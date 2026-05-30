from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    import kagglesdk.kaggle_env as kaggle_env

    if not hasattr(kaggle_env, "get_web_endpoint"):
        kaggle_env.get_web_endpoint = kaggle_env.get_endpoint
except ImportError:
    pass

import kagglehub
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, PolynomialFeatures, StandardScaler


@dataclass
class RecommendationRunResult:
    summary: pd.DataFrame
    model_results: dict[str, dict[str, Any]]
    final_result_key: str
    final_predictions: pd.DataFrame
    baseline_product: str
    baseline_match_rate: float
    product_prior: pd.Series


def load_dataset_from_kaggle(
    dataset: str = "imakash3011/customer-personality-analysis",
    file_name: str = "marketing_campaign.csv",
) -> pd.DataFrame:
    """
    Load the Customer Personality Analysis dataset from Kaggle.
    """
    path = Path(kagglehub.dataset_download(dataset))
    return pd.read_csv(path / file_name, sep="\t").copy()


def load_dataset_from_path(path: str | Path, sep: str = "\t") -> pd.DataFrame:
    """
    Load the marketing campaign dataset from a local CSV-like text file.
    """
    return pd.read_csv(path, sep=sep).copy()


def normalize_predicted_ratios(raw_pred: Any) -> np.ndarray:
    """
    Clip negative multi-output regression predictions and normalize each row.
    """
    pred = np.clip(np.asarray(raw_pred, dtype=float), 0, None)
    if pred.ndim == 1:
        pred = pred.reshape(-1, 1)

    row_sums = pred.sum(axis=1, keepdims=True)
    return np.divide(pred, row_sums, out=np.zeros_like(pred), where=row_sums != 0)


def relative_preference_score(
    ratio_df: pd.DataFrame,
    product_prior: pd.Series,
) -> pd.DataFrame:
    """
    Score product preference relative to the train-set average product share.
    """
    score_df = ratio_df.div(product_prior, axis=1)
    return score_df.replace([np.inf, -np.inf], np.nan).fillna(0)


def banner_ad_order(score_df: pd.DataFrame) -> pd.Series:
    """
    Create a display order string from highest to lowest recommendation score.
    """
    return score_df.apply(
        lambda row: " > ".join(row.sort_values(ascending=False).index),
        axis=1,
    )


def banner_ad_scores(score_df: pd.DataFrame) -> pd.Series:
    """
    Create a compact score string from highest to lowest recommendation score.
    """
    return score_df.apply(
        lambda row: " | ".join(
            f"{product}:{score:.3f}"
            for product, score in row.sort_values(ascending=False).items()
        ),
        axis=1,
    )


def best_by_business_model(summary_df: pd.DataFrame) -> pd.DataFrame:
    """
    Return the best row per business model from a model summary table.
    """
    return (
        summary_df.sort_values(
            by=["Business Model", "Recommendation Match Rate", "Test R2"],
            ascending=[True, False, False],
        )
        .groupby("Business Model", as_index=False)
        .head(1)
    )


class MarketingItemRecommendationRegression:
    """
    Reusable pipeline for product-category recommendation regression.

    The model predicts each customer's spending ratio across six product groups.
    The final recommendation is based on predicted ratio divided by the train-set
    product prior, so generally popular categories do not always dominate.
    """

    def __init__(
        self,
        figure_dir: str | Path = "figures",
        result_dir: str | Path = "results",
        random_state: int = 42,
    ):
        self.figure_dir = Path(figure_dir)
        self.result_dir = Path(result_dir)
        self.random_state = random_state

        self.raw_df: pd.DataFrame | None = None
        self.df: pd.DataFrame | None = None
        self.X: pd.DataFrame | None = None
        self.y: pd.DataFrame | None = None
        self.last_result: RecommendationRunResult | None = None

    def load_dataset_from_kaggle(self) -> pd.DataFrame:
        self.raw_df = load_dataset_from_kaggle()
        return self.raw_df.copy()

    def load_dataset_from_path(self, path: str | Path, sep: str = "\t") -> pd.DataFrame:
        self.raw_df = load_dataset_from_path(path, sep=sep)
        return self.raw_df.copy()

    def prepare_dataframe(
        self,
        df: pd.DataFrame,
        min_age: int = 18,
        max_age: int = 100,
        max_income: int = 300000,
    ) -> pd.DataFrame:
        """
        Create reusable customer features and remove clear outliers.
        """
        self._validate_columns(
            df,
            [
                "Year_Birth",
                "Income",
                "Education",
                "Marital_Status",
                "Kidhome",
                "Teenhome",
            ],
        )

        prepared = df.copy()
        prepared["Age"] = 2014 - prepared["Year_Birth"]
        prepared = prepared[(prepared["Age"] >= min_age) & (prepared["Age"] <= max_age)]
        prepared = prepared[prepared["Income"].isna() | (prepared["Income"] <= max_income)]

        prepared["Education"] = prepared["Education"].astype("category")

        rare_status = ["Alone", "Absurd", "YOLO"]
        prepared["Marital_Status"] = prepared["Marital_Status"].replace(rare_status, "Other")
        marital_mode = prepared["Marital_Status"].mode(dropna=True)
        marital_fill = marital_mode.iloc[0] if not marital_mode.empty else "Other"
        prepared["Marital_Status"] = prepared["Marital_Status"].fillna(marital_fill)
        prepared["Marital_Status"] = prepared["Marital_Status"].astype("category")

        prepared["HasPartner"] = prepared["Marital_Status"].isin(["Married", "Together"]).astype(int)
        prepared["TotalChildren"] = prepared["Kidhome"] + prepared["Teenhome"]

        return prepared

    def build_feature_matrix(
        self,
        df: pd.DataFrame,
        feature_sets: Mapping[str, Sequence[str]] | None = None,
    ) -> pd.DataFrame:
        """
        Build the full feature matrix containing every feature used by any stage.
        """
        feature_sets = self._copy_feature_sets(feature_sets)
        all_feature_cols = sorted({col for cols in feature_sets.values() for col in cols})
        self._validate_columns(df, all_feature_cols)
        return df[all_feature_cols].copy()

    def build_targets(
        self,
        df: pd.DataFrame,
        product_cols: Sequence[str] | None = None,
    ) -> tuple[pd.DataFrame, pd.Index, pd.Series]:
        """
        Convert product spending amounts into per-customer product share targets.
        """
        if product_cols is None:
            product_cols = [
                "MntWines",
                "MntFruits",
                "MntMeatProducts",
                "MntFishProducts",
                "MntSweetProducts",
                "MntGoldProds",
            ]
        product_cols = list(product_cols)
        self._validate_columns(df, product_cols)

        total_spending = df[product_cols].sum(axis=1)
        valid_target_index = total_spending[total_spending > 0].index
        y = df.loc[valid_target_index, product_cols].div(
            total_spending.loc[valid_target_index],
            axis=0,
        )
        return y, valid_target_index, total_spending

    def build_stage_preprocessor(
        self,
        stage_features: Sequence[str],
        log_numeric_candidates: Sequence[str] | None = None,
        standard_numeric_candidates: Sequence[str] | None = None,
        categorical_candidates: Sequence[str] | None = None,
    ) -> ColumnTransformer:
        """
        Build a leakage-safe preprocessor for one business-stage feature set.
        """
        if log_numeric_candidates is None:
            log_numeric_candidates = [
                "Income",
                "NumWebVisitsMonth",
                "NumWebPurchases",
                "NumCatalogPurchases",
                "NumStorePurchases",
            ]
        if standard_numeric_candidates is None:
            standard_numeric_candidates = [
                "Age",
                "Kidhome",
                "Teenhome",
                "Recency",
            ]
        if categorical_candidates is None:
            categorical_candidates = [
                "Education",
                "Marital_Status",
            ]

        stage_features = list(stage_features)
        log_numeric_features = [
            col for col in log_numeric_candidates
            if col in stage_features
        ]
        standard_numeric_features = [
            col for col in standard_numeric_candidates
            if col in stage_features
        ]
        categorical_features = [
            col for col in categorical_candidates
            if col in stage_features
        ]

        transformers = []

        if log_numeric_features:
            transformers.append(
                (
                    "log_num",
                    Pipeline(
                        steps=[
                            ("imputer", SimpleImputer(strategy="median")),
                            ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
                            ("scaler", StandardScaler()),
                        ]
                    ),
                    log_numeric_features,
                )
            )

        if standard_numeric_features:
            transformers.append(
                (
                    "num",
                    Pipeline(
                        steps=[
                            ("imputer", SimpleImputer(strategy="median")),
                            ("scaler", StandardScaler()),
                        ]
                    ),
                    standard_numeric_features,
                )
            )

        if categorical_features:
            transformers.append(
                (
                    "cat",
                    Pipeline(
                        steps=[
                            ("imputer", SimpleImputer(strategy="most_frequent")),
                            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                        ]
                    ),
                    categorical_features,
                )
            )

        if not transformers:
            raise ValueError("No usable stage features were found for preprocessing.")

        return ColumnTransformer(transformers=transformers)

    def default_models(self) -> dict[str, Any]:
        """
        Return the regression models used in the notebook experiment.
        """
        return {
            "Multiple Linear Regression": LinearRegression(),
            "Polynomial Regression degree 2": Pipeline(
                steps=[
                    ("poly", PolynomialFeatures(degree=2, include_bias=False)),
                    ("linear", LinearRegression()),
                ]
            ),
        }

    def run_pipeline(
        self,
        df: pd.DataFrame | None = None,
        feature_sets: Mapping[str, Sequence[str]] | None = None,
        models: Mapping[str, Any] | None = None,
        product_cols: Sequence[str] | None = None,
        test_size: float = 0.2,
        cv_splits: int = 5,
        verbose: bool = True,
        save_results: bool = False,
        plot: bool = False,
    ) -> RecommendationRunResult:
        """
        Run preprocessing, model comparison, and final recommendation selection.
        """
        if df is None:
            df = self.raw_df if self.raw_df is not None else self.load_dataset_from_kaggle()

        feature_sets = self._copy_feature_sets(feature_sets)
        prepared_df = self.prepare_dataframe(df)
        X = self.build_feature_matrix(prepared_df, feature_sets)
        y, valid_target_index, _ = self.build_targets(prepared_df, product_cols)
        X_model = X.loc[valid_target_index].copy()

        result = self.evaluate_models(
            X_model=X_model,
            y=y,
            feature_sets=feature_sets,
            models=models,
            product_cols=product_cols,
            test_size=test_size,
            cv_splits=cv_splits,
            verbose=verbose,
        )

        self.df = prepared_df
        self.X = X_model
        self.y = y
        self.last_result = result

        if save_results:
            self.save_result_tables(result)

        if plot:
            self.plot_model_summary(result)
            self.plot_final_diagnostics(result)

        return result

    def evaluate_models(
        self,
        X_model: pd.DataFrame,
        y: pd.DataFrame,
        feature_sets: Mapping[str, Sequence[str]] | None = None,
        models: Mapping[str, Any] | None = None,
        product_cols: Sequence[str] | None = None,
        test_size: float = 0.2,
        cv_splits: int = 5,
        verbose: bool = True,
    ) -> RecommendationRunResult:
        """
        Compare all stage/model combinations and keep fitted pipelines.
        """
        feature_sets = self._copy_feature_sets(feature_sets)
        models = dict(models) if models is not None else self.default_models()
        if product_cols is None:
            product_cols = [
                "MntWines",
                "MntFruits",
                "MntMeatProducts",
                "MntFishProducts",
                "MntSweetProducts",
                "MntGoldProds",
            ]
        product_cols = list(product_cols)

        train_index, test_index = train_test_split(
            X_model.index,
            test_size=test_size,
            random_state=self.random_state,
        )

        X_train_all = X_model.loc[train_index]
        X_test_all = X_model.loc[test_index]
        y_train = y.loc[train_index]
        y_test = y.loc[test_index]

        product_prior = y_train.mean().replace(0, np.nan)
        y_train_relative_score = relative_preference_score(y_train, product_prior)
        y_test_relative_score = relative_preference_score(y_test, product_prior)

        baseline_product = y_train_relative_score.idxmax(axis=1).mode()[0]
        actual_relative_product = y_test_relative_score.idxmax(axis=1)
        baseline_match_rate = float((actual_relative_product == baseline_product).mean())
        cv = KFold(n_splits=cv_splits, shuffle=True, random_state=self.random_state)

        model_results: dict[str, dict[str, Any]] = {}
        model_summary = []

        if verbose:
            print("Baseline product:", baseline_product)
            print("Baseline match rate:", baseline_match_rate)
            print("Product prior:")
            print(product_prior)

        for stage_name, stage_features in feature_sets.items():
            stage_features = list(stage_features)
            X_stage = X_model[stage_features]
            X_train = X_train_all[stage_features]
            X_test = X_test_all[stage_features]

            for model_name, model in models.items():
                reg = Pipeline(
                    steps=[
                        ("preprocessor", self.build_stage_preprocessor(stage_features)),
                        ("model", clone(model)),
                    ]
                )

                cv_scores = cross_val_score(reg, X_stage, y, cv=cv, scoring="r2")

                reg.fit(X_train, y_train)
                y_pred = normalize_predicted_ratios(reg.predict(X_test))
                y_pred_df = pd.DataFrame(y_pred, columns=product_cols, index=y_test.index)
                recommendation_score_df = relative_preference_score(y_pred_df, product_prior)

                recommended_product = recommendation_score_df.idxmax(axis=1)
                recommendation_match_rate = float(
                    (actual_relative_product == recommended_product).mean()
                )

                test_r2 = float(r2_score(y_test, y_pred))
                test_mae = float(mean_absolute_error(y_test, y_pred))
                test_rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))

                result_key = f"{stage_name} | {model_name}"

                if verbose:
                    print(f"\n--- {result_key} ---")
                    print("Features:", stage_features)
                    print("5-fold CV R2:", cv_scores)
                    print("Mean CV R2:", cv_scores.mean())
                    print("Test R2:", test_r2)
                    print("Test MAE:", test_mae)
                    print("Test RMSE:", test_rmse)
                    print("Relative recommendation match rate:", recommendation_match_rate)

                model_summary.append(
                    {
                        "Business Model": stage_name,
                        "Regression Model": model_name,
                        "Feature Count": len(stage_features),
                        "Features": ", ".join(stage_features),
                        "CV R2 Mean": float(cv_scores.mean()),
                        "CV R2 Std": float(cv_scores.std()),
                        "Test R2": test_r2,
                        "Test MAE": test_mae,
                        "Test RMSE": test_rmse,
                        "Recommendation Match Rate": recommendation_match_rate,
                        "Baseline Match Rate": baseline_match_rate,
                        "Lift vs Baseline": recommendation_match_rate - baseline_match_rate,
                    }
                )

                result_df = self.build_prediction_frame(
                    y_test=y_test,
                    y_pred_df=y_pred_df,
                    y_test_relative_score=y_test_relative_score,
                    recommendation_score_df=recommendation_score_df,
                    actual_product=actual_relative_product,
                    recommended_product=recommended_product,
                )

                model_results[result_key] = {
                    "business_model": stage_name,
                    "regression_model": model_name,
                    "features": stage_features,
                    "model": reg,
                    "predictions": result_df,
                }

        summary_df = pd.DataFrame(model_summary).sort_values(
            by=["Recommendation Match Rate", "Test R2"],
            ascending=False,
        )
        final_result_key = (
            summary_df.iloc[0]["Business Model"]
            + " | "
            + summary_df.iloc[0]["Regression Model"]
        )
        final_predictions = model_results[final_result_key]["predictions"]

        return RecommendationRunResult(
            summary=summary_df,
            model_results=model_results,
            final_result_key=final_result_key,
            final_predictions=final_predictions,
            baseline_product=baseline_product,
            baseline_match_rate=baseline_match_rate,
            product_prior=product_prior,
        )

    def build_prediction_frame(
        self,
        y_test: pd.DataFrame,
        y_pred_df: pd.DataFrame,
        y_test_relative_score: pd.DataFrame,
        recommendation_score_df: pd.DataFrame,
        actual_product: pd.Series,
        recommended_product: pd.Series,
    ) -> pd.DataFrame:
        """
        Combine actual ratios, predicted ratios, and final recommendation fields.
        """
        result_df = pd.concat(
            [
                y_test.add_prefix("ActualRatio_"),
                y_pred_df.add_prefix("PredictedRatio_"),
                y_test_relative_score.add_prefix("ActualRelativeScore_"),
                recommendation_score_df.add_prefix("RecommendationScore_"),
            ],
            axis=1,
        )
        result_df["Actual_Product"] = actual_product
        result_df["Recommended_Product"] = recommended_product
        result_df["Recommendation_Match"] = actual_product == recommended_product
        result_df["Banner_Ad_Order"] = banner_ad_order(recommendation_score_df)
        result_df["Banner_Ad_Scores"] = banner_ad_scores(recommendation_score_df)
        return result_df

    def recommend_customers(
        self,
        df: pd.DataFrame,
        result: RecommendationRunResult | None = None,
        result_key: str | None = None,
        prepare: bool = True,
        product_cols: Sequence[str] | None = None,
    ) -> pd.DataFrame:
        """
        Recommend product-category ad order for new or existing customer rows.
        """
        if product_cols is None:
            product_cols = [
                "MntWines",
                "MntFruits",
                "MntMeatProducts",
                "MntFishProducts",
                "MntSweetProducts",
                "MntGoldProds",
            ]
        result = self._require_result(result)
        result_key = result_key or result.final_result_key
        model_info = result.model_results[result_key]
        stage_features = list(model_info["features"])

        scoring_df = self.prepare_dataframe(df) if prepare else df.copy()
        self._validate_columns(scoring_df, stage_features)

        raw_pred = model_info["model"].predict(scoring_df[stage_features])
        y_pred = normalize_predicted_ratios(raw_pred)
        ratio_df = pd.DataFrame(y_pred, columns=list(product_cols), index=scoring_df.index)
        score_df = relative_preference_score(ratio_df, result.product_prior)

        output_df = pd.concat(
            [
                ratio_df.add_prefix("PredictedRatio_"),
                score_df.add_prefix("RecommendationScore_"),
            ],
            axis=1,
        )
        output_df["Recommended_Product"] = score_df.idxmax(axis=1)
        output_df["Banner_Ad_Order"] = banner_ad_order(score_df)
        output_df["Banner_Ad_Scores"] = banner_ad_scores(score_df)
        return output_df

    def get_banner_recommendation_output(
        self,
        final_predictions: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """
        Return the final banner-ad fields used in the notebook.
        """
        if final_predictions is None:
            final_predictions = self._require_result().final_predictions
        return final_predictions[
            [
                "Actual_Product",
                "Recommended_Product",
                "Recommendation_Match",
                "Banner_Ad_Order",
                "Banner_Ad_Scores",
            ]
        ].copy()

    def get_customer_score_table(
        self,
        customer_id: Any | None = None,
        final_predictions: pd.DataFrame | None = None,
        product_cols: Sequence[str] | None = None,
    ) -> pd.DataFrame:
        """
        Return one customer's product recommendation scores in rank order.
        """
        if product_cols is None:
            product_cols = [
                "MntWines",
                "MntFruits",
                "MntMeatProducts",
                "MntFishProducts",
                "MntSweetProducts",
                "MntGoldProds",
            ]
        if final_predictions is None:
            final_predictions = self._require_result().final_predictions
        if customer_id is None:
            customer_id = final_predictions.index[0]

        score_cols = [f"RecommendationScore_{col}" for col in product_cols]
        score_table = final_predictions.loc[[customer_id], score_cols].T.reset_index()
        score_table.columns = ["Product", "RecommendationScore"]
        score_table["Product"] = score_table["Product"].str.replace(
            "RecommendationScore_",
            "",
            regex=False,
        )
        return score_table.sort_values("RecommendationScore", ascending=False)

    def save_result_tables(
        self,
        result: RecommendationRunResult | None = None,
    ) -> dict[str, Path]:
        """
        Save reusable CSV outputs from the latest run.
        """
        result = self._require_result(result)
        self.result_dir.mkdir(exist_ok=True)

        paths = {
            "summary": self.result_dir / "item_recommendation_model_summary.csv",
            "best_by_business_model": self.result_dir / "item_recommendation_best_by_business_model.csv",
            "final_predictions": self.result_dir / "item_recommendation_final_predictions.csv",
            "banner_output": self.result_dir / "item_recommendation_banner_output.csv",
        }
        result.summary.to_csv(paths["summary"], index=False)
        best_by_business_model(result.summary).to_csv(paths["best_by_business_model"], index=False)
        result.final_predictions.to_csv(paths["final_predictions"], index=True)
        self.get_banner_recommendation_output(result.final_predictions).to_csv(
            paths["banner_output"],
            index=True,
        )
        return paths

    def plot_model_summary(
        self,
        result: RecommendationRunResult | None = None,
        save: bool = True,
        show: bool = True,
    ) -> None:
        """
        Plot match-rate, R2, and baseline-lift summaries.
        """
        import seaborn as sns

        result = self._require_result(result)
        summary_df = result.summary

        plt.figure(figsize=(10, 5))
        sns.barplot(
            data=summary_df,
            x="Business Model",
            y="Recommendation Match Rate",
            hue="Regression Model",
        )
        plt.axhline(
            result.baseline_match_rate,
            color="red",
            linestyle="--",
            label=f"Baseline ({result.baseline_match_rate:.3f})",
        )
        plt.title("Relative Preference Match Rate by Business Model")
        plt.xlabel("Business Model")
        plt.ylabel("Match Rate")
        plt.ylim(0, 1)
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()
        self._finish_plot("relative_preference_match_rate_by_business_model", save, show)

        r2_plot_df = summary_df.melt(
            id_vars=["Business Model", "Regression Model"],
            value_vars=["CV R2 Mean", "Test R2"],
            var_name="Metric",
            value_name="R2",
        )

        plt.figure(figsize=(10, 5))
        sns.barplot(
            data=r2_plot_df,
            x="Business Model",
            y="R2",
            hue="Metric",
        )
        plt.title("Regression R2 by Business Model")
        plt.xlabel("Business Model")
        plt.ylabel("R2")
        plt.tight_layout()
        self._finish_plot("regression_r2_by_business_model", save, show)

        plt.figure(figsize=(10, 5))
        sns.barplot(
            data=summary_df,
            x="Business Model",
            y="Lift vs Baseline",
            hue="Regression Model",
        )
        plt.axhline(0, color="black", linewidth=1)
        plt.title("Relative Recommendation Lift vs Baseline")
        plt.xlabel("Business Model")
        plt.ylabel("Match Rate Lift")
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()
        self._finish_plot("relative_recommendation_lift_vs_baseline", save, show)

    def plot_final_diagnostics(
        self,
        result: RecommendationRunResult | None = None,
        product_cols: Sequence[str] | None = None,
        save: bool = True,
        show: bool = True,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Plot confusion-rate heatmap and actual-vs-recommended distribution.
        """
        import seaborn as sns

        result = self._require_result(result)
        if product_cols is None:
            product_cols = [
                "MntWines",
                "MntFruits",
                "MntMeatProducts",
                "MntFishProducts",
                "MntSweetProducts",
                "MntGoldProds",
            ]
        product_cols = list(product_cols)
        final_predictions = result.final_predictions

        confusion_counts = pd.crosstab(
            final_predictions["Actual_Product"],
            final_predictions["Recommended_Product"],
        ).reindex(index=product_cols, columns=product_cols, fill_value=0)

        confusion_rate = pd.crosstab(
            final_predictions["Actual_Product"],
            final_predictions["Recommended_Product"],
            normalize="index",
        ).reindex(index=product_cols, columns=product_cols, fill_value=0)

        plt.figure(figsize=(8, 6))
        sns.heatmap(confusion_rate, annot=True, fmt=".2f", cmap="Blues")
        plt.title(f"Actual vs Recommended Relative Product Ratio ({result.final_result_key})")
        plt.xlabel("Recommended Product")
        plt.ylabel("Actual Product")
        plt.tight_layout()
        self._finish_plot("actual_vs_recommended_relative_product_ratio", save, show)

        product_distribution_df = pd.DataFrame(
            {
                "Actual": final_predictions["Actual_Product"].value_counts(normalize=True),
                "Recommended": final_predictions["Recommended_Product"].value_counts(normalize=True),
            }
        ).reindex(product_cols).fillna(0)

        product_distribution_df.plot(kind="bar", figsize=(9, 5))
        plt.title("Actual vs Recommended Relative Product Distribution")
        plt.xlabel("Product")
        plt.ylabel("Customer Share")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        self._finish_plot("actual_vs_recommended_relative_product_distribution", save, show)

        return confusion_counts, product_distribution_df

    def _finish_plot(self, filename: str, save: bool, show: bool) -> None:
        if save:
            self.figure_dir.mkdir(exist_ok=True)
            plt.savefig(self.figure_dir / f"{filename}.png", dpi=300, bbox_inches="tight")
        if show:
            plt.show()
        plt.close()

    def _require_result(
        self,
        result: RecommendationRunResult | None = None,
    ) -> RecommendationRunResult:
        if result is not None:
            return result
        if self.last_result is None:
            raise ValueError("No recommendation result is available. Run run_pipeline() first.")
        return self.last_result

    @staticmethod
    def _copy_feature_sets(
        feature_sets: Mapping[str, Sequence[str]] | None = None,
    ) -> dict[str, list[str]]:
        if feature_sets is None:
            feature_sets = {
                "Cold-start": [
                    "Age",
                    "Income",
                    "Education",
                    "Marital_Status",
                    "Kidhome",
                    "Teenhome",
                ],
                "Early-behavior": [
                    "Age",
                    "Income",
                    "Education",
                    "Marital_Status",
                    "Kidhome",
                    "Teenhome",
                    "NumWebVisitsMonth",
                ],
                "Existing-customer": [
                    "Age",
                    "Income",
                    "Education",
                    "Marital_Status",
                    "Kidhome",
                    "Teenhome",
                    "Recency",
                    "NumWebPurchases",
                    "NumCatalogPurchases",
                    "NumStorePurchases",
                ],
            }
        source = feature_sets
        return {name: list(features) for name, features in source.items()}

    @staticmethod
    def _validate_columns(df: pd.DataFrame, columns: Sequence[str]) -> None:
        missing_cols = [col for col in columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")


# Backward-compatible name used by the original file.
ItemRecommendation = MarketingItemRecommendationRegression


def main() -> None:
    recommender = MarketingItemRecommendationRegression()
    result = recommender.run_pipeline(verbose=True, save_results=True)

    print("\nFinal model selected:", result.final_result_key)
    print(recommender.get_banner_recommendation_output(result.final_predictions).head(10))

    sample_customer_id = result.final_predictions.index[0]
    print(f"\nSample customer index: {sample_customer_id}")
    print("Banner ad display order:")
    for rank, product in enumerate(
        result.final_predictions.loc[sample_customer_id, "Banner_Ad_Order"].split(" > "),
        start=1,
    ):
        print(f"{rank}. {product}")

    print("\nRecommendation score table:")
    print(recommender.get_customer_score_table(sample_customer_id, result.final_predictions))


if __name__ == "__main__":
    main()
