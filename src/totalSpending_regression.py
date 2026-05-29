from pathlib import Path

import kagglehub
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
    PolynomialFeatures,
    RobustScaler,
    StandardScaler,
)
from sklearn.impute import SimpleImputer

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

class TotalSpendingRegressor:
    def __init__(self):
        self.figure_dir = Path("figures")
        self.result_dir = Path("results")
        self.raw_df = self.__load_dataset_from_kaggle()


    def __load_dataset_from_kaggle(self):
        """
        Load Customer Personality Analysis dataset.
        Kaggle original file is a CSV-like text file.
        """
        path = kagglehub.dataset_download("imakash3011/customer-personality-analysis")
        df = pd.read_csv(path + '/marketing_campaign.csv', sep='\t')
        return df.copy()
    def preprocessing(self):
        temp_df = self.create_features(self.raw_df)
        self.df = self.clean_data(temp_df)

    def save_and_show_plot(self, filename):
        """
        Save the current matplotlib figure to the figures directory and show it.
        """
        self.figure_dir.mkdir(exist_ok=True)
        safe_filename = "".join(
            char.lower() if char.isalnum() else "_"
            for char in filename
        ).strip("_")
        figure_path = self.figure_dir / f"{safe_filename}.png"
        plt.savefig(figure_path, dpi=300, bbox_inches="tight")
        print(f"Saved figure: {figure_path}")
        plt.show()
        plt.close()

    def create_features(self, df):
        """
        Create derived variables according to the team proposal.
        """
        df = df.copy()

        df["Dt_Customer"] = pd.to_datetime(
            df["Dt_Customer"],
            format="%d-%m-%Y",
            errors="coerce"
        )

        # Create Age from Year_Birth.
        df["Age"] = 2014 - df["Year_Birth"]

        # Customer tenure in days.
        reference_date = df["Dt_Customer"].max()
        df["Customer_Days"] = (reference_date - df["Dt_Customer"]).dt.days

        # Family-related feature.
        df["TotalChildren"] = df["Kidhome"] + df["Teenhome"]

        # Total spending from six product spending columns.
        spending_cols = [
            "MntWines",
            "MntFruits",
            "MntMeatProducts",
            "MntFishProducts",
            "MntSweetProducts",
            "MntGoldProds",
        ]
        df["TotalSpending"] = df[spending_cols].sum(axis=1)

        # Total purchases from actual purchase count variables.
        purchase_cols = [
            "NumDealsPurchases",
            "NumWebPurchases",
            "NumCatalogPurchases",
            "NumStorePurchases",
        ]
        df["TotalPurchases"] = df[purchase_cols].sum(axis=1)

        # Total number of accepted previous campaigns.
        campaign_cols = [
            "AcceptedCmp1",
            "AcceptedCmp2",
            "AcceptedCmp3",
            "AcceptedCmp4",
            "AcceptedCmp5",
        ]
        df["CampaignAcceptedTotal"] = df[campaign_cols].sum(axis=1)

        return df

    def clean_data(self, df):
        """
        Clean dirty data and remove unnecessary columns.
        """
        df = df.copy()

        # Group rare marital status categories into Other.
        rare_status = ["Alone", "Absurd", "YOLO"]
        df["Marital_Status"] = df["Marital_Status"].replace(rare_status, "Other")

        # Remove unrealistic ages.
        df = df[(df["Age"] >= 18) & (df["Age"] <= 100)]

        # Remove extreme income outlier.
        # The proposal notes Income has a maximum value of 666,666.
        df = df[df["Income"].isna() | (df["Income"] <= 200000)]


        # Remove identifier and constant/date columns.
        drop_cols = [
            "ID",
            "Year_Birth",
            "Dt_Customer",
            "Z_CostContact",
            "Z_Revenue",
        ]
        df = df.drop(columns=[col for col in drop_cols if col in df.columns])

        return df

    def data_inspection(self, df=None):
        if df is None:
            df = self.raw_df
        corr_matrix = df.corr(method='pearson', numeric_only=True)
        print("--- correlation matrix ---")
        print(corr_matrix)

        plt.figure(figsize=(20, 20))

        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', vmin=-1, vmax=1, linewidths=0.5)
        plt.title('Correlation Matrix Heatmap', fontsize=15)
        self.save_and_show_plot("correlation_matrix_heatmap")

    def build_preprocessor(self, X, scaler_name="standard", encoder_name="onehot"):
        """
        Build preprocessing pipeline:
        - Numeric columns: median imputation + selected scaling method
        - Categorical columns: most frequent imputation + selected encoding method
        """
        categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
        numeric_cols = X.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns.tolist()

        scalers = {
            "standard": StandardScaler(),
            "minmax": MinMaxScaler(),
            "robust": RobustScaler(),
        }
        encoders = {
            "onehot": OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            "ordinal": OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=-1,
            ),
        }

        numeric_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", scalers[scaler_name]),
            ]
        )

        categorical_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", encoders[encoder_name]),
            ]
        )

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, numeric_cols),
                ("cat", categorical_transformer, categorical_cols),
            ]
        )

        return preprocessor

    def find_best_combinations(self, df, top_n=5):
        """
        Run model selection experiments under one top-level function.

        It evaluates combinations of:
        - numeric scaling methods
        - categorical encoding methods
        - learning models and model parameters
        - cross-validation and hold-out test metrics
        """
        print("\n==============================")
        print("Model Selection: Top Combinations")
        print("==============================")

        selected_features = [
            "Income",
            "Age",
            "Kidhome",
            "Teenhome",
            "Customer_Days",
            "TotalChildren",
            "Education",
            "Marital_Status",
        ]
        X = df[selected_features]
        y = df["TotalSpending"]

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
        )
        cv = KFold(n_splits=5, shuffle=True, random_state=42)

        model_configs = [
            ("LinearRegression", LinearRegression()),
            ("Ridge_alpha_0.1", Ridge(alpha=0.1)),
            ("Ridge_alpha_1.0", Ridge(alpha=1.0)),
            ("Ridge_alpha_10.0", Ridge(alpha=10.0)),
            ("Lasso_alpha_0.1", Lasso(alpha=0.1, max_iter=10000)),
            ("Lasso_alpha_1.0", Lasso(alpha=1.0, max_iter=10000)),
            (
                "RandomForest_depth_5",
                RandomForestRegressor(
                    n_estimators=200,
                    max_depth=5,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
            (
                "RandomForest_depth_10",
                RandomForestRegressor(
                    n_estimators=200,
                    max_depth=10,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
            (
                "GradientBoosting_lr_0.05",
                GradientBoostingRegressor(
                    learning_rate=0.05,
                    n_estimators=200,
                    max_depth=3,
                    random_state=42,
                ),
            ),
            (
                "GradientBoosting_lr_0.1",
                GradientBoostingRegressor(
                    learning_rate=0.1,
                    n_estimators=200,
                    max_depth=3,
                    random_state=42,
                ),
            ),
        ]

        results = []
        for scaler_name in ["standard", "minmax", "robust"]:
            for encoder_name in ["onehot", "ordinal"]:
                preprocessor = self.build_preprocessor(
                    X,
                    scaler_name=scaler_name,
                    encoder_name=encoder_name,
                )

                for model_name, model in model_configs:
                    pipeline = Pipeline(
                        steps=[
                            ("preprocessor", preprocessor),
                            ("model", model),
                        ]
                    )

                    cv_r2_scores = cross_val_score(
                        pipeline,
                        X,
                        y,
                        cv=cv,
                        scoring="r2",
                    )
                    cv_mae_scores = -cross_val_score(
                        pipeline,
                        X,
                        y,
                        cv=cv,
                        scoring="neg_mean_absolute_error",
                    )

                    pipeline.fit(X_train, y_train)
                    y_pred = pipeline.predict(X_test)

                    test_mse = mean_squared_error(y_test, y_pred)
                    results.append(
                        {
                            "scaler": scaler_name,
                            "encoder": encoder_name,
                            "model": model_name,
                            "cv_r2_mean": cv_r2_scores.mean(),
                            "cv_r2_std": cv_r2_scores.std(),
                            "cv_mae_mean": cv_mae_scores.mean(),
                            "test_r2": r2_score(y_test, y_pred),
                            "test_mae": mean_absolute_error(y_test, y_pred),
                            "test_rmse": np.sqrt(test_mse),
                        }
                    )

        results_df = pd.DataFrame(results).sort_values(
            by=["cv_r2_mean", "test_r2"],
            ascending=False,
        )

        print(f"\nTop {top_n} combinations by mean CV R2:")
        top_results = results_df.head(top_n)
        print(top_results.to_string(index=False))

        self.result_dir.mkdir(exist_ok=True)
        results_df.to_csv(self.result_dir / "model_selection_results.csv", index=False)
        top_results.to_csv(self.result_dir / "top_5_model_combinations.csv", index=False)
        print(f"\nSaved results: {self.result_dir / 'model_selection_results.csv'}")
        print(f"Saved top {top_n}: {self.result_dir / 'top_5_model_combinations.csv'}")

        self.model_selection_results = results_df
        return results_df

    def run_regression(self, df):
        """
        Regression task:
        Predict TotalSpending.
        """
        print("\n==============================")
        print("Regression: Predict TotalSpending")
        print("==============================")

        y = df["TotalSpending"]

        # Proposal-based simple regression input features.
        selected_features = ["Income", "Age", "Kidhome", "Teenhome"]
        X = df[selected_features]

        preprocessor = self.build_preprocessor(X)

        models = {
            "Multiple Linear Regression": LinearRegression(),
            "Polynomial Regression degree 2": Pipeline(
                steps=[
                    ("poly", PolynomialFeatures(degree=2, include_bias=False)),
                    ("linear", LinearRegression()),
                ]
            ),
        }

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
        )

        cv = KFold(n_splits=5, shuffle=True, random_state=42)

        for model_name, model in models.items():
            print(f"\n--- {model_name} ---")

            reg = Pipeline(
                steps=[
                    ("preprocessor", preprocessor),
                    ("model", model),
                ]
            )

            cv_scores = cross_val_score(reg, X, y, cv=cv, scoring="r2")
            print("5-fold CV R2:", cv_scores)
            print("Mean CV R2:", cv_scores.mean())

            reg.fit(X_train, y_train)
            y_pred = reg.predict(X_test)

            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            print("Test R2:", r2)
            print("MAE:", mae)
            print("RMSE:", rmse)

            if model_name == "Multiple Linear Regression":
                coef_df = pd.DataFrame(
                    {
                        "Feature": selected_features,
                        "Coefficient": reg.named_steps["model"].coef_,
                    }
                ).sort_values("Coefficient", ascending=False)

                print("\nCoefficient:")
                print(coef_df)

            result_df = pd.DataFrame(
                {
                    "Actual_TotalSpending": y_test,
                    "Predicted_TotalSpending": y_pred,
                    "Error": y_test - y_pred,
                }
            )
            print("\nPrediction result sample:")
            print(result_df.head(10))

            self.plot_actual_vs_predicted(y_test, y_pred, model_name)
            self.plot_regression_curve(reg, X, y, selected_features, model_name)

    def plot_actual_vs_predicted(self, y_test, y_pred, model_name):
        plt.figure(figsize=(6, 6))
        plt.scatter(y_test, y_pred, alpha=0.6)
        plt.plot(
            [y_test.min(), y_test.max()],
            [y_test.min(), y_test.max()],
            color="red",
        )
        plt.xlabel("Actual TotalSpending")
        plt.ylabel("Predicted TotalSpending")
        plt.title(f"Actual vs Predicted TotalSpending ({model_name})")
        plt.tight_layout()
        self.save_and_show_plot(f"actual_vs_predicted_{model_name}")

    def plot_regression_curve(self, fitted_model, X, y, selected_features, model_name, feature="Income"):
        """
        Plot the fitted regression curve for one feature while holding the other
        model features at their median values.
        """
        X_curve = pd.DataFrame(
            np.tile(X[selected_features].median().values, (200, 1)),
            columns=selected_features,
        )
        X_curve[feature] = np.linspace(X[feature].min(), X[feature].max(), 200)
        y_curve = fitted_model.predict(X_curve)

        plt.figure(figsize=(8, 5))
        plt.scatter(X[feature], y, alpha=0.35, label="Actual")
        plt.plot(X_curve[feature], y_curve, color="red", linewidth=2, label="Regression curve")
        plt.xlabel(feature)
        plt.ylabel("TotalSpending")
        plt.title(f"{model_name}: {feature} vs TotalSpending")
        plt.legend()
        plt.tight_layout()
        self.save_and_show_plot(f"{model_name}_{feature}_regression_curve")


