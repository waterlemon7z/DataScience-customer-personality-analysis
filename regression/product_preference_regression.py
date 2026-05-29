"""
Product preference regression and banner recommendation.

Business goal:
    Predict each customer's expected purchase share for six product categories,
    then rank product banner ads by relative preference.

Leakage rule:
    Product spending columns (MntWines ... MntGoldProds), TotalSpending, and
    product share targets are never used as input features.

Business models:
    1. New-user: demographic and family information only.
    2. Early-behavior: new-user features + NumWebVisitsMonth.
    3. Existing-customer: new-user features + Recency and purchase-channel counts.

Usage:
    python regression/product_preference_regression.py --input ../marketing_campaign.xlsx
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")
os.environ.setdefault("OMP_NUM_THREADS", "4")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


RANDOM_STATE = 42

PRODUCT_SPENDING_COLUMNS = {
    "Wine": "MntWines",
    "Fruit": "MntFruits",
    "Meat": "MntMeatProducts",
    "Fish": "MntFishProducts",
    "Sweet": "MntSweetProducts",
    "Gold": "MntGoldProds",
}

PRODUCT_SHARE_COLUMNS = [f"{product}Share" for product in PRODUCT_SPENDING_COLUMNS]

BUSINESS_FEATURE_SETS = {
    "new_user": [
        "Age",
        "Income",
        "Education",
        "Marital_Status",
        "Kidhome",
        "Teenhome",
    ],
    "early_behavior": [
        "Age",
        "Income",
        "Education",
        "Marital_Status",
        "Kidhome",
        "Teenhome",
        "NumWebVisitsMonth",
    ],
    "existing_customer": [
        "Age",
        "Income",
        "Education",
        "Marital_Status",
        "Kidhome",
        "Teenhome",
        "Recency",
        "NumDealsPurchases",
        "NumWebPurchases",
        "NumCatalogPurchases",
        "NumStorePurchases",
    ],
}


def make_one_hot_encoder() -> OneHotEncoder:
    """Create OneHotEncoder for both recent and older scikit-learn versions."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def load_dataset(input_path: Path) -> pd.DataFrame:
    suffix = input_path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(input_path)
    if suffix == ".csv":
        df = pd.read_csv(input_path, sep="\t")
        if df.shape[1] == 1:
            df = pd.read_csv(input_path)
        return df
    raise ValueError("Unsupported input format. Use .xlsx, .xls, or .csv.")


def add_regression_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create age and product share targets."""
    df = df.copy()
    df["Dt_Customer"] = pd.to_datetime(df["Dt_Customer"], dayfirst=True, errors="coerce")
    reference_year = int(df["Dt_Customer"].dt.year.max())
    df["Age"] = reference_year - df["Year_Birth"]

    spending_cols = list(PRODUCT_SPENDING_COLUMNS.values())
    df["TotalSpending"] = df[spending_cols].sum(axis=1)

    for product, spending_col in PRODUCT_SPENDING_COLUMNS.items():
        share_col = f"{product}Share"
        df[share_col] = np.where(
            df["TotalSpending"] > 0,
            df[spending_col] / df["TotalSpending"],
            0,
        )

    rare_status = {"Alone", "Absurd", "YOLO"}
    df["Marital_Status"] = df["Marital_Status"].replace(
        {status: "Other" for status in rare_status}
    )

    return df


def clean_regression_data(df: pd.DataFrame) -> pd.DataFrame:
    """Apply common cleaning used by the team project."""
    df = df.copy()
    before_rows = len(df)
    df = df[df["Age"].between(18, 100)]
    df = df[df["Income"].isna() | (df["Income"] <= 200000)]
    df = df[df["TotalSpending"] > 0]
    df.attrs["removed_rows"] = before_rows - len(df)
    return df


def build_preprocessor(x: pd.DataFrame) -> ColumnTransformer:
    categorical_cols = x.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    numeric_cols = [col for col in x.columns if col not in categorical_cols]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", make_one_hot_encoder()),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_cols),
            ("categorical", categorical_pipeline, categorical_cols),
        ]
    )


def normalize_predictions(predictions: np.ndarray) -> np.ndarray:
    """Clip negative share predictions and normalize each row to sum to 1."""
    clipped = np.clip(predictions, 0, None)
    row_sums = clipped.sum(axis=1, keepdims=True)
    zero_rows = row_sums.squeeze() == 0
    if zero_rows.any():
        clipped[zero_rows] = 1 / clipped.shape[1]
        row_sums = clipped.sum(axis=1, keepdims=True)
    return clipped / row_sums


def rank_banners(
    predicted_shares: np.ndarray,
    average_train_shares: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rank product banners by average-corrected relative preference."""
    pred_df = pd.DataFrame(predicted_shares, columns=PRODUCT_SHARE_COLUMNS)
    relative_df = pred_df.div(average_train_shares[PRODUCT_SHARE_COLUMNS], axis=1)

    rank_rows = []
    for idx, row in relative_df.iterrows():
        ordered_share_cols = row.sort_values(ascending=False).index.tolist()
        ordered_products = [col.replace("Share", "") for col in ordered_share_cols]
        rank_rows.append(
            {
                "RecommendedBannerOrder": " > ".join(ordered_products),
                "Top1Product": ordered_products[0],
                "Top2Product": ordered_products[1],
                "Top3Product": ordered_products[2],
            }
        )

    return pred_df, pd.DataFrame(rank_rows)


def evaluate_business_model(
    df: pd.DataFrame,
    model_name: str,
    feature_columns: list[str],
    output_dir: Path,
) -> tuple[list[dict], pd.DataFrame]:
    """Train six linear regression models and build banner recommendations."""
    x = df[feature_columns]
    y = df[PRODUCT_SHARE_COLUMNS]

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
    )

    preprocessor = build_preprocessor(x)
    product_predictions = []
    metric_rows = []

    for target_col in PRODUCT_SHARE_COLUMNS:
        pipeline = Pipeline(
            steps=[
                ("preprocess", preprocessor),
                ("model", LinearRegression()),
            ]
        )
        pipeline.fit(x_train, y_train[target_col])
        y_pred_raw = pipeline.predict(x_test)
        product_predictions.append(y_pred_raw)

    y_pred = np.column_stack(product_predictions)
    y_pred_normalized = normalize_predictions(y_pred)

    for col_idx, target_col in enumerate(PRODUCT_SHARE_COLUMNS):
        actual = y_test[target_col].to_numpy()
        pred = y_pred_normalized[:, col_idx]
        metric_rows.append(
            {
                "business_model": model_name,
                "target": target_col,
                "mae": mean_absolute_error(actual, pred),
                "rmse": np.sqrt(mean_squared_error(actual, pred)),
                "r2": r2_score(actual, pred),
            }
        )

    average_train_shares = y_train.mean()
    pred_df, ranking_df = rank_banners(y_pred_normalized, average_train_shares)

    recommendation_sample = x_test.reset_index(drop=True).copy()
    for col in PRODUCT_SHARE_COLUMNS:
        recommendation_sample[f"Actual_{col}"] = y_test.reset_index(drop=True)[col]
        recommendation_sample[f"Predicted_{col}"] = pred_df[col]
        recommendation_sample[f"Relative_{col}"] = (
            pred_df[col] / average_train_shares[col]
        )
    recommendation_sample = pd.concat([recommendation_sample, ranking_df], axis=1)
    recommendation_sample.head(30).to_csv(
        output_dir / f"recommendation_sample_{model_name}.csv",
        index=False,
        encoding="utf-8-sig",
    )

    avg_share_row = average_train_shares.rename("train_average_share").reset_index()
    avg_share_row.columns = ["target", "train_average_share"]
    avg_share_row.to_csv(
        output_dir / f"train_average_product_shares_{model_name}.csv",
        index=False,
        encoding="utf-8-sig",
    )

    top1_counts = ranking_df["Top1Product"].value_counts().rename_axis("product").reset_index(name="count")
    top1_counts["business_model"] = model_name
    top1_counts.to_csv(
        output_dir / f"top1_banner_distribution_{model_name}.csv",
        index=False,
        encoding="utf-8-sig",
    )

    return metric_rows, top1_counts


def plot_metric_summary(metrics: pd.DataFrame, output_dir: Path) -> None:
    pivot = metrics.pivot(index="target", columns="business_model", values="rmse")
    ax = pivot.plot(kind="bar", figsize=(10, 5))
    ax.set_title("RMSE by Product Share Target")
    ax.set_xlabel("Product share target")
    ax.set_ylabel("RMSE")
    ax.legend(title="Business model")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(output_dir / "regression_rmse_by_target.png", dpi=160)
    plt.close()


def plot_top1_distribution(top1_counts: pd.DataFrame, output_dir: Path) -> None:
    pivot = top1_counts.pivot_table(
        index="product",
        columns="business_model",
        values="count",
        fill_value=0,
        aggfunc="sum",
    )
    ax = pivot.plot(kind="bar", figsize=(9, 5))
    ax.set_title("Top-1 Banner Product Distribution")
    ax.set_xlabel("Product")
    ax.set_ylabel("Number of test customers")
    ax.legend(title="Business model")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_dir / "top1_banner_distribution.png", dpi=160)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Product preference regression")
    parser.add_argument("--input", required=True, help="Path to marketing campaign data")
    parser.add_argument("--output", default="regression/outputs", help="Output directory")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_df = load_dataset(Path(args.input))
    df = add_regression_features(raw_df)
    df = clean_regression_data(df)

    summary = {
        "business_goal": "Predict product preference shares and rank personalized ad banners.",
        "raw_shape": list(raw_df.shape),
        "cleaned_shape": list(df.shape),
        "removed_rows": int(df.attrs.get("removed_rows", 0)),
        "targets": PRODUCT_SHARE_COLUMNS,
        "feature_sets": BUSINESS_FEATURE_SETS,
        "leakage_rule": "Mnt product spending columns and TotalSpending are target-only variables, not input features.",
    }

    metric_rows = []
    top1_frames = []
    for model_name, features in BUSINESS_FEATURE_SETS.items():
        rows, top1_counts = evaluate_business_model(df, model_name, features, output_dir)
        metric_rows.extend(rows)
        top1_frames.append(top1_counts)

    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(output_dir / "regression_product_share_metrics.csv", index=False, encoding="utf-8-sig")

    model_summary = (
        metrics.groupby("business_model")
        .agg(mean_mae=("mae", "mean"), mean_rmse=("rmse", "mean"), mean_r2=("r2", "mean"))
        .reset_index()
        .sort_values("mean_rmse")
    )
    model_summary.to_csv(output_dir / "regression_business_model_summary.csv", index=False, encoding="utf-8-sig")

    top1_counts = pd.concat(top1_frames, ignore_index=True)
    top1_counts.to_csv(output_dir / "top1_banner_distribution_all_models.csv", index=False, encoding="utf-8-sig")

    plot_metric_summary(metrics, output_dir)
    plot_top1_distribution(top1_counts, output_dir)

    with (output_dir / "regression_data_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Regression data summary")
    print(json.dumps(summary, indent=2))
    print("\nBusiness model summary")
    print(model_summary.round(4).to_string(index=False))
    print("\nProduct-level metrics")
    print(metrics.round(4).to_string(index=False))
    print("\nTop-1 banner distribution")
    print(top1_counts.to_string(index=False))
    print(f"\nSaved outputs to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
