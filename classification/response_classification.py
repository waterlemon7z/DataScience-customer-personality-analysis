"""
Customer Personality Analysis - response classification.

This script predicts whether a customer accepted the last campaign
(`Response`) using demographic, family, tenure, purchase behavior, and
previous campaign response features.

Key project choices:
    - No data augmentation is used.
    - StratifiedKFold is used because Response is imbalanced.
    - class_weight="balanced" is tested for Logistic Regression and
      Decision Tree.
    - CampaignAcceptedTotal is built only from AcceptedCmp1-5, never Response.

Usage:
    python classification/response_classification.py --input ../marketing_campaign.xlsx
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
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier


RANDOM_STATE = 42

SPENDING_COLUMNS = [
    "MntWines",
    "MntFruits",
    "MntMeatProducts",
    "MntFishProducts",
    "MntSweetProducts",
    "MntGoldProds",
]

PURCHASE_COLUMNS = [
    "NumDealsPurchases",
    "NumWebPurchases",
    "NumCatalogPurchases",
    "NumStorePurchases",
]

PREVIOUS_CAMPAIGN_COLUMNS = [
    "AcceptedCmp1",
    "AcceptedCmp2",
    "AcceptedCmp3",
    "AcceptedCmp4",
    "AcceptedCmp5",
]


def make_one_hot_encoder() -> OneHotEncoder:
    """Create OneHotEncoder for both recent and older scikit-learn versions."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def load_dataset(input_path: Path) -> pd.DataFrame:
    """Load xlsx/csv/tsv data."""
    suffix = input_path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(input_path)
    if suffix == ".csv":
        return pd.read_csv(input_path)
    if suffix in {".txt", ".tsv"}:
        return pd.read_csv(input_path, sep="\t")
    raise ValueError(f"Unsupported file type: {input_path.suffix}")


def add_project_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add proposal-based derived features."""
    df = df.copy()
    df["Dt_Customer"] = pd.to_datetime(df["Dt_Customer"], dayfirst=True, errors="coerce")

    # Use the dataset period, not the current year, to represent customer age at observation time.
    reference_year = int(df["Dt_Customer"].dt.year.max())
    reference_date = df["Dt_Customer"].max()

    df["Age"] = reference_year - df["Year_Birth"]
    df["CustomerTenure"] = (reference_date - df["Dt_Customer"]).dt.days
    df["TotalChildren"] = df["Kidhome"] + df["Teenhome"]
    df["TotalSpending"] = df[SPENDING_COLUMNS].sum(axis=1)
    df["TotalPurchases"] = df[PURCHASE_COLUMNS].sum(axis=1)
    df["CampaignAcceptedTotal"] = df[PREVIOUS_CAMPAIGN_COLUMNS].sum(axis=1)

    rare_status = {"Alone", "Absurd", "YOLO"}
    df["Marital_Status"] = df["Marital_Status"].replace(
        {status: "Other" for status in rare_status}
    )

    return df


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Remove identifiers, constants, and extreme outliers."""
    df = df.copy()

    before_rows = len(df)
    df = df[df["Age"].between(18, 100)]
    df = df[df["Income"].isna() | (df["Income"] <= 200000)]

    drop_columns = [
        "ID",
        "Year_Birth",
        "Dt_Customer",
        "Z_CostContact",
        "Z_Revenue",
    ]
    df = df.drop(columns=[col for col in drop_columns if col in df.columns])
    df.attrs["removed_rows"] = before_rows - len(df)

    return df


def build_feature_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split data into X and y for Response classification."""
    y = df["Response"].astype(int)

    feature_columns = [
        "Education",
        "Marital_Status",
        "Income",
        "Age",
        "Kidhome",
        "Teenhome",
        "TotalChildren",
        "CustomerTenure",
        "Recency",
        "MntWines",
        "MntFruits",
        "MntMeatProducts",
        "MntFishProducts",
        "MntSweetProducts",
        "MntGoldProds",
        "TotalSpending",
        "NumDealsPurchases",
        "NumWebPurchases",
        "NumCatalogPurchases",
        "NumStorePurchases",
        "NumWebVisitsMonth",
        "TotalPurchases",
        "AcceptedCmp1",
        "AcceptedCmp2",
        "AcceptedCmp3",
        "AcceptedCmp4",
        "AcceptedCmp5",
        "CampaignAcceptedTotal",
        "Complain",
    ]
    existing_columns = [col for col in feature_columns if col in df.columns]
    return df[existing_columns], y


def build_preprocessor(x: pd.DataFrame) -> ColumnTransformer:
    """Create train-fold-only preprocessing pipeline."""
    categorical_columns = x.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    numeric_columns = [col for col in x.columns if col not in categorical_columns]

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
            ("numeric", numeric_pipeline, numeric_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ]
    )


def model_candidates() -> dict[str, object]:
    """Models limited to simple course-level classifiers."""
    return {
        "dummy_most_frequent": DummyClassifier(strategy="most_frequent"),
        "logistic_regression": LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        "logistic_regression_balanced": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "knn_k15": KNeighborsClassifier(n_neighbors=15),
        "decision_tree_balanced": DecisionTreeClassifier(
            max_depth=5,
            min_samples_leaf=20,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
    }


def save_confusion_matrix_plot(
    matrix: np.ndarray,
    model_name: str,
    output_dir: Path,
) -> None:
    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=["No Response", "Response"],
    )
    display.plot(cmap="Blues", values_format="d")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.tight_layout()
    plt.savefig(output_dir / f"confusion_matrix_{model_name}.png", dpi=160)
    plt.close()


def evaluate_models(x: pd.DataFrame, y: pd.Series, output_dir: Path) -> pd.DataFrame:
    """Run stratified CV and holdout-test evaluation."""
    output_dir.mkdir(parents=True, exist_ok=True)

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scoring = {
        "accuracy": "accuracy",
        "precision": make_scorer(precision_score, zero_division=0),
        "recall": make_scorer(recall_score, zero_division=0),
        "f1": make_scorer(f1_score, zero_division=0),
        "roc_auc": "roc_auc",
    }
    rows = []

    for model_name, model in model_candidates().items():
        pipeline = Pipeline(
            steps=[
                ("preprocess", build_preprocessor(x)),
                ("model", model),
            ]
        )

        cv_scores = cross_validate(
            pipeline,
            x,
            y,
            cv=cv,
            scoring=scoring,
            error_score="raise",
        )

        pipeline.fit(x_train, y_train)
        y_pred = pipeline.predict(x_test)

        if hasattr(pipeline, "predict_proba"):
            y_score = pipeline.predict_proba(x_test)[:, 1]
        else:
            y_score = y_pred

        matrix = confusion_matrix(y_test, y_pred)
        pd.DataFrame(
            matrix,
            index=["actual_0", "actual_1"],
            columns=["pred_0", "pred_1"],
        ).to_csv(output_dir / f"confusion_matrix_{model_name}.csv", encoding="utf-8-sig")
        save_confusion_matrix_plot(matrix, model_name, output_dir)

        report = classification_report(
            y_test,
            y_pred,
            target_names=["No Response", "Response"],
            zero_division=0,
            output_dict=True,
        )
        with (output_dir / f"classification_report_{model_name}.json").open(
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(report, f, indent=2)

        prediction_sample = x_test.copy()
        prediction_sample["actual_Response"] = y_test.values
        prediction_sample["predicted_Response"] = y_pred
        prediction_sample.head(30).to_csv(
            output_dir / f"prediction_sample_{model_name}.csv",
            index=False,
            encoding="utf-8-sig",
        )

        rows.append(
            {
                "model": model_name,
                "test_accuracy": accuracy_score(y_test, y_pred),
                "test_precision": precision_score(y_test, y_pred, zero_division=0),
                "test_recall": recall_score(y_test, y_pred, zero_division=0),
                "test_f1": f1_score(y_test, y_pred, zero_division=0),
                "test_roc_auc": roc_auc_score(y_test, y_score),
                "cv_accuracy_mean": cv_scores["test_accuracy"].mean(),
                "cv_precision_mean": cv_scores["test_precision"].mean(),
                "cv_recall_mean": cv_scores["test_recall"].mean(),
                "cv_f1_mean": cv_scores["test_f1"].mean(),
                "cv_roc_auc_mean": cv_scores["test_roc_auc"].mean(),
            }
        )

    results = pd.DataFrame(rows).sort_values("test_f1", ascending=False)
    results.to_csv(output_dir / "classification_model_results.csv", index=False, encoding="utf-8-sig")
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to marketing_campaign file")
    parser.add_argument("--output", default="classification/outputs", help="Output directory")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output)

    raw_df = load_dataset(input_path)
    df = add_project_features(raw_df)
    df = clean_dataset(df)
    x, y = build_feature_matrix(df)

    summary = {
        "raw_shape": list(raw_df.shape),
        "cleaned_shape": list(df.shape),
        "removed_outlier_rows": int(df.attrs.get("removed_rows", 0)),
        "feature_count": int(x.shape[1]),
        "target_counts": y.value_counts().sort_index().to_dict(),
        "target_rates": y.value_counts(normalize=True).sort_index().round(4).to_dict(),
        "missing_values_used_for_imputation": x.isna().sum()[x.isna().sum() > 0].to_dict(),
        "features": list(x.columns),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "classification_data_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    results = evaluate_models(x, y, output_dir)

    print("Data summary")
    print(json.dumps(summary, indent=2))
    print("\nClassification model results")
    print(results.round(4).to_string(index=False))
    print(f"\nSaved outputs to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
