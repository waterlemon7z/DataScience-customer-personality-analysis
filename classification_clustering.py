"""
Customer Personality Analysis: classification + clustering

Usage:
    python classification_clustering.py --input marketing_campaign.xlsx
    python classification_clustering.py --input "C:/path/to/marketing_campaign.xlsx" --output outputs

Outputs:
    outputs/classification_results.csv
    outputs/confusion_matrix_<model>.csv
    outputs/clustering_k_scores.csv
    outputs/cluster_profile.csv
    outputs/clustered_customers.csv
"""

from __future__ import annotations

import os
import argparse
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")
os.environ.setdefault("OMP_NUM_THREADS", "4")

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    silhouette_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier


RANDOM_STATE = 42


def make_one_hot_encoder() -> OneHotEncoder:
    """Create OneHotEncoder that works across recent and older sklearn versions."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def load_data(input_path: Path) -> pd.DataFrame:
    df = pd.read_excel(input_path)
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["Dt_Customer"] = pd.to_datetime(df["Dt_Customer"], dayfirst=True, errors="coerce")
    reference_year = 2014
    reference_date = df["Dt_Customer"].max()

    spending_cols = [
        "MntWines",
        "MntFruits",
        "MntMeatProducts",
        "MntFishProducts",
        "MntSweetProducts",
        "MntGoldProds",
    ]
    purchase_cols = [
        "NumDealsPurchases",
        "NumWebPurchases",
        "NumCatalogPurchases",
        "NumStorePurchases",
    ]
    campaign_cols = [
        "AcceptedCmp1",
        "AcceptedCmp2",
        "AcceptedCmp3",
        "AcceptedCmp4",
        "AcceptedCmp5",
    ]

    df["Age"] = reference_year - df["Year_Birth"]
    df["Customer_Days"] = (reference_date - df["Dt_Customer"]).dt.days
    df["TotalSpending"] = df[spending_cols].sum(axis=1)
    df["TotalPurchases"] = df[purchase_cols].sum(axis=1)
    df["Children"] = df["Kidhome"] + df["Teenhome"]
    df["AcceptedCmpTotal"] = df[campaign_cols].sum(axis=1)

    rare_marital_status = {"Alone", "Absurd", "YOLO"}
    df["Marital_Status"] = df["Marital_Status"].replace(
        {value: "Other" for value in rare_marital_status}
    )

    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Remove unrealistic age outliers and the most extreme income outliers.
    income_99 = df["Income"].quantile(0.99)
    df = df[df["Age"].between(18, 100)]
    df = df[df["Income"].isna() | (df["Income"] < income_99)]

    # Constants and identifiers do not help prediction.
    drop_cols = ["ID", "Year_Birth", "Dt_Customer", "Z_CostContact", "Z_Revenue"]
    return df.drop(columns=[col for col in drop_cols if col in df.columns])


def build_preprocessor(x: pd.DataFrame) -> ColumnTransformer:
    categorical_cols = x.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    numeric_cols = [col for col in x.columns if col not in categorical_cols]

    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", make_one_hot_encoder()),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ]
    )


def evaluate_classification(df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    y = df["Response"].astype(int)
    x = df.drop(columns=["Response"])

    preprocessor = build_preprocessor(x)
    models = {
        "logistic_regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "decision_tree": DecisionTreeClassifier(
            max_depth=5,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "knn": KNeighborsClassifier(n_neighbors=15),
    }

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    rows = []

    for model_name, model in models.items():
        pipeline = Pipeline(
            steps=[
                ("preprocess", preprocessor),
                ("model", model),
            ]
        )

        cv_scores = cross_validate(
            pipeline,
            x,
            y,
            cv=cv,
            scoring=["accuracy", "precision", "recall", "f1", "roc_auc"],
            error_score="raise",
        )

        pipeline.fit(x_train, y_train)
        y_pred = pipeline.predict(x_test)

        if hasattr(pipeline, "predict_proba"):
            y_score = pipeline.predict_proba(x_test)[:, 1]
        else:
            y_score = y_pred

        cm = confusion_matrix(y_test, y_pred)
        pd.DataFrame(
            cm,
            index=["actual_0", "actual_1"],
            columns=["pred_0", "pred_1"],
        ).to_csv(output_dir / f"confusion_matrix_{model_name}.csv", encoding="utf-8-sig")

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
    results.to_csv(output_dir / "classification_results.csv", index=False, encoding="utf-8-sig")
    return results


def run_clustering(df: pd.DataFrame, output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    cluster_features = [
        "Income",
        "Age",
        "Children",
        "Recency",
        "TotalSpending",
        "TotalPurchases",
        "NumWebVisitsMonth",
        "AcceptedCmpTotal",
        "NumDealsPurchases",
        "NumWebPurchases",
        "NumCatalogPurchases",
        "NumStorePurchases",
    ]

    x_cluster = df[cluster_features].copy()
    cluster_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    x_scaled = cluster_pipe.fit_transform(x_cluster)

    k_rows = []
    best_k = None
    best_score = -1

    for k in range(2, 8):
        kmeans = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=20)
        labels = kmeans.fit_predict(x_scaled)
        score = silhouette_score(x_scaled, labels)
        k_rows.append({"k": k, "silhouette_score": score, "inertia": kmeans.inertia_})

        if score > best_score:
            best_score = score
            best_k = k

    k_scores = pd.DataFrame(k_rows)
    k_scores.to_csv(output_dir / "clustering_k_scores.csv", index=False, encoding="utf-8-sig")

    final_kmeans = KMeans(n_clusters=best_k, random_state=RANDOM_STATE, n_init=20)
    df_clustered = df.copy()
    df_clustered["Cluster"] = final_kmeans.fit_predict(x_scaled)

    profile_cols = [
        "Income",
        "Age",
        "Children",
        "Recency",
        "TotalSpending",
        "TotalPurchases",
        "NumWebVisitsMonth",
        "AcceptedCmpTotal",
        "Response",
    ]
    cluster_profile = df_clustered.groupby("Cluster")[profile_cols].mean().round(2)
    cluster_profile["customer_count"] = df_clustered.groupby("Cluster").size()
    cluster_profile = cluster_profile.reset_index()

    cluster_profile.to_csv(output_dir / "cluster_profile.csv", index=False, encoding="utf-8-sig")
    df_clustered.to_csv(output_dir / "clustered_customers.csv", index=False, encoding="utf-8-sig")

    return k_scores, cluster_profile


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to marketing_campaign.xlsx")
    parser.add_argument("--output", default="outputs", help="Directory to save result files")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_data(input_path)
    df = add_features(df)
    df = clean_data(df)

    print(f"Rows after cleaning: {len(df):,}")
    print(f"Response positive rate: {df['Response'].mean():.4f}")

    classification_results = evaluate_classification(df, output_dir)
    k_scores, cluster_profile = run_clustering(df, output_dir)

    print("\nClassification results")
    print(classification_results.round(4).to_string(index=False))

    print("\nClustering k scores")
    print(k_scores.round(4).to_string(index=False))

    print("\nCluster profile")
    print(cluster_profile.to_string(index=False))

    print(f"\nSaved result files to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
