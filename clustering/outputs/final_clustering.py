import argparse
import os
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")
os.environ.setdefault("OMP_NUM_THREADS", "9")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


RANDOM_STATE = 42
REFERENCE_YEAR = 2014
FINAL_K = 4
K_RANGE = range(2, 9)

CLUSTER_FEATURES = [
    "Income",
    "Age",
    "Recency",
    "TotalChildren",
    "TotalSpending",
    "TotalPurchases",
    "CampaignAcceptedTotal",
    "NumWebVisitsMonth",
    "CustomerTenure",
    "NumDealsPurchases",
    "NumWebPurchases",
    "NumCatalogPurchases",
    "NumStorePurchases",
]

PROFILE_BASE_COLUMNS = [
    "Income",
    "Age",
    "TotalChildren",
    "CustomerTenure",
    "Recency",
    "TotalSpending",
    "TotalPurchases",
    "CampaignAcceptedTotal",
    "NumWebVisitsMonth",
]

def load_data(file_path) -> pd.DataFrame:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Input file not found: {path}. "
            "Check the file name or pass --input with the correct path."
        )

    if path.suffix.lower() in [".xlsx", ".xls"]:
        df = pd.read_excel(path)
    elif path.suffix.lower() == ".csv":
        df = pd.read_csv(path, sep="\t")
        if df.shape[1] == 1:
            df = pd.read_csv(path)
    else:
        raise ValueError("Unsupported input format. Use .xlsx, .xls, or .csv")

    print("Dataset shape:", df.shape)
    print("Missing values:")
    print(df.isna().sum()[df.isna().sum() > 0])
    return df

def add_common_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["Dt_Customer"] = pd.to_datetime(
        df["Dt_Customer"], format="%d-%m-%Y", errors="coerce"
    )
    ref_date = df["Dt_Customer"].max()
    df["CustomerTenure"] = (ref_date - df["Dt_Customer"]).dt.days

    df["Age"] = REFERENCE_YEAR - df["Year_Birth"]

    df["TotalChildren"] = df["Kidhome"] + df["Teenhome"]

    spending_cols = [
        "MntWines",
        "MntFruits",
        "MntMeatProducts",
        "MntFishProducts",
        "MntSweetProducts",
        "MntGoldProds",
    ]
    df["TotalSpending"] = df[spending_cols].sum(axis=1)

    purchase_cols = [
        "NumDealsPurchases",
        "NumWebPurchases",
        "NumCatalogPurchases",
        "NumStorePurchases",
    ]
    df["TotalPurchases"] = df[purchase_cols].sum(axis=1)

    campaign_cols = [
        "AcceptedCmp1",
        "AcceptedCmp2",
        "AcceptedCmp3",
        "AcceptedCmp4",
        "AcceptedCmp5",
    ]
    df["CampaignAcceptedTotal"] = df[campaign_cols].sum(axis=1)

    if "Marital_Status" in df.columns:
        df["Marital_Status"] = df["Marital_Status"].replace(
            {"Alone": "Other", "Absurd": "Other", "YOLO": "Other"}
        )

    return df


def clean_common_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    before = len(df)

    df = df[(df["Age"] >= 18) & (df["Age"] <= 100)]

    df = df[(df["Income"].isna()) | (df["Income"] <= 200_000)]

    df = df[df["CustomerTenure"].notna()]

    after = len(df)
    print(f"Rows after common cleaning: {before} -> {after}")
    return df

def validate_clustering_columns(df: pd.DataFrame) -> None:
    missing = [col for col in CLUSTER_FEATURES if col not in df.columns]
    if missing:
        raise KeyError(f"Missing required clustering columns: {missing}")
    if "Response" not in df.columns:
        raise KeyError("Response column is required for post-clustering interpretation.")


def preprocess_clustering_features(df: pd.DataFrame):
    validate_clustering_columns(df)
    x_cluster = df[CLUSTER_FEATURES].copy()

    imputer = SimpleImputer(strategy="median")
    x_imputed = imputer.fit_transform(x_cluster)

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x_imputed)

    return x_cluster, x_scaled, imputer, scaler


def calculate_k_scores(x_scaled: np.ndarray, k_range=K_RANGE) -> pd.DataFrame:
    rows = []
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = kmeans.fit_predict(x_scaled)
        rows.append(
            {
                "k": k,
                "silhouette_score": silhouette_score(x_scaled, labels),
                "inertia": kmeans.inertia_,
            }
        )
    return pd.DataFrame(rows)


def run_clustering(df: pd.DataFrame, final_k: int = FINAL_K):
    _, x_scaled, _, _ = preprocess_clustering_features(df)

    kmeans = KMeans(n_clusters=final_k, random_state=RANDOM_STATE, n_init=10)
    clustered_df = df.copy()
    clustered_df["Cluster"] = kmeans.fit_predict(x_scaled)

    k_scores = calculate_k_scores(x_scaled, K_RANGE)
    return clustered_df, k_scores, x_scaled, kmeans

def _make_persona_labels(profile: pd.DataFrame):
    if len(profile) != 4:
        labels = {}
        for _, row in profile.iterrows():
            income = "High-income" if row["Income_mean"] >= profile["Income_mean"].median() else "Low-income"
            spending = (
                "High-spending"
                if row["TotalSpending_mean"] >= profile["TotalSpending_mean"].median()
                else "Low-spending"
            )
            response = (
                "Responsive"
                if row["Response_Rate"] >= profile["Response_Rate"].median()
                else "Less-responsive"
            )
            labels[int(row["Cluster"])] = f"{income}, {spending}, {response} customers"
        return labels

    temp = profile.copy()
    temp["value_score"] = (
        temp["Income_mean"].rank(pct=True)
        + temp["TotalSpending_mean"].rank(pct=True)
        + temp["TotalPurchases_mean"].rank(pct=True)
    )

    labels = {}

    low_cluster = int(temp.sort_values("value_score").iloc[0]["Cluster"])
    labels[low_cluster] = "Low-income, low-spending, less-responsive customers"

    remaining = temp[~temp["Cluster"].isin([low_cluster])].copy()

    high_resp_cluster = int(remaining.sort_values("Response_Rate", ascending=False).iloc[0]["Cluster"])
    labels[high_resp_cluster] = "High-income, high-spending, responsive customers"

    remaining = remaining[~remaining["Cluster"].isin([high_resp_cluster])].copy()

    high_value_cluster = int(remaining.sort_values("value_score", ascending=False).iloc[0]["Cluster"])
    labels[high_value_cluster] = "High-income, high-spending, less-responsive customers"

    remaining = remaining[~remaining["Cluster"].isin([high_value_cluster])].copy()

    mid_cluster = int(remaining.iloc[0]["Cluster"])
    labels[mid_cluster] = "Middle-income, middle-spending customers"

    return labels


def create_cluster_profile(clustered_df: pd.DataFrame) -> pd.DataFrame:
    profile = (
        clustered_df.groupby("Cluster")[PROFILE_BASE_COLUMNS]
        .mean()
        .round(2)
        .reset_index()
    )

    profile = profile.rename(
        columns={
            "Income": "Income_mean",
            "Age": "Age_mean",
            "TotalChildren": "TotalChildren_mean",
            "CustomerTenure": "CustomerTenure_mean",
            "Recency": "Recency_mean",
            "TotalSpending": "TotalSpending_mean",
            "TotalPurchases": "TotalPurchases_mean",
            "CampaignAcceptedTotal": "CampaignAcceptedTotal_mean",
            "NumWebVisitsMonth": "NumWebVisitsMonth_mean",
        }
    )

    counts = clustered_df.groupby("Cluster").size().rename("CustomerCount")
    response_rate = clustered_df.groupby("Cluster")["Response"].mean().round(3).rename("Response_Rate")

    profile = profile.merge(counts, on="Cluster").merge(response_rate, on="Cluster")

    persona_map = _make_persona_labels(profile)
    profile["Persona_Label"] = profile["Cluster"].map(persona_map)

    ordered_cols = [
        "Cluster",
        "CustomerCount",
        "Income_mean",
        "Age_mean",
        "TotalChildren_mean",
        "CustomerTenure_mean",
        "Recency_mean",
        "TotalSpending_mean",
        "TotalPurchases_mean",
        "CampaignAcceptedTotal_mean",
        "NumWebVisitsMonth_mean",
        "Response_Rate",
        "Persona_Label",
    ]
    return profile[ordered_cols].sort_values("Cluster").reset_index(drop=True)

def plot_silhouette_scores(k_scores: pd.DataFrame, output_dir: Path) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(k_scores["k"], k_scores["silhouette_score"], marker="o")
    plt.axvline(FINAL_K, linestyle="--", alpha=0.7, label=f"Chosen k={FINAL_K}")
    plt.title("Silhouette Score by k")
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Silhouette score")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "silhouette_score_plot.png", dpi=150)
    plt.close()


def plot_elbow(k_scores: pd.DataFrame, output_dir: Path) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(k_scores["k"], k_scores["inertia"], marker="o")
    plt.axvline(FINAL_K, linestyle="--", alpha=0.7, label=f"Chosen k={FINAL_K}")
    plt.title("Elbow Method")
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Inertia")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "elbow_plot.png", dpi=150)
    plt.close()


def plot_response_rate(profile: pd.DataFrame, output_dir: Path) -> None:
    plt.figure(figsize=(8, 5))
    x_labels = profile["Cluster"].astype(str)
    plt.bar(x_labels, profile["Response_Rate"])
    plt.title("Response Rate by Cluster")
    plt.xlabel("Cluster")
    plt.ylabel("Response rate")
    plt.ylim(0, max(0.05, profile["Response_Rate"].max() * 1.2))
    plt.tight_layout()
    plt.savefig(output_dir / "response_rate_by_cluster.png", dpi=150)
    plt.close()


def plot_cluster_heatmap(profile: pd.DataFrame, output_dir: Path) -> None:
    heatmap_cols = {
        "Income_mean": "Income",
        "Age_mean": "Age",
        "TotalChildren_mean": "Children",
        "CustomerTenure_mean": "Tenure",
        "Recency_mean": "Recency",
        "TotalSpending_mean": "Spending",
        "TotalPurchases_mean": "Purchases",
        "CampaignAcceptedTotal_mean": "Accepted",
        "NumWebVisitsMonth_mean": "Web visits",
        "Response_Rate": "Response",
    }
    columns = list(heatmap_cols.keys())
    heatmap_data = profile.set_index("Cluster")[columns].copy()

    standardized = (heatmap_data - heatmap_data.mean()) / heatmap_data.std(ddof=0)
    standardized = standardized.replace([np.inf, -np.inf], 0).fillna(0)

    fig, ax = plt.subplots(figsize=(13, 5.5))
    image = ax.imshow(standardized.values, aspect="auto", cmap="RdBu_r", vmin=-1.8, vmax=1.8)
    fig.colorbar(image, ax=ax, label="Standardized value")

    ax.set_xticks(range(len(columns)))
    ax.set_xticklabels([heatmap_cols[col] for col in columns], rotation=35, ha="right")
    ax.set_yticks(range(len(profile)))
    ax.set_yticklabels([f"Cluster {c}" for c in profile["Cluster"]])

    for row_idx in range(heatmap_data.shape[0]):
        for col_idx in range(heatmap_data.shape[1]):
            value = heatmap_data.iloc[row_idx, col_idx]
            text_color = "white" if abs(standardized.iloc[row_idx, col_idx]) > 1.0 else "black"
            ax.text(col_idx, row_idx, f"{value:.1f}", ha="center", va="center", color=text_color, fontsize=8)

    plt.title("Cluster Profile Heatmap")
    plt.tight_layout()
    plt.savefig(output_dir / "cluster_profile_heatmap.png", dpi=150)
    plt.close()


def plot_pca_clusters(
    x_scaled: np.ndarray, clustered_df: pd.DataFrame, output_dir: Path
) -> pd.DataFrame:
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    coords = pca.fit_transform(x_scaled)

    pca_df = pd.DataFrame(coords, columns=["PC1", "PC2"])
    pca_df["Cluster"] = clustered_df["Cluster"].values

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        pca_df["PC1"],
        pca_df["PC2"],
        c=pca_df["Cluster"],
        alpha=0.7,
    )
    plt.title(
        "Customer Segments by PCA "
        f"(Explained variance: {pca.explained_variance_ratio_.sum():.1%})"
    )
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend(*scatter.legend_elements(), title="Cluster")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "clustering_pca.png", dpi=150)
    plt.close()

    return pca_df

def main() -> None:
    parser = argparse.ArgumentParser(description="Final K-Means clustering workflow")
    parser.add_argument(
        "--input",
        default="marketing_campaign.xlsx",
        help="Path to marketing campaign data (.xlsx or .csv).",
    )
    parser.add_argument(
        "--output",
        default="outputs",
        help="Directory where output csv/png files will be saved.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_data(args.input)
    df = add_common_features(df)
    df = clean_common_data(df)

    clustered_df, k_scores, x_scaled, _ = run_clustering(df, final_k=FINAL_K)
    profile = create_cluster_profile(clustered_df)

    k_scores.to_csv(output_dir / "clustering_k_scores.csv", index=False, encoding="utf-8-sig")
    profile.to_csv(output_dir / "clustering_profile.csv", index=False, encoding="utf-8-sig")
    clustered_df.to_csv(
        output_dir / "marketing_campaign_with_clusters.csv",
        index=False,
        encoding="utf-8-sig",
    )

    plot_silhouette_scores(k_scores, output_dir)
    plot_elbow(k_scores, output_dir)
    plot_response_rate(profile, output_dir)
    plot_cluster_heatmap(profile, output_dir)
    plot_pca_clusters(x_scaled, clustered_df, output_dir)

    print("\nClustering k scores:")
    print(k_scores.round(4).to_string(index=False))
    print("\nCluster profile:")
    print(profile.to_string(index=False))
    print(f"\nSaved output files to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
