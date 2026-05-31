import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


RANDOM_STATE = 42  # Fixed random seed for reproducible clustering results
REFERENCE_YEAR = 2014  # Reference year used to calculate customer age
FINAL_K = 4  # Final k selected for interpretable customer personas
K_RANGE = range(2, 9)  # Candidate k values used for silhouette and inertia comparison

CLUSTER_FEATURES = [  # Input features for K-Means; Response is excluded to keep clustering unsupervised
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

PROFILE_BASE_COLUMNS = [  # Main variables summarized after clustering for customer profile interpretation
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
    path = Path(file_path)  # Convert input path string to a Path object for safer file handling
    if not path.exists():  # Stop execution early if the input file path is wrong
        raise FileNotFoundError(
            f"Input file not found: {path}. "
            "Check the file name or pass --input with the correct path."
        )

    if path.suffix.lower() in [".xlsx", ".xls"]:  # Read Excel files used in the team project
        df = pd.read_excel(path)
    elif path.suffix.lower() == ".csv":  # Read Kaggle-style csv files when the dataset is saved as text/csv
        df = pd.read_csv(path, sep="\t")
        if df.shape[1] == 1:  # Retry normal comma-separated csv if tab-separated reading fails
            df = pd.read_csv(path)
    else:
        raise ValueError("Unsupported input format. Use .xlsx, .xls, or .csv")

    print("Dataset shape:", df.shape)  # Show the original data size before preprocessing
    print("Missing values:")
    print(df.isna().sum()[df.isna().sum() > 0])  # Print only columns that contain missing values
    return df


def add_common_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()  # Work on a copy to avoid changing the original dataframe directly

    df["Dt_Customer"] = pd.to_datetime(  # Convert enrollment date so customer tenure can be calculated
        df["Dt_Customer"], format="%d-%m-%Y", errors="coerce"
    )
    ref_date = df["Dt_Customer"].max()  # Use the latest enrollment date as the reference point
    df["CustomerTenure"] = (ref_date - df["Dt_Customer"]).dt.days  # Longer value means longer customer history

    df["Age"] = REFERENCE_YEAR - df["Year_Birth"]  # Convert birth year into customer age

    df["TotalChildren"] = df["Kidhome"] + df["Teenhome"]  # Combine children and teenagers at home

    spending_cols = [  # Product spending columns used to create total spending
        "MntWines",
        "MntFruits",
        "MntMeatProducts",
        "MntFishProducts",
        "MntSweetProducts",
        "MntGoldProds",
    ]
    df["TotalSpending"] = df[spending_cols].sum(axis=1)  # Total amount spent across all product categories

    purchase_cols = [  # Purchase channel count columns used to create total purchase activity
        "NumDealsPurchases",
        "NumWebPurchases",
        "NumCatalogPurchases",
        "NumStorePurchases",
    ]
    df["TotalPurchases"] = df[purchase_cols].sum(axis=1)  # Total number of purchases across all channels

    campaign_cols = [  # Past campaign acceptance columns, excluding the current Response target
        "AcceptedCmp1",
        "AcceptedCmp2",
        "AcceptedCmp3",
        "AcceptedCmp4",
        "AcceptedCmp5",
    ]
    df["CampaignAcceptedTotal"] = df[campaign_cols].sum(axis=1)  # Past campaign acceptance count

    if "Marital_Status" in df.columns:  # Group rare categories for consistency with team preprocessing
        df["Marital_Status"] = df["Marital_Status"].replace(
            {"Alone": "Other", "Absurd": "Other", "YOLO": "Other"}
        )

    return df


def clean_common_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()  # Keep the original dataframe unchanged
    before = len(df)  # Store row count before cleaning for comparison

    df = df[(df["Age"] >= 18) & (df["Age"] <= 100)]  # Keep only realistic adult customer ages

    df = df[(df["Income"].isna()) | (df["Income"] <= 200_000)]  # Remove extreme income outliers but keep missing values for imputation

    df = df[df["CustomerTenure"].notna()]  # Remove rows where date parsing failed and tenure cannot be calculated

    after = len(df)  # Store row count after cleaning
    print(f"Rows after common cleaning: {before} -> {after}")
    return df


def validate_clustering_columns(df: pd.DataFrame) -> None:
    missing = [col for col in CLUSTER_FEATURES if col not in df.columns]  # Check whether all required clustering features exist
    if missing:
        raise KeyError(f"Missing required clustering columns: {missing}")
    if "Response" not in df.columns:  # Response is needed only after clustering to interpret response rate
        raise KeyError("Response column is required for post-clustering interpretation.")


def preprocess_clustering_features(df: pd.DataFrame):
    validate_clustering_columns(df)  # Validate feature availability before model preprocessing
    x_cluster = df[CLUSTER_FEATURES].copy()  # Use only clustering features; Response is not an input variable

    imputer = SimpleImputer(strategy="median")  # Fill missing numeric values using the median, which is robust to outliers
    x_imputed = imputer.fit_transform(x_cluster)

    scaler = StandardScaler()  # Scale variables so K-Means distance is not dominated by large-value features
    x_scaled = scaler.fit_transform(x_imputed)

    return x_cluster, x_scaled, imputer, scaler


def calculate_k_scores(x_scaled: np.ndarray, k_range=K_RANGE) -> pd.DataFrame:
    rows = []  # Store silhouette score and inertia for each candidate k
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)  # Fit K-Means for each candidate k
        labels = kmeans.fit_predict(x_scaled)
        rows.append(
            {
                "k": k,
                "silhouette_score": silhouette_score(x_scaled, labels),  # Higher value means clearer cluster separation
                "inertia": kmeans.inertia_,  # Lower value means smaller within-cluster distance
            }
        )
    return pd.DataFrame(rows)


def run_clustering(df: pd.DataFrame, final_k: int = FINAL_K):
    _, x_scaled, _, _ = preprocess_clustering_features(df)  # Prepare scaled features before distance-based clustering

    kmeans = KMeans(n_clusters=final_k, random_state=RANDOM_STATE, n_init=10)  # Final model uses k=4 for marketing personas
    clustered_df = df.copy()
    clustered_df["Cluster"] = kmeans.fit_predict(x_scaled)  # Assign one cluster label to each customer

    k_scores = calculate_k_scores(x_scaled, K_RANGE)  # Calculate supporting k-selection metrics for k=2 to k=8
    return clustered_df, k_scores, x_scaled, kmeans


def _make_persona_labels(profile: pd.DataFrame):
    if len(profile) != 4:  # Fallback rule when the final number of clusters is not four
        labels = {}
        for _, row in profile.iterrows():
            income = "High-income" if row["Income_mean"] >= profile["Income_mean"].median() else "Low-income"  # Compare income to cluster median
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

    temp = profile.copy()  # Create a temporary profile table for persona assignment
    temp["value_score"] = (  # Approximate customer value using income, spending, and purchase activity ranks
        temp["Income_mean"].rank(pct=True)
        + temp["TotalSpending_mean"].rank(pct=True)
        + temp["TotalPurchases_mean"].rank(pct=True)
    )

    labels = {}  # Store persona label for each cluster number

    low_cluster = int(temp.sort_values("value_score").iloc[0]["Cluster"])  # Lowest value segment becomes low-income and low-spending persona
    labels[low_cluster] = "Low-income, low-spending, less-responsive customers"

    remaining = temp[~temp["Cluster"].isin([low_cluster])].copy()  # Remove already labeled low-value cluster

    high_resp_cluster = int(remaining.sort_values("Response_Rate", ascending=False).iloc[0]["Cluster"])  # Highest response segment becomes responsive persona
    labels[high_resp_cluster] = "High-income, high-spending, responsive customers"

    remaining = remaining[~remaining["Cluster"].isin([high_resp_cluster])].copy()  # Remove already labeled responsive cluster

    high_value_cluster = int(remaining.sort_values("value_score", ascending=False).iloc[0]["Cluster"])  # Highest remaining value becomes high-value low-response persona
    labels[high_value_cluster] = "High-income, high-spending, less-responsive customers"

    remaining = remaining[~remaining["Cluster"].isin([high_value_cluster])].copy()  # Leave the middle segment for the last persona

    mid_cluster = int(remaining.iloc[0]["Cluster"])  # Remaining cluster is interpreted as the middle segment
    labels[mid_cluster] = "Middle-income, middle-spending customers"

    return labels


def create_cluster_profile(clustered_df: pd.DataFrame) -> pd.DataFrame:
    profile = (  # Average values summarize each cluster for business interpretation
        clustered_df.groupby("Cluster")[PROFILE_BASE_COLUMNS]
        .mean()
        .round(2)
        .reset_index()
    )

    profile = profile.rename(  # Rename columns so the profile clearly shows that values are cluster means
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

    counts = clustered_df.groupby("Cluster").size().rename("CustomerCount")  # Count customers in each segment
    response_rate = clustered_df.groupby("Cluster")["Response"].mean().round(3).rename("Response_Rate")  # Use Response only after clustering for interpretation

    profile = profile.merge(counts, on="Cluster").merge(response_rate, on="Cluster")  # Combine profile means, cluster size, and response rate

    persona_map = _make_persona_labels(profile)  # Assign readable persona names based on cluster profiles
    profile["Persona_Label"] = profile["Cluster"].map(persona_map)

    ordered_cols = [  # Arrange output columns in the order used for the report
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
    plt.figure(figsize=(8, 5))  # Create a figure for comparing silhouette scores by k
    plt.plot(k_scores["k"], k_scores["silhouette_score"], marker="o")
    plt.axvline(FINAL_K, linestyle="--", alpha=0.7, label=f"Chosen k={FINAL_K}")  # Mark the selected final k
    plt.title("Silhouette Score by k")
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Silhouette score")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "silhouette_score_plot.png", dpi=150)  # Save plot for PPT or report output
    plt.close()


def plot_elbow(k_scores: pd.DataFrame, output_dir: Path) -> None:
    plt.figure(figsize=(8, 5))  # Create a figure for the elbow method
    plt.plot(k_scores["k"], k_scores["inertia"], marker="o")
    plt.axvline(FINAL_K, linestyle="--", alpha=0.7, label=f"Chosen k={FINAL_K}")  # Mark k=4 even if inertia keeps decreasing
    plt.title("Elbow Method")
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Inertia")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "elbow_plot.png", dpi=150)  # Save inertia plot as a k-selection supporting output
    plt.close()


def plot_response_rate(profile: pd.DataFrame, output_dir: Path) -> None:
    plt.figure(figsize=(8, 5))  # Create a bar chart for cluster-level campaign response rate
    x_labels = profile["Cluster"].astype(str)
    plt.bar(x_labels, profile["Response_Rate"])
    plt.title("Response Rate by Cluster")
    plt.xlabel("Cluster")
    plt.ylabel("Response rate")
    plt.ylim(0, max(0.05, profile["Response_Rate"].max() * 1.2))
    plt.tight_layout()
    plt.savefig(output_dir / "response_rate_by_cluster.png", dpi=150)  # Save response comparison across clusters
    plt.close()


def plot_cluster_heatmap(profile: pd.DataFrame, output_dir: Path) -> None:
    heatmap_cols = {  # Short display names make the heatmap easier to read
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
    heatmap_data = profile.set_index("Cluster")[columns].copy()  # Use cluster profile values as heatmap data

    standardized = (heatmap_data - heatmap_data.mean()) / heatmap_data.std(ddof=0)  # Standardize profile columns for fair color comparison
    standardized = standardized.replace([np.inf, -np.inf], 0).fillna(0)  # Prevent invalid values from breaking the heatmap

    fig, ax = plt.subplots(figsize=(13, 5.5))
    image = ax.imshow(standardized.values, aspect="auto", cmap="RdBu_r", vmin=-1.8, vmax=1.8)  # Show standardized profile strength by color
    fig.colorbar(image, ax=ax, label="Standardized value")

    ax.set_xticks(range(len(columns)))
    ax.set_xticklabels([heatmap_cols[col] for col in columns], rotation=35, ha="right")
    ax.set_yticks(range(len(profile)))
    ax.set_yticklabels([f"Cluster {c}" for c in profile["Cluster"]])

    for row_idx in range(heatmap_data.shape[0]):
        for col_idx in range(heatmap_data.shape[1]):
            value = heatmap_data.iloc[row_idx, col_idx]
            text_color = "white" if abs(standardized.iloc[row_idx, col_idx]) > 1.0 else "black"  # Keep labels readable on dark cells
            ax.text(col_idx, row_idx, f"{value:.1f}", ha="center", va="center", color=text_color, fontsize=8)

    plt.title("Cluster Profile Heatmap")
    plt.tight_layout()
    plt.savefig(output_dir / "cluster_profile_heatmap.png", dpi=150)  # Save heatmap for comparing cluster profiles visually
    plt.close()


def plot_pca_clusters(
    x_scaled: np.ndarray, clustered_df: pd.DataFrame, output_dir: Path
) -> pd.DataFrame:
    pca = PCA(n_components=2, random_state=RANDOM_STATE)  # Reduce scaled features to PC1 and PC2 for visualization only
    coords = pca.fit_transform(x_scaled)

    pca_df = pd.DataFrame(coords, columns=["PC1", "PC2"])  # PC1 and PC2 are combinations of scaled features, not original columns
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
    plt.savefig(output_dir / "clustering_pca.png", dpi=150)  # Save 2D visualization of customer segments
    plt.close()

    return pca_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Final K-Means clustering workflow")  # Allow input and output paths from PowerShell
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

    output_dir = Path(args.output)  # Output folder for csv and png results
    output_dir.mkdir(parents=True, exist_ok=True)  # Create the output folder if it does not exist

    df = load_data(args.input)  # Load the marketing campaign dataset
    df = add_common_features(df)  # Create project-level engineered features
    df = clean_common_data(df)  # Remove clear outliers and invalid date rows

    clustered_df, k_scores, x_scaled, _ = run_clustering(df, final_k=FINAL_K)  # Run final k=4 K-Means clustering
    profile = create_cluster_profile(clustered_df)  # Build cluster summary table and persona labels

    k_scores.to_csv(output_dir / "clustering_k_scores.csv", index=False, encoding="utf-8-sig")  # Save k-selection metrics
    profile.to_csv(output_dir / "clustering_profile.csv", index=False, encoding="utf-8-sig")  # Save cluster interpretation table
    clustered_df.to_csv(
        output_dir / "marketing_campaign_with_clusters.csv",
        index=False,
        encoding="utf-8-sig",
    )  # Save original data with assigned cluster labels

    plot_silhouette_scores(k_scores, output_dir)  # Save silhouette score plot
    plot_elbow(k_scores, output_dir)  # Save elbow plot using inertia
    plot_response_rate(profile, output_dir)  # Save response rate comparison plot
    plot_cluster_heatmap(profile, output_dir)  # Save profile heatmap
    plot_pca_clusters(x_scaled, clustered_df, output_dir)  # Save PCA cluster plot

    print("\nClustering k scores:")
    print(k_scores.round(4).to_string(index=False))
    print("\nCluster profile:")
    print(profile.to_string(index=False))
    print(f"\nSaved output files to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
