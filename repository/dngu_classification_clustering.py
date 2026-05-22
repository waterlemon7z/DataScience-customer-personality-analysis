import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    make_scorer,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier


DATA_FILE = "marketing_campaign.xlsx"
KAGGLE_DATASET = "imakash3011/customer-personality-analysis"
RANDOM_STATE = 42
SELECTED_K = 4
REFERENCE_YEAR = 2014


def load_data(file_path=DATA_FILE):
    # Load data
    file_path = Path(file_path)

    if file_path.exists():
        if file_path.suffix == ".xlsx":
            df = pd.read_excel(file_path)
        else:
            df = pd.read_csv(file_path, sep="\t")
    else:
        import kagglehub

        path = kagglehub.dataset_download(KAGGLE_DATASET)
        csv_path = Path(path) / "marketing_campaign.csv"
        df = pd.read_csv(csv_path, sep="\t")

    print("\nDataset shape:")
    print(df.shape)

    print("\nMissing values:")
    missing_values = df.isnull().sum()
    print(missing_values[missing_values > 0])

    print("\nTarget distribution:")
    print(df["Response"].value_counts())
    print(df["Response"].value_counts(normalize=True).round(4))

    return df


def add_common_features(df):
    # Date feature creation
    df = df.copy()
    df["Dt_Customer"] = pd.to_datetime(df["Dt_Customer"], format="%d-%m-%Y")    # Date conversion
    df["CustomerTenure"] = (df["Dt_Customer"].max() - df["Dt_Customer"]).dt.days    # Customer tenure

    # Feature engineering
    df["Age"] = REFERENCE_YEAR - df["Year_Birth"]    # Age calculation
    df["TotalChildren"] = df["Kidhome"] + df["Teenhome"]    # Children total

    df["TotalSpending"] = (    # Spending total
        df["MntWines"]
        + df["MntFruits"]
        + df["MntMeatProducts"]
        + df["MntFishProducts"]
        + df["MntSweetProducts"]
        + df["MntGoldProds"]
    )

    df["TotalPurchases"] = (    # Purchase total
        df["NumDealsPurchases"]
        + df["NumWebPurchases"]
        + df["NumCatalogPurchases"]
        + df["NumStorePurchases"]
    )

    df["CampaignAcceptedTotal"] = (    # Campaign total
        df["AcceptedCmp1"]
        + df["AcceptedCmp2"]
        + df["AcceptedCmp3"]
        + df["AcceptedCmp4"]
        + df["AcceptedCmp5"]
    )

    return df


def clean_common_data(df):
    # Outlier check
    before_rows = len(df)
    age_outliers = (df["Age"] > 100).sum()
    income_outliers = (df["Income"] > 200000).sum()

    # Outlier removal
    df = df[df["Age"] <= 100]
    df = df[(df["Income"].isnull()) | (df["Income"] <= 200000)]
    after_rows = len(df)

    print("\nOutlier handling:")
    print(f"Age > 100 row: {age_outliers}")
    print(f"Income > 200000 row: {income_outliers}")
    print(f"Rows after cleaning: {after_rows} / {before_rows}")

    # Rare category grouping
    rare_marital_status = ["Alone", "Absurd", "YOLO"]
    df["Marital_Status"] = df["Marital_Status"].replace(rare_marital_status, "Other")

    return df


def prepare_classification_data(df):
    # Classification preprocessing
    drop_columns = [
        "ID",
        "Dt_Customer",
        "Z_CostContact",
        "Z_Revenue",
        "Response",
    ]

    X = df.drop(columns=drop_columns)
    y = df["Response"]

    # Feature type split
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

    print("\nClassification feature:")
    print(f"Numerical: {len(numeric_features)}, Categorical: {len(categorical_features)}")
    print("Feature count is after feature engineering and column dropping.")

    return X, y, numeric_features, categorical_features


def build_classification_preprocessor(numeric_features, categorical_features):
    # Numeric pipeline
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    # Category pipeline
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    # Full preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    return preprocessor


def get_classification_models():
    # Model list
    return {
        "Baseline Dummy Classifier": DummyClassifier(strategy="most_frequent"),
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "K-Nearest Neighbors (KNN)": KNeighborsClassifier(n_neighbors=5),
        "Decision Tree": DecisionTreeClassifier(
            random_state=RANDOM_STATE,
            class_weight="balanced",
        ),
    }


def run_cross_validation(X_train, y_train, preprocessor, models):
    # K-fold setting
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    # Evaluation metric
    scoring = {
        "accuracy": "accuracy",
        "precision": make_scorer(precision_score, zero_division=0),
        "recall": make_scorer(recall_score, zero_division=0),
    }

    cv_results_list = []

    for model_name, model in models.items():
        # Model pipeline
        classifier = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ])

        # Cross validation
        cv_results = cross_validate(
            classifier,
            X_train,
            y_train,
            cv=cv,
            scoring=scoring,
        )

        # CV result append
        cv_results_list.append({
            "Model": model_name,
            "CV_Accuracy_Mean": cv_results["test_accuracy"].mean(),
            "CV_Precision_Mean": cv_results["test_precision"].mean(),
            "CV_Recall_Mean": cv_results["test_recall"].mean(),
        })

    cv_results_df = pd.DataFrame(cv_results_list)
    cv_results_df.to_csv("classification_cv_results.csv", index=False, encoding="utf-8-sig")

    print("\nCross validation results:")
    print(cv_results_df)

    return cv_results_df


def evaluate_test_set(X_train, X_test, y_train, y_test, preprocessor, models):
    # Test evaluation
    test_results_list = []
    reports = {}

    for model_name, model in models.items():
        # Model pipeline
        classifier = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ])

        # Final training
        classifier.fit(X_train, y_train)

        # Prediction
        y_pred = classifier.predict(X_test)

        # Metric calculation
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
        recall = recall_score(y_test, y_pred, pos_label=1, zero_division=0)

        # Test result append
        test_results_list.append({
            "Model": model_name,
            "Test_Accuracy": accuracy,
            "Test_Precision_Response_1": precision,
            "Test_Recall_Response_1": recall,
        })

        # Report output
        cm = confusion_matrix(y_test, y_pred)
        report = pd.DataFrame({
            "Class": [0, 1],
            "Precision": precision_score(
                y_test,
                y_pred,
                labels=[0, 1],
                average=None,
                zero_division=0,
            ),
            "Recall": recall_score(
                y_test,
                y_pred,
                labels=[0, 1],
                average=None,
                zero_division=0,
            ),
        })
        reports[model_name] = report

        # Confusion matrix image
        display = ConfusionMatrixDisplay(confusion_matrix=cm)
        display.plot()
        plt.title(f"Confusion Matrix - {model_name}")

        file_name = (
            "confusion_matrix_"
            + model_name.lower()
            .replace(" ", "_")
            .replace("-", "")
            .replace("=", "")
            + ".png"
        )

        plt.savefig(file_name, bbox_inches="tight")
        plt.close()

    # Test result save
    test_results_df = pd.DataFrame(test_results_list)
    test_results_df.to_csv("classification_test_results.csv", index=False, encoding="utf-8-sig")

    print("\nFinal test results:")
    print(test_results_df)

    return test_results_df, reports


def summarize_classification_results(test_results_df):
    # Best model
    best_accuracy_model = test_results_df.loc[test_results_df["Test_Accuracy"].idxmax()]
    best_recall_model = test_results_df.loc[test_results_df["Test_Recall_Response_1"].idxmax()]

    # Summary dict
    summary = {
        "best_accuracy_model": best_accuracy_model["Model"],
        "best_accuracy": best_accuracy_model["Test_Accuracy"],
        "best_recall_model": best_recall_model["Model"],
        "best_recall": best_recall_model["Test_Recall_Response_1"],
    }

    print("\nClassification interpretation:")
    print(
        f"The model with the highest test accuracy is {summary['best_accuracy_model']} "
        f"with accuracy {summary['best_accuracy']:.4f}."
    )
    print(
        f"The model with the highest recall for Response=1 is {summary['best_recall_model']} "
        f"with recall {summary['best_recall']:.4f}."
    )
    return summary


def run_classification(df):
    # Classification full process
    X, y, numeric_features, categorical_features = prepare_classification_data(df)
    preprocessor = build_classification_preprocessor(numeric_features, categorical_features)
    models = get_classification_models()

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # CV and test
    cv_results = run_cross_validation(X_train, y_train, preprocessor, models)
    test_results, reports = evaluate_test_set(
        X_train,
        X_test,
        y_train,
        y_test,
        preprocessor,
        models,
    )
    summary = summarize_classification_results(test_results)

    # Result return
    return {
        "X_shape": X.shape,
        "target_ratio": y.value_counts(normalize=True),
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "cv_results": cv_results,
        "test_results": test_results,
        "reports": reports,
        "summary": summary,
    }


def prepare_clustering_data(df):
    # Proposal-based clustering features
    cluster_features = [
        "Income",
        "Recency",
        "CustomerTenure",
        "Age",
        "TotalChildren",
        "TotalSpending",
        "TotalPurchases",
        "CampaignAcceptedTotal",
        "NumWebVisitsMonth",
        "NumDealsPurchases",
        "NumWebPurchases",
        "NumCatalogPurchases",
        "NumStorePurchases",
    ]

    X_cluster = df[cluster_features]

    # Missing value
    imputer = SimpleImputer(strategy="median")
    X_imputed = imputer.fit_transform(X_cluster)

    # Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    print("\nClustering feature:")
    print(f"{len(cluster_features)} features used")

    return X_cluster, X_scaled, cluster_features


def fit_final_clustering(df, X_scaled, cluster_features, selected_k=SELECTED_K):
    # Final model
    kmeans = KMeans(n_clusters=selected_k, random_state=RANDOM_STATE, n_init=10)
    df = df.copy()
    df["Cluster"] = kmeans.fit_predict(X_scaled)

    print(f"\n K: {selected_k}")
    print("K is set for customer segment interpretation based on the proposal.")

    print("\nCluster counts:")
    print(df["Cluster"].value_counts().sort_index())

    # Profile column
    profile_columns = [
        "Income",
        "Age",
        "TotalChildren",
        "CustomerTenure",
        "Recency",
        "TotalSpending",
        "TotalPurchases",
        "CampaignAcceptedTotal",
        "NumWebVisitsMonth",
        "Response",
    ]

    # Cluster profile
    cluster_profile = df.groupby("Cluster")[profile_columns].mean().round(2)
    cluster_profile["CustomerCount"] = df["Cluster"].value_counts().sort_index()
    cluster_profile = cluster_profile.reset_index()

    # Cluster result save
    cluster_profile.to_csv("clustering_profile.csv", index=False, encoding="utf-8-sig")
    df.to_csv("marketing_campaign_with_clusters.csv", index=False, encoding="utf-8-sig")

    print("\nCluster profile:")
    print(cluster_profile)

    # PCA transform
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    pca_result = pca.fit_transform(X_scaled)

    # PCA dataframe
    pca_df = pd.DataFrame(pca_result, columns=["PC1", "PC2"])
    pca_df["Cluster"] = df["Cluster"].values

    # PCA plot
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        pca_df["PC1"],
        pca_df["PC2"],
        c=pca_df["Cluster"],
        cmap="tab10",
        alpha=0.7,
    )
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("Customer Segments by PCA")
    plt.legend(*scatter.legend_elements(), title="Cluster")
    plt.grid(alpha=0.3)
    plt.savefig("clustering_pca.png", bbox_inches="tight")
    plt.close()

    # Cluster center
    centers = pd.DataFrame(kmeans.cluster_centers_, columns=cluster_features)
    centers.to_csv("clustering_scaled_centers.csv", index=False, encoding="utf-8-sig")

    return df, cluster_profile


def run_clustering(df, selected_k=SELECTED_K):
    # Clustering full process
    X_cluster, X_scaled, cluster_features = prepare_clustering_data(df)

    # Final clustering
    clustered_df, cluster_profile = fit_final_clustering(
        df,
        X_scaled,
        cluster_features,
        selected_k=selected_k,
    )

    # Result return
    return {
        "raw_shape": df.shape,
        "features": X_cluster.columns.tolist(),
        "clustered_df": clustered_df,
        "cluster_profile": cluster_profile,
    }


def run_full_analysis(file_path=DATA_FILE, selected_k=SELECTED_K):
    # Data preparation
    df = load_data(file_path)
    df = add_common_features(df)
    df = clean_common_data(df)

    # Classification part
    print("\nClassification Analysis")

    classification_results = run_classification(df)

    # Clustering part
    print("\nClustering Analysis")

    clustering_results = run_clustering(df, selected_k=selected_k)

    # Final summary
    print("\nFinal Summary")

    print("\nBest classification model:")
    print(classification_results["summary"])

    print("\nGenerated files:")
    print("- classification_cv_results.csv")
    print("- classification_test_results.csv")
    print("- confusion_matrix_*.png")
    print("- clustering_profile.csv")
    print("- marketing_campaign_with_clusters.csv")
    print("- clustering_scaled_centers.csv")
    print("- clustering_pca.png")

    # Result return
    return {
        "classification": classification_results,
        "clustering": clustering_results,
    }


if __name__ == "__main__":
    run_full_analysis()
