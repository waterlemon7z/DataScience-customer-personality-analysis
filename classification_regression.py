import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, StratifiedKFold, KFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    mean_squared_error,
    mean_absolute_error,
    r2_score
)


DATA_PATH = "data/marketing_campaign.csv"


def load_dataset(path=DATA_PATH):
    """
    Load Customer Personality Analysis dataset.
    Kaggle original file is a CSV-like text file.
    """
    df = pd.read_csv(path, sep="\t")
    return df


def create_features(df):
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
    df["Age"] = 2026 - df["Year_Birth"]

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


def clean_data(df):
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


def build_preprocessor(X):
    """
    Build preprocessing pipeline:
    - Numeric columns: median imputation + StandardScaler
    - Categorical columns: most frequent imputation + OneHotEncoder
    """
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_cols = X.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns.tolist()

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )

    return preprocessor


def run_classification(df):
    """
    Classification task:
    Predict Response.
    """
    print("\n==============================")
    print("Classification: Predict Response")
    print("==============================")

    y = df["Response"]

    # Do not include regression target as a feature.
    X = df.drop(columns=["Response", "TotalSpending"])

    preprocessor = build_preprocessor(X)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=5),
    }

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for model_name, model in models.items():
        print(f"\n--- {model_name} ---")

        clf = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", model),
            ]
        )

        cv_scores = cross_val_score(clf, X, y, cv=cv, scoring="accuracy")
        print("5-fold CV Accuracy:", cv_scores)
        print("Mean CV Accuracy:", cv_scores.mean())

        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        print("Test Accuracy:", accuracy_score(y_test, y_pred))
        print("Precision:", precision_score(y_test, y_pred, zero_division=0))
        print("Recall:", recall_score(y_test, y_pred, zero_division=0))
        print("F1-score:", f1_score(y_test, y_pred, zero_division=0))
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        print("Classification Report:")
        print(classification_report(y_test, y_pred, zero_division=0))


def run_regression(df):
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

    preprocessor = build_preprocessor(X)

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


def main():
    df = load_dataset()
    print("Original shape:", df.shape)

    df = create_features(df)
    df = clean_data(df)

    print("Cleaned shape:", df.shape)

    print("\nMissing values after cleaning:")
    missing_values = df.isnull().sum()
    print(missing_values[missing_values > 0])

    print("\nTarget distribution:")
    print(df["Response"].value_counts())
    print(df["Response"].value_counts(normalize=True))

    run_classification(df)
    run_regression(df)


if __name__ == "__main__":
    main()