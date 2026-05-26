import kagglehub
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LinearRegression

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

class IncomeRegressor:
    def __init__(self):
        self.raw_df = self.__load_dataset_from_kaggle()


    def __load_dataset_from_kaggle(self):
        """
        Load Customer Personality Analysis dataset.
        Kaggle original file is a CSV-like text file.
        """
        path = kagglehub.dataset_download("imakash3011/customer-personality-analysis")
        df = pd.read_csv(path + '/marketing_campaign.csv', sep='\t')
        return df.copy()
    def preprocessing(self, ):
        temp_df = self.create_features(self.raw_df)
        self.df = self.clean_data(temp_df)

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

    def data_inspection(self, df= None):
        if df is None:
            df = self.raw_df
        corr_matrix = df.corr(method='pearson', numeric_only=True)
        print("--- cov matrix ---")
        print(corr_matrix)

        plt.figure(figsize=(20, 20))

        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', vmin=-1, vmax=1, linewidths=0.5)
        plt.title('Correlation Matrix Heatmap', fontsize=15)
        plt.show()

    def build_preprocessor(self, X):
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
        plt.show()

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
        plt.show()


def main():
    reg = IncomeRegressor()
    reg.preprocessing()

    print("Original shape:", reg.raw_df.shape)
    print("Preprocessed shape:", reg.df.shape)

    print("\nMissing values after preprocessing:")
    missing_values = reg.df.isnull().sum()
    print(missing_values[missing_values > 0])

    reg.data_inspection(reg.df)
    reg.run_regression(reg.df)


if __name__ == "__main__":
    main()
