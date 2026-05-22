# Classification + Clustering

This folder contains the classification and clustering code for the Customer Personality Analysis term project.

## Goal

- Classification: predict the binary target variable `Response`.
- Clustering: segment customers into interpretable marketing groups.

## How to run

Install packages:

```bash
pip install -r requirements.txt
```

Run the analysis:

```bash
python classification_clustering.py --input marketing_campaign.xlsx
```

If the dataset is in another folder:

```bash
python classification_clustering.py --input "C:/path/to/marketing_campaign.xlsx" --output outputs
```

## Main preprocessing steps

- Convert `Dt_Customer` to datetime.
- Create derived variables:
  - `Age`
  - `Customer_Days`
  - `TotalSpending`
  - `TotalPurchases`
  - `Children`
  - `AcceptedCmpTotal`
- Group rare `Marital_Status` values into `Other`.
- Remove unrealistic age outliers and extreme income outliers.
- Impute missing numeric values with median.
- Impute missing categorical values with most frequent value.
- One-hot encode categorical variables.
- Scale numeric variables.

## Classification models

- Logistic Regression
- Decision Tree
- K-Nearest Neighbors

The target variable `Response` is imbalanced, so the code reports:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion matrix
- 5-fold cross-validation metrics

## Clustering model

- K-means clustering
- Tests `k = 2` to `k = 7`
- Uses silhouette score and inertia
- Saves cluster profiles for interpretation

## Output files

The script saves result files in the `outputs` folder:

- `classification_results.csv`
- `confusion_matrix_logistic_regression.csv`
- `confusion_matrix_decision_tree.csv`
- `confusion_matrix_knn.csv`
- `clustering_k_scores.csv`
- `cluster_profile.csv`
- `clustered_customers.csv`
