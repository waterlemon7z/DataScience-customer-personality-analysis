# Customer Personality Analysis

This project analyzes the Customer Personality Analysis dataset for marketing decision support.

## Objective

The main objective is to predict customer campaign response and segment customers into meaningful groups.

- Classification: predict `Response`
- Clustering: create customer segments
- Business goal: support targeted marketing strategy

## Dataset

The dataset contains customer demographic information, purchase behavior, campaign acceptance history, and response result.

Main data issues:

- Missing values in `Income`
- Possible outliers in `Age` and `Income`
- Numerical and categorical features

## Preprocessing

The following preprocessing steps are applied:

- Convert `Dt_Customer` to datetime
- Create `Age`
- Create `CustomerTenure`
- Create `TotalChildren`
- Create `TotalSpending`
- Create `TotalPurchases`
- Create `CampaignAcceptedTotal`
- Remove outliers
- Handle missing values
- Apply encoding and scaling for classification
- Apply scaling for clustering

## Classification

The target variable is `Response`.

Models used:

- Baseline Dummy Classifier
- Logistic Regression
- K-Nearest Neighbors
- Decision Tree

Evaluation metrics:

- Accuracy
- Precision
- Recall
- Confusion Matrix

K-fold cross validation is used for classification model comparison.

## Clustering

K-means clustering is used to group customers based on customer profile and purchase behavior.

Clustering features include:

- Income
- Age
- TotalChildren
- TotalSpending
- TotalPurchases
- CampaignAcceptedTotal
- Purchase channel variables

Cluster results are interpreted using:

- Cluster size
- Average income
- Average spending
- Average response rate
- Customer profile

## Output Files

The code generates:

- `classification_cv_results.csv`
- `classification_test_results.csv`
- `confusion_matrix_*.png`
- `clustering_profile.csv`
- `marketing_campaign_with_clusters.csv`
- `clustering_scaled_centers.csv`
- `clustering_pca.png`