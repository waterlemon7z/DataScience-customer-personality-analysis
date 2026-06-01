# Customer Personality Analysis - DS Team 9

This repository contains the source code, reusable workflows, and output artifacts for a data science term project based on the Kaggle `Customer Personality Analysis` dataset. The project supports targeted marketing decisions through three connected modeling tracks: campaign response classification, customer clustering, and product-category banner recommendation with regression.

## Project Scope

The project focuses on three business tasks:

- Classification: predict whether a customer will accept the latest marketing campaign using the binary target `Response`.
- Clustering: divide customers into interpretable marketing segments and compare each segment's response rate.
- Regression: rank product-category banner ads for personalized targeting based on customer profile and behavior features.

The business objective is to move beyond sending the same campaign to every customer. The analysis helps identify high-potential customers, understand customer groups with different value and campaign sensitivity, and rank product-category ads by predicted relative preference.

## Dataset

- Dataset name: Customer Personality Analysis
- Source: Kaggle, `imakash3011/customer-personality-analysis`
- Kaggle URL: <https://www.kaggle.com/datasets/imakash3011/customer-personality-analysis/data>
- Local file names used in the project: `marketing_campaign.xlsx`, `marketing_campaign.csv`

Dataset summary:

| Item | Value |
|---|---:|
| Raw size | 2,240 rows, 29 columns |
| Cleaned size | 2,236 rows |
| Removed outlier rows | 4 |
| Missing `Income` values | 24 |
| `Response=0` after cleaning | 1,902 customers, 85.06% |
| `Response=1` after cleaning | 334 customers, 14.94% |

`Income` missing values are handled by median imputation inside preprocessing pipelines.

Main dataset fields:

- Demographics: `Year_Birth`, `Education`, `Marital_Status`, `Income`
- Family information: `Kidhome`, `Teenhome`
- Product spending: `MntWines`, `MntFruits`, `MntMeatProducts`, `MntFishProducts`, `MntSweetProducts`, `MntGoldProds`
- Purchase channels: `NumDealsPurchases`, `NumWebPurchases`, `NumCatalogPurchases`, `NumStorePurchases`, `NumWebVisitsMonth`
- Campaign response: `AcceptedCmp1` to `AcceptedCmp5`, `Response`

This dataset is useful for the project because it includes numeric variables, categorical variables, missing values, outliers, purchase behavior, and a clear binary target for classification. It also supports unsupervised customer segmentation.

## Repository Structure

```text
.
|-- main.py
|-- pyproject.toml
|-- requirements.txt
|-- README.md
|-- output/
|   |-- classification/
|   |-- clustering/
|   `-- regression/
`-- src/
    |-- classification/
    |   |-- response_classification.py
    |   `-- README_classification.md
    |-- clustering/
    |   `-- final_clustering.py
    |-- regression/
    |   |-- marketingItemRecommendationRegression.py
    |   |-- regression_business_model_summary.md
    |   `-- Regression_README.md
    `-- util/
        `-- api.py
```

## Workflow Architecture

The source code is organized as reusable workflows instead of one-time notebook cells. Each modeling workflow follows the same general pattern: load data, create features, clean data, preprocess features, fit models, evaluate results, and save outputs.

| Layer | Code component | Role |
|---|---|---|
| Input layer | `src.util.api.load_dataset_from_kaggle()` | Loads the Kaggle dataset used by classification and regression modules. |
| Feature engineering layer | `add_project_features`, `add_common_features`, `prepare_dataframe` | Creates shared customer-level features. |
| Cleaning layer | `clean_dataset`, `clean_common_data` | Handles rare categories and removes extreme outlier rows. |
| Preprocessing layer | `build_preprocessor`, `preprocess_clustering_features`, `build_stage_preprocessor` | Applies imputation, encoding, scaling, and selected log transforms. |
| Modeling layer | `evaluate_models`, `run_clustering`, `run_pipeline` | Trains classification, clustering, and regression/recommendation models. |
| Evaluation layer | `cross_validate`, `calculate_k_scores`, `create_cluster_profile` | Calculates model metrics and interpretation tables. |
| Output layer | `to_csv`, JSON reports, `savefig` | Saves metrics, reports, plots, samples, and cluster assignments. |

## Shared Preprocessing and Feature Engineering

Preprocessing is not identical across all three modeling tracks.

Classification and clustering share these customer-level preprocessing choices:

- Converted `Dt_Customer` to a datetime value.
- Created `CustomerTenure` as the number of days from customer enrollment to the latest enrollment date in the dataset.
- Created `Age` from `Year_Birth`.
- Created `TotalChildren` as `Kidhome + Teenhome`.
- Created `TotalSpending` from the six product spending columns.
- Created `TotalPurchases` from deal, web, catalog, and store purchase counts.
- Created `CampaignAcceptedTotal` only from `AcceptedCmp1` through `AcceptedCmp5`.
- Excluded `Response` from `CampaignAcceptedTotal` to prevent target leakage.
- Grouped rare `Marital_Status` values `Alone`, `Absurd`, and `YOLO` into `Other`.
- Removed rows outside the adult-age range and removed extreme income values.

The regression workflow uses a stage-specific preprocessing path:

- Created `Age` using the dataset reference year.
- Created `TotalChildren` as `Kidhome + Teenhome`.
- Grouped rare `Marital_Status` values `Alone`, `Absurd`, and `YOLO` into `Other`.
- Removed rows outside the adult-age range and removed extreme income values.
- Applied median imputation, selected log transforms, scaling, and one-hot encoding inside scikit-learn pipelines.
- Did not use `Dt_Customer`, `CustomerTenure`, `TotalPurchases`, or `CampaignAcceptedTotal` in the current business feature sets.

## 1. Response Classification

File: `src/classification/response_classification.py`

This module predicts `Response`, a binary target indicating whether a customer accepted the latest campaign. Because the target is imbalanced, accuracy alone is misleading. The evaluation emphasizes precision, recall, F1, ROC-AUC, and confusion matrices.

Classification preprocessing:

- Numeric variables: median imputation and `StandardScaler`
- Categorical variables: most-frequent imputation and `OneHotEncoder`
- The preprocessing is inside a scikit-learn `Pipeline`, so imputation, encoding, and scaling are fit only on training data in each split.

Classification setup:

| Item | Value |
|---|---|
| Target variable | `Response` |
| Final feature count | 29 |
| Train/test split | 80/20 with stratification |
| Cross validation | `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` |

Compared models:

| Model | Key setting |
|---|---|
| `DummyClassifier` | `strategy="most_frequent"` |
| `LogisticRegression` | default class weight |
| `LogisticRegression` | `class_weight="balanced"` |
| `KNeighborsClassifier` | `n_neighbors=15` |
| `DecisionTreeClassifier` | `max_depth=5`, `min_samples_leaf=20`, `class_weight="balanced"` |

Current classification results:

| Model | Test Acc. | Test Prec. | Test Recall | Test F1 | Test ROC-AUC | CV F1 | CV ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| logistic_regression_balanced | 0.8192 | 0.4397 | 0.7612 | 0.5574 | 0.8917 | 0.5651 | 0.8969 |
| logistic_regression | 0.8862 | 0.6905 | 0.4328 | 0.5321 | 0.8871 | 0.5569 | 0.8953 |
| decision_tree_balanced | 0.7768 | 0.3540 | 0.5970 | 0.4444 | 0.7756 | 0.4803 | 0.8059 |
| knn_k15 | 0.8750 | 0.7037 | 0.2836 | 0.4043 | 0.8346 | 0.3948 | 0.8310 |
| dummy_most_frequent | 0.8504 | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0.0000 | 0.5000 |

Interpretation:

- `logistic_regression_balanced` is the most appropriate model for the business goal because it has the strongest balance of F1 and recall.
- The balanced logistic regression model identifies more positive-response customers than the unbalanced alternatives.
- The dummy baseline has high accuracy because the data is imbalanced, but it predicts no positive-response customers.

Main classification outputs:

- `output/classification/classification_model_results.csv`
- `output/classification/classification_data_summary.json`
- `output/classification/confusion_matrix_<model>.csv`
- `output/classification/confusion_matrix_<model>.png`
- `output/classification/classification_report_<model>.json`
- `output/classification/prediction_sample_<model>.csv`

## 2. Customer Clustering

File: `src/clustering/final_clustering.py`

This module segments customers with K-Means clustering. `Response` is excluded from the clustering input and used only after clustering to interpret segment-level response rates.

Clustering features:

- `Income`
- `Age`
- `Recency`
- `TotalChildren`
- `TotalSpending`
- `TotalPurchases`
- `CampaignAcceptedTotal`
- `NumWebVisitsMonth`
- `CustomerTenure`
- `NumDealsPurchases`
- `NumWebPurchases`
- `NumCatalogPurchases`
- `NumStorePurchases`

Clustering preprocessing:

- Median imputation for missing numeric values
- `StandardScaler` before K-Means because K-Means is distance-based

Clustering setup:

| Item | Value |
|---|---|
| Candidate `k` values | 2 to 8 |
| Final model | `KMeans(n_clusters=4, random_state=42, n_init=10)` |

K selection summary:

| k | Silhouette score | Inertia |
|---:|---:|---:|
| 2 | 0.2847 | 20,213.95 |
| 3 | 0.2378 | 17,575.72 |
| 4 | 0.2285 | 16,382.29 |
| 5 | 0.1496 | 15,398.25 |
| 6 | 0.1437 | 14,521.81 |
| 7 | 0.1402 | 13,945.24 |
| 8 | 0.1296 | 13,468.32 |

Although the highest silhouette score came from the broadest segmentation, the final model was selected because it provides more useful marketing personas: low-value customers, middle-value customers, high-value responsive customers, and high-value less-responsive customers.

Current cluster profile:

| Cluster | Count | Avg Income | Avg Spending | Avg Purchases | Avg Accepted | Response Rate | Persona |
|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | 993 | 34,269 | 92 | 7.56 | 0.08 | 0.085 | Low-income, low-spending, less-responsive customers |
| 1 | 451 | 53,036 | 666 | 21.03 | 0.25 | 0.164 | Middle-income, middle-spending customers |
| 2 | 223 | 81,401 | 1,644 | 21.60 | 1.84 | 0.507 | High-income, high-spending, responsive customers |
| 3 | 569 | 70,161 | 1,048 | 20.11 | 0.11 | 0.111 | High-income, high-spending, less-responsive customers |

Interpretation:

- The high-income, high-spending responsive segment is the strongest campaign target candidate.
- The low-income, low-spending segment is large but less responsive.
- One high-spending segment has a low final campaign response rate, so high spending alone does not guarantee campaign acceptance.

Main clustering outputs:

- `output/clustering/clustering_k_scores.csv`
- `output/clustering/clustering_profile.csv`
- `output/clustering/marketing_campaign_with_clusters.csv`
- `output/clustering/silhouette_score_plot.png`
- `output/clustering/elbow_plot.png`
- `output/clustering/response_rate_by_cluster.png`
- `output/clustering/cluster_profile_heatmap.png`
- `output/clustering/clustering_pca.png`

## 3. Marketing Item Recommendation Regression

File: `src/regression/marketingItemRecommendationRegression.py`

This module recommends personalized product-category banner display order. It predicts product-category spending-share preferences across six product categories and converts those predicted shares into relative preference scores.

Target product categories:

- `MntWines`
- `MntFruits`
- `MntMeatProducts`
- `MntFishProducts`
- `MntSweetProducts`
- `MntGoldProds`

Regression target:

```text
ProductShare = ProductSpending / TotalSpending
```

The target is a six-output spending-share vector. The model predicts all six product-category shares at the same time instead of training a separate model for each category.

Recommendation score:

```text
RecommendationScore = PredictedProductShare / TrainAverageProductShare
```

This converts the prediction from expected product share to relative preference. The final banner order is created by sorting the six product categories by `RecommendationScore` in descending order. This avoids simply recommending generally popular or high-spending categories to everyone.

Business feature sets:

| Business Model | Scenario | Features |
|---|---|---|
| New-user | New customer with no purchase history | `Age`, `Income`, `Education`, `Marital_Status`, `Kidhome`, `Teenhome` |
| Early-behavior | Customer with early web-visit behavior | New-user features + `NumWebVisitsMonth` |
| Existing-customer | Customer with purchase-channel history | New-user features + `Recency`, `NumWebPurchases`, `NumCatalogPurchases`, `NumStorePurchases` |

Regression preprocessing:

- Median imputation for missing numeric values
- Log transform for skewed variables such as `Income` and count-based behavior features where appropriate
- `StandardScaler` for numeric features
- `OneHotEncoder(handle_unknown="ignore")` for categorical features such as `Education` and `Marital_Status`

Compared models:

| Model | Key setting |
|---|---|
| Multiple Linear Regression | baseline linear model |
| Polynomial Regression | degree 2 |

Evaluation metrics:

- CV R2 Mean / CV R2 Std
- Test R2
- Test MAE
- Test RMSE
- Recommendation Match Rate
- Baseline Match Rate
- Lift vs Baseline

Interpretation:

- The final regression model is selected by `Recommendation Match Rate` because the business goal is recommendation ranking, not only regression accuracy.
- The regression models perform above the non-personalized baseline, showing that customer-profile and behavior features improve personalized banner ranking.
- The recommendation DataFrame includes `Recommended_Product`, `Banner_Ad_Order`, and `Banner_Ad_Scores` for each customer.
- The current script saves diagnostic plots to `output/regression` and prints sample recommendation rows; it does not write the per-customer recommendation table to CSV by default.
- The dataset does not include real ad impressions, clicks, or conversions, so this model does not directly predict click-through rate. Historical purchase behavior is used as a proxy for product-category ad preference.

Main regression outputs:

- `output/regression/relative_preference_match_rate_by_business_model.png`
- `output/regression/regression_r2_by_business_model.png`
- `output/regression/relative_recommendation_lift_vs_baseline.png`
- `output/regression/actual_vs_recommended_relative_product_ratio.png`
- `output/regression/actual_vs_recommended_relative_product_distribution.png`

## Important Libraries and APIs

| Library or API | Project usage |
|---|---|
| `pandas` | Data loading, cleaning, feature engineering, CSV outputs |
| `numpy` | Numeric operations and prediction post-processing |
| `scikit-learn Pipeline` | Connects preprocessing and modeling while reducing data leakage risk |
| `ColumnTransformer` | Applies different preprocessing to numeric and categorical columns |
| `SimpleImputer` | Handles missing values such as missing `Income` |
| `StandardScaler` | Standardizes numeric features for distance-based and regularized models |
| `OneHotEncoder` | Encodes categorical variables such as `Education` and `Marital_Status` |
| `LogisticRegression` | Main campaign response classification model |
| `KNeighborsClassifier` | Classification comparison model |
| `DecisionTreeClassifier` | Tree-based classification comparison model |
| `DummyClassifier` | Majority-class baseline |
| `KMeans` | Customer segmentation model |
| `silhouette_score` and `inertia_` | Clustering k-selection support |
| `PCA` | Two-dimensional clustering visualization |
| `LinearRegression` and `PolynomialFeatures` | Regression-based recommendation models |
| `matplotlib` and `seaborn` | Output plot generation |

## How to Run

Runtime requirement:

| Requirement | Value |
|---|---|
| Python | 3.12 or later |

Install dependencies:

```bash
pip install -r requirements.txt
```

Or, if using `uv`:

```bash
uv sync
```

Run response classification:

```bash
python -m src.classification.response_classification --output output/classification
```

Run customer clustering:

```bash
python -m src.clustering.final_clustering --input path/to/marketing_campaign.csv --output output/clustering
```

Run regression-based recommendation:

```bash
python -m src.regression.marketingItemRecommendationRegression
```

Run the integrated script:

```bash
python main.py
```

Note: `main.py` calls the clustering module with its default input path and output directory. This expects `marketing_campaign.xlsx` in the project root and saves clustering files to `outputs/`, not `output/clustering`. To save clustering files under `output/clustering`, run the clustering module separately and pass `--input` and `--output output/clustering`.

## Difficulties, Solutions, and Learning

- Class imbalance: the positive-response class is a minority, so accuracy alone was misleading. The solution was to use stratified splitting and report precision, recall, F1, ROC-AUC, and confusion matrices.
- Data leakage risk: `CampaignAcceptedTotal` could accidentally include `Response`. The solution was to construct it only from previous campaign columns.
- Clustering k selection: the highest-silhouette option was too broad for marketing personas. The solution was to use silhouette and elbow results as supporting evidence, then justify the final choice by interpretability.
- Outliers and missing values: `Income` had missing values and both `Age` and `Income` had extreme values. The solution was median imputation inside pipelines and removal of clear extreme rows.
- Evaluation alignment: each model needed metrics that matched its business purpose. Classification used F1/recall/ROC-AUC, clustering used profile interpretation and response-rate comparison, and regression recommendation used match rate and lift over baseline.

## Key Takeaways

- Balanced logistic regression is the best classification option in the current results because it identifies more positive-response customers than the unbalanced models.
- The high-income, high-spending responsive segment should be treated as the strongest campaign target candidate.
- High spending does not automatically mean high campaign sensitivity.
- The regression recommender ranks product categories by relative preference, so it can avoid recommending only generally popular categories.

## Limitations

- The dataset does not contain real ad impressions, clicks, or conversions, so the recommendation model does not directly measure ad performance.
- The final cluster count is chosen for marketing interpretability, not purely by the highest silhouette score.
- The classification results are based on the current split and fixed random state; additional validation and tuning would be needed before production use.
- The regression model uses historical purchase behavior as a proxy for product-category ad preference, so the recommendation should be validated with real marketing outcome data before deployment.

## Sources

- Dataset: Customer Personality Analysis, Kaggle, <https://www.kaggle.com/datasets/imakash3011/customer-personality-analysis/data>
- Libraries: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`
- Project repository: <https://github.com/waterlemon7z/DataScience-customer-personality-analysis>
