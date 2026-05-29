# Product Preference Regression

This workflow predicts each customer's expected purchase share for six product categories and converts the predictions into a personalized banner advertisement order.

## Business models

| Business model | Target customer | Available features |
|---|---|---|
| `new_user` | New customers with no purchase history | `Age`, `Income`, `Education`, `Marital_Status`, `Kidhome`, `Teenhome` |
| `early_behavior` | New customers with early web activity | New-user features + `NumWebVisitsMonth` |
| `existing_customer` | Existing customers with purchase-channel history | New-user features + `Recency`, `NumDealsPurchases`, `NumWebPurchases`, `NumCatalogPurchases`, `NumStorePurchases` |

## Target

Six regression targets are created from product spending shares:

- `WineShare`
- `FruitShare`
- `MeatShare`
- `FishShare`
- `SweetShare`
- `GoldShare`

## Leakage rule

The product spending columns (`MntWines` to `MntGoldProds`) and `TotalSpending` are used only to create the target shares. They are not used as input features.

## Recommendation rule

The model predicts product shares for each customer. To avoid recommending categories that are generally large for everyone, each predicted share is divided by the average product share from the training set. The products are then sorted by this relative preference score and used as the personalized banner order.

## Run

```bash
python regression/product_preference_regression.py --input ../marketing_campaign.xlsx --output regression/outputs
```

## Outputs

- `regression_product_share_metrics.csv`
- `regression_business_model_summary.csv`
- `recommendation_sample_<business_model>.csv`
- `train_average_product_shares_<business_model>.csv`
- `top1_banner_distribution_<business_model>.csv`
- `regression_rmse_by_target.png`
- `top1_banner_distribution.png`

