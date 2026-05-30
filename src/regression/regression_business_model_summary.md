# Regression - Business Model Summary
고객 특성 기반 제품군별 광고 노출 순위 추천 알고리즘   
과거 구매 패턴을 바탕으로 고객별 반응 가능성이 높은 제품군 광고 순서를 추천한다
## 1. Business Objective

This regression model is used to decide the display order of product-category banner ads for customers.

The main business question is:

> Given customer information, which product-category ads should be shown first?

The model supports three customer situations.

| Business Model    | Target Customer | Available Features |
|-------------------|---|---|
| New-user          | New customers with no purchase history | Age, Income, Education, Marital_Status, Kidhome, Teenhome |
| Early-behavior    | New customers with early web activity | New-user features + NumWebVisitsMonth |
| Existing-customer | Existing customers with purchase-channel history | New-user features + Recency, NumWebPurchases, NumCatalogPurchases, NumStorePurchases |

The final output is not just one recommended product category. Instead, the model produces a ranked order of all six product categories for banner display.

---

## 2. Product Categories

The target product categories are the six `Mnt...` columns from the Customer Personality Analysis dataset.

```python
product_cols = [
    "MntWines",
    "MntFruits",
    "MntMeatProducts",
    "MntFishProducts",
    "MntSweetProducts",
    "MntGoldProds",
]
```

These columns are not used as input features because that would cause target leakage. They are used only to create the regression target.

---

## 3. Regression Target

The model predicts each customer's product-category spending share.

For each customer:

```text
ProductShare = ProductSpending / TotalSpending
```

Example:

| Product | Spending | Spending Share |
|---|---:|---:|
| MntWines | 500 | 0.50 |
| MntFruits | 50 | 0.05 |
| MntMeatProducts | 300 | 0.30 |
| MntFishProducts | 100 | 0.10 |
| MntSweetProducts | 50 | 0.05 |
| MntGoldProds | 0 | 0.00 |

So the regression target `y` is a multi-output target with six values per customer.

```python
total_spending = df[product_cols].sum(axis=1)

y = df.loc[valid_target_index, product_cols].div(
    total_spending.loc[valid_target_index],
    axis=0,
)
```

This means the model predicts product preference as a spending-share vector.

이것만 적용 시 문제되는 것은 평균적으로 비싼 가격의 제품 카테고리(와인)는 가끔씩 구매해도 비중이 올라가게 된다. (아래 4번에서 이어 설명한다.)

---

## 4. Why Raw Product Share Is Not Enough

If the model simply recommends the product category with the highest predicted share, expensive or generally popular categories such as `MntWines` can dominate the recommendation.

That creates a weak business result:

> The model may behave similarly to a simple baseline that recommends Wine to almost everyone.

To avoid this, the model does not use raw predicted share directly as the final recommendation score.

---

## 5. Recommendation Score

The final recommendation score is calculated by comparing the predicted product share with the train-set average product share.

```text
RecommendationScore = PredictedProductShare / TrainAverageProductShare
```

In code:

```python
product_prior = y_train.mean().replace(0, np.nan)
recommendation_score_df = y_pred_df.div(product_prior, axis=1)
```

This changes the meaning of the recommendation.

Raw predicted share asks:

> Which category will this customer spend the most on?

Relative recommendation score asks:

> Which category is this customer expected to prefer more strongly than the average customer?

This reduces the dominance of generally popular or high-spending categories and makes the recommendation more personalized.

---

## 6. Banner Ad Display Order

For each customer, the six product categories are sorted by `RecommendationScore` in descending order.

```python
result_df["Banner_Ad_Order"] = recommendation_score_df.apply(
    lambda row: " > ".join(row.sort_values(ascending=False).index),
    axis=1,
)
```

Example output:

```text
Sample customer index: 1755
Banner ad display order:
1. MntGoldProds
2. MntFruits
3. MntFishProducts
4. MntSweetProducts
5. MntMeatProducts
6. MntWines
```

This does not mean the customer is expected to spend the most money on `MntGoldProds`. It means `MntGoldProds` has the strongest predicted preference relative to the average customer.

Therefore, the banner display strategy is:

```text
Show the highest relative-preference product category first.
Then show the remaining categories in descending RecommendationScore order.
```

---

## 7. Evaluation

The model is evaluated in two ways.

### Regression Metrics

These measure how close the predicted product-share vector is to the actual product-share vector.

```python
r2_score(y_test, y_pred)
mean_absolute_error(y_test, y_pred)
np.sqrt(mean_squared_error(y_test, y_pred))
```

Metrics used:

- Test R2
- Test MAE
- Test RMSE
- 5-fold CV R2

### Recommendation Match Rate

The recommendation target is also converted to relative preference using the same train-set product prior.

```text
ActualRelativeScore = ActualProductShare / TrainAverageProductShare
PredictedRelativeScore = PredictedProductShare / TrainAverageProductShare
```

Then the model checks whether the highest predicted relative-preference category matches the highest actual relative-preference category.

```python
actual_product = actual_relative_score.idxmax(axis=1)
recommended_product = recommendation_score_df.idxmax(axis=1)
recommendation_match_rate = (actual_product == recommended_product).mean()
```

---

## 8. Baseline

The baseline is a non-personalized recommendation rule.

It always recommends the most common actual relative-preference category from the train set.

```python
baseline_product = y_train_relative_score.idxmax(axis=1).mode()[0]
baseline_match_rate = (actual_relative_product == baseline_product).mean()
```

The model is useful only if it performs better than this baseline.

In the current result, the baseline is about `0.299`, while the regression models reach around `0.37` to `0.40`. This means the model provides a measurable improvement over a simple non-personalized recommendation.

---

## 9. Final Interpretation

The regression model predicts product-category spending shares from customer information. Then it converts those predicted shares into relative preference scores by dividing by the average product share in the training data.

The final business output is a ranked list of product categories for banner advertising.

A concise project description is:

> This model predicts each customer's product-category preference using regression and ranks the six product categories by relative preference score. The resulting order is used as the personalized banner-ad display order for cold-start, early-behavior, and existing-customer scenarios.

---

## 10. Important Limitation

The dataset does not contain actual ad impressions, clicks, or conversion labels. Therefore, this model does not directly predict ad click-through rate or campaign response.

Instead, it uses historical purchase behavior as a proxy for product-category advertising preference.

This should be stated clearly in the report or presentation.
