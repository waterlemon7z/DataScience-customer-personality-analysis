꼬덱스로 썼슴니다!
# Classification Flow

이 문서는 `classification.ipynb`의 분류 과제 흐름을 전처리부터 최종 모델 선택까지 정리한 것이다. 목표 변수는 `Response`이며, 캠페인에 반응할 고객(`Response = 1`)을 예측하는 것이 목적이다.

## 1. Data Loading

- KaggleHub를 통해 `customer-personality-analysis` 데이터를 다운로드한다.
- `marketing_campaign.csv`는 tab-separated 형식이므로 `sep='\t'`로 읽는다.
- 원본 데이터프레임은 `df`로 관리한다.

## 2. Initial Feature Inspection

- 숫자형 컬럼에 대해 Pearson correlation matrix를 계산한다.
- heatmap을 통해 변수 간 상관관계를 확인한다.
- 이 단계는 모델에 넣을 후보 변수를 파악하기 위한 EDA 역할을 한다.

## 3. Feature Engineering

원본 컬럼을 그대로 쓰기보다 모델에 더 의미 있는 파생변수를 만든다.

- `Age`
  - `2014 - Year_Birth`
  - 고객의 나이를 나타낸다.

- `Enrollment_Days`
  - 기준일 `2014-08-01`에서 `Dt_Customer`를 뺀 값
  - 고객이 등록된 지 얼마나 오래되었는지를 나타낸다.

- `Total_Spent`
  - `MntWines`, `MntFruits`, `MntMeatProducts`, `MntFishProducts`, `MntSweetProducts`, `MntGoldProds`의 합
  - 전체 소비 규모를 나타낸다.

- `Total_Acp`
  - `AcceptedCmp1`부터 `AcceptedCmp5`까지의 합
  - 과거 캠페인 수락 횟수를 나타낸다.

각 파생변수는 histogram으로 분포를 확인한다.

## 4. Dirty Data Cleaning

모델링에 불필요하거나 중복되는 컬럼을 제거한다.

- 제거한 상수성/비정보성 컬럼
  - `Z_CostContact`
  - `Z_Revenue`

- `Total_Acp`로 합쳐진 개별 캠페인 컬럼 제거
  - `AcceptedCmp1` ~ `AcceptedCmp5`

- `Total_Spent`로 합쳐진 개별 소비 컬럼 제거
  - `MntWines`
  - `MntFruits`
  - `MntMeatProducts`
  - `MntFishProducts`
  - `MntSweetProducts`
  - `MntGoldProds`

- 결측치는 `dropna()`로 제거한다.

## 5. Modeling Feature Selection

분류 모델에는 인구통계, 가족 구성, 구매 행동, 캠페인 관련 변수를 사용한다.

사용 변수:

```python
model_features = [
    'Age', 'Income', 'Education', 'Marital_Status',
    'Kidhome', 'Teenhome', 'Recency', 'Enrollment_Days',
    'Total_Spent', 'Total_Acp',
    'NumDealsPurchases', 'NumWebPurchases', 'NumCatalogPurchases',
    'NumStorePurchases', 'NumWebVisitsMonth', 'Complain'
]
```

추가 처리:

- `Total_Spent`는 오른쪽으로 치우친 분포를 가지므로 `np.log1p()`를 적용해 `Total_Spent_log`로 변환한다.
- 원본 `Total_Spent`는 제거한다.
- `NumCatalogPurchases > 8`인 행은 outlier로 보고 제거한다.

## 6. Preprocessing

모델 입력값은 `X`, 목표 변수는 `y = Response`로 분리한다.

현재 전처리 전략:

- `Education`
  - 순서가 있는 범주형 변수로 보고 `OrdinalEncoder` 사용
  - 순서: `Basic -> 2n Cycle -> Graduation -> Master -> PhD`

- `Marital_Status`
  - 현재 모델에서는 제외한다.
  - 순서형 변수가 아니기 때문에 숫자형처럼 처리하지 않는다.

- 숫자형 변수
  - `StandardScaler` 적용

전처리는 `ColumnTransformer`로 구성한다.

## 7. Train/Test Split

- `train_test_split` 사용
- `test_size=0.2`
- `random_state=42`
- `stratify=y`

`Response = 1` 비율이 낮은 불균형 데이터이므로 stratified split을 사용해 train/test의 클래스 비율을 유지한다.

## 8. Baseline Model

`DummyClassifier(strategy='most_frequent')`를 baseline으로 사용한다.

이 baseline은 항상 다수 클래스인 `Response = 0`을 예측한다. 따라서 accuracy는 높아 보일 수 있지만, positive class recall은 0이다. 이 과제에서는 baseline accuracy만 보고 모델을 평가하면 안 된다.

## 9. Candidate Models

후보 모델:

- Logistic Regression
- KNN
- Decision Tree
- Random Forest
- Gradient Boosting

공통 구조:

- `Pipeline`
  - `preprocess`
  - `pca`
  - `model`

PCA 처리:

- Logistic Regression, KNN은 PCA 사용 여부를 GridSearch 후보에 포함한다.
- Tree 계열 모델은 PCA가 보통 큰 이득이 없으므로 `passthrough`로 고정한다.

## 10. Model Selection Metric

이 데이터는 `Response = 1`이 소수 클래스이므로 accuracy만으로는 부족하다.

주요 지표:

- `Precision`
  - 1이라고 예측한 고객 중 실제 응답 고객의 비율
  - 캠페인 비용 절감 관점에서 중요하다.

- `Recall`
  - 실제 응답 고객 중 모델이 잡아낸 비율
  - 응답 고객을 놓치지 않는 관점에서 중요하다.

- `F1`
  - precision과 recall을 동일하게 고려한 균형 지표

- `F2`
  - recall을 precision보다 더 중요하게 고려한 지표
  - 이 과제처럼 positive class를 놓치는 것이 부담될 때 적합하다.

- `PR-AUC`
  - 불균형 데이터에서 positive class 구분력을 보는 데 유용하다.

현재 GridSearch는 `F2`를 기준으로 `refit`한다.

## 11. Test Set Evaluation

각 모델의 test set 성능을 비교한다.

최근 비교 결과:

| Model | Accuracy | Precision | Recall | F1 | F2 | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Decision Tree | 0.793 | 0.380 | 0.793 | 0.514 | 0.652 | 0.552 |
| Logistic Regression | 0.803 | 0.389 | 0.759 | 0.515 | 0.638 | 0.680 |
| Random Forest | 0.850 | 0.471 | 0.690 | 0.559 | 0.631 | 0.688 |
| Gradient Boosting | 0.912 | 0.800 | 0.483 | 0.602 | 0.524 | 0.726 |
| KNN | 0.893 | 0.667 | 0.448 | 0.536 | 0.480 | 0.506 |

## 12. Trade-off Interpretation

모델 선택은 목적에 따라 달라진다.

- Recall 최우선
  - `Decision Tree`
  - 실제 응답 고객을 가장 많이 잡지만, 오탐도 많다.

- Precision과 campaign efficiency 중시
  - `Gradient Boosting`
  - 1이라고 예측한 고객의 정확도가 높다.
  - 다만 실제 응답 고객 중 놓치는 비율이 높다.

- 중간 균형
  - `Random Forest`
  - Decision Tree보다 precision이 높고, Gradient Boosting보다 recall이 높다.

## 13. Final Model Choice

최종적으로 `Gradient Boosting`을 선택할 수 있다.

최종 성능:

```text
Best model: Gradient Boosting

              precision    recall  f1-score   support

           0       0.92      0.98      0.95       363
           1       0.80      0.48      0.60        58

    accuracy                           0.91       421
   macro avg       0.86      0.73      0.78       421
weighted avg       0.91      0.91      0.90       421
```

선택 근거:

- `Response = 1` precision이 `0.80`으로 가장 높다.
- 전체 accuracy와 F1도 가장 안정적이다.
- PR-AUC도 가장 높아 positive class를 확률적으로 구분하는 능력이 좋다.
- 캠페인 비용이나 타겟팅 효율을 고려하면, 무작정 recall만 높은 모델보다 더 현실적인 선택이다.

단, 응답 고객을 최대한 놓치지 않는 것이 최우선 목표라면 `Decision Tree` 또는 `Random Forest`를 함께 후보로 제시하는 것이 좋다.

## 14. Summary

이 과제의 핵심은 불균형 분류 문제에서 accuracy만 보지 않고, `Response = 1`에 대한 precision-recall trade-off를 함께 보는 것이다.

최종 정리:

- Baseline accuracy는 높지만 positive class를 전혀 잡지 못한다.
- Recall만 보면 Decision Tree가 가장 높다.
- Precision과 PR-AUC를 보면 Gradient Boosting이 가장 좋다.
- 실무적인 캠페인 효율까지 고려하면 Gradient Boosting을 최종 모델로 선택할 수 있다.
- 다만 과제 목표가 recall maximization이라면 Decision Tree 또는 Random Forest 결과도 같이 해석해야 한다.
