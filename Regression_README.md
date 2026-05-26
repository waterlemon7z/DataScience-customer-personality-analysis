## Merge two Reg. Methods   
What is different? (MS vs. JS)
 - Age 기준 (2014 vs 2026)
 - income none 값 처리방식 (dropna vs SimpleImputer)
 - income outlier 처리방식 (없음 vs <= 200000)
 - Age outlier 처리방식 (없음 vs Age >= 18 && Age <= 100)

## Preprocessing

- Dt_Customer 날짜 변환
- Age 생성 (기준 년도 2014)
- Customer_Days 생성
- TotalChildren 생성
- TotalSpending 생성
- TotalPurchases 생성
- CampaignAcceptedTotal 생성
- 희귀 Marital_Status 통합
- Age outlier 제거: 18 <= Age <= 100
- Income outlier 제거: Income <= 200000, NaN은 유지
- 불필요 컬럼 제거
- Regression feature 선택: Income, Age, Kidhome, Teenhome
- Income 결측값 median imputation
- StandardScaler 적용
- Polynomial 모델에서는 degree=2 feature 확장

## Regression

Test R2:
- Linear:     0.7526
- Polynomial: 0.7625

MAE:
- Linear:     234.97
- Polynomial: 230.27

RMSE:
- Linear:     305.36
- Polynomial: 299.18