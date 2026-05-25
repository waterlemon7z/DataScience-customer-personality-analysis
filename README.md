### 1. 라이브러리 설치

```bash
pip install -r requirements.txt
```

### 2. 데이터 다운로드

[Kaggle](https://www.kaggle.com/datasets/imakash3011/customer-personality-analysis/data)에서
`marketing_campaign.csv` 파일을 다운로드해서 프로젝트 폴더에 넣어주세요.

### 3. 실행

두 파일은 독립적으로 실행됩니다:

```bash
# 분류만 실행
python classification.py

# 군집화만 실행
python clustering.py
```

모든 결과(그래프, 표)는 `outputs/` 폴더에 자동 저장됩니다.

## Methodology

### 공통 전처리 (Preprocessing)

- `Dt_Customer` 파싱 후 `Customer_Tenure_Days` (가입 후 경과일) 생성
- 파생변수 생성: `Age`, `TotalChildren`, `TotalSpending`, `TotalPurchases`, `CampaignAcceptedTotal`
- 이상치 제거: `Year_Birth < 1940` (3명), `Income > 200,000` (1명)
- 희소 범주 통합: `Marital_Status`의 Alone, Absurd, YOLO → "Other"
- 상수 컬럼 제거: `Z_CostContact`, `Z_Revenue`
- CV의 각 fold 내에서 결측치 처리/인코딩/스케일링 수행 → **데이터 누수 방지**

### `classification.py` — 응답 예측

- **모델**: Logistic Regression, KNN, Decision Tree, Random Forest
- **검증**: 5-fold Stratified Cross-Validation
- **불균형 처리**: `class_weight='balanced'` (양성 클래스 비율 14.9%)
- **평가 지표**: Accuracy, Precision, Recall, F1, ROC-AUC
- **모델 선택**: ROC-AUC 기준 (불균형 데이터에 강함)
- 최종 모델에 대해 20% 홀드아웃 테스트셋으로 confusion matrix까지 확인

### `clustering.py` — 고객 세분화

- **알고리즘**: K-Means (Euclidean distance라 StandardScaler 적용)
- **k 선택**: k=2~8 범위에서 Elbow + Silhouette 비교
- **최종 k=4**: Silhouette 점수만 보면 k=2지만, k=2는 "고소득 vs 저소득" 정도로만 나뉘어 마케팅 페르소나로 활용하기엔 너무 거칠어서 k=4 선택
- **Response 제외**: 군집화 변수에서 Response를 빼고, 군집 형성 **후** 각 군집의 응답률을 해석에만 사용
- **시각화**: Elbow, Silhouette, PCA 2D 투영, 군집별 응답률, 히트맵

## Key Results

### Classification

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| **Logistic Regression**  | 0.819 | 0.443 | **0.784** | 0.565 | **0.897** |
| KNN | 0.871 | 0.665 | 0.281 | 0.395 | 0.831 |
| Decision Tree | 0.765 | 0.357 | 0.698 | 0.471 | 0.750 |
| Random Forest | 0.882 | 0.699 | 0.377 | 0.489 | 0.891 |

Logistic Regression이 ROC-AUC 0.897로 최고 성능을 보였으며,
양성 클래스(응답자) Recall 78%를 달성해 캠페인 타겟팅에 적합합니다.

### Clustering (4 Customer Personas)

| Cluster | Persona | Size | Income | Spending | Response Rate |
|---|---|---|---|---|---|
| **3** | VIP 충성 고객 | 193 | $80,749 | $1,663 | **56.5%** |
| **1** | 자녀 있는 가족 고객 | 448 | $54,393 | $706 | 16.7% |
| **0** | 고소득 무관심층 | 591 | $70,885 | $1,045 | 11.2% |
| **2** | 저소득·소액 고객 | 1,004 | $34,127 | $100 | 8.4% |

###  마케팅 전략 제안

- **Cluster 3 (VIP)**: 프리미엄 상품 제안, VIP 전용 혜택, 로열티 프로그램.
  ROI가 가장 높은 세그먼트로 새 캠페인 우선 타겟.
- **Cluster 1 (자녀 가족)**: 할인 프로모션, 번들 상품, 가족 친화적 상품.
- **Cluster 0 (고소득 무관심)**: 응답률이 낮은 이유 분석 필요.
  채널이나 상품 카테고리 다양화 시도.
- **Cluster 2 (저소득)**: 비용이 큰 직접 캠페인은 비효율.
  저비용 웹 기반 채널 위주로 운영.

## Dataset

- **Source**: [Customer Personality Analysis (Kaggle)](https://www.kaggle.com/datasets/imakash3011/customer-personality-analysis/data)
- **Records**: 2,240명
- **Features**: 29개 (인구통계, 가족, 상품별 구매금액, 구매채널, 캠페인 응답)
- **Target**: `Response` (1 = 마지막 캠페인 수락, 0 = 수락 안 함)
