# Response Classification

This code predicts `Response`, the binary target showing whether a customer accepted the last marketing campaign.

## Main choices

- Target: `Response`
- Features: demographic, family, customer tenure, purchase behavior, previous campaign response
- Imbalance handling: `StratifiedKFold` and `class_weight="balanced"` for Logistic Regression and Decision Tree
- No data augmentation
- Important leakage rule: `CampaignAcceptedTotal` is calculated only from `AcceptedCmp1` to `AcceptedCmp5`; it does not include `Response`

## Run

```bash
python classification/response_classification.py --input ../marketing_campaign.xlsx
```

## Outputs

- `classification_model_results.csv`
- `classification_data_summary.json`
- `confusion_matrix_<model>.csv`
- `confusion_matrix_<model>.png`
- `classification_report_<model>.json`
- `prediction_sample_<model>.csv`

