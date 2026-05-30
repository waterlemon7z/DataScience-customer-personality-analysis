# 202235134 최민석 최종 수정
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    import kagglesdk.kaggle_env as kaggle_env

    if not hasattr(kaggle_env, "get_web_endpoint"):
        kaggle_env.get_web_endpoint = kaggle_env.get_endpoint
except ImportError:
    pass

import kagglehub
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, PolynomialFeatures, StandardScaler
import src.util.api as api

@dataclass
class RecommendationRunResult:
    """
    - Name - RecommendationRunResult
    - Type - Class
    - Params - None
    - Returns - None
    - Description - Save user based recommendation results.
    """
    summary: pd.DataFrame
    model_results: dict[str, dict[str, Any]]
    final_result_key: str
    final_predictions: pd.DataFrame
    baseline_product: str
    baseline_match_rate: float
    product_prior: pd.Series


def normalize_predicted_ratios(raw_pred: Any) -> np.ndarray:
    """
    - Name - normalize_predicted_ratios
    - Type - Function
    - Params - raw_pred: Any
    - Returns - numpy
    - Description - numpy
    """
    pred = np.clip(np.asarray(raw_pred, dtype=float), 0, None)
    if pred.ndim == 1:
        pred = pred.reshape(-1, 1)

    row_sums = pred.sum(axis=1, keepdims=True)
    return np.divide(pred, row_sums, out=np.zeros_like(pred), where=row_sums != 0)


def relative_preference_score(ratio_df: pd.DataFrame, product_prior: pd.Series, ) -> pd.DataFrame:
    """
    - Name - relative_preference_score
    - Type - Function
    - Params - ratio_df: pd.DataFrame, product_prior: pd.Series
    - Returns - pd.DataFrame
    - Description - 상대 선호 점수 = 고객의 예측 제품군 비율 / 전체 평균 제품군 비율
    """
    score_df = ratio_df.div(product_prior, axis=1)
    return score_df.replace([np.inf, -np.inf], np.nan).fillna(0)


def banner_ad_order(score_df: pd.DataFrame) -> pd.Series:
    """
    - Name - banner_ad_order
    - Type - Function
    - Params - score_df: pd.DataFrame
    - Returns - pd.Series
    - Description - 광고 배너 순서 표시
    """
    return score_df.apply(
        lambda row: " > ".join(row.sort_values(ascending=False).index),
        axis=1,
    )


def banner_ad_scores(score_df: pd.DataFrame) -> pd.Series:
    """
    - Name - banner_ad_scores
    - Type - Function
    - Params - score_df: pd.DataFrame
    - Returns - pd.Series
    - Description - 광고 배너 순서 표시 기준 스코어 표시
    """
    return score_df.apply(
        lambda row: " | ".join(
            f"{product}:{score:.3f}"
            for product, score in row.sort_values(ascending=False).items()
        ),
        axis=1,
    )


def best_by_business_model(summary_df: pd.DataFrame) -> pd.DataFrame:
    """
    - Name - best_by_business_model
    - Type - Function
    - Params - summary_df: pd.DataFrame
    - Returns - pd.Series
    - Description - Return the best row per business model from a model summary table.
    """
    return (
        summary_df.sort_values(
            by=["Business Model", "Recommendation Match Rate", "Test R2"],
            ascending=[True, False, False],
        )
        .groupby("Business Model", as_index=False)
        .head(1)
    )


class MarketingItemRecommendationRegression:
    """
    - Name - MarketingItemRecommendationRegression
    - Type - Class
    - Params - None
    - Returns - None
    - Description - 마케팅 시 유저 특화된 광고 배너를 띄우기 위한 모델 정의
    """

    def __init__(self, random_state: int = 42, ):
        """
        - Name - __init__
        - Type - Functions
        - Params - random_state: int = 42
        - Returns - None
        - Description - Initialize.
        """
        self.figure_dir = Path(__file__).resolve().parents[2] / "output" / "regression"
        self.random_state = random_state

        self.raw_df: pd.DataFrame | None = None
        self.df: pd.DataFrame | None = None
        self.X: pd.DataFrame | None = None
        self.y: pd.DataFrame | None = None
        self.last_result: RecommendationRunResult | None = None

    def load_dataset_from_kaggle(self) -> pd.DataFrame:
        """
        - Name - load_dataset_from_kaggle
        - Type - Functions
        - Params - None
        - Returns - pd.DataFrame
        - Description - 캐글 서버에서 csv 파일을 받아온다.
        """
        self.raw_df = api.load_dataset_from_kaggle()
        return self.raw_df.copy()

    def prepare_dataframe(self, df: pd.DataFrame, min_age: int = 18, max_age: int = 100,
                          max_income: int = 300000, ) -> pd.DataFrame:
        """
        - Name - prepare_dataframe
        - Type - Functions
        - Params - df: pd.DataFrame, min_age: int = 18, max_age: int = 100, max_income: int = 300000
        - Returns - pd.DataFrame
        - Description - 기본적으로 사용할 feature들을 정의하고 일부 feature의 경우 다른 측도로 변환
        """
        self._validate_columns(
            df,
            [
                "Year_Birth",
                "Income",
                "Education",
                "Marital_Status",
                "Kidhome",
                "Teenhome",
            ],
        )

        prepared_df = df.copy()
        prepared_df["Age"] = 2014 - prepared_df["Year_Birth"]  # Year Birth to Age
        prepared_df = prepared_df[
            (prepared_df["Age"] >= min_age) & (prepared_df["Age"] <= max_age)]  # Remove Age Outlier
        prepared_df = prepared_df[
            prepared_df["Income"].isna() | (prepared_df["Income"] <= max_income)]  # Remove Income Outlier

        prepared_df["Education"] = prepared_df["Education"].astype("category")  # 범주형으로 변경

        rare_status = ["Alone", "Absurd", "YOLO"]  # 데이터 수가 적거나 의미가 모호한 결혼 상태 값
        prepared_df["Marital_Status"] = prepared_df["Marital_Status"].replace(rare_status,
                                                                              "Other")  # 희귀 결혼 상태를 Other로 통합
        marital_mode = prepared_df["Marital_Status"].mode(dropna=True)  # 결측값 대체에 사용할 최빈 결혼 상태 계산
        marital_fill = marital_mode.iloc[0] if not marital_mode.empty else "Other"  # 최빈값이 없으면 Other를 기본값으로 사용
        prepared_df["Marital_Status"] = prepared_df["Marital_Status"].fillna(marital_fill)  # 결혼 상태 결측값을 최빈값으로 대체
        prepared_df["Marital_Status"] = prepared_df["Marital_Status"].astype("category")  # 결혼 상태를 범주형 변수로 변환

        prepared_df["HasPartner"] = prepared_df["Marital_Status"].isin(["Married", "Together"]).astype(
            int)  # 배우자 여부를 0 또는 1로 생성
        prepared_df["TotalChildren"] = prepared_df["Kidhome"] + prepared_df["Teenhome"]  # 집에 있는 아동과 청소년 수를 합산

        return prepared_df

    def build_feature_matrix(self, df: pd.DataFrame,
                             feature_sets: Mapping[str, Sequence[str]] | None = None, ) -> pd.DataFrame:
        """
        - Name - build_feature_matrix
        - Type - Functions
        - Params - df: pd.DataFrame, feature_sets: Mapping[str, Sequence[str]] | None = None
        - Returns - pd.DataFrame
        - Description - build_feature_matrix
        """
        feature_sets = self._copy_feature_sets(feature_sets)
        all_feature_cols = sorted({col for cols in feature_sets.values() for col in cols})
        self._validate_columns(df, all_feature_cols)
        return df[all_feature_cols].copy()

    def build_targets(self, df: pd.DataFrame, product_cols: Sequence[str] | None = None, ) -> tuple[
        pd.DataFrame, pd.Index, pd.Series]:
        """
        - Name - build_targets
        - Type - Functions
        - Params - df: pd.DataFrame, product_cols: Sequence[str] | None = None
        - Returns - tuple[pd.DataFrame, pd.Index, pd.Series]
        - Description - target 변수를 지정 "MntWines", "MntFruits", "MntMeatProducts", "MntFishProducts", "MntSweetProducts", "MntGoldProds",
        """
        if product_cols is None:
            product_cols = [
                "MntWines",
                "MntFruits",
                "MntMeatProducts",
                "MntFishProducts",
                "MntSweetProducts",
                "MntGoldProds",
            ]
        product_cols = list(product_cols)
        self._validate_columns(df, product_cols)

        total_spending = df[product_cols].sum(axis=1)
        valid_target_index = total_spending[total_spending > 0].index
        y = df.loc[valid_target_index, product_cols].div(
            total_spending.loc[valid_target_index],
            axis=0,
        )
        return y, valid_target_index, total_spending

    def build_stage_preprocessor(self, stage_features: Sequence[str],
                                 log_numeric_candidates: Sequence[str] | None = None,
                                 standard_numeric_candidates: Sequence[str] | None = None,
                                 categorical_candidates: Sequence[str] | None = None) -> ColumnTransformer:
        """
        - Name - build_stage_preprocessor
        - Type - Functions
        - Params - stage_features: Sequence[str], log_numeric_candidates: Sequence[str] | None = None, standard_numeric_candidates: Sequence[str] | None = None, categorical_candidates: Sequence[str] | None = None
        - Returns - ColumnTransformer
        - Description - feature 별로 적용할 preprocessing 나누기
        """
        if log_numeric_candidates is None:  # 로그 변환을 적용할 숫자형 후보 컬럼이 따로 없으면 기본값 사용
            log_numeric_candidates = [
                "Income",
                "NumWebVisitsMonth",
                "NumWebPurchases",
                "NumCatalogPurchases",
                "NumStorePurchases",
            ]
        if standard_numeric_candidates is None:  # 일반 표준화만 적용할 숫자형 후보 컬럼이 따로 없으면 기본값 사용
            standard_numeric_candidates = [
                "Age",
                "Kidhome",
                "Teenhome",
                "Recency",
            ]
        if categorical_candidates is None:  # 원핫 인코딩할 범주형 후보 컬럼이 따로 없으면 기본값 사용
            categorical_candidates = [
                "Education",
                "Marital_Status",
            ]

        stage_features = list(stage_features)  # 현재 비즈니스 단계에서 사용할 feature 목록을 리스트로 변환
        log_numeric_features = [
            col for col in log_numeric_candidates
            if col in stage_features
        ]  # 현재 단계 feature 중 로그 변환 대상 숫자형 컬럼만 선택
        standard_numeric_features = [
            col for col in standard_numeric_candidates
            if col in stage_features
        ]  # 현재 단계 feature 중 일반 숫자형 컬럼만 선택
        categorical_features = [
            col for col in categorical_candidates
            if col in stage_features
        ]  # 현재 단계 feature 중 범주형 컬럼만 선택

        transformers = []  # ColumnTransformer에 넣을 전처리 파이프라인 목록

        if log_numeric_features:  # 로그 변환 대상 숫자형 컬럼이 있으면 해당 전처리 파이프라인 추가
            transformers.append(
                (
                    "log_num",  # ColumnTransformer 안에서 사용할 파이프라인 이름
                    Pipeline(
                        steps=[
                            ("imputer", SimpleImputer(strategy="median")),  # 결측값을 중앙값으로 대체
                            ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
                            # 큰 값의 영향을 줄이기 위해 log(1+x) 변환
                            ("scaler", StandardScaler()),  # 평균 0, 표준편차 1로 표준화
                        ]
                    ),
                    log_numeric_features,  # 위 파이프라인을 적용할 컬럼 목록
                )
            )

        if standard_numeric_features:  # 일반 숫자형 컬럼이 있으면 해당 전처리 파이프라인 추가
            transformers.append(
                (
                    "num",  # ColumnTransformer 안에서 사용할 파이프라인 이름
                    Pipeline(
                        steps=[
                            ("imputer", SimpleImputer(strategy="median")),  # 결측값을 중앙값으로 대체
                            ("scaler", StandardScaler()),  # 평균 0, 표준편차 1로 표준화
                        ]
                    ),
                    standard_numeric_features,  # 위 파이프라인을 적용할 컬럼 목록
                )
            )

        if categorical_features:  # 범주형 컬럼이 있으면 해당 전처리 파이프라인 추가
            transformers.append(
                (
                    "cat",  # ColumnTransformer 안에서 사용할 파이프라인 이름
                    Pipeline(
                        steps=[
                            ("imputer", SimpleImputer(strategy="most_frequent")),  # 결측값을 최빈값으로 대체
                            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),  # 범주형 값을 원핫 인코딩
                        ]
                    ),
                    categorical_features,  # 위 파이프라인을 적용할 컬럼 목록
                )
            )

        if not transformers:  # 적용할 전처리 대상 컬럼이 하나도 없으면 오류 처리
            raise ValueError("No usable stage features were found for preprocessing.")  # 잘못된 feature set 입력을 알림

        return ColumnTransformer(transformers=transformers)  # 숫자형/범주형 전처리를 컬럼별로 묶은 전처리기 반환

    def default_models(self) -> dict[str, Any]:
        """
        - Name - default_models
        - Type - Function
        - Params - None
        - Returns - dict[str, Any]
        - Description - Return models which we use
        """
        return {
            "Multiple Linear Regression": LinearRegression(),
            "Polynomial Regression degree 2": Pipeline(
                steps=[
                    ("poly", PolynomialFeatures(degree=2, include_bias=False)),
                    ("linear", LinearRegression()),
                ]
            ),
        }

    def run_pipeline(
            self, df: pd.DataFrame | None = None, feature_sets: Mapping[str, Sequence[str]] | None = None,
            models: Mapping[str, Any] | None = None, product_cols: Sequence[str] | None = None, test_size: float = 0.2,
            cv_splits: int = 5, verbose: bool = True, plot: bool = True,
    ) -> RecommendationRunResult:
        """
        - Name - run_pipeline
        - Type - Function
        - Params - df: pd.DataFrame | None = None, feature_sets: Mapping[str, Sequence[str]] | None = None,
            models: Mapping[str, Any] | None = None, product_cols: Sequence[str] | None = None, test_size: float = 0.2,
            cv_splits: int = 5, verbose: bool = True, plot: bool = True,
        - Returns - RecommendationRunResult
        - Description - running preprocessing and model training
        """
        if df is None:  # df 없으면
            df = self.raw_df if self.raw_df is not None else self.load_dataset_from_kaggle()  # 기존 로드 데이터가 있으면 쓰고, 없으면 Kaggle에서 로드

        feature_sets = self._copy_feature_sets(feature_sets)  # 사용할 고객 단계별 feature set을 준비
        prepared_df = self.prepare_dataframe(df)  # preprocessing
        X = self.build_feature_matrix(prepared_df, feature_sets)  # feature set에 필요한 입력 컬럼만 모아 X 생성
        y, valid_target_index, _ = self.build_targets(prepared_df, product_cols)  # 제품군별 지출 비율 target과 유효 고객 인덱스 생성
        X_model = X.loc[valid_target_index].copy()  # 결측값 제거한 index 적용

        result = self.evaluate_models(  # training model and save result
            X_model=X_model,
            y=y,
            feature_sets=feature_sets,
            models=models,
            product_cols=product_cols,
            test_size=test_size,
            cv_splits=cv_splits,
            verbose=verbose,
        )

        self.df = prepared_df
        self.X = X_model
        self.y = y
        self.last_result = result

        if plot:
            self.plot_model_summary(result)
            self.plot_final_diagnostics(result)

        return result

    def evaluate_models(self, X_model: pd.DataFrame, y: pd.DataFrame,
                        feature_sets: Mapping[str, Sequence[str]] | None = None,
                        models: Mapping[str, Any] | None = None, product_cols: Sequence[str] | None = None,
                        test_size: float = 0.2, cv_splits: int = 5, verbose: bool = True, ) -> RecommendationRunResult:
        """
        - Name - evaluate_models
        - Type - Function
        - Params - X_model: pd.DataFrame,y: pd.DataFrame,feature_sets: Mapping[str, Sequence[str]] | None = None,models: Mapping[str, Any] | None = None,product_cols: Sequence[str] | None = None,test_size: float = 0.2,cv_splits: int = 5,verbose: bool = True,
        - Returns - RecommendationRunResult
        - Description - evaluate models
            Evaluation metrics:
            1. CV R2 Mean / CV R2 Std - KFold cross validation R2 average and standard deviation
            2. Test R2 - R2 score on the hold-out test set
            3. Test MAE - mean absolute error on the hold-out test set
            4. Test RMSE - root mean squared error on the hold-out test set
            5. Recommendation Match Rate - actual relative-preference product and recommended product match rate
            6. Baseline Match Rate - match rate when recommending the most common train-set relative-preference product
            7. Lift vs Baseline - Recommendation Match Rate minus Baseline Match Rate
        """
        feature_sets = self._copy_feature_sets(feature_sets)  # 고객 단계별 feature set 준비
        models = dict(models) if models is not None else self.default_models()  # 전달된 모델이 없으면 기본 모델 사용
        if product_cols is None:  # 제품군 컬럼을 따로 넘기지 않으면 기본 6개 제품군 사용
            product_cols = [
                "MntWines",
                "MntFruits",
                "MntMeatProducts",
                "MntFishProducts",
                "MntSweetProducts",
                "MntGoldProds",
            ]
        product_cols = list(product_cols)  # DataFrame 컬럼명으로 사용할 수 있도록 리스트로 변환

        train_index, test_index = train_test_split(  # 같은 고객 기준으로 X와 y를 train/test로 나누기 위한 인덱스 분리
            X_model.index,  # 모델 입력 데이터의 고객 인덱스
            test_size=test_size,
            random_state=self.random_state,
        )

        X_train_all = X_model.loc[train_index]  # train 입력 데이터
        X_test_all = X_model.loc[test_index]  # test 입력 데이터
        y_train = y.loc[train_index]  # train target 비율
        y_test = y.loc[test_index]  # test target 비율

        product_prior = y_train.mean().replace(0, np.nan)  # train set의 제품군별 평균 비율
        y_train_relative_score = relative_preference_score(y_train, product_prior)  # train 실제값을 평균 대비 상대 선호 점수로 변환
        y_test_relative_score = relative_preference_score(y_test, product_prior)  # test 실제값을 평균 대비 상대 선호 점수로 변환

        baseline_product = y_train_relative_score.idxmax(axis=1).mode()[0]  # train에서 가장 자주 실제 선호 1위였던 제품군
        actual_relative_product = y_test_relative_score.idxmax(axis=1)  # test 고객별 실제 상대 선호 1위 제품군
        baseline_match_rate = float((actual_relative_product == baseline_product).mean())  # baseline 추천이 맞은 비율
        cv = KFold(n_splits=cv_splits, shuffle=True, random_state=self.random_state)  # 교차검증 설정

        model_results: dict[str, dict[str, Any]] = {}  # 학습된 모델과 예측 결과 저장
        model_summary = []  # 모델별 성능 요약 저장

        if verbose:
            print("Baseline product:", baseline_product)
            print("Baseline match rate:", baseline_match_rate)
            print("Product prior:")
            print(product_prior)

        for stage_name, stage_features in feature_sets.items():  # New-user, Early-behavior, Existing-customer별 반복
            stage_features = list(stage_features)  # 현재 단계에서 사용할 feature 목록
            X_stage = X_model[stage_features]  # 교차검증에 사용할 현재 단계 입력 데이터
            X_train = X_train_all[stage_features]  # 현재 단계 train 입력 데이터
            X_test = X_test_all[stage_features]  # 현재 단계 test 입력 데이터

            for model_name, model in models.items():  # 각 회귀 모델별 반복
                reg = Pipeline(
                    steps=[
                        ("preprocessor", self.build_stage_preprocessor(stage_features)),  # 현재 feature set에 맞는 전처리기
                        ("model", clone(model)),  # 원본 모델 객체를 복제해서 독립적으로 학습
                    ]
                )

                cv_scores = cross_val_score(reg, X_stage, y, cv=cv, scoring="r2")  # KFold CV로 R2 점수 계산

                reg.fit(X_train, y_train)  # train 데이터로 모델 학습
                y_pred = normalize_predicted_ratios(reg.predict(X_test))  # test 예측값을 제품군별 비율로 보정
                y_pred_df = pd.DataFrame(y_pred, columns=product_cols, index=y_test.index)  # 예측 비율을 DataFrame으로 변환
                recommendation_score_df = relative_preference_score(y_pred_df, product_prior)  # 예측 비율을 평균 대비 추천 점수로 변환

                recommended_product = recommendation_score_df.idxmax(axis=1)  # 고객별 추천 점수 1위 제품군
                recommendation_match_rate = float(
                    (actual_relative_product == recommended_product).mean()
                )  # 실제 상대 선호 1위와 추천 1위가 일치한 비율

                test_r2 = float(r2_score(y_test, y_pred))  # test set R2
                test_mae = float(mean_absolute_error(y_test, y_pred))  # test set MAE
                test_rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))  # test set RMSE

                result_key = f"{stage_name} | {model_name}"  # 결과를 구분하기 위한 key

                if verbose:
                    print(f"\n--- {result_key} ---")
                    print("Features:", stage_features)
                    print("5-fold CV R2:", cv_scores)
                    print("Mean CV R2:", cv_scores.mean())
                    print("Test R2:", test_r2)
                    print("Test MAE:", test_mae)
                    print("Test RMSE:", test_rmse)
                    print("Relative recommendation match rate:", recommendation_match_rate)

                model_summary.append(  # 현재 stage/model 조합의 평가 지표 저장
                    {
                        "Business Model": stage_name,
                        "Regression Model": model_name,
                        "Feature Count": len(stage_features),
                        "Features": ", ".join(stage_features),
                        "CV R2 Mean": float(cv_scores.mean()),
                        "CV R2 Std": float(cv_scores.std()),
                        "Test R2": test_r2,
                        "Test MAE": test_mae,
                        "Test RMSE": test_rmse,
                        "Recommendation Match Rate": recommendation_match_rate,
                        "Baseline Match Rate": baseline_match_rate,
                        "Lift vs Baseline": recommendation_match_rate - baseline_match_rate,
                    }
                )

                result_df = self.build_prediction_frame(  # 고객별 실제/예측/추천 결과 테이블 생성
                    y_test=y_test,
                    y_pred_df=y_pred_df,
                    y_test_relative_score=y_test_relative_score,
                    recommendation_score_df=recommendation_score_df,
                    actual_product=actual_relative_product,
                    recommended_product=recommended_product,
                )

                model_results[result_key] = {  # 학습된 모델과 예측 결과를 나중에 재사용할 수 있게 저장
                    "business_model": stage_name,
                    "regression_model": model_name,
                    "features": stage_features,
                    "model": reg,
                    "predictions": result_df,
                }

        summary_df = pd.DataFrame(model_summary).sort_values(  # 성능 요약을 DataFrame으로 만들고 좋은 모델 순으로 정렬
            by=["Recommendation Match Rate", "Test R2"],  # 추천 일치율을 우선하고, 동률이면 Test R2 사용
            ascending=False,
        )
        final_result_key = (  # 가장 좋은 모델 조합의 key 생성
                summary_df.iloc[0]["Business Model"]
                + " | "
                + summary_df.iloc[0]["Regression Model"]
        )
        final_predictions = model_results[final_result_key]["predictions"]  # 최종 선택 모델의 고객별 추천 결과

        return RecommendationRunResult(  # 모델 비교 결과와 최종 추천 결과를 하나의 객체로 반환
            summary=summary_df,  # 모델별 성능 요약
            model_results=model_results,  # 모델별 상세 결과
            final_result_key=final_result_key,  # 최종 선택 모델 key
            final_predictions=final_predictions,  # 최종 선택 모델의 예측/추천 결과
            baseline_product=baseline_product,  # baseline 추천 제품군
            baseline_match_rate=baseline_match_rate,  # baseline 추천 일치율
            product_prior=product_prior,  # 제품군별 train 평균 비율
        )

    def build_prediction_frame(self, y_test: pd.DataFrame, y_pred_df: pd.DataFrame, y_test_relative_score: pd.DataFrame,
                               recommendation_score_df: pd.DataFrame, actual_product: pd.Series,
                               recommended_product: pd.Series, ) -> pd.DataFrame:
        """
           - Name - build_prediction_frame
           - Type - Functions
           - Params - y_test: pd.DataFrame, y_pred_df: pd.DataFrame, y_test_relative_score: pd.DataFrame, recommendation_score_df: pd.DataFrame, actual_product: pd.Series, recommended_product: pd.Series,
           - Returns - pd.DataFrame
           - Description - 예측한 결과를 리턴하는 함수
       """
        result_df = pd.concat(
            [
                y_test.add_prefix("ActualRatio_"),
                y_pred_df.add_prefix("PredictedRatio_"),
                y_test_relative_score.add_prefix("ActualRelativeScore_"),
                recommendation_score_df.add_prefix("RecommendationScore_"),
            ],
            axis=1,
        )
        result_df["Actual_Product"] = actual_product
        result_df["Recommended_Product"] = recommended_product
        result_df["Recommendation_Match"] = actual_product == recommended_product
        result_df["Banner_Ad_Order"] = banner_ad_order(recommendation_score_df)
        result_df["Banner_Ad_Scores"] = banner_ad_scores(recommendation_score_df)
        return result_df

    def recommend_customers(self, df: pd.DataFrame, result: RecommendationRunResult | None = None,
                            result_key: str | None = None, prepare: bool = True,
                            product_cols: Sequence[str] | None = None, ) -> pd.DataFrame:
        """
           - Name - recommend_customers
           - Type - Functions
           - Params - df: pd.DataFrame, result: RecommendationRunResult | None = None,
                            result_key: str | None = None, prepare: bool = True,
                            product_cols: Sequence[str] | None = None,
           - Returns - pd.DataFrame
           - Description - 학습된 추천 모델을 사용해 입력 고객별 제품군 예측 비율과 상대 선호 점수를 계산하고,
                           가장 높은 추천 점수의 제품군과 광고 배너 노출 순서를 반환한다.
       """
        if product_cols is None:
            product_cols = [
                "MntWines",
                "MntFruits",
                "MntMeatProducts",
                "MntFishProducts",
                "MntSweetProducts",
                "MntGoldProds",
            ]
        result = self._require_result(result)
        result_key = result_key or result.final_result_key
        model_info = result.model_results[result_key]
        stage_features = list(model_info["features"])

        scoring_df = self.prepare_dataframe(df) if prepare else df.copy()
        self._validate_columns(scoring_df, stage_features)

        raw_pred = model_info["model"].predict(scoring_df[stage_features])
        y_pred = normalize_predicted_ratios(raw_pred)
        ratio_df = pd.DataFrame(y_pred, columns=list(product_cols), index=scoring_df.index)
        score_df = relative_preference_score(ratio_df, result.product_prior)

        output_df = pd.concat(
            [
                ratio_df.add_prefix("PredictedRatio_"),
                score_df.add_prefix("RecommendationScore_"),
            ],
            axis=1,
        )
        output_df["Recommended_Product"] = score_df.idxmax(axis=1)
        output_df["Banner_Ad_Order"] = banner_ad_order(score_df)
        output_df["Banner_Ad_Scores"] = banner_ad_scores(score_df)
        return output_df

    def get_banner_recommendation_output(self, final_predictions: pd.DataFrame | None = None, ) -> pd.DataFrame:
        """
           - Name - get_banner_recommendation_output
           - Type - Functions
           - Params - df: final_predictions: pd.DataFrame | None = None,
           - Returns - pd.DataFrame
           - Description - csv출력을 위한 df 반환
       """
        if final_predictions is None:
            final_predictions = self._require_result().final_predictions
        return final_predictions[
            [
                "Actual_Product",
                "Recommended_Product",
                "Recommendation_Match",
                "Banner_Ad_Order",
                "Banner_Ad_Scores",
            ]
        ].copy()

    def get_customer_score_table(self, customer_id: Any | None = None, final_predictions: pd.DataFrame | None = None,
                                 product_cols: Sequence[str] | None = None, ) -> pd.DataFrame:
        """
           - Name - get_customer_score_table
           - Type - Functions
           - Params - customer_id: Any | None = None, final_predictions: pd.DataFrame | None = None,
                                 product_cols: Sequence[str] | None = None
           - Returns - pd.DataFrame
           - Description - 제품군의 비율별 정렬을 해서 가장 높은 순으로 정렬
       """
        if product_cols is None:
            product_cols = [
                "MntWines",
                "MntFruits",
                "MntMeatProducts",
                "MntFishProducts",
                "MntSweetProducts",
                "MntGoldProds",
            ]
        if final_predictions is None:
            final_predictions = self._require_result().final_predictions  # 최종 예측 저장
        if customer_id is None:
            customer_id = final_predictions.index[0]

        score_cols = [f"RecommendationScore_{col}" for col in product_cols]
        score_table = final_predictions.loc[[customer_id], score_cols].T.reset_index()
        score_table.columns = ["Product", "RecommendationScore"]
        score_table["Product"] = score_table["Product"].str.replace(
            "RecommendationScore_",
            "",
            regex=False,
        )
        return score_table.sort_values("RecommendationScore", ascending=False)

    def plot_model_summary(self, result: RecommendationRunResult | None = None, save: bool = True,
                           show: bool = False, ) -> None:
        """
           - Name - plot_model_summary
           - Type - Functions
           - Params - result: RecommendationRunResult | None = None, save: bool = True,
                           show: bool = False
           - Returns - None
           - Description - 모델의 요약 plot을 저장
       """
        import seaborn as sns  # 막대그래프 시각화를 위한 라이브러리

        result = self._require_result(result)  # 전달된 결과가 없으면 마지막 학습 결과를 사용
        summary_df = result.summary  # 모델별 평가 지표가 저장된 요약 테이블

        plt.figure(figsize=(10, 5))  # 추천 일치율 그래프 크기 설정
        sns.barplot(
            data=summary_df,  # 모델 평가 요약 데이터 사용
            x="Business Model",  # 고객 단계별 모델 구분
            y="Recommendation Match Rate",  # 실제 선호 제품군과 추천 제품군이 일치한 비율
            hue="Regression Model",  # 회귀 모델 종류별로 색상 구분
        )
        plt.axhline(
            result.baseline_match_rate,  # 비교 기준이 되는 baseline 추천 일치율
            color="red",
            linestyle="--",
            label=f"Baseline ({result.baseline_match_rate:.3f})",
        )
        plt.title("Relative Preference Match Rate by Business Model")
        plt.xlabel("Business Model")
        plt.ylabel("Match Rate")
        plt.ylim(0, 1)
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()
        self._finish_plot("relative_preference_match_rate_by_business_model", save, show)  # 그래프 저장/출력 처리

        r2_plot_df = summary_df.melt(  # CV R2와 Test R2를 한 그래프에 그리기 위해 long format으로 변환
            id_vars=["Business Model", "Regression Model"],  # 그래프에서 유지할 구분 컬럼
            value_vars=["CV R2 Mean", "Test R2"],  # 비교할 R2 지표 컬럼
            var_name="Metric",  # CV R2/Test R2 구분 이름
            value_name="R2",  # 실제 R2 값 컬럼 이름
        )

        plt.figure(figsize=(10, 5))  # R2 비교 그래프 크기 설정
        sns.barplot(
            data=r2_plot_df,  # 변환된 R2 지표 데이터 사용
            x="Business Model",  # 고객 단계별 모델 구분
            y="R2",  # 회귀 설명력 지표
            hue="Metric",  # CV R2와 Test R2를 색상으로 구분
        )
        plt.title("Regression R2 by Business Model")
        plt.xlabel("Business Model")
        plt.ylabel("R2")
        plt.tight_layout()
        self._finish_plot("regression_r2_by_business_model", save, show)  # 그래프 저장/출력 처리

        plt.figure(figsize=(10, 5))  # baseline 대비 향상도 그래프 크기 설정
        sns.barplot(
            data=summary_df,  # 모델 평가 요약 데이터 사용
            x="Business Model",  # 고객 단계별 모델 구분
            y="Lift vs Baseline",  # baseline 대비 추천 일치율 증가분
            hue="Regression Model",  # 회귀 모델 종류별로 색상 구분
        )
        plt.axhline(0, color="black", linewidth=1)  # baseline과 성능이 같은 기준선
        plt.title("Relative Recommendation Lift vs Baseline")
        plt.xlabel("Business Model")
        plt.ylabel("Match Rate Lift")
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()
        self._finish_plot("relative_recommendation_lift_vs_baseline", save, show)  # 그래프 저장/출력 처리

    def plot_final_diagnostics(self, result: RecommendationRunResult | None = None,
                               product_cols: Sequence[str] | None = None, save: bool = True, show: bool = False, ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
           - Name - plot_final_diagnostics
           - Type - Functions
           - Params - result: RecommendationRunResult | None = None,
                               product_cols: Sequence[str] | None = None, save: bool = True, show: bool = False
           - Returns - tuple[pd.DataFrame, pd.DataFrame]
           - Description - 최종 선택 모델의 추천 결과를 진단하기 위해 실제 선호 제품군과 추천 제품군의
                           혼동행렬 및 실제/추천 제품군 분포 그래프를 생성한다.
       """
        import seaborn as sns  # heatmap 시각화를 위한 라이브러리

        result = self._require_result(result)  # 전달된 결과가 없으면 마지막 학습 결과를 사용
        if product_cols is None:  # 제품군 컬럼을 따로 넘기지 않으면 기본 6개 제품군 사용
            product_cols = [
                "MntWines",
                "MntFruits",
                "MntMeatProducts",
                "MntFishProducts",
                "MntSweetProducts",
                "MntGoldProds",
            ]
        product_cols = list(product_cols)  # reindex에 사용할 수 있도록 리스트로 변환
        final_predictions = result.final_predictions  # 최종 선택 모델의 테스트셋 추천 결과

        confusion_counts = pd.crosstab(  # 실제 선호 제품군과 추천 제품군의 건수 기준 교차표 생성
            final_predictions["Actual_Product"],  # 행: 실제 상대 선호 제품군
            final_predictions["Recommended_Product"],  # 열: 모델이 추천한 제품군
        ).reindex(index=product_cols, columns=product_cols, fill_value=0)  # 모든 제품군 순서를 고정하고 없는 값은 0으로 채움

        confusion_rate = pd.crosstab(  # 실제 제품군별 추천 비율을 보기 위한 정규화 교차표 생성
            final_predictions["Actual_Product"],  # 행: 실제 상대 선호 제품군
            final_predictions["Recommended_Product"],  # 열: 모델이 추천한 제품군
            normalize="index",  # 각 실제 제품군 행의 합이 1이 되도록 비율화
        ).reindex(index=product_cols, columns=product_cols, fill_value=0)  # 제품군 순서를 고정하고 없는 값은 0으로 채움

        plt.figure(figsize=(8, 6))  # 혼동행렬 heatmap 크기 설정
        sns.heatmap(confusion_rate, annot=True, fmt=".2f", cmap="Blues")  # 실제 대비 추천 비율을 heatmap으로 표시
        plt.title(f"Actual vs Recommended Relative Product Ratio ({result.final_result_key})")
        plt.xlabel("Recommended Product")
        plt.ylabel("Actual Product")
        plt.tight_layout()
        self._finish_plot("actual_vs_recommended_relative_product_ratio", save, show)  # 그래프 저장/출력 처리

        product_distribution_df = pd.DataFrame(  # 실제 선호 제품군 분포와 추천 제품군 분포를 비교할 데이터 생성
            {
                "Actual": final_predictions["Actual_Product"].value_counts(normalize=True),  # 실제 상대 선호 제품군 비율
                "Recommended": final_predictions["Recommended_Product"].value_counts(normalize=True),  # 추천 제품군 비율
            }
        ).reindex(product_cols).fillna(0)  # 제품군 순서를 고정하고 없는 값은 0으로 채움

        product_distribution_df.plot(kind="bar", figsize=(9, 5))  # 실제/추천 제품군 분포를 막대그래프로 비교
        plt.title("Actual vs Recommended Relative Product Distribution")
        plt.xlabel("Product")
        plt.ylabel("Customer Share")
        plt.xticks(rotation=30, ha="right")  # 제품군 이름이 겹치지 않도록 x축 라벨 회전
        plt.tight_layout()
        self._finish_plot("actual_vs_recommended_relative_product_distribution", save, show)  # 그래프 저장/출력 처리

        return confusion_counts, product_distribution_df  # 표 형태 진단 결과도 반환

    def _finish_plot(self, filename: str, save: bool, show: bool) -> None:
        if save:  # save=True이면 그래프를 파일로 저장
            self.figure_dir.mkdir(exist_ok=True)  # figures 폴더가 없으면 생성
            plt.savefig(self.figure_dir / f"{filename}.png", dpi=300, bbox_inches="tight")  # 현재 그래프를 png 파일로 저장
        if show:  # show=True이면 화면에 그래프 출력
            plt.show()
        plt.close()  # 현재 figure를 닫아서 다음 그래프와 겹치지 않게 처리

    def _require_result(
            self,
            result: RecommendationRunResult | None = None,
    ) -> RecommendationRunResult:
        if result is not None:  # 함수 인자로 결과 객체가 들어온 경우
            return result  # 전달받은 결과를 그대로 사용
        if self.last_result is None:  # 인자로 받은 결과도 없고, 이전 실행 결과도 없는 경우
            raise ValueError("No recommendation result is available. Run run_pipeline() first.")  # 먼저 학습 파이프라인을 실행하라고 안내
        return self.last_result  # 마지막으로 실행된 추천 결과를 사용

    @staticmethod
    def _copy_feature_sets(
            feature_sets: Mapping[str, Sequence[str]] | None = None,
    ) -> dict[str, list[str]]:
        if feature_sets is None:  # 사용자가 feature set을 따로 넘기지 않은 경우 기본 feature set 사용
            feature_sets = {
                "New-user": [  # 신규 고객처럼 기본 인구통계 정보만 있는 경우
                    "Age",
                    "Income",
                    "Education",
                    "Marital_Status",
                    "Kidhome",
                    "Teenhome",
                ],
                "Early-behavior": [  # 웹 방문 정보까지 있는 초기 행동 고객
                    "Age",
                    "Income",
                    "Education",
                    "Marital_Status",
                    "Kidhome",
                    "Teenhome",
                    "NumWebVisitsMonth",
                ],
                "Existing-customer": [  # 구매 채널/최근성 정보까지 있는 기존 고객
                    "Age",
                    "Income",
                    "Education",
                    "Marital_Status",
                    "Kidhome",
                    "Teenhome",
                    "Recency",
                    "NumWebPurchases",
                    "NumCatalogPurchases",
                    "NumStorePurchases",
                ],
            }
        source = feature_sets  # 기본값 또는 사용자 입력 feature set
        return {name: list(features) for name, features in source.items()}  # 원본 수정 방지를 위해 리스트 복사본으로 반환

    @staticmethod
    def _validate_columns(df: pd.DataFrame, columns: Sequence[str]) -> None:
        missing_cols = [col for col in columns if col not in df.columns]  # 필요한 컬럼 중 데이터프레임에 없는 컬럼 확인
        if missing_cols:  # 누락된 컬럼이 있으면
            raise ValueError(f"Missing required columns: {missing_cols}")  # 어떤 컬럼이 없는지 에러로 알려줌


def main() -> None:
    recommender = MarketingItemRecommendationRegression()
    result = recommender.run_pipeline(verbose=True)

    print("\nFinal model selected:", result.final_result_key)
    print(recommender.get_banner_recommendation_output(result.final_predictions).head(10))

    sample_customer_id = result.final_predictions.index[0]
    print(f"\nSample customer index: {sample_customer_id}")
    print("Banner ad display order:")
    for rank, product in enumerate(
            result.final_predictions.loc[sample_customer_id, "Banner_Ad_Order"].split(" > "),
            start=1,
    ):
        print(f"{rank}. {product}")

    print("\nRecommendation score table:")
    print(recommender.get_customer_score_table(sample_customer_id, result.final_predictions))


if __name__ == "__main__":
    main()
