# GradientBoosting_pipeline_02_training.py
# -*- coding: utf-8 -*-
"""
[Pipeline 02] GradientBoosting 모델 학습 및 평가
"""

from __future__ import annotations

import os
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.utils.class_weight import compute_sample_weight


Metrics = Optional[dict[str, list[str]]]


def run_model_training_and_evaluation(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    best_params: Optional[dict[str, float | int]] = None,
) -> Metrics:
    """
    GradientBoosting 모델을 학습하고 성능 지표를 계산합니다.

    Args:
        X_train (pd.DataFrame): 학습용 피처 데이터.
        X_test (pd.DataFrame): 검증용 피처 데이터.
        y_train (pd.Series): 학습용 정답 데이터.
        y_test (pd.Series): 검증용 정답 데이터.
        best_params (dict[str, float | int] | None): 튜닝으로 찾은 모델 파라미터.

    Returns:
        Metrics: Train/Test 평가 지표 딕셔너리. 입력 오류가 있으면 None.
    """
    if X_train is None:
        print("[오류] 입력 데이터가 올바르지 않습니다.")
        return None

    params = best_params or {}
    model = GradientBoostingClassifier(
        learning_rate=float(params.get("learning_rate", 0.05)),
        n_estimators=int(params.get("n_estimators", 150)),
        max_depth=int(params.get("max_depth", 3)),
        random_state=42,
    )

    print(f"최종 모델 파라미터: {model.get_params()}")

    sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)
    model.fit(X_train, y_train, sample_weight=sample_weight)
    print("GradientBoosting 모델 학습 완료")

    y_train_pred = model.predict(X_train)
    y_train_proba = model.predict_proba(X_train)[:, 1]
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    print("\n------------------ [ GradientBoosting 모델 평가 결과 ] ------------------")
    print(f"1. 정확도(Accuracy): {accuracy_score(y_test, y_pred):.4f}")
    print(f"2. ROC-AUC 점수    : {roc_auc_score(y_test, y_pred_proba):.4f}")
    print("\n3. 분류 리포트(Classification Report):")
    print(classification_report(y_test, y_pred))
    print("-----------------------------------------------------------------------")

    importance_df = pd.DataFrame(
        {"Feature": X_train.columns, "Importance": model.feature_importances_}
    ).sort_values(by="Importance", ascending=False)

    sns.set_theme(style="whitegrid")
    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False

    plt.figure(figsize=(10, 6))
    sns.barplot(
        x="Importance",
        y="Feature",
        data=importance_df.head(20),
        palette="viridis",
        hue="Feature",
        legend=False,
    )
    plt.title("GradientBoosting Feature Importance for Churn Prediction", fontsize=14)
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()

    output_image = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gb_feature_importance.png")
    plt.savefig(output_image, dpi=150)
    plt.close()
    print(f"Feature Importance 그래프 저장 완료: {output_image}")

    return {
        "평가 지표(Metric)": ["정확도(Accuracy)", "정밀도(Precision)", "재현율(Recall)", "F1-Score", "ROC-AUC"],
        "훈련 데이터 성능 (Train)": [
            f"{accuracy_score(y_train, y_train_pred):.4f}",
            f"{precision_score(y_train, y_train_pred):.4f}",
            f"{recall_score(y_train, y_train_pred):.4f}",
            f"{f1_score(y_train, y_train_pred):.4f}",
            f"{roc_auc_score(y_train, y_train_proba):.4f}",
        ],
        "검증 데이터 성능 (Test)": [
            f"{accuracy_score(y_test, y_pred):.4f}",
            f"{precision_score(y_test, y_pred):.4f}",
            f"{recall_score(y_test, y_pred):.4f}",
            f"{f1_score(y_test, y_pred):.4f}",
            f"{roc_auc_score(y_test, y_pred_proba):.4f}",
        ],
    }


if __name__ == "__main__":
    try:
        import GradientBoosting_pipeline_01_preprocessing as p1

        split_data = p1.run_preprocessing()
        if split_data[0] is not None:
            run_model_training_and_evaluation(*split_data)
    except ImportError as error:
        print(f"[오류] 전처리 모듈을 불러올 수 없습니다: {error}")
