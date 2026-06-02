# GradientBoosting_pipeline_03_tuning.py
# -*- coding: utf-8 -*-
"""
[Pipeline 03] GradientBoosting 하이퍼파라미터 튜닝
"""

from __future__ import annotations

import os
import time
from typing import Optional

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.utils.class_weight import compute_sample_weight

import GradientBoosting_pipeline_01_preprocessing as p1


def run_parameter_tuning() -> Optional[pd.DataFrame]:
    """
    여러 GradientBoosting 파라미터 조합을 평가합니다.

    Args:
        없음.

    Returns:
        pd.DataFrame | None: ROC-AUC 기준 상위 10개 튜닝 결과.
            전처리에 실패하면 None을 반환합니다.
    """
    tuning_start_time = time.perf_counter()
    X_train, X_test, y_train, y_test = p1.run_preprocessing()
    if X_train is None:
        return None

    learning_rates = [0.03, 0.05, 0.1]
    n_estimators_list = [80, 120, 160]
    max_depths = [2, 3]

    X_train_sample = X_train.sample(n=min(15000, len(X_train)), random_state=42)
    y_train_sample = y_train.loc[X_train_sample.index]
    sample_weight = compute_sample_weight(class_weight="balanced", y=y_train_sample)

    results: list[dict[str, float | int]] = []
    trial_count = 1

    for learning_rate in learning_rates:
        for n_estimators in n_estimators_list:
            for max_depth in max_depths:
                trial_start_time = time.perf_counter()
                print(
                    f"[Trial {trial_count:02d}] 검증 진행 중 -> "
                    f"learning_rate: {learning_rate}, "
                    f"n_estimators: {n_estimators}, "
                    f"max_depth: {max_depth}"
                )

                model = GradientBoostingClassifier(
                    learning_rate=learning_rate,
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    random_state=42,
                )
                model.fit(X_train_sample, y_train_sample, sample_weight=sample_weight)

                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                elapsed_seconds = time.perf_counter() - trial_start_time

                results.append(
                    {
                        "Trial": trial_count,
                        "learning_rate": learning_rate,
                        "n_estimators": n_estimators,
                        "max_depth": max_depth,
                        "Accuracy": round(accuracy_score(y_test, y_pred), 4),
                        "ROC-AUC": round(roc_auc_score(y_test, y_pred_proba), 4),
                        "ElapsedSeconds": round(elapsed_seconds, 2),
                    }
                )
                trial_count += 1

    top_10_results = (
        pd.DataFrame(results)
        .sort_values(by=["ROC-AUC", "Accuracy"], ascending=False)
        .head(10)
        .reset_index(drop=True)
    )
    top_10_results.index = top_10_results.index + 1
    top_10_results.index.name = "Rank"

    print("-------------------------------------------------------------------------")
    print("GradientBoosting 하이퍼파라미터 튜닝 상위 10개 결과")
    print("-------------------------------------------------------------------------")
    print(top_10_results.to_string())
    print("-------------------------------------------------------------------------")

    output_csv = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gb_tuning_top10_results.csv")
    top_10_results.to_csv(output_csv, encoding="utf-8-sig")
    print(f"튜닝 상위 10개 결과 저장 완료: {output_csv}")
    print(f"전체 하이퍼파라미터 튜닝 소요 시간: {time.perf_counter() - tuning_start_time:.2f}초")

    return top_10_results


if __name__ == "__main__":
    run_parameter_tuning()
