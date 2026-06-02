# GradientBoosting_pipeline_01_preprocessing.py
# -*- coding: utf-8 -*-
"""
[Pipeline 01] GradientBoosting 데이터 전처리

GradientBoosting1 버전은 전처리 단계에서 범주형 변수를 직접 원-핫 인코딩한 뒤,
모델 학습 파일로 숫자형 데이터만 넘깁니다.
"""
import os
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


SplitData = tuple[
    Optional[pd.DataFrame],
    Optional[pd.DataFrame],
    Optional[pd.Series],
    Optional[pd.Series],
]


def add_gradient_boosting_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    GradientBoosting 성능 개선에 사용할 파생변수를 추가합니다.

    Args:
        df (pd.DataFrame): Churn 컬럼을 포함한 원본 고객 데이터.

    Returns:
        pd.DataFrame: 요금 관련 파생변수와 구간 변수가 추가된 데이터.
    """
    df = df.copy()

    df["TotalCharges_Clipped"] = df["TotalCharges"].clip(lower=0)
    df["AvgMonthlyChargeFromTotal"] = (
        df["TotalCharges_Clipped"] / df["Tenure"].replace(0, np.nan)
    )
    df["ChargeGap"] = df["TotalCharges_Clipped"] - (
        df["MonthlyCharges"] * df["Tenure"]
    )
    df["MonthlyToAverageChargeRatio"] = (
        df["MonthlyCharges"] / df["AvgMonthlyChargeFromTotal"].replace(0, np.nan)
    )

    df["AgeGroup"] = pd.cut(
        df["Age"],
        bins=[17, 29, 44, 59, 80],
        labels=["18-29", "30-44", "45-59", "60+"],
        include_lowest=True,
    )
    df["TenureGroup"] = pd.cut(
        df["Tenure"],
        bins=[0, 12, 24, 48, 72],
        labels=["0-12", "13-24", "25-48", "49-72"],
        include_lowest=True,
    )

    numeric_columns = df.select_dtypes(include=[np.number]).columns
    df[numeric_columns] = df[numeric_columns].replace([np.inf, -np.inf], np.nan)
    df[numeric_columns] = df[numeric_columns].fillna(0)

    return df


def run_preprocessing(data_filename: str = "synthetic_customer_churn_100k.csv") -> SplitData:
    """
    CSV 파일을 읽고 GradientBoosting1 학습용 데이터를 반환합니다.

    Args:
        data_filename (str): 현재 파일과 같은 폴더에 있는 CSV 파일명.

    Returns:
        SplitData: X_train, X_test, y_train, y_test.
            오류가 발생하면 네 값을 모두 None으로 반환합니다.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, data_filename)

    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"[오류] 데이터 파일을 찾을 수 없습니다: {data_path}")
        return None, None, None, None
    except pd.errors.ParserError as error:
        print(f"[오류] CSV 파일을 읽는 중 문제가 발생했습니다: {error}")
        return None, None, None, None

    print(f"데이터 크기: {df.shape}")

    if "CustomerID" in df.columns:
        df = df.drop("CustomerID", axis=1)

    if "Churn" not in df.columns:
        print("[오류] 타깃 컬럼 'Churn'이 없습니다.")
        return None, None, None, None

    if not pd.api.types.is_numeric_dtype(df["Churn"]):
        df["Churn"] = df["Churn"].astype(str).str.strip().map({"Yes": 1, "No": 0})
        print("타깃 변수 Churn 변환: Yes -> 1, No -> 0")

    if df["Churn"].isna().any():
        print("[오류] Churn 컬럼에 변환할 수 없는 값이 있습니다.")
        return None, None, None, None

    df = add_gradient_boosting_features(df)
    print("파생변수 추가 완료")

    X = df.drop("Churn", axis=1)
    y = df["Churn"].astype(int)

    categorical_columns = X.select_dtypes(include=["object", "category"]).columns.tolist()
    if categorical_columns:
        X = pd.get_dummies(X, columns=categorical_columns, drop_first=False, dtype=int)
        print(f"범주형 변수 원-핫 인코딩 완료: {categorical_columns}")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    print(f"Train 데이터 크기: {X_train.shape}, Test 데이터 크기: {X_test.shape}")
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    run_preprocessing()
