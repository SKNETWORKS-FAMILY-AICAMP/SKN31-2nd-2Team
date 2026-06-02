# app.py
# -*- coding: utf-8 -*-
"""
고객 이탈 예측 시스템 - Streamlit 웹 대시보드 (app.py)
========================================================================
[실행 방법]
1. 필수 라이브러리 설치:
   pip install streamlit pandas numpy xgboost matplotlib seaborn scikit-learn
2. 터미널에서 실행: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import os

# XGBoost 파이프라인 모듈 임포트
try:
    import XGBoost_pipeline_01_preprocessing as pipe1
    import XGBoost_pipeline_02_training      as pipe2
    import XGBoost_pipeline_03_tuning        as pipe3
except ImportError:
    st.error("❌ 'XGBoost_pipeline_01~03.py' 파일이 같은 폴더에 있어야 합니다.")
    st.stop()

st.set_page_config(
    page_title="고객 이탈 예측 시스템",
    page_icon="🔮",
    layout="wide"
)

# -------------------------------------------------------------------------
# 사이드바 메뉴
# -------------------------------------------------------------------------
st.sidebar.title("🔮 Churn Dashboard")
st.sidebar.markdown("---")
page = st.sidebar.radio("모드를 선택하세요", ["👤 사용자 모드 (개별 이탈 예측)", "⚙️ 관리자 모드 (모델 학습/튜닝)"])
st.sidebar.markdown("---")
st.sidebar.info("💡 본 시스템은 XGBoost 모델 기반의 고객 이탈 예측 솔루션입니다.")

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'synthetic_customer_churn_100k.csv')


# -------------------------------------------------------------------------
# [1페이지] 사용자 모드: 개별 고객 정보 입력 → 실시간 이탈 예측
# -------------------------------------------------------------------------
if page == "👤 사용자 모드 (개별 이탈 예측)":
    st.title("👤 사용자 모드 (User Mode)")
    st.subheader("실시간 고객 이탈 위험도 예측")
    st.write("상담원이나 영업 사원이 개별 고객의 정보를 입력하여 이탈 가능성을 실시간으로 조회하는 화면입니다.")
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 📋 고객 기본 정보")
        age             = st.slider("고객 나이 (Age)", 18, 100, 40)
        gender          = st.selectbox("성별 (Gender)", ["Female", "Male", "Other"])
        tenure          = st.number_input("가입 기간 (Tenure, 개월 수)", min_value=0, max_value=120, value=12)

    with col2:
        st.markdown("### 💳 계약 및 요금 정보")
        contract        = st.selectbox("계약 유형 (Contract)", ["Month-to-month", "One year", "Two year"])
        payment_method  = st.selectbox("결제 수단 (PaymentMethod)", ["Electronic check", "Mailed check", "Bank transfer", "Credit card"])
        monthly_charges = st.number_input("월 청구 금액 (MonthlyCharges, $)", min_value=0.0, max_value=300.0, value=65.0)
        total_charges   = st.number_input("총 청구 금액 (TotalCharges, $)", min_value=0.0, max_value=30000.0, value=monthly_charges * tenure)

    st.markdown("---")

    if st.button("🔮 이탈 위험도 분석 실행", type="primary"):
        with st.spinner("🧠 모델 로딩 및 실시간 분석 중..."):

            # 전처리 파이프라인 실행 (processor 포함 5개 반환)
            X_train, X_test, y_train, y_test, processor = pipe1.run_preprocessing()

            if X_train is not None:
                # 튜닝 결과 파일이 있으면 최적 파라미터 사용, 없으면 기본값
                TUNING_FILE = os.path.join(BASE_DIR, 'xgb_tuning_top10_results.csv')
                best_params = None
                if os.path.exists(TUNING_FILE):
                    top10 = pd.read_csv(TUNING_FILE)
                    best_params = {
                        'n_estimators'    : 300,
                        'learning_rate'   : float(top10.iloc[0]['learning_rate']),
                        'max_depth'       : int(top10.iloc[0]['max_depth']),
                        'subsample'       : float(top10.iloc[0]['subsample']),
                        'colsample_bytree': float(top10.iloc[0]['colsample_bytree']),
                    }

                # XGBoost 모델 학습
                from xgboost import XGBClassifier
                scale_pos = (y_train == 0).sum() / (y_train == 1).sum()

                n_estimators     = best_params.get('n_estimators',     100) if best_params else 100
                max_depth        = best_params.get('max_depth',          6) if best_params else 6
                learning_rate    = best_params.get('learning_rate',    0.1) if best_params else 0.1
                subsample        = best_params.get('subsample',        0.8) if best_params else 0.8
                colsample_bytree = best_params.get('colsample_bytree', 0.8) if best_params else 0.8

                model = XGBClassifier(
                    n_estimators     = n_estimators,
                    max_depth        = max_depth,
                    learning_rate    = learning_rate,
                    subsample        = subsample,
                    colsample_bytree = colsample_bytree,
                    scale_pos_weight = scale_pos,
                    eval_metric      = 'logloss',
                    random_state     = 42,
                    n_jobs           = -1,
                    verbosity        = 0
                )
                model.fit(X_train, y_train)

                # 입력 데이터를 processor로 동일하게 전처리
                input_df = pd.DataFrame([{
                    'Age'           : age,
                    'Gender'        : gender,
                    'Tenure'        : tenure,
                    'MonthlyCharges': monthly_charges,
                    'Contract'      : contract,
                    'PaymentMethod' : payment_method,
                    'TotalCharges'  : total_charges
                }])
                input_processed = processor.transform(input_df)

                # 예측
                prob = model.predict_proba(input_processed)[0][1] * 100

                # 결과 출력
                st.markdown("## 📊 분석 결과")
                if prob >= 50:
                    st.error(f"🚨 이탈 위험군: 본 고객이 이탈할 확률이 **{prob:.0f}%** 로 매우 높습니다!")
                    st.progress(int(prob))
                    st.markdown("⚠️ **추천 조치:** 특별 할인 프로모션 제안, 장기 계약 전환 상담 유도 필요.")
                else:
                    st.success(f"✅ 안정 유지군: 본 고객이 이탈할 확률은 **{prob:.0f}%** 로 안정적입니다.")
                    st.progress(int(prob))


# -------------------------------------------------------------------------
# [2페이지] 관리자 모드: 데이터 확인 및 모델 튜닝/학습 제어
# -------------------------------------------------------------------------
elif page == "⚙️ 관리자 모드 (모델 학습/튜닝)":
    st.title("⚙️ 관리자 모드 (Admin Mode)")
    st.subheader("데이터 모니터링 및 예측 파이프라인 관리")
    st.write("원천 데이터를 확인하고 머신러닝 파이프라인(전처리 ➔ 튜닝 ➔ 학습)을 원클릭으로 가동하는 제어판입니다.")
    st.markdown("---")

    # 데이터셋 프리뷰
    st.markdown("### 📂 1. 데이터셋 프리뷰 (`synthetic_customer_churn_100k.csv`)")
    if os.path.exists(DATA_PATH):
        df_preview = pd.read_csv(DATA_PATH, nrows=5)
        st.dataframe(df_preview)
        st.success("✔ 현재 원천 데이터가 정상 인식되고 있습니다. (총 100,000개 행 보유)")
    else:
        st.error(f"❌ '{DATA_PATH}' 파일을 찾을 수 없습니다. 데이터셋을 같은 폴더에 배치해 주세요.")
        st.stop()

    st.markdown("---")

    # 파이프라인 제어 버튼
    st.markdown("### 🚀 2. 머신러닝 오토메이션 파이프라인 제어")
    col_btn1, col_btn2 = st.columns(2)

    with col_btn1:
        st.markdown("**[실험 단계] 하이퍼파라미터 그리드 서치**")
        st.write("배치된 조합을 순회 연산하여 최적의 파라미터 조합 Top 10을 추출합니다.")
        if st.button("🔥 튜닝 파이프라인 가동 (Pipeline 03)"):
            with st.spinner("🔄 파라미터 조합별 성능 점수 연산 중... (약 10~20초 소요)"):
                top_10_results = pipe3.run_parameter_tuning()
                st.write("🏆 **연산 완료! 최적 하이퍼파라미터 순위 테이블 (Top 10)**")
                st.dataframe(top_10_results)
                st.success("✔ 상위 10개 결과가 'xgb_tuning_top10_results.csv' 파일로 저장되었습니다.")

    with col_btn2:
        st.markdown("**[최종 배포] 최적 조합 기반 대규모 최종 학습**")
        st.write("튜닝 1위 파라미터를 가져와 최종 모델을 생성하고 평가 결과를 출력합니다.")
        if st.button("🚀 최종 트레이닝 가동 (Pipeline 02)", key="admin_btn2"):
            TUNING_FILE = os.path.join(BASE_DIR, 'xgb_tuning_top10_results.csv')

            if not os.path.exists(TUNING_FILE):
                st.warning("⚠️ 먼저 왼쪽의 '튜닝 파이프라인 가동' 버튼을 눌러 최적 파라미터를 선출해 주세요. (기본값으로 학습을 우선 시작합니다.)")
                best_params = None
            else:
                top10 = pd.read_csv(TUNING_FILE)
                best_params = {
                    'n_estimators'    : 300,
                    'learning_rate'   : float(top10.iloc[0]['learning_rate']),
                    'max_depth'       : int(top10.iloc[0]['max_depth']),
                    'subsample'       : float(top10.iloc[0]['subsample']),
                    'colsample_bytree': float(top10.iloc[0]['colsample_bytree']),
                }

            with st.spinner("🏋️ 10만 건 전체 데이터로 최종 모델 학습 중..."):
                # 전처리 (processor 포함 5개 반환)
                X_train, X_test, y_train, y_test, processor = pipe1.run_preprocessing()

                # 최종 학습 및 평가 (model 객체 반환)
                model = pipe2.run_model_training_and_evaluation(
                    X_train, X_test, y_train, y_test,
                    processor,
                    best_params=best_params
                )

                st.success("🎉 최종 학습 완료!")

                # 평가 결과 테이블 — model_comparison_results.csv 읽어서 출력
                RESULT_PATH = os.path.join(BASE_DIR, 'model_comparison_results.csv')
                if os.path.exists(RESULT_PATH):
                    st.markdown("### 📋 최종 모델 성능 평가 결과")
                    df_metrics = pd.read_csv(RESULT_PATH, index_col='Model')
                    st.dataframe(df_metrics, use_container_width=True)
                else:
                    st.error("❌ 평가 결과 파일을 찾을 수 없습니다.")

                st.markdown("---")

                # 피처 중요도 그래프
                IMG_PATH = os.path.join(BASE_DIR, 'xgb_feature_importance.png')
                if os.path.exists(IMG_PATH):
                    st.markdown("### 📊 모델 평가 결과 - 피처 중요도 (Feature Importance)")
                    st.image(IMG_PATH, caption="XGBoost가 분석한 고객 이탈 핵심 변수 순위")
                else:
                    st.warning("⚠️ 피처 중요도 그래프 파일이 없습니다. 학습을 먼저 실행해 주세요.")