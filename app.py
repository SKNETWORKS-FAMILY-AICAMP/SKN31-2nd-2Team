# app.py
# -*- coding: utf-8 -*-
"""
고객 이탈 예측 시스템 - Streamlit 웹 대시보드 (상단 레이아웃 버전)

"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

# LightGBM 기존 모듈 불러오기
try:
    from models.LightGBM import LightGBM_01_preprocessing as pipe1
    from models.LightGBM import LightGBM_02_training as pipe2
    from models.LightGBM import LightGBM_03_tuning as pipe3
except ImportError:
    st.error("❌ 'LightGBM_01_preprocessing.py', 'LightGBM_02_training.py', 'LightGBM_03_tuning.py' 파일을 찾을 수 없습니다.")
    st.stop()

# 스트림릿 페이지 설정
st.set_page_config(
    page_title="고객 이탈 예측 시스템",
    page_icon="🔮",
    layout="wide"
)

# 데이터 세트 경로 정의
DATA_PATH = 'data/synthetic_customer_churn_100k.csv'

# -------------------------------------------------------------------------
# [수정] 최상단 대시보드 타이틀 및 모드 선택 가로 배치
# -------------------------------------------------------------------------
st.image("main_image.png", use_container_width=True)

st.title("🔮 Telco Churn Dashboard")
st.caption("본 시스템은 LightGBM 모델 기반의 고객 이탈 예측 솔루션입니다.")

# 대용량 메뉴 셀렉트박스를 상단에 넓게 배치
page = st.selectbox(
    "모드를 선택하세요", 
    ["👤 사용자 모드 (개별 이탈 예측)", "⚙️ 관리자 모드 (모델 학습/튜닝)"]
)
st.markdown("---")


# -------------------------------------------------------------------------
# [1페이지] 사용자 모드: 개별 고객의 정보를 입력받아 이탈을 실시간 예측
# -------------------------------------------------------------------------
if page == "👤 사용자 모드 (개별 이탈 예측)":
    st.subheader("👤 실시간 고객 이탈 위험도 예측")
    st.write("상담사를 위한 고객 이탈 가능성을 실시간으로 조회하는 화면입니다.")
    st.write("")
    
    # [수정] 입력 폼을 4개의 열(Column)로 나누어 상단에 가로로 길게 배치
    st.markdown("#### 📋 고객 정보 입력 (가로 입력판)")
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    
    with m_col1:
        age = st.slider("고객 나이 (Age)", 18, 100, 40)
        gender = st.selectbox("성별 (Gender)", ["Female", "Male", "Other"])
        
    with m_col2:
        tenure = st.number_input("가입 기간 (Tenure, 개월 수)", min_value=0, max_value=120, value=12)
        contract = st.selectbox("계약 유형 (Contract)", ["Month-to-month", "One year", "Two year"])
        
    with m_col3:
        payment_method = st.selectbox("결제 수단 (PaymentMethod)", ["Electronic check", "Mailed check", "Bank transfer", "Credit card"])
        monthly_charges = st.number_input("월 청구 금액 (MonthlyCharges, $)", min_value=0.0, max_value=300.0, value=65.0)
        
    with m_col4:
        # 총 금액은 월 금액 * 가입기간으로 기본값 계산
        total_charges = st.number_input("총 청구 금액 (TotalCharges, $)", min_value=0.0, max_value=30000.0, value=monthly_charges * tenure)

    st.markdown("---")
    
    # 예측 버튼 클릭 이벤트
    if st.button("🔮 이탈 위험도 분석 실행", type="primary"):
        TUNING_FILE = 'lgb_tuning_top10_results.csv'
        
        with st.spinner("🧠 연동 모델로 실시간 분석 중..."):
            input_data = pd.DataFrame([{
                'Age': age, 'Gender': gender, 'Tenure': tenure,
                'MonthlyCharges': monthly_charges, 'Contract': contract,
                'PaymentMethod': payment_method, 'TotalCharges': total_charges
            }])
            
            X_train, X_test, y_train, y_test = pipe1.run_preprocessing()
            
            if X_train is not None:
                best_params = None
                if os.path.exists(TUNING_FILE):
                    top10 = pd.read_csv(TUNING_FILE)
                    best_params = {
                        'learning_rate': float(top10.iloc[0]['learning_rate']),
                        'num_leaves': int(top10.iloc[0]['num_leaves']),
                        'max_depth': int(top10.iloc[0]['max_depth'])
                    }
                
                import lightgbm as lgb
                lr = best_params['learning_rate'] if best_params else 0.05
                num_leaves = best_params['num_leaves'] if best_params else 31
                max_depth = best_params['max_depth'] if best_params else -1
                
                for col in ['Gender', 'Contract', 'PaymentMethod']:
                    input_data[col] = input_data[col].astype('category')
                
                model = lgb.LGBMClassifier(n_estimators=100, learning_rate=lr, num_leaves=num_leaves, max_depth=max_depth, random_state=42, class_weight='balanced', verbose=-1)
                model.fit(X_train, y_train)
                
                prob = model.predict_proba(input_data)[0][1] * 100
                
                st.markdown("## 📊 분석 결과")
                if prob >= 50:
                    st.error(f"🚨 이탈 위험군: 본 고객이 이탈할 확률이 {prob:.0f}% 로 매우 높습니다!")
                    st.progress(int(prob))
                    st.markdown("⚠️ **추천 조치:** 특별 할인 프로모션 제안, 장기 계약 전환 상담 유도 필요.")
                else:
                    st.success(f"✅ 안정 유지군: 본 고객이 이탈할 확률은 {prob:.0f}% 로 안정적입니다.")
                    st.progress(int(prob))


# -------------------------------------------------------------------------
# [2페이지] 관리자 모드: 데이터 확인 및 모델 튜닝/트레이닝 제어
# -------------------------------------------------------------------------
elif page == "⚙️ 관리자 모드 (모델 학습/튜닝)":
    st.subheader("⚙️ 데이터 모니터링 및 예측 모델 관리")
    st.write("원본 데이터를 확인하고 머신러닝(전처리 ➔ 튜닝 ➔ 학습)을 원클릭으로 가동하는 제어판입니다.")
    st.markdown("---")
    
    # 2-1. 원본 데이터프레임 확인
    st.markdown("### 📂 1. 데이터셋 프리뷰 (`synthetic_customer_churn_100k.csv`)")
    if os.path.exists(DATA_PATH):
        df_preview = pd.read_csv(DATA_PATH, nrows=5)
        st.dataframe(df_preview, use_container_width=True)
        st.success(f"✔ 현재 원본 데이터가 정상 인식되고 있습니다. (총 100,000개 행 보유)")
    else:
        st.error(f"❌ '{DATA_PATH}' 파일을 찾을 수 없습니다. 데이터셋을 같은 폴더에 배치해 주세요.")
        st.stop()
        
    st.markdown("---")
    
    # 2-2. 모델 제어
    st.markdown("### 🚀 2. LightGBM 모델 제어")
    
    col_btn1, col_btn2 = st.columns(2)
    
    with col_btn1:
        st.markdown("**[실험 단계] 하이퍼파라미터 그리드 서치**")
        st.write("배치된 조합을 순회 연산하여 최적의 파라미터 조합 Top 10을 추출합니다.")
        if st.button("🔥 튜닝 모듈 가동", use_container_width=True):
            with st.spinner("🔄 여러 개의 모델을 생성하며 조합별 성능 점수를 연산 중입니다... (약 10~20초 소요)"):
                top_10_results = pipe3.run_parameter_tuning()
                st.write("🏆 **연산 완료! 최적 하이퍼파라미터 순위 테이블 (Top 10)**")
                st.dataframe(top_10_results, use_container_width=True)
                st.success("✔ 상위 10개 모델 테이블이 'lgb_tuning_top10_results.csv' 파일로 자동 업데이트되었습니다.")
                
    with col_btn2:
        st.markdown("**[최종 학습] 대규모 최종 학습**")
        st.write("튜닝 1위 파라미터를 이용하여 500개 트리와 Early Stopping 기반의 최종 모델을 생성합니다.")
        if st.button("🚀 최종 트레이닝 모듈 가동", key="admin_btn2", use_container_width=True):
            TUNING_FILE = 'models/LightGBM/saved_data/lgb_tuning_top10_results.csv'
            if not os.path.exists(TUNING_FILE):
                st.warning("⚠️ 먼저 왼쪽의 '튜닝 모듈 가동' 버튼을 눌러 최적의 파라미터를 선출해 주세요. (기본값으로 학습을 우선 시작합니다.)")
                best_params = None
            else:
                top10 = pd.read_csv(TUNING_FILE)
                best_params = {
                    'learning_rate': float(top10.iloc[0]['learning_rate']),
                    'num_leaves': int(top10.iloc[0]['num_leaves']),
                    'max_depth': int(top10.iloc[0]['max_depth'])
                }
                
            with st.spinner("🏋️‍♂️ 10만 건 전체 데이터를 이용하여 최종 예측 모델 학습 및 평가 지표 산출 중..."):
                X_train, X_test, y_train, y_test = pipe1.run_preprocessing()
                metrics = pipe2.run_model_training_and_evaluation(X_train, X_test, y_train, y_test, best_params=best_params)
                
                st.success("🎉 최종 학습 완수!")
                
                st.markdown("### 📋 최종 모델 성능 평가 결과")
                st.markdown("⚙️ **LightGBM 최적 파라미터 (튜닝 1순위)**")
                if best_params:
                    p_col1, p_col2, p_col3 = st.columns(3)
                    p_col1.markdown(f"**Learning Rate:** `{best_params['learning_rate']}`")
                    p_col2.markdown(f"**Num Leaves:** `{best_params['num_leaves']}`")
                    p_col3.markdown(f"**Max Depth:** `{best_params['max_depth']}`")
                else:
                    st.info("💡 튜닝 결과 파일이 없어 기본 파라미터로 실행되었습니다.")
                
                st.write("")
                
                if metrics and isinstance(metrics, dict):
                    df_metrics = pd.DataFrame(metrics)
                    st.dataframe(df_metrics, hide_index=True, use_container_width=True)
                else:
                    st.error("❌ 모델 평가 지표를 불러오지 못했습니다. 리턴값을 확인해 주세요.")
                
                st.markdown("---")
                
                IMG_PATH = 'models/LightGBM/saved_data/lgb_feature_importance.png'
                if os.path.exists(IMG_PATH):
                    st.markdown("### 📊 모델 평가 결과 - 피쳐 중요도 (Feature Importance)")
                    st.image(IMG_PATH, caption="LightGBM이 분석한 고객 이탈 핵심 변수 순위")