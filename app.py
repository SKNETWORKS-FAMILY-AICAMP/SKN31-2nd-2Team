# app.py
# -*- coding: utf-8 -*-
"""
고객 이탈 예측 시스템 - Streamlit 웹 대시보드 (커스텀 디자인 적용)
========================================================================
[실행 방법]
1. pip install streamlit pandas numpy lightgbm matplotlib seaborn scikit-learn
2. streamlit run app.py

[로고 이미지]
- 프로젝트 루트에 logo.png 파일을 배치하면 헤더에 자동으로 표시됩니다.
- 없으면 텍스트 로고로 대체됩니다.
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import base64
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# ── LightGBM 파이프라인 모듈 import ─────────────────────────────────────────
try:
    from models.LightGBM import LightGBM_01_preprocessing as pipe1
    from models.LightGBM import LightGBM_02_training      as pipe2
    from models.LightGBM import LightGBM_03_tuning        as pipe3
except ImportError:
    st.error("❌ models/LightGBM/ 하위의 파이프라인 파일을 찾을 수 없습니다.")
    st.stop()

# ── 경로 상수 ────────────────────────────────────────────────────────────────
DATA_PATH    = "data/synthetic_customer_churn_100k.csv"
TUNING_FILE  = "models/04_LightGBM/saved_data/lgb_tuning_top10_results.csv"
IMG_PATH     = "models/04_LightGBM/saved_data/lgb_feature_importance.png"
LOGO_PATH    = "logo.png"   # ← 로고 파일을 여기에 배치하세요

# ── 페이지 기본 설정 ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="고객 이탈 예측 시스템",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ═══════════════════════════════════════════════════════════════════════════════
#  전역 CSS 주입 (네이비 컬러 시스템 + 커스텀 컴포넌트)
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
/* ── 구글 폰트 ── */
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@400;500;700;900&display=swap');

/* ── 전역 변수 & 리셋 ── */
:root {
    --navy:        #1a1f5e;
    --navy-mid:    #2a3080;
    --blue-acc:    #3b5bdb;
    --blue-light:  #e8eeff;
    --red-acc:     #e03131;
    --red-light:   #fff0f0;
    --green-acc:   #2f9e44;
    --green-light: #ebfbee;
    --gray-50:     #f8f9fa;
    --gray-100:    #f1f3f5;
    --gray-200:    #e9ecef;
    --gray-600:    #868e96;
    --gray-800:    #343a40;
    --white:       #ffffff;
    --radius:      10px;
    --radius-lg:   16px;
    --shadow:      0 2px 12px rgba(26,31,94,0.08);
}

html, body, [class*="css"] {
    font-family: 'Noto Sans KR', sans-serif !important;
}

/* ── Streamlit 기본 패딩/여백 제거 ── */
.block-container {
    padding-top: 0 !important;
    padding-left: 3rem !important;
    padding-right:  3rem !important;
    max-width: 100% !important;
}
header[data-testid="stHeader"] { display: none !important; }
#MainMenu, footer { display: none !important; }
section[data-testid="stSidebar"] { display: none !important; }

/* ── 헤더 ── */
.custom-header {
    background: var(--white);
    border-bottom: 2.5px solid var(--navy);
    padding: 0 40px;
    height: 110px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    position: sticky;
    top: 0;
    z-index: 999;
    box-shadow: 0 2px 8px rgba(26,31,94,0.07);
}
.header-logo-area { display: flex; align-items: center; gap: 14px; }
.header-logo-box {
    width: 200px; height:100px;
    background: var(--white);
    border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    overflow: hidden; flex-shrink: 0;
}
.header-logo-box img { width: 100%; height: 100%; object-fit: contain; }
.header-logo-text { color: white; font-size: 17px; font-weight: 900; letter-spacing: -1px; }
.header-title-wrap { display: flex; flex-direction: column; gap: 1px; }
.header-title-wrap strong { font-size: 17px; font-weight: 900; color: var(--navy); letter-spacing: -0.5px; }
.header-title-wrap span  { font-size: 11px; color: var(--gray-600); }
.header-badges { display: flex; gap: 10px; }
.header-badge {
    font-size: 12px; color: var(--gray-600); font-weight: 500;
    padding: 4px 12px; border: 1px solid var(--gray-200);
    border-radius: 20px; background: var(--gray-50);
}

/* ── 탭 네비게이션 ── */
.custom-nav {
    background: var(--white);
    border-bottom: 1px solid var(--gray-200);
    padding: 0 40px;
    display: flex; gap: 0;
}
.nav-tab-item {
    padding: 14px 28px;
    font-size: 14px; font-weight: 700;
    color: var(--gray-600);
    cursor: pointer;
    border-bottom: 3px solid transparent;
    letter-spacing: -0.3px;
    transition: color 0.15s;
    text-decoration: none;
}
.nav-tab-item.active {
    color: var(--navy);
    border-bottom-color: var(--navy);
}

/* ── 히어로 배너 ── */
.hero-banner {
    background: linear-gradient(135deg, #1a1f5e 0%, #2a3080 55%, #3b5bdb 100%);
    padding: 100px 48px;
    color: white;
    margin-left:-48px;
    margin-right: -48px;      
}
.hero-inner { display: flex; align-items: flex-end; justify-content: space-between; width: 100%; max-width: 1200px; }
.hero-h1 { font-size: 36px; font-weight: 900; line-height: 1.2; letter-spacing: -1.5px; margin-bottom: 10px; }
.hero-sub { font-size: 15px; opacity: 0.75; font-weight: 400; }
.hero-stats { display: flex; gap: 16px; }
.hero-stat-box {
    background: rgba(255,255,255,0.15);
    border: 1px solid rgba(255,255,255,0.25);
    border-radius: 10px;
    padding: 14px 22px;
    text-align: center;
}
.hero-stat-num  { font-size: 26px; font-weight: 900; letter-spacing: -1px; }
.hero-stat-lbl  { font-size: 11px; opacity: 0.7; margin-top: 2px; }

/* ── 섹션 타이틀 ── */
.section-title {
    font-size: 17px; font-weight: 900; color: var(--navy);
    letter-spacing: -0.5px;
    display: flex; align-items: center; gap: 10px;
    margin-bottom: 16px; margin-top: 8px;
}
.section-title::before {
    content: ''; display: block;
    width: 4px; height: 20px;
    background: var(--navy); border-radius: 2px;
}

/* ── 카드 ── */
.custom-card {
    background: var(--white);
    border-radius: var(--radius-lg);
    border: 1px solid var(--gray-200);
    padding: 28px 32px;
    box-shadow: var(--shadow);
    margin-bottom: 20px;
}

/* ── 결과 패널 ── */
.result-prob-high { font-size: 44px; font-weight: 900; letter-spacing: -2px; color: var(--red-acc); }
.result-prob-low  { font-size: 44px; font-weight: 900; letter-spacing: -2px; color: var(--green-acc); }
.risk-badge-high {
    display: inline-block; padding: 7px 20px; border-radius: 8px;
    background: var(--red-light); color: var(--red-acc);
    border: 1.5px solid #ffc9c9; font-size: 14px; font-weight: 900;
}
.risk-badge-low {
    display: inline-block; padding: 7px 20px; border-radius: 8px;
    background: var(--green-light); color: var(--green-acc);
    border: 1.5px solid #b2f2bb; font-size: 14px; font-weight: 900;
}
.result-tip-high {
    padding: 14px 18px; border-radius: 10px;
    background: var(--red-light); color: #c92a2a;
    border-left: 4px solid var(--red-acc);
    font-size: 14px; line-height: 1.6; font-weight: 500;
    margin-top: 12px;
}
.result-tip-low {
    padding: 14px 18px; border-radius: 10px;
    background: var(--green-light); color: #2b8a3e;
    border-left: 4px solid var(--green-acc);
    font-size: 14px; line-height: 1.6; font-weight: 500;
    margin-top: 12px;
}

/* ── 스탯 카드 (관리자) ── */
.stat-card {
    background: var(--white);
    border-radius: var(--radius-lg);
    border: 1px solid var(--gray-200);
    padding: 22px 24px;
    box-shadow: var(--shadow);
}
.stat-card .sc-label { font-size: 11px; color: var(--gray-600); font-weight: 700; letter-spacing: 0.5px; text-transform: uppercase; margin-bottom: 6px; }
.stat-card .sc-value { font-size: 26px; font-weight: 900; color: var(--navy); letter-spacing: -1.5px; }
.stat-card .sc-delta { font-size: 11px; color: var(--green-acc); margin-top: 4px; font-weight: 500; }

/* ── 어드민 컨트롤 카드 ── */
.admin-ctrl-card {
    background: var(--white);
    border-radius: var(--radius-lg);
    border: 1px solid var(--gray-200);
    padding: 26px 28px;
    box-shadow: var(--shadow);
    height: 100%;
}
.admin-ctrl-title { font-size: 15px; font-weight: 900; color: var(--navy); letter-spacing: -0.3px; margin-bottom: 8px; }
.admin-ctrl-desc  { font-size: 13px; color: var(--gray-600); line-height: 1.6; margin-bottom: 0; }

/* ── 버튼 오버라이드 ── */
div[data-testid="stButton"] > button {
    background: var(--navy) !important;
    color: white !important;
    border: none !important;
    border-radius: var(--radius) !important;
    font-weight: 900 !important;
    font-size: 15px !important;
    letter-spacing: -0.3px !important;
    padding: 14px 24px !important;
    width: auto !important;
    transition: background 0.15s !important;
    font-family: 'Noto Sans KR', sans-serif !important;
}
div[data-testid="stButton"] > button:hover {
    background: var(--navy-mid) !important;
}

/* ── 입력 필드 ── */
div[data-testid="stNumberInput"] input,
div[data-testid="stSelectbox"] > div > div {
    border-radius: var(--radius) !important;
    border: 1.5px solid var(--gray-200) !important;
}

/* ── 슬라이더 ── */
div[data-testid="stSlider"] > div > div > div > div {
    background: var(--navy) !important;
}

/* ── 구분선 ── */
.custom-divider { height: 1px; background: var(--gray-200); margin: 28px 0; }

/* ── 콘텐츠 래퍼 ── */
.content-wrap { padding: 36px 40px; max-width: 1400px; margin: 0 auto; }

/* ── 파라미터 뱃지 ── */
.param-badge {
    display: inline-block;
    background: #eef0ff; color: var(--navy);
    border-radius: 8px; padding: 8px 18px;
    font-size: 14px; font-weight: 700;
    margin: 4px;
}

/*─────────────────────────────────────────────────────────────────────
 하단 고정 상담 배너                                     
 position: fixed / bottom: 0 → 항상 화면 하단에 고정              
 보라색 그라디언트 배경 (#3D1FC2 → #4A28D4)                      
 .bottom-banner-btn : 반투명 둥근 버튼 (채팅/전화 2개)            
───────────────────────────────────────────────────────────────────── */
.bottom-banner {
    position: fixed;
    bottom: 0;
    left: 0;
    right: 0;
    z-index: 1000;
    background: linear-gradient(90deg, #3D1FC2 0%, #4A28D4 50%, #3A1AB8 100%);
    padding: 0.85rem 2.5rem;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 1.5rem;
    box-shadow: 0 -4px 24px rgba(61,31,194,0.35);
}
.bottom-banner-text {
    font-size: 1rem;
    font-weight: 700;
    color: #FFFFFF;
    letter-spacing: -0.01em;
    white-space: nowrap;
}
.bottom-banner-btns {
    display: flex;
    gap: 10px;
}
.bottom-banner-btn {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(255,255,255,0.18);
    border: 1.5px solid rgba(255,255,255,0.45);
    color: #FFFFFF !important;
    font-size: 0.875rem;
    font-weight: 600;
    padding: 0.5rem 1.1rem;
    border-radius: 22px;
    cursor: pointer;
    text-decoration: none  !important;
    white-space: nowrap;
    transition: background 0.2s, border-color 0.2s;
    backdrop-filter: blur(6px);
}
.bottom-banner-btn:hover {
    background: rgba(255,255,255,0.28);
    border-color: rgba(255,255,255,0.7);
}

/* ─────────────────────────────────────────────────────────────────────
    하단 배너 가림 방지                                     
  - 페이지 맨 아래 콘텐츠가 고정 배너에 가리지 않도록 여백 추가      
   ───────────────────────────────────────────────────────────────────── */
.content-wrap {
    padding-bottom: 80px !important;
}

</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  유틸: 로고 이미지 → base64
# ═══════════════════════════════════════════════════════════════════════════════
def get_logo_html() -> str:
    """logo.png 가 있으면 <img> 태그, 없으면 텍스트 로고를 반환합니다."""
    if os.path.exists(LOGO_PATH):
        with open(LOGO_PATH, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        ext = LOGO_PATH.rsplit(".", 1)[-1].lower()
        mime = "image/svg+xml" if ext == "svg" else f"image/{ext}"
        return f'<img src="data:{mime};base64,{b64}" alt="Logo">'
    return '<span class="header-logo-text">AI</span>'


# ═══════════════════════════════════════════════════════════════════════════════
#  공통 헤더 렌더링
# ═══════════════════════════════════════════════════════════════════════════════
def render_header():
    logo_html = get_logo_html()
    st.markdown(f"""
    <div class="custom-header">
        <div class="header-logo-area">
            <div class="header-logo-box">{logo_html}</div>
            <div class="header-title-wrap">
                <div style="height:30px;"></div>
                <span>고객 이탈 예측 시스템</span>
            </div>
        </div>
        <div class="header-badges">
            <span class="header-badge">LightGBM v3.x</span>
            <span class="header-badge">데이터: 100,000건</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  탭 네비게이션 (session_state 기반 페이지 전환)
# ═══════════════════════════════════════════════════════════════════════════════
if "page" not in st.session_state:
    st.session_state.page = "user"

render_header()

st.markdown("<div style='height:50px'></div>", unsafe_allow_html=True)

col_nav1, col_nav2, *_ = st.columns([1, 1, 8])
with col_nav1:
    if st.button("사용자 모드", key="nav_user"):
        st.session_state.page = "user"
with col_nav2:
    if st.button("관리자 모드", key="nav_admin"):
        st.session_state.page = "admin"

#탭 하이라이트 CSS (active 탭 표시)
# active_user  = "border-bottom: 3px solid #1a1f5e; color: #1a1f5e;" if st.session_state.page == "user"  else ""
# active_admin = "border-bottom: 3px solid #1a1f5e; color: #1a1f5e;" if st.session_state.page == "admin" else ""

# st.markdown(f"""
# <style>
# div[data-testid="stButton"]:nth-of-type(1) > button {{
#     background: transparent !important;
#     color: {'#1a1f5e' if st.session_state.page == 'user' else '#868e96'} !important;
#     border: none !important;
#     border-bottom: {'3px solid #1a1f5e' if st.session_state.page == 'user' else '3px solid transparent'} !important;
#     border-radius: 0 !important;
#     padding: 14px 20px !important;
#     font-size: 14px !important;
#     width: auto !important;
#     box-shadow: none !important;
# }}
# div[data-testid="stButton"]:nth-of-type(2) > button {{
#     background: transparent !important;
#     color: {'#1a1f5e' if st.session_state.page == 'admin' else '#868e96'} !important;
#     border: none !important;
#     border-bottom: {'3px solid #1a1f5e' if st.session_state.page == 'admin' else '3px solid transparent'} !important;
#     border-radius: 0 !important;
#     padding: 14px 20px !important;
#     font-size: 14px !important;
#     width: auto !important;
#     box-shadow: none !important;
# }}
# </style>
# """, unsafe_allow_html=True)

st.markdown('<div style="border-bottom:1px solid #e9ecef;margin-bottom:0"></div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE 1 ── 사용자 모드
# ═══════════════════════════════════════════════════════════════════════════════
if st.session_state.page == "user":

    # 히어로 배너
    st.markdown("""
    <div class="hero-banner">
      <div class="hero-inner">
        <div>
          <div class="hero-h1">실시간 고객 이탈<br>위험도 예측</div>
          <div class="hero-sub">상담사를 위한 고객 이탈 가능성 조회 시스템 · LightGBM 기반</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # 본문
    st.markdown('<div class="content-wrap">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">고객 정보 입력</div>', unsafe_allow_html=True)
    
    # st.markdown('<div class="custom-card">', unsafe_allow_html=True)

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown("**📋 기본 정보**")
        age     = st.slider("고객 나이 (Age)", 18, 100, 40)
        gender  = st.selectbox("성별 (Gender)", ["Female", "Male", "Other"])
        tenure  = st.number_input("가입 기간 (Tenure, 개월)", min_value=0, max_value=120, value=12)

    with col2:
        st.markdown("**💳 계약 및 요금 정보**")
        contract       = st.selectbox("계약 유형 (Contract)", ["Month-to-month", "One year", "Two year"])
        payment_method = st.selectbox("결제 수단 (PaymentMethod)",
                                      ["Electronic check", "Mailed check", "Bank transfer", "Credit card"])
        monthly_charges = st.number_input("월 청구 금액 (MonthlyCharges, $)",
                                          min_value=0.0, max_value=300.0, value=65.0)
        total_charges   = st.number_input("총 청구 금액 (TotalCharges, $)",
                                          min_value=0.0, max_value=30000.0, value=monthly_charges * tenure)

    # st.markdown('</div >', unsafe_allow_html=True)  # .custom-card
    # ── 예측 버튼 ──────────────────────────────────────────────────────────────
    run_btn = st.button("🔮  이탈 위험도 분석 실행", key="predict_btn", type="primary")

    if run_btn:
        with st.spinner("🧠 연동 모델로 실시간 분석 중..."):
            # 입력 데이터를 모델이 읽을 수 있는 데이터프레임 구조로 변환
            input_data = pd.DataFrame([{
                "Age": age, "Gender": gender, "Tenure": tenure,
                "MonthlyCharges": monthly_charges, "Contract": contract,
                "PaymentMethod": payment_method, "TotalCharges": total_charges
            }])
            # 전처리 모듈을 통해 기존 10만 건 데이터를 읽어와 구조 맞추기
            X_train, X_test, y_train, y_test = pipe1.run_preprocessing()

            if X_train is not None:
                import lightgbm as lgb

                # 저장된 튜닝 파라미터 로드 (없으면 기본값)
                best_params = None
                if os.path.exists(TUNING_FILE):
                    top10 = pd.read_csv(TUNING_FILE)
                    best_params = {
                        "learning_rate": float(top10.iloc[0]["learning_rate"]),
                        "num_leaves":    int(top10.iloc[0]["num_leaves"]),
                        "max_depth":     int(top10.iloc[0]["max_depth"]),
                    }
                # 가볍게 실시간 학습 모델 빌드 (사용자가 수정한 최신 파라미터 반영 목적)
                lr         = best_params["learning_rate"] if best_params else 0.05
                num_leaves = best_params["num_leaves"]    if best_params else 31
                max_depth  = best_params["max_depth"]     if best_params else -1
                
                # 카테고리형 변수 매핑 처리
                for col in ["Gender", "Contract", "PaymentMethod"]:
                    input_data[col] = input_data[col].astype("category")

                model = lgb.LGBMClassifier(
                    n_estimators=100, learning_rate=lr,
                    num_leaves=num_leaves, max_depth=max_depth,
                    random_state=42, class_weight="balanced", verbose=-1
                )
                model.fit(X_train, y_train)
                # 결과 예측
                prob = model.predict_proba(input_data)[0][1] * 100

                # ── 결과 출력 ────────────────────────────────────────────────
                st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
                st.markdown('<div class="section-title">분석 결과</div>', unsafe_allow_html=True)
                # st.markdown('<div class="custom-card">', unsafe_allow_html=True)
                # 게이지바 형태 시각화 대신 스트림릿 컴포넌트 활용
                if prob >= 50:
                    st.markdown(
                        f'<div style="display:flex;align-items:baseline;gap:16px;margin-bottom:4px">'
                        f'<span class="result-prob-high">{prob:.0f}%</span>'
                        f'<span class="risk-badge-high">🚨 이탈 위험군</span>'
                        f'</div>',
                        unsafe_allow_html=True
                    )
                    st.progress(int(prob))
                    st.markdown(
                        '<div class="result-tip-high">⚠️ <strong>추천 조치:</strong> '
                        '특별 할인 프로모션 제안, 장기 계약 전환 상담 유도가 필요합니다. '
                        '즉각적인 리텐션 액션을 취하세요.</div>',
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f'<div style="display:flex;align-items:baseline;gap:16px;margin-bottom:4px">'
                        f'<span class="result-prob-low">{prob:.0f}%</span>'
                        f'<span class="risk-badge-low">✅ 안정 유지군</span>'
                        f'</div>',
                        unsafe_allow_html=True
                    )
                    st.progress(int(prob))
                    st.markdown(
                        '<div class="result-tip-low">✔ <strong>현재 상태:</strong> '
                        '고객이 안정적으로 유지되고 있습니다. '
                        '정기적인 만족도 확인과 혜택 안내를 지속하세요.</div>',
                        unsafe_allow_html=True
                    )

                # st.markdown('</div>', unsafe_allow_html=True)  # result card

            else:
                st.error("❌ 전처리 모듈에서 데이터를 불러오지 못했습니다.")

    st.markdown('</div>', unsafe_allow_html=True)  # .content-wrap


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE 2 ── 관리자 모드
# ═══════════════════════════════════════════════════════════════════════════════
elif st.session_state.page == "admin":

    # 히어로 배너
    st.markdown("""
    <div class="hero-banner">
        <div class="hero-inner">
            <div>
                <div class="hero-h1">관리자 모드<br>모델 학습 / 튜닝</div>
                <div class="hero-sub">데이터 모니터링 및 예측 모델 관리 제어판 · LightGBM Pipeline</div>
            </div>
            <div class="hero-stats">
                <div class="hero-stat-box">
                    <div class="hero-stat-num">76.0%</div>
                    <div class="hero-stat-lbl">모델 정확도</div>
                </div>
                <div class="hero-stat-box">
                    <div class="hero-stat-num">0.80</div>
                    <div class="hero-stat-lbl">ROC-AUC</div>
                </div>   
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="content-wrap">', unsafe_allow_html=True)

    # ── 스탯 카드 4개 ──────────────────────────────────────────────────────────
    sc1, sc2, sc3, sc4 = st.columns(4, gap="medium")
    for col, label, value, delta in [
        (sc1, "총 데이터 행",     "100,000",  "synthetic_customer_churn.csv"),
        (sc2, "이탈 고객 비율",   "26.5%",    "class_weight='balanced' 적용"),
        (sc3, "Feature 수",       "7",        "Age · Tenure · Charges 등"),
        (sc4, "파이프라인 단계",  "3-Step",   "전처리 → 튜닝 → 최종학습"),
    ]:
        with col:
            st.markdown(
                f'<div class="stat-card">'
                f'<div class="sc-label">{label}</div>'
                f'<div class="sc-value">{value}</div>'
                f'<div class="sc-delta">{delta}</div>'
                f'</div>',
                unsafe_allow_html=True
            )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── 1. 데이터 프리뷰 ───────────────────────────────────────────────────────
    #  원본 데이터프레임 확인
    st.markdown('<div class="section-title">데이터셋 프리뷰</div>', unsafe_allow_html=True)

    if os.path.exists(DATA_PATH):
        df_preview = pd.read_csv(DATA_PATH, nrows=5)
        st.dataframe(df_preview, use_container_width=True)
        st.success("✔ 원본 데이터가 정상 인식되고 있습니다. (총 100,000개 행)")
    else:
        st.error(f"❌ '{DATA_PATH}' 파일을 찾을 수 없습니다. 데이터셋을 해당 경로에 배치해 주세요.")
        st.stop()

    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

    # ── 2. 모델 제어 ───────────────────────────────────────────────────────────
    st.markdown('<div class="section-title">LightGBM 모델 제어</div>', unsafe_allow_html=True)

    col_btn1, col_btn2 = st.columns(2, gap="large")

    # ── 튜닝 ──
    with col_btn1:
        st.markdown(
            '<div class="admin-ctrl-card">'
            '<div class="admin-ctrl-title">🔥 하이퍼파라미터 그리드 서치</div>'
            '<div class="admin-ctrl-desc">배치된 조합을 순회 연산하여 최적의 파라미터 조합 Top 10을 추출합니다.<br>'
            '결과는 <code>lgb_tuning_top10_results.csv</code>로 자동 저장됩니다.</div>'
            '</div>',
            unsafe_allow_html=True
        )
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🔥  튜닝 모듈 가동", key="btn_tuning"):
            with st.spinner("🔄 파라미터 조합별 성능 점수 연산 중... (약 10~20초 소요)"):
                top_10_results = pipe3.run_parameter_tuning()
            st.markdown('<div class="section-title">튜닝 결과 Top 10</div>', unsafe_allow_html=True)
            st.dataframe(top_10_results, use_container_width=True)
            st.success("✔ 상위 10개 결과가 'lgb_tuning_top10_results.csv' 파일로 자동 업데이트되었습니다.")

    # ── 최종 학습 ──
    with col_btn2:
        st.markdown(
            '<div class="admin-ctrl-card">'
            '<div class="admin-ctrl-title">🚀 최종 대규모 학습</div>'
            '<div class="admin-ctrl-desc">튜닝 1위 파라미터로 500개 트리 · Early Stopping 기반의 최종 모델을 생성합니다.<br>'
            '학습 완료 후 성능 지표 테이블과 피처 중요도 그래프를 출력합니다.</div>'
            '</div>',
            unsafe_allow_html=True
        )
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🚀  최종 트레이닝 모듈 가동", key="btn_training"):

            # 튜닝 파라미터 로드
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



            with st.spinner("🏋️ 10만 건 전체 데이터를 이용하여 최종 예측 모델 학습 및 평가 지표 산출 중..."):
                # 1. 전처리 모듈 가동하여 데이터 로드
                X_train, X_test, y_train, y_test = pipe1.run_preprocessing()
                # 2. 수정된 함수를 실행하여 실제 모델 평가 스코어(딕셔너리)를 받아오기.
                metrics = pipe2.run_model_training_and_evaluation(
                    X_train, X_test, y_train, y_test, best_params=best_params
                )

            st.success("🎉 최종 학습 완수!")

            # ── 사용 파라미터 표시 ──
            st.markdown('<div class="section-title">ightGBM 최적 파라미터 (튜닝 1순위)</div>', unsafe_allow_html=True)
            if best_params:
                # 열단위로 구분하여 작성
                p_col1, p_col2, p_col3 = st.columns(3)
                p_col1.markdown(f'<span class="param-badge">Learning Rate: {best_params["learning_rate"]}</span>', unsafe_allow_html=True)
                p_col2.markdown(f'<span class="param-badge">Num Leaves: {best_params["num_leaves"]}</span>',    unsafe_allow_html=True)
                p_col3.markdown(f'<span class="param-badge">Max Depth: {best_params["max_depth"]}</span>',      unsafe_allow_html=True)
            else:
                st.info("💡 튜닝 결과 파일이 없어 기본 파라미터로 실행되었습니다.")

            # ── 성능 지표 테이블 ──
            st.markdown('<div class="section-title">최종 모델 성능 평가</div>', unsafe_allow_html=True)
            # 평가 모듈에서 정상적으로 딕셔너리가 반환되었을 경우 테이블로 변환하여 출력
            if metrics and isinstance(metrics, dict):
                df_metrics = pd.DataFrame(metrics)
                # hide_index=True로 옵션명 변경
                st.dataframe(df_metrics, hide_index=True, use_container_width=True)
            else:
                st.error("❌ 모델 평가 지표를 불러오지 못했습니다. 리턴값을 확인해 주세요.")

            # ── 피처 중요도 그래프 ──
            # 현재 폴더에 저장하기
            IMG_PATH = 'models/LightGBM/saved_data/lgb_feature_importance.png'
            if os.path.exists(IMG_PATH):
                st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
                st.markdown('<div class="section-title">피처 중요도 (Feature Importance)</div>', unsafe_allow_html=True)
                st.image(IMG_PATH, caption="LightGBM이 분석한 고객 이탈 핵심 변수 순위",
                         use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)  # .content-wrap

# _________________________________________________

st.markdown("""
<div class="bottom-banner">
    <span class="bottom-banner-text">전문 상담사가 서비스 도입을 도와드려요!</span>
    <div class="bottom-banner-btns">
        <a class="bottom-banner-btn" href="#">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none"
                 stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
                <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>
            </svg>
            1:1 채팅 상담 &rsaquo;
        </a>
        <a class="bottom-banner-btn" href="#">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none"
                 stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
                <path d="M22 16.92v3a2 2 0 0 1-2.18 2 19.79 19.79 0 0 1-8.63-3.07A19.5 19.5 0 0 1 4.15 12 19.79 19.79 0 0 1 1.07 3.38 2 2 0 0 1 3.05 1h3a2 2 0 0 1 2 1.72c.13.96.36 1.9.7 2.81a2 2 0 0 1-.45 2.11L7.09 8.91a16 16 0 0 0 6 6l1.27-1.27a2 2 0 0 1 2.11-.45c.91.34 1.85.57 2.81.7A2 2 0 0 1 22 16.92z"/>
            </svg>
            전화 상담 &rsaquo;
        </a>
    </div>
</div>
""", unsafe_allow_html=True)

