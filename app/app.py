import streamlit as st
import pandas as pd
import numpy as np
import time

# 1. 페이지 기본 설정
st.set_page_config(
    page_title="Spotify 유저 이탈 예측 시스템",
    page_icon="🎵",
    layout="wide"
)

# 데이터 로드 (스포티파이 특성을 반영한 가상 데이터 생성)
@st.cache_data
def load_spotify_data():
    np.random.seed(24)
    data = pd.DataFrame({
        '유저 ID': [f"SPOT-{i:04d}" for i in range(1, 101)],
        '요금제 유형': np.random.choice(['Premium Individual', 'Premium Duo', 'Premium Family', 'Free (광고형)'], 100),
        '주간 스트리밍 시간(시간)': np.random.randint(1, 40, 100),
        '곡 건너뛰기 비율(%)': np.random.randint(5, 85, 100),
        '생성한 플레이리스트 수': np.random.randint(0, 15, 100),
        '최근 30일 접속 일수': np.random.randint(1, 31, 100),
        '이탈 위험도(%)': np.random.randint(5, 95, 100)
    })
    return data

df = load_spotify_data()

# 2. 사이드바 - 스포티파이 테마 적용
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/2/26/Spotify_logo_with_text.svg", width=150)
st.sidebar.title("CRM 예측 시스템")
menu = st.sidebar.radio("메뉴 이동", ["📊 전체 대시보드", "🔍 유저별 이탈 예측", "📈 스트리밍 패턴 분석"])

# 3. 메인 화면 구성
st.title("🎵 Spotify User Churn Analytics")
st.caption("음악 스트리밍 패턴 및 구독 데이터를 활용한 탈퇴 위험도 실시간 예측 시스템")
st.write("---")

# ----------------- MENU 1: 전체 대시보드 -----------------
if menu == "📊 전체 대시보드":
    st.subheader("📈 당월 구독자 유지(Retention) 현황")
    
    # 핵심 지표 (KPI)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("분석 대상 유저 수", f"{len(df)}명", "안정군 72%")
    with col2:
        high_risk = len(df[df['이탈 위험도(%)'] >= 75])
        st.metric("🚨 이탈 고위험 유저", f"{high_risk}명", f"전체 중 {high_risk/len(df)*100:.1f}%", delta_color="inverse")
    with col3:
        st.metric("평균 곡 건너뛰기 비율", f"{df['곡 건너뛰기 비율(%)'].mean():.1f}%", "+2.1%", delta_color="inverse")
    with col4:
        st.metric("평균 생성 플레이리스트", f"{df['생성한 플레이리스트 수'].mean():.1f}개", "+0.4개")
        
    st.write("### 🟥 요주의 이탈 경고 유저 Top 5 (구독 해지 확률 최고조)")
    top_churners = df.sort_values(by='이탈 위험도(%)', ascending=False).head(5)
    st.dataframe(top_churners, use_container_width=True)

# ----------------- MENU 2: 유저별 이탈 예측 -----------------
elif menu == "🔍 유저별 이탈 예측":
    st.subheader("🧠 머신러닝 기반 실시간 이탈 위험도 시뮬레이터")
    st.write("유저의 음악 청취 습관과 앱 활동 로그를 입력하여 탈퇴 확률을 계산합니다.")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.info("🎧 유저 행동 데이터 입력")
        plan = st.selectbox("현재 가입 요금제", ['Premium Individual', 'Premium Duo', 'Premium Family', 'Free (광고형)'])
        weekly_hours = st.slider("주간 평균 스트리밍 시간 (시간)", min_value=0, max_value=100, value=15)
        skip_rate = st.slider("곡 건너뛰기 비율 (Skip Rate %)", min_value=0, max_value=100, value=65)
        playlists = st.number_input("유저가 직접 생성한 플레이리스트 수", min_value=0, max_value=50, value=2)
        offline_download = st.radio("오프라인 다운로드 기능 사용 여부", ["사용함", "사용 안 함 (또는 Free 권한 없음)"])
        
        predict_btn = st.button("🔮 이탈 가능성 추정", type="primary")

    with col2:
        st.info("📊 ML 모델 분석 결과")
        if predict_btn:
            with st.spinner('스포티파이 유저 행동 모델 계산 중...'):
                time.sleep(0.8)
                
                # 스포티파이 도메인 지식 기반 가상 로직
                # (플레이리스트가 적고, 스킵율이 높고, 오프라인 다운로드가 없으면 이탈 확률 상승)
                base_risk = 45
                if weekly_hours < 5: base_risk += 20
                if skip_rate > 60: base_risk += 15
                if playlists <= 1: base_risk += 15
                if offline_download == "사용 안 함 (또는 Free 권한 없음)": base_risk += 10
                
                risk_score = min(max(base_risk, 3), 97)
                
                # 예측 결과 반응형 UI
                st.metric(label="이탈 확률 (Churn Score)", value=f"{risk_score}%")
                
                if risk_score >= 75:
                    st.error("🚨 **위험 등급: 초고위험군 (Churn Imminent)**\n\n이 유저는 서비스 권태기이거나 개인 맞춤형 추천(Discover Weekly)에 불만족할 가능성이 큽니다. 즉시 3개월 할인 패키지 푸시 발송을 권장합니다.")
                    st.progress(risk_score / 100)
                elif risk_score >= 45:
                    st.warning("⚠️ **주의 등급: 관심 필요 (Muted Engagement)**\n\n최근 청취 다양성이 떨어지고 스킵률이 오르고 있습니다. 유저 취향에 맞는 신보 발매 알림이나 독점 팟캐스트 추천이 필요합니다.")
                    st.progress(risk_score / 100)
                else:
                    st.success("✅ **안정 등급: 충성 고객 (Active Enthusiast)**\n\n자신만의 플레이리스트를 적극적으로 소비하고 있으며 락인(Lock-in) 효과가 강력한 상태입니다.")
                    st.progress(risk_score / 100)
                
                # 가상 Feature Importance 시각화
                st.write("---")
                st.write("**🤖 이탈 판단에 영향을 준 주요 변수**")
                importance = pd.DataFrame({
                    '행동 지표': ['곡 건너뛰기 비율', '주간 스트리밍 시간', '플레이리스트 보유량', '요금제 락인 효과'],
                    '영향도': [0.42, 0.28, 0.20, 0.10]
                })
                st.bar_chart(importance.set_index('행동 지표'))
        else:
            st.write("👈 좌측에서 유저 시나리오를 설정한 뒤 버튼을 클릭해 주세요.")

# ----------------- MENU 3: 스트리밍 패턴 분석 -----------------
elif menu == "📈 스트리밍 패턴 분석":
    st.subheader("🔍 변수 간 상관관계 및 데이터 분포")
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("**[곡 건너뛰기 비율 vs 이탈 위험도 산점도]**")
        st.write("💡 스킵 비율이 높은 유저일수록 추천 시스템에 실망해 이탈할 확률이 급증합니다.")
        st.scatter_chart(data=df, x='곡 건너뛰기 비율(%)', y='이탈 위험도(%)', color='요금제 유형')
        
    with col2:
        st.write("**[생성한 플레이리스트 수에 따른 이탈 위험도]**")
        st.write("💡 자신만의 플레이리스트를 많이 만들수록 플랫폼 전환 비용이 커져 이탈율이 낮아집니다.")
        st.line_chart(data=df.set_index('생성한 플레이리스트 수')['이탈 위험도(%)'])

    st.write("### 📑 전체 분석 대상 유저 원본 데이터")
    st.dataframe(df, use_container_width=True)