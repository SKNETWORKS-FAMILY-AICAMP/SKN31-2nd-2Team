# SKN31-2nd-2Team
SKN31기 2차프로젝트 2팀

# 🎵 Spotify 이용자 이탈 예측 머신러닝 프로젝트

본 프로젝트는 Spotify 유저들의 이용 행태 및 결제 데이터를 분석하여, 향후 서비스 탈퇴(이탈) 가능성이 높은 사용자를 예측하는 머신러닝 모델 개발 프로젝트입니다. 데이터 전처리, 탐색적 데이터 분석(EDA), 그리고 예측 모델링 과정을 포함하고 있습니다.

---

## 📂 저장소 구조 (Repository Structure)

```text
├── app/
│   └── app.py                    # streamlit 화면 구현
├── data/
│   └── spotify_churn_dataset.csv # 원본 데이터셋 (Raw Data)
├── notebooks/
│   ├── 01_eda.ipynb              # 탐색적 데이터 분석 및 시각화 코드
│   └── 02_preprocessing.ipynb    # 데이터 전처리 및 피처 엔지니어링 코드
├── models/
│   └── churn_predict_model.pkl   # 학습이 완료된 머신러닝 모델 파일
├── src/ 
│   ├── .gitignore
│   └── README.md                 # 프로젝트 소개 및 데이터 명세서
```

---

## 📊 데이터 명세서 (Data Dictionary)

본 문서는 Spotify 이용자 이탈 예측 머신러닝 모델 개발 프로젝트에 사용되는 원시 데이터(Raw Data)의 컬럼 정보와 전처리 가이드를 담고 있습니다.

### 📋 데이터 구조 개요
- **데이터셋 파일명:** `spotify_churn_dataset.csv`
- **목적:** 사용자 행동 패턴 및 결제 데이터를 기반으로 한 이탈 여부(`is_churned`) 이진 분류(Binary Classification)
- **대상 변수 (Target):** `is_churned` (0: 유지, 1: 이탈)

---

### 🔍 컬럼별 상세 명세 (Specification Table)

| 컬럼명 (Column Name) | 데이터 타입 (Type) | 변수 유형 (Class) | 설명 (Description) | 예시 값 (Example) | 전처리 시 고려사항 및 비고 (Pre-processing Notes) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **user_id** | 수치형 (Integer) | 식별자 (ID) | 고유 사용자 식별 번호 | `1`, `7971` | 단순 식별용 변수이므로 **모델 학습 시 제외(Drop)** 필요. |
| **gender** | 범주형 (String) | 독립 변수 (Feature) | 사용자의 성별 | `Female`, `Male`, `Other` | 다중 범주형 변수. **원-핫 인코딩(One-Hot Encoding)** 적용 필요. |
| **age** | 수치형 (Integer) | 독립 변수 (Feature) | 사용자의 나이 (만 나이) | `18`, `54`, `58` | 데이터 분포 확인 후 **스케일링(Scaling)** 또는 연령대별 **범주화(Binning)** 검토. |
| **country** | 범주형 (String) | 독립 변수 (Feature) | 거주 국가 (ISO 국가 코드 2자리) | `US`, `CA`, `DE`, `IN` | 범주가 많을 경우 고차원 방지를 위해 원-핫 인코딩 외에 **빈도수 기반 인코딩(Frequency/Target Encoding)** 고려. |
| **subscription_type**| 범주형 (String) | 독립 변수 (Feature) | 현재 이용 중인 구독 요금제 유형 | `Free`, `Premium`, `Family`, `Student` | 이탈률과 밀접한 핵심 변수. 순서형(Ordinal) 성격 여부 판단 후 인코딩 방식 결정. |
| **listening_time** | 수치형 (Integer) | 독립 변수 (Feature) | 총 청취 시간 (분 단위) | `26`, `141`, `280` | 연속형 수치 데이터. 이상치(Outlier) 제거 및 **MinMax/Standard 스케일링** 필요. |
| **songs_played_per_day**| 수치형 (Integer) | 독립 변수 (Feature) | 하루 평균 음악 재생 곡 수 | `3`, `23`, `62` | `listening_time`과 강한 상관관계를 보일 수 있으므로 **다중공선성(Multicollinearity)** 확인 필요. |
| **skip_rate** | 수치형 (Float) | 독립 변수 (Feature) | 곡 건너뛰기 비율 (0.0 ~ 1.0) | `0.04`, `0.20`, `0.46` | 이미 0과 1 사이로 정규화된 형태. 사용자 만족도를 나타내는 프록시(Proxy) 지표로 활용 가능. |
| **device_type** | 범주형 (String) | 독립 변수 (Feature) | 주로 사용하는 디바이스 플랫폼 | `Desktop`, `Mobile`, `Web` | 범주형 변수. **원-핫 인코딩(One-Hot Encoding)** 적용. |
| **ads_listened_per_week**| 수치형 (Integer) | 독립 변수 (Feature) | 주당 광고 청취 횟수 | `0`, `13`, `44` | `subscription_type`이 'Free'인 유저에게서만 높게 나타나는 경향 확인 및 교차 효과 분석 필요. |
| **offline_listening** | 범주형/이진 (Binary) | 독립 변수 (Feature) | 오프라인 다운로드/청취 기능 사용 여부 | `0` (미사용), `1` (사용) | 이미 0과 1로 이진 인코딩(Binary Encoded) 완료되어 변환 없이 수치 데이터로 바로 사용 가능. |
| **is_churned** | 범주형/이진 (Binary) | **종속 변수 (Target)** | 서비스 탈퇴(이탈) 여부 | `0` (유지), `1` (이탈) | **모델의 예측 목표.** 전체 데이터에서 0과 1의 비율을 확인하여 **클래스 불균형(Class Imbalance)** 대응 필요. |

---

### 🛠️ 데이터 탐색(EDA) 및 전처리 체크리스트

1. **[ ] 결측치 확인 (`df.isnull().sum()`)**
   - 각 컬럼에 누락된 값이 있는지 확인하고, 발견 시 대체(Imputation) 혹은 제거 전략 수립.
2. **[ ] 클래스 불균형 확인 (`df['is_churned'].value_counts(normalize=True)`)**
   - 이탈자 비율이 너무 적을 경우 모델이 0으로만 예측하는 편향이 생길 수 있으므로, 평가지표로 Accuracy 대신 **F1-Score / AUC-ROC** 채택 및 **SMOTE** 같은 복원 추출 알고리즘 고려.
3. **[ ] 파생 변수(Feature Engineering) 아이디어**
   - `listening_time` 대비 `songs_played_per_day` 비율을 계산하여 한 곡당 평균 청취 시간 지표 생성 가능.
   - `subscription_type`이 유료 요금제(Premium, Family, Student)임에도 불구하고 `offline_listening`이 0인 미사용 유저층 분석.