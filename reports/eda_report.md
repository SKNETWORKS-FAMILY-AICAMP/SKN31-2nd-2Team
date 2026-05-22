# 📊 Spotify Churn Analysis: EDA Report

본 리포트는 `spotify_churn_dataset.csv` 데이터셋을 기반으로 사용자 이탈에 영향을 미치는 주요 요인을 분석한 결과입니다.

---

## 1. 데이터 요약 및 이탈 현황
<p align="center">
  <img src="figures\churn_distribution.png" width="500">
</p>

- **전체 사용자 수**: 8,000명
- **이탈 유저 수**: 2,071명
- **평균 이탈률**: **약 25.89%**
- **데이터 품질**: 모든 컬럼에서 결측치가 발견되지 않았으며, 8,000개의 행이 분석에 활용되었습니다.

---

## 2. 요금제 및 환경별 이탈 분석

### 2.1 구독 유형(Subscription Type)별 이탈률
<p align="center">
  <img src="figures\subscription_type.png" width="500">
</p>

유료 모델인 Family와 Student 플랜에서 평균보다 높은 이탈률이 관찰되었습니다.
- **Family**: 27.52%
- **Student**: 26.19%
- **Premium**: 25.06%
- **Free**: 24.93%

### 2.2 행동 지표와 이탈의 상관관계
<p align="center">
  <img src="figures\heatmap.png" width="500">
</p>

이탈 여부(`is_churned`)와 양(+)의 상관관계가 가장 높은 지표는 다음과 같습니다.
- **skip_rate (0.016)**: 곡을 자주 건너뛸수록 이탈 가능성이 높아지는 경향을 보입니다.
- **offline_listening (0.012)**: 오프라인 청취 기능 사용 여부와 약한 상관관계가 존재합니다.

---

## 3. 그룹별 평균 비교 (유지 vs 이탈)

이탈한 유저 그룹과 유지 중인 유저 그룹 간의 주요 지표 비교 결과입니다.

| 지표 (Average) | 유지 유저 (0) | 이탈 유저 (1) |
| :--- | :--- | :--- |
| **청취 시간 (listening_time)** | 154.45분 | 152.98분 |
| **곡 건너뛰기 비율 (skip_rate)** | 29.85% | 30.49% |
| **주당 광고 청취 (ads_listened)** | 6.96회 | 6.89회 |
| **평균 나이 (age)** | 37.6세 | 37.7세 |

> **분석 결과**: 이탈 유저 그룹은 유지 유저 그룹에 비해 **평균 청취 시간이 약 1.5분 짧고, 곡 건너뛰기 비율이 더 높게** 나타납니다.

---

## 4. 모델링을 위한 전처리 전략

1. **클래스 불균형 대응**: 이탈률이 약 26%이므로, 단순 정확도(Accuracy)보다는 **F1-Score**를 주요 평가지표로 설정합니다.
2. **범주형 변수 처리**: `subscription_type`, `device_type`, `country`, `gender`에 대해 **One-Hot Encoding**을 수행합니다.
3. **수치형 변수 스케일링**: `listening_time`과 `age` 등 단위 차이가 큰 변수들에 대해 **StandardScaler**를 적용하여 모델의 수렴 속도를 높입니다.
4. **특성 공학(Feature Engineering)**: 상관관계가 높은 `skip_rate`와 `listening_time`을 조합하여 사용자의 서비스 만족도를 점수화하는 파생 변수 생성을 검토합니다.