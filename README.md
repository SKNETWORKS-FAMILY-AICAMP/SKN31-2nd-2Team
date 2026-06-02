# SKN31-2nd-2Team


**프로젝트 명:** 고객 이탈(Churn) 예측 모델 구축을 위한 데이터 탐색

**분석 데이터:** synthetic_customer_churn_100k.csv  
(출처 : https://www.kaggle.com/datasets/dhrubangtalukdar/telco-customer-churn-data)

본 프로젝트는 Telco 유저들의 이용 행태 및 결제 데이터를 분석하여, 향후 서비스 탈퇴(이탈) 가능성이 높은 사용자를 예측하는 머신러닝 모델 개발 프로젝트입니다. 데이터 전처리, 탐색적 데이터 분석(EDA), 그리고 예측 모델링 과정을 포함하고 있습니다.

---

## 저장소 구조 (Repository Structure)

```
SKN31-2nd-2Team
├─ app
│  └─ app.py                                    # streamlit 화면 구현
├─ data
│  ├─ eda_plots
│  │  ├─ 01_churn_distribution.png
│  │  ├─ 02_numerical_distributions.png
│  │  ├─ 03_categorical_churn_analysis.png
│  │  └─ 04_correlation_matrix.png
│  └─ synthetic_customer_churn_100k.csv
├─ models
│  ├─ 01_RandomForest                           # RandomForest 모델링 데이터
│  │  ├─ data
│  │  │  └─ synthetic_customer_churn_100k.csv   # 원본 데이터셋 (Raw Data)
│  │  ├─ data_scaling.py
│  │  ├─ eval.py
│  │  ├─ guide.md
│  │  ├─ modeling.py
│  │  └─ saved_models
│  │     ├─ randomforest_model.pkl
│  │     ├─ x_test.pkl
│  │     ├─ x_train.pkl
│  │     ├─ x_val.pkl
│  │     ├─ y_test.pkl
│  │     ├─ y_train.pkl
│  │     └─ y_val.pkl
│  ├─ 02_GradientBoosting                       # GradientBoosting 모델링 데이터
│  │  ├─ gb_feature_importance.png              # GradientBoosting 특성 중요도 이미지
│  │  ├─ gb_tuning_top10_results.csv            # GradientBoosting 하이퍼파라미터 튜닝 상위 10개 결과 csv
│  │  ├─ GradientBoosting_main.py      
│  │  ├─ GradientBoosting_pipeline_01_preprocessing.py   #  GradientBoosting 전처리 모듈
│  │  └─ GradientBoosting_pipeline_02_training.py     #  GradientBoosting 학습, 검증, 평가 모듈
│  ├─ 03_XGBoost                                # XGBoost 모델링 데이터
│  │  ├─ app.py
│  │  ├─ data_scaling.py
│  │  ├─ guide.md
│  │  ├─ model_comparison_results.csv
│  │  ├─ synthetic_customer_churn_100k.csv
│  │  ├─ XGBoost_main.py
│  │  ├─ XGBoost_pipeline_01_preprocessing.py
│  │  ├─ XGBoost_pipeline_02_training.py
│  │  ├─ XGBoost_pipeline_03_tuning.py
│  │  ├─ xgb_feature_importance.png
│  │  └─ xgb_tuning_top10_results.csv
│  ├─ 04_LightGBM                               # LightGBM 모델링 데이터
│  │  ├─ guide.md
│  │  ├─ LightGBM_01_preprocessing.py
│  │  ├─ LightGBM_02_training.py
│  │  ├─ LightGBM_03_tuning.py
│  │  ├─ LightGBM_main.py
│  │  └─ saved_data
│  │     ├─ lgb_feature_importance.png
│  │     └─ lgb_tuning_top10_results.csv
│  └─ 05_DeepLearning                           # DeepLearning 모델링 데이터
│     ├─ data
│     │  └─ synthetic_customer_churn_100k.csv
│     ├─ data_scaling.py
│     ├─ deep_learning_model.py
│     ├─ final_test.py
│     ├─ guide.md
│     ├─ saved_model
│     │  └─ deep_model.pt
│     ├─ train.py
│     └─ training.py
├─ outputs
│  ├─ 1_preprocessing_report.md                 # 전처리 결과서
│  └─ 2_model_training_report.md                # 학습결과서
├─ README.md                                    # 프로젝트 소개 및 데이터 명세서
└─ requirements.txt

```
---
## Tech Stack

---
## 1. 프로젝트 배경

오늘날 이동통신 시장은 신규 고객 유치 비용(CAC, Customer Acquisition Cost)이 기존 고객을 유지하는 비용보다 수 배 이상 높게 발생하는 '성숙기·포화 시장'의 특성을 띄고 있습니다. 무제한 요금제 경쟁, 결합 상품 다변화, 그리고 다양한 알뜰폰(MVNO) 사업자의 등장은 고객이 언제든 더 나은 조건의 경쟁사로 번호이동을 할 수 있는 환경을 제공합니다. 

특히 본 프로젝트에서 활용한 대규모 고객 데이터 분석 결과, **전체 고객의 약 33.1%에 달하는 인원이 이탈(Churn)하는 심각한 시그널이 관측**되었습니다. 이는 기업의 장기적인 매출 안정성을 크게 위협하는 수치입니다. 따라서 사후에 이탈한 고객을 붙잡는 소모적인 마케팅에서 벗어나, 데이터에 기반해 **'이탈 징후가 보이기 직전의 고객'을 선제적으로 감지하고 방어할 수 있는 데이터 기반의 정밀한 이탈 예측 시스템**이 요구되는 시점입니다.

---

## 2. 프로젝트 필요성

* **이상치 정제 및 데이터 신뢰성 확보**
  * 초기 데이터 탐색(EDA) 결과, 고객의 총 청구 금액(`TotalCharges`) 데이터 중 일부에서 **음수(-) 값의 기록 오류(이상치)가 발견**되었습니다. 현업 데이터에서 발생할 수 있는 이러한 노이즈를 방치한 채 모델을 학습시키면 예측 성능이 왜곡됩니다. 이에 따라 체계적인 데이터 정제(Data Cleaning)와 신뢰할 수 있는 데이터 파이프라인 구축이 필수적입니다.
* **다중공선성 및 복잡한 변수 관계의 극복**
  * 가입 기간(`Tenure`), 월 요금(`MonthlyCharges`), 총 청구 금액(`TotalCharges`)은 서로 매우 강한 선형 상관관계(다중공선성)를 가집니다. 일반적인 통계 모델로는 분석이 어려운 복잡한 관계를 효과적으로 다루기 위해, 다중공선성에 강건하면서도 대용량 데이터 처리 속도가 압도적인 머신러닝 알고리즘(LightGBM, XGBoost 등)을 도입한 고성능 모델 개발이 필요합니다.
* **타겟 마케팅으로의 패러다임 전환**
  * 모든 고객에게 무차별적으로 혜택이나 프로모션을 제공하는 비용 낭비형 마케팅에서 탈피해야 합니다. 이탈 확률이 높은 고위험군 고객 리스트와 이탈에 영향을 미친 주요 요인(Feature Importance)을 정밀 타격하여, 한정된 마케팅 예산 안에서 방어 효율을 극대화(ROI 최적화)해야 합니다.

---

## 3. 핵심 분석 및 기대효과

### 핵심 분석 내용
* **대용량 데이터셋 기반의 다차원 EDA:** 100,000명의 대규모 익명 고객 데이터를 바탕으로 나이 분포, 계약 형태(Contract), 결제 수단(PaymentMethod)과 이탈률 간의 비즈니스적 인과관계를 입증했습니다.
* **LightGBM 기반 고속 파이프라인 구축:** 트리 기반 앙상블 알고리즘을 활용하여, 데이터 스케일링 단계를 최소화하면서도 정확도와 변별력(ROC-AUC 지표 등) 측면에서 모두 최적의 성능을 내는 파라미터를 자동 튜닝(Hyperparameter Tuning)했습니다.
* **이탈 영향 인자 도출:** 단순 예측에 그치지 않고, 고객이 이탈 결정을 내리게 만드는 핵심 변수가 무엇인지 분석하여 비즈니스 부서가 즉각적인 액션 플랜(예: 계약 형태 변경 유도 프로모션 등)을 세울 수 있도록 지원합니다.

### 기대 효과
* **마케팅 비용 절감 (ROI 최적화):** 이탈 확률 상위 5~10%의 고위험군 고객만을 정밀 선별하여 개인화 혜택(쿠폰, 요금제 업그레이드 등)을 집중 제공함으로써 무분별한 마케팅 리소스 낭비를 막습니다.
* **LTV (고객 생애 가치) 증대:** 이탈률을 단 1~2%만 낮추더라도 장기 약정 고객을 확보하는 효과를 낳으며, 이는 매월 고정적인 반복 매출(ARR)의 안정적인 상승으로 직결됩니다.
* **이탈 방어 프로세스 자동화:** 개발된 머신러닝 모델을 주기적인 배치(Batch) 형태로 연동하여, 매월 혹은 매주 이탈 위험도가 높아진 고객 리스트를 고객 센터(CRM)나 마케팅 시스템에 자동으로 전달하는 자동화 인프라를 확립할 수 있습니다.
---

## 4. 데이터 개요 및 구조 분석

* **데이터 규모:** 총 100,000행(Rows), 9열(Columns)
* **결측치 현황:** 모든 변수의 Non-Null Count가 100,000개로 일치하여 결측치 미존재
* **변수 타입 구성:**
  * **수치형(Numerical) 변수:** `CustomerID`, `Age`, `Tenure`, `MonthlyCharges`, `TotalCharges` (5개)
  * **범주형(Categorical) 변수:** `Gender`, `Contract`, `PaymentMethod`, `Churn` (4개)

### 4-1. 데이터 변수 설명

| 컬럼명 | 설명 | 데이터 타입 | 예시 |
|---|---|---|---|
| CustomerID | 고객 고유 식별자 | int | 1, 2, …, 100000 |
| Age | 고객 나이 (18–80세) | int | 51 |
| Gender | 고객 성별 | string | Male / Female / Other |
| Tenure | 서비스 이용 기간 (월, 1–72) | int | 58 |
| MonthlyCharges | 월 청구 금액 (USD, 약 10–150) | float | 95.92 |
| TotalCharges | 누적 청구 금액 (Tenure × MonthlyCharges + 노이즈) | float | 5530.46 |
| Contract | 계약 유형 | string | 월별 / 1년 / 2년 |
| PaymentMethod | 결제 수단 | string | 전자수표 / 우편수표 / 계좌이체 / 신용카드 |
| Churn | 이탈 여부 (타깃 변수) | string | Yes / No |

---

## 5. 수치형 변수 기초 통계량 및 데이터 이상치 분석

각 수치형 변수의 기술 통계 정보(`df.describe()`)를 분석한 결과, 머신러닝 모델링 전에 반드시 인지해야 할 몇 가지 특이사항과 오류가 발견되었습니다.

| 변수명 | 평균값 (Mean) | 중위값 (50%) | 최솟값 (Min) | 최댓값 (Max) | 주요 특징 분석 |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Age** (나이) | 49.0세 | 49.0세 | 18.0세 | 80.0세 | 18세부터 80세까지 아주 고르게 분포된 정상 데이터입니다. |
| **Tenure** (가입기간) | 36.5개월 | 37.0개월 | 1.0개월 | 72.0개월 | 최소 1개월에서 최대 6년(72개월)까지의 가입 기간을 가집니다. |
| **MonthlyCharges** (월요금) | 79.9달러 | 80.0달러 | 10.0달러 | 150.0달러 | 최소 10달러~최대 150달러 사이에서 균등하게 청구되고 있습니다. |
| **TotalCharges** (총요금) | 2,926.1달러 | 2,268.0달러 | **-118.43달러** | 10,831.5달러 | **[오류 발견]** **최솟값이 음수(-118.43)** 인 데이터가 존재합니다. |

---

## 6. 타겟 변수(Churn) 분포 분석

모델이 예측해야 하는 핵심 타겟 변수인 **고객 이탈 여부 (`Churn`)** 의 비율을 파악했습니다.

![이탈 분포](../SKN31-2nd-2Team/data/eda_plots/01_churn_distribution.png)

* **이탈하지 않은 고객 (No):** 66,856명 (**66.86%**)
* **이탈한 고객 (Yes):** 33,144명 (**33.14%**)

### 클래스 불균형 분석
일반적인 기업의 이탈 데이터는 이탈자 비율이 5~10% 미만인 극심한 불균형 데이터가 많으나, 본 데이터는 이탈률이 **약 33%** 로 꽤 높은 편입니다. 따라서 모델 학습 시 적절한 불균형 전처리를 적용함으로써 모델이 안정적으로 이탈 패턴을 학습할 수 있는 환경을 만들어 주는 것이 필요합니다.

---

## 7. 변수 간 상관관계 분석 (Correlation Matrix)

`Churn` 변수를 수치화(Yes=1, No=0)하여 연속형 변수들과의 선형 상관관계를 분석한 결과입니다.

![수치형 분포](../SKN31-2nd-2Team/data/eda_plots/02_numerical_distributions.png)

![범주형 분석](../SKN31-2nd-2Team/data/eda_plots/03_categorical_churn_analysis.png)

![상관관계](../SKN31-2nd-2Team/data/eda_plots/04_correlation_matrix.png)

* **다중공선성(Multicollinearity) 확인:** `TotalCharges`(총 청구액)는 `Tenure`(가입 기간) 및 `MonthlyCharges`(월 청구액)와 매우 강한 양의 상관관계를 보입니다. (논리 공식인 $TotalCharges \approx Tenure \times MonthlyCharges$가 성립함을 보여줌.)
* **모델 연계:** Random Forest, Gradient Boosting, XGBoost, LightGBM은 트리 기반 앙상블 모델이기 때문에 성능(예측력) 자체에는 거의 영향이 없습니다.  하지만 변수중요도 분석을 해야한다면 높은 상관관계의 변수를 제거하는 것이 좋습니다.  Deep Learning은 수학적으로 경사하강법(Gradient Descent)과 가중치 연산을 사용하기 때문에 성능과 학습 안정성에 직접적인 타격을 받을 수 있습니다.  따라서 높은 상관관계의 변수를 제거하거나 강력한 규제를 가하는 것이 필요합니다.
---

## 8. 데이터 전처리

1. **이상치 처리**
   - `TotalCharges` 열에서 발견된 음수(-) 값을 0으로 대체
   - 총 100,000개 중 265개의 음수(-)을 0으로 대체

2. **변수 분리 및 인코딩(Encoding)**  
- feature와 target 분리(X, y 분리)
   - 총 컬럼(9개): CustomerID, Age, Gender, Tenure, MonthlyCharges, Contract, PaymentMethod, TotalCharges, Churn
   - feature(7개): Age, Gender, Tenure, MonthlyCharges, Contract, PaymentMethod, TotalCharges<br>
   - categorical_columns = ['Gender', 'Contract', 'PaymentMethod']<br>
   - numeric_columns = ['Tenure', 'MonthlyCharges', 'TotalCharges', 'Age']

- target(1개): Churn
   - target은 LabelEncoding을 통해 Yes를 0으로 , No를 1로 변환

   | 구분 | 레이블 | 의미 | 건수 |
   |------|--------|------|-----:|
   | 변환 전 | No | 유지 | 66,856 |
   | 변환 전 | Yes | 이탈 | 33,144 |
   | 변환 후 | 0 | 유지 | 66,856 |
   | 변환 후 | 1 | 이탈 | 33,144 |

3. **train/validation/test set 분리**

- train set(60%), validation set(20%), test set(20%)로 분리하여 데이터 준비

   | 변수 | 크기 |
   |------|-----:|
   | X_train | 60,000 |
   | y_train | 60,000 |
   | X_val | 20,000 |
   | y_val | 20,000 |
   | X_test | 20,000 |
   | y_test | 20,000 |

4. **데이터 스케일링(Scaling) 생략 가능**
- Random Forest, Gradient Boosting, XGBoost, LightGBM은 트리 기반 앙상블 모델을 사용하였으므로 수치형(numeric) 변수에 대해서는 별도의 Scaling을 하지 않음  

- 머신러닝 Encoding과 Scaling 정리  

   | 척도 | | 선형 알고리즘 | 트리 알고리즘 |
   |---|---|---|---|
   | 범주형 | 명목척도 | `OneHotEncoder(drop='')` | `OneHotEncoder -> LabelEncoder()` |
   | 범주형 | 순서척도 | `LabelEncoder()` | `LabelEncoder())` |
   | 수치형 | 간격척도 | Scaling 필요 | Scaling 불필요 |
   | 수치형 | 비율척도 | Scaling 필요 | Scaling 불필요 |

5. **파생 변수(Feature Engineering) 추가**
   * 고객의 가입 기간 대비 총 지불 금액의 비율을 나타내는 `TotalCharges / Tenure` (즉, 실제 인당 평균 월 지불 가치) 등의 파생 변수를 추가하면 이탈 모델의 예측력을 높일 수 있을 것으로 예상합니다.

## . 모델 학습 결과
