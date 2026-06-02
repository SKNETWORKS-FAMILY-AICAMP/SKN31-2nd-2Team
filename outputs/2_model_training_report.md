# 산출물 2: 모델 학습 결과 보고서

## 1. 모델링 전략

### 1-1. 평가지표 선정

본 프로젝트는 Telco 고객 이탈(Churn) 예측 문제를 다룬다.  
이탈 고객을 놓치면 고객 유지 마케팅 기회를 잃을 수 있고, 비이탈 고객을 이탈 고객으로 잘못 예측하면 불필요한 마케팅 비용이 발생할 수 있다. 따라서 Accuracy만으로 모델을 평가하지 않고, 이탈 고객 탐지력과 예측 신뢰도를 함께 확인한다.

| 지표 | 선정 이유 |
| --------- | ------------------------------------ |
| Recall | 실제 이탈 고객을 놓치지 않는 것이 중요하기 때문에 우선적으로 고려 |
| Precision | 바이탈 고객을 이탈 고객으로 잘못 판단하는 경우를 관리하기 위해 필요 |
| F1-Score | Precision과 Recall의 균형 확인 |
| ROC-AUC | 전체적인 분류 구분력 확인 |
| PR-AUC | 불균형 데이터에서 Fraud 탐지 성능을 평가하는 데 적합 |
| Accuracy | 참고 지표로 활용 |

> Telco 고객 이탈(Churn) 예측에서는 False Negative, 즉 실제 Fraud 거래를 정상 거래로 판단하는 경우가 큰 손실로 이어질 수 있다.
> 따라서 Recall을 중요하게 보고, Precision과 PR-AUC를 함께 확인하였다.

---

### 1-2. 후보 모델 선정

| 모델 | 유형 | 선정 근거 |
| ------------------- | ----- | ------------------------------------------ |
| Random Forest | 앙상블 | 여러 Decision Tree를 결합하여 안정적인 성능을 기대할 수 있고 Feature Importance 확인 가능 |
| Gradient Boosting | 부스팅 | 이전 모델의 오차를 순차적으로 보정하며 정형 데이터 분류에서 성능 확인 가능 |
| XGBoost | 부스팅 | 정형 데이터 분류 문제에서 높은 성능을 기대할 수 있는 대표 부스팅 모델 |
| LightGBM | 부스팅 | 빠른 학습 속도와 대용량 데이터 처리에 강점 |
| Deep Learning | 신경망 | 비선형 패턴 학습 가능성 확인 |

---

### 1-3. 실험 계획

TO-DO: 최종모델 선정 후 작성

* 기본 전처리 데이터를 기반으로 여러 머신러닝 모델을 학습하였다.
* Validation Set 기준으로 Accuracy, Precision, Recall, F1-Score, ROC-AUC를 비교하였다.
* 고객 이탈 예측 목적을 고려하여 Recall과 F1-Score를 중요하게 확인하였다.

---

## 2. 머신러닝 모델 학습 결과

### 2-1. Random Forest

TO-DO: 최종 소스코드받으면 작성

**하이퍼파라미터**

| 파라미터 | 값 |
| ------------ | --------- |
| n_estimators | 정리 예정 |
| max_depth | 정리 예정 |
| random_state | 42 |

**성능 결과**

| 지표 | Validation/Test |
| --- | ---: |
| F1-Score | 0.6112 |
| Average Precision | 0.7055 |
| ROC-AUC | 0.8247 |

**특이사항**

* 

---

### 2-2. Gradient Boosting

TO-DO: 최종 소스코드받으면 작성

**하이퍼파라미터**

| 파라미터 | 값 |
| --- | --- |
| learning_rate | 0.03 |
| n_estimators | 80 |
| max_depth | 3 |

**성능 결과**

| 지표 | Validation/Test |
| --- | ---: |
| Accuracy | 0.7523 |
| ROC-AUC | 0.8063 |

**특이사항**

* 현재 정리된 튜닝 결과 기준 ROC-AUC 0.8063으로 확인되었다.
* 얕은 트리 구조에서 비교적 안정적인 성능을 보였다.

---

### 2-3. XGBoost

TO-DO: 최종 소스코드받으면 작성

**하이퍼파라미터**

| 파라미터 | 값 |
| --- | --- |
| n_estimators | 정리 예정 |
| max_depth | 정리 예정 |
| learning_rate | 정리 예정 |
| subsample | 정리 예정 |
| colsample_bytree | 정리 예정 |

**성능 결과**

| 지표 | Validation/Test |
| --- | ---: |
| Accuracy | - |
| Precision | - |
| Recall | - |
| F1-Score | - |
| ROC-AUC | - |

**특이사항**

* 정형 데이터 분류 문제에서 높은 성능을 기대할 수 있는 후보 모델이다.
* Gradient Boosting, LightGBM과 함께 부스팅 계열 모델로 비교한다.
* 결과 정리 후 최종 비교표에 반영한다.

---

### 2-4. LightGBM

TO-DO: 최종 소스코드받으면 작성

**하이퍼파라미터**

| 파라미터 | 값 |
| --- | --- |
| learning_rate | 0.05 |
| num_leaves | 15 |
| max_depth | 6 |

**성능 결과**

| 지표 | Validation/Test |
| --- | ---: |
| Accuracy | 0.7396 |
| ROC-AUC | 0.8062 |

**특이사항**

* 현재 정리된 튜닝 결과 기준 ROC-AUC 0.8062로 확인되었다.
* Gradient Boosting과 유사한 수준의 ROC-AUC를 보였다.
* 학습 속도와 운영 적용 가능성을 함께 고려할 수 있다.

---

### 2-5. 주요 머신러닝 모델 비교

TO-DO: 최종 소스코드받으면 작성

| 모델 | Accuracy | Precision | Recall | F1-Score | ROC-AUC | 비고 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Random Forest | - | - | - | 0.6112 | 0.8247 | 결과 일부 정리 |
| Gradient Boosting | 0.7523 | - | - | - | 0.8063 | 튜닝 결과 기준 |
| XGBoost | - | - | - | - | - | 정리 예정 |
| LightGBM | 0.7396 | - | - | - | 0.8062 | 튜닝 결과 기준 |

**해석**

* 현재 정리된 결과 기준 Random Forest의 ROC-AUC가 가장 높다.
* Gradient Boosting과 LightGBM은 ROC-AUC가 약 0.806 수준으로 유사하다.
* XGBoost 결과가 아직 표에 정리되지 않아 최종 판단은 보류한다.
* TO-DO: 최종모델 선정 후 동일 조건 기준 최종 성능 비교 작성

---

### 2-6. 주요 부스팅 모델 비교

TO-DO: 최종 소스코드받으면 작성

| 모델                | F1-Score | ROC-AUC | PR-AUC |
| ----------------- | -------: | ------: | -----: |
| Gradient Boosting |   0.8677 |  0.9946 | 0.8777 |
| XGBoost           |   0.5027 |  0.9987 | 0.9284 |
| LightGBM          |   0.4576 |  0.9976 | 0.9188 |
| CatBoost          |   0.3876 |  0.9984 | 0.8844 |

**해석**

* Gradient Boosting은 F1-Score가 높게 나타났다.
* XGBoost는 PR-AUC와 ROC-AUC가 높아 불균형 데이터에서 Fraud 탐지 후보 모델로 적합하다고 판단하였다.
* 최종 모델은 PR-AUC와 Recall 중심 운영 가능성을 고려하여 XGBoost를 중심으로 튜닝하였다.

---

### 2-6. XGBoost 하이퍼파라미터 튜닝

TO-DO: 최종 소스코드받으면 작성

**튜닝 방법**

| 항목           | 내용                  |
| ------------ | ------------------- |
| 튜닝 도구        | Optuna              |
| Trial 수      | 100                 |
| 최적화 기준       | PR-AUC              |
| 불균형 보정       | scale_pos_weight 적용 |
| random_state | 42                  |

**탐색 파라미터**

| 파라미터             | 탐색 범위       |
| ---------------- | ----------- |
| n_estimators     | 100 ~ 500   |
| max_depth        | 3 ~ 10      |
| learning_rate    | 0.01 ~ 0.3  |
| subsample        | 0.6 ~ 1.0   |
| colsample_bytree | 0.6 ~ 1.0   |
| min_child_weight | 1 ~ 10      |
| gamma            | 0.0 ~ 1.0   |
| reg_alpha        | 1e-4 ~ 10.0 |
| reg_lambda       | 1e-4 ~ 10.0 |

**튜닝 전/후 비교**

| 지표      |   튜닝 전 |   튜닝 후 |      개선 |
| ------- | -----: | -----: | ------: |
| PR-AUC  | 0.9284 | 0.9717 | +0.0433 |
| ROC-AUC | 0.9987 | 0.9994 | +0.0007 |

**튜닝 후 주요 성능**

| 지표        |      값 |
| --------- | -----: |
| Precision | 0.9166 |
| Recall    | 0.9367 |
| F1-Score  | 0.9265 |
| F2-Score  | 0.9326 |
| PR-AUC    | 0.9717 |
| ROC-AUC   | 0.9994 |

---

## 3. 딥러닝 모델 학습 결과

### 3-1. Deep Learning 모델 개요

TO-DO: 최종 소스코드받으면 작성

정형 데이터 기반 고객 이탈 예측에서 신경망 모델의 성능을 확인하기 위해 Deep Learning 모델을 후보군에 포함하였다.

---

### 3-2. 모델 구조

TO-DO: 최종 소스코드받으면 작성

**네트워크 아키텍처**

```text
입력층 (17개 수치형 특성)
    │
    ├─ Linear(17 → 16)
    ├─ Linear(16 → 8)
    ├─ Linear(8 → 16)
    └─ Linear(16 → 17)
```

| 항목 | 내용 |
| ------- | ---------------------------------------- |
| 입력층 | 정리 예정 |
| 은닉층 | 정리 예정 |
| 출력층 | Binary Classification |
| 활성화 함수 | 정리 예정 |

---

### 3-3. 학습 설정

TO-DO: 최종 소스코드받으면 작성

| 파라미터 | 값 |
| ------- | ---------------------------------------- |
| 손실 함수 | 정리 예정 |
| 옵티마이저 | 정리 예정 |
| Epoch | 정리 예정 |
| Batch Size | 정리 예정 |
| Scaling | 적용 예정 |

---

### 3-4. Deep Learning 사용 결과

TO-DO: 최종 소스코드받으면 작성

| 결과             | 설명                          |
| -------------- | --------------------------- |
| Anomaly Score  | 복원 오차 기반 이상 점수              |
| 활용 목적          | XGBoost가 포착하지 못하는 비정상 패턴 보완 |
| Final Score 반영 | Anomaly Score를 20% 가중치로 반영  |

AutoEncoder는 최종 분류 모델이라기보다, Final Score를 구성하는 보조 이상 탐지 모듈로 활용하였다.

---

## 4. 모델 종합 비교 및 최적 모델 선정

### 4-1. 전체 모델 성능 비교표

TO-DO: 최종 소스코드받으면 작성

| 모델                  | F1-Score | ROC-AUC | PR-AUC |
| ------------------- | -------: | ------: | -----: |
| Logistic Regression |   0.0765 |  0.9374 | 0.2329 |
| Decision Tree       |   0.2809 |  0.9795 | 0.5203 |
| Random Forest       |   0.3150 |  0.9794 | 0.6838 |
| Extra Trees         |   0.0716 |  0.9170 | 0.4128 |
| Gradient Boosting   |   0.8677 |  0.9946 | 0.8777 |
| XGBoost             |   0.5027 |  0.9987 | 0.9284 |
| LightGBM            |   0.4576 |  0.9976 | 0.9188 |
| CatBoost            |   0.3876 |  0.9984 | 0.8844 |
| Stacking            |   0.4411 |  0.9986 | 0.9182 |
| Soft Voting         |   0.4990 |  0.9976 | 0.8808 |
| Hard Voting         |   0.5037 |       - |      - |
| XGBoost Tuned       |   0.9265 |  0.9994 | 0.9717 |

---

### 4-2. 지표별 최고 성능 모델

TO-DO: 최종 소스코드받으면 작성

| 지표       | 최고 모델         |      값 |
| -------- | ------------- | -----: |
| F1-Score | XGBoost Tuned | 0.9265 |
| F2-Score | XGBoost Tuned | 0.9326 |
| PR-AUC   | XGBoost Tuned | 0.9717 |
| ROC-AUC  | XGBoost Tuned | 0.9994 |

---

### 4-3. ROC Curve 비교

TO-DO: 최종 소스코드받으면 작성

---

### 4-4. Precision-Recall Curve 비교

TO-DO: 최종 소스코드받으면 작성

---

### 4-5. 최적 모델 선정

**선정 모델**: TO-DO: 최종모델 선정 후 작성

**선정 근거**

1. TO-DO: 최종모델 선정 후 주요 성능 지표 작성
2. TO-DO: 최종모델 선정 후 비교 모델 대비 우수한 점 작성
3. TO-DO: 최종모델 선정 후 고객 이탈 예측 관점의 선정 근거 작성
4. TO-DO: 최종모델 선정 후 운영 및 해석 가능성 작성

---

### 4-6. 최적 임계값 결정

TO-DO: 최종모델 선정 후 작성

| 항목 | 값 |
| --- | --- |
| 기본 Threshold | 0.50 |
| 최적 Threshold | TO-DO: 최종모델 선정 후 작성 |
| 기준 지표 | TO-DO: 최종모델 선정 후 작성 |

---

### 4-7. Threshold Trade-off 분석

TO-DO: 최종모델 선정 후 작성

| Threshold 변화 | 영향 |
| --- | --- |
| 낮게 설정 | Recall 증가, 이탈 고객 미탐 감소, 오탐 증가 가능 |
| 높게 설정 | Precision 증가, 오탐 감소, 이탈 고객 미탐 증가 가능 |

---

## 5. 모델 해석

### 5-1. 특성 중요도

TO-DO: 최종모델 선정 후 작성
최종 모델인 OO 기준 Feature Importance를 확인하였다.

<p align="center">
  <img src="" width="90%" alt="Feature Importance">
</p>

주요 Feature는 다음과 같다.

| 주요 Feature       | 해석                             |
| ---------------- | ------------------------------ |
| `amt`            | 거래 금액은 Fraud 탐지에 중요한 변수        |
| `category` 관련 변수 | 특정 업종에서 Fraud 패턴이 다르게 나타날 수 있음 |
| `trans_hour`     | 시간대별 이상거래 패턴 반영                |
| `amt_zscore`     | 고객 평균 대비 고액 거래 여부 반영           |
| `hour_dev`       | 평소 거래 시간과 다른 거래 패턴 반영          |
| `distance_km`    | 고객 위치와 가맹점 위치 간 거리 반영          |
| `high_amt_far`   | 고액 + 장거리 복합 위험 반영              |

---

### 5-2. 주요 변수 기반 해석

TO-DO: 최종모델 선정 후 작성

| 주요 Feature 후보 | 해석 방향 |
| --- | --- |
| `Tenure` | 가입 기간에 따른 이탈 위험 |
| `MonthlyCharges` | 월 요금 수준과 이탈 가능성 |
| `TotalCharges` | 누적 청구 금액과 고객 유지 관계 |
| `Contract` | 계약 유형별 이탈 패턴 |
| `PaymentMethod` | 결제 방식별 이탈 패턴 |

---

### 5-3. 고위험 고객 프로파일

TO-DO: 최종모델 선정 후 작성

* 가입 기간이 짧은 고객
* 월 요금이 높은 고객
* 특정 계약 유형에 속한 고객
* 특정 결제 방식을 사용하는 고객
* 모델 예측 확률이 높은 고객

---

## 6. 결론

본 보고서에서는 Telco 고객 이탈 예측을 위한 5개 후보 모델의 학습 결과와 현재까지의 비교 현황을 정리하였다.

현재까지 정리된 결과에서는 Random Forest의 ROC-AUC가 0.8247로 가장 높게 확인되었고, Gradient Boosting과 LightGBM은 튜닝 결과 기준 ROC-AUC가 약 0.806 수준으로 유사하게 나타났다. XGBoost와 Deep Learning 결과는 추가 정리 후 비교표에 반영한다.

TO-DO: 최종모델 선정 후 최종 모델명, 선정 근거, 성능 지표, 해석 결과를 추가 작성한다.

