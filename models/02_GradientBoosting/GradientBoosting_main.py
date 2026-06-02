# LightGBM_main.py
# -*- coding: utf-8 -*-
"""
[필수 라이브러리 설치 안내]
이 프로젝트를 실행하기 위해 아래 라이브러리들이 필요합니다.
터미널(Terminal)에서 다음 명령어를 실행하여 한 번에 설치할 수 있습니다:

pip install streamlit pandas numpy scikit-learn lightgbm matplotlib seaborn
-------------------------------------------------------------------------
-------------------------------------------------------------------------
고객 이탈 예측 메인(main.py)
-------------------------------------------------------------------------
1. Pipeline 01: 데이터 로드 및 전처리/분할
2. Pipeline 03: 하이퍼파라미터 튜닝 수행 ➔ 상위 10개 출력 및 최적 조합 도출
3. Pipeline 02: 도출된 최적 파라미터 주입 ➔ 최종 대규모 학습 및 평가
"""
import GradientBoosting_pipeline_01_preprocessing as pipe1
import GradientBoosting_pipeline_02_training as pipe2
import GradientBoosting_pipeline_03_tuning as pipe3


def main() -> None:
    """
    GradientBoosting1 전체 파이프라인을 실행합니다.

    Args:
        없음.

    Returns:
        None: 전처리, 튜닝, 최종 학습 결과를 콘솔과 파일로 출력합니다.
    """
    print("<GradientBoosting 데이터 전처리>")
    X_train, X_test, y_train, y_test = pipe1.run_preprocessing()

    if X_train is None:
        print("[오류] 데이터 전처리에 실패하여 파이프라인을 중단합니다.")
        return

    print("\n" + "-------------------------------------------------------------------------")
    print("<GradientBoosting 하이퍼파라미터 튜닝 진행>")
    print("-------------------------------------------------------------------------")

    top_10_df = pipe3.run_parameter_tuning()
    if top_10_df is None or top_10_df.empty:
        print("[오류] 튜닝 결과가 없어 파이프라인을 중단합니다.")
        return

    best_params = {
        "learning_rate": float(top_10_df.iloc[0]["learning_rate"]),
        "n_estimators": int(top_10_df.iloc[0]["n_estimators"]),
        "max_depth": int(top_10_df.iloc[0]["max_depth"]),
    }
    print(f"[튜닝 완료] 최적 파라미터 조합: {best_params}")

    print("\n" + "-------------------------------------------------------------------------")
    print("<최적 파라미터 기반 최종 GradientBoosting 모델 학습 진행>")
    print("-------------------------------------------------------------------------")

    pipe2.run_model_training_and_evaluation(
        X_train,
        X_test,
        y_train,
        y_test,
        best_params=best_params,
    )
    print("\n[전체 GradientBoosting 예측 파이프라인 완료]")


if __name__ == "__main__":
    main()
