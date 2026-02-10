"""
모델 학습 및 평가 실행 스크립트 (다변량 예측)

Training/Test 모드 지원:
- training 모드: model_training_data.xlsx로 학습
- test 모드: model_test_data.xlsx로 학습
"""
from model_trainer_v2 import ModelTrainer
import os
import argparse


def main():
    """메인 실행 함수"""

    # 커맨드라인 인자 파싱
    parser = argparse.ArgumentParser(
        description='코팅 L/L 제어 모델 학습 및 평가',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  # Training 모드 (기본)
  python train_models.py
  python train_models.py --mode training

  # Test 모드
  python train_models.py --mode test

  # 데이터 파일 직접 지정
  python train_models.py --data-file outputs/custom_data.xlsx

  # 출력 디렉토리 지정
  python train_models.py --mode test --output-dir outputs/test_models
        """
    )

    parser.add_argument(
        '--mode',
        type=str,
        choices=['training', 'test'],
        default='training',
        help='학습 모드 (training: 학습 데이터, test: 테스트 데이터)'
    )
    parser.add_argument(
        '--data-file',
        type=str,
        default=None,
        help='데이터 파일 경로 (None이면 mode 기반 자동 생성)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='모델 저장 디렉토리 (None이면 mode 기반 자동 생성)'
    )
    parser.add_argument(
        '--random-state',
        type=int,
        default=42,
        help='랜덤 시드'
    )

    args = parser.parse_args()

    # 데이터 파일 경로 결정
    if args.data_file is None:
        data_file = f'./outputs/model_{args.mode}_data.xlsx'
    else:
        data_file = args.data_file

    # 출력 디렉토리 결정
    if args.output_dir is None:
        output_dir = f'./outputs/models_{args.mode}'
    else:
        output_dir = args.output_dir

    # 헤더 출력
    print("="*80)
    print(f"코팅 L/L 제어 모델 학습 및 평가 ({args.mode.upper()} 모드)")
    print("="*80)
    print(f"데이터 파일: {data_file}")
    print(f"출력 디렉토리: {output_dir}")
    print("="*80)
    print()

    # 데이터 파일 존재 확인
    if not os.path.exists(data_file):
        print(f"ERROR: 데이터 파일을 찾을 수 없습니다: {data_file}")
        print(f"먼저 전처리 파이프라인을 실행하여 model_{args.mode}_data.xlsx를 생성하세요.")
        print()
        print(f"전처리 실행 예시:")
        print(f"  python main.py --mode {args.mode} --apc data/raw/apc_data.xlsx --densitometer data/raw/densitometer_data.csv")
        return 1

    # ModelTrainer 초기화
    trainer = ModelTrainer(
        data_file=data_file,
        output_dir=output_dir,
        random_state=args.random_state
    )

    # ======================================================================
    # Step 1: 데이터 로드 및 전처리
    # ======================================================================
    print("\n" + "="*80)
    print("[Step 1] 데이터 로드 및 전처리")
    print("="*80)
    trainer.load_data()
    trainer.prepare_data(
        test_size=0.2,
        scale_features=True
    )

    # ======================================================================
    # Step 2: 모델 학습 (여러 모델 비교)
    # ======================================================================
    print("\n" + "="*80)
    print(f"[Step 2] 모델 학습 ({args.mode.upper()} 데이터)")
    print("="*80)

    # 2-1. CatBoost (Independent)
    print("\n  [2-1] CatBoost (Independent)...")
    trainer.train_catboost(
        model_name='CatBoost_independent',
        multioutput_strategy='independent'
    )

    # 2-2. CatBoost (Chain)
    print("\n  [2-2] CatBoost (Chain)...")
    trainer.train_catboost(
        model_name='CatBoost_chain',
        multioutput_strategy='chain'
    )

    # 2-3. CatBoost (MultiRMSE)
    print("\n  [2-3] CatBoost (MultiRMSE - 네이티브 다변량)...")
    trainer.train_catboost_multi(
        model_name='CatBoost_multi'
    )

    # 2-4. XGBoost (Independent)
    print("\n  [2-4] XGBoost (Independent)...")
    trainer.train_xgboost(
        model_name='XGBoost_independent',
        multioutput_strategy='independent'
    )

    # 2-5. Random Forest (Independent)
    print("\n  [2-5] Random Forest (Independent)...")
    trainer.train_random_forest(
        model_name='RandomForest_independent',
        multioutput_strategy='independent'
    )

    # 2-6. MLP (Independent)
    print("\n  [2-6] MLP (Independent)...")
    trainer.train_mlp(
        model_name='MLP_independent',
        multioutput_strategy='independent'
    )

    # 2-7. PyTorch MLP with Constraints
    print("\n  [2-7] PyTorch MLP (물리적 제약 포함)...")
    trainer.train_pytorch_mlp(
        model_name='MLP_constrained'
    )

    # ======================================================================
    # Step 3: 교차 검증 (선택)
    # ======================================================================
    print("\n" + "="*80)
    print("[Step 3] 교차 검증 (선택적)")
    print("="*80)
    print("  교차 검증은 시간이 오래 걸리므로 skip...")
    # trainer.cross_validate('CatBoost_multi', cv=5)

    # ======================================================================
    # Step 4: 테스트 세트 평가
    # ======================================================================
    print("\n" + "="*80)
    print("[Step 4] 테스트 세트 평가")
    print("="*80)
    for model_name in trainer.models.keys():
        print(f"\n  평가 중: {model_name}...")
        trainer.evaluate_model(model_name)

    # ======================================================================
    # Step 5: 특성 중요도 분석
    # ======================================================================
    print("\n" + "="*80)
    print("[Step 5] 특성 중요도 분석")
    print("="*80)
    print("  [5-1] CatBoost_multi 특성 중요도...")
    trainer.plot_feature_importance('CatBoost_multi')

    # ======================================================================
    # Step 6: 잔차 분석
    # ======================================================================
    print("\n" + "="*80)
    print("[Step 6] 잔차 분석")
    print("="*80)
    print("  [6-1] CatBoost_multi 잔차 분석...")
    trainer.plot_residuals('CatBoost_multi')

    # ======================================================================
    # Step 7: 모델 성능 비교
    # ======================================================================
    print("\n" + "="*80)
    print("[Step 7] 모델 성능 비교")
    print("="*80)
    comparison_df = trainer.compare_models()

    # ======================================================================
    # Step 8: 시각화
    # ======================================================================
    print("\n" + "="*80)
    print("[Step 8] 결과 시각화")
    print("="*80)

    print("  [8-1] 예측 결과 산점도...")
    trainer.plot_predictions()

    print("  [8-2] 예측값 합 분포...")
    trainer.plot_sum_distribution()

    # ======================================================================
    # Step 9: 예측 결과 저장
    # ======================================================================
    print("\n" + "="*80)
    print("[Step 9] 예측 결과 저장")
    print("="*80)
    trainer.save_predictions()

    # ======================================================================
    # Step 10: 종합 리포트 생성
    # ======================================================================
    print("\n" + "="*80)
    print("[Step 10] 종합 리포트 생성")
    print("="*80)
    report = trainer.generate_report()

    # ======================================================================
    # 완료 - 결과 요약
    # ======================================================================
    print("\n" + "="*80)
    print("모든 작업 완료!")
    print("="*80)
    print(f"\n학습 모드: {args.mode.upper()}")
    print(f"데이터 파일: {data_file}")
    print(f"결과 저장 위치: {trainer.output_dir}")
    print("\n생성된 파일:")
    print("  - model_comparison.csv: 모델 성능 비교")
    print("  - predictions_scatter.png: 예측 결과 산점도")
    print("  - sum_distribution.png: 예측값 합 분포")
    print("  - *_predictions.csv: 각 모델의 상세 예측 결과")
    print("  - training_report.txt: 종합 리포트")
    print("  - training.log: 학습 로그")
    print()

    # 최고 성능 모델 출력
    print("="*80)
    print("최고 성능 모델 (RMSE 기준):")
    print("="*80)
    best_idx = comparison_df['RMSE'].idxmin()
    print(comparison_df.iloc[best_idx].to_string())

    print("\n" + "="*80)
    print("물리적 제약 준수도 (합 제약, 절대값 기준):")
    print("="*80)
    best_constraint_idx = comparison_df['합_평균'].abs().idxmin()
    print(comparison_df.iloc[best_constraint_idx].to_string())

    print("\n" + "="*80)
    print("\n추천 모델:")
    print("-"*80)
    print("1. 정확도 우선: " + comparison_df.iloc[best_idx]['모델'])
    print("2. 물리적 제약 준수 우선: " + comparison_df.iloc[best_constraint_idx]['모델'])
    print("3. 균형: CatBoost_multi 또는 MLP_constrained 권장")
    print("\n설명:")
    print("- Independent: 각 출력을 독립적으로 예측 (가장 빠르지만 상관관계 무시)")
    print("- Chain: 순차적 예측으로 출력 간 종속성 학습")
    print("- MultiRMSE: CatBoost 네이티브 다변량 회귀 (트리 기반 + 상관관계)")
    print("- Constrained: PyTorch MLP + 물리적 제약 손실 함수 (가장 강력)")
    print("\n" + "="*80)

    return 0

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)
