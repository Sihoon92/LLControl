"""
APC 최적화 엔진 통합 테스트

MVP (Minimum Viable Product) 범위의 통합 테스트:
1. Cost Function 평가
2. Differential Evolution 최적화
3. Multi-zone 제어 평가
4. 최적화 결과 분석
"""

import sys
import logging
import argparse
import numpy as np
from pathlib import Path

# 모듈 import
sys.path.insert(0, str(Path(__file__).parent))

from apc_optimization import (
    N_ZONES, N_GV,
    CostFunctionEvaluator,
    CatBoostModelManager,
    MultiZoneController,
    DifferentialEvolutionOptimizer,
    create_config_summary,
)

logger = logging.getLogger(__name__)


def test_cost_function():
    """비용 함수 테스트"""
    logger.info("="*80)
    logger.info("Test 1: Cost Function 평가")
    logger.info("="*80)

    # 비용 함수 인스턴스 생성
    cost_evaluator = CostFunctionEvaluator()

    # 테스트 데이터
    p_low = np.random.uniform(0, 0.2, N_ZONES)
    p_mid = np.random.uniform(0.6, 1.0, N_ZONES)
    p_high = 1.0 - p_low - p_mid
    p_high = np.clip(p_high, 0, 1)

    delta_gv = np.random.uniform(-0.5, 0.5, 11)
    delta_rpm = np.random.uniform(-20, 20, 1)[0]

    # 평가
    total_cost, breakdown = cost_evaluator.evaluate_total_cost(
        p_low, p_mid, p_high, delta_gv, delta_rpm
    )

    logger.info(f"✓ 비용 평가 완료")
    logger.info(f"  총 비용: {total_cost:.6f}")
    logger.info(f"  품질: {breakdown['quality_cost']:.4f}")
    logger.info(f"  균형: {breakdown['balance_cost']:.4f}")
    logger.info(f"  제어: {breakdown['control_cost']:.4f}")
    logger.info(f"  안전: {breakdown['safety_cost']:.4f}")

    return True


def test_model_interface():
    """모델 인터페이스 테스트"""
    logger.info("="*80)
    logger.info("Test 2: Model Interface (CatBoost)")
    logger.info("="*80)

    # 모델 로드 - 명시적인 경로 설정
    from apc_optimization.config import MODEL_DIR, MODEL_PARAMS
    model_path = MODEL_DIR / f"{MODEL_PARAMS['model_name']}.pkl"
    model_manager = CatBoostModelManager(model_path=str(model_path))
    logger.info(f"✓ 모델 로드 완료: {type(model_manager.model).__name__}")

    # 배치 예측 테스트
    X_test = np.random.randn(5, 33)  # 임시 입력 (실제 모델에 맞춰 수정)
    predictions = model_manager.predict_batch(X_test)
    logger.info(f"✓ 배치 예측 완료")
    logger.info(f"  입력 shape: {X_test.shape}")
    logger.info(f"  예측 shape: {predictions.shape}")

    # Inverse CLR 테스트
    current_clr = np.random.randn(N_ZONES, 3)
    delta_clr = np.random.randn(N_ZONES, 3) * 0.1
    probabilities = model_manager.apply_inverse_clr_transform(current_clr, delta_clr)
    logger.info(f"✓ Inverse CLR 변환 완료")
    logger.info(f"  출력 shape: {probabilities.shape}")
    logger.info(f"  확률 합 (샘플): {np.sum(probabilities[0]):.4f}")

    return True


def test_multi_zone_controller():
    """다중 Zone 제어기 테스트"""
    logger.info("="*80)
    logger.info("Test 3: Multi-Zone Controller")
    logger.info("="*80)

    # 모델 매니저 및 제어기 생성
    model_manager = CatBoostModelManager()
    controller = MultiZoneController(model_manager)

    # Zone 정보 출력
    controller.print_zone_summary()

    # 제어값
    x_test = np.concatenate([
        np.random.uniform(-0.5, 0.5, 11),  # △GV
        np.array([10])                      # △RPM
    ])

    # 현재 상태
    current_state = {
        'current_clr': np.random.randn(N_ZONES, 3)
    }

    # 제어 평가
    result = controller.evaluate_control(x_test, current_state)
    logger.info(f"✓ 제어 평가 완료")
    logger.info(f"  P_Low 범위: [{result['p_low'].min():.4f}, {result['p_low'].max():.4f}]")
    logger.info(f"  P_Mid 범위: [{result['p_mid'].min():.4f}, {result['p_mid'].max():.4f}]")
    logger.info(f"  P_High 범위: [{result['p_high'].min():.4f}, {result['p_high'].max():.4f}]")

    return True


def test_optimizer_quick():
    """최적화 엔진 빠른 테스트 (10회 반복)"""
    logger.info("="*80)
    logger.info("Test 4: Differential Evolution Optimizer (Quick Test)")
    logger.info("="*80)

    # 모듈 초기화
    model_manager = CatBoostModelManager()
    cost_evaluator = CostFunctionEvaluator()

    # 현재 상태
    current_state = {
        'current_clr': np.random.randn(N_ZONES, 3)
    }

    # 최적화기 생성 (빠른 테스트용 - 10회 반복)
    optimizer = DifferentialEvolutionOptimizer(
        model_manager, cost_evaluator, current_state,
        optimizer_params={
            'strategy': 'best1bin',
            'maxiter': 10,  # 빠른 테스트용
            'popsize': 5,
            'tol': 0.001,
            'seed': 42,
            'workers': 1,
        }
    )

    # 최적화 실행
    result = optimizer.run_optimization()

    logger.info(f"✓ 최적화 완료 (빠른 테스트)")
    logger.info(f"  최적 비용: {result.cost_opt:.6f}")
    logger.info(f"  평가 횟수: {result.n_evaluations}")
    logger.info(f"  소요 시간: {result.optimization_time:.2f}초")
    logger.info(f"  최적해: {result.x_opt}")

    # 수렴 정보
    convergence_info = optimizer.get_convergence_info()
    initial = convergence_info.get('initial_cost', float('nan'))
    final = convergence_info.get('final_cost', float('nan'))
    logger.info(f"  초기 비용: {initial:.6f}" if isinstance(initial, float) else f"  초기 비용: {initial}")
    logger.info(f"  최종 비용: {final:.6f}" if isinstance(final, float) else f"  최종 비용: {final}")

    return True


def setup_logging(verbose: bool = False):
    """
    로깅 설정

    Args:
        verbose: True면 DEBUG 레벨, False면 INFO 레벨
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return level


def main():
    """모든 테스트 실행"""
    # 커맨드라인 인자 파싱
    parser = argparse.ArgumentParser(
        description='APC 최적화 엔진 통합 테스트',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  python apc_optimization_test.py              # 기본 실행 (INFO 레벨)
  python apc_optimization_test.py --verbose    # 상세 모드 (DEBUG 레벨)
  python apc_optimization_test.py -v           # 상세 모드 (단축)
        """
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='상세 로깅 활성화 (DEBUG 레벨 - logger.debug() 내용 표시)'
    )

    args = parser.parse_args()

    # 로깅 설정
    log_level = setup_logging(verbose=args.verbose)
    log_level_name = "DEBUG" if args.verbose else "INFO"

    logger.info("\n")
    logger.info("╔" + "="*78 + "╗")
    logger.info("║" + " "*20 + "APC 최적화 엔진 통합 테스트" + " "*30 + "║")
    logger.info("╚" + "="*78 + "╝")
    logger.info(f"로깅 레벨: {log_level_name} {'(--verbose 활성화됨)' if args.verbose else ''}\n")

    # 설정 요약 출력
    logger.info(create_config_summary())

    # 테스트 실행
    tests = [
        ("Cost Function", test_cost_function),
        ("Model Interface", test_model_interface),
        ("Multi-Zone Controller", test_multi_zone_controller),
        ("Optimizer (Quick)", test_optimizer_quick),
    ]

    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            logger.error(f"테스트 실패: {test_name}")
            logger.error(f"  오류: {e}", exc_info=True)
            results[test_name] = False

    # 결과 요약
    logger.info("\n")
    logger.info("="*80)
    logger.info("테스트 결과 요약")
    logger.info("="*80)

    for test_name, passed in results.items():
        status = "✓ 통과" if passed else "✗ 실패"
        logger.info(f"{status}: {test_name}")

    total_passed = sum(results.values())
    total_tests = len(results)
    logger.info(f"\n{total_passed}/{total_tests} 테스트 통과")

    if total_passed == total_tests:
        logger.info("\n🎉 모든 테스트 통과!")
        return 0
    else:
        logger.warning(f"\n⚠️  {total_tests - total_passed}개 테스트 실패")
        return 1


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
