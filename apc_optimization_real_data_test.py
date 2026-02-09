"""
실제 Test Data로 APC 최적화 성능 평가

Baseline(실제 제어값) vs Optimized(최적화된 제어값) 비교
"""

import sys
import logging
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns

# 모듈 import
sys.path.insert(0, str(Path(__file__).parent))

from apc_optimization import (
    N_ZONES, N_GV,
    CostFunctionEvaluator,
    CatBoostModelManager,
    MultiZoneController,
    DifferentialEvolutionOptimizer,
)
from apc_optimization.evaluation_metrics import OptimizationEvaluator

logger = logging.getLogger(__name__)


def load_test_data(test_data_file: str) -> List[Dict]:
    """
    Test data 로드 및 전처리

    Parameters:
    -----------
    test_data_file : str
        model_test_data.xlsx 파일 경로

    Returns:
    --------
    List[Dict]
        테스트 샘플 리스트
    """
    logger.info("="*80)
    logger.info("Test Data 로드")
    logger.info("="*80)

    # 데이터 로드
    df = pd.read_excel(test_data_file)
    logger.info(f"로드된 데이터: {len(df)} 행")

    # 필요한 칼럼 확인
    required_cols = [
        'group_id', 'zone_id',
        'current_CLR_1', 'current_CLR_2', 'current_CLR_3',
        'delta_GV_self', 'delta_GV_left1', 'delta_GV_right1',
        'delta_GV_left2', 'delta_GV_right2', 'delta_RPM',
        'after_CLR_1', 'after_CLR_2', 'after_CLR_3',
        'before_ratio_1', 'before_ratio_2', 'before_ratio_3',
        'after_ratio_1', 'after_ratio_2', 'after_ratio_3'
    ]

    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        logger.error(f"누락된 칼럼: {missing_cols}")
        return []

    # 샘플별로 변환
    test_samples = []

    for _, row in df.iterrows():
        sample = {
            'group_id': row['group_id'],
            'zone_id': int(row['zone_id']),

            # 현재 상태 (before)
            'current_clr': np.array([
                row['current_CLR_1'],
                row['current_CLR_2'],
                row['current_CLR_3']
            ]),
            'current_probs': np.array([
                row['before_ratio_1'],
                row['before_ratio_2'],
                row['before_ratio_3']
            ]),

            # 실제 제어값
            'actual_control': {
                'delta_gv_self': row['delta_GV_self'],
                'delta_gv_left1': row['delta_GV_left1'],
                'delta_gv_right1': row['delta_GV_right1'],
                'delta_gv_left2': row['delta_GV_left2'],
                'delta_gv_right2': row['delta_GV_right2'],
                'delta_rpm': row['delta_RPM']
            },

            # 실제 결과 상태 (after)
            'actual_after_clr': np.array([
                row['after_CLR_1'],
                row['after_CLR_2'],
                row['after_CLR_3']
            ]),
            'actual_after_probs': np.array([
                row['after_ratio_1'],
                row['after_ratio_2'],
                row['after_ratio_3']
            ])
        }

        test_samples.append(sample)

    logger.info(f"변환된 샘플 수: {len(test_samples)}")
    logger.info(f"Group 수: {df['group_id'].nunique()}")
    logger.info(f"Zone 수: {df['zone_id'].nunique()}")

    return test_samples


def reconstruct_control_vector_from_sample(sample: Dict, all_samples: List[Dict]) -> np.ndarray:
    """
    샘플에서 전체 제어 벡터 [△GV₁~₁₁, △RPM] 재구성

    Parameters:
    -----------
    sample : Dict
        현재 샘플 (특정 zone)
    all_samples : List[Dict]
        전체 샘플 (같은 group_id의 다른 zone 포함)

    Returns:
    --------
    np.ndarray
        제어 벡터 [△GV₁~₁₁, △RPM]
    """
    group_id = sample['group_id']
    zone_id = sample['zone_id']

    # 같은 group의 모든 zone 데이터 찾기
    group_samples = [s for s in all_samples if s['group_id'] == group_id]

    # GV 제어값 초기화
    delta_gv = np.zeros(N_GV)

    for gs in group_samples:
        z_id = gs['zone_id']
        # zone_id는 1~11, 배열 인덱스는 0~10
        if 1 <= z_id <= N_GV:
            delta_gv[z_id - 1] = gs['actual_control']['delta_gv_self']

    # RPM은 전역 제어값이므로 현재 샘플에서 가져오기
    delta_rpm = sample['actual_control']['delta_rpm']

    # 제어 벡터 생성
    control_vector = np.concatenate([delta_gv, [delta_rpm]])

    return control_vector


def evaluate_baseline_performance(
    test_samples: List[Dict],
    model_manager: CatBoostModelManager,
    cost_evaluator: CostFunctionEvaluator
) -> List[Dict]:
    """
    Baseline 성능 평가 (실제 제어값 사용)

    Parameters:
    -----------
    test_samples : List[Dict]
        테스트 샘플 리스트
    model_manager : CatBoostModelManager
        모델 매니저
    cost_evaluator : CostFunctionEvaluator
        비용 평가기

    Returns:
    --------
    List[Dict]
        Baseline 결과 리스트
    """
    logger.info("="*80)
    logger.info("Baseline 성능 평가 (실제 제어값)")
    logger.info("="*80)

    results = []

    for idx, sample in enumerate(test_samples):
        # 실제 제어값 재구성
        actual_control = reconstruct_control_vector_from_sample(sample, test_samples)

        # 현재 상태에서 모델 예측
        # (여기서는 실제 after 상태를 사용)
        actual_probs = sample['actual_after_probs']

        # 비용 계산
        p_low, p_mid, p_high = actual_probs[0], actual_probs[1], actual_probs[2]

        # Zone별로 하나의 값이므로 확장
        p_low_arr = np.array([p_low] * N_ZONES)
        p_mid_arr = np.array([p_mid] * N_ZONES)
        p_high_arr = np.array([p_high] * N_ZONES)

        delta_gv = actual_control[:N_GV]
        delta_rpm = actual_control[N_GV]

        total_cost, breakdown = cost_evaluator.evaluate_total_cost(
            p_low_arr, p_mid_arr, p_high_arr, delta_gv, delta_rpm
        )

        result = {
            'group_id': sample['group_id'],
            'zone_id': sample['zone_id'],
            'actual_control': actual_control,
            'predicted_probs': actual_probs,  # 실제 = 예측으로 간주
            'actual_probs': actual_probs,
            'cost': total_cost,
            'cost_breakdown': breakdown
        }

        results.append(result)

        if (idx + 1) % 10 == 0:
            logger.info(f"  진행: {idx + 1}/{len(test_samples)} 샘플")

    logger.info(f"✓ Baseline 평가 완료: {len(results)} 샘플")
    avg_cost = np.mean([r['cost'] for r in results])
    logger.info(f"  평균 비용: {avg_cost:.6f}")

    return results


def evaluate_optimized_performance(
    test_samples: List[Dict],
    model_manager: CatBoostModelManager,
    cost_evaluator: CostFunctionEvaluator,
    max_samples: int = None
) -> List[Dict]:
    """
    최적화 성능 평가 (최적 제어값 탐색)

    Parameters:
    -----------
    test_samples : List[Dict]
        테스트 샘플 리스트
    model_manager : CatBoostModelManager
        모델 매니저
    cost_evaluator : CostFunctionEvaluator
        비용 평가기
    max_samples : int, optional
        최대 샘플 수 (테스트용)

    Returns:
    --------
    List[Dict]
        최적화 결과 리스트
    """
    logger.info("="*80)
    logger.info("최적화 성능 평가 (최적 제어값 탐색)")
    logger.info("="*80)

    if max_samples:
        test_samples = test_samples[:max_samples]
        logger.info(f"테스트 샘플 제한: {max_samples}개")

    results = []

    # Group별로 처리 (같은 group은 한번에 최적화)
    groups = {}
    for sample in test_samples:
        gid = sample['group_id']
        if gid not in groups:
            groups[gid] = []
        groups[gid].append(sample)

    logger.info(f"총 Group 수: {len(groups)}")

    for gidx, (group_id, group_samples) in enumerate(groups.items()):
        logger.info(f"\n[Group {group_id}] 최적화 시작 ({gidx+1}/{len(groups)})")

        # 현재 상태 준비 (전체 11개 zone의 CLR)
        current_state = {
            'current_clr': np.zeros((N_ZONES, 3))  # 11 zones x 3 CLR
        }

        # 각 zone의 current_clr 채우기
        for gs in group_samples:
            z_id = gs['zone_id']
            if 1 <= z_id <= N_ZONES:
                current_state['current_clr'][z_id - 1, :] = gs['current_clr']

        # 최적화기 생성 (빠른 테스트용 파라미터)
        optimizer = DifferentialEvolutionOptimizer(
            model_manager,
            cost_evaluator,
            current_state,
            optimizer_params={
                'strategy': 'best1bin',
                'maxiter': 20,  # 빠른 테스트용
                'popsize': 10,
                'tol': 0.01,
                'seed': 42,
                'workers': 1,
            }
        )

        # 최적화 실행
        try:
            opt_result = optimizer.run_optimization()

            # 최적 제어값으로 예측
            controller = MultiZoneController(model_manager)
            control_result = controller.evaluate_control(opt_result.x_opt, current_state)

            # 각 zone별 결과 저장
            for gs in group_samples:
                z_id = gs['zone_id']
                if 1 <= z_id <= N_ZONES:
                    z_idx = z_id - 1

                    predicted_probs = np.array([
                        control_result['p_low'][z_idx],
                        control_result['p_mid'][z_idx],
                        control_result['p_high'][z_idx]
                    ])

                    result = {
                        'group_id': group_id,
                        'zone_id': z_id,
                        'optimized_control': opt_result.x_opt.copy(),
                        'predicted_probs': predicted_probs,
                        'actual_probs': gs['actual_after_probs'],
                        'cost': opt_result.cost_opt,
                        'cost_breakdown': opt_result.final_cost_breakdown
                    }

                    results.append(result)

            logger.info(f"  ✓ Group {group_id} 최적화 완료, 비용: {opt_result.cost_opt:.6f}")

        except Exception as e:
            logger.error(f"  ✗ Group {group_id} 최적화 실패: {e}")
            continue

    logger.info(f"\n✓ 최적화 평가 완료: {len(results)} 샘플")
    avg_cost = np.mean([r['cost'] for r in results])
    logger.info(f"  평균 비용: {avg_cost:.6f}")

    return results


def compare_and_report(
    baseline_results: List[Dict],
    optimized_results: List[Dict],
    evaluator: OptimizationEvaluator,
    output_dir: str = './outputs'
) -> Dict:
    """
    결과 비교 및 리포트 생성

    Parameters:
    -----------
    baseline_results : List[Dict]
        Baseline 결과
    optimized_results : List[Dict]
        최적화 결과
    evaluator : OptimizationEvaluator
        평가기
    output_dir : str
        출력 디렉토리

    Returns:
    --------
    Dict
        비교 리포트
    """
    logger.info("="*80)
    logger.info("결과 비교 및 리포트 생성")
    logger.info("="*80)

    # 전체 비교
    overall_comparison = evaluator.compare_overall(baseline_results, optimized_results)

    # Zone별 비교
    zone_comparison = evaluator.compare_by_zone(baseline_results, optimized_results)

    # 리포트 생성
    report = evaluator.generate_evaluation_report({
        'overall': overall_comparison,
        'by_zone': zone_comparison,
        'baseline': baseline_results,
        'optimized': optimized_results
    })

    logger.info(report)

    # 리포트 저장
    report_file = Path(output_dir) / 'optimization_evaluation_report.txt'
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    logger.info(f"✓ 리포트 저장: {report_file}")

    # Zone별 결과 저장
    zone_file = Path(output_dir) / 'zone_comparison.xlsx'
    zone_comparison.to_excel(zone_file, index=False)
    logger.info(f"✓ Zone별 비교 저장: {zone_file}")

    return {
        'overall': overall_comparison,
        'by_zone': zone_comparison
    }


def visualize_comparison(
    baseline_results: List[Dict],
    optimized_results: List[Dict],
    output_dir: str = './outputs'
):
    """
    비교 결과 시각화

    Parameters:
    -----------
    baseline_results : List[Dict]
        Baseline 결과
    optimized_results : List[Dict]
        최적화 결과
    output_dir : str
        출력 디렉토리
    """
    logger.info("="*80)
    logger.info("결과 시각화")
    logger.info("="*80)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 1. 비용 비교 scatter plot
    fig, ax = plt.subplots(figsize=(10, 6))

    baseline_costs = [r['cost'] for r in baseline_results]
    optimized_costs = [r['cost'] for r in optimized_results]

    ax.scatter(baseline_costs, optimized_costs, alpha=0.5)
    ax.plot([min(baseline_costs), max(baseline_costs)],
            [min(baseline_costs), max(baseline_costs)],
            'r--', label='y=x (동일선)')

    ax.set_xlabel('Baseline Cost')
    ax.set_ylabel('Optimized Cost')
    ax.set_title('Baseline vs Optimized Cost Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plot_file = output_path / 'cost_comparison_scatter.png'
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    logger.info(f"✓ 비용 비교 plot 저장: {plot_file}")
    plt.close()

    # 2. 비용 개선도 bar chart
    fig, ax = plt.subplots(figsize=(12, 6))

    improvements = [(b['cost'] - o['cost']) / b['cost'] * 100
                   for b, o in zip(baseline_results, optimized_results)]

    ax.bar(range(len(improvements)), improvements, alpha=0.7)
    ax.axhline(y=0, color='r', linestyle='--', linewidth=1)
    ax.set_xlabel('Sample Index')
    ax.set_ylabel('Cost Improvement (%)')
    ax.set_title('Cost Improvement per Sample')
    ax.grid(True, alpha=0.3, axis='y')

    plot_file = output_path / 'cost_improvement_bar.png'
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    logger.info(f"✓ 개선도 bar chart 저장: {plot_file}")
    plt.close()

    logger.info("✓ 시각화 완료")


def setup_logging(verbose: bool = False):
    """로깅 설정"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # apc_optimization 패키지 로거 레벨 설정
    apc_logger = logging.getLogger('apc_optimization')
    apc_logger.setLevel(level)

    return level


def main():
    """메인 실행 함수"""

    # 커맨드라인 인자 파싱
    parser = argparse.ArgumentParser(
        description='실제 Test Data로 APC 최적화 성능 평가',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  python apc_optimization_real_data_test.py --test-data outputs/model_test_data.xlsx
  python apc_optimization_real_data_test.py --test-data outputs/model_test_data.xlsx --max-samples 10 --verbose
        """
    )

    parser.add_argument(
        '--test-data',
        type=str,
        default='./outputs/model_test_data.xlsx',
        help='테스트 데이터 파일 경로 (model_test_data.xlsx)'
    )
    parser.add_argument(
        '--max-samples',
        type=int,
        default=None,
        help='최대 샘플 수 (테스트용, 기본값: 전체)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./outputs',
        help='결과 저장 디렉토리'
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='상세 로깅 활성화 (DEBUG 레벨)'
    )

    args = parser.parse_args()

    # 로깅 설정
    setup_logging(verbose=args.verbose)

    logger.info("\n")
    logger.info("╔" + "="*78 + "╗")
    logger.info("║" + " "*15 + "APC 최적화 실제 데이터 성능 평가" + " "*29 + "║")
    logger.info("╚" + "="*78 + "╝")
    logger.info("")

    # 1. Test data 로드
    test_samples = load_test_data(args.test_data)

    if not test_samples:
        logger.error("테스트 데이터 로드 실패")
        return 1

    # 샘플 제한 (테스트용)
    if args.max_samples:
        test_samples = test_samples[:args.max_samples]
        logger.info(f"샘플 수 제한: {args.max_samples}개")

    # 2. 모델 및 평가기 초기화
    logger.info("\n모델 및 평가기 초기화")
    model_manager = CatBoostModelManager()
    cost_evaluator = CostFunctionEvaluator()
    evaluator = OptimizationEvaluator(cost_evaluator)

    # 3. Baseline 평가
    baseline_results = evaluate_baseline_performance(
        test_samples,
        model_manager,
        cost_evaluator
    )

    # 4. 최적화 평가
    optimized_results = evaluate_optimized_performance(
        test_samples,
        model_manager,
        cost_evaluator,
        max_samples=args.max_samples
    )

    if len(baseline_results) != len(optimized_results):
        logger.warning(f"결과 수 불일치: Baseline={len(baseline_results)}, Optimized={len(optimized_results)}")
        # 일치시키기 (group_id, zone_id 기준)
        baseline_keys = {(r['group_id'], r['zone_id']) for r in baseline_results}
        optimized_keys = {(r['group_id'], r['zone_id']) for r in optimized_results}
        common_keys = baseline_keys & optimized_keys

        baseline_results = [r for r in baseline_results if (r['group_id'], r['zone_id']) in common_keys]
        optimized_results = [r for r in optimized_results if (r['group_id'], r['zone_id']) in common_keys]

        logger.info(f"일치된 샘플 수: {len(baseline_results)}")

    # 5. 비교 및 리포트
    report = compare_and_report(
        baseline_results,
        optimized_results,
        evaluator,
        output_dir=args.output_dir
    )

    # 6. 시각화
    visualize_comparison(
        baseline_results,
        optimized_results,
        output_dir=args.output_dir
    )

    logger.info("\n")
    logger.info("="*80)
    logger.info("평가 완료!")
    logger.info("="*80)

    return 0


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
