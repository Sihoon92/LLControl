"""
CEM (Cross Entropy Method) 최적 제어 실행 스크립트

학습된 MBRL 앙상블 모델을 로드하여 CEM 최적화를 실행하고 결과를 출력합니다.

사용법:
  python run_cem.py
  python run_cem.py --model-path outputs/mbrl/training/models/best_model.pt
  python run_cem.py --horizon 3 --n-samples 100  # 빠른 테스트
  python run_cem.py --benchmark                  # DE vs CEM 비교
"""

import sys
import argparse
import logging
import numpy as np
from pathlib import Path

# 프로젝트 루트를 Python path에 추가
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from apc_optimization.mbrl import (
    PerZoneProbabilisticEnsemble,
    CEMPlanner,
    CEMBenchmark,
    DYNAMICS_MODEL_CONFIG,
    ENSEMBLE_CONFIG,
    PLANNER_CONFIG,
    MBRL_OUTPUT_DIR,
)
from apc_optimization.cost_function import CostFunctionEvaluator
from apc_optimization.model_interface import CatBoostModelManager


# =============================================================================
# 로깅 설정
# =============================================================================

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S',
    )
    return logging.getLogger('run_cem')


# =============================================================================
# 앙상블 모델 로드
# =============================================================================

def load_ensemble(model_path: str, device: str = 'cpu') -> PerZoneProbabilisticEnsemble:
    """
    학습된 앙상블 모델 로드

    Args:
        model_path: best_model.pt 경로
        device: 'cpu' | 'cuda' | 'mps'

    Returns:
        PerZoneProbabilisticEnsemble (가중치 로드 완료)
    """
    logger = logging.getLogger('run_cem')

    model = PerZoneProbabilisticEnsemble(
        n_ensembles=ENSEMBLE_CONFIG['n_ensembles'],          # 5
        input_dim=DYNAMICS_MODEL_CONFIG['input_dim'],        # 11
        output_dim=DYNAMICS_MODEL_CONFIG['output_dim'],      # 3
        hidden_dims=DYNAMICS_MODEL_CONFIG['hidden_dims'],    # [512, 256, 256, 128]
        device=device,
        activation=DYNAMICS_MODEL_CONFIG['activation'],
        use_layer_norm=DYNAMICS_MODEL_CONFIG['use_layer_norm'],
        dropout=DYNAMICS_MODEL_CONFIG['dropout'],
    )

    model_path = Path(model_path)
    if not model_path.exists():
        logger.error(f"모델 파일 없음: {model_path}")
        logger.error("먼저 앙상블 모델을 학습하세요:")
        logger.error("  python -m apc_optimization.mbrl.train --data-file outputs/model_training_data.xlsx")
        sys.exit(1)

    model.load(str(model_path))
    logger.info(f"앙상블 모델 로드 완료: {model_path}")

    info = model.get_model_info()
    logger.info(f"  앙상블 수: {info['n_ensembles']}")
    logger.info(f"  총 파라미터: {info['total_parameters']:,}")

    return model


# =============================================================================
# 현재 공정 상태 구성
# =============================================================================

def build_current_state(clr_csv: str = None) -> dict:
    """
    현재 공정 CLR 상태 구성

    Args:
        clr_csv: CLR 값이 담긴 CSV 파일 경로 (없으면 예시 값 사용)
                 형식: 11행 × 3열 (CLR_1, CLR_2, CLR_3)

    Returns:
        {'current_clr': np.ndarray (11, 3)}
    """
    logger = logging.getLogger('run_cem')

    if clr_csv is not None and Path(clr_csv).exists():
        import pandas as pd
        df = pd.read_csv(clr_csv, header=None)
        current_clr = df.values.astype(np.float32)  # (11, 3)
        logger.info(f"CLR 상태 로드: {clr_csv}")
    else:
        # 예시 초기 상태: P_Mid 편향 (정상 공정에 가까운 CLR)
        # CLR 계산식: clr_i = log(p_i) - mean(log(p))
        # P_Low=0.1, P_Mid=0.8, P_High=0.1 → CLR ≈ [-1.79, 0.95, -1.79] (정규화 후)
        p = np.array([0.1, 0.8, 0.1])
        clr_base = np.log(p) - np.mean(np.log(p))   # (3,)

        # 존별 소폭 노이즈 추가
        rng = np.random.default_rng(42)
        noise = rng.normal(0, 0.05, (11, 3))
        current_clr = (np.tile(clr_base, (11, 1)) + noise).astype(np.float32)

        logger.info("현재 CLR 상태: 예시 값 사용 (P_Mid 편향)")
        logger.info("  실제 데이터 사용 시: --clr-file path/to/clr.csv")

    logger.info(f"  CLR shape: {current_clr.shape}  (11 zones × 3 components)")

    return {'current_clr': current_clr}


# =============================================================================
# CEM 실행 및 결과 출력
# =============================================================================

def run_cem(
    model: PerZoneProbabilisticEnsemble,
    cost_fn: CostFunctionEvaluator,
    current_state: dict,
    planner_config: dict,
):
    """CEM 최적화 실행 후 결과 출력"""
    logger = logging.getLogger('run_cem')

    logger.info("=" * 70)
    logger.info("CEM 최적화 시작")
    logger.info(f"  horizon={planner_config['horizon']}, "
                f"n_samples={planner_config['n_samples']}, "
                f"n_elite={planner_config['n_elite']}, "
                f"n_iterations={planner_config['n_iterations']}")
    logger.info("=" * 70)

    cem = CEMPlanner(
        dynamics_model=model,
        cost_evaluator=cost_fn,
        current_state=current_state,
        planner_config=planner_config,
    )

    result = cem.run_optimization()

    # ── 결과 출력 ─────────────────────────────────────────────────────────
    print()
    print("=" * 70)
    print("  CEM 최적화 결과")
    print("=" * 70)
    print(f"  최적 비용:   {result.cost_opt:.6f}")
    print(f"  소요 시간:   {result.optimization_time:.2f}초")
    print(f"  총 롤아웃:   {result.n_evaluations:,}회")
    print()
    print("  [최적 제어값]")
    print(f"  {'Zone':<10} {'ΔGV (mm)':<15}")
    print(f"  {'-'*25}")
    for i, gv in enumerate(result.x_opt[:11], 1):
        bar = '▶' * int(abs(gv) * 5)
        sign = '+' if gv >= 0 else '-'
        print(f"  Zone{i:02d}      {sign}{abs(gv):.4f}  {bar}")
    print(f"  {'ΔRPM':<10} {result.x_opt[11]:+.2f}")
    print()

    if result.final_cost_breakdown:
        print("  [비용 상세]")
        for k, v in result.final_cost_breakdown.items():
            print(f"  {k:<20}: {v:.6f}")

    print("=" * 70)

    return result


# =============================================================================
# 벤치마크 실행 (DE vs CEM)
# =============================================================================

def run_benchmark_mode(
    model: PerZoneProbabilisticEnsemble,
    cost_fn: CostFunctionEvaluator,
    n_scenarios: int,
    sim_horizon: int,
    output_dir: Path,
):
    """DE+CatBoost vs CEM+MBRL 비교 벤치마크 실행"""
    logger = logging.getLogger('run_cem')

    logger.info("=" * 70)
    logger.info("벤치마크 모드: DE+CatBoost  vs  CEM+MBRL")
    logger.info(f"  시나리오: {n_scenarios}개, sim_horizon: {sim_horizon}스텝")
    logger.info("=" * 70)

    # CatBoost 모델 로드 (DE용, 없으면 Mock)
    catboost_manager = CatBoostModelManager()

    bench = CEMBenchmark(
        dynamics_model=model,
        cost_evaluator=cost_fn,
        model_manager=catboost_manager,
        sim_horizon=sim_horizon,
    )

    df = bench.run(n_scenarios=n_scenarios)

    # CSV 저장
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "de_vs_cem_benchmark.csv"
    CEMBenchmark.save_results(df, str(csv_path))
    print(f"\n  결과 저장: {csv_path}")

    return df


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='CEM 최적 제어 실행 스크립트',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  # 기본 실행 (config.py 기본 설정)
  python run_cem.py

  # 모델 경로 직접 지정
  python run_cem.py --model-path outputs/mbrl/training/models/best_model.pt

  # 빠른 테스트 (파라미터 축소)
  python run_cem.py --horizon 3 --n-samples 100 --n-iterations 3

  # 현재 CLR 상태를 CSV로 입력
  python run_cem.py --clr-file data/current_clr.csv

  # DE vs CEM 벤치마크 비교
  python run_cem.py --benchmark --n-scenarios 20

  # GPU 사용
  python run_cem.py --device cuda
        """,
    )

    # 모델 설정
    parser.add_argument(
        '--model-path', type=str,
        default='outputs/mbrl/training/models/best_model.pt',
        help='학습된 앙상블 모델 경로 (기본: outputs/mbrl/training/models/best_model.pt)',
    )
    parser.add_argument(
        '--device', type=str, choices=['cpu', 'cuda', 'mps'], default='cpu',
        help='PyTorch 디바이스 (기본: cpu)',
    )

    # 현재 상태
    parser.add_argument(
        '--clr-file', type=str, default=None,
        help='현재 CLR 상태 CSV 파일 (11행 × 3열, 없으면 예시 값 사용)',
    )

    # CEM 파라미터 (기본값은 config.py PLANNER_CONFIG)
    parser.add_argument(
        '--horizon', type=int, default=None,
        help=f"롤아웃 horizon (기본: {PLANNER_CONFIG['horizon']})",
    )
    parser.add_argument(
        '--n-samples', type=int, default=None,
        help=f"샘플 수 (기본: {PLANNER_CONFIG['n_samples']})",
    )
    parser.add_argument(
        '--n-elite', type=int, default=None,
        help=f"Elite 수 (기본: {PLANNER_CONFIG['n_elite']})",
    )
    parser.add_argument(
        '--n-iterations', type=int, default=None,
        help=f"CEM 반복 횟수 (기본: {PLANNER_CONFIG['n_iterations']})",
    )
    parser.add_argument(
        '--uncertainty-penalty', type=float, default=None,
        help=f"불확실성 페널티 (기본: {PLANNER_CONFIG['uncertainty_penalty']})",
    )

    # 벤치마크 모드
    parser.add_argument(
        '--benchmark', action='store_true',
        help='DE+CatBoost vs CEM+MBRL 벤치마크 비교 실행',
    )
    parser.add_argument(
        '--n-scenarios', type=int, default=20,
        help='벤치마크 시나리오 수 (기본: 20)',
    )
    parser.add_argument(
        '--sim-horizon', type=int, default=5,
        help='벤치마크 시뮬레이션 스텝 수 (기본: 5)',
    )

    # 출력
    parser.add_argument(
        '--output-dir', type=str, default='outputs/mbrl',
        help='결과 저장 디렉토리 (기본: outputs/mbrl)',
    )

    return parser.parse_args()


# =============================================================================
# 메인
# =============================================================================

def main():
    logger = setup_logging()
    args = parse_args()

    print()
    print("=" * 70)
    print("  CEM 기반 MBRL 최적 제어")
    print("=" * 70)
    print(f"  모델 경로: {args.model_path}")
    print(f"  디바이스:  {args.device}")
    print(f"  모드:      {'벤치마크' if args.benchmark else 'CEM 최적화'}")
    print()

    # ── 1. 앙상블 모델 로드 ────────────────────────────────────────────────
    model = load_ensemble(args.model_path, args.device)

    # ── 2. 비용 함수 ───────────────────────────────────────────────────────
    cost_fn = CostFunctionEvaluator()

    output_dir = Path(args.output_dir)

    # ── 3. 벤치마크 모드 ───────────────────────────────────────────────────
    if args.benchmark:
        run_benchmark_mode(
            model=model,
            cost_fn=cost_fn,
            n_scenarios=args.n_scenarios,
            sim_horizon=args.sim_horizon,
            output_dir=output_dir,
        )
        return

    # ── 4. CEM 최적화 모드 ─────────────────────────────────────────────────

    # 현재 공정 상태
    current_state = build_current_state(args.clr_file)

    # CEM 파라미터 (CLI 인자가 있으면 config 기본값 덮어씀)
    planner_config = dict(PLANNER_CONFIG)  # 복사
    if args.horizon is not None:
        planner_config['horizon'] = args.horizon
    if args.n_samples is not None:
        planner_config['n_samples'] = args.n_samples
    if args.n_elite is not None:
        planner_config['n_elite'] = args.n_elite
    if args.n_iterations is not None:
        planner_config['n_iterations'] = args.n_iterations
    if args.uncertainty_penalty is not None:
        planner_config['uncertainty_penalty'] = args.uncertainty_penalty

    # CEM 실행
    result = run_cem(model, cost_fn, current_state, planner_config)

    # 결과 저장 (npy)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "cem_x_opt.npy", result.x_opt)
    print(f"\n  최적 제어값 저장: {output_dir / 'cem_x_opt.npy'}")


if __name__ == '__main__':
    main()
