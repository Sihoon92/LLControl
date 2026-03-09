"""
CEM (Cross Entropy Method) 최적 제어 실행 스크립트

학습된 MBRL 앙상블 모델을 로드하여 CEM 최적화를 실행하고 결과를 출력합니다.

사용법:
  python run_cem.py
  python run_cem.py --model-path outputs/mbrl/training/models/best_model.pt
  python run_cem.py --horizon 3 --n-samples 100  # 빠른 테스트
  python run_cem.py --benchmark                  # DE vs CEM 비교

  # 실제 데이터 기반 비교 모드
  python run_cem.py --data-file outputs/model_training_data.xlsx --list-groups
  python run_cem.py --data-file outputs/model_training_data.xlsx --group-id 42
  python run_cem.py --data-file outputs/model_training_data.xlsx  # 랜덤 group
"""

import sys
import argparse
import logging
import numpy as np
import pandas as pd
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
import utils


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
        n_ensembles=ENSEMBLE_CONFIG['n_ensembles'],
        input_dim=DYNAMICS_MODEL_CONFIG['input_dim'],
        output_dim=DYNAMICS_MODEL_CONFIG['output_dim'],
        hidden_dims=DYNAMICS_MODEL_CONFIG['hidden_dims'],
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
# 현재 공정 상태 구성 (예시 / CSV)
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
        df = pd.read_csv(clr_csv, header=None)
        current_clr = df.values.astype(np.float32)
        logger.info(f"CLR 상태 로드: {clr_csv}")
    else:
        # 예시: P_Low=0.1, P_Mid=0.8, P_High=0.1 기반 CLR
        p = np.array([0.1, 0.8, 0.1])
        clr_base = np.log(p) - np.mean(np.log(p))
        rng = np.random.default_rng(42)
        noise = rng.normal(0, 0.05, (11, 3))
        current_clr = (np.tile(clr_base, (11, 1)) + noise).astype(np.float32)
        logger.info("현재 CLR 상태: 예시 값 사용 (P_Mid 편향)")
        logger.info("  실제 데이터 사용 시: --clr-file path/to/clr.csv")

    logger.info(f"  CLR shape: {current_clr.shape}  (11 zones × 3 components)")

    return {'current_clr': current_clr}


# =============================================================================
# 실제 데이터 기반 상태 구성
# =============================================================================

def load_group_from_data(data_file: str, group_id: int = None, list_groups: bool = False) -> pd.DataFrame:
    """
    model_training_data.xlsx에서 특정 group_id의 11개 zone 행을 로드

    Args:
        data_file:   xlsx / parquet / csv 경로
        group_id:    로드할 group_id (None이면 랜덤 선택)
        list_groups: True면 group_id 목록만 출력하고 종료

    Returns:
        zone_id 기준 정렬된 DataFrame (11행)
    """
    logger = logging.getLogger('run_cem')

    data_path = Path(data_file)
    if not data_path.exists():
        logger.error(f"데이터 파일 없음: {data_file}")
        sys.exit(1)

    logger.info(f"데이터 로드: {data_file}")
    df = utils.load_file(str(data_file), logger=logger)

    # None 컬럼명 처리 (xlwings 로드 시 발생)
    none_cols = [c for c in df.columns if c is None]
    if none_cols:
        df.rename(columns={c: f'unnamed_{i}' for i, c in enumerate(none_cols)}, inplace=True)

    required = ['group_id', 'zone_id',
                'current_CLR_1', 'current_CLR_2', 'current_CLR_3',
                'before_ratio_1', 'before_ratio_2', 'before_ratio_3',
                'delta_GV_self', 'delta_RPM',
                'diff_CLR_1', 'diff_CLR_2', 'diff_CLR_3',
                'after_ratio_1', 'after_ratio_2', 'after_ratio_3']

    missing = [c for c in required if c not in df.columns]
    if missing:
        logger.error(f"필요한 컬럼 없음: {missing}")
        sys.exit(1)

    all_groups = sorted(df['group_id'].unique())
    logger.info(f"  전체 데이터: {len(df)}행, group 수: {len(all_groups)}")

    # --list-groups 모드
    if list_groups:
        print()
        print("=" * 50)
        print(f"  사용 가능한 group_id ({len(all_groups)}개)")
        print("=" * 50)
        for i, gid in enumerate(all_groups):
            n_zones = (df['group_id'] == gid).sum()
            print(f"  {gid}  ({n_zones} zones)")
        print()
        sys.exit(0)

    # group 선택
    if group_id is None:
        group_id = np.random.choice(all_groups)
        logger.info(f"  group_id 랜덤 선택: {group_id}")
    elif group_id not in all_groups:
        logger.error(f"group_id={group_id} 없음. 사용 가능: {all_groups[:10]} ...")
        sys.exit(1)
    else:
        logger.info(f"  group_id: {group_id}")

    group_df = df[df['group_id'] == group_id].copy()
    group_df = group_df.sort_values('zone_id').reset_index(drop=True)

    n_zones = len(group_df)
    logger.info(f"  추출된 zone 수: {n_zones}")
    if n_zones != 11:
        logger.warning(f"  zone 수가 11개가 아님 ({n_zones}개). 결과 해석 주의.")

    return group_df


def build_state_from_group(group_df: pd.DataFrame) -> dict:
    """
    group DataFrame에서 CEM 입력 및 비교용 실제 정보를 추출

    반환 dict:
        group_id          : int
        current_clr       : (11, 3)  float32  ← PETS 모델 입력
        before_probs      : (11, 3)  float32  ← [P_Low, P_Mid, P_High] 제어 전 확률
        actual_delta_gv   : (11,)    float32  ← 실제 GV 변화량
        actual_delta_rpm  : float             ← 실제 RPM 변화량 (전역)
        actual_diff_clr   : (11, 3)  float32  ← 실제 CLR 변화량 (ground truth)
        actual_after_probs: (11, 3)  float32  ← 실제 제어 후 확률
    """
    # PETS 모델 입력: CLR 공간
    current_clr = group_df[['current_CLR_1', 'current_CLR_2', 'current_CLR_3']].values.astype(np.float32)

    # 비용 함수 입력: 확률 공간 (before)
    before_probs = group_df[['before_ratio_1', 'before_ratio_2', 'before_ratio_3']].values.astype(np.float32)

    # 실제 제어값
    # zone_id는 1~11이므로 배열 인덱스(0~10)에 맞게 zone_id 순 정렬된 상태 가정
    actual_delta_gv = group_df['delta_GV_self'].values.astype(np.float32)  # (11,)
    actual_delta_rpm = float(group_df['delta_RPM'].iloc[0])                # 전역값, 1행에서 추출

    # 실제 결과 (ground truth)
    actual_diff_clr = group_df[['diff_CLR_1', 'diff_CLR_2', 'diff_CLR_3']].values.astype(np.float32)
    actual_after_probs = group_df[['after_ratio_1', 'after_ratio_2', 'after_ratio_3']].values.astype(np.float32)

    return {
        'group_id':           group_df['group_id'].iloc[0],
        'current_clr':        current_clr,
        'before_probs':       before_probs,
        'actual_delta_gv':    actual_delta_gv,
        'actual_delta_rpm':   actual_delta_rpm,
        'actual_diff_clr':    actual_diff_clr,
        'actual_after_probs': actual_after_probs,
    }


# =============================================================================
# 실제 제어 비용 평가
# =============================================================================

def evaluate_actual_control(cost_fn: CostFunctionEvaluator, state: dict) -> tuple:
    """
    실제 제어값 + 실제 after_ratio로 비용 평가 (PETS 모델 사용 안 함)

    Returns:
        (actual_cost: float, actual_breakdown: dict)
    """
    p_low  = state['actual_after_probs'][:, 0]
    p_mid  = state['actual_after_probs'][:, 1]
    p_high = state['actual_after_probs'][:, 2]

    actual_cost, actual_breakdown = cost_fn.evaluate_total_cost(
        p_low, p_mid, p_high,
        state['actual_delta_gv'],
        state['actual_delta_rpm'],
    )
    return actual_cost, actual_breakdown


# =============================================================================
# CEM 최적 제어의 예측 결과 평가
# =============================================================================

def evaluate_cem_prediction(
    model: PerZoneProbabilisticEnsemble,
    cost_fn: CostFunctionEvaluator,
    state: dict,
    x_opt: np.ndarray,
) -> tuple:
    """
    CEM 최적 제어값을 PETS 앙상블으로 예측하여 비용 평가

    Args:
        x_opt: (12,) CEM 최적 제어값 [ΔGV×11, ΔRPM]

    Returns:
        (cem_cost: float, cem_breakdown: dict, cem_after_probs: np.ndarray (11, 3))
    """
    cem_delta_gv  = x_opt[:11]
    cem_delta_rpm = float(x_opt[11])

    # PETS 앙상블으로 다음 CLR 예측
    pred = model.predict_all_zones(
        current_clr_all=state['current_clr'],
        delta_gv=cem_delta_gv,
        delta_rpm=cem_delta_rpm,
        return_uncertainty=False,
    )
    next_clr = pred['next_clr']  # (11, 3)

    # CLR → 확률 역변환
    cem_after_probs = CatBoostModelManager.inverse_clr(next_clr)  # (11, 3)

    p_low  = cem_after_probs[:, 0]
    p_mid  = cem_after_probs[:, 1]
    p_high = cem_after_probs[:, 2]

    cem_cost, cem_breakdown = cost_fn.evaluate_total_cost(
        p_low, p_mid, p_high,
        cem_delta_gv,
        cem_delta_rpm,
    )
    return cem_cost, cem_breakdown, cem_after_probs


# =============================================================================
# 비교 출력
# =============================================================================

def print_comparison(
    state: dict,
    actual_cost: float,
    actual_bd: dict,
    cem_cost: float,
    cem_bd: dict,
    cem_after_probs: np.ndarray,
    x_opt: np.ndarray,
):
    """실제 제어 vs CEM 최적 제어 비교 결과 출력"""

    W = 72

    def improvement(actual_val: float, cem_val: float) -> str:
        if actual_val == 0:
            return "  N/A"
        delta = (cem_val - actual_val) / actual_val * 100
        arrow = '↑ 개선' if delta < 0 else ('↓ 악화' if delta > 0 else '-')
        return f"{delta:+.1f}%  {arrow}"

    # ── 섹션 1: 비용 비교 ────────────────────────────────────────────────────
    print()
    print("=" * W)
    print("  실제 제어 vs CEM 최적 제어  비교 결과")
    print(f"  Group ID: {state['group_id']}")
    print("=" * W)
    print()
    print("  [1] 비용 비교")
    print(f"  {'항목':<16} {'실제 제어':>12} {'CEM 최적':>12}   {'개선도'}")
    print(f"  {'-'*60}")

    cost_items = [
        ('총 비용',   actual_cost,              cem_cost),
        ('품질 비용', actual_bd['quality_cost'], cem_bd['quality_cost']),
        ('균형 비용', actual_bd['balance_cost'], cem_bd['balance_cost']),
        ('제어 비용', actual_bd['control_cost'], cem_bd['control_cost']),
        ('안전 비용', actual_bd['safety_cost'],  cem_bd['safety_cost']),
    ]

    for label, a_val, c_val in cost_items:
        imp = improvement(a_val, c_val)
        print(f"  {label:<16} {a_val:>12.6f} {c_val:>12.6f}   {imp}")

    # ── 섹션 2: P_Mid 변화 ───────────────────────────────────────────────────
    print()
    print("  [2] P_Mid 평균 변화 (품질 지표)")
    print(f"  {'-'*60}")

    p_mid_before = float(state['before_probs'][:, 1].mean())
    p_mid_actual = float(state['actual_after_probs'][:, 1].mean())
    p_mid_cem    = float(cem_after_probs[:, 1].mean())

    delta_actual = (p_mid_actual - p_mid_before) * 100
    delta_cem    = (p_mid_cem    - p_mid_before) * 100
    delta_vs_actual = (p_mid_cem - p_mid_actual) * 100

    print(f"  {'제어 전       ':20} {p_mid_before:.4f}")
    print(f"  {'실제 제어 후  ':20} {p_mid_actual:.4f}  ({delta_actual:+.2f}%p)")
    print(f"  {'CEM 예측 후   ':20} {p_mid_cem:.4f}  ({delta_cem:+.2f}%p, 실제 대비 {delta_vs_actual:+.2f}%p)")

    # Zone별 P_Mid 출력
    print()
    print(f"  {'Zone':<8} {'P_Mid 전':>10} {'P_Mid 실제 후':>14} {'P_Mid CEM 후':>13}")
    print(f"  {'-'*48}")
    for i in range(len(state['before_probs'])):
        b  = state['before_probs'][i, 1]
        a  = state['actual_after_probs'][i, 1]
        c  = cem_after_probs[i, 1]
        print(f"  Zone{i+1:02d}   {b:>10.4f} {a:>14.4f} {c:>13.4f}")

    # ── 섹션 3: Zone별 제어값 비교 ──────────────────────────────────────────
    print()
    print("  [3] Zone별 제어값 비교")
    print(f"  {'Zone':<8} {'실제 ΔGV':>10} {'CEM ΔGV':>10} {'차이':>8}")
    print(f"  {'-'*40}")

    actual_gv = state['actual_delta_gv']
    cem_gv    = x_opt[:11]

    for i in range(11):
        diff = cem_gv[i] - actual_gv[i]
        diff_str = f"{diff:+.0f}" if diff != 0 else "  0"
        print(f"  Zone{i+1:02d}   {actual_gv[i]:>10.0f} {cem_gv[i]:>10.0f} {diff_str:>8}")

    actual_rpm = state['actual_delta_rpm']
    cem_rpm    = float(x_opt[11])
    rpm_diff   = cem_rpm - actual_rpm
    print(f"  {'ΔRPM':<8} {actual_rpm:>10.1f} {cem_rpm:>10.1f} {rpm_diff:>+8.1f}")

    # ── 섹션 4: 실제 CLR 변화 (ground truth) vs CEM 예측 ───────────────────
    print()
    print("  [4] 실제 diff_CLR (ground truth) — 참고용")
    print(f"  {'Zone':<8} {'실제 ΔCLR_1':>12} {'실제 ΔCLR_2':>12} {'실제 ΔCLR_3':>12}")
    print(f"  {'-'*52}")
    for i in range(len(state['actual_diff_clr'])):
        d = state['actual_diff_clr'][i]
        print(f"  Zone{i+1:02d}   {d[0]:>12.4f} {d[1]:>12.4f} {d[2]:>12.4f}")

    print()
    print("=" * W)


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
        print(f"  Zone{i:02d}      {sign}{abs(gv):.0f}  {bar}")
    print(f"  {'ΔRPM':<10} {result.x_opt[11]:+.2f}")
    print()

    if result.final_cost_breakdown:
        print("  [비용 상세]")
        for k, v in result.final_cost_breakdown.items():
            if isinstance(v, (int, float, np.floating, np.integer)):
                print(f"  {k:<20}: {float(v):.6f}")

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

    catboost_manager = CatBoostModelManager()

    bench = CEMBenchmark(
        dynamics_model=model,
        cost_evaluator=cost_fn,
        model_manager=catboost_manager,
        sim_horizon=sim_horizon,
    )

    df = bench.run(n_scenarios=n_scenarios)

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
  # 예시 CLR로 실행 (기본)
  python run_cem.py

  # 실제 데이터 기반 — group_id 목록 확인
  python run_cem.py --data-file outputs/model_training_data.xlsx --list-groups

  # 실제 데이터 기반 — 특정 group 비교
  python run_cem.py --data-file outputs/model_training_data.xlsx --group-id 42

  # 실제 데이터 기반 — 랜덤 group
  python run_cem.py --data-file outputs/model_training_data.xlsx

  # CEM 파라미터 조정
  python run_cem.py --data-file outputs/model_training_data.xlsx --group-id 5 \\
                    --horizon 3 --n-samples 200 --n-iterations 5

  # DE vs CEM 벤치마크 비교
  python run_cem.py --benchmark --n-scenarios 20

  # GPU 사용
  python run_cem.py --data-file outputs/model_training_data.xlsx --device cuda
        """,
    )

    # 모델 설정
    parser.add_argument(
        '--model-path', type=str,
        default='outputs/mbrl/training/models/best_model.pt',
        help='학습된 앙상블 모델 경로',
    )
    parser.add_argument(
        '--device', type=str, choices=['cpu', 'cuda', 'mps'], default='cpu',
        help='PyTorch 디바이스 (기본: cpu)',
    )

    # 실제 데이터 기반 모드
    parser.add_argument(
        '--data-file', type=str, default=None,
        help='model_training_data.xlsx 경로 (지정 시 실제 데이터 기반 비교 모드)',
    )
    parser.add_argument(
        '--group-id', type=int, default=None,
        help='비교할 group_id (--data-file 필요, 미지정 시 랜덤 선택)',
    )
    parser.add_argument(
        '--list-groups', action='store_true',
        help='사용 가능한 group_id 목록 출력 후 종료 (--data-file 필요)',
    )

    # 기존 CLR 파일 방식 (하위 호환)
    parser.add_argument(
        '--clr-file', type=str, default=None,
        help='현재 CLR 상태 CSV 파일 (11행 × 3열, --data-file 없을 때 사용)',
    )

    # CEM 파라미터
    parser.add_argument('--horizon',             type=int,   default=None)
    parser.add_argument('--n-samples',           type=int,   default=None)
    parser.add_argument('--n-elite',             type=int,   default=None)
    parser.add_argument('--n-iterations',        type=int,   default=None)
    parser.add_argument('--uncertainty-penalty', type=float, default=None)

    # 벤치마크 모드
    parser.add_argument('--benchmark',    action='store_true')
    parser.add_argument('--n-scenarios',  type=int, default=20)
    parser.add_argument('--sim-horizon',  type=int, default=5)

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

    # --list-groups는 --data-file 필수
    if args.list_groups and not args.data_file:
        print("오류: --list-groups 사용 시 --data-file 필요")
        sys.exit(1)

    print()
    print("=" * 72)
    print("  CEM 기반 MBRL 최적 제어")
    print("=" * 72)
    print(f"  모델 경로: {args.model_path}")
    print(f"  디바이스:  {args.device}")
    if args.benchmark:
        print(f"  모드:      벤치마크 (DE vs CEM)")
    elif args.data_file:
        print(f"  모드:      실제 데이터 기반 비교")
        print(f"  데이터:    {args.data_file}")
    else:
        print(f"  모드:      CEM 최적화 (예시/CSV 상태)")
    print()

    # ── 1. 앙상블 모델 로드 ────────────────────────────────────────────────
    model = load_ensemble(args.model_path, args.device)

    # ── 2. 비용 함수 ───────────────────────────────────────────────────────
    cost_fn = CostFunctionEvaluator()
    output_dir = Path(args.output_dir)

    # ── 3. CEM 파라미터 구성 ───────────────────────────────────────────────
    planner_config = dict(PLANNER_CONFIG)
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

    # ── 4. 벤치마크 모드 ───────────────────────────────────────────────────
    if args.benchmark:
        run_benchmark_mode(
            model=model, cost_fn=cost_fn,
            n_scenarios=args.n_scenarios,
            sim_horizon=args.sim_horizon,
            output_dir=output_dir,
        )
        return

    # ── 5. 실제 데이터 기반 비교 모드 ─────────────────────────────────────
    if args.data_file:
        group_df = load_group_from_data(
            data_file=args.data_file,
            group_id=args.group_id,
            list_groups=args.list_groups,
        )
        state = build_state_from_group(group_df)
        current_state = {'current_clr': state['current_clr']}

        # CEM 최적화
        cem_result = run_cem(model, cost_fn, current_state, planner_config)

        # 실제 제어 비용 평가
        actual_cost, actual_bd = evaluate_actual_control(cost_fn, state)

        # CEM 예측 비용 평가
        cem_cost, cem_bd, cem_after_probs = evaluate_cem_prediction(
            model, cost_fn, state, cem_result.x_opt
        )

        # 비교 출력
        print_comparison(
            state=state,
            actual_cost=actual_cost,
            actual_bd=actual_bd,
            cem_cost=cem_cost,
            cem_bd=cem_bd,
            cem_after_probs=cem_after_probs,
            x_opt=cem_result.x_opt,
        )

        # 결과 저장
        output_dir.mkdir(parents=True, exist_ok=True)
        np.save(output_dir / "cem_x_opt.npy", cem_result.x_opt)
        print(f"  최적 제어값 저장: {output_dir / 'cem_x_opt.npy'}")
        return

    # ── 6. 기존 방식 (예시 / CSV CLR) ─────────────────────────────────────
    current_state = build_current_state(args.clr_file)
    result = run_cem(model, cost_fn, current_state, planner_config)

    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "cem_x_opt.npy", result.x_opt)
    print(f"\n  최적 제어값 저장: {output_dir / 'cem_x_opt.npy'}")


if __name__ == '__main__':
    main()
