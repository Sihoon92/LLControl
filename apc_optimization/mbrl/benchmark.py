"""
DE+ML vs CEM+MBRL 성능 비교 벤치마크

"정답 없는 최적 제어" 비교 방법론:
──────────────────────────────────
1. 단일 스텝 비용 (t=0):
   두 제어기가 내놓은 제어값을 동일한 앙상블로 평가 → 공정 비교

2. H-step 시뮬레이션 누적 비용:
   앙상블을 "가상 공장(Virtual Plant)"으로 사용
   DE 제어값 → 앙상블 H번 롤아웃 → 결과 측정
   CEM 제어값 → 앙상블 H번 롤아웃 → 결과 측정
   (CEM은 H-step을 미리 내다보고 최적화, DE는 1-step만 봄)

3. 다양한 초기 조건에서 강건성:
   balanced / p_low_heavy / p_high_heavy / random 4가지 유형의
   초기 CLR 상태로 양쪽을 실행 → 평균 성능 및 분산 비교

공정성 원칙:
- 두 제어기 모두 동일한 앙상블 모델로 결과 평가
- 동일한 초기 조건
- 동일한 비용 함수 (CostFunctionEvaluator)
"""

import numpy as np
import time
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

import pandas as pd

from .dynamics_model import PerZoneProbabilisticEnsemble
from .cem_planner import CEMPlanner
from ..optimizer_engine import DifferentialEvolutionOptimizer, OptimizationResult
from ..cost_function import CostFunctionEvaluator
from ..model_interface import CatBoostModelManager
from ..config import N_ZONES, N_GV, RANDOM_SEED

logger = logging.getLogger(__name__)


# =============================================================================
# 결과 데이터 클래스
# =============================================================================

@dataclass
class ScenarioResult:
    """단일 시나리오(초기 조건)에서의 DE vs CEM 비교 결과"""
    scenario_id: int
    scenario_type: str = ""

    # 단일 스텝 비용 (t=0)
    de_cost_t0: float = 0.0
    cem_cost_t0: float = 0.0

    # H-step 시뮬레이션 누적 비용
    de_simulated_cost_th: float = 0.0
    cem_simulated_cost_th: float = 0.0

    # H-step 후 P_Mid 품질
    de_p_mid_mean_final: float = 0.0
    cem_p_mid_mean_final: float = 0.0
    de_p_mid_std_final: float = 0.0    # 존 간 불균형 (낮을수록 좋음)
    cem_p_mid_std_final: float = 0.0

    # 누적 불확실성 (CEM은 이를 회피하도록 최적화됨)
    de_uncertainty_accumulated: float = 0.0
    cem_uncertainty_accumulated: float = 0.0

    # 계산 효율
    de_wall_time_s: float = 0.0
    cem_wall_time_s: float = 0.0
    de_n_evaluations: int = 0
    cem_n_evaluations: int = 0

    # 최적 제어값 (참고용)
    de_x_opt: Optional[np.ndarray] = field(default=None, repr=False)
    cem_x_opt: Optional[np.ndarray] = field(default=None, repr=False)


# =============================================================================
# 벤치마크 클래스
# =============================================================================

class CEMBenchmark:
    """
    DE+CatBoost vs CEM+MBRL 성능 비교 벤치마크

    사용 예시:
    ----------
    bench = CEMBenchmark(
        dynamics_model=trained_ensemble,
        cost_evaluator=cost_fn,
        model_manager=catboost_manager,  # DE용
    )
    df = bench.run(n_scenarios=20)
    bench.save_results(df, "outputs/mbrl/benchmark_results.csv")
    """

    def __init__(
        self,
        dynamics_model: PerZoneProbabilisticEnsemble,
        cost_evaluator: CostFunctionEvaluator,
        model_manager: Optional[CatBoostModelManager] = None,
        sim_horizon: int = 5,
        random_seed: int = RANDOM_SEED,
    ):
        """
        Args:
            dynamics_model: 학습된 PETS 앙상블 (가상 공장 + CEM 세계 모델 역할)
            cost_evaluator: 비용 함수 평가자 (DE/CEM 공통 사용)
            model_manager: DE에서 사용할 CatBoostModelManager (None이면 Mock)
            sim_horizon: H-step 시뮬레이션 스텝 수
            random_seed: 재현성 시드
        """
        self.dynamics_model = dynamics_model
        self.cost_evaluator = cost_evaluator
        self.model_manager  = model_manager or CatBoostModelManager()
        self.sim_horizon    = sim_horizon
        self.rng            = np.random.default_rng(random_seed)

        logger.info(f"CEMBenchmark 초기화 (sim_horizon={sim_horizon})")

    # =========================================================================
    # 시나리오 생성
    # =========================================================================

    def generate_test_scenarios(
        self,
        n_scenarios: int = 20,
        scenario_types: Optional[List[str]] = None,
    ) -> List[Dict]:
        """
        다양한 초기 CLR 상태 시나리오 생성

        시나리오 유형:
          - 'balanced'     : P_Mid ≈ 0.8, 균형 잡힌 상태
          - 'p_low_heavy'  : P_Low ≈ 0.5, 저밀도 편향 상태
          - 'p_high_heavy' : P_High ≈ 0.5, 고밀도 편향 상태
          - 'random'       : 랜덤 CLR

        Returns:
            list of {'id': int, 'type': str, 'current_clr': np.ndarray (11, 3)}
        """
        if scenario_types is None:
            scenario_types = ['balanced', 'p_low_heavy', 'p_high_heavy', 'random']

        scenarios: List[Dict] = []
        n_per_type = max(1, n_scenarios // len(scenario_types))

        for s_type in scenario_types:
            for _ in range(n_per_type):
                if len(scenarios) >= n_scenarios:
                    break

                if s_type == 'balanced':
                    p   = np.array([0.1, 0.8, 0.1])
                    clr = np.log(p) - np.mean(np.log(p))
                    noise = self.rng.normal(0, 0.1, (N_ZONES, 3))
                    current_clr = np.tile(clr, (N_ZONES, 1)) + noise

                elif s_type == 'p_low_heavy':
                    p   = np.array([0.5, 0.3, 0.2])
                    clr = np.log(p) - np.mean(np.log(p))
                    noise = self.rng.normal(0, 0.15, (N_ZONES, 3))
                    current_clr = np.tile(clr, (N_ZONES, 1)) + noise

                elif s_type == 'p_high_heavy':
                    p   = np.array([0.2, 0.3, 0.5])
                    clr = np.log(p) - np.mean(np.log(p))
                    noise = self.rng.normal(0, 0.15, (N_ZONES, 3))
                    current_clr = np.tile(clr, (N_ZONES, 1)) + noise

                else:  # random
                    current_clr = self.rng.normal(0, 1.0, (N_ZONES, 3))

                scenarios.append({
                    'id':          len(scenarios),
                    'type':        s_type,
                    'current_clr': current_clr.astype(np.float32),
                })

            if len(scenarios) >= n_scenarios:
                break

        logger.info(f"시나리오 생성: {len(scenarios)}개 "
                    f"(types={scenario_types})")
        return scenarios

    # =========================================================================
    # 시뮬레이션 유틸리티
    # =========================================================================

    def _get_single_step_cost(
        self,
        initial_clr: np.ndarray,  # (11, 3)
        x_opt: np.ndarray,        # (12,)
    ) -> float:
        """
        단일 스텝 제어 결과 비용 계산
        앙상블(가상 공장)으로 평가 → 두 제어기 공통 기준
        """
        delta_gv  = x_opt[:N_GV]
        delta_rpm = float(x_opt[N_GV])

        result = self.dynamics_model.predict_all_zones(
            current_clr_all=initial_clr,
            delta_gv=delta_gv,
            delta_rpm=delta_rpm,
            return_uncertainty=False,
        )
        probs = CatBoostModelManager.inverse_clr(result['next_clr'])
        cost, _ = self.cost_evaluator.evaluate_total_cost(
            probs[:, 0], probs[:, 1], probs[:, 2], delta_gv, delta_rpm
        )
        return float(cost)

    def _simulate_h_steps(
        self,
        initial_clr: np.ndarray,  # (11, 3)
        x_opt: np.ndarray,        # (12,) — 동일 제어값 반복 적용
    ) -> Dict:
        """
        앙상블을 가상 공장으로 사용한 H-step 시뮬레이션

        동일한 x_opt를 H번 반복 적용하여 누적 비용, 최종 품질 측정.
        (DE는 단순 비교용이므로 동일 제어값 반복, CEM도 동일 조건 적용)

        Returns:
            dict: total_cost, total_uncertainty,
                  final_p_mid_mean, final_p_mid_std, final_clr
        """
        clr_t         = initial_clr.copy()
        total_cost    = 0.0
        total_uncertainty = 0.0
        delta_gv      = x_opt[:N_GV]
        delta_rpm     = float(x_opt[N_GV])

        for _ in range(self.sim_horizon):
            result = self.dynamics_model.predict_all_zones(
                current_clr_all=clr_t,
                delta_gv=delta_gv,
                delta_rpm=delta_rpm,
                return_uncertainty=True,
            )
            next_clr = result['next_clr']
            probs    = CatBoostModelManager.inverse_clr(next_clr)

            step_cost, _ = self.cost_evaluator.evaluate_total_cost(
                probs[:, 0], probs[:, 1], probs[:, 2], delta_gv, delta_rpm
            )
            total_cost += step_cost

            if result['diff_clr_uncertainty'] is not None:
                total_uncertainty += float(result['diff_clr_uncertainty'].mean())

            clr_t = next_clr

        # 최종 상태 P_Mid 통계
        final_probs = CatBoostModelManager.inverse_clr(clr_t)
        final_p_mid = final_probs[:, 1]  # (11,)

        return {
            'total_cost':          total_cost,
            'total_uncertainty':   total_uncertainty,
            'final_p_mid_mean':    float(final_p_mid.mean()),
            'final_p_mid_std':     float(final_p_mid.std()),
            'final_clr':           clr_t,
        }

    # =========================================================================
    # 단일 시나리오 비교
    # =========================================================================

    def _run_single_scenario(
        self,
        scenario: Dict,
        cem_planner_config: Optional[Dict] = None,
        de_optimizer_params: Optional[Dict] = None,
    ) -> ScenarioResult:
        """단일 시나리오에서 DE와 CEM 실행 및 비교"""
        current_state = {'current_clr': scenario['current_clr']}
        initial_clr   = scenario['current_clr']
        res = ScenarioResult(
            scenario_id=scenario['id'],
            scenario_type=scenario['type'],
        )

        # ── DE 실행 ───────────────────────────────────────────────────────────
        logger.info(f"  DE 실행 (scenario {scenario['id']}, type={scenario['type']})...")
        t0 = time.time()
        try:
            de_opt    = DifferentialEvolutionOptimizer(
                model_manager=self.model_manager,
                cost_evaluator=self.cost_evaluator,
                current_state=current_state,
                optimizer_params=de_optimizer_params,
            )
            de_result = de_opt.run_optimization()
            res.de_wall_time_s    = time.time() - t0
            res.de_n_evaluations  = de_result.n_evaluations
            res.de_x_opt          = de_result.x_opt

            # 단일 스텝 비용 (앙상블 평가)
            res.de_cost_t0 = self._get_single_step_cost(initial_clr, de_result.x_opt)

            # H-step 시뮬레이션
            de_sim = self._simulate_h_steps(initial_clr, de_result.x_opt)
            res.de_simulated_cost_th       = de_sim['total_cost']
            res.de_p_mid_mean_final        = de_sim['final_p_mid_mean']
            res.de_p_mid_std_final         = de_sim['final_p_mid_std']
            res.de_uncertainty_accumulated = de_sim['total_uncertainty']

        except Exception as e:
            logger.error(f"  DE 실패: {e}")
            res.de_cost_t0            = 1e6
            res.de_simulated_cost_th  = 1e6

        # ── CEM 실행 ──────────────────────────────────────────────────────────
        logger.info(f"  CEM 실행 (scenario {scenario['id']})...")
        t0 = time.time()
        try:
            cem       = CEMPlanner(
                dynamics_model=self.dynamics_model,
                cost_evaluator=self.cost_evaluator,
                current_state=current_state,
                planner_config=cem_planner_config,
            )
            cem_result = cem.run_optimization()
            res.cem_wall_time_s    = time.time() - t0
            res.cem_n_evaluations  = cem_result.n_evaluations
            res.cem_x_opt          = cem_result.x_opt

            # 단일 스텝 비용 (앙상블 평가)
            res.cem_cost_t0 = self._get_single_step_cost(initial_clr, cem_result.x_opt)

            # H-step 시뮬레이션
            cem_sim = self._simulate_h_steps(initial_clr, cem_result.x_opt)
            res.cem_simulated_cost_th       = cem_sim['total_cost']
            res.cem_p_mid_mean_final        = cem_sim['final_p_mid_mean']
            res.cem_p_mid_std_final         = cem_sim['final_p_mid_std']
            res.cem_uncertainty_accumulated = cem_sim['total_uncertainty']

        except Exception as e:
            logger.error(f"  CEM 실패: {e}")
            res.cem_cost_t0            = 1e6
            res.cem_simulated_cost_th  = 1e6

        return res

    # =========================================================================
    # 전체 벤치마크 실행
    # =========================================================================

    def run(
        self,
        n_scenarios: int = 20,
        scenario_types: Optional[List[str]] = None,
        cem_planner_config: Optional[Dict] = None,
        de_optimizer_params: Optional[Dict] = None,
    ) -> pd.DataFrame:
        """
        전체 벤치마크 실행

        Args:
            n_scenarios:         비교할 시나리오 수
            scenario_types:      시나리오 유형 필터 (None이면 전체 4종)
            cem_planner_config:  CEM 하이퍼파라미터 오버라이드
            de_optimizer_params: DE 하이퍼파라미터 오버라이드

        Returns:
            pd.DataFrame: 시나리오별 비교 결과 (행=시나리오, 열=지표)
        """
        logger.info("=" * 80)
        logger.info("DE+ML  vs  CEM+MBRL  벤치마크 시작")
        logger.info(f"  시나리오 수: {n_scenarios}, sim_horizon: {self.sim_horizon}")
        logger.info("=" * 80)

        scenarios = self.generate_test_scenarios(n_scenarios, scenario_types)
        results: List[ScenarioResult] = []

        for i, scenario in enumerate(scenarios):
            logger.info(f"\n[{i + 1}/{len(scenarios)}] "
                        f"시나리오 {scenario['id']} (type={scenario['type']})")
            r = self._run_single_scenario(
                scenario, cem_planner_config, de_optimizer_params
            )
            results.append(r)

        df = self._to_dataframe(results)
        self.print_summary(df)
        return df

    # =========================================================================
    # 결과 처리
    # =========================================================================

    def _to_dataframe(self, results: List[ScenarioResult]) -> pd.DataFrame:
        """ScenarioResult 리스트 → DataFrame"""
        rows = []
        for r in results:
            de_t0  = r.de_cost_t0
            cem_t0 = r.cem_cost_t0
            de_th  = r.de_simulated_cost_th
            cem_th = r.cem_simulated_cost_th
            rows.append({
                'scenario_id':                  r.scenario_id,
                'scenario_type':                r.scenario_type,

                # 단일 스텝 비용
                'de_cost_t0':                   de_t0,
                'cem_cost_t0':                  cem_t0,
                'cost_t0_improvement_%':
                    (de_t0 - cem_t0) / (de_t0 + 1e-9) * 100,

                # H-step 누적 비용
                'de_simulated_cost_th':         de_th,
                'cem_simulated_cost_th':        cem_th,
                'cost_th_improvement_%':
                    (de_th - cem_th) / (de_th + 1e-9) * 100,

                # 최종 P_Mid 품질
                'de_p_mid_mean_final':          r.de_p_mid_mean_final,
                'cem_p_mid_mean_final':         r.cem_p_mid_mean_final,
                'de_p_mid_std_final':           r.de_p_mid_std_final,
                'cem_p_mid_std_final':          r.cem_p_mid_std_final,

                # 불확실성 누적
                'de_uncertainty_accumulated':   r.de_uncertainty_accumulated,
                'cem_uncertainty_accumulated':  r.cem_uncertainty_accumulated,

                # 계산 비용
                'de_wall_time_s':               r.de_wall_time_s,
                'cem_wall_time_s':              r.cem_wall_time_s,
                'de_n_evaluations':             r.de_n_evaluations,
                'cem_n_evaluations':            r.cem_n_evaluations,
            })
        return pd.DataFrame(rows)

    def print_summary(self, df: pd.DataFrame):
        """벤치마크 집계 결과 출력"""
        logger.info("\n" + "=" * 80)
        logger.info("벤치마크 결과 요약")
        logger.info("=" * 80)

        def _fmt(col):
            return f"{df[col].mean():.4f} ± {df[col].std():.4f}"

        logger.info("\n[1] 단일 스텝 비용 (t=0)  — 동일 앙상블로 평가")
        logger.info(f"    DE  : {_fmt('de_cost_t0')}")
        logger.info(f"    CEM : {_fmt('cem_cost_t0')}")
        logger.info(f"    CEM 개선율: {df['cost_t0_improvement_%'].mean():.2f}%")

        logger.info(f"\n[2] H-step 시뮬레이션 누적 비용 (H={self.sim_horizon})")
        logger.info(f"    DE  : {_fmt('de_simulated_cost_th')}")
        logger.info(f"    CEM : {_fmt('cem_simulated_cost_th')}")
        logger.info(f"    CEM 개선율: {df['cost_th_improvement_%'].mean():.2f}%")

        logger.info(f"\n[3] 최종 P_Mid 품질 (H스텝 후)")
        logger.info(f"    DE  P_Mid 평균: {df['de_p_mid_mean_final'].mean():.4f} "
                    f"(존 불균형 std={df['de_p_mid_std_final'].mean():.4f})")
        logger.info(f"    CEM P_Mid 평균: {df['cem_p_mid_mean_final'].mean():.4f} "
                    f"(존 불균형 std={df['cem_p_mid_std_final'].mean():.4f})")

        logger.info(f"\n[4] 경로 불확실성 누적 (낮을수록 안전한 제어)")
        logger.info(f"    DE  : {df['de_uncertainty_accumulated'].mean():.4f}")
        logger.info(f"    CEM : {df['cem_uncertainty_accumulated'].mean():.4f}")

        logger.info(f"\n[5] 계산 비용")
        logger.info(f"    DE  : 평균 {df['de_wall_time_s'].mean():.2f}s  "
                    f"({df['de_n_evaluations'].mean():.0f}회 평가)")
        logger.info(f"    CEM : 평균 {df['cem_wall_time_s'].mean():.2f}s  "
                    f"({df['cem_n_evaluations'].mean():.0f}회 롤아웃)")

        logger.info("\n[6] 시나리오 유형별 요약")
        for s_type in df['scenario_type'].unique():
            sub = df[df['scenario_type'] == s_type]
            logger.info(f"    [{s_type}] n={len(sub)}  "
                        f"CEM t=0 개선: {sub['cost_t0_improvement_%'].mean():.2f}%  "
                        f"CEM H-step 개선: {sub['cost_th_improvement_%'].mean():.2f}%")

        logger.info("=" * 80)

    @staticmethod
    def save_results(df: pd.DataFrame, path: str):
        """벤치마크 결과 CSV 저장"""
        df.to_csv(path, index=False)
        logger.info(f"벤치마크 결과 저장: {path}")


# =============================================================================
# 간편 실행 함수
# =============================================================================

def run_benchmark(
    dynamics_model: PerZoneProbabilisticEnsemble,
    cost_evaluator: CostFunctionEvaluator,
    model_manager: Optional[CatBoostModelManager] = None,
    n_scenarios: int = 20,
    sim_horizon: int = 5,
    output_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    벤치마크 간편 실행 래퍼

    Args:
        dynamics_model: 학습된 PETS 앙상블
        cost_evaluator: 비용 함수 평가자
        model_manager:  DE용 CatBoost 모델 (None이면 Mock)
        n_scenarios:    비교할 시나리오 수
        sim_horizon:    H-step 시뮬레이션 스텝 수
        output_path:    CSV 저장 경로 (None이면 저장 안 함)

    Returns:
        pd.DataFrame: 벤치마크 결과
    """
    bench = CEMBenchmark(
        dynamics_model=dynamics_model,
        cost_evaluator=cost_evaluator,
        model_manager=model_manager,
        sim_horizon=sim_horizon,
    )
    df = bench.run(n_scenarios=n_scenarios)

    if output_path:
        CEMBenchmark.save_results(df, output_path)

    return df
