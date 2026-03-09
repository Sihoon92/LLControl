"""
CEM (Cross Entropy Method) Planner for MBRL

PETS(Probabilistic Ensembles with Trajectory Sampling) 기반의
CEM 최적 제어 플래너

DE와의 핵심 차이:
- DE:  단일 스텝 예측 + 제어값 1개 최적화 (점 예측, 불확실성 없음)
- CEM: H-step 앞까지 롤아웃 시뮬레이션 + 불확실성 비용 반영

CEM 알고리즘:
  1. 제어 시퀀스 분포 N(μ, σ²) 초기화  (shape: horizon × 12)
  2. n_samples 개의 시퀀스 샘플링
  3. 각 시퀀스를 앙상블로 H-step 롤아웃 → 누적 비용 계산
  4. 비용 낮은 상위 n_elite개 선택 (elite)
  5. elite로 분포 업데이트 (momentum 적용)
  6. 2~5 반복 (n_iterations회)
  7. 최적 시퀀스의 첫 번째 스텝 제어값만 실행 (MPC 원칙)
"""

import numpy as np
import time
import logging
from typing import Dict, List, Tuple, Optional

from .dynamics_model import PerZoneProbabilisticEnsemble
from ..config import (
    N_ZONES, N_GV, N_CONTROL_VARS,
    get_bounds_array,
    GV_ADJACENT_MAX_DIFF, GV_TOTAL_CHANGE_MAX,
    RANDOM_SEED,
)
from ..cost_function import CostFunctionEvaluator
from ..model_interface import CatBoostModelManager
from ..optimizer_engine import OptimizationResult
from .config import PLANNER_CONFIG

logger = logging.getLogger(__name__)


class CEMPlanner:
    """
    Cross Entropy Method (CEM) 기반 최적 제어 플래너

    PerZoneProbabilisticEnsemble을 세계 모델(dynamics model)로 사용하여
    H-step 앞을 내다보는 최적 제어 시퀀스를 탐색한다.

    불확실성 활용:
    - 앙상블의 epistemic uncertainty를 비용에 반영
    - 학습 데이터 밖의 영역(불확실성 높음)을 자동으로 회피

    DE와 동일한 OptimizationResult를 반환하므로 drop-in replacement 가능.
    """

    def __init__(
        self,
        dynamics_model: PerZoneProbabilisticEnsemble,
        cost_evaluator: CostFunctionEvaluator,
        current_state: Dict[str, np.ndarray],
        planner_config: Optional[Dict] = None,
        random_seed: int = RANDOM_SEED,
    ):
        """
        Args:
            dynamics_model: 학습된 PETS 앙상블 모델
            cost_evaluator: 비용 함수 평가자 (DE와 동일 인스턴스 재사용 가능)
            current_state: {'current_clr': np.ndarray (11, 3)}
            planner_config: CEM 하이퍼파라미터 (None이면 config.py 기본값)
            random_seed: 재현성 시드
        """
        self.dynamics_model = dynamics_model
        self.cost_evaluator = cost_evaluator
        self.current_state = current_state

        # CEM 하이퍼파라미터 (PLANNER_CONFIG 기본값)
        cfg = planner_config or PLANNER_CONFIG
        self.horizon            = cfg['horizon']               # 5
        self.n_samples          = cfg['n_samples']             # 500
        self.n_elite            = cfg['n_elite']               # 50
        self.n_iterations       = cfg['n_iterations']          # 5
        self.alpha              = cfg['alpha']                 # 0.1  (momentum)
        self.uncertainty_penalty = cfg['uncertainty_penalty']  # 0.1
        self.init_mean          = cfg.get('init_mean', 0.0)
        self.init_std           = cfg.get('init_std', 0.5)

        # 경계값
        lower_list, upper_list = get_bounds_array()
        self.bounds_lower = np.array(lower_list, dtype=np.float64)  # (12,)
        self.bounds_upper = np.array(upper_list, dtype=np.float64)  # (12,)

        # 재현성
        self.rng = np.random.default_rng(random_seed)

        # 통계 추적
        self.eval_count = 0
        self.best_cost_history: List[float] = []

        logger.info("CEMPlanner 초기화")
        logger.info(f"  horizon={self.horizon}, n_samples={self.n_samples}, "
                    f"n_elite={self.n_elite}, n_iterations={self.n_iterations}")
        logger.info(f"  alpha={self.alpha}, uncertainty_penalty={self.uncertainty_penalty}")

    # =========================================================================
    # 제약 조건 검증 (DifferentialEvolutionOptimizer 와 동일 로직)
    # =========================================================================

    def check_constraints(self, x: np.ndarray) -> Tuple[bool, str]:
        """
        단일 제어값 벡터의 제약 조건 검증

        Args:
            x: (12,) - [ΔGV₁...₁₁, ΔRPM]

        Returns:
            (is_feasible, violation_message)
        """
        violations = []

        # 1. 경계값
        for i in range(N_CONTROL_VARS):
            if x[i] < self.bounds_lower[i] or x[i] > self.bounds_upper[i]:
                violations.append(
                    f"x[{i}]={x[i]:.3f} 범위 초과 "
                    f"[{self.bounds_lower[i]}, {self.bounds_upper[i]}]"
                )

        # 2. 인접 GV 차이
        delta_gv = x[:N_GV]
        for i in range(N_GV - 1):
            diff = abs(delta_gv[i] - delta_gv[i + 1])
            if diff > GV_ADJACENT_MAX_DIFF:
                violations.append(
                    f"|ΔGV[{i}]-ΔGV[{i+1}]|={diff:.3f} > {GV_ADJACENT_MAX_DIFF}"
                )

        # 3. 전체 변화량
        total = np.sum(np.abs(delta_gv))
        if total > GV_TOTAL_CHANGE_MAX:
            violations.append(f"Σ|ΔGV|={total:.3f} > {GV_TOTAL_CHANGE_MAX}")

        is_feasible = len(violations) == 0
        msg = " | ".join(violations) if violations else "제약 만족"
        return is_feasible, msg

    # =========================================================================
    # CEM 핵심 메서드
    # =========================================================================

    def _sample_action_sequences(
        self,
        mean: np.ndarray,   # (horizon, 12)
        std: np.ndarray,    # (horizon, 12)
    ) -> np.ndarray:        # (n_samples, horizon, 12)
        """
        현재 Gaussian 분포에서 액션 시퀀스 샘플링 후 bounds 클램핑
        """
        sequences = self.rng.normal(
            loc=mean,
            scale=std,
            size=(self.n_samples, self.horizon, N_CONTROL_VARS),
        )
        # delta_GV 정수 반올림 (학습 데이터가 정수값으로 구성되어 있으므로 OOD 방지)
        sequences[:, :, :N_GV] = np.round(sequences[:, :, :N_GV])
        # bounds 클램핑
        sequences = np.clip(sequences, self.bounds_lower, self.bounds_upper)
        return sequences  # (n_samples, horizon, 12)

    def _rollout_trajectory(
        self,
        action_sequence: np.ndarray,  # (horizon, 12)
        initial_clr: np.ndarray,      # (11, 3)
    ) -> Tuple[float, float]:
        """
        단일 액션 시퀀스로 H-step 롤아웃

        - 각 스텝에서 앙상블이 다음 상태(CLR)를 예측
        - 비용 + 불확실성 페널티를 horizon에 걸쳐 누적
        - 제약 위반 스텝에서 즉시 중단 (거부 방식)

        Returns:
            (total_cost, total_uncertainty)
        """
        clr_t = initial_clr.copy()
        total_cost = 0.0
        total_uncertainty = 0.0

        for t in range(self.horizon):
            action    = action_sequence[t]         # (12,)
            delta_gv  = action[:N_GV]              # (11,)
            delta_rpm = float(action[N_GV])        # scalar

            # 제약 위반 시 큰 페널티 + 롤아웃 중단 (거부 방식)
            is_feasible, _ = self.check_constraints(action)
            if not is_feasible:
                total_cost += 1e6
                break

            # 앙상블로 다음 상태 예측
            result = self.dynamics_model.predict_all_zones(
                current_clr_all=clr_t,
                delta_gv=delta_gv,
                delta_rpm=delta_rpm,
                return_uncertainty=True,
            )
            # result keys: 'diff_clr_mean'(11,3), 'diff_clr_uncertainty'(11,3), 'next_clr'(11,3)

            next_clr = result['next_clr']  # (11, 3)

            # CLR → 확률 변환 (static method, CatBoost 모델 불필요)
            probs  = CatBoostModelManager.inverse_clr(next_clr)  # (11, 3)
            p_low  = probs[:, 0]
            p_mid  = probs[:, 1]
            p_high = probs[:, 2]

            # 스텝 비용
            step_cost, _ = self.cost_evaluator.evaluate_total_cost(
                p_low, p_mid, p_high, delta_gv, delta_rpm
            )

            # 불확실성 페널티 (epistemic + aleatoric)
            if result['diff_clr_uncertainty'] is not None:
                step_unc   = float(result['diff_clr_uncertainty'].mean())
                step_cost += self.uncertainty_penalty * step_unc
                total_uncertainty += step_unc

            total_cost += step_cost
            clr_t = next_clr  # 상태 전이

        self.eval_count += 1
        return total_cost, total_uncertainty

    def _evaluate_batch_trajectories(
        self,
        action_sequences: np.ndarray,  # (n_samples, horizon, 12)
        initial_clr: np.ndarray,       # (11, 3)
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        모든 후보 시퀀스를 순차 평가

        Returns:
            costs:         (n_samples,)
            uncertainties: (n_samples,)
        """
        n = len(action_sequences)
        costs         = np.zeros(n)
        uncertainties = np.zeros(n)

        for i, seq in enumerate(action_sequences):
            costs[i], uncertainties[i] = self._rollout_trajectory(seq, initial_clr)

        return costs, uncertainties

    def _update_distribution(
        self,
        action_sequences: np.ndarray,  # (n_samples, horizon, 12)
        costs: np.ndarray,             # (n_samples,)
        current_mean: np.ndarray,      # (horizon, 12)
        current_std: np.ndarray,       # (horizon, 12)
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Elite 시퀀스로 분포 업데이트 (Momentum 블렌딩)

        1. 비용 낮은 상위 n_elite개 선택
        2. 새 mean, std 계산
        3. Momentum: new = α·old + (1-α)·elite_stat
        4. Distribution collapse 방지: std ≥ 1e-4
        """
        elite_idx  = np.argsort(costs)[: self.n_elite]
        elite_seqs = action_sequences[elite_idx]  # (n_elite, horizon, 12)

        new_mean = elite_seqs.mean(axis=0)         # (horizon, 12)
        new_std  = elite_seqs.std(axis=0)          # (horizon, 12)

        updated_mean = self.alpha * current_mean + (1 - self.alpha) * new_mean
        updated_std  = self.alpha * current_std  + (1 - self.alpha) * new_std

        # Distribution collapse 방지
        updated_std = np.maximum(updated_std, 1e-4)

        return updated_mean, updated_std

    # =========================================================================
    # 메인 최적화 엔진
    # =========================================================================

    def run_optimization(self) -> OptimizationResult:
        """
        CEM 최적화 실행

        DifferentialEvolutionOptimizer.run_optimization()과 동일한
        OptimizationResult를 반환 → drop-in replacement 가능

        Returns:
            OptimizationResult
        """
        logger.info("=" * 80)
        logger.info("CEM 최적화 시작")
        logger.info(f"  horizon={self.horizon}, n_samples={self.n_samples}, "
                    f"n_elite={self.n_elite}, n_iterations={self.n_iterations}")
        logger.info(f"  예상 총 롤아웃: {self.n_samples * self.n_iterations:,}회 "
                    f"(각 {self.horizon}스텝)")
        logger.info("=" * 80)

        start_time = time.time()
        self.eval_count = 0
        self.best_cost_history = []

        initial_clr = self.current_state['current_clr']  # (11, 3)

        # ── 분포 초기화 ──────────────────────────────────────────────────────
        mean = np.full((self.horizon, N_CONTROL_VARS), self.init_mean)
        std  = np.full((self.horizon, N_CONTROL_VARS), self.init_std,
                       dtype=np.float64)

        # std 상한: 제어 범위의 절반을 넘지 않도록
        action_range = (self.bounds_upper - self.bounds_lower) / 2.0
        std = np.minimum(std, action_range)

        best_cost     = float('inf')
        best_sequence = None

        # ── CEM 반복 ─────────────────────────────────────────────────────────
        for iteration in range(self.n_iterations):
            iter_start = time.time()

            # 1. 샘플링
            sequences = self._sample_action_sequences(mean, std)

            # 2. 평가
            costs, uncertainties = self._evaluate_batch_trajectories(
                sequences, initial_clr
            )

            # 3. 분포 업데이트
            mean, std = self._update_distribution(sequences, costs, mean, std)

            # 4. 이터레이션 최선값 추적
            iter_best_idx  = int(np.argmin(costs))
            iter_best_cost = float(costs[iter_best_idx])

            if iter_best_cost < best_cost:
                best_cost     = iter_best_cost
                best_sequence = sequences[iter_best_idx].copy()

            self.best_cost_history.append(best_cost)

            logger.info(
                f"  Iter [{iteration + 1}/{self.n_iterations}]  "
                f"Best={iter_best_cost:.6f}  "
                f"Mean={costs.mean():.6f}  "
                f"σ_mean={std.mean():.4f}  "
                f"Unc_mean={uncertainties.mean():.4f}  "
                f"Time={time.time() - iter_start:.2f}s"
            )

        elapsed = time.time() - start_time

        # ── 최적해 추출 (MPC: 첫 번째 스텝만 실행) ──────────────────────────
        x_opt = np.clip(best_sequence[0], self.bounds_lower, self.bounds_upper)

        delta_gv_opt  = x_opt[:N_GV]
        delta_rpm_opt = float(x_opt[N_GV])

        # 최종 비용 분석 (단일 스텝, breakdown용)
        result_t1 = self.dynamics_model.predict_all_zones(
            current_clr_all=initial_clr,
            delta_gv=delta_gv_opt,
            delta_rpm=delta_rpm_opt,
            return_uncertainty=True,
        )
        probs_t1 = CatBoostModelManager.inverse_clr(result_t1['next_clr'])
        _, final_breakdown = self.cost_evaluator.evaluate_total_cost(
            probs_t1[:, 0], probs_t1[:, 1], probs_t1[:, 2],
            delta_gv_opt, delta_rpm_opt,
        )

        logger.info("=" * 80)
        logger.info("CEM 최적화 완료")
        logger.info(f"  총 롤아웃 횟수: {self.eval_count:,}")
        logger.info(f"  소요 시간: {elapsed:.2f}초")
        logger.info(f"  최적 비용: {best_cost:.6f}")
        logger.info(f"  최적 ΔGV: {np.round(x_opt[:N_GV], 4)}")
        logger.info(f"  최적 ΔRPM: {x_opt[N_GV]:.2f}")
        logger.info("=" * 80)

        return OptimizationResult(
            x_opt=x_opt,
            cost_opt=best_cost,
            scipy_result=None,
            n_evaluations=self.eval_count,
            optimization_time=elapsed,
            success=True,
            message=f"CEM 수렴 완료 ({self.n_iterations} iterations, "
                    f"horizon={self.horizon})",
            final_cost_breakdown=final_breakdown,
        )
