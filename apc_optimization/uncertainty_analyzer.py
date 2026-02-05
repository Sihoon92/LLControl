"""
불확실성 분석기 (Uncertainty Analyzer)

Monte Carlo Simulation을 통한 최적화 결과의 신뢰도 평가

불확실성 소스:
1. 모델 예측 오차 (학습 데이터 기반 RMSE)
2. 측정 오차 (±5% relative noise)

프로세스:
1. 최적해 x*에서의 예측값 계산
2. N번 반복:
   a. 모델 예측 오차 추가: pred' = pred + δ_pred ~ N(0, σ²)
   b. 측정 오차 추가: prob' = prob + δ_meas ~ N(0, 0.05²)
   c. 비용 재계산
3. 통계 분석: 평균, 표준편차, 신뢰구간
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from scipy import stats

from .config import MC_SIMULATION_PARAMS, N_ZONES
from .cost_function import CostFunctionEvaluator
from .multi_zone_controller import MultiZoneController
from .model_interface import CatBoostModelManager

logger = logging.getLogger(__name__)


@dataclass
class MonteCarloResults:
    """Monte Carlo 시뮬레이션 결과"""
    x_opt: np.ndarray                      # 최적해
    base_cost: float                       # 기본 비용값

    # 확률 분포
    p_low_samples: np.ndarray              # Shape (n_sims, n_zones)
    p_mid_samples: np.ndarray              # Shape (n_sims, n_zones)
    p_high_samples: np.ndarray             # Shape (n_sims, n_zones)

    # 비용 분포
    cost_samples: np.ndarray               # Shape (n_sims,)

    # 통계
    statistics: Dict = field(default_factory=dict)

    n_simulations: int = 100

    def get_p_mid_stats(self) -> Dict:
        """P_Mid의 통계"""
        p_mid_mean = np.mean(self.p_mid_samples, axis=0)  # Zone별 평균
        p_mid_std = np.std(self.p_mid_samples, axis=0)    # Zone별 표준편차

        return {
            'mean': p_mid_mean,
            'std': p_mid_std,
            'global_mean': np.mean(p_mid_mean),
            'global_std': np.mean(p_mid_std),
        }

    def get_cost_stats(self) -> Dict:
        """비용의 통계"""
        costs = self.cost_samples

        ci_lower = np.percentile(costs, 2.5)
        ci_upper = np.percentile(costs, 97.5)

        return {
            'mean': np.mean(costs),
            'std': np.std(costs),
            'min': np.min(costs),
            'max': np.max(costs),
            'median': np.median(costs),
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'ci_width': ci_upper - ci_lower,
        }


class MonteCarloUncertaintyAnalyzer:
    """
    Monte Carlo 불확실성 분석기
    """

    def __init__(self,
                 model_manager: CatBoostModelManager,
                 cost_evaluator: CostFunctionEvaluator,
                 controller: MultiZoneController,
                 mc_params: Optional[Dict] = None):
        """
        초기화

        Args:
            model_manager: CatBoostModelManager
            cost_evaluator: CostFunctionEvaluator
            controller: MultiZoneController
            mc_params: Monte Carlo 파라미터 (없으면 config 사용)
        """
        self.model = model_manager
        self.cost_evaluator = cost_evaluator
        self.controller = controller
        self.mc_params = mc_params or MC_SIMULATION_PARAMS

        self.n_simulations = self.mc_params.get('n_simulations', 100)
        self.model_rmse = self.mc_params.get('model_rmse', 0.05)
        self.measurement_noise_ratio = self.mc_params.get('measurement_noise_ratio', 0.05)
        self.confidence_level = self.mc_params.get('confidence_level', 0.95)

        logger.info(f"MonteCarloUncertaintyAnalyzer 초기화")
        logger.info(f"  시뮬레이션 횟수: {self.n_simulations}")
        logger.info(f"  모델 RMSE: {self.model_rmse:.4f}")
        logger.info(f"  측정 오차: {self.measurement_noise_ratio*100:.1f}%")
        logger.info(f"  신뢰도: {self.confidence_level*100:.1f}%")

    # ========================================================================
    # Monte Carlo 시뮬레이션
    # ========================================================================

    def run_simulations(self,
                       x_opt: np.ndarray,
                       current_state: Dict,
                       n_simulations: Optional[int] = None) -> MonteCarloResults:
        """
        Monte Carlo 시뮬레이션 실행

        Args:
            x_opt: Shape (12,) - 최적해
            current_state: 현재 상태 dict
            n_simulations: 시뮬레이션 횟수 (없으면 기본값)

        Returns:
            MonteCarloResults
        """
        n_sims = n_simulations or self.n_simulations

        logger.info("="*80)
        logger.info(f"Monte Carlo 시뮬레이션 시작 (N={n_sims})")
        logger.info("="*80)

        # 기본 예측 (오차 없음)
        base_control = self.controller.evaluate_control(x_opt, current_state)
        base_p_low = base_control['p_low']
        base_p_mid = base_control['p_mid']
        base_p_high = base_control['p_high']

        # 기본 비용
        _, base_breakdown = self.cost_evaluator.evaluate_total_cost(
            base_p_low, base_p_mid, base_p_high,
            x_opt[:11], x_opt[11]
        )
        base_cost = base_breakdown['total_cost']

        # 샘플 저장소
        p_low_samples = np.zeros((n_sims, N_ZONES))
        p_mid_samples = np.zeros((n_sims, N_ZONES))
        p_high_samples = np.zeros((n_sims, N_ZONES))
        cost_samples = np.zeros(n_sims)

        # 시뮬레이션
        for i in range(n_sims):
            # 1. 모델 예측 오차 추가
            # △CLR에 가우시안 노이즈 추가
            delta_clr_noisy = base_control['delta_clr'].copy()
            delta_clr_noisy += np.random.normal(0, self.model_rmse, delta_clr_noisy.shape)

            # 2. Inverse CLR (noise 포함)
            current_clr = current_state.get('current_clr', np.zeros((N_ZONES, 3)))
            new_clr = current_clr + delta_clr_noisy
            probabilities_noisy = self.model.apply_inverse_clr_transform(
                current_clr, delta_clr_noisy
            )

            # 3. 측정 오차 추가 (relative noise)
            probabilities_noisy += np.random.normal(
                0, self.measurement_noise_ratio, probabilities_noisy.shape
            )

            # 4. 확률 정규화 (합=1)
            probabilities_noisy = np.clip(probabilities_noisy, 0, 1)
            prob_sum = np.sum(probabilities_noisy, axis=1, keepdims=True)
            prob_sum = np.where(prob_sum > 0, prob_sum, 1.0)
            probabilities_noisy = probabilities_noisy / prob_sum

            p_low_noisy = probabilities_noisy[:, 0]
            p_mid_noisy = probabilities_noisy[:, 1]
            p_high_noisy = probabilities_noisy[:, 2]

            # 5. 비용 재계산
            _, breakdown = self.cost_evaluator.evaluate_total_cost(
                p_low_noisy, p_mid_noisy, p_high_noisy,
                x_opt[:11], x_opt[11]
            )

            # 저장
            p_low_samples[i] = p_low_noisy
            p_mid_samples[i] = p_mid_noisy
            p_high_samples[i] = p_high_noisy
            cost_samples[i] = breakdown['total_cost']

            # 진행률 출력
            if (i + 1) % (n_sims // 10) == 0 or i == 0:
                logger.info(f"  [{i+1:4d}/{n_sims}] Cost: {breakdown['total_cost']:.6f}, "
                           f"P_Mid: {np.mean(p_mid_noisy):.4f}")

        # 결과 생성
        result = MonteCarloResults(
            x_opt=x_opt,
            base_cost=base_cost,
            p_low_samples=p_low_samples,
            p_mid_samples=p_mid_samples,
            p_high_samples=p_high_samples,
            cost_samples=cost_samples,
            n_simulations=n_sims,
        )

        # 통계 계산
        result.statistics = self._compute_statistics(result)

        logger.info("="*80)
        logger.info("시뮬레이션 완료")
        logger.info("="*80)

        return result

    # ========================================================================
    # 통계 분석
    # ========================================================================

    def _compute_statistics(self, results: MonteCarloResults) -> Dict:
        """
        통계 계산
        """
        p_mid_stats = results.get_p_mid_stats()
        cost_stats = results.get_cost_stats()

        # 제약 위반 확률 (P_Mid < 0.8을 실패로 간주)
        success_threshold = 0.8
        success_rate = np.mean(results.p_mid_samples > success_threshold)

        # P_Mid 균일성 (표준편차 작을수록 좋음)
        p_mid_uniformity = np.mean(p_mid_stats['std'])

        statistics = {
            'p_mid': p_mid_stats,
            'cost': cost_stats,
            'success_rate': success_rate,           # P_Mid > 0.8 달성 비율
            'p_mid_uniformity': p_mid_uniformity,  # Zone 간 균일성
            'confidence_level': self.confidence_level,
        }

        return statistics

    # ========================================================================
    # 결과 시각화용 출력
    # ========================================================================

    def print_simulation_summary(self, results: MonteCarloResults) -> str:
        """
        시뮬레이션 결과 요약 출력
        """
        stats = results.statistics
        p_mid_stats = stats['p_mid']
        cost_stats = stats['cost']

        summary = f"""
        ============================================================
        Monte Carlo 시뮬레이션 결과 (N={results.n_simulations})
        ============================================================

        기본 비용: {results.base_cost:.6f}

        P_Mid 분석:
          - 평균: {p_mid_stats['global_mean']:.4f}
          - 표준편차: {p_mid_stats['global_std']:.4f}
          - 균일성 (Zone σ 평균): {stats['p_mid_uniformity']:.4f}
          - 성공률 (P_Mid > 0.8): {stats['success_rate']*100:.1f}%

        비용 분포:
          - 평균: {cost_stats['mean']:.6f}
          - 표준편차: {cost_stats['std']:.6f}
          - 95% 신뢰구간: [{cost_stats['ci_lower']:.6f}, {cost_stats['ci_upper']:.6f}]
          - 폭: {cost_stats['ci_width']:.6f}
          - 범위: [{cost_stats['min']:.6f}, {cost_stats['max']:.6f}]

        Zone별 P_Mid 통계:
        """

        # Zone별 상세 통계
        for i in range(min(N_ZONES, 5)):  # 처음 5개 Zone만 표시
            mean = p_mid_stats['mean'][i]
            std = p_mid_stats['std'][i]
            summary += f"\n          Zone{i+1:02d}: {mean:.4f} ± {std:.4f}"

        if N_ZONES > 5:
            summary += f"\n          ... (총 {N_ZONES}개 Zone)"

        summary += f"""

        ============================================================
        """

        return summary

    def get_constraint_violation_probability(self,
                                            results: MonteCarloResults,
                                            p_mid_threshold: float = 0.8) -> Dict:
        """
        제약 위반 확률 계산

        Args:
            results: MonteCarloResults
            p_mid_threshold: P_Mid 하한값

        Returns:
            dict: 위반 확률 통계
        """
        # P_Mid 위반 (한 개 이상의 Zone이 임계값 미만)
        p_mid_violations = np.any(results.p_mid_samples < p_mid_threshold, axis=1)
        p_mid_violation_prob = np.mean(p_mid_violations)

        # 비용 상한 초과 (기본값에서 25% 이상 증가)
        cost_threshold = results.base_cost * 1.25
        cost_violations = results.cost_samples > cost_threshold
        cost_violation_prob = np.mean(cost_violations)

        violation_stats = {
            'p_mid_violation_prob': p_mid_violation_prob,
            'cost_violation_prob': cost_violation_prob,
            'joint_violation_prob': np.mean(p_mid_violations & cost_violations),
            'base_cost': results.base_cost,
            'cost_threshold': cost_threshold,
        }

        return violation_stats


# ============================================================================
# 테스트 코드
# ============================================================================

if __name__ == '__main__':
    import sys
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    from .model_interface import CatBoostModelManager
    from .cost_function import CostFunctionEvaluator
    from .multi_zone_controller import MultiZoneController

    # 초기화
    model_manager = CatBoostModelManager()
    cost_evaluator = CostFunctionEvaluator()
    controller = MultiZoneController(model_manager)

    # 불확실성 분석기
    analyzer = MonteCarloUncertaintyAnalyzer(
        model_manager, cost_evaluator, controller,
        mc_params={'n_simulations': 50}  # 빠른 테스트
    )

    # 최적해 (임의)
    x_opt = np.concatenate([
        np.random.uniform(-0.5, 0.5, 11),
        np.array([10])
    ])

    # 현재 상태
    current_state = {
        'current_clr': np.random.randn(N_ZONES, 3)
    }

    # 시뮬레이션
    results = analyzer.run_simulations(x_opt, current_state)

    # 결과 출력
    print(analyzer.print_simulation_summary(results))

    # 위반 확률
    violation_stats = analyzer.get_constraint_violation_probability(results)
    print(f"\n제약 위반 확률:")
    print(f"  P_Mid < 0.8: {violation_stats['p_mid_violation_prob']*100:.1f}%")
    print(f"  Cost > 1.25×기본값: {violation_stats['cost_violation_prob']*100:.1f}%")
