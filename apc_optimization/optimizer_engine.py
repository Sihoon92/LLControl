"""
최적화 엔진 (Optimization Engine)

Differential Evolution (DE) 알고리즘을 사용하여 최적 제어값을 탐색

목적함수 최소화:
  min_x f(x) = Cost_Function(x)
  where x = [△GV₁, ..., △GV₁₁, △RPM]

제약조건:
  - 경계값: △GV ∈ [-2, 2], △RPM ∈ [-50, 50]
  - 인접 차이: |△GV_i - △GV_(i±1)| ≤ 0.5
  - 총 변화량: Σ|△GV_i| ≤ 10.0
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Callable, Any
from dataclasses import dataclass, field
from scipy.optimize import differential_evolution, OptimizeResult
import time

from .config import (
    N_ZONES, N_GV, N_CONTROL_VARS,
    get_bounds_array, DE_OPTIMIZER_PARAMS, CONSTRAINT_PARAMS,
    GV_ADJACENT_MAX_DIFF, GV_TOTAL_CHANGE_MAX, RANDOM_SEED
)
from .cost_function import CostFunctionEvaluator
from .multi_zone_controller import MultiZoneController
from .model_interface import CatBoostModelManager
from .output_transformer import OutputTransformer, TransformConfig
from .normalizer import ControlVariableNormalizer
from .config import OUTPUT_TRANSFORM_CONFIG

logger = logging.getLogger(__name__)


@dataclass
class OptimizationResult:
    """최적화 결과 데이터 클래스"""
    x_opt: np.ndarray                    # 최적해 [△GV₁~₁₁, △RPM]
    cost_opt: float                      # 최적 비용값
    scipy_result: Optional[OptimizeResult] = None  # scipy 결과 객체
    n_evaluations: int = 0               # 평가 횟수
    optimization_time: float = 0.0       # 최적화 소요 시간
    success: bool = False                # 수렴 성공 여부
    message: str = ""                    # 수렴 메시지
    final_cost_breakdown: Dict = field(default_factory=dict)  # 최종 비용 분석


class DifferentialEvolutionOptimizer:
    """
    Differential Evolution 최적화 엔진

    최적해 탐색 프로세스:
    1. 제어값 후보 생성 [△GV₁~₁₁, △RPM]
    2. Multi-zone fan-out: 11개 Zone 입력 구성
    3. CatBoost 배치 예측: △CLR 예측
    4. Inverse CLR: 확률 분포로 변환
    5. Cost 평가: 4개 비용 항목 계산
    6. 진화: DE 알고리즘으로 다음 세대 생성
    7. 반복: 수렴까지
    """

    def __init__(self,
                 model_manager: CatBoostModelManager,
                 cost_evaluator: CostFunctionEvaluator,
                 current_state: Dict[str, np.ndarray],
                 optimizer_params: Optional[Dict] = None,
                 constraint_params: Optional[Dict] = None):
        """
        초기화

        Args:
            model_manager: CatBoostModelManager 인스턴스
            cost_evaluator: CostFunctionEvaluator 인스턴스
            current_state: 현재 공정 상태 dict
            optimizer_params: DE 파라미터 (없으면 config 기본값)
            constraint_params: 제약 파라미터 (없으면 config 기본값)
        """
        self.model = model_manager
        self.cost_evaluator = cost_evaluator
        self.current_state = current_state
        self.optimizer_params = optimizer_params or DE_OPTIMIZER_PARAMS
        self.constraint_params = constraint_params or CONSTRAINT_PARAMS

        # Multi-zone 제어기
        self.controller = MultiZoneController(model_manager)

        # 통합 정규화 클래스 초기화 (model_manager의 scaler 포함)
        # ★ Phase 3: 예측 모델과 비용 함수의 정규화 일관성 보장
        self.normalizer = ControlVariableNormalizer(
            gv_max=cost_evaluator.normalizer.gv_max,
            rpm_max=cost_evaluator.normalizer.rpm_max,
            scaler=model_manager.scaler  # ★ StandardScaler 전달
        )
        logger.info(f"✓ 통합 정규화 클래스 초기화: "
                   f"{self.normalizer.get_description().split(chr(10))[0]}")

        # 출력 변환기 (정수화)
        transform_config = TransformConfig(**OUTPUT_TRANSFORM_CONFIG)
        self.output_transformer = OutputTransformer(transform_config)

        # 최적화 히스토리
        self.evaluation_history = []
        self.best_cost_history = []
        self.eval_count = 0

        # 경계값 설정
        self.bounds_lower, self.bounds_upper = get_bounds_array()

        logger.info(f"Differential Evolution Optimizer 초기화")
        logger.info(f"제어 변수: {N_CONTROL_VARS}개")
        logger.info(f"경계: GV [{self.bounds_lower[0]}, {self.bounds_upper[0]}], "
                    f"RPM [{self.bounds_lower[-1]}, {self.bounds_upper[-1]}]")

        # 출력 변환기 상태
        if self.output_transformer.config.enable:
            logger.info(f"✓ 출력 변환 활성화: GV {self.output_transformer.config.delta_gv_method}() "
                       f"decimals={self.output_transformer.config.delta_gv_decimals}")

    # ========================================================================
    # 제약 조건 검증
    # ========================================================================

    def check_constraints(self, x: np.ndarray) -> Tuple[bool, str]:
        """
        제약 조건 검증

        Args:
            x: Shape (12,) - [△GV₁, ..., △GV₁₁, △RPM]

        Returns:
            (is_feasible, violation_message)
        """
        violations = []

        # 1. 경계값 검사
        if self.constraint_params.get('enforce_bounds', True):
            for i in range(N_CONTROL_VARS):
                if x[i] < self.bounds_lower[i] or x[i] > self.bounds_upper[i]:
                    violations.append(f"x[{i}]={x[i]:.3f} 범위 초과 "
                                    f"[{self.bounds_lower[i]}, {self.bounds_upper[i]}]")

        # 2. 인접 GV 차이 검사
        if self.constraint_params.get('enforce_adjacent', True):
            delta_gv = x[:N_GV]
            for i in range(N_GV - 1):
                diff = abs(delta_gv[i] - delta_gv[i+1])
                if diff > GV_ADJACENT_MAX_DIFF:
                    violations.append(f"인접 차이 위반: |△GV[{i}] - △GV[{i+1}]| = {diff:.3f} > {GV_ADJACENT_MAX_DIFF}")

        # 3. 전체 GV 변화량 검사
        if self.constraint_params.get('enforce_total', True):
            delta_gv = x[:N_GV]
            total_change = np.sum(np.abs(delta_gv))
            if total_change > GV_TOTAL_CHANGE_MAX:
                violations.append(f"총 변화량 위반: Σ|△GV| = {total_change:.3f} > {GV_TOTAL_CHANGE_MAX}")

        is_feasible = len(violations) == 0
        message = " | ".join(violations) if violations else "제약 만족"

        return is_feasible, message

    # ========================================================================
    # 목적함수
    # ========================================================================

    def objective_function(self, x: np.ndarray) -> float:
        """
        목적함수 (최소화할 값)

        프로세스:
        1. 제약 검증
        2. Multi-zone 제어 평가
        3. 비용 평가
        4. 위반 페널티 추가

        Args:
            x: Shape (12,) - 제어값

        Returns:
            float - 목적함수 값 (최소화 대상)
        """
        self.eval_count += 1

        # 0. 출력 변환 (정수화)
        x_transformed = self.output_transformer.transform(x, x_type='control_vector')

        # 1. 제약 조건 검증 (정수화된 값으로)
        is_feasible, constraint_msg = self.check_constraints(x_transformed)
        penalty = 0.0

        if not is_feasible:
            # 제약 위반 시 페널티 부과
            penalty_multiplier = self.constraint_params.get('penalty_multiplier', 1e6)
            penalty = penalty_multiplier
            logger.debug(f"[Eval {self.eval_count}] 제약 위반: {constraint_msg}")

        # 2. Multi-zone 평가 (정수화된 값으로)
        try:
            control_result = self.controller.evaluate_control(x_transformed, self.current_state)
            p_low = control_result['p_low']
            p_mid = control_result['p_mid']
            p_high = control_result['p_high']
        except Exception as e:
            logger.error(f"제어 평가 실패: {e}")
            return 1e10 + penalty

        # 3. 비용 평가 (정수화된 값으로)
        delta_gv = x_transformed[:N_GV]
        delta_rpm = x_transformed[N_GV]

        try:
            cost, breakdown = self.cost_evaluator.evaluate_total_cost(
                p_low, p_mid, p_high, delta_gv, delta_rpm
            )
        except Exception as e:
            logger.error(f"비용 평가 실패: {e}")
            return 1e10 + penalty

        # 4. 총 비용 (비용 + 페널티)
        total_cost = cost + penalty

        # 히스토리 기록
        self.evaluation_history.append({
            'eval_count': self.eval_count,
            'x': x.copy(),
            'x_transformed': x_transformed.copy(),
            'cost': cost,
            'penalty': penalty,
            'total_cost': total_cost,
            'p_mid_mean': np.mean(p_mid),
            'feasible': is_feasible,
            'transform_active': self.output_transformer.config.enable,
        })

        # 최선 비용 추적
        if len(self.best_cost_history) == 0 or total_cost < self.best_cost_history[-1]:
            self.best_cost_history.append(total_cost)
        else:
            self.best_cost_history.append(self.best_cost_history[-1])

        # 100 평가마다 로그
        if self.eval_count % 100 == 0:
            logger.info(f"[Eval {self.eval_count}] Cost={cost:.6f}, "
                       f"Penalty={penalty:.2e}, Total={total_cost:.6f}, "
                       f"P_Mid_mean={np.mean(p_mid):.4f}")

        return total_cost

    # ========================================================================
    # 최적화 실행
    # ========================================================================

    def run_optimization(self) -> OptimizationResult:
        """
        Differential Evolution 최적화 실행

        Returns:
            OptimizationResult - 최적화 결과
        """
        logger.info("="*80)
        logger.info("Differential Evolution 최적화 시작")
        logger.info("="*80)

        # 경계값
        bounds = list(zip(self.bounds_lower, self.bounds_upper))

        # 최적화 파라미터
        de_kwargs = {
            'strategy': self.optimizer_params.get('strategy', 'best1bin'),
            'maxiter': self.optimizer_params.get('maxiter', 100),
            'popsize': self.optimizer_params.get('popsize', 15),
            'tol': self.optimizer_params.get('tol', 0.001),
            'atol': self.optimizer_params.get('atol', 0.001),
            'seed': self.optimizer_params.get('seed', RANDOM_SEED),
            'workers': self.optimizer_params.get('workers', 1),
            'updating': self.optimizer_params.get('updating', 'immediate'),
            'polish': self.optimizer_params.get('polish', True),
            'init': self.optimizer_params.get('init', 'latinhypercube'),
        }

        logger.info(f"DE 파라미터: {de_kwargs}")
        logger.info(f"예상 평가 횟수: ~{de_kwargs['popsize'] * de_kwargs['maxiter']}")

        # 최적화 실행
        start_time = time.time()
        try:
            scipy_result = differential_evolution(
                func=self.objective_function,
                bounds=bounds,
                **de_kwargs
            )
            elapsed_time = time.time() - start_time

            # 최적해에서의 비용 분석
            self.eval_count_at_best = len([e for e in self.evaluation_history if e['cost'] < scipy_result.fun + 0.001])

            # 최종 해를 정수화
            x_opt_transformed = self.output_transformer.transform(scipy_result.x, x_type='control_vector')

            # 최종 비용 분석 (정수화된 값으로)
            final_control = self.controller.evaluate_control(x_opt_transformed, self.current_state)
            p_low_final = final_control['p_low']
            p_mid_final = final_control['p_mid']
            p_high_final = final_control['p_high']

            _, final_breakdown = self.cost_evaluator.evaluate_total_cost(
                p_low_final, p_mid_final, p_high_final,
                x_opt_transformed[:N_GV], x_opt_transformed[N_GV]
            )

            # 결과 생성
            result = OptimizationResult(
                x_opt=x_opt_transformed,  # 정수화된 최적해
                cost_opt=scipy_result.fun,
                scipy_result=scipy_result,
                n_evaluations=self.eval_count,
                optimization_time=elapsed_time,
                success=scipy_result.success,
                message=scipy_result.message,
                final_cost_breakdown=final_breakdown,
            )

            logger.info("="*80)
            logger.info("최적화 완료")
            logger.info("="*80)
            logger.info(f"성공: {scipy_result.success}")
            logger.info(f"메시지: {scipy_result.message}")
            logger.info(f"총 평가 횟수: {self.eval_count}")
            logger.info(f"소요 시간: {elapsed_time:.2f}초")
            logger.info(f"최적 비용: {scipy_result.fun:.6f}")

            if self.output_transformer.config.enable:
                logger.info(f"최적해 (변환 전): {scipy_result.x}")
                logger.info(f"최적해 (변환 후): {x_opt_transformed}")
            else:
                logger.info(f"최적해: {scipy_result.x}")

            return result

        except Exception as e:
            logger.error(f"최적화 실패: {e}", exc_info=True)
            elapsed_time = time.time() - start_time

            return OptimizationResult(
                x_opt=np.zeros(N_CONTROL_VARS),
                cost_opt=float('inf'),
                scipy_result=None,
                n_evaluations=self.eval_count,
                optimization_time=elapsed_time,
                success=False,
                message=f"최적화 실패: {str(e)}",
            )

    # ========================================================================
    # 히스토리 분석
    # ========================================================================

    def get_convergence_info(self) -> Dict[str, Any]:
        """
        수렴 정보 반환
        """
        if not self.best_cost_history:
            return {}

        costs = np.array(self.best_cost_history)

        info = {
            'n_evaluations': len(self.best_cost_history),
            'initial_cost': costs[0],
            'final_cost': costs[-1],
            'improvement': (costs[0] - costs[-1]) / costs[0] * 100,
            'improvement_after_500': 0.0,
            'median_cost': np.median(costs),
            'std_cost': np.std(costs),
        }

        # 500회 이후 개선도
        if len(costs) > 500:
            info['improvement_after_500'] = (costs[499] - costs[-1]) / costs[499] * 100

        return info

    def print_optimization_summary(self, result: OptimizationResult) -> str:
        """
        최적화 결과 요약 출력
        """
        convergence_info = self.get_convergence_info()

        summary = f"""
        ============================================================
        최적화 결과 요약
        ============================================================

        최적해:
          △GV: {result.x_opt[:N_GV]}
          △RPM: {result.x_opt[N_GV]:.2f}

        성능:
          최적 비용: {result.cost_opt:.6f}
          평가 횟수: {result.n_evaluations}
          소요 시간: {result.optimization_time:.2f}초
          수렴 성공: {result.success}
          메시지: {result.message}

        수렴 정보:
          초기 비용: {convergence_info.get('initial_cost', 'N/A'):.6f}
          최종 비용: {convergence_info.get('final_cost', 'N/A'):.6f}
          개선도: {convergence_info.get('improvement', 0):.2f}%

        비용 분석:
          {self.cost_evaluator.print_cost_summary(result.final_cost_breakdown)}

        ============================================================
        """
        return summary


# ============================================================================
# 테스트 코드
# ============================================================================

if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # 모델 및 비용 함수 초기화
    model_manager = CatBoostModelManager()
    cost_evaluator = CostFunctionEvaluator()

    # 현재 상태 (임의)
    current_state = {
        'current_clr': np.random.randn(N_ZONES, 3)
    }

    # 최적화기 생성
    optimizer = DifferentialEvolutionOptimizer(
        model_manager, cost_evaluator, current_state
    )

    # 최적화 실행
    result = optimizer.run_optimization()

    # 결과 출력
    print(optimizer.print_optimization_summary(result))
