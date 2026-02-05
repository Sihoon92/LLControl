"""
검증 프레임워크 (Validation Framework)

오프라인 검증 (Hold-out 테스트 데이터에서):
1. 각 테스트 샘플에 대해 최적화 실행
2. 예측된 결과 vs 실제 결과 비교
3. 성능 지표 계산

검증 지표:
- RMSE (P_Mid)
- Success Rate (P_Mid > 0.8)
- Constraint Violation Rate
- Mean Absolute Error (MAE)
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from pathlib import Path

from .config import VALIDATION_PARAMS, N_ZONES, OPTIMIZATION_OUTPUT_DIR
from .optimizer_engine import OptimizationResult, DifferentialEvolutionOptimizer
from .cost_function import CostFunctionEvaluator
from .model_interface import CatBoostModelManager
from .multi_zone_controller import MultiZoneController

logger = logging.getLogger(__name__)


@dataclass
class ValidationMetrics:
    """검증 지표"""
    n_samples: int                     # 검증 샘플 수
    rmse_p_mid: float                  # RMSE of P_Mid
    mae_p_mid: float                   # MAE of P_Mid
    success_rate: float                # P_Mid > 0.8 달성 비율
    constraint_violation_rate: float   # 제약 위반 비율
    r2_score: float                    # R² 점수 (설명력)

    # 상세 결과
    predictions: Dict = field(default_factory=dict)  # 개별 예측
    errors: Dict = field(default_factory=dict)       # 오차 분석


class OfflineValidationFramework:
    """
    오프라인 검증 프레임워크
    """

    def __init__(self,
                 test_data: pd.DataFrame,
                 model_manager: CatBoostModelManager,
                 cost_evaluator: CostFunctionEvaluator,
                 validation_params: Optional[Dict] = None):
        """
        초기화

        Args:
            test_data: 테스트 데이터 DataFrame
            model_manager: CatBoostModelManager
            cost_evaluator: CostFunctionEvaluator
            validation_params: 검증 파라미터
        """
        self.test_data = test_data
        self.model = model_manager
        self.cost_evaluator = cost_evaluator
        self.validation_params = validation_params or VALIDATION_PARAMS

        self.controller = MultiZoneController(model_manager)

        logger.info(f"OfflineValidationFramework 초기화")
        logger.info(f"  테스트 샘플: {len(test_data)}")

    # ========================================================================
    # 검증 실행
    # ========================================================================

    def run_validation(self,
                      n_samples: Optional[int] = None,
                      verbose: bool = True) -> ValidationMetrics:
        """
        오프라인 검증 실행

        Args:
            n_samples: 검증할 샘플 수 (None이면 전체)
            verbose: 상세 로그 출력

        Returns:
            ValidationMetrics
        """
        logger.info("="*80)
        logger.info("오프라인 검증 시작")
        logger.info("="*80)

        n_test = len(self.test_data)
        n_samples = n_samples or n_test
        n_samples = min(n_samples, n_test)

        # 테스트 샘플 선택
        test_indices = np.random.choice(n_test, n_samples, replace=False)

        predictions_list = []
        errors_list = []
        constraint_violations = []

        for idx, sample_idx in enumerate(test_indices):
            sample = self.test_data.iloc[sample_idx]

            # 1. 테스트 샘플에서 현재 상태 추출
            current_state = self._extract_current_state(sample)

            # 2. 최적화 실행 (빠른 테스트용 - 짧은 반복)
            try:
                opt_result = self._run_quick_optimization(current_state)
            except Exception as e:
                logger.warning(f"샘플 {idx} 최적화 실패: {e}")
                continue

            # 3. 예측 결과 추출
            pred_control = self.controller.evaluate_control(
                opt_result.x_opt, current_state
            )
            pred_p_mid = pred_control['p_mid']

            # 4. 실제 결과 추출 (테스트 데이터에서)
            actual_p_mid = self._extract_actual_p_mid(sample)

            # 5. 오차 계산
            error = np.abs(pred_p_mid - actual_p_mid)
            rmse = np.sqrt(np.mean(error ** 2))
            mae = np.mean(error)

            # 6. 제약 위반 검사
            violation = np.any(pred_p_mid < 0.8)  # Success threshold

            predictions_list.append({
                'sample_idx': sample_idx,
                'pred_p_mid': pred_p_mid,
                'actual_p_mid': actual_p_mid,
                'error': error,
                'rmse': rmse,
                'mae': mae,
            })

            constraint_violations.append(violation)

            if verbose and (idx + 1) % max(1, n_samples // 10) == 0:
                logger.info(f"  [{idx+1}/{n_samples}] RMSE={rmse:.6f}, MAE={mae:.6f}, "
                           f"Violation={violation}")

        # 통계 계산
        metrics = self._compute_metrics(predictions_list, constraint_violations)

        logger.info("="*80)
        logger.info("검증 완료")
        logger.info("="*80)

        return metrics

    # ========================================================================
    # 헬퍼 함수
    # ========================================================================

    def _extract_current_state(self, sample: pd.Series) -> Dict:
        """
        테스트 샘플에서 현재 상태 추출

        Args:
            sample: DataFrame 행

        Returns:
            dict: {'current_clr': Shape (n_zones, 3)}
        """
        # 예상 칼럼명: 'current_CLR_1', 'current_CLR_2', 'current_CLR_3'
        current_clr = np.zeros((N_ZONES, 3))

        for zone_id in range(N_ZONES):
            for clr_id in range(3):
                col_name = f'current_CLR_{clr_id+1}_Zone{zone_id+1:02d}'
                if col_name in sample.index:
                    current_clr[zone_id, clr_id] = sample[col_name]

        return {'current_clr': current_clr}

    def _extract_actual_p_mid(self, sample: pd.Series) -> np.ndarray:
        """
        테스트 샘플에서 실제 P_Mid 추출

        Args:
            sample: DataFrame 행

        Returns:
            Shape (n_zones,)
        """
        p_mid = np.zeros(N_ZONES)

        for zone_id in range(N_ZONES):
            col_name = f'actual_P_Mid_Zone{zone_id+1:02d}'
            if col_name in sample.index:
                p_mid[zone_id] = sample[col_name]
            else:
                # 기본값
                p_mid[zone_id] = 0.5

        return p_mid

    def _run_quick_optimization(self, current_state: Dict) -> OptimizationResult:
        """
        빠른 최적화 실행 (검증용 - 적은 반복)

        Args:
            current_state: 현재 상태

        Returns:
            OptimizationResult
        """
        optimizer = DifferentialEvolutionOptimizer(
            self.model, self.cost_evaluator, current_state,
            optimizer_params={
                'strategy': 'best1bin',
                'maxiter': 5,      # 빠른 테스트용
                'popsize': 10,
                'tol': 0.01,
                'seed': 42,
                'workers': 1,
            }
        )

        result = optimizer.run_optimization()
        return result

    def _compute_metrics(self,
                        predictions_list: List[Dict],
                        constraint_violations: List[bool]) -> ValidationMetrics:
        """
        검증 지표 계산

        Args:
            predictions_list: 예측 결과 리스트
            constraint_violations: 제약 위반 리스트

        Returns:
            ValidationMetrics
        """
        if not predictions_list:
            logger.warning("검증 결과가 없습니다")
            return ValidationMetrics(
                n_samples=0,
                rmse_p_mid=float('nan'),
                mae_p_mid=float('nan'),
                success_rate=0.0,
                constraint_violation_rate=float('nan'),
                r2_score=float('nan'),
            )

        # 수치 모음
        rmse_values = [p['rmse'] for p in predictions_list]
        mae_values = [p['mae'] for p in predictions_list]

        # RMSE, MAE
        rmse_p_mid = np.mean(rmse_values)
        mae_p_mid = np.mean(mae_values)

        # Success rate (P_Mid > 0.8 달성)
        success_count = sum(1 for p in predictions_list
                           if np.mean(p['pred_p_mid']) > 0.8)
        success_rate = success_count / len(predictions_list)

        # Constraint violation rate
        violation_rate = np.mean(constraint_violations)

        # R² score (설명력)
        # R² = 1 - (SS_res / SS_tot)
        all_errors = np.concatenate([p['error'] for p in predictions_list])
        ss_res = np.sum(all_errors ** 2)
        mean_error = np.mean(all_errors)
        ss_tot = np.sum((all_errors - mean_error) ** 2)
        r2_score = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

        metrics = ValidationMetrics(
            n_samples=len(predictions_list),
            rmse_p_mid=rmse_p_mid,
            mae_p_mid=mae_p_mid,
            success_rate=success_rate,
            constraint_violation_rate=violation_rate,
            r2_score=r2_score,
            predictions={'samples': predictions_list},
            errors={
                'rmse_list': rmse_values,
                'mae_list': mae_values,
            }
        )

        return metrics

    # ========================================================================
    # 결과 출력 및 저장
    # ========================================================================

    def print_validation_report(self, metrics: ValidationMetrics) -> str:
        """
        검증 결과 리포트 출력

        Args:
            metrics: ValidationMetrics

        Returns:
            str - 리포트 텍스트
        """
        report = f"""
        ============================================================
        오프라인 검증 결과
        ============================================================

        검증 통계:
          - 테스트 샘플: {metrics.n_samples}개
          - RMSE (P_Mid): {metrics.rmse_p_mid:.6f}
          - MAE (P_Mid): {metrics.mae_p_mid:.6f}
          - 성공률 (P_Mid > 0.8): {metrics.success_rate*100:.1f}%
          - 제약 위반률: {metrics.constraint_violation_rate*100:.1f}%
          - R² 점수: {metrics.r2_score:.4f}

        해석:
          - RMSE < 0.1: 예측 정확도 높음 ✓
          - 성공률 > 80%: 제어 목표 달성 가능 ✓
          - 제약 위반률 < 10%: 제약 만족 가능 ✓

        ============================================================
        """

        return report

    def save_validation_report(self,
                              metrics: ValidationMetrics,
                              output_path: Optional[Path] = None) -> Path:
        """
        검증 결과를 파일로 저장

        Args:
            metrics: ValidationMetrics
            output_path: 저장 경로

        Returns:
            저장된 파일 경로
        """
        if output_path is None:
            output_path = OPTIMIZATION_OUTPUT_DIR / 'validation_report.txt'

        report = self.print_validation_report(metrics)

        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report)
            logger.info(f"✓ 검증 결과 저장: {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"파일 저장 실패: {e}")
            return output_path


# ============================================================================
# 테스트 코드
# ============================================================================

if __name__ == '__main__':
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # 테스트 데이터 생성 (더미)
    test_data = pd.DataFrame({
        'current_CLR_1_Zone01': np.random.randn(10),
        'current_CLR_2_Zone01': np.random.randn(10),
        'current_CLR_3_Zone01': np.random.randn(10),
        'actual_P_Mid_Zone01': np.random.uniform(0.6, 1.0, 10),
    })

    # 모듈 초기화
    model_manager = CatBoostModelManager()
    cost_evaluator = CostFunctionEvaluator()

    # 검증 프레임워크
    framework = OfflineValidationFramework(
        test_data, model_manager, cost_evaluator
    )

    # 검증 실행 (빠른 테스트 - 2개 샘플)
    metrics = framework.run_validation(n_samples=2, verbose=True)

    # 결과 출력
    print(framework.print_validation_report(metrics))

    # 파일 저장
    framework.save_validation_report(metrics)
