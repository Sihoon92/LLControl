"""
최적화 성능 평가 메트릭
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
import logging

logger = logging.getLogger(__name__)


class OptimizationEvaluator:
    """
    최적화 성능 평가 클래스

    Baseline(실제 제어값) vs Optimized(최적화된 제어값) 비교
    """

    def __init__(self, cost_evaluator):
        """
        Parameters:
        -----------
        cost_evaluator : CostFunctionEvaluator
            비용 함수 평가기
        """
        self.cost_evaluator = cost_evaluator

    def evaluate_control_difference(
        self,
        actual_control: np.ndarray,
        optimized_control: np.ndarray
    ) -> Dict[str, float]:
        """
        제어값 차이 계산

        Parameters:
        -----------
        actual_control : np.ndarray
            실제 제어값 [△GV₁~₁₁, △RPM]
        optimized_control : np.ndarray
            최적화된 제어값 [△GV₁~₁₁, △RPM]

        Returns:
        --------
        Dict[str, float]
            차이 메트릭
        """
        diff = optimized_control - actual_control

        metrics = {
            'mae': np.mean(np.abs(diff)),
            'rmse': np.sqrt(np.mean(diff ** 2)),
            'max_error': np.max(np.abs(diff)),
            'gv_mae': np.mean(np.abs(diff[:11])),  # GV만
            'rpm_error': np.abs(diff[11]),  # RPM만
        }

        return metrics

    def evaluate_cost_improvement(
        self,
        baseline_cost: float,
        baseline_breakdown: Dict[str, float],
        optimized_cost: float,
        optimized_breakdown: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        비용 개선도 계산

        Parameters:
        -----------
        baseline_cost : float
            Baseline 총 비용
        baseline_breakdown : Dict
            Baseline 비용 항목별 분석
        optimized_cost : float
            최적화 총 비용
        optimized_breakdown : Dict
            최적화 비용 항목별 분석

        Returns:
        --------
        Dict[str, Any]
            개선도 메트릭
        """
        # 총 비용 개선
        absolute_improvement = baseline_cost - optimized_cost
        relative_improvement = (absolute_improvement / baseline_cost * 100) if baseline_cost > 0 else 0.0

        # 항목별 개선
        item_improvements = {}
        for key in baseline_breakdown.keys():
            if key in optimized_breakdown:
                baseline_val = baseline_breakdown[key]
                optimized_val = optimized_breakdown[key]
                abs_imp = baseline_val - optimized_val
                rel_imp = (abs_imp / baseline_val * 100) if baseline_val > 0 else 0.0

                item_improvements[key] = {
                    'baseline': baseline_val,
                    'optimized': optimized_val,
                    'absolute_improvement': abs_imp,
                    'relative_improvement': rel_imp
                }

        metrics = {
            'baseline_cost': baseline_cost,
            'optimized_cost': optimized_cost,
            'absolute_improvement': absolute_improvement,
            'relative_improvement': relative_improvement,
            'item_improvements': item_improvements
        }

        return metrics

    def evaluate_prediction_accuracy(
        self,
        predicted_probs: np.ndarray,
        actual_probs: np.ndarray
    ) -> Dict[str, float]:
        """
        예측 정확도 평가 (확률 분포 비교)

        Parameters:
        -----------
        predicted_probs : np.ndarray
            예측된 확률 분포 [p_low, p_mid, p_high] x N_zones
        actual_probs : np.ndarray
            실제 확률 분포 [p_low, p_mid, p_high] x N_zones

        Returns:
        --------
        Dict[str, float]
            정확도 메트릭
        """
        # MAE (Mean Absolute Error)
        mae = np.mean(np.abs(predicted_probs - actual_probs))

        # RMSE (Root Mean Squared Error)
        rmse = np.sqrt(np.mean((predicted_probs - actual_probs) ** 2))

        # KL Divergence (확률 분포 간 거리)
        epsilon = 1e-10
        predicted_probs = np.clip(predicted_probs, epsilon, 1.0)
        actual_probs = np.clip(actual_probs, epsilon, 1.0)

        kl_divergence = np.sum(actual_probs * np.log(actual_probs / predicted_probs))

        metrics = {
            'mae': mae,
            'rmse': rmse,
            'kl_divergence': kl_divergence
        }

        return metrics

    def compare_overall(
        self,
        baseline_results: List[Dict],
        optimized_results: List[Dict]
    ) -> Dict[str, Any]:
        """
        전체 결과 비교

        Parameters:
        -----------
        baseline_results : List[Dict]
            Baseline 결과 리스트
        optimized_results : List[Dict]
            최적화 결과 리스트

        Returns:
        --------
        Dict[str, Any]
            전체 비교 결과
        """
        n_samples = len(baseline_results)

        # 비용 개선도 집계
        cost_improvements = []
        control_diffs = []

        for baseline, optimized in zip(baseline_results, optimized_results):
            # 비용 개선
            cost_imp = self.evaluate_cost_improvement(
                baseline['cost'],
                baseline['cost_breakdown'],
                optimized['cost'],
                optimized['cost_breakdown']
            )
            cost_improvements.append(cost_imp)

            # 제어값 차이
            control_diff = self.evaluate_control_difference(
                baseline['actual_control'],
                optimized['optimized_control']
            )
            control_diffs.append(control_diff)

        # 평균 계산
        avg_relative_improvement = np.mean([ci['relative_improvement'] for ci in cost_improvements])
        avg_absolute_improvement = np.mean([ci['absolute_improvement'] for ci in cost_improvements])

        avg_control_mae = np.mean([cd['mae'] for cd in control_diffs])
        avg_control_rmse = np.mean([cd['rmse'] for cd in control_diffs])

        summary = {
            'n_samples': n_samples,
            'avg_cost_improvement': {
                'relative': avg_relative_improvement,
                'absolute': avg_absolute_improvement
            },
            'avg_control_difference': {
                'mae': avg_control_mae,
                'rmse': avg_control_rmse
            },
            'cost_improvements': cost_improvements,
            'control_diffs': control_diffs
        }

        return summary

    def compare_by_zone(
        self,
        baseline_results: List[Dict],
        optimized_results: List[Dict]
    ) -> pd.DataFrame:
        """
        Zone별 비교

        Parameters:
        -----------
        baseline_results : List[Dict]
            Baseline 결과 리스트
        optimized_results : List[Dict]
            최적화 결과 리스트

        Returns:
        --------
        pd.DataFrame
            Zone별 비교 결과
        """
        zone_stats = []

        for baseline, optimized in zip(baseline_results, optimized_results):
            zone_id = baseline.get('zone_id', -1)
            group_id = baseline.get('group_id', -1)

            # 비용 개선
            cost_imp = (baseline['cost'] - optimized['cost']) / baseline['cost'] * 100 if baseline['cost'] > 0 else 0.0

            # 제어값 차이
            control_diff = self.evaluate_control_difference(
                baseline['actual_control'],
                optimized['optimized_control']
            )

            zone_stats.append({
                'group_id': group_id,
                'zone_id': zone_id,
                'baseline_cost': baseline['cost'],
                'optimized_cost': optimized['cost'],
                'cost_improvement_%': cost_imp,
                'control_mae': control_diff['mae'],
                'control_rmse': control_diff['rmse']
            })

        df = pd.DataFrame(zone_stats)

        # Zone별 평균
        if 'zone_id' in df.columns:
            zone_summary = df.groupby('zone_id').agg({
                'baseline_cost': 'mean',
                'optimized_cost': 'mean',
                'cost_improvement_%': 'mean',
                'control_mae': 'mean',
                'control_rmse': 'mean'
            }).reset_index()

            return zone_summary

        return df

    def generate_evaluation_report(
        self,
        test_results: Dict[str, Any]
    ) -> str:
        """
        평가 리포트 생성

        Parameters:
        -----------
        test_results : Dict
            테스트 결과 딕셔너리

        Returns:
        --------
        str
            평가 리포트 텍스트
        """
        overall = test_results['overall']

        report = f"""
========================================================================
                    최적화 성능 평가 리포트
========================================================================

총 샘플 수: {overall['n_samples']}

------------------------------------------------------------------------
1. 비용 개선도
------------------------------------------------------------------------
  평균 상대 개선도: {overall['avg_cost_improvement']['relative']:.2f}%
  평균 절대 개선도: {overall['avg_cost_improvement']['absolute']:.6f}

------------------------------------------------------------------------
2. 제어값 차이
------------------------------------------------------------------------
  평균 MAE: {overall['avg_control_difference']['mae']:.4f}
  평균 RMSE: {overall['avg_control_difference']['rmse']:.4f}

------------------------------------------------------------------------
3. Zone별 분석
------------------------------------------------------------------------
"""

        if 'by_zone' in test_results:
            zone_df = test_results['by_zone']
            report += zone_df.to_string(index=False)

        report += "\n\n========================================================================"

        return report


# ============================================================================
# 테스트 코드
# ============================================================================

if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # 간단한 테스트
    from apc_optimization import CostFunctionEvaluator

    cost_evaluator = CostFunctionEvaluator()
    evaluator = OptimizationEvaluator(cost_evaluator)

    # 더미 데이터
    actual_control = np.random.uniform(-0.5, 0.5, 12)
    optimized_control = np.random.uniform(-0.5, 0.5, 12)

    control_diff = evaluator.evaluate_control_difference(actual_control, optimized_control)
    logger.info(f"제어값 차이: {control_diff}")
