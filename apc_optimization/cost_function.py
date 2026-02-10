"""
다목적 비용 함수 (Multi-Objective Cost Function)

Total_Cost = w₁·Quality_Cost + w₂·Balance_Cost + w₃·Control_Cost + w₄·Safety_Cost

각 비용 항목은 [0, 1] 범위로 정규화됨
"""

import numpy as np
from typing import Dict, Tuple, List, Optional
import logging
from .config import (
    COST_WEIGHTS, QUALITY_COST_PARAMS, BALANCE_COST_PARAMS,
    CONTROL_COST_PARAMS, SAFETY_COST_PARAMS, CONTROL_LIMITS,
    N_ZONES, GV_ADJACENT_MAX_DIFF, GV_TOTAL_CHANGE_MAX
)
from .normalizer import ControlVariableNormalizer

logger = logging.getLogger(__name__)


class CostFunctionEvaluator:
    """
    다목적 비용 함수 평가자

    입력: Zone별 확률 분포 (P_Low, P_Mid, P_High), 제어값 (△GV, △RPM)
    출력: 정규화된 총 비용값 + 상세 비용 분석
    """

    def __init__(self,
                 weights: Optional[Dict[str, float]] = None,
                 ucl: float = CONTROL_LIMITS['ucl'],
                 lcl: float = CONTROL_LIMITS['lcl'],
                 normalizer: Optional[ControlVariableNormalizer] = None):
        """
        초기화

        Args:
            weights: 비용 가중치 (없으면 config 기본값 사용)
            ucl: Upper Control Limit
            lcl: Lower Control Limit
            normalizer: ControlVariableNormalizer 인스턴스 (없으면 생성)
        """
        self.weights = weights or COST_WEIGHTS
        self.ucl = ucl
        self.lcl = lcl

        # 통합 정규화 클래스 초기화
        if normalizer is None:
            self.normalizer = ControlVariableNormalizer(
                gv_max=CONTROL_COST_PARAMS['gv_max'],
                rpm_max=CONTROL_COST_PARAMS['rpm_max']
            )
        else:
            self.normalizer = normalizer

        # 가중치 정규화 (합이 1이 되도록)
        total_weight = sum(self.weights.values())
        self.weights_normalized = {k: v/total_weight for k, v in self.weights.items()}

        logger.info(f"Cost Function Evaluator 초기화")
        logger.info(f"가중치: {self.weights}")
        logger.info(f"정규화된 가중치: {self.weights_normalized}")
        logger.info(f"정규화: {self.normalizer.get_description()}")

    # ========================================================================
    # 1. 품질 비용 (Quality Cost)
    # ========================================================================

    def quality_cost(self,
                     p_low_array: np.ndarray,
                     p_mid_array: np.ndarray,
                     p_high_array: np.ndarray) -> Tuple[float, Dict]:
        """
        품질 비용 계산

        각 Zone i에 대해:
            Q_i = (1 - P_Mid(i))² + α·[P_Low(i)² + P_High(i)²]

        Quality_Cost = mean(Q_i) / max_possible_Q

        Args:
            p_low_array: Shape (n_zones,) - 각 Zone의 P_Low
            p_mid_array: Shape (n_zones,) - 각 Zone의 P_Mid
            p_high_array: Shape (n_zones,) - 각 Zone의 P_High

        Returns:
            (quality_cost, detail_dict)
        """
        alpha = QUALITY_COST_PARAMS['alpha']
        target_p_mid = QUALITY_COST_PARAMS['target_p_mid']

        # 개별 비용 계산
        q_values = (
            (target_p_mid - p_mid_array) ** 2 +
            alpha * (p_low_array ** 2 + p_high_array ** 2)
        )

        # 최악의 경우: P_Mid=0 (균등 분포에서 P=0.33)
        # Q_worst = (1 - 0)² + α·(0.33² + 0.33²) ≈ 1 + 0.218 = 1.218
        q_worst = (target_p_mid - 0.0) ** 2 + alpha * 2 * (1.0/3) ** 2

        # 정규화
        quality_cost_normalized = np.mean(q_values) / q_worst
        quality_cost_normalized = np.clip(quality_cost_normalized, 0.0, 1.0)

        details = {
            'q_values': q_values,
            'mean_q': np.mean(q_values),
            'worst_q': q_worst,
            'p_mid_mean': np.mean(p_mid_array),
            'p_mid_std': np.std(p_mid_array),
        }

        return quality_cost_normalized, details

    # ========================================================================
    # 2. 균형 비용 (Balance Cost)
    # ========================================================================

    def balance_cost(self, p_mid_array: np.ndarray) -> Tuple[float, Dict]:
        """
        Zone 간 균일성 비용

        Balance_Cost = σ² / μ²  (변동계수의 제곱)

        모든 Zone이 동일한 P_Mid를 가지면: σ=0 → Cost=0
        극단적 차이가 나면: Cost → 1

        Args:
            p_mid_array: Shape (n_zones,) - 각 Zone의 P_Mid

        Returns:
            (balance_cost, detail_dict)
        """
        mu = np.mean(p_mid_array)
        sigma_sq = np.var(p_mid_array)

        # 분모 보호 (매우 작은 mu 값 처리)
        min_mean = BALANCE_COST_PARAMS['min_mean']
        mu_safe = max(mu, min_mean)

        balance_cost = sigma_sq / (mu_safe ** 2)
        balance_cost = np.clip(balance_cost, 0.0, 1.0)

        details = {
            'mu': mu,
            'sigma_sq': sigma_sq,
            'coefficient_of_variation_sq': balance_cost,
            'std': np.sqrt(sigma_sq),
            'max_p_mid': np.max(p_mid_array),
            'min_p_mid': np.min(p_mid_array),
        }

        return balance_cost, details

    # ========================================================================
    # 3. 제어 비용 (Control Cost)
    # ========================================================================

    def control_cost(self,
                     delta_gv: np.ndarray,
                     delta_rpm: float) -> Tuple[float, Dict]:
        """
        제어 변화량 최소화 비용

        통합 정규화 클래스를 사용한 MinMax 정규화:
        GV_norm = [Σ(|△GV_i| / gv_max)²] / 11
        RPM_norm = (|△RPM| / rpm_max)²
        Control_Cost = beta·GV_norm + gamma·RPM_norm

        Args:
            delta_gv: Shape (11,) - 각 GV의 변화량 (mm)
            delta_rpm: Scalar - RPM 변화량

        Returns:
            (control_cost, detail_dict)
        """
        # 통합 정규화 클래스 사용
        gv_normalized, rpm_normalized = self.normalizer.normalize_for_cost(
            delta_gv, delta_rpm
        )

        # 제어 비용 계산 (정규화된 값의 제곱)
        beta = CONTROL_COST_PARAMS['beta']
        gamma = CONTROL_COST_PARAMS['gamma']

        gv_norm = np.mean(gv_normalized ** 2)
        rpm_norm = rpm_normalized ** 2

        # 가중 합
        control_cost = beta * gv_norm + gamma * rpm_norm
        control_cost = np.clip(control_cost, 0.0, 1.0)

        details = {
            'gv_normalized': gv_normalized,      # ★ 새로 추가
            'rpm_normalized': rpm_normalized,    # ★ 새로 추가
            'gv_norm': gv_norm,
            'rpm_norm': rpm_norm,
            'gv_values': delta_gv,
            'rpm_value': delta_rpm,
            'gv_sum_abs': np.sum(np.abs(delta_gv)),
            'gv_max_abs': np.max(np.abs(delta_gv)),
        }

        return control_cost, details

    # ========================================================================
    # 4. 안전 비용 (Safety Cost)
    # ========================================================================

    def safety_cost(self,
                    delta_gv: np.ndarray,
                    delta_rpm: float,
                    p_high_array: np.ndarray,
                    p_low_array: np.ndarray) -> Tuple[float, Dict]:
        """
        제약 위반 페널티

        V₁ = Σ max(0, |△GV_i - △GV_(i±1)| - 0.5)  (인접 차이)
        V₂ = max(0, Σ|△GV_i| - 10.0)              (총 변화량)
        V₃ = max(0, max(P_High) - UCL)            (상한 초과)
        V₄ = max(0, LCL - min(P_Low))             (하한 미달)

        Safety_Cost = (V₁ + V₂ + V₃ + V₄) / 4

        Args:
            delta_gv: Shape (11,) - GV 변화량
            delta_rpm: Scalar - RPM 변화량
            p_high_array: Shape (n_zones,) - P_High 값
            p_low_array: Shape (n_zones,) - P_Low 값

        Returns:
            (safety_cost, detail_dict)
        """
        violations = {}

        # V₁: 인접 Zone GV 차이
        v1 = 0.0
        for i in range(len(delta_gv) - 1):
            diff = abs(delta_gv[i] - delta_gv[i+1])
            if diff > GV_ADJACENT_MAX_DIFF:
                v1 += (diff - GV_ADJACENT_MAX_DIFF)
        violations['v1_adjacent_diff'] = v1

        # V₂: 전체 GV 변화량
        total_change = np.sum(np.abs(delta_gv))
        v2 = max(0, total_change - GV_TOTAL_CHANGE_MAX)
        violations['v2_total_change'] = v2

        # V₃: P_High 상한 초과
        max_p_high = np.max(p_high_array)
        v3 = max(0, max_p_high - self.ucl)
        violations['v3_ucl_excess'] = v3

        # V₄: P_Low 하한 미달
        min_p_low = np.min(p_low_array)
        v4 = max(0, self.lcl - min_p_low)
        violations['v4_lcl_deficit'] = v4

        # 총 위반량 (정규화)
        max_violation = max(
            GV_ADJACENT_MAX_DIFF * len(delta_gv),  # V₁ 최악
            GV_TOTAL_CHANGE_MAX,                   # V₂ 최악
            1.0,                                    # V₃ 최악
            1.0,                                    # V₄ 최악
        )

        total_violation = v1 + v2 + v3 + v4
        safety_cost = min(total_violation / 4, 1.0)  # 4로 정규화

        details = {
            'violations': violations,
            'total_violation': total_violation,
            'adjacent_violations_count': sum(1 for i in range(len(delta_gv)-1)
                                             if abs(delta_gv[i] - delta_gv[i+1]) > GV_ADJACENT_MAX_DIFF),
            'total_change': total_change,
            'max_p_high': max_p_high,
            'min_p_low': min_p_low,
        }

        return safety_cost, details

    # ========================================================================
    # 5. 통합 비용 함수
    # ========================================================================

    def evaluate_total_cost(self,
                           p_low_array: np.ndarray,
                           p_mid_array: np.ndarray,
                           p_high_array: np.ndarray,
                           delta_gv: np.ndarray,
                           delta_rpm: float) -> Tuple[float, Dict]:
        """
        전체 비용 평가

        Total_Cost = w₁·Q + w₂·B + w₃·C + w₄·S

        Args:
            p_low_array: Shape (n_zones,)
            p_mid_array: Shape (n_zones,)
            p_high_array: Shape (n_zones,)
            delta_gv: Shape (11,) - GV 변화량
            delta_rpm: Scalar - RPM 변화량

        Returns:
            (total_cost, cost_breakdown_dict)
        """
        # 각 비용 항목 계산
        q_cost, q_detail = self.quality_cost(p_low_array, p_mid_array, p_high_array)
        b_cost, b_detail = self.balance_cost(p_mid_array)
        c_cost, c_detail = self.control_cost(delta_gv, delta_rpm)
        s_cost, s_detail = self.safety_cost(delta_gv, delta_rpm, p_high_array, p_low_array)

        # 가중 합 (비정규화된 가중치)
        total_cost = (
            self.weights['quality'] * q_cost +
            self.weights['balance'] * b_cost +
            self.weights['control'] * c_cost +
            self.weights['safety'] * s_cost
        )

        # 전체 가중치로 정규화
        total_weight = sum(self.weights.values())
        total_cost_normalized = total_cost / total_weight

        # 상세 정보
        breakdown = {
            'total_cost': total_cost_normalized,
            'quality_cost': q_cost,
            'balance_cost': b_cost,
            'control_cost': c_cost,
            'safety_cost': s_cost,
            'quality_detail': q_detail,
            'balance_detail': b_detail,
            'control_detail': c_detail,
            'safety_detail': s_detail,
            'weights': self.weights_normalized,
        }

        return total_cost_normalized, breakdown

    # ========================================================================
    # 유틸리티 함수
    # ========================================================================

    def evaluate_batch(self,
                      p_low_batch: np.ndarray,
                      p_mid_batch: np.ndarray,
                      p_high_batch: np.ndarray,
                      delta_gv_batch: np.ndarray,
                      delta_rpm_batch: np.ndarray) -> Tuple[np.ndarray, List[Dict]]:
        """
        배치 비용 평가 (최적화 중 여러 후보 평가)

        Args:
            p_low_batch: Shape (n_samples, n_zones)
            p_mid_batch: Shape (n_samples, n_zones)
            p_high_batch: Shape (n_samples, n_zones)
            delta_gv_batch: Shape (n_samples, 11)
            delta_rpm_batch: Shape (n_samples,)

        Returns:
            (costs, breakdowns)
        """
        n_samples = len(p_low_batch)
        costs = np.zeros(n_samples)
        breakdowns = []

        for i in range(n_samples):
            cost, breakdown = self.evaluate_total_cost(
                p_low_batch[i],
                p_mid_batch[i],
                p_high_batch[i],
                delta_gv_batch[i],
                delta_rpm_batch[i]
            )
            costs[i] = cost
            breakdowns.append(breakdown)

        return costs, breakdowns

    def print_cost_summary(self, breakdown: Dict) -> str:
        """
        비용 분석 결과 요약 출력
        """
        summary = f"""
        ============================================================
        비용 함수 분석 결과
        ============================================================

        총 비용: {breakdown['total_cost']:.4f}

        개별 비용:
          - 품질 비용 (Quality): {breakdown['quality_cost']:.4f}
          - 균형 비용 (Balance): {breakdown['balance_cost']:.4f}
          - 제어 비용 (Control): {breakdown['control_cost']:.4f}
          - 안전 비용 (Safety): {breakdown['safety_cost']:.4f}

        가중치:
          - 품질: {breakdown['weights']['quality']:.3f}
          - 균형: {breakdown['weights']['balance']:.3f}
          - 제어: {breakdown['weights']['control']:.3f}
          - 안전: {breakdown['weights']['safety']:.3f}

        품질 비용 상세:
          - P_Mid 평균: {breakdown['quality_detail']['p_mid_mean']:.4f}
          - P_Mid 표준편차: {breakdown['quality_detail']['p_mid_std']:.4f}

        균형 비용 상세:
          - P_Mid 변동계수²: {breakdown['balance_detail']['coefficient_of_variation_sq']:.4f}
          - P_Mid 범위: [{breakdown['balance_detail']['min_p_mid']:.4f}, {breakdown['balance_detail']['max_p_mid']:.4f}]

        제어 비용 상세:
          - GV 정규화: {breakdown['control_detail']['gv_norm']:.4f}
          - RPM 정규화: {breakdown['control_detail']['rpm_norm']:.4f}
          - GV 총 변화량: {breakdown['control_detail']['gv_sum_abs']:.4f}

        안전 비용 상세:
          - 위반 항목: {breakdown['safety_detail']['violations']}
          - 총 위반량: {breakdown['safety_detail']['total_violation']:.4f}

        ============================================================
        """
        return summary


# ============================================================================
# 테스트 코드
# ============================================================================

if __name__ == '__main__':
    import sys
    logging.basicConfig(level=logging.INFO)

    # Cost Function 인스턴스 생성
    evaluator = CostFunctionEvaluator()

    # 테스트 데이터
    p_low = np.random.uniform(0, 0.2, N_ZONES)
    p_mid = np.random.uniform(0.6, 1.0, N_ZONES)
    p_high = 1.0 - p_low - p_mid
    p_high = np.clip(p_high, 0, 1)  # 범위 제한

    delta_gv = np.random.uniform(-0.5, 0.5, 11)
    delta_rpm = np.random.uniform(-20, 20, 1)[0]

    # 평가
    total_cost, breakdown = evaluator.evaluate_total_cost(
        p_low, p_mid, p_high, delta_gv, delta_rpm
    )

    print(evaluator.print_cost_summary(breakdown))
