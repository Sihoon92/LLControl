"""
제어 변수 정규화 통합 관리자 (Unified Control Variable Normalizer)

두 시스템이 동일한 정규화 기준을 사용하도록 관리:
- 예측 모델 (model_interface.py) - StandardScaler
- 최적화 모델 (cost_function.py) - MinMax (절댓값 기준)

정규화 방식:
1. MinMax (절댓값 기준): normalized = |value| / max_value, 범위 [0, 1]
2. StandardScaler: (value - μ) / σ, 범위 (-∞, +∞)
"""

import numpy as np
from typing import Tuple, Dict, Union, Optional
import logging

logger = logging.getLogger(__name__)


class ControlVariableNormalizer:
    """
    제어 변수(△GV, △RPM) 정규화 통합 관리자

    역할:
    - 예측 모델과 최적화 모델 간 정규화 기준 통일
    - MinMax 정규화 (절댓값 기준) 또는 StandardScaler 정규화 제공
    - 양방향 변환 지원 (정규화 ↔ 역정규화)

    Parameters:
        gv_max (float): GV 정규화 기준값 (mm) - 기본값: 2.0
        rpm_max (float): RPM 정규화 기준값 - 기본값: 50
        scaler: scikit-learn StandardScaler 인스턴스 (선택사항)

    Example:
        >>> normalizer = ControlVariableNormalizer(gv_max=2.0, rpm_max=50)
        >>> delta_gv = np.array([0.5, 1.0, 2.0])
        >>> delta_rpm = 25.0

        # MinMax 정규화 (비용 함수용)
        >>> gv_norm, rpm_norm = normalizer.normalize_for_cost(delta_gv, delta_rpm)
        >>> print(gv_norm)  # [0.25, 0.5, 1.0]

        # 역정규화
        >>> gv_back, rpm_back = normalizer.denormalize_control_vars(gv_norm, rpm_norm)
        >>> print(gv_back)  # [0.5, 1.0, 2.0]
    """

    def __init__(self,
                 gv_max: float = 2.0,
                 rpm_max: float = 50.0,
                 scaler: Optional[object] = None):
        """
        초기화

        Args:
            gv_max: GV 정규화 기준값 (mm)
            rpm_max: RPM 정규화 기준값
            scaler: scikit-learn StandardScaler 인스턴스 (예측 모델용)

        Raises:
            ValueError: gv_max 또는 rpm_max가 0 이하일 때
        """
        # 입력 검증
        if gv_max <= 0 or rpm_max <= 0:
            raise ValueError(f"gv_max와 rpm_max는 양수여야 합니다. "
                           f"gv_max={gv_max}, rpm_max={rpm_max}")

        self.gv_max = gv_max
        self.rpm_max = rpm_max
        self.scaler = scaler  # StandardScaler (예측 모델용)

        logger.info(f"ControlVariableNormalizer 초기화: "
                   f"gv_max={gv_max}, rpm_max={rpm_max}, "
                   f"scaler={'있음' if scaler else '없음'}")

    # ====================================================================
    # MinMax 정규화 메서드 (비용 함수용)
    # ====================================================================

    def normalize_control_vars(self,
                              delta_gv: np.ndarray,
                              delta_rpm: float) -> Tuple[np.ndarray, float]:
        """
        제어 변수 정규화 (MinMax: [0, 1])

        정규화 공식:
            gv_normalized = |delta_gv| / gv_max
            rpm_normalized = |delta_rpm| / rpm_max

        Args:
            delta_gv: Shape (n_gv,) - GV 변화량 (mm)
                     또는 Shape (n_samples, n_gv) - 배치 처리
            delta_rpm: Scalar - RPM 변화량
                     또는 Shape (n_samples,) - 배치 처리

        Returns:
            (gv_normalized, rpm_normalized)
            - gv_normalized: Shape와 동일하게 반환
            - rpm_normalized: Scalar 또는 배열

        Raises:
            ValueError: 입력값이 nan 또는 inf를 포함할 때
        """
        return self.normalize_for_cost(delta_gv, delta_rpm)

    def normalize_for_cost(self,
                          delta_gv: np.ndarray,
                          delta_rpm: Union[float, np.ndarray]) -> Tuple[np.ndarray, Union[float, np.ndarray]]:
        """
        비용 함수용 정규화 (MinMax: [0, 1])

        Args:
            delta_gv: GV 변화량
            delta_rpm: RPM 변화량

        Returns:
            (gv_normalized, rpm_normalized)
        """
        # 입력 검증
        delta_gv = np.asarray(delta_gv)
        delta_rpm = np.asarray(delta_rpm)

        if np.any(np.isnan(delta_gv)) or np.any(np.isnan(delta_rpm)):
            raise ValueError("정규화 입력에 NaN이 포함되어 있습니다")

        if np.any(np.isinf(delta_gv)) or np.any(np.isinf(delta_rpm)):
            raise ValueError("정규화 입력에 Inf가 포함되어 있습니다")

        # 절댓값 기준 정규화
        gv_normalized = np.abs(delta_gv) / self.gv_max
        rpm_normalized = np.abs(delta_rpm) / self.rpm_max

        # 범위 클립 [0, 1]
        gv_normalized = np.clip(gv_normalized, 0.0, 1.0)
        rpm_normalized = np.clip(rpm_normalized, 0.0, 1.0)

        # 스칼라로 반환
        if isinstance(rpm_normalized, np.ndarray) and rpm_normalized.size == 1:
            rpm_normalized = float(rpm_normalized)

        return gv_normalized, rpm_normalized

    # ====================================================================
    # StandardScaler 정규화 메서드 (예측 모델용)
    # ====================================================================

    def normalize_for_prediction(self,
                               delta_gv: np.ndarray,
                               delta_rpm: Union[float, np.ndarray]
                               ) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
        """
        예측 모델용 정규화 (StandardScaler)

        예측 모델과 동일한 스케일로 정규화하기 위해 StandardScaler 사용.
        StandardScaler를 사용할 수 없는 경우 MinMax로 Fallback.

        Args:
            delta_gv: GV 변화량
            delta_rpm: RPM 변화량

        Returns:
            (gv_normalized, rpm_normalized)
        """
        if self.scaler is not None:
            # StandardScaler 사용
            try:
                # 입력을 2D 배열로 변환
                X = np.atleast_2d([delta_gv, delta_rpm] if np.isscalar(delta_gv) else
                                 [[gv, delta_rpm] for gv in delta_gv])
                X_scaled = self.scaler.transform(X)

                # 결과 반환
                if X_scaled.shape[0] == 1:
                    return X_scaled[0, 0], X_scaled[0, 1]
                else:
                    return X_scaled[:, 0], X_scaled[:, 1]
            except Exception as e:
                logger.warning(f"StandardScaler 적용 실패: {e}. MinMax로 대체합니다.")
                return self.normalize_for_cost(delta_gv, delta_rpm)
        else:
            # Fallback: MinMax 정규화
            logger.debug("StandardScaler가 없어 MinMax 정규화를 사용합니다.")
            return self.normalize_for_cost(delta_gv, delta_rpm)

    # ====================================================================
    # 역정규화 메서드
    # ====================================================================

    def denormalize_control_vars(self,
                                gv_normalized: np.ndarray,
                                rpm_normalized: Union[float, np.ndarray]
                                ) -> Tuple[np.ndarray, Union[float, np.ndarray]]:
        """
        역정규화 (정규화된 값 → 원본 값)

        역정규화 공식 (MinMax 기반):
            delta_gv = gv_normalized * gv_max
            delta_rpm = rpm_normalized * rpm_max

        Args:
            gv_normalized: 정규화된 GV 값 [0, 1]
            rpm_normalized: 정규화된 RPM 값 [0, 1]

        Returns:
            (delta_gv, delta_rpm)

        Example:
            >>> gv_norm = np.array([0.25, 0.5, 1.0])
            >>> rpm_norm = 0.5
            >>> delta_gv, delta_rpm = normalizer.denormalize_control_vars(gv_norm, rpm_norm)
            >>> delta_gv  # [0.5, 1.0, 2.0]
            >>> delta_rpm  # 25.0
        """
        gv_normalized = np.asarray(gv_normalized)
        rpm_normalized = np.asarray(rpm_normalized)

        # 역정규화
        delta_gv = gv_normalized * self.gv_max
        delta_rpm = rpm_normalized * self.rpm_max

        # 스칼라로 반환
        if isinstance(delta_rpm, np.ndarray) and delta_rpm.size == 1:
            delta_rpm = float(delta_rpm)

        return delta_gv, delta_rpm

    # ====================================================================
    # 유틸리티 메서드
    # ====================================================================

    def get_config_dict(self) -> Dict[str, float]:
        """
        설정 사전 반환 (config.py의 CONTROL_COST_PARAMS와 동기화)

        Returns:
            dict: {'gv_max': float, 'rpm_max': float}

        Example:
            >>> config = normalizer.get_config_dict()
            >>> config['gv_max']  # 2.0
            >>> config['rpm_max']  # 50
        """
        return {
            'gv_max': self.gv_max,
            'rpm_max': self.rpm_max
        }

    def get_description(self) -> str:
        """정규화 설정 설명 반환"""
        return (f"ControlVariableNormalizer(gv_max={self.gv_max}, "
               f"rpm_max={self.rpm_max}, scaler={'있음' if self.scaler else '없음'})\n"
               f"정규화 방식:\n"
               f"  - MinMax (비용함수): normalized = |value| / max_value, 범위 [0, 1]\n"
               f"  - StandardScaler (예측모델): (value - μ) / σ")


# ============================================================================
# 테스트 코드
# ============================================================================

if __name__ == '__main__':
    import logging
    logging.basicConfig(level=logging.INFO)

    # 정규화 테스트
    normalizer = ControlVariableNormalizer(gv_max=2.0, rpm_max=50)

    # 테스트 데이터
    delta_gv = np.array([0.5, 1.0, 2.0])
    delta_rpm = 25.0

    print(normalizer.get_description())

    # MinMax 정규화
    gv_norm, rpm_norm = normalizer.normalize_for_cost(delta_gv, delta_rpm)
    print(f"\nMinMax 정규화 (비용함수):")
    print(f"  delta_gv: {delta_gv} → {gv_norm}")
    print(f"  delta_rpm: {delta_rpm} → {rpm_norm}")

    # 역정규화
    gv_back, rpm_back = normalizer.denormalize_control_vars(gv_norm, rpm_norm)
    print(f"\n역정규화:")
    print(f"  gv_norm: {gv_norm} → {gv_back}")
    print(f"  rpm_norm: {rpm_norm} → {rpm_back}")

    # 예측 모델용 정규화 (StandardScaler 없음)
    gv_pred, rpm_pred = normalizer.normalize_for_prediction(delta_gv, delta_rpm)
    print(f"\n예측모델용 정규화 (StandardScaler 없음, Fallback MinMax):")
    print(f"  delta_gv: {delta_gv} → {gv_pred}")
    print(f"  delta_rpm: {delta_rpm} → {rpm_pred}")
