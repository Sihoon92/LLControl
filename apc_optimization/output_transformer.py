"""
출력 변환기 (Output Transformer)

모델 예측값의 후처리 및 정수화를 담당
- 모델별 다른 변환 규칙 적용 가능
- Config 기반 중앙 관리
- 선택적으로 활성화/비활성화 가능

주요 기능:
1. 정수화 (Rounding/Floor/Ceil)
2. 경계값 클램핑 (Bounds Clamping)
3. 확률 정규화 (Probability Normalization)
"""

import numpy as np
import logging
from typing import Dict, Optional, Literal, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class TransformConfig:
    """변환 설정"""
    enable: bool = False                                    # 변환 활성화 여부
    apply_delta_gv: bool = True                             # GV 변화량 변환
    delta_gv_method: Literal['round', 'floor', 'ceil'] = 'round'  # 변환 방식
    delta_gv_decimals: int = 0                              # 소수점 자리
    apply_delta_rpm: bool = False                           # RPM 변화량 변환
    delta_rpm_method: Literal['round', 'floor', 'ceil'] = 'round'
    delta_rpm_decimals: int = 0
    validate_bounds: bool = True                            # 경계값 재검증
    clamp_bounds: bool = True                               # 경계값 클램핑
    gv_bounds: Tuple[float, float] = (-2.0, 2.0)           # GV 경계
    rpm_bounds: Tuple[float, float] = (-50.0, 50.0)        # RPM 경계


class OutputTransformer:
    """
    모델 출력 후처리 변환기

    정수화, 경계값 처리, 정규화 등을 수행
    """

    def __init__(self, config: Optional[TransformConfig] = None):
        """
        초기화

        Args:
            config: 변환 설정. None이면 기본값 사용
        """
        self.config = config or TransformConfig()
        self.violation_count = 0

        if self.config.enable:
            logger.info("OutputTransformer 활성화됨")
            logger.info(f"  - GV 변환: {self.config.delta_gv_method} "
                       f"({self.config.delta_gv_decimals} decimals)")
            logger.info(f"  - RPM 변환: {self.config.delta_rpm_method} "
                       f"({self.config.delta_rpm_decimals} decimals)")
            logger.info(f"  - 경계값 재검증: {self.config.validate_bounds}")

    def transform(self,
                  x: np.ndarray,
                  x_type: str = 'control_vector') -> np.ndarray:
        """
        제어값 변환

        Args:
            x: Shape (12,) = [△GV₁~₁₁, △RPM]
            x_type: 'control_vector' - 제어값 벡터

        Returns:
            변환된 제어값
        """
        if not self.config.enable:
            return x

        if x_type != 'control_vector':
            return x

        x_transformed = x.copy()

        # △GV 변환 (처음 11개)
        if self.config.apply_delta_gv:
            x_transformed[:11] = self._transform_value(
                x_transformed[:11],
                method=self.config.delta_gv_method,
                decimals=self.config.delta_gv_decimals,
                bounds=self.config.gv_bounds,
                name='GV'
            )

        # △RPM 변환 (마지막 1개)
        if self.config.apply_delta_rpm:
            x_transformed[11] = self._transform_value(
                np.array([x_transformed[11]]),
                method=self.config.delta_rpm_method,
                decimals=self.config.delta_rpm_decimals,
                bounds=self.config.rpm_bounds,
                name='RPM'
            )[0]

        # 경계값 클램핑
        if self.config.clamp_bounds:
            x_transformed = self._clamp_bounds(x_transformed)

        return x_transformed

    def _transform_value(self,
                        values: np.ndarray,
                        method: str,
                        decimals: int,
                        bounds: Tuple[float, float],
                        name: str = 'Value') -> np.ndarray:
        """
        값 변환 수행

        Args:
            values: 변환할 값들
            method: 'round', 'floor', 'ceil'
            decimals: 소수점 자리
            bounds: (min, max) 경계
            name: 로깅용 이름

        Returns:
            변환된 값들
        """
        result = values.copy()

        # 변환 수행
        if decimals == 0:
            # 정수화
            if method == 'round':
                result = np.round(result)
            elif method == 'floor':
                result = np.floor(result)
            elif method == 'ceil':
                result = np.ceil(result)
        else:
            # 소수점 자리까지 변환
            multiplier = 10 ** decimals
            if method == 'round':
                result = np.round(result * multiplier) / multiplier
            elif method == 'floor':
                result = np.floor(result * multiplier) / multiplier
            elif method == 'ceil':
                result = np.ceil(result * multiplier) / multiplier

        # 경계값 검증
        if self.config.validate_bounds:
            violations = np.where((result < bounds[0]) | (result > bounds[1]))[0]
            if len(violations) > 0:
                self.violation_count += len(violations)
                logger.warning(
                    f"{name} 경계 위반: {len(violations)}개 값이 "
                    f"[{bounds[0]}, {bounds[1]}] 범위를 벗어남"
                )

        return result

    def _clamp_bounds(self, x: np.ndarray) -> np.ndarray:
        """경계값에 따라 클램핑"""
        x_clamped = x.copy()

        # △GV 클램핑 (처음 11개)
        gv_min, gv_max = self.config.gv_bounds
        x_clamped[:11] = np.clip(x_clamped[:11], gv_min, gv_max)

        # △RPM 클램핑 (마지막 1개)
        rpm_min, rpm_max = self.config.rpm_bounds
        x_clamped[11] = np.clip(x_clamped[11], rpm_min, rpm_max)

        return x_clamped

    def get_statistics(self) -> Dict:
        """변환 통계 반환"""
        return {
            'violation_count': self.violation_count,
            'enabled': self.config.enable,
            'gv_method': self.config.delta_gv_method,
            'rpm_method': self.config.delta_rpm_method,
        }

    def reset_statistics(self):
        """통계 초기화"""
        self.violation_count = 0


# ============================================================================
# 테스트 코드
# ============================================================================

if __name__ == '__main__':
    import sys
    from pathlib import Path

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # 테스트 1: 비활성화 (변환 없음)
    print("\n" + "="*60)
    print("Test 1: 변환 비활성화")
    print("="*60)
    transformer_disabled = OutputTransformer(TransformConfig(enable=False))
    x_test = np.array([0.4, 0.6, -0.3, 0.7, 0.1, -0.5, 0.2, 0.8, -0.1, 0.3, 0.5, 15.5])
    x_result = transformer_disabled.transform(x_test)
    print(f"Input:  {x_test}")
    print(f"Output: {x_result}")
    print(f"변화 없음: {np.allclose(x_test, x_result)}")

    # 테스트 2: 정수화 활성화
    print("\n" + "="*60)
    print("Test 2: 정수화 활성화 (round)")
    print("="*60)
    config = TransformConfig(
        enable=True,
        apply_delta_gv=True,
        delta_gv_method='round',
        delta_gv_decimals=0,
        apply_delta_rpm=True,
        delta_rpm_method='round',
        delta_rpm_decimals=0,
        clamp_bounds=True
    )
    transformer_enabled = OutputTransformer(config)
    x_result = transformer_enabled.transform(x_test)
    print(f"Input:  {x_test}")
    print(f"Output: {x_result}")
    print(f"통계: {transformer_enabled.get_statistics()}")

    # 테스트 3: 경계 위반
    print("\n" + "="*60)
    print("Test 3: 경계 위반 검증")
    print("="*60)
    x_violation = np.array([3.0, -3.0, 0.5, 1.0, -1.0, 0.0, 0.5, 1.0, -0.5, 0.2, 0.3, 100.0])
    print(f"Input (경계 위반): {x_violation}")
    x_result = transformer_enabled.transform(x_violation)
    print(f"Output (클램핑 후): {x_result}")
    print(f"통계: {transformer_enabled.get_statistics()}")

    # 테스트 4: Floor 변환
    print("\n" + "="*60)
    print("Test 4: Floor 변환")
    print("="*60)
    config_floor = TransformConfig(
        enable=True,
        apply_delta_gv=True,
        delta_gv_method='floor',
        delta_gv_decimals=0,
        clamp_bounds=False
    )
    transformer_floor = OutputTransformer(config_floor)
    x_result = transformer_floor.transform(x_test)
    print(f"Input:  {x_test[:11]} (GV only)")
    print(f"Output: {x_result[:11]}")
