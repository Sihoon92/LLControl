"""
다중 Zone 제어기 (Multi-Zone Controller)

Fan-out 패턴:
  1개 제어값 [△GV₁, ..., △GV₁₁, △RPM] → 11개 Zone 입력 구성 → 배치 예측

각 Zone i의 입력 (총 11개 특성):
  - 위치 특성 (4개): distance_i, edge_distance_i, normalized_position, normalized_distance
  - 상태 특성 (3개): current_CLR_i (측정값)
  - 제어 특성 (4개): △GV_(i-1), △GV_i, △GV_(i+1), △RPM
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

from .config import (
    N_ZONES, N_GV, CLR_COMPONENTS, PROBABILITY_COMPONENTS,
    INPUT_FEATURE_PATTERNS
)
from .model_interface import CatBoostModelManager

logger = logging.getLogger(__name__)


@dataclass
class ZoneProperties:
    """Zone별 고정 속성"""
    zone_id: int              # 1~11
    zone_name: str            # "Zone01" ~ "Zone11"
    distance: float           # 상대 거리 (mm 단위)
    edge_distance: float      # 중앙에서의 거리


class MultiZoneController:
    """
    다중 Zone 제어기

    제어값 [△GV₁, ..., △GV₁₁, △RPM] → 11개 Zone의 최종 확률 분포로 변환
    """

    def __init__(self,
                 model_manager: CatBoostModelManager,
                 zone_properties: Optional[List[ZoneProperties]] = None):
        """
        초기화

        Args:
            model_manager: CatBoostModelManager 인스턴스
            zone_properties: Zone별 고정 속성 (없으면 생성)
        """
        self.model = model_manager
        self.zone_properties = zone_properties or self._create_default_zone_properties()

        # Zone 입력 특성 구성
        self._construct_input_features()

        logger.info(f"MultiZoneController 초기화 (Zone 수: {N_ZONES})")

    def _create_default_zone_properties(self) -> List[ZoneProperties]:
        """
        기본 Zone 속성 생성 (실제 공정값은 별도로 설정 필요)
        """
        properties = []
        center = N_ZONES / 2

        for i in range(N_ZONES):
            zone_id = i + 1
            properties.append(ZoneProperties(
                zone_id=zone_id,
                zone_name=f"Zone{zone_id:02d}",
                distance=i * 100,  # 100mm 간격
                edge_distance=abs(i - (center - 0.5))
            ))

        return properties

    def _construct_input_features(self):
        """
        모델의 입력 특성 구조 파악

        현재 코드에서 각 Zone의 입력 특성 구조:
          Zone당 11개 특성:
            - 위치 특성: 4개 (distance, edge_distance, normalized_position, normalized_distance)
            - 상태 특성: 3개 (current_CLR_1, current_CLR_2, current_CLR_3)
            - 제어 특성: 4개 (GV_{i-1}, GV_i, GV_{i+1}, RPM)

          배치 예측: 11개 Zone × 11개 특성 = 총 121개 특성 (배치 크기 제외)

        실제로는 모델 입력 특성명을 로드해야 함
        """
        # TODO: 모델에서 입력 특성명 조회
        self.input_feature_names = None

    # ========================================================================
    # Zone 입력 벡터 구성 (Fan-out)
    # ========================================================================

    def construct_zone_inputs(self,
                             x: np.ndarray,
                             current_state: Dict[str, np.ndarray]) -> np.ndarray:
        """
        제어값 x로부터 각 Zone의 입력 벡터 구성

        Args:
            x: Shape (12,) - [△GV₁, ..., △GV₁₁, △RPM]
            current_state: 현재 상태 dict
              {
                'current_clr': Shape (n_zones, 3),  # CLR_1, CLR_2, CLR_3
              }

        Returns:
            Shape (n_zones, n_features) - 각 Zone의 모델 입력
        """
        # 제어값 분해
        delta_gv = x[:N_GV]      # Shape (11,)
        delta_rpm = x[N_GV]       # Scalar

        # 현재 CLR 상태
        current_clr = current_state.get('current_clr', np.zeros((N_ZONES, 3)))

        # Zone별 입력 구성
        zone_inputs_list = []

        for i in range(N_ZONES):
            zone_input = self._construct_single_zone_input(
                zone_id=i,
                current_clr_values=current_clr[i],  # Shape (3,)
                delta_gv=delta_gv,
                delta_rpm=delta_rpm
            )
            zone_inputs_list.append(zone_input)

        # 전체 Zone 입력 배열
        zone_inputs = np.array(zone_inputs_list)  # Shape (n_zones, n_features)

        return zone_inputs

    def _construct_single_zone_input(self,
                                     zone_id: int,
                                     current_clr_values: np.ndarray,
                                     delta_gv: np.ndarray,
                                     delta_rpm: float) -> np.ndarray:
        """
        단일 Zone의 입력 벡터 구성

        입력 구성:
          [Zone_위치_특성(4): distance, edge_distance, normalized_position, normalized_distance]
          + [현재_CLR(3)]
          + [인접_GV_변화(3): GV_{i-1}, GV_i, GV_{i+1}]
          + [RPM_변화(1)]
          = 총 11개

        Args:
            zone_id: 0~10
            current_clr_values: Shape (3,)
            delta_gv: Shape (11,)
            delta_rpm: Scalar

        Returns:
            Shape (11,) - Zone 입력 벡터
        """
        zone_prop = self.zone_properties[zone_id]
        center = N_ZONES / 2

        # 위치 특성 (4개)
        zone_distance_from_center = abs(zone_id - (center - 0.5))
        is_edge = 1.0 if (zone_id == 0 or zone_id == N_ZONES - 1) else 0.0
        normalized_position = zone_id / (N_ZONES - 1)  # 0~1 정규화
        normalized_distance = zone_distance_from_center / (center - 0.5)  # 0~1 정규화

        position_features = np.array([
            zone_prop.distance,
            zone_prop.edge_distance,
            normalized_position,
            normalized_distance,
        ])

        # 현재 상태 특성 (CLR)
        state_features = current_clr_values  # Shape (3,)

        # 제어 특성 - 인접 GV (i-1, i, i+1)
        # 경계 처리: 0, 1, ..., 10
        gv_idx_left = max(0, zone_id - 1)
        gv_idx_center = zone_id
        gv_idx_right = min(N_GV - 1, zone_id + 1)

        control_features_gv = np.array([
            delta_gv[gv_idx_left],
            delta_gv[gv_idx_center],
            delta_gv[gv_idx_right],
        ])

        # RPM 특성
        control_features_rpm = np.array([delta_rpm])

        # 전체 특성 결합
        zone_input = np.concatenate([
            position_features,           # 4개
            state_features,              # 3개
            control_features_gv,         # 3개
            control_features_rpm,        # 1개
        ])  # 총 11개

        return zone_input

    # ========================================================================
    # 배치 예측
    # ========================================================================

    def batch_predict(self, zone_inputs: np.ndarray) -> np.ndarray:
        """
        Zone 입력에 대한 배치 예측

        Args:
            zone_inputs: Shape (n_zones, n_features)

        Returns:
            Shape (n_zones, 3) - 각 Zone의 [△CLR_1, △CLR_2, △CLR_3]
        """
        # 모델 예측
        predictions = self.model.predict_batch(zone_inputs)

        return predictions

    # ========================================================================
    # Inverse CLR 변환
    # ========================================================================

    def apply_inverse_clr(self,
                         current_clr: np.ndarray,
                         predicted_delta_clr: np.ndarray) -> np.ndarray:
        """
        Inverse CLR 변환

        Args:
            current_clr: Shape (n_zones, 3)
            predicted_delta_clr: Shape (n_zones, 3)

        Returns:
            Shape (n_zones, 3) - [P_Low, P_Mid, P_High]
        """
        probabilities = self.model.apply_inverse_clr_transform(
            current_clr, predicted_delta_clr
        )

        return probabilities

    # ========================================================================
    # 전체 제어 평가
    # ========================================================================

    def evaluate_control(self,
                        x: np.ndarray,
                        current_state: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """
        제어값 x에 대한 전체 평가

        Args:
            x: Shape (12,) - [△GV₁, ..., △GV₁₁, △RPM]
            current_state: 현재 상태 dict

        Returns:
            dict: {
                'p_low': Shape (n_zones,),
                'p_mid': Shape (n_zones,),
                'p_high': Shape (n_zones,),
                'probabilities': Shape (n_zones, 3),
                'delta_clr': Shape (n_zones, 3),
                'zone_inputs': Shape (n_zones, n_features),
            }
        """
        # 1. Zone 입력 구성
        zone_inputs = self.construct_zone_inputs(x, current_state)

        # 2. 배치 예측 (△CLR 예측)
        predicted_delta_clr = self.batch_predict(zone_inputs)

        # 3. Inverse CLR 변환
        current_clr = current_state.get('current_clr', np.zeros((N_ZONES, 3)))
        probabilities = self.apply_inverse_clr(current_clr, predicted_delta_clr)

        # 4. 확률 분해
        p_low = probabilities[:, 0]
        p_mid = probabilities[:, 1]
        p_high = probabilities[:, 2]

        # 5. 결과 정리
        result = {
            'p_low': p_low,
            'p_mid': p_mid,
            'p_high': p_high,
            'probabilities': probabilities,
            'delta_clr': predicted_delta_clr,
            'zone_inputs': zone_inputs,
        }

        return result

    # ========================================================================
    # 유틸리티
    # ========================================================================

    def print_zone_summary(self):
        """
        Zone 정보 출력
        """
        print(f"\n{'='*60}")
        print(f"Zone 설정 정보 (총 {N_ZONES}개 Zone)")
        print(f"{'='*60}")
        print(f"{'Zone':<10} {'Zone ID':<10} {'Distance':<12} {'Edge Dist':<12}")
        print(f"{'-'*60}")
        for prop in self.zone_properties:
            print(f"{prop.zone_name:<10} {prop.zone_id:<10} {prop.distance:<12.1f} {prop.edge_distance:<12.1f}")
        print(f"{'='*60}\n")

    def get_zone_properties_dict(self) -> Dict[str, Dict]:
        """
        Zone 속성을 딕셔너리로 반환
        """
        return {
            prop.zone_name: {
                'zone_id': prop.zone_id,
                'distance': prop.distance,
                'edge_distance': prop.edge_distance,
            }
            for prop in self.zone_properties
        }


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

    # 모델 매니저 로드
    model_manager = CatBoostModelManager()

    # 제어기 생성
    controller = MultiZoneController(model_manager)
    controller.print_zone_summary()

    # 테스트: 제어값
    x_test = np.concatenate([
        np.random.uniform(-0.5, 0.5, N_GV),  # △GV
        np.array([10])                        # △RPM
    ])

    # 현재 상태 (임의 CLR 값)
    current_state = {
        'current_clr': np.random.randn(N_ZONES, 3)
    }

    # 평가
    result = controller.evaluate_control(x_test, current_state)

    print(f"\n제어 평가 결과:")
    print(f"  P_Low 범위: [{result['p_low'].min():.4f}, {result['p_low'].max():.4f}]")
    print(f"  P_Mid 범위: [{result['p_mid'].min():.4f}, {result['p_mid'].max():.4f}]")
    print(f"  P_High 범위: [{result['p_high'].min():.4f}, {result['p_high'].max():.4f}]")
    print(f"  확률 합: {np.sum(result['probabilities'], axis=1)}")
