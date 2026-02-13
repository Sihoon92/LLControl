"""
Per-Zone Probabilistic Dynamics Model

CatBoost의 MultiZoneController와 유사한 구조로
11개 Zone의 상태 전이를 예측하는 확률적 모델

주요 기능:
1. Zone별 입력 구성 (CatBoost와 동일)
2. 앙상블 예측
3. 불확실성 계산
4. CatBoost 인터페이스와 호환
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass

from .ensemble_nn import EnsembleWrapper
from ..config import N_ZONES, N_GV

logger = logging.getLogger(__name__)


@dataclass
class ZoneProperties:
    """Zone별 고정 속성 (CatBoost와 동일)"""
    zone_id: int
    zone_name: str
    distance: float
    edge_distance: float


class PerZoneProbabilisticEnsemble:
    """
    Per-Zone 확률적 앙상블 Dynamics Model

    CatBoost의 MultiZoneController와 동일한 방식으로
    각 Zone을 독립적으로 예측하되, 인접 정보는 입력에 포함

    차이점:
    - CatBoost: 점 예측 (단일 값)
    - PETS: 확률 분포 예측 (mean + uncertainty)
    """

    def __init__(
        self,
        n_ensembles: int = 5,
        input_dim: int = 11,
        output_dim: int = 3,
        hidden_dims: List[int] = [128, 128],
        zone_properties: Optional[List[ZoneProperties]] = None,
        device: str = 'cpu',
        **model_kwargs
    ):
        """
        Args:
            n_ensembles: 앙상블 개수
            input_dim: 입력 차원 (11)
            output_dim: 출력 차원 (3: diff_CLR)
            hidden_dims: Hidden layer 구조
            zone_properties: Zone 속성 (None이면 기본 생성)
            device: 디바이스
        """
        self.n_ensembles = n_ensembles
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.device = device

        # Zone 속성
        self.zone_properties = zone_properties or self._create_default_zone_properties()

        # 앙상블 래퍼
        self.ensemble = EnsembleWrapper(
            n_ensembles=n_ensembles,
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dims=hidden_dims,
            device=device,
            **model_kwargs
        )

        logger.info(f"Per-Zone Probabilistic Ensemble 초기화 (Zone 수: {N_ZONES}, 앙상블: {n_ensembles})")

    def _create_default_zone_properties(self) -> List[ZoneProperties]:
        """기본 Zone 속성 생성 (CatBoost와 동일)"""
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

    # ========================================================================
    # Zone 입력 구성 (CatBoost와 동일)
    # ========================================================================

    def _get_position_features(self, zone_i: int) -> np.ndarray:
        """
        위치 특성 추출 (4개)

        CatBoost의 _construct_single_zone_input와 동일
        """
        prop = self.zone_properties[zone_i]
        center = N_ZONES / 2

        zone_distance_from_center = abs(zone_i - (center - 0.5))
        normalized_position = zone_i / (N_ZONES - 1)
        normalized_distance = zone_distance_from_center / (center - 0.5)

        return np.array([
            prop.distance,
            prop.edge_distance,
            normalized_position,
            normalized_distance,
        ], dtype=np.float32)

    def _get_control_features(
        self,
        zone_i: int,
        delta_gv: np.ndarray,
        delta_rpm: float
    ) -> np.ndarray:
        """
        제어 특성 추출 (4개: 인접 GV + RPM)

        CatBoost와 동일: [GV_{i-1}, GV_i, GV_{i+1}, RPM]
        """
        # 인접 인덱스 (경계 처리)
        gv_idx_left = max(0, zone_i - 1)
        gv_idx_center = zone_i
        gv_idx_right = min(N_GV - 1, zone_i + 1)

        gv_left = delta_gv[gv_idx_left]
        gv_center = delta_gv[gv_idx_center]
        gv_right = delta_gv[gv_idx_right]

        return np.array([gv_left, gv_center, gv_right, delta_rpm], dtype=np.float32)

    def construct_zone_input(
        self,
        zone_i: int,
        current_clr: np.ndarray,
        delta_gv: np.ndarray,
        delta_rpm: float
    ) -> np.ndarray:
        """
        단일 Zone의 입력 벡터 구성

        Args:
            zone_i: Zone 인덱스 (0~10)
            current_clr: (3,) - 현재 CLR [Low, Mid, High]
            delta_gv: (11,) - GV 변화량
            delta_rpm: Scalar - RPM 변화량

        Returns:
            input_vector: (11,) - [위치(4), CLR(3), 제어(4)]
        """
        # 1. 위치 특성 (4개)
        position_features = self._get_position_features(zone_i)

        # 2. 현재 상태 (3개)
        state_features = current_clr.astype(np.float32)

        # 3. 제어 특성 (4개)
        control_features = self._get_control_features(zone_i, delta_gv, delta_rpm)

        # 결합
        input_vector = np.concatenate([
            position_features,   # 4개
            state_features,      # 3개
            control_features,    # 4개
        ])  # 총 11개

        return input_vector

    # ========================================================================
    # 예측 인터페이스
    # ========================================================================

    def predict_single_zone(
        self,
        zone_i: int,
        current_clr: np.ndarray,
        delta_gv: np.ndarray,
        delta_rpm: float,
        return_uncertainty: bool = True
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        단일 Zone 예측

        Args:
            zone_i: Zone 인덱스 (0~10)
            current_clr: (3,) - 현재 CLR
            delta_gv: (11,) - GV 변화량
            delta_rpm: Scalar - RPM 변화량
            return_uncertainty: 불확실성 반환 여부

        Returns:
            mean: (3,) - diff_CLR 평균
            uncertainty: (3,) - 불확실성 (옵션)
        """
        # 입력 구성
        x = self.construct_zone_input(zone_i, current_clr, delta_gv, delta_rpm)

        # Tensor 변환
        x_t = torch.FloatTensor(x).unsqueeze(0).to(self.device)  # (1, 11)

        # 앙상블 예측
        mean, uncertainty = self.ensemble.predict(x_t, return_uncertainty)

        # Numpy 변환
        mean = mean.squeeze(0).cpu().numpy()  # (3,)

        if uncertainty is not None:
            uncertainty = uncertainty.squeeze(0).cpu().numpy()  # (3,)

        return mean, uncertainty

    def predict_all_zones(
        self,
        current_clr_all: np.ndarray,
        delta_gv: np.ndarray,
        delta_rpm: float,
        return_uncertainty: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        전체 11개 Zone 예측

        CatBoost의 MultiZoneController.evaluate_control()와 유사

        Args:
            current_clr_all: (11, 3) - 모든 Zone의 현재 CLR
            delta_gv: (11,) - GV 변화량
            delta_rpm: Scalar - RPM 변화량
            return_uncertainty: 불확실성 반환 여부

        Returns:
            dict: {
                'diff_clr_mean': (11, 3),
                'diff_clr_uncertainty': (11, 3) or None,
                'next_clr': (11, 3),
            }
        """
        diff_clr_means = []
        diff_clr_uncertainties = []

        for zone_i in range(N_ZONES):
            current_clr = current_clr_all[zone_i]

            mean, uncertainty = self.predict_single_zone(
                zone_i, current_clr, delta_gv, delta_rpm, return_uncertainty
            )

            diff_clr_means.append(mean)

            if uncertainty is not None:
                diff_clr_uncertainties.append(uncertainty)

        # 결과 정리
        diff_clr_means = np.array(diff_clr_means)  # (11, 3)

        result = {
            'diff_clr_mean': diff_clr_means,
            'diff_clr_uncertainty': np.array(diff_clr_uncertainties) if diff_clr_uncertainties else None,
            'next_clr': current_clr_all + diff_clr_means,  # 다음 상태
        }

        return result

    # ========================================================================
    # 학습 인터페이스
    # ========================================================================

    def train_on_batch(
        self,
        batch_inputs: np.ndarray,
        batch_targets: np.ndarray
    ) -> dict:
        """
        배치 학습

        Args:
            batch_inputs: (batch, 11) - 입력 벡터들
            batch_targets: (batch, 3) - diff_CLR 타겟들

        Returns:
            metrics: 학습 메트릭
        """
        # Tensor 변환
        x = torch.FloatTensor(batch_inputs).to(self.device)
        y = torch.FloatTensor(batch_targets).to(self.device)

        # 앙상블 학습
        metrics = self.ensemble.train_step(x, y)

        return metrics

    def evaluate_on_batch(
        self,
        batch_inputs: np.ndarray,
        batch_targets: np.ndarray
    ) -> dict:
        """
        배치 평가

        Args:
            batch_inputs: (batch, 11)
            batch_targets: (batch, 3)

        Returns:
            metrics: 평가 메트릭
        """
        x = torch.FloatTensor(batch_inputs).to(self.device)
        y_true = batch_targets

        # 예측
        with torch.no_grad():
            y_pred, uncertainty = self.ensemble.predict(x, return_uncertainty=True)

        y_pred = y_pred.cpu().numpy()
        uncertainty = uncertainty.cpu().numpy()

        # 메트릭 계산
        errors = y_true - y_pred
        mse = np.mean(errors ** 2)
        mae = np.mean(np.abs(errors))
        rmse = np.sqrt(mse)

        # R² score
        ss_res = np.sum(errors ** 2)
        ss_tot = np.sum((y_true - y_true.mean()) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        metrics = {
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'mean_uncertainty': uncertainty.mean(),
        }

        return metrics

    # ========================================================================
    # 유틸리티
    # ========================================================================

    def save(self, path: str):
        """모델 저장"""
        self.ensemble.save(path)
        logger.info(f"모델 저장: {path}")

    def load(self, path: str):
        """모델 로드"""
        self.ensemble.load(path)
        logger.info(f"모델 로드: {path}")

    def init_optimizers(self, lr: float = 1e-3, weight_decay: float = 1e-5):
        """Optimizer 초기화"""
        self.ensemble.init_optimizers(lr, weight_decay)

    def get_model_info(self) -> dict:
        """모델 정보 반환"""
        total_params = sum(
            p.numel() for model in self.ensemble.models for p in model.parameters()
        )

        return {
            'n_ensembles': self.n_ensembles,
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'total_parameters': total_params,
            'parameters_per_model': total_params // self.n_ensembles,
        }


# ============================================================================
# 테스트 코드
# ============================================================================

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    print("="*80)
    print("Per-Zone Probabilistic Ensemble 테스트")
    print("="*80)

    # 모델 생성
    model = PerZoneProbabilisticEnsemble(
        n_ensembles=5,
        hidden_dims=[128, 128],
        device='cpu'
    )

    model.init_optimizers(lr=1e-3)

    # 모델 정보
    info = model.get_model_info()
    print(f"\n모델 정보:")
    for k, v in info.items():
        print(f"  {k}: {v}")

    # 더미 데이터
    current_clr_all = np.random.randn(11, 3)
    delta_gv = np.random.randn(11) * 0.5
    delta_rpm = np.random.randn() * 10

    # 단일 Zone 예측
    print(f"\n[단일 Zone 예측]")
    zone_i = 5
    mean, uncertainty = model.predict_single_zone(
        zone_i, current_clr_all[zone_i], delta_gv, delta_rpm
    )
    print(f"Zone {zone_i+1}:")
    print(f"  diff_CLR mean: {mean}")
    print(f"  uncertainty: {uncertainty}")

    # 전체 Zone 예측
    print(f"\n[전체 Zone 예측]")
    result = model.predict_all_zones(current_clr_all, delta_gv, delta_rpm)
    print(f"diff_clr_mean shape: {result['diff_clr_mean'].shape}")
    print(f"diff_clr_uncertainty shape: {result['diff_clr_uncertainty'].shape}")
    print(f"next_clr shape: {result['next_clr'].shape}")

    # 학습 테스트
    print(f"\n[학습 테스트]")
    batch_size = 32
    batch_inputs = np.random.randn(batch_size, 11)
    batch_targets = np.random.randn(batch_size, 3)

    metrics = model.train_on_batch(batch_inputs, batch_targets)
    print(f"Training metrics: {metrics}")

    # 평가 테스트
    eval_metrics = model.evaluate_on_batch(batch_inputs, batch_targets)
    print(f"Evaluation metrics: {eval_metrics}")

    print("\n✓ 모든 테스트 통과!")
