"""
Per-Zone Dynamics Neural Network

CatBoost와 동일한 입력/출력 구조를 사용하는 Dynamics Model

입력 (11개):
  - 위치 특성 (4): distance, edge_distance, normalized_position, normalized_distance
  - 현재 CLR (3): CLR_1, CLR_2, CLR_3
  - 제어 변화 (4): delta_GV_{i-1}, delta_GV_i, delta_GV_{i+1}, delta_RPM

출력 (3 × 2 = 6개):
  - diff_CLR mean (3): mean_1, mean_2, mean_3
  - diff_CLR log_var (3): log_var_1, log_var_2, log_var_3
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional
import numpy as np


class PerZoneDynamicsNN(nn.Module):
    """
    Per-Zone Dynamics Neural Network

    단일 Zone의 상태 전이를 예측하는 확률적 신경망

    아키텍처:
        Input (11) → Hidden Layers → Output (6: mean + log_var)
    """

    def __init__(
        self,
        input_dim: int = 11,
        output_dim: int = 3,
        hidden_dims: List[int] = [128, 128],
        activation: str = 'relu',
        use_layer_norm: bool = True,
        dropout: float = 0.0,
        log_var_min: float = -10,
        log_var_max: float = 2
    ):
        """
        Args:
            input_dim: 입력 차원 (11: 위치 4 + CLR 3 + 제어 4)
            output_dim: 출력 차원 (3: diff_CLR)
            hidden_dims: Hidden layer 차원 리스트
            activation: 활성화 함수 ('relu', 'tanh', 'elu')
            use_layer_norm: LayerNorm 사용 여부
            dropout: Dropout 비율
            log_var_min: Log variance 하한
            log_var_max: Log variance 상한
        """
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.log_var_min = log_var_min
        self.log_var_max = log_var_max

        # 활성화 함수 선택
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'elu':
            self.activation = nn.ELU()
        else:
            raise ValueError(f"Unknown activation: {activation}")

        # 네트워크 구성
        layers = []
        prev_dim = input_dim

        for i, hidden_dim in enumerate(hidden_dims):
            # Linear layer
            layers.append(nn.Linear(prev_dim, hidden_dim))

            # Activation
            layers.append(self.activation)

            # LayerNorm (옵션)
            if use_layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))

            # Dropout (옵션)
            if dropout > 0:
                layers.append(nn.Dropout(dropout))

            prev_dim = hidden_dim

        # 출력 레이어 (mean + log_var)
        output_total = output_dim * 2  # mean (3) + log_var (3) = 6
        layers.append(nn.Linear(prev_dim, output_total))

        self.network = nn.Sequential(*layers)

        # 가중치 초기화
        self._init_weights()

    def _init_weights(self):
        """He initialization"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(
        self,
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass

        Args:
            x: (batch, input_dim) - [위치(4), CLR(3), 제어(4)]

        Returns:
            mean: (batch, output_dim) - diff_CLR 평균
            log_var: (batch, output_dim) - diff_CLR 로그 분산
        """
        # 네트워크 통과
        output = self.network(x)

        # mean과 log_var 분리
        mean, log_var = torch.chunk(output, 2, dim=-1)

        # log_var 안정화 (너무 크거나 작지 않도록)
        log_var = torch.clamp(log_var, min=self.log_var_min, max=self.log_var_max)

        return mean, log_var

    def predict(
        self,
        x: torch.Tensor,
        deterministic: bool = False
    ) -> torch.Tensor:
        """
        예측 (샘플링 옵션)

        Args:
            x: (batch, input_dim)
            deterministic: True면 평균만, False면 샘플링

        Returns:
            prediction: (batch, output_dim)
        """
        mean, log_var = self.forward(x)

        if deterministic:
            return mean
        else:
            # Reparameterization trick
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            sample = mean + eps * std
            return sample

    def compute_loss(
        self,
        x: torch.Tensor,
        y: torch.Tensor
    ) -> Tuple[torch.Tensor, dict]:
        """
        Negative Log-Likelihood Loss

        Loss = 0.5 * [log(2π) + log(σ²) + (y - μ)² / σ²]

        Args:
            x: (batch, input_dim) - 입력
            y: (batch, output_dim) - 타겟 (diff_CLR)

        Returns:
            loss: Scalar
            info: 상세 정보 dict
        """
        mean, log_var = self.forward(x)

        # NLL Loss
        inv_var = torch.exp(-log_var)
        mse_loss = ((y - mean) ** 2) * inv_var
        var_loss = log_var

        # Total loss (평균)
        nll = 0.5 * (mse_loss + var_loss).mean()

        # 추가 정보
        with torch.no_grad():
            mse = ((y - mean) ** 2).mean()
            mae = torch.abs(y - mean).mean()
            mean_log_var = log_var.mean()

        info = {
            'nll': nll.item(),
            'mse': mse.item(),
            'mae': mae.item(),
            'mean_log_var': mean_log_var.item(),
        }

        return nll, info


class EnsembleWrapper:
    """
    앙상블 래퍼 클래스

    여러 개의 PerZoneDynamicsNN을 관리하고 앙상블 예측 수행
    """

    def __init__(
        self,
        n_ensembles: int = 5,
        input_dim: int = 11,
        output_dim: int = 3,
        hidden_dims: List[int] = [128, 128],
        device: str = 'cpu',
        **model_kwargs
    ):
        """
        Args:
            n_ensembles: 앙상블 개수
            input_dim: 입력 차원
            output_dim: 출력 차원
            hidden_dims: Hidden layer 차원
            device: 디바이스 ('cpu', 'cuda', 'mps')
            **model_kwargs: PerZoneDynamicsNN에 전달할 추가 인자
        """
        self.n_ensembles = n_ensembles
        self.device = device

        # 앙상블 모델 생성
        self.models = [
            PerZoneDynamicsNN(
                input_dim=input_dim,
                output_dim=output_dim,
                hidden_dims=hidden_dims,
                **model_kwargs
            ).to(device)
            for _ in range(n_ensembles)
        ]

        # Optimizer (각 모델별)
        self.optimizers = None

    def init_optimizers(
        self,
        lr: float = 1e-3,
        weight_decay: float = 1e-5
    ):
        """Optimizer 초기화"""
        self.optimizers = [
            torch.optim.Adam(
                model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
            for model in self.models
        ]

    def train_step(
        self,
        x: torch.Tensor,
        y: torch.Tensor
    ) -> dict:
        """
        앙상블 학습 1 step

        Args:
            x: (batch, input_dim)
            y: (batch, output_dim)

        Returns:
            metrics: 평균 메트릭
        """
        total_loss = 0.0
        total_mse = 0.0
        total_mae = 0.0

        for model, optimizer in zip(self.models, self.optimizers):
            model.train()
            optimizer.zero_grad()

            # Forward & Loss
            loss, info = model.compute_loss(x, y)

            # Backward
            loss.backward()

            # Gradient clipping (옵션)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # Update
            optimizer.step()

            total_loss += info['nll']
            total_mse += info['mse']
            total_mae += info['mae']

        # 평균
        metrics = {
            'loss': total_loss / self.n_ensembles,
            'mse': total_mse / self.n_ensembles,
            'mae': total_mae / self.n_ensembles,
        }

        return metrics

    def predict(
        self,
        x: torch.Tensor,
        return_uncertainty: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        앙상블 예측

        Args:
            x: (batch, input_dim)
            return_uncertainty: 불확실성 반환 여부

        Returns:
            mean: (batch, output_dim) - 앙상블 평균
            uncertainty: (batch, output_dim) - 총 불확실성 (옵션)
        """
        means = []
        variances = []

        for model in self.models:
            model.eval()
            with torch.no_grad():
                mean, log_var = model(x)
                means.append(mean)
                variances.append(torch.exp(log_var))

        # 앙상블 통계
        means = torch.stack(means, dim=0)  # (n_ens, batch, output_dim)
        variances = torch.stack(variances, dim=0)

        ensemble_mean = means.mean(dim=0)  # (batch, output_dim)

        if return_uncertainty:
            # 총 불확실성 = Aleatoric + Epistemic
            aleatoric = variances.mean(dim=0)  # 평균 분산
            epistemic = means.var(dim=0)       # 예측 간 분산
            total_uncertainty = aleatoric + epistemic

            return ensemble_mean, total_uncertainty

        return ensemble_mean, None

    def save(self, path: str):
        """앙상블 모델 저장"""
        state = {
            'n_ensembles': self.n_ensembles,
            'models': [model.state_dict() for model in self.models],
        }
        torch.save(state, path)

    def load(self, path: str):
        """앙상블 모델 로드"""
        state = torch.load(path, map_location=self.device)

        if state['n_ensembles'] != self.n_ensembles:
            raise ValueError(
                f"Ensemble size mismatch: {state['n_ensembles']} vs {self.n_ensembles}"
            )

        for model, state_dict in zip(self.models, state['models']):
            model.load_state_dict(state_dict)


# ============================================================================
# 테스트 코드
# ============================================================================

if __name__ == '__main__':
    # 단일 모델 테스트
    print("="*80)
    print("Per-Zone Dynamics NN 테스트")
    print("="*80)

    model = PerZoneDynamicsNN(
        input_dim=11,
        output_dim=3,
        hidden_dims=[128, 128]
    )

    # 더미 데이터
    batch_size = 32
    x = torch.randn(batch_size, 11)
    y = torch.randn(batch_size, 3)

    # Forward
    mean, log_var = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output mean shape: {mean.shape}")
    print(f"Output log_var shape: {log_var.shape}")

    # Loss
    loss, info = model.compute_loss(x, y)
    print(f"\nLoss: {loss.item():.6f}")
    print(f"Info: {info}")

    # 앙상블 테스트
    print("\n" + "="*80)
    print("Ensemble 테스트")
    print("="*80)

    ensemble = EnsembleWrapper(
        n_ensembles=5,
        input_dim=11,
        output_dim=3,
        hidden_dims=[128, 128],
        device='cpu'
    )

    ensemble.init_optimizers(lr=1e-3)

    # 학습 step
    metrics = ensemble.train_step(x, y)
    print(f"Training metrics: {metrics}")

    # 예측
    pred_mean, pred_uncertainty = ensemble.predict(x, return_uncertainty=True)
    print(f"\nPrediction mean shape: {pred_mean.shape}")
    print(f"Uncertainty shape: {pred_uncertainty.shape}")
    print(f"Mean uncertainty: {pred_uncertainty.mean().item():.6f}")

    # 파라미터 수
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n단일 모델 파라미터 수: {total_params:,}")
    print(f"앙상블 총 파라미터 수: {total_params * 5:,}")
