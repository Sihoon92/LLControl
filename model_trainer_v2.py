import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
import os
import pickle
import logging
from datetime import datetime
from pathlib import Path
import sys

# 프로젝트 루트를 경로에 추가 (utils 임포트용)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import utils

# 모델 라이브러리
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor, RegressorChain

# Gaussian Process
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Matern, WhiteKernel

# XGBoost
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Warning: XGBoost not installed. Install with: pip install xgboost")

# CatBoost
try:
    from catboost import CatBoostRegressor
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    print("Warning: CatBoost not installed. Install with: pip install catboost")

# Random Forest
from sklearn.ensemble import RandomForestRegressor

# Neural Network
from sklearn.neural_network import MLPRegressor

# PyTorch for Custom MLP
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import TensorDataset, DataLoader
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    print("Warning: PyTorch not installed. Install with: pip install torch")

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

class ConstrainedMLP(nn.Module):
    """
    물리적 제약 조건을 고려한 MLP
    출력의 합이 0에 가까워지도록 제약
    """
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: List[int] = [128, 64]):
        super(ConstrainedMLP, self).__init__()

        layers = []
        prev_dim = input_dim

        # Hidden layers
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            prev_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class ConstrainedLoss(nn.Module):
    """
    물리적 제약을 포함한 손실 함수
    MSE + 합 제약 + 분산 제약
    """
    def __init__(self, sum_weight: float = 1.0, variance_weight: float = 0.1):
        super(ConstrainedLoss, self).__init__()
        self.sum_weight = sum_weight
        self.variance_weight = variance_weight
        self.mse = nn.MSELoss()

    def forward(self, pred, target):
        # 기본 MSE
        mse_loss = self.mse(pred, target)

        # 합 제약: 예측값의 합이 0에 가까워야 함 (CLR 특성상)
        sum_constraint = torch.mean(torch.abs(torch.sum(pred, dim=1)))

        # 분산 제약: 타겟의 분산과 예측의 분산이 유사해야 함
        target_var = torch.var(target, dim=1)
        pred_var = torch.var(pred, dim=1)
        variance_loss = torch.mean(torch.abs(target_var - pred_var))

        # 총 손실
        total_loss = mse_loss + self.sum_weight * sum_constraint + self.variance_weight * variance_loss

        return total_loss, mse_loss, sum_constraint, variance_loss

class ModelTrainer:
    """
    모델 학습 및 평가 클래스 (다변량 예측 기능 추가)
    """
    def __init__(
        self,
        data_file: str,
        output_dir: str = './outputs/models',
        logger: logging.Logger = None,
        random_state: int = 42
    ):
        """
        Parameters:
        -----------
        data_file : str
            모델 학습용 데이터 파일 경로
        output_dir : str
            결과 저장 디렉토리
        logger : logging.Logger
            로거 객체
        random_state : int
            랜덤 시드
        """
        self.data_file = data_file
        self.output_dir = output_dir
        self.random_state = random_state

        # 로거 설정
        if logger is None:
            self.logger = self._setup_logger()
        else:
            self.logger = logger

        # 출력 디렉토리 생성
        os.makedirs(output_dir, exist_ok=True)

        # 데이터 로드
        self.data = None
        self.X_train = None
        self.X_test = None
        self.Y_train = None
        self.Y_test = None
        self.X_train_scaled = None
        self.X_test_scaled = None
        self.scaler = None

        # 모델 저장
        self.models = {}
        self.predictions = {}
        self.metrics = {}

        # 특성 이름
        self.input_features = []
        self.output_features = []

    def _setup_logger(self):
        """로거 설정"""
        logger = logging.getLogger('model_trainer')
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )

            # 콘솔 핸들러
            ch = logging.StreamHandler()
            ch.setFormatter(formatter)
            logger.addHandler(ch)

            # 파일 핸들러
            log_file = os.path.join(self.output_dir, 'training.log')
            fh = logging.FileHandler(log_file, encoding='utf-8')
            fh.setFormatter(formatter)
            logger.addHandler(fh)

        return logger

    def load_and_prepare_data(
        self,
        test_size: float = 0.3,
        scale_features: bool = True
    ):
        """
        데이터 로드 및 전처리
        """
        self.logger.info("="*80)
        self.logger.info("데이터 로드 및 전처리")
        self.logger.info("="*80)

        # 데이터 로드 (xlwings 기반 load_excel_file 사용)
        self.logger.info(f"데이터 파일: {self.data_file}")
        self.data = utils.load_excel_file(self.data_file, logger=self.logger)
        self.logger.info(f"  ✓ {len(self.data)} 샘플 로드")

        # None 컬럼 이름 처리 (xlwings 로드 시 발생할 수 있는 unnamed 컬럼)
        none_cols = [col for col in self.data.columns if col is None]
        if none_cols:
            rename_map = {col: f'unnamed_{i}' for i, col in enumerate(none_cols)}
            self.data.rename(columns=rename_map, inplace=True)
            self.logger.info(f"  None 컬럼 {len(none_cols)}개 → 임시 이름 부여: {list(rename_map.values())}")

        # 입력/출력 특성 분리
        # position_features: Zone 위치 관련 물리적 특성
        position_features = [col for col in self.data.columns if col in [
            'zone_distance_from_center', 'is_edge',
            'normalized_position', 'normalized_distance'
        ]]

        # state_features: 현재 공정 상태 (current CLR)
        state_features = [col for col in self.data.columns if 'current_CLR' in col]

        # global_features: 전역 제어 변수 (RPM)
        global_features = [col for col in self.data.columns if 'delta_RPM' in col]

        # local_features: 국소 제어 변수 (GV 변화량, 자신 및 인접 zone)
        local_features = [col for col in self.data.columns if 'delta_GV' in col]

        self.input_features = position_features + state_features + global_features + local_features

        self.output_features = [col for col in self.data.columns if 'diff_CLR' in col]

        self.logger.info(f"입력 특성: {len(self.input_features)}개")
        self.logger.info(f"  - position_features ({len(position_features)}개): {position_features}")
        self.logger.info(f"  - state_features    ({len(state_features)}개): {state_features}")
        self.logger.info(f"  - global_features   ({len(global_features)}개): {global_features}")
        self.logger.info(f"  - local_features    ({len(local_features)}개): {local_features}")
        self.logger.info(f"출력 특성: {len(self.output_features)}개: {self.output_features}")

        # X, Y 분리
        X = self.data[self.input_features].values
        Y = self.data[self.output_features].values

        # Train/Test 분리
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(
            X, Y,
            test_size=test_size,
            random_state=self.random_state
        )

        self.logger.info(f"학습 데이터: {len(self.X_train)} 샘플")
        self.logger.info(f"테스트 데이터: {len(self.X_test)} 샘플")

        # 특성 스케일링
        if scale_features:
            self.scaler = StandardScaler()
            self.X_train_scaled = self.scaler.fit_transform(self.X_train)
            self.X_test_scaled = self.scaler.transform(self.X_test)
            self.logger.info("  ✓ 특성 스케일링 완료")
        else:
            self.X_train_scaled = self.X_train
            self.X_test_scaled = self.X_test

        # 데이터 통계
        self.logger.info("\n입력 특성 통계:")
        self.logger.info(f"  평균: {self.X_train.mean():.4f}")
        self.logger.info(f"  표준편차: {self.X_train.std():.4f}")

        self.logger.info("\n출력 특성 통계:")
        self.logger.info(f"  평균: {self.Y_train.mean():.4f}")
        self.logger.info(f"  표준편차: {self.Y_train.std():.4f}")

        # 출력 간 상관관계 확인
        self.logger.info("\n출력 변수 간 상관관계:")
        corr_matrix = np.corrcoef(self.Y_train.T)
        for i in range(len(self.output_features)):
            for j in range(i+1, len(self.output_features)):
                self.logger.info(f"  {self.output_features[i]} vs {self.output_features[j]}: {corr_matrix[i,j]:.3f}")

        self.logger.info("="*80)

    def train_gpr(
        self,
        kernel_type: str = 'rbf',
        length_scale: float = 1.0,
        alpha: float = 1e-6
    ):
        """Gaussian Process Regression 학습 (MultiOutputRegressor 방식)"""
        self.logger.info("="*80)
        self.logger.info("Gaussian Process Regression 학습")
        self.logger.info("="*80)

        # 커널 설정
        if kernel_type == 'rbf':
            kernel = ConstantKernel(1.0) * RBF(length_scale=length_scale)
        elif kernel_type == 'matern':
            kernel = ConstantKernel(1.0) * Matern(length_scale=length_scale, nu=1.5)
        else:
            raise ValueError(f"Unknown kernel type: {kernel_type}")

        self.logger.info(f"커널: {kernel}")

        # MultiOutputRegressor 방식
        gpr_models = []
        predictions = []

        for i, output_name in enumerate(self.output_features):
            self.logger.info(f"\n[{i+1}/{len(self.output_features)}] {output_name} 학습 중...")

            gpr = GaussianProcessRegressor(
                kernel=kernel,
                alpha=alpha,
                random_state=self.random_state,
                n_restarts_optimizer=10
            )

            gpr.fit(self.X_train_scaled, self.Y_train[:, i])
            y_pred = gpr.predict(self.X_test_scaled)
            predictions.append(y_pred)
            gpr_models.append(gpr)

            self.logger.info(f"  학습된 커널: {gpr.kernel_}")

        self.models['GPR'] = gpr_models
        self.predictions['GPR'] = np.column_stack(predictions)

        self.logger.info("\n✓ GPR 학습 완료")
        self.logger.info("="*80)

    def train_xgboost(
        self,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        method: str = 'independent'
    ):
        """
        XGBoost 학습
        method: 'independent' (독립) or 'chain' (순차)
        """
        if not XGBOOST_AVAILABLE:
            self.logger.warning("XGBoost가 설치되지 않았습니다.")
            return

        model_name = f'XGBoost_{method}'

        self.logger.info("="*80)
        self.logger.info(f"XGBoost 학습 (방법: {method})")
        self.logger.info("="*80)

        base_model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=self.random_state,
            n_jobs=-1
        )

        if method == 'independent':
            # 독립 모델 (기존 방식)
            xgb_models = []
            predictions = []

            for i, output_name in enumerate(self.output_features):
                self.logger.info(f"\n[{i+1}/{len(self.output_features)}] {output_name} 학습 중...")

                model = xgb.XGBRegressor(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    learning_rate=learning_rate,
                    random_state=self.random_state,
                    n_jobs=-1
                )

                model.fit(
                    self.X_train_scaled,
                    self.Y_train[:, i],
                    eval_set=[(self.X_test_scaled, self.Y_test[:, i])],
                    verbose=False
                )

                y_pred = model.predict(self.X_test_scaled)
                predictions.append(y_pred)
                xgb_models.append(model)

            self.models[model_name] = xgb_models
            self.predictions[model_name] = np.column_stack(predictions)

        elif method == 'chain':
            # RegressorChain 방식 (순차 예측)
            self.logger.info("RegressorChain 방식으로 학습...")
            self.logger.info("순서: " + " -> ".join(self.output_features))

            chain = RegressorChain(
                base_model,
                order=list(range(len(self.output_features))),
                random_state=self.random_state
            )

            chain.fit(self.X_train_scaled, self.Y_train)
            predictions = chain.predict(self.X_test_scaled)

            self.models[model_name] = chain
            self.predictions[model_name] = predictions

        self.logger.info(f"\n✓ {model_name} 학습 완료")
        self.logger.info("="*80)

    def train_catboost(
        self,
        iterations: int = 1000,
        depth: int = 6,
        learning_rate: float = 0.1,
        method: str = 'independent'
    ):
        """
        CatBoost 학습
        method: 'independent', 'chain', or 'multi' (MultiRMSE)
        """
        if not CATBOOST_AVAILABLE:
            self.logger.warning("CatBoost가 설치되지 않았습니다.")
            return

        model_name = f'CatBoost_{method}'

        self.logger.info("="*80)
        self.logger.info(f"CatBoost 학습 (방법: {method})")
        self.logger.info("="*80)

        if method == 'independent':
            # 독립 모델
            cat_models = []
            predictions = []

            for i, output_name in enumerate(self.output_features):
                self.logger.info(f"\n[{i+1}/{len(self.output_features)}] {output_name} 학습 중...")

                model = CatBoostRegressor(
                    iterations=iterations,
                    depth=depth,
                    learning_rate=learning_rate,
                    random_state=self.random_state,
                    verbose=False
                )

                model.fit(
                    self.X_train_scaled,
                    self.Y_train[:, i],
                    eval_set=(self.X_test_scaled, self.Y_test[:, i]),
                    verbose=False
                )

                y_pred = model.predict(self.X_test_scaled)
                predictions.append(y_pred)
                cat_models.append(model)

            self.models[model_name] = cat_models
            self.predictions[model_name] = np.column_stack(predictions)

        elif method == 'chain':
            # RegressorChain
            base_model = CatBoostRegressor(
                iterations=iterations,
                depth=depth,
                learning_rate=learning_rate,
                random_state=self.random_state,
                verbose=False
            )

            self.logger.info("RegressorChain 방식으로 학습...")

            chain = RegressorChain(
                base_model,
                order=list(range(len(self.output_features))),
                random_state=self.random_state
            )

            chain.fit(self.X_train_scaled, self.Y_train)
            predictions = chain.predict(self.X_test_scaled)

            self.models[model_name] = chain
            self.predictions[model_name] = predictions

        elif method == 'multi':
            # MultiRMSE (네이티브 다변량)
            self.logger.info("MultiRMSE 네이티브 다변량 회귀...")

            model = CatBoostRegressor(
                iterations=iterations,
                depth=depth,
                learning_rate=learning_rate,
                loss_function='MultiRMSE',
                random_state=self.random_state,
                verbose=False
            )

            model.fit(
                self.X_train_scaled,
                self.Y_train,
                eval_set=(self.X_test_scaled, self.Y_test),
                verbose=False
            )

            predictions = model.predict(self.X_test_scaled)

            self.models[model_name] = model
            self.predictions[model_name] = predictions

        self.logger.info(f"\n✓ {model_name} 학습 완료")
        self.logger.info("="*80)

    def train_random_forest(
        self,
        n_estimators: int = 100,
        max_depth: int = 10,
        method: str = 'independent'
    ):
        """Random Forest 학습"""
        model_name = f'RandomForest_{method}'

        self.logger.info("="*80)
        self.logger.info(f"Random Forest 학습 (방법: {method})")
        self.logger.info("="*80)

        if method == 'independent':
            rf_models = []
            predictions = []

            for i, output_name in enumerate(self.output_features):
                self.logger.info(f"\n[{i+1}/{len(self.output_features)}] {output_name} 학습 중...")

                model = RandomForestRegressor(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    random_state=self.random_state,
                    n_jobs=-1
                )

                model.fit(self.X_train_scaled, self.Y_train[:, i])
                y_pred = model.predict(self.X_test_scaled)
                predictions.append(y_pred)
                rf_models.append(model)

            self.models[model_name] = rf_models
            self.predictions[model_name] = np.column_stack(predictions)

        elif method == 'chain':
            base_model = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=self.random_state,
                n_jobs=-1
            )

            chain = RegressorChain(
                base_model,
                order=list(range(len(self.output_features))),
                random_state=self.random_state
            )

            chain.fit(self.X_train_scaled, self.Y_train)
            predictions = chain.predict(self.X_test_scaled)

            self.models[model_name] = chain
            self.predictions[model_name] = predictions

        self.logger.info(f"\n✓ {model_name} 학습 완료")
        self.logger.info("="*80)

    def train_mlp_sklearn(
        self,
        hidden_layer_sizes: Tuple[int, ...] = (100, 50),
        max_iter: int = 1000,
        learning_rate_init: float = 0.001
    ):
        """sklearn MLP (기존 방식)"""
        self.logger.info("="*80)
        self.logger.info("Multi-Layer Perceptron (sklearn) 학습")
        self.logger.info("="*80)

        mlp_models = []
        predictions = []

        for i, output_name in enumerate(self.output_features):
            self.logger.info(f"\n[{i+1}/{len(self.output_features)}] {output_name} 학습 중...")

            model = MLPRegressor(
                hidden_layer_sizes=hidden_layer_sizes,
                max_iter=max_iter,
                learning_rate_init=learning_rate_init,
                random_state=self.random_state,
                early_stopping=True,
                validation_fraction=0.1,
                verbose=False
            )

            model.fit(self.X_train_scaled, self.Y_train[:, i])
            y_pred = model.predict(self.X_test_scaled)
            predictions.append(y_pred)
            mlp_models.append(model)

        self.models['MLP_sklearn'] = mlp_models
        self.predictions['MLP_sklearn'] = np.column_stack(predictions)

        self.logger.info("\n✓ MLP (sklearn) 학습 완료")
        self.logger.info("="*80)

    def train_mlp_constrained(
        self,
        hidden_dims: List[int] = [128, 64],
        epochs: int = 200,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        sum_weight: float = 1.0,
        variance_weight: float = 0.1
    ):
        """물리적 제약 조건을 고려한 MLP (PyTorch)"""
        if not PYTORCH_AVAILABLE:
            self.logger.warning("PyTorch가 설치되지 않았습니다.")
            return

        self.logger.info("="*80)
        self.logger.info("Constrained MLP (PyTorch) 학습")
        self.logger.info("="*80)
        self.logger.info(f"은닉층: {hidden_dims}")
        self.logger.info(f"에포크: {epochs}")
        self.logger.info(f"합 제약 가중치: {sum_weight}")
        self.logger.info(f"분산 제약 가중치: {variance_weight}")

        # 데이터를 PyTorch Tensor로 변환
        X_train_tensor = torch.FloatTensor(self.X_train_scaled)
        Y_train_tensor = torch.FloatTensor(self.Y_train)
        X_test_tensor = torch.FloatTensor(self.X_test_scaled)
        Y_test_tensor = torch.FloatTensor(self.Y_test)

        # DataLoader 생성
        train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # 모델 생성
        input_dim = self.X_train_scaled.shape[1]
        output_dim = self.Y_train.shape[1]

        model = ConstrainedMLP(input_dim, output_dim, hidden_dims)
        criterion = ConstrainedLoss(sum_weight, variance_weight)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # 학습
        self.logger.info("\n학습 시작...")
        train_losses = []

        for epoch in range(epochs):
            model.train()
            epoch_loss = 0
            epoch_mse = 0
            epoch_sum_constraint = 0
            epoch_var_loss = 0

            for batch_X, batch_Y in train_loader:
                optimizer.zero_grad()

                predictions = model(batch_X)
                total_loss, mse_loss, sum_constraint, variance_loss = criterion(predictions, batch_Y)

                total_loss.backward()
                optimizer.step()

                epoch_loss += total_loss.item()
                epoch_mse += mse_loss.item()
                epoch_sum_constraint += sum_constraint.item()
                epoch_var_loss += variance_loss.item()

            # 평균 손실
            n_batches = len(train_loader)
            avg_loss = epoch_loss / n_batches
            avg_mse = epoch_mse / n_batches
            avg_sum = epoch_sum_constraint / n_batches
            avg_var = epoch_var_loss / n_batches

            train_losses.append(avg_loss)

            if (epoch + 1) % 20 == 0:
                self.logger.info(f"  Epoch [{epoch+1}/{epochs}] - "
                               f"Loss: {avg_loss:.6f}, MSE: {avg_mse:.6f}, "
                               f"Sum: {avg_sum:.6f}, Var: {avg_var:.6f}")

        # 예측
        model.eval()
        with torch.no_grad():
            predictions = model(X_test_tensor).numpy()

        self.models['MLP_constrained'] = model
        self.predictions['MLP_constrained'] = predictions

        # 학습 곡선 시각화
        self._plot_training_curve(train_losses, 'MLP_constrained')

        # 제약 조건 검증
        self._validate_constraints(predictions)

        self.logger.info("\n✓ Constrained MLP 학습 완료")
        self.logger.info("="*80)

    def _plot_training_curve(self, losses: List[float], model_name: str):
        """학습 곡선 시각화"""
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(losses, linewidth=2)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title(f'{model_name} 학습 곡선', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        output_file = os.path.join(self.output_dir, f'{model_name}_training_curve.png')
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()

        self.logger.info(f"  ✓ 학습 곡선 저장: {output_file}")

    def _validate_constraints(self, predictions: np.ndarray):
        """물리적 제약 조건 검증"""
        self.logger.info("\n제약 조건 검증:")

        # 합 제약 검증
        pred_sums = predictions.sum(axis=1)
        mean_sum = np.mean(np.abs(pred_sums))
        max_sum = np.max(np.abs(pred_sums))

        self.logger.info(f"  예측값 합의 절대값:")
        self.logger.info(f"    평균: {mean_sum:.6f}")
        self.logger.info(f"    최대: {max_sum:.6f}")

        # 분산 비교
        target_var = np.var(self.Y_test, axis=1).mean()
        pred_var = np.var(predictions, axis=1).mean()

        self.logger.info(f"  분산 비교:")
        self.logger.info(f"    타겟 분산: {target_var:.6f}")
        self.logger.info(f"    예측 분산: {pred_var:.6f}")
        self.logger.info(f"    차이: {abs(target_var - pred_var):.6f}")

    def evaluate_models(self):
        """모든 모델 평가"""
        self.logger.info("="*80)
        self.logger.info("모델 성능 평가")
        self.logger.info("="*80)

        for model_name in self.models.keys():
            self.logger.info(f"\n{'='*80}")
            self.logger.info(f"{model_name} 평가")
            self.logger.info(f"{'='*80}")

            y_pred = self.predictions[model_name]

            # 전체 메트릭
            rmse = np.sqrt(mean_squared_error(self.Y_test, y_pred))
            mae = mean_absolute_error(self.Y_test, y_pred)
            r2 = r2_score(self.Y_test, y_pred)

            self.logger.info(f"\n전체 성능:")
            self.logger.info(f"  RMSE: {rmse:.6f}")
            self.logger.info(f"  MAE:  {mae:.6f}")
            self.logger.info(f"  R²:   {r2:.6f}")

            # 출력 차원별 메트릭
            self.logger.info(f"\n출력 차원별 성능:")
            dimension_metrics = []

            for i, output_name in enumerate(self.output_features):
                rmse_i = np.sqrt(mean_squared_error(self.Y_test[:, i], y_pred[:, i]))
                mae_i = mean_absolute_error(self.Y_test[:, i], y_pred[:, i])
                r2_i = r2_score(self.Y_test[:, i], y_pred[:, i])

                self.logger.info(f"  {output_name}:")
                self.logger.info(f"    RMSE: {rmse_i:.6f}")
                self.logger.info(f"    MAE:  {mae_i:.6f}")
                self.logger.info(f"    R²:   {r2_i:.6f}")

                dimension_metrics.append({
                    'output': output_name,
                    'rmse': rmse_i,
                    'mae': mae_i,
                    'r2': r2_i
                })

            # 물리적 제약 검증
            self.logger.info(f"\n물리적 제약 검증:")
            pred_sums = y_pred.sum(axis=1)
            self.logger.info(f"  예측값 합의 평균: {pred_sums.mean():.6f}")
            self.logger.info(f"  예측값 합의 표준편차: {pred_sums.std():.6f}")

            # 저장
            self.metrics[model_name] = {
                'overall': {
                    'rmse': rmse,
                    'mae': mae,
                    'r2': r2,
                    'sum_mean': pred_sums.mean(),
                    'sum_std': pred_sums.std()
                },
                'by_dimension': dimension_metrics
            }

        self.logger.info("\n" + "="*80)

    def compare_models(self):
        """모델 성능 비교"""
        self.logger.info("="*80)
        self.logger.info("모델 성능 비교")
        self.logger.info("="*80)

        comparison = []

        for model_name in self.models.keys():
            metrics = self.metrics[model_name]['overall']
            comparison.append({
                '모델': model_name,
                'RMSE': metrics['rmse'],
                'MAE': metrics['mae'],
                'R²': metrics['r2'],
                '합_평균': metrics['sum_mean'],
                '합_표준편차': metrics['sum_std']
            })

        # DataFrame으로 변환
        df_comparison = pd.DataFrame(comparison)
        df_comparison = df_comparison.sort_values('RMSE')

        self.logger.info("\n" + df_comparison.to_string(index=False))

        # 저장
        output_file = os.path.join(self.output_dir, 'model_comparison.csv')
        df_comparison.to_csv(output_file, index=False, encoding='utf-8-sig')
        self.logger.info(f"\n✓ 비교 결과 저장: {output_file}")

        self.logger.info("="*80)

        return df_comparison

    def plot_predictions(self):
        """예측 결과 시각화"""
        self.logger.info("="*80)
        self.logger.info("예측 결과 시각화")
        self.logger.info("="*80)

        n_models = len(self.models)
        n_outputs = len(self.output_features)

        fig, axes = plt.subplots(n_outputs, n_models, figsize=(5*n_models, 4*n_outputs))

        if n_outputs == 1:
            axes = axes.reshape(1, -1)
        if n_models == 1:
            axes = axes.reshape(-1, 1)

        for j, model_name in enumerate(self.models.keys()):
            y_pred = self.predictions[model_name]

            for i, output_name in enumerate(self.output_features):
                ax = axes[i, j]

                # Scatter plot
                ax.scatter(
                    self.Y_test[:, i],
                    y_pred[:, i],
                    alpha=0.5,
                    s=30,
                    edgecolors='k',
                    linewidths=0.5
                )

                # 대각선
                min_val = min(self.Y_test[:, i].min(), y_pred[:, i].min())
                max_val = max(self.Y_test[:, i].max(), y_pred[:, i].max())
                ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)

                # 메트릭
                r2 = r2_score(self.Y_test[:, i], y_pred[:, i])
                rmse = np.sqrt(mean_squared_error(self.Y_test[:, i], y_pred[:, i]))

                ax.set_xlabel('실제 값', fontsize=10)
                ax.set_ylabel('예측 값', fontsize=10)
                ax.set_title(f'{model_name} - {output_name}\nR²={r2:.3f}, RMSE={rmse:.4f}',
                           fontsize=11, fontweight='bold')
                ax.grid(True, alpha=0.3)

        plt.tight_layout()

        # 저장
        output_file = os.path.join(self.output_dir, 'predictions_scatter.png')
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()

        self.logger.info(f"✓ 시각화 저장: {output_file}")
        self.logger.info("="*80)

    def plot_sum_distribution(self):
        """예측값 합 분포 비교"""
        self.logger.info("="*80)
        self.logger.info("예측값 합 분포 시각화")
        self.logger.info("="*80)

        fig, axes = plt.subplots(2, 1, figsize=(12, 10))

        # 실제값 합
        target_sums = self.Y_test.sum(axis=1)

        # 박스플롯
        ax1 = axes[0]
        data_to_plot = [target_sums]
        labels = ['Target']

        for model_name in self.models.keys():
            pred_sums = self.predictions[model_name].sum(axis=1)
            data_to_plot.append(pred_sums)
            labels.append(model_name)

        bp = ax1.boxplot(data_to_plot, labels=labels, patch_artist=True)
        ax1.axhline(y=0, color='r', linestyle='--', linewidth=2, label='이상적 합 (0)')
        ax1.set_ylabel('예측값 합', fontsize=12)
        ax1.set_title('모델별 예측값 합 분포 비교 (BoxPlot)', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')
        plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=45, ha='right')

        # 히스토그램
        ax2 = axes[1]
        ax2.hist(target_sums, bins=30, alpha=0.5, label='Target', edgecolor='black')

        for model_name in self.models.keys():
            pred_sums = self.predictions[model_name].sum(axis=1)
            ax2.hist(pred_sums, bins=30, alpha=0.5, label=model_name, edgecolor='black')

        ax2.axvline(x=0, color='r', linestyle='--', linewidth=2, label='이상적 합 (0)')
        ax2.set_xlabel('예측값 합', fontsize=12)
        ax2.set_ylabel('빈도', fontsize=12)
        ax2.set_title('모델별 예측값 합 분포 비교 (Histogram)', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        output_file = os.path.join(self.output_dir, 'sum_distribution.png')
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()

        self.logger.info(f"✓ 시각화 저장: {output_file}")
        self.logger.info("="*80)

    def save_predictions(self):
        """예측 결과 저장"""
        self.logger.info("="*80)
        self.logger.info("예측 결과 저장")
        self.logger.info("="*80)

        for model_name in self.models.keys():
            results = pd.DataFrame()

            for i, output_name in enumerate(self.output_features):
                results[f'{output_name}_actual'] = self.Y_test[:, i]
                results[f'{output_name}_pred'] = self.predictions[model_name][:, i]
                results[f'{output_name}_residual'] = (
                    self.Y_test[:, i] - self.predictions[model_name][:, i]
                )

            # 합 계산
            results['sum_actual'] = self.Y_test.sum(axis=1)
            results['sum_pred'] = self.predictions[model_name].sum(axis=1)

            output_file = os.path.join(self.output_dir, f'{model_name}_predictions.csv')
            results.to_csv(output_file, index=False, encoding='utf-8-sig')
            self.logger.info(f"✓ {model_name} 예측 결과 저장: {output_file}")

        self.logger.info("="*80)

    def generate_report(self):
        """종합 리포트 생성"""
        self.logger.info("="*80)
        self.logger.info("종합 리포트 생성")
        self.logger.info("="*80)

        report_lines = []
        report_lines.append("="*80)
        report_lines.append("모델 학습 및 평가 종합 리포트 (다변량 예측)")
        report_lines.append("="*80)
        report_lines.append(f"생성 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")

        # 데이터 정보
        report_lines.append("1. 데이터 정보")
        report_lines.append("-"*80)
        report_lines.append(f"  데이터 파일: {self.data_file}")
        report_lines.append(f"  전체 샘플: {len(self.data)}")
        report_lines.append(f"  학습 샘플: {len(self.X_train)}")
        report_lines.append(f"  테스트 샘플: {len(self.X_test)}")
        report_lines.append(f"  입력 특성: {len(self.input_features)}")
        report_lines.append(f"  출력 특성: {len(self.output_features)}")
        report_lines.append("")

        # 모델 성능
        report_lines.append("2. 모델 성능 비교")
        report_lines.append("-"*80)

        for model_name in self.models.keys():
            metrics = self.metrics[model_name]['overall']
            report_lines.append(f"\n  {model_name}:")
            report_lines.append(f"    RMSE: {metrics['rmse']:.6f}")
            report_lines.append(f"    MAE:  {metrics['mae']:.6f}")
            report_lines.append(f"    R²:   {metrics['r2']:.6f}")
            report_lines.append(f"    합 평균: {metrics['sum_mean']:.6f}")
            report_lines.append(f"    합 표준편차: {metrics['sum_std']:.6f}")

        report_lines.append("")

        # 최고 성능 모델
        best_model = min(self.metrics.items(),
                        key=lambda x: x[1]['overall']['rmse'])
        report_lines.append("3. 최고 성능 모델 (RMSE 기준)")
        report_lines.append("-"*80)
        report_lines.append(f"  모델: {best_model[0]}")
        report_lines.append(f"  RMSE: {best_model[1]['overall']['rmse']:.6f}")
        report_lines.append(f"  R²: {best_model[1]['overall']['r2']:.6f}")
        report_lines.append("")

        # 제약 조건 준수도
        best_constrained = min(self.metrics.items(),
                              key=lambda x: abs(x[1]['overall']['sum_mean']))
        report_lines.append("4. 물리적 제약 준수도 (합 제약)")
        report_lines.append("-"*80)
        report_lines.append(f"  최우수 모델: {best_constrained[0]}")
        report_lines.append(f"  합 평균: {best_constrained[1]['overall']['sum_mean']:.6f}")
        report_lines.append(f"  합 표준편차: {best_constrained[1]['overall']['sum_std']:.6f}")
        report_lines.append("")

        # 저장
        report_text = "\n".join(report_lines)
        output_file = os.path.join(self.output_dir, 'training_report.txt')

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report_text)

        self.logger.info(f"✓ 리포트 저장: {output_file}")
        self.logger.info("="*80)

        return report_text

    def save_models(
        self,
        save_dir: str = None,
        model_names: list = None
    ) -> list:
        """
        학습된 모델을 pickle 형식으로 저장

        CatBoostModelManager._load_model()이 기대하는 dict 포맷으로 저장:
            {
                'model':           학습된 모델 객체,
                'scaler':          StandardScaler (입력 스케일링),
                'input_features':  입력 특성 이름 리스트,
                'output_features': 출력 특성 이름 리스트,
                'metadata':        학습 메타데이터 dict,
            }

        저장 경로: <save_dir>/<model_name>.pkl
        파일명이 *CatBoost* 패턴을 포함해야 model_interface.py가 자동 탐색 가능.

        Parameters
        ----------
        save_dir : str, optional
            모델 저장 디렉토리. None이면 self.output_dir 사용.
        model_names : list, optional
            저장할 모델 이름 리스트. None이면 self.models 전체 저장.

        Returns
        -------
        list
            저장된 파일 경로 리스트
        """
        self.logger.info("="*80)
        self.logger.info("학습된 모델 파라미터 저장")
        self.logger.info("="*80)

        if save_dir is None:
            save_dir = self.output_dir

        Path(save_dir).mkdir(parents=True, exist_ok=True)

        if model_names is None:
            model_names = list(self.models.keys())

        saved_files = []

        for name in model_names:
            if name not in self.models:
                self.logger.warning(f"모델 없음, 건너뜀: {name}")
                continue

            # CatBoostModelManager가 기대하는 포맷으로 패키징
            model_data = {
                'model': self.models[name],
                'scaler': self.scaler,
                'input_features': self.input_features,
                'output_features': self.output_features,
                'metadata': {
                    'model_name': name,
                    'trained_at': datetime.now().isoformat(),
                    'data_file': self.data_file,
                    'n_train_samples': len(self.X_train),
                    'n_test_samples': len(self.X_test),
                    'n_input_features': len(self.input_features),
                    'n_output_features': len(self.output_features),
                    'random_state': self.random_state,
                    'metrics': self.metrics.get(name, {}),
                },
            }

            save_path = Path(save_dir) / f'{name}.pkl'
            with open(save_path, 'wb') as f:
                pickle.dump(model_data, f)

            saved_files.append(str(save_path))
            self.logger.info(f"  ✓ 저장 완료: {save_path}")

        self.logger.info(f"\n총 {len(saved_files)}개 모델 저장됨 → {save_dir}")
        self.logger.info("="*80)

        return saved_files
