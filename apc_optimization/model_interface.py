"""
모델 인터페이스 (Model Interface)

CatBoost 학습된 모델을 로드하고, 배치 예측을 수행
추가적으로 Inverse CLR 변환을 통해 확률 분포로 변환
"""

import numpy as np
import pandas as pd
import pickle
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import sys

from .config import (
    PROJECT_ROOT, MODEL_DIR, CLR_PARAMS, N_ZONES, N_GV,
    INPUT_FEATURE_PATTERNS, OUTPUT_FEATURE_PATTERNS,
    CLR_COMPONENTS, PROBABILITY_COMPONENTS, OUTPUT_TRANSFORM_CONFIG
)
from .output_transformer import OutputTransformer, TransformConfig

logger = logging.getLogger(__name__)


class CatBoostModelManager:
    """
    CatBoost 학습된 모델 관리자

    - 모델 로드
    - 배치 예측 (벡터화)
    - Inverse CLR 변환 (CLR → 확률)
    - 특성 중요도 조회
    """

    def __init__(self, model_path: Optional[str] = None,
                 enable_output_transform: bool = False):
        """
        초기화 및 모델 로드

        Args:
            model_path: 모델 파일 경로. 없으면 자동으로 MODEL_DIR에서 찾음
            enable_output_transform: 출력 변환 활성화 여부
        """
        self.model = None
        self.scaler = None
        self.input_features = None
        self.output_features = None
        self.feature_importance = None
        self.model_metadata = {}

        # 출력 변환기 초기화
        transform_config = TransformConfig(**OUTPUT_TRANSFORM_CONFIG)
        if enable_output_transform:
            transform_config.enable = True
        self.output_transformer = OutputTransformer(transform_config)

        self._load_model(model_path)

    def _load_model(self, model_path: Optional[str] = None):
        """
        학습된 모델 로드

        Args:
            model_path: 모델 경로
        """
        if model_path is None:
            # 자동으로 MODEL_DIR에서 CatBoost 모델 찾기
            model_files = list(MODEL_DIR.glob('*CatBoost*'))
            if not model_files:
                logger.warning(f"모델을 찾을 수 없습니다: {MODEL_DIR}")
                logger.info("모델이 없을 경우 MockCatBoostModel을 사용합니다.")
                self.model = MockCatBoostModel()
                return

            model_path = str(model_files[0])
            logger.info(f"발견된 모델: {model_path}")

        model_path = Path(model_path)

        try:
            # 모델 로드 (pickle 형식)
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)

            # 모델 객체 또는 dict 형식 처리
            if isinstance(model_data, dict):
                self.model = model_data.get('model')
                self.scaler = model_data.get('scaler')
                self.input_features = model_data.get('input_features')
                self.output_features = model_data.get('output_features')
                self.model_metadata = model_data.get('metadata', {})
            else:
                self.model = model_data

            logger.info(f"✓ 모델 로드 완료: {model_path}")

            # 모델 정보 출력
            self._print_model_info()

        except Exception as e:
            logger.error(f"모델 로드 실패: {e}")
            logger.warning("MockCatBoostModel을 사용합니다.")
            self.model = MockCatBoostModel()

    def _print_model_info(self):
        """
        로드된 모델의 정보 출력
        """
        info = f"""
        ============================================================
        모델 정보
        ============================================================
        모델 타입: {type(self.model).__name__}
        입력 특성 수: {len(self.input_features) if self.input_features else 'Unknown'}
        출력 특성 수: {len(self.output_features) if self.output_features else 'Unknown'}
        스케일러: {'사용' if self.scaler else '미사용'}
        메타데이터: {self.model_metadata}
        ============================================================
        """
        logger.info(info)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        단일 샘플 예측

        Args:
            X: Shape (n_features,) 또는 (1, n_features)

        Returns:
            Shape (n_outputs,) 또는 (1, n_outputs)
        """
        X = np.atleast_2d(X)

        # 스케일링
        if self.scaler is not None:
            X = self.scaler.transform(X)

        # 예측
        if self.model is None:
            raise RuntimeError("모델이 로드되지 않았습니다.")

        prediction = self.model.predict(X)

        return prediction[0] if prediction.shape[0] == 1 else prediction

    def predict_batch(self, X: np.ndarray, apply_transform: bool = True) -> np.ndarray:
        """
        배치 예측 (벡터화된 예측)

        Args:
            X: Shape (n_samples, n_features)
            apply_transform: 출력 변환 적용 여부

        Returns:
            Shape (n_samples, n_outputs)
        """
        if X.ndim == 1:
            X = X.reshape(1, -1)

        # 스케일링
        if self.scaler is not None:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X

        # 배치 예측
        if self.model is None:
            raise RuntimeError("모델이 로드되지 않았습니다.")

        predictions = self.model.predict(X_scaled)

        # 출력 변환 (필요시)
        if apply_transform and self.output_transformer.config.enable:
            predictions = self._apply_output_transform(predictions)

        return predictions

    def _apply_output_transform(self, predictions: np.ndarray) -> np.ndarray:
        """
        모델 출력 변환

        Args:
            predictions: Shape (n_samples, n_outputs)

        Returns:
            변환된 예측값
        """
        # Note: 예측값은 △CLR이므로 일반적으로 변환하지 않음
        # 필요시 여기에 변환 로직 추가 가능
        return predictions

    def predict_with_uncertainty(self, X: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        불확실성을 포함한 예측 (CatBoost는 확률 모델이 아니므로 기본 예측만 반환)

        Args:
            X: Shape (n_samples, n_features)

        Returns:
            (predictions, std) - std는 None (CatBoost는 자체 불확실성 추정 불가)
        """
        predictions = self.predict_batch(X)
        return predictions, None

    # ========================================================================
    # Inverse CLR 변환
    # ========================================================================

    @staticmethod
    def inverse_clr(clr_values: np.ndarray) -> np.ndarray:
        """
        Center-Log-Ratio (CLR) 역변환

        CLR 정의:
            CLR_i = log(x_i / GM)  where GM = (x_1 * x_2 * ... * x_D)^(1/D)

        역변환:
            x_i = exp(CLR_i) / sum(exp(CLR_j)) for all j

        Args:
            clr_values: Shape (..., n_components) - CLR 값

        Returns:
            Shape (..., n_components) - 확률 분포 (합=1)
        """
        # exp 연산
        exp_clr = np.exp(clr_values)

        # 정규화 (합=1)
        # keepdims=True를 사용하여 브로드캐스팅 가능하게 유지
        sum_exp = np.sum(exp_clr, axis=-1, keepdims=True)

        # 0으로 나누기 방지
        sum_exp = np.where(sum_exp == 0, 1.0, sum_exp)

        probabilities = exp_clr / sum_exp

        return probabilities

    def apply_inverse_clr_transform(self,
                                    current_clr: np.ndarray,
                                    predicted_delta_clr: np.ndarray) -> np.ndarray:
        """
        현재 CLR + 예측된 △CLR → 새로운 확률 분포

        Args:
            current_clr: Shape (n_zones, 3) - 현재 CLR 값 (CLR_1, CLR_2, CLR_3)
            predicted_delta_clr: Shape (n_zones, 3) - 예측된 △CLR (모델 출력)

        Returns:
            Shape (n_zones, 3) - 새로운 확률 분포 (P_Low, P_Mid, P_High)
        """
        # 새로운 CLR = 현재 CLR + △CLR
        new_clr = current_clr + predicted_delta_clr

        # Inverse CLR 변환
        probabilities = self.inverse_clr(new_clr)

        # 안정성 체크: 합이 1에 가까운지 확인
        prob_sum = np.sum(probabilities, axis=-1)
        if not np.allclose(prob_sum, 1.0, atol=1e-5):
            logger.warning(f"확률 합이 1이 아닙니다: {prob_sum}")

        return probabilities

    # ========================================================================
    # 특성 중요도 조회
    # ========================================================================

    def get_feature_importance(self) -> Dict[str, float]:
        """
        특성 중요도 조회

        Returns:
            dict: 특성명 → 중요도 (%)
        """
        if not hasattr(self.model, 'get_feature_importance'):
            logger.warning("모델이 특성 중요도를 지원하지 않습니다.")
            return {}

        try:
            importances = self.model.get_feature_importance()
            feature_names = self.input_features or [f"feature_{i}" for i in range(len(importances))]

            importance_dict = {
                name: importance
                for name, importance in zip(feature_names, importances)
            }

            return importance_dict

        except Exception as e:
            logger.error(f"특성 중요도 조회 실패: {e}")
            return {}

    def print_feature_importance(self, top_n: int = 10):
        """
        특성 중요도 상위 N개 출력
        """
        importances = self.get_feature_importance()
        if not importances:
            logger.info("사용 가능한 특성 중요도 없음")
            return

        sorted_importances = sorted(importances.items(), key=lambda x: x[1], reverse=True)

        print(f"\n{'특성명':<30} {'중요도':<10} {'순위'}")
        print("="*50)
        for rank, (feature, importance) in enumerate(sorted_importances[:top_n], 1):
            print(f"{feature:<30} {importance:>8.2f}% {rank:>3}")

    # ========================================================================
    # 유틸리티
    # ========================================================================

    def get_model_summary(self) -> str:
        """
        모델 요약 정보 반환
        """
        summary = f"""
        ============================================================
        CatBoost 모델 요약
        ============================================================
        모델 타입: {type(self.model).__name__}
        입력 특성: {len(self.input_features) if self.input_features else 'Unknown'}개
        출력 특성: {len(self.output_features) if self.output_features else 'Unknown'}개
        스케일러: {'사용' if self.scaler else '미사용'}
        ============================================================
        """
        return summary


# ============================================================================
# Mock CatBoost Model (테스트용)
# ============================================================================

class MockCatBoostModel:
    """
    테스트 및 디버깅용 Mock CatBoost 모델

    무작위로 예측값을 반환 (실제 학습된 모델이 없을 때)
    """

    def __init__(self):
        self.n_features = 33  # 예상 입력 특성 수
        self.n_outputs = 3    # 출력: CLR_1, CLR_2, CLR_3

        logger.warning("Mock CatBoost Model 사용 중 (학습된 모델 없음)")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        무작위 예측 (정상 분포 샘플링)
        """
        n_samples = X.shape[0] if X.ndim > 1 else 1
        predictions = np.random.normal(0, 0.1, size=(n_samples, self.n_outputs))
        return predictions

    def get_feature_importance(self) -> np.ndarray:
        """
        무작위 중요도
        """
        return np.random.uniform(0, 100, self.n_features)


# ============================================================================
# 테스트 코드
# ============================================================================

if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # 모델 로드
    manager = CatBoostModelManager()

    print(manager.get_model_summary())

    # 배치 예측 테스트
    X_test = np.random.randn(5, 33)  # 임시 테스트 데이터
    predictions = manager.predict_batch(X_test)
    print(f"\n배치 예측 결과 shape: {predictions.shape}")
    print(f"예측값 샘플:\n{predictions[:2]}")

    # Inverse CLR 테스트
    current_clr = np.random.randn(N_ZONES, 3)
    predicted_delta_clr = np.random.randn(N_ZONES, 3) * 0.1
    probabilities = manager.apply_inverse_clr_transform(current_clr, predicted_delta_clr)
    print(f"\nInverse CLR 결과 shape: {probabilities.shape}")
    print(f"확률 분포 샘플 (합):\n{np.sum(probabilities, axis=-1)[:3]}")
