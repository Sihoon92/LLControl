"""
Data Processor for PETS

기존 CatBoost 학습 데이터 (model_training_data.xlsx)를
Per-Zone PETS 모델 학습용 형태로 변환

데이터 구조:
- Input: [위치(4), current_CLR(3), 제어(4)] = 11개
- Output: diff_CLR (3개)
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, List, Optional
from pathlib import Path
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from ..config import N_ZONES, N_GV
from utils import load_file
from feature_extractor import extract_features as extract_canonical_features

logger = logging.getLogger(__name__)


class PETSDataProcessor:
    """
    PETS 데이터 처리기

    CatBoost 학습 데이터를 Per-Zone PETS 형태로 변환
    """

    def __init__(
        self,
        normalize: bool = True,
        validation_split: float = 0.2,
        test_split: float = 0.1,
        random_seed: int = 42
    ):
        """
        Args:
            normalize: 정규화 여부
            validation_split: 검증 데이터 비율
            test_split: 테스트 데이터 비율
            random_seed: 랜덤 시드
        """
        self.normalize = normalize
        self.validation_split = validation_split
        self.test_split = test_split
        self.random_seed = random_seed

        # Scaler (정규화용)
        self.input_scaler = StandardScaler() if normalize else None
        self.output_scaler = StandardScaler() if normalize else None

        self.is_fitted = False

    def load_data(
        self,
        data_file: str,
        mode: str = 'training'
    ) -> pd.DataFrame:
        """
        데이터 로드

        Args:
            data_file: 데이터 파일 경로
            mode: 'training' 또는 'test'

        Returns:
            DataFrame
        """
        logger.info(f"데이터 로드: {data_file} (mode={mode})")

        df = load_file(data_file, logger=logger)

        logger.info(f"  로드된 데이터: {len(df)} 행, {len(df.columns)} 열")

        return df

    def extract_features(
        self,
        df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        데이터프레임에서 입력/출력 특성 추출

        feature_extractor.py 공통 정의를 사용하여
        ModelTrainer와 동일한 피처(11개)를 추출한다.

        Args:
            df: 학습 데이터 DataFrame

        Returns:
            X: (N, 11) - 입력
            y: (N, 3)  - 출력
        """
        logger.info("특성 추출 중... [feature_extractor 공통 정의 사용]")

        X, y = extract_canonical_features(df)

        logger.info(f"  추출 완료: X shape={X.shape}, y shape={y.shape}")

        if np.isnan(X).any():
            logger.warning(f"  입력에 NaN 발견: {np.isnan(X).sum()}개")
        if np.isnan(y).any():
            logger.warning(f"  출력에 NaN 발견: {np.isnan(y).sum()}개")

        return X, y

    def split_data(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        데이터 분할 (train/val/test)

        Args:
            X: (N, 11) - 입력
            y: (N, 3) - 출력

        Returns:
            dict: {
                'train': (X_train, y_train),
                'val': (X_val, y_val),
                'test': (X_test, y_test),
            }
        """
        logger.info("데이터 분할 중...")

        # Train + Val / Test
        test_size = self.test_split
        X_trainval, X_test, y_trainval, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_seed
        )

        # Train / Val
        val_size = self.validation_split / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_trainval, y_trainval, test_size=val_size, random_state=self.random_seed
        )

        logger.info(f"  Train: {len(X_train)} 샘플")
        logger.info(f"  Val:   {len(X_val)} 샘플")
        logger.info(f"  Test:  {len(X_test)} 샘플")

        return {
            'train': (X_train, y_train),
            'val': (X_val, y_val),
            'test': (X_test, y_test),
        }

    def normalize_data(
        self,
        data_dict: Dict[str, Tuple[np.ndarray, np.ndarray]],
        fit_on_train: bool = True
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        데이터 정규화 (StandardScaler)

        Args:
            data_dict: {'train': (X, y), 'val': (X, y), 'test': (X, y)}
            fit_on_train: Train 데이터로 Scaler 학습 여부

        Returns:
            정규화된 data_dict
        """
        if not self.normalize:
            logger.info("정규화 스킵")
            return data_dict

        logger.info("데이터 정규화 중...")

        X_train, y_train = data_dict['train']

        if fit_on_train:
            # Train 데이터로 Scaler 학습
            self.input_scaler.fit(X_train)
            self.output_scaler.fit(y_train)
            self.is_fitted = True
            logger.info("  Scaler 학습 완료")

        # 정규화 적용
        normalized_dict = {}

        for split_name, (X, y) in data_dict.items():
            X_norm = self.input_scaler.transform(X)
            y_norm = self.output_scaler.transform(y)

            normalized_dict[split_name] = (X_norm, y_norm)

            logger.info(f"  {split_name}: X mean={X_norm.mean():.4f}, std={X_norm.std():.4f}")

        return normalized_dict

    def inverse_transform_output(
        self,
        y_normalized: np.ndarray
    ) -> np.ndarray:
        """
        정규화된 출력을 원본 스케일로 변환

        Args:
            y_normalized: (N, 3) - 정규화된 diff_CLR

        Returns:
            y_original: (N, 3) - 원본 스케일 diff_CLR
        """
        if not self.normalize or not self.is_fitted:
            return y_normalized

        return self.output_scaler.inverse_transform(y_normalized)

    def process(
        self,
        data_file: str,
        mode: str = 'training'
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        전체 데이터 처리 파이프라인

        Args:
            data_file: 데이터 파일 경로
            mode: 'training' 또는 'test'

        Returns:
            dict: {'train': (X, y), 'val': (X, y), 'test': (X, y)}
        """
        logger.info("="*80)
        logger.info("PETS 데이터 처리 시작")
        logger.info("="*80)

        # 1. 데이터 로드
        df = self.load_data(data_file, mode)

        # 2. 특성 추출
        X, y = self.extract_features(df)

        # 3. 데이터 분할
        data_dict = self.split_data(X, y)

        # 4. 정규화
        data_dict = self.normalize_data(data_dict, fit_on_train=True)

        logger.info("="*80)
        logger.info("데이터 처리 완료")
        logger.info("="*80)

        return data_dict

    def save_scaler(self, path: str):
        """Scaler 저장"""
        import joblib
        if self.is_fitted:
            joblib.dump({
                'input_scaler': self.input_scaler,
                'output_scaler': self.output_scaler
            }, path)
            logger.info(f"Scaler 저장: {path}")

    def load_scaler(self, path: str):
        """Scaler 로드"""
        import joblib
        state = joblib.load(path)
        self.input_scaler = state['input_scaler']
        self.output_scaler = state['output_scaler']
        self.is_fitted = True
        logger.info(f"Scaler 로드: {path}")


# ============================================================================
# 테스트 코드
# ============================================================================

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    print("="*80)
    print("PETS Data Processor 테스트")
    print("="*80)

    # 더미 데이터 생성
    n_samples = 1000

    data = {
        'zone_id': np.random.randint(1, 12, n_samples),
        'current_CLR_1': np.random.randn(n_samples),
        'current_CLR_2': np.random.randn(n_samples),
        'current_CLR_3': np.random.randn(n_samples),
        'delta_GV_left1': np.random.randn(n_samples) * 0.5,
        'delta_GV_self': np.random.randn(n_samples) * 0.5,
        'delta_GV_right1': np.random.randn(n_samples) * 0.5,
        'delta_RPM': np.random.randn(n_samples) * 10,
        'diff_CLR_1': np.random.randn(n_samples) * 0.1,
        'diff_CLR_2': np.random.randn(n_samples) * 0.1,
        'diff_CLR_3': np.random.randn(n_samples) * 0.1,
    }

    df = pd.DataFrame(data)

    # 임시 저장
    temp_file = '/tmp/test_data.csv'
    df.to_csv(temp_file, index=False)

    # 데이터 처리기 생성
    processor = PETSDataProcessor(
        normalize=True,
        validation_split=0.2,
        test_split=0.1,
        random_seed=42
    )

    # 처리 실행
    data_dict = processor.process(temp_file, mode='training')

    # 결과 확인
    print(f"\nTrain 데이터:")
    X_train, y_train = data_dict['train']
    print(f"  X_train shape: {X_train.shape}")
    print(f"  y_train shape: {y_train.shape}")
    print(f"  X_train mean: {X_train.mean():.4f}, std: {X_train.std():.4f}")

    print(f"\nVal 데이터:")
    X_val, y_val = data_dict['val']
    print(f"  X_val shape: {X_val.shape}")
    print(f"  y_val shape: {y_val.shape}")

    print(f"\nTest 데이터:")
    X_test, y_test = data_dict['test']
    print(f"  X_test shape: {X_test.shape}")
    print(f"  y_test shape: {y_test.shape}")

    # 역변환 테스트
    y_pred_norm = np.random.randn(10, 3)
    y_pred_original = processor.inverse_transform_output(y_pred_norm)
    print(f"\n역변환 테스트:")
    print(f"  정규화: {y_pred_norm[0]}")
    print(f"  원본: {y_pred_original[0]}")

    # Cleanup
    import os
    os.remove(temp_file)

    print("\n✓ 모든 테스트 통과!")
