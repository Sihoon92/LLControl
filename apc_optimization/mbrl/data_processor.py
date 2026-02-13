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

        if data_file.endswith('.xlsx'):
            df = pd.read_excel(data_file)
        elif data_file.endswith('.csv'):
            df = pd.read_csv(data_file)
        else:
            raise ValueError(f"Unsupported file format: {data_file}")

        logger.info(f"  로드된 데이터: {len(df)} 행, {len(df.columns)} 열")

        return df

    def extract_features(
        self,
        df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        데이터프레임에서 입력/출력 특성 추출

        Args:
            df: 학습 데이터 DataFrame

        Returns:
            X: (N, 11) - 입력
            y: (N, 3) - 출력
        """
        logger.info("특성 추출 중...")

        required_cols = [
            # 위치 특성 (4개) - 계산으로 생성
            # 'distance', 'edge_distance', 'normalized_position', 'normalized_distance'

            # Zone ID
            'zone_id',

            # 현재 CLR (3개)
            'current_CLR_1', 'current_CLR_2', 'current_CLR_3',

            # 제어 변화 (인접 포함)
            'delta_GV_self',
            'delta_GV_left1', 'delta_GV_right1',
            'delta_RPM',

            # 출력 (diff_CLR)
            'diff_CLR_1', 'diff_CLR_2', 'diff_CLR_3',
        ]

        # 필수 칼럼 확인
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            logger.warning(f"누락된 칼럼: {missing_cols}")

            # 대체 칼럼 확인
            if 'delta_GV_left1' not in df.columns and 'delta_GV_left' in df.columns:
                df['delta_GV_left1'] = df['delta_GV_left']
                logger.info("  delta_GV_left1 <- delta_GV_left")

            if 'delta_GV_right1' not in df.columns and 'delta_GV_right' in df.columns:
                df['delta_GV_right1'] = df['delta_GV_right']
                logger.info("  delta_GV_right1 <- delta_GV_right")

        # 입력 특성 구성
        X_list = []
        y_list = []

        for idx, row in df.iterrows():
            zone_id = int(row['zone_id']) - 1  # 0-based

            # 1. 위치 특성 (4개)
            position_features = self._compute_position_features(zone_id)

            # 2. 현재 CLR (3개)
            current_clr = np.array([
                row['current_CLR_1'],
                row['current_CLR_2'],
                row['current_CLR_3']
            ], dtype=np.float32)

            # 3. 제어 특성 (4개)
            control_features = np.array([
                row.get('delta_GV_left1', 0.0),
                row['delta_GV_self'],
                row.get('delta_GV_right1', 0.0),
                row['delta_RPM']
            ], dtype=np.float32)

            # 입력 벡터 결합
            x = np.concatenate([position_features, current_clr, control_features])

            # 출력 (diff_CLR)
            y = np.array([
                row['diff_CLR_1'],
                row['diff_CLR_2'],
                row['diff_CLR_3']
            ], dtype=np.float32)

            X_list.append(x)
            y_list.append(y)

        X = np.array(X_list)  # (N, 11)
        y = np.array(y_list)  # (N, 3)

        logger.info(f"  추출 완료: X shape={X.shape}, y shape={y.shape}")

        # NaN 체크
        if np.isnan(X).any():
            logger.warning(f"  입력에 NaN 발견: {np.isnan(X).sum()}개")
        if np.isnan(y).any():
            logger.warning(f"  출력에 NaN 발견: {np.isnan(y).sum()}개")

        return X, y

    def _compute_position_features(self, zone_i: int) -> np.ndarray:
        """
        위치 특성 계산 (4개)

        dynamics_model.py의 _get_position_features와 동일
        """
        center = N_ZONES / 2
        distance = zone_i * 100  # 100mm 간격
        edge_distance = abs(zone_i - (center - 0.5))
        normalized_position = zone_i / (N_ZONES - 1)
        normalized_distance = edge_distance / (center - 0.5)

        return np.array([
            distance,
            edge_distance,
            normalized_position,
            normalized_distance
        ], dtype=np.float32)

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
