"""
CQL 데이터 처리기

offline_rl_data_preparator.py가 생성한 MDP parquet 데이터를
d3rlpy MDPDataset으로 변환
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import logging
import re

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from utils import load_file
from .config import DATA_CONFIG, TRAINING_CONFIG

logger = logging.getLogger(__name__)


class CQLDataProcessor:
    """
    CQL 데이터 처리기

    MDP parquet → d3rlpy MDPDataset 변환
    """

    def __init__(
        self,
        normalize_observations: bool = None,
        normalize_actions: bool = None,
        normalize_rewards: bool = None,
        validation_split: float = None,
        random_seed: int = None
    ):
        if normalize_observations is None:
            normalize_observations = DATA_CONFIG['normalize_observations']
        if normalize_actions is None:
            normalize_actions = DATA_CONFIG['normalize_actions']
        if normalize_rewards is None:
            normalize_rewards = DATA_CONFIG['normalize_rewards']
        if validation_split is None:
            validation_split = TRAINING_CONFIG['validation_split']
        if random_seed is None:
            random_seed = TRAINING_CONFIG['random_seed']

        self.normalize_observations = normalize_observations
        self.normalize_actions = normalize_actions
        self.normalize_rewards = normalize_rewards
        self.validation_split = validation_split
        self.random_seed = random_seed

        # Scalers
        self.obs_scaler = StandardScaler() if normalize_observations else None
        self.act_scaler = StandardScaler() if normalize_actions else None
        self.rew_scaler = StandardScaler() if normalize_rewards else None
        self.is_fitted = False

        # 칼럼 정보 (동적 탐지 결과)
        self.state_columns: List[str] = []
        self.action_columns: List[str] = []
        self.next_state_columns: List[str] = []
        self.obs_dim: int = 0
        self.act_dim: int = 0

    def load_mdp_data(self, data_file: str) -> pd.DataFrame:
        """MDP 데이터 로드"""
        logger.info(f"MDP 데이터 로드: {data_file}")
        df = load_file(data_file, logger=logger)
        logger.info(f"  로드 완료: {len(df)} 행, {len(df.columns)} 열")
        return df

    def extract_sars(
        self, df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        DataFrame에서 S, A, R, S', terminals 추출

        칼럼명 패턴으로 동적 탐지:
        - State: S_Z{n}_CLR_div_{d}
        - Action: delta_GV_GAP{nn}, delta_RPM
        - Reward: reward
        - Next State: NS_Z{n}_CLR_div_{d}

        Returns:
            (observations, actions, rewards, next_observations, terminals)
        """
        # 칼럼 동적 탐지
        self.state_columns = sorted(
            [c for c in df.columns if c.startswith('S_Z')],
            key=self._sort_key_state
        )
        self.action_columns = sorted(
            [c for c in df.columns if c.startswith('delta_')]
        )
        self.next_state_columns = sorted(
            [c for c in df.columns if c.startswith('NS_Z')],
            key=self._sort_key_state
        )

        self.obs_dim = len(self.state_columns)
        self.act_dim = len(self.action_columns)

        logger.info(f"  State 칼럼: {self.obs_dim}개 {self.state_columns[:3]}...")
        logger.info(f"  Action 칼럼: {self.act_dim}개 {self.action_columns}")
        logger.info(f"  Next State 칼럼: {len(self.next_state_columns)}개")

        # ndarray 변환
        observations = df[self.state_columns].values.astype(np.float32)
        actions = df[self.action_columns].values.astype(np.float32)
        rewards = df['reward'].values.astype(np.float32)
        next_observations = df[self.next_state_columns].values.astype(np.float32)

        # 각 행은 독립적인 1-step 에피소드
        terminals = np.ones(len(df), dtype=np.float32)

        # NaN 처리
        nan_obs = np.isnan(observations).sum()
        nan_act = np.isnan(actions).sum()
        if nan_obs > 0:
            logger.warning(f"  Observations에 NaN {nan_obs}개 → 0으로 대체")
            observations = np.nan_to_num(observations, nan=0.0)
        if nan_act > 0:
            logger.warning(f"  Actions에 NaN {nan_act}개 → 0으로 대체")
            actions = np.nan_to_num(actions, nan=0.0)

        logger.info(f"  추출 완료: obs={observations.shape}, act={actions.shape}, "
                     f"rew={rewards.shape}")

        return observations, actions, rewards, next_observations, terminals

    def normalize_data(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        fit: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """데이터 정규화 (학습 데이터 기준으로 fit)"""
        if fit:
            if self.obs_scaler:
                self.obs_scaler.fit(observations)
            if self.act_scaler:
                self.act_scaler.fit(actions)
            if self.rew_scaler:
                self.rew_scaler.fit(rewards.reshape(-1, 1))
            self.is_fitted = True

        obs_norm = self.obs_scaler.transform(observations) if self.obs_scaler else observations
        act_norm = self.act_scaler.transform(actions) if self.act_scaler else actions
        rew_norm = (self.rew_scaler.transform(rewards.reshape(-1, 1)).flatten()
                    if self.rew_scaler else rewards)

        return (obs_norm.astype(np.float32),
                act_norm.astype(np.float32),
                rew_norm.astype(np.float32))

    def create_dataset(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        terminals: np.ndarray
    ):
        """d3rlpy MDPDataset 생성"""
        import d3rlpy

        dataset = d3rlpy.dataset.MDPDataset(
            observations=observations,
            actions=actions,
            rewards=rewards,
            terminals=terminals,
        )
        logger.info(f"  MDPDataset 생성 완료: {len(observations)} transitions")
        return dataset

    def process(self, data_file: str) -> Dict:
        """
        전체 데이터 처리 파이프라인

        Returns:
            dict: {
                'train_dataset': MDPDataset,
                'val_dataset': MDPDataset,
                'obs_dim': int,
                'act_dim': int,
                'state_columns': list,
                'action_columns': list,
                'n_transitions': int,
            }
        """
        logger.info("=" * 80)
        logger.info("CQL 데이터 처리 시작")
        logger.info("=" * 80)

        # 1. 로드
        df = self.load_mdp_data(data_file)

        # 2. S/A/R/S'/T 추출
        obs, act, rew, next_obs, terminals = self.extract_sars(df)

        # 3. Train/Val 분할
        indices = np.arange(len(obs))
        train_idx, val_idx = train_test_split(
            indices,
            test_size=self.validation_split,
            random_state=self.random_seed
        )

        logger.info(f"  Train: {len(train_idx)} 샘플, Val: {len(val_idx)} 샘플")

        # 4. 정규화 (train 기준으로 fit)
        obs_train, act_train, rew_train = self.normalize_data(
            obs[train_idx], act[train_idx], rew[train_idx], fit=True
        )
        obs_val, act_val, rew_val = self.normalize_data(
            obs[val_idx], act[val_idx], rew[val_idx], fit=False
        )

        # 5. MDPDataset 생성
        train_dataset = self.create_dataset(
            obs_train, act_train, rew_train, terminals[train_idx]
        )
        val_dataset = self.create_dataset(
            obs_val, act_val, rew_val, terminals[val_idx]
        )

        logger.info("=" * 80)
        logger.info("CQL 데이터 처리 완료")
        logger.info("=" * 80)

        return {
            'train_dataset': train_dataset,
            'val_dataset': val_dataset,
            'obs_dim': self.obs_dim,
            'act_dim': self.act_dim,
            'state_columns': self.state_columns,
            'action_columns': self.action_columns,
            'n_transitions': len(obs),
        }

    def inverse_transform_action(self, normalized_action: np.ndarray) -> np.ndarray:
        """정규화된 액션을 원본 스케일로 복원"""
        if self.act_scaler and self.is_fitted:
            if normalized_action.ndim == 1:
                return self.act_scaler.inverse_transform(
                    normalized_action.reshape(1, -1)
                ).flatten().astype(np.float32)
            return self.act_scaler.inverse_transform(normalized_action).astype(np.float32)
        return normalized_action

    def inverse_transform_observation(self, normalized_obs: np.ndarray) -> np.ndarray:
        """정규화된 관측을 원본 스케일로 복원"""
        if self.obs_scaler and self.is_fitted:
            if normalized_obs.ndim == 1:
                return self.obs_scaler.inverse_transform(
                    normalized_obs.reshape(1, -1)
                ).flatten().astype(np.float32)
            return self.obs_scaler.inverse_transform(normalized_obs).astype(np.float32)
        return normalized_obs

    def transform_observation(self, observation: np.ndarray) -> np.ndarray:
        """원본 관측을 정규화"""
        if self.obs_scaler and self.is_fitted:
            if observation.ndim == 1:
                return self.obs_scaler.transform(
                    observation.reshape(1, -1)
                ).flatten().astype(np.float32)
            return self.obs_scaler.transform(observation).astype(np.float32)
        return observation

    def save_scaler(self, path: str):
        """스케일러 및 칼럼 메타데이터 저장"""
        import joblib
        if self.is_fitted:
            joblib.dump({
                'obs_scaler': self.obs_scaler,
                'act_scaler': self.act_scaler,
                'rew_scaler': self.rew_scaler,
                'state_columns': self.state_columns,
                'action_columns': self.action_columns,
                'next_state_columns': self.next_state_columns,
                'obs_dim': self.obs_dim,
                'act_dim': self.act_dim,
            }, path)
            logger.info(f"스케일러 저장: {path}")

    def load_scaler(self, path: str):
        """스케일러 및 칼럼 메타데이터 로드"""
        import joblib
        state = joblib.load(path)
        self.obs_scaler = state['obs_scaler']
        self.act_scaler = state['act_scaler']
        self.rew_scaler = state['rew_scaler']
        self.state_columns = state['state_columns']
        self.action_columns = state['action_columns']
        self.next_state_columns = state['next_state_columns']
        self.obs_dim = state['obs_dim']
        self.act_dim = state['act_dim']
        self.is_fitted = True
        logger.info(f"스케일러 로드: {path}")

    @staticmethod
    def _sort_key_state(col: str) -> Tuple[int, int]:
        """State 칼럼 정렬 키: zone 번호 → div 번호 순"""
        match = re.search(r'Z(\d+)_CLR_div_(\d+)', col)
        if match:
            return int(match.group(1)), int(match.group(2))
        return (0, 0)
