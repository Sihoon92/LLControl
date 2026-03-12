"""
CQL 정책 (Policy) 모듈

학습된 CQL 모델을 로드하여 액션을 예측하고,
공정 제약 조건을 적용
"""

import numpy as np
from typing import Dict, Optional
from pathlib import Path
import logging

from .config import ACTION_CONSTRAINT_CONFIG

logger = logging.getLogger(__name__)


class CQLPolicy:
    """
    CQL 정책: 학습된 모델로 액션 예측 + 제약 적용

    사용 예시:
        policy = CQLPolicy(model_path='outputs/cql/training/models/best_model.d3',
                           scaler_path='outputs/cql/training/scaler.pkl')
        action = policy.predict(current_observation)
    """

    def __init__(
        self,
        model_path: str,
        scaler_path: str,
        constraint_config: Optional[Dict] = None
    ):
        """
        Args:
            model_path: 학습된 모델 경로 (.d3 파일)
            scaler_path: 스케일러 경로 (.pkl 파일)
            constraint_config: 액션 제약 설정 (None이면 기본값 사용)
        """
        import d3rlpy
        import joblib

        # 모델 로드
        self.model = d3rlpy.load_learnable(model_path)
        logger.info(f"CQL 모델 로드: {model_path}")

        # 스케일러 로드
        scaler_state = joblib.load(scaler_path)
        self.obs_scaler = scaler_state['obs_scaler']
        self.act_scaler = scaler_state['act_scaler']
        self.state_columns = scaler_state['state_columns']
        self.action_columns = scaler_state['action_columns']
        self.obs_dim = scaler_state['obs_dim']
        self.act_dim = scaler_state['act_dim']
        logger.info(f"스케일러 로드: {scaler_path}")
        logger.info(f"  obs_dim={self.obs_dim}, act_dim={self.act_dim}")

        # 제약 조건
        self.constraints = constraint_config or ACTION_CONSTRAINT_CONFIG
        self.gv_bounds = self.constraints['gv_bounds']
        self.rpm_bounds = self.constraints['rpm_bounds']
        self.gv_adjacent_max_diff = self.constraints['gv_adjacent_max_diff']
        self.gv_total_change_max = self.constraints['gv_total_change_max']

        # GV/RPM 인덱스 분리
        self.gv_indices = [
            i for i, c in enumerate(self.action_columns)
            if c.startswith('delta_GV')
        ]
        self.rpm_index = next(
            (i for i, c in enumerate(self.action_columns) if c == 'delta_RPM'),
            None
        )

    def predict(self, observation: np.ndarray) -> np.ndarray:
        """
        관측에서 액션 예측 (정규화 → 추론 → 역정규화 → 제약 적용)

        Args:
            observation: 원본 스케일 관측 (obs_dim,) 또는 (1, obs_dim)

        Returns:
            constrained_action: 제약 적용된 액션 (act_dim,)
        """
        obs = np.asarray(observation, dtype=np.float32)
        if obs.ndim == 1:
            obs = obs.reshape(1, -1)

        # 정규화
        if self.obs_scaler:
            obs_norm = self.obs_scaler.transform(obs).astype(np.float32)
        else:
            obs_norm = obs

        # 모델 추론
        raw_action = self.model.predict(obs_norm)[0]

        # 역정규화
        if self.act_scaler:
            action = self.act_scaler.inverse_transform(
                raw_action.reshape(1, -1)
            ).flatten().astype(np.float32)
        else:
            action = raw_action

        # 제약 적용
        return self._enforce_constraints(action)

    def predict_batch(self, observations: np.ndarray) -> np.ndarray:
        """배치 예측"""
        obs = np.asarray(observations, dtype=np.float32)
        if obs.ndim == 1:
            obs = obs.reshape(1, -1)

        if self.obs_scaler:
            obs_norm = self.obs_scaler.transform(obs).astype(np.float32)
        else:
            obs_norm = obs

        raw_actions = self.model.predict(obs_norm)

        if self.act_scaler:
            actions = self.act_scaler.inverse_transform(raw_actions).astype(np.float32)
        else:
            actions = raw_actions

        return np.array([self._enforce_constraints(a) for a in actions])

    def predict_with_info(self, observation: np.ndarray) -> Dict:
        """디버깅용 상세 결과 반환"""
        obs = np.asarray(observation, dtype=np.float32)
        if obs.ndim == 1:
            obs = obs.reshape(1, -1)

        if self.obs_scaler:
            obs_norm = self.obs_scaler.transform(obs).astype(np.float32)
        else:
            obs_norm = obs

        raw_action_norm = self.model.predict(obs_norm)[0]

        if self.act_scaler:
            raw_action = self.act_scaler.inverse_transform(
                raw_action_norm.reshape(1, -1)
            ).flatten().astype(np.float32)
        else:
            raw_action = raw_action_norm

        constrained_action = self._enforce_constraints(raw_action.copy())
        constraints_violated = not np.allclose(raw_action, constrained_action)

        # 칼럼별 매핑
        action_dict = {
            col: float(constrained_action[i])
            for i, col in enumerate(self.action_columns)
        }

        return {
            'action': constrained_action,
            'raw_action': raw_action,
            'constraints_violated': constraints_violated,
            'action_dict': action_dict,
        }

    def _enforce_constraints(self, action: np.ndarray) -> np.ndarray:
        """
        액션 제약 적용

        1. GV 값 범위 클리핑
        2. RPM 값 범위 클리핑
        3. 인접 GV 차이 제약 (반복 프로젝션)
        4. 총 GV 변화량 제약 (비례 스케일링)
        """
        action = action.copy()

        # 1. GV 범위 클리핑
        for idx in self.gv_indices:
            action[idx] = np.clip(action[idx], self.gv_bounds[0], self.gv_bounds[1])

        # 2. RPM 범위 클리핑
        if self.rpm_index is not None:
            action[self.rpm_index] = np.clip(
                action[self.rpm_index], self.rpm_bounds[0], self.rpm_bounds[1]
            )

        # 3. 인접 GV 차이 제약 (반복 프로젝션)
        if len(self.gv_indices) > 1:
            for _ in range(3):  # 최대 3회 반복
                satisfied = True
                for i in range(len(self.gv_indices) - 1):
                    idx_a = self.gv_indices[i]
                    idx_b = self.gv_indices[i + 1]
                    diff = action[idx_b] - action[idx_a]

                    if abs(diff) > self.gv_adjacent_max_diff:
                        satisfied = False
                        mid = (action[idx_a] + action[idx_b]) / 2
                        half_max = self.gv_adjacent_max_diff / 2
                        action[idx_a] = mid - half_max * np.sign(diff)
                        action[idx_b] = mid + half_max * np.sign(diff)

                if satisfied:
                    break

            # 인접 제약 후 범위 다시 클리핑
            for idx in self.gv_indices:
                action[idx] = np.clip(action[idx], self.gv_bounds[0], self.gv_bounds[1])

        # 4. 총 GV 변화량 제약 (비례 스케일링)
        gv_values = action[self.gv_indices]
        total_change = np.sum(np.abs(gv_values))
        if total_change > self.gv_total_change_max:
            scale = self.gv_total_change_max / total_change
            action[self.gv_indices] = gv_values * scale

        return action

    def get_action_column_mapping(self) -> Dict[str, int]:
        """칼럼명 → 인덱스 매핑"""
        return {col: i for i, col in enumerate(self.action_columns)}
