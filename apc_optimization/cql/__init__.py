"""
CQL (Conservative Q-Learning) 패키지

d3rlpy 기반 Offline RL 학습 및 추론

주요 모듈:
- config: CQL 하이퍼파라미터 및 경로 설정
- data_processor: MDP parquet → d3rlpy MDPDataset 변환
- train: CQLTrainer 학습/평가/저장
- policy: CQLPolicy 추론 + 액션 제약
"""

__version__ = '0.1.0'
__author__ = 'LLControl Team'

from .config import (
    CQL_ALGORITHM_CONFIG,
    NETWORK_CONFIG,
    TRAINING_CONFIG,
    DATA_CONFIG,
    ACTION_CONSTRAINT_CONFIG,
    EXPERIMENT_CONFIG,
    CQL_OUTPUT_DIR,
    CQL_MODEL_DIR,
    CQL_LOG_DIR,
    get_model_save_path,
    get_config_summary,
)

from .data_processor import CQLDataProcessor
from .policy import CQLPolicy

__all__ = [
    # Config
    'CQL_ALGORITHM_CONFIG',
    'NETWORK_CONFIG',
    'TRAINING_CONFIG',
    'DATA_CONFIG',
    'ACTION_CONSTRAINT_CONFIG',
    'EXPERIMENT_CONFIG',
    'CQL_OUTPUT_DIR',
    'CQL_MODEL_DIR',
    'CQL_LOG_DIR',
    'get_model_save_path',
    'get_config_summary',

    # Data Processing
    'CQLDataProcessor',

    # Policy
    'CQLPolicy',
]
