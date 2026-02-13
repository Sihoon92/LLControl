"""
MBRL (Model-Based Reinforcement Learning) 설정 파일

PETS (Probabilistic Ensembles with Trajectory Sampling) 알고리즘을 위한
하이퍼파라미터 및 설정값
"""

from pathlib import Path
from typing import Dict, List

# ============================================================================
# 1. 모델 아키텍처 설정
# ============================================================================

# Per-Zone Dynamics Model
DYNAMICS_MODEL_CONFIG = {
    'input_dim': 11,           # 위치(4) + CLR(3) + 제어(4)
    'output_dim': 3,           # diff_CLR (3)
    'hidden_dims': [128, 128], # Hidden layer 구조
    'activation': 'relu',      # 활성화 함수
    'use_layer_norm': True,    # LayerNorm 사용 여부
    'dropout': 0.0,            # Dropout 비율 (0 = 사용 안함)
}

# 앙상블 설정
ENSEMBLE_CONFIG = {
    'n_ensembles': 5,          # 앙상블 개수 (5~7 권장)
    'bootstrap': True,         # Bootstrap sampling 사용
    'bootstrap_ratio': 0.8,    # Bootstrap 샘플링 비율
}

# ============================================================================
# 2. 학습 설정
# ============================================================================

TRAINING_CONFIG = {
    # 옵티마이저
    'optimizer': 'adam',
    'learning_rate': 1e-3,
    'weight_decay': 1e-5,
    'lr_scheduler': 'cosine',  # 'step', 'cosine', 'none'
    'lr_decay_steps': [50, 80],
    'lr_decay_gamma': 0.1,

    # 학습 파라미터
    'batch_size': 256,
    'epochs': 100,
    'early_stopping_patience': 20,
    'min_delta': 1e-6,

    # 정규화
    'grad_clip': 1.0,          # Gradient clipping
    'log_var_min': -10,        # Log variance 하한
    'log_var_max': 2,          # Log variance 상한

    # 검증
    'validation_split': 0.2,
    'test_split': 0.1,
}

# ============================================================================
# 3. 데이터 처리 설정
# ============================================================================

DATA_CONFIG = {
    # 입력 정규화
    'normalize_inputs': True,
    'normalization_method': 'standard',  # 'standard', 'minmax'

    # 출력 정규화
    'normalize_outputs': True,
    'output_normalization_method': 'standard',

    # 시퀀스 설정
    'sequence_length': 1,      # 단일 step 예측
    'horizon': 5,              # Planning horizon

    # 데이터 증강
    'data_augmentation': False,
    'noise_std': 0.01,
}

# ============================================================================
# 4. Planner 설정 (CEM)
# ============================================================================

PLANNER_CONFIG = {
    # CEM 파라미터
    'horizon': 5,              # 예측 구간 (step)
    'n_samples': 500,          # 샘플 수
    'n_elite': 50,             # Elite 수
    'n_iterations': 5,         # CEM 반복 횟수
    'alpha': 0.1,              # 분포 업데이트 가중치

    # 액션 제약
    'action_bounds': {
        'gv': (-2.0, 2.0),     # GV 범위 (mm)
        'rpm': (-50, 50),      # RPM 범위
    },

    # 초기 분포
    'init_mean': 0.0,
    'init_std': 0.5,

    # 불확실성 페널티
    'uncertainty_penalty': 0.1,  # 불확실성에 대한 가중치
}

# ============================================================================
# 5. 경로 설정
# ============================================================================

# 프로젝트 루트 (apc_optimization의 상위)
import sys
from pathlib import Path

# apc_optimization 패키지의 상위 경로
MBRL_PACKAGE_DIR = Path(__file__).parent
APC_OPT_DIR = MBRL_PACKAGE_DIR.parent
PROJECT_ROOT = APC_OPT_DIR.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"
MODEL_DIR = OUTPUT_DIR / "models_v2"

MBRL_OUTPUT_DIR = OUTPUT_DIR / "mbrl"
MBRL_OUTPUT_DIR.mkdir(exist_ok=True)

MBRL_MODEL_DIR = MODEL_DIR / "mbrl"
MBRL_MODEL_DIR.mkdir(exist_ok=True)

MBRL_LOG_DIR = OUTPUT_DIR / "logs" / "mbrl"
MBRL_LOG_DIR.mkdir(exist_ok=True, parents=True)

# ============================================================================
# 6. 로깅 설정
# ============================================================================

LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'log_file': MBRL_LOG_DIR / 'mbrl_training.log',
    'tensorboard_dir': MBRL_LOG_DIR / 'tensorboard',
}

# ============================================================================
# 7. 실험 설정
# ============================================================================

EXPERIMENT_CONFIG = {
    'experiment_name': 'pets_per_zone_v1',
    'random_seed': 42,
    'device': 'cpu',           # 'cpu', 'cuda', 'mps'
    'deterministic': True,

    # 체크포인트
    'save_checkpoints': True,
    'checkpoint_interval': 10,  # epoch마다
    'keep_last_n_checkpoints': 3,

    # 평가
    'evaluate_interval': 5,    # epoch마다
    'n_evaluation_samples': 100,
}

# ============================================================================
# 8. 비교 설정 (CatBoost vs PETS)
# ============================================================================

COMPARISON_CONFIG = {
    # 평가 지표
    'metrics': [
        'mse',
        'mae',
        'rmse',
        'r2_score',
        'nll',               # PETS만
        'calibration_ece',   # PETS만
    ],

    # Open-loop 시뮬레이션
    'open_loop_steps': 20,
    'open_loop_n_trials': 10,

    # 시각화
    'plot_predictions': True,
    'plot_uncertainty': True,
    'plot_comparison': True,
}

# ============================================================================
# 9. 유틸리티 함수
# ============================================================================

def get_model_save_path(experiment_name: str, epoch: int = None) -> Path:
    """모델 저장 경로 생성"""
    if epoch is not None:
        filename = f"{experiment_name}_epoch{epoch}.pt"
    else:
        filename = f"{experiment_name}_best.pt"
    return MBRL_MODEL_DIR / filename


def get_config_summary() -> str:
    """설정 요약 문자열 생성"""
    summary = f"""
    ============================================================================
    MBRL (PETS) 설정 요약
    ============================================================================

    1. 모델 아키텍처
       - Input: {DYNAMICS_MODEL_CONFIG['input_dim']}개
       - Hidden: {DYNAMICS_MODEL_CONFIG['hidden_dims']}
       - Output: {DYNAMICS_MODEL_CONFIG['output_dim']}개
       - 앙상블 개수: {ENSEMBLE_CONFIG['n_ensembles']}

    2. 학습 설정
       - Batch Size: {TRAINING_CONFIG['batch_size']}
       - Learning Rate: {TRAINING_CONFIG['learning_rate']}
       - Epochs: {TRAINING_CONFIG['epochs']}
       - Validation Split: {TRAINING_CONFIG['validation_split']}

    3. Planner (CEM)
       - Horizon: {PLANNER_CONFIG['horizon']} steps
       - Samples: {PLANNER_CONFIG['n_samples']}
       - Elite: {PLANNER_CONFIG['n_elite']}
       - Iterations: {PLANNER_CONFIG['n_iterations']}

    4. 경로
       - 출력: {MBRL_OUTPUT_DIR}
       - 모델: {MBRL_MODEL_DIR}
       - 로그: {MBRL_LOG_DIR}

    ============================================================================
    """
    return summary


if __name__ == '__main__':
    print(get_config_summary())
    print(f"\nModel save path example: {get_model_save_path('test', 10)}")
