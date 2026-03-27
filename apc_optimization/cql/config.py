"""
CQL (Conservative Q-Learning) 설정 파일

d3rlpy 기반 Offline RL 학습을 위한 하이퍼파라미터 및 설정값
"""

from pathlib import Path

# ============================================================================
# 1. CQL 알고리즘 설정
# ============================================================================

CQL_ALGORITHM_CONFIG = {
    'actor_learning_rate': 1e-4,
    'critic_learning_rate': 3e-4,
    'alpha_learning_rate': 1e-4,
    'alpha_threshold': 10.0,       # CQL conservative penalty 임계값
    'conservative_weight': 5.0,    # 보수성 가중치 (높을수록 보수적)
    'n_action_samples': 10,        # CQL 손실 계산용 액션 샘플 수
    'batch_size': 256,
    'gamma': 0.99,                 # 할인율 (1-step 에피소드에서는 영향 미미)
    'tau': 0.005,                  # Soft target update 비율
}

# ============================================================================
# 2. 네트워크 설정
# ============================================================================

NETWORK_CONFIG = {
    'actor_hidden_dims': [256, 256, 256],
    'critic_hidden_dims': [256, 256, 256],
    'activation': 'relu',
}

# ============================================================================
# 3. 학습 설정
# ============================================================================

TRAINING_CONFIG = {
    'n_steps': 100000,             # 총 gradient steps
    'n_steps_per_epoch': 1000,     # 에폭당 steps (로깅 단위)
    'validation_split': 0.2,
    'random_seed': 42,
}

# ============================================================================
# 4. 데이터 처리 설정
# ============================================================================

DATA_CONFIG = {
    'normalize_observations': True,
    'normalize_actions': True,
    'normalize_rewards': False,    # reward는 이미 적절한 스케일
    'normalization_method': 'standard',  # 'standard' (StandardScaler)
}

# ============================================================================
# 5. 액션 제약 설정
# ============================================================================

ACTION_CONSTRAINT_CONFIG = {
    'gv_bounds': (-2.0, 2.0),          # GV_GAP 범위 (mm)
    'rpm_bounds': (-50, 50),            # PUMP RPM 범위
    'gv_adjacent_max_diff': 0.5,        # 인접 GV 최대 차이 (mm)
    'gv_total_change_max': 10.0,        # 전체 GV 변화량 상한 (mm)
}

# ============================================================================
# 6. 경로 설정
# ============================================================================

CQL_PACKAGE_DIR = Path(__file__).parent
APC_OPT_DIR = CQL_PACKAGE_DIR.parent
PROJECT_ROOT = APC_OPT_DIR.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"
MODEL_DIR = OUTPUT_DIR / "models_v2"

CQL_OUTPUT_DIR = OUTPUT_DIR / "cql"
CQL_OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

CQL_MODEL_DIR = MODEL_DIR / "cql"
CQL_MODEL_DIR.mkdir(exist_ok=True, parents=True)

CQL_LOG_DIR = OUTPUT_DIR / "logs" / "cql"
CQL_LOG_DIR.mkdir(exist_ok=True, parents=True)

# ============================================================================
# 7. 로깅 설정
# ============================================================================

LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'log_file': CQL_LOG_DIR / 'cql_training.log',
}

# ============================================================================
# 8. 실험 설정
# ============================================================================

EXPERIMENT_CONFIG = {
    'experiment_name': 'cql_offline_rl_v1',
    'random_seed': 42,
    'device': 'cpu',               # 'cpu', 'cuda', 'mps'
    'deterministic': True,
}

# ============================================================================
# 9. 유틸리티 함수
# ============================================================================


def get_model_save_path(experiment_name: str = None) -> Path:
    """모델 저장 경로 생성"""
    if experiment_name is None:
        experiment_name = EXPERIMENT_CONFIG['experiment_name']
    return CQL_MODEL_DIR / f"{experiment_name}_best.d3"


def get_config_summary() -> str:
    """설정 요약 문자열 생성"""
    summary = f"""
    ============================================================================
    CQL (Conservative Q-Learning) 설정 요약
    ============================================================================

    1. 알고리즘 설정
       - Actor LR: {CQL_ALGORITHM_CONFIG['actor_learning_rate']}
       - Critic LR: {CQL_ALGORITHM_CONFIG['critic_learning_rate']}
       - Conservative Weight: {CQL_ALGORITHM_CONFIG['conservative_weight']}
       - Batch Size: {CQL_ALGORITHM_CONFIG['batch_size']}
       - Gamma: {CQL_ALGORITHM_CONFIG['gamma']}

    2. 네트워크 설정
       - Actor Hidden: {NETWORK_CONFIG['actor_hidden_dims']}
       - Critic Hidden: {NETWORK_CONFIG['critic_hidden_dims']}

    3. 학습 설정
       - Total Steps: {TRAINING_CONFIG['n_steps']}
       - Steps per Epoch: {TRAINING_CONFIG['n_steps_per_epoch']}
       - Validation Split: {TRAINING_CONFIG['validation_split']}

    4. 액션 제약
       - GV 범위: {ACTION_CONSTRAINT_CONFIG['gv_bounds']} mm
       - RPM 범위: {ACTION_CONSTRAINT_CONFIG['rpm_bounds']}
       - 인접 차이: ≤ {ACTION_CONSTRAINT_CONFIG['gv_adjacent_max_diff']} mm
       - 총 변화량: ≤ {ACTION_CONSTRAINT_CONFIG['gv_total_change_max']} mm

    5. 경로
       - 출력: {CQL_OUTPUT_DIR}
       - 모델: {CQL_MODEL_DIR}
       - 로그: {CQL_LOG_DIR}

    ============================================================================
    """
    return summary


if __name__ == '__main__':
    print(get_config_summary())
    print(f"\nModel save path: {get_model_save_path()}")
