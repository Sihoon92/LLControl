"""
MBRL (Model-Based Reinforcement Learning) 패키지

PETS (Probabilistic Ensembles with Trajectory Sampling) 알고리즘 구현

Per-Zone 접근법:
- CatBoost와 동일한 입력/출력 구조
- Zone별 독립 예측 (인접 정보 포함)
- 확률적 앙상블로 불확실성 제공

주요 모듈:
- config: PETS 설정
- ensemble_nn: Per-Zone Neural Network
- dynamics_model: 확률적 Dynamics Model
- data_processor: 데이터 전처리
"""

__version__ = '0.1.0'
__author__ = 'LLControl Team'

from .config import (
    DYNAMICS_MODEL_CONFIG,
    ENSEMBLE_CONFIG,
    TRAINING_CONFIG,
    DATA_CONFIG,
    PLANNER_CONFIG,
    EXPERIMENT_CONFIG,
    COMPARISON_CONFIG,
    MBRL_OUTPUT_DIR,
    MBRL_MODEL_DIR,
    MBRL_LOG_DIR,
    get_model_save_path,
    get_config_summary,
)

from .ensemble_nn import (
    PerZoneDynamicsNN,
    EnsembleWrapper,
)

from .dynamics_model import (
    PerZoneProbabilisticEnsemble,
    ZoneProperties,
)

from .data_processor import (
    PETSDataProcessor,
)

__all__ = [
    # Config
    'DYNAMICS_MODEL_CONFIG',
    'ENSEMBLE_CONFIG',
    'TRAINING_CONFIG',
    'DATA_CONFIG',
    'PLANNER_CONFIG',
    'EXPERIMENT_CONFIG',
    'COMPARISON_CONFIG',
    'MBRL_OUTPUT_DIR',
    'MBRL_MODEL_DIR',
    'MBRL_LOG_DIR',
    'get_model_save_path',
    'get_config_summary',

    # Neural Networks
    'PerZoneDynamicsNN',
    'EnsembleWrapper',

    # Dynamics Model
    'PerZoneProbabilisticEnsemble',
    'ZoneProperties',

    # Data Processing
    'PETSDataProcessor',
]


def print_package_info():
    """패키지 정보 출력"""
    print("="*80)
    print("MBRL (PETS) 패키지 정보")
    print("="*80)
    print(f"버전: {__version__}")
    print(f"저자: {__author__}")
    print()
    print("주요 클래스:")
    print("  - PerZoneProbabilisticEnsemble: 확률적 Dynamics Model")
    print("  - PETSDataProcessor: 데이터 처리기")
    print("  - PerZoneDynamicsNN: Neural Network")
    print()
    print("출력 디렉토리:")
    print(f"  - 결과: {MBRL_OUTPUT_DIR}")
    print(f"  - 모델: {MBRL_MODEL_DIR}")
    print(f"  - 로그: {MBRL_LOG_DIR}")
    print("="*80)


if __name__ == '__main__':
    print_package_info()
    print()
    print(get_config_summary())
