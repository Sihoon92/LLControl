"""
APC 최적화 엔진 패키지

3-Layer 아키텍처:
  Layer 1: Prediction Engine (CatBoost 모델)
  Layer 2: Optimization Engine (Differential Evolution)
  Layer 3: Decision Support System

주요 모듈:
  - config: 설정값 및 하이퍼파라미터
  - cost_function: 다목적 비용함수
  - model_interface: CatBoost 모델 인터페이스
  - multi_zone_controller: 11-zone 제어기 (Fan-out 패턴)
  - optimizer_engine: DE 최적화 엔진
  - uncertainty_analyzer: Monte Carlo 불확실성 분석
  - decision_support: 의사결정 지원 시스템
  - validation_framework: 검증 프레임워크
"""

__version__ = '0.1.0'
__author__ = 'LLControl Team'

from .config import (
    N_ZONES, N_GV, N_CONTROL_VARS,
    COST_WEIGHTS, DE_OPTIMIZER_PARAMS, CONTROL_LIMITS,
    get_bounds_array, get_zone_properties, create_config_summary,
    PROJECT_ROOT, OUTPUT_DIR, OPTIMIZATION_OUTPUT_DIR, LOG_DIR, MODEL_DIR
)

from .cost_function import CostFunctionEvaluator

from .model_interface import CatBoostModelManager, MockCatBoostModel

from .output_transformer import OutputTransformer, TransformConfig

from .normalizer import ControlVariableNormalizer

from .multi_zone_controller import MultiZoneController, ZoneProperties

from .optimizer_engine import (
    DifferentialEvolutionOptimizer,
    OptimizationResult
)

from .uncertainty_analyzer import (
    MonteCarloUncertaintyAnalyzer,
    MonteCarloResults
)

from .decision_support import DecisionSupportSystem, Scenario

from .validation_framework import OfflineValidationFramework, ValidationMetrics

__all__ = [
    # Config
    'N_ZONES', 'N_GV', 'N_CONTROL_VARS',
    'COST_WEIGHTS', 'DE_OPTIMIZER_PARAMS', 'CONTROL_LIMITS',
    'get_bounds_array', 'get_zone_properties', 'create_config_summary',
    'PROJECT_ROOT', 'OUTPUT_DIR', 'OPTIMIZATION_OUTPUT_DIR', 'LOG_DIR', 'MODEL_DIR',

    # Cost Function
    'CostFunctionEvaluator',

    # Model Interface
    'CatBoostModelManager', 'MockCatBoostModel',

    # Output Transformer
    'OutputTransformer', 'TransformConfig',

    # Normalizer
    'ControlVariableNormalizer',

    # Multi-Zone Controller
    'MultiZoneController', 'ZoneProperties',

    # Optimizer
    'DifferentialEvolutionOptimizer', 'OptimizationResult',

    # Uncertainty Analysis
    'MonteCarloUncertaintyAnalyzer', 'MonteCarloResults',

    # Decision Support
    'DecisionSupportSystem', 'Scenario',

    # Validation
    'OfflineValidationFramework', 'ValidationMetrics',
]
