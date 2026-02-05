"""
APC 최적화 엔진 설정 파일
3-Layer 아키텍처의 모든 하이퍼파라미터 정의
"""

import os
from pathlib import Path
from typing import Dict, List, Tuple

# ============================================================================
# 1. 시스템 설정
# ============================================================================

# 프로젝트 루트 경로
PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# 최적화 결과 저장 경로
OPTIMIZATION_OUTPUT_DIR = OUTPUT_DIR / "optimization_results"
OPTIMIZATION_OUTPUT_DIR.mkdir(exist_ok=True)

# 로그 디렉토리
LOG_DIR = OUTPUT_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

# 모델 저장 경로
MODEL_DIR = OUTPUT_DIR / "models_v2"
MODEL_DIR.mkdir(exist_ok=True)

# 재현성을 위한 시드 값
RANDOM_SEED = 42

# ============================================================================
# 2. 공정 설정 (Zone 및 제어 변수)
# ============================================================================

# Zone 설정
N_ZONES = 11  # 11개 Zone (01~11)
ZONE_NAMES = [f"Zone{i:02d}" for i in range(1, N_ZONES + 1)]

# 제어 변수
N_GV = 11  # GV_GAP 01~11
N_RPM = 1  # PUMP RPM
N_CONTROL_VARS = N_GV + N_RPM  # 총 12개

# ============================================================================
# 3. 제약 조건 (Constraint)
# ============================================================================

# 경계값 (Bounds)
GV_GAP_BOUNDS = {
    'lower': -2.0,  # mm
    'upper': 2.0,   # mm
}

RPM_BOUNDS = {
    'lower': -50,  # rpm
    'upper': 50,   # rpm
}

# 제약 조건
GV_ADJACENT_MAX_DIFF = 0.5  # 인접 Zone 최대 차이 (mm)
GV_TOTAL_CHANGE_MAX = 10.0  # 전체 변화량 상한 (mm)

# UCL/LCL 범위 (부분적 제어 시 사용)
CONTROL_LIMITS = {
    'ucl': 1.0,  # Upper Control Limit (정규화된 범위)
    'lcl': 0.0,  # Lower Control Limit
}

# ============================================================================
# 4. 비용 함수 (Cost Function)
# ============================================================================

# 4.1 가중치 (Multi-Objective)
COST_WEIGHTS = {
    'quality': 10.0,   # w₁: 품질 최우선
    'balance': 5.0,    # w₂: 균형 중요
    'control': 1.0,    # w₃: 제어 최소화
    'safety': 15.0,    # w₄: 안전 최우선
}

# 4.2 품질 비용 (Quality Cost)
QUALITY_COST_PARAMS = {
    'alpha': 0.5,      # P_Low, P_High 페널티 가중치
    'target_p_mid': 1.0,  # 목표 P_Mid 값
    'target_p_low_high': 0.0,  # 목표 P_Low, P_High 값
}

# 4.3 균형 비용 (Balance Cost)
# 변동계수의 제곱을 사용: σ² / μ²
BALANCE_COST_PARAMS = {
    'min_mean': 0.1,   # 분모 보호 (div by zero 방지)
}

# 4.4 제어 비용 (Control Cost)
CONTROL_COST_PARAMS = {
    'gv_max': 2.0,     # GV 정규화 기준값 (mm)
    'rpm_max': 50,     # RPM 정규화 기준값
    'beta': 0.7,       # GV 가중치
    'gamma': 0.3,      # RPM 가중치
}

# 4.5 안전 비용 (Safety Cost)
SAFETY_COST_PARAMS = {
    'adjacent_diff_penalty': 1.0,  # V₁ 가중치
    'total_change_penalty': 1.0,   # V₂ 가중치
    'ucl_violation_penalty': 1.0,  # V₃ 가중치
    'lcl_violation_penalty': 1.0,  # V₄ 가중치
}

# ============================================================================
# 5. 최적화 엔진 (Optimizer Engine)
# ============================================================================

# 5.1 Differential Evolution 설정
DE_OPTIMIZER_PARAMS = {
    'strategy': 'best1bin',      # 전략
    'maxiter': 100,              # 최대 반복 횟수
    'popsize': 15,               # 인구수 = 15 × dimension = 15 × 12 = 180
    'tol': 0.001,                # 공차
    'atol': 0.001,               # 절대 공차
    'seed': RANDOM_SEED,         # 시드
    'workers': -1,               # 병렬화 (모든 CPU 사용)
    'updating': 'immediate',     # 즉시 업데이트
    'polish': True,              # 최종 정제 (L-BFGS)
    'init': 'latinhypercube',    # 초기화 방식 (scipy 형식)
    'verbose': True,             # 상세 출력
}

# 5.2 제약 조건 함수 설정
CONSTRAINT_PARAMS = {
    'enforce_bounds': True,      # 경계값 강제
    'enforce_adjacent': True,    # 인접 차이 강제
    'enforce_total': True,       # 총 변화량 강제
    'penalty_multiplier': 1e6,   # 위반 페널티 배수
}

# ============================================================================
# 6. CLR (Center-Log-Ratio) 변환 설정
# ============================================================================

CLR_PARAMS = {
    'epsilon': 1e-6,           # 0 값 대체
    'n_components': 3,         # CLR 성분 수 (Low, Mid, High)
}

# ============================================================================
# 7. Monte Carlo Simulation
# ============================================================================

MC_SIMULATION_PARAMS = {
    'n_simulations': 100,          # 시뮬레이션 횟수
    'model_rmse': 0.05,            # 모델 예측 오차 (RMSE) - 학습 후 업데이트
    'measurement_noise_ratio': 0.05,  # 측정 오차 (±5% relative)
    'confidence_level': 0.95,       # 신뢰도 (95% CI)
}

# ============================================================================
# 8. Decision Support System
# ============================================================================

DSS_PARAMS = {
    'top_n_scenarios': 3,          # Top-N 시나리오 수
    'risk_thresholds': {
        'low': 0.05,               # 위험도 LOW 임계값 (< 5%)
        'medium': 0.15,            # 위험도 MEDIUM 임계값 (< 15%)
        'high': 1.0,               # 위험도 HIGH (>= 15%)
    },
    'constraint_violation_threshold': 0.15,  # 제약 위반 용허 임계값 (15%)
}

# ============================================================================
# 9. 검증 프레임워크 (Validation Framework)
# ============================================================================

VALIDATION_PARAMS = {
    'test_size': 0.3,              # 테스트 데이터 비율
    'success_threshold': 0.8,      # 성공 기준 (P_Mid > 0.8)
    'constraint_tolerance': 0.05,  # 제약 위반 허용도 (5%)
}

# ============================================================================
# 10. 모델 인터페이스
# ============================================================================

MODEL_PARAMS = {
    'model_type': 'catboost',      # 모델 타입: 'catboost', 'xgboost', 'gpr'
    'model_name': 'CatBoost_multi',  # 학습된 모델 이름
    'batch_size': 1000,            # 배치 예측 크기
    'use_gpu': False,              # GPU 사용 여부 (CatBoost)
}

# ============================================================================
# 11. 입출력 데이터 구조
# ============================================================================

# 모델 입력 특성 패턴
INPUT_FEATURE_PATTERNS = {
    'current_clr': 'current_CLR',  # 현재 CLR 값
    'delta_gv': 'delta_GV',        # GV 변화량
    'delta_rpm': 'delta_RPM',      # RPM 변화량
}

# 모델 출력 특성 패턴
OUTPUT_FEATURE_PATTERNS = {
    'diff_clr': 'diff_CLR',        # CLR 변화량
}

# CLR 성분명
CLR_COMPONENTS = ['CLR_1', 'CLR_2', 'CLR_3']  # Low, Mid, High

# 확률 성분명 (역변환 후)
PROBABILITY_COMPONENTS = ['P_Low', 'P_Mid', 'P_High']

# ============================================================================
# 12. 로깅 설정
# ============================================================================

LOGGING_PARAMS = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file': LOG_DIR / 'optimization.log',
    'max_bytes': 10 * 1024 * 1024,  # 10MB
    'backup_count': 5,
}

# ============================================================================
# 13. 유틸리티 함수
# ============================================================================

def get_bounds_array() -> Tuple[List[float], List[float]]:
    """
    최적화 변수의 경계값 배열 반환
    Returns:
        (lower_bounds, upper_bounds): 각각 길이 12의 배열
    """
    bounds_lower = [GV_GAP_BOUNDS['lower']] * N_GV + [RPM_BOUNDS['lower']]
    bounds_upper = [GV_GAP_BOUNDS['upper']] * N_GV + [RPM_BOUNDS['upper']]
    return bounds_lower, bounds_upper


def get_zone_properties() -> Dict[str, Dict]:
    """
    Zone별 고정 속성 반환
    Returns:
        dict: Zone별 거리, 위치 등의 속성
    """
    properties = {}
    for i in range(N_ZONES):
        zone_name = ZONE_NAMES[i]
        properties[zone_name] = {
            'zone_id': i + 1,
            'distance': i * 100,  # 상대 거리 (가상값, 실제 공정값으로 수정)
            'edge_distance': abs(i - N_ZONES // 2),  # 중앙으로부터의 거리
        }
    return properties


def create_config_summary() -> str:
    """
    현재 설정값의 요약 문자열 생성
    """
    summary = f"""
    ============================================================================
    APC 최적화 엔진 설정 요약
    ============================================================================

    1. 공정 설정
       - Zone 수: {N_ZONES}
       - 제어 변수: GV {N_GV}개 + RPM {N_RPM}개 = {N_CONTROL_VARS}개

    2. 제약 조건
       - GV 범위: [{GV_GAP_BOUNDS['lower']}, {GV_GAP_BOUNDS['upper']}] mm
       - RPM 범위: [{RPM_BOUNDS['lower']}, {RPM_BOUNDS['upper']}] rpm
       - 인접 차이: ≤ {GV_ADJACENT_MAX_DIFF} mm
       - 총 변화량: ≤ {GV_TOTAL_CHANGE_MAX} mm

    3. 비용 함수 (가중치)
       - 품질: {COST_WEIGHTS['quality']}
       - 균형: {COST_WEIGHTS['balance']}
       - 제어: {COST_WEIGHTS['control']}
       - 안전: {COST_WEIGHTS['safety']}

    4. 최적화 엔진
       - 알고리즘: Differential Evolution
       - 최대 반복: {DE_OPTIMIZER_PARAMS['maxiter']}
       - 인구수: {DE_OPTIMIZER_PARAMS['popsize'] * N_CONTROL_VARS}

    5. Monte Carlo
       - 시뮬레이션 횟수: {MC_SIMULATION_PARAMS['n_simulations']}
       - 신뢰도: {MC_SIMULATION_PARAMS['confidence_level'] * 100}%

    6. 의사결정 지원
       - Top-N 시나리오: {DSS_PARAMS['top_n_scenarios']}
       - 위험도 임계값: LOW < {DSS_PARAMS['risk_thresholds']['low'] * 100}%, MEDIUM < {DSS_PARAMS['risk_thresholds']['medium'] * 100}%

    ============================================================================
    """
    return summary


if __name__ == '__main__':
    print(create_config_summary())
    print(f"프로젝트 루트: {PROJECT_ROOT}")
    print(f"출력 디렉토리: {OUTPUT_DIR}")
    print(f"경계값: {get_bounds_array()}")
