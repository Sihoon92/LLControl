"""
공통 피처 추출 모듈 (Single Source of Truth)

ModelTrainer와 MBRL이 동일한 피처 정의를 공유하여
정확한 성능 비교가 가능하도록 한다.

피처 구성 (총 11개):
  - 위치 특성 (4): ModelTrainer 방식 — 데이터 컬럼 직접 사용
  - 현재 상태 (3): current_CLR_1/2/3
  - 제어 GV   (3): delta_GV_left1 / delta_GV_self / delta_GV_right1
                    (delta_GV_left2 / right2 제외)
  - 제어 RPM  (1): delta_RPM
"""

import numpy as np
import pandas as pd
from typing import List, Tuple

# ============================================================================
# 정규 피처 정의
# ============================================================================

POSITION_FEATURES: List[str] = [
    'zone_distance_from_center',
    'is_edge',
    'normalized_position',
    'normalized_distance',
]

STATE_FEATURES: List[str] = [
    'current_CLR_1',
    'current_CLR_2',
    'current_CLR_3',
]

CONTROL_FEATURES: List[str] = [
    'delta_GV_left1',
    'delta_GV_self',
    'delta_GV_right1',
    'delta_RPM',
]

OUTPUT_FEATURES: List[str] = [
    'diff_CLR_1',
    'diff_CLR_2',
    'diff_CLR_3',
]

# 전체 입력 피처 (11개)
CANONICAL_INPUT_FEATURES: List[str] = (
    POSITION_FEATURES + STATE_FEATURES + CONTROL_FEATURES
)


# ============================================================================
# 피처 추출 함수
# ============================================================================

def extract_features(
    df: pd.DataFrame,
    input_features: List[str] = None,
    output_features: List[str] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    DataFrame에서 표준 입출력 피처를 추출한다.

    Parameters
    ----------
    df : pd.DataFrame
        원본 데이터 (model_training_data.xlsx 로드 결과)
    input_features : list, optional
        입력 피처 리스트. None이면 CANONICAL_INPUT_FEATURES(11개) 사용.
    output_features : list, optional
        출력 피처 리스트. None이면 OUTPUT_FEATURES(3개) 사용.

    Returns
    -------
    X : np.ndarray, shape (N, 11)
    Y : np.ndarray, shape (N, 3)

    Raises
    ------
    ValueError
        필요한 컬럼이 DataFrame에 없을 경우.
    """
    if input_features is None:
        input_features = CANONICAL_INPUT_FEATURES
    if output_features is None:
        output_features = OUTPUT_FEATURES

    missing = [c for c in input_features + output_features if c not in df.columns]
    if missing:
        raise ValueError(
            f"데이터에 필요한 컬럼이 없습니다: {missing}\n"
            f"사용 가능한 컬럼: {list(df.columns)}"
        )

    X = df[input_features].values.astype(np.float32)
    Y = df[output_features].values.astype(np.float32)
    return X, Y
