# 통합 정규화 구현 계획 (Option A)

## 1. 현황 분석

### 1.1 문제 상황
```
예측 모델 (model_interface.py)
    ↓
StandardScaler: (x - μ) / σ
범위: (-∞, +∞)
    ↓
모델 입력값: 데이터 통계 기반 정규화
    ✗ 최적화 모델과 기준 불일치

최적화 모델 (cost_function.py)
    ↓
MinMax (절댓값): x / max
범위: [0, 1]
    ↓
비용 함수 입력값: 고정값 기반 정규화 (gv_max=2.0, rpm_max=50)
    ✗ 예측 모델과 기준 불일치
```

### 1.2 영향 범위
- **evaluation_metrics.py**: 비용 분석 (evaluation_metrics.py:34-80)
- **cost_function.py**: 제어 비용 계산 (cost_function.py:148-190)
- **model_interface.py**: 모델 예측 (model_interface.py:123-177)
- **optimizer_engine.py**: 최적화 실행 (optimizer_engine.py에서 cost_function 사용)

---

## 2. 통합 정규화 클래스 설계

### 2.1 클래스 구조

```python
# apc_optimization/normalizer.py

class ControlVariableNormalizer:
    """
    제어 변수(△GV, △RPM) 정규화 통합 관리자

    역할:
    - 예측 모델과 최적화 모델 간 정규화 기준 통일
    - MinMax 스타일 정규화 (절댓값 기준)
    - 양방향 변환 지원 (정규화 ↔ 역정규화)
    """

    def __init__(self, gv_max: float = 2.0, rpm_max: float = 50.0)
    def normalize_control_vars(self, delta_gv, delta_rpm)
    def denormalize_control_vars(self, gv_normalized, rpm_normalized)
    def get_config_dict(self)
```

### 2.2 설계 원칙

| 원칙 | 설명 |
|------|------|
| **단일 진실 공급원** | 정규화 기준은 한곳에서만 관리 |
| **일관성** | 예측 모델과 최적화 모델이 동일한 기준 사용 |
| **확장성** | 향후 다른 정규화 방식 추가 가능 |
| **추적성** | config.py의 CONTROL_COST_PARAMS와 연동 |

---

## 3. 단계별 수정 계획

### Phase 1: 통합 정규화 클래스 구현 (1단계)

#### 1.1 파일 생성: apc_optimization/normalizer.py

```python
"""
제어 변수 정규화 통합 관리자

두 시스템이 동일한 정규화 기준을 사용하도록 관리:
- 예측 모델 (model_interface.py)
- 최적화 모델 (cost_function.py)
"""

import numpy as np
from typing import Tuple, Dict, Union
import logging

logger = logging.getLogger(__name__)


class ControlVariableNormalizer:
    """
    제어 변수(△GV, △RPM) 정규화 통합 관리자

    정규화 방식: MinMax (절댓값 기준)
    공식:
        - normalized_value = value / max_value
        - 범위: [0, 1]

    Parameters:
        gv_max (float): GV 정규화 기준값 (mm) - 기본값: 2.0
        rpm_max (float): RPM 정규화 기준값 - 기본값: 50

    Example:
        >>> normalizer = ControlVariableNormalizer(gv_max=2.0, rpm_max=50)
        >>> delta_gv = np.array([0.5, 1.0, 2.0])
        >>> delta_rpm = 25.0
        >>> gv_norm, rpm_norm = normalizer.normalize_control_vars(delta_gv, delta_rpm)
        >>> print(gv_norm)  # [0.25, 0.5, 1.0]
        >>> print(rpm_norm)  # 0.5
    """

    def __init__(self, gv_max: float = 2.0, rpm_max: float = 50.0):
        """
        초기화

        Args:
            gv_max: GV 정규화 기준값 (mm)
            rpm_max: RPM 정규화 기준값
        """
        self.gv_max = gv_max
        self.rpm_max = rpm_max

        # 입력 검증
        if gv_max <= 0 or rpm_max <= 0:
            raise ValueError(f"gv_max와 rpm_max는 양수여야 합니다. "
                           f"gv_max={gv_max}, rpm_max={rpm_max}")

        logger.info(f"ControlVariableNormalizer 초기화: "
                   f"gv_max={gv_max}, rpm_max={rpm_max}")

    # ====================================================================
    # 정규화 메서드
    # ====================================================================

    def normalize_control_vars(self,
                              delta_gv: np.ndarray,
                              delta_rpm: float) -> Tuple[np.ndarray, float]:
        """
        제어 변수 정규화 (MinMax: [0, 1])

        정규화 공식:
            gv_normalized = |delta_gv| / gv_max
            rpm_normalized = |delta_rpm| / rpm_max

        Args:
            delta_gv: Shape (n_gv,) - GV 변화량 (mm)
                     또는 Shape (n_samples, n_gv) - 배치 처리
            delta_rpm: Scalar - RPM 변화량
                     또는 Shape (n_samples,) - 배치 처리

        Returns:
            (gv_normalized, rpm_normalized)
            - gv_normalized: Shape와 동일하게 반환
            - rpm_normalized: Scalar 또는 배열

        Raises:
            ValueError: 입력값이 nan 또는 inf를 포함할 때

        Example:
            >>> delta_gv = np.array([0.5, 1.0, 2.0])
            >>> delta_rpm = 25.0
            >>> gv_norm, rpm_norm = normalizer.normalize_control_vars(delta_gv, delta_rpm)
            >>> gv_norm  # [0.25, 0.5, 1.0]
            >>> rpm_norm  # 0.5
        """
        # 입력 검증
        delta_gv = np.asarray(delta_gv)
        delta_rpm = np.asarray(delta_rpm)

        if np.any(np.isnan(delta_gv)) or np.any(np.isnan(delta_rpm)):
            raise ValueError("정규화 입력에 NaN이 포함되어 있습니다")

        if np.any(np.isinf(delta_gv)) or np.any(np.isinf(delta_rpm)):
            raise ValueError("정규화 입력에 Inf가 포함되어 있습니다")

        # 절댓값 기준 정규화
        gv_normalized = np.abs(delta_gv) / self.gv_max
        rpm_normalized = np.abs(delta_rpm) / self.rpm_max

        # 범위 클립 [0, 1]
        gv_normalized = np.clip(gv_normalized, 0.0, 1.0)
        rpm_normalized = np.clip(rpm_normalized, 0.0, 1.0)

        # 스칼라로 반환
        if isinstance(rpm_normalized, np.ndarray) and rpm_normalized.size == 1:
            rpm_normalized = float(rpm_normalized)

        return gv_normalized, rpm_normalized

    def denormalize_control_vars(self,
                                gv_normalized: np.ndarray,
                                rpm_normalized: Union[float, np.ndarray]
                                ) -> Tuple[np.ndarray, Union[float, np.ndarray]]:
        """
        역정규화 (정규화된 값 → 원본 값)

        역정규화 공식:
            delta_gv = gv_normalized * gv_max
            delta_rpm = rpm_normalized * rpm_max

        Args:
            gv_normalized: 정규화된 GV 값 [0, 1]
            rpm_normalized: 정규화된 RPM 값 [0, 1]

        Returns:
            (delta_gv, delta_rpm)

        Example:
            >>> gv_norm = np.array([0.25, 0.5, 1.0])
            >>> rpm_norm = 0.5
            >>> delta_gv, delta_rpm = normalizer.denormalize_control_vars(gv_norm, rpm_norm)
            >>> delta_gv  # [0.5, 1.0, 2.0]
            >>> delta_rpm  # 25.0
        """
        gv_normalized = np.asarray(gv_normalized)
        rpm_normalized = np.asarray(rpm_normalized)

        # 역정규화
        delta_gv = gv_normalized * self.gv_max
        delta_rpm = rpm_normalized * self.rpm_max

        # 스칼라로 반환
        if isinstance(delta_rpm, np.ndarray) and delta_rpm.size == 1:
            delta_rpm = float(delta_rpm)

        return delta_gv, delta_rpm

    # ====================================================================
    # 유틸리티 메서드
    # ====================================================================

    def get_config_dict(self) -> Dict[str, float]:
        """
        설정 사전 반환 (config.py의 CONTROL_COST_PARAMS와 동기화)

        Returns:
            dict: {'gv_max': float, 'rpm_max': float}

        Example:
            >>> config = normalizer.get_config_dict()
            >>> config['gv_max']  # 2.0
            >>> config['rpm_max']  # 50
        """
        return {
            'gv_max': self.gv_max,
            'rpm_max': self.rpm_max
        }

    def get_description(self) -> str:
        """정규화 설정 설명 반환"""
        return (f"ControlVariableNormalizer(gv_max={self.gv_max}, "
               f"rpm_max={self.rpm_max})\n"
               f"정규화 방식: MinMax (절댓값 기준)\n"
               f"범위: [0, 1]")


# ============================================================================
# 테스트 코드
# ============================================================================

if __name__ == '__main__':
    import logging
    logging.basicConfig(level=logging.INFO)

    # 정규화 테스트
    normalizer = ControlVariableNormalizer(gv_max=2.0, rpm_max=50)

    # 테스트 데이터
    delta_gv = np.array([0.5, 1.0, 2.0])
    delta_rpm = 25.0

    print(normalizer.get_description())

    # 정규화
    gv_norm, rpm_norm = normalizer.normalize_control_vars(delta_gv, delta_rpm)
    print(f"\n정규화:")
    print(f"  delta_gv: {delta_gv} → {gv_norm}")
    print(f"  delta_rpm: {delta_rpm} → {rpm_norm}")

    # 역정규화
    gv_back, rpm_back = normalizer.denormalize_control_vars(gv_norm, rpm_norm)
    print(f"\n역정규화:")
    print(f"  gv_norm: {gv_norm} → {gv_back}")
    print(f"  rpm_norm: {rpm_norm} → {rpm_back}")
```

#### 1.2 수정 사항: apc_optimization/__init__.py

```python
# 추가 import
from .normalizer import ControlVariableNormalizer

__all__ = [
    'ControlVariableNormalizer',
    # ... 기존 exports
]
```

---

### Phase 2: cost_function.py 수정 (2단계)

#### 2.1 수정 위치: cost_function.py (라인 148-190)

**Before:**
```python
def control_cost(self, delta_gv: np.ndarray, delta_rpm: float) -> Tuple[float, Dict]:
    gv_max = CONTROL_COST_PARAMS['gv_max']      # 2.0
    rpm_max = CONTROL_COST_PARAMS['rpm_max']    # 50
    beta = CONTROL_COST_PARAMS['beta']          # 0.7
    gamma = CONTROL_COST_PARAMS['gamma']        # 0.3

    # GV 정규화
    gv_normalized = (delta_gv / gv_max) ** 2
    gv_norm = np.mean(gv_normalized)

    # RPM 정규화
    rpm_norm = (delta_rpm / rpm_max) ** 2

    # 가중 합
    control_cost = beta * gv_norm + gamma * rpm_norm
    control_cost = np.clip(control_cost, 0.0, 1.0)

    details = {
        'gv_norm': gv_norm,
        'rpm_norm': rpm_norm,
        'gv_values': delta_gv,
        'rpm_value': delta_rpm,
        'gv_sum_abs': np.sum(np.abs(delta_gv)),
        'gv_max_abs': np.max(np.abs(delta_gv)),
    }

    return control_cost, details
```

**After:**
```python
def control_cost(self, delta_gv: np.ndarray, delta_rpm: float) -> Tuple[float, Dict]:
    # 통합 정규화 클래스 사용
    gv_normalized, rpm_normalized = self.normalizer.normalize_control_vars(
        delta_gv, delta_rpm
    )

    # 제어 비용 계산 (정규화된 값의 제곱)
    beta = CONTROL_COST_PARAMS['beta']
    gamma = CONTROL_COST_PARAMS['gamma']

    gv_norm = np.mean(gv_normalized ** 2)
    rpm_norm = rpm_normalized ** 2

    # 가중 합
    control_cost = beta * gv_norm + gamma * rpm_norm
    control_cost = np.clip(control_cost, 0.0, 1.0)

    details = {
        'gv_normalized': gv_normalized,
        'rpm_normalized': rpm_normalized,
        'gv_norm': gv_norm,
        'rpm_norm': rpm_norm,
        'gv_values': delta_gv,
        'rpm_value': delta_rpm,
        'gv_sum_abs': np.sum(np.abs(delta_gv)),
        'gv_max_abs': np.max(np.abs(delta_gv)),
    }

    return control_cost, details
```

#### 2.2 CostFunctionEvaluator.__init__ 수정

**Before:**
```python
def __init__(self,
             weights: Optional[Dict[str, float]] = None,
             ucl: float = CONTROL_LIMITS['ucl'],
             lcl: float = CONTROL_LIMITS['lcl']):
    self.weights = weights or COST_WEIGHTS
    self.ucl = ucl
    self.lcl = lcl

    total_weight = sum(self.weights.values())
    self.weights_normalized = {k: v/total_weight for k, v in self.weights.items()}

    logger.info(f"Cost Function Evaluator 초기화")
```

**After:**
```python
def __init__(self,
             weights: Optional[Dict[str, float]] = None,
             ucl: float = CONTROL_LIMITS['ucl'],
             lcl: float = CONTROL_LIMITS['lcl'],
             normalizer: Optional['ControlVariableNormalizer'] = None):
    self.weights = weights or COST_WEIGHTS
    self.ucl = ucl
    self.lcl = lcl

    # 통합 정규화 클래스 초기화
    if normalizer is None:
        from .normalizer import ControlVariableNormalizer
        self.normalizer = ControlVariableNormalizer(
            gv_max=CONTROL_COST_PARAMS['gv_max'],
            rpm_max=CONTROL_COST_PARAMS['rpm_max']
        )
    else:
        self.normalizer = normalizer

    total_weight = sum(self.weights.values())
    self.weights_normalized = {k: v/total_weight for k, v in self.weights.items()}

    logger.info(f"Cost Function Evaluator 초기화 (정규화: {self.normalizer.get_description()})")
```

#### 2.3 Import 추가

```python
# cost_function.py 최상단
from .normalizer import ControlVariableNormalizer
```

---

### Phase 3: model_interface.py 수정 (3단계) - 선택적

#### 3.1 상황 분석

현재 model_interface.py는 **pickle에 저장된 scaler** 사용:
```python
if self.scaler is not None:
    X = self.scaler.transform(X)
```

**고려사항**:
- 기존 학습된 모델은 StandardScaler로 정규화된 데이터로 학습됨
- 예측 시에도 동일한 StandardScaler 사용 필요 (모델 변경 없음)
- **최적화 모델에서만 ControlVariableNormalizer 사용**

#### 3.2 수정 방안 (선택 1: 최소 변경)

model_interface.py는 **변경하지 않음** (기존 StandardScaler 유지)

**이유**:
- 예측 모델은 이미 StandardScaler로 학습됨
- 현재 방식이 올바름 (학습 데이터와 동일한 정규화)
- 최적화 과정에서 ControlVariableNormalizer 사용

#### 3.3 수정 방안 (선택 2: 통일)

**장기적으로** 예측 모델도 ControlVariableNormalizer 사용:

```python
# model_interface.py - 향후 수정 (지금은 미실시)
class CatBoostModelManager:
    def __init__(self, ...):
        ...
        # 통합 정규화 사용 (미래 작업)
        # self.normalizer = ControlVariableNormalizer(...)
```

**현재 권장사항**: Phase 3는 **건너뛰고**, Phase 1-2만 실시

---

### Phase 4: 테스트 코드 작성 (4단계)

#### 4.1 파일 생성: tests/test_normalizer.py

```python
"""
ControlVariableNormalizer 테스트
"""

import numpy as np
import pytest
import sys
sys.path.insert(0, '/home/user/LLControl')

from apc_optimization.normalizer import ControlVariableNormalizer


class TestControlVariableNormalizer:
    """정규화 클래스 테스트"""

    @pytest.fixture
    def normalizer(self):
        """정규화기 인스턴스 반환"""
        return ControlVariableNormalizer(gv_max=2.0, rpm_max=50)

    # ====================================================================
    # 정규화 테스트
    # ====================================================================

    def test_normalize_basic(self, normalizer):
        """기본 정규화 테스트"""
        delta_gv = np.array([0.5, 1.0, 2.0])
        delta_rpm = 25.0

        gv_norm, rpm_norm = normalizer.normalize_control_vars(delta_gv, delta_rpm)

        np.testing.assert_array_almost_equal(gv_norm, np.array([0.25, 0.5, 1.0]))
        assert np.isclose(rpm_norm, 0.5)

    def test_normalize_negative_values(self, normalizer):
        """음수 값 정규화 (절댓값 사용)"""
        delta_gv = np.array([-0.5, -1.0, -2.0])
        delta_rpm = -25.0

        gv_norm, rpm_norm = normalizer.normalize_control_vars(delta_gv, delta_rpm)

        np.testing.assert_array_almost_equal(gv_norm, np.array([0.25, 0.5, 1.0]))
        assert np.isclose(rpm_norm, 0.5)

    def test_normalize_clipping(self, normalizer):
        """범위 클립 테스트"""
        delta_gv = np.array([5.0, 10.0])  # 범위 초과
        delta_rpm = 100.0  # 범위 초과

        gv_norm, rpm_norm = normalizer.normalize_control_vars(delta_gv, delta_rpm)

        assert np.all(gv_norm <= 1.0)
        assert np.all(gv_norm >= 0.0)
        assert rpm_norm <= 1.0
        assert rpm_norm >= 0.0

    def test_normalize_zero(self, normalizer):
        """0 값 정규화 테스트"""
        delta_gv = np.array([0.0, 0.0])
        delta_rpm = 0.0

        gv_norm, rpm_norm = normalizer.normalize_control_vars(delta_gv, delta_rpm)

        np.testing.assert_array_almost_equal(gv_norm, np.array([0.0, 0.0]))
        assert np.isclose(rpm_norm, 0.0)

    # ====================================================================
    # 역정규화 테스트
    # ====================================================================

    def test_denormalize_basic(self, normalizer):
        """기본 역정규화 테스트"""
        gv_norm = np.array([0.25, 0.5, 1.0])
        rpm_norm = 0.5

        delta_gv, delta_rpm = normalizer.denormalize_control_vars(gv_norm, rpm_norm)

        np.testing.assert_array_almost_equal(delta_gv, np.array([0.5, 1.0, 2.0]))
        assert np.isclose(delta_rpm, 25.0)

    def test_roundtrip_consistency(self, normalizer):
        """정규화 → 역정규화 일관성 테스트"""
        original_gv = np.array([0.3, 0.7, 1.5])
        original_rpm = 35.0

        # 정규화
        gv_norm, rpm_norm = normalizer.normalize_control_vars(original_gv, original_rpm)

        # 역정규화
        gv_back, rpm_back = normalizer.denormalize_control_vars(gv_norm, rpm_norm)

        # 원본과 동일해야 함
        np.testing.assert_array_almost_equal(gv_back, np.abs(original_gv))
        assert np.isclose(rpm_back, np.abs(original_rpm))

    # ====================================================================
    # 에러 처리 테스트
    # ====================================================================

    def test_invalid_initialization(self):
        """잘못된 초기화 테스트"""
        with pytest.raises(ValueError):
            ControlVariableNormalizer(gv_max=-1.0, rpm_max=50)

        with pytest.raises(ValueError):
            ControlVariableNormalizer(gv_max=2.0, rpm_max=0)

    def test_nan_input(self, normalizer):
        """NaN 입력 에러 처리"""
        delta_gv = np.array([0.5, np.nan, 1.0])
        delta_rpm = 25.0

        with pytest.raises(ValueError):
            normalizer.normalize_control_vars(delta_gv, delta_rpm)

    def test_inf_input(self, normalizer):
        """Inf 입력 에러 처리"""
        delta_gv = np.array([0.5, np.inf, 1.0])
        delta_rpm = 25.0

        with pytest.raises(ValueError):
            normalizer.normalize_control_vars(delta_gv, delta_rpm)

    # ====================================================================
    # 유틸리티 메서드 테스트
    # ====================================================================

    def test_get_config_dict(self, normalizer):
        """설정 사전 반환 테스트"""
        config = normalizer.get_config_dict()

        assert config['gv_max'] == 2.0
        assert config['rpm_max'] == 50
        assert isinstance(config, dict)


class TestCostFunctionWithNormalizer:
    """cost_function.py와 normalizer 통합 테스트"""

    def test_cost_function_with_normalizer(self):
        """cost_function이 normalizer를 올바르게 사용하는지 테스트"""
        from apc_optimization.cost_function import CostFunctionEvaluator

        evaluator = CostFunctionEvaluator()

        # 테스트 데이터
        delta_gv = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
        delta_rpm = 25.0

        control_cost, details = evaluator.control_cost(delta_gv, delta_rpm)

        # 결과 검증
        assert isinstance(control_cost, (float, np.floating))
        assert 0.0 <= control_cost <= 1.0
        assert 'gv_normalized' in details
        assert 'rpm_normalized' in details
        assert 'gv_norm' in details
        assert 'rpm_norm' in details
```

#### 4.2 테스트 실행

```bash
# pytest 설치 (필요시)
pip install pytest

# 테스트 실행
pytest tests/test_normalizer.py -v

# 또는 직접 실행
python -m pytest tests/test_normalizer.py -v
```

---

### Phase 5: 최적화 엔진 통합 확인 (5단계)

#### 5.1 영향받는 파일

```
optimizer_engine.py
    ↓
    CostFunctionEvaluator 사용
        ↓
        evaluate_total_cost()
            ↓
            control_cost() ← ControlVariableNormalizer 사용
```

#### 5.2 확인 사항

**optimizer_engine.py 수정 필요 여부**: NO
- optimizer_engine.py는 CostFunctionEvaluator를 통해서만 비용 함수 호출
- CostFunctionEvaluator 내부에서 normalizer 초기화
- 기존 API 유지 (변경 불필요)

---

## 4. 수정 파일 목록 및 변경 요약

| Phase | 파일명 | 작업 | 우선순위 |
|-------|--------|------|---------|
| 1 | `apc_optimization/normalizer.py` | 신규 생성 | 🔴 필수 |
| 1 | `apc_optimization/__init__.py` | import 추가 | 🔴 필수 |
| 2 | `apc_optimization/cost_function.py` | control_cost() 수정 | 🔴 필수 |
| 2 | `apc_optimization/cost_function.py` | __init__() 수정 | 🔴 필수 |
| 4 | `tests/test_normalizer.py` | 테스트 코드 작성 | 🟡 권장 |
| 3 | `apc_optimization/model_interface.py` | (건너뜀) | 🔵 선택 |

---

## 5. 예상 효과

### 5.1 개선 사항

| 항목 | 변경 전 | 변경 후 |
|------|--------|--------|
| **정규화 기준** | 불일치 (StandardScaler vs 고정값) | ✅ 일치 (ControlVariableNormalizer) |
| **코드 중복** | 있음 (gv_max, rpm_max 여러 곳) | ✅ 제거 (한곳에서 관리) |
| **유지보수성** | 낮음 | ✅ 높음 (단일 진실 공급원) |
| **확장성** | 낮음 | ✅ 높음 (새 정규화 방식 추가 용이) |
| **일관성** | 낮음 | ✅ 높음 (모든 시스템 동일 기준) |

### 5.2 성능 영향

- **런타임 오버헤드**: 무시할 수 있음 (단순 연산)
- **메모리 사용**: 변화 없음 (단순 클래스 인스턴스)
- **정확도**: 향상 가능 (일관된 정규화)

---

## 6. 구현 순서 및 타임라인

### 권장 순서

```
1단계: normalizer.py 작성 및 테스트
    ├─ ControlVariableNormalizer 클래스 구현
    ├─ 기본 정규화/역정규화 테스트
    └─ 에러 처리 확인

2단계: cost_function.py 수정
    ├─ control_cost() 메서드 수정
    ├─ __init__() 메서드 수정
    ├─ 기존 테스트 패스 확인
    └─ 비용 함수 출력값 검증

3단계: 통합 테스트
    ├─ optimizer_engine과의 연동 확인
    ├─ 최적화 결과 검증
    └─ 성능 회귀 테스트

4단계: 문서화 및 커밋
    ├─ 구현 완료 문서 작성
    ├─ API 문서 업데이트
    └─ Git 커밋 및 Push
```

---

## 7. 주의사항

### 7.1 주의할 점

1. **기존 모델과의 호환성**
   - 예측 모델은 StandardScaler로 학습됨
   - model_interface.py는 현재 건너뜀 (향후 재검토 필요)

2. **수치 안정성**
   - NaN, Inf 값 처리 (normalizer.py에 구현됨)
   - 클립핑 [0, 1] 범위 유지

3. **역호환성**
   - 기존 code_function.py API 유지
   - 기존 테스트 코드 패스 확인 필수

### 7.2 롤백 계획

수정 후 문제 발생 시:

```bash
# 최근 커밋 되돌리기
git revert <commit_hash>

# 또는 이전 버전으로 복구
git checkout <branch> -- apc_optimization/cost_function.py
```

---

## 8. 체크리스트

### 구현 전

- [ ] 현재 코드 백업 (git에서 자동 관리)
- [ ] 기존 테스트 케이스 확인
- [ ] 영향받는 모듈 파악

### Phase 1 완료 후

- [ ] normalizer.py 작성 완료
- [ ] 기본 테스트 통과
- [ ] __init__.py import 추가
- [ ] 로컬에서 import 확인

### Phase 2 완료 후

- [ ] cost_function.py 수정
- [ ] 기존 테스트 패스 확인
- [ ] 출력값 검증
- [ ] 통합 테스트 실시

### Phase 4 완료 후

- [ ] 전체 테스트 통과
- [ ] 통합 테스트 성공
- [ ] 문서화 완료

### 최종

- [ ] 코드 리뷰
- [ ] Git 커밋 및 Push
- [ ] PR 생성 (선택사항)

---

## 9. 참고 자료

### 관련 문서
- [CONTROL_COST_NORMALIZATION_ANALYSIS.md](./CONTROL_COST_NORMALIZATION_ANALYSIS.md)
- [config.py](./apc_optimization/config.py) - CONTROL_COST_PARAMS 정의
- [cost_function.py](./apc_optimization/cost_function.py) - 원본 구현

### 주요 파라미터
- `gv_max`: 2.0 (mm) - GV 정규화 기준
- `rpm_max`: 50 - RPM 정규화 기준
- `beta`: 0.7 - GV 가중치
- `gamma`: 0.3 - RPM 가중치

