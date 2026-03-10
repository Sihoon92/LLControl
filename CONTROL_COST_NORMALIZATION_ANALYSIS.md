# CONTROL_COST_PARAMS 정규화 일관성 분석

## 1. 문제 개요

**핵심 이슈**: 예측 모델과 최적화 모델에서 `delta_gv`와 `delta_rpm` 값의 정규화 기준이 **불일치**합니다.

- **예측 모델** (model_interface.py): StandardScaler 사용 (평균=0, 표준편차=1 기준)
- **최적화 모델** (cost_function.py): 고정값 기준 (gv_max=2.0, rpm_max=50)

이로 인해 **두 시스템이 서로 다른 스케일의 입력값을 기준**으로 동작하게 됩니다.

---

## 2. 코드 분석

### 2.1 예측 모델의 정규화 (model_interface.py)

```python
# model_interface.py:136-137
def predict(self, X: np.ndarray) -> np.ndarray:
    X = np.atleast_2d(X)

    # 스케일링
    if self.scaler is not None:
        X = self.scaler.transform(X)  # ← StandardScaler (또는 다른 sklearn scaler)

    prediction = self.model.predict(X)
    return prediction[0] if prediction.shape[0] == 1 else prediction
```

**특징**:
- Pickle에서 로드된 scaler 사용
- 일반적으로 **StandardScaler** (평균 제거, 표준편차로 정규화)
- **데이터 의존적**: 학습 데이터의 통계에 따라 결정됨

### 2.2 최적화 모델의 정규화 (cost_function.py)

```python
# cost_function.py:148-190
def control_cost(self, delta_gv: np.ndarray, delta_rpm: float) -> Tuple[float, Dict]:
    gv_max = CONTROL_COST_PARAMS['gv_max']      # 2.0 mm
    rpm_max = CONTROL_COST_PARAMS['rpm_max']    # 50 rpm
    beta = CONTROL_COST_PARAMS['beta']          # 0.7
    gamma = CONTROL_COST_PARAMS['gamma']        # 0.3

    # GV 정규화: (value / 2.0)²
    gv_normalized = (delta_gv / gv_max) ** 2
    gv_norm = np.mean(gv_normalized)

    # RPM 정규화: (value / 50)²
    rpm_norm = (delta_rpm / rpm_max) ** 2

    # 가중 합
    control_cost = beta * gv_norm + gamma * rpm_norm
    return control_cost, details
```

**특징**:
- **절댓값 기준** (MinMax 스타일, 정규화 범위: [0, 1])
- 범위: (value / max) ∈ [0, 1]
- **고정값**: 공정 제약 조건에 기반

### 2.3 CONTROL_COST_PARAMS 정의 (config.py:98-103)

```python
CONTROL_COST_PARAMS = {
    'gv_max': 2.0,     # GV 정규화 기준값 (mm)
    'rpm_max': 50,     # RPM 정규화 기준값
    'beta': 0.7,       # GV 가중치
    'gamma': 0.3,      # RPM 가중치
}
```

---

## 3. 정규화 방식 비교

| 항목 | 예측 모델 | 최적화 모델 |
|------|---------|-----------|
| **정규화 방식** | StandardScaler | MinMax (절댓값 기준) |
| **정규화 공식** | (x - μ) / σ | x / max |
| **범위** | (-∞, +∞) | [0, 1] |
| **기준** | 데이터 통계 | 공정 제약 |
| **학습 의존성** | 있음 | 없음 |
| **재현성** | 모델마다 다름 | 일관성 있음 |

---

## 4. 불일치의 영향

### 4.1 문제 시나리오

**시나리오 1**: delta_gv = 0.5 mm
- 예측 모델 스케일: (0.5 - μ_train) / σ_train = ?
- 최적화 모델 스케일: 0.5 / 2.0 = 0.25

→ **다른 스케일의 입력값**이 모델에 전달됨

### 4.2 영향

1. **모델 예측 정확도 저하**:
   - 예측 모델은 StandardScaler로 학습됨
   - 하지만 최적화 모델에서 정규화된 값이 아닌 원본 또는 다르게 정규화된 값이 입력될 수 있음

2. **비용 함수 계산 오류**:
   - control_cost()에서 사용하는 값의 스케일이 예측 모델과 다름
   - 최적화 결과의 신뢰성 감소

3. **최적화 수렴 문제**:
   - 다른 스케일의 입력으로 인해 최적화 알고리즘의 수렴 성능 저하

---

## 5. 개선 방안

### 옵션 A: 통합 정규화 클래스 생성 (권장)

**전략**: 예측 모델과 최적화 모델이 **동일한 scaler 인스턴스** 공유

```python
# apc_optimization/normalizer.py (신규 파일)
class ControlVariableNormalizer:
    """제어 변수 정규화 통합 관리자"""

    def __init__(self, use_fixed_bounds=True):
        """
        Args:
            use_fixed_bounds: True면 고정값 사용, False면 StandardScaler
        """
        self.use_fixed_bounds = use_fixed_bounds
        self.scaler = None  # StandardScaler (use_fixed_bounds=False일 때)
        self.gv_max = 2.0
        self.rpm_max = 50

    def normalize(self, delta_gv, delta_rpm):
        """정규화"""
        if self.use_fixed_bounds:
            gv_norm = (delta_gv / self.gv_max) ** 2
            rpm_norm = (delta_rpm / self.rpm_max) ** 2
        else:
            # StandardScaler 사용
            ...
        return gv_norm, rpm_norm

    def denormalize(self, gv_norm, rpm_norm):
        """역정규화"""
        ...
```

**장점**:
- ✅ 단일 진실 공급원 (Single Source of Truth)
- ✅ 예측 모델과 최적화 모델 간 일관성 보장
- ✅ 유지보수 용이

### 옵션 B: StandardScaler를 CONTROL_COST_PARAMS 계산에 사용

**전략**: cost_function.py에서 model의 scaler를 재사용

```python
def control_cost(self, delta_gv, delta_rpm, scaler=None):
    if scaler is not None:
        # 모델의 scaler 사용
        X = np.array([[delta_gv, delta_rpm]])
        X_scaled = scaler.transform(X)
        gv_norm = X_scaled[0, 0] ** 2
        rpm_norm = X_scaled[0, 1] ** 2
    else:
        # config의 고정값 사용
        ...
```

**장점**:
- ✅ 최소 변경
- ❌ 모델 스케일러 의존성 증가

### 옵션 C: MinMax 정규화를 예측 모델에도 적용

**전략**: 모델 학습 시 StandardScaler 대신 MinMax 정규화 사용

**장점**:
- ✅ 공정 제약 조건과 일치
- ❌ 기존 모델 재학습 필요

---

## 6. 권장 해결 방안

### 즉시 조치

**선택**: **옵션 A (통합 정규화 클래스)** 권장

1. `ControlVariableNormalizer` 클래스 생성 (apc_optimization/normalizer.py)
2. model_interface.py에서 사용
3. cost_function.py에서 사용
4. 두 시스템 간 정규화 기준 통일

### 코드 예시

```python
# apc_optimization/normalizer.py
from typing import Tuple
import numpy as np

class ControlVariableNormalizer:
    """제어 변수(△GV, △RPM) 정규화 통합 관리자"""

    def __init__(self, gv_max: float = 2.0, rpm_max: float = 50.0):
        self.gv_max = gv_max
        self.rpm_max = rpm_max

    def normalize_control_vars(self, delta_gv: np.ndarray,
                               delta_rpm: float) -> Tuple[np.ndarray, float]:
        """
        제어 변수 정규화 (MinMax: [0, 1])

        Args:
            delta_gv: Shape (n_gv,) - GV 변화량 (mm)
            delta_rpm: Scalar - RPM 변화량

        Returns:
            (gv_normalized, rpm_normalized)
        """
        gv_normalized = delta_gv / self.gv_max
        rpm_normalized = delta_rpm / self.rpm_max
        return gv_normalized, rpm_normalized

    def denormalize_control_vars(self, gv_normalized: np.ndarray,
                                 rpm_normalized: float) -> Tuple[np.ndarray, float]:
        """역정규화"""
        gv = gv_normalized * self.gv_max
        rpm = rpm_normalized * self.rpm_max
        return gv, rpm

    def get_config_dict(self) -> dict:
        """설정 사전 반환"""
        return {
            'gv_max': self.gv_max,
            'rpm_max': self.rpm_max
        }
```

```python
# cost_function.py에서 사용
from apc_optimization.normalizer import ControlVariableNormalizer

class CostFunctionEvaluator:
    def __init__(self, ...):
        self.normalizer = ControlVariableNormalizer(
            gv_max=CONTROL_COST_PARAMS['gv_max'],
            rpm_max=CONTROL_COST_PARAMS['rpm_max']
        )

    def control_cost(self, delta_gv, delta_rpm):
        gv_norm, rpm_norm = self.normalizer.normalize_control_vars(delta_gv, delta_rpm)
        gv_norm_sq = gv_norm ** 2
        rpm_norm_sq = rpm_norm ** 2
        control_cost = CONTROL_COST_PARAMS['beta'] * np.mean(gv_norm_sq) + \
                       CONTROL_COST_PARAMS['gamma'] * rpm_norm_sq
        return control_cost, {...}
```

---

## 7. 검증 계획

### 검증 테스트

```python
def test_normalization_consistency():
    """정규화 일관성 테스트"""
    normalizer = ControlVariableNormalizer(gv_max=2.0, rpm_max=50)

    # 테스트 케이스
    delta_gv = np.array([0.5, 1.0, 2.0])
    delta_rpm = 25.0

    # 정규화
    gv_norm, rpm_norm = normalizer.normalize_control_vars(delta_gv, delta_rpm)

    # 예상값
    assert np.allclose(gv_norm, np.array([0.25, 0.5, 1.0]))
    assert np.isclose(rpm_norm, 0.5)

    # 역정규화
    gv_back, rpm_back = normalizer.denormalize_control_vars(gv_norm, rpm_norm)
    assert np.allclose(gv_back, delta_gv)
    assert np.isclose(rpm_back, delta_rpm)

    print("✓ 정규화 일관성 테스트 통과")
```

---

## 8. 결론

**현재 상태**:
- 예측 모델: StandardScaler 사용 (데이터 통계 기준)
- 최적화 모델: MinMax 스타일 고정값 사용 (공정 제약 기준)
- **→ 불일치 발생**

**권장 조치**:
1. 통합 정규화 클래스 (`ControlVariableNormalizer`) 생성
2. 예측 모델과 최적화 모델에서 동일한 정규화 기준 적용
3. 테스트 코드로 일관성 검증

**예상 효과**:
- ✅ 모델 예측 정확도 향상
- ✅ 최적화 수렴 성능 개선
- ✅ 시스템 신뢰성 증가
- ✅ 유지보수 용이성 개선
