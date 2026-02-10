# 통합 정규화 구현 빠른 시작 가이드

> **전체 계획**: [UNIFIED_NORMALIZATION_IMPLEMENTATION_PLAN.md](./UNIFIED_NORMALIZATION_IMPLEMENTATION_PLAN.md) 참고

## Phase 1: 통합 정규화 클래스 생성 ✅ 준비 완료

### 단계 1.1: normalizer.py 파일 생성

**파일**: `apc_optimization/normalizer.py`

**내용**: [계획 문서의 Phase 1 참고](#)

**핵심 클래스**:
```python
class ControlVariableNormalizer:
    def __init__(self, gv_max=2.0, rpm_max=50.0)
    def normalize_control_vars(self, delta_gv, delta_rpm)
    def denormalize_control_vars(self, gv_normalized, rpm_normalized)
    def get_config_dict(self)
    def get_description(self)
```

**작업 명령어**:
```bash
# 1. 파일 생성 (계획 문서 복사)
# apc_optimization/normalizer.py 생성

# 2. 기본 테스트 (인터프리터에서)
python3
>>> from apc_optimization.normalizer import ControlVariableNormalizer
>>> normalizer = ControlVariableNormalizer()
>>> import numpy as np
>>> gv_norm, rpm_norm = normalizer.normalize_control_vars(np.array([0.5, 1.0]), 25.0)
>>> print(gv_norm, rpm_norm)  # [0.25, 0.5] 0.5
```

---

## Phase 2: cost_function.py 수정 ✅ 준비 완료

### 단계 2.1: normalizer import 추가

**파일**: `apc_optimization/cost_function.py` (라인 12-15)

**수정 전**:
```python
from .config import (
    COST_WEIGHTS, QUALITY_COST_PARAMS, BALANCE_COST_PARAMS,
    CONTROL_COST_PARAMS, SAFETY_COST_PARAMS, CONTROL_LIMITS,
    N_ZONES, GV_ADJACENT_MAX_DIFF, GV_TOTAL_CHANGE_MAX
)
```

**수정 후**:
```python
from .config import (
    COST_WEIGHTS, QUALITY_COST_PARAMS, BALANCE_COST_PARAMS,
    CONTROL_COST_PARAMS, SAFETY_COST_PARAMS, CONTROL_LIMITS,
    N_ZONES, GV_ADJACENT_MAX_DIFF, GV_TOTAL_CHANGE_MAX
)
from .normalizer import ControlVariableNormalizer
```

### 단계 2.2: CostFunctionEvaluator.__init__ 수정

**파일**: `apc_optimization/cost_function.py` (라인 29-51)

**수정 전**:
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

**수정 후**:
```python
def __init__(self,
             weights: Optional[Dict[str, float]] = None,
             ucl: float = CONTROL_LIMITS['ucl'],
             lcl: float = CONTROL_LIMITS['lcl'],
             normalizer: Optional[ControlVariableNormalizer] = None):
    self.weights = weights or COST_WEIGHTS
    self.ucl = ucl
    self.lcl = lcl

    # 통합 정규화 클래스 초기화
    if normalizer is None:
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

### 단계 2.3: control_cost() 메서드 수정

**파일**: `apc_optimization/cost_function.py` (라인 148-190)

**수정 전**:
```python
def control_cost(self, delta_gv: np.ndarray, delta_rpm: float) -> Tuple[float, Dict]:
    gv_max = CONTROL_COST_PARAMS['gv_max']
    rpm_max = CONTROL_COST_PARAMS['rpm_max']
    beta = CONTROL_COST_PARAMS['beta']
    gamma = CONTROL_COST_PARAMS['gamma']

    gv_normalized = (delta_gv / gv_max) ** 2
    gv_norm = np.mean(gv_normalized)

    rpm_norm = (delta_rpm / rpm_max) ** 2

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

**수정 후**:
```python
def control_cost(self, delta_gv: np.ndarray, delta_rpm: float) -> Tuple[float, Dict]:
    # 통합 정규화 클래스 사용
    gv_normalized, rpm_normalized = self.normalizer.normalize_control_vars(
        delta_gv, delta_rpm
    )

    # 제어 비용 계산
    beta = CONTROL_COST_PARAMS['beta']
    gamma = CONTROL_COST_PARAMS['gamma']

    gv_norm = np.mean(gv_normalized ** 2)
    rpm_norm = rpm_normalized ** 2

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

### 단계 2.4: 수정 후 테스트

```bash
# cost_function.py 테스트
python3
>>> from apc_optimization.cost_function import CostFunctionEvaluator
>>> evaluator = CostFunctionEvaluator()
>>> import numpy as np
>>> delta_gv = np.array([0.5] * 11)
>>> delta_rpm = 25.0
>>> cost, details = evaluator.control_cost(delta_gv, delta_rpm)
>>> print(f"Control Cost: {cost:.4f}")
>>> print(f"GV Normalized: {details['gv_normalized']}")
>>> print(f"RPM Normalized: {details['rpm_normalized']}")
```

---

## Phase 3: (건너뜀) model_interface.py

**현재**: 변경하지 않음 (기존 StandardScaler 유지)

**이유**: 예측 모델은 StandardScaler로 학습됨 - 호환성 유지 필요

**향후 검토**: 예측 모델을 ControlVariableNormalizer로 재학습할 시점에 수정

---

## Phase 4: 테스트 코드 작성 ✅ 준비 완료

### 단계 4.1: tests/test_normalizer.py 생성

**파일**: `tests/test_normalizer.py`

**내용**: [계획 문서의 Phase 4 참고](#)

**테스트 실행**:
```bash
# pytest 설치 (필요시)
pip install pytest

# 테스트 실행
cd /home/user/LLControl
pytest tests/test_normalizer.py -v

# 또는 직접 실행
python -m pytest tests/test_normalizer.py -v
```

**예상 출력**:
```
tests/test_normalizer.py::TestControlVariableNormalizer::test_normalize_basic PASSED
tests/test_normalizer.py::TestControlVariableNormalizer::test_normalize_negative_values PASSED
tests/test_normalizer.py::TestControlVariableNormalizer::test_normalize_clipping PASSED
tests/test_normalizer.py::TestControlVariableNormalizer::test_normalize_zero PASSED
tests/test_normalizer.py::TestControlVariableNormalizer::test_denormalize_basic PASSED
tests/test_normalizer.py::TestControlVariableNormalizer::test_roundtrip_consistency PASSED
tests/test_normalizer.py::TestControlVariableNormalizer::test_invalid_initialization PASSED
tests/test_normalizer.py::TestControlVariableNormalizer::test_nan_input PASSED
tests/test_normalizer.py::TestControlVariableNormalizer::test_inf_input PASSED
tests/test_normalizer.py::TestControlVariableNormalizer::test_get_config_dict PASSED
tests/test_normalizer.py::TestCostFunctionWithNormalizer::test_cost_function_with_normalizer PASSED

======================== 11 passed in 0.15s ========================
```

---

## Phase 5: 최적화 엔진 통합 확인

### 단계 5.1: 기존 테스트 실행

```bash
# 기존 테스트 실행 (패스 확인)
cd /home/user/LLControl
python apc_optimization_test.py

# 또는 최적화 전체 테스트
python apc_optimization_full_test.py
```

### 단계 5.2: 검증 사항

- [ ] 기존 테스트 모두 패스
- [ ] 최적화 결과 생성 확인
- [ ] 비용 함수 값 합리적인지 확인
- [ ] 에러/경고 메시지 없음

---

## 체크리스트

### Before Starting

- [ ] 현재 branch 확인: `claude/explain-control-cost-params-mNbLz`
- [ ] Working tree clean 확인: `git status`
- [ ] 기존 테스트 패스 확인

### Phase 1 완료

- [ ] `apc_optimization/normalizer.py` 파일 생성
- [ ] 클래스 구현 완료
- [ ] 기본 동작 테스트 통과
- [ ] Import 가능 확인

### Phase 2 완료

- [ ] `cost_function.py` import 추가
- [ ] `__init__()` 메서드 수정
- [ ] `control_cost()` 메서드 수정
- [ ] 기존 테스트 패스 확인

### Phase 4 완료

- [ ] `tests/test_normalizer.py` 생성
- [ ] 전체 테스트 패스

### Phase 5 완료

- [ ] 기존 최적화 테스트 패스
- [ ] 최적화 결과 검증
- [ ] 회귀 테스트 통과

### 최종

- [ ] 코드 리뷰 완료
- [ ] Git 커밋 메시지 작성
- [ ] Push 완료
- [ ] 기타 branch와 충돌 없음 확인

---

## 핵심 코드 스니펫

### normalizer 사용 예시

```python
from apc_optimization.normalizer import ControlVariableNormalizer
import numpy as np

# 초기화
normalizer = ControlVariableNormalizer(gv_max=2.0, rpm_max=50)

# 정규화
delta_gv = np.array([0.5, 1.0, 1.5])
delta_rpm = 25.0
gv_norm, rpm_norm = normalizer.normalize_control_vars(delta_gv, delta_rpm)

# 역정규화
delta_gv_back, delta_rpm_back = normalizer.denormalize_control_vars(gv_norm, rpm_norm)

# 설정 확인
config = normalizer.get_config_dict()
```

### cost_function 수정 확인

```python
from apc_optimization.cost_function import CostFunctionEvaluator

evaluator = CostFunctionEvaluator()

# normalizer 확인
print(evaluator.normalizer.get_description())

# 제어 비용 계산 (normalizer 자동 사용)
control_cost, details = evaluator.control_cost(delta_gv, delta_rpm)
print(details['gv_normalized'])  # 새로 추가된 필드
```

---

## 문제 해결

### 문제 1: Import 에러

```
ModuleNotFoundError: No module named 'apc_optimization.normalizer'
```

**해결**:
1. `apc_optimization/normalizer.py` 파일이 있는지 확인
2. `apc_optimization/__init__.py`에 import 추가 확인
3. PYTHONPATH 확인

### 문제 2: 테스트 실패

```bash
FAILED tests/test_normalizer.py::test_normalize_basic
```

**해결**:
1. normalizer.py 구현 재확인
2. 테스트 데이터 확인
3. 계산 로직 재검토

### 문제 3: 기존 테스트 실패

```
FAILED apc_optimization_test.py
```

**해결**:
1. cost_function.py 수정 재확인
2. 수정 전 테스트와 비교
3. 롤백 후 재시도

---

## 다음 단계

1. ✅ Phase 1-5 완료 후
2. 📝 구현 완료 보고서 작성
3. 📊 성능 비교 분석 (수정 전/후)
4. 🔄 코드 리뷰 및 피드백 반영
5. 🎯 최적화 모델 재검증

---

## 참고 문서

- [통합 정규화 구현 계획 (상세)](./UNIFIED_NORMALIZATION_IMPLEMENTATION_PLAN.md)
- [CONTROL_COST_PARAMS 분석](./CONTROL_COST_NORMALIZATION_ANALYSIS.md)
- [config.py](./apc_optimization/config.py) - 설정값
- [cost_function.py](./apc_optimization/cost_function.py) - 원본 구현

