# Phase 3 재검토: 정규화 불일치가 최적화에 미치는 영향 분석

## 1. 문제 재확인: 정규화 불일치의 실제 흐름

### 1.1 최적화 과정의 정규화 흐름

```
최적화 단계 (optimizer_engine.py)
    ↓
x = [△GV₁~₁₁, △RPM] 생성 (정수화됨)
    ↓
controller.evaluate_control(x, current_state)
    ├─ zone_inputs = construct_zone_inputs(x)
    │  └─ zone_inputs 구성: [위치특성, 현재CLR, 제어특성(△GV, △RPM)]
    │
    ├─ predicted_delta_clr = batch_predict(zone_inputs)
    │  └─ model.predict_batch(zone_inputs) 호출
    │     └─ scaler.transform(zone_inputs) ← ★ StandardScaler 적용!
    │
    └─ probabilities = inverse_clr(current_clr, predicted_delta_clr)

    반환: p_low, p_mid, p_high

이후:
    ↓
cost_evaluator.evaluate_total_cost(p_low, p_mid, p_high, delta_gv, delta_rpm)
    └─ control_cost(delta_gv, delta_rpm)
       └─ normalizer.normalize_control_vars() ← ★ MinMax 정규화 (Phase 2)!
```

### 1.2 정규화 불일치 발생 지점

```
zone_inputs에 포함된 제어값 (delta_gv, delta_rpm):
  ↓
  모델 예측 시: StandardScaler로 정규화
    공식: (value - μ_train) / σ_train
    범위: (-∞, +∞)
    기준: 학습 데이터의 통계

vs

비용 함수의 제어값 (delta_gv, delta_rpm):
  ↓
  비용 계산 시: MinMax (절댓값)로 정규화
    공식: |value| / max_value
    범위: [0, 1]
    기준: 공정 제약 조건
```

---

## 2. 영향 분석: 이 불일치가 문제인가?

### 2.1 YES - 잠재적 문제가 존재한다

#### 문제 1: 예측 모델과 비용 함수의 스케일 불일치

**시나리오**:
```
△GV = 1.0 mm라는 제어값이 있을 때:

1. 최적화 과정 (예측 모델 사용):
   zone_inputs에 △GV = 1.0 포함
   → model.predict_batch() 내부에서:
      △GV_normalized = (1.0 - μ) / σ
      (학습 데이터: μ=0.2, σ=0.5라고 가정)
      → △GV_normalized = (1.0 - 0.2) / 0.5 = 1.6
   → 모델이 1.6 스케일로 처리

2. 비용 함수 (ControlVariableNormalizer 사용):
   △GV = 1.0 입력
   → normalizer.normalize():
      △GV_normalized = 1.0 / 2.0 = 0.5
   → 비용 함수가 0.5 스케일로 처리

결과: 예측 모델과 비용 함수가 서로 다른 "의미"의 정규화된 값으로 작동
```

#### 문제 2: 최적화 수렴 성능 저하

- 예측 모델의 출력 (확률 분포)은 StandardScaler 정규화된 입력을 기반으로 함
- 최적화 알고리즘이 선택하는 제어값은 MinMax 정규화 기준으로 평가됨
- 두 스케일이 다르면, 최적화 알고리즘이 올바른 방향으로 수렴하지 못할 수 있음

#### 문제 3: 최적화 결과의 신뢰성 저하

- 최적화에서 선택된 제어값이 예측 모델과 비용 함수에서 일관되게 평가되지 않음
- 실제 공정에 적용할 때, 예상과 다른 결과 발생 가능성

### 2.2 영향도 평가

| 시나리오 | 심각도 | 설명 |
|--------|-------|------|
| **작은 제어값** (0.1-0.5) | 중간 | StandardScaler와 MinMax 간 편차 커짐 |
| **큰 제어값** (1.5-2.0) | 높음 | 스케일 불일치로 인한 오류 증가 |
| **경계값** (±2.0) | 매우 높음 | 최대한 다른 스케일로 변환됨 |

---

## 3. 현재 계획 (Phase 1-2만)의 문제점

### 3.1 What is actually happening now?

```
현재 코드 (Phase 1-2 후):

최적화:
  x = [△GV₁~₁₁, △RPM] 생성
    ↓
  controller.evaluate_control(x)
    ├─ zone_inputs 구성 (제어값 포함)
    ├─ model.predict_batch(zone_inputs)
    │  └─ StandardScaler.transform() ← 모델 내부 적용
    └─ 확률 분포 반환

  cost_evaluator.evaluate_total_cost(p_low, p_mid, p_high, delta_gv, delta_rpm)
    └─ control_cost()
       └─ normalizer.normalize_control_vars(delta_gv, delta_rpm)
          └─ MinMax 정규화 ← Phase 2에서 추가
```

**문제**:
- zone_inputs 내의 제어값: StandardScaler로 정규화
- 비용 함수의 제어값: MinMax로 정규화
- 두 정규화 기준이 다름!

---

## 4. Phase 3 재정의: 진정한 해결책

### 4.1 근본 원인

예측 모델이 이미 StandardScaler로 학습되었기 때문에, **zone_inputs에 포함된 제어값도 학습 데이터와 동일한 스케일로 정규화되어야 함**.

### 4.2 3가지 해결 방안

#### 방안 A: 최적화에서도 StandardScaler 사용 (권장)

```python
# optimizer_engine.py 수정
class DifferentialEvolutionOptimizer:
    def __init__(self, model_manager, cost_evaluator, ...):
        ...
        # ★ StandardScaler 로드 (모델에서 추출)
        self.input_scaler = model_manager.scaler
        self.cost_normalizer = ControlVariableNormalizer()

    def objective_function(self, x):
        x_transformed = self.output_transformer.transform(x)

        # ★ Zone 입력 구성 시 StandardScaler 적용
        zone_inputs = self.controller.construct_zone_inputs(
            x_transformed, self.current_state
        )
        # zone_inputs 내의 제어값이 이미 StandardScaler로 정규화됨
        # (construct_zone_inputs 내부에서 또는 외부에서)

        # 1. 예측 (StandardScaler 정규화된 입력)
        control_result = self.controller.evaluate_control(
            x_transformed, self.current_state
        )

        # 2. 비용 계산
        # ★ 비용 함수에서 ControlVariableNormalizer 사용 (Phase 2)
        cost = self.cost_evaluator.evaluate_total_cost(
            p_low, p_mid, p_high, x_transformed[:N_GV], x_transformed[N_GV]
        )
```

**장점**:
- ✅ 예측 모델과의 완벽한 호환성
- ✅ 명확한 정규화 기준

**문제점**:
- ❌ 최적화 과정에서 공정 제약 조건(gv_max, rpm_max)과의 연결성 저하
- ❌ 비용 함수와 최적화 제약 조건이 다른 스케일 사용

---

#### 방안 B: construct_zone_inputs에서 명시적 정규화

```python
# multi_zone_controller.py 수정
class MultiZoneController:
    def construct_zone_inputs(self, x, current_state,
                             normalizer=None):  # ★ normalizer 추가
        """
        제어값 정규화를 명시적으로 처리
        """
        delta_gv = x[:N_GV]
        delta_rpm = x[N_GV]

        # ★ 제어값 정규화
        if normalizer is not None:
            # ControlVariableNormalizer 사용 (절댓값 기준)
            delta_gv_norm, delta_rpm_norm = normalizer.normalize_control_vars(
                delta_gv, delta_rpm
            )
            # 이 정규화된 값을 zone_inputs에 포함
        else:
            # StandardScaler는 모델 내부에서 자동 적용
            delta_gv_norm = delta_gv
            delta_rpm_norm = delta_rpm

        # Zone 입력 구성... (정규화된 제어값 사용)
```

**장점**:
- ✅ 명시적 제어 가능
- ✅ 유연한 정규화 방식 선택

**문제점**:
- ❌ 추가 복잡도
- ❌ 모델 입력과의 정규화 기준 여전히 불일치

---

#### 방안 C: 통합 정규화 클래스 확장 (최고 권장)

**새로운 ControlVariableNormalizer 기능**:

```python
class ControlVariableNormalizer:
    def __init__(self, gv_max=2.0, rpm_max=50.0, scaler=None):
        self.gv_max = gv_max
        self.rpm_max = rpm_max
        self.scaler = scaler  # ★ StandardScaler (선택사항)

    def normalize_for_prediction(self, delta_gv, delta_rpm):
        """
        예측 모델용 정규화 (StandardScaler)
        """
        if self.scaler is not None:
            X = np.array([[delta_gv, delta_rpm]])
            X_scaled = self.scaler.transform(X)
            return X_scaled[0, 0], X_scaled[0, 1]
        else:
            # Fallback: 절댓값 기준
            return np.abs(delta_gv) / self.gv_max, np.abs(delta_rpm) / self.rpm_max

    def normalize_for_cost(self, delta_gv, delta_rpm):
        """
        비용 함수용 정규화 (MinMax)
        """
        return np.abs(delta_gv) / self.gv_max, np.abs(delta_rpm) / self.rpm_max
```

**사용**:

```python
# Phase 2 (cost_function.py)
class CostFunctionEvaluator:
    def control_cost(self, delta_gv, delta_rpm):
        gv_norm, rpm_norm = self.normalizer.normalize_for_cost(
            delta_gv, delta_rpm
        )
        # ... 비용 계산
```

```python
# Phase 3 (optimizer_engine.py)
class DifferentialEvolutionOptimizer:
    def objective_function(self, x):
        # 예측용 정규화
        delta_gv_pred, delta_rpm_pred = self.normalizer.normalize_for_prediction(
            x[:N_GV], x[N_GV]
        )
        # zone_inputs에 정규화된 제어값 포함
```

**장점**:
- ✅ 단일 클래스에서 모든 정규화 관리
- ✅ 예측과 비용 함수 모두 일관성 있게 처리
- ✅ 향후 확장 용이

---

## 5. 수정된 구현 계획

### 새로운 Phase 3: 정규화 일관성 보장

#### Phase 3A: ControlVariableNormalizer 확장 (1시간)

```python
# apc_optimization/normalizer.py 수정
class ControlVariableNormalizer:
    def __init__(self, gv_max=2.0, rpm_max=50.0, scaler=None):
        ...
        self.scaler = scaler  # ★ StandardScaler 추가

    # 기존 메서드
    def normalize_control_vars(self, delta_gv, delta_rpm):
        # MinMax 정규화 (기존)
        ...

    # 새로운 메서드
    def normalize_for_prediction(self, delta_gv, delta_rpm):
        # StandardScaler 정규화 (예측 모델용)
        if self.scaler:
            ...

    def normalize_for_cost(self, delta_gv, delta_rpm):
        # MinMax 정규화 (비용 함수용)
        ...
```

#### Phase 3B: optimizer_engine.py 수정 (30분)

- construct_zone_inputs에서 제어값 정규화 명시
- 예측 모델과 비용 함수 간 일관성 보장

#### Phase 3C: 통합 테스트 (30분)

- 예측 모델과 비용 함수의 정규화 일관성 검증
- 최적화 결과 검증

---

## 6. 최종 권장사항

### ✅ Phase 3는 반드시 포함되어야 한다

**근거**:
1. 현재 계획 (Phase 1-2만): 정규화 불일치로 인해 최적화 결과 신뢰성 저하
2. 방안 C (ControlVariableNormalizer 확장): 최선의 해결책
3. 추가 소요 시간: ~2시간 (Phase 3A + 3B + 3C)

### 수정된 총 구현 일정

| Phase | 작업 | 소요시간 |
|-------|------|---------|
| 1 | normalizer.py 생성 (기본) | 1-2h |
| 2 | cost_function.py 수정 | 30m |
| **3** | **optimizer_engine.py 수정 + 정규화 일관성** | **2h** |
| 4 | 테스트 코드 | 1h |
| 5 | 통합 검증 | 30m |
| | **총** | **5-6h** |

---

## 7. 결론

사용자의 지적이 정확합니다.

**현재 계획의 문제**:
- ❌ Phase 3을 건너뛰면, 예측 모델과 비용 함수의 정규화 불일치 발생
- ❌ 이는 최적화 수렴 성능 저하 및 결과 신뢰성 감소로 이어짐

**수정된 계획**:
- ✅ Phase 3 포함 (최고 권장: 방안 C)
- ✅ ControlVariableNormalizer에 양방향 정규화 기능 추가
- ✅ optimizer_engine.py와 cost_function.py 모두 동일한 정규화 기준 사용
- ✅ 예측 모델과의 완벽한 호환성 보장

