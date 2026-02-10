# 최종 정규화 분석: 현재 구현의 한계와 완전한 해결책

## 1. 현재 구현의 문제점 재확인

### 1.1 제어값이 두 가지 다른 스케일로 사용되는 상황

```
제어값 x = [△GV₁~₁₁, △RPM]

1️⃣ 예측 모델 경로 (optimizer_engine.py → multi_zone_controller.py)
   x (원본)
   ↓
   construct_zone_inputs(x)
   ├─ zone_input = [..., delta_gv[...], delta_rpm, ...]
   └─ zone_inputs에 x의 제어값을 그대로 포함
   ↓
   model.predict_batch(zone_inputs)
   ├─ scaler.transform(zone_inputs)  ← ★ StandardScaler 적용!
   │  (학습 데이터: μ, σ 기준)
   └─ 예측 결과 반환

2️⃣ 비용 함수 경로 (cost_function.py)
   x (원본)
   ↓
   control_cost(△GV, △RPM)
   └─ normalizer.normalize_for_cost(△GV, △RPM)  ← ★ MinMax 정규화!
      (절댓값 기준: gv_max=2.0, rpm_max=50)

결과:
  ✗ 동일한 제어값 x가 두 가지 다른 정규화 스케일로 사용됨!
```

### 1.2 이것이 문제인가?

**다양한 관점에서의 분석:**

#### 관점 1: 의도적 설계인가?
```
가능한 해석:
- 예측 모델: 모델이 학습한 방식대로 StandardScaler 정규화
- 비용 함수: 공정 제약 기준의 MinMax 정규화
- 이 두 가지는 별개의 목적이므로 다를 수 있다?
```

#### 관점 2: 실제 영향은?
```
최적화 알고리즘 관점:
- DE(Differential Evolution) 알고리즘은 목적함수의 값을 최소화
- 목적함수 = 비용 함수 (MinMax 스케일)
- 예측 모델 입력 (StandardScaler 스케일)은 중간 계산값

문제점:
- 최적화가 선택한 제어값 x
  → 예측 모델에는 StandardScaler로 들어감
  → 비용 함수에는 MinMax로 들어감
  → 최적화 품질에 영향 가능성
```

#### 관점 3: 코드 인스턴스
```
model.predict_batch() 내부:
  X_scaled = self.scaler.transform(X)  # 자동 적용

cost_function.control_cost() 내부:
  gv_norm, rpm_norm = self.normalizer.normalize_for_cost(...)  # 명시적 적용

→ 한 곳은 자동, 한 곳은 명시적으로 다르게 정규화됨!
```

---

## 2. 진정한 문제: 불명확한 정규화 기준

### 2.1 현재의 모호함

```
question: zone_inputs의 제어값이 어떤 스케일로 예측 모델에 들어가는가?

답변 옵션:
A) 원본 스케일 (그 후 StandardScaler 자동 적용)
B) 비용 함수와 동일한 MinMax 스케일
C) 의도적으로 다른 스케일?

현재 코드: A (원본 스케일 후 자동 StandardScaler 적용)
하지만: B와 C의 가능성도 모호함
```

### 2.2 이것이 최적화에 미치는 영향

```
시나리오: 최적화 알고리즘이 x = [△GV=1.0, △RPM=30]을 선택했을 때

1. 예측 모델 계산:
   zone_inputs에 △GV=1.0, △RPM=30 포함
   → model.predict_batch()
   → StandardScaler.transform()
   → (1.0 - μ_gv) / σ_gv = ?
   → (30 - μ_rpm) / σ_rpm = ?
   (실제 값은 학습 데이터에 따라 다름)

2. 비용 함수 계산:
   control_cost(△GV=1.0, △RPM=30)
   → normalizer.normalize_for_cost()
   → gv_norm = 1.0 / 2.0 = 0.5
   → rpm_norm = 30 / 50 = 0.6

3. 최적화 알고리즘:
   - 비용 함수 값 기준으로 수렴 (MinMax 스케일)
   - 하지만 예측 모델은 StandardScaler 스케일로 작동

→ 최적화 그래디언트와 예측 모델 입력 스케일이 맞지 않음!
```

---

## 3. 완전한 해결책

### 3.1 최선의 방법: 명시적 정규화

```
현재 문제의 근본 원인:
- zone_inputs 구성 시 제어값을 원본 스케일로 유지
- 모델이 자동으로 StandardScaler 적용
- 비용 함수는 MinMax 적용
→ 두 경로가 서로 다른 스케일 사용

해결책:
zone_inputs 구성 단계에서 제어값을 명시적으로 처리
```

### 3.2 구현 방법: Option A (권장)

```python
# multi_zone_controller.py 수정

class MultiZoneController:
    def __init__(self, model_manager, zone_properties=None, normalizer=None):
        self.model = model_manager
        self.normalizer = normalizer  # ★ ControlVariableNormalizer 추가
        ...

    def construct_zone_inputs(self, x, current_state,
                             use_normalized_control=False):  # ★ 옵션 추가
        """
        Args:
            use_normalized_control: True면 MinMax 정규화된 제어값 사용,
                                  False면 원본 값 사용
        """
        delta_gv = x[:N_GV]
        delta_rpm = x[N_GV]

        # ★ 제어값 정규화 여부 선택
        if use_normalized_control and self.normalizer:
            delta_gv_to_use, delta_rpm_to_use = self.normalizer.normalize_for_cost(
                delta_gv, delta_rpm
            )
        else:
            delta_gv_to_use = delta_gv
            delta_rpm_to_use = delta_rpm

        # zone_inputs 구성... (정규화된 또는 원본 제어값 사용)
        for i in range(N_ZONES):
            control_features_gv = np.array([
                delta_gv_to_use[gv_idx_left],
                delta_gv_to_use[gv_idx_center],
                delta_gv_to_use[gv_idx_right],
            ])
            control_features_rpm = np.array([delta_rpm_to_use])
            ...
```

### 3.3 구현 방법: Option B (대안)

```python
# optimizer_engine.py에서 명시적 처리

def objective_function(self, x):
    x_transformed = self.output_transformer.transform(x)

    # ★ 제어값 스케일 명시적 처리
    delta_gv = x_transformed[:N_GV]
    delta_rpm = x_transformed[N_GV]

    # 제어값이 예측 모델에 어떻게 들어갈지 명시
    # Option 1: 원본 값 (model이 StandardScaler 적용)
    # Option 2: 정규화된 값 (MinMax, 일관성 유지)

    control_result = self.controller.evaluate_control(x_transformed, ...)
    # ...
    cost = self.cost_evaluator.evaluate_total_cost(
        p_low, p_mid, p_high, delta_gv, delta_rpm
    )
```

---

## 4. 권장 결정: 현재 상태 평가

### 4.1 현재 구현의 안전성 평가

```
상황: 두 경로가 다른 스케일 사용

일반적인 우려 ❌
  - 예측 모델은 이미 StandardScaler로 학습됨 (변경 불가)
  - 따라서 zone_inputs의 제어값은 원본이어야 함
  - 비용 함수는 MinMax로 정규화 (공정 제약 기준)
  - 이는 의도적이고 필요한 설계

하지만, 불명확함 ✓
  - zone_inputs의 제어값이 어떤 스케일로 사용되는지 명확하지 않음
  - 코드 주석이 부족함
  - 향후 유지보수 시 혼동 가능성

권장: 명시적 주석과 선택적으로 Option A/B 적용
```

### 4.2 즉시 필요한 개선사항

```
1️⃣ 문서화 (지금 당장)
   - zone_inputs의 제어값이 원본임을 명시
   - 모델이 StandardScaler를 자동으로 적용함을 명시
   - 비용 함수는 별도의 MinMax 정규화를 사용함을 명시

2️⃣ 주석 추가 (지금 당장)
   - multi_zone_controller.py에 주석 추가
   - optimizer_engine.py에 주석 추가
   - 정규화 프로세스를 명확히 문서화

3️⃣ 선택사항: Option A/B 구현
   - 시간 있을 경우 구현
   - 현재 구현이 작동하고 있으므로 성급한 변경 불필요
```

---

## 5. 결론

### 현재 상태

```
✅ 작동: 예측 모델과 비용 함수가 각각 의도대로 작동
✅ 안전: 모델이 학습한 방식 유지 (StandardScaler)
✅ 기능: 최적화가 목적함수(비용 함수) 기준으로 수렴

❌ 불명확: 정규화 기준이 명확하게 문서화되지 않음
⚠️  위험: 향후 수정 시 혼동 가능성
```

### 권장사항

```
단기 (당장): 명시적 주석과 문서화
장기 (여유있을 때): Option A 구현

이유:
- 현재 구현이 기본적으로 작동함
- 성급한 변경은 예측 불가능한 결과 야기 가능
- 명확한 문서화가 더 중요함
- 향후 리팩토링 시 Option A/B를 고려
```

---

## 6. 액션 아이템

### 즉시 필요한 개선사항

```python
# multi_zone_controller.py에 추가할 주석

"""
⚠️ 정규화 기준 설명

zone_inputs 구성 시 제어값(delta_gv, delta_rpm)은 원본 스케일로 포함됩니다.
이후 model.predict_batch()에서 자동으로 StandardScaler가 적용됩니다.

이는 의도적 설계입니다:
1. 예측 모델은 StandardScaler로 학습됨
2. 따라서 zone_inputs의 제어값은 원본이어야 함
3. 모델이 자동으로 학습 데이터의 mean/std로 정규화

별도 경로:
- 비용 함수는 ControlVariableNormalizer(MinMax)를 사용
- 이는 공정 제약 조건(gv_max=2.0, rpm_max=50) 기반
- 두 정규화는 다른 목적이므로 서로 다름
"""
```

### 향후 개선사항

```
Option A: explicit normalization in construct_zone_inputs
  - normalizer 파라미터 추가
  - use_normalized_control 옵션 추가
  - 명시적 제어값 정규화 선택 가능

Option B: explicit handling in optimizer_engine
  - 제어값 정규화 방식을 명확히 기록
  - 예측 모델과 비용 함수의 정규화 프로세스 분리
```

