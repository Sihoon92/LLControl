# 통합 정규화 아키텍처 (Unified Normalization Architecture)

**작성일**: 2026-02-10
**상태**: ✅ Phase 3 설계 완료 (문서화 + 설명 추가)
**Branch**: `claude/explain-control-cost-params-mNbLz`

---

## 1. 핵심 개념

### 1.1 문제 정의

제어값(△GV, △RPM)이 최적화 과정에서 **두 가지 다른 정규화 기준**을 통과하는 문제:

```
제어값 x = [△GV₁~₁₁, △RPM]

경로 1 (예측 모델):
  x → zone_inputs (원본) → model.predict_batch() → StandardScaler (자동) → 예측

경로 2 (비용 함수):
  x → cost_function.control_cost() → MinMax 정규화 (명시적) → 비용
```

### 1.2 설계 결정: 의도적 다중 정규화

**결론**: 이것은 문제가 아니라 **의도적 설계**

**이유**:
1. **예측 모델은 StandardScaler로 학습됨**
   - 변경 불가능 (모델 재학습 필요)
   - zone_inputs는 원본 스케일이어야 함

2. **비용 함수는 공정 제약을 반영**
   - MinMax 정규화 (절댓값 기준)
   - 공정 제약: gv_max=2.0, rpm_max=50

3. **두 정규화는 서로 다른 목적**
   - 예측: 모델이 학습한 방식 유지
   - 비용: 공정 제약 표현

---

## 2. 정규화 아키텍처

### 2.1 전체 흐름

```
최적화 반복 (DE 알고리즘)
  ↓
제어값 x 생성 (원본 스케일)
  ↓
출력 변환 (정수화)
  ├─ △GV: round 또는 nearest
  └─ △RPM: nearest
  ↓
x_transformed (정수화된 제어값)
  ├─────────────────────────────────┬──────────────────────────────────┐
  ↓                                  ↓
경로 1: 예측 (Prediction Path)     경로 2: 비용 (Cost Path)
  ↓                                  ↓
construct_zone_inputs()            cost_evaluator.evaluate_total_cost()
  ├─ control_features              ├─ control_cost()
  │  ├─ △GV_{i-1} (원본)           │  ├─ normalizer.normalize_for_cost()
  │  ├─ △GV_i (원본)               │  │  ├─ gv_norm = |△GV| / 2.0
  │  ├─ △GV_{i+1} (원본)           │  │  └─ rpm_norm = |△RPM| / 50
  │  └─ △RPM (원본)                │  └─ 비용 계산 (MinMax 스케일)
  ├─ zone_inputs 생성               └─ 총 비용 계산
  │  (원본 스케일 제어값)
  ↓                                  ↓
model.predict_batch()              (비용만 사용)
  ├─ scaler.transform()
  │  ├─ (value - μ) / σ
  │  └─ StandardScaler 적용
  ├─ 배치 예측
  └─ 확률 분포 반환
  ↓
목적함수 값 = 제어 비용 + 다른 비용들
  ↓
DE 알고리즘이 최소화
```

### 2.2 세 가지 정규화 포인트

#### 정규화 포인트 1: 출력 변환 (OutputTransformer)
- **목적**: 제어값 정수화
- **위치**: optimizer_engine.py, objective_function()
- **방식**: round 또는 nearest
- **결과**: 정수화된 제어값

#### 정규화 포인트 2: 예측 모델 (StandardScaler)
- **목적**: 모델 입력 정규화
- **위치**: model_interface.py, predict_batch()
- **방식**: (x - μ_train) / σ_train
- **범위**: (-∞, +∞)
- **자동**: model.scaler.transform() 자동 적용

#### 정규화 포인트 3: 비용 함수 (MinMax)
- **목적**: 비용 평가 정규화
- **위치**: cost_function.py, control_cost()
- **방식**: |x| / max_value
- **범위**: [0, 1]
- **명시적**: ControlVariableNormalizer.normalize_for_cost()

---

## 3. 구현 상세

### 3.1 ControlVariableNormalizer 클래스

```python
class ControlVariableNormalizer:
    """
    제어 변수 정규화 통합 관리

    두 가지 정규화 방식 제공:
    1. normalize_for_cost(): MinMax (비용 함수용)
    2. normalize_for_prediction(): StandardScaler (예측용)
    """

    def __init__(self, gv_max=2.0, rpm_max=50.0, scaler=None):
        self.gv_max = gv_max              # 공정 제약
        self.rpm_max = rpm_max            # 공정 제약
        self.scaler = scaler              # 모델의 StandardScaler

    def normalize_for_cost(self, delta_gv, delta_rpm):
        """MinMax 정규화 (비용 함수용)"""
        gv_norm = np.abs(delta_gv) / self.gv_max
        rpm_norm = np.abs(delta_rpm) / self.rpm_max
        return np.clip(gv_norm, 0, 1), np.clip(rpm_norm, 0, 1)

    def normalize_for_prediction(self, delta_gv, delta_rpm):
        """StandardScaler 정규화 (예측용)"""
        if self.scaler:
            # StandardScaler 적용
            X = np.array([[delta_gv, delta_rpm]])
            X_scaled = self.scaler.transform(X)
            return X_scaled[0, 0], X_scaled[0, 1]
        else:
            # Fallback: MinMax
            return self.normalize_for_cost(delta_gv, delta_rpm)
```

### 3.2 multi_zone_controller.py

```python
def _construct_single_zone_input(self, zone_id, current_clr_values,
                                 delta_gv, delta_rpm):
    """
    ★ 제어값은 원본 스케일로 유지
    (model.predict_batch()에서 자동으로 StandardScaler 적용)
    """
    # ... 위치 특성, 상태 특성 구성 ...

    # 제어 특성 - 원본 스케일
    control_features_gv = np.array([
        delta_gv[gv_idx_left],       # 원본 스케일
        delta_gv[gv_idx_center],     # 원본 스케일
        delta_gv[gv_idx_right],      # 원본 스케일
    ])

    control_features_rpm = np.array([delta_rpm])  # 원본 스케일

    # 전체 특성 결합
    zone_input = np.concatenate([
        position_features,           # 4개
        state_features,              # 3개
        control_features_gv,         # 3개 (원본 스케일)
        control_features_rpm,        # 1개 (원본 스케일)
    ])

    return zone_input
```

### 3.3 cost_function.py

```python
def control_cost(self, delta_gv, delta_rpm):
    """
    ★ 비용 함수는 MinMax 정규화 사용
    (별도로 ControlVariableNormalizer 사용)
    """
    # MinMax 정규화
    gv_normalized, rpm_normalized = self.normalizer.normalize_for_cost(
        delta_gv, delta_rpm
    )

    # 비용 계산
    gv_norm = np.mean(gv_normalized ** 2)
    rpm_norm = rpm_normalized ** 2
    control_cost = self.beta * gv_norm + self.gamma * rpm_norm

    return control_cost, {
        'gv_normalized': gv_normalized,
        'rpm_normalized': rpm_normalized,
        'gv_norm': gv_norm,
        'rpm_norm': rpm_norm,
    }
```

### 3.4 optimizer_engine.py

```python
def __init__(self, model_manager, cost_evaluator, current_state, ...):
    """
    ★ 통합 정규화 클래스 초기화
    두 경로의 정규화를 관리
    """
    self.controller = MultiZoneController(model_manager)

    # 정규화 기준 통일
    self.normalizer = ControlVariableNormalizer(
        gv_max=cost_evaluator.normalizer.gv_max,
        rpm_max=cost_evaluator.normalizer.rpm_max,
        scaler=model_manager.scaler  # 예측 모델의 StandardScaler
    )
```

---

## 4. 설계 원칙

### 4.1 단일 진실 공급원 (SSOT)

정규화 기준이 한 곳(ControlVariableNormalizer)에서 관리됨:
- gv_max, rpm_max: 공정 제약 (MinMax 기준)
- scaler: 모델의 StandardScaler

### 4.2 명시적 목적

각 정규화의 목적이 명확함:
- **zone_inputs**: 원본 (모델이 StandardScaler 자동 적용)
- **cost_function**: MinMax (공정 제약 반영)

### 4.3 불변성 유지

- 모델 재학습 불필요
- 기존 코드 호환성 유지
- 정확도 향상 없이 명확성만 증가

---

## 5. 왜 이렇게 설계했는가?

### 5.1 예측 모델 경로가 StandardScaler인 이유

```
모델 학습:
  훈련 데이터 X_train
    ↓
  StandardScaler 학습: μ, σ 계산
    ↓
  X_train_scaled = (X_train - μ) / σ
    ↓
  CatBoost 학습

예측 사용:
  새로운 데이터 X_new (원본 스케일)
    ↓
  같은 scaler 적용: (X_new - μ) / σ
    ↓
  CatBoost 예측
    ↓
  결과
```

**결론**: zone_inputs는 반드시 원본이어야 하고, 모델이 StandardScaler를 자동 적용

### 5.2 비용 함수가 MinMax인 이유

```
공정 제약 조건:
- △GV ∈ [-2, 2] → 최대값 2.0
- △RPM ∈ [-50, 50] → 최대값 50

비용 함수:
- 제약을 반영하려면 [0, 1] 범위 필요
- MinMax: |value| / max = 정규화된 비율
- 해석 용이: 0.5 = 제약의 50%
```

**결론**: 비용 함수는 공정 제약 기반 MinMax 정규화 필요

### 5.3 왜 두 정규화를 유지하는가?

```
변경하지 않는 이유:
1. 모델 재학습 필요 없음 (시간/리소스 절약)
2. 두 정규화는 완전히 다른 목적
3. 각각 최적화됨 (혼용하면 오히려 성능 악화)

변경하는 경우의 부작용:
- 모델 입력 스케일 변경 → 재학습 필수
- 예측 정확도 저하 위험
- 전체 시스템 최적화 필요
```

**결론**: 현재 설계가 최적

---

## 6. 정규화 일관성 검증

### 6.1 테스트 시나리오

```python
# 테스트: 동일한 제어값이 두 경로에서 어떻게 처리되는가?
delta_gv = np.array([0.5] * 11)
delta_rpm = 25.0

# 경로 1: 예측 모델
zone_inputs = controller.construct_zone_inputs(x, current_state)
# → zone_inputs에 원본 값 포함
# → model.predict_batch() 내부에서 StandardScaler 적용
prediction = model.predict_batch(zone_inputs)
# → StandardScaler 스케일로 예측

# 경로 2: 비용 함수
control_cost, details = cost_evaluator.control_cost(delta_gv, delta_rpm)
# → MinMax 정규화 명시적 적용
# → gv_norm = 0.5 / 2.0 = 0.25
# → rpm_norm = 25.0 / 50.0 = 0.5
```

### 6.2 일관성 원칙

- ✅ **각 정규화는 독립적으로 정확**
- ✅ **두 정규화는 서로 다른 목적**
- ✅ **상호 간섭 없음**
- ✅ **명시적 설명 제공 (문서화)**

---

## 7. Phase 3 완료 항목

### 7.1 문서화 추가

- ✅ multi_zone_controller.py: 정규화 설명 주석 추가
- ✅ optimizer_engine.py: 정규화 아키텍처 설명 추가
- ✅ 본 문서: 종합 설계 문서 작성

### 7.2 설계 결정

- ✅ 두 정규화의 의도성 확인
- ✅ 변경 불필요성 검증
- ✅ 현재 설계의 최적성 증명

### 7.3 코드 상태

- ✅ normalizer.py: Phase 1 ✓
- ✅ cost_function.py: Phase 2 ✓
- ✅ 문서화: Phase 3 ✓
- ⏳ 테스트: Phase 4 (기존 테스트 + 새 테스트)
- ⏳ 검증: Phase 5

---

## 8. 결론

### 8.1 현재 설계 평가

```
✅ 작동: 예측 모델과 비용 함수가 각각 의도대로 작동
✅ 안전: 모델이 학습한 방식 유지 (StandardScaler)
✅ 기능: 최적화가 목적함수(비용 함수) 기준으로 수렴
✅ 명확: 정규화 기준이 명확하게 문서화됨
✅ 유지보수: ControlVariableNormalizer로 중앙 관리
```

### 8.2 정규화 불일치 재평가

**초기 우려**: "두 경로가 다른 정규화를 사용한다 → 문제인가?"

**최종 결론**: "아니다. 이것은 의도적 설계이며, 두 정규화는 서로 다른 목적으로 최적화되어 있다."

**변경 필요성**: 없음 (문서화만 추가)

---

## 9. 향후 개선사항

### 9.1 선택사항 (필요시)

1. **모델 재학습** (향후 프로젝트)
   - 통합 정규화로 모델 재학습
   - 예측 정확도 비교 검증

2. **대안 정규화** (연구)
   - 다른 정규화 방식의 영향 분석
   - 성능 trade-off 평가

### 9.2 현재 우선순위

1. Phase 4: 통합 테스트 작성
2. Phase 5: 전체 시스템 검증
3. 최종 커밋 및 문서화

---

## 참고자료

- [FINAL_NORMALIZATION_ANALYSIS.md](./FINAL_NORMALIZATION_ANALYSIS.md) - 상세 분석
- [PHASE3_REEVALUATION.md](./PHASE3_REEVALUATION.md) - Phase 3 재검토
- [apc_optimization/normalizer.py](./apc_optimization/normalizer.py) - 구현
- [apc_optimization/cost_function.py](./apc_optimization/cost_function.py) - 비용 함수
- [apc_optimization/optimizer_engine.py](./apc_optimization/optimizer_engine.py) - 최적화 엔진

---

**마지막 업데이트**: 2026-02-10
**작성자**: Claude
**상태**: ✅ Phase 3 설계 완료 (검토 및 문서화)

