# MBRL 성능 악영향 요소 및 수정 방법

## 개요

MBRL(Model-Based Reinforcement Learning) / PETS Probabilistic Ensemble은 NLL(Negative Log-Likelihood) 손실을 사용하는 확률적 모델이다.
이 특성으로 인해 일반 회귀 모델(GPR, XGBoost, CatBoost 등)에는 문제가 되지 않는 요소들이 MBRL에서는 심각한 성능 저하로 이어질 수 있다.

---

## 1. Y 미정규화 (핵심 문제)

### 문제

NLL 손실 수식:

```
NLL = 0.5 * [log(2π) + log(σ²) + (y - μ)² / σ²]
```

네트워크는 `log_var`를 예측하고, 내부적으로 `inv_var = exp(-log_var)`를 계산한다.
Y가 원본 스케일(예: 값 범위 수십~수백)로 들어오면:

- `(y - μ)²`의 크기가 너무 커서 NLL이 `log_var` 항을 무시하고 MSE처럼 동작
- 네트워크가 불확실성(variance)을 제대로 학습하지 못함
- `exp(-log_var)` 오버플로우/언더플로우 위험
- 기존 모델과 달리 MBRL만 학습이 불안정해져 불공정한 비교

### 수정 (적용 완료)

`train_mbrl_ensemble()` 내부에서 Y에 `StandardScaler`를 적용:

```python
from sklearn.preprocessing import StandardScaler
y_scaler = StandardScaler()
Y_tr_s  = y_scaler.fit_transform(Y_tr)   # train으로 fit
Y_val_s = y_scaler.transform(Y_val)      # transform만 (데이터 누수 방지)

# 학습 후 예측값을 원본 스케일로 복원
mean_pred = y_scaler.inverse_transform(mean_pred_norm)
```

**비교 공정성:** 다른 모델들은 원본 스케일로 비교 → MBRL도 역변환 후 원본 스케일로 비교.

---

## 2. log_var 범위 제한이 실제 데이터에 부적합

### 문제

현재 설정:

```python
log_var_min: -10
log_var_max:  2
```

Y를 정규화하면 `σ² ≈ 1`이므로 `log_var ≈ 0`이 적절한 초기값이다.
Y를 정규화하지 않은 경우, 원본 스케일 분산이 `log_var_max=2`(즉, `σ² ≈ 7.4`)보다 훨씬 클 수 있어 clamp가 최적값 탐색을 막는다.

### 수정 방법

Y 정규화 적용 후에는 현재 범위(`-10`, `2`)가 적절.
필요 시 `log_var_max`를 약간 올려(`4~5`) 유연성 확보:

```python
# apc_optimization/mbrl/config.py
'log_var_min': -10,
'log_var_max': 4,   # 필요 시 조정
```

---

## 3. Validation MSE가 정규화 공간에서만 기록됨

### 문제

Early stopping 기준으로 사용하는 `val_mse`는 정규화된 Y 공간에서 계산된다.
로그 메시지가 "정규화 공간"이라고 명시하므로 혼동은 없으나, **최종 평가와 단위가 달라** 해석 시 주의 필요.

```python
self.logger.info(f"  Best Val MSE (정규화 공간): {best_val_loss:.6f}")
```

Early stopping 자체는 정규화 공간에서 해도 무방(순서는 동일).
최종 test 성능은 역변환 후 `evaluate_models()`에서 원본 스케일로 표시된다.

### 수정 방법 (선택)

로그에 원본 스케일 MSE도 추가하려면:

```python
val_mean_orig = y_scaler.inverse_transform(val_mean.cpu().numpy())
val_mse_orig  = float(((val_mean_orig - Y_val) ** 2).mean())
self.logger.info(f"  Best Val MSE (원본 스케일): {val_mse_orig:.6f}")
```

---

## 4. Bootstrap 비율과 데이터 크기 불균형

### 문제

`bootstrap_ratio=0.8`은 각 앙상블 멤버가 전체 train 데이터의 80%를 사용함을 의미한다.
데이터가 **적을수록**(< 500 샘플) 멤버 간 다양성이 부족해 Epistemic uncertainty 추정이 부정확해진다.

### 수정 방법

데이터가 적은 경우 비율을 낮추거나, 앙상블 수를 늘린다:

```python
# 데이터 < 500 샘플 → bootstrap_ratio 낮춤
trainer.train_mbrl_ensemble(
    bootstrap_ratio=0.6,  # 더 다양한 서브셋
    n_ensembles=7         # 멤버 수 증가
)
```

---

## 5. Learning Rate가 네트워크 깊이에 비해 너무 큼

### 문제

`hidden_dims=[512, 256, 256, 128]` (4 layers)의 깊은 네트워크에서
`lr=5e-4`는 초기 학습이 불안정할 수 있다(특히 Y 정규화 이후 NLL 스케일 변화).

### 수정 방법

Learning rate를 낮추거나 Warmup을 추가:

```python
trainer.train_mbrl_ensemble(
    learning_rate=1e-4,   # 보수적 LR
    epochs=300,           # 더 많은 epoch으로 보완
)
```

또는 `config.py`에서 cosine scheduler 활성화 (현재 정의됨, 미구현):

```python
TRAINING_CONFIG = {
    'lr_scheduler': 'cosine',  # 현재 train.py에서만 사용, model_trainer_v2에 미적용
    ...
}
```

---

## 6. Gradient Clipping 임계값

### 문제

현재 `clip_grad_norm_(..., max_norm=1.0)` 적용 중.
NLL 손실에서 `inv_var = exp(-log_var)` 항이 gradient를 증폭시킬 수 있어,
max_norm=1.0이 너무 관대할 수 있다.

### 수정 방법

더 엄격한 clipping 적용:

```python
torch.nn.utils.clip_grad_norm_(m.parameters(), max_norm=0.5)
```

---

## 7. 위치 특성(distance)의 스케일 불균형

### 문제

입력 피처 중 위치 특성:

```
distance:           0, 100, 200, ..., 1000  (mm 단위, 범위: 0~1000)
edge_distance:      0.5~5.5
normalized_position: 0~1
normalized_distance: 0~1
```

`distance` 피처가 다른 피처(CLR, delta_GV 등) 대비 1000배 큰 스케일을 가져
X 정규화가 필요하다. 현재 `self.X_train_scaled`를 사용하므로 X는 정규화됨.
단, MBRL 전용 `PETSDataProcessor`를 사용하는 `train.py` 경로에서도 동일하게 X 정규화가 이루어지는지 확인 필요.

### 확인 방법

```python
# data_processor.py
self.input_scaler = StandardScaler() if normalize else None
# normalize=True (DATA_CONFIG['normalize_inputs']=True) 확인
```

---

## 요약 테이블

| # | 요소 | 심각도 | 상태 | 수정 필요 여부 |
|---|------|--------|------|--------------|
| 1 | Y 미정규화 (NLL 불안정) | 🔴 높음 | ✅ 수정 완료 | 완료 |
| 2 | log_var 범위 부적합 | 🟡 중간 | Y 정규화 후 개선됨 | 선택적 |
| 3 | Val MSE 단위 혼동 | 🟢 낮음 | 로그 명시됨 | 선택적 |
| 4 | Bootstrap 비율/데이터 불균형 | 🟡 중간 | 데이터 수에 따라 조정 | 조건부 |
| 5 | LR이 깊은 네트워크에 과다 | 🟡 중간 | Y 정규화 후 개선됨 | 선택적 |
| 6 | Gradient Clipping 임계값 | 🟢 낮음 | 현재 동작 | 선택적 |
| 7 | 위치 특성 스케일 불균형 (X) | 🟡 중간 | X 정규화로 처리됨 | 확인 완료 |

---

## 비교 기준 명확화

| 단계 | 기존 모델 (GPR, XGBoost 등) | MBRL |
|------|----------------------------|------|
| 학습 Y | 원본 스케일 | **정규화 (수정 후)** |
| 예측 Y | 원본 스케일 | 정규화 공간 → **역변환 → 원본 스케일** |
| 비교 Y | 원본 스케일 (`Y_test`) | 원본 스케일 (`Y_test`) |

→ **공정한 비교 가능**
