# MBRL (Model-Based Reinforcement Learning) 모듈

Per-Zone PETS (Probabilistic Ensembles with Trajectory Sampling) 구현

## 📁 파일 구조

```
mbrl/
├── __init__.py           # 패키지 초기화
├── config.py             # PETS 설정
├── ensemble_nn.py        # Per-Zone Neural Network
├── dynamics_model.py     # 확률적 Dynamics Model
├── data_processor.py     # 데이터 전처리
└── README.md            # 이 파일
```

## 🎯 설계 철학

### CatBoost와 동일한 구조 (공정한 비교)

**Per-Zone 모델링**:
- 각 Zone을 독립적으로 예측
- 인접 Zone 정보는 입력에 포함 (i-1, i, i+1)
- 위치 특성으로 물리적 특성 반영

**입력 구조** (11개):
```
[
  위치 특성 (4): distance, edge_distance, normalized_position, normalized_distance
  현재 CLR (3): CLR_1, CLR_2, CLR_3
  제어 변화 (4): delta_GV_{i-1}, delta_GV_i, delta_GV_{i+1}, delta_RPM
]
```

**출력 구조** (3개):
```
diff_CLR = [diff_CLR_1, diff_CLR_2, diff_CLR_3]
```

## 🔧 주요 차이점: CatBoost vs PETS

| 특징 | CatBoost | PETS |
|------|----------|------|
| 예측 방식 | 점 예측 (단일 값) | 확률 분포 (mean + variance) |
| 불확실성 | ❌ 없음 | ✅ Aleatoric + Epistemic |
| 모델 타입 | Gradient Boosting | Ensemble Neural Network |
| 학습 방식 | 트리 앙상블 | Backpropagation |
| 출력 | y | (μ, σ²) |

## 🚀 사용법

### 1. 환경 설정

```bash
# PyTorch 설치 (필수)
pip install torch torchvision

# 기타 의존성
pip install numpy pandas scikit-learn joblib
```

### 2. 데이터 준비

```python
from apc_optimization.mbrl import PETSDataProcessor

# 데이터 처리기 생성
processor = PETSDataProcessor(
    normalize=True,
    validation_split=0.2,
    test_split=0.1,
    random_seed=42
)

# 데이터 로드 및 전처리
data_dict = processor.process(
    data_file='outputs/model_training_data.xlsx',
    mode='training'
)

# 결과: {'train': (X, y), 'val': (X, y), 'test': (X, y)}
X_train, y_train = data_dict['train']
X_val, y_val = data_dict['val']
X_test, y_test = data_dict['test']
```

### 3. 모델 학습

```python
from apc_optimization.mbrl import PerZoneProbabilisticEnsemble
import torch

# 모델 생성
model = PerZoneProbabilisticEnsemble(
    n_ensembles=5,
    input_dim=11,
    output_dim=3,
    hidden_dims=[128, 128],
    device='cpu'  # or 'cuda' if available
)

# Optimizer 초기화
model.init_optimizers(lr=1e-3, weight_decay=1e-5)

# 학습 루프
n_epochs = 100
batch_size = 256

for epoch in range(n_epochs):
    # 배치 샘플링
    indices = np.random.choice(len(X_train), batch_size, replace=False)
    X_batch = X_train[indices]
    y_batch = y_train[indices]

    # 학습 step
    metrics = model.train_on_batch(X_batch, y_batch)

    if (epoch + 1) % 10 == 0:
        # 검증
        val_metrics = model.evaluate_on_batch(X_val, y_val)
        print(f"Epoch {epoch+1}: Train Loss={metrics['loss']:.6f}, Val MSE={val_metrics['mse']:.6f}")

# 모델 저장
model.save('outputs/models_v2/mbrl/pets_per_zone_best.pt')
```

### 4. 예측 (전체 11개 Zone)

```python
import numpy as np

# 현재 상태
current_clr_all = np.random.randn(11, 3)  # 11 zones × 3 CLR
delta_gv = np.array([0.5, 0.3, ..., 0.1])  # 11개 GV 변화
delta_rpm = 10.0

# 예측
result = model.predict_all_zones(
    current_clr_all,
    delta_gv,
    delta_rpm,
    return_uncertainty=True
)

# 결과
diff_clr_mean = result['diff_clr_mean']          # (11, 3) - 예측 평균
diff_clr_uncertainty = result['diff_clr_uncertainty']  # (11, 3) - 불확실성
next_clr = result['next_clr']                     # (11, 3) - 다음 상태

# 불확실성 활용
high_uncertainty_zones = np.where(diff_clr_uncertainty.mean(axis=1) > threshold)[0]
print(f"High uncertainty zones: {high_uncertainty_zones + 1}")  # Zone ID (1-based)
```

### 5. CatBoost와 비교

```python
from apc_optimization import CatBoostModelManager

# CatBoost 모델 로드
catboost_model = CatBoostModelManager()

# PETS 모델 로드
pets_model = PerZoneProbabilisticEnsemble(...)
pets_model.load('outputs/models_v2/mbrl/pets_per_zone_best.pt')

# 테스트 데이터 평가
for model_name, model in [('CatBoost', catboost_model), ('PETS', pets_model)]:
    if model_name == 'CatBoost':
        # CatBoost 예측 (구현 필요)
        pass
    else:
        metrics = model.evaluate_on_batch(X_test, y_test)
        print(f"{model_name}: MSE={metrics['mse']:.6f}, MAE={metrics['mae']:.6f}")
        print(f"  R²={metrics['r2']:.4f}, Uncertainty={metrics['mean_uncertainty']:.6f}")
```

## 📊 성능 비교 지표

### 예측 정확도
- **MSE** (Mean Squared Error): 작을수록 좋음
- **MAE** (Mean Absolute Error): 작을수록 좋음
- **R² Score**: 1에 가까울수록 좋음

### 불확실성 (PETS만)
- **Calibration (ECE)**: 예측 불확실성이 실제 오차와 일치하는가?
- **NLL** (Negative Log-Likelihood): 작을수록 좋음

### Open-loop 시뮬레이션
- 실제 궤적을 얼마나 정확히 재현하는가?

## ⚙️ 하이퍼파라미터 튜닝

`config.py`에서 설정 가능:

```python
# 모델 구조
DYNAMICS_MODEL_CONFIG = {
    'hidden_dims': [128, 128],  # Hidden layer 크기
    'activation': 'relu',
    'use_layer_norm': True,
    'dropout': 0.0,
}

# 앙상블
ENSEMBLE_CONFIG = {
    'n_ensembles': 5,  # 5~7 권장
}

# 학습
TRAINING_CONFIG = {
    'batch_size': 256,
    'epochs': 100,
    'learning_rate': 1e-3,
    'weight_decay': 1e-5,
}
```

## 🔬 실험 권장사항

### Phase 1: Dynamics Model 단독 평가
1. PETS 학습
2. CatBoost와 예측 정확도 비교
3. 불확실성 교정 검증

**목표**: PETS MSE ≤ CatBoost MSE

### Phase 2: 제어기 통합 (미래)
1. Planner (CEM) 구현
2. CatBoost + DE vs PETS + CEM 비교

## 📝 TODO

- [ ] Planner (CEM) 구현
- [ ] Trajectory Sampler 구현
- [ ] 모델 비교 프레임워크 구현
- [ ] 시각화 도구 추가
- [ ] 성능 벤치마크 실험

## 📚 참고 문헌

- PETS: [Chua et al., 2018](https://arxiv.org/abs/1805.12114)
- CatBoost: [Prokhorenkova et al., 2018](https://arxiv.org/abs/1706.09516)

---

**Author**: LLControl Team
**Version**: 0.1.0
**Last Updated**: 2024-02-13
