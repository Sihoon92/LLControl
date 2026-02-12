# 밀도계 데이터 추출 워크플로우

## 올바른 사용 방법

### 전체 프로세스

```
1. APC 전처리 (3rd 파일 생성)
   ↓
2. 제어 구간 정보 생성 (4th 파일 생성) ← 여기서 before/after 시간 설정!
   ↓
3. 비제어 구간 샘플링 (5th 파일 생성)
   ↓
4. 밀도계 데이터 추출 (6th 파일 생성)
   ↓
5. 통계 분석
```

---

## 1단계: APC 전처리

```python
from preprocessor.apc_preprocessor import APCPreprocessor
from preprocessor.preprocess_config import PreprocessConfig

config = PreprocessConfig()
apc_processor = APCPreprocessor(config, logger)

# APC 데이터 전처리 실행
result = apc_processor.run('data/apc_raw.xlsx')

# 생성 파일: outputs/3rd_meaningful_changes.xlsx
```

**3rd 파일 내용**:
- group_id, start_time, end_time (제어 시작/종료 시점)
- 변수별 before/after 값

---

## 2단계: 제어 구간 정보 생성 (중요!)

**여기서 before/after 시간 범위를 설정합니다!**

```python
# 3rd 파일 읽기
meaningful_df = pd.read_excel('outputs/3rd_meaningful_changes.xlsx')

# 제어 구간 정보 생성
control_regions_df = apc_processor.create_control_regions_info(
    meaningful_df,
    before_minutes=1,   # ← 제어 전 1분
    after_minutes=6     # ← 제어 후 6분
)

# 4th 파일 저장
control_regions_df.to_excel('outputs/4th_control_regions.xlsx', index=False)
```

**4th 파일 내용**:
```
| group_id | control_type | start_time | end_time | control_start | control_end |
|----------|--------------|------------|----------|---------------|-------------|
| 1        | controlled   | 09:59:00   | 10:07:00 | 10:00:00      | 10:01:00    |
```

**핵심**:
- `start_time` = control_start - 1분 = 09:59:00 (실제 추출 시작)
- `end_time` = control_end + 6분 = 10:07:00 (실제 추출 종료)
- **이미 before/after가 적용된 범위입니다!**

---

## 3단계: 비제어 구간 샘플링

```python
# 비제어 구간 샘플링 (대조군)
no_control_regions_df = apc_processor.sample_no_control_regions(
    original_df=original_df,
    grouped_changes=grouped_changes,
    n_samples=10,
    before_minutes=1,   # 제어 구간과 동일하게 설정
    after_minutes=6,
    min_gap_minutes=10
)

# 5th 파일 저장
no_control_regions_df.to_excel('outputs/5th_no_control_regions.xlsx', index=False)
```

**5th 파일 내용**:
```
| group_id | control_type | start_time | end_time | reference_point |
|----------|--------------|------------|----------|-----------------|
| 1        | no_control   | 11:00:00   | 11:08:00 | 11:01:00        |
```

---

## 4단계: 밀도계 데이터 추출

```python
from preprocessor.densitometer_preprocessor import DensitometerPreprocessor

densitometer_processor = DensitometerPreprocessor(config, logger)

# 밀도계 데이터 추출 (제어 + 비제어)
extracted_data = densitometer_processor.run(
    control_regions_file='outputs/4th_control_regions.xlsx',
    no_control_regions_file='outputs/5th_no_control_regions.xlsx',
    raw_data_file='data/densitometer_raw.xlsx'
)

# 생성 파일: outputs/extracted_densitometer_data.xlsx
```

**동작 원리**:
```python
# 4th 파일의 start_time/end_time을 그대로 사용
# 이미 before/after가 적용된 범위이므로 추가 계산 불필요
mask = (raw_df['datetime'] >= start_time) & (raw_df['datetime'] <= end_time)
group_data = raw_df[mask].copy()
```

**추출 데이터 구조**:
```
| group_id | control_type | before/after | time     | Value1 | Value2 |
|----------|--------------|--------------|----------|--------|--------|
| 1        | controlled   | before       | 09:59:30 | 1.234  | 1.235  |
| 1        | controlled   | after        | 10:01:30 | 1.240  | 1.241  |
| 2        | no_control   | before       | 11:00:30 | 1.236  | 1.237  |
| 2        | no_control   | after        | 11:01:30 | 1.238  | 1.239  |
```

---

## 5단계: 통계 분석

```python
from preprocessor.zone_analyzer import ZoneAnalyzer

zone_analyzer = ZoneAnalyzer(config, logger)

# 통계 분석 실행
analysis_results = zone_analyzer.run(
    densitometer_file='outputs/extracted_densitometer_data.xlsx'
)

# 생성 파일: outputs/statistical_analysis_results.xlsx
```

---

## 핵심 설계 원칙

### 단일 책임 원칙

1. **apc_preprocessor**:
   - 제어 구간 식별
   - before/after 시간 범위 계산
   - 4th/5th 파일 생성

2. **densitometer_preprocessor**:
   - 4th/5th 파일 읽기
   - 이미 계산된 범위로 밀도계 데이터 추출

### 데이터 흐름

```
3rd 파일 (제어 시작/종료)
   ↓
[apc_preprocessor.create_control_regions_info()]
   - before_minutes, after_minutes 적용
   - start_time = control_start - before_minutes
   - end_time = control_end + after_minutes
   ↓
4th 파일 (이미 before/after 적용된 범위)
   ↓
[densitometer_preprocessor.extract_densitometer_data()]
   - start_time ~ end_time 그대로 사용
   - 추가 계산 없음
   ↓
추출된 밀도계 데이터
```

---

## 시간 범위 설정

### before_minutes / after_minutes는 2단계에서 설정!

```python
# ✅ 올바른 방법: 4th 파일 생성 시 설정
control_regions_df = apc_processor.create_control_regions_info(
    meaningful_df,
    before_minutes=1,   # 여기서 설정!
    after_minutes=6     # 여기서 설정!
)

# ❌ 잘못된 방법: 밀도계 추출 시 추가 설정 (중복 계산!)
extracted_data = densitometer_processor.run(
    control_regions_file='...',
    raw_data_file='...',
    before_minutes=1,   # 불필요! 이미 4th 파일에 적용됨
    after_minutes=6
)
```

### 예제: 제어 전 1분 / 후 6분 데이터 추출

```python
# Step 1: APC 전처리
apc_processor.run('data/apc_raw.xlsx')

# Step 2: 제어 구간 정보 생성 (여기서 시간 설정!)
meaningful_df = pd.read_excel('outputs/3rd_meaningful_changes.xlsx')
control_regions_df = apc_processor.create_control_regions_info(
    meaningful_df,
    before_minutes=1,   # 제어 전 1분
    after_minutes=6     # 제어 후 6분
)
control_regions_df.to_excel('outputs/4th_control_regions.xlsx', index=False)

# Step 3: 비제어 구간 샘플링 (동일한 시간 범위)
no_control_regions_df = apc_processor.sample_no_control_regions(
    original_df=original_df,
    grouped_changes=grouped_changes,
    before_minutes=1,   # 제어 구간과 동일
    after_minutes=6
)
no_control_regions_df.to_excel('outputs/5th_no_control_regions.xlsx', index=False)

# Step 4: 밀도계 데이터 추출 (4th/5th 파일 그대로 사용)
extracted_data = densitometer_processor.run(
    control_regions_file='outputs/4th_control_regions.xlsx',
    no_control_regions_file='outputs/5th_no_control_regions.xlsx',
    raw_data_file='data/densitometer_raw.xlsx'
)
```

---

## FAQ

### Q1: 3rd 파일에서 직접 밀도계 데이터를 추출할 수 없나요?

**A**: 아니요, 반드시 4th 파일을 먼저 생성해야 합니다.

**이유**:
- 3rd 파일은 제어 시작/종료 시점만 저장
- 실제 밀도계 데이터는 제어 전/후를 포함한 더 넓은 범위가 필요
- 4th 파일 생성 시 이 범위를 계산하고 저장

**올바른 워크플로우**:
```
3rd → 4th (before/after 적용) → densitometer 추출
```

### Q2: before_minutes/after_minutes를 변경하고 싶어요

**A**: 4th, 5th 파일을 다시 생성하세요.

```python
# 새로운 시간 범위로 4th 파일 재생성
control_regions_df = apc_processor.create_control_regions_info(
    meaningful_df,
    before_minutes=2,   # 변경: 2분
    after_minutes=10    # 변경: 10분
)
control_regions_df.to_excel('outputs/4th_control_regions.xlsx', index=False)

# 그 다음 밀도계 데이터 재추출
extracted_data = densitometer_processor.run(...)
```

### Q3: 왜 이렇게 설계했나요?

**A**: 단일 책임 원칙과 중복 계산 방지를 위해

**장점**:
1. **명확한 역할 분담**
   - apc_preprocessor: 시간 범위 계산
   - densitometer_preprocessor: 데이터 추출

2. **중복 계산 방지**
   - before/after를 한 곳에서만 계산

3. **재사용성**
   - 4th 파일을 여러 번 재사용 가능
   - 밀도계 추출을 여러 번 실행해도 동일한 결과

---

## 참고

- 전체 테스트: `test_statistical_analysis.py`
- 통계 분석 가이드: `STATISTICAL_ANALYSIS_GUIDE.md`
