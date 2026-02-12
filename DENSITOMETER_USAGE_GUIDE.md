# 밀도계 데이터 추출 가이드

## 개요

densitometer_preprocessor는 APC 제어 구간에 해당하는 밀도계 raw data를 추출하는 모듈입니다.
v1.3부터 두 가지 사용 방법을 지원합니다.

---

## 사용 방법

### 방법 1: 3rd_meaningful_changes 파일 사용 (v1.2 호환)

**용도**: APC 전처리 결과에서 직접 밀도계 데이터 추출

**특징**:
- 제어 전/후 시간을 지정할 수 있음 (예: 전 1분, 후 6분)
- v1.2와 동일한 방식으로 동작

**코드 예제**:

```python
from preprocessor.densitometer_preprocessor import DensitometerPreprocessor
from preprocessor.preprocess_config import PreprocessConfig
import logging

# Config 및 Logger 설정
config = PreprocessConfig()
logger = logging.getLogger('test')
logging.basicConfig(level=logging.INFO)

# Preprocessor 생성
processor = DensitometerPreprocessor(config, logger)

# 밀도계 데이터 추출
extracted_data = processor.run_from_meaningful_changes(
    meaningful_changes_file='outputs/3rd_meaningful_changes.xlsx',
    raw_data_file='data/densitometer_raw.xlsx',
    before_minutes=1,   # 제어 전 1분
    after_minutes=6     # 제어 후 6분
)

print(f"추출된 데이터: {len(extracted_data)}행")
```

**동작 원리**:
1. 3rd_meaningful_changes에서 start_time, end_time 읽기
2. 실제 추출 범위 계산:
   - `extract_start = start_time - before_minutes`
   - `extract_end = end_time + after_minutes`
3. 해당 범위의 밀도계 데이터 추출
4. start_time을 기준으로 before/after 구분

**출력 데이터 구조**:
```
| group_id | before/after | time     | Value1 | Value2 | ... |
|----------|--------------|----------|--------|--------|-----|
| 1        | before       | 10:00:30 | 1.234  | 1.235  | ... |
| 1        | after        | 10:01:30 | 1.240  | 1.241  | ... |
```

---

### 방법 2: 4th/5th 파일 사용 (제어/비제어 구간 비교)

**용도**: 통계 분석을 위한 제어 구간과 비제어 구간 데이터 동시 추출

**특징**:
- 제어 구간과 비제어 구간(대조군)을 동시에 처리
- control_type 칼럼으로 구분 ('controlled' vs 'no_control')
- 통계 분석에 최적화

**코드 예제**:

```python
from preprocessor.densitometer_preprocessor import DensitometerPreprocessor
from preprocessor.preprocess_config import PreprocessConfig
import logging

# Config 및 Logger 설정
config = PreprocessConfig()
logger = logging.getLogger('test')
logging.basicConfig(level=logging.INFO)

# Preprocessor 생성
processor = DensitometerPreprocessor(config, logger)

# 밀도계 데이터 추출 (제어 + 비제어)
extracted_data = processor.run(
    control_regions_file='outputs/4th_control_regions.xlsx',
    no_control_regions_file='outputs/5th_no_control_regions.xlsx',
    raw_data_file='data/densitometer_raw.xlsx',
    before_minutes=0,   # 4th/5th 파일에 이미 범위가 설정되어 있으면 0
    after_minutes=0     # 추가 확장이 필요하면 값 지정
)

print(f"추출된 데이터: {len(extracted_data)}행")
print(f"제어 구간: {len(extracted_data[extracted_data['control_type']=='controlled'])}행")
print(f"비제어 구간: {len(extracted_data[extracted_data['control_type']=='no_control'])}행")
```

**동작 원리**:
1. 4th_control_regions.xlsx: 제어 구간 정보 읽기
2. 5th_no_control_regions.xlsx: 비제어 구간 정보 읽기
3. 각 구간의 start_time, end_time 범위의 데이터 추출
4. control_start 또는 reference_point 기준으로 before/after 구분

**출력 데이터 구조**:
```
| group_id | control_type | before/after | time     | Value1 | Value2 | ... |
|----------|--------------|--------------|----------|--------|--------|-----|
| 1        | controlled   | before       | 10:00:30 | 1.234  | 1.235  | ... |
| 1        | controlled   | after        | 10:01:30 | 1.240  | 1.241  | ... |
| 2        | no_control   | before       | 11:00:30 | 1.236  | 1.237  | ... |
| 2        | no_control   | after        | 11:01:30 | 1.238  | 1.239  | ... |
```

---

## 파라미터 설명

### before_minutes / after_minutes

**의미**: start_time/end_time으로부터 추가로 확장할 시간 (분)

**사용 시나리오**:

#### 시나리오 1: 3rd_meaningful_changes 직접 사용
```python
# 3rd 파일의 start_time = 10:00:00, end_time = 10:01:00
# before_minutes=1, after_minutes=6 설정

# 실제 추출 범위:
# - extract_start = 10:00:00 - 1분 = 09:59:00
# - extract_end = 10:01:00 + 6분 = 10:07:00

extracted_data = processor.run_from_meaningful_changes(
    meaningful_changes_file='outputs/3rd_meaningful_changes.xlsx',
    raw_data_file='data/densitometer_raw.xlsx',
    before_minutes=1,
    after_minutes=6
)
```

#### 시나리오 2: 4th/5th 파일 사용
```python
# 4th 파일의 start_time이 이미 before/after가 적용된 시간
# (apc_preprocessor가 생성할 때 이미 계산됨)

# Case A: 추가 확장 불필요
extracted_data = processor.run(
    control_regions_file='outputs/4th_control_regions.xlsx',
    no_control_regions_file='outputs/5th_no_control_regions.xlsx',
    raw_data_file='data/densitometer_raw.xlsx',
    before_minutes=0,   # 확장 안 함
    after_minutes=0
)

# Case B: 추가 확장 필요 (더 넓은 범위 분석)
extracted_data = processor.run(
    control_regions_file='outputs/4th_control_regions.xlsx',
    no_control_regions_file='outputs/5th_no_control_regions.xlsx',
    raw_data_file='data/densitometer_raw.xlsx',
    before_minutes=2,   # 추가로 2분 확장
    after_minutes=3     # 추가로 3분 확장
)
```

---

## before/after 구분 로직

### 제어 구간 (controlled)

1. **control_start가 있는 경우** (4th 파일):
   ```python
   before/after = 'before' if datetime < control_start else 'after'
   ```

2. **control_start가 없는 경우** (3rd 파일):
   ```python
   before/after = 'before' if datetime < start_time else 'after'
   ```

### 비제어 구간 (no_control)

1. **reference_point가 있는 경우** (5th 파일):
   ```python
   before/after = 'before' if datetime < reference_point else 'after'
   ```

2. **reference_point가 없는 경우**:
   ```python
   # 중간 지점 사용
   midpoint = start_time + (end_time - start_time) / 2
   before/after = 'before' if datetime < midpoint else 'after'
   ```

---

## 전체 워크플로우

### v1.2 스타일 (단순 추출)

```
1. APC 전처리
   ↓
2. 3rd_meaningful_changes.xlsx 생성
   ↓
3. densitometer_preprocessor.run_from_meaningful_changes()
   ↓
4. extracted_densitometer_data.xlsx 생성
```

### v1.3 스타일 (통계 분석 포함)

```
1. APC 전처리
   ↓
2. 3rd_meaningful_changes.xlsx 생성
   ↓
3. 4th_control_regions.xlsx 생성 (제어 구간)
   ↓
4. 5th_no_control_regions.xlsx 생성 (비제어 구간)
   ↓
5. densitometer_preprocessor.run()
   ↓
6. extracted_densitometer_data.xlsx 생성 (제어 + 비제어)
   ↓
7. zone_analyzer 통계 분석
   ↓
8. statistical_analysis_results.xlsx 생성
```

---

## 설정 (preprocess_config.py)

```python
class PreprocessConfig:
    # 시간 설정
    BEFORE_MINUTES = 1   # 기본값: 제어 전 1분
    AFTER_MINUTES = 6    # 기본값: 제어 후 6분

    # 출력 파일명
    OUTPUT_3RD = '3rd_meaningful_changes.xlsx'
    OUTPUT_4TH_CONTROL = '4th_control_regions.xlsx'
    OUTPUT_5TH_NO_CONTROL = '5th_no_control_regions.xlsx'
    OUTPUT_DENSITOMETER = 'extracted_densitometer_data.xlsx'
```

---

## 주의사항

1. **시간 범위 설정**
   - 너무 짧으면: 데이터 부족으로 통계 분석 어려움
   - 너무 길면: 다른 제어 효과와 겹칠 가능성

2. **파일 칼럼 구조**
   - 3rd 파일: `group_id`, `start_time`, `end_time` 필수
   - 4th 파일: `group_id`, `start_time`, `end_time`, `control_start`, `control_end` 권장
   - 5th 파일: `group_id`, `start_time`, `end_time`, `reference_point` 권장

3. **데이터 품질**
   - 중복 행은 자동 제거됨
   - 시간 형식은 HH:MM:SS 또는 datetime 형식 지원
   - 밀도계 raw data의 첫 번째 칼럼은 반드시 time 칼럼

---

## 트러블슈팅

### Q1: "필수 칼럼이 없습니다" 오류

**원인**: 파일에 `group_id`, `start_time`, `end_time` 칼럼이 없음

**해결**:
```python
# 파일 칼럼 확인
import pandas as pd
df = pd.read_excel('outputs/3rd_meaningful_changes.xlsx')
print(df.columns)

# 칼럼명이 다른 경우 수정 필요
```

### Q2: 추출된 데이터가 없음

**원인**: 시간 범위가 밀도계 데이터 범위와 겹치지 않음

**해결**:
```python
# 시간 범위 확인
print(f"3rd 파일 시간 범위: {df['start_time'].min()} ~ {df['end_time'].max()}")

# 밀도계 데이터 시간 범위 확인
raw_df = pd.read_excel('data/densitometer_raw.xlsx')
print(f"밀도계 시간 범위: {raw_df.iloc[:, 0].min()} ~ {raw_df.iloc[:, 0].max()}")
```

### Q3: before/after 비율이 이상함

**원인**: before_minutes/after_minutes 설정이 맞지 않음

**해결**:
```python
# 추출 범위 로그 확인
# "start_time 기준 X분 전부터 추출" 메시지 확인

# 적절한 값으로 조정
extracted_data = processor.run_from_meaningful_changes(
    meaningful_changes_file='...',
    raw_data_file='...',
    before_minutes=5,   # 늘림
    after_minutes=5     # 줄임
)
```

---

## 참고

- v1.2 스타일 사용 예제: `test_densitometer_v1_2_compatible.py`
- v1.3 스타일 사용 예제: `test_statistical_analysis.py`
- 전체 가이드: `STATISTICAL_ANALYSIS_GUIDE.md`
