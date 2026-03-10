"""
전처리 파이프라인 데이터 정합성 검증 테스트

outputs/ 디렉토리의 실제 결과 파일을 대상으로 데이터 무결성을 검증합니다.
pytest tests/test_data_integrity.py -v 로 실행합니다.
"""

import numpy as np
import pandas as pd
import pytest
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils import load_file

# PreprocessConfig를 직접 import하면 xlwings 등 무거운 의존성이 딸려옴.
# 테스트에 필요한 상수만 직접 정의한다.
N_ZONES = 11
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs')

# 출력 파일명
OUTPUT_ZONE_ANALYSIS = 'zone_analysis_results.xlsx'
OUTPUT_STATISTICAL_ANALYSIS = 'statistical_analysis_results.xlsx'
OUTPUT_MODEL_DATA = 'model_training_data.xlsx'
OUTPUT_3RD = '3rd_meaningful_changes.xlsx'
OUTPUT_DENSITOMETER = 'extracted_densitometer_data.xlsx'


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture(scope="module")
def zone_analysis_df():
    """zone_analysis_results.xlsx 로드"""
    path = os.path.join(OUTPUT_DIR, OUTPUT_ZONE_ANALYSIS)
    if not os.path.exists(path):
        pytest.skip(f"파일 없음: {path}")
    return load_file(path)


@pytest.fixture(scope="module")
def statistical_analysis_df():
    """statistical_analysis_results.xlsx 로드"""
    path = os.path.join(OUTPUT_DIR, OUTPUT_STATISTICAL_ANALYSIS)
    if not os.path.exists(path):
        pytest.skip(f"파일 없음: {path}")
    return load_file(path)


@pytest.fixture(scope="module")
def model_data_df():
    """model_training_data.xlsx 또는 model_test_data.xlsx 로드"""
    for fname in [OUTPUT_MODEL_DATA, 'model_test_data.xlsx']:
        path = os.path.join(OUTPUT_DIR, fname)
        if os.path.exists(path):
            return load_file(path)
    pytest.skip(f"모델 데이터 파일 없음: {OUTPUT_DIR}")


@pytest.fixture(scope="module")
def meaningful_changes_df():
    """3rd_meaningful_changes.xlsx 로드"""
    path = os.path.join(OUTPUT_DIR, OUTPUT_3RD)
    if not os.path.exists(path):
        pytest.skip(f"파일 없음: {path}")
    return load_file(path)


@pytest.fixture(scope="module")
def densitometer_df():
    """extracted_densitometer_data.xlsx 로드"""
    path = os.path.join(OUTPUT_DIR, OUTPUT_DENSITOMETER)
    if not os.path.exists(path):
        pytest.skip(f"파일 없음: {path}")
    return load_file(path)


# ============================================================================
# 1. Zone Analysis Results 정합성 검증
# ============================================================================

class TestZoneAnalysisIntegrity:
    """zone_analysis_results.xlsx 데이터 정합성 테스트"""

    def test_before_ratio_sum_equals_one(self, zone_analysis_df):
        """각 (group_id, zone_id)마다 div_*_before_ratio 합이 1이어야 한다"""
        df = zone_analysis_df
        n_divisions = int(df['n_divisions'].iloc[0])
        before_ratio_cols = [f'div_{i}_before_ratio' for i in range(1, n_divisions + 1)]

        for _, row in df.iterrows():
            ratio_sum = sum(row[col] for col in before_ratio_cols)
            assert abs(ratio_sum - 1.0) < 1e-6, (
                f"Group {row['group_id']}, Zone {row['zone_id']}: "
                f"before_ratio 합 = {ratio_sum:.6f} (expected 1.0)"
            )

    def test_after_ratio_sum_equals_one(self, zone_analysis_df):
        """각 (group_id, zone_id)마다 div_*_after_ratio 합이 1이어야 한다"""
        df = zone_analysis_df
        n_divisions = int(df['n_divisions'].iloc[0])
        after_ratio_cols = [f'div_{i}_after_ratio' for i in range(1, n_divisions + 1)]

        for _, row in df.iterrows():
            ratio_sum = sum(row[col] for col in after_ratio_cols)
            assert abs(ratio_sum - 1.0) < 1e-6, (
                f"Group {row['group_id']}, Zone {row['zone_id']}: "
                f"after_ratio 합 = {ratio_sum:.6f} (expected 1.0)"
            )

    def test_each_group_has_all_zones(self, zone_analysis_df):
        """각 group_id에 N_ZONES(11)개의 zone_id가 있어야 한다"""
        df = zone_analysis_df
        expected_zones = set(range(1, N_ZONES + 1))

        for group_id, group_df in df.groupby('group_id'):
            actual_zones = set(group_df['zone_id'].values)
            assert actual_zones == expected_zones, (
                f"Group {group_id}: zone_id = {sorted(actual_zones)}, "
                f"expected {sorted(expected_zones)}"
            )

    def test_usl_lsl_target_consistent_per_group(self, zone_analysis_df):
        """각 group 내에서 ucl, lcl, target 값이 모두 동일해야 한다"""
        df = zone_analysis_df

        for group_id, group_df in df.groupby('group_id'):
            assert group_df['ucl'].nunique() == 1, (
                f"Group {group_id}: ucl 값이 일관되지 않음 {group_df['ucl'].unique()}"
            )
            assert group_df['lcl'].nunique() == 1, (
                f"Group {group_id}: lcl 값이 일관되지 않음 {group_df['lcl'].unique()}"
            )
            # target은 NaN일 수 있으므로 NaN을 제외하고 비교
            target_vals = group_df['target'].dropna().unique()
            assert len(target_vals) <= 1, (
                f"Group {group_id}: target 값이 일관되지 않음 {target_vals}"
            )

    def test_ucl_greater_than_lcl(self, zone_analysis_df):
        """ucl > lcl 이어야 한다"""
        df = zone_analysis_df
        violations = df[df['ucl'] <= df['lcl']]
        assert len(violations) == 0, (
            f"ucl <= lcl인 행 {len(violations)}건 발견: "
            f"group_ids = {violations['group_id'].unique().tolist()}"
        )

    def test_ratio_change_equals_after_minus_before(self, zone_analysis_df):
        """div_*_ratio_change = div_*_after_ratio - div_*_before_ratio 이어야 한다"""
        df = zone_analysis_df
        n_divisions = int(df['n_divisions'].iloc[0])

        for i in range(1, n_divisions + 1):
            expected = df[f'div_{i}_after_ratio'] - df[f'div_{i}_before_ratio']
            actual = df[f'div_{i}_ratio_change']
            np.testing.assert_allclose(
                actual.values, expected.values, atol=1e-6,
                err_msg=f"div_{i}_ratio_change != after - before"
            )

    def test_before_count_sum_equals_total(self, zone_analysis_df):
        """div_*_before_count 합이 before_total과 일치해야 한다"""
        df = zone_analysis_df
        n_divisions = int(df['n_divisions'].iloc[0])
        count_cols = [f'div_{i}_before_count' for i in range(1, n_divisions + 1)]

        for _, row in df.iterrows():
            count_sum = sum(row[col] for col in count_cols)
            assert count_sum == row['before_total'], (
                f"Group {row['group_id']}, Zone {row['zone_id']}: "
                f"before_count 합 = {count_sum}, before_total = {row['before_total']}"
            )

    def test_after_count_sum_equals_total(self, zone_analysis_df):
        """div_*_after_count 합이 after_total과 일치해야 한다"""
        df = zone_analysis_df
        n_divisions = int(df['n_divisions'].iloc[0])
        count_cols = [f'div_{i}_after_count' for i in range(1, n_divisions + 1)]

        for _, row in df.iterrows():
            count_sum = sum(row[col] for col in count_cols)
            assert count_sum == row['after_total'], (
                f"Group {row['group_id']}, Zone {row['zone_id']}: "
                f"after_count 합 = {count_sum}, after_total = {row['after_total']}"
            )

    def test_no_zero_counts_in_any_division(self, zone_analysis_df):
        """모든 division의 before/after count가 0보다 커야 한다 (zone_analyzer 필터링 규칙)"""
        df = zone_analysis_df
        n_divisions = int(df['n_divisions'].iloc[0])

        for _, row in df.iterrows():
            for i in range(1, n_divisions + 1):
                assert row[f'div_{i}_before_count'] > 0, (
                    f"Group {row['group_id']}, Zone {row['zone_id']}: "
                    f"div_{i}_before_count = 0"
                )
                assert row[f'div_{i}_after_count'] > 0, (
                    f"Group {row['group_id']}, Zone {row['zone_id']}: "
                    f"div_{i}_after_count = 0"
                )

    def test_ratios_between_zero_and_one(self, zone_analysis_df):
        """모든 ratio 값이 [0, 1] 범위여야 한다"""
        df = zone_analysis_df
        n_divisions = int(df['n_divisions'].iloc[0])

        for prefix in ['before', 'after']:
            for i in range(1, n_divisions + 1):
                col = f'div_{i}_{prefix}_ratio'
                assert (df[col] >= 0).all(), f"{col}에 음수 값 존재"
                assert (df[col] <= 1).all(), f"{col}에 1 초과 값 존재"

    def test_mean_within_spec_range(self, zone_analysis_df):
        """before_mean, after_mean이 [lcl, ucl] 범위 내에 있어야 한다"""
        df = zone_analysis_df

        for _, row in df.iterrows():
            lcl, ucl = row['lcl'], row['ucl']
            if not np.isnan(row['before_mean']):
                assert lcl <= row['before_mean'] <= ucl, (
                    f"Group {row['group_id']}, Zone {row['zone_id']}: "
                    f"before_mean={row['before_mean']:.4f} not in [{lcl:.4f}, {ucl:.4f}]"
                )
            if not np.isnan(row['after_mean']):
                assert lcl <= row['after_mean'] <= ucl, (
                    f"Group {row['group_id']}, Zone {row['zone_id']}: "
                    f"after_mean={row['after_mean']:.4f} not in [{lcl:.4f}, {ucl:.4f}]"
                )


# ============================================================================
# 2. Statistical Analysis Results 정합성 검증
# ============================================================================

class TestStatisticalAnalysisIntegrity:
    """statistical_analysis_results.xlsx 데이터 정합성 테스트"""

    def test_low_mid_high_ratio_sum_equals_one(self, statistical_analysis_df):
        """before/after Low+Mid+High 비율 합이 1이어야 한다"""
        df = statistical_analysis_df

        for _, row in df.iterrows():
            if pd.isna(row.get('before_low_ratio')):
                continue

            before_sum = row['before_low_ratio'] + row['before_mid_ratio'] + row['before_high_ratio']
            assert abs(before_sum - 1.0) < 1e-6, (
                f"Group {row['group_id']}, Zone {row['zone_id']}: "
                f"before L+M+H = {before_sum:.6f}"
            )

            after_sum = row['after_low_ratio'] + row['after_mid_ratio'] + row['after_high_ratio']
            assert abs(after_sum - 1.0) < 1e-6, (
                f"Group {row['group_id']}, Zone {row['zone_id']}: "
                f"after L+M+H = {after_sum:.6f}"
            )

    def test_ratio_change_consistency(self, statistical_analysis_df):
        """ratio_change = after_ratio - before_ratio 이어야 한다"""
        df = statistical_analysis_df

        for level in ['low', 'mid', 'high']:
            before_col = f'before_{level}_ratio'
            after_col = f'after_{level}_ratio'
            change_col = f'{level}_ratio_change'

            valid = df[df[before_col].notna()]
            if len(valid) == 0:
                continue

            expected = valid[after_col] - valid[before_col]
            np.testing.assert_allclose(
                valid[change_col].values, expected.values, atol=1e-6,
                err_msg=f"{change_col} != {after_col} - {before_col}"
            )

    def test_mid_ratio_improved_flag(self, statistical_analysis_df):
        """mid_ratio_improved가 after_mid > before_mid 와 일치해야 한다"""
        df = statistical_analysis_df
        valid = df[df['before_mid_ratio'].notna()].copy()
        if len(valid) == 0:
            pytest.skip("분포 검정 결과 없음")

        expected = valid['after_mid_ratio'] > valid['before_mid_ratio']
        actual = valid['mid_ratio_improved'].astype(bool)
        mismatches = (expected != actual).sum()
        assert mismatches == 0, (
            f"mid_ratio_improved 불일치 {mismatches}건"
        )

    def test_pvalue_range(self, statistical_analysis_df):
        """p-value는 [0, 1] 범위여야 한다"""
        df = statistical_analysis_df
        pvalue_cols = ['ttest_pvalue', 'mannwhitney_pvalue', 'ks_test_pvalue', 'chi2_pvalue']

        for col in pvalue_cols:
            if col not in df.columns:
                continue
            valid = df[col].dropna()
            if len(valid) == 0:
                continue
            assert (valid >= 0).all() and (valid <= 1).all(), (
                f"{col}: 범위 벗어남 [{valid.min():.6f}, {valid.max():.6f}]"
            )

    def test_cohens_d_has_effect_category(self, statistical_analysis_df):
        """cohens_d가 있으면 effect_size_category도 존재해야 한다"""
        df = statistical_analysis_df
        has_d = df['cohens_d'].notna()
        has_cat = df['effect_size_category'].notna()
        mismatches = (has_d & ~has_cat).sum()
        assert mismatches == 0, (
            f"cohens_d는 있지만 effect_size_category가 없는 행 {mismatches}건"
        )

    def test_control_type_values(self, statistical_analysis_df):
        """control_type은 'controlled' 또는 'no_control'만 허용"""
        df = statistical_analysis_df
        valid_types = {'controlled', 'no_control'}
        actual_types = set(df['control_type'].unique())
        assert actual_types.issubset(valid_types), (
            f"허용되지 않은 control_type: {actual_types - valid_types}"
        )


# ============================================================================
# 3. Model Data 정합성 검증
# ============================================================================

class TestModelDataIntegrity:
    """model_training_data.xlsx 데이터 정합성 테스트"""

    def test_before_ratio_sum_equals_one(self, model_data_df):
        """before_ratio 합이 1이어야 한다"""
        df = model_data_df
        ratio_cols = [col for col in df.columns if col.startswith('before_ratio_')]
        if not ratio_cols:
            pytest.skip("before_ratio 칼럼 없음")

        for _, row in df.iterrows():
            ratio_sum = sum(row[col] for col in ratio_cols)
            assert abs(ratio_sum - 1.0) < 1e-6, (
                f"Group {row['group_id']}, Zone {row['zone_id']}: "
                f"before_ratio 합 = {ratio_sum:.6f}"
            )

    def test_after_ratio_sum_equals_one(self, model_data_df):
        """after_ratio 합이 1이어야 한다"""
        df = model_data_df
        ratio_cols = [col for col in df.columns if col.startswith('after_ratio_')]
        if not ratio_cols:
            pytest.skip("after_ratio 칼럼 없음")

        for _, row in df.iterrows():
            ratio_sum = sum(row[col] for col in ratio_cols)
            assert abs(ratio_sum - 1.0) < 1e-6, (
                f"Group {row['group_id']}, Zone {row['zone_id']}: "
                f"after_ratio 합 = {ratio_sum:.6f}"
            )

    def test_clr_sum_equals_zero(self, model_data_df):
        """CLR 변환값의 합은 0이어야 한다 (CLR 속성)"""
        df = model_data_df
        before_clr_cols = sorted([col for col in df.columns if col.startswith('before_CLR_')])
        after_clr_cols = sorted([col for col in df.columns if col.startswith('after_CLR_')])
        current_clr_cols = sorted([col for col in df.columns if col.startswith('current_CLR_')])

        for clr_cols, label in [
            (before_clr_cols, 'before_CLR'),
            (after_clr_cols, 'after_CLR'),
            (current_clr_cols, 'current_CLR'),
        ]:
            if not clr_cols:
                continue
            for _, row in df.iterrows():
                clr_sum = sum(row[col] for col in clr_cols)
                assert abs(clr_sum) < 1e-4, (
                    f"Group {row['group_id']}, Zone {row['zone_id']}: "
                    f"{label} 합 = {clr_sum:.6f} (expected 0)"
                )

    def test_diff_clr_equals_after_minus_before(self, model_data_df):
        """diff_CLR = after_CLR - before_CLR 이어야 한다"""
        df = model_data_df
        diff_cols = sorted([col for col in df.columns if col.startswith('diff_CLR_')])
        if not diff_cols:
            pytest.skip("diff_CLR 칼럼 없음")

        n = len(diff_cols)
        for _, row in df.iterrows():
            for i in range(1, n + 1):
                expected = row[f'after_CLR_{i}'] - row[f'before_CLR_{i}']
                actual = row[f'diff_CLR_{i}']
                assert abs(actual - expected) < 1e-6, (
                    f"Group {row['group_id']}, Zone {row['zone_id']}: "
                    f"diff_CLR_{i} = {actual:.6f}, expected {expected:.6f}"
                )

    def test_each_group_has_all_zones(self, model_data_df):
        """각 group_id에 N_ZONES(11)개의 zone_id가 있어야 한다"""
        df = model_data_df
        expected_zones = set(range(1, N_ZONES + 1))

        for group_id, group_df in df.groupby('group_id'):
            actual_zones = set(group_df['zone_id'].values)
            assert actual_zones == expected_zones, (
                f"Group {group_id}: zone_id = {sorted(actual_zones)}, "
                f"expected {sorted(expected_zones)}"
            )

    def test_no_nan_in_features(self, model_data_df):
        """입출력 특성 칼럼에 NaN이 없어야 한다"""
        df = model_data_df
        feature_patterns = ['current_CLR_', 'diff_CLR_', 'delta_GV_', 'delta_RPM']
        feature_cols = [
            col for col in df.columns
            if any(col.startswith(p) for p in feature_patterns)
        ]

        for col in feature_cols:
            nan_count = df[col].isna().sum()
            assert nan_count == 0, (
                f"{col}: NaN {nan_count}건 발견"
            )

    def test_is_edge_only_at_zone_1_and_11(self, model_data_df):
        """is_edge=1은 zone_id 1과 11에서만 True여야 한다"""
        df = model_data_df
        if 'is_edge' not in df.columns:
            pytest.skip("is_edge 칼럼 없음")

        edge_rows = df[df['is_edge'] == 1]
        non_edge_zones = edge_rows[~edge_rows['zone_id'].isin([1, N_ZONES])]
        assert len(non_edge_zones) == 0, (
            f"zone_id {non_edge_zones['zone_id'].unique().tolist()}에서 is_edge=1 발견"
        )

        # 역방향: zone 1, 11은 모두 is_edge=1이어야 함
        edge_zone_rows = df[df['zone_id'].isin([1, N_ZONES])]
        non_flagged = edge_zone_rows[edge_zone_rows['is_edge'] != 1]
        assert len(non_flagged) == 0, (
            f"zone_id 1 또는 {N_ZONES}인데 is_edge=0인 행 {len(non_flagged)}건"
        )


# ============================================================================
# 4. Meaningful Changes 정합성 검증
# ============================================================================

class TestMeaningfulChangesIntegrity:
    """3rd_meaningful_changes.xlsx 데이터 정합성 테스트"""

    def test_group_id_unique(self, meaningful_changes_df):
        """group_id는 유일해야 한다"""
        df = meaningful_changes_df
        duplicates = df[df['group_id'].duplicated()]
        assert len(duplicates) == 0, (
            f"중복 group_id: {duplicates['group_id'].tolist()}"
        )

    def test_start_before_end(self, meaningful_changes_df):
        """start_time < end_time 이어야 한다"""
        df = meaningful_changes_df
        violations = df[df['start_time'] >= df['end_time']]
        assert len(violations) == 0, (
            f"start_time >= end_time인 group_id: {violations['group_id'].tolist()}"
        )

    def test_ucl_lcl_present(self, meaningful_changes_df):
        """UCL, LCL 칼럼이 존재하고 NaN이 아닌 행이 있어야 한다"""
        df = meaningful_changes_df
        assert 'UCL' in df.columns, "UCL 칼럼 없음"
        assert 'LCL' in df.columns, "LCL 칼럼 없음"
        assert df['UCL'].notna().sum() > 0, "UCL 값이 모두 NaN"
        assert df['LCL'].notna().sum() > 0, "LCL 값이 모두 NaN"

    def test_ucl_greater_than_lcl(self, meaningful_changes_df):
        """UCL > LCL 이어야 한다"""
        df = meaningful_changes_df
        valid = df[df['UCL'].notna() & df['LCL'].notna()]
        violations = valid[valid['UCL'] <= valid['LCL']]
        assert len(violations) == 0, (
            f"UCL <= LCL인 group_id: {violations['group_id'].tolist()}"
        )


# ============================================================================
# 5. Densitometer Data 정합성 검증
# ============================================================================

class TestDensitometerDataIntegrity:
    """extracted_densitometer_data.xlsx 데이터 정합성 테스트"""

    def test_required_columns_exist(self, densitometer_df):
        """필수 칼럼이 존재해야 한다"""
        required = ['group_id', 'control_type', 'before/after']
        for col in required:
            assert col in densitometer_df.columns, f"필수 칼럼 없음: {col}"

    def test_control_type_values(self, densitometer_df):
        """control_type은 'controlled' 또는 'no_control'만 허용"""
        valid_types = {'controlled', 'no_control'}
        actual = set(densitometer_df['control_type'].unique())
        assert actual.issubset(valid_types), (
            f"허용되지 않은 control_type: {actual - valid_types}"
        )

    def test_before_after_values(self, densitometer_df):
        """before/after 칼럼은 'before' 또는 'after'만 허용"""
        valid = {'before', 'after'}
        actual = set(densitometer_df['before/after'].unique())
        assert actual.issubset(valid), (
            f"허용되지 않은 before/after 값: {actual - valid}"
        )

    def test_each_group_has_before_and_after(self, densitometer_df):
        """각 group_id에 before와 after 데이터가 모두 존재해야 한다"""
        df = densitometer_df
        for group_id, group_df in df.groupby('group_id'):
            phases = set(group_df['before/after'].unique())
            assert 'before' in phases, (
                f"Group {group_id}: before 데이터 없음"
            )
            assert 'after' in phases, (
                f"Group {group_id}: after 데이터 없음"
            )


# ============================================================================
# 6. 파이프라인 간 Cross-Validation
# ============================================================================

class TestCrossFileConsistency:
    """파일 간 데이터 일관성 검증"""

    def test_zone_analysis_groups_subset_of_meaningful(
        self, zone_analysis_df, meaningful_changes_df
    ):
        """zone_analysis의 group_id는 meaningful_changes에 존재해야 한다"""
        zone_groups = set(zone_analysis_df['group_id'].unique())
        meaningful_groups = set(meaningful_changes_df['group_id'].unique())
        missing = zone_groups - meaningful_groups
        assert len(missing) == 0, (
            f"meaningful_changes에 없는 group_id: {sorted(missing)}"
        )

    def test_model_data_groups_subset_of_zone_analysis(
        self, model_data_df, zone_analysis_df
    ):
        """model_data의 group_id는 zone_analysis에 존재해야 한다"""
        model_groups = set(model_data_df['group_id'].unique())
        zone_groups = set(zone_analysis_df['group_id'].unique())
        missing = model_groups - zone_groups
        assert len(missing) == 0, (
            f"zone_analysis에 없는 group_id: {sorted(missing)}"
        )

    def test_ucl_lcl_consistent_across_files(
        self, zone_analysis_df, meaningful_changes_df
    ):
        """zone_analysis와 meaningful_changes의 UCL/LCL이 일치해야 한다"""
        za = zone_analysis_df.drop_duplicates('group_id')[['group_id', 'ucl', 'lcl']]
        mc = meaningful_changes_df[meaningful_changes_df['UCL'].notna()][['group_id', 'UCL', 'LCL']]

        merged = za.merge(mc, on='group_id', how='inner')
        if len(merged) == 0:
            pytest.skip("공통 group_id 없음")

        ucl_mismatch = merged[~np.isclose(merged['ucl'], merged['UCL'])]
        assert len(ucl_mismatch) == 0, (
            f"UCL 불일치 group_id: {ucl_mismatch['group_id'].tolist()}"
        )

        lcl_mismatch = merged[~np.isclose(merged['lcl'], merged['LCL'])]
        assert len(lcl_mismatch) == 0, (
            f"LCL 불일치 group_id: {lcl_mismatch['group_id'].tolist()}"
        )
