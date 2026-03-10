"""
통계 분석 모듈 v2.0
제어 효과 검증을 위한 통계적 분석 기능

v2.0 변경사항:
- Low/Mid/High 분포 비율 변화 검정 추가 (카이제곱 검정)
- 종합 판단에 분포 유의성 반영
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, Tuple, Optional
import logging


class StatisticalAnalyzer:
    """통계적 유의성 검정 클래스"""

    def __init__(self, logger: logging.Logger = None):
        """
        Parameters:
        -----------
        logger : logging.Logger
            로거 객체
        """
        self.logger = logger or logging.getLogger('coating_preprocessor.statistics')

    def analyze_control_effect(
        self,
        before_data: np.ndarray,
        after_data: np.ndarray,
        ucl: float,
        lcl: float,
        target: Optional[float] = None,
        alpha: float = 0.05
    ) -> Dict:
        """
        제어 효과 종합 분석

        Parameters:
        -----------
        before_data : np.ndarray
            제어 전 데이터
        after_data : np.ndarray
            제어 후 데이터
        ucl : float
            Upper Control Limit
        lcl : float
            Lower Control Limit
        target : float, optional
            Target 값
        alpha : float
            유의수준 (기본값: 0.05 = 95% 신뢰도)

        Returns:
        --------
        Dict
            통계 분석 결과
        """
        # 데이터 검증
        before_data = self._clean_data(before_data)
        after_data = self._clean_data(after_data)

        if len(before_data) == 0 or len(after_data) == 0:
            return self._empty_result()

        # 기본 통계량
        basic_stats = self.compute_basic_statistics(before_data, after_data)

        # p-value 검정
        p_values = self.compute_p_values(before_data, after_data)

        # Effect Size
        effect_sizes = self.compute_effect_sizes(before_data, after_data)

        # 분산 분석
        variance_analysis = self.compute_variance_analysis(before_data, after_data)

        # 공정 능력 지수
        cpk_analysis = self.compute_cpk_analysis(
            before_data, after_data, ucl, lcl, target
        )

        # Low/Mid/High 분포 비율 변화 검정 (카이제곱)
        distribution_test = self.compute_distribution_test(
            before_data, after_data, ucl, lcl
        )

        # 종합 판단
        summary = self._create_summary(
            basic_stats, p_values, effect_sizes,
            variance_analysis, cpk_analysis, distribution_test, alpha
        )

        # 결과 통합
        result = {
            **basic_stats,
            **p_values,
            **effect_sizes,
            **variance_analysis,
            **cpk_analysis,
            **distribution_test,
            **summary
        }

        return result

    def compute_basic_statistics(
        self,
        before_data: np.ndarray,
        after_data: np.ndarray
    ) -> Dict:
        """
        기본 통계량 계산

        Parameters:
        -----------
        before_data : np.ndarray
            제어 전 데이터
        after_data : np.ndarray
            제어 후 데이터

        Returns:
        --------
        Dict
            기본 통계량
        """
        return {
            'before_count': len(before_data),
            'after_count': len(after_data),
            'before_mean': float(np.mean(before_data)),
            'after_mean': float(np.mean(after_data)),
            'before_std': float(np.std(before_data, ddof=1)),
            'after_std': float(np.std(after_data, ddof=1)),
            'before_median': float(np.median(before_data)),
            'after_median': float(np.median(after_data)),
            'before_min': float(np.min(before_data)),
            'after_min': float(np.min(after_data)),
            'before_max': float(np.max(before_data)),
            'after_max': float(np.max(after_data)),
            'mean_change': float(np.mean(after_data) - np.mean(before_data)),
            'mean_change_percent': float((np.mean(after_data) - np.mean(before_data)) / np.mean(before_data) * 100) if np.mean(before_data) != 0 else 0,
        }

    def compute_p_values(
        self,
        before_data: np.ndarray,
        after_data: np.ndarray
    ) -> Dict:
        """
        p-value 검정 (여러 방법)

        Parameters:
        -----------
        before_data : np.ndarray
            제어 전 데이터
        after_data : np.ndarray
            제어 후 데이터

        Returns:
        --------
        Dict
            p-value 검정 결과
        """
        result = {}

        # 1. t-test (모수 검정)
        try:
            t_stat, t_pvalue = stats.ttest_ind(before_data, after_data)
            result['ttest_statistic'] = float(t_stat)
            result['ttest_pvalue'] = float(t_pvalue)
        except Exception as e:
            self.logger.warning(f"t-test 계산 오류: {e}")
            result['ttest_statistic'] = np.nan
            result['ttest_pvalue'] = np.nan

        # 2. Mann-Whitney U test (비모수 검정) - 추천
        try:
            u_stat, u_pvalue = stats.mannwhitneyu(
                before_data, after_data, alternative='two-sided'
            )
            result['mannwhitney_statistic'] = float(u_stat)
            result['mannwhitney_pvalue'] = float(u_pvalue)
        except Exception as e:
            self.logger.warning(f"Mann-Whitney U test 계산 오류: {e}")
            result['mannwhitney_statistic'] = np.nan
            result['mannwhitney_pvalue'] = np.nan

        # 3. Kolmogorov-Smirnov test (분포 차이 검정)
        try:
            ks_stat, ks_pvalue = stats.ks_2samp(before_data, after_data)
            result['ks_test_statistic'] = float(ks_stat)
            result['ks_test_pvalue'] = float(ks_pvalue)
        except Exception as e:
            self.logger.warning(f"KS test 계산 오류: {e}")
            result['ks_test_statistic'] = np.nan
            result['ks_test_pvalue'] = np.nan

        return result

    def compute_effect_sizes(
        self,
        before_data: np.ndarray,
        after_data: np.ndarray
    ) -> Dict:
        """
        Effect Size 계산

        Parameters:
        -----------
        before_data : np.ndarray
            제어 전 데이터
        after_data : np.ndarray
            제어 후 데이터

        Returns:
        --------
        Dict
            Effect size 결과
        """
        result = {}

        # Cohen's d
        mean_diff = np.mean(after_data) - np.mean(before_data)
        pooled_std = np.sqrt(
            ((len(before_data) - 1) * np.var(before_data, ddof=1) +
             (len(after_data) - 1) * np.var(after_data, ddof=1)) /
            (len(before_data) + len(after_data) - 2)
        )

        if pooled_std != 0:
            cohens_d = mean_diff / pooled_std
            result['cohens_d'] = float(cohens_d)

            # Cohen's d 해석
            abs_d = abs(cohens_d)
            if abs_d < 0.2:
                result['effect_size_category'] = 'negligible'
            elif abs_d < 0.5:
                result['effect_size_category'] = 'small'
            elif abs_d < 0.8:
                result['effect_size_category'] = 'medium'
            else:
                result['effect_size_category'] = 'large'
        else:
            result['cohens_d'] = 0.0
            result['effect_size_category'] = 'negligible'

        # 평균 차이 비율
        if np.mean(before_data) != 0:
            result['mean_difference_ratio'] = float(
                (np.mean(after_data) - np.mean(before_data)) / np.mean(before_data)
            )
        else:
            result['mean_difference_ratio'] = 0.0

        return result

    def compute_variance_analysis(
        self,
        before_data: np.ndarray,
        after_data: np.ndarray
    ) -> Dict:
        """
        분산 분석

        Parameters:
        -----------
        before_data : np.ndarray
            제어 전 데이터
        after_data : np.ndarray
            제어 후 데이터

        Returns:
        --------
        Dict
            분산 분석 결과
        """
        result = {}

        before_var = np.var(before_data, ddof=1)
        after_var = np.var(after_data, ddof=1)

        # 분산 비율
        if before_var != 0:
            result['variance_ratio'] = float(after_var / before_var)
            result['variance_reduced'] = after_var < before_var
        else:
            result['variance_ratio'] = np.nan
            result['variance_reduced'] = False

        # F-test (분산 동질성 검정)
        try:
            # F = var1 / var2 (큰 분산 / 작은 분산)
            if before_var > after_var:
                f_stat = before_var / after_var if after_var != 0 else np.nan
                df1, df2 = len(before_data) - 1, len(after_data) - 1
            else:
                f_stat = after_var / before_var if before_var != 0 else np.nan
                df1, df2 = len(after_data) - 1, len(before_data) - 1

            if not np.isnan(f_stat):
                f_pvalue = 2 * min(
                    stats.f.cdf(f_stat, df1, df2),
                    1 - stats.f.cdf(f_stat, df1, df2)
                )
                result['f_test_statistic'] = float(f_stat)
                result['f_test_pvalue'] = float(f_pvalue)
            else:
                result['f_test_statistic'] = np.nan
                result['f_test_pvalue'] = np.nan
        except Exception as e:
            self.logger.warning(f"F-test 계산 오류: {e}")
            result['f_test_statistic'] = np.nan
            result['f_test_pvalue'] = np.nan

        return result

    def compute_cpk_analysis(
        self,
        before_data: np.ndarray,
        after_data: np.ndarray,
        ucl: float,
        lcl: float,
        target: Optional[float] = None
    ) -> Dict:
        """
        공정 능력 지수 (Cpk) 분석

        Parameters:
        -----------
        before_data : np.ndarray
            제어 전 데이터
        after_data : np.ndarray
            제어 후 데이터
        ucl : float
            Upper Control Limit (USL)
        lcl : float
            Lower Control Limit (LSL)
        target : float, optional
            Target 값

        Returns:
        --------
        Dict
            Cpk 분석 결과
        """
        result = {}

        # Before Cpk
        before_cpk = self._calculate_cpk(before_data, ucl, lcl, target)
        result['before_cpk'] = before_cpk

        # After Cpk
        after_cpk = self._calculate_cpk(after_data, ucl, lcl, target)
        result['after_cpk'] = after_cpk

        # Cpk 개선도
        if not np.isnan(before_cpk) and not np.isnan(after_cpk):
            result['cpk_change'] = float(after_cpk - before_cpk)
            result['cpk_improved'] = after_cpk > before_cpk
            result['cpk_change_percent'] = float(
                (after_cpk - before_cpk) / before_cpk * 100
            ) if before_cpk != 0 else 0
        else:
            result['cpk_change'] = np.nan
            result['cpk_improved'] = False
            result['cpk_change_percent'] = np.nan

        # Cpk 평가
        result['before_cpk_grade'] = self._grade_cpk(before_cpk)
        result['after_cpk_grade'] = self._grade_cpk(after_cpk)

        return result

    def compute_distribution_test(
        self,
        before_data: np.ndarray,
        after_data: np.ndarray,
        ucl: float,
        lcl: float,
        alpha: float = 0.05
    ) -> Dict:
        """
        Low/Mid/High 분포 비율 변화의 유의성 검정

        UCL~LCL 범위를 3등분하여 Low/Mid/High 구간을 정의하고,
        제어 전/후 데이터가 각 구간에 몇 개씩 분포하는지 count한 뒤
        카이제곱 검정으로 분포 변화가 통계적으로 유의미한지 검증합니다.

        이 검정이 필요한 이유:
        - 기존 t-test/Mann-Whitney는 평균/중앙값 변화만 검정
        - MBRL 모델의 목표는 "Low/Mid/High 비율 변화 예측"이므로
          비율 자체의 변화가 유의미한지 직접 검정해야 함

        Parameters:
        -----------
        before_data : np.ndarray
            제어 전 밀도계 raw 데이터
        after_data : np.ndarray
            제어 후 밀도계 raw 데이터
        ucl : float
            Upper Control Limit
        lcl : float
            Lower Control Limit
        alpha : float
            유의수준 (기본값: 0.05)

        Returns:
        --------
        Dict
            분포 검정 결과 (비율, 카이제곱 통계량, p-value 등)
        """
        result = {}

        spec_range = ucl - lcl
        if spec_range <= 0:
            self.logger.warning(f"UCL({ucl}) <= LCL({lcl}): 분포 검정 불가")
            return self._empty_distribution_result()

        # 3등분 경계: Low / Mid / High
        boundary_low_mid = lcl + spec_range / 3.0
        boundary_mid_high = lcl + 2.0 * spec_range / 3.0

        # 구간 분류 함수
        def classify(data: np.ndarray) -> np.ndarray:
            """0=Low, 1=Mid, 2=High"""
            labels = np.zeros(len(data), dtype=int)
            labels[data >= boundary_low_mid] = 1
            labels[data >= boundary_mid_high] = 2
            return labels

        before_labels = classify(before_data)
        after_labels = classify(after_data)

        # 각 구간별 count
        before_counts = np.array([
            np.sum(before_labels == 0),
            np.sum(before_labels == 1),
            np.sum(before_labels == 2),
        ])
        after_counts = np.array([
            np.sum(after_labels == 0),
            np.sum(after_labels == 1),
            np.sum(after_labels == 2),
        ])

        before_total = max(len(before_data), 1)
        after_total = max(len(after_data), 1)

        # 비율 계산
        before_ratios = before_counts / before_total
        after_ratios = after_counts / after_total

        result['before_low_ratio'] = float(before_ratios[0])
        result['before_mid_ratio'] = float(before_ratios[1])
        result['before_high_ratio'] = float(before_ratios[2])
        result['after_low_ratio'] = float(after_ratios[0])
        result['after_mid_ratio'] = float(after_ratios[1])
        result['after_high_ratio'] = float(after_ratios[2])

        # 비율 변화량
        result['low_ratio_change'] = float(after_ratios[0] - before_ratios[0])
        result['mid_ratio_change'] = float(after_ratios[1] - before_ratios[1])
        result['high_ratio_change'] = float(after_ratios[2] - before_ratios[2])

        # Mid 비율 개선 여부
        result['mid_ratio_improved'] = bool(after_ratios[1] > before_ratios[1])

        # 카이제곱 검정: 2×3 분할표 (before/after × Low/Mid/High)
        contingency_table = np.array([before_counts, after_counts])

        # 모든 셀이 0이면 검정 불가
        if contingency_table.sum() == 0 or np.all(contingency_table.sum(axis=0) == 0):
            result['chi2_statistic'] = np.nan
            result['chi2_pvalue'] = np.nan
            result['chi2_dof'] = np.nan
            result['distribution_significant'] = False
            return result

        # 기대빈도가 5 미만인 셀이 있으면 Fisher exact 대신 경고 후 진행
        try:
            chi2_stat, chi2_pvalue, dof, expected = stats.chi2_contingency(
                contingency_table
            )
            result['chi2_statistic'] = float(chi2_stat)
            result['chi2_pvalue'] = float(chi2_pvalue)
            result['chi2_dof'] = int(dof)
            result['distribution_significant'] = bool(chi2_pvalue < alpha)

            # 기대빈도 경고
            min_expected = expected.min()
            if min_expected < 5:
                self.logger.warning(
                    f"카이제곱 검정: 최소 기대빈도={min_expected:.1f} < 5. "
                    f"데이터 수가 적어 결과 해석에 주의 필요."
                )
                result['chi2_low_expected_warning'] = True
            else:
                result['chi2_low_expected_warning'] = False

        except Exception as e:
            self.logger.warning(f"카이제곱 검정 오류: {e}")
            result['chi2_statistic'] = np.nan
            result['chi2_pvalue'] = np.nan
            result['chi2_dof'] = np.nan
            result['distribution_significant'] = False
            result['chi2_low_expected_warning'] = True

        # Cramér's V (효과 크기 — 카이제곱 검정의 실질적 유의성)
        n_total = contingency_table.sum()
        k = min(contingency_table.shape) - 1  # min(rows, cols) - 1
        if n_total > 0 and k > 0 and not np.isnan(result.get('chi2_statistic', np.nan)):
            cramers_v = np.sqrt(result['chi2_statistic'] / (n_total * k))
            result['cramers_v'] = float(cramers_v)
            # Cramér's V 해석: <0.1 negligible, <0.3 small, <0.5 medium, ≥0.5 large
            if cramers_v < 0.1:
                result['distribution_effect_size'] = 'negligible'
            elif cramers_v < 0.3:
                result['distribution_effect_size'] = 'small'
            elif cramers_v < 0.5:
                result['distribution_effect_size'] = 'medium'
            else:
                result['distribution_effect_size'] = 'large'
        else:
            result['cramers_v'] = np.nan
            result['distribution_effect_size'] = 'unknown'

        return result

    def _empty_distribution_result(self) -> Dict:
        """분포 검정 빈 결과"""
        return {
            'before_low_ratio': np.nan,
            'before_mid_ratio': np.nan,
            'before_high_ratio': np.nan,
            'after_low_ratio': np.nan,
            'after_mid_ratio': np.nan,
            'after_high_ratio': np.nan,
            'low_ratio_change': np.nan,
            'mid_ratio_change': np.nan,
            'high_ratio_change': np.nan,
            'mid_ratio_improved': False,
            'chi2_statistic': np.nan,
            'chi2_pvalue': np.nan,
            'chi2_dof': np.nan,
            'distribution_significant': False,
            'chi2_low_expected_warning': True,
            'cramers_v': np.nan,
            'distribution_effect_size': 'unknown',
        }

    def _calculate_cpk(
        self,
        data: np.ndarray,
        usl: float,
        lsl: float,
        target: Optional[float] = None
    ) -> float:
        """
        Cpk 계산

        Parameters:
        -----------
        data : np.ndarray
            데이터
        usl : float
            Upper Specification Limit
        lsl : float
            Lower Specification Limit
        target : float, optional
            Target 값

        Returns:
        --------
        float
            Cpk 값
        """
        if len(data) == 0:
            return np.nan

        mean = np.mean(data)
        std = np.std(data, ddof=1)

        if std == 0:
            return np.nan

        # Cpu: Upper Process Capability
        cpu = (usl - mean) / (3 * std)

        # Cpl: Lower Process Capability
        cpl = (mean - lsl) / (3 * std)

        # Cpk: Minimum of Cpu and Cpl
        cpk = min(cpu, cpl)

        return float(cpk)

    def _grade_cpk(self, cpk: float) -> str:
        """
        Cpk 등급 평가

        Parameters:
        -----------
        cpk : float
            Cpk 값

        Returns:
        --------
        str
            등급
        """
        if np.isnan(cpk):
            return 'unknown'
        elif cpk < 1.0:
            return 'poor'
        elif cpk < 1.33:
            return 'fair'
        elif cpk < 1.67:
            return 'good'
        else:
            return 'excellent'

    def _create_summary(
        self,
        basic_stats: Dict,
        p_values: Dict,
        effect_sizes: Dict,
        variance_analysis: Dict,
        cpk_analysis: Dict,
        distribution_test: Dict,
        alpha: float = 0.05
    ) -> Dict:
        """
        종합 판단 생성

        Parameters:
        -----------
        basic_stats : Dict
            기본 통계량
        p_values : Dict
            p-value 검정 결과
        effect_sizes : Dict
            Effect size 결과
        variance_analysis : Dict
            분산 분석 결과
        cpk_analysis : Dict
            Cpk 분석 결과
        distribution_test : Dict
            Low/Mid/High 분포 비율 변화 검정 결과
        alpha : float
            유의수준

        Returns:
        --------
        Dict
            종합 판단
        """
        summary = {}

        # 통계적 유의성 (Mann-Whitney U test 기준)
        mannwhitney_pvalue = p_values.get('mannwhitney_pvalue', np.nan)
        if not np.isnan(mannwhitney_pvalue):
            summary['statistically_significant'] = mannwhitney_pvalue < alpha
            summary['significance_level'] = '***' if mannwhitney_pvalue < 0.001 else \
                                            '**' if mannwhitney_pvalue < 0.01 else \
                                            '*' if mannwhitney_pvalue < 0.05 else 'ns'
        else:
            summary['statistically_significant'] = False
            summary['significance_level'] = 'unknown'

        # 실질적 유의성 (Effect size 기준)
        effect_category = effect_sizes.get('effect_size_category', 'negligible')
        summary['practically_significant'] = effect_category in ['medium', 'large']

        # 분산 안정성 개선
        variance_reduced = variance_analysis.get('variance_reduced', False)
        summary['variance_improved'] = variance_reduced

        # Cpk 개선
        cpk_improved = cpk_analysis.get('cpk_improved', False)
        summary['cpk_improved'] = cpk_improved

        # 분포 비율 유의성 (카이제곱 검정)
        dist_significant = distribution_test.get('distribution_significant', False)
        summary['distribution_significant'] = dist_significant

        # Mid 비율 개선 여부
        mid_improved = distribution_test.get('mid_ratio_improved', False)
        summary['mid_ratio_improved'] = mid_improved

        # 종합 판단: 제어 효과 있음
        # 조건: (통계적 유의 OR 실질적 유의 OR 분포 유의) AND (분산 개선 OR Cpk 개선 OR Mid 개선)
        summary['control_effective'] = (
            (summary['statistically_significant'] or
             summary['practically_significant'] or
             summary['distribution_significant']) and
            (summary['variance_improved'] or
             summary['cpk_improved'] or
             summary['mid_ratio_improved'])
        )

        return summary

    def _clean_data(self, data: np.ndarray) -> np.ndarray:
        """
        데이터 정제 (NaN, Inf 제거)

        Parameters:
        -----------
        data : np.ndarray
            원본 데이터

        Returns:
        --------
        np.ndarray
            정제된 데이터
        """
        data = np.array(data).flatten()
        data = data[~np.isnan(data)]
        data = data[~np.isinf(data)]
        return data

    def _empty_result(self) -> Dict:
        """
        빈 결과 반환

        Returns:
        --------
        Dict
            빈 통계 결과
        """
        return {
            'before_count': 0,
            'after_count': 0,
            'before_mean': np.nan,
            'after_mean': np.nan,
            'before_std': np.nan,
            'after_std': np.nan,
            'before_median': np.nan,
            'after_median': np.nan,
            'before_min': np.nan,
            'after_min': np.nan,
            'before_max': np.nan,
            'after_max': np.nan,
            'mean_change': np.nan,
            'mean_change_percent': np.nan,
            'ttest_statistic': np.nan,
            'ttest_pvalue': np.nan,
            'mannwhitney_statistic': np.nan,
            'mannwhitney_pvalue': np.nan,
            'ks_test_statistic': np.nan,
            'ks_test_pvalue': np.nan,
            'cohens_d': np.nan,
            'effect_size_category': 'unknown',
            'mean_difference_ratio': np.nan,
            'variance_ratio': np.nan,
            'variance_reduced': False,
            'f_test_statistic': np.nan,
            'f_test_pvalue': np.nan,
            'before_cpk': np.nan,
            'after_cpk': np.nan,
            'cpk_change': np.nan,
            'cpk_improved': False,
            'cpk_change_percent': np.nan,
            'before_cpk_grade': 'unknown',
            'after_cpk_grade': 'unknown',
            'before_low_ratio': np.nan,
            'before_mid_ratio': np.nan,
            'before_high_ratio': np.nan,
            'after_low_ratio': np.nan,
            'after_mid_ratio': np.nan,
            'after_high_ratio': np.nan,
            'low_ratio_change': np.nan,
            'mid_ratio_change': np.nan,
            'high_ratio_change': np.nan,
            'mid_ratio_improved': False,
            'chi2_statistic': np.nan,
            'chi2_pvalue': np.nan,
            'chi2_dof': np.nan,
            'distribution_significant': False,
            'chi2_low_expected_warning': True,
            'cramers_v': np.nan,
            'distribution_effect_size': 'unknown',
            'statistically_significant': False,
            'significance_level': 'unknown',
            'practically_significant': False,
            'variance_improved': False,
            'distribution_significant': False,
            'mid_ratio_improved': False,
            'control_effective': False
        }

    def compare_control_vs_no_control(
        self,
        control_result: Dict,
        no_control_result: Dict
    ) -> Dict:
        """
        제어 구간 vs 비제어 구간 효과 비교

        Parameters:
        -----------
        control_result : Dict
            제어 구간 분석 결과
        no_control_result : Dict
            비제어 구간 분석 결과

        Returns:
        --------
        Dict
            비교 분석 결과
        """
        comparison = {}

        # 평균 변화량 비교
        control_mean_change = control_result.get('mean_change', 0)
        no_control_mean_change = no_control_result.get('mean_change', 0)

        comparison['control_mean_change'] = control_mean_change
        comparison['no_control_mean_change'] = no_control_mean_change
        comparison['net_control_effect'] = control_mean_change - no_control_mean_change

        # 통계적 유의성 비교
        comparison['control_significant'] = control_result.get('statistically_significant', False)
        comparison['no_control_significant'] = no_control_result.get('statistically_significant', False)

        # Effect size 비교
        control_cohens_d = control_result.get('cohens_d', 0)
        no_control_cohens_d = no_control_result.get('cohens_d', 0)

        comparison['control_cohens_d'] = control_cohens_d
        comparison['no_control_cohens_d'] = no_control_cohens_d
        comparison['cohens_d_difference'] = control_cohens_d - no_control_cohens_d

        # Cpk 개선 비교
        control_cpk_change = control_result.get('cpk_change', 0)
        no_control_cpk_change = no_control_result.get('cpk_change', 0)

        comparison['control_cpk_change'] = control_cpk_change
        comparison['no_control_cpk_change'] = no_control_cpk_change
        comparison['cpk_change_difference'] = control_cpk_change - no_control_cpk_change

        # 종합 판단
        comparison['control_more_effective'] = (
            comparison['control_significant'] and
            not comparison['no_control_significant'] and
            abs(control_cohens_d) > abs(no_control_cohens_d)
        )

        return comparison
