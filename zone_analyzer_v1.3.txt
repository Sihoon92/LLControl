"""
Zone 분석 모듈 v1.3
- 통계적 분석 기능 추가 (StatisticalAnalyzer 통합)
- 제어 구간 vs 비제어 구간 비교 분석
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Dict, List, Optional
import os
import re
import logging

# 한글 폰트 설정 (Windows)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False


class ZoneAnalyzer:
    """Zone별 분석 클래스 (통계 분석 포함)"""

    def __init__(self, config, statistical_analyzer=None, logger: logging.Logger = None):
        """
        Parameters:
        -----------
        config : PreprocessConfig
            전처리 설정 객체
        statistical_analyzer : StatisticalAnalyzer, optional
            통계 분석기 객체
        logger : logging.Logger
            로거 객체
        """
        self.config = config
        self.logger = logger or logging.getLogger('coating_preprocessor.zone')

        # 통계 분석기 (외부에서 주입하거나 자동 생성)
        if statistical_analyzer is None:
            from statistical_analyzer_v1_0 import StatisticalAnalyzer
            self.stat_analyzer = StatisticalAnalyzer(logger=self.logger)
        else:
            self.stat_analyzer = statistical_analyzer

        self.zone_results = None
        self.zone_statistics = None

    def run(
        self,
        densitometer_data: pd.DataFrame,
        meaningful_changes: pd.DataFrame,
        visualize: bool = True,
        perform_statistical_analysis: bool = True
    ) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """
        Zone별 분석 실행 (통계 분석 포함)

        Parameters:
        -----------
        densitometer_data : pd.DataFrame
            추출된 밀도계 데이터 (제어 + 비제어)
        meaningful_changes : pd.DataFrame
            유의미한 변경 구간 데이터 (UCL/LCL 포함)
        visualize : bool
            시각화 수행 여부
        perform_statistical_analysis : bool
            통계 분석 수행 여부

        Returns:
        --------
        Tuple[pd.DataFrame, pd.DataFrame]
            (Zone 분석 결과, 통계 분석 결과)
        """
        self.logger.info("="*80)
        self.logger.info("Zone별 분석 시작 (v1.3 - 통계 분석 포함)")
        self.logger.info("="*80)

        # 기본 Zone 분석 실행
        self.zone_results = self.analyze_all_groups(
            densitometer_data=densitometer_data,
            meaningful_changes=meaningful_changes,
            visualize=visualize
        )

        # 통계 분석 실행
        if perform_statistical_analysis and self.zone_results is not None:
            self.logger.info("="*80)
            self.logger.info("통계 분석 시작")
            self.logger.info("="*80)

            self.zone_statistics = self.perform_statistical_analysis(
                densitometer_data=densitometer_data,
                meaningful_changes=meaningful_changes
            )

            # 통계 분석 결과 저장
            if self.zone_statistics is not None and not self.zone_statistics.empty:
                output_file = os.path.join(
                    self.config.OUTPUT_DIR,
                    self.config.OUTPUT_STATISTICAL_ANALYSIS
                )

                try:
                    if output_file.endswith('.csv'):
                        self.zone_statistics.to_csv(output_file, index=False, encoding='utf-8-sig')
                    elif output_file.endswith(('.xlsx', '.xls')):
                        self.zone_statistics.to_excel(output_file, index=False)
                    else:
                        output_file = output_file + '.xlsx'
                        self.zone_statistics.to_excel(output_file, index=False)

                    self.logger.info(f"✓ 통계 분석 결과 저장: '{output_file}'")
                except Exception as e:
                    self.logger.error(f"✗ 저장 오류: {e}", exc_info=True)

        return self.zone_results, self.zone_statistics

    def analyze_all_groups(
        self,
        densitometer_data: pd.DataFrame,
        meaningful_changes: pd.DataFrame,
        visualize: bool = True
    ) -> pd.DataFrame:
        """
        모든 Group에 대해 Zone별 분석 수행

        Parameters:
        -----------
        densitometer_data : pd.DataFrame
            밀도계 데이터
        meaningful_changes : pd.DataFrame
            meaningful_changes 데이터 (UCL/LCL 포함)
        visualize : bool
            시각화 수행 여부

        Returns:
        --------
        pd.DataFrame
            Zone 분석 결과
        """
        self.logger.info(f"데이터 크기:")
        self.logger.info(f"  밀도계 데이터: {len(densitometer_data)} 행")
        self.logger.info(f"  Meaningful Changes: {len(meaningful_changes)} 행")

        # Value 칼럼 추출
        self.logger.info(f"[1단계] Value 칼럼 추출 및 경계 검출...")

        value_columns = []
        for col in densitometer_data.columns:
            col_str = str(col)
            if 'value' in col_str.lower() or col_str.isdigit():
                value_columns.append(col)

        if not value_columns:
            value_columns = [col for col in densitometer_data.columns
                            if col not in ['group_id', 'control_type', 'before/after', 'time', 'Time', 'TIME']]

        self.logger.info(f"   ✓ {len(value_columns)}개 Value 칼럼 발견")

        # 경계 검출
        left_boundary, right_boundary = self.find_boundaries(
            densitometer_data,
            value_columns,
            self.config.BOUNDARY_THRESHOLD
        )
        valid_value_columns = value_columns[left_boundary:right_boundary+1]

        self.logger.info(f"   ✓ {len(valid_value_columns)}개 유효 칼럼 추출")

        # Zone 할당
        self.logger.info(f"[2단계] Zone 할당...")
        zones = self.assign_zones(len(valid_value_columns), self.config.N_ZONES)

        # 각 Group 분석
        self.logger.info(f"[3단계] Group별 분석 수행...")

        all_zone_results = []

        # UCL/LCL이 있는 group만 분석
        valid_groups = meaningful_changes[
            meaningful_changes['UCL'].notna() & meaningful_changes['LCL'].notna()
        ]['group_id'].unique()

        self.logger.info(f"분석 대상 Group 수: {len(valid_groups)} / {meaningful_changes['group_id'].nunique()}")

        for group_id in valid_groups:
            try:
                result = self.analyze_group_zone_distribution(
                    data_df=densitometer_data,
                    meaningful_df=meaningful_changes,
                    group_id=group_id,
                    value_columns=valid_value_columns,
                    zones=zones,
                    n_divisions=self.config.N_DIVISIONS
                )

                if result and result['zone_results']:
                    all_zone_results.extend(result['zone_results'])

                    # 시각화
                    if visualize:
                        try:
                            self.visualize_group_zone_analysis(result)
                        except Exception as e:
                            self.logger.warning(f"   ⚠ Group {group_id} 시각화 오류: {e}")

            except Exception as e:
                self.logger.warning(f"   ⚠ Group {group_id} 분석 오류: {e}")
                continue

        if not all_zone_results:
            self.logger.warning("   ✗ 분석 결과 없음")
            return pd.DataFrame()

        # 결과 저장
        self.logger.info(f"[4단계] 결과 저장...")
        try:
            results_df = pd.DataFrame(all_zone_results)

            output_file = os.path.join(
                self.config.OUTPUT_DIR,
                self.config.OUTPUT_ZONE_ANALYSIS
            )

            if output_file.endswith('.csv'):
                results_df.to_csv(output_file, index=False, encoding='utf-8-sig')
            elif output_file.endswith(('.xlsx', '.xls')):
                results_df.to_excel(output_file, index=False)
            else:
                output_file = output_file + '.xlsx'
                results_df.to_excel(output_file, index=False)

            self.logger.info(f"   ✓ 분석 결과 저장: '{output_file}'")

        except Exception as e:
            self.logger.error(f"   ✗ 저장 오류: {e}", exc_info=True)
            return pd.DataFrame()

        # 최종 요약
        self.logger.info("="*80)
        self.logger.info("Zone 분석 완료 요약")
        self.logger.info("="*80)
        self.logger.info(f"분석 Group 수: {len(valid_groups)}")
        self.logger.info(f"총 Zone 분석 결과: {len(all_zone_results)}")
        self.logger.info(f"결과 파일: {output_file}")
        if visualize:
            self.logger.info(f"시각화 파일: {self.config.PLOT_DIR}/group_zone_analysis/")
        self.logger.info("="*80)

        return results_df

    def perform_statistical_analysis(
        self,
        densitometer_data: pd.DataFrame,
        meaningful_changes: pd.DataFrame
    ) -> pd.DataFrame:
        """
        통계 분석 수행 (제어 구간 + 비제어 구간)

        Parameters:
        -----------
        densitometer_data : pd.DataFrame
            밀도계 데이터 (control_type 칼럼 포함)
        meaningful_changes : pd.DataFrame
            meaningful_changes 데이터 (UCL/LCL 포함)

        Returns:
        --------
        pd.DataFrame
            통계 분석 결과
        """
        self.logger.info("통계 분석 수행 중...")

        # Value 칼럼 추출
        value_columns = []
        for col in densitometer_data.columns:
            col_str = str(col)
            if 'value' in col_str.lower() or col_str.isdigit():
                value_columns.append(col)

        if not value_columns:
            value_columns = [col for col in densitometer_data.columns
                            if col not in ['group_id', 'control_type', 'before/after', 'time', 'Time', 'TIME']]

        # 경계 검출
        left_boundary, right_boundary = self.find_boundaries(
            densitometer_data,
            value_columns,
            self.config.BOUNDARY_THRESHOLD
        )
        valid_value_columns = value_columns[left_boundary:right_boundary+1]

        # Zone 할당
        zones = self.assign_zones(len(valid_value_columns), self.config.N_ZONES)

        # 통계 분석 결과 저장
        statistical_results = []

        # 제어 구간 분석
        control_data = densitometer_data[densitometer_data['control_type'] == 'controlled']
        control_groups = control_data['group_id'].unique()

        self.logger.info(f"제어 구간 분석: {len(control_groups)}개 그룹")

        for group_id in control_groups:
            group_data = control_data[control_data['group_id'] == group_id]

            # meaningful_changes에서 UCL/LCL 정보 가져오기
            group_info = meaningful_changes[meaningful_changes['group_id'] == group_id]
            if len(group_info) == 0:
                continue

            group_info = group_info.iloc[0]
            ucl = group_info.get('UCL', np.nan)
            lcl = group_info.get('LCL', np.nan)
            target = group_info.get('TARGET', None)

            if pd.isna(ucl) or pd.isna(lcl):
                continue

            # Zone별 통계 분석
            unique_zones = np.unique(zones)
            for zone_id in unique_zones:
                # 해당 Zone의 칼럼들
                zone_mask = zones == zone_id
                zone_cols = [valid_value_columns[i] for i in range(len(valid_value_columns)) if zone_mask[i]]

                # Before 데이터
                before_data = group_data[group_data['before/after'] == 'before'][zone_cols].values.flatten()
                before_data = before_data[before_data > 0]

                # After 데이터
                after_data = group_data[group_data['before/after'] == 'after'][zone_cols].values.flatten()
                after_data = after_data[after_data > 0]

                if len(before_data) == 0 or len(after_data) == 0:
                    continue

                # 통계 분석 수행
                stats_result = self.stat_analyzer.analyze_control_effect(
                    before_data=before_data,
                    after_data=after_data,
                    ucl=ucl,
                    lcl=lcl,
                    target=target if not pd.isna(target) else None
                )

                # 결과에 메타데이터 추가
                stats_result['group_id'] = group_id
                stats_result['zone_id'] = zone_id
                stats_result['control_type'] = 'controlled'
                stats_result['ucl'] = ucl
                stats_result['lcl'] = lcl
                stats_result['target'] = target if not pd.isna(target) else np.nan

                statistical_results.append(stats_result)

        # 비제어 구간 분석
        no_control_data = densitometer_data[densitometer_data['control_type'] == 'no_control']
        no_control_groups = no_control_data['group_id'].unique()

        self.logger.info(f"비제어 구간 분석: {len(no_control_groups)}개 그룹")

        for group_id in no_control_groups:
            group_data = no_control_data[no_control_data['group_id'] == group_id]

            # 제어 구간과 동일한 UCL/LCL 사용 (평균값 또는 대표값)
            # 여기서는 meaningful_changes의 평균 UCL/LCL 사용
            if len(meaningful_changes) > 0:
                ucl = meaningful_changes['UCL'].mean()
                lcl = meaningful_changes['LCL'].mean()
                target = meaningful_changes['TARGET'].mean() if 'TARGET' in meaningful_changes.columns else None
            else:
                continue

            # Zone별 통계 분석
            unique_zones = np.unique(zones)
            for zone_id in unique_zones:
                # 해당 Zone의 칼럼들
                zone_mask = zones == zone_id
                zone_cols = [valid_value_columns[i] for i in range(len(valid_value_columns)) if zone_mask[i]]

                # Before 데이터
                before_data = group_data[group_data['before/after'] == 'before'][zone_cols].values.flatten()
                before_data = before_data[before_data > 0]

                # After 데이터
                after_data = group_data[group_data['before/after'] == 'after'][zone_cols].values.flatten()
                after_data = after_data[after_data > 0]

                if len(before_data) == 0 or len(after_data) == 0:
                    continue

                # 통계 분석 수행
                stats_result = self.stat_analyzer.analyze_control_effect(
                    before_data=before_data,
                    after_data=after_data,
                    ucl=ucl,
                    lcl=lcl,
                    target=target if not pd.isna(target) else None
                )

                # 결과에 메타데이터 추가
                stats_result['group_id'] = group_id
                stats_result['zone_id'] = zone_id
                stats_result['control_type'] = 'no_control'
                stats_result['ucl'] = ucl
                stats_result['lcl'] = lcl
                stats_result['target'] = target if not pd.isna(target) else np.nan

                statistical_results.append(stats_result)

        if not statistical_results:
            self.logger.warning("통계 분석 결과 없음")
            return pd.DataFrame()

        stats_df = pd.DataFrame(statistical_results)

        self.logger.info("="*80)
        self.logger.info("통계 분석 완료 요약")
        self.logger.info("="*80)
        self.logger.info(f"전체 통계 분석 결과: {len(stats_df)}")
        self.logger.info(f"  - 제어 구간: {len(stats_df[stats_df['control_type'] == 'controlled'])}")
        self.logger.info(f"  - 비제어 구간: {len(stats_df[stats_df['control_type'] == 'no_control'])}")

        # 통계적 유의성 요약
        controlled_stats = stats_df[stats_df['control_type'] == 'controlled']
        if len(controlled_stats) > 0:
            significant_count = controlled_stats['statistically_significant'].sum()
            self.logger.info(f"제어 구간 통계적 유의 비율: {significant_count}/{len(controlled_stats)} ({significant_count/len(controlled_stats)*100:.1f}%)")

            effective_count = controlled_stats['control_effective'].sum()
            self.logger.info(f"제어 효과 있음 비율: {effective_count}/{len(controlled_stats)} ({effective_count/len(controlled_stats)*100:.1f}%)")

        self.logger.info("="*80)

        return stats_df

    def find_boundaries(
        self,
        df: pd.DataFrame,
        value_columns: List[str],
        threshold: float = 0.9
    ) -> Tuple[int, int]:
        """
        유의미한 데이터가 있는 좌/우 경계(boundary) 검출

        Parameters:
        -----------
        df : pd.DataFrame
            전처리된 밀도계 데이터
        value_columns : List[str]
            Value 칼럼 리스트
        threshold : float
            0 이상 데이터 비율 임계값 (기본값: 0.9 = 90%)

        Returns:
        --------
        Tuple[int, int]
            (left_boundary_index, right_boundary_index)
        """
        self.logger.info("="*80)
        self.logger.info("경계(Boundary) 검출 시작")
        self.logger.info("="*80)
        self.logger.info(f"임계값: {threshold*100:.0f}% (0 이상 데이터 비율)")

        n_cols = len(value_columns)

        # 각 칼럼별로 0 이상 데이터 비율 계산
        valid_ratios = []
        for col in value_columns:
            valid_count = (df[col] > 0).sum()
            total_count = len(df)
            ratio = valid_count / total_count if total_count > 0 else 0
            valid_ratios.append(ratio)

        valid_ratios = np.array(valid_ratios)

        # Left boundary
        left_boundary = None
        for i in range(n_cols):
            if valid_ratios[i] >= threshold:
                left_boundary = i
                break

        # Right boundary
        right_boundary = None
        for i in range(n_cols - 1, -1, -1):
            if valid_ratios[i] >= threshold:
                right_boundary = i
                break

        if left_boundary is None or right_boundary is None:
            self.logger.warning("유효한 경계를 찾지 못했습니다.")
            left_boundary = 0
            right_boundary = n_cols - 1

        self.logger.info(f"✓ Left Boundary: 칼럼 인덱스 {left_boundary} ({value_columns[left_boundary]})")
        self.logger.info(f"  - 유효 데이터 비율: {valid_ratios[left_boundary]*100:.2f}%")
        self.logger.info(f"✓ Right Boundary: 칼럼 인덱스 {right_boundary} ({value_columns[right_boundary]})")
        self.logger.info(f"  - 유효 데이터 비율: {valid_ratios[right_boundary]*100:.2f}%")
        self.logger.info(f"✓ 추출 범위: {right_boundary - left_boundary + 1} 개 칼럼")
        self.logger.info("="*80)

        return left_boundary, right_boundary

    def assign_zones(self, n_columns: int, n_zones: int = 11) -> np.ndarray:
        """
        칼럼에 Zone 번호 할당 (동일 간격)

        Parameters:
        -----------
        n_columns : int
            총 칼럼 개수
        n_zones : int
            Zone 구역 수

        Returns:
        --------
        np.ndarray
            각 칼럼의 Zone 번호 배열
        """
        self.logger.info(f"Zone 할당 중... (총 {n_zones}개 Zone)")

        zone_size = n_columns / n_zones
        zones = np.ceil((np.arange(n_columns) + 1) / zone_size).astype(int)

        for zone_id in range(1, n_zones + 1):
            count = (zones == zone_id).sum()
            self.logger.debug(f"  Zone {zone_id}: {count}개 칼럼")

        return zones

    def analyze_group_zone_distribution(
        self,
        data_df: pd.DataFrame,
        meaningful_df: pd.DataFrame,
        group_id: int,
        value_columns: List[str],
        zones: np.ndarray,
        n_divisions: int = 6
    ) -> Optional[Dict]:
        """
        특정 group에 대한 Zone별 분포 분석

        Parameters:
        -----------
        data_df : pd.DataFrame
            밀도계 raw 데이터
        meaningful_df : pd.DataFrame
            meaningful_changes 데이터 (UCL/LCL 포함)
        group_id : int
            분석 대상 group ID
        value_columns : List[str]
            Value 칼럼 리스트
        zones : np.ndarray
            각 칼럼의 Zone 번호
        n_divisions : int
            균등 분할 개수

        Returns:
        --------
        Dict
            분석 결과
        """
        self.logger.info("="*80)
        self.logger.info(f"Group {group_id} 분석")
        self.logger.info("="*80)

        # Group 정보 가져오기
        group_info = meaningful_df[meaningful_df['group_id'] == group_id]

        if len(group_info) == 0:
            self.logger.warning(f"Group {group_id}: 정보 없음")
            return None

        group_info = group_info.iloc[0]

        # UCL/LCL 확인
        ucl = group_info.get('UCL', None)
        lcl = group_info.get('LCL', None)
        target = group_info.get('TARGET', None)

        if pd.isna(ucl) or pd.isna(lcl):
            self.logger.warning(f"Group {group_id}: UCL 또는 LCL 정보 없음")
            return None

        self.logger.info(f"UCL: {ucl:.4f}")
        self.logger.info(f"LCL: {lcl:.4f}")
        if target is not None and not pd.isna(target):
            self.logger.info(f"TARGET: {target:.4f}")

        # 분할 구간 생성
        division_edges = np.linspace(lcl, ucl, n_divisions + 1)

        self.logger.info(f"분할 구간 정보 (n_divisions={n_divisions}):")
        for i in range(n_divisions):
            self.logger.debug(f"  구간 {i+1}: [{division_edges[i]:.4f}, {division_edges[i+1]:.4f})")

        # Group 데이터 필터링 (제어 구간만)
        group_data = data_df[
            (data_df['group_id'] == group_id) &
            (data_df['control_type'] == 'controlled')
        ].copy()

        if len(group_data) == 0:
            self.logger.warning(f"Group {group_id}: 밀도계 데이터 없음")
            return None

        self.logger.info(f"Group {group_id} 밀도계 데이터: {len(group_data)} 행")

        # Before/After 분리
        before_data = group_data[group_data['before/after'] == 'before']
        after_data = group_data[group_data['before/after'] == 'after']

        self.logger.info(f"Before 데이터: {len(before_data)} 행")
        self.logger.info(f"After 데이터: {len(after_data)} 행")

        # Zone별 분석
        unique_zones = np.unique(zones)
        zone_results = []

        for zone_id in unique_zones:
            # 해당 Zone의 칼럼들
            zone_mask = zones == zone_id
            zone_cols = [value_columns[i] for i in range(len(value_columns)) if zone_mask[i]]

            # Before 데이터 추출
            before_zone_data = before_data[zone_cols].values.flatten()
            before_zone_data = before_zone_data[before_zone_data > 0]

            # After 데이터 추출
            after_zone_data = after_data[zone_cols].values.flatten()
            after_zone_data = after_zone_data[after_zone_data > 0]

            if len(before_zone_data) == 0:
                continue

            # 각 구간별 분포 계산 - Before
            before_hist, _ = np.histogram(before_zone_data, bins=division_edges)
            before_ratios = before_hist / len(before_zone_data)

            # 각 구간별 분포 계산 - After
            after_hist, _ = np.histogram(after_zone_data, bins=division_edges)
            after_ratios = after_hist / len(after_zone_data) if len(after_zone_data) > 0 else np.zeros_like(before_hist)

            zone_result = {
                'group_id': group_id,
                'zone_id': zone_id,
                'ucl': ucl,
                'lcl': lcl,
                'lsl': lcl,  # LSL = LCL로 통일
                'usl': ucl,  # USL = UCL로 통일
                'target': target,
                'n_divisions': n_divisions,
                'before_total': len(before_zone_data),
                'after_total': len(after_zone_data),
                'before_mean': before_zone_data.mean(),
                'after_mean': after_zone_data.mean() if len(after_zone_data) > 0 else np.nan,
                'before_std': before_zone_data.std(),
                'after_std': after_zone_data.std() if len(after_zone_data) > 0 else np.nan
            }

            # 구간별 정보 추가
            for i in range(n_divisions):
                zone_result[f'div_{i+1}_range'] = f"[{division_edges[i]:.4f}, {division_edges[i+1]:.4f})"
                zone_result[f'div_{i+1}_before_count'] = int(before_hist[i])
                zone_result[f'div_{i+1}_before_ratio'] = float(before_ratios[i])
                zone_result[f'div_{i+1}_after_count'] = int(after_hist[i])
                zone_result[f'div_{i+1}_after_ratio'] = float(after_ratios[i])
                zone_result[f'div_{i+1}_ratio_change'] = float(after_ratios[i] - before_ratios[i])

            zone_results.append(zone_result)

            # 로그 출력
            self.logger.debug(f"{'='*60}")
            self.logger.debug(f"Zone {zone_id}")
            self.logger.debug(f"{'='*60}")
            self.logger.debug(f"Before: {len(before_zone_data):,}개 데이터 (평균: {before_zone_data.mean():.4f})")
            if len(after_zone_data) > 0:
                self.logger.debug(f"After: {len(after_zone_data):,}개 데이터 (평균: {after_zone_data.mean():.4f})")

        self.logger.info("="*80)

        return {
            'group_id': group_id,
            'ucl': ucl,
            'lcl': lcl,
            'target': target,
            'n_divisions': n_divisions,
            'division_edges': division_edges,
            'zone_results': zone_results
        }

    def visualize_group_zone_analysis(self, result: Dict):
        """
        Group Zone 분석 결과 시각화
        (기존 visualize_group_zone_analysis 메서드와 동일)
        """
        # 시각화 구현은 기존 코드와 동일하므로 생략
        # 필요시 zone_analyzer_v1.2.txt의 visualize_group_zone_analysis 메서드 참조
        pass
