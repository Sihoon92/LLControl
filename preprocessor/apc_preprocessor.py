"""
APC 데이터 전처리 모듈 v1.3
- 비제어 구간(대조군) 추출 기능 추가
- 통계적 분석을 위한 데이터 수집
"""

import xlwings as xw
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Tuple, Optional, List
import os
import logging
import utils


class APCPreprocessor:
    """APC 데이터 전처리 클래스"""

    def __init__(self, config, logger: logging.Logger = None):
        """
        Parameters:
        -----------
        config : PreprocessConfig
            전처리 설정 객체
        logger : logging.Logger
            로거 객체
        """
        self.config = config
        self.logger = logger or logging.getLogger('coating_preprocessor.apc')
        self.changes_df = None
        self.original_df = None
        self.grouped_changes = None
        self.meaningful_df = None
        self.control_regions_df = None
        self.no_control_regions_df = None

    def run(self, input_file: str, llspec_file: Optional[str] = None) -> pd.DataFrame:
        """
        전체 APC 전처리 프로세스 실행

        Parameters:
        -----------
        input_file : str
            APC 원본 데이터 파일 경로
        llspec_file : str, optional
            LLspec 파일 경로

        Returns:
        --------
        pd.DataFrame
            유의미한 변경 구간 데이터 (3rd_meaningful_changes)
        """
        self.logger.info("="*80)
        self.logger.info("APC 데이터 전처리 시작 (v1.3 - 비제어 구간 추출 포함)")
        self.logger.info("="*80)
        self.logger.info(f"입력 파일: {input_file}")
        if llspec_file:
            self.logger.info(f"LLspec 파일: {llspec_file}")
        self.logger.info("="*80)

        # 1차 전처리: 변경점 감지
        self.logger.info("[1차 전처리] 변경점 감지")
        self.changes_df, self.original_df = self.detect_parameter_changes(input_file)

        if self.changes_df is None:
            self.logger.warning("변경점이 없어 종료합니다.")
            return None

        # 1차 결과 저장
        output_1st = os.path.join(self.config.OUTPUT_DIR, self.config.OUTPUT_1ST)
        utils.save_to_excel(self.changes_df, output_1st, sheet_name='All_Changes', logger=self.logger)

        # 2차 전처리: 그룹화
        self.logger.info("[2차 전처리] 변경 구간 그룹화")
        self.grouped_changes = self.group_changes_by_time(
            self.changes_df,
            self.config.TIME_THRESHOLD_MINUTES
        )

        # 그룹 요약 정보 생성
        summary_df = self.create_grouped_summary(self.grouped_changes)
        output_2nd = os.path.join(self.config.OUTPUT_DIR, self.config.OUTPUT_2ND)
        utils.save_to_excel(summary_df, output_2nd, sheet_name='Grouped_Changes', logger=self.logger)

        # 3차 전처리: 유의미한 변경 구간 필터링
        self.logger.info("[3차 전처리] 유의미한 변경 구간 필터링")
        self.meaningful_df = self.filter_meaningful_changes(
            self.grouped_changes,
            self.original_df,
            self.config.BEFORE_MINUTES,
            self.config.AFTER_MINUTES
        )

        if self.meaningful_df is None or self.meaningful_df.empty:
            self.logger.warning("유의미한 변경 구간이 없어 종료합니다.")
            return None

        # LLspec 데이터 통합 (옵션)
        if llspec_file:
            self.logger.info("[LLspec 통합] LLspec 데이터 통합")
            llspec_df = self.load_llspec_data(llspec_file, self.config.SIDE_FILTER)

            if llspec_df is not None:
                self.meaningful_df = self.match_llspec_to_groups(
                    self.meaningful_df,
                    llspec_df
                )

        # 최종 결과 저장
        output_3rd = os.path.join(self.config.OUTPUT_DIR, self.config.OUTPUT_3RD)
        utils.save_to_excel(self.meaningful_df, output_3rd, sheet_name='Meaningful_Changes', logger=self.logger)

        # ===== NEW: 4차 전처리 - 제어 구간 정보 저장 =====
        self.logger.info("[4차 전처리] 제어 구간 정보 저장")
        self.control_regions_df = self.create_control_regions_info(
            self.meaningful_df,
            self.config.BEFORE_MINUTES,
            self.config.AFTER_MINUTES
        )
        output_4th = os.path.join(self.config.OUTPUT_DIR, self.config.OUTPUT_4TH_CONTROL)
        utils.save_to_excel(self.control_regions_df, output_4th, sheet_name='Control_Regions', logger=self.logger)

        # ===== NEW: 5차 전처리 - 비제어 구간 샘플링 =====
        self.logger.info("[5차 전처리] 비제어 구간 샘플링")
        self.no_control_regions_df = self.sample_no_control_regions(
            self.original_df,
            self.grouped_changes,
            n_samples=len(self.meaningful_df),
            before_minutes=self.config.BEFORE_MINUTES,
            after_minutes=self.config.AFTER_MINUTES
        )

        if self.no_control_regions_df is not None and not self.no_control_regions_df.empty:
            output_5th = os.path.join(self.config.OUTPUT_DIR, self.config.OUTPUT_5TH_NO_CONTROL)
            utils.save_to_excel(self.no_control_regions_df, output_5th, sheet_name='No_Control_Regions', logger=self.logger)

        # 최종 통계
        self.logger.info("="*80)
        self.logger.info("APC 전처리 완료!")
        self.logger.info("="*80)
        self.logger.info(f"전체 변경점: {len(self.changes_df)}")
        self.logger.info(f"그룹화된 변경 구간: {len(self.grouped_changes)}")
        self.logger.info(f"유의미한 변경 구간: {len(self.meaningful_df)}")

        if llspec_file and 'UCL' in self.meaningful_df.columns:
            matched_count = self.meaningful_df['UCL'].notna().sum()
            self.logger.info(f"LLspec 매칭 성공: {matched_count} / {len(self.meaningful_df)}")

        self.logger.info(f"제어 구간 정보: {len(self.control_regions_df)}")
        if self.no_control_regions_df is not None:
            self.logger.info(f"비제어 구간 샘플: {len(self.no_control_regions_df)}")

        self.logger.info("="*80)

        return self.meaningful_df

    def detect_parameter_changes(
        self,
        file_path: str,
        sheet_name: Optional[str] = None
    ) -> Tuple[Optional[pd.DataFrame], pd.DataFrame]:
        """
        1차 전처리: GV_GAP 및 PUMP RPM 값 변경 감지

        Parameters:
        -----------
        file_path : str
            데이터 파일 경로 (Excel, Parquet, CSV 지원)
        sheet_name : str, optional
            시트 이름 (Excel 파일의 경우, None이면 첫 번째 시트)

        Returns:
        --------
        Tuple[pd.DataFrame, pd.DataFrame]
            (변경 사항 DataFrame, 필터링된 원본 DataFrame)
        """
        try:
            # 파일 형식에 따라 자동 로드 (Excel/Parquet/CSV)
            # Excel: xlwings, Parquet/CSV: pandas
            df = utils.load_file(
                file_path,
                sheet_name=sheet_name if sheet_name else 0,
                logger=self.logger
            )

            # SIDE 필터링
            self.logger.info(f"원본 데이터 행 수: {len(df)}")

            if self.config.SIDE_COL in df.columns:
                df = df[df[self.config.SIDE_COL] == self.config.SIDE_FILTER].reset_index(drop=True)
                self.logger.info(f"SIDE='{self.config.SIDE_FILTER}' 필터링 후 행 수: {len(df)}")
            else:
                self.logger.warning(f"'{self.config.SIDE_COL}' 칼럼이 없습니다. 필터링을 건너뜁니다.")

            # TIME 칼럼을 datetime으로 변환
            df[self.config.TIME_COL] = pd.to_datetime(df[self.config.TIME_COL])

            # GV_GAP 칼럼 찾기 (0 패딩 형식)
            gv_gap_columns = sorted([
                col for col in df.columns
                if self.config.GV_GAP_PATTERN in str(col)
            ])

            # 감지 대상 칼럼 = GV_GAP + PUMP RPM
            monitor_columns = gv_gap_columns + [self.config.PUMP_RPM_COL]

            self.logger.info(f"감지 대상 칼럼 수: {len(monitor_columns)}")
            self.logger.info(f"  GV_GAP: {len(gv_gap_columns)}개")
            self.logger.info(f"  PUMP RPM: 1개")
            self.logger.info("="*80)

            # 변경 사항 저장
            changes = []

            # 각 행을 순회하며 변경 감지
            for idx in range(1, len(df)):
                current_row = df.iloc[idx]
                previous_row = df.iloc[idx - 1]

                # 각 모니터링 칼럼 체크
                for col in monitor_columns:
                    prev_value = previous_row[col]
                    curr_value = current_row[col]

                    # 값이 변경되었는지 확인
                    if prev_value != curr_value:
                        # NaN 체크
                        if pd.isna(prev_value) and pd.isna(curr_value):
                            continue

                        change_info = {
                            'original_index': idx,
                            'TIME': current_row[self.config.TIME_COL],
                            'column': col,
                            'prev_value': prev_value,
                            'curr_value': curr_value
                        }
                        changes.append(change_info)

                        # 로그 출력
                        self.logger.debug(f"[변경 감지 #{len(changes)}]")
                        self.logger.debug(f"  시점: {current_row[self.config.TIME_COL]}")
                        self.logger.debug(f"  변수: {col}")
                        self.logger.debug(f"  변경: {prev_value} → {curr_value}")
                        self.logger.debug("-"*80)

            self.logger.info(f"총 {len(changes)}개의 변경 사항이 감지되었습니다.")

            if changes:
                result_df = pd.DataFrame(changes)
                return result_df, df
            else:
                self.logger.warning("변경 사항이 없습니다.")
                return None, df

        except FileNotFoundError:
            self.logger.error(f"파일을 찾을 수 없습니다: {file_path}")
            return None, None
        except Exception as e:
            self.logger.error(f"오류: {e}", exc_info=True)
            return None, None

    def group_changes_by_time(
        self,
        changes_df: pd.DataFrame,
        time_threshold_minutes: int = 2
    ) -> List[dict]:
        """
        2차 전처리: 시간 기준으로 변경점 그룹화

        Parameters:
        -----------
        changes_df : pd.DataFrame
            1차 전처리 결과 DataFrame
        time_threshold_minutes : int
            그룹화 시간 임계값 (분)

        Returns:
        --------
        List[dict]
            그룹화된 변경 구간 리스트
        """
        if changes_df is None or changes_df.empty:
            return []

        # TIME 기준으로 정렬
        changes_df = changes_df.sort_values('TIME').reset_index(drop=True)

        grouped_changes = []
        current_group = {
            'start_time': None,
            'end_time': None,
            'changes': []
        }

        for idx, row in changes_df.iterrows():
            if current_group['start_time'] is None:
                # 첫 번째 그룹 시작
                current_group['start_time'] = row['TIME']
                current_group['end_time'] = row['TIME']
                current_group['changes'].append(row)
            else:
                # 이전 변경점과의 시간 차이 계산
                time_diff = (row['TIME'] - current_group['end_time']).total_seconds() / 60

                if time_diff <= time_threshold_minutes:
                    # 같은 그룹에 추가
                    current_group['end_time'] = row['TIME']
                    current_group['changes'].append(row)
                else:
                    # 이전 그룹 저장하고 새 그룹 시작
                    grouped_changes.append(current_group.copy())
                    current_group = {
                        'start_time': row['TIME'],
                        'end_time': row['TIME'],
                        'changes': [row]
                    }

        # 마지막 그룹 저장
        if current_group['start_time'] is not None:
            grouped_changes.append(current_group)

        self.logger.info(f"총 {len(grouped_changes)}개의 변경 구간이 그룹화되었습니다.")

        return grouped_changes

    def create_grouped_summary(self, grouped_changes: List[dict]) -> pd.DataFrame:
        """
        그룹화된 변경 구간의 요약 정보 생성
        모든 GV_GAP{01~13}_before/after 칼럼을 미리 생성하고 NaN으로 채움

        Parameters:
        -----------
        grouped_changes : List[dict]
            그룹화된 변경 구간 리스트

        Returns:
        --------
        pd.DataFrame
            요약 DataFrame
        """
        summary_list = []

        # 모든 가능한 GV_GAP 칼럼 정의 (01~13, 0 패딩 적용)
        all_gv_gap_columns = [f'GV_GAP{i:02d}' for i in range(1, 14)]

        for group_idx, group in enumerate(grouped_changes):
            # 각 변수별로 시작 값과 종료 값 정리
            variable_changes = {}

            for change in group['changes']:
                var_name = change['column']

                if var_name not in variable_changes:
                    variable_changes[var_name] = {
                        'initial_value': change['prev_value'],
                        'final_value': change['curr_value']
                    }
                else:
                    # 같은 변수가 여러 번 변경된 경우 최종 값만 업데이트
                    variable_changes[var_name]['final_value'] = change['curr_value']

            # 요약 정보 생성
            summary_info = {
                'group_id': group_idx + 1,
                'start_time': group['start_time'],
                'end_time': group['end_time'],
                'duration_seconds': (group['end_time'] - group['start_time']).total_seconds(),
                'num_changes': len(group['changes']),
                'changed_variables': ', '.join(variable_changes.keys())
            }

            # 모든 GV_GAP{01~13}_before/after 칼럼을 NaN으로 초기화
            for gv_col in all_gv_gap_columns:
                summary_info[f'{gv_col}_before'] = np.nan
                summary_info[f'{gv_col}_after'] = np.nan

            # PUMP RPM_before/after도 NaN으로 초기화
            summary_info[f'{self.config.PUMP_RPM_COL}_before'] = np.nan
            summary_info[f'{self.config.PUMP_RPM_COL}_after'] = np.nan

            # 실제 변경된 변수들의 값 업데이트
            for var_name, values in variable_changes.items():
                summary_info[f'{var_name}_before'] = values['initial_value']
                summary_info[f'{var_name}_after'] = values['final_value']

            summary_list.append(summary_info)

        return pd.DataFrame(summary_list)

    def filter_meaningful_changes(
        self,
        grouped_changes: List[dict],
        original_df: pd.DataFrame,
        before_minutes: int = 5,
        after_minutes: int = 5
    ) -> pd.DataFrame:
        """
        3차 전처리: 유의미한 변경 구간 필터링 및 GV_ZONE 평균값/표준편차 계산

        Parameters:
        -----------
        grouped_changes : List[dict]
            그룹화된 변경 구간 리스트
        original_df : pd.DataFrame
            원본 데이터 DataFrame (SIDE 필터링된)
        before_minutes : int
            이전 확인 시간 (분)
        after_minutes : int
            이후 확인 시간 (분)

        Returns:
        --------
        pd.DataFrame
            유의미한 변경 구간 DataFrame
        """
        # GV_ZONE 칼럼 찾기
        gv_zone_columns = sorted([
            col for col in original_df.columns
            if self.config.GV_ZONE_PATTERN in str(col)
        ])

        # 모든 가능한 GV_GAP 칼럼 정의 (01~13, 0 패딩 적용)
        all_gv_gap_columns = [f'GV_GAP{i:02d}' for i in range(1, 14)]

        # 모든 가능한 GV_ZONE 칼럼 정의 (01~13, 0 패딩 적용)
        all_gv_zone_columns = [f'GV_ZONE{i:02d}' for i in range(1, 14)]

        meaningful_changes = []

        for group_idx, group in enumerate(grouped_changes):
            start_time = group['start_time']
            end_time = group['end_time']

            # 이전 5분과 이후 5분 범위 설정
            before_range_start = start_time - timedelta(minutes=before_minutes)
            before_range_end = start_time
            after_range_start = end_time
            after_range_end = end_time + timedelta(minutes=after_minutes)

            # 다른 변경점 확인
            has_conflict = False
            for other_group in grouped_changes:
                if other_group == group:
                    continue

                other_start = other_group['start_time']
                other_end = other_group['end_time']

                # 이전 5분 범위에 다른 변경점이 있는지 확인
                if before_range_start <= other_start < before_range_end or \
                   before_range_start < other_end <= before_range_end:
                    has_conflict = True
                    break

                # 이후 5분 범위에 다른 변경점이 있는지 확인
                if after_range_start < other_start <= after_range_end or \
                   after_range_start < other_end <= after_range_end:
                    has_conflict = True
                    break

            if not has_conflict:
                # 유의미한 변경 구간
                self.logger.info(f"유의미한 구간 발견: Group {group_idx + 1} - {start_time} ~ {end_time}")

                # 이전 5분간 데이터 추출
                before_data = original_df[
                    (original_df[self.config.TIME_COL] >= before_range_start) &
                    (original_df[self.config.TIME_COL] < before_range_end)
                ]

                # 이후 5분간 데이터 추출
                after_data = original_df[
                    (original_df[self.config.TIME_COL] > after_range_start) &
                    (original_df[self.config.TIME_COL] <= after_range_end)
                ]

                # 각 변수별로 시작 값과 종료 값 정리
                variable_changes = {}
                for change in group['changes']:
                    var_name = change['column']
                    if var_name not in variable_changes:
                        variable_changes[var_name] = {
                            'initial_value': change['prev_value'],
                            'final_value': change['curr_value']
                        }
                    else:
                        variable_changes[var_name]['final_value'] = change['curr_value']

                # 기본 정보
                change_info = {
                    'group_id': group_idx + 1,
                    'start_time': start_time,
                    'end_time': end_time,
                    'duration_seconds': (end_time - start_time).total_seconds(),
                    'num_changes': len(group['changes']),
                    'changed_variables': ', '.join(variable_changes.keys())
                }

                # 모든 GV_GAP{01~13}_before/after 칼럼을 NaN으로 초기화
                for gv_col in all_gv_gap_columns:
                    change_info[f'{gv_col}_before'] = np.nan
                    change_info[f'{gv_col}_after'] = np.nan

                # PUMP RPM_before/after도 NaN으로 초기화
                change_info[f'{self.config.PUMP_RPM_COL}_before'] = np.nan
                change_info[f'{self.config.PUMP_RPM_COL}_after'] = np.nan

                # 실제 변경된 변수들의 값 업데이트
                for var_name, values in variable_changes.items():
                    change_info[f'{var_name}_before'] = values['initial_value']
                    change_info[f'{var_name}_after'] = values['final_value']

                # 모든 GV_ZONE{01~13}_before/after_5min_avg/std 칼럼을 NaN으로 초기화
                for zone_col in all_gv_zone_columns:
                    change_info[f'{zone_col}_before_5min_avg'] = np.nan
                    change_info[f'{zone_col}_after_5min_avg'] = np.nan
                    change_info[f'{zone_col}_before_5min_std'] = np.nan
                    change_info[f'{zone_col}_after_5min_std'] = np.nan

                # 실제 존재하는 GV_ZONE 칼럼에 대해 평균값 및 표준편차 계산
                for zone_col in gv_zone_columns:
                    # 이전 5분 평균 및 표준편차
                    if not before_data.empty and zone_col in before_data.columns:
                        before_mean = before_data[zone_col].mean()
                        before_std = before_data[zone_col].std()
                        change_info[f'{zone_col}_before_5min_avg'] = before_mean
                        change_info[f'{zone_col}_before_5min_std'] = before_std

                    # 이후 5분 평균 및 표준편차
                    if not after_data.empty and zone_col in after_data.columns:
                        after_mean = after_data[zone_col].mean()
                        after_std = after_data[zone_col].std()
                        change_info[f'{zone_col}_after_5min_avg'] = after_mean
                        change_info[f'{zone_col}_after_5min_std'] = after_std

                meaningful_changes.append(change_info)

        if meaningful_changes:
            self.logger.info(f"총 {len(meaningful_changes)}개의 유의미한 변경 구간이 발견되었습니다.")
            return pd.DataFrame(meaningful_changes)
        else:
            self.logger.warning("유의미한 변경 구간이 없습니다.")
            return pd.DataFrame()

    def create_control_regions_info(
        self,
        meaningful_df: pd.DataFrame,
        before_minutes: int = 5,
        after_minutes: int = 5
    ) -> pd.DataFrame:
        """
        4차 전처리: 제어 구간 정보 저장

        Parameters:
        -----------
        meaningful_df : pd.DataFrame
            유의미한 변경 구간 DataFrame
        before_minutes : int
            이전 버퍼 시간 (분)
        after_minutes : int
            이후 버퍼 시간 (분)

        Returns:
        --------
        pd.DataFrame
            제어 구간 정보 (group_id, start_time, end_time, control_start, control_end)
        """
        self.logger.info("="*80)
        self.logger.info("제어 구간 정보 생성")
        self.logger.info("="*80)

        control_regions = []

        for idx, row in meaningful_df.iterrows():
            group_id = row['group_id']
            control_start = row['start_time']
            control_end = row['end_time']

            # 분석 구간 (before ~ after)
            start_time = control_start - timedelta(minutes=before_minutes)
            end_time = control_end + timedelta(minutes=after_minutes)

            control_info = {
                'group_id': group_id,
                'control_type': 'controlled',
                'start_time': start_time,
                'end_time': end_time,
                'control_start': control_start,
                'control_end': control_end,
                'before_minutes': before_minutes,
                'after_minutes': after_minutes,
                'total_duration_minutes': (end_time - start_time).total_seconds() / 60
            }

            control_regions.append(control_info)

        control_regions_df = pd.DataFrame(control_regions)

        self.logger.info(f"총 {len(control_regions_df)}개의 제어 구간 정보 생성")
        self.logger.info("="*80)

        return control_regions_df

    def sample_no_control_regions(
        self,
        original_df: pd.DataFrame,
        grouped_changes: List[dict],
        n_samples: int = 10,
        before_minutes: int = 5,
        after_minutes: int = 5,
        min_gap_minutes: int = 10
    ) -> pd.DataFrame:
        """
        5차 전처리: 비제어 구간 샘플링 (대조군)

        Parameters:
        -----------
        original_df : pd.DataFrame
            원본 데이터 DataFrame
        grouped_changes : List[dict]
            모든 그룹화된 변경 구간 리스트
        n_samples : int
            샘플링할 비제어 구간 개수
        before_minutes : int
            이전 버퍼 시간 (분)
        after_minutes : int
            이후 버퍼 시간 (분)
        min_gap_minutes : int
            제어점으로부터 최소 간격 (분)

        Returns:
        --------
        pd.DataFrame
            비제어 구간 정보
        """
        self.logger.info("="*80)
        self.logger.info("비제어 구간 샘플링")
        self.logger.info("="*80)
        self.logger.info(f"목표 샘플 수: {n_samples}")
        self.logger.info(f"분석 구간: 기준점 전 {before_minutes}분 ~ 후 {after_minutes}분")
        self.logger.info(f"제어점 최소 간격: {min_gap_minutes}분")

        # 데이터 시간 범위
        data_start = original_df[self.config.TIME_COL].min()
        data_end = original_df[self.config.TIME_COL].max()

        self.logger.info(f"데이터 시간 범위: {data_start} ~ {data_end}")

        # 필요한 총 시간 (분)
        required_duration = before_minutes + after_minutes

        # 제어 구간 목록 생성 (모든 grouped_changes)
        control_periods = []
        for group in grouped_changes:
            control_periods.append({
                'start': group['start_time'] - timedelta(minutes=min_gap_minutes),
                'end': group['end_time'] + timedelta(minutes=min_gap_minutes)
            })

        self.logger.info(f"제외할 제어 구간 수: {len(control_periods)}")

        # 비제어 구간 후보 샘플링
        no_control_regions = []
        max_attempts = n_samples * 100  # 최대 시도 횟수
        attempts = 0

        while len(no_control_regions) < n_samples and attempts < max_attempts:
            attempts += 1

            # 랜덤 시작 시점 선택
            time_range = (data_end - data_start).total_seconds()
            random_offset = np.random.uniform(0, time_range - required_duration * 60)
            candidate_start = data_start + timedelta(seconds=random_offset)
            candidate_end = candidate_start + timedelta(minutes=required_duration)

            # 범위 체크
            if candidate_end > data_end:
                continue

            # 제어 구간과 겹치는지 확인
            is_valid = True
            for control in control_periods:
                # 겹침 조건: candidate_start < control_end AND candidate_end > control_start
                if candidate_start < control['end'] and candidate_end > control['start']:
                    is_valid = False
                    break

            if is_valid:
                # 기준점 (중간 지점)
                reference_point = candidate_start + timedelta(minutes=before_minutes)

                region_info = {
                    'group_id': len(no_control_regions) + 1,
                    'control_type': 'no_control',
                    'start_time': candidate_start,
                    'end_time': candidate_end,
                    'reference_point': reference_point,  # 제어가 없는 시점
                    'before_minutes': before_minutes,
                    'after_minutes': after_minutes,
                    'total_duration_minutes': required_duration
                }

                no_control_regions.append(region_info)

                self.logger.debug(f"비제어 구간 {len(no_control_regions)}: {candidate_start} ~ {candidate_end}")

        if len(no_control_regions) == 0:
            self.logger.warning("비제어 구간을 찾지 못했습니다.")
            return pd.DataFrame()

        no_control_df = pd.DataFrame(no_control_regions)

        self.logger.info(f"총 {len(no_control_df)}개의 비제어 구간 샘플링 완료 (시도 횟수: {attempts})")
        self.logger.info("="*80)

        return no_control_df

    def load_llspec_data(
        self,
        llspec_file: str,
        side: str = 'C'
    ) -> Optional[pd.DataFrame]:
        """
        Loading Level Spec 데이터 로드 및 전처리

        Parameters:
        -----------
        llspec_file : str
            LLspec 엑셀 파일 경로
        side : str
            필터링할 SIDE 값 ('C' 또는 'D')

        Returns:
        --------
        pd.DataFrame
            전처리된 LLspec DataFrame (TIME, UCL, TARGET, LCL)
        """
        self.logger.info("="*80)
        self.logger.info("Loading Level Spec 데이터 로드 중...")
        self.logger.info("="*80)

        # 파일 확장자 확인
        file_ext = os.path.splitext(llspec_file)[1].lower()

        # Excel 파일의 경우 멀티레벨 헤더 처리를 위해 xlwings 사용
        if file_ext in ['.xlsx', '.xls']:
            app = xw.App(visible=False)
            wb = None
            try:
                wb = xw.Book(llspec_file)
                sheet = wb.sheets[0]

                # 데이터 읽기
                used_range = sheet.used_range
                data = used_range.value

                # 헤더 처리 (멀티레벨 헤더 가능성)
                headers_row1 = data[0]
                headers_row2 = data[1] if len(data) > 1 else None

                # DataFrame 생성
                if headers_row2:
                    # 멀티레벨 헤더 처리
                    df = pd.DataFrame(data[2:], columns=headers_row2)
                else:
                    df = pd.DataFrame(data[1:], columns=headers_row1)

                self.logger.info(f"Excel 파일 로드 완료 (xlwings)")
            except Exception as e:
                self.logger.error(f"Excel 파일 로드 오류: {e}", exc_info=True)
                return None
            finally:
                if wb:
                    wb.close()
                app.quit()
        else:
            # Parquet/CSV 파일의 경우 utils.load_file() 사용
            try:
                df = utils.load_file(llspec_file, logger=self.logger)
                self.logger.info(f"파일 로드 완료 (utils.load_file)")
            except Exception as e:
                self.logger.error(f"파일 로드 오류: {e}", exc_info=True)
                return None

        self.logger.info(f"원본 LLspec 데이터 행 수: {len(df)}")
        self.logger.debug(f"원본 칼럼: {list(df.columns)}")

        try:

            # SIDE 필터링
            if self.config.SIDE_COL in df.columns:
                df = df[df[self.config.SIDE_COL] == side].reset_index(drop=True)
                self.logger.info(f"SIDE='{side}' 필터링 후 행 수: {len(df)}")
            else:
                self.logger.warning(f"{self.config.SIDE_COL} 칼럼이 없습니다. 필터링을 건너뜁니다.")

            # TIME 칼럼을 datetime으로 변환
            if self.config.TIME_COL in df.columns:
                df[self.config.TIME_COL] = pd.to_datetime(df[self.config.TIME_COL])
            else:
                self.logger.error(f"{self.config.TIME_COL} 칼럼이 없습니다.")
                return None

            # UCL, TARGET, LCL 칼럼 찾기
            spec_cols = []

            # 다양한 칼럼명 패턴 시도
            col_patterns = {
                'UCL': ['UCL', 'ucl', 'Upper Control Limit'],
                'TARGET': ['TARGET', 'target', 'Target'],
                'LCL': ['LCL', 'lcl', 'Lower Control Limit']
            }

            for spec_name, patterns in col_patterns.items():
                found = False
                for pattern in patterns:
                    if pattern in df.columns:
                        spec_cols.append(pattern)
                        found = True
                        break

                if not found:
                    self.logger.warning(f"{spec_name} 칼럼을 찾을 수 없습니다.")

            if len(spec_cols) == 0:
                self.logger.error("UCL, TARGET, LCL 칼럼을 찾을 수 없습니다.")
                return None

            # 필요한 칼럼만 추출
            selected_cols = [self.config.TIME_COL] + spec_cols
            llspec_df = df[selected_cols].copy()

            # 칼럼명 표준화
            rename_dict = {}
            for col in spec_cols:
                if 'UCL' in col.upper():
                    rename_dict[col] = 'UCL'
                elif 'TARGET' in col.upper():
                    rename_dict[col] = 'TARGET'
                elif 'LCL' in col.upper():
                    rename_dict[col] = 'LCL'

            llspec_df = llspec_df.rename(columns=rename_dict)

            # NaN 제거
            llspec_df = llspec_df.dropna(subset=[self.config.TIME_COL]).reset_index(drop=True)

            self.logger.info(f"최종 LLspec 데이터 행 수: {len(llspec_df)}")
            self.logger.info(f"칼럼: {list(llspec_df.columns)}")
            self.logger.info(f"시간 범위: {llspec_df[self.config.TIME_COL].min()} ~ {llspec_df[self.config.TIME_COL].max()}")

            if 'UCL' in llspec_df.columns and 'LCL' in llspec_df.columns:
                self.logger.info(f"UCL 범위: {llspec_df['UCL'].min():.4f} ~ {llspec_df['UCL'].max():.4f}")
                self.logger.info(f"LCL 범위: {llspec_df['LCL'].min():.4f} ~ {llspec_df['LCL'].max():.4f}")

            if 'TARGET' in llspec_df.columns:
                self.logger.info(f"TARGET 범위: {llspec_df['TARGET'].min():.4f} ~ {llspec_df['TARGET'].max():.4f}")

            self.logger.info("="*80)

            return llspec_df

        except FileNotFoundError:
            self.logger.error(f"'{llspec_file}' 파일을 찾을 수 없습니다.")
            return None
        except Exception as e:
            self.logger.error(f"오류: {e}", exc_info=True)
            return None

    def match_llspec_to_groups(
        self,
        meaningful_df: pd.DataFrame,
        llspec_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        meaningful_df의 각 group에 LLspec 값 매칭

        Parameters:
        -----------
        meaningful_df : pd.DataFrame
            유의미한 변경 구간 DataFrame
        llspec_df : pd.DataFrame
            LLspec DataFrame (TIME, UCL, TARGET, LCL)

        Returns:
        --------
        pd.DataFrame
            LLspec 정보가 추가된 DataFrame
        """
        self.logger.info("="*80)
        self.logger.info("LLspec 데이터 매칭 중...")
        self.logger.info("="*80)

        if llspec_df is None or llspec_df.empty:
            self.logger.warning("LLspec 데이터가 없습니다. 매칭을 건너뜁니다.")
            return meaningful_df

        # LLspec 칼럼 확인
        available_spec_cols = [col for col in ['UCL', 'TARGET', 'LCL'] if col in llspec_df.columns]

        if not available_spec_cols:
            self.logger.warning("UCL, TARGET, LCL 칼럼이 없습니다. 매칭을 건너뜁니다.")
            return meaningful_df

        # meaningful_df에 LLspec 칼럼 추가 (NaN으로 초기화)
        for col in available_spec_cols:
            meaningful_df[col] = np.nan

        # 각 group에 대해 start_time 기준으로 가장 가까운 LLspec 값 매칭
        for idx, row in meaningful_df.iterrows():
            start_time = row['start_time']

            # start_time과 가장 가까운 TIME 찾기
            time_diffs = (llspec_df[self.config.TIME_COL] - start_time).abs()
            closest_idx = time_diffs.idxmin()

            # 시간 차이 확인 (너무 멀면 매칭하지 않음)
            time_diff_seconds = time_diffs.iloc[closest_idx].total_seconds()

            if time_diff_seconds > 300:  # 5분 이상 차이나면 매칭하지 않음
                self.logger.warning(f"Group {row['group_id']}: 가장 가까운 LLspec 데이터와 {time_diff_seconds/60:.1f}분 차이 (매칭 건너뜀)")
                continue

            # LLspec 값 매칭
            for col in available_spec_cols:
                meaningful_df.at[idx, col] = llspec_df.at[closest_idx, col]

            self.logger.info(f"Group {row['group_id']}: LLspec 매칭 완료 (시간 차이: {time_diff_seconds:.1f}초)")
            if 'UCL' in available_spec_cols and 'LCL' in available_spec_cols:
                self.logger.debug(f"  UCL: {meaningful_df.at[idx, 'UCL']:.4f}, LCL: {meaningful_df.at[idx, 'LCL']:.4f}")

        # 매칭 통계
        matched_count = meaningful_df[available_spec_cols[0]].notna().sum()
        total_count = len(meaningful_df)

        self.logger.info("="*80)
        self.logger.info("매칭 결과 요약")
        self.logger.info("="*80)
        self.logger.info(f"전체 그룹 수: {total_count}")
        self.logger.info(f"매칭 성공: {matched_count} ({matched_count/total_count*100:.1f}%)")
        self.logger.info(f"매칭 실패: {total_count - matched_count} ({(total_count - matched_count)/total_count*100:.1f}%)")
        self.logger.info("="*80)

        return meaningful_df
