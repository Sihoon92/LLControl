"""
여러 파일을 통합하는 유틸리티
"""

import pandas as pd
from typing import List
import os
import logging
import utils


class DataMerger:
    """데이터 파일 통합 클래스"""

    def __init__(self, logger: logging.Logger = None):
        """
        Parameters:
        -----------
        logger : logging.Logger
            로거 객체
        """
        self.logger = logger or logging.getLogger('coating_preprocessor.merger')

    def merge_apc_files(self, file_list: List[str]) -> pd.DataFrame:
        """
        여러 APC 파일을 하나로 통합

        Parameters:
        -----------
        file_list : List[str]
            통합할 파일 경로 리스트

        Returns:
        --------
        pd.DataFrame
            통합된 데이터프레임
        """
        self.logger.info("="*80)
        self.logger.info("APC 파일 통합 시작")
        self.logger.info("="*80)

        all_data = []

        for file_path in file_list:
            self.logger.info(f"처리 중: {os.path.basename(file_path)}")

            try:
                # 파일 형식에 따라 자동 로드 (Excel: xlwings, CSV/Parquet: pandas)
                df = utils.load_file(file_path, logger=self.logger)

                self.logger.info(f"  ✓ {len(df)} 행 로드")
                all_data.append(df)

            except Exception as e:
                self.logger.error(f"  ✗ 오류: {e}")
                continue

        if not all_data:
            self.logger.error("통합할 데이터가 없습니다.")
            return None

        # 데이터 통합
        merged_df = pd.concat(all_data, ignore_index=True)

        # TIME 칼럼으로 정렬
        if 'TIME' in merged_df.columns:
            merged_df['TIME'] = pd.to_datetime(merged_df['TIME'])
            merged_df = merged_df.sort_values('TIME').reset_index(drop=True)

        self.logger.info("="*80)
        self.logger.info(f"통합 완료: {len(merged_df)} 행")
        self.logger.info("="*80)

        return merged_df

    def merge_densitometer_files(self, file_list: List[str]) -> pd.DataFrame:
        """
        여러 밀도계 파일을 하나로 통합

        Parameters:
        -----------
        file_list : List[str]
            통합할 파일 경로 리스트

        Returns:
        --------
        pd.DataFrame
            통합된 데이터프레임
        """
        self.logger.info("="*80)
        self.logger.info("밀도계 파일 통합 시작")
        self.logger.info("="*80)

        all_data = []

        for file_path in file_list:
            self.logger.info(f"처리 중: {os.path.basename(file_path)}")

            try:
                # 파일 형식에 따라 자동 로드 (Excel: xlwings, CSV/Parquet: pandas)
                df = utils.load_file(file_path, logger=self.logger)

                self.logger.info(f"  ✓ {len(df)} 행 로드")
                all_data.append(df)

            except Exception as e:
                self.logger.error(f"  ✗ 오류: {e}")
                continue

        if not all_data:
            self.logger.error("통합할 데이터가 없습니다.")
            return None

        # 데이터 통합
        merged_df = pd.concat(all_data, ignore_index=True)

        # 첫 번째 칼럼(시간)으로 정렬
        time_col = merged_df.columns[0]
        merged_df[time_col] = pd.to_datetime(merged_df[time_col])
        merged_df = merged_df.sort_values(time_col).reset_index(drop=True)

        self.logger.info("="*80)
        self.logger.info(f"통합 완료: {len(merged_df)} 행")
        self.logger.info("="*80)

        return merged_df

    def merge_llspec_files(self, file_list: List[str]) -> pd.DataFrame:
        """
        여러 LLspec 파일을 하나로 통합

        Parameters:
        -----------
        file_list : List[str]
            통합할 파일 경로 리스트

        Returns:
        --------
        pd.DataFrame
            통합된 데이터프레임
        """
        self.logger.info("="*80)
        self.logger.info("LLspec 파일 통합 시작")
        self.logger.info("="*80)

        all_data = []

        for file_path in file_list:
            self.logger.info(f"처리 중: {os.path.basename(file_path)}")

            try:
                # 파일 형식에 따라 자동 로드 (Excel: xlwings, CSV/Parquet: pandas)
                df = utils.load_file(file_path, logger=self.logger)

                self.logger.info(f"  ✓ {len(df)} 행 로드")
                all_data.append(df)

            except Exception as e:
                self.logger.error(f"  ✗ 오류: {e}")
                continue

        if not all_data:
            self.logger.error("통합할 데이터가 없습니다.")
            return None

        # 데이터 통합
        merged_df = pd.concat(all_data, ignore_index=True)

        # TIME 칼럼으로 정렬
        if 'TIME' in merged_df.columns:
            merged_df['TIME'] = pd.to_datetime(merged_df['TIME'])
            merged_df = merged_df.sort_values('TIME').reset_index(drop=True)

        self.logger.info("="*80)
        self.logger.info(f"통합 완료: {len(merged_df)} 행")
        self.logger.info("="*80)

        return merged_df
