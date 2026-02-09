"""
전처리 파이프라인 메인 클래스 (모델 데이터 준비 기능 추가)
"""

import pandas as pd
from typing import List, Union, Optional
import os
import logging

from config import PreprocessConfig
from utils import save_to_excel, ensure_directory, setup_logger, get_files_from_folder
# 아래 모듈들은 사용자가 구현해야 함 (문맥상 import 경로 유지)
from preprocessor.apc_preprocessor import APCPreprocessor
from preprocessor.densitometer_preprocessor import DensitometerPreprocessor
from preprocessor.zone_analyzer import ZoneAnalyzer
from preprocessor.data_merger import DataMerger
from preprocessor.model_data_preparator import ModelDataPreparator

class CoatingPreprocessPipeline:
    """
    코팅 L/L 제어 전처리 파이프라인
    전체 전처리 과정을 통합 관리하는 메인 클래스
    """

    def __init__(self, config: PreprocessConfig = None, logger: logging.Logger = None):
        """
        Parameters:
        -----------
        config : PreprocessConfig, optional
            설정 객체 (None이면 기본 설정 사용)
        logger : logging.Logger, optional
            로거 객체 (None이면 새로 생성)
        """
        self.config = config if config else PreprocessConfig()

        # 로거 설정
        if logger is None:
            ensure_directory(self.config.LOG_DIR)
            log_file = os.path.join(self.config.LOG_DIR, self.config.LOG_FILE)
            self.logger = setup_logger(
                name='coating_preprocessor',
                log_file=log_file,
                level=self.config.LOG_LEVEL,
                console_output=self.config.LOG_CONSOLE_OUTPUT
            )
        else:
            self.logger = logger

        # 전처리 모듈 초기화
        self.apc_preprocessor = APCPreprocessor(self.config, self.logger)
        self.densitometer_preprocessor = DensitometerPreprocessor(self.config, self.logger)
        self.zone_analyzer = ZoneAnalyzer(self.config, self.logger)
        self.data_merger = DataMerger(self.logger)
        self.model_data_preparator = ModelDataPreparator(self.config, self.logger)

        # 결과 저장용
        self.results = {}

        # 출력 디렉토리 생성
        ensure_directory(self.config.OUTPUT_DIR, self.logger)
        ensure_directory(self.config.PLOT_DIR, self.logger)

    def run_single_file(
        self,
        apc_file: str,
        densitometer_file: str,
        llspec_file: Optional[str] = None,
        visualize: bool = True,
        prepare_model_data: bool = True,
        mode: str = 'training'
    ):
        """
        단일 파일에 대한 전체 전처리 실행

        Parameters:
        -----------
        apc_file : str
            APC 데이터 파일 경로
        densitometer_file : str
            밀도계 데이터 파일 경로
        llspec_file : str, optional
            LLspec 데이터 파일 경로
        visualize : bool
            시각화 여부
        prepare_model_data : bool
            모델 데이터 준비 여부
        mode : str
            데이터 모드 ('training' 또는 'test'), 기본값: 'training'
        """
        self.logger.info("="*80)
        self.logger.info(f"전처리 파이프라인 시작 (단일 파일 모드 - {mode.upper()})")
        self.logger.info("="*80)
        self.logger.info(f"APC 파일: {apc_file}")
        self.logger.info(f"밀도계 파일: {densitometer_file}")
        if llspec_file:
            self.logger.info(f"LLspec 파일: {llspec_file}")
        self.logger.info("="*80)

        # Step 1: APC 데이터 전처리
        self.logger.info("[Step 1] APC 데이터 전처리")
        meaningful_changes = self.apc_preprocessor.run(
            input_file=apc_file,
            llspec_file=llspec_file
        )

        if meaningful_changes is None or meaningful_changes.empty:
            self.logger.error("APC 전처리 실패 또는 유의미한 변경 구간 없음")
            return

        self.results['meaningful_changes'] = meaningful_changes

        # Step 2: 밀도계 데이터 추출
        self.logger.info("[Step 2] 밀도계 데이터 추출")
        extracted_data = self.densitometer_preprocessor.run(
            changes_file=os.path.join(
                self.config.OUTPUT_DIR,
                self.config.OUTPUT_3RD
            ),
            raw_data_file=densitometer_file
        )

        if extracted_data is None or extracted_data.empty:
            self.logger.error("밀도계 데이터 추출 실패")
            return

        self.results['extracted_densitometer'] = extracted_data

        # Step 3: Zone 분석
        self.logger.info("[Step 3] Zone별 분석")
        zone_results = self.zone_analyzer.run(
            densitometer_data=extracted_data,
            meaningful_changes=meaningful_changes,
            visualize=visualize
        )

        if zone_results is None or zone_results.empty:
            self.logger.error("Zone 분석 실패")
            return

        self.results['zone_analysis'] = zone_results

        # Step 4: 모델 학습용 데이터 준비 (옵션)
        if prepare_model_data:
            self.logger.info(f"[Step 4] 모델 {mode.upper()} 데이터 준비")

            zone_analysis_file = os.path.join(
                self.config.OUTPUT_DIR,
                self.config.OUTPUT_ZONE_ANALYSIS
            )
            meaningful_changes_file = os.path.join(
                self.config.OUTPUT_DIR,
                self.config.OUTPUT_3RD
            )

            # mode 파라미터 전달
            model_data = self.model_data_preparator.run(
                zone_analysis_file=zone_analysis_file,
                meaningful_changes_file=meaningful_changes_file,
                mode=mode  # mode 전달
            )

            if model_data is not None and not model_data.empty:
                self.results[f'model_{mode}_data'] = model_data
                self.logger.info(f"  ✓ {mode.upper()} 데이터 준비 완료: {len(model_data)} 샘플")
            else:
                self.logger.warning(f"  ⚠ 모델 {mode.upper()} 데이터 준비 실패")

        # 최종 요약
        self.logger.info("="*80)
        self.logger.info("전처리 파이프라인 완료")
        self.logger.info("="*80)
        self.logger.info(f"유의미한 변경 구간: {len(meaningful_changes)}")
        self.logger.info(f"추출된 밀도계 데이터: {len(extracted_data)} 행")
        self.logger.info(f"Zone 분석 결과: {len(zone_results)} 행")

        if prepare_model_data and f'model_{mode}_data' in self.results:
            self.logger.info(f"모델 {mode.upper()} 데이터: {len(self.results[f'model_{mode}_data'])} 샘플")

        self.logger.info(f"결과 저장 위치: {self.config.OUTPUT_DIR}")
        self.logger.info(f"로그 파일: {os.path.join(self.config.LOG_DIR, self.config.LOG_FILE)}")
        self.logger.info("="*80)

    def run_multiple_files(
        self,
        apc_files: List[str],
        densitometer_files: List[str],
        llspec_files: Optional[List[str]] = None,
        visualize: bool = True,
        prepare_model_data: bool = True,
        mode: str = 'training'
    ):
        """
        여러 파일을 통합하여 전처리 실행

        Parameters:
        -----------
        apc_files : List[str]
            APC 파일 경로 리스트
        densitometer_files : List[str]
            밀도계 파일 경로 리스트
        llspec_files : List[str], optional
            LLspec 파일 경로 리스트
        visualize : bool
            시각화 여부
        prepare_model_data : bool
            모델 데이터 준비 여부
        mode : str
            데이터 모드 ('training' 또는 'test'), 기본값: 'training'
        """
        self.logger.info("="*80)
        self.logger.info(f"전처리 파이프라인 시작 (다중 파일 모드 - {mode.upper()})")
        self.logger.info("="*80)
        self.logger.info(f"APC 파일 수: {len(apc_files)}")
        self.logger.info(f"밀도계 파일 수: {len(densitometer_files)}")
        if llspec_files:
            self.logger.info(f"LLspec 파일 수: {len(llspec_files)}")
        self.logger.info("="*80)

        # Step 0: 파일 통합
        self.logger.info("[Step 0] 데이터 파일 통합")

        # APC 파일 통합
        merged_apc = self.data_merger.merge_apc_files(apc_files)
        if merged_apc is None:
            self.logger.error("APC 파일 통합 실패")
            return

        # 통합된 APC 파일 임시 저장
        temp_apc_file = os.path.join(self.config.OUTPUT_DIR, 'temp_merged_apc.xlsx')
        save_to_excel(merged_apc, temp_apc_file, 'Merged_APC', self.logger)

        # 밀도계 파일 통합
        merged_densitometer = self.data_merger.merge_densitometer_files(densitometer_files)
        if merged_densitometer is None:
            self.logger.error("밀도계 파일 통합 실패")
            return

        # 통합된 밀도계 파일 임시 저장
        temp_densitometer_file = os.path.join(
            self.config.OUTPUT_DIR,
            'temp_merged_densitometer.xlsx'
        )
        save_to_excel(merged_densitometer, temp_densitometer_file, 'Merged_Densitometer', self.logger)

        # LLspec 파일 통합 (있는 경우)
        temp_llspec_file = None
        if llspec_files:
            merged_llspec = self.data_merger.merge_llspec_files(llspec_files)
            if merged_llspec is not None:
                temp_llspec_file = os.path.join(
                    self.config.OUTPUT_DIR,
                    'temp_merged_llspec.xlsx'
                )
                save_to_excel(merged_llspec, temp_llspec_file, 'Merged_LLspec', self.logger)

        # 단일 파일 모드로 전처리 실행 (mode 전달)
        self.run_single_file(
            apc_file=temp_apc_file,
            densitometer_file=temp_densitometer_file,
            llspec_file=temp_llspec_file,
            visualize=visualize,
            prepare_model_data=prepare_model_data,
            mode=mode  # mode 전달
        )

    def run_from_folder(
        self,
        folder_path: str,
        apc_pattern: str = 'apc*.xlsx',
        densitometer_pattern: str = 'densitometer*.xlsx',
        llspec_pattern: str = 'llspec*.xlsx',
        visualize: bool = True,
        prepare_model_data: bool = True,
        mode: str = 'training'
    ):
        """
        폴더에서 패턴에 맞는 파일들을 자동으로 찾아서 전처리 실행

        Parameters:
        -----------
        folder_path : str
            데이터 폴더 경로
        apc_pattern : str
            APC 파일 패턴
        densitometer_pattern : str
            밀도계 파일 패턴
        llspec_pattern : str
            LLspec 파일 패턴
        visualize : bool
            시각화 여부
        prepare_model_data : bool
            모델 데이터 준비 여부
        mode : str
            데이터 모드 ('training' 또는 'test'), 기본값: 'training'
        """
        self.logger.info("="*80)
        self.logger.info(f"전처리 파이프라인 시작 (폴더 모드 - {mode.upper()})")
        self.logger.info("="*80)
        self.logger.info(f"폴더 경로: {folder_path}")
        self.logger.info("="*80)

        # 파일 검색
        apc_files = get_files_from_folder(folder_path, apc_pattern, self.logger)
        densitometer_files = get_files_from_folder(folder_path, densitometer_pattern, self.logger)
        llspec_files = get_files_from_folder(folder_path, llspec_pattern, self.logger)

        self.logger.info(f"발견된 파일:")
        self.logger.info(f"  APC: {len(apc_files)}개")
        self.logger.info(f"  밀도계: {len(densitometer_files)}개")
        self.logger.info(f"  LLspec: {len(llspec_files)}개")

        if not apc_files or not densitometer_files:
            self.logger.error("필수 파일이 없습니다.")
            return

        # 다중 파일 모드로 실행 (mode 전달)
        self.run_multiple_files(
            apc_files=apc_files,
            densitometer_files=densitometer_files,
            llspec_files=llspec_files if llspec_files else None,
            visualize=visualize,
            prepare_model_data=prepare_model_data,
            mode=mode  # mode 전달
        )

    def get_results(self):
        """전처리 결과 반환"""
        return self.results
