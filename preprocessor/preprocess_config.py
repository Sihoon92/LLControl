"""
전처리 파이프라인 설정
"""

import logging

class PreprocessConfig:
    """전처리 설정 클래스"""

    # 시간 설정
    TIME_THRESHOLD_MINUTES = 2      # 그룹화 시간 임계값
    BEFORE_MINUTES = 5              # 이전 버퍼 시간
    AFTER_MINUTES = 5               # 이후 버퍼 시간

    # 필터링 설정
    SIDE_FILTER = 'C'               # SIDE 필터 값
    BOUNDARY_THRESHOLD = 0.9        # 경계 검출 임계값

    # Zone 설정
    N_ZONES = 11                    # Zone 개수
    N_DIVISIONS = 6                 # 균등 분할 개수

    # 파일 경로
    OUTPUT_DIR = './outputs'
    PLOT_DIR = './outputs/plots'
    LOG_DIR = './logs'

    # 로그 설정
    LOG_FILE = 'preprocessing.log'
    LOG_LEVEL = logging.INFO
    LOG_CONSOLE_OUTPUT = True

    # 출력 파일명
    OUTPUT_1ST = '1st_all_changes.xlsx'
    OUTPUT_2ND = '2nd_grouped_changes.xlsx'
    OUTPUT_3RD = '3rd_meaningful_changes.xlsx'
    OUTPUT_4TH_CONTROL = '4th_control_regions.xlsx'
    OUTPUT_5TH_NO_CONTROL = '5th_no_control_regions.xlsx'
    OUTPUT_DENSITOMETER = 'extracted_densitometer_data.xlsx'
    OUTPUT_ZONE_ANALYSIS = 'zone_analysis_results.xlsx'
    OUTPUT_STATISTICAL_ANALYSIS = 'statistical_analysis_results.xlsx'
    OUTPUT_FINAL_STATS = 'final_zone_statistics.xlsx'
    OUTPUT_FINAL_DATA = 'final_preprocessed_data.xlsx'
    OUTPUT_MODEL_DATA = 'model_training_data.xlsx'

    # 칼럼 패턴
    GV_GAP_PATTERN = 'GV_GAP'
    GV_ZONE_PATTERN = 'GV_ZONE'
    PUMP_RPM_COL = 'PUMP RPM'
    TIME_COL = 'TIME'
    SIDE_COL = 'SIDE'
