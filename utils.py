"""
공통 유틸리티 함수
"""

import pandas as pd
import xlwings as xw
from datetime import datetime
from typing import List, Union
import os
import logging
from logging.handlers import RotatingFileHandler

def setup_logger(
    name: str = 'coating_preprocessor',
    log_file: str = 'preprocessing.log',
    level: int = logging.INFO,
    console_output: bool = True
) -> logging.Logger:
    """
    로거 설정

    Parameters:
    -----------
    name : str
        로거 이름
    log_file : str
        로그 파일 경로
    level : int
        로깅 레벨
    console_output : bool
        콘솔 출력 여부

    Returns:
    --------
    logging.Logger
        설정된 로거
    """
    # 로거 생성
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # 기존 핸들러 제거 (중복 방지)
    if logger.handlers:
        logger.handlers.clear()

    # 포맷터 설정
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # 파일 핸들러 (최대 10MB, 백업 5개)
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=10*1024*1024,
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # 콘솔 핸들러
    if console_output:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger

def save_to_excel(df: pd.DataFrame, output_path: str, sheet_name: str = 'Sheet1', logger: logging.Logger = None):
    """DataFrame을 엑셀로 저장"""
    if logger is None:
        logger = logging.getLogger('coating_preprocessor')

    if df is None or df.empty:
        logger.warning(f"저장할 데이터가 없습니다: {output_path}")
        return

    app = xw.App(visible=False)
    try:
        wb = xw.Book()
        sheet = wb.sheets[0]
        sheet.name = sheet_name

        sheet.range('A1').value = df
        sheet.range('A1').expand('right').api.Font.Bold = True
        sheet.autofit()

        wb.save(output_path)
        logger.info(f"저장 완료: '{output_path}'")

    except Exception as e:
        logger.error(f"저장 오류: {e}")
        raise
    finally:
        wb.close()
        app.quit()

def parse_time(time_str, reference_date='2024-01-01'):
    """시간 문자열을 datetime으로 변환"""
    if isinstance(time_str, str):
        return pd.to_datetime(f"{reference_date} {time_str}")
    return time_str

def load_excel_file(file_path: str, sheet_name: Union[str, int] = 0, logger: logging.Logger = None) -> pd.DataFrame:
    """엑셀 파일 로드"""
    if logger is None:
        logger = logging.getLogger('coating_preprocessor')

    try:
        if file_path.endswith('.csv'):
            return pd.read_csv(file_path)
        elif file_path.endswith(('.xlsx', '.xls')):
            return pd.read_excel(file_path, sheet_name=sheet_name)
        else:
            raise ValueError(f"지원하지 않는 파일 형식: {file_path}")
    except FileNotFoundError:
        logger.error(f"파일을 찾을 수 없습니다: {file_path}")
        return None
    except Exception as e:
        logger.error(f"파일 로드 오류: {e}")
        return None

def ensure_directory(directory: str, logger: logging.Logger = None):
    """디렉토리가 없으면 생성"""
    if logger is None:
        logger = logging.getLogger('coating_preprocessor')

    os.makedirs(directory, exist_ok=True)
    logger.debug(f"디렉토리 확인/생성: {directory}")

def get_files_from_folder(folder_path: str, pattern: str = '*.xlsx', logger: logging.Logger = None) -> List[str]:
    """폴더에서 패턴에 맞는 파일 목록 반환"""
    if logger is None:
        logger = logging.getLogger('coating_preprocessor')

    import glob
    files = glob.glob(os.path.join(folder_path, pattern))
    logger.debug(f"폴더 '{folder_path}'에서 패턴 '{pattern}' 매칭: {len(files)}개 파일 발견")
    return files
