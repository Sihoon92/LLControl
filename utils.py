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

def _load_excel_with_xlwings(
    file_path: str,
    sheet_name: Union[str, int] = 0,
    logger: logging.Logger = None
) -> pd.DataFrame:
    """
    xlwings를 사용하여 엑셀 파일 로드 (회사 내부 이슈로 pandas.read_excel 사용 불가)

    Parameters:
    -----------
    file_path : str
        엑셀 파일 경로
    sheet_name : Union[str, int]
        시트 이름 또는 인덱스 (기본값: 0)
    logger : logging.Logger
        로거

    Returns:
    --------
    pd.DataFrame
        로드된 데이터프레임
    """
    if logger is None:
        logger = logging.getLogger('coating_preprocessor')

    app = xw.App(visible=False)
    wb = None
    try:
        # 엑셀 파일 열기
        wb = app.books.open(file_path)

        # 시트 선택
        if isinstance(sheet_name, int):
            sheet = wb.sheets[sheet_name]
        else:
            sheet = wb.sheets[sheet_name]

        # 데이터 읽기 (헤더 포함)
        data = sheet.used_range.options(pd.DataFrame, header=1, index=False).value

        logger.debug(f"Excel 로드 완료: {file_path} (시트: {sheet_name}, 행: {len(data) if data is not None else 0})")

        return data

    except Exception as e:
        logger.error(f"xlwings Excel 로드 오류 ({file_path}): {e}")
        raise
    finally:
        # 리소스 정리
        if wb:
            wb.close()
        app.quit()


def load_file(
    file_path: str,
    sheet_name: Union[str, int] = 0,
    logger: logging.Logger = None
) -> pd.DataFrame:
    """
    파일 형식에 따라 자동으로 로드

    지원 형식:
    - Excel (.xlsx, .xls): xlwings로 로드 (회사 내부 이슈로 pandas 사용 불가)
    - Parquet (.parquet): pandas로 로드
    - CSV (.csv): pandas로 로드

    Parameters:
    -----------
    file_path : str
        파일 경로
    sheet_name : Union[str, int]
        엑셀 시트 이름 또는 인덱스 (엑셀 파일만 해당, 기본값: 0)
    logger : logging.Logger
        로거

    Returns:
    --------
    pd.DataFrame
        로드된 데이터프레임

    Raises:
    -------
    FileNotFoundError
        파일이 존재하지 않을 때
    ValueError
        지원하지 않는 파일 형식일 때

    Examples:
    ---------
    >>> # Excel 파일 로드 (xlwings)
    >>> df = load_file('data.xlsx')
    >>> df = load_file('data.xlsx', sheet_name='Sheet2')
    >>> df = load_file('data.xlsx', sheet_name=1)
    >>>
    >>> # CSV 파일 로드
    >>> df = load_file('data.csv')
    >>>
    >>> # Parquet 파일 로드
    >>> df = load_file('data.parquet')
    """
    if logger is None:
        logger = logging.getLogger('coating_preprocessor')

    try:
        # 파일 존재 확인
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"파일을 찾을 수 없습니다: {file_path}")

        # 파일 확장자 확인
        file_ext = os.path.splitext(file_path)[1].lower()

        if file_ext == '.csv':
            logger.debug(f"CSV 파일 로드: {file_path}")
            return pd.read_csv(file_path)

        elif file_ext == '.parquet':
            logger.debug(f"Parquet 파일 로드: {file_path}")
            return pd.read_parquet(file_path)

        elif file_ext in ['.xlsx', '.xls']:
            logger.debug(f"Excel 파일 로드 (xlwings): {file_path}")
            return _load_excel_with_xlwings(file_path, sheet_name, logger)

        else:
            raise ValueError(f"지원하지 않는 파일 형식: {file_ext} (지원: .xlsx, .xls, .csv, .parquet)")

    except FileNotFoundError as e:
        logger.error(str(e))
        raise
    except Exception as e:
        logger.error(f"파일 로드 오류 ({file_path}): {e}")
        raise


def load_excel_file(file_path: str, sheet_name: Union[str, int] = 0, logger: logging.Logger = None) -> pd.DataFrame:
    """
    엑셀 파일 로드 (하위 호환성을 위한 래퍼 함수)

    ⚠️  Deprecated: load_file() 사용을 권장합니다.

    Parameters:
    -----------
    file_path : str
        파일 경로
    sheet_name : Union[str, int]
        엑셀 시트 이름 또는 인덱스
    logger : logging.Logger
        로거

    Returns:
    --------
    pd.DataFrame
        로드된 데이터프레임
    """
    return load_file(file_path, sheet_name=sheet_name, logger=logger)

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
