"""
v1.2 호환 테스트 스크립트
3rd_meaningful_changes 파일에서 직접 밀도계 데이터 추출
"""

import sys
import os
import logging

# 프로젝트 경로 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from preprocessor.densitometer_preprocessor import DensitometerPreprocessor
from preprocessor.preprocess_config import PreprocessConfig


def setup_logger():
    """로거 설정"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/densitometer_test.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger('densitometer_test')


def main():
    """메인 실행 함수"""

    # 로거 설정
    logger = setup_logger()
    logger.info("="*80)
    logger.info("밀도계 데이터 추출 테스트 (v1.2 호환)")
    logger.info("="*80)

    # Config 생성
    config = PreprocessConfig()

    # 출력 디렉토리 생성
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    os.makedirs(config.LOG_DIR, exist_ok=True)

    try:
        # ===== 방법 1: 3rd_meaningful_changes 파일 사용 (v1.2 호환) =====
        logger.info("\n[방법 1] 3rd_meaningful_changes 파일 사용")
        logger.info("-" * 80)

        densitometer_processor = DensitometerPreprocessor(config, logger)

        meaningful_changes_file = os.path.join(config.OUTPUT_DIR, config.OUTPUT_3RD)
        raw_data_file = 'data/densitometer_raw.xlsx'  # TODO: 실제 경로로 변경

        if not os.path.exists(meaningful_changes_file):
            logger.error(f"3rd_meaningful_changes 파일을 찾을 수 없습니다: {meaningful_changes_file}")
            logger.info("먼저 APC 전처리를 실행하세요.")
            return

        if not os.path.exists(raw_data_file):
            logger.error(f"밀도계 raw data 파일을 찾을 수 없습니다: {raw_data_file}")
            return

        # 밀도계 데이터 추출
        # 제어 전 1분 / 후 6분 데이터 추출
        extracted_data = densitometer_processor.run_from_meaningful_changes(
            meaningful_changes_file=meaningful_changes_file,
            raw_data_file=raw_data_file,
            before_minutes=1,   # 제어 전 1분
            after_minutes=6     # 제어 후 6분
        )

        if extracted_data is None or extracted_data.empty:
            logger.error("밀도계 데이터 추출 실패")
            return

        logger.info(f"✓ 밀도계 데이터 추출 완료: {len(extracted_data)}행")
        logger.info(f"  - Before 데이터: {(extracted_data['before/after'] == 'before').sum()}행")
        logger.info(f"  - After 데이터: {(extracted_data['before/after'] == 'after').sum()}행")

        # ===== 완료 =====
        logger.info("\n" + "="*80)
        logger.info("밀도계 데이터 추출 완료!")
        logger.info("="*80)
        logger.info(f"\n생성된 파일: {config.OUTPUT_DIR}/{config.OUTPUT_DENSITOMETER}")

    except Exception as e:
        logger.error(f"오류 발생: {e}", exc_info=True)
        raise


if __name__ == '__main__':
    main()
