"""
통계 분석 기능 테스트 스크립트
"""

import sys
import os
import logging

# 프로젝트 경로 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from preprocessor.apc_preprocessor import APCPreprocessor
from preprocessor.densitometer_preprocessor import DensitometerPreprocessor
from preprocessor.zone_analyzer import ZoneAnalyzer
from preprocessor.statistical_analyzer import StatisticalAnalyzer
from preprocessor.statistical_visualizer import StatisticalVisualizer
from preprocessor.preprocess_config import PreprocessConfig


def setup_logger():
    """로거 설정"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/statistical_analysis_test.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger('statistical_analysis_test')


def main():
    """메인 실행 함수"""

    # 로거 설정
    logger = setup_logger()
    logger.info("="*80)
    logger.info("통계 분석 기능 테스트 시작")
    logger.info("="*80)

    # Config 생성
    config = PreprocessConfig()

    # 출력 디렉토리 생성
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    os.makedirs(config.PLOT_DIR, exist_ok=True)
    os.makedirs(config.LOG_DIR, exist_ok=True)

    try:
        # ===== Step 1: APC 전처리 (비제어 구간 샘플링 포함) =====
        logger.info("\n[Step 1] APC 데이터 전처리")
        logger.info("-" * 80)

        apc_processor = APCPreprocessor(config, logger)

        # 실제 파일 경로로 변경 필요
        input_file = 'data/apc_data.xlsx'  # TODO: 실제 경로로 변경
        llspec_file = 'data/llspec.xlsx'    # TODO: 실제 경로로 변경 (옵션)

        if not os.path.exists(input_file):
            logger.error(f"입력 파일을 찾을 수 없습니다: {input_file}")
            logger.info("테스트를 위해 샘플 데이터 경로를 확인하세요.")
            return

        meaningful_df = apc_processor.run(
            input_file=input_file,
            llspec_file=llspec_file if os.path.exists(llspec_file) else None
        )

        if meaningful_df is None:
            logger.error("APC 전처리 실패")
            return

        logger.info(f"✓ APC 전처리 완료: {len(meaningful_df)}개 유의미한 구간")

        # ===== Step 2: 밀도계 데이터 추출 (제어 + 비제어) =====
        logger.info("\n[Step 2] 밀도계 데이터 추출")
        logger.info("-" * 80)

        densitometer_processor = DensitometerPreprocessor(config, logger)

        control_regions_file = os.path.join(config.OUTPUT_DIR, config.OUTPUT_4TH_CONTROL)
        no_control_regions_file = os.path.join(config.OUTPUT_DIR, config.OUTPUT_5TH_NO_CONTROL)
        raw_data_file = 'data/densitometer_raw.xlsx'  # TODO: 실제 경로로 변경

        if not os.path.exists(raw_data_file):
            logger.error(f"밀도계 raw data 파일을 찾을 수 없습니다: {raw_data_file}")
            return

        extracted_data = densitometer_processor.run(
            control_regions_file=control_regions_file,
            no_control_regions_file=no_control_regions_file,
            raw_data_file=raw_data_file
        )

        if extracted_data is None or extracted_data.empty:
            logger.error("밀도계 데이터 추출 실패")
            return

        logger.info(f"✓ 밀도계 데이터 추출 완료: {len(extracted_data)}행")

        # ===== Step 3: 통계 분석 실행 =====
        logger.info("\n[Step 3] 통계 분석 수행")
        logger.info("-" * 80)

        stat_analyzer = StatisticalAnalyzer(logger)
        zone_analyzer = ZoneAnalyzer(config, stat_analyzer, logger)

        zone_results, stat_results = zone_analyzer.run(
            densitometer_data=extracted_data,
            meaningful_changes=meaningful_df,
            visualize=True,
            perform_statistical_analysis=True
        )

        if stat_results is None or stat_results.empty:
            logger.warning("통계 분석 결과가 없습니다.")
        else:
            logger.info(f"✓ 통계 분석 완료: {len(stat_results)}개 결과")

            # 통계 요약
            controlled_stats = stat_results[stat_results['control_type'] == 'controlled']
            if len(controlled_stats) > 0:
                significant_count = controlled_stats['statistically_significant'].sum()
                effective_count = controlled_stats['control_effective'].sum()

                logger.info("\n[통계 분석 요약]")
                logger.info(f"  전체 분석 수: {len(controlled_stats)}")
                logger.info(f"  통계적 유의: {significant_count} ({significant_count/len(controlled_stats)*100:.1f}%)")
                logger.info(f"  제어 효과 있음: {effective_count} ({effective_count/len(controlled_stats)*100:.1f}%)")

        # ===== Step 4: 시각화 생성 =====
        logger.info("\n[Step 4] 시각화 생성")
        logger.info("-" * 80)

        if stat_results is not None and not stat_results.empty:
            visualizer = StatisticalVisualizer(config, logger)
            visualizer.create_statistical_summary_plots(stat_results)
            logger.info(f"✓ 시각화 완료: {config.PLOT_DIR}/statistical_analysis/")

        # ===== 완료 =====
        logger.info("\n" + "="*80)
        logger.info("통계 분석 테스트 완료!")
        logger.info("="*80)
        logger.info("\n생성된 파일:")
        logger.info(f"  - {config.OUTPUT_DIR}/4th_control_regions.xlsx")
        logger.info(f"  - {config.OUTPUT_DIR}/5th_no_control_regions.xlsx")
        logger.info(f"  - {config.OUTPUT_DIR}/extracted_densitometer_data.xlsx")
        logger.info(f"  - {config.OUTPUT_DIR}/zone_analysis_results.xlsx")
        logger.info(f"  - {config.OUTPUT_DIR}/statistical_analysis_results.xlsx")
        logger.info(f"  - {config.PLOT_DIR}/statistical_analysis/*.png")

    except Exception as e:
        logger.error(f"오류 발생: {e}", exc_info=True)
        raise


if __name__ == '__main__':
    main()
