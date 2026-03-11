"""
전처리 파이프라인 실행 스크립트
"""

from pipeline import CoatingPreprocessPipeline
from config import PreprocessConfig
import logging
import argparse


def main():
    """메인 실행 함수"""

    # 커맨드라인 인자 파싱
    parser = argparse.ArgumentParser(
        description='코팅 L/L 제어 전처리 파이프라인',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  # Training 데이터 생성 (기본 - 병합된 파일 사용)
  python main.py --mode training

  # Test 데이터 생성
  python main.py --mode test

  # APC/밀도계 파일 직접 지정
  python main.py --mode training --apc data/raw/apc_data.xlsx --densitometer data/raw/densitometer_data.csv

  # 다중 파일 병합만 수행
  python main.py --merge --apc-multiple data/raw/apc_*.xlsx --densitometer-multiple data/raw/densitometer_*.csv
        """
    )

    # 모드 선택
    parser.add_argument(
        '--mode',
        type=str,
        choices=['training', 'test'],
        default='training',
        help='데이터 생성 모드 (training: 학습 데이터, test: 테스트 데이터)'
    )

    # 단일 파일 모드 (기본값: 병합된 파일)
    parser.add_argument(
        '--apc',
        type=str,
        default='outputs/temp_merged_apc.parquet',
        help='APC 데이터 파일 경로 (기본: outputs/temp_merged_apc.parquet)'
    )
    parser.add_argument(
        '--densitometer',
        type=str,
        default='outputs/temp_merged_densitometer.parquet',
        help='밀도계 데이터 파일 경로 (기본: outputs/temp_merged_densitometer.parquet)'
    )
    parser.add_argument(
        '--llspec',
        type=str,
        default='outputs/temp_merged_llspec.parquet',
        help='LLspec 데이터 파일 경로 (기본: outputs/temp_merged_llspec.parquet)'
    )

    # 다중 파일 병합 모드
    parser.add_argument(
        '--merge',
        action='store_true',
        help='다중 파일 병합만 수행 (전처리 파이프라인은 실행하지 않음)'
    )
    parser.add_argument(
        '--apc-multiple',
        nargs='+',
        help='APC 데이터 파일 경로 리스트 (여러 파일)'
    )
    parser.add_argument(
        '--densitometer-multiple',
        nargs='+',
        help='밀도계 데이터 파일 경로 리스트 (여러 파일)'
    )
    parser.add_argument(
        '--llspec-multiple',
        nargs='+',
        default=None,
        help='LLspec 데이터 파일 경로 리스트 (선택)'
    )

    # 기타 옵션
    parser.add_argument(
        '--no-visualize',
        action='store_true',
        help='시각화 비활성화'
    )
    parser.add_argument(
        '--no-model-data',
        action='store_true',
        help='모델 데이터 생성 비활성화'
    )
    parser.add_argument(
        '--no-offline-rl-data',
        action='store_true',
        help='Offline RL MDP 데이터 생성 비활성화'
    )

    args = parser.parse_args()

    # 설정 초기화
    config = PreprocessConfig()

    # 파이프라인 초기화 (로거는 자동으로 생성됨)
    pipeline = CoatingPreprocessPipeline(config)

    pipeline.logger.info("코팅 L/L 전처리 파이프라인을 시작합니다.")
    pipeline.logger.info(f"모드: {args.mode.upper()}")

    # 실행 모드 결정
    visualize = not args.no_visualize
    prepare_model_data = not args.no_model_data
    prepare_offline_rl_data = not args.no_offline_rl_data

    # ===================================================================
    # 실행 모드 선택
    # ===================================================================

    if args.merge:
        # 병합 모드: 다중 파일 병합만 수행
        if args.apc_multiple and args.densitometer_multiple:
            pipeline.merge_multiple_files(
                apc_files=args.apc_multiple,
                densitometer_files=args.densitometer_multiple,
                llspec_files=args.llspec_multiple
            )
        else:
            pipeline.logger.error("--merge 사용 시 --apc-multiple/--densitometer-multiple을 지정해주세요.")
            parser.print_help()
            return

    else:
        # 전처리 실행 (기본: 병합된 parquet 파일 사용)
        pipeline.run_single_file(
            apc_file=args.apc,
            densitometer_file=args.densitometer,
            llspec_file=args.llspec,
            visualize=visualize,
            prepare_model_data=prepare_model_data,
            prepare_offline_rl_data=prepare_offline_rl_data,
            mode=args.mode
        )

    # 결과 확인
    results = pipeline.get_results()

    pipeline.logger.info("전처리 완료!")
    pipeline.logger.info(f"결과 항목: {list(results.keys())}")


if __name__ == "__main__":
    main()
