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
  # Training 데이터 생성 (기본)
  python main.py --mode training --apc data/raw/apc_data.xlsx --densitometer data/raw/densitometer_data.csv

  # Test 데이터 생성
  python main.py --mode test --apc data/raw/apc_test.xlsx --densitometer data/raw/densitometer_test.csv

  # 다중 파일 모드 (Training)
  python main.py --mode training --apc-multiple data/raw/apc_*.xlsx --densitometer-multiple data/raw/densitometer_*.csv

  # 폴더 모드 (Test)
  python main.py --mode test --folder data/raw/
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

    # 단일 파일 모드
    parser.add_argument(
        '--apc',
        type=str,
        help='APC 데이터 파일 경로'
    )
    parser.add_argument(
        '--densitometer',
        type=str,
        help='밀도계 데이터 파일 경로'
    )
    parser.add_argument(
        '--llspec',
        type=str,
        default=None,
        help='LLspec 데이터 파일 경로 (선택)'
    )

    # 다중 파일 모드
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

    # 폴더 모드
    parser.add_argument(
        '--folder',
        type=str,
        help='데이터 폴더 경로 (자동 검색)'
    )
    parser.add_argument(
        '--apc-pattern',
        type=str,
        default='apc*.xlsx',
        help='APC 파일 패턴 (폴더 모드 사용 시)'
    )
    parser.add_argument(
        '--densitometer-pattern',
        type=str,
        default='densitometer*.csv',
        help='밀도계 파일 패턴 (폴더 모드 사용 시)'
    )
    parser.add_argument(
        '--llspec-pattern',
        type=str,
        default='llspec*.xlsx',
        help='LLspec 파일 패턴 (폴더 모드 사용 시)'
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

    # ===================================================================
    # 실행 모드 선택
    # ===================================================================

    if args.folder:
        # 옵션 3: 폴더 자동 검색 모드
        pipeline.run_from_folder(
            folder_path=args.folder,
            apc_pattern=args.apc_pattern,
            densitometer_pattern=args.densitometer_pattern,
            llspec_pattern=args.llspec_pattern,
            visualize=visualize,
            prepare_model_data=prepare_model_data,
            mode=args.mode
        )

    elif args.apc_multiple and args.densitometer_multiple:
        # 옵션 2: 다중 파일 모드
        pipeline.run_multiple_files(
            apc_files=args.apc_multiple,
            densitometer_files=args.densitometer_multiple,
            llspec_files=args.llspec_multiple,
            visualize=visualize,
            prepare_model_data=prepare_model_data,
            mode=args.mode
        )

    elif args.apc and args.densitometer:
        # 옵션 1: 단일 파일 모드
        pipeline.run_single_file(
            apc_file=args.apc,
            densitometer_file=args.densitometer,
            llspec_file=args.llspec,
            visualize=visualize,
            prepare_model_data=prepare_model_data,
            mode=args.mode
        )

    else:
        pipeline.logger.error("입력 파일을 지정해주세요. --help로 사용법을 확인하세요.")
        parser.print_help()
        return

    # 결과 확인
    results = pipeline.get_results()

    pipeline.logger.info("전처리 완료!")
    pipeline.logger.info(f"결과 항목: {list(results.keys())}")


if __name__ == "__main__":
    main()
