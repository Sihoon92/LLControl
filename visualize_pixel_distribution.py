"""
픽셀 분포 시계열 시각화 스크립트

제어 구간(4th_control_regions) 기반으로 밀도계 raw data에서
USL/LSL 사이 3구간(High/Mid/Low) 픽셀 분포의 시간별 변화를 시각화.

사용법:
  python visualize_pixel_distribution.py \
    --densitometer data/raw/densitometer_data.csv \
    --n_samples 5 \
    --seed 42

  python visualize_pixel_distribution.py \
    --densitometer data/raw/densitometer_data.csv \
    --control_regions outputs/4th_control_regions.xlsx \
    --meaningful_changes outputs/3rd_meaningful_changes.xlsx \
    --n_samples 3
"""

import argparse
import logging
import os
import random
import sys

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

# 프로젝트 내 유틸리티 재사용
import utils
from preprocessor.preprocess_config import PreprocessConfig

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger('pixel_distribution')


def parse_time(time_val, reference_date='2024-01-01'):
    """
    다양한 형태의 시간 값을 timezone-naive pandas Timestamp로 통일 변환.

    지원 형태:
    - str "HH:MM:SS" → reference_date + HH:MM:SS
    - str "YYYY-MM-DD HH:MM:SS" → 그대로 파싱
    - datetime / pd.Timestamp → 그대로 반환 (tz 제거)
    """
    if pd.isna(time_val):
        return pd.NaT

    # 이미 Timestamp/datetime 계열이면 tz만 제거하고 반환
    if isinstance(time_val, (pd.Timestamp,)):
        return time_val.tz_localize(None) if time_val.tzinfo else time_val
    if hasattr(time_val, 'year'):  # datetime.datetime 등
        return pd.Timestamp(time_val).tz_localize(None)

    # 문자열 처리
    time_str = str(time_val).strip()
    # "HH:MM:SS" 형식 (날짜 없음) → reference_date 붙이기
    if len(time_str) <= 8 and ':' in time_str and '-' not in time_str:
        return pd.Timestamp(f"{reference_date} {time_str}")
    # 그 외 ("YYYY-MM-DD HH:MM:SS" 등) → 직접 파싱
    return pd.Timestamp(time_str)


def identify_value_columns(df):
    """밀도계 DataFrame에서 Value 칼럼 목록을 식별"""
    time_col = df.columns[0]
    value_columns = []
    for col in df.columns[1:]:
        col_str = str(col)
        if 'value' in col_str.lower() or col_str.isdigit():
            value_columns.append(col)
    if not value_columns:
        value_columns = [col for col in df.columns if col != time_col]
    return time_col, value_columns


def compute_pixel_distribution(row_values, lsl, usl):
    """
    단일 시간대의 픽셀 값들을 USL/LSL 기준으로 3구간 분류.

    Returns:
        dict with keys 'low', 'mid', 'high', 'out_of_range'
        각 값은 비율 (0~1)
    """
    values = row_values[row_values > 0]  # 0 이하(무효) 제외
    total = len(values)
    if total == 0:
        return {'low': 0.0, 'mid': 0.0, 'high': 0.0, 'out_of_range': 0.0}

    range_size = (usl - lsl) / 3.0
    boundary_low_mid = lsl + range_size
    boundary_mid_high = lsl + 2 * range_size

    low_count = ((values >= lsl) & (values < boundary_low_mid)).sum()
    mid_count = ((values >= boundary_low_mid) & (values < boundary_mid_high)).sum()
    high_count = ((values >= boundary_mid_high) & (values <= usl)).sum()
    out_count = ((values < lsl) | (values > usl)).sum()

    return {
        'low': low_count / total,
        'mid': mid_count / total,
        'high': high_count / total,
        'out_of_range': out_count / total,
    }


def process_group(raw_df, time_col, value_columns, group_info, usl, lsl):
    """
    하나의 group_id에 대해 시간별 픽셀 분포를 계산.

    Returns:
        pd.DataFrame: columns = ['datetime', 'low', 'mid', 'high', 'out_of_range']
    """
    start_time = parse_time(group_info['start_time'])
    end_time = parse_time(group_info['end_time'])

    mask = (raw_df['datetime'] >= start_time) & (raw_df['datetime'] <= end_time)
    group_data = raw_df[mask].copy()

    if group_data.empty:
        logger.warning(f"  Group {group_info['group_id']}: 데이터 없음 ({start_time} ~ {end_time})")
        return None

    # 1분 단위로 리샘플링 (이미 1분 간격이면 그대로)
    results = []
    for _, row in group_data.iterrows():
        dist = compute_pixel_distribution(
            row[value_columns].values.astype(float), lsl, usl
        )
        dist['datetime'] = row['datetime']
        results.append(dist)

    return pd.DataFrame(results)


def plot_group(ax, dist_df, group_id, usl, lsl, control_start=None, control_end=None):
    """하나의 group_id에 대한 시계열 분포 그래프를 그린다."""

    ax.plot(dist_df['datetime'], dist_df['high'] * 100,
            color='#e74c3c', linewidth=1.5, label='High', marker='o', markersize=2)
    ax.plot(dist_df['datetime'], dist_df['mid'] * 100,
            color='#2ecc71', linewidth=1.5, label='Mid', marker='s', markersize=2)
    ax.plot(dist_df['datetime'], dist_df['low'] * 100,
            color='#3498db', linewidth=1.5, label='Low', marker='^', markersize=2)
    ax.plot(dist_df['datetime'], dist_df['out_of_range'] * 100,
            color='#95a5a6', linewidth=1.0, linestyle='--', label='Out of range',
            marker='x', markersize=2, alpha=0.7)

    # 제어 구간 표시
    if control_start is not None and control_end is not None:
        cs = parse_time(control_start)
        ce = parse_time(control_end)
        ax.axvspan(cs, ce, alpha=0.15, color='#f39c12', label='Control period')
        ax.axvline(cs, color='#f39c12', linestyle='--', linewidth=1, alpha=0.6)
        ax.axvline(ce, color='#f39c12', linestyle='--', linewidth=1, alpha=0.6)

    ax.set_title(
        f'Group {group_id}  |  USL={usl:.4f}  LSL={lsl:.4f}',
        fontsize=11, fontweight='bold'
    )
    ax.set_ylabel('Ratio (%)', fontsize=10)
    ax.set_ylim(-2, 102)
    ax.legend(loc='upper right', fontsize=8, ncol=5)
    ax.grid(axis='y', alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.tick_params(axis='x', rotation=30)


def main():
    parser = argparse.ArgumentParser(
        description='픽셀 분포 시계열 시각화 (USL/LSL 3구간: High/Mid/Low)'
    )
    parser.add_argument('--densitometer', type=str, required=True,
                        help='밀도계 raw data 파일 경로')
    parser.add_argument('--control_regions', type=str, default=None,
                        help='4th_control_regions 파일 경로 (기본: outputs/4th_control_regions.xlsx)')
    parser.add_argument('--meaningful_changes', type=str, default=None,
                        help='3rd_meaningful_changes 파일 경로 (기본: outputs/3rd_meaningful_changes.xlsx)')
    parser.add_argument('--n_samples', type=int, default=5,
                        help='랜덤 샘플링할 group_id 수 (기본: 5)')
    parser.add_argument('--seed', type=int, default=None,
                        help='랜덤 시드 (재현성)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='출력 디렉토리 (기본: outputs/plots/pixel_distribution)')
    args = parser.parse_args()

    config = PreprocessConfig()

    # 파일 경로 기본값 설정
    control_regions_file = args.control_regions or os.path.join(
        config.OUTPUT_DIR, config.OUTPUT_4TH_CONTROL
    )
    meaningful_changes_file = args.meaningful_changes or os.path.join(
        config.OUTPUT_DIR, config.OUTPUT_3RD
    )
    output_dir = args.output_dir or os.path.join(config.PLOT_DIR, 'pixel_distribution')
    os.makedirs(output_dir, exist_ok=True)

    # ================================================================
    # 1. 데이터 로딩
    # ================================================================
    logger.info("=" * 60)
    logger.info("픽셀 분포 시계열 시각화 시작")
    logger.info("=" * 60)

    # 1-1. 제어 구간 정보
    logger.info(f"[1] 제어 구간 파일 로드: {control_regions_file}")
    regions_df = utils.load_file(control_regions_file, logger=logger)
    logger.info(f"    -> {len(regions_df)} 개 구간, group_id: {sorted(regions_df['group_id'].unique())}")

    # 1-2. meaningful_changes (USL/LSL 소스)
    logger.info(f"[2] Meaningful changes 로드: {meaningful_changes_file}")
    mc_df = utils.load_file(meaningful_changes_file, logger=logger)
    if 'UCL' not in mc_df.columns or 'LCL' not in mc_df.columns:
        logger.error("    UCL/LCL 칼럼이 없습니다. LLspec 매칭이 필요합니다.")
        sys.exit(1)
    logger.info(f"    -> {len(mc_df)} 개 항목")

    # 1-3. 밀도계 raw data
    logger.info(f"[3] 밀도계 raw data 로드: {args.densitometer}")
    raw_df = utils.load_file(args.densitometer, logger=logger)
    time_col, value_columns = identify_value_columns(raw_df)
    logger.info(f"    -> {len(raw_df)} 행, {len(value_columns)} Value 칼럼")

    # 시간 파싱 (모든 값을 timezone-naive Timestamp로 통일)
    raw_df['datetime'] = pd.to_datetime(raw_df[time_col].apply(parse_time))
    logger.info(f"    시간 범위: {raw_df['datetime'].min()} ~ {raw_df['datetime'].max()}")

    # ================================================================
    # 2. group_id 매칭 및 랜덤 샘플링
    # ================================================================
    # UCL/LCL이 있는 group_id만 필터
    valid_mc = mc_df[mc_df['UCL'].notna() & mc_df['LCL'].notna()]
    valid_group_ids = set(valid_mc['group_id'].unique()) & set(regions_df['group_id'].unique())
    valid_group_ids = sorted(valid_group_ids)

    if not valid_group_ids:
        logger.error("USL/LSL 정보가 있는 유효한 group_id가 없습니다.")
        sys.exit(1)

    logger.info(f"[4] 유효한 group_id 수: {len(valid_group_ids)}")

    # 랜덤 샘플링
    n_samples = min(args.n_samples, len(valid_group_ids))
    if args.seed is not None:
        random.seed(args.seed)
    sampled_ids = sorted(random.sample(valid_group_ids, n_samples))
    logger.info(f"    샘플링된 group_id ({n_samples}개): {sampled_ids}")

    # ================================================================
    # 3. 시각화
    # ================================================================
    fig, axes = plt.subplots(n_samples, 1, figsize=(16, 5 * n_samples), squeeze=False)

    for i, group_id in enumerate(sampled_ids):
        ax = axes[i, 0]
        logger.info(f"[5-{i+1}] Group {group_id} 처리 중...")

        # USL/LSL 가져오기
        group_mc = valid_mc[valid_mc['group_id'] == group_id].iloc[0]
        usl = float(group_mc['UCL'])
        lsl = float(group_mc['LCL'])

        # 제어 구간 정보
        group_region = regions_df[regions_df['group_id'] == group_id].iloc[0]
        control_start = group_region.get('control_start', None)
        control_end = group_region.get('control_end', None)
        if pd.isna(control_start):
            control_start = None
        if pd.isna(control_end):
            control_end = None

        # 시간별 분포 계산
        dist_df = process_group(raw_df, time_col, value_columns, group_region, usl, lsl)

        if dist_df is None or dist_df.empty:
            ax.text(0.5, 0.5, f'Group {group_id}: No data',
                    ha='center', va='center', transform=ax.transAxes, fontsize=14)
            continue

        logger.info(f"    -> {len(dist_df)} 시간대, USL={usl:.4f}, LSL={lsl:.4f}")

        # 그래프 그리기
        plot_group(ax, dist_df, group_id, usl, lsl, control_start, control_end)

    axes[-1, 0].set_xlabel('Time', fontsize=11)

    fig.suptitle(
        'Pixel Distribution Over Time (USL/LSL 3-Region: High / Mid / Low)',
        fontsize=14, fontweight='bold', y=1.01
    )
    plt.tight_layout()

    output_path = os.path.join(output_dir, 'pixel_distribution_timeseries.png')
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)

    logger.info("=" * 60)
    logger.info(f"시각화 완료! 저장 위치: {output_path}")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
