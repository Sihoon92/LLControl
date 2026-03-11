"""
픽셀 분포 시각화 스크립트

1) 시계열 시각화: 제어 구간(4th_control_regions) 기반으로 밀도계 raw data에서
   USL/LSL 사이 3구간(High/Mid/Low) 픽셀 분포의 시간별 변화를 시각화.
2) Zone 히트맵: zone_analysis_results.xlsx 기반으로 존별 제어 전/후
   Division 분포 변화를 히트맵으로 시각화.

사용법:
  python visualize_pixel_distribution.py \
    --densitometer data/raw/densitometer_data.csv \
    --n_samples 5 \
    --seed 42

  python visualize_pixel_distribution.py \
    --densitometer data/raw/densitometer_data.csv \
    --zone_results outputs/zone_analysis_results.xlsx \
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
import seaborn as sns

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

    분모: USL~LSL 사이 데이터 수 (out of range 제외)
    분자: 각 구간(High/Mid/Low)의 데이터 수

    Returns:
        dict with keys 'low', 'mid', 'high'
        각 값은 비율 (0~1), 합계 = 1.0
    """
    values = row_values[row_values > 0]  # 0 이하(무효) 제외

    # USL~LSL 사이 데이터만 대상 (out of range 제외)
    in_range = values[(values >= lsl) & (values <= usl)]
    total = len(in_range)
    if total == 0:
        return {'low': 0.0, 'mid': 0.0, 'high': 0.0}

    range_size = (usl - lsl) / 3.0
    boundary_low_mid = lsl + range_size
    boundary_mid_high = lsl + 2 * range_size

    low_count = ((in_range >= lsl) & (in_range < boundary_low_mid)).sum()
    mid_count = ((in_range >= boundary_low_mid) & (in_range < boundary_mid_high)).sum()
    high_count = ((in_range >= boundary_mid_high) & (in_range <= usl)).sum()

    return {
        'low': low_count / total,
        'mid': mid_count / total,
        'high': high_count / total,
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

    # 1분 단위로 floor하여 그룹핑
    group_data['minute'] = group_data['datetime'].dt.floor('1min')

    results = []
    for minute_ts, minute_group in group_data.groupby('minute'):
        # 해당 1분 윈도우의 모든 픽셀 값을 합침
        all_values = minute_group[value_columns].values.astype(float).flatten()
        dist = compute_pixel_distribution(all_values, lsl, usl)
        dist['datetime'] = minute_ts
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


def extract_heatmap_matrices(group_df, n_divisions):
    """
    특정 group_id의 zone_analysis_results에서 before/after/change 행렬을 추출.

    Returns:
        tuple: (before_matrix, after_matrix, change_matrix, zone_ids, div_labels)
    """
    group_df = group_df.sort_values('zone_id')
    zone_ids = group_df['zone_id'].values

    before_matrix = np.zeros((len(zone_ids), n_divisions))
    after_matrix = np.zeros((len(zone_ids), n_divisions))
    change_matrix = np.zeros((len(zone_ids), n_divisions))
    div_labels = []

    for j in range(n_divisions):
        div_idx = j + 1
        before_matrix[:, j] = group_df[f'div_{div_idx}_before_ratio'].values
        after_matrix[:, j] = group_df[f'div_{div_idx}_after_ratio'].values
        change_matrix[:, j] = group_df[f'div_{div_idx}_ratio_change'].values

        range_str = group_df[f'div_{div_idx}_range'].iloc[0]
        div_labels.append(f'Div{div_idx}\n{range_str}')

    return before_matrix, after_matrix, change_matrix, zone_ids, div_labels


def plot_zone_heatmaps(before_matrix, after_matrix, change_matrix,
                       zone_ids, div_labels, group_id, ucl, lcl, output_dir):
    """하나의 group_id에 대해 Before/After/Change 3개 히트맵을 생성하여 저장."""
    fig, axes = plt.subplots(1, 3, figsize=(22, 8))

    vmax_ba = max(before_matrix.max(), after_matrix.max())

    abs_max_change = max(abs(change_matrix.min()), abs(change_matrix.max()))
    if abs_max_change == 0:
        abs_max_change = 0.01

    sns.heatmap(
        before_matrix, annot=True, fmt='.2f', cmap='YlOrRd',
        vmin=0, vmax=vmax_ba, xticklabels=div_labels, yticklabels=zone_ids,
        linewidths=0.5, cbar_kws={'label': 'Ratio'}, ax=axes[0]
    )
    axes[0].set_title('Before (제어 전)', fontsize=13, fontweight='bold')
    axes[0].set_ylabel('Zone ID', fontsize=11)
    axes[0].set_xlabel('Division', fontsize=11)

    sns.heatmap(
        after_matrix, annot=True, fmt='.2f', cmap='YlOrRd',
        vmin=0, vmax=vmax_ba, xticklabels=div_labels, yticklabels=zone_ids,
        linewidths=0.5, cbar_kws={'label': 'Ratio'}, ax=axes[1]
    )
    axes[1].set_title('After (제어 후)', fontsize=13, fontweight='bold')
    axes[1].set_ylabel('Zone ID', fontsize=11)
    axes[1].set_xlabel('Division', fontsize=11)

    sns.heatmap(
        change_matrix, annot=True, fmt='.2f', cmap='RdBu_r',
        vmin=-abs_max_change, vmax=abs_max_change, center=0,
        xticklabels=div_labels, yticklabels=zone_ids,
        linewidths=0.5, cbar_kws={'label': 'Ratio Change'}, ax=axes[2]
    )
    axes[2].set_title('Change (After - Before)', fontsize=13, fontweight='bold')
    axes[2].set_ylabel('Zone ID', fontsize=11)
    axes[2].set_xlabel('Division', fontsize=11)

    fig.suptitle(
        f'Group {group_id}  |  UCL={ucl:.4f}  LCL={lcl:.4f}\n'
        f'Zone별 Division 분포 변화 히트맵',
        fontsize=15, fontweight='bold', y=1.02
    )
    plt.tight_layout()

    output_path = os.path.join(output_dir, f'group_{group_id}_zone_heatmap.png')
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description='픽셀 분포 시각화 (시계열 + Zone 히트맵)'
    )
    parser.add_argument('--densitometer', type=str, required=True,
                        help='밀도계 raw data 파일 경로')
    parser.add_argument('--control_regions', type=str, default=None,
                        help='4th_control_regions 파일 경로 (기본: outputs/4th_control_regions.xlsx)')
    parser.add_argument('--meaningful_changes', type=str, default=None,
                        help='3rd_meaningful_changes 파일 경로 (기본: outputs/3rd_meaningful_changes.xlsx)')
    parser.add_argument('--zone_results', type=str, default=None,
                        help='zone_analysis_results.xlsx 경로 (기본: outputs/zone_analysis_results.xlsx)')
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
    zone_results_file = args.zone_results or os.path.join(
        config.OUTPUT_DIR, config.OUTPUT_ZONE_ANALYSIS
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

    logger.info(f"[6] 시계열 시각화 완료: {output_path}")

    # ================================================================
    # 4. Zone 히트맵 시각화
    # ================================================================
    logger.info("=" * 60)
    logger.info("Zone 히트맵 시각화 시작")
    logger.info("=" * 60)

    if not os.path.exists(zone_results_file):
        logger.warning(f"Zone 분석 결과 파일 없음: {zone_results_file} (히트맵 생략)")
    else:
        zone_df = utils.load_file(zone_results_file, logger=logger)
        logger.info(f"[7] Zone 분석 결과 로드: {len(zone_df)} 행")

        n_divisions = int(zone_df['n_divisions'].iloc[0])
        available_zone_groups = set(zone_df['group_id'].unique())

        # 시계열에서 사용한 sampled_ids 중 zone_analysis에도 존재하는 것 사용
        heatmap_ids = sorted(set(sampled_ids) & available_zone_groups)

        if not heatmap_ids:
            logger.warning("시계열 샘플 group_id와 zone_analysis의 group_id가 겹치지 않습니다.")
        else:
            logger.info(f"    히트맵 대상 group_id ({len(heatmap_ids)}개): {heatmap_ids}")

            heatmap_dir = os.path.join(output_dir, 'zone_heatmap')
            os.makedirs(heatmap_dir, exist_ok=True)

            for i, group_id in enumerate(heatmap_ids):
                logger.info(f"[8-{i+1}] Group {group_id} 히트맵 생성 중...")

                group_zone_df = zone_df[zone_df['group_id'] == group_id]
                ucl = float(group_zone_df['ucl'].iloc[0])
                lcl = float(group_zone_df['lcl'].iloc[0])

                before_mat, after_mat, change_mat, zone_ids, div_labels = \
                    extract_heatmap_matrices(group_zone_df, n_divisions)

                heatmap_path = plot_zone_heatmaps(
                    before_mat, after_mat, change_mat,
                    zone_ids, div_labels,
                    group_id, ucl, lcl, heatmap_dir
                )
                logger.info(f"    -> 저장: {heatmap_path}")

                max_change_idx = np.unravel_index(
                    np.argmax(np.abs(change_mat)), change_mat.shape
                )
                max_zone = zone_ids[max_change_idx[0]]
                max_div = max_change_idx[1] + 1
                max_val = change_mat[max_change_idx]
                logger.info(f"    -> 최대 변화: Zone {max_zone}, Div {max_div} (change={max_val:+.4f})")

    logger.info("=" * 60)
    logger.info(f"전체 시각화 완료! 저장 위치: {output_dir}")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
