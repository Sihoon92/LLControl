"""
Zone별 Before/After 분포 변화 히트맵 시각화 스크립트

zone_analysis_results.xlsx 데이터를 기반으로, 랜덤 선택된 group_id에 대해
각 zone_id별 6개 구간(div_1~div_6)의 before/after ratio를 히트맵으로 시각화.

사용법:
  python visualize_zone_heatmap.py

  python visualize_zone_heatmap.py \
    --zone_results outputs/zone_analysis_results.xlsx \
    --n_samples 3 \
    --seed 42
"""

import argparse
import logging
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import utils
from preprocessor.preprocess_config import PreprocessConfig

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger('zone_heatmap')


def extract_matrices(group_df, n_divisions):
    """
    특정 group_id의 DataFrame에서 before/after/change 행렬을 추출.

    Parameters:
    -----------
    group_df : pd.DataFrame
        단일 group_id에 해당하는 zone_analysis_results 행들
    n_divisions : int
        구간 수 (기본 6)

    Returns:
    --------
    tuple: (before_matrix, after_matrix, change_matrix, zone_ids, div_labels)
        각 matrix shape: (n_zones, n_divisions)
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

        # 구간 범위 라벨 (첫 번째 행 기준)
        range_str = group_df[f'div_{div_idx}_range'].iloc[0]
        div_labels.append(f'Div{div_idx}\n{range_str}')

    return before_matrix, after_matrix, change_matrix, zone_ids, div_labels


def plot_group_heatmaps(before_matrix, after_matrix, change_matrix,
                        zone_ids, div_labels, group_id, ucl, lcl, output_dir):
    """
    하나의 group_id에 대해 Before/After/Change 3개 히트맵을 생성하여 저장.
    """
    fig, axes = plt.subplots(1, 3, figsize=(22, 8))

    # Before/After 공통 스케일
    vmax_ba = max(before_matrix.max(), after_matrix.max())

    # Change 대칭 스케일
    abs_max_change = max(abs(change_matrix.min()), abs(change_matrix.max()))
    if abs_max_change == 0:
        abs_max_change = 0.01

    # --- Before 히트맵 ---
    sns.heatmap(
        before_matrix,
        annot=True, fmt='.2f',
        cmap='YlOrRd',
        vmin=0, vmax=vmax_ba,
        xticklabels=div_labels,
        yticklabels=zone_ids,
        linewidths=0.5,
        cbar_kws={'label': 'Ratio'},
        ax=axes[0]
    )
    axes[0].set_title('Before (제어 전)', fontsize=13, fontweight='bold')
    axes[0].set_ylabel('Zone ID', fontsize=11)
    axes[0].set_xlabel('Division', fontsize=11)

    # --- After 히트맵 ---
    sns.heatmap(
        after_matrix,
        annot=True, fmt='.2f',
        cmap='YlOrRd',
        vmin=0, vmax=vmax_ba,
        xticklabels=div_labels,
        yticklabels=zone_ids,
        linewidths=0.5,
        cbar_kws={'label': 'Ratio'},
        ax=axes[1]
    )
    axes[1].set_title('After (제어 후)', fontsize=13, fontweight='bold')
    axes[1].set_ylabel('Zone ID', fontsize=11)
    axes[1].set_xlabel('Division', fontsize=11)

    # --- Change 히트맵 ---
    sns.heatmap(
        change_matrix,
        annot=True, fmt='.2f',
        cmap='RdBu_r',
        vmin=-abs_max_change, vmax=abs_max_change,
        center=0,
        xticklabels=div_labels,
        yticklabels=zone_ids,
        linewidths=0.5,
        cbar_kws={'label': 'Ratio Change'},
        ax=axes[2]
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
        description='Zone별 Before/After 분포 변화 히트맵 시각화'
    )
    parser.add_argument('--zone_results', type=str, default=None,
                        help='zone_analysis_results.xlsx 경로 (기본: outputs/zone_analysis_results.xlsx)')
    parser.add_argument('--n_samples', type=int, default=3,
                        help='랜덤 샘플링할 group_id 수 (기본: 3)')
    parser.add_argument('--seed', type=int, default=None,
                        help='랜덤 시드 (재현성)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='출력 디렉토리 (기본: outputs/plots/zone_heatmap)')
    args = parser.parse_args()

    config = PreprocessConfig()

    zone_results_file = args.zone_results or os.path.join(
        config.OUTPUT_DIR, config.OUTPUT_ZONE_ANALYSIS
    )
    output_dir = args.output_dir or os.path.join(config.PLOT_DIR, 'zone_heatmap')
    os.makedirs(output_dir, exist_ok=True)

    # ================================================================
    # 1. 데이터 로딩
    # ================================================================
    logger.info("=" * 60)
    logger.info("Zone 히트맵 시각화 시작")
    logger.info("=" * 60)

    logger.info(f"[1] Zone 분석 결과 로드: {zone_results_file}")
    zone_df = utils.load_file(zone_results_file, logger=logger)
    logger.info(f"    -> {len(zone_df)} 행 로드")

    # n_divisions 확인
    n_divisions = int(zone_df['n_divisions'].iloc[0])
    logger.info(f"    -> n_divisions: {n_divisions}")

    # ================================================================
    # 2. group_id 샘플링
    # ================================================================
    valid_group_ids = sorted(zone_df['group_id'].unique())
    logger.info(f"[2] 유효한 group_id 수: {len(valid_group_ids)}")

    n_samples = min(args.n_samples, len(valid_group_ids))
    if args.seed is not None:
        random.seed(args.seed)
    sampled_ids = sorted(random.sample(valid_group_ids, n_samples))
    logger.info(f"    샘플링된 group_id ({n_samples}개): {sampled_ids}")

    # ================================================================
    # 3. 시각화
    # ================================================================
    for i, group_id in enumerate(sampled_ids):
        logger.info(f"[3-{i+1}] Group {group_id} 히트맵 생성 중...")

        group_df = zone_df[zone_df['group_id'] == group_id]
        ucl = float(group_df['ucl'].iloc[0])
        lcl = float(group_df['lcl'].iloc[0])

        before_mat, after_mat, change_mat, zone_ids, div_labels = \
            extract_matrices(group_df, n_divisions)

        output_path = plot_group_heatmaps(
            before_mat, after_mat, change_mat,
            zone_ids, div_labels,
            group_id, ucl, lcl, output_dir
        )
        logger.info(f"    -> 저장: {output_path}")

        # 주요 변화 요약
        max_change_idx = np.unravel_index(np.argmax(np.abs(change_mat)), change_mat.shape)
        max_zone = zone_ids[max_change_idx[0]]
        max_div = max_change_idx[1] + 1
        max_val = change_mat[max_change_idx]
        logger.info(f"    -> 최대 변화: Zone {max_zone}, Div {max_div} (change={max_val:+.4f})")

    logger.info("=" * 60)
    logger.info(f"시각화 완료! 저장 위치: {output_dir}")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
