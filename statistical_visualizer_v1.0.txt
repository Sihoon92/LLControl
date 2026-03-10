"""
통계 분석 시각화 모듈 v1.0
통계 분석 결과를 다양한 방식으로 시각화
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional
import os
import logging

# 한글 폰트 설정 (Windows)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False


class StatisticalVisualizer:
    """통계 분석 결과 시각화 클래스"""

    def __init__(self, config, logger: logging.Logger = None):
        """
        Parameters:
        -----------
        config : PreprocessConfig
            전처리 설정 객체
        logger : logging.Logger
            로거 객체
        """
        self.config = config
        self.logger = logger or logging.getLogger('coating_preprocessor.stat_viz')

        # 시각화 디렉토리 생성
        self.viz_dir = os.path.join(self.config.PLOT_DIR, 'statistical_analysis')
        os.makedirs(self.viz_dir, exist_ok=True)

    def create_statistical_summary_plots(
        self,
        stats_df: pd.DataFrame
    ):
        """
        통계 분석 결과 요약 시각화

        Parameters:
        -----------
        stats_df : pd.DataFrame
            통계 분석 결과 DataFrame
        """
        self.logger.info("="*80)
        self.logger.info("통계 분석 시각화 시작")
        self.logger.info("="*80)

        # 1. p-value 히트맵 (Zone별)
        self.plot_pvalue_heatmap(stats_df)

        # 2. Effect Size 비교 차트
        self.plot_effect_size_comparison(stats_df)

        # 3. Cpk 개선도 차트
        self.plot_cpk_improvement(stats_df)

        # 4. 제어 효과 요약 차트
        self.plot_control_effectiveness_summary(stats_df)

        # 5. 제어 vs 비제어 비교
        self.plot_control_vs_no_control(stats_df)

        self.logger.info(f"시각화 파일 저장 위치: {self.viz_dir}")
        self.logger.info("="*80)

    def plot_pvalue_heatmap(self, stats_df: pd.DataFrame):
        """
        p-value 히트맵 (Zone별, Group별)

        Parameters:
        -----------
        stats_df : pd.DataFrame
            통계 분석 결과
        """
        try:
            # 제어 구간만 필터링
            controlled_stats = stats_df[stats_df['control_type'] == 'controlled'].copy()

            if len(controlled_stats) == 0:
                self.logger.warning("제어 구간 데이터가 없어 p-value 히트맵을 생성할 수 없습니다.")
                return

            # Pivot 테이블 생성 (Group x Zone)
            pivot_data = controlled_stats.pivot_table(
                index='group_id',
                columns='zone_id',
                values='mannwhitney_pvalue',
                aggfunc='first'
            )

            # 시각화
            fig, ax = plt.subplots(figsize=(14, 8))

            # 히트맵 생성
            sns.heatmap(
                pivot_data,
                annot=True,
                fmt='.4f',
                cmap='RdYlGn_r',  # 낮을수록 녹색 (유의미)
                vmin=0,
                vmax=0.1,
                cbar_kws={'label': 'p-value (Mann-Whitney U test)'},
                linewidths=0.5,
                ax=ax
            )

            ax.set_title('통계적 유의성 히트맵 (p-value)\np < 0.05: 통계적으로 유의미한 변화',
                        fontsize=14, fontweight='bold')
            ax.set_xlabel('Zone ID', fontsize=12)
            ax.set_ylabel('Group ID', fontsize=12)

            # 유의수준 표시선
            ax.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5)
            ax.text(pivot_data.shape[1] + 0.5, 0, 'α = 0.05', color='red', fontsize=10)

            plt.tight_layout()
            output_path = os.path.join(self.viz_dir, 'pvalue_heatmap.png')
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()

            self.logger.info(f"✓ p-value 히트맵 저장: {output_path}")

        except Exception as e:
            self.logger.error(f"✗ p-value 히트맵 생성 오류: {e}", exc_info=True)

    def plot_effect_size_comparison(self, stats_df: pd.DataFrame):
        """
        Effect Size (Cohen's d) 비교 차트

        Parameters:
        -----------
        stats_df : pd.DataFrame
            통계 분석 결과
        """
        try:
            # 제어 구간만 필터링
            controlled_stats = stats_df[stats_df['control_type'] == 'controlled'].copy()

            if len(controlled_stats) == 0:
                self.logger.warning("제어 구간 데이터가 없어 Effect Size 차트를 생성할 수 없습니다.")
                return

            # Zone별 평균 Cohen's d
            zone_effect_sizes = controlled_stats.groupby('zone_id')['cohens_d'].mean().reset_index()

            # 시각화
            fig, ax = plt.subplots(figsize=(12, 6))

            # Bar plot
            colors = []
            for d in zone_effect_sizes['cohens_d']:
                abs_d = abs(d)
                if abs_d < 0.2:
                    colors.append('lightgray')  # negligible
                elif abs_d < 0.5:
                    colors.append('skyblue')  # small
                elif abs_d < 0.8:
                    colors.append('orange')  # medium
                else:
                    colors.append('red')  # large

            ax.bar(zone_effect_sizes['zone_id'], zone_effect_sizes['cohens_d'], color=colors, edgecolor='black')

            # 기준선 표시
            ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
            ax.axhline(y=0.2, color='blue', linestyle='--', linewidth=1, alpha=0.5, label='Small (0.2)')
            ax.axhline(y=0.5, color='orange', linestyle='--', linewidth=1, alpha=0.5, label='Medium (0.5)')
            ax.axhline(y=0.8, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Large (0.8)')
            ax.axhline(y=-0.2, color='blue', linestyle='--', linewidth=1, alpha=0.5)
            ax.axhline(y=-0.5, color='orange', linestyle='--', linewidth=1, alpha=0.5)
            ax.axhline(y=-0.8, color='red', linestyle='--', linewidth=1, alpha=0.5)

            ax.set_title("Zone별 Effect Size (Cohen's d)\n효과 크기: Small (0.2), Medium (0.5), Large (0.8)",
                        fontsize=14, fontweight='bold')
            ax.set_xlabel('Zone ID', fontsize=12)
            ax.set_ylabel("Cohen's d", fontsize=12)
            ax.legend()
            ax.grid(axis='y', alpha=0.3)

            plt.tight_layout()
            output_path = os.path.join(self.viz_dir, 'effect_size_comparison.png')
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()

            self.logger.info(f"✓ Effect Size 비교 차트 저장: {output_path}")

        except Exception as e:
            self.logger.error(f"✗ Effect Size 차트 생성 오류: {e}", exc_info=True)

    def plot_cpk_improvement(self, stats_df: pd.DataFrame):
        """
        Cpk 개선도 차트

        Parameters:
        -----------
        stats_df : pd.DataFrame
            통계 분석 결과
        """
        try:
            # 제어 구간만 필터링
            controlled_stats = stats_df[stats_df['control_type'] == 'controlled'].copy()

            if len(controlled_stats) == 0:
                self.logger.warning("제어 구간 데이터가 없어 Cpk 개선도 차트를 생성할 수 없습니다.")
                return

            # Zone별 평균 Cpk
            zone_cpk = controlled_stats.groupby('zone_id').agg({
                'before_cpk': 'mean',
                'after_cpk': 'mean'
            }).reset_index()

            # 시각화
            fig, ax = plt.subplots(figsize=(12, 6))

            x = np.arange(len(zone_cpk))
            width = 0.35

            bars1 = ax.bar(x - width/2, zone_cpk['before_cpk'], width, label='Before', color='lightcoral', edgecolor='black')
            bars2 = ax.bar(x + width/2, zone_cpk['after_cpk'], width, label='After', color='lightgreen', edgecolor='black')

            # Cpk 기준선
            ax.axhline(y=1.0, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Poor (< 1.0)')
            ax.axhline(y=1.33, color='orange', linestyle='--', linewidth=1, alpha=0.5, label='Fair (1.33)')
            ax.axhline(y=1.67, color='blue', linestyle='--', linewidth=1, alpha=0.5, label='Good (1.67)')

            ax.set_title('Zone별 Cpk 개선도\nCpk > 1.33: 양호한 공정 능력',
                        fontsize=14, fontweight='bold')
            ax.set_xlabel('Zone ID', fontsize=12)
            ax.set_ylabel('Cpk', fontsize=12)
            ax.set_xticks(x)
            ax.set_xticklabels(zone_cpk['zone_id'])
            ax.legend()
            ax.grid(axis='y', alpha=0.3)

            plt.tight_layout()
            output_path = os.path.join(self.viz_dir, 'cpk_improvement.png')
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()

            self.logger.info(f"✓ Cpk 개선도 차트 저장: {output_path}")

        except Exception as e:
            self.logger.error(f"✗ Cpk 개선도 차트 생성 오류: {e}", exc_info=True)

    def plot_control_effectiveness_summary(self, stats_df: pd.DataFrame):
        """
        제어 효과 요약 차트

        Parameters:
        -----------
        stats_df : pd.DataFrame
            통계 분석 결과
        """
        try:
            # 제어 구간만 필터링
            controlled_stats = stats_df[stats_df['control_type'] == 'controlled'].copy()

            if len(controlled_stats) == 0:
                self.logger.warning("제어 구간 데이터가 없어 제어 효과 요약 차트를 생성할 수 없습니다.")
                return

            # 요약 통계
            summary = {
                '통계적 유의': controlled_stats['statistically_significant'].sum(),
                '실질적 유의': controlled_stats['practically_significant'].sum(),
                '분산 개선': controlled_stats['variance_improved'].sum(),
                'Cpk 개선': controlled_stats['cpk_improved'].sum(),
                '제어 효과 있음': controlled_stats['control_effective'].sum()
            }

            total = len(controlled_stats)

            # 시각화
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

            # 1. 개수 막대 그래프
            categories = list(summary.keys())
            counts = list(summary.values())
            colors_bar = ['skyblue', 'lightgreen', 'orange', 'pink', 'red']

            bars = ax1.bar(categories, counts, color=colors_bar, edgecolor='black')
            ax1.axhline(y=total, color='gray', linestyle='--', linewidth=1, alpha=0.5, label=f'Total: {total}')

            # 막대 위에 값 표시
            for bar in bars:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height)}\n({height/total*100:.1f}%)',
                        ha='center', va='bottom', fontsize=10)

            ax1.set_title('제어 효과 요약 (개수)', fontsize=14, fontweight='bold')
            ax1.set_ylabel('개수', fontsize=12)
            ax1.legend()
            ax1.grid(axis='y', alpha=0.3)
            ax1.tick_params(axis='x', rotation=15)

            # 2. 비율 파이 차트
            effective_count = summary['제어 효과 있음']
            not_effective_count = total - effective_count

            sizes = [effective_count, not_effective_count]
            labels = [f'제어 효과 있음\n({effective_count}/{total})', f'제어 효과 없음\n({not_effective_count}/{total})']
            colors_pie = ['lightgreen', 'lightcoral']
            explode = (0.1, 0)

            ax2.pie(sizes, explode=explode, labels=labels, colors=colors_pie,
                   autopct='%1.1f%%', shadow=True, startangle=90, textprops={'fontsize': 12})
            ax2.set_title('제어 효과 비율', fontsize=14, fontweight='bold')

            plt.tight_layout()
            output_path = os.path.join(self.viz_dir, 'control_effectiveness_summary.png')
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()

            self.logger.info(f"✓ 제어 효과 요약 차트 저장: {output_path}")

        except Exception as e:
            self.logger.error(f"✗ 제어 효과 요약 차트 생성 오류: {e}", exc_info=True)

    def plot_control_vs_no_control(self, stats_df: pd.DataFrame):
        """
        제어 vs 비제어 구간 비교

        Parameters:
        -----------
        stats_df : pd.DataFrame
            통계 분석 결과
        """
        try:
            # 제어 구간과 비제어 구간 분리
            controlled_stats = stats_df[stats_df['control_type'] == 'controlled'].copy()
            no_control_stats = stats_df[stats_df['control_type'] == 'no_control'].copy()

            if len(controlled_stats) == 0 or len(no_control_stats) == 0:
                self.logger.warning("제어 또는 비제어 구간 데이터가 없어 비교 차트를 생성할 수 없습니다.")
                return

            # Zone별 평균 변화량
            control_mean_change = controlled_stats.groupby('zone_id')['mean_change'].mean()
            no_control_mean_change = no_control_stats.groupby('zone_id')['mean_change'].mean()

            # 공통 Zone만 추출
            common_zones = control_mean_change.index.intersection(no_control_mean_change.index)

            if len(common_zones) == 0:
                self.logger.warning("공통 Zone이 없어 비교 차트를 생성할 수 없습니다.")
                return

            control_mean_change = control_mean_change[common_zones]
            no_control_mean_change = no_control_mean_change[common_zones]

            # 시각화
            fig, ax = plt.subplots(figsize=(12, 6))

            x = np.arange(len(common_zones))
            width = 0.35

            bars1 = ax.bar(x - width/2, control_mean_change.values, width,
                          label='제어 구간', color='lightblue', edgecolor='black')
            bars2 = ax.bar(x + width/2, no_control_mean_change.values, width,
                          label='비제어 구간 (대조군)', color='lightgray', edgecolor='black')

            ax.axhline(y=0, color='black', linestyle='-', linewidth=1)

            ax.set_title('제어 vs 비제어 구간 평균 변화량 비교\n제어 구간의 변화가 비제어 구간보다 크면 제어 효과 있음',
                        fontsize=14, fontweight='bold')
            ax.set_xlabel('Zone ID', fontsize=12)
            ax.set_ylabel('평균 변화량 (After - Before)', fontsize=12)
            ax.set_xticks(x)
            ax.set_xticklabels(common_zones)
            ax.legend()
            ax.grid(axis='y', alpha=0.3)

            plt.tight_layout()
            output_path = os.path.join(self.viz_dir, 'control_vs_no_control.png')
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()

            self.logger.info(f"✓ 제어 vs 비제어 비교 차트 저장: {output_path}")

        except Exception as e:
            self.logger.error(f"✗ 제어 vs 비제어 비교 차트 생성 오류: {e}", exc_info=True)
