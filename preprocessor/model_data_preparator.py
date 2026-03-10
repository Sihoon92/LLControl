import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import os
import logging
import utils

class ModelDataPreparator:
    """모델 학습용 데이터 준비 클래스"""

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
        self.logger = logger or logging.getLogger('coating_preprocessor.model_data')

        # Zone과 GV_GAP 매핑 (zone_1 -> GV_GAP02, zone_2 -> GV_GAP03, ...)
        self.zone_to_gv_gap = {i: f'GV_GAP{i+1:02d}' for i in range(1, 12)}

        # 상수
        self.EPSILON = 1e-4  # 제로 데이터 대체값
        self.N_ZONES = 11
        self.CENTER_ZONE = 6  # 중앙 zone

    def run(
        self,
        zone_analysis_file: str,
        meaningful_changes_file: str,
        output_file: str = None,
        mode: str = 'training'
    ) -> pd.DataFrame:
        """
        모델 학습용 데이터 준비 실행

        Parameters:
        -----------
        zone_analysis_file : str
            Zone 분석 결과 파일 경로
        meaningful_changes_file : str
            유의미한 변경 구간 파일 경로
        output_file : str, optional
            출력 파일 경로 (None이면 mode 기반 자동 생성)
        mode : str
            데이터 모드 ('training' 또는 'test'), 기본값: 'training'

        Returns:
        --------
        pd.DataFrame
            학습 준비된 데이터
        """
        self.logger.info("="*80)
        self.logger.info(f"모델 {mode.upper()} 데이터 준비 시작")
        self.logger.info("="*80)

        # 데이터 로드
        self.logger.info("[1단계] 데이터 로드")
        zone_df = self._load_data(zone_analysis_file)
        changes_df = self._load_data(meaningful_changes_file)

        if zone_df is None or changes_df is None:
            self.logger.error("데이터 로드 실패")
            return None

        self.logger.info(f"  Zone 분석 데이터: {len(zone_df)} 행")
        self.logger.info(f"  변경 구간 데이터: {len(changes_df)} 행")

        # Zone-wise 데이터 생성
        self.logger.info("[2단계] Zone-wise 데이터 구조화")
        model_data = self._prepare_zone_wise_data(zone_df, changes_df)

        if model_data is None or model_data.empty:
            self.logger.error("Zone-wise 데이터 생성 실패")
            return None

        self.logger.info(f"  생성된 {mode} 데이터: {len(model_data)} 행")

        # 결과 저장
        if output_file is None:
            # mode 기반 자동 파일명 생성
            filename = f'model_{mode}_data.xlsx'
            output_file = os.path.join(
                self.config.OUTPUT_DIR,
                filename
            )

        self._save_data(model_data, output_file)

        # 최종 요약
        self.logger.info("="*80)
        self.logger.info(f"모델 {mode.upper()} 데이터 준비 완료")
        self.logger.info("="*80)
        self.logger.info(f"총 샘플 수: {len(model_data)}")
        self.logger.info(f"Group 수: {model_data['group_id'].nunique()}")
        self.logger.info(f"Zone 수: {model_data['zone_id'].nunique()}")
        self.logger.info(f"출력 파일: {output_file}")
        self.logger.info("="*80)

        return model_data

    def _prepare_zone_wise_data(
        self,
        zone_df: pd.DataFrame,
        changes_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Zone-wise 데이터 구조로 변환

        Parameters:
        -----------
        zone_df : pd.DataFrame
            Zone 분석 결과
        changes_df : pd.DataFrame
            변경 구간 데이터

        Returns:
        --------
        pd.DataFrame
            Zone-wise 학습 데이터
        """
        all_rows = []

        # 각 group에 대해 처리
        for group_id in zone_df['group_id'].unique():
            group_zone_data = zone_df[zone_df['group_id'] == group_id]
            group_changes = changes_df[changes_df['group_id'] == group_id]

            if len(group_changes) == 0:
                self.logger.warning(f"  Group {group_id}: 변경 정보 없음")
                continue

            group_change = group_changes.iloc[0]

            # 각 zone에 대해 처리
            for _, zone_row in group_zone_data.iterrows():
                try:
                    zone_data = self._prepare_single_zone(
                        zone_row,
                        group_change,
                        changes_df
                    )

                    if zone_data is not None:
                        all_rows.append(zone_data)

                except Exception as e:
                    self.logger.warning(
                        f"  Group {group_id}, Zone {zone_row['zone_id']} 처리 오류: {e}"
                    )
                    continue

        if not all_rows:
            return pd.DataFrame()

        return pd.DataFrame(all_rows)

    def _prepare_single_zone(
        self,
        zone_row: pd.Series,
        group_change: pd.Series,
        changes_df: pd.DataFrame
    ) -> Optional[Dict]:
        """
        단일 Zone의 학습 데이터 준비

        Parameters:
        -----------
        zone_row : pd.Series
            Zone 분석 결과 행
        group_change : pd.Series
            해당 group의 변경 정보
        changes_df : pd.DataFrame
            전체 변경 정보 (인접 zone의 GV 값 참조용)

        Returns:
        --------
        Dict
            학습 데이터 딕셔너리
        """
        group_id = zone_row['group_id']
        zone_id = int(zone_row['zone_id'])
        n_divisions = int(zone_row['n_divisions'])

        # 최대 거리 (중앙 기준): zone 1 또는 11에서 중앙 6까지 = 5
        max_distance = self.CENTER_ZONE - 1  # 6 - 1 = 5

        # Step 1: Before/After 비율 추출 및 재정규화
        before_ratios = []
        after_ratios = []

        for i in range(1, n_divisions + 1):
            before_key = f'div_{i}_before_ratio'
            after_key = f'div_{i}_after_ratio'

            if before_key in zone_row and after_key in zone_row:
                before_ratios.append(zone_row[before_key])
                after_ratios.append(zone_row[after_key])

        if len(before_ratios) != n_divisions:
            self.logger.debug(f"  Group {group_id}, Zone {zone_id}: 비율 데이터 불완전")
            return None

        # 재정규화 (합이 1이 되도록)
        before_ratios = self._renormalize(before_ratios)
        after_ratios = self._renormalize(after_ratios)

        # Step 2: 제로 데이터 처리
        before_ratios = self._replace_zeros(before_ratios)
        after_ratios = self._replace_zeros(after_ratios)

        # Step 3: CLR 변환
        before_clr = self._compute_clr(before_ratios)
        after_clr = self._compute_clr(after_ratios)

        # Step 4: Delta CLR 계산
        diff_clr = [after - before for after, before in zip(after_clr, before_clr)]

        # Step 5: GV_GAP 변화량 추출
        gv_gap_col = self.zone_to_gv_gap[zone_id]

        delta_gv_self = self._get_gv_delta(group_change, gv_gap_col)

        delta_gv_left1 = self._get_gv_delta(
            group_change,
            self.zone_to_gv_gap.get(zone_id - 1, None)
        ) if zone_id > 1 else 0.0

        delta_gv_right1 = self._get_gv_delta(
            group_change,
            self.zone_to_gv_gap.get(zone_id + 1, None)
        ) if zone_id < self.N_ZONES else 0.0

        delta_gv_left2 = self._get_gv_delta(
            group_change,
            self.zone_to_gv_gap.get(zone_id - 2, None)
        ) if zone_id > 2 else 0.0

        delta_gv_right2 = self._get_gv_delta(
            group_change,
            self.zone_to_gv_gap.get(zone_id + 2, None)
        ) if zone_id < self.N_ZONES - 1 else 0.0

        # Step 6: RPM 변화량 추출 (전역 제어 변수)
        delta_rpm = self._get_rpm_delta(group_change)

        # Step 7: Zone 위치 특성 계산
        distance_from_center = abs(zone_id - self.CENTER_ZONE)
        is_edge = 1 if (zone_id == 1 or zone_id == self.N_ZONES) else 0
        normalized_position = (zone_id - 1) / (self.N_ZONES - 1)       # 0.0 (zone1) ~ 1.0 (zone11)
        normalized_distance = distance_from_center / max_distance        # 0.0 (중앙) ~ 1.0 (엣지)

        # 데이터 구조화
        zone_data = {
            'group_id': group_id,
            'zone_id': zone_id,
            'start_time': group_change.get('start_time', None),
            'end_time': group_change.get('end_time', None),
            'ucl': zone_row.get('ucl', None),
            'lcl': zone_row.get('lcl', None),
            'target': zone_row.get('target', None),
            'n_divisions': n_divisions
        }

        # Zone 위치 특성 (물리적 대칭성 반영)
        zone_data['zone_distance_from_center'] = distance_from_center
        zone_data['is_edge'] = is_edge
        zone_data['normalized_position'] = normalized_position
        zone_data['normalized_distance'] = normalized_distance

        # 입력 특성 (X) - 현재 상태 (CLR)
        for i, clr_val in enumerate(before_clr, 1):
            zone_data[f'current_CLR_{i}'] = clr_val

        # 입력 특성 (X) - 제어 변수 (전역)
        zone_data['delta_RPM'] = delta_rpm

        # 입력 특성 (X) - 제어 변수 (국소)
        zone_data['delta_GV_self'] = delta_gv_self
        zone_data['delta_GV_left1'] = delta_gv_left1
        zone_data['delta_GV_right1'] = delta_gv_right1
        zone_data['delta_GV_left2'] = delta_gv_left2
        zone_data['delta_GV_right2'] = delta_gv_right2

        # 출력 특성 (Y) - CLR 변화량
        for i, diff_val in enumerate(diff_clr, 1):
            zone_data[f'diff_CLR_{i}'] = diff_val

        # Before/After 원본 비율도 저장 (참고용)
        for i, (before, after) in enumerate(zip(before_ratios, after_ratios), 1):
            zone_data[f'before_ratio_{i}'] = before
            zone_data[f'after_ratio_{i}'] = after

        # Before/After CLR 값도 저장 (참고용)
        for i, (before, after) in enumerate(zip(before_clr, after_clr), 1):
            zone_data[f'before_CLR_{i}'] = before
            zone_data[f'after_CLR_{i}'] = after

        return zone_data

    def _renormalize(self, ratios: List[float]) -> List[float]:
        """
        비율을 재정규화 (합이 1이 되도록)

        Parameters:
        -----------
        ratios : List[float]
            원본 비율 리스트

        Returns:
        --------
        List[float]
            재정규화된 비율
        """
        total = sum(ratios)
        if total == 0 or np.isnan(total):
            # 모두 0이거나 NaN이면 균등 분배
            return [1.0 / len(ratios)] * len(ratios)

        return [r / total for r in ratios]

    def _replace_zeros(self, ratios: List[float]) -> List[float]:
        """
        제로 데이터를 작은 값으로 대체

        Parameters:
        -----------
        ratios : List[float]
            비율 리스트

        Returns:
        --------
        List[float]
            제로 대체된 비율
        """
        result = []
        n = len(ratios)
        n_zeros = sum(1 for r in ratios if r == 0 or np.isnan(r))

        if n_zeros == 0:
            return ratios

        # 0이 아닌 값들의 합
        non_zero_sum = sum(r for r in ratios if r > 0 and not np.isnan(r))

        # 남은 비율을 0이 아닌 값들에서 빌려옴
        borrow_per_zero = self.EPSILON
        total_borrow = borrow_per_zero * n_zeros

        if non_zero_sum > total_borrow:
            # 0이 아닌 값들에서 비례적으로 빌림
            adjustment_factor = (non_zero_sum - total_borrow) / non_zero_sum
        else:
            # 빌릴 수 없으면 균등 분배
            adjustment_factor = 0
            borrow_per_zero = 1.0 / n

        for r in ratios:
            if r == 0 or np.isnan(r):
                result.append(borrow_per_zero)
            else:
                result.append(r * adjustment_factor)

        # 재정규화하여 합이 정확히 1이 되도록
        return self._renormalize(result)

    def _compute_clr(self, ratios: List[float]) -> List[float]:
        """
        CLR (Centered Log-Ratio) 변환

        Parameters:
        -----------
        ratios : List[float]
            비율 리스트

        Returns:
        --------
        List[float]
            CLR 변환된 값
        """
        # 기하평균 계산
        log_ratios = [np.log(r) for r in ratios]
        geom_mean = np.exp(np.mean(log_ratios))

        # CLR 변환
        clr_values = [np.log(r / geom_mean) for r in ratios]

        return clr_values

    def _get_gv_delta(
        self,
        group_change: pd.Series,
        gv_gap_col: Optional[str]
    ) -> float:
        """
        GV_GAP 변화량 추출

        Parameters:
        -----------
        group_change : pd.Series
            변경 정보
        gv_gap_col : str
            GV_GAP 칼럼명

        Returns:
        --------
        float
            변화량 (after - before)
        """
        if gv_gap_col is None:
            return 0.0

        before_col = f'{gv_gap_col}_before'
        after_col = f'{gv_gap_col}_after'

        before_val = group_change.get(before_col, np.nan)
        after_val = group_change.get(after_col, np.nan)

        if pd.isna(before_val) or pd.isna(after_val):
            return 0.0

        return float(after_val - before_val)

    def _get_rpm_delta(self, group_change: pd.Series) -> float:
        """
        PUMP RPM 변화량 추출 (전역 제어 변수)

        Parameters:
        -----------
        group_change : pd.Series
            변경 정보

        Returns:
        --------
        float
            RPM 변화량 (after - before)
        """
        rpm_col = 'PUMP RPM'
        before_col = f'{rpm_col}_before'
        after_col = f'{rpm_col}_after'

        before_val = group_change.get(before_col, np.nan)
        after_val = group_change.get(after_col, np.nan)

        if pd.isna(before_val) or pd.isna(after_val):
            return 0.0

        return float(after_val - before_val)

    def _load_data(self, file_path: str) -> Optional[pd.DataFrame]:
        """
        데이터 파일 로드

        파일 형식에 따라 자동으로 로드:
        - Excel (.xlsx, .xls): xlwings로 로드 (회사 내부 이슈)
        - Parquet (.parquet): pandas로 로드
        - CSV (.csv): pandas로 로드
        """
        try:
            return utils.load_file(file_path, logger=self.logger)
        except Exception as e:
            self.logger.error(f"파일 로드 오류: {e}")
            return None

    def _save_data(self, df: pd.DataFrame, output_path: str):
        """데이터 저장"""
        try:
            if output_path.endswith('.csv'):
                df.to_csv(output_path, index=False, encoding='utf-8-sig')
            elif output_path.endswith(('.xlsx', '.xls')):
                df.to_excel(output_path, index=False)
            else:
                output_path = output_path + '.xlsx'
                df.to_excel(output_path, index=False)

            self.logger.info(f"  ✓ 데이터 저장: {output_path}")

        except Exception as e:
            self.logger.error(f"  ✗ 저장 오류: {e}")
            raise

    def inverse_clr_transform(
        self,
        predicted_diff_clr: List[float],
        current_clr: List[float]
    ) -> List[float]:
        """
        CLR 역변환 (예측용)

        Parameters:
        -----------
        predicted_diff_clr : List[float]
            모델이 예측한 CLR 변화량
        current_clr : List[float]
            현재 상태의 CLR 값

        Returns:
        --------
        List[float]
            예측된 비율 (%)
        """
        # 예측된 CLR 값 = 현재 + 변화량
        predicted_clr = [curr + diff for curr, diff
                        in zip(current_clr, predicted_diff_clr)]

        # 지수 변환
        exp_vals = [np.exp(clr) for clr in predicted_clr]

        # Softmax (정규화)
        total = sum(exp_vals)
        predicted_ratios = [exp_val / total for exp_val in exp_vals]

        return predicted_ratios
