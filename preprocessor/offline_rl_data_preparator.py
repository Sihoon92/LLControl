"""
Offline RL 학습을 위한 시계열 + 전체 Zone MDP 데이터 전처리 모듈

1행 = 1제어 이벤트 (Control Event)
- State S_t: 제어 직전 5분간 시계열 (5 timesteps x 11 zones x 3 CLR = 165차원)
- Action A_t: GV_GAP 변화량 11개 + RPM 변화량 1개 = 12차원
- Reward R_t: 제어 후 Mid 비율 보상 - Low/High 패널티 = 1차원
- Next State S_{t+1}: 제어 후 5분간 시계열 (165차원)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import os
import logging
import utils
from preprocessor.zone_analyzer import ZoneAnalyzer


class OfflineRLDataPreparator:
    """Offline RL (MDP 튜플) 데이터 생성 클래스"""

    CLR_COMPONENTS = ['Low', 'Mid', 'High']
    EPSILON = 1e-4

    def __init__(self, config, logger: Optional[logging.Logger] = None):
        self.config = config
        self.logger = logger or logging.getLogger('coating_preprocessor.offline_rl')

        self.n_zones = config.N_ZONES  # 11
        self.n_timesteps = config.OFFLINE_RL_TIMESTEPS  # 5
        self.n_divisions = config.OFFLINE_RL_N_DIVISIONS  # 3 (Low/Mid/High)
        self.dead_time = config.OFFLINE_RL_DEAD_TIME_MINUTES  # 0
        self.reward_alpha = config.OFFLINE_RL_REWARD_ALPHA  # 1.0

        # Zone -> GV_GAP 매핑 (zone_1 -> GV_GAP02, ...)
        self.zone_to_gv_gap = {i: f'GV_GAP{i+1:02d}' for i in range(1, 12)}

    def run(
        self,
        densitometer_data: pd.DataFrame,
        meaningful_changes: pd.DataFrame,
        zone_analysis_results: Optional[pd.DataFrame] = None,
        output_file: Optional[str] = None,
        mode: str = 'training'
    ) -> Optional[pd.DataFrame]:
        """
        Offline RL용 MDP 데이터 생성 실행

        Parameters
        ----------
        densitometer_data : pd.DataFrame
            추출된 밀도계 데이터 (group_id, control_type, before/after, time, Value1..N)
        meaningful_changes : pd.DataFrame
            유의미한 변경 구간 (group_id, UCL, LCL, GV_GAP*_before/after, PUMP RPM_before/after)
        zone_analysis_results : pd.DataFrame
            Zone 분석 결과 (group_id, zone_id 등 - 유효 group 확인용)
        output_file : str, optional
            출력 파일 경로
        mode : str
            'training' or 'test'

        Returns
        -------
        pd.DataFrame or None
        """
        self.logger.info("=" * 80)
        self.logger.info(f"Offline RL {mode.upper()} 데이터 준비 시작")
        self.logger.info("=" * 80)

        # 1. 입력 검증
        if densitometer_data is None or densitometer_data.empty:
            self.logger.error("밀도계 데이터가 비어있습니다.")
            return None
        if meaningful_changes is None or meaningful_changes.empty:
            self.logger.error("meaningful_changes 데이터가 비어있습니다.")
            return None

        # 2. Value 칼럼 식별
        meta_cols = {'group_id', 'control_type', 'before/after'}
        time_col = self._find_time_column(densitometer_data)
        if time_col is None:
            self.logger.error("시간 칼럼을 찾을 수 없습니다.")
            return None

        value_columns = [
            col for col in densitometer_data.columns
            if col not in meta_cols and col != time_col
            and ('value' in str(col).lower() or str(col).isdigit())
        ]
        self.logger.info(f"  전체 Value 칼럼 수: {len(value_columns)}")

        # 2-1. Boundary 필터링 (양 끝단 노이즈 칼럼 제거)
        zone_analyzer = ZoneAnalyzer(self.config, self.logger)
        left_boundary, right_boundary = zone_analyzer.find_boundaries(
            densitometer_data, value_columns,
            self.config.BOUNDARY_THRESHOLD,
            meaningful_changes=meaningful_changes
        )
        value_columns = value_columns[left_boundary:right_boundary + 1]
        self.logger.info(f"  유효 Value 칼럼 수: {len(value_columns)} (boundary: [{left_boundary}, {right_boundary}])")

        # 3. Zone 할당 (칼럼 -> Zone 매핑)
        zones = self._assign_zones(len(value_columns), self.n_zones)
        zone_cols_map = {}
        for zone_id in range(1, self.n_zones + 1):
            zone_mask = zones == zone_id
            zone_cols_map[zone_id] = [
                value_columns[i] for i in range(len(value_columns)) if zone_mask[i]
            ]

        # 4. 유효 group_id 추출 (controlled 타입만, zone_analysis에 존재하는 것만)
        controlled_groups = set(
            densitometer_data[
                densitometer_data['control_type'] == 'controlled'
            ]['group_id'].unique()
        )

        if zone_analysis_results is not None and not zone_analysis_results.empty:
            valid_zone_groups = set(zone_analysis_results['group_id'].unique())
            valid_groups = sorted(controlled_groups & valid_zone_groups)
        else:
            valid_groups = sorted(controlled_groups)

        changes_groups = set(meaningful_changes['group_id'].unique())
        valid_groups = [g for g in valid_groups if g in changes_groups]

        self.logger.info(f"  유효 Group 수: {len(valid_groups)}")

        if not valid_groups:
            self.logger.warning("유효한 Group이 없습니다.")
            return None

        # 5. 각 group에 대해 MDP 튜플 생성
        mdp_rows = []
        skipped = 0

        for group_id in valid_groups:
            try:
                row = self._build_mdp_tuple(
                    group_id=group_id,
                    densitometer_data=densitometer_data,
                    meaningful_changes=meaningful_changes,
                    zone_cols_map=zone_cols_map,
                    time_col=time_col
                )
                if row is not None:
                    mdp_rows.append(row)
                else:
                    skipped += 1
            except Exception as e:
                self.logger.warning(f"  Group {group_id} MDP 생성 실패: {e}")
                skipped += 1

        if not mdp_rows:
            self.logger.error("MDP 데이터 생성 결과가 없습니다.")
            return None

        result_df = pd.DataFrame(mdp_rows)

        self.logger.info(f"  생성된 MDP 행: {len(result_df)}, 건너뛴 Group: {skipped}")
        self.logger.info(f"  총 컬럼 수: {len(result_df.columns)}")

        # 6. 저장
        if output_file is None:
            filename = self.config.OUTPUT_OFFLINE_RL_DATA
            if mode == 'test':
                base, ext = os.path.splitext(filename)
                filename = f'{base}_test{ext}'
            output_file = os.path.join(self.config.OUTPUT_DIR, filename)

        self._save_data(result_df, output_file)

        self.logger.info("=" * 80)
        self.logger.info(f"Offline RL {mode.upper()} 데이터 준비 완료")
        self.logger.info(f"  MDP 행 수: {len(result_df)}")
        self.logger.info(f"  출력 파일: {output_file}")
        self.logger.info("=" * 80)

        return result_df

    def _build_mdp_tuple(
        self,
        group_id: int,
        densitometer_data: pd.DataFrame,
        meaningful_changes: pd.DataFrame,
        zone_cols_map: Dict[int, List[str]],
        time_col: str
    ) -> Optional[Dict]:
        """하나의 제어 이벤트에 대한 MDP 튜플 (S, A, R, S') 생성"""

        # 1. 변경 정보 가져오기
        group_changes = meaningful_changes[meaningful_changes['group_id'] == group_id]
        if group_changes.empty:
            return None
        group_change = group_changes.iloc[0]

        # UCL/LCL 가져오기
        ucl = group_change.get('UCL', np.nan)
        lcl = group_change.get('LCL', np.nan)
        target = group_change.get('TARGET', np.nan)

        if pd.isna(ucl) or pd.isna(lcl):
            self.logger.debug(f"  Group {group_id}: UCL/LCL 정보 없음")
            return None

        # 3구간 경계 생성 (Low/Mid/High)
        division_edges = np.linspace(lcl, ucl, self.n_divisions + 1)

        # 2. 밀도계 데이터 필터링 (controlled만)
        group_data = densitometer_data[
            (densitometer_data['group_id'] == group_id) &
            (densitometer_data['control_type'] == 'controlled')
        ].copy()

        if group_data.empty:
            return None

        # 시간 칼럼 datetime 변환
        group_data['_datetime'] = pd.to_datetime(group_data[time_col], errors='coerce')
        group_data = group_data.dropna(subset=['_datetime']).sort_values('_datetime')

        # Before/After 분리
        before_data = group_data[group_data['before/after'] == 'before'].copy()
        after_data = group_data[group_data['before/after'] == 'after'].copy()

        if before_data.empty or after_data.empty:
            self.logger.debug(f"  Group {group_id}: before 또는 after 데이터 없음")
            return None

        # 3. State 구성 (제어 전 5분)
        state_features, state_ratios = self._compute_timestep_features(
            data=before_data,
            zone_cols_map=zone_cols_map,
            division_edges=division_edges,
            prefix='S_tminus',
            reverse_time_index=True  # t-5, t-4, ..., t-1
        )
        if state_features is None:
            self.logger.debug(f"  Group {group_id}: State 특성 생성 실패")
            return None

        # 4. Action 구성
        action_features = self._compute_action_features(group_change)

        # 5. Next State 구성 (제어 후 5분, dead_time 적용)
        if self.dead_time > 0 and len(after_data) > 0:
            dead_time_cutoff = after_data['_datetime'].min() + pd.Timedelta(minutes=self.dead_time)
            after_data_trimmed = after_data[after_data['_datetime'] >= dead_time_cutoff]
        else:
            after_data_trimmed = after_data

        next_state_features, next_state_ratios = self._compute_timestep_features(
            data=after_data_trimmed,
            zone_cols_map=zone_cols_map,
            division_edges=division_edges,
            prefix='S_tplus',
            reverse_time_index=False  # t+1, t+2, ..., t+5
        )
        if next_state_features is None or next_state_ratios is None:
            self.logger.debug(f"  Group {group_id}: Next State 특성 생성 실패")
            return None

        # 6. Reward 계산
        reward = self._compute_reward(next_state_ratios)

        # 7. 메타데이터
        timestamp = before_data['_datetime'].max()  # 제어 직전 시점

        row = {
            'group_id': group_id,
            'timestamp': timestamp,
            'ucl': ucl,
            'lcl': lcl,
            'target': target if not pd.isna(target) else np.nan,
            'n_divisions': self.n_divisions,
        }

        row.update(state_features)
        row.update(action_features)
        row.update(next_state_features)
        row['reward'] = reward

        return row

    def _compute_timestep_features(
        self,
        data: pd.DataFrame,
        zone_cols_map: Dict[int, List[str]],
        division_edges: np.ndarray,
        prefix: str,
        reverse_time_index: bool = False
    ) -> Tuple[Optional[Dict], Optional[List[Dict]]]:
        """
        시계열 데이터에서 타임스텝별 Zone CLR 특성을 계산

        Parameters
        ----------
        data : pd.DataFrame
            before 또는 after 밀도계 데이터 (_datetime 칼럼 필요)
        zone_cols_map : dict
            zone_id -> [value 칼럼 리스트]
        division_edges : np.ndarray
            3구간 경계값 배열 (4개 값: [LCL, low_mid_boundary, mid_high_boundary, UCL])
        prefix : str
            'S_tminus' (State) 또는 'S_tplus' (Next State)
        reverse_time_index : bool
            True면 가장 오래된 것부터 1,2,3,...,5로 번호 부여 (t-5, t-4, ...)

        Returns
        -------
        (features_dict, ratios_list) or (None, None)
        """
        if data.empty:
            return None, None

        # 1분 간격 리샘플링
        timestep_data = self._resample_to_minutes(data, n_minutes=self.n_timesteps)

        if timestep_data is None or len(timestep_data) < self.n_timesteps:
            return None, None

        features = {}
        all_ratios = []

        for ts_idx in range(self.n_timesteps):
            if reverse_time_index:
                # State: ts_idx=0 -> t-5 (가장 먼 과거), ts_idx=4 -> t-1 (직전)
                time_label = self.n_timesteps - ts_idx
            else:
                # Next State: ts_idx=0 -> t+1, ts_idx=4 -> t+5
                time_label = ts_idx + 1

            ts_row = timestep_data.iloc[ts_idx]
            ts_ratios = {}

            for zone_id in range(1, self.n_zones + 1):
                zone_cols = zone_cols_map[zone_id]
                zone_values = ts_row[zone_cols].values.astype(float)

                # 유효값만 (0보다 큰 값)
                valid_values = zone_values[zone_values > 0]

                if len(valid_values) == 0:
                    # 데이터 없으면 균등 분배
                    ratios = [1.0 / self.n_divisions] * self.n_divisions
                else:
                    # 3구간 히스토그램
                    hist, _ = np.histogram(valid_values, bins=division_edges)
                    total = hist.sum()
                    if total == 0:
                        ratios = [1.0 / self.n_divisions] * self.n_divisions
                    else:
                        ratios = (hist / total).tolist()

                # 재정규화 + 제로 처리 + CLR 변환
                ratios = self._renormalize(ratios)
                ratios = self._replace_zeros(ratios)
                clr_values = self._compute_clr(ratios)

                # 비율 저장 (Reward 계산용)
                for c_idx, comp in enumerate(self.CLR_COMPONENTS):
                    ts_ratios[f'Z{zone_id}_{comp}'] = ratios[c_idx]

                # CLR 피처 저장
                for c_idx, comp in enumerate(self.CLR_COMPONENTS):
                    col_name = f'{prefix}{time_label}_Z{zone_id}_CLR_{comp}'
                    features[col_name] = clr_values[c_idx]

            all_ratios.append(ts_ratios)

        return features, all_ratios

    def _compute_action_features(self, group_change: pd.Series) -> Dict:
        """GV_GAP 변화량 (11개) + RPM 변화량 (1개) 추출"""
        features = {}

        for zone_id in range(1, self.n_zones + 1):
            gv_col = self.zone_to_gv_gap[zone_id]
            features[f'delta_GV_{zone_id}'] = self._get_gv_delta(group_change, gv_col)

        features['delta_RPM'] = self._get_rpm_delta(group_change)

        return features

    def _compute_reward(self, ratios_list: List[Dict]) -> float:
        """
        보상 함수 계산

        R = Σ(Mid 비율) - α × Σ(Low + High 비율) 의 5분 평균

        Parameters
        ----------
        ratios_list : list of dict
            각 타임스텝별 Zone 비율 딕셔너리 목록

        Returns
        -------
        float
        """
        if not ratios_list:
            return 0.0

        total_mid = 0.0
        total_penalty = 0.0
        count = 0

        for ts_ratios in ratios_list:
            for zone_id in range(1, self.n_zones + 1):
                mid = ts_ratios.get(f'Z{zone_id}_Mid', 0.0)
                low = ts_ratios.get(f'Z{zone_id}_Low', 0.0)
                high = ts_ratios.get(f'Z{zone_id}_High', 0.0)
                total_mid += mid
                total_penalty += (low + high)
                count += 1

        if count == 0:
            return 0.0

        # 평균으로 정규화
        n_timesteps = len(ratios_list)
        avg_mid = total_mid / n_timesteps
        avg_penalty = total_penalty / n_timesteps

        reward = avg_mid - self.reward_alpha * avg_penalty

        return float(reward)

    # ===================================================================
    # 유틸리티 메서드
    # ===================================================================

    def _resample_to_minutes(
        self,
        data: pd.DataFrame,
        n_minutes: int
    ) -> Optional[pd.DataFrame]:
        """
        데이터를 1분 간격으로 리샘플링하여 n_minutes개 타임스텝 추출

        가장 가까운 시점의 데이터를 사용하며, 충분한 시간 범위가 없으면
        가능한 만큼의 데이터를 균등 간격으로 샘플링합니다.
        """
        if data.empty or '_datetime' not in data.columns:
            return None

        data_sorted = data.sort_values('_datetime').reset_index(drop=True)

        if len(data_sorted) >= n_minutes:
            # 충분한 데이터가 있으면 균등 간격으로 샘플링
            indices = np.linspace(0, len(data_sorted) - 1, n_minutes, dtype=int)
            return data_sorted.iloc[indices].reset_index(drop=True)
        else:
            # 데이터 부족
            return None

    def _find_time_column(self, df: pd.DataFrame) -> Optional[str]:
        """밀도계 데이터에서 시간 칼럼 찾기"""
        # 'TIME' 우선
        for col in df.columns:
            if col.upper() == 'TIME':
                return col
        # 첫 번째 non-meta 칼럼
        meta_cols = {'group_id', 'control_type', 'before/after'}
        for col in df.columns:
            if col not in meta_cols:
                return col
        return None

    def _assign_zones(self, n_columns: int, n_zones: int = 11) -> np.ndarray:
        """칼럼에 Zone 번호 할당 (동일 간격)"""
        zone_size = n_columns / n_zones
        return np.ceil((np.arange(n_columns) + 1) / zone_size).astype(int)

    def _renormalize(self, ratios: List[float]) -> List[float]:
        """비율 재정규화 (합 = 1)"""
        total = sum(ratios)
        if total == 0 or np.isnan(total):
            return [1.0 / len(ratios)] * len(ratios)
        return [r / total for r in ratios]

    def _replace_zeros(self, ratios: List[float]) -> List[float]:
        """제로값을 EPSILON으로 대체 (CLR의 log(0) 방지)"""
        n = len(ratios)
        n_zeros = sum(1 for r in ratios if r == 0 or np.isnan(r))

        if n_zeros == 0:
            return ratios

        non_zero_sum = sum(r for r in ratios if r > 0 and not np.isnan(r))
        borrow_per_zero = self.EPSILON
        total_borrow = borrow_per_zero * n_zeros

        if non_zero_sum > total_borrow:
            adjustment_factor = (non_zero_sum - total_borrow) / non_zero_sum
        else:
            adjustment_factor = 0
            borrow_per_zero = 1.0 / n

        result = []
        for r in ratios:
            if r == 0 or np.isnan(r):
                result.append(borrow_per_zero)
            else:
                result.append(r * adjustment_factor)

        return self._renormalize(result)

    def _compute_clr(self, ratios: List[float]) -> List[float]:
        """CLR (Centered Log-Ratio) 변환"""
        log_ratios = [np.log(r) for r in ratios]
        geom_mean = np.exp(np.mean(log_ratios))
        return [np.log(r / geom_mean) for r in ratios]

    def _get_gv_delta(self, group_change: pd.Series, gv_gap_col: str) -> float:
        """GV_GAP 변화량 추출 (after - before)"""
        before_col = f'{gv_gap_col}_before'
        after_col = f'{gv_gap_col}_after'

        before_val = group_change.get(before_col, np.nan)
        after_val = group_change.get(after_col, np.nan)

        if pd.isna(before_val) or pd.isna(after_val):
            return 0.0

        return float(after_val - before_val)

    def _get_rpm_delta(self, group_change: pd.Series) -> float:
        """PUMP RPM 변화량 추출 (after - before)"""
        before_col = 'PUMP RPM_before'
        after_col = 'PUMP RPM_after'

        before_val = group_change.get(before_col, np.nan)
        after_val = group_change.get(after_col, np.nan)

        if pd.isna(before_val) or pd.isna(after_val):
            return 0.0

        return float(after_val - before_val)

    def _save_data(self, df: pd.DataFrame, output_path: str):
        """데이터 저장 (parquet 또는 excel)"""
        try:
            if output_path.endswith('.parquet'):
                df.to_parquet(output_path, index=False)
            elif output_path.endswith('.csv'):
                df.to_csv(output_path, index=False, encoding='utf-8-sig')
            elif output_path.endswith(('.xlsx', '.xls')):
                df.to_excel(output_path, index=False)
            else:
                # 기본: parquet
                df.to_parquet(output_path, index=False)

            self.logger.info(f"  ✓ 데이터 저장: {output_path}")

        except Exception as e:
            self.logger.error(f"  ✗ 저장 오류: {e}")
            raise
