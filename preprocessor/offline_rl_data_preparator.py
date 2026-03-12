"""
Offline RL 학습을 위한 전체 Zone MDP 데이터 전처리 모듈

zone_analysis_results의 존별 3구간(Low/Mid/High) 비율을 활용하여
전체 11 Zone을 하나의 행으로 구성

1행 = 1제어 이벤트 (Control Event)
- State S_t: 11 zones × 3 CLR = 33차원 (제어 전 5분 통합)
- Action A_t: GV_GAP 변화량 11개 + RPM 변화량 1개 = 12차원
- Reward R_t: Mid 비율 보상 - Low/High 패널티 = 1차원
- Next State S_{t+1}: 11 zones × 3 CLR = 33차원 (제어 후 5분 통합)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import os
import logging
import utils


class OfflineRLDataPreparator:
    """Offline RL (MDP 튜플) 데이터 생성 클래스"""

    CLR_COMPONENTS = ['Low', 'Mid', 'High']
    EPSILON = 1e-4

    def __init__(self, config, logger: Optional[logging.Logger] = None):
        self.config = config
        self.logger = logger or logging.getLogger('coating_preprocessor.offline_rl')

        self.n_zones = config.N_ZONES  # 11
        self.n_divisions = config.N_DIVISIONS  # 3 (Low/Mid/High)
        self.reward_alpha = config.OFFLINE_RL_REWARD_ALPHA  # 1.0

        # Zone -> GV_GAP 매핑 (zone_1 -> GV_GAP02, ...)
        self.zone_to_gv_gap = {i: f'GV_GAP{i+1:02d}' for i in range(1, 12)}

    def run(
        self,
        zone_analysis_results: pd.DataFrame,
        meaningful_changes: pd.DataFrame,
        output_file: Optional[str] = None,
        mode: str = 'training'
    ) -> Optional[pd.DataFrame]:
        """
        Offline RL용 MDP 데이터 생성 실행

        Parameters
        ----------
        zone_analysis_results : pd.DataFrame
            Zone 분석 결과 (group_id, zone_id, div_{n}_before/after_ratio 등)
        meaningful_changes : pd.DataFrame
            유의미한 변경 구간 (group_id, UCL, LCL, GV_GAP*_before/after, PUMP RPM_before/after)
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
        if zone_analysis_results is None or zone_analysis_results.empty:
            self.logger.error("zone_analysis_results 데이터가 비어있습니다.")
            return None
        if meaningful_changes is None or meaningful_changes.empty:
            self.logger.error("meaningful_changes 데이터가 비어있습니다.")
            return None

        # 2. 유효 group_id 추출 (zone_analysis와 meaningful_changes의 교집합)
        zone_groups = set(zone_analysis_results['group_id'].unique())
        changes_groups = set(meaningful_changes['group_id'].unique())
        valid_groups = sorted(zone_groups & changes_groups)

        self.logger.info(f"  유효 Group 수: {len(valid_groups)}")

        if not valid_groups:
            self.logger.warning("유효한 Group이 없습니다.")
            return None

        # 3. 각 group에 대해 MDP 튜플 생성
        mdp_rows = []
        skipped = 0

        for group_id in valid_groups:
            try:
                row = self._build_mdp_tuple(
                    group_id=group_id,
                    zone_analysis_results=zone_analysis_results,
                    meaningful_changes=meaningful_changes
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

        # 4. 저장
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
        zone_analysis_results: pd.DataFrame,
        meaningful_changes: pd.DataFrame
    ) -> Optional[Dict]:
        """하나의 제어 이벤트에 대한 MDP 튜플 (S, A, R, S') 생성"""

        # 1. 변경 정보 가져오기
        group_changes = meaningful_changes[meaningful_changes['group_id'] == group_id]
        if group_changes.empty:
            return None
        group_change = group_changes.iloc[0]

        # 2. 해당 group의 zone 분석 결과
        group_zones = zone_analysis_results[zone_analysis_results['group_id'] == group_id]
        if group_zones.empty or len(group_zones) < self.n_zones:
            self.logger.debug(f"  Group {group_id}: zone 분석 결과 부족 ({len(group_zones)}/{self.n_zones})")
            return None

        # UCL/LCL 가져오기
        ucl = group_change.get('UCL', np.nan)
        lcl = group_change.get('LCL', np.nan)
        target = group_change.get('TARGET', np.nan)

        if pd.isna(ucl) or pd.isna(lcl):
            self.logger.debug(f"  Group {group_id}: UCL/LCL 정보 없음")
            return None

        # 3. State 구성 (제어 전 5분 통합)
        state_features, state_ratios = self._compute_features_from_zone_analysis(
            group_zones, period='before', prefix='S'
        )
        if state_features is None:
            self.logger.debug(f"  Group {group_id}: State 특성 생성 실패")
            return None

        # 4. Action 구성
        action_features = self._compute_action_features(group_change)

        # 5. Next State 구성 (제어 후 5분 통합)
        ns_features, ns_ratios = self._compute_features_from_zone_analysis(
            group_zones, period='after', prefix='NS'
        )
        if ns_features is None:
            self.logger.debug(f"  Group {group_id}: Next State 특성 생성 실패")
            return None

        # 6. Reward 계산 (Next State 비율 기반)
        reward = self._compute_reward(ns_ratios)

        # 7. 메타데이터
        timestamp = group_change.get('start_time', None)

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
        row.update(ns_features)
        row['reward'] = reward

        return row

    def _compute_features_from_zone_analysis(
        self,
        group_zones: pd.DataFrame,
        period: str,
        prefix: str
    ) -> Tuple[Optional[Dict], Optional[Dict]]:
        """
        zone_analysis_results에서 직접 ratio 추출 후 CLR 변환

        Parameters
        ----------
        group_zones : pd.DataFrame
            해당 group의 zone_analysis 결과 (11개 행, zone_id 1~11)
        period : str
            'before' 또는 'after'
        prefix : str
            'S' (State) 또는 'NS' (Next State)

        Returns
        -------
        (features_dict, ratios_dict) or (None, None)
        """
        features = {}
        ratios_dict = {}

        for _, zone_row in group_zones.iterrows():
            zone_id = int(zone_row['zone_id'])
            n_div = int(zone_row['n_divisions'])

            # zone_analysis_results에서 ratio 직접 추출
            ratios = []
            for i in range(1, n_div + 1):
                ratio_key = f'div_{i}_{period}_ratio'
                if ratio_key not in zone_row:
                    return None, None
                ratios.append(float(zone_row[ratio_key]))

            # 재정규화 + 제로 처리 + CLR 변환
            ratios = self._renormalize(ratios)
            ratios = self._replace_zeros(ratios)
            clr_values = self._compute_clr(ratios)

            # CLR 피처 저장
            for c_idx, comp in enumerate(self.CLR_COMPONENTS):
                features[f'{prefix}_Z{zone_id}_CLR_{comp}'] = clr_values[c_idx]

            # 비율 저장 (Reward 계산용)
            for c_idx, comp in enumerate(self.CLR_COMPONENTS):
                ratios_dict[f'Z{zone_id}_{comp}'] = ratios[c_idx]

        return features, ratios_dict

    def _compute_action_features(self, group_change: pd.Series) -> Dict:
        """GV_GAP 변화량 (11개) + RPM 변화량 (1개) 추출"""
        features = {}

        for zone_id in range(1, self.n_zones + 1):
            gv_col = self.zone_to_gv_gap[zone_id]
            features[f'delta_GV_{zone_id}'] = self._get_gv_delta(group_change, gv_col)

        features['delta_RPM'] = self._get_rpm_delta(group_change)

        return features

    def _compute_reward(self, ratios: Dict[str, float]) -> float:
        """
        보상 함수 계산

        R = Σ(Mid 비율) - α × Σ(Low + High 비율)

        Parameters
        ----------
        ratios : dict
            Zone별 비율 딕셔너리 (Z{n}_Low, Z{n}_Mid, Z{n}_High)

        Returns
        -------
        float
        """
        if not ratios:
            return 0.0

        total_mid = sum(ratios.get(f'Z{z}_Mid', 0.0) for z in range(1, self.n_zones + 1))
        total_penalty = sum(
            ratios.get(f'Z{z}_Low', 0.0) + ratios.get(f'Z{z}_High', 0.0)
            for z in range(1, self.n_zones + 1)
        )

        reward = total_mid - self.reward_alpha * total_penalty

        return float(reward)

    # ===================================================================
    # 유틸리티 메서드
    # ===================================================================

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
        """데이터 저장 (엑셀 + 파케이 동시 저장)"""
        utils.save_to_excel(df, output_path, format='both', logger=self.logger)
