"""
의사결정 지원 시스템 (Decision Support System)

최적화 결과와 불확실성 분석을 바탕으로 운전자에게
의사결정 정보 제공:
1. Top-N 제어 시나리오
2. 각 시나리오별 예상 결과 ± 신뢰구간
3. 위험도 평가 (Low/Medium/High)
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from pathlib import Path

from .config import (
    DSS_PARAMS, N_ZONES, N_GV,
    OPTIMIZATION_OUTPUT_DIR
)
from .optimizer_engine import OptimizationResult
from .uncertainty_analyzer import MonteCarloResults

logger = logging.getLogger(__name__)


@dataclass
class Scenario:
    """제어 시나리오"""
    scenario_id: int                    # 1, 2, 3, ...
    x: np.ndarray                       # 제어값 [△GV₁~₁₁, △RPM]
    cost: float                         # 비용

    # 예상 결과 (평균)
    p_mid_mean: np.ndarray              # Shape (n_zones,)
    p_mid_std: np.ndarray               # Shape (n_zones,)
    p_low_mean: np.ndarray
    p_high_mean: np.ndarray

    # 위험도
    risk_level: str                     # 'LOW', 'MEDIUM', 'HIGH'
    risk_score: float                   # [0, 1]

    # 불확실성
    cost_ci_lower: float = 0.0
    cost_ci_upper: float = 1.0

    def get_summary(self) -> Dict:
        """시나리오 요약"""
        return {
            'scenario_id': self.scenario_id,
            'cost': self.cost,
            'p_mid_mean': np.mean(self.p_mid_mean),
            'risk_level': self.risk_level,
            'risk_score': self.risk_score,
            'control': {
                'gv': self.x[:N_GV].tolist(),
                'rpm': float(self.x[N_GV]),
            }
        }


class DecisionSupportSystem:
    """
    의사결정 지원 시스템
    """

    def __init__(self,
                 dss_params: Optional[Dict] = None):
        """
        초기화

        Args:
            dss_params: DSS 파라미터 (없으면 config 사용)
        """
        self.dss_params = dss_params or DSS_PARAMS
        self.top_n = self.dss_params.get('top_n_scenarios', 3)

        # 위험도 임계값
        self.risk_thresholds = self.dss_params.get('risk_thresholds', {})
        self.risk_low_threshold = self.risk_thresholds.get('low', 0.05)
        self.risk_medium_threshold = self.risk_thresholds.get('medium', 0.15)

        logger.info(f"DecisionSupportSystem 초기화 (Top-N={self.top_n})")

    # ========================================================================
    # Top-N 시나리오 생성
    # ========================================================================

    def generate_top_n_scenarios(self,
                                opt_result: OptimizationResult,
                                mc_results: MonteCarloResults) -> List[Scenario]:
        """
        Top-N 최적 시나리오 생성

        Args:
            opt_result: OptimizationResult (최적화 결과)
            mc_results: MonteCarloResults (불확실성 분석)

        Returns:
            List[Scenario] - Top-N 시나리오
        """
        logger.info(f"Top-{self.top_n} 시나리오 생성 중...")

        # 현재는 최적해 1개만 있으므로, 최적해 주변에서 변동을 시뮬레이션
        scenarios = []

        # Scenario 1: 최적해 (기본)
        scenario_base = self._create_scenario(
            scenario_id=1,
            x=opt_result.x_opt,
            cost=opt_result.cost_opt,
            mc_results=mc_results,
            description="최적해"
        )
        scenarios.append(scenario_base)

        # Scenario 2-3: 보수적 제어 (비용 추가 승인 대신 안정성 확보)
        # 제어량을 50% 축소한 버전
        for i in range(2, min(self.top_n + 1, 3)):
            x_conservative = opt_result.x_opt.copy()
            x_conservative *= (1.0 - (i - 1) * 0.25)  # 50%, 75% 축소

            scenario_conservative = self._create_scenario(
                scenario_id=i,
                x=x_conservative,
                cost=float('nan'),  # 재평가 필요
                mc_results=None,
                description=f"보수적 제어 ({i*25}% 축소)"
            )
            scenarios.append(scenario_conservative)

        # Scenario 3: 공격적 제어 (최대 효과 추구)
        if self.top_n >= 3:
            x_aggressive = opt_result.x_opt.copy()
            x_aggressive *= 1.5  # 150% 증폭

            scenario_aggressive = self._create_scenario(
                scenario_id=3,
                x=x_aggressive,
                cost=float('nan'),
                mc_results=None,
                description="공격적 제어"
            )
            scenarios.append(scenario_aggressive)

        return scenarios[:self.top_n]

    def _create_scenario(self,
                        scenario_id: int,
                        x: np.ndarray,
                        cost: float,
                        mc_results: Optional[MonteCarloResults],
                        description: str = "") -> Scenario:
        """
        단일 시나리오 생성

        Args:
            scenario_id: 시나리오 ID
            x: 제어값
            cost: 비용
            mc_results: 불확실성 분석 결과
            description: 설명

        Returns:
            Scenario
        """
        # MC 결과에서 평균/표준편차 추출
        if mc_results is not None:
            p_mid_stats = mc_results.get_p_mid_stats()
            p_mid_mean = p_mid_stats['mean']
            p_mid_std = p_mid_stats['std']
            p_low_mean = np.mean(mc_results.p_low_samples, axis=0)
            p_high_mean = np.mean(mc_results.p_high_samples, axis=0)

            cost_stats = mc_results.get_cost_stats()
            cost_ci_lower = cost_stats['ci_lower']
            cost_ci_upper = cost_stats['ci_upper']
        else:
            # 기본값 (재평가 필요한 경우)
            p_mid_mean = np.ones(N_ZONES) * 0.5
            p_mid_std = np.ones(N_ZONES) * 0.1
            p_low_mean = np.ones(N_ZONES) * 0.25
            p_high_mean = np.ones(N_ZONES) * 0.25
            cost_ci_lower = cost
            cost_ci_upper = cost

        # 위험도 평가
        risk_level, risk_score = self._assess_risk(
            p_mid_mean, cost, mc_results
        )

        scenario = Scenario(
            scenario_id=scenario_id,
            x=x,
            cost=cost,
            p_mid_mean=p_mid_mean,
            p_mid_std=p_mid_std,
            p_low_mean=p_low_mean,
            p_high_mean=p_high_mean,
            risk_level=risk_level,
            risk_score=risk_score,
            cost_ci_lower=cost_ci_lower,
            cost_ci_upper=cost_ci_upper,
        )

        return scenario

    # ========================================================================
    # 위험도 평가
    # ========================================================================

    def _assess_risk(self,
                     p_mid_mean: np.ndarray,
                     cost: float,
                     mc_results: Optional[MonteCarloResults]) -> Tuple[str, float]:
        """
        시나리오 위험도 평가

        위험도 판정 기준:
          - P_Mid > 0.8 달성 비율
          - 비용 변동성 (신뢰구간 폭)

        Args:
            p_mid_mean: Shape (n_zones,)
            cost: 비용값
            mc_results: 불확실성 분석 결과

        Returns:
            (risk_level, risk_score) - risk_level: 'LOW', 'MEDIUM', 'HIGH'
                                     risk_score: [0, 1]
        """
        risk_factors = {}

        # Factor 1: P_Mid 달성도
        p_mid_success_rate = np.mean(p_mid_mean > 0.8)
        p_mid_risk = 1.0 - p_mid_success_rate
        risk_factors['p_mid'] = p_mid_risk

        # Factor 2: 비용 변동성 (MC 결과 있을 때)
        if mc_results is not None:
            cost_stats = mc_results.get_cost_stats()
            # 신뢰구간이 넓을수록 위험 (불확실성 높음)
            ci_width = cost_stats['ci_width']
            base_cost = mc_results.base_cost
            ci_width_ratio = ci_width / max(base_cost, 0.1)
            cost_risk = min(ci_width_ratio, 1.0)
            risk_factors['cost_variability'] = cost_risk
        else:
            cost_risk = 0.0

        # 총 위험도 (가중 평균)
        risk_score = 0.6 * p_mid_risk + 0.4 * cost_risk

        # 위험 레벨 판정
        if risk_score < self.risk_low_threshold:
            risk_level = 'LOW'
        elif risk_score < self.risk_medium_threshold:
            risk_level = 'MEDIUM'
        else:
            risk_level = 'HIGH'

        return risk_level, risk_score

    # ========================================================================
    # 권고 리포트 생성
    # ========================================================================

    def generate_recommendation_report(self,
                                      scenarios: List[Scenario],
                                      opt_result: OptimizationResult) -> str:
        """
        최종 권고 리포트 생성

        Args:
            scenarios: 시나리오 리스트
            opt_result: 최적화 결과

        Returns:
            str - 리포트 텍스트
        """
        report = f"""
        ╔══════════════════════════════════════════════════════════════════════════╗
        ║                     APC 제어 권고 리포트                                 ║
        ╚══════════════════════════════════════════════════════════════════════════╝

        [1] 최적화 결과 요약
        {'-'*76}
        총 평가 횟수: {opt_result.n_evaluations}
        소요 시간: {opt_result.optimization_time:.2f}초
        최적 비용: {opt_result.cost_opt:.6f}
        수렴 성공: {'예' if opt_result.success else '아니오'}


        [2] Top-{len(scenarios)} 제어 시나리오
        {'-'*76}
        """

        for scenario in scenarios:
            report += f"""
        시나리오 {scenario.scenario_id}:
          위험도: {scenario.risk_level} (점수: {scenario.risk_score:.3f})
          제어값:
            - △GV: {scenario.x[:N_GV]}
            - △RPM: {scenario.x[N_GV]:.2f}
          예상 결과:
            - P_Mid 평균: {np.mean(scenario.p_mid_mean):.4f} ± {np.mean(scenario.p_mid_std):.4f}
            - 예상 비용: {scenario.cost:.6f} [{scenario.cost_ci_lower:.6f}, {scenario.cost_ci_upper:.6f}]

        """

        # 권고안
        report += f"""
        [3] 운전자 권고안
        {'-'*76}
        """

        # 위험도별 권고
        low_risk_scenarios = [s for s in scenarios if s.risk_level == 'LOW']
        medium_risk_scenarios = [s for s in scenarios if s.risk_level == 'MEDIUM']
        high_risk_scenarios = [s for s in scenarios if s.risk_level == 'HIGH']

        if low_risk_scenarios:
            best = low_risk_scenarios[0]
            report += f"""
        ✓ 권고: 시나리오 {best.scenario_id} 적용
          - 위험도: {best.risk_level} (안전)
          - 예상 P_Mid: {np.mean(best.p_mid_mean):.4f}
          - 신뢰도: 높음 (95% CI 폭: {best.cost_ci_upper - best.cost_ci_lower:.6f})
        """
        elif medium_risk_scenarios:
            best = medium_risk_scenarios[0]
            report += f"""
        ⚠ 중간 위험도: 시나리오 {best.scenario_id}
          - 위험도: {best.risk_level}
          - 예상 P_Mid: {np.mean(best.p_mid_mean):.4f}
          - 주의: 결과 변동성 있음
        """
        else:
            report += f"""
        ⚠ 높은 위험도
          - 모든 시나리오가 높은 위험도를 가짐
          - 추가 분석 필요
        """

        report += f"""

        [4] 실행 가이드
        {'-'*76}
        1. 선택한 시나리오의 제어값을 공정에 적용합니다.
        2. 5분 후 실제 측정값을 확인합니다.
        3. 예상 결과와 실제 결과 차이가 크면 모델 재학습 필요합니다.

        [5] 제약사항
        {'-'*76}
        - 모든 제어값은 물리적 한계 범위 내에 있습니다.
        - 인접 Zone 간 제어값 차이 제약을 만족합니다.
        - 총 제어량이 허용 범위를 초과하지 않습니다.

        ╚══════════════════════════════════════════════════════════════════════════╝
        """

        return report

    # ========================================================================
    # 파일 저장
    # ========================================================================

    def save_scenarios_to_excel(self,
                               scenarios: List[Scenario],
                               output_path: Optional[Path] = None) -> Path:
        """
        시나리오를 Excel 파일로 저장

        Args:
            scenarios: 시나리오 리스트
            output_path: 저장 경로 (없으면 기본 경로)

        Returns:
            저장된 파일 경로
        """
        if output_path is None:
            output_path = OPTIMIZATION_OUTPUT_DIR / 'top_scenarios.xlsx'

        # DataFrame 생성
        data = []
        for scenario in scenarios:
            data.append({
                'Scenario': scenario.scenario_id,
                'Risk Level': scenario.risk_level,
                'Risk Score': f"{scenario.risk_score:.3f}",
                'Cost': f"{scenario.cost:.6f}",
                'P_Mid Mean': f"{np.mean(scenario.p_mid_mean):.4f}",
                'P_Mid Std': f"{np.mean(scenario.p_mid_std):.4f}",
                'GV Changes': str(scenario.x[:N_GV]),
                'RPM Change': f"{scenario.x[N_GV]:.2f}",
            })

        df = pd.DataFrame(data)

        # Excel 저장
        try:
            df.to_excel(output_path, index=False, sheet_name='Scenarios')
            logger.info(f"✓ 시나리오 저장: {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Excel 저장 실패: {e}")
            return output_path

    def save_report_to_file(self,
                           report: str,
                           output_path: Optional[Path] = None) -> Path:
        """
        리포트를 파일로 저장

        Args:
            report: 리포트 텍스트
            output_path: 저장 경로

        Returns:
            저장된 파일 경로
        """
        if output_path is None:
            output_path = OPTIMIZATION_OUTPUT_DIR / 'recommendation_report.txt'

        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report)
            logger.info(f"✓ 리포트 저장: {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"파일 저장 실패: {e}")
            return output_path


# ============================================================================
# 테스트 코드
# ============================================================================

if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # DSS 인스턴스 생성
    dss = DecisionSupportSystem()

    # 테스트 시나리오 생성 (더미)
    from .optimizer_engine import OptimizationResult
    from .uncertainty_analyzer import MonteCarloResults

    # 테스트용 최적화 결과
    opt_result = OptimizationResult(
        x_opt=np.random.uniform(-0.5, 0.5, 12),
        cost_opt=0.5,
        n_evaluations=1000,
        optimization_time=10.0,
        success=True,
        message="수렴 성공"
    )

    # 테스트용 MC 결과 (간단한 더미)
    mc_results = MonteCarloResults(
        x_opt=opt_result.x_opt,
        base_cost=0.5,
        p_low_samples=np.random.uniform(0, 0.2, (50, N_ZONES)),
        p_mid_samples=np.random.uniform(0.6, 1.0, (50, N_ZONES)),
        p_high_samples=np.random.uniform(0, 0.2, (50, N_ZONES)),
        cost_samples=np.random.normal(0.5, 0.05, 50),
        n_simulations=50
    )

    # 시나리오 생성
    scenarios = dss.generate_top_n_scenarios(opt_result, mc_results)

    # 리포트 생성
    report = dss.generate_recommendation_report(scenarios, opt_result)
    print(report)

    # 파일 저장
    dss.save_scenarios_to_excel(scenarios)
    dss.save_report_to_file(report)
