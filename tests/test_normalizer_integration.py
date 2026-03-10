"""
통합 정규화 테스트

예측 모델과 최적화 모델의 정규화 일관성을 검증합니다.
"""

import numpy as np
import pytest
import sys
sys.path.insert(0, '/home/user/LLControl')

from apc_optimization.normalizer import ControlVariableNormalizer
from apc_optimization.cost_function import CostFunctionEvaluator
from apc_optimization.optimizer_engine import DifferentialEvolutionOptimizer
from apc_optimization.model_interface import CatBoostModelManager


class TestControlVariableNormalizer:
    """ControlVariableNormalizer 테스트"""

    @pytest.fixture
    def normalizer(self):
        """정규화기 인스턴스 반환"""
        return ControlVariableNormalizer(gv_max=2.0, rpm_max=50)

    # ====================================================================
    # MinMax 정규화 테스트
    # ====================================================================

    def test_normalize_for_cost_basic(self, normalizer):
        """기본 MinMax 정규화 테스트"""
        delta_gv = np.array([0.5, 1.0, 2.0])
        delta_rpm = 25.0

        gv_norm, rpm_norm = normalizer.normalize_for_cost(delta_gv, delta_rpm)

        np.testing.assert_array_almost_equal(gv_norm, np.array([0.25, 0.5, 1.0]))
        assert np.isclose(rpm_norm, 0.5)

    def test_normalize_for_cost_negative_values(self, normalizer):
        """음수 값 정규화 (절댓값 사용)"""
        delta_gv = np.array([-0.5, -1.0, -2.0])
        delta_rpm = -25.0

        gv_norm, rpm_norm = normalizer.normalize_for_cost(delta_gv, delta_rpm)

        np.testing.assert_array_almost_equal(gv_norm, np.array([0.25, 0.5, 1.0]))
        assert np.isclose(rpm_norm, 0.5)

    def test_normalize_for_cost_clipping(self, normalizer):
        """범위 클립 테스트"""
        delta_gv = np.array([5.0, 10.0])
        delta_rpm = 100.0

        gv_norm, rpm_norm = normalizer.normalize_for_cost(delta_gv, delta_rpm)

        assert np.all(gv_norm <= 1.0)
        assert np.all(gv_norm >= 0.0)
        assert rpm_norm <= 1.0

    def test_normalize_for_cost_zero(self, normalizer):
        """0 값 정규화 테스트"""
        delta_gv = np.array([0.0, 0.0])
        delta_rpm = 0.0

        gv_norm, rpm_norm = normalizer.normalize_for_cost(delta_gv, delta_rpm)

        np.testing.assert_array_almost_equal(gv_norm, np.array([0.0, 0.0]))
        assert np.isclose(rpm_norm, 0.0)

    # ====================================================================
    # 역정규화 테스트
    # ====================================================================

    def test_denormalize_basic(self, normalizer):
        """기본 역정규화 테스트"""
        gv_norm = np.array([0.25, 0.5, 1.0])
        rpm_norm = 0.5

        delta_gv, delta_rpm = normalizer.denormalize_control_vars(gv_norm, rpm_norm)

        np.testing.assert_array_almost_equal(delta_gv, np.array([0.5, 1.0, 2.0]))
        assert np.isclose(delta_rpm, 25.0)

    def test_roundtrip_consistency(self, normalizer):
        """정규화 → 역정규화 일관성 테스트"""
        original_gv = np.array([0.3, 0.7, 1.5])
        original_rpm = 35.0

        # 정규화
        gv_norm, rpm_norm = normalizer.normalize_for_cost(original_gv, original_rpm)

        # 역정규화
        gv_back, rpm_back = normalizer.denormalize_control_vars(gv_norm, rpm_norm)

        # 원본과 동일해야 함
        np.testing.assert_array_almost_equal(gv_back, np.abs(original_gv))
        assert np.isclose(rpm_back, np.abs(original_rpm))

    # ====================================================================
    # 에러 처리 테스트
    # ====================================================================

    def test_invalid_initialization(self):
        """잘못된 초기화 테스트"""
        with pytest.raises(ValueError):
            ControlVariableNormalizer(gv_max=-1.0, rpm_max=50)

        with pytest.raises(ValueError):
            ControlVariableNormalizer(gv_max=2.0, rpm_max=0)

    def test_nan_input(self, normalizer):
        """NaN 입력 에러 처리"""
        delta_gv = np.array([0.5, np.nan, 1.0])
        delta_rpm = 25.0

        with pytest.raises(ValueError):
            normalizer.normalize_for_cost(delta_gv, delta_rpm)

    def test_inf_input(self, normalizer):
        """Inf 입력 에러 처리"""
        delta_gv = np.array([0.5, np.inf, 1.0])
        delta_rpm = 25.0

        with pytest.raises(ValueError):
            normalizer.normalize_for_cost(delta_gv, delta_rpm)


class TestCostFunctionNormalization:
    """CostFunctionEvaluator와 정규화 테스트"""

    def test_cost_function_with_normalizer(self):
        """cost_function이 normalizer를 올바르게 사용하는지 테스트"""
        evaluator = CostFunctionEvaluator()

        # 테스트 데이터
        delta_gv = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
        delta_rpm = 25.0

        control_cost, details = evaluator.control_cost(delta_gv, delta_rpm)

        # 결과 검증
        assert isinstance(control_cost, (float, np.floating))
        assert 0.0 <= control_cost <= 1.0
        assert 'gv_normalized' in details
        assert 'rpm_normalized' in details
        assert 'gv_norm' in details
        assert 'rpm_norm' in details

    def test_normalized_control_values_in_cost_details(self):
        """정규화된 제어값이 cost details에 포함되어 있는지 검증"""
        evaluator = CostFunctionEvaluator()

        delta_gv = np.array([0.5] * 11)
        delta_rpm = 25.0

        _, details = evaluator.control_cost(delta_gv, delta_rpm)

        # 정규화된 값 확인
        expected_gv_norm = np.array([0.25] * 11)
        expected_rpm_norm = 0.5

        np.testing.assert_array_almost_equal(
            details['gv_normalized'], expected_gv_norm
        )
        assert np.isclose(details['rpm_normalized'], expected_rpm_norm)

    def test_cost_consistency_across_calls(self):
        """여러 번 호출해도 동일한 결과인지 확인"""
        evaluator = CostFunctionEvaluator()

        delta_gv = np.array([0.3, 0.4, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
        delta_rpm = 20.0

        # 첫 번째 호출
        cost1, details1 = evaluator.control_cost(delta_gv, delta_rpm)

        # 두 번째 호출
        cost2, details2 = evaluator.control_cost(delta_gv, delta_rpm)

        # 동일한 결과
        assert np.isclose(cost1, cost2)
        np.testing.assert_array_almost_equal(
            details1['gv_normalized'], details2['gv_normalized']
        )


class TestOptimizerNormalization:
    """DifferentialEvolutionOptimizer와 정규화 테스트"""

    def test_optimizer_initializes_with_normalizer(self):
        """optimizer가 정규화기를 제대로 초기화하는지 테스트"""
        model_manager = CatBoostModelManager()
        cost_evaluator = CostFunctionEvaluator()
        current_state = {'current_clr': np.random.randn(11, 3)}

        optimizer = DifferentialEvolutionOptimizer(
            model_manager=model_manager,
            cost_evaluator=cost_evaluator,
            current_state=current_state
        )

        # 정규화기 확인
        assert hasattr(optimizer, 'normalizer')
        assert optimizer.normalizer is not None
        assert optimizer.normalizer.gv_max == 2.0
        assert optimizer.normalizer.rpm_max == 50

    def test_normalizer_consistency_between_cost_and_optimizer(self):
        """cost_evaluator와 optimizer의 정규화기 설정이 일치하는지 확인"""
        model_manager = CatBoostModelManager()
        cost_evaluator = CostFunctionEvaluator()
        current_state = {'current_clr': np.random.randn(11, 3)}

        optimizer = DifferentialEvolutionOptimizer(
            model_manager=model_manager,
            cost_evaluator=cost_evaluator,
            current_state=current_state
        )

        # 정규화 기준 일치 확인
        assert optimizer.normalizer.gv_max == cost_evaluator.normalizer.gv_max
        assert optimizer.normalizer.rpm_max == cost_evaluator.normalizer.rpm_max


class TestNormalizationConsistency:
    """전체 시스템의 정규화 일관성 테스트"""

    def test_end_to_end_normalization_consistency(self):
        """
        End-to-end 정규화 일관성 테스트

        제어값 → 정규화 → 비용 계산 과정에서의 일관성 확인
        """
        # 시스템 초기화
        model_manager = CatBoostModelManager()
        cost_evaluator = CostFunctionEvaluator()
        current_state = {'current_clr': np.random.randn(11, 3)}

        optimizer = DifferentialEvolutionOptimizer(
            model_manager=model_manager,
            cost_evaluator=cost_evaluator,
            current_state=current_state
        )

        # 테스트 제어값
        delta_gv = np.array([0.5, 0.6, 0.7, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
        delta_rpm = 20.0

        # 비용 계산 (cost_evaluator의 normalizer 사용)
        cost, details = cost_evaluator.control_cost(delta_gv, delta_rpm)

        # 정규화 확인
        gv_norm_expected = np.abs(delta_gv) / 2.0
        rpm_norm_expected = np.abs(delta_rpm) / 50.0

        np.testing.assert_array_almost_equal(
            details['gv_normalized'], gv_norm_expected
        )
        assert np.isclose(details['rpm_normalized'], rpm_norm_expected)

        # 비용이 유효한 범위 내인지 확인
        assert 0.0 <= cost <= 1.0

    def test_normalizer_parameter_propagation(self):
        """정규화 파라미터가 제대로 전파되는지 확인"""
        custom_gv_max = 3.0
        custom_rpm_max = 60

        # 커스텀 normalizer 생성
        normalizer = ControlVariableNormalizer(
            gv_max=custom_gv_max,
            rpm_max=custom_rpm_max
        )

        # cost_evaluator 초기화 (커스텀 normalizer 사용)
        cost_evaluator = CostFunctionEvaluator(normalizer=normalizer)

        # 파라미터 확인
        assert cost_evaluator.normalizer.gv_max == custom_gv_max
        assert cost_evaluator.normalizer.rpm_max == custom_rpm_max


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
