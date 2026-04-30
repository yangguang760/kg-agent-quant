"""Tests for evaluation module"""

import sys
sys.path.insert(0, 'src')

import pytest
import numpy as np
import pandas as pd
from kg_quant.evaluation.metrics import (
    compute_ic,
    compute_rank_ic,
    compute_icir,
    compute_arr,
    compute_mdd,
    compute_ir,
    compute_calmar,
    FactorEvaluator,
)


class TestMetrics:
    """Tests for metric functions"""

    def test_compute_ic(self):
        """Test IC computation with arrays"""
        np.random.seed(42)
        n = 100

        # Positive correlation
        x = np.random.randn(n)
        y = x + np.random.randn(n) * 0.5

        ic = compute_ic(x, y)
        assert ic > 0

        # Negative correlation
        y_neg = -x + np.random.randn(n) * 0.5
        ic_neg = compute_ic(x, y_neg)
        assert ic_neg < 0

    def test_compute_ic_with_nan(self):
        """Test IC computation with NaN values"""
        x = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        ic = compute_ic(x, y)
        assert not np.isnan(ic)

    def test_compute_rank_ic(self):
        """Test RankIC computation"""
        np.random.seed(42)
        n = 100

        x = np.random.randn(n)
        y = np.random.randn(n)

        rank_ic = compute_rank_ic(x, y)
        assert -1 <= rank_ic <= 1

    def test_compute_icir(self):
        """Test ICIR computation"""
        ic_series = [0.05, 0.03, 0.08, 0.02, 0.06]

        icir = compute_icir(ic_series)
        assert icir >= 0

    def test_compute_arr(self):
        """Test ARR computation"""
        np.random.seed(42)
        returns = np.random.randn(252) * 0.01

        arr = compute_arr(returns)
        assert isinstance(arr, float)

    def test_compute_arr_single_return(self):
        """Test ARR with single return"""
        arr = compute_arr([0.05])
        assert arr > 0

    def test_compute_mdd(self):
        """Test MDD computation"""
        np.random.seed(42)
        returns = np.random.randn(100) * 0.01

        mdd = compute_mdd(returns)
        assert mdd >= 0

    def test_compute_ir(self):
        """Test IR computation"""
        np.random.seed(42)
        returns = np.random.randn(252) * 0.01

        ir = compute_ir(returns)
        assert isinstance(ir, float)

    def test_compute_calmar(self):
        """Test Calmar ratio computation"""
        np.random.seed(42)
        returns = np.random.randn(252) * 0.01

        calmar = compute_calmar(returns)
        assert isinstance(calmar, float)


class TestFactorEvaluator:
    """Tests for FactorEvaluator class"""

    def test_init(self):
        """Test evaluator initialization"""
        evaluator = FactorEvaluator()

        assert evaluator.annualization_factor == 252

        evaluator_custom = FactorEvaluator(annualization_factor=365)
        assert evaluator_custom.annualization_factor == 365

    def test_evaluate_strategy(self):
        """Test strategy evaluation"""
        evaluator = FactorEvaluator()

        np.random.seed(42)
        returns = pd.Series(np.random.randn(100) * 0.01)

        metrics = evaluator.evaluate_strategy(returns)

        assert 'arr' in metrics
        assert 'mdd' in metrics
        assert 'ir' in metrics
        assert 'calmar_ratio' in metrics


if __name__ == "__main__":
    pytest.main([__file__, "-v"])