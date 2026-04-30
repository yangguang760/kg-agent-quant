#!/usr/bin/env python3
"""
Test suite for KG-AgentQuant
"""

import sys
sys.path.insert(0, 'src')

import pytest
import numpy as np
import pandas as pd
from kg_quant import (
    QLIBExpressionEvaluator,
    KGExplainer,
    SemanticConsistencyChecker,
    FactorASTParser,
    FactorEvaluator,
)
from kg_quant.evaluation.metrics import (
    compute_ic,
    compute_rank_ic,
    compute_arr,
    compute_mdd,
)


class TestQLIBExpressionEvaluator:
    """Tests for QLIBExpressionEvaluator"""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing"""
        dates = pd.date_range(start="2023-01-01", periods=50, freq='B')
        stocks = [f"STOCK_{i:03d}" for i in range(10)]

        np.random.seed(42)
        n = len(dates) * len(stocks)

        return pd.DataFrame({
            '$close': np.random.randn(n) * 10 + 100,
            '$open': np.random.randn(n) * 10 + 100,
            '$high': np.random.randn(n) * 10 + 105,
            '$low': np.random.randn(n) * 10 + 95,
            '$volume': np.random.rand(n) * 1e6,
            '$roe': np.random.randn(n) * 0.1 + 0.15,
            '$pe': np.random.rand(n) * 20 + 10,
        }, index=pd.MultiIndex.from_product(
            [dates, stocks],
            names=['datetime', 'instrument']
        ))

    def test_evaluate_simple_expression(self, sample_data):
        """Test evaluating a simple expression"""
        evaluator = QLIBExpressionEvaluator()
        evaluator.begin_batch(sample_data)

        result = evaluator.evaluate("RANK($close)", sample_data)
        assert len(result) == len(sample_data)
        assert result.min() >= 0
        assert result.max() <= 1

        evaluator.end_batch()

    def test_evaluate_ts_mean(self, sample_data):
        """Test TS_MEAN operator"""
        evaluator = QLIBExpressionEvaluator()
        evaluator.begin_batch(sample_data)

        result = evaluator.evaluate("TS_MEAN($close, 5)", sample_data)
        assert len(result) == len(sample_data)
        # First 4 values should be NaN due to window
        assert pd.isna(result.iloc[:4]).all()

        evaluator.end_batch()

    def test_evaluate_arithmetic(self, sample_data):
        """Test arithmetic operations"""
        evaluator = QLIBExpressionEvaluator()
        evaluator.begin_batch(sample_data)

        result = evaluator.evaluate("$close / $open", sample_data)
        assert len(result) == len(sample_data)

        evaluator.end_batch()

    def test_evaluate_complex_expression(self, sample_data):
        """Test complex nested expression"""
        evaluator = QLIBExpressionEvaluator()
        evaluator.begin_batch(sample_data)

        result = evaluator.evaluate("RANK(TS_MEAN($close, 10))", sample_data)
        assert len(result) == len(sample_data)

        evaluator.end_batch()


class TestFactorEvaluator:
    """Tests for FactorEvaluator"""

    def test_compute_ic(self):
        """Test IC computation"""
        np.random.seed(42)
        n = 100
        factor = np.random.randn(n) * 0.1
        returns = factor * 0.5 + np.random.randn(n) * 0.02

        ic = compute_ic(factor, returns)
        assert -1 <= ic <= 1

    def test_compute_rank_ic(self):
        """Test RankIC computation"""
        np.random.seed(42)
        n = 100
        factor = np.random.randn(n)
        returns = np.random.randn(n)

        rank_ic = compute_rank_ic(factor, returns)
        assert -1 <= rank_ic <= 1

    def test_compute_arr(self):
        """Test ARR computation"""
        np.random.seed(42)
        returns = np.random.randn(100) * 0.01

        arr = compute_arr(returns)
        assert isinstance(arr, float)

    def test_compute_mdd(self):
        """Test MDD computation"""
        np.random.seed(42)
        returns = np.random.randn(100) * 0.01

        mdd = compute_mdd(returns)
        assert mdd >= 0


class TestKGExplainer:
    """Tests for KGExplainer"""

    def test_explain_factor(self):
        """Test factor explanation"""
        explainer = KGExplainer()

        explanation = explainer.explain_factor("RANK(TS_MEAN($roe, 20))")

        assert explanation.factor_expression == "RANK(TS_MEAN($roe, 20))"
        assert len(explanation.used_indicators) > 0
        assert explanation.explanation_confidence > 0

    def test_explain_batch(self):
        """Test batch explanation"""
        explainer = KGExplainer()

        factors = [
            "RANK(TS_MEAN($roe, 20))",
            "RANK(1/$pe)",
        ]

        explanations = explainer.explain_batch(factors)
        assert len(explanations) == 2


class TestSemanticConsistencyChecker:
    """Tests for SemanticConsistencyChecker"""

    def test_check_hypothesis(self):
        """Test hypothesis checking"""
        checker = SemanticConsistencyChecker()

        result = checker.check("ROE is positively correlated with stock returns")

        assert result.consistency_level.value in ['consistent', 'partial', 'inconsistent', 'unknown']
        assert 0 <= result.confidence <= 1

    def test_check_batch(self):
        """Test batch checking"""
        checker = SemanticConsistencyChecker()

        hypotheses = [
            "ROE is a profitability indicator",
            "Low PE stocks may be undervalued",
        ]

        results = checker.check_batch(hypotheses)
        assert len(results) == 2


class TestFactorASTParser:
    """Tests for FactorASTParser"""

    def test_parse_valid_expression(self):
        """Test parsing valid expression"""
        parser = FactorASTParser()

        tree = parser.parse_expression("RANK(TS_MEAN($close, 10))")
        assert tree is not None

    def test_validate_constraints(self):
        """Test constraint validation"""
        parser = FactorASTParser()

        tree = parser.parse_expression("RANK(TS_MEAN($close, 10))")
        valid, analysis = parser.validate_constraints(tree)

        assert valid
        assert analysis['valid']

    def test_compute_complexity(self):
        """Test complexity computation"""
        parser = FactorASTParser()

        tree = parser.parse_expression("RANK(TS_MEAN($close, 10))")
        complexity = parser.compute_complexity(tree)

        assert complexity > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])