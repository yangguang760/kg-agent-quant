"""Tests for KG module"""

import sys
sys.path.insert(0, 'src')

import pytest
import numpy as np
import pandas as pd
from kg_quant.kg import (
    KGRetriever,
    KGFeatureGenerator,
    SemanticConsistencyChecker,
    KGExplainer,
)


class TestKGRetriever:
    """Tests for KGRetriever"""

    def test_load_kg(self):
        """Test loading knowledge graph"""
        retriever = KGRetriever(kg_dir="data/kg")

        assert retriever.layer1_concepts is not None
        assert retriever.layer2_relations is not None

    def test_get_statistics(self):
        """Test getting statistics"""
        retriever = KGRetriever(kg_dir="data/kg")

        stats = retriever.get_statistics()

        assert 'total_concepts' in stats
        assert 'total_relations' in stats
        assert stats['total_concepts'] > 0

    def test_search_concepts(self):
        """Test concept search"""
        retriever = KGRetriever(kg_dir="data/kg")

        results = retriever.search_concepts("ROE")

        assert isinstance(results, list)


class TestKGFeatureGenerator:
    """Tests for KGFeatureGenerator"""

    def test_init(self):
        """Test initialization"""
        generator = KGFeatureGenerator(
            kg_dir="data/kg",
            factor_json_path="data/sample/factors_sample.json"
        )

        assert generator.retriever is not None
        assert generator.evaluator is not None

    def test_resolve_valid_factors(self):
        """Test resolving valid factors"""
        generator = KGFeatureGenerator(
            kg_dir="data/kg",
            factor_json_path="data/sample/factors_sample.json"
        )

        factors = generator.resolve_valid_factors("quality", n_features=5)

        assert isinstance(factors, list)
        assert len(factors) <= 5

    def test_generate_features(self):
        """Test feature generation"""
        generator = KGFeatureGenerator(
            kg_dir="data/kg",
            factor_json_path="data/sample/factors_sample.json"
        )

        # Generate sample data
        data = generator._generate_sample_data(n_stocks=10, n_days=50)

        features = generator.generate_kg_features(
            factor_type="quality",
            n_features=5,
            data=data
        )

        assert isinstance(features, pd.DataFrame)
        if not features.empty:
            assert features.shape[0] == len(data)


class TestKGExplainer:
    """Tests for KGExplainer"""

    def test_explain_factor(self):
        """Test explaining a factor"""
        explainer = KGExplainer(kg_dir="data/kg")

        explanation = explainer.explain_factor("RANK(TS_MEAN($roe, 20))")

        assert explanation.factor_expression == "RANK(TS_MEAN($roe, 20))"
        assert explanation.explanation_confidence >= 0


class TestSemanticConsistencyChecker:
    """Tests for SemanticConsistencyChecker"""

    def test_check(self):
        """Test checking consistency"""
        checker = SemanticConsistencyChecker(kg_dir="data/kg")

        result = checker.check("ROE is a profitability indicator")

        assert result.hypothesis == "ROE is a profitability indicator"
        assert result.confidence >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])