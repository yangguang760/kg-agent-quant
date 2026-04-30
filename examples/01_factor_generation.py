#!/usr/bin/env python3
"""
Example 1: Factor Generation

Demonstrates how to generate alpha factors using the KG-Enhanced pipeline.
"""

import sys
sys.path.insert(0, 'src')

import pandas as pd
import numpy as np
from kg_quant import KGFeatureGenerator, QLIBExpressionEvaluator
from kg_quant.kg.explainer import KGExplainer

def main():
    print("=" * 60)
    print("KG-AgentQuant Example 1: Factor Generation")
    print("=" * 60)

    # Create generator with sample data
    generator = KGFeatureGenerator(
        kg_dir="data/kg",
        factor_json_path="data/sample/factors_sample.json"
    )

    # Generate sample market data
    print("\n[1] Generating sample market data...")
    n_stocks = 20
    n_days = 100
    dates = pd.date_range(start="2023-01-01", periods=n_days, freq='B')
    stocks = [f"SH{str(i).zfill(6)}" for i in range(n_stocks)]

    np.random.seed(42)
    n = len(dates) * len(stocks)

    data = pd.DataFrame({
        '$close': np.random.randn(n) * 10 + 100,
        '$open': np.random.randn(n) * 10 + 100,
        '$high': np.random.randn(n) * 10 + 105,
        '$low': np.random.randn(n) * 10 + 95,
        '$volume': np.random.rand(n) * 1e6,
        '$vwap': np.random.randn(n) * 10 + 100,
        '$returns': np.random.randn(n) * 0.02,
        '$roe': np.random.randn(n) * 0.1 + 0.15,
        '$roa': np.random.randn(n) * 0.05 + 0.08,
        '$pe': np.random.rand(n) * 20 + 10,
        '$pb': np.random.rand(n) * 3 + 0.5,
    }, index=pd.MultiIndex.from_product(
        [dates, stocks],
        names=['datetime', 'instrument']
    ))
    print(f"  Generated {len(data)} samples for {n_stocks} stocks, {n_days} days")

    # Test expression evaluator
    print("\n[2] Testing QLIB Expression Evaluator...")
    evaluator = QLIBExpressionEvaluator()
    evaluator.begin_batch(data)

    test_exprs = [
        "RANK(TS_MEAN($close, 10))",
        "RANK(1/$pe)",
        "TS_DELTA($roe, 1)",
    ]

    for expr in test_exprs:
        result = evaluator.evaluate(expr, data)
        print(f"  {expr}: shape={result.shape}, mean={result.mean():.4f}")

    evaluator.end_batch()

    # Generate quality factors
    print("\n[3] Generating Quality Factors...")
    quality_features = generator.generate_kg_features(
        factor_type="quality",
        n_features=5,
        data=data
    )
    print(f"  Generated {quality_features.shape[1]} quality factors")
    print(f"  Columns: {list(quality_features.columns)}")

    # Generate value factors
    print("\n[4] Generating Value Factors...")
    value_features = generator.generate_kg_features(
        factor_type="value",
        n_features=3,
        data=data
    )
    print(f"  Generated {value_features.shape[1]} value factors")
    print(f"  Columns: {list(value_features.columns)}")

    # Explain a factor
    print("\n[5] Explaining Factors...")
    explainer = KGExplainer(kg_dir="data/kg")

    test_factors = [
        "RANK(TS_MEAN($roe, 20))",
        "RANK(1/$pe)",
        "RANK(-TS_STD($returns, 20))",
    ]

    for factor in test_factors:
        explanation = explainer.explain_factor(factor)
        print(f"\n  Factor: {factor}")
        print(f"    Name: {explanation.factor_name}")
        print(f"    Indicators: {', '.join(explanation.used_indicators)}")
        print(f"    Theories: {', '.join(explanation.used_theories)}")
        print(f"    Patterns: {', '.join(explanation.used_patterns)}")
        print(f"    Confidence: {explanation.explanation_confidence:.2f}")

    # Get KG statistics
    print("\n[6] Knowledge Graph Statistics...")
    stats = generator.retriever.get_statistics() if generator.retriever else {}
    for key, value in stats.items():
        print(f"  {key}: {value}")

    print("\n" + "=" * 60)
    print("Example 1 completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()