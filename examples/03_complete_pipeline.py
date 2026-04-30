#!/usr/bin/env python3
"""
Example 3: Complete Pipeline

Demonstrates the complete KG-Enhanced alpha factor research pipeline.
"""

import sys
sys.path.insert(0, 'src')

import json
import pandas as pd
import numpy as np
from pathlib import Path

from kg_quant import (
    KGFeatureGenerator,
    QLIBExpressionEvaluator,
    KGExplainer,
    SemanticConsistencyChecker,
    FactorEvaluator
)
from kg_quant.evaluation.metrics import compute_ic, compute_rank_ic


def generate_sample_market_data(n_stocks=30, n_days=120, seed=42):
    """Generate realistic sample market data"""
    np.random.seed(seed)

    dates = pd.date_range(start="2023-01-01", periods=n_days, freq='B')
    stocks = [f"SH{str(i).zfill(6)}" for i in range(n_stocks)]

    n = len(dates) * len(stocks)

    # Generate price series with trends
    all_data = {}

    for i, stock in enumerate(stocks):
        start_price = 100 + np.random.rand() * 50
        returns = np.random.randn(n_days) * 0.02

        # Add momentum component
        momentum = 0.01
        for t in range(1, n_days):
            returns[t] += momentum * returns[t-1]

        price_series = start_price * np.cumprod(np.concatenate([[1], 1 + returns]))

        for j, date in enumerate(dates):
            idx = i * n_days + j
            all_data[idx] = {
                'datetime': date,
                'instrument': stock,
                '$close': price_series[j],
                '$open': price_series[j] * (1 + np.random.randn() * 0.01),
                '$high': max(price_series[j], price_series[j] * (1 + abs(np.random.randn()) * 0.02)),
                '$low': min(price_series[j], price_series[j] * (1 - abs(np.random.randn()) * 0.02)),
                '$volume': np.random.rand() * 1e7,
                '$vwap': price_series[j] * (1 + np.random.randn() * 0.005),
                '$returns': returns[j] if j > 0 else 0,
                '$roe': np.clip(np.random.randn() * 0.1 + 0.15, 0, 1),
                '$roa': np.clip(np.random.randn() * 0.05 + 0.08, 0, 0.5),
                '$pe': np.clip(np.random.rand() * 30 + 5, 5, 100),
                '$pb': np.clip(np.random.rand() * 5 + 0.5, 0.1, 20),
            }

    df = pd.DataFrame.from_dict(all_data, orient='index')
    df = df.set_index(['datetime', 'instrument'])

    return df


def main():
    print("=" * 70)
    print("KG-AgentQuant Example 3: Complete Pipeline")
    print("=" * 70)

    # Configuration
    kg_dir = "data/kg"
    factor_json = "data/sample/factors_sample.json"

    # Step 1: Initialize Components
    print("\n[Step 1] Initializing Components...")
    generator = KGFeatureGenerator(kg_dir=kg_dir, factor_json_path=factor_json)
    evaluator = QLIBExpressionEvaluator()
    explainer = KGExplainer(kg_dir=kg_dir)
    checker = SemanticConsistencyChecker(kg_dir=kg_dir)
    metric_eval = FactorEvaluator()

    print("  ✓ KGFeatureGenerator")
    print("  ✓ QLIBExpressionEvaluator")
    print("  ✓ KGExplainer")
    print("  ✓ SemanticConsistencyChecker")
    print("  ✓ FactorEvaluator")

    # Step 2: Load Knowledge Graph
    print("\n[Step 2] Loading Knowledge Graph...")
    kg_stats = generator.retriever.get_statistics()
    print(f"  Total concepts: {kg_stats.get('total_concepts', 'N/A')}")
    print(f"  Total relations: {kg_stats.get('total_relations', 'N/A')}")
    print(f"  Relation types: {kg_stats.get('relation_types', 'N/A')}")

    # Step 3: Generate Market Data
    print("\n[Step 3] Generating Sample Market Data...")
    market_data = generate_sample_market_data(n_stocks=30, n_days=120)
    print(f"  Data shape: {market_data.shape}")
    print(f"  Date range: {market_data.index.get_level_values('datetime').min()} to {market_data.index.get_level_values('datetime').max()}")
    print(f"  Number of stocks: {len(market_data.index.get_level_values('instrument').unique())}")

    # Step 4: Generate Factors
    print("\n[Step 4] Generating Alpha Factors...")

    factor_types = ["quality", "value", "momentum"]
    all_features = {}

    for factor_type in factor_types:
        features = generator.generate_kg_features(
            factor_type=factor_type,
            n_features=5,
            data=market_data
        )
        if not features.empty:
            all_features[factor_type] = features
            print(f"  {factor_type}: {features.shape[1]} factors generated")

    # Combine all features
    if all_features:
        combined_features = pd.concat(all_features.values(), axis=1)
        print(f"\n  Total combined features: {combined_features.shape[1]}")

    # Step 5: Evaluate Factors
    print("\n[Step 5] Evaluating Factor Quality...")

    # Use returns as label for evaluation
    future_returns = market_data['$returns'].shift(-1)

    ic_results = []
    for col in combined_features.columns:
        ic = compute_ic(combined_features[col].values, future_returns.reindex(combined_features.index).values)
        rank_ic = compute_rank_ic(combined_features[col].values, future_returns.reindex(combined_features.index).values)
        ic_results.append({
            'factor': col,
            'IC': ic,
            'RankIC': rank_ic
        })

    ic_df = pd.DataFrame(ic_results).sort_values('IC', ascending=False)
    print(f"\n  Evaluated {len(ic_df)} factors")
    print(f"  Mean IC: {ic_df['IC'].mean():.4f}")
    print(f"  Mean RankIC: {ic_df['RankIC'].mean():.4f}")

    # Step 6: Explain Top Factors
    print("\n[Step 6] Explaining Top Factors...")

    top_factor_expr = "RANK(TS_MEAN($roe, 20))"  # Example
    explanation = explainer.explain_factor(top_factor_expr)

    print(f"\n  Factor: {top_factor_expr}")
    print(f"  Name: {explanation.factor_name}")
    print(f"  Economic Logic: {explanation.economic_logic[:100]}...")
    print(f"  Confidence: {explanation.explanation_confidence:.2f}")
    print(f"  Patterns: {', '.join(explanation.used_patterns)}")

    # Step 7: Verify Hypothesis Consistency
    print("\n[Step 7] Verifying Hypothesis Consistency...")

    test_hypotheses = [
        "ROE is positively correlated with stock returns",
        "Low PE stocks outperform high PE stocks",
        "Price momentum predicts future returns",
    ]

    for hypothesis in test_hypotheses:
        result = checker.check(hypothesis)
        print(f"\n  Hypothesis: {hypothesis[:50]}...")
        print(f"    Consistency: {result.consistency_level.value}")
        print(f"    Confidence: {result.confidence:.2f}")

    # Step 8: Summary
    print("\n" + "=" * 70)
    print("Pipeline Summary")
    print("=" * 70)
    print(f"  Concepts in KG: {kg_stats.get('total_concepts', 'N/A')}")
    print(f"  Relations in KG: {kg_stats.get('total_relations', 'N/A')}")
    print(f"  Factors generated: {combined_features.shape[1]}")
    print(f"  Mean IC: {ic_df['IC'].mean():.4f}")
    print(f"  Mean RankIC: {ic_df['RankIC'].mean():.4f}")
    print(f"  Top factor IC: {ic_df['IC'].max():.4f}")

    print("\n" + "=" * 70)
    print("Example 3 completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()