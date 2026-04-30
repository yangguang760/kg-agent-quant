#!/usr/bin/env python3
"""
Example 2: Factor Evaluation

Demonstrates how to evaluate alpha factors using various metrics.
"""

import sys
sys.path.insert(0, 'src')

import pandas as pd
import numpy as np
from kg_quant.evaluation.metrics import (
    compute_ic, compute_rank_ic, compute_arr, compute_mdd,
    compute_ir, compute_calmar, FactorEvaluator
)


def generate_sample_data(n_stocks=50, n_days=100, seed=42):
    """Generate sample factor and return data for evaluation"""
    np.random.seed(seed)

    dates = pd.date_range(start="2023-01-01", periods=n_days, freq='B')
    stocks = [f"SH{str(i).zfill(6)}" for i in range(n_stocks)]

    # Factor values (some predictive, some random)
    factor_base = np.random.randn(n_days) * 0.02
    factor_values = np.zeros((n_days, n_stocks))
    for i in range(n_days):
        # Factor has some predictive power with noise
        factor_values[i, :] = factor_base[i] + np.random.randn(n_stocks) * 0.1

    # Returns (correlated with factor)
    returns = factor_values * 0.5 + np.random.randn(n_days, n_stocks) * 0.02

    # Create DataFrames with MultiIndex
    factor_df = pd.DataFrame(
        factor_values,
        index=dates,
        columns=stocks
    ).T

    returns_df = pd.DataFrame(
        returns,
        index=dates,
        columns=stocks
    ).T

    return factor_df, returns_df


def main():
    print("=" * 60)
    print("KG-AgentQuant Example 2: Factor Evaluation")
    print("=" * 60)

    # Generate sample data
    print("\n[1] Generating sample data...")
    factor_df, returns_df = generate_sample_data(n_stocks=50, n_days=100)
    print(f"  Factor shape: {factor_df.shape}")
    print(f"  Returns shape: {returns_df.shape}")

    # Compute point-in-time metrics
    print("\n[2] Computing Factor Metrics...")

    ic_list = []
    rank_ic_list = []

    for date in factor_df.columns:
        fv = factor_df[date].dropna()
        rv = returns_df[date].dropna()

        common = fv.index.intersection(rv.index)
        if len(common) >= 3:
            ic = compute_ic(fv[common], rv[common])
            rank_ic = compute_rank_ic(fv[common], rv[common])
            if not np.isnan(ic):
                ic_list.append(ic)
            if not np.isnan(rank_ic):
                rank_ic_list.append(rank_ic)

    if ic_list:
        print(f"  Mean IC: {np.mean(ic_list):.4f}")
        print(f"  IC Std: {np.std(ic_list):.4f}")
        print(f"  ICIR: {np.mean(ic_list) / np.std(ic_list) if np.std(ic_list) > 0 else 0:.4f}")
        print(f"  Mean RankIC: {np.mean(rank_ic_list):.4f}")
        print(f"  RankICIR: {np.mean(rank_ic_list) / np.std(rank_ic_list) if np.std(rank_ic_list) > 0 else 0:.4f}")

    # Create sample portfolio returns
    print("\n[3] Computing Portfolio Metrics...")
    daily_returns = returns_df.mean(axis=0).values

    arr = compute_arr(daily_returns)
    mdd = compute_mdd(daily_returns)
    ir = compute_ir(daily_returns)
    calmar = compute_calmar(daily_returns)

    print(f"  Annualized Return (ARR): {arr:.4f} ({arr*100:.2f}%)")
    print(f"  Maximum Drawdown (MDD): {mdd:.4f} ({mdd*100:.2f}%)")
    print(f"  Information Ratio (IR): {ir:.4f}")
    print(f"  Calmar Ratio: {calmar:.4f}")

    # Use FactorEvaluator class
    print("\n[4] Using FactorEvaluator Class...")

    # Create proper MultiIndex data for FactorEvaluator
    dates = pd.date_range(start="2023-01-01", periods=100, freq='B')
    stocks = [f"SH{str(i).zfill(6)}" for i in range(50)]

    np.random.seed(42)

    # Create data with MultiIndex (datetime, instrument) like QLib expects
    all_data = []
    for date in dates:
        for stock in stocks:
            all_data.append({
                'datetime': date,
                'instrument': stock,
                'factor': np.random.randn(),
                'return': np.random.randn() * 0.02
            })

    df = pd.DataFrame(all_data)
    df = df.set_index(['datetime', 'instrument'])

    # The evaluator expects MultiIndex (datetime, instrument)
    factor_values = df['factor'].unstack(level='instrument')
    returns = df['return'].unstack(level='instrument')

    # Stack back to get proper format
    factor_values = factor_values.stack().swaplevel().sort_index()
    returns = returns.stack().swaplevel().sort_index()

    evaluator = FactorEvaluator(annualization_factor=252)

    print("  Testing FactorEvaluator...")
    metrics = evaluator.evaluate_factor(factor_values, returns)
    print(f"  IC Mean: {metrics['ic_mean']:.4f}")
    print(f"  RankIC Mean: {metrics['rank_ic_mean']:.4f}")

    # Visualize IC time series
    print("\n[5] IC Time Series Summary...")
    print(f"  Number of days: {len(ic_list)}")
    print(f"  Positive IC days: {sum(1 for x in ic_list if x > 0)}")
    print(f"  Negative IC days: {sum(1 for x in ic_list if x < 0)}")
    print(f"  Hit Rate: {sum(1 for x in ic_list if x > 0) / len(ic_list) * 100:.1f}%")

    print("\n" + "=" * 60)
    print("Example 2 completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()