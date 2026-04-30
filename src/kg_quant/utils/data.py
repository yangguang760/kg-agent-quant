"""
Data Utilities

Provides data generation and loading utilities.
"""

import numpy as np
import pandas as pd
from typing import Optional, List


def generate_sample_data(
    n_stocks: int = 50,
    n_days: int = 100,
    start_date: str = "2023-01-01",
    seed: int = 42,
    include_fundamentals: bool = True
) -> pd.DataFrame:
    """
    Generate sample market data for testing and demonstration.

    Args:
        n_stocks: Number of stocks
        n_days: Number of trading days
        start_date: Start date
        seed: Random seed
        include_fundamentals: Include fundamental data fields

    Returns:
        DataFrame with MultiIndex (instrument, datetime)
    """
    np.random.seed(seed)

    dates = pd.date_range(start=start_date, periods=n_days, freq='B')
    stocks = [f"SH{str(i).zfill(6)}" for i in range(n_stocks)]

    n = len(dates) * len(stocks)

    # Base prices
    base_prices = 100 + np.random.rand(n_stocks) * 100

    # Generate price series
    all_data = {}

    for i, stock in enumerate(stocks):
        start_price = base_prices[i]
        returns = np.random.randn(n_days) * 0.02

        # Add some momentum
        for t in range(1, n_days):
            returns[t] += 0.01 * returns[t-1]

        prices = start_price * np.cumprod(np.concatenate([[1], 1 + returns]))

        for j, date in enumerate(dates):
            idx = i * n_days + j
            all_data[idx] = {
                'datetime': date,
                'instrument': stock,
                '$close': prices[j],
                '$open': prices[j] * (1 + np.random.randn() * 0.005),
                '$high': max(prices[j], prices[j] * (1 + abs(np.random.randn()) * 0.01)),
                '$low': min(prices[j], prices[j] * (1 - abs(np.random.randn()) * 0.01)),
                '$volume': np.random.rand() * 1e7,
                '$vwap': prices[j] * (1 + np.random.randn() * 0.002),
                '$returns': returns[j] if j > 0 else 0,
            }

    df = pd.DataFrame.from_dict(all_data, orient='index')

    # Add fundamental data if requested
    if include_fundamentals:
        # ROE - varies per stock but stable over time
        roe_base = np.random.rand(n_stocks) * 0.3
        roe_noise = np.random.randn(n) * 0.02

        # PE - varies per stock
        pe_base = np.random.rand(n_stocks) * 40 + 10
        pe_noise = np.random.randn(n) * 2

        # PB - varies per stock
        pb_base = np.random.rand(n_stocks) * 5 + 0.5
        pb_noise = np.random.randn(n) * 0.3

        df['$roe'] = 0
        df['$pe'] = 0
        df['$pb'] = 0

        for i, stock in enumerate(stocks):
            mask = df['instrument'] == stock
            df.loc[mask, '$roe'] = np.clip(roe_base[i] + roe_noise[mask.values], 0, 1)
            df.loc[mask, '$pe'] = np.clip(pe_base[i] + pe_noise[mask.values], 1, 100)
            df.loc[mask, '$pb'] = np.clip(pb_base[i] + pb_noise[mask.values], 0.1, 20)

    df = df.set_index(['datetime', 'instrument'])
    return df


def load_qlib_data(
    qlib_data_dir: str,
    instruments: str = "csi300",
    fields: Optional[List[str]] = None,
    start_time: str = "2023-01-01",
    end_time: str = "2023-12-31"
) -> pd.DataFrame:
    """
    Load data from QLib.

    Args:
        qlib_data_dir: QLib data directory
        instruments: Market or instrument list
        fields: List of fields to load
        start_time: Start time
        end_time: End time

    Returns:
        DataFrame with loaded data
    """
    try:
        import qlib
        from qlib.data import D

        qlib.init(provider_uri=qlib_data_dir, region="cn")

        if fields is None:
            fields = ["$close", "$open", "$high", "$low", "$volume"]

        data = D.features(
            instruments=instruments,
            fields=fields,
            start_time=start_time,
            end_time=end_time,
            freq="day"
        )

        if hasattr(data, 'to_dataframe'):
            return data.to_dataframe()
        return data

    except ImportError:
        raise ImportError("Qlib is not installed. Install with: pip install pyqlib")
    except Exception as e:
        raise RuntimeError(f"Failed to load QLib data: {e}")


__all__ = ['generate_sample_data', 'load_qlib_data']