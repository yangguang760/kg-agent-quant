"""
Evaluation Metrics for KG-AgentQuant

Provides comprehensive evaluation metrics for factor research:
- IC (Information Coefficient)
- RankIC (Spearman correlation)
- IR (Information Ratio)
- ARR (Annualized Return)
- MDD (Maximum Drawdown)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union
from scipy.stats import spearmanr


def compute_ic(
    factor_values: Union[np.ndarray, pd.Series],
    future_returns: Union[np.ndarray, pd.Series],
    method: str = 'pearson'
) -> float:
    """
    Compute Information Coefficient (IC).

    Args:
        factor_values: Factor prediction values
        future_returns: Actual future returns
        method: 'pearson' or 'spearman'

    Returns:
        IC value in range [-1, 1]
    """
    factor_arr = np.asarray(factor_values).flatten()
    returns_arr = np.asarray(future_returns).flatten()

    min_len = min(len(factor_arr), len(returns_arr))
    if min_len < 2:
        return 0.0

    factor_arr = factor_arr[:min_len]
    returns_arr = returns_arr[:min_len]

    mask = ~(np.isnan(factor_arr) | np.isnan(returns_arr))
    factor_arr = factor_arr[mask]
    returns_arr = returns_arr[mask]

    if len(factor_arr) < 2:
        return 0.0

    if method == 'pearson':
        corr_matrix = np.corrcoef(factor_arr, returns_arr)
        if corr_matrix.shape[0] < 2:
            return 0.0
        ic = corr_matrix[0, 1]
    elif method == 'spearman':
        ic, _ = spearmanr(factor_arr, returns_arr)
    else:
        raise ValueError(f"Unknown method: {method}")

    return 0.0 if np.isnan(ic) else ic


def compute_rank_ic(factor_values: Union[np.ndarray, pd.Series], future_returns: Union[np.ndarray, pd.Series]) -> float:
    """Compute Rank IC (Spearman correlation)"""
    return compute_ic(factor_values, future_returns, method='spearman')


def compute_icir(ic_series: Union[np.ndarray, pd.Series, List[float]]) -> float:
    """
    Compute IC Information Ratio (ICIR).

    Args:
        ic_series: Time series of IC values

    Returns:
        ICIR = mean(IC) / std(IC)
    """
    ic_array = np.array(ic_series)
    ic_array = ic_array[~np.isnan(ic_array)]

    if len(ic_array) < 2 or np.std(ic_array) == 0:
        return 0.0

    return float(np.mean(ic_array) / np.std(ic_array))


def compute_arr(
    returns: Union[np.ndarray, pd.Series, List[float]],
    annualization_factor: int = 252
) -> float:
    """
    Compute Annualized Rate of Return (ARR).

    Args:
        returns: List or array of period returns
        annualization_factor: Number of periods per year (default 252 for daily)

    Returns:
        Annualized return rate
    """
    returns_array = np.array(returns)
    returns_array = returns_array[~np.isnan(returns_array)]

    if len(returns_array) == 0:
        return 0.0

    cumulative_return = np.prod(1 + returns_array) - 1
    n_periods = len(returns_array)

    if n_periods == 0:
        return 0.0

    return float((1 + cumulative_return) ** (annualization_factor / n_periods) - 1)


def compute_mdd(returns: Union[np.ndarray, pd.Series, List[float]]) -> float:
    """
    Compute Maximum Drawdown (MDD).

    Args:
        returns: List or array of period returns

    Returns:
        Maximum drawdown as a positive value
    """
    returns_array = np.array(returns)
    returns_array = returns_array[~np.isnan(returns_array)]

    if len(returns_array) == 0:
        return 0.0

    cumulative = np.cumprod(1 + returns_array)
    running_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - running_max) / running_max
    return float(-np.min(drawdown))


def compute_ir(returns: Union[np.ndarray, pd.Series, List[float]], annualization_factor: int = 252) -> float:
    """
    Compute Information Ratio (IR).

    Args:
        returns: Period returns
        annualization_factor: Number of periods per year

    Returns:
        IR = mean(returns) / std(returns) * sqrt(annualization_factor)
    """
    returns_array = np.array(returns)
    returns_array = returns_array[~np.isnan(returns_array)]

    if len(returns_array) == 0 or np.std(returns_array) == 0:
        return 0.0

    mean_ret = np.mean(returns_array)
    std_ret = np.std(returns_array)
    return float(mean_ret / std_ret * np.sqrt(annualization_factor))


def compute_calmar(returns: Union[np.ndarray, pd.Series, List[float]], annualization_factor: int = 252) -> float:
    """
    Compute Calmar Ratio.

    Args:
        returns: Period returns
        annualization_factor: Number of periods per year

    Returns:
        Calmar = ARR / MDD
    """
    arr = compute_arr(returns, annualization_factor)
    mdd = compute_mdd(returns)
    return float(arr / mdd) if mdd != 0 else 0.0


class FactorEvaluator:
    """Factor evaluator with comprehensive metrics"""

    def __init__(self, annualization_factor: int = 252):
        """
        Initialize factor evaluator.

        Args:
            annualization_factor: Days per year (252 for trading days)
        """
        self.annualization_factor = annualization_factor

    def evaluate_factor(
        self,
        factor_values: pd.DataFrame,
        future_returns: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Evaluate factor with multiple metrics.

        Args:
            factor_values: DataFrame with MultiIndex (instrument, datetime)
            future_returns: Future returns with same index structure

        Returns:
            Dictionary of metrics
        """
        ic_list = []
        rank_ic_list = []

        common_dates = factor_values.index.get_level_values('datetime').intersection(
            future_returns.index.get_level_values('datetime')
        )

        for date in common_dates:
            fv = factor_values.xs(date, level='datetime').dropna()
            fr = future_returns.xs(date, level='datetime').dropna()

            common_instruments = fv.index.intersection(fr.index)
            if len(common_instruments) < 3:
                continue

            ic = compute_ic(fv.loc[common_instruments].values, fr.loc[common_instruments].values)
            rank_ic = compute_rank_ic(fv.loc[common_instruments].values, fr.loc[common_instruments].values)

            if not np.isnan(ic):
                ic_list.append(ic)
            if not np.isnan(rank_ic):
                rank_ic_list.append(rank_ic)

        return {
            'ic_mean': float(np.mean(ic_list)) if ic_list else 0.0,
            'ic_std': float(np.std(ic_list)) if ic_list else 0.0,
            'icir': compute_icir(ic_list),
            'rank_ic_mean': float(np.mean(rank_ic_list)) if rank_ic_list else 0.0,
            'rank_ic_std': float(np.std(rank_ic_list)) if rank_ic_list else 0.0,
            'rank_icir': compute_icir(rank_ic_list),
            'n_days': len(ic_list),
        }

    def evaluate_strategy(self, portfolio_returns: pd.Series) -> Dict[str, float]:
        """
        Evaluate strategy performance.

        Args:
            portfolio_returns: Series of portfolio returns

        Returns:
            Dictionary of performance metrics
        """
        arr = compute_arr(portfolio_returns, self.annualization_factor)
        mdd = compute_mdd(portfolio_returns)
        ir = compute_ir(portfolio_returns, self.annualization_factor)

        return {
            'arr': arr,
            'mdd': mdd,
            'ir': ir,
            'calmar_ratio': compute_calmar(portfolio_returns, self.annualization_factor),
            'win_rate': float((portfolio_returns > 0).mean()) if len(portfolio_returns) > 0 else 0.0,
            'mean_return': float(portfolio_returns.mean()) if len(portfolio_returns) > 0 else 0.0,
            'volatility': float(portfolio_returns.std()) if len(portfolio_returns) > 0 else 0.0,
        }

    def evaluate(
        self,
        factor_values: Optional[pd.DataFrame] = None,
        future_returns: Optional[pd.DataFrame] = None,
        portfolio_returns: Optional[pd.Series] = None
    ) -> Dict[str, float]:
        """
        Comprehensive evaluation combining factor and strategy metrics.

        Args:
            factor_values: Factor predictions
            future_returns: Future returns
            portfolio_returns: Portfolio returns

        Returns:
            Combined metrics dictionary
        """
        metrics = {}

        if factor_values is not None and future_returns is not None:
            metrics.update(self.evaluate_factor(factor_values, future_returns))

        if portfolio_returns is not None:
            metrics.update(self.evaluate_strategy(portfolio_returns))

        return metrics


__all__ = [
    'compute_ic',
    'compute_rank_ic',
    'compute_icir',
    'compute_arr',
    'compute_mdd',
    'compute_ir',
    'compute_calmar',
    'FactorEvaluator',
]