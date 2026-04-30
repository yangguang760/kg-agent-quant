"""
Core Evaluation Module

Unified evaluator that provides consistent metrics across all experiments.
"""

from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
from scipy.stats import spearmanr


class Evaluator:
    """
    Unified evaluator for all KG-AgentQuant experiments.

    Provides consistent metrics including:
    - IC, RankIC: Factor prediction quality
    - ICIR, RankICIR: Stability of prediction quality
    - ARR, IR: Portfolio performance
    - MDD: Risk metric
    """

    METRICS = ['IC', 'ICIR', 'RankIC', 'RankICIR', 'ARR', 'IR', 'MDD', 'CR']

    def __init__(self, eval_config: Optional[Dict] = None):
        """
        Initialize evaluator.

        Args:
            eval_config: Optional configuration dict
        """
        self.config = eval_config or {}
        self.commission = self.config.get("commission", 0.001)

    def evaluate(self, results: Dict) -> Dict[str, float]:
        """
        Evaluate experiment results.

        Args:
            results: Dict containing predictions, labels, positions, etc.

        Returns:
            Dictionary of computed metrics
        """
        metrics = {}

        predictions = results.get('predictions')
        labels = results.get('labels')
        positions = results.get('positions')

        if predictions is not None and labels is not None:
            pred_metrics = self._evaluate_factor_predictions(predictions, labels)
            metrics.update(pred_metrics)

        if positions is not None and labels is not None:
            strategy_metrics = self._evaluate_strategy(positions, labels)
            metrics.update(strategy_metrics)

        if 'backtest_returns' in results:
            backtest_metrics = self._evaluate_backtest(results['backtest_returns'])
            metrics.update(backtest_metrics)

        return metrics

    def _evaluate_factor_predictions(
        self,
        predictions: pd.DataFrame,
        labels: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Evaluate factor prediction quality using IC and RankIC.

        Args:
            predictions: Factor prediction DataFrame
            labels: Ground truth labels DataFrame

        Returns:
            Dictionary with IC, RankIC, ICIR, RankICIR
        """
        ic_list = []
        rank_ic_list = []

        common_dates = predictions.index.intersection(labels.index)

        for date in common_dates:
            pred = predictions.loc[date].dropna()
            label = labels.loc[date].dropna()

            common = pred.index.intersection(label.index)
            if len(common) < 3:
                continue

            ic, _ = self._pearson_corr(pred[common].values, label[common].values)
            if not np.isnan(ic):
                ic_list.append(ic)

            rank_ic, _ = spearmanr(pred[common].values, label[common].values)
            if not np.isnan(rank_ic):
                rank_ic_list.append(rank_ic)

        if not ic_list:
            return {m: 0.0 for m in ['IC', 'ICIR', 'RankIC', 'RankICIR']}

        ic_arr = np.array(ic_list)
        rank_ic_arr = np.array(rank_ic_list)

        return {
            'IC': float(np.mean(ic_arr)),
            'ICIR': float(np.mean(ic_arr) / np.std(ic_arr)) if np.std(ic_arr) > 0 else 0.0,
            'RankIC': float(np.mean(rank_ic_arr)),
            'RankICIR': float(np.mean(rank_ic_arr) / np.std(rank_ic_arr)) if np.std(rank_ic_arr) > 0 else 0.0,
        }

    def _pearson_corr(self, x: np.ndarray, y: np.ndarray):
        """Compute Pearson correlation coefficient."""
        if len(x) < 3:
            return np.nan, np.nan
        return np.corrcoef(x, y)[0, 1], None

    def _evaluate_strategy(
        self,
        positions: pd.DataFrame,
        labels: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Evaluate strategy performance using portfolio returns.

        Args:
            positions: Portfolio positions DataFrame
            labels: Returns labels DataFrame

        Returns:
            Dictionary with ARR, IR, MDD, CR
        """
        strategy_returns = self._calculate_portfolio_returns(positions, labels)

        if len(strategy_returns) == 0:
            return {m: 0.0 for m in ['ARR', 'IR', 'MDD', 'CR']}

        cumulative = (1 + strategy_returns).prod()
        n_days = len(strategy_returns)
        arr = cumulative ** (252 / n_days) - 1

        mean_ret = strategy_returns.mean()
        std_ret = strategy_returns.std()
        ir = mean_ret / std_ret * np.sqrt(252) if std_ret > 0 else 0.0

        cumprod = (1 + strategy_returns).cumprod()
        running_max = cumprod.cummax()
        drawdown = (cumprod - running_max) / running_max
        mdd = drawdown.min()

        cr = arr / abs(mdd) if mdd != 0 else 0.0

        return {
            'ARR': float(arr),
            'IR': float(ir),
            'MDD': float(mdd),
            'CR': float(cr),
        }

    def _calculate_portfolio_returns(
        self,
        positions: pd.DataFrame,
        labels: pd.DataFrame
    ) -> pd.Series:
        """Calculate portfolio returns from positions and labels."""
        returns_list = []
        dates = positions.index.intersection(labels.index)

        for date in dates:
            pos = positions.loc[date].dropna()
            ret = labels.loc[date].dropna()

            common = pos.index.intersection(ret.index)
            if len(common) == 0:
                continue

            portfolio_ret = (pos[common] * ret[common]).sum()
            returns_list.append({'date': date, 'return': portfolio_ret})

        if returns_list:
            return pd.DataFrame(returns_list).set_index('date')['return']
        return pd.Series()

    def _evaluate_backtest(self, returns: pd.Series) -> Dict[str, float]:
        """Evaluate backtest results."""
        return {
            'total_return': float((1 + returns).prod() - 1),
            'win_rate': float((returns > 0).mean()) if len(returns) > 0 else 0.0,
            'avg_return': float(returns.mean()) if len(returns) > 0 else 0.0,
            'volatility': float(returns.std()) if len(returns) > 0 else 0.0,
        }


__all__ = ['Evaluator']