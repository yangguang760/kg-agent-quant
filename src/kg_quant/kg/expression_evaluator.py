#!/usr/bin/env python3
"""
QLIB Expression Evaluator for KG-AgentQuant

Implements QLIB-style expression evaluation using pandas/numpy.
Supports: RANK, CS_RANK, TS_MEAN, TS_STD, TS_DELTA, TS_MAX, TS_MIN, TS_SUM, LOG, etc.
"""

import re
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any


class QLIBExpressionEvaluator:
    """
    Evaluate QLIB-style factor expressions on pandas DataFrame.

    Example:
        >>> evaluator = QLIBExpressionEvaluator()
        >>> data = pd.DataFrame({'$close': [1, 2, 3, 4, 5], '$volume': [100, 200, 300, 400, 500]})
        >>> result = evaluator.evaluate("TS_MEAN($close, 3)", data)
        >>> print(result)
        0    NaN
        1    NaN
        2    2.0
        3    3.0
        4    4.0
        dtype: float64
    """

    # Supported operators
    OPERATORS = {
        # Time series operators
        'TS_MEAN', 'TS_STD', 'TS_STD_DEV', 'TS_MIN', 'TS_MAX',
        'TS_SUM', 'TS_RANK', 'TS_DELTA', 'TS_DELAY',
        # Cross-section operators
        'RANK', 'CS_RANK', 'ZSCORE', 'SCALE',
        # Math operators
        'ABS', 'SIGN', 'LOG', 'LOG1P', 'EXP', 'SQRT', 'POW', 'POWER',
        # Basic math
        'SUM', 'MEAN', 'STD', 'MIN', 'MAX',
        # Logical operators
        'GT', 'LT', 'GE', 'LE', 'EQ', 'NE',
        'AND', 'OR', 'NOT',
        # Conditional
        'IF', 'WHERE',
        # Auxiliary
        'COUNT', 'SUMIF', 'FILTER', 'DELAY',
        # Industry/Sector operators (simplified)
        'SECTOR_MEAN', 'SECTOR_STD', 'SECTOR_RANK',
    }

    def __init__(self):
        self._operator_cache: Dict[str, pd.Series] = {}
        self._batch_data_id: Optional[int] = None
        self._eval_cache: Dict[str, pd.Series] = {}
        self._batch_active = False

    def begin_batch(self, data: pd.DataFrame) -> None:
        """Start a run-local evaluation batch for one immutable data frame."""
        self._batch_active = True
        self._batch_data_id = id(data)
        self._eval_cache = {}

    def end_batch(self) -> None:
        """Clear any run-local cached intermediates/results."""
        self._batch_active = False
        self._batch_data_id = None
        self._eval_cache = {}

    def _get_eval_cache(self, data: pd.DataFrame) -> Optional[Dict[str, pd.Series]]:
        """Return the active run-local cache when it is safe to reuse."""
        if not self._batch_active:
            return None

        data_id = id(data)
        if self._batch_data_id != data_id:
            self._batch_data_id = data_id
            self._eval_cache = {}

        return self._eval_cache

    def evaluate(self, expression: str, data: pd.DataFrame) -> pd.Series:
        """
        Evaluate expression on data.

        Args:
            expression: QLIB-style expression (e.g., "RANK(TS_MEAN($close, 10))")
            data: DataFrame with MultiIndex (instrument, datetime) or single index

        Returns:
            Series with computed factor values
        """
        try:
            cache = self._get_eval_cache(data)
            normalized_expression = expression.strip()
            if cache is not None and normalized_expression in cache:
                return cache[normalized_expression]

            result = self._eval_expr(normalized_expression, data, cache=cache)
            if cache is not None:
                cache[normalized_expression] = result
            return result
        except Exception as e:
            raise ValueError(f"Failed to evaluate expression '{expression}': {e}")

    def _eval_expr(
        self, expr: str, data: pd.DataFrame, cache: Optional[Dict[str, pd.Series]] = None
    ) -> pd.Series:
        """Internal expression evaluation"""
        expr = expr.strip()
        if cache is not None and expr in cache:
            return cache[expr]

        result: Optional[pd.Series] = None

        # Step 0: Handle unary minus at the start (e.g., "-TS_STD($returns, 20)" or "-$close")
        if expr.startswith('-'):
            inner_expr = expr[1:].strip()
            inner_result = self._eval_expr(inner_expr, data, cache=cache)
            result = -inner_result
            if cache is not None:
                cache[expr] = result
            return result

        # Step 1: Try to parse as binary operation FIRST (outside parentheses)
        binary_ops = ['+', '-', '*', '/', '>', '<', '>=', '<=', '==', '!=', '&', '|']
        for op in sorted(binary_ops, key=len, reverse=True):
            pos = self._find_operator(expr, op)
            if pos > 0:  # Binary op must have left operand
                left = expr[:pos].strip()
                right = expr[pos+len(op):].strip()

                if right:  # Right side must exist
                    left_val = self._eval_expr(left, data, cache=cache)
                    right_val = self._eval_expr(right, data, cache=cache)
                    result = self._apply_binary(op, left_val, right_val)
                    break

        # Step 2: Try to parse as function call
        if result is None:
            match = re.match(r'^([A-Za-z_][A-Za-z0-9_]*)\s*\((.*)\)$', expr, re.DOTALL)
            if match:
                func_name = match.group(1)
                args_str = match.group(2)

                args = self._parse_args(args_str, data, cache=cache)
                result = self._call_operator(func_name, args, data)

        # Step 3: Try to parse as parenthesized expression
        if result is None and expr.startswith('(') and expr.endswith(')'):
            result = self._eval_expr(expr[1:-1], data, cache=cache)

        # Step 4: Try to parse as field reference
        if result is None and (expr.startswith('$') or expr in data.columns):
            result = self._get_field(expr, data)

        # Step 5: Try to parse as number
        if result is None:
            try:
                result = pd.Series(float(expr), index=data.index)
            except ValueError:
                pass

        if result is None:
            raise ValueError(f"Cannot parse expression: {expr}")

        if cache is not None:
            cache[expr] = result
        return result

    def _parse_args(
        self, args_str: str, data: pd.DataFrame, cache: Optional[Dict[str, pd.Series]] = None
    ) -> List[Any]:
        """Parse function arguments"""
        args = []
        current = ""
        depth = 0

        for char in args_str:
            if char == '(':
                depth += 1
                current += char
            elif char == ')':
                depth -= 1
                current += char
            elif char == ',' and depth == 0:
                args.append(self._parse_arg(current.strip(), data, cache=cache))
                current = ""
            else:
                current += char

        if current.strip():
            args.append(self._parse_arg(current.strip(), data, cache=cache))

        return args

    def _parse_arg(
        self, arg_str: str, data: pd.DataFrame, cache: Optional[Dict[str, pd.Series]] = None
    ) -> Any:
        """Parse single argument"""
        # Try as number
        try:
            if '.' in arg_str:
                return float(arg_str)
            return int(arg_str)
        except ValueError:
            pass

        # Try as expression
        return self._eval_expr(arg_str, data, cache=cache)

    def _call_operator(self, func_name: str, args: List[Any], data: pd.DataFrame) -> pd.Series:
        """Call operator function"""
        func_name = func_name.upper()

        if func_name == 'CS_RANK':
            return self._cs_rank(args[0])
        elif func_name == 'RANK':
            if len(args) == 1:
                return self._rank(args[0])
            else:
                return self._ts_rank(args[0], int(args[1]))
        elif func_name == 'TS_RANK':
            return self._ts_rank(args[0], int(args[1]))
        elif func_name in ('MEAN', 'TS_MEAN'):
            return self._ts_mean(args[0], int(args[1]))
        elif func_name in ('STD', 'TS_STD', 'TS_STD_DEV'):
            return self._ts_std(args[0], int(args[1]))
        elif func_name in ('DELTA', 'TS_DELTA'):
            return self._ts_delta(args[0], int(args[1]))
        elif func_name in ('DELAY', 'REF', 'TS_DELAY'):
            return self._ts_delay(args[0], int(args[1]))
        elif func_name in ('SUM', 'TS_SUM'):
            return self._ts_sum(args[0], int(args[1]))
        elif func_name in ('MIN', 'TS_MIN'):
            return self._ts_min(args[0], int(args[1]))
        elif func_name in ('MAX', 'TS_MAX'):
            return self._ts_max(args[0], int(args[1]))
        elif func_name == 'LOG':
            return self._log(args[0])
        elif func_name == 'LOG1P':
            return self._log1p(args[0])
        elif func_name == 'ABS':
            return self._abs(args[0])
        elif func_name == 'SIGN':
            return self._sign(args[0])
        elif func_name == 'SQRT':
            return self._sqrt(args[0])
        elif func_name == 'EXP':
            return self._exp(args[0])
        elif func_name in ('POW', 'POWER'):
            return self._pow(args[0], args[1])
        elif func_name == 'COUNT':
            return self._count(args[0], int(args[1]))
        elif func_name == 'GT':
            return self._apply_binary('>', args[0], args[1])
        elif func_name == 'LT':
            return self._apply_binary('<', args[0], args[1])
        elif func_name == 'GE':
            return self._apply_binary('>=', args[0], args[1])
        elif func_name == 'LE':
            return self._apply_binary('<=', args[0], args[1])
        elif func_name == 'EQ':
            return self._apply_binary('==', args[0], args[1])
        elif func_name == 'NE':
            return self._apply_binary('!=', args[0], args[1])
        elif func_name == 'AND':
            return self._apply_binary('and', args[0], args[1])
        elif func_name == 'OR':
            return self._apply_binary('or', args[0], args[1])
        elif func_name == 'IF':
            return self._where(args[0], args[1], args[2])
        elif func_name == 'WHERE':
            return self._where(args[0], args[1], args[2])
        elif func_name == 'ZSCORE':
            return self._zscore(args[0])
        elif func_name == 'SCALE':
            return self._scale(args[0])
        elif func_name == 'SECTOR_MEAN':
            return self._sector_mean(args[0])
        elif func_name == 'SECTOR_STD':
            return self._sector_std(args[0])
        elif func_name == 'SECTOR_RANK':
            return self._sector_rank(args[0])
        else:
            raise ValueError(f"Unknown operator: {func_name}")

    def _get_field(self, field: str, data: pd.DataFrame) -> pd.Series:
        """Get field from data"""
        field = field.strip()

        if field.startswith('$'):
            field = field[1:]

        if field in data.columns:
            return data[field]

        if f'${field}' in data.columns:
            return data[f'${field}']

        raise KeyError(f"Field '{field}' not found in data. Available: {list(data.columns)}")

    # ========================================================================
    # Operator implementations
    # ========================================================================

    def _rank(self, x: pd.Series) -> pd.Series:
        """Cross-sectional rank (percentile)"""
        return x.groupby(level=0 if isinstance(x.index, pd.MultiIndex) else x.index).rank(pct=True)

    def _cs_rank(self, x: pd.Series) -> pd.Series:
        """Cross-sectional rank (percentile within each datetime group)."""
        if isinstance(x.index, pd.MultiIndex):
            return x.groupby(level=1).rank(pct=True)
        else:
            return x.rank(pct=True)

    def _ts_mean(self, x: pd.Series, window: int) -> pd.Series:
        """Time series mean"""
        return x.rolling(window).mean()

    def _ts_std(self, x: pd.Series, window: int) -> pd.Series:
        """Time series standard deviation"""
        return x.rolling(window).std()

    def _ts_min(self, x: pd.Series, window: int) -> pd.Series:
        """Time series minimum"""
        return x.rolling(window).min()

    def _ts_max(self, x: pd.Series, window: int) -> pd.Series:
        """Time series maximum"""
        return x.rolling(window).max()

    def _ts_sum(self, x: pd.Series, window: int) -> pd.Series:
        """Time series sum"""
        return x.rolling(window).sum()

    def _ts_rank(self, x: pd.Series, window: int) -> pd.Series:
        """Time series rank (percentile within window)"""
        def rank_in_window(window_data):
            if len(window_data) < 2 or pd.isna(window_data.iloc[-1]):
                return np.nan
            values = window_data.dropna()
            if len(values) == 0:
                return np.nan
            last_val = values.iloc[-1]
            return (values <= last_val).sum() / len(values)

        return x.rolling(window).apply(rank_in_window, raw=False)

    def _ts_delta(self, x: pd.Series, period: int) -> pd.Series:
        """Time series delta (difference)"""
        return x.diff(period)

    def _ts_delay(self, x: pd.Series, period: int) -> pd.Series:
        """Time series delay (lag)"""
        return x.shift(period)

    def _count(self, x: pd.Series, window: int) -> pd.Series:
        """Count of values in window"""
        if x.dtype == bool:
            x = x.astype(int)
        return x.rolling(window).sum()

    def _log(self, x: pd.Series) -> pd.Series:
        """Natural logarithm"""
        return np.log(x)

    def _log1p(self, x: pd.Series) -> pd.Series:
        """Log(1 + x)"""
        return np.log1p(x)

    def _abs(self, x: pd.Series) -> pd.Series:
        """Absolute value"""
        return np.abs(x)

    def _sign(self, x: pd.Series) -> pd.Series:
        """Sign function"""
        return np.sign(x)

    def _sqrt(self, x: pd.Series) -> pd.Series:
        """Square root"""
        return np.sqrt(x)

    def _exp(self, x: pd.Series) -> pd.Series:
        """Exponential"""
        return np.exp(x)

    def _pow(self, x: pd.Series, y: Union[float, pd.Series]) -> pd.Series:
        """Power function"""
        if isinstance(y, pd.Series):
            return np.power(x, y)
        return np.power(x, float(y))

    def _zscore(self, x: pd.Series) -> pd.Series:
        """Z-score normalization"""
        return (x - x.mean()) / (x.std() + 1e-6)

    def _scale(self, x: pd.Series) -> pd.Series:
        """Scale to [-1, 1]"""
        min_val = x.min()
        max_val = x.max()
        if max_val - min_val < 1e-6:
            return x * 0
        return 2 * (x - min_val) / (max_val - min_val) - 1

    def _where(self, cond: pd.Series, x: pd.Series, y: pd.Series) -> pd.Series:
        """Conditional selection"""
        return x.where(cond, y)

    def _sector_mean(self, x: pd.Series) -> pd.Series:
        """Sector mean (simplified: use market-wide mean)"""
        if isinstance(x.index, pd.MultiIndex):
            return x.groupby(level=1).transform('mean')
        else:
            return pd.Series(x.mean(), index=x.index)

    def _sector_std(self, x: pd.Series) -> pd.Series:
        """Sector std (simplified: use market-wide std)"""
        if isinstance(x.index, pd.MultiIndex):
            return x.groupby(level=1).transform('std')
        else:
            return pd.Series(x.std(), index=x.index)

    def _sector_rank(self, x: pd.Series) -> pd.Series:
        """Sector rank (simplified: use market-wide rank)"""
        return self._rank(x)

    # ========================================================================
    # Binary operations
    # ========================================================================

    def _apply_binary(self, op: str, left: pd.Series, right: pd.Series) -> pd.Series:
        """Apply binary operation"""
        if not isinstance(right, pd.Series):
            right = pd.Series(right, index=left.index)

        if op == '+':
            return left + right
        elif op == '-':
            return left - right
        elif op == '*':
            return left * right
        elif op == '/':
            return left / (right + 1e-6)
        elif op == '>':
            return left > right
        elif op == '<':
            return left < right
        elif op == '>=':
            return left >= right
        elif op == '<=':
            return left <= right
        elif op == '==':
            return left == right
        elif op == '&':
            return left.astype(bool) & right.astype(bool)
        elif op == '|':
            return left.astype(bool) | right.astype(bool)
        elif op == 'and':
            return left.astype(bool) & right.astype(bool)
        elif op == 'or':
            return left.astype(bool) | right.astype(bool)
        else:
            raise ValueError(f"Unknown binary operator: {op}")

    def _find_operator(self, expr: str, op: str) -> int:
        """Find operator position (not inside parentheses)"""
        depth = 0
        i = 0
        while i < len(expr) - len(op) + 1:
            if expr[i] == '(':
                depth += 1
            elif expr[i] == ')':
                depth -= 1
            elif depth == 0 and expr[i:i+len(op)] == op:
                # Special case: operator followed by $ is always an operator (field name)
                if i + len(op) < len(expr) and expr[i+len(op)] == '$':
                    return i
                # Skip if preceded by alphanumeric (not $, which starts field names)
                if i > 0 and (expr[i-1].isalnum() or expr[i-1] == '_'):
                    i += 1
                    continue
                # Skip if followed by alphanumeric (not $, which starts field names)
                if i + len(op) < len(expr) and (expr[i+len(op)].isalnum() or expr[i+len(op)] == '_'):
                    i += 1
                    continue
                return i
            i += 1
        return -1


def evaluate_expression(expression: str, data: pd.DataFrame) -> pd.Series:
    """Evaluate QLIB expression on data"""
    evaluator = QLIBExpressionEvaluator()
    return evaluator.evaluate(expression, data)


__all__ = ['QLIBExpressionEvaluator', 'evaluate_expression']