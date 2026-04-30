"""
AST-based Factor Parser for KG-AgentQuant

Implements factor expression parsing and validation using Python's AST module.
"""

import ast
import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class FactorConstraint:
    """Factor complexity constraints"""
    max_symbol_length: int = 250
    max_base_features: int = 6
    max_free_args_ratio: float = 0.5


class FactorASTParser:
    """
    Parse and validate factor expressions using Python's AST module.

    This parser:
    1. Parses QLIB-style expressions (e.g., "RANK(TS_MEAN($close, 10))")
    2. Validates against complexity constraints
    3. Computes complexity scores

    Example:
        >>> parser = FactorASTParser()
        >>> tree = parser.parse_expression("RANK(TS_MEAN($close, 10))")
        >>> valid, analysis = parser.validate_constraints(tree)
        >>> print(f"Valid: {valid}, Complexity: {parser.compute_complexity(tree):.2f}")
    """

    # Supported operators
    OPERATORS = {
        # Time series operators
        'DELTA', 'DELAY', 'TS_MEAN', 'TS_STD', 'TS_MIN', 'TS_MAX',
        'TS_SUM', 'TS_RANK',
        # Cross-section operators
        'RANK', 'CS_RANK', 'ZSCORE', 'SCALE',
        # Math operators
        'ABS', 'SIGN', 'LOG', 'SQRT', 'POW',
        # Technical indicators
        'SMA', 'EMA', 'WMA', 'MACD', 'RSI',
        # Logical operators
        'GT', 'LT', 'GE', 'LE', 'AND', 'OR', 'NOT',
        # Auxiliary
        'COUNT', 'SUMIF', 'FILTER',
        # Industry/Sector operators
        'SECTOR_MEAN', 'SECTOR_STD', 'SECTOR_RANK',
    }

    # Base features
    BASE_FEATURES = {'$open', '$high', '$low', '$close', '$volume', '$vwap', '$amount'}

    def __init__(self, constraints: Optional[FactorConstraint] = None):
        """
        Initialize the parser.

        Args:
            constraints: Optional complexity constraints
        """
        self.constraints = constraints or FactorConstraint()

    def parse_expression(self, expr: str) -> ast.AST:
        """
        Parse factor expression string to AST.

        Args:
            expr: Factor expression (e.g., "RANK(TS_MEAN($close, 10))")

        Returns:
            AST tree

        Raises:
            ValueError: If expression is invalid
        """
        sanitized = self._sanitize_expression(expr)
        try:
            tree = ast.parse(sanitized, mode='eval')
            return tree
        except SyntaxError as e:
            raise ValueError(f"Invalid factor expression: {expr}\nError: {e}")

    def _sanitize_expression(self, expr: str) -> str:
        """Sanitize expression for AST parsing by replacing $field with f_field."""
        return re.sub(r'\$(\w+)', r'f_\1', expr)

    def validate_constraints(self, tree: ast.AST) -> Tuple[bool, Dict]:
        """
        Validate factor against complexity constraints.

        Args:
            tree: AST tree from parse_expression

        Returns:
            Tuple of (is_valid, analysis_dict)
        """
        result = {
            'symbol_length': 0,
            'base_features': 0,
            'free_args': 0,
            'operators': [],
            'valid': True,
            'errors': []
        }

        # Count symbol length
        try:
            result['symbol_length'] = len(ast.unparse(tree))
        except Exception:
            result['symbol_length'] = len(str(tree))

        if result['symbol_length'] > self.constraints.max_symbol_length:
            result['valid'] = False
            result['errors'].append(
                f"Symbol length {result['symbol_length']} > {self.constraints.max_symbol_length}"
            )

        # Analyze AST
        self._analyze_ast(tree, result)

        # Check base features
        if result['base_features'] > self.constraints.max_base_features:
            result['valid'] = False
            result['errors'].append(
                f"Base features {result['base_features']} > {self.constraints.max_base_features}"
            )

        return result['valid'], result

    def _analyze_ast(self, tree: ast.AST, result: Dict) -> None:
        """Analyze AST to extract features and operators."""
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                name = node.id
                if name.startswith('f_'):
                    result['base_features'] += 1
                elif name.upper() in self.OPERATORS:
                    result['operators'].append(name.upper())
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id.upper() in self.OPERATORS:
                        result['free_args'] += len(node.args)

    def compute_complexity(self, tree: ast.AST) -> float:
        """
        Compute factor complexity score.

        Formula: C(f) = α1·SL(f) + α2·PC(f) + α3·log(1+|Ff|)

        Args:
            tree: AST tree

        Returns:
            Complexity score
        """
        _, analysis = self.validate_constraints(tree)

        alpha1, alpha2, alpha3 = 0.1, 0.2, 0.3
        complexity = (
            alpha1 * analysis['symbol_length'] +
            alpha2 * analysis['free_args'] +
            alpha3 * (1 + analysis['base_features'])
        )
        return complexity

    def generate_expression(self, tree: ast.AST) -> str:
        """
        Generate factor expression string from AST.

        Args:
            tree: AST tree

        Returns:
            Factor expression string
        """
        try:
            expr = ast.unparse(tree)
        except Exception:
            expr = str(tree)
        # Restore $field names
        return re.sub(r'f_(\w+)', r'$\1', expr)


def validate_factor_expression(expression: str) -> Tuple[bool, str]:
    """
    Validate a factor expression and return status.

    Args:
        expression: Factor expression to validate

    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        parser = FactorASTParser()
        tree = parser.parse_expression(expression)
        valid, analysis = parser.validate_constraints(tree)
        if valid:
            return True, "Valid"
        else:
            return False, "; ".join(analysis['errors'])
    except Exception as e:
        return False, str(e)


__all__ = ['FactorASTParser', 'FactorConstraint', 'validate_factor_expression']