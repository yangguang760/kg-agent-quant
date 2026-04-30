#!/usr/bin/env python3
"""
KG Feature Generator

Generates knowledge graph enhanced alpha factors from validated factor expressions.
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import numpy as np

from .retriever import KGRetriever
from .expression_evaluator import QLIBExpressionEvaluator


# Category to factor type mapping
_CAT_TO_TYPE: Dict[str, str] = {
    "估值指标": "value",
    "盈利能力指标": "quality",
    "偿债能力指标": "quality",
    "营运能力指标": "quality",
    "成长能力指标": "momentum",
    "其他重要指标": "size",
}

_TYPE_TO_CATS: Dict[str, List[str]] = {
    "value": ["value", "估值指标"],
    "quality": ["quality", "盈利能力指标", "偿债能力指标", "营运能力指标"],
    "size": ["size", "其他重要指标"],
    "momentum": ["momentum", "成长能力指标"],
}


class JSONFactorLoader:
    """
    Load factors from JSON file, filter by factor type.
    """

    def __init__(self, json_path: Optional[str] = None):
        """
        Initialize factor loader.

        Args:
            json_path: Path to factor JSON file
        """
        if json_path:
            self.json_path = Path(json_path)
        else:
            # Use sample data as default
            self.json_path = Path(__file__).parent.parent.parent.parent / "data" / "sample" / "factors_sample.json"

        self._factors: List[Dict] = []
        self._load()

    def _load(self) -> None:
        """Load factors from JSON file"""
        if not self.json_path.exists():
            print(f"[JSONFactorLoader] File not found: {self.json_path}, using empty loader")
            self._factors = []
            return

        with open(self.json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        factors = data.get("factors", [])
        if isinstance(factors, dict):
            self._factors = [
                {"expression": expr, "metadata": meta, "result": {"expression": expr}}
                for expr, meta in factors.items()
            ]
        else:
            self._factors = factors

        print(f"[JSONFactorLoader] Loaded {len(self._factors)} factors from {self.json_path.name}")

    def get_factors_by_type(self, factor_type: str, n_features: int, seed: int = 42) -> List[Tuple[int, str, Dict]]:
        """
        Get factors by type.

        Args:
            factor_type: "value" | "quality" | "size" | "momentum"
            n_features: Maximum number to return
            seed: Random seed

        Returns:
            List of (index, expression, metadata) tuples
        """
        cats = _TYPE_TO_CATS.get(factor_type, [])

        matched = []
        for f in self._factors:
            cat = f.get("metadata", {}).get("category", "unknown")
            if cat in cats:
                expr = f.get("result", {}).get("expression", f.get("expression", ""))
                if expr:
                    matched.append((f.get("metadata", {}).get("index", 0), expr, f))

        if not matched:
            print(f"[JSONFactorLoader] No factors for type '{factor_type}'")
            return []

        # Limit to n_features
        selected = matched[:min(n_features, len(matched))]
        print(f"[JSONFactorLoader] Selected {len(selected)} factors for '{factor_type}'")
        return selected

    def get_all_expressions(self) -> List[Tuple[int, str, Dict]]:
        """Get all factor expressions"""
        return [
            (i, f.get("result", {}).get("expression", f.get("expression", "")), f)
            for i, f in enumerate(self._factors)
            if f.get("result", {}).get("expression")
        ]

    @property
    def total_count(self) -> int:
        """Total number of factors"""
        return len(self._factors)


class KGFeatureGenerator:
    """
    KG Feature Generator

    Generates KG-enhanced alpha factors from validated expressions.
    """

    def __init__(
        self,
        kg_dir: str = "data/kg",
        factor_json_path: Optional[str] = None,
    ):
        """
        Initialize feature generator.

        Args:
            kg_dir: Knowledge graph directory
            factor_json_path: Path to factor JSON file
        """
        self.kg_dir = Path(kg_dir)
        self.retriever = KGRetriever(kg_dir) if kg_dir and Path(kg_dir).exists() else None
        self.evaluator = QLIBExpressionEvaluator()
        self.factor_loader = JSONFactorLoader(factor_json_path)

        # Feature cache
        self._feature_cache: Dict[str, pd.DataFrame] = {}

    def resolve_valid_factors(
        self,
        factor_type: str,
        n_features: Optional[int] = None,
        seed: int = 42,
    ) -> List[Tuple[int, str, Dict]]:
        """
        Resolve valid factors for a given type.

        Args:
            factor_type: Factor type
            n_features: Max features to return
            seed: Random seed

        Returns:
            List of valid (index, expression, metadata) tuples
        """
        loader_n = n_features if n_features is not None else 1000
        json_factors = self.factor_loader.get_factors_by_type(factor_type, loader_n, seed)
        valid_expressions: List[Tuple[int, str, Dict]] = []

        for idx, expr, meta in json_factors:
            if expr and isinstance(expr, str):
                valid_expressions.append((idx, expr, meta))

        return valid_expressions

    def generate_kg_features(
        self,
        factor_type: str,
        n_features: int = 10,
        data: Optional[pd.DataFrame] = None,
        seed: int = 42,
    ) -> pd.DataFrame:
        """
        Generate KG features.

        Args:
            factor_type: Factor type to generate
            n_features: Number of features to generate
            data: DataFrame with price/volume data
            seed: Random seed

        Returns:
            DataFrame with generated features
        """
        if data is None:
            # Generate sample data if not provided
            data = self._generate_sample_data()

        # Get valid expressions
        valid_expressions = self.resolve_valid_factors(factor_type, n_features, seed)

        if not valid_expressions:
            print(f"[KGFeatureGenerator] No valid factors for type '{factor_type}'")
            return pd.DataFrame(index=data.index)

        # Compute features
        feature_dict: Dict[str, pd.Series] = {}
        self.evaluator.begin_batch(data)

        try:
            for i, (idx, expr, meta) in enumerate(valid_expressions):
                col_name = f"kg_{factor_type}_{i:03d}"
                try:
                    feature_values = self.evaluator.evaluate(expr, data)
                    feature_dict[col_name] = feature_values
                except Exception as e:
                    print(f"[KGFeatureGenerator] Error evaluating '{expr[:40]}...': {e}")

        finally:
            self.evaluator.end_batch()

        if feature_dict:
            return pd.DataFrame(feature_dict)
        return pd.DataFrame(index=data.index)

    def _generate_sample_data(
        self,
        n_stocks: int = 10,
        n_days: int = 100,
        start_date: str = "2023-01-01"
    ) -> pd.DataFrame:
        """Generate sample market data for testing"""
        dates = pd.date_range(start=start_date, periods=n_days, freq='B')
        stocks = [f"STOCK_{i:04d}" for i in range(n_stocks)]

        np.random.seed(42)
        n = len(dates) * len(stocks)

        data = pd.DataFrame({
            '$close': np.random.randn(n) * 10 + 100,
            '$open': np.random.randn(n) * 10 + 100,
            '$high': np.random.randn(n) * 10 + 105,
            '$low': np.random.randn(n) * 10 + 95,
            '$volume': np.random.rand(n) * 1e6,
            '$roe': np.random.randn(n) * 0.1 + 0.15,
            '$roa': np.random.randn(n) * 0.05 + 0.08,
            '$pe': np.random.rand(n) * 20 + 10,
            '$pb': np.random.rand(n) * 3 + 0.5,
        }, index=pd.MultiIndex.from_product(
            [dates, stocks],
            names=['datetime', 'instrument']
        ))

        return data

    def get_feature_metadata(self) -> Dict:
        """Get feature generation metadata"""
        return {
            "kg_dir": str(self.kg_dir) if self.kg_dir else None,
            "factor_json": str(self.factor_loader.json_path),
            "factor_count": self.factor_loader.total_count,
            "supported_factor_types": ["value", "quality", "size", "momentum"],
        }


def create_generator(
    kg_dir: str = "data/kg",
    factor_json_path: Optional[str] = None,
) -> KGFeatureGenerator:
    """Create a feature generator instance"""
    return KGFeatureGenerator(kg_dir, factor_json_path)


__all__ = ['KGFeatureGenerator', 'JSONFactorLoader', 'create_generator']