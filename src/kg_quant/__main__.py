#!/usr/bin/env python3
"""
KG-AgentQuant

Knowledge Graph Enhanced Alpha Factor Research with LLM Verification
"""

from kg_quant import (
    KGFeatureGenerator,
    QLIBExpressionEvaluator,
    KGRetriever,
    KGExplainer,
    SemanticConsistencyChecker,
    FactorASTParser,
    FactorEvaluator,
    Evaluator,
)

__version__ = "0.1.0"
__author__ = "Research Team"
__all__ = [
    "KGFeatureGenerator",
    "QLIBExpressionEvaluator",
    "KGRetriever",
    "KGExplainer",
    "SemanticConsistencyChecker",
    "FactorASTParser",
    "FactorEvaluator",
    "Evaluator",
]