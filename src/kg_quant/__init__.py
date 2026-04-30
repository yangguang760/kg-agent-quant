"""
KG-AgentQuant

Knowledge Graph Enhanced Alpha Factor Research with LLM Verification

A multi-stage pipeline for discovering and validating quantitative alpha factors
using Large Language Models with stage-wise independent verification.
"""

from kg_quant.kg import (
    KGFeatureGenerator,
    JSONFactorLoader,
    QLIBExpressionEvaluator,
    KGRetriever,
    KGExplainer,
    FactorExplanation,
    SemanticConsistencyChecker,
    ConsistencyLevel,
)
from kg_quant.factor import FactorASTParser, FactorConstraint
from kg_quant.evaluation.metrics import (
    FactorEvaluator,
    compute_ic,
    compute_rank_ic,
    compute_arr,
    compute_mdd,
    compute_ir,
    compute_calmar,
)
from kg_quant.core import Evaluator, ConfigManager, get_config_manager
from kg_quant.utils import setup_logger, get_logger
from kg_quant.llm import (
    LLMConfig,
    LLMConfigManager,
    ConceptGenerator,
    RelationGenerator,
    HypothesisGenerator,
)

__version__ = "0.1.0"

__all__ = [
    # KG Module
    "KGFeatureGenerator",
    "JSONFactorLoader",
    "QLIBExpressionEvaluator",
    "KGRetriever",
    "KGExplainer",
    "FactorExplanation",
    "SemanticConsistencyChecker",
    "ConsistencyLevel",
    # Factor Module
    "FactorASTParser",
    "FactorConstraint",
    # Evaluation
    "FactorEvaluator",
    "Evaluator",
    # Metrics
    "compute_ic",
    "compute_rank_ic",
    "compute_arr",
    "compute_mdd",
    "compute_ir",
    "compute_calmar",
    # Core
    "ConfigManager",
    "get_config_manager",
    # Utils
    "setup_logger",
    "get_logger",
    # LLM Module
    "LLMConfig",
    "LLMConfigManager",
    "ConceptGenerator",
    "RelationGenerator",
    "HypothesisGenerator",
]