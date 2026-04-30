"""
LLM Module for KG-AgentQuant

Provides LLM-based generation of financial concepts, relations, and hypotheses.
"""

from .config import (
    LLMConfig,
    LLMConfigManager,
    create_llm_client,
    load_llm_config,
    MockLLMClient,
)
from .generators import (
    GeneratedConcept,
    GeneratedRelation,
    GeneratedHypothesis,
    ConceptGenerator,
    RelationGenerator,
    HypothesisGenerator,
    TOPIC_PROMPTS,
    RELATION_PROMPTS,
    HYPOTHESIS_PROMPTS,
)

__all__ = [
    # Config
    'LLMConfig',
    'LLMConfigManager',
    'create_llm_client',
    'load_llm_config',
    'MockLLMClient',
    # Data classes
    'GeneratedConcept',
    'GeneratedRelation',
    'GeneratedHypothesis',
    # Generators
    'ConceptGenerator',
    'RelationGenerator',
    'HypothesisGenerator',
    # Prompts
    'TOPIC_PROMPTS',
    'RELATION_PROMPTS',
    'HYPOTHESIS_PROMPTS',
]
