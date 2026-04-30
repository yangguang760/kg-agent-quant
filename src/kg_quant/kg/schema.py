"""
Knowledge Graph Schema for KG-AgentQuant

Defines the data structures and types for the financial knowledge graph.
"""

from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any


class EntityType(Enum):
    """Types of entities in the financial knowledge graph."""
    FINANCIAL_METRIC = auto()
    ECONOMIC_THEORY = auto()
    FACTOR_PATTERN = auto()
    MARKET_INSTITUTION = auto()
    CONCEPT = auto()


class RelationType(Enum):
    """Types of relations between entities."""
    CAUSAL = auto()
    CORRELATED_WITH = auto()
    THEORY_SUPPORTS = auto()
    THEORY_OPPOSES = auto()
    PREDICTS = auto()
    AFFECTS = auto()


@dataclass
class Entity:
    """A node in the knowledge graph."""
    id: str
    name: str
    category: str
    description: str
    properties: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.8

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "name": self.name,
            "category": self.category,
            "description": self.description,
            "properties": self.properties,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'Entity':
        return cls(
            id=data.get("id", ""),
            name=data.get("name", ""),
            category=data.get("category", ""),
            description=data.get("description", ""),
            properties=data.get("properties", {}),
            confidence=data.get("confidence", 0.8),
        )


@dataclass
class Relation:
    """An edge in the knowledge graph."""
    source: str
    target: str
    relation_type: str
    description: str
    confidence: float = 0.8
    evidence: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "source": self.source,
            "target": self.target,
            "relation_type": self.relation_type,
            "description": self.description,
            "confidence": self.confidence,
            "evidence": self.evidence,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'Relation':
        return cls(
            source=data.get("source", ""),
            target=data.get("target", ""),
            relation_type=data.get("relation_type", ""),
            description=data.get("description", ""),
            confidence=data.get("confidence", 0.8),
            evidence=data.get("evidence", []),
            metadata=data.get("metadata", {}),
        )


@dataclass
class Hypothesis:
    """An investment hypothesis derived from the KG."""
    id: str
    statement: str
    variable_left: str
    operator: str
    variable_right: str
    economic_logic: str
    confidence: float = 0.8
    supporting_entities: List[str] = field(default_factory=list)
    supporting_relations: List[str] = field(default_factory=list)
    risks: List[str] = field(default_factory=list)
    expression: Optional[str] = None

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "statement": self.statement,
            "variable_left": self.variable_left,
            "operator": self.operator,
            "variable_right": self.variable_right,
            "economic_logic": self.economic_logic,
            "confidence": self.confidence,
            "supporting_entities": self.supporting_entities,
            "supporting_relations": self.supporting_relations,
            "risks": self.risks,
            "expression": self.expression,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'Hypothesis':
        return cls(
            id=data.get("id", ""),
            statement=data.get("statement", ""),
            variable_left=data.get("variable_left", ""),
            operator=data.get("operator", ""),
            variable_right=data.get("variable_right", ""),
            economic_logic=data.get("economic_logic", ""),
            confidence=data.get("confidence", 0.8),
            supporting_entities=data.get("supporting_entities", []),
            supporting_relations=data.get("supporting_relations", []),
            risks=data.get("risks", []),
            expression=data.get("expression"),
        )


class FinanceKGSchema:
    """
    Schema validator and manager for the financial knowledge graph.

    Example:
        >>> schema = FinanceKGSchema()
        >>> entity = Entity(id="roe", name="ROE", category="financial_metric", description="...")
        >>> schema.validate_entity(entity)
        True
    """

    VALID_CATEGORIES = [
        "financial_metric",
        "economic_theory",
        "factor_pattern",
        "market_institution",
        "concept",
    ]

    VALID_RELATION_TYPES = [
        "causal",
        "correlated_with",
        "theory_supports",
        "theory_opposes",
        "predicts",
        "affects",
    ]

    def validate_entity(self, entity: Entity) -> bool:
        """Validate an entity."""
        if not entity.id:
            return False
        if not entity.name:
            return False
        if entity.category not in self.VALID_CATEGORIES:
            return False
        if not 0 <= entity.confidence <= 1:
            return False
        return True

    def validate_relation(self, relation: Relation) -> bool:
        """Validate a relation."""
        if not relation.source:
            return False
        if not relation.target:
            return False
        if relation.relation_type not in self.VALID_RELATION_TYPES:
            return False
        if not 0 <= relation.confidence <= 1:
            return False
        return True

    def validate_hypothesis(self, hypothesis: Hypothesis) -> bool:
        """Validate a hypothesis."""
        if not hypothesis.id:
            return False
        if not hypothesis.statement:
            return False
        if not hypothesis.variable_left:
            return False
        if not hypothesis.variable_right:
            return False
        return True


__all__ = [
    'EntityType',
    'RelationType',
    'Entity',
    'Relation',
    'Hypothesis',
    'FinanceKGSchema',
]