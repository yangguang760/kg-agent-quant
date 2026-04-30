#!/usr/bin/env python3
"""
KG Retriever Module

Provides retrieval and search capabilities over the KG-AgentQuant knowledge graph.
Supports querying by factor type, entity ID, and relation type.
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Set
from collections import defaultdict


class KGRetriever:
    """
    KG Retriever for Knowledge Graph Enhanced Alpha Factor Research.

    Retrieves concepts, relations, and evidence from the KG to constrain
    hypothesis generation and reduce hallucination.
    """

    def __init__(self, kg_dir: str = "data/kg"):
        """
        Initialize KG Retriever.

        Args:
            kg_dir: Directory containing KG JSON files
        """
        self.kg_dir = Path(kg_dir)

        # Load KG data
        self.layer1_concepts = self._load_layer1()
        self.layer2_relations = self._load_layer2()
        self.layer3_data = self._load_layer3()

        # Build indices for fast retrieval
        self._build_indices()

    def _load_layer1(self) -> Dict:
        """Load Layer 1 concepts"""
        layer1_path = self.kg_dir / "layer1_concepts.json"
        if not layer1_path.exists():
            return {"topics": []}

        with open(layer1_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _load_layer2(self) -> Dict:
        """Load Layer 2 relations"""
        layer2_path = self.kg_dir / "layer2_relations_final.json"
        if not layer2_path.exists():
            return {"relations": []}

        with open(layer2_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _load_layer3(self) -> Dict:
        """Load Layer 3 validation data"""
        layer3_path = self.kg_dir / "layer3_frequentsave.json"
        if not layer3_path.exists():
            return {}

        with open(layer3_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _build_indices(self):
        """Build indices for fast retrieval"""
        # Index concepts by name and type
        self.concept_by_name: Dict[str, Dict] = {}
        self.concepts_by_type: Dict[str, List[Dict]] = defaultdict(list)
        self.concepts_by_category: Dict[str, List[Dict]] = defaultdict(list)

        for topic in self.layer1_concepts.get('topics', []):
            entity_type = topic.get('entity_type', 'Unknown')
            for concept in topic.get('concepts', []):
                name = concept.get('name', '')
                category = concept.get('category', '')

                self.concept_by_name[name] = concept
                self.concepts_by_type[entity_type].append(concept)
                if category:
                    self.concepts_by_category[category].append(concept)

        # Index relations by head, tail, and type
        self.relations_by_head: Dict[str, List[Dict]] = defaultdict(list)
        self.relations_by_tail: Dict[str, List[Dict]] = defaultdict(list)
        self.relations_by_type: Dict[str, List[Dict]] = defaultdict(list)

        for rel in self.layer2_relations.get('relations', []):
            head = rel.get('head', '')
            tail = rel.get('tail', '')
            rel_type = rel.get('type', '')

            self.relations_by_head[head].append(rel)
            self.relations_by_tail[tail].append(rel)
            self.relations_by_type[rel_type].append(rel)

    def retrieve_related_concepts(self, factor_type: str, limit: int = 10) -> List[Dict]:
        """
        Retrieve concepts related to a factor type.

        Args:
            factor_type: Type of factor (e.g., "value", "growth", "momentum", "quality", "size")
            limit: Maximum number of concepts to return

        Returns:
            List of related concepts with their properties
        """
        # Map factor types to KG entity types
        factor_type_mapping = {
            "value": ["FinancialIndicator", "EconomicTheory"],
            "growth": ["FinancialIndicator", "FactorPattern"],
            "momentum": ["FactorPattern", "StatisticalPattern"],
            "quality": ["FinancialIndicator"],
            "size": ["FinancialIndicator", "MarketRegime"],
        }

        relevant_types = factor_type_mapping.get(factor_type.lower(), ["FinancialIndicator"])

        # Collect concepts from relevant types
        concepts = []
        for entity_type in relevant_types:
            concepts.extend(self.concepts_by_type.get(entity_type, []))

        # Rank by relevance (simple: prefer concepts with more relations)
        concept_scores = []
        for concept in concepts:
            name = concept.get('name', '')
            relation_count = len(self.relations_by_head.get(name, [])) + \
                           len(self.relations_by_tail.get(name, []))
            concept_scores.append((concept, relation_count))

        # Sort by relation count (descending)
        concept_scores.sort(key=lambda x: x[1], reverse=True)

        return [concept for concept, score in concept_scores[:limit]]

    def get_evidence(self, entity_id: str) -> List[Dict]:
        """
        Get evidence for a specific entity.

        Args:
            entity_id: Entity name/ID

        Returns:
            List of evidence items (relations with their sources and confidence)
        """
        evidence = []

        # Get relations where this entity is head or tail
        head_relations = self.relations_by_head.get(entity_id, [])
        tail_relations = self.relations_by_tail.get(entity_id, [])

        for rel in head_relations + tail_relations:
            evidence_item = {
                "relation_type": rel.get('type', ''),
                "head": rel.get('head', ''),
                "tail": rel.get('tail', ''),
                "confidence": rel.get('confidence', rel.get('weight', 0.0)),
                "source": rel.get('source', rel.get('evidence', '')),
                "is_head": rel.get('head', '') == entity_id
            }
            evidence.append(evidence_item)

        return evidence

    def get_entity_details(self, entity_name: str) -> Optional[Dict]:
        """
        Get detailed information about an entity.

        Args:
            entity_name: Name of the entity

        Returns:
            Entity details or None if not found
        """
        return self.concept_by_name.get(entity_name)

    def search_concepts(self, query: str) -> List[Dict]:
        """
        Search concepts by name (fuzzy match).

        Args:
            query: Search query

        Returns:
            List of matching concepts
        """
        query_lower = query.lower()
        matches = []

        for name, concept in self.concept_by_name.items():
            if query_lower in name.lower():
                matches.append(concept)

        return matches

    def get_relations_for_pair(self, entity1: str, entity2: str) -> List[Dict]:
        """
        Get relations between two entities.

        Args:
            entity1: First entity name
            entity2: Second entity name

        Returns:
            List of relations connecting the two entities
        """
        relations = []

        # Check if entity1 -> entity2
        for rel in self.relations_by_head.get(entity1, []):
            if rel.get('tail', '') == entity2:
                relations.append(rel)

        # Check if entity2 -> entity1
        for rel in self.relations_by_head.get(entity2, []):
            if rel.get('head', '') == entity1:
                relations.append(rel)

        return relations

    def get_statistics(self) -> Dict:
        """Get KG statistics"""
        return {
            "total_concepts": len(self.concept_by_name),
            "total_relations": len(self.layer2_relations.get('relations', [])),
            "entity_types": len(self.concepts_by_type),
            "relation_types": len(self.relations_by_type),
            "layer3_verified": self.layer3_data.get('verified_total', 0)
        }

    def get_concepts_by_category(self, category: str) -> List[Dict]:
        """Get all concepts in a specific category."""
        return self.concepts_by_category.get(category, [])

    def get_relation_types(self) -> List[str]:
        """Get all unique relation types."""
        return list(self.relations_by_type.keys())

    def get_related_entities(self, entity_name: str, max_results: int = 20) -> List[Dict]:
        """
        Get all entities related to a given entity.

        Args:
            entity_name: Name of the entity
            max_results: Maximum number of results

        Returns:
            List of related entities with relation information
        """
        results = []

        for rel in self.relations_by_head.get(entity_name, []):
            results.append({
                'entity': rel.get('tail', ''),
                'relation_type': rel.get('type', ''),
                'direction': 'outgoing',
                'confidence': rel.get('confidence', rel.get('weight', 0.0))
            })

        for rel in self.relations_by_tail.get(entity_name, []):
            results.append({
                'entity': rel.get('head', ''),
                'relation_type': rel.get('type', ''),
                'direction': 'incoming',
                'confidence': rel.get('confidence', rel.get('weight', 0.0))
            })

        # Sort by confidence and limit results
        results.sort(key=lambda x: x['confidence'], reverse=True)
        return results[:max_results]


def create_retriever(kg_dir: str = "data/kg") -> KGRetriever:
    """Create a KG retriever instance."""
    return KGRetriever(kg_dir)


__all__ = ['KGRetriever', 'create_retriever']