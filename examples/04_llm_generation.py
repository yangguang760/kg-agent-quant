#!/usr/bin/env python3
"""
Example 4: LLM-Powered Generation (Full Pipeline)

Demonstrates how to use the LLM module to generate financial concepts,
relations, and hypotheses with real API.
"""

import sys
sys.path.insert(0, 'src')

import json
from kg_quant.llm.config import LLMConfig, load_llm_config
from kg_quant.llm.generators import (
    ConceptGenerator,
    RelationGenerator,
    HypothesisGenerator,
)


def main():
    print("=" * 70)
    print("KG-AgentQuant Example 4: LLM-Powered Generation (Full Pipeline)")
    print("=" * 70)

    # Step 1: Configure LLM from local config file
    print("\n[Step 1] Configuring LLM...")

    try:
        config = load_llm_config("yunnetC")
        print(f"  Provider: custom")
        print(f"  Model: {config.model}")
        print(f"  API Base: {config.api_base}")
        if config.api_key == "YOUR_API_KEY_HERE":
            print("  WARNING: Please set your API key in config/llm.json")
            print("  Falling back to mock client for demonstration")
            config = LLMConfig(provider="mock", model="mock-model")
    except Exception as e:
        print(f"  Failed to load config: {e}")
        print("  Using mock client")
        config = LLMConfig(provider="mock", model="mock-model")

    # Step 2: Generate Concepts
    print("\n[Step 2] Generating Financial Concepts...")

    concept_gen = ConceptGenerator(config=config, language="en")

    # Generate concepts for financial metrics topic
    print("  Generating concepts for: financial_metrics")
    concepts = concept_gen.generate(topic="financial_metrics", min_concepts=10, max_concepts=15)

    if concepts:
        print(f"  Generated {len(concepts)} concepts:")
        for c in concepts[:5]:
            print(f"    - {c.name}: {c.description[:60]}...")
        if len(concepts) > 5:
            print(f"    ... and {len(concepts) - 5} more")
    else:
        print("  WARNING: No concepts generated")

    # Save concepts to file
    concepts_file = "data/generated/concepts.json"
    import os
    os.makedirs("data/generated", exist_ok=True)
    with open(concepts_file, 'w') as f:
        json.dump(concept_gen.to_json(), f, indent=2, ensure_ascii=False)
    print(f"  Saved concepts to: {concepts_file}")

    # Step 3: Generate Relations
    print("\n[Step 3] Generating Relations...")

    relation_gen = RelationGenerator(config=config, language="en")

    # Use the generated concepts
    entities = [{"name": c.name, "category": c.category} for c in concepts]
    if not entities:
        # Fallback to basic entities
        entities = [
            {"name": "ROE", "category": "financial_metric"},
            {"name": "PE", "category": "valuation"},
            {"name": "PB", "category": "valuation"},
            {"name": "returns", "category": "outcome"},
        ]
        print("  Using fallback entities for demonstration")

    print(f"  Generating relations between {len(entities)} entities...")
    relations = relation_gen.generate_relations(
        concepts=entities,
        relation_type="correlated",
        min_confidence=0.5,
    )

    if relations:
        print(f"  Generated {len(relations)} relations:")
        for r in relations[:5]:
            print(f"    {r.source} --({r.relation_type})--> {r.target} (conf: {r.confidence:.2f})")
        if len(relations) > 5:
            print(f"    ... and {len(relations) - 5} more")
    else:
        print("  WARNING: No relations generated")

    # Save relations to file
    relations_file = "data/generated/relations.json"
    with open(relations_file, 'w') as f:
        json.dump(relation_gen.to_json(), f, indent=2, ensure_ascii=False)
    print(f"  Saved relations to: {relations_file}")

    # Step 4: Generate Hypotheses
    print("\n[Step 4] Generating Investment Hypotheses...")

    hypothesis_gen = HypothesisGenerator(config=config, language="en")

    hypotheses = hypothesis_gen.generate(
        entities=entities,
        hypothesis_type="financial_metric",
        min_hypotheses=5,
    )

    if hypotheses:
        print(f"  Generated {len(hypotheses)} hypotheses:")
        for h in hypotheses[:3]:
            print(f"    - {h.statement}")
            print(f"      Logic: {h.economic_logic[:60]}...")
        if len(hypotheses) > 3:
            print(f"    ... and {len(hypotheses) - 3} more")
    else:
        print("  WARNING: No hypotheses generated")

    # Save hypotheses to file
    hypotheses_file = "data/generated/hypotheses.json"
    with open(hypotheses_file, 'w') as f:
        json.dump(hypothesis_gen.to_json(), f, indent=2, ensure_ascii=False)
    print(f"  Saved hypotheses to: {hypotheses_file}")

    # Step 5: Summary
    print("\n" + "=" * 70)
    print("LLM Generation Summary")
    print("=" * 70)
    print(f"  API Provider: {config.provider}")
    print(f"  API Model: {config.model}")
    print(f"  Concepts Generated: {len(concepts)}")
    print(f"  Relations Generated: {len(relations)}")
    print(f"  Hypotheses Generated: {len(hypotheses)}")
    print(f"\n  Output Files:")
    print(f"    - {concepts_file}")
    print(f"    - {relations_file}")
    print(f"    - {hypotheses_file}")

    print("\n" + "=" * 70)
    print("Example 4 completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
