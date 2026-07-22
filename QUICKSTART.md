# Quick Start

## 5-Minute Setup

```bash
pip install -e .
```

## Multi-Agent Verification Demo

```python
from kg_quant.agents import (
    AgentHarness, build_default_harness_config,
    AgentRole, Artifact
)

# Create harness with Generator + CSC/EQ/SC Scorers
config = build_default_harness_config()
harness = AgentHarness(config=config)
harness.start_session()
harness.register_default_agents()

# Route an artifact through verification
artifact = Artifact(
    artifact_type="relation",
    content={"head": "ROE", "tail": "PE"},
    reasoning_trace="Both relate to earnings..."
)
artifact.add_provenance(AgentRole.GENERATOR, "generated")
artifact.quality_scores = {"csc": 0.88, "eq": 0.82, "sc": 0.90}

# Check quality gates
passed, reasons = harness.gate_enforcer.check_all(artifact)
print(f"Gate check: {'PASSED' if passed else 'FAILED'}")

# Test deliberation trigger
from kg_quant.agents import DeliberationConfig
dc = DeliberationConfig()
print(f"Should deliberate [0.3, 0.9]: {dc.should_deliberate([0.3, 0.9])}")
```

## Run Full Demo

```bash
python examples/demo_agent_harness.py
```

## Run Deliberation (requires API key)

```bash
export DELIB_API_KEY=sk-...
python examples/run_deliberation_live.py
```

## More Examples

| File | Description |
|------|-------------|
| `examples/01_factor_generation.py` | Basic factor generation |
| `examples/04_llm_generation.py` | LLM-powered concept/relation generation |
| `examples/demo_agent_harness.py` | Full multi-agent topology demonstration |
| `examples/run_deliberation_live.py` | Real multi-turn deliberation experiment |
| `examples/run_feedback_loop_live.py` | Agentic feedback loop experiment |
| `examples/run_heterogeneity_study.py` | 5-model heterogeneity study |
| `examples/run_eq_sc_deliberation.py` | EQ + SC deliberation experiment |
