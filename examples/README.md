# Examples

| File | Description | Requires API Key |
|------|-------------|-----------------|
| `01_factor_generation.py` | Basic factor generation from KG | No |
| `02_evaluation.py` | Factor evaluation metrics | No |
| `03_complete_pipeline.py` | End-to-end pipeline demo | No |
| `04_llm_generation.py` | LLM-powered concept/relation/hypothesis generation | Yes |
| `demo_agent_harness.py` | Full multi-agent verification topology | No |
| `run_deliberation_live.py` | Real multi-turn deliberation experiment | Yes |
| `run_feedback_loop_live.py` | Agentic feedback loop experiment | Yes |
| `run_heterogeneity_study.py` | 5-model heterogeneity study | Yes |

## Experiment Data

The `data/experiments/` directory contains key result JSONs:
- `unified_deliberation_results.json` — CSC deliberation (70 relations, 93% convergence)
- `heterogeneity_study.json` — 5-model pairwise Spearman matrix
- `human_scores_70.json` — 70-relation human evaluation data
- `threshold_sweep_summary.json` — 19 threshold sensitivity variants
- `extended_summary.json` — Cross-market portfolio results
- `eq_sc_deliberation_results.json` — EQ (91%) + SC (50%) deliberation
