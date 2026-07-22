# Architecture

## Overview

KG-AgentQuant implements a **multi-agent verification architecture** for LLM-driven factor discovery pipelines. The core innovation is **deliberative consensus**: when independent scorer agents disagree on the quality of an intermediate artifact, they engage in structured multi-turn debate with reasoning exchange — rather than defaulting to simple score averaging.

The system is described in our DAI 2026 Industry Track paper: *"From Topics to Factors: Multi-Agent Verification with Deliberative Consensus."*

## Pipeline Architecture

```
Topics ──→ Entities ──[CSC Gate]──→ Relations ──[EQ Gate]──→ Hypotheses ──[SC Gate]──→ Factors → Portfolio
              │                      │                      │
         Two scorers             Two scorers            Two scorers
         (GLM-5 + Kimi-K2.5)    (GLM-5 + DS-V4-Pro)    (DS-V4-Pro × 2)
              │                      │                      │
         ┌────┴────┐           ┌────┴────┐           ┌────┴────┐
         │ agree?   │           │ agree?   │           │ agree?   │
         │ σ ≤ 0.2 → PASS      │ σ ≤ 0.2 → PASS      │ σ ≤ 0.2 → PASS
         │ σ > 0.2 → DELIBERATE│ σ > 0.2 → DELIBERATE│ σ > 0.2 → DELIBERATE
         └─────────┘           └─────────┘           └─────────┘
```

**Five pipeline stages**, **three verification gates**. Each gate employs two independent scorer LLM instances. When scorers agree (std ≤ 0.2), the artifact passes. When they disagree, deliberation is triggered.

## Deliberative Consensus Protocol

At each gate, the two scorer instances independently evaluate the artifact. If their scores diverge (σ > 0.2), the protocol executes:

1. **Primary scoring**: Each scorer produces a confidence score (0–1) with reasoning.
2. **Deliberation trigger**: If σ > 0.2, each scorer receives a structured prompt containing the artifact, their own score and reasoning, and their peer's score and reasoning.
3. **Independent re-evaluation**: Scorers re-evaluate. They may maintain, revise, or partially adjust their score, but must explain their reasoning in response to peer critiques.
4. **Convergence check**: If σ ≤ 0.1 after re-evaluation, consensus is declared. Otherwise, the loop continues (max 3 rounds).
5. **Controversial flag**: If consensus is not reached after 3 rounds, the artifact is flagged for human review.

## Deliberation Results

| Gate | Artifacts | Disputed | Convergence |
|------|-----------|----------|-------------|
| CSC (Relations) | 809 | 9% (70) | 93% |
| EQ (Hypotheses) | 509 | 45% (228) | 91% |
| SC (Factors) | 1,179 | 89% (1,052) | 50% |

**Key finding**: Reasoning exchange (not mere score comparison) is the active ingredient. A minimal-deliberation condition (scores only, no reasoning) achieves only 55% convergence — a 5.1× degradation in standard deviation reduction.

## Agent Module (`kg_quant/agents/`)

The multi-agent verification infrastructure is implemented in four modules:

### protocol.py
Defines the inter-agent communication layer:
- `AgentRole`: GENERATOR, CSC_SCORER, EQ_SCORER, SC_SCORER, ORCHESTRATOR
- `AgentMessage`: Typed message envelope (10 message types) with artifact provenance tracking
- `AgentCard`: A2A-inspired capability advertisement
- `Artifact`: Unit of work with full provenance chain, reasoning trace, and quality scores
- `DeliberationConfig`: Protocol parameters (θ_d=0.2, θ_c=0.1, max rounds=3)

### deliberation.py
Multi-turn deliberative consensus engine:
- `DeliberationEngine`: Orchestrates the deliberation loop
- `DeliberationResult`: Per-artifact trajectory (round-by-round scores, convergence status)
- `DeliberationSummary`: Aggregate metrics across deliberation sessions

### feedback_loop.py
Agentic self-correction loop:
- `FeedbackLoopEngine`: Routes rejected artifacts back to Generator with scorer critique
- `RevisionRecord`: Per-revision tracking (score before/after, revision type)
- `FeedbackLoopSummary`: Aggregate revision metrics

### harness.py
Central orchestration infrastructure:
- `AgentHarness`: Manages agent lifecycle, message routing, gate enforcement
- `AgentRegistry`: A2A-inspired agent registration and discovery
- `MessageRouter`: Routes and logs all inter-agent messages
- `QualityGateEnforcer`: Enforces CSC/EQ/SC thresholds

## Key Empirical Findings

| Finding | Evidence |
|---------|----------|
| Simple averaging degrades accuracy in 55% of disagreement cases | 5-model heterogeneity study (n=130) |
| Deliberation with reasoning exchange resolves 91–93% of CSC/EQ disagreements | Full-scale experiments |
| Scorer quality dominates scorer quantity | Best 2-model consensus (r=0.90) > 5-model (r=0.81) |
| Heterogeneous scorers outperform homogeneous | Homo deliberation: +2% human error |
| LLM scorers approach human consistency | LLM-human r=0.61 vs inter-human r=0.71 |
| Post-deliberation consensus often wrong (68% of cases) | Motivates CONTROVERSIAL flag for human review |

## Heterogeneity Study

Five LLM scorer models (Qwen3.7-Max, GLM-5.2, Kimi-K2.7, MiniMax-M3, DeepSeek-V4-Pro) independently scored 30 relations with human ground truth. Key findings:

- Scorer quality varies dramatically: Qwen r=0.89, Kimi r=0.91 vs. GLM r=−0.15 (negatively correlated with human judgment)
- Adding poor scorers degrades consensus: best 2-model (r=0.90) > 3-model (r=0.85) > 5-model (r=0.81)
- Even the best model pair (Qwen+Kimi, r=0.91) suffers: averaging degrades 62% of their disagreement cases

## Deployment

- **Market**: Chinese A-shares (CSI300, CSI500, CSI800, CSI1000)
- **Period**: 2022–2025 (368 out-of-sample trading days)
- **Model**: LightGBM with fixed hyperparameters
- **Portfolio**: TopK-Dropout (top 50, 5 dropped daily, 10 bps cost)
- **Cost**: ~$2.50–$5.00 per 1,000 relations in API calls
- **Code**: ~2,000 lines of Python, MIT licensed
