# Architecture

## Overview

KG-AgentQuant implements a multi-stage LLM-assisted quantitative factor discovery pipeline with stage-wise independent verification.

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                      KG-AgentQuant Pipeline                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   Topics ──────► Entity Expansion ──────► Relation Construction      │
│                      (Layer 1)                  (Layer 2)            │
│                        │                          │                  │
│                        ▼                          ▼                  │
│                   [CSC Filter]              [EQ Filter]              │
│              (Consensus Calibration)     (Explanation Quality)       │
│                        │                          │                  │
│                        ▼                          ▼                  │
│               Hypothesis Generation ─────► Expression Instantiation  │
│                    (Hypotheses)                  (Factors)          │
│                        │                          │                  │
│                        └────────────┬───────────────────────────────┤
│                                     │                                │
│                                     ▼                                │
│                              [SC Filter]                             │
│                       (Semantic Consistency)                         │
│                                     │                                │
│                                     ▼                                │
│                              Validated Alpha Factors                 │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

## Component Architecture

### 1. Knowledge Graph Module (`kg_quant/kg/`)

#### KGRetriever
- Loads and indexes KG data (Layer 1-3)
- Provides retrieval APIs for concepts, relations, and evidence
- Supports fuzzy search and filtering

#### QLIBExpressionEvaluator
- Evaluates QLIB-style factor expressions
- Supports 30+ operators (TS_MEAN, RANK, etc.)
- Batch evaluation with caching

#### KGExplainer
- Generates human-readable explanations for factors
- Traces factor logic from KG concepts
- Computes explanation confidence

#### SemanticConsistencyChecker
- Validates hypothesis consistency against KG
- Detects conflicting knowledge
- Provides correction suggestions

### 2. Factor Module (`kg_quant/factor/`)

#### FactorASTParser
- Parses factor expressions to AST
- Validates complexity constraints
- Computes complexity scores

### 3. Evaluation Module (`kg_quant/evaluation/`)

#### Metrics
- IC (Information Coefficient)
- RankIC (Spearman correlation)
- ARR (Annualized Return)
- MDD (Maximum Drawdown)
- IR (Information Ratio)
- Calmar Ratio

#### FactorEvaluator
- Comprehensive factor evaluation
- Factor + strategy metrics
- Customizable annualization

### 4. Core Module (`kg_quant/core/`)

#### Evaluator
- Unified evaluation interface
- Multi-metric support
- Batch evaluation

#### ConfigManager
- YAML configuration loading
- Environment variable resolution
- Cross-reference resolution

## Quality Control System

### CSC (Consensus Calibration Score)
- Evaluates relation credibility at entity level
- Uses scorer-side LLMs (separated from generators)
- Multiple models for consensus

### EQ (Explanation Quality)
- Evaluates hypothesis coherence
- Checks interpretability
- Financial sufficiency validation

### SC (Semantic Consistency)
- Measures hypothesis-expression fidelity
- Detects semantic drift
- Ensures expression fidelity

## Data Flow

```
1. Load KG Data (Layer 1-3)
   │
2. Generate Entities from Topics
   │
3. Construct Relations
   │
4. Filter with CSC Threshold
   │
5. Generate Hypotheses
   │
6. Filter with EQ Threshold
   │
7. Instantiate Expressions
   │
8. Filter with SC Threshold
   │
9. Compute Factor Values
   │
10. Evaluate & Backtest
```

## Integration Points

### QLib Integration
- Native DatasetH support
- Alpha158 handler compatibility
- Custom model wrappers (XGBoost, CatBoost)

### LLM Integration
- OpenAI compatible
- Anthropic Claude support
- Separated generator-scorer architecture

## Extensibility

The system is designed for extensibility:

1. **New Operators**: Extend `QLIBExpressionEvaluator`
2. **New Quality Metrics**: Extend evaluation module
3. **New Models**: Add to model registry
4. **New Data Sources**: Extend data loader