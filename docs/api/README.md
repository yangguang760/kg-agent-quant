# API Reference

## Core Classes

### KGFeatureGenerator

```python
class KGFeatureGenerator:
    def __init__(
        self,
        kg_dir: str = "data/kg",
        factor_json_path: Optional[str] = None
    )
```

Generate knowledge graph enhanced alpha factors.

**Methods:**

#### `generate_kg_features()`
```python
def generate_kg_features(
    self,
    factor_type: str,
    n_features: int = 10,
    data: Optional[pd.DataFrame] = None,
    seed: int = 42
) -> pd.DataFrame
```

Generate KG features of specified type.

#### `resolve_valid_factors()`
```python
def resolve_valid_factors(
    self,
    factor_type: str,
    n_features: Optional[int] = None,
    seed: int = 42
) -> List[Tuple[int, str, Dict]]
```

Resolve valid factors for a given type.

#### `get_feature_metadata()`
```python
def get_feature_metadata(self) -> Dict
```

Get feature generation metadata.

---

### QLIBExpressionEvaluator

```python
class QLIBExpressionEvaluator:
    def __init__(self)
```

Evaluate QLIB-style factor expressions.

**Methods:**

#### `evaluate()`
```python
def evaluate(
    self,
    expression: str,
    data: pd.DataFrame
) -> pd.Series
```

Evaluate expression on data.

**Supported Operators:**
- Time series: `TS_MEAN`, `TS_STD`, `TS_DELTA`, `TS_DELAY`, `TS_SUM`, `TS_RANK`
- Cross-section: `RANK`, `CS_RANK`, `ZSCORE`, `SCALE`
- Math: `LOG`, `EXP`, `ABS`, `SQRT`, `POW`
- Logical: `IF`, `GT`, `LT`, `AND`, `OR`

---

### KGRetriever

```python
class KGRetriever:
    def __init__(self, kg_dir: str = "data/kg")
```

Retrieve concepts and relations from the knowledge graph.

**Methods:**

#### `retrieve_related_concepts()`
```python
def retrieve_related_concepts(
    self,
    factor_type: str,
    limit: int = 10
) -> List[Dict]
```

#### `get_evidence()`
```python
def get_evidence(self, entity_id: str) -> List[Dict]
```

#### `search_concepts()`
```python
def search_concepts(self, query: str) -> List[Dict]
```

#### `get_statistics()`
```python
def get_statistics(self) -> Dict
```

---

### KGExplainer

```python
class KGExplainer:
    def __init__(self, kg_dir: str = "data/kg")
```

Generate explanations for alpha factors.

**Methods:**

#### `explain_factor()`
```python
def explain_factor(
    self,
    factor_expression: str,
    factor_name: str = ""
) -> FactorExplanation
```

#### `explain_batch()`
```python
def explain_batch(
    self,
    factor_expressions: List[str]
) -> List[FactorExplanation]
```

---

### SemanticConsistencyChecker

```python
class SemanticConsistencyChecker:
    def __init__(self, kg_dir: str = "data/kg")
```

Check semantic consistency of hypotheses.

**Methods:**

#### `check()`
```python
def check(self, hypothesis: str) -> ConsistencyResult
```

#### `check_batch()`
```python
def check_batch(self, hypotheses: List[str]) -> List[ConsistencyResult]
```

---

## Evaluation Metrics

### compute_ic()

```python
def compute_ic(
    factor_values: Union[np.ndarray, pd.Series],
    future_returns: Union[np.ndarray, pd.Series],
    method: str = 'pearson'
) -> float
```

Compute Information Coefficient.

### compute_rank_ic()

```python
def compute_rank_ic(
    factor_values: Union[np.ndarray, pd.Series],
    future_returns: Union[np.ndarray, pd.Series]
) -> float
```

Compute Rank IC (Spearman correlation).

### compute_arr()

```python
def compute_arr(
    returns: Union[np.ndarray, pd.Series, List[float]],
    annualization_factor: int = 252
) -> float
```

Compute Annualized Rate of Return.

### compute_mdd()

```python
def compute_mdd(
    returns: Union[np.ndarray, pd.Series, List[float]]
) -> float
```

Compute Maximum Drawdown.

### FactorEvaluator

```python
class FactorEvaluator:
    def __init__(self, annualization_factor: int = 252)
```

**Methods:**

#### `evaluate_factor()`
```python
def evaluate_factor(
    self,
    factor_values: pd.DataFrame,
    future_returns: pd.DataFrame
) -> Dict[str, float]
```

#### `evaluate_strategy()`
```python
def evaluate_strategy(
    self,
    portfolio_returns: pd.Series
) -> Dict[str, float]
```

---

## Data Classes

### FactorExplanation

```python
@dataclass
class FactorExplanation:
    factor_expression: str
    factor_name: str
    used_indicators: List[str]
    used_theories: List[str]
    used_patterns: List[str]
    economic_logic: str
    logic_chain: List[str]
    supporting_evidence: List[str]
    evidence_sources: List[str]
    applicable_markets: List[str]
    market_regimes: List[str]
    constraints: List[str]
    explanation_confidence: float
```

### ConsistencyResult

```python
@dataclass
class ConsistencyResult:
    hypothesis: str
    consistency_level: ConsistencyLevel
    confidence: float
    conflicting_knowledge: List[str]
    supporting_knowledge: List[str]
    suggestion: Optional[str]
```

### ConsistencyLevel

```python
class ConsistencyLevel(Enum):
    CONSISTENT = "consistent"
    PARTIALLY_CONSISTENT = "partial"
    INCONSISTENT = "inconsistent"
    UNKNOWN = "unknown"
```

---

## LLM Module

### LLMConfig

```python
@dataclass
class LLMConfig:
    provider: str = "openai"
    model: str = "gpt-4o-mini"
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    max_tokens: int = 4096
    temperature: float = 0.7
    timeout: int = 120
    max_retries: int = 3
```

### LLMConfigManager

```python
class LLMConfigManager:
    PRESETS = {
        "fast": {"provider": "openai", "model": "gpt-4o-mini", "temperature": 0.3},
        "balanced": {"provider": "openai", "model": "gpt-4o", "temperature": 0.7},
        "creative": {"provider": "openai", "model": "gpt-4o", "temperature": 1.0},
        "deepseek": {"provider": "deepseek", "model": "deepseek-chat"},
    }

    def get_preset(self, name: str) -> LLMConfig
    def get_config(self, name: str) -> LLMConfig
    def save_config(self, name: str, config: LLMConfig) -> None
    def list_configs(self) -> List[str]
```

---

### ConceptGenerator

```python
class ConceptGenerator:
    def __init__(
        self,
        config: Optional[LLMConfig] = None,
        language: str = "en",
        callback: Optional[Callable] = None,
    )

    def generate(
        self,
        topic: str = "financial_metrics",
        min_concepts: int = 15,
        max_concepts: int = 50,
    ) -> List[GeneratedConcept]

    def generate_all_topics(self, min_per_topic: int = 10) -> Dict[str, List[GeneratedConcept]]
```

**Topics:** `financial_metrics`, `economic_theories`, `factor_patterns`, `market_institutions`

---

### RelationGenerator

```python
class RelationGenerator:
    def __init__(
        self,
        config: Optional[LLMConfig] = None,
        language: str = "en",
        max_workers: int = 4,
        callback: Optional[Callable] = None,
    )

    def generate_relation(
        self,
        source: str,
        target: str,
        relation_type: str = "correlated",
    ) -> Optional[GeneratedRelation]

    def generate_relations(
        self,
        concepts: List[Dict],
        relation_type: str = "correlated",
        min_confidence: float = 0.6,
    ) -> List[GeneratedRelation]

    def generate_relations_batch(
        self,
        concepts: List[Dict],
        relation_type: str = "correlated",
        min_confidence: float = 0.6,
        max_pairs: Optional[int] = None,
    ) -> List[GeneratedRelation]
```

**Relation types:** `causal`, `correlated`

---

### HypothesisGenerator

```python
class HypothesisGenerator:
    def __init__(
        self,
        config: Optional[LLMConfig] = None,
        language: str = "en",
        callback: Optional[Callable] = None,
    )

    def generate(
        self,
        entities: Optional[List[Dict]] = None,
        hypothesis_type: str = "financial_metric",
        min_hypotheses: int = 10,
    ) -> List[GeneratedHypothesis]

    def generate_from_kg(
        self,
        kg_path: str,
        min_hypotheses: int = 10,
    ) -> List[GeneratedHypothesis]
```

**Hypothesis types:** `financial_metric`, `technical_pattern`

---

### GeneratedConcept

```python
@dataclass
class GeneratedConcept:
    id: str
    name: str
    category: str
    description: str
    properties: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.8
    source: str = "llm_generation"
```

### GeneratedRelation

```python
@dataclass
class GeneratedRelation:
    source: str
    target: str
    relation_type: str
    description: str
    confidence: float = 0.8
    evidence: List[str] = field(default_factory=list)
```

### GeneratedHypothesis

```python
@dataclass
class GeneratedHypothesis:
    statement: str
    variable: str
    operator: str
    target: str
    economic_logic: str
    confidence: float = 0.8
    supporting_entities: List[str] = field(default_factory=list)
    risks: List[str] = field(default_factory=list)
```

---

## KG Schema

### Entity

```python
@dataclass
class Entity:
    id: str
    name: str
    category: str
    description: str
    properties: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.8
```

### Relation

```python
@dataclass
class Relation:
    source: str
    target: str
    relation_type: str
    description: str
    confidence: float = 0.8
    evidence: List[str] = field(default_factory=list)
```

### EntityType

```python
class EntityType(Enum):
    FINANCIAL_METRIC
    ECONOMIC_THEORY
    FACTOR_PATTERN
    MARKET_INSTITUTION
    CONCEPT
```

### RelationType

```python
class RelationType(Enum):
    CAUSAL
    CORRELATED_WITH
    THEORY_SUPPORTS
    THEORY_OPPOSES
    PREDICTS
    AFFECTS
```