"""
LLM-powered Generators for KG-AgentQuant

Generates financial concepts, relations, and hypotheses using LLMs.
"""

import re
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed

from .config import LLMConfig, create_llm_client

logger = logging.getLogger(__name__)


@dataclass
class GeneratedConcept:
    """A generated financial concept."""
    id: str
    name: str
    category: str
    description: str
    properties: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.8
    source: str = "llm_generation"


@dataclass
class GeneratedRelation:
    """A generated relation between concepts."""
    source: str
    target: str
    relation_type: str
    description: str
    confidence: float = 0.8
    evidence: List[str] = field(default_factory=list)
    generation_source: str = "llm_generation"


@dataclass
class GeneratedHypothesis:
    """A generated investment hypothesis."""
    statement: str
    variable: str
    operator: str
    target: str
    economic_logic: str
    confidence: float = 0.8
    supporting_entities: List[str] = field(default_factory=list)
    risks: List[str] = field(default_factory=list)


# =============================================================================
# Topic Prompts for Concept Generation
# =============================================================================

TOPIC_PROMPTS = {
    "financial_metrics": {
        "zh": """你是一个金融量化研究专家。请基于以下主题生成金融概念实体。

主题：财务指标

请生成至少{min_concepts}个金融领域的核心概念，包括：
1. 估值指标：如PE、PB、PS、PCF等
2. 盈利指标：如ROE、ROA、ROIC、毛利率、净利率等
3. 成长指标：如营收增速、利润增速、PEG等
4. 质量指标：如现金流、资产周转、存货周转等
5. 风险指标：如波动率、BETA、VAR等

每个概念需要包含：ID、名称、类型、描述、关键属性。

输出格式（JSON数组）：
[
  {{
    "id": "concept_xxx",
    "name": "概念名称",
    "category": "financial_metric",
    "description": "概念描述",
    "properties": {{
      "formula": "计算公式（如有）",
      "typical_range": "典型取值范围",
      "unit": "单位"
    }}
  }}
]""",
        "en": """You are a quantitative finance expert. Generate financial concept entities.

Topic: Financial Metrics

Generate at least {min_concepts} core concepts in finance:
1. Valuation: PE, PB, PS, PCF, etc.
2. Profitability: ROE, ROA, ROIC, Gross Margin, Net Margin, etc.
3. Growth: Revenue Growth, Profit Growth, PEG, etc.
4. Quality: Cash Flow, Asset Turnover, Inventory Turnover, etc.
5. Risk: Volatility, BETA, VaR, etc.

Output format (JSON array)""",
    },
    "economic_theories": {
        "zh": """你是一个金融量化研究专家。请基于以下主题生成金融概念实体。

主题：经济理论

请生成至少{min_concepts}个经济学相关概念，包括：
1. 价值投资理论：内在价值、安全边际等
2. 成长投资理论：竞争优势、护城河等
3. 因子理论：价值因子、质量因子、动量因子等
4. 行为金融：锚定效应、羊群效应等
5. 宏观因素：利率、通胀、汇率等

输出格式（JSON数组）：
[
  {{
    "id": "concept_xxx",
    "name": "概念名称",
    "category": "economic_theory",
    "description": "概念描述"
  }}
]""",
        "en": """You are a quantitative finance expert. Generate financial concept entities.

Topic: Economic Theories

Generate concepts related to:
1. Value investing: intrinsic value, margin of safety
2. Growth investing: competitive advantage, moat
3. Factor theory: value, quality, momentum factors
4. Behavioral finance: anchoring, herding
5. Macro factors: interest rates, inflation, exchange rates""",
    },
    "factor_patterns": {
        "zh": """你是一个金融量化研究专家。请基于以下主题生成金融概念实体。

主题：因子模式

请生成至少{min_concepts}个量化因子相关概念，包括：
1. 时间序列模式：趋势、均值回复、季节性等
2. 截面模式：行业轮动、市值效应等
3. 复合模式：多因子组合、因子正交化等
4. 信号模式：金叉死叉、突破等
5. 风险模式：黑天鹅、尾部风险等

输出格式（JSON数组）""",
        "en": """You are a quantitative finance expert. Generate financial concept entities.

Topic: Factor Patterns

Generate concepts related to:
1. Time series patterns: trend, mean reversion, seasonality
2. Cross-sectional patterns: sector rotation, size effect
3. Composite patterns: multi-factor, orthogonalization
4. Signal patterns: golden cross, breakouts
5. Risk patterns: black swan, tail risk""",
    },
    "market_institutions": {
        "zh": """你是一个金融量化研究专家。请基于以下主题生成金融概念实体。

主题：市场制度

请生成至少{min_concepts}个市场制度相关概念，包括：
1. 交易制度：涨跌停、T+1、熔断等
2. 监管制度：信息披露、关联交易等
3. 指数编制：指数调整、权重计算等
4. 产品设计：ETF、期货、期权等
5. 市场结构：做市商、竞价交易等

输出格式（JSON数组）""",
        "en": """You are a quantitative finance expert. Generate financial concept entities.

Topic: Market Institutions

Generate concepts related to:
1. Trading rules: price limits, T+1, circuit breakers
2. Regulatory systems: disclosure, related party transactions
3. Index methodology: rebalancing, weight calculation
4. Product design: ETF, futures, options
5. Market structure: market makers, auction""",
    },
}


# =============================================================================
# Relation Generation Templates
# =============================================================================

RELATION_PROMPTS = {
    "causal": {
        "zh": """分析以下两个金融概念之间的因果关系：

源概念：{source_name}
目标概念：{target_name}

请判断：
1. 是否存在因果关系？
2. 方向是什么（源→目标 还是 目标→源）？
3. 传导机制是什么？
4. 置信度是多少（0-1）？

输出格式（JSON）：
{{
  "has_relation": true/false,
  "direction": "source->target" 或 "target->source" 或 "bidirectional",
  "relation_type": "causal",
  "mechanism": "传导机制描述",
  "confidence": 0.85,
  "evidence": ["证据1", "证据2"]
}}""",
        "en": """Analyze causal relationship between financial concepts:

Source: {source_name}
Target: {target_name}

Output format:
{{
  "has_relation": true/false,
  "direction": "source->target",
  "mechanism": "transmission mechanism",
  "confidence": 0.85
}}""",
    },
    "correlated": {
        "zh": """分析以下两个金融概念之间的相关性：

概念A：{source_name}
概念B：{target_name}

请判断：
1. 是否存在相关性？
2. 相关方向（正相关/负相关）？
3. 相关强度（强/中/弱）？
4. 可能的解释是什么？

输出格式（JSON）：
{{
  "has_relation": true/false,
  "direction": "positive/negative",
  "strength": "strong/medium/weak",
  "relation_type": "correlated_with",
  "explanation": "解释",
  "confidence": 0.8
}}""",
        "en": """Analyze correlation between financial concepts:

Concept A: {source_name}
Concept B: {target_name}

Output format:
{{
  "has_relation": true/false,
  "direction": "positive/negative",
  "confidence": 0.8
}}""",
    },
}


# =============================================================================
# Hypothesis Generation Templates
# =============================================================================

HYPOTHESIS_PROMPTS = {
    "financial_metric": {
        "zh": """你是一个量化投资研究员。请基于金融概念生成可验证的投资假设。

已有关键念：
- {entities}

请生成至少{min_hypotheses}个投资假设，格式如下：
1. 假设陈述：清晰描述预期关系
2. 左变量：被解释变量（如收益率）
3. 操作符：关系类型（如正相关、负相关）
4. 右变量：解释变量（如ROE）
5. 经济逻辑：为什么这个假设合理
6. 潜在风险：可能失效的原因

输出格式（JSON数组）：
[
  {{
    "statement": "高ROE股票具有更高收益率",
    "variable_left": "future_return",
    "operator": "positively_correlated",
    "variable_right": "roe",
    "economic_logic": "高ROE代表高效资本利用...",
    "confidence": 0.8,
    "supporting_entities": ["ROE", "盈利质量"],
    "risks": ["ROE可能被人为操控"]
  }}
]""",
        "en": """You are a quantitative investment researcher. Generate testable investment hypotheses in JSON format.

Entities: {entities}

Generate at least {min_hypotheses} hypotheses in the following JSON format:
[
  {{
    "statement": "Clear description of the hypothesis",
    "variable_left": "future_return",
    "operator": "positively_correlated",
    "variable_right": "roe",
    "economic_logic": "Why this hypothesis makes economic sense",
    "confidence": 0.8,
    "supporting_entities": ["ROE", "Profitability"],
    "risks": ["Risk factor 1", "Risk factor 2"]
  }}
]

IMPORTANT: Return ONLY valid JSON, no other text.""",
    },
    "technical_pattern": {
        "zh": """你是一个量化投资研究员。请基于技术分析模式生成可验证的投资假设。

请生成基于技术指标的假设，例如：
- 均线金叉/死叉策略
- 突破策略
- 均值回复策略

输出格式（JSON数组）：
[
  {{
    "statement": "短期均线上穿长期均线预示正收益",
    "variable_left": "future_return",
    "operator": "positively_correlated",
    "variable_right": "ma_cross_signal",
    "economic_logic": "动量效应...",
    "confidence": 0.7
  }}
]""",
        "en": """You are a quantitative investment researcher. Generate technical analysis hypotheses.

Generate hypotheses about:
- Moving average crossovers
- Breakout patterns
- Mean reversion patterns""",
    },
}


# =============================================================================
# Generator Base Class
# =============================================================================

class BaseGenerator:
    """Base class for LLM-powered generators."""

    def __init__(
        self,
        config: Optional[LLMConfig] = None,
        language: str = "en",
        callback: Optional[Callable] = None,
    ):
        self.config = config or LLMConfig(provider="mock")
        self.language = language
        self.callback = callback
        self._client = create_llm_client(self.config)

    def _call_llm(self, prompt: str, max_retries: int = 3, retry_delay: float = 5.0, **kwargs) -> Dict:
        """Call LLM with prompt and retry logic."""
        messages = [{"role": "user", "content": prompt}]
        last_error = None

        for attempt in range(max_retries):
            try:
                # Check if it's a mock client first
                if hasattr(self._client, 'call_count'):
                    response = self._client.chat(messages, **kwargs)
                else:
                    # OpenAI-compatible client
                    response = self._client.chat.completions.create(
                        messages=messages,
                        model=self.config.model,
                        temperature=self.config.temperature,
                        max_tokens=self.config.max_tokens,
                        **kwargs
                    )
                    if hasattr(response, 'choices'):
                        response = {"content": response.choices[0].message.content}

                content = response.get("content", "")
                return self._parse_response(content)
            except Exception as e:
                last_error = e
                error_str = str(e)
                # Check if it's a rate limit error
                if "429" in error_str or "rate_limit" in error_str.lower() or "cooldown" in error_str.lower():
                    if attempt < max_retries - 1:
                        import time
                        wait_time = retry_delay * (2 ** attempt)  # Exponential backoff
                        logger.warning(f"Rate limited, retrying in {wait_time:.1f}s...")
                        time.sleep(wait_time)
                        continue
                logger.error(f"LLM call failed: {e}")
                return {"error": str(e)}

        return {"error": str(last_error) if last_error else "Max retries exceeded"}

    def _parse_response(self, content: str) -> Dict:
        """Parse LLM response, extracting JSON."""
        content = content.strip()

        # Try to find JSON in code blocks
        match = re.search(r'```(?:json)?\s*([\s\S]*?)```', content)
        if match:
            content = match.group(1).strip()

        # Try to find and parse JSON arrays or objects
        # Handle both {"key": [...]} and [...]
        for pattern in [r'\[\s*\{[\s\S]*\}\s*\]', r'\{\s*"[\w]+"\s*:\s*\[']:
            match = re.search(pattern, content)
            if match:
                try:
                    start = match.start()
                    bracket_count = 0
                    end = start
                    for i in range(start, len(content)):
                        if content[i] == '[' or content[i] == '{':
                            bracket_count += 1
                        elif content[i] == ']' or content[i] == '}':
                            bracket_count -= 1
                            if bracket_count == 0:
                                end = i + 1
                                break
                    json_str = content[start:end]
                    parsed = json.loads(json_str)
                    # If it's a list, wrap it in a dict with generic key
                    if isinstance(parsed, list):
                        return {"items": parsed}
                    return parsed
                except json.JSONDecodeError:
                    pass

        # Try to parse the entire content as JSON
        try:
            parsed = json.loads(content)
            if isinstance(parsed, list):
                return {"items": parsed}
            return parsed
        except json.JSONDecodeError:
            pass

        return {"raw": content}


# =============================================================================
# Concept Generator
# =============================================================================

class ConceptGenerator(BaseGenerator):
    """
    Generate financial concepts using LLM.

    Example:
        >>> generator = ConceptGenerator()
        >>> concepts = generator.generate(topic="financial_metrics", min_concepts=20)
        >>> print(f"Generated {len(concepts)} concepts")
    """

    def __init__(
        self,
        config: Optional[LLMConfig] = None,
        language: str = "en",
        callback: Optional[Callable] = None,
    ):
        super().__init__(config, language, callback)
        self._generated_concepts: Dict[str, GeneratedConcept] = {}

    def generate(
        self,
        topic: str = "financial_metrics",
        min_concepts: int = 15,
        max_concepts: int = 50,
    ) -> List[GeneratedConcept]:
        """
        Generate financial concepts for a given topic.

        Args:
            topic: Topic key (financial_metrics, economic_theories, factor_patterns, market_institutions)
            min_concepts: Minimum number of concepts to generate
            max_concepts: Maximum number of concepts to generate

        Returns:
            List of GeneratedConcept objects
        """
        if topic not in TOPIC_PROMPTS:
            raise ValueError(f"Unknown topic: {topic}. Available: {list(TOPIC_PROMPTS.keys())}")

        prompt_template = TOPIC_PROMPTS[topic][self.language]
        prompt = prompt_template.format(min_concepts=min_concepts, max_concepts=max_concepts)

        result = self._call_llm(prompt)

        concepts = []
        if "error" in result:
            logger.warning(f"LLM call failed: {result['error']}")
            return concepts

        data = result.get("concepts", result.get("entities", result.get("items", [])))

        # Handle various LLM response formats
        flat_concepts = []
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    # Format 1: {"category": "Valuation", "concepts": [...]}
                    if "concepts" in item and isinstance(item["concepts"], list):
                        category = item.get("category", topic)
                        for concept_item in item["concepts"]:
                            if isinstance(concept_item, str):
                                flat_concepts.append({"name": concept_item, "category": category})
                            elif isinstance(concept_item, dict):
                                concept_item["category"] = concept_item.get("category", category)
                                flat_concepts.append(concept_item)
                    # Format 2: {"category": "...", "concept": "...", ...}
                    elif "concept" in item:
                        flat_concepts.append({
                            "name": item.get("concept", ""),
                            "category": item.get("category", topic),
                            "description": item.get("interpretation", ""),
                            "properties": {"formula": item.get("formula", "")},
                        })
                    # Format 3: {"name": "...", ...}
                    elif "name" in item:
                        flat_concepts.append(item)

        if flat_concepts:
            data = flat_concepts
        elif not isinstance(data, list):
            data = []

        if isinstance(data, list):
            for i, item in enumerate(data):
                concept = GeneratedConcept(
                    id=item.get("id", item.get("ticker_like", f"concept_{i}")),
                    name=item.get("name", item.get("ticker_like", "")),
                    category=item.get("category", topic),
                    description=item.get("description", ""),
                    properties=item.get("properties", {}),
                    confidence=item.get("confidence", 0.8),
                    source="llm_generation",
                )
                if concept.name:
                    concepts.append(concept)
                    self._generated_concepts[concept.id] = concept

        if self.callback:
            self.callback("concepts", len(concepts))

        return concepts

    def generate_all_topics(self, min_per_topic: int = 10) -> Dict[str, List[GeneratedConcept]]:
        """Generate concepts for all predefined topics."""
        all_concepts = {}
        for topic in TOPIC_PROMPTS.keys():
            concepts = self.generate(topic=topic, min_concepts=min_per_topic)
            all_concepts[topic] = concepts
        return all_concepts

    def to_json(self) -> List[Dict]:
        """Export generated concepts as list of dicts."""
        return [
            {
                "id": c.id,
                "name": c.name,
                "category": c.category,
                "description": c.description,
                "properties": c.properties,
                "confidence": c.confidence,
            }
            for c in self._generated_concepts.values()
        ]


# =============================================================================
# Relation Generator
# =============================================================================

class RelationGenerator(BaseGenerator):
    """
    Generate relations between financial concepts using LLM.

    Example:
        >>> generator = RelationGenerator()
        >>> concepts = [{"name": "ROE"}, {"name": "returns"}]
        >>> relations = generator.generate_relations(concepts, relation_type="causal")
        >>> print(f"Generated {len(relations)} relations")
    """

    def __init__(
        self,
        config: Optional[LLMConfig] = None,
        language: str = "en",
        max_workers: int = 4,
        callback: Optional[Callable] = None,
    ):
        super().__init__(config, language, callback)
        self.max_workers = max_workers
        self._generated_relations: List[GeneratedRelation] = []

    def generate_relation(
        self,
        source: str,
        target: str,
        relation_type: str = "correlated",
    ) -> Optional[GeneratedRelation]:
        """
        Generate a single relation between two concepts.

        Args:
            source: Source concept name
            target: Target concept name
            relation_type: Type of relation (causal, correlated, etc.)

        Returns:
            GeneratedRelation or None if generation failed
        """
        prompt_template = RELATION_PROMPTS.get(relation_type, RELATION_PROMPTS["correlated"])[self.language]
        prompt = prompt_template.format(source_name=source, target_name=target)

        result = self._call_llm(prompt)

        if "error" in result:
            logger.warning(f"LLM call failed for {source}->{target}: {result['error']}")
            return None

        relation = GeneratedRelation(
            source=source,
            target=target,
            relation_type=result.get("relation_type", relation_type),
            description=result.get("explanation", result.get("mechanism", "")),
            confidence=result.get("confidence", 0.8),
            evidence=result.get("evidence", []),
        )

        if result.get("has_relation", True):
            self._generated_relations.append(relation)

        return relation

    def generate_relations(
        self,
        concepts: List[Dict],
        relation_type: str = "correlated",
        min_confidence: float = 0.6,
    ) -> List[GeneratedRelation]:
        """
        Generate relations between multiple concepts.

        Args:
            concepts: List of concept dicts with 'name' field
            relation_type: Type of relations to generate
            min_confidence: Minimum confidence threshold

        Returns:
            List of GeneratedRelation objects
        """
        relations = []

        for i, c1 in enumerate(concepts):
            for c2 in concepts[i+1:]:
                name1 = c1.get("name", "")
                name2 = c2.get("name", "")

                if not name1 or not name2:
                    continue

                relation = self.generate_relation(name1, name2, relation_type)

                if relation and relation.confidence >= min_confidence:
                    relations.append(relation)

                if self.callback:
                    self.callback("relation", len(relations))

        return relations

    def generate_relations_batch(
        self,
        concepts: List[Dict],
        relation_type: str = "correlated",
        min_confidence: float = 0.6,
        max_pairs: Optional[int] = None,
    ) -> List[GeneratedRelation]:
        """
        Generate relations between concepts using parallel processing.

        Args:
            concepts: List of concept dicts with 'name' field
            relation_type: Type of relations to generate
            min_confidence: Minimum confidence threshold
            max_pairs: Maximum number of concept pairs to process

        Returns:
            List of GeneratedRelation objects
        """
        # Generate all pairs
        pairs = []
        for i, c1 in enumerate(concepts):
            for c2 in concepts[i+1:]:
                name1 = c1.get("name", "")
                name2 = c2.get("name", "")
                if name1 and name2:
                    pairs.append((name1, name2))

        if max_pairs:
            pairs = pairs[:max_pairs]

        # Process in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self.generate_relation, s, t, relation_type): (s, t)
                for s, t in pairs
            }

            for future in as_completed(futures):
                relation = future.result()
                if relation and relation.confidence >= min_confidence:
                    self._generated_relations.append(relation)

                if self.callback:
                    self.callback("relation_batch", len(self._generated_relations))

        return self._generated_relations

    def to_json(self) -> List[Dict]:
        """Export generated relations as list of dicts."""
        return [
            {
                "source": r.source,
                "target": r.target,
                "relation_type": r.relation_type,
                "description": r.description,
                "confidence": r.confidence,
                "evidence": r.evidence,
            }
            for r in self._generated_relations
        ]


# =============================================================================
# Hypothesis Generator
# =============================================================================

class HypothesisGenerator(BaseGenerator):
    """
    Generate investment hypotheses using LLM.

    Example:
        >>> generator = HypothesisGenerator()
        >>> entities = [{"name": "ROE"}, {"name": "returns"}]
        >>> hypotheses = generator.generate(entities=entities, min_hypotheses=10)
        >>> print(f"Generated {len(hypotheses)} hypotheses")
    """

    def __init__(
        self,
        config: Optional[LLMConfig] = None,
        language: str = "en",
        callback: Optional[Callable] = None,
    ):
        super().__init__(config, language, callback)
        self._generated_hypotheses: List[GeneratedHypothesis] = []

    def generate(
        self,
        entities: Optional[List[Dict]] = None,
        hypothesis_type: str = "financial_metric",
        min_hypotheses: int = 10,
    ) -> List[GeneratedHypothesis]:
        """
        Generate investment hypotheses.

        Args:
            entities: List of entity dicts with 'name' field
            hypothesis_type: Type of hypotheses (financial_metric, technical_pattern)
            min_hypotheses: Minimum number of hypotheses to generate

        Returns:
            List of GeneratedHypothesis objects
        """
        prompt_template = HYPOTHESIS_PROMPTS.get(hypothesis_type, HYPOTHESIS_PROMPTS["financial_metric"])[self.language]

        if entities:
            entity_names = [e.get("name", "") for e in entities if e.get("name")]
            entities_str = ", ".join(entity_names[:20])  # Limit to 20 entities
        else:
            entities_str = "ROE, PE, PB, returns, volatility, momentum"

        prompt = prompt_template.format(
            entities=entities_str,
            min_hypotheses=min_hypotheses,
        )

        result = self._call_llm(prompt)

        hypotheses = []
        if "error" in result:
            logger.warning(f"LLM call failed: {result['error']}")
            return hypotheses

        data = result.get("hypotheses", result.get("items", []))
        if not isinstance(data, list):
            data = [data]

        for i, item in enumerate(data):
            hypothesis = GeneratedHypothesis(
                statement=item.get("statement", item.get("hypothesis", "")),
                variable=item.get("variable_left", item.get("variable", "")),
                operator=item.get("operator", ""),
                target=item.get("variable_right", item.get("target", "")),
                economic_logic=item.get("economic_logic", ""),
                confidence=item.get("confidence", 0.8),
                supporting_entities=item.get("supporting_entities", []),
                risks=item.get("risks", []),
            )

            if hypothesis.statement:
                hypotheses.append(hypothesis)
                self._generated_hypotheses.append(hypothesis)

        if self.callback:
            self.callback("hypotheses", len(hypotheses))

        return hypotheses

    def generate_from_kg(
        self,
        kg_path: str,
        min_hypotheses: int = 10,
    ) -> List[GeneratedHypothesis]:
        """
        Generate hypotheses from existing knowledge graph.

        Args:
            kg_path: Path to knowledge graph JSON file
            min_hypotheses: Minimum number of hypotheses to generate

        Returns:
            List of GeneratedHypothesis objects
        """
        try:
            with open(kg_path, 'r') as f:
                kg_data = json.load(f)

            entities = kg_data.get("entities", kg_data.get("concepts", []))
            return self.generate(entities=entities, min_hypotheses=min_hypotheses)
        except Exception as e:
            logger.error(f"Failed to load KG from {kg_path}: {e}")
            return []

    def to_json(self) -> List[Dict]:
        """Export generated hypotheses as list of dicts."""
        return [
            {
                "statement": h.statement,
                "variable_left": h.variable,
                "operator": h.operator,
                "variable_right": h.target,
                "economic_logic": h.economic_logic,
                "confidence": h.confidence,
                "supporting_entities": h.supporting_entities,
                "risks": h.risks,
            }
            for h in self._generated_hypotheses
        ]


__all__ = [
    'GeneratedConcept',
    'GeneratedRelation',
    'GeneratedHypothesis',
    'ConceptGenerator',
    'RelationGenerator',
    'HypothesisGenerator',
    'TOPIC_PROMPTS',
    'RELATION_PROMPTS',
    'HYPOTHESIS_PROMPTS',
]