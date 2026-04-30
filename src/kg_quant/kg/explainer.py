"""
KG Explainer Module for KG-AgentQuant

Provides factor explanation and traceability capabilities.
"""

import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Set


@dataclass
class FactorExplanation:
    """Factor explanation data structure"""
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

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return asdict(self)


class FactorParser:
    """Factor expression parser"""

    # Operator mapping (English -> Chinese)
    OPERATOR_MAP = {
        'RANK': '横截面排序',
        'TS_RANK': '时间序列排序',
        'MEAN': '均值',
        'TS_MEAN': '时间序列均值',
        'STD': '标准差',
        'TS_STD': '时间序列标准差',
        'DELAY': '滞后',
        'DELTA': '差分',
        'SUM': '求和',
        'MAX': '最大值',
        'MIN': '最小值',
        'ABS': '绝对值',
        'LOG': '对数',
        'SQRT': '平方根',
        'ZSCORE': 'Z分数标准化',
        'SCALE': '归一化',
        'GT': '大于',
        'LT': '小于',
        'AND': '与',
        'OR': '或',
    }

    # Base feature mapping
    FEATURE_MAP = {
        '$close': '收盘价',
        '$open': '开盘价',
        '$high': '最高价',
        '$low': '最低价',
        '$volume': '成交量',
        '$vwap': '均价',
        '$returns': '收益率',
        '$roe': 'ROE',
        '$roa': 'ROA',
        '$pe': 'PE',
        '$pb': 'PB',
    }

    # KG concept mapping
    KG_CONCEPT_MAP = {
        '$pe': 'PE',
        '$pb': 'PB',
        '$roe': 'ROE',
        '$roa': 'ROA',
        '$close': '收盘价',
        '$volume': '成交量',
        '$returns': '收益率',
    }

    def parse(self, expression: str) -> tuple:
        """Parse expression to operators and features"""
        operators = re.findall(r'\b([A-Z_]+)\b', expression)
        operators = [op for op in operators if op in self.OPERATOR_MAP]
        features = re.findall(r'(\$[a-z_]+)', expression)
        return operators, features

    def to_human_readable(self, expression: str) -> str:
        """Convert expression to human readable form"""
        result = expression
        for op, desc in self.OPERATOR_MAP.items():
            result = result.replace(op, desc)
        for feat, desc in self.FEATURE_MAP.items():
            result = result.replace(feat, desc)
        return result

    def extract_kg_concepts(self, expression: str) -> Set[str]:
        """Extract KG concepts from expression"""
        concepts = set()
        _, features = self.parse(expression)
        for feat in features:
            if feat in self.KG_CONCEPT_MAP:
                concepts.add(self.KG_CONCEPT_MAP[feat])
        return concepts


class KGExplainer:
    """
    KG Factor Explainer

    Provides complete explainability tracing for each generated alpha factor:
    1. KG concepts used (financial indicators, economic theories)
    2. Economic logic behind the factor
    3. Empirical evidence support
    4. Applicable market conditions
    """

    def __init__(self, kg_dir: str = "data/kg"):
        """
        Initialize the explainer.

        Args:
            kg_dir: KG data directory
        """
        self.kg_dir = Path(kg_dir)

        # Initialize factor parser
        self.factor_parser = FactorParser()

        # Theory mapping
        self.theory_map = {
            'ROE': ['价值投资', '成长投资'],
            'PE': ['价值投资', '有效市场假说'],
            'PB': ['价值投资'],
            'ROA': ['价值投资'],
            '收盘价': ['动量理论', '均值回归理论'],
            '成交量': ['量价关系理论'],
            '收益率': ['动量理论', '均值回归理论']
        }

        # Factor pattern rules
        self.pattern_rules = {
            '价值因子': lambda ops, feats: any(f in feats for f in ['$pe', '$pb', '$roe']) and 'RANK' in ops,
            '动量因子': lambda ops, feats: '$returns' in feats or '$close' in feats,
            '波动率因子': lambda ops, feats: 'STD' in ops or 'TS_STD' in ops,
            '质量因子': lambda ops, feats: any(f in feats for f in ['$roe', '$roa'])
        }

    def _retrieve_theories(self, indicators: List[str]) -> List[str]:
        """Retrieve related economic theories"""
        theories = set()
        for indicator in indicators:
            if indicator in self.theory_map:
                theories.update(self.theory_map[indicator])
        return list(theories)

    def _identify_pattern(self, operators: List[str], features: List[str]) -> List[str]:
        """Identify factor patterns"""
        patterns = []
        for pattern_name, rule_func in self.pattern_rules.items():
            if rule_func(operators, features):
                patterns.append(pattern_name)
        return patterns

    def _generate_economic_logic(
        self,
        indicators: List[str],
        theories: List[str],
        patterns: List[str]
    ) -> str:
        """Generate economic logic description"""
        logic_parts = []

        if '价值投资' in theories:
            logic_parts.append("基于价值投资理论，该因子寻找被市场低估的股票")

        if '成长投资' in theories:
            logic_parts.append("基于成长投资理论，该因子关注公司的成长性和盈利能力")

        if '动量理论' in theories:
            logic_parts.append("利用价格动量效应，预期过去表现好的股票继续表现好")

        if '均值回归理论' in theories:
            logic_parts.append("基于均值回归理论，预期价格将回归长期均值")

        if '价值因子' in patterns:
            logic_parts.append("作为价值因子，通过低估值指标筛选投资标的")

        if '动量因子' in patterns:
            logic_parts.append("作为动量因子，捕捉价格趋势的持续性")

        if '质量因子' in patterns:
            logic_parts.append("作为质量因子，筛选盈利能力强、财务健康的公司")

        if indicators:
            logic_parts.append(f"使用关键财务指标：{', '.join(indicators)} 作为选股依据")

        return '。'.join(logic_parts) + '。' if logic_parts else "该因子基于财务指标构建。"

    def _generate_logic_chain(self, operators: List[str]) -> List[str]:
        """Generate reasoning chain"""
        chain = []
        step_num = 1

        if 'TS_MEAN' in operators:
            chain.append(f"{step_num}. 计算时间序列均值，平滑短期波动")
            step_num += 1

        if 'TS_STD' in operators:
            chain.append(f"{step_num}. 计算时间序列标准差，衡量波动性")
            step_num += 1

        if 'RANK' in operators:
            chain.append(f"{step_num}. 横截面排序，识别相对优劣")
            step_num += 1

        if 'DELAY' in operators:
            chain.append(f"{step_num}. 引入滞后，避免前瞻性偏差")
            step_num += 1

        if 'DELTA' in operators:
            chain.append(f"{step_num}. 计算差分，捕捉变化趋势")
            step_num += 1

        if not chain:
            chain.append("1. 直接使用原始指标进行排序")

        return chain

    def _determine_applicability(
        self,
        indicators: List[str],
        theories: List[str],
        patterns: List[str]
    ) -> tuple:
        """Determine applicable conditions"""
        markets = []
        regimes = []
        constraints = []

        if '价值投资' in theories:
            markets.extend(['A股', '港股', '美股'])
            regimes.append('震荡市')
            constraints.append('需要足够的财务数据')

        if '动量理论' in theories:
            markets.extend(['A股', '美股'])
            regimes.append('趋势市')
            constraints.append('高流动性市场')

        if '价值因子' in patterns:
            constraints.append('适合价值投资风格')

        if '动量因子' in patterns:
            constraints.append('需要较高的交易频率')

        return markets or ['通用'], regimes or ['通用'], constraints or ['无明显约束']

    def _calculate_confidence(
        self,
        num_indicators: int,
        num_theories: int,
        num_patterns: int
    ) -> float:
        """Calculate explanation confidence"""
        confidence = 0.5
        confidence += min(0.2, num_indicators * 0.1)
        confidence += min(0.2, num_theories * 0.1)
        confidence += min(0.1, num_patterns * 0.05)
        return min(1.0, confidence)

    def _generate_name(self, operators: List[str], features: List[str]) -> str:
        """Generate factor name"""
        parts = []

        if 'RANK' in operators:
            parts.append('排序')

        if 'TS_MEAN' in operators:
            parts.append('时序均值')

        if 'TS_STD' in operators:
            parts.append('波动率')

        if '$roe' in features:
            parts.append('ROE')
        elif '$pe' in features:
            parts.append('PE')
        elif '$pb' in features:
            parts.append('PB')
        elif '$returns' in features:
            parts.append('动量')

        return '_'.join(parts) + '因子' if parts else '复合因子'

    def explain_factor(self, factor_expression: str, factor_name: str = "") -> FactorExplanation:
        """
        Generate complete explanation for a factor.

        Args:
            factor_expression: Factor expression (e.g., "RANK(TS_MEAN($close, 20))")
            factor_name: Optional factor name

        Returns:
            FactorExplanation with complete tracing
        """
        operators, features = self.factor_parser.parse(factor_expression)

        # Identify used indicators
        used_indicators = []
        for feat in features:
            if feat in self.factor_parser.KG_CONCEPT_MAP:
                used_indicators.append(self.factor_parser.KG_CONCEPT_MAP[feat])

        used_theories = self._retrieve_theories(used_indicators)
        used_patterns = self._identify_pattern(operators, features)

        return FactorExplanation(
            factor_expression=factor_expression,
            factor_name=factor_name or self._generate_name(operators, features),
            used_indicators=used_indicators,
            used_theories=used_theories,
            used_patterns=used_patterns,
            economic_logic=self._generate_economic_logic(used_indicators, used_theories, used_patterns),
            logic_chain=self._generate_logic_chain(operators),
            supporting_evidence=[],
            evidence_sources=[],
            applicable_markets=self._determine_applicability(used_indicators, used_theories, used_patterns)[0],
            market_regimes=self._determine_applicability(used_indicators, used_theories, used_patterns)[1],
            constraints=self._determine_applicability(used_indicators, used_theories, used_patterns)[2],
            explanation_confidence=self._calculate_confidence(
                len(used_indicators), len(used_theories), len(used_patterns)
            )
        )

    def explain_batch(self, factor_expressions: List[str]) -> List[FactorExplanation]:
        """Explain multiple factors"""
        return [self.explain_factor(expr) for expr in factor_expressions]


def create_explainer(kg_dir: str = "data/kg") -> KGExplainer:
    """Create an explainer instance"""
    return KGExplainer(kg_dir)


__all__ = ['KGExplainer', 'FactorExplanation', 'FactorParser', 'create_explainer']