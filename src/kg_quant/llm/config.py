"""
LLM Configuration Management for KG-AgentQuant

Provides centralized LLM API configuration with presets, audit logging, and
provider-agnostic interface.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Literal
from dataclasses import dataclass, field, asdict
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class LLMConfig:
    """Configuration for an LLM provider."""

    provider: Literal["openai", "anthropic", "azure", "deepseek", "local", "mock", "custom"] = "openai"
    model: str = "gpt-4o-mini"
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    max_tokens: int = 4096
    temperature: float = 0.7
    timeout: int = 120
    max_retries: int = 3
    extra_params: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> bool:
        """Validate configuration."""
        if self.provider not in ["openai", "anthropic", "azure", "deepseek", "local", "mock", "custom"]:
            return False
        if self.max_tokens < 1:
            return False
        if self.temperature < 0 or self.temperature > 2:
            return False
        return True


class LLMConfigManager:
    """
    Centralized LLM configuration manager with presets and audit logging.

    Example:
        >>> manager = LLMConfigManager()
        >>> config = manager.get_config("fast")
        >>> manager.save_request("concept_generation", "ROE", {"tokens": 500})
    """

    PRESETS: Dict[str, Dict[str, Any]] = {
        "fast": {
            "provider": "openai",
            "model": "gpt-4o-mini",
            "temperature": 0.3,
            "max_tokens": 2048,
        },
        "balanced": {
            "provider": "openai",
            "model": "gpt-4o",
            "temperature": 0.7,
            "max_tokens": 4096,
        },
        "creative": {
            "provider": "openai",
            "model": "gpt-4o",
            "temperature": 1.0,
            "max_tokens": 8192,
        },
        "deepseek": {
            "provider": "deepseek",
            "model": "deepseek-chat",
            "api_base": "https://api.deepseek.com",
            "temperature": 0.5,
            "max_tokens": 4096,
        },
    }

    def __init__(self, config_dir: Optional[Path] = None):
        self.config_dir = config_dir or Path.home() / ".kg_quant" / "llm_configs"
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.audit_log_path = self.config_dir / "audit_log.jsonl"
        self._configs: Dict[str, LLMConfig] = {}
        self._load_configs()

    def _load_configs(self) -> None:
        """Load saved configurations from disk."""
        configs_file = self.config_dir / "configs.json"
        if configs_file.exists():
            try:
                with open(configs_file, 'r') as f:
                    data = json.load(f)
                    for name, cfg in data.items():
                        self._configs[name] = LLMConfig(**cfg)
            except Exception as e:
                logger.warning(f"Failed to load configs: {e}")

    def _save_configs(self) -> None:
        """Save configurations to disk."""
        configs_file = self.config_dir / "configs.json"
        data = {name: asdict(cfg) for name, cfg in self._configs.items()}
        with open(configs_file, 'w') as f:
            json.dump(data, f, indent=2)

    def get_preset(self, name: str) -> LLMConfig:
        """Get a preset configuration."""
        if name not in self.PRESETS:
            raise ValueError(f"Unknown preset: {name}. Available: {list(self.PRESETS.keys())}")
        return LLMConfig(**self.PRESETS[name])

    def get_config(self, name: str = "balanced") -> LLMConfig:
        """Get configuration by name (preset or saved)."""
        if name in self.PRESETS:
            return self.get_preset(name)

        if name in self._configs:
            return self._configs[name]

        raise ValueError(f"Unknown config: {name}")

    def save_config(self, name: str, config: LLMConfig) -> None:
        """Save a named configuration."""
        config.api_key = config.api_key or os.getenv(f"{config.provider.upper()}_API_KEY")
        if not config.validate():
            raise ValueError("Invalid configuration")
        self._configs[name] = config
        self._save_configs()

    def list_configs(self) -> List[str]:
        """List all available configurations."""
        return list(self.PRESETS.keys()) + list(self._configs.keys())

    def save_request(self, task_type: str, task_id: str, metadata: Dict[str, Any]) -> None:
        """Log an LLM request for audit purposes."""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "task_type": task_type,
            "task_id": task_id,
            "metadata": metadata,
        }
        with open(self.audit_log_path, 'a') as f:
            f.write(json.dumps(log_entry) + "\n")

    def get_audit_log(self, limit: Optional[int] = None) -> List[Dict]:
        """Retrieve audit log entries."""
        entries = []
        if self.audit_log_path.exists():
            with open(self.audit_log_path, 'r') as f:
                for line in f:
                    entries.append(json.loads(line))
        if limit:
            return entries[-limit:]
        return entries


def create_llm_client(config: LLMConfig) -> Any:
    """
    Create an LLM client based on configuration.

    Example:
        >>> config = LLMConfig(provider="openai", model="gpt-4o-mini")
        >>> client = create_llm_client(config)
    """
    try:
        if config.provider == "openai":
            from openai import OpenAI
            client = OpenAI(
                api_key=config.api_key or os.getenv("OPENAI_API_KEY"),
                base_url=config.api_base,
                timeout=config.timeout,
                max_retries=config.max_retries,
            )
            return client
        elif config.provider == "anthropic":
            from anthropic import Anthropic
            client = Anthropic(
                api_key=config.api_key or os.getenv("ANTHROPIC_API_KEY"),
                timeout=config.timeout,
                max_retries=config.max_retries,
            )
            return client
        elif config.provider == "deepseek":
            from openai import OpenAI
            client = OpenAI(
                api_key=config.api_key or os.getenv("DEEPSEEK_API_KEY"),
                base_url=config.api_base or "https://api.deepseek.com",
                timeout=config.timeout,
            )
            return client
        elif config.provider in ("custom", "local"):
            from openai import OpenAI
            client = OpenAI(
                api_key=config.api_key,
                base_url=config.api_base,
                timeout=config.timeout,
                max_retries=config.max_retries,
            )
            return client
        elif config.provider == "mock":
            return MockLLMClient()
        else:
            raise ValueError(f"Unknown provider: {config.provider}")
    except ImportError as e:
        logger.warning(f"Provider not available: {e}. Using mock client.")
        return MockLLMClient()


def load_llm_config(config_name: str = "yunnetC") -> LLMConfig:
    """Load LLM configuration from local config file."""
    # Path: src/kg_quant/llm/config.py -> project_root/config/llm.json
    config_path = Path(__file__).parent.parent.parent.parent / "config" / "llm.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        configs = json.load(f)

    if config_name not in configs:
        raise ValueError(f"Unknown config: {config_name}. Available: {list(configs.keys())}")

    cfg = configs[config_name]
    return LLMConfig(
        provider="custom",
        model=cfg.get("model", "gpt-5.3-codex"),
        api_key=cfg.get("api_key"),
        api_base=cfg.get("api_base"),
        temperature=cfg.get("temperature", 0.5),
        max_tokens=cfg.get("max_tokens", 4096),
    )


class MockLLMClient:
    """Mock LLM client for testing without API access."""

    def __init__(self):
        self.call_count = 0

    def chat(self, messages: List[Dict], **kwargs) -> Dict:
        """Return mock response based on input."""
        self.call_count += 1
        prompt = messages[-1]["content"] if messages else ""

        if "financial_metrics" in prompt or "Financial Metrics" in prompt or "财务指标" in prompt:
            return {
                "content": json.dumps({
                    "concepts": [
                        {"id": "concept_roe", "name": "ROE", "category": "financial_metric", "description": "Return on Equity - measures profitability relative to shareholder equity", "properties": {"formula": "Net Income / Shareholders' Equity", "typical_range": "0.05-0.25"}},
                        {"id": "concept_pe", "name": "PE", "category": "financial_metric", "description": "Price-to-Earnings ratio - valuation metric comparing stock price to earnings", "properties": {"formula": "Stock Price / Earnings Per Share", "typical_range": "10-30"}},
                        {"id": "concept_pb", "name": "PB", "category": "financial_metric", "description": "Price-to-Book ratio - compares market value to book value", "properties": {"formula": "Market Cap / Book Value", "typical_range": "1-5"}},
                        {"id": "concept_roa", "name": "ROA", "category": "financial_metric", "description": "Return on Assets - measures efficiency in using assets", "properties": {"formula": "Net Income / Total Assets", "typical_range": "0.02-0.15"}},
                        {"id": "concept_gross_margin", "name": "Gross_Margin", "category": "financial_metric", "description": "Gross profit margin - profitability after cost of goods sold", "properties": {"formula": "(Revenue - COGS) / Revenue", "typical_range": "0.2-0.6"}},
                        {"id": "concept_debt_ratio", "name": "Debt_Ratio", "category": "financial_metric", "description": "Debt-to-Asset ratio - financial leverage measure", "properties": {"formula": "Total Debt / Total Assets", "typical_range": "0.2-0.6"}},
                        {"id": "concept_current_ratio", "name": "Current_Ratio", "category": "financial_metric", "description": "Current ratio - short-term liquidity measure", "properties": {"formula": "Current Assets / Current Liabilities", "typical_range": "1-3"}},
                        {"id": "concept_eps_growth", "name": "EPS_Growth", "category": "financial_metric", "description": "Earnings Per Share growth rate", "properties": {"formula": "(EPS_t - EPS_t-1) / EPS_t-1", "typical_range": "-0.2-0.5"}},
                    ]
                })
            }
        elif "economic_theories" in prompt or "Economic Theories" in prompt or "经济理论" in prompt:
            return {
                "content": json.dumps({
                    "concepts": [
                        {"id": "concept_value", "name": "Value_Premium", "category": "economic_theory", "description": "Value stocks outperform growth stocks over long term"},
                        {"id": "concept_momentum", "name": "Momentum_Effect", "category": "economic_theory", "description": "Past winners continue to outperform in short term"},
                        {"id": "concept_size", "name": "Size_Effect", "category": "economic_theory", "description": "Small caps historically outperform large caps"},
                    ]
                })
            }
        elif "relation" in prompt.lower() or "关系" in prompt:
            # Check if this is correlation or causal
            if "causal" in prompt.lower():
                return {
                    "content": json.dumps({
                        "has_relation": True,
                        "direction": "source->target",
                        "relation_type": "causal",
                        "mechanism": "Higher profitability (ROE) leads to better market valuation (higher PB)",
                        "confidence": 0.85,
                        "evidence": ["Academic research shows ROE predicts future returns"]
                    })
                }
            else:
                return {
                    "content": json.dumps({
                        "has_relation": True,
                        "direction": "positive",
                        "strength": "strong",
                        "relation_type": "correlated_with",
                        "explanation": "ROE and PE often move together as both reflect profitability expectations",
                        "confidence": 0.82
                    })
                }
        elif "hypothesis" in prompt.lower() or "invest" in prompt.lower() or "假设" in prompt:
            return {
                "content": json.dumps({
                    "hypotheses": [
                        {
                            "statement": "Companies with higher ROE will generate higher future stock returns",
                            "variable_left": "future_return",
                            "operator": "positively_correlated",
                            "variable_right": "roe",
                            "economic_logic": "High ROE indicates efficient capital allocation and strong competitive advantages, which should translate to shareholder value creation",
                            "confidence": 0.82,
                            "supporting_entities": ["ROE", "Return on Equity"],
                            "risks": ["ROE can be manipulated through financial engineering", "Market conditions may override fundamentals"]
                        },
                        {
                            "statement": "Low valuation stocks (low PE) will outperform high valuation stocks",
                            "variable_left": "future_return",
                            "operator": "negatively_correlated",
                            "variable_right": "pe",
                            "economic_logic": "Value premium suggests mean reversion in valuations; low PE stocks are undervalued relative to earnings",
                            "confidence": 0.75,
                            "supporting_entities": ["PE", "Price-to-Earnings"],
                            "risks": ["Low PE may signal financial distress", "Growth stocks can continue outperforming"]
                        },
                        {
                            "statement": "Stocks with improving profitability will outperform",
                            "variable_left": "future_return",
                            "operator": "positively_correlated",
                            "variable_right": "roa_change",
                            "economic_logic": "Improving operational efficiency signals management effectiveness and competitive moat strengthening",
                            "confidence": 0.78,
                            "supporting_entities": ["ROA", "Profitability"],
                            "risks": ["Single quarter improvement may not persist"]
                        },
                        {
                            "statement": "High gross margin companies have lower downside risk",
                            "variable_left": "future_return",
                            "operator": "positively_correlated",
                            "variable_right": "gross_margin",
                            "economic_logic": "High gross margins indicate pricing power and competitive advantages that protect against cost pressures",
                            "confidence": 0.72,
                            "supporting_entities": ["Gross_Margin", "Profitability"],
                            "risks": ["Gross margin can be industry-specific"]
                        },
                        {
                            "statement": "Companies with improving earnings growth outperform",
                            "variable_left": "future_return",
                            "operator": "positively_correlated",
                            "variable_right": "eps_growth",
                            "economic_logic": "Accelerating earnings indicate business momentum and positive fundamental developments",
                            "confidence": 0.80,
                            "supporting_entities": ["EPS_Growth", "Earnings Momentum"],
                            "risks": ["Earnings growth may not translate to stock performance if already priced in"]
                        },
                    ]
                })
            }
        elif "factor_pattern" in prompt or "Factor Patterns" in prompt or "因子模式" in prompt:
            return {
                "content": json.dumps({
                    "concepts": [
                        {"id": "concept_momentum", "name": "Price_Momentum", "category": "factor_pattern", "description": "Relative strength over past N months"},
                        {"id": "concept_reversal", "name": "Short_Term_Reversal", "category": "factor_pattern", "description": "Mean reversion in short-term returns"},
                    ]
                })
            }
        elif "market_institution" in prompt or "Market Institutions" in prompt or "市场制度" in prompt:
            return {
                "content": json.dumps({
                    "concepts": [
                        {"id": "concept_limit", "name": "Price_Limit", "category": "market_institution", "description": "Daily price change limit"},
                        {"id": "concept_t1", "name": "T1_Settlement", "category": "market_institution", "description": "T+1 trading settlement system"},
                    ]
                })
            }
        else:
            return {"content": '{"result": "mock_response"}'}

    @property
    def models(self):
        return MockModels()


class MockModels:
    def list(self):
        return type('obj', (object,), {'data': []})()


__all__ = ['LLMConfig', 'LLMConfigManager', 'create_llm_client', 'MockLLMClient', 'load_llm_config']