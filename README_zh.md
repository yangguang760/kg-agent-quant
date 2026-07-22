# KG-AgentQuant

<div align="center">

**基于知识图谱与多智能体验证的量化因子研究平台**

*使用大语言模型和审议共识协议来发现和验证量化alpha因子的多阶段流水线。*

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-green.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-34%20passed-brightgreen.svg)](#)

</div>

---

## 概述

KG-AgentQuant 实现了一个**多智能体验证架构**。当独立的评分器LLM对产物质量存在分歧时，它们不会简单平均（我们发现简单平均在55%的分歧案例中反而降低准确性），而是进入**审议共识协议**——带推理交换的结构化多轮辩论。

### 核心创新

**阶段性验证 + 审议共识。** 三个验证门（CSC、EQ、SC）各使用**双独立评分器**。分歧时（σ > 0.2），审议而非投票：

| 门 | 检查内容 | 评分器对 | 收敛率 |
|------|--------|-------------|-------------|
| **CSC** | 关系可信度 | GLM-5 + Kimi-K2.5 | 93% |
| **EQ** | 假设一致性 | GLM-5 + DeepSeek-V4-Pro | 91% |
| **SC** | 因子-假设忠实度 | DeepSeek-V4-Pro × 2 | 50% |

### 架构

```
主题 → 实体 → [CSC门] → 关系 → [EQ门] → 假设 → [SC门] → 因子 → 组合
                 │                    │                  │
            双评分器              双评分器            双评分器
            一致? → 通过          一致? → 通过        一致? → 通过
            分歧? → 审议          分歧? → 审议        分歧? → 审议
          (层级1)       (层级2)       (假设)        (因子)
                       [CSC过滤]    [EQ过滤]     [SC过滤]
```

## 特性

- **LLM驱动生成**：使用大语言模型生成金融概念、关系和假设
- **三层知识图谱**：结构化的金融概念、关系和LLM验证的证据
- **QLIB风格表达式评估器**：30+操作符，包括RANK、TS_MEAN、TS_STD等
- **因子可解释性**：从主题到可执行因子的完整可追溯性
- **语义一致性检查**：验证假设-表达式的一致性
- **综合指标**：IC、RankIC、ARR、MDD、IR、Calmar比率

## LLM集成

KG-AgentQuant支持多种LLM提供商来生成金融知识：

```python
from kg_quant.llm import load_llm_config, ConceptGenerator

# 从本地配置文件加载
config = load_llm_config("yunnetC")  # 使用 gpt-5.3-codex

# 生成金融概念
concept_gen = ConceptGenerator(config=config, language="en")
concepts = concept_gen.generate(topic="financial_metrics", min_concepts=20)

# 生成投资假设
from kg_quant.llm import HypothesisGenerator
hyp_gen = HypothesisGenerator(config=config)
hypotheses = hyp_gen.generate(entities=concepts, min_hypotheses=10)
```

### 支持的提供商

| 提供商 | 模型 | 说明 |
|----------|--------|-------|
| yunnetC | gpt-5.3-codex | 高性价比 |
| yunnet | claude-opus-4-6 | 高质量 |
| DeepSeek | deepseek-chat | 成本效益 |
| OpenAI | GPT-4o, GPT-4o-mini | 设置环境变量 |
| Mock | - | 仅用于测试 |

### API配置

在 `config/llm.json` 中配置您的API密钥：

```python
from kg_quant.llm import load_llm_config

config = load_llm_config("yunnetC")  # gpt-5.3-codex
config = load_llm_config("deepseek")  # deepseek-chat
```

## 安装

```bash
# 从源码安装
git clone https://github.com/YOUR_ORG/kg-agent-quant.git
cd kg-agent-quant
pip install -e .

# 安装所有依赖
pip install -e ".[all]"
```

## 快速开始

### 生成Alpha因子

```python
from kg_quant import KGFeatureGenerator, KGExplainer
import pandas as pd

# 初始化生成器
generator = KGFeatureGenerator(
    kg_dir="data/kg",
    factor_json_path="data/sample/factors_sample.json"
)

# 生成样本数据
data = generator._generate_sample_data(n_stocks=50, n_days=100)

# 生成质量因子
features = generator.generate_kg_features(
    factor_type="quality",
    n_features=10,
    data=data
)

# 解释因子
explainer = KGExplainer()
explanation = explainer.explain_factor("RANK(TS_MEAN($roe, 20))")

print(f"逻辑: {explanation.economic_logic}")
print(f"置信度: {explanation.explanation_confidence:.2f}")
```

### 评估因子

```python
from kg_quant.evaluation.metrics import FactorEvaluator

evaluator = FactorEvaluator(annualization_factor=252)

# 评估因子质量
metrics = evaluator.evaluate_factor(factor_values, future_returns)

print(f"IC: {metrics['ic_mean']:.4f}")
print(f"RankIC: {metrics['rank_ic_mean']:.4f}")
print(f"ICIR: {metrics['icir']:.4f}")
```

## 表达式语法

KG-AgentQuant使用QLIB风格的表达式：

```python
# 时间序列操作符
TS_MEAN($close, 20)    # 20日移动平均
TS_STD($returns, 20)   # 20日滚动标准差
TS_DELTA($roe, 1)      # 1期变化
TS_DELAY($close, 5)    # 5期滞后

# 截面操作符
RANK($roe)             # 截面排名
ZSCORE($returns)      # Z-score标准化

# 逻辑操作符
IF($returns > 0, $roe, -$roe)  # 条件表达式
```

## 因子类型

| 类型 | 描述 | 示例 |
|------|-------------|---------|
| `quality` | 盈利质量因子 | ROE, ROA, 利润率 |
| `value` | 估值因子 | PE, PB, PS |
| `momentum` | 趋势因子 | 收益率, 价格变动 |
| `size` | 规模因子 | 市值 |

## 示例

```bash
# 运行所有示例
python examples/01_factor_generation.py  # 因子生成
python examples/02_evaluation.py          # 因子评估
python examples/03_complete_pipeline.py   # 完整流水线
python examples/04_llm_generation.py     # LLM生成

# 运行测试
pytest tests/ -v
```

## 多 Agent 验证（新功能）

本包包含 **分布式多 Agent 验证框架**：

```python
from kg_quant.agents import AgentHarness, build_default_harness_config
from kg_quant.agents import AgentRole, Artifact

# 创建包含 Generator + CSC/EQ/SC Scorer 的 Agent 拓扑
config = build_default_harness_config()
harness = AgentHarness(config=config)
harness.start_session()
harness.register_default_agents()

# 将 artifact 路由通过验证
artifact = Artifact(artifact_type="relation", 
    content={"head": "ROE", "tail": "PE"}, 
    reasoning_trace="ROE与PE通过盈利相关联...")
artifact.add_provenance(AgentRole.GENERATOR, "generated")

# 质量门检查
passed, reasons = harness.gate_enforcer.check_all(artifact)
```

详见 `examples/demo_agent_harness.py`。

## 项目结构

```
kg_agent_quant/
├── src/kg_quant/               # 核心包 (~5000行)
│   ├── core/                  # 核心框架
│   ├── kg/                    # 知识图谱模块
│   ├── agents/                # 多Agent验证（新增）
│   │   ├── protocol.py       # Agent角色、消息、A2A协议
│   │   ├── deliberation.py   # 多轮审议共识引擎
│   │   ├── feedback_loop.py  # Agentic自修正闭环
│   │   └── harness.py        # Agent注册、消息路由、质量门
│   ├── llm/                   # LLM生成模块
│   ├── factor/               # 因子解析
│   └── evaluation/           # 评估指标
├── examples/                  # 示例脚本
│   ├── demo_agent_harness.py    # Agent拓扑演示
│   ├── run_deliberation_live.py # 真实多轮审议实验
│   ├── run_feedback_loop_live.py # 反馈闭环实验
│   └── run_heterogeneity_study.py # 异质性研究
├── data/
│   ├── kg/                   # 知识图谱数据
│   │   └── layer2_relations_final.json  # 856个关系
│   └── sample/               # 样本数据
│       └── factors_sample.json  # 10个样本因子
├── config/                    # LLM配置文件
├── examples/                  # 示例脚本 (1-4)
├── docs/                      # 文档
├── tests/                    # 测试套件 (34个测试)
└── pyproject.toml            # 项目配置
```

## 知识图谱

内置知识图谱包含：

- **64个金融实体**：ROE、PE、PB、ROA、利润率等
- **856个关系**：CORRELATED_WITH、THEORY_SUPPORTS等
- **6种关系类型**：经过质量验证的关系

## 文档

- [用户指南](docs/guide/README.md) - 入门和教程
- [API参考](docs/api/README.md) - 完整API文档
- [架构设计](docs/ARCHITECTURE.md) - 系统设计
- [QLib操作符](docs/qlib_operators.md) - 表达式语法

## 许可证

MIT许可证 - 详见 [LICENSE](LICENSE)

---

<div align="center">

**为量化金融研究而构建 ❤️**

</div>
