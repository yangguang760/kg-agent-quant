# KG-AgentQuant

<div align="center">

**基于知识图谱与LLM验证的量化因子研究平台**

*使用大语言模型和阶段性独立验证来发现和验证量化alpha因子的多阶段流水线。*

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-green.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-34%20passed-brightgreen.svg)](#)
[![Code Size](https://img.shields.io/badge/code%20size-~3k%20lines-blue.svg)](#)

</div>

---

## 概述

KG-AgentQuant 实现了一个新颖的**多阶段LLM辅助量化因子发现流水线**，具有阶段性独立验证机制。通过在每个中间阶段引入质量控制，解决了传统多阶段LLM流水线中的误差传播问题。

### 核心创新

核心创新是**分离的生成器-评分器架构**和**三个质量控制指标**：

| 指标 | 用途 | 描述 |
|--------|---------|-------------|
| **CSC** | 共识校准分数 | 评估实体层面的关系可信度 |
| **EQ** | 解释质量 | 验证假设的一致性和可解释性 |
| **SC** | 语义一致性 | 确保表达式忠实于假设 |

### 架构图

<p align="center">
  <img src="docs/fig1.jpg" alt="KG-AgentQuant 架构图" width="800"/>
</p>

```
主题 → 实体扩展 → 关系构建 → 假设生成 → 表达式实例化
              ↓            ↓              ↓              ↓
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

## 项目结构

```
kg_agent_quant/
├── src/kg_quant/               # 核心包 (~3200行)
│   ├── core/                  # 核心框架
│   │   ├── config.py          # 配置管理
│   │   └── evaluator.py       # 统一评估器
│   ├── kg/                    # 知识图谱模块
│   │   ├── retriever.py      # KG检索
│   │   ├── feature_generator.py  # 特征生成
│   │   ├── expression_evaluator.py  # QLIB表达式
│   │   ├── explainer.py      # 因子解释
│   │   ├── schema.py        # KG schema定义
│   │   └── consistency_checker.py  # 语义检查
│   ├── llm/                   # LLM生成模块
│   │   ├── config.py         # LLM配置
│   │   └── generators.py     # 概念/关系/假设生成器
│   ├── factor/               # 因子解析
│   │   └── ast_parser.py     # AST解析器
│   └── evaluation/           # 评估指标
│       └── metrics.py        # IC, RankIC, ARR等
├── data/
│   ├── kg/                   # 知识图谱数据
│   │   ├── layer1_concepts.json    # 64个金融实体
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
