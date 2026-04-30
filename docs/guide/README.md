# User Guide

## Installation

### From Source

```bash
# Clone the repository
git clone https://github.com/your-org/kg-agent-quant.git
cd kg-agent-quant

# Install in development mode
pip install -e .

# Or install with all dependencies
pip install -e ".[all]"
```

### Dependencies

**Core Dependencies:**
- numpy >= 1.21.0
- pandas >= 1.5.0
- scikit-learn >= 1.0.0
- scipy >= 1.7.0
- pyyaml >= 6.0

**Optional Dependencies:**
- pyqlib >= 0.9.0 (for QLib integration)
- lightgbm >= 3.0.0 (for LightGBM models)
- openai >= 1.0.0 (for LLM features)
- matplotlib >= 3.5.0 (for visualization)

## Quick Start

### Basic Usage

```python
from kg_quant import KGFeatureGenerator, KGExplainer

# Initialize generator
generator = KGFeatureGenerator(
    kg_dir="data/kg",
    factor_json_path="data/sample/factors_sample.json"
)

# Generate sample data
data = generator._generate_sample_data(n_stocks=50, n_days=100)

# Generate factors
features = generator.generate_kg_features(
    factor_type="quality",
    n_features=10,
    data=data
)

# Explain a factor
explainer = KGExplainer()
explanation = explainer.explain_factor("RANK(TS_MEAN($roe, 20))")
print(explanation.economic_logic)
```

### Factor Evaluation

```python
from kg_quant.evaluation.metrics import FactorEvaluator, compute_ic

# Create evaluator
evaluator = FactorEvaluator(annualization_factor=252)

# Evaluate factor
metrics = evaluator.evaluate_factor(factor_values, future_returns)
print(f"IC: {metrics['ic_mean']:.4f}")
print(f"RankIC: {metrics['rank_ic_mean']:.4f}")
```

## Configuration

### YAML Configuration

Create a `configs/` directory with:

```yaml
# configs/env.yaml
data_root: ./data
qlib_data_dir: /path/to/qlib_data

# configs/data.yaml
datasets:
  csi300:
    qlib_market: csi300
    train_period: [2020, 2021]
    test_period: [2022, 2023]
```

## Factor Types

KG-AgentQuant supports four factor types:

| Type | Description | Typical Indicators |
|------|-------------|-------------------|
| value | Valuation factors | PE, PB, PS |
| quality | Profitability factors | ROE, ROA, Margin |
| momentum | Trend factors | Returns, Price change |
| size | Size factors | Market cap, Float |

## Expression Syntax

KG-AgentQuant uses QLIB-style expressions:

```python
# Time series operators
TS_MEAN($close, 20)    # 20-day moving average
TS_STD($returns, 20)   # 20-day standard deviation
TS_DELTA($roe, 1)      # 1-period change
TS_DELAY($close, 5)    # 5-period lag

# Cross-sectional operators
RANK($roe)             # Cross-sectional rank
ZSCORE($returns)       # Z-score normalization

# Math operators
LOG($volume)           # Natural logarithm
ABS($returns)          # Absolute value

# Logical operators
IF($returns > 0, $roe, -$roe)  # Conditional
```

## Examples

See the `examples/` directory for:

1. `01_factor_generation.py` - Basic factor generation
2. `02_evaluation.py` - Factor evaluation metrics
3. `03_complete_pipeline.py` - End-to-end pipeline

```bash
# Run examples
python examples/01_factor_generation.py
python examples/02_evaluation.py
python examples/03_complete_pipeline.py
```

## CLI Usage

```bash
# Generate factors
kg-factor generate --type quality --n-features 20

# Explain factor
kg-factor explain "RANK(TS_MEAN($roe, 20))"

# Show KG statistics
kg-factor stats
```

## Common Issues

### QLib Not Found
```python
# Install QLib
pip install pyqlib

# Initialize with correct path
import qlib
qlib.init(provider_uri="/path/to/qlib_data", region="cn")
```

### Import Errors
```bash
# Ensure package is installed
pip install -e .

# Or add to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
```