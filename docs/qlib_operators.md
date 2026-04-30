# QLib Operators Reference

## Time Series Operators

### TS_Mean
```python
TS_MEAN($close, 20)  # 20-day moving average of close price
```

### TS_Std
```python
TS_STD($returns, 20)  # 20-day rolling standard deviation
```

### TS_Delta
```python
TS_DELTA($roe, 1)  # 1-period change in ROE
```

### TS_Delay / REF
```python
TS_DELAY($close, 5)  # Close price 5 periods ago
REF($close, 5)      # Same as TS_DELAY
```

### TS_Sum
```python
TS_SUM($returns, 20)  # 20-day sum of returns
```

### TS_Rank
```python
TS_RANK($volume, 10)  # Rank of volume in past 10 periods
```

### TS_Max / TS_Min
```python
TS_MAX($close, 20)  # Maximum close price in past 20 days
TS_MIN($close, 20)  # Minimum close price in past 20 days
```

## Cross-Sectional Operators

### Rank
```python
RANK($roe)  # Cross-sectional rank of ROE
```

### CS_Rank
```python
CS_RANK($pe)  # Cross-sectional rank within each datetime
```

### ZScore
```python
ZSCORE($returns)  # Z-score normalization
```

### Scale
```python
SCALE($roe)  # Scale to [-1, 1] range
```

## Math Operators

### Basic Math
```python
$close - $open     # Difference
$close * $volume   # Product
$close / $pe       # Division
LOG($volume)       # Natural log
EXP($roe)          # Exponential
ABS($returns)      # Absolute value
SQRT($volume)      # Square root
POW($roe, 2)       # Power
```

### Log/Exp
```python
LOG($volume)    # Natural logarithm
LOG1P($volume)   # log(1 + x)
EXP($roe)        # e^x
```

## Logical Operators

### Comparison
```python
$returns > 0      # Greater than
$roe < 0.1       # Less than
$pe >= 10        # Greater than or equal
$pb <= 3         # Less than or equal
```

### Conditional
```python
IF($returns > 0, $roe, -$roe)  # If returns > 0, use ROE, else use -ROE
WHERE($returns > 0, $roe, -$roe)  # Same as IF
```

## Sector Operators

### Sector Mean/Std
```python
SECTOR_MEAN($roe)  # Mean of ROE across sector (simplified)
SECTOR_STD($returns)  # Standard deviation across sector
SECTOR_RANK($pe)    # Rank within sector
```

## Examples

### Value Factor
```python
RANK(1/$pe)  # Inverse PE for value screening
```

### Quality Factor
```python
RANK(TS_MEAN($roe, 20))  # Rank of 20-day average ROE
```

### Momentum Factor
```python
RANK(TS_SUM($returns, 20))  # Rank of 20-day cumulative returns
```

### Low Volatility Factor
```python
RANK(-TS_STD($returns, 20))  # Rank of negative volatility
```

### Combined Factor
```python
RANK(TS_MEAN($roe, 20)) + RANK(1/$pe)  # Combined quality and value
```