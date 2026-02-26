# ANTI-BIAS FRAMEWORK – Integration Guide

## Overview

The Anti-Bias Framework has been successfully integrated into `BITCOIN4Traders`. It consists of four main modules:

1. **Walk-Forward CV** (`src/validation/antib.py`)
2. **Transaction Costs**ias_walkforward (`src/costs/antibias_costs.py`)
3. **Reward Functions** (`src/reward/antibias_rewards.py`)
4. **Validator** (`src/evaluation/antibias_validator.py`)

## Integration Status

### ✅ Fully Integrated:

- **src/validation/antibias_walkforward.py** – Purged Walk-Forward CV, PurgedScaler, LeakDetector
- **src/costs/antibias_costs.py** – Realistic transaction costs (Spot/Futures)
- **src/reward/antibias_rewards.py** – Risk-adjusted reward functions
- **src/evaluation/antibias_validator.py** – Statistical validation (CPCV, Permutation, DSR)

### ✅ Integrated into Existing Systems:

1. **realistic_trading_env.py**
   - New reward functions (SharpeIncrement, CostAware, RegimeAware)
   - Anti-Bias Cost Engine optionally available
   - Configuration via `TradingEnvConfig`

2. **performance_calculator.py**
   - Added `validate_with_antibias()` method
   - Full ValidationReport available

3. **walkforward_engine.py**
   - Added `create_purged_splits()` method for Purged CV

## Usage

### 1. Purged Walk-Forward CV

```python
from src.validation.antibias_walkforward import PurgedWalkForwardCV, WalkForwardConfig

# Configuration
cv = PurgedWalkForwardCV(WalkForwardConfig(
    n_splits=5,
    feature_lookback=100,  # Max feature window
    embargo_pct=0.01,      # 1% embargo
    holdout_pct=0.15,      # 15% holdout
    purge=True,
))

# Create splits
folds, holdout_idx = cv.split(n_samples=len(df))

for fold in folds:
    X_train = X[fold.train_idx]
    X_test = X[fold.test_idx]
    
    # CRITICAL: Fit scaler only on Train!
    scaler = PurgedScaler("zscore")
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)  # ONLY transform!
    
    # Training...
```

### 2. Realistic Transaction Costs

```python
from src.costs.antibias_costs import (
    TransactionCostEngine, CostConfig,
    MarketType, Timeframe, OrderType
)

# Create engine
engine = TransactionCostEngine(CostConfig(
    market_type=MarketType.FUTURES,
    timeframe=Timeframe.H1,
    order_type=OrderType.MARKET,
    holding_bars=4,
))

# Calculate costs
cost = engine.total_cost(
    price=30_000,
    quantity=0.1,
    adv=500_000_000,  # 24h Volume
    volatility=0.02,
)

print(cost.total_roundtrip)  # ~0.12%
print(cost.min_required_edge)  # ~0.18%
```

### 3. Risk-Adjusted Rewards

```python
from src.reward.antibias_rewards import (
    SharpeIncrementReward,
    CostAwareReward,
    RegimeAwareReward,
    RegimeState,
)

# Option 1: Sharpe-based Reward
reward_fn = SharpeIncrementReward(window=50)

# Option 2: Cost-aware Reward (recommended)
reward_fn = CostAwareReward(
    lambda_cost=2.0,    # Churning penalty
    lambda_draw=5.0,    # Drawdown penalty
    cost_rate=0.001,    # 0.1% per trade
)

# Option 3: Regime-aware Reward (best for multi-TF)
reward_fn = RegimeAwareReward(
    lambda_cost=2.0,
    lambda_draw=3.0,
    lambda_regime=0.5,  # Regime congruence bonus
)

# Set regime (for RegimeAwareReward)
reward_fn.set_regime(RegimeState(
    regime=2,           # 0=Bear, 1=Neutral, 2=Bull
    vol_regime=0,       # 0=Low, 1=High
    trend_strength=0.7,
))

# Calculate reward
reward = reward_fn.compute(
    pnl=50.0,
    position=1.0,
    prev_position=0.0,
    equity=10_500,
    cost_this_bar=10.0,
)
```

### 4. Statistical Validation

```python
from src.evaluation.antibias_validator import BacktestValidator

# Create validator
validator = BacktestValidator(
    n_cpcv_splits=6,
    n_permutations=1000,
    n_trials_tested=20,  # For DSR
)

# Run validation
report = validator.validate(returns_array, positions_array)

# Result
print(report)
# ════════════════════════════════════════════════════════════
#   BACKTEST VALIDATION REPORT
# ════════════════════════════════════════════════════════════
#   Overall: ✅ PASSES ALL CHECKS  /  ❌ FAILS
#
#   [2] CPCV Stability: mean_sharpe=0.85, stability=83.3%
#   [3] Permutation Test: p=0.003 z=2.88 ✅ SIGNIFICANT
#   [4] Deflated Sharpe:  DSR=0.89 ✅ ACCEPTABLE
#   [5] Min. Track Record: 312 bars @ 1h = 13 days

# Check before live trading
assert report.passes_all, "System failed validation!"
```

### 5. Integration into Trading Environment

```python
from src.environment.realistic_trading_env import TradingEnvConfig

# Environment with Anti-Bias features
config = TradingEnvConfig(
    initial_capital=100000,
    transaction_cost_bps=5.0,
    
    # Anti-Bias Settings
    use_antibias_rewards=True,
    reward_type="cost_aware",  # or "sharpe", "regime_aware"
    use_antibias_costs=True,
)

env = RealisticTradingEnv(price_data, features, config)
```

### 6. Performance Calculator with Validation

```python
from src.backtesting.performance_calculator import PerformanceCalculator

calc = PerformanceCalculator()

# Calculate standard metrics
metrics = calc.calculate_from_equity_curve(equity_series)

# Anti-Bias validation
report = calc.validate_with_antibias(
    returns=returns_array,
    positions=positions_array,
    n_cpcv_splits=6,
    n_permutations=1000,
)

if report:
    calc.print_validation_report(report)
```

## Break-Even Analysis

```python
from src.costs.antibias_costs import BreakEvenAnalyzer

# Shows realistically how hard profitability is
print(BreakEvenAnalyzer.analyze_all_scenarios())

# ════════════════════════════════════════════════════════════════════════
#   BREAK-EVEN ANALYSIS
#   Symbol: BTCUSDT  |  Order: $10,000  |  ADV: $500M
# ════════════════════════════════════════════════════════════════════════
#   Market   TF    Type    RT Cost  Min Edge  Trades/d@BE
# ────────────────────────────────────────────────────────────────────────
#   SPOT     1m    market  0.2500%  0.3750%        12.0
#   SPOT     5m    market  0.1800%  0.2700%        16.7
#   FUTURES  H1    market  0.1200%  0.1800%        25.0
#   FUTURES  H1    limit   0.0800%  0.1200%        37.5
#   FUTURES  H4    limit   0.0500%  0.0750%        60.0
# ════════════════════════════════════════════════════════════════════════
```

## Tests

Run tests:

```bash
cd /home/hp17/Tradingbot/Quantrivo/BITCOIN4Traders
python -m pytest tests/test_antibias_integration.py -v
```

## Checklist Before Go-Live

```python
from src.evaluation.antibias_validator import BacktestValidator

# 1. CPCV Stability > 70%
# 2. Permutation Test p < 0.05
# 3. Deflated Sharpe > 0.64
# 4. Min Track Record reached

validator = BacktestValidator(
    n_cpcv_splits=6,
    n_permutations=1000,
    n_trials_tested=50,
)

report = validator.validate(returns, positions)

assert report.passes_all, "❌ DO NOT deploy live!"
print("✅ System ready for live trading")
```

## Most Important Rules

1. **PurgedScaler**: Always fit only on Train, never on Test/Holdout
2. **Walk-Forward**: At least 15% Holdout, never touch
3. **Costs**: Model realistically (0.08-0.12% round-trip on 1h)
4. **Validation**: All 4 tests must pass before live trading
5. **Reward**: Use Sharpe or CostAware, never naive return

## File Structure

```
BITCOIN4Traders/
├── src/
│   ├── validation/
│   │   ├── __init__.py
│   │   └── antibias_walkforward.py    ← Purged CV + Scaler + LeakDetector
│   ├── costs/
│   │   ├── __init__.py
│   │   └── antibias_costs.py          ← Transaction Cost Engine
│   ├── reward/
│   │   ├── __init__.py
│   │   └── antibias_rewards.py        ← Risk-Adjusted Rewards
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── antibias_validator.py      ← CPCV + Permutation + DSR
│   ├── environment/
│   │   └── realistic_trading_env.py   ← Integrated Anti-Bias Rewards
│   └── backtesting/
│       ├── walkforward_engine.py      ← Purged CV methods
│       └── performance_calculator.py  ← Anti-Bias validation
└── tests/
    └── test_antibias_integration.py   ← Integration Tests
```

## Support

For questions or issues with integration:
- Run tests: `pytest tests/test_antibias_integration.py -v`
- Enable logging: `logging.getLogger("antibias").setLevel(logging.DEBUG)`
- Project path: `/home/hp17/Tradingbot/Quantrivo/BITCOIN4Traders`
