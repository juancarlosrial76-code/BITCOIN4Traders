# ANTI-BIAS FRAMEWORK – Integration Guide

## Übersicht

Das Anti-Bias Framework wurde erfolgreich in `BITCOIN4Traders` integriert. Es umfasst vier Hauptmodule:

1. **Walk-Forward CV** (`src/validation/antibias_walkforward.py`)
2. **Transaction Costs** (`src/costs/antibias_costs.py`)
3. **Reward Functions** (`src/reward/antibias_rewards.py`)
4. **Validator** (`src/evaluation/antibias_validator.py`)

## Integration Status

### ✅ Vollständig integriert:

- **src/validation/antibias_walkforward.py** – Purged Walk-Forward CV, PurgedScaler, LeakDetector
- **src/costs/antibias_costs.py** – Realistische Transaktionskosten (Spot/Futures)
- **src/reward/antibias_rewards.py** – Risikobereinigte Reward-Funktionen
- **src/evaluation/antibias_validator.py** – Statistische Validierung (CPCV, Permutation, DSR)

### ✅ In bestehende Systeme integriert:

1. **realistic_trading_env.py**
   - Neue Reward-Funktionen (SharpeIncrement, CostAware, RegimeAware)
   - Anti-Bias Cost Engine optional verfügbar
   - Konfiguration über `TradingEnvConfig`

2. **performance_calculator.py**
   - `validate_with_antibias()` Methode hinzugefügt
   - Vollständiger ValidationReport verfügbar

3. **walkforward_engine.py**
   - `create_purged_splits()` Methode für Purged CV

## Verwendung

### 1. Purged Walk-Forward CV

```python
from src.validation.antibias_walkforward import PurgedWalkForwardCV, WalkForwardConfig

# Konfiguration
cv = PurgedWalkForwardCV(WalkForwardConfig(
    n_splits=5,
    feature_lookback=100,  # Max. Feature-Window
    embargo_pct=0.01,      # 1% Embargo
    holdout_pct=0.15,      # 15% Holdout
    purge=True,
))

# Splits erstellen
folds, holdout_idx = cv.split(n_samples=len(df))

for fold in folds:
    X_train = X[fold.train_idx]
    X_test = X[fold.test_idx]
    
    # KRITISCH: Scaler nur auf Train fitten!
    scaler = PurgedScaler("zscore")
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)  # NUR transform!
    
    # Training...
```

### 2. Realistische Transaktionskosten

```python
from src.costs.antibias_costs import (
    TransactionCostEngine, CostConfig,
    MarketType, Timeframe, OrderType
)

# Engine erstellen
engine = TransactionCostEngine(CostConfig(
    market_type=MarketType.FUTURES,
    timeframe=Timeframe.H1,
    order_type=OrderType.MARKET,
    holding_bars=4,
))

# Kosten berechnen
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

# Option 1: Sharpe-basierter Reward
reward_fn = SharpeIncrementReward(window=50)

# Option 2: Kosten-bewusster Reward (empfohlen)
reward_fn = CostAwareReward(
    lambda_cost=2.0,    # Churning-Strafe
    lambda_draw=5.0,    # Drawdown-Strafe
    cost_rate=0.001,    # 0.1% pro Trade
)

# Option 3: Regime-bewusster Reward (beste für Multi-TF)
reward_fn = RegimeAwareReward(
    lambda_cost=2.0,
    lambda_draw=3.0,
    lambda_regime=0.5,  # Regime-Kongruenz-Bonus
)

# Regime setzen (für RegimeAwareReward)
reward_fn.set_regime(RegimeState(
    regime=2,           # 0=Bear, 1=Neutral, 2=Bull
    vol_regime=0,       # 0=Low, 1=High
    trend_strength=0.7,
))

# Reward berechnen
reward = reward_fn.compute(
    pnl=50.0,
    position=1.0,
    prev_position=0.0,
    equity=10_500,
    cost_this_bar=10.0,
)
```

### 4. Statistische Validierung

```python
from src.evaluation.antibias_validator import BacktestValidator

# Validator erstellen
validator = BacktestValidator(
    n_cpcv_splits=6,
    n_permutations=1000,
    n_trials_tested=20,  # Für DSR
)

# Validierung durchführen
report = validator.validate(returns_array, positions_array)

# Ergebnis
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

# Check vor Live-Trading
assert report.passes_all, "System besteht Validierung nicht!"
```

### 5. Integration in Trading Environment

```python
from src.environment.realistic_trading_env import TradingEnvConfig

# Environment mit Anti-Bias Features
config = TradingEnvConfig(
    initial_capital=100000,
    transaction_cost_bps=5.0,
    
    # Anti-Bias Settings
    use_antibias_rewards=True,
    reward_type="cost_aware",  # oder "sharpe", "regime_aware"
    use_antibias_costs=True,
)

env = RealisticTradingEnv(price_data, features, config)
```

### 6. Performance Calculator mit Validierung

```python
from src.backtesting.performance_calculator import PerformanceCalculator

calc = PerformanceCalculator()

# Standard-Metriken berechnen
metrics = calc.calculate_from_equity_curve(equity_series)

# Anti-Bias Validierung
report = calc.validate_with_antibias(
    returns=returns_array,
    positions=positions_array,
    n_cpcv_splits=6,
    n_permutations=1000,
)

if report:
    calc.print_validation_report(report)
```

## Break-Even Analyse

```python
from src.costs.antibias_costs import BreakEvenAnalyzer

# Zeigt realistisch wie schwer Profitabilität ist
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

Tests ausführen:

```bash
cd /home/hp17/Tradingbot/BITCOIN4Traders
python -m pytest tests/test_antibias_integration.py -v
```

## Checkliste vor Go-Live

```python
from src.evaluation.antibias_validator import BacktestValidator

# 1. CPCV Stability > 70%
# 2. Permutation Test p < 0.05
# 3. Deflated Sharpe > 0.64
# 4. Min Track Record erreicht

validator = BacktestValidator(
    n_cpcv_splits=6,
    n_permutations=1000,
    n_trials_tested=50,
)

report = validator.validate(returns, positions)

assert report.passes_all, "❌ NICHT live deployen!"
print("✅ System bereit für Live-Trading")
```

## Wichtigste Regeln

1. **PurgedScaler**: Immer nur auf Train fitten, nie auf Test/Holdout
2. **Walk-Forward**: Mindestens 15% Holdout, nie anfassen
3. **Kosten**: Realistisch modellieren (0.08-0.12% Round-Trip auf 1h)
4. **Validierung**: Alle 4 Tests müssen passen vor Live-Trading
5. **Reward**: Sharpe oder CostAware verwenden, nie naiver Return

## Dateistruktur

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
│   │   └── realistic_trading_env.py   ← Integriert Anti-Bias Rewards
│   └── backtesting/
│       ├── walkforward_engine.py      ← Purged CV Methoden
│       └── performance_calculator.py  ← Anti-Bias Validierung
└── tests/
    └── test_antibias_integration.py   ← Integration Tests
```

## Support

Bei Fragen oder Problemen mit der Integration:
- Tests laufen lassen: `pytest tests/test_antibias_integration.py -v`
- Logging aktivieren: `logging.getLogger("antibias").setLevel(logging.DEBUG)`
- Projektpfad: `/home/hp17/Tradingbot/BITCOIN4Traders`
