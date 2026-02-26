# BITCOIN4Traders — Backtesting Guide

## Table of Contents

1. [Walk-Forward Validation: Purpose and Methodology](#1-walk-forward-validation)
2. [Stress Testing: What Scenarios Are Tested](#2-stress-testing)
3. [Performance Metrics Explained](#3-performance-metrics)
4. [Anti-Bias Validation Framework](#4-anti-bias-validation)
5. [Purged Walk-Forward Cross-Validation](#5-purged-walk-forward-cv)
6. [How to Interpret Backtesting Results](#6-interpreting-results)
7. [Common Backtesting Pitfalls](#7-common-pitfalls)
8. [When Is a Strategy Ready for Live Trading?](#8-readiness-criteria)

---

## 1. Walk-Forward Validation

### Purpose

Walk-forward validation is the most reliable method for evaluating a trading strategy on historical data. It directly simulates what will happen in live trading: the strategy is trained on past data and tested on data it has never seen.

### Why Not Just Split Train/Test Once?

A single train/test split has a critical weakness: the test set covers only one market period. If that period happens to be similar to the training data, the results are overly optimistic. If it differs significantly, results are overly pessimistic. Neither tells you how the strategy will behave across different market regimes.

Walk-forward uses many sequential splits, covering multiple market conditions.

### Methodology

```
Full dataset (3 years of hourly BTC/USDT data = 26,280 bars)

Window 1:
  Train: bars 1–8,760   (Year 1)         │ IS Sharpe: 1.84
  Test:  bars 8,761–11,688 (3 months)    │ OOS Sharpe: 1.21

Window 2:
  Train: bars 721–9,480   (rolling)      │ IS Sharpe: 2.01
  Test:  bars 9,481–12,408               │ OOS Sharpe: 1.35

Window 3:
  Train: bars 1,441–10,200               │ IS Sharpe: 1.73
  Test:  bars 10,201–13,128              │ OOS Sharpe: 0.94

...continues until all data is covered...

Average OOS Sharpe: 1.17
Average Degradation: 38%   (IS - OOS) / IS
```

The key metrics are:
1. **Average OOS Sharpe** — the honest estimate of live performance
2. **Degradation ratio** — how much performance decays on unseen data
3. **OOS Max Drawdown** — worst drawdown in test periods
4. **Consistency** — are all windows positive, or is performance driven by one period?

### WalkForwardEngine Configuration

```python
from src.backtesting.walkforward_engine import WalkForwardEngine, WalkForwardConfig

config = WalkForwardConfig(
    train_window_days=365,      # 1 year training per window
    test_window_days=90,        # 3 months test per window
    step_days=30,               # Advance 1 month between windows
    train_iterations=200,       # Training iterations per window
    results_dir="data/backtests",
    min_trades=10,              # Minimum trades for a valid test
    max_drawdown_threshold=0.30 # Flag windows with DD > 30%
)

engine = WalkForwardEngine(agent, env_config, config)
results = engine.run(df)
results.save("data/backtests/run_2025_01_15.json")
```

---

## 2. Stress Testing

### Purpose

Walk-forward validation tests the strategy on real historical data. Stress testing tests it on **worst-case synthetic scenarios** that may not have appeared in history but are physically possible.

The `stress_tester.py` module runs the strategy through six scenario categories.

### Scenario 1: Extreme Volatility

```
Scenario: Volatility multiplied by 3× for 30 days
Example:  Normal daily vol 2% → Stress vol 6%
This simulates: COVID March 2020, FTX collapse, flash crashes
```

Checks: Does the strategy survive without blowing the account? Does it reduce position size automatically via ATR stops?

### Scenario 2: Prolonged Trend (Bull/Bear)

```
Scenario: Smooth trend of +50% or −50% over 60 days
This simulates: 2021 bull run, 2022 bear market
```

Checks: Does a mean-reversion strategy avoid trading against a strong trend? Does a trend-following strategy capture most of the move?

### Scenario 3: Sideways Chop (Range-Bound)

```
Scenario: Price oscillates ±2% around a fixed level for 60 days
This simulates: Accumulation phases, low-conviction markets
```

Checks: Does the strategy avoid excessive trading when there is no directional edge? High Profit Factor still required in chop.

### Scenario 4: Flash Crash and Recovery

```
Scenario: −30% price drop in 15 minutes, full recovery in 2 hours
This simulates: May 2021 BTC flash crash, exchange liquidations
```

Checks: Are stop losses triggered appropriately? Does the circuit breaker engage? What is the actual fill price during the crash (slippage simulation)?

### Scenario 5: Data Gaps (Exchange Downtime)

```
Scenario: 6-hour data gap followed by resume at ±10% price
This simulates: Exchange maintenance, API failures
```

Checks: Does the system handle missing bars gracefully? Are open positions managed during gaps?

### Scenario 6: Transaction Cost Amplification

```
Scenario: 5× normal transaction costs (slippage + fees)
This simulates: Low-liquidity periods, wide spreads, market impact
```

Checks: Does the strategy remain profitable when costs are high? Strategies that only work with zero/minimal costs are rejected.

### Running Stress Tests

```python
from src.backtesting.stress_tester import StressTester, StressConfig

stress = StressTester()

results = stress.run_all_scenarios(agent, df)

# Print summary
for scenario, metrics in results.items():
    status = "PASS" if metrics["survived"] else "FAIL"
    print(f"{scenario:30s} {status:6s} Sharpe: {metrics['sharpe']:.2f} "
          f"MaxDD: {metrics['max_drawdown']:.1%}")
```

---

## 3. Performance Metrics Explained

### Sharpe Ratio

**Formula:**

```
Sharpe = (mean_return - risk_free_rate) / std(returns)
         annualised = daily_sharpe × sqrt(252)   (trading days)
         or hourly_sharpe × sqrt(8760)           (hourly bars)
```

**Interpretation:**

| Sharpe | Interpretation |
|---|---|
| < 0 | Strategy loses money on a risk-adjusted basis |
| 0.0–0.5 | Poor; not worth trading |
| 0.5–1.0 | Acceptable; requires careful validation |
| 1.0–2.0 | Good; institutional-grade |
| > 2.0 | Excellent; likely overfitted — scrutinise carefully |

**Limitation:** Sharpe treats upside and downside volatility equally. A strategy that has occasional large gains but consistent small losses can have a low Sharpe despite being profitable.

**In this system:** OOS Sharpe ≥ 0.5 is the minimum promotion criterion.

### Calmar Ratio

**Formula:**

```
Calmar = Annualised Return / |Maximum Drawdown|

Example: +24% annual return, −15% max drawdown → Calmar = 24/15 = 1.6
```

**Interpretation:** How much return per unit of drawdown pain. Calmar penalises strategies that achieve returns via large drawdowns.

| Calmar | Interpretation |
|---|---|
| < 0.5 | Poor risk/return trade-off |
| 0.5–1.0 | Acceptable |
| 1.0–2.0 | Good |
| > 2.0 | Excellent |

**In this system:** Calmar ≥ 0.3 minimum for promotion.

### Sortino Ratio

**Formula:**

```
Sortino = (mean_return - risk_free_rate) / downside_deviation

downside_deviation = std(returns[returns < 0]) × sqrt(252)
```

**Why better than Sharpe:** Only penalises downside volatility. Upside volatility (big wins) is not a problem — Sharpe penalises it anyway.

A strategy with consistent small gains and occasional large gains has a high Sortino but moderate Sharpe. Sortino correctly identifies this as good.

### Maximum Drawdown

**Formula:**

```
For each point t in equity curve:
  peak = max(equity[0..t])
  drawdown[t] = (peak - equity[t]) / peak

Max Drawdown = max(drawdown)
```

**Example:**

```
Equity history: $10,000 → $12,000 → $11,000 → $9,500 → $11,500

Peak so far:    $10,000   $12,000   $12,000   $12,000   $12,000
Drawdown:            0%        0%     8.3%     20.8%      4.2%

Max Drawdown = 20.8%  (from $12,000 peak to $9,500 trough)
```

**In this system:** Max DD ≤ 25% required for promotion.

### Profit Factor

**Formula:**

```
Profit Factor = Gross Profit / |Gross Loss|
              = sum(winning trades) / |sum(losing trades)|
```

**Interpretation:**

| Profit Factor | Interpretation |
|---|---|
| < 1.0 | Losing strategy |
| 1.0–1.3 | Marginal; not worth the risk |
| 1.3–2.0 | Good |
| > 2.0 | Excellent (or overfitted — check) |

**Why this matters:** A strategy with a 60% win rate but small wins and large losses can have a Profit Factor < 1.0 and lose money overall. PF directly measures whether the strategy makes more than it costs.

**In this system:** Profit Factor ≥ 1.3 minimum for promotion.

---

## 4. Anti-Bias Validation Framework

The Anti-Bias Framework is a set of mechanisms that reduce the risk of deploying a strategy that performed well only due to data snooping or other biases.

### Bias Type 1: Lookahead Bias

**What it is:** Using future data in a calculation that should only use past data.

**Examples:**
- Fitting a scaler on the entire dataset (including test data) before training
- Using end-of-bar price as entry price (in reality you can only enter at bar open)
- Calculating stop loss using the next bar's low (which you don't know yet)

**How this system prevents it:**

```python
# WRONG (introduces lookahead bias):
engine.fit_transform(entire_dataset)   # Scaler sees future data

# CORRECT (no lookahead):
train_features = engine.fit_transform(train_split)   # Fit only on train
test_features  = engine.transform(test_split)         # Apply saved stats
```

### Bias Type 2: Survivorship Bias

**What it is:** Testing only on assets that survived (e.g., only coins that are still trading). Coins that crashed or were delisted are excluded, making results look better.

**How this system addresses it:** BTC/USDT is the primary asset. Bitcoin has not been delisted. For multi-asset expansion, only use assets that existed for the full backtest period.

### Bias Type 3: Overfitting (Data Snooping)

**What it is:** Running hundreds of parameter combinations and reporting only the best. Any sufficiently large parameter space will produce a great-looking strategy by chance.

**How this system prevents it:**

1. **Walk-forward validation:** The best parameters from training are tested on future data they've never seen. Random luck does not persist across multiple out-of-sample windows.

2. **Multi-metric fitness:** Fitness = 0.35×Sharpe + 0.25×Calmar + 0.20×Sortino + 0.20×PF. Overfitted strategies typically score well on one metric but poorly on others.

3. **Minimum trade count:** At least 10 trades required in each walk-forward window. Results from fewer trades are statistically unreliable.

4. **Adversarial training:** The adversary exposes strategies that only work in a narrow set of conditions.

---

## 5. Purged Walk-Forward Cross-Validation

### The Problem: Label Leakage

In standard cross-validation, consecutive samples can share overlapping labels. For example, if a 20-period moving average is used as a feature, the feature at bar t overlaps with bars t-19 through t. If bar t is in the training set and bar t-5 is in the test set, the test sample's feature was partly computed from training data.

This is subtle but real — it inflates OOS performance estimates.

### The Solution: Purging

`PurgedWalkForwardCV` (from `validation/antibias_walkforward.py`) introduces a **gap period** between training and test windows:

```
Training window:    [Day 1 ─────────────────── Day 365]
Gap (purged):                                            [Day 366─370]
Test window:                                                           [Day 371 ─ Day 460]
```

The gap length depends on the longest lookback period used in features. If the longest feature uses a 20-bar window, the gap should be at least 20 bars.

### Implementation

```python
from validation.antibias_walkforward import PurgedWalkForwardCV, WalkForwardConfig

config = WalkForwardConfig(
    n_splits=5,
    train_fraction=0.8,
    gap_periods=24,           # 24-bar gap (= 24 hours for 1h data)
    purge_method="label",     # Purge based on label overlap
)

cv = PurgedWalkForwardCV(config)

for train_idx, test_idx in cv.split(features_df):
    train_df = features_df.iloc[train_idx]
    test_df  = features_df.iloc[test_idx]
    
    # Verify gap is respected
    gap_between = test_df.index[0] - train_df.index[-1]
    assert gap_between >= pd.Timedelta(hours=24)
    
    # Train and evaluate
    ...
```

---

## 6. Interpreting Backtesting Results

### Reading the Walk-Forward Report

```
═══════════════════════════════════════════════════════════════════════
WALK-FORWARD RESULTS SUMMARY
═══════════════════════════════════════════════════════════════════════
Strategy:  RSI_scout_gen8_ind3
Data:      BTC/USDT 1h, 2022-01-01 → 2024-12-31
Windows:   5 (365-day train, 90-day test, 30-day step)
───────────────────────────────────────────────────────────────────────
Win │ IS Sharpe │ OOS Sharpe │ OOS Return │ MaxDD  │ Trades │ Degrad.
  1 │     1.84  │      1.21  │    +12.3%  │  -8.1% │     47 │  34.2%
  2 │     2.01  │      1.35  │    +15.8%  │  -6.3% │     52 │  32.8%
  3 │     1.73  │      0.89  │     +7.2%  │ -11.4% │     38 │  48.6%
  4 │     2.15  │      1.42  │    +17.1%  │  -5.8% │     55 │  33.9%
  5 │     1.92  │      1.18  │    +11.6%  │  -9.1% │     43 │  38.5%
───────────────────────────────────────────────────────────────────────
AVG │     1.93  │      1.21  │    +12.8%  │  -8.1% │     47 │  37.6%
═══════════════════════════════════════════════════════════════════════
VERDICT: APPROVED (all criteria met)
```

**Reading window 3 (weakest):**
- IS Sharpe 1.73 is good, OOS 0.89 is lower but still positive
- Degradation 48.6% is higher than other windows — investigate what market regime this period covered
- 38 trades is above the 10-trade minimum
- Max DD -11.4% is within the 25% limit

**What to look for:**
- All OOS Sharpe values positive → no windows lost money
- Degradation consistent across windows → no single window outlier
- OOS return positively correlated with IS return → ranking is preserved

### Red Flags

```
Red flag 1: One window with very negative OOS Sharpe
   Meaning: Strategy fails in certain market regimes
   Action:  Identify the regime, add regime filter

Red flag 2: Degradation > 70% in all windows
   Meaning: Heavy overfitting; strategy is learning noise
   Action:  Reduce model complexity, increase regularisation

Red flag 3: IS Sharpe >> 3.0 consistently
   Meaning: Likely lookahead bias or transaction cost error
   Action:  Audit feature engineering and cost modelling

Red flag 4: Trades < 10 per window
   Meaning: Too few trades to measure significance
   Action:  Use shorter test window or longer data period
```

---

## 7. Common Backtesting Pitfalls

### Pitfall 1: Lookahead Bias

**Description:** Using information that would not have been available at the time of the trade.

**Most common mistake:** Fitting normalisers on the full dataset:

```python
# WRONG:
scaler.fit(entire_df)
features = scaler.transform(entire_df)
# Split AFTER transforming → test data was used to fit the scaler

# CORRECT:
scaler.fit(train_df)
train_features = scaler.transform(train_df)
test_features  = scaler.transform(test_df)  # Same scaler, test data only
```

**Detection:** OOS performance is unrealistically close to IS performance (degradation < 10%).

### Pitfall 2: Transaction Cost Neglect

**Description:** Testing a strategy without accounting for fees, slippage, or spread.

**Example:** A high-frequency strategy that trades 20 times per day appears profitable at +0.2% per trade before costs. With 0.1% fee per side + 0.05% slippage per trade = 0.25% cost, the net result is −0.05% per trade.

**This system's defence:** `RealisticTradingEnv` includes:
- Binance maker/taker fees (0.1% by default)
- Slippage model (fixed BPS, volume-based, or order-book)
- Bid-ask spread impact

### Pitfall 3: Overfitting (Backtesting Overfitting)

**Description:** Testing hundreds of parameter combinations and selecting the best. The best-looking result is largely due to luck.

**Detection:** Strategy performs well in-sample but fails in walk-forward OOS windows.

**Prevention:** Walk-forward validation, multi-metric fitness, minimum trade count requirement.

### Pitfall 4: Outlier Dependence

**Description:** Strategy performance is driven by 2-3 exceptional trades. Without those trades, results are mediocre.

**Detection:** Check the distribution of individual trade PnL. If removing the top 5 trades changes Profit Factor from 2.0 to 0.9, the strategy is outlier-dependent.

**Prevention:** Ensure consistent performance across many trades. Minimum of 30 trades for meaningful statistics.

### Pitfall 5: Reporting IS Performance as OOS

**Description:** Reporting in-sample metrics as if they were out-of-sample. Common in simplistic backtests.

**Prevention:** Always run walk-forward; always distinguish IS Sharpe from OOS Sharpe.

---

## 8. Readiness Criteria

A strategy is ready for live trading when it passes **all** of the following:

### Quantitative Criteria

| Metric | Minimum | Source |
|---|---|---|
| Average OOS Sharpe | ≥ 0.5 | Walk-forward validation |
| OOS Profit Factor | ≥ 1.3 | Walk-forward validation |
| OOS Max Drawdown | ≤ 25% | Walk-forward validation |
| IS/OOS Degradation | ≤ 60% | Walk-forward validation |
| Min trades per window | ≥ 10 | Walk-forward validation |
| All stress tests | PASS | stress_tester.py |
| Risk Guard check | APPROVED | LiveTradingGuard |
| No walk-forward window | negative Sharpe | Walk-forward |

### Qualitative Criteria

| Check | Description |
|---|---|
| Explainable edge | The reason the strategy should work is understood |
| Market regime coverage | Walk-forward windows cover bull, bear, and sideways markets |
| Cost robustness | Profitable even with 3× the expected transaction costs |
| Position sizing tested | 1%-rule and Kelly correctly applied in backtest |
| Data quality verified | DataQualityAssessor run; no gaps or outliers |

### Promotion Process

```
1. Complete walk-forward run (5+ windows)
2. Run stress tests (all 6 scenarios)
3. LiveTradingGuard.check(strategy, df) → APPROVED
4. Compare fitness score vs. current champion
5. If higher: save to champion_meta.json
6. Upload to Drive (for Colab awareness)
7. GitHub push (version control)
8. Start with paper trading for 2 weeks
9. Promote to live trading after paper trading confirms results
```

The final paper trading step is optional but strongly recommended for new strategy types. It costs nothing and catches any remaining issues with live data quality, API behaviour, or timing assumptions.
