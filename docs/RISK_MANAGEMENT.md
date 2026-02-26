# BITCOIN4Traders — Risk Management

## Table of Contents

1. [The 1% Rule Explained](#1-the-1-rule)
2. [All 7 Protection Mechanisms](#2-all-7-protection-mechanisms)
3. [Kelly Criterion: Formula, Fractional Kelly, Examples](#3-kelly-criterion)
4. [ATR-Based Stop Loss Calculation](#4-atr-based-stop-loss)
5. [Circuit Breaker Logic](#5-circuit-breaker-logic)
6. [Position Sizing Walkthrough](#6-position-sizing-walkthrough)
7. [Risk Metrics Logging](#7-risk-metrics-logging)
8. [Configuring Risk Parameters (RiskConfig)](#8-configuring-risk-parameters)

---

## 1. The 1% Rule

### The Principle

**Never risk more than 1% of current account equity on any single trade.**

This is the most important rule in the entire system. It is enforced as a hard constraint — not a guideline, not a soft limit.

### Why 1%?

Consider two scenarios:

| Scenario | Risk per Trade | After 10 consecutive losses |
|---|---|---|
| Aggressive (5% risk) | $500 on $10,000 | $5,987 remaining (−40.1%) |
| Conservative (1% risk) | $100 on $10,000 | $9,044 remaining (−9.6%) |
| **This system (1% rule)** | **$100 on $10,000** | **$9,044 remaining (−9.6%)** |

Even 10 consecutive losing trades (an extreme but possible event) reduces the account by less than 10% when using 1% risk. This gives the system time to recover.

### How Position Size is Calculated

The 1% rule converts a dollar risk into a position size:

```
dollar_risk   = equity × 0.01
position_size = dollar_risk / |entry_price - stop_loss_price|
```

**Example:**

```
Equity:      $10,000
Entry price: $45,000 (BTC/USDT)
Stop loss:   $44,100 (2% below entry)

dollar_risk   = $10,000 × 0.01 = $100
stop_distance = $45,000 - $44,100 = $900
position_size = $100 / $900 = 0.111 BTC

Trade value:  0.111 × $45,000 = $5,000
If stop hit:  0.111 × $900 loss = $100 = exactly 1% of equity
```

---

## 2. All 7 Protection Mechanisms

All seven gates must pass for a trade to be executed. They run in sequence; failing any gate blocks the trade.

### Gate 1: 1%-Rule (Capital Risk Limit)

```python
max_dollar_risk = equity × 0.01
required_size   = max_dollar_risk / stop_distance

# Trade is approved only if proposed_size <= required_size
```

- **Purpose:** Cap worst-case loss per trade
- **Blocks when:** Proposed position size would risk more than 1% of equity
- **Action:** Automatically downscales the position to comply

### Gate 2: Kelly Ceiling

```python
kelly_fraction = (win_rate * avg_win - loss_rate * avg_loss) / avg_win
fractional_kelly = kelly_fraction * 0.5    # Half-Kelly for safety
max_kelly_size = equity * fractional_kelly

# Trade approved only if proposed_size <= max_kelly_size
```

- **Purpose:** Limit position to what Kelly formula deems mathematically optimal
- **Blocks when:** Position would exceed half-Kelly size
- **Note:** Kelly uses recent win rate and avg win/loss from trade history

### Gate 3: Stop-Loss Required

```python
if stop_loss_price is None:
    raise RiskViolation("No stop-loss defined. Trade rejected.")
```

- **Purpose:** Every trade must have a defined exit point
- **Blocks when:** Stop loss price is not provided
- **No exceptions:** A signal without a stop loss is always rejected

### Gate 4: Take-Profit (Risk/Reward Check)

```python
risk   = abs(entry - stop_loss)
reward = abs(take_profit - entry)
rr_ratio = reward / risk

if rr_ratio < min_rr_ratio:  # min_rr_ratio = 2.0
    raise RiskViolation(f"RR ratio {rr_ratio:.1f} below minimum 2.0")
```

- **Purpose:** Ensure positive expectancy (potential gain ≥ 2× potential loss)
- **Blocks when:** Take-profit is too close to entry
- **Example:** Entry $45,000, Stop $44,100 (risk=$900) → Take-profit must be ≥ $46,800

### Gate 5: Circuit Breaker (Daily Loss + Consecutive Losses)

```python
daily_loss_pct = (start_of_day_equity - current_equity) / start_of_day_equity

if daily_loss_pct > max_daily_loss_pct:       # default: 5%
    self.halt_trading = True
    self.halt_reason = f"Daily loss {daily_loss_pct:.1%} exceeds limit"

if self.consecutive_losses >= max_consecutive_losses:  # default: 5
    self.halt_trading = True
    self.halt_reason = f"{self.consecutive_losses} consecutive losses"
```

- **Purpose:** Stop trading during bad market conditions before damage accumulates
- **Blocks when:** Daily loss > 5% OR 5 consecutive losses (configurable)
- **Reset:** Daily loss circuit resets at midnight UTC; consecutive loss counter resets on a winning trade

### Gate 6: Drawdown Limit

```python
drawdown = (peak_equity - current_equity) / peak_equity

if drawdown > max_drawdown:  # default: 20%
    self.halt_trading = True
    self.halt_reason = f"Drawdown {drawdown:.1%} exceeds maximum {max_drawdown:.1%}"
```

- **Purpose:** Prevent catastrophic account destruction
- **Blocks when:** Current equity is more than `max_drawdown` below the peak
- **Example:** Peak equity $12,000 × (1 - 0.20) = $9,600 → halt if equity falls below $9,600

### Gate 7: Minimum Capital Gate

```python
min_capital = initial_capital * min_capital_threshold  # default: 30%

if current_equity < min_capital:
    self.halt_trading = True
    self.halt_reason = "Equity below minimum capital threshold"
```

- **Purpose:** Ultimate safety net — do not trade when most capital is gone
- **Blocks when:** Current equity < 30% of starting capital
- **Example:** Started with $10,000 → stop trading if equity falls below $3,000

---

## 3. Kelly Criterion

### The Formula

The Kelly Criterion provides the mathematically optimal position size that maximises long-run compound growth:

```
f* = (p × b - q) / b

Where:
  f* = fraction of capital to bet
  p  = probability of winning (win rate)
  q  = probability of losing = 1 - p
  b  = net odds = avg_win / avg_loss
```

### Example Calculation

```
Recent trade history: 60 trades
  Wins:   36  (win rate p = 36/60 = 0.60)
  Losses: 24  (loss rate q = 0.40)
  
Average winning trade:  +$520
Average losing trade:   −$280
Net odds: b = 520/280 = 1.857

Full Kelly:
  f* = (0.60 × 1.857 - 0.40) / 1.857
  f* = (1.114 - 0.40) / 1.857
  f* = 0.714 / 1.857
  f* = 0.385   →  38.5% of capital per trade

Half Kelly (fractional_kelly = 0.5):
  f* = 0.385 × 0.5 = 0.193  →  19.3% of capital
```

### Why Fractional Kelly?

Full Kelly (38.5% per trade in the example above) is mathematically optimal only if:
1. Win rate is estimated perfectly (it never is — estimates are noisy)
2. Returns are i.i.d. (markets are not)
3. No sequence-of-losses risk matters

In practice, win rates estimated from limited trade history are noisy. If the true win rate is slightly lower, full Kelly can cause catastrophic drawdowns. Half Kelly (50% of the full formula) provides a substantial safety margin while still capturing most of the growth benefit.

### Kelly in the System

The Kelly calculation uses a rolling window of the last 20 trades:

```python
from src.math_tools.kelly_criterion import KellyCriterion, KellyParameters

kelly = KellyCriterion()

params = KellyParameters(
    win_rate=0.60,
    avg_win=520.0,
    avg_loss=280.0,
    kelly_fraction=0.5,          # Half Kelly
    max_position_size=0.25,      # Cap at 25% regardless
)

f_star = kelly.calculate(params)
max_position = equity * f_star
```

The result is capped at `max_position_size` (default 25%) to prevent any single position from dominating the portfolio.

---

## 4. ATR-Based Stop Loss

### What is ATR?

Average True Range (ATR) measures market volatility. True Range for each bar is:

```
TR = max(
    high - low,                  # Bar range
    |high - prev_close|,         # Gap up
    |low - prev_close|           # Gap down
)

ATR(14) = exponential moving average of TR over 14 periods
```

A high ATR means the market is moving a lot; a low ATR means it is quiet.

### ATR Stop Placement

```python
atr_period = 14
atr_multiplier = 2.0            # Stop = 2× ATR below/above entry

# For a LONG trade:
stop_loss = entry_price - (atr * atr_multiplier)

# For a SHORT trade:
stop_loss = entry_price + (atr * atr_multiplier)
```

**Why ATR-based?**

Fixed percentage stops (e.g., always 2% below entry) ignore market conditions. During low-volatility periods, a 2% stop is very wide; during high-volatility periods, it is too tight and gets triggered by normal noise. ATR-based stops automatically adjust:

| Market Condition | ATR | ATR Stop (2×) | Interpretation |
|---|---|---|---|
| Quiet (overnight) | $300 | $600 below entry | Tight stop, normal for calm market |
| Normal | $900 | $1,800 below entry | Standard stop |
| Volatile (news event) | $2,400 | $4,800 below entry | Wide stop, avoids false triggers |

**Example:**

```
BTC/USDT at $45,000
ATR(14) = $1,200 (current 14-period ATR)
ATR multiplier = 2.0

Stop loss = $45,000 - (2.0 × $1,200) = $45,000 - $2,400 = $42,600

Dollar risk on 0.1 BTC = 0.1 × $2,400 = $240

With 1%-rule on $50,000 equity: max_risk = $500
Position size = $500 / $2,400 = 0.208 BTC
```

---

## 5. Circuit Breaker Logic

The circuit breaker monitors two conditions simultaneously and halts trading if either is triggered.

### Condition 1: Daily Loss Limit

```
Daily PnL = (current_equity - start_of_day_equity) / start_of_day_equity

HALT if Daily PnL < -5%   (configurable: max_drawdown_per_session)
```

The daily loss counter resets at midnight UTC. After a halt, no new trades are opened until the next trading day.

### Condition 2: Consecutive Loss Streak

```
consecutive_losses = number of trades in a row that lost money

HALT if consecutive_losses >= 5   (configurable: max_consecutive_losses)
```

Three consecutive losses is a warning; five triggers the halt. The counter resets on the first profitable trade.

### State Diagram

```
              Trade submitted
                    │
                    ▼
            ┌───────────────┐
            │  Normal State  │
            │  Trading OK   │
            └───────────────┘
                    │
          ┌─────────┴─────────┐
          │                   │
     Win trade           Lose trade
          │                   │
    reset consec.       consec_losses++
    losses = 0               │
          │            ┌──────┴──────┐
          │            │             │
          │      consec < 5    consec >= 5
          │            │             │
          │        Continue      CIRCUIT
          │                      BREAK
          │                          │
          │                   ┌──────▼──────┐
          │                   │  HALTED     │
          │                   │  No trades  │
          │                   │  allowed    │
          │                   └──────┬──────┘
          │                          │
          │                     Next day OR
          │                     manual reset
          └──────────────────────────┘
```

### Resuming After a Circuit Breaker Trip

The circuit breaker does NOT automatically resume. After a halt:

1. The `RiskManager.halt_trading` flag is `True`
2. All trade validation calls return `False`
3. Check the reason: `risk_mgr.get_halt_reason()`
4. To resume: `risk_mgr.reset_circuit_breaker()` (manual confirmation required)

This is intentional — a human should review what caused the halt before resuming.

---

## 6. Position Sizing Walkthrough

A complete position sizing example from signal to order size.

**Setup:**

```
Account equity:    $25,000
Signal:            BUY BTC/USDT
Entry price:       $45,000
Recent trade log:  65% win rate, avg win $480, avg loss $260
ATR(14):           $1,100
RR target:         2.5:1
```

**Step 1: Compute ATR stop**

```
stop_loss = $45,000 - (2.0 × $1,100) = $42,800
stop_distance = $45,000 - $42,800 = $2,200
```

**Step 2: Apply 1%-rule**

```
max_dollar_risk = $25,000 × 0.01 = $250
size_from_1pct = $250 / $2,200 = 0.1136 BTC
```

**Step 3: Compute Kelly size**

```
b = avg_win / avg_loss = 480/260 = 1.846
p = 0.65, q = 0.35

f* = (0.65 × 1.846 - 0.35) / 1.846 = (1.200 - 0.35) / 1.846 = 0.460

Half Kelly: f* = 0.460 × 0.5 = 0.230
Max Kelly position = $25,000 × 0.230 = $5,750
Kelly size = $5,750 / $45,000 = 0.1278 BTC
```

**Step 4: Take the minimum**

```
size_from_1pct = 0.1136 BTC
size_from_kelly = 0.1278 BTC

Final position size = min(0.1136, 0.1278) = 0.1136 BTC
```

**Step 5: Compute take-profit (RR 2.5:1)**

```
reward = stop_distance × RR = $2,200 × 2.5 = $5,500
take_profit = entry + reward = $45,000 + $5,500 = $50,500
```

**Step 6: Verify all 7 gates**

| Gate | Check | Result |
|---|---|---|
| 1: 1%-rule | 0.1136 × $2,200 = $249.92 ≤ $250 | PASS |
| 2: Kelly | 0.1136 ≤ 0.1278 | PASS |
| 3: Stop defined | stop = $42,800 | PASS |
| 4: RR ≥ 2.0 | RR = 2.5 | PASS |
| 5: Circuit breaker | No daily halt, no streak | PASS |
| 6: Drawdown | Current DD = 2.1% < 20% | PASS |
| 7: Min capital | $25,000 > $25,000 × 0.30 = $7,500 | PASS |

**Final order:**

```
MARKET BUY  0.1136 BTC  @ ~$45,000
Stop:        $42,800
Take-profit: $50,500
Max loss:    $249.92 (1.00% of equity)
Max gain:    $624.80 (2.50% of equity)
```

---

## 7. Risk Metrics Logging

The `RiskManager` tracks and logs the following metrics after every trade and at regular intervals.

### Per-Trade Metrics (saved to SQLite `trades` table)

| Metric | Description |
|---|---|
| `entry_price` | Actual fill price |
| `exit_price` | Close price (stop/TP/signal exit) |
| `quantity` | Position size in base currency |
| `pnl_dollar` | Realised profit/loss in USD |
| `pnl_pct` | PnL as percentage of equity at entry |
| `fee` | Exchange fee paid |
| `slippage` | Difference between signal price and fill price |
| `hold_duration` | Time in position (seconds) |
| `used_1pct_rule` | Whether 1%-rule was binding |
| `used_kelly` | Whether Kelly was binding |

### Portfolio Metrics (hourly snapshots, `portfolio_snapshots` table)

| Metric | Computation |
|---|---|
| `current_equity` | Cash + unrealised position value |
| `peak_equity` | Highest equity seen since start |
| `current_drawdown` | (peak - current) / peak |
| `consecutive_losses` | Running count of losing trades |
| `var_95` | Value at Risk (95% confidence, 1-day) |
| `daily_pnl` | Equity change since midnight UTC |
| `sharpe_7d` | Rolling 7-day Sharpe ratio |

### Value at Risk (VaR)

```python
# Historical simulation VaR (95% confidence)
returns = equity_history.pct_change().dropna()
var_95 = np.percentile(returns, 5) * current_equity
# Interpretation: with 95% confidence, next-day loss will not exceed var_95
```

---

## 8. Configuring Risk Parameters

### RiskConfig Dataclass

```python
from src.risk.risk_manager import RiskConfig

# Default configuration
config = RiskConfig(
    # Circuit Breaker
    max_drawdown_per_session=0.02,    # 2% max daily loss
    max_consecutive_losses=5,         # Stop after 5 losses in a row

    # Position Sizing
    max_position_size=0.25,           # Never more than 25% of capital
    kelly_fraction=0.5,               # Half Kelly

    # Value at Risk
    var_confidence=0.95,              # 95% VaR
    var_lookback=100,                 # Rolling 100-bar window

    # Safety Nets
    min_capital_threshold=0.3,        # Halt if equity < 30% of start
    enable_circuit_breaker=True,      # Set False to disable (NOT recommended)
)
```

### Conservative vs. Aggressive Profiles

**Conservative (new account, testing):**

```python
config = RiskConfig(
    max_drawdown_per_session=0.01,    # 1% daily loss limit
    max_consecutive_losses=3,         # Halt after 3 losses
    max_position_size=0.10,           # Max 10% per position
    kelly_fraction=0.25,              # Quarter Kelly
    min_capital_threshold=0.5,        # Halt at 50% drawdown
)
```

**Standard (proven strategy, live trading):**

```python
config = RiskConfig(
    max_drawdown_per_session=0.05,    # 5% daily loss limit
    max_consecutive_losses=5,         # Halt after 5 losses
    max_position_size=0.25,           # Max 25% per position
    kelly_fraction=0.5,               # Half Kelly
    min_capital_threshold=0.3,        # Halt at 70% drawdown
)
```

### Initialising the Risk Manager

```python
from src.risk.risk_manager import RiskManager, RiskConfig

config = RiskConfig()
risk_mgr = RiskManager(config, initial_capital=10_000.0)

# Before every trade:
approved, adjusted_size = risk_mgr.validate_position_size(
    proposed_size=500.0,        # Dollar value
    current_capital=10_200.0,
    stop_loss_pct=0.02,         # 2% stop
)

if approved:
    place_order(size=adjusted_size)
else:
    print(f"Trade rejected: {risk_mgr.get_halt_reason()}")

# After every trade:
risk_mgr.update_state(
    current_equity=10_200.0,
    trade_result=+200.0,        # positive = win, negative = loss
)

# Check if halted:
if risk_mgr.should_halt_trading():
    print("Trading halted:", risk_mgr.halt_reason)
```
