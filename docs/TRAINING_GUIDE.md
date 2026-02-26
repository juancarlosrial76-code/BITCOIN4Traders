# BITCOIN4Traders — Training Guide

## Table of Contents

1. [Prerequisites and Setup](#1-prerequisites-and-setup)
2. [Local Training: darwin_engine.py and auto_12h_train.py](#2-local-training)
3. [Colab GPU Training: Notebook Walkthrough](#3-colab-gpu-training)
4. [Hyperparameter Tuning Guide](#4-hyperparameter-tuning-guide)
5. [Walk-Forward Validation: Interpreting Results](#5-walk-forward-validation)
6. [Champion Evaluation and Promotion Criteria](#6-champion-evaluation-and-promotion)
7. [Troubleshooting Training Issues](#7-troubleshooting)

---

## 1. Prerequisites and Setup

### Software Requirements

```
Python       >= 3.10
PyTorch      >= 2.0
gymnasium    >= 0.29
stable-baselines3  (optional, for SB3 baselines)
ccxt         >= 4.0
pandas       >= 2.0
numpy        >= 1.24
numba        >= 0.57     (critical for Darwin Engine speed)
joblib       >= 1.3
loguru       >= 0.7
hydra-core   >= 1.3
pyarrow      >= 12.0     (Parquet support)
tensorboard  >= 2.13     (optional, for training curves)
```

Install all dependencies:

```bash
cd /home/hp17/Tradingbot/Quantrivo/BITCOIN4Traders
pip install -r requirements.txt
```

### Environment Variables

The following environment variables should be set before training:

```bash
# Exchange API (read-only for data fetching; write access for live trading)
export BINANCE_API_KEY="your_api_key_here"
export BINANCE_API_SECRET="your_secret_here"

# Google Drive Service Account (for Drive sync)
export GOOGLE_SA_CREDENTIALS="/path/to/service_account.json"

# Telegram Bot (for alerts; optional)
export TELEGRAM_BOT_TOKEN="your_bot_token"
export TELEGRAM_CHAT_ID="your_chat_id"

# Optional: override default SQLite path
export SQLITE_DB_PATH="/home/hp17/Tradingbot/data/sqlite/trading.db"
```

### Directory Structure

Verify the working directories exist before starting:

```bash
mkdir -p data/cache data/processed data/models data/backtests logs
```

### Data Prerequisite

Training requires at least 90 days of OHLCV data. Download it once:

```python
from src.data.ccxt_loader import CCXTDataLoader, DataLoaderConfig
from pathlib import Path

config = DataLoaderConfig(
    exchange_id="binance",
    exchange_type="spot",
    rate_limit_ms=100,
    cache_dir=Path("data/cache"),
    processed_dir=Path("data/processed"),
)
loader = CCXTDataLoader(config)

df = loader.download_and_cache(
    symbol="BTC/USDT",
    timeframe="1h",
    start_date="2022-01-01",
)
print(f"Downloaded {len(df)} candles")
```

---

## 2. Local Training

The Linux PC runs two complementary training scripts. They are designed to be scheduled via `cron` or `systemd` and run without human intervention.

### 2.1 auto_12h_train.py

This is the **primary local training loop**. It runs every 12 hours and executes the complete training-validation-promotion cycle.

```bash
python auto_12h_train.py
```

**What it does, step by step:**

```
Step 1: Load or download OHLCV data
         → CCXTDataLoader checks cache first
         → Downloads from Binance only if cache is stale

Step 2: Feature engineering
         → FeatureEngine.fit_transform(train_split)
         → FeatureEngine.transform(test_split)
         → Scaler fitted on train only (no lookahead)

Step 3: Darwin evolutionary run
         → DarwinArena(generations=10, pop_size=20)
         → Evaluates RSI, MACD, Bollinger, EMA scouts
         → Numba JIT + joblib parallelism (all CPU cores - 1)
         → EliteEvaluator scores: Sharpe + Calmar + Sortino + PF

Step 4: Walk-forward validation
         → WalkForwardValidator(n_splits=5)
         → Checks IS/OOS degradation ratio
         → Rejects strategy if OOS Sharpe < 0.5 × IS Sharpe

Step 5: Risk check
         → LiveTradingGuard.check(champion, data)
         → Must return "APPROVED" to continue

Step 6: Save champion to disk
         → data/cache/multiverse_champion_meta.json

Step 7: Upload to Google Drive
         → DriveManager.run_sync("up")
```

**Scheduling with cron:**

```bash
# Edit crontab
crontab -e

# Add entry to run every 12 hours
0 */12 * * * cd /home/hp17/Tradingbot/Quantrivo/BITCOIN4Traders && python auto_12h_train.py >> logs/train_cron.log 2>&1
```

### 2.2 darwin_engine.py

The Darwin Engine can be run standalone for focused evolutionary optimisation without the full pipeline:

```bash
python -m src.agents.darwin_engine
```

**Core concept:** The Darwin Engine runs a genetic algorithm over a population of classical trading strategies (RSI Scout, MACD Scout, Bollinger Scout, EMA Scout). Each strategy is a set of parameters (e.g., RSI period, oversold threshold, overbought threshold). The engine:

1. Initialises a random population of N parameter sets
2. Evaluates each on historical data using a Numba-compiled backtesting kernel
3. Selects the top performers (elite fraction)
4. Generates the next generation via crossover + mutation
5. Repeats for the specified number of generations

**Speed:** The `_kernel_simulate()` function runs at approximately 27,000× the speed of a naive pandas loop due to Numba JIT compilation. A full run of 10 generations × 20 candidates on 3 years of hourly data completes in under 60 seconds on a modern CPU.

**Key classes:**

| Class | Description |
|---|---|
| `DarwinArena` | Main entry point; manages the evolutionary loop |
| `RSIScout` | RSI mean-reversion strategy with configurable parameters |
| `MACDScout` | MACD crossover strategy |
| `BollingerScout` | Bollinger Band breakout/reversion |
| `EMAScout` | Exponential MA crossover |
| `EliteEvaluator` | Multi-metric fitness scoring (Sharpe, Calmar, Sortino, PF) |

---

## 3. Colab GPU Training

GPU training on Google Colab uses the notebook `BITCOIN4Traders_Colab.ipynb`. Open it in Colab, connect to a GPU runtime (Runtime → Change runtime type → T4 GPU), and run cells in order.

### Cell-by-Cell Walkthrough

**Cell 0 — Setup and Mount Drive**

```python
from google.colab import drive
drive.mount('/content/drive')

# Clone or pull latest code
import subprocess
subprocess.run(["git", "clone", "https://github.com/youruser/BITCOIN4Traders.git"])

# Install dependencies
subprocess.run(["pip", "install", "-r", "BITCOIN4Traders/requirements.txt", "-q"])
```

This cell mounts Google Drive (shared state with the Linux PC) and installs all Python dependencies. Takes 2-3 minutes.

**Cell 1 — Load Data**

```python
# Try loading from Drive first; fall back to direct CCXT download
import os
drive_data_path = "/content/drive/MyDrive/BITCOIN4Traders/data/"

if os.path.exists(drive_data_path + "btc_1h.parquet"):
    df = pd.read_parquet(drive_data_path + "btc_1h.parquet")
    print(f"Loaded {len(df)} rows from Drive")
else:
    # Download directly from Binance via CCXT
    loader = CCXTDataLoader(config)
    df = loader.download_and_cache("BTC/USDT", "1h", "2022-01-01")
```

**Cell 2 — Build Trading Environment**

```python
from src.environment.trading_env import RealisticTradingEnv, EnvConfig

env_config = EnvConfig(
    initial_capital=100_000.0,
    slippage_bps=5.0,          # 5 basis points slippage
    maker_fee=0.001,            # 0.1% maker fee
    taker_fee=0.001,            # 0.1% taker fee
    reward_type="sharpe",       # SharpeIncrementReward
    use_order_book=True,        # Enable synthetic L2 order book
)

env = RealisticTradingEnv(features_df, env_config)
print(f"Observation space: {env.observation_space.shape}")
print(f"Action space: {env.action_space.n}")   # 3: Sell, Hold, Buy
```

**Cell 3 — Create PPO Agent**

```python
from src.agents.ppo_agent import PPOAgent, PPOConfig

ppo_config = PPOConfig(
    hidden_dim=256,
    n_layers=3,
    recurrent=True,             # GRU for temporal memory
    recurrent_type="gru",
    dropout=0.1,
    learning_rate=3e-4,
    batch_size=64,
    n_epochs=10,
    clip_epsilon=0.2,           # PPO clipping
    gae_lambda=0.95,
    target_kl=0.01,             # Early stop on KL divergence
    entropy_coef=0.01,
)

agent = PPOAgent(
    obs_dim=env.observation_space.shape[0],
    action_dim=env.action_space.n,
    config=ppo_config,
)
print(f"Agent parameters: {sum(p.numel() for p in agent.policy_net.parameters()):,}")
```

**Cell 4 — Adversarial Training**

```python
from src.training.adversarial_trainer import AdversarialTrainer, AdversarialConfig

adv_config = AdversarialConfig(
    n_iterations=500,
    steps_per_iteration=2048,
    adversary_start_iteration=100,   # Warm-up: trader trains alone first
    adversary_strength=0.1,
    save_frequency=50,
    checkpoint_dir="/content/drive/MyDrive/BITCOIN4Traders/models/adversarial",
)

trainer = AdversarialTrainer(env, adv_config, agent)
trainer.train()
```

The adversarial training loop:
- Iterations 1–99: Only the Trader trains (warm-up phase)
- Iterations 100+: Adversary starts training; both agents improve
- Every 50 iterations: Checkpoints saved to Drive
- Every 10 iterations: Metrics logged (Trader return, Adversary loss)

**Cell N — Walk-Forward Validation**

```python
from src.backtesting.walkforward_engine import WalkForwardEngine, WalkForwardConfig

wf_config = WalkForwardConfig(
    train_window_days=365,
    test_window_days=90,
    step_days=30,
    train_iterations=200,
)

engine = WalkForwardEngine(agent, env_config, wf_config)
results = engine.run(df)

print(f"Average OOS Sharpe: {results.avg_oos_sharpe:.3f}")
print(f"IS/OOS degradation: {results.avg_degradation:.1%}")
```

**Heartbeat Cell (runs in loop)**

```python
import json, time
from datetime import datetime

status_path = "/content/drive/MyDrive/BITCOIN4Traders/colab_status.json"

while training:
    status = {
        "timestamp": datetime.utcnow().isoformat(),
        "iteration": current_iter,
        "trader_return": float(trader_return),
        "loss": float(current_loss),
        "status": "running",
    }
    with open(status_path, "w") as f:
        json.dump(status, f)
    time.sleep(300)   # Update every 5 minutes
```

---

## 4. Hyperparameter Tuning Guide

### PPO Hyperparameters

| Parameter | Default | Range | Effect |
|---|---|---|---|
| `learning_rate` | 3e-4 | 1e-5 to 1e-3 | Higher = faster learning, more instability |
| `batch_size` | 64 | 32 to 512 | Larger = more stable gradients, more memory |
| `n_epochs` | 10 | 3 to 20 | More epochs per rollout = sample efficiency vs. overfitting |
| `clip_epsilon` | 0.2 | 0.1 to 0.3 | PPO trust region size; 0.2 is canonical |
| `gae_lambda` | 0.95 | 0.9 to 0.99 | GAE bias-variance trade-off; 0.95 is canonical |
| `target_kl` | 0.01 | 0.005 to 0.05 | Early stop threshold; lower = more conservative updates |
| `hidden_dim` | 256 | 64 to 512 | Network capacity; scale with data complexity |
| `n_layers` | 3 | 2 to 5 | Depth; 3 is sufficient for most cases |
| `dropout` | 0.1 | 0.0 to 0.3 | Regularisation; increase if overfitting |
| `entropy_coef` | 0.01 | 0.001 to 0.05 | Exploration bonus; increase for sparse rewards |

**Tuning priority order:**

1. `learning_rate` — most impactful; if training diverges, reduce by 10×
2. `batch_size` — if OOM on Colab, halve it
3. `target_kl` — if policy collapses, reduce to 0.005
4. `dropout` — if OOS Sharpe << IS Sharpe, increase to 0.2
5. Everything else — rarely needs changing

### Darwin Engine Hyperparameters

| Parameter | Default | Range | Effect |
|---|---|---|---|
| `generations` | 10 | 5 to 50 | More generations = better convergence, more time |
| `pop_size` | 20 | 10 to 100 | Larger population = more diversity, more compute |
| `elite_fraction` | 0.2 | 0.1 to 0.4 | Fraction kept as parents; 0.2 is typical |
| `mutation_rate` | 0.1 | 0.05 to 0.3 | How much offspring deviate from parents |
| `crossover_rate` | 0.7 | 0.5 to 0.9 | Probability of gene exchange between parents |

### Adversarial Training Hyperparameters

| Parameter | Default | Effect if Increased |
|---|---|---|
| `n_iterations` | 500 | More total training; diminishing returns after ~1000 |
| `steps_per_iteration` | 2048 | More data per update; better gradient estimates |
| `adversary_start_iteration` | 100 | Longer warm-up = stronger trader baseline |
| `adversary_strength` | 0.1 | Harder scenarios; too high (>0.3) prevents learning |

**Warning:** If `adversary_strength` > 0.3 and the Trader's reward collapses to near-zero and stays there, the adversary is too strong. Reduce to 0.05 and restart.

### Walk-Forward Hyperparameters

| Parameter | Default | Recommendation |
|---|---|---|
| `train_window_days` | 365 | Minimum 180; more is generally better |
| `test_window_days` | 90 | Should represent at least one market regime |
| `step_days` | 30 | Smaller step = more windows = more robust but slower |
| `train_iterations` | 200 | 200 is a good compromise for per-window training |

---

## 5. Walk-Forward Validation

### What It Is

Walk-forward validation is the gold standard for testing trading strategies. Instead of training on the full dataset and testing on the same data (which guarantees overfitting), it splits time into sequential windows:

```
Full dataset (e.g., 3 years)
│
├─ Window 1  Train: Jan Y1 → Dec Y1   Test: Jan Y2 → Mar Y2
├─ Window 2  Train: Apr Y1 → Mar Y2   Test: Apr Y2 → Jun Y2
├─ Window 3  Train: Jul Y1 → Jun Y2   Test: Jul Y2 → Sep Y2
│   ...
└─ Window N  Train: ...               Test: (most recent 90 days)
```

Test windows never overlap with train windows. Each test window is strictly out-of-sample (OOS).

### Reading the Results Table

After a walk-forward run, the engine prints a results table like this:

```
Window | Train Sharpe | OOS Sharpe | OOS Return | Max DD | Trades | Degradation
-------|--------------|------------|------------|--------|--------|------------
     1 |        1.82  |      1.21  |    +12.3%  |  -8.1% |     47 |      33.5%
     2 |        2.14  |      1.45  |    +18.7%  |  -6.2% |     52 |      32.2%
     3 |        1.73  |      0.94  |     +8.1%  |  -9.8% |     38 |      45.7%
     4 |        2.01  |      1.31  |    +14.2%  |  -7.3% |     44 |      34.8%
     5 |        1.95  |      1.18  |    +11.4%  |  -8.9% |     41 |      39.5%
-------|--------------|------------|------------|--------|--------|------------
  AVG  |        1.93  |      1.22  |    +12.9%  |  -8.1% |     44 |      37.1%
```

**How to interpret each column:**

- **Train Sharpe:** In-sample performance. Expected to be higher than OOS.
- **OOS Sharpe:** Out-of-sample performance. This is the honest estimate.
- **OOS Return:** Actual return in the test period.
- **Max DD:** Worst drawdown during the test period.
- **Trades:** Number of completed round-trips. Below 10 is statistically unreliable.
- **Degradation:** `(Train Sharpe - OOS Sharpe) / Train Sharpe`. Lower is better.

### Acceptance Criteria

A strategy passes walk-forward validation if:

```
Average OOS Sharpe     >= 0.5
IS/OOS degradation     <= 60%   (OOS Sharpe >= 0.4 × IS Sharpe)
Average Max Drawdown   <= 25%
Minimum trades/window  >= 10
No window with negative OOS Sharpe (unless isolated)
```

If degradation is consistently above 60%, the strategy is overfitted to the training windows.

### PurgedWalkForwardCV (Anti-Leakage)

The `PurgedWalkForwardCV` adds a gap period between training and test windows:

```
Train: |===================|   (365 days)
Gap:                        |==|  (5 days, purged)
Test:                           |=========|  (90 days)
```

The gap ensures no training samples whose labels "bleed" into the test period. This prevents a subtle form of lookahead bias where the model learns from samples whose outcomes depend on data in the test window.

---

## 6. Champion Evaluation and Promotion

### Multi-Metric Fitness Score

The `EliteEvaluator` ranks every strategy by a composite fitness score:

```python
fitness = (
    0.35 * sharpe_ratio
  + 0.25 * calmar_ratio
  + 0.20 * sortino_ratio
  + 0.20 * profit_factor
)
```

**Why four metrics instead of one?**

Sharpe alone is gameable (e.g., a strategy that holds cash most of the time has high Sharpe but low returns). Calmar penalises drawdown severity. Sortino focuses only on downside volatility. Profit factor directly measures whether the strategy makes more than it loses.

### Promotion Checklist

Before a new candidate replaces the current champion:

| Check | Criterion | Reason |
|---|---|---|
| OOS Sharpe | >= 0.5 | Minimum meaningful risk-adjusted return |
| Calmar Ratio | >= 0.3 | Max DD not too large relative to return |
| Profit Factor | >= 1.3 | Gross profit > 1.3× gross loss |
| Max Drawdown | <= 25% | Capital preservation |
| Min Trades | >= 30 | Statistical significance |
| Walk-Forward | Pass (see §5) | Anti-overfitting |
| Risk Guard | APPROVED | All 7 risk gates pass |
| Beats Champion | Fitness score higher | Only promote improvements |

### Champion Metadata

When promoted, champion metadata is saved to `data/cache/multiverse_champion_meta.json`:

```json
{
  "name": "RSI_scout_gen8_ind3",
  "strategy": "RSI",
  "sharpe": 1.22,
  "calmar": 0.61,
  "sortino": 1.89,
  "profit_factor": 1.47,
  "total_return": 0.187,
  "win_rate": 0.54,
  "max_drawdown": -0.082,
  "n_trades": 47,
  "params": {"rsi_period": 14, "oversold": 28, "overbought": 72},
  "promoted_at": "2025-01-15T09:32:00Z",
  "source": "local_master"
}
```

---

## 7. Troubleshooting

### Training Diverges (NaN loss)

**Symptoms:** Loss becomes NaN after a few iterations; agent stops improving.

**Causes and fixes:**

```
Cause 1: Learning rate too high
Fix:     Reduce learning_rate by 10× (e.g., 3e-4 → 3e-5)

Cause 2: Reward scale too large (large absolute reward values)
Fix:     Normalise rewards in the environment (divide by rolling std)

Cause 3: Gradient explosion in recurrent layer
Fix:     Add gradient clipping: clip_grad_norm_(params, max_norm=0.5)

Cause 4: NaN in input features
Fix:     Run df.isna().sum() on features; check FeatureEngine output
```

The auto-repair system on the Linux PC detects NaN loss from Colab error reports and automatically patches the notebook by reducing `learning_rate` by 10×.

### Out of Memory (OOM) on Colab

**Symptoms:** CUDA out of memory error; Colab crashes.

**Fixes:**

```
1. Reduce batch_size (e.g., 64 → 32)
2. Reduce hidden_dim (e.g., 256 → 128)
3. Disable recurrent layer (recurrent=False)
4. Reduce steps_per_iteration (e.g., 2048 → 1024)
```

The auto-repair system detects OOM and automatically halves `batch_size` in the patched notebook.

### Walk-Forward All Windows Failing

**Symptoms:** Every window shows OOS Sharpe < 0.3 or negative.

**Possible causes:**

```
1. Data issue: Check for gaps or outliers in OHLCV data
   Fix: Run DataQualityAssessor on the dataset

2. Feature leakage: Scaler fitted on entire dataset
   Fix: Verify FeatureEngine.fit() is called only on training split

3. Strategy not suitable for the data period
   Fix: Try different timeframes (4h or 1d instead of 1h)

4. Transaction costs too high for strategy frequency
   Fix: Check average trade duration; increase slippage_bps in env

5. Train window too short
   Fix: Increase train_window_days to 365+
```

### Darwin Engine Converges Prematurely

**Symptoms:** Fitness score stops improving after 2-3 generations.

**Fixes:**

```
1. Increase mutation_rate (e.g., 0.1 → 0.2)
2. Increase pop_size (e.g., 20 → 50)
3. Reduce elite_fraction (e.g., 0.2 → 0.1) for more diversity
4. Run for more generations
```

### Champion Not Promoting

If `auto_12h_train.py` logs "champion not promoted" repeatedly:

```
1. Check LiveTradingGuard output — which of the 7 risk gates is failing?
2. Check walk-forward degradation — above 60%?
3. Lower the min_trades threshold if data period is short
4. Review EliteEvaluator fitness — is the current champion genuinely better?
```

### Colab Session Timeout

Colab free tier disconnects after 90 minutes of inactivity. The `ColabWatchdog` on the Linux PC detects this via the heartbeat file and sends a restart signal via Drive. If automatic restart is not working:

```
1. Check Drive credentials are valid
2. Verify the heartbeat cell is running in Colab
3. Manually open BITCOIN4Traders_Colab.ipynb and run all cells
```
