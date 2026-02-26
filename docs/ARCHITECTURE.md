# BITCOIN4Traders — System Architecture

## Table of Contents

1. [Project Philosophy](#1-project-philosophy)
2. [Four-Pillar Zero-Cost Infrastructure](#2-four-pillar-zero-cost-infrastructure)
3. [Complete Data Flow](#3-complete-data-flow)
4. [Layer Architecture](#4-layer-architecture)
5. [Component Interaction Map](#5-component-interaction-map)
6. [Training Pipeline](#6-training-pipeline)
7. [Error Recovery and Self-Healing Loop](#7-error-recovery-and-self-healing-loop)
8. [Key Design Decisions and Trade-offs](#8-key-design-decisions-and-trade-offs)

---

## 1. Project Philosophy

BITCOIN4Traders is a **reinforcement learning trading system** for Bitcoin/crypto markets built around three core principles:

**Principle 1 — Evolutionary Selection**
No single handcrafted strategy survives changing market conditions. The system therefore runs a continuous Darwinian competition: many candidate strategies are evaluated in parallel, the winners are selected, crossed, and mutated, and the process repeats. Only the strategy that currently performs best is promoted to live trading — the "champion".

**Principle 2 — Risk First**
The algorithm decides *what* to trade. The risk engine decides *how much* and *whether at all*. These are deliberately separated: `darwin_engine.py` generates signals, `risk_engine.py` gates every single trade through seven independent protection mechanisms before any capital is committed. A signal alone is never sufficient to place a trade.

**Principle 3 — Anti-Overfitting by Design**
The most dangerous failure mode in algorithmic trading is a strategy that looks excellent on historical data but fails in production. Every component of this system includes explicit defences: walk-forward validation, purged cross-validation to prevent data leakage, realistic transaction-cost modelling with slippage, adversarial training to expose hidden weaknesses, and multi-metric fitness scoring that penalises overfitted Sharpe ratios.

---

## 2. Four-Pillar Zero-Cost Infrastructure

The entire system runs at **zero recurring cost** by combining four free resources into a coherent distributed system.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    BITCOIN4Traders Zero-Cost Infrastructure                 │
│                                                                             │
│  ┌─────────────────┐   ┌─────────────────┐   ┌─────────────────────────┐  │
│  │   PILLAR 1      │   │   PILLAR 2      │   │      PILLAR 3           │  │
│  │   Linux PC      │   │  Google Drive   │   │    Google Colab GPU     │  │
│  │   (Brain)       │   │   (Memory)      │   │     (Muscle)            │  │
│  │                 │   │                 │   │                         │  │
│  │ • 24/7 uptime   │──▶│ • Models        │──▶│ • Deep RL training      │  │
│  │ • Scheduling    │   │ • Logs          │   │ • PPO / SAC / TD3       │  │
│  │ • Risk engine   │◀──│ • Heartbeats    │◀──│ • Free Tesla T4 GPU     │  │
│  │ • Live trading  │   │ • Signals       │   │ • Adversarial training  │  │
│  │ • Monitoring    │   │                 │   │                         │  │
│  └────────┬────────┘   └─────────────────┘   └─────────────────────────┘  │
│           │                                                                 │
│           ▼                                                                 │
│  ┌─────────────────┐                                                        │
│  │   PILLAR 4      │                                                        │
│  │    GitHub       │                                                        │
│  │   (Backup)      │                                                        │
│  │                 │                                                        │
│  │ • Code version  │                                                        │
│  │ • Champion push │                                                        │
│  │ • Full recovery │                                                        │
│  └─────────────────┘                                                        │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Pillar 1 — Linux PC (Brain)

The local machine is the always-on control centre. It runs the Master Orchestrator (`infrastructure/master.py`) as a `systemd` service, which means it survives reboots and restarts automatically on failure. The PC handles:

- Continuous lightweight training via `auto_12h_train.py`
- Darwin evolutionary engine (CPU-bound, multi-core via joblib)
- Live trading execution via the Live Engine
- Risk management and order gating
- Scheduling of all infrastructure tasks

### Pillar 2 — Google Drive (Memory)

Google Drive acts as the shared state between the Linux PC and Colab. The Drive API is free up to one billion requests per day. It stores:

- Champion model weights (`.pth` files)
- Training logs and metrics
- Heartbeat files (proof of life from both sides)
- Colab status files (last training iteration, loss, etc.)
- Restart signals (commands from PC to Colab)
- Error reports (Colab → PC when training fails)

The `drive_manager.py` component handles all authentication via a Service Account (no OAuth browser flow required) and ensures upsert semantics — uploading a file never creates a duplicate, it always overwrites the existing version.

### Pillar 3 — Google Colab GPU (Muscle)

Colab provides free access to NVIDIA Tesla T4 / P100 GPUs. This is where heavy deep RL training takes place: PPO, SAC, TD3, and Adversarial Training all benefit enormously from GPU acceleration. The Colab notebook (`BITCOIN4Traders_Colab.ipynb`) is the entry point. The `colab_watchdog.py` component on the Linux PC monitors Colab status and sends a restart signal via Drive if Colab goes silent for more than 90 minutes.

### Pillar 4 — GitHub (Backup)

Every six hours, the Linux PC pushes champion model metadata and code to GitHub via `sync_champion.sh`. This ensures full recovery is possible even if local storage is lost. GitHub also serves as the versioning system for all code changes.

---

## 3. Complete Data Flow

### End-to-End Flow: Raw Data → Live Trading Signal

```
  EXTERNAL WORLD
  ════════════════════════════════════════════════════════════════════════
  │  Binance REST API  │  Binance WebSocket  │  CCXT (other exchanges)  │
  ════════════════════════════════════════════════════════════════════════
                │                  │                      │
                ▼                  ▼                      ▼
  ┌─────────────────────────────────────────────────────────────────────┐
  │                         DATA LAYER                                  │
  │                                                                     │
  │  CCXTDataLoader ──────────────────▶ Parquet Cache (snappy)          │
  │  HistoricalDataDownloader ────────▶ PostgreSQL / SQLite             │
  │  BinanceWSConnector (live) ───────▶ In-memory tick stream           │
  └─────────────────────┬───────────────────────────────────────────────┘
                         │
                         ▼
  ┌─────────────────────────────────────────────────────────────────────┐
  │                      FEATURE LAYER                                  │
  │                                                                     │
  │  FeatureEngine.fit_transform()                                      │
  │    ├─ Log returns (stationarity)                                    │
  │    ├─ Volatility (rolling std)                                      │
  │    ├─ OU process mean-reversion signal                              │
  │    ├─ StandardScaler / RobustScaler (no future leakage)             │
  │    └─ NaN handling                                                  │
  │                                                                     │
  │  MultiTimeframeAnalyzer                                             │
  │    ├─ M1, M5, M15, H1, H4, D1, W1 signals                          │
  │    └─ Top-down alignment score                                      │
  │                                                                     │
  │  MicrostructureAnalyzer                                             │
  │    ├─ Bid/ask spread                                                │
  │    ├─ Order flow imbalance                                          │
  │    └─ Market impact estimates                                       │
  └─────────────────────┬───────────────────────────────────────────────┘
                         │
                         ▼
  ┌─────────────────────────────────────────────────────────────────────┐
  │                    ENVIRONMENT LAYER                                │
  │                                                                     │
  │  RealisticTradingEnv (gym.Env)                                      │
  │    ├─ State: [price features | position | portfolio metrics |       │
  │    │          order book features]                                  │
  │    ├─ Actions: 0=Sell  1=Hold  2=Buy                                │
  │    ├─ OrderBookSimulator (10-level synthetic L2)                    │
  │    ├─ SlippageModel (volume-based / orderbook / volatility)         │
  │    └─ TransactionCostEngine (maker/taker + spread + funding)        │
  │                                                                     │
  │  Anti-Bias Reward Functions                                         │
  │    ├─ SharpeIncrementReward (normalise by rolling vol)              │
  │    ├─ CostAwareReward (penalise churn)                              │
  │    └─ RegimeAwareReward (adapt to HMM market state)                 │
  └─────────────────────┬───────────────────────────────────────────────┘
                         │
                         ▼
  ┌─────────────────────────────────────────────────────────────────────┐
  │                       AGENT LAYER                                   │
  │                                                                     │
  │  PPOAgent (primary)                                                 │
  │    ├─ BaseNetwork: Linear → LayerNorm → ReLU → Dropout              │
  │    ├─ GRU / LSTM recurrent layer (temporal memory)                  │
  │    ├─ Actor head: π(a|s) with Categorical distribution              │
  │    ├─ Critic head: V(s)                                             │
  │    ├─ GAE advantage estimation (λ=0.95)                             │
  │    └─ Clipped surrogate objective (ε=0.2), KL early stop            │
  │                                                                     │
  │  Adversarial Training                                               │
  │    ├─ Trader agent: maximise risk-adjusted return                   │
  │    └─ Adversary agent: create hardest possible market conditions    │
  │                                                                     │
  │  Optional: DQN / SAC / A2C / TD3 / Ensemble / Transformer          │
  └─────────────────────┬───────────────────────────────────────────────┘
                         │
                         ▼
  ┌─────────────────────────────────────────────────────────────────────┐
  │                      MATH TOOLS LAYER                               │
  │                                                                     │
  │  KalmanFilter      ── noise-filtered trend state                    │
  │  GARCHModel        ── conditional volatility forecast               │
  │  HMMRegimeDetector ── market regime (bull/bear/transition)          │
  │  KellyCriterion    ── optimal position fraction                     │
  │  HurstExponent     ── trending vs mean-reverting classification     │
  │  SpectralAnalysis  ── cyclical pattern detection                    │
  │  Cointegration     ── pairs trading relationships                   │
  │  OrnsteinUhlenbeck ── mean-reversion speed estimation               │
  └─────────────────────┬───────────────────────────────────────────────┘
                         │
                         ▼
  ┌─────────────────────────────────────────────────────────────────────┐
  │                       RISK LAYER                                    │
  │                                                                     │
  │  RiskEngine (7 gates — all must pass)                               │
  │    ├─ [1] 1%-rule: max_risk = equity × 0.01                         │
  │    ├─ [2] Kelly ceiling: f* = (p·b − q) / b × 0.5                  │
  │    ├─ [3] ATR stop-loss (or fixed %)                                │
  │    ├─ [4] Take-profit: entry risk × RR-ratio (≥ 2.0)               │
  │    ├─ [5] Circuit breaker: halt if daily loss > 5%                  │
  │    │       or 3 consecutive losses                                  │
  │    ├─ [6] Drawdown limit: halt if equity < (peak × max_dd)          │
  │    └─ [7] Min-capital gate: halt if equity < 30% of start           │
  │                                                                     │
  │  ManipulationDetector                                               │
  │    ├─ Spoofing / layering / wash trading detection                  │
  │    └─ Order book toxicity score → execution adaptation              │
  └─────────────────────┬───────────────────────────────────────────────┘
                         │
                         ▼
  ┌─────────────────────────────────────────────────────────────────────┐
  │                    EXECUTION LAYER                                  │
  │                                                                     │
  │  LiveEngine                                                         │
  │    ├─ BinanceWSConnector: real-time tick + user stream              │
  │    ├─ OrderManager: order lifecycle (submit/fill/cancel)            │
  │    ├─ Position tracker: VWAP average cost, realized PnL             │
  │    └─ ExecutionAlgorithms: TWAP, VWAP, Iceberg                      │
  │                                                                     │
  │  ProductionMonitor                                                  │
  │    ├─ TelegramNotifier: async fill alerts                           │
  │    ├─ Prometheus metrics: Grafana-ready                             │
  │    └─ Rolling P&L dashboard                                         │
  └─────────────────────────────────────────────────────────────────────┘
```

---

## 4. Layer Architecture

### Layer 1 — Data

**Purpose:** Acquire, cache, and validate all market data.

| Component | Role |
|---|---|
| `CCXTDataLoader` | Fetches OHLCV from any CCXT exchange, Parquet caching |
| `HistoricalDataDownloader` | Pulls years of history, stores in PostgreSQL or SQLite |
| `BinanceConnector` | Spot/futures REST API with retry logic and rate limiting |
| `BinanceWSConnector` | Asynchronous WebSocket for live ticks + user stream |
| `DataManager` | Unified access layer, data validation, quality checks |
| `DataQualityAssessor` | Detects gaps, outliers, and distribution shifts |
| `LiveDataMonitor` | Streaming quality gate for production data |

**Key invariant:** `FeatureEngine.fit()` is only ever called on training data. `transform()` is called on test and live data, using statistics from the training set. This prevents lookahead bias at the data-ingestion boundary.

### Layer 2 — Features

**Purpose:** Transform raw OHLCV into stationary, normalised features suitable for neural networks.

The `FeatureEngine` follows the scikit-learn `fit_transform` / `transform` pattern:

```
Training phase:   engine.fit_transform(train_df)   → fits scaler, stores stats
Test/Live phase:  engine.transform(live_df)         → applies stored stats ONLY
```

Feature groups computed:
- **Returns:** Log returns for stationarity
- **Volatility:** Rolling standard deviation with configurable window
- **Mean-reversion:** Ornstein-Uhlenbeck signal (deviation from equilibrium)
- **Scaling:** StandardScaler, MinMaxScaler, or RobustScaler — choice is configurable via Hydra

`MultiTimeframeAnalyzer` aggregates signals across eight timeframes (1m to 1w) and computes a top-down alignment score. `MicrostructureAnalyzer` adds bid-ask spread, order flow imbalance, and market impact estimates from L2 order book data.

### Layer 3 — Environment

**Purpose:** Simulate realistic trading conditions so the agent experiences realistic costs during training, preventing the agent from learning a strategy that only works when costs are ignored.

`RealisticTradingEnv` (gymnasium-compatible) has:

- **State space:** Concatenation of price features (from FeatureEngine), current position (long/short/flat), portfolio metrics (PnL, drawdown, unrealised), and order book features (spread, depth imbalance)
- **Action space:** Discrete {0=Sell, 1=Hold, 2=Buy}
- **Reward functions** (Anti-Bias Framework): `SharpeIncrementReward` normalises each step's return by local rolling volatility; `CostAwareReward` explicitly penalises excessive transaction costs; `RegimeAwareReward` adjusts risk tolerance based on the detected HMM regime

`OrderBookSimulator` generates a synthetic 10-level L2 book around the mid price, with volume and volatility-dependent spread. `SlippageModel` has four modes: fixed BPS, volume-based, volatility-adjusted, and full order-book walk. `TransactionCostEngine` covers Binance maker/taker fees, bid-ask spread (with timeframe multipliers), market impact, and futures funding rates.

### Layer 4 — Agent

**Purpose:** Learn a policy that maximises risk-adjusted returns in the environment.

**Primary agent: PPOAgent**

PPO (Proximal Policy Optimization, Schulman et al. 2017) was chosen as the primary algorithm because it is stable, sample-efficient, and well-understood. The implementation adds:

- Optional GRU or LSTM recurrent layer for temporal memory (market context across bars)
- Layer Normalization and Dropout for regularisation
- Separate actor and critic learning rates
- Exponential learning rate decay
- Early stopping when KL divergence exceeds `target_kl = 0.01`
- Generalised Advantage Estimation (GAE, λ=0.95)

**Adversarial Training** (`AdversarialTrainer`): A second agent — the Adversary — is trained concurrently to create market scenarios that maximally hurt the Trader. The Trader must then adapt. This self-play loop produces strategies that are robust to adversarial and out-of-distribution conditions. The Adversary starts after 100 warm-up iterations so the Trader first learns a baseline.

**Additional agents** (all in `src/agents/drl_agents.py`): DQN (discrete, experience replay, target network), A2C (synchronous advantage actor-critic), SAC (maximum entropy, off-policy, continuous), TD3 (twin delayed, target policy smoothing). The `AgentEnsemble` module combines any set of agents via voting, weighted averaging, stacking (meta-learner), or bagging.

### Layer 5 — Mathematical Tools

**Purpose:** Provide rigorous statistical and mathematical primitives that inform risk management, regime detection, and feature engineering.

| Module | Algorithm | Primary Use |
|---|---|---|
| `KalmanFilter1D` | State-space optimal estimation | Price trend extraction |
| `GARCHModel` | GARCH(1,1) max-likelihood | Volatility forecasting for VaR |
| `HMMRegimeDetector` | Hidden Markov Model | Bull/bear/transition regime |
| `KellyCriterion` | f* = (p·b - q) / b | Optimal position fraction |
| `HurstExponent` | R/S analysis | Trending vs mean-reverting |
| `SpectralAnalysis` | FFT / Lomb-Scargle | Cyclical pattern detection |
| `Cointegration` | Engle-Granger, Johansen | Pairs trading relationships |
| `OrnsteinUhlenbeck` | OU SDE parameter estimation | Mean-reversion speed |

### Layer 6 — Risk

**Purpose:** Act as the final gatekeeper before any trade reaches the exchange. No signal from any agent, however confident, bypasses the risk layer.

**Seven gates** (all must pass for a trade to be executed):

1. **1%-rule** — Never risk more than 1% of current equity on a single trade. Position size = (equity × 0.01) / |entry - stop|.
2. **Kelly ceiling** — The Kelly formula additionally limits position size. Fractional Kelly (50%) is used for safety margin.
3. **Stop-loss** — Automatically computed from ATR or fixed percentage. No trade without a stop.
4. **Take-profit** — Set at entry_risk × RR-ratio (minimum 2.0). Ensures positive expectancy.
5. **Circuit breaker** — Trading halts if daily loss exceeds 5% or if there are 3 consecutive losing trades.
6. **Drawdown limit** — Trading halts if equity falls below (peak equity × (1 - max_drawdown)).
7. **Min-capital gate** — No trading if equity is below 30% of starting capital.

### Layer 7 — Execution

**Purpose:** Convert approved trade signals into actual exchange orders with minimal slippage and correct lifecycle management.

`LiveEngine` orchestrates the full execution cycle:

```
WS tick → FeatureEngine.transform() → Agent.predict()
        → Risk checks → OrderManager.submit() → Fill handlers
```

`OrderManager` handles the complete order lifecycle: submission, partial fills, full fills, cancellation, and rejection. `Position` tracks VWAP average entry cost and realised PnL per symbol. `ExecutionAlgorithms` provides TWAP, VWAP, and Iceberg execution for large orders to minimise market impact.

---

## 5. Component Interaction Map

```
                        ┌──────────────────────────────┐
                        │      Master Orchestrator      │
                        │   (infrastructure/master.py)  │
                        │                              │
                        │  Scheduler                   │
                        │  ├─ Every 15 min: training   │
                        │  ├─ Every 60 min: drive sync │
                        │  ├─ Every 60 min: watchdog   │
                        │  ├─ Every 6 hrs: github push │
                        │  └─ Every 24 hrs: status rpt │
                        └────────────┬─────────────────┘
                                     │
            ┌────────────────────────┼──────────────────────────┐
            │                        │                          │
            ▼                        ▼                          ▼
  ┌──────────────────┐    ┌──────────────────┐    ┌──────────────────────┐
  │  DriveManager    │    │  ColabWatchdog   │    │   AlertManager       │
  │                  │    │                  │    │                      │
  │ upload_champion  │    │ read_colab_status│    │ INFO / WARNING /     │
  │ upload_logs      │    │ is_colab_alive   │    │ CRITICAL             │
  │ sync_notebook    │    │ send_restart_sig │    │                      │
  │ write_heartbeat  │    │                  │    │ Channels:            │
  └──────────────────┘    └──────────────────┘    │  • Telegram          │
                                                  │  • Log file          │
                                                  │  • Desktop notify    │
                                                  └──────────────────────┘
                                                           │
                                                           ▼
                                                  ┌──────────────────────┐
                                                  │  Listener (Flask)    │
                                                  │                      │
                                                  │ Port 5001            │
                                                  │ Token auth (HMAC)    │
                                                  │ Rate limiting (10/m) │
                                                  │ Drive fallback chan.  │
                                                  └──────────┬───────────┘
                                                             │
                                                             ▼
                                                  ┌──────────────────────┐
                                                  │  ErrorRepair Engine  │
                                                  │                      │
                                                  │ classify_error()     │
                                                  │ repair_oom()         │
                                                  │ repair_nan_loss()    │
                                                  │ repair_connection()  │
                                                  │ repair_timeout()     │
                                                  │ repair_import()      │
                                                  │ patch_notebook()     │
                                                  │ upload_patched_nb()  │
                                                  └──────────────────────┘


  TRAINING COMPONENTS
  ════════════════════════════════════════════════════════════════════════

  DarwinEngine                       RiskEngine
  ┌────────────────────────┐         ┌────────────────────────────────┐
  │ DarwinArena            │         │ StrategyTournament             │
  │  ├─ RSIScout           │         │  └─▶ LiveTradingGuard          │
  │  ├─ MACDScout          │    ──▶  │       └─▶ RiskEngine           │
  │  ├─ BollingerScout     │         │             └─▶ TradingSession │
  │  ├─ EMAScout           │         └────────────────────────────────┘
  │  └─ EliteEvaluator     │
  │      (Sharpe/Calmar/   │
  │       Sortino/PFR)     │
  └────────────────────────┘
```

---

## 6. Training Pipeline

### Local Training (Linux PC — CPU, continuous)

```
auto_12h_train.py  (runs every 12 hours)
  │
  ├─ 1. Load or download latest OHLCV data
  │       CCXTDataLoader → Parquet cache
  │
  ├─ 2. Feature engineering
  │       FeatureEngine.fit_transform(train) → FeatureEngine.transform(test)
  │
  ├─ 3. Darwin evolutionary run
  │       DarwinArena(generations=10, pop_size=20)
  │       Numba JIT kernels + joblib parallel (all CPU cores - 1)
  │       EliteEvaluator: Sharpe + Calmar + Sortino + Profit Factor
  │
  ├─ 4. Walk-forward validation
  │       WalkForwardValidator(n_splits=5)
  │       IS/OOS degradation check (overfitting detection)
  │
  ├─ 5. Risk check
  │       LiveTradingGuard.check(champion, df) → must be APPROVED
  │
  ├─ 6. Save champion
  │       data/cache/multiverse_champion_meta.json
  │
  └─ 7. Upload to Drive
          DriveManager.run_sync("up")
```

### GPU Training (Google Colab — T4/P100, on-demand)

```
BITCOIN4Traders_Colab.ipynb
  │
  ├─ Cell 0: Mount Drive, clone repo, install deps
  │
  ├─ Cell 1: Load data from Drive / CCXT
  │
  ├─ Cell 2: Build RealisticTradingEnv
  │             OrderBookSimulator + SlippageModel
  │             Anti-Bias reward functions
  │
  ├─ Cell 3: Create PPOAgent (GRU recurrent, LayerNorm)
  │
  ├─ Cell 4: AdversarialTrainer
  │             n_iterations=500, steps_per_iteration=2048
  │             Adversary starts at iteration 100 (warm-up)
  │
  ├─ Cell N: WalkForward validation
  │             PurgedWalkForwardCV (gap period prevents leakage)
  │
  ├─ Continuous: Write colab_status.json to Drive (heartbeat)
  │
  └─ On error: POST to Linux listener / write error_report to Drive
                  → Auto-repair patches notebook parameters
                  → Restart signal triggers new session
```

### Walk-Forward Validation (Anti-Overfitting Core)

```
Full dataset (e.g., 3 years)
│
├─ Window 1 ─ Train [Y1-Jan..Y1-Dec]  Test [Y2-Jan..Y2-Mar]
├─ Window 2 ─ Train [Y1-Apr..Y2-Mar]  Test [Y2-Apr..Y2-Jun]
├─ Window 3 ─ Train [Y1-Jul..Y2-Jun]  Test [Y2-Jul..Y2-Sep]
│   ...
└─ Window N ─ Train [...]              Test [...]

For each window:
  1. Train DarwinArena / PPOAgent on IS window
  2. Evaluate champion on OOS window
  3. Compute IS Sharpe, OOS Sharpe, degradation ratio
  
Overfitting signal: IS Sharpe >> OOS Sharpe → strategy rejected
Acceptance criterion: OOS Sharpe ≥ 0.5 × IS Sharpe
```

The `PurgedWalkForwardCV` (from `validation/antibias_walkforward.py`) additionally introduces a gap period between train and test windows. This purges any samples whose labels overlap with the test period — a common source of subtle leakage in financial ML.

---

## 7. Error Recovery and Self-Healing Loop

The system is designed to recover from all foreseeable failures without human intervention.

```
  ┌─────────────────────────────────────────────────────────────────────┐
  │                     SELF-HEALING LOOP                               │
  │                                                                     │
  │  COLAB                          LINUX PC                            │
  │                                                                     │
  │  Training running               Master Orchestrator running         │
  │       │                              │                              │
  │       │ Every N iterations           │ Every 60 min                 │
  │       ▼                              ▼                              │
  │  Write colab_status.json ──▶ Drive ──▶ ColabWatchdog.read_status()  │
  │       │                                    │                        │
  │       │                              is_colab_alive?                │
  │       │                              │           │                  │
  │       │                             YES          NO                 │
  │       │                              │           │                  │
  │       │                          Continue    Send alert to Telegram │
  │       │                                      Send restart_signal    │
  │       │                                      to Drive               │
  │       │                                           │                 │
  │       ◀────────── Colab reads restart_requested.json               │
  │       │                                                             │
  │  Error occurs                                                       │
  │       │                                                             │
  │       ├─ Try HTTP POST to Linux listener (port 5001)                │
  │       │   (fast path, works on local network)                       │
  │       │                                                             │
  │       └─ Fallback: write error_report.json to Drive                 │
  │           (works even without direct network access)                │
  │                │                                                    │
  │                ▼                                                    │
  │         Linux Listener receives report                              │
  │                │                                                    │
  │                ▼                                                    │
  │         ErrorRepair.classify_error()                                │
  │                │                                                    │
  │                ├─ OOM       → halve batch_size in notebook          │
  │                ├─ NaN loss  → reduce learning_rate by 10x           │
  │                ├─ Timeout   → increase timeout parameter            │
  │                ├─ Import    → inject pip install cell               │
  │                └─ Connection→ send restart signal                   │
  │                │                                                    │
  │                ▼                                                    │
  │         Upload patched notebook to Drive                            │
  │                │                                                    │
  │                ▼                                                    │
  │         Colab picks up patched notebook on next session             │
  └─────────────────────────────────────────────────────────────────────┘
```

### Alert Severity Levels

| Severity | Minimum interval | Channels |
|---|---|---|
| INFO | 60 min | Log file |
| WARNING | 30 min | Log file + Desktop notification |
| CRITICAL | 5 min | Log file + Desktop + Telegram |

Deduplication prevents alert spam: the same message key will not be sent more frequently than the minimum interval for its severity level.

---

## 8. Key Design Decisions and Trade-offs

### Decision 1: Darwin Engine for Strategy Selection

**Why:** Neural network agents (PPO, SAC) need many samples and a GPU. For lightweight continuous local training, evolutionary strategies (RSI, MACD, Bollinger, EMA crossover) can be evaluated at >27,000× the speed of naive pandas loops using Numba JIT compilation and joblib parallelism. This allows the Linux PC to continuously find the best parameterisation without cloud GPU time.

**Trade-off:** Classical technical indicators are less expressive than deep RL agents. The solution is to run both: Darwin Engine on the PC for continuous cheap refinement, deep RL on Colab for expressive policy learning.

### Decision 2: Numba JIT for Backtesting Kernels

**Why:** The backtesting inner loop (`_kernel_simulate`) iterates bar by bar over thousands of candles, millions of times across populations and generations. Numba `@njit` compiles this to native machine code, yielding a ~100× speedup. The system degrades gracefully — if numba is unavailable, the same function runs as pure Python.

**Trade-off:** Numba adds a compilation overhead on the first call (~seconds). This is negligible for long training runs.

### Decision 3: Anti-Bias Framework

**Why:** The most common failure mode in backtested strategies is lookahead bias — using future data in training. The Anti-Bias Framework addresses this at three levels: (1) `PurgedWalkForwardCV` with a gap period between train and test, (2) `fit_transform`/`transform` separation in FeatureEngine, and (3) transaction cost modelling with realistic slippage that punishes overfitted high-frequency strategies.

**Trade-off:** More conservative validation reduces the number of strategies that pass. Accepting this false-negative rate is correct — the cost of deploying an overfitted strategy vastly exceeds the cost of discarding a slightly profitable one.

### Decision 4: Dual-Channel Error Reporting

**Why:** Colab sessions run behind Google's NAT. Direct inbound connections are impossible. The original plan (ngrok + open port) had two problems: (1) ngrok free tier changes URLs every 8 hours, and (2) an open port is a security risk. The dual-channel approach uses HTTP as the fast path when a tunnel is available, and Google Drive as the reliable fallback.

**Trade-off:** Drive polling adds 2-minute latency for error reports. This is acceptable since auto-repair patches the notebook for the *next* session, not the current one.

### Decision 5: 1%-Rule as Hard Constraint

**Why:** The Kelly formula alone does not prevent catastrophic drawdowns in practice because win-rate estimates are noisy. The 1%-rule is a simple, robust constraint that caps the worst-case per-trade loss regardless of Kelly. The combined system (Kelly sizing, capped at 1%-rule) provides two independent lines of defence.

**Trade-off:** The 1%-rule may undersize positions relative to Kelly in high-confidence scenarios. Fractional Kelly (50%) already halves the theoretical optimal, so the combination is conservative by design.

### Decision 6: Adversarial Training

**Why:** Standard RL training on historical data tends to produce strategies that exploit specific patterns in the training distribution. The adversary forces the trader to develop more general defensive strategies by actively creating the most damaging market conditions it can.

**Trade-off:** Adversarial training doubles the training compute and requires careful tuning of `adversary_strength` (default: 0.1). Too strong an adversary prevents the trader from learning at all; too weak an adversary provides no benefit over standard training. The 100-iteration warm-up period ensures the trader has a usable baseline before the adversary starts.
