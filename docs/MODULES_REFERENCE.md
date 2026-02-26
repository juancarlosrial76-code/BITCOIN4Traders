# BITCOIN4Traders — Modules Reference

Complete reference for every module in `src/`. Each entry covers the module's purpose, primary classes and functions, inputs/outputs, and important parameters.

---

## Table of Contents

1. [agents/](#1-agents)
2. [backtesting/](#2-backtesting)
3. [causal/](#3-causal)
4. [connectors/](#4-connectors)
5. [costs/](#5-costs)
6. [data/](#6-data)
7. [data_processors/](#7-data_processors)
8. [data_quality/](#8-data_quality)
9. [ensemble/](#9-ensemble)
10. [environment/](#10-environment)
11. [evaluation/](#11-evaluation)
12. [execution/](#12-execution)
13. [features/](#13-features)
14. [math_tools/](#14-math_tools)
15. [meta_learning/](#15-meta_learning)
16. [monitoring/](#16-monitoring)
17. [orders/](#17-orders)
18. [portfolio/](#18-portfolio)
19. [research/](#19-research)
20. [reward/](#20-reward)
21. [risk/](#21-risk)
22. [surveillance/](#22-surveillance)
23. [training/](#23-training)
24. [transformer/](#24-transformer)
25. [utils/](#25-utils)
26. [validation/](#26-validation)

---

## 1. agents/

### `ppo_agent.py` — Proximal Policy Optimization

**Purpose:** Primary RL agent. Implements PPO (Schulman et al. 2017) with recurrent extensions for temporal market context.

#### `PPOConfig` (dataclass)

All hyperparameters in one place. Key parameters:

| Parameter | Default | Description |
|---|---|---|
| `state_dim` | required | Input feature dimensionality |
| `hidden_dim` | 128 | Width of MLP layers |
| `n_actions` | 3 | Buy / Hold / Sell |
| `use_recurrent` | True | Enable GRU/LSTM temporal memory |
| `rnn_type` | "GRU" | "GRU" or "LSTM" |
| `actor_lr` | 3e-4 | Actor learning rate |
| `critic_lr` | 1e-3 | Critic learning rate (higher for faster value convergence) |
| `gamma` | 0.99 | Discount factor |
| `gae_lambda` | 0.95 | GAE smoothing parameter |
| `clip_epsilon` | 0.2 | PPO surrogate clip ratio |
| `n_epochs` | 10 | Training epochs per rollout |
| `entropy_coef` | 0.01 | Entropy bonus (exploration) |
| `target_kl` | 0.01 | Early stop if KL exceeds this |

#### `BaseNetwork` (nn.Module)

Shared feature extractor used by both Actor and Critic.

Architecture:
```
Linear(state_dim, hidden_dim)
→ LayerNorm (optional)
→ ReLU
→ Dropout (optional)
→ Linear(hidden_dim, hidden_dim)
→ LayerNorm (optional)
→ ReLU
→ Dropout (optional)
→ GRU / LSTM (optional recurrent layer)
→ output_dim
```

#### `PPOAgent`

Main agent class.

**Key methods:**

| Method | Input | Output | Description |
|---|---|---|---|
| `select_action(state, hidden)` | state array, hidden state | action int, log_prob, value, new_hidden | Forward pass, samples from policy |
| `update(rollout_buffer)` | RolloutBuffer | dict of losses | PPO update step with GAE |
| `save(path)` / `load(path)` | file path | — | Checkpoint save/load |

---

### `drl_agents.py` — Extended DRL Algorithms

**Purpose:** Additional algorithms for experimentation and ensemble construction.

#### `DQNAgent`

Deep Q-Network for discrete action spaces.

| Parameter | Default | Description |
|---|---|---|
| `epsilon_start` | 1.0 | Initial exploration rate |
| `epsilon_end` | 0.01 | Minimum exploration rate |
| `epsilon_decay` | 0.995 | Multiplicative decay per step |
| `buffer_size` | 100000 | Experience replay buffer size |
| `target_update` | 1000 | Steps between target network sync |

**Key methods:** `select_action(state)`, `train_step()`, `update_target()`

#### `A2CAgent`

Advantage Actor-Critic. Synchronous n-step returns, no replay buffer.

**Key methods:** `select_action(state)`, `update(states, actions, returns)`

#### `SACAgent`

Soft Actor-Critic. Off-policy, maximum entropy. Suitable for continuous action spaces or discrete via Gumbel-Softmax. Includes automatic entropy temperature tuning.

**Key methods:** `select_action(state, deterministic)`, `update(batch)`

#### `TD3Agent`

Twin Delayed DDPG. Two Q-networks to reduce overestimation. Delayed policy updates (every 2 critic updates). Target policy smoothing (Gaussian noise on target actions).

**Key methods:** `select_action(state, noise)`, `update(batch)`

#### `ReplayBuffer`

Experience replay shared across DQN, SAC, TD3.

**Key methods:** `push(state, action, reward, next_state, done)`, `sample(batch_size)`

---

## 2. backtesting/

### `strategy_evolution.py` — Darwinian Strategy Selection

**Purpose:** Evaluate multiple classical technical strategies on historical data and select the winner by fee-adjusted profit factor.

#### `StrategyResult` (dataclass)

Output of a single strategy evaluation:
- `name`, `profit_factor`, `fee_adjusted_pf`, `win_rate`, `n_trades`
- `total_return`, `sharpe`, `status` ("KEEP" or "DISCARD")
- `signal_column`: which DataFrame column carries the signal

#### `EvolutionReport` (dataclass)

- `results`: list of all StrategyResult
- `best_strategy`: top StrategyResult by fee-adjusted PF
- `ranking` (property): sorted list, best first
- `to_dataframe()`: returns formatted DataFrame for display

#### `SignalBuilder`

Generates raw entry/exit signals from OHLCV data for a named strategy.

**Input:** OHLCV DataFrame + strategy name  
**Output:** DataFrame with signal column added

#### `SignalBacktester`

Runs a bar-by-bar backtest on a signal column.

**Input:** OHLCV DataFrame with signal column, fee rate  
**Output:** `StrategyResult`

---

### `walkforward_engine.py` — Rolling Walk-Forward Validation

**Purpose:** Prevent overfitting by training and evaluating on non-overlapping time windows.

#### `WalkForwardConfig` (dataclass)

| Parameter | Default | Description |
|---|---|---|
| `train_window_days` | 365 | Training window length |
| `test_window_days` | 90 | Out-of-sample test window |
| `step_days` | 30 | Step size (window advance) |
| `train_iterations` | 200 | RL training iterations per window |
| `min_trades` | 10 | Minimum trades in test period |
| `max_drawdown_threshold` | 0.30 | Reject if OOS drawdown > 30% |

#### `WalkForwardEngine`

**Key methods:**

| Method | Description |
|---|---|
| `run(data, agent_config)` | Execute all windows, return list of results |
| `analyze_results(results)` | Compute IS/OOS ratio, degradation |
| `save_results(path)` | Persist results as JSON |

**Anti-Bias integration:** If `validation/antibias_walkforward.py` is available, uses `PurgedWalkForwardCV` with a configurable gap period between train and test to prevent label leakage.

---

### `performance_calculator.py` — Metrics

**Purpose:** Compute the full set of risk-adjusted performance metrics from a trade log or equity curve.

**Key functions:**

| Function | Input | Output |
|---|---|---|
| `sharpe_ratio(returns, rf)` | daily returns array, risk-free rate | annualised Sharpe |
| `calmar_ratio(equity_curve)` | equity curve | annual return / max drawdown |
| `sortino_ratio(returns, target)` | returns, minimum acceptable return | Sortino ratio |
| `max_drawdown(equity_curve)` | equity curve | maximum drawdown fraction |
| `profit_factor(wins, losses)` | gross wins, gross losses | PF = sum(wins) / sum(losses) |
| `var_historical(returns, confidence)` | returns array, confidence level | VaR estimate |

---

### `stress_tester.py` — Black Swan Testing

**Purpose:** Inject extreme scenarios into historical data to test strategy robustness.

#### `CrashSimulator`

**Scenarios (configured via `config/stress_test.yaml`):**

| Scenario | Description | Default parameters |
|---|---|---|
| `flash_crash` | Linearly ramp price down over N candles | `drop_pct=0.20`, `duration_candles=5` |
| `volatility_spike` | Multiply all returns by a factor | `multiplier=3.0` |

**Key methods:** `inject_anomalies(df)` → modified DataFrame

#### `StressTester`

Runs the full evolution pipeline on stress-injected data and compares results to clean-data baseline.

**Key method:** `run(df)` → dict with clean metrics vs stressed metrics

---

### `visualizer.py` — Backtesting Charts

**Purpose:** Generate equity curves, drawdown charts, and performance summaries.

**Key functions:** `plot_equity_curve(equity)`, `plot_drawdown(equity)`, `plot_trade_distribution(trades)`, `performance_table(results)`

---

## 3. causal/

### `causal_inference.py` — Causal Relationship Discovery

**Purpose:** Go beyond correlation — identify which variables causally drive price movements using DoWhy-inspired methods.

#### `CausalEffect` (dataclass)

Represents a discovered causal relationship:
- `treatment`, `outcome`, `effect_size`, `p_value`
- `confidence_interval`, `method`, `valid_instruments`

#### `CausalDiscovery`

Discovers the causal graph from observational data using conditional independence tests.

**Key methods:**

| Method | Input | Output |
|---|---|---|
| `fit(data)` | DataFrame with features | Fitted causal graph (NetworkX DiGraph) |
| `discover_causes(target)` | target variable name | List of causal variables |
| `get_causal_graph()` | — | NetworkX DiGraph |

#### `InstrumentalVariables`

Estimates causal effect using instrumental variables (avoids confounding bias).

**Input:** treatment variable, outcome variable, instrument variable, DataFrame  
**Output:** `CausalEffect` with IV estimate

#### `CounterfactualReasoning`

Answers "what would have happened if...?" questions by intervening on causal variables.

**Key method:** `compute_counterfactual(data, intervention)` → counterfactual DataFrame

---

## 4. connectors/

### `binance_connector.py` — REST API Connector

**Purpose:** Production-ready Binance REST API client for spot and futures trading.

**Authentication:** `BINANCE_API_KEY` and `BINANCE_API_SECRET` from environment variables.

#### `Order` (dataclass)

Fields: `symbol`, `side` (BUY/SELL), `order_type` (MARKET/LIMIT), `quantity`, `price`, `order_id`, `status`, `filled_qty`, `avg_price`

#### `Position` (dataclass)

Fields: `symbol`, `side`, `quantity`, `entry_price`, `current_price`, `unrealised_pnl`, `leverage`

#### `BinanceConnector`

| Method | Description |
|---|---|
| `place_market_order(symbol, side, quantity)` | Submit market order |
| `place_limit_order(symbol, side, quantity, price)` | Submit limit order |
| `cancel_order(symbol, order_id)` | Cancel open order |
| `get_position(symbol)` | Current position |
| `get_account_balance()` | Current equity and available margin |
| `get_historical_klines(symbol, interval, start, end)` | OHLCV history |
| `subscribe_ticker(symbol, callback)` | Real-time price stream |

**Retry logic:** Exponential backoff on `BinanceAPIException`. Rate limiting compliance via configurable `rate_limit_ms`.

---

### `binance_ws_connector.py` — WebSocket Connector

**Purpose:** Asynchronous WebSocket for real-time market data and user account stream.

#### `ReconnectPolicy` (dataclass)

| Parameter | Default | Description |
|---|---|---|
| `max_retries` | 10 | Maximum reconnect attempts |
| `base_delay` | 1.0 | Initial backoff delay (seconds) |
| `max_delay` | 60.0 | Maximum backoff delay |
| `jitter` | True | Randomise reconnect delay |

#### `BinanceWSConnector`

**Key methods:**

| Method | Description |
|---|---|
| `connect()` | Establish WebSocket connection |
| `subscribe_trades(symbol)` | Real-time trade stream |
| `subscribe_orderbook(symbol, depth)` | Order book snapshot + updates |
| `subscribe_user_stream()` | Order fills, position updates |
| `disconnect()` | Clean shutdown |

Internally manages reconnection with exponential backoff, message queuing during reconnects, and sequence number validation for order book consistency.

---

## 5. costs/

### `antibias_costs.py` — Realistic Transaction Cost Engine

**Purpose:** Model every component of real trading costs to prevent agents from learning strategies that only work when costs are ignored.

#### Enums

- `MarketType`: SPOT, FUTURES
- `Timeframe`: M1, M5, M15, H1, H4
- `OrderType`: MARKET, LIMIT

#### Fee schedules (hardcoded from Binance, conservative)

| Market | Taker | Maker |
|---|---|---|
| Spot | 0.10% | 0.10% |
| Futures | 0.05% | 0.02% |

#### `CostConfig` (dataclass)

| Parameter | Default | Description |
|---|---|---|
| `market_type` | SPOT | Spot or Futures |
| `timeframe` | H1 | Affects spread multiplier |
| `order_type` | MARKET | Market or limit |
| `include_funding` | True | Include futures funding rate |
| `include_impact` | True | Include market impact |

#### `TransactionCostEngine`

**Key method:** `compute(position_size, price, volatility, volume)` → total cost in basis points

Cost components added:
1. Exchange fee (maker or taker based on order type)
2. Bid-ask spread × timeframe multiplier (shorter timeframes have wider relative spreads)
3. Market impact (quadratic in order size / daily volume)
4. Funding rate (futures only): sampled from historical BTC perpetual distribution (mean=0.01%, std=0.03%, max=0.75% per 8-hour period)

---

## 6. data/

### `ccxt_loader.py` — CCXT Data Loader

**Purpose:** Fetch OHLCV from any CCXT-compatible exchange with Parquet caching.

#### `DataLoaderConfig` (dataclass)

| Parameter | Description |
|---|---|
| `exchange_id` | e.g. "binance", "bybit" |
| `exchange_type` | "spot" or "futures" |
| `rate_limit_ms` | Milliseconds between requests |
| `cache_dir` | Path to Parquet cache |
| `compression` | "snappy" (default, fast) |

#### `CCXTDataLoader`

**Key methods:**

| Method | Input | Output |
|---|---|---|
| `fetch(symbol, timeframe, since, limit)` | Symbol string, timeframe, start date | OHLCV DataFrame |
| `fetch_with_cache(symbol, timeframe, since)` | — | Cached or freshly fetched DataFrame |
| `invalidate_cache(symbol, timeframe)` | — | Deletes cached file |

**Cache format:** `{cache_dir}/{exchange}_{symbol}_{timeframe}.parquet`  
**Atomic writes:** Writes to `.tmp` first, then renames to prevent partial file corruption.

---

### `data_manager.py` — Unified Data Access

**Purpose:** Single entry point for all data access. Abstracts over CCXTDataLoader, HistoricalDownloader, and SQLite/PostgreSQL.

**Key methods:** `get_ohlcv(symbol, timeframe, start, end)`, `get_latest(symbol, timeframe, n_bars)`, `update(symbol, timeframe)`, `validate(df)`

---

### `database.py` — Database Manager

**Purpose:** PostgreSQL storage for historical market data.

#### `MarketData` (SQLAlchemy model)

Fields: `id`, `symbol`, `timeframe`, `timestamp`, `open`, `high`, `low`, `close`, `volume`

#### `DatabaseManager`

**Key methods:** `store(df, symbol, timeframe)`, `query(symbol, timeframe, start, end)`, `count(symbol, timeframe)`, `delete_old(days)`

---

### `historical_pipeline.py` — Historical Downloader

**Purpose:** Download years of historical data from Binance and store in the database.

#### `HistoricalDataDownloader`

**Key methods:**

| Method | Input | Description |
|---|---|---|
| `download_symbol(symbol, timeframe, start, end)` | Symbol, TF, date range | Downloads and stores all bars |
| `download_all(symbols, timeframes)` | Lists | Downloads all combinations |
| `update_to_latest(symbol, timeframe)` | Symbol, TF | Incremental update from last stored bar |

**Rate limiting:** Configurable delay between API calls. Progress bar via `tqdm`.

---

### `sqlite_local.py` — SQLite Local Storage

**Purpose:** Lightweight local alternative to PostgreSQL for development and single-machine deployments.

**Key class:** `SQLiteManager` — same interface as `DatabaseManager` but backed by SQLite.

---

## 7. data_processors/

### `processors.py` — Data Preprocessing Pipeline

**Purpose:** Clean, normalise, and split raw OHLCV data before it reaches the feature engine.

**Key functions:**
- `remove_duplicates(df)` — deduplicate by timestamp
- `fill_gaps(df, method)` — forward-fill or interpolate missing bars
- `detect_outliers(df, sigma)` — flag extreme price moves (default: 5σ)
- `train_test_split(df, test_fraction)` — time-series aware split (no shuffling)
- `normalise_volume(df)` — scale volume to [0, 1] range

---

## 8. data_quality/

### `assessor.py` — Data Quality Assessment

**Purpose:** Validate historical data before it enters the training pipeline.

**Key class:** `DataQualityAssessor`

**Checks performed:**
- Missing bar detection (gaps in timestamp index)
- Duplicate timestamps
- Price/volume outliers (configurable sigma threshold)
- OHLC integrity (high ≥ open, close, low ≥ 0, etc.)
- Distribution shift detection between train and live data

**Output:** `QualityReport` dataclass with pass/fail per check and a summary score.

---

### `live_monitor.py` — Streaming Quality Gate

**Purpose:** Monitor incoming live ticks for anomalies before feeding them to the live trading pipeline.

**Key class:** `LiveDataMonitor`

Raises alerts (via AlertManager) when:
- Tick arrives outside expected time window
- Price deviates more than N sigma from recent rolling mean
- Volume is zero or negative
- Stale data (no tick for more than `timeout_seconds`)

---

## 9. ensemble/

### `ensemble_agents.py` — Agent Ensemble Methods

**Purpose:** Combine predictions from multiple trained agents to reduce variance and improve decision robustness.

#### `EnsembleConfig` (dataclass)

| Parameter | Default | Description |
|---|---|---|
| `method` | "voting" | "voting", "weighted", "stacking", "bagging" |
| `weights` | None | Per-agent weights for weighted ensemble |
| `window_size` | 10 | Rolling performance window for adaptive weights |
| `temperature` | 1.0 | Softmax temperature for weight sharpening |

#### `AgentEnsemble`

**Key methods:**

| Method | Description |
|---|---|
| `predict(state)` | Return consensus action |
| `predict_with_confidence(state)` | Return action + confidence score |
| `update_weights(rewards)` | Adapt weights based on recent agent performance |

**Ensemble methods:**

- **Voting:** Majority vote. Ties broken by highest-confidence agent.
- **Weighted:** Softmax-weighted average of Q-values / policy logits.
- **Stacking:** A meta-learner (linear model) trained on agent outputs.
- **Bagging:** Bootstrap aggregating — each agent trained on a bootstrapped data subset.

**Adaptive weighting:** Tracks rolling reward per agent. Agents outperforming the mean get higher weights; underperforming agents are down-weighted.

---

## 10. environment/

### `realistic_trading_env.py` — Primary Trading Environment

**Purpose:** Gymnasium-compatible trading environment with realistic execution modelling. "If the model loses here, it won't make money live."

#### `TradingEnvConfig` (dataclass)

| Parameter | Default | Description |
|---|---|---|
| `initial_capital` | 100,000 | Starting capital in USD |
| `transaction_cost_bps` | 5.0 | Transaction cost (0.05%) |
| `slippage_model` | "volume_based" | fixed / volume_based / volatility / orderbook |
| `use_orderbook` | True | Simulate L2 order book |
| `orderbook_levels` | 10 | Number of price levels |
| `max_position_size` | 1.0 | Maximum 100% of capital |
| `lookback_window` | 50 | Feature lookback bars |
| `max_steps` | 5000 | Episode length in bars |
| `max_drawdown` | 0.20 | Episode terminates at 20% drawdown |
| `use_antibias_rewards` | True | Enable risk-adjusted rewards |
| `reward_type` | "cost_aware" | "sharpe" / "cost_aware" / "regime_aware" |

#### `RealisticTradingEnv` (gym.Env)

**State space:** Concatenation of:
- Price features from `FeatureEngine` (normalised, stationary)
- Current position: -1 (short), 0 (flat), +1 (long)
- Portfolio metrics: PnL, unrealised PnL, current drawdown
- Order book features: spread, bid depth, ask depth, imbalance

**Action space:** `Discrete(3)` — 0=Sell/Short, 1=Hold/Flat, 2=Buy/Long

**Reward:** Depends on `reward_type`:
- `"sharpe"`: (return - rolling_mean) / rolling_std
- `"cost_aware"`: return - transaction_costs - drawdown_penalty
- `"regime_aware"`: scaled by HMM regime confidence

**Key methods:** `reset()`, `step(action)` → (obs, reward, done, info)

---

### `crypto_futures_env.py` — Futures-Specific Environment

**Purpose:** Extends `RealisticTradingEnv` with futures-specific features: leverage, funding rate costs, liquidation risk, and margin tracking.

---

### `config_integrated_env.py` — Hydra-Configured Environment

**Purpose:** Wraps `RealisticTradingEnv` to receive its full configuration from Hydra config files rather than constructor arguments. Used in the Colab training pipeline.

---

### `order_book.py` — Level 2 Order Book Simulator

**Purpose:** Simulate a realistic 10-level bid/ask book to calculate execution price with proper slippage.

#### `OrderBookConfig` (dataclass)

| Parameter | Default | Description |
|---|---|---|
| `n_levels` | 10 | Price levels on each side |
| `base_spread_bps` | 5.0 | Base bid/ask spread in basis points |
| `depth_factor` | 1.0 | Liquidity depth multiplier |
| `impact_coefficient` | 0.1 | Price impact coefficient |

#### `OrderBookSimulator`

**Key methods:**

| Method | Input | Output |
|---|---|---|
| `generate_order_book(mid_price, volatility, volume)` | Market state | (bid_prices, bid_vols, ask_prices, ask_vols) |
| `calculate_execution_price(side, quantity, order_book)` | Order details | Actual execution price |
| `calculate_market_impact(side, quantity)` | Order details | Price impact in BPS |

---

### `slippage_model.py` — Slippage Calculation

**Purpose:** Compute the difference between expected and actual execution price for each order type.

#### `SlippageConfig` (dataclass)

| Parameter | Default | Description |
|---|---|---|
| `model_type` | "volume_based" | fixed / volume_based / volatility / orderbook |
| `fixed_slippage_bps` | 5.0 | For fixed model |
| `volatility_multiplier` | 2.0 | For volatility model |
| `volume_impact_coef` | 0.1 | For volume-based model |

#### `SlippageModel`

**Key method:** `calculate_slippage(side, quantity, price, volume, volatility)` → (execution_price, slippage_bps)

**Models:**
- **Fixed:** `execution_price = price ± fixed_bps`
- **Volume-based:** `slippage ∝ quantity / daily_volume × impact_coef`
- **Volatility:** `slippage ∝ current_volatility × multiplier`
- **Order book:** Walks the synthetic order book to compute realistic fill

#### `TransactionCostModel`

Wraps SlippageModel and adds exchange fees.

**Key method:** `total_cost(side, quantity, price, volume, volatility)` → (net_price, total_cost_bps)

---

## 11. evaluation/

### `antibias_validator.py` — Bias Detection Validator

**Purpose:** Systematic detection of common ML biases in trained trading strategies before promotion to live trading.

#### `AntiBiasValidator`

**Checks performed:**

| Check | Description |
|---|---|
| Lookahead bias | Verifies no future data in feature computation |
| Survivorship bias | Tests on delisted/failed assets if available |
| Data snooping | Counts strategy variations tested (Bonferroni correction) |
| Overfitting score | Compares IS vs OOS Sharpe ratio |
| Transaction cost sensitivity | Re-runs backtest at 2× and 5× stated costs |

**Key method:** `validate(strategy, train_data, test_data)` → `ValidationReport`

---

## 12. execution/

### `live_engine.py` — Live Trading Orchestrator

**Purpose:** The Phase 7 execution engine. Coordinates WebSocket data, feature computation, agent inference, risk checks, and order submission in a real-time async loop.

#### `Position` (dataclass)

| Field | Type | Description |
|---|---|---|
| `symbol` | str | Trading symbol |
| `qty` | Decimal | Signed quantity (+long, -short) |
| `avg_cost` | Decimal | VWAP average entry price |
| `realized_pnl` | Decimal | Cumulative closed P&L |

**Key method:** `update_fill(side, fill)` → realized PnL of fill  
Uses VWAP cost basis for partial fills.

#### `LiveEngine`

**Main execution loop per tick:**
```
1. WS tick received
2. FeatureEngine.transform(tick)
3. agent.select_action(features)
4. risk_manager.validate(signal, position, equity)
5. order_manager.submit(order)
6. Handle fill callback
7. Update position and P&L
8. Send monitoring update
```

**Key methods:** `start()`, `stop()`, `on_tick(tick_data)`, `on_fill(fill_event)`

---

### `execution_algorithms.py` — Smart Order Routing

**Purpose:** Execute large orders with minimal market impact using institutional execution algorithms.

#### `TWAPExecutor`

Time-Weighted Average Price: splits order into equal slices over a time window.

**Parameters:** `total_quantity`, `duration_seconds`, `n_slices`

#### `VWAPExecutor`

Volume-Weighted Average Price: sizes slices proportional to expected volume profile.

**Parameters:** `total_quantity`, `duration_seconds`, `volume_profile` (intraday volume curve)

#### `IcebergExecutor`

Shows only a small visible portion of the order at a time.

**Parameters:** `total_quantity`, `visible_quantity`, `price_tolerance_bps`

---

## 13. features/

### `feature_engine.py` — Institutional Feature Engineering

**Purpose:** Transform raw OHLCV into normalised, stationary features using a scikit-learn-style `fit`/`transform` pattern that prevents data leakage.

#### `FeatureConfig` (dataclass, Hydra-injected)

| Parameter | Description |
|---|---|
| `volatility_window` | Bars for rolling volatility |
| `ou_window` | Bars for OU mean-reversion signal |
| `rolling_mean_window` | Bars for rolling mean |
| `use_log_returns` | Use log returns (recommended: True) |
| `scaler_type` | "standard" / "minmax" / "robust" |
| `save_scaler` | Persist scaler to disk |
| `dropna_strategy` | "forward_fill" / "drop" / "zero" |

#### `FeatureEngine`

**Key methods:**

| Method | Input | Output | Notes |
|---|---|---|---|
| `fit_transform(df)` | Training DataFrame | Feature array | Fits scaler, stores stats |
| `transform(df)` | Any DataFrame | Feature array | Uses ONLY training stats |
| `get_feature_names()` | — | List of str | For interpretability |
| `save(path)` / `load(path)` | file path | — | Serialises with pickle |

**Features computed (Numba-accelerated where possible):**
- Log returns
- Rolling volatility (std of returns over `volatility_window` bars)
- Rolling mean return
- OU process deviation (price - rolling mean, normalised by volatility)
- Optional: RSI, MACD, Bollinger Bands, EMA crossover signals

---

### `multi_timeframe.py` — Multi-Timeframe Analysis

**Purpose:** Aggregate signals across multiple timeframes for top-down market analysis.

#### `Timeframe` (Enum)

M1, M5, M15, M30, H1, H4, D1, W1

#### `SignalStrength` (dataclass)

Fields: `timeframe`, `direction` (-1/0/+1), `strength` (0.0–1.0), `confidence` (0.0–1.0)

#### `MultiTimeframeAnalyzer`

**Key methods:**

| Method | Input | Output |
|---|---|---|
| `analyze(df_by_tf)` | Dict[Timeframe, DataFrame] | Dict[Timeframe, SignalStrength] |
| `get_alignment_score()` | — | Float: how aligned are all timeframes |
| `get_dominant_direction()` | — | -1, 0, or +1 |

**Top-down analysis logic:** Weekly and daily signals carry higher weight. An entry signal on H1 is only taken if it agrees with the D1 and W1 direction.

---

### `microstructure.py` — Market Microstructure Features

**Purpose:** Extract trading-relevant signals from L2 order book data.

#### `MicrostructureAnalyzer`

**Features computed:**

| Feature | Description |
|---|---|
| Bid-ask spread | Current spread in BPS |
| Order flow imbalance | (buy_vol - sell_vol) / (buy_vol + sell_vol) |
| Quote-to-trade ratio | Proxy for spoofing/manipulation intensity |
| Effective spread | Twice the signed distance from mid to trade price |
| Kyle's lambda | Price impact per unit of order flow |
| Amihud illiquidity | Price change / volume ratio |

---

## 14. math_tools/

### `kalman_filter.py` — Kalman Filter

**Purpose:** Optimal recursive state estimation for price trend extraction.

#### `KalmanFilterConfig` (dataclass)

| Parameter | Default | Description |
|---|---|---|
| `Q` | 0.001 | Process noise (state change uncertainty) |
| `R` | 0.1 | Measurement noise (observation uncertainty) |
| `A` | 1.0 | State transition coefficient |
| `H` | 1.0 | Observation coefficient |

#### `KalmanFilter1D`

**Key methods:**
- `update(observation)` → (filtered_state, uncertainty)
- `predict()` → (predicted_state, predicted_uncertainty)
- `smooth(observations)` → smoothed state array (Rauch-Tung-Striebel smoother)

#### `KalmanPairsTrader`

Extension for dynamic hedge ratio estimation in pairs trading. Tracks the ratio of two cointegrated assets as a time-varying Kalman state.

---

### `garch_models.py` — GARCH Volatility Models

**Purpose:** Model volatility clustering for VaR calculation and position sizing.

**Mathematical model:**  
σ²_t = ω + α·ε²_{t-1} + β·σ²_{t-1}

#### `GARCHModel`

**Key methods:**

| Method | Input | Output |
|---|---|---|
| `fit(returns)` | Return series | Fitted parameters (ω, α, β) |
| `forecast(h)` | Horizon (steps) | Conditional variance forecast |
| `var(confidence)` | Confidence level | Value at Risk estimate |

**Parameter estimation:** Maximum likelihood via `scipy.optimize.minimize` with BFGS. Stationarity constraint: α + β < 1.

#### `EGARCHModel`

Exponential GARCH — captures asymmetric volatility (bad news increases vol more than good news).

#### `GARCHForecaster`

Rolling GARCH estimation with automatic re-fitting every N bars.

---

### `hmm_regime.py` — Hidden Markov Market Regimes

**Purpose:** Classify the current market state (bull/bear/transition) using an HMM trained on returns, volatility, and volume.

#### `MarketRegime` (dataclass)

Fields: `id`, `name`, `mean_return`, `mean_volatility`, `probability`

#### `HMMRegimeDetector`

**Key methods:**

| Method | Input | Output |
|---|---|---|
| `fit(features)` | Feature array | Fitted HMM |
| `predict(features)` | Feature array | Array of regime IDs |
| `predict_proba(features)` | Feature array | Regime probability matrix |
| `current_regime(features)` | Current feature vector | `MarketRegime` with highest probability |

**Typical regimes (n_regimes=3):**
- Regime 0: Low volatility, positive drift (bull)
- Regime 1: High volatility, negative drift (bear)
- Regime 2: Medium volatility, near-zero drift (transition/sideways)

**Integration with rewards:** `RegimeAwareReward` uses `predict_proba()` to scale risk tolerance per step.

---

### `kelly_criterion.py` — Optimal Position Sizing

**Purpose:** Compute the theoretically optimal fraction of capital to allocate per trade.

**Formula:** f* = (p·b - q) / b  
Where p = win probability, q = 1-p, b = win/loss ratio.

#### `KellyParameters` (dataclass)

| Parameter | Default | Description |
|---|---|---|
| `win_probability` | required | Historical win rate |
| `win_loss_ratio` | required | avg_win / avg_loss |
| `kelly_fraction` | 0.5 | Fractional Kelly (50% of full Kelly) |
| `max_position` | 0.25 | Hard cap: max 25% of capital |

#### `KellyCriterion`

**Key methods:**

| Method | Input | Output |
|---|---|---|
| `calculate_kelly_fraction(win_prob, wl_ratio)` | p, b | Full Kelly fraction |
| `estimate_parameters(trade_history)` | List of trade outcomes | `KellyParameters` |
| `calculate_position_size(capital, params)` | Capital, KellyParameters | Position size in USD |
| `dynamic_kelly(capital, recent_win_rate, recent_pf)` | — | Dynamically adjusted size |

**Safety rule:** Always uses Fractional Kelly (50%) and applies `max_position` cap. Never recommends > 25% of capital in a single trade.

---

### `hurst_exponent.py` — Hurst Exponent

**Purpose:** Classify price series as trending (H > 0.5), mean-reverting (H < 0.5), or random walk (H ≈ 0.5).

**Key method:** `compute_hurst(prices, lags)` → float in [0, 1]

**Interpretation:**
- H > 0.6: Strong trend — use trend-following strategies
- H ≈ 0.5: Random walk — discretion required
- H < 0.4: Strong mean-reversion — use mean-reversion strategies

---

### `spectral_analysis.py` — Frequency Domain Analysis

**Purpose:** Detect cyclical patterns in price series using Fourier and Lomb-Scargle methods.

**Key functions:**
- `fft_analysis(prices)` → dominant frequencies and their amplitudes
- `lomb_scargle(times, prices, frequencies)` → power spectrum for unevenly spaced data
- `detect_cycles(prices, min_period, max_period)` → list of (period, strength) tuples

---

### `cointegration.py` — Cointegration Analysis

**Purpose:** Identify pairs of assets whose price spread is stationary (mean-reverting), enabling pairs trading.

**Key functions:**
- `engle_granger(y1, y2)` → (cointegration stat, p-value, hedge ratio)
- `johansen_test(price_matrix, rank)` → trace statistic and critical values
- `estimate_hedge_ratio(y1, y2)` → optimal hedge ratio via OLS or Kalman

---

### `ornstein_uhlenbeck.py` — Ornstein-Uhlenbeck Process

**Purpose:** Model and estimate mean-reversion speed for spread trading.

**SDE:** dX_t = θ(μ - X_t)dt + σ dW_t  
Where θ = reversion speed, μ = long-run mean, σ = volatility.

**Key class:** `OrnsteinUhlenbeck`

**Key methods:**
- `fit(series)` → (θ, μ, σ) by MLE
- `half_life()` → ln(2) / θ (time for deviation to halve)
- `simulate(n_steps)` → simulated OU path

---

## 15. meta_learning/

### `meta_trader.py` — MAML Meta-Learning

**Purpose:** Learn a policy initialisation that adapts to any new market with only a few gradient steps. "Learning to learn."

#### `MetaLearningConfig` (dataclass)

| Parameter | Default | Description |
|---|---|---|
| `inner_lr` | 0.01 | Learning rate for per-task adaptation |
| `meta_lr` | 0.001 | Learning rate for meta-parameter update |
| `inner_steps` | 5 | Gradient steps per task |
| `meta_batch_size` | 4 | Number of tasks per meta-update |
| `first_order` | False | First-order MAML (faster, less memory) |

#### `MAMLTrader`

**Key methods:**

| Method | Description |
|---|---|
| `inner_update(task_data)` | Adapt parameters to a single task (market condition) |
| `meta_update(tasks)` | Update meta-parameters from a batch of tasks |
| `adapt(new_market_data, n_steps)` | Rapidly adapt to a new market with few examples |

**Continual learning:** `ContinualLearner` subclass uses Elastic Weight Consolidation (EWC) to prevent catastrophic forgetting when adapting to new markets.

---

## 16. monitoring/

### `monitor.py` — Real-Time Monitoring

**Purpose:** Production monitoring with Telegram alerts, Prometheus metrics, and live P&L dashboard.

#### `AlertLevel`

Constants: INFO, WARNING, CRITICAL, FILL, ERROR (with emoji indicators).

#### `TelegramNotifier`

Sends alerts to a Telegram chat asynchronously via an internal message queue (bounded at 200 messages to prevent memory bloat).

**Key methods:** `send(message, level)`, `start()`, `stop()`

**Configuration:** `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` environment variables.

#### `PrometheusMetrics`

Exposes metrics for Grafana dashboards:
- `trading_equity_total`: Current account equity
- `trading_pnl_daily`: Daily P&L
- `trading_position_size`: Current position
- `trading_drawdown_pct`: Current drawdown
- `trading_orders_total`: Order count by status

**Key method:** `expose(port)` → starts HTTP server at `/metrics`

#### `PnLDashboard`

Rolling console display of P&L, drawdown, position, and recent trades.

---

### `production_monitor.py` — Production-Grade Monitor

**Purpose:** Higher-level orchestration of monitoring in production. Coordinates TelegramNotifier, PrometheusMetrics, structured JSON logging, and anomaly detection.

**Key class:** `ProductionMonitor`

**Monitored events:** Order fills, circuit breaker triggers, drawdown thresholds, agent action anomalies, data quality issues.

---

## 17. orders/

### `order_manager.py` — Order Lifecycle Manager

**Purpose:** Track the full lifecycle of every order from submission to final fill or cancellation.

#### Enums

- `OrderSide`: BUY, SELL
- `OrderType`: MARKET, LIMIT, STOP, STOP_LIMIT
- `OrderStatus`: PENDING, SUBMITTED, PARTIAL, FILLED, CANCELLED, REJECTED
- `TimeInForce`: GTC, IOC, FOK

#### `Order` (dataclass)

Fields: `order_id`, `symbol`, `side`, `type`, `qty`, `price`, `status`, `fills`, `created_at`

#### `Fill` (dataclass)

Fields: `fill_id`, `order_id`, `qty`, `price`, `fee`, `timestamp`

#### `OrderManager`

**Key methods:**

| Method | Input | Output |
|---|---|---|
| `submit(order)` | Order | order_id string |
| `on_fill(fill)` | Fill event | Updated Order |
| `cancel(order_id)` | order_id | bool: success |
| `get_open_orders()` | — | List[Order] |
| `get_position(symbol)` | symbol | net quantity |

---

### `paper_order_manager.py` — Paper Trading Order Manager

**Purpose:** Simulates order execution for paper trading without connecting to a real exchange. Uses mid-price fills with configurable slippage.

Same interface as `OrderManager`. Drop-in replacement for testing.

---

## 18. portfolio/

### `portfolio_env.py` — Multi-Asset Portfolio Environment

**Purpose:** Extends the single-asset environment to manage a portfolio of multiple crypto assets simultaneously.

**State space:** Concatenation of per-asset feature vectors + portfolio-level state (weights, correlation matrix, total equity).

**Action space:** Portfolio weights allocation (continuous, sums to 1.0).

**Reward:** Portfolio Sharpe increment with cross-asset correlation penalty.

---

### `portfolio_risk_manager.py` — Portfolio Risk Manager

**Purpose:** Apply risk constraints at the portfolio level, considering cross-asset correlations.

**Key constraints:**
- Maximum single-asset weight (e.g., 40%)
- Portfolio VaR (historical simulation)
- Maximum correlation between held positions
- Factor exposure limits

**Key method:** `optimise_weights(signals, constraints)` → optimal weight vector

---

## 19. research/

### `alpha_research.py` — Alpha Factor Research

**Purpose:** Systematic evaluation of potential alpha factors (signals that predict future returns).

**Key functions:**
- `factor_ic(factor, forward_returns, periods)` → Information Coefficient at each horizon
- `factor_turnover(factor)` → How quickly the factor changes (cost proxy)
- `factor_decay(factor, forward_returns)` → Decay curve of predictive power over time
- `icir(factor, forward_returns)` → IC information ratio (IC / std(IC))

---

## 20. reward/

### `antibias_rewards.py` — Risk-Adjusted Reward Functions

**Purpose:** Replace naive raw-return rewards with risk-adjusted alternatives that prevent the agent from learning pathological behaviours (excessive risk, ignoring costs, churn).

#### `BaseReward` (ABC)

Abstract interface. All rewards must implement `compute(pnl, position, prev_position, equity, cost_this_bar)`.

#### `SharpeIncrementReward`

**Formula:** reward_t = (r_t - μ_rolling) / σ_rolling

Normalises each step's return by local rolling volatility. Rewards consistent low-volatility gains over lucky spikes.

**Parameters:** `window=50`, `risk_free_rate=0.0`

#### `CostAwareReward`

**Formula:** reward_t = r_t - λ_cost · costs_t - λ_dd · max(0, drawdown_t)

Explicitly penalises transaction costs (discourages excessive trading) and drawdowns (discourages risk-taking).

**Parameters:** `cost_penalty_lambda`, `drawdown_penalty_lambda`

#### `RegimeAwareReward`

Scales the reward by the confidence of the current HMM regime. In uncertain/transition regimes, the reward signal is attenuated to discourage overconfident positions.

**Parameters:** `hmm_detector` (HMMRegimeDetector instance)

---

## 21. risk/

### `risk_manager.py` — The Guardian

**Purpose:** Prevent capital destruction. Acts as the final gatekeeper before every trade reaches the exchange.

#### `RiskConfig` (dataclass)

| Parameter | Default | Description |
|---|---|---|
| `max_drawdown_per_session` | 0.02 | 2% max drawdown — halt trading |
| `max_consecutive_losses` | 5 | Halt after N consecutive losses |
| `max_position_size` | 0.25 | Max 25% of capital per position |
| `kelly_fraction` | 0.5 | Half Kelly for safety |
| `var_confidence` | 0.95 | VaR confidence level |
| `var_lookback` | 100 | Rolling VaR window |
| `min_capital_threshold` | 0.30 | Halt if equity < 30% of initial |
| `enable_circuit_breaker` | True | Enable halt mechanisms |

#### `RiskState` (dataclass)

Live state: `current_equity`, `initial_equity`, `peak_equity`, `current_drawdown`, `consecutive_losses`, `var_95`, `halt_trading`, `halt_reason`

#### `RiskManager`

**Key methods:**

| Method | Input | Output |
|---|---|---|
| `validate_position_size(proposed_size, capital)` | Position size in USD | (approved: bool, adjusted_size: float) |
| `update_state(equity, trade_result)` | New equity, PnL of last trade | Updated RiskState |
| `check_circuit_breaker()` | — | (halt: bool, reason: str) |
| `calculate_var(returns)` | Return series | VaR at configured confidence |

---

### `risk_metrics_logger.py` — Risk Metrics Logging

**Purpose:** Persist risk state snapshots at configurable intervals for post-hoc analysis and regulatory compliance.

**Key class:** `RiskMetricsLogger`

**Logged fields per interval:** timestamp, equity, drawdown, VaR, position, consecutive_losses, halt_status

**Output formats:** JSON Lines (`.jsonl`), CSV, database insert.

---

## 22. surveillance/

### `manipulation_detection.py` — Market Manipulation Detection

**Purpose:** Detect exchange-level manipulation patterns in real time to protect capital and adapt execution.

#### `ManipulationAlert` (dataclass)

Fields: `timestamp`, `type`, `severity` (low/medium/high/critical), `description`, `evidence`, `affected_orders`, `recommended_action`

#### `OrderBookState` (dataclass)

Snapshot: `timestamp`, `bids`, `asks`, `mid_price`, `spread`, `bid_depth`, `ask_depth`, `imbalance`

#### `ManipulationDetector`

**Detected patterns:**

| Pattern | Detection method |
|---|---|
| Spoofing | Large orders appear then cancel before fill |
| Layering | Multiple fake levels on one side of book |
| Wash trading | Same entity on both sides; volume without price movement |
| Quote stuffing | Order-to-trade ratio anomaly |
| Pump and dump | Abnormal volume + price velocity pattern |
| Flash crash | Instantaneous price drop followed by recovery |

**Key methods:**
- `analyze_orderbook(state)` → List[ManipulationAlert]
- `get_toxicity_score()` → float [0, 1]
- `recommend_action(alerts)` → "continue" / "reduce_size" / "pause" / "exit"

---

### `fraud_prevention.py` — Fraud Prevention

**Purpose:** Detect and prevent fraudulent account activity and API key compromise.

**Checks performed:** Unusual login times, abnormal trading patterns, unexpected position changes, rapid consecutive order submissions indicating automated attack.

---

## 23. training/

### `adversarial_trainer.py` — Adversarial Self-Play Training

**Purpose:** Train a robust Trader agent by simultaneously training an Adversary that creates maximally challenging market conditions.

#### `AdversarialConfig` (dataclass)

| Parameter | Default | Description |
|---|---|---|
| `n_iterations` | 500 | Total training iterations |
| `steps_per_iteration` | 2048 | Environment steps per iteration |
| `adversary_start_iteration` | 100 | Warm-up: adversary inactive until then |
| `adversary_strength` | 0.1 | How much adversary can modify market |
| `save_frequency` | 50 | Save checkpoint every N iterations |
| `checkpoint_dir` | "data/models/adversarial" | Where to save models |

#### `AdversarialTrainer`

**Training loop:**
1. Trader collects `steps_per_iteration` steps in the environment
2. If `iteration >= adversary_start_iteration`: Adversary observes Trader's trajectory and updates its policy to maximise Trader's losses
3. Trader performs PPO update on collected rollout
4. Log metrics, save checkpoint if due

**Key methods:**

| Method | Description |
|---|---|
| `train()` | Run full training loop |
| `evaluate(n_episodes)` | Evaluate Trader without Adversary, return metrics |
| `save_checkpoint(iteration)` | Save both agents |
| `load_checkpoint(path)` | Restore both agents |

---

## 24. transformer/

### `trading_transformer.py` — Attention-Based Trading Model

**Purpose:** Capture long-range temporal dependencies in financial time series using a Transformer encoder architecture.

#### `TransformerConfig` (dataclass)

| Parameter | Default | Description |
|---|---|---|
| `input_dim` | 64 | Features per timestep |
| `d_model` | 256 | Internal representation dimension |
| `nhead` | 8 | Multi-head attention heads |
| `num_layers` | 6 | Stacked encoder layers |
| `dim_feedforward` | 1024 | Feed-forward layer size |
| `max_seq_len` | 500 | Maximum input sequence length |
| `output_dim` | 3 | Buy / Hold / Sell |
| `use_adaptive_attention` | True | Adaptive attention span |

#### `PositionalEncoding`

Learnable temporal embeddings (preferred over fixed sinusoidal for trading, as market time is non-uniform).

#### `TradingTransformer` (nn.Module)

**Architecture:**
```
Input(seq_len, input_dim)
→ Linear projection to d_model
→ PositionalEncoding
→ TransformerEncoder (num_layers × multi-head attention + FFN)
→ [CLS] token pooling
→ Linear(d_model, output_dim)
→ Softmax → [P(Buy), P(Hold), P(Sell)]
```

**Key innovation:** Causal masking prevents the model from attending to future timesteps. Adaptive attention spans learn which history length is most relevant for each head.

**Key methods:** `forward(x)`, `get_attention_weights(x)` (for interpretability)

---

## 25. utils/

### `memory_utils.py` — Memory Management

**Purpose:** GPU and CPU memory management utilities for long training runs.

**Key functions:**
- `clear_gpu_cache()` — torch.cuda.empty_cache() + gc.collect()
- `get_memory_usage()` → dict with CPU/GPU memory stats
- `memory_guard(threshold_gb)` — decorator that clears cache if usage exceeds threshold
- `profile_memory(func)` — decorator that logs memory before/after a function call

---

## 26. validation/

### `antibias_walkforward.py` — Purged Walk-Forward Cross-Validation

**Purpose:** Walk-forward validation with a configurable "purge gap" that removes samples whose labels overlap with the test period, preventing the most subtle form of lookahead bias in financial ML.

#### `WalkForwardConfig` (dataclass, Antibias variant)

| Parameter | Description |
|---|---|
| `train_pct` | Fraction of data for training |
| `test_pct` | Fraction for test |
| `gap_pct` | Fraction to purge between train and test |
| `n_splits` | Number of walk-forward splits |

#### `PurgedWalkForwardCV`

**Key methods:**

| Method | Input | Output |
|---|---|---|
| `split(X, y)` | Feature array, labels | Iterator of (train_idx, test_idx) |
| `get_n_splits()` | — | Total number of splits |

**Why purging matters:** In financial ML, features at time t often incorporate information from bars t+1...t+k (e.g., a return label uses the next-bar close price). If the test set starts immediately after the training set, the last training samples and first test samples share overlapping information. The gap period removes this overlap.

#### `PurgedScaler`

A StandardScaler that only fits on the purged training set, ensuring test-set statistics never influence the normalisation.
