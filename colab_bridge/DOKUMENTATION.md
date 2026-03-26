# colab_bridge — Complete Documentation

**Version:** 1.0.0  
**Project:** BITCOIN4Traders (Reinforcement Learning Trading Bot)  
**Purpose:** Communication bridge between local machine (paper trading executor) and Google Colab / Cloud (RL model inference)

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Architecture Diagram](#2-architecture-diagram)
3. [Directory Structure](#3-directory-structure)
4. [Channel Schema](#4-channel-schema)
5. [Quick Start (5 minutes)](#5-quick-start-5-minutes)
6. [Module Reference](#6-module-reference)
   - 6.1 [transport_base.py — Abstract Interface](#61-transport_basepy--abstract-interface)
   - 6.2 [module_a_local.py — Local Engine](#62-module_a_localpy--local-engine)
   - 6.3 [module_b_colab.py — Colab RL Engine](#63-module_b_colabpy--colab-rl-engine)
   - 6.4 [control_plane.py — Control Layer](#64-control_planepy--control-layer)
   - 6.5 [colab_extension.py — Colab Extension](#65-colab_extensionpy--colab-extension)
   - 6.6 [transports/ — All 4 Transport Options](#66-transports--all-4-transport-options)
7. [Colab Extension — Detailed Reference](#7-colab-extension--detailed-reference)
   - 7.1 [classify_error()](#71-classify_error)
   - 7.2 [InProcessRepair](#72-inprocessrepair)
   - 7.3 [Reporter](#73-reporter)
   - 7.4 [IterationController](#74-iterationcontroller)
   - 7.5 [MemoryMonitor](#75-memorymonitor)
   - 7.6 [ColabKeepalive](#76-colabkeepalive)
   - 7.7 [ExceptionHook](#77-exceptionhook)
   - 7.8 [BT4TExtension / bt4t (public API)](#78-bt4textension--bt4t-public-api)
8. [Command Reference](#8-command-reference)
9. [Error Handling Table](#9-error-handling-table)
10. [Environment Variables](#10-environment-variables)
11. [Dependencies](#11-dependencies)
12. [Copy-Paste Colab Cells](#12-copy-paste-colab-cells)
13. [Troubleshooting](#13-troubleshooting)

---

## 1. System Overview

The `colab_bridge` system solves a fundamental problem in RL trading: the trained model runs in the cloud (Google Colab / GPU server), but the actual paper trader should work locally with real market data and execute trades.

**The system consists of three layers:**

| Layer         | Component                          | Runs on              |
| ------------- | ---------------------------------- | -------------------- |
| **Execution** | Module A (`module_a_local.py`)     | Local machine (24/7) |
| **Inference** | Module B (`module_b_colab.py`)     | Google Colab / Cloud |
| **Control**   | Control Plane (`control_plane.py`) | Local + Colab        |
| **Extension** | `colab_extension.py`               | Colab only           |

**Data flow (simplified):**

```
Local Machine                      Google Colab
─────────────────                  ────────────
Binance OHLCV
    ↓
Feature Computation
    ↓
Module A  ──── bt4t:market ────→  Module B
               (Ably/Redis/…)          ↓
                                    RL Inference
                                       ↓
Module A  ←─── bt4t:signals ────  Signal (BUY/SELL/HOLD)
    ↓
Execute Paper Order
    ↓
Portfolio State ─── bt4t:portfolio → Module B (context)
```

---

## 2. Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│  LOCAL MACHINE                                                       │
│                                                                      │
│  ┌──────────────────┐      ┌─────────────────────────────────────┐  │
│  │    Module A       │      │        Control Server               │  │
│  │ (module_a_local)  │      │       (FastAPI :8765)               │  │
│  │                  │      │                                     │  │
│  │ - Binance OHLCV  │      │  GET  /colab/command  ← Colab polls │  │
│  │ - RSI/BB/MACD    │      │  POST /colab/command  → Commands    │  │
│  │ - LocalPortfolio │      │  POST /colab/status   ← Status      │  │
│  │ - Signal Check   │      │  GET  /positions      → Portfolio   │  │
│  │ - Paper Orders   │      │  POST /trading/pause               │  │
│  └────────┬─────────┘      └──────────────┬────────────────────┘  │
│           │                               │                         │
│           │  Ably / Redis / Telegram /    │  Cloudflare Tunnel      │
│           │  Google Drive                 │  (HTTPS public URL)     │
└───────────┼───────────────────────────────┼─────────────────────────┘
            │                               │
            │         INTERNET              │
            │                               │
┌───────────┼───────────────────────────────┼─────────────────────────┐
│  GOOGLE COLAB                             │                          │
│           │                               │ HTTP Poll (every 5s)    │
│  ┌────────┴──────────┐      ┌─────────────┴──────────────────────┐  │
│  │    Module B        │      │       Control Client               │  │
│  │ (module_b_colab)   │      │   (Colab side of ControlPlane)    │  │
│  │                   │      │                                    │  │
│  │ - ModelAdapter    │      │  - polls commands every 5s        │  │
│  │   Darwin/.pth/SB3 │      │  - executes PAUSE/RESUME/etc.     │  │
│  │ - Obs-Buffer(60)  │      │  - sends status back              │  │
│  │ - RL Inference    │      │                                    │  │
│  │ - Heartbeat (10s) │      └────────────────────────────────────┘  │
│  └───────────────────┘                                              │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │          colab_extension (bt4t Singleton)                     │   │
│  │                                                              │   │
│  │  Reporter        → HTTP POST → Drive Fallback                │   │
│  │  InProcessRepair → batch_size / LR / gradient_clip           │   │
│  │  IterationCtrl   → PAUSE/RESUME/CHANGE_LR/CHANGE_BS          │   │
│  │  MemoryMonitor   → GPU check every 60s                       │   │
│  │  ColabKeepalive  → numpy compute every 600s                  │   │
│  │  ExceptionHook   → sys.excepthook (global)                   │   │
│  └──────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 3. Directory Structure

```
colab_bridge/
│
├── __init__.py              # Package init, Version 1.0.0
├── transport_base.py        # Abstract interface (ABC) for all transports
├── module_a_local.py        # Local execution engine
├── module_b_colab.py        # Colab RL inference engine
├── control_plane.py         # FastAPI ControlServer + ControlClient
├── colab_extension.py       # Colab Extension (bt4t Singleton)
│
├── DOKUMENTATION.md         # This file
├── TRANSPORT_GUIDE.md       # Transport comparison and setup
│
└── transports/
    ├── __init__.py           # Factory get_transport()
    ├── transport_ably.py     # Option 1: Ably Pub/Sub (50–150ms)
    ├── transport_redis.py    # Option 2: Redis + Cloudflare (30–150ms)
    ├── transport_telegram.py # Option 3: Telegram Bot (200–800ms)
    └── transport_gdrive.py   # Option 4: Google Drive (2–15s)
```

---

## 4. Channel Schema

All transports use identical channel names (defined in `transport_base.py`):

| Constant       | Channel name           | Direction     | Content                                   |
| -------------- | ---------------------- | ------------- | ----------------------------------------- |
| `CH_MARKET`    | `bt4t:market:{symbol}` | Local → Colab | OHLCV + Features (RSI/BB/MACD)            |
| `CH_SIGNALS`   | `bt4t:signals`         | Colab → Local | Trade signal (BUY/SELL/HOLD + Confidence) |
| `CH_PORTFOLIO` | `bt4t:portfolio:state` | Local → Colab | Portfolio state (Equity, Position, P&L)   |
| `CH_HEALTH`    | `bt4t:health`          | Colab → Local | Heartbeat (every 10s)                     |
| `CH_CONTROL`   | `bt4t:control:cmd`     | Local → Colab | Control commands (PAUSE/RESUME/RELOAD/…)  |
| `CH_ACK`       | `bt4t:control:ack`     | Colab → Local | Acknowledgement for commands              |

### Signal Payload (`bt4t:signals`)

```json
{
  "action": "BUY",
  "symbol": "BTCUSDT",
  "confidence": 0.7823,
  "model_version": "MV_Gen4_BB_breakout_p11_std1.91_9",
  "timestamp_utc": "2024-01-15T10:30:00+00:00",
  "inference_latency_ms": 12.3,
  "signal_n": 42,
  "portfolio_equity": 10500.0,
  "portfolio_return_pct": 5.0
}
```

### Market Payload (`bt4t:market:BTCUSDT`)

```json
{
  "timestamp": "2024-01-15T10:30:00+00:00",
  "symbol": "BTCUSDT",
  "open": 42000.0, "high": 42500.0, "low": 41800.0,
  "close": 42300.0, "volume": 1234.5,
  "rsi14": 58.3,
  "bb_pct": 0.62,
  "bb_upper": 43000.0, "bb_lower": 41000.0,
  "macd": 120.5,
  "vol_ratio": 1.3,
  "return_1h": 0.0071, "return_4h": 0.0152, "return_24h": 0.0234,
  "sma20": 42100.0,
  "close_60": [41500.0, 41600.0, ..., 42300.0]
}
```

---

## 5. Quick Start (5 minutes)

### Step 1: Create Ably account (free)

```
1. https://ably.com → "Sign up for free"
2. Create App → "BITCOIN4Traders"
3. API Keys → Copy Root Key
4. Add to .env: ABLY_API_KEY=your_root_key_here
```

### Step 2: Start local stack

```bash
# Terminal 1: Cloudflare Tunnel (optional, for Control Plane)
./cloudflared tunnel --url http://localhost:8765

# Terminal 2: Full local stack
python colab_bridge/control_plane.py full \
    --ably-key $ABLY_API_KEY \
    --capital 10000

# OR: Module A only (without Control Server)
python colab_bridge/module_a_local.py \
    --ably-key $ABLY_API_KEY \
    --symbol BTC/USDT \
    --interval 30
```

### Step 3: Set up Google Colab

First cell in Colab notebook (copy-paste, one-time):

```python
# ─── Cell 1: Setup ───────────────────────────────────────────
import sys, os
from google.colab import drive
drive.mount('/content/drive')

sys.path.insert(0, '/content/drive/MyDrive/BITCOIN4Traders')

# Configuration (copy URL from Terminal 1)
os.environ['BT4T_LISTENER_URL'] = 'https://abc-def.trycloudflare.com'
os.environ['BT4T_API_TOKEN']    = 'bt4t-secret-token'
os.environ['BT4T_NOTEBOOK_ID']  = 'training_v1'
os.environ['ABLY_API_KEY']      = 'your_ably_key'

# Load bt4t Extension
from colab_bridge.colab_extension import bt4t
bt4t.install()

# Start Module B
from colab_bridge.module_b_colab import ModuleB
import asyncio

engine = ModuleB(
    ably_key=os.environ['ABLY_API_KEY'],
    model_path='/content/drive/MyDrive/BITCOIN4Traders/data/cache/multiverse_champion.pkl',
)
await engine.run()
```

---

## 6. Module Reference

### 6.1 `transport_base.py` — Abstract Interface

Defines the shared interface for all 4 transports and the channel constants.

#### Class: `TransportBase` (ABC)

Each transport must implement these 5 methods:

| Method          | Signature                                                   | Description                                          |
| --------------- | ----------------------------------------------------------- | ---------------------------------------------------- |
| `connect`       | `async connect() -> None`                                   | Establish connection. Raises exception on error.     |
| `disconnect`    | `async disconnect() -> None`                                | Cleanly disconnect.                                  |
| `publish`       | `async publish(channel: str, payload: dict) -> None`        | Send message on channel.                             |
| `subscribe`     | `async subscribe(channel: str, callback: Callable) -> None` | Register callback for incoming messages.             |
| `name`          | `@property -> str`                                          | Name of transport for logging.                       |
| `latency_class` | `@property -> str`                                          | Latency class: `'ms'` / `'sub-second'` / `'seconds'` |

#### Helper methods (static):

```python
TransportBase.encode(payload: dict) -> str     # dict → JSON string
TransportBase.decode(data: str|bytes|dict) -> dict  # JSON → dict
```

---

### 6.2 `module_a_local.py` — Local Engine

Runs on the local machine (24/7). Connects exchange with Colab via Ably.

#### Class: `LocalPaperPortfolio`

In-memory paper portfolio. No external database needed.

```python
portfolio = LocalPaperPortfolio(
    initial_capital=10_000.0,  # Starting capital in USDT
    fee_rate=0.001             # 0.1% trading fee
)
```

| Method                             | Description                                           |
| ---------------------------------- | ----------------------------------------------------- |
| `execute(side, price, confidence)` | Simulates market order. Returns trade dict or `None`. |
| `current_equity(price)`            | Calculates current total value (cash + position).     |
| `state_dict(price)`                | Returns complete portfolio state as dict.             |
| `pause()` / `resume()`             | Stops/starts trading (on heartbeat timeout).          |

**Trade Dict (returned by `execute()`):**

```python
{
    "timestamp": "2024-01-15T10:30:00+00:00",
    "action": "BUY",    # BUY / SELL / SHORT / COVER
    "side": "buy",
    "qty": 0.0023,      # BTC quantity
    "price": 42315.0,   # Fill price (incl. 0.03% slippage)
    "fee": 0.09,
    "pnl": 0.0,         # Realized P&L (0 on opening)
    "confidence": 0.78  # Signal confidence
}
```

#### Function: `compute_features(df: pd.DataFrame) -> dict`

Computes trading features from OHLCV DataFrame (200 bars recommended).

**Computed Features:**

| Feature                 | Description                       | Formula                             |
| ----------------------- | --------------------------------- | ----------------------------------- |
| `rsi14`                 | RSI with 14 periods               | Wilder's RSI                        |
| `bb_pct`                | Position in Bollinger Band (0–1)  | `(close - lower) / (upper - lower)` |
| `bb_upper` / `bb_lower` | Bollinger Bands (20, 2σ)          | SMA20 ± 2×STD20                     |
| `macd`                  | MACD line                         | EMA12 − EMA26                       |
| `vol_ratio`             | Volume relative to 20-bar average | `vol / mean(vol[-20:])`             |
| `return_1h`             | 1-bar return                      | `close[-1]/close[-2] - 1`           |
| `return_4h`             | 4-bar return                      | `close[-1]/close[-5] - 1`           |
| `return_24h`            | 24-bar return                     | `close[-1]/close[-25] - 1`          |
| `sma20`                 | Simple moving average (20)        | `mean(close[-20:])`                 |
| `close_60`              | Last 60 close values              | Array for RL observation            |

#### Class: `ModuleA`

```python
engine = ModuleA(
    ably_key="your_key",          # ABLY_API_KEY
    symbol="BTC/USDT",            # CCXT format
    timeframe="1h",               # OHLCV timeframe
    exchange_id="binance",        # CCXT exchange ID
    poll_interval_s=30.0,         # Seconds between OHLCV requests
    initial_capital=10_000.0,     # Starting capital USDT
)
await engine.run()                # Blocks until Ctrl+C
```

**Internal Loop (every `poll_interval_s`):**

1. Fetch OHLCV via CCXT (200 bars)
2. Call `compute_features()`
3. Publish features to `bt4t:market:BTCUSDT`
4. Every 5 ticks: publish portfolio state to `bt4t:portfolio:state`
5. Check heartbeat age (timeout: 90s)
6. Every 10 ticks: print status log

**Signal validation (automatic in `_on_signal()`):**

- Staleness check: signal older than 10s → discard
- Confidence check: below 0.55 → discard

**Send commands to Colab:**

```python
await engine.send_command("PAUSE_INFERENCE")
await engine.send_command("RELOAD_MODEL", {"model_path": "/path/to/model.pkl"})
await engine.send_command("SHUTDOWN")
```

**CLI:**

```bash
python colab_bridge/module_a_local.py [--symbol BTC/USDT] [--timeframe 1h]
    [--exchange binance] [--interval 30] [--capital 10000] [--ably-key KEY]
```

---

### 6.3 `module_b_colab.py` — Colab RL Engine

Runs in Google Colab. Receives market data, runs RL inference.

#### Class: `ModelAdapter`

Abstraction layer supporting 4 model types:

```python
adapter = ModelAdapter(
    model_path="/path/to/model.pkl",  # or .pth, .zip, None
    model_type="auto"                  # auto / darwin / sb3 / pytorch / rsi_fallback
)
action, confidence = adapter.predict(close_array, features_dict)
```

**Load priority (auto):**

| Order | File extension          | Model type     | Description                              |
| ----- | ----------------------- | -------------- | ---------------------------------------- |
| 1     | `.pkl`                  | `darwin`       | Darwin Champion from Multiverse training |
| 2     | `.pth`                  | `pytorch`      | PyTorch checkpoint (falls back to RSI)   |
| 3     | `.zip` or `sb3` in path | `sb3`          | Stable-Baselines3 PPO                    |
| 4     | no path / load failed   | `rsi_fallback` | RSI signal (no model needed)             |

**Darwin inference:**

```python
# internally: model.compute_signals(close_array)
# Confidence = Clip(abs(sigs[-10:]).mean() + 0.5, 0, 1)
# Signal = sign(sigs[-1]) → 1=BUY, -1=SELL, 0=HOLD
```

**RSI fallback inference:**

```python
# RSI < 30 → BUY  with confidence = 0.5 + (30-rsi)/60
# RSI > 70 → SELL with confidence = 0.5 + (rsi-70)/60
# else     → HOLD
```

**Reload model:**

```python
adapter.reload("/path/to/new_model.pkl")  # or None = same path
```

#### Class: `ModuleB`

```python
engine = ModuleB(
    ably_key="your_key",
    model_path="/content/drive/MyDrive/.../multiverse_champion.pkl",
    symbol="BTCUSDT",          # Without slash (Ably channel format)
    min_confidence=0.55,       # Signals below this threshold → HOLD
)
await engine.run()             # Blocks until SHUTDOWN or Ctrl+C
```

**Observation buffer:**

- Rolling window of last 70 market data updates (`deque(maxlen=70)`)
- Inference starts when buffer ≥ 20 entries
- RL observation = last 60 bars + 6 scalar features

**Heartbeat (automatic every 10s):**

```json
{
  "model_loaded": true,
  "model_version": "MV_Gen4_BB...",
  "inference_count": 142,
  "signal_count": 23,
  "obs_buffer_size": 70,
  "paused": false,
  "last_market_data_age_s": 8.2,
  "status": "COLAB_READY"
}
```

**Received commands (from `bt4t:control:cmd`):**

| Command           | Effect                              |
| ----------------- | ----------------------------------- |
| `PAUSE_INFERENCE` | Stops RL inference (no signals)     |
| `RESUME`          | Resumes inference                   |
| `RELOAD_MODEL`    | Reloads model (`params.model_path`) |
| `SHUTDOWN`        | Terminates Module B                 |
| `STATUS`          | Sends status ACK                    |

**Helper functions:**

```python
from colab_bridge.module_b_colab import colab_setup, find_champion_on_drive

project_path = colab_setup(drive_mount=True)   # Mounts Drive, returns path
champion = find_champion_on_drive(project_path) # Searches for .pkl file
```

**CLI (for vast.ai / local GPU):**

```bash
python colab_bridge/module_b_colab.py \
    --ably-key $ABLY_API_KEY \
    --model data/cache/multiverse_champion.pkl \
    --symbol BTCUSDT \
    --min-conf 0.55
```

---

### 6.4 `control_plane.py` — Control Layer

#### Class: `ControlServer`

FastAPI server that runs locally and is accessible via Cloudflare Tunnel.

```python
server = ControlServer(
    port=8765,
    token="bt4t-secret-token",  # CONTROL_API_TOKEN
    module_a=engine,             # Optional: ModuleA reference for portfolio state
)
await server.start()
```

**API endpoints:**

| Method | Endpoint          | Auth   | Description                               |
| ------ | ----------------- | ------ | ----------------------------------------- |
| `GET`  | `/health`         | no     | Health check                              |
| `GET`  | `/status`         | Bearer | Uptime, queue length, Colab status        |
| `GET`  | `/positions`      | Bearer | Current portfolio state                   |
| `GET`  | `/colab/command`  | Bearer | Next command for Colab (Colab polls here) |
| `POST` | `/colab/command`  | Bearer | Send command to Colab                     |
| `POST` | `/colab/status`   | Bearer | Colab reports status                      |
| `POST` | `/trading/pause`  | Bearer | Pause portfolio + inference               |
| `POST` | `/trading/resume` | Bearer | Resume portfolio + inference              |

**Send command to Colab (HTTP):**

```bash
curl -X POST http://localhost:8765/colab/command \
  -H "Authorization: Bearer bt4t-secret-token" \
  -H "Content-Type: application/json" \
  -d '{"cmd": "PAUSE_INFERENCE", "params": {}}'
```

**Valid commands for `POST /colab/command`:**
`PAUSE_INFERENCE` | `RESUME` | `RELOAD_MODEL` | `SHUTDOWN` | `STATUS`

**Command expiry:** Commands waiting in the queue for more than 60s are automatically discarded.

#### Class: `ControlClient`

Runs in Colab alongside Module B. Polls ControlServer every 5s.

```python
client = ControlClient(
    server_url="https://abc-def.trycloudflare.com",
    module_b=engine,
    token="bt4t-secret-token",
    poll_interval_s=5.0,
)
asyncio.create_task(client.run())  # Start as async task
```

#### Function: `start_cloudflare_tunnel(port) -> Optional[str]`

Starts `cloudflared` process and returns the public HTTPS URL.

```python
url = await start_cloudflare_tunnel(port=8765)
# → "https://abc-def.trycloudflare.com"
```

Prerequisite: `cloudflared` must be installed.

#### Function: `start_full_local_stack()`

Starts Module A + ControlServer + Cloudflare Tunnel in one call:

```python
await start_full_local_stack(
    ably_key="your_key",
    capital=10_000.0,
    symbol="BTC/USDT",
    timeframe="1h",
    exchange_id="binance",
    poll_interval_s=30.0,
    api_token="bt4t-secret-token",
    start_tunnel=True,
)
```

**CLI:**

```bash
python colab_bridge/control_plane.py server   # Control Server only
python colab_bridge/control_plane.py tunnel   # Cloudflare Tunnel only
python colab_bridge/control_plane.py full \   # Everything together
    --ably-key $ABLY_API_KEY \
    --capital 10000 --symbol BTC/USDT
```

---

### 6.5 `colab_extension.py` — Colab Extension

The extension hooks into any existing Colab notebook with **a single import**. No code restructuring needed.

**Singleton import:**

```python
from colab_bridge.colab_extension import bt4t
```

The `bt4t` object is a global singleton instance of the `BT4TExtension` class.

The detailed reference for all classes can be found in [Chapter 7](#7-colab-extension--detailed-reference).

---

### 6.6 `transports/` — All 4 Transport Options

#### Factory Function: `get_transport()`

```python
from colab_bridge.transports import get_transport

# Ably
t = get_transport("ably", api_key="root:xxx")

# Redis
t = get_transport("redis", side="local")    # on local machine
t = get_transport("redis", side="colab",    # in Colab
    redis_url="https://tunnel.dev/redis")

# Telegram
t = get_transport("telegram",
    bot_token="123:ABC",
    chat_id="-100123456")

# Google Drive
t = get_transport("gdrive", side="local")
t = get_transport("gdrive", side="colab")
```

#### Comparison of the 4 options:

| Option           | Latency   | Cost                | Account    | Best for            |
| ---------------- | --------- | ------------------- | ---------- | ------------------- |
| **Ably**         | 50–150ms  | Free Tier: 100 Conn | Yes (Ably) | Primary transport   |
| **Redis**        | 30–150ms  | Free                | No         | No external account |
| **Telegram**     | 200–800ms | Free                | Bot token  | Already in .env     |
| **Google Drive** | 2–15s     | Free                | Google     | Fallback / Debug    |

Detailed setup instructions: [TRANSPORT_GUIDE.md](TRANSPORT_GUIDE.md)

---

## 7. Colab Extension — Detailed Reference

### 7.1 `classify_error()`

```python
error_type, repair_action, severity = classify_error(exc)
```

Classifies an exception based on regex patterns.

**Recognized error types:**

| Error type    | Detection pattern                                   | Action             | Severity |
| ------------- | --------------------------------------------------- | ------------------ | -------- |
| `OOM`         | `CUDA out of memory`, `OutOfMemoryError`, `OOM`     | `halve_batch_size` | high     |
| `NAN_LOSS`    | `nan`, `NaN`, `inf.*loss/reward/gradient`           | `reduce_lr`        | high     |
| `EXPLODING`   | `gradient.*explod`, `loss.*explod`, `overflow`      | `clip_gradients`   | high     |
| `IMPORT`      | `ModuleNotFoundError`, `No module named`            | `pip_install`      | medium   |
| `TIMEOUT`     | `TimeoutError`, `ReadTimeout`, `socket.timeout`     | `increase_timeout` | low      |
| `CONNECTION`  | `ConnectionError`, `ConnectTimeout`                 | `retry_connection` | medium   |
| `CUDA_ERROR`  | `RuntimeError.*CUDA`, `device-side assert`          | `halve_batch_size` | high     |
| `DATA_ERROR`  | `KeyError`, `IndexError`, `ValueError.*batch/data`  | `skip_batch`       | medium   |
| `IO_ERROR`    | `PermissionError`, `FileNotFoundError.*drive/model` | `retry_io`         | medium   |
| `INTERRUPTED` | `KeyboardInterrupt`                                 | `none`             | low      |
| `UNKNOWN`     | (no pattern matches)                                | `none`             | medium   |

**Example:**

```python
try:
    train(model, data)
except Exception as exc:
    error_type, action, severity = classify_error(exc)
    # error_type = "OOM", action = "halve_batch_size", severity = "high"
```

---

### 7.2 `InProcessRepair`

Repairs hyperparameters directly in the running Python process without notebook restart.  
Searches for variables in the global namespace (`__main__`).

```python
repair = InProcessRepair(repair_log=[])
result = repair.apply("halve_batch_size", context={})
# result = {"action": "halve_batch_size", "changes": ["BATCH_SIZE: 32 → 16"], "success": True}
```

**Method `apply(action, context) -> dict`:**

| Action             | What gets changed                                                                    |
| ------------------ | ------------------------------------------------------------------------------------ |
| `halve_batch_size` | Halves all `batch_size` variables in global NS. Clears CUDA cache.                   |
| `reduce_lr`        | Reduces `learning_rate`/`lr` variables by factor 10. Also patches PyTorch optimizer. |
| `clip_gradients`   | Sets `gradient_clip_val` to min(current, 0.5). New: `GRADIENT_CLIP_VAL = 0.5`.       |
| `pip_install`      | Installs missing package from error message via `pip`.                               |
| `increase_timeout` | Doubles all `timeout` variables in global NS.                                        |
| `skip_batch`       | Logs warning (signal for notebook code).                                             |
| `retry_connection` | Waits 30s.                                                                           |
| `retry_io`         | Waits 10s.                                                                           |
| `none`             | No action.                                                                           |

**Return dict:**

```python
{
    "timestamp": "2024-01-15T10:30:00+00:00",
    "action": "halve_batch_size",
    "changes": ["BATCH_SIZE: 32 → 16", "optimizer[opt].lr: 1.00e-03 → 1.00e-04"],
    "success": True
}
```

**Important:** The repair only takes effect if the variables exist in the global namespace. Local variables inside functions are not captured.

---

### 7.3 `Reporter`

Sends reports to the local machine. Dual channel: HTTP POST → Google Drive fallback.

```python
reporter = Reporter(
    listener_url="https://tunnel.trycloudflare.com",
    api_token="bt4t-secret-token",
    notebook_id="training_v1"
)
reporter.start()   # Start background send thread
reporter.stop()    # Stop thread
```

**Public methods:**

| Method                                          | Description                                    |
| ----------------------------------------------- | ---------------------------------------------- |
| `report_error(exc, error_type, repair_applied)` | Sends error report with stack trace            |
| `report_progress(data: dict)`                   | Sends training progress (epoch, loss, reward)  |
| `report_heartbeat(extra={})`                    | Sends heartbeat with status `COLAB_ALIVE`      |
| `poll_commands() -> list[dict]`                 | Synchronously polls local machine for commands |

**Send mechanism:**

1. All reports are put in internal queue
2. Background thread (`_send_loop`) continuously drains queue
3. Each message: HTTP POST to `{listener_url}/report_error`
4. On HTTP error: JSON file in `/content/drive/MyDrive/BITCOIN4Traders/bt4t_bus/reports/`

**Poll commands (`poll_commands()`):**

```python
commands = reporter.poll_commands()
# → [{"cmd": "PAUSE", "params": {}}]   or []
```

Requests `GET {listener_url}/colab/command` (max. 5s timeout).

---

### 7.4 `IterationController`

Controls the training loop from outside. Called internally by `bt4t.step()`.

```python
controller = IterationController(reporter=reporter, repair=repair)
should_continue = controller.process(epoch=5, step=100, loss=0.42, reward=1.5)
```

**Method `process(...) -> bool`:**

Called at every training step. Returns `False` when training should stop.

Internal steps:

1. Update state (epoch, step, loss, reward)
2. Every 10 steps: send progress via Reporter
3. Every 5s: poll commands + execute
4. When paused: wait loop (checks every 5s for RESUME)
5. Return: `not stop_requested`

**`IterationState` fields:**

```python
state.epoch            # Current epoch
state.step             # Current step
state.loss             # Last loss value
state.reward           # Last reward value
state.paused           # True when PAUSE command received
state.stop_requested   # True when STOP command received
state.lr               # Remotely set learning rate
state.batch_size       # Remotely set batch size
state.checkpoint_every # Checkpoint every N steps (default: 50)
```

**Processed commands:**

| Command             | Effect on state                                                             |
| ------------------- | --------------------------------------------------------------------------- |
| `PAUSE`             | `paused = True` → wait loop                                                 |
| `RESUME`            | `paused = False` → exit wait loop                                           |
| `STOP` / `SHUTDOWN` | `stop_requested = True` → loop returns False                                |
| `CHANGE_LR`         | Sets LR in global NS + PyTorch optimizer                                    |
| `CHANGE_BS`         | Sets `batch_size` in global NS                                              |
| `RELOAD_MODEL`      | Sets `BT4T_RELOAD_REQUESTED = True` + `BT4T_RELOAD_MODEL_PATH` in global NS |
| `STATUS`            | Sends current state as progress report                                      |

---

### 7.5 `MemoryMonitor`

Monitors GPU memory every 60s and performs prophylactic cleanup.

```python
monitor = MemoryMonitor(
    repair=repair,
    warn_pct=85.0    # Warning threshold in % (default: 85%)
)
monitor.start()   # Start background thread
monitor.stop()    # Stop thread
```

**Automatic behavior:**

- `GPU > 85%`: `gc.collect()` + `torch.cuda.empty_cache()` → log output
- `GPU > 95%` after cleanup: `repair.apply("halve_batch_size", {})` → halve batch size

Only works if `torch` is installed and GPU is available. Otherwise: no-op.

---

### 7.6 `ColabKeepalive`

Prevents Colab session timeout through real compute tasks.

```python
keepalive = ColabKeepalive(
    reporter=reporter,
    interval_s=600.0    # Every 10 minutes (default)
)
keepalive.start()
keepalive.stop()
```

**Mechanism:**

- Every `interval_s` seconds: `numpy.random.randn(1000, 1000).mean()` (real compute task)
- No sleep trick — Colab detects active kernel through GPU-CPU activity
- Sends heartbeat with `{"keepalive_tick": "HH:MM:SS"}`

**Note:** Colab kills sessions due to **inactivity** (no output/compute), not after a fixed time. The numpy computation simulates real activity.

---

### 7.7 `ExceptionHook`

Global `sys.excepthook` — catches all unhandled exceptions.

```python
hook = ExceptionHook(reporter=reporter, repair=repair)
hook.install()    # Override sys.excepthook
hook.uninstall()  # Restore original hook
```

**Flow on exception:**

1. `KeyboardInterrupt` → pass to original hook (no handling)
2. Call `classify_error(exc)` → error type + action + severity
3. Increment repair counter for this error type
4. If `action != "none"` and `count <= 5`: call `repair.apply(action, ctx)`
5. If `count > 5`: warning "no further repair" (prevents infinite loop)
6. Call `reporter.report_error(exc, error_type, repair_result)`
7. Call original hook (print traceback in notebook)

---

### 7.8 `BT4TExtension` / `bt4t` (public API)

The singleton `bt4t` is the only public interface of the extension.

```python
from colab_bridge.colab_extension import bt4t
```

#### `bt4t.install()` — Set up extension

```python
bt4t.install(
    listener_url="https://abc-def.trycloudflare.com",  # Optional (also via ENV)
    api_token="bt4t-secret-token",    # Optional
    notebook_id="training_v1",        # Optional
    keepalive=True,                   # Enable Colab keepalive
    memory_monitor=True,              # Enable GPU monitoring
    exception_hook=True,              # Install global exception hook
    keepalive_interval_s=600.0,       # Keepalive interval
    memory_warn_pct=85.0,             # GPU warning threshold %
)
```

- Returns `self` (method chaining possible)
- Idempotent: second call logs warning and does nothing
- Sends `COLAB_READY` heartbeat after successful installation
- Reads configuration from environment variables (override possible via parameters)

**Output after `install()`:**

```
═══════════════════════════════════════════════════════
  bt4t Extension installed
  Notebook  : training_v1
  Listener  : https://abc-def.trycloudflare.com
  Keepalive : on
  Memory    : on
  ExcHook   : on
═══════════════════════════════════════════════════════
```

#### `bt4t.step()` — Report training step

```python
should_continue = bt4t.step(
    epoch=5,
    step=100,
    loss=0.42,
    reward=1.5,
    # any additional kwargs are sent as "extra"
    accuracy=0.94,
)
# False when STOP command received

# Typical usage:
for epoch in range(100):
    loss = train_one_epoch(model, data)
    if not bt4t.step(epoch=epoch, loss=float(loss)):
        break
```

If extension not installed (`bt4t.install()` not called): always returns `True`.

#### `bt4t.guard` — Decorator

```python
@bt4t.guard
def run_training():
    for epoch in range(100):
        loss = train(model, data)
        bt4t.step(epoch=epoch, loss=loss)
```

- Max. 3 retries on error
- Classifies error + performs repair
- Sends error report
- On `high` severity or last attempt: re-raise
- `KeyboardInterrupt` is always passed through immediately

#### `bt4t.session()` — Context manager

```python
with bt4t.session("experiment_42"):
    train_model(model, data)
```

- Sends `SESSION_START` heartbeat on entry
- Sends `SESSION_END` with status `OK` / `ERROR` / `INTERRUPTED` on exit
- On exception: classify error + repair + report, then re-raise

#### `bt4t.send_checkpoint()` — Report checkpoint

```python
bt4t.send_checkpoint(
    model_path="/content/drive/.../model_ep50.pth",
    metrics={"loss": 0.42, "reward": 1.5, "sharpe": 1.2}
)
```

Sends `CHECKPOINT` event with current epoch/step and optional metrics.

#### `bt4t.send_alert()` — Manual message

```python
bt4t.send_alert("Epoch 50 completed", level="INFO")
bt4t.send_alert("WARNING: Loss diverging!", level="WARNING")
```

#### `bt4t.status()` — Query current status

```python
status = bt4t.status()
# {
#   "installed": True,
#   "notebook_id": "training_v1",
#   "listener_url": "https://...",
#   "epoch": 42,
#   "step": 1234,
#   "paused": False,
#   "stop_requested": False,
#   "repairs_done": 2
# }
```

#### `bt4t.repair_log()` — Repair log

```python
repairs = bt4t.repair_log()
# [
#   {"timestamp": "...", "action": "halve_batch_size",
#    "changes": ["BATCH_SIZE: 32 → 16"], "success": True},
#   ...
# ]
```

#### `bt4t.should_stop` / `bt4t.is_paused` — Properties

```python
if bt4t.should_stop:
    print("Training should be stopped")
if bt4t.is_paused:
    print("Training is paused")
```

#### `bt4t.uninstall()` — Remove extension

```python
bt4t.uninstall()  # Remove all hooks, stop all threads
```

---

## 8. Command Reference

### Commands Local → Colab (via `bt4t:control:cmd` or `POST /colab/command`)

| Command           | Parameters        | Processed by            | Description                                       |
| ----------------- | ----------------- | ----------------------- | ------------------------------------------------- |
| `PAUSE_INFERENCE` | –                 | ModuleB                 | Stops RL inference in Module B                    |
| `RESUME`          | –                 | ModuleB + IterationCtrl | Resumes inference + training                      |
| `RELOAD_MODEL`    | `model_path: str` | ModuleB + IterationCtrl | Loads new model                                   |
| `SHUTDOWN`        | –                 | ModuleB                 | Cleanly shuts down Module B                       |
| `STATUS`          | –                 | ModuleB + IterationCtrl | Requests status report                            |
| `PAUSE`           | –                 | IterationController     | Pauses training loop                              |
| `STOP`            | –                 | IterationController     | Stops training loop (returns `False` at `step()`) |
| `CHANGE_LR`       | `value: float`    | IterationController     | Changes learning rate live                        |
| `CHANGE_BS`       | `value: int`      | IterationController     | Changes batch size live                           |

### Commands Colab → Local (via `bt4t:control:ack`)

| Event | Content              | Description                                |
| ----- | -------------------- | ------------------------------------------ |
| ACK   | `{cmd, status, msg}` | Acknowledgement for every received command |

### Events Colab → Local (via Reporter HTTP POST)

| Type         | Fields                                                    | Description                                         |
| ------------ | --------------------------------------------------------- | --------------------------------------------------- |
| `heartbeat`  | `{status, notebook_id, timestamp_utc}`                    | Sign of life (also: `COLAB_READY`, `SESSION_START`) |
| `progress`   | `{epoch, step, loss, reward}`                             | Training progress (every 10 steps)                  |
| `error`      | `{error_type, error_message, stacktrace, repair_applied}` | Error report                                        |
| `CHECKPOINT` | `{model_path, metrics, epoch, step}`                      | Checkpoint notification                             |
| `ALERT`      | `{level, message}`                                        | Manual message                                      |

---

## 9. Error Handling Table

| Error occurs       | Recognized as | Automatic action   | Result                            |
| ------------------ | ------------- | ------------------ | --------------------------------- |
| GPU memory full    | `OOM`         | `halve_batch_size` | BATCH_SIZE /2, CUDA cache cleared |
| Loss is NaN        | `NAN_LOSS`    | `reduce_lr`        | LR /10, optimizer LR adjusted     |
| Gradient explosion | `EXPLODING`   | `clip_gradients`   | GRADIENT_CLIP_VAL = 0.5           |
| Missing library    | `IMPORT`      | `pip_install`      | Automatic pip install             |
| Network timeout    | `TIMEOUT`     | `increase_timeout` | Timeout variables ×2              |
| Connection error   | `CONNECTION`  | `retry_connection` | Wait 30s                          |
| CUDA runtime error | `CUDA_ERROR`  | `halve_batch_size` | Like OOM                          |
| Invalid data       | `DATA_ERROR`  | `skip_batch`       | Set flag                          |
| Drive unreachable  | `IO_ERROR`    | `retry_io`         | Wait 10s                          |
| Same error >5x     | Any           | No further repair  | Infinite loop prevented           |

**Three-level escalation:**

1. **ExceptionHook** catches unhandled exceptions (automatic)
2. **`bt4t.guard` decorator** tries up to 3 times (opt-in)
3. **`bt4t.session` context manager** catches + reports (opt-in)

---

## 10. Environment Variables

### For the Colab Extension (`colab_extension.py`)

| Variable            | Default               | Description                                               |
| ------------------- | --------------------- | --------------------------------------------------------- |
| `BT4T_LISTENER_URL` | `""`                  | HTTPS URL of the local Control Server (Cloudflare Tunnel) |
| `BT4T_API_TOKEN`    | `"bt4t-secret-token"` | Shared secret (must match ControlServer)                  |
| `BT4T_NOTEBOOK_ID`  | `"colab_notebook"`    | Notebook name for logs/reports                            |

### For the entire system

| Variable             | File      | Description                                        |
| -------------------- | --------- | -------------------------------------------------- |
| `ABLY_API_KEY`       | `.env`    | Ably Root API Key (free)                           |
| `CONTROL_API_TOKEN`  | `.env`    | Bearer token for ControlServer auth                |
| `CONTROL_PORT`       | `.env`    | ControlServer port (default: 8765)                 |
| `CONTROL_SERVER_URL` | Colab ENV | URL of the Cloudflare Tunnel (set in Colab)        |
| `TELEGRAM_BOT_TOKEN` | `.env`    | Bot token for Telegram transport (already in .env) |
| `TELEGRAM_CHAT_ID`   | `.env`    | Telegram chat ID (already in .env, commented out)  |

---

## 11. Dependencies

### Required (local)

```bash
pip install ably ccxt numpy pandas loguru
```

### Required (Colab, Module B)

```bash
pip install ably loguru numpy
```

### Optional (for respective features)

| Package             | Feature                                            |
| ------------------- | -------------------------------------------------- |
| `fastapi uvicorn`   | ControlServer                                      |
| `httpx`             | ControlClient + Reporter HTTP                      |
| `torch`             | PyTorch models + GPU monitor                       |
| `stable-baselines3` | SB3 models                                         |
| `redis`             | Redis transport                                    |
| `cloudflared`       | Cloudflare Tunnel (CLI tool, not a Python package) |

### Graceful Degradation

All optional packages are handled with `try/except ImportError`. The system runs without them — only the corresponding features are disabled.

---

## 12. Copy-Paste Colab Cells

### Cell 1: Run once (Setup)

```python
# ─── bt4t Extension Setup ──────────────────────────────────────────
import sys, os
from google.colab import drive
drive.mount('/content/drive')
sys.path.insert(0, '/content/drive/MyDrive/BITCOIN4Traders')

# Set configuration (copy URL from local terminal)
os.environ.setdefault('BT4T_LISTENER_URL', 'https://YOUR_URL.trycloudflare.com')
os.environ.setdefault('BT4T_API_TOKEN',    'bt4t-secret-token')
os.environ.setdefault('BT4T_NOTEBOOK_ID',  'training_v1')
os.environ.setdefault('ABLY_API_KEY',      'YOUR_ABLY_KEY')

from colab_bridge.colab_extension import bt4t
bt4t.install()
print("Setup complete:", bt4t.status())
```

### Cell 2: Start Module B

```python
# ─── Module B (RL Inference) start ─────────────────────────────────
from colab_bridge.module_b_colab import ModuleB, find_champion_on_drive

champion_path = find_champion_on_drive('/content/drive/MyDrive/BITCOIN4Traders')

engine = ModuleB(
    ably_key=os.environ['ABLY_API_KEY'],
    model_path=champion_path,
    symbol='BTCUSDT',
    min_confidence=0.55,
)
await engine.run()
```

### Cell 3: RL training with bt4t.step() (pattern)

```python
# ─── Training with extension monitoring ────────────────────────────
from colab_bridge.colab_extension import bt4t

# Normal training code — just add bt4t.step()
for epoch in range(1000):
    loss = train_one_epoch(model, optimizer, data)
    reward = evaluate(model, env)

    # bt4t.step() returns False when STOP command received
    if not bt4t.step(epoch=epoch, loss=float(loss), reward=float(reward)):
        print("Training stopped from local machine")
        break

    # Checkpoint every 100 epochs
    if epoch % 100 == 0:
        save_path = f"/content/drive/MyDrive/.../model_ep{epoch}.pth"
        torch.save(model.state_dict(), save_path)
        bt4t.send_checkpoint(save_path, {"loss": loss, "reward": reward})
```

### Cell 4: Training with decorator

```python
# ─── Training with bt4t.guard decorator ────────────────────────────
from colab_bridge.colab_extension import bt4t

@bt4t.guard   # Automatic retry + error handling
def run_full_training():
    for epoch in range(1000):
        loss = train_one_epoch(model, optimizer, data)
        bt4t.step(epoch=epoch, loss=float(loss))

run_full_training()
```

### Cell 5: Training with context manager

```python
# ─── Training with context manager ─────────────────────────────────
from colab_bridge.colab_extension import bt4t

with bt4t.session("experiment_v3_bollinger"):
    for epoch in range(1000):
        loss = train_one_epoch(model, optimizer, data)
        if not bt4t.step(epoch=epoch, loss=float(loss)):
            break
```

### Cell 6: Send command from Colab

```python
# ─── Query status ───────────────────────────────────────────────────
print(bt4t.status())
print("Repairs:", bt4t.repair_log())

# Send manual alert
bt4t.send_alert("Epoch 500 reached — please check metrics", level="INFO")
```

---

## 13. Troubleshooting

### "No Ably API Key"

```
ERROR: No Ably API Key!
```

**Solution:** Set `ABLY_API_KEY`:

```bash
# In .env:
ABLY_API_KEY=root.XXXXX:YYYYY

# In Colab:
os.environ['ABLY_API_KEY'] = 'root.XXXXX:YYYYY'
```

---

### Colab receives no market data

**Symptom:** `obs_buffer_size` stays 0, `last_market_data_age_s` increases.

**Checklist:**

1. Is Module A running locally? `python colab_bridge/module_a_local.py`
2. Same Ably key on both sides?
3. Symbol format correct? Module A: `BTC/USDT`, Module B: `BTCUSDT`
4. Ably connection active? See `connected` log message

---

### Local machine receives no signals

**Symptom:** No "Signal received" in Module A log.

**Checklist:**

1. Is Module B running in Colab?
2. Is confidence threshold sufficient? Currently: 0.55 (lower with `--min-conf 0.4`)
3. Is observation buffer sufficient? Inference only starts at ≥ 20 entries
4. Ably channel correct? `bt4t:signals`

---

### Trading is paused even though Colab is running

**Symptom:** "Portfolio: Trading PAUSED (no Colab heartbeat)"

**Cause:** No heartbeat from Colab in the last 90 seconds.

**Solution:**

```python
# In Colab: check heartbeat interval (default 10s)
# Is Module B running? Execute await engine.run()

# Resume manually (HTTP):
curl -X POST http://localhost:8765/trading/resume \
  -H "Authorization: Bearer bt4t-secret-token"
```

---

### bt4t Extension sends no reports

**Symptom:** "HTTP failed" + "Drive Fallback"

**Checklist:**

1. `BT4T_LISTENER_URL` set and correct?
2. Cloudflare Tunnel running locally? (`./cloudflared tunnel --url http://localhost:8765`)
3. ControlServer running? (`python colab_bridge/control_plane.py server`)
4. If Drive fallback is sufficient: messages in `/content/drive/MyDrive/BITCOIN4Traders/bt4t_bus/reports/`

---

### OOM despite MemoryMonitor

**Symptom:** Training crashes with OOM even though monitor is running.

**Cause:** Monitor only checks every 60s. OOM can occur in between.

**Solution:** Use `bt4t.guard` or `bt4t.session` additionally — the ExceptionHook catches OOM and halves `batch_size` in the running process.

---

### "Extension already installed"

```
WARNING | [bt4t] Extension already installed — skipping
```

Not an error. `bt4t.install()` is idempotent — multiple calls are safe.

---

### Repair fails ("no batch_size variable found")

**Cause:** Batch size variable has an unusual name or is a local variable.

**Solution:** Rename variable:

```python
# Instead of:
bs = 32
# Use:
BATCH_SIZE = 32   # or batch_size = 32
```

Or adjust manually after repair:

```python
# After InProcessRepair: check if change was desired
print(bt4t.repair_log())
```

---

_Documentation generated: 2026-03-12 | BITCOIN4Traders v1.0.0_
