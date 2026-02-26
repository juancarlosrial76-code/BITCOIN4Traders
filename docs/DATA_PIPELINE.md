# BITCOIN4Traders — Data Pipeline

## Table of Contents

1. [Data Sources](#1-data-sources)
2. [OHLCV Data Fetching with ccxt_loader.py](#2-ohlcv-data-fetching)
3. [Data Quality Assessment](#3-data-quality-assessment)
4. [Feature Engineering Pipeline](#4-feature-engineering-pipeline)
5. [Storage Layer](#5-storage-layer)
6. [Data Versioning and Cache Management](#6-data-versioning-and-cache-management)
7. [How to Add a New Data Source](#7-how-to-add-a-new-data-source)

---

## 1. Data Sources

### Primary Source: Binance via CCXT

The system uses [CCXT](https://github.com/ccxt/ccxt) (CryptoCurrency eXchange Trading Library) as its primary data interface. CCXT provides a unified API across 100+ exchanges. The system is configured to use **Binance Spot** by default.

**What data is fetched:**

| Data Type | Source | Format | Timeframes |
|---|---|---|---|
| OHLCV candles | Binance REST API | `[timestamp, open, high, low, close, volume]` | 1m, 5m, 15m, 1h, 4h, 1d |
| Live ticks | Binance WebSocket | `{price, volume, side, timestamp}` | Real-time |
| Order book | Binance REST/WS | L2 10-level bid/ask | Real-time |

**Supported exchanges** (CCXT provides unified access to all):

```python
import ccxt
print([e for e in ccxt.exchanges if 'binance' in e or 'kraken' in e or 'coinbase' in e])
# ['binance', 'binanceus', 'binancecoinm', 'binanceusdm', 'kraken', 'coinbasepro', ...]
```

Switching exchange requires only changing `exchange_id` in the Hydra config:

```yaml
# config/data/exchange.yaml
exchange:
  id: "kraken"          # was "binance"
  type: "spot"
  rate_limit_ms: 200    # kraken is slower
```

### API Rate Limits

Binance REST API limits: 1200 requests per minute, 1 request per 100ms recommended.
The `CCXTDataLoader` respects this with `enableRateLimit: True` and `rateLimit: 100` (ms).

**No API key required for historical OHLCV data** (public endpoint).
API keys are only needed for placing live orders.

---

## 2. OHLCV Data Fetching

### CCXTDataLoader (`src/data/ccxt_loader.py`)

The `CCXTDataLoader` handles all historical data acquisition with Parquet-based caching.

**Architecture:**

```
download_and_cache(symbol, timeframe, start_date)
        │
        ├─ Check cache exists?
        │   YES → load_local() → return DataFrame
        │
        └─ NO → Fetch from exchange in batches of 1000 candles
                    │
                    ├─ ccxt.fetch_ohlcv() with exponential backoff retry
                    ├─ Advance `since` pointer after each batch
                    ├─ Stop when: partial batch received OR end_date reached
                    │
                    └─ _save_to_cache() → Parquet (snappy, atomic write)
```

**Batch fetching logic:**

Binance returns a maximum of 1000 candles per request. For 3 years of hourly data (26,280 candles), the loader makes ~27 sequential API calls, advancing the `since` timestamp after each batch:

```python
while True:
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=1000)
    all_ohlcv.extend(ohlcv)
    since = ohlcv[-1][0] + 1    # Advance past last candle
    if len(ohlcv) < 1000:       # Partial batch = no more data
        break
```

**Retry logic:**

Network errors trigger exponential backoff (2^retry_count seconds). After 3 failures, the download aborts with an error. Exchange errors (e.g., invalid symbol) are propagated immediately.

**Atomic write:**

Cache files are written to a temporary `.tmp` path first, then atomically renamed. This prevents partially-written files from corrupting the cache if the process is interrupted.

```python
temp_path = cache_path.with_suffix(".tmp")
df.to_parquet(temp_path, ...)
temp_path.replace(cache_path)   # Atomic on POSIX systems
```

**Cache naming:**

Cache filenames include an MD5 hash of the parameters to auto-invalidate stale entries:

```
BTC_USDT_1h_a3f2b891.parquet
     │     │       │
  symbol  tf   MD5(exchange+symbol+tf+dates)[:8]
```

### Downloading Data

**Quick start:**

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

# Download and cache (or load from cache if exists)
df = loader.download_and_cache(
    symbol="BTC/USDT",
    timeframe="1h",
    start_date="2022-01-01",
    end_date="2024-12-31",
    force_refresh=False,   # True to re-download
)
print(df.shape)        # (26280, 5)
print(df.dtypes)       # All float64
print(df.index)        # DatetimeIndex
```

**Force refresh:**

```python
# Re-download even if cache exists (use when you need today's data)
df = loader.download_and_cache(
    "BTC/USDT", "1h", "2022-01-01", force_refresh=True
)
```

---

## 3. Data Quality Assessment

### DataQualityAssessor (`src/data/assessor.py`)

Before features are computed, all data passes through quality checks. The assessor detects and flags:

**Check 1: Missing bars (gaps)**

```python
expected_freq = pd.Timedelta("1h")
actual_gaps = df.index.to_series().diff().dropna()
large_gaps = actual_gaps[actual_gaps > expected_freq * 2]
# Reports: "3 gaps found. Largest: 2025-03-15 (4h missing)"
```

Large gaps indicate exchange downtime or API issues. Gaps under 3 bars are forward-filled; larger gaps are flagged for human review.

**Check 2: Price outliers (spike detection)**

```python
log_returns = np.log(df['close'] / df['close'].shift(1))
z_scores = (log_returns - log_returns.mean()) / log_returns.std()
outliers = df[np.abs(z_scores) > 5]
# |z| > 5: >5 standard deviations from mean → likely data error
```

**Check 3: Volume anomalies**

Zero-volume bars and bars with volume > 20× rolling median are flagged. These often indicate exchange issues rather than real trading.

**Check 4: OHLC consistency**

```python
invalid = df[(df['high'] < df['low']) |   # High < Low (impossible)
             (df['open'] > df['high']) |   # Open above High
             (df['close'] > df['high']) |  # Close above High
             (df['open'] < df['low']) |    # Open below Low
             (df['close'] < df['low'])]    # Close below Low
```

**Check 5: Distribution shift (concept drift detection)**

For live monitoring, the assessor compares the statistical distribution of recent data against the training distribution using Population Stability Index (PSI):

```
PSI < 0.1:  Distribution stable (green)
PSI 0.1-0.2: Minor shift (yellow, monitor)
PSI > 0.2:  Significant shift (red, retrain recommended)
```

---

## 4. Feature Engineering Pipeline

Raw OHLCV data is transformed into a feature matrix through three sequential stages. Each stage must be run in order.

### Stage 1: FeatureEngine (`src/features/feature_engine.py`)

**Critical principle:** `fit()` is called only on training data. `transform()` uses the statistics from the fit step. This is identical to the scikit-learn pattern and prevents lookahead bias.

```
Training:  engine.fit_transform(train_df)   → computes mean, std; stores them
Test/Live: engine.transform(live_df)        → uses stored stats ONLY
```

Features computed by FeatureEngine:

| Feature | Computation | Purpose |
|---|---|---|
| `log_return` | `ln(close_t / close_{t-1})` | Stationary price changes |
| `volatility` | `rolling_std(log_return, window=20)` | Local risk estimate |
| `OU_signal` | OrnsteinUhlenbeck deviation from equilibrium | Mean-reversion signal |
| `norm_return` | `log_return / volatility` | Volatility-normalised returns |
| `volume_ratio` | `volume / rolling_mean(volume, 20)` | Volume anomaly indicator |

After feature computation, all features are scaled using one of:
- `StandardScaler` (default): `(x - mean) / std`
- `RobustScaler`: `(x - median) / IQR` (outlier-resistant)
- `MinMaxScaler`: `(x - min) / (max - min)` (bounded to [0,1])

The scaler type is configured via Hydra: `features.scaler: standard`.

### Stage 2: MultiTimeframeAnalyzer (`src/features/multi_timeframe.py`)

OHLCV data at multiple timeframes is analysed to produce a **top-down alignment score**. The idea is that a trade has higher probability when all timeframes agree on direction.

**Timeframes analysed:** 1m, 5m, 15m, 1h, 4h, 1d, 1w

**Alignment score computation:**

```python
signals = {
    "1m":  rsi_signal(df_1m),   # +1, 0, -1
    "5m":  rsi_signal(df_5m),
    "15m": ema_signal(df_15m),
    "1h":  macd_signal(df_1h),
    "4h":  macd_signal(df_4h),
    "1d":  trend_signal(df_1d),
}

weights = {"1m": 0.05, "5m": 0.10, "15m": 0.15, "1h": 0.25, "4h": 0.25, "1d": 0.20}
alignment = sum(w * signals[tf] for tf, w in weights.items())
# Range: [-1.0, +1.0]. +1.0 = all timeframes bullish
```

The alignment score is appended as an additional feature to the FeatureEngine output.

### Stage 3: MicrostructureAnalyzer (`src/features/microstructure.py`)

Market microstructure features capture the trading mechanics that institutional traders exploit:

| Feature | Computation | Interpretation |
|---|---|---|
| `bid_ask_spread` | `(ask - bid) / mid_price` | Transaction cost proxy |
| `order_flow_imbalance` | `(buy_vol - sell_vol) / total_vol` | Buying vs. selling pressure |
| `market_impact` | `|trade_size| / avg_daily_volume × volatility` | Cost of executing a trade |
| `quote_depth` | `sum(volume at top 5 levels)` | Liquidity available |
| `toxicity_score` | Abnormal order flow pattern score | Manipulation risk indicator |

These features are particularly valuable for the execution layer (timing entries to avoid toxic flow).

### Complete Pipeline Example

```python
from src.features.feature_engine import FeatureEngine
from src.features.multi_timeframe import MultiTimeframeAnalyzer
from src.features.microstructure import MicrostructureAnalyzer

# Stage 1: Base features (fit on train only)
engine = FeatureEngine()
train_features = engine.fit_transform(train_df)
test_features  = engine.transform(test_df)

# Stage 2: Multi-timeframe alignment
mtf = MultiTimeframeAnalyzer(timeframes=["1h", "4h", "1d"])
train_features["alignment"] = mtf.compute(train_df)
test_features["alignment"]  = mtf.compute(test_df)

# Stage 3: Microstructure (if order book data available)
ms = MicrostructureAnalyzer()
train_features = ms.add_features(train_features, order_book_train)
test_features  = ms.add_features(test_features, order_book_test)

print(f"Feature matrix shape: {train_features.shape}")
# e.g., (8760, 47)  = 8760 hourly bars, 47 features
```

---

## 5. Storage Layer

### SQLite Local Database (`src/data/sqlite_local.py`)

The primary storage for the Linux PC is a local SQLite database. It holds all persistent data.

**Tables:**

| Table | Content | Key Columns |
|---|---|---|
| `market_data` | OHLCV price bars | `symbol`, `timeframe`, `timestamp` (ms) |
| `champion_history` | All past champion entries | `sharpe`, `calmar`, `sortino`, `profit_factor` |
| `trades` | Executed live trades | `symbol`, `side`, `price`, `quantity`, `fee` |
| `portfolio_snapshots` | Hourly equity snapshots | `total_value`, `daily_pnl`, `max_drawdown` |
| `heartbeat` | System liveness log | `source`, `status`, `signal` |

**Performance characteristics:**

- WAL mode (`PRAGMA journal_mode=WAL`): multiple concurrent readers, one writer
- 64 MB in-memory page cache (`PRAGMA cache_size=-64000`)
- Unique index on `(symbol, timeframe, timestamp)`: fast deduplication
- Query of 1,000,000 bars by symbol+timeframe: < 10ms

**Usage examples:**

```python
from src.data.sqlite_local import get_local_db

db = get_local_db()   # Singleton, auto-creates if not exists

# Save OHLCV (idempotent: INSERT OR IGNORE)
db.save_ohlcv(df, symbol="BTC/USDT", timeframe="1h")

# Load last 90 days
df = db.load_ohlcv("BTC/USDT", timeframe="1h", days=90)

# Load last 500 bars
df = db.load_ohlcv("BTC/USDT", timeframe="1h", limit=500)

# Save a champion entry
db.save_champion(champion_meta, source="local_master")

# Save an executed trade
db.save_trade({
    "trade_id": "abc123",
    "symbol": "BTC/USDT",
    "side": "BUY",
    "quantity": 0.01,
    "price": 45000.0,
    "fee": 0.45,
})

# Database stats
print(db.stats())
# {'db_size_mb': 124.5, 'market_data_rows': 2628000, 'symbols': ['BTC/USDT']}
```

### Parquet Cache

Short-term data cache for the feature pipeline uses Parquet format:

- **Location:** `data/cache/`
- **Compression:** Snappy (fastest decompression, moderate compression)
- **Load speed:** ~50,000 rows/second (vs. ~5,000 rows/second for CSV)
- **Dtype preservation:** All float64 columns preserved exactly

The Parquet cache is populated by `CCXTDataLoader.download_and_cache()` and read by `load_local()`.

### Google Drive Sync

Drive is the shared state between the Linux PC and Google Colab. The `DriveManager` syncs:

| File | Direction | Frequency | Purpose |
|---|---|---|---|
| `multiverse_champion_meta.json` | PC → Drive | After each training run | Champion parameters for Colab |
| `btc_1h.parquet` | PC → Drive | Daily | Data for Colab training |
| `colab_status.json` | Colab → Drive | Every 5 min | Heartbeat from Colab |
| `restart_requested.json` | PC → Drive | On timeout detection | Command to restart Colab |
| `error_report.json` | Colab → Drive | On error | Error details for auto-repair |
| Model weights (`.pth`) | Colab → Drive | Every 50 iterations | Checkpoint for recovery |

---

## 6. Data Versioning and Cache Management

### Automatic Cache Invalidation

Cache files are named with a hash of all parameters that affect the data content:

```
{exchange}_{symbol}_{timeframe}_{start}_{end} → MD5 → first 8 chars
```

If any parameter changes, the hash changes, a new file is created, and the old file is ignored. This prevents silent use of stale data.

### Manual Cache Management

```python
# Check what's in the cache
import os
from pathlib import Path

cache_dir = Path("data/cache")
for f in sorted(cache_dir.glob("*.parquet")):
    size_mb = f.stat().st_size / 1024 / 1024
    print(f"{f.name}: {size_mb:.1f} MB")

# Delete old cache entries (> 30 days)
db = get_local_db()
deleted = db.delete_old_data(days=365)
print(f"Deleted {deleted} rows older than 365 days")

# Compact the SQLite file
db.vacuum()
```

### Recommended Retention Policy

| Data | Retention | Reason |
|---|---|---|
| OHLCV (SQLite) | 2 years | Sufficient for walk-forward validation |
| OHLCV (Parquet) | Indefinite | Small file size; no reason to delete |
| Trade history | Indefinite | Required for tax/audit |
| Champion history | Indefinite | Track strategy evolution |
| Portfolio snapshots | 1 year | Rolling performance analysis |

---

## 7. How to Add a New Data Source

### Step 1: Create a data loader class

```python
# src/data/my_source_loader.py

from src.data.ccxt_loader import DataLoaderConfig
import pandas as pd

class MySourceLoader:
    """
    Custom data loader for a new source.
    Must produce a DataFrame with DatetimeIndex and columns:
    [open, high, low, close, volume]
    """

    def __init__(self, config: DataLoaderConfig):
        self.config = config

    def download_and_cache(
        self,
        symbol: str,
        timeframe: str,
        start_date: str,
        end_date: str = None,
    ) -> pd.DataFrame:
        # Fetch from your source
        raw_data = self._fetch(symbol, timeframe, start_date, end_date)

        # Convert to standard format
        df = pd.DataFrame(raw_data)
        df.index = pd.to_datetime(df["timestamp"], unit="ms")
        df = df[["open", "high", "low", "close", "volume"]].astype("float64")
        df.sort_index(inplace=True)

        return df
```

### Step 2: Register in Hydra config

```yaml
# config/data/source.yaml
source:
  type: "my_source"
  exchange_id: "my_exchange"
  rate_limit_ms: 200
```

### Step 3: Wire into DataManager

```python
# src/data/data_manager.py

def _create_loader(cfg):
    if cfg.data.source.type == "ccxt":
        return CCXTDataLoader(config)
    elif cfg.data.source.type == "my_source":
        return MySourceLoader(config)
    else:
        raise ValueError(f"Unknown data source: {cfg.data.source.type}")
```

### Step 4: Run the quality assessor

Before using new data in training, validate it:

```python
from src.data.assessor import DataQualityAssessor

assessor = DataQualityAssessor()
report = assessor.assess(df)

print(f"Quality score: {report.score}/100")
print(f"Gaps found: {len(report.gaps)}")
print(f"Outliers: {len(report.outliers)}")

if report.score < 80:
    print("WARNING: Data quality below threshold. Review before training.")
```

### Step 5: Test end-to-end

```python
# Verify the new source produces correct features
from src.features.feature_engine import FeatureEngine

engine = FeatureEngine()
features = engine.fit_transform(df)

assert not features.isna().any().any(), "NaN in features after fit_transform"
assert (features.std() > 0).all(), "Zero-variance features detected"
print(f"Feature matrix: {features.shape}, no NaN, all features have variance")
```
