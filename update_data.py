"""
update_data.py — Keep Binance data up to date
==============================================
Loads only NEW bars since the last saved timestamp.
Runs on the Linux Local Master (no geo-block).

Usage:
    python3 update_data.py              # Update all pairs
    python3 update_data.py --push       # Then automatically git-push
    python3 update_data.py --symbol BTC # BTC only

Crontab (daily at 00:05 UTC):
    5 0 * * * cd /home/hp17/Tradingbot/Quantrivo/BITCOIN4Traders && python3 update_data.py --push >> logs/data_update.log 2>&1
"""

import sys
import time
import argparse
import subprocess
from pathlib import Path
from datetime import datetime

import ccxt
import pandas as pd

try:
    from loguru import logger
except ImportError:
    import logging

    logger = logging.getLogger(__name__)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
    )

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CACHE_DIR = Path(__file__).parent / "data" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Pairs and timeframes we maintain
PAIRS = [
    ("BTC/USDT", "1h"),
    ("BTC/USDT", "4h"),
    ("BTC/USDT", "1d"),
    ("ETH/USDT", "1h"),
]


def parquet_path(symbol: str, timeframe: str) -> Path:
    fname = f"{symbol.replace('/', '_')}_{timeframe}_binance.parquet"
    return CACHE_DIR / fname


def load_existing(symbol: str, timeframe: str) -> pd.DataFrame:
    path = parquet_path(symbol, timeframe)
    if path.exists():
        return pd.read_parquet(path)
    return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])


def fetch_new_bars(
    exchange: ccxt.Exchange,
    symbol: str,
    timeframe: str,
    since_ms: int,
) -> pd.DataFrame:
    """Loads all bars from since_ms with pagination."""
    all_ohlcv = []
    limit = 1000
    while True:
        batch = exchange.fetch_ohlcv(symbol, timeframe, since=since_ms, limit=limit)
        if not batch:
            break
        all_ohlcv.extend(batch)
        since_ms = batch[-1][0] + 1
        if len(batch) < limit:
            break
        time.sleep(0.12)

    if not all_ohlcv:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    df = pd.DataFrame(
        all_ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"]
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df = df.set_index("timestamp").sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df.astype("float64")


def update_pair(exchange: ccxt.Exchange, symbol: str, timeframe: str) -> int:
    """
    Updates a file with new bars.
    Returns the number of new bars.
    """
    path = parquet_path(symbol, timeframe)
    existing = load_existing(symbol, timeframe)

    if existing.empty:
        # First-time download: everything since Binance launch
        since_ms = exchange.parse8601("2017-08-01T00:00:00Z")
        logger.info(f"{symbol} {timeframe}: Downloading for the first time...")
    else:
        # Incremental: from last bar + 1 tick
        last_ts = existing.index[-1]
        since_ms = int(last_ts.timestamp() * 1000) + 1
        logger.info(
            f"{symbol} {timeframe}: Update from {last_ts.date()} "
            f"({len(existing):,} bars present)..."
        )

    new_bars = fetch_new_bars(exchange, symbol, timeframe, since_ms)

    if new_bars.empty:
        logger.info(f"  No new bars.")
        return 0

    # Merge and remove duplicates
    if existing.empty:
        combined = new_bars
    else:
        combined = pd.concat([existing, new_bars])
        combined = combined[~combined.index.duplicated(keep="last")].sort_index()
        combined = combined.astype("float64")

    # Save
    combined.to_parquet(path, engine="pyarrow", compression="snappy")
    size_mb = path.stat().st_size / 1024**2

    n_new = len(new_bars)
    logger.success(
        f"  {symbol} {timeframe}: +{n_new} new bars | "
        f"Total {len(combined):,} | {size_mb:.2f} MB | "
        f"up to {combined.index[-1].date()}"
    )
    return n_new


def git_push(n_updated: int) -> None:
    """Pushes updated Parquet files to GitHub."""
    repo = Path(__file__).parent
    try:
        # Stage only data/cache/*.parquet
        subprocess.run(
            ["git", "add", "data/cache/*.parquet"],
            cwd=repo,
            check=True,
            capture_output=True,
        )
        # Check if there is anything to commit
        result = subprocess.run(
            ["git", "diff", "--cached", "--name-only"],
            cwd=repo,
            capture_output=True,
            text=True,
        )
        if not result.stdout.strip():
            logger.info("Git: No changes in data/cache/ - no push needed.")
            return

        msg = f"Data update: {n_updated} new bars ({datetime.utcnow().strftime('%Y-%m-%d %H:%M')} UTC)"
        subprocess.run(
            ["git", "commit", "-m", msg], cwd=repo, check=True, capture_output=True
        )
        subprocess.run(
            ["git", "push", "origin", "main"], cwd=repo, check=True, capture_output=True
        )
        logger.success(f"Git push successful: '{msg}'")
    except subprocess.CalledProcessError as e:
        logger.error(f"Git error: {e.stderr.decode() if e.stderr else e}")


def main():
    parser = argparse.ArgumentParser(description="Update Binance data")
    parser.add_argument(
        "--push", action="store_true", help="Push to GitHub after update"
    )
    parser.add_argument("--symbol", default=None, help="Only this symbol (e.g. 'BTC')")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("  Binance Data Update")
    logger.info(f"  {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
    logger.info("=" * 60)

    exchange = ccxt.binance({"enableRateLimit": True})

    total_new = 0
    pairs = PAIRS
    if args.symbol:
        pairs = [(s, tf) for s, tf in PAIRS if args.symbol.upper() in s]

    for symbol, timeframe in pairs:
        try:
            n = update_pair(exchange, symbol, timeframe)
            total_new += n
        except Exception as e:
            logger.error(f"{symbol} {timeframe} failed: {e}")

    logger.info(f"\nTotal new bars: {total_new:,}")

    # Overview of all files
    logger.info("\nFiles in data/cache/:")
    for f in sorted(CACHE_DIR.glob("*_binance.parquet")):
        mb = f.stat().st_size / 1024**2
        df = pd.read_parquet(f)
        logger.info(
            f"  {f.name:40s} {len(df):>7,} bars | {mb:.2f} MB | "
            f"{df.index[0].date()} - {df.index[-1].date()}"
        )

    if args.push and total_new > 0:
        git_push(total_new)
    elif args.push:
        logger.info("No push (no new bars).")


if __name__ == "__main__":
    main()
