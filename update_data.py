"""
update_data.py — Binance Daten aktuell halten
==============================================
Laedt nur NEUE Bars seit dem letzten gespeicherten Timestamp.
Laeuft auf dem Linux Local Master (kein Geo-Block).

Aufruf:
    python3 update_data.py              # Alle Paare aktualisieren
    python3 update_data.py --push       # Danach automatisch Git-Push
    python3 update_data.py --symbol BTC # Nur BTC

Crontab (taeglich um 00:05 UTC):
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
# Konfiguration
# ---------------------------------------------------------------------------
CACHE_DIR = Path(__file__).parent / "data" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Paare und Timeframes die wir pflegen
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
    """Laedt alle Bars ab since_ms mit Paginierung."""
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
    return df.astype("float32")


def update_pair(exchange: ccxt.Exchange, symbol: str, timeframe: str) -> int:
    """
    Aktualisiert eine Datei mit neuen Bars.
    Gibt Anzahl neuer Bars zurueck.
    """
    path = parquet_path(symbol, timeframe)
    existing = load_existing(symbol, timeframe)

    if existing.empty:
        # Erstmaliger Download: alles seit Binance-Start
        since_ms = exchange.parse8601("2017-08-01T00:00:00Z")
        logger.info(f"{symbol} {timeframe}: Erstmalig herunterladen...")
    else:
        # Inkrementell: ab letztem Bar + 1 Tick
        last_ts = existing.index[-1]
        since_ms = int(last_ts.timestamp() * 1000) + 1
        logger.info(
            f"{symbol} {timeframe}: Update ab {last_ts.date()} "
            f"({len(existing):,} Bars vorhanden)..."
        )

    new_bars = fetch_new_bars(exchange, symbol, timeframe, since_ms)

    if new_bars.empty:
        logger.info(f"  Keine neuen Bars.")
        return 0

    # Zusammenfuehren und Duplikate entfernen
    if existing.empty:
        combined = new_bars
    else:
        combined = pd.concat([existing, new_bars])
        combined = combined[~combined.index.duplicated(keep="last")].sort_index()
        combined = combined.astype("float32")

    # Speichern
    combined.to_parquet(path, engine="pyarrow", compression="snappy")
    size_mb = path.stat().st_size / 1024**2

    n_new = len(new_bars)
    logger.success(
        f"  {symbol} {timeframe}: +{n_new} neue Bars | "
        f"Gesamt {len(combined):,} | {size_mb:.2f} MB | "
        f"bis {combined.index[-1].date()}"
    )
    return n_new


def git_push(n_updated: int) -> None:
    """Pusht aktualisierte Parquet-Dateien auf GitHub."""
    repo = Path(__file__).parent
    try:
        # Nur data/cache/*.parquet stagen
        subprocess.run(
            ["git", "add", "data/cache/*.parquet"],
            cwd=repo,
            check=True,
            capture_output=True,
        )
        # Pruefen ob etwas zu committen ist
        result = subprocess.run(
            ["git", "diff", "--cached", "--name-only"],
            cwd=repo,
            capture_output=True,
            text=True,
        )
        if not result.stdout.strip():
            logger.info("Git: Keine Aenderungen in data/cache/ - kein Push noetig.")
            return

        msg = f"Data update: {n_updated} new bars ({datetime.utcnow().strftime('%Y-%m-%d %H:%M')} UTC)"
        subprocess.run(
            ["git", "commit", "-m", msg], cwd=repo, check=True, capture_output=True
        )
        subprocess.run(
            ["git", "push", "origin", "main"], cwd=repo, check=True, capture_output=True
        )
        logger.success(f"Git push erfolgreich: '{msg}'")
    except subprocess.CalledProcessError as e:
        logger.error(f"Git-Fehler: {e.stderr.decode() if e.stderr else e}")


def main():
    parser = argparse.ArgumentParser(description="Binance Daten aktualisieren")
    parser.add_argument(
        "--push", action="store_true", help="Nach Update auf GitHub pushen"
    )
    parser.add_argument("--symbol", default=None, help="Nur dieses Symbol (z.B. 'BTC')")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("  Binance Daten-Update")
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
            logger.error(f"{symbol} {timeframe} fehlgeschlagen: {e}")

    logger.info(f"\nGesamt neue Bars: {total_new:,}")

    # Uebersicht aller Dateien
    logger.info("\nDateien in data/cache/:")
    for f in sorted(CACHE_DIR.glob("*_binance.parquet")):
        mb = f.stat().st_size / 1024**2
        df = pd.read_parquet(f)
        logger.info(
            f"  {f.name:40s} {len(df):>7,} Bars | {mb:.2f} MB | "
            f"{df.index[0].date()} - {df.index[-1].date()}"
        )

    if args.push and total_new > 0:
        git_push(total_new)
    elif args.push:
        logger.info("Kein Push (keine neuen Bars).")


if __name__ == "__main__":
    main()
