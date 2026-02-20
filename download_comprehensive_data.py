"""
Comprehensive Historical Data Downloader
=========================================
Download extensive historical data for backtesting and training.

Features:
- Multiple symbols (BTC, ETH, major alts)
- Multiple timeframes (1h, 4h, 1d)
- 1+ years of data
- Automatic database storage
- Progress tracking
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict
from loguru import logger
import time
import sys

sys.path.insert(0, "/home/hp17/Tradingbot/BITCOIN4Traders")

from src.data.database import DatabaseManager


def download_binance_klines(
    symbol: str,
    interval: str,
    start_date: datetime,
    end_date: datetime = None,
) -> pd.DataFrame:
    """Download kline data from Binance public API."""
    if end_date is None:
        end_date = datetime.now()

    base_url = "https://api.binance.com/api/v3/klines"
    all_data = []

    start_ts = int(start_date.timestamp() * 1000)
    end_ts = int(end_date.timestamp() * 1000)

    logger.info(
        f"Downloading {symbol} {interval} ({start_date.date()} to {end_date.date()})"
    )

    while start_ts < end_ts:
        try:
            params = {
                "symbol": symbol,
                "interval": interval,
                "startTime": start_ts,
                "limit": 1000,
            }

            response = requests.get(base_url, params=params, timeout=30)
            response.raise_for_status()

            klines = response.json()

            if not klines:
                break

            all_data.extend(klines)

            # Move to next batch
            start_ts = klines[-1][0] + 1

            # Progress indicator
            if len(all_data) % 10000 == 0:
                logger.info(f"  ... downloaded {len(all_data)} candles")

            # Rate limiting
            time.sleep(0.05)

        except Exception as e:
            logger.error(f"Error downloading: {e}")
            time.sleep(1)  # Wait before retry
            continue

    if not all_data:
        logger.warning(f"No data downloaded for {symbol}")
        return pd.DataFrame()

    # Convert to DataFrame
    df = pd.DataFrame(
        all_data,
        columns=[
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time",
            "quote_volume",
            "trades",
            "taker_buy_base",
            "taker_buy_quote",
            "ignore",
        ],
    )

    # Convert types
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col])

    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("timestamp", inplace=True)

    logger.success(f"Downloaded {len(df)} candles for {symbol} {interval}")

    return df[["open", "high", "low", "close", "volume"]]


def download_comprehensive_dataset(
    symbols: List[str] = None,
    timeframes: List[str] = None,
    days: int = 365,
):
    """
    Download comprehensive historical dataset.

    Args:
        symbols: List of trading pairs
        timeframes: List of timeframes
        days: Number of days to download
    """
    if symbols is None:
        symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"]

    if timeframes is None:
        timeframes = ["1h", "4h", "1d"]

    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)

    logger.info("=" * 70)
    logger.info(f"COMPREHENSIVE DATA DOWNLOAD")
    logger.info(f"Symbols: {symbols}")
    logger.info(f"Timeframes: {timeframes}")
    logger.info(f"Period: {days} days ({start_date.date()} to {end_date.date()})")
    logger.info("=" * 70)

    # Initialize database
    db = DatabaseManager()

    total_records = 0
    results = {}

    for symbol in symbols:
        results[symbol] = {}

        for timeframe in timeframes:
            try:
                logger.info(f"\n[{symbol} - {timeframe}]")

                # Download data
                df = download_binance_klines(symbol, timeframe, start_date, end_date)

                if not df.empty:
                    # Save to database
                    db.save_market_data(df, symbol, timeframe)

                    # Track results
                    results[symbol][timeframe] = {
                        "records": len(df),
                        "start": df.index[0],
                        "end": df.index[-1],
                    }
                    total_records += len(df)

                    logger.info(f"  Records: {len(df)}")
                    logger.info(f"  Range: {df.index[0]} to {df.index[-1]}")
                    logger.info(
                        f"  Price: ${df['low'].min():,.2f} - ${df['high'].max():,.2f}"
                    )
                else:
                    results[symbol][timeframe] = {"records": 0}

            except Exception as e:
                logger.error(f"Failed to download {symbol} {timeframe}: {e}")
                results[symbol][timeframe] = {"records": 0, "error": str(e)}

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("DOWNLOAD SUMMARY")
    logger.info("=" * 70)

    for symbol in symbols:
        logger.info(f"\n{symbol}:")
        for timeframe in timeframes:
            info = results[symbol].get(timeframe, {"records": 0})
            if info["records"] > 0:
                logger.info(f"  {timeframe}: {info['records']:,} records")

    logger.info(f"\nTotal records downloaded: {total_records:,}")

    # Verify database
    session = db.get_session()
    from src.data.database import MarketData

    db_count = session.query(MarketData).count()
    session.close()

    logger.info(f"Total records in database: {db_count:,}")
    logger.info("=" * 70)
    logger.success("Comprehensive download complete!")

    return results


def main():
    """Main download function."""
    import argparse

    parser = argparse.ArgumentParser(description="Download historical market data")
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=["BTCUSDT", "ETHUSDT"],
        help="Symbols to download (default: BTCUSDT ETHUSDT)",
    )
    parser.add_argument(
        "--timeframes", nargs="+", default=["1h"], help="Timeframes (default: 1h)"
    )
    parser.add_argument(
        "--days", type=int, default=365, help="Number of days (default: 365)"
    )

    args = parser.parse_args()

    download_comprehensive_dataset(
        symbols=args.symbols, timeframes=args.timeframes, days=args.days
    )


if __name__ == "__main__":
    main()
