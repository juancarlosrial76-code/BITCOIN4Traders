"""
Simple Historical Data Downloader
==================================
Download historical market data from Binance public API (no API key needed).
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
from loguru import logger
import time


def download_binance_klines(
    symbol: str,
    interval: str = "1h",
    start_date: datetime = None,
    end_date: datetime = None,
) -> pd.DataFrame:
    """
    Download kline data from Binance public API.

    No API key required - uses public endpoints.

    Args:
        symbol: Trading pair (e.g., 'BTCUSDT')
        interval: Candle interval (1m, 5m, 1h, 4h, 1d)
        start_date: Start date
        end_date: End date

    Returns:
        DataFrame with OHLCV data
    """
    if start_date is None:
        start_date = datetime.now() - timedelta(days=30)
    if end_date is None:
        end_date = datetime.now()

    logger.info(f"Downloading {symbol} {interval} from {start_date} to {end_date}")

    base_url = "https://api.binance.com/api/v3/klines"
    all_data = []

    # Convert to timestamps
    start_ts = int(start_date.timestamp() * 1000)
    end_ts = int(end_date.timestamp() * 1000)

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

            logger.debug(f"Downloaded {len(klines)} candles, total: {len(all_data)}")

            # Rate limiting
            time.sleep(0.1)

        except Exception as e:
            logger.error(f"Error downloading: {e}")
            break

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
    numeric_cols = ["open", "high", "low", "close", "volume"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col])

    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("timestamp", inplace=True)

    logger.success(f"Downloaded {len(df)} candles for {symbol}")

    return df[["open", "high", "low", "close", "volume"]]


def main():
    """Download and display sample data."""
    import sys

    sys.path.insert(0, "/home/hp17/Tradingbot/BITCOIN4Traders")

    from src.data.database import DatabaseManager

    logger.info("=" * 60)
    logger.info("Historical Data Download - BITCOIN4Traders")
    logger.info("=" * 60)

    # Initialize database
    db = DatabaseManager()
    logger.info("Database connected")

    # Download BTC data
    logger.info("\n1. Downloading BTCUSDT 1h data (30 days)...")
    btc_start = datetime.now() - timedelta(days=30)
    btc_data = download_binance_klines("BTCUSDT", "1h", start_date=btc_start)

    if not btc_data.empty:
        logger.info(f"   Data shape: {btc_data.shape}")
        logger.info(f"   Date range: {btc_data.index[0]} to {btc_data.index[-1]}")
        logger.info(
            f"   Price range: ${btc_data['low'].min():,.2f} - ${btc_data['high'].max():,.2f}"
        )

        # Save to database
        db.save_market_data(btc_data, "BTCUSDT", "1h")
        logger.success("   Saved to database")

    # Download ETH data
    logger.info("\n2. Downloading ETHUSDT 1h data (30 days)...")
    eth_start = datetime.now() - timedelta(days=30)
    eth_data = download_binance_klines("ETHUSDT", "1h", start_date=eth_start)

    if not eth_data.empty:
        logger.info(f"   Data shape: {eth_data.shape}")
        logger.info(f"   Date range: {eth_data.index[0]} to {eth_data.index[-1]}")
        logger.info(
            f"   Price range: ${eth_data['low'].min():,.2f} - ${eth_data['high'].max():,.2f}"
        )

        # Save to database
        db.save_market_data(eth_data, "ETHUSDT", "1h")
        logger.success("   Saved to database")

    # Verify data in database
    logger.info("\n3. Verifying database contents...")
    session = db.get_session()
    from src.data.database import MarketData

    count = session.query(MarketData).count()
    session.close()
    logger.info(f"   Total records in database: {count}")

    logger.info("\n" + "=" * 60)
    logger.success("Historical data download complete!")
    logger.info("=" * 60)

    # Show sample
    if not btc_data.empty:
        logger.info("\nSample BTC data (last 5 candles):")
        print(btc_data.tail().to_string())


if __name__ == "__main__":
    main()
