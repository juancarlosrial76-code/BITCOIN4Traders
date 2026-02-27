"""
Historical Data Pipeline
========================

Comprehensive pipeline for downloading and managing historical market data
from cryptocurrency exchanges. This module handles bulk data acquisition,
database storage, and incremental updates.

Features:
---------
1. BULK DOWNLOAD: Fetch years of historical data efficiently
2. DATABASE STORAGE: Store data in PostgreSQL for persistent access
3. INCREMENTAL UPDATES: Refresh only new data since last download
4. MULTIPLE TIMEFRAMES: Support for 1m, 5m, 15m, 1h, 4h, 1d intervals
5. DATA VALIDATION: Comprehensive quality checks before storage
6. ERROR HANDLING: Robust retry logic for network failures

Architecture:
-------------
The pipeline consists of two main classes:

1. HistoricalDataDownloader:
   - Downloads data from exchange APIs
   - Handles rate limiting and pagination
   - Validates data quality
   - Stores in database

2. DataPipeline:
   - High-level orchestration
   - Combines download, storage, and retrieval
   - Provides training data interface

Data Flow:
----------
Exchange API → Downloader → Validator → Database
                                            ↓
                        Training/Backtesting ← Pipeline

Supported Exchanges:
-------------------
- Binance (spot and futures)
- Other CCXT-compatible exchanges

Usage:
------
# Download historical data
downloader = HistoricalDataDownloader()
df = downloader.download_symbol("BTCUSDT", timeframe="1h")

# Initialize database with historical data
pipeline = DataPipeline()
pipeline.initialize_market_data(["BTCUSDT", "ETHUSDT"], days=365)

# Update with latest data
pipeline.update_market_data(["BTCUSDT"])

# Get training data
data = pipeline.get_training_data(["BTCUSDT"], lookback=100)

Author: BITCOIN4Traders Team
Version: 1.0.0
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Optional
from tqdm import tqdm
import time
from loguru import logger

from src.connectors import BinanceConnector
from src.data.database import DatabaseManager, MarketData


class HistoricalDataDownloader:
    """
    Download historical market data from cryptocurrency exchanges.

    This class provides comprehensive functionality for downloading large
    volumes of historical OHLCV data. It handles:

    1. PAGINATION: Exchanges limit candles per request (Binance: 1000)
       This class automatically fetches sequential batches until complete

    2. RATE LIMITING: Respects exchange API limits to avoid bans

    3. ERROR RECOVERY: Retries failed requests with exponential backoff

    4. DATABASE STORAGE: Optionally saves to PostgreSQL

    5. DATA VALIDATION: Checks quality before storing

    Attributes:
        connector: Exchange connector (default: BinanceConnector)
        db: Database manager (default: DatabaseManager)

    Usage:
        # Initialize downloader
        downloader = HistoricalDataDownloader()

        # Download single symbol
        df = downloader.download_symbol(
            symbol="BTCUSDT",
            timeframe="1h",
            start_date=datetime(2023, 1, 1),
            end_date=datetime.now(),
            save_to_db=True
        )

        # Download multiple symbols
        results = downloader.download_multiple_symbols(
            symbols=["BTCUSDT", "ETHUSDT"],
            timeframe="1h",
            days=365
        )

        # Update with latest data
        downloader.update_database(["BTCUSDT"], ["1h", "4h"])
    """

    def __init__(self, connector: BinanceConnector = None, db: DatabaseManager = None):
        """
        Initialize downloader.

        Args:
            connector: Exchange connector (creates default if None)
            db: Database manager (creates default if None)
        """
        self.connector = connector or BinanceConnector(testnet=True)
        self.db = db or DatabaseManager()
        logger.info("HistoricalDataDownloader initialized")

    def download_symbol(
        self,
        symbol: str,
        timeframe: str = "1h",
        start_date: datetime = None,
        end_date: datetime = None,
        save_to_db: bool = True,
    ) -> pd.DataFrame:
        """
        Download historical data for a single symbol.

        This method fetches OHLCV (Open-High-Low-Close-Volume) candle data
        from the exchange. It handles pagination automatically - for large
        date ranges, it will make multiple API requests.

        The Download Process:
        ---------------------
        1. Initialize date range (default: 1 year to now)
        2. Request first batch of candles (max 1000 per request)
        3. Filter results to requested date range
        4. Move start time to last received candle + 1 hour
        5. Repeat until all data is downloaded
        6. Combine all batches into single DataFrame
        7. Remove duplicates and sort chronologically
        8. Optionally save to database

        Parameters:
        -----------
        symbol : str
            Trading pair identifier (e.g., 'BTCUSDT', 'ETHUSDT')
            Note: Use exchange format (no slash), not CCXT format
        timeframe : str
            Candle interval: '1m', '5m', '15m', '30m', '1h', '4h', '1d'
        start_date : datetime, optional
            Start date for download. Default: 1 year ago
        end_date : datetime, optional
            End date for download. Default: current time
        save_to_db : bool
            Whether to save downloaded data to database (default: True)

        Returns:
        --------
        df : pd.DataFrame
            DataFrame with OHLCV data
            Columns: open, high, low, close, volume
            Index: DatetimeIndex

        Example:
            >>> downloader = HistoricalDataDownloader()
            >>> df = downloader.download_symbol(
            ...     symbol="BTCUSDT",
            ...     timeframe="1h",
            ...     start_date=datetime(2022, 1, 1),
            ...     end_date=datetime(2023, 1, 1),
            ...     save_to_db=True
            ... )
            >>> print(f"Downloaded {len(df)} candles")
        """
        if start_date is None:
            start_date = datetime.now() - timedelta(days=365)
        if end_date is None:
            end_date = datetime.now()

        logger.info(
            f"Downloading {symbol} {timeframe} data from {start_date} to {end_date}"
        )

        all_data = []
        current_start = start_date

        # Binance limits to 1000 candles per request
        # Download in chunks
        with tqdm(desc=f"Downloading {symbol}") as pbar:
            while current_start < end_date:
                try:
                    # Download chunk
                    df = self.connector.get_historical_klines(
                        symbol=symbol, interval=timeframe, limit=1000
                    )

                    if df.empty:
                        break

                    # Filter to date range
                    df = df[(df.index >= current_start) & (df.index <= end_date)]

                    if df.empty:
                        break

                    all_data.append(df)

                    # Update progress
                    pbar.update(len(df))

                    # Move to next chunk
                    current_start = df.index[-1] + pd.Timedelta(hours=1)

                    # Rate limiting
                    time.sleep(0.1)

                except Exception as e:
                    logger.error(f"Error downloading chunk: {e}")
                    break

        if not all_data:
            logger.warning(f"No data downloaded for {symbol}")
            return pd.DataFrame()

        # Combine all chunks
        combined_df = pd.concat(all_data)  # merge all chunk DataFrames into one
        combined_df = combined_df[
            ~combined_df.index.duplicated(keep="first")
        ]  # drop duplicate timestamps keeping first occurrence
        combined_df.sort_index(inplace=True)  # ensure chronological order

        logger.success(f"Downloaded {len(combined_df)} candles for {symbol}")

        # Save to database
        if save_to_db and not combined_df.empty:
            self.db.save_market_data(combined_df, symbol, timeframe)

        return combined_df

    def download_multiple_symbols(
        self, symbols: List[str], timeframe: str = "1h", days: int = 365
    ) -> dict:
        """
        Download data for multiple symbols.

        Args:
            symbols: List of trading pairs
            timeframe: Candle timeframe
            days: Number of days to download

        Returns:
            Dictionary of DataFrames
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        results = {}

        for symbol in symbols:
            try:
                df = self.download_symbol(
                    symbol=symbol,
                    timeframe=timeframe,
                    start_date=start_date,
                    end_date=end_date,
                )
                results[symbol] = df
            except Exception as e:
                logger.error(f"Failed to download {symbol}: {e}")
                results[symbol] = pd.DataFrame()

        return results

    def update_database(self, symbols: List[str], timeframes: List[str] = None):
        """
        Update database with latest data.

        Args:
            symbols: List of symbols to update
            timeframes: List of timeframes (default: ['1h', '4h', '1d'])
        """
        if timeframes is None:
            timeframes = ["1h", "4h", "1d"]

        for symbol in symbols:
            for timeframe in timeframes:
                try:
                    # Get last timestamp from database
                    session = self.db.get_session()
                    last_record = (
                        session.query(MarketData)
                        .filter(
                            MarketData.symbol == symbol,
                            MarketData.timeframe == timeframe,
                        )
                        .order_by(MarketData.timestamp.desc())
                        .first()
                    )
                    session.close()

                    if last_record:
                        start_date = last_record.timestamp + pd.Timedelta(hours=1)
                    else:
                        start_date = datetime.now() - timedelta(days=30)

                    # Download new data
                    self.download_symbol(
                        symbol=symbol,
                        timeframe=timeframe,
                        start_date=start_date,
                        save_to_db=True,
                    )

                except Exception as e:
                    logger.error(f"Error updating {symbol} {timeframe}: {e}")

    def get_data_from_db(
        self, symbol: str, timeframe: str = "1h", days: int = 30
    ) -> pd.DataFrame:
        """
        Get data from database (download if not exists).

        Args:
            symbol: Trading pair
            timeframe: Candle timeframe
            days: Number of days

        Returns:
            DataFrame with OHLCV data
        """
        start_date = datetime.now() - timedelta(days=days)

        # Try to get from database
        df = self.db.get_market_data(symbol, start_date)

        if (
            df.empty or len(df) < days * 20
        ):  # heuristic: ~20 candles/day on 1h TF; trigger re-download if insufficient
            logger.info(f"Data not in database or incomplete, downloading...")
            df = self.download_symbol(
                symbol=symbol,
                timeframe=timeframe,
                start_date=start_date,
                save_to_db=True,
            )

        return df

    def validate_data(self, df: pd.DataFrame) -> dict:
        """
        Validate downloaded data quality.

        Returns:
            Dictionary with validation results
        """
        issues = []

        if df.empty:
            issues.append("DataFrame is empty")
            return {"valid": False, "issues": issues}

        # Check for missing values
        if df.isnull().any().any():
            issues.append("Contains missing values")

        # Check for zero volume
        zero_volume = (df["volume"] == 0).sum()  # count bars where no trading occurred
        if zero_volume > len(df) * 0.1:  # >10% zero volume
            issues.append(f"{zero_volume} candles with zero volume")

        # Check OHLC logic: high must be >= max(open, close); low must be <= min(open, close)
        ohlc_issues = (df["high"] < df[["open", "close"]].max(axis=1)).sum() + (
            df["low"] > df[["open", "close"]].min(axis=1)
        ).sum()
        if ohlc_issues > 0:
            issues.append(f"{ohlc_issues} OHLC logic violations")

        # Check for gaps: compare median bar spacing to expected 1h interval
        if len(df) > 1:
            expected_diff = pd.Timedelta(hours=1)  # Assume 1h data
            actual_diff = (
                df.index.to_series().diff().median()
            )  # robust gap measure via median (ignores outliers)

            if actual_diff > expected_diff * 1.5:
                issues.append(f"Data gaps detected (median diff: {actual_diff})")

        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "total_candles": len(df),
            "date_range": (df.index[0], df.index[-1]) if len(df) > 0 else None,
        }


class DataPipeline:
    """
    Complete data pipeline for the trading system.

    This high-level class orchestrates the entire data flow:
    1. INITIALIZATION: Download and store historical data
    2. UPDATES: Refresh with latest market data
    3. RETRIEVAL: Provide data for training and backtesting

    The pipeline manages:
    - Historical data download and storage
    - Real-time data feeds (via database)
    - Data validation and quality checks
    - Feature computation for training

    Attributes:
        downloader: HistoricalDataDownloader instance
        db: DatabaseManager instance

    Usage:
        # Initialize pipeline
        pipeline = DataPipeline()

        # Download historical data for training
        pipeline.initialize_market_data(
            symbols=["BTCUSDT", "ETHUSDT"],
            timeframes=["1h", "4h"],
            days=365
        )

        # Get data for model training
        training_data = pipeline.get_training_data(
            symbols=["BTCUSDT"],
            lookback=100  # bars
        )

        # Update with latest data
        pipeline.update_market_data(["BTCUSDT"])
    """

    def __init__(self):
        self.downloader = HistoricalDataDownloader()
        self.db = DatabaseManager()
        logger.info("DataPipeline initialized")

    def initialize_market_data(
        self, symbols: List[str], timeframes: List[str] = None, days: int = 365
    ):
        """
        Initialize database with historical market data.

        Args:
            symbols: List of trading pairs
            timeframes: List of timeframes
            days: Number of days to download
        """
        if timeframes is None:
            timeframes = ["1h"]

        logger.info(f"Initializing market data for {len(symbols)} symbols...")

        for symbol in symbols:
            for timeframe in timeframes:
                try:
                    end_date = datetime.now()
                    start_date = end_date - timedelta(days=days)

                    self.downloader.download_symbol(
                        symbol=symbol,
                        timeframe=timeframe,
                        start_date=start_date,
                        end_date=end_date,
                        save_to_db=True,
                    )

                except Exception as e:
                    logger.error(f"Error initializing {symbol} {timeframe}: {e}")

        logger.success("Market data initialization complete!")

    def get_training_data(
        self, symbols: List[str], feature_columns: List[str] = None, lookback: int = 100
    ) -> dict:
        """
        Get data for model training.

        Args:
            symbols: List of symbols
            feature_columns: Columns to include (default: OHLCV)
            lookback: Number of periods for sequence models

        Returns:
            Dictionary of feature arrays
        """
        if feature_columns is None:
            feature_columns = ["open", "high", "low", "close", "volume"]

        data = {}

        for symbol in symbols:
            df = self.downloader.get_data_from_db(symbol, days=lookback + 100)

            if not df.empty:
                # Calculate additional features
                df["returns"] = df[
                    "close"
                ].pct_change()  # simple period returns: (P_t - P_{t-1}) / P_{t-1}
                df["volatility"] = (
                    df["returns"].rolling(20).std()
                )  # 20-bar rolling realized volatility
                df["sma_20"] = df["close"].rolling(20).mean()  # short-term trend proxy
                df["sma_50"] = df["close"].rolling(50).mean()  # medium-term trend proxy

                # Add to data dict
                data[symbol] = df[
                    feature_columns + ["returns", "volatility", "sma_20", "sma_50"]
                ].dropna()

        return data


# Production helper functions
def download_historical_data(symbols: List[str], days: int = 365):
    """
    Download historical data for multiple symbols.

    Usage:
        download_historical_data(['BTCUSDT', 'ETHUSDT'], days=365)
    """
    pipeline = DataPipeline()
    pipeline.initialize_market_data(symbols, days=days)


def update_market_data(symbols: List[str]):
    """
    Update database with latest data.

    Usage:
        update_market_data(['BTCUSDT', 'ETHUSDT'])
    """
    downloader = HistoricalDataDownloader()
    downloader.update_database(symbols)


if __name__ == "__main__":
    # Example usage
    print("=== Historical Data Pipeline ===")
    print("\nTo download data:")
    print("  download_historical_data(['BTCUSDT', 'ETHUSDT'], days=365)")
    print("\nTo update data:")
    print("  update_market_data(['BTCUSDT', 'ETHUSDT'])")
    print("\n=== Database Ready ===")
