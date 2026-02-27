"""
Data Processors for Multiple Data Sources
=========================================
Comprehensive data processing pipeline for trading systems.

This module provides unified data acquisition and processing from various
sources, converting raw market data into features suitable for ML models.

SUPPORTED DATA SOURCES:
----------------------
1. YAHOO FINANCE (yfinance)
   - Wide range of equities, ETFs, crypto
   - Historical daily data
   - Free and easy to use

2. BINANCE (via CCXT)
   - Cryptocurrency exchange data
   - High-frequency data (1m, 5m, 1h, etc.)
   - Real-time streaming support

3. ALPHA VANTAGE
   - Fundamental data
   - Economic indicators
   - Requires API key

4. QUANDL
   - Alternative data
   - Institutional data
   - Requires subscription

5. LOCAL CSV FILES
   - Custom data sources
   - Backtesting with proprietary data

PROCESSING FEATURES:
------------------
- Technical indicator calculation (MACD, RSI, ADX, Bollinger Bands, VWAP)
- Covariance matrix computation
- Data normalization (z-score, min-max)
- Train/validation/test splitting

Usage:
    from src.data_processors.processors import (
        create_data_processor,
        DataProcessorConfig,
        YahooFinanceProcessor,
        BinanceProcessor,
        CSVProcessor
    )

    # Configure processor
    config = DataProcessorConfig(
        start_date="2020-01-01",
        use_technical_indicators=True,
        normalize=True
    )

    # Create processor
    processor = create_data_processor("yahoo", config)

    # Download and process
    data = processor.download_data(["BTC-USD", "ETH-USD"])
    processed = processor.process(data)

Author: BITCOIN4Traders Team
License: Proprietary - Internal Use Only
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod
import yfinance as yf


@dataclass
class DataProcessorConfig:
    """
    Configuration for data processor.

    Controls all aspects of data acquisition and processing.

    Attributes:
        start_date: Start date for historical data (YYYY-MM-DD)
        end_date: End date for historical data (YYYY-MM-DD or None for now)
        use_technical_indicators: Whether to calculate technical indicators
        tech_indicator_list: List of indicators to compute
        use_covariance: Whether to compute rolling covariance
        covariance_window: Window size for covariance calculation
        normalize: Whether to normalize features
        normalization_method: 'zscore' or 'minmax'

    Example:
        >>> config = DataProcessorConfig(
        ...     start_date="2020-01-01",
        ...     end_date="2024-01-01",
        ...     use_technical_indicators=True,
        ...     normalize=True
        ... )
    """

    # Time range
    start_date: str = "2010-01-01"
    end_date: Optional[str] = None

    # Technical indicators
    use_technical_indicators: bool = True
    tech_indicator_list: List[str] = None

    # Feature engineering
    use_covariance: bool = True
    covariance_window: int = 60

    # Normalization
    normalize: bool = True
    normalization_method: str = "zscore"  # zscore, minmax

    def __post_init__(self):
        if self.tech_indicator_list is None:
            self.tech_indicator_list = [
                "macd",
                "rsi",
                "cci",
                "adx",
                "bbs",
            ]  # default indicator set


class BaseDataProcessor(ABC):
    """
    Abstract base class for data processors.

    Defines the interface that all data source processors must implement.
    Provides common processing functionality.

    PROCESSING PIPELINE:
    ------------------
    1. Download raw data from source
    2. Add technical indicators
    3. Add covariance features
    4. Normalize data
    5. Split into train/val/test

    SUBCLASSES:
    ----------
    - YahooFinanceProcessor: Yahoo Finance data
    - BinanceProcessor: Binance exchange data
    - CSVProcessor: Local CSV files
    """

    def __init__(self, config: DataProcessorConfig):
        """
        Initialize processor with configuration.

        Args:
            config: DataProcessorConfig instance
        """
        self.config = config

    @abstractmethod
    def download_data(self, tickers: List[str], **kwargs) -> pd.DataFrame:
        """
        Download data from the source.

        Args:
            tickers: List of ticker symbols

        Returns:
            DataFrame with OHLCV data
        """
        pass

    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add technical indicators to dataframe.

        Calculates popular technical analysis indicators:
        - MACD: Moving Average Convergence Divergence
        - RSI: Relative Strength Index
        - ADX: Average Directional Index
        - Bollinger Bands: Price volatility bands
        - VWAP: Volume Weighted Average Price

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with additional indicator columns
        """
        from ta.trend import MACD, ADXIndicator
        from ta.momentum import RSIIndicator
        from ta.volatility import BollingerBands
        from ta.volume import VolumeWeightedAveragePrice

        for ticker in self._get_unique_tickers(df):
            # Get price columns for this ticker
            close_col = f"close_{ticker}" if "close_0" in df.columns else "close"
            high_col = f"high_{ticker}" if "high_0" in df.columns else "high"
            low_col = f"low_{ticker}" if "low_0" in df.columns else "low"
            open_col = f"open_{ticker}" if "open_0" in df.columns else "open"
            volume_col = f"volume_{ticker}" if "volume_0" in df.columns else "volume"

            # MACD
            if "macd" in self.config.tech_indicator_list:
                macd = MACD(close=df[close_col])
                df[f"macd_{ticker}"] = macd.macd()
                df[f"macd_signal_{ticker}"] = macd.macd_signal()
                df[f"macd_diff_{ticker}"] = macd.macd_diff()

            # RSI
            if "rsi" in self.config.tech_indicator_list:
                rsi = RSIIndicator(close=df[close_col])
                df[f"rsi_{ticker}"] = rsi.rsi()

            # ADX
            if "adx" in self.config.tech_indicator_list:
                adx = ADXIndicator(
                    high=df[high_col], low=df[low_col], close=df[close_col]
                )
                df[f"adx_{ticker}"] = adx.adx()

            # Bollinger Bands
            if "bbs" in self.config.tech_indicator_list:
                bb = BollingerBands(close=df[close_col])
                df[f"bb_high_{ticker}"] = bb.bollinger_hband()
                df[f"bb_low_{ticker}"] = bb.bollinger_lband()
                df[f"bb_width_{ticker}"] = bb.bollinger_wband()

            # VWAP
            if "vwap" in self.config.tech_indicator_list:
                vwap = VolumeWeightedAveragePrice(
                    high=df[high_col],
                    low=df[low_col],
                    close=df[close_col],
                    volume=df[volume_col],
                )
                df[f"vwap_{ticker}"] = vwap.volume_weighted_average_price()

        return df

    def add_covariance_matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add rolling covariance matrix as features.

        Computes pairwise covariances between asset returns over
        a rolling window, capturing cross-asset relationships.

        Args:
            df: DataFrame with price data

        Returns:
            DataFrame with covariance features
        """
        if not self.config.use_covariance:
            return df

        tickers = self._get_unique_tickers(df)
        window = self.config.covariance_window

        # Calculate returns for each ticker
        returns_cols = []
        for ticker in tickers:
            close_col = f"close_{ticker}" if "close_0" in df.columns else "close"
            returns_col = f"returns_{ticker}"
            df[returns_col] = df[close_col].pct_change()  # simple percentage returns
            returns_cols.append(returns_col)

        # Rolling covariance (for each day, use past window)
        for i in range(len(df)):
            if i >= window:
                window_data = df[returns_cols].iloc[i - window : i].dropna()
                if len(window_data) > 1:
                    cov_matrix = (
                        window_data.cov()
                    )  # n×n covariance matrix across tickers

                    # Flatten covariance matrix and add to dataframe (one col per pair)
                    for j, ticker1 in enumerate(tickers):
                        for k, ticker2 in enumerate(tickers):
                            col_name = f"cov_{ticker1}_{ticker2}"
                            if col_name not in df.columns:
                                df[col_name] = (
                                    np.nan
                                )  # initialise column with NaN for early rows
                            df.loc[i, col_name] = cov_matrix.iloc[j, k]

        # Drop intermediate returns columns
        df = df.drop(
            columns=returns_cols, errors="ignore"
        )  # keep only cov features, not raw returns

        return df

    def normalize_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize data using specified method.

        Normalization methods:
        - Z-SCORE: (x - mean) / std (standardizes to ~N(0,1))
        - MIN-MAX: (x - min) / (max - min) (scales to [0,1])

        Args:
            df: DataFrame with numeric columns

        Returns:
            Normalized DataFrame
        """
        if not self.config.normalize:
            return df

        numeric_cols = df.select_dtypes(
            include=[np.number]
        ).columns  # only numeric columns

        if self.config.normalization_method == "zscore":
            for col in numeric_cols:
                mean = df[col].mean()
                std = df[col].std()
                if std > 0:
                    df[col] = (
                        df[col] - mean
                    ) / std  # standardise to zero mean, unit variance
        elif self.config.normalization_method == "minmax":
            for col in numeric_cols:
                min_val = df[col].min()
                max_val = df[col].max()
                if max_val > min_val:
                    df[col] = (df[col] - min_val) / (
                        max_val - min_val
                    )  # scale to [0, 1]

        return df

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Complete data processing pipeline.

        Applies all configured processing steps in order:
        1. Technical indicators
        2. Covariance features
        3. Drop NaN rows
        4. Normalization

        Args:
            df: Raw OHLCV data

        Returns:
            Fully processed DataFrame
        """
        # Add technical indicators
        if self.config.use_technical_indicators:
            df = self.add_technical_indicators(df)

        # Add covariance matrix
        if self.config.use_covariance:
            df = self.add_covariance_matrix(df)

        # Drop NaN values
        df = df.dropna()

        # Normalize
        df = self.normalize_data(df)

        return df

    def _get_unique_tickers(self, df: pd.DataFrame) -> List[str]:
        """
        Extract unique tickers from dataframe columns.

        Handles both single-ticker and multi-ticker data formats.

        Args:
            df: DataFrame with price data

        Returns:
            List of ticker identifiers
        """
        # Check if using multi-ticker format
        if "close_0" in df.columns:
            # Extract ticker indices
            tickers = []
            for col in df.columns:
                if col.startswith("close_"):
                    tickers.append(col.split("_")[1])
            return tickers
        else:
            # Single ticker
            return ["0"]

    def split_data(
        self, df: pd.DataFrame, train_ratio: float = 0.7, val_ratio: float = 0.15
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data into train/validation/test sets.

        Uses temporal splitting (no shuffling) to preserve
        chronological order for time series data.

        Args:
            df: Processed DataFrame
            train_ratio: Fraction for training (default: 70%)
            val_ratio: Fraction for validation (default: 15%)

        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        n = len(df)
        train_end = int(n * train_ratio)
        val_end = int(
            n * (train_ratio + val_ratio)
        )  # val starts after train, test gets the remainder

        # Sequential (non-shuffled) split to preserve temporal order
        train_df = df.iloc[:train_end].reset_index(drop=True)
        val_df = df.iloc[train_end:val_end].reset_index(drop=True)
        test_df = df.iloc[val_end:].reset_index(
            drop=True
        )  # remaining ~15% as hold-out test

        return train_df, val_df, test_df


class YahooFinanceProcessor(BaseDataProcessor):
    """
    Data processor for Yahoo Finance.

    Downloads historical data using the yfinance library.

    FEATURES:
    --------
    - Wide asset coverage (stocks, ETFs, crypto, forex)
    - Daily, weekly, monthly data
    - Fundamental data available
    - Free and requires no API key

    LIMITATIONS:
    -----------
    - Delayed data for some instruments
    - Rate limiting
    - Not suitable for real-time trading

    Example:
        >>> processor = YahooFinanceProcessor(config)
        >>> data = processor.download_data(["BTC-USD", "ETH-USD"], interval="1d")
    """

    def download_data(
        self, tickers: List[str], interval: str = "1d", **kwargs
    ) -> pd.DataFrame:
        """
        Download data from Yahoo Finance.

        Args:
            tickers: List of Yahoo Finance ticker symbols
            interval: Data frequency ('1d', '1wk', '1mo', etc.)
            **kwargs: Additional arguments for yfinance

        Returns:
            DataFrame with OHLCV data for all tickers
        """
        all_data = []

        for i, ticker in enumerate(tickers):
            print(f"Downloading {ticker}...")

            # Download using yfinance
            data = yf.download(
                ticker,
                start=self.config.start_date,
                end=self.config.end_date,
                interval=interval,
                progress=False,
            )

            # Rename columns
            data.columns = [f"{col.lower()}_{i}" for col in data.columns]
            data = data.rename(
                columns={
                    f"open_{i}": f"open_{i}",
                    f"high_{i}": f"high_{i}",
                    f"low_{i}": f"low_{i}",
                    f"close_{i}": f"close_{i}",
                    f"adj close_{i}": f"adj_close_{i}",
                    f"volume_{i}": f"volume_{i}",
                }
            )

            all_data.append(data)

        # Merge all tickers
        df = pd.concat(all_data, axis=1)

        # Add date column
        df["date"] = df.index
        df = df.reset_index(drop=True)

        return df


class BinanceProcessor(BaseDataProcessor):
    """
    Data processor for Binance (via CCXT).

    Downloads cryptocurrency market data from Binance exchange.

    FEATURES:
    --------
    - High-frequency data (1m, 5m, 15m, 1h, etc.)
    - Real-time data
    - Wide crypto coverage
    - Low latency

    LIMITATIONS:
    -----------
    - Requires CCXT library
    - Only crypto assets
    - API rate limits

    Example:
        >>> processor = BinanceProcessor(config)
        >>> data = processor.download_data(["BTC/USDT", "ETH/USDT"], timeframe="1h")
    """

    def __init__(self, config: DataProcessorConfig):
        """
        Initialize Binance processor.

        Args:
            config: DataProcessorConfig

        Raises:
            ImportError: If CCXT not installed
        """
        super().__init__(config)
        try:
            import ccxt

            self.exchange = ccxt.binance()
        except ImportError:
            raise ImportError("CCXT not installed. Run: pip install ccxt")

    def download_data(
        self, tickers: List[str], timeframe: str = "1h", **kwargs
    ) -> pd.DataFrame:
        """
        Download data from Binance.

        Args:
            tickers: List of Binance ticker symbols (e.g., 'BTC/USDT')
            timeframe: OHLCV timeframe ('1m', '5m', '1h', '1d')
            **kwargs: Additional arguments

        Returns:
            DataFrame with OHLCV data
        """
        all_data = []

        for i, ticker in enumerate(tickers):
            print(f"Downloading {ticker}...")

            # Fetch OHLCV data
            since = self.exchange.parse8601(f"{self.config.start_date}T00:00:00Z")

            ohlcv = self.exchange.fetch_ohlcv(
                ticker, timeframe=timeframe, since=since, limit=1000
            )

            # Convert to DataFrame
            df = pd.DataFrame(
                ohlcv,
                columns=[
                    "timestamp",
                    f"open_{i}",
                    f"high_{i}",
                    f"low_{i}",
                    f"close_{i}",
                    f"volume_{i}",
                ],
            )

            # Convert timestamp to datetime
            df["date"] = pd.to_datetime(df["timestamp"], unit="ms")
            df = df.drop("timestamp", axis=1)

            all_data.append(df)

        # Merge on date
        df = all_data[0]
        for i in range(1, len(all_data)):
            df = pd.merge(df, all_data[i], on="date", how="inner")

        return df.reset_index(drop=True)


class CSVProcessor(BaseDataProcessor):
    """
    Data processor for local CSV files.

    Loads custom data from CSV files for backtesting
    with proprietary or alternative data sources.

    FEATURES:
    --------
    - Custom data formats
    - No external dependencies
    - Full control over data

    LIMITATIONS:
    -----------
    - Manual data preparation required
    - No automatic updates

    Example:
        >>> processor = CSVProcessor(config)
        >>> data = processor.download_data(["data/btc.csv", "data/eth.csv"])
    """

    def download_data(self, file_paths: List[str], **kwargs) -> pd.DataFrame:
        """
        Load data from CSV files.

        Args:
            file_paths: List of paths to CSV files
            **kwargs: Additional arguments for pd.read_csv

        Returns:
            DataFrame with loaded data
        """
        all_data = []

        for i, file_path in enumerate(file_paths):
            print(f"Loading {file_path}...")

            df = pd.read_csv(file_path)

            # Rename columns
            df.columns = [f"{col.lower()}_{i}" for col in df.columns]

            all_data.append(df)

        # Merge
        df = pd.concat(all_data, axis=1)

        return df


# Factory function
def create_data_processor(
    source: str, config: DataProcessorConfig
) -> BaseDataProcessor:
    """
    Factory function to create data processor.

    Args:
        source: Data source ('yahoo', 'binance', 'csv')
        config: Processor configuration

    Returns:
        Data processor instance for the specified source

    Raises:
        ValueError: If unknown source specified

    Example:
        >>> config = DataProcessorConfig(start_date="2020-01-01")
        >>> processor = create_data_processor("yahoo", config)
    """
    source = source.lower()

    if source == "yahoo":
        return YahooFinanceProcessor(config)
    elif source == "binance":
        return BinanceProcessor(config)
    elif source == "csv":
        return CSVProcessor(config)
    else:
        raise ValueError(f"Unknown data source: {source}")
