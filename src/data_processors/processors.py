"""
Data Processors for Multiple Data Sources
==========================================
Process data from various sources into standardized format.

Supported sources:
- Yahoo Finance (yfinance)
- Binance (CCXT)
- Alpha Vantage
- Quandl
- Local CSV files
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod
import yfinance as yf


@dataclass
class DataProcessorConfig:
    """Configuration for data processor."""

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
    """Abstract base class for data processors."""

    def __init__(self, config: DataProcessorConfig):
        self.config = config

    @abstractmethod
    def download_data(self, tickers: List[str], **kwargs) -> pd.DataFrame:
        """Download data from source."""
        pass

    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to dataframe."""
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
        """Add covariance matrix as feature."""
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

        tickers = self._get_unique_tickers(df)
        window = self.config.covariance_window

        # Calculate returns for each ticker
        returns_cols = []
        for ticker in tickers:
            close_col = f"close_{ticker}" if "close_0" in df.columns else "close"
            returns_col = f"returns_{ticker}"
            df[returns_col] = df[close_col].pct_change()
            returns_cols.append(returns_col)

        # Rolling covariance (for each day, use past window)
        for i in range(len(df)):
            if i >= window:
                window_data = df[returns_cols].iloc[i - window : i].dropna()
                if len(window_data) > 1:
                    cov_matrix = window_data.cov()

                    # Flatten covariance matrix and add to dataframe
                    for j, ticker1 in enumerate(tickers):
                        for k, ticker2 in enumerate(tickers):
                            col_name = f"cov_{ticker1}_{ticker2}"
                            if col_name not in df.columns:
                                df[col_name] = np.nan
                            df.loc[i, col_name] = cov_matrix.iloc[j, k]

        # Drop intermediate returns columns
        df = df.drop(columns=returns_cols, errors="ignore")

        return df

    def normalize_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize data."""
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
        """Complete data processing pipeline."""
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
        """Extract unique tickers from dataframe columns."""
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
        """Split data into train/val/test sets."""
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
    """Data processor for Yahoo Finance."""

    def download_data(
        self, tickers: List[str], interval: str = "1d", **kwargs
    ) -> pd.DataFrame:
        """Download data from Yahoo Finance."""
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
    """Data processor for Binance (via CCXT)."""

    def __init__(self, config: DataProcessorConfig):
        super().__init__(config)
        try:
            import ccxt

            self.exchange = ccxt.binance()
        except ImportError:
            raise ImportError("CCXT not installed. Run: pip install ccxt")

    def download_data(
        self, tickers: List[str], timeframe: str = "1h", **kwargs
    ) -> pd.DataFrame:
        """Download data from Binance."""
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
    """Data processor for local CSV files."""

    def download_data(self, file_paths: List[str], **kwargs) -> pd.DataFrame:
        """Load data from CSV files."""
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
        Data processor instance
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
