"""
Data Manager - CCXT Integration & Parquet Caching
==================================================
Professional data loading with:
- CCXT exchange integration
- Parquet format for efficient storage
- Automatic caching
- Data validation & quality checks
- Rate limit handling
"""

import ccxt
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Tuple
from datetime import datetime, timedelta
from loguru import logger
import time
from dataclasses import dataclass


@dataclass
class DataConfig:
    """Configuration for data management."""
    exchange_id: str = "binance"
    symbols: List[str] = None
    timeframe: str = "1h"
    start_date: str = "2020-01-01"
    end_date: Optional[str] = None
    cache_dir: Path = Path("data/cache")
    processed_dir: Path = Path("data/processed")
    rate_limit_delay: float = 0.1  # seconds between requests
    max_retries: int = 3
    
    def __post_init__(self):
        if self.symbols is None:
            self.symbols = ["BTC/USDT", "ETH/USDT"]
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)


class DataManager:
    """
    Professional data management with CCXT and Parquet caching.
    
    Features:
    - Loads OHLCV data from multiple exchanges
    - Caches data locally in Parquet format
    - Validates data quality
    - Handles rate limits and network errors
    - Incremental updates
    """
    
    def __init__(self, config: DataConfig):
        self.config = config
        self.exchange = self._init_exchange()
        
        logger.info(f"DataManager initialized: {config.exchange_id}")
        logger.info(f"Symbols: {config.symbols}")
        logger.info(f"Timeframe: {config.timeframe}")
        
    def _init_exchange(self) -> ccxt.Exchange:
        """Initialize CCXT exchange connection."""
        try:
            exchange_class = getattr(ccxt, self.config.exchange_id)
            exchange = exchange_class({
                'enableRateLimit': True,
                'rateLimit': int(self.config.rate_limit_delay * 1000),
                'options': {
                    'defaultType': 'spot',
                }
            })
            
            # Test connection
            exchange.load_markets()
            logger.success(f"Connected to {self.config.exchange_id}")
            
            return exchange
            
        except Exception as e:
            logger.error(f"Failed to connect to {self.config.exchange_id}: {e}")
            raise
    
    def fetch_ohlcv(
        self,
        symbol: str,
        since: Optional[datetime] = None,
        limit: int = 1000,
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data for a symbol.
        
        Parameters:
        -----------
        symbol : str
            Trading pair (e.g., 'BTC/USDT')
        since : datetime, optional
            Start date for data
        limit : int
            Max number of candles per request
        use_cache : bool
            Whether to use cached data
            
        Returns:
        --------
        df : pd.DataFrame
            OHLCV data with columns: [open, high, low, close, volume]
        """
        # Check cache first
        if use_cache:
            cached_df = self._load_from_cache(symbol)
            if cached_df is not None:
                logger.info(f"Loaded {len(cached_df)} rows from cache: {symbol}")
                
                # Check if we need to update
                if since is None or cached_df.index.max() >= pd.Timestamp(since):
                    return cached_df
                else:
                    logger.info("Cache outdated, fetching new data...")
        
        # Fetch from exchange
        logger.info(f"Fetching {symbol} from {self.config.exchange_id}...")
        
        since_ms = int(since.timestamp() * 1000) if since else None
        all_data = []
        
        retries = 0
        while retries < self.config.max_retries:
            try:
                ohlcv = self.exchange.fetch_ohlcv(
                    symbol=symbol,
                    timeframe=self.config.timeframe,
                    since=since_ms,
                    limit=limit
                )
                
                if not ohlcv:
                    break
                
                all_data.extend(ohlcv)
                
                # Update since for next batch
                since_ms = ohlcv[-1][0] + 1
                
                # Check if we have all data
                if len(ohlcv) < limit:
                    break
                
                # Rate limiting
                time.sleep(self.config.rate_limit_delay)
                
            except ccxt.NetworkError as e:
                retries += 1
                logger.warning(f"Network error ({retries}/{self.config.max_retries}): {e}")
                time.sleep(2 ** retries)  # Exponential backoff
                
            except ccxt.ExchangeError as e:
                logger.error(f"Exchange error: {e}")
                break
        
        if not all_data:
            logger.warning(f"No data fetched for {symbol}")
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame(
            all_data,
            columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
        )
        
        # Set timestamp as index
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        df.sort_index(inplace=True)
        
        # Remove duplicates
        df = df[~df.index.duplicated(keep='last')]
        
        logger.success(f"Fetched {len(df)} candles for {symbol}")
        
        # Cache data
        self._save_to_cache(symbol, df)
        
        return df
    
    def _load_from_cache(self, symbol: str) -> Optional[pd.DataFrame]:
        """Load data from Parquet cache."""
        cache_file = self._get_cache_path(symbol)
        
        if cache_file.exists():
            try:
                df = pd.read_parquet(cache_file)
                return df
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")
                return None
        
        return None
    
    def _save_to_cache(self, symbol: str, df: pd.DataFrame):
        """Save data to Parquet cache."""
        cache_file = self._get_cache_path(symbol)
        
        try:
            df.to_parquet(
                cache_file,
                engine='pyarrow',
                compression='snappy'
            )
            logger.debug(f"Cached {len(df)} rows to {cache_file}")
        except Exception as e:
            logger.error(f"Failed to save cache: {e}")
    
    def _get_cache_path(self, symbol: str) -> Path:
        """Get cache file path for symbol."""
        safe_symbol = symbol.replace('/', '_')
        filename = f"{self.config.exchange_id}_{safe_symbol}_{self.config.timeframe}.parquet"
        return self.config.cache_dir / filename
    
    def load_multiple_symbols(
        self,
        symbols: Optional[List[str]] = None,
        start_date: Optional[str] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Load data for multiple symbols.
        
        Parameters:
        -----------
        symbols : List[str], optional
            List of symbols (default: from config)
        start_date : str, optional
            Start date YYYY-MM-DD
            
        Returns:
        --------
        data : Dict[str, pd.DataFrame]
            Dictionary mapping symbol -> DataFrame
        """
        if symbols is None:
            symbols = self.config.symbols
        
        since = pd.to_datetime(start_date or self.config.start_date)
        
        data = {}
        for symbol in symbols:
            try:
                df = self.fetch_ohlcv(symbol, since=since)
                if not df.empty:
                    data[symbol] = df
            except Exception as e:
                logger.error(f"Failed to load {symbol}: {e}")
        
        return data
    
    def validate_data(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Validate data quality.
        
        Returns:
        --------
        is_valid : bool
        issues : List[str]
            List of data quality issues
        """
        issues = []
        
        # Check for missing values
        if df.isnull().any().any():
            null_cols = df.columns[df.isnull().any()].tolist()
            issues.append(f"Missing values in columns: {null_cols}")
        
        # Check for negative prices
        price_cols = ['open', 'high', 'low', 'close']
        for col in price_cols:
            if (df[col] <= 0).any():
                issues.append(f"Negative or zero values in {col}")
        
        # Check high >= low
        if (df['high'] < df['low']).any():
            issues.append("High < Low inconsistency detected")
        
        # Check close within [low, high]
        if ((df['close'] < df['low']) | (df['close'] > df['high'])).any():
            issues.append("Close price outside [low, high] range")
        
        # Check for duplicate timestamps
        if df.index.duplicated().any():
            issues.append(f"Duplicate timestamps: {df.index.duplicated().sum()}")
        
        # Check for gaps in timeline
        expected_freq = pd.Timedelta(self.config.timeframe)
        actual_diffs = df.index.to_series().diff()
        gaps = actual_diffs[actual_diffs > expected_freq * 1.5]
        if len(gaps) > 0:
            issues.append(f"Timeline gaps detected: {len(gaps)} instances")
        
        is_valid = len(issues) == 0
        
        if is_valid:
            logger.success("Data validation passed")
        else:
            logger.warning(f"Data validation found {len(issues)} issues:")
            for issue in issues:
                logger.warning(f"  - {issue}")
        
        return is_valid, issues
    
    def get_data_summary(self, df: pd.DataFrame) -> Dict:
        """Get summary statistics for data."""
        return {
            'rows': len(df),
            'start_date': df.index.min(),
            'end_date': df.index.max(),
            'duration_days': (df.index.max() - df.index.min()).days,
            'null_values': df.isnull().sum().to_dict(),
            'price_range': {
                'min': df['close'].min(),
                'max': df['close'].max(),
                'mean': df['close'].mean(),
                'std': df['close'].std()
            },
            'volume_stats': {
                'mean': df['volume'].mean(),
                'total': df['volume'].sum()
            }
        }


# ============================================================================
# FEATURE ENGINEERING PIPELINE
# ============================================================================

class FeatureEngine:
    """
    Feature engineering for trading signals.
    
    Computes:
    - Log returns
    - Rolling volatility (20, 50 windows)
    - Technical indicators
    - Regime features
    """
    
    def __init__(self, volatility_windows: List[int] = None):
        self.volatility_windows = volatility_windows or [20, 50]
        logger.info(f"FeatureEngine initialized with windows: {self.volatility_windows}")
    
    def compute_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute log returns."""
        df = df.copy()
        df['returns'] = np.log(df['close'] / df['close'].shift(1))
        return df
    
    def compute_volatility(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute rolling volatility (annualized)."""
        df = df.copy()
        
        if 'returns' not in df.columns:
            df = self.compute_returns(df)
        
        for window in self.volatility_windows:
            col_name = f'volatility_{window}'
            # Annualized volatility (assuming 24 hours per day for crypto)
            df[col_name] = df['returns'].rolling(window=window).std() * np.sqrt(365 * 24)
        
        return df
    
    def compute_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute basic technical indicators."""
        df = df.copy()
        
        # Simple Moving Averages
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()
        
        # Exponential Moving Averages
        df['ema_12'] = df['close'].ewm(span=12, adjust=False).mean()
        df['ema_26'] = df['close'].ewm(span=26, adjust=False).mean()
        
        # MACD
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        
        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        bb_std = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + 2 * bb_std
        df['bb_lower'] = df['bb_middle'] - 2 * bb_std
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        return df
    
    def compute_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute all features."""
        logger.info("Computing features...")
        
        df = self.compute_returns(df)
        df = self.compute_volatility(df)
        df = self.compute_technical_indicators(df)
        
        # Drop NaN rows from rolling calculations
        initial_rows = len(df)
        df = df.dropna()
        dropped_rows = initial_rows - len(df)
        
        if dropped_rows > 0:
            logger.info(f"Dropped {dropped_rows} rows due to NaN values")
        
        logger.success(f"Features computed: {len(df)} rows, {len(df.columns)} columns")
        
        return df
    
    def save_features(self, df: pd.DataFrame, symbol: str, output_dir: Path):
        """Save processed features to Parquet."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        safe_symbol = symbol.replace('/', '_')
        output_file = output_dir / f"{safe_symbol}_features.parquet"
        
        df.to_parquet(
            output_file,
            engine='pyarrow',
            compression='snappy'
        )
        
        logger.success(f"Saved features to {output_file}")
        return output_file


# ============================================================================
# EXAMPLE USAGE & TESTING
# ============================================================================

if __name__ == "__main__":
    # Configure logger
    logger.add("logs/data_manager_{time}.log", rotation="1 day")
    
    print("="*80)
    print("DATA MANAGER - PHASE 1 TEST")
    print("="*80)
    
    # Configuration
    config = DataConfig(
        exchange_id="binance",
        symbols=["BTC/USDT", "ETH/USDT"],
        timeframe="1h",
        start_date="2023-01-01",
        cache_dir=Path("data/cache"),
        processed_dir=Path("data/processed")
    )
    
    # Initialize DataManager
    dm = DataManager(config)
    
    # Test 1: Fetch single symbol
    print("\n[TEST 1] Fetching BTC/USDT...")
    btc_data = dm.fetch_ohlcv(
        symbol="BTC/USDT",
        since=pd.to_datetime("2023-01-01"),
        use_cache=True
    )
    
    print(f"✓ Loaded {len(btc_data)} rows")
    print(f"  Date range: {btc_data.index.min()} to {btc_data.index.max()}")
    print(f"  Columns: {btc_data.columns.tolist()}")
    
    # Test 2: Data validation
    print("\n[TEST 2] Validating data quality...")
    is_valid, issues = dm.validate_data(btc_data)
    
    if is_valid:
        print("✓ Data validation passed")
    else:
        print(f"⚠ Found {len(issues)} issues")
    
    # Test 3: Data summary
    print("\n[TEST 3] Data summary...")
    summary = dm.get_data_summary(btc_data)
    print(f"  Rows: {summary['rows']}")
    print(f"  Duration: {summary['duration_days']} days")
    print(f"  Price range: ${summary['price_range']['min']:.2f} - ${summary['price_range']['max']:.2f}")
    
    # Test 4: Feature engineering
    print("\n[TEST 4] Computing features...")
    fe = FeatureEngine(volatility_windows=[20, 50])
    btc_features = fe.compute_all_features(btc_data)
    
    print(f"✓ Features computed: {btc_features.shape[1]} columns")
    print(f"  Columns: {btc_features.columns.tolist()}")
    
    # Save features
    output_file = fe.save_features(
        btc_features,
        symbol="BTC/USDT",
        output_dir=Path("data/processed")
    )
    print(f"✓ Saved to {output_file}")
    
    # Test 5: Load multiple symbols
    print("\n[TEST 5] Loading multiple symbols...")
    all_data = dm.load_multiple_symbols(
        symbols=["BTC/USDT", "ETH/USDT"],
        start_date="2023-01-01"
    )
    
    print(f"✓ Loaded {len(all_data)} symbols:")
    for symbol, df in all_data.items():
        print(f"  {symbol}: {len(df)} rows")
    
    print("\n" + "="*80)
    print("✓ PHASE 1 COMPLETE - DATA INFRASTRUCTURE READY")
    print("="*80)
    
    print("\nNext steps:")
    print("1. Verify data in data/cache/ (Parquet files)")
    print("2. Check features in data/processed/")
    print("3. Review logs in logs/")
    print("\nReady for PHASE 2: Market Simulation (Gym Environment)")
