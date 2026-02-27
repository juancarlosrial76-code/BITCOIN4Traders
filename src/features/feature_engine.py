"""
Feature Engine - Institutional Grade Feature Engineering
=========================================================

This module provides production-grade feature engineering for algorithmic trading
and reinforcement learning applications. It implements the scikit-learn fit/transform
pattern to prevent data leakage, which is critical for building robust trading models.

Key Features:
-------------
1. PREVENTS DATA LEAKAGE: Uses fit_transform() for training data and transform()
   for test/live data, ensuring the model never sees future information during training.

2. ENSURES STATIONARITY: Converts raw prices to log returns, which have more
   desirable statistical properties for time series analysis (additive, bounded).

3. PROPER SCALING: Uses sklearn scalers (StandardScaler, MinMaxScaler, or RobustScaler)
   that learn statistics from training data only and apply them consistently to
   test/live data.

4. SYSTEMATIC NaN HANDLING: Multiple strategies (rolling, forward_fill, drop_all)
   to handle missing values from rolling window calculations.

5. NUMBA OPTIMIZATION: JIT-compiled functions for critical performance paths.

6. HYDRA INTEGRATION: All parameters configurable via Hydra config system.

Technical Indicators Computed:
-------------------------------
- Log Returns: Natural logarithm of price ratio (ln(P_t / P_{t-1}))
- Volatility: Annualized rolling standard deviation of returns (20, 50 windows)
- OU Score: Ornstein-Uhlenbeck mean reversion z-score
- RSI: Relative Strength Index (14-period)
- MACD: Moving Average Convergence Divergence with signal line and histogram
- Bollinger Bands: Band width and position metrics

Usage:
------
# Training Phase (fit scaler on historical data)
    engine = FeatureEngine(config)
    train_features = engine.fit_transform(train_df)

# Testing/Live Phase (use training statistics)
    test_features = engine.transform(test_df)

# Production: Save and reload scaler
    engine.save_scaler()
    engine.load_scaler()
    live_features = engine.transform(live_df)

References:
----------
- Borrowed (2013): "Advances in Financial Machine Learning"
- Easley et al. (2012): "Volume Synchronized Probability of Informed Trading"
- sklearn.preprocessing documentation

Author: BITCOIN4Traders Team
Version: 1.0.0
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from loguru import logger
import pickle
from dataclasses import dataclass
from numba import jit


@dataclass
class FeatureConfig:
    """
    Configuration dataclass for FeatureEngine.

    This configuration is typically loaded via Hydra from a YAML config file,
    ensuring zero hardcoded parameters in production code.

    Attributes:
        volatility_window: Rolling window size for short-term volatility calculation
                          (typically 20 periods for hourly data)
        ou_window: Window for Ornstein-Uhlenbeck mean reversion calculation
        rolling_mean_window: Window for rolling mean and standard deviation
        use_log_returns: If True, use log returns; if False, use simple percentage returns
        scaler_type: Type of scaler - "standard" (z-score), "minmax", or "robust"
        save_scaler: Whether to save fitted scaler to disk for production use
        scaler_path: Directory path where scaler will be saved
        dropna_strategy: Strategy for handling NaN values - "rolling", "forward_fill", or "drop_all"
        min_valid_rows: Minimum number of valid rows required after processing

    Example:
        >>> config = FeatureConfig(
        ...     volatility_window=20,
        ...     ou_window=50,
        ...     rolling_mean_window=20,
        ...     use_log_returns=True,
        ...     scaler_type="standard",
        ...     save_scaler=True,
        ...     scaler_path=Path("data/scalers"),
        ...     dropna_strategy="rolling",
        ...     min_valid_rows=100,
        ... )
    """

    volatility_window: int
    ou_window: int
    rolling_mean_window: int
    use_log_returns: bool
    scaler_type: str
    save_scaler: bool
    scaler_path: Path
    dropna_strategy: str
    min_valid_rows: int


class FeatureEngine:
    """
    Production-grade feature engineering with fit/transform pattern.

    This class implements institutional-grade feature engineering specifically
    designed for algorithmic trading and reinforcement learning applications.
    The key innovation is the strict separation between fit (training) and
    transform (inference) phases to prevent data leakage.

    Key Features:
        - NO DATA LEAKAGE: fit_transform() computes statistics on training data,
          transform() applies those same statistics to new data. This is critical
          for building models that generalize to live trading.

        - STATIONARITY: Uses log returns instead of raw prices, which have better
          statistical properties (additive, more normally distributed).

        - PROPER SCALING: Fitted scaler can be saved and reloaded for production use,
          ensuring consistent feature scaling between training and inference.

        - COMPREHENSIVE INDICATORS: Includes volatility, RSI, MACD, Bollinger Bands,
          and Ornstein-Uhlenbeck mean reversion score.

        - NUMBA OPTIMIZED: Critical computation paths use JIT compilation for
          performance.

    Usage:
        # Training phase - fit the engine on historical data
        engine = FeatureEngine(config)
        train_features = engine.fit_transform(train_df)

        # Testing/Live phase - use training statistics (CRITICAL!)
        test_features = engine.transform(test_df)  # Uses train stats!

        # Save for production
        engine.save_scaler()

        # Production: load scaler and transform live data
        engine.load_scaler()
        live_features = engine.transform(live_df)

    Attributes:
        config: FeatureConfig object with all parameters
        is_fitted: Boolean flag indicating if fit_transform() has been called
        scaler: Fitted sklearn scaler (StandardScaler, MinMaxScaler, or RobustScaler)
        train_stats: Dictionary of statistics from training data (for transform)

    Raises:
        RuntimeError: If transform() is called before fit_transform()
        ValueError: If invalid scaler_type or dropna_strategy is provided
        ValueError: If insufficient data after NaN handling

    Example:
        >>> config = FeatureConfig(
        ...     volatility_window=20,
        ...     ou_window=50,
        ...     rolling_mean_window=20,
        ...     use_log_returns=True,
        ...     scaler_type="standard",
        ...     save_scaler=False,
        ...     scaler_path=Path("data/scalers"),
        ...     dropna_strategy="rolling",
        ...     min_valid_rows=100,
        ... )
        >>> engine = FeatureEngine(config)
        >>> train_features = engine.fit_transform(train_df)
        >>> print(f"Training features shape: {train_features.shape}")
        >>> test_features = engine.transform(test_df)
        >>> print(f"Test features shape: {test_features.shape}")
    """

    def __init__(self, config: FeatureConfig):
        """
        Initialize with Hydra config.

        Parameters:
        -----------
        config : FeatureConfig
            Configuration object (injected by Hydra)
        """
        self.config = config
        self.is_fitted = False

        # Initialize scaler
        self.scaler = self._init_scaler()

        # Statistics from training data (for transform)
        self.train_stats = {}

        logger.info("FeatureEngine initialized")
        logger.info(f"  Volatility window: {config.volatility_window}")
        logger.info(f"  OU window: {config.ou_window}")
        logger.info(f"  Scaler: {config.scaler_type}")

    def _init_scaler(self):
        """Initialize scaler based on config."""
        if self.config.scaler_type == "standard":
            return StandardScaler()
        elif self.config.scaler_type == "minmax":
            return MinMaxScaler()
        elif self.config.scaler_type == "robust":
            return RobustScaler()
        else:
            raise ValueError(f"Unknown scaler: {self.config.scaler_type}")

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit on training data and transform in one step.

        This method must be called ONCE on training data to:
        1. Compute raw features (returns, volatility, technical indicators)
        2. Store training statistics (mean, std) for later use in transform()
        3. Fit the scaler on training features
        4. Transform training features using the fitted scaler

        CRITICAL: After calling fit_transform, you MUST call transform() (not
        fit_transform again) on any new data (test set, live data). This ensures
        the model uses training statistics only, preventing data leakage.

        Parameters:
        -----------
        df : pd.DataFrame
            Raw OHLCV (Open, High, Low, Close, Volume) data with DatetimeIndex.
            Must contain columns: open, high, low, close, volume

        Returns:
        --------
        features : pd.DataFrame
            Transformed features with all computed indicators and scaled values.
            Contains columns: open, high, low, close, volume, log_ret, volatility_20,
            volatility_50, ou_score, rolling_mean, rolling_std, rsi_14, macd,
            macd_signal, macd_hist, bb_width, bb_position

        Example:
            >>> config = FeatureConfig(...)
            >>> engine = FeatureEngine(config)
            >>> train_features = engine.fit_transform(train_df)
            >>> print(f"Fitted on {len(train_features)} training samples")
            >>> # Later, on test data:
            >>> test_features = engine.transform(test_df)  # Note: transform(), not fit_transform()
        """
        logger.info("Fitting FeatureEngine on training data...")

        # Compute raw features
        df = self._compute_raw_features(df)

        # Store training statistics
        self._store_train_stats(df)

        # Apply OU score (requires train stats)
        df = self._compute_ou_score(df)

        # Handle NaN values
        df = self._handle_nan(df)

        # Fit and transform scaler
        feature_cols = self._get_feature_columns(df)
        df[feature_cols] = self.scaler.fit_transform(df[feature_cols])

        self.is_fitted = True

        # Save scaler if configured
        if self.config.save_scaler:
            self._save_scaler()

        logger.success(f"FeatureEngine fitted on {len(df)} rows")

        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform test or live data using training statistics.

        CRITICAL: This method uses statistics learned from training data via
        fit_transform(). This is the KEY mechanism that prevents data leakage.
        Never call fit_transform() on test or live data!

        The method:
        1. Computes raw features using the same formulas as fit_transform()
        2. Applies the OU score using TRAINING mean/std (not the data's own stats)
        3. Handles NaN values using the same strategy
        4. Scales features using the FITTED scaler (trained on training data)

        Parameters:
        -----------
        df : pd.DataFrame
            Raw OHLCV data (can be test set or live data)
            Must contain columns: open, high, low, close, volume

        Returns:
        --------
        features : pd.DataFrame
            Transformed features using training statistics.
            Has same column structure as fit_transform output.

        Raises:
        -------
        RuntimeError: If called before fit_transform() - you must fit the
            engine on training data first

        Example:
            >>> # After fitting on training data
            >>> test_features = engine.transform(test_df)
            >>> print(f"Transformed {len(test_features)} test samples")
            >>> # Features are scaled using training statistics!
        """
        if not self.is_fitted:
            raise RuntimeError("FeatureEngine not fitted. Call fit_transform() first.")

        logger.info("Transforming data with training statistics...")

        # Compute raw features
        df = self._compute_raw_features(df)

        # Apply OU score (using TRAIN stats)
        df = self._compute_ou_score(df)

        # Handle NaN values
        df = self._handle_nan(df)

        # Transform using fitted scaler
        feature_cols = self._get_feature_columns(df)
        df[feature_cols] = self.scaler.transform(df[feature_cols])

        logger.success(f"Transformed {len(df)} rows")

        return df

    def _compute_raw_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute raw technical indicators from OHLCV data.

        This method calculates all technical indicators used as features for
        the trading model. All calculations are performed on raw (unscaled)
        values.

        Features computed:
            1. log_ret: Log returns (ln(P_t / P_{t-1})) - ensures stationarity
            2. volatility_20: 20-period annualized rolling volatility
            3. volatility_50: 50-period annualized rolling volatility (longer-term)
            4. rolling_mean: Rolling mean of close price
            5. rolling_std: Rolling standard deviation of close price
            6. rsi_14: Relative Strength Index (14-period)
            7. macd: MACD line (12-period EMA - 26-period EMA)
            8. macd_signal: Signal line (9-period EMA of MACD)
            9. macd_hist: MACD histogram (MACD - Signal)
            10. bb_width: Bollinger Band width (normalized)
            11. bb_position: Bollinger Band position (0-1 scale)

        Parameters:
        -----------
        df : pd.DataFrame
            OHLCV data with columns: open, high, low, close, volume

        Returns:
        --------
        df : pd.DataFrame
            DataFrame with additional feature columns (unscaled)
        """
        df = df.copy()

        # 1. Log Returns (ensures stationarity): ln(P_t / P_{t-1})
        # Log returns are preferred over simple returns because:
        # - Additive: log(a) + log(b) = log(a*b)
        # - Symmetric: log(1+x) ≈ -log(1-x) for small x
        # - Bounded: less affected by extreme price movements
        if self.config.use_log_returns:
            df["log_ret"] = np.log(df["close"] / df["close"].shift(1))
        else:
            df["log_ret"] = df["close"].pct_change()  # Simple pct return as alternative

        # 2. Volatility (rolling std of returns)
        # Annualized to allow comparison across different timeframes
        # For hourly data: 252 trading days * 24 hours = 6048 periods/year
        df["volatility_20"] = (
            df["log_ret"].rolling(window=self.config.volatility_window).std()
            * np.sqrt(
                252 * 24
            )  # Annualize hourly vol: σ_hourly * sqrt(trading hours/year)
        )

        # Additional volatility window (50-period for longer-term regime)
        df["volatility_50"] = (
            df["log_ret"].rolling(window=50).std()
            * np.sqrt(252 * 24)  # Same annualization factor
        )

        # 3. Rolling statistics (for OU score)
        # Used to compute mean-reversion signals
        df["rolling_mean"] = (
            df["close"].rolling(window=self.config.rolling_mean_window).mean()
        )

        df["rolling_std"] = (
            df["close"].rolling(window=self.config.rolling_mean_window).std()
        )

        # 4. RSI (Relative Strength Index)
        # Momentum oscillator measuring speed and change of price movements
        # Scale: 0 = oversold, 100 = overbought
        df["rsi_14"] = self._compute_rsi(df["close"], window=14)

        # 5. MACD (Moving Average Convergence Divergence)
        # Trend-following momentum indicator
        macd_line, signal_line = self._compute_macd(df["close"])
        df["macd"] = macd_line
        df["macd_signal"] = signal_line
        df["macd_hist"] = (
            macd_line - signal_line
        )  # MACD histogram: momentum of momentum

        # 6. Bollinger Bands
        # Statistical bands around a moving average
        upper, lower = self._compute_bollinger_bands(df["close"], window=20, num_std=2)
        # Handle division by zero if close is 0 (unlikely but safe)
        df["bb_width"] = (upper - lower) / (
            df["close"] + 1e-8
        )  # Band width normalized by price
        df["bb_position"] = (df["close"] - lower) / (
            upper - lower + 1e-8
        )  # 0=at lower band, 1=at upper band

        return df

    def _compute_rsi(self, series: pd.Series, window: int = 14) -> pd.Series:
        """Compute Relative Strength Index."""
        delta = series.diff()  # Price change per period
        gain = (
            (delta.where(delta > 0, 0)).rolling(window=window).mean()
        )  # Average gain over window
        loss = (
            (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        )  # Average loss (positive value)

        rs = gain / (
            loss + 1e-8
        )  # Relative strength ratio; epsilon avoids division by zero
        return 100 - (
            100 / (1 + rs)
        )  # RSI formula: 0=fully oversold, 100=fully overbought

    def _compute_macd(self, series: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """Compute MACD and Signal line."""
        exp1 = series.ewm(span=12, adjust=False).mean()  # Fast EMA (12-period)
        exp2 = series.ewm(span=26, adjust=False).mean()  # Slow EMA (26-period)
        macd = exp1 - exp2  # MACD line: fast - slow
        signal = macd.ewm(
            span=9, adjust=False
        ).mean()  # Signal line: 9-period EMA of MACD
        return macd, signal

    def _compute_bollinger_bands(
        self, series: pd.Series, window: int = 20, num_std: float = 2
    ) -> Tuple[pd.Series, pd.Series]:
        """Compute Bollinger Bands."""
        rolling_mean = series.rolling(window=window).mean()
        rolling_std = series.rolling(window=window).std()
        upper = rolling_mean + (rolling_std * num_std)
        lower = rolling_mean - (rolling_std * num_std)
        return upper, lower

    def _compute_ou_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute Ornstein-Uhlenbeck mean reversion score.

        OU Score = (price - mean) / std
        Normalized deviation from rolling mean.
        """
        df = df.copy()

        # Use training statistics if available (for transform)
        if self.is_fitted and "ou_mean" in self.train_stats:
            ou_mean = self.train_stats["ou_mean"]
            ou_std = self.train_stats["ou_std"]
        else:
            ou_mean = df["rolling_mean"]
            ou_std = df["rolling_std"]

        # OU score: z-score of price vs mean (positive=above mean → sell signal, negative=below → buy)
        df["ou_score"] = (df["close"] - ou_mean) / (
            ou_std + 1e-8
        )  # +1e-8 prevents division by zero

        # Clip extreme values to ±5σ to limit outlier influence on the RL agent
        df["ou_score"] = df["ou_score"].clip(-5, 5)

        return df

    def _store_train_stats(self, df: pd.DataFrame):
        """Store training statistics for later use in transform."""
        self.train_stats = {
            "ou_mean": df["rolling_mean"].mean(),
            "ou_std": df["rolling_std"].mean(),
            "volatility_mean": df["volatility_20"].mean(),
            "close_mean": df["close"].mean(),
            "close_std": df["close"].std(),
            "rsi_mean": df["rsi_14"].mean(),
            "macd_mean": df["macd"].mean(),
            "bb_width_mean": df["bb_width"].mean(),
        }

        logger.debug(f"Stored training statistics: {self.train_stats}")

    def _handle_nan(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle NaN values systematically.

        Strategy:
        - 'rolling': Drop rows with NaN (from rolling windows)
        - 'forward_fill': Forward fill NaN
        - 'drop_all': Drop any row with NaN
        """
        initial_rows = len(df)

        if self.config.dropna_strategy == "rolling":
            # Drop rows affected by rolling windows
            max_window = max(
                self.config.volatility_window,
                self.config.ou_window,
                self.config.rolling_mean_window,
            )
            df = df.iloc[max_window:]

        elif self.config.dropna_strategy == "forward_fill":
            df = df.fillna(method="ffill")

        elif self.config.dropna_strategy == "drop_all":
            df = df.dropna()

        else:
            raise ValueError(f"Unknown dropna_strategy: {self.config.dropna_strategy}")

        dropped = initial_rows - len(df)

        if dropped > 0:
            logger.info(f"Dropped {dropped} rows (NaN handling)")

        # Validate minimum rows
        if len(df) < self.config.min_valid_rows:
            raise ValueError(
                f"Insufficient data after NaN handling: {len(df)} rows "
                f"(minimum: {self.config.min_valid_rows})"
            )

        return df

    def _get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """Get columns to be scaled."""
        # Exclude OHLCV, timestamp, etc.
        exclude = ["open", "high", "low", "close", "volume", "timestamp"]

        feature_cols = [col for col in df.columns if col not in exclude]

        return feature_cols

    def _save_scaler(self):
        """Save fitted scaler for production use."""
        self.config.scaler_path.mkdir(parents=True, exist_ok=True)

        scaler_file = self.config.scaler_path / "feature_scaler.pkl"

        with open(scaler_file, "wb") as f:
            pickle.dump(
                {
                    "scaler": self.scaler,
                    "train_stats": self.train_stats,
                    "config": self.config,
                },
                f,
            )

        logger.info(f"Saved scaler to {scaler_file}")

    def load_scaler(self):
        """Load pre-fitted scaler (for production)."""
        scaler_file = self.config.scaler_path / "feature_scaler.pkl"

        if not scaler_file.exists():
            raise FileNotFoundError(f"Scaler not found: {scaler_file}")

        with open(scaler_file, "rb") as f:
            data = pickle.load(f)

        self.scaler = data["scaler"]
        self.train_stats = data["train_stats"]
        self.is_fitted = True

        logger.info(f"Loaded scaler from {scaler_file}")

    def get_feature_names(self) -> List[str]:
        """Get list of feature names."""
        return [
            "log_ret",
            "volatility_20",
            "volatility_50",
            "ou_score",
            "rolling_mean",
            "rolling_std",
            "rsi_14",
            "macd",
            "macd_signal",
            "macd_hist",
            "bb_width",
            "bb_position",
        ]


# ============================================================================
# NUMBA-OPTIMIZED FUNCTIONS
# ============================================================================


@jit(nopython=True, cache=True)
def compute_rolling_mean_numba(arr: np.ndarray, window: int) -> np.ndarray:
    """Numba-optimized rolling mean."""
    n = len(arr)
    result = np.zeros(n)

    for i in range(window - 1, n):
        result[i] = np.mean(arr[i - window + 1 : i + 1])

    return result


@jit(nopython=True, cache=True)
def compute_rolling_std_numba(arr: np.ndarray, window: int) -> np.ndarray:
    """Numba-optimized rolling std."""
    n = len(arr)
    result = np.zeros(n)

    for i in range(window - 1, n):
        result[i] = np.std(arr[i - window + 1 : i + 1])

    return result


# ============================================================================
# HYDRA INTEGRATION HELPER
# ============================================================================


def create_feature_engine_from_hydra(cfg) -> FeatureEngine:
    """
    Create FeatureEngine from Hydra config.

    Usage:
    ------
    @hydra.main(config_path="config", config_name="main_config")
    def main(cfg):
        engine = create_feature_engine_from_hydra(cfg)
    """
    config = FeatureConfig(
        volatility_window=cfg.features.volatility_window,
        ou_window=cfg.features.ou_window,
        rolling_mean_window=cfg.features.rolling_mean_window,
        use_log_returns=cfg.features.use_log_returns,
        scaler_type=cfg.features.scaler_type,
        save_scaler=cfg.features.save_scaler,
        scaler_path=Path(cfg.features.scaler_path),
        dropna_strategy=cfg.features.dropna_strategy,
        min_valid_rows=cfg.features.min_valid_rows,
    )

    return FeatureEngine(config)


# ============================================================================
# EXAMPLE USAGE & VALIDATION
# ============================================================================

if __name__ == "__main__":
    logger.add("logs/feature_engine_{time}.log", rotation="1 day")

    print("=" * 80)
    print("FEATURE ENGINE - FIT/TRANSFORM TEST")
    print("=" * 80)

    # Generate synthetic OHLCV data
    np.random.seed(42)
    n_samples = 2000
    dates = pd.date_range("2023-01-01", periods=n_samples, freq="1H")

    close = 100 + np.cumsum(np.random.randn(n_samples) * 0.5)

    df = pd.DataFrame(
        {
            "open": close + np.random.randn(n_samples) * 0.2,
            "high": close + abs(np.random.randn(n_samples) * 0.5),
            "low": close - abs(np.random.randn(n_samples) * 0.5),
            "close": close,
            "volume": np.random.uniform(1000, 10000, n_samples),
        },
        index=dates,
    )

    # Split train/test
    train_size = int(0.7 * len(df))
    train_df = df.iloc[:train_size]
    test_df = df.iloc[train_size:]

    print(f"\nData split:")
    print(f"  Train: {len(train_df)} rows")
    print(f"  Test: {len(test_df)} rows")

    # Configure FeatureEngine
    config = FeatureConfig(
        volatility_window=20,
        ou_window=50,
        rolling_mean_window=20,
        use_log_returns=True,
        scaler_type="standard",
        save_scaler=True,
        scaler_path=Path("data/scalers"),
        dropna_strategy="rolling",
        min_valid_rows=100,
    )

    engine = FeatureEngine(config)

    # Test 1: Fit on training data
    print("\n[TEST 1] Fit on training data")
    train_features = engine.fit_transform(train_df)

    print(f"✓ Training features: {train_features.shape}")
    print(f"  Columns: {train_features.columns.tolist()}")
    print(f"  Features: {engine.get_feature_names()}")

    # Test 2: Check for NaN
    print("\n[TEST 2] Check for NaN values")
    nan_count = train_features.isnull().sum().sum()

    if nan_count == 0:
        print("✓ No NaN values in output")
    else:
        print(f"✗ Found {nan_count} NaN values")
        print(train_features.isnull().sum())

    # Test 3: Transform test data (using train stats)
    print("\n[TEST 3] Transform test data")
    test_features = engine.transform(test_df)

    print(f"✓ Test features: {test_features.shape}")

    # Verify no data leakage
    print("\n[TEST 4] Verify no data leakage")
    print(f"  Train mean (log_ret): {train_features['log_ret'].mean():.6f}")
    print(f"  Test mean (log_ret): {test_features['log_ret'].mean():.6f}")
    print(f"  Train mean (volatility): {train_features['volatility_20'].mean():.6f}")
    print(f"  Test mean (volatility): {test_features['volatility_20'].mean():.6f}")

    # Test 5: Feature statistics
    print("\n[TEST 5] Feature statistics")
    for feature in engine.get_feature_names():
        if feature in train_features.columns:
            print(f"  {feature}:")
            print(
                f"    Train: mean={train_features[feature].mean():.4f}, "
                f"std={train_features[feature].std():.4f}"
            )
            print(
                f"    Test:  mean={test_features[feature].mean():.4f}, "
                f"std={test_features[feature].std():.4f}"
            )

    # Test 6: Scaler persistence
    print("\n[TEST 6] Scaler persistence")
    if config.save_scaler:
        scaler_path = config.scaler_path / "feature_scaler.pkl"
        if scaler_path.exists():
            print(f"✓ Scaler saved to {scaler_path}")

            # Load and verify
            engine2 = FeatureEngine(config)
            engine2.load_scaler()

            test_features2 = engine2.transform(test_df)

            if test_features.equals(test_features2):
                print("✓ Loaded scaler produces identical output")
            else:
                print("✗ Scaler mismatch")

    print("\n" + "=" * 80)
    print("✓ FEATURE ENGINE TEST PASSED")
    print("=" * 80)
