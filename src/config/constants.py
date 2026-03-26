"""
BITCOIN4Traders — Centralized Constants
========================================
Single source of truth for all magic numbers across the codebase.
Import from here instead of scattering literals throughout modules.

Usage:
    from src.config.constants import RSI_PERIOD, CRYPTO_ANNUALIZATION
"""

# ============================================================
# Feature Engineering
# ============================================================
CRYPTO_ANNUALIZATION = 365 * 1440   # minutes per year (24/7 crypto, not 252 trading days)
RSI_PERIOD = 14
RSI_ALPHA = 1.0 / RSI_PERIOD        # Wilder EWM smoothing
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
BOLLINGER_WINDOW = 20
BOLLINGER_STD = 2.0
ATR_PERIOD = 14
VWAP_WINDOW = 20
VOLATILITY_LONG_WINDOW = 50         # Long-window volatility rolling std
ROLLING_MEAN_WINDOW = 20

# Hurst / DFA parameters
HURST_MAX_LAG = 100
HURST_MAX_ROWS = 5000               # Downsample above this for speed
HURST_MIN_ROWS = 500                # Minimum rows for reliable Hurst
HURST_MIN_LENGTH = 8                # Minimum series length before fallback

# GARCH
GARCH_WINDOW = 100
GARCH_NORM_DIVISOR = 0.10           # Normalize GARCH forecast into [-1,1] range

# OU Process clip range for score feature
OU_SCORE_CLIP = 5.0                 # Clip OU z-score to [-5, 5]

# Numerical epsilons
LOG_RETURN_EPS = 1e-10              # Denominator guard in log-return: log(p/(prev+eps))
FEATURE_EPS = 1e-10                 # General epsilon for feature computations
UNIT_ROOT_EPS = 1e-10               # GARCH integrated-process fallback denominator

# Excluded feature columns (dropped before passing to agent)
EXCLUDED_FEATURES = [
    "open",
    "high",
    "low",
    "close",
    "volume",
]

# ============================================================
# Hurst Regime Thresholds
# ============================================================
HURST_STRONG_MR = 0.40      # Strong mean-reversion
HURST_WEAK_MR = 0.45        # Weak mean-reversion
HURST_WEAK_TREND = 0.55     # Weak trend-following
HURST_STRONG_TREND = 0.60   # Strong trend-following
HURST_HIGH_CONF = 0.35      # High-confidence MR cutoff
HURST_STRONG_CONF = 0.65    # Strong trend confidence cutoff

# ============================================================
# PPO Agent
# ============================================================
DELTA_CLIP = 10.0               # GAE delta clipping
ADVANTAGE_CLIP = 10.0           # Advantage clipping before normalization
STD_EPS = 1e-8                  # Epsilon added to std for numerical stability
LOG_RATIO_CLAMP = (-10.0, 10.0) # log-prob ratio clamp range

# ============================================================
# GARCH Model
# ============================================================
MAX_ALPHA = 0.5                 # Maximum ARCH coefficient
MAX_BETA = 0.999                # Maximum GARCH coefficient
COVARIANCE_EPS = 1e-10          # Minimum covariance (Kalman stability)

# ============================================================
# Kalman Filter
# ============================================================
DEFAULT_PROCESS_NOISE = 0.001
DEFAULT_MEASUREMENT_NOISE = 0.1

# ============================================================
# Risk Management
# ============================================================
SESSION_DRAWDOWN_LIMIT = 0.02   # Max drawdown per live session (circuit breaker)
ACCOUNT_DRAWDOWN_LIMIT = 0.15   # Realistic env episode termination
TRAINING_DRAWDOWN_LIMIT = 0.20  # Standard training env episode termination
MAX_POSITION_SIZE = 0.25        # Max fraction of equity in one position
DEFAULT_KELLY_FRACTION = 0.5    # Fractional Kelly (half-Kelly is standard)
RISK_EPS = 1e-8                 # General risk math epsilon
KELLY_EPS = 1e-9                # Kelly criterion denominator guard

# ============================================================
# Live Engine
# ============================================================
RECONNECT_DELAY_SEC = 60        # WebSocket reconnect delay
THROTTLE_SEC = 10               # Order throttle interval
PRICE_THRESHOLD = 0.0001        # Minimum meaningful price change (BPS)
STACKING_LIMIT = 0.8            # Max position stacking limit (fraction)
