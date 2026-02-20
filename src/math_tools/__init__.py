"""Mathematical Tools Module."""

from src.math_tools.ornstein_uhlenbeck import (
    OrnsteinUhlenbeckProcess,
    calculate_ou_score_vectorized as calculate_ou_score,
)

from src.math_tools.hmm_regime import (
    HMMRegimeDetector,
    MarketRegime as RegimeState,
)

from src.math_tools.kelly_criterion import (
    KellyCriterion,
    calculate_kelly_numba as kelly_fraction,
)

from src.math_tools.kalman_filter import (
    KalmanFilter1D,
    KalmanFilterPairs,
    KalmanTrendDetector,
    KalmanFilterConfig,
    apply_kalman_smoothing,
    detect_price_jumps,
    calculate_dynamic_beta,
)

from src.math_tools.cointegration import (
    CointegrationTest,
    PairsTradingStrategy,
    StatisticalArbitragePortfolio,
    calculate_hedge_ratio,
    calculate_spread,
    find_best_pairs,
)

from src.math_tools.garch_models import (
    GARCHModel,
    VolatilityRegimeDetector,
    VolatilityTargeting,
    calculate_realized_volatility,
    forecast_volatility_garch,
    calculate_var_garch,
)

from src.math_tools.hurst_exponent import (
    HurstExponent,
    RollingHurst,
    MultiScaleHurst,
    quick_hurst_check,
    hurst_adaptive_strategy,
)

from src.math_tools.spectral_analysis import (
    SpectralAnalyzer,
    HilbertTransformAnalyzer,
    AdaptiveCycleIndicator,
    SeasonalityAnalyzer,
    compute_dominant_cycle,
    cycle_based_signal,
    remove_seasonality,
    spectral_edge_detection,
)

__all__ = [
    # Ornstein-Uhlenbeck
    "OrnsteinUhlenbeckProcess",
    "calculate_ou_score",
    # HMM Regime
    "HMMRegimeDetector",
    "RegimeState",
    # Kelly Criterion
    "KellyCriterion",
    "kelly_fraction",
    "kelly_position_size",
    # Kalman Filter
    "KalmanFilter1D",
    "KalmanFilterPairs",
    "KalmanTrendDetector",
    "KalmanFilterConfig",
    "apply_kalman_smoothing",
    "detect_price_jumps",
    "calculate_dynamic_beta",
    # Cointegration
    "CointegrationTest",
    "PairsTradingStrategy",
    "StatisticalArbitragePortfolio",
    "calculate_hedge_ratio",
    "calculate_spread",
    "find_best_pairs",
    # GARCH
    "GARCHModel",
    "VolatilityRegimeDetector",
    "VolatilityTargeting",
    "calculate_realized_volatility",
    "forecast_volatility_garch",
    "calculate_var_garch",
    # Hurst Exponent
    "HurstExponent",
    "RollingHurst",
    "MultiScaleHurst",
    "quick_hurst_check",
    "hurst_adaptive_strategy",
    # Spectral Analysis
    "SpectralAnalyzer",
    "HilbertTransformAnalyzer",
    "AdaptiveCycleIndicator",
    "SeasonalityAnalyzer",
    "compute_dominant_cycle",
    "cycle_based_signal",
    "remove_seasonality",
    "spectral_edge_detection",
]
