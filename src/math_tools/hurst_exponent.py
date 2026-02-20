"""
Hurst Exponent for Trend Analysis
==================================
Measures long-term memory and persistence in time series.

Mathematical Interpretation:
- H = 0.5: Random walk (Brownian motion)
- H > 0.5: Trending/persistent (trend following works)
- H < 0.5: Mean-reverting/anti-persistent (mean reversion works)

Used for:
- Trend vs. mean-reversion detection
- Strategy selection (trend vs. MR)
- Market regime identification
- Trading horizon optimization
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional
from scipy import stats
import warnings

warnings.filterwarnings("ignore")


class HurstExponent:
    """
    Calculate Hurst exponent using multiple methods.

    The Hurst exponent (H) characterizes the time series:
    - H < 0.5: Mean-reverting (anti-persistent)
    - H = 0.5: Random walk (no memory)
    - H > 0.5: Trending (persistent)
    """

    def __init__(self, max_lag: int = 100):
        """
        Initialize Hurst calculator.

        Args:
            max_lag: Maximum lag for calculations
        """
        self.max_lag = max_lag

    def rescaled_range(self, ts: np.ndarray) -> float:
        """
        Calculate Hurst exponent using R/S analysis (classic method).

        Mathematical Method:
            1. Divide series into chunks
            2. For each chunk: calculate mean, std, and cumulative deviation
            3. Calculate range R = max - min of cumulative deviations
            4. Calculate R/S statistic
            5. Regress log(R/S) vs log(chunk size)
            6. Slope = Hurst exponent

        Args:
            ts: Time series (prices or returns)

        Returns:
            Hurst exponent
        """
        lags = range(2, min(self.max_lag, len(ts) // 4))
        tau = [np.sqrt(np.std(np.subtract(ts[lag:], ts[:-lag]))) for lag in lags]

        # Linear regression on log-log plot
        poly = np.polyfit(np.log(lags), np.log(tau), 1)

        return poly[0] * 2.0  # Hurst exponent

    def detrended_fluctuation_analysis(self, ts: np.ndarray) -> float:
        """
        Calculate Hurst using Detrended Fluctuation Analysis (DFA).

        DFA is more robust to non-stationarities than R/S method.

        Args:
            ts: Time series

        Returns:
            Hurst exponent
        """
        # Create profile (cumulative sum of deviations from mean)
        profile = np.cumsum(ts - np.mean(ts))

        # Try different window sizes
        lags = range(4, min(self.max_lag, len(ts) // 4))
        flucts = []

        for lag in lags:
            # Split into windows
            n_windows = len(profile) // lag

            fluct = 0
            for i in range(n_windows):
                # Extract window
                window = profile[i * lag : (i + 1) * lag]

                # Fit linear trend
                x = np.arange(lag)
                slope, intercept, _, _, _ = stats.linregress(x, window)
                trend = slope * x + intercept

                # Calculate fluctuation (RMS of detrended series)
                fluct += np.mean((window - trend) ** 2)

            flucts.append(np.sqrt(fluct / n_windows))

        # Linear fit on log-log scale
        poly = np.polyfit(np.log(list(lags)), np.log(flucts), 1)

        return poly[0]

    def variance_method(self, ts: np.ndarray) -> float:
        """
        Calculate Hurst using aggregated variance method.

        Args:
            ts: Time series

        Returns:
            Hurst exponent
        """
        lags = range(2, min(self.max_lag, len(ts) // 4))
        variances = []

        for lag in lags:
            # Aggregate series
            aggregated = [np.mean(ts[i : i + lag]) for i in range(0, len(ts), lag)]
            variances.append(np.var(aggregated))

        # Linear fit
        poly = np.polyfit(np.log(lags), np.log(variances), 1)

        # Hurst = 1 - slope/2
        return 1 - poly[0] / 2

    def calculate(self, ts: np.ndarray, method: str = "dfa") -> float:
        """
        Calculate Hurst exponent using specified method.

        Args:
            ts: Time series
            method: 'rs', 'dfa', 'variance', or 'aggregate'

        Returns:
            Hurst exponent
        """
        if method == "rs":
            return self.rescaled_range(ts)
        elif method == "dfa":
            return self.detrended_fluctuation_analysis(ts)
        elif method == "variance":
            return self.variance_method(ts)
        elif method == "aggregate":
            # Average of all methods
            rs_h = self.rescaled_range(ts)
            dfa_h = self.detrended_fluctuation_analysis(ts)
            var_h = self.variance_method(ts)
            return np.median([rs_h, dfa_h, var_h])
        else:
            raise ValueError(f"Unknown method: {method}")

    def get_regime(self, hurst: float) -> str:
        """
        Interpret Hurst value.

        Args:
            hurst: Hurst exponent value

        Returns:
            Regime description
        """
        if hurst < 0.4:
            return "strong_mean_reversion"
        elif hurst < 0.45:
            return "mean_reversion"
        elif hurst < 0.55:
            return "random_walk"
        elif hurst < 0.6:
            return "trending"
        else:
            return "strong_trend"

    def get_trading_recommendation(self, hurst: float) -> dict:
        """
        Get trading strategy recommendation based on Hurst.

        Args:
            hurst: Hurst exponent

        Returns:
            Dictionary with recommendations
        """
        regime = self.get_regime(hurst)

        recommendations = {
            "strong_mean_reversion": {
                "strategy": "Mean Reversion",
                "style": "Bollinger Bands, RSI, Z-Score",
                "holding_period": "Short (hours to days)",
                "risk": "Breakouts can be severe",
                "confidence": "High" if hurst < 0.35 else "Medium",
            },
            "mean_reversion": {
                "strategy": "Mean Reversion",
                "style": "Pairs trading, Statistical arbitrage",
                "holding_period": "Medium (days to weeks)",
                "risk": "Moderate",
                "confidence": "Medium",
            },
            "random_walk": {
                "strategy": "None / Momentum",
                "style": "Trend following on higher timeframe",
                "holding_period": "Long (weeks to months)",
                "risk": "High (no edge)",
                "confidence": "Low",
            },
            "trending": {
                "strategy": "Trend Following",
                "style": "Moving averages, Breakouts",
                "holding_period": "Medium (days to weeks)",
                "risk": "Moderate",
                "confidence": "Medium",
            },
            "strong_trend": {
                "strategy": "Trend Following",
                "style": "Momentum, Position trading",
                "holding_period": "Long (weeks to months)",
                "risk": "Reversals can be sharp",
                "confidence": "High" if hurst > 0.65 else "Medium",
            },
        }

        return recommendations.get(regime, recommendations["random_walk"])


class RollingHurst:
    """
    Calculate rolling Hurst exponent for regime detection.

    Tracks how market regime changes over time.
    """

    def __init__(self, window: int = 200, step: int = 20):
        """
        Initialize rolling Hurst calculator.

        Args:
            window: Rolling window size
            step: Step size between calculations
        """
        self.window = window
        self.step = step
        self.hurst_calc = HurstExponent(max_lag=min(100, window // 4))

    def calculate(self, prices: pd.Series) -> pd.Series:
        """
        Calculate rolling Hurst exponent.

        Args:
            prices: Price series

        Returns:
            Series of Hurst values
        """
        hurst_values = []
        indices = []

        # Calculate returns
        returns = prices.pct_change().dropna().values

        for i in range(self.window, len(returns), self.step):
            window_data = returns[i - self.window : i]

            try:
                hurst = self.hurst_calc.calculate(window_data, method="dfa")

                # Sanity check
                if 0 < hurst < 1:
                    hurst_values.append(hurst)
                    indices.append(prices.index[i])
            except:
                continue

        return pd.Series(hurst_values, index=indices)

    def detect_regime_change(
        self, hurst_series: pd.Series, threshold: float = 0.1
    ) -> list:
        """
        Detect significant regime changes.

        Args:
            hurst_series: Series of Hurst values
            threshold: Minimum change to trigger regime change

        Returns:
            List of regime change timestamps
        """
        changes = []

        for i in range(1, len(hurst_series)):
            current_regime = self.hurst_calc.get_regime(hurst_series.iloc[i])
            prev_regime = self.hurst_calc.get_regime(hurst_series.iloc[i - 1])

            if current_regime != prev_regime:
                changes.append(
                    {
                        "timestamp": hurst_series.index[i],
                        "from_regime": prev_regime,
                        "to_regime": current_regime,
                        "hurst": hurst_series.iloc[i],
                    }
                )

        return changes


class MultiScaleHurst:
    """
    Analyze Hurst exponent at multiple time scales.

    Professional traders analyze multiple timeframes to confirm signals.
    """

    def __init__(self):
        """Initialize multi-scale Hurst analyzer."""
        self.hurst_calc = HurstExponent()

    def analyze_scales(self, prices: pd.Series, scales: list = None) -> dict:
        """
        Calculate Hurst at multiple time scales.

        Args:
            prices: Price series
            scales: List of time scales (e.g., ['1H', '4H', '1D'])

        Returns:
            Dictionary with Hurst values per scale
        """
        if scales is None:
            scales = ["1H", "4H", "1D", "1W"]

        results = {}

        for scale in scales:
            try:
                # Resample to scale
                resampled = prices.resample(scale).last().dropna()

                if len(resampled) < 100:
                    continue

                # Calculate returns
                returns = resampled.pct_change().dropna().values

                # Calculate Hurst
                hurst = self.hurst_calc.calculate(returns, method="dfa")

                results[scale] = {
                    "hurst": hurst,
                    "regime": self.hurst_calc.get_regime(hurst),
                    "recommendation": self.hurst_calc.get_trading_recommendation(hurst),
                }
            except:
                continue

        return results

    def get_consensus(self, scale_results: dict) -> dict:
        """
        Get consensus view across time scales.

        Args:
            scale_results: Results from analyze_scales()

        Returns:
            Consensus recommendation
        """
        hurst_values = [r["hurst"] for r in scale_results.values()]

        if not hurst_values:
            return {"consensus": "unknown", "confidence": 0}

        avg_hurst = np.mean(hurst_values)
        std_hurst = np.std(hurst_values)

        # Check if all scales agree
        regimes = [r["regime"] for r in scale_results.values()]
        unique_regimes = set(regimes)

        if len(unique_regimes) == 1:
            agreement = "strong"
        elif len(unique_regimes) == 2:
            agreement = "mixed"
        else:
            agreement = "conflicting"

        return {
            "consensus_hurst": avg_hurst,
            "consensus_regime": self.hurst_calc.get_regime(avg_hurst),
            "agreement": agreement,
            "confidence": 1 - min(1, std_hurst * 2),  # Higher std = lower confidence
            "recommendation": self.hurst_calc.get_trading_recommendation(avg_hurst),
        }


# Utility functions
def quick_hurst_check(prices: pd.Series) -> dict:
    """
    Quick Hurst analysis for immediate decision making.

    Args:
        prices: Price series

    Returns:
        Dictionary with key metrics
    """
    calc = HurstExponent()

    returns = prices.pct_change().dropna().values

    if len(returns) < 100:
        return {"error": "Insufficient data (need >100 points)"}

    hurst = calc.calculate(returns, method="dfa")
    regime = calc.get_regime(hurst)
    recommendation = calc.get_trading_recommendation(hurst)

    return {
        "hurst_exponent": round(hurst, 4),
        "regime": regime,
        "interpretation": "Trending"
        if hurst > 0.5
        else "Mean-Reverting"
        if hurst < 0.5
        else "Random",
        "strategy": recommendation["strategy"],
        "confidence": recommendation["confidence"],
        "holding_period": recommendation["holding_period"],
    }


def hurst_adaptive_strategy(prices: pd.Series, base_signal: float) -> float:
    """
    Adjust trading signal based on Hurst regime.

    Args:
        prices: Price series
        base_signal: Original signal (-1 to 1)

    Returns:
        Adjusted signal
    """
    calc = HurstExponent()
    returns = prices.pct_change().dropna().values

    if len(returns) < 100:
        return base_signal

    hurst = calc.calculate(returns, method="dfa")

    # Adjust signal based on regime
    if hurst < 0.4:  # Strong mean reversion
        # Flip signal (mean reversion)
        return -base_signal * 1.2
    elif hurst < 0.45:  # Mean reversion
        # Reduce trend following signal
        return base_signal * 0.5
    elif hurst > 0.6:  # Strong trend
        # Amplify trend signal
        return base_signal * 1.3
    elif hurst > 0.55:  # Trending
        # Standard trend following
        return base_signal
    else:
        # Random walk - reduce position
        return base_signal * 0.3
