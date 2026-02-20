"""
GARCH Models for Volatility Forecasting
========================================
Generalized Autoregressive Conditional Heteroskedasticity

Used for:
- Volatility clustering modeling
- Risk management (VaR calculations)
- Option pricing
- Position sizing based on volatility
- Market regime detection

Mathematical Model:
    σ²_t = ω + α*ε²_{t-1} + β*σ²_{t-1}

Where:
    σ²_t: Conditional variance at time t
    ω: Long-term average variance
    α: ARCH parameter (impact of shocks)
    β: GARCH parameter (persistence)
    ε_{t-1}: Previous return shock
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, List
from scipy.optimize import minimize
from scipy.stats import norm
import warnings

warnings.filterwarnings("ignore")


class GARCHModel:
    """
    GARCH(1,1) model for volatility forecasting.

    The GARCH(1,1) is the industry standard for volatility modeling
    due to its parsimony and effectiveness.
    """

    def __init__(self, p: int = 1, q: int = 1):
        """
        Initialize GARCH model.

        Args:
            p: GARCH order (past variances)
            q: ARCH order (past squared returns)
        """
        self.p = p
        self.q = q

        # Parameters: omega, alpha, beta
        self.omega = None
        self.alpha = None
        self.beta = None

        self.is_fitted = False
        self.sigma_history = []
        self.log_likelihood = None

    def _garch_variance(self, params: np.ndarray, returns: np.ndarray) -> np.ndarray:
        """
        Calculate conditional variance series.

        Args:
            params: [omega, alpha, beta]
            returns: Return series

        Returns:
            Variance series
        """
        omega, alpha, beta = params
        T = len(returns)

        sigma2 = np.zeros(T)
        sigma2[0] = np.var(returns)  # Initialize with unconditional variance

        for t in range(1, T):
            sigma2[t] = omega + alpha * returns[t - 1] ** 2 + beta * sigma2[t - 1]

        return sigma2

    def _log_likelihood(self, params: np.ndarray, returns: np.ndarray) -> float:
        """
        Calculate negative log-likelihood (for minimization).

        Args:
            params: [omega, alpha, beta]
            returns: Return series

        Returns:
            Negative log-likelihood
        """
        sigma2 = self._garch_variance(params, returns)

        # Avoid log of zero
        sigma2 = np.maximum(sigma2, 1e-10)

        # Log-likelihood for normal distribution
        ll = -0.5 * np.sum(np.log(2 * np.pi * sigma2) + returns**2 / sigma2)

        return -ll  # Return negative for minimization

    def fit(self, returns: np.ndarray) -> dict:
        """
        Fit GARCH model to return data.

        Args:
            returns: Array of returns

        Returns:
            Dictionary with fitted parameters and statistics
        """
        # Initial parameter guesses
        # omega: unconditional variance * (1 - alpha - beta)
        # alpha: 0.1 (typical for financial data)
        # beta: 0.85 (typical persistence)

        uncond_var = np.var(returns)
        init_params = np.array(
            [
                uncond_var * 0.05,  # omega
                0.1,  # alpha
                0.85,  # beta
            ]
        )

        # Constraints: omega > 0, alpha > 0, beta > 0, alpha + beta < 1
        bounds = [
            (1e-6, None),  # omega > 0
            (0, 0.5),  # 0 < alpha < 0.5
            (0, 0.999),  # 0 < beta < 1
        ]

        # Optimization
        result = minimize(
            self._log_likelihood,
            init_params,
            args=(returns,),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 1000},
        )

        if result.success:
            self.omega, self.alpha, self.beta = result.x
            self.is_fitted = True
            self.log_likelihood = -result.fun

            # Calculate variance history
            self.sigma_history = self._garch_variance(result.x, returns)

            # Calculate persistence
            persistence = self.alpha + self.beta
            half_life = np.log(0.5) / np.log(persistence) if persistence < 1 else np.inf

            return {
                "omega": self.omega,
                "alpha": self.alpha,
                "beta": self.beta,
                "persistence": persistence,
                "half_life": half_life,
                "log_likelihood": self.log_likelihood,
                "aic": -2 * self.log_likelihood + 2 * 3,  # AIC = -2*LL + 2*k
                "bic": -2 * self.log_likelihood + 3 * np.log(len(returns)),
                "success": True,
            }
        else:
            return {"success": False, "message": result.message}

    def forecast(self, steps: int = 1) -> np.ndarray:
        """
        Forecast future volatility.

        Args:
            steps: Number of steps to forecast

        Returns:
            Array of forecasted volatilities
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before forecasting")

        if len(self.sigma_history) == 0:
            raise ValueError("No variance history available")

        # Long-run variance
        long_run_var = self.omega / (1 - self.alpha - self.beta)

        # Current variance
        current_var = self.sigma_history[-1]

        forecasts = np.zeros(steps)

        for i in range(steps):
            # GARCH(1,1) forecast converges to long-run variance
            forecasts[i] = long_run_var + (self.alpha + self.beta) ** i * (
                current_var - long_run_var
            )

        return np.sqrt(forecasts)  # Return standard deviation (volatility)

    def get_conditional_volatility(self) -> np.ndarray:
        """Get fitted conditional volatility series."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        return np.sqrt(self.sigma_history)

    def calculate_var(self, confidence: float = 0.95, horizon: int = 1) -> float:
        """
        Calculate Value at Risk (VaR).

        Args:
            confidence: Confidence level (e.g., 0.95 for 95%)
            horizon: Time horizon in periods

        Returns:
            VaR as positive number
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")

        # Forecast volatility for horizon
        vol_forecast = self.forecast(horizon)[-1]

        # VaR = z-score * volatility
        z_score = norm.ppf(confidence)
        var = z_score * vol_forecast

        return var


class VolatilityRegimeDetector:
    """
    Detect market regimes based on volatility using GARCH.
    """

    def __init__(
        self, high_vol_threshold: float = 0.02, low_vol_threshold: float = 0.005
    ):
        """
        Initialize regime detector.

        Args:
            high_vol_threshold: Threshold for high volatility (daily)
            low_vol_threshold: Threshold for low volatility (daily)
        """
        self.high_threshold = high_vol_threshold
        self.low_threshold = low_vol_threshold
        self.garch_model = GARCHModel()

    def fit(self, returns: np.ndarray) -> dict:
        """Fit GARCH and determine regimes."""
        result = self.garch_model.fit(returns)

        if not result["success"]:
            return {"success": False}

        # Get conditional volatility
        cond_vol = self.garch_model.get_conditional_volatility()

        # Classify regimes
        regimes = np.where(
            cond_vol > self.high_threshold,
            2,  # High vol
            np.where(cond_vol < self.low_threshold, 0, 1),  # Low vol, Normal
        )

        # Calculate regime statistics
        regime_counts = np.bincount(regimes, minlength=3)
        total = len(regimes)

        return {
            "success": True,
            "regimes": regimes,
            "current_regime": regimes[-1],
            "regime_names": {0: "Low Vol", 1: "Normal", 2: "High Vol"},
            "regime_probabilities": {
                "low_vol": regime_counts[0] / total,
                "normal": regime_counts[1] / total,
                "high_vol": regime_counts[2] / total,
            },
            "current_volatility": cond_vol[-1],
            "garch_params": result,
        }

    def get_position_size_multiplier(self) -> float:
        """
        Get position size adjustment based on current volatility.

        Returns:
            Multiplier (1.0 = normal, <1 = reduce size, >1 = increase size)
        """
        if not self.garch_model.is_fitted:
            return 1.0

        current_vol = self.garch_model.get_conditional_volatility()[-1]

        # Reduce position size in high volatility
        if current_vol > self.high_threshold:
            return 0.5  # Half size
        elif current_vol > self.low_threshold * 2:
            return 0.75  # 75% size
        else:
            return 1.0  # Full size


class VolatilityTargeting:
    """
    Target constant volatility in portfolio.

    Adjusts position sizes to maintain target volatility level.
    """

    def __init__(
        self,
        target_volatility: float = 0.15,  # 15% annualized
        lookback: int = 60,
    ):
        """
        Initialize volatility targeting.

        Args:
            target_volatility: Target annualized volatility
            lookback: Lookback period for calculation
        """
        self.target_vol = target_volatility
        self.lookback = lookback
        self.garch_model = GARCHModel()

    def fit(self, returns: np.ndarray):
        """Fit GARCH model."""
        self.garch_model.fit(returns)

    def get_position_scalar(self) -> float:
        """
        Calculate position size scalar.

        Returns:
            Scalar to multiply base position size
        """
        if not self.garch_model.is_fitted:
            return 1.0

        # Forecast volatility
        forecast_vol = self.garch_model.forecast(1)[0]

        # Annualize (assuming daily data)
        annual_vol = forecast_vol * np.sqrt(252)

        # Calculate scalar
        scalar = self.target_vol / annual_vol

        # Limit extremes
        return np.clip(scalar, 0.1, 3.0)

    def apply_volatility_targeting(self, signal: float, returns: np.ndarray) -> float:
        """
        Apply volatility targeting to trading signal.

        Args:
            signal: Raw trading signal (-1 to 1)
            returns: Historical returns for estimation

        Returns:
            Adjusted signal
        """
        self.fit(returns)
        scalar = self.get_position_scalar()
        return signal * scalar


# Utility functions
def calculate_realized_volatility(returns: np.ndarray, window: int = 20) -> pd.Series:
    """
    Calculate rolling realized volatility.

    Args:
        returns: Return series
        window: Rolling window size

    Returns:
        Series of realized volatilities
    """
    return pd.Series(returns).rolling(window).std() * np.sqrt(252)


def forecast_volatility_garch(returns: np.ndarray, steps: int = 5) -> np.ndarray:
    """
    Convenience function to forecast volatility.

    Args:
        returns: Historical returns
        steps: Number of steps to forecast

    Returns:
        Volatility forecasts
    """
    model = GARCHModel()
    result = model.fit(returns)

    if result["success"]:
        return model.forecast(steps)
    else:
        # Fallback to historical volatility
        hist_vol = np.std(returns) * np.ones(steps)
        return hist_vol


def calculate_var_garch(
    returns: np.ndarray, confidence: float = 0.95, horizon: int = 1
) -> Tuple[float, float]:
    """
    Calculate Value at Risk using GARCH.

    Args:
        returns: Historical returns
        confidence: Confidence level
        horizon: Time horizon

    Returns:
        (VaR, Conditional VaR)
    """
    model = GARCHModel()
    result = model.fit(returns)

    if not result["success"]:
        # Fallback to historical VaR
        var = np.percentile(returns, (1 - confidence) * 100)
        return var, var

    # Calculate VaR
    var = model.calculate_var(confidence, horizon)

    # Calculate CVaR (Expected Shortfall)
    # Simplified calculation
    z_score = norm.ppf(confidence)
    cvar = var + (norm.pdf(z_score) / (1 - confidence)) * model.forecast(horizon)[-1]

    return var, cvar
