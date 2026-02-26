"""
GARCH Models for Volatility Forecasting
=======================================
Generalized Autoregressive Conditional Heteroskedasticity

Mathematical Foundation:
------------------------
GARCH models capture volatility clustering - the empirical observation that
large price changes tend to follow large changes, and small tend to follow small.

GARCH(1,1) Model:
    σ²_t = ω + α·ε²_{t-1} + β·σ²_{t-1}

    where:
        σ²_t: Conditional variance at time t (forecasted volatility²)
        ω (omega): Long-term average variance (baseline)
        α (alpha): ARCH parameter - impact of yesterday's shock
        β (beta): GARCH parameter - persistence of variance
        ε_{t-1}: Squared return at t-1 (shock)

    Constraint: α + β < 1 (ensures stationarity)

Integrated GARCH (IGARCH):
    If α + β = 1, variance follows random walk
    Today's variance = yesterday's variance + ω + shock²

Volatility Persistence:
    P = α + β (sum of coefficients)

    - P → 1: High persistence (shocks decay slowly)
    - P < 1: Stationary (shocks decay exponentially)

Half-Life of Volatility Shocks:
    t_½ = ln(0.5) / ln(α + β)

    Time for a volatility shock to decay by 50%

Long-Run (Unconditional) Variance:
    Var(∞) = ω / (1 - α - β)

    This is the variance the process reverts to over time

Log-Likelihood (Gaussian):
    LL = -½Σ[log(2π) + log(σ²_t) + ε²_t/σ²_t]

    Maximized during estimation

Value at Risk (VaR):
    VaR_α = z_α × σ_t

    where z_α is the standard normal quantile

Conditional VaR (Expected Shortfall):
    CVaR_α = E[loss | loss > VaR_α]
           = σ_t × [φ(z_α) / (1-α)]

    where φ is the PDF

Used for:
- Volatility clustering modeling
- Risk management (VaR calculations)
- Option pricing
- Position sizing based on volatility
- Market regime detection

References:
- Engle, R.F. (1982) "Autoregressive Conditional Heteroskedasticity with Estimates of UK Inflation"
- Bollerslev, T. (1986) "Generalized Autoregressive Conditional Heteroskedasticity"
- Tsay, R.S. (2010) "Analysis of Financial Time Series"

Dependencies:
- numpy: Numerical computations
- pandas: Time series handling
- scipy: Optimization and statistics
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

    Mathematical Background:
    -----------------------
    GARCH(1,1) models time-varying volatility as:

        σ²_t = ω + α·r²_{t-1} + β·σ²_{t-1}

    where:
        σ²_t: Conditional variance at time t
        r_{t-1}: Return at time t-1
        σ²_{t-1}: Conditional variance at t-1

    Key Properties:

    1. Volatility Persistence (α + β):
       - Close to 1 = high persistence (volatility shocks linger)
       - Financial data typically: 0.90 - 0.99

    2. Half-Life:
       t_½ = ln(0.5) / ln(α + β)

       Example: If α + β = 0.95
                t_½ = ln(0.5) / ln(0.95) ≈ 13.5 periods

    3. Long-Run Variance:
       σ²_∞ = ω / (1 - α - β)

       The variance the process reverts to

    4. Forecasting:
       σ²_{t+h} = σ²_∞ + (α+β)^h × (σ²_t - σ²_∞)

    Parameter Estimation:
    --------------------
    Uses Maximum Likelihood Estimation (MLE):

    1. Initialize: σ²_0 = sample variance
    2. Iterate: σ²_t = ω + αr²_{t-1} + βσ²_{t-1}
    3. Compute: LL = -½Σ[log(2π) + log(σ²_t) + r²_t/σ²_t]
    4. Optimize: Maximize LL w.r.t. (ω, α, β)

    Constraints:
       ω > 0, α ≥ 0, β ≥ 0, α + β < 1

    Parameters:
    -----------
    p : int
        GARCH order (number of lagged variance terms). Default: 1
        Typically p=1 is sufficient for financial data

    q : int
        ARCH order (number of lagged squared returns). Default: 1
        Typically q=1 is sufficient

    Attributes:
    ----------
    omega : float
        Long-term variance coefficient (ω)

    alpha : float
        ARCH coefficient (α) - impact of recent returns

    beta : float
        GARCH coefficient (β) - volatility persistence

    sigma_history : np.ndarray
        Fitted conditional volatility series

    log_likelihood : float
        Log-likelihood at fitted parameters

    is_fitted : bool
        Whether model has been fitted

    Methods:
    --------
    fit(returns)
        Estimate parameters using MLE

    forecast(steps)
        Forecast future volatility

    get_conditional_volatility()
        Get historical conditional volatility

    calculate_var(confidence, horizon)
        Calculate Value at Risk

    Returns:
    --------
    Dictionary containing:
        - omega, alpha, beta: Fitted parameters
        - persistence: α + β
        - half_life: Mean reversion half-life in periods
        - log_likelihood: Model fit
        - aic, bic: Information criteria

    Example:
    -------
    >>> import numpy as np
    >>>
    >>> # Generate returns with volatility clustering
    >>> np.random.seed(42)
    >>> returns = np.random.randn(1000) * 0.01
    >>> returns[500:510] *= 5  # Add volatility cluster
    >>>
    >>> # Fit GARCH
    >>> model = GARCHModel(p=1, q=1)
    >>> result = model.fit(returns)
    >>>
    >>> print(f"Alpha: {result['alpha']:.4f}")
    >>> print(f"Beta: {result['beta']:.4f}")
    >>> print(f"Persistence: {result['persistence']:.4f}")
    >>> print(f"Half-life: {result['half_life']:.1f} periods")
    >>>
    >>> # Forecast volatility
    >>> vol_forecast = model.forecast(5)
    >>> print(f"5-day volatility forecast: {vol_forecast[-1]:.4f}")
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
        Calculate conditional variance series given parameters.

        Uses the GARCH(1,1) recursion to compute variance at each time step.

        Mathematical Formula:
        --------------------
        σ²_t = ω + α·r²_{t-1} + β·σ²_{t-1}

        where:
            σ²_t: Conditional variance at time t
            r²_{t-1}: Squared return at t-1 (recent "shock")
            σ²_{t-1}: Previous conditional variance
            ω: Long-term variance baseline
            α: Weight on recent shock (ARCH term)
            β: Weight on past variance (GARCH term)

        This is a recursive definition - each variance depends on the previous.

        Args:
            params: Parameter vector [omega, alpha, beta]
            returns: Return series (not squared)

        Returns:
            Variance series of same length as returns
        """
        omega, alpha, beta = params
        T = len(returns)

        # Initialize with unconditional (sample) variance
        # This is σ²_0 = Var(r) before we have any observations
        sigma2 = np.zeros(T)
        sigma2[0] = np.var(returns)

        # GARCH(1,1) recursion: σ²_t = ω + α·r²_{t-1} + β·σ²_{t-1}
        #
        # Interpretation:
        # - ω (omega): Baseline variance - the long-run average
        # - α (alpha): How much yesterday's return shock affects today's variance
        # - β (beta): How much yesterday's variance persists to today
        # - α + β: Volatility persistence (should be < 1 for stationarity)
        for t in range(1, T):
            # Squared return is the "shock" or "innovation"
            # Large |return| → large squared return → increased variance
            shock = returns[t - 1] ** 2

            # Recursive update:
            # New variance = baseline + α*(recent shock) + β*(past variance)
            sigma2[t] = omega + alpha * shock + beta * sigma2[t - 1]

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

        # Gaussian log-likelihood: -0.5 * Σ[log(2π σ²) + ε²/σ²]
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

            # Calculate persistence: α+β < 1 required for stationarity
            persistence = self.alpha + self.beta
            # Half-life: time for shock to decay to half its initial size
            half_life = np.log(0.5) / np.log(persistence) if persistence < 1 else np.inf

            return {
                "omega": self.omega,
                "alpha": self.alpha,
                "beta": self.beta,
                "persistence": persistence,
                "half_life": half_life,
                "log_likelihood": self.log_likelihood,
                "aic": -2 * self.log_likelihood
                + 2 * 3,  # AIC = -2*LL + 2*k, penalizes complexity
                "bic": -2 * self.log_likelihood
                + 3 * np.log(len(returns)),  # BIC penalizes more heavily
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

        # Long-run (unconditional) variance: ω / (1 - α - β)
        long_run_var = self.omega / (1 - self.alpha - self.beta)

        # Most recent conditional variance (starting point for forecast)
        current_var = self.sigma_history[-1]

        forecasts = np.zeros(steps)

        for i in range(steps):
            # GARCH(1,1) multi-step forecast: mean-reverts to long-run variance at rate (α+β)^i
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

        # VaR = z-score * volatility  (normal distribution assumption)
        z_score = norm.ppf(confidence)  # Inverse CDF: e.g. 1.645 for 95% confidence
        var = z_score * vol_forecast  # VaR in return units

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

        # Annualize (assuming daily data): σ_daily * sqrt(252 trading days)
        annual_vol = forecast_vol * np.sqrt(252)

        # Position scalar: target_vol / realized_vol  (scale up when vol is low, scale down when high)
        scalar = self.target_vol / annual_vol

        # Limit extremes to prevent excessive leverage or zero exposure
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
