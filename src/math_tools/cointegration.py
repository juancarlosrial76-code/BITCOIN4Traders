"""
Cointegration and Pairs Trading
===============================
Statistical arbitrage between two or more assets.

Mathematical Foundation:
------------------------
Cointegration captures long-run equilibrium relationships between non-stationary
time series. While individual assets may follow random walks, their linear
combination may be stationary.

Key Definitions:
    Unit Root: A time series Y_t is non-stationary if Y_t = ρY_{t-1} + ε_t
                with ρ = 1 (random walk)

    Cointegration: Two series X_t and Y_t are cointegrated if there exists
                  a linear combination αX_t + βY_t that is stationary.

    Error Correction Model (ECM):
        ΔY_t = α(Y_{t-1} - βX_{t-1}) + ε_t
        where (Y_{t-1} - βX_{t-1}) is the error correction term

Engle-Granger Two-Step Test:
    Step 1: Regress Y_t = α + βX_t + ε_t
            Get hedge ratio β and residuals (spread) ε_t

    Step 2: Test residuals ε_t for stationarity using ADF test
            ADF regression: Δε_t = γε_{t-1} + Σδ_jΔε_{t-j} + η_t
            H0: γ = 0 (unit root, no cointegration)
            H1: γ < 0 (stationary, cointegrated)

Johansen Test (for multiple assets):
    Tests for r cointegrating relationships among k variables
    Uses eigenvalue decomposition of the error correction matrix
    Provides trace and max-eigenvalue statistics

Half-Life of Mean Reversion:
    From AR(1) on spread: ε_t = ρε_{t-1} + η_t
    ρ ≈ e^(-θ) where θ is mean-reversion speed
    t_½ = -ln(2)/ln(ρ) = ln(2)/θ

Used for:
- Statistical arbitrage (pairs trading)
- Market-neutral strategies
- Relative value trading
- Hedge ratio calculation
- Beta estimation

References:
- Engle, R.F. & Granger, C.W.J. (1987) "Cointegration and error correction"
- Johansen, S. (1991) "Estimation and hypothesis testing of cointegration vectors"
- Gatev, E. et al. (2006) "Pairs Trading: Performance of a Relative Value Arbitrage Rule"

Dependencies:
- numpy: Numerical computations
- pandas: DataFrame handling
- scipy: Statistical tests
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, List
from scipy import stats
from scipy.stats import jarque_bera
import warnings

warnings.filterwarnings("ignore")


class CointegrationTest:
    """
    Test for cointegration between two time series.

    Cointegration means that while individual series may be non-stationary,
    a linear combination of them is stationary. This is the foundation of
    pairs trading strategies.

    Mathematical Background:
    -----------------------
    Engle-Granger Two-Step Procedure:

    Step 1: Estimate the cointegrating regression
        Y_t = α + βX_t + ε_t

        where:
            Y_t: Dependent variable (price of asset 1)
            X_t: Independent variable (price of asset 2)
            α: Intercept (spread adjustment)
            β: Hedge ratio (beta)
            ε_t: Residuals (the spread)

    Step 2: Test residuals for stationarity using ADF
        Δε_t = γε_{t-1} + Σδ_jΔε_{t-j} + η_t

        Test statistic: t_γ = γ̂ / SE(γ̂)

        Critical values (5%):
            -3.34 (more negative = stronger evidence)

        If |t-stat| > critical value → reject H0 → cointegrated

    Augmented Dickey-Fuller (ADF) Test:
        Tests for presence of unit root (non-stationarity)
        H0: Series has a unit root (non-stationary)
        H1: Series is stationary

        Test regression:
            Δy_t = α + βy_{t-1} + Σγ_jΔy_{t-j} + ε_t

        where β < 0 indicates stationarity

    Parameters:
    -----------
    significance : float
        Significance level for hypothesis testing (default: 0.05).
        Common values:
        - 0.01: 99% confidence
        - 0.05: 95% confidence (standard)
        - 0.10: 90% confidence

    Attributes:
    ----------
    critical_values : dict
        ADF test critical values for 1%, 5%, 10% significance levels

    Methods:
    --------
    adf_test(series, maxlag)
        Augmented Dickey-Fuller test for stationarity

    engle_granger_test(y, x)
        Two-step cointegration test

    find_cointegrated_pairs(prices_df)
        Find all cointegrated pairs in a price matrix

    Returns:
    --------
    Dictionary containing:
        - cointegrated: Boolean indicating cointegration
        - adf_statistic: ADF test statistic
        - p_value: Approximate p-value
        - alpha: Intercept
        - beta/hedge_ratio: Slope coefficient
        - residuals: Spread series
        - half_life: Mean reversion half-life
        - ar1_coefficient: Autocorrelation of spread
        - correlation: Pearson correlation

    Example:
    -------
    >>> import numpy as np
    >>>
    >>> # Generate cointegrated series
    >>> np.random.seed(42)
    >>> x = np.cumsum(np.random.randn(200))
    >>> y = 2*x + 1 + np.cumsum(np.random.randn(200) * 0.1)
    >>>
    >>> test = CointegrationTest(significance=0.05)
    >>> result = test.engle_granger_test(y, x)
    >>>
    >>> print(f"Cointegrated: {result['cointegrated']}")
    >>> print(f"ADF Statistic: {result['adf_statistic']:.3f}")
    >>> print(f"Hedge Ratio: {result['hedge_ratio']:.4f}")
    >>> print(f"Half-life: {result['half_life']:.2f} periods")
    """

    def __init__(self, significance: float = 0.05):
        """
        Initialize cointegration test.

        Args:
            significance: Significance level (default 5%)
        """
        self.significance = significance
        self.critical_values = {"1%": -3.90, "5%": -3.34, "10%": -3.04}

    def adf_test(
        self, series: np.ndarray, maxlag: int = 1
    ) -> Tuple[float, float, bool]:
        """
        Augmented Dickey-Fuller test for stationarity.

        Tests whether a time series has a unit root (is non-stationary)
        or is stationary (mean-reverting).

        Mathematical Formulation:
        -------------------------
        The ADF test estimates the regression:

            Δy_t = α + β*y_{t-1} + Σγ_j*Δy_{t-j} + ε_t

            where:
                y_t: Time series value at t
                Δy_t = y_t - y_{t-1}: First difference
                α: Drift term
                β: Coefficient testing for unit root
                γ_j: Coefficients on lagged differences (for autocorrelation)
                ε_t: White noise error

        Hypothesis:
            H0 (null): β = 0 → Unit root → Non-stationary
            H1 (alternative): β < 0 → Stationary

        Test Statistic:
            t_β = β̂ / SE(β̂)

            More negative = stronger evidence against unit root

        Critical Values (5% significance):
            -3.34 (for n > 100)

        If t-statistic < critical value → reject H0 → stationary

        Args:
            series: Time series to test (must have at least 2 observations)
            maxlag: Maximum number of lagged difference terms to include.
                   Higher = more flexible but less power. Default: 1

        Returns:
            Tuple of (adf_statistic, p_value, is_stationary)
                - adf_statistic: The t-statistic for β
                - p_value: Approximate p-value
                - is_stationary: True if reject unit root at 10% level
        """
        # Calculate first differences: Δy_t = y_t - y_{t-1}
        diff = np.diff(series)

        # Lagged level: y_{t-1} (first element removed)
        lagged = series[:-1]

        # ADF regression: Δy_t = α + β*y_{t-1} + ε_t
        # This tests if β < 0 (stationary) vs β = 0 (unit root)
        X = np.column_stack([np.ones(len(lagged)), lagged])
        y = diff

        # OLS estimation using normal equations: β = (X'X)^{-1} X'y
        beta = np.linalg.lstsq(X, y, rcond=None)[0]

        # Calculate residuals: ε = y - Xβ
        residuals = y - X @ beta

        # Calculate standard error of β (for t-statistic)
        n = len(y)  # Sample size
        k = X.shape[1]  # Number of parameters (2: intercept + slope)

        # Mean squared error: MSE = Σε²/(n-k)
        mse = np.sum(residuals**2) / (n - k)

        # Variance-covariance matrix: Var(β) = MSE * (X'X)^{-1}
        var_beta = mse * np.linalg.inv(X.T @ X)

        # ADF t-statistic: t_β = β[1] / SE(β[1])
        # More negative = more evidence against unit root
        adf_stat = beta[1] / np.sqrt(var_beta[1, 1])

        # Approximate p-value based on critical values
        # In practice, use statsmodels for exact p-values from Dickey-Fuller distribution
        # The distribution is non-standard (not t), so we use critical value approximations
        if adf_stat < self.critical_values["1%"]:
            p_value = 0.01
            is_stationary = True
        elif adf_stat < self.critical_values["5%"]:
            p_value = 0.05
            is_stationary = True
        elif adf_stat < self.critical_values["10%"]:
            p_value = 0.10
            is_stationary = True
        else:
            p_value = 0.20
            is_stationary = False

        return adf_stat, p_value, is_stationary

    def engle_granger_test(self, y: np.ndarray, x: np.ndarray) -> dict:
        """
        Engle-Granger two-step cointegration test.

        Args:
            y: First time series (dependent variable)
            x: Second time series (independent variable)

        Returns:
            Dictionary with test results
        """
        # Step 1: Regress y on x
        X = np.column_stack([np.ones(len(x)), x])
        beta = np.linalg.lstsq(X, y, rcond=None)[0]

        alpha = beta[0]  # Intercept
        beta_coef = beta[1]  # Slope (hedge ratio)

        # Calculate residuals (spread)
        residuals = y - (alpha + beta_coef * x)

        # Step 2: Test residuals for stationarity
        adf_stat, p_value, is_stationary = self.adf_test(residuals)

        # Calculate half-life of mean reversion
        # From OU process: dS = -θ(S - μ)dt + σdW
        # Half-life = ln(2) / θ
        # θ ≈ -ln(ρ) where ρ is AR(1) coefficient

        if len(residuals) > 1:
            # AR(1) regression: ε_t = ρ*ε_{t-1} + η_t  (autocorrelation of spread)
            lagged = residuals[:-1]
            current = residuals[1:]

            rho = (
                np.corrcoef(lagged, current)[0, 1] if np.std(lagged) > 0 else 0
            )  # AR(1) coefficient

            if rho < 1 and rho > 0:
                theta = -np.log(rho)  # Mean-reversion speed from OU process
                half_life = (
                    np.log(2) / theta if theta > 0 else np.inf
                )  # t_half = ln(2)/θ
            else:
                half_life = np.inf
        else:
            half_life = np.inf
            rho = 0

        return {
            "cointegrated": is_stationary,
            "adf_statistic": adf_stat,
            "p_value": p_value,
            "alpha": alpha,
            "beta": beta_coef,
            "hedge_ratio": beta_coef,
            "residuals": residuals,
            "half_life": half_life,
            "ar1_coefficient": rho,
            "correlation": np.corrcoef(y, x)[0, 1],
        }

    def find_cointegrated_pairs(self, prices_df: pd.DataFrame) -> pd.DataFrame:
        """
        Find all cointegrated pairs in a dataframe of prices.

        Args:
            prices_df: DataFrame with asset prices as columns

        Returns:
            DataFrame with cointegration results
        """
        assets = prices_df.columns
        n = len(assets)
        results = []

        for i in range(n):
            for j in range(i + 1, n):
                asset1 = assets[i]
                asset2 = assets[j]

                y = prices_df[asset1].values
                x = prices_df[asset2].values

                # Remove NaN
                mask = ~(np.isnan(y) | np.isnan(x))
                y = y[mask]
                x = x[mask]

                if len(y) < 30:  # Need sufficient data
                    continue

                result = self.engle_granger_test(y, x)

                results.append(
                    {
                        "asset1": asset1,
                        "asset2": asset2,
                        "cointegrated": result["cointegrated"],
                        "adf_statistic": result["adf_statistic"],
                        "p_value": result["p_value"],
                        "hedge_ratio": result["hedge_ratio"],
                        "half_life": result["half_life"],
                        "correlation": result["correlation"],
                    }
                )

        return pd.DataFrame(results)


class PairsTradingStrategy:
    """
    Pairs trading strategy using cointegration.

    Strategy:
    1. Calculate spread: S = Asset1 - β*Asset2 - α
    2. When spread > upper threshold: Short Asset1, Long Asset2
    3. When spread < lower threshold: Long Asset1, Short Asset2
    4. Exit when spread returns to mean
    """

    def __init__(self, entry_zscore: float = 2.0, exit_zscore: float = 0.5):
        """
        Initialize pairs trading strategy.

        Args:
            entry_zscore: Z-score threshold for entry
            exit_zscore: Z-score threshold for exit
        """
        self.entry_zscore = entry_zscore
        self.exit_zscore = exit_zscore

        self.alpha = 0.0
        self.beta = 1.0
        self.spread_mean = 0.0
        self.spread_std = 1.0
        self.is_trained = False

    def fit(self, asset1: np.ndarray, asset2: np.ndarray):
        """
        Calibrate strategy on historical data.

        Args:
            asset1: Historical prices of first asset
            asset2: Historical prices of second asset
        """
        # Calculate cointegration
        coint_test = CointegrationTest()
        result = coint_test.engle_granger_test(asset1, asset2)

        self.alpha = result["alpha"]
        self.beta = result["beta"]

        # Calculate spread statistics
        spread = result["residuals"]
        self.spread_mean = np.mean(spread)
        self.spread_std = np.std(spread)

        self.is_trained = True

    def get_spread(self, asset1: float, asset2: float) -> float:
        """Calculate current spread."""
        return asset1 - self.beta * asset2 - self.alpha

    def get_zscore(self, asset1: float, asset2: float) -> float:
        """Calculate z-score of spread."""
        spread = self.get_spread(asset1, asset2)
        return (
            spread - self.spread_mean
        ) / self.spread_std  # Standardize spread: how many σ from mean

    def generate_signal(self, asset1: float, asset2: float) -> int:
        """
        Generate trading signal.

        Returns:
            -1: Short spread (Short Asset1, Long Asset2)
             0: No trade
             1: Long spread (Long Asset1, Short Asset2)
        """
        if not self.is_trained:
            return 0

        zscore = self.get_zscore(asset1, asset2)

        if zscore > self.entry_zscore:
            return -1  # Short spread
        elif zscore < -self.entry_zscore:
            return 1  # Long spread
        else:
            return 0  # No trade

    def should_exit(self, asset1: float, asset2: float, position: int) -> bool:
        """Check if position should be closed."""
        zscore = self.get_zscore(asset1, asset2)

        if position == 1 and zscore >= -self.exit_zscore:
            return True
        elif position == -1 and zscore <= self.exit_zscore:
            return True

        return False


class StatisticalArbitragePortfolio:
    """
    Portfolio of multiple cointegrated pairs.

    Manages multiple pairs trades with position sizing.
    """

    def __init__(self, max_pairs: int = 5, capital_per_pair: float = 10000.0):
        """
        Initialize statistical arbitrage portfolio.

        Args:
            max_pairs: Maximum number of pairs to trade
            capital_per_pair: Capital allocated per pair
        """
        self.max_pairs = max_pairs
        self.capital_per_pair = capital_per_pair
        self.pairs = {}  # Dictionary of PairsTradingStrategy
        self.positions = {}  # Current positions
        self.active_pairs = []

    def select_pairs(self, prices_df: pd.DataFrame, min_correlation: float = 0.8):
        """
        Select best cointegrated pairs from universe.

        Args:
            prices_df: DataFrame with asset prices
            min_correlation: Minimum correlation threshold
        """
        # Find all cointegrated pairs
        coint_test = CointegrationTest()
        results = coint_test.find_cointegrated_pairs(prices_df)

        # Filter by correlation and sort by half-life
        valid_pairs = results[
            (results["cointegrated"] == True)
            & (results["correlation"].abs() >= min_correlation)
        ].sort_values("half_life")

        # Select top pairs
        selected = valid_pairs.head(self.max_pairs)

        # Create strategies for selected pairs
        for _, row in selected.iterrows():
            pair_key = f"{row['asset1']}_{row['asset2']}"

            strategy = PairsTradingStrategy()
            strategy.fit(
                prices_df[row["asset1"]].values, prices_df[row["asset2"]].values
            )

            self.pairs[pair_key] = {
                "strategy": strategy,
                "asset1": row["asset1"],
                "asset2": row["asset2"],
                "half_life": row["half_life"],
            }
            self.positions[pair_key] = 0

        self.active_pairs = list(self.pairs.keys())

        print(f"Selected {len(self.active_pairs)} pairs for trading")
        for pair in self.active_pairs:
            info = self.pairs[pair]
            print(f"  {pair}: half-life={info['half_life']:.1f} bars")

    def update(self, prices: dict) -> dict:
        """
        Update all pairs with new prices and generate signals.

        Args:
            prices: Dictionary of current prices {asset: price}

        Returns:
            Dictionary of signals {pair: signal}
        """
        signals = {}

        for pair_key, pair_info in self.pairs.items():
            strategy = pair_info["strategy"]
            asset1 = pair_info["asset1"]
            asset2 = pair_info["asset2"]

            if asset1 not in prices or asset2 not in prices:
                continue

            price1 = prices[asset1]
            price2 = prices[asset2]
            current_pos = self.positions[pair_key]

            # Check for exit
            if current_pos != 0 and strategy.should_exit(price1, price2, current_pos):
                signals[pair_key] = 0  # Exit signal
                self.positions[pair_key] = 0

            # Check for entry
            elif current_pos == 0:
                signal = strategy.generate_signal(price1, price2)
                if signal != 0:
                    signals[pair_key] = signal
                    self.positions[pair_key] = signal

        return signals


# Utility functions
def calculate_hedge_ratio(asset1: pd.Series, asset2: pd.Series) -> float:
    """
    Calculate static hedge ratio using OLS regression.

    Args:
        asset1: First asset prices
        asset2: Second asset prices

    Returns:
        Hedge ratio (beta)
    """
    X = np.column_stack([np.ones(len(asset2)), asset2.values])
    beta = np.linalg.lstsq(X, asset1.values, rcond=None)[0]
    return beta[1]


def calculate_spread(
    asset1: pd.Series, asset2: pd.Series, hedge_ratio: Optional[float] = None
) -> pd.Series:
    """
    Calculate spread between two assets.

    Args:
        asset1: First asset
        asset2: Second asset
        hedge_ratio: Optional hedge ratio (calculated if None)

    Returns:
        Spread series
    """
    if hedge_ratio is None:
        hedge_ratio = calculate_hedge_ratio(asset1, asset2)

    spread = asset1 - hedge_ratio * asset2
    return spread


def find_best_pairs(prices_df: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    """
    Find best cointegrated pairs in a price universe.

    Args:
        prices_df: DataFrame with asset prices
        top_n: Number of top pairs to return

    Returns:
        DataFrame with best pairs
    """
    coint_test = CointegrationTest()
    results = coint_test.find_cointegrated_pairs(prices_df)

    # Filter and rank
    best = (
        results[
            (results["cointegrated"] == True)
            & (results["half_life"] > 0)
            & (results["half_life"] < 100)  # Reasonable half-life
        ]
        .sort_values(["half_life", "correlation"], ascending=[True, False])
        .head(top_n)
    )

    return best
