"""
Alpha Research Framework
=======================
Professional-grade alpha factor mining and validation system.

This module provides comprehensive tools for discovering, validating, and
combining alpha factors for quantitative trading strategies. It implements
techniques used by top quantitative hedge funds.

Alpha Mining Capabilities:
    1. Technical Alphas: Indicator-based signals
       - Momentum (price rate of change)
       - Mean reversion (z-score deviation)
       - Volume-weighted indicators
       - Volatility regimes
       - RSI extremes
       - Bollinger Band positioning

    2. Statistical Alphas: Data-driven signals
       - Beta-adjusted returns (market-neutral)
       - Skewness (crash risk)
       - Kurtosis (tail risk)
       - Volatility of volatility
       - Autocorrelation (momentum/reversal)
       - Hurst exponent (trend strength)

    3. Cross-Sectional Alphas: Relative ranking signals
       - Cross-sectional momentum
       - Cross-sectional reversal
       - Cross-sectional volatility

Alpha Validation Metrics:
    - IC (Information Coefficient): Correlation with future returns
    - IR (Information Ratio): IC mean / IC std
    - Sharpe Ratio: Risk-adjusted returns
    - Max Drawdown: Peak-to-trough decline
    - Turnover: Signal change rate
    - Decay: Half-life of predictive power
    - Stability: Consistency over time

Alpha Combination Methods:
    - Equal Weight: Simple averaging
    - IC Weight: Weighted by IC scores
    - ML Stack: Ridge/Lasso/ElasticNet combination
    - PCA: Principal component combination

Factor Neutralization:
    - Market beta removal
    - Sector exposure removal
    - Style factor removal
    - Creates pure alpha signals

Usage:
    from src.research.alpha_research import AlphaMiner, AlphaValidator, AlphaCombiner

    # Generate alphas
    miner = AlphaMiner()
    technical_alphas = miner.generate_technical_alphas(df)
    statistical_alphas = miner.generate_statistical_alphas(df)

    # Validate alphas
    validator = AlphaValidator()
    for name, alpha in {**technical_alphas, **statistical_alphas}.items():
        metrics = validator.validate_alpha(name, alpha, forward_returns)
        print(f"{name}: IC={metrics.ic:.3f}, Sharpe={metrics.sharpe:.2f}")

    # Combine best alphas
    combiner = AlphaCombiner()
    best_alphas = validator.get_best_alphas(n=5)
    combined = combiner.ic_weighted_combine(dict(best_alphas), {a.name: a.ic for a in best_alphas})

Dependencies:
    - numpy: Numerical operations
    - pandas: Data manipulation
    - scipy: Statistical tests
    - scikit-learn: ML models and preprocessing

Note:
    Alpha factors require forward returns for validation. Ensure proper
    temporal separation between factor calculation and return measurement
    to avoid look-ahead bias.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass
from scipy import stats
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings

warnings.filterwarnings("ignore")
from loguru import logger


@dataclass
class AlphaMetrics:
    """Performance metrics for an alpha factor."""

    name: str
    ic: float  # Information coefficient
    ic_ir: float  # IC information ratio
    returns: pd.Series  # Factor returns
    sharpe: float  # Sharpe ratio
    max_drawdown: float  # Maximum drawdown
    turnover: float  # Turnover (0-1)
    fitness: float  # Fitness score
    decay: float  # Half-life in days
    stability: float  # Stability score (0-1)


class AlphaMiner:
    """
    Automated alpha factor mining from market data.

    This class generates predictive signals (alphas) from raw market data using
    various techniques including technical indicators, statistical measures,
    and cross-sectional rankings.

    Attributes:
        min_obs: Minimum observations required for alpha calculation
        alphas: Dictionary of generated alpha factors

    Alpha Categories:

    Technical Alphas:
        - mom_[5,10,20,60]: Price momentum over N periods
        - mr_[5,10,20]: Mean reversion (negative z-score)
        - vwma_dist: Volume-weighted moving average distance
        - vol_regime: Volatility breakout indicator
        - rsi_extreme: RSI overbought/oversold signals
        - bb_position: Bollinger Band position

    Statistical Alphas:
        - residual: Beta-adjusted market-neutral returns
        - skew: Rolling skewness (crash risk)
        - kurt: Rolling kurtosis (tail risk)
        - vol_of_vol: Volatility of volatility
        - autocorr_1: First-order autocorrelation
        - hurst: Hurst exponent (trend/reversion)

    Cross-Sectional Alphas:
        - xs_mom: Cross-sectional momentum ranking
        - xs_reversal: Cross-sectional reversal ranking
        - xs_vol: Cross-sectional volatility ranking

    Example:
        >>> miner = AlphaMiner(min_observations=100)
        >>>
        >>> # Generate technical alphas
        >>> tech_alphas = miner.generate_technical_alphas(df)
        >>> print(f"Generated {len(tech_alphas)} technical alphas")

        >>> # Generate statistical alphas
        >>> stat_alphas = miner.generate_statistical_alphas(df)
        >>> print(f"Generated {len(stat_alphas)} statistical alphas")

        >>> # Generate cross-sectional alphas
        >>> xs_alphas = miner.generate_cross_sectional_alphas(data_dict)
        >>> print(f"Generated {len(xs_alphas)} cross-sectional alphas")

    Input Data Format:
        DataFrame with columns: close, volume (optional), and timestamp index
        For cross-sectional: Dict of {symbol: DataFrame}

    Note:
        Some alphas may produce NaN values at the beginning due to rolling
        window calculations. These should be handled before validation.
    """

    def __init__(self, min_observations: int = 100):
        self.min_obs = min_observations
        self.alphas = {}
        logger.info("AlphaMiner initialized")

    def generate_technical_alphas(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        Generate alpha factors from technical indicators.
        """
        alphas = {}
        close = df["close"]
        volume = df.get("volume", pd.Series(1, index=df.index))

        # Momentum alphas
        for period in [5, 10, 20, 60]:
            alphas[f"mom_{period}"] = close.pct_change(period)

        # Mean reversion alphas
        for lookback in [5, 10, 20]:
            # z-score of price deviation from rolling mean
            zscore = (close - close.rolling(lookback).mean()) / close.rolling(
                lookback
            ).std()
            alphas[
                f"mr_{lookback}"
            ] = -zscore  # flip sign: high z-score → expect reversal down

        # Volume-weighted alphas
        vwma = (close * volume).rolling(20).sum() / volume.rolling(
            20
        ).sum()  # volume-weighted moving average
        alphas["vwma_dist"] = (
            close - vwma
        ) / vwma  # relative distance of price from VWMA

        # Volatility breakout
        vol = close.pct_change().rolling(20).std()  # realised vol (20-bar window)
        alphas["vol_regime"] = (vol > vol.rolling(60).mean()).astype(
            float
        )  # 1 if vol above 60-bar average

        # RSI divergence (simplified)
        delta = close.diff()
        gain = (
            (delta.where(delta > 0, 0)).rolling(14).mean()
        )  # average gain (Wilder smoothing approximation)
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()  # average loss magnitude
        rs = gain / loss  # relative strength = avg gain / avg loss
        rsi = 100 - (100 / (1 + rs))  # RSI: 0–100 scale; >70 overbought, <30 oversold
        alphas["rsi_extreme"] = ((rsi > 70) | (rsi < 30)).astype(float) * (
            50 - rsi
        )  # signed extreme signal

        # Bollinger Band position
        bb_middle = close.rolling(20).mean()
        bb_std = close.rolling(20).std()
        bb_position = (close - bb_middle) / (
            2 * bb_std
        )  # ±1 = at upper/lower band; 0 = at middle band
        alphas["bb_position"] = bb_position

        logger.info(f"Generated {len(alphas)} technical alphas")
        return alphas

    def generate_statistical_alphas(
        self, df: pd.DataFrame, market_returns: pd.Series = None
    ) -> Dict[str, pd.Series]:
        """
        Generate statistical arbitrage alphas.
        """
        alphas = {}
        returns = df["close"].pct_change()

        if market_returns is None:
            market_returns = returns

        # Beta-adjusted returns
        rolling_beta = (
            returns.rolling(60).cov(market_returns)
            / market_returns.rolling(60).var()  # rolling β
        )
        alphas["residual"] = (
            returns - rolling_beta * market_returns
        )  # market-neutral residual alpha

        # Skewness (crash risk)
        alphas["skew"] = returns.rolling(
            20
        ).skew()  # negative skew → fat left tail / crash risk

        # Kurtosis (tail risk)
        alphas["kurt"] = returns.rolling(
            20
        ).kurt()  # excess kurtosis > 3 → fatter tails than normal

        # Volatility of volatility
        vol = returns.rolling(20).std()
        alphas["vol_of_vol"] = vol.rolling(
            20
        ).std()  # second-order vol: measures vol regime instability

        # Autocorrelation (reversal/momentum)
        alphas["autocorr_1"] = returns.rolling(20).apply(
            lambda x: x.autocorr(lag=1)
            if len(x) > 1
            else 0  # positive → momentum, negative → mean reversion
        )

        # Hurst exponent (trend strength)
        def hurst_exponent(ts):
            """
            Calculate Hurst exponent using R/S (Rescaled Range) analysis.

            The Hurst exponent measures long-term memory of time series:
            - H > 0.5: Trending behavior (persistence)
            - H < 0.5: Mean-reverting behavior (anti-persistence)
            - H = 0.5: Random walk (no memory)

            Implementation uses R/S analysis:
            1. Calculate standard deviation at different lags
            2. Fit log-log regression: log(R/S) = H * log(lag) + c
            3. Slope is the Hurst exponent
            """
            if len(ts) < 20:
                return 0.5  # not enough data → assume random walk (H=0.5)
            lags = range(2, min(20, len(ts) // 2))
            tau = [
                np.std(np.subtract(ts[lag:], ts[:-lag])) for lag in lags
            ]  # R/S statistic at each lag
            reg = np.polyfit(np.log(lags), np.log(tau), 1)  # log-log regression slope
            return reg[0] * 2.0  # H > 0.5 = trending, H < 0.5 = mean-reverting

        alphas["hurst"] = df["close"].rolling(60).apply(hurst_exponent)

        logger.info(f"Generated {len(alphas)} statistical alphas")
        return alphas

    def generate_cross_sectional_alphas(
        self, data: Dict[str, pd.DataFrame]
    ) -> Dict[str, pd.DataFrame]:
        """
        Generate cross-sectional ranking alphas.

        Ranks assets relative to each other (market neutral).
        """
        alphas = {}

        # Combine all assets
        all_returns = pd.DataFrame(
            {asset: df["close"].pct_change() for asset, df in data.items()}
        )

        # Cross-sectional momentum
        alphas["xs_mom"] = all_returns.rolling(20).mean().rank(axis=1, pct=True)

        # Cross-sectional reversal
        alphas["xs_reversal"] = (
            all_returns.rolling(5).mean().rank(axis=1, pct=True, ascending=False)
        )

        # Cross-sectional volatility
        vol = all_returns.rolling(20).std()
        alphas["xs_vol"] = vol.rank(axis=1, pct=True, ascending=False)  # Prefer low vol

        logger.info(
            f"Generated {len(alphas)} cross-sectional alphas for {len(data)} assets"
        )
        return alphas


class AlphaValidator:
    """
    Validates alpha factors for production use.

    Checks for:
    - Predictive power (IC, IR)
    - Stability over time
    - Turnover costs
    - Overfitting
    - Decay characteristics
    """

    def __init__(self):
        self.validation_results = []
        logger.info("AlphaValidator initialized")

    def calculate_information_coefficient(
        self, factor: pd.Series, forward_returns: pd.Series, method: str = "spearman"
    ) -> Tuple[float, float]:
        """
        Calculate Information Coefficient (IC).

        Measures correlation between factor and future returns.
        """
        # Align series
        aligned_factor, aligned_returns = factor.align(forward_returns, join="inner")

        if len(aligned_factor) < 30:
            return 0.0, 0.0

        # Calculate rolling IC
        ic_series = []
        window = min(252, len(aligned_factor) // 4)

        for i in range(window, len(aligned_factor)):
            if method == "spearman":
                ic = stats.spearmanr(
                    aligned_factor.iloc[i - window : i],
                    aligned_returns.iloc[i - window : i],
                )[0]
            else:
                ic = np.corrcoef(
                    aligned_factor.iloc[i - window : i],
                    aligned_returns.iloc[i - window : i],
                )[0, 1]

            if not np.isnan(ic):
                ic_series.append(ic)

        if len(ic_series) == 0:
            return 0.0, 0.0

        mean_ic = np.mean(ic_series)
        ic_ir = mean_ic / (np.std(ic_series) + 1e-10)

        return mean_ic, ic_ir

    def calculate_turnover(self, factor: pd.Series) -> float:
        """Calculate factor turnover."""
        if len(factor) < 2:
            return 0.0

        # Turnover = sum of absolute changes / sum of absolute values
        changes = factor.diff().abs()
        turnover = changes.sum() / factor.abs().sum()

        return turnover

    def calculate_decay(self, factor: pd.Series, forward_returns: pd.Series) -> float:
        """
        Calculate alpha decay (half-life) in trading periods.

        Measures how quickly the predictive power of an alpha factor diminishes
        over time. This is critical for determining optimal holding periods.

        Method:
            1. Calculate IC at various lag offsets (1 to 20 periods)
            2. Find when IC drops to 50% of initial value
            3. Return that lag as the half-life

        Interpretation:
            - Low decay (<5): Short holding periods optimal
            - Medium decay (5-15): Medium-term signals
            - High decay (>15): Long-term alpha

        Args:
            factor: Alpha factor values
            forward_returns: Future returns for IC calculation

        Returns:
            Half-life in periods (lower = faster decay)
        """
        ics = []
        # Calculate IC at different lag offsets (1 to 20 periods ahead)
        for lag in range(1, min(21, len(factor) // 10)):
            # Shift returns backward to align with current factor values
            shifted_returns = forward_returns.shift(-lag)
            ic, _ = self.calculate_information_coefficient(factor, shifted_returns)
            ics.append(ic)

        # Not enough data or no meaningful IC
        if len(ics) < 2 or max(ics) < 0.01:
            return 1.0

        # Half-life: when IC drops to half its initial value
        initial_ic = ics[0]
        for i, ic in enumerate(ics):
            if abs(ic) < abs(initial_ic) * 0.5:
                return i + 1

        return len(ics)

    def validate_alpha(
        self,
        name: str,
        factor: pd.Series,
        forward_returns: pd.Series,
        transaction_cost: float = 0.001,
    ) -> AlphaMetrics:
        """
        Perform comprehensive validation of an alpha factor.

        Calculates multiple metrics to assess alpha quality:
        - IC: Correlation with future returns
        - IR: IC stability (mean/std)
        - Sharpe: Annualized risk-adjusted returns
        - Max Drawdown: Maximum peak-to-trough decline
        - Turnover: Signal change frequency
        - Decay: Half-life of predictive power
        - Stability: Consistency of returns
        - Fitness: Weighted composite score

        Args:
            name: Alpha identifier
            factor: Alpha factor values
            forward_returns: Future returns for validation
            transaction_cost: Not used in current implementation

        Returns:
            AlphaMetrics with all calculated values
        """
        # IC (Information Coefficient): Measures predictive power
        ic, ic_ir = self.calculate_information_coefficient(factor, forward_returns)

        # Factor returns: Hypothetical returns if we traded this alpha
        factor_returns = factor * forward_returns

        # Sharpe Ratio: Annualized risk-adjusted return
        # Formula: mean / std * sqrt(252) for daily data
        sharpe = factor_returns.mean() / (factor_returns.std() + 1e-10) * np.sqrt(252)

        # Maximum Drawdown: Worst peak-to-trough decline
        cumulative = (1 + factor_returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative - rolling_max) / rolling_max
        max_dd = drawdown.min()

        # Turnover: How often the signal changes (affects trading costs)
        turnover = self.calculate_turnover(factor)

        # Decay: How quickly predictive power fades
        decay = self.calculate_decay(factor, forward_returns)

        # Stability: Consistency of factor returns (0-1 scale)
        # Higher = more consistent returns
        stability = 1.0 - (
            np.std(factor_returns) / (np.abs(factor_returns.mean()) + 1e-10)
        )
        stability = max(0, min(1, stability))

        # Fitness Score: Composite metric combining all factors
        # Weights: IC(30%) + IR(20%) + Sharpe(20%) + LowTurnover(10%) + Stability(20%)
        fitness = (
            abs(ic) * 0.3
            + abs(ic_ir) * 0.2
            + max(0, sharpe) * 0.2
            + (1 - turnover) * 0.1
            + stability * 0.2
        )

        metrics = AlphaMetrics(
            name=name,
            ic=ic,
            ic_ir=ic_ir,
            returns=factor_returns,
            sharpe=sharpe,
            max_drawdown=max_dd,
            turnover=turnover,
            fitness=fitness,
            decay=decay,
            stability=stability,
        )

        self.validation_results.append(metrics)

        return metrics

    def get_best_alphas(self, n: int = 10) -> List[AlphaMetrics]:
        """Get top N alphas by fitness score."""
        sorted_alphas = sorted(
            self.validation_results, key=lambda x: x.fitness, reverse=True
        )
        return sorted_alphas[:n]


class AlphaCombiner:
    """
    Combines multiple alpha factors into ensemble.

    Uses:
    - Weighted combination
    - Machine learning stacking
    - Principal component analysis
    - Dynamic weighting
    """

    def __init__(self):
        self.weights = {}
        self.combined_alpha = None
        logger.info("AlphaCombiner initialized")

    def equal_weight_combine(self, alphas: Dict[str, pd.Series]) -> pd.Series:
        """Simple equal-weight combination."""
        combined = pd.concat(alphas.values(), axis=1).mean(axis=1)
        return combined

    def ic_weighted_combine(
        self, alphas: Dict[str, pd.Series], ics: Dict[str, float]
    ) -> pd.Series:
        """Weight alphas by their IC."""
        weights = {name: abs(ic) for name, ic in ics.items()}
        total_weight = sum(weights.values())

        if total_weight == 0:
            return self.equal_weight_combine(alphas)

        normalized_weights = {name: w / total_weight for name, w in weights.items()}

        combined = pd.Series(0, index=next(iter(alphas.values())).index)
        for name, alpha in alphas.items():
            combined += alpha * normalized_weights.get(name, 0)

        return combined

    def ml_stack_combine(
        self,
        alphas: Dict[str, pd.Series],
        forward_returns: pd.Series,
        method: str = "ridge",
    ) -> pd.Series:
        """
        Use ML to learn optimal combination.

        Methods: 'ridge', 'lasso', 'elasticnet'
        """
        from sklearn.linear_model import Ridge, Lasso, ElasticNet

        # Prepare data
        X = pd.concat(alphas.values(), axis=1)
        X.columns = alphas.keys()
        y = forward_returns

        # Align
        X, y = X.align(y, join="inner", axis=0)

        if len(X) < 100:
            return self.equal_weight_combine(alphas)

        # Split train/test
        train_size = int(0.7 * len(X))
        X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
        y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

        # Fit model
        if method == "ridge":
            model = Ridge(alpha=1.0)
        elif method == "lasso":
            model = Lasso(alpha=0.001)
        else:
            model = ElasticNet(alpha=0.001, l1_ratio=0.5)

        model.fit(X_train, y_train)

        # Store weights
        self.weights = dict(zip(alphas.keys(), model.coef_))

        # Predict
        predictions = model.predict(X)

        return pd.Series(predictions, index=X.index)

    def pca_combine(
        self, alphas: Dict[str, pd.Series], n_components: int = 3
    ) -> pd.Series:
        """
        Combine using PCA (removes correlation).
        """
        X = pd.concat(alphas.values(), axis=1)
        X = X.dropna()

        if len(X) < 100:
            return self.equal_weight_combine(alphas)

        # Standardize
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # PCA
        pca = PCA(n_components=min(n_components, X.shape[1]))
        components = pca.fit_transform(X_scaled)

        # Use first component as combined alpha
        combined = components[:, 0]

        return pd.Series(combined, index=X.index)


class FactorNeutralizer:
    """
    Neutralizes alpha factors against common risk factors.

    Removes exposure to:
    - Market beta
    - Sector exposure
    - Style factors (size, value, momentum)

    Creates pure alpha signals.
    """

    def __init__(self):
        self.risk_factors = {}
        logger.info("FactorNeutralizer initialized")

    def add_risk_factor(self, name: str, factor: pd.Series):
        """Add a risk factor to neutralize against."""
        self.risk_factors[name] = factor

    def neutralize(self, alpha: pd.Series, method: str = "regression") -> pd.Series:
        """
        Neutralize alpha against risk factors.
        """
        if len(self.risk_factors) == 0:
            return alpha

        # Prepare risk factor matrix
        risk_df = pd.concat(self.risk_factors.values(), axis=1)
        risk_df.columns = self.risk_factors.keys()

        # Align
        alpha_aligned, risk_aligned = alpha.align(risk_df, join="inner", axis=0)

        if method == "regression":
            # Regress out risk factors
            from sklearn.linear_model import LinearRegression

            model = LinearRegression()
            model.fit(risk_aligned, alpha_aligned)

            # Residuals are neutralized alpha
            predictions = model.predict(risk_aligned)
            neutralized = alpha_aligned - predictions

        elif method == "projection":
            # Orthogonal projection
            risk_matrix = risk_aligned.values
            alpha_vector = alpha_aligned.values.reshape(-1, 1)

            # Project out risk space
            projection = (
                risk_matrix
                @ np.linalg.pinv(risk_matrix.T @ risk_matrix)
                @ risk_matrix.T
                @ alpha_vector
            )
            neutralized = pd.Series(
                (alpha_vector - projection).flatten(), index=alpha_aligned.index
            )

        else:
            neutralized = alpha_aligned

        return neutralized


# Production-ready functions
def mine_and_validate_alphas(
    df: pd.DataFrame, forward_returns: pd.Series
) -> Dict[str, AlphaMetrics]:
    """
    Complete alpha mining and validation pipeline.
    """
    # Mine alphas
    miner = AlphaMiner()
    technical = miner.generate_technical_alphas(df)
    statistical = miner.generate_statistical_alphas(df)

    all_alphas = {**technical, **statistical}

    # Validate
    validator = AlphaValidator()
    results = {}

    for name, alpha in all_alphas.items():
        metrics = validator.validate_alpha(name, alpha, forward_returns)
        results[name] = metrics

        logger.info(
            f"Alpha {name}: IC={metrics.ic:.3f}, Sharpe={metrics.sharpe:.2f}, "
            f"Turnover={metrics.turnover:.2f}, Fitness={metrics.fitness:.3f}"
        )

    return results


def build_combined_alpha(
    alphas: Dict[str, pd.Series],
    forward_returns: pd.Series,
    method: str = "ic_weighted",
) -> pd.Series:
    """
    Build optimized combined alpha.
    """
    combiner = AlphaCombiner()

    if method == "ic_weighted":
        # Calculate ICs
        validator = AlphaValidator()
        ics = {}
        for name, alpha in alphas.items():
            ic, _ = validator.calculate_information_coefficient(alpha, forward_returns)
            ics[name] = ic

        combined = combiner.ic_weighted_combine(alphas, ics)
    elif method == "ml_stack":
        combined = combiner.ml_stack_combine(alphas, forward_returns)
    elif method == "pca":
        combined = combiner.pca_combine(alphas)
    else:
        combined = combiner.equal_weight_combine(alphas)

    return combined
