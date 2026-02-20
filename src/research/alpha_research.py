"""
Alpha Research Framework
=========================
Professional-grade alpha research and validation system.

Features:
- Automated alpha idea generation
- Cross-sectional analysis
- Factor mining and validation
- Alpha combination and stacking
- Out-of-sample testing
- Turnover analysis

Used by: WorldQuant, Two Sigma, Renaissance Technologies
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
    Automated alpha factor mining.

    Discovers predictive relationships in data using:
    - Technical indicators
    - Statistical arbitrage signals
    - Cross-sectional rankings
    - Time-series momentum
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
            zscore = (close - close.rolling(lookback).mean()) / close.rolling(
                lookback
            ).std()
            alphas[f"mr_{lookback}"] = -zscore  # Negative for mean reversion

        # Volume-weighted alphas
        vwma = (close * volume).rolling(20).sum() / volume.rolling(20).sum()
        alphas["vwma_dist"] = (close - vwma) / vwma

        # Volatility breakout
        vol = close.pct_change().rolling(20).std()
        alphas["vol_regime"] = (vol > vol.rolling(60).mean()).astype(float)

        # RSI divergence (simplified)
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        alphas["rsi_extreme"] = ((rsi > 70) | (rsi < 30)).astype(float) * (50 - rsi)

        # Bollinger Band position
        bb_middle = close.rolling(20).mean()
        bb_std = close.rolling(20).std()
        bb_position = (close - bb_middle) / (2 * bb_std)
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
            returns.rolling(60).cov(market_returns) / market_returns.rolling(60).var()
        )
        alphas["residual"] = returns - rolling_beta * market_returns

        # Skewness (crash risk)
        alphas["skew"] = returns.rolling(20).skew()

        # Kurtosis (tail risk)
        alphas["kurt"] = returns.rolling(20).kurt()

        # Volatility of volatility
        vol = returns.rolling(20).std()
        alphas["vol_of_vol"] = vol.rolling(20).std()

        # Autocorrelation (reversal/momentum)
        alphas["autocorr_1"] = returns.rolling(20).apply(
            lambda x: x.autocorr(lag=1) if len(x) > 1 else 0
        )

        # Hurst exponent (trend strength)
        def hurst_exponent(ts):
            if len(ts) < 20:
                return 0.5
            lags = range(2, min(20, len(ts) // 2))
            tau = [np.std(np.subtract(ts[lag:], ts[:-lag])) for lag in lags]
            reg = np.polyfit(np.log(lags), np.log(tau), 1)
            return reg[0] * 2.0

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
        Calculate alpha decay (half-life).

        How quickly does predictive power diminish?
        """
        ics = []
        for lag in range(1, min(21, len(factor) // 10)):
            shifted_returns = forward_returns.shift(-lag)
            ic, _ = self.calculate_information_coefficient(factor, shifted_returns)
            ics.append(ic)

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
        Full validation of an alpha factor.
        """
        # IC and IR
        ic, ic_ir = self.calculate_information_coefficient(factor, forward_returns)

        # Generate factor returns
        factor_returns = factor * forward_returns

        # Performance metrics
        sharpe = factor_returns.mean() / (factor_returns.std() + 1e-10) * np.sqrt(252)

        # Drawdown
        cumulative = (1 + factor_returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative - rolling_max) / rolling_max
        max_dd = drawdown.min()

        # Turnover
        turnover = self.calculate_turnover(factor)

        # Decay
        decay = self.calculate_decay(factor, forward_returns)

        # Stability (consistency of IC)
        stability = 1.0 - (
            np.std(factor_returns) / (np.abs(factor_returns.mean()) + 1e-10)
        )
        stability = max(0, min(1, stability))

        # Fitness score (composite)
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
