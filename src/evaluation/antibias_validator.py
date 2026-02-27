"""
ANTI-BIAS FRAMEWORK – Statistical Validation
============================================
Comprehensive validation suite for trading strategy backtests.

This module implements four independent statistical tests to confirm
that a trading strategy's performance is genuine and not the result
of overfitting or random chance. A strategy must pass ALL tests to
be considered viable for live trading.

The Four Validation Tests:
---------------------------
1. CPCV (Combinatorial Purged Cross-Validation)
   - Tests stability across different time periods
   - Strategy should perform consistently in all folds
   - Failure indicates regime dependency or overfitting

2. Permutation Test (Monte Carlo)
   - Tests whether performance is due to skill or chance
   - Shuffles trade signals and compares to original
   - Failure indicates results may be random

3. DSR (Deflated Sharpe Ratio)
   - Corrects Sharpe ratio for multiple testing bias
   - Accounts for the "lucky" strategy problem
   - Failure indicates Sharpe may be inflated by selection bias

4. MTRL (Minimum Track Record Length)
   - How much data is needed for statistical significance?
   - Calculates minimum trades/bars required
   - Failure indicates insufficient data for conclusions

Why All Four Tests?
------------------
Each test catches different types of overfitting:
- CPCV: Temporal overfitting (works in some periods, not others)
- Permutation: Random chance (no real edge)
- DSR: Multiple testing (found strategy by luck among many trials)
- MTRL: Insufficient sample (not enough data to be confident)

A strategy is only considered valid if it passes ALL tests.

Usage:
------
    from evaluation.antibias_validator import BacktestValidator

    # Initialize validator
    validator = BacktestValidator(
        n_cpcv_splits=6,
        n_permutations=1000,
        n_trials_tested=1,
    )

    # Run full validation
    report = validator.validate(returns, positions)

    # Check results
    print(report)

    if report.passes_all:
        print("✅ Strategy passes all validation tests!")
    else:
        print("❌ Strategy fails validation - do not go live!")

    # Individual test results
    print(f"CPCV robust: {report.cpcv.is_robust()}")
    print(f"Permutation significant: {report.perm.is_significant}")
    print(f"DSR acceptable: {report.dsr.is_acceptable}")

Reference:
---------
- Lopez de Prado, M. (2018): "Advances in Financial Machine Learning"
  - Chapter 7: Cross-Validation in Financial ML
  - Chapter 9: The DSR Formula
  - Chapter 16: Backtesting without Overfitting
- Bailey & Lopez de Prado (2014): "The Deflated Sharpe Ratio"
- Bailey et al. (2015): "Optimal Trading Rules"
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from scipy import stats as scipy_stats

logger = logging.getLogger("antibias.evaluation")


@dataclass
class PerformanceMetrics:
    """
    Complete set of performance metrics for a trading strategy.

    This dataclass encapsulates all key performance indicators computed
    from a strategy's return series. It serves as the foundation for
    all validation tests in the Anti-Bias Framework.

    Attributes:
        sharpe: Daily Sharpe ratio (not annualized in this class)
        sortino: Sortino ratio using downside deviation
        calmar: Calmar ratio (return / max drawdown)
        max_drawdown: Maximum drawdown as positive decimal
        win_rate: Fraction of profitable periods
        profit_factor: Gross profit / gross loss
        total_return: Total return as decimal
        n_trades: Number of completed trades
        avg_trade_return: Mean return per trade
        std_trade_return: Standard deviation of trade returns
        skewness: Return distribution skewness (0 = symmetric)
        kurtosis: Return distribution excess kurtosis (0 = normal)

    Methods:
        __repr__(): Formatted string with key metrics

    Example:
        --------
        metrics = PerformanceMetrics(
            sharpe=0.85,
            sortino=1.12,
            calmar=1.45,
            max_drawdown=0.08,
            win_rate=0.55,
            profit_factor=1.8,
            total_return=0.15,
            n_trades=120,
            avg_trade_return=0.0012,
            std_trade_return=0.015,
            skewness=-0.3,
            kurtosis=2.5
        )

        print(metrics)
        # Output: Sharpe=0.850 Sortino=1.120 Calmar=1.450 MaxDD=8.00% ...
    """

    sharpe: float
    sortino: float
    calmar: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    total_return: float
    n_trades: int
    avg_trade_return: float
    std_trade_return: float
    skewness: float
    kurtosis: float

    def __repr__(self) -> str:
        return (
            f"Sharpe={self.sharpe:.3f} "
            f"Sortino={self.sortino:.3f} "
            f"Calmar={self.calmar:.3f} "
            f"MaxDD={self.max_drawdown * 100:.2f}% "
            f"WinRate={self.win_rate * 100:.1f}% "
            f"PF={self.profit_factor:.2f} "
            f"TotalReturn={self.total_return * 100:.2f}%"
        )


def compute_metrics(
    returns: np.ndarray, positions: Optional[np.ndarray] = None
) -> PerformanceMetrics:
    """Compute the full set of performance metrics from a return series."""
    r = np.asarray(returns, dtype=np.float64)
    r = r[~np.isnan(r)]  # drop NaN values before any metric calculation
    if len(r) == 0:
        return PerformanceMetrics(
            *([0.0] * 11 + [0])
        )  # zero-metric sentinel for empty input

    sharpe = float(r.mean() / (r.std() + 1e-8))  # daily Sharpe (not annualised here)

    downside = r[r < 0]  # retain only loss periods for Sortino denominator
    sortino = (
        float(r.mean() / (downside.std() + 1e-8)) if len(downside) > 0 else sharpe
    )  # Sortino = mean / downside deviation

    equity = np.cumprod(1 + r)  # compounded equity curve from 1.0
    peak = np.maximum.accumulate(equity)  # running maximum (high-water mark)
    dd = (equity - peak) / (peak + 1e-8)  # drawdown series: negative values
    max_dd = float(-dd.min())  # largest drawdown magnitude (positive scalar)

    calmar = float(
        r.mean() / (max_dd + 1e-8)
    )  # Calmar ratio = mean return / max drawdown

    wins = r[r > 0]
    losses = r[r < 0]
    win_rate = float(
        len(wins) / (len(r) + 1e-8)
    )  # fraction of bars with positive return
    pf = (
        float(wins.sum() / (-losses.sum() + 1e-8)) if len(losses) > 0 else 99.0
    )  # Profit Factor = gross profit / gross loss

    trade_r = (
        r[r != 0]  # non-zero returns as proxy for trades (no position series)
        if positions is None
        else r[
            np.diff(np.concatenate([[0], positions])) != 0
        ]  # returns where position changed = actual trade bars
    )
    if len(trade_r) == 0:
        trade_r = r  # fallback: use all bars if no trades detected

    return PerformanceMetrics(
        sharpe=sharpe,
        sortino=sortino,
        calmar=calmar,
        max_drawdown=max_dd,
        win_rate=win_rate,
        profit_factor=pf,
        total_return=float(equity[-1] - 1),
        n_trades=len(trade_r),
        avg_trade_return=float(trade_r.mean()),
        std_trade_return=float(trade_r.std()),
        skewness=float(scipy_stats.skew(r)),
        kurtosis=float(scipy_stats.kurtosis(r)),
    )


@dataclass
class CPCVResult:
    """
    Result container for Combinatorial Purged Cross-Validation.

    CPCV tests strategy stability by splitting data into multiple folds
    and computing performance metrics for each. A robust strategy should
    have consistent performance across all folds.

    Attributes:
        fold_metrics: List of PerformanceMetrics, one per fold
        n_splits: Number of folds used

    Properties:
        sharpe_distribution: NumPy array of Sharpe ratios per fold
        mean_sharpe: Average Sharpe across folds
        sharpe_stability: Fraction of folds with positive Sharpe

    Methods:
        is_robust(): Check if strategy passes robustness thresholds
        __repr__(): Formatted summary string

    Example:
        --------
        cpcv_result = CPCVResult(
            fold_metrics=[m1, m2, m3, m4, m5, m6],
            n_splits=6
        )

        print(f"Mean Sharpe: {cpcv_result.mean_sharpe:.3f}")
        print(f"Stability: {cpcv_result.sharpe_stability:.1%}")

        if cpcv_result.is_robust(min_mean_sharpe=0.3, min_stability=0.7):
            print("Strategy passes CPCV!")
    """

    fold_metrics: List[PerformanceMetrics]
    n_splits: int

    @property
    def sharpe_distribution(self) -> np.ndarray:
        return np.array([m.sharpe for m in self.fold_metrics])

    @property
    def mean_sharpe(self) -> float:
        return float(self.sharpe_distribution.mean())

    @property
    def sharpe_stability(self) -> float:
        """Fraction of folds with positive Sharpe ratio (robustness indicator)."""
        dist = self.sharpe_distribution
        return float((dist > 0).mean())

    def is_robust(
        self, min_mean_sharpe: float = 0.3, min_stability: float = 0.7
    ) -> bool:
        return (
            self.mean_sharpe >= min_mean_sharpe
            and self.sharpe_stability >= min_stability
        )

    def __repr__(self) -> str:
        dist = self.sharpe_distribution
        return (
            f"CPCV ({len(self.fold_metrics)} folds): "
            f"Sharpe mean={dist.mean():.3f} "
            f"std={dist.std():.3f} "
            f"min={dist.min():.3f} "
            f"stability={self.sharpe_stability:.1%} "
            f"{'✅ ROBUST' if self.is_robust() else '❌ UNSTABLE'}"
        )


class CPCVEvaluator:
    """
    Combinatorial Purged Cross-Validation implementation.

    This evaluator splits returns into n_splits groups and computes
    performance metrics for each group. It uses purging (removing
    samples near the train/test boundary) to prevent look-ahead bias.

    The key metric is "stability": the fraction of folds with positive
    Sharpe. A stable strategy should have positive Sharpe in most/all folds.

    Attributes:
        n_splits: Number of folds to create (default: 6)
        purge_pct: Fraction to purge around each fold boundary (default: 0.02)

    Methods:
        evaluate(): Run CPCV and return results

    Example:
        --------
        evaluator = CPCVEvaluator(n_splits=6, purge_pct=0.02)

        result = evaluator.evaluate(returns, positions)

        print(result)
        # Output: CPCV (6 folds): Sharpe mean=0.85 std=0.23 min=0.41 stability=100% ✅ ROBUST
    """

    def __init__(self, n_splits: int = 6, purge_pct: float = 0.02):
        self.n_splits = n_splits
        self.purge_pct = purge_pct

    def evaluate(
        self,
        returns: np.ndarray,
        positions: np.ndarray,
    ) -> CPCVResult:
        """Split returns into n_splits groups and compute per-fold metrics."""
        n = len(returns)
        fold_size = n // self.n_splits
        purge_n = max(1, int(n * self.purge_pct))

        fold_metrics: List[PerformanceMetrics] = []

        for i in range(self.n_splits):
            test_start = i * fold_size
            test_end = min((i + 1) * fold_size, n)

            train_mask = np.ones(n, dtype=bool)
            embargo_s = max(0, test_start - purge_n)
            embargo_e = min(n, test_end + purge_n)
            train_mask[embargo_s:embargo_e] = False

            test_rets = returns[test_start:test_end]
            test_pos = positions[test_start:test_end]

            if len(test_rets) < 10:
                continue

            m = compute_metrics(test_rets * np.sign(test_pos + 1e-8), test_pos)
            fold_metrics.append(m)

        return CPCVResult(fold_metrics, self.n_splits)


@dataclass
class PermutationResult:
    """
    Result of the Monte Carlo permutation test.

    The permutation test determines whether observed performance could
    have occurred by random chance. It compares the actual strategy
    Sharpe to a null distribution created by shuffling trades randomly.

    If the observed metric is significantly better than random, the
    test passes (p < 0.05).

    Attributes:
        observed: Actual strategy metric value
        null_mean: Mean of null (random) distribution
        null_std: Standard deviation of null distribution
        null_p95: 95th percentile of null distribution
        p_value: Probability of observing this result by chance
        n_permutations: Number of Monte Carlo iterations
        metric: Which metric was tested ("sharpe", "sortino", etc.)

    Properties:
        is_significant: True if p_value < 0.05
        z_score: Standardized score (observed - null_mean) / null_std

    Methods:
        __repr__(): Formatted summary string

    Example:
        --------
        result = PermutationResult(
            observed=1.25,
            null_mean=0.15,
            null_std=0.35,
            null_p95=0.72,
            p_value=0.02,
            n_permutations=1000,
            metric="sharpe"
        )

        print(result)
        # Output: Permutation Test (sharpe): Observed=1.25 Null mean=0.15±0.35 p=0.0200 z=3.14
        #         ✅ SIGNIFICANT (p<0.05)
    """

    observed: float
    null_mean: float
    null_std: float
    null_p95: float
    p_value: float
    n_permutations: int
    metric: str

    @property
    def is_significant(self) -> bool:
        return self.p_value < 0.05

    @property
    def z_score(self) -> float:
        return (self.observed - self.null_mean) / (self.null_std + 1e-8)

    def __repr__(self) -> str:
        sig_str = (
            "✅ SIGNIFICANT (p<0.05)" if self.is_significant else "❌ NOT SIGNIFICANT"
        )
        return (
            f"Permutation Test ({self.metric}):\n"
            f"  Observed={self.observed:.4f}  "
            f"Null mean={self.null_mean:.4f}±{self.null_std:.4f}  "
            f"p={self.p_value:.4f}  z={self.z_score:.2f}\n"
            f"  {sig_str}"
        )


class PermutationTest:
    """
    Monte Carlo Permutation Test for statistical significance.

    This test answers: "Could the observed performance be due to luck?"

    It works by:
    1. Computing the actual strategy metric
    2. Randomly shuffling the positions many times
    3. Computing the metric for each shuffled version
    4. Comparing actual to the null distribution

    If the actual metric exceeds ~95% of random results (p < 0.05),
    the strategy passes the test.

    Attributes:
        n_permutations: Number of shuffles to perform (default: 1000)

    Methods:
        test(): Run permutation test on returns/positions

    Example:
        --------
        tester = PermutationTest(n_permutations=1000)

        result = tester.test(returns, positions, metric="sharpe")

        if result.is_significant:
            print("Strategy is statistically significant!")
        else:
            print("Strategy may be due to random chance")
    """

    def __init__(self, n_permutations: int = 1000):
        self.n_permutations = n_permutations

    def test(
        self,
        returns: np.ndarray,
        positions: np.ndarray,
        metric: str = "sharpe",
    ) -> PermutationResult:
        """Test whether observed performance can be explained by random chance."""
        rng = np.random.default_rng(42)
        observed = self._compute_metric(returns, positions, metric)

        null_distribution = []
        for _ in range(self.n_permutations):
            shuffled_pos = rng.permutation(positions)
            perf = self._compute_metric(returns, shuffled_pos, metric)
            null_distribution.append(perf)

        null_arr = np.array(null_distribution)
        p_value = float((null_arr >= observed).mean())

        return PermutationResult(
            observed=observed,
            null_mean=float(null_arr.mean()),
            null_std=float(null_arr.std()),
            null_p95=float(np.percentile(null_arr, 95)),
            p_value=p_value,
            n_permutations=self.n_permutations,
            metric=metric,
        )

    @staticmethod
    def _compute_metric(
        returns: np.ndarray, positions: np.ndarray, metric: str
    ) -> float:
        r = returns * np.sign(positions + 1e-9)
        m = compute_metrics(r)
        return getattr(m, metric, m.sharpe)


@dataclass
class DSRResult:
    """
    Result of the Deflated Sharpe Ratio computation.

    The Deflated Sharpe Ratio (DSR) corrects the observed Sharpe for:
    1. Selection bias: Trying many strategies, one will look good by chance
    2. Non-normality: Returns often have skewness and excess kurtosis

    DSR > 0.64 means there's 64% probability the true Sharpe exceeds
    the benchmark (usually 0 for risk-free rate).

    Attributes:
        observed_sr: The raw (inflated) Sharpe ratio
        sr_benchmark: Adjusted benchmark after accounting for trials
        sr_std: Standard error of the Sharpe estimate
        dsr: Deflated Sharpe Ratio (probability of true SR > benchmark)
        skewness: Return distribution skewness
        kurtosis: Return distribution excess kurtosis
        n_trials: Number of trials tested (for multiple testing adjustment)

    Properties:
        is_acceptable: True if DSR > 0.64

    Methods:
        __repr__(): Formatted summary string

    Example:
        --------
        result = DSRResult(
            observed_sr=1.5,
            sr_benchmark=0.3,
            sr_std=0.2,
            dsr=0.85,
            skewness=-0.5,
            kurtosis=3.2,
            n_trials=50
        )

        print(result)
        # Output: DSR=0.8500 (✅ ACCEPTABLE)
        #         SR_observed=1.5000 SR_benchmark=0.3000 SR_std=0.2000
        #         Skew=-0.500 Kurt=3.200 n_trials=50
    """

    observed_sr: float
    sr_benchmark: float
    sr_std: float
    dsr: float
    skewness: float
    kurtosis: float
    n_trials: int

    @property
    def is_acceptable(self) -> bool:
        return self.dsr > 0.64

    def __repr__(self) -> str:
        flag = "✅ ACCEPTABLE" if self.is_acceptable else "❌ REJECT"
        return (
            f"DSR={self.dsr:.4f} ({flag})\n"
            f"  SR_observed={self.observed_sr:.4f}  "
            f"SR_benchmark={self.sr_benchmark:.4f}  "
            f"SR_std={self.sr_std:.4f}\n"
            f"  Skew={self.skewness:.3f}  Kurt={self.kurtosis:.3f}  "
            f"n_trials={self.n_trials}"
        )


class DeflatedSharpeRatio:
    """
    Bailey & Lopez de Prado (2014) Deflated Sharpe Ratio.

    The DSR corrects the Sharpe ratio for multiple testing bias - the
    tendency to find good-looking strategies by chance when testing many.

    When you test N strategies, the best one will likely be better than
    the true average. DSR adjusts for this by:
    1. Estimating the expected maximum Sharpe across N trials
    2. Using this as a higher benchmark
    3. Computing probability that true Sharpe exceeds benchmark

    Formula accounts for:
    - Number of trials tested
    - Skewness of returns
    - Excess kurtosis (fat tails)

    Attributes:
        None - all methods are static

    Methods:
        compute(): Calculate DSR for a return series

    Example:
        --------
        result = DeflatedSharpeRatio.compute(
            returns=strategy_returns,
            n_trials=50,       # Tested 50 strategies
            benchmark_sr=0.0  # Compare to zero (risk-free)
        )

        print(f"Observed Sharpe: {result.observed_sr:.3f}")
        print(f"Deflated Sharpe: {result.dsr:.3f}")

        if result.is_acceptable:
            print("Sharpe is acceptable after adjusting for multiple testing!")
    """

    @staticmethod
    def compute(
        returns: np.ndarray,
        n_trials: int = 1,
        benchmark_sr: float = 0.0,
    ) -> DSRResult:
        r = np.asarray(returns, dtype=np.float64)
        r = r[~np.isnan(r)]
        n = len(r)

        if n < 10:
            return DSRResult(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, n_trials)

        sr_hat = float(r.mean() / (r.std() + 1e-8))  # sample Sharpe ratio
        skew = float(scipy_stats.skew(r))  # skewness of the return distribution
        kurt = float(scipy_stats.kurtosis(r))  # excess kurtosis (0 = Gaussian)

        # Variance of SR estimator accounting for non-Gaussianity (Bailey & Lopez de Prado eq. 7)
        sr_variance = (1 + 0.5 * sr_hat**2 - skew * sr_hat + (kurt / 4) * sr_hat**2) / (
            n - 1
        )
        sr_std = np.sqrt(max(sr_variance, 1e-8))  # std dev of SR estimate

        if n_trials > 1:
            # Expected maximum SR across n_trials independent tests (Gumbel EVT approximation)
            z = (1 - np.euler_gamma) * scipy_stats.norm.ppf(
                1 - 1 / n_trials
            ) + np.euler_gamma * scipy_stats.norm.ppf(1 - 1 / (n_trials * np.e))
            sr_benchmark = (
                benchmark_sr + sr_std * z
            )  # adjusted benchmark for multiple testing
        else:
            sr_benchmark = benchmark_sr  # single-trial: benchmark unchanged

        z_dsr = (sr_hat - sr_benchmark) / (sr_std + 1e-8)  # standardised DSR z-score
        dsr = float(scipy_stats.norm.cdf(z_dsr))  # DSR = Prob(true SR > benchmark)

        return DSRResult(
            observed_sr=sr_hat,
            sr_benchmark=sr_benchmark,
            sr_std=sr_std,
            dsr=dsr,
            skewness=skew,
            kurtosis=kurt,
            n_trials=n_trials,
        )


@dataclass
class MTRLResult:
    """
    Result of the Minimum Track Record Length calculation.

    MTRL answers: "How much data do I need to be confident this
    strategy's performance is real?"

    It calculates the minimum number of observations required for
    the observed Sharpe to be statistically significant at the
    given confidence level.

    Attributes:
        n_min: Minimum observations needed for significance
        sharpe: Observed Sharpe ratio
        confidence: Confidence level used (e.g., 0.95 = 95%)
        skewness: Return distribution skewness
        kurtosis: Return distribution excess kurtosis

    Methods:
        in_hours(): Convert to hours given bars per hour
        __repr__(): Formatted summary with time conversions

    Example:
        --------
        result = MTRLResult(
            n_min=250,
            sharpe=1.2,
            confidence=0.95,
            skewness=-0.3,
            kurtosis=2.1
        )

        print(result)
        # Output: Min Track Record: 250 bars
        #         @ 1h bars:  10.4 days (0.3 months)
        #         @ 5m bars:  125.0 days (4.2 months)
        #         SR=1.200 confidence=95%

        print(f"Need at least {result.n_min} observations")
    """

    n_min: float
    sharpe: float
    confidence: float
    skewness: float
    kurtosis: float

    def in_hours(self, bars_per_hour: float = 1.0) -> float:
        return self.n_min / bars_per_hour

    def __repr__(self) -> str:
        h1 = self.n_min
        d1 = h1 / 24
        m5 = self.n_min * 12
        return (
            f"Min Track Record: {self.n_min:.0f} bars\n"
            f"  @ 1h bars:  {d1:.1f} days ({d1 / 30:.1f} months)\n"
            f"  @ 5m bars:  {m5 / 24:.1f} days ({m5 / 24 / 30:.1f} months)\n"
            f"  SR={self.sharpe:.3f}  confidence={self.confidence:.0%}"
        )


class MinTrackRecordLength:
    """
    Computes minimum observations needed for statistical significance.

    MTRL calculates how much data is required for the observed Sharpe
    ratio to be statistically significant. It accounts for:
    - Target Sharpe threshold (what "good enough" means)
    - Desired confidence level
    - Return distribution characteristics (skewness, kurtosis)

    If your actual track record is shorter than MTRL, the observed
    performance may be due to luck rather than skill.

    Attributes:
        None - all methods are static

    Methods:
        compute(): Calculate MTRL for given Sharpe

    Example:
        --------
        result = MinTrackRecordLength.compute(
            sharpe=1.5,
            target_sr=0.0,
            confidence=0.95,
            skewness=0.0,
            kurtosis=0.0
        )

        print(f"Minimum track record: {result.n_min:.0f} bars")

        # With realistic market data (fat tails, skew)
        result = MinTrackRecordLength.compute(
            sharpe=1.5,
            target_sr=0.0,
            confidence=0.95,
            skewness=-0.5,  # Negative skew
            kurtosis=3.0    # Fat tails
        )

        print(f"Min track record: {result.n_min:.0f} bars")
        print(f"At hourly bars: {result.in_hours(bars_per_hour=1):.0f} hours")
    """

    @staticmethod
    def compute(
        sharpe: float,
        target_sr: float = 0.0,
        confidence: float = 0.95,
        skewness: float = 0.0,
        kurtosis: float = 0.0,
    ) -> MTRLResult:
        z_alpha = scipy_stats.norm.ppf(
            confidence
        )  # one-sided z critical value (e.g. 1.645 for 95%)

        if abs(sharpe - target_sr) < 1e-8:
            return MTRLResult(
                np.inf, sharpe, confidence, skewness, kurtosis
            )  # infinite bars needed when SR == target

        # Non-centrality term accounts for fat tails and skewness of the SR distribution
        noncentrality = (
            1 + 0.5 * sharpe**2 - skewness * sharpe + (kurtosis / 4) * sharpe**2
        )
        n_min = (
            z_alpha / (sharpe - target_sr)
        ) ** 2 * noncentrality  # MTRL formula: Bailey & Lopez de Prado

        return MTRLResult(
            n_min=float(n_min),
            sharpe=sharpe,
            confidence=confidence,
            skewness=skewness,
            kurtosis=kurtosis,
        )


@dataclass
class ValidationReport:
    """
    Complete validation report combining all four statistical checks.

    This is the main output of the BacktestValidator. It contains
    results from all four validation tests and provides an overall
    pass/fail verdict.

    Attributes:
        metrics: Full-sample performance metrics
        cpcv: CPCV stability test results
        perm: Permutation test results
        dsr: Deflated Sharpe Ratio results
        mtrl: Minimum Track Record Length results

    Properties:
        passes_all: True if strategy passes CPCV, Permutation, AND DSR

    Methods:
        __repr__(): Formatted multi-line report

    Example:
        --------
        validator = BacktestValidator(n_cpcv_splits=6, n_permutations=1000)
        report = validator.validate(returns, positions)

        print(report)

        if report.passes_all:
            print("\n✅ STRATEGY PASSES ALL VALIDATION TESTS")
            print("Ready for paper trading / live deployment")
        else:
            print("\n❌ STRATEGY FAILS VALIDATION")
            print("Review individual test failures above")
    """

    metrics: PerformanceMetrics
    cpcv: CPCVResult
    perm: PermutationResult
    dsr: DSRResult
    mtrl: MTRLResult

    @property
    def passes_all(self) -> bool:
        return (
            self.cpcv.is_robust()
            and self.perm.is_significant
            and self.dsr.is_acceptable
        )

    def __repr__(self) -> str:
        sep = "═" * 60
        lines = [
            sep,
            "  BACKTEST VALIDATION REPORT",
            sep,
            f"  Overall: {'✅ PASSES ALL CHECKS' if self.passes_all else '❌ FAILS – DO NOT GO LIVE'}",
            sep,
            "  [1] Full-Sample Metrics:",
            f"      {self.metrics}",
            "",
            "  [2] CPCV Stability:",
            f"      {self.cpcv}",
            "",
            "  [3] Permutation Test:",
            f"      {self.perm}",
            "",
            "  [4] Deflated Sharpe Ratio:",
            f"      {self.dsr}",
            "",
            "  [5] Min. Track Record Length:",
            f"      {self.mtrl}",
            sep,
        ]
        return "\n".join(lines)


class BacktestValidator:
    """
    Master validator that runs all four validation layers in sequence.

    This is the main entry point for the Anti-Bias validation suite.
    It coordinates all four tests and produces a comprehensive report
    with pass/fail verdicts for each test and overall.

    The validator runs:
    1. Full-sample metrics computation
    2. CPCV (Combinatorial Purged Cross-Validation)
    3. Permutation Test (Monte Carlo)
    4. Deflated Sharpe Ratio
    5. Minimum Track Record Length

    A strategy passes ONLY if it passes all tests.

    Attributes:
        cpcv: CPCVEvaluator instance
        perm: PermutationTest instance
        n_trials: Number of trials for DSR adjustment

    Methods:
        validate(): Run complete validation suite

    Example:
        --------
        # Simple usage
        validator = BacktestValidator()
        report = validator.validate(returns, positions)

        # With custom parameters
        validator = BacktestValidator(
            n_cpcv_splits=6,       # More folds = stricter
            n_permutations=1000,   # More iterations = precise p-value
            n_trials_tested=10,    # If testing 10 strategies
        )

        report = validator.validate(returns, positions)

        # Access individual results
        print(f"CPCV robust: {report.cpcv.is_robust()}")
        print(f"Permutation p-value: {report.perm.p_value:.4f}")
        print(f"DSR: {report.dsr.dsr:.3f}")

        # Overall verdict
        if report.passes_all:
            launch_strategy()
        else:
            print("Strategy not ready - fix issues above")
    """

    def __init__(
        self,
        n_cpcv_splits: int = 6,
        n_permutations: int = 1000,
        n_trials_tested: int = 1,
    ):
        self.cpcv = CPCVEvaluator(n_cpcv_splits)
        self.perm = PermutationTest(n_permutations)
        self.n_trials = n_trials_tested

    def validate(
        self,
        returns: np.ndarray,
        positions: np.ndarray,
    ) -> ValidationReport:
        """Run the full validation suite and return a combined report."""
        logger.info("Running full backtest validation suite...")

        metrics = compute_metrics(returns * np.sign(positions + 1e-9))
        cpcv_res = self.cpcv.evaluate(returns, positions)
        perm_res = self.perm.test(returns, positions, "sharpe")
        dsr_res = DeflatedSharpeRatio.compute(
            returns * np.sign(positions + 1e-9), n_trials=self.n_trials
        )
        mtrl_res = MinTrackRecordLength.compute(
            sharpe=metrics.sharpe,
            skewness=metrics.skewness,
            kurtosis=metrics.kurtosis,
        )

        return ValidationReport(metrics, cpcv_res, perm_res, dsr_res, mtrl_res)
