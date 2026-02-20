"""
ANTI-BIAS FRAMEWORK – Statistical Validation
=============================================
Validierungssuite: CPCV + Permutation + DSR + MTRL.
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
    """Performance-Metriken für eine Strategie."""

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
    """Berechnet vollständige Performance-Metriken."""
    r = np.asarray(returns, dtype=np.float64)
    r = r[~np.isnan(r)]
    if len(r) == 0:
        return PerformanceMetrics(*([0.0] * 11 + [0]))

    sharpe = float(r.mean() / (r.std() + 1e-8))

    downside = r[r < 0]
    sortino = float(r.mean() / (downside.std() + 1e-8)) if len(downside) > 0 else sharpe

    equity = np.cumprod(1 + r)
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / (peak + 1e-8)
    max_dd = float(-dd.min())

    calmar = float(r.mean() / (max_dd + 1e-8))

    wins = r[r > 0]
    losses = r[r < 0]
    win_rate = float(len(wins) / (len(r) + 1e-8))
    pf = float(wins.sum() / (-losses.sum() + 1e-8)) if len(losses) > 0 else 99.0

    trade_r = (
        r[r != 0]
        if positions is None
        else r[np.diff(np.concatenate([[0], positions])) != 0]
    )
    if len(trade_r) == 0:
        trade_r = r

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
    """Ergebnis des Combinatorial Purged CV."""

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
        """Anteil Folds mit positivem Sharpe."""
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
    """Combinatorial Purged Cross-Validation."""

    def __init__(self, n_splits: int = 6, purge_pct: float = 0.02):
        self.n_splits = n_splits
        self.purge_pct = purge_pct

    def evaluate(
        self,
        returns: np.ndarray,
        positions: np.ndarray,
    ) -> CPCVResult:
        """Teilt Returns in n_splits Gruppen und berechnet Metriken."""
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
    """Ergebnis des Permutation Tests."""

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
    """Monte Carlo Permutationstest."""

    def __init__(self, n_permutations: int = 1000):
        self.n_permutations = n_permutations

    def test(
        self,
        returns: np.ndarray,
        positions: np.ndarray,
        metric: str = "sharpe",
    ) -> PermutationResult:
        """Testet ob Performance durch Zufall erklärbar ist."""
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
    """Ergebnis des Deflated Sharpe Ratio."""

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
    Korrigiert Sharpe für Multiple Testing.
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

        sr_hat = float(r.mean() / (r.std() + 1e-8))
        skew = float(scipy_stats.skew(r))
        kurt = float(scipy_stats.kurtosis(r))

        sr_variance = (1 + 0.5 * sr_hat**2 - skew * sr_hat + (kurt / 4) * sr_hat**2) / (
            n - 1
        )
        sr_std = np.sqrt(max(sr_variance, 1e-8))

        if n_trials > 1:
            z = (1 - np.euler_gamma) * scipy_stats.norm.ppf(
                1 - 1 / n_trials
            ) + np.euler_gamma * scipy_stats.norm.ppf(1 - 1 / (n_trials * np.e))
            sr_benchmark = benchmark_sr + sr_std * z
        else:
            sr_benchmark = benchmark_sr

        z_dsr = (sr_hat - sr_benchmark) / (sr_std + 1e-8)
        dsr = float(scipy_stats.norm.cdf(z_dsr))

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
    """Minimum Track Record Length Ergebnis."""

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
    """Berechnet minimale Track Record Länge für statistische Signifikanz."""

    @staticmethod
    def compute(
        sharpe: float,
        target_sr: float = 0.0,
        confidence: float = 0.95,
        skewness: float = 0.0,
        kurtosis: float = 0.0,
    ) -> MTRLResult:
        z_alpha = scipy_stats.norm.ppf(confidence)

        if abs(sharpe - target_sr) < 1e-8:
            return MTRLResult(np.inf, sharpe, confidence, skewness, kurtosis)

        noncentrality = (
            1 + 0.5 * sharpe**2 - skewness * sharpe + (kurtosis / 4) * sharpe**2
        )
        n_min = (z_alpha / (sharpe - target_sr)) ** 2 * noncentrality

        return MTRLResult(
            n_min=float(n_min),
            sharpe=sharpe,
            confidence=confidence,
            skewness=skewness,
            kurtosis=kurtosis,
        )


@dataclass
class ValidationReport:
    """Vollständiger Validierungsbericht."""

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
    Master Validator – führt alle 4 Validierungsebenen durch.
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
        """Führt vollständige Validierung durch."""
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
