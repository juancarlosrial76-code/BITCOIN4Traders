"""
Performance Metrics Calculator
===============================
Comprehensive trading performance analysis with Anti-Bias validation.

Metrics Categories:
1. Return Metrics (CAGR, Total Return, Monthly Returns)
2. Risk Metrics (Volatility, Downside Deviation, VaR, CVaR)
3. Risk-Adjusted Metrics (Sharpe, Sortino, Calmar, Omega)
4. Drawdown Metrics (Max DD, Avg DD, Recovery Time)
5. Trade Metrics (Win Rate, Profit Factor, Avg Trade)
6. Anti-Bias Validation (CPCV, Permutation, DSR, MTRL)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from scipy import stats
from loguru import logger

# Anti-Bias Framework integration
try:
    from evaluation.antibias_validator import (
        BacktestValidator,
        ValidationReport,
        compute_metrics as antibias_compute_metrics,
    )

    ANTIBIAS_AVAILABLE = True
except ImportError:
    ANTIBIAS_AVAILABLE = False
    logger.warning("Anti-Bias Validator not available")


@dataclass
class PerformanceMetrics:
    """Complete performance metrics."""

    # Return metrics
    total_return: float
    cagr: float
    monthly_return: float

    # Risk metrics
    volatility: float
    downside_deviation: float
    var_95: float
    cvar_95: float

    # Risk-adjusted metrics
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    omega_ratio: float

    # Drawdown metrics
    max_drawdown: float
    avg_drawdown: float
    max_drawdown_duration: int
    recovery_time: int

    # Trade metrics
    total_trades: int
    win_rate: float
    profit_factor: float
    avg_trade: float
    avg_win: float
    avg_loss: float
    largest_win: float
    largest_loss: float

    # Consistency metrics
    positive_months: int
    negative_months: int
    consistency_score: float


class PerformanceCalculator:
    """
    Calculate comprehensive trading performance metrics.

    Usage:
    ------
    calc = PerformanceCalculator()

    # From equity curve
    metrics = calc.calculate_from_equity_curve(equity_series)

    # From trades
    metrics = calc.calculate_from_trades(trades_df)

    # Print report
    calc.print_report(metrics)
    """

    def __init__(self, risk_free_rate: float = 0.0):
        """
        Initialize calculator.

        Parameters:
        -----------
        risk_free_rate : float
            Annual risk-free rate (default: 0.0)
        """
        self.risk_free_rate = risk_free_rate
        logger.info("PerformanceCalculator initialized")

    def calculate_from_equity_curve(
        self, equity: pd.Series, trades: Optional[pd.DataFrame] = None
    ) -> PerformanceMetrics:
        """
        Calculate metrics from equity curve.

        Parameters:
        -----------
        equity : pd.Series
            Equity curve (datetime index)
        trades : pd.DataFrame, optional
            Trade-level data

        Returns:
        --------
        metrics : PerformanceMetrics
            Complete performance metrics
        """
        # Calculate returns
        returns = equity.pct_change().dropna()

        # Return metrics
        total_return = (equity.iloc[-1] - equity.iloc[0]) / equity.iloc[0]

        years = (equity.index[-1] - equity.index[0]).days / 365.25
        cagr = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0.0

        monthly_return = cagr / 12

        # Risk metrics
        volatility = returns.std() * np.sqrt(252)

        negative_returns = returns[returns < 0]
        downside_deviation = (
            negative_returns.std() * np.sqrt(252) if len(negative_returns) > 0 else 0.0
        )

        var_95 = np.percentile(returns, 5)
        cvar_95 = returns[returns <= var_95].mean()

        # Risk-adjusted metrics
        excess_returns = returns - self.risk_free_rate / 252
        sharpe_ratio = (
            (excess_returns.mean() / returns.std() * np.sqrt(252))
            if returns.std() > 0
            else 0.0
        )

        sortino_ratio = (
            (excess_returns.mean() / downside_deviation * np.sqrt(252))
            if downside_deviation > 0
            else 0.0
        )

        # Drawdown metrics
        dd_metrics = self._calculate_drawdown_metrics(equity)

        calmar_ratio = (
            cagr / abs(dd_metrics["max_drawdown"])
            if dd_metrics["max_drawdown"] < 0
            else 0.0
        )

        # Omega ratio
        omega_ratio = self._calculate_omega_ratio(returns)

        # Trade metrics (if available)
        if trades is not None and len(trades) > 0:
            trade_metrics = self._calculate_trade_metrics(trades)
        else:
            trade_metrics = {
                "total_trades": 0,
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "avg_trade": 0.0,
                "avg_win": 0.0,
                "avg_loss": 0.0,
                "largest_win": 0.0,
                "largest_loss": 0.0,
            }

        # Consistency metrics
        monthly_returns = equity.resample("M").last().pct_change().dropna()
        positive_months = (monthly_returns > 0).sum()
        negative_months = (monthly_returns < 0).sum()
        consistency_score = (
            positive_months / len(monthly_returns) if len(monthly_returns) > 0 else 0.0
        )

        # Build metrics object
        metrics = PerformanceMetrics(
            total_return=total_return,
            cagr=cagr,
            monthly_return=monthly_return,
            volatility=volatility,
            downside_deviation=downside_deviation,
            var_95=var_95,
            cvar_95=cvar_95,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            omega_ratio=omega_ratio,
            max_drawdown=dd_metrics["max_drawdown"],
            avg_drawdown=dd_metrics["avg_drawdown"],
            max_drawdown_duration=dd_metrics["max_drawdown_duration"],
            recovery_time=dd_metrics["recovery_time"],
            total_trades=trade_metrics["total_trades"],
            win_rate=trade_metrics["win_rate"],
            profit_factor=trade_metrics["profit_factor"],
            avg_trade=trade_metrics["avg_trade"],
            avg_win=trade_metrics["avg_win"],
            avg_loss=trade_metrics["avg_loss"],
            largest_win=trade_metrics["largest_win"],
            largest_loss=trade_metrics["largest_loss"],
            positive_months=positive_months,
            negative_months=negative_months,
            consistency_score=consistency_score,
        )

        return metrics

    def _calculate_drawdown_metrics(self, equity: pd.Series) -> Dict:
        """Calculate drawdown metrics."""
        # Calculate drawdown series
        running_max = equity.expanding().max()
        drawdown = (equity - running_max) / running_max

        # Max drawdown
        max_drawdown = drawdown.min()

        # Average drawdown
        drawdowns = drawdown[drawdown < 0]
        avg_drawdown = drawdowns.mean() if len(drawdowns) > 0 else 0.0

        # Drawdown duration
        in_drawdown = drawdown < 0

        # Find longest drawdown period
        max_duration = 0
        current_duration = 0

        for is_dd in in_drawdown:
            if is_dd:
                current_duration += 1
                max_duration = max(max_duration, current_duration)
            else:
                current_duration = 0

        # Recovery time (from max drawdown to recovery)
        max_dd_idx = drawdown.idxmin()
        recovery_idx = equity[max_dd_idx:][
            equity[max_dd_idx:] >= running_max[max_dd_idx]
        ].index

        if len(recovery_idx) > 0:
            recovery_time = (recovery_idx[0] - max_dd_idx).days
        else:
            recovery_time = -1  # Not yet recovered

        return {
            "max_drawdown": max_drawdown,
            "avg_drawdown": avg_drawdown,
            "max_drawdown_duration": max_duration,
            "recovery_time": recovery_time,
        }

    def _calculate_omega_ratio(
        self, returns: pd.Series, threshold: float = 0.0
    ) -> float:
        """
        Calculate Omega ratio.

        Omega = Probability-weighted gains / Probability-weighted losses
        """
        returns_above = returns[returns > threshold] - threshold
        returns_below = threshold - returns[returns < threshold]

        if len(returns_below) == 0:
            return float("inf")

        omega = returns_above.sum() / returns_below.sum()

        return float(omega)

    def _calculate_trade_metrics(self, trades: pd.DataFrame) -> Dict:
        """Calculate trade-level metrics."""
        # Ensure pnl column exists
        if "pnl" not in trades.columns:
            return {
                "total_trades": 0,
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "avg_trade": 0.0,
                "avg_win": 0.0,
                "avg_loss": 0.0,
                "largest_win": 0.0,
                "largest_loss": 0.0,
            }

        total_trades = len(trades)

        # Separate wins and losses
        wins = trades[trades["pnl"] > 0]
        losses = trades[trades["pnl"] < 0]

        # Win rate
        win_rate = len(wins) / total_trades if total_trades > 0 else 0.0

        # Profit factor
        gross_profit = wins["pnl"].sum() if len(wins) > 0 else 0.0
        gross_loss = abs(losses["pnl"].sum()) if len(losses) > 0 else 0.0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0.0

        # Average metrics
        avg_trade = trades["pnl"].mean()
        avg_win = wins["pnl"].mean() if len(wins) > 0 else 0.0
        avg_loss = losses["pnl"].mean() if len(losses) > 0 else 0.0

        # Extremes
        largest_win = wins["pnl"].max() if len(wins) > 0 else 0.0
        largest_loss = losses["pnl"].min() if len(losses) > 0 else 0.0

        return {
            "total_trades": total_trades,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "avg_trade": avg_trade,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "largest_win": largest_win,
            "largest_loss": largest_loss,
        }

    def print_report(self, metrics: PerformanceMetrics):
        """Print formatted performance report."""
        print("\n" + "=" * 80)
        print("PERFORMANCE REPORT")
        print("=" * 80)

        print("\nRETURN METRICS:")
        print(f"  Total Return:        {metrics.total_return * 100:>10.2f}%")
        print(f"  CAGR:                {metrics.cagr * 100:>10.2f}%")
        print(f"  Monthly Return:      {metrics.monthly_return * 100:>10.2f}%")

        print("\nRISK METRICS:")
        print(f"  Volatility (Ann.):   {metrics.volatility * 100:>10.2f}%")
        print(f"  Downside Deviation:  {metrics.downside_deviation * 100:>10.2f}%")
        print(f"  VaR (95%):           {metrics.var_95 * 100:>10.2f}%")
        print(f"  CVaR (95%):          {metrics.cvar_95 * 100:>10.2f}%")

        print("\nRISK-ADJUSTED METRICS:")
        print(f"  Sharpe Ratio:        {metrics.sharpe_ratio:>10.2f}")
        print(f"  Sortino Ratio:       {metrics.sortino_ratio:>10.2f}")
        print(f"  Calmar Ratio:        {metrics.calmar_ratio:>10.2f}")
        print(f"  Omega Ratio:         {metrics.omega_ratio:>10.2f}")

        print("\nDRAWDOWN METRICS:")
        print(f"  Max Drawdown:        {metrics.max_drawdown * 100:>10.2f}%")
        print(f"  Avg Drawdown:        {metrics.avg_drawdown * 100:>10.2f}%")
        print(f"  Max DD Duration:     {metrics.max_drawdown_duration:>10d} days")
        print(f"  Recovery Time:       {metrics.recovery_time:>10d} days")

        print("\nTRADE METRICS:")
        print(f"  Total Trades:        {metrics.total_trades:>10d}")
        print(f"  Win Rate:            {metrics.win_rate * 100:>10.1f}%")
        print(f"  Profit Factor:       {metrics.profit_factor:>10.2f}")
        print(f"  Avg Trade:           ${metrics.avg_trade:>10.2f}")
        print(f"  Avg Win:             ${metrics.avg_win:>10.2f}")
        print(f"  Avg Loss:            ${metrics.avg_loss:>10.2f}")
        print(f"  Largest Win:         ${metrics.largest_win:>10.2f}")
        print(f"  Largest Loss:        ${metrics.largest_loss:>10.2f}")

        print("\nCONSISTENCY METRICS:")
        print(f"  Positive Months:     {metrics.positive_months:>10d}")
        print(f"  Negative Months:     {metrics.negative_months:>10d}")
        print(f"  Consistency Score:   {metrics.consistency_score * 100:>10.1f}%")

        print("\n" + "=" * 80)

    def compare_strategies(
        self, metrics_list: List[PerformanceMetrics], labels: List[str]
    ) -> pd.DataFrame:
        """
        Compare multiple strategies.

        Parameters:
        -----------
        metrics_list : List[PerformanceMetrics]
            List of metrics for each strategy
        labels : List[str]
            Strategy names

        Returns:
        --------
        comparison : pd.DataFrame
            Comparison table
        """
        data = []

        for metrics, label in zip(metrics_list, labels):
            data.append(
                {
                    "Strategy": label,
                    "Total Return": f"{metrics.total_return * 100:.2f}%",
                    "CAGR": f"{metrics.cagr * 100:.2f}%",
                    "Sharpe": f"{metrics.sharpe_ratio:.2f}",
                    "Sortino": f"{metrics.sortino_ratio:.2f}",
                    "Calmar": f"{metrics.calmar_ratio:.2f}",
                    "Max DD": f"{metrics.max_drawdown * 100:.2f}%",
                    "Win Rate": f"{metrics.win_rate * 100:.1f}%",
                    "Trades": metrics.total_trades,
                }
            )

        df = pd.DataFrame(data)

        return df

    def validate_with_antibias(
        self,
        returns: np.ndarray,
        positions: np.ndarray,
        n_cpcv_splits: int = 6,
        n_permutations: int = 1000,
        n_trials_tested: int = 1,
    ) -> Optional[ValidationReport]:
        """
        Validate strategy using Anti-Bias Framework.

        Args:
            returns: Per-bar returns
            positions: Position array (-1, 0, 1)
            n_cpcv_splits: Number of CPCV splits
            n_permutations: Number of permutations for test
            n_trials_tested: Number of trials tested (for DSR)

        Returns:
            ValidationReport or None if antibias not available
        """
        if not ANTIBIAS_AVAILABLE:
            logger.warning("Anti-Bias Framework not available, skipping validation")
            return None

        logger.info("Running Anti-Bias validation...")
        validator = BacktestValidator(
            n_cpcv_splits=n_cpcv_splits,
            n_permutations=n_permutations,
            n_trials_tested=n_trials_tested,
        )

        report = validator.validate(returns, positions)

        logger.info(
            f"Validation complete: {'✅ PASSES' if report.passes_all else '❌ FAILS'}"
        )
        return report

    def print_validation_report(self, report: ValidationReport):
        """Print Anti-Bias validation report."""
        print("\n" + "=" * 80)
        print("ANTI-BIAS VALIDATION REPORT")
        print("=" * 80)
        print(report)


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("PERFORMANCE CALCULATOR TEST")
    print("=" * 80)

    # Generate synthetic equity curve
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", "2024-01-01", freq="D")

    # Simulate returns with drift
    returns = np.random.normal(0.0005, 0.01, len(dates))
    equity = pd.Series(100000 * (1 + returns).cumprod(), index=dates)

    # Generate synthetic trades
    n_trades = 100
    trades = pd.DataFrame({"pnl": np.random.normal(100, 500, n_trades)})

    print(f"\n✓ Generated synthetic data")
    print(f"  Equity curve: {len(equity)} days")
    print(f"  Trades: {len(trades)}")

    # Calculate metrics
    calc = PerformanceCalculator(risk_free_rate=0.02)
    metrics = calc.calculate_from_equity_curve(equity, trades)

    # Print report
    calc.print_report(metrics)

    print("\n" + "=" * 80)
    print("✓ PERFORMANCE CALCULATOR TEST PASSED")
    print("=" * 80)
