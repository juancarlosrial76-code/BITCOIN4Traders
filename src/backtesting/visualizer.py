"""
Backtest Visualizer & Report Generator
=======================================
Creates comprehensive visual reports for backtest results.

Visualizations:
1. Equity Curve
2. Drawdown Chart
3. Monthly Returns Heatmap
4. Trade Distribution
5. Rolling Sharpe Ratio
6. Win/Loss Analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional
from pathlib import Path
from datetime import datetime
from loguru import logger

# Set style
plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("husl")


class BacktestVisualizer:
    """
    Create visual reports for backtest results.

    Usage:
    ------
    viz = BacktestVisualizer()

    # Create full report
    viz.create_report(
        equity=equity_series,
        trades=trades_df,
        metrics=performance_metrics,
        save_path='reports/backtest_2024.png'
    )
    """

    def __init__(self, figsize: tuple = (16, 12)):
        """
        Initialize visualizer.

        Parameters:
        -----------
        figsize : tuple
            Figure size for reports
        """
        self.figsize = figsize
        logger.info("BacktestVisualizer initialized")

    def create_report(
        self,
        equity: pd.Series,
        trades: Optional[pd.DataFrame] = None,
        metrics: Optional[Dict] = None,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Create comprehensive backtest report.

        Parameters:
        -----------
        equity : pd.Series
            Equity curve
        trades : pd.DataFrame, optional
            Trade data
        metrics : dict, optional
            Performance metrics
        save_path : str, optional
            Path to save figure

        Returns:
        --------
        fig : plt.Figure
            Report figure
        """
        fig = plt.figure(figsize=self.figsize)

        # Create grid
        gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)

        # 1. Equity curve (large, top)
        ax1 = fig.add_subplot(gs[0, :])
        self._plot_equity_curve(ax1, equity)

        # 2. Drawdown (below equity)
        ax2 = fig.add_subplot(gs[1, :])
        self._plot_drawdown(ax2, equity)

        # 3. Monthly returns heatmap
        ax3 = fig.add_subplot(gs[2, :2])
        self._plot_monthly_returns(ax3, equity)

        # 4. Trade distribution
        if trades is not None:
            ax4 = fig.add_subplot(gs[2, 2])
            self._plot_trade_distribution(ax4, trades)

        # 5. Rolling Sharpe
        ax5 = fig.add_subplot(gs[3, :2])
        self._plot_rolling_sharpe(ax5, equity)

        # 6. Metrics summary
        ax6 = fig.add_subplot(gs[3, 2])
        if metrics is not None:
            self._plot_metrics_summary(ax6, metrics)
        ax6.axis("off")

        # Title
        fig.suptitle("Backtest Performance Report", fontsize=16, fontweight="bold")

        # Save if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Report saved: {save_path}")

        return fig

    def _plot_equity_curve(self, ax: plt.Axes, equity: pd.Series):
        """Plot equity curve."""
        ax.plot(equity.index, equity.values, linewidth=2, label="Equity")

        # Add buy & hold benchmark if possible
        returns = equity.pct_change()
        if len(returns) > 0:
            # constant-growth benchmark: compound the mean return at every step
            benchmark = (1 + returns.mean()).cumprod() * equity.iloc[0]
            ax.plot(
                equity.index,
                benchmark,
                linewidth=1,
                alpha=0.5,
                linestyle="--",
                label="Mean Return",
            )

        ax.set_title("Equity Curve", fontsize=12, fontweight="bold")
        ax.set_xlabel("Date")
        ax.set_ylabel("Equity ($)")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Format y-axis
        ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, p: f"${x:,.0f}")
        )  # dollar format with commas

    def _plot_drawdown(self, ax: plt.Axes, equity: pd.Series):
        """Plot drawdown chart."""
        running_max = (
            equity.expanding().max()
        )  # highest equity seen so far at each point
        drawdown = (
            equity - running_max
        ) / running_max  # underwater fraction (negative values)

        ax.fill_between(
            drawdown.index,
            0,
            drawdown.values * 100,
            alpha=0.3,
            color="red",
            label="Drawdown",
        )
        ax.plot(drawdown.index, drawdown.values * 100, color="red", linewidth=1)

        ax.set_title("Drawdown", fontsize=12, fontweight="bold")
        ax.set_xlabel("Date")
        ax.set_ylabel("Drawdown (%)")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Highlight max drawdown
        max_dd = drawdown.min()
        max_dd_date = drawdown.idxmin()
        ax.axhline(max_dd * 100, color="red", linestyle="--", alpha=0.7, linewidth=1)
        ax.text(
            drawdown.index[len(drawdown) // 2],
            max_dd * 100,
            f"Max DD: {max_dd * 100:.2f}%",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

    def _plot_monthly_returns(self, ax: plt.Axes, equity: pd.Series):
        """Plot monthly returns heatmap."""
        # Calculate monthly returns
        monthly = (
            equity.resample("M").last().pct_change() * 100
        )  # last price per month, then % change

        # Reshape into years x months
        monthly_df = monthly.to_frame("return")
        monthly_df["year"] = monthly_df.index.year
        monthly_df["month"] = monthly_df.index.month

        # Pivot table: rows=years, columns=months → calendar heatmap layout
        pivot = monthly_df.pivot(index="year", columns="month", values="return")

        # Month names
        month_names = [
            "Jan",
            "Feb",
            "Mar",
            "Apr",
            "May",
            "Jun",
            "Jul",
            "Aug",
            "Sep",
            "Oct",
            "Nov",
            "Dec",
        ]
        pivot.columns = [month_names[m - 1] for m in pivot.columns]

        # Heatmap
        sns.heatmap(
            pivot,
            annot=True,
            fmt=".1f",
            cmap="RdYlGn",
            center=0,
            cbar_kws={"label": "Return (%)"},
            ax=ax,
        )

        ax.set_title("Monthly Returns (%)", fontsize=12, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("")

    def _plot_trade_distribution(self, ax: plt.Axes, trades: pd.DataFrame):
        """Plot trade P&L distribution."""
        if "pnl" not in trades.columns:
            ax.text(0.5, 0.5, "No trade data", ha="center", va="center")
            return

        pnl = trades["pnl"].values

        # Histogram
        ax.hist(pnl, bins=30, alpha=0.7, edgecolor="black")
        ax.axvline(0, color="red", linestyle="--", linewidth=2, alpha=0.7)
        ax.axvline(pnl.mean(), color="green", linestyle="--", linewidth=2, alpha=0.7)

        ax.set_title("Trade Distribution", fontsize=12, fontweight="bold")
        ax.set_xlabel("P&L ($)")
        ax.set_ylabel("Frequency")
        ax.grid(True, alpha=0.3)

        # Add statistics
        textstr = f"Mean: ${pnl.mean():.0f}\nMedian: ${np.median(pnl):.0f}"
        ax.text(
            0.05,
            0.95,
            textstr,
            transform=ax.transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

    def _plot_rolling_sharpe(self, ax: plt.Axes, equity: pd.Series, window: int = 60):
        """Plot rolling Sharpe ratio."""
        returns = equity.pct_change()

        rolling_sharpe = (
            returns.rolling(window).mean()
            / returns.rolling(window).std()
            * np.sqrt(252)
        )

        ax.plot(rolling_sharpe.index, rolling_sharpe.values, linewidth=2)
        ax.axhline(0, color="red", linestyle="--", alpha=0.5)
        ax.axhline(1, color="green", linestyle="--", alpha=0.5, label="Sharpe = 1")
        ax.axhline(2, color="blue", linestyle="--", alpha=0.5, label="Sharpe = 2")

        ax.set_title(
            f"Rolling Sharpe Ratio ({window}d)", fontsize=12, fontweight="bold"
        )
        ax.set_xlabel("Date")
        ax.set_ylabel("Sharpe Ratio")
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_metrics_summary(self, ax: plt.Axes, metrics: Dict):
        """Plot key metrics summary."""
        text = []

        # Extract key metrics
        if hasattr(metrics, "total_return"):
            text.append(f"Total Return: {metrics.total_return * 100:.2f}%")
            text.append(f"CAGR: {metrics.cagr * 100:.2f}%")
            text.append(f"")
            text.append(f"Sharpe: {metrics.sharpe_ratio:.2f}")
            text.append(f"Sortino: {metrics.sortino_ratio:.2f}")
            text.append(f"Calmar: {metrics.calmar_ratio:.2f}")
            text.append(f"")
            text.append(f"Max DD: {metrics.max_drawdown * 100:.2f}%")
            text.append(f"Volatility: {metrics.volatility * 100:.2f}%")
            text.append(f"")
            text.append(f"Trades: {metrics.total_trades}")
            text.append(f"Win Rate: {metrics.win_rate * 100:.1f}%")
            text.append(f"Profit Factor: {metrics.profit_factor:.2f}")
        else:
            # Assume dict format
            for key, value in metrics.items():
                text.append(f"{key}: {value}")

        # Display as text box
        text_str = "\n".join(text)
        ax.text(
            0.1,
            0.9,
            text_str,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.5),
            family="monospace",
        )

        ax.set_title("Key Metrics", fontsize=12, fontweight="bold")

    def plot_walk_forward_results(
        self, results: List, save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot walk-forward analysis results.

        Parameters:
        -----------
        results : List[WindowResult]
            Walk-forward results
        save_path : str, optional
            Path to save figure

        Returns:
        --------
        fig : plt.Figure
            Results figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Extract data
        window_ids = [r.window_id for r in results]
        train_returns = [r.train_return * 100 for r in results]
        test_returns = [r.test_return * 100 for r in results]
        test_sharpes = [r.test_sharpe for r in results]
        test_dds = [r.test_max_drawdown * 100 for r in results]

        # 1. Train vs Test Returns
        ax = axes[0, 0]
        ax.plot(window_ids, train_returns, "o-", label="Train", alpha=0.7)
        ax.plot(window_ids, test_returns, "s-", label="Test (OOS)", alpha=0.7)
        ax.axhline(0, color="red", linestyle="--", alpha=0.5)
        ax.set_title("Returns: Train vs Test", fontweight="bold")
        ax.set_xlabel("Window")
        ax.set_ylabel("Return (%)")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 2. Test Sharpe Ratio
        ax = axes[0, 1]
        ax.bar(window_ids, test_sharpes, alpha=0.7)
        ax.axhline(1, color="green", linestyle="--", alpha=0.5, label="Sharpe = 1")
        ax.set_title("Test Sharpe Ratio", fontweight="bold")
        ax.set_xlabel("Window")
        ax.set_ylabel("Sharpe Ratio")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 3. Test Max Drawdown
        ax = axes[1, 0]
        ax.bar(window_ids, test_dds, color="red", alpha=0.7)
        ax.set_title("Test Max Drawdown", fontweight="bold")
        ax.set_xlabel("Window")
        ax.set_ylabel("Max DD (%)")
        ax.grid(True, alpha=0.3)

        # 4. Distribution of Test Returns
        ax = axes[1, 1]
        ax.hist(test_returns, bins=15, alpha=0.7, edgecolor="black")
        ax.axvline(
            np.mean(test_returns),
            color="green",
            linestyle="--",
            linewidth=2,
            label=f"Mean: {np.mean(test_returns):.2f}%",
        )
        ax.axvline(0, color="red", linestyle="--", linewidth=2, alpha=0.5)
        ax.set_title("Distribution of Test Returns", fontweight="bold")
        ax.set_xlabel("Return (%)")
        ax.set_ylabel("Frequency")
        ax.legend()
        ax.grid(True, alpha=0.3)

        fig.suptitle("Walk-Forward Analysis Results", fontsize=16, fontweight="bold")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Walk-forward plot saved: {save_path}")

        return fig


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("BACKTEST VISUALIZER TEST")
    print("=" * 80)

    # Generate synthetic data
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", "2024-01-01", freq="D")

    returns = np.random.normal(0.0005, 0.01, len(dates))
    equity = pd.Series(100000 * (1 + returns).cumprod(), index=dates)

    trades = pd.DataFrame({"pnl": np.random.normal(100, 500, 100)})

    from backtesting.performance_calculator import PerformanceCalculator

    calc = PerformanceCalculator()
    metrics = calc.calculate_from_equity_curve(equity, trades)

    print("\n✓ Generated synthetic data")

    # Create visualizer
    viz = BacktestVisualizer()

    # Create report
    print("\n[TEST] Creating visual report...")
    fig = viz.create_report(
        equity=equity,
        trades=trades,
        metrics=metrics,
        save_path="test_backtest_report.png",
    )

    print("  ✓ Report created")

    # Show
    plt.show()

    print("\n" + "=" * 80)
    print("✓ BACKTEST VISUALIZER TEST PASSED")
    print("=" * 80)
