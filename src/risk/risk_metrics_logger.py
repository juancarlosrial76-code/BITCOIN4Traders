"""
Risk Metrics Logger
===================
Comprehensive tracking and logging system for risk metrics during
reinforcement learning training and backtesting sessions.

This module provides detailed risk analytics essential for evaluating
trading strategy performance. It tracks multiple risk-adjusted metrics
that help identify whether returns are generated through genuine edge
or excessive risk-taking.

Metrics Tracked:
----------------
- Drawdown: Current, maximum, and historical drawdown tracking
- Sharpe Ratio: Risk-adjusted return using total volatility
- Sortino Ratio: Risk-adjusted return using downside deviation
- Calmar Ratio: Return normalized by maximum drawdown
- Win/Loss Streaks: Consecutive win/loss tracking with historical max
- Kelly Fraction History: Position sizing evolution over time
- Risk-adjusted returns: Comprehensive return analysis

Why These Metrics Matter:
-------------------------
1. Sharpe Ratio: Standard risk-adjusted metric. Values > 1.0 indicate
   good risk-adjusted returns; > 2.0 is excellent.

2. Sortino Ratio: More appropriate than Sharpe for asymmetric returns.
   Only penalizes downside volatility, not upside.

3. Calmar Ratio: Essential for trading strategies. Measures how much
   return is generated per unit of maximum drawdown risk.

4. Drawdown Tracking: Maximum drawdown is often more relevant than
   volatility for trading strategies that require capital preservation.

Usage Example:
--------------
    from src.risk.risk_metrics_logger import RiskMetricsLogger

    # Initialize logger with lookback window
    logger = RiskMetricsLogger(lookback=50, risk_free_rate=0.0)

    # Update with each step/trade
    logger.update(
        equity=98500,
        trade_result=-500,
        kelly_fraction=0.12
    )

    # Get current risk metrics
    metrics = logger.get_current_metrics()
    print(f"Sharpe: {metrics['sharpe_ratio']:.2f}")
    print(f"Drawdown: {metrics['current_drawdown']*100:.1f}%")

    # Get full metrics history for analysis
    history_df = logger.get_metrics_history()

Dependencies:
-------------
- numpy: Numerical calculations
- pandas: DataFrame for metrics history
- loguru: Structured logging

Author: BITCOIN4Traders Team
Version: 1.0.0
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from collections import deque
from loguru import logger


@dataclass
class MetricsSnapshot:
    """
    Single point-in-time snapshot of all risk metrics.

    This dataclass represents a comprehensive view of risk metrics at
    a specific moment during trading. Snapshots are created periodically
    (every N steps) to enable historical analysis without excessive
    memory usage.

    Attributes:
        timestamp: Step number when snapshot was taken.
        equity: Account equity at snapshot time.
        drawdown: Current drawdown as decimal.
        sharpe_ratio: Rolling Sharpe ratio.
        sortino_ratio: Rolling Sortino ratio.
        calmar_ratio: Calmar ratio.
        win_streak: Current consecutive winning trades.
        loss_streak: Current consecutive losing trades.
        kelly_fraction: Current Kelly fraction being used.
        var_95: Value at Risk at 95% confidence.
    """

    # Timing
    timestamp: int

    # Equity metrics
    equity: float

    # Risk metrics
    drawdown: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float

    # Streak metrics
    win_streak: int
    loss_streak: int

    # Position sizing
    kelly_fraction: float

    # VaR placeholder
    var_95: float


class RiskMetricsLogger:
    """
    Comprehensive risk metrics tracking system for trading strategies.

    This class provides real-time tracking of all essential risk metrics
    during RL training or backtesting. It maintains rolling windows of
    equity and returns data to calculate dynamic risk-adjusted performance
    indicators.

    Key Features:
    -------------
    1. Rolling Ratios:
       - Sharpe Ratio: Uses configurable lookback window
       - Sortino Ratio: Focuses on downside risk
       - Calmar Ratio: Annualized return / max drawdown

    2. Drawdown Tracking:
       - Current drawdown from peak
       - Maximum drawdown over history
       - Running peak (high water mark)

    3. Streak Analysis:
       - Current and maximum win streak
       - Current and maximum loss streak
       - Useful for identifying regime changes

    4. Metrics History:
       - Periodic snapshots for historical analysis
       - DataFrame export for external analysis

    Usage:
    ------
    logger = RiskMetricsLogger(lookback=50, risk_free_rate=0.0)

    # Update each step
    logger.update(
        equity=98500,
        trade_result=-500,
        kelly_fraction=0.12
    )

    # Get current metrics
    metrics = logger.get_current_metrics()

    # Get full history
    history = logger.get_metrics_history()

    Attributes:
        lookback: Rolling window size for ratio calculations.
        risk_free_rate: Annual risk-free rate for Sharpe calculation.
        equity_history: Deque of equity values.
        returns_history: Deque of period returns.
        trade_history: Deque of trade results.
        kelly_history: Deque of Kelly fractions.
        current_win_streak: Current consecutive wins.
        current_loss_streak: Current consecutive losses.
        max_win_streak: Historical maximum win streak.
        max_loss_streak: Historical maximum loss streak.
        snapshots: List of MetricsSnapshot objects.
    """

    def __init__(self, lookback: int = 50, risk_free_rate: float = 0.0):
        """
        Initialize the metrics logger.

        Parameters:
        -----------
        lookback : int, optional
            Rolling window size for ratio calculations (default: 50).
            Larger values give more stable ratios but react slower to
            changing market conditions.
        risk_free_rate : float, optional
            Annual risk-free rate for Sharpe ratio calculation (default: 0.0).
            Set to current treasury yield (e.g., 0.05 for 5%) for realistic
            risk-adjusted performance measurement.

        Example:
            >>> logger = RiskMetricsLogger(lookback=100, risk_free_rate=0.04)
            >>> # Uses 100-period lookback, 4% annual risk-free rate
        """
        self.lookback = lookback
        self.risk_free_rate = risk_free_rate

        # History buffers using deque for efficient rolling window
        # maxlen=1000 limits memory while preserving sufficient history
        self.equity_history: deque = deque(maxlen=1000)
        self.returns_history: deque = deque(maxlen=1000)
        self.trade_history: deque = deque(maxlen=1000)
        self.kelly_history: deque = deque(maxlen=1000)

        # Streak tracking - current and historical maxima
        self.current_win_streak = 0
        self.current_loss_streak = 0
        self.max_win_streak = 0
        self.max_loss_streak = 0

        # Metrics snapshots - created periodically for history
        self.snapshots: List[MetricsSnapshot] = []

        # Peak tracking for drawdown calculation
        self.peak_equity = 0.0
        self.initial_equity = 0.0

        logger.info(f"RiskMetricsLogger initialized (lookback={lookback})")

    def update(
        self,
        equity: float,
        trade_result: Optional[float] = None,
        kelly_fraction: Optional[float] = None,
    ):
        """
        Update metrics with new data point.

        This method should be called after each trading period (step/bar)
        to maintain accurate risk metrics. It calculates returns, updates
        streak counters, and periodically creates metric snapshots.

        Parameters:
        -----------
        equity : float
            Current account equity in dollars. This is the primary
            driver for drawdown and return calculations.
        trade_result : float, optional
            Profit/loss from the most recent trade. Positive for wins,
            negative for losses. If provided, updates streak counters.
        kelly_fraction : float, optional
            Current Kelly fraction being used for position sizing.
            Useful for tracking how position sizing evolves over time.

        Side Effects:
        -------------
        - Updates equity_history and returns_history
        - Updates trade_history and streak counters (if trade_result provided)
        - Updates kelly_history (if kelly_fraction provided)
        - Creates metrics snapshot every 10 steps

        Note:
            On first call (empty history), initializes peak_equity and
            initial_equity to the provided equity value.

        Example:
            >>> # After a winning trade
            >>> logger.update(equity=101500, trade_result=1500, kelly_fraction=0.15)
            >>>
            >>> # After a losing trade
            >>> logger.update(equity=98500, trade_result=-500, kelly_fraction=0.12)
        """
        # Initialize on first update - set baseline values
        if len(self.equity_history) == 0:
            self.initial_equity = equity
            self.peak_equity = equity

        # Update peak equity (high water mark)
        # This is used as denominator for drawdown calculation
        if equity > self.peak_equity:
            self.peak_equity = equity

        # Calculate period return if we have previous equity
        # Return = (current - previous) / previous
        if len(self.equity_history) > 0:
            ret = (equity - self.equity_history[-1]) / self.equity_history[-1]
            self.returns_history.append(ret)

        # Update equity history
        self.equity_history.append(equity)

        # Update trade history and streaks if trade result provided
        if trade_result is not None:
            self.trade_history.append(trade_result)
            self._update_streaks(trade_result)

        # Track Kelly fraction evolution if provided
        if kelly_fraction is not None:
            self.kelly_history.append(kelly_fraction)

        # Create snapshot periodically (every 10 steps)
        # This balances memory usage with historical detail
        if len(self.equity_history) % 10 == 0:
            self._create_snapshot()

    def _update_streaks(self, trade_result: float):
        """
        Update win/loss streak counters based on trade result.

        Maintains both current streak (active) and maximum streak
        (historical) for both wins and losses. This helps identify
        regime changes - extended win streaks often precede reversals.

        Parameters:
        -----------
        trade_result : float
            Trade profit/loss. Positive = win, negative = loss.

        Logic:
            - Winning trade: increment win streak, reset loss streak
            - Losing trade: increment loss streak, reset win streak
            - Update historical maxima accordingly
        """
        if trade_result > 0:
            # Winning trade - increment win streak, reset loss streak
            self.current_win_streak += 1
            self.current_loss_streak = 0
            # Update historical maximum if current streak is higher
            self.max_win_streak = max(self.max_win_streak, self.current_win_streak)
        elif trade_result < 0:
            # Losing trade - increment loss streak, reset win streak
            self.current_loss_streak += 1
            self.current_win_streak = 0
            # Update historical maximum if current streak is higher
            self.max_loss_streak = max(self.max_loss_streak, self.current_loss_streak)

    def _create_snapshot(self):
        """Create metrics snapshot."""
        if len(self.equity_history) < 2:
            return

        snapshot = MetricsSnapshot(
            timestamp=len(self.equity_history),
            equity=self.equity_history[-1],
            drawdown=self._calculate_current_drawdown(),
            sharpe_ratio=self.calculate_sharpe_ratio(),
            sortino_ratio=self.calculate_sortino_ratio(),
            calmar_ratio=self.calculate_calmar_ratio(),
            win_streak=self.current_win_streak,
            loss_streak=self.current_loss_streak,
            kelly_fraction=self.kelly_history[-1] if self.kelly_history else 0.0,
            var_95=0.0,  # Placeholder, calculated elsewhere
        )

        self.snapshots.append(snapshot)

    def calculate_sharpe_ratio(self, annualization: float = 252.0) -> float:
        """
        Calculate rolling Sharpe ratio for risk-adjusted return measurement.

        The Sharpe ratio measures how much excess return is generated
        per unit of total volatility. Higher values indicate better
        risk-adjusted performance.

        Formula:
            Sharpe = (Mean Return - Risk Free Rate) / Std(Returns)

        The result is then annualized by multiplying by sqrt(annualization)
        to convert from per-period to per-year terms.

        Parameters:
        -----------
        annualization : float, optional
            Annualization factor (default: 252 for daily data).
            Common values:
            - 252 for daily data (trading days/year)
            - 252*24 for hourly data
            - 52 for weekly data

        Returns:
        --------
        sharpe : float
            Annualized Sharpe ratio. Typical interpretations:
            - < 0: Negative risk-adjusted returns
            - 0-1: Poor risk-adjusted returns
            - 1-2: Good risk-adjusted returns
            - > 2: Excellent risk-adjusted returns

        Note:
            Returns 0.0 if insufficient data (< 2 returns) or if
            standard deviation is zero (no variation in returns).

        Example:
            >>> sharpe = logger.calculate_sharpe_ratio(annualization=252)
            >>> print(f"Sharpe Ratio: {sharpe:.2f}")
            Sharpe Ratio: 1.45
        """
        # Need at least 2 data points for meaningful statistics
        if len(self.returns_history) < 2:
            return 0.0

        # Get recent returns within lookback window
        returns = np.array(list(self.returns_history)[-self.lookback :])

        if len(returns) < 2:
            return 0.0

        # Calculate mean and standard deviation of returns
        mean_return = np.mean(returns)
        std_return = np.std(returns)

        # Handle edge case of zero volatility
        if std_return == 0:
            return 0.0

        # Calculate Sharpe ratio:
        # (mean_return - rfr_per_period) / std_return
        # Risk-free rate is divided by annualization to get per-period rate
        sharpe = (mean_return - self.risk_free_rate / annualization) / std_return

        # Annualize by multiplying by sqrt(annualization factor)
        # This converts from per-period to annualized terms
        sharpe = sharpe * np.sqrt(annualization)

        return float(sharpe)

    def calculate_sortino_ratio(self, annualization: float = 252.0) -> float:
        """
        Calculate rolling Sortino ratio using downside deviation.

        The Sortino ratio is similar to Sharpe but only penalizes
        downside volatility (negative returns). This makes it more
        appropriate for strategies with asymmetric return distributions
        where upside volatility is desirable.

        Formula:
            Sortino = (Mean Return - Risk Free Rate) / Downside Deviation

        Downside deviation only considers negative returns, treating
        positive returns as zero for the calculation.

        Parameters:
        -----------
        annualization : float, optional
            Annualization factor (default: 252).

        Returns:
        --------
        sortino : float
            Annualized Sortino ratio. Interpretation:
            - > 2: Excellent risk-adjusted returns
            - 1-2: Good risk-adjusted returns
            - < 1: Poor risk-adjusted returns

        Note:
            Returns 10.0 (capped) if no negative returns in period,
            indicating very favorable conditions.

        Example:
            >>> sortino = logger.calculate_sortino_ratio()
            >>> print(f"Sortino Ratio: {sortino:.2f}")
            Sortino Ratio: 2.15
        """
        # Need sufficient history
        if len(self.returns_history) < 2:
            return 0.0

        returns = np.array(list(self.returns_history)[-self.lookback :])

        if len(returns) < 2:
            return 0.0

        # Calculate mean return
        mean_return = np.mean(returns)

        # Calculate downside deviation (only negative returns matter)
        # Filter to only negative returns for downside std calculation
        negative_returns = returns[returns < 0]

        if len(negative_returns) == 0:
            # No downside in this period - infinite Sortino (cap at 10)
            # This is actually a very favorable sign!
            return 10.0

        # Standard deviation of negative returns only
        downside_std = np.std(negative_returns)

        if downside_std == 0:
            return 0.0

        # Calculate Sortino using downside deviation
        sortino = (mean_return - self.risk_free_rate / annualization) / downside_std

        # Annualize same as Sharpe
        sortino = sortino * np.sqrt(annualization)

        return float(sortino)

    def calculate_calmar_ratio(self) -> float:
        """
        Calculate Calmar ratio for drawdown-normalized returns.

        The Calmar ratio measures annualized return relative to maximum
        drawdown. It's particularly important for trading strategies
        where drawdown (not volatility) is the primary risk.

        Formula:
            Calmar = Annualized Return / Maximum Drawdown

        Higher values indicate better returns per unit of drawdown risk.

        Returns:
        --------
        calmar : float
            Calmar ratio. Interpretation:
            - > 3: Excellent (rare)
            - 1-3: Good
            - < 1: Poor

        Note:
            Returns 0.0 if maximum drawdown is zero (equity never declined).

        Example:
            >>> calmar = logger.calculate_calmar_ratio()
            >>> print(f"Calmar Ratio: {calmar:.2f}")
            Calmar Ratio: 1.85
        """
        if len(self.equity_history) < 2:
            return 0.0

        # Calculate total return from initial to current equity
        # Return = (current - initial) / initial
        total_return = (
            self.equity_history[-1] - self.initial_equity
        ) / self.initial_equity

        n_periods = len(self.equity_history)

        # Annualize using geometric compounding:
        # annualized = (1 + total_return)^(252/n_periods) - 1
        # This properly accounts for compounding effects
        annualized_return = (1 + total_return) ** (252 / n_periods) - 1

        # Get maximum drawdown over full history
        max_dd = self._calculate_max_drawdown()

        if max_dd == 0:
            return 0.0

        # Calmar = annualized return / max drawdown
        # Use absolute value of drawdown (it's always negative in our convention)
        calmar = annualized_return / abs(max_dd)

        return float(calmar)

    def _calculate_current_drawdown(self) -> float:
        """Calculate current drawdown from peak."""
        if self.peak_equity == 0:
            return 0.0

        current_equity = self.equity_history[-1]
        drawdown = (self.peak_equity - current_equity) / self.peak_equity

        return float(drawdown)

    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown in history."""
        if len(self.equity_history) < 2:
            return 0.0

        equity_array = np.array(self.equity_history)
        running_max = np.maximum.accumulate(equity_array)  # Running peak equity
        drawdowns = (running_max - equity_array) / running_max  # Drawdown at each point

        max_dd = np.max(drawdowns)  # Largest drawdown over the entire history

        return float(max_dd)

    def get_current_metrics(self) -> Dict:
        """
        Get current risk metrics.

        Returns:
        --------
        metrics : dict
            Current metrics snapshot
        """
        return {
            "equity": self.equity_history[-1] if self.equity_history else 0.0,
            "peak_equity": self.peak_equity,
            "current_drawdown": self._calculate_current_drawdown(),
            "max_drawdown": self._calculate_max_drawdown(),
            "sharpe_ratio": self.calculate_sharpe_ratio(),
            "sortino_ratio": self.calculate_sortino_ratio(),
            "calmar_ratio": self.calculate_calmar_ratio(),
            "win_streak": self.current_win_streak,
            "loss_streak": self.current_loss_streak,
            "max_win_streak": self.max_win_streak,
            "max_loss_streak": self.max_loss_streak,
            "total_trades": len(self.trade_history),
            "avg_kelly_fraction": np.mean(self.kelly_history)
            if self.kelly_history
            else 0.0,
        }

    def get_metrics_history(self) -> pd.DataFrame:
        """
        Get full metrics history as DataFrame.

        Returns:
        --------
        df : pd.DataFrame
            Metrics history
        """
        if not self.snapshots:
            return pd.DataFrame()

        data = []
        for snap in self.snapshots:
            data.append(
                {
                    "timestamp": snap.timestamp,
                    "equity": snap.equity,
                    "drawdown": snap.drawdown,
                    "sharpe_ratio": snap.sharpe_ratio,
                    "sortino_ratio": snap.sortino_ratio,
                    "calmar_ratio": snap.calmar_ratio,
                    "win_streak": snap.win_streak,
                    "loss_streak": snap.loss_streak,
                    "kelly_fraction": snap.kelly_fraction,
                }
            )

        return pd.DataFrame(data)

    def reset(self):
        """Reset all metrics."""
        self.equity_history.clear()
        self.returns_history.clear()
        self.trade_history.clear()
        self.kelly_history.clear()

        self.current_win_streak = 0
        self.current_loss_streak = 0
        self.max_win_streak = 0
        self.max_loss_streak = 0

        self.snapshots = []
        self.peak_equity = 0.0
        self.initial_equity = 0.0


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("RISK METRICS LOGGER TEST")
    print("=" * 80)

    # Initialize
    logger = RiskMetricsLogger(lookback=50)

    # Simulate trading episode
    np.random.seed(42)
    initial_capital = 100000
    equity = initial_capital

    print(f"\n✓ Starting simulation with ${initial_capital:,.0f}")

    for i in range(200):
        # Simulate trade
        if np.random.rand() < 0.55:  # 55% win rate
            trade_result = np.random.uniform(100, 500)
        else:
            trade_result = np.random.uniform(-300, -100)

        equity += trade_result

        # Kelly fraction (simplified)
        kelly_frac = 0.15 if trade_result > 0 else 0.10

        # Update logger
        logger.update(
            equity=equity, trade_result=trade_result, kelly_fraction=kelly_frac
        )

    print(f"\n✓ Simulated 200 trades")

    # Get current metrics
    print("\n[CURRENT METRICS]")
    metrics = logger.get_current_metrics()

    print(f"  Final equity: ${metrics['equity']:,.0f}")
    print(f"  Peak equity: ${metrics['peak_equity']:,.0f}")
    print(f"  Current drawdown: {metrics['current_drawdown'] * 100:.2f}%")
    print(f"  Max drawdown: {metrics['max_drawdown'] * 100:.2f}%")
    print(f"  Sharpe ratio: {metrics['sharpe_ratio']:.2f}")
    print(f"  Sortino ratio: {metrics['sortino_ratio']:.2f}")
    print(f"  Calmar ratio: {metrics['calmar_ratio']:.2f}")
    print(f"  Current win streak: {metrics['win_streak']}")
    print(f"  Max win streak: {metrics['max_win_streak']}")
    print(f"  Max loss streak: {metrics['max_loss_streak']}")
    print(f"  Total trades: {metrics['total_trades']}")
    print(f"  Avg Kelly fraction: {metrics['avg_kelly_fraction']:.2f}")

    # Get history
    print("\n[METRICS HISTORY]")
    history = logger.get_metrics_history()

    if not history.empty:
        print(f"  Snapshots recorded: {len(history)}")
        print(
            f"  Sharpe range: [{history['sharpe_ratio'].min():.2f}, {history['sharpe_ratio'].max():.2f}]"
        )
        print(
            f"  Drawdown range: [{history['drawdown'].min() * 100:.2f}%, {history['drawdown'].max() * 100:.2f}%]"
        )

    print("\n" + "=" * 80)
    print("✓ RISK METRICS LOGGER TEST PASSED")
    print("=" * 80)
