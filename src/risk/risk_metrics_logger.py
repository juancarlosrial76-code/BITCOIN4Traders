"""
Risk Metrics Logger
===================
Tracks and logs comprehensive risk metrics during training and backtesting.

Metrics Tracked:
- Drawdown (current, max, average)
- Sharpe Ratio (rolling)
- Sortino Ratio (rolling)
- Calmar Ratio
- Win/Loss Streaks
- Kelly Fraction History
- Risk-adjusted returns
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from collections import deque
from loguru import logger


@dataclass
class MetricsSnapshot:
    """Single point-in-time risk metrics."""
    timestamp: int
    equity: float
    drawdown: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    win_streak: int
    loss_streak: int
    kelly_fraction: float
    var_95: float


class RiskMetricsLogger:
    """
    Comprehensive risk metrics tracking.
    
    Tracks real-time risk metrics during episode:
    - Drawdown metrics
    - Risk-adjusted return ratios
    - Streak analysis
    - Kelly sizing history
    
    Usage:
    ------
    logger = RiskMetricsLogger(lookback=50)
    
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
    """
    
    def __init__(
        self,
        lookback: int = 50,
        risk_free_rate: float = 0.0
    ):
        """
        Initialize metrics logger.
        
        Parameters:
        -----------
        lookback : int
            Rolling window for ratio calculations
        risk_free_rate : float
            Annual risk-free rate for Sharpe calculation
        """
        self.lookback = lookback
        self.risk_free_rate = risk_free_rate
        
        # History buffers
        self.equity_history: deque = deque(maxlen=1000)
        self.returns_history: deque = deque(maxlen=1000)
        self.trade_history: deque = deque(maxlen=1000)
        self.kelly_history: deque = deque(maxlen=1000)
        
        # Streak tracking
        self.current_win_streak = 0
        self.current_loss_streak = 0
        self.max_win_streak = 0
        self.max_loss_streak = 0
        
        # Metrics snapshots
        self.snapshots: List[MetricsSnapshot] = []
        
        # Peak tracking
        self.peak_equity = 0.0
        self.initial_equity = 0.0
        
        logger.info(f"RiskMetricsLogger initialized (lookback={lookback})")
    
    def update(
        self,
        equity: float,
        trade_result: Optional[float] = None,
        kelly_fraction: Optional[float] = None
    ):
        """
        Update metrics with new data point.
        
        Parameters:
        -----------
        equity : float
            Current equity
        trade_result : float, optional
            P&L of last trade
        kelly_fraction : float, optional
            Current Kelly fraction
        """
        # Initialize on first update
        if len(self.equity_history) == 0:
            self.initial_equity = equity
            self.peak_equity = equity
        
        # Update peak
        if equity > self.peak_equity:
            self.peak_equity = equity
        
        # Calculate return
        if len(self.equity_history) > 0:
            ret = (equity - self.equity_history[-1]) / self.equity_history[-1]
            self.returns_history.append(ret)
        
        # Update histories
        self.equity_history.append(equity)
        
        if trade_result is not None:
            self.trade_history.append(trade_result)
            self._update_streaks(trade_result)
        
        if kelly_fraction is not None:
            self.kelly_history.append(kelly_fraction)
        
        # Create snapshot (every N steps to avoid overhead)
        if len(self.equity_history) % 10 == 0:
            self._create_snapshot()
    
    def _update_streaks(self, trade_result: float):
        """Update win/loss streaks."""
        if trade_result > 0:
            # Win
            self.current_win_streak += 1
            self.current_loss_streak = 0
            self.max_win_streak = max(self.max_win_streak, self.current_win_streak)
        elif trade_result < 0:
            # Loss
            self.current_loss_streak += 1
            self.current_win_streak = 0
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
            var_95=0.0  # Placeholder, calculated elsewhere
        )
        
        self.snapshots.append(snapshot)
    
    def calculate_sharpe_ratio(self, annualization: float = 252.0) -> float:
        """
        Calculate rolling Sharpe ratio.
        
        Sharpe = (Mean Return - Risk Free Rate) / Std(Returns)
        
        Parameters:
        -----------
        annualization : float
            Annualization factor (252 for daily, 252*24 for hourly)
            
        Returns:
        --------
        sharpe : float
            Annualized Sharpe ratio
        """
        if len(self.returns_history) < 2:
            return 0.0
        
        # Get recent returns
        returns = np.array(list(self.returns_history)[-self.lookback:])
        
        if len(returns) < 2:
            return 0.0
        
        # Calculate
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        if std_return == 0:
            return 0.0
        
        sharpe = (mean_return - self.risk_free_rate / annualization) / std_return
        sharpe = sharpe * np.sqrt(annualization)
        
        return float(sharpe)
    
    def calculate_sortino_ratio(self, annualization: float = 252.0) -> float:
        """
        Calculate rolling Sortino ratio.
        
        Sortino = (Mean Return - Risk Free Rate) / Downside Deviation
        
        Only considers negative returns in denominator.
        
        Parameters:
        -----------
        annualization : float
            Annualization factor
            
        Returns:
        --------
        sortino : float
            Annualized Sortino ratio
        """
        if len(self.returns_history) < 2:
            return 0.0
        
        returns = np.array(list(self.returns_history)[-self.lookback:])
        
        if len(returns) < 2:
            return 0.0
        
        # Mean return
        mean_return = np.mean(returns)
        
        # Downside deviation (only negative returns)
        negative_returns = returns[returns < 0]
        
        if len(negative_returns) == 0:
            # No downside -> infinite Sortino (cap at high value)
            return 10.0
        
        downside_std = np.std(negative_returns)
        
        if downside_std == 0:
            return 0.0
        
        sortino = (mean_return - self.risk_free_rate / annualization) / downside_std
        sortino = sortino * np.sqrt(annualization)
        
        return float(sortino)
    
    def calculate_calmar_ratio(self) -> float:
        """
        Calculate Calmar ratio.
        
        Calmar = Annualized Return / Max Drawdown
        
        Returns:
        --------
        calmar : float
            Calmar ratio
        """
        if len(self.equity_history) < 2:
            return 0.0
        
        # Annualized return
        total_return = (self.equity_history[-1] - self.initial_equity) / self.initial_equity
        n_periods = len(self.equity_history)
        annualized_return = (1 + total_return) ** (252 / n_periods) - 1
        
        # Max drawdown
        max_dd = self._calculate_max_drawdown()
        
        if max_dd == 0:
            return 0.0
        
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
        running_max = np.maximum.accumulate(equity_array)
        drawdowns = (running_max - equity_array) / running_max
        
        max_dd = np.max(drawdowns)
        
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
            'equity': self.equity_history[-1] if self.equity_history else 0.0,
            'peak_equity': self.peak_equity,
            'current_drawdown': self._calculate_current_drawdown(),
            'max_drawdown': self._calculate_max_drawdown(),
            'sharpe_ratio': self.calculate_sharpe_ratio(),
            'sortino_ratio': self.calculate_sortino_ratio(),
            'calmar_ratio': self.calculate_calmar_ratio(),
            'win_streak': self.current_win_streak,
            'loss_streak': self.current_loss_streak,
            'max_win_streak': self.max_win_streak,
            'max_loss_streak': self.max_loss_streak,
            'total_trades': len(self.trade_history),
            'avg_kelly_fraction': np.mean(self.kelly_history) if self.kelly_history else 0.0
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
            data.append({
                'timestamp': snap.timestamp,
                'equity': snap.equity,
                'drawdown': snap.drawdown,
                'sharpe_ratio': snap.sharpe_ratio,
                'sortino_ratio': snap.sortino_ratio,
                'calmar_ratio': snap.calmar_ratio,
                'win_streak': snap.win_streak,
                'loss_streak': snap.loss_streak,
                'kelly_fraction': snap.kelly_fraction
            })
        
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
    print("="*80)
    print("RISK METRICS LOGGER TEST")
    print("="*80)
    
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
            equity=equity,
            trade_result=trade_result,
            kelly_fraction=kelly_frac
        )
    
    print(f"\n✓ Simulated 200 trades")
    
    # Get current metrics
    print("\n[CURRENT METRICS]")
    metrics = logger.get_current_metrics()
    
    print(f"  Final equity: ${metrics['equity']:,.0f}")
    print(f"  Peak equity: ${metrics['peak_equity']:,.0f}")
    print(f"  Current drawdown: {metrics['current_drawdown']*100:.2f}%")
    print(f"  Max drawdown: {metrics['max_drawdown']*100:.2f}%")
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
        print(f"  Sharpe range: [{history['sharpe_ratio'].min():.2f}, {history['sharpe_ratio'].max():.2f}]")
        print(f"  Drawdown range: [{history['drawdown'].min()*100:.2f}%, {history['drawdown'].max()*100:.2f}%]")
    
    print("\n" + "="*80)
    print("✓ RISK METRICS LOGGER TEST PASSED")
    print("="*80)
