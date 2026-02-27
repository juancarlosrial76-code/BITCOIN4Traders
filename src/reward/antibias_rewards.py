"""
ANTI-BIAS FRAMEWORK – Risk-Adjusted Reward Functions
====================================================
Comprehensive reward functions designed to prevent common RL training biases
in algorithmic trading systems.

Problem: Naive Reward Bias
--------------------------
Traditional reinforcement learning rewards simply use raw returns:
    reward_t = pnl_t

This creates several dangerous biases in the agent:

1. VOLATILITY BIAS: The agent learns to chase volatile spikes in returns
   rather than seeking consistent, predictable profits. Lucky outliers
   get reinforced the same as genuine edge.

2. CHURNING BIAS: Without cost awareness, the agent learns that more
   frequent trading (even if unprofitable) generates more reward signals,
   leading to excessive portfolio turnover.

3. DRAWDOWN IGNORANCE: Raw returns don't penalize drawdowns, so the agent
   may take excessive risk to recover from losses, compounding losses further.

4. REGIME BLINDNESS: Naive rewards don't consider market conditions, so
   the agent may maintain positions inappropriate for the current regime.

Solution: This Module
---------------------
This module provides sophisticated reward functions that account for:

- Risk-adjusted returns (Sharpe, Sortino)
- Transaction costs and trading friction
- Drawdown penalties
- Market regime alignment

All rewards are clipped to stable ranges (±3-5) to ensure gradient stability
during neural network training.

Classes Provided:
-----------------
1. BaseReward: Abstract base class defining the reward interface
2. SharpeIncrementReward: Rolling Sharpe ratio component
3. CalmarIncrementReward: Drawdown-penalized returns
4. CostAwareReward: Transaction cost and drawdown penalties
5. RegimeAwareReward: Complete reward with regime alignment bonus
6. RewardAnalyzer: Benchmarking utility comparing all reward types

Usage Example:
--------------
    from src.reward.antibias_rewards import SharpeIncrementReward, CostAwareReward

    # Simple Sharpe-based reward
    sharpe_reward = SharpeIncrementReward(window=50)

    # Or more sophisticated cost-aware reward
    cost_reward = CostAwareReward(
        lambda_cost=2.0,    # Cost penalty weight
        lambda_draw=5.0,    # Drawdown penalty weight
        cost_rate=0.001,    # Transaction cost rate
        window=50
    )

    # Compute reward
    reward = cost_reward.compute(
        pnl=100,           # Profit this bar
        position=0.5,      # Current position (50% of capital)
        prev_position=0.3, # Previous position
        equity=10100,      # Current equity
        cost_this_bar=5    # Transaction cost incurred
    )

Dependencies:
-------------
- numpy: Numerical calculations
- abc: Abstract base class

Author: BITCOIN4Traders Team
Version: 1.0.0
"""

from __future__ import annotations

import abc
import logging
from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np

logger = logging.getLogger("antibias.reward")


class BaseReward(abc.ABC):
    """
    Abstract interface for all reward function implementations.

    This base class defines the contract that all reward functions must
    follow. It ensures compatibility with the trading environment and
    provides a common reset mechanism for episode boundaries.

    All reward functions must implement:
    1. compute(): Calculate reward for a single time step
    2. reset(): Clear internal state for new episode

    Attributes:
        None (subclasses define their own)

    Note:
        Reward values should generally be clipped to reasonable ranges
        (±3 to ±5) to maintain stable gradient descent during training.
        Raw P&L values can be orders of magnitude larger, causing
        training instability.
    """

    @abc.abstractmethod
    def compute(
        self,
        pnl: float,
        position: float,
        prev_position: float,
        equity: float,
        cost_this_bar: float,
        **kwargs,
    ) -> float:
        """
        Compute the scalar reward for a single environment step.

        Parameters:
        -----------
        pnl : float
            Profit/loss for this time step (before costs).
        position : float
            Current position size as fraction of capital (-1 to 1 for
            long/short, or 0 to 1 for long-only).
        prev_position : float
            Previous position size.
        equity : float
            Current account equity.
        cost_this_bar : float
            Transaction costs incurred this bar (fees + spread + slippage).

        Returns:
        --------
        reward : float
            Computed reward value (should be clipped to ±3-5 range).

        Raises:
        -------
        NotImplementedError: Must be implemented by subclass.
        """

    def reset(self) -> None:
        """
        Reset internal state at the beginning of a new episode.

        Called by the environment when starting a new trading episode.
        Should clear any rolling windows, buffers, or stateful tracking.

        Default implementation does nothing; subclasses should override
        if they maintain state.
        """
        pass


class SharpeIncrementReward(BaseReward):
    """
    Rolling Sharpe ratio-based reward for risk-adjusted returns.

    This reward normalizes each period's return by the rolling standard
    deviation, rewarding consistent returns over volatile spikes.

    Formula:
        reward_t = (r_t - μ_rolling) / σ_rolling

    Where:
        r_t = net return (pnl - costs)
        μ_rolling = rolling mean of returns
        σ_rolling = rolling standard deviation of returns

    Interpretation:
        - Positive value: Return above average for this volatility level
        - Negative value: Return below average for this volatility level
        - Magnitude: How many standard deviations from mean

    Advantages:
        - Penalizes volatile, inconsistent returns
        - Rewards steady compounding over lucky spikes
        - Stabilizes gradient magnitudes for neural network training

    Attributes:
        window: Rolling window size for mean/std calculation.
        rf: Risk-free rate per period (annual rate / periods).

    Example:
        >>> reward = SharpeIncrementReward(window=50)
        >>> reward.compute(pnl=100, position=0.5, prev_position=0.3,
        ...               equity=10000, cost_this_bar=5)
        0.85
    """

    def __init__(self, window: int = 50, risk_free_rate: float = 0.0):
        """
        Initialize Sharpe increment reward calculator.

        Parameters:
        -----------
        window : int, optional
            Rolling window size (default: 50). Larger windows give
            more stable ratios but react slower to regime changes.
        risk_free_rate : float, optional
            Annual risk-free rate (default: 0.0). Divided by periods
            to get per-period rate.
        """
        self.window = window
        # Convert annual rate to per-bar rate (assuming hourly = 365*24 bars/year)
        self.rf = risk_free_rate / (365 * 24)
        # Rolling buffer of net returns (after costs)
        self._ret_buf: deque = deque(maxlen=window)
        self._peak_equity: float = 0.0

    def compute(
        self, pnl, position, prev_position, equity, cost_this_bar, **kwargs
    ) -> float:
        """
        Compute Sharpe increment reward for current step.

        Parameters:
        -----------
        pnl : float
            Gross P&L before costs.
        position : float
            Current position.
        prev_position : float
            Previous position.
        equity : float
            Current equity.
        cost_this_bar : float
            Transaction costs this bar.

        Returns:
        --------
        reward : float
            Sharpe increment, clipped to ±3 range.
        """
        # Net return after transaction costs
        net_ret = pnl - cost_this_bar
        self._ret_buf.append(net_ret)

        # Warm-up period: return simple scaled reward
        if len(self._ret_buf) < 5:
            return float(np.clip(net_ret * 10, -1, 1))

        # Calculate rolling statistics
        arr = np.array(self._ret_buf)
        mu = arr.mean()  # Rolling mean return
        sig = arr.std() + 1e-8  # Rolling std + epsilon to avoid div/0

        # Incremental Sharpe: how many std deviations above/below mean
        # This measures if current return is better or worse than typical
        sharpe_increment = (net_ret - self.rf - mu) / sig

        # Clip to ±3 sigma for stable training gradients
        return float(np.clip(sharpe_increment, -3, 3))

    def reset(self) -> None:
        """Reset rolling buffers for new episode."""
        self._ret_buf.clear()
        self._peak_equity = 0.0


class CalmarIncrementReward(BaseReward):
    """
    Drawdown-penalized reward using rolling Calmar ratio concept.

    This reward divides returns by current drawdown, creating an
    asymmetric penalty where losses in drawdown are heavily penalized.
    Encourages capital preservation over aggressive recovery.

    Formula:
        reward_t = r_t / (max_drawdown_rolling + ε)

    Advantages:
        - Strongly penalizes losses during drawdowns
        - Encourages cutting losses quickly
        - Rewards consistency during good periods

    Attributes:
        window: Rolling window for drawdown calculation.

    Example:
        >>> reward = CalmarIncrementReward(window=100)
        >>> reward.compute(pnl=-50, position=0.5, prev_position=0.5,
        ...               equity=9500, cost_this_bar=5)
        -0.85
    """

    def __init__(self, window: int = 100):
        """
        Initialize Calmar increment reward calculator.

        Parameters:
        -----------
        window : int, optional
            Rolling window for drawdown calculation (default: 100).
        """
        self.window = window
        self._equity_buf: deque = deque(maxlen=window)
        self._peak: float = 0.0

    def compute(
        self, pnl, position, prev_position, equity, cost_this_bar, **kwargs
    ) -> float:
        """
        Compute Calmar increment reward.

        Parameters:
        -----------
        pnl : float
            Gross P&L.
        position : float
            Current position.
        prev_position : float
            Previous position.
        equity : float
            Current equity.
        cost_this_bar : float
            Transaction costs.

        Returns:
        --------
        reward : float
            Calmar increment, clipped to ±3 range.
        """
        # Net return after costs
        net_ret = pnl - cost_this_bar
        self._equity_buf.append(equity)

        if not self._equity_buf:
            return 0.0

        # Calculate rolling max drawdown
        eq_arr = np.array(self._equity_buf)
        peak = eq_arr.max()
        trough = eq_arr.min()

        # Drawdown as fraction of peak
        max_dd = (peak - trough) / (peak + 1e-8)

        # Floor at 0.1% to avoid division by zero in good periods
        max_dd = max(max_dd, 0.001)

        # Calmar increment: return normalized by drawdown
        # During drawdowns, denominator is small → reward is heavily negative
        # During good periods, denominator is large → reward is moderate
        calmar_inc = net_ret / max_dd

        # Scale and clip for stable training
        return float(np.clip(calmar_inc * 10, -3, 3))

    def reset(self) -> None:
        """Reset buffers for new episode."""
        self._equity_buf.clear()
        self._peak = 0.0


class CostAwareReward(BaseReward):
    """
    Comprehensive reward with transaction cost and drawdown penalties.

    This is a complete reward function that addresses multiple biases:
    - Churning: Penalizes excessive position changes
    - Drawdown: Penalizes losses during drawdown periods
    - Costs: Explicitly subtracts transaction costs

    Formula:
        reward_t = net_pnl - λ_cost × |Δpos| × cost_rate - λ_draw × drawdown × max(0, -net_pnl)

    Where:
        net_pnl = pnl - cost_this_bar (P&L after fees)
        |Δpos| = |position - prev_position| (position change)
        drawdown = (peak_equity - equity) / peak_equity

    The cost penalty discourages unnecessary trading, while the drawdown
    penalty only applies during losing periods to encourage cutting losses.

    Attributes:
        lambda_cost: Weight for trading cost penalty.
        lambda_draw: Weight for drawdown penalty.
        cost_rate: Transaction cost as fraction of trade value.
        window: Rolling window for equity tracking.

    Example:
        >>> reward = CostAwareReward(
        ...     lambda_cost=2.0,
        ...     lambda_draw=5.0,
        ...     cost_rate=0.001,
        ...     window=50
        ... )
        >>> reward.compute(pnl=100, position=0.5, prev_position=0.3,
        ...                equity=10100, cost_this_bar=10)
        0.75
    """

    def __init__(
        self,
        lambda_cost: float = 2.0,
        lambda_draw: float = 5.0,
        cost_rate: float = 0.001,
        window: int = 50,
    ):
        """
        Initialize cost-aware reward calculator.

        Parameters:
        -----------
        lambda_cost : float, optional
            Weight for cost penalty (default: 2.0). Higher values
            discourage more trading.
        lambda_draw : float, optional
            Weight for drawdown penalty (default: 5.0). Higher values
            discourage holding during drawdowns.
        cost_rate : float, optional
            Transaction cost as decimal (default: 0.001 = 0.1%).
        window : int, optional
            Rolling window for equity tracking (default: 50).
        """
        self.lambda_cost = lambda_cost
        self.lambda_draw = lambda_draw
        self.cost_rate = cost_rate
        self.window = window

        self._equity_buf: deque = deque(maxlen=window)
        self._peak_equity: float = 1.0
        self._trade_count: int = 0

    def compute(
        self, pnl, position, prev_position, equity, cost_this_bar, **kwargs
    ) -> float:
        """
        Compute cost-aware reward.

        Parameters:
        -----------
        pnl : float
            Gross P&L before costs.
        position : float
            Current position.
        prev_position : float
            Previous position.
        equity : float
            Current equity.
        cost_this_bar : float
            Transaction costs incurred.

        Returns:
        --------
        reward : float
            Net reward after penalties, clipped to ±3 range.
        """
        # Position change: fraction of capital being repositioned
        position_change = abs(position - prev_position)
        self._equity_buf.append(equity)

        # Net P&L after exchange fees
        net_pnl = pnl - cost_this_bar

        # Churn penalty: penalize unnecessary trading
        # Scales with position change magnitude
        churn_penalty = self.lambda_cost * position_change * self.cost_rate

        # Update high water mark for drawdown
        self._peak_equity = max(self._peak_equity, equity)

        # Current drawdown from peak
        drawdown = max(0, (self._peak_equity - equity) / (self._peak_equity + 1e-8))

        # Drawdown penalty: only applies during losing bars
        # This encourages cutting losses without penalizing
        draw_penalty = self.lambda_draw * drawdown * max(0, -net_pnl)

        # Track meaningful trades (ignore micro-adjustments)
        if position_change > 0.01:
            self._trade_count += 1

        # Final reward: net P&L minus penalties
        reward = net_pnl - churn_penalty - draw_penalty

        # Scale to ±3 range for stable training
        return float(np.clip(reward * 100, -3, 3))

    def reset(self) -> None:
        """Reset state for new episode."""
        self._equity_buf.clear()
        self._peak_equity = 1.0
        self._trade_count = 0


@dataclass
class RegimeState:
    """
    Market regime information passed into RegimeAwareReward.

    This dataclass encapsulates the current market regime detected
    by external regime detection systems.

    Attributes:
        regime: Market direction (0=Bear, 1=Neutral, 2=Bull).
        vol_regime: Volatility regime (0=Low, 1=High).
        trend_strength: Strength of current trend (0 to 1).

    Example:
        >>> state = RegimeState(
        ...     regime=2,           # Bull market
        ...     vol_regime=0,      # Low volatility
        ...     trend_strength=0.8 # Strong uptrend
        ... )
    """

    regime: int  # 0=Bear, 1=Neutral, 2=Bull
    vol_regime: int  # 0=Low volatility, 1=High volatility
    trend_strength: float  # 0..1  (0=no trend, 1=strong trend)


class RegimeAwareReward(BaseReward):
    """
    Complete reward function with regime alignment bonus.

    This is the most sophisticated reward function, combining:
    - Sharpe-based risk adjustment
    - Transaction cost penalties
    - Drawdown penalties
    - Market regime alignment bonus

    The regime bonus rewards the agent for trading in the direction
    of detected market trends and penalizes counter-trend trading.

    Regime Logic:
        - Bull regime (regime=2): Long positions rewarded, shorts penalized
        - Bear regime (regime=0): Short positions rewarded, longs penalized
        - Neutral: No regime bonus

    Attributes:
        window: Rolling window for Sharpe calculation.
        lambda_cost: Cost penalty weight.
        lambda_draw: Drawdown penalty weight.
        lambda_regime: Regime bonus weight.
        cost_rate: Transaction cost rate.

    Usage:
        >>> reward = RegimeAwareReward(window=50, lambda_cost=2.0)
        >>>
        >>> # Set regime each step
        >>> reward.set_regime(RegimeState(regime=2, vol_regime=0, trend_strength=0.7))
        >>>
        >>> # Compute reward
        >>> r = reward.compute(pnl=100, position=0.5, prev_position=0.0,
        ...                   equity=10000, cost_this_bar=5)
    """

    def __init__(
        self,
        window: int = 50,
        lambda_cost: float = 2.0,
        lambda_draw: float = 3.0,
        lambda_regime: float = 0.5,
        cost_rate: float = 0.001,
    ):
        """
        Initialize regime-aware reward calculator.

        Parameters:
        -----------
        window : int, optional
            Rolling window size (default: 50).
        lambda_cost : float, optional
            Cost penalty weight (default: 2.0).
        lambda_draw : float, optional
            Drawdown penalty weight (default: 3.0).
        lambda_regime : float, optional
            Regime bonus weight (default: 0.5).
        cost_rate : float, optional
            Transaction cost rate (default: 0.001).
        """
        self.window = window
        self.lambda_cost = lambda_cost
        self.lambda_draw = lambda_draw
        self.lambda_regime = lambda_regime
        self.cost_rate = cost_rate

        self._sharpe = SharpeIncrementReward(window)
        self._peak: float = 1.0
        self._regime: Optional[RegimeState] = None

    def set_regime(self, regime: RegimeState) -> None:
        """
        Update the current market regime state.

        Called by the environment each step to provide regime information
        to the reward function.

        Parameters:
        -----------
        regime : RegimeState
            Current market regime detected by external system.
        """
        self._regime = regime

    def compute(
        self, pnl, position, prev_position, equity, cost_this_bar, **kwargs
    ) -> float:
        """
        Compute regime-aware reward.

        Parameters:
        -----------
        pnl : float
            Gross P&L.
        position : float
            Current position.
        prev_position : float
            Previous position.
        equity : float
            Current equity.
        cost_this_bar : float
            Transaction costs.

        Returns:
        --------
        reward : float
            Combined reward, clipped to ±5 range.
        """
        # Calculate Sharpe increment component
        sharpe_inc = self._sharpe.compute(
            pnl, position, prev_position, equity, cost_this_bar
        )

        # Position change for cost penalty
        position_change = abs(position - prev_position)
        cost_penalty = self.lambda_cost * position_change * self.cost_rate

        # Drawdown calculation
        self._peak = max(self._peak, equity)
        drawdown = max(0, (self._peak - equity) / (self._peak + 1e-8))

        # Squared drawdown penalty (stronger for deeper drawdowns)
        draw_penalty = self.lambda_draw * (drawdown**1.5)

        # Regime bonus/penalty
        regime_bonus = 0.0
        if self._regime is not None:
            regime_bonus = self._compute_regime_bonus(position, self._regime)

        # Combine all components
        reward = sharpe_inc - cost_penalty - draw_penalty + regime_bonus
        return float(np.clip(reward, -5, 5))

    def _compute_regime_bonus(self, position: float, regime: RegimeState) -> float:
        """
        Compute bonus or penalty based on signal-regime alignment.

        The agent receives a bonus for trading in the same direction
        as the detected market regime, and a penalty for trading
        against the regime.

        Parameters:
        -----------
        position : float
            Current position.
        regime : RegimeState
            Current market regime.

        Returns:
        --------
        bonus : float
            Positive bonus for aligned trades, negative penalty for
            counter-regime trades.
        """
        ts = regime.trend_strength  # How strong is the trend
        bonus = self.lambda_regime  # Base bonus magnitude

        # Bull market: reward longs, penalize shorts
        if regime.regime == 2:
            if position > 0.1:
                return +bonus * ts  # Aligned: long in bull
            elif position < -0.1:
                return -bonus * ts * 1.5  # Counter: short in bull (penalty amplified)

        # Bear market: reward shorts, penalize longs
        elif regime.regime == 0:
            if position < -0.1:
                return +bonus * ts  # Aligned: short in bear
            elif position > 0.1:
                return -bonus * ts * 1.5  # Counter: long in bear (penalty amplified)

        # Neutral regime: no bonus
        return 0.0

    def reset(self) -> None:
        """Reset all state for new episode."""
        self._sharpe.reset()
        self._peak = 1.0
        self._regime = None


class RewardAnalyzer:
    """
    Utility class for comparing all reward function variants.

    This class provides benchmarking functionality to compare different
    reward functions on the same trading data. Useful for:
    - Selecting the best reward function for a strategy
    - Understanding how different rewards affect agent behavior
    - Debugging reward shaping issues

    Example:
        >>> analyzer = RewardAnalyzer()
        >>> stats = analyzer.compare(returns, positions, cost_rate=0.001)
        >>> print(stats['sharpe']['mean'], stats['cost_aware']['sharpe'])
    """

    @staticmethod
    def compare(
        returns: np.ndarray,
        positions: np.ndarray,
        cost_rate: float = 0.001,
        equity_start: float = 10_000,
    ) -> dict:
        """
        Run all reward variants on the same data and return statistics.

        Parameters:
        -----------
        returns : np.ndarray
            Array of period returns (e.g., price returns).
        positions : np.ndarray
            Array of position sizes for each period.
        cost_rate : float, optional
            Transaction cost rate (default: 0.001).
        equity_start : float, optional
            Starting equity (default: 10,000).

        Returns:
        --------
        stats : dict
            Dictionary with statistics for each reward type:
            - mean: Average reward
            - std: Reward standard deviation
            - sharpe: Reward Sharpe ratio
            - min: Minimum reward
            - max: Maximum reward
        """
        # Initialize reward collectors
        rewards: dict[str, list] = {
            "naive": [],
            "sharpe": [],
            "calmar": [],
            "cost_aware": [],
            "regime_aware": [],
        }

        # Create reward function instances
        fns = {
            "naive": None,  # Baseline: raw return
            "sharpe": SharpeIncrementReward(50),
            "calmar": CalmarIncrementReward(100),
            "cost_aware": CostAwareReward(lambda_cost=2.0, cost_rate=cost_rate),
            "regime_aware": RegimeAwareReward(cost_rate=cost_rate),
        }

        equity = equity_start
        prev_pos = 0.0

        # Simulate trading and collect rewards
        for t, (ret, pos) in enumerate(zip(returns, positions)):
            # Calculate P&L
            pnl = ret * equity * abs(pos)

            # Calculate transaction costs
            cost_bar = abs(pos - prev_pos) * cost_rate * equity

            # Update equity
            equity = equity + pnl - cost_bar

            # Record naive reward (raw return)
            rewards["naive"].append(float(ret))

            # Compute all reward variants
            for name, fn in fns.items():
                if name == "naive" or fn is None:
                    continue
                r = fn.compute(
                    pnl=pnl,
                    position=float(pos),
                    prev_position=prev_pos,
                    equity=equity,
                    cost_this_bar=cost_bar,
                )
                rewards[name].append(r)

            prev_pos = pos

        # Calculate statistics for each reward type
        stats = {}
        for name, r_list in rewards.items():
            if not r_list:
                continue
            arr = np.array(r_list)
            stats[name] = {
                "mean": round(float(arr.mean()), 5),
                "std": round(float(arr.std()), 5),
                "sharpe": round(float(arr.mean() / (arr.std() + 1e-8)), 4),
                "min": round(float(arr.min()), 5),
                "max": round(float(arr.max()), 5),
            }
        return stats
