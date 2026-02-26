"""
ANTI-BIAS FRAMEWORK – Risk-Adjusted Reward Functions
=====================================================
Replaces the naive return-based reward with risk-adjusted alternatives.

Naive reward (just raw return) causes the agent to:
- Take excessive risk for short-term gains
- Ignore transaction costs (churn)
- Ignore drawdowns

These reward functions fix that by penalising risk and costs.
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
    """Abstract interface for all reward function implementations."""

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
        """Compute the scalar reward for a single environment step."""

    def reset(self) -> None:
        """Reset internal state at the beginning of a new episode."""
        pass


class SharpeIncrementReward(BaseReward):
    """
    reward_t = (r_t - μ_rolling) / σ_rolling

    Normalises the return by local rolling volatility.
    This rewards consistent, low-volatility returns over lucky spikes.
    """

    def __init__(self, window: int = 50, risk_free_rate: float = 0.0):
        self.window = window
        self.rf = risk_free_rate / (
            365 * 24
        )  # per-bar risk-free rate (hourly fraction)
        self._ret_buf: deque = deque(maxlen=window)  # rolling window of net returns
        self._peak_equity: float = 0.0

    def compute(
        self, pnl, position, prev_position, equity, cost_this_bar, **kwargs
    ) -> float:
        net_ret = pnl - cost_this_bar  # return after transaction costs
        self._ret_buf.append(net_ret)

        if len(self._ret_buf) < 5:
            return float(
                np.clip(net_ret * 10, -1, 1)
            )  # fallback: simple scaled reward during warm-up

        arr = np.array(self._ret_buf)
        mu = arr.mean()  # rolling mean return
        sig = arr.std() + 1e-8  # rolling std dev + epsilon

        # Standardised excess return: reward = (r_t - rf - μ) / σ  (incremental Sharpe contribution)
        sharpe_increment = (net_ret - self.rf - mu) / sig
        return float(
            np.clip(sharpe_increment, -3, 3)
        )  # clip to ±3 σ for stable training

    def reset(self) -> None:
        self._ret_buf.clear()
        self._peak_equity = 0.0


class CalmarIncrementReward(BaseReward):
    """
    reward_t = r_t / (max_drawdown_rolling + ε)

    Asymmetric penalty for drawdowns: the deeper the drawdown,
    the harder it is to earn a positive reward. Encourages capital preservation.
    """

    def __init__(self, window: int = 100):
        self.window = window
        self._equity_buf: deque = deque(maxlen=window)
        self._peak: float = 0.0

    def compute(
        self, pnl, position, prev_position, equity, cost_this_bar, **kwargs
    ) -> float:
        net_ret = pnl - cost_this_bar  # return net of costs
        self._equity_buf.append(equity)

        if not self._equity_buf:
            return 0.0

        eq_arr = np.array(self._equity_buf)
        peak = eq_arr.max()
        trough = eq_arr.min()
        max_dd = (peak - trough) / (peak + 1e-8)  # rolling max drawdown magnitude
        max_dd = max(max_dd, 0.001)  # floor at 0.1% to avoid div/0 in benign regimes

        calmar_inc = (
            net_ret / max_dd
        )  # Calmar increment: positive reward divided by recent worst drawdown
        return float(np.clip(calmar_inc * 10, -3, 3))

    def reset(self) -> None:
        self._equity_buf.clear()
        self._peak = 0.0


class CostAwareReward(BaseReward):
    """
    reward_t = net_pnl - λ_cost × |Δpos| × cost_rate - λ_draw × drawdown_penalty

    Prevents churning (excessive trading) and penalises drawdowns.
    The cost penalty grows with every position change, discouraging
    the agent from trading for no reason.
    """

    def __init__(
        self,
        lambda_cost: float = 2.0,
        lambda_draw: float = 5.0,
        cost_rate: float = 0.001,
        window: int = 50,
    ):
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
        position_change = abs(
            position - prev_position
        )  # fraction of capital repositioned
        self._equity_buf.append(equity)

        net_pnl = pnl - cost_this_bar  # P&L after exchange fees
        churn_penalty = (
            self.lambda_cost * position_change * self.cost_rate
        )  # penalise unnecessary trades

        self._peak_equity = max(self._peak_equity, equity)  # update high-water mark
        drawdown = max(
            0, (self._peak_equity - equity) / (self._peak_equity + 1e-8)
        )  # current drawdown fraction
        draw_penalty = (
            self.lambda_draw * drawdown * max(0, -net_pnl)
        )  # extra penalty only during losing drawdown bars

        if position_change > 0.01:
            self._trade_count += 1  # count meaningful trades (ignore micro-adjustments)

        reward = net_pnl - churn_penalty - draw_penalty
        return float(
            np.clip(reward * 100, -3, 3)
        )  # scale to ±3 range for stable RL gradients

    def reset(self) -> None:
        self._equity_buf.clear()
        self._peak_equity = 1.0
        self._trade_count = 0


@dataclass
class RegimeState:
    """Market regime information passed into RegimeAwareReward."""

    regime: int  # 0=Bear, 1=Neutral, 2=Bull
    vol_regime: int  # 0=Low volatility, 1=High volatility
    trend_strength: float  # 0..1  (0=no trend, 1=strong trend)


class RegimeAwareReward(BaseReward):
    """
    Combines all reward components plus a regime-congruence bonus.

    Recommended for multi-timeframe systems.
    Bonus: if the agent trades in the direction of the detected market regime,
    it receives extra reward. Penalty if it trades against the regime.
    """

    def __init__(
        self,
        window: int = 50,
        lambda_cost: float = 2.0,
        lambda_draw: float = 3.0,
        lambda_regime: float = 0.5,
        cost_rate: float = 0.001,
    ):
        self.window = window
        self.lambda_cost = lambda_cost
        self.lambda_draw = lambda_draw
        self.lambda_regime = lambda_regime
        self.cost_rate = cost_rate

        self._sharpe = SharpeIncrementReward(window)
        self._peak: float = 1.0
        self._regime: Optional[RegimeState] = None

    def set_regime(self, regime: RegimeState) -> None:
        """Update the current market regime state (called by environment each step)."""
        self._regime = regime

    def compute(
        self, pnl, position, prev_position, equity, cost_this_bar, **kwargs
    ) -> float:
        sharpe_inc = self._sharpe.compute(
            pnl, position, prev_position, equity, cost_this_bar
        )

        position_change = abs(position - prev_position)
        cost_penalty = self.lambda_cost * position_change * self.cost_rate

        self._peak = max(self._peak, equity)
        drawdown = max(0, (self._peak - equity) / (self._peak + 1e-8))
        draw_penalty = self.lambda_draw * (drawdown**1.5)

        regime_bonus = 0.0
        if self._regime is not None:
            regime_bonus = self._compute_regime_bonus(position, self._regime)

        reward = sharpe_inc - cost_penalty - draw_penalty + regime_bonus
        return float(np.clip(reward, -5, 5))

    def _compute_regime_bonus(self, position: float, regime: RegimeState) -> float:
        """Compute bonus/penalty based on signal-regime alignment."""
        ts = regime.trend_strength
        bonus = self.lambda_regime

        if regime.regime == 2:  # Bull
            if position > 0.1:
                return +bonus * ts
            elif position < -0.1:
                return -bonus * ts * 1.5
        elif regime.regime == 0:  # Bear
            if position < -0.1:
                return +bonus * ts
            elif position > 0.1:
                return -bonus * ts * 1.5
        return 0.0

    def reset(self) -> None:
        self._sharpe.reset()
        self._peak = 1.0
        self._regime = None


class RewardAnalyzer:
    """Compares all reward function variants on the same data for benchmarking."""

    @staticmethod
    def compare(
        returns: np.ndarray,
        positions: np.ndarray,
        cost_rate: float = 0.001,
        equity_start: float = 10_000,
    ) -> dict:
        """Run all reward variants on the same return/position sequence and return statistics."""
        rewards: dict[str, list] = {
            "naive": [],
            "sharpe": [],
            "calmar": [],
            "cost_aware": [],
            "regime_aware": [],
        }

        fns = {
            "naive": None,
            "sharpe": SharpeIncrementReward(50),
            "calmar": CalmarIncrementReward(100),
            "cost_aware": CostAwareReward(lambda_cost=2.0, cost_rate=cost_rate),
            "regime_aware": RegimeAwareReward(cost_rate=cost_rate),
        }

        equity = equity_start
        prev_pos = 0.0

        for t, (ret, pos) in enumerate(zip(returns, positions)):
            pnl = ret * equity * abs(pos)
            cost_bar = abs(pos - prev_pos) * cost_rate * equity
            equity = equity + pnl - cost_bar

            rewards["naive"].append(float(ret))

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
