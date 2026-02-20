"""
ANTI-BIAS FRAMEWORK – Risk-Adjusted Reward Functions
=====================================================
Ersetzt naiven Return-Reward durch risikobereinigte Funktionen.
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
    """Interface für alle Reward-Funktionen."""

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
        """Berechnet den Reward für einen Schritt."""

    def reset(self) -> None:
        """Reset am Episode-Anfang."""
        pass


class SharpeIncrementReward(BaseReward):
    """
    reward_t = (r_t - μ_rolling) / σ_rolling

    Normalisiert Return durch lokale Volatilität.
    """

    def __init__(self, window: int = 50, risk_free_rate: float = 0.0):
        self.window = window
        self.rf = risk_free_rate / (365 * 24)
        self._ret_buf: deque = deque(maxlen=window)
        self._peak_equity: float = 0.0

    def compute(
        self, pnl, position, prev_position, equity, cost_this_bar, **kwargs
    ) -> float:
        net_ret = pnl - cost_this_bar
        self._ret_buf.append(net_ret)

        if len(self._ret_buf) < 5:
            return float(np.clip(net_ret * 10, -1, 1))

        arr = np.array(self._ret_buf)
        mu = arr.mean()
        sig = arr.std() + 1e-8

        sharpe_increment = (net_ret - self.rf - mu) / sig
        return float(np.clip(sharpe_increment, -3, 3))

    def reset(self) -> None:
        self._ret_buf.clear()
        self._peak_equity = 0.0


class CalmarIncrementReward(BaseReward):
    """
    reward_t = r_t / (max_drawdown_rolling + ε)

    Asymmetrische Bestrafung von Drawdowns.
    """

    def __init__(self, window: int = 100):
        self.window = window
        self._equity_buf: deque = deque(maxlen=window)
        self._peak: float = 0.0

    def compute(
        self, pnl, position, prev_position, equity, cost_this_bar, **kwargs
    ) -> float:
        net_ret = pnl - cost_this_bar
        self._equity_buf.append(equity)

        if not self._equity_buf:
            return 0.0

        eq_arr = np.array(self._equity_buf)
        peak = eq_arr.max()
        trough = eq_arr.min()
        max_dd = (peak - trough) / (peak + 1e-8)
        max_dd = max(max_dd, 0.001)

        calmar_inc = net_ret / max_dd
        return float(np.clip(calmar_inc * 10, -3, 3))

    def reset(self) -> None:
        self._equity_buf.clear()
        self._peak = 0.0


class CostAwareReward(BaseReward):
    """
    reward_t = net_pnl - λ_cost × |Δpos| × cost_rate - λ_draw × drawdown_penalty

    Verhindert Churning und bestraft Drawdowns.
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
        position_change = abs(position - prev_position)
        self._equity_buf.append(equity)

        net_pnl = pnl - cost_this_bar
        churn_penalty = self.lambda_cost * position_change * self.cost_rate

        self._peak_equity = max(self._peak_equity, equity)
        drawdown = max(0, (self._peak_equity - equity) / (self._peak_equity + 1e-8))
        draw_penalty = self.lambda_draw * drawdown * max(0, -net_pnl)

        if position_change > 0.01:
            self._trade_count += 1

        reward = net_pnl - churn_penalty - draw_penalty
        return float(np.clip(reward * 100, -3, 3))

    def reset(self) -> None:
        self._equity_buf.clear()
        self._peak_equity = 1.0
        self._trade_count = 0


@dataclass
class RegimeState:
    """Regime-Information für RegimeAwareReward."""

    regime: int  # 0=Bear, 1=Neutral, 2=Bull
    vol_regime: int  # 0=Low Vol, 1=High Vol
    trend_strength: float  # 0..1


class RegimeAwareReward(BaseReward):
    """
    Kombiniert alle Reward-Komponenten + Regime-Kongruenz-Bonus.

    Empfohlen für Multi-Timeframe Systeme.
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
        """Aktualisiert den Regime-State."""
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
        """Berechnet Bonus/Malus basierend auf Signal-Regime-Kongruenz."""
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
    """Vergleicht verschiedene Reward-Funktionen."""

    @staticmethod
    def compare(
        returns: np.ndarray,
        positions: np.ndarray,
        cost_rate: float = 0.001,
        equity_start: float = 10_000,
    ) -> dict:
        """Berechnet alle Reward-Varianten auf denselben Daten."""
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
