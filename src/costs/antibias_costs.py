"""
ANTI-BIAS FRAMEWORK – Transaction Costs
========================================
Realistic transaction cost engine for Spot and Futures markets.

Covers all cost components that reduce actual profitability:
- Exchange fees (maker/taker)
- Bid-ask spread
- Market impact (how large orders move the price)
- Funding rate (for futures positions held overnight)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

import numpy as np

logger = logging.getLogger("antibias.costs")


class MarketType(Enum):
    SPOT = auto()
    FUTURES = auto()


class Timeframe(Enum):
    M1 = "1m"
    M5 = "5m"
    M15 = "15m"
    H1 = "1h"
    H4 = "4h"


class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"


# Binance Fee Schedule (conservative, no VIP tier)
BINANCE_SPOT_FEES = {
    "taker": 0.0010,
    "maker": 0.0010,
}

BINANCE_FUTURES_FEES = {
    "taker": 0.00050,
    "maker": 0.00020,
}

# Funding Rate statistics (historical BTC perpetual, 8h period)
FUNDING_MEAN_8H = 0.0001
FUNDING_STD_8H = 0.0003
FUNDING_MAX_8H = 0.0075

# Spread multipliers per timeframe (shorter TF = wider relative spread)
SPREAD_MULT = {
    Timeframe.M1: 3.5,
    Timeframe.M5: 2.5,
    Timeframe.M15: 1.8,
    Timeframe.H1: 1.0,
    Timeframe.H4: 0.8,
}

# BTC/USDT typical bid-ask spread in basis points (1h baseline)
BASE_SPREAD_BPS = {
    "BTCUSDT": 1.0,
    "ETHUSDT": 1.5,
    "DEFAULT": 3.0,
}


@dataclass
class CostConfig:
    market_type: MarketType = MarketType.FUTURES
    timeframe: Timeframe = Timeframe.H1
    order_type: OrderType = OrderType.MARKET
    symbol: str = "BTCUSDT"
    fee_override: Optional[float] = None
    spread_override: Optional[float] = None
    enable_impact: bool = True
    impact_coefficient: float = 0.1
    enable_funding: bool = True
    holding_bars: int = 1
    funding_scenario: str = "mean"
    slippage_model: str = "sqrt"
    fixed_slippage_bps: float = 3.0


@dataclass
class CostBreakdown:
    fee_one_way: float
    spread_one_way: float
    impact_one_way: float
    funding_holding: float
    total_one_way: float
    total_roundtrip: float
    min_required_edge: float
    notional_usd: float
    config: CostConfig

    def __repr__(self) -> str:
        lines = [
            f"┌─ Cost Breakdown ({self.config.market_type.name} "
            f"{self.config.timeframe.value} {self.config.symbol}) ─",
            f"│  Fee (one-way):     {self.fee_one_way * 100:6.4f}%",
            f"│  Spread (one-way):  {self.spread_one_way * 100:6.4f}%",
            f"│  Impact (one-way):  {self.impact_one_way * 100:6.4f}%",
            f"│  Funding (hold):    {self.funding_holding * 100:6.4f}%",
            f"│  ─────────────────────────────────",
            f"│  Total Round-Trip:  {self.total_roundtrip * 100:6.4f}%",
            f"│  Min. Edge Needed:  {self.min_required_edge * 100:6.4f}%",
            f"└──────────────────────────────────────",
        ]
        return "\n".join(lines)


class TransactionCostEngine:
    """
    Computes complete round-trip transaction costs for a single trade.

    A round-trip includes entry and exit costs:
    - Fee (paid twice: open + close)
    - Spread (paid twice: open + close)
    - Market impact (paid twice)
    - Funding rate (paid once per holding period)
    """

    def __init__(self, config: Optional[CostConfig] = None):
        self.config = config or CostConfig()

    def total_cost(
        self,
        price: float,
        quantity: float,
        adv: float = 50_000_000,
        volatility: float = 0.02,
        side: str = "buy",
    ) -> CostBreakdown:
        """
        Compute total transaction costs for a single trade.

        Args:
            price: Order price in USD
            quantity: Order size in base asset units
            adv: 24-hour average daily volume in USD
            volatility: Daily volatility (e.g. 0.02 = 2%)
            side: Trade direction ('buy' or 'sell')
        """
        cfg = self.config
        notional = price * quantity  # total trade value in USD

        fee = self._compute_fee()
        spread = self._compute_spread()

        impact = 0.0
        if cfg.enable_impact and adv > 0:
            impact = self._compute_impact(
                notional, adv, volatility
            )  # market-impact slippage

        funding = 0.0
        if cfg.enable_funding and cfg.market_type == MarketType.FUTURES:
            funding = (
                self._compute_funding()
            )  # perpetual funding rate cost (futures only)

        one_way = fee + spread + impact  # cost for one leg (entry or exit)
        roundtrip = (
            2 * (fee + spread + impact) + funding
        )  # full round-trip (entry + exit + funding)
        min_edge_req = (
            roundtrip * 1.5
        )  # required edge = 1.5× round-trip cost (safety margin)

        return CostBreakdown(
            fee_one_way=fee,
            spread_one_way=spread,
            impact_one_way=impact,
            funding_holding=funding,
            total_one_way=one_way,
            total_roundtrip=roundtrip,
            min_required_edge=min_edge_req,
            notional_usd=notional,
            config=cfg,
        )

    def _compute_fee(self) -> float:
        cfg = self.config
        if cfg.fee_override is not None:
            return cfg.fee_override

        if cfg.market_type == MarketType.SPOT:
            table = BINANCE_SPOT_FEES
        else:
            table = BINANCE_FUTURES_FEES

        return table["taker"] if cfg.order_type == OrderType.MARKET else table["maker"]

    def _compute_spread(self) -> float:
        cfg = self.config
        if cfg.spread_override is not None:
            return (
                cfg.spread_override / 2
            )  # divide by 2: one-way is half the full spread

        base_bps = BASE_SPREAD_BPS.get(cfg.symbol, BASE_SPREAD_BPS["DEFAULT"])
        mult = SPREAD_MULT.get(
            cfg.timeframe, 1.0
        )  # shorter timeframes have wider relative spreads
        return (
            (base_bps * mult) / 10_000 / 2
        )  # convert bps to fraction, then one-way half-spread

    def _compute_impact(
        self,
        notional: float,
        adv: float,
        volatility: float,
    ) -> float:
        """Square-Root Market Impact Model."""
        cfg = self.config
        if cfg.slippage_model == "fixed":
            return cfg.fixed_slippage_bps / 10_000  # convert bps to fraction

        participation = notional / (
            adv + 1e-8
        )  # fraction of daily volume; 1e-8 avoids div-by-zero

        if cfg.slippage_model == "sqrt":
            # Almgren-Chriss square-root model: impact scales with sqrt(participation)
            impact = cfg.impact_coefficient * volatility * np.sqrt(participation)
        elif cfg.slippage_model == "linear":
            impact = (
                cfg.impact_coefficient * volatility * participation
            )  # simplified linear model
        else:
            impact = 0.0

        return min(impact, 0.01)  # cap at 1% to prevent unrealistic estimates

    def _compute_funding(self) -> float:
        """Compute funding rate costs for the holding period."""
        cfg = self.config
        # number of bars in one 8-hour funding interval per timeframe
        bars_per_8h = {
            Timeframe.M1: 480,
            Timeframe.M5: 96,
            Timeframe.M15: 32,
            Timeframe.H1: 8,
            Timeframe.H4: 2,
        }
        b_per_8h = bars_per_8h.get(cfg.timeframe, 8)
        n_funding_payments = (
            cfg.holding_bars / b_per_8h
        )  # fractional number of 8h funding periods

        if cfg.funding_scenario == "mean":
            rate_per_8h = FUNDING_MEAN_8H  # typical market conditions
        elif cfg.funding_scenario == "stress":
            rate_per_8h = (
                FUNDING_MEAN_8H + 2 * FUNDING_STD_8H
            )  # 2-sigma stress scenario
        elif cfg.funding_scenario == "extreme":
            rate_per_8h = FUNDING_MAX_8H  # historical maximum observed rate
        else:
            rate_per_8h = FUNDING_MEAN_8H

        return abs(
            rate_per_8h * n_funding_payments
        )  # abs() because long/short both pay funding cost


class BreakEvenAnalyzer:
    """Computes the minimum required edge (break-even) for various trading scenarios."""

    @staticmethod
    def analyze_all_scenarios(
        symbol: str = "BTCUSDT",
        order_size_usd: float = 10_000,
        adv_usd: float = 500_000_000,
        daily_vol: float = 0.025,
    ) -> str:
        scenarios = [
            (MarketType.SPOT, Timeframe.M1, OrderType.MARKET),
            (MarketType.SPOT, Timeframe.M5, OrderType.MARKET),
            (MarketType.FUTURES, Timeframe.M5, OrderType.MARKET),
            (MarketType.FUTURES, Timeframe.H1, OrderType.MARKET),
            (MarketType.FUTURES, Timeframe.H1, OrderType.LIMIT),
            (MarketType.FUTURES, Timeframe.H4, OrderType.LIMIT),
        ]

        lines = [
            "═" * 72,
            "  BREAK-EVEN ANALYSIS",
            f"  Symbol: {symbol}  |  Order: ${order_size_usd:,.0f}  |  ADV: ${adv_usd / 1e6:.0f}M",
            "═" * 72,
            f"  {'Market':<8} {'TF':<5} {'Type':<7} {'RT Cost':>8} {'Min Edge':>9} {'Trades/d@BE':>12}",
            "─" * 72,
        ]

        for market, tf, ot in scenarios:
            cfg = CostConfig(
                market_type=market,
                timeframe=tf,
                order_type=ot,
                symbol=symbol,
                holding_bars=1,
            )
            eng = TransactionCostEngine(cfg)
            cost = eng.total_cost(
                price=30_000,
                quantity=order_size_usd / 30_000,
                adv=adv_usd,
                volatility=daily_vol,
            )
            daily_vol_abs = 30_000 * daily_vol
            trades_be = daily_vol_abs / (order_size_usd * cost.total_roundtrip + 1e-8)
            lines.append(
                f"  {market.name:<8} {tf.value:<5} {ot.value:<7} "
                f"{cost.total_roundtrip * 100:>7.4f}% "
                f"{cost.min_required_edge * 100:>8.4f}% "
                f"{trades_be:>11.1f}"
            )

        lines += [
            "═" * 72,
            "  Interpretation:",
            "  ─────────────────────────────────────────────────────────────",
            "  1m MARKET orders: >0.15% edge per trade required → extremely hard",
            "  1h LIMIT orders:  ~0.04% edge required → achievable with good signal",
            "  4h LIMIT orders:  ~0.03% edge required → realistic target",
            "═" * 72,
        ]
        return "\n".join(lines)
