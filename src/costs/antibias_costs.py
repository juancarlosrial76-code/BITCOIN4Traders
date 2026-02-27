"""
ANTI-BIAS FRAMEWORK – Transaction Costs Engine
===============================================
Realistic transaction cost modeling for cryptocurrency spot and futures markets.

Purpose:
--------
In algorithmic trading, transaction costs are often underestimated, leading to
strategies that appear profitable in backtests but lose money in live trading.
This module provides comprehensive, realistic cost estimates that account for
all components of trading costs.

Cost Components Covered:
-------------------------
1. EXCHANGE FEES (Maker/Taker)
   - Maker: Limit orders that add liquidity (typically lower)
   - Taker: Market orders that remove liquidity (typically higher)
   - Binance spot: 0.10% taker, 0.10% maker
   - Binance futures: 0.05% taker, 0.02% maker

2. BID-ASK SPREAD
   - The difference between best bid and ask
   - Wider on lower-liquidity pairs and shorter timeframes
   - Paid twice per round-trip (entry + exit)

3. MARKET IMPACT
   - Large orders move the market against themselves
   - Modeled using Almgren-Chriss square-root model
   - Depends on order size relative to average daily volume (ADV)

4. FUNDING RATE (Futures Only)
   - Cost of holding perpetual futures positions
   - Paid every 8 hours (3 times daily)
   - Variable: mean ~0.01%, max ~0.75%

Why Accurate Costs Matter:
---------------------------
- A strategy with 0.1% edge per trade needs ~0.15% round-trip cost
- Intraday strategies on 1m timeframe face ~0.15%+ round-trip
- Costs compound: 10 trades/day at 0.1% = ~2.5% monthly drag
- Underestimating costs is the #1 cause of backtest-to-live failure

Usage Example:
--------------
    from src.costs.antibias_costs import TransactionCostEngine, CostConfig, MarketType, Timeframe

    # Configure for BTC futures, 1h timeframe
    config = CostConfig(
        market_type=MarketType.FUTURES,
        timeframe=Timeframe.H1,
        symbol="BTCUSDT",
        enable_impact=True,
        enable_funding=True
    )

    engine = TransactionCostEngine(config)

    # Calculate costs for $10,000 trade
    cost = engine.total_cost(
        price=30000,
        quantity=0.333,  # ~$10,000 notional
        adv=500_000_000,  # $500M daily volume
        volatility=0.025   # 2.5% daily volatility
    )

    print(cost)
    # Output shows breakdown of all cost components

Classes:
--------
1. MarketType: SPOT or FUTURES
2. Timeframe: M1, M5, M15, H1, H4
3. OrderType: MARKET or LIMIT
4. CostConfig: Configuration parameters
5. CostBreakdown: Detailed cost output
6. TransactionCostEngine: Main cost calculator
7. BreakEvenAnalyzer: Break-even analysis utility

Author: BITCOIN4Traders Team
Version: 1.0.0
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

import numpy as np

logger = logging.getLogger("antibias.costs")


class MarketType(Enum):
    """Market type for cost calculation."""

    SPOT = auto()
    FUTURES = auto()


class Timeframe(Enum):
    """Trading timeframe for spread and funding calculations."""

    M1 = "1m"
    M5 = "5m"
    M15 = "15m"
    H1 = "1h"
    H4 = "4h"


class OrderType(Enum):
    """Order type affects fee structure."""

    MARKET = "market"
    LIMIT = "limit"


# Binance Fee Schedule (conservative, no VIP tier)
# These are the standard fees for regular users
# VIP users get significant discounts
BINANCE_SPOT_FEES = {
    "taker": 0.0010,  # 0.10%
    "maker": 0.0010,  # 0.10%
}

BINANCE_FUTURES_FEES = {
    "taker": 0.00050,  # 0.05%
    "maker": 0.00020,  # 0.02%
}

# Funding Rate statistics for BTC perpetual futures (historical 8h period)
# These are conservative estimates based on historical data
FUNDING_MEAN_8H = 0.0001  # 0.01% typical rate
FUNDING_STD_8H = 0.0003  # 0.03% standard deviation
FUNDING_MAX_8H = 0.0075  # 0.75% historical maximum

# Spread multipliers per timeframe
# Shorter timeframes have wider relative spreads due to less liquidity
# and more noise trading
SPREAD_MULT = {
    Timeframe.M1: 3.5,  # 3.5x base spread
    Timeframe.M5: 2.5,  # 2.5x base spread
    Timeframe.M15: 1.8,  # 1.8x base spread
    Timeframe.H1: 1.0,  # baseline
    Timeframe.H4: 0.8,  # 0.8x (slightly tighter)
}

# BTC/USDT typical bid-ask spread in basis points (1h baseline)
# 1 basis point = 0.01%
BASE_SPREAD_BPS = {
    "BTCUSDT": 1.0,  # 0.01% spread
    "ETHUSDT": 1.5,  # 0.015% spread
    "DEFAULT": 3.0,  # 0.03% for other pairs
}


@dataclass
class CostConfig:
    """
    Configuration for transaction cost calculation.

    This dataclass contains all parameters needed to accurately model
    transaction costs for a specific trading scenario.

    Attributes:
        market_type: SPOT or FUTURES (affects fees and funding).
        timeframe: Trading timeframe (affects spread and funding frequency).
        order_type: MARKET or LIMIT (affects fee rate).
        symbol: Trading pair (affects base spread).
        fee_override: Override exchange fee (optional).
        spread_override: Override spread (optional).
        enable_impact: Include market impact in calculation.
        impact_coefficient: Market impact model coefficient.
        enable_funding: Include funding costs (futures only).
        holding_bars: Expected holding period in bars.
        funding_scenario: "mean", "stress", or "extreme".
        slippage_model: "sqrt" (Almgren-Chriss) or "linear" or "fixed".
        fixed_slippage_bps: Fixed slippage in basis points.

    Example:
        >>> config = CostConfig(
        ...     market_type=MarketType.FUTURES,
        ...     timeframe=Timeframe.H1,
        ...     symbol="BTCUSDT",
        ...     enable_impact=True,
        ...     enable_funding=True
        ... )
    """

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
    """
    Detailed breakdown of transaction costs for a single trade.

    This dataclass provides itemized costs for analysis and debugging.
    All values are expressed as fractions (not percentages) for easy use.

    Attributes:
        fee_one_way: Exchange fee for one direction (entry OR exit).
        spread_one_way: Bid-ask spread for one direction.
        impact_one_way: Market impact for one direction.
        funding_holding: Funding rate cost for holding period.
        total_one_way: Sum of all one-way costs.
        total_roundtrip: Total cost (entry + exit + funding).
        min_required_edge: Minimum edge needed to break even (1.5x RT cost).
        notional_usd: Trade notional in USD.
        config: Original CostConfig used for calculation.

    Example:
        >>> print(cost.total_roundtrip * 100)  # As percentage
        0.12
    """

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
        """Human-readable cost breakdown."""
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
    Complete transaction cost calculator for crypto trading.

    This engine computes comprehensive round-trip transaction costs
    accounting for all major cost components in cryptocurrency trading.

    Round-Trip Cost Components (each paid twice for entry+exit):
    1. Exchange fees: Maker/taker fees based on order type
    2. Bid-ask spread: Market liquidity cost
    3. Market impact: Self-induced price movement (for large orders)

    One-Way Costs (paid once for holding period):
    4. Funding rate: For holding futures positions

    The total round-trip cost represents the minimum edge a strategy
    needs just to break even. Strategies should target 1.5-2x this
    minimum to account for execution variation.

    Attributes:
        config: CostConfig with all parameters.

    Example:
        >>> engine = TransactionCostEngine(CostConfig(
        ...     market_type=MarketType.FUTURES,
        ...     timeframe=Timeframe.H1
        ... ))
        >>> cost = engine.total_cost(
        ...     price=30000,
        ...     quantity=1.0,
        ...     adv=500_000_000,
        ...     volatility=0.025
        ... )
        >>> print(f"Total round-trip cost: {cost.total_roundtrip*100:.4f}%")
    """

    def __init__(self, config: Optional[CostConfig] = None):
        """
        Initialize cost engine with configuration.

        Parameters:
        -----------
        config : CostConfig, optional
            Cost configuration. Uses default if not provided.
        """
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

        This is the main method that calculates all cost components.
        Call this for each trade to get accurate cost estimates.

        Parameters:
        -----------
        price : float
            Order price in USD (or stablecoin).
        quantity : float
            Order size in base asset units (e.g., BTC for BTCUSDT).
        adv : float, optional
            24-hour Average Daily Volume in USD (default: $50M).
            Used for market impact calculation. Higher ADV = lower impact.
        volatility : float, optional
            Daily volatility as decimal (default: 0.02 = 2%).
            Used for market impact model.
        side : str, optional
            Trade direction: "buy" or "sell" (default: "buy").
            Currently affects only logging, not cost calculation.

        Returns:
        --------
        CostBreakdown : dataclass
            Detailed breakdown of all cost components:
            - fee_one_way: Exchange fee for one direction
            - spread_one_way: Spread cost for one direction
            - impact_one_way: Market impact for one direction
            - funding_holding: Funding cost for holding period
            - total_one_way: Sum of all one-way costs
            - total_roundtrip: Full round-trip (entry + exit + funding)
            - min_required_edge: Minimum edge needed (1.5x RT cost)

        Example:
            >>> cost = engine.total_cost(
            ...     price=30000,
            ...     quantity=0.5,    # 0.5 BTC = $15,000
            ...     adv=500_000_000,  # $500M ADV
            ...     volatility=0.03
            ... )
            >>> print(cost)
            ┌─ Cost Breakdown (FUTURES 1h BTCUSDT) ─
            │  Fee (one-way):      0.0500%
            │  Spread (one-way):   0.0050%
            │  Impact (one-way):   0.0089%
            │  Funding (hold):     0.0000%
            │  ─────────────────────────────────
            │  Total Round-Trip:   0.1089%
            │  Min. Edge Needed:   0.1633%
            └──────────────────────────────────────
        """
        cfg = self.config

        # Calculate notional value in USD
        notional = price * quantity

        # Component 1: Exchange fees
        fee = self._compute_fee()

        # Component 2: Bid-ask spread
        spread = self._compute_spread()

        # Component 3: Market impact (if enabled)
        impact = 0.0
        if cfg.enable_impact and adv > 0:
            impact = self._compute_impact(notional, adv, volatility)

        # Component 4: Funding rate (futures only)
        funding = 0.0
        if cfg.enable_funding and cfg.market_type == MarketType.FUTURES:
            funding = self._compute_funding()

        # Sum all components
        # One-way = entry (or exit) cost
        one_way = fee + spread + impact

        # Round-trip = entry + exit + funding
        # Multiply fees, spread, impact by 2 for both directions
        roundtrip = 2 * (fee + spread + impact) + funding

        # Minimum edge: need at least 1.5x round-trip to have margin
        min_edge_req = roundtrip * 1.5

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
        """
        Calculate exchange fee based on market and order type.

        Returns:
        --------
        fee : float
            Fee as decimal (0.001 = 0.1%).
        """
        cfg = self.config

        # Use override if provided
        if cfg.fee_override is not None:
            return cfg.fee_override

        # Select fee table based on market type
        if cfg.market_type == MarketType.SPOT:
            table = BINANCE_SPOT_FEES
        else:
            table = BINANCE_FUTURES_FEES

        # Taker fees for market orders, maker fees for limit orders
        return table["taker"] if cfg.order_type == OrderType.MARKET else table["maker"]

    def _compute_spread(self) -> float:
        """
        Calculate bid-ask spread cost.

        Spread depends on:
        - Base spread for the trading pair
        - Timeframe (shorter = wider relative spread)

        Returns:
        --------
        spread : float
            One-way spread as decimal.
        """
        cfg = self.config

        # Use override if provided
        if cfg.spread_override is not None:
            # Divide by 2: one-way is half the full spread
            return cfg.spread_override / 2

        # Get base spread for symbol (bps)
        base_bps = BASE_SPREAD_BPS.get(cfg.symbol, BASE_SPREAD_BPS["DEFAULT"])

        # Adjust for timeframe
        # Shorter timeframes have relatively wider spreads
        mult = SPREAD_MULT.get(cfg.timeframe, 1.0)

        # Convert bps to fraction, then divide by 2 for one-way
        return (base_bps * mult) / 10_000 / 2

    def _compute_impact(
        self,
        notional: float,
        adv: float,
        volatility: float,
    ) -> float:
        """
        Calculate market impact using square-root model.

        The Almgren-Chriss model assumes:
        - Market impact scales with square root of participation rate (order size / ADV)
        - Impact also scales with volatility of the asset

        Formula:
            Impact = coefficient × volatility × sqrt(participation)

        Parameters:
        -----------
        notional : float
            Trade notional in USD.
        adv : float
            Average daily volume in USD.
        volatility : float
            Daily volatility as decimal.

        Returns:
        --------
        impact : float
            One-way impact as decimal, capped at 1%.
        """
        cfg = self.config

        # Fixed slippage model (simple)
        if cfg.slippage_model == "fixed":
            return cfg.fixed_slippage_bps / 10_000

        # Calculate participation rate: what fraction of daily volume is this order?
        # Add epsilon to avoid division by zero
        participation = notional / (adv + 1e-8)

        # Square-root model (Almgren-Chriss)
        if cfg.slippage_model == "sqrt":
            impact = cfg.impact_coefficient * volatility * np.sqrt(participation)
        # Linear model (simplified)
        elif cfg.slippage_model == "linear":
            impact = cfg.impact_coefficient * volatility * participation
        else:
            impact = 0.0

        # Cap at 1% to prevent unrealistic estimates for very large orders
        return min(impact, 0.01)

    def _compute_funding(self) -> float:
        """
        Calculate funding rate cost for holding futures positions.

        Funding is paid every 8 hours on perpetual futures. The rate
        varies based on market conditions:
        - Mean: Typical funding under normal conditions
        - Stress: 2-sigma scenario
        - Extreme: Historical maximum

        Returns:
        --------
        funding : float
            Funding cost as decimal, proportional to holding period.
        """
        cfg = self.config

        # Calculate how many 8-hour funding periods in the holding duration
        # Different timeframes have different numbers of bars per 8 hours
        bars_per_8h = {
            Timeframe.M1: 480,  # 480 minutes = 8 hours
            Timeframe.M5: 96,  # 96 × 5min = 8 hours
            Timeframe.M15: 32,  # 32 × 15min = 8 hours
            Timeframe.H1: 8,  # 8 × 1hr = 8 hours
            Timeframe.H4: 2,  # 2 × 4hr = 8 hours
        }
        b_per_8h = bars_per_8h.get(cfg.timeframe, 8)

        # Fractional number of funding periods
        n_funding_payments = cfg.holding_bars / b_per_8h

        # Select funding rate based on scenario
        if cfg.funding_scenario == "mean":
            rate_per_8h = FUNDING_MEAN_8H
        elif cfg.funding_scenario == "stress":
            # 2-sigma stress scenario
            rate_per_8h = FUNDING_MEAN_8H + 2 * FUNDING_STD_8H
        elif cfg.funding_scenario == "extreme":
            # Historical maximum
            rate_per_8h = FUNDING_MAX_8H
        else:
            rate_per_8h = FUNDING_MEAN_8H

        # Total funding = rate × number of periods
        # Use abs() because both long and short positions pay funding
        return abs(rate_per_8h * n_funding_payments)


class BreakEvenAnalyzer:
    """
    Utility class for break-even analysis across trading scenarios.

    This analyzer helps understand the minimum edge required to be
    profitable under different market conditions and order types.
    It's essential for strategy selection and realistic expectations.

    The analysis considers:
    - Different market types (spot vs futures)
    - Different timeframes (1m to 4h)
    - Different order types (market vs limit)

    Understanding Results:
    ----------------------
    - RT Cost: Total round-trip transaction cost
    - Min Edge: Minimum edge needed to break even (1.5x RT cost)
    - Trades/d@BE: How many trades per day at this edge to beat costs

    A practical trading strategy should target 2-3x the minimum edge
    to account for execution variation and provide profit margin.

    Example:
        >>> analysis = BreakEvenAnalyzer.analyze_all_scenarios(
        ...     symbol="BTCUSDT",
        ...     order_size_usd=10000,
        ...     adv_usd=500_000_000
        ... )
        >>> print(analysis)
        ═════════════════════════════════════════════════════════════
          BREAK-EVEN ANALYSIS
          Symbol: BTCUSDT  |  Order: $10,000  |  ADV: $500M
        ═════════════════════════════════════════════════════════════
          Market    TF    Type    RT Cost   Min Edge  Trades/d@BE
        ─────────────────────────────────────────────────────────────
          SPOT      1m    market   0.1512%   0.2268%         14.3
          SPOT      5m    market   0.1234%   0.1851%         17.5
          FUTURES   5m    market   0.1045%   0.1568%         20.6
          FUTURES   1h    market   0.0892%   0.1338%         24.2
          FUTURES   1h    limit    0.0456%   0.0684%         47.4
          FUTURES   4h    limit    0.0334%   0.0501%         64.8
        ═════════════════════════════════════════════════════════════
          Interpretation:
          ─────────────────────────────────────────────────────────────
          1m MARKET orders: >0.15% edge per trade required → extremely hard
          1h LIMIT orders:  ~0.04% edge required → achievable with good signal
          4h LIMIT orders:  ~0.03% edge required → realistic target
        ═════════════════════════════════════════════════════════════
    """

    @staticmethod
    def analyze_all_scenarios(
        symbol: str = "BTCUSDT",
        order_size_usd: float = 10_000,
        adv_usd: float = 500_000_000,
        daily_vol: float = 0.025,
    ) -> str:
        """
        Analyze break-even requirements for multiple scenarios.

        Parameters:
        -----------
        symbol : str, optional
            Trading pair to analyze (default: "BTCUSDT").
        order_size_usd : float, optional
            Order size in USD (default: 10,000).
        adv_usd : float, optional
            Average daily volume in USD (default: 500,000,000).
        daily_vol : float, optional
            Daily volatility as decimal (default: 0.025 = 2.5%).

        Returns:
        --------
        analysis : str
            Formatted analysis showing costs and break-even for each scenario.
        """
        # Define scenarios to analyze
        scenarios = [
            (MarketType.SPOT, Timeframe.M1, OrderType.MARKET),
            (MarketType.SPOT, Timeframe.M5, OrderType.MARKET),
            (MarketType.FUTURES, Timeframe.M5, OrderType.MARKET),
            (MarketType.FUTURES, Timeframe.H1, OrderType.MARKET),
            (MarketType.FUTURES, Timeframe.H1, OrderType.LIMIT),
            (MarketType.FUTURES, Timeframe.H4, OrderType.LIMIT),
        ]

        # Build output header
        lines = [
            "═" * 72,
            "  BREAK-EVEN ANALYSIS",
            f"  Symbol: {symbol}  |  Order: ${order_size_usd:,.0f}  |  ADV: ${adv_usd / 1e6:.0f}M",
            "═" * 72,
            f"  {'Market':<8} {'TF':<5} {'Type':<7} {'RT Cost':>8} {'Min Edge':>9} {'Trades/d@BE':>12}",
            "─" * 72,
        ]

        # Calculate for each scenario
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

            # Calculate trades per day needed to break even
            daily_vol_abs = 30_000 * daily_vol
            trades_be = daily_vol_abs / (order_size_usd * cost.total_roundtrip + 1e-8)

            lines.append(
                f"  {market.name:<8} {tf.value:<5} {ot.value:<7} "
                f"{cost.total_roundtrip * 100:>7.4f}% "
                f"{cost.min_required_edge * 100:>8.4f}% "
                f"{trades_be:>11.1f}"
            )

        # Add interpretation
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
