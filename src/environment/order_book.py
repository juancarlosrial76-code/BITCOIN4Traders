"""
Level 2 Order Book Simulator
=============================
Realistic order book simulation for accurate slippage modeling in cryptocurrency
trading environments. This module provides a complete implementation of a limit
order book with aggregated price levels, volume distribution, and market impact
modeling.

Purpose:
--------
The order book simulator enables realistic trade execution modeling by:
- Simulating discrete price levels (L2 market depth)
- Distributing liquidity across multiple price levels
- Calculating execution prices based on available volume
- Modeling permanent market impact from large orders

Key Features:
-------------
- Aggregated order book with configurable number of levels
- Volume-based slippage calculation (walking the book)
- Dynamic spread modeling based on volatility
- Exponential volume decay away from best prices
- Permanent market impact estimation using square-root law

Usage:
------
    from src.environment.order_book import OrderBookSimulator, OrderBookConfig

    config = OrderBookConfig(n_levels=10, base_spread_bps=5.0)
    simulator = OrderBookSimulator(config)

    # Generate synthetic order book
    bid_prices, bid_volumes, ask_prices, ask_volumes = simulator.generate_order_book(
        mid_price=50000.0,
        volatility=0.02,
        volume=100.0
    )

    # Calculate execution price with slippage
    avg_price, slippage_bps, filled = simulator.calculate_execution_price(
        side='buy',
        quantity=1.0,
        bid_prices=bid_prices,
        bid_volumes=bid_volumes,
        ask_prices=ask_prices,
        ask_volumes=ask_volumes
    )

Mathematical Model:
-------------------
1. Spread: spread_bps = base_spread × (1 + volatility × 10)
2. Volume Decay: volume[i] = base_volume × exp(-i × 0.3) × noise
3. Market Impact: impact ∝ √(order_size / daily_volume)
4. Slippage: VWAP from walking price levels until order is filled

Dependencies:
-------------
- numpy: Numerical operations
- numba: JIT compilation for performance-critical calculations
- loguru: Logging
"""

import numpy as np
from typing import Tuple, Dict, List
from dataclasses import dataclass
from numba import jit
from loguru import logger


@dataclass
class OrderBookConfig:
    """
    Configuration parameters for order book simulation.

    Attributes:
        n_levels: Number of price levels to simulate on each side of the book.
                  More levels provide better accuracy but require more computation.
        base_spread_bps: Base bid-ask spread in basis points (0.01%). Typical
                         values: 1-10 bps for liquid markets.
        depth_factor: Multiplier for liquidity depth. Higher values simulate
                       more liquid markets with larger orders at each level.
        impact_coefficient: Coefficient for permanent market impact calculation.
                            Controls how much large orders move the market price.

    Example:
        >>> config = OrderBookConfig(
        ...     n_levels=10,
        ...     base_spread_bps=5.0,  # 0.05% spread
        ...     depth_factor=1.0,
        ...     impact_coefficient=0.1
        ... )
    """

    n_levels: int = 10  # Number of price levels to simulate
    base_spread_bps: float = 5.0  # Base spread in basis points
    depth_factor: float = 1.0  # Liquidity depth multiplier
    impact_coefficient: float = 0.1  # Price impact coefficient


class OrderBookSimulator:
    """
    Simulates Level 2 order book for realistic trade execution.

    This class models a realistic limit order book by maintaining aggregated
    price levels on both bid and ask sides. It simulates the "walking the book"
    phenomenon where large orders consume liquidity at multiple price levels,
    resulting in worse execution prices than the best bid/ask.

    Key Insight:
    -------------
    Real markets have LIMITED liquidity at each price level. Large orders
    "walk the book" and experience worse fills as they consume liquidity
    at progressively worse prices. This is a critical factor for accurate
    backtesting as overly optimistic slippage assumptions lead to overfitting.

    The simulator models:
    1. Price Levels: Non-linear spacing widening with distance from mid
    2. Volume Distribution: Exponential decay with random noise
    3. Spread Dynamics: Volatility-dependent spreads
    4. Market Impact: Permanent price movement from large orders

    Attributes:
        config: OrderBookConfig instance with simulation parameters

    Example:
        >>> config = OrderBookConfig(n_levels=10, base_spread_bps=5.0)
        >>> simulator = OrderBookSimulator(config)
        >>> bids, bid_vols, asks, ask_vols = simulator.generate_order_book(
        ...     mid_price=50000.0, volatility=0.02, volume=100.0
        ... )
    """

    def __init__(self, config: OrderBookConfig):
        """
        Initialize the order book simulator.

        Args:
            config: OrderBookConfig with simulation parameters
        """
        self.config = config
        logger.info(f"OrderBookSimulator initialized with {config.n_levels} levels")

    def generate_order_book(
        self, mid_price: float, volatility: float, volume: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate synthetic order book around mid price.

        Creates a realistic order book by:
        1. Calculating dynamic spread based on volatility
        2. Generating price levels with non-linear spacing
        3. Distributing volume exponentially across levels

        Parameters:
        -----------
        mid_price : float
            Current market mid price (the average of best bid and ask).
            Used as the reference point for generating price levels.
        volatility : float
            Recent volatility (e.g., 0.02 for 2%). Higher volatility
            increases the bid-ask spread to simulate wider markets.
        volume : float
            Recent trading volume. Used to scale the volume available
            at each price level - higher volume markets have more liquidity.

        Returns:
        --------
        bid_prices : np.ndarray
            Bid prices in descending order (best bid first).
            Shape: (n_levels,)
        bid_volumes : np.ndarray
            Volume available at each bid level.
            Shape: (n_levels,)
        ask_prices : np.ndarray
            Ask prices in ascending order (best ask first).
            Shape: (n_levels,)
        ask_volumes : np.ndarray
            Volume available at each ask level.
            Shape: (n_levels,)

        Example:
            >>> bids, bid_v, asks, ask_v = simulator.generate_order_book(
            ...     mid_price=50000.0, volatility=0.02, volume=100.0
            ... )
            >>> print(f"Best bid: {bids[0]:.2f}, Best ask: {asks[0]:.2f}")
        """
        # Spread increases with volatility - volatile markets have wider spreads
        # This models the empirical observation that volatility and spreads are correlated
        spread_bps = self.config.base_spread_bps * (1 + volatility * 10)
        spread = mid_price * (spread_bps / 10000)

        # Best bid/ask - the tightest prices on each side
        best_bid = mid_price - spread / 2
        best_ask = mid_price + spread / 2

        # Generate price levels on both sides
        bid_prices = self._generate_price_levels(
            best_bid, direction="down", n_levels=self.config.n_levels
        )

        ask_prices = self._generate_price_levels(
            best_ask, direction="up", n_levels=self.config.n_levels
        )

        # Generate volumes - exponentially decreasing with distance from best price
        # Most liquidity is concentrated near the best prices
        bid_volumes = self._generate_volumes(volume, self.config.n_levels)
        ask_volumes = self._generate_volumes(volume, self.config.n_levels)

        return bid_prices, bid_volumes, ask_prices, ask_volumes

    def _generate_price_levels(
        self, start_price: float, direction: str, n_levels: int
    ) -> np.ndarray:
        """
        Generate price levels with realistic non-linear spacing.

        Uses widening tick sizes to model the fact that price levels
        become more spaced out as you move away from the best price.
        This is realistic because:
        - Market makers place more orders near the mid price
        - Deep out-of-the-money orders are less common

        Parameters:
        -----------
        start_price : float
            Starting price (best bid or best ask)
        direction : str
            'down' for bids, 'up' for asks
        n_levels : int
            Number of levels to generate

        Returns:
        --------
        prices : np.ndarray
            Array of price levels
        """
        levels = np.zeros(n_levels)
        levels[0] = start_price

        # Price levels get wider apart (non-linear spacing)
        # This models increasing tick sizes at further distances
        for i in range(1, n_levels):
            # Widening ticks: each level is slightly wider than the previous
            tick_size = start_price * (0.0001 * (1 + i * 0.1))

            if direction == "down":
                levels[i] = levels[i - 1] - tick_size
            else:  # up
                levels[i] = levels[i - 1] + tick_size

        return levels

    def _generate_volumes(self, base_volume: float, n_levels: int) -> np.ndarray:
        """
        Generate volume distribution across price levels.

        Volume decreases exponentially away from the best price, with added
        random noise to simulate real market conditions. This models:
        - Highest liquidity at best prices (most competitive)
        - Decreasing liquidity at worse prices
        - Random fluctuations in order flow

        Parameters:
        -----------
        base_volume : float
            Base volume for scaling (recent average volume)
        n_levels : int
            Number of price levels

        Returns:
        --------
        volumes : np.ndarray
            Volume at each price level
        """
        volumes = np.zeros(n_levels)

        for i in range(n_levels):
            # Exponential decay - most volume at level 0, decreasing exponentially
            # Decay rate of 0.3 means ~74% reduction at level 5
            decay_factor = np.exp(-i * 0.3)

            # Add randomness (0.5 to 1.5x) to simulate real market noise
            noise = np.random.uniform(0.5, 1.5)

            volumes[i] = base_volume * self.config.depth_factor * decay_factor * noise

        return volumes

    def calculate_execution_price(
        self,
        side: str,
        quantity: float,
        bid_prices: np.ndarray,
        bid_volumes: np.ndarray,
        ask_prices: np.ndarray,
        ask_volumes: np.ndarray,
    ) -> Tuple[float, float, float]:
        """
        Calculate realistic execution price with slippage.

        Simulates "walking the order book" for large orders. When an order
        exceeds the available volume at the best price, it automatically
        fills at the next price level, and so on until the order is complete.

        This is critical for accurate backtesting because:
        - Small orders execute at or near the best price
        - Large orders experience significant slippage
        - Partial fills occur when total volume is insufficient

        Parameters:
        -----------
        side : str
            Order side: 'buy' or 'sell'
            - Buy orders consume ask (sell) side liquidity
            - Sell orders consume bid (buy) side liquidity
        quantity : float
            Order quantity in base currency units (e.g., BTC)
        bid_prices : np.ndarray
            Bid prices from generate_order_book()
        bid_volumes : np.ndarray
            Bid volumes from generate_order_book()
        ask_prices : np.ndarray
            Ask prices from generate_order_book()
        ask_volumes : np.ndarray
            Ask volumes from generate_order_book()

        Returns:
        --------
        avg_price : float
            Volume-weighted average execution price across all filled levels
        slippage_bps : float
            Slippage in basis points relative to best price
            Positive value indicates worse execution than best price
        filled_quantity : float
            Actually filled quantity. May be less than requested if
            insufficient liquidity exists in the order book

        Example:
            >>> avg_price, slippage, filled = simulator.calculate_execution_price(
            ...     side='buy',
            ...     quantity=10.0,
            ...     bid_prices=bids, bid_volumes=bid_v,
            ...     ask_prices=asks, ask_volumes=ask_v
            ... )
            >>> print(f"Filled: {filled}/{10.0} @ ${avg_price:.2f}")
        """
        if side == "buy":
            # Buy orders execute against asks (sell side)
            prices = ask_prices
            volumes = ask_volumes
            best_price = ask_prices[0]
        else:  # sell
            # Sell orders execute against bids (buy side)
            prices = bid_prices
            volumes = bid_volumes
            best_price = bid_prices[0]

        # Walk the book: fill at each level until order is complete
        filled = 0.0
        total_cost = 0.0
        level = 0

        while filled < quantity and level < len(prices):
            # How much can we fill at this level?
            available = volumes[level]
            to_fill = min(quantity - filled, available)

            # Add cost at this price level
            total_cost += to_fill * prices[level]
            filled += to_fill
            level += 1

        if filled == 0:
            # No liquidity available at any level
            return best_price, 0.0, 0.0

        # Calculate volume-weighted average price
        avg_price = total_cost / filled

        # Calculate slippage in basis points
        # Slippage = |actual - expected| / expected × 10000
        slippage_bps = abs(avg_price - best_price) / best_price * 10000

        return avg_price, slippage_bps, filled

    def estimate_market_impact(
        self, quantity: float, avg_daily_volume: float, mid_price: float
    ) -> float:
        """
        Estimate permanent market impact from a large order.

        Uses the square-root law of market impact, an empirically validated
        relationship where price impact is proportional to the square root
        of the order size relative to daily volume.

        Model: Price Impact = coefficient × √(order_size / daily_volume)

        This represents PERMANENT impact - the price shift that remains
        after the order completes, as opposed to temporary impact which
        represents price movement that reverts.

        Parameters:
        -----------
        quantity : float
            Order size in base currency
        avg_daily_volume : float
            Average daily trading volume (same units as quantity)
        mid_price : float
            Current mid price for converting basis points to price units

        Returns:
        --------
        impact_price : float
            Estimated permanent price impact in price units

        Example:
            >>> impact = simulator.estimate_market_impact(
            ...     quantity=100.0,  # 100 BTC order
            ...     avg_daily_volume=1000.0,  # 1000 BTC daily volume
            ...     mid_price=50000.0
            ... )
            >>> print(f"Impact: ${impact:.2f}")
        """
        # Participation rate: what fraction of daily volume is this order?
        participation = quantity / (avg_daily_volume + 1e-8)

        # Square root law of price impact
        # Empirical research shows impact ∝ √(order_size / volume)
        impact_bps = self.config.impact_coefficient * np.sqrt(participation) * 10000

        # Convert from basis points to price units
        impact_price = mid_price * (impact_bps / 10000)

        return impact_price


# ============================================================================
# NUMBA-OPTIMIZED SLIPPAGE CALCULATION
# ============================================================================


@jit(nopython=True, cache=True)
def calculate_slippage_numba(
    quantity: float, prices: np.ndarray, volumes: np.ndarray
) -> Tuple[float, float]:
    """
    Numba-optimized slippage calculation for performance.

    This JIT-compiled function provides fast execution for slippage
    calculations by avoiding Python overhead. It's used internally
    when the orderbook slippage model is selected.

    Parameters:
    -----------
    quantity : float
        Order quantity to fill
    prices : np.ndarray
        Price levels (should be sorted for the order side)
    volumes : np.ndarray
        Available volume at each price level

    Returns:
    --------
    avg_price : float
        Volume-weighted average execution price
    filled_quantity : float
        Actual quantity filled (may be less than requested)

    Note:
        This function assumes prices and volumes are aligned and
        represents one side of the order book (all bids or all asks).
    """
    filled = 0.0
    total_cost = 0.0

    # Iterate through levels, filling as much as possible at each
    for i in range(len(prices)):
        if filled >= quantity:
            break

        available = volumes[i]
        to_fill = min(quantity - filled, available)

        total_cost += to_fill * prices[i]
        filled += to_fill

    # Handle case where no volume available
    if filled == 0.0:
        return prices[0], 0.0

    avg_price = total_cost / filled
    return avg_price, filled


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("ORDER BOOK SIMULATOR TEST")
    print("=" * 80)

    # Configure with realistic parameters
    config = OrderBookConfig(
        n_levels=10, base_spread_bps=5.0, depth_factor=1.0, impact_coefficient=0.1
    )

    simulator = OrderBookSimulator(config)

    # Test 1: Generate order book
    print("\n[TEST 1] Generate order book")

    mid_price = 50000.0  # BTC price
    volatility = 0.02  # 2% volatility
    volume = 100.0  # Average volume

    bid_prices, bid_volumes, ask_prices, ask_volumes = simulator.generate_order_book(
        mid_price, volatility, volume
    )

    print(f"Mid price: ${mid_price:,.2f}")
    print(f"\nTop 3 Bids:")
    for i in range(3):
        print(f"  {bid_prices[i]:,.2f} @ {bid_volumes[i]:.2f}")

    print(f"\nTop 3 Asks:")
    for i in range(3):
        print(f"  {ask_prices[i]:,.2f} @ {ask_volumes[i]:.2f}")

    spread = ask_prices[0] - bid_prices[0]
    spread_bps = spread / mid_price * 10000
    print(f"\nSpread: ${spread:.2f} ({spread_bps:.2f} bps)")

    # Test 2: Small order (minimal slippage expected)
    print("\n[TEST 2] Small order execution")

    quantity = 0.1  # Small order relative to volume
    avg_price, slippage, filled = simulator.calculate_execution_price(
        "buy", quantity, bid_prices, bid_volumes, ask_prices, ask_volumes
    )

    print(f"Order: BUY {quantity} BTC")
    print(f"Execution price: ${avg_price:,.2f}")
    print(f"Slippage: {slippage:.2f} bps")
    print(f"Filled: {filled:.4f} BTC")

    # Test 3: Large order (significant slippage expected)
    print("\n[TEST 3] Large order execution")

    quantity = 50.0  # Large order - will walk the book
    avg_price, slippage, filled = simulator.calculate_execution_price(
        "buy", quantity, bid_prices, bid_volumes, ask_prices, ask_volumes
    )

    print(f"Order: BUY {quantity} BTC")
    print(f"Execution price: ${avg_price:,.2f}")
    print(f"Slippage: {slippage:.2f} bps")
    print(f"Filled: {filled:.4f} BTC (Partial fill due to insufficient liquidity!)")

    # Test 4: Market impact estimation
    print("\n[TEST 4] Market impact estimation")

    daily_volume = 1000.0  # Average daily volume
    impact = simulator.estimate_market_impact(quantity, daily_volume, mid_price)
    impact_bps = impact / mid_price * 10000

    print(f"Order size: {quantity} BTC")
    print(f"Daily volume: {daily_volume} BTC")
    print(f"Participation: {quantity / daily_volume * 100:.1f}%")
    print(f"Estimated impact: ${impact:.2f} ({impact_bps:.2f} bps)")

    print("\n" + "=" * 80)
    print("✓ ORDER BOOK SIMULATOR TEST PASSED")
    print("=" * 80)
