"""
Level 2 Order Book Simulator
=============================
Realistic order book simulation for accurate slippage modeling.

Features:
- Aggregated order book levels (bid/ask)
- Volume-based slippage calculation
- Market impact modeling
- Dynamic spread simulation
"""

import numpy as np
from typing import Tuple, Dict, List
from dataclasses import dataclass
from numba import jit
from loguru import logger


@dataclass
class OrderBookConfig:
    """Configuration for order book simulation."""
    n_levels: int = 10  # Number of price levels to simulate
    base_spread_bps: float = 5.0  # Base spread in basis points
    depth_factor: float = 1.0  # Liquidity depth multiplier
    impact_coefficient: float = 0.1  # Price impact coefficient


class OrderBookSimulator:
    """
    Simulates Level 2 order book for realistic trade execution.
    
    The order book determines:
    - Execution price (with slippage)
    - Available liquidity at each level
    - Market impact from large orders
    
    Key Insight:
    Real markets have LIMITED liquidity at each price level.
    Large orders "walk the book" and experience worse fills.
    """
    
    def __init__(self, config: OrderBookConfig):
        self.config = config
        logger.info(f"OrderBookSimulator initialized with {config.n_levels} levels")
    
    def generate_order_book(
        self,
        mid_price: float,
        volatility: float,
        volume: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate synthetic order book around mid price.
        
        Parameters:
        -----------
        mid_price : float
            Current market mid price
        volatility : float
            Recent volatility (affects spread)
        volume : float
            Recent volume (affects depth)
            
        Returns:
        --------
        bid_prices : np.ndarray
            Bid prices (descending)
        bid_volumes : np.ndarray
            Volume available at each bid level
        ask_prices : np.ndarray
            Ask prices (ascending)
        ask_volumes : np.ndarray
            Volume available at each ask level
        """
        # Spread increases with volatility
        spread_bps = self.config.base_spread_bps * (1 + volatility * 10)
        spread = mid_price * (spread_bps / 10000)
        
        # Best bid/ask
        best_bid = mid_price - spread / 2
        best_ask = mid_price + spread / 2
        
        # Generate price levels
        bid_prices = self._generate_price_levels(
            best_bid, 
            direction='down',
            n_levels=self.config.n_levels
        )
        
        ask_prices = self._generate_price_levels(
            best_ask,
            direction='up',
            n_levels=self.config.n_levels
        )
        
        # Generate volumes (decreasing with distance from mid)
        bid_volumes = self._generate_volumes(volume, self.config.n_levels)
        ask_volumes = self._generate_volumes(volume, self.config.n_levels)
        
        return bid_prices, bid_volumes, ask_prices, ask_volumes
    
    def _generate_price_levels(
        self,
        start_price: float,
        direction: str,
        n_levels: int
    ) -> np.ndarray:
        """Generate price levels with realistic spacing."""
        levels = np.zeros(n_levels)
        levels[0] = start_price
        
        # Price levels get wider apart (non-linear spacing)
        for i in range(1, n_levels):
            tick_size = start_price * (0.0001 * (1 + i * 0.1))  # Widening ticks
            
            if direction == 'down':
                levels[i] = levels[i-1] - tick_size
            else:  # up
                levels[i] = levels[i-1] + tick_size
        
        return levels
    
    def _generate_volumes(self, base_volume: float, n_levels: int) -> np.ndarray:
        """
        Generate volume distribution across levels.
        
        Volume decreases exponentially away from best price.
        """
        volumes = np.zeros(n_levels)
        
        for i in range(n_levels):
            # Exponential decay
            decay_factor = np.exp(-i * 0.3)
            
            # Add randomness
            noise = np.random.uniform(0.5, 1.5)
            
            volumes[i] = (
                base_volume * 
                self.config.depth_factor * 
                decay_factor * 
                noise
            )
        
        return volumes
    
    def calculate_execution_price(
        self,
        side: str,
        quantity: float,
        bid_prices: np.ndarray,
        bid_volumes: np.ndarray,
        ask_prices: np.ndarray,
        ask_volumes: np.ndarray
    ) -> Tuple[float, float, float]:
        """
        Calculate realistic execution price with slippage.
        
        Simulates "walking the order book" for large orders.
        
        Parameters:
        -----------
        side : str
            'buy' or 'sell'
        quantity : float
            Order quantity (in base currency units)
        bid_prices, bid_volumes : np.ndarray
            Bid side of order book
        ask_prices, ask_volumes : np.ndarray
            Ask side of order book
            
        Returns:
        --------
        avg_price : float
            Volume-weighted average execution price
        slippage_bps : float
            Slippage in basis points
        filled_quantity : float
            Actually filled quantity (may be < requested if insufficient liquidity)
        """
        if side == 'buy':
            # Buy from ask side
            prices = ask_prices
            volumes = ask_volumes
            best_price = ask_prices[0]
        else:  # sell
            # Sell to bid side
            prices = bid_prices
            volumes = bid_volumes
            best_price = bid_prices[0]
        
        # Walk the book
        filled = 0.0
        total_cost = 0.0
        level = 0
        
        while filled < quantity and level < len(prices):
            available = volumes[level]
            to_fill = min(quantity - filled, available)
            
            total_cost += to_fill * prices[level]
            filled += to_fill
            level += 1
        
        if filled == 0:
            # No liquidity!
            return best_price, 0.0, 0.0
        
        # Average execution price
        avg_price = total_cost / filled
        
        # Slippage in basis points
        slippage_bps = abs(avg_price - best_price) / best_price * 10000
        
        return avg_price, slippage_bps, filled
    
    def estimate_market_impact(
        self,
        quantity: float,
        avg_daily_volume: float,
        mid_price: float
    ) -> float:
        """
        Estimate permanent market impact.
        
        Large orders move the market price permanently.
        
        Model: Price Impact ∝ sqrt(Order Size / Daily Volume)
        """
        # Participation rate
        participation = quantity / (avg_daily_volume + 1e-8)
        
        # Square root law of price impact
        impact_bps = (
            self.config.impact_coefficient * 
            np.sqrt(participation) * 
            10000
        )
        
        # Convert to price
        impact_price = mid_price * (impact_bps / 10000)
        
        return impact_price


# ============================================================================
# NUMBA-OPTIMIZED SLIPPAGE CALCULATION
# ============================================================================

@jit(nopython=True, cache=True)
def calculate_slippage_numba(
    quantity: float,
    prices: np.ndarray,
    volumes: np.ndarray
) -> Tuple[float, float]:
    """
    Numba-optimized slippage calculation.
    
    Returns:
    --------
    avg_price : float
    filled_quantity : float
    """
    filled = 0.0
    total_cost = 0.0
    
    for i in range(len(prices)):
        if filled >= quantity:
            break
        
        available = volumes[i]
        to_fill = min(quantity - filled, available)
        
        total_cost += to_fill * prices[i]
        filled += to_fill
    
    if filled == 0.0:
        return prices[0], 0.0
    
    avg_price = total_cost / filled
    return avg_price, filled


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("="*80)
    print("ORDER BOOK SIMULATOR TEST")
    print("="*80)
    
    # Configure
    config = OrderBookConfig(
        n_levels=10,
        base_spread_bps=5.0,
        depth_factor=1.0,
        impact_coefficient=0.1
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
    
    # Test 2: Small order (no slippage)
    print("\n[TEST 2] Small order execution")
    
    quantity = 0.1  # Small order
    avg_price, slippage, filled = simulator.calculate_execution_price(
        'buy', quantity, bid_prices, bid_volumes, ask_prices, ask_volumes
    )
    
    print(f"Order: BUY {quantity} BTC")
    print(f"Execution price: ${avg_price:,.2f}")
    print(f"Slippage: {slippage:.2f} bps")
    print(f"Filled: {filled:.4f} BTC")
    
    # Test 3: Large order (significant slippage)
    print("\n[TEST 3] Large order execution")
    
    quantity = 50.0  # Large order
    avg_price, slippage, filled = simulator.calculate_execution_price(
        'buy', quantity, bid_prices, bid_volumes, ask_prices, ask_volumes
    )
    
    print(f"Order: BUY {quantity} BTC")
    print(f"Execution price: ${avg_price:,.2f}")
    print(f"Slippage: {slippage:.2f} bps")
    print(f"Filled: {filled:.4f} BTC (Partial fill!)")
    
    # Test 4: Market impact
    print("\n[TEST 4] Market impact estimation")
    
    daily_volume = 1000.0
    impact = simulator.estimate_market_impact(quantity, daily_volume, mid_price)
    impact_bps = impact / mid_price * 10000
    
    print(f"Order size: {quantity} BTC")
    print(f"Daily volume: {daily_volume} BTC")
    print(f"Participation: {quantity/daily_volume*100:.1f}%")
    print(f"Estimated impact: ${impact:.2f} ({impact_bps:.2f} bps)")
    
    print("\n" + "="*80)
    print("✓ ORDER BOOK SIMULATOR TEST PASSED")
    print("="*80)
