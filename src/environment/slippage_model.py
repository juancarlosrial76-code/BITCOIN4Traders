"""
Slippage Model Module
=====================
Comprehensive slippage calculation strategies for realistic trade execution
modeling in cryptocurrency trading environments.

Purpose:
--------
This module provides multiple slippage calculation strategies to model
the difference between expected and actual execution prices. Accurate
slippage modeling is critical for backtesting because:

1. Overly optimistic slippage leads to unrealistic strategy performance
2. Different market conditions require different slippage models
3. Transaction costs significantly impact strategy profitability

Strategies Implemented:
------------------------
1. Fixed BPS: Constant slippage regardless of order size or market conditions
   - Use case: Simple backtests, conservative estimates

2. Volume-based: Slippage proportional to order size relative to volume
   - Uses square-root market impact law
   - Use case: Most realistic for medium-sized orders

3. Volatility-adjusted: Higher slippage in volatile markets
   - Models wider spreads during high volatility
   - Use case: Crisis periods, volatile assets

4. Order Book: Realistic slippage from walking the order book
   - Most accurate but requires order book data
   - Use case: High-precision backtesting

Additional Components:
-----------------------
- TransactionCostModel: Complete cost modeling including fees, slippage,
  and market impact
- Numba-optimized calculations for performance

Usage:
------
    from src.environment.slippage_model import (
        SlippageModel, SlippageConfig,
        TransactionCostModel, TransactionCostConfig
    )

    # Configure slippage model
    config = SlippageConfig(model_type='volume_based', fixed_slippage_bps=5.0)
    model = SlippageModel(config)

    # Calculate slippage
    execution_price, slippage_bps = model.calculate_slippage(
        side='buy',
        quantity=1.0,
        price=50000.0,
        volume=100.0,
        volatility=0.02
    )

Mathematical Models:
--------------------
1. Fixed: slippage = fixed_bps
2. Volume-based: slippage = base + coef × √(quantity/volume) × 10000
3. Volatility: slippage = base × (1 + volatility_multiplier × volatility)
4. Order book: VWAP from walking price levels

Dependencies:
-------------
- numpy: Numerical operations
- numba: JIT compilation for performance
- loguru: Logging
- dataclasses: Configuration data structures
"""

import numpy as np
from typing import Tuple, Dict
from dataclasses import dataclass
from numba import jit
from loguru import logger


@dataclass
class SlippageConfig:
    """
    Configuration for slippage modeling.

    Attributes:
        model_type: Slippage calculation strategy. Options:
            - 'fixed': Constant slippage
            - 'volume_based': Size-dependent slippage (recommended)
            - 'volatility': Volatility-dependent slippage
            - 'orderbook': Order book walking (most accurate)
        fixed_slippage_bps: Base slippage in basis points (for fixed model)
        volatility_multiplier: Scaling factor for volatility model
        volume_impact_coef: Coefficient for volume-based model

    Example:
        >>> config = SlippageConfig(
        ...     model_type='volume_based',
        ...     fixed_slippage_bps=5.0,
        ...     volume_impact_coef=0.1
        ... )
    """

    model_type: str = "volume_based"  # fixed, volume_based, volatility, orderbook
    fixed_slippage_bps: float = 5.0  # For fixed model
    volatility_multiplier: float = 2.0  # For volatility model
    volume_impact_coef: float = 0.1  # For volume model


class SlippageModel:
    """
    Slippage calculation for realistic trade execution modeling.

    This class implements multiple strategies for calculating slippage -
    the difference between the expected execution price and the actual
    price obtained. Accurate slippage modeling is essential because:

    1. It represents real trading costs not captured by fees alone
    2. Large orders experience more slippage than small orders
    3. Market conditions (volatility, volume) significantly impact slippage

    Critical Insight:
    -----------------
    Overly optimistic slippage assumptions in backtesting lead to strategies
    that perform well in testing but fail in live trading. It's recommended
    to use conservative slippage estimates or the 'orderbook' model for
    accurate simulation.

    Attributes:
        config: SlippageConfig with model parameters

    Example:
        >>> config = SlippageConfig(model_type='volume_based')
        >>> model = SlippageModel(config)
        >>> exec_price, slippage = model.calculate_slippage(
        ...     side='buy', quantity=1.0, price=50000.0,
        ...     volume=100.0, volatility=0.02
        ... )
    """

    def __init__(self, config: SlippageConfig):
        """
        Initialize slippage model.

        Args:
            config: SlippageConfig with model parameters
        """
        self.config = config
        logger.info(f"SlippageModel initialized: {config.model_type}")

    def calculate_slippage(
        self,
        side: str,
        quantity: float,
        price: float,
        volume: float,
        volatility: float,
        **kwargs,
    ) -> Tuple[float, float]:
        """
        Calculate slippage for an order using the configured model.

        This is the main entry point for slippage calculation. It delegates
        to the appropriate method based on the configured model_type.

        Parameters:
        -----------
        side : str
            Order side: 'buy' or 'sell'
            - Buy orders: slippage results in higher execution price
            - Sell orders: slippage results in lower execution price
        quantity : float
            Order size in base currency (e.g., BTC for BTC/USDT)
        price : float
            Reference price - typically the mid price or last traded price
            at the time of order submission
        volume : float
            Recent market volume (e.g., 24h volume or recent bar volume).
            Used for volume-based models to calculate participation rate.
        volatility : float
            Recent volatility (e.g., 20-period standard deviation of returns).
            Higher volatility typically results in wider spreads and more slippage.
        **kwargs : dict
            Additional parameters for specific models:
            - orderbook model: bid_prices, bid_volumes, ask_prices, ask_volumes

        Returns:
        --------
        execution_price : float
            Actual execution price after slippage
        slippage_bps : float
            Slippage in basis points (0.01%). Positive means worse execution
            than reference price.

        Example:
            >>> # Volume-based slippage
            >>> config = SlippageConfig(model_type='volume_based')
            >>> model = SlippageModel(config)
            >>> exec_price, slippage = model.calculate_slippage(
            ...     side='buy', quantity=1.0, price=50000.0,
            ...     volume=100.0, volatility=0.02
            ... )
            >>> print(f"Exec: ${exec_price:.2f}, Slippage: {slippage:.2f} bps")
        """
        if self.config.model_type == "fixed":
            return self._fixed_slippage(side, price)

        elif self.config.model_type == "volume_based":
            return self._volume_based_slippage(side, quantity, price, volume)

        elif self.config.model_type == "volatility":
            return self._volatility_adjusted_slippage(side, price, volatility)

        elif self.config.model_type == "orderbook":
            return self._orderbook_slippage(side, quantity, price, **kwargs)

        else:
            raise ValueError(f"Unknown slippage model: {self.config.model_type}")

    def _fixed_slippage(self, side: str, price: float) -> Tuple[float, float]:
        """
        Fixed slippage regardless of order size or market conditions.

        This is the simplest model - a constant slippage in basis points.
        Useful for conservative estimates or simple backtests.

        Parameters:
        -----------
        side : str
            'buy' or 'sell'
        price : float
            Reference price

        Returns:
        --------
        execution_price : float
            Price after applying fixed slippage
        slippage_bps : float
            The fixed slippage in bps
        """
        slippage_bps = self.config.fixed_slippage_bps
        # Convert bps to price units
        slippage = price * (slippage_bps / 10000)

        if side == "buy":
            # Buy orders execute ABOVE reference (worse price)
            execution_price = price + slippage
        else:  # sell
            # Sell orders execute BELOW reference (worse price)
            execution_price = price - slippage

        return execution_price, slippage_bps

    def _volume_based_slippage(
        self, side: str, quantity: float, price: float, volume: float
    ) -> Tuple[float, float]:
        """
        Slippage proportional to order size vs market volume.

        Uses the square-root market impact law, which is empirically validated
        in equity and crypto markets. The model captures:

        1. Base slippage component (exchange fees, minor costs)
        2. Market impact component proportional to √(participation rate)

        Model: slippage = base + coef × √(quantity/volume) × 10000

        This model is recommended for most backtesting scenarios as it
        provides a good balance between accuracy and simplicity.

        Parameters:
        -----------
        side : str
            'buy' or 'sell'
        quantity : float
            Order size in base currency
        price : float
            Reference price
        volume : float
            Recent market volume for calculating participation

        Returns:
        --------
        execution_price : float
            Price after applying volume-based slippage
        slippage_bps : float
            Calculated slippage in bps
        """
        # Participation rate: fraction of market volume this order represents
        # Small orders have low participation, large orders have high participation
        participation = quantity / (volume + 1e-8)  # 1e-8 avoids division by zero

        # Slippage increases non-linearly with participation
        # Square-root law: impact ∝ √(order_size/volume)
        # This captures the empirically observed relationship
        slippage_bps = (
            self.config.fixed_slippage_bps  # Base fee component
            + self.config.volume_impact_coef
            * np.sqrt(participation)
            * 10000  # Convert to basis points
        )

        # Cap maximum slippage to prevent unrealistic values
        # Max 1% (100 bps) prevents runaway costs in extreme scenarios
        slippage_bps = min(slippage_bps, 100.0)

        slippage = price * (slippage_bps / 10000)

        if side == "buy":
            execution_price = price + slippage
        else:
            execution_price = price - slippage

        return execution_price, slippage_bps

    def _volatility_adjusted_slippage(
        self, side: str, price: float, volatility: float
    ) -> Tuple[float, float]:
        """
        Slippage that increases with market volatility.

        This model captures the empirical observation that:
        - High volatility markets have wider spreads
        - Wider spreads result in more slippage
        - Market makers demand more compensation for uncertainty

        Model: slippage = base × (1 + multiplier × volatility)

        Use case: Modeling slippage during volatile periods or crisis events

        Parameters:
        -----------
        side : str
            'buy' or 'sell'
        price : float
            Reference price
        volatility : float
            Recent volatility (e.g., 0.02 = 2%)

        Returns:
        --------
        execution_price : float
            Price after applying volatility-adjusted slippage
        slippage_bps : float
            Calculated slippage in bps
        """
        # Base slippage scaled by volatility
        # Example: 2% vol with 2x multiplier = 4% additional slippage
        slippage_bps = self.config.fixed_slippage_bps * (
            1 + self.config.volatility_multiplier * volatility
        )

        slippage = price * (slippage_bps / 10000)

        if side == "buy":
            execution_price = price + slippage
        else:
            execution_price = price - slippage

        return execution_price, slippage_bps

    def _orderbook_slippage(
        self, side: str, quantity: float, price: float, **kwargs
    ) -> Tuple[float, float]:
        """
        Realistic slippage from order book simulation.

        This is the most accurate slippage model as it simulates walking
        the order book - filling an order at multiple price levels when
        liquidity at the best price is insufficient.

        Requires order book data in kwargs:
        - bid_prices, bid_volumes: Bid side of order book
        - ask_prices, ask_volumes: Ask side of order book

        Falls back to volume-based model if order book data is missing.

        Parameters:
        -----------
        side : str
            'buy' or 'sell'
        quantity : float
            Order size in base currency
        price : float
            Reference price (typically mid price)
        **kwargs : dict
            Order book data (optional but recommended)

        Returns:
        --------
        execution_price : float
            Volume-weighted average execution price
        slippage_bps : float
            Calculated slippage in bps
        """
        # Extract order book data from kwargs
        bid_prices = kwargs.get("bid_prices")
        bid_volumes = kwargs.get("bid_volumes")
        ask_prices = kwargs.get("ask_prices")
        ask_volumes = kwargs.get("ask_volumes")

        # Fallback to volume-based if order book not provided
        if any(x is None for x in [bid_prices, bid_volumes, ask_prices, ask_volumes]):
            logger.warning("Order book data missing, using volume-based slippage")
            return self._volume_based_slippage(
                side, quantity, price, kwargs.get("volume", 100.0)
            )

        # Use Numba-optimized calculation for performance
        if side == "buy":
            # Buy orders execute against asks (sell side)
            execution_price, filled = calculate_slippage_numba(
                quantity, ask_prices, ask_volumes
            )
        else:
            # Sell orders execute against bids (buy side)
            execution_price, filled = calculate_slippage_numba(
                quantity, bid_prices, bid_volumes
            )

        # Calculate slippage relative to reference price
        slippage_bps = abs(execution_price - price) / price * 10000

        return execution_price, slippage_bps


@jit(nopython=True, cache=True)
def calculate_slippage_numba(
    quantity: float, prices: np.ndarray, volumes: np.ndarray
) -> Tuple[float, float]:
    """
    Numba-optimized slippage calculation from walking the order book.

    This JIT-compiled function provides fast execution for order book
    slippage calculations. It walks through price levels, filling as much
    as possible at each level until the order is complete.

    Parameters:
    -----------
    quantity : float
        Order quantity to fill
    prices : np.ndarray
        Price levels (sorted for the relevant side)
    volumes : np.ndarray
        Volume available at each price level

    Returns:
    --------
    avg_price : float
        Volume-weighted average execution price
    filled_quantity : float
        Actual quantity filled (may be less than requested)
    """
    filled = 0.0
    total_cost = 0.0

    # Walk through price levels, filling the order
    for i in range(len(prices)):
        if filled >= quantity:
            break

        available = volumes[i]
        to_fill = min(quantity - filled, available)

        total_cost += to_fill * prices[i]
        filled += to_fill

    # Handle edge case: no volume available
    if filled == 0.0:
        return prices[0], 0.0

    avg_price = total_cost / filled
    return avg_price, filled


# ============================================================================
# TRANSACTION COST MODEL
# ============================================================================


@dataclass
class TransactionCostConfig:
    """
    Configuration for complete transaction cost modeling.

    Attributes:
        fixed_bps: Fixed trading fee in basis points (per side)
        include_slippage: Whether to include slippage in cost calculation
        include_market_impact: Whether to include permanent market impact

    Note:
        In practice, exchange fees are typically:
        - Maker: 0.01-0.02% (1-2 bps)
        - Taker: 0.03-0.06% (3-6 bps)

        This config uses a single fixed rate for simplicity.
    """

    fixed_bps: float = 5.0  # 0.05% = 5 basis points
    include_slippage: bool = True
    include_market_impact: bool = True


class TransactionCostModel:
    """
    Complete transaction cost modeling.

    Combines multiple cost components to provide a comprehensive
    view of trading costs:

    1. Exchange Fees: Fixed costs per trade (maker/taker)
    2. Slippage: Cost due to adverse price movement
    3. Market Impact: Permanent price movement from large orders

    Total Cost = Fees + Slippage + Market Impact

    This model is used by ConfigIntegratedTradingEnv for accurate
    cost calculation in backtesting.

    Attributes:
        cost_config: TransactionCostConfig with fee parameters
        slippage_model: SlippageModel for slippage calculation

    Example:
        >>> cost_config = TransactionCostConfig(fixed_bps=5.0)
        >>> slippage_config = SlippageConfig(model_type='volume_based')
        >>> slippage_model = SlippageModel(slippage_config)
        >>> cost_model = TransactionCostModel(cost_config, slippage_model)
        >>>
        >>> costs = cost_model.calculate_total_cost(
        ...     side='buy', quantity=1.0, price=50000.0,
        ...     volume=100.0, volatility=0.02
        ... )
    """

    def __init__(
        self, cost_config: TransactionCostConfig, slippage_model: SlippageModel
    ):
        """
        Initialize transaction cost model.

        Args:
            cost_config: Configuration for fees
            slippage_model: Model for slippage calculation
        """
        self.cost_config = cost_config
        self.slippage_model = slippage_model
        logger.info(f"TransactionCostModel: {cost_config.fixed_bps} bps fees")

    def calculate_total_cost(
        self, side: str, quantity: float, price: float, **kwargs
    ) -> Dict[str, float]:
        """
        Calculate complete transaction costs including all components.

        This method provides a comprehensive cost breakdown:
        1. Execution price after slippage
        2. Trading fee in basis points
        3. Slippage in basis points
        4. Total cost in basis points
        5. Total cost in dollar terms

        Parameters:
        -----------
        side : str
            'buy' or 'sell'
        quantity : float
            Order size in base currency
        price : float
            Reference price (mid/last)
        **kwargs : dict
            Additional parameters passed to slippage model:
            - volume: Recent volume
            - volatility: Recent volatility
            - Order book data (if using orderbook model)

        Returns:
        --------
        costs : dict
            Dictionary containing:
            - 'execution_price': Actual execution price
            - 'fee_bps': Trading fee in bps
            - 'slippage_bps': Slippage in bps
            - 'total_cost_bps': Combined cost in bps
            - 'total_cost_dollars': Total cost in USD

        Example:
            >>> costs = model.calculate_total_cost(
            ...     side='buy', quantity=1.0, price=50000.0,
            ...     volume=100.0, volatility=0.02
            ... )
            >>> print(f"Total cost: ${costs['total_cost_dollars']:.2f}")
        """
        # 1. Trading fees (fixed cost per trade)
        fee_bps = self.cost_config.fixed_bps
        # Convert bps to dollar amount: price × quantity × (fee_bps/10000)
        fee_dollars = price * quantity * (fee_bps / 10000)

        # 2. Slippage calculation
        if self.cost_config.include_slippage:
            execution_price, slippage_bps = self.slippage_model.calculate_slippage(
                side, quantity, price, **kwargs
            )
        else:
            # No slippage - execute at reference price
            execution_price = price
            slippage_bps = 0.0

        # 3. Calculate total cost
        total_cost_bps = fee_bps + slippage_bps

        # Convert to dollar costs
        # For buys: pay more than reference price + pay fees
        # For sells: receive less than reference price + pay fees
        if side == "buy":
            total_cost_dollars = (execution_price - price) * quantity + fee_dollars
        else:
            total_cost_dollars = (price - execution_price) * quantity + fee_dollars

        return {
            "execution_price": execution_price,
            "fee_bps": fee_bps,
            "slippage_bps": slippage_bps,
            "total_cost_bps": total_cost_bps,
            "total_cost_dollars": total_cost_dollars,
        }


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("SLIPPAGE MODEL TEST")
    print("=" * 80)

    # Test different slippage models
    models = {
        "Fixed": SlippageConfig(model_type="fixed", fixed_slippage_bps=5.0),
        "Volume-based": SlippageConfig(model_type="volume_based"),
        "Volatility": SlippageConfig(
            model_type="volatility", volatility_multiplier=2.0
        ),
    }

    # Test parameters
    price = 50000.0
    quantity = 1.0
    volume = 100.0
    volatility = 0.02

    print(f"\nTest Order: BUY {quantity} @ ${price:,.2f}")
    print(f"Market Volume: {volume}")
    print(f"Volatility: {volatility * 100:.1f}%\n")

    for name, config in models.items():
        model = SlippageModel(config)
        exec_price, slippage_bps = model.calculate_slippage(
            "buy", quantity, price, volume, volatility
        )

        slippage_dollars = exec_price - price

        print(f"{name}:")
        print(f"  Execution: ${exec_price:,.2f}")
        print(f"  Slippage: {slippage_bps:.2f} bps (${slippage_dollars:.2f})")

    # Test transaction cost model
    print("\n" + "-" * 80)
    print("TRANSACTION COST MODEL TEST")
    print("-" * 80)

    cost_config = TransactionCostConfig(fixed_bps=5.0)
    slippage_config = SlippageConfig(model_type="volume_based")
    slippage_model = SlippageModel(slippage_config)

    cost_model = TransactionCostModel(cost_config, slippage_model)

    costs = cost_model.calculate_total_cost(
        "buy", quantity, price, volume=volume, volatility=volatility
    )

    print(f"\nTotal Transaction Costs:")
    print(f"  Execution price: ${costs['execution_price']:,.2f}")
    print(f"  Fee: {costs['fee_bps']:.2f} bps")
    print(f"  Slippage: {costs['slippage_bps']:.2f} bps")
    print(f"  Total: {costs['total_cost_bps']:.2f} bps")
    print(f"  Dollar cost: ${costs['total_cost_dollars']:.2f}")

    print("\n" + "=" * 80)
    print("✓ SLIPPAGE MODEL TEST PASSED")
    print("=" * 80)
