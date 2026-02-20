"""
Slippage Model
==============
Multiple slippage calculation strategies for realistic execution modeling.

Strategies:
1. Fixed BPS: Constant slippage
2. Volume-based: Slippage proportional to order size vs available volume
3. Volatility-adjusted: Higher slippage in volatile markets
4. Order Book: Realistic slippage from walking the book
"""

import numpy as np
from typing import Tuple, Dict
from dataclasses import dataclass
from numba import jit
from loguru import logger


@dataclass
class SlippageConfig:
    """Configuration for slippage modeling."""

    model_type: str = "volume_based"  # fixed, volume_based, volatility, orderbook
    fixed_slippage_bps: float = 5.0  # For fixed model
    volatility_multiplier: float = 2.0  # For volatility model
    volume_impact_coef: float = 0.1  # For volume model


class SlippageModel:
    """
    Slippage calculation for realistic trade execution.

    Slippage represents the difference between:
    - Expected price (mid/last price)
    - Actual execution price

    Critical for backtesting: Overly optimistic slippage → Overfitting
    """

    def __init__(self, config: SlippageConfig):
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
        Calculate slippage for an order.

        Parameters:
        -----------
        side : str
            'buy' or 'sell'
        quantity : float
            Order size (in base currency)
        price : float
            Reference price (mid/last)
        volume : float
            Recent market volume
        volatility : float
            Recent volatility
        **kwargs : dict
            Additional parameters (e.g., order book data)

        Returns:
        --------
        execution_price : float
            Actual execution price
        slippage_bps : float
            Slippage in basis points
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
        """Fixed slippage in basis points."""
        slippage_bps = self.config.fixed_slippage_bps
        slippage = price * (slippage_bps / 10000)

        if side == "buy":
            execution_price = price + slippage
        else:  # sell
            execution_price = price - slippage

        return execution_price, slippage_bps

    def _volume_based_slippage(
        self, side: str, quantity: float, price: float, volume: float
    ) -> Tuple[float, float]:
        """
        Slippage proportional to order size vs market volume.

        Larger orders relative to volume → More slippage
        """
        # Participation rate
        participation = quantity / (volume + 1e-8)

        # Slippage increases non-linearly with participation
        slippage_bps = (
            self.config.fixed_slippage_bps
            + self.config.volume_impact_coef * np.sqrt(participation) * 10000
        )

        # Cap maximum slippage
        slippage_bps = min(slippage_bps, 100.0)  # Max 1%

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
        Slippage increases with volatility.

        High volatility → Wider spreads → More slippage
        """
        # Base slippage adjusted by volatility
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

        Requires: bid_prices, bid_volumes, ask_prices, ask_volumes in kwargs
        """
        bid_prices = kwargs.get("bid_prices")
        bid_volumes = kwargs.get("bid_volumes")
        ask_prices = kwargs.get("ask_prices")
        ask_volumes = kwargs.get("ask_volumes")

        if any(x is None for x in [bid_prices, bid_volumes, ask_prices, ask_volumes]):
            # Fallback to volume-based
            logger.warning("Order book data missing, using volume-based slippage")
            return self._volume_based_slippage(
                side, quantity, price, kwargs.get("volume", 100.0)
            )

        # Use Numba-optimized calculation
        if side == "buy":
            execution_price, filled = calculate_slippage_numba(
                quantity, ask_prices, ask_volumes
            )
        else:
            execution_price, filled = calculate_slippage_numba(
                quantity, bid_prices, bid_volumes
            )

        # Calculate slippage
        slippage_bps = abs(execution_price - price) / price * 10000

        return execution_price, slippage_bps


@jit(nopython=True, cache=True)
def calculate_slippage_numba(
    quantity: float, prices: np.ndarray, volumes: np.ndarray
) -> Tuple[float, float]:
    """
    Numba-optimized slippage from walking the order book.

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
# TRANSACTION COST MODEL
# ============================================================================


@dataclass
class TransactionCostConfig:
    """Configuration for transaction costs."""

    fixed_bps: float = 5.0  # 0.05% = 5 basis points
    include_slippage: bool = True
    include_market_impact: bool = True


class TransactionCostModel:
    """
    Complete transaction cost modeling.

    Total Cost = Fees + Slippage + Market Impact
    """

    def __init__(
        self, cost_config: TransactionCostConfig, slippage_model: SlippageModel
    ):
        self.cost_config = cost_config
        self.slippage_model = slippage_model
        logger.info(f"TransactionCostModel: {cost_config.fixed_bps} bps fees")

    def calculate_total_cost(
        self, side: str, quantity: float, price: float, **kwargs
    ) -> Dict[str, float]:
        """
        Calculate complete transaction costs.

        Returns:
        --------
        costs : dict
            {
                'execution_price': float,
                'fee_bps': float,
                'slippage_bps': float,
                'total_cost_bps': float,
                'total_cost_dollars': float
            }
        """
        # 1. Trading fees
        fee_bps = self.cost_config.fixed_bps
        fee_dollars = price * quantity * (fee_bps / 10000)

        # 2. Slippage
        if self.cost_config.include_slippage:
            execution_price, slippage_bps = self.slippage_model.calculate_slippage(
                side, quantity, price, **kwargs
            )
        else:
            execution_price = price
            slippage_bps = 0.0

        # 3. Total cost
        total_cost_bps = fee_bps + slippage_bps

        # Dollar cost
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
