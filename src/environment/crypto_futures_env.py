"""
Cryptocurrency Perpetual Futures Trading Environment
=====================================================

Purpose:
--------
This module implements a Gymnasium-compatible environment for trading
cryptocurrency perpetual futures contracts. It provides a realistic
simulation of perpetual futures trading including leverage, funding
rates, margin mechanics, and liquidation.

Key Features:
-------------
1. Perpetual Futures: Simulates BTC-PERP, ETH-PERP, etc.
2. Leverage: Supports 1x to 125x leverage (Binance-style)
3. Funding Rates: 8-hour funding intervals (simulated or from data)
4. Long/Short: Full support for both directions
5. Liquidation: Realistic liquidation with penalty
6. Margin: Initial and maintenance margin mechanics
7. Realistic Fees: Maker/taker differentiation (Binance rates)

Differences from Spot Trading:
------------------------------
Perpetual futures have unique characteristics:
- No expiration date (unlike delivery futures)
- Funding payments every 8 hours (longs pay shorts or vice versa)
- Can go long or short with leverage
- Liquidation risk increases with leverage
- Notional value = position size × price

State Space:
------------
The observation vector includes:
- OHLCV data (5 features): normalized open, high, low, close, volume
- Position info (5): side, size/equity ratio, leveraged PnL, margin usage, cash
- Funding rate (1): current funding rate scaled
- Order book (4): simulated spread, depth metrics, imbalance
Total: 15 features

Action Space:
-------------
Discrete 6-action space:
- 0: Close position (go flat)
- 1: Hold (no action)
- 2: Open/Increase Long
- 3: Open/Increase Short
- 4: Decrease Long (partial close)
- 5: Decrease Short (partial close)

Reward:
-------
- Realized PnL from trades
- Funding rate payments/collections
- Penalty for liquidation (-50)
- Small penalty per trade (discourages overtrading)
- Clipped to [-50, 50] to prevent extreme gradients

Usage:
------
    import gymnasium as gym
    from src.environment.crypto_futures_env import CryptoFuturesEnv, CryptoFuturesConfig

    # Create environment
    config = CryptoFuturesConfig(
        initial_capital=10000.0,
        max_leverage=20.0,
        default_leverage=10.0
    )
    env = CryptoFuturesEnv(df, funding_rates, config)

    # Run episode
    obs, info = env.reset()
    for _ in range(1000):
        action = env.action_space.sample()  # Or use policy
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break

    # Get performance summary
    summary = env.get_performance_summary()

Dependencies:
-------------
- gymnasium: RL environment interface
- numpy: Numerical operations
- pandas: Data handling
- loguru: Logging

References:
-----------
- Binance Futures: https://www.binance.com/en/futures
- Perpetual Swaps: https://docs.perpetual-protocol/
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass
from enum import Enum
from loguru import logger


class PositionSide(Enum):
    """
    Position direction indicator.

    Attributes:
        LONG: Long position (profit if price rises)
        SHORT: Short position (profit if price falls)
        FLAT: No position (neutral)
    """

    LONG = 1
    SHORT = -1
    FLAT = 0


@dataclass
class CryptoFuturesConfig:
    """
    Configuration for crypto futures environment.

    All parameters are adjustable to match different trading scenarios
    and exchange specifications.

    Attributes:
        initial_capital: Starting USDT balance
        symbol: Trading pair (e.g., 'BTCUSDT')
        max_leverage: Maximum allowed leverage (Binance max is 125x)
        default_leverage: Leverage to use when opening positions
        initial_margin_rate: Margin required to open position
            - 5% = 20x leverage
            - 10% = 10x leverage
        maintenance_margin_rate: Minimum margin to avoid liquidation
            - Binance uses 2.5% (0.025) for most contracts
        maker_fee: Fee for limit orders (lower than taker)
        taker_fee: Fee for market orders
        funding_interval: Steps between funding payments
            - 288 for 1-minute bars (8 hours)
            - Adjust based on your data timeframe
        max_position_pct: Maximum position as % of capital
        liquidation_penalty: Penalty % on liquidation
        timeframe: Data timeframe (for reference)
        use_orderbook: Whether to simulate order book
        orderbook_levels: Number of levels to simulate
        use_antibias_costs: Include anti-bias costs

    Example:
        >>> config = CryptoFuturesConfig(
        ...     initial_capital=10000.0,
        ...     max_leverage=125.0,
        ...     default_leverage=20.0,
        ...     initial_margin_rate=0.05,  # 5% = 20x
        ...     maintenance_margin_rate=0.025,
        ...     maker_fee=0.0002,
        ...     taker_fee=0.0004,
        ... )
    """

    # Trading settings
    initial_capital: float = 10000.0  # USDT
    symbol: str = "BTCUSDT"

    # Leverage
    max_leverage: float = 125.0  # Binance max
    default_leverage: float = 20.0

    # Margin
    initial_margin_rate: float = 0.05  # 5% for 20x leverage
    maintenance_margin_rate: float = 0.025  # 2.5%

    # Fees (Binance Futures Tier 0)
    maker_fee: float = 0.0002  # 0.02%
    taker_fee: float = 0.0004  # 0.04%

    # Funding rate
    funding_interval: int = 288  # 8h in 1-minute bars (8 * 60 = 480)
    # Note: Adjust based on your timeframe

    # Risk limits
    max_position_pct: float = 0.95  # Max 95% of capital
    liquidation_penalty: float = 0.01  # 1% penalty on liquidation

    # Timeframe
    timeframe: str = "1h"  # 1m, 5m, 15m, 1h, 4h

    # Features
    use_orderbook: bool = True
    orderbook_levels: int = 5

    # Anti-bias
    use_antibias_costs: bool = True


class CryptoFuturesEnv(gym.Env):
    """
    Perpetual Futures Trading Environment.

    This environment simulates cryptocurrency perpetual futures trading
    with realistic mechanics including leverage, margin, funding,
    and liquidation.

    State Space (19 features):
    ---------------------------
    1-5. Market Features:
        - open/close ratio - 1
        - high/close ratio - 1
        - low/close ratio - 1
        - close (reference) - 1
        - volume deviation from mean - 1

    6-10. Position Features:
        - position side (1=long, -1=short, 0=flat) - 1
        - position value / initial capital - 1
        - leveraged unrealized PnL - 1
        - margin usage ratio - 1
        - available cash ratio - 1

    11. Funding:
        - funding rate × 10000 - 1

    12-15. Order Book:
        - spread (bps) - 1
        - ask depth - 1
        - bid depth - 1
        - order imbalance - 1

    Action Space:
    -------------
    Discrete(6) with actions:
    - 0: Close position (go flat)
    - 1: Hold (no action)
    - 2: Open/Increase Long
    - 3: Open/Increase Short
    - 4: Decrease Long (close 50%)
    - 5: Decrease Short (close 50%)

    Reward:
    -------
    The reward is calculated as:
    1. Equity change normalized by capital
    2. Minus 0.01 per trade (trading penalty)
    3. Minus 50 if liquidated
    4. Clipped to [-50, 50]

    Lifecycle:
    ----------
    1. reset(): Initialize/reset episode state
    2. step(action): Execute one trading step
    3. _execute_action(): Process trading logic
    4. _apply_funding_rate(): Handle funding payments
    5. _check_liquidation(): Verify margin requirements
    6. _calculate_reward(): Compute reward signal
    7. _get_observation(): Return state vector

    Attributes:
        df: OHLCV price data
        funding_rates: Optional funding rate history
        config: Environment configuration
        action_space: Discrete(6) action space
        observation_space: Box(19,) for state

    Example:
        >>> config = CryptoFuturesConfig(initial_capital=10000.0)
        >>> env = CryptoFuturesEnv(df, funding_rates, config)
        >>> obs, info = env.reset()
        >>> action = 2  # Open long
        >>> obs, reward, term, trunc, info = env.step(action)
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        df: pd.DataFrame,
        funding_rates: Optional[pd.DataFrame] = None,
        config: Optional[CryptoFuturesConfig] = None,
    ):
        """
        Initialize futures trading environment.

        Args:
            df: OHLCV data with columns [open, high, low, close, volume]
                Must have DatetimeIndex or integer index
            funding_rates: Optional DataFrame with funding rates
                Should have 'timestamp' and 'funding_rate' columns
            config: Environment configuration (uses defaults if None)

        Raises:
            ValueError: If required columns are missing from df
        """
        super().__init__()

        self.df = df.reset_index(drop=True)
        self.funding_rates = funding_rates
        self.config = config or CryptoFuturesConfig()

        # Validate data
        required_cols = ["open", "high", "low", "close", "volume"]
        for col in required_cols:
            if col not in self.df.columns:
                raise ValueError(f"Missing required column: {col}")

        # Action space (discrete 6 actions)
        self.action_space = spaces.Discrete(6)

        # State space: 5 OHLCV + 5 position + 1 funding + 4 orderbook = 15
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(15,), dtype=np.float32
        )

        # Initialize state
        self.current_step = 0
        self.balance = self.config.initial_capital
        self.position_size = 0.0  # In contracts (notional value)
        self.position_side = PositionSide.FLAT
        self.entry_price = 0.0
        self.leverage = self.config.default_leverage

        # Margin tracking
        self.initial_margin = 0.0
        self.maintenance_margin = 0.0
        self.available_balance = self.config.initial_capital

        # Funding tracking
        self.total_funding_paid = 0.0
        self.total_funding_received = 0.0
        self.last_funding_step = 0

        # Performance tracking
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_realized_pnl = 0.0
        self.total_fees = 0.0

        # Episode tracking
        self.equity_curve = [self.config.initial_capital]
        self.trade_history = []
        self.liquidated = False

        logger.info(f"CryptoFuturesEnv initialized: {self.config.symbol}")
        logger.info(f"  Leverage: {self.leverage}x")
        logger.info(f"  Timeframe: {self.config.timeframe}")
        logger.info(f"  Data points: {len(self.df)}")

    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        """
        Reset environment to initial state for new episode.

        Called at the start of each episode to reset all state variables
        to their initial values.

        Args:
            seed: Random seed for reproducibility (optional)
            options: Additional reset options (optional)

        Returns:
            observation: Initial state vector
            info: Additional information dict

        Example:
            >>> obs, info = env.reset()
            >>> # Or with seed
            >>> obs, info = env.reset(seed=42)
        """
        super().reset(seed=seed)

        # Reset to initial state
        self.current_step = 0
        self.balance = self.config.initial_capital
        self.position_size = 0.0
        self.position_side = PositionSide.FLAT
        self.entry_price = 0.0
        self.leverage = self.config.default_leverage

        self.initial_margin = 0.0
        self.maintenance_margin = 0.0
        self.available_balance = self.config.initial_capital

        self.total_funding_paid = 0.0
        self.total_funding_received = 0.0
        self.last_funding_step = 0

        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_realized_pnl = 0.0
        self.total_fees = 0.0

        self.equity_curve = [self.config.initial_capital]
        self.trade_history = []
        self.liquidated = False

        obs = self._get_observation()
        info = self._get_info()

        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one step in the environment.

        This is the main interaction method. The agent selects an action,
        the environment executes it, and returns the new state, reward,
        and episode status.

        Args:
            action: Integer in range [0, 5]
                - 0: Close position (go flat)
                - 1: Hold (do nothing)
                - 2: Open/Increase Long
                - 3: Open/Increase Short
                - 4: Decrease Long (close 50%)
                - 5: Decrease Short (close 50%)

        Returns:
            observation: State vector (15 features)
            reward: Step reward (clipped to [-50, 50])
            terminated: True if episode ended due to liquidation
            truncated: True if episode ended due to time limit
            info: Dict with additional information

        Example:
            >>> obs, reward, terminated, truncated, info = env.step(2)
            >>> print(f"Reward: {reward:.2f}, Done: {terminated}")
        """
        # Store old equity for reward calculation
        old_equity = self._calculate_equity()

        # Execute action
        trade_pnl, trade_fee = self._execute_action(action)

        # Apply funding rate if interval reached
        funding_pnl = self._apply_funding_rate()

        # Move to next step
        self.current_step += 1

        # Check termination
        terminated = False
        truncated = False

        # Check liquidation
        if self._check_liquidation():
            self._liquidate_position()
            terminated = True
            self.liquidated = True
            logger.warning(f"LIQUIDATION at step {self.current_step}!")

        # Check end of data
        if self.current_step >= len(self.df) - 1:
            truncated = True

        # Calculate reward
        new_equity = self._calculate_equity()
        reward = self._calculate_reward(old_equity, new_equity, trade_pnl, funding_pnl)

        # Update equity curve
        self.equity_curve.append(new_equity)

        # Get observation and info
        obs = self._get_observation()
        info = self._get_info()
        info["trade_pnl"] = trade_pnl
        info["funding_pnl"] = funding_pnl
        info["liquidated"] = self.liquidated

        return obs, reward, terminated, truncated, info

    def _execute_action(self, action: int) -> Tuple[float, float]:
        """
        Execute the selected trading action.

        Maps action integers to position management operations:
        - Close: Close entire position
        - Hold: No changes
        - Long: Open/increase long position
        - Short: Open/increase short position
        - Decrease: Partially close position

        Args:
            action: Action integer from step()

        Returns:
            trade_pnl: Realized PnL from the trade (0 if no trade)
            trade_fee: Trading fee paid
        """
        current_price = self.df.iloc[self.current_step]["close"]
        trade_pnl = 0.0
        trade_fee = 0.0

        if action == 0:  # Close position
            if self.position_side != PositionSide.FLAT:
                trade_pnl, trade_fee = self._close_position(current_price)

        elif action == 1:  # Hold
            pass

        elif action == 2:  # Open/Increase Long
            trade_pnl, trade_fee = self._open_long(current_price)

        elif action == 3:  # Open/Increase Short
            trade_pnl, trade_fee = self._open_short(current_price)

        elif action == 4:  # Decrease Long
            if self.position_side == PositionSide.LONG:
                trade_pnl, trade_fee = self._decrease_position(current_price, 0.5)

        elif action == 5:  # Decrease Short
            if self.position_side == PositionSide.SHORT:
                trade_pnl, trade_fee = self._decrease_position(current_price, 0.5)

        return trade_pnl, trade_fee

    def _open_long(self, price: float) -> Tuple[float, float]:
        """
        Open or increase a long position.

        If flat, opens new long position. If already long, increases size.
        Uses available balance × leverage for max position.

        Args:
            price: Current market price for execution

        Returns:
            trade_pnl: Realized PnL (0 for new position)
            trade_fee: Trading fee paid
        """
        # Calculate max position size based on available margin
        max_notional = (
            self.available_balance * self.leverage * self.config.max_position_pct
        )

        # If flat, open new position
        if self.position_side == PositionSide.FLAT:
            position_size = max_notional / price
            notional_value = position_size * price

            # Calculate margin required
            initial_margin = notional_value / self.leverage
            fee = notional_value * self.config.taker_fee

            if initial_margin + fee <= self.available_balance:
                self.position_size = position_size
                self.position_side = PositionSide.LONG
                self.entry_price = price
                self.initial_margin = initial_margin
                self.available_balance -= initial_margin + fee
                self.total_fees += fee
                self.total_trades += 1

                return 0.0, fee

        # If already long, increase position
        elif self.position_side == PositionSide.LONG:
            additional_size = (max_notional / price) - self.position_size
            if additional_size > 0:
                notional_value = additional_size * price
                initial_margin = notional_value / self.leverage
                fee = notional_value * self.config.taker_fee

                if initial_margin + fee <= self.available_balance:
                    # Compute new volume-weighted average entry price
                    total_notional = (
                        self.position_size
                        * self.entry_price  # Existing position notional
                    ) + notional_value  # Plus new position notional
                    self.position_size += additional_size
                    self.entry_price = (
                        total_notional / self.position_size
                    )  # VWAP entry price
                    self.initial_margin += initial_margin
                    self.available_balance -= initial_margin + fee
                    self.total_fees += fee

                    return 0.0, fee

        return 0.0, 0.0

    def _open_short(self, price: float) -> Tuple[float, float]:
        """
        Open or increase a short position.

        Similar logic to _open_long but for short positions.

        Args:
            price: Current market price

        Returns:
            trade_pnl: Realized PnL
            trade_fee: Trading fee
        """
        max_notional = (
            self.available_balance * self.leverage * self.config.max_position_pct
        )

        if self.position_side == PositionSide.FLAT:
            position_size = max_notional / price
            notional_value = position_size * price

            initial_margin = notional_value / self.leverage
            fee = notional_value * self.config.taker_fee

            if initial_margin + fee <= self.available_balance:
                self.position_size = position_size
                self.position_side = PositionSide.SHORT
                self.entry_price = price
                self.initial_margin = initial_margin
                self.available_balance -= initial_margin + fee
                self.total_fees += fee
                self.total_trades += 1

                return 0.0, fee

        elif self.position_side == PositionSide.SHORT:
            additional_size = (max_notional / price) - self.position_size
            if additional_size > 0:
                notional_value = additional_size * price
                initial_margin = notional_value / self.leverage
                fee = notional_value * self.config.taker_fee

                if initial_margin + fee <= self.available_balance:
                    total_notional = (
                        self.position_size * self.entry_price
                    ) + notional_value
                    self.position_size += additional_size
                    self.entry_price = total_notional / self.position_size
                    self.initial_margin += initial_margin
                    self.available_balance -= initial_margin + fee
                    self.total_fees += fee

                    return 0.0, fee

        return 0.0, 0.0

    def _close_position(self, price: float) -> Tuple[float, float]:
        """
        Close entire position at market price.

        Calculates PnL based on entry vs exit price, returns margin,
        and records trade in history.

        Args:
            price: Exit price

        Returns:
            pnl: Realized profit/loss
            fee: Trading fee paid
        """
        if self.position_side == PositionSide.FLAT:
            return 0.0, 0.0

        notional_value = self.position_size * price
        fee = notional_value * self.config.taker_fee

        # Calculate PnL based on direction
        if self.position_side == PositionSide.LONG:
            pnl = (price - self.entry_price) * self.position_size
        else:  # SHORT
            pnl = (self.entry_price - price) * self.position_size

        # Update balance
        self.balance += pnl - fee
        self.available_balance += self.initial_margin + pnl - fee
        self.total_fees += fee
        self.total_realized_pnl += pnl

        # Track win/loss
        if pnl > 0:
            self.winning_trades += 1
        else:
            self.losing_trades += 1

        # Record trade
        self.trade_history.append(
            {
                "step": self.current_step,
                "side": self.position_side.name,
                "entry_price": self.entry_price,
                "exit_price": price,
                "size": self.position_size,
                "pnl": pnl,
                "fee": fee,
            }
        )

        # Reset position
        self.position_size = 0.0
        self.position_side = PositionSide.FLAT
        self.entry_price = 0.0
        self.initial_margin = 0.0

        return pnl, fee

    def _decrease_position(self, price: float, pct: float) -> Tuple[float, float]:
        """
        Decrease position by percentage.

        Partially closes position, returning proportional margin and PnL.

        Args:
            price: Exit price
            pct: Fraction to close (0.5 = close 50%)

        Returns:
            pnl: Realized PnL from closed portion
            fee: Trading fee
        """
        if self.position_side == PositionSide.FLAT:
            return 0.0, 0.0

        close_size = self.position_size * pct
        notional_value = close_size * price
        fee = notional_value * self.config.taker_fee

        # Calculate PnL for closed portion
        if self.position_side == PositionSide.LONG:
            pnl = (price - self.entry_price) * close_size
        else:
            pnl = (self.entry_price - price) * close_size

        # Update
        close_margin = self.initial_margin * pct
        self.balance += pnl - fee
        self.available_balance += close_margin + pnl - fee
        self.total_fees += fee
        self.total_realized_pnl += pnl

        self.position_size -= close_size
        self.initial_margin -= close_margin

        if pnl > 0:
            self.winning_trades += 1
        else:
            self.losing_trades += 1

        return pnl, fee

    def _apply_funding_rate(self) -> float:
        """
        Apply funding rate payment/collection.

        Perpetual futures have funding payments every 8 hours (typically).
        Longs pay shorts when funding rate is positive, and vice versa.

        Returns:
            funding_pnl: Funding payment (negative for payment, positive for receipt)
        """
        if self.position_side == PositionSide.FLAT:
            return 0.0

        # Check if funding interval reached
        steps_since_last = self.current_step - self.last_funding_step
        if steps_since_last < self.config.funding_interval:
            return 0.0

        # Get funding rate
        funding_rate = self._get_current_funding_rate()

        # Calculate funding payment
        notional_value = self.position_size * self.df.iloc[self.current_step]["close"]
        funding_payment = notional_value * funding_rate

        # Long pays, Short receives (when funding_rate > 0)
        if self.position_side == PositionSide.LONG:
            if funding_rate > 0:
                self.balance -= funding_payment
                self.total_funding_paid += funding_payment
            else:
                self.balance += abs(funding_payment)
                self.total_funding_received += abs(funding_payment)
        else:  # SHORT
            if funding_rate > 0:
                self.balance += funding_payment
                self.total_funding_received += funding_payment
            else:
                self.balance -= abs(funding_payment)
                self.total_funding_paid += abs(funding_payment)

        self.last_funding_step = self.current_step

        # Return PnL impact (negative = cost)
        return (
            -funding_payment
            if self.position_side == PositionSide.LONG
            else funding_payment
        )

    def _get_current_funding_rate(self) -> float:
        """
        Get funding rate for current step.

        If funding_rates DataFrame provided, uses historical data.
        Otherwise simulates realistic funding rate.

        Binance funding rates typically range from -0.1% to +0.1%

        Returns:
            funding_rate: Current funding rate (e.g., 0.0001 = 0.01%)
        """
        if self.funding_rates is not None:
            # Look up funding rate from data
            idx = min(self.current_step, len(self.funding_rates) - 1)
            return self.funding_rates.iloc[idx].get("funding_rate", 0.0001)

        # Simulate realistic funding rate
        # Mean around 0.01% (positive = longs pay shorts)
        base_rate = 0.0001  # 0.01% base
        noise = np.random.normal(0, 0.0003)  # 0.03% std
        return base_rate + noise

    def _check_liquidation(self) -> bool:
        """
        Check if position should be liquidated.

        Liquidation occurs when:
        1. Price reaches liquidation price (for position direction)
        2. Wallet balance falls below maintenance margin

        Liquidation price formulas:
        - Long: entry × (1 - 1/leverage + maintenance_margin_rate)
        - Short: entry × (1 + 1/leverage - maintenance_margin_rate)

        Returns:
            liquidated: True if liquidation should occur
        """
        if self.position_side == PositionSide.FLAT:
            return False

        current_price = self.df.iloc[self.current_step]["close"]
        notional_value = self.position_size * current_price

        # Calculate unrealized PnL
        if self.position_side == PositionSide.LONG:
            unrealized_pnl = (current_price - self.entry_price) * self.position_size
            # Long liquidation: price falls below threshold
            liquidation_price = self.entry_price * (
                1 - 1 / self.leverage + self.config.maintenance_margin_rate
            )
            liquidated = current_price <= liquidation_price
        else:  # SHORT
            unrealized_pnl = (self.entry_price - current_price) * self.position_size
            # Short liquidation: price rises above threshold
            liquidation_price = self.entry_price * (
                1 + 1 / self.leverage - self.config.maintenance_margin_rate
            )
            liquidated = current_price >= liquidation_price

        # Check margin ratio
        wallet_balance = self.balance + unrealized_pnl
        maintenance_margin = notional_value * self.config.maintenance_margin_rate

        if wallet_balance < maintenance_margin:
            return True

        return liquidated

    def _liquidate_position(self):
        """
        Execute forced liquidation of position.

        When margin falls below maintenance threshold, position is
        automatically closed with a penalty. The penalty is typically
        used to cover liquidation costs.
        """
        current_price = self.df.iloc[self.current_step]["close"]

        # Calculate final PnL with liquidation penalty
        if self.position_side == PositionSide.LONG:
            pnl = (current_price - self.entry_price) * self.position_size
        else:
            pnl = (self.entry_price - current_price) * self.position_size

        # Apply liquidation penalty
        penalty = abs(pnl) * self.config.liquidation_penalty
        final_pnl = pnl - penalty

        self.balance += final_pnl
        self.total_realized_pnl += final_pnl
        self.losing_trades += 1

        # Reset position
        self.position_size = 0.0
        self.position_side = PositionSide.FLAT
        self.entry_price = 0.0
        self.initial_margin = 0.0
        self.available_balance = self.balance

        logger.warning(f"Position liquidated! Loss: {final_pnl:.2f} USDT")

    def _calculate_equity(self) -> float:
        """
        Calculate total account equity.

        Equity = Balance + Unrealized PnL

        This is the total account value including unrealized gains/losses
        from open positions.

        Returns:
            equity: Total equity in USDT
        """
        equity = self.balance

        if self.position_side != PositionSide.FLAT:
            current_price = self.df.iloc[self.current_step]["close"]
            if self.position_side == PositionSide.LONG:
                unrealized_pnl = (current_price - self.entry_price) * self.position_size
            else:
                unrealized_pnl = (self.entry_price - current_price) * self.position_size
            equity += unrealized_pnl

        return max(0.0, equity)

    def _calculate_reward(
        self, old_equity: float, new_equity: float, trade_pnl: float, funding_pnl: float
    ) -> float:
        """
        Calculate reward signal for the step.

        Reward components:
        1. Equity change normalized (main reward)
        2. Small penalty per trade (discourages overtrading)
        3. Large penalty if liquidated

        Final reward is clipped to [-50, 50] to prevent extreme gradients.

        Args:
            old_equity: Equity before step
            new_equity: Equity after step
            trade_pnl: PnL from trade
            funding_pnl: Funding payment/receipt

        Returns:
            reward: Step reward
        """
        # Base return: equity change normalized
        equity_change = new_equity - old_equity

        # Penalty for liquidation
        liquidation_penalty = 0.0
        if self.liquidated:
            liquidation_penalty = -50.0

        # Penalty for excessive trading (discourages overtrading)
        trading_penalty = 0.0
        if abs(trade_pnl) > 0:
            trading_penalty = -0.01

        # Combined reward
        # Normalize by 1% of capital so reward ≈ 1.0 per 1% gain
        reward = equity_change / (self.config.initial_capital * 0.01)
        reward += liquidation_penalty
        reward += trading_penalty

        return float(np.clip(reward, -50, 50))

    def _get_observation(self) -> np.ndarray:
        """
        Construct observation vector for current state.

        Returns 15-dimensional state vector with market, position,
        funding, and order book features.

        Returns:
            obs: State vector (15 features, float32)
        """
        current = self.df.iloc[self.current_step]

        # Market features (normalized)
        obs = np.array(
            [
                current["open"] / current["close"] - 1,  # Open/close ratio
                current["high"] / current["close"] - 1,  # High/close ratio
                current["low"] / current["close"] - 1,  # Low/close ratio
                0.0,  # Close is reference (already normalized)
                current["volume"] / self.df["volume"].mean() - 1,  # Volume ratio
            ]
        )

        # Position features
        if self.position_side == PositionSide.FLAT:
            position_features = np.array([0, 0, 0, 0, 0])
        else:
            current_price = current["close"]
            unrealized_pnl_pct = 0.0
            if self.position_side == PositionSide.LONG:
                unrealized_pnl_pct = (
                    current_price - self.entry_price
                ) / self.entry_price
            else:
                unrealized_pnl_pct = (
                    self.entry_price - current_price
                ) / self.entry_price

            position_features = np.array(
                [
                    1 if self.position_side == PositionSide.LONG else -1,
                    self.position_size * current_price / self.config.initial_capital,
                    unrealized_pnl_pct * self.leverage,
                    self.initial_margin / self.config.initial_capital,
                    self.available_balance / self.config.initial_capital,
                ]
            )

        # Funding rate (scaled up for numerical stability)
        funding_rate = self._get_current_funding_rate() * 10000

        # Order book features (simulated)
        spread = 0.0001  # 0.01%
        depth_ask = 1.0
        depth_bid = 1.0
        imbalance = 0.0

        orderbook_features = np.array([spread * 10000, depth_ask, depth_bid, imbalance])

        # Combine all features
        obs = np.concatenate(
            [obs, position_features, [funding_rate], orderbook_features]
        )

        return obs.astype(np.float32)

    def _get_info(self) -> Dict:
        """
        Get additional information about current state.

        Returns dictionary with detailed state information useful
        for logging, debugging, and analysis.

        Returns:
            info: Dict with state information
        """
        current_price = self.df.iloc[self.current_step]["close"]

        info = {
            "step": self.current_step,
            "balance": self.balance,
            "equity": self._calculate_equity(),
            "position_size": self.position_size,
            "position_side": self.position_side.name
            if self.position_side != PositionSide.FLAT
            else "FLAT",
            "entry_price": self.entry_price,
            "leverage": self.leverage,
            "available_balance": self.available_balance,
            "margin_ratio": self.initial_margin / (self.balance + 1e-8)
            if self.balance > 0
            else 0,
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "win_rate": self.winning_trades / max(1, self.total_trades),
            "total_realized_pnl": self.total_realized_pnl,
            "total_fees": self.total_fees,
            "total_funding_paid": self.total_funding_paid,
            "total_funding_received": self.total_funding_received,
            "liquidated": self.liquidated,
        }

        # Add unrealized PnL if in position
        if self.position_side != PositionSide.FLAT:
            if self.position_side == PositionSide.LONG:
                unrealized_pnl = (current_price - self.entry_price) * self.position_size
            else:
                unrealized_pnl = (self.entry_price - current_price) * self.position_size
            info["unrealized_pnl"] = unrealized_pnl

        return info

    def render(self, mode="human"):
        """
        Render environment state (human-readable).

        Prints current state to console for debugging.

        Args:
            mode: Rendering mode ('human' supported)
        """
        if mode == "human":
            info = self._get_info()
            print(f"\n{'=' * 60}")
            print(f"Step: {info['step']}")
            print(f"Price: {self.df.iloc[self.current_step]['close']:.2f}")
            print(f"Balance: {info['balance']:.2f} USDT")
            print(f"Equity: {info['equity']:.2f} USDT")
            print(
                f"Position: {info['position_side']} {info['position_size']:.4f} @ {info['entry_price']:.2f}"
            )
            print(f"Leverage: {info['leverage']}x")
            print(f"Available: {info['available_balance']:.2f} USDT")
            print(
                f"Win Rate: {info['win_rate'] * 100:.1f}% ({info['winning_trades']}/{info['total_trades']})"
            )
            print(f"Total PnL: {info['total_realized_pnl']:.2f} USDT")
            print(f"Total Fees: {info['total_fees']:.2f} USDT")
            if info["liquidated"]:
                print(f"⚠️  LIQUIDATED!")
            print(f"{'=' * 60}")

    def get_performance_summary(self) -> Dict:
        """
        Get episode performance summary.

        Returns comprehensive metrics for the completed episode including
        returns, risk metrics, and trading statistics.

        Returns:
            summary: Dict with performance metrics

        Metrics:
            - total_return: Absolute return
            - total_return_pct: Return percentage
            - sharpe_ratio: Annualized Sharpe-like ratio
            - max_drawdown: Maximum drawdown (fraction)
            - max_drawdown_pct: Max drawdown percentage
            - total_trades: Number of trades
            - win_rate: Winning trade percentage
            - profit_factor: Gross profits / gross losses
            - total_fees: Sum of all fees paid
            - total_funding_paid: Funding payments
            - total_funding_received: Funding receipts
            - liquidated: Whether position was liquidated
            - final_equity: Final account equity
        """
        total_return = (
            self.equity_curve[-1] - self.config.initial_capital
        ) / self.config.initial_capital

        # Calculate Sharpe-like metric
        if len(self.equity_curve) > 1:
            returns = np.diff(self.equity_curve) / np.array(self.equity_curve[:-1])
            sharpe = (
                np.mean(returns)
                / (np.std(returns) + 1e-8)
                * np.sqrt(365 * 24)  # Annualize from hourly
            )
        else:
            sharpe = 0.0

        # Max drawdown
        peak = np.maximum.accumulate(self.equity_curve)
        drawdown = (np.array(self.equity_curve) - peak) / peak
        max_drawdown = np.min(drawdown)

        return {
            "total_return": total_return,
            "total_return_pct": total_return * 100,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_drawdown,
            "max_drawdown_pct": max_drawdown * 100,
            "total_trades": self.total_trades,
            "win_rate": self.winning_trades / max(1, self.total_trades),
            "profit_factor": abs(self.total_realized_pnl)
            / (abs(self.total_realized_pnl) + 1e-8)
            if self.total_realized_pnl > 0
            else 0,
            "total_fees": self.total_fees,
            "total_funding_paid": self.total_funding_paid,
            "total_funding_received": self.total_funding_received,
            "liquidated": self.liquidated,
            "final_equity": self.equity_curve[-1],
        }


# Convenience functions
def create_crypto_futures_env(
    df: pd.DataFrame, funding_rates: Optional[pd.DataFrame] = None, **kwargs
) -> CryptoFuturesEnv:
    """
    Factory function to create crypto futures environment.

    Convenience wrapper for environment instantiation with kwargs.

    Args:
        df: OHLCV price data
        funding_rates: Optional funding rate history
        **kwargs: Arguments passed to CryptoFuturesConfig

    Returns:
        env: Configured CryptoFuturesEnv

    Example:
        >>> env = create_crypto_futures_env(
        ...     df,
        ...     funding_rates=funding_rates,
        ...     initial_capital=10000.0,
        ...     max_leverage=20.0
        ... )
    """
    config = CryptoFuturesConfig(**kwargs)
    return CryptoFuturesEnv(df, funding_rates, config)
