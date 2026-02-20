"""
Cryptocurrency Perpetual Futures Trading Environment
=====================================================

Specialized environment for trading perpetual futures contracts
with leverage, funding rates, and realistic margin mechanics.

Features:
- Perpetual futures (BTC-PERP, ETH-PERP, etc.)
- Leverage (1x - 125x)
- Funding rates (8h intervals)
- Long/Short positions
- Liquidation logic
- Margin requirements
- Realistic Binance fees

Based on Binance Futures specifications.
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
    """Position side."""

    LONG = 1
    SHORT = -1
    FLAT = 0


@dataclass
class CryptoFuturesConfig:
    """Configuration for crypto futures environment."""

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

    State Space:
    - OHLCV data
    - Funding rate information
    - Position details (size, side, entry price)
    - Margin metrics (available, used, liquidation price)
    - Order book features (spread, depth)
    - Account metrics (balance, unrealized PnL)

    Action Space:
    Discrete actions:
    - 0: Close position (go flat)
    - 1: Hold
    - 2: Open/Increase Long
    - 3: Open/Increase Short
    - 4: Decrease Long
    - 5: Decrease Short

    Or continuous (optional):
    - Position size [-1, 1] (-1 = max short, 1 = max long)

    Reward:
    - Realized PnL from trades
    - Funding rate payments/collections
    - Risk-adjusted returns
    - Penalty for liquidation
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        df: pd.DataFrame,
        funding_rates: Optional[pd.DataFrame] = None,
        config: Optional[CryptoFuturesConfig] = None,
    ):
        """
        Initialize futures environment.

        Args:
            df: OHLCV data with columns [open, high, low, close, volume]
            funding_rates: Funding rates with columns [timestamp, funding_rate]
            config: Environment configuration
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

        # Action space (discrete)
        self.action_space = spaces.Discrete(6)

        # State space
        # OHLCV (5) + position info (5) + margin info (4) + funding (1) + orderbook (4) = 19
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(19,), dtype=np.float32
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
        """Reset environment."""
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
        Execute one step.

        Args:
            action: 0=Close, 1=Hold, 2=Long, 3=Short, 4=Decrease Long, 5=Decrease Short

        Returns:
            observation, reward, terminated, truncated, info
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
        Execute trading action.

        Returns:
            (trade_pnl, trade_fee)
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
        """Open or increase long position."""
        # Calculate max position size
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
                    # Update average entry price
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

    def _open_short(self, price: float) -> Tuple[float, float]:
        """Open or increase short position."""
        # Similar logic to _open_long but for short
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
        """Close entire position."""
        if self.position_side == PositionSide.FLAT:
            return 0.0, 0.0

        notional_value = self.position_size * price
        fee = notional_value * self.config.taker_fee

        # Calculate PnL
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
        """Decrease position by percentage."""
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
        """Apply funding rate payment/collection."""
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
        # Short pays, Long receives (when funding_rate < 0)
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

        return (
            -funding_payment
            if self.position_side == PositionSide.LONG
            else funding_payment
        )

    def _get_current_funding_rate(self) -> float:
        """Get funding rate for current step."""
        if self.funding_rates is not None:
            # Look up funding rate from data
            idx = min(self.current_step, len(self.funding_rates) - 1)
            return self.funding_rates.iloc[idx].get("funding_rate", 0.0001)

        # Simulate realistic funding rate
        # Binance funding rates typically range from -0.1% to +0.1%
        # Mean around 0.01% (positive = longs pay shorts)
        base_rate = 0.0001  # 0.01%
        noise = np.random.normal(0, 0.0003)  # 0.03% std
        return base_rate + noise

    def _check_liquidation(self) -> bool:
        """Check if position should be liquidated."""
        if self.position_side == PositionSide.FLAT:
            return False

        current_price = self.df.iloc[self.current_step]["close"]
        notional_value = self.position_size * current_price

        # Calculate unrealized PnL
        if self.position_side == PositionSide.LONG:
            unrealized_pnl = (current_price - self.entry_price) * self.position_size
            liquidation_price = self.entry_price * (
                1 - 1 / self.leverage + self.config.maintenance_margin_rate
            )
            liquidated = current_price <= liquidation_price
        else:  # SHORT
            unrealized_pnl = (self.entry_price - current_price) * self.position_size
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
        """Liquidate position."""
        current_price = self.df.iloc[self.current_step]["close"]

        # Calculate final PnL with penalty
        if self.position_side == PositionSide.LONG:
            pnl = (current_price - self.entry_price) * self.position_size
        else:
            pnl = (self.entry_price - current_price) * self.position_size

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
        """Calculate total equity (balance + unrealized PnL)."""
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
        """Calculate reward signal."""
        # Base return
        equity_change = new_equity - old_equity

        # Penalty for liquidation
        liquidation_penalty = 0.0
        if self.liquidated:
            liquidation_penalty = -100.0

        # Penalty for excessive trading
        trading_penalty = 0.0
        if abs(trade_pnl) > 0:
            trading_penalty = -0.01

        # Combined reward
        reward = equity_change / (
            self.config.initial_capital * 0.01
        )  # Scale by 1% of capital
        reward += liquidation_penalty
        reward += trading_penalty

        return float(np.clip(reward, -50, 50))

    def _get_observation(self) -> np.ndarray:
        """Construct observation vector."""
        # Get current market data
        current = self.df.iloc[self.current_step]

        # Market features (normalized)
        obs = np.array(
            [
                current["open"] / current["close"] - 1,  # Open/close ratio
                current["high"] / current["close"] - 1,  # High/close ratio
                current["low"] / current["close"] - 1,  # Low/close ratio
                0.0,  # Close is reference
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
                    self.position_size
                    * current_price
                    / self.config.initial_capital,  # Position value / capital
                    unrealized_pnl_pct * self.leverage,  # Leveraged PnL
                    self.initial_margin / self.config.initial_capital,  # Margin usage
                    self.available_balance
                    / self.config.initial_capital,  # Available ratio
                ]
            )

        # Funding rate
        funding_rate = self._get_current_funding_rate() * 10000  # Scale up

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
        """Get additional information."""
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
        """Render environment state."""
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
        """Get performance summary for episode."""
        total_return = (
            self.equity_curve[-1] - self.config.initial_capital
        ) / self.config.initial_capital

        # Calculate Sharpe-like metric
        if len(self.equity_curve) > 1:
            returns = np.diff(self.equity_curve) / np.array(self.equity_curve[:-1])
            sharpe = (
                np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(365 * 24)
            )  # Hourly to annual
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
    """Factory function to create crypto futures environment."""
    config = CryptoFuturesConfig(**kwargs)
    return CryptoFuturesEnv(df, funding_rates, config)
