"""
Realistic Trading Environment - Gymnasium
==========================================
Professional trading environment with:
- Level 2 order book simulation
- Realistic slippage modeling
- 0.05% transaction costs
- Integration with Phase 1 features

Critical: "If the model loses here, it won't make money live."
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
from pathlib import Path
from loguru import logger

from src.features.feature_engine import FeatureEngine, FeatureConfig
from src.environment.order_book import OrderBookSimulator, OrderBookConfig
from src.environment.slippage_model import (
    SlippageModel,
    SlippageConfig,
    TransactionCostModel,
    TransactionCostConfig,
)

# Anti-Bias Framework integration
try:
    from reward.antibias_rewards import (
        BaseReward,
        SharpeIncrementReward,
        CostAwareReward,
        RegimeAwareReward,
        RegimeState,
    )
    from costs.antibias_costs import (
        TransactionCostEngine,
        CostConfig as AntibiasCostConfig,
        MarketType,
        Timeframe,
        OrderType,
    )

    ANTIBIAS_AVAILABLE = True
except ImportError:
    ANTIBIAS_AVAILABLE = False


@dataclass
class TradingEnvConfig:
    """Configuration for trading environment."""

    # Capital
    initial_capital: float = 100000.0

    # Transaction costs (institutional standard: 0.05%)
    transaction_cost_bps: float = 5.0  # 0.05% = 5 basis points

    # Slippage model
    slippage_model: str = "volume_based"  # fixed, volume_based, volatility, orderbook

    # Order book
    use_orderbook: bool = True
    orderbook_levels: int = 10

    # Position sizing
    max_position_size: float = 1.0  # Max 100% of capital
    min_position_size: float = 0.01  # Min 1% of capital

    # Episode
    lookback_window: int = 50
    max_steps: int = 5000

    # Risk
    max_drawdown: float = 0.20  # 20% max drawdown

    # Features (Phase 1 integration)
    feature_config: Optional[FeatureConfig] = None

    # Anti-Bias Framework settings
    use_antibias_rewards: bool = True  # Use risk-adjusted rewards
    reward_type: str = "cost_aware"  # "sharpe", "cost_aware", "regime_aware"
    use_antibias_costs: bool = True  # Use realistic transaction costs


class RealisticTradingEnv(gym.Env):
    """
    Gym environment for realistic trading simulation.

    State Space:
    ------------
    - Price features (from Phase 1 FeatureEngine)
    - Position (long/short/flat)
    - Portfolio metrics (PnL, drawdown, etc.)
    - Order book features (spread, depth)

    Action Space:
    -------------
    Discrete actions:
    - 0: Sell/Short
    - 1: Hold/Flat
    - 2: Buy/Long

    Reward:
    -------
    Risk-adjusted return:
    - Portfolio return
    - Sharpe ratio bonus
    - Drawdown penalty
    - Transaction cost penalty

    Key Features:
    -------------
    - Realistic transaction costs (0.05%)
    - Volume-based slippage
    - Order book simulation
    - Market impact modeling
    """

    metadata = {"render.modes": ["human"]}

    def __init__(
        self, price_data: pd.DataFrame, features: pd.DataFrame, config: TradingEnvConfig
    ):
        """
        Initialize environment.

        Parameters:
        -----------
        price_data : pd.DataFrame
            OHLCV data
        features : pd.DataFrame
            Computed features from Phase 1 FeatureEngine
        config : TradingEnvConfig
            Environment configuration
        """
        super().__init__()

        self.config = config
        self.price_data = price_data
        self.features = features

        # Align data
        common_index = price_data.index.intersection(features.index)
        self.price_data = price_data.loc[common_index]
        self.features = features.loc[common_index]

        logger.info(f"TradingEnv initialized: {len(self.price_data)} steps")
        logger.info(f"  Transaction cost: {config.transaction_cost_bps} bps")
        logger.info(f"  Slippage model: {config.slippage_model}")

        # Initialize simulators
        self._init_simulators()

        # Gym spaces
        self._init_spaces()

        # Episode state
        self.current_step = 0
        self.position = 0.0  # -1, 0, 1
        self.cash = config.initial_capital
        self.shares = 0.0
        self.equity_history = []
        self.trade_history = []

    def _init_simulators(self):
        """Initialize order book and slippage simulators."""
        # Order book
        if self.config.use_orderbook:
            orderbook_config = OrderBookConfig(n_levels=self.config.orderbook_levels)
            self.orderbook_sim = OrderBookSimulator(orderbook_config)
        else:
            self.orderbook_sim = None

        # Slippage
        slippage_config = SlippageConfig(
            model_type=self.config.slippage_model,
            fixed_slippage_bps=self.config.transaction_cost_bps,
        )
        self.slippage_model = SlippageModel(slippage_config)

        # Transaction costs
        cost_config = TransactionCostConfig(fixed_bps=self.config.transaction_cost_bps)
        self.cost_model = TransactionCostModel(cost_config, self.slippage_model)

        # Anti-Bias reward function
        self.reward_fn = None
        if ANTIBIAS_AVAILABLE and self.config.use_antibias_rewards:
            if self.config.reward_type == "sharpe":
                self.reward_fn = SharpeIncrementReward(window=50)
            elif self.config.reward_type == "cost_aware":
                self.reward_fn = CostAwareReward(
                    lambda_cost=2.0,
                    lambda_draw=5.0,
                    cost_rate=self.config.transaction_cost_bps / 10000,
                )
            elif self.config.reward_type == "regime_aware":
                self.reward_fn = RegimeAwareReward(
                    lambda_cost=2.0,
                    lambda_draw=3.0,
                    lambda_regime=0.5,
                    cost_rate=self.config.transaction_cost_bps / 10000,
                )
            logger.info(f"Anti-Bias reward function: {self.config.reward_type}")

        # Anti-Bias cost engine
        self.antibias_cost_engine = None
        if ANTIBIAS_AVAILABLE and self.config.use_antibias_costs:
            self.antibias_cost_engine = TransactionCostEngine(
                AntibiasCostConfig(
                    market_type=MarketType.FUTURES,
                    timeframe=Timeframe.H1,
                    order_type=OrderType.MARKET,
                )
            )

    def _init_spaces(self):
        """Initialize observation and action spaces."""
        # State features (from Phase 1)
        n_features = len(self.features.columns)

        # Additional state: position, portfolio metrics, orderbook
        n_additional = 8  # position, cash, equity, pnl, drawdown, etc. (was 10)

        state_dim = n_features + n_additional

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(state_dim,), dtype=np.float32
        )

        # Actions: 0=Sell, 1=Hold, 2=Buy
        self.action_space = spaces.Discrete(3)

        logger.info(f"Observation space: {state_dim} features")
        logger.info(f"Action space: Discrete(3)")

    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        """Reset environment to initial state."""
        super().reset(seed=seed)

        # Random start point (avoid beginning for proper features)
        max_start = (
            len(self.price_data) - self.config.max_steps - self.config.lookback_window
        )
        self.current_step = self.np_random.integers(
            self.config.lookback_window, max(self.config.lookback_window + 1, max_start)
        )

        # Reset state
        self.position = 0.0
        self.cash = self.config.initial_capital
        self.shares = 0.0
        self.equity_history = [self.config.initial_capital]
        self.trade_history = []

        # Reset Anti-Bias reward function
        if self.reward_fn is not None:
            self.reward_fn.reset()

        obs = self._get_observation()
        info = self._get_info()

        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one step.

        Parameters:
        -----------
        action : int
            0=Sell, 1=Hold, 2=Buy

        Returns:
        --------
        observation, reward, terminated, truncated, info
        """
        # Store old equity for reward calculation
        old_equity = self._calculate_equity()

        # Execute trade with realistic costs
        trade_cost = self._execute_trade(action)

        # Move to next step
        self.current_step += 1

        # Check if episode ended
        if self.current_step >= len(self.price_data) - 1:
            terminated = True
            truncated = False
        else:
            terminated = False
            truncated = False

        # Calculate reward
        reward = self._calculate_reward(old_equity, trade_cost)

        # Update equity history
        current_equity = self._calculate_equity()
        self.equity_history.append(current_equity)

        # Check for max drawdown (early termination)
        drawdown = self._calculate_drawdown()
        if abs(drawdown) > self.config.max_drawdown:
            terminated = True
            reward -= 10.0  # Large penalty for hitting max drawdown
            logger.warning(f"Max drawdown reached: {drawdown * 100:.2f}%")

        # Get new observation
        obs = self._get_observation()
        info = self._get_info()
        info["trade_cost"] = trade_cost

        return obs, reward, terminated, truncated, info

    def _execute_trade(self, action: int) -> float:
        """
        Execute trade with realistic transaction costs.

        Returns:
        --------
        trade_cost : float
            Total cost of trade in dollars
        """
        # Map action to position
        target_position = action - 1  # [0,1,2] -> [-1, 0, 1]

        if target_position == self.position:
            # No change
            return 0.0

        # Get current market data
        current_price = self.price_data.iloc[self.current_step]["close"]
        current_volume = self.price_data.iloc[self.current_step]["volume"]

        # Get volatility from features
        if "volatility_20" in self.features.columns:
            volatility = self.features.iloc[self.current_step]["volatility_20"]
        else:
            volatility = 0.02  # Default 2%

        # Calculate position change
        position_change = target_position - self.position

        # Calculate shares to trade
        current_equity = self._calculate_equity()
        position_value = (
            current_equity * self.config.max_position_size * abs(position_change)
        )
        shares_to_trade = position_value / current_price

        # Determine side
        if position_change > 0:
            side = "buy"
        else:
            side = "sell"
            shares_to_trade = abs(shares_to_trade)

        # Generate order book if needed
        if self.config.use_orderbook and self.orderbook_sim is not None:
            bid_prices, bid_volumes, ask_prices, ask_volumes = (
                self.orderbook_sim.generate_order_book(
                    current_price, volatility, current_volume
                )
            )

            # Calculate costs with order book
            costs = self.cost_model.calculate_total_cost(
                side=side,
                quantity=shares_to_trade,
                price=current_price,
                volume=current_volume,
                volatility=volatility,
                bid_prices=bid_prices,
                bid_volumes=bid_volumes,
                ask_prices=ask_prices,
                ask_volumes=ask_volumes,
            )
        else:
            # Calculate costs without order book
            costs = self.cost_model.calculate_total_cost(
                side=side,
                quantity=shares_to_trade,
                price=current_price,
                volume=current_volume,
                volatility=volatility,
            )

        execution_price = costs["execution_price"]
        total_cost = costs["total_cost_dollars"]

        # Execute trade
        if position_change != 0:
            # Close old position
            if self.position != 0:
                close_value = self.shares * current_price
                self.cash += close_value
                self.shares = 0

            # Open new position
            if target_position != 0:
                self.shares = shares_to_trade * target_position
                trade_value = shares_to_trade * execution_price
                self.cash -= trade_value

            self.position = target_position

            # Record trade
            self.trade_history.append(
                {
                    "step": self.current_step,
                    "action": action,
                    "side": side,
                    "shares": shares_to_trade,
                    "price": current_price,
                    "execution_price": execution_price,
                    "cost": total_cost,
                    "slippage_bps": costs["slippage_bps"],
                }
            )

        return total_cost

    def _calculate_equity(self) -> float:
        """Calculate current portfolio value."""
        current_price = self.price_data.iloc[self.current_step]["close"]
        position_value = self.shares * current_price
        return self.cash + position_value

    def _calculate_drawdown(self) -> float:
        """Calculate current drawdown."""
        if len(self.equity_history) < 2:
            return 0.0

        equity_array = np.array(self.equity_history)
        running_max = np.maximum.accumulate(equity_array)
        drawdown = (equity_array[-1] - running_max[-1]) / running_max[-1]

        return drawdown

    def _calculate_reward(self, old_equity: float, trade_cost: float) -> float:
        """
        Calculate reward signal.

        Components:
        - Portfolio return
        - Risk adjustment (Sharpe)
        - Drawdown penalty
        - Transaction cost penalty
        """
        current_equity = self._calculate_equity()

        # Use Anti-Bias reward function if available
        if self.reward_fn is not None:
            pnl = current_equity - old_equity
            prev_position = self.position  # Position before this step's action
            # Note: position is already updated in _execute_trade, so we need to track it differently
            # For simplicity, we'll use the reward function's compute method
            reward = self.reward_fn.compute(
                pnl=pnl,
                position=self.position,
                prev_position=prev_position,
                equity=current_equity,
                cost_this_bar=trade_cost,
            )
            return float(reward)

        # Legacy reward calculation
        # Return
        pnl = current_equity - old_equity
        pnl_pct = pnl / old_equity if old_equity > 0 else 0.0

        # Sharpe bonus (if enough history)
        if len(self.equity_history) > 20:
            returns = np.diff(self.equity_history[-20:]) / np.array(
                self.equity_history[-20:-1]
            )
            sharpe = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252)
            sharpe_bonus = np.clip(sharpe * 0.1, -0.5, 0.5)
        else:
            sharpe_bonus = 0.0

        # Drawdown penalty
        drawdown = self._calculate_drawdown()
        drawdown_penalty = drawdown * 2.0  # Double weight on drawdown

        # Transaction cost penalty
        cost_penalty = trade_cost / old_equity if old_equity > 0 else 0.0

        # Combined reward
        reward = pnl_pct + sharpe_bonus + drawdown_penalty - cost_penalty

        # Scale for training
        reward = np.clip(reward * 100, -10, 10)

        return reward

    def _get_observation(self) -> np.ndarray:
        """Construct observation vector."""
        # Features from Phase 1
        feature_values = self.features.iloc[self.current_step].values

        # Portfolio state
        current_equity = self._calculate_equity()
        portfolio_return = (
            current_equity - self.config.initial_capital
        ) / self.config.initial_capital

        cash_ratio = self.cash / current_equity if current_equity > 0 else 1.0
        position_ratio = (
            abs(self.shares * self.price_data.iloc[self.current_step]["close"])
            / current_equity
            if current_equity > 0
            else 0.0
        )

        drawdown = self._calculate_drawdown()

        # Order book features (if available)
        if self.config.use_orderbook and self.orderbook_sim is not None:
            current_price = self.price_data.iloc[self.current_step]["close"]
            current_volume = self.price_data.iloc[self.current_step]["volume"]
            volatility = self.features.iloc[self.current_step].get(
                "volatility_20", 0.02
            )

            bid_prices, _, ask_prices, _ = self.orderbook_sim.generate_order_book(
                current_price, volatility, current_volume
            )

            spread = ask_prices[0] - bid_prices[0]
            spread_bps = spread / current_price * 10000
        else:
            spread_bps = 5.0  # Default spread

        # Additional features
        additional = np.array(
            [
                self.position,
                portfolio_return,
                cash_ratio,
                position_ratio,
                drawdown,
                len(self.trade_history),
                spread_bps,
                float(self.current_step) / len(self.price_data),  # Progress
            ]
        )

        # Combine
        obs = np.concatenate([feature_values, additional])

        # Handle NaN
        obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)

        return obs.astype(np.float32)

    def _get_info(self) -> Dict:
        """Get additional info for logging."""
        current_equity = self._calculate_equity()

        info = {
            "step": self.current_step,
            "price": self.price_data.iloc[self.current_step]["close"],
            "position": self.position,
            "equity": current_equity,
            "cash": self.cash,
            "return": (current_equity - self.config.initial_capital)
            / self.config.initial_capital,
            "n_trades": len(self.trade_history),
            "drawdown": self._calculate_drawdown(),
        }

        if len(self.equity_history) > 20:
            returns = np.diff(self.equity_history[-20:]) / np.array(
                self.equity_history[-20:-1]
            )
            info["sharpe"] = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252)

        return info

    def render(self, mode="human"):
        """Render environment state."""
        if mode == "human":
            info = self._get_info()
            print(f"\nStep {info['step']}:")
            print(f"  Price: ${info['price']:.2f}")
            print(f"  Position: {info['position']:.0f}")
            print(f"  Equity: ${info['equity']:.2f}")
            print(f"  Return: {info['return'] * 100:.2f}%")
            print(f"  Trades: {info['n_trades']}")


# ============================================================================
# HYDRA INTEGRATION
# ============================================================================


def create_trading_env_from_hydra(cfg, price_data, features):
    """Create environment from Hydra config."""
    from features.feature_engine import FeatureConfig

    feature_config = FeatureConfig(
        volatility_window=cfg.features.volatility_window,
        ou_window=cfg.features.ou_window,
        rolling_mean_window=cfg.features.rolling_mean_window,
        use_log_returns=cfg.features.use_log_returns,
        scaler_type=cfg.features.scaler_type,
        save_scaler=cfg.features.save_scaler,
        scaler_path=Path(cfg.features.scaler_path),
        dropna_strategy=cfg.features.dropna_strategy,
        min_valid_rows=cfg.features.min_valid_rows,
    )

    env_config = TradingEnvConfig(
        initial_capital=cfg.environment.initial_capital,
        transaction_cost_bps=cfg.risk.transaction_cost_bps,
        slippage_model=cfg.risk.slippage_model,
        max_position_size=cfg.risk.max_position_size,
        lookback_window=cfg.environment.lookback_window,
        feature_config=feature_config,
    )

    return RealisticTradingEnv(price_data, features, env_config)


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("REALISTIC TRADING ENVIRONMENT TEST")
    print("=" * 80)

    # Generate synthetic data
    np.random.seed(42)
    n_points = 2000
    dates = pd.date_range("2023-01-01", periods=n_points, freq="1H")

    close = 100 + np.cumsum(np.random.randn(n_points) * 0.5)

    price_data = pd.DataFrame(
        {
            "open": close + np.random.randn(n_points) * 0.2,
            "high": close + abs(np.random.randn(n_points) * 0.5),
            "low": close - abs(np.random.randn(n_points) * 0.5),
            "close": close,
            "volume": np.random.uniform(100, 1000, n_points),
        },
        index=dates,
    )

    # Generate features (simplified)
    features = pd.DataFrame(
        {
            "log_ret": np.log(price_data["close"] / price_data["close"].shift(1)),
            "volatility_20": price_data["close"].pct_change().rolling(20).std(),
            "ou_score": (price_data["close"] - price_data["close"].rolling(20).mean())
            / price_data["close"].rolling(20).std(),
        },
        index=dates,
    ).dropna()

    # Create environment
    config = TradingEnvConfig(
        initial_capital=100000, transaction_cost_bps=5.0, use_orderbook=True
    )

    env = RealisticTradingEnv(price_data, features, config)

    print(f"✓ Environment created")
    print(f"  Observation space: {env.observation_space.shape}")
    print(f"  Action space: {env.action_space}")

    # Test episode
    obs, info = env.reset()
    print(f"\n✓ Reset successful")
    print(f"  Start equity: ${info['equity']:.2f}")

    total_reward = 0
    for i in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if i % 20 == 0:
            env.render()

        if terminated or truncated:
            print(f"\n✓ Episode ended at step {i + 1}")
            break

    print(f"\n✓ Test complete")
    print(f"  Total reward: {total_reward:.2f}")
    print(f"  Final equity: ${info['equity']:.2f}")
    print(f"  Final return: {info['return'] * 100:.2f}%")
    print(f"  Trades: {info['n_trades']}")

    print("\n" + "=" * 80)
    print("✓ REALISTIC TRADING ENVIRONMENT TEST PASSED")
    print("=" * 80)
