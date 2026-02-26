"""
Realistic Trading Environment - FULLY CONFIG-INTEGRATED
=========================================================

Purpose:
--------
This module implements the main trading environment for the BITCOIN4Traders
project, fully integrated with YAML configuration. It provides a Gymnasium-
compatible RL environment with realistic trading mechanics, risk management,
and comprehensive cost modeling.

Key Improvements over Basic Version:
---------------------------------
1. 7 Discrete Position Actions: Richer action space with Kelly-inspired sizing
2. Maker/Taker Fee Differentiation: Realistic exchange fee modeling
3. Dynamic Reward Calculation: Configurable reward components from YAML
4. Market Regime Simulation: Volatility and volume regime variations
5. Full Hydra/YAML Integration: All parameters from config files
6. Two-Layer Risk Management: Pre-trade and post-trade risk checks
7. Order Book Integration: Realistic slippage from L2 data

Architecture:
------------
The environment integrates multiple components:
- Order Book Simulator: L2 market simulation for slippage
- Slippage Model: Multiple slippage calculation strategies
- Transaction Cost Model: Complete cost breakdown
- Risk Manager: Position sizing and circuit breakers
- Kelly Criterion: Optimal position sizing

State Space:
------------
The observation vector combines:
- Feature Data: Technical indicators and custom features
- Portfolio State: Position, equity, cash ratios
- Risk Metrics: Drawdown, consecutive losses
- Market Regime: Volatility and volume factors
- Progress: Episode progress indicator

Action Space:
-------------
7 discrete actions for position sizing:
    0: Short 100% (full short)
    1: Short 50% (half short)
    2: Neutral (flat)
    3: Long 33% (quarter Kelly-inspired)
    4: Long 50% (half position)
    5: Long 75% (three-quarter)
    6: Long 100% (full long)

Risk Management (Two-Layer):
----------------------------
LAYER 1 - Pre-Trade Check:
    - Executed BEFORE trade is processed
    - Checks if circuit breaker is already triggered
    - Returns -50 penalty if halted
    - Prevents continued trading in risk-limited state

LAYER 2 - Post-Trade Check:
    - Executed AFTER trade AND RiskManager update
    - Checks for NEW limit breaches from current trade
    - Applies -50 penalty if limits exceeded
    - Catches new drawdown/loss limit violations

Usage:
------
    from src.environment.config_integrated_env import ConfigIntegratedTradingEnv
    from src.environment.config_system import load_environment_config_from_yaml

    # Load configuration
    config = load_environment_config_from_yaml('config/environment/realistic_env.yaml')

    # Create environment
    env = ConfigIntegratedTradingEnv(price_data, features, config)

    # Run episode
    obs, info = env.reset()
    for _ in range(1000):
        action = policy(obs)  # Or env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break

Dependencies:
-------------
- gymnasium: RL environment interface
- numpy: Numerical operations
- pandas: Data handling
- loguru: Logging
- config_system: YAML configuration loader
- order_book: L2 market simulation
- slippage_model: Slippage calculations
- position_actions: Action mapping
- risk_manager: Risk management (Phase 4)
- risk_metrics_logger: Risk metrics tracking
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
from pathlib import Path
from loguru import logger

from src.environment.config_system import EnvironmentConfig, MarketRegime
from src.environment.order_book import OrderBookSimulator
from src.environment.slippage_model import SlippageModel, TransactionCostModel
from src.environment.position_actions import (
    PositionActionMapper,
    ActionConfig,
    POSITION_SIZES,
)

from src.risk.risk_manager import RiskManager, RiskConfig
from src.risk.risk_metrics_logger import RiskMetricsLogger


class ConfigIntegratedTradingEnv(gym.Env):
    """
    Fully config-integrated trading environment.

    This is the main environment class for the BITCOIN4Traders project.
    It provides a realistic RL trading environment with comprehensive
    configuration through YAML files.

    Features:
    ---------
    - Configurable action space (7 discrete positions)
    - Maker/taker fee differentiation
    - Dynamic reward from configurable components
    - Market regime simulation
    - Order book slippage modeling
    - Two-layer risk management
    - Kelly Criterion position sizing

    Attributes:
        config: EnvironmentConfig from YAML
        price_data: OHLCV price data
        features: Computed technical features
        orderbook_sim: OrderBookSimulator (if enabled)
        slippage_model: SlippageModel instance
        cost_model: EnhancedTransactionCostModel
        risk_manager: RiskManager for position sizing
        risk_metrics: RiskMetricsLogger for tracking
        position_mapper: PositionActionMapper for action conversion

    State Vector:
    -------------
    Total features = n_features + 9 additional

    Additional features:
    - position: Current position (-1 to 1)
    - portfolio_return: Return since episode start
    - cash_ratio: Cash / equity ratio
    - drawdown: Current drawdown
    - n_trades: Trade count
    - consecutive_losses: Loss streak
    - regime_vol_factor: Volatility regime factor
    - regime_volume_factor: Volume regime factor
    - progress: Episode progress (0 to 1)

    Example:
        >>> config = load_environment_config_from_yaml('config.yaml')
        >>> env = ConfigIntegratedTradingEnv(prices, features, config)
        >>> obs, info = env.reset()
    """

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        price_data: pd.DataFrame,
        features: pd.DataFrame,
        config: EnvironmentConfig,
    ):
        """
        Initialize with EnvironmentConfig from YAML.

        Args:
            price_data: OHLCV data with columns [open, high, low, close, volume]
            features: Computed technical indicators and features
            config: Complete EnvironmentConfig from YAML

        Raises:
            ValueError: If required columns missing from data
        """
        super().__init__()

        self.config = config
        self.price_data = price_data
        self.features = features

        # Align data - ensure price and features match
        common_index = price_data.index.intersection(features.index)
        self.price_data = price_data.loc[common_index]
        self.features = features.loc[common_index]

        logger.info(
            f"ConfigIntegratedTradingEnv initialized: {len(self.price_data)} steps"
        )
        logger.info(f"  Type: {config.type}")
        logger.info(f"  Maker Fee: {config.transaction_costs.maker_fee_bps} bps")
        logger.info(f"  Taker Fee: {config.transaction_costs.taker_fee_bps} bps")
        logger.info(f"  Reward components: {len(config.reward.components)}")

        # Initialize simulators (order book, slippage, costs)
        self._init_simulators()

        # Initialize Risk Management (Phase 4)
        self._init_risk_management()

        # Initialize spaces
        self._init_spaces()

        # Episode state
        self.reset()

    def _init_simulators(self):
        """
        Initialize order book and cost models from config.

        Creates instances of:
        - OrderBookSimulator: For L2 slippage (if enabled)
        - SlippageModel: For slippage calculation
        - EnhancedTransactionCostModel: For complete cost modeling
        """
        # Order book simulator
        if self.config.orderbook.enabled:
            self.orderbook_sim = OrderBookSimulator(self.config.orderbook)
        else:
            self.orderbook_sim = None

        # Slippage model
        self.slippage_model = SlippageModel(self.config.slippage)

        # Transaction costs (with maker/taker differentiation)
        self.cost_model = EnhancedTransactionCostModel(
            self.config.transaction_costs, self.slippage_model
        )

    def _init_risk_management(self):
        """
        Initialize Risk Management system (Phase 4).

        Creates RiskManager and RiskMetricsLogger for:
        - Position sizing validation
        - Kelly Criterion estimation
        - Circuit breaker logic
        - Drawdown tracking
        - Consecutive loss counting
        """
        # Create RiskConfig from EnvironmentConfig
        risk_config = RiskConfig(
            max_drawdown_per_session=self.config.max_drawdown,
            max_consecutive_losses=self.config.max_consecutive_losses,
            max_position_size=self.config.max_position_size,
            kelly_fraction=0.5,  # From transaction_costs or default
            enable_circuit_breaker=True,
        )

        # Initialize RiskManager
        self.risk_manager = RiskManager(
            config=risk_config, initial_capital=self.config.initial_capital
        )

        # Initialize RiskMetricsLogger
        self.risk_metrics = RiskMetricsLogger(lookback=50, risk_free_rate=0.0)

        logger.info("Risk Management initialized")
        logger.info(
            f"  Max drawdown: {risk_config.max_drawdown_per_session * 100:.1f}%"
        )
        logger.info(f"  Max position: {risk_config.max_position_size * 100:.0f}%")

    def _init_spaces(self):
        """
        Initialize observation and action spaces.

        Sets up:
        - observation_space: Box with feature + additional dimensions
        - action_space: Discrete(7) for position sizing
        - position_mapper: For action-to-position conversion
        """
        # Calculate state dimension
        n_features = len(self.features.columns)
        n_additional = 9  # Extended features
        state_dim = n_features + n_additional

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(state_dim,), dtype=np.float32
        )

        # Actions: 7 discrete position sizing actions
        # 0: Short 100%, 1: Short 50%, 2: Neutral, 3: Long 33%
        # 4: Long 50%, 5: Long 75%, 6: Long 100%
        self.action_space = spaces.Discrete(7)

        # Initialize position action mapper
        action_config = ActionConfig(
            use_kelly_override=False,  # Disabled - use discrete values
            kelly_fraction=0.5,
            min_position_size=0.0,
            max_position_size=self.config.max_position_size,
            strategy="discrete",
        )
        self.position_mapper = PositionActionMapper(action_config)

        logger.info(f"Observation space: {state_dim} features")
        logger.info(f"Action space: Discrete(7) with position sizing")
        logger.info(f"  Actions: Short100%(0), Short50%(1), Neutral(2), Long33%(3)")
        logger.info(f"           Long50%(4), Long75%(5), Long100%(6)")

    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        """
        Reset environment to start new episode.

        Resets all state variables and initializes random starting point
        within the data. Episode starts at random index to prevent
        overfitting to specific time periods.

        Args:
            seed: Random seed (optional)
            options: Additional options (optional)

        Returns:
            observation: Initial state vector
            info: Initial information dict
        """
        super().reset(seed=seed)

        # Random start – episode ends after max_steps steps
        # Ensure enough data for lookback window
        max_start = (
            len(self.price_data) - self.config.max_steps - self.config.lookback_window
        )
        self.current_step = np.random.randint(
            self.config.lookback_window, max(self.config.lookback_window + 1, max_start)
        )
        self._episode_start_step = self.current_step

        # Reset state
        self.position = 0.0
        self.cash = self.config.initial_capital
        self.shares = 0.0
        self.equity_history = [self.config.initial_capital]
        self.trade_history = []

        # Reset Risk Management
        self.risk_manager.reset()
        self.risk_metrics.reset()

        # Market regime (sampled for episode)
        self.current_regime = self._sample_market_regime()

        obs = self._get_observation()
        info = self._get_info()

        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one step with config-driven behavior and risk management.

        This is the main interaction method. It implements a two-layer
        risk management system:

        Risk Management Flow (Two-Layer Protection):
        ===========================================

        LAYER 1 - Pre-Trade Check (lines ~257-272):
        ---------------------------------------------
        Purpose:    "Can we even trade?"
        When:       BEFORE executing the trade
        Checks:     RiskManager state BEFORE update_state()
        Action:     Immediate termination with -50 penalty if halted
        Reason:     Circuit breaker was already triggered from previous steps
                    Prevents continued trading when already in risk-limited state

        LAYER 2 - Post-Trade Check (lines ~309-319):
        ----------------------------------------------
        Purpose:    "Did this trade breach any limits?"
        When:       AFTER executing trade AND updating RiskManager state
        Checks:     RiskManager state AFTER update_state()
        Action:     Termination + -50 penalty if limits exceeded
        Reason:     This is where NEW drawdown/loss limits are triggered

        Why Both Layers?
        ================
        - Pre-Trade:  Prevents continued trading when ALREADY halted
        - Post-Trade: Catches NEW limit breaches from the CURRENT trade
        - Together:   Complete protection against all risk scenarios
        - Penalty:    Consistent -50 for both (unified reinforcement)

        Parameters:
        -----------
        action : int
            Action from agent (0-6 for position sizing)

        Returns:
        --------
        observation : np.ndarray
            Current market state (feature vector)
        reward : float
            Trading reward (includes risk penalties)
        terminated : bool
            Episode ended (risk limits reached or end of data)
        truncated : bool
            Episode truncated (max steps reached)
        info : dict
            Additional information (position, equity, risk_metrics, etc.)

        Example:
            >>> obs, reward, terminated, truncated, info = env.step(4)
            >>> print(f"Reward: {reward:.2f}, Position: {info['position']}")
        """
        # Store old equity for reward calculation
        old_equity = self._calculate_equity()

        # ============================================================
        # LAYER 1: PRE-TRADE CIRCUIT BREAKER CHECK
        # Purpose: Abort if already halted from previous step
        # ============================================================
        if self.risk_manager.should_halt_trading():
            terminated = True
            truncated = False
            reward = -50.0  # Consistent penalty

            obs = self._get_observation()
            info = self._get_info()
            info["circuit_breaker"] = True
            info["halt_reason"] = self.risk_manager.get_halt_reason()

            logger.critical(f"CIRCUIT BREAKER TRIGGERED: {info['halt_reason']}")

            return obs, reward, terminated, truncated, info

        # Execute trade with risk management validation
        trade_info = self._execute_trade_enhanced(action)

        # Move to next step
        self.current_step += 1

        # Check episode end: end of data OR max_steps reached
        steps_in_episode = self.current_step - self._episode_start_step
        if self.current_step >= len(self.price_data) - 1:
            terminated = True
            truncated = False
        elif steps_in_episode >= self.config.max_steps:
            terminated = False
            truncated = True  # Time-limited episode
        else:
            terminated = False
            truncated = False

        # Calculate reward (dynamic from config)
        reward = self._calculate_reward_dynamic(old_equity, trade_info)

        # Update equity
        current_equity = self._calculate_equity()
        self.equity_history.append(current_equity)

        # Update Risk Management
        trade_pnl = trade_info.get("pnl", 0.0)
        self.risk_manager.update_state(current_equity, trade_pnl)
        self.risk_metrics.update(
            equity=current_equity,
            trade_result=trade_pnl if trade_info.get("trade_executed") else None,
            kelly_fraction=trade_info.get("kelly_fraction", 0.0),
        )

        # ============================================================
        # LAYER 2: POST-TRADE RISK LIMIT CHECK
        # Purpose: Catch NEW limit breaches from this trade
        # ============================================================
        if self.risk_manager.should_halt_trading():
            terminated = True
            halt_reason = self.risk_manager.get_halt_reason()
            logger.warning(f"Risk limits reached: {halt_reason}")

            # Consistent penalty (unified with Layer 1)
            reward -= 50.0

        obs = self._get_observation()
        info = self._get_info()
        info.update(trade_info)

        # Add risk metrics to info
        info["risk_metrics"] = self.risk_metrics.get_current_metrics()

        return obs, reward, terminated, truncated, info

    def _execute_trade_enhanced(self, action: int) -> Dict:
        """
        Execute trade with position sizing and maker/taker differentiation.

        Maps discrete action to position size, calculates costs with
        realistic fee modeling, and executes the trade.

        Actions:
            0: Short 100%, 1: Short 50%, 2: Neutral, 3: Long 33%
            4: Long 50%, 5: Long 75%, 6: Long 100%

        Features:
        - Kelly parameter estimation for optimal sizing
        - Risk Manager validation
        - Order type determination (maker vs taker)
        - Order book slippage (if enabled)
        - Complete cost calculation

        Returns:
        --------
        trade_info : dict
            Contains trade details, costs, order type, execution price
        """
        # Get Kelly parameters for optimal position sizing
        kelly_params = self.risk_manager.kelly.estimate_parameters(
            self.risk_manager.trade_history[-20:]
            if len(self.risk_manager.trade_history) >= 5
            else []
        )

        # Map action to position size using predefined values
        target_position = POSITION_SIZES[action]

        # Skip if already at target position
        if abs(target_position - self.position) < 0.01:
            return {"trade_executed": False, "cost": 0.0, "order_type": "none"}

        # Get current market data
        current_price = self.price_data.iloc[self.current_step]["close"]
        current_volume = self.price_data.iloc[self.current_step]["volume"]
        volatility = self.features.iloc[self.current_step].get("volatility_20", 0.02)

        # Apply market regime: scale volatility and volume
        regime = self.current_regime
        volatility *= regime.volatility / 0.02
        current_volume *= regime.volume / 500.0

        # Calculate position change
        position_change = target_position - self.position
        current_equity = self._calculate_equity()

        # Calculate position value based on target size
        position_value = current_equity * abs(target_position)
        shares_to_trade = position_value / current_price

        # PHASE 4: Validate with RiskManager
        win_prob = kelly_params.win_probability if kelly_params else None
        win_loss_ratio = kelly_params.win_loss_ratio if kelly_params else None

        approved, adjusted_position_value = self.risk_manager.validate_position_size(
            proposed_size=position_value,
            current_capital=current_equity,
            win_probability=win_prob,
            win_loss_ratio=win_loss_ratio,
        )

        if not approved:
            return {
                "trade_executed": False,
                "cost": 0.0,
                "order_type": "rejected",
                "pnl": 0.0,
                "kelly_fraction": 0.0,
                "rejection_reason": "Risk Manager rejected trade",
            }

        # Apply Risk Manager adjustments
        if adjusted_position_value < position_value:
            logger.info(
                f"Position adjusted: ${position_value:.0f} -> ${adjusted_position_value:.0f} (Risk Manager)"
            )
            position_value = adjusted_position_value
            # Recalculate target position based on adjusted value
            adjusted_target = np.sign(target_position) * (
                adjusted_position_value / current_equity
            )
            target_position = adjusted_target
            position_change = target_position - self.position

        # Calculate Kelly fraction for tracking
        kelly_fraction = (
            kelly_params.kelly_fraction
            if kelly_params
            else (position_value / current_equity if current_equity > 0 else 0.0)
        )

        # Determine side
        if position_change > 0:
            side = "buy"
        else:
            side = "sell"
            shares_to_trade = abs(shares_to_trade)

        # Determine order type (maker vs taker)
        # Small orders use limit (maker), large orders use market (taker)
        participation_rate = (
            shares_to_trade * current_price / (current_volume * current_price + 1e-8)
        )

        if participation_rate < 0.01:  # < 1% of volume = maker
            order_type = "maker"
            fee_bps = self.config.transaction_costs.maker_fee_bps
        else:  # Large order = taker
            order_type = "taker"
            fee_bps = self.config.transaction_costs.taker_fee_bps

        # Generate order book if enabled
        if self.config.orderbook.enabled and self.orderbook_sim:
            bid_prices, bid_volumes, ask_prices, ask_volumes = (
                self.orderbook_sim.generate_order_book(
                    current_price, volatility, current_volume
                )
            )

            costs = self.cost_model.calculate_total_cost_enhanced(
                side=side,
                quantity=shares_to_trade,
                price=current_price,
                order_type=order_type,
                volume=current_volume,
                volatility=volatility,
                bid_prices=bid_prices,
                bid_volumes=bid_volumes,
                ask_prices=ask_prices,
                ask_volumes=ask_volumes,
            )
        else:
            costs = self.cost_model.calculate_total_cost_enhanced(
                side=side,
                quantity=shares_to_trade,
                price=current_price,
                order_type=order_type,
                volume=current_volume,
                volatility=volatility,
            )

        execution_price = costs["execution_price"]
        total_cost = costs["total_cost_dollars"]

        # Execute trade
        old_position_value = 0
        if self.position != 0:
            old_position_value = self.shares * current_price
            self.cash += old_position_value
            self.shares = 0

        if target_position != 0:
            self.shares = shares_to_trade
            trade_value = shares_to_trade * execution_price
            self.cash -= trade_value

        self.position = target_position

        # Calculate Trade PnL
        # PnL = Change in position value - Transaction costs
        new_position_value = self.shares * current_price if target_position != 0 else 0
        position_change_value = new_position_value - old_position_value
        pnl = position_change_value - total_cost

        # Record trade
        trade_info = {
            "trade_executed": True,
            "action": action,
            "side": side,
            "order_type": order_type,
            "shares": shares_to_trade,
            "price": current_price,
            "execution_price": execution_price,
            "cost": total_cost,
            "fee_bps": fee_bps,
            "slippage_bps": costs.get("slippage_bps", 0),
            "pnl": pnl,
            "kelly_fraction": kelly_fraction,
        }

        self.trade_history.append(trade_info)

        return trade_info

    def _calculate_reward_dynamic(self, old_equity: float, trade_info: Dict) -> float:
        """
        Calculate reward dynamically from config components.

        Uses reward.components from YAML to build the reward signal.
        This allows customization of reward shaping without code changes.

        Available Components:
        - 'return': Portfolio return (PnL / equity)
        - 'sharpe': Sharpe ratio bonus
        - 'drawdown': Drawdown penalty
        - 'transaction_cost': Cost penalty

        Args:
            old_equity: Equity before step
            trade_info: Trade execution details

        Returns:
            reward: Combined reward from all components
        """
        current_equity = self._calculate_equity()
        components_values = {}

        for comp in self.config.reward.components:
            if comp.name == "return":
                # Portfolio return
                pnl = current_equity - old_equity
                pnl_pct = pnl / old_equity if old_equity > 0 else 0.0
                components_values["return"] = pnl_pct * comp.weight

            elif comp.name == "sharpe":
                # Sharpe bonus
                lookback = comp.lookback or 20
                if len(self.equity_history) > lookback:
                    returns = np.diff(self.equity_history[-lookback:]) / np.array(
                        self.equity_history[-lookback:-1]
                    )
                    sharpe = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252)
                    components_values["sharpe"] = np.clip(
                        sharpe * comp.weight,
                        -0.5,
                        0.5,
                    )
                else:
                    components_values["sharpe"] = 0.0

            elif comp.name == "drawdown":
                # Drawdown penalty
                drawdown = self._calculate_drawdown()
                components_values["drawdown"] = drawdown * comp.weight

            elif comp.name == "transaction_cost":
                # Cost penalty
                cost_penalty = (
                    trade_info.get("cost", 0.0) / old_equity if old_equity > 0 else 0.0
                )
                components_values["transaction_cost"] = -cost_penalty * comp.weight

        # Sum all components
        reward = sum(components_values.values())

        # Scale and clip reward
        reward = np.clip(
            reward * self.config.reward.scale,
            self.config.reward.clip_min,
            self.config.reward.clip_max,
        )

        return reward

    def _sample_market_regime(self) -> MarketRegime:
        """
        Sample market regime for episode.

        Market regimes define different volatility and volume conditions
        to simulate various market environments. This adds realism and
        helps train more robust strategies.

        Returns:
            regime: MarketRegime with volatility, volume, spread
        """
        regime_names = list(self.config.market.vol_regimes.keys())

        if not regime_names:
            # Default if no regimes defined
            return MarketRegime("normal", 0.02, 500.0, 5.0)

        # Sample regime (can be made more sophisticated)
        regime_name = np.random.choice(regime_names)
        return self.config.market.get_regime(regime_name)

    def _calculate_equity(self) -> float:
        """
        Calculate current portfolio value.

        Equity = Cash + Position Value

        Returns:
            equity: Total portfolio value
        """
        current_price = self.price_data.iloc[self.current_step]["close"]
        position_value = self.shares * current_price
        return self.cash + position_value

    def _calculate_drawdown(self) -> float:
        """
        Calculate current drawdown from peak equity.

        Drawdown = (Current Equity - Peak Equity) / Peak Equity

        Returns:
            drawdown: Current drawdown (negative = below peak)
        """
        if len(self.equity_history) < 2:
            return 0.0

        equity_array = np.array(self.equity_history)
        running_max = np.maximum.accumulate(equity_array)
        drawdown = (equity_array[-1] - running_max[-1]) / running_max[-1]

        return drawdown

    def _get_observation(self) -> np.ndarray:
        """
        Construct observation vector.

        Combines feature data with portfolio state, risk metrics,
        and regime information.

        Returns:
            obs: State vector (float32)
        """
        # Features from Phase 1
        feature_values = self.features.iloc[self.current_step].values

        # Portfolio state
        current_equity = self._calculate_equity()
        portfolio_return = (
            current_equity - self.config.initial_capital
        ) / self.config.initial_capital
        cash_ratio = self.cash / current_equity if current_equity > 0 else 1.0
        drawdown = self._calculate_drawdown()

        # Market regime factors
        regime_vol_factor = self.current_regime.volatility / 0.02
        regime_volume_factor = self.current_regime.volume / 500.0

        # Additional features
        additional = np.array(
            [
                self.position,
                portfolio_return,
                cash_ratio,
                drawdown,
                len(self.trade_history),
                self.risk_manager.consecutive_losses,
                regime_vol_factor,
                regime_volume_factor,
                float(self.current_step) / len(self.price_data),
            ]
        )

        obs = np.concatenate([feature_values, additional])
        obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)

        return obs.astype(np.float32)

    def _get_info(self) -> Dict:
        """
        Get info dict with current state.

        Returns:
            info: Dict with step, price, position, equity, etc.
        """
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
            "regime": self.current_regime.name,
        }

        return info

    def render(self, mode="human"):
        """
        Render environment state.

        Args:
            mode: Rendering mode ('human' supported)
        """
        if mode == "human":
            info = self._get_info()
            print(f"\nStep {info['step']}:")
            print(f"  Price: ${info['price']:.2f}")
            print(f"  Position: {info['position']:.0f}")
            print(f"  Equity: ${info['equity']:.2f}")
            print(f"  Return: {info['return'] * 100:.2f}%")
            print(f"  Regime: {info['regime']}")


class EnhancedTransactionCostModel:
    """
    Enhanced transaction cost model with maker/taker differentiation.

    This model provides complete cost breakdown including:
    - Maker fees (for limit orders)
    - Taker fees (for market orders)
    - Slippage (various models)

    The key improvement is distinguishing between maker and taker
    orders, which have significantly different fees on most exchanges.

    Attributes:
        config: TransactionCostConfig with fee parameters
        slippage_model: SlippageModel instance

    Example:
        >>> cost_model = EnhancedTransactionCostModel(config, slippage_model)
        >>> costs = cost_model.calculate_total_cost_enhanced(
        ...     side='buy', quantity=1.0, price=50000.0,
        ...     order_type='taker', volume=100.0
        ... )
    """

    def __init__(self, config, slippage_model):
        self.config = config
        self.slippage_model = slippage_model

    def calculate_total_cost_enhanced(
        self,
        side: str,
        quantity: float,
        price: float,
        order_type: str,  # 'maker' or 'taker'
        **kwargs,
    ) -> Dict:
        """
        Calculate costs with maker/taker differentiation.

        Args:
            side: 'buy' or 'sell'
            quantity: Order size
            price: Reference price
            order_type: 'maker' (limit) or 'taker' (market)
            **kwargs: Additional parameters for slippage

        Returns:
            costs: Dict with execution_price, fee_bps, slippage_bps, etc.
        """
        # Select appropriate fee based on order type
        if order_type == "maker":
            fee_bps = self.config.maker_fee_bps
        else:  # taker
            fee_bps = self.config.taker_fee_bps

        # Calculate fee in dollars
        fee_dollars = price * quantity * (fee_bps / 10000)

        # Calculate slippage (if enabled)
        if self.config.include_slippage:
            execution_price, slippage_bps = self.slippage_model.calculate_slippage(
                side, quantity, price, **kwargs
            )
        else:
            execution_price = price
            slippage_bps = 0.0

        # Total cost in bps
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
            "order_type": order_type,
        }


if __name__ == "__main__":
    print("=" * 80)
    print("CONFIG-INTEGRATED ENVIRONMENT TEST")
    print("=" * 80)

    # Load config
    from environment.config_system import load_environment_config_from_yaml

    config_path = (
        Path(__file__).parent.parent.parent
        / "config"
        / "environment"
        / "realistic_env.yaml"
    )

    if config_path.exists():
        config = load_environment_config_from_yaml(str(config_path))
        print("\n✓ Config loaded")
        print(f"  Maker fee: {config.transaction_costs.maker_fee_bps} bps")
        print(f"  Taker fee: {config.transaction_costs.taker_fee_bps} bps")
        print(f"  Reward components: {len(config.reward.components)}")
    else:
        print("⚠ Using default config")
        config = EnvironmentConfig()

    # Generate test data
    np.random.seed(42)
    n_points = 1000
    dates = pd.date_range("2023-01-01", periods=n_points, freq="1H")

    close = 50000 + np.cumsum(np.random.randn(n_points) * 100)
    price_data = pd.DataFrame(
        {
            "open": close + np.random.randn(n_points) * 50,
            "high": close + abs(np.random.randn(n_points) * 100),
            "low": close - abs(np.random.randn(n_points) * 100),
            "close": close,
            "volume": np.random.uniform(100, 1000, n_points),
        },
        index=dates,
    )

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
    env = ConfigIntegratedTradingEnv(price_data, features, config)

    print(f"\n✓ Environment created")

    # Test episode
    obs, info = env.reset()
    print(f"\n✓ Reset successful")
    print(f"  Regime: {info['regime']}")

    total_reward = 0
    for i in range(50):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if i % 10 == 0:
            env.render()

        if terminated or truncated:
            break

    print(f"\n✓ Test complete")
    print(f"  Total reward: {total_reward:.2f}")
    print(f"  Final return: {info['return'] * 100:.2f}%")

    print("\n" + "=" * 80)
    print("✓ CONFIG-INTEGRATED ENVIRONMENT TEST PASSED")
    print("=" * 80)
