"""
Portfolio Allocation Environment for Reinforcement Learning
==========================================================
Multi-asset portfolio optimization using reinforcement learning.

This module provides OpenAI Gym-compatible environments for training RL agents
to optimize portfolio allocation across multiple assets. It models realistic
trading conditions including transaction costs, position limits, and risk-adjusted
rewards.

Features:
  - Portfolio Weight Allocation: Continuous action space for optimal asset distribution
  - Covariance Matrix Features: State includes risk-relevant market features
  - Transaction Cost Modeling: Realistic slippage and commission modeling
  - Rebalancing Logic: Configurable rebalancing frequency
  - Risk-Adjusted Rewards: Sharpe ratio and other risk metrics for training
  - Multi-Stock Support: Handles portfolios of 30+ assets (e.g., Dow 30)

Environment Types:
  1. PortfolioAllocationEnv: Continuous weight allocation (softmax normalized)
  2. MultiStockTradingEnv: Discrete actions per stock (buy/sell/hold)

State Space Components:
  - Covariance matrix (stock_dim × stock_dim) - flattened
  - Technical indicators (MACD, RSI, CCI, ADX) per stock
  - Current portfolio weights
  - Cash position

Action Space:
  - Continuous weights [0, 1] normalized via softmax to sum to 1
  - Alternative: Discrete {hold, buy, sell} for each stock

Reward Functions:
  - Portfolio return
  - Risk-adjusted return (Sharpe-like ratio)
  - Customizable combinations

Usage:
    from src.portfolio.portfolio_env import PortfolioAllocationEnv, PortfolioEnvConfig

    config = PortfolioEnvConfig(
        initial_capital=100000,
        stock_dim=30,
        transaction_cost_pct=0.001,
        rebalance_window=63
    )

    env = PortfolioAllocationEnv(df, config)
    state, info = env.reset()

    for _ in range(1000):
        action = agent.predict(state)
        state, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            state, info = env.reset()

References:
  - "Deep Reinforcement Learning for Trading" by Jiang et al.
  - "Portfolio Allocation with RL" by Moody & Saffell
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class PortfolioEnvConfig:
    """
    Configuration for portfolio environment.

    Defines all hyperparameters and settings for the portfolio allocation
    environment including capital, transaction costs, and feature selection.

    Attributes:
        initial_capital: Starting capital in USD. Default is 100,000.
        stock_dim: Number of stocks in portfolio. Default is 30 (Dow 30).
        transaction_cost_pct: Transaction cost as percentage (0.001 = 0.1%).
                            Applied to both buys and sells.
        rebalance_window: Days between rebalancing decisions. Default is 63 (~3 months).
        validation_window: Days for validation. Default is 63.
        tech_indicator_list: List of technical indicators to include in state.
                           Default includes MACD, RSI, CCI, ADX.
        risk_free_rate: Annual risk-free rate for Sharpe calculation. Default 0.03 (3%).

    Example:
        # Aggressive trading config
        config = PortfolioEnvConfig(
            initial_capital=1_000_000,
            stock_dim=50,
            transaction_cost_pct=0.0005,  # 0.05% - low cost
            rebalance_window=21,  # Monthly rebalancing
            risk_free_rate=0.02
        )

        # Conservative config
        config = PortfolioEnvConfig(
            initial_capital=10_000,
            stock_dim=10,
            transaction_cost_pct=0.002,  # 0.2% - high cost
            rebalance_window=252,  # Yearly rebalancing
            risk_free_rate=0.04
        )
    """

    # Portfolio settings
    initial_capital: float = 100000.0
    stock_dim: int = 30  # Number of stocks (e.g., Dow 30)

    # Transaction costs
    transaction_cost_pct: float = 0.001  # 0.1%

    # Rebalancing
    rebalance_window: int = 63  # Days between rebalancing
    validation_window: int = 63  # Days for validation

    # Features
    tech_indicator_list: List[str] = None

    # Risk
    risk_free_rate: float = 0.03  # Annual risk-free rate

    def __post_init__(self):
        if self.tech_indicator_list is None:
            self.tech_indicator_list = [
                "macd",
                "rsi",
                "cci",
                "adx",
            ]  # Default indicator set


class PortfolioAllocationEnv(gym.Env):
    """
    Gymnasium-compatible portfolio allocation environment for DRL.

    This environment models a multi-asset portfolio where an RL agent learns
    to allocate capital across multiple stocks to maximize risk-adjusted returns.

    State Space (Observation):
        The state is a concatenated vector containing:
        - Flattened covariance matrix (stock_dim × stock_dim)
        - Technical indicators for each stock (macd, rsi, cci, adx)
        - Current portfolio weights
        - Cash position

        Total dimension: stock_dim² + stock_dim × num_indicators + stock_dim

    Action Space (Continuous):
        Continuous values [0, 1] for each stock representing raw weights.
        Actions are normalized via softmax to ensure valid probability distribution
        (weights sum to 1). This allows the agent to express preferences without
        worrying about normalization.

    Reward Function:
        Configurable. Default is a Sharpe-like ratio:
        - Uses rolling 20-step window for volatility estimation
        - Returns portfolio_return / (volatility + ε) during warm-up
        - Returns simple scaled return during initial episodes

    Episode Termination:
        - Episode ends when all data has been processed (day >= len(df))
        - Or when portfolio value drops to zero

    Key Methods:
        - reset(): Initialize environment to starting state
        - step(action): Execute one trading period, return new state and reward
        - render(): Print current portfolio state
        - get_portfolio_performance(): Calculate final performance metrics

    Example:
        # Create environment
        config = PortfolioEnvConfig(initial_capital=100000, stock_dim=30)
        env = PortfolioAllocationEnv(df, config)

        # Training loop
        state, info = env.reset()
        total_reward = 0

        for step in range(1000):
            action = agent.select_action(state)  # Get agent's allocation
            state, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            if terminated or truncated:
                # Calculate performance
                perf = env.get_portfolio_performance()
                print(f"Sharpe: {perf['sharpe_ratio']:.2f}")
                state, info = env.reset()
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, df: pd.DataFrame, config: PortfolioEnvConfig, day: int = 0):
        """
        Initialize portfolio environment.

        Args:
            df: DataFrame with stock data
            config: Environment configuration
            day: Starting day index
        """
        super().__init__()

        self.df = df
        self.config = config
        self.day = day
        self.stock_dim = config.stock_dim

        # Action space: portfolio weights for each stock
        # Continuous [0, 1], will be normalized to sum to 1
        self.action_space = spaces.Box(
            low=0, high=1, shape=(self.stock_dim,), dtype=np.float32
        )

        # State space
        # Covariance matrix + technical indicators + weights
        n_features = (
            self.stock_dim * self.stock_dim  # Covariance matrix (flattened)
            + self.stock_dim * len(config.tech_indicator_list)  # Per-stock indicators
            + self.stock_dim  # Current weights
        )

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(n_features,), dtype=np.float32
        )

        # Initialize state
        self.data = self.df.loc[self.day, :]  # Current day's row
        self.covs = None  # Will hold covariance matrix
        self.state = None  # Will hold observation vector
        self.portfolio_value = config.initial_capital
        self.portfolio_values = [config.initial_capital]  # History of portfolio values
        self.asset_memory = [config.initial_capital]  # Tracks equity curve
        self.rewards_memory = []  # History of rewards
        self.actions_memory = []  # History of weight allocations
        self.date_memory = []  # History of dates

        # Terminal state flag
        self.terminal = False

        # Risk-free rate per day (convert annual to daily using compound formula)
        self.daily_risk_free = (1 + config.risk_free_rate) ** (1 / 252) - 1

        # Transaction cost
        self.transaction_cost_pct = config.transaction_cost_pct

    def _calculate_covariance(self, window: int = 60) -> np.ndarray:
        """
        Calculate covariance matrix of historical returns.

        Computes the covariance matrix from historical stock returns over
        a rolling window. The matrix is used as a risk feature in the state
        to help the agent understand asset correlations.

        Args:
            window: Number of historical days to calculate covariance over.
                   Default is 60 days. If less data available, uses what's available.

        Returns:
            Stock_dim × stock_dim numpy array representing covariance matrix.
            Small identity matrix added to ensure positive definiteness.
        """
        if self.day < window:
            # Use available data if less than window
            window = self.day + 1

        # Get past returns
        returns = []
        for i in range(window):
            day_idx = self.day - i  # Walk backward through history
            if day_idx >= 0:
                day_data = self.df.loc[day_idx, :]
                # Calculate returns for each stock
                day_returns = []
                for j in range(self.stock_dim):
                    if day_idx > 0:
                        prev_close = self.df.loc[day_idx - 1, f"close_{j}"]
                        curr_close = day_data[f"close_{j}"]
                        ret = (curr_close - prev_close) / prev_close  # Simple return
                    else:
                        ret = 0.0  # No return for first day
                    day_returns.append(ret)
                returns.append(day_returns)

        returns = np.array(returns)
        cov_matrix = np.cov(returns.T)  # Covariance across stocks (columns)

        # Ensure positive definite by adding small identity matrix
        cov_matrix += np.eye(self.stock_dim) * 1e-6

        return cov_matrix

    def _get_state(self) -> np.ndarray:
        """
        Construct current state vector for the agent.

        Builds the observation state by concatenating:
        1. Flattened covariance matrix (risk features)
        2. Technical indicators per stock
        3. Current portfolio weights

        Returns:
            numpy array of shape (n_features,) containing all state features
        """
        # Calculate covariance matrix
        cov_matrix = self._calculate_covariance()
        self.covs = cov_matrix

        # Flatten covariance matrix into 1D vector
        cov_flat = cov_matrix.flatten()

        # Get technical indicators for all stocks
        tech_features = []
        for i in range(self.stock_dim):
            for indicator in self.config.tech_indicator_list:
                tech_features.append(self.data[f"{indicator}_{i}"])  # e.g., "macd_0"

        # Current portfolio weights (if any)
        if len(self.actions_memory) > 0:
            current_weights = self.actions_memory[-1]  # Last allocation
        else:
            current_weights = np.zeros(self.stock_dim)  # Start with flat/empty weights

        # Combine all features into a single observation vector
        state = np.concatenate([cov_flat, np.array(tech_features), current_weights])

        return state.astype(np.float32)

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        """
        Reset environment to initial state.

        Initializes the environment for a new episode, resetting portfolio value,
        historical tracking, and day counter to their initial values.

        Args:
            seed: Random seed for reproducibility (passed to gym.Env)
            options: Optional dictionary with 'initial_capital' override

        Returns:
            Initial state observation and info dict
        """
        super().reset(seed=seed)

        self.day = 0
        self.data = self.df.loc[self.day, :]
        self.covs = self._calculate_covariance()
        self.state = self._get_state()

        if options and "initial_capital" in options:
            self.portfolio_value = options["initial_capital"]
        else:
            self.portfolio_value = self.config.initial_capital

        self.portfolio_values = [self.portfolio_value]
        self.asset_memory = [self.portfolio_value]
        self.rewards_memory = []
        self.actions_memory = []
        self.date_memory = [self._get_date()]
        self.terminal = False

        return self.state, {}

    def _get_date(self):
        """Get current date from data."""
        if "date" in self.data.index:
            return self.data["date"]
        return self.day  # Fall back to day index if no date column

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one trading step.

        Processes the agent's portfolio allocation action, advances time by one day,
        calculates returns and transaction costs, and returns the new state.

        Args:
            action: Array of raw weights (one per stock), will be normalized
                   via softmax to sum to 1.0

        Returns:
            Tuple containing:
            - state: New observation after action
            - reward: Calculated reward (risk-adjusted return)
            - terminated: True if episode complete (no more data)
            - truncated: False (not used in this env)
            - info: Dict with portfolio_value, return, cost, sharpe, weights

        Process:
            1. Normalize action via softmax to get valid weight distribution
            2. Calculate individual stock returns for the day
            3. Calculate portfolio return as weighted sum
            4. Subtract transaction costs based on turnover (weight changes)
            5. Update portfolio value
            6. Calculate reward (Sharpe-like during warmup, return scaled)
            7. Check termination condition
            8. Return new state and info
        """
        # Normalize action to sum to 1 using softmax (ensures valid probability distribution)
        action = np.exp(action) / np.sum(np.exp(action))
        self.actions_memory.append(action)

        # Store previous portfolio value
        prev_portfolio_value = self.portfolio_value

        # Move to next day
        self.day += 1
        self.data = self.df.loc[self.day, :]
        self.date_memory.append(self._get_date())

        # Calculate individual stock returns for this day
        individual_returns = []
        for i in range(self.stock_dim):
            prev_close = self.df.loc[self.day - 1, f"close_{i}"]
            curr_close = self.data[f"close_{i}"]
            ret = (curr_close - prev_close) / prev_close  # Daily return
            individual_returns.append(ret)

        individual_returns = np.array(individual_returns)

        # Portfolio return (weighted sum of individual returns)
        portfolio_return = np.sum(action * individual_returns)

        # Calculate transaction costs (if rebalancing)
        if len(self.actions_memory) >= 2:
            prev_weights = self.actions_memory[-2]
            turnover = np.sum(np.abs(action - prev_weights))  # Total weight shift
            transaction_cost = turnover * self.transaction_cost_pct
        else:
            transaction_cost = 0.0  # No cost on first allocation

        # Net portfolio return after transaction costs
        net_return = portfolio_return - transaction_cost

        # Update portfolio value (compound)
        self.portfolio_value = self.portfolio_value * (1 + net_return)
        self.portfolio_values.append(self.portfolio_value)
        self.asset_memory.append(self.portfolio_value)

        # Calculate reward (can be customized)
        reward = self._calculate_reward(net_return)
        self.rewards_memory.append(reward)

        # Check terminal state
        self.terminal = self.day >= len(self.df) - 1

        # Get new state
        self.state = self._get_state()

        # Info dict
        info = {
            "portfolio_value": self.portfolio_value,
            "portfolio_return": net_return,
            "transaction_cost": transaction_cost,
            "sharpe": self._calculate_sharpe(),
            "weights": action,
        }

        return self.state, reward, self.terminal, False, info

    def _calculate_reward(self, portfolio_return: float) -> float:
        """
        Calculate reward.

        Can be customized to use:
        - Simple return
        - Risk-adjusted return (Sharpe)
        - Differential Sharpe ratio
        """
        # Simple return
        # return portfolio_return * 100  # Scale up

        # Risk-adjusted return (approximation)
        if len(self.rewards_memory) > 20:
            recent_returns = np.array(
                self.rewards_memory[-20:]
            )  # Rolling 20-step window
            volatility = np.std(recent_returns) + 1e-6  # Add epsilon to avoid /0
            sharpe_like = portfolio_return / volatility  # Sharpe-like ratio
            return sharpe_like

        return portfolio_return * 100  # Scale up during warm-up period

    def _calculate_sharpe(self) -> float:
        """Calculate Sharpe ratio."""
        if len(self.rewards_memory) < 2:
            return 0.0

        returns = np.array(self.rewards_memory)
        excess_returns = returns - self.daily_risk_free  # Subtract risk-free benchmark

        if np.std(returns) == 0:
            return 0.0  # Avoid division by zero when returns are constant

        sharpe = np.mean(excess_returns) / (np.std(returns) + 1e-6)
        return sharpe * np.sqrt(252)  # Annualize by scaling with sqrt of trading days

    def render(self, mode="human"):
        """Render environment state."""
        if mode == "human":
            print(f"\nDay: {self.day}")
            print(f"Portfolio Value: ${self.portfolio_value:,.2f}")
            if len(self.actions_memory) > 0:
                weights = self.actions_memory[-1]
                print(f"Weights: {weights}")
            print(f"Sharpe: {self._calculate_sharpe():.2f}")

    def get_portfolio_performance(self) -> Dict:
        """Get final portfolio performance metrics."""
        portfolio_returns = np.array(self.rewards_memory)

        # Total return
        total_return = (
            self.portfolio_value - self.config.initial_capital
        ) / self.config.initial_capital

        # Annualized return
        n_days = len(self.rewards_memory)
        annual_return = (1 + total_return) ** (252 / n_days) - 1 if n_days > 0 else 0.0

        # Volatility (annualized)
        volatility = (
            np.std(portfolio_returns) * np.sqrt(252)
            if len(portfolio_returns) > 0
            else 0.0
        )

        # Sharpe ratio
        sharpe = self._calculate_sharpe()

        # Max drawdown
        portfolio_values = np.array(self.portfolio_values)
        peak = np.maximum.accumulate(portfolio_values)  # Running peak
        drawdown = (portfolio_values - peak) / peak  # Drawdown from peak
        max_drawdown = np.min(drawdown)  # Worst drawdown

        return {
            "total_return": total_return,
            "annual_return": annual_return,
            "volatility": volatility,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_drawdown,
            "final_value": self.portfolio_value,
        }


class MultiStockTradingEnv(PortfolioAllocationEnv):
    """
    Multi-stock trading environment with discrete actions.

    This environment extends PortfolioAllocationEnv with a discrete action space
    where the agent chooses one of three actions per stock:
    - HOLD (0): Do nothing for this stock
    - BUY (1): Buy max possible shares (up to hmax limit)
    - SELL (2): Sell all shares of this stock

    Key Differences from PortfolioAllocationEnv:
        - Discrete MultiDiscrete action space instead of continuous Box
        - Tracks actual share holdings and cash balance
        - Simulates realistic order execution with fills at current price
        - Transaction costs applied to each trade

    State Space:
        - Technical indicators per stock
        - Current stock prices
        - Current holdings (number of shares per stock)
        - Cash balance

    Action Space:
        - MultiDiscrete([3] × stock_dim)
        - Each dimension: 0=hold, 1=buy, 2=sell

    Example:
        config = PortfolioEnvConfig(initial_capital=100000, stock_dim=10)
        env = MultiStockTradingEnv(df, config, hmax=100)

        # Action: buy BTC, hold ETH, sell all AAPL
        action = np.array([1, 0, 2, 0, 0, 0, 0, 0, 0, 0])
        state, reward, terminated, truncated, info = env.step(action)
    """

    def __init__(
        self,
        df: pd.DataFrame,
        config: PortfolioEnvConfig,
        day: int = 0,
        hmax: int = 100,  # Max shares per trade
    ):
        super().__init__(df, config, day)

        self.hmax = hmax  # Maximum shares allowed per single trade

        # Discrete action space: 3 actions per stock (buy, sell, hold)
        self.action_space = spaces.MultiDiscrete([3] * self.stock_dim)

        # Update observation space to include stock holdings
        n_features = (
            self.stock_dim * len(config.tech_indicator_list)
            + self.stock_dim * 2  # Holdings + prices
        )

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(n_features,), dtype=np.float32
        )

        # Track holdings
        self.holdings = np.zeros(self.stock_dim)  # Shares held for each stock
        self.cash = config.initial_capital  # Available cash

    def _get_state(self) -> np.ndarray:
        """Construct state for multi-stock trading."""
        # Technical indicators
        tech_features = []
        for i in range(self.stock_dim):
            for indicator in self.config.tech_indicator_list:
                tech_features.append(self.data[f"{indicator}_{i}"])

        # Stock prices
        prices = []
        for i in range(self.stock_dim):
            prices.append(self.data[f"close_{i}"])

        # Combine features: indicators + prices + holdings
        state = np.concatenate(
            [np.array(tech_features), np.array(prices), self.holdings]
        )

        return state.astype(np.float32)

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        """Reset environment."""
        super().reset(seed=seed, options=options)

        self.holdings = np.zeros(self.stock_dim)  # Clear all positions
        self.cash = self.config.initial_capital  # Restore full capital

        return self.state, {}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute trading actions.

        Args:
            action: Array of actions (0=hold, 1=buy, 2=sell) for each stock
        """
        # Map actions: 0=hold, 1=buy, 2=sell
        prev_holdings = self.holdings.copy()  # Snapshot for return calc
        prev_cash = self.cash

        # Execute trades
        for i in range(self.stock_dim):
            price = self.data[f"close_{i}"]

            if action[i] == 1:  # Buy
                # Buy max possible shares (capped at hmax)
                max_shares = min(self.hmax, int(self.cash / price))
                cost = (
                    max_shares * price * (1 + self.transaction_cost_pct)
                )  # Include fee
                if cost <= self.cash:
                    self.holdings[i] += max_shares
                    self.cash -= cost

            elif action[i] == 2:  # Sell
                # Sell all holdings
                if self.holdings[i] > 0:
                    revenue = (
                        self.holdings[i] * price * (1 - self.transaction_cost_pct)
                    )  # Include fee
                    self.cash += revenue
                    self.holdings[i] = 0  # Clear position

        # Move to next day
        self.day += 1
        self.data = self.df.loc[self.day, :]

        # Calculate new portfolio value (cash + stock values)
        stock_values = []
        for i in range(self.stock_dim):
            price = self.data[f"close_{i}"]
            stock_values.append(self.holdings[i] * price)  # Market value of holding

        self.portfolio_value = self.cash + np.sum(stock_values)
        self.portfolio_values.append(self.portfolio_value)

        # Calculate reward (simplified return estimate)
        portfolio_return = (
            self.portfolio_value
            - prev_cash
            - np.sum(
                prev_holdings * self.data[f"close_0"]
            )  # Approximate prev stock value
        ) / (prev_cash + np.sum(prev_holdings * self.data[f"close_0"]))
        reward = portfolio_return * 100  # Scale up for training signal
        self.rewards_memory.append(reward)

        # Check terminal
        self.terminal = self.day >= len(self.df) - 1

        # Get new state
        self.state = self._get_state()

        info = {
            "portfolio_value": self.portfolio_value,
            "cash": self.cash,
            "holdings": self.holdings,
            "sharpe": self._calculate_sharpe(),
        }

        return self.state, reward, self.terminal, False, info
