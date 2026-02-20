"""
Portfolio Allocation Environment
=================================
Multi-asset portfolio optimization using reinforcement learning.

Features:
- Portfolio weight allocation (continuous actions)
- Covariance matrix as state feature
- Transaction cost modeling
- Rebalancing logic
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class PortfolioEnvConfig:
    """Configuration for portfolio environment."""

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
            self.tech_indicator_list = ["macd", "rsi", "cci", "adx"]


class PortfolioAllocationEnv(gym.Env):
    """
    Portfolio allocation environment for DRL.

    State Space:
    - Covariance matrix (stock_dim x stock_dim)
    - Technical indicators (macd, rsi, cci, adx) for each stock
    - Current portfolio weights
    - Cash position

    Action Space:
    - Continuous weights for each stock [0, 1]
    - Weights are normalized to sum to 1 using softmax

    Reward:
    - Portfolio return
    - Risk-adjusted return (Sharpe)
    - Transaction cost penalty
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
            self.stock_dim * self.stock_dim  # Covariance matrix
            + self.stock_dim * len(config.tech_indicator_list)  # Indicators
            + self.stock_dim  # Current weights
        )

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(n_features,), dtype=np.float32
        )

        # Initialize state
        self.data = self.df.loc[self.day, :]
        self.covs = None
        self.state = None
        self.portfolio_value = config.initial_capital
        self.portfolio_values = [config.initial_capital]
        self.asset_memory = [config.initial_capital]
        self.rewards_memory = []
        self.actions_memory = []
        self.date_memory = []

        # Terminal state flag
        self.terminal = False

        # Risk-free rate per day
        self.daily_risk_free = (1 + config.risk_free_rate) ** (1 / 252) - 1

        # Transaction cost
        self.transaction_cost_pct = config.transaction_cost_pct

    def _calculate_covariance(self, window: int = 60) -> np.ndarray:
        """Calculate covariance matrix of returns."""
        if self.day < window:
            # Use available data if less than window
            window = self.day + 1

        # Get past returns
        returns = []
        for i in range(window):
            day_idx = self.day - i
            if day_idx >= 0:
                day_data = self.df.loc[day_idx, :]
                # Calculate returns for each stock
                day_returns = []
                for j in range(self.stock_dim):
                    if day_idx > 0:
                        prev_close = self.df.loc[day_idx - 1, f"close_{j}"]
                        curr_close = day_data[f"close_{j}"]
                        ret = (curr_close - prev_close) / prev_close
                    else:
                        ret = 0.0
                    day_returns.append(ret)
                returns.append(day_returns)

        returns = np.array(returns)
        cov_matrix = np.cov(returns.T)

        # Ensure positive definite
        cov_matrix += np.eye(self.stock_dim) * 1e-6

        return cov_matrix

    def _get_state(self) -> np.ndarray:
        """Construct state vector."""
        # Calculate covariance matrix
        cov_matrix = self._calculate_covariance()
        self.covs = cov_matrix

        # Flatten covariance matrix
        cov_flat = cov_matrix.flatten()

        # Get technical indicators for all stocks
        tech_features = []
        for i in range(self.stock_dim):
            for indicator in self.config.tech_indicator_list:
                tech_features.append(self.data[f"{indicator}_{i}"])

        # Current portfolio weights (if any)
        if len(self.actions_memory) > 0:
            current_weights = self.actions_memory[-1]
        else:
            current_weights = np.zeros(self.stock_dim)

        # Combine all features
        state = np.concatenate([cov_flat, np.array(tech_features), current_weights])

        return state.astype(np.float32)

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        """Reset environment."""
        super().reset(seed=seed)

        self.day = 0
        self.data = self.df.loc[self.day, :]
        self.covs = self._calculate_covariance()
        self.state = self._get_state()
        self.portfolio_value = self.config.initial_capital
        self.portfolio_values = [self.config.initial_capital]
        self.asset_memory = [self.config.initial_capital]
        self.rewards_memory = []
        self.actions_memory = []
        self.date_memory = [self._get_date()]
        self.terminal = False

        return self.state, {}

    def _get_date(self):
        """Get current date from data."""
        if "date" in self.data.index:
            return self.data["date"]
        return self.day

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one step.

        Args:
            action: Portfolio weights (will be normalized with softmax)

        Returns:
            state, reward, terminated, truncated, info
        """
        # Normalize action to sum to 1 using softmax
        action = np.exp(action) / np.sum(np.exp(action))
        self.actions_memory.append(action)

        # Store previous portfolio value
        prev_portfolio_value = self.portfolio_value

        # Move to next day
        self.day += 1
        self.data = self.df.loc[self.day, :]
        self.date_memory.append(self._get_date())

        # Calculate portfolio return
        individual_returns = []
        for i in range(self.stock_dim):
            prev_close = self.df.loc[self.day - 1, f"close_{i}"]
            curr_close = self.data[f"close_{i}"]
            ret = (curr_close - prev_close) / prev_close
            individual_returns.append(ret)

        individual_returns = np.array(individual_returns)

        # Portfolio return (weighted sum of individual returns)
        portfolio_return = np.sum(action * individual_returns)

        # Calculate transaction costs (if rebalancing)
        if len(self.actions_memory) >= 2:
            prev_weights = self.actions_memory[-2]
            turnover = np.sum(np.abs(action - prev_weights))
            transaction_cost = turnover * self.transaction_cost_pct
        else:
            transaction_cost = 0.0

        # Net portfolio return
        net_return = portfolio_return - transaction_cost

        # Update portfolio value
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
            recent_returns = np.array(self.rewards_memory[-20:])
            volatility = np.std(recent_returns) + 1e-6
            sharpe_like = portfolio_return / volatility
            return sharpe_like

        return portfolio_return * 100

    def _calculate_sharpe(self) -> float:
        """Calculate Sharpe ratio."""
        if len(self.rewards_memory) < 2:
            return 0.0

        returns = np.array(self.rewards_memory)
        excess_returns = returns - self.daily_risk_free

        if np.std(returns) == 0:
            return 0.0

        sharpe = np.mean(excess_returns) / (np.std(returns) + 1e-6)
        return sharpe * np.sqrt(252)  # Annualize

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

        # Volatility
        volatility = (
            np.std(portfolio_returns) * np.sqrt(252)
            if len(portfolio_returns) > 0
            else 0.0
        )

        # Sharpe ratio
        sharpe = self._calculate_sharpe()

        # Max drawdown
        portfolio_values = np.array(self.portfolio_values)
        peak = np.maximum.accumulate(portfolio_values)
        drawdown = (portfolio_values - peak) / peak
        max_drawdown = np.min(drawdown)

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
    Multi-stock trading environment (discrete actions).

    Similar to portfolio allocation but with discrete actions
    for each stock (buy, sell, hold).
    """

    def __init__(
        self,
        df: pd.DataFrame,
        config: PortfolioEnvConfig,
        day: int = 0,
        hmax: int = 100,  # Max shares per trade
    ):
        super().__init__(df, config, day)

        self.hmax = hmax

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
        self.holdings = np.zeros(self.stock_dim)
        self.cash = config.initial_capital

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

        # Combine features
        state = np.concatenate(
            [np.array(tech_features), np.array(prices), self.holdings]
        )

        return state.astype(np.float32)

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        """Reset environment."""
        super().reset(seed=seed, options=options)

        self.holdings = np.zeros(self.stock_dim)
        self.cash = self.config.initial_capital

        return self.state, {}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute trading actions.

        Args:
            action: Array of actions (0=hold, 1=buy, 2=sell) for each stock
        """
        # Map actions: 0=hold, 1=buy, 2=sell
        prev_holdings = self.holdings.copy()
        prev_cash = self.cash

        # Execute trades
        for i in range(self.stock_dim):
            price = self.data[f"close_{i}"]

            if action[i] == 1:  # Buy
                # Buy max possible shares
                max_shares = min(self.hmax, int(self.cash / price))
                cost = max_shares * price * (1 + self.transaction_cost_pct)
                if cost <= self.cash:
                    self.holdings[i] += max_shares
                    self.cash -= cost

            elif action[i] == 2:  # Sell
                # Sell all holdings
                if self.holdings[i] > 0:
                    revenue = self.holdings[i] * price * (1 - self.transaction_cost_pct)
                    self.cash += revenue
                    self.holdings[i] = 0

        # Move to next day
        self.day += 1
        self.data = self.df.loc[self.day, :]

        # Calculate new portfolio value
        stock_values = []
        for i in range(self.stock_dim):
            price = self.data[f"close_{i}"]
            stock_values.append(self.holdings[i] * price)

        self.portfolio_value = self.cash + np.sum(stock_values)
        self.portfolio_values.append(self.portfolio_value)

        # Calculate reward
        portfolio_return = (
            self.portfolio_value
            - prev_cash
            - np.sum(prev_holdings * self.data[f"close_0"])
        ) / (prev_cash + np.sum(prev_holdings * self.data[f"close_0"]))
        reward = portfolio_return * 100
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
