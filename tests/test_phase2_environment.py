"""
Phase 2: Environment Tests
==========================

Tests for the trading environment including:
- Config-integrated environment
- Order book simulation
- Slippage models
- Transaction cost calculation
"""

import pytest
import numpy as np
import pandas as pd

from environment.config_integrated_env import ConfigIntegratedTradingEnv
from environment.config_system import (
    EnvironmentConfig,
    TransactionCostConfig,
    SlippageConfig,
    OrderBookConfig,
)
from environment.order_book import OrderBookSimulator, OrderBookConfig as OBConfig
from environment.slippage_model import SlippageModel, TransactionCostModel


@pytest.fixture
def sample_price_data():
    """Create sample price data."""
    np.random.seed(42)
    n_points = 200
    dates = pd.date_range("2023-01-01", periods=n_points, freq="1H")

    close = 50000 + np.cumsum(np.random.randn(n_points) * 100)

    return pd.DataFrame(
        {
            "open": close + np.random.randn(n_points) * 50,
            "high": close + abs(np.random.randn(n_points) * 100),
            "low": close - abs(np.random.randn(n_points) * 100),
            "close": close,
            "volume": np.random.uniform(100, 1000, n_points),
        },
        index=dates,
    )


@pytest.fixture
def sample_features():
    """Create sample features."""
    np.random.seed(42)
    n_points = 200
    dates = pd.date_range("2023-01-01", periods=n_points, freq="1H")

    return pd.DataFrame(
        {
            "log_ret": np.random.randn(n_points) * 0.01,
            "volatility_20": np.abs(np.random.randn(n_points)) * 0.02,
            "ou_score": np.random.randn(n_points),
        },
        index=dates,
    )


@pytest.fixture
def env_config():
    """Create environment configuration."""
    return EnvironmentConfig(
        initial_capital=100000,
        max_position_size=0.25,
        max_drawdown=0.20,
        max_steps=100,
        lookback_window=20,
    )


class TestEnvironmentConfig:
    """Test environment configuration."""

    def test_default_config(self):
        """Test default configuration."""
        config = EnvironmentConfig()
        assert config.initial_capital == 100000
        assert config.max_position_size == 1.0
        assert config.max_drawdown == 0.20

    def test_transaction_costs(self):
        """Test transaction cost configuration."""
        config = EnvironmentConfig()
        assert config.transaction_costs.maker_fee_bps == 2
        assert config.transaction_costs.taker_fee_bps == 5

    def test_slippage_config(self):
        """Test slippage configuration."""
        config = EnvironmentConfig()
        assert config.slippage.model_type == "volume_based"
        assert config.slippage.fixed_slippage_bps == 5.0


class TestConfigIntegratedTradingEnv:
    """Test the main trading environment."""

    def test_initialization(self, sample_price_data, sample_features, env_config):
        """Test environment initialization."""
        # Align indices
        common_idx = sample_price_data.index.intersection(sample_features.index)
        price_data = sample_price_data.loc[common_idx]
        features = sample_features.loc[common_idx]

        env = ConfigIntegratedTradingEnv(price_data, features, env_config)

        assert env.config == env_config
        assert env.observation_space is not None
        assert env.action_space is not None
        assert env.action_space.n == 7  # 7 position sizing actions

    def test_reset(self, sample_price_data, sample_features, env_config):
        """Test environment reset."""
        common_idx = sample_price_data.index.intersection(sample_features.index)
        price_data = sample_price_data.loc[common_idx]
        features = sample_features.loc[common_idx]

        env = ConfigIntegratedTradingEnv(price_data, features, env_config)
        obs, info = env.reset()

        assert isinstance(obs, np.ndarray)
        assert obs.shape[0] > 0
        assert isinstance(info, dict)
        assert info["position"] == 0
        assert info["equity"] == env_config.initial_capital

    def test_step_buy(self, sample_price_data, sample_features, env_config):
        """Test buy action."""
        common_idx = sample_price_data.index.intersection(sample_features.index)
        price_data = sample_price_data.loc[common_idx]
        features = sample_features.loc[common_idx]

        env = ConfigIntegratedTradingEnv(price_data, features, env_config)
        env.reset()

        # Buy action
        obs, reward, terminated, truncated, info = env.step(2)

        assert isinstance(obs, np.ndarray)
        assert isinstance(reward, (float, np.floating))
        assert isinstance(terminated, bool)
        assert info["position"] in [-1, 0, 1]

    def test_step_sell(self, sample_price_data, sample_features, env_config):
        """Test sell action."""
        common_idx = sample_price_data.index.intersection(sample_features.index)
        price_data = sample_price_data.loc[common_idx]
        features = sample_features.loc[common_idx]

        env = ConfigIntegratedTradingEnv(price_data, features, env_config)
        env.reset()

        # Sell action
        obs, reward, terminated, truncated, info = env.step(0)

        assert isinstance(obs, np.ndarray)
        assert isinstance(reward, (float, np.floating))

    def test_step_hold(self, sample_price_data, sample_features, env_config):
        """Test hold action."""
        common_idx = sample_price_data.index.intersection(sample_features.index)
        price_data = sample_price_data.loc[common_idx]
        features = sample_features.loc[common_idx]

        env = ConfigIntegratedTradingEnv(price_data, features, env_config)
        env.reset()

        # Neutral (action=2) should result in zero/flat position
        obs, reward, terminated, truncated, info = env.step(2)

        # Neutral action should result in zero position
        assert info["position"] == 0

    def test_equity_calculation(self, sample_price_data, sample_features, env_config):
        """Test equity calculation."""
        common_idx = sample_price_data.index.intersection(sample_features.index)
        price_data = sample_price_data.loc[common_idx]
        features = sample_features.loc[common_idx]

        env = ConfigIntegratedTradingEnv(price_data, features, env_config)
        env.reset()

        initial_equity = env._calculate_equity()
        assert initial_equity == env_config.initial_capital

    def test_episode_termination(self, sample_price_data, sample_features, env_config):
        """Test episode termination."""
        common_idx = sample_price_data.index.intersection(sample_features.index)
        price_data = sample_price_data.loc[common_idx]
        features = sample_features.loc[common_idx]

        env = ConfigIntegratedTradingEnv(price_data, features, env_config)
        env.reset()

        # Run until episode ends
        terminated = False
        steps = 0
        max_steps = 1000

        while not terminated and steps < max_steps:
            obs, reward, terminated, truncated, info = env.step(1)  # Hold
            steps += 1

        assert steps > 0
        assert steps <= max_steps


class TestOrderBookSimulator:
    """Test order book simulation."""

    def test_initialization(self):
        """Test order book initialization."""
        config = OBConfig(n_levels=10, base_spread_bps=2)
        ob = OrderBookSimulator(config)

        assert ob.config.n_levels == 10
        assert ob.config.base_spread_bps == 2

    def test_generate_order_book(self):
        """Test order book generation."""
        config = OBConfig(n_levels=5, base_spread_bps=2)
        ob = OrderBookSimulator(config)

        mid_price = 50000
        volatility = 0.02
        volume = 1000

        bid_prices, bid_volumes, ask_prices, ask_volumes = ob.generate_order_book(
            mid_price, volatility, volume
        )

        assert len(bid_prices) == config.n_levels
        assert len(ask_prices) == config.n_levels
        assert all(bid_prices < mid_price)
        assert all(ask_prices > mid_price)


class TestSlippageModel:
    """Test slippage calculation."""

    def test_fixed_slippage(self):
        """Test fixed slippage calculation."""
        from environment.slippage_model import (
            SlippageConfig as SlipCfg,
            SlippageModel as SlipModel,
        )

        config = SlipCfg(model_type="fixed", fixed_slippage_bps=5)
        model = SlipModel(config)

        price, slippage = model.calculate_slippage(
            "buy", 1.0, 50000, volume=1000, volatility=0.02
        )

        assert price > 50000  # Buy price should be higher
        assert slippage > 0

    def test_volume_based_slippage(self):
        """Test volume-based slippage."""
        from environment.slippage_model import (
            SlippageConfig as SlipCfg,
            SlippageModel as SlipModel,
        )

        config = SlipCfg(
            model_type="volume_based", fixed_slippage_bps=5, volume_impact_coef=0.1
        )
        model = SlipModel(config)

        price, slippage = model.calculate_slippage(
            "buy", 1.0, 50000, volume=1000, volatility=0.02
        )

        assert price > 50000
        assert slippage > 0

    def test_volatility_adjusted_slippage(self):
        """Test volatility-adjusted slippage."""
        from environment.slippage_model import (
            SlippageConfig as SlipCfg,
            SlippageModel as SlipModel,
        )

        config = SlipCfg(
            model_type="volatility",
            fixed_slippage_bps=5,
            volatility_multiplier=10,
        )
        model = SlipModel(config)

        # High volatility
        price1, slippage1 = model.calculate_slippage(
            "buy", 1.0, 50000, volume=1000, volatility=0.05
        )

        # Low volatility
        price2, slippage2 = model.calculate_slippage(
            "buy", 1.0, 50000, volume=1000, volatility=0.01
        )

        assert slippage1 > slippage2  # Higher vol = higher slippage


class TestTransactionCostModel:
    """Test transaction cost calculation."""

    def test_maker_fee_calculation(self):
        """Test maker fee calculation."""
        from environment.slippage_model import (
            TransactionCostConfig as TCConfig,
            SlippageConfig as SlipCfg,
            SlippageModel as SlipModel,
            TransactionCostModel as TCModel,
        )

        config = TCConfig(fixed_bps=2, include_slippage=False)
        slippage_config = SlipCfg(model_type="fixed", fixed_slippage_bps=0)
        slippage_model = SlipModel(slippage_config)

        cost_model = TCModel(config, slippage_model)

        costs = cost_model.calculate_total_cost(
            side="buy", quantity=1.0, price=50000, volume=1000, volatility=0.02
        )

        assert costs["fee_bps"] == 2
        assert costs["total_cost_dollars"] > 0

    def test_taker_fee_calculation(self):
        """Test taker fee calculation."""
        from environment.slippage_model import (
            TransactionCostConfig as TCConfig,
            SlippageConfig as SlipCfg,
            SlippageModel as SlipModel,
            TransactionCostModel as TCModel,
        )

        config = TCConfig(fixed_bps=5, include_slippage=False)
        slippage_config = SlipCfg(model_type="fixed", fixed_slippage_bps=0)
        slippage_model = SlipModel(slippage_config)

        cost_model = TCModel(config, slippage_model)

        costs = cost_model.calculate_total_cost(
            side="buy", quantity=1.0, price=50000, volume=1000, volatility=0.02
        )

        assert costs["fee_bps"] == 5
        assert costs["total_cost_dollars"] > 0

    def test_taker_fee_higher_than_maker(self):
        """Test that taker fee is higher than maker fee."""
        from environment.slippage_model import (
            TransactionCostConfig as TCConfig,
            SlippageConfig as SlipCfg,
            SlippageModel as SlipModel,
            TransactionCostModel as TCModel,
        )

        maker_config = TCConfig(fixed_bps=2, include_slippage=False)
        taker_config = TCConfig(fixed_bps=5, include_slippage=False)
        slippage_config = SlipCfg(model_type="fixed", fixed_slippage_bps=0)

        maker_model = TCModel(maker_config, SlipModel(slippage_config))
        taker_model = TCModel(taker_config, SlipModel(slippage_config))

        maker_costs = maker_model.calculate_total_cost(
            side="buy", quantity=1.0, price=50000, volume=1000, volatility=0.02
        )

        taker_costs = taker_model.calculate_total_cost(
            side="buy", quantity=1.0, price=50000, volume=1000, volatility=0.02
        )

        assert taker_costs["total_cost_dollars"] > maker_costs["total_cost_dollars"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
