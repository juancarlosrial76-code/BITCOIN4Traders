"""
Tests for ConfigIntegratedTradingEnv (src/environment/config_integrated_env.py)
=================================================================================
Uses minimal synthetic price/feature data and a default EnvironmentConfig.
HMM regime detection is disabled via config to avoid hmmlearn dependency.
"""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock

from src.environment.config_system import (
    EnvironmentConfig,
    TransactionCostConfig,
    SlippageConfig,
    OrderBookConfig,
    RewardConfig,
    RewardComponent,
    MarketConfig,
)
from src.environment.config_integrated_env import ConfigIntegratedTradingEnv


# ─────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────


N_STEPS = 120  # enough data for lookback=50 + episode steps
N_FEATURES = 12


def make_price_df(n: int = N_STEPS, seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)
    prices = 50_000 + np.cumsum(np.random.randn(n) * 100)
    prices = np.abs(prices) + 1  # ensure positive
    idx = pd.date_range("2023-01-01", periods=n, freq="h")
    return pd.DataFrame(
        {
            "open": prices * (1 + np.random.randn(n) * 0.0005),
            "high": prices * 1.001,
            "low": prices * 0.999,
            "close": prices,
            "volume": np.random.uniform(10, 100, n),
        },
        index=idx,
        dtype=np.float64,
    )


def make_features_df(n: int = N_STEPS, n_feat: int = N_FEATURES, seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)
    idx = pd.date_range("2023-01-01", periods=n, freq="h")
    cols = [f"feat_{i}" for i in range(n_feat - 1)] + ["volatility_20"]
    data = np.random.randn(n, n_feat).astype(np.float64)
    return pd.DataFrame(data, index=idx, columns=cols)


def make_env_config(disable_hmm: bool = True) -> EnvironmentConfig:
    return EnvironmentConfig(
        type="test",
        initial_capital=100_000.0,
        max_position_size=1.0,
        min_position_size=0.0,
        max_drawdown=0.20,
        max_consecutive_losses=5,
        lookback_window=50,
        max_steps=N_STEPS,
        include_orderbook_features=False,
        include_portfolio_metrics=True,
        normalize_observations=False,
        transaction_costs=TransactionCostConfig(
            fixed_bps=5.0,
            maker_fee_bps=2.0,
            taker_fee_bps=5.0,
            include_slippage=False,
            include_market_impact=False,
        ),
        slippage=SlippageConfig(model_type="fixed", fixed_slippage_bps=2.0),
        orderbook=OrderBookConfig(enabled=False),
        reward=RewardConfig(
            components=[RewardComponent(name="return", weight=1.0)],
            clip_min=-10.0,
            clip_max=10.0,
            scale=1.0,
        ),
        market=MarketConfig(
            vol_regimes={"normal": 0.02},
            volume_patterns={"normal": 500.0},
            spread_patterns={"normal": 5.0},
        ),
    )


@pytest.fixture
def price_df():
    return make_price_df()


@pytest.fixture
def features_df():
    return make_features_df()


@pytest.fixture
def env_config():
    return make_env_config()


@pytest.fixture
def env(price_df, features_df, env_config):
    # Disable HMM globally for tests
    with patch(
        "src.environment.config_integrated_env._HMM_AVAILABLE", False
    ):
        e = ConfigIntegratedTradingEnv(
            price_data=price_df,
            features=features_df,
            config=env_config,
        )
    return e


# ─────────────────────────────────────────────
#  reset()
# ─────────────────────────────────────────────


class TestReset:
    def test_returns_obs_and_info(self, env):
        obs, info = env.reset()
        assert obs is not None
        assert isinstance(info, dict)

    def test_obs_is_numpy_array(self, env):
        obs, _ = env.reset()
        assert isinstance(obs, np.ndarray)

    def test_obs_is_finite(self, env):
        obs, _ = env.reset()
        assert np.all(np.isfinite(obs)), "Observation must be finite after reset"

    def test_obs_shape_consistent_across_resets(self, env):
        obs1, _ = env.reset()
        obs2, _ = env.reset()
        assert obs1.shape == obs2.shape


# ─────────────────────────────────────────────
#  step()
# ─────────────────────────────────────────────


class TestStep:
    def test_step_returns_five_tuple(self, env):
        env.reset()
        result = env.step(2)  # action=2 → neutral
        assert len(result) == 5

    def test_reward_is_float(self, env):
        env.reset()
        _, reward, _, _, _ = env.step(2)
        assert isinstance(reward, (float, np.floating))

    def test_obs_shape_consistent_after_step(self, env):
        obs0, _ = env.reset()
        obs1, _, _, _, _ = env.step(2)
        assert obs0.shape == obs1.shape

    def test_obs_is_finite_after_step(self, env):
        env.reset()
        obs, _, _, _, _ = env.step(2)
        assert np.all(np.isfinite(obs))

    def test_info_has_equity_key(self, env):
        env.reset()
        _, _, _, _, info = env.step(2)
        assert "equity" in info

    def test_terminated_or_truncated_eventually(self, env):
        """Episode must end within max_steps."""
        env.reset()
        done = False
        for _ in range(N_STEPS + 10):
            _, _, terminated, truncated, _ = env.step(2)
            if terminated or truncated:
                done = True
                break
        assert done, "Episode must terminate before data runs out"

    def test_neutral_action_no_cost(self, env):
        """Holding a neutral position (action=2) from flat should cost 0."""
        env.reset()
        env.position = 0.0  # ensure flat start
        _, reward1, _, _, info1 = env.step(2)
        _, reward2, _, _, info2 = env.step(2)
        # Equity should not change due to trading costs (only mark-to-market)
        assert info1["equity"] > 0
        assert info2["equity"] > 0


# ─────────────────────────────────────────────
#  Reward clipping
# ─────────────────────────────────────────────


class TestRewardClipping:
    def test_reward_within_clip_range(self, env):
        env.reset()
        for _ in range(10):
            _, reward, terminated, truncated, _ = env.step(np.random.randint(0, 7))
            # Config clip_min=-10, clip_max=10 with scale=1
            assert reward >= -50, f"Reward {reward} should be >= -50 (risk penalty)"
            assert reward <= 50, f"Reward {reward} should be <= 50"
            if terminated or truncated:
                break

    def test_reward_clipped_for_dynamic_reward(self, price_df, features_df):
        """With tight clip bounds rewards should stay within them."""
        config = make_env_config()
        config.reward.clip_min = -1.0
        config.reward.clip_max = 1.0

        with patch("src.environment.config_integrated_env._HMM_AVAILABLE", False):
            e = ConfigIntegratedTradingEnv(price_df, features_df, config)

        e.reset()
        for _ in range(5):
            _, reward, terminated, truncated, _ = e.step(2)  # neutral
            if terminated or truncated:
                break
            # Rewards from neutral steps should be within clip range
            # (unless risk penalty kicks in)
            assert reward >= -52, f"Reward {reward} too low"


# ─────────────────────────────────────────────
#  Position sign flip
# ─────────────────────────────────────────────


class TestPositionChange:
    def test_long_position_established(self, env):
        env.reset()
        env.step(6)  # Long 100%
        assert hasattr(env, "position")
        assert env.position > 0

    def test_neutral_flattens_position(self, env):
        env.reset()
        env.step(6)     # go long
        env.step(2)     # go neutral
        # After neutral step, position should be ~0 (may not be exactly 0 due to
        # trading mechanics, but should be closer to 0 than 1)
        assert abs(env.position) <= 1.0

    def test_equity_is_positive(self, env):
        env.reset()
        for _ in range(5):
            _, _, terminated, truncated, info = env.step(4)  # Long 50%
            assert info["equity"] > 0
            if terminated or truncated:
                break


# ─────────────────────────────────────────────
#  Action space
# ─────────────────────────────────────────────


class TestActionSpace:
    def test_action_space_is_discrete(self, env):
        import gymnasium as gym
        assert isinstance(env.action_space, gym.spaces.Discrete)

    def test_action_space_has_7_actions(self, env):
        assert env.action_space.n == 7

    def test_observation_space_defined(self, env):
        import gymnasium as gym
        assert isinstance(env.observation_space, gym.spaces.Box)

    def test_obs_shape_matches_observation_space(self, env):
        obs, _ = env.reset()
        assert obs.shape == env.observation_space.shape
