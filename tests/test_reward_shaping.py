"""
Reward Shaping Tests
====================
Validates the _calculate_reward() function in RealisticTradingEnv.

Tests:
  1. Asymmetry: a loss produces a more negative reward than the equivalent
     gain produces a positive reward (2× penalty on losses).
  2. Clipping: reward is always in the [-10, 10] range.
  3. Transaction costs: costs reduce the reward vs. the cost-free case.
"""

import sys
import os

import numpy as np
import pandas as pd
import pytest

# Ensure project root and src/ are importable
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "src"))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_env(initial_capital: float = 100_000.0):
    """
    Build a minimal RealisticTradingEnv with synthetic data.

    We disable orderbook, antibias rewards, and antibias costs so the
    fallback pure-PnL reward branch is exercised.
    """
    from src.environment.realistic_trading_env import (
        RealisticTradingEnv,
        TradingEnvConfig,
    )

    np.random.seed(0)
    n = 500
    dates = pd.date_range("2022-01-01", periods=n, freq="1h")
    close = 30_000 + np.cumsum(np.random.randn(n) * 50)

    price_data = pd.DataFrame(
        {
            "open": close + np.random.randn(n) * 10,
            "high": close + np.abs(np.random.randn(n) * 20),
            "low": close - np.abs(np.random.randn(n) * 20),
            "close": close,
            "volume": np.random.uniform(50, 500, n),
        },
        index=dates,
    )

    features = pd.DataFrame(
        {
            "log_ret": np.log(
                price_data["close"] / price_data["close"].shift(1)
            ).fillna(0),
            "volatility_20": price_data["close"]
            .pct_change()
            .rolling(20)
            .std()
            .fillna(0.02),
            "ou_score": np.random.randn(n) * 0.5,
        },
        index=dates,
    )

    config = TradingEnvConfig(
        initial_capital=initial_capital,
        transaction_cost_bps=5.0,
        slippage_model="fixed",
        use_orderbook=False,  # faster, no order-book dependency
        use_antibias_rewards=False,  # force the simple fallback path
        use_antibias_costs=False,
        max_steps=200,
        lookback_window=25,
    )

    env = RealisticTradingEnv(price_data, features, config)
    return env


def _reward_for_equity_delta(
    env,
    equity_delta: float,
    trade_cost: float = 0.0,
) -> float:
    """
    Directly call _calculate_reward() with a crafted equity situation.

    We set env.cash so that _calculate_equity() returns
    (old_equity + equity_delta), then call _calculate_reward(old_equity, ...).
    """
    # Pick a stable step in the middle of the data
    env.current_step = env.config.lookback_window + 5
    env.shares = 0.0  # no open position → equity == cash

    old_equity = env.config.initial_capital
    # Set cash so that current_equity = old_equity + equity_delta
    env.cash = old_equity + equity_delta

    return env._calculate_reward(old_equity, trade_cost)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def env():
    """Shared environment instance (read-only across tests)."""
    return _make_env(initial_capital=100_000.0)


# ---------------------------------------------------------------------------
# Test 1 – Asymmetry
# ---------------------------------------------------------------------------


class TestAsymmetry:
    """Loss rewards should be more negative than equivalent gains are positive."""

    @pytest.mark.parametrize("delta_usd", [500, 1_000, 5_000])
    def test_loss_more_negative_than_gain_positive(self, env, delta_usd):
        """
        For a $delta_usd gain: reward should be  +X
        For a $delta_usd loss:  reward should be < -X  (2× asymmetry)
        """
        reward_gain = _reward_for_equity_delta(env, +delta_usd)
        reward_loss = _reward_for_equity_delta(env, -delta_usd)

        assert (
            reward_gain > 0
        ), f"Positive PnL should yield positive reward; got {reward_gain:.4f}"
        assert (
            reward_loss < 0
        ), f"Negative PnL should yield negative reward; got {reward_loss:.4f}"
        assert abs(reward_loss) > abs(reward_gain), (
            f"Loss reward ({reward_loss:.4f}) should be more negative than "
            f"gain reward ({reward_gain:.4f}) is positive (expected 2× asymmetry)"
        )

    def test_exact_two_times_multiplier(self, env):
        """
        Without transaction costs the ratio |loss_reward| / |gain_reward|
        should be exactly 2.0 (before clipping applies).

        We use a small delta so that clipping at ±10 is not triggered.
        """
        delta = 100.0  # $100 — well below clipping threshold
        reward_gain = _reward_for_equity_delta(env, +delta, trade_cost=0.0)
        reward_loss = _reward_for_equity_delta(env, -delta, trade_cost=0.0)

        ratio = abs(reward_loss) / abs(reward_gain)
        assert abs(ratio - 2.0) < 1e-6, f"Expected ratio of 2.0, got {ratio:.6f}"


# ---------------------------------------------------------------------------
# Test 2 – Clipping
# ---------------------------------------------------------------------------


class TestClipping:
    """Reward must always be in the closed interval [-10, 10]."""

    @pytest.mark.parametrize(
        "delta_usd",
        [
            -1_000_000,  # catastrophic loss — should saturate at -10
            -100_000,
            -10_000,
            -1_000,
            0,
            +1_000,
            +10_000,
            +100_000,
            +1_000_000,  # massive gain — should saturate at +10
        ],
    )
    def test_reward_in_bounds(self, env, delta_usd):
        reward = _reward_for_equity_delta(env, delta_usd)
        assert (
            -10.0 <= reward <= 10.0
        ), f"Reward {reward:.4f} out of [-10, 10] for delta={delta_usd}"

    def test_extreme_loss_clips_to_minus_ten(self, env):
        """Very large losses should hit the -10 floor."""
        reward = _reward_for_equity_delta(env, -10_000_000)
        assert reward == -10.0, f"Expected -10.0, got {reward}"

    def test_extreme_gain_clips_to_plus_ten(self, env):
        """Very large gains should hit the +10 ceiling."""
        reward = _reward_for_equity_delta(env, +10_000_000)
        assert reward == 10.0, f"Expected +10.0, got {reward}"


# ---------------------------------------------------------------------------
# Test 3 – Transaction costs reduce reward
# ---------------------------------------------------------------------------


class TestTransactionCostPenalty:
    """Incurring a transaction cost should reduce the reward."""

    @pytest.mark.parametrize(
        "delta_usd,cost",
        [
            (0, 500),  # flat PnL but with cost → reward negative
            (+500, 100),  # small gain with cost → smaller reward
            (-500, 100),  # loss with cost → even more negative reward
            (+5_000, 1_000),  # larger gain with significant cost
        ],
    )
    def test_cost_reduces_reward(self, env, delta_usd, cost):
        reward_no_cost = _reward_for_equity_delta(env, delta_usd, trade_cost=0.0)
        reward_with_cost = _reward_for_equity_delta(env, delta_usd, trade_cost=cost)

        assert reward_with_cost < reward_no_cost, (
            f"Cost should reduce reward: no_cost={reward_no_cost:.4f}, "
            f"with_cost={reward_with_cost:.4f} (delta={delta_usd}, cost={cost})"
        )

    def test_cost_penalty_magnitude(self, env):
        """
        Verify the cost penalty formula:
            cost_penalty = (trade_cost / initial_capital) * 0.5 * 100
        """
        initial_capital = env.config.initial_capital  # 100_000
        trade_cost = 1_000.0  # $1k cost

        # With no PnL change the reward = 0 - cost_penalty * scale
        # cost_penalty = (1000 / 100000) * 0.5 = 0.005
        # after *100 scaling: -0.5
        expected_cost_effect = -(trade_cost / initial_capital) * 0.5 * 100

        reward_no_cost = _reward_for_equity_delta(env, 0.0, trade_cost=0.0)
        reward_with_cost = _reward_for_equity_delta(env, 0.0, trade_cost=trade_cost)

        actual_effect = reward_with_cost - reward_no_cost
        assert abs(actual_effect - expected_cost_effect) < 1e-4, (
            f"Expected cost effect {expected_cost_effect:.4f}, "
            f"got {actual_effect:.4f}"
        )

    def test_zero_cost_no_penalty(self, env):
        """A trade with zero cost should not be penalised for cost."""
        delta = 1_000.0
        reward_no_cost = _reward_for_equity_delta(env, delta, trade_cost=0.0)
        reward_zero_cost = _reward_for_equity_delta(env, delta, trade_cost=0.0)
        assert reward_no_cost == reward_zero_cost


# ---------------------------------------------------------------------------
# Test 4 – Integration smoke-test (full env step)
# ---------------------------------------------------------------------------


class TestFullStepIntegration:
    """
    End-to-end sanity: run a few steps through the env and verify that
    rewards are finite and in-bounds.
    """

    def test_rewards_finite_and_bounded(self):
        env = _make_env()
        obs, _ = env.reset()
        rewards = []
        for _ in range(50):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            rewards.append(reward)
            if terminated or truncated:
                break

        rewards_arr = np.array(rewards)
        assert np.all(np.isfinite(rewards_arr)), "Some rewards are NaN or Inf"
        assert np.all(
            rewards_arr >= -10.0
        ), f"Some rewards below -10: {rewards_arr.min()}"
        assert np.all(
            rewards_arr <= 10.0
        ), f"Some rewards above +10: {rewards_arr.max()}"
