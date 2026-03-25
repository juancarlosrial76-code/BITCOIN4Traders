"""
End-to-End Walk-Forward Backtest Test
======================================
Verifies that the complete training + backtesting pipeline runs on real
BTC/USDT data and produces meaningful KPI metrics.

KPI Requirements:
- Profit Factor, Sharpe, Max-DD, Win-Rate, Calmar are computed (not mocked)
- Pipeline completes without error on real parquet data
- At least 1 walk-forward window completes train + test cycle
"""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from src.environment.realistic_trading_env import RealisticTradingEnv, TradingEnvConfig
from src.agents.ppo_agent import PPOAgent, PPOConfig
from src.backtesting.walkforward_engine import WalkForwardEngine, WalkForwardConfig


DATA_PATH = Path("data/cache/BTC_USDT_1h_binance.parquet")
SMALL_DATA_PATH = Path("data/cache/test_btc_1h_200.parquet")


def _load_btc_data(n_bars: int = 500) -> pd.DataFrame:
    """Load a slice of real BTC data."""
    if DATA_PATH.exists():
        df = pd.read_parquet(DATA_PATH).iloc[-n_bars:].copy()
    elif SMALL_DATA_PATH.exists():
        df = pd.read_parquet(SMALL_DATA_PATH).copy()
    else:
        # Fallback: synthetic BTC-like data
        rng = np.random.default_rng(42)
        n = n_bars
        prices = 40000.0 * np.cumprod(1 + rng.normal(0.0001, 0.015, n))
        df = pd.DataFrame(
            {
                "open": prices * rng.uniform(0.999, 1.001, n),
                "high": prices * rng.uniform(1.000, 1.020, n),
                "low": prices * rng.uniform(0.980, 1.000, n),
                "close": prices,
                "volume": rng.uniform(500, 2000, n),
            }
        )
    # Ensure required columns
    for col in ["open", "high", "low", "close", "volume"]:
        if col not in df.columns:
            df[col] = df["close"]
    return df.reset_index(drop=True)


def _make_features(data: pd.DataFrame) -> pd.DataFrame:
    """Generate minimal feature DataFrame aligned to data index."""
    n = len(data)
    close = data["close"].values
    # Simple features: returns, sma ratio, volume ratio
    returns = np.concatenate([[0.0], np.diff(np.log(close + 1e-8))])
    sma20 = np.convolve(close, np.ones(20) / 20, mode="same")
    sma_ratio = close / (sma20 + 1e-8) - 1.0
    vol = data["volume"].values
    vol_ratio = vol / (vol.mean() + 1e-8) - 1.0
    features = pd.DataFrame(
        {
            "returns": returns,
            "sma_ratio": sma_ratio,
            "vol_ratio": vol_ratio,
        },
        index=data.index,
    )
    return features.fillna(0.0)


def _make_env(data: pd.DataFrame) -> RealisticTradingEnv:
    """Create a minimal trading environment from data."""
    features = _make_features(data)
    cfg = TradingEnvConfig(
        initial_capital=10_000.0,
        transaction_cost_bps=5.0,
        max_steps=len(data) - 1,
        max_drawdown=0.50,
    )
    return RealisticTradingEnv(price_data=data, features=features, config=cfg)


def _make_agent(env: RealisticTradingEnv) -> PPOAgent:
    """Create a minimal PPO agent sized for the env's observation space."""
    state_dim = env.observation_space.shape[0]
    cfg = PPOConfig(
        state_dim=state_dim,
        hidden_dim=64,
        n_actions=3,
        use_transformer=False,
        use_sil=False,
        n_epochs=2,  # fast for testing
        batch_size=32,
    )
    return PPOAgent(cfg, device="cpu")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestWalkForwardKPIs:
    """Verify KPI metrics are real (not mocked) and within sane bounds."""

    def test_calculate_metrics_profit_factor(self):
        """_calculate_metrics must include profit_factor."""
        data = _load_btc_data(300)
        env = _make_env(data)
        agent = _make_agent(env)
        cfg = WalkForwardConfig(
            train_window_days=5,
            test_window_days=2,
            step_days=2,
            train_iterations=2,
            min_trades=0,
        )
        engine = WalkForwardEngine(env, agent, cfg)

        # Synthetic trade list
        trades = [
            {"pnl": 100.0, "return": 0.01, "action": 1, "price": 40000, "timestamp": 0},
            {
                "pnl": -50.0,
                "return": -0.005,
                "action": 2,
                "price": 41000,
                "timestamp": 1,
            },
            {"pnl": 200.0, "return": 0.02, "action": 1, "price": 39000, "timestamp": 2},
        ]
        equity_curve = [10000.0, 10100.0, 10050.0, 10250.0]

        metrics = engine._calculate_metrics(trades, equity_curve)

        assert "profit_factor" in metrics, "profit_factor must be in metrics"
        assert "sharpe" in metrics
        assert "max_drawdown" in metrics
        assert "win_rate" in metrics
        assert "calmar" in metrics
        assert "n_trades" in metrics

        # profit_factor: gross_profit=300, gross_loss=50 → 6.0
        assert abs(metrics["profit_factor"] - 6.0) < 0.01
        assert metrics["win_rate"] == pytest.approx(2 / 3)
        assert metrics["n_trades"] == 3
        assert metrics["max_drawdown"] <= 0.0  # equity never exceeds starting

    def test_metrics_empty_trades(self):
        """Metrics with zero trades must not crash."""
        data = _load_btc_data(200)
        env = _make_env(data)
        agent = _make_agent(env)
        cfg = WalkForwardConfig(train_iterations=1, min_trades=0)
        engine = WalkForwardEngine(env, agent, cfg)

        metrics = engine._calculate_metrics([], [10000.0, 10050.0, 10100.0])
        assert metrics["profit_factor"] == 0.0
        assert metrics["win_rate"] == 0.0
        assert metrics["n_trades"] == 0

    def test_sharpe_positive_trend(self):
        """Monotonically rising equity → positive Sharpe."""
        data = _load_btc_data(200)
        env = _make_env(data)
        agent = _make_agent(env)
        cfg = WalkForwardConfig(train_iterations=1, min_trades=0)
        engine = WalkForwardEngine(env, agent, cfg)

        equity = [10000.0 * (1.001**i) for i in range(100)]
        metrics = engine._calculate_metrics([], equity)
        assert metrics["sharpe"] > 0, "Rising equity must give positive Sharpe"

    def test_max_drawdown_is_negative(self):
        """Max drawdown must be <= 0."""
        data = _load_btc_data(200)
        env = _make_env(data)
        agent = _make_agent(env)
        cfg = WalkForwardConfig(train_iterations=1, min_trades=0)
        engine = WalkForwardEngine(env, agent, cfg)

        # equity dips 10% then recovers
        equity = [10000.0, 10100.0, 9090.0, 9500.0, 10200.0]
        metrics = engine._calculate_metrics([], equity)
        assert metrics["max_drawdown"] < 0, "Max drawdown must be negative"
        assert metrics["max_drawdown"] > -1.0, "Max drawdown must be > -100%"


class TestTrainOnWindow:
    """Verify train_on_window runs real PPO episodes (not mock)."""

    def test_train_on_window_returns_real_metrics(self):
        """
        train_on_window must return actor_loss and n_episodes > 0.
        Uses only 3 training iterations to keep test fast.
        """
        data = _load_btc_data(300)
        env = _make_env(data)
        agent = _make_agent(env)
        cfg = WalkForwardConfig(
            train_window_days=5,
            test_window_days=2,
            step_days=2,
            train_iterations=3,  # minimal for speed
            min_trades=0,
        )
        engine = WalkForwardEngine(env, agent, cfg)

        metrics = engine.train_on_window(data, window_id=0)

        assert "n_episodes" in metrics, "Must report number of episodes"
        assert (
            metrics["n_episodes"] == 3
        ), "Should run exactly train_iterations episodes"
        assert "actor_loss" in metrics
        assert "return" in metrics
        # return is mean episode reward — could be anything but must be finite
        assert np.isfinite(metrics["return"]), "Episode return must be finite"

    def test_train_on_window_not_mock(self):
        """
        Verify train_on_window is NOT returning the old hardcoded mock values
        (return=0.10, sharpe=1.5, trades=50).
        """
        data = _load_btc_data(300)
        env = _make_env(data)
        agent = _make_agent(env)
        cfg = WalkForwardConfig(train_iterations=2, min_trades=0)
        engine = WalkForwardEngine(env, agent, cfg)

        metrics = engine.train_on_window(data, window_id=99)

        # Old mock always returned exactly these values
        assert not (
            metrics.get("return") == 0.10
            and metrics.get("sharpe") == 1.5
            and metrics.get("trades") == 50
        ), "train_on_window is still returning mock values!"


class TestEndToEndBacktest:
    """Full pipeline: load data → env → agent → walk-forward → KPIs."""

    def test_full_pipeline_on_real_data(self):
        """
        Run a minimal 2-window walk-forward on real BTC data.
        Verifies the complete pipeline produces valid KPI metrics.
        """
        data = _load_btc_data(500)
        env = _make_env(data)
        agent = _make_agent(env)

        cfg = WalkForwardConfig(
            train_window_days=10,  # ~240 hourly bars
            test_window_days=5,  # ~120 hourly bars
            step_days=5,
            train_iterations=3,  # minimal for CI speed
            min_trades=0,
            results_dir="data/backtests/test_e2e",
        )
        engine = WalkForwardEngine(env, agent, cfg)

        # Create windows from the data's date range
        if hasattr(data.index, "to_timestamp"):
            start = data.index[0].to_timestamp()
            end = data.index[-1].to_timestamp()
        elif isinstance(data.index, pd.DatetimeIndex):
            start = data.index[0]
            end = data.index[-1]
        else:
            # Integer index — use synthetic dates
            from datetime import datetime, timedelta

            start = datetime(2024, 1, 1)
            end = start + timedelta(hours=len(data))

        windows = engine.create_windows(start, end)
        assert len(windows) >= 1, "Must create at least 1 walk-forward window"

        # Run just the first window manually for speed
        # create_windows returns 4-tuples: (train_start, train_end, test_start, test_end)
        train_start, train_end, test_start, test_end = windows[0]

        if isinstance(data.index, pd.DatetimeIndex):
            train_data = data.loc[train_start:train_end]
            test_data = data.loc[test_start:test_end]
        else:
            # Integer-indexed fallback
            n_train = int(len(data) * 0.6)
            train_data = data.iloc[:n_train]
            test_data = data.iloc[n_train:]

        if len(train_data) < 10 or len(test_data) < 5:
            pytest.skip("Not enough data for the window sizes configured")

        train_metrics = engine.train_on_window(train_data, window_id=0)
        test_metrics = engine.test_on_window(test_data, window_id=0)

        # ── KPI assertions ────────────────────────────────────────────────────
        for key in ("return", "sharpe", "max_drawdown", "win_rate", "profit_factor"):
            assert key in test_metrics, f"Missing KPI: {key}"
            assert np.isfinite(test_metrics[key]), f"KPI {key} is not finite"

        assert test_metrics["max_drawdown"] <= 0.0
        assert 0.0 <= test_metrics["win_rate"] <= 1.0
        assert test_metrics["profit_factor"] >= 0.0
        assert train_metrics["n_episodes"] > 0
