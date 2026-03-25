"""
Tests for WalkForwardEngine (src/backtesting/walkforward_engine.py)
===================================================================
Uses lightweight mocks for env and agent — no training occurs.
"""

from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.backtesting.walkforward_engine import (
    WalkForwardConfig,
    WalkForwardEngine,
    WindowResult,
)


# ─────────────────────────────────────────────
#  Fixtures
# ─────────────────────────────────────────────


@pytest.fixture
def default_config(tmp_path) -> WalkForwardConfig:
    return WalkForwardConfig(
        train_window_days=365,
        test_window_days=90,
        step_days=90,
        train_iterations=2,
        results_dir=str(tmp_path / "results"),
    )


@pytest.fixture
def mock_env():
    env = MagicMock()
    obs = np.zeros(20, dtype=np.float32)
    env.reset.return_value = (obs, {})
    env.step.return_value = (obs, 0.0, True, False, {"equity": 100_000})
    return env


@pytest.fixture
def mock_agent():
    agent = MagicMock()
    agent.select_action.return_value = (0, 0.5, 0.5, None)
    agent.train.return_value = {"actor_loss": 0.1, "critic_loss": 0.2}
    return agent


@pytest.fixture
def engine(mock_env, mock_agent, default_config) -> WalkForwardEngine:
    return WalkForwardEngine(env=mock_env, agent=mock_agent, config=default_config)


# ─────────────────────────────────────────────
#  create_windows
# ─────────────────────────────────────────────


class TestCreateWindows:
    def test_creates_at_least_one_window(self, engine):
        start = datetime(2020, 1, 1)
        end = datetime(2022, 12, 31)
        windows = engine.create_windows(start, end)
        assert len(windows) >= 1

    def test_window_structure_four_tuple(self, engine):
        start = datetime(2020, 1, 1)
        end = datetime(2022, 12, 31)
        windows = engine.create_windows(start, end)
        train_s, train_e, test_s, test_e = windows[0]
        assert isinstance(train_s, datetime)
        assert isinstance(test_e, datetime)

    def test_train_end_equals_test_start(self, engine):
        windows = engine.create_windows(datetime(2020, 1, 1), datetime(2022, 12, 31))
        for train_s, train_e, test_s, test_e in windows:
            assert train_e == test_s, "test window must start immediately after training"

    def test_windows_do_not_exceed_end_date(self, engine):
        end = datetime(2022, 12, 31)
        windows = engine.create_windows(datetime(2020, 1, 1), end)
        for _, _, _, test_e in windows:
            assert test_e <= end

    def test_train_window_length_correct(self, engine, default_config):
        windows = engine.create_windows(datetime(2020, 1, 1), datetime(2023, 12, 31))
        train_s, train_e, _, _ = windows[0]
        delta = (train_e - train_s).days
        assert delta == default_config.train_window_days

    def test_test_window_length_correct(self, engine, default_config):
        windows = engine.create_windows(datetime(2020, 1, 1), datetime(2023, 12, 31))
        _, _, test_s, test_e = windows[0]
        delta = (test_e - test_s).days
        assert delta == default_config.test_window_days

    def test_windows_step_forward_by_step_days(self, engine, default_config):
        windows = engine.create_windows(datetime(2020, 1, 1), datetime(2023, 12, 31))
        if len(windows) >= 2:
            step_actual = (windows[1][0] - windows[0][0]).days
            assert step_actual == default_config.step_days

    def test_empty_range_returns_no_windows(self, engine):
        start = datetime(2020, 1, 1)
        # end is before one full train+test window
        end = start + timedelta(days=100)
        windows = engine.create_windows(start, end)
        assert windows == []

    def test_two_windows_non_overlapping_test_periods(self, engine, default_config):
        """With step_days == test_window_days, test periods should not overlap."""
        config = WalkForwardConfig(
            train_window_days=365,
            test_window_days=90,
            step_days=90,  # non-overlapping
            results_dir="/tmp/wf_test",
        )
        eng = WalkForwardEngine(engine.env, engine.agent, config)
        windows = eng.create_windows(datetime(2020, 1, 1), datetime(2022, 12, 31))
        if len(windows) >= 2:
            _, _, _, test_e1 = windows[0]
            _, _, test_s2, _ = windows[1]
            assert test_s2 >= test_e1


# ─────────────────────────────────────────────
#  _calculate_metrics
# ─────────────────────────────────────────────


class TestCalculateMetrics:
    def test_positive_return(self, engine):
        equity = [100_000, 105_000, 110_000]
        trades = [{"pnl": 2500}, {"pnl": 2500}]
        result = engine._calculate_metrics(trades, equity)
        assert result["return"] == pytest.approx(0.10, rel=1e-3)

    def test_negative_return(self, engine):
        equity = [100_000, 95_000, 90_000]
        trades = [{"pnl": -5000}]
        result = engine._calculate_metrics(trades, equity)
        assert result["return"] == pytest.approx(-0.10, rel=1e-3)

    def test_sharpe_present_in_result(self, engine):
        equity = list(np.linspace(100_000, 110_000, 50))
        result = engine._calculate_metrics([], equity)
        assert "sharpe" in result

    def test_win_rate_all_wins(self, engine):
        equity = [100_000, 110_000]
        trades = [{"pnl": 500}, {"pnl": 300}, {"pnl": 200}]
        result = engine._calculate_metrics(trades, equity)
        assert result["win_rate"] == pytest.approx(1.0)

    def test_win_rate_mixed(self, engine):
        equity = [100_000, 103_000]
        trades = [{"pnl": 300}, {"pnl": -100}, {"pnl": -50}, {"pnl": 200}]
        result = engine._calculate_metrics(trades, equity)
        assert result["win_rate"] == pytest.approx(0.5)

    def test_max_drawdown_is_non_positive(self, engine):
        equity = [100_000, 90_000, 80_000, 95_000]
        result = engine._calculate_metrics([], equity)
        assert result["max_drawdown"] <= 0

    def test_no_trades_returns_zero_win_rate(self, engine):
        equity = [100_000, 105_000]
        result = engine._calculate_metrics([], equity)
        assert result["win_rate"] == 0.0
        assert result["n_trades"] == 0


# ─────────────────────────────────────────────
#  summarize_results
# ─────────────────────────────────────────────


class TestSummarizeResults:
    def _make_result(self, window_id, test_return, test_sharpe, test_max_dd, win_rate):
        return WindowResult(
            window_id=window_id,
            train_start=datetime(2020, 1, 1),
            train_end=datetime(2021, 1, 1),
            test_start=datetime(2021, 1, 1),
            test_end=datetime(2021, 4, 1),
            train_return=0.05,
            train_sharpe=1.0,
            train_trades=50,
            test_return=test_return,
            test_sharpe=test_sharpe,
            test_sortino=0.8,
            test_calmar=0.5,
            test_max_drawdown=test_max_dd,
            test_trades=20,
            test_win_rate=win_rate,
            trades=[],
            equity_curve=pd.Series([100_000, 105_000]),
        )

    def test_n_windows(self, engine):
        results = [
            self._make_result(0, 0.05, 1.5, -0.10, 0.60),
            self._make_result(1, -0.02, -0.5, -0.08, 0.45),
        ]
        summary = engine.summarize_results(results)
        assert summary["n_windows"] == 2

    def test_mean_test_return(self, engine):
        results = [
            self._make_result(0, 0.10, 1.0, -0.05, 0.60),
            self._make_result(1, 0.04, 0.8, -0.03, 0.55),
        ]
        summary = engine.summarize_results(results)
        assert summary["mean_test_return"] == pytest.approx(0.07, rel=1e-3)

    def test_positive_and_negative_windows(self, engine):
        results = [
            self._make_result(0, 0.10, 1.0, -0.05, 0.60),
            self._make_result(1, -0.03, -0.5, -0.10, 0.40),
        ]
        summary = engine.summarize_results(results)
        assert summary["positive_windows"] == 1
        assert summary["negative_windows"] == 1

    def test_worst_max_dd(self, engine):
        results = [
            self._make_result(0, 0.05, 1.0, -0.05, 0.60),
            self._make_result(1, 0.03, 0.8, -0.20, 0.55),
        ]
        summary = engine.summarize_results(results)
        assert summary["worst_test_max_dd"] == pytest.approx(-0.20, rel=1e-3)

    def test_mean_win_rate(self, engine):
        results = [
            self._make_result(0, 0.05, 1.0, -0.05, 0.60),
            self._make_result(1, 0.03, 0.8, -0.10, 0.40),
        ]
        summary = engine.summarize_results(results)
        assert summary["mean_win_rate"] == pytest.approx(0.50, rel=1e-3)


# ─────────────────────────────────────────────
#  create_purged_splits
# ─────────────────────────────────────────────


class TestCreatePurgedSplits:
    def test_returns_tuple_of_two(self, engine):
        result = engine.create_purged_splits(n_samples=500)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_holdout_is_ndarray(self, engine):
        folds, holdout = engine.create_purged_splits(n_samples=500)
        assert isinstance(holdout, np.ndarray)

    def test_holdout_indices_within_bounds(self, engine):
        n = 500
        _, holdout = engine.create_purged_splits(n_samples=n)
        assert holdout.max() < n
        assert holdout.min() >= 0

    def test_holdout_does_not_overlap_folds(self, engine):
        n = 1000
        folds, holdout = engine.create_purged_splits(n_samples=n)
        if len(folds) == 0:
            pytest.skip("Anti-Bias Framework not available, skipping overlap check")
        holdout_set = set(holdout.tolist())
        for fold in folds:
            # FoldSplit objects use train_idx / test_idx attributes
            if hasattr(fold, "train_idx"):
                train_idx = fold.train_idx
                val_idx = fold.test_idx
            elif hasattr(fold, "train"):
                train_idx = fold.train
                val_idx = fold.val
            else:
                train_idx, val_idx = fold
            train_set = set(np.asarray(train_idx).tolist())
            val_set = set(np.asarray(val_idx).tolist())
            assert len(holdout_set & train_set) == 0, "holdout overlaps training"
            assert len(holdout_set & val_set) == 0, "holdout overlaps validation"
