"""
Tests for PerformanceCalculator (src/backtesting/performance_calculator.py)
===========================================================================
All tests use synthetic in-memory data. No file I/O or live API calls.
"""

import numpy as np
import pandas as pd
import pytest

from src.backtesting.performance_calculator import PerformanceCalculator, PerformanceMetrics


# ─────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────


def make_equity_series(values, start="2020-01-01", freq="D") -> pd.Series:
    idx = pd.date_range(start=start, periods=len(values), freq=freq)
    return pd.Series(values, index=idx)


def make_trades_df(pnls: list) -> pd.DataFrame:
    return pd.DataFrame({"pnl": pnls})


# ─────────────────────────────────────────────
#  calculate_from_equity_curve
# ─────────────────────────────────────────────


class TestCalculateFromEquityCurve:
    def setup_method(self):
        self.calc = PerformanceCalculator(risk_free_rate=0.0)

    def test_total_return_positive(self):
        equity = make_equity_series([100_000, 105_000, 110_000, 115_000, 120_000])
        metrics = self.calc.calculate_from_equity_curve(equity)
        assert metrics.total_return == pytest.approx(0.20, rel=1e-3)

    def test_total_return_negative(self):
        equity = make_equity_series([100_000, 95_000, 90_000])
        metrics = self.calc.calculate_from_equity_curve(equity)
        assert metrics.total_return == pytest.approx(-0.10, rel=1e-3)

    def test_max_drawdown_is_negative(self):
        # Peak at 110k, trough at 90k → dd = (90-110)/110 ≈ -0.1818
        equity = make_equity_series([100_000, 110_000, 90_000, 95_000])
        metrics = self.calc.calculate_from_equity_curve(equity)
        assert metrics.max_drawdown < 0

    def test_max_drawdown_known_value(self):
        # Peak 100 → trough 80 → drawdown = -0.20
        equity = make_equity_series([100, 100, 80, 90])
        metrics = self.calc.calculate_from_equity_curve(equity)
        assert metrics.max_drawdown == pytest.approx(-0.20, abs=1e-6)

    def test_sharpe_flat_equity_is_zero(self):
        equity = make_equity_series([100_000] * 100)
        metrics = self.calc.calculate_from_equity_curve(equity)
        assert metrics.sharpe_ratio == pytest.approx(0.0)

    def test_volatility_positive_for_varying_equity(self):
        np.random.seed(42)
        prices = 100_000 + np.cumsum(np.random.randn(252) * 500)
        equity = make_equity_series(prices)
        metrics = self.calc.calculate_from_equity_curve(equity)
        assert metrics.volatility > 0

    def test_sharpe_reasonable_range(self):
        """Sharpe for a linear uptrend should be positive."""
        equity = make_equity_series(np.linspace(100_000, 120_000, 252))
        metrics = self.calc.calculate_from_equity_curve(equity)
        assert metrics.sharpe_ratio > 0

    def test_with_trade_metrics(self):
        equity = make_equity_series(np.linspace(100_000, 110_000, 100))
        trades = make_trades_df([500, -200, 300, -100, 400, -50])
        metrics = self.calc.calculate_from_equity_curve(equity, trades)
        assert metrics.total_trades == 6
        assert 0 < metrics.win_rate < 1
        assert metrics.profit_factor > 0


# ─────────────────────────────────────────────
#  Edge Cases
# ─────────────────────────────────────────────


class TestEdgeCases:
    def setup_method(self):
        self.calc = PerformanceCalculator()

    def test_empty_equity_returns_empty_metrics(self):
        equity = pd.Series([], dtype=float)
        metrics = self.calc.calculate_from_equity_curve(equity)
        assert isinstance(metrics, PerformanceMetrics)
        assert metrics.total_return == 0.0
        assert metrics.sharpe_ratio == 0.0

    def test_single_element_equity_no_crash(self):
        equity = make_equity_series([100_000])
        metrics = self.calc.calculate_from_equity_curve(equity)
        assert isinstance(metrics, PerformanceMetrics)
        assert metrics.total_return == 0.0

    def test_two_element_equity_no_crash(self):
        equity = make_equity_series([100_000, 105_000])
        metrics = self.calc.calculate_from_equity_curve(equity)
        assert isinstance(metrics, PerformanceMetrics)

    def test_empty_metrics_all_zeros(self):
        metrics = self.calc._empty_metrics()
        assert metrics.total_return == 0.0
        assert metrics.sharpe_ratio == 0.0
        assert metrics.max_drawdown == 0.0
        assert metrics.total_trades == 0
        assert metrics.recovery_time == -1  # convention for not-yet-recovered


# ─────────────────────────────────────────────
#  _calculate_drawdown_metrics
# ─────────────────────────────────────────────


class TestDrawdownMetrics:
    def setup_method(self):
        self.calc = PerformanceCalculator()

    def test_no_drawdown_peak_series(self):
        equity = make_equity_series([100, 110, 120, 130])
        result = self.calc._calculate_drawdown_metrics(equity)
        assert result["max_drawdown"] == pytest.approx(0.0, abs=1e-9)

    def test_known_max_drawdown(self):
        equity = make_equity_series([100, 120, 90, 100])
        result = self.calc._calculate_drawdown_metrics(equity)
        # peak=120, trough=90 → dd = (90-120)/120 = -0.25
        assert result["max_drawdown"] == pytest.approx(-0.25, abs=1e-6)

    def test_avg_drawdown_negative(self):
        equity = make_equity_series([100, 90, 80, 100])
        result = self.calc._calculate_drawdown_metrics(equity)
        assert result["avg_drawdown"] < 0

    def test_max_drawdown_duration_positive(self):
        equity = make_equity_series([100, 90, 85, 80, 100])
        result = self.calc._calculate_drawdown_metrics(equity)
        assert result["max_drawdown_duration"] >= 3


# ─────────────────────────────────────────────
#  _calculate_trade_metrics
# ─────────────────────────────────────────────


class TestTradeMetrics:
    def setup_method(self):
        self.calc = PerformanceCalculator()

    def test_win_rate_mixed(self):
        trades = make_trades_df([100, -50, 200, -100, 150])
        result = self.calc._calculate_trade_metrics(trades)
        assert result["win_rate"] == pytest.approx(0.6)

    def test_profit_factor(self):
        trades = make_trades_df([300, -100])  # gross profit 300, gross loss 100
        result = self.calc._calculate_trade_metrics(trades)
        assert result["profit_factor"] == pytest.approx(3.0)

    def test_all_winners(self):
        trades = make_trades_df([100, 200, 300])
        result = self.calc._calculate_trade_metrics(trades)
        assert result["win_rate"] == pytest.approx(1.0)
        assert result["profit_factor"] == 0.0  # no losses

    def test_all_losers(self):
        trades = make_trades_df([-100, -200])
        result = self.calc._calculate_trade_metrics(trades)
        assert result["win_rate"] == 0.0
        assert result["profit_factor"] == 0.0

    def test_total_trades_count(self):
        trades = make_trades_df([100, -50, 200, -100])
        result = self.calc._calculate_trade_metrics(trades)
        assert result["total_trades"] == 4

    def test_missing_pnl_column_returns_zeros(self):
        trades = pd.DataFrame({"entry": [40000], "exit": [41000]})
        result = self.calc._calculate_trade_metrics(trades)
        assert result["total_trades"] == 0
        assert result["win_rate"] == 0.0

    def test_largest_win_and_loss(self):
        trades = make_trades_df([500, 100, -200, -50])
        result = self.calc._calculate_trade_metrics(trades)
        assert result["largest_win"] == pytest.approx(500)
        assert result["largest_loss"] == pytest.approx(-200)


# ─────────────────────────────────────────────
#  Fees Scenario
# ─────────────────────────────────────────────


class TestFeesScenario:
    def test_lower_return_with_fees(self):
        calc_no_fee = PerformanceCalculator()
        calc_fee = PerformanceCalculator()

        equity_no_fee = make_equity_series(np.linspace(100_000, 110_000, 252))
        # Simulate a net-of-fee equity curve that grows slower
        equity_with_fee = make_equity_series(np.linspace(100_000, 107_000, 252))

        m_no = calc_no_fee.calculate_from_equity_curve(equity_no_fee)
        m_fee = calc_fee.calculate_from_equity_curve(equity_with_fee)

        assert m_no.total_return > m_fee.total_return
