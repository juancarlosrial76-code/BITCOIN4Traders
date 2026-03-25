"""
Tests for RiskManager circuit breaker (src/risk/risk_manager.py)
================================================================
Tests the check_circuit_breaker() method, validate_position_size() under halt,
and reset(). No network calls needed.
"""

import pytest
from unittest.mock import patch, MagicMock

from src.risk.risk_manager import RiskConfig, RiskManager


# ─────────────────────────────────────────────
#  Fixtures
# ─────────────────────────────────────────────


@pytest.fixture
def config():
    return RiskConfig(
        max_drawdown_per_session=0.02,     # 2% drawdown limit
        max_consecutive_losses=5,
        max_position_size=0.25,
        kelly_fraction=0.5,
        min_capital_threshold=0.30,
        enable_circuit_breaker=True,
    )


@pytest.fixture
def risk_mgr(config):
    return RiskManager(config=config, initial_capital=100_000.0)


# ─────────────────────────────────────────────
#  check_circuit_breaker
# ─────────────────────────────────────────────


class TestCheckCircuitBreaker:
    def test_no_trigger_under_normal_conditions(self, risk_mgr):
        # equity just below threshold but not quite
        should_halt, reason = risk_mgr.check_circuit_breaker(99_000.0)
        assert should_halt is False
        assert reason is None

    def test_triggers_on_drawdown_exceeds_max(self, risk_mgr):
        # Draw down 3% from initial capital (limit is 2%)
        risk_mgr.state.peak_equity = 100_000.0
        should_halt, reason = risk_mgr.check_circuit_breaker(97_500.0)
        assert should_halt is True
        assert reason is not None
        assert "drawdown" in reason.lower() or "Drawdown" in reason

    def test_triggers_on_consecutive_losses_exceeds_limit(self, risk_mgr):
        risk_mgr.state.consecutive_losses = 5  # equal to limit
        should_halt, reason = risk_mgr.check_circuit_breaker(99_000.0)
        assert should_halt is True
        assert reason is not None

    def test_triggers_on_consecutive_losses_above_limit(self, risk_mgr):
        risk_mgr.state.consecutive_losses = 7
        should_halt, reason = risk_mgr.check_circuit_breaker(99_000.0)
        assert should_halt is True

    def test_triggers_when_equity_below_min_capital_threshold(self, risk_mgr):
        # threshold is 30% of initial (100k → threshold is 30k)
        should_halt, reason = risk_mgr.check_circuit_breaker(25_000.0)
        assert should_halt is True
        assert reason is not None

    def test_no_trigger_when_circuit_breaker_disabled(self):
        config = RiskConfig(
            enable_circuit_breaker=False,
            max_drawdown_per_session=0.02,
            max_consecutive_losses=5,
            min_capital_threshold=0.30,
        )
        rm = RiskManager(config=config, initial_capital=100_000.0)
        # Even with massive drawdown it must NOT trigger
        should_halt, reason = risk_mgr_disabled(config) if False else (
            rm.check_circuit_breaker(1.0)  # 99.999% drawdown
        )
        assert should_halt is False
        assert reason is None

    @pytest.mark.parametrize("equity,should_trip", [
        (100_000, False),   # no drawdown
        (98_001, False),    # just under 2% drawdown (strict > check)
        (97_999, True),     # just over 2% drawdown
        (95_000, True),     # 5% drawdown
    ])
    def test_drawdown_boundary(self, config, equity, should_trip):
        rm = RiskManager(config=config, initial_capital=100_000.0)
        rm.state.peak_equity = 100_000.0
        should_halt, _ = rm.check_circuit_breaker(equity)
        assert should_halt is should_trip


def risk_mgr_disabled(config):  # helper, not a test
    pass


# ─────────────────────────────────────────────
#  validate_position_size under halt
# ─────────────────────────────────────────────


class TestValidatePositionSizeHalt:
    def test_size_reduced_to_zero_when_halted(self, risk_mgr):
        risk_mgr.state.halt_trading = True
        risk_mgr.state.halt_reason = "Test halt"
        approved, size = risk_mgr.validate_position_size(
            proposed_size=10_000.0,
            current_capital=100_000.0,
        )
        assert approved is False
        assert size == 0.0

    def test_approved_when_not_halted(self, risk_mgr):
        assert risk_mgr.state.halt_trading is False
        approved, size = risk_mgr.validate_position_size(
            proposed_size=5_000.0,
            current_capital=100_000.0,
        )
        assert approved is True
        assert size > 0

    def test_hard_cap_enforced(self, risk_mgr):
        # Proposed $50k when cap is 25% of $100k = $25k
        approved, size = risk_mgr.validate_position_size(
            proposed_size=50_000.0,
            current_capital=100_000.0,
        )
        assert approved is True
        assert size <= 25_000.0


# ─────────────────────────────────────────────
#  reset
# ─────────────────────────────────────────────


class TestRiskManagerReset:
    def test_reset_clears_halt_flag(self, risk_mgr):
        risk_mgr.state.halt_trading = True
        risk_mgr.state.halt_reason = "Test"
        risk_mgr.reset()
        assert risk_mgr.state.halt_trading is False
        assert risk_mgr.state.halt_reason is None

    def test_reset_clears_consecutive_losses(self, risk_mgr):
        risk_mgr.state.consecutive_losses = 7
        risk_mgr.reset()
        assert risk_mgr.state.consecutive_losses == 0

    def test_reset_restores_equity_to_initial(self, risk_mgr):
        risk_mgr.state.current_equity = 70_000.0
        risk_mgr.reset()
        assert risk_mgr.state.current_equity == risk_mgr.initial_capital

    def test_reset_clears_trade_history(self, risk_mgr):
        risk_mgr.trade_history = [-100, -200, 50]
        risk_mgr.reset()
        assert len(risk_mgr.trade_history) == 0


# ─────────────────────────────────────────────
#  update_state circuit breaker integration
# ─────────────────────────────────────────────


class TestUpdateStateCircuitBreaker:
    def test_halt_triggered_via_update_state_drawdown(self, risk_mgr):
        """update_state() should automatically trip the halt flag on >2% drawdown."""
        risk_mgr.state.peak_equity = 100_000.0
        risk_mgr.update_state(current_equity=97_000.0)
        assert risk_mgr.should_halt_trading() is True

    def test_consecutive_losses_accumulated_via_update_state(self, risk_mgr):
        for _ in range(3):
            risk_mgr.update_state(current_equity=99_000.0, trade_result=-100.0)
        assert risk_mgr.state.consecutive_losses == 3

    def test_win_resets_consecutive_losses(self, risk_mgr):
        risk_mgr.update_state(current_equity=99_000.0, trade_result=-100.0)
        risk_mgr.update_state(current_equity=99_500.0, trade_result=200.0)
        assert risk_mgr.state.consecutive_losses == 0
