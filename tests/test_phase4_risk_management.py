"""
Phase 4: Risk Management Tests
==============================

Comprehensive tests for the risk management system.
"""

import pytest
import numpy as np

from risk.risk_manager import RiskManager, RiskConfig, RiskState
from risk.risk_metrics_logger import RiskMetricsLogger
from math_tools.kelly_criterion import KellyCriterion, KellyParameters


class TestRiskConfig:
    """Test RiskConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = RiskConfig()
        assert config.max_drawdown_per_session == 0.02  # 2% daily drawdown limit
        assert config.max_consecutive_losses == 5  # Stop after 5 losing trades
        assert config.max_position_size == 0.25  # Max 25% of capital per position
        assert config.kelly_fraction == 0.5  # Half-Kelly position sizing
        assert config.enable_circuit_breaker is True  # Circuit breaker on by default

    def test_custom_config(self):
        """Test custom configuration values."""
        config = RiskConfig(
            max_drawdown_per_session=0.10,
            max_consecutive_losses=3,
            max_position_size=0.5,
            kelly_fraction=0.25,
            enable_circuit_breaker=False,
        )
        assert config.max_drawdown_per_session == 0.10
        assert config.max_consecutive_losses == 3
        assert config.max_position_size == 0.5
        assert config.kelly_fraction == 0.25
        assert config.enable_circuit_breaker is False


class TestRiskManager:
    """Test RiskManager functionality."""

    @pytest.fixture
    def risk_manager(self):
        """Create a fresh RiskManager instance."""
        config = RiskConfig(
            max_drawdown_per_session=0.10,  # 10% drawdown limit
            max_consecutive_losses=3,
            max_position_size=0.5,
            kelly_fraction=0.5,
            enable_circuit_breaker=True,
        )
        return RiskManager(config, initial_capital=100000)

    def test_initialization(self, risk_manager):
        """Test proper initialization."""
        assert risk_manager.config is not None
        assert risk_manager.initial_capital == 100000
        assert risk_manager.state.current_equity == 100000
        assert risk_manager.state.initial_equity == 100000
        assert risk_manager.state.peak_equity == 100000  # Peak starts at initial
        assert risk_manager.state.consecutive_losses == 0
        assert not risk_manager.state.halt_trading  # Trading allowed at start

    def test_update_state_with_profit(self, risk_manager):
        """Test state update with profitable trade."""
        risk_manager.update_state(105000, 5000)  # Equity increased by $5000
        assert risk_manager.state.current_equity == 105000
        assert risk_manager.state.peak_equity == 105000  # New peak
        assert risk_manager.state.consecutive_losses == 0  # Loss streak reset

    def test_update_state_with_loss(self, risk_manager):
        """Test state update with losing trade."""
        risk_manager.update_state(95000, -5000)  # Equity dropped by $5000
        assert risk_manager.state.current_equity == 95000
        assert risk_manager.state.peak_equity == 100000  # Peak unchanged
        assert risk_manager.state.consecutive_losses == 1  # Loss streak started

    def test_consecutive_losses_tracking(self, risk_manager):
        """Test consecutive losses tracking."""
        # Three consecutive losses
        risk_manager.update_state(99000, -1000)
        risk_manager.update_state(98000, -1000)
        risk_manager.update_state(97000, -1000)

        assert risk_manager.state.consecutive_losses == 3
        assert risk_manager.should_halt_trading()  # Meets max_consecutive_losses=3

    def test_circuit_breaker_drawdown(self, risk_manager):
        """Test circuit breaker triggers on drawdown."""
        # 15% drawdown (exceeds 10% limit)
        risk_manager.update_state(85000, -15000)

        assert risk_manager.state.halt_trading  # Circuit breaker fired
        assert risk_manager.should_halt_trading()

    def test_circuit_breaker_no_trigger(self, risk_manager):
        """Test circuit breaker doesn't trigger prematurely."""
        # 5% drawdown (within 10% limit)
        risk_manager.update_state(95000, -5000)

        assert not risk_manager.state.halt_trading  # Still within limits
        assert not risk_manager.should_halt_trading()

    def test_reset(self, risk_manager):
        """Test reset functionality."""
        risk_manager.update_state(85000, -15000)
        assert risk_manager.state.halt_trading

        risk_manager.reset()  # Reset for new session
        assert risk_manager.state.current_equity == 100000
        assert risk_manager.state.peak_equity == 100000
        assert not risk_manager.state.halt_trading  # Trading re-enabled
        assert risk_manager.state.consecutive_losses == 0

    def test_get_halt_reason_drawdown(self, risk_manager):
        """Test halt reason for drawdown."""
        risk_manager.update_state(85000, -15000)
        reason = risk_manager.get_halt_reason()

        assert reason is not None
        assert len(reason) > 0  # Should return a descriptive message

    def test_get_halt_reason_consecutive_losses(self, risk_manager):
        """Test halt reason for consecutive losses."""
        for _ in range(3):
            risk_manager.update_state(risk_manager.state.current_equity - 1000, -1000)

        reason = risk_manager.get_halt_reason()
        assert reason is not None
        assert len(reason) > 0

    def test_validate_position_size_approved(self, risk_manager):
        """Test position size validation - approved."""
        approved, size = risk_manager.validate_position_size(
            proposed_size=10000,
            current_capital=100000,
            win_probability=0.6,
            win_loss_ratio=2.0,
        )

        assert approved is True
        assert size > 0  # A valid size was computed

    def test_validate_position_size_rejected(self, risk_manager):
        """Test position size validation - rejected."""
        # Trigger circuit breaker first
        risk_manager.update_state(85000, -15000)

        approved, size = risk_manager.validate_position_size(
            proposed_size=10000, current_capital=85000
        )

        assert approved is False  # Halted = no new positions
        assert size == 0

    def test_validate_position_size_kelly_capped(self, risk_manager):
        """Test Kelly-based position sizing is capped."""
        approved, size = risk_manager.validate_position_size(
            proposed_size=60000,  # 60% of capital (exceeds max_position_size=50%)
            current_capital=100000,
            win_probability=0.8,  # High win rate
            win_loss_ratio=3.0,  # Good risk/reward
        )

        assert approved is True
        # Should be capped at max_position_size (50%)
        assert size <= 50000


class TestKellyCriterion:
    """Test Kelly Criterion calculations."""

    @pytest.fixture
    def kelly(self):
        """Create KellyCriterion instance."""
        return KellyCriterion()

    def test_calculate_kelly_fraction(self, kelly):
        """Test Kelly fraction calculation."""
        # 60% win rate, 2:1 reward/risk
        kelly_fraction = kelly.calculate_kelly_fraction(
            win_prob=0.6, win_loss_ratio=2.0
        )

        # Kelly = (bp - q) / b = (0.6*2 - 0.4) / 2 = 0.4
        assert abs(kelly_fraction - 0.4) < 0.01

    def test_calculate_position_size(self, kelly):
        """Test position size calculation."""
        params = KellyParameters(
            win_probability=0.6, win_loss_ratio=2.0, kelly_fraction=0.4
        )

        position_size = kelly.calculate_position_size(capital=100000, params=params)

        assert position_size > 0
        assert position_size <= 100000  # Cannot bet more than total capital

    def test_estimate_parameters_with_trades(self, kelly):
        """Test parameter estimation from trade history."""
        trades = [1000, -500, 1500, -300, 2000, -400]  # Mix of wins and losses

        params = kelly.estimate_parameters(trades)

        assert isinstance(params, KellyParameters)
        assert 0 < params.win_probability < 1  # Valid probability
        assert params.win_loss_ratio > 0
        assert params.kelly_fraction is not None

    def test_estimate_parameters_insufficient_data(self, kelly):
        """Test parameter estimation with insufficient data (single trade = all wins, no losses).

        The implementation returns default parameters rather than None when there are no
        wins or no losses in the history (graceful fallback). We verify it returns a
        valid KellyParameters object with sensible defaults.
        """
        trades = [1000]  # Only one trade — no losses to compute loss ratio

        params = kelly.estimate_parameters(trades)

        # Implementation returns defaults (not None) when wins or losses are missing
        assert params is not None
        assert isinstance(params, KellyParameters)
        assert 0 < params.win_probability < 1  # Still a valid probability

    def test_dynamic_kelly(self, kelly):
        """Test dynamic Kelly calculation."""
        position_size = kelly.dynamic_kelly(
            capital=100000, recent_win_rate=0.55, recent_profit_factor=1.8
        )

        assert position_size > 0
        assert position_size <= 100000  # Cannot exceed available capital


class TestRiskMetricsLogger:
    """Test RiskMetricsLogger functionality."""

    @pytest.fixture
    def metrics_logger(self):
        """Create RiskMetricsLogger instance."""
        return RiskMetricsLogger(lookback=50, risk_free_rate=0.0)

    def test_initialization(self, metrics_logger):
        """Test proper initialization."""
        assert metrics_logger.lookback == 50
        assert metrics_logger.risk_free_rate == 0.0

    def test_update_equity(self, metrics_logger):
        """Test equity update."""
        metrics_logger.update(equity=100000)
        assert len(metrics_logger.equity_history) == 1

    def test_update_with_trade(self, metrics_logger):
        """Test update with trade result."""
        metrics_logger.update(equity=100000, trade_result=1000)
        assert len(metrics_logger.trade_history) == 1  # Trade result recorded

    def test_sharpe_ratio_calculation(self, metrics_logger):
        """Test Sharpe ratio calculation."""
        # Add some equity data with positive returns
        for i in range(60):
            equity = 100000 + i * 100  # Monotonically increasing equity
            metrics_logger.update(equity=equity)

        metrics = metrics_logger.get_current_metrics()

        assert "sharpe_ratio" in metrics

    def test_drawdown_calculation(self, metrics_logger):
        """Test drawdown calculation."""
        # Peak at 110000, then drop
        for equity in [100000, 105000, 110000, 105000, 100000, 95000]:
            metrics_logger.update(equity=equity)

        metrics = metrics_logger.get_current_metrics()

        assert "max_drawdown" in metrics
        # max_drawdown is stored as a positive fraction (e.g. 0.136 means ~13.6% drawdown)
        assert metrics["max_drawdown"] >= 0

    def test_calmar_ratio_calculation(self, metrics_logger):
        """Test Calmar ratio calculation."""
        for i in range(60):
            equity = 100000 + i * 50
            metrics_logger.update(equity=equity)

        metrics = metrics_logger.get_current_metrics()

        assert "calmar_ratio" in metrics  # Calmar = annualized return / max drawdown

    def test_reset(self, metrics_logger):
        """Test reset functionality."""
        metrics_logger.update(equity=100000, trade_result=1000)
        assert len(metrics_logger.equity_history) == 1

        metrics_logger.reset()
        assert len(metrics_logger.equity_history) == 0  # History cleared
        assert len(metrics_logger.trade_history) == 0


class TestRiskManagerIntegration:
    """Integration tests for risk management."""

    def test_full_trading_session(self):
        """Test a complete trading session with risk management."""
        config = RiskConfig(
            max_drawdown_per_session=0.15,
            max_consecutive_losses=5,
            max_position_size=0.3,
            enable_circuit_breaker=True,
        )

        risk_manager = RiskManager(config, initial_capital=100000)

        # Simulate trading session
        trades = [1000, 1500, -500, -300, 2000, -400, 1800, -600, -700, -800]

        for trade_pnl in trades:
            if risk_manager.should_halt_trading():
                break

            new_capital = risk_manager.state.current_equity + trade_pnl
            risk_manager.update_state(new_capital, trade_pnl)

            # Validate each trade
            approved, _ = risk_manager.validate_position_size(
                proposed_size=10000, current_capital=risk_manager.state.current_equity
            )

            if risk_manager.should_halt_trading():
                assert not approved, "Should reject trades when halted"

        # Verify final state
        expected_capital = 100000 + sum(trades)
        assert risk_manager.state.current_equity == expected_capital

    def test_risk_metrics_with_logger(self):
        """Test risk metrics integration with RiskManager."""
        risk_manager = RiskManager(RiskConfig(), initial_capital=100000)
        metrics_logger = RiskMetricsLogger(lookback=50)

        # Simulate trading
        equity_values = [100000, 101000, 102000, 101500, 103000, 102500, 104000]

        for equity in equity_values:
            pnl = equity - risk_manager.state.current_equity
            risk_manager.update_state(equity, pnl)
            metrics_logger.update(equity=equity, trade_result=pnl if pnl != 0 else None)

        # Get risk metrics
        metrics = metrics_logger.get_current_metrics()

        assert "sharpe_ratio" in metrics
        assert "max_drawdown" in metrics

        # Verify RiskManager still functioning
        assert not risk_manager.should_halt_trading()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
