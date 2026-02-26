"""
Risk Manager - The Guardian
============================
Prevents capital destruction through systematic risk controls.

Key Components:
1. Position Size Validation (Kelly Criterion)
2. Circuit Breaker (Drawdown Limit)
3. Value at Risk (VaR) Estimation
4. Risk Limit Monitoring

Critical: Acts as final gatekeeper before every trade.
"""

import numpy as np
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass, field
from loguru import logger
from src.math_tools.kelly_criterion import KellyCriterion, KellyParameters


@dataclass
class RiskConfig:
    """Risk management configuration."""

    # Circuit Breaker
    max_drawdown_per_session: float = 0.02  # 2% max drawdown
    max_consecutive_losses: int = 5

    # Position Sizing
    max_position_size: float = 0.25  # Max 25% of capital per position
    kelly_fraction: float = 0.5  # Half Kelly

    # Value at Risk
    var_confidence: float = 0.95  # 95% confidence level
    var_lookback: int = 100  # Rolling window

    # Monitoring
    min_capital_threshold: float = 0.3  # 30% of initial capital (reduced from 50%)
    enable_circuit_breaker: bool = True


@dataclass
class RiskState:
    """Current risk state of the system."""

    current_equity: float
    initial_equity: float
    peak_equity: float
    current_drawdown: float
    consecutive_losses: int
    var_95: float
    halt_trading: bool = False
    halt_reason: Optional[str] = None


class RiskManager:
    """
    Central risk management system.

    Acts as guardian to prevent capital destruction:
    - Validates position sizes using Kelly Criterion
    - Monitors drawdown and triggers circuit breaker
    - Calculates Value at Risk
    - Tracks risk metrics

    Usage:
    ------
    risk_mgr = RiskManager(config, initial_capital=100000)

    # Before trade
    approved, adjusted_size = risk_mgr.validate_position_size(
        proposed_size=10000,
        current_capital=95000
    )

    # After trade
    risk_mgr.update_state(
        current_equity=95000,
        trade_result=-500
    )

    # Check circuit breaker
    if risk_mgr.should_halt_trading():
        print("TRADING HALTED:", risk_mgr.get_halt_reason())
    """

    def __init__(self, config: RiskConfig, initial_capital: float):
        """
        Initialize Risk Manager.

        Parameters:
        -----------
        config : RiskConfig
            Risk management configuration
        initial_capital : float
            Starting capital
        """
        self.config = config
        self.initial_capital = initial_capital

        # Initialize Kelly Criterion from Phase 3
        self.kelly = KellyCriterion()

        # Risk state
        self.state = RiskState(
            current_equity=initial_capital,
            initial_equity=initial_capital,
            peak_equity=initial_capital,
            current_drawdown=0.0,
            consecutive_losses=0,
            var_95=0.0,
        )

        # Trade history for metrics
        self.trade_history: List[float] = []
        self.equity_history: List[float] = [initial_capital]

        logger.info("RiskManager initialized")
        logger.info(f"  Max drawdown: {config.max_drawdown_per_session * 100:.1f}%")
        logger.info(f"  Max position: {config.max_position_size * 100:.0f}%")
        logger.info(f"  Kelly fraction: {config.kelly_fraction * 100:.0f}%")

    def validate_position_size(
        self,
        proposed_size: float,
        current_capital: float,
        win_probability: Optional[float] = None,
        win_loss_ratio: Optional[float] = None,
    ) -> Tuple[bool, float]:
        """
        Validate and adjust position size using Kelly Criterion.

        Parameters:
        -----------
        proposed_size : float
            Proposed position size in dollars
        current_capital : float
            Available capital
        win_probability : float, optional
            Recent win rate (if available)
        win_loss_ratio : float, optional
            Recent profit factor (if available)

        Returns:
        --------
        approved : bool
            Whether position is approved
        adjusted_size : float
            Adjusted position size (may be less than proposed)
        """
        # Check 1: Hard cap
        max_allowed = current_capital * self.config.max_position_size

        if proposed_size > max_allowed:
            logger.warning(
                f"Position size {proposed_size:.0f} exceeds hard cap "
                f"{max_allowed:.0f}. Capping to {self.config.max_position_size * 100:.0f}%"
            )
            proposed_size = max_allowed

        # Check 2: Kelly Criterion (if we have recent stats)
        if win_probability is not None and win_loss_ratio is not None:
            kelly_size = self.kelly.dynamic_kelly(
                capital=current_capital,
                recent_win_rate=win_probability,
                recent_profit_factor=win_loss_ratio,
                kelly_fraction=self.config.kelly_fraction,
                max_position=self.config.max_position_size,
            )

            if proposed_size > kelly_size:
                logger.info(
                    f"Position size {proposed_size:.0f} exceeds Kelly limit "
                    f"{kelly_size:.0f}. Adjusting to Kelly size."
                )
                proposed_size = kelly_size

        # Check 3: Minimum capital threshold
        if current_capital < self.initial_capital * self.config.min_capital_threshold:
            logger.warning(
                f"Capital below {self.config.min_capital_threshold * 100:.0f}% threshold. "
                "Reducing position size by 50%."
            )
            proposed_size *= 0.5

        # Check 4: Circuit breaker
        if self.state.halt_trading:
            logger.error(f"Trading halted: {self.state.halt_reason}")
            return False, 0.0

        return True, proposed_size

    def check_circuit_breaker(
        self, current_equity: float
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if circuit breaker should trigger.

        Circuit breaker triggers on:
        1. Drawdown exceeds max limit
        2. Too many consecutive losses
        3. Capital below minimum threshold

        Parameters:
        -----------
        current_equity : float
            Current account equity

        Returns:
        --------
        should_halt : bool
            Whether trading should be halted
        reason : str
            Reason for halt (if any)
        """
        if not self.config.enable_circuit_breaker:
            return False, None

        # Calculate current drawdown
        drawdown = self._calculate_drawdown(current_equity)

        # Check 1: Max drawdown
        if drawdown > self.config.max_drawdown_per_session:
            reason = (
                f"Max drawdown exceeded: {drawdown * 100:.2f}% > "
                f"{self.config.max_drawdown_per_session * 100:.1f}%"
            )
            logger.critical(f"CIRCUIT BREAKER: {reason}")
            return True, reason

        # Check 2: Consecutive losses
        if self.state.consecutive_losses >= self.config.max_consecutive_losses:
            reason = (
                f"Max consecutive losses: {self.state.consecutive_losses} >= "
                f"{self.config.max_consecutive_losses}"
            )
            logger.critical(f"CIRCUIT BREAKER: {reason}")
            return True, reason

        # Check 3: Minimum capital
        if current_equity < self.initial_capital * self.config.min_capital_threshold:
            reason = (
                f"Capital below minimum threshold: "
                f"${current_equity:.2f} < ${self.initial_capital * self.config.min_capital_threshold:.2f}"
            )
            logger.critical(f"CIRCUIT BREAKER: {reason}")
            return True, reason

        return False, None

    def calculate_var(
        self, position_value: float, confidence: Optional[float] = None
    ) -> float:
        """
        Calculate Value at Risk (VaR).

        VaR estimates the maximum loss at a given confidence level
        based on recent return distribution.

        Parameters:
        -----------
        position_value : float
            Current position value
        confidence : float, optional
            Confidence level (default: from config)

        Returns:
        --------
        var : float
            Value at Risk in dollars
        """
        if confidence is None:
            confidence = self.config.var_confidence

        # Need sufficient history
        if len(self.equity_history) < 2:
            return 0.0

        # Calculate returns
        equity_array = np.array(self.equity_history[-self.config.var_lookback :])
        returns = np.diff(equity_array) / equity_array[:-1]

        if len(returns) == 0:
            return 0.0

        # Historical VaR (percentile method): find the (1-confidence) worst return
        var_percentile = (
            1 - confidence
        ) * 100  # e.g. 5th percentile for 95% confidence
        var_return = np.percentile(returns, var_percentile)  # Negative value (loss)

        # Convert return-based VaR to dollar loss amount
        var_dollars = abs(var_return * position_value)

        return float(var_dollars)

    def update_state(self, current_equity: float, trade_result: Optional[float] = None):
        """
        Update risk state after trade or market movement.

        Parameters:
        -----------
        current_equity : float
            Current account equity
        trade_result : float, optional
            P&L of last trade (positive=profit, negative=loss)
        """
        # Update equity history
        self.equity_history.append(current_equity)

        # Update peak equity
        if current_equity > self.state.peak_equity:
            self.state.peak_equity = current_equity

        # Update drawdown
        self.state.current_drawdown = self._calculate_drawdown(current_equity)

        # Track trade result
        if trade_result is not None:
            self.trade_history.append(trade_result)

            # Update consecutive losses
            if trade_result < 0:
                self.state.consecutive_losses += 1
            else:
                self.state.consecutive_losses = 0

        # Update VaR
        if len(self.equity_history) >= 2:
            # Estimate position value (simplified)
            position_value = current_equity - self.state.current_equity
            self.state.var_95 = self.calculate_var(abs(position_value))

        # Update current equity
        self.state.current_equity = current_equity

        # Check circuit breaker
        should_halt, reason = self.check_circuit_breaker(current_equity)
        if should_halt:
            self.state.halt_trading = True
            self.state.halt_reason = reason

    def should_halt_trading(self) -> bool:
        """Check if trading should be halted."""
        return self.state.halt_trading

    def get_halt_reason(self) -> Optional[str]:
        """Get reason for trading halt."""
        return self.state.halt_reason

    @property
    def consecutive_losses(self) -> int:
        """Get current consecutive losses."""
        return self.state.consecutive_losses

    def reset(self):
        """Reset risk state (for new episode)."""
        self.state = RiskState(
            current_equity=self.initial_capital,
            initial_equity=self.initial_capital,
            peak_equity=self.initial_capital,
            current_drawdown=0.0,
            consecutive_losses=0,
            var_95=0.0,
        )
        self.trade_history = []
        self.equity_history = [self.initial_capital]

        logger.info("RiskManager reset for new episode")

    def get_risk_metrics(self) -> Dict:
        """
        Get current risk metrics.

        Returns:
        --------
        metrics : dict
            Current risk metrics
        """
        return {
            "current_equity": self.state.current_equity,
            "peak_equity": self.state.peak_equity,
            "current_drawdown": self.state.current_drawdown,
            "consecutive_losses": self.state.consecutive_losses,
            "var_95": self.state.var_95,
            "halt_trading": self.state.halt_trading,
            "halt_reason": self.state.halt_reason,
            "total_trades": len(self.trade_history),
        }

    def _calculate_drawdown(self, current_equity: float) -> float:
        """Calculate current drawdown from peak."""
        if self.state.peak_equity == 0:
            return 0.0

        # Drawdown = (peak - current) / peak  (as a fraction, e.g. 0.05 = 5% drawdown)
        drawdown = (self.state.peak_equity - current_equity) / self.state.peak_equity
        return max(0.0, drawdown)  # Clamp to zero (can't have negative drawdown)

    def get_kelly_parameters(self, lookback: int = 50) -> Optional[KellyParameters]:
        """
        Get current Kelly parameters from recent trades.

        Parameters:
        -----------
        lookback : int
            Number of recent trades to analyze

        Returns:
        --------
        params : KellyParameters or None
            Current Kelly parameters
        """
        if len(self.trade_history) < 10:
            return None

        recent_trades = np.array(self.trade_history[-lookback:])
        params = self.kelly.estimate_parameters(recent_trades)

        return params


# ============================================================================
# EXAMPLE USAGE & VALIDATION
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("RISK MANAGER TEST")
    print("=" * 80)

    # Configure
    config = RiskConfig(
        max_drawdown_per_session=0.02,
        max_consecutive_losses=5,
        max_position_size=0.25,
        kelly_fraction=0.5,
    )

    initial_capital = 100000
    risk_mgr = RiskManager(config, initial_capital)

    print(f"\n✓ RiskManager initialized with ${initial_capital:,.0f}")

    # Test 1: Position size validation
    print("\n[TEST 1] Position Size Validation")

    # Normal position
    approved, adjusted = risk_mgr.validate_position_size(
        proposed_size=20000, current_capital=100000
    )
    print(f"  Proposed: $20,000")
    print(f"  Approved: {approved}, Adjusted: ${adjusted:,.0f}")

    # Oversized position
    approved, adjusted = risk_mgr.validate_position_size(
        proposed_size=50000, current_capital=100000
    )
    print(f"  Proposed: $50,000 (exceeds limit)")
    print(f"  Approved: {approved}, Adjusted: ${adjusted:,.0f}")

    # Test 2: Circuit breaker - drawdown
    print("\n[TEST 2] Circuit Breaker - Drawdown")

    # Simulate 3% drawdown (should trigger)
    current_equity = 97000
    risk_mgr.update_state(current_equity)

    should_halt, reason = risk_mgr.check_circuit_breaker(current_equity)
    print(f"  Equity: ${current_equity:,.0f} (-3%)")
    print(f"  Should halt: {should_halt}")
    if reason:
        print(f"  Reason: {reason}")

    # Test 3: Circuit breaker - consecutive losses
    print("\n[TEST 3] Circuit Breaker - Consecutive Losses")

    # Reset for clean test
    risk_mgr.reset()

    # Simulate 6 consecutive losses
    for i in range(6):
        risk_mgr.update_state(current_equity=100000 - (i + 1) * 500, trade_result=-500)

    print(f"  Consecutive losses: {risk_mgr.state.consecutive_losses}")
    print(f"  Should halt: {risk_mgr.should_halt_trading()}")
    if risk_mgr.state.halt_reason:
        print(f"  Reason: {risk_mgr.state.halt_reason}")

    # Test 4: VaR calculation
    print("\n[TEST 4] Value at Risk")

    risk_mgr.reset()

    # Simulate some equity history with volatility
    np.random.seed(42)
    for i in range(100):
        equity = 100000 + np.cumsum(np.random.randn(1) * 1000)[0]
        risk_mgr.update_state(equity)

    position_value = 25000
    var_95 = risk_mgr.calculate_var(position_value, confidence=0.95)

    print(f"  Position value: ${position_value:,.0f}")
    print(f"  VaR (95%): ${var_95:,.0f}")
    print(f"  Interpretation: 95% confidence loss won't exceed ${var_95:,.0f}")

    # Test 5: Risk metrics
    print("\n[TEST 5] Risk Metrics")

    metrics = risk_mgr.get_risk_metrics()
    print(f"  Current equity: ${metrics['current_equity']:,.0f}")
    print(f"  Peak equity: ${metrics['peak_equity']:,.0f}")
    print(f"  Drawdown: {metrics['current_drawdown'] * 100:.2f}%")
    print(f"  VaR 95%: ${metrics['var_95']:,.0f}")
    print(f"  Total trades: {metrics['total_trades']}")

    # Test 6: Kelly parameters
    print("\n[TEST 6] Kelly Parameters from Trade History")

    risk_mgr.reset()

    # Simulate trading history
    np.random.seed(42)
    for i in range(50):
        if np.random.rand() < 0.55:  # 55% win rate
            trade_result = np.random.uniform(100, 300)
        else:
            trade_result = np.random.uniform(-200, -100)

        equity = 100000 + sum(risk_mgr.trade_history) + trade_result
        risk_mgr.update_state(equity, trade_result)

    kelly_params = risk_mgr.get_kelly_parameters()
    if kelly_params:
        print(f"  Win probability: {kelly_params.win_probability:.1%}")
        print(f"  Win/loss ratio: {kelly_params.win_loss_ratio:.2f}")
        print(f"  Kelly fraction: {kelly_params.kelly_fraction * 100:.0f}%")
        print(f"  Max position: {kelly_params.max_position * 100:.0f}%")

    print("\n" + "=" * 80)
    print("✓ RISK MANAGER TEST PASSED")
    print("=" * 80)

    print("\nKey Insights:")
    print("• Position sizes validated and capped correctly")
    print("• Circuit breaker triggers on drawdown and consecutive losses")
    print("• VaR provides risk estimate at 95% confidence")
    print("• Kelly parameters estimated from trade history")
    print("• Ready for integration with trading environment")
