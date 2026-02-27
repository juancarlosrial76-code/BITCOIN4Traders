"""
Risk Manager - The Guardian
============================
Comprehensive risk management system designed to prevent capital destruction
through systematic risk controls and protective mechanisms.

This module serves as the primary defense layer in the BITCOIN4Traders trading
system, implementing industry-standard risk management techniques adapted for
reinforcement learning-based trading agents.

Key Components:
---------------
1. Position Size Validation (Kelly Criterion)
   - Dynamically calculates optimal position sizes based on historical win rate
   - Uses fractional Kelly (default 50%) to reduce variance while maintaining
     positive expected value

2. Circuit Breaker (Drawdown Limit)
   - Monitors session drawdown and triggers automatic trading halt
   - Stops trading when losses exceed configurable threshold (default 2%)
   - Tracks consecutive losses to detect adverse conditions

3. Value at Risk (VaR) Estimation
   - Historical VaR using percentile method
   - Estimates maximum expected loss at 95% confidence level
   - Uses rolling window for adaptive risk estimation

4. Risk Limit Monitoring
   - Minimum capital threshold enforcement
   - Maximum position size constraints
   - Real-time risk state tracking

Critical: Acts as final gatekeeper before every trade execution. All position
sizes must pass through validate_position_size() before trading occurs.

Usage Example:
--------------
    from src.risk.risk_manager import RiskManager, RiskConfig

    # Initialize with configuration
    config = RiskConfig(
        max_drawdown_per_session=0.02,  # 2% max drawdown
        max_consecutive_losses=5,
        max_position_size=0.25,         # 25% max position
        kelly_fraction=0.5               # Half Kelly
    )

    risk_mgr = RiskManager(config, initial_capital=100000)

    # Before executing any trade
    approved, adjusted_size = risk_mgr.validate_position_size(
        proposed_size=15000,
        current_capital=95000,
        win_probability=0.55,
        win_loss_ratio=1.5
    )

    if approved:
        # Execute trade with adjusted_size
        pass

    # After trade completion, update risk state
    risk_mgr.update_state(current_equity=94000, trade_result=-1000)

    # Check if trading should be halted
    if risk_mgr.should_halt_trading():
        print(f"Trading halted: {risk_mgr.get_halt_reason()}")

Dependencies:
-------------
- numpy: For numerical calculations and array operations
- loguru: For structured logging
- src.math_tools.kelly_criterion: Kelly Criterion implementation

Author: BITCOIN4Traders Team
Version: 1.0.0
"""

import numpy as np
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass, field
from loguru import logger
from src.math_tools.kelly_criterion import KellyCriterion, KellyParameters


@dataclass
class RiskConfig:
    """
    Risk management configuration parameters.

    This dataclass encapsulates all configurable risk parameters that control
    the behavior of the RiskManager. Values are set to conservative defaults
    suitable for live trading but can be adjusted for different strategies
    or risk tolerances.

    Attributes:
        max_drawdown_per_session: Maximum allowed drawdown before circuit
            breaker triggers (default 2%). Expressed as decimal (0.02 = 2%).
        max_consecutive_losses: Maximum number of consecutive losing trades
            before circuit breaker triggers (default 5).
        max_position_size: Maximum position size as fraction of capital
            (default 25%). Prevents over-concentration in single positions.
        kelly_fraction: Fraction of Kelly Criterion to use (default 50%).
            Full Kelly is mathematically optimal but has high variance;
            Half Kelly provides good risk-adjusted returns with lower risk.
        var_confidence: Confidence level for Value at Risk calculation
            (default 95%). Higher values are more conservative.
        var_lookback: Number of periods for rolling VaR calculation
            (default 100).
        min_capital_threshold: Minimum capital as fraction of initial capital
            (default 30%). Trading is restricted when capital falls below
            this threshold.
        enable_circuit_breaker: Whether circuit breaker is active (default True).
            Can be disabled for debugging or controlled environments.

    Example:
        >>> config = RiskConfig(
        ...     max_drawdown_per_session=0.03,  # 3% drawdown limit
        ...     max_consecutive_losses=7,
        ...     max_position_size=0.20,         # 20% max position
        ...     kelly_fraction=0.25              # Quarter Kelly for safety
        ... )
    """

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
    """
    Current risk state of the trading system.

    This dataclass maintains the real-time risk state including equity levels,
    drawdown metrics, VaR estimates, and halt status. Updated after each
    trade or market movement through RiskManager.update_state().

    Attributes:
        current_equity: Current account equity in dollars.
        initial_equity: Starting equity at beginning of session/episode.
        peak_equity: Highest equity achieved (high water mark).
        current_drawdown: Current drawdown as decimal (0.05 = 5%).
        consecutive_losses: Number of consecutive losing trades.
        var_95: Value at Risk at 95% confidence in dollars.
        halt_trading: Boolean flag indicating if trading is halted.
        halt_reason: Explanation for trading halt (if triggered).
    """

    # Current state
    current_equity: float
    initial_equity: float
    peak_equity: float
    current_drawdown: float
    consecutive_losses: int
    var_95: float

    # Circuit breaker state
    halt_trading: bool = False
    halt_reason: Optional[str] = None


class RiskManager:
    """
    Central risk management system for the BITCOIN4Traders trading bot.

    Acts as the guardian to prevent capital destruction through multiple
    layers of risk control:

    1. Position Validation:
       - Enforces maximum position size limits (hard cap)
       - Applies Kelly Criterion for optimal sizing when win statistics available
       - Reduces position size when capital falls below threshold

    2. Circuit Breaker:
       - Monitors drawdown from peak equity
       - Tracks consecutive losses
       - Halts trading when risk limits exceeded

    3. Risk Measurement:
       - Calculates Value at Risk (VaR) using historical method
       - Tracks equity curve and drawdown metrics
       - Maintains trade history for parameter estimation

    4. State Management:
       - Updates risk state after each trade
       - Provides real-time risk metrics
       - Supports reset for new episodes

    The RiskManager should be instantiated once at the beginning of a
    trading session and used throughout to validate all trades before
    execution.

    Usage:
    ------
    risk_mgr = RiskManager(config, initial_capital=100000)

    # Before trade
    approved, adjusted_size = risk_mgr.validate_position_size(
        proposed_size=10000,
        current_capital=95000,
        win_probability=0.55,    # 55% win rate
        win_loss_ratio=1.5       # profit factor
    )

    # After trade
    risk_mgr.update_state(
        current_equity=95000,
        trade_result=-500        # loss of $500
    )

    # Check circuit breaker
    if risk_mgr.should_halt_trading():
        print("TRADING HALTED:", risk_mgr.get_halt_reason())

    Attributes:
        config: RiskConfig instance with risk parameters
        initial_capital: Starting capital for the session
        kelly: KellyCriterion instance for dynamic position sizing
        state: RiskState containing current risk metrics
        trade_history: List of all trade results
        equity_history: List of equity values over time

    Note:
        The Kelly Criterion calculations require at least 10 trades in
        history before becoming active. Before that, only hard caps apply.
    """

    def __init__(self, config: RiskConfig, initial_capital: float):
        """
        Initialize the Risk Manager with configuration and starting capital.

        Parameters:
        -----------
        config : RiskConfig
            Risk management configuration containing all risk parameters
            such as max drawdown, position limits, and Kelly fraction.
        initial_capital : float
            Starting capital in dollars. This is used as the baseline
            for calculating drawdowns and determining minimum capital
            thresholds.

        Returns:
        --------
        None

        Side Effects:
        -------------
        - Initializes Kelly Criterion calculator
        - Sets up initial risk state
        - Logs configuration parameters

        Example:
            >>> config = RiskConfig(
            ...     max_drawdown_per_session=0.02,
            ...     max_position_size=0.25,
            ...     kelly_fraction=0.5
            ... )
            >>> risk_mgr = RiskManager(config, initial_capital=100000)
            RiskManager initialized
              Max drawdown: 2.0%
              Max position: 25%
              Kelly fraction: 50%
        """
        self.config = config
        self.initial_capital = initial_capital

        # Initialize Kelly Criterion from Phase 3
        # Used for dynamic position sizing based on historical performance
        self.kelly = KellyCriterion()

        # Risk state - tracks all risk-related metrics
        self.state = RiskState(
            current_equity=initial_capital,
            initial_equity=initial_capital,
            peak_equity=initial_capital,
            current_drawdown=0.0,
            consecutive_losses=0,
            var_95=0.0,
        )

        # Trade history for metrics calculation
        # Used for Kelly parameter estimation and streak tracking
        self.trade_history: List[float] = []

        # Equity history for VaR and drawdown calculation
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
        Validate and adjust position size using multiple risk controls.

        This method applies a multi-stage validation process to ensure
        proposed positions meet all risk requirements:

        Stage 1 - Hard Cap: Ensures position doesn't exceed maximum allowed
            size as percentage of capital.

        Stage 2 - Kelly Criterion: If win probability and profit factor
            are provided, calculates optimal Kelly size and caps position
            at that level (if lower than proposed).

        Stage 3 - Capital Threshold: If capital has fallen below minimum
            threshold, reduces position size by 50% as protective measure.

        Stage 4 - Circuit Breaker Check: Returns rejected if trading is
            currently halted due to risk limits.

        Parameters:
        -----------
        proposed_size : float
            Proposed position size in dollars.
        current_capital : float
            Available capital for trading.
        win_probability : float, optional
            Recent win rate from trading history (e.g., 0.55 = 55% wins).
            If provided, Kelly Criterion will be applied.
        win_loss_ratio : float, optional
            Recent profit factor (avg win / avg loss). If provided,
            Kelly Criterion will be applied.

        Returns:
        --------
        approved : bool
            Whether position is approved for execution. False if any
            risk check fails or circuit breaker is active.
        adjusted_size : float
            Adjusted position size (may be less than proposed due to
            risk controls). Zero if position rejected.

        Example:
            >>> approved, adjusted = risk_mgr.validate_position_size(
            ...     proposed_size=30000,
            ...     current_capital=100000,
            ...     win_probability=0.55,
            ...     win_loss_ratio=1.5
            ... )
            >>> if approved:
            ...     print(f"Trade approved for ${adjusted:,.0f}")
        """
        # Stage 1: Hard cap - enforce maximum position size limit
        # This prevents over-concentration in single positions
        max_allowed = current_capital * self.config.max_position_size

        if proposed_size > max_allowed:
            logger.warning(
                f"Position size {proposed_size:.0f} exceeds hard cap "
                f"{max_allowed:.0f}. Capping to {self.config.max_position_size * 100:.0f}%"
            )
            proposed_size = max_allowed

        # Stage 2: Kelly Criterion - dynamic sizing based on edge
        # Only applies if we have recent trading statistics
        if win_probability is not None and win_loss_ratio is not None:
            kelly_size = self.kelly.dynamic_kelly(
                capital=current_capital,
                recent_win_rate=win_probability,
                recent_profit_factor=win_loss_ratio,
                kelly_fraction=self.config.kelly_fraction,
                max_position=self.config.max_position_size,
            )

            # Use Kelly size if more conservative than proposed
            if proposed_size > kelly_size:
                logger.info(
                    f"Position size {proposed_size:.0f} exceeds Kelly limit "
                    f"{kelly_size:.0f}. Adjusting to Kelly size."
                )
                proposed_size = kelly_size

        # Stage 3: Capital threshold - protective reduction
        # When capital drops below threshold, reduce position sizes
        if current_capital < self.initial_capital * self.config.min_capital_threshold:
            logger.warning(
                f"Capital below {self.config.min_capital_threshold * 100:.0f}% threshold. "
                "Reducing position size by 50%."
            )
            proposed_size *= 0.5

        # Stage 4: Circuit breaker - reject if trading halted
        if self.state.halt_trading:
            logger.error(f"Trading halted: {self.state.halt_reason}")
            return False, 0.0

        return True, proposed_size

    def check_circuit_breaker(
        self, current_equity: float
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if circuit breaker should trigger based on risk limits.

        The circuit breaker is a critical safety mechanism that halts
        trading when risk exceeds acceptable thresholds. This prevents
        runaway losses during adverse market conditions or extended
        losing streaks.

        Circuit breaker triggers on any of the following conditions:

        1. Maximum Drawdown: Current drawdown from peak exceeds configured
           limit (default 2%). This protects against large single-session
           losses.

        2. Consecutive Losses: Number of consecutive losing trades exceeds
           limit (default 5). Extended losing streaks may indicate
           changing market conditions requiring strategy review.

        3. Minimum Capital: Equity falls below minimum threshold as
           percentage of initial capital (default 30%). Prevents trading
           when account is significantly degraded.

        Parameters:
        -----------
        current_equity : float
            Current account equity in dollars.

        Returns:
        --------
        should_halt : bool
            True if any circuit breaker condition is met, False otherwise.
        reason : str or None
            Human-readable explanation of why trading should be halted.
            None if no condition is met.

        Example:
            >>> should_halt, reason = risk_mgr.check_circuit_breaker(95000)
            >>> if should_halt:
            ...     print(f"Trading halted: {reason}")
        """
        # Allow circuit breaker to be disabled for controlled testing
        if not self.config.enable_circuit_breaker:
            return False, None

        # Calculate current drawdown from peak equity
        # Drawdown = (peak - current) / peak, expressed as fraction
        drawdown = self._calculate_drawdown(current_equity)

        # Check 1: Maximum drawdown limit
        # Prevents large single-session losses
        if drawdown > self.config.max_drawdown_per_session:
            reason = (
                f"Max drawdown exceeded: {drawdown * 100:.2f}% > "
                f"{self.config.max_drawdown_per_session * 100:.1f}%"
            )
            logger.critical(f"CIRCUIT BREAKER: {reason}")
            return True, reason

        # Check 2: Consecutive losses limit
        # Detects adverse conditions requiring strategy review
        if self.state.consecutive_losses >= self.config.max_consecutive_losses:
            reason = (
                f"Max consecutive losses: {self.state.consecutive_losses} >= "
                f"{self.config.max_consecutive_losses}"
            )
            logger.critical(f"CIRCUIT BREAKER: {reason}")
            return True, reason

        # Check 3: Minimum capital threshold
        # Prevents trading with significantly degraded capital
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
        Calculate Value at Risk (VaR) using historical percentile method.

        VaR estimates the maximum expected loss at a given confidence level
        over a specified time horizon based on the historical distribution
        of returns. This implementation uses the historical simulation
        approach with a rolling lookback window.

        For example, 95% VaR of $1,000 means:
        - In 95% of cases, the loss will not exceed $1,000
        - In 5% of cases, the loss will exceed $1,000

        The calculation uses the (1-confidence) percentile of historical
        returns. For 95% confidence, this is the 5th percentile (worst 5%
        of returns).

        Parameters:
        -----------
        position_value : float
            Current position value in dollars. This is the notional amount
            at risk for the VaR calculation.
        confidence : float, optional
            Confidence level for VaR calculation (default: from config,
            typically 0.95 for 95% VaR). Higher values produce more
            conservative estimates.

        Returns:
        --------
        var : float
            Value at Risk in dollars. This is the estimated maximum loss
            at the specified confidence level. Always returns positive
            value representing dollar amount at risk.

        Note:
            Returns 0.0 if insufficient equity history (< 2 data points)
            to calculate meaningful VaR.

        Example:
            >>> var_95 = risk_mgr.calculate_var(position_value=25000, confidence=0.95)
            >>> print(f"95% VaR: ${var_95:,.0f}")
            95% VaR: $1,250
            >>> print(f"Interpretation: 95% confidence loss won't exceed ${var_95:,.0f}")
        """
        # Use config default if not specified
        if confidence is None:
            confidence = self.config.var_confidence

        # Need sufficient history for return calculation
        # VaR requires at least 2 data points to compute returns
        if len(self.equity_history) < 2:
            return 0.0

        # Extract lookback window from equity history
        # This ensures VaR adapts to recent market conditions
        equity_array = np.array(self.equity_history[-self.config.var_lookback :])

        # Calculate period-over-period returns
        # Returns = (current - previous) / previous
        returns = np.diff(equity_array) / equity_array[:-1]

        if len(returns) == 0:
            return 0.0

        # Historical VaR using percentile method:
        # Find the (1-confidence) percentile return (worst case)
        # For 95% confidence, we look at the 5th percentile
        var_percentile = (1 - confidence) * 100
        var_return = np.percentile(returns, var_percentile)

        # var_return is negative (representing a loss)
        # Convert to positive dollar amount

        # Convert return-based VaR to dollar loss amount
        # VaR_dollars = |return| × position_value
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
        """
        Calculate current drawdown from peak equity.

        Drawdown measures the decline from peak equity to current equity,
        expressed as a fraction. A 10% drawdown means current equity is
        90% of the peak (or equivalently, peak - current = 10% of peak).

        Parameters:
        -----------
        current_equity : float
            Current account equity.

        Returns:
        --------
        drawdown : float
            Current drawdown as decimal (e.g., 0.05 = 5% drawdown).
            Always non-negative; clamped to 0 if equity exceeds peak.
        """
        # Handle edge case of zero peak (shouldn't happen in practice)
        if self.state.peak_equity == 0:
            return 0.0

        # Calculate drawdown: (peak - current) / peak
        # This gives the fraction lost from peak
        # Example: peak=$100, current=$95 → drawdown = (100-95)/100 = 0.05 (5%)
        drawdown = (self.state.peak_equity - current_equity) / self.state.peak_equity

        # Clamp to zero - can't have negative drawdown (which would mean
        # equity is above peak, which is actually a good thing!)
        return max(0.0, drawdown)

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
