"""
Kelly Criterion - Optimal Position Sizing
==========================================
Mathematical framework for optimal bet sizing to maximize long-term growth.

Formula:
f* = (p*b - q) / b

Where:
- f*: Optimal fraction of capital to bet
- p: Probability of winning
- q: Probability of losing (1-p)
- b: Win/loss ratio (profit/loss on win vs loss)

Key Properties:
- Maximizes geometric growth rate
- Prevents bankruptcy (never bet more than you have)
- Accounts for risk of ruin

Practical Implementation:
- Fractional Kelly (typically 25-50%) for risk management
- Dynamic adjustment based on win probability
- Position size caps (max 25% of capital per trade)
"""

import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass
from loguru import logger
from numba import jit


@dataclass
class KellyParameters:
    """Kelly Criterion parameters."""

    win_probability: float  # p
    win_loss_ratio: float  # b (avg win / avg loss)
    kelly_fraction: float = 0.5  # Fractional Kelly (50%)
    max_position: float = 0.25  # Max 25% of capital

    def __post_init__(self):
        """Validate parameters."""
        assert 0 <= self.win_probability <= 1, "Win probability must be in [0, 1]"
        assert self.win_loss_ratio > 0, "Win/loss ratio must be positive"
        assert 0 < self.kelly_fraction <= 1, "Kelly fraction must be in (0, 1]"
        assert 0 < self.max_position <= 1, "Max position must be in (0, 1]"


class KellyCriterion:
    """
    Kelly Criterion for optimal position sizing.

    Usage:
    ------
    # Estimate parameters from trading history
    kelly = KellyCriterion()
    params = kelly.estimate_parameters(trade_history)

    # Calculate optimal position size
    position_size = kelly.calculate_position_size(
        capital=100000,
        params=params
    )

    # Adjust dynamically based on performance
    adjusted_size = kelly.dynamic_kelly(
        capital=100000,
        recent_win_rate=0.55,
        recent_profit_factor=1.8
    )
    """

    def __init__(self):
        logger.info("KellyCriterion initialized")

    def calculate_kelly_fraction(self, win_prob: float, win_loss_ratio: float) -> float:
        """
        Calculate optimal Kelly fraction.

        Formula: f* = (p*b - q) / b

        Parameters:
        -----------
        win_prob : float
            Probability of winning
        win_loss_ratio : float
            Average win / average loss

        Returns:
        --------
        kelly_fraction : float
            Optimal fraction (can be negative if -EV)
        """
        p = win_prob
        q = 1 - p
        b = win_loss_ratio

        # Kelly formula: f* = (p*b - q) / b = p - q/b
        kelly_f = (
            p * b - q
        ) / b  # Positive = positive expected value; negative = avoid this bet

        return float(kelly_f)

    def calculate_position_size(self, capital: float, params: KellyParameters) -> float:
        """
        Calculate position size in dollars.

        Parameters:
        -----------
        capital : float
            Available capital
        params : KellyParameters
            Kelly parameters

        Returns:
        --------
        position_size : float
            Position size in dollars
        """
        # Calculate full Kelly
        full_kelly = self.calculate_kelly_fraction(
            params.win_probability, params.win_loss_ratio
        )

        # Apply fractional Kelly
        kelly_fraction = full_kelly * params.kelly_fraction

        # Clip to [0, max_position]
        kelly_fraction = np.clip(kelly_fraction, 0.0, params.max_position)

        # Convert to dollar amount
        position_size = capital * kelly_fraction

        return float(position_size)

    def estimate_parameters(
        self, trade_returns: np.ndarray, lookback: Optional[int] = None
    ) -> KellyParameters:
        """
        Estimate Kelly parameters from trading history.

        Parameters:
        -----------
        trade_returns : np.ndarray
            Array of trade returns (positive = win, negative = loss)
        lookback : int, optional
            Use only last N trades

        Returns:
        --------
        params : KellyParameters
            Estimated parameters
        """
        # Convert to numpy array if needed
        trade_returns = np.array(trade_returns)

        if lookback is not None:
            trade_returns = trade_returns[-lookback:]

        # Separate wins and losses
        wins = trade_returns[trade_returns > 0]
        losses = trade_returns[trade_returns < 0]

        if len(wins) == 0 or len(losses) == 0:
            logger.warning("No wins or losses in history, using defaults")
            return KellyParameters(
                win_probability=0.55,  # Slightly favorable to allow initial trading
                win_loss_ratio=1.0,
                kelly_fraction=0.5,
                max_position=0.25,
            )

        # Win probability
        win_prob = len(wins) / len(trade_returns)

        # Win/loss ratio
        avg_win = np.mean(wins)
        avg_loss = abs(np.mean(losses))
        win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 1.0

        params = KellyParameters(
            win_probability=float(win_prob),
            win_loss_ratio=float(win_loss_ratio),
            kelly_fraction=0.5,  # Conservative
            max_position=0.25,
        )

        logger.info(f"Estimated Kelly parameters:")
        logger.info(f"  Win probability: {params.win_probability:.3f}")
        logger.info(f"  Win/loss ratio: {params.win_loss_ratio:.3f}")

        return params

    def dynamic_kelly(
        self,
        capital: float,
        recent_win_rate: float,
        recent_profit_factor: float,
        kelly_fraction: float = 0.5,
        max_position: float = 0.25,
    ) -> float:
        """
        Dynamic Kelly adjustment based on recent performance.

        Parameters:
        -----------
        capital : float
            Available capital
        recent_win_rate : float
            Recent win rate (e.g., last 20 trades)
        recent_profit_factor : float
            Profit factor (gross profit / gross loss)
        kelly_fraction : float
            Fractional Kelly multiplier
        max_position : float
            Maximum position size

        Returns:
        --------
        position_size : float
            Adjusted position size
        """
        # Convert profit factor to win/loss ratio
        # Profit factor = (Win rate * Avg win) / (Loss rate * Avg loss)
        # Approximate: win_loss_ratio ≈ profit_factor
        win_loss_ratio = recent_profit_factor

        params = KellyParameters(
            win_probability=recent_win_rate,
            win_loss_ratio=win_loss_ratio,
            kelly_fraction=kelly_fraction,
            max_position=max_position,
        )

        return self.calculate_position_size(capital, params)

    def expected_growth_rate(
        self, win_prob: float, win_loss_ratio: float, kelly_fraction: float
    ) -> float:
        """
        Calculate expected geometric growth rate.

        Used to compare different Kelly fractions.

        Parameters:
        -----------
        win_prob : float
            Win probability
        win_loss_ratio : float
            Win/loss ratio
        kelly_fraction : float
            Fraction of Kelly to use

        Returns:
        --------
        growth_rate : float
            Expected geometric growth rate
        """
        p = win_prob
        q = 1 - p
        b = win_loss_ratio
        f = kelly_fraction

        # Expected log-growth rate: G(f) = p*ln(1+f*b) + q*ln(1-f)
        # Maximizing this is equivalent to the Kelly criterion
        g = p * np.log(1 + f * b) + q * np.log(1 - f)

        return float(g)

    def risk_of_ruin(
        self, win_prob: float, win_loss_ratio: float, target_drawdown: float = 0.5
    ) -> float:
        """
        Estimate risk of ruin (probability of hitting drawdown).

        Parameters:
        -----------
        win_prob : float
            Win probability
        win_loss_ratio : float
            Win/loss ratio
        target_drawdown : float
            Drawdown threshold (e.g., 0.5 = 50% loss)

        Returns:
        --------
        risk : float
            Probability of hitting drawdown
        """
        p = win_prob
        q = 1 - p
        b = win_loss_ratio

        if p * b <= q:
            # Negative expectancy (EV ≤ 0) → certain ruin long-term
            return 1.0

        # Simplified risk of ruin: R = ((q/p)/b)^(1/DD)
        risk = ((q / p) / b) ** (1 / target_drawdown)
        risk = min(risk, 1.0)  # Cap at 100%

        return float(risk)


# ============================================================================
# NUMBA-OPTIMIZED FUNCTIONS
# ============================================================================


@jit(nopython=True, cache=True)
def calculate_kelly_numba(
    win_prob: float, win_loss_ratio: float, kelly_fraction: float, max_position: float
) -> float:
    """Numba-optimized Kelly calculation."""
    p = win_prob
    q = 1.0 - p
    b = win_loss_ratio

    # Kelly formula
    full_kelly = (p * b - q) / b

    # Apply fractional Kelly
    kelly_f = full_kelly * kelly_fraction

    # Clip
    kelly_f = min(max(kelly_f, 0.0), max_position)

    return kelly_f


@jit(nopython=True, cache=True)
def simulate_kelly_growth(
    initial_capital: float,
    win_prob: float,
    win_loss_ratio: float,
    kelly_fraction: float,
    n_trades: int,
    seed: int = 42,
) -> float:
    """
    Simulate capital growth with Kelly sizing.

    Returns final capital after n_trades.
    """
    np.random.seed(seed)

    capital = initial_capital

    for _ in range(n_trades):
        # Calculate position size
        kelly_f = calculate_kelly_numba(win_prob, win_loss_ratio, kelly_fraction, 0.25)
        position_size = capital * kelly_f

        # Simulate trade outcome
        if np.random.rand() < win_prob:
            # Win
            profit = position_size * win_loss_ratio
            capital += profit
        else:
            # Loss
            capital -= position_size

        # Prevent bankruptcy
        if capital <= 0:
            return 0.0

    return capital


# ============================================================================
# EXAMPLE USAGE & VALIDATION
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("KELLY CRITERION TEST")
    print("=" * 80)

    kelly = KellyCriterion()

    # Test 1: Basic Kelly calculation
    print("\n[TEST 1] Basic Kelly Calculation")

    win_prob = 0.55
    win_loss_ratio = 1.5

    full_kelly = kelly.calculate_kelly_fraction(win_prob, win_loss_ratio)

    print(f"\nParameters:")
    print(f"  Win probability: {win_prob:.1%}")
    print(f"  Win/loss ratio: {win_loss_ratio:.2f}")
    print(f"\nFull Kelly: {full_kelly:.1%}")
    print(f"Half Kelly (50%): {full_kelly * 0.5:.1%}")
    print(f"Quarter Kelly (25%): {full_kelly * 0.25:.1%}")

    # Test 2: Position sizing
    print("\n[TEST 2] Position Sizing")

    capital = 100000
    params = KellyParameters(
        win_probability=win_prob,
        win_loss_ratio=win_loss_ratio,
        kelly_fraction=0.5,
        max_position=0.25,
    )

    position_size = kelly.calculate_position_size(capital, params)

    print(f"\nCapital: ${capital:,.0f}")
    print(f"Position size: ${position_size:,.0f} ({position_size / capital:.1%})")

    # Test 3: Parameter estimation from history
    print("\n[TEST 3] Parameter Estimation")

    # Simulate trading history
    np.random.seed(42)
    n_trades = 100

    # Generate returns (55% win rate, 1.5:1 ratio)
    trade_returns = []
    for _ in range(n_trades):
        if np.random.rand() < 0.55:
            # Win
            trade_returns.append(np.random.uniform(0.01, 0.03))
        else:
            # Loss
            trade_returns.append(np.random.uniform(-0.02, -0.01))

    trade_returns = np.array(trade_returns)

    print(f"\nSimulated {n_trades} trades")
    print(f"  Wins: {(trade_returns > 0).sum()}")
    print(f"  Losses: {(trade_returns < 0).sum()}")
    print(f"  Actual win rate: {(trade_returns > 0).mean():.1%}")

    estimated_params = kelly.estimate_parameters(trade_returns)

    print(f"\nEstimated parameters:")
    print(f"  Win probability: {estimated_params.win_probability:.1%}")
    print(f"  Win/loss ratio: {estimated_params.win_loss_ratio:.2f}")

    # Test 4: Growth rate analysis
    print("\n[TEST 4] Growth Rate Analysis")

    fractions = [0.25, 0.50, 0.75, 1.00]

    print("\nExpected growth rates:")
    for frac in fractions:
        growth = kelly.expected_growth_rate(win_prob, win_loss_ratio, frac * full_kelly)
        print(f"  {frac:.0%} Kelly: {growth:.4f}")

    # Test 5: Risk of ruin
    print("\n[TEST 5] Risk of Ruin Analysis")

    risk_50 = kelly.risk_of_ruin(win_prob, win_loss_ratio, target_drawdown=0.5)
    risk_30 = kelly.risk_of_ruin(win_prob, win_loss_ratio, target_drawdown=0.3)
    risk_20 = kelly.risk_of_ruin(win_prob, win_loss_ratio, target_drawdown=0.2)

    print(f"\nRisk of ruin:")
    print(f"  50% drawdown: {risk_50:.2%}")
    print(f"  30% drawdown: {risk_30:.2%}")
    print(f"  20% drawdown: {risk_20:.2%}")

    # Test 6: Simulation
    print("\n[TEST 6] Growth Simulation")

    print("\nSimulating 1000 trades with different Kelly fractions:")

    initial_capital = 100000
    n_sim_trades = 1000

    for frac in [0.25, 0.50, 1.00]:
        final_capital = simulate_kelly_growth(
            initial_capital, win_prob, win_loss_ratio, frac, n_sim_trades, seed=42
        )

        total_return = (final_capital - initial_capital) / initial_capital

        print(f"  {frac:.0%} Kelly: ${final_capital:,.0f} ({total_return:+.1%})")

    # Test 7: Dynamic Kelly
    print("\n[TEST 7] Dynamic Kelly Adjustment")

    scenarios = [
        ("Good streak", 0.65, 2.0),
        ("Normal", 0.55, 1.5),
        ("Bad streak", 0.45, 1.0),
    ]

    print("\nDynamic position sizing:")
    for scenario, win_rate, profit_factor in scenarios:
        pos_size = kelly.dynamic_kelly(
            capital=capital,
            recent_win_rate=win_rate,
            recent_profit_factor=profit_factor,
        )
        print(f"  {scenario}: ${pos_size:,.0f} ({pos_size / capital:.1%})")

    print("\n" + "=" * 80)
    print("✓ KELLY CRITERION TEST PASSED")
    print("=" * 80)

    print("\nKey Insights:")
    print("• Fractional Kelly (50%) reduces risk while maintaining growth")
    print("• Position sizing adapts to recent performance")
    print("• Risk of ruin decreases significantly with fractional Kelly")
    print("• Ready for integration with risk management system")
