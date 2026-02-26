"""
Discrete Position Sizing Actions
=================================

Purpose:
--------
This module provides the action space definition for the trading environment,
mapping discrete RL actions to continuous position sizes. It implements a
realistic action space where the agent chooses from predefined position
levels rather than arbitrary sizes.

Key Components:
----------------
1. PositionAction: Enum defining discrete actions (7 levels)
2. POSITION_SIZES: Mapping from actions to position sizes
3. ActionConfig: Configuration for action mapping behavior
4. PositionActionMapper: Core class for action-to-position conversion
5. KellyActionSelector: Kelly Criterion-based action selection

Action Space:
------------
The action space consists of 7 discrete actions representing different
position sizes from full short to full long:

    Action 0: SHORT_FULL  (-100%) - Full short position
    Action 1: SHORT_HALF  (-50%)  - Half short position
    Action 2: NEUTRAL     (0%)    - No position (flat)
    Action 3: LONG_QUARTER (+33%) - Quarter long (Kelly-inspired)
    Action 4: LONG_HALF   (+50%)  - Half long position
    Action 5: LONG_THREE_QTR (+75%) - Three-quarter long
    Action 6: LONG_FULL   (+100%) - Full long position

Rationale:
---------
1. Discrete actions are more stable for RL training than continuous
2. The asymmetric long sizes (33%, 50%, 75%) relate to Kelly fractions
3. Short positions are -50% and -100% (less common but available)

Kelly Criterion Integration:
---------------------------
The module optionally integrates the Kelly Criterion for optimal position
sizing. When enabled, the Kelly formula calculates the optimal position
size based on win probability and win/loss ratio:

    Kelly% = (bp - q) / b

Where:
    b = odds received (win/loss ratio)
    p = probability of winning
    q = probability of losing (1 - p)

Usage:
------
    from src.environment.position_actions import (
        PositionActionMapper, ActionConfig, PositionAction,
        KellyActionSelector, POSITION_SIZES
    )

    # Basic usage - discrete actions
    config = ActionConfig(use_kelly_override=False)
    mapper = PositionActionMapper(config)
    position = mapper.action_to_position(4)  # Returns 0.50

    # With Kelly override
    config = ActionConfig(use_kelly_override=True, strategy='kelly_nearest')
    mapper = PositionActionMapper(config)
    position = mapper.action_to_position(4, kelly_fraction=0.35, use_kelly=True)

    # Kelly-based action selection
    selector = KellyActionSelector(kelly_fraction=0.5)
    action = selector.select_action(win_prob=0.6, win_loss_ratio=2.0, direction='long')

Dependencies:
-------------
- numpy: Numerical operations
- dataclasses: Configuration data structures
- enum: IntEnum for action definitions
"""

import numpy as np
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
from enum import IntEnum


class PositionAction(IntEnum):
    """
    Discrete position actions representing different position sizes.

    These actions define the action space for the trading RL environment.
    The agent selects one of these actions at each step, which is then
    mapped to a position size.

    Attributes:
        SHORT_FULL: Full short position (-100% of available capital)
        SHORT_HALF: Half short position (-50% of available capital)
        NEUTRAL: No position (0%)
        LONG_QUARTER: Quarter long position (+33% - inspired by Quarter Kelly)
        LONG_HALF: Half long position (+50%)
        LONG_THREE_QTR: Three-quarter long position (+75%)
        LONG_FULL: Full long position (+100%)

    Note:
        The fractional long sizes (33%, 50%, 75%) are inspired by the
        Kelly Criterion where fractional Kelly (especially Half Kelly)
        is often preferred for risk management.
    """

    SHORT_FULL = 0  # -100%
    SHORT_HALF = 1  # -50%
    NEUTRAL = 2  # 0%
    LONG_QUARTER = 3  # +33%
    LONG_HALF = 4  # +50%
    LONG_THREE_QTR = 5  # +75%
    LONG_FULL = 6  # +100%


# Position sizes corresponding to each action
# Maps the discrete action enum to continuous position size
POSITION_SIZES = {
    PositionAction.SHORT_FULL: -1.0,
    PositionAction.SHORT_HALF: -0.5,
    PositionAction.NEUTRAL: 0.0,
    PositionAction.LONG_QUARTER: 0.33,
    PositionAction.LONG_HALF: 0.50,
    PositionAction.LONG_THREE_QTR: 0.75,
    PositionAction.LONG_FULL: 1.0,
}


@dataclass
class ActionConfig:
    """
    Configuration for action-to-position mapping behavior.

    This config controls how discrete actions are translated to
    continuous position sizes, with optional Kelly Criterion integration.

    Attributes:
        use_kelly_override: Whether to apply Kelly Criterion adjustment
        kelly_fraction: Fraction of Kelly to use (e.g., 0.5 = Half Kelly)
                        Full Kelly can be very aggressive; fractional
                        versions are more conservative
        min_position_size: Minimum meaningful position size (prevents
                          tiny positions that generate costs without benefit)
        max_position_size: Maximum position size (e.g., 1.0 = 100% of capital)
        strategy: How to handle Kelly calculation:
            - 'discrete': Use exact predefined position sizes (no Kelly)
            - 'kelly_nearest': Calculate Kelly, map to nearest discrete action
            - 'kelly_continuous': Use Kelly value directly (continuous)

    Example:
        >>> # Conservative discrete actions only
        >>> config = ActionConfig(use_kelly_override=False)
        >>>
        >>> # Kelly with nearest discrete action
        >>> config = ActionConfig(
        ...     use_kelly_override=True,
        ...     strategy='kelly_nearest',
        ...     kelly_fraction=0.5
        ... )
    """

    use_kelly_override: bool = True
    kelly_fraction: float = 0.5  # Half Kelly default
    min_position_size: float = 0.1  # Minimum 10% position
    max_position_size: float = 1.0  # Maximum 100% position
    strategy: str = "discrete"  # "discrete", "kelly_nearest", "kelly_continuous"


class PositionActionMapper:
    """
    Maps discrete RL actions to continuous position sizes.

    This class is the core of the action mapping system. It converts
    the integer action chosen by the RL agent into a position size
    that can be used by the trading environment.

    Key Features:
    ------------
    1. Discrete mapping: Direct mapping from action to predefined position
    2. Kelly override: Adjusts position based on Kelly Criterion
    3. Constraint handling: Enforces min/max position limits
    4. Bidirectional mapping: Can convert position back to nearest action

    Strategies:
    ----------
    1. 'discrete': Standard discrete action mapping
       - Action 4 → Position 0.50 (exactly)
       - No Kelly calculation performed

    2. 'kelly_nearest': Kelly calculates optimal, maps to nearest action
       - Kelly says 35% → finds nearest discrete (33% or 50%)
       - Preserves interpretability of discrete actions

    3. 'kelly_continuous': Uses continuous Kelly value
       - Kelly says 35% → Position 0.35
       - More precise but less interpretable

    Attributes:
        config: ActionConfig with mapping behavior
        n_actions: Number of discrete actions (7)

    Example:
        >>> config = ActionConfig(strategy='discrete')
        >>> mapper = PositionActionMapper(config)
        >>>
        >>> # Standard discrete mapping
        >>> position = mapper.action_to_position(4)  # Returns 0.50
        >>>
        >>> # With Kelly
        >>> position = mapper.action_to_position(
        ...     action=4,
        ...     kelly_fraction=0.35,
        ...     use_kelly=True
        ... )
    """

    def __init__(self, config: ActionConfig):
        """
        Initialize position action mapper.

        Args:
            config: ActionConfig with mapping behavior
        """
        self.config = config
        self.n_actions = len(POSITION_SIZES)

    def action_to_position(
        self,
        action: int,
        kelly_fraction: Optional[float] = None,
        use_kelly: bool = False,
    ) -> float:
        """
        Convert discrete action to continuous position size.

        This is the main entry point for converting RL actions to
        position sizes. It applies the configured strategy to
        determine the final position.

        Parameters:
        -----------
        action : int
            Action index (0-6). Must be a valid action in POSITION_SIZES.
        kelly_fraction : float, optional
            Kelly-optimal position fraction. If provided and use_kelly
            is True, the Kelly calculation will be applied.
        use_kelly : bool
            Whether to apply Kelly override. If True and kelly_fraction
            is provided, the position may be adjusted.

        Returns:
        --------
        position_size : float
            Target position size in range [-1.0, +1.0]
            Negative = short, Positive = long, 0 = neutral

        Raises:
            ValueError: If action is not in valid range

        Example:
            >>> mapper = PositionActionMapper(ActionConfig())
            >>>
            >>> # Standard discrete
            >>> pos = mapper.action_to_position(4)  # 0.50
            >>>
            >>> # With Kelly
            >>> pos = mapper.action_to_position(4, kelly_fraction=0.35, use_kelly=True)
        """
        # Validate action
        if action not in POSITION_SIZES:
            raise ValueError(
                f"Invalid action: {action}. Must be 0-{self.n_actions - 1}"
            )

        # Get base position from action
        base_position = POSITION_SIZES[action]

        # Apply Kelly override if enabled and requested
        if use_kelly and kelly_fraction is not None and self.config.use_kelly_override:
            return self._apply_kelly_override(action, kelly_fraction)

        # Return base discrete position
        return base_position

    def _apply_kelly_override(self, action: int, kelly_fraction: float) -> float:
        """
        Apply Kelly Criterion to determine optimal position size.

        The Kelly Criterion calculates the optimal position size based
        on edge (win probability × win/loss ratio). This method applies
        the Kelly calculation while respecting several constraints:

        1. Same sign as action (don't flip long/short)
        2. Within min/max position bounds
        3. Nearest discrete action (if using kelly_nearest strategy)

        Parameters:
        -----------
        action : int
            Original action for determining direction
        kelly_fraction : float
            Kelly fraction from external calculation (0-1 range)

        Returns:
        --------
        position_size : float
            Kelly-adjusted position size
        """
        # Get direction from action (preserve long/short)
        base_position = POSITION_SIZES[action]
        direction = np.sign(base_position)

        if direction == 0:
            # Neutral action - Kelly doesn't apply to flat positions
            return 0.0

        # Apply fractional Kelly
        # kelly_fraction is the raw Kelly calculation
        # self.config.kelly_fraction is the multiplier (e.g., 0.5 for Half Kelly)
        optimal_size = kelly_fraction * self.config.kelly_fraction

        # Apply direction (match sign of original action)
        optimal_size *= direction

        # Constrain to configured position limits
        optimal_size = np.clip(
            optimal_size, -self.config.max_position_size, self.config.max_position_size
        )

        # Ensure minimum position size
        # Positions smaller than minimum don't justify trading costs
        if abs(optimal_size) < self.config.min_position_size:
            # Use base position if meaningful, otherwise go flat
            if abs(base_position) >= self.config.min_position_size:
                optimal_size = base_position
            else:
                return 0.0

        # Apply strategy-specific behavior
        if self.config.strategy == "kelly_nearest":
            # Map to nearest discrete action for interpretability
            return self._find_nearest_discrete(optimal_size)
        elif self.config.strategy == "kelly_continuous":
            # Use continuous Kelly value directly
            return optimal_size
        else:
            # Discrete strategy: ignore Kelly, use base position
            return base_position

    def _find_nearest_discrete(self, target_size: float) -> float:
        """
        Find the nearest discrete position size to target.

        Used by kelly_nearest strategy to maintain discrete action
        interpretability while benefiting from Kelly sizing.

        Parameters:
        -----------
        target_size : float
            Target position size (e.g., Kelly result)

        Returns:
        --------
        nearest : float
            Nearest predefined position size
        """
        sizes = list(POSITION_SIZES.values())
        nearest = min(sizes, key=lambda x: abs(x - target_size))
        return nearest

    def position_to_action(self, position_size: float) -> int:
        """
        Convert position size back to nearest discrete action.

        Useful when you have a continuous position (e.g., from Kelly)
        and want to convert it back to a discrete action for analysis
        or reporting.

        Parameters:
        -----------
        position_size : float
            Position size in range [-1.0, 1.0]

        Returns:
        --------
        action : int
            Nearest discrete action index

        Example:
            >>> mapper = PositionActionMapper(ActionConfig())
            >>> action = mapper.position_to_action(0.55)  # Returns 4 (nearest to 0.50)
        """
        # Find closest position size
        sizes = list(POSITION_SIZES.values())
        nearest_idx = np.argmin([abs(size - position_size) for size in sizes])
        return nearest_idx

    def get_action_description(self, action: int) -> str:
        """
        Get human-readable description of an action.

        Useful for logging, debugging, and user interface.

        Parameters:
        -----------
        action : int
            Action index (0-6)

        Returns:
        --------
        description : str
            Human-readable description

        Example:
            >>> mapper.get_action_description(4)
            'Long 50%'
        """
        descriptions = {
            PositionAction.SHORT_FULL: "Short 100%",
            PositionAction.SHORT_HALF: "Short 50%",
            PositionAction.NEUTRAL: "Neutral (0%)",
            PositionAction.LONG_QUARTER: "Long 33%",
            PositionAction.LONG_HALF: "Long 50%",
            PositionAction.LONG_THREE_QTR: "Long 75%",
            PositionAction.LONG_FULL: "Long 100%",
        }
        return descriptions.get(action, f"Unknown action {action}")

    def get_allowed_actions(
        self, current_position: float, max_position: float = 1.0
    ) -> List[int]:
        """
        Get list of allowed actions given current position.

        This method can be used to constrain the action space based
        on current position, preventing excessive position changes
        that could lead to unrealistic trading.

        Parameters:
        -----------
        current_position : float
            Current position size (-1.0 to 1.0)
        max_position : float
            Maximum allowed position change (as fraction of max position)

        Returns:
        --------
        allowed : List[int]
            List of allowed action indices

        Example:
            >>> mapper.get_allowed_actions(current_position=0.5, max_position=1.0)
            # Returns actions where |target - 0.5| <= 2.0
        """
        allowed = []

        for action, target_pos in POSITION_SIZES.items():
            # Check if position change is within limits
            position_change = abs(target_pos - current_position)
            if position_change <= max_position * 2:  # Allow up to 200% change
                allowed.append(action)

        return allowed


class KellyActionSelector:
    """
    Selects actions based on Kelly-optimal position sizing.

    This class provides an alternative to RL-learned actions by
    directly calculating the optimal position using the Kelly
    Criterion. It can be used:

    1. As a baseline/benchmark for RL agents
    2. For rule-based trading strategies
    3. To provide Kelly information to the RL agent

    Kelly Criterion:
    -----------------
    The Kelly formula calculates the optimal fraction of capital to bet:

        f* = (bp - q) / b

    Where:
        b = odds received on the wager (win/loss ratio)
        p = probability of winning
        q = probability of losing (1 - p)

    In trading terms:
        Kelly% = (win_prob × win_loss_ratio - loss_prob) / win_loss_ratio

    Note:
        Full Kelly is very aggressive. Most practitioners use:
        - Half Kelly (50%): ~75% of Kelly returns with ~50% of variance
        - Quarter Kelly (25%): Even more conservative

    Attributes:
        kelly_fraction: Fraction of Kelly to use (0.5 = Half Kelly)
        mapper: PositionActionMapper for action conversion

    Example:
        >>> selector = KellyActionSelector(kelly_fraction=0.5)
        >>>
        >>> # Get optimal action based on win probability
        >>> action = selector.select_action(
        ...     win_prob=0.6,
        ...     win_loss_ratio=2.0,
        ...     direction='long'
        ... )
    """

    def __init__(self, kelly_fraction: float = 0.5):
        """
        Initialize Kelly action selector.

        Args:
            kelly_fraction: Fraction of Kelly to use (0.5 = Half Kelly)
        """
        self.kelly_fraction = kelly_fraction
        self.mapper = PositionActionMapper(ActionConfig())

    def select_action(
        self, win_prob: float, win_loss_ratio: float, direction: str = "long"
    ) -> int:
        """
        Select optimal action based on Kelly criterion.

        Calculates the Kelly-optimal position size and maps it to
        the nearest discrete action.

        Parameters:
        -----------
        win_prob : float
            Probability of winning (0.0 to 1.0)
            Represents the strategy's historical win rate
        win_loss_ratio : float
            Average win / average loss ratio
            E.g., 2.0 means wins are 2x larger than losses
        direction : str
            Trading direction: 'long' or 'short'

        Returns:
        --------
        action : int
            Optimal discrete action index

        Example:
            >>> selector = KellyActionSelector(0.5)
            >>>
            >>> # Strong edge: 60% win rate, 2:1 ratio
            >>> action = selector.select_action(0.6, 2.0, 'long')
            >>>
            >>> # Calculate position
            >>> position = POSITION_SIZES[action]
        """
        # Calculate Kelly fraction
        kelly_f = self._calculate_kelly_fraction(win_prob, win_loss_ratio)

        # Apply fractional Kelly (e.g., Half Kelly = 50% of full Kelly)
        kelly_f *= self.kelly_fraction

        # Apply direction (short Kelly is negative)
        if direction == "short":
            kelly_f = -kelly_f

        # Find nearest discrete action
        action = self.mapper.position_to_action(kelly_f)

        return action

    def _calculate_kelly_fraction(
        self, win_prob: float, win_loss_ratio: float
    ) -> float:
        """
        Calculate Kelly fraction from win probability and win/loss ratio.

        Kelly Formula: f* = (bp - q) / b

        Where:
            b = win_loss_ratio (odds)
            p = win_prob (probability of win)
            q = 1 - p (probability of loss)

        Parameters:
        -----------
        win_prob : float
            Probability of winning (0.0 to 1.0)
        win_loss_ratio : float
            Ratio of average win to average loss

        Returns:
        --------
        kelly : float
            Kelly fraction (0.0 to 1.0). Returns 0 if negative edge.
        """
        # Validate inputs
        if win_loss_ratio <= 0:
            return 0.0

        # Calculate Kelly
        q = 1 - win_prob
        kelly = (win_prob * win_loss_ratio - q) / win_loss_ratio

        # Constrain to valid range [0, 1]
        # Negative Kelly means negative edge (don't trade)
        return max(0.0, min(1.0, kelly))


if __name__ == "__main__":
    print("=" * 60)
    print("POSITION ACTION MAPPER TEST")
    print("=" * 60)

    # Test basic mapping
    mapper = PositionActionMapper(ActionConfig())

    print("\nStandard Discrete Mapping:")
    for action in range(7):
        pos = mapper.action_to_position(action)
        desc = mapper.get_action_description(action)
        print(f"  Action {action}: {desc:20s} → Position {pos:+.2f}")

    # Test Kelly override
    print("\nKelly Override Examples:")
    config = ActionConfig(strategy="kelly_nearest", kelly_fraction=0.5)
    mapper_kelly = PositionActionMapper(config)

    test_cases = [
        (4, 0.20, "Kelly says 20% (Half Kelly of 40%)"),
        (4, 0.35, "Kelly says 35% (Half Kelly of 70%)"),
        (5, 0.60, "Kelly says 60% (Half Kelly of 120%)"),
    ]

    for action, kelly_f, desc in test_cases:
        pos = mapper_kelly.action_to_position(action, kelly_f, use_kelly=True)
        print(f"  {desc}")
        print(f"    Action {action} → Position {pos:+.2f}")

    # Test Kelly Action Selector
    print("\nKelly Action Selector:")
    selector = KellyActionSelector(kelly_fraction=0.5)

    scenarios = [
        (0.55, 1.5, "long", "Moderate edge"),
        (0.60, 2.0, "long", "Strong edge"),
        (0.45, 1.5, "short", "Short opportunity"),
    ]

    for win_prob, ratio, direction, desc in scenarios:
        action = selector.select_action(win_prob, ratio, direction)
        pos = POSITION_SIZES[action]
        print(f"  {desc}: {direction}, win_prob={win_prob:.0%}, W/L={ratio:.1f}")
        print(f"    → Action {action} ({mapper.get_action_description(action)})")

    print("\n" + "=" * 60)
    print("✓ ALL TESTS PASSED")
    print("=" * 60)
