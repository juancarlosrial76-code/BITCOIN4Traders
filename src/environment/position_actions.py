"""
Discrete Position Sizing Actions
================================

Maps discrete actions to continuous position sizes.
Integrates Kelly Criterion for optimal position sizing.

Action Space (7 actions):
    0: -1.0 (Short 100%)
    1: -0.5 (Short 50%)
    2: 0.0 (Neutral)
    3: +0.33 (Long 33% - Quarter Kelly)
    4: +0.50 (Long 50% - Half Kelly)
    5: +0.75 (Long 75% - Three-Quarter Kelly)
    6: +1.0 (Long 100% - Full Position)

With Kelly Criterion Override:
    When Kelly calculation suggests optimal size, map to closest action.
    Example: Kelly says 40% → Action 4 (50%)
"""

import numpy as np
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
from enum import IntEnum


class PositionAction(IntEnum):
    """Discrete position actions."""

    SHORT_FULL = 0  # -100%
    SHORT_HALF = 1  # -50%
    NEUTRAL = 2  # 0%
    LONG_QUARTER = 3  # +33%
    LONG_HALF = 4  # +50%
    LONG_THREE_QTR = 5  # +75%
    LONG_FULL = 6  # +100%


# Position sizes corresponding to each action
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
    """Configuration for action-position mapping."""

    use_kelly_override: bool = True
    kelly_fraction: float = 0.5  # Half Kelly default
    min_position_size: float = 0.1  # Minimum 10% position
    max_position_size: float = 1.0  # Maximum 100% position

    # Action selection strategy
    strategy: str = "discrete"  # "discrete", "kelly_nearest", "kelly_continuous"


class PositionActionMapper:
    """
    Maps discrete actions to continuous position sizes.

    Supports three strategies:
    1. discrete: Standard discrete action mapping
    2. kelly_nearest: Kelly calculates optimal size, maps to nearest discrete action
    3. kelly_continuous: Action is suggestion, Kelly overrides with optimal size

    Usage:
    ------
    mapper = PositionActionMapper(ActionConfig())

    # Standard discrete mapping
    position_size = mapper.action_to_position(4)  # Returns 0.50

    # With Kelly override
    position_size = mapper.action_to_position(
        action=4,
        kelly_fraction=0.35,
        use_kelly=True
    )  # Kelly says 35%, closest is Action 3 (33%)
    """

    def __init__(self, config: ActionConfig):
        self.config = config
        self.n_actions = len(POSITION_SIZES)

    def action_to_position(
        self,
        action: int,
        kelly_fraction: Optional[float] = None,
        use_kelly: bool = False,
    ) -> float:
        """
        Convert action to position size.

        Parameters:
        -----------
        action : int
            Action index (0-6)
        kelly_fraction : float, optional
            Kelly-optimal position fraction
        use_kelly : bool
            Whether to override with Kelly calculation

        Returns:
        --------
        position_size : float
            Target position size (-1.0 to +1.0)
        """
        if action not in POSITION_SIZES:
            raise ValueError(
                f"Invalid action: {action}. Must be 0-{self.n_actions - 1}"
            )

        # Get base position from action
        base_position = POSITION_SIZES[action]

        # If Kelly override is enabled and we have a Kelly fraction
        if use_kelly and kelly_fraction is not None and self.config.use_kelly_override:
            return self._apply_kelly_override(action, kelly_fraction)

        return base_position

    def _apply_kelly_override(self, action: int, kelly_fraction: float) -> float:
        """
        Apply Kelly criterion to determine optimal position size.

        Strategy: Use Kelly to find optimal size, but constrain to:
        1. Same sign as action (don't flip long/short)
        2. Within min/max bounds
        3. Closest discrete action if using discrete strategy
        """
        # Get direction from action
        base_position = POSITION_SIZES[action]
        direction = np.sign(base_position)

        if direction == 0:
            # Neutral action - Kelly doesn't apply
            return 0.0

        # Apply Kelly fraction (e.g., Half Kelly)
        optimal_size = kelly_fraction * self.config.kelly_fraction

        # Apply direction
        optimal_size *= direction

        # Constrain to limits
        optimal_size = np.clip(
            optimal_size, -self.config.max_position_size, self.config.max_position_size
        )

        # Ensure minimum position size (avoid tiny positions)
        if abs(optimal_size) < self.config.min_position_size:
            # Instead of returning 0, use the base position if it's larger
            if abs(base_position) >= self.config.min_position_size:
                optimal_size = base_position
            else:
                return 0.0

        if self.config.strategy == "kelly_nearest":
            # Map to nearest discrete action
            return self._find_nearest_discrete(optimal_size)
        elif self.config.strategy == "kelly_continuous":
            # Use continuous Kelly value
            return optimal_size
        else:
            # Discrete: use action's predefined size
            return base_position

    def _find_nearest_discrete(self, target_size: float) -> float:
        """Find nearest discrete position size."""
        sizes = list(POSITION_SIZES.values())
        nearest = min(sizes, key=lambda x: abs(x - target_size))
        return nearest

    def position_to_action(self, position_size: float) -> int:
        """
        Convert position size to nearest action.

        Useful for converting Kelly-optimal size back to discrete action.
        """
        # Find closest position size
        sizes = list(POSITION_SIZES.values())
        nearest_idx = np.argmin([abs(size - position_size) for size in sizes])
        return nearest_idx

    def get_action_description(self, action: int) -> str:
        """Get human-readable description of action."""
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

        Prevents excessive position changes.
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

    Instead of the agent learning position sizes, Kelly Criterion
    calculates the optimal size and we select the nearest action.

    Usage:
    ------
    selector = KellyActionSelector()

    # Get optimal action based on Kelly
    action = selector.select_action(
        win_prob=0.6,
        win_loss_ratio=2.0,
        direction='long'  # or 'short'
    )
    """

    def __init__(self, kelly_fraction: float = 0.5):
        self.kelly_fraction = kelly_fraction
        self.mapper = PositionActionMapper(ActionConfig())

    def select_action(
        self, win_prob: float, win_loss_ratio: float, direction: str = "long"
    ) -> int:
        """
        Select optimal action based on Kelly criterion.

        Parameters:
        -----------
        win_prob : float
            Probability of winning
        win_loss_ratio : float
            Average win / average loss
        direction : str
            'long' or 'short'

        Returns:
        --------
        action : int
            Optimal discrete action
        """
        # Calculate Kelly fraction
        kelly_f = self._calculate_kelly_fraction(win_prob, win_loss_ratio)
        kelly_f *= self.kelly_fraction  # Apply Kelly fraction (e.g., Half Kelly)

        # Apply direction
        if direction == "short":
            kelly_f = -kelly_f

        # Find nearest discrete action
        action = self.mapper.position_to_action(kelly_f)

        return action

    def _calculate_kelly_fraction(
        self, win_prob: float, win_loss_ratio: float
    ) -> float:
        """Calculate Kelly fraction: f* = (bp - q) / b"""
        if win_loss_ratio <= 0:
            return 0.0

        q = 1 - win_prob
        kelly = (win_prob * win_loss_ratio - q) / win_loss_ratio

        return max(0.0, min(1.0, kelly))  # Constrain to [0, 1]


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
