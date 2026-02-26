"""
Configuration System - YAML Integration
=====================================

Purpose:
--------
This module provides the complete configuration system for the BITCOIN4Traders
environment. It enables full YAML-based configuration of all trading environment
parameters, including transaction costs, slippage models, order book settings,
reward shaping, and market regimes.

Key Features:
-------------
1. Transaction Cost Configuration: Maker/taker fees
2. Slippage Configuration: Multiple model types
3. Order Book Configuration: L2 market simulation
4. Reward Configuration: Dynamic reward components
5. Market Regime Configuration: Volatility/volume patterns
6. YAML Loading: Both Hydra and standalone modes

Configuration Hierarchy:
-----------------------
EnvironmentConfig
├── transaction_costs: TransactionCostConfig
│   ├── fixed_bps
│   ├── maker_fee_bps
│   ├── taker_fee_bps
│   └── include_slippage
├── slippage: SlippageConfig
│   ├── model_type
│   ├── fixed_slippage_bps
│   └── volume_impact_coef
├── orderbook: OrderBookConfig
│   ├── enabled
│   ├── n_levels
│   └── base_spread_bps
├── reward: RewardConfig
│   ├── components[]
│   │   ├── name
│   │   └── weight
│   └── clip_min/max
└── market: MarketConfig
    ├── vol_regimes{}
    └── volume_patterns{}

Usage:
------
    from src.environment.config_system import (
        EnvironmentConfig,
        load_environment_config_from_yaml
    )

    # Load from YAML (standalone mode)
    config = load_environment_config_from_yaml('config/environment/realistic_env.yaml')

    # Load from YAML (Hydra mode)
    config = EnvironmentConfig.from_yaml(cfg)

    # Access configuration
    print(f"Maker fee: {config.transaction_costs.maker_fee_bps} bps")
    print(f"Max position: {config.max_position_size}")

YAML Structure:
--------------
    environment:
      type: realistic
      initial_capital: 100000.0
      max_position_size: 1.0
      max_drawdown: 0.20

    transaction_costs:
      fixed_bps: 5.0
      maker_fee_bps: 2.0
      taker_fee_bps: 5.0
      include_slippage: true

    slippage:
      model_type: volume_based
      fixed_slippage_bps: 5.0
      volume_impact_coef: 0.1

    orderbook:
      enabled: true
      n_levels: 10
      base_spread_bps: 5.0

    reward:
      components:
        - name: return
          weight: 1.0
        - name: sharpe
          weight: 0.1
          lookback: 20
      clip_min: -10.0
      clip_max: 10.0
      scale: 100.0

    market:
      vol_regimes:
        low_vol: 0.01
        normal: 0.02
        high_vol: 0.05

Dependencies:
-------------
- dataclasses: Configuration data structures
- yaml: YAML parsing
- omegaconf: DictConfig for Hydra integration
- pathlib: Path handling
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
from pathlib import Path
import yaml
from omegaconf import DictConfig


@dataclass
class TransactionCostConfig:
    """
    Transaction cost configuration.

    Defines all fees associated with trading:
    - Fixed fees (exchange fees)
    - Maker fees (limit orders, typically lower)
    - Taker fees (market orders, typically higher)

    Attributes:
        fixed_bps: Base transaction cost in basis points (for simple model)
        maker_fee_bps: Fee for limit orders (typically 1-2 bps)
        taker_fee_bps: Fee for market orders (typically 3-6 bps)
        include_slippage: Whether to include slippage in calculations
        include_market_impact: Whether to include permanent market impact

    Example:
        >>> config = TransactionCostConfig(
        ...     maker_fee_bps=2.0,
        ...     taker_fee_bps=5.0,
        ...     include_slippage=True
        ... )
    """

    # Fixed costs
    fixed_bps: float = 5.0  # Base transaction cost

    # Maker/Taker differentiation
    maker_fee_bps: float = 2.0  # Limit order fee
    taker_fee_bps: float = 5.0  # Market order fee

    # Include components
    include_slippage: bool = True
    include_market_impact: bool = True

    @classmethod
    def from_yaml(cls, cfg: DictConfig):
        """
        Create from Hydra config.

        Args:
            cfg: Hydra DictConfig with transaction_costs section

        Returns:
            TransactionCostConfig instance
        """
        return cls(
            fixed_bps=cfg.transaction_costs.fixed_bps,
            maker_fee_bps=cfg.transaction_costs.maker_fee_bps,
            taker_fee_bps=cfg.transaction_costs.taker_fee_bps,
            include_slippage=cfg.transaction_costs.include_slippage,
            include_market_impact=cfg.transaction_costs.include_market_impact,
        )


@dataclass
class SlippageConfig:
    """
    Slippage configuration for execution quality modeling.

    Slippage is the difference between expected and actual execution price.
    This config controls how slippage is calculated.

    Attributes:
        model_type: Slippage calculation method
            - 'fixed': Constant slippage
            - 'volume_based': Based on order size vs volume (recommended)
            - 'volatility': Based on market volatility
            - 'orderbook': From walking the order book (most accurate)
        fixed_slippage_bps: Base slippage for fixed model
        volume_impact_coef: Coefficient for volume-based model
        volatility_multiplier: Multiplier for volatility model
        max_slippage_bps: Maximum slippage cap

    Example:
        >>> config = SlippageConfig(
        ...     model_type='volume_based',
        ...     fixed_slippage_bps=5.0,
        ...     volume_impact_coef=0.1
        ... )
    """

    model_type: str = "volume_based"
    fixed_slippage_bps: float = 5.0
    volume_impact_coef: float = 0.1
    volatility_multiplier: float = 2.0
    max_slippage_bps: float = 100.0

    @classmethod
    def from_yaml(cls, cfg: DictConfig):
        """
        Create from Hydra config.

        Args:
            cfg: Hydra DictConfig with slippage section

        Returns:
            SlippageConfig instance
        """
        return cls(
            model_type=cfg.slippage.model_type,
            fixed_slippage_bps=cfg.slippage.fixed_slippage_bps,
            volume_impact_coef=cfg.slippage.volume_impact_coef,
            volatility_multiplier=cfg.slippage.volatility_multiplier,
            max_slippage_bps=cfg.slippage.max_slippage_bps,
        )


@dataclass
class OrderBookConfig:
    """
    Order book simulation configuration.

    Controls the Level 2 market simulation for realistic slippage
    and execution quality modeling.

    Attributes:
        enabled: Whether to simulate order book
        n_levels: Number of price levels to simulate
        base_spread_bps: Base bid-ask spread in basis points
        volatility_spread_multiplier: How much volatility affects spread
        depth_factor: Liquidity depth multiplier
        depth_decay_rate: Rate at which volume decreases with distance
        impact_coefficient: Market impact coefficient

    Example:
        >>> config = OrderBookConfig(
        ...     enabled=True,
        ...     n_levels=10,
        ...     base_spread_bps=5.0
        ... )
    """

    enabled: bool = True
    n_levels: int = 10

    # Spread dynamics
    base_spread_bps: float = 5.0
    volatility_spread_multiplier: float = 10.0

    # Depth dynamics
    depth_factor: float = 1.0
    depth_decay_rate: float = 0.3

    # Market impact
    impact_coefficient: float = 0.1

    @classmethod
    def from_yaml(cls, cfg: DictConfig):
        """
        Create from Hydra config.

        Args:
            cfg: Hydra DictConfig with orderbook section

        Returns:
            OrderBookConfig instance
        """
        return cls(
            enabled=cfg.orderbook.enabled,
            n_levels=cfg.orderbook.n_levels,
            base_spread_bps=cfg.orderbook.base_spread_bps,
            volatility_spread_multiplier=cfg.orderbook.volatility_spread_multiplier,
            depth_factor=cfg.orderbook.depth_factor,
            depth_decay_rate=cfg.orderbook.depth_decay_rate,
            impact_coefficient=cfg.orderbook.impact_coefficient,
        )


@dataclass
class RewardComponent:
    """
    Single reward component for dynamic reward shaping.

    Defines a component of the reward signal that can be combined
    with other components to create custom reward functions.

    Attributes:
        name: Component name
            - 'return': Portfolio return
            - 'sharpe': Sharpe ratio bonus
            - 'drawdown': Drawdown penalty
            - 'transaction_cost': Trading cost penalty
        weight: Component weight in final reward
        lookback: Period for calculating component (for sharpe)

    Example:
        >>> comp = RewardComponent(name='sharpe', weight=0.1, lookback=20)
    """

    name: str
    weight: float
    lookback: Optional[int] = None


@dataclass
class RewardConfig:
    """
    Reward shaping configuration.

    Allows customization of the reward signal through weighted
    components. This enables fine-tuning of agent behavior.

    Attributes:
        components: List of RewardComponent objects
        clip_min: Minimum reward value after scaling
        clip_max: Maximum reward value after scaling
        scale: Multiplier for final reward

    Example:
        >>> config = RewardConfig(
        ...     components=[
        ...         RewardComponent('return', 1.0),
        ...         RewardComponent('sharpe', 0.1, lookback=20)
        ...     ],
        ...     clip_min=-10.0,
        ...     clip_max=10.0,
        ...     scale=100.0
        ... )
    """

    components: List[RewardComponent] = field(default_factory=list)
    clip_min: float = -10.0
    clip_max: float = 10.0
    scale: float = 100.0

    @classmethod
    def from_yaml(cls, cfg: DictConfig):
        """
        Create from Hydra config.

        Args:
            cfg: Hydra DictConfig with reward section

        Returns:
            RewardConfig instance
        """
        components = []
        for comp_cfg in cfg.reward.components:
            components.append(
                RewardComponent(
                    name=comp_cfg.name,
                    weight=comp_cfg.weight,
                    lookback=comp_cfg.get("lookback", None),
                )
            )

        return cls(
            components=components,
            clip_min=cfg.reward.clip_min,
            clip_max=cfg.reward.clip_max,
            scale=cfg.reward.scale,
        )


@dataclass
class MarketRegime:
    """
    Market regime definition.

    Represents a distinct market condition with specific
    volatility and volume characteristics.

    Attributes:
        name: Regime identifier (e.g., 'low_vol', 'high_vol')
        volatility: Typical volatility (e.g., 0.02 = 2%)
        volume: Typical volume level
        spread: Typical spread in bps

    Example:
        >>> regime = MarketRegime('high_vol', 0.05, 1000.0, 10.0)
    """

    name: str
    volatility: float
    volume: float
    spread: float


@dataclass
class MarketConfig:
    """
    Market dynamics configuration.

    Defines market regimes and their characteristics for
    simulation and training diversity.

    Attributes:
        vol_regimes: Dict mapping regime name to volatility value
            Can be simple: {'low_vol': 0.01, 'high_vol': 0.05}
            Or complex: {'low_vol': {'volatility': 0.01, 'volume': 500}}
        volume_patterns: Dict of volume patterns by regime
        spread_patterns: Dict of spread patterns by regime

    Example:
        >>> config = MarketConfig(
        ...     vol_regimes={'normal': 0.02, 'high_vol': 0.05},
        ...     volume_patterns={'normal': 500.0, 'high_vol': 1000.0},
        ...     spread_patterns={'normal': 5.0, 'high_vol': 15.0}
        ... )
    """

    # Volatility regimes
    vol_regimes: Dict[str, float] = field(default_factory=dict)

    # Volume patterns
    volume_patterns: Dict[str, float] = field(default_factory=dict)

    # Spread patterns
    spread_patterns: Dict[str, float] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, cfg: DictConfig):
        """
        Create from Hydra config.

        Args:
            cfg: Hydra DictConfig with market section

        Returns:
            MarketConfig instance
        """
        return cls(
            vol_regimes=dict(cfg.market.vol_regimes),
            volume_patterns=dict(cfg.market.volume_patterns),
            spread_patterns=dict(cfg.market.spread_patterns),
        )

    def get_regime(self, regime_name: str) -> MarketRegime:
        """
        Get market regime by name.

        Extracts regime parameters and returns MarketRegime object.
        Handles both simple (float) and complex (dict) regime definitions.

        Args:
            regime_name: Name of regime to retrieve

        Returns:
            MarketRegime with regime parameters

        Example:
            >>> regime = config.get_regime('high_vol')
            >>> print(f"Vol: {regime.volatility}")
        """
        # Handle different regime definition formats

        # Get volatility value
        vol_data = self.vol_regimes.get(
            regime_name, next(iter(self.vol_regimes.values()), {})
        )
        if isinstance(vol_data, dict):
            volatility = vol_data.get("volatility", 0.02)
        else:
            volatility = float(vol_data) if vol_data else 0.02

        # Get volume value
        vol_for_volume = self.vol_regimes.get(
            regime_name, next(iter(self.vol_regimes.values()), {})
        )
        if isinstance(vol_for_volume, dict):
            volume = vol_for_volume.get("volume", 500.0)
        else:
            volume = self.volume_patterns.get(regime_name, 500.0)

        # Get spread value
        spread = self.spread_patterns.get(regime_name, 5.0)

        return MarketRegime(
            name=regime_name,
            volatility=volatility,
            volume=volume,
            spread=spread,
        )


@dataclass
class EnvironmentConfig:
    """
    Complete environment configuration.

    This is the top-level configuration class that contains
    all settings for the trading environment.

    General Settings:
    -----------------
    - type: Environment type identifier
    - initial_capital: Starting capital
    - max_position_size: Maximum position as fraction of capital
    - min_position_size: Minimum position threshold

    Risk Controls:
    --------------
    - max_drawdown: Maximum allowed drawdown before halting
    - max_consecutive_losses: Maximum loss streak before halting

    Episode Settings:
    -----------------
    - lookback_window: Historical data for initial state
    - max_steps: Maximum steps per episode

    Observation Settings:
    ---------------------
    - include_orderbook_features: Include L2 features
    - include_portfolio_metrics: Include portfolio state
    - normalize_observations: Normalize feature values

    Sub-Configs:
    ------------
    - transaction_costs: Fee configuration
    - slippage: Slippage model configuration
    - orderbook: Order book simulation
    - reward: Reward shaping
    - market: Market regime definitions

    Example:
        >>> config = EnvironmentConfig(
        ...     type='realistic',
        ...     initial_capital=100000.0,
        ...     max_position_size=1.0,
        ...     max_drawdown=0.20
        ... )
    """

    # Environment type
    type: str = "realistic"

    # Capital
    initial_capital: float = 100000.0

    # Position sizing
    max_position_size: float = 1.0
    min_position_size: float = 0.01

    # Risk controls
    max_drawdown: float = 0.20
    max_consecutive_losses: int = 5

    # Episode
    lookback_window: int = 50
    max_steps: int = 5000

    # Observation
    include_orderbook_features: bool = True
    include_portfolio_metrics: bool = True
    normalize_observations: bool = True

    # Sub-configs
    transaction_costs: TransactionCostConfig = field(
        default_factory=TransactionCostConfig
    )
    slippage: SlippageConfig = field(default_factory=SlippageConfig)
    orderbook: OrderBookConfig = field(default_factory=OrderBookConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    market: MarketConfig = field(default_factory=MarketConfig)

    @classmethod
    def from_yaml(cls, cfg: DictConfig):
        """
        Create complete config from Hydra config.

        Args:
            cfg: Hydra DictConfig with environment section

        Returns:
            EnvironmentConfig with all parameters loaded

        Example:
            >>> config = EnvironmentConfig.from_yaml(cfg)
        """
        return cls(
            type=cfg.environment.type,
            initial_capital=cfg.environment.initial_capital,
            max_position_size=cfg.environment.max_position_size,
            min_position_size=cfg.environment.min_position_size,
            max_drawdown=cfg.environment.max_drawdown,
            max_consecutive_losses=cfg.environment.max_consecutive_losses,
            lookback_window=cfg.environment.lookback_window,
            max_steps=cfg.environment.max_steps,
            include_orderbook_features=cfg.environment.include_orderbook_features,
            include_portfolio_metrics=cfg.environment.include_portfolio_metrics,
            normalize_observations=cfg.environment.normalize_observations,
            transaction_costs=TransactionCostConfig.from_yaml(cfg),
            slippage=SlippageConfig.from_yaml(cfg),
            orderbook=OrderBookConfig.from_yaml(cfg),
            reward=RewardConfig.from_yaml(cfg),
            market=MarketConfig.from_yaml(cfg),
        )


# ============================================================================
# YAML LOADER (Standalone)
# ============================================================================


def load_environment_config_from_yaml(yaml_path: str) -> EnvironmentConfig:
    """
    Load environment config from YAML file without Hydra.

    This function allows loading configuration from YAML files
    in standalone mode (without Hydra framework). It supports
    both nested and flat YAML structures.

    Args:
        yaml_path: Path to YAML configuration file

    Returns:
        EnvironmentConfig with all parameters loaded

    Raises:
        FileNotFoundError: If YAML file doesn't exist
        yaml.YAMLError: If YAML is malformed

    Example:
        >>> config = load_environment_config_from_yaml('config/realistic_env.yaml')
        >>> print(f"Maker fee: {config.transaction_costs.maker_fee_bps}")
    """
    with open(yaml_path, "r") as f:
        yaml_dict = yaml.safe_load(f)

    # Manual construction (when not using Hydra)
    config = EnvironmentConfig()

    # Handle both nested and flat YAML formats
    env_cfg = yaml_dict.get("environment", yaml_dict)

    # General settings
    config.type = env_cfg.get("type", "realistic")
    config.initial_capital = env_cfg.get("initial_capital", 100000.0)
    config.max_position_size = env_cfg.get("max_position_size", 1.0)
    config.max_drawdown = env_cfg.get("max_drawdown", 0.20)
    config.max_consecutive_losses = env_cfg.get("max_consecutive_losses", 5)
    config.lookback_window = env_cfg.get("lookback_window", 50)
    config.max_steps = env_cfg.get("max_steps", 5000)

    # Transaction costs
    tc_cfg = yaml_dict.get("transaction_costs", env_cfg.get("transaction_costs", {}))
    config.transaction_costs = TransactionCostConfig(
        fixed_bps=tc_cfg.get("fixed_bps", 5.0),
        maker_fee_bps=tc_cfg.get("maker_fee_bps", 2.0),
        taker_fee_bps=tc_cfg.get("taker_fee_bps", 5.0),
        include_slippage=tc_cfg.get("include_slippage", True),
        include_market_impact=tc_cfg.get("include_market_impact", True),
    )

    # Slippage configuration
    slip_cfg = yaml_dict.get("slippage", env_cfg.get("slippage", {}))
    config.slippage = SlippageConfig(
        model_type=slip_cfg.get("model_type", "volume_based"),
        fixed_slippage_bps=slip_cfg.get("fixed_slippage_bps", 5.0),
        volume_impact_coef=slip_cfg.get("volume_impact_coef", 0.1),
        volatility_multiplier=slip_cfg.get("volatility_multiplier", 2.0),
        max_slippage_bps=slip_cfg.get("max_slippage_bps", 100.0),
    )

    # Order book configuration
    ob_cfg = yaml_dict.get("orderbook", env_cfg.get("orderbook", {}))
    config.orderbook = OrderBookConfig(
        enabled=ob_cfg.get("enabled", True),
        n_levels=ob_cfg.get("n_levels", 10),
        base_spread_bps=ob_cfg.get("base_spread_bps", 5.0),
        volatility_spread_multiplier=ob_cfg.get("volatility_spread_multiplier", 10.0),
        depth_factor=ob_cfg.get("depth_factor", 1.0),
        depth_decay_rate=ob_cfg.get("depth_decay_rate", 0.3),
        impact_coefficient=ob_cfg.get("impact_coefficient", 0.1),
    )

    # Reward configuration
    reward_cfg = yaml_dict.get("reward", env_cfg.get("reward", {}))
    components = []
    for comp in reward_cfg.get("components", []):
        components.append(
            RewardComponent(
                name=comp["name"],
                weight=comp["weight"],
                lookback=comp.get("lookback", None),
            )
        )

    config.reward = RewardConfig(
        components=components,
        clip_min=reward_cfg.get("clip_min", -10.0),
        clip_max=reward_cfg.get("clip_max", 10.0),
        scale=reward_cfg.get("scale", 100.0),
    )

    # Market configuration
    market_cfg = yaml_dict.get("market", {})
    config.market = MarketConfig(
        vol_regimes=market_cfg.get("vol_regimes", {}),
        volume_patterns=market_cfg.get("volume_patterns", {}),
        spread_patterns=market_cfg.get("spread_patterns", {}),
    )

    return config


if __name__ == "__main__":
    print("=" * 80)
    print("CONFIG SYSTEM TEST")
    print("=" * 80)

    # Test loading from YAML
    config_path = (
        Path(__file__).parent.parent.parent
        / "config"
        / "environment"
        / "realistic_env.yaml"
    )

    if config_path.exists():
        config = load_environment_config_from_yaml(str(config_path))

        print("\n✓ Config loaded from YAML")
        print(f"\nEnvironment Type: {config.type}")
        print(f"Initial Capital: ${config.initial_capital:,.0f}")
        print(f"\nTransaction Costs:")
        print(f"  Maker Fee: {config.transaction_costs.maker_fee_bps} bps")
        print(f"  Taker Fee: {config.transaction_costs.taker_fee_bps} bps")
        print(f"\nSlippage Model: {config.slippage.model_type}")
        print(f"\nOrder Book:")
        print(f"  Enabled: {config.orderbook.enabled}")
        print(f"  Levels: {config.orderbook.n_levels}")
        print(f"  Decay Rate: {config.orderbook.depth_decay_rate}")
        print(f"\nReward Components:")
        for comp in config.reward.components:
            print(f"  {comp.name}: weight={comp.weight}")
        print(f"\nMarket Regimes:")
        for regime_name in config.market.vol_regimes.keys():
            regime = config.market.get_regime(regime_name)
            print(f"  {regime.name}: vol={regime.volatility}, volume={regime.volume}")
    else:
        print(f"⚠ Config file not found: {config_path}")

    print("\n" + "=" * 80)
    print("✓ CONFIG SYSTEM TEST COMPLETE")
    print("=" * 80)
