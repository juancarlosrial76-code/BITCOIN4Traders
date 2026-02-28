"""
Complete Configuration System - YAML Integration
=================================================
Fully connects YAML configuration to code implementation.

Fixes:
1. Maker/Taker fee differentiation
2. Dynamic reward components
3. Market regime simulation
4. Hydra integration
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
from pathlib import Path
import yaml
from omegaconf import DictConfig


@dataclass
class TransactionCostConfig:
    """Transaction cost configuration from YAML."""

    # Fixed costs
    fixed_bps: float = 5.0  # Base transaction cost

    # Maker/Taker differentiation (NEW!)
    maker_fee_bps: float = 2.0  # Limit order fee
    taker_fee_bps: float = 5.0  # Market order fee

    # Include components
    include_slippage: bool = True
    include_market_impact: bool = True

    @classmethod
    def from_yaml(cls, cfg: DictConfig):
        """Create from Hydra config."""
        return cls(
            fixed_bps=cfg.transaction_costs.fixed_bps,
            maker_fee_bps=cfg.transaction_costs.maker_fee_bps,
            taker_fee_bps=cfg.transaction_costs.taker_fee_bps,
            include_slippage=cfg.transaction_costs.include_slippage,
            include_market_impact=cfg.transaction_costs.include_market_impact,
        )


@dataclass
class SlippageConfig:
    """Slippage configuration from YAML."""

    model_type: str = "volume_based"
    fixed_slippage_bps: float = 5.0
    volume_impact_coef: float = 0.1
    volatility_multiplier: float = 2.0
    max_slippage_bps: float = 100.0

    @classmethod
    def from_yaml(cls, cfg: DictConfig):
        """Create from Hydra config."""
        return cls(
            model_type=cfg.slippage.model_type,
            fixed_slippage_bps=cfg.slippage.fixed_slippage_bps,
            volume_impact_coef=cfg.slippage.volume_impact_coef,
            volatility_multiplier=cfg.slippage.volatility_multiplier,
            max_slippage_bps=cfg.slippage.max_slippage_bps,
        )


@dataclass
class OrderBookConfig:
    """Order book configuration from YAML."""

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
        """Create from Hydra config."""
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
    """Single reward component."""

    name: str
    weight: float
    lookback: Optional[int] = None


@dataclass
class RewardConfig:
    """Reward shaping configuration from YAML."""

    components: List[RewardComponent] = field(default_factory=list)
    clip_min: float = -10.0
    clip_max: float = 10.0
    scale: float = 100.0

    @classmethod
    def from_yaml(cls, cfg: DictConfig):
        """Create from Hydra config."""
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
    """Market regime definition."""

    name: str
    volatility: float
    volume: float
    spread: float


@dataclass
class MarketConfig:
    """Market dynamics configuration from YAML."""

    # Volatility regimes
    vol_regimes: Dict[str, float] = field(default_factory=dict)

    # Volume patterns
    volume_patterns: Dict[str, float] = field(default_factory=dict)

    # Spread patterns
    spread_patterns: Dict[str, float] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, cfg: DictConfig):
        """Create from Hydra config."""
        return cls(
            vol_regimes=dict(cfg.market.vol_regimes),
            volume_patterns=dict(cfg.market.volume_patterns),
            spread_patterns=dict(cfg.market.spread_patterns),
        )

    def get_regime(self, regime_name: str) -> MarketRegime:
        """Get market regime by name."""
        # Fallback: first available regime, then hardcoded defaults
        _first_vol = next(iter(self.vol_regimes.values()), 0.02) if self.vol_regimes else 0.02
        _first_vol = (
            _first_vol if not isinstance(_first_vol, dict) else _first_vol.get("volatility", 0.02)
        )
        _first_vol_full = self.vol_regimes.get(
            regime_name, next(iter(self.vol_regimes.values()), {})
        )
        if isinstance(_first_vol_full, dict):
            vol = _first_vol_full.get("volatility", 0.02)
        else:
            vol = float(_first_vol_full) if _first_vol_full else 0.02

        _first_vol_map = self.vol_regimes.get(
            regime_name, next(iter(self.vol_regimes.values()), {})
        )
        volume = (
            _first_vol_map.get("volume", 500.0)
            if isinstance(_first_vol_map, dict)
            else self.volume_patterns.get(regime_name, 500.0)
        )
        spread = self.spread_patterns.get(regime_name, 5.0)

        return MarketRegime(
            name=regime_name,
            volatility=vol,
            volume=volume,
            spread=spread,
        )


@dataclass
class EnvironmentConfig:
    """Complete environment configuration from YAML."""

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

    # Sub-configs (NEW!)
    transaction_costs: TransactionCostConfig = field(default_factory=TransactionCostConfig)
    slippage: SlippageConfig = field(default_factory=SlippageConfig)
    orderbook: OrderBookConfig = field(default_factory=OrderBookConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    market: MarketConfig = field(default_factory=MarketConfig)

    @classmethod
    def from_yaml(cls, cfg: DictConfig):
        """Create complete config from Hydra config."""
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
    Load environment config from YAML file (without Hydra).

    For testing/debugging without Hydra framework.
    """
    with open(yaml_path, "r") as f:
        yaml_dict = yaml.safe_load(f)

    # Manual construction (when not using Hydra)
    config = EnvironmentConfig()

    # Check if YAML has 'environment' key or is flat
    env_cfg = yaml_dict.get("environment", yaml_dict)
    config.type = env_cfg.get("type", "realistic")
    config.initial_capital = env_cfg.get("initial_capital", 100000.0)
    config.max_position_size = env_cfg.get("max_position_size", 1.0)
    config.max_drawdown = env_cfg.get("max_drawdown", 0.20)
    config.max_consecutive_losses = env_cfg.get("max_consecutive_losses", 5)
    config.lookback_window = env_cfg.get("lookback_window", 50)
    config.max_steps = env_cfg.get("max_steps", 5000)

    # Transaction costs (check both nested and flat)
    tc_cfg = yaml_dict.get("transaction_costs", env_cfg.get("transaction_costs", {}))
    config.transaction_costs = TransactionCostConfig(
        fixed_bps=tc_cfg.get("fixed_bps", 5.0),
        maker_fee_bps=tc_cfg.get("maker_fee_bps", 2.0),
        taker_fee_bps=tc_cfg.get("taker_fee_bps", 5.0),
        include_slippage=tc_cfg.get("include_slippage", True),
        include_market_impact=tc_cfg.get("include_market_impact", True),
    )

    # Slippage
    slip_cfg = yaml_dict.get("slippage", env_cfg.get("slippage", {}))
    config.slippage = SlippageConfig(
        model_type=slip_cfg.get("model_type", "volume_based"),
        fixed_slippage_bps=slip_cfg.get("fixed_slippage_bps", 5.0),
        volume_impact_coef=slip_cfg.get("volume_impact_coef", 0.1),
        volatility_multiplier=slip_cfg.get("volatility_multiplier", 2.0),
        max_slippage_bps=slip_cfg.get("max_slippage_bps", 100.0),
    )

    # Order book
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

    # Reward
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

    # Market
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
        Path(__file__).parent.parent.parent / "config" / "environment" / "realistic_env.yaml"
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
