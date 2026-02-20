"""
Advanced Feature Engineering Module
====================================
Professional-grade features for quantitative trading.
"""

from src.features.multi_timeframe import (
    MultiTimeframeAnalyzer,
    MarketStructureAnalyzer,
    Timeframe,
    SignalStrength,
    create_multi_timeframe_features,
)

from src.features.microstructure import (
    MicrostructureAnalyzer,
    LiquidityAnalyzer,
    OrderFlowMetrics,
    calculate_spread_metrics,
    create_microstructure_features,
)

from src.features.feature_engine import (
    FeatureEngine,
    FeatureConfig,
)

__all__ = [
    # Multi-timeframe Analysis
    "MultiTimeframeAnalyzer",
    "MarketStructureAnalyzer",
    "Timeframe",
    "SignalStrength",
    "create_multi_timeframe_features",
    # Microstructure Analysis
    "MicrostructureAnalyzer",
    "LiquidityAnalyzer",
    "OrderFlowMetrics",
    "calculate_spread_metrics",
    "create_microstructure_features",
    # Core Feature Engine
    "FeatureEngine",
    "FeatureConfig",
]
