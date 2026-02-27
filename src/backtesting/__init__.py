"""Backtesting package for BITCOIN4Traders."""

from src.backtesting.performance_calculator import (
    PerformanceCalculator,
    PerformanceMetrics,
)
from src.backtesting.visualizer import BacktestVisualizer
from src.backtesting.walkforward_engine import WalkForwardEngine, WalkForwardConfig
from src.backtesting.strategy_evolution import (
    StrategyEvolution,
    WalkForwardStrategyEvolution,
    EvolutionReport,
    StrategyResult,
    evolve_strategy,
    load_evolution_config,
)
from src.backtesting.stress_tester import (
    StressTestEngine,
    run_black_swan_test,
)

__all__ = [
    # Performance
    "PerformanceCalculator",
    "PerformanceMetrics",
    # Visualizer
    "BacktestVisualizer",
    # Walk-Forward
    "WalkForwardEngine",
    "WalkForwardConfig",
    # Strategy Evolution (Darwin-Ansatz)
    "StrategyEvolution",
    "WalkForwardStrategyEvolution",
    "EvolutionReport",
    "StrategyResult",
    "evolve_strategy",
    "load_evolution_config",
    # Stress Testing (Black Swan)
    "StressTestEngine",
    "run_black_swan_test",
]
