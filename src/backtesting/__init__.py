"""Backtesting package for BITCOIN4Traders."""

from backtesting.performance_calculator import PerformanceCalculator, PerformanceMetrics
from backtesting.visualizer import BacktestVisualizer
from backtesting.walkforward_engine import WalkForwardEngine, WalkForwardConfig
from backtesting.strategy_evolution import (
    StrategyEvolution,
    WalkForwardStrategyEvolution,
    EvolutionReport,
    StrategyResult,
    evolve_strategy,
    load_evolution_config,
)
from backtesting.stress_tester import (
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
