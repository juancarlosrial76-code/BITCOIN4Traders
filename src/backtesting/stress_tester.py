"""
Black Swan Stress-Testing Module
================================
Stress testing framework for trading strategies under extreme market conditions.

This module implements a "Darwinian" approach to strategy validation by
exposing trading strategies to simulated Black Swan events - rare but
catastrophic market conditions that standard backtesting cannot anticipate.

The goal is the "Elite Arena" concept: strategies must survive extreme
stress scenarios to be considered viable for live trading.

Stress Scenarios Implemented:
-----------------------------
1. Flash Crash Simulation
   - Sudden, severe price drop over short period
   - Configurable drop percentage and duration
   - Tests stop-loss execution and liquidity assumptions

2. Volatility Spike Simulation
   - Dramatic increase in price volatility
   - Multiplier applied to historical volatility
   - Tests position sizing and risk management

Why Stress Testing Matters:
---------------------------
Standard backtesting assumes historical patterns continue. However,
markets can experience:
- Flash crashes (e.g., 2010 Flash Crash, COVID-19 March 2020)
- Volatility regime changes (sudden increase in turbulence)
- Correlation breakdowns (usually uncorrelated assets move together)
- Liquidity crises (unable to exit positions at desired prices)

Strategies that look excellent in backtesting may fail catastrophically
in these scenarios. Stress testing identifies these weaknesses BEFORE
live trading.

Elite Criteria:
---------------
A strategy qualifies for live trading only if it maintains:
- Maximum drawdown within acceptable limits (e.g., -15%)
- Sharpe ratio above minimum threshold (e.g., 1.0)

This is the "survival of the fittest" approach - only robust strategies
proceed to live trading.

Usage:
------
    from backtesting.stress_tester import StressTestEngine, run_black_swan_test

    # Using the full engine with custom configuration
    engine = StressTestEngine(stress_cfg_path="config/stress_test.yaml")
    results = engine.run_arena(market_data)

    # Quick test with default settings
    results = run_black_swan_test(market_data)

    # Analyze results
    for r in results:
        print(f"{r.name}: {r.status} (Sharpe: {r.sharpe:.2f}, MaxDD: {r.max_drawdown:.2%})")

Configuration File (stress_test.yaml):
---------------------------------------
    scenarios:
        flash_crash:
            enabled: true
            drop_pct: 0.20      # 20% price drop
            duration_candles: 5 # Over 5 candles

        volatility_spike:
            enabled: true
            multiplier: 3.0     # 3x normal volatility

    elite_criteria:
        max_drawdown_limit: -0.15   # Max 15% drawdown allowed
        min_sharpe_ratio: 1.0       # Min Sharpe of 1.0

Reference:
---------
- Taleb, N.N. (2007): "The Black Swan"
- Lopez de Prado, M. (2018): "Advances in Financial Machine Learning"
- Markowitz, H. (1952): "Portfolio Selection"
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml
from loguru import logger

from backtesting.strategy_evolution import (
    SignalBacktester,
    SignalBuilder,
    StrategyResult,
)


def load_stress_config(cfg_path: str = "config/stress_test.yaml") -> dict:
    p = Path(cfg_path)
    if not p.exists():
        return {}
    return yaml.safe_load(p.read_text()) or {}


class CrashSimulator:
    """
    Simulates extreme market crash scenarios for stress testing.

    This class injects artificial anomalies into historical price data
    to simulate Black Swan events. It modifies the closing prices in
    ways that represent realistic but extreme market conditions.

    The simulator reads configuration from a YAML file defining:
    - Flash crash parameters (drop percentage, duration)
    - Volatility spike parameters (multiplier)

    Attributes:
        config: Dictionary containing scenario configurations from YAML

    Methods:
        inject_anomalies(): Apply all enabled stress scenarios to data
        _simulate_flash_crash(): Inject sudden price drop
        _simulate_volatility_spike(): Inject volatility increase

    Example:
        --------
        config = {
            'scenarios': {
                'flash_crash': {
                    'enabled': True,
                    'drop_pct': 0.20,
                    'duration_candles': 5
                },
                'volatility_spike': {
                    'enabled': True,
                    'multiplier': 3.0
                }
            }
        }

        simulator = CrashSimulator(config)
        stressed_data = simulator.inject_anomalies(original_df)

    Note:
        This class always returns a copy of the input DataFrame,
        never modifying the original data.
    """

    def __init__(self, config: dict):
        """
        Initialize the crash simulator with configuration.

        Parameters:
            config: Dictionary containing scenario definitions from YAML.
                   Should have 'scenarios' key with 'flash_crash' and/or
                   'volatility_spike' sub-configurations.
        """
        self._cfg = config.get("scenarios", {})

    def inject_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()  # never mutate the original dataframe
        if self._cfg.get("flash_crash", {}).get("enabled", True):
            df = self._simulate_flash_crash(df)
        if self._cfg.get("volatility_spike", {}).get("enabled", True):
            df = self._simulate_volatility_spike(df)
        return df

    def _simulate_flash_crash(self, df: pd.DataFrame) -> pd.DataFrame:
        cfg = self._cfg.get("flash_crash", {})
        drop, dur = cfg.get("drop_pct", 0.20), cfg.get("duration_candles", 5)
        if len(df) < dur * 2:
            return df  # not enough bars to inject crash
        start = len(df) // 2  # inject crash in the middle of the series
        # linearly ramp close price down by `drop` fraction over `dur` candles
        for i, f in enumerate(np.linspace(1.0, 1.0 - drop, dur)):
            df.iloc[start + i, df.columns.get_loc("close")] *= f
        return df

    def _simulate_volatility_spike(self, df: pd.DataFrame) -> pd.DataFrame:
        mult = self._cfg.get("volatility_spike", {}).get("multiplier", 3.0)
        # noise scaled to `mult` × historical volatility, applied to every bar
        noise = np.random.normal(0, df["close"].pct_change().std() * mult, len(df))
        df["close"] *= 1 + noise
        return df


class StressTestEngine:
    """
    Main stress testing engine implementing the Elite Arena concept.

    This engine runs trading strategies through a gauntlet of stress
    scenarios to determine which strategies survive extreme market
    conditions. Only strategies meeting the "elite criteria" are
    marked as qualified for live trading.

    The engine:
    1. Loads stress scenario configuration
    2. Creates a CrashSimulator for generating extreme conditions
    3. Builds all enabled strategies from configuration
    4. Backtests each strategy on stressed data
    5. Evaluates against elite criteria (max drawdown, min Sharpe)
    6. Returns qualified/eliminated status for each strategy

    Attributes:
        config: Full configuration dictionary from YAML
        simulator: CrashSimulator instance for generating anomalies
        criteria: Dictionary of elite criteria thresholds

    Methods:
        run_arena(): Execute the complete stress testing gauntlet
        _calculate_stress_metrics(): Compute performance under stress

    Example:
        --------
        # With custom configuration file
        engine = StressTestEngine(stress_cfg_path="config/my_stress.yaml")
        results = engine.run_arena(market_data)

        # Analyze results
        qualified = [r for r in results if "QUALIFIED" in r.status]
        eliminated = [r for r in results if "ELIMINATED" in r.status]

        print(f"Qualified: {len(qualified)}, Eliminated: {len(eliminated)}")

        # Quick test with defaults
        results = run_black_swan_test(market_data)

    Elite Criteria:
    --------------
    By default, a strategy must satisfy BOTH conditions:
    - Maximum Drawdown > -15% (i.e., not losing more than 15%)
    - Sharpe Ratio > 1.0 (risk-adjusted returns acceptable)

    These can be customized in the stress_test.yaml configuration file.
    """

    def __init__(self, stress_cfg_path: str = "config/stress_test.yaml"):
        """
        Initialize the stress test engine.

        Parameters:
            stress_cfg_path: Path to YAML configuration file with stress
                            scenarios and elite criteria.
        """
        self.config = load_stress_config(stress_cfg_path)
        self.simulator = CrashSimulator(self.config)
        self.criteria = self.config.get("elite_criteria", {})
        from backtesting.strategy_evolution import load_evolution_config

        evo_cfg = load_evolution_config()
        self.builder = SignalBuilder(evo_cfg.get("strategies", {}))
        self.backtester = SignalBacktester(
            fee_bps=10, slippage_bps=30
        )  # High slippage stress

    def run_arena(self, df: pd.DataFrame) -> List[StrategyResult]:
        logger.info("ELITE ARENA START")
        stressed_df = self.simulator.inject_anomalies(
            df
        )  # apply synthetic crash/spike scenarios
        stressed_df, signal_cols = self.builder.build_all(stressed_df)
        results = []
        for col in signal_cols:
            bt = self.backtester.backtest(
                stressed_df, col, min_trades=5
            )  # low min_trades for stress test
            max_dd, sharpe = self._calculate_stress_metrics(stressed_df, col)
            # strategy qualifies only if drawdown is within limit AND Sharpe is above minimum
            status = (
                "🏆 QUALIFIED"
                if (
                    max_dd > self.criteria.get("max_drawdown_limit", -0.15)
                    and sharpe > self.criteria.get("min_sharpe_ratio", 1.0)
                )
                else "💀 ELIMINATED"
            )
            results.append(
                StrategyResult(
                    col.replace("_sig_", ""),
                    bt["profit_factor"],
                    bt["profit_factor"],
                    bt["win_rate"],
                    bt["n_trades"],
                    max_dd,
                    sharpe,
                    status,
                    col,
                )
            )
        return results

    def _calculate_stress_metrics(self, df, col):
        returns = (
            df[col].shift(1) * df["close"].pct_change()
        )  # lagged signal × bar return
        cum_ret = (1 + returns.fillna(0)).cumprod()  # running equity index
        max_dd = (
            cum_ret - cum_ret.cummax()
        ) / cum_ret.cummax()  # drawdown series (negative values)
        sharpe = (
            (
                returns.mean() / (returns.std() + 1e-8) * np.sqrt(252 * 24)
            )  # annualised Sharpe (hourly bars)
            if returns.std() > 0
            else 0
        )
        return float(max_dd.min()), float(sharpe)


def run_black_swan_test(df: pd.DataFrame):
    """
    Convenience function to run stress test with default configuration.

    This is a one-liner wrapper that creates a StressTestEngine with
    default settings and runs the full arena. For more control, use
    StressTestEngine directly.

    Parameters:
        df: DataFrame with market data (must have 'close' column)

    Returns:
        List of StrategyResult objects, one per strategy tested.
        Each result includes performance metrics and qualified/eliminated status.

    Example:
        --------
        results = run_black_swan_test(price_data)

        for r in results:
            print(f"{r.name}: {r.status}")
    """
    return StressTestEngine().run_arena(df)
