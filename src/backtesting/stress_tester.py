"""
Black Swan Stress-Testing
=========================
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml
from loguru import logger

from backtesting.strategy_evolution import SignalBacktester, SignalBuilder, StrategyResult

def load_stress_config(cfg_path: str = "config/stress_test.yaml") -> dict:
    p = Path(cfg_path)
    if not p.exists(): return {}
    return yaml.safe_load(p.read_text()) or {}

class CrashSimulator:
    def __init__(self, config: dict):
        self._cfg = config.get("scenarios", {})

    def inject_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if self._cfg.get("flash_crash", {}).get("enabled", True):
            df = self._simulate_flash_crash(df)
        if self._cfg.get("volatility_spike", {}).get("enabled", True):
            df = self._simulate_volatility_spike(df)
        return df

    def _simulate_flash_crash(self, df: pd.DataFrame) -> pd.DataFrame:
        cfg = self._cfg.get("flash_crash", {})
        drop, dur = cfg.get("drop_pct", 0.20), cfg.get("duration_candles", 5)
        if len(df) < dur * 2: return df
        start = len(df) // 2
        for i, f in enumerate(np.linspace(1.0, 1.0 - drop, dur)):
            df.iloc[start + i, df.columns.get_loc('close')] *= f
        return df

    def _simulate_volatility_spike(self, df: pd.DataFrame) -> pd.DataFrame:
        mult = self._cfg.get("volatility_spike", {}).get("multiplier", 3.0)
        noise = np.random.normal(0, df['close'].pct_change().std() * mult, len(df))
        df['close'] *= (1 + noise)
        return df

class StressTestEngine:
    def __init__(self, stress_cfg_path: str = "config/stress_test.yaml"):
        self.config = load_stress_config(stress_cfg_path)
        self.simulator = CrashSimulator(self.config)
        self.criteria = self.config.get("elite_criteria", {})
        from backtesting.strategy_evolution import load_evolution_config
        evo_cfg = load_evolution_config()
        self.builder = SignalBuilder(evo_cfg.get("strategies", {}))
        self.backtester = SignalBacktester(fee_bps=10, slippage_bps=30) # High slippage stress

    def run_arena(self, df: pd.DataFrame) -> List[StrategyResult]:
        logger.info("ELITE ARENA START")
        stressed_df = self.simulator.inject_anomalies(df)
        stressed_df, signal_cols = self.builder.build_all(stressed_df)
        results = []
        for col in signal_cols:
            bt = self.backtester.backtest(stressed_df, col, min_trades=5)
            max_dd, sharpe = self._calculate_stress_metrics(stressed_df, col)
            status = "🏆 QUALIFIZIERT" if (max_dd > self.criteria.get("max_drawdown_limit", -0.15) and sharpe > self.criteria.get("min_sharpe_ratio", 1.0)) else "💀 ELIMINIERT"
            results.append(StrategyResult(col.replace("_sig_",""), bt["profit_factor"], bt["profit_factor"], bt["win_rate"], bt["n_trades"], max_dd, sharpe, status, col))
        return results

    def _calculate_stress_metrics(self, df, col):
        returns = df[col].shift(1) * df['close'].pct_change()
        cum_ret = (1 + returns.fillna(0)).cumprod()
        max_dd = (cum_ret - cum_ret.cummax()) / cum_ret.cummax()
        sharpe = (returns.mean() / (returns.std() + 1e-8) * np.sqrt(252*24)) if returns.std() > 0 else 0
        return float(max_dd.min()), float(sharpe)

def run_black_swan_test(df: pd.DataFrame):
    return StressTestEngine().run_arena(df)
