"""
Strategy Evolution Module
=========================
"Darwin-Ansatz" für BITCOIN4Traders.
"""

from __future__ import annotations

import gc
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml
from loguru import logger

_DEFAULT_CFG_PATH = "config/strategy_evolution.yaml"

def load_evolution_config(cfg_path: str = _DEFAULT_CFG_PATH) -> dict:
    p = Path(cfg_path)
    if not p.exists():
        return {}
    return yaml.safe_load(p.read_text()) or {}

@dataclass
class StrategyResult:
    name: str
    profit_factor: float
    fee_adjusted_pf: float
    win_rate: float
    n_trades: int
    total_return: float
    sharpe: float
    status: str
    signal_column: str

    def is_valid(self) -> bool:
        return self.status == "KEEP"

@dataclass
class EvolutionReport:
    results: List[StrategyResult]
    best_strategy: Optional[StrategyResult]
    n_candles_used: int
    timestamp: str
    config_path: str

    @property
    def ranking(self) -> List[StrategyResult]:
        return sorted(self.results, key=lambda r: r.fee_adjusted_pf, reverse=True)

    def to_dataframe(self) -> pd.DataFrame:
        rows = []
        for r in self.ranking:
            rows.append({
                "Strategie": r.name,
                "PF (brutto)": round(r.profit_factor, 3),
                "PF (nach Gebühr)": round(r.fee_adjusted_pf, 3),
                "Win-Rate %": round(r.win_rate * 100, 1),
                "Trades": r.n_trades,
                "Return %": round(r.total_return * 100, 2),
                "Sharpe": round(r.sharpe, 2),
                "Status": r.status,
            })
        return pd.DataFrame(rows)

class SignalBuilder:
    def __init__(self, strategies_cfg: dict):
        self._cfg = strategies_cfg

    def build_all(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        df = df.copy()
        signal_cols: List[str] = []
        builders = {
            "RSI_Oversold": self._rsi_signal,
            "SMA_Cross": self._sma_cross_signal,
            "Bollinger_Lower": self._bollinger_signal,
            "MACD_Momentum": self._macd_signal,
            "OU_MeanReversion": self._ou_signal,
        }
        for name, scfg in self._cfg.items():
            if not scfg.get("enabled", True): continue
            if name not in builders: continue
            try:
                col_name = f"_sig_{name}"
                df = builders[name](df, scfg, col_name)
                signal_cols.append(col_name)
            except Exception: pass
        return df, signal_cols

    def _rsi_signal(self, df, cfg, col):
        ind = cfg.get("indicator", "rsi_14")
        lower = cfg.get("lower_threshold", 30)
        signal = np.zeros(len(df))
        signal[df[ind] < lower] = 1
        df[col] = signal
        return df

    def _sma_cross_signal(self, df, cfg, col):
        fast_w, slow_w = cfg.get("fast_window", 20), cfg.get("slow_window", 50)
        signal = np.zeros(len(df))
        signal[df["close"].rolling(fast_w).mean() > df["close"].rolling(slow_w).mean()] = 1
        df[col] = signal
        return df

    def _bollinger_signal(self, df, cfg, col):
        ind, lower = cfg.get("indicator", "bb_position"), cfg.get("lower_threshold", 0.1)
        signal = np.zeros(len(df))
        signal[df[ind] < lower] = 1
        df[col] = signal
        return df

    def _macd_signal(self, df, cfg, col):
        ind = cfg.get("indicator", "macd_hist")
        signal = np.zeros(len(df))
        signal[df[ind] > 0] = 1
        df[col] = signal
        return df

    def _ou_signal(self, df, cfg, col):
        ind, lower = cfg.get("indicator", "ou_score"), cfg.get("lower_threshold", -1.5)
        signal = np.zeros(len(df))
        signal[df[ind] < lower] = 1
        df[col] = signal
        return df

class SignalBacktester:
    def __init__(self, fee_bps=10.0, slippage_bps=3.0, signal_shift=1):
        self.fee_bps, self.slippage_bps, self.signal_shift = fee_bps, slippage_bps, signal_shift
        self.total_cost_pct = (fee_bps + slippage_bps) / 10_000

    def backtest(self, df: pd.DataFrame, signal_col: str, min_trades: int = 20) -> Dict:
        if signal_col not in df.columns: return self._empty_result()
        returns = df["close"].pct_change()
        signal = df[signal_col].shift(self.signal_shift).fillna(0)
        strat_returns = signal * returns
        trades_mask = signal.diff().abs() > 0
        n_trades = int(trades_mask.sum())
        if n_trades < min_trades: return {**self._empty_result(), "n_trades": n_trades}
        strat_returns.loc[trades_mask] -= self.total_cost_pct
        gains, losses = strat_returns[strat_returns > 0], strat_returns[strat_returns < 0]
        pf = gains.sum() / abs(losses.sum()) if losses.sum() < 0 else 0.0
        return {
            "profit_factor": pf, "win_rate": len(gains)/(len(gains)+len(losses)) if (len(gains)+len(losses))>0 else 0.0,
            "n_trades": n_trades, "total_return": (1+strat_returns.dropna()).prod()-1,
            "sharpe": (strat_returns.mean()/strat_returns.std()*np.sqrt(252*24)) if strat_returns.std()>0 else 0.0,
            "status": "OK"
        }

    def _empty_result(self): return {"profit_factor": 0, "win_rate": 0, "n_trades": 0, "total_return": 0, "sharpe": 0, "status": "INSUFFICIENT_DATA"}

class StrategyEvolution:
    def __init__(self, cfg_path=_DEFAULT_CFG_PATH):
        self._raw_cfg = load_evolution_config(cfg_path)
        self._cfg_path = cfg_path
        self._val_cfg = self._raw_cfg.get("validation", {})
        self.min_candles, self.min_trades = self._val_cfg.get("min_candles", 500), self._val_cfg.get("min_trades", 20)
        self.pf_threshold, self.fee_pf_thresh = self._val_cfg.get("profit_factor_threshold", 1.5), self._val_cfg.get("fee_adjusted_threshold", 1.2)
        self.lookback, self.reeval_interval = self._raw_cfg.get("backtest", {}).get("lookback_window", 200), self._raw_cfg.get("adaptive", {}).get("reeval_every_n_candles", 50)
        self._builder = SignalBuilder(self._raw_cfg.get("strategies", {}))
        self._backtester = SignalBacktester(self._raw_cfg.get("costs", {}).get("fee_bps", 10))
        self._current_best = None
        self._candle_count = 0

    def evolve(self, df: pd.DataFrame) -> EvolutionReport:
        window_df = df.tail(self.lookback).copy()
        window_df, cols = self._builder.build_all(window_df)
        results = []
        for col in cols:
            bt = self._backtester.backtest(window_df, col, self.min_trades)
            status = "KEEP" if bt["profit_factor"] >= self.pf_threshold else "ELIMINATE"
            if bt["status"] == "INSUFFICIENT_DATA": status = "INSUFFICIENT_DATA"
            results.append(StrategyResult(col.replace("_sig_",""), bt["profit_factor"], bt["profit_factor"], bt["win_rate"], bt["n_trades"], bt["total_return"], bt["sharpe"], status, col))
        results = sorted(results, key=lambda r: r.profit_factor, reverse=True)
        best = next((r for r in results if r.status == "KEEP"), None)
        self._current_best = best
        return EvolutionReport(results, best, len(window_df), "", self._cfg_path)

    def should_reeval(self):
        self._candle_count += 1
        return self.reeval_interval > 0 and self._candle_count % self.reeval_interval == 0

    def get_current_signal(self, df):
        if not self._current_best: return 0
        df_sig, _ = self._builder.build_all(df.tail(10))
        return int(df_sig[self._current_best.signal_column].iloc[-1]) if self._current_best.signal_column in df_sig.columns else 0

def evolve_strategy(df, cfg_path=_DEFAULT_CFG_PATH):
    evo = StrategyEvolution(cfg_path)
    report = evo.evolve(df)
    return report, (report.best_strategy.name if report.best_strategy else None)

class WalkForwardStrategyEvolution:
    def __init__(self, df, n_folds=5, test_pct=0.2, cfg_path=_DEFAULT_CFG_PATH):
        self.df, self.n_folds, self.test_pct, self._evo = df, n_folds, test_pct, StrategyEvolution(cfg_path)
    def run(self):
        # Implementation for notebook analysis
        return pd.DataFrame()
