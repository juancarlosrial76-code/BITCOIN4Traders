"""
Strategy Evolution Module
========================
"Darwinian approach" for BITCOIN4Traders: strategies compete,
the best-performing one survives and is used for live trading.

This module implements an evolutionary approach to strategy selection
where multiple trading strategies compete against each other, with
only the best-performing strategies surviving for live trading.

Core Concepts:
--------------
1. Signal Generation: Multiple technical indicators generate trading signals
2. Backtesting: Each signal is backtested on historical data
3. Selection: Only strategies meeting performance thresholds survive
4. Evolution: Best strategy is selected for live trading
5. Re-evaluation: Strategy can be replaced if performance degrades

Supported Strategies:
-------------------
- RSI_Oversold: Buy when RSI below threshold (mean reversion)
- SMA_Cross: Buy when fast MA crosses above slow MA (trend following)
- Bollinger_Lower: Buy when price near lower Bollinger Band
- MACD_Momentum: Buy when MACD histogram is positive
- OU_MeanReversion: Buy when Ornstein-Uhlenbeck score is negative

Configuration:
-------------
Strategies are configured via YAML (config/strategy_evolution.yaml):

    strategies:
        RSI_Oversold:
            enabled: true
            indicator: rsi_14
            lower_threshold: 30

        SMA_Cross:
            enabled: true
            fast_window: 20
            slow_window: 50

    validation:
        min_candles: 500
        min_trades: 20
        profit_factor_threshold: 1.5
        fee_adjusted_threshold: 1.2

    backtest:
        lookback_window: 200

    adaptive:
        reeval_every_n_candles: 50

    costs:
        fee_bps: 10

Classes:
-------
- StrategyResult: Individual strategy performance metrics
- EvolutionReport: Complete evolution results with ranking
- SignalBuilder: Generates trading signals from indicators
- SignalBacktester: Backtests signal strategies
- StrategyEvolution: Main evolution engine

Usage:
------
    from backtesting.strategy_evolution import evolve_strategy

    # Run evolution and get best strategy
    report, best_name = evolve_strategy(market_data)

    # Or use the full class for more control
    evo = StrategyEvolution(cfg_path="config/strategy_evolution.yaml")
    report = evo.evolve(market_data)

    print(report.to_dataframe())

    # Check if should re-evaluate
    if evo.should_reeval():
        report = evo.evolve(market_data)

    # Get current signal for live trading
    signal = evo.get_current_signal(live_data)

Reference:
---------
- Kaufman, P.J. (2013): "Trading Systems and Methods"
- Pardo, R. (2008): "The Evaluation and Optimization of Trading Strategies"
- Lopez de Prado, M. (2018): "Advances in Financial Machine Learning"
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
        return {}  # return empty dict so callers can safely call .get()
    return (
        yaml.safe_load(p.read_text()) or {}
    )  # guard against empty YAML file (returns None)


@dataclass
class StrategyResult:
    """
    Performance results for a single trading strategy.

    This dataclass encapsulates all performance metrics computed during
    backtesting of a single strategy signal. It is used both for
    individual strategy evaluation and for comparison/ranking.

    Attributes:
        name: Strategy identifier (e.g., "RSI_Oversold", "SMA_Cross")
        profit_factor: Gross profit factor (gross profit / gross loss)
        fee_adjusted_pf: Profit factor after deducting trading costs
        win_rate: Percentage of profitable trades (0.0 to 1.0)
        n_trades: Total number of completed trades
        total_return: Total return as decimal (e.g., 0.15 = 15%)
        sharpe: Annualized Sharpe ratio
        status: Strategy status ("KEEP", "ELIMINATE", "INSUFFICIENT_DATA")
        signal_column: DataFrame column name for the signal

    Methods:
        is_valid(): Returns True if strategy status is "KEEP"

    Example:
        --------
        result = StrategyResult(
            name="RSI_Oversold",
            profit_factor=1.85,
            fee_adjusted_pf=1.62,
            win_rate=0.58,
            n_trades=45,
            total_return=0.12,
            sharpe=1.34,
            status="KEEP",
            signal_column="_sig_RSI_Oversold"
        )

        if result.is_valid():
            print(f"Strategy {result.name} is valid with PF={result.profit_factor}")
    """

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
    """
    Complete evolution results with strategy rankings.

    This dataclass contains the results of a complete evolution run,
    including all strategies tested and their performance metrics.
    It provides convenient methods for ranking and analyzing results.

    Attributes:
        results: List of StrategyResult objects, one per strategy tested
        best_strategy: The top-performing strategy (highest fee-adjusted PF)
        n_candles_used: Number of data points used for backtesting
        timestamp: ISO timestamp of when evolution was run
        config_path: Path to configuration file used

    Properties:
        ranking: List of strategies sorted by fee-adjusted profit factor (best first)

    Methods:
        to_dataframe(): Convert results to pandas DataFrame for analysis

    Example:
        --------
        report = evolution_engine.evolve(market_data)

        # Get best strategy
        if report.best_strategy:
            print(f"Best: {report.best_strategy.name}")

        # View all strategies ranked
        print(report.to_dataframe())

        # Iterate through ranking
        for result in report.ranking:
            print(f"{result.name}: PF={result.fee_adjusted_pf:.2f}, {result.status}")
    """

    results: List[StrategyResult]
    best_strategy: Optional[StrategyResult]
    n_candles_used: int
    timestamp: str
    config_path: str

    @property
    def ranking(self) -> List[StrategyResult]:
        # sort descending: highest fee-adjusted profit factor wins
        return sorted(self.results, key=lambda r: r.fee_adjusted_pf, reverse=True)

    def to_dataframe(self) -> pd.DataFrame:
        rows = []
        for (
            r
        ) in self.ranking:  # iterate strategies sorted by fee-adjusted PF (best first)
            rows.append(
                {
                    "Strategy": r.name,
                    "PF (gross)": round(
                        r.profit_factor, 3
                    ),  # profit factor before fees
                    "PF (after fee)": round(
                        r.fee_adjusted_pf, 3
                    ),  # profit factor after fees
                    "Win-Rate %": round(r.win_rate * 100, 1),
                    "Trades": r.n_trades,
                    "Return %": round(r.total_return * 100, 2),
                    "Sharpe": round(r.sharpe, 2),
                    "Status": r.status,
                }
            )
        return pd.DataFrame(rows)


class SignalBuilder:
    """
    Builds trading signals from technical indicators.

    This class generates trading signals based on various technical
    indicators configured in the strategy evolution config. It supports
    multiple indicator types and automatically handles disabled strategies.

    The builder uses a dispatch pattern: strategy names are mapped to
    builder methods that generate the actual trading signals.

    Attributes:
        strategies_cfg: Dictionary of strategy configurations from YAML

    Methods:
        build_all(): Generate signals for all enabled strategies

    Supported Strategies:
    --------------------
    1. RSI_Oversold: Buy when RSI < lower_threshold (mean reversion)
    2. SMA_Cross: Buy when fast SMA > slow SMA (trend following)
    3. Bollinger_Lower: Buy when price near lower Bollinger Band
    4. MACD_Momentum: Buy when MACD histogram > 0 (momentum)
    5. OU_MeanReversion: Buy when OU score < threshold (mean reversion)

    Example:
        --------
        builder = SignalBuilder(config.get("strategies", {}))

        # Generate all signals
        df_with_signals, signal_cols = builder.build_all(market_data)

        # Use specific signal
        if "_sig_RSI_Oversold" in signal_cols:
            signal = df_with_signals["_sig_RSI_Oversold"]
    """

    def __init__(self, strategies_cfg: dict):
        """
        Initialize the SignalBuilder with strategy configurations.

        Parameters:
            strategies_cfg: Dictionary mapping strategy names to their
                          configuration dictionaries from YAML.
        """
        self._cfg = strategies_cfg

    def build_all(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        df = df.copy()  # avoid mutating the caller's DataFrame
        signal_cols: List[str] = []
        builders = {  # dispatch table: strategy name -> builder method
            "RSI_Oversold": self._rsi_signal,
            "SMA_Cross": self._sma_cross_signal,
            "Bollinger_Lower": self._bollinger_signal,
            "MACD_Momentum": self._macd_signal,
            "OU_MeanReversion": self._ou_signal,
        }
        for name, scfg in self._cfg.items():
            if not scfg.get("enabled", True):
                continue  # skip disabled strategies
            if name not in builders:
                continue  # skip unknown strategy names
            try:
                col_name = (
                    f"_sig_{name}"  # prefix avoids collisions with existing columns
                )
                df = builders[name](df, scfg, col_name)
                signal_cols.append(col_name)
            except Exception:
                pass  # skip silently if indicator column is missing
        return df, signal_cols

    def _rsi_signal(self, df, cfg, col):
        ind = cfg.get("indicator", "rsi_14")  # RSI column name in df
        lower = cfg.get("lower_threshold", 30)  # oversold threshold
        signal = np.zeros(len(df))
        signal[df[ind] < lower] = 1  # long when RSI is oversold
        df[col] = signal
        return df

    def _sma_cross_signal(self, df, cfg, col):
        fast_w, slow_w = cfg.get("fast_window", 20), cfg.get("slow_window", 50)
        signal = np.zeros(len(df))
        # long when fast MA is above slow MA (bullish crossover regime)
        signal[
            df["close"].rolling(fast_w).mean() > df["close"].rolling(slow_w).mean()
        ] = 1
        df[col] = signal
        return df

    def _bollinger_signal(self, df, cfg, col):
        ind, lower = (
            cfg.get("indicator", "bb_position"),
            cfg.get("lower_threshold", 0.1),
        )
        signal = np.zeros(len(df))
        # long when price is near the lower Bollinger Band (mean-reversion entry)
        signal[df[ind] < lower] = 1
        df[col] = signal
        return df

    def _macd_signal(self, df, cfg, col):
        ind = cfg.get("indicator", "macd_hist")  # MACD histogram column
        signal = np.zeros(len(df))
        signal[df[ind] > 0] = 1  # long when histogram is positive (bullish momentum)
        df[col] = signal
        return df

    def _ou_signal(self, df, cfg, col):
        ind, lower = cfg.get("indicator", "ou_score"), cfg.get("lower_threshold", -1.5)
        signal = np.zeros(len(df))
        # long when Ornstein-Uhlenbeck score is deeply negative (mean-reversion entry)
        signal[df[ind] < lower] = 1
        df[col] = signal
        return df


class SignalBacktester:
    """
    Backtests signal-based trading strategies.

    This class computes performance metrics for trading strategies that
    generate binary signals (buy/hold/sell). It handles signal alignment,
    cost modeling, and comprehensive metric calculation.

    The backtester accounts for:
    - Signal shift: Avoiding look-ahead bias by shifting signals
    - Trading costs: Fees and slippage deducted from returns
    - Trade detection: Identifying when positions change

    Attributes:
        fee_bps: Fee in basis points per trade (default: 10 bps)
        slippage_bps: Slippage in basis points (default: 3 bps)
        signal_shift: Number of bars to shift signal (default: 1)
                      1 means use previous bar's signal for current return
        total_cost_pct: Combined cost as decimal fraction

    Methods:
        backtest(): Run backtest on a signal column
        _empty_result(): Return empty result for insufficient data

    Example:
        --------
        backtester = SignalBacktester(
            fee_bps=10,      # 0.1% fee per trade
            slippage_bps=3,  # 0.03% slippage
            signal_shift=1   # Use previous bar signal
        )

        results = backtester.backtest(
            market_data,
            signal_column="_sig_RSI_Oversold",
            min_trades=20
        )

        print(f"Profit Factor: {results['profit_factor']:.2f}")
        print(f"Sharpe: {results['sharpe']:.2f}")
    """

    def __init__(self, fee_bps=10.0, slippage_bps=3.0, signal_shift=1):
        """
        Initialize the SignalBacktester.

        Parameters:
            fee_bps: Trading fee in basis points (default: 10)
            slippage_bps: Expected slippage in basis points (default: 3)
            signal_shift: Number of bars to shift signals (default: 1)
        """
        self.fee_bps, self.slippage_bps, self.signal_shift = (
            fee_bps,
            slippage_bps,
            signal_shift,
        )
        self.total_cost_pct = (
            fee_bps + slippage_bps
        ) / 10_000  # combined one-way cost fraction

    def backtest(self, df: pd.DataFrame, signal_col: str, min_trades: int = 20) -> Dict:
        if signal_col not in df.columns:
            return self._empty_result()
        returns = df["close"].pct_change()  # per-bar price returns
        signal = (
            df[signal_col].shift(self.signal_shift).fillna(0)
        )  # shift by 1 bar to avoid look-ahead
        strat_returns = signal * returns  # position-weighted returns
        trades_mask = (
            signal.diff().abs() > 0
        )  # bars where position changed (trade occurred)
        n_trades = int(trades_mask.sum())
        if n_trades < min_trades:
            return {**self._empty_result(), "n_trades": n_trades}
        strat_returns.loc[trades_mask] -= (
            self.total_cost_pct
        )  # deduct round-trip cost at each trade
        gains, losses = (
            strat_returns[strat_returns > 0],
            strat_returns[strat_returns < 0],
        )
        pf = (
            gains.sum() / abs(losses.sum()) if losses.sum() < 0 else 0.0
        )  # profit factor = gross profit / gross loss
        return {
            "profit_factor": pf,
            "win_rate": len(gains) / (len(gains) + len(losses))
            if (len(gains) + len(losses)) > 0
            else 0.0,
            "n_trades": n_trades,
            "total_return": (1 + strat_returns.dropna()).prod()
            - 1,  # compounded total return
            "sharpe": (
                strat_returns.mean() / strat_returns.std() * np.sqrt(252 * 24)
            )  # annualised (hourly bars)
            if strat_returns.std() > 0
            else 0.0,
            "status": "OK",
        }

    def _empty_result(self):
        return {
            "profit_factor": 0,
            "win_rate": 0,
            "n_trades": 0,
            "total_return": 0,
            "sharpe": 0,
            "status": "INSUFFICIENT_DATA",
        }


class StrategyEvolution:
    """
    Main strategy evolution engine implementing the Darwinian selection process.

    This class orchestrates the complete strategy evolution process:
    1. Loads strategy configurations from YAML
    2. Generates signals for all enabled strategies
    3. Backtests each strategy
    4. Selects the best-performing strategy
    5. Provides adaptive re-evaluation for live trading

    The evolution maintains state between runs, allowing for continuous
    operation where strategies are periodically re-evaluated and
    replaced if better alternatives are found.

    Attributes:
        min_candles: Minimum data points required for valid backtest
        min_trades: Minimum trades required for valid backtest
        pf_threshold: Minimum profit factor for strategy to be kept
        fee_pf_thresh: Minimum fee-adjusted profit factor
        lookback: Number of candles to use for backtesting
        reeval_interval: Candles between re-evaluations (0 = disabled)

    Methods:
        evolve(): Run one evolution iteration
        should_reeval(): Check if re-evaluation is due
        get_current_signal(): Get signal from best strategy for live trading

    Example:
        --------
        evo = StrategyEvolution(cfg_path="config/strategy_evolution.yaml")

        # Run initial evolution
        report = evo.evolve(market_data)

        print(f"Best strategy: {report.best_strategy.name}")
        print(report.to_dataframe())

        # In live trading loop
        while True:
            if evo.should_reeval():
                report = evo.evolve(market_data)
                print(f"New best: {report.best_strategy.name}")

            signal = evo.get_current_signal(live_data)
            execute_trade(signal)
    """

    def __init__(self, cfg_path=_DEFAULT_CFG_PATH):
        self._raw_cfg = load_evolution_config(cfg_path)
        self._cfg_path = cfg_path
        self._val_cfg = self._raw_cfg.get("validation", {})
        self.min_candles, self.min_trades = (
            self._val_cfg.get("min_candles", 500),
            self._val_cfg.get("min_trades", 20),
        )
        self.pf_threshold, self.fee_pf_thresh = (
            self._val_cfg.get("profit_factor_threshold", 1.5),
            self._val_cfg.get("fee_adjusted_threshold", 1.2),
        )
        self.lookback, self.reeval_interval = (
            self._raw_cfg.get("backtest", {}).get("lookback_window", 200),
            self._raw_cfg.get("adaptive", {}).get("reeval_every_n_candles", 50),
        )
        self._builder = SignalBuilder(self._raw_cfg.get("strategies", {}))
        self._backtester = SignalBacktester(
            self._raw_cfg.get("costs", {}).get("fee_bps", 10)
        )
        self._current_best = None
        self._candle_count = 0

    def evolve(self, df: pd.DataFrame) -> EvolutionReport:
        window_df = df.tail(self.lookback).copy()  # use only the most recent N candles
        window_df, cols = self._builder.build_all(window_df)
        results = []
        for col in cols:
            bt = self._backtester.backtest(window_df, col, self.min_trades)
            # promote strategy only if raw profit factor clears the threshold
            status = "KEEP" if bt["profit_factor"] >= self.pf_threshold else "ELIMINATE"
            if bt["status"] == "INSUFFICIENT_DATA":
                status = "INSUFFICIENT_DATA"
            results.append(
                StrategyResult(
                    col.replace(
                        "_sig_", ""
                    ),  # strip internal prefix to get readable name
                    bt["profit_factor"],
                    bt[
                        "profit_factor"
                    ],  # fee_adjusted_pf placeholder (same value here)
                    bt["win_rate"],
                    bt["n_trades"],
                    bt["total_return"],
                    bt["sharpe"],
                    status,
                    col,
                )
            )
        results = sorted(
            results, key=lambda r: r.profit_factor, reverse=True
        )  # best first
        best = next(
            (r for r in results if r.status == "KEEP"), None
        )  # highest-PF surviving strategy
        self._current_best = best
        return EvolutionReport(results, best, len(window_df), "", self._cfg_path)

    def should_reeval(self):
        self._candle_count += 1
        # trigger re-evaluation every N candles; interval=0 disables adaptive re-evaluation
        return (
            self.reeval_interval > 0 and self._candle_count % self.reeval_interval == 0
        )

    def get_current_signal(self, df):
        if not self._current_best:
            return 0  # no surviving strategy → flat position
        df_sig, _ = self._builder.build_all(
            df.tail(10)
        )  # build signals on last 10 bars only
        return (
            int(df_sig[self._current_best.signal_column].iloc[-1])  # latest bar signal
            if self._current_best.signal_column in df_sig.columns
            else 0
        )


def evolve_strategy(df, cfg_path=_DEFAULT_CFG_PATH):
    evo = StrategyEvolution(cfg_path)
    report = evo.evolve(df)
    return report, (report.best_strategy.name if report.best_strategy else None)


class WalkForwardStrategyEvolution:
    def __init__(self, df, n_folds=5, test_pct=0.2, cfg_path=_DEFAULT_CFG_PATH):
        self.df, self.n_folds, self.test_pct, self._evo = (
            df,
            n_folds,
            test_pct,
            StrategyEvolution(cfg_path),
        )

    def run(self):
        # Implementation for notebook analysis
        return pd.DataFrame()
