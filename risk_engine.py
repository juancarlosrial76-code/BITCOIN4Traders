"""
Risk Engine - Champion-to-Live Bridge
======================================
Links the winning algorithm (DarwinBot Champion) with a
complete risk management system following the professional 1%-rule.

Philosophy:
    "The algorithm decides WHAT is traded.
     The Risk Engine decides HOW MUCH and WHETHER at all."

Pipeline:
    StrategyTournament  ->  LiveTradingGuard  ->  RiskEngine  ->  TradingSession
         (Selection)            (Approval)          (Protection)     (Execution)

Components:
    RiskConfig        : All configurable thresholds (1%-rule, Kelly, Stop-Loss)
    PositionOrder     : Output of the Risk Engine: size, SL, TP, reason
    RiskEngine        : Core - calculates position size and checks all gates
    TradingSession    : Connects Champion + RiskEngine for bar-by-bar simulation
    SessionReport     : Complete performance and risk report

Mathematics:
    1%-rule     : max_risk_per_trade = account_equity * 0.01
    Position size = max_risk_per_trade / |entry_price - stop_loss_price|
    Kelly       : f* = (p*b - q) / b  (Fractional Kelly = 50% for safety)
    Stop-Loss   : ATR-based (Average True Range) or fixed percentage
    Take-Profit : Risk/Reward Ratio >= 2.0 (earns 2x the risk)

Protection Mechanisms:
    [1] 1%-rule          : Risk at most 1% of account per trade
    [2] Kelly-Ceiling    : Kelly formula additionally limits position size
    [3] Stop-Loss        : Automatically set (ATR or fixed %)
    [4] Take-Profit      : Automatically set (Risk * RR-Ratio)
    [5] Circuit-Breaker  : Trading halt at > 5% daily loss or 3 consecutive losses
    [6] Drawdown-Limit   : Trading halt when account > max_drawdown below peak
    [7] Min-Capital-Gate : No trading when account < 30% of starting capital

Usage (Colab):
    from risk_engine import RiskEngine, TradingSession, RiskConfig
    from darwin_engine import StrategyTournament, LiveTradingGuard, load_live_data

    df = load_live_data("BTC/USDT", "1h", limit=2000)
    t  = StrategyTournament(df); t.run()
    g  = LiveTradingGuard(); assert g.check(t.champion, df).APPROVED

    session = TradingSession(
        champion=t.champion,
        data=df,
        initial_capital=10_000,
        risk_config=RiskConfig(),  # 1%-rule activated
    )
    report = session.run()
    report.print_summary()

Author: BITCOIN4Traders Project
"""

from __future__ import annotations

import sys
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import gc

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
try:
    from loguru import logger
except ImportError:
    import logging

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    logger = logging.getLogger("risk_engine")

# ---------------------------------------------------------------------------
# Import project components
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(_PROJECT_ROOT))

# Darwin Engine
from darwin_engine import (
    DarwinBot,
    StrategyTournament,
    LiveTradingGuard,
    TournamentResult,
    ArenaConfig,
    generate_synthetic_btc,
    load_live_data,
    _kernel_profit_factor,
)

# Existing project RiskManager (graceful fallback if unavailable)
try:
    from src.math_tools.kelly_criterion import KellyCriterion, KellyParameters
    from src.risk.risk_manager import RiskManager, RiskConfig as _SrcRiskConfig

    _SRC_RISK_AVAILABLE = True
    logger.info("src/risk/risk_manager loaded - full Kelly integration active.")
except Exception as _e:
    _SRC_RISK_AVAILABLE = False
    logger.warning(f"src/risk not available ({_e}). Using built-in risk calculations.")


# ============================================================================
# CONFIGURATION
# ============================================================================


@dataclass
class RiskConfig:
    """
    All configurable parameters for risk management.
    Not a single magic number - everything explicit and documented.

    Defaults correspond to the professional 1%-rule for retail investors.
    """

    # --- Core: 1%-rule ---
    risk_per_trade_pct: float = 0.01  # Risk 1% of account per trade
    max_position_pct: float = 0.10  # Never more than 10% of account in one position

    # --- Kelly Criterion ---
    kelly_fraction: float = (
        0.50  # Fractional Kelly (50% = conservative, 100% = aggressive)
    )
    kelly_lookback: int = 20  # Last N trades for Kelly estimation

    # --- Stop-Loss ---
    sl_mode: str = "atr"  # "atr" | "pct"
    sl_atr_multiplier: float = 2.0  # Stop = entry ± (ATR * multiplier)
    sl_pct: float = 0.02  # Fixed stop: 2% from entry
    atr_period: int = 14  # ATR calculation window

    # --- Take-Profit ---
    rr_ratio: float = 2.0  # Risk/Reward: TP = entry + (risk * rr_ratio)
    tp_enabled: bool = True  # Enable take-profit

    # --- Circuit-Breaker ---
    max_daily_loss_pct: float = 0.05  # Trading halt at 5% daily loss
    max_consecutive_losses: int = 3  # Trading halt after 3 consecutive losses
    max_drawdown_pct: float = 0.15  # Trading halt at 15% drawdown from peak
    min_capital_pct: float = 0.30  # No trading below 30% of starting capital

    # --- Costs ---
    fee_rate: float = 0.001  # 0.1% taker fee (Binance standard)
    slippage_rate: float = 0.0005  # 0.05% slippage

    # --- Minimum trades for statistics ---
    min_trades_for_kelly: int = 10  # Apply Kelly only after 10 trades


# ============================================================================
# DATA CLASSES
# ============================================================================


@dataclass
class PositionOrder:
    """
    Output of the Risk Engine for each signal bar.
    Contains everything a broker needs for the order.
    """

    timestamp: pd.Timestamp
    signal: int  # 1=Long, -1=Short, 0=Flat
    approved: bool  # False = no trade (Risk-Gate blocked)
    block_reason: str  # Why blocked (empty if approved)

    entry_price: float = 0.0
    stop_loss_price: float = 0.0  # Absolute SL price
    take_profit_price: float = 0.0  # Absolute TP price
    position_size_usd: float = 0.0  # Position size in USD
    position_size_pct: float = 0.0  # Position size as % of account
    risk_amount_usd: float = 0.0  # Maximum risked amount (1%-rule)
    kelly_fraction: float = 0.0  # Kelly fraction used
    atr: float = 0.0  # ATR at time of signal

    def summary(self) -> str:
        if not self.approved:
            return f"[BLOCKED] {self.timestamp} | {self.block_reason}"
        side = "LONG" if self.signal == 1 else "SHORT" if self.signal == -1 else "FLAT"
        return (
            f"[{side}] {self.timestamp} | "
            f"Entry={self.entry_price:.2f} | "
            f"SL={self.stop_loss_price:.2f} | "
            f"TP={self.take_profit_price:.2f} | "
            f"Size={self.position_size_usd:.2f}$ ({self.position_size_pct:.2%}) | "
            f"Risk={self.risk_amount_usd:.2f}$ | "
            f"Kelly={self.kelly_fraction:.2%}"
        )


@dataclass
class RiskState:
    """Running state of the Risk Engine (updated after each trade)."""

    equity: float
    peak_equity: float
    day_start_equity: float
    consecutive_losses: int = 0
    consecutive_wins: int = 0
    trade_history: List[float] = field(default_factory=list)  # Trade P&L in %
    halted: bool = False
    halt_reason: str = ""

    @property
    def current_drawdown(self) -> float:
        return (self.peak_equity - self.equity) / self.peak_equity

    @property
    def daily_loss(self) -> float:
        return (self.day_start_equity - self.equity) / self.day_start_equity

    @property
    def total_return(self) -> float:
        return (self.equity - self.peak_equity) / self.peak_equity


@dataclass
class SessionReport:
    """Complete performance and risk report for a TradingSession."""

    initial_capital: float
    final_capital: float
    total_return: float
    max_drawdown: float
    sharpe_ratio: float
    profit_factor: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_win_pct: float
    avg_loss_pct: float
    circuit_breaker_hits: int
    blocked_signals: int
    champion_name: str
    champion_type: str
    risk_config: RiskConfig
    equity_curve: pd.Series = field(default_factory=pd.Series)
    orders: List[PositionOrder] = field(default_factory=list)

    def print_summary(self) -> None:
        print(f"\n{'=' * 64}")
        print(f"  TRADING SESSION REPORT")
        print(f"{'=' * 64}")
        print(f"  Champion     : {self.champion_name}")
        print(f"  Strategy     : {self.champion_type}")
        print(f"{'─' * 64}")
        print(
            f"  Capital      : {self.initial_capital:>10,.2f}$  ->  {self.final_capital:>10,.2f}$"
        )
        print(f"  Total Return : {self.total_return:>+10.2%}")
        print(f"  Max Drawdown : {self.max_drawdown:>10.2%}")
        print(f"  Sharpe Ratio : {self.sharpe_ratio:>10.2f}")
        print(f"  Profit Factor: {self.profit_factor:>10.3f}")
        print(f"{'─' * 64}")
        print(
            f"  Trades       : {self.total_trades:>10d}  "
            f"(W:{self.winning_trades} / L:{self.losing_trades})"
        )
        print(f"  Win Rate     : {self.win_rate:>10.1%}")
        print(f"  Avg Win      : {self.avg_win_pct:>+10.2%}")
        print(f"  Avg Loss     : {self.avg_loss_pct:>+10.2%}")
        print(f"{'─' * 64}")
        print(f"  Risk/Trade   : {self.risk_config.risk_per_trade_pct:.1%}  (1%-rule)")
        print(f"  Kelly Frac.  : {self.risk_config.kelly_fraction:.0%}  (Half-Kelly)")
        print(
            f"  SL Mode      : {self.risk_config.sl_mode.upper()}  "
            f"({'x' + str(self.risk_config.sl_atr_multiplier) + ' ATR' if self.risk_config.sl_mode == 'atr' else str(self.risk_config.sl_pct) + '%'})"
        )
        print(f"  RR Ratio     : 1:{self.risk_config.rr_ratio:.1f}")
        print(f"  Circuit Hits : {self.circuit_breaker_hits:>10d}")
        print(
            f"  Blocked      : {self.blocked_signals:>10d}  signals blocked by Risk-Engine"
        )
        print(f"{'=' * 64}\n")


# ============================================================================
# CORE: RISK ENGINE
# ============================================================================


class RiskEngine:
    """
    The account protector.

    Takes the raw signal from the champion algorithm and decides:
        1. May this trade happen? (Circuit-Breaker, Drawdown-Gate)
        2. How large is the position? (1%-rule + Kelly-Ceiling)
        3. Where is the Stop-Loss? (ATR or fixed %)
        4. Where is the Take-Profit? (Risk * RR-Ratio)

    Returns a PositionOrder object - never raw numbers.
    This enforces that no trade happens without a risk check.
    """

    def __init__(self, config: RiskConfig, initial_capital: float):
        self.config = config
        self.state = RiskState(
            equity=initial_capital,
            peak_equity=initial_capital,
            day_start_equity=initial_capital,
        )
        self._initial_capital = initial_capital

        # Integrate existing src/risk/RiskManager if available
        if _SRC_RISK_AVAILABLE:
            src_cfg = _SrcRiskConfig(
                max_drawdown_per_session=config.max_drawdown_pct,
                max_consecutive_losses=config.max_consecutive_losses,
                max_position_size=config.max_position_pct,
                kelly_fraction=config.kelly_fraction,
                min_capital_threshold=config.min_capital_pct,
            )
            self._src_risk = RiskManager(src_cfg, initial_capital)
            self._kelly = KellyCriterion()
        else:
            self._src_risk = None
            self._kelly = None

        logger.info(
            f"RiskEngine initialized | Capital={initial_capital:,.2f}$ | "
            f"1%-rule active | Kelly={config.kelly_fraction:.0%} | "
            f"SL={config.sl_mode.upper()}"
        )

    # ------------------------------------------------------------------
    # ATR calculation
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_atr(
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        period: int,
        idx: int,
    ) -> float:
        """
        Average True Range at position idx.
        True Range = max(High-Low, |High-PrevClose|, |Low-PrevClose|)
        """
        start = max(1, idx - period + 1)
        if start >= idx:
            return float(high[idx] - low[idx])
        tr_list = []
        for i in range(start, idx + 1):
            tr = max(
                high[i] - low[i],
                abs(high[i] - close[i - 1]),
                abs(low[i] - close[i - 1]),
            )
            tr_list.append(tr)
        return float(np.mean(tr_list))

    # ------------------------------------------------------------------
    # Estimate Kelly fraction from trade history
    # ------------------------------------------------------------------

    def _compute_kelly_fraction(self) -> float:
        """
        Estimate Kelly fraction from the last N trades.
        Before min_trades_for_kelly, uses fixed value from config.
        """
        history = self.state.trade_history
        n = len(history)
        if n < self.config.min_trades_for_kelly:
            return self.config.kelly_fraction  # Conservative until enough data

        recent = np.array(history[-self.config.kelly_lookback :])
        wins = recent[recent > 0]
        losses = recent[recent < 0]

        if len(wins) == 0 or len(losses) == 0:
            return self.config.kelly_fraction

        p = len(wins) / len(recent)
        b = np.mean(wins) / max(abs(np.mean(losses)), 1e-9)
        q = 1.0 - p
        f_star = (p * b - q) / b  # Full Kelly

        # Fractional Kelly + Hard-Cap
        frac = np.clip(
            f_star * self.config.kelly_fraction, 0.0, self.config.max_position_pct
        )
        return float(frac)

    # ------------------------------------------------------------------
    # Circuit-Breaker check
    # ------------------------------------------------------------------

    def _check_circuit_breaker(self) -> Tuple[bool, str]:
        """
        Check all circuit-breaker conditions.

        Returns
        -------
        (should_halt: bool, reason: str)
        """
        cfg = self.config
        st = self.state

        # Gate 1: Daily loss
        if st.daily_loss >= cfg.max_daily_loss_pct:
            return (
                True,
                f"Daily loss {st.daily_loss:.2%} >= limit {cfg.max_daily_loss_pct:.2%}",
            )

        # Gate 2: Consecutive losses
        if st.consecutive_losses >= cfg.max_consecutive_losses:
            return (
                True,
                f"{st.consecutive_losses} consecutive losses >= limit {cfg.max_consecutive_losses}",
            )

        # Gate 3: Drawdown from peak
        if st.current_drawdown >= cfg.max_drawdown_pct:
            return (
                True,
                f"Drawdown {st.current_drawdown:.2%} >= limit {cfg.max_drawdown_pct:.2%}",
            )

        # Gate 4: Minimum capital
        if st.equity < self._initial_capital * cfg.min_capital_pct:
            return True, (
                f"Capital {st.equity:.2f}$ < "
                f"minimum capital {self._initial_capital * cfg.min_capital_pct:.2f}$"
            )

        return False, ""

    # ------------------------------------------------------------------
    # Main method: compute_order()
    # ------------------------------------------------------------------

    def compute_order(
        self,
        timestamp: pd.Timestamp,
        signal: int,
        entry_price: float,
        high_arr: np.ndarray,
        low_arr: np.ndarray,
        close_arr: np.ndarray,
        bar_idx: int,
    ) -> PositionOrder:
        """
        Compute a complete position order for a signal.

        Parameters
        ----------
        timestamp    : Timestamp of the signal
        signal       : 1 (Long), -1 (Short), 0 (Flat/no trade)
        entry_price  : Entry price (current close)
        high_arr     : Full high array (for ATR)
        low_arr      : Full low array (for ATR)
        close_arr    : Full close array (for ATR)
        bar_idx      : Current bar index

        Returns
        -------
        PositionOrder
            Contains position size, SL, TP and whether the trade is approved.
        """
        # Flat signal: no trade, no check needed
        if signal == 0:
            return PositionOrder(
                timestamp=timestamp,
                signal=0,
                approved=False,
                block_reason="No signal (Flat)",
                entry_price=entry_price,
            )

        # --- Circuit-Breaker ---
        halted, reason = self._check_circuit_breaker()
        if halted:
            self.state.halted = True
            self.state.halt_reason = reason
            logger.warning(f"CIRCUIT BREAKER: {reason}")
            return PositionOrder(
                timestamp=timestamp,
                signal=signal,
                approved=False,
                block_reason=f"Circuit-Breaker: {reason}",
                entry_price=entry_price,
            )

        equity = self.state.equity

        # --- Compute ATR ---
        atr = self._compute_atr(
            high_arr, low_arr, close_arr, self.config.atr_period, bar_idx
        )

        # --- Compute Stop-Loss ---
        if self.config.sl_mode == "atr":
            sl_distance = atr * self.config.sl_atr_multiplier
        else:
            sl_distance = entry_price * self.config.sl_pct

        sl_distance = max(sl_distance, entry_price * 0.001)  # Minimum SL: 0.1%

        if signal == 1:  # Long
            sl_price = entry_price - sl_distance
            tp_price = entry_price + sl_distance * self.config.rr_ratio
        else:  # Short
            sl_price = entry_price + sl_distance
            tp_price = entry_price - sl_distance * self.config.rr_ratio

        # --- 1%-rule: max. risked amount ---
        max_risk_usd = (
            equity * self.config.risk_per_trade_pct
        )  # e.g. 10,000 * 0.01 = 100$

        # --- Position size from 1%-rule ---
        # Position size = risk budget / (|Entry - SL| / Entry)
        risk_per_unit = sl_distance / entry_price  # relative loss per unit
        position_by_1pct = max_risk_usd / max(risk_per_unit, 1e-9)
        position_by_1pct = min(position_by_1pct, equity)  # never more than the account

        # --- Kelly-Ceiling: Kelly further limits the 1%-position ---
        kelly_f = self._compute_kelly_fraction()
        kelly_usd = equity * kelly_f

        # Final position size: minimum of 1%-rule and Kelly
        position_usd = min(position_by_1pct, kelly_usd)
        position_usd = min(
            position_usd, equity * self.config.max_position_pct
        )  # Hard-Cap

        # --- Integrate src/risk/RiskManager if available ---
        if (
            self._src_risk is not None
            and len(self.state.trade_history) >= self.config.min_trades_for_kelly
        ):
            recent = np.array(self.state.trade_history[-self.config.kelly_lookback :])
            wins = recent[recent > 0]
            losses = recent[recent < 0]
            if len(wins) > 0 and len(losses) > 0:
                win_prob = len(wins) / len(recent)
                wl_ratio = float(np.mean(wins)) / max(float(abs(np.mean(losses))), 1e-9)
                approved_src, kelly_src = self._src_risk.validate_position_size(
                    proposed_size=position_usd,
                    current_capital=equity,
                    win_probability=win_prob,
                    win_loss_ratio=wl_ratio,
                )
                if not approved_src:
                    return PositionOrder(
                        timestamp=timestamp,
                        signal=signal,
                        approved=False,
                        block_reason="src/RiskManager: Position rejected",
                        entry_price=entry_price,
                        stop_loss_price=sl_price,
                        take_profit_price=tp_price,
                        atr=atr,
                    )
                position_usd = min(position_usd, kelly_src)

        return PositionOrder(
            timestamp=timestamp,
            signal=signal,
            approved=True,
            block_reason="",
            entry_price=entry_price,
            stop_loss_price=round(sl_price, 4),
            take_profit_price=round(tp_price, 4) if self.config.tp_enabled else 0.0,
            position_size_usd=round(position_usd, 2),
            position_size_pct=position_usd / max(equity, 1.0),
            risk_amount_usd=round(min(max_risk_usd, position_usd * risk_per_unit), 2),
            kelly_fraction=kelly_f,
            atr=round(atr, 4),
        )

    # ------------------------------------------------------------------
    # State update after trade completion
    # ------------------------------------------------------------------

    def update_after_trade(self, pnl_pct: float, new_equity: float) -> None:
        """
        Update the risk state after a completed trade.

        Parameters
        ----------
        pnl_pct    : P&L of the trade as a fraction of capital (e.g. -0.008 = -0.8%)
        new_equity : New account balance after the trade
        """
        self.state.equity = new_equity
        self.state.peak_equity = max(self.state.peak_equity, new_equity)
        self.state.trade_history.append(pnl_pct)

        if pnl_pct > 0:
            self.state.consecutive_losses = 0
            self.state.consecutive_wins += 1
        else:
            self.state.consecutive_losses += 1
            self.state.consecutive_wins = 0

        # Sync src/RiskManager
        if self._src_risk is not None:
            self._src_risk.update_state(new_equity, pnl_pct * new_equity)

    def reset_daily(self, current_equity: float) -> None:
        """Daily reset: reset daily loss counter."""
        self.state.day_start_equity = current_equity
        self.state.halted = False
        self.state.halt_reason = ""
        logger.info(f"Daily reset: Equity={current_equity:.2f}$")


# ============================================================================
# TRADING SESSION
# ============================================================================


class TradingSession:
    """
    Complete bar-by-bar simulation of the champion algorithm
    with active risk management.

    For each bar:
        1. Champion algorithm computes signal
        2. RiskEngine checks signal and computes position size
        3. If approved: trade is simulated
        4. Equity is updated, Stop-Loss/Take-Profit are checked

    This is the bridge between backtest and live trading:
    The identical code runs in backtest AND in paper-/live-trading.
    """

    def __init__(
        self,
        champion: DarwinBot,
        data: pd.DataFrame,
        initial_capital: float = 10_000.0,
        risk_config: Optional[RiskConfig] = None,
        verbose: bool = True,
    ):
        if "close" not in data.columns:
            raise ValueError("DataFrame must contain 'close'.")
        required = {"open", "high", "low", "close"}
        missing = required - set(data.columns)
        if missing:
            # Synthetic OHLC if only close is available
            logger.warning(f"Missing columns {missing} - using Close as OHLC proxy.")
            for col in missing:
                data = data.copy()
                data[col] = data["close"]

        self.champion = champion
        self.data = data.copy()
        self.initial_capital = initial_capital
        self.risk_config = risk_config or RiskConfig()
        self.verbose = verbose
        self.engine = RiskEngine(self.risk_config, initial_capital)

    def run(self) -> SessionReport:
        """
        Run the complete session.

        Returns
        -------
        SessionReport  with all performance and risk metrics.
        """
        df = self.data
        closes = df["close"].values.astype(np.float64)
        highs = df["high"].values.astype(np.float64)
        lows = df["low"].values.astype(np.float64)
        n = len(df)
        idx = df.index

        # Pre-compute all signals (JIT kernel)
        signals = self.champion.compute_signals(closes)

        equity = self.initial_capital
        equity_curve = [equity]
        orders: List[PositionOrder] = []

        in_position = 0
        entry_price = 0.0
        entry_equity = equity
        sl_price = 0.0
        tp_price = 0.0

        wins: List[float] = []
        losses: List[float] = []
        cb_hits = 0
        blocked = 0

        last_day: Optional[pd.Timestamp] = None

        for i in range(1, n):
            bar_ts = idx[i]
            price = closes[i]

            # Daily reset (for daily loss limit)
            bar_day = getattr(bar_ts, "date", lambda: bar_ts)()
            if last_day is not None and bar_day != last_day:
                self.engine.reset_daily(equity)
            last_day = bar_day

            # --- SL/TP-Check for open position ---
            if in_position != 0:
                sl_hit = (in_position == 1 and price <= sl_price) or (
                    in_position == -1 and price >= sl_price
                )
                tp_hit = self.risk_config.tp_enabled and (
                    (in_position == 1 and price >= tp_price)
                    or (in_position == -1 and price <= tp_price)
                )

                if sl_hit or tp_hit:
                    exit_price = sl_price if sl_hit else tp_price
                    pnl_pct = in_position * (exit_price - entry_price) / entry_price
                    pnl_pct -= self.risk_config.fee_rate  # Exit-Fee

                    trade_pnl = entry_equity * pnl_pct
                    equity += trade_pnl

                    if pnl_pct > 0:
                        wins.append(pnl_pct)
                    else:
                        losses.append(pnl_pct)

                    self.engine.update_after_trade(pnl_pct, equity)

                    if self.verbose:
                        tag = "SL" if sl_hit else "TP"
                        logger.debug(
                            f"{tag} | {bar_ts} | exit={exit_price:.2f} | "
                            f"pnl={pnl_pct:+.2%} | equity={equity:.2f}$"
                        )
                    in_position = 0
                    entry_price = 0.0

            # --- New signal from champion ---
            sig = int(signals[i])

            if sig != in_position and sig != 0:
                # Signal change -> query Risk Engine
                order = self.engine.compute_order(
                    timestamp=bar_ts,
                    signal=sig,
                    entry_price=price,
                    high_arr=highs,
                    low_arr=lows,
                    close_arr=closes,
                    bar_idx=i,
                )
                orders.append(order)

                if order.approved:
                    # Open trade
                    entry_equity = equity
                    in_position = sig
                    entry_price = price
                    sl_price = order.stop_loss_price
                    tp_price = order.take_profit_price
                    # Entry-Fee
                    equity *= 1.0 - self.risk_config.fee_rate

                    if self.verbose:
                        logger.debug(order.summary())
                else:
                    blocked += 1
                    if "Circuit" in order.block_reason:
                        cb_hits += 1
                    if self.verbose:
                        logger.debug(order.summary())

            elif sig == 0 and in_position != 0:
                # Flat signal: close open position
                pnl_pct = in_position * (price - entry_price) / entry_price
                pnl_pct -= self.risk_config.fee_rate

                trade_pnl = entry_equity * pnl_pct
                equity += trade_pnl

                if pnl_pct > 0:
                    wins.append(pnl_pct)
                else:
                    losses.append(pnl_pct)

                self.engine.update_after_trade(pnl_pct, equity)
                in_position = 0

            equity_curve.append(max(equity, 0.0))

        # --- Metrics ---
        eq_series = pd.Series(equity_curve, index=df.index[: len(equity_curve)])
        returns = eq_series.pct_change().dropna()
        total_ret = (eq_series.iloc[-1] / eq_series.iloc[0]) - 1
        running_max = eq_series.cummax()
        max_dd = float(((eq_series - running_max) / running_max).min())
        sharpe = (
            float(returns.mean()) / float(returns.std()) * np.sqrt(8760)
            if returns.std() > 0
            else 0.0
        )  # Annualized to hours

        n_wins = len(wins)
        n_losses = len(losses)
        n_trades = n_wins + n_losses
        pf = (sum(wins) / max(abs(sum(losses)), 1e-9)) if losses else 0.0

        report = SessionReport(
            initial_capital=self.initial_capital,
            final_capital=round(equity, 2),
            total_return=total_ret,
            max_drawdown=abs(max_dd),
            sharpe_ratio=sharpe,
            profit_factor=round(pf, 3),
            total_trades=n_trades,
            winning_trades=n_wins,
            losing_trades=n_losses,
            win_rate=n_wins / max(n_trades, 1),
            avg_win_pct=float(np.mean(wins)) if wins else 0.0,
            avg_loss_pct=float(np.mean(losses)) if losses else 0.0,
            circuit_breaker_hits=cb_hits,
            blocked_signals=blocked,
            champion_name=self.champion.name,
            champion_type=type(self.champion).__name__,
            risk_config=self.risk_config,
            equity_curve=eq_series,
            orders=orders,
        )

        gc.collect()
        return report


# ============================================================================
# FULL PIPELINE: Tournament -> Guard -> RiskEngine -> Session
# ============================================================================


def run_full_pipeline(
    symbol: str = "BTC/USDT",
    timeframe: str = "1h",
    limit: int = 2000,
    initial_capital: float = 10_000.0,
    risk_per_trade: float = 0.01,  # 1%-Regel
    kelly_fraction: float = 0.50,  # Half-Kelly
    rr_ratio: float = 2.0,  # Risk/Reward 1:2
    exchange_id: str = "binance",
    cache_dir: str = "data/cache",
    generations: int = 5,
    pop_size: int = 12,
    use_synthetic: bool = False,
    verbose: bool = True,
) -> Tuple[Optional[DarwinBot], Optional[SessionReport]]:
    """
    Complete pipeline in a single function call.

    For Colab:
        from risk_engine import run_full_pipeline
        champion, report = run_full_pipeline(
            symbol="BTC/USDT", initial_capital=10_000, risk_per_trade=0.01
        )
        report.print_summary()

    Steps:
        1. Load data (Binance or synthetic)
        2. Strategy Tournament (RSI vs MACD vs BB vs EMA)
        3. Live-Trading Guard (6-gate check)
        4. Build Risk Engine (1%-rule + Kelly)
        5. Simulate trading session
        6. Print report

    Returns
    -------
    (champion, report) or (None, None) if Guard blocked
    """
    print(f"\n{'=' * 64}")
    print(f"  DARWIN RISK PIPELINE")
    print(
        f"  Capital: {initial_capital:,.2f}$  |  Rule: {risk_per_trade:.0%} per trade"
    )
    print(f"{'=' * 64}\n")

    # --- 1. Data ---
    print("[1/5] Loading market data ...")
    if use_synthetic:
        df = generate_synthetic_btc(n_bars=limit, seed=42)
        print(f"      Synthetic: {len(df)} bars")
    else:
        cache = (
            Path(cache_dir)
            / f"{exchange_id}_{symbol.replace('/', '_')}_{timeframe}_{limit}.parquet"
        )
        try:
            df = load_live_data(symbol, timeframe, limit, exchange_id, cache_path=cache)
            print(
                f"      Live: {len(df)} bars | {df.index[0].date()} -> {df.index[-1].date()}"
            )
        except Exception as e:
            print(f"      Live fetch failed ({e}) -> Synthetic")
            df = generate_synthetic_btc(n_bars=limit, seed=42)

    # --- 2. Tournament ---
    print("\n[2/5] Strategy Tournament (RSI vs MACD vs Bollinger vs EMA) ...")
    tournament = StrategyTournament(
        df,
        min_bars=500,
        pf_threshold=1.2,
        max_dd_threshold=0.25,
        fee_rate=0.001,
    )
    ranking = tournament.run(
        arena_config=ArenaConfig(generations=generations, pop_size=pop_size, seed=42),
        verbose=verbose,
    )

    if tournament.champion is None:
        print("   No winner qualified - pipeline aborted.")
        return None, None

    champion = tournament.champion
    print(f"      Winner: {champion.name}  ({type(champion).__name__})")

    # --- 3. Guard ---
    print("\n[3/5] Live-Trading Guard (6-Gate-Check) ...")
    guard = LiveTradingGuard(
        min_bars=500,
        min_pf=1.2,
        max_dd=0.25,
        min_sharpe=0.3,
        max_wfv_degradation=0.90,
        wfv_splits=3,
    )
    guard_report = guard.check(champion, df, verbose=verbose)

    if not guard_report.APPROVED:
        print("   Guard: NOT APPROVED - pipeline aborted.")
        return champion, None

    print("   Guard: APPROVED")

    # --- 4. Risk Config ---
    print("\n[4/5] Configuring Risk Engine ...")
    risk_cfg = RiskConfig(
        risk_per_trade_pct=risk_per_trade,
        kelly_fraction=kelly_fraction,
        rr_ratio=rr_ratio,
        sl_mode="atr",
        sl_atr_multiplier=2.0,
        fee_rate=0.001,
        slippage_rate=0.0005,
        max_daily_loss_pct=0.05,
        max_consecutive_losses=3,
        max_drawdown_pct=0.15,
    )
    print(
        f"      1%-rule: max {risk_per_trade:.0%} | Kelly: {kelly_fraction:.0%} | RR: 1:{rr_ratio}"
    )

    # --- 5. Session ---
    print("\n[5/5] Trading Session ...")
    session = TradingSession(
        champion=champion,
        data=df,
        initial_capital=initial_capital,
        risk_config=risk_cfg,
        verbose=False,
    )
    report = session.run()

    # --- 6. Report ---
    report.print_summary()

    return champion, report


# ============================================================================
# STANDALONE ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    print("Risk Engine - Self Test")
    print("=" * 64)

    # Quick test with synthetic data
    champion, report = run_full_pipeline(
        symbol="BTC/USDT",
        limit=1500,
        initial_capital=10_000.0,
        risk_per_trade=0.01,  # 1%-rule
        kelly_fraction=0.50,  # Half-Kelly
        rr_ratio=2.0,  # 1:2 Risk/Reward
        use_synthetic=True,  # Synthetic for test
        generations=3,
        pop_size=8,
        verbose=True,
    )

    if report is not None:
        print(f"Test successful. Champion: {champion.name}")
        print(
            f"Final equity: {report.final_capital:,.2f}$ "
            f"(Start: {report.initial_capital:,.2f}$)"
        )
    else:
        print("No champion qualified - guards blocked.")
