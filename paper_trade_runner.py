#!/usr/bin/env python3
"""
Paper Trading Runner & Observer
===============================
Comprehensive paper trading execution with real-time observation and metrics.

Usage:
    python paper_trade_runner.py --symbol BTC/USDT --capital 10000 --timeframe 1h

Features:
    - Multi-exchange paper trading via ccxt
    - Real-time P&L tracking
    - Trade logging to parquet
    - Metrics observation and alerting
    - Signal validation
"""

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from loguru import logger

from src.execution.multi_exchange_paper_trader import (
    MultiExchangePaperTrader,
    PaperTraderConfig,
)
from src.monitoring.production_monitor import (
    ProductionMonitor,
    AlertLevel,
    AlertType,
    Alert,
    TradingMetrics,
)


@dataclass
class PaperTradingState:
    """Current state of paper trading session."""

    session_id: str
    start_time: datetime
    is_running: bool = False

    initial_capital: float = 0.0
    current_capital: float = 0.0
    total_pnl: float = 0.0
    daily_pnl: float = 0.0

    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0

    current_position: float = 0.0
    exposure_pct: float = 0.0

    max_drawdown_pct: float = 0.0
    peak_capital: float = 0.0

    last_signal: int = 0
    last_signal_time: Optional[datetime] = None

    metrics_history: List[Dict] = field(default_factory=list)
    trades: List[Dict] = field(default_factory=list)


class RandomSignalBot:
    """Fallback bot that generates random signals when no champion is available."""

    name = "RandomSignalBot"

    def __init__(self):
        import random

        self._random = random

    def compute_signals(self, close_prices):
        """Generate random signals."""
        n = len(close_prices)
        signals = np.zeros(n)
        for i in range(1, n):
            signals[i] = self._random.choice([-1, 0, 1])
        return signals


class PaperTradingObserver:
    """
    Observer for paper trading sessions.
    Tracks metrics, detects anomalies, and provides real-time insights.
    """

    def __init__(self, state: PaperTradingState):
        self.state = state
        self._start_of_day = datetime.now()

    def on_trade(self, trade: Dict):
        """Called when a trade is executed."""
        self.state.total_trades += 1
        self.state.trades.append(
            {
                **trade,
                "session_id": self.state.session_id,
                "timestamp": datetime.now().isoformat(),
            }
        )

        if trade.get("pnl", 0) > 0:
            self.state.winning_trades += 1
        else:
            self.state.losing_trades += 1

        if self.state.total_trades > 0:
            self.state.win_rate = self.state.winning_trades / self.state.total_trades

        logger.info(
            f"Trade executed: {trade.get('side')} {trade.get('quantity')} @ {trade.get('price')} | PnL: {trade.get('pnl', 0):.2f}"
        )

    def on_signal(self, signal: int, timestamp: datetime):
        """Called when a new signal is received."""
        self.state.last_signal = signal
        self.state.last_signal_time = timestamp
        signal_label = {1: "LONG", -1: "SHORT", 0: "FLAT"}.get(signal, "UNKNOWN")
        logger.info(f"Signal received: {signal_label} at {timestamp}")

    def update_metrics(self, portfolio_value: float, position: float):
        """Update metrics based on current state."""
        self.state.current_capital = portfolio_value

        if portfolio_value > self.state.peak_capital:
            self.state.peak_capital = portfolio_value

        self.state.total_pnl = portfolio_value - self.state.initial_capital

        drawdown = (
            (self.state.peak_capital - portfolio_value) / self.state.peak_capital
            if self.state.peak_capital > 0
            else 0
        )
        self.state.max_drawdown_pct = max(self.state.max_drawdown_pct, drawdown)

        now = datetime.now()
        if now.date() != self._start_of_day.date():
            self.state.daily_pnl = 0.0
            self._start_of_day = now

        self.state.current_position = position
        self.state.exposure_pct = (
            abs(position * portfolio_value / self.state.initial_capital)
            if self.state.initial_capital > 0
            else 0
        )

    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics as dictionary."""
        return {
            "session_id": self.state.session_id,
            "timestamp": datetime.now().isoformat(),
            "is_running": self.state.is_running,
            "initial_capital": self.state.initial_capital,
            "current_capital": self.state.current_capital,
            "total_pnl": self.state.total_pnl,
            "daily_pnl": self.state.daily_pnl,
            "total_trades": self.state.total_trades,
            "winning_trades": self.state.winning_trades,
            "losing_trades": self.state.losing_trades,
            "win_rate": self.state.win_rate,
            "current_position": self.state.current_position,
            "exposure_pct": self.state.exposure_pct,
            "max_drawdown_pct": self.state.max_drawdown_pct,
            "last_signal": self.state.last_signal,
            "last_signal_time": self.state.last_signal_time.isoformat()
            if self.state.last_signal_time
            else None,
        }

    def get_trading_metrics(self) -> TradingMetrics:
        """Get TradingMetrics dataclass for ProductionMonitor."""
        uptime = (datetime.now() - self.state.start_time).total_seconds()

        return TradingMetrics(
            timestamp=datetime.now(),
            total_pnl=self.state.total_pnl,
            daily_pnl=self.state.daily_pnl,
            open_positions=1 if abs(self.state.current_position) > 0.001 else 0,
            exposure_pct=self.state.exposure_pct,
            margin_used_pct=self.state.exposure_pct,
            sharpe_24h=0.0,
            max_drawdown_pct=self.state.max_drawdown_pct,
            win_rate=self.state.win_rate,
            avg_trade_size=self.state.current_capital / max(self.state.total_trades, 1),
            latency_ms=0.0,
            uptime_seconds=uptime,
        )

    def check_alerts(self, monitor: ProductionMonitor) -> List[str]:
        """Check for alert conditions."""
        alerts = []

        if self.state.max_drawdown_pct > 0.15:
            alerts.append(
                f"WARNING: Max drawdown {self.state.max_drawdown_pct * 100:.1f}% exceeds 15%"
            )
            monitor.record_alert(
                Alert(
                    timestamp=datetime.now(),
                    level=AlertLevel.WARNING,
                    type=AlertType.DRAWDOWN,
                    message=f"Max drawdown {self.state.max_drawdown_pct * 100:.1f}% exceeds 15%",
                    value=self.state.max_drawdown_pct,
                    threshold=0.15,
                )
            )

        if self.state.exposure_pct > 0.9:
            alerts.append(f"WARNING: Exposure {self.state.exposure_pct * 100:.1f}% exceeds 90%")

        if self.state.daily_pnl < -self.state.initial_capital * 0.05:
            alerts.append(f"CRITICAL: Daily loss {self.state.daily_pnl:.2f} exceeds 5%")
            monitor.record_alert(
                Alert(
                    timestamp=datetime.now(),
                    level=AlertLevel.CRITICAL,
                    type=AlertType.PNL,
                    message=f"Daily loss {self.state.daily_pnl:.2f} exceeds 5%",
                    value=self.state.daily_pnl,
                    threshold=-self.state.initial_capital * 0.05,
                )
            )

        return alerts

    def save_session(self, output_dir: Path):
        """Save session data to files."""
        output_dir.mkdir(parents=True, exist_ok=True)

        metrics_file = output_dir / f"session_{self.state.session_id}_metrics.json"
        with open(metrics_file, "w") as f:
            json.dump(self.get_metrics(), f, indent=2)

        if self.state.trades:
            trades_file = output_dir / f"session_{self.state.session_id}_trades.parquet"
            pd.DataFrame(self.state.trades).to_parquet(trades_file, index=False)
            logger.info(f"Session data saved to {output_dir}")


def run_paper_trading(
    symbol: str = "BTC/USDT",
    timeframe: str = "1h",
    initial_capital: float = 10000.0,
    exchanges: List[str] = None,
    max_ticks: int = 0,
    output_dir: str = "data/paper_trades",
    champion_path: str = "data/cache/multiverse_champion.pkl",
):
    """Run paper trading session with observation."""

    if exchanges is None:
        exchanges = ["binance", "kucoin", "bybit"]

    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    state = PaperTradingState(
        session_id=session_id,
        start_time=datetime.now(),
        initial_capital=initial_capital,
        current_capital=initial_capital,
        is_running=True,
    )

    observer = PaperTradingObserver(state)

    monitor = ProductionMonitor(
        check_interval_seconds=30.0,
        metrics_history_size=10000,
    )
    monitor.set_threshold("max_drawdown_pct", 0.15)
    monitor.set_threshold("daily_loss_limit_pct", 0.05)
    monitor.start_monitoring()

    config = PaperTraderConfig(
        symbol=symbol,
        timeframe=timeframe,
        initial_capital=initial_capital,
        exchanges=exchanges,
        primary_exchange="binance",
        poll_interval_s=5.0,
        ohlcv_lookback=200,
        data_dir=output_dir,
        risk_per_trade=0.01,
        max_drawdown=0.20,
    )

    logger.info(f"Starting paper trading session {session_id}")
    logger.info(f"Config: {symbol} {timeframe} ${initial_capital}")

    try:
        from src.math_tools.archive.darwin_legacy import ChampionPersistence
        import os

        champion = None
        champion_meta = Path(champion_path).with_suffix(".json")

        if Path(champion_path).exists():
            try:
                champion = ChampionPersistence.load(
                    str(champion_path),
                    str(champion_meta) if champion_meta.exists() else None,
                )
                logger.info(f"Loaded champion: {champion.name}")
            except Exception as e:
                logger.warning(f"Could not load champion: {e}")

        if champion is None:
            logger.warning("No champion found, using random signals")
            champion = RandomSignalBot()

        trader = MultiExchangePaperTrader(config)

        original_tick = trader._tick

        def tracked_tick(bot):
            result = original_tick(bot)

            summary = trader.get_summary()
            total_equity = sum(s.get("equity", 0) for s in summary.values())
            total_position = sum(s.get("position", 0) for s in summary.values())

            observer.update_metrics(
                portfolio_value=total_equity,
                position=total_position,
            )

            metrics = observer.get_trading_metrics()
            monitor.record_metrics(metrics)

            alerts = observer.check_alerts(monitor)
            for alert in alerts:
                logger.warning(alert)

            return result

        trader._tick = tracked_tick

        trader.run(champion, max_ticks=max_ticks)

    except KeyboardInterrupt:
        logger.info("Paper trading interrupted by user")
    except Exception as e:
        logger.error(f"Paper trading error: {e}")
        raise
    finally:
        state.is_running = False
        observer.save_session(Path(output_dir))
        monitor.stop_monitoring()

        logger.info(f"\n{'=' * 50}")
        logger.info(f"Paper Trading Session Summary")
        logger.info(f"{'=' * 50}")
        logger.info(f"Session ID: {session_id}")
        logger.info(
            f"Duration: {(datetime.now() - state.start_time).total_seconds() / 3600:.2f} hours"
        )
        logger.info(f"Initial Capital: ${state.initial_capital:,.2f}")
        logger.info(f"Final Capital: ${state.current_capital:,.2f}")
        logger.info(
            f"Total PnL: ${state.total_pnl:,.2f} ({state.total_pnl / state.initial_capital * 100:.2f}%)"
        )
        logger.info(f"Daily PnL: ${state.daily_pnl:,.2f}")
        logger.info(f"Total Trades: {state.total_trades}")
        logger.info(f"Win Rate: {state.win_rate * 100:.1f}%")
        logger.info(f"Max Drawdown: {state.max_drawdown_pct * 100:.2f}%")
        logger.info(f"{'=' * 50}")


def main():
    parser = argparse.ArgumentParser(description="Paper Trading Runner & Observer")
    parser.add_argument("--symbol", type=str, default="BTC/USDT", help="Trading symbol")
    parser.add_argument(
        "--timeframe", type=str, default="1h", help="Timeframe (1m, 5m, 1h, 4h, 1d)"
    )
    parser.add_argument("--capital", type=float, default=10000.0, help="Initial capital")
    parser.add_argument(
        "--exchanges", type=str, default="binance", help="Comma-separated exchanges"
    )
    parser.add_argument("--ticks", type=int, default=0, help="Number of ticks (0 = infinite)")
    parser.add_argument("--output", type=str, default="data/paper_trades", help="Output directory")
    parser.add_argument(
        "--champion", type=str, default="data/cache/multiverse_champion.pkl", help="Champion path"
    )

    args = parser.parse_args()

    exchanges = [e.strip() for e in args.exchanges.split(",")]

    run_paper_trading(
        symbol=args.symbol,
        timeframe=args.timeframe,
        initial_capital=args.capital,
        exchanges=exchanges,
        max_ticks=args.ticks,
        output_dir=args.output,
        champion_path=args.champion,
    )


if __name__ == "__main__":
    main()
