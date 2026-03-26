"""
papertrade_local.py — Local Paper-Trading Starter
==================================================
Starts the Multi-Exchange Paper Trader with the Darwin Champion as signal engine.
No API key required — runs entirely on public exchange data.

Usage:
    python papertrade_local.py                     # Default: Binance, BTC/USDT, 1h
    python papertrade_local.py --capital 5000      # Starting capital in USDT
    python papertrade_local.py --exchange binance  # Select exchange
    python papertrade_local.py --interval 60       # Poll interval in seconds
    python papertrade_local.py --timeframe 15m     # Timeframe (1m, 5m, 15m, 1h, 4h)
    python papertrade_local.py --symbol ETH/USDT   # Different trading pair

Ctrl+C: Clean shutdown + save trade log as Parquet.
"""

from __future__ import annotations

import argparse
import os
import sys
import signal
import time
from datetime import datetime, timezone
from pathlib import Path

# ── Add project root to sys.path ─────────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from loguru import logger

# ── Configure logger ─────────────────────────────────────────────────────────
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | <cyan>{message}</cyan>",
    level="INFO",
    colorize=True,
)
logger.add(
    LOG_DIR / "papertrade_local.log",
    rotation="50 MB",
    retention="14 days",
    level="DEBUG",
    encoding="utf-8",
)


# ── Load champion ────────────────────────────────────────────────────────────


def load_champion():
    """Loads the Darwin Champion from cache."""
    champion_pkl = ROOT / "data/cache/multiverse_champion.pkl"
    champion_meta = ROOT / "data/cache/multiverse_champion_meta.json"

    if not champion_pkl.exists():
        logger.warning("No saved champion found -> using synthetic mode")
        return None

    try:
        from src.math_tools.archive.darwin_legacy import ChampionPersistence

        champion = ChampionPersistence.load(
            str(champion_pkl),
            str(champion_meta) if champion_meta.exists() else None,
        )
        logger.success(
            f"Champion loaded: {getattr(champion, 'name', type(champion).__name__)}"
        )
        return champion
    except Exception as e:
        logger.error(f"Failed to load champion: {e}")
        logger.warning("Fallback: using synthetic signal")
        return None


# ── Signal wrapper (Champion or fallback) ────────────────────────────────────


class ChampionSignalAdapter:
    """
    Adapter between DarwinBot Champion and MultiExchangePaperTrader.
    Implements the minimal interface: compute_signals(close_array) -> array
    """

    def __init__(self, champion):
        self.champion = champion
        self._tick = 0

    def compute_signals(self, close: "np.ndarray") -> "np.ndarray":
        """Returns signal array: 1=Long, -1=Short, 0=Flat."""
        import numpy as np

        if self.champion is not None:
            try:
                return self.champion.compute_signals(close)
            except Exception as e:
                logger.warning(f"Champion compute_signals error: {e}")

        # Fallback: simple RSI logic (for demo only)
        self._tick += 1
        n = len(close)
        signals = np.zeros(n)
        if n >= 14:
            delta = np.diff(close)
            gain = np.where(delta > 0, delta, 0)
            loss = np.where(delta < 0, -delta, 0)
            avg_gain = np.convolve(gain, np.ones(14) / 14, mode="valid")
            avg_loss = np.convolve(loss, np.ones(14) / 14, mode="valid")
            rs = np.where(avg_loss == 0, 100, avg_gain / (avg_loss + 1e-10))
            rsi = 100 - (100 / (1 + rs))
            offset = n - len(rsi)
            signals[offset:] = np.where(rsi < 30, 1, np.where(rsi > 70, -1, 0))
        return signals


# ── Console dashboard ────────────────────────────────────────────────────────


def print_dashboard(
    portfolios: dict, prices: dict, trade_counts: dict, start_time: float
):
    """Prints a simple terminal dashboard."""
    elapsed = time.time() - start_time
    h, m, s = int(elapsed // 3600), int((elapsed % 3600) // 60), int(elapsed % 60)

    print("\n" + "═" * 65)
    print(
        f"  BITCOIN4Traders — Paper Trade Dashboard    {datetime.now().strftime('%H:%M:%S')}"
    )
    print(f"  Runtime: {h:02d}h {m:02d}m {s:02d}s")
    print("═" * 65)
    print(f"  {'Exchange':<12} {'Preis':>12} {'Equity':>12} {'P&L%':>8} {'Trades':>7}")
    print("─" * 65)

    for ex, portfolio in portfolios.items():
        price = prices.get(ex, 0.0)
        if price > 0:
            eq = portfolio.current_equity(price)
            initial = portfolio.initial_cap
            pnl_pct = (eq - initial) / initial * 100
            n_trades = trade_counts.get(ex, 0)
            pnl_color = "+" if pnl_pct >= 0 else ""
            print(
                f"  {ex:<12} {price:>12,.2f} {eq:>12,.2f} "
                f"{pnl_color}{pnl_pct:>7.2f}% {n_trades:>7}"
            )
    print("═" * 65)
    print("  Ctrl+C for clean shutdown")


# ── Main trading loop ────────────────────────────────────────────────────────


def run_papertrade(
    symbol: str = "BTC/USDT",
    timeframe: str = "1h",
    capital: float = 10_000.0,
    exchanges: list = None,
    poll_interval: float = 30.0,
    dashboard_interval: float = 60.0,
    risk_per_trade: float = 0.01,
    max_drawdown: float = 0.20,
):
    """
    Main paper-trading loop.

    Parameters
    ----------
    symbol          : Trading pair, e.g. 'BTC/USDT'
    timeframe       : OHLCV timeframe
    capital         : Starting capital (USDT, simulated)
    exchanges       : List of exchanges (default: ['binance'])
    poll_interval   : Seconds between ticker requests
    dashboard_interval: Seconds between dashboard updates
    risk_per_trade  : Max. capital % per trade (default: 1%)
    max_drawdown    : Circuit-breaker threshold (default: 20%)
    """
    if exchanges is None:
        exchanges = ["binance"]

    from src.execution.multi_exchange_paper_trader import (
        MultiExchangePaperTrader,
        PaperTraderConfig,
    )

    cfg = PaperTraderConfig(
        symbol=symbol,
        timeframe=timeframe,
        initial_capital=capital,
        exchanges=exchanges,
        primary_exchange=exchanges[0],
        poll_interval_s=poll_interval,
        ohlcv_lookback=200,
        data_dir="data/paper_trades",
        risk_per_trade=risk_per_trade,
        max_drawdown=max_drawdown,
        log_ohlcv=True,
    )

    champion = load_champion()
    bot = ChampionSignalAdapter(champion)

    trader = MultiExchangePaperTrader(cfg)

    # ── Signal handler for clean shutdown ────────────────────────────────────
    def _on_signal(sig, frame):
        logger.info("Stop signal received — shutting down paper trading...")
        trader.stop()  # sets self._running = False in the trader

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    logger.success("=" * 60)
    logger.success("  BITCOIN4Traders — Paper Trade started")
    logger.success(f"  Symbol    : {symbol}")
    logger.success(f"  Timeframe : {timeframe}")
    logger.success(f"  Capital   : ${capital:,.0f} USDT (simulated)")
    logger.success(f"  Exchanges : {', '.join(exchanges)}")
    logger.success(
        f"  Champion  : {getattr(champion, 'name', 'RSI-Fallback') if champion else 'RSI-Fallback'}"
    )
    logger.success(f"  Risk/Trade  : {risk_per_trade * 100:.1f}%")
    logger.success(f"  CB-Drawdown : {max_drawdown * 100:.0f}%")
    logger.success("=" * 60)

    # ── Start trading ────────────────────────────────────────────────────────
    try:
        trader.run(bot)  # blocks until Ctrl+C or trader.stop()
    except KeyboardInterrupt:
        trader.stop()
    except Exception as e:
        logger.error(f"Critical error in trading loop: {e}", exc_info=True)
    finally:
        logger.success("Paper trading stopped. Trade log in data/paper_trades/")


# ── CLI ───────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="BITCOIN4Traders — Local Paper Trader",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python papertrade_local.py
  python papertrade_local.py --capital 5000 --exchange binance kucoin
  python papertrade_local.py --timeframe 15m --interval 30
  python papertrade_local.py --symbol ETH/USDT --capital 2000
        """,
    )
    parser.add_argument(
        "--symbol", default="BTC/USDT", help="Trading pair (default: BTC/USDT)"
    )
    parser.add_argument(
        "--timeframe",
        default="1h",
        choices=["1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "12h", "1d"],
        help="OHLCV timeframe (default: 1h)",
    )
    parser.add_argument(
        "--capital",
        type=float,
        default=10_000.0,
        help="Starting capital in USDT (default: 10000)",
    )
    parser.add_argument(
        "--exchange",
        nargs="+",
        default=["binance"],
        choices=["binance", "kucoin", "bybit"],
        help="Exchanges (default: binance). Multiple allowed: --exchange binance kucoin",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=30.0,
        help="Poll interval in seconds (default: 30)",
    )
    parser.add_argument(
        "--risk", type=float, default=1.0, help="Risk per trade in %% (default: 1.0)"
    )
    parser.add_argument(
        "--max-drawdown",
        type=float,
        default=20.0,
        help="Circuit-breaker drawdown in %% (default: 20)",
    )

    args = parser.parse_args()

    run_papertrade(
        symbol=args.symbol,
        timeframe=args.timeframe,
        capital=args.capital,
        exchanges=args.exchange,
        poll_interval=args.interval,
        risk_per_trade=args.risk / 100.0,
        max_drawdown=args.max_drawdown / 100.0,
    )


if __name__ == "__main__":
    main()
