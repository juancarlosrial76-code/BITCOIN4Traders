"""
Module A — Lokale Ausführungs-Engine mit Ably-Bridge
=====================================================
Läuft auf dem lokalen Rechner (24/7).

Aufgaben:
  1. Marktdaten von Exchange holen (CCXT, public endpoint)
  2. Features berechnen
  3. Marktdaten → Ably → Colab publishen (bt4t/market/BTCUSDT)
  4. Signale von Colab abonnieren (bt4t/signals)
  5. Signale validieren (Staleness, Confidence)
  6. Paper-Order ausführen
  7. Portfolio-State → Ably publishen (bt4t/portfolio/state)
  8. Heartbeat von Colab überwachen → bei Timeout pausieren

Voraussetzungen:
  pip install ably ccxt loguru

Umgebungsvariablen (.env):
  ABLY_API_KEY=your_ably_root_key   # https://ably.com → free tier

Verwendung:
  python colab_bridge/module_a_local.py
  python colab_bridge/module_a_local.py --symbol ETHUSDT --interval 15
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# ── Abhängigkeiten ────────────────────────────────────────────────────────────
try:
    from ably import AblyRealtime

    _ABLY_OK = True
except ImportError:
    _ABLY_OK = False

try:
    import ccxt

    _CCXT_OK = True
except ImportError:
    _CCXT_OK = False

import numpy as np
import pandas as pd
from loguru import logger

# ── Logging ───────────────────────────────────────────────────────────────────
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
    LOG_DIR / "module_a.log", rotation="20 MB", retention="7 days", level="DEBUG"
)

# ── Kanal-Namen (müssen mit Module B übereinstimmen) ─────────────────────────
CH_MARKET = "bt4t:market:{symbol}"  # Lokal → Colab
CH_SIGNALS = "bt4t:signals"  # Colab → Lokal
CH_PORTFOLIO = "bt4t:portfolio:state"  # Lokal → Colab
CH_HEALTH = "bt4t:health"  # Colab → Lokal (Heartbeat)
CH_CONTROL = "bt4t:control:cmd"  # Lokal → Colab (Befehle)
CH_ACK = "bt4t:control:ack"  # Colab → Lokal (Bestätigung)

# ── Signal-Validierung ────────────────────────────────────────────────────────
MAX_SIGNAL_AGE_S = 10.0  # Signal älter als 10s → verwerfen
MIN_CONFIDENCE = 0.55  # Mindest-Konfidenz für Ausführung
HEARTBEAT_TIMEOUT_S = 90.0  # Kein Heartbeat > 90s → Trading pausieren

# ── Paper-Portfolio (in-memory) ───────────────────────────────────────────────


class LocalPaperPortfolio:
    """Einfaches in-memory Paper-Portfolio für Module A."""

    def __init__(self, initial_capital: float = 10_000.0, fee_rate: float = 0.001):
        self.cash = initial_capital
        self.initial_cap = initial_capital
        self.position = 0.0  # BTC-Menge
        self.entry_price = 0.0
        self.fee_rate = fee_rate
        self.trades = []
        self.equity_history = []
        self._paused = False

    @property
    def paused(self) -> bool:
        return self._paused

    def pause(self):
        self._paused = True
        logger.warning("Portfolio: Trading PAUSIERT (kein Colab-Heartbeat)")

    def resume(self):
        self._paused = False
        logger.success("Portfolio: Trading FORTGESETZT")

    def execute(
        self, side: str, price: float, confidence: float = 1.0
    ) -> Optional[dict]:
        """Simuliert einen Market-Order. Gibt Trade-Dict zurück oder None."""
        if self._paused:
            logger.debug("Trade übersprungen — Portfolio pausiert")
            return None

        slip = 0.0003  # 0.03% Slippage
        fee_rate = self.fee_rate
        fill = price * (1 + slip) if side == "buy" else price * (1 - slip)
        risk_pct = 0.01 * confidence  # Skaliert Risiko mit Konfidenz

        pnl = 0.0

        if side == "buy" and self.position <= 0:
            if self.position < 0:
                # Short cover
                qty = abs(self.position)
                cost = qty * fill
                fee = cost * fee_rate
                pnl = (self.entry_price - fill) * qty - fee
                self.cash -= cost + fee
                self.position = 0.0
                self.entry_price = 0.0
                action = "COVER"
            else:
                # Long eröffnen
                spend = self.cash * risk_pct
                if spend < 5:
                    return None
                qty = spend / fill
                fee = qty * fill * fee_rate
                if qty * fill + fee > self.cash:
                    return None
                self.cash -= qty * fill + fee
                self.position = qty
                self.entry_price = fill
                action = "BUY"

        elif side == "sell" and self.position >= 0:
            if self.position > 0:
                # Long schließen
                qty = self.position
                proceeds = qty * fill
                fee = proceeds * fee_rate
                pnl = proceeds - fee - (self.entry_price * qty)
                self.cash += proceeds - fee
                self.position = 0.0
                self.entry_price = 0.0
                action = "SELL"
            else:
                # Short eröffnen
                margin = self.cash * risk_pct
                if margin < 5:
                    return None
                qty = margin / fill
                fee = qty * fill * fee_rate
                self.cash += qty * fill - fee
                self.position = -qty
                self.entry_price = fill
                pnl = 0.0
                action = "SHORT"
        else:
            return None

        trade = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "action": action,
            "side": side,
            "qty": qty,
            "price": fill,
            "fee": fee,
            "pnl": pnl,
            "confidence": confidence,
        }
        self.trades.append(trade)
        return trade

    def current_equity(self, price: float) -> float:
        return self.cash + self.position * price

    def state_dict(self, price: float) -> dict:
        eq = self.current_equity(price)
        initial = self.initial_cap
        trades = self.trades
        wins = [t for t in trades if t["pnl"] > 0]
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "equity": eq,
            "cash": self.cash,
            "position_btc": self.position,
            "entry_price": self.entry_price,
            "current_price": price,
            "return_pct": (eq - initial) / initial * 100,
            "n_trades": len(trades),
            "win_rate": len(wins) / len(trades) if trades else 0.0,
            "total_pnl": sum(t["pnl"] for t in trades),
            "paused": self._paused,
        }


# ── Feature-Berechnung ────────────────────────────────────────────────────────


def compute_features(df: pd.DataFrame) -> dict:
    """
    Berechnet Standard-Features aus OHLCV.
    Gibt dict zurück das als Ably-Message publisht wird.
    """
    close = df["close"].values.astype(float)
    volume = df["volume"].values.astype(float)
    n = len(close)

    # RSI(14)
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    if len(gain) >= 14:
        avg_gain = np.convolve(gain, np.ones(14) / 14, mode="valid")[-1]
        avg_loss = np.convolve(loss, np.ones(14) / 14, mode="valid")[-1]
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
    else:
        rsi = 50.0

    # Bollinger Bands (20, 2)
    if n >= 20:
        sma20 = close[-20:].mean()
        std20 = close[-20:].std()
        bb_upper = sma20 + 2 * std20
        bb_lower = sma20 - 2 * std20
        bb_pct = (close[-1] - bb_lower) / (bb_upper - bb_lower + 1e-10)
    else:
        sma20 = close[-1]
        bb_upper = bb_lower = close[-1]
        bb_pct = 0.5

    # MACD (12,26,9)
    if n >= 26:
        ema12 = pd.Series(close).ewm(span=12, adjust=False).mean().iloc[-1]
        ema26 = pd.Series(close).ewm(span=26, adjust=False).mean().iloc[-1]
        macd = ema12 - ema26
    else:
        macd = 0.0

    # Volume-Ratio (aktuell vs. 20er Durchschnitt)
    vol_ratio = volume[-1] / (volume[-20:].mean() + 1e-10) if n >= 20 else 1.0

    # Returns (normalisiert)
    returns_1h = (close[-1] / close[-2] - 1) if n >= 2 else 0.0
    returns_4h = (close[-1] / close[-5] - 1) if n >= 5 else 0.0
    returns_24h = (close[-1] / close[-25] - 1) if n >= 25 else 0.0

    # Raw OHLCV (letzter Bar)
    last = df.iloc[-1]

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "symbol": "BTCUSDT",
        # Raw
        "open": float(last["open"]),
        "high": float(last["high"]),
        "low": float(last["low"]),
        "close": float(last["close"]),
        "volume": float(last["volume"]),
        # Features
        "rsi14": float(rsi),
        "bb_pct": float(bb_pct),
        "bb_upper": float(bb_upper),
        "bb_lower": float(bb_lower),
        "macd": float(macd),
        "vol_ratio": float(vol_ratio),
        "return_1h": float(returns_1h),
        "return_4h": float(returns_4h),
        "return_24h": float(returns_24h),
        "sma20": float(sma20),
        # Close-Array (letzte 60 Bars für RL-Observation)
        "close_60": close[-60:].tolist() if n >= 60 else close.tolist(),
    }


# ── Haupt-Engine ──────────────────────────────────────────────────────────────


class ModuleA:
    """
    Lokale Ausführungs-Engine.

    Verbindet Exchange (CCXT) mit Colab (Ably).
    Führt Paper-Orders aus basierend auf Colab-Signalen.
    """

    def __init__(
        self,
        ably_key: str,
        symbol: str = "BTC/USDT",
        timeframe: str = "1h",
        exchange_id: str = "binance",
        poll_interval_s: float = 30.0,
        initial_capital: float = 10_000.0,
    ):
        if not _ABLY_OK:
            raise ImportError("pip install ably")
        if not _CCXT_OK:
            raise ImportError("pip install ccxt")

        self.ably_key = ably_key
        self.symbol = symbol
        self.symbol_ccxt = symbol  # z.B. 'BTC/USDT'
        self.symbol_clean = symbol.replace("/", "")  # 'BTCUSDT'
        self.timeframe = timeframe
        self.exchange_id = exchange_id
        self.poll_interval = poll_interval_s

        # Exchange-Connector
        ex_cls = getattr(ccxt, exchange_id)
        self.exchange = ex_cls({"enableRateLimit": True})
        logger.info(f"Exchange: {exchange_id} verbunden (Public-only)")

        # Portfolio
        self.portfolio = LocalPaperPortfolio(initial_capital=initial_capital)

        # State
        self._last_heartbeat: float = time.time()
        self._last_price: float = 0.0
        self._running = False
        self._ably: Optional[AblyRealtime] = None
        self._last_signal: Optional[dict] = None

        logger.success(
            f"Module A bereit | {symbol} | {exchange_id} | ${initial_capital:,.0f}"
        )

    # ── Ably-Verbindung ───────────────────────────────────────────────────────

    async def _connect_ably(self):
        """Stellt Ably-Verbindung her und abonniert Kanäle."""
        self._ably = AblyRealtime(self.ably_key)
        await self._ably.connection.once_async("connected")
        logger.success("Ably verbunden")

        # Signale von Colab abonnieren
        ch_signals = self._ably.channels.get(CH_SIGNALS)
        await ch_signals.subscribe(self._on_signal)

        # Heartbeat von Colab abonnieren
        ch_health = self._ably.channels.get(CH_HEALTH)
        await ch_health.subscribe(self._on_heartbeat)

        # Control-Acks von Colab abonnieren
        ch_ack = self._ably.channels.get(CH_ACK)
        await ch_ack.subscribe(self._on_ack)

        logger.info(f"Abonniert: {CH_SIGNALS}, {CH_HEALTH}, {CH_ACK}")

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def _on_signal(self, message):
        """Empfängt Handelssignal von Colab und führt Paper-Order aus."""
        try:
            data = (
                json.loads(message.data)
                if isinstance(message.data, str)
                else message.data
            )

            # ── Staleness-Check ───────────────────────────────────────────────
            sig_time_str = data.get("timestamp_utc", "")
            if sig_time_str:
                sig_time = datetime.fromisoformat(sig_time_str.replace("Z", "+00:00"))
                age_s = (datetime.now(timezone.utc) - sig_time).total_seconds()
                if age_s > MAX_SIGNAL_AGE_S:
                    logger.warning(f"Signal zu alt ({age_s:.1f}s) — verworfen")
                    return

            # ── Konfidenz-Check ───────────────────────────────────────────────
            confidence = float(data.get("confidence", 0.0))
            if confidence < MIN_CONFIDENCE:
                logger.debug(
                    f"Confidence {confidence:.2f} < {MIN_CONFIDENCE} — verworfen"
                )
                return

            action = data.get("action", "HOLD")  # BUY / SELL / HOLD
            model_version = data.get("model_version", "?")

            logger.info(
                f"Signal empfangen: {action} | confidence={confidence:.2f} | model={model_version}"
            )
            self._last_signal = data

            if self._last_price <= 0:
                logger.warning("Kein aktueller Preis — Signal übersprungen")
                return

            # ── Order ausführen ───────────────────────────────────────────────
            side = None
            if action == "BUY":
                side = "buy"
            elif action == "SELL":
                side = "sell"

            if side:
                trade = self.portfolio.execute(side, self._last_price, confidence)
                if trade:
                    eq = self.portfolio.current_equity(self._last_price)
                    logger.success(
                        f"TRADE: {trade['action']} {trade['qty']:.6f} BTC @ "
                        f"${trade['price']:,.2f} | P&L=${trade['pnl']:+.2f} | "
                        f"Equity=${eq:,.2f}"
                    )

        except Exception as e:
            logger.error(f"Signal-Verarbeitung Fehler: {e}", exc_info=True)

    def _on_heartbeat(self, message):
        """Aktualisiert Heartbeat-Timestamp und setzt Portfolio fort."""
        self._last_heartbeat = time.time()
        if self.portfolio.paused:
            self.portfolio.resume()

    def _on_ack(self, message):
        """Bestätigung von Colab für gesendete Befehle."""
        try:
            data = (
                json.loads(message.data)
                if isinstance(message.data, str)
                else message.data
            )
            logger.info(
                f"Colab-ACK: {data.get('cmd', '?')} → {data.get('status', '?')}"
            )
        except Exception:
            pass

    # ── Marktdaten publishen ──────────────────────────────────────────────────

    async def _publish_market_data(self, features: dict):
        """Publisht Feature-Dict auf Ably-Kanal → Colab."""
        ch_name = CH_MARKET.format(symbol=self.symbol_clean)
        channel = self._ably.channels.get(ch_name)
        await channel.publish("market_update", json.dumps(features))

    async def _publish_portfolio_state(self):
        """Publisht aktuellen Portfolio-State auf Ably → Colab."""
        if self._last_price <= 0:
            return
        state = self.portfolio.state_dict(self._last_price)
        channel = self._ably.channels.get(CH_PORTFOLIO)
        await channel.publish("portfolio_update", json.dumps(state))

    # ── Heartbeat-Überwachung ─────────────────────────────────────────────────

    def _check_heartbeat(self):
        """Pausiert Trading wenn Colab zu lange kein Heartbeat sendet."""
        age = time.time() - self._last_heartbeat
        if age > HEARTBEAT_TIMEOUT_S and not self.portfolio.paused:
            logger.warning(f"Kein Colab-Heartbeat seit {age:.0f}s — pausiere Trading")
            self.portfolio.pause()

    # ── Befehle an Colab senden ───────────────────────────────────────────────

    async def send_command(self, cmd: str, params: dict = None):
        """
        Sendet Steuerbefehl an Colab.

        Commands: PAUSE_INFERENCE, RESUME, RELOAD_MODEL, SHUTDOWN
        """
        payload = {
            "cmd": cmd,
            "params": params or {},
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        }
        channel = self._ably.channels.get(CH_CONTROL)
        await channel.publish("command", json.dumps(payload))
        logger.info(f"Befehl an Colab: {cmd}")

    # ── Status-Anzeige ────────────────────────────────────────────────────────

    def _print_status(self):
        """Gibt Kurzstatus in der Konsole aus."""
        if self._last_price <= 0:
            return
        state = self.portfolio.state_dict(self._last_price)
        hb_age = time.time() - self._last_heartbeat
        hb_status = f"{hb_age:.0f}s" if hb_age < 999 else "OFFLINE"
        sig_info = "–"
        if self._last_signal:
            sig_info = f"{self._last_signal.get('action', '?')} conf={self._last_signal.get('confidence', 0):.2f}"

        logger.info(
            f"STATUS | Preis=${self._last_price:,.2f} | "
            f"Equity=${state['equity']:,.2f} | "
            f"Return={state['return_pct']:+.2f}% | "
            f"Trades={state['n_trades']} | "
            f"Colab-HB={hb_status} | "
            f"Signal={sig_info}"
        )

    # ── Haupt-Loop ────────────────────────────────────────────────────────────

    async def run(self):
        """Startet den Haupt-Polling-Loop."""
        await self._connect_ably()
        self._running = True
        tick = 0

        logger.success("=" * 60)
        logger.success("  Module A gestartet — lokale Ausführungs-Engine")
        logger.success(f"  Publisht auf : {CH_MARKET.format(symbol=self.symbol_clean)}")
        logger.success(f"  Abonniert    : {CH_SIGNALS}, {CH_HEALTH}")
        logger.success("  Warte auf Colab-Signale...")
        logger.success("=" * 60)

        try:
            while self._running:
                try:
                    # 1. OHLCV holen
                    raw = self.exchange.fetch_ohlcv(
                        self.symbol_ccxt, self.timeframe, limit=200
                    )
                    if raw and len(raw) >= 10:
                        df = pd.DataFrame(
                            raw,
                            columns=[
                                "timestamp",
                                "open",
                                "high",
                                "low",
                                "close",
                                "volume",
                            ],
                        )
                        df["timestamp"] = pd.to_datetime(
                            df["timestamp"], unit="ms", utc=True
                        )
                        df.set_index("timestamp", inplace=True)

                        self._last_price = float(df["close"].iloc[-1])

                        # 2. Features berechnen
                        features = compute_features(df)

                        # 3. Marktdaten → Ably → Colab
                        await self._publish_market_data(features)

                        # 4. Portfolio-State → Ably
                        if tick % 5 == 0:  # alle 5 Ticks
                            await self._publish_portfolio_state()

                    # 5. Heartbeat-Check
                    self._check_heartbeat()

                    # 6. Status-Log alle 10 Ticks
                    if tick % 10 == 0:
                        self._print_status()

                    tick += 1

                except Exception as e:
                    logger.warning(f"Tick-Fehler: {e}")

                await asyncio.sleep(self.poll_interval)

        except asyncio.CancelledError:
            pass
        finally:
            await self._shutdown()

    async def _shutdown(self):
        """Sauberes Beenden."""
        self._running = False
        logger.info("Module A: Beende...")
        if self._ably:
            await self._ably.close()
        self.portfolio._print_summary()
        logger.success("Module A: Beendet")

    def _print_summary(self):
        if self._last_price <= 0:
            return
        state = self.portfolio.state_dict(self._last_price)
        print("\n" + "=" * 55)
        print("  MODULE A — ABSCHLUSS-ZUSAMMENFASSUNG")
        print("=" * 55)
        print(f"  Equity:     ${state['equity']:>10,.2f}")
        print(f"  Return:     {state['return_pct']:>+8.2f}%")
        print(f"  Trades:     {state['n_trades']}")
        print(f"  Win-Rate:   {state['win_rate'] * 100:>5.1f}%")
        print(f"  Total P&L: ${state['total_pnl']:>+9.2f}")
        print("=" * 55)


# ── CLI ───────────────────────────────────────────────────────────────────────


async def main():
    parser = argparse.ArgumentParser(description="Module A — Lokale Ausführungs-Engine")
    parser.add_argument("--symbol", default="BTC/USDT", help="Handelspaar")
    parser.add_argument("--timeframe", default="1h", help="OHLCV-Timeframe")
    parser.add_argument("--exchange", default="binance", help="Exchange-ID (ccxt)")
    parser.add_argument(
        "--interval", type=float, default=30.0, help="Poll-Intervall in Sekunden"
    )
    parser.add_argument(
        "--capital", type=float, default=10_000.0, help="Startkapital USDT"
    )
    parser.add_argument(
        "--ably-key", default=os.getenv("ABLY_API_KEY", ""), help="Ably API Key"
    )
    args = parser.parse_args()

    if not args.ably_key:
        logger.error(
            "Kein Ably API Key! "
            "Setze ABLY_API_KEY in .env oder --ably-key <key>\n"
            "Kostenloser Key: https://ably.com → Sign up → Create App → API Keys"
        )
        sys.exit(1)

    engine = ModuleA(
        ably_key=args.ably_key,
        symbol=args.symbol,
        timeframe=args.timeframe,
        exchange_id=args.exchange,
        poll_interval_s=args.interval,
        initial_capital=args.capital,
    )

    try:
        await engine.run()
    except KeyboardInterrupt:
        logger.info("Strg+C — beende Module A")


if __name__ == "__main__":
    asyncio.run(main())
