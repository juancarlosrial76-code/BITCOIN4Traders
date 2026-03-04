"""
Multi-Exchange Paper Trading Engine
=====================================
Verbindet KuCoin, Binance und Bybit gleichzeitig via ccxt im Paper-Trading
Modus.  Zweck: Trainingsdaten sammeln, Strategien validieren, OHNE echtes
Kapital zu riskieren.

Architektur
-----------
                ┌─────────────────────────────────────────────┐
                │         MultiExchangePaperTrader            │
                ├──────────────┬──────────────┬───────────────┤
                │   KuCoin     │   Binance    │    Bybit      │
                │  (ccxt REST) │  (ccxt REST) │  (ccxt REST)  │
                └──────┬───────┴──────┬───────┴──────┬────────┘
                       │              │               │
                 OHLCV-Ticker   OHLCV-Ticker    OHLCV-Ticker
                       │              │               │
                 FeatureEngine (gemeinsam)
                       │
                 DarwinBot / PPOAgent
                       │
                 PaperPortfolio (simulierte Orders)
                       │
                 TrainingDataCollector (OHLCV + Signale speichern)

Paper-Trading-Logik
-------------------
- Kein echtes Kapital: Alle Orders werden lokal simuliert.
- Echter Preis-Feed von allen drei Exchanges via ccxt `fetch_ticker()`.
- Slippage + Fees werden realitätsnah simuliert (konfigurierbar pro Exchange).
- Alle Trades werden in `data/paper_trades/` als Parquet gespeichert.
- Telemetriedaten (OHLCV + Feature-Vektoren) werden kontinuierlich gelogt.

Verwendung
----------
    from src.execution.multi_exchange_paper_trader import (
        MultiExchangePaperTrader, PaperTraderConfig
    )

    cfg = PaperTraderConfig(
        symbol='BTC/USDT',
        timeframe='1h',
        initial_capital=10_000.0,
        exchanges=['kucoin', 'binance', 'bybit'],
    )
    trader = MultiExchangePaperTrader(cfg)
    trader.run(champion_bot)   # blockierender Loop (Ctrl+C zum Stoppen)

    # Oder in async:
    await trader.run_async(champion_bot)

Umgebungsvariablen (optional — ohne Keys nur Public-Feed)
----------------------------------------------------------
    KUCOIN_API_KEY / KUCOIN_API_SECRET / KUCOIN_PASSPHRASE
    BINANCE_API_KEY / BINANCE_API_SECRET
    BYBIT_API_KEY / BYBIT_API_SECRET

Ohne Keys: nur öffentliche Marktdaten (ausreichend für Paper-Trading).
"""

from __future__ import annotations

import asyncio
import os
import time
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
from loguru import logger

# ── ccxt (graceful import) ────────────────────────────────────────────────────
try:
    import ccxt

    _CCXT_OK = True
except ImportError:
    _CCXT_OK = False
    logger.warning("ccxt nicht installiert.  pip install ccxt")


# ─────────────────────────────────────────────────────────────────────────────
# Konfiguration
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class ExchangeFeeConfig:
    """Reale Gebührenstruktur pro Exchange."""

    maker_fee: float = 0.001  # 0.1%
    taker_fee: float = 0.001
    slippage: float = 0.0005  # 0.05% Slippage


_DEFAULT_FEES: Dict[str, ExchangeFeeConfig] = {
    "kucoin": ExchangeFeeConfig(maker_fee=0.0008, taker_fee=0.0010, slippage=0.0004),
    "binance": ExchangeFeeConfig(maker_fee=0.0002, taker_fee=0.0004, slippage=0.0003),
    "bybit": ExchangeFeeConfig(maker_fee=0.0001, taker_fee=0.0006, slippage=0.0004),
}


@dataclass
class PaperTraderConfig:
    """
    Konfiguration für den Multi-Exchange Paper Trader.

    Attributes
    ----------
    symbol : str
        Handelspaar, z.B. 'BTC/USDT'
    timeframe : str
        OHLCV-Timeframe, z.B. '1h'
    initial_capital : float
        Startkapital in USDT (simuliert)
    exchanges : List[str]
        Liste der Exchanges.  Verfügbar: 'kucoin', 'binance', 'bybit'
    primary_exchange : str
        Exchange für Signalgenerierung (Preis-Feed für Bot)
    poll_interval_s : float
        Sekunden zwischen Ticker-Abfragen (pro Exchange)
    ohlcv_lookback : int
        Anzahl historischer Bars für Feature-Berechnung
    data_dir : str
        Verzeichnis zum Speichern von Paper-Trades + Trainingsdaten
    risk_per_trade : float
        Max. Kapital-% pro Trade (1%-Regel)
    max_drawdown : float
        Circuit-Breaker: Stopp wenn Drawdown diesen Wert überschreitet
    fee_config : Dict[str, ExchangeFeeConfig]
        Gebührenstruktur pro Exchange (Standard = reale Werte)
    """

    symbol: str = "BTC/USDT"
    timeframe: str = "1h"
    initial_capital: float = 10_000.0
    exchanges: List[str] = field(default_factory=lambda: ["kucoin", "binance", "bybit"])
    primary_exchange: str = "binance"
    poll_interval_s: float = 5.0
    ohlcv_lookback: int = 200
    data_dir: str = "data/paper_trades"
    risk_per_trade: float = 0.01  # 1%-Regel
    max_drawdown: float = 0.20  # 20%
    fee_config: Dict[str, ExchangeFeeConfig] = field(
        default_factory=lambda: dict(_DEFAULT_FEES)
    )
    log_ohlcv: bool = True  # OHLCV + Signale als Trainingsdaten loggen


# ─────────────────────────────────────────────────────────────────────────────
# Paper Portfolio (simulierte Kontoführung)
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class PaperTrade:
    """Einzelner simulierter Trade."""

    timestamp: datetime
    exchange: str
    symbol: str
    side: str  # 'buy' | 'sell'
    qty: float
    price: float
    fee: float
    slippage: float
    pnl: float  # realisierter P&L (nach Fee + Slippage)


class PaperPortfolio:
    """
    Simuliertes Portfolio mit realistischer Fee- und Slippage-Modellierung.

    Führt Buch über Cash, offene Position und alle Trades.
    """

    def __init__(self, initial_capital: float, fee_cfg: ExchangeFeeConfig):
        self.cash = initial_capital
        self.initial_cap = initial_capital
        self.fee_cfg = fee_cfg
        self.position = 0.0  # Anzahl BTC
        self.entry_price = 0.0
        self.trades: List[PaperTrade] = []
        self.equity_curve: List[Tuple[datetime, float]] = []

    @property
    def equity(self, price: float = 0.0) -> float:
        """Aktuelles Eigenkapital (Cash + Positionswert)."""
        return self.cash + self.position * price

    def current_equity(self, price: float) -> float:
        return self.cash + self.position * price

    @property
    def drawdown(self) -> float:
        if not self.equity_curve:
            return 0.0
        vals = [e for _, e in self.equity_curve]
        peak = max(vals)
        cur = vals[-1]
        return (peak - cur) / peak if peak > 0 else 0.0

    def execute(
        self,
        side: str,
        price: float,
        exchange: str,
        symbol: str,
        risk_pct: float = 0.01,
    ) -> Optional[PaperTrade]:
        """
        Führt einen simulierten Market-Order aus.

        Parameters
        ----------
        side        : 'buy' | 'sell'
        price       : Aktueller Marktpreis
        exchange    : Exchange-Name (für Gebühren)
        symbol      : Handelspaar
        risk_pct    : Anteil des Kapitals das eingesetzt wird
        """
        fee_cfg = self.fee_cfg  # globale fee config wird pro-call überschrieben
        fee_rate = fee_cfg.taker_fee
        slip = fee_cfg.slippage

        # Slippage-angepasster Preis
        fill_price = price * (1 + slip) if side == "buy" else price * (1 - slip)

        if side == "buy" and self.position <= 0:
            # Kaufen: max risk_pct des Kapitals
            spend = self.cash * risk_pct
            if spend < 10:  # Mindestbetrag
                return None
            qty = spend / fill_price
            fee = qty * fill_price * fee_rate
            total_cost = qty * fill_price + fee
            if total_cost > self.cash:
                return None
            self.cash -= total_cost
            self.position = qty
            self.entry_price = fill_price
            pnl = 0.0

        elif side == "sell" and self.position > 0:
            qty = self.position
            proceeds = qty * fill_price
            fee = proceeds * fee_rate
            net = proceeds - fee
            pnl = net - (self.entry_price * qty)  # realisierter P&L
            self.cash += net
            self.position = 0.0
            self.entry_price = 0.0
        else:
            return None  # Kein Signal oder Position falsch

        trade = PaperTrade(
            timestamp=datetime.now(timezone.utc),
            exchange=exchange,
            symbol=symbol,
            side=side,
            qty=qty,
            price=fill_price,
            fee=fee,
            slippage=slip * price,
            pnl=pnl if side == "sell" else 0.0,
        )
        self.trades.append(trade)
        return trade

    def update_equity(self, price: float) -> float:
        eq = self.current_equity(price)
        self.equity_curve.append((datetime.now(timezone.utc), eq))
        return eq

    def summary(self) -> Dict[str, Any]:
        if not self.equity_curve:
            return {}
        vals = [e for _, e in self.equity_curve]
        total_pnl = sum(t.pnl for t in self.trades)
        wins = [t for t in self.trades if t.pnl > 0]
        total_fee = sum(t.fee for t in self.trades)
        return {
            "n_trades": len(self.trades),
            "win_rate": len(wins) / len(self.trades) if self.trades else 0.0,
            "total_pnl": total_pnl,
            "total_fee": total_fee,
            "max_drawdown": self.drawdown,
            "final_equity": vals[-1],
            "return_pct": (vals[-1] - self.initial_cap) / self.initial_cap * 100,
        }

    def to_dataframe(self) -> pd.DataFrame:
        if not self.trades:
            return pd.DataFrame()
        return pd.DataFrame(
            [
                {
                    "timestamp": t.timestamp.isoformat(),
                    "exchange": t.exchange,
                    "symbol": t.symbol,
                    "side": t.side,
                    "qty": t.qty,
                    "price": t.price,
                    "fee": t.fee,
                    "slippage": t.slippage,
                    "pnl": t.pnl,
                }
                for t in self.trades
            ]
        )


# ─────────────────────────────────────────────────────────────────────────────
# Exchange-Connector (ccxt, synchron)
# ─────────────────────────────────────────────────────────────────────────────


class CCXTConnector:
    """
    Leichtgewichtiger ccxt-Wrapper für einen einzelnen Exchange.

    Unterstützt: kucoin, binance, bybit (und jeden anderen ccxt-Exchange).
    Ohne API-Keys: nur öffentliche Daten (ausreichend für Paper-Trading).
    """

    def __init__(self, exchange_id: str):
        if not _CCXT_OK:
            raise ImportError("ccxt nicht installiert.  pip install ccxt")

        self.exchange_id = exchange_id
        cls = getattr(ccxt, exchange_id, None)
        if cls is None:
            raise ValueError(f"Unbekannte Exchange: {exchange_id}")

        # Credentials aus Umgebungsvariablen (optional)
        prefix = exchange_id.upper()
        kwargs: Dict[str, Any] = {"enableRateLimit": True}

        api_key = os.getenv(f"{prefix}_API_KEY")
        api_secret = os.getenv(f"{prefix}_API_SECRET")
        passphrase = os.getenv(f"{prefix}_PASSPHRASE")  # KuCoin

        if api_key:
            kwargs["apiKey"] = api_key
            kwargs["secret"] = api_secret or ""
        if passphrase:
            kwargs["password"] = passphrase

        self._ex: ccxt.Exchange = cls(kwargs)
        logger.info(
            f"CCXTConnector: {exchange_id} initialisiert "
            f"({'mit Keys' if api_key else 'Public-only'})"
        )

    def fetch_ticker(self, symbol: str) -> Optional[Dict]:
        """Aktueller Preis (last, bid, ask)."""
        try:
            return self._ex.fetch_ticker(symbol)
        except Exception as e:
            logger.warning(f"[{self.exchange_id}] fetch_ticker failed: {e}")
            return None

    def fetch_ohlcv(
        self, symbol: str, timeframe: str, limit: int = 200
    ) -> Optional[pd.DataFrame]:
        """OHLCV-Daten als DataFrame."""
        try:
            raw = self._ex.fetch_ohlcv(symbol, timeframe, limit=limit)
            if not raw:
                return None
            df = pd.DataFrame(
                raw, columns=["timestamp", "open", "high", "low", "close", "volume"]
            )
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
            df.set_index("timestamp", inplace=True)
            return df
        except Exception as e:
            logger.warning(f"[{self.exchange_id}] fetch_ohlcv failed: {e}")
            return None

    def fetch_price(self, symbol: str) -> Optional[float]:
        """Letzter Handelspreis."""
        t = self.fetch_ticker(symbol)
        return float(t["last"]) if t and t.get("last") else None


# ─────────────────────────────────────────────────────────────────────────────
# Training Data Collector
# ─────────────────────────────────────────────────────────────────────────────


class TrainingDataCollector:
    """
    Speichert OHLCV-Daten + Bot-Signale für späteres RL-Training.

    Schema pro Zeile:
        timestamp | exchange | open | high | low | close | volume |
        signal (-1/0/1) | bot_name | equity
    """

    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._rows: List[Dict] = []
        self._flush_every = 100  # Auto-flush nach N Zeilen

    def record(
        self,
        exchange: str,
        ohlcv_row: Dict[str, float],
        signal: int,
        bot_name: str,
        equity: float,
        price: float,
    ) -> None:
        row = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "exchange": exchange,
            **ohlcv_row,
            "signal": signal,
            "bot_name": bot_name,
            "equity": equity,
            "price": price,
        }
        self._rows.append(row)
        if len(self._rows) >= self._flush_every:
            self.flush()

    def flush(self) -> None:
        if not self._rows:
            return
        df = pd.DataFrame(self._rows)
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        out = self.data_dir / f"training_data_{ts}.parquet"
        df.to_parquet(out, index=False)
        logger.info(f"TrainingData: {len(df)} Zeilen -> {out}")
        self._rows.clear()


# ─────────────────────────────────────────────────────────────────────────────
# Haupt-Engine
# ─────────────────────────────────────────────────────────────────────────────


class MultiExchangePaperTrader:
    """
    Multi-Exchange Paper Trading Engine.

    Verbindet KuCoin, Binance und Bybit gleichzeitig.  Beobachtet echte
    Preise, generiert Signale mit dem übergebenen Bot/Champion und führt
    simulierte Orders aus.

    Parameters
    ----------
    config : PaperTraderConfig
        Vollständige Konfiguration.

    Example
    -------
        cfg = PaperTraderConfig(
            symbol='BTC/USDT',
            exchanges=['kucoin', 'binance', 'bybit'],
            initial_capital=10_000.0,
        )
        trader = MultiExchangePaperTrader(cfg)
        trader.run(champion_bot)
    """

    def __init__(self, config: PaperTraderConfig):
        self.cfg = config

        # Exchange-Connectors
        self.connectors: Dict[str, CCXTConnector] = {}
        for ex_id in config.exchanges:
            try:
                self.connectors[ex_id] = CCXTConnector(ex_id)
            except Exception as e:
                logger.error(f"Exchange {ex_id} konnte nicht verbunden werden: {e}")

        if not self.connectors:
            raise RuntimeError("Keine Exchange konnte verbunden werden.")

        # Sicherstellen dass primary_exchange verfügbar ist
        if config.primary_exchange not in self.connectors:
            config.primary_exchange = next(iter(self.connectors))
            logger.warning(
                f"primary_exchange nicht verfügbar — verwende '{config.primary_exchange}'"
            )

        # Ein Portfolio pro Exchange (separate Buchhaltung)
        self.portfolios: Dict[str, PaperPortfolio] = {
            ex_id: PaperPortfolio(
                initial_capital=config.initial_capital,
                fee_cfg=config.fee_config.get(ex_id, ExchangeFeeConfig()),
            )
            for ex_id in self.connectors
        }

        # Trainingsdaten-Sammler
        self.collector = TrainingDataCollector(config.data_dir)

        # Zustand
        self._running = False
        self._tick_count = 0
        self._last_prices: Dict[str, float] = {}

        logger.info(
            f"MultiExchangePaperTrader bereit | "
            f"Exchanges: {list(self.connectors.keys())} | "
            f"Symbol: {config.symbol} | "
            f"Kapital: ${config.initial_capital:,.0f}"
        )

    # ------------------------------------------------------------------
    # Öffentliche API
    # ------------------------------------------------------------------

    def run(self, bot, max_ticks: int = 0) -> None:
        """
        Blockierender Trading-Loop.

        Parameters
        ----------
        bot       : DarwinBot oder PPOAgent — muss compute_signals(df) haben
        max_ticks : 0 = unbegrenzt (Ctrl+C zum Stoppen)
        """
        self._running = True
        logger.warning(
            f"Paper Trading GESTARTET | {self.cfg.symbol} | Ctrl+C zum Stoppen"
        )
        try:
            while self._running:
                if max_ticks > 0 and self._tick_count >= max_ticks:
                    break
                self._tick(bot)
                self._tick_count += 1
                time.sleep(self.cfg.poll_interval_s)
        except KeyboardInterrupt:
            logger.warning("Paper Trading gestoppt (KeyboardInterrupt)")
        finally:
            self._shutdown()

    def stop(self) -> None:
        self._running = False

    def get_summary(self) -> Dict[str, Any]:
        """Summary aller Portfolios."""
        result = {}
        for ex_id, pf in self.portfolios.items():
            price = self._last_prices.get(ex_id, 0.0)
            result[ex_id] = {
                "exchange": ex_id,
                "price": price,
                **pf.summary(),
                "equity": pf.current_equity(price),
            }
        return result

    def print_summary(self) -> None:
        print("\n" + "=" * 60)
        print("  PAPER TRADING ZUSAMMENFASSUNG")
        print("=" * 60)
        for ex_id, s in self.get_summary().items():
            print(f"\n  {ex_id.upper()}")
            print(f"    Kapital:     ${s.get('equity', 0):>10,.2f}")
            print(f"    Return:      {s.get('return_pct', 0):>+8.2f}%")
            print(f"    Trades:      {s.get('n_trades', 0)}")
            print(f"    Win-Rate:    {s.get('win_rate', 0) * 100:>6.1f}%")
            print(f"    Max DD:      {s.get('max_drawdown', 0) * 100:>6.1f}%")
            print(f"    Fees gezahlt:${s.get('total_fee', 0):>8.2f}")
        print("=" * 60)

    # ------------------------------------------------------------------
    # Interner Tick-Loop
    # ------------------------------------------------------------------

    def _tick(self, bot) -> None:
        """Ein Polling-Zyklus: Preise abrufen → Signal → Order."""
        for ex_id, conn in self.connectors.items():
            try:
                self._process_exchange(ex_id, conn, bot)
            except Exception as e:
                logger.warning(f"[{ex_id}] Tick-Fehler: {e}")

    def _process_exchange(self, ex_id: str, conn: CCXTConnector, bot) -> None:
        """Verarbeitet einen Tick für eine einzelne Exchange."""
        symbol = self.cfg.symbol
        tf = self.cfg.timeframe
        pf = self.portfolios[ex_id]

        # 1. OHLCV holen (für Feature-Berechnung)
        df = conn.fetch_ohlcv(symbol, tf, limit=self.cfg.ohlcv_lookback)
        if df is None or len(df) < 10:
            return

        price = float(df["close"].iloc[-1])
        self._last_prices[ex_id] = price

        # 2. Equity aktualisieren (für DD-Check)
        equity = pf.update_equity(price)

        # 3. Circuit-Breaker: Drawdown-Check
        if pf.drawdown > self.cfg.max_drawdown:
            logger.warning(
                f"[{ex_id}] Circuit-Breaker: DD={pf.drawdown * 100:.1f}% > "
                f"{self.cfg.max_drawdown * 100:.1f}%"
            )
            if pf.position > 0:
                pf.execute(
                    "sell", price, ex_id, symbol, risk_pct=self.cfg.risk_per_trade
                )
            return

        # 4. Signal vom Bot
        signal = self._get_signal(bot, df, ex_id)

        # 5. Trainingsdaten loggen
        if self.cfg.log_ohlcv:
            last_row = df.iloc[-1]
            self.collector.record(
                exchange=ex_id,
                ohlcv_row={
                    "open": float(last_row["open"]),
                    "high": float(last_row["high"]),
                    "low": float(last_row["low"]),
                    "close": price,
                    "volume": float(last_row["volume"]),
                },
                signal=signal,
                bot_name=getattr(bot, "name", str(type(bot).__name__)),
                equity=equity,
                price=price,
            )

        # 6. Order ausführen
        if signal == 1 and pf.position <= 0:
            trade = pf.execute(
                "buy", price, ex_id, symbol, risk_pct=self.cfg.risk_per_trade
            )
            if trade:
                logger.info(
                    f"[{ex_id}] BUY  {trade.qty:.6f} BTC @ ${trade.price:,.2f} "
                    f"| Fee=${trade.fee:.2f}"
                )

        elif signal == -1 and pf.position > 0:
            trade = pf.execute(
                "sell", price, ex_id, symbol, risk_pct=self.cfg.risk_per_trade
            )
            if trade:
                logger.info(
                    f"[{ex_id}] SELL {trade.qty:.6f} BTC @ ${trade.price:,.2f} "
                    f"| P&L=${trade.pnl:+.2f}"
                )

        # 7. Periodisches Status-Log
        if self._tick_count % 12 == 0:  # ca. jede Stunde bei 5s Poll
            logger.warning(
                f"[{ex_id}] Tick={self._tick_count:,} | "
                f"Preis=${price:,.2f} | "
                f"Equity=${equity:,.2f} | "
                f"DD={pf.drawdown * 100:.1f}%"
            )

    def _get_signal(self, bot, df: pd.DataFrame, ex_id: str) -> int:
        """
        Ruft das Signal vom Bot ab.  Unterstützt DarwinBot und PPOAgent.

        Returns
        -------
        int  -1 (short/sell), 0 (neutral), +1 (long/buy)
        """
        try:
            # DarwinBot API
            if hasattr(bot, "compute_signals"):
                close = df["close"].values.astype(np.float64)
                sig_arr = bot.compute_signals(close)
                return (
                    int(np.sign(sig_arr[-1]))
                    if sig_arr is not None and len(sig_arr)
                    else 0
                )

            # PPOAgent API
            if hasattr(bot, "select_action"):
                # Minimale Feature-Extraktion: Normalisierte Returns
                close = df["close"].values
                returns = np.diff(np.log(close + 1e-8))
                state = returns[-20:].astype(np.float32)
                if len(state) < 20:
                    state = np.pad(state, (20 - len(state), 0))
                action, _, _, _ = bot.select_action(state)
                # Action-Space: 0=Short100, 1=Short50, 2=Neutral, 3=Long33,
                #               4=Long50,  5=Long75, 6=Long100
                if action <= 1:
                    return -1
                elif action == 2:
                    return 0
                else:
                    return 1

        except Exception as e:
            logger.debug(f"[{ex_id}] Signal-Fehler: {e}")
        return 0

    def _shutdown(self) -> None:
        """Abschluss: Trades speichern, Summary ausgeben."""
        self.collector.flush()
        # Trade-Log als Parquet speichern
        out_dir = Path(self.cfg.data_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        for ex_id, pf in self.portfolios.items():
            df_trades = pf.to_dataframe()
            if not df_trades.empty:
                ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
                out = out_dir / f"paper_trades_{ex_id}_{ts}.parquet"
                df_trades.to_parquet(out, index=False)
                logger.info(f"Trades gespeichert: {out}")
        self.print_summary()


# ─────────────────────────────────────────────────────────────────────────────
# Factory-Funktion (für Colab / Skripte)
# ─────────────────────────────────────────────────────────────────────────────


def create_paper_trader(
    symbol: str = "BTC/USDT",
    timeframe: str = "1h",
    initial_capital: float = 10_000.0,
    exchanges: Optional[List[str]] = None,
    data_dir: str = "data/paper_trades",
    poll_interval_s: float = 5.0,
) -> MultiExchangePaperTrader:
    """
    Erstellt einen MultiExchangePaperTrader mit Standardkonfiguration.

    Parameters
    ----------
    exchanges : Liste der Exchanges.  None = alle drei (kucoin, binance, bybit)

    Example
    -------
        trader = create_paper_trader(symbol='BTC/USDT', initial_capital=10_000)
        trader.run(champion_bot)
    """
    if exchanges is None:
        exchanges = ["kucoin", "binance", "bybit"]

    cfg = PaperTraderConfig(
        symbol=symbol,
        timeframe=timeframe,
        initial_capital=initial_capital,
        exchanges=exchanges,
        primary_exchange="binance",
        poll_interval_s=poll_interval_s,
        data_dir=data_dir,
    )
    return MultiExchangePaperTrader(cfg)
