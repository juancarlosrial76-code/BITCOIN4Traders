"""
Live WebSocket Trader using ccxt.pro
=====================================
Latency-optimized async execution engine for live crypto trading.
Uses ccxt.pro's WebSocket interface for real-time data and order execution.

Architecture:
    LiveTraderConfig  →  static configuration (exchange, symbol, risk params)
    LiveTrader        →  async event loop that subscribes to a WS ticker,
                         feeds price data through a user-supplied agent,
                         and submits orders via ccxt.pro.

Action space (agent output):
    0  → Hold  (no order)
    1  → Buy   (long entry / close short)
    2  → Sell  (short entry / close long)   [also accepts -1 for short]

Latency measurement:
    time.perf_counter() timestamps signal_start → order_ack.
    If elapsed > config.max_latency_ms a WARNING is emitted.

Paper trading:
    When paper_trading=True the engine logs the would-be order
    but never calls exchange.create_order().

Reconnection:
    Exponential back-off starting at reconnect_delay (s),
    capped at 5 × reconnect_delay.

State construction (_get_state):
    Returns np.ndarray([price, volume, price_change_pct]).
    Replace/subclass this method to plug in a real FeatureEngine.

Usage:
    import asyncio
    from src.execution.live_trader import LiveTrader, LiveTraderConfig

    config = LiveTraderConfig(
        exchange_id="binance",
        symbol="BTC/USDT",
        api_key="...",
        api_secret="...",
        paper_trading=True,
        order_size_usd=100.0,
    )

    class DummyAgent:
        def predict(self, state):
            return 0  # always hold

    trader = LiveTrader(config=config, agent=DummyAgent())
    asyncio.run(trader.start())
"""

from __future__ import annotations

import asyncio
import collections
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

import numpy as np
from loguru import logger

# ── ccxt.pro import with graceful fallback ──────────────────────────────────
try:
    import ccxt.pro as ccxtpro  # type: ignore

    _CCXT_PRO = True
    logger.debug("ccxt.pro loaded – WebSocket streaming available.")
except ImportError:  # pragma: no cover
    import ccxt as ccxtpro  # type: ignore  # noqa: F401

    _CCXT_PRO = False
    logger.warning(
        "ccxt.pro not available, falling back to ccxt (no WebSocket). "
        "Install 'ccxt[pro]' for low-latency streaming."
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Configuration
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class LiveTraderConfig:
    """
    Immutable configuration for LiveTrader.

    Attributes:
        exchange_id:     ccxt exchange id, e.g. "binance", "bybit", "kucoin".
        symbol:          Market symbol in ccxt format, e.g. "BTC/USDT".
        api_key:         Exchange API key (leave empty for public-only / paper).
        api_secret:      Exchange API secret.
        paper_trading:   When True, orders are logged but NOT sent to exchange.
        order_size_usd:  Notional USD value per order (used to size qty).
        max_latency_ms:  Latency threshold in ms; a WARNING is logged if exceeded.
        reconnect_delay: Base delay (seconds) before reconnect; doubles on failure,
                         capped at 5 × reconnect_delay.
    """

    exchange_id: str = "binance"
    symbol: str = "BTC/USDT"
    api_key: str = ""
    api_secret: str = ""
    paper_trading: bool = True
    order_size_usd: float = 100.0
    max_latency_ms: float = 100.0
    reconnect_delay: float = 5.0


# ─────────────────────────────────────────────────────────────────────────────
#  LiveTrader
# ─────────────────────────────────────────────────────────────────────────────


class LiveTrader:
    """
    Async WebSocket-based live trader powered by ccxt.pro.

    Lifecycle:
        await trader.start()   – connects, subscribes, runs until stop()
        await trader.stop()    – graceful shutdown (cancels tasks, closes WS)

    Execution flow per tick:
        WS ticker  →  _process_ticker()
                        ├─ _get_state()        build np.ndarray for agent
                        ├─ agent.predict()     get action (0/1/2 or -1)
                        └─ _execute_action()   submit / log order
                            └─ _place_order()  ccxt create_order (live only)

    Latency is measured from the moment a ticker arrives until the
    order acknowledgement is received from the exchange.
    """

    # Action constants
    HOLD: int = 0
    BUY: int = 1
    SELL: int = 2  # also accepts -1

    def __init__(
        self,
        config: LiveTraderConfig,
        agent: Any,
        feature_fn: Optional[Callable[[Dict[str, Any]], np.ndarray]] = None,
    ) -> None:
        """
        Args:
            config:     LiveTraderConfig instance.
            agent:      Any object with a ``predict(state: np.ndarray) -> int``
                        method.  Expected to return 0 (hold), 1 (buy), or
                        2 / -1 (sell/short).
            feature_fn: Optional callable ``f(ticker) -> np.ndarray`` that
                        replaces the built-in _get_state() placeholder.
                        Signature: takes the raw ccxt ticker dict and returns
                        a 1-D numpy array ready for agent.predict().
        """
        self._cfg = config
        self._agent = agent
        self._feature_fn = feature_fn  # Optional override for state construction

        # Exchange handle – created lazily in start() so the constructor is sync
        self._exchange: Optional[Any] = None

        # Internal state
        self._running: bool = False
        self._prev_price: Optional[float] = None  # For price_change_pct
        self._tick_count: int = 0
        self._order_count: int = 0

        # Latency statistics (perf_counter seconds)
        self._latency_samples: collections.deque = collections.deque(maxlen=1000)

        mode_tag = "[PAPER]" if config.paper_trading else "[LIVE]"
        logger.info(
            "{} LiveTrader initialised | exchange={} symbol={} order_size=${:.2f}",
            mode_tag,
            config.exchange_id,
            config.symbol,
            config.order_size_usd,
        )

    # ── Public API ───────────────────────────────────────────────────────────

    async def start(self) -> None:
        """
        Main entry point.  Creates the exchange connection and enters the
        WebSocket subscription loop with exponential-backoff reconnection.
        """
        self._running = True
        self._exchange = self._build_exchange()

        if self._cfg.paper_trading:
            logger.info("[PAPER] Paper-trading mode active – no real orders placed.")

        delay = self._cfg.reconnect_delay
        attempt = 0

        while self._running:
            try:
                attempt += 1
                logger.info(
                    "Connecting to {} (attempt {})…", self._cfg.exchange_id, attempt
                )
                await self._subscribe_and_trade()

                # If _subscribe_and_trade() exits cleanly (e.g. stop() called)
                # we break out of the reconnect loop.
                if not self._running:
                    break

                # Unexpected clean exit → reconnect
                logger.warning("WS loop exited unexpectedly, reconnecting…")

            except asyncio.CancelledError:
                logger.info("LiveTrader task cancelled.")
                break

            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "WebSocket error (attempt {}): {} – reconnecting in {:.1f}s",
                    attempt,
                    exc,
                    delay,
                    exc_info=True,
                )

            if self._running:
                await asyncio.sleep(delay)
                # Exponential back-off, capped at 5× base delay
                delay = min(delay * 2, self._cfg.reconnect_delay * 5)

        await self._close_exchange()
        self._print_session_summary()

    async def stop(self) -> None:
        """
        Signal the trader to shut down gracefully.
        The current ticker iteration finishes, then the loop exits.
        """
        logger.warning("LiveTrader stop() called – shutting down.")
        self._running = False
        await self._close_exchange()

    # ── WebSocket Subscription Loop ──────────────────────────────────────────

    async def _subscribe_and_trade(self) -> None:
        """
        Subscribes to the exchange ticker stream via ccxt.pro watchTicker()
        and drives the trading loop until _running is False or an exception
        is raised.

        ccxt.pro's ``watchTicker()`` blocks until a new ticker update arrives,
        then returns the full ticker dict.  We call it in a tight loop so each
        new update is processed sequentially with minimal overhead.
        """
        logger.info(
            "Subscribing to {} ticker on {}…",
            self._cfg.symbol,
            self._cfg.exchange_id,
        )

        while self._running:
            # watchTicker blocks until the next WS update arrives.
            # Falls back to fetch_ticker() if ccxt.pro is unavailable.
            if _CCXT_PRO and hasattr(self._exchange, "watchTicker"):
                ticker: Dict[str, Any] = await self._exchange.watchTicker(
                    self._cfg.symbol
                )
            else:
                # Fallback: polling via REST (no true WebSocket)
                ticker = await self._exchange.fetch_ticker(self._cfg.symbol)
                await asyncio.sleep(1.0)  # Throttle REST polling to 1 Hz

            await self._process_ticker(ticker)

    # ── Tick Processing ──────────────────────────────────────────────────────

    async def _process_ticker(self, ticker: Dict[str, Any]) -> None:
        """
        Called for every new ticker update from the WebSocket.

        1. Extracts price from ticker.
        2. Builds state array via _get_state().
        3. Asks the agent for an action.
        4. Executes the action if it is not HOLD.
        5. Logs latency if it exceeds max_latency_ms.

        Args:
            ticker: Raw ccxt ticker dict with keys such as
                    'last', 'bid', 'ask', 'baseVolume', 'timestamp', …
        """
        signal_start = time.perf_counter()
        self._tick_count += 1

        price: Optional[float] = ticker.get("last") or ticker.get("close")
        if price is None:
            logger.debug(
                "Ticker has no 'last' price – skipping tick {}.", self._tick_count
            )
            return

        price = float(price)

        # Build state for the agent
        state = await self._get_state(ticker)

        # Agent inference
        try:
            action: int = int(self._agent.predict(state))
        except Exception as exc:  # noqa: BLE001
            logger.error("Agent predict() raised: {}", exc, exc_info=True)
            self._prev_price = price
            return

        # Normalise -1 → SELL
        if action == -1:
            action = self.SELL

        logger.debug(
            "tick={} price={:.4f} action={}",
            self._tick_count,
            price,
            {0: "HOLD", 1: "BUY", 2: "SELL"}.get(action, str(action)),
        )

        # Execute (skips on HOLD)
        await self._execute_action(action, price)

        # ── Latency measurement ───────────────────────────────────────────
        elapsed_ms = (time.perf_counter() - signal_start) * 1_000
        self._latency_samples.append(elapsed_ms)

        if elapsed_ms > self._cfg.max_latency_ms:
            logger.warning(
                "HIGH LATENCY | signal_to_order={:.2f} ms > threshold={:.2f} ms "
                "| tick={} price={:.4f} action={}",
                elapsed_ms,
                self._cfg.max_latency_ms,
                self._tick_count,
                price,
                action,
            )
        else:
            logger.debug("Latency OK: {:.2f} ms", elapsed_ms)

        self._prev_price = price

    # ── State Construction ───────────────────────────────────────────────────

    async def _get_state(self, ticker: Dict[str, Any]) -> np.ndarray:
        """
        Convert a raw ccxt ticker dict into a 1-D numpy state array.

        Default placeholder returns [price, volume, price_change_pct].
        Override or inject ``feature_fn`` for real feature engineering.

        Args:
            ticker: ccxt ticker dict.

        Returns:
            np.ndarray of shape (3,) with dtype float32.
        """
        if self._feature_fn is not None:
            # User-supplied feature function takes precedence
            return np.asarray(self._feature_fn(ticker), dtype=np.float32)

        price: float = float(ticker.get("last") or ticker.get("close") or 0.0)
        volume: float = float(ticker.get("baseVolume") or 0.0)

        if self._prev_price and self._prev_price != 0.0:
            price_change_pct = (price - self._prev_price) / self._prev_price
        else:
            price_change_pct = 0.0

        return np.array([price, volume, price_change_pct], dtype=np.float32)

    # ── Order Execution ──────────────────────────────────────────────────────

    async def _execute_action(self, action: int, price: float) -> None:
        """
        Translate a discrete action to an order and submit it.

        Args:
            action: 0=Hold, 1=Buy, 2=Sell (−1 is pre-normalised to 2).
            price:  Current market price (used to size the order).
        """
        if action == self.HOLD:
            return  # Nothing to do

        side = "buy" if action == self.BUY else "sell"
        await self._place_order(side=side, price=price)

    async def _place_order(self, side: str, price: float) -> Optional[Dict[str, Any]]:
        """
        Submit a market order to the exchange (or log it in paper mode).

        Quantity is sized so the notional ≈ order_size_usd.
        Uses a market order for minimal latency.

        Args:
            side:  "buy" or "sell".
            price: Reference price used to calculate quantity.

        Returns:
            The ccxt order dict if live, or a simulated dict if paper.
            Returns None if price is zero or on error.
        """
        if price <= 0.0:
            logger.error(
                "_place_order called with non-positive price={:.6f} – skipping.", price
            )
            return None

        # Size calculation: qty = notional / price
        qty = round(self._cfg.order_size_usd / price, 6)

        if self._cfg.paper_trading:
            simulated_order: Dict[str, Any] = {
                "id": f"paper_{self._order_count + 1}",
                "symbol": self._cfg.symbol,
                "side": side,
                "type": "market",
                "amount": qty,
                "price": price,
                "notional_usd": qty * price,
                "status": "simulated",
                "timestamp": time.time(),
            }
            self._order_count += 1
            logger.info(
                "[PAPER] Would place {} market order | symbol={} qty={:.6f} "
                "price={:.4f} notional=${:.2f} | order_id={}",
                side.upper(),
                self._cfg.symbol,
                qty,
                price,
                qty * price,
                simulated_order["id"],
            )
            return simulated_order

        # ── Live order ────────────────────────────────────────────────────
        try:
            order = await self._exchange.create_order(
                symbol=self._cfg.symbol,
                type="market",
                side=side,
                amount=qty,
            )
            self._order_count += 1
            logger.info(
                "[LIVE] Order submitted | id={} side={} qty={:.6f} price={:.4f}",
                order.get("id", "?"),
                side.upper(),
                qty,
                price,
            )
            return order

        except Exception as exc:  # noqa: BLE001
            logger.error(
                "create_order failed | side={} qty={:.6f} price={:.4f} | {}",
                side,
                qty,
                price,
                exc,
                exc_info=True,
            )
            return None

    # ── Internal Helpers ─────────────────────────────────────────────────────

    def _build_exchange(self) -> Any:
        """
        Instantiate the ccxt.pro exchange object from config.

        Credentials are only attached when api_key is non-empty.
        Sandbox / testnet mode is NOT enabled automatically; users should
        set config.exchange_id to the sandbox exchange id if needed.
        """
        exchange_class = getattr(ccxtpro, self._cfg.exchange_id, None)
        if exchange_class is None:
            raise ValueError(
                f"Exchange '{self._cfg.exchange_id}' not found in ccxt"
                f"{'pro' if _CCXT_PRO else ''}.  "
                f"Available: {list(ccxtpro.exchanges)[:10]} …"
            )

        params: Dict[str, Any] = {
            "enableRateLimit": True,  # Respect exchange rate limits
        }

        if self._cfg.api_key:
            params["apiKey"] = self._cfg.api_key
        if self._cfg.api_secret:
            params["secret"] = self._cfg.api_secret

        exchange = exchange_class(params)
        logger.debug("Exchange instance created: {}", self._cfg.exchange_id)
        return exchange

    async def _close_exchange(self) -> None:
        """Gracefully close the exchange WS connection."""
        if self._exchange is not None:
            try:
                if hasattr(self._exchange, "close"):
                    await self._exchange.close()
                    logger.debug("Exchange connection closed.")
            except Exception as exc:  # noqa: BLE001
                logger.warning("Error closing exchange connection: {}", exc)
            finally:
                self._exchange = None

    def _print_session_summary(self) -> None:
        """Log a human-readable session summary."""
        if self._latency_samples:
            avg_ms = float(np.mean(self._latency_samples))
            p95_ms = float(np.percentile(self._latency_samples, 95))
            max_ms = float(np.max(self._latency_samples))
        else:
            avg_ms = p95_ms = max_ms = 0.0

        mode = "PAPER" if self._cfg.paper_trading else "LIVE"

        logger.info(
            "\n╔══════════════════════════════════════════╗\n"
            "║      LIVE TRADER – SESSION SUMMARY       ║\n"
            "╠══════════════════════════════════════════╣\n"
            "║  Mode             : {:<20}║\n"
            "║  Exchange         : {:<20}║\n"
            "║  Symbol           : {:<20}║\n"
            "║  Ticks processed  : {:<20}║\n"
            "║  Orders placed    : {:<20}║\n"
            "║  Avg latency (ms) : {:<20.2f}║\n"
            "║  P95 latency (ms) : {:<20.2f}║\n"
            "║  Max latency (ms) : {:<20.2f}║\n"
            "╚══════════════════════════════════════════╝",
            mode,
            self._cfg.exchange_id,
            self._cfg.symbol,
            self._tick_count,
            self._order_count,
            avg_ms,
            p95_ms,
            max_ms,
        )

    # ── Properties (read-only diagnostics) ──────────────────────────────────

    @property
    def tick_count(self) -> int:
        """Total number of ticker updates processed."""
        return self._tick_count

    @property
    def order_count(self) -> int:
        """Total number of orders placed (or simulated in paper mode)."""
        return self._order_count

    @property
    def is_running(self) -> bool:
        """True while the trading loop is active."""
        return self._running

    @property
    def avg_latency_ms(self) -> float:
        """Mean signal-to-order latency in milliseconds (0 if no data yet)."""
        return float(np.mean(self._latency_samples)) if self._latency_samples else 0.0
