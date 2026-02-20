"""
PHASE 7 – MONITORING & ALERTING
=================================
Real-time monitoring with:
  - Telegram bot alerts (fills, errors, circuit breaker)
  - Prometheus metrics exposition (für Grafana)
  - Rolling P&L console dashboard
  - Structured JSON log enrichment
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Dict, List, Optional

import aiohttp

logger = logging.getLogger("phase7.monitoring")


# ─────────────────────────────────────────────
#  Alert Levels
# ─────────────────────────────────────────────

class AlertLevel:
    INFO     = "ℹ️"
    WARNING  = "⚠️"
    CRITICAL = "🚨"
    FILL     = "✅"
    ERROR    = "❌"


# ─────────────────────────────────────────────
#  Telegram Notifier
# ─────────────────────────────────────────────

class TelegramNotifier:
    """
    Sends alerts to a Telegram chat.
    Queues messages and sends asynchronously to avoid blocking the engine.
    """

    _API = "https://api.telegram.org/bot{token}/sendMessage"

    def __init__(self, bot_token: str, chat_id: str):
        self._token   = bot_token
        self._chat_id = chat_id
        self._queue:  asyncio.Queue = asyncio.Queue(maxsize=200)
        self._session: Optional[aiohttp.ClientSession] = None
        self._task:    Optional[asyncio.Task] = None

    async def start(self) -> None:
        self._session = aiohttp.ClientSession()
        self._task    = asyncio.create_task(self._send_loop(), name="telegram_sender")
        logger.info("Telegram notifier started.")

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
        if self._session and not self._session.closed:
            await self._session.close()

    def send(self, level: str, message: str) -> None:
        """Non-blocking enqueue."""
        text = f"{level} *PHASE7*\n{message}"
        try:
            self._queue.put_nowait(text)
        except asyncio.QueueFull:
            logger.warning("Telegram queue full – dropping alert.")

    async def _send_loop(self) -> None:
        while True:
            try:
                text = await self._queue.get()
                await self._send_one(text)
                await asyncio.sleep(0.3)    # Telegram rate limit ~30 msg/s
            except asyncio.CancelledError:
                return
            except Exception as exc:
                logger.error("Telegram send error: %s", exc)

    async def _send_one(self, text: str) -> None:
        url = self._API.format(token=self._token)
        payload = {
            "chat_id":    self._chat_id,
            "text":       text,
            "parse_mode": "Markdown",
        }
        try:
            async with self._session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    logger.warning("Telegram API %d: %s", resp.status, body[:200])
        except Exception as exc:
            logger.error("Telegram HTTP error: %s", exc)


# ─────────────────────────────────────────────
#  Metrics (Prometheus-compatible exposition)
# ─────────────────────────────────────────────

@dataclass
class EngineMetrics:
    ticks_processed:    int     = 0
    orders_submitted:   int     = 0
    orders_filled:      int     = 0
    orders_canceled:    int     = 0
    orders_rejected:    int     = 0
    total_fills:        int     = 0
    realized_pnl_usd:   float   = 0.0
    unrealized_pnl_usd: float   = 0.0
    circuit_breaker_trips: int  = 0
    ws_reconnects:      int     = 0
    last_updated:       float   = field(default_factory=time.time)

    def to_prometheus(self) -> str:
        lines = []
        for k, v in self.__dict__.items():
            if isinstance(v, (int, float)):
                metric_name = f"phase7_{k}"
                lines.append(f"# TYPE {metric_name} gauge")
                lines.append(f"{metric_name} {v}")
        return "\n".join(lines) + "\n"

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


# ─────────────────────────────────────────────
#  Monitor
# ─────────────────────────────────────────────

class EngineMonitor:
    """
    Central monitoring hub.
    Attach to LiveExecutionEngine to get full observability.
    """

    def __init__(
        self,
        telegram: Optional[TelegramNotifier] = None,
        metrics_port: int = 9090,
        dashboard_interval_s: int = 30,
    ):
        self._telegram          = telegram
        self._metrics           = EngineMetrics()
        self._dashboard_interval = dashboard_interval_s
        self._dashboard_task: Optional[asyncio.Task] = None
        self._metrics_task: Optional[asyncio.Task]   = None

    # ── Lifecycle ───────────────────────────

    async def start(self) -> None:
        if self._telegram:
            await self._telegram.start()
        self._dashboard_task = asyncio.create_task(
            self._dashboard_loop(), name="dashboard"
        )
        logger.info("Monitor started.")
        self.alert(AlertLevel.INFO, "Phase 7 Live Engine started ✅")

    async def stop(self) -> None:
        if self._dashboard_task:
            self._dashboard_task.cancel()
        if self._telegram:
            await self._telegram.stop()

    # ── Event Hooks (call from engine) ──────

    def on_tick(self) -> None:
        self._metrics.ticks_processed += 1

    def on_order_submitted(self) -> None:
        self._metrics.orders_submitted += 1

    def on_fill(self, qty: Decimal, price: Decimal, pnl: Decimal) -> None:
        self._metrics.total_fills += 1
        self._metrics.realized_pnl_usd += float(pnl)
        self.alert(
            AlertLevel.FILL,
            f"Fill: qty={qty} @{price} | realized PnL=${float(pnl):+.2f}"
        )

    def on_order_canceled(self, reason: str = "") -> None:
        self._metrics.orders_canceled += 1

    def on_order_rejected(self, reason: str) -> None:
        self._metrics.orders_rejected += 1
        self.alert(AlertLevel.ERROR, f"Order REJECTED: {reason}")

    def on_circuit_breaker(self, reason: str) -> None:
        self._metrics.circuit_breaker_trips += 1
        self.alert(AlertLevel.CRITICAL, f"CIRCUIT BREAKER TRIPPED\n{reason}")

    def on_reconnect(self, attempt: int) -> None:
        self._metrics.ws_reconnects += 1
        self.alert(AlertLevel.WARNING, f"WS Reconnect attempt #{attempt}")

    def on_error(self, msg: str) -> None:
        self.alert(AlertLevel.ERROR, msg)

    def update_pnl(self, realized: float, unrealized: float) -> None:
        self._metrics.realized_pnl_usd   = realized
        self._metrics.unrealized_pnl_usd = unrealized
        self._metrics.last_updated = time.time()

    # ── Alerting ─────────────────────────────

    def alert(self, level: str, message: str) -> None:
        log_fn = logger.info if level in (AlertLevel.INFO, AlertLevel.FILL) else logger.warning
        log_fn("[ALERT %s] %s", level, message)
        if self._telegram:
            self._telegram.send(level, message)

    # ── Dashboard ────────────────────────────

    async def _dashboard_loop(self) -> None:
        while True:
            try:
                await asyncio.sleep(self._dashboard_interval)
                self._print_dashboard()
            except asyncio.CancelledError:
                return
            except Exception as exc:
                logger.error("Dashboard error: %s", exc)

    def _print_dashboard(self) -> None:
        m = self._metrics
        total_pnl = m.realized_pnl_usd + m.unrealized_pnl_usd
        pnl_sign  = "+" if total_pnl >= 0 else ""
        logger.info(
            "\n┌─────────────────────────────────────────┐\n"
            "│         PHASE 7  │  LIVE DASHBOARD       │\n"
            "├─────────────────────────────────────────┤\n"
            "│ Ticks:    %-8d  Orders:  %-10d │\n"
            "│ Fills:    %-8d  Cancels: %-10d │\n"
            "│ Realized: $%-9.2f Unreal: $%-9.2f│\n"
            "│ Total PnL: %s$%-9.2f  CB Trips: %-6d│\n"
            "│ WS Reconnects: %-6d                    │\n"
            "└─────────────────────────────────────────┘",
            m.ticks_processed, m.orders_submitted,
            m.total_fills,     m.orders_canceled,
            m.realized_pnl_usd, m.unrealized_pnl_usd,
            pnl_sign, total_pnl, m.circuit_breaker_trips,
            m.ws_reconnects,
        )

    @property
    def metrics(self) -> EngineMetrics:
        return self._metrics
