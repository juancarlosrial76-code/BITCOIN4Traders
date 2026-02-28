"""
PAPER TRADING – SIMULATED ORDER MANAGER
=========================================
Drop-in Ersatz für OrderManager im dry_run Modus.
Kein echter Binance REST-Call – alle Orders werden lokal simuliert.

Fill-Logik:
  - MARKET  → sofort zum aktuellen mid-price gefüllt
  - LIMIT   → sofort gefüllt (konservativ: zum Limit-Preis)
  - Kommission: 0.1 % (Binance Standard)
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Callable, Dict, List, Optional, Tuple
from uuid import uuid4

from src.orders.order_manager import (
    Fill,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    TimeInForce,
)

logger = logging.getLogger("phase7.paper_oms")

COMMISSION_RATE = Decimal("0.001")  # 0.1 %


class PaperOrderManager:
    """
    Simulierter OMS für Paper Trading.
    Identische öffentliche API wie OrderManager.
    """

    def __init__(self, api_key: str, api_secret: str, initial_cash: Decimal = Decimal(10_000)):
        self._session = None  # nicht benötigt
        self._orders: Dict[str, Order] = {}
        self._exchange_id_map: Dict[int, str] = {}
        self._on_fill_cbs: List[Callable[[Order, Fill], None]] = []
        self._on_status_cbs: List[Callable[[Order, OrderStatus], None]] = []
        self._lock = asyncio.Lock()
        self._next_eid = 1

        # Paper-Trading Kontostand
        self.cash: Decimal = initial_cash
        self.trade_log: List[dict] = []

        logger.info(
            "[PAPER] PaperOrderManager initialisiert – Startkapital: $%.2f", float(initial_cash)
        )

    # ── Lifecycle ───────────────────────────

    async def start(self) -> None:
        logger.info("[PAPER] OrderManager gestartet (Simulation).")

    async def stop(self) -> None:
        logger.info("[PAPER] OrderManager gestoppt.")
        self._print_trade_log()

    # ── Callbacks ───────────────────────────

    def on_fill(self, cb: Callable[[Order, Fill], None]) -> None:
        self._on_fill_cbs.append(cb)

    def on_status_change(self, cb: Callable[[Order, OrderStatus], None]) -> None:
        self._on_status_cbs.append(cb)

    # ── Public OMS API ───────────────────────

    async def submit_order(self, order: Order) -> Order:
        async with self._lock:
            self._orders[order.client_order_id] = order

            # Simuliere Exchange-ACK
            eid = self._next_eid
            self._next_eid += 1
            order.exchange_order_id = eid
            self._exchange_id_map[eid] = order.client_order_id
            order.transition(OrderStatus.NEW, note=f"[PAPER] exchangeOrderId={eid}")

            # Fill-Preis bestimmen
            fill_price = order.limit_price if order.limit_price else Decimal(0)
            if fill_price == 0:
                logger.warning(
                    "[PAPER] Kein Preis für %s – überspringe Fill.", order.client_order_id
                )
                return order

            # Kommission berechnen
            commission = fill_price * order.quantity * COMMISSION_RATE

            fill = Fill(
                price=fill_price,
                qty=order.quantity,
                commission=commission,
                commission_asset="USDT",
                trade_id=eid,
                timestamp_ms=int(time.time() * 1000),
            )
            order.add_fill(fill)

            # Kontostand aktualisieren
            notional = fill_price * order.quantity
            if order.side == OrderSide.BUY:
                self.cash -= notional + commission
            else:
                self.cash += notional - commission

            # Trade-Log Eintrag
            self.trade_log.append(
                {
                    "time": time.strftime("%H:%M:%S"),
                    "symbol": order.symbol,
                    "side": order.side.value,
                    "qty": float(order.quantity),
                    "price": float(fill_price),
                    "notional": float(notional),
                    "commission": float(commission),
                    "cash_after": float(self.cash),
                }
            )

            order.transition(OrderStatus.FILLED, note="[PAPER] Sofort gefüllt")

            logger.info(
                "[PAPER] FILL: %s %s %.6f %s @ $%.4f | Kommission=$%.4f | Kasse=$%.2f",
                order.symbol,
                order.side.value,
                float(order.quantity),
                order.symbol,
                float(fill_price),
                float(commission),
                float(self.cash),
            )

        # Callbacks außerhalb des Locks aufrufen
        for cb in self._on_fill_cbs:
            try:
                cb(order, fill)
            except Exception as exc:
                logger.error("[PAPER] on_fill callback Fehler: %s", exc)

        for cb in self._on_status_cbs:
            try:
                cb(order, OrderStatus.FILLED)
            except Exception as exc:
                logger.error("[PAPER] on_status callback Fehler: %s", exc)

        return order

    async def cancel_order(self, client_order_id: str) -> Order:
        async with self._lock:
            order = self._orders.get(client_order_id)
            if order is None:
                raise KeyError(f"Unbekannte Order: {client_order_id}")
            if order.is_terminal:
                return order
            order.transition(OrderStatus.CANCELED, note="[PAPER] Cancel simuliert")
        logger.info("[PAPER] Order storniert: %s", client_order_id)
        return order

    async def cancel_all(self, symbol: str) -> List[Order]:
        canceled = []
        async with self._lock:
            for order in self._orders.values():
                if order.symbol == symbol and not order.is_terminal:
                    order.transition(OrderStatus.CANCELED, note="[PAPER] cancel_all")
                    canceled.append(order)
        logger.info("[PAPER] cancel_all(%s): %d Orders storniert.", symbol, len(canceled))
        return canceled

    async def get_open_orders(self, symbol: Optional[str] = None) -> List[Order]:
        async with self._lock:
            return [
                o
                for o in self._orders.values()
                if not o.is_terminal and (symbol is None or o.symbol == symbol)
            ]

    def get_order(self, client_order_id: str) -> Optional[Order]:
        return self._orders.get(client_order_id)

    def get_all_orders(self) -> List[Order]:
        return list(self._orders.values())

    # ── WS Handler (No-op im Paper Mode) ────

    async def handle_execution_report(self, data: dict) -> None:
        """Im Paper Mode werden keine echten WS-Execution-Reports empfangen."""
        pass

    # ── Hilfsmethoden ───────────────────────

    def _print_trade_log(self) -> None:
        if not self.trade_log:
            logger.info("[PAPER] Keine Trades ausgeführt.")
            return

        logger.info("\n[PAPER] ══ TRADE LOG ════════════════════════════")
        logger.info(
            "  %-8s %-8s %-5s %-10s %-12s %-12s %-10s",
            "Zeit",
            "Symbol",
            "Seite",
            "Menge",
            "Preis",
            "Notional",
            "Kasse danach",
        )
        for t in self.trade_log:
            logger.info(
                "  %-8s %-8s %-5s %-10.5f %-12.2f %-12.2f %-10.2f",
                t["time"],
                t["symbol"],
                t["side"],
                t["qty"],
                t["price"],
                t["notional"],
                t["cash_after"],
            )
        logger.info("[PAPER] ════════════════════════════════════════")
