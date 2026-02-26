"""
PHASE 7 – ORDER MANAGEMENT SYSTEM (OMS)
=========================================
Responsibilities:
  - Maintain canonical order state (lifecycle FSM)
  - Submit / cancel / amend orders via Binance REST
  - Reconcile REST state with WebSocket execution reports
  - Track fills, partial fills, slippage vs. expected price
  - Thread-safe order book with event callbacks
  - Full audit log per order
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import logging
import time
import urllib.parse
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum, auto
from typing import Callable, Dict, List, Optional, Tuple
from uuid import uuid4

import aiohttp

from src.connectors.binance_ws_connector import (
    AuthenticationError,
    ConnectorError,
    RateLimitError,
)

logger = logging.getLogger("phase7.oms")


# ─────────────────────────────────────────────
#  Order Lifecycle FSM
# ─────────────────────────────────────────────


class OrderStatus(Enum):
    PENDING_NEW = auto()  # Created locally, not yet sent
    NEW = auto()  # ACK received from exchange
    PARTIALLY_FILLED = auto()
    FILLED = auto()
    PENDING_CANCEL = auto()
    CANCELED = auto()
    REJECTED = auto()
    EXPIRED = auto()


# Valid transitions
_TRANSITIONS: Dict[OrderStatus, List[OrderStatus]] = {
    OrderStatus.PENDING_NEW: [OrderStatus.NEW, OrderStatus.REJECTED],
    OrderStatus.NEW: [
        OrderStatus.PARTIALLY_FILLED,
        OrderStatus.FILLED,
        OrderStatus.PENDING_CANCEL,
        OrderStatus.CANCELED,
        OrderStatus.EXPIRED,
    ],
    OrderStatus.PARTIALLY_FILLED: [
        OrderStatus.FILLED,
        OrderStatus.PENDING_CANCEL,
        OrderStatus.CANCELED,
    ],
    OrderStatus.PENDING_CANCEL: [OrderStatus.CANCELED, OrderStatus.FILLED],
    # Terminal states – no transitions
    OrderStatus.FILLED: [],
    OrderStatus.CANCELED: [],
    OrderStatus.REJECTED: [],
    OrderStatus.EXPIRED: [],
}


class OrderSide(Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderType(Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    LIMIT_MAKER = "LIMIT_MAKER"
    STOP_LOSS_LIMIT = "STOP_LOSS_LIMIT"


class TimeInForce(Enum):
    GTC = "GTC"  # Good Till Cancel
    IOC = "IOC"  # Immediate Or Cancel
    FOK = "FOK"  # Fill Or Kill


# ─────────────────────────────────────────────
#  Order & Fill Data Classes
# ─────────────────────────────────────────────


@dataclass
class Fill:
    price: Decimal
    qty: Decimal
    commission: Decimal
    commission_asset: str
    trade_id: int
    timestamp_ms: int


@dataclass
class Order:
    # Identity
    client_order_id: str = field(default_factory=lambda: f"p7_{uuid4().hex[:12]}")
    exchange_order_id: Optional[int] = None

    # Instrument
    symbol: str = ""
    side: OrderSide = OrderSide.BUY
    type: OrderType = OrderType.LIMIT
    tif: TimeInForce = TimeInForce.GTC

    # Sizing
    quantity: Decimal = Decimal(0)
    limit_price: Optional[Decimal] = None
    stop_price: Optional[Decimal] = None

    # State
    status: OrderStatus = OrderStatus.PENDING_NEW
    filled_qty: Decimal = Decimal(0)
    avg_fill_price: Decimal = Decimal(0)
    cumulative_quote: Decimal = Decimal(0)

    # Fills
    fills: List[Fill] = field(default_factory=list)

    # Metadata
    created_at_ms: int = field(default_factory=lambda: int(time.time() * 1000))
    updated_at_ms: int = field(default_factory=lambda: int(time.time() * 1000))
    error_msg: Optional[str] = None

    # Audit log
    audit: List[Tuple[int, str]] = field(default_factory=list)

    # ── Computed ────────────────────────────

    @property
    def remaining_qty(self) -> Decimal:
        return self.quantity - self.filled_qty

    @property
    def is_terminal(self) -> bool:
        return self.status in (
            OrderStatus.FILLED,
            OrderStatus.CANCELED,
            OrderStatus.REJECTED,
            OrderStatus.EXPIRED,
        )

    @property
    def slippage_bps(self) -> Optional[Decimal]:
        """Slippage vs. limit price in basis points (limit orders only)."""
        if not self.limit_price or self.avg_fill_price == 0:
            return None
        # Raw price deviation: positive = filled above limit (bad for buys)
        diff = (self.avg_fill_price - self.limit_price) / self.limit_price
        if self.side == OrderSide.SELL:
            diff = (
                -diff
            )  # For sells: above limit is good; flip sign for consistent convention
        return diff * Decimal(10_000)  # Convert to basis points (1 bps = 0.01%)

    # ── FSM ─────────────────────────────────

    def transition(self, new_status: OrderStatus, note: str = "") -> None:
        allowed = _TRANSITIONS.get(self.status, [])
        if new_status not in allowed:
            raise ValueError(
                f"Illegal transition {self.status} → {new_status} for order {self.client_order_id}"
            )
        ts = int(time.time() * 1000)
        self.audit.append((ts, f"{self.status.name} → {new_status.name}. {note}"))
        self.status = new_status
        self.updated_at_ms = ts

    def add_fill(self, fill: Fill) -> None:
        self.fills.append(fill)
        self.filled_qty += fill.qty  # Accumulate total filled quantity
        self.cumulative_quote += (
            fill.price * fill.qty
        )  # Accumulate total cost (price * qty)
        if self.filled_qty > 0:
            self.avg_fill_price = (
                self.cumulative_quote / self.filled_qty
            )  # VWAP of all fills
        self.audit.append(
            (
                fill.timestamp_ms,
                f"FILL {fill.qty}@{fill.price} (cumQty={self.filled_qty})",
            )
        )
        self.updated_at_ms = fill.timestamp_ms

    def __repr__(self) -> str:
        return (
            f"<Order {self.client_order_id} "
            f"{self.side.value} {self.quantity} {self.symbol} "
            f"@ {self.limit_price or 'MKT'} | {self.status.name}>"
        )


# ─────────────────────────────────────────────
#  Order Manager
# ─────────────────────────────────────────────


class OrderManager:
    """
    Central OMS: submit, track, cancel orders.
    Integrates with BinanceWSConnector user-data events.
    """

    _REST_BASE = "https://api.binance.com"

    def __init__(self, api_key: str, api_secret: str):
        self._api_key = api_key
        self._api_secret = api_secret
        self._session: Optional[aiohttp.ClientSession] = None

        # order_id → Order
        self._orders: Dict[str, Order] = {}
        # exchange_order_id → client_order_id
        self._exchange_id_map: Dict[int, str] = {}

        # Event callbacks
        self._on_fill_cbs: List[Callable[[Order, Fill], None]] = []
        self._on_status_cbs: List[Callable[[Order, OrderStatus], None]] = []

        self._lock = asyncio.Lock()

    # ── Lifecycle ───────────────────────────

    async def start(self) -> None:
        self._session = aiohttp.ClientSession()
        logger.info("OrderManager started.")

    async def stop(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()
        logger.info("OrderManager stopped.")

    # ── Callbacks ───────────────────────────

    def on_fill(self, cb: Callable[[Order, Fill], None]) -> None:
        self._on_fill_cbs.append(cb)

    def on_status_change(self, cb: Callable[[Order, OrderStatus], None]) -> None:
        self._on_status_cbs.append(cb)

    # ── Public OMS API ───────────────────────

    async def submit_order(self, order: Order) -> Order:
        """
        Send order to Binance. Returns the order with exchange_order_id populated.
        """
        async with self._lock:
            self._orders[order.client_order_id] = order

        params = self._build_order_params(order)
        logger.info("Submitting: %s", order)

        try:
            resp = await self._signed_post("/api/v3/order", params)
        except RateLimitError:
            logger.warning(
                "Rate limit hit submitting %s – will retry in 5 s.",
                order.client_order_id,
            )
            await asyncio.sleep(5)
            resp = await self._signed_post("/api/v3/order", params)
        except AuthenticationError as exc:
            async with self._lock:
                order.error_msg = str(exc)
                order.transition(OrderStatus.REJECTED, note=str(exc))
            raise

        async with self._lock:
            order.exchange_order_id = resp["orderId"]
            self._exchange_id_map[resp["orderId"]] = order.client_order_id
            order.transition(OrderStatus.NEW, note=f"exchangeOrderId={resp['orderId']}")

            # Handle immediate fills (MARKET orders, IOC)
            for fill_data in resp.get("fills", []):
                self._apply_fill_from_rest(order, fill_data)

            if resp["status"] == "FILLED":
                order.transition(OrderStatus.FILLED, note="Immediately filled")
            elif resp["status"] == "PARTIALLY_FILLED":
                order.transition(OrderStatus.PARTIALLY_FILLED)

        logger.info("Order ACK: %s", order)
        return order

    async def cancel_order(self, client_order_id: str) -> Order:
        """Cancel an open order."""
        async with self._lock:
            order = self._orders.get(client_order_id)
            if order is None:
                raise KeyError(f"Unknown order: {client_order_id}")
            if order.is_terminal:
                logger.warning("Cancel attempted on terminal order %s", client_order_id)
                return order
            order.transition(OrderStatus.PENDING_CANCEL, note="Cancel requested")

        params = {
            "symbol": order.symbol,
            "origClientOrderId": order.client_order_id,
        }
        try:
            await self._signed_delete("/api/v3/order", params)
        except ConnectorError as exc:
            logger.error("Cancel failed for %s: %s", client_order_id, exc)
            raise

        async with self._lock:
            order.transition(OrderStatus.CANCELED, note="Cancel confirmed by REST")

        logger.info("Order canceled: %s", client_order_id)
        return order

    async def cancel_all(self, symbol: str) -> List[Order]:
        """Cancel all open orders for a symbol."""
        params = {"symbol": symbol}
        await self._signed_delete("/api/v3/openOrders", params)

        canceled = []
        async with self._lock:
            for order in self._orders.values():
                if order.symbol == symbol and not order.is_terminal:
                    order.transition(OrderStatus.CANCELED, note="cancel_all")
                    canceled.append(order)
        logger.info("cancel_all(%s): %d orders canceled.", symbol, len(canceled))
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

    # ── WebSocket Event Handler ──────────────

    async def handle_execution_report(self, data: dict) -> None:
        """
        Called by BinanceWSConnector for 'executionReport' user-data events.
        """
        ex_order_id = data["i"]  # exchange order id
        c_order_id = data.get("c") or self._exchange_id_map.get(ex_order_id)
        exec_type = data["x"]  # NEW, CANCELED, TRADE, EXPIRED, REJECTED
        order_status = data["X"]  # order status string

        async with self._lock:
            order = self._orders.get(c_order_id)
            if order is None:
                logger.warning(
                    "Execution report for unknown order: cid=%s, eid=%s",
                    c_order_id,
                    ex_order_id,
                )
                return

            # Register exchange id mapping if missing
            if ex_order_id not in self._exchange_id_map:
                self._exchange_id_map[ex_order_id] = c_order_id

            prev_status = order.status

            if exec_type == "TRADE":
                fill = Fill(
                    price=Decimal(str(data["L"])),
                    qty=Decimal(str(data["l"])),
                    commission=Decimal(str(data["n"])),
                    commission_asset=data["N"] or "",
                    trade_id=data["t"],
                    timestamp_ms=data["T"],
                )
                order.add_fill(fill)
                for cb in self._on_fill_cbs:
                    try:
                        cb(order, fill)
                    except Exception as exc:
                        logger.error("on_fill callback error: %s", exc)

            new_status = self._map_exchange_status(order_status)
            if new_status != prev_status and new_status in _TRANSITIONS.get(
                prev_status, []
            ):
                order.transition(new_status, note=f"WS exec_type={exec_type}")
                for cb in self._on_status_cbs:
                    try:
                        cb(order, new_status)
                    except Exception as exc:
                        logger.error("on_status callback error: %s", exc)

        logger.debug(
            "Execution report processed: %s → %s", c_order_id, order.status.name
        )

    # ── REST helpers ────────────────────────

    def _build_order_params(self, order: Order) -> dict:
        params: dict = {
            "symbol": order.symbol,
            "side": order.side.value,
            "type": order.type.value,
            "quantity": str(order.quantity),
            "newClientOrderId": order.client_order_id,
            "newOrderRespType": "FULL",
        }
        if order.type != OrderType.MARKET:
            params["timeInForce"] = order.tif.value
        if order.limit_price is not None:
            params["price"] = str(order.limit_price)
        if order.stop_price is not None:
            params["stopPrice"] = str(order.stop_price)
        return params

    def _apply_fill_from_rest(self, order: Order, fill_data: dict) -> None:
        fill = Fill(
            price=Decimal(str(fill_data["price"])),
            qty=Decimal(str(fill_data["qty"])),
            commission=Decimal(str(fill_data["commission"])),
            commission_asset=fill_data["commissionAsset"],
            trade_id=fill_data["tradeId"],
            timestamp_ms=int(time.time() * 1000),
        )
        order.add_fill(fill)

    @staticmethod
    def _map_exchange_status(status: str) -> OrderStatus:
        return {
            "NEW": OrderStatus.NEW,
            "PARTIALLY_FILLED": OrderStatus.PARTIALLY_FILLED,
            "FILLED": OrderStatus.FILLED,
            "CANCELED": OrderStatus.CANCELED,
            "PENDING_CANCEL": OrderStatus.PENDING_CANCEL,
            "REJECTED": OrderStatus.REJECTED,
            "EXPIRED": OrderStatus.EXPIRED,
        }.get(status, OrderStatus.NEW)

    def _sign(self, params: dict) -> str:
        """HMAC-SHA256 sign the query string with the API secret (Binance auth requirement)."""
        query = urllib.parse.urlencode(params)  # Encode params as URL query string
        return hmac.new(
            self._api_secret.encode(), query.encode(), hashlib.sha256
        ).hexdigest()  # Return hex-encoded signature

    async def _signed_post(self, path: str, params: dict) -> dict:
        params["timestamp"] = int(time.time() * 1000)
        params["signature"] = self._sign(params)
        headers = {"X-MBX-APIKEY": self._api_key}
        async with self._session.post(
            f"{self._REST_BASE}{path}", params=params, headers=headers
        ) as resp:
            body = await resp.json()
            if resp.status == 401:
                raise AuthenticationError(body)
            if resp.status in (429, 418):
                raise RateLimitError(body)
            if resp.status >= 400:
                raise ConnectorError(f"REST {resp.status}: {body}")
            return body

    async def _signed_delete(self, path: str, params: dict) -> dict:
        params["timestamp"] = int(time.time() * 1000)
        params["signature"] = self._sign(params)
        headers = {"X-MBX-APIKEY": self._api_key}
        async with self._session.delete(
            f"{self._REST_BASE}{path}", params=params, headers=headers
        ) as resp:
            if resp.status >= 400:
                body = await resp.json()
                raise ConnectorError(f"REST DELETE {resp.status}: {body}")
            return await resp.json()
