"""
Paper Trading Order Manager - Simulated Order Execution
========================================================
Drop-in replacement for OrderManager for dry-run and backtesting.

This module provides a simulated order management system that mimics the
real OrderManager API but executes orders locally without connecting to
the Binance exchange. It is designed for:
    - Paper trading (trading with fake money)
    - Backtesting strategies with realistic fill simulation
    - Development and testing without risking real capital
    - Strategy validation before live deployment

Key Features:
    - Identical API to OrderManager for seamless switching
    - Simulated order fills at realistic prices
    - Account balance tracking with virtual cash
    - Configurable commission rates (default: 0.1% = Binance taker fee)
    - Trade logging for performance analysis
    - No network dependencies - runs completely offline

Fill Simulation Logic:
    - MARKET orders: Filled immediately at current price (requires price data)
    - LIMIT orders: Filled immediately at limit price (conservative assumption)
    - The fill price must be provided or available - MARKET orders without
      price data will not be filled and will return early with a warning

Commission Structure:
    - Default: 0.1% (matching Binance standard taker fee)
    - Applied to both BUY and SELL orders
    - Deducted from cash for BUY orders
    - Subtracted from proceeds for SELL orders

Usage:
    from src.orders.paper_order_manager import PaperOrderManager
    from src.orders.order_manager import Order, OrderSide, OrderType

    # Initialize with virtual capital
    pom = PaperOrderManager(
        api_key="dummy",
        api_secret="dummy",
        initial_cash=Decimal("10000")  # $10,000 starting capital
    )
    await pom.start()

    # Submit orders just like with real OrderManager
    order = Order(
        symbol="BTCUSDT",
        side=OrderSide.BUY,
        type=OrderType.MARKET,
        quantity=Decimal("0.001")
    )
    await pom.submit_order(order)

    # Check virtual cash balance
    print(f"Remaining cash: ${pom.cash}")

    await pom.stop()  # Prints trade log summary

Architecture:
    PaperOrderManager maintains identical internal data structures to OrderManager
    for maximum compatibility. The key difference is that submit_order() simulates
    exchange behavior locally rather than making API calls.

Warning:
    Paper trading results do not guarantee similar live trading performance.
    Market conditions, slippage, and liquidity may differ significantly between
    simulated and live execution.
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

COMMISSION_RATE = Decimal("0.001")  # 0.1% Binance taker fee


class PaperOrderManager:
    """
    Simulated Order Manager for paper trading and testing.

    This class provides a complete drop-in replacement for OrderManager that
    simulates order execution locally. It maintains identical public API and
    data structures, allowing seamless switching between paper and live trading.

    Attributes:
        cash: Virtual cash balance in quote currency (e.g., USDT)
        trade_log: List of executed trade records with timestamps
        _orders: Dict mapping client_order_id -> Order objects
        _exchange_id_map: Dict mapping simulated exchange IDs -> client IDs
        _on_fill_cbs: List of fill event callbacks
        _on_status_cbs: List of status change callbacks

    Account Balance:
        - Initial cash defaults to $10,000 USDT
        - BUY orders deduct (price * quantity + commission) from cash
        - SELL orders add (price * quantity - commission) to cash
        - Cash balance updates immediately upon order fill

    Trade Log:
        Each executed trade records:
        - Time, symbol, side, quantity, price
        - Notional value, commission paid
        - Cash balance after trade

    Example:
        >>> from decimal import Decimal
        >>> pom = PaperOrderManager(
        ...     api_key="test",
        ...     api_secret="test",
        ...     initial_cash=Decimal("50000")
        ... )
        >>> import asyncio
        >>> asyncio.run(pom.start())
        >>> # ... submit orders ...
        >>> asyncio.run(pom.stop())
        [PAPER] ══ TRADE LOG ════════════════════════════
        ...

    Note:
        PaperOrderManager does NOT require price data for LIMIT orders - it will
        fill at the specified limit price. For MARKET orders, you must ensure
        price data is available or provide it via the order's limit_price field.
    """

    def __init__(
        self, api_key: str, api_secret: str, initial_cash: Decimal = Decimal(10_000)
    ):
        self._session = None  # Not needed for paper trading
        self._orders: Dict[str, Order] = {}
        self._exchange_id_map: Dict[int, str] = {}
        self._on_fill_cbs: List[Callable[[Order, Fill], None]] = []
        self._on_status_cbs: List[Callable[[Order, OrderStatus], None]] = []
        self._lock = asyncio.Lock()
        self._next_eid = 1  # Auto-incrementing simulated exchange order ID

        # Paper trading account balance
        self.cash: Decimal = initial_cash
        self.trade_log: List[dict] = []

        logger.info(
            "[PAPER] PaperOrderManager initialized – starting capital: $%.2f",
            float(initial_cash),
        )

    # ── Lifecycle ───────────────────────────

    async def start(self) -> None:
        logger.info("[PAPER] OrderManager started (simulation).")

    async def stop(self) -> None:
        logger.info("[PAPER] OrderManager stopped.")
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

            # Simulate exchange ACK
            eid = self._next_eid
            self._next_eid += 1
            order.exchange_order_id = eid
            self._exchange_id_map[eid] = order.client_order_id
            order.transition(OrderStatus.NEW, note=f"[PAPER] exchangeOrderId={eid}")

            # Determine fill price: use limit price if set, else 0 (market needs mid)
            fill_price = order.limit_price if order.limit_price else Decimal(0)
            if fill_price == 0:
                logger.warning(
                    "[PAPER] No price for %s – skipping fill.", order.client_order_id
                )
                return order

            # Calculate commission: price * qty * rate
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

            # Update account balance
            notional = (
                fill_price * order.quantity
            )  # Total trade value in quote currency
            if order.side == OrderSide.BUY:
                self.cash -= notional + commission  # Deduct cost + fees for buy
            else:
                self.cash += notional - commission  # Add proceeds minus fees for sell

            # Trade log entry
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

            order.transition(OrderStatus.FILLED, note="[PAPER] Immediately filled")

            logger.info(
                "[PAPER] FILL: %s %s %.6f %s @ $%.4f | Commission=$%.4f | Cash=$%.2f",
                order.symbol,
                order.side.value,
                float(order.quantity),
                order.symbol,
                float(fill_price),
                float(commission),
                float(self.cash),
            )

        # Invoke callbacks outside of the lock to avoid deadlock
        for cb in self._on_fill_cbs:
            try:
                cb(order, fill)
            except Exception as exc:
                logger.error("[PAPER] on_fill callback error: %s", exc)

        for cb in self._on_status_cbs:
            try:
                cb(order, OrderStatus.FILLED)
            except Exception as exc:
                logger.error("[PAPER] on_status callback error: %s", exc)

        return order

    async def cancel_order(self, client_order_id: str) -> Order:
        async with self._lock:
            order = self._orders.get(client_order_id)
            if order is None:
                raise KeyError(f"Unknown order: {client_order_id}")
            if order.is_terminal:
                return order
            order.transition(OrderStatus.CANCELED, note="[PAPER] Cancel simulated")
        logger.info("[PAPER] Order canceled: %s", client_order_id)
        return order

    async def cancel_all(self, symbol: str) -> List[Order]:
        canceled = []
        async with self._lock:
            for order in self._orders.values():
                if order.symbol == symbol and not order.is_terminal:
                    order.transition(OrderStatus.CANCELED, note="[PAPER] cancel_all")
                    canceled.append(order)
        logger.info(
            "[PAPER] cancel_all(%s): %d orders canceled.", symbol, len(canceled)
        )
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

    # ── WS Handler (no-op in paper mode) ────

    async def handle_execution_report(self, data: dict) -> None:
        """In paper mode no real WS execution reports are received."""
        pass

    # ── Helper methods ───────────────────────

    def _print_trade_log(self) -> None:
        if not self.trade_log:
            logger.info("[PAPER] No trades executed.")
            return

        logger.info("\n[PAPER] ══ TRADE LOG ════════════════════════════")
        logger.info(
            "  %-8s %-8s %-5s %-10s %-12s %-12s %-10s",
            "Time",
            "Symbol",
            "Side",
            "Qty",
            "Price",
            "Notional",
            "Cash After",
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
