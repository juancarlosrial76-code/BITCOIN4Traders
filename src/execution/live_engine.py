"""
PHASE 7 – LIVE EXECUTION ENGINE
================================
Orchestrates:
  1. BinanceWSConnector  – market data + user stream
  2. OrderManager        – order lifecycle
  3. RL Agent            – signal generation (from Phase 5)
  4. Risk Layer          – position limits, circuit breaker (Phase 4)
  5. Monitoring          – P&L tracking, alerting

Flow per tick:
  WS tick  →  FeatureEngine.transform()  →  Agent.predict()
           →  Risk checks  →  OMS.submit()  →  Fill handlers
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Dict, List, Optional

from ..connectors.binance_ws_connector import (
    BinanceWSConnector,
    ReconnectPolicy,
)
from ..orders.order_manager import (
    Fill,
    Order,
    OrderManager,
    OrderSide,
    OrderStatus,
    OrderType,
    TimeInForce,
)

logger = logging.getLogger("phase7.engine")


# ─────────────────────────────────────────────
#  Position Tracker
# ─────────────────────────────────────────────

@dataclass
class Position:
    symbol:      str
    qty:         Decimal = Decimal(0)      # + long, - short
    avg_cost:    Decimal = Decimal(0)
    realized_pnl: Decimal = Decimal(0)

    def update_fill(self, side: OrderSide, fill: Fill) -> Decimal:
        """Update position on fill. Returns realized PnL of this fill."""
        qty  = fill.qty if side == OrderSide.BUY else -fill.qty
        cost = fill.price

        realized = Decimal(0)
        if self.qty != 0 and (qty < 0 < self.qty or qty > 0 > self.qty):
            # Closing or reversing trade
            close_qty = min(abs(qty), abs(self.qty))
            realized  = close_qty * (cost - self.avg_cost) * (1 if self.qty > 0 else -1)
            self.realized_pnl += realized

        # Update position
        new_qty = self.qty + qty
        if new_qty == 0:
            self.avg_cost = Decimal(0)
        elif (self.qty >= 0 and qty > 0) or (self.qty <= 0 and qty < 0):
            # Adding to position → weighted average cost
            total_cost = self.qty * self.avg_cost + qty * cost
            self.avg_cost = total_cost / new_qty
        else:
            # Partial close – avg_cost unchanged for remaining
            pass

        self.qty = new_qty
        return realized

    def unrealized_pnl(self, current_price: Decimal) -> Decimal:
        if self.qty == 0 or self.avg_cost == 0:
            return Decimal(0)
        return self.qty * (current_price - self.avg_cost)


# ─────────────────────────────────────────────
#  Engine Config
# ─────────────────────────────────────────────

@dataclass
class EngineConfig:
    symbols:         List[str]
    api_key:         str
    api_secret:      str

    # Risk limits
    max_position_usd:    Decimal = Decimal(10_000)
    max_order_usd:       Decimal = Decimal(2_000)
    circuit_breaker_pct: Decimal = Decimal("0.02")   # 2% drawdown → halt
    daily_loss_limit_usd: Decimal = Decimal(500)

    # Execution
    use_limit_orders:     bool  = True
    limit_order_offset_bps: int = 2    # place limit 2 bps inside mid
    order_timeout_s:      float = 30.0

    # Reconnect
    reconnect_policy: ReconnectPolicy = field(default_factory=ReconnectPolicy)


# ─────────────────────────────────────────────
#  Circuit Breaker
# ─────────────────────────────────────────────

class CircuitBreaker:
    def __init__(self, max_drawdown_pct: Decimal, daily_loss_usd: Decimal):
        self._max_dd       = max_drawdown_pct
        self._daily_loss   = daily_loss_usd
        self._peak_equity  = Decimal(0)
        self._day_start_eq = Decimal(0)
        self._tripped      = False
        self._trip_reason  = ""

    def update_equity(self, equity: Decimal) -> None:
        self._peak_equity   = max(self._peak_equity, equity)
        if self._day_start_eq == 0:
            self._day_start_eq = equity

    def check(self, equity: Decimal) -> bool:
        """Returns True if trading should HALT."""
        if self._tripped:
            return True

        if self._peak_equity > 0:
            dd = (self._peak_equity - equity) / self._peak_equity
            if dd >= self._max_dd:
                self._tripped    = True
                self._trip_reason = f"Drawdown {dd*100:.2f}% ≥ limit {self._max_dd*100:.2f}%"
                logger.critical("🔴 CIRCUIT BREAKER TRIPPED: %s", self._trip_reason)
                return True

        if self._day_start_eq > 0:
            daily_loss = self._day_start_eq - equity
            if daily_loss >= self._daily_loss:
                self._tripped    = True
                self._trip_reason = f"Daily loss ${daily_loss:.2f} ≥ limit ${self._daily_loss}"
                logger.critical("🔴 CIRCUIT BREAKER TRIPPED: %s", self._trip_reason)
                return True

        return False

    @property
    def is_tripped(self) -> bool:
        return self._tripped

    @property
    def trip_reason(self) -> str:
        return self._trip_reason

    def reset(self) -> None:
        """Manual reset only – requires explicit operator confirmation."""
        logger.warning("Circuit breaker manually reset.")
        self._tripped = False


# ─────────────────────────────────────────────
#  Live Execution Engine
# ─────────────────────────────────────────────

class LiveExecutionEngine:
    """
    Main orchestrator for live trading.

    Usage:
        engine = LiveExecutionEngine(config, agent, feature_engine)
        await engine.start()
        # runs until engine.stop() is called or circuit breaker trips
    """

    def __init__(self, config: EngineConfig, agent, feature_engine):
        self._cfg    = config
        self._agent  = agent
        self._feat   = feature_engine

        self._connector = BinanceWSConnector(
            api_key          = config.api_key,
            api_secret       = config.api_secret,
            reconnect_policy = config.reconnect_policy,
        )
        self._oms = OrderManager(config.api_key, config.api_secret)

        self._positions: Dict[str, Position] = {s: Position(s) for s in config.symbols}
        self._last_prices: Dict[str, Decimal] = {}
        self._initial_equity: Optional[Decimal] = None
        self._cash = Decimal(0)

        self._breaker = CircuitBreaker(
            config.circuit_breaker_pct,
            config.daily_loss_limit_usd,
        )

        self._running = False
        self._tick_count = 0
        self._order_timeout_tasks: Dict[str, asyncio.Task] = {}

    # ── Public API ───────────────────────────

    async def start(self) -> None:
        logger.info("═══ PHASE 7 LIVE ENGINE STARTING ═══")
        await self._oms.start()

        # Wire callbacks
        self._oms.on_fill(self._on_fill)
        self._oms.on_status_change(self._on_status_change)

        # Register WS handlers
        for symbol in self._cfg.symbols:
            stream = f"{symbol.lower()}@bookTicker"
            self._connector.on(stream, self._on_book_ticker)

        self._connector.on("executionReport", self._oms.handle_execution_report)

        # Connect
        await self._connector.connect()

        # Subscribe market data streams
        streams = [f"{s.lower()}@bookTicker" for s in self._cfg.symbols]
        await self._connector.subscribe(streams)

        self._running = True
        logger.info("✅ Engine live. Symbols: %s", self._cfg.symbols)

        # Run until stopped
        await self._run_loop()

    async def stop(self, reason: str = "manual") -> None:
        logger.warning("ENGINE STOPPING: %s", reason)
        self._running = False

        # Cancel all open orders
        for symbol in self._cfg.symbols:
            try:
                canceled = await self._oms.cancel_all(symbol)
                if canceled:
                    logger.info("Canceled %d orders for %s on shutdown.", len(canceled), symbol)
            except Exception as exc:
                logger.error("cancel_all failed on shutdown for %s: %s", symbol, exc)

        await self._connector.disconnect()
        await self._oms.stop()
        self._print_session_summary()
        logger.info("═══ ENGINE STOPPED ═══")

    # ── Main Loop ────────────────────────────

    async def _run_loop(self) -> None:
        """
        The engine is event-driven via WS callbacks.
        This loop checks for circuit breaker and watchdog conditions.
        """
        while self._running:
            await asyncio.sleep(1)

            equity = self._compute_equity()
            if self._initial_equity is None and equity > 0:
                self._initial_equity = equity
                self._breaker.update_equity(equity)
                logger.info("Initial equity locked: $%.2f", float(equity))
            elif equity > 0:
                self._breaker.update_equity(equity)
                if self._breaker.check(equity):
                    await self.stop(reason=self._breaker.trip_reason)
                    return

    # ── Market Data Handler ──────────────────

    async def _on_book_ticker(self, data: dict) -> None:
        """
        Fired on every best bid/ask update.
        { "s": "BTCUSDT", "b": "29999.00", "B": "10.000", "a": "30001.00", "A": "5.000" }
        """
        symbol = data["s"]
        bid    = Decimal(data["b"])
        ask    = Decimal(data["a"])
        mid    = (bid + ask) / 2

        self._last_prices[symbol] = mid
        self._tick_count += 1

        # Only run agent every N ticks to avoid over-trading
        if self._tick_count % 10 != 0:
            return

        if self._breaker.is_tripped:
            return

        await self._on_tick(symbol, bid, ask, mid)

    async def _on_tick(self, symbol: str, bid: Decimal, ask: Decimal, mid: Decimal) -> None:
        """Full tick processing pipeline."""
        try:
            # 1. Build feature vector
            features = self._feat.transform_single(symbol, float(mid))
            if features is None:
                return   # Not enough history yet

            # 2. Agent signal: -1 (short), 0 (flat), +1 (long)
            signal = self._agent.predict(features)

            # 3. Risk pre-check
            if not self._pre_trade_risk_check(symbol, signal, mid):
                return

            # 4. Execute
            await self._execute_signal(symbol, signal, bid, ask)

        except Exception as exc:
            logger.error("Tick processing error for %s: %s", symbol, exc, exc_info=True)

    # ── Signal Execution ─────────────────────

    async def _execute_signal(
        self,
        symbol: str,
        signal: int,
        bid: Decimal,
        ask: Decimal,
    ) -> None:
        pos = self._positions[symbol]
        current_qty = pos.qty

        target_qty = self._signal_to_target_qty(signal, symbol, (bid + ask) / 2)
        delta_qty  = target_qty - current_qty

        if abs(delta_qty) < Decimal("0.0001"):
            return   # No meaningful change

        side = OrderSide.BUY if delta_qty > 0 else OrderSide.SELL
        qty  = abs(delta_qty)

        if self._cfg.use_limit_orders:
            # Place limit inside the spread
            offset = (ask - bid) * Decimal(self._cfg.limit_order_offset_bps) / Decimal(10_000)
            price  = (bid + offset) if side == OrderSide.BUY else (ask - offset)
            order  = Order(
                symbol      = symbol,
                side        = side,
                type        = OrderType.LIMIT,
                tif         = TimeInForce.GTC,
                quantity    = qty,
                limit_price = price,
            )
        else:
            order = Order(
                symbol   = symbol,
                side     = side,
                type     = OrderType.MARKET,
                quantity = qty,
            )

        submitted = await self._oms.submit_order(order)
        logger.info("Signal %+d → %s", signal, submitted)

        # Start timeout watchdog for limit orders
        if order.type == OrderType.LIMIT:
            task = asyncio.create_task(
                self._order_timeout_watchdog(submitted.client_order_id),
                name=f"timeout_{submitted.client_order_id}"
            )
            self._order_timeout_tasks[submitted.client_order_id] = task

    def _signal_to_target_qty(self, signal: int, symbol: str, price: Decimal) -> Decimal:
        """Convert discrete signal to target position size."""
        if signal == 0 or price == 0:
            return Decimal(0)
        max_notional = self._cfg.max_order_usd
        qty = (max_notional / price).quantize(Decimal("0.001"))
        return qty if signal > 0 else -qty

    # ── Risk Checks ──────────────────────────

    def _pre_trade_risk_check(self, symbol: str, signal: int, price: Decimal) -> bool:
        if signal == 0:
            return False

        pos = self._positions[symbol]
        proposed_notional = abs(self._signal_to_target_qty(signal, symbol, price)) * price

        # Check position limit
        if proposed_notional > self._cfg.max_position_usd:
            logger.warning("Position limit breach for %s: $%.2f > $%.2f",
                           symbol, proposed_notional, self._cfg.max_position_usd)
            return False

        # Don't stack same direction
        if (signal > 0 and pos.qty >= proposed_notional / price * Decimal("0.8")):
            return False
        if (signal < 0 and pos.qty <= -proposed_notional / price * Decimal("0.8")):
            return False

        return True

    # ── Fill / Status Handlers ───────────────

    def _on_fill(self, order: Order, fill: Fill) -> None:
        pos = self._positions.get(order.symbol)
        if pos is None:
            return
        realized = pos.update_fill(order.side, fill)
        logger.info(
            "FILL: %s %s qty=%s @%s | realized_pnl=$%.4f",
            order.symbol, order.side.value, fill.qty, fill.price, float(realized)
        )

    def _on_status_change(self, order: Order, new_status: OrderStatus) -> None:
        logger.info("Order %s → %s", order.client_order_id, new_status.name)
        if new_status in (OrderStatus.FILLED, OrderStatus.CANCELED):
            task = self._order_timeout_tasks.pop(order.client_order_id, None)
            if task and not task.done():
                task.cancel()

    # ── Order Timeout Watchdog ───────────────

    async def _order_timeout_watchdog(self, client_order_id: str) -> None:
        """Cancel limit order if not filled within timeout."""
        await asyncio.sleep(self._cfg.order_timeout_s)
        order = self._oms.get_order(client_order_id)
        if order and not order.is_terminal:
            logger.warning("Order %s timeout – canceling.", client_order_id)
            try:
                await self._oms.cancel_order(client_order_id)
            except Exception as exc:
                logger.error("Timeout cancel failed: %s", exc)

    # ── Equity / PnL ────────────────────────

    def _compute_equity(self) -> Decimal:
        unrealized = Decimal(0)
        for symbol, pos in self._positions.items():
            px = self._last_prices.get(symbol, Decimal(0))
            unrealized += pos.unrealized_pnl(px)
        realized = sum(p.realized_pnl for p in self._positions.values())
        return self._cash + realized + unrealized

    def _print_session_summary(self) -> None:
        realized = sum(float(p.realized_pnl) for p in self._positions.values())
        unrealized = sum(
            float(p.unrealized_pnl(self._last_prices.get(s, Decimal(0))))
            for s, p in self._positions.items()
        )
        orders = self._oms.get_all_orders()
        fills  = sum(len(o.fills) for o in orders)

        logger.info(
            "\n╔══════════════════════════════════════╗\n"
            "║        SESSION SUMMARY – PHASE 7     ║\n"
            "╠══════════════════════════════════════╣\n"
            "║  Ticks processed : %-18d║\n"
            "║  Orders submitted: %-18d║\n"
            "║  Total fills     : %-18d║\n"
            "║  Realized PnL    : $%-17.2f║\n"
            "║  Unrealized PnL  : $%-17.2f║\n"
            "╚══════════════════════════════════════╝",
            self._tick_count,
            len(orders),
            fills,
            realized,
            unrealized,
        )
