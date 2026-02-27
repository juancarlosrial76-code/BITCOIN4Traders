"""
Live Execution Engine (Phase 7)
============================
Production-ready orchestration layer for RL-based trading on Binance.

This module provides the main execution engine that coordinates all components
of the live trading system including market data ingestion, signal generation,
risk management, order execution, and monitoring.

Architecture:
  1. BinanceWSConnector: Market data + user data stream WebSocket
  2. OrderManager: Order lifecycle management
  3. RL Agent: Signal generation (from Phase 5 training)
  4. Risk Layer: Position limits, circuit breaker (Phase 4)
  5. Monitor: P&L tracking, alerting (Phase 7)

Execution Flow per Tick:
  1. WS tick received → FeatureEngine.transform()
  2. Agent.predict(features) → signal (-1/0/+1)
  3. Risk pre-check → validate position
  4. OMS.submit_order() → exchange
  5. Fill handlers → update positions & P&L
  6. Monitor callbacks → update metrics

Key Components:
  - LiveExecutionEngine: Main orchestrator
  - CircuitBreaker: Automatic trading halt on losses
  - Position: Per-symbol position tracking with P&L
  - EngineConfig: Configuration for all engine parameters

Usage:
    from src.execution.live_engine import LiveExecutionEngine, EngineConfig
    from src.connectors.binance_ws_connector import ReconnectPolicy

    config = EngineConfig(
        symbols=['BTCUSDT', 'ETHUSDT'],
        api_key='your_key',
        api_secret='your_secret',
        max_position_usd=Decimal('10000'),
        circuit_breaker_pct=Decimal('0.02'),
        daily_loss_limit_usd=Decimal('500')
    )

    engine = LiveExecutionEngine(
        config=config,
        agent=trained_agent,
        feature_engine=feature_engine,
        paper_trading=True
    )

    await engine.start()
    # Runs until stopped or circuit breaker trips
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
    """
    Represents a trading position for a single symbol.

    Tracks position quantity, average cost basis, and calculates both
    realized and unrealized P&L.

    Attributes:
        symbol: Trading pair symbol (e.g., 'BTCUSDT')
        qty: Position quantity (positive = long, negative = short)
        avg_cost: Volume-weighted average entry price
        realized_pnl: Cumulative realized P&L from closed trades

    P&L Calculation:
        - Unrealized P&L = qty × (current_price - avg_cost)
        - Realized P&L = sum of all closed trade P&Ls

    Example:
        pos = Position('BTCUSDT')
        pos.qty = Decimal('0.5')
        pos.avg_cost = Decimal('40000')

        # Calculate unrealized P&L
        unreal = pos.unrealized_pnl(Decimal('45000'))  # = 0.5 × 5000 = 2500
    """

    symbol: str
    qty: Decimal = Decimal(0)  # + long, - short
    avg_cost: Decimal = Decimal(0)  # Volume-weighted average entry price
    realized_pnl: Decimal = Decimal(0)  # Cumulative closed P&L for this symbol

    def update_fill(self, side: OrderSide, fill: Fill) -> Decimal:
        """Update position on fill. Returns realized PnL of this fill."""
        # Convert fill quantity to signed delta (+buy, -sell)
        qty = fill.qty if side == OrderSide.BUY else -fill.qty
        cost = fill.price

        realized = Decimal(0)
        # Detect closing/reversing scenario: opposite sign means we are reducing exposure
        if self.qty != 0 and (qty < 0 < self.qty or qty > 0 > self.qty):
            # Closing or reversing trade
            close_qty = min(abs(qty), abs(self.qty))  # Only close what we have
            realized = close_qty * (cost - self.avg_cost) * (1 if self.qty > 0 else -1)
            self.realized_pnl += realized

        # Update position
        new_qty = self.qty + qty
        if new_qty == 0:
            self.avg_cost = Decimal(0)  # Flat – reset cost basis
        elif (self.qty >= 0 and qty > 0) or (self.qty <= 0 and qty < 0):
            # Adding to position → weighted average cost
            total_cost = self.qty * self.avg_cost + qty * cost
            self.avg_cost = total_cost / new_qty  # New VWAP entry price
        else:
            # Partial close – avg_cost unchanged for remaining position
            pass

        self.qty = new_qty
        return realized

    def unrealized_pnl(self, current_price: Decimal) -> Decimal:
        """Calculate mark-to-market unrealized P&L."""
        if self.qty == 0 or self.avg_cost == 0:
            return Decimal(0)
        return self.qty * (
            current_price - self.avg_cost
        )  # Positive for profitable longs


# ─────────────────────────────────────────────
#  Engine Config
# ─────────────────────────────────────────────


@dataclass
class EngineConfig:
    """
    Configuration for LiveExecutionEngine.

    Defines all parameters for the live trading engine including
    risk limits, execution parameters, and reconnection behavior.

    Risk Limits:
        - max_position_usd: Maximum notional per position (default $10,000)
        - max_order_usd: Maximum notional per single order (default $2,000)
        - circuit_breaker_pct: Drawdown percentage to trigger halt (default 2%)
        - daily_loss_limit_usd: Dollar loss to trigger halt (default $500)

    Execution Settings:
        - use_limit_orders: Use limit orders vs market orders (default True)
        - limit_order_offset_bps: Limit price offset in basis points (default 2)
        - order_timeout_s: Cancel unfilled limit orders after seconds (default 30)

    Example:
        config = EngineConfig(
            symbols=['BTCUSDT', 'ETHUSDT'],
            api_key='mvxxx',
            api_secret='abc123',
            max_position_usd=Decimal('50000'),
            max_order_usd=Decimal('10000'),
            circuit_breaker_pct=Decimal('0.05'),  # 5% drawdown
            daily_loss_limit_usd=Decimal('1000'),   # $1000 daily loss
            use_limit_orders=True,
            limit_order_offset_bps=5,
            reconnect_policy=ReconnectPolicy(max_attempts=10)
        )
    """

    symbols: List[str]
    api_key: str
    api_secret: str

    # Risk limits
    max_position_usd: Decimal = Decimal(10_000)  # Maximum notional per position
    max_order_usd: Decimal = Decimal(2_000)  # Maximum notional per single order
    circuit_breaker_pct: Decimal = Decimal("0.02")  # 2% drawdown → halt
    daily_loss_limit_usd: Decimal = Decimal(500)  # Hard stop on daily loss

    # Execution
    use_limit_orders: bool = True
    limit_order_offset_bps: int = 2  # Place limit 2 bps inside the mid price
    order_timeout_s: float = 30.0  # Cancel unfilled limit orders after this delay

    # Reconnect
    reconnect_policy: ReconnectPolicy = field(default_factory=ReconnectPolicy)


# ─────────────────────────────────────────────
#  Circuit Breaker
# ─────────────────────────────────────────────


class CircuitBreaker:
    """
    Trading circuit breaker for automatic risk control.

    Automatically halts trading when predefined loss thresholds are exceeded,
    preventing further losses during adverse market conditions.

    Trigger Conditions:
        - Drawdown exceeds max_drawdown_pct from peak equity
        - Daily loss exceeds daily_loss_usd from session start

    Behavior:
        - Once tripped, remains latched until manually reset
        - Provides reason for trip via trip_reason property
        - Designed for manual reset by operator

    Example:
        breaker = CircuitBreaker(
            max_drawdown_pct=Decimal('0.02'),  # 2% drawdown
            daily_loss_usd=Decimal('500')      # $500 daily loss
        )

        # Update with current equity
        breaker.update_equity(Decimal('10000'))

        # Check if should halt
        equity = Decimal('9500')  # 5% drawdown
        if breaker.check(equity):
            print(f"STOP TRADING: {breaker.trip_reason}")

        # Reset for new session
        breaker.reset()
    """

    def __init__(self, max_drawdown_pct: Decimal, daily_loss_usd: Decimal):
        self._max_dd = max_drawdown_pct  # Drawdown fraction that trips the breaker
        self._daily_loss = daily_loss_usd  # Dollar loss that trips the breaker
        self._peak_equity = Decimal(0)  # Running maximum equity seen
        self._day_start_eq = Decimal(0)  # Equity at session start
        self._tripped = False  # Latching flag – stays True once triggered
        self._trip_reason = ""

    def update_equity(self, equity: Decimal) -> None:
        """Track the running peak and set the day-start reference on first call."""
        self._peak_equity = max(self._peak_equity, equity)
        if self._day_start_eq == 0:
            self._day_start_eq = equity  # Capture starting equity once

    def check(self, equity: Decimal) -> bool:
        """Returns True if trading should HALT."""
        if self._tripped:
            return True  # Already tripped – remain halted

        if self._peak_equity > 0:
            dd = (self._peak_equity - equity) / self._peak_equity  # Drawdown fraction
            if dd >= self._max_dd:
                self._tripped = True
                self._trip_reason = (
                    f"Drawdown {dd * 100:.2f}% ≥ limit {self._max_dd * 100:.2f}%"
                )
                logger.critical("🔴 CIRCUIT BREAKER TRIPPED: %s", self._trip_reason)
                return True

        if self._day_start_eq > 0:
            daily_loss = self._day_start_eq - equity  # Positive = loss
            if daily_loss >= self._daily_loss:
                self._tripped = True
                self._trip_reason = (
                    f"Daily loss ${daily_loss:.2f} ≥ limit ${self._daily_loss}"
                )
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
    Main orchestrator for live RL-based trading on Binance.

    Coordinates all components of the live trading system, managing the
    complete lifecycle from market data ingestion to order execution
    with comprehensive risk controls.

    Key Responsibilities:
        - WebSocket connection management
        - Market data processing and feature transformation
        - Signal generation from RL agent
        - Pre-trade risk validation
        - Order submission and tracking
        - Position and P&L management
        - Circuit breaker monitoring
        - Session reporting

    Trading Modes:
        - Live Trading: Real orders sent to Binance
        - Paper Trading: Simulated orders with virtual cash

    Event-Driven Architecture:
        - Book ticker updates trigger signal generation
        - Agent signals converted to orders
        - Order fills update positions
        - Circuit breaker checked on each tick

    Example:
        # Initialize
        config = EngineConfig(
            symbols=['BTCUSDT'],
            api_key='key',
            api_secret='secret'
        )

        engine = LiveExecutionEngine(
            config=config,
            agent=trained_agent,
            feature_engine=features,
            paper_trading=True
        )

        # Start trading
        await engine.start()

        # Runs until:
        # - await engine.stop() called
        # - Circuit breaker trips
        # - Unhandled exception

        # Session summary automatically printed on stop
    """

    def __init__(
        self, config: EngineConfig, agent, feature_engine, paper_trading: bool = False
    ):
        self._cfg = config
        self._agent = agent
        self._feat = feature_engine
        self._paper = paper_trading  # Paper mode sends no real orders

        # Build the WebSocket connector with reconnect policy
        self._connector = BinanceWSConnector(
            api_key=config.api_key,
            api_secret=config.api_secret,
            reconnect_policy=config.reconnect_policy,
            paper_trading=paper_trading,
        )

        if paper_trading:
            from ..orders.paper_order_manager import PaperOrderManager

            # Use a simulated order manager with virtual cash balance
            self._oms = PaperOrderManager(
                config.api_key,
                config.api_secret,
                initial_cash=config.max_position_usd,
            )
            logger.info("[PAPER] Paper-trading mode active – no real trades executed.")
        else:
            self._oms = OrderManager(config.api_key, config.api_secret)  # Live OMS

        # One Position object per symbol for P&L tracking
        self._positions: Dict[str, Position] = {s: Position(s) for s in config.symbols}
        self._last_prices: Dict[str, Decimal] = {}  # Latest mid prices per symbol
        self._initial_equity: Optional[Decimal] = None  # Locked at first equity read
        self._cash = Decimal(0)  # Cash balance (live mode)

        self._breaker = CircuitBreaker(
            config.circuit_breaker_pct,
            config.daily_loss_limit_usd,
        )

        self._running = False
        self._tick_count = 0  # Total book-ticker events received
        self._order_timeout_tasks: Dict[
            str, asyncio.Task
        ] = {}  # Watchdog tasks keyed by client_order_id

    # ── Public API ───────────────────────────

    async def start(self) -> None:
        logger.info("═══ PHASE 7 LIVE ENGINE STARTING ═══")
        await self._oms.start()  # Initialize order manager (REST session, etc.)

        # Wire callbacks
        self._oms.on_fill(self._on_fill)
        self._oms.on_status_change(self._on_status_change)

        # Register WS handlers for each symbol's book-ticker stream
        for symbol in self._cfg.symbols:
            stream = f"{symbol.lower()}@bookTicker"
            self._connector.on(stream, self._on_book_ticker)

        # Route execution reports from user stream to OMS
        self._connector.on("executionReport", self._oms.handle_execution_report)

        # Connect
        await self._connector.connect()

        # Subscribe market data streams for all configured symbols
        streams = [f"{s.lower()}@bookTicker" for s in self._cfg.symbols]
        await self._connector.subscribe(streams)

        self._running = True
        logger.info("✅ Engine live. Symbols: %s", self._cfg.symbols)

        # Run until stopped
        await self._run_loop()

    async def stop(self, reason: str = "manual") -> None:
        logger.warning("ENGINE STOPPING: %s", reason)
        self._running = False

        # Cancel all open orders before disconnecting
        for symbol in self._cfg.symbols:
            try:
                canceled = await self._oms.cancel_all(symbol)
                if canceled:
                    logger.info(
                        "Canceled %d orders for %s on shutdown.", len(canceled), symbol
                    )
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
            await asyncio.sleep(1)  # 1-second heartbeat interval

            equity = self._compute_equity()
            if self._initial_equity is None and equity > 0:
                # Lock the starting equity for drawdown calculations
                self._initial_equity = equity
                self._breaker.update_equity(equity)
                logger.info("Initial equity locked: $%.2f", float(equity))
            elif equity > 0:
                self._breaker.update_equity(equity)
                if self._breaker.check(equity):
                    # Circuit breaker tripped – halt immediately
                    await self.stop(reason=self._breaker.trip_reason)
                    return

    # ── Market Data Handler ──────────────────

    async def _on_book_ticker(self, data: dict) -> None:
        """
        Fired on every best bid/ask update.
        { "s": "BTCUSDT", "b": "29999.00", "B": "10.000", "a": "30001.00", "A": "5.000" }
        """
        symbol = data["s"]
        bid = Decimal(data["b"])
        ask = Decimal(data["a"])
        mid = (bid + ask) / 2  # Simple mid price for feature computation

        self._last_prices[symbol] = mid
        self._tick_count += 1

        # Only run agent every N ticks to avoid over-trading
        if self._tick_count % 10 != 0:
            return  # Throttle: process only 1 in 10 ticks

        if self._breaker.is_tripped:
            return  # Do not generate signals when halted

        await self._on_tick(symbol, bid, ask, mid)

    async def _on_tick(
        self, symbol: str, bid: Decimal, ask: Decimal, mid: Decimal
    ) -> None:
        """Full tick processing pipeline."""
        try:
            # 1. Build feature vector from raw price
            features = self._feat.transform_single(symbol, float(mid))
            if features is None:
                return  # Not enough history yet

            # 2. Agent signal: -1 (short), 0 (flat), +1 (long)
            signal = self._agent.predict(features)

            # 3. Risk pre-check before submitting
            if not self._pre_trade_risk_check(symbol, signal, mid):
                return

            # 4. Execute the signal as an order
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
        delta_qty = target_qty - current_qty  # Size adjustment needed

        if abs(delta_qty) < Decimal("0.0001"):
            return  # No meaningful change – skip to avoid noise trades

        side = OrderSide.BUY if delta_qty > 0 else OrderSide.SELL
        qty = abs(delta_qty)

        if self._cfg.use_limit_orders:
            # Place limit inside the spread to get maker rebate
            offset = (
                (ask - bid)
                * Decimal(self._cfg.limit_order_offset_bps)
                / Decimal(10_000)
            )
            # Buy slightly above bid; sell slightly below ask
            price = (bid + offset) if side == OrderSide.BUY else (ask - offset)
            order = Order(
                symbol=symbol,
                side=side,
                type=OrderType.LIMIT,
                tif=TimeInForce.GTC,
                quantity=qty,
                limit_price=price,
            )
        else:
            # Market order for urgent execution
            order = Order(
                symbol=symbol,
                side=side,
                type=OrderType.MARKET,
                quantity=qty,
            )

        submitted = await self._oms.submit_order(order)
        logger.info("Signal %+d → %s", signal, submitted)

        # Start timeout watchdog for limit orders (cancel if not filled in time)
        if order.type == OrderType.LIMIT:
            task = asyncio.create_task(
                self._order_timeout_watchdog(submitted.client_order_id),
                name=f"timeout_{submitted.client_order_id}",
            )
            self._order_timeout_tasks[submitted.client_order_id] = task

    def _signal_to_target_qty(
        self, signal: int, symbol: str, price: Decimal
    ) -> Decimal:
        """Convert discrete signal to target position size."""
        if signal == 0 or price == 0:
            return Decimal(0)  # Flat signal → zero position
        max_notional = self._cfg.max_order_usd
        qty = (max_notional / price).quantize(Decimal("0.001"))  # Round to 3 decimals
        return qty if signal > 0 else -qty  # Negative for short positions

    # ── Risk Checks ──────────────────────────

    def _pre_trade_risk_check(self, symbol: str, signal: int, price: Decimal) -> bool:
        if signal == 0:
            return False  # No action needed for flat signal

        pos = self._positions[symbol]
        proposed_notional = (
            abs(self._signal_to_target_qty(signal, symbol, price)) * price
        )

        # Check position limit
        if proposed_notional > self._cfg.max_position_usd:
            logger.warning(
                "Position limit breach for %s: $%.2f > $%.2f",
                symbol,
                proposed_notional,
                self._cfg.max_position_usd,
            )
            return False

        # Don't stack same direction – require meaningful distance before adding
        if signal > 0 and pos.qty >= proposed_notional / price * Decimal("0.8"):
            return False  # Already 80%+ long – skip
        if signal < 0 and pos.qty <= -proposed_notional / price * Decimal("0.8"):
            return False  # Already 80%+ short – skip

        return True

    # ── Fill / Status Handlers ───────────────

    def _on_fill(self, order: Order, fill: Fill) -> None:
        pos = self._positions.get(order.symbol)
        if pos is None:
            return
        realized = pos.update_fill(order.side, fill)
        logger.info(
            "FILL: %s %s qty=%s @%s | realized_pnl=$%.4f",
            order.symbol,
            order.side.value,
            fill.qty,
            fill.price,
            float(realized),
        )

    def _on_status_change(self, order: Order, new_status: OrderStatus) -> None:
        logger.info("Order %s → %s", order.client_order_id, new_status.name)
        if new_status in (OrderStatus.FILLED, OrderStatus.CANCELED):
            # Cancel watchdog task for terminal orders to avoid spurious cancels
            task = self._order_timeout_tasks.pop(order.client_order_id, None)
            if task and not task.done():
                task.cancel()

    # ── Order Timeout Watchdog ───────────────

    async def _order_timeout_watchdog(self, client_order_id: str) -> None:
        """Cancel limit order if not filled within timeout."""
        await asyncio.sleep(self._cfg.order_timeout_s)  # Wait for fill window
        order = self._oms.get_order(client_order_id)
        if order and not order.is_terminal:
            logger.warning("Order %s timeout – canceling.", client_order_id)
            try:
                await self._oms.cancel_order(client_order_id)
            except Exception as exc:
                logger.error("Timeout cancel failed: %s", exc)

    # ── Equity / PnL ────────────────────────

    def _compute_equity(self) -> Decimal:
        # In paper-trading mode: Cash + mark-to-market value of open positions
        # Cash accounting: BUY → cash decreases, SELL → cash increases (incl. short proceeds)
        # Correct equity = Cash + (qty * current_price) for all positions
        # For short positions (qty < 0): qty*px is negative → correctly reduces equity
        if self._paper:
            from ..orders.paper_order_manager import PaperOrderManager

            if isinstance(self._oms, PaperOrderManager):
                cash = self._oms.cash
                open_position_value = Decimal(0)
                for symbol, pos in self._positions.items():
                    if pos.qty != 0:
                        px = self._last_prices.get(symbol, Decimal(0))
                        open_position_value += pos.qty * px  # Mark-to-market
                return cash + open_position_value

        # Live mode: sum unrealized P&L across all positions
        unrealized = Decimal(0)
        for symbol, pos in self._positions.items():
            px = self._last_prices.get(symbol, Decimal(0))
            unrealized += pos.unrealized_pnl(px)
        realized = sum(p.realized_pnl for p in self._positions.values())
        return self._cash + realized + unrealized  # Total equity

    def _print_session_summary(self) -> None:
        realized = sum(float(p.realized_pnl) for p in self._positions.values())
        unrealized = sum(
            float(p.unrealized_pnl(self._last_prices.get(s, Decimal(0))))
            for s, p in self._positions.items()
        )
        orders = self._oms.get_all_orders()
        fills = sum(len(o.fills) for o in orders)  # Total number of fill events

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
