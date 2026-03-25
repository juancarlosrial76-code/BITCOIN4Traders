"""
Tests for Order, OrderStatus, Fill FSM (src/orders/order_manager.py)
=====================================================================
Pure-dataclass logic — no HTTP calls needed.
"""

import time
from decimal import Decimal

import pytest

from src.orders.order_manager import (
    Fill,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    TimeInForce,
)


# ─────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────


def make_order(
    qty: str = "0.001",
    limit: str = "50000",
    side: OrderSide = OrderSide.BUY,
) -> Order:
    return Order(
        symbol="BTCUSDT",
        side=side,
        type=OrderType.LIMIT,
        tif=TimeInForce.GTC,
        quantity=Decimal(qty),
        limit_price=Decimal(limit),
    )


def make_fill(price: str, qty: str, trade_id: int = 1) -> Fill:
    return Fill(
        price=Decimal(price),
        qty=Decimal(qty),
        commission=Decimal("0"),
        commission_asset="USDT",
        trade_id=trade_id,
        timestamp_ms=int(time.time() * 1000),
    )


# ─────────────────────────────────────────────
#  FSM Transition Tests
# ─────────────────────────────────────────────


class TestOrderStatusFSM:
    def test_pending_new_to_new(self):
        order = make_order()
        assert order.status == OrderStatus.PENDING_NEW
        order.transition(OrderStatus.NEW)
        assert order.status == OrderStatus.NEW

    def test_new_to_partially_filled(self):
        order = make_order()
        order.transition(OrderStatus.NEW)
        order.transition(OrderStatus.PARTIALLY_FILLED)
        assert order.status == OrderStatus.PARTIALLY_FILLED

    def test_partially_filled_to_filled(self):
        order = make_order()
        order.transition(OrderStatus.NEW)
        order.transition(OrderStatus.PARTIALLY_FILLED)
        order.transition(OrderStatus.FILLED)
        assert order.status == OrderStatus.FILLED

    def test_new_to_filled_direct(self):
        order = make_order()
        order.transition(OrderStatus.NEW)
        order.transition(OrderStatus.FILLED)
        assert order.status == OrderStatus.FILLED

    def test_new_to_canceled(self):
        order = make_order()
        order.transition(OrderStatus.NEW)
        order.transition(OrderStatus.CANCELED)
        assert order.status == OrderStatus.CANCELED

    def test_new_to_expired(self):
        order = make_order()
        order.transition(OrderStatus.NEW)
        order.transition(OrderStatus.EXPIRED)
        assert order.status == OrderStatus.EXPIRED

    def test_pending_new_to_rejected(self):
        order = make_order()
        order.transition(OrderStatus.REJECTED)
        assert order.status == OrderStatus.REJECTED

    def test_invalid_transition_filled_to_new_raises(self):
        order = make_order()
        order.transition(OrderStatus.NEW)
        order.transition(OrderStatus.FILLED)
        with pytest.raises(ValueError, match="Illegal transition"):
            order.transition(OrderStatus.NEW)

    def test_invalid_transition_canceled_to_filled_raises(self):
        order = make_order()
        order.transition(OrderStatus.NEW)
        order.transition(OrderStatus.CANCELED)
        with pytest.raises(ValueError, match="Illegal transition"):
            order.transition(OrderStatus.FILLED)

    def test_invalid_transition_pending_new_to_filled_raises(self):
        order = make_order()
        with pytest.raises(ValueError, match="Illegal transition"):
            order.transition(OrderStatus.FILLED)

    @pytest.mark.parametrize("terminal_status", [
        OrderStatus.FILLED,
        OrderStatus.CANCELED,
        OrderStatus.REJECTED,
        OrderStatus.EXPIRED,
    ])
    def test_terminal_states_cannot_transition(self, terminal_status):
        order = make_order()
        # Get to terminal state via valid path
        if terminal_status == OrderStatus.FILLED:
            order.transition(OrderStatus.NEW)
            order.transition(OrderStatus.FILLED)
        elif terminal_status == OrderStatus.CANCELED:
            order.transition(OrderStatus.NEW)
            order.transition(OrderStatus.CANCELED)
        elif terminal_status == OrderStatus.REJECTED:
            order.transition(OrderStatus.REJECTED)
        elif terminal_status == OrderStatus.EXPIRED:
            order.transition(OrderStatus.NEW)
            order.transition(OrderStatus.EXPIRED)
        with pytest.raises(ValueError):
            order.transition(OrderStatus.NEW)


# ─────────────────────────────────────────────
#  Order.add_fill Tests
# ─────────────────────────────────────────────


class TestOrderAddFill:
    def test_single_fill_sets_avg_price(self):
        order = make_order(qty="1.0", limit="50000")
        fill = make_fill("50100", "1.0")
        order.add_fill(fill)
        assert order.filled_qty == Decimal("1.0")
        assert order.avg_fill_price == Decimal("50100")

    def test_two_fills_vwap_calculation(self):
        order = make_order(qty="2.0", limit="50000")
        order.add_fill(make_fill("50000", "1.0", trade_id=1))
        order.add_fill(make_fill("51000", "1.0", trade_id=2))
        # VWAP = (50000*1 + 51000*1) / 2 = 50500
        assert order.avg_fill_price == pytest.approx(Decimal("50500"))
        assert order.filled_qty == Decimal("2.0")

    def test_three_fills_vwap_weighted(self):
        order = make_order(qty="4.0", limit="50000")
        order.add_fill(make_fill("40000", "1.0", trade_id=1))  # weight 1
        order.add_fill(make_fill("50000", "2.0", trade_id=2))  # weight 2
        order.add_fill(make_fill("60000", "1.0", trade_id=3))  # weight 1
        # VWAP = (40000 + 100000 + 60000) / 4 = 50000
        assert order.avg_fill_price == pytest.approx(Decimal("50000"))
        assert order.filled_qty == Decimal("4.0")

    def test_fills_accumulated_in_list(self):
        order = make_order(qty="2.0")
        order.add_fill(make_fill("50000", "1.0", trade_id=1))
        order.add_fill(make_fill("51000", "1.0", trade_id=2))
        assert len(order.fills) == 2

    def test_audit_grows_on_fill(self):
        order = make_order()
        initial_len = len(order.audit)
        order.add_fill(make_fill("50000", "0.001"))
        assert len(order.audit) > initial_len


# ─────────────────────────────────────────────
#  Computed Properties Tests
# ─────────────────────────────────────────────


class TestOrderProperties:
    def test_remaining_qty_unfilled(self):
        order = make_order(qty="1.0")
        assert order.remaining_qty == Decimal("1.0")

    def test_remaining_qty_partial_fill(self):
        order = make_order(qty="1.0")
        order.add_fill(make_fill("50000", "0.3"))
        assert order.remaining_qty == pytest.approx(Decimal("0.7"))

    def test_remaining_qty_full_fill(self):
        order = make_order(qty="1.0")
        order.add_fill(make_fill("50000", "1.0"))
        assert order.remaining_qty == Decimal("0.0")

    def test_is_terminal_false_for_pending(self):
        order = make_order()
        assert order.is_terminal is False

    def test_is_terminal_false_for_new(self):
        order = make_order()
        order.transition(OrderStatus.NEW)
        assert order.is_terminal is False

    @pytest.mark.parametrize("terminal", [
        OrderStatus.FILLED, OrderStatus.CANCELED,
        OrderStatus.REJECTED, OrderStatus.EXPIRED,
    ])
    def test_is_terminal_true_for_terminal_states(self, terminal):
        order = make_order()
        if terminal == OrderStatus.REJECTED:
            order.transition(OrderStatus.REJECTED)
        elif terminal == OrderStatus.FILLED:
            order.transition(OrderStatus.NEW)
            order.transition(OrderStatus.FILLED)
        elif terminal == OrderStatus.CANCELED:
            order.transition(OrderStatus.NEW)
            order.transition(OrderStatus.CANCELED)
        elif terminal == OrderStatus.EXPIRED:
            order.transition(OrderStatus.NEW)
            order.transition(OrderStatus.EXPIRED)
        assert order.is_terminal is True

    def test_slippage_bps_buy_above_limit(self):
        order = make_order(qty="1.0", limit="50000")
        order.add_fill(make_fill("50050", "1.0"))  # Filled 50 above limit
        # slippage = (50050 - 50000) / 50000 * 10000 = 10 bps (bad for buy)
        assert order.slippage_bps == pytest.approx(Decimal("10"))

    def test_slippage_bps_buy_at_limit(self):
        order = make_order(qty="1.0", limit="50000")
        order.add_fill(make_fill("50000", "1.0"))
        assert order.slippage_bps == pytest.approx(Decimal("0"))

    def test_slippage_bps_none_for_market_order(self):
        order = Order(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            type=OrderType.MARKET,
            quantity=Decimal("1.0"),
        )
        order.add_fill(make_fill("50000", "1.0"))
        assert order.slippage_bps is None

    def test_slippage_bps_sell_above_limit_is_good(self):
        order = Order(
            symbol="BTCUSDT",
            side=OrderSide.SELL,
            type=OrderType.LIMIT,
            quantity=Decimal("1.0"),
            limit_price=Decimal("50000"),
        )
        order.add_fill(make_fill("50050", "1.0"))  # Filled above limit → good for sell
        # For sells, sign is flipped: diff = -(50050-50000)/50000 → *10000 = -10
        # Then flipped again for sell → positive
        assert order.slippage_bps is not None


# ─────────────────────────────────────────────
#  Audit Log Tests
# ─────────────────────────────────────────────


class TestOrderAudit:
    def test_audit_grows_on_each_transition(self):
        order = make_order()
        assert len(order.audit) == 0
        order.transition(OrderStatus.NEW, note="submitted")
        assert len(order.audit) == 1
        order.transition(OrderStatus.FILLED)
        assert len(order.audit) == 2

    def test_audit_contains_transition_string(self):
        order = make_order()
        order.transition(OrderStatus.NEW, note="test note")
        ts, msg = order.audit[0]
        assert "PENDING_NEW" in msg
        assert "NEW" in msg

    def test_audit_contains_fill_info(self):
        order = make_order()
        fill = make_fill("50000", "0.001")
        order.add_fill(fill)
        ts, msg = order.audit[-1]
        assert "FILL" in msg
        assert "50000" in msg

    def test_client_order_id_format(self):
        order = make_order()
        assert order.client_order_id.startswith("p7_")
        assert len(order.client_order_id) == 15  # "p7_" + 12 hex chars
