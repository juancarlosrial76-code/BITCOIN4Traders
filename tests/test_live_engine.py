"""
Tests for CircuitBreaker and Position (src/execution/live_engine.py)
=====================================================================
Unit tests for the pure-dataclass components of the live execution engine.
No external I/O or async calls needed for these tests.
"""

from decimal import Decimal

import pytest

from src.execution.live_engine import CircuitBreaker, Position
from src.orders.order_manager import Fill, OrderSide


# ─────────────────────────────────────────────
#  Fixtures
# ─────────────────────────────────────────────


def make_fill(price: str, qty: str, trade_id: int = 1) -> Fill:
    return Fill(
        price=Decimal(price),
        qty=Decimal(qty),
        commission=Decimal("0"),
        commission_asset="USDT",
        trade_id=trade_id,
        timestamp_ms=1_700_000_000_000,
    )


# ─────────────────────────────────────────────
#  CircuitBreaker Tests
# ─────────────────────────────────────────────


class TestCircuitBreaker:
    def _make_breaker(
        self,
        max_dd: str = "0.05",
        daily_loss: str = "500",
    ) -> CircuitBreaker:
        return CircuitBreaker(
            max_drawdown_pct=Decimal(max_dd),
            daily_loss_usd=Decimal(daily_loss),
        )

    def test_no_trip_under_normal_conditions(self):
        cb = self._make_breaker()
        cb.update_equity(Decimal("10000"))
        assert cb.check(Decimal("9600")) is False  # 4% drawdown, limit is 5%
        assert cb.is_tripped is False

    def test_trips_on_drawdown_exceeds_max(self):
        cb = self._make_breaker(max_dd="0.05")
        cb.update_equity(Decimal("10000"))
        # 6% drawdown → should trip
        assert cb.check(Decimal("9400")) is True
        assert cb.is_tripped is True

    def test_trips_at_exact_drawdown_boundary(self):
        cb = self._make_breaker(max_dd="0.05")
        cb.update_equity(Decimal("10000"))
        # exactly 5% drawdown → should trip (>= comparison)
        assert cb.check(Decimal("9500")) is True

    def test_trips_on_daily_loss_exceeds_limit(self):
        # max_dd set high so drawdown does not trip first
        cb = self._make_breaker(max_dd="0.99", daily_loss="500")
        cb.update_equity(Decimal("10000"))  # sets day-start eq
        # loss of $600 > limit of $500 → trip
        assert cb.check(Decimal("9400")) is True
        assert "Daily loss" in cb.trip_reason

    def test_trips_on_daily_loss_exact_boundary(self):
        cb = self._make_breaker(daily_loss="500")
        cb.update_equity(Decimal("10000"))
        assert cb.check(Decimal("9500")) is True  # exactly $500 loss

    def test_latches_after_trip(self):
        """Once tripped, stays tripped even after equity recovers."""
        cb = self._make_breaker(max_dd="0.05")
        cb.update_equity(Decimal("10000"))
        cb.check(Decimal("9000"))  # trip it
        assert cb.is_tripped is True
        # Equity "recovers" — breaker must stay tripped
        assert cb.check(Decimal("10500")) is True

    def test_reset_clears_tripped_flag(self):
        cb = self._make_breaker(max_dd="0.05")
        cb.update_equity(Decimal("10000"))
        cb.check(Decimal("9000"))
        assert cb.is_tripped is True
        cb.reset()
        assert cb.is_tripped is False

    def test_trip_reason_set_on_drawdown(self):
        cb = self._make_breaker(max_dd="0.05")
        cb.update_equity(Decimal("10000"))
        cb.check(Decimal("9000"))
        assert "Drawdown" in cb.trip_reason
        assert "%" in cb.trip_reason

    def test_trip_reason_set_on_daily_loss(self):
        cb = self._make_breaker(max_dd="0.99", daily_loss="100")
        cb.update_equity(Decimal("10000"))
        cb.check(Decimal("9800"))  # $200 loss > $100 limit
        assert "Daily loss" in cb.trip_reason

    @pytest.mark.parametrize("drawdown_pct,should_trip", [
        ("0.01", False),
        ("0.04", False),
        ("0.05", True),  # exact boundary
        ("0.10", True),
    ])
    def test_drawdown_boundary_parametrize(self, drawdown_pct, should_trip):
        cb = self._make_breaker(max_dd="0.05")
        cb.update_equity(Decimal("10000"))
        equity = Decimal("10000") * (1 - Decimal(drawdown_pct))
        assert cb.check(equity) is should_trip


# ─────────────────────────────────────────────
#  Position Tests
# ─────────────────────────────────────────────


class TestPosition:
    def test_long_entry_sets_avg_cost(self):
        pos = Position("BTCUSDT")
        fill = make_fill("40000", "0.5")
        pos.update_fill(OrderSide.BUY, fill)
        assert pos.qty == Decimal("0.5")
        assert pos.avg_cost == Decimal("40000")

    def test_partial_close_realizes_pnl(self):
        pos = Position("BTCUSDT")
        pos.update_fill(OrderSide.BUY, make_fill("40000", "1.0"))
        # Close half at a higher price
        realized = pos.update_fill(OrderSide.SELL, make_fill("45000", "0.5"))
        # realized = 0.5 × (45000 - 40000) = 2500
        assert realized == pytest.approx(Decimal("2500"))
        assert pos.qty == Decimal("0.5")
        assert pos.realized_pnl == Decimal("2500")

    def test_full_close_sets_qty_zero(self):
        pos = Position("BTCUSDT")
        pos.update_fill(OrderSide.BUY, make_fill("40000", "1.0"))
        pos.update_fill(OrderSide.SELL, make_fill("42000", "1.0"))
        assert pos.qty == Decimal("0")
        assert pos.avg_cost == Decimal("0")

    def test_vwap_avg_cost_on_adding_to_position(self):
        pos = Position("BTCUSDT")
        pos.update_fill(OrderSide.BUY, make_fill("40000", "1.0"))
        pos.update_fill(OrderSide.BUY, make_fill("50000", "1.0"))
        # VWAP = (40000*1 + 50000*1) / 2 = 45000
        assert pos.avg_cost == pytest.approx(Decimal("45000"))
        assert pos.qty == Decimal("2.0")

    def test_unrealized_pnl_long_profitable(self):
        pos = Position("BTCUSDT")
        pos.update_fill(OrderSide.BUY, make_fill("40000", "0.5"))
        pnl = pos.unrealized_pnl(Decimal("45000"))
        # 0.5 × (45000 - 40000) = 2500
        assert pnl == pytest.approx(Decimal("2500"))

    def test_unrealized_pnl_long_loss(self):
        pos = Position("BTCUSDT")
        pos.update_fill(OrderSide.BUY, make_fill("40000", "1.0"))
        pnl = pos.unrealized_pnl(Decimal("38000"))
        assert pnl == pytest.approx(Decimal("-2000"))

    def test_unrealized_pnl_short_profitable(self):
        pos = Position("BTCUSDT")
        pos.update_fill(OrderSide.SELL, make_fill("50000", "1.0"))
        pnl = pos.unrealized_pnl(Decimal("45000"))
        # short: qty=-1, (45000-50000) * (-1) = 5000  → pnl = -1 * (45000 - 50000) = 5000
        assert pnl == pytest.approx(Decimal("5000"))

    def test_unrealized_pnl_flat_position(self):
        pos = Position("BTCUSDT")
        assert pos.unrealized_pnl(Decimal("50000")) == Decimal("0")

    def test_realized_pnl_accumulates(self):
        pos = Position("BTCUSDT")
        pos.update_fill(OrderSide.BUY, make_fill("40000", "2.0"))
        pos.update_fill(OrderSide.SELL, make_fill("41000", "1.0"))
        pos.update_fill(OrderSide.SELL, make_fill("42000", "1.0"))
        # First close: 1.0 × (41000-40000) = 1000
        # Second close: 1.0 × (42000-40000) = 2000
        assert pos.realized_pnl == pytest.approx(Decimal("3000"))
