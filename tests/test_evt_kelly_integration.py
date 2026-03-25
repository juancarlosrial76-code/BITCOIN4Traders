"""
Tests: EVT-Kelly Integration
==============================
Verifies that EVTRiskManager tail-risk signals dynamically scale down
Kelly position sizing inside RiskManager.

Coverage:
---------
1. compute_evt_kelly_scalar() tiers (ES_99 at every threshold boundary)
2. Position size is actually reduced when EVT is stressed
3. EVT scalar = 1.0 (no reduction) under benign conditions
4. update_evt() / get_evt_metrics() public API
5. EVT scalar applied inside validate_position_size()
"""

import numpy as np
import pytest
from unittest.mock import patch, PropertyMock

from src.risk.risk_manager import RiskManager, RiskConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_risk_manager() -> RiskManager:
    config = RiskConfig(
        max_drawdown_per_session=0.05,
        max_consecutive_losses=10,
        max_position_size=0.25,
        kelly_fraction=0.5,
        enable_circuit_breaker=False,  # keep circuit breaker out of the way
    )
    return RiskManager(config, initial_capital=100_000)


def _patch_es99(risk_mgr: RiskManager, es_99_value: float, is_critical: bool = False):
    """
    Context-manager that monkey-patches evt_manager.compute_evt_risk_metrics
    so we can inject a synthetic ES_99 value without needing 100+ observations.
    """
    metrics = {
        "VaR_99": es_99_value * 0.8,
        "ES_99": es_99_value,
        "is_critical": is_critical,
    }
    return patch.object(
        risk_mgr.evt_manager,
        "compute_evt_risk_metrics",
        return_value=metrics,
    )


# ---------------------------------------------------------------------------
# Test 1 – Scalar tiers for compute_evt_kelly_scalar()
# ---------------------------------------------------------------------------


class TestComputeEvtKellyScalar:
    """Unit-tests for the five ES_99 tiers."""

    def test_scalar_benign_regime(self):
        """ES_99 = 0.5% (well below 1%) → scalar must be exactly 1.0."""
        rm = make_risk_manager()
        with _patch_es99(rm, 0.005):
            scalar = rm.compute_evt_kelly_scalar()
        assert scalar == 1.0, f"Expected 1.0, got {scalar}"

    def test_scalar_mild_stress(self):
        """ES_99 = 1.5% (between 1% and 2%) → scalar = 0.75."""
        rm = make_risk_manager()
        with _patch_es99(rm, 0.015):
            scalar = rm.compute_evt_kelly_scalar()
        assert scalar == 0.75, f"Expected 0.75, got {scalar}"

    def test_scalar_moderate_stress(self):
        """ES_99 = 2.5% (between 2% and 3%) → scalar = 0.50."""
        rm = make_risk_manager()
        with _patch_es99(rm, 0.025):
            scalar = rm.compute_evt_kelly_scalar()
        assert scalar == 0.50, f"Expected 0.50, got {scalar}"

    def test_scalar_high_stress(self):
        """ES_99 = 4% (between 3% and 5%) → scalar = 0.25 (quartered)."""
        rm = make_risk_manager()
        with _patch_es99(rm, 0.04):
            scalar = rm.compute_evt_kelly_scalar()
        assert scalar == 0.25, f"Expected 0.25, got {scalar}"

    def test_scalar_critical_exactly_5pct(self):
        """ES_99 = 5% (at the boundary) → scalar = 0.25 (still in 3%–5% tier)."""
        rm = make_risk_manager()
        with _patch_es99(rm, 0.05):
            scalar = rm.compute_evt_kelly_scalar()
        # 0.05 > 0.03 and 0.05 <= 0.05, so returns 0.25
        assert scalar == 0.25, f"Expected 0.25 at exactly 5%, got {scalar}"

    def test_scalar_critical_above_5pct(self):
        """ES_99 > 5% → scalar = 0.0 (close all)."""
        rm = make_risk_manager()
        with _patch_es99(rm, 0.06, is_critical=True):
            scalar = rm.compute_evt_kelly_scalar()
        assert scalar == 0.0, f"Expected 0.0, got {scalar}"

    def test_scalar_critical_flag_overrides(self):
        """is_critical=True should force scalar=0.0 even at ES_99=3%."""
        rm = make_risk_manager()
        with _patch_es99(rm, 0.03, is_critical=True):
            scalar = rm.compute_evt_kelly_scalar()
        assert scalar == 0.0, f"is_critical should force 0.0, got {scalar}"


# ---------------------------------------------------------------------------
# Test 2 – Position size is actually reduced
# ---------------------------------------------------------------------------


class TestPositionSizeReduction:
    """Integration tests: EVT scalar flows through validate_position_size."""

    def test_position_unchanged_in_benign_regime(self):
        """Benign EVT (ES_99=0.5%) → position not reduced by EVT."""
        rm = make_risk_manager()
        with _patch_es99(rm, 0.005):
            approved, size = rm.validate_position_size(
                proposed_size=10_000,
                current_capital=100_000,
                win_probability=0.55,
                win_loss_ratio=1.5,
            )
        assert approved is True
        # In benign regime the EVT scalar is 1.0 so Kelly sizing is the only limit.
        # Just verify size > 0 and not reduced below Kelly floor.
        assert size > 0

    def test_position_reduced_under_high_stress(self):
        """High EVT stress (ES_99=4%) should produce a smaller position than benign."""
        rm_benign = make_risk_manager()
        rm_stressed = make_risk_manager()

        with _patch_es99(rm_benign, 0.005):
            _, size_benign = rm_benign.validate_position_size(
                proposed_size=25_000,
                current_capital=100_000,
                win_probability=0.55,
                win_loss_ratio=1.5,
            )

        with _patch_es99(rm_stressed, 0.04):
            _, size_stressed = rm_stressed.validate_position_size(
                proposed_size=25_000,
                current_capital=100_000,
                win_probability=0.55,
                win_loss_ratio=1.5,
            )

        assert size_stressed < size_benign, (
            f"Stressed position ({size_stressed:.0f}) should be smaller "
            f"than benign position ({size_benign:.0f})"
        )

    def test_position_zero_when_critical(self):
        """is_critical EVT → position size = 0, trade rejected."""
        rm = make_risk_manager()
        with _patch_es99(rm, 0.06, is_critical=True):
            approved, size = rm.validate_position_size(
                proposed_size=10_000,
                current_capital=100_000,
                win_probability=0.55,
                win_loss_ratio=1.5,
            )
        # EVT scalar is 0.0 → kelly_size * 0 = 0
        assert size == pytest.approx(
            0.0, abs=1e-6
        ), f"Critical EVT should zero out position, got {size}"

    def test_position_quartered_at_4pct_es99(self):
        """ES_99 = 4% maps to scalar=0.25; final position ≈ kelly_size * 0.25."""
        rm_ref = make_risk_manager()
        rm_evt = make_risk_manager()

        # Benign reference size
        with _patch_es99(rm_ref, 0.005):
            _, size_ref = rm_ref.validate_position_size(
                proposed_size=25_000,
                current_capital=100_000,
                win_probability=0.55,
                win_loss_ratio=1.5,
            )

        # Stressed size
        with _patch_es99(rm_evt, 0.04):
            _, size_stressed = rm_evt.validate_position_size(
                proposed_size=25_000,
                current_capital=100_000,
                win_probability=0.55,
                win_loss_ratio=1.5,
            )

        expected = size_ref * 0.25
        assert size_stressed == pytest.approx(
            expected, rel=1e-6
        ), f"Expected ~{expected:.2f}, got {size_stressed:.2f}"


# ---------------------------------------------------------------------------
# Test 3 – update_evt() and get_evt_metrics() API
# ---------------------------------------------------------------------------


class TestEvtPublicAPI:
    def test_update_evt_feeds_returns(self):
        """update_evt() should increase returns_history length."""
        rm = make_risk_manager()
        initial_len = len(rm.evt_manager.returns_history)
        rm.update_evt(-0.01)
        rm.update_evt(0.005)
        assert len(rm.evt_manager.returns_history) == initial_len + 2

    def test_get_evt_metrics_returns_dict(self):
        """get_evt_metrics() must return a dict with expected keys."""
        rm = make_risk_manager()
        metrics = rm.get_evt_metrics()
        assert isinstance(metrics, dict)
        assert "VaR_99" in metrics
        assert "ES_99" in metrics
        assert "is_critical" in metrics
        assert "evt_kelly_scalar" in metrics

    def test_get_evt_metrics_scalar_matches_compute(self):
        """evt_kelly_scalar in metrics matches direct compute_evt_kelly_scalar()."""
        rm = make_risk_manager()
        with _patch_es99(rm, 0.025):
            metrics = rm.get_evt_metrics()
            direct = rm.compute_evt_kelly_scalar()
        assert metrics["evt_kelly_scalar"] == direct

    def test_evt_manager_initialized(self):
        """RiskManager must expose an evt_manager of the correct type."""
        from src.risk.evt import EVTRiskManager

        rm = make_risk_manager()
        assert hasattr(rm, "evt_manager")
        assert isinstance(rm.evt_manager, EVTRiskManager)

    def test_get_evt_metrics_insufficient_data(self):
        """With < 100 observations EVT returns safe defaults (no crash)."""
        rm = make_risk_manager()
        # Feed 50 returns — not enough for EVT to fit GPD
        for _ in range(50):
            rm.update_evt(np.random.randn() * 0.01)
        metrics = rm.get_evt_metrics()
        # Should not raise; defaults should be zero / False
        assert metrics["ES_99"] == pytest.approx(0.0)
        assert metrics["is_critical"] is False
        # Scalar should be 1.0 (no reduction) when ES_99 == 0.0
        assert metrics["evt_kelly_scalar"] == 1.0


# ---------------------------------------------------------------------------
# Test 4 – End-to-end with real EVT data (smoke test)
# ---------------------------------------------------------------------------


class TestEndToEndRealEvt:
    def test_evt_scalar_with_real_returns(self):
        """
        Feed 200 synthetic high-volatility returns so EVT actually runs.
        Verifies no crashes and that the scalar is in [0, 1].
        """
        np.random.seed(42)
        rm = make_risk_manager()

        # Simulate a volatile period
        for _ in range(200):
            # Mix of normal daily returns + occasional large losses
            r = np.random.choice(
                [np.random.randn() * 0.01, -np.random.uniform(0.03, 0.08)],
                p=[0.85, 0.15],
            )
            rm.update_evt(float(r))

        scalar = rm.compute_evt_kelly_scalar()
        assert 0.0 <= scalar <= 1.0, f"Scalar out of bounds: {scalar}"

    def test_validate_position_size_with_real_evt(self):
        """
        Full pipeline smoke test: position size stays non-negative after
        validate_position_size() with real EVT history.
        """
        np.random.seed(0)
        rm = make_risk_manager()

        for _ in range(200):
            r = np.random.randn() * 0.015
            rm.update_evt(float(r))

        approved, size = rm.validate_position_size(
            proposed_size=20_000,
            current_capital=100_000,
            win_probability=0.55,
            win_loss_ratio=1.5,
        )
        assert approved is True
        assert size >= 0.0
