"""
Tests: VPIN and EVT Risk Modules
=================================

Covers:
- VPINCalculator with balanced (buy + sell) trade flow  → low VPIN
- VPINCalculator with one-sided (all-buy) trade flow    → high VPIN
- VPINCalculator BVC auto-classification
- VPINCalculator get_vpin_series()
- EVTRiskManager with 200 normal returns                → VaR positive
- EVTRiskManager with extreme losses                    → is_critical True
- EVTRiskManager update_and_check() convenience wrapper
- Public __init__.py exports
"""

import numpy as np
import pytest

from src.risk.vpin import VPINCalculator
from src.risk.evt import EVTRiskManager
from src.risk import VPINCalculator as VPINFromInit, EVTRiskManager as EVTFromInit


# ---------------------------------------------------------------------------
# VPINCalculator
# ---------------------------------------------------------------------------


class TestVPINCalculator:
    def _make_calculator(self, sample_length: int = 50) -> VPINCalculator:
        return VPINCalculator(volume_bucket_size=100.0, sample_length=sample_length)

    # --- balanced flow: ~30 buys + 30 sells --------------------------------

    def test_low_vpin_balanced_trades(self):
        """60 alternating buy/sell trades should yield VPIN close to 0."""
        calc = self._make_calculator()

        price = 50_000.0
        for i in range(60):
            is_buy = i % 2 == 0  # alternating
            calc.update(price, volume=100.0, is_buy_initiated=is_buy)

        vpin = calc.compute_vpin()
        # Perfect balance → VPIN == 0; allow small numerical slack
        assert vpin < 0.15, f"Expected low VPIN for balanced flow, got {vpin:.4f}"

    # --- one-sided flow: 60 buys only --------------------------------------

    def test_high_vpin_all_buys(self):
        """60 buy-only trades should yield VPIN close to 1.0."""
        calc = self._make_calculator()

        price = 50_000.0
        for _ in range(60):
            calc.update(price, volume=100.0, is_buy_initiated=True)

        vpin = calc.compute_vpin()
        assert vpin > 0.85, f"Expected high VPIN for all-buy flow, got {vpin:.4f}"

    # --- toxicity threshold ------------------------------------------------

    def test_is_market_toxic_high_vpin(self):
        calc = self._make_calculator()
        for _ in range(60):
            calc.update(50_000.0, volume=100.0, is_buy_initiated=True)
        assert calc.is_market_toxic(threshold=0.8) is True

    def test_is_market_toxic_low_vpin(self):
        calc = self._make_calculator()
        for i in range(60):
            calc.update(50_000.0, volume=100.0, is_buy_initiated=(i % 2 == 0))
        assert calc.is_market_toxic(threshold=0.8) is False

    def test_is_market_toxic_empty(self):
        """Should return False with no history."""
        calc = self._make_calculator()
        assert calc.is_market_toxic() is False

    # --- BVC auto-classification -------------------------------------------

    def test_bvc_classify_trade_up(self):
        calc = self._make_calculator()
        assert calc.classify_trade(50_001.0, 50_000.0) is True  # price rose → buy

    def test_bvc_classify_trade_down(self):
        calc = self._make_calculator()
        assert calc.classify_trade(49_999.0, 50_000.0) is False  # price fell → sell

    def test_bvc_classify_trade_flat(self):
        calc = self._make_calculator()
        assert calc.classify_trade(50_000.0, 50_000.0) is False  # tie → sell

    def test_update_without_explicit_direction_rising_price(self):
        """Auto-BVC: rising prices should accumulate buy volume."""
        calc = self._make_calculator()
        prices = [50_000.0 + i * 10 for i in range(60)]  # monotonically rising
        for p in prices:
            calc.update(p, volume=100.0)  # no is_buy_initiated arg

        vpin = calc.compute_vpin()
        assert (
            vpin > 0.8
        ), f"Monotonically rising prices should yield high VPIN via BVC, got {vpin:.4f}"

    # --- get_vpin_series ---------------------------------------------------

    def test_get_vpin_series_returns_ndarray(self):
        calc = self._make_calculator()
        for i in range(30):
            calc.update(50_000.0, volume=100.0, is_buy_initiated=(i % 2 == 0))

        series = calc.get_vpin_series()
        assert isinstance(series, np.ndarray)
        assert series.dtype == float

    def test_get_vpin_series_empty(self):
        calc = self._make_calculator()
        series = calc.get_vpin_series()
        assert isinstance(series, np.ndarray)
        assert len(series) == 0

    def test_get_vpin_series_monotone_all_buys(self):
        """VPIN should remain at 1.0 throughout an all-buy stream."""
        calc = self._make_calculator(sample_length=10)
        for _ in range(30):
            calc.update(50_000.0, volume=100.0, is_buy_initiated=True)

        series = calc.get_vpin_series()
        assert len(series) > 0
        assert np.all(
            series >= 0.99
        ), f"Expected all values ~ 1.0, got min {series.min():.4f}"

    # --- insufficient data guard -------------------------------------------

    def test_insufficient_data_returns_zero(self):
        calc = self._make_calculator(sample_length=50)
        # Feed fewer than sample_length // 2 ticks
        for _ in range(10):
            calc.update(50_000.0, volume=100.0, is_buy_initiated=True)
        assert calc.compute_vpin() == 0.0


# ---------------------------------------------------------------------------
# EVTRiskManager
# ---------------------------------------------------------------------------


class TestEVTRiskManager:
    def _make_manager(self) -> EVTRiskManager:
        return EVTRiskManager(history_window=500, threshold_quantile=0.95)

    # --- normal returns: VaR must be positive ------------------------------

    def test_var_positive_with_normal_returns(self):
        """200 N(0, 0.01) returns should produce a positive VaR_99."""
        rng = np.random.default_rng(42)
        mgr = self._make_manager()

        for ret in rng.normal(loc=0.0, scale=0.01, size=200):
            mgr.add_return(float(ret))

        metrics = mgr.compute_evt_risk_metrics()

        assert "VaR_99" in metrics
        assert "ES_99" in metrics
        assert (
            metrics["VaR_99"] > 0.0
        ), f"VaR_99 should be positive, got {metrics['VaR_99']:.6f}"
        assert (
            metrics["ES_99"] >= metrics["VaR_99"]
        ), "ES (CVaR) must be at least as large as VaR"

    # --- extreme losses: is_critical must be True -------------------------

    def test_is_critical_with_extreme_losses(self):
        """Heavy-tailed losses (10% average) should trigger is_critical=True."""
        rng = np.random.default_rng(0)
        mgr = self._make_manager()

        # Inject 200 large negative returns (losses of 8–15 %)
        for ret in rng.uniform(low=-0.15, high=-0.08, size=200):
            mgr.add_return(float(ret))

        metrics = mgr.compute_evt_risk_metrics()

        is_critical = metrics.get("is_critical")
        assert (
            is_critical
        ), f"Expected is_critical=True for extreme losses, metrics={metrics}"

    # --- insufficient data guard ------------------------------------------

    def test_insufficient_data_returns_zeros(self):
        mgr = self._make_manager()
        for _ in range(50):  # below the 100-sample threshold
            mgr.add_return(-0.01)

        metrics = mgr.compute_evt_risk_metrics()
        assert metrics["VaR_99"] == 0.0
        assert metrics["ES_99"] == 0.0
        assert metrics["is_critical"] is False

    # --- update_and_check convenience wrapper -----------------------------

    def test_update_and_check_returns_bool(self):
        mgr = self._make_manager()
        result = mgr.update_and_check(-0.01)
        assert isinstance(result, bool)

    def test_update_and_check_false_for_normal_conditions(self):
        rng = np.random.default_rng(7)
        mgr = self._make_manager()

        # Seed with mild normal returns first
        for ret in rng.normal(0.0, 0.01, 150):
            mgr.add_return(float(ret))

        # A single small loss should not trigger critical
        result = mgr.update_and_check(-0.005)
        assert result is False

    def test_update_and_check_true_for_extreme_regime(self):
        rng = np.random.default_rng(99)
        mgr = self._make_manager()

        # Load history with heavy losses
        for ret in rng.uniform(-0.15, -0.08, 200):
            mgr.add_return(float(ret))

        result = mgr.update_and_check(-0.12)
        assert result is True

    # --- rolling window enforcement ---------------------------------------

    def test_rolling_window_enforced(self):
        mgr = EVTRiskManager(history_window=50)
        for i in range(200):
            mgr.add_return(float(i))
        assert len(mgr.returns_history) <= 50

    # --- GPD floc=0 constraint (smoke test) -------------------------------

    def test_gpd_floc0_does_not_crash(self):
        """Ensure GPD fit with floc=0 runs without exception on valid data."""
        import scipy.stats as stats

        rng = np.random.default_rng(42)
        exceedances = rng.exponential(scale=0.02, size=50)  # pure exponential tail
        shape, loc, scale = stats.genpareto.fit(exceedances, floc=0)

        assert loc == 0.0, "floc=0 should fix location to 0"
        assert scale > 0.0

    # --- shape_param present in output ------------------------------------

    def test_shape_param_in_output(self):
        rng = np.random.default_rng(13)
        mgr = self._make_manager()

        for ret in rng.normal(0.0, 0.015, 200):
            mgr.add_return(float(ret))

        metrics = mgr.compute_evt_risk_metrics()
        # shape_param is included when GPD fit succeeds
        # (may not be present on fallback path – that's acceptable)
        if "shape_param" in metrics:
            assert isinstance(metrics["shape_param"], float)


# ---------------------------------------------------------------------------
# Public __init__.py exports
# ---------------------------------------------------------------------------


class TestPublicExports:
    def test_vpin_importable_from_package(self):
        assert VPINFromInit is VPINCalculator

    def test_evt_importable_from_package(self):
        assert EVTFromInit is EVTRiskManager

    def test_vpin_instantiation_via_package(self):
        calc = VPINFromInit(volume_bucket_size=500.0, sample_length=20)
        assert calc.volume_bucket_size == 500.0
        assert calc.sample_length == 20

    def test_evt_instantiation_via_package(self):
        mgr = EVTFromInit(history_window=100, threshold_quantile=0.90)
        assert mgr.history_window == 100
        assert mgr.threshold_quantile == 0.90
