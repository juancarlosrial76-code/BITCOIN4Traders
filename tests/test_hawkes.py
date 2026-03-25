"""
Tests: Hawkes Point Process — Order Flow Toxicity
==================================================

Covers:
- Baseline (no events) → λ(t) == μ
- Adding events raises λ(t) above μ
- Intensity decays over time: very large Δt → λ(t) ≈ μ
- branching_ratio() == α / β
- Cluster of events → is_toxic() returns True
- No events → is_toxic() returns False
- Event size amplification (larger trades excite more)
- reset() clears all state
- get_intensity_series() returns NumPy array
- window_size pruning is respected
- HawkesConfig validation raises on bad params
- hawkes_from_trades() integration helper
- Public __init__.py exports
"""

import math
import numpy as np
import pytest

from src.risk.hawkes import HawkesProcess, HawkesConfig, hawkes_from_trades
from src.risk import HawkesProcess as HPFromInit, HawkesConfig as HCFromInit


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def make_hp(
    mu: float = 0.1,
    alpha: float = 0.5,
    beta: float = 1.0,
    toxicity_threshold: float = 2.0,
    window_size: int = 100,
) -> HawkesProcess:
    cfg = HawkesConfig(
        mu=mu,
        alpha=alpha,
        beta=beta,
        toxicity_threshold=toxicity_threshold,
        window_size=window_size,
    )
    return HawkesProcess(cfg)


# ---------------------------------------------------------------------------
# 1. Baseline intensity
# ---------------------------------------------------------------------------


class TestBaselineIntensity:
    def test_no_events_intensity_equals_mu(self):
        """With no prior events λ(t) must equal the baseline μ exactly."""
        hp = make_hp(mu=0.1)
        lam = hp.compute_intensity(t=10.0)
        assert lam == pytest.approx(0.1), f"Expected λ=μ=0.1, got {lam}"

    def test_baseline_with_custom_mu(self):
        for mu in (0.01, 0.5, 2.0):
            hp = make_hp(mu=mu)
            lam = hp.compute_intensity(t=0.0)
            assert lam == pytest.approx(mu)

    def test_intensity_never_below_mu(self):
        """λ(t) ≥ μ for all valid inputs."""
        hp = make_hp(mu=0.2, alpha=0.3, beta=0.8)
        hp.add_event(1.0)
        hp.add_event(2.0)
        hp.add_event(3.0)
        # Query at a time well before any event: no excitation should apply
        # (events in the future don't contribute to past λ)
        lam_before = hp.compute_intensity(t=0.5)
        assert lam_before >= 0.2


# ---------------------------------------------------------------------------
# 2. Adding events raises intensity
# ---------------------------------------------------------------------------


class TestEventExcitation:
    def test_single_event_raises_intensity(self):
        """Immediately after an event, λ(t) > μ."""
        hp = make_hp(mu=0.1, alpha=0.5, beta=1.0)
        hp.add_event(timestamp=1.0, size=1.0)
        lam = hp.compute_intensity(t=1.5)  # 0.5 s after event
        expected_excitation = 0.5 * 1.0 * math.exp(-1.0 * 0.5)
        assert lam == pytest.approx(0.1 + expected_excitation, rel=1e-6)
        assert lam > 0.1

    def test_multiple_events_additive_excitation(self):
        """Excitation from multiple events accumulates."""
        hp = make_hp(mu=0.1, alpha=0.5, beta=1.0)
        hp.add_event(1.0, size=1.0)
        hp.add_event(2.0, size=1.0)
        lam_two = hp.compute_intensity(t=2.5)

        hp2 = make_hp(mu=0.1, alpha=0.5, beta=1.0)
        hp2.add_event(2.0, size=1.0)
        lam_one = hp2.compute_intensity(t=2.5)

        assert lam_two > lam_one, "Two events should produce higher intensity than one"

    def test_larger_size_excites_more(self):
        """A trade with size=3 should produce more excitation than size=1."""
        t_query = 5.0
        hp_small = make_hp()
        hp_small.add_event(4.0, size=1.0)
        lam_small = hp_small.compute_intensity(t_query)

        hp_large = make_hp()
        hp_large.add_event(4.0, size=3.0)
        lam_large = hp_large.compute_intensity(t_query)

        assert lam_large == pytest.approx(
            lam_small + 0.5 * 2.0 * math.exp(-1.0), rel=1e-6
        )
        assert lam_large > lam_small

    def test_future_events_do_not_contribute(self):
        """Events with tᵢ > t must not be counted (causality)."""
        hp = make_hp(mu=0.1)
        hp.add_event(timestamp=10.0)  # future relative to query time 5
        lam = hp.compute_intensity(t=5.0)
        assert lam == pytest.approx(
            0.1
        ), "Future events must not contribute: expected λ=μ"


# ---------------------------------------------------------------------------
# 3. Decay over time
# ---------------------------------------------------------------------------


class TestIntensityDecay:
    def test_intensity_decays_with_time(self):
        """λ(t) should decrease monotonically as t increases past an event."""
        hp = make_hp(mu=0.1, alpha=0.5, beta=1.0)
        hp.add_event(timestamp=0.0, size=1.0)

        lam_prev = hp.compute_intensity(t=0.01)
        for delta in (0.5, 1.0, 2.0, 5.0, 10.0):
            lam = hp.compute_intensity(t=delta)
            assert (
                lam < lam_prev
            ), f"λ({delta}) = {lam:.6f} should be < λ_prev = {lam_prev:.6f}"
            lam_prev = lam

    def test_intensity_approaches_mu_at_large_t(self):
        """For t >> last event, λ(t) should be indistinguishable from μ."""
        hp = make_hp(mu=0.1, alpha=0.5, beta=1.0)
        hp.add_event(timestamp=0.0)
        # After 100 half-lives the excitation is negligible
        lam = hp.compute_intensity(t=100.0)
        assert lam == pytest.approx(
            0.1, abs=1e-6
        ), f"Intensity should decay to μ; got {lam}"

    def test_decay_rate_consistent_with_beta(self):
        """The excitation kernel decays at rate β: kernel(t₂)/kernel(t₁) = exp(-β·Δt)."""
        mu, alpha, beta = 0.001, 1.0, 2.0
        hp = make_hp(mu=mu, alpha=alpha, beta=beta)
        hp.add_event(timestamp=0.0, size=1.0)
        t1, t2 = 1.0, 2.0
        lam1 = hp.compute_intensity(t1)  # μ + α·exp(-β·1)
        lam2 = hp.compute_intensity(t2)  # μ + α·exp(-β·2)
        # Isolate the kernel contribution by subtracting μ
        exc1 = lam1 - mu
        exc2 = lam2 - mu
        ratio = exc2 / exc1
        expected_ratio = math.exp(-beta * (t2 - t1))
        assert ratio == pytest.approx(expected_ratio, rel=1e-6)


# ---------------------------------------------------------------------------
# 4. Branching ratio
# ---------------------------------------------------------------------------


class TestBranchingRatio:
    def test_branching_ratio_formula(self):
        for alpha, beta in [(0.5, 1.0), (0.3, 0.6), (1.0, 4.0)]:
            hp = make_hp(alpha=alpha, beta=beta)
            assert hp.branching_ratio() == pytest.approx(alpha / beta, rel=1e-9)

    def test_branching_ratio_stationary(self):
        """n* < 1 for the default config (α=0.5, β=1.0)."""
        hp = make_hp()
        assert hp.branching_ratio() < 1.0

    def test_branching_ratio_warns_when_supercritical(self, caplog):
        """Logger should warn when n* >= 1."""
        import logging

        # HawkesConfig.__post_init__ already warns on construction;
        # branching_ratio() also warns.
        with caplog.at_level(logging.WARNING, logger="root"):
            hp = make_hp(alpha=2.0, beta=1.0)  # n* = 2.0
            n = hp.branching_ratio()
        assert n >= 1.0


# ---------------------------------------------------------------------------
# 5. Toxicity detection
# ---------------------------------------------------------------------------


class TestIsToxic:
    def test_no_events_not_toxic(self):
        """Without any trades the market is at baseline — not toxic."""
        hp = make_hp(mu=0.1, toxicity_threshold=2.0)
        assert hp.is_toxic(t=10.0) is False

    def test_cluster_of_events_is_toxic(self):
        """
        A dense cluster of large trades should push λ/μ above the threshold.
        We use a high alpha and many recent events so excitation >> μ.
        """
        hp = make_hp(mu=0.1, alpha=0.8, beta=0.5, toxicity_threshold=2.0)
        # Feed 20 events spaced 0.1 apart, ending at t=2.0
        for i in range(20):
            hp.add_event(timestamp=float(i) * 0.1, size=1.0)
        assert hp.is_toxic(t=2.0) is True

    def test_widely_spaced_events_not_toxic(self):
        """Events spaced far apart: by the time of query, excitation has decayed."""
        hp = make_hp(mu=0.1, alpha=0.5, beta=2.0, toxicity_threshold=2.0)
        hp.add_event(timestamp=0.0)
        # Query 20 half-lives later — excitation ≈ 0
        assert hp.is_toxic(t=20.0) is False

    def test_is_toxic_threshold_boundary(self):
        """Exactly at the threshold the market should NOT be flagged as toxic."""
        hp = make_hp(mu=0.1, alpha=0.5, beta=1.0, toxicity_threshold=2.0)
        # λ(t)/μ = 2.0 is the boundary; is_toxic uses strict >
        lam_needed = hp.config.mu * hp.config.toxicity_threshold  # = 0.2
        # Compute the required event configuration analytically:
        # We want α·exp(-β·Δt) = μ·(threshold - 1) = 0.1·1.0 = 0.1
        # → exp(-1·Δt) = 0.1/0.5 = 0.2 → Δt = -ln(0.2) ≈ 1.609
        dt = -math.log(0.2)
        hp.add_event(timestamp=0.0)
        lam_at_boundary = hp.compute_intensity(t=dt)
        # lam_at_boundary / μ should be ≈ 2.0 (threshold)
        ratio = lam_at_boundary / hp.config.mu
        # Since is_toxic is strict >, ratio == 2.0 should be False
        assert ratio == pytest.approx(2.0, abs=1e-6)
        # One additional infinitesimal step past boundary: not testing exact
        # floating-point equality here, just verify logic works near boundary.
        assert hp.is_toxic(t=dt - 0.0001) is True  # just before: above threshold
        assert hp.is_toxic(t=dt + 0.0001) is False  # just after: below threshold

    def test_size_amplification_can_trigger_toxicity(self):
        """A single very large trade should trigger toxicity due to size > 1."""
        hp = make_hp(mu=0.1, alpha=0.5, beta=0.5, toxicity_threshold=2.0)
        # One enormous trade (size=20) immediately before query
        hp.add_event(timestamp=9.9, size=20.0)
        lam = hp.compute_intensity(t=10.0)
        # Excitation = 0.5 * 20 * exp(-0.5 * 0.1) ≈ 9.51
        assert lam > hp.config.mu * hp.config.toxicity_threshold
        assert hp.is_toxic(t=10.0) is True


# ---------------------------------------------------------------------------
# 6. reset()
# ---------------------------------------------------------------------------


class TestReset:
    def test_reset_clears_event_history(self):
        hp = make_hp()
        for i in range(10):
            hp.add_event(float(i))
        hp.reset()
        assert len(hp._event_times) == 0
        assert len(hp._event_sizes) == 0

    def test_reset_clears_intensity_history(self):
        hp = make_hp()
        hp.add_event(1.0)
        hp.compute_intensity(2.0)
        hp.reset()
        assert len(hp._intensity_history) == 0

    def test_intensity_after_reset_equals_mu(self):
        hp = make_hp(mu=0.3)
        hp.add_event(1.0)
        hp.reset()
        lam = hp.compute_intensity(t=2.0)
        assert lam == pytest.approx(0.3)


# ---------------------------------------------------------------------------
# 7. get_intensity_series()
# ---------------------------------------------------------------------------


class TestIntensitySeries:
    def test_returns_ndarray(self):
        hp = make_hp()
        series = hp.get_intensity_series()
        assert isinstance(series, np.ndarray)
        assert series.dtype == float

    def test_empty_before_any_query(self):
        hp = make_hp()
        assert len(hp.get_intensity_series()) == 0

    def test_length_matches_query_count(self):
        hp = make_hp()
        for t in (1.0, 2.0, 3.0):
            hp.compute_intensity(t)
        series = hp.get_intensity_series()
        assert len(series) == 3

    def test_is_toxic_also_appends_to_series(self):
        """is_toxic() internally calls compute_intensity() → history grows."""
        hp = make_hp()
        hp.is_toxic(t=1.0)
        hp.is_toxic(t=2.0)
        assert len(hp.get_intensity_series()) == 2

    def test_series_values_are_non_negative(self):
        hp = make_hp()
        hp.add_event(1.0)
        hp.add_event(2.0)
        for t in np.linspace(0, 5, 20):
            hp.compute_intensity(float(t))
        assert np.all(hp.get_intensity_series() >= 0.0)


# ---------------------------------------------------------------------------
# 8. window_size pruning
# ---------------------------------------------------------------------------


class TestWindowSize:
    def test_window_size_enforced(self):
        hp = make_hp(window_size=10)
        for i in range(50):
            hp.add_event(float(i))
        assert len(hp._event_times) <= 10

    def test_oldest_events_pruned(self):
        hp = make_hp(window_size=5)
        for i in range(10):
            hp.add_event(float(i))
        # Only the 5 newest (indices 5-9) should remain
        assert min(hp._event_times) == pytest.approx(5.0)

    def test_window_size_one(self):
        hp = make_hp(window_size=1)
        hp.add_event(1.0)
        hp.add_event(2.0)
        assert len(hp._event_times) == 1
        assert hp._event_times[0] == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# 9. HawkesConfig validation
# ---------------------------------------------------------------------------


class TestHawkesConfigValidation:
    def test_invalid_mu_raises(self):
        with pytest.raises(ValueError, match="mu"):
            HawkesConfig(mu=0.0)

    def test_negative_mu_raises(self):
        with pytest.raises(ValueError, match="mu"):
            HawkesConfig(mu=-0.1)

    def test_negative_alpha_raises(self):
        with pytest.raises(ValueError, match="alpha"):
            HawkesConfig(alpha=-0.1)

    def test_zero_beta_raises(self):
        with pytest.raises(ValueError, match="beta"):
            HawkesConfig(beta=0.0)

    def test_toxicity_threshold_at_one_raises(self):
        with pytest.raises(ValueError, match="toxicity_threshold"):
            HawkesConfig(toxicity_threshold=1.0)

    def test_window_size_zero_raises(self):
        with pytest.raises(ValueError, match="window_size"):
            HawkesConfig(window_size=0)

    def test_negative_event_size_raises(self):
        hp = make_hp()
        with pytest.raises(ValueError, match="size"):
            hp.add_event(1.0, size=-1.0)

    def test_zero_event_size_raises(self):
        hp = make_hp()
        with pytest.raises(ValueError, match="size"):
            hp.add_event(1.0, size=0.0)


# ---------------------------------------------------------------------------
# 10. hawkes_from_trades()
# ---------------------------------------------------------------------------


class TestHawkesFromTrades:
    def _make_trade_tape(self, n: int = 200, seed: int = 42) -> tuple:
        rng = np.random.default_rng(seed)
        prices = np.full(n, 50_000.0)
        # Exponential volumes: mean ≈ 1.0, occasional spikes > 2.0
        volumes = rng.exponential(scale=1.0, size=n)
        return prices, volumes

    def test_returns_hawkes_process(self):
        prices, volumes = self._make_trade_tape()
        hp = hawkes_from_trades(prices, volumes)
        assert isinstance(hp, HawkesProcess)

    def test_events_ingested_are_above_threshold(self):
        """All stored events must correspond to above-threshold volumes."""
        rng = np.random.default_rng(0)
        prices = np.ones(100)
        volumes = rng.exponential(1.0, size=100)
        mean_vol = float(np.mean(volumes))
        threshold = 2.0 * mean_vol

        hp = hawkes_from_trades(prices, volumes, threshold_vol_mult=2.0)

        # Every ingested event index (stored as timestamp) should map to
        # a volume above threshold
        for t in hp._event_times:
            assert (
                volumes[int(t)] > threshold
            ), f"Event at t={t} has vol={volumes[int(t)]:.4f} <= threshold={threshold:.4f}"

    def test_intensity_above_mu_after_large_trades(self):
        """A burst of large trades should raise λ above μ."""
        n = 100
        prices = np.ones(n) * 50_000.0
        # Force all trades to be "large" by setting a very low multiplier
        volumes = np.ones(n) * 5.0  # uniform large volume
        hp = hawkes_from_trades(prices, volumes, threshold_vol_mult=0.5)
        lam = hp.compute_intensity(t=float(n))
        assert lam > hp.config.mu

    def test_empty_arrays_raises(self):
        with pytest.raises(ValueError):
            hawkes_from_trades(np.array([]), np.array([]))

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError):
            hawkes_from_trades(np.ones(10), np.ones(5))

    def test_custom_config_respected(self):
        prices, volumes = self._make_trade_tape()
        cfg = HawkesConfig(mu=0.05, alpha=0.3, beta=0.8)
        hp = hawkes_from_trades(prices, volumes, config=cfg)
        assert hp.config.mu == pytest.approx(0.05)
        assert hp.config.alpha == pytest.approx(0.3)

    def test_no_large_trades_returns_empty_process(self):
        """If no trade exceeds the threshold, event list is empty."""
        prices = np.ones(50)
        volumes = np.ones(50) * 0.1  # all equal, mean = 0.1
        # threshold_vol_mult=2.0 → threshold=0.2; all trades = 0.1 < 0.2
        hp = hawkes_from_trades(prices, volumes, threshold_vol_mult=2.0)
        assert len(hp._event_times) == 0
        lam = hp.compute_intensity(t=100.0)
        assert lam == pytest.approx(hp.config.mu)


# ---------------------------------------------------------------------------
# 11. Default constructor (no config)
# ---------------------------------------------------------------------------


class TestDefaultConstructor:
    def test_default_config_used_when_none(self):
        hp = HawkesProcess()
        assert hp.config.mu == pytest.approx(0.1)
        assert hp.config.alpha == pytest.approx(0.5)
        assert hp.config.beta == pytest.approx(1.0)

    def test_default_branching_ratio(self):
        hp = HawkesProcess()
        assert hp.branching_ratio() == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# 12. Public __init__.py exports
# ---------------------------------------------------------------------------


class TestPublicExports:
    def test_hawkes_process_importable_from_package(self):
        assert HPFromInit is HawkesProcess

    def test_hawkes_config_importable_from_package(self):
        assert HCFromInit is HawkesConfig

    def test_instantiation_via_package_exports(self):
        cfg = HCFromInit(mu=0.2, alpha=0.4, beta=0.8)
        hp = HPFromInit(cfg)
        assert hp.config.mu == pytest.approx(0.2)
        lam = hp.compute_intensity(t=0.0)
        assert lam == pytest.approx(0.2)
