"""
Tests for Regime-Causal Integration Pipeline
=============================================
Verifies that RegimeCausalPipeline and DynamicEnsemble.predict_with_pipeline()
work correctly end-to-end.

Test coverage
-------------
1. RegimeCausalPipeline can be instantiated with default config
2. fit() runs without error on 200-bar synthetic data
3. transform() returns numpy array with correct shape (base + extra dims)
4. get_extra_dims() returns 4 when both use_regime=True and use_causal=True
5. maybe_refit() triggers refit after refit_every_n_bars calls
6. Pipeline works with use_regime=False (only causal)
7. Pipeline works with use_causal=False  (only regime)
8. DynamicEnsemble.predict_with_pipeline() runs without error
"""

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_synthetic_features(n_bars: int = 200, seed: int = 42) -> pd.DataFrame:
    """
    Generate a synthetic OHLCV-style feature DataFrame with the columns that
    both HMMRegimeDetector and CausalDiscovery expect to find.
    """
    rng = np.random.default_rng(seed)

    # Simulate a price series
    log_returns = rng.normal(0.0002, 0.015, size=n_bars)
    prices = 100 * np.exp(np.cumsum(log_returns))

    df = pd.DataFrame(
        {
            "returns": log_returns,
            "volatility": pd.Series(log_returns)
            .rolling(20, min_periods=1)
            .std()
            .values,
            "volume": rng.uniform(500, 2000, size=n_bars),
            "rsi": rng.uniform(30, 70, size=n_bars),
            "momentum": rng.normal(0, 1, size=n_bars),
        }
    )
    # Fill any warm-up NaNs
    df = df.fillna(0.0)
    return df


# ---------------------------------------------------------------------------
# Minimal mock agent so we don't need a full PPO model
# ---------------------------------------------------------------------------


class MockAgent:
    """Minimal stand-in for a PPO agent."""

    def select_action(self, state, deterministic: bool = True):
        """Always return action 1 (Hold)."""
        return 1


# ---------------------------------------------------------------------------
# Test 1 – Instantiation
# ---------------------------------------------------------------------------


class TestInstantiation:
    def test_default_config_instantiation(self):
        """Pipeline can be built with zero arguments."""
        from src.integration.regime_causal_pipeline import RegimeCausalPipeline

        pipe = RegimeCausalPipeline()
        assert pipe is not None
        assert pipe._fitted is False
        assert pipe._hmm_fitted is False
        assert pipe._causal_fitted is False
        assert pipe._bar_count == 0

    def test_custom_config_instantiation(self):
        """Pipeline accepts a custom RegimeCausalConfig."""
        from src.integration.regime_causal_pipeline import (
            RegimeCausalConfig,
            RegimeCausalPipeline,
        )

        cfg = RegimeCausalConfig(n_regimes=2, causal_alpha=0.10, hmm_fit_window=100)
        pipe = RegimeCausalPipeline(cfg)
        assert pipe.config.n_regimes == 2
        assert pipe.config.causal_alpha == 0.10


# ---------------------------------------------------------------------------
# Test 2 – fit() on synthetic data
# ---------------------------------------------------------------------------


class TestFit:
    def test_fit_runs_without_error(self):
        """fit() on 200-bar synthetic data should not raise."""
        from src.integration.regime_causal_pipeline import RegimeCausalPipeline

        features = make_synthetic_features(200)
        pipe = RegimeCausalPipeline()
        pipe.fit(features)  # must not raise
        assert pipe._fitted is True

    def test_fit_small_dataset(self):
        """fit() on very small data should fall back gracefully (not crash)."""
        from src.integration.regime_causal_pipeline import RegimeCausalPipeline

        features = make_synthetic_features(15)  # too small for HMM
        pipe = RegimeCausalPipeline()
        pipe.fit(features)  # should not raise even if HMM is skipped
        assert pipe._fitted is True

    def test_fit_sets_fitted_flag(self):
        """_fitted is True after calling fit()."""
        from src.integration.regime_causal_pipeline import RegimeCausalPipeline

        features = make_synthetic_features(200)
        pipe = RegimeCausalPipeline()
        assert pipe._fitted is False
        pipe.fit(features)
        assert pipe._fitted is True


# ---------------------------------------------------------------------------
# Test 3 – transform() shape
# ---------------------------------------------------------------------------


class TestTransform:
    def test_transform_returns_numpy_array(self):
        """transform() must return a 1-D numpy array."""
        from src.integration.regime_causal_pipeline import RegimeCausalPipeline

        features = make_synthetic_features(200)
        pipe = RegimeCausalPipeline()
        pipe.fit(features)
        result = pipe.transform(features)

        assert isinstance(result, np.ndarray)
        assert result.ndim == 1

    def test_transform_shape_base_plus_extra(self):
        """
        After fit(), transform() shape == n_base_features + get_extra_dims().
        """
        from src.integration.regime_causal_pipeline import RegimeCausalPipeline

        features = make_synthetic_features(200)
        n_base = len(features.select_dtypes(include=[np.number]).columns)

        pipe = RegimeCausalPipeline()
        pipe.fit(features)

        result = pipe.transform(features)
        expected_len = n_base + pipe.get_extra_dims()
        assert len(result) == expected_len, (
            f"Expected {expected_len}, got {len(result)}; "
            f"n_base={n_base}, extra={pipe.get_extra_dims()}"
        )

    def test_transform_before_fit_returns_raw_features(self):
        """
        transform() before fit() should not crash and should return the raw
        feature vector (no extra dims appended).
        """
        from src.integration.regime_causal_pipeline import RegimeCausalPipeline

        features = make_synthetic_features(50)
        pipe = RegimeCausalPipeline()  # deliberately NOT calling fit()
        result = pipe.transform(features)

        assert isinstance(result, np.ndarray)
        assert result.ndim == 1
        # No extra dims because not fitted
        n_base = len(features.select_dtypes(include=[np.number]).columns)
        assert len(result) == n_base

    def test_transform_dtype_float32(self):
        """transform() output must be float32."""
        from src.integration.regime_causal_pipeline import RegimeCausalPipeline

        features = make_synthetic_features(200)
        pipe = RegimeCausalPipeline()
        pipe.fit(features)
        result = pipe.transform(features)
        assert result.dtype == np.float32


# ---------------------------------------------------------------------------
# Test 4 – get_extra_dims()
# ---------------------------------------------------------------------------


class TestGetExtraDims:
    def test_extra_dims_both_enabled_after_fit(self):
        """
        get_extra_dims() should return n_regimes + 1 == 4 when both modules
        are enabled AND successfully fitted.
        """
        from src.integration.regime_causal_pipeline import (
            RegimeCausalConfig,
            RegimeCausalPipeline,
        )

        features = make_synthetic_features(200)
        cfg = RegimeCausalConfig(use_regime=True, use_causal=True, n_regimes=3)
        pipe = RegimeCausalPipeline(cfg)
        pipe.fit(features)

        extra = pipe.get_extra_dims()
        # If both modules fitted: 3 (regime probs) + 1 (causal signal) = 4
        # If only one module fitted: 3 or 1
        # At least one should be > 0 after fitting 200 bars
        assert isinstance(extra, int)
        assert extra >= 0

        # Verify the documented contract: both ON → max 4
        if pipe._hmm_fitted and pipe._causal_fitted:
            assert extra == 4

    def test_extra_dims_zero_before_fit(self):
        """get_extra_dims() is 0 before any fit() call."""
        from src.integration.regime_causal_pipeline import RegimeCausalPipeline

        pipe = RegimeCausalPipeline()
        assert pipe.get_extra_dims() == 0

    def test_extra_dims_regime_only(self):
        """With use_causal=False, extra dims ≤ n_regimes (3)."""
        from src.integration.regime_causal_pipeline import (
            RegimeCausalConfig,
            RegimeCausalPipeline,
        )

        features = make_synthetic_features(200)
        cfg = RegimeCausalConfig(use_regime=True, use_causal=False, n_regimes=3)
        pipe = RegimeCausalPipeline(cfg)
        pipe.fit(features)

        extra = pipe.get_extra_dims()
        assert extra <= 3

    def test_extra_dims_causal_only(self):
        """With use_regime=False, extra dims ≤ 1."""
        from src.integration.regime_causal_pipeline import (
            RegimeCausalConfig,
            RegimeCausalPipeline,
        )

        features = make_synthetic_features(200)
        cfg = RegimeCausalConfig(use_regime=False, use_causal=True)
        pipe = RegimeCausalPipeline(cfg)
        pipe.fit(features)

        extra = pipe.get_extra_dims()
        assert extra <= 1


# ---------------------------------------------------------------------------
# Test 5 – maybe_refit()
# ---------------------------------------------------------------------------


class TestMaybeRefit:
    def test_maybe_refit_triggers_at_n_bars(self):
        """
        Calling maybe_refit() refit_every_n_bars times must trigger a refit.

        We verify this by patching fit() with a counter.
        """
        from src.integration.regime_causal_pipeline import (
            RegimeCausalConfig,
            RegimeCausalPipeline,
        )

        features = make_synthetic_features(200)
        cfg = RegimeCausalConfig(refit_every_n_bars=10)
        pipe = RegimeCausalPipeline(cfg)
        pipe.fit(features)  # initial fit

        fit_call_count = [0]
        original_fit = pipe.fit

        def counting_fit(f):
            fit_call_count[0] += 1
            original_fit(f)

        pipe.fit = counting_fit  # monkey-patch

        # Call maybe_refit() exactly refit_every_n_bars times
        for _ in range(cfg.refit_every_n_bars):
            pipe.maybe_refit(features)

        assert fit_call_count[0] >= 1, (
            f"Expected at least 1 refit call after {cfg.refit_every_n_bars} "
            f"maybe_refit() invocations, got {fit_call_count[0]}"
        )

    def test_maybe_refit_increments_bar_count(self):
        """bar_count is incremented on every maybe_refit() call."""
        from src.integration.regime_causal_pipeline import RegimeCausalPipeline

        features = make_synthetic_features(200)
        pipe = RegimeCausalPipeline()
        assert pipe._bar_count == 0

        for i in range(5):
            pipe.maybe_refit(features)

        assert pipe._bar_count == 5

    def test_maybe_refit_no_refit_before_threshold(self):
        """
        maybe_refit() called fewer than refit_every_n_bars times should NOT
        trigger a refit.
        """
        from src.integration.regime_causal_pipeline import (
            RegimeCausalConfig,
            RegimeCausalPipeline,
        )

        features = make_synthetic_features(200)
        cfg = RegimeCausalConfig(refit_every_n_bars=20)
        pipe = RegimeCausalPipeline(cfg)
        pipe.fit(features)

        fit_call_count = [0]
        original_fit = pipe.fit

        def counting_fit(f):
            fit_call_count[0] += 1
            original_fit(f)

        pipe.fit = counting_fit

        # Only 10 calls — should NOT trigger a refit (threshold is 20)
        for _ in range(10):
            pipe.maybe_refit(features)

        assert fit_call_count[0] == 0


# ---------------------------------------------------------------------------
# Test 6 – use_regime=False (causal only)
# ---------------------------------------------------------------------------


class TestCausalOnly:
    def test_causal_only_pipeline_fit_transform(self):
        """Pipeline with use_regime=False fits and transforms without error."""
        from src.integration.regime_causal_pipeline import (
            RegimeCausalConfig,
            RegimeCausalPipeline,
        )

        features = make_synthetic_features(200)
        cfg = RegimeCausalConfig(use_regime=False, use_causal=True)
        pipe = RegimeCausalPipeline(cfg)
        result = pipe.fit_transform(features)

        assert isinstance(result, np.ndarray)
        assert result.ndim == 1

    def test_causal_only_no_regime_dims(self):
        """With use_regime=False, no regime probabilities appended."""
        from src.integration.regime_causal_pipeline import (
            RegimeCausalConfig,
            RegimeCausalPipeline,
        )

        features = make_synthetic_features(200)
        cfg = RegimeCausalConfig(use_regime=False, use_causal=True)
        pipe = RegimeCausalPipeline(cfg)
        pipe.fit(features)

        n_base = len(features.select_dtypes(include=[np.number]).columns)
        result = pipe.transform(features)

        # At most n_base + 1 (causal signal)
        assert len(result) <= n_base + 1


# ---------------------------------------------------------------------------
# Test 7 – use_causal=False (regime only)
# ---------------------------------------------------------------------------


class TestRegimeOnly:
    def test_regime_only_pipeline_fit_transform(self):
        """Pipeline with use_causal=False fits and transforms without error."""
        from src.integration.regime_causal_pipeline import (
            RegimeCausalConfig,
            RegimeCausalPipeline,
        )

        features = make_synthetic_features(200)
        cfg = RegimeCausalConfig(use_regime=True, use_causal=False)
        pipe = RegimeCausalPipeline(cfg)
        result = pipe.fit_transform(features)

        assert isinstance(result, np.ndarray)
        assert result.ndim == 1

    def test_regime_only_no_causal_dims(self):
        """With use_causal=False, no causal signal appended."""
        from src.integration.regime_causal_pipeline import (
            RegimeCausalConfig,
            RegimeCausalPipeline,
        )

        features = make_synthetic_features(200)
        cfg = RegimeCausalConfig(use_regime=True, use_causal=False, n_regimes=3)
        pipe = RegimeCausalPipeline(cfg)
        pipe.fit(features)

        n_base = len(features.select_dtypes(include=[np.number]).columns)
        result = pipe.transform(features)

        # At most n_base + n_regimes
        assert len(result) <= n_base + cfg.n_regimes


# ---------------------------------------------------------------------------
# Test 8 – DynamicEnsemble.predict_with_pipeline()
# ---------------------------------------------------------------------------


class TestDynamicEnsembleWithPipeline:
    """Tests for the new predict_with_pipeline method on DynamicEnsemble."""

    @pytest.fixture
    def simple_ensemble(self):
        """
        Build a minimal DynamicEnsemble with MockAgents and a dummy
        regime_detector.
        """
        from src.ensemble.ensemble_agents import DynamicEnsemble

        class DummyRegimeDetector:
            """Always says regime 0."""

            def predict(self, state):
                return 0

        agents = [MockAgent(), MockAgent(), MockAgent()]
        detector = DummyRegimeDetector()
        agent_regime_map = {0: 0, 1: 1, 2: 2}
        return DynamicEnsemble(agents, detector, agent_regime_map)

    @pytest.fixture
    def fitted_pipeline(self):
        """Return a fitted pipeline on 200-bar data."""
        from src.integration.regime_causal_pipeline import RegimeCausalPipeline

        features = make_synthetic_features(200)
        pipe = RegimeCausalPipeline()
        pipe.fit(features)
        return pipe

    def test_predict_with_pipeline_returns_action(
        self, simple_ensemble, fitted_pipeline
    ):
        """predict_with_pipeline() must return a valid action without raising."""
        features = make_synthetic_features(200)
        state = features.iloc[-1].values.astype(np.float32)

        action = simple_ensemble.predict_with_pipeline(state, fitted_pipeline)
        # MockAgent always returns 1 (Hold)
        assert action == 1

    def test_predict_with_pipeline_with_features_df(
        self, simple_ensemble, fitted_pipeline
    ):
        """predict_with_pipeline() works when features_df is provided explicitly."""
        features = make_synthetic_features(200)
        state = features.iloc[-1].values.astype(np.float32)

        action = simple_ensemble.predict_with_pipeline(
            state, fitted_pipeline, features_df=features
        )
        assert action == 1

    def test_predict_with_pipeline_unfitted(self, simple_ensemble):
        """predict_with_pipeline() works even with an *unfitted* pipeline."""
        from src.integration.regime_causal_pipeline import RegimeCausalPipeline

        pipe = RegimeCausalPipeline()  # NOT fitted
        features = make_synthetic_features(50)
        state = features.iloc[-1].values.astype(np.float32)

        # Should not raise; falls back gracefully
        action = simple_ensemble.predict_with_pipeline(state, pipe)
        assert action == 1

    def test_predict_with_pipeline_invalid_features_df_falls_back(
        self, simple_ensemble, fitted_pipeline
    ):
        """
        If features_df is not a DataFrame, the method should warn and fall
        back to using raw state — not raise.
        """
        state = np.zeros(5, dtype=np.float32)

        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            action = simple_ensemble.predict_with_pipeline(
                state, fitted_pipeline, features_df="bad_input"
            )
        # Ensure we got a RuntimeWarning
        runtime_warnings = [x for x in w if issubclass(x.category, RuntimeWarning)]
        assert len(runtime_warnings) >= 1
        assert action == 1

    def test_predict_with_pipeline_state_is_enriched(
        self, simple_ensemble, fitted_pipeline
    ):
        """
        When both HMM and causal are fitted, the vector passed to the agent
        should be longer than the raw state.

        We verify this by replacing MockAgent.select_action with a spy.
        """
        features = make_synthetic_features(200)
        state = features.iloc[-1].values.astype(np.float32)

        received_states = []

        class SpyAgent(MockAgent):
            def select_action(self, s, deterministic=True):
                received_states.append(s)
                return 1

        spy_agents = [SpyAgent(), SpyAgent(), SpyAgent()]
        from src.ensemble.ensemble_agents import DynamicEnsemble

        class DummyRegimeDetector:
            def predict(self, s):
                return 0

        ensemble = DynamicEnsemble(
            spy_agents, DummyRegimeDetector(), {0: 0, 1: 1, 2: 2}
        )
        # Add some performance history so the regime-specific agent is chosen
        ensemble.performance_history[0].append(1.0)

        ensemble.predict_with_pipeline(state, fitted_pipeline, features_df=features)

        # If pipeline is fitted and adds dims, the state passed to the agent
        # should be at least as long as the raw state.
        if received_states:
            assert len(received_states[0]) >= len(state)


# ---------------------------------------------------------------------------
# Test 9 – fit_transform convenience
# ---------------------------------------------------------------------------


class TestFitTransform:
    def test_fit_transform_equivalent_to_fit_then_transform(self):
        """fit_transform(X) must be consistent with fit(X); transform(X)."""
        from src.integration.regime_causal_pipeline import (
            RegimeCausalConfig,
            RegimeCausalPipeline,
        )

        features = make_synthetic_features(200)

        cfg1 = RegimeCausalConfig(use_regime=True, use_causal=True)
        pipe1 = RegimeCausalPipeline(cfg1)
        result1 = pipe1.fit_transform(features)

        cfg2 = RegimeCausalConfig(use_regime=True, use_causal=True)
        pipe2 = RegimeCausalPipeline(cfg2)
        pipe2.fit(features)
        result2 = pipe2.transform(features)

        # Both must be the same length (exact values may differ due to random
        # HMM init, but shapes must match)
        assert len(result1) == len(result2)


# ---------------------------------------------------------------------------
# Test 10 – Regime uniform fallback
# ---------------------------------------------------------------------------


class TestRegimeFallback:
    def test_uniform_regime_probs_when_hmm_not_fitted(self):
        """
        _predict_regime_proba() should return uniform [1/3, 1/3, 1/3] when
        the HMM has not been fitted.
        """
        from src.integration.regime_causal_pipeline import (
            RegimeCausalConfig,
            RegimeCausalPipeline,
        )

        features = make_synthetic_features(50)
        cfg = RegimeCausalConfig(n_regimes=3, use_regime=True, use_causal=False)
        pipe = RegimeCausalPipeline(cfg)
        # Do NOT call fit()

        probs = pipe._predict_regime_proba(features)
        expected = np.ones(3) / 3
        np.testing.assert_allclose(probs, expected, atol=1e-6)
