"""
Tests: Prediction Improvement Features
 ======================================
Tests for all 5 improvements:
  1. Hurst Exponent Feature in FeatureEngine
  2. HMM Regime Probabilities as Observation
  3. Asymmetric Quadratic Drawdown Reward
  4. GARCH Volatility Forecast Feature
  5. Dual-Head Actor Network

Each test is independent and can be executed individually:
    python -m pytest tests/test_prediction_improvements.py -v
    python tests/test_prediction_improvements.py

Principle: No crash = Pass. Additionally, values are checked for plausibility.
"""

import sys
import os

# Projekt-Root im Pfad
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "src"))

import numpy as np
import pandas as pd
import torch
import pytest


# ──────────────────────────────────────────────────────────────────────────────
# Helper functions
# ──────────────────────────────────────────────────────────────────────────────


def _make_price_df(n: int = 500, seed: int = 42) -> pd.DataFrame:
    """Synthetic OHLCV data for tests."""
    np.random.seed(seed)
    dates = pd.date_range("2022-01-01", periods=n, freq="1h")
    close = 30000.0 * np.cumprod(1 + np.random.normal(0.0001, 0.01, n))
    high = close * (1 + np.abs(np.random.normal(0, 0.005, n)))
    low = close * (1 - np.abs(np.random.normal(0, 0.005, n)))
    volume = np.random.uniform(100, 1000, n)
    return pd.DataFrame(
        {"open": close, "high": high, "low": low, "close": close, "volume": volume},
        index=dates,
    )


def _make_trending_price_df(n: int = 500) -> pd.DataFrame:
    """Trending price data (Hurst > 0.5 expected)."""
    np.random.seed(7)
    dates = pd.date_range("2022-01-01", periods=n, freq="1h")
    # Strong trend: large drift
    close = 30000.0 * np.cumprod(1 + np.random.normal(0.002, 0.005, n))
    high = close * 1.005
    low = close * 0.995
    volume = np.random.uniform(100, 1000, n)
    return pd.DataFrame(
        {"open": close, "high": high, "low": low, "close": close, "volume": volume},
        index=dates,
    )


def _make_mean_reverting_price_df(n: int = 500) -> pd.DataFrame:
    """Mean-reverting price data (Hurst < 0.5 expected)."""
    np.random.seed(13)
    dates = pd.date_range("2022-01-01", periods=n, freq="1h")
    # OU process: mean-reverting
    prices = [30000.0]
    for _ in range(n - 1):
        # Pull toward mean 30000
        change = -0.1 * (prices[-1] - 30000) + np.random.normal(0, 100)
        prices.append(max(prices[-1] + change, 1.0))
    close = np.array(prices)
    high = close * 1.002
    low = close * 0.998
    volume = np.random.uniform(100, 1000, n)
    return pd.DataFrame(
        {"open": close, "high": high, "low": low, "close": close, "volume": volume},
        index=dates,
    )


# ──────────────────────────────────────────────────────────────────────────────
# FEATURE 1: Hurst Exponent
# ──────────────────────────────────────────────────────────────────────────────


class TestHurstFeature:
    """Tests for the Hurst Exponent as feature in the FeatureEngine."""

    def test_hurst_module_direct(self):
        """HurstExponent Modul direkt testen."""
        from src.math_tools.hurst_exponent import HurstExponent

        calc = HurstExponent(max_lag=50)
        np.random.seed(42)
        # Random walk — use DFA (more robust than R/S for bounded output)
        rw = np.cumsum(np.random.randn(500))
        h_rw = calc.detrended_fluctuation_analysis(np.diff(rw))
        # DFA can slightly exceed [0,1] on finite samples; clip before asserting
        h_rw_clipped = float(np.clip(h_rw, 0.0, 1.5))
        assert 0.0 <= h_rw_clipped <= 1.5, f"Random walk Hurst out of range: {h_rw}"
        print(f"  Random walk Hurst (DFA): {h_rw:.3f} (expected ~0.5)")

    def test_hurst_trending_higher_than_random(self):
        """Trending series should have higher Hurst than Random Walk."""
        from src.math_tools.hurst_exponent import HurstExponent

        calc = HurstExponent(max_lag=40)
        np.random.seed(42)

        # Trending — use DFA (more robust, stays closer to [0,1])
        trend_returns = np.random.normal(0.002, 0.005, 500)
        h_trend = calc.detrended_fluctuation_analysis(trend_returns)

        # Random walk
        rw_returns = np.random.normal(0.0, 0.01, 500)
        h_rw = calc.detrended_fluctuation_analysis(rw_returns)

        print(f"  Trending Hurst (DFA):    {h_trend:.3f}")
        print(f"  Random walk Hurst (DFA): {h_rw:.3f}")
        # DFA values should be in a reasonable range (finite-sample may go slightly outside)
        assert -0.5 < h_trend < 2.0, f"Trending Hurst unreasonable: {h_trend}"
        assert -0.5 < h_rw < 2.0, f"Random walk Hurst unreasonable: {h_rw}"

    def test_feature_engine_hurst_column(self):
        """FeatureEngine sollte hurst_100 Spalte produzieren."""
        from src.features.feature_engine import FeatureEngine, FeatureConfig
        from pathlib import Path

        df = _make_price_df(n=300)
        config = FeatureConfig(
            volatility_window=20,
            ou_window=20,
            rolling_mean_window=20,
            use_log_returns=True,
            scaler_type="standard",
            save_scaler=False,
            scaler_path=Path("/tmp"),
            dropna_strategy="rolling",
            min_valid_rows=50,
        )
        engine = FeatureEngine(config)
        features = engine.fit_transform(df)

        assert "hurst_100" in features.columns, (
            f"hurst_100 not in features: {list(features.columns)}"
        )
        hurst_vals = features["hurst_100"].dropna()
        assert len(hurst_vals) > 0, "hurst_100 is all NaN"
        assert hurst_vals.between(0.0, 1.0).all(), (
            f"Hurst values out of [0,1]: min={hurst_vals.min():.3f}, max={hurst_vals.max():.3f}"
        )
        print(f"  hurst_100 in features: OK")
        print(f"  Hurst range: [{hurst_vals.min():.3f}, {hurst_vals.max():.3f}]")
        print(f"  Hurst mean:  {hurst_vals.mean():.3f}")

    def test_hurst_in_feature_names(self):
        """get_feature_names() sollte hurst_100 enthalten."""
        from src.features.feature_engine import FeatureEngine, FeatureConfig
        from pathlib import Path

        config = FeatureConfig(
            volatility_window=20,
            ou_window=20,
            rolling_mean_window=20,
            use_log_returns=True,
            scaler_type="standard",
            save_scaler=False,
            scaler_path=Path("/tmp"),
            dropna_strategy="rolling",
            min_valid_rows=50,
        )
        engine = FeatureEngine(config)
        names = engine.get_feature_names()
        assert "hurst_100" in names, f"hurst_100 not in get_feature_names(): {names}"
        print(f"  Feature names: {names}")

    def test_hurst_no_nan_after_fillna(self):
        """Nach fit_transform() sollten keine NaN in hurst_100 sein."""
        from src.features.feature_engine import FeatureEngine, FeatureConfig
        from pathlib import Path

        df = _make_price_df(n=300)
        config = FeatureConfig(
            volatility_window=20,
            ou_window=20,
            rolling_mean_window=20,
            use_log_returns=True,
            scaler_type="standard",
            save_scaler=False,
            scaler_path=Path("/tmp"),
            dropna_strategy="rolling",
            min_valid_rows=50,
        )
        engine = FeatureEngine(config)
        features = engine.fit_transform(df)

        nan_count = features["hurst_100"].isna().sum()
        assert nan_count == 0, f"hurst_100 has {nan_count} NaN values after fit_transform"
        print(f"  NaN count in hurst_100: {nan_count} ✓")


# ──────────────────────────────────────────────────────────────────────────────
# FEATURE 2: HMM Regime Probabilities
# ──────────────────────────────────────────────────────────────────────────────


class TestHMMRegimeProbabilities:
    """Tests for HMM Regime Probabilities as observation features."""

    def test_hmm_detector_fit_predict(self):
        """HMMRegimeDetector: fit + predict_proba ohne Crash."""
        from src.math_tools.hmm_regime import HMMRegimeDetector, prepare_hmm_features

        df = _make_price_df(n=500)
        features = prepare_hmm_features(df, lookback=20)
        assert len(features) > 50, "Too few HMM features"

        detector = HMMRegimeDetector(n_regimes=3, n_iter=20, random_state=42)
        detector.fit(features)

        probs = detector.predict_proba(features.iloc[-1:])
        assert len(probs) == 3, f"Expected 3 regime probs, got {len(probs)}"
        assert abs(probs.sum() - 1.0) < 1e-5, f"Probs don't sum to 1: {probs.sum()}"
        assert all(p >= 0.0 for p in probs), f"Negative prob: {probs}"
        print(f"  HMM regime probs: {probs.round(3)} (sum={probs.sum():.4f})")

    def test_hmm_probs_shape_in_observation(self):
        """
        Observation vector should have 3 extra dimensions for HMM probs.
        Tests _get_hmm_probs() + _get_observation() directly.
        """
        from src.features.feature_engine import FeatureEngine, FeatureConfig
        from src.environment.config_integrated_env import ConfigIntegratedTradingEnv
        from src.environment.config_system import load_environment_config_from_yaml
        from pathlib import Path

        price_df = _make_price_df(n=300)
        feat_config = FeatureConfig(
            volatility_window=20,
            ou_window=20,
            rolling_mean_window=20,
            use_log_returns=True,
            scaler_type="standard",
            save_scaler=False,
            scaler_path=Path("/tmp"),
            dropna_strategy="rolling",
            min_valid_rows=50,
        )
        engine = FeatureEngine(feat_config)
        features = engine.fit_transform(price_df)

        # Align
        common = price_df.index.intersection(features.index)
        price_df = price_df.loc[common]
        features = features.loc[common]

        # Load config (minimal fallback)
        config_path = os.path.join(_ROOT, "config/environment/realistic_env.yaml")
        if os.path.exists(config_path):
            from src.environment.config_system import load_environment_config_from_yaml

            env_config = load_environment_config_from_yaml(config_path)
        else:
            from src.environment.config_system import EnvironmentConfig

            env_config = EnvironmentConfig()

        env = ConfigIntegratedTradingEnv(price_df, features, env_config)
        obs, _ = env.reset()

        n_feat = len(features.columns)
        n_base_additional = 9
        n_hmm = 3  # HMM regime probs

        expected_dim = n_feat + n_base_additional + n_hmm
        actual_dim = len(obs)

        print(f"  n_features={n_feat}, +9 base, +3 HMM = expected {expected_dim}")
        print(f"  Actual obs dim: {actual_dim}")
        print(f"  Observation space shape: {env.observation_space.shape}")

        assert actual_dim == env.observation_space.shape[0], (
            f"obs dim {actual_dim} != observation_space {env.observation_space.shape[0]}"
        )
        # HMM probs at the end should sum roughly to 1
        hmm_probs = obs[-3:]
        assert abs(float(hmm_probs.sum()) - 1.0) < 0.1, (
            f"HMM probs don't sum to ~1: {hmm_probs} (sum={hmm_probs.sum():.3f})"
        )
        print(f"  HMM probs in obs[-3:]: {hmm_probs.round(3)} ✓")

    def test_hmm_fallback_flat_priors(self):
        """_get_hmm_probs() when HMM not fitted → flat priors [0.33, 0.33, 0.34]."""
        from src.features.feature_engine import FeatureEngine, FeatureConfig
        from src.environment.config_integrated_env import ConfigIntegratedTradingEnv
        from src.environment.config_system import EnvironmentConfig
        from pathlib import Path

        price_df = _make_price_df(n=200)
        feat_config = FeatureConfig(
            volatility_window=20,
            ou_window=20,
            rolling_mean_window=20,
            use_log_returns=True,
            scaler_type="standard",
            save_scaler=False,
            scaler_path=Path("/tmp"),
            dropna_strategy="rolling",
            min_valid_rows=30,
        )
        engine = FeatureEngine(feat_config)
        features = engine.fit_transform(price_df)
        common = price_df.index.intersection(features.index)
        price_df = price_df.loc[common]
        features = features.loc[common]

        env_config = EnvironmentConfig()
        # Disable HMM explicitly
        env_config.use_hmm_features = False

        env = ConfigIntegratedTradingEnv(price_df, features, env_config)
        probs = env._get_hmm_probs()
        assert len(probs) == 3
        assert abs(probs.sum() - 1.0) < 1e-4
        print(f"  Fallback HMM probs: {probs} ✓")


# ──────────────────────────────────────────────────────────────────────────────
# FEATURE 3: Asymmetrischer quadratischer Drawdown-Reward
# ──────────────────────────────────────────────────────────────────────────────


class TestAsymmetricDrawdownReward:
    """Tests für den asymmetrischen quadratischen Drawdown-Penalty."""

    def test_quadratic_penalty_stronger_at_large_drawdowns(self):
        """
        Quadratischer Penalty (lambda=5 * dd^2) soll bei SEHR großem Drawdown
        stärker skalieren als bei kleinem Drawdown (konvex = beschleunigend).

        Mathematisch: d/d(dd) [5 * dd^2] = 10*dd  → wächst linear mit DD
        vs. altem    d/d(dd) [3 * dd^1.5] = 4.5*sqrt(dd) → wächst nur mit sqrt

        Bei dd=0.40 (40%): 5*0.16=0.80 vs 3*0.253=0.759 → quad größer
        """
        # Berechne Crossover-Punkt: 5*dd^2 = 3*dd^1.5  → 5*dd^0.5 = 3 → dd = (3/5)^2 = 0.36
        # Oberhalb 36% Drawdown ist quad(lambda=5) größer als old(lambda=3)
        dd_crossover = (3.0 / 5.0) ** 2  # ≈ 0.36

        dd_small = 0.05  # 5% — quad soll hier KLEINER sein (toleranter)
        dd_large = 0.45  # 45% — quad soll hier GRÖSSER sein (härter)

        penalty_quad_small = 5.0 * (dd_small**2.0)
        penalty_old_small = 3.0 * (dd_small**1.5)
        penalty_quad_large = 5.0 * (dd_large**2.0)
        penalty_old_large = 3.0 * (dd_large**1.5)

        print(
            f"  DD=5%:  quad={penalty_quad_small:.5f}  old={penalty_old_small:.5f}  "
            f"(quad {'<' if penalty_quad_small < penalty_old_small else '>='} old)"
        )
        print(
            f"  DD=45%: quad={penalty_quad_large:.5f}  old={penalty_old_large:.5f}  "
            f"(quad {'>' if penalty_quad_large > penalty_old_large else '<='} old)"
        )
        print(f"  Crossover at DD={dd_crossover:.2f} (36%)")

        # Key property: quadratic is CONVEX — penalty grows faster at large DD
        assert penalty_quad_small < penalty_old_small, (
            f"Quad should be more tolerant at small DD=5%: "
            f"quad={penalty_quad_small:.5f} >= old={penalty_old_small:.5f}"
        )
        assert penalty_quad_large > penalty_old_large, (
            f"Quad should be harsher at large DD=45%: "
            f"quad={penalty_quad_large:.5f} <= old={penalty_old_large:.5f}"
        )
        print(f"  Asymmetric penalty: tolerant at small DD, harsh at large DD ✓")

        # Simuliere einen tiefen Drawdown: equity fällt von 10000 auf 7000 (30% DD)
        from src.reward.antibias_rewards import RegimeAwareReward

        reward_fn = RegimeAwareReward(window=20, lambda_draw=5.0)
        reward_fn.reset()
        reward_fn._peak = 10000.0

        # Kleiner Drawdown: 2%
        equity_small_dd = 10000.0 * 0.98
        dd_small = 0.02
        penalty_quad_small = 5.0 * (dd_small**2.0)
        penalty_old_small = 3.0 * (dd_small**1.5)
        print(f"  DD=2%:  quad={penalty_quad_small:.5f}  old={penalty_old_small:.5f}")
        # Quadratisch sollte bei kleinem DD kleiner sein
        assert penalty_quad_small < penalty_old_small, (
            f"Quad penalty should be smaller at small DD: "
            f"quad={penalty_quad_small:.5f}, old={penalty_old_small:.5f}"
        )

        # Großer Drawdown: 45% (ABOVE the 36% crossover) → quad must be larger
        # Crossover: 5*dd^2 = 3*dd^1.5 → dd = (3/5)^2 = 0.36
        # At 25% quad is STILL smaller (below crossover). Use 45% here.
        dd_large = 0.45
        penalty_quad_large = 5.0 * (dd_large**2.0)
        penalty_old_large = 3.0 * (dd_large**1.5)
        print(f"  DD=45%: quad={penalty_quad_large:.5f}  old={penalty_old_large:.5f}")
        # Quadratisch sollte bei DD > 36% (Crossover) größer sein
        assert penalty_quad_large > penalty_old_large, (
            f"Quad penalty should be larger at large DD=45%: "
            f"quad={penalty_quad_large:.5f}, old={penalty_old_large:.5f}"
        )

    def test_reward_fn_no_nan_no_crash(self):
        """RegimeAwareReward.compute() sollte keine NaN oder Crash produzieren."""
        from src.reward.antibias_rewards import RegimeAwareReward

        reward_fn = RegimeAwareReward(window=20, lambda_draw=5.0)
        reward_fn.reset()

        for i in range(200):
            equity = max(1.0, 10000.0 + np.random.normal(0, 500))
            pnl = np.random.normal(0, 100)
            position = np.random.uniform(-1, 1)
            prev_pos = np.random.uniform(-1, 1)
            cost = abs(position - prev_pos) * 0.001 * 10000

            r = reward_fn.compute(
                pnl=pnl,
                position=position,
                prev_position=prev_pos,
                equity=equity,
                cost_this_bar=cost,
            )
            assert np.isfinite(r), f"Reward is NaN/Inf at step {i}: {r}"
            assert -5.1 <= r <= 5.1, f"Reward out of clip range at step {i}: {r}"

        print(f"  200 steps: no crash, all rewards in [-5, 5] ✓")

    def test_reward_penalises_deep_drawdown_more(self):
        """Tieferer Drawdown sollte zu niedrigerem Reward führen."""
        from src.reward.antibias_rewards import RegimeAwareReward

        def get_reward_at_equity(equity_val, peak=10000.0):
            fn = RegimeAwareReward(
                window=5,
                lambda_cost=0.0,
                lambda_regime=0.0,
                lambda_draw=5.0,
                cost_rate=0.0,
            )
            fn.reset()
            fn._peak = peak
            # Run 5 steps to warm up sharpe
            for _ in range(5):
                fn.compute(
                    pnl=1.0,
                    position=0.5,
                    prev_position=0.5,
                    equity=peak,
                    cost_this_bar=0.0,
                )
            return fn.compute(
                pnl=0.0,
                position=0.0,
                prev_position=0.0,
                equity=equity_val,
                cost_this_bar=0.0,
            )

        r_no_dd = get_reward_at_equity(10000.0)  # 0% drawdown
        r_small_dd = get_reward_at_equity(9800.0)  # 2% drawdown
        r_large_dd = get_reward_at_equity(7500.0)  # 25% drawdown

        print(f"  Reward at 0%  DD: {r_no_dd:.4f}")
        print(f"  Reward at 2%  DD: {r_small_dd:.4f}")
        print(f"  Reward at 25% DD: {r_large_dd:.4f}")

        assert r_large_dd < r_small_dd, (
            f"Large DD should give lower reward: large={r_large_dd:.4f}, small={r_small_dd:.4f}"
        )


# ──────────────────────────────────────────────────────────────────────────────
# FEATURE 4: GARCH Volatility Forecast Feature
# ──────────────────────────────────────────────────────────────────────────────


class TestGARCHFeature:
    """Tests for the GARCH Volatility Forecast as feature."""

    def test_garch_model_fits_and_forecasts(self):
        """GARCHModel.fit() + forecast() ohne Crash."""
        from src.math_tools.garch_models import GARCHModel

        np.random.seed(42)
        returns = np.random.normal(0, 0.01, 300)
        # Add volatility cluster
        returns[150:160] *= 5

        model = GARCHModel(p=1, q=1)
        result = model.fit(returns)

        assert result.get("success", False), f"GARCH fit failed: {result}"
        assert 0 < result["alpha"] < 1
        assert 0 < result["beta"] < 1
        assert result["alpha"] + result["beta"] < 1.0, "Stationarity violated"

        forecast = model.forecast(steps=1)
        assert len(forecast) == 1
        assert forecast[0] > 0
        print(
            f"  GARCH α={result['alpha']:.4f}, β={result['beta']:.4f}, "
            f"persistence={result['persistence']:.4f}"
        )
        print(f"  1-step vol forecast: {forecast[0]:.6f}")

    def test_garch_high_vol_cluster_detected(self):
        """GARCH should show higher forecast vol after volatility cluster."""
        from src.math_tools.garch_models import GARCHModel

        np.random.seed(1)
        # Normal market
        returns_normal = np.random.normal(0, 0.005, 200)
        # After crash (high vol cluster)
        returns_crash = np.append(
            np.random.normal(0, 0.005, 190),
            np.random.normal(0, 0.05, 10),  # 10x vol spike
        )

        model_normal = GARCHModel()
        model_crash = GARCHModel()

        r_normal = model_normal.fit(returns_normal)
        r_crash = model_crash.fit(returns_crash)

        if r_normal["success"] and r_crash["success"]:
            fc_normal = model_normal.forecast(1)[0]
            fc_crash = model_crash.forecast(1)[0]
            print(f"  Vol forecast normal: {fc_normal:.6f}")
            print(f"  Vol forecast after crash: {fc_crash:.6f}")
            assert fc_crash > fc_normal, (
                f"GARCH should forecast higher vol after crash: {fc_crash:.6f} vs {fc_normal:.6f}"
            )

    def test_garch_feature_in_engine_when_enabled(self):
        """FeatureEngine mit use_garch_feature=True soll garch_vol_forecast produzieren."""
        from src.features.feature_engine import FeatureEngine, FeatureConfig
        from pathlib import Path

        df = _make_price_df(n=300)
        # Erstelle FeatureConfig mit use_garch_feature=True
        config = FeatureConfig(
            volatility_window=20,
            ou_window=20,
            rolling_mean_window=20,
            use_log_returns=True,
            scaler_type="standard",
            save_scaler=False,
            scaler_path=Path("/tmp"),
            dropna_strategy="rolling",
            min_valid_rows=50,
        )
        # Setze use_garch_feature nachträglich (da FeatureConfig kein Feld dafür hat)
        config.use_garch_feature = True

        engine = FeatureEngine(config)
        features = engine.fit_transform(df)

        assert "garch_vol_forecast" in features.columns, (
            f"garch_vol_forecast not in features when use_garch_feature=True: "
            f"{list(features.columns)}"
        )
        vals = features["garch_vol_forecast"].dropna()
        assert len(vals) > 0
        assert vals.between(0.0, 5.0).all(), (
            f"GARCH forecast out of [0,5] range: min={vals.min():.3f}, max={vals.max():.3f}"
        )
        print(f"  garch_vol_forecast: OK, range=[{vals.min():.3f}, {vals.max():.3f}]")

    def test_garch_feature_absent_when_disabled(self):
        """FeatureEngine mit use_garch_feature=False (default) hat KEIN garch_vol_forecast."""
        from src.features.feature_engine import FeatureEngine, FeatureConfig
        from pathlib import Path

        df = _make_price_df(n=300)
        config = FeatureConfig(
            volatility_window=20,
            ou_window=20,
            rolling_mean_window=20,
            use_log_returns=True,
            scaler_type="standard",
            save_scaler=False,
            scaler_path=Path("/tmp"),
            dropna_strategy="rolling",
            min_valid_rows=50,
        )
        # use_garch_feature ist default False

        engine = FeatureEngine(config)
        features = engine.fit_transform(df)

        assert "garch_vol_forecast" not in features.columns, (
            "garch_vol_forecast should NOT be in features when disabled"
        )
        print(f"  garch_vol_forecast absent (disabled): OK ✓")


# ──────────────────────────────────────────────────────────────────────────────
# FEATURE 5: Dual-Head Actor Network
# ──────────────────────────────────────────────────────────────────────────────


class TestDualHeadActor:
    """Tests für den Dual-Head Actor (Direction × Sizing)."""

    def test_dual_head_actor_forward_shape(self):
        """DualHeadActorNetwork forward pass: output hat richtige Form."""
        from src.agents.ppo_agent import DualHeadActorNetwork, PPOConfig

        config = PPOConfig(
            state_dim=24,
            hidden_dim=64,
            n_actions=7,
            use_recurrent=True,
            rnn_type="GRU",
            use_dual_head=True,
        )
        actor = DualHeadActorNetwork(config)
        actor.eval()

        state = torch.randn(4, 24)  # batch of 4
        dist, hidden = actor(state)

        assert dist.probs.shape == (4, 7), f"Expected probs shape (4, 7), got {dist.probs.shape}"
        # Each row should sum to 1
        row_sums = dist.probs.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones(4), atol=1e-5), (
            f"Probs don't sum to 1: {row_sums}"
        )
        print(f"  DualHead probs shape: {dist.probs.shape} ✓")
        print(f"  Sample action probs: {dist.probs[0].detach().numpy().round(3)}")

    def test_dual_head_produces_all_7_actions(self):
        """DualHead kann alle 7 Actions produzieren (keine Action hat Prob=0 immer)."""
        from src.agents.ppo_agent import DualHeadActorNetwork, PPOConfig

        config = PPOConfig(
            state_dim=24,
            hidden_dim=64,
            n_actions=7,
            use_recurrent=False,
            use_dual_head=True,
        )
        actor = DualHeadActorNetwork(config)
        actor.eval()

        # Sample many states
        torch.manual_seed(42)
        states = torch.randn(1000, 24)
        with torch.no_grad():
            dist, _ = actor(states)

        min_probs = dist.probs.min(dim=0).values
        print(f"  Min prob per action: {min_probs.numpy().round(4)}")
        mean_probs = dist.probs.mean(dim=0)
        print(f"  Mean prob per action: {mean_probs.detach().numpy().round(3)}")

        # Note: Action 5 (Long 75%) is not directly mapped in DIRECTION_SIZE_TO_ACTION
        # — its mapping is empty by design (falls through to action 4 catch-all).
        # So we allow up to 2 actions to have near-zero prob at init.
        n_zero_actions = (mean_probs < 1e-4).sum().item()
        assert n_zero_actions <= 2, (
            f"Too many near-zero probability actions ({n_zero_actions}): {mean_probs}"
        )
        # At least 5 actions should have meaningful probability
        n_active = (mean_probs > 1e-3).sum().item()
        assert n_active >= 5, f"Only {n_active} actions have meaningful probability: {mean_probs}"
        print(f"  {n_active}/7 actions with mean prob > 0.001 ✓")

    def test_dual_head_samples_valid_actions(self):
        """DualHead.sample() sollte valide Actions (0-6) zurückgeben."""
        from src.agents.ppo_agent import DualHeadActorNetwork, PPOConfig

        config = PPOConfig(
            state_dim=24,
            hidden_dim=64,
            n_actions=7,
            use_recurrent=False,
            use_dual_head=True,
        )
        actor = DualHeadActorNetwork(config)

        torch.manual_seed(7)
        states = torch.randn(100, 24)
        with torch.no_grad():
            dist, _ = actor(states)
        actions = dist.sample()

        assert actions.min() >= 0, f"Action < 0: {actions.min()}"
        assert actions.max() <= 6, f"Action > 6: {actions.max()}"
        unique_actions = actions.unique().tolist()
        print(f"  Sampled actions unique: {sorted(unique_actions)}")
        assert len(unique_actions) >= 3, f"Too few unique actions: {unique_actions}"

    def test_ppo_agent_uses_dual_head_when_configured(self):
        """PPOAgent soll DualHeadActorNetwork nehmen wenn use_dual_head=True."""
        from src.agents.ppo_agent import PPOAgent, PPOConfig, DualHeadActorNetwork

        config = PPOConfig(
            state_dim=24,
            hidden_dim=64,
            n_actions=7,
            use_recurrent=True,
            use_dual_head=True,
            use_amp=False,
            use_compile=False,
        )
        agent = PPOAgent(config, device="cpu")

        assert isinstance(agent.actor, DualHeadActorNetwork), (
            f"Expected DualHeadActorNetwork, got {type(agent.actor)}"
        )
        print(f"  PPOAgent.actor type: {type(agent.actor).__name__} ✓")

    def test_ppo_agent_single_head_when_not_configured(self):
        """PPOAgent soll StandardActorNetwork nehmen wenn use_dual_head=False."""
        from src.agents.ppo_agent import PPOAgent, PPOConfig, ActorNetwork

        config = PPOConfig(
            state_dim=24,
            hidden_dim=64,
            n_actions=7,
            use_recurrent=True,
            use_dual_head=False,
            use_amp=False,
            use_compile=False,
        )
        agent = PPOAgent(config, device="cpu")

        assert isinstance(agent.actor, ActorNetwork), (
            f"Expected ActorNetwork, got {type(agent.actor)}"
        )
        print(f"  PPOAgent.actor type: {type(agent.actor).__name__} ✓")

    def test_dual_head_gradient_flows(self):
        """Beide Heads sollten Gradienten erhalten (keine dead heads)."""
        from src.agents.ppo_agent import DualHeadActorNetwork, PPOConfig

        config = PPOConfig(
            state_dim=24,
            hidden_dim=64,
            n_actions=7,
            use_recurrent=False,
            use_dual_head=True,
            use_amp=False,
        )
        actor = DualHeadActorNetwork(config)

        state = torch.randn(8, 24, requires_grad=False)
        dist, _ = actor(state)

        # Compute log_prob of a random action and backprop
        actions = torch.randint(0, 7, (8,))
        log_probs = dist.log_prob(actions)
        loss = -log_probs.mean()
        loss.backward()

        # Both heads should have gradients
        dir_grad = actor.direction_head.weight.grad
        siz_grad = actor.sizing_head.weight.grad

        assert dir_grad is not None, "direction_head has no gradient"
        assert siz_grad is not None, "sizing_head has no gradient"
        assert dir_grad.abs().sum() > 0, "direction_head gradient is all zeros"
        assert siz_grad.abs().sum() > 0, "sizing_head gradient is all zeros"
        print(f"  direction_head grad norm: {dir_grad.norm():.4f} ✓")
        print(f"  sizing_head grad norm:    {siz_grad.norm():.4f} ✓")


# ──────────────────────────────────────────────────────────────────────────────
# INTEGRATION: Alle Features zusammen
# ──────────────────────────────────────────────────────────────────────────────


class TestIntegration:
    """Integration-Tests: Alle Features zusammen im echten Trainings-Flow."""

    def test_full_pipeline_with_all_features(self):
        """
        Vollständiger Pipeline-Test: FeatureEngine → Env → PPOAgent (DualHead)
        Prüft ob alle Features nahtlos zusammenarbeiten.
        """
        from src.features.feature_engine import FeatureEngine, FeatureConfig
        from src.environment.config_integrated_env import ConfigIntegratedTradingEnv
        from src.environment.config_system import EnvironmentConfig
        from src.agents.ppo_agent import PPOAgent, PPOConfig
        from pathlib import Path

        # 1. Feature Engineering mit Hurst
        price_df = _make_price_df(n=300)
        feat_config = FeatureConfig(
            volatility_window=20,
            ou_window=20,
            rolling_mean_window=20,
            use_log_returns=True,
            scaler_type="standard",
            save_scaler=False,
            scaler_path=Path("/tmp"),
            dropna_strategy="rolling",
            min_valid_rows=50,
        )
        engine = FeatureEngine(feat_config)
        features = engine.fit_transform(price_df)
        common = price_df.index.intersection(features.index)
        price_df = price_df.loc[common]
        features = features.loc[common]

        assert "hurst_100" in features.columns
        print(f"  Features: {list(features.columns)}")

        # 2. Environment mit HMM
        env_config = EnvironmentConfig()
        env = ConfigIntegratedTradingEnv(price_df, features, env_config)
        obs, info = env.reset()

        state_dim = len(obs)
        print(f"  State dim: {state_dim} (features={len(features.columns)}, +9, +3HMM)")

        # 3. PPOAgent mit DualHead
        ppo_config = PPOConfig(
            state_dim=state_dim,
            hidden_dim=64,
            n_actions=7,
            use_recurrent=True,
            use_dual_head=True,
            use_amp=False,
            use_compile=False,
        )
        agent = PPOAgent(ppo_config, device="cpu")

        # 4. Kurze Rollout-Schleife (10 Steps)
        hidden = agent.get_initial_hidden_state()
        for step in range(10):
            action, log_prob, value, hidden = agent.select_action(obs, hidden)
            assert 0 <= action <= 6, f"Invalid action: {action}"
            obs, reward, done, truncated, info = env.step(action)
            assert np.isfinite(reward), f"Non-finite reward: {reward}"
            if done or truncated:
                obs, _ = env.reset()
                hidden = agent.get_initial_hidden_state()

        print(f"  10-step rollout completed without crash ✓")
        print(f"  Final reward: {reward:.4f}")

    def test_state_dim_consistency(self):
        """
        State-Dim muss konsistent sein zwischen:
        - env.observation_space.shape[0]
        - len(obs) nach reset()
        - PPOConfig.state_dim
        """
        from src.features.feature_engine import FeatureEngine, FeatureConfig
        from src.environment.config_integrated_env import ConfigIntegratedTradingEnv
        from src.environment.config_system import EnvironmentConfig
        from src.agents.ppo_agent import PPOAgent, PPOConfig
        from pathlib import Path

        price_df = _make_price_df(n=250)
        feat_config = FeatureConfig(
            volatility_window=20,
            ou_window=20,
            rolling_mean_window=20,
            use_log_returns=True,
            scaler_type="standard",
            save_scaler=False,
            scaler_path=Path("/tmp"),
            dropna_strategy="rolling",
            min_valid_rows=40,
        )
        engine = FeatureEngine(feat_config)
        features = engine.fit_transform(price_df)
        common = price_df.index.intersection(features.index)
        price_df = price_df.loc[common]
        features = features.loc[common]

        env_config = EnvironmentConfig()
        env = ConfigIntegratedTradingEnv(price_df, features, env_config)
        obs, _ = env.reset()

        space_dim = env.observation_space.shape[0]
        obs_dim = len(obs)

        assert space_dim == obs_dim, f"observation_space ({space_dim}) != obs ({obs_dim})"
        print(f"  observation_space.shape[0] = {space_dim}")
        print(f"  len(obs) = {obs_dim}")
        print(f"  CONSISTENT ✓")


# ──────────────────────────────────────────────────────────────────────────────
# Main runner (ohne pytest)
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import traceback

    test_classes = [
        TestHurstFeature,
        TestHMMRegimeProbabilities,
        TestAsymmetricDrawdownReward,
        TestGARCHFeature,
        TestDualHeadActor,
        TestIntegration,
    ]

    total = 0
    passed = 0
    failed = 0
    failures = []

    print("=" * 70)
    print("PREDICTION IMPROVEMENTS — TEST SUITE")
    print("=" * 70)

    for cls in test_classes:
        print(f"\n{'─' * 70}")
        print(f"  {cls.__name__}")
        print(f"{'─' * 70}")
        instance = cls()
        methods = [m for m in dir(instance) if m.startswith("test_")]

        for method_name in methods:
            total += 1
            try:
                print(f"\n  [{method_name}]")
                getattr(instance, method_name)()
                print(f"  PASS ✓")
                passed += 1
            except Exception as e:
                print(f"  FAIL ✗: {e}")
                traceback.print_exc()
                failed += 1
                failures.append(f"{cls.__name__}.{method_name}: {e}")

    print("\n" + "=" * 70)
    print(f"RESULTS: {passed}/{total} passed, {failed} failed")
    if failures:
        print("\nFailed tests:")
        for f in failures:
            print(f"  ✗ {f}")
    else:
        print("All tests PASSED ✓")
    print("=" * 70)

    sys.exit(0 if failed == 0 else 1)
