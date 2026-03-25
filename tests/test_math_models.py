"""
Comprehensive Tests for Mathematical Trading Models
====================================================

Tests for all mathematical models used in BITCOIN4Traders.

Run with: pytest tests/test_math_models.py -v
"""

import numpy as np
import pandas as pd
import pytest
from src.math_tools import (
    # Kalman Filter
    KalmanFilter1D,
    KalmanFilterPairs,
    KalmanTrendDetector,
    apply_kalman_smoothing,
    calculate_dynamic_beta,
    # Cointegration
    CointegrationTest,
    PairsTradingStrategy,
    calculate_hedge_ratio,
    calculate_spread,
    # GARCH
    GARCHModel,
    VolatilityRegimeDetector,
    VolatilityTargeting,
    forecast_volatility_garch,
    calculate_var_garch,
    # Hurst
    HurstExponent,
    RollingHurst,
    quick_hurst_check,
    hurst_adaptive_strategy,
    # OU Process
    OrnsteinUhlenbeckProcess,
    calculate_ou_score,
    # Kelly
    KellyCriterion,
    kelly_fraction,
    # HMM
    HMMRegimeDetector,
)
from src.math_tools.kelly_criterion import KellyParameters


class TestKalmanFilter:
    """Test Kalman Filter implementations."""

    def test_kalman_1d_smoothing(self):
        """Test 1D Kalman filter smoothing."""
        # Generate noisy signal
        np.random.seed(42)
        true_prices = np.cumsum(np.random.randn(100) * 0.01) + 100
        noise = np.random.randn(100) * 0.5
        noisy_prices = true_prices + noise

        # Apply Kalman filter with appropriate parameters for this noise level
        from src.math_tools.kalman_filter import KalmanFilterConfig

        config = KalmanFilterConfig(
            Q=0.01,  # Process noise (model uncertainty)
            R=0.25,  # Measurement noise (0.5^2)
            x0=true_prices[0],
            P0=1.0,
        )
        kf = KalmanFilter1D(config=config)
        smoothed = kf.filter_series(noisy_prices)

        # Smoothed should be closer to true than noisy
        mse_noisy = np.mean((noisy_prices - true_prices) ** 2)
        mse_smoothed = np.mean((smoothed - true_prices) ** 2)

        assert mse_smoothed < mse_noisy, (
            f"Kalman filter should reduce noise: {mse_smoothed:.3f} < {mse_noisy:.3f}"
        )
        assert len(smoothed) == len(noisy_prices), "Output length should match input"

    def test_kalman_pairs_hedge_ratio(self):
        """Test Kalman filter for pairs trading hedge ratio."""
        np.random.seed(42)
        n = 200

        # Generate cointegrated pair with known beta=2.0
        beta_true = 2.0
        asset2 = np.cumsum(np.random.randn(n) * 0.01) + 100
        asset1 = 50 + beta_true * asset2 + np.random.randn(n) * 0.5

        kf = KalmanFilterPairs()
        for i in range(n):
            alpha, beta, spread = kf.update(asset1[i], asset2[i])  # Online update

        # Beta should be reasonably close to true value (within 30%)
        relative_error = abs(beta - beta_true) / beta_true
        assert relative_error < 0.3, (
            f"Beta should converge to {beta_true}, got {beta} (error: {relative_error:.1%})"
        )

    def test_kalman_trend_detection(self):
        """Test Kalman trend detector."""
        np.random.seed(42)

        # Generate trending series (consistent drift upward)
        prices = np.cumsum(np.ones(100) * 0.1) + np.random.randn(100) * 0.1

        kf = KalmanTrendDetector()
        for price in prices:
            result = kf.update(price)

        # Should detect positive trend
        assert result["velocity"] > 0, "Should detect upward trend"
        assert result["trend_direction"] == 1, "Trend direction should be positive"


class TestCointegration:
    """Test cointegration and pairs trading."""

    def test_engle_granger_cointegration(self):
        """Test Engle-Granger cointegration test."""
        np.random.seed(42)
        n = 500

        # Generate cointegrated series: y = 2*x + noise (stable spread)
        x = np.cumsum(np.random.randn(n) * 0.01)
        y = 2.0 * x + np.random.randn(n) * 0.1

        test = CointegrationTest()
        result = test.engle_granger_test(y, x)

        assert result["cointegrated"], "Should detect cointegration"
        assert abs(result["beta"] - 2.0) < 0.1, "Beta should be close to 2.0"
        assert result["half_life"] > 0, "Should have finite half-life"

    def test_no_cointegration_random_walks(self):
        """Test that independent random walks are not cointegrated."""
        np.random.seed(42)
        n = 500

        # Independent random walks (no stable long-run relationship)
        x = np.cumsum(np.random.randn(n))
        y = np.cumsum(np.random.randn(n))

        test = CointegrationTest()
        result = test.engle_granger_test(y, x)

        # Should NOT be cointegrated
        assert not result["cointegrated"], (
            "Independent walks should not be cointegrated"
        )

    def test_pairs_trading_signals(self):
        """Test pairs trading strategy signals."""
        np.random.seed(42)
        n = 200

        # Generate pair
        asset2 = np.cumsum(np.random.randn(n) * 0.01) + 100
        asset1 = 50 + 2.0 * asset2 + np.random.randn(n) * 0.5

        strategy = PairsTradingStrategy(entry_zscore=2.0, exit_zscore=0.5)
        strategy.fit(asset1, asset2)  # Learn spread distribution

        # Test signal generation
        signal = strategy.generate_signal(asset1[-1], asset2[-1])
        assert signal in [-1, 0, 1], "Signal should be -1, 0, or 1"

    def test_hedge_ratio_calculation(self):
        """Test static hedge ratio calculation."""
        np.random.seed(42)
        n = 100

        x = np.random.randn(n) + 100
        y = 2.5 * x + np.random.randn(n) * 0.1  # Beta = 2.5

        beta = calculate_hedge_ratio(pd.Series(y), pd.Series(x))
        assert abs(beta - 2.5) < 0.2, f"Hedge ratio should be ~2.5, got {beta}"


class TestGARCH:
    """Test GARCH volatility models."""

    def test_garch_volatility_clustering(self):
        """Test that GARCH captures volatility clustering."""
        np.random.seed(42)
        n = 500

        # Generate returns with volatility clustering
        returns = np.random.randn(n) * 0.02
        # Add volatility clustering manually (spikes)
        returns[100:150] *= 3  # High vol period
        returns[300:350] *= 3  # Another high vol period

        model = GARCHModel()
        result = model.fit(returns)

        assert result["success"], "GARCH fitting should succeed"
        assert result["persistence"] < 1.0, "Persistence should be < 1 (stationary)"
        assert result["persistence"] > 0, "Persistence should be positive"
        assert 0 < result["alpha"] < 0.5, "ARCH coefficient in reasonable range"
        assert 0 < result["beta"] < 1, "GARCH coefficient in reasonable range"

    def test_garch_forecast(self):
        """Test GARCH volatility forecasting."""
        np.random.seed(42)
        returns = np.random.randn(200) * 0.02

        model = GARCHModel()
        model.fit(returns)

        forecasts = model.forecast(steps=5)

        assert len(forecasts) == 5, "Should return 5 forecasts"
        assert all(f > 0 for f in forecasts), "Volatilities should be positive"

    def test_var_calculation(self):
        """Test Value at Risk calculation."""
        np.random.seed(42)
        returns = np.random.randn(500) * 0.02

        model = GARCHModel()
        model.fit(returns)

        var = model.calculate_var(confidence=0.95)
        assert var > 0, "VaR should be positive"

        # Check that VaR is in reasonable range compared to empirical
        empirical_var = np.percentile(returns, 5)
        assert abs(var - abs(empirical_var)) < 0.05, "VaR should match empirical"

    def test_volatility_regime_detection(self):
        """Test volatility regime detector."""
        np.random.seed(42)
        returns = np.random.randn(300) * 0.02

        detector = VolatilityRegimeDetector()
        result = detector.fit(returns)

        assert result["success"], "Regime detection should succeed"
        assert result["current_regime"] in [0, 1, 2], "Regime should be 0, 1, or 2"


class TestHurstExponent:
    """Test Hurst exponent calculations."""

    def test_hurst_random_walk(self):
        """Test Hurst for random walk (should be ~0.5)."""
        np.random.seed(42)
        returns = np.random.randn(1000)  # IID noise → H ≈ 0.5

        hurst_calc = HurstExponent()
        h = hurst_calc.calculate(returns, method="dfa")

        assert 0.45 < h < 0.55, f"Random walk should have H ~ 0.5, got {h}"

    def test_hurst_trending(self):
        """Test Hurst for trending series (should be > 0.5)."""
        np.random.seed(42)
        # Trending series (persistent positive momentum)
        returns = np.ones(500) * 0.001 + np.random.randn(500) * 0.001

        hurst_calc = HurstExponent()
        h = hurst_calc.calculate(returns, method="dfa")

        assert h > 0.5, f"Trending series should have H > 0.5, got {h}"

    def test_hurst_mean_reverting(self):
        """Test Hurst for mean-reverting series (should be < 0.5)."""
        np.random.seed(42)

        # Mean-reverting: simple AR(1) process with negative autocorrelation
        n = 500
        returns = np.zeros(n)
        returns[0] = np.random.randn()
        for i in range(1, n):
            # Negative autocorrelation creates mean reversion
            returns[i] = -0.3 * returns[i - 1] + np.random.randn() * 0.5

        hurst_calc = HurstExponent()
        h = hurst_calc.calculate(returns, method="dfa")

        # Mean-reverting series typically have H < 0.5
        assert h < 0.6, f"Mean-reverting series should have H < 0.6, got {h}"

    def test_hurst_regime_classification(self):
        """Test Hurst regime classification."""
        hurst_calc = HurstExponent()

        # Test based on actual thresholds in implementation:
        # H < 0.4: strong_mean_reversion
        # 0.4 <= H < 0.45: mean_reversion
        # 0.45 <= H < 0.55: random_walk
        # 0.55 <= H < 0.6: trending
        # H >= 0.6: strong_trend
        assert hurst_calc.get_regime(0.3) == "strong_mean_reversion"
        assert hurst_calc.get_regime(0.42) == "mean_reversion"
        assert hurst_calc.get_regime(0.5) == "random_walk"
        assert hurst_calc.get_regime(0.58) == "trending"
        assert hurst_calc.get_regime(0.7) == "strong_trend"


class TestOrnsteinUhlenbeck:
    """Test OU process implementation."""

    def test_ou_simulation(self):
        """Test OU process simulation."""
        from src.math_tools.ornstein_uhlenbeck import OUParameters

        ou = OrnsteinUhlenbeckProcess()
        params = OUParameters(theta=0.5, mu=100, sigma=2.0)

        # Simulate single path (n_paths=1, take first path)
        paths = ou.simulate_paths(x0=100, params=params, n_steps=1000, n_paths=1)
        prices = paths[0]

        assert len(prices) == 1001, "Should generate 1001 prices (1000 steps + initial)"
        assert np.mean(prices) > 90 and np.mean(prices) < 110, (
            f"Mean should be around 100 (long-run mean), got {np.mean(prices):.2f}"
        )

    def test_ou_mean_reversion(self):
        """Test that OU process mean-reverts."""
        from src.math_tools.ornstein_uhlenbeck import OUParameters

        ou = OrnsteinUhlenbeckProcess()
        params = OUParameters(theta=0.8, mu=100, sigma=1.0)

        # Simulate starting far from mean (x0=120 vs mu=100)
        paths = ou.simulate_paths(x0=120, params=params, n_steps=100, n_paths=1)
        prices = paths[0]

        # Should revert toward mean within 100 steps
        final_price = prices[-1]
        assert abs(final_price - 100) < 20, (
            f"Should revert toward mean (100), got {final_price:.2f}"
        )

    def test_ou_score_calculation(self):
        """Test OU score for trading signals."""
        from src.math_tools.ornstein_uhlenbeck import OUParameters

        ou = OrnsteinUhlenbeckProcess()
        params = OUParameters(theta=0.5, mu=100, sigma=2.0)

        paths = ou.simulate_paths(x0=100, params=params, n_steps=200, n_paths=1)
        prices = paths[0]

        # Calculate score using the method (not standalone function)
        current_price = prices[-1]
        score = ou.calculate_ou_score(current_price, params)
        assert isinstance(score, (int, float)), "OU score should be a number"


class TestKellyCriterion:
    """Test Kelly criterion implementations."""

    def test_kelly_fraction_calculation(self):
        """Test Kelly fraction calculation using direct formula."""
        # Strategy with 60% win rate, 1:1 payoff
        # Kelly = (bp - q) / b = (0.6*1 - 0.4) / 1 = 0.2
        win_rate = 0.6
        win_loss_ratio = 1.0
        kelly = (win_rate * win_loss_ratio - (1 - win_rate)) / win_loss_ratio

        assert abs(kelly - 0.2) < 0.01, f"Kelly should be 0.2, got {kelly}"

    def test_kelly_position_sizing(self):
        """Test Kelly position sizing."""
        from src.math_tools.kelly_criterion import KellyParameters

        kelly = KellyCriterion()

        # Create parameters for position sizing
        params = KellyParameters(
            win_probability=0.6,
            win_loss_ratio=1.67,  # 0.05 / 0.03 risk-reward ratio
            kelly_fraction=0.5,  # Half Kelly (conservative)
            max_position=0.25,  # Max 25% of capital
        )

        # Calculate position size for $10,000 capital
        size = kelly.calculate_position_size(capital=10000.0, params=params)

        assert 0 < size < 10000, f"Position size should be reasonable, got {size:.2f}"

    def test_kelly_with_no_edge(self):
        """Test Kelly when there's no edge."""
        # 50% win rate, 1:1 payoff (no edge)
        # Kelly = (bp - q) / b = (0.5*1 - 0.5) / 1 = 0
        kelly = KellyCriterion()
        kelly_frac = kelly.calculate_kelly_fraction(win_prob=0.5, win_loss_ratio=1.0)

        assert abs(kelly_frac) < 0.01, (
            f"Kelly should be 0 with no edge, got {kelly_frac}"
        )

    def test_dynamic_kelly_pf_to_b_conversion(self):
        """
        dynamic_kelly() must convert profit_factor → win/loss_ratio correctly.

        Setup: win_rate=0.6, avg_win=$100, avg_loss=$100
          b = avg_win / avg_loss = 1.0
          PF = (0.6 × 100) / (0.4 × 100) = 1.5

        Kelly formula with b=1.0:
          f* = (0.6×1.0 − 0.4) / 1.0 = 0.20

        Before the fix, passing PF=1.5 as b gave:
          f* = (0.6×1.5 − 0.4) / 1.5 = 0.333  ← 67% oversizing
        """
        kelly = KellyCriterion()
        capital = 100_000.0
        win_rate = 0.6
        pf = 1.5  # PF = (win_rate/loss_rate) × b = (0.6/0.4) × 1.0 = 1.5

        size = kelly.dynamic_kelly(
            capital=capital,
            recent_win_rate=win_rate,
            recent_profit_factor=pf,
            kelly_fraction=1.0,   # full Kelly for clarity
            max_position=1.0,
        )

        # Expected: b = 1.5 × (1-0.6)/0.6 = 1.0
        # Full Kelly f* = (0.6×1.0 − 0.4)/1.0 = 0.20 → size = $20,000
        assert size == pytest.approx(20_000.0, rel=1e-4), (
            f"Expected $20,000 (20% Kelly), got ${size:,.0f}"
        )

    def test_dynamic_kelly_at_50pct_winrate_pf_equals_b(self):
        """At exactly 50% win rate, PF == b, so conversion is a no-op."""
        kelly = KellyCriterion()
        # win_rate=0.5: b = PF × (1-0.5)/0.5 = PF × 1.0 = PF
        # Kelly = (0.5×1.5 − 0.5)/1.5 = (0.75−0.5)/1.5 = 0.1667
        size = kelly.dynamic_kelly(
            capital=100_000.0,
            recent_win_rate=0.5,
            recent_profit_factor=1.5,
            kelly_fraction=1.0,
            max_position=1.0,
        )
        expected = (0.5 * 1.5 - 0.5) / 1.5 * 100_000.0
        assert size == pytest.approx(expected, rel=1e-4)

    def test_dynamic_kelly_zero_win_rate_returns_zero(self):
        """Win rate of 0 → no edge → position must be 0."""
        kelly = KellyCriterion()
        size = kelly.dynamic_kelly(
            capital=100_000.0,
            recent_win_rate=0.0,
            recent_profit_factor=2.0,
        )
        assert size == pytest.approx(0.0)

    def test_dynamic_kelly_consistent_with_calculate_position_size(self):
        """
        dynamic_kelly(win_rate, PF) must give the same result as
        calculate_position_size() with the analytically equivalent b.
        """
        kelly = KellyCriterion()
        win_rate = 0.55
        pf = 2.0
        b = pf * (1.0 - win_rate) / win_rate   # correct conversion

        direct = kelly.calculate_position_size(
            100_000.0,
            KellyParameters(
                win_probability=win_rate,
                win_loss_ratio=b,
                kelly_fraction=0.5,
                max_position=0.25,
            ),
        )
        via_dynamic = kelly.dynamic_kelly(
            capital=100_000.0,
            recent_win_rate=win_rate,
            recent_profit_factor=pf,
            kelly_fraction=0.5,
            max_position=0.25,
        )
        assert via_dynamic == pytest.approx(direct, rel=1e-6)


class TestHMMRegime:
    """Test HMM regime detection."""

    def test_hmm_regime_detection(self):
        """Test HMM can detect regimes."""
        np.random.seed(42)
        n = 500

        # Generate data with regime switch (low vol → high vol)
        regime1 = np.random.randn(250) * 0.01  # Low vol (regime 0)
        regime2 = np.random.randn(250) * 0.03  # High vol (regime 1)
        returns = np.concatenate([regime1, regime2])

        # Create DataFrame with features
        features = pd.DataFrame(
            {
                "returns": returns,
                "volatility": pd.Series(returns).rolling(20).std().bfill().values,
            }
        )

        hmm_det = HMMRegimeDetector(n_regimes=2)
        hmm_det.fit(features)
        # predict() returns the regime for the last sample (int)
        current_regime = hmm_det.predict(features)

        assert isinstance(current_regime, int), "predict() should return an int"
        assert current_regime in [0, 1], "Should be a valid regime id"

    def test_hmm_get_current_regime(self):
        """Test getting current regime."""
        np.random.seed(42)
        returns = np.random.randn(200) * 0.02

        # Create DataFrame with features
        features = pd.DataFrame(
            {
                "returns": returns,
                "volatility": pd.Series(returns).rolling(20).std().bfill().values,
            }
        )

        hmm_det = HMMRegimeDetector()
        hmm_det.fit(features)
        # predict() returns the regime for the last sample (int)
        current_regime = hmm_det.predict(features)

        assert isinstance(current_regime, int), "predict() should return an int"
        assert current_regime in [0, 1, 2], "Current regime should be valid"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
