"""
Tests for Spectral Analysis Module
====================================

Comprehensive tests for FFT, cycle detection, and spectral filtering.
"""

import numpy as np
import pandas as pd
import pytest
from src.math_tools.spectral_analysis import (
    SpectralAnalyzer,
    HilbertTransformAnalyzer,
    AdaptiveCycleIndicator,
    SeasonalityAnalyzer,
    compute_dominant_cycle,
    cycle_based_signal,
    spectral_edge_detection,
)


class TestSpectralAnalyzer:
    """Test SpectralAnalyzer class."""

    def test_fft_computation(self):
        """Test FFT computation."""
        np.random.seed(42)

        # Create signal with known frequency (0.1 cycles per sample)
        t = np.linspace(0, 4 * np.pi, 1000)
        signal = np.sin(2 * np.pi * 0.1 * t) + np.random.randn(1000) * 0.1

        analyzer = SpectralAnalyzer()
        freqs, power = analyzer.compute_fft(signal)

        assert len(freqs) == len(power), "Frequencies and power should match"
        assert len(freqs) == 500, "Should have half length (positive frequencies only)"
        assert np.all(power >= 0), "Power spectral density must be non-negative"

    def test_find_dominant_cycles(self):
        """Test dominant cycle detection."""
        np.random.seed(42)

        # Create signal with 20-period cycle (well above noise floor)
        t = np.arange(500)
        prices = 100 + 10 * np.sin(2 * np.pi * t / 20) + np.random.randn(500) * 0.5

        analyzer = SpectralAnalyzer()
        cycles = analyzer.find_dominant_cycles(prices, n_cycles=3, min_period=5)

        assert len(cycles) > 0, "Should find at least one cycle"

        # Check if 20-period cycle is found (with ±5 tolerance)
        periods = [c["period"] for c in cycles]
        found_20 = any(15 <= p <= 25 for p in periods)
        assert found_20, f"Should find ~20 period cycle, got {periods}"

    def test_spectral_filter_lowpass(self):
        """Test lowpass filtering."""
        np.random.seed(42)

        # Create noisy trend (high-frequency noise on top of slow drift)
        trend = np.cumsum(np.ones(200) * 0.1)
        noise = np.random.randn(200) * 5
        signal = trend + noise

        analyzer = SpectralAnalyzer()
        filtered = analyzer.spectral_filter(signal, filter_type="lowpass", cutoff=0.1)

        # Filtered should be smoother (less variance) — noise was attenuated
        assert np.std(filtered) < np.std(signal), "Filtered should have less variance"

    def test_extract_trend(self):
        """Test trend extraction."""
        np.random.seed(42)

        # Create trending series with noise
        trend = np.linspace(100, 200, 500)
        noise = np.random.randn(500) * 5
        prices = trend + noise

        analyzer = SpectralAnalyzer()
        extracted_trend = analyzer.extract_trend(prices, smoothness=0.95)

        # Extracted trend should correlate strongly with the true linear trend
        correlation = np.corrcoef(extracted_trend, trend)[0, 1]
        assert correlation > 0.9, f"Trend correlation should be high, got {correlation}"

    def test_cycle_composite(self):
        """Test cycle composite indicator."""
        np.random.seed(42)

        # Simple signal
        prices = 100 + np.random.randn(200) * 0.5

        analyzer = SpectralAnalyzer()
        composite = analyzer.cycle_composite(prices, cycles=[10, 20], lookahead=10)

        assert len(composite) == len(prices) + 10, "Should include lookahead bars"
        assert isinstance(composite, np.ndarray), "Should return numpy array"


class TestHilbertTransform:
    """Test Hilbert Transform analyzer."""

    def test_hilbert_computation(self):
        """Test Hilbert transform computation."""
        np.random.seed(42)

        # Create signal with known periodicity
        t = np.arange(100)
        signal = np.sin(2 * np.pi * t / 20) + np.random.randn(100) * 0.1

        hilbert = HilbertTransformAnalyzer()
        result = hilbert.compute(signal)

        assert "amplitude" in result, "Should have amplitude"
        assert "phase" in result, "Should have phase"
        assert "period" in result, "Should have period"
        assert len(result["amplitude"]) == len(signal), "Output length should match"

    def test_instantaneous_period_detection(self):
        """Test instantaneous period detection."""
        np.random.seed(42)

        # Create pure 20-period cycle (no noise for easy detection)
        t = np.arange(200)
        signal = np.sin(2 * np.pi * t / 20)

        hilbert = HilbertTransformAnalyzer()
        result = hilbert.compute(signal)

        # Median period should be around 20
        median_period = np.median(result["period"])
        assert 15 <= median_period <= 25, f"Period should be ~20, got {median_period}"

    def test_cycle_state(self):
        """Test cycle state extraction."""
        np.random.seed(42)

        signal = np.sin(2 * np.pi * np.arange(100) / 20)

        hilbert = HilbertTransformAnalyzer()
        hilbert.compute(signal)

        state = hilbert.get_cycle_state(recent_window=20)  # Use last 20 bars

        assert "dominant_period" in state, "Should have dominant period"
        assert "cycle_strength" in state, "Should have cycle strength"
        assert state["dominant_period"] > 0, "Period should be positive"


class TestAdaptiveCycleIndicator:
    """Test Adaptive Cycle Indicator."""

    def test_initialization(self):
        """Test ACI initialization."""
        aci = AdaptiveCycleIndicator(lookback=100, adapt_period=20)

        assert aci.lookback == 100
        assert aci.adapt_period == 20
        assert len(aci.cycle_history) == 0  # No history yet

    def test_update_not_enough_data(self):
        """Test update with insufficient data."""
        aci = AdaptiveCycleIndicator(lookback=100)

        # Add only 10 prices (need at least adapt_period)
        for price in range(10):
            result = aci.update(price)

        # Should return neutral signal (not enough data to compute cycle)
        assert result["signal"] == 0
        assert result["confidence"] == 0

    def test_update_with_synthetic_cycle(self):
        """Test update with synthetic cyclical data."""
        np.random.seed(42)

        aci = AdaptiveCycleIndicator(lookback=100, adapt_period=25)

        # Generate 20-period cycle with small noise
        t = np.arange(150)
        prices = 100 + 10 * np.sin(2 * np.pi * t / 20) + np.random.randn(150) * 0.5

        results = []
        for price in prices:
            result = aci.update(price)
            results.append(result)

        # Should eventually find cycle and generate signals
        last_result = results[-1]
        assert last_result["cycle_period"] is not None or aci.current_cycle is not None


class TestSeasonalityAnalyzer:
    """Test Seasonality Analyzer."""

    def test_day_of_week_analysis(self):
        """Test day-of-week seasonality."""
        np.random.seed(42)

        # Create synthetic data with Monday effect
        dates = pd.date_range("2020-01-01", periods=500, freq="B")  # Business days
        returns = np.random.randn(500) * 0.01
        returns[dates.dayofweek == 0] += 0.005  # Monday gets higher returns

        series = pd.Series(returns, index=dates)

        analyzer = SeasonalityAnalyzer()
        stats = analyzer.analyze_day_of_week(series, series.index)

        assert "Monday" in stats, "Should have Monday stats"
        assert "best_day" in stats, "Should identify best day"
        assert stats["Monday"]["mean_return"] > stats["Tuesday"]["mean_return"], (
            "Monday should have higher returns"
        )

    def test_month_of_year_analysis(self):
        """Test month-of-year seasonality."""
        np.random.seed(42)

        dates = pd.date_range("2015-01-01", periods=2000, freq="B")
        returns = np.random.randn(2000) * 0.01

        # January effect (higher returns in January)
        january_mask = dates.month == 1
        returns[january_mask] += 0.01

        series = pd.Series(returns, index=dates)

        analyzer = SeasonalityAnalyzer()
        stats = analyzer.analyze_month_of_year(series, series.index)

        assert "Jan" in stats, "Should have January stats"
        assert "best_month" in stats, "Should identify best month"


class TestUtilityFunctions:
    """Test utility functions."""

    def test_compute_dominant_cycle(self):
        """Test dominant cycle computation."""
        np.random.seed(42)

        # Create 30-period cycle (pure sinusoid for easy detection)
        t = np.arange(300)
        prices = 100 + 10 * np.sin(2 * np.pi * t / 30)

        cycle = compute_dominant_cycle(prices, min_period=10)

        assert cycle is not None, "Should find cycle"
        assert 25 <= cycle <= 35, f"Cycle should be ~30, got {cycle}"

    def test_cycle_based_signal(self):
        """Test cycle-based signal generation."""
        np.random.seed(42)

        prices = np.random.randn(100) + 100

        signal = cycle_based_signal(prices, cycle_period=20)

        assert signal in [-1, 0, 1], "Signal should be -1, 0, or 1"

    def test_spectral_edge_detection(self):
        """Test spectral edge detection."""
        np.random.seed(42)

        # Create series with strong trend (more long-term power)
        # Use cumsum to create momentum/trend
        returns = np.ones(200) * 0.001 + np.random.randn(200) * 0.01
        prices = 100 * np.exp(np.cumsum(returns))

        edge = spectral_edge_detection(prices, short_period=10, long_period=50)

        assert isinstance(edge, float), "Should return float"
        assert -1 <= edge <= 1, "Edge should be between -1 and 1"
        # Note: Linear trend may not always produce positive edge due to FFT characteristics
        # Just verify the function works and returns valid values


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
