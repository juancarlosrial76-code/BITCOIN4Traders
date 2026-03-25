"""Tests for fast_kernels JIT-compiled math primitives."""

import numpy as np
import pytest

from src.math_tools.fast_kernels import (
    _kernel_ema,
    _kernel_rolling_mean_std,
    _kernel_rsi_wilder,
)


# ---------------------------------------------------------------------------
# _kernel_ema
# ---------------------------------------------------------------------------


class TestKernelEma:
    """Tests for _kernel_ema (standard EMA, alpha = 2 / (span + 1))."""

    def test_output_length_equals_input(self) -> None:
        """Output array must have the same length as the input array."""
        closes = np.array([10.0, 11.0, 12.0, 11.5, 13.0])
        ema = _kernel_ema(closes, span=3)
        assert len(ema) == len(closes)

    def test_alpha_calculation(self) -> None:
        """First update must use alpha = 2 / (span + 1)."""
        closes = np.array([10.0, 11.0, 12.0, 11.5, 13.0])
        span = 3
        # alpha = 2 / (3 + 1) = 0.5
        alpha = 2.0 / (span + 1)
        ema = _kernel_ema(closes, span=span)
        expected_ema1 = alpha * closes[1] + (1.0 - alpha) * closes[0]
        np.testing.assert_allclose(ema[1], expected_ema1)

    def test_first_value_equals_first_close(self) -> None:
        """EMA[0] must be seeded with the first close price."""
        closes = np.array([10.0, 11.0, 12.0, 11.5, 13.0])
        ema = _kernel_ema(closes, span=3)
        assert ema[0] == closes[0]

    def test_known_values(self) -> None:
        """Spot-check computed EMA values against manual calculation."""
        closes = np.array([10.0, 11.0, 12.0, 11.5, 13.0])
        span = 3
        alpha = 2.0 / (span + 1)  # 0.5
        ema = _kernel_ema(closes, span=span)

        # ema[0] = 10.0
        # ema[1] = 0.5 * 11.0 + 0.5 * 10.0 = 10.5
        # ema[2] = 0.5 * 12.0 + 0.5 * 10.5 = 11.25
        expected = [10.0, 10.5, 11.25]
        for i, exp in enumerate(expected):
            np.testing.assert_allclose(
                ema[i], exp, rtol=1e-9, err_msg=f"ema[{i}] mismatch"
            )

    def test_single_element(self) -> None:
        """A single-element input must return that same element."""
        closes = np.array([42.0])
        ema = _kernel_ema(closes, span=5)
        assert len(ema) == 1
        assert ema[0] == 42.0


# ---------------------------------------------------------------------------
# _kernel_rolling_mean_std
# ---------------------------------------------------------------------------


class TestKernelRollingMeanStd:
    """Tests for _kernel_rolling_mean_std (Bessel-corrected rolling stats)."""

    def test_nan_for_first_period_minus_one_elements(self) -> None:
        """The first (period - 1) elements must be NaN for both mean and std."""
        closes = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        period = 3
        means, stds = _kernel_rolling_mean_std(closes, period=period)

        # Indices 0 .. period-2 must be NaN
        for i in range(period - 1):
            assert np.isnan(means[i]), f"means[{i}] should be NaN"
            assert np.isnan(stds[i]), f"stds[{i}] should be NaN"

    def test_correct_mean(self) -> None:
        """Rolling mean at index (period - 1) must equal the arithmetic mean of the window."""
        closes = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        means, _ = _kernel_rolling_mean_std(closes, period=3)
        # Window [1, 2, 3] -> mean = 2.0
        np.testing.assert_allclose(means[2], 2.0)
        # Window [2, 3, 4] -> mean = 3.0
        np.testing.assert_allclose(means[3], 3.0)
        # Window [3, 4, 5] -> mean = 4.0
        np.testing.assert_allclose(means[4], 4.0)

    def test_correct_std_bessel_corrected(self) -> None:
        """Rolling std must use Bessel correction (ddof=1)."""
        closes = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        _, stds = _kernel_rolling_mean_std(closes, period=3)
        # std([1, 2, 3], ddof=1) == 1.0
        np.testing.assert_allclose(stds[2], 1.0, rtol=1e-9)

    def test_output_length_equals_input(self) -> None:
        """Both output arrays must have the same length as the input."""
        closes = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        means, stds = _kernel_rolling_mean_std(closes, period=3)
        assert len(means) == len(closes)
        assert len(stds) == len(closes)

    def test_minimum_valid_period(self) -> None:
        """period=2 is the minimum valid value (Bessel correction uses period-1)."""
        closes = np.array([4.0, 6.0, 8.0])
        means, stds = _kernel_rolling_mean_std(closes, period=2)
        # Index 0 must be NaN (only period-1=1 warmup bar)
        assert np.isnan(means[0])
        # Index 1: window [4, 6] -> mean=5, std(ddof=1)=sqrt(2)
        np.testing.assert_allclose(means[1], 5.0)
        np.testing.assert_allclose(stds[1], np.std([4.0, 6.0], ddof=1))


# ---------------------------------------------------------------------------
# _kernel_rsi_wilder
# ---------------------------------------------------------------------------


class TestKernelRsiWilder:
    """Tests for _kernel_rsi_wilder (Wilder's smoothed RSI)."""

    def test_output_length_equals_input(self) -> None:
        """Output array must have the same length as the input array."""
        closes = np.array([10.0, 11.0, 12.0, 13.0, 14.0, 15.0])
        rsi = _kernel_rsi_wilder(closes, period=3)
        assert len(rsi) == len(closes)

    def test_nan_at_index_zero(self) -> None:
        """RSI[0] must be NaN (no prior bar to compute a difference)."""
        closes = np.array([10.0, 11.0, 12.0, 13.0, 14.0, 15.0])
        rsi = _kernel_rsi_wilder(closes, period=3)
        assert np.isnan(rsi[0])

    def test_warmup_bars_are_nan(self) -> None:
        """Bars 0 .. period-1 must be NaN (insufficient history)."""
        closes = np.array([10.0, 11.0, 12.0, 13.0, 14.0, 15.0])
        period = 3
        rsi = _kernel_rsi_wilder(closes, period=period)
        for i in range(period):
            assert np.isnan(rsi[i]), f"rsi[{i}] should be NaN during warmup"

    def test_range_0_to_100(self) -> None:
        """All non-NaN RSI values must be within [0, 100]."""
        closes = np.array([10.0, 11.0, 12.0, 13.0, 14.0, 15.0])
        rsi = _kernel_rsi_wilder(closes, period=3)
        valid = rsi[~np.isnan(rsi)]
        assert np.all(valid >= 0.0), "RSI below 0 detected"
        assert np.all(valid <= 100.0), "RSI above 100 detected"

    def test_all_gains_gives_rsi_100(self) -> None:
        """A monotonically rising series with no losses must produce RSI = 100."""
        closes = np.array([10.0, 11.0, 12.0, 13.0, 14.0, 15.0])
        rsi = _kernel_rsi_wilder(closes, period=3)
        # From index `period` onward there are only gains -> RSI must be 100
        valid = rsi[~np.isnan(rsi)]
        np.testing.assert_allclose(valid, 100.0)

    def test_all_losses_gives_rsi_0(self) -> None:
        """A monotonically falling series with no gains must produce RSI = 0."""
        closes = np.array([15.0, 14.0, 13.0, 12.0, 11.0, 10.0])
        rsi = _kernel_rsi_wilder(closes, period=3)
        valid = rsi[~np.isnan(rsi)]
        np.testing.assert_allclose(valid, 0.0)

    def test_insufficient_data_returns_all_nan(self) -> None:
        """When len(closes) < period + 1, every element must be NaN."""
        closes = np.array([10.0, 11.0, 12.0])
        rsi = _kernel_rsi_wilder(closes, period=5)
        assert np.all(np.isnan(rsi))
