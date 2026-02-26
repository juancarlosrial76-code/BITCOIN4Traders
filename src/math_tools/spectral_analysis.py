"""
Spectral Analysis & Fourier Transform for Financial Time Series
===============================================================

Frequency domain analysis for detecting dominant cycles, seasonality,
and filtering noise from financial data.

Mathematical Foundation:
------------------------
Spectral analysis transforms time-domain signals into frequency-domain
representations to identify periodic components.

Fast Fourier Transform (FFT):
    X_k = Σ_{n=0}^{N-1} x_n * e^{-i2πkn/N}

    where:
        N: Number of samples
        k: Frequency index (0 to N-1)
        x_n: Time-domain signal at sample n
        X_k: Frequency-domain representation at frequency k

Power Spectral Density (PSD):
    P(k) = |X_k|² = X_k * conj(X_k)

    Measures the power (energy) at each frequency component.

Hilbert Transform:
    x̂(t) = (1/π) * P.V. ∫[x(τ)/(t-τ)]dτ

    Creates analytic signal: z(t) = x(t) + i*x̂(t)
    Enables computation of instantaneous amplitude and phase.

Key Mathematical Relationships:
    Period: T = 1/f  (frequency to period conversion)
    Nyquist Frequency: f_N = Fs/2 (maximum detectable frequency)
    Frequency Resolution: Δf = Fs/N (frequency bins)

Professional Use Cases:
- Dominant cycle detection (e.g., 20-day, 4-year crypto cycles)
- Seasonality analysis (day-of-week, month-of-year effects)
- Noise reduction via frequency filtering
- Cycle composite indicators for trading signals
- Market regime identification via spectral features

Academic References:
- Ehlers, J.F. "Rocket Science for Traders" (John Wiley & Sons, 2001)
- Oppenheim & Schafer "Digital Signal Processing"
- Granger & Hatanaka "Spectral Analysis of Economic Time Series"

Dependencies:
- numpy: Numerical computations and FFT
- scipy.signal: Signal processing (hilbert, butter, filtfilt, welch)
- scipy.interpolate: Interpolation for cycle analysis
- pandas: Time series handling
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, List, Dict
from scipy import fft
from scipy.signal import hilbert, butter, filtfilt, welch
from scipy.interpolate import interp1d
import warnings

warnings.filterwarnings("ignore")


class SpectralAnalyzer:
    """
    Comprehensive spectral analysis for financial time series.

    Detects dominant cycles, seasonality patterns, and performs
    frequency domain filtering for trend extraction.

    Mathematical Background:
    -----------------------
    The SpectralAnalyzer uses FFT to decompose a time series into its
    frequency components. Key operations:

    1. FFT Computation:
       X[k] = Σ_{n=0}^{N-1} x[n] * e^{-j2πkn/N}

       - Converts time domain to frequency domain
       - Returns complex values (magnitude + phase)

    2. Power Spectrum:
       P[k] = |X[k]|²

       - Shows power (energy) at each frequency
       - Peak frequencies indicate dominant cycles

    3. Windowing:
       w[n] = window function (Hanning, Hamming, Blackman)

       - Reduces spectral leakage at signal boundaries
       - Improves accuracy of frequency estimates

    4. Filtering in Frequency Domain:
       y_filtered = IFFT( X * H(f) )

       where H(f) is the frequency response function:
       - Low-pass:  H(f) = 1 for |f| < f_c, else 0
       - High-pass: H(f) = 0 for |f| < f_c, else 1
       - Band-pass: H(f) = 1 for f_l < |f| < f_h, else 0

    Parameters:
    -----------
    sampling_rate : float
        Sampling frequency (default: 1.0 for daily data).
        Common values:
        - 1.0: Daily data (1 sample per day)
        - 24.0: Hourly data (24 samples per day)
        - 1440: Minute data (1440 samples per day)

    Attributes:
    ----------
    frequencies : np.ndarray
        Array of frequency values (positive frequencies only)

    power_spectrum : np.ndarray
        Power at each frequency (proportional to amplitude²)

    dominated_cycles : List[Dict]
        List of detected dominant cycles with properties

    Methods:
    --------
    compute_fft(signal, detrend, window)
        Compute FFT and power spectrum

    find_dominant_cycles(signal, n_cycles, min_period, max_period)
        Find strongest cyclical components

    cycle_composite(signal, cycles, lookahead)
        Reconstruct signal from selected cycles

    spectral_filter(signal, filter_type, cutoff)
        Apply frequency-domain filter

    extract_trend(signal, smoothness)
        Extract low-frequency trend component

    extract_cycles(signal, min_period, max_period)
        Extract cyclical component

    Example:
    -------
    >>> import numpy as np
    >>>
    >>> # Generate signal with known cycles
    >>> t = np.linspace(0, 200, 200)
    >>> signal = np.sin(2*np.pi*t/20) + 0.5*np.sin(2*np.pi*t/50)
    >>>
    >>> analyzer = SpectralAnalyzer(sampling_rate=1.0)
    >>> freqs, power = analyzer.compute_fft(signal)
    >>>
    >>> # Find dominant cycles
    >>> cycles = analyzer.find_dominant_cycles(signal, n_cycles=3)
    >>> for c in cycles:
    ...     print(f"Period: {c['period']:.1f}, Power: {c['power']:.3f}")
    """

    def __init__(self, sampling_rate: float = 1.0):
        """
        Initialize spectral analyzer.

        Args:
            sampling_rate: Sampling frequency (1.0 for daily data)
        """
        self.sampling_rate = sampling_rate
        self.frequencies = None
        self.power_spectrum = None
        self.dominated_cycles = None

    def compute_fft(
        self, signal: np.ndarray, detrend: bool = True, window: str = "hanning"
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Fast Fourier Transform and power spectrum.

        The FFT decomposes a time-domain signal into frequency components.

        Mathematical Background:
        ------------------------
        Discrete Fourier Transform:
            X[k] = Σ_{n=0}^{N-1} x[n] × e^{-j2πkn/N}

            where:
                N: Number of samples
                k: Frequency index (0 to N-1)
                x[n]: Time-domain sample n
                X[k]: Complex frequency coefficient at frequency k

        The FFT is an O(N log N) algorithm to compute this efficiently.

        Power Spectrum:
            P[k] = |X[k]|² = X[k] × conj(X[k])

            Represents the power (energy) at each frequency.
            Peak frequencies indicate dominant cycles.

        Frequency Resolution:
            Δf = Fs / N

            where:
                Fs: Sampling frequency
                N: Number of samples

            Longer signals = finer frequency resolution.

        Windowing:
            Applies a window function to reduce spectral leakage.
            Without windowing, discontinuities at signal edges create
            artificial frequency components.

            Window functions:
            - Hanning (default): Good general purpose, moderate leakage
            - Hamming: Slightly less leakage than Hanning
            - Blackman: Best leakage reduction, but broader peaks

        Args:
            signal: Input time series (prices or returns)
            detrend: Remove linear trend before FFT (reduces low-freq artifact)
            window: Window function ('hanning', 'hamming', 'blackman', or 'none')

        Returns:
            Tuple of (frequencies, power_spectrum)
                - frequencies: Array of frequency values (positive only)
                - power_spectrum: Power at each frequency
        """
        n = len(signal)

        # Step 1: Detrend
        # Remove linear trend to prevent low-frequency artifact in spectrum
        # The trend would appear as large low-frequency component
        if detrend:
            x = np.arange(n)
            # Linear regression: signal = slope * time + intercept
            slope, intercept = np.polyfit(x, signal, 1)
            # Remove the trend component
            signal = signal - (slope * x + intercept)

        # Step 2: Window
        # Apply window function to reduce spectral leakage
        # Without windowing, discontinuities at edges create artifacts
        if window == "hanning":
            w = np.hanning(n)
        elif window == "hamming":
            w = np.hamming(n)
        elif window == "blackman":
            w = np.blackman(n)
        else:
            w = np.ones(n)

        # Apply window: multiply signal by window to taper edges
        # This reduces discontinuities at the boundaries
        windowed_signal = signal * w

        # Step 3: Compute FFT
        # FFT computes the frequency-domain representation
        # Returns complex values: real + imaginary
        # |FFT| = amplitude, angle(FFT) = phase
        fft_values = fft.fft(windowed_signal)

        # Step 4: Compute Power Spectrum
        # Power = |X(f)|² = magnitude squared
        # This shows how much "power" is at each frequency
        power = np.abs(fft_values) ** 2

        # Step 5: Compute Frequency Axis
        # FFT frequencies: f[k] = k / (N * dt)
        # where dt = 1/sampling_rate
        # For sampling_rate=1 (daily data): f[k] = k/N cycles per day
        freqs = fft.fftfreq(n, d=1.0 / self.sampling_rate)

        # Step 6: Keep only positive frequencies
        # For real-valued signals, the spectrum is symmetric:
        # X(-f) = conj(X(f))
        # Positive frequencies contain all the information
        # Nyquist frequency (Fs/2) is the maximum detectable frequency
        positive_freqs = freqs[: n // 2]
        positive_power = power[: n // 2]

        # Store for later use
        self.frequencies = positive_freqs
        self.power_spectrum = positive_power

        return positive_freqs, positive_power

    def find_dominant_cycles(
        self,
        signal: np.ndarray,
        n_cycles: int = 3,
        min_period: float = 5,
        max_period: float = None,
    ) -> List[Dict]:
        """
        Find dominant cycles in the time series.

        Args:
            signal: Price or return series
            n_cycles: Number of dominant cycles to find
            min_period: Minimum cycle period (e.g., 5 days)
            max_period: Maximum cycle period (default: len(signal)/2)

        Returns:
            List of dictionaries with cycle information
            [{'period': 20, 'power': 0.85, 'frequency': 0.05}, ...]
        """
        if max_period is None:
            max_period = len(signal) / 2

        # Compute spectrum
        freqs, power = self.compute_fft(signal)

        # Convert frequencies to periods
        periods = 1.0 / (freqs + 1e-10)  # Add epsilon to avoid division by zero

        # Filter by period range
        mask = (periods >= min_period) & (periods <= max_period)
        filtered_periods = periods[mask]
        filtered_power = power[mask]
        filtered_freqs = freqs[mask]

        if len(filtered_power) == 0:
            return []

        # Find peaks
        from scipy.signal import find_peaks

        peaks, properties = find_peaks(
            filtered_power, height=np.percentile(filtered_power, 75)
        )

        # Sort by power
        peak_powers = filtered_power[peaks]
        sorted_indices = np.argsort(peak_powers)[::-1]

        # Get top N cycles
        top_cycles = []
        for i in sorted_indices[:n_cycles]:
            peak_idx = peaks[i]
            top_cycles.append(
                {
                    "period": float(filtered_periods[peak_idx]),
                    "power": float(
                        filtered_power[peak_idx] / np.max(filtered_power)
                    ),  # Normalized
                    "frequency": float(filtered_freqs[peak_idx]),
                    "amplitude": float(np.sqrt(filtered_power[peak_idx])),
                }
            )

        self.dominated_cycles = top_cycles
        return top_cycles

    def cycle_composite(
        self, signal: np.ndarray, cycles: List[float], lookahead: int = 0
    ) -> np.ndarray:
        """
        Create composite indicator from multiple cycles.

        Mathematical Formula:
            Composite(t) = Σ [Amplitude_i × sin(2π × t / Period_i + Phase_i)]

        Args:
            signal: Input series
            cycles: List of cycle periods to use (e.g., [20, 50, 200])
            lookahead: Number of periods to forecast

        Returns:
            Composite cycle indicator
        """
        n = len(signal)
        composite = np.zeros(n + lookahead)

        # Compute FFT once
        freqs, power = self.compute_fft(signal)
        fft_values = fft.fft(signal)

        for cycle_period in cycles:
            # Find closest frequency
            target_freq = 1.0 / cycle_period
            idx = np.argmin(np.abs(freqs - target_freq))

            # Extract amplitude and phase
            amplitude = np.abs(fft_values[idx]) / n
            phase = np.angle(fft_values[idx])

            # Reconstruct cycle
            t = np.arange(n + lookahead)
            cycle_component = amplitude * np.sin(2 * np.pi * t / cycle_period + phase)
            composite += cycle_component

        return composite

    def spectral_filter(
        self, signal: np.ndarray, filter_type: str = "lowpass", cutoff: float = 0.1
    ) -> np.ndarray:
        """
        Filter signal in frequency domain.

        Args:
            signal: Input series
            filter_type: 'lowpass', 'highpass', 'bandpass'
            cutoff: Cutoff frequency (0-0.5, where 0.5 = Nyquist)

        Returns:
            Filtered signal
        """
        n = len(signal)

        # Compute FFT
        fft_values = fft.fft(signal)
        freqs = fft.fftfreq(n)

        # Create filter mask
        if filter_type == "lowpass":
            # Keep low frequencies (trend)
            mask = np.abs(freqs) <= cutoff
        elif filter_type == "highpass":
            # Keep high frequencies (noise/cycles)
            mask = np.abs(freqs) >= cutoff
        elif filter_type == "bandpass":
            # Keep specific frequency band
            if isinstance(cutoff, tuple):
                low, high = cutoff
                mask = (np.abs(freqs) >= low) & (np.abs(freqs) <= high)
            else:
                mask = np.abs(freqs) <= cutoff
        else:
            mask = np.ones(n, dtype=bool)

        # Apply filter
        filtered_fft = fft_values.copy()
        filtered_fft[~mask] = 0

        # Inverse FFT
        filtered_signal = np.real(fft.ifft(filtered_fft))

        return filtered_signal

    def extract_trend(self, signal: np.ndarray, smoothness: float = 0.95) -> np.ndarray:
        """
        Extract trend component using lowpass filtering.

        Args:
            signal: Price series
            smoothness: 0-1, higher = smoother trend

        Returns:
            Trend component
        """
        cutoff = 1.0 - smoothness
        return self.spectral_filter(signal, "lowpass", cutoff=cutoff)

    def extract_cycles(
        self, signal: np.ndarray, min_period: float = 5, max_period: float = 100
    ) -> np.ndarray:
        """
        Extract cyclical component.

        Args:
            signal: Price series
            min_period: Minimum cycle period
            max_period: Maximum cycle period

        Returns:
            Cycle component
        """
        n = len(signal)

        # Convert periods to frequencies
        freq_low = 1.0 / max_period
        freq_high = 1.0 / min_period

        # Normalize to Nyquist frequency
        nyquist = 0.5
        cutoff_low = freq_low * nyquist
        cutoff_high = min(freq_high * nyquist, nyquist)

        return self.spectral_filter(
            signal, "bandpass", cutoff=(cutoff_low, cutoff_high)
        )


class HilbertTransformAnalyzer:
    """
    Hilbert Transform analysis for instantaneous frequency and amplitude.

    Provides real-time cycle analysis without look-ahead bias.
    """

    def __init__(self):
        """Initialize Hilbert analyzer."""
        self.analytic_signal = None
        self.instantaneous_phase = None
        self.instantaneous_frequency = None
        self.instantaneous_amplitude = None

    def compute(self, signal: np.ndarray, window: int = 20) -> Dict:
        """
        Compute Hilbert transform and derived metrics.

        The Hilbert transform creates an analytic signal that allows
        us to compute instantaneous amplitude and phase - properties
        that change continuously over time (unlike FFT which gives
        global frequency content).

        Mathematical Background:
        ------------------------
        Hilbert Transform:
            x̂(t) = (1/π) × P.V. ∫[x(τ)/(t-τ)] dτ

            where P.V. denotes Cauchy principal value integral.

        This is equivalent to shifting each frequency component by 90°.

        Analytic Signal:
            z(t) = x(t) + j × x̂(t)

        This is a complex-valued signal where:
            |z(t)| = instantaneous amplitude (envelope)
            arg(z(t)) = instantaneous phase

        Instantaneous Amplitude:
            A(t) = |z(t)| = √[x²(t) + x̂²(t)]

            This is the "envelope" of the signal - the smooth curve
            that surrounds the oscillations.

        Instantaneous Phase:
            φ(t) = arg(z(t)) = arctan[x̂(t)/x(t)]

        Instantaneous Frequency:
            ω(t) = dφ/dt

            The rate of change of phase gives instantaneous frequency.

            f(t) = ω(t) / (2π)

        Advantages over FFT:
        - Provides time-localized information (no look-ahead bias)
        - Can track changing frequencies within a signal
        - Useful for real-time trading applications

        Args:
            signal: Input time series (typically price returns)
            window: Window size for smoothing (not used in basic implementation)

        Returns:
            Dictionary with keys:
                - amplitude: Instantaneous amplitude (envelope)
                - phase: Instantaneous phase (unwrapped)
                - frequency: Instantaneous frequency
                - period: Instantaneous period = 1/frequency
                - current_period: Most recent period value
                - current_amplitude: Most recent amplitude value
                - signal: Real part of analytic signal
        """
        # Apply Hilbert transform using scipy.signal.hilbert
        # This returns the analytic signal: x + j*H[x]
        # The imaginary part is the Hilbert transform of x
        analytic_signal = hilbert(signal)

        # Instantaneous amplitude: |z(t)| = √(x² + x̂²)
        # This is the envelope - how far the signal deviates from zero
        # High amplitude = strong cycle, low amplitude = weak cycle
        amplitude = np.abs(analytic_signal)

        # Instantaneous phase: arg(z(t))
        # Use np.unwrap to remove 2π discontinuities
        # This gives a continuously increasing/decreasing phase
        phase = np.unwrap(np.angle(analytic_signal))

        # Instantaneous frequency: dφ/dt / 2π
        # The derivative of phase gives angular frequency
        # Convert to cycles per sample by dividing by 2π
        # This tells us the dominant frequency at each point in time
        freq = np.diff(phase) / (2.0 * np.pi)

        # Pad last value to maintain original signal length
        freq = np.append(freq, freq[-1])

        # Convert frequency to period: T = 1/f
        # This is more intuitive for trading (e.g., "20-day cycle")
        # Add small epsilon to avoid division by zero
        period = 1.0 / (np.abs(freq) + 1e-10)

        # Clip to valid range:
        # - Minimum: 2 samples (can't have period shorter than 2 points)
        # - Maximum: entire signal length
        period = np.clip(period, 2, len(signal))

        # Store for later access
        self.analytic_signal = analytic_signal
        self.instantaneous_amplitude = amplitude
        self.instantaneous_phase = phase
        self.instantaneous_frequency = freq
        self.instantaneous_period = period

        return {
            "amplitude": amplitude,
            "phase": phase,
            "frequency": freq,
            "period": period,
            "current_period": float(period[-1]),
            "current_amplitude": float(amplitude[-1]),
            "signal": np.real(analytic_signal),
        }

    def get_cycle_state(self, recent_window: int = 20) -> Dict:
        """
        Get current cycle state for trading.

        Args:
            recent_window: Number of recent periods to analyze

        Returns:
            Cycle state dictionary
        """
        if self.instantaneous_period is None:
            raise ValueError("Must call compute() first")

        recent_periods = self.instantaneous_period[-recent_window:]
        recent_amplitudes = self.instantaneous_amplitude[-recent_window:]

        return {
            "dominant_period": float(np.median(recent_periods)),
            "period_std": float(np.std(recent_periods)),
            "amplitude": float(np.mean(recent_amplitudes)),
            "amplitude_trend": "increasing"
            if recent_amplitudes[-1] > recent_amplitudes[0]
            else "decreasing",
            "cycle_strength": float(
                np.mean(recent_amplitudes) / (np.std(recent_amplitudes) + 1e-6)
            ),
        }


class AdaptiveCycleIndicator:
    """
    Adaptive cycle indicator that adjusts to changing market conditions.

    Automatically detects dominant cycle and generates trading signals.
    """

    def __init__(self, lookback: int = 200, adapt_period: int = 50):
        """
        Initialize adaptive cycle indicator.

        Args:
            lookback: Lookback window for cycle detection
            adapt_period: Period for cycle adaptation
        """
        self.lookback = lookback
        self.adapt_period = adapt_period
        self.current_cycle = None
        self.cycle_history = []
        self.spectral_analyzer = SpectralAnalyzer()

    def update(self, price: float, timestamp=None) -> Dict:
        """
        Update indicator with new price.

        Args:
            price: Current price
            timestamp: Optional timestamp

        Returns:
            Current cycle state and signal
        """
        self.cycle_history.append(price)

        # Keep only lookback period
        if len(self.cycle_history) > self.lookback:
            self.cycle_history = self.cycle_history[-self.lookback :]

        # Not enough data yet
        if len(self.cycle_history) < self.adapt_period:
            return {"signal": 0, "cycle_period": None, "confidence": 0}

        # Adapt cycle every adapt_period
        if len(self.cycle_history) % self.adapt_period == 0:
            signal_array = np.array(self.cycle_history)

            # Find dominant cycle
            cycles = self.spectral_analyzer.find_dominant_cycles(
                signal_array,
                n_cycles=1,
                min_period=5,
                max_period=len(signal_array) // 3,
            )

            if cycles:
                self.current_cycle = cycles[0]["period"]

        # Generate signal if we have a cycle
        if self.current_cycle:
            # Compute phase in current cycle
            position_in_cycle = len(self.cycle_history) % int(self.current_cycle)
            phase = position_in_cycle / self.current_cycle

            # Generate signal based on cycle phase
            # Buy at trough (phase ~ 0.75-1.0), Sell at peak (phase ~ 0.25-0.5)
            if phase > 0.75 or phase < 0.1:
                signal = 1  # Buy
            elif 0.25 < phase < 0.5:
                signal = -1  # Sell
            else:
                signal = 0  # Neutral

            return {
                "signal": signal,
                "cycle_period": self.current_cycle,
                "phase": phase,
                "confidence": 0.7 if self.current_cycle else 0,
            }

        return {"signal": 0, "cycle_period": None, "confidence": 0}


class SeasonalityAnalyzer:
    """
    Analyze seasonal patterns in financial data.

    Detects day-of-week, month-of-year, and other seasonal effects.
    """

    def __init__(self):
        """Initialize seasonality analyzer."""
        self.patterns = {}

    def analyze_day_of_week(self, returns: pd.Series, index: pd.DatetimeIndex) -> Dict:
        """
        Analyze day-of-week seasonality.

        Args:
            returns: Return series
            index: Datetime index

        Returns:
            Dictionary with day-of-week statistics
        """
        df = pd.DataFrame({"returns": returns, "day": index.dayofweek})

        day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
        stats = {}

        for day in range(5):  # Monday to Friday
            day_returns = df[df["day"] == day]["returns"]

            stats[day_names[day]] = {
                "mean_return": float(day_returns.mean()),
                "win_rate": float((day_returns > 0).mean()),
                "sharpe": float(day_returns.mean() / (day_returns.std() + 1e-6)),
                "count": int(len(day_returns)),
            }

        # Find best and worst days
        means = {day: stats[day]["mean_return"] for day in day_names}
        stats["best_day"] = max(means, key=means.get)
        stats["worst_day"] = min(means, key=means.get)

        return stats

    def analyze_month_of_year(
        self, returns: pd.Series, index: pd.DatetimeIndex
    ) -> Dict:
        """
        Analyze month-of-year seasonality.

        Args:
            returns: Return series
            index: Datetime index

        Returns:
            Month-of-year statistics
        """
        df = pd.DataFrame({"returns": returns, "month": index.month})

        month_names = [
            "Jan",
            "Feb",
            "Mar",
            "Apr",
            "May",
            "Jun",
            "Jul",
            "Aug",
            "Sep",
            "Oct",
            "Nov",
            "Dec",
        ]
        stats = {}

        for month in range(1, 13):
            month_returns = df[df["month"] == month]["returns"]

            if len(month_returns) > 0:
                stats[month_names[month - 1]] = {
                    "mean_return": float(month_returns.mean()),
                    "win_rate": float((month_returns > 0).mean()),
                    "volatility": float(month_returns.std()),
                    "count": int(len(month_returns)),
                }

        # Find best and worst months
        means = {month: stats[month]["mean_return"] for month in month_names}
        stats["best_month"] = max(means, key=means.get)
        stats["worst_month"] = min(means, key=means.get)

        return stats

    def detect_holiday_effects(
        self, prices: pd.Series, index: pd.DatetimeIndex, holidays: List[str] = None
    ) -> Dict:
        """
        Detect effects around holidays or special dates.

        Args:
            prices: Price series
            index: Datetime index
            holidays: List of holiday dates (e.g., ['01-01', '12-25'])

        Returns:
            Holiday effect statistics
        """
        if holidays is None:
            holidays = [
                "01-01",
                "07-04",
                "12-25",
            ]  # New Year, Independence Day, Christmas

        effects = {}

        for holiday in holidays:
            month, day = map(int, holiday.split("-"))

            # Find dates around holiday
            mask = (index.month == month) & (index.day == day)
            holiday_dates = index[mask]

            pre_returns = []
            post_returns = []

            for holiday_date in holiday_dates:
                try:
                    # Get price before and after
                    loc = prices.index.get_loc(holiday_date)
                    if loc > 0 and loc < len(prices) - 1:
                        pre_ret = prices.iloc[loc] / prices.iloc[loc - 1] - 1
                        post_ret = prices.iloc[loc + 1] / prices.iloc[loc] - 1
                        pre_returns.append(pre_ret)
                        post_returns.append(post_ret)
                except:
                    continue

            if pre_returns and post_returns:
                effects[holiday] = {
                    "pre_holiday_mean": float(np.mean(pre_returns)),
                    "post_holiday_mean": float(np.mean(post_returns)),
                    "pre_holiday_winrate": float(np.mean([r > 0 for r in pre_returns])),
                    "post_holiday_winrate": float(
                        np.mean([r > 0 for r in post_returns])
                    ),
                }

        return effects


# Utility functions
def compute_dominant_cycle(prices: np.ndarray, min_period: int = 5) -> Optional[float]:
    """
    Quick function to find dominant cycle period.

    Args:
        prices: Price series
        min_period: Minimum cycle period

    Returns:
        Dominant cycle period or None
    """
    analyzer = SpectralAnalyzer()
    cycles = analyzer.find_dominant_cycles(prices, n_cycles=1, min_period=min_period)

    if cycles:
        return cycles[0]["period"]
    return None


def cycle_based_signal(prices: np.ndarray, cycle_period: float) -> int:
    """
    Generate trading signal based on cycle position.

    Args:
        prices: Price series
        cycle_period: Cycle period

    Returns:
        Trading signal (-1, 0, 1)
    """
    n = len(prices)
    position = n % int(cycle_period)
    phase = position / cycle_period

    # Buy at cycle bottom (phase ~0.75-1.0)
    if phase > 0.75 or phase < 0.1:
        return 1
    # Sell at cycle top (phase ~0.25-0.5)
    elif 0.25 < phase < 0.5:
        return -1
    else:
        return 0


def remove_seasonality(prices: pd.Series, period: int = 20) -> pd.Series:
    """
    Remove seasonal component from prices.

    Args:
        prices: Price series
        period: Seasonal period (e.g., 20 for monthly in trading days)

    Returns:
        Deseasonalized prices
    """
    # Calculate seasonal component
    seasonal = prices.groupby(prices.index % period).transform("mean")

    # Remove seasonal effect
    deseasonalized = prices - seasonal + prices.mean()

    return deseasonalized


def spectral_edge_detection(
    prices: np.ndarray, short_period: int = 10, long_period: int = 50
) -> float:
    """
    Detect trend changes using spectral analysis.

    Compares power in short cycles vs long cycles to detect
    trend strength changes.

    Args:
        prices: Price series
        short_period: Short cycle period
        long_period: Long cycle period

    Returns:
        Edge strength (-1 to 1, positive = uptrend)
    """
    analyzer = SpectralAnalyzer()
    freqs, power = analyzer.compute_fft(prices)

    # Find power in short and long cycles
    short_freq = 1.0 / short_period
    long_freq = 1.0 / long_period

    short_power = np.sum(
        power[(freqs >= short_freq * 0.8) & (freqs <= short_freq * 1.2)]
    )
    long_power = np.sum(power[(freqs >= long_freq * 0.8) & (freqs <= long_freq * 1.2)])

    # Normalize
    total_power = short_power + long_power
    if total_power > 0:
        edge = (long_power - short_power) / total_power
    else:
        edge = 0

    return float(edge)
