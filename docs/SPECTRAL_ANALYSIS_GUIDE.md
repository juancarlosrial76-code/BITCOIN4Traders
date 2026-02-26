# Spectral Analysis & Fourier Transform Guide

## Overview

**Spectral Analysis** transforms time series into the frequency domain to discover hidden cycles, seasonalities, and trends.

### Why Spectral Analysis Matters:

1. **Discover Dominant Cycles** (e.g., Bitcoin 4-year halving cycle)
2. **Remove Noise** (separate trend from noise)
3. **Find Seasonalities** (Monday effect, month-end effect)
4. **Identify Market Regimes** (cyclic vs. trend-based phases)

**Mathematical Foundation:** Fourier Transform & Power Spectral Density (PSD)

---

## 1. Core Concepts

### 1.1 Fourier Transform

Transforms time series from time domain to frequency domain:

```
X(f) = ∫ x(t) * e^(-2πift) dt

Where:
x(t) = Price at time t
X(f) = Frequency component at frequency f
```

### 1.2 Power Spectral Density (PSD)

Shows the strength of each frequency:

```
PSD(f) = |X(f)|²

High values at f = 0.05 → Dominant cycle at period = 1/0.05 = 20 days
```

### 1.3 Dominant Cycles

Frequencies with highest power are the dominant market cycles.

---

## 2. Implementations

### 2.1 SpectralAnalyzer - Basic Analysis

```python
from math_tools import SpectralAnalyzer

# Initialize
analyzer = SpectralAnalyzer(sampling_rate=1.0)  # 1.0 for daily data

# Compute FFT
freqs, power = analyzer.compute_fft(prices, detrend=True, window='hanning')

# Find dominant cycles
dominant_cycles = analyzer.find_dominant_cycles(
    prices, 
    n_cycles=3,
    min_period=10,    # At least 10 days
    max_period=500    # At most 500 days
)

print("Dominant Cycles:")
for cycle in dominant_cycles:
    print(f"  Period: {cycle['period']:.1f} days")
    print(f"  Power: {cycle['power']:.2f}")
    print(f"  Amplitude: ${cycle['amplitude']:.2f}")
```

**Example Output:**
```
Dominant Cycles:
  Period: 140.0 days (20 weeks)
  Power: 0.85
  Amplitude: $850.00
  
  Period: 730.0 days (4 years)
  Power: 0.72
  Amplitude: $2,400.00
  
  Period: 30.0 days (1 month)
  Power: 0.45
  Amplitude: $320.00
```

---

### 2.2 Cycle Composite Indicator

Combines multiple cycles into a signal:

```python
# Combine cycles
cycles = [20, 50, 200]  # 20-day, 50-day, 200-day cycles
composite = analyzer.cycle_composite(prices, cycles, lookahead=10)

# Trading Signal
current_value = composite[-1]
previous_value = composite[-2]

if current_value > previous_value and current_value < 0:
    signal = 1  # Long (cycle low)
elif current_value < previous_value and current_value > 0:
    signal = -1  # Short (cycle high)
else:
    signal = 0  # Neutral
```

---

### 2.3 Spectral Filtering

Separate trend from noise:

```python
# Lowpass Filter - Extract trend
trend = analyzer.spectral_filter(prices, filter_type='lowpass', cutoff=0.05)

# Highpass Filter - Extract cycles
cycles = analyzer.spectral_filter(prices, filter_type='highpass', cutoff=0.1)

# Bandpass Filter - Specific cycles
specific_cycles = analyzer.spectral_filter(
    prices, 
    filter_type='bandpass', 
    cutoff=(0.02, 0.1)  # 10-50 day cycles
)

# Practical application
from math_tools import SpectralAnalyzer

analyzer = SpectralAnalyzer()

# Extract noise-free trend
clean_trend = analyzer.extract_trend(prices, smoothness=0.95)

# Extract cyclical component
cyclical = analyzer.extract_cycles(prices, min_period=5, max_period=100)
```

---

### 2.4 Hilbert Transform - Real-Time Cycle Analysis

For real-time trading without look-ahead bias:

```python
from math_tools import HilbertTransformAnalyzer

# Hilbert Transform for Instantaneous Frequency
hilbert = HilbertTransformAnalyzer()
result = hilbert.compute(prices, window=20)

# Instantaneous values
amplitude = result['amplitude'][-1]      # Current amplitude
phase = result['phase'][-1]            # Current phase
frequency = result['frequency'][-1]     # Instantaneous frequency
period = result['current_period']       # Instantaneous period

print(f"Current Cycle: {period:.1f} days")
print(f"Amplitude: ${amplitude:.2f}")
print(f"Phase: {phase:.2f} rad")

# Cycle State for Trading
cycle_state = hilbert.get_cycle_state(recent_window=20)
print(f"\nDominant Period: {cycle_state['dominant_period']:.1f}")
print(f"Cycle Strength: {cycle_state['cycle_strength']:.2f}")
print(f"Amplitude Trend: {cycle_state['amplitude_trend']}")
```

---

### 2.5 Adaptive Cycle Indicator

Automatic adaptation to changing cycles:

```python
from math_tools import AdaptiveCycleIndicator

# Adaptive Cycle Indicator
aci = AdaptiveCycleIndicator(lookback=200, adapt_period=50)

# In Trading Loop
for price in live_prices:
    result = aci.update(price)
    
    signal = result['signal']           # -1, 0, 1
    cycle_period = result['cycle_period']  # Current cycle
    phase = result['phase']              # Position in cycle
    confidence = result['confidence']    # 0-1
    
    if signal == 1 and confidence > 0.6:
        place_buy_order()
    elif signal == -1 and confidence > 0.6:
        place_sell_order()
```

**How it works:**
1. Continuously analyzes the last 200 periods
2. Recalculates dominant cycle every 50 periods
3. Generates signals based on cycle phase
4. Buys at cycle low (phase ~0.9), sells at high (phase ~0.3)

---

### 2.6 Seasonality Analysis

Discover seasonal patterns:

```python
from math_tools import SeasonalityAnalyzer
import pandas as pd

# Data with DatetimeIndex
returns = pd.Series(returns_array, index=pd.to_datetime(dates))

analyzer = SeasonalityAnalyzer()

# Day-of-week analysis
dow_stats = analyzer.analyze_day_of_week(returns, returns.index)
print("Day-of-Week Effects:")
for day, stats in dow_stats.items():
    if day not in ['best_day', 'worst_day']:
        print(f"  {day}: {stats['mean_return']:.3f}% (Win Rate: {stats['win_rate']:.1%})")

print(f"\nBest Day: {dow_stats['best_day']}")
print(f"Worst Day: {dow_stats['worst_day']}")

# Month-of-year analysis
month_stats = analyzer.analyze_month_of_year(returns, returns.index)

# Holiday Effects
holiday_effects = analyzer.detect_holiday_effects(
    prices, 
    prices.index,
    holidays=['01-01', '07-04', '12-25']  # New Year, July 4th, Christmas
)
```

**Typical Results for BTC:**
```
Day-of-Week Effects:
  Monday: 0.25% (Win Rate: 54.2%)
  Tuesday: 0.18% (Win Rate: 52.1%)
  Wednesday: -0.05% (Win Rate: 48.9%)
  Thursday: 0.32% (Win Rate: 55.8%)  ← Best day
  Friday: -0.12% (Win Rate: 47.3%)

Best Day: Thursday
Worst Day: Friday

Holiday Effects:
  01-01: Pre-holiday +0.45%, Post-holiday +0.82%
  12-25: Pre-holiday -0.23%, Post-holiday +1.12%
```

---

## 3. Practical Trading Strategies

### 3.1 Crypto Cycle Strategy

Based on Bitcoin's 4-year halving cycle:

```python
from math_tools import SpectralAnalyzer, compute_dominant_cycle

class CryptoCycleStrategy:
    def __init__(self):
        self.analyzer = SpectralAnalyzer()
        self.cycle_period = None
        self.entry_phase = 0.85  # 85% through cycle (low point)
        self.exit_phase = 0.25  # 25% through cycle (high point)
    
    def update(self, prices):
        # Recalculate cycle every 100 bars
        if len(prices) % 100 == 0:
            self.cycle_period = compute_dominant_cycle(prices[-500:], min_period=100)
        
        if self.cycle_period is None:
            return 0
        
        # Position in cycle
        position = len(prices) % int(self.cycle_period)
        phase = position / self.cycle_period
        
        # Signals
        if phase > self.entry_phase:
            return 1  # Long at cycle low
        elif phase < self.exit_phase:
            return -1  # Short/Exit at cycle high
        
        return 0
```

---

### 3.2 Spectral Mean Reversion

Combines Spectral Analysis with Mean-Reversion:

```python
from math_tools import SpectralAnalyzer, OrnsteinUhlenbeckProcess

def spectral_mean_reversion(prices, lookback=200):
    analyzer = SpectralAnalyzer()
    
    # 1. Extract trend
    trend = analyzer.extract_trend(prices[-lookback:], smoothness=0.95)
    
    # 2. Cycle component
    cycle = analyzer.extract_cycles(prices[-lookback:], min_period=5, max_period=50)
    
    # 3. Apply Mean-Reversion to cycle
    ou = OrnsteinUhlenbeckProcess()
    score = ou.calculate_score(cycle[-20:])
    
    # 4. Signal
    if score > 0.8:  # Far above trend
        return -1  # Short (expect return to trend)
    elif score < -0.8:  # Far below trend
        return 1   # Long
    
    return 0
```

---

### 3.3 Multi-Timeframe Cycle Confirmation

Cycle confirmation across multiple timeframes:

```python
from math_tools import SpectralAnalyzer

def multi_tf_cycle_signal(prices_daily, prices_weekly, prices_monthly):
    analyzer = SpectralAnalyzer()
    
    # Daily cycles
    daily_cycles = analyzer.find_dominant_cycles(prices_daily, n_cycles=2)
    
    # Weekly cycles
    weekly_cycles = analyzer.find_dominant_cycles(prices_weekly, n_cycles=2)
    
    # Monthly cycles  
    monthly_cycles = analyzer.find_dominant_cycles(prices_monthly, n_cycles=1)
    
    # Check consistency
    daily_long_cycle = daily_cycles[0]['period'] if daily_cycles else None
    weekly_long_cycle = weekly_cycles[0]['period'] * 5 if weekly_cycles else None  # Convert to days
    
    # If Daily and Weekly agree (±20%), boost signal
    if daily_long_cycle and weekly_long_cycle:
        diff = abs(daily_long_cycle - weekly_long_cycle) / daily_long_cycle
        if diff < 0.2:
            confidence = 1.5  # Agreement!
        else:
            confidence = 1.0
    else:
        confidence = 0.5
    
    # Base signal
    composite_daily = analyzer.cycle_composite(prices_daily, 
                                               [c['period'] for c in daily_cycles[:2]])
    signal = 1 if composite_daily[-1] > 0 else -1
    
    return signal * confidence
```

---

## 4. Advantages & Limitations

### ✅ Advantages:

1. **Objective Cycle Detection** (not subjective like visual analysis)
2. **Noise Suppression** (filtering in frequency domain more effective)
3. **Seasonality Proof** (statistically significant patterns)
4. **Multi-Cycle Combination** (different timeframes simultaneously)
5. **Adaptive Adjustment** (responds to changing market cycles)

### ⚠️ Limitations:

1. **Needs Lots of Data** (at least 2x the longest period)
2. **Not a Daytrading Strategy** (too much noise on short timeframes)
3. **Cycles Change** (must be regularly recalibrated)
4. **Not Suitable for All Markets** (works best with clear cycles like Crypto)
5. **Spectral Leakage** (window functions necessary)

### 🎯 Best Use Cases:

- **Swing Trading** (1W - 3M holding period)
- **Crypto Trading** (4-year halving cycle)
- **Stock Seasonalities** (Sell in May, January effect)
- **Volatility Cycles** (VIX-Seasonality)
- **Commodity Cycles** (harvest cycles, weather)

---

## 5. Integration with Other Models

### Combination with HMM:

```python
from math_tools import SpectralAnalyzer, HMMRegimeDetector

# Cycle-based regime detection
analyzer = SpectralAnalyzer()
cycles = analyzer.find_dominant_cycles(prices, n_cycles=1)

if cycles and cycles[0]['period'] > 200:
    # Long cycles = Trending Regime
    regime = "trending"
else:
    # Short cycles = Mean-Reversion Regime
    regime = "mean_reverting"
```

### Combination with Kelly:

```python
from math_tools import SpectralAnalyzer, KellyCriterion

# Position Size based on cycle confidence
analyzer = SpectralAnalyzer()
dominant_cycle = analyzer.find_dominant_cycles(returns, n_cycles=1)

if dominant_cycle and dominant_cycle[0]['power'] > 0.7:
    # Strong cycle → Normal position
    kelly = KellyCriterion()
    size = kelly.calculate_position_size(returns)
else:
    # Weak cycle → Reduced position
    size = 0.5
```

---

## 6. Summary

**Spectral Analysis provides:**

✅ **Frequency Domain Analysis** for cycle detection  
✅ **Noise Reduction** via filtering  
✅ **Seasonality Analysis** (day, month, holiday)  
✅ **Adaptive Cycle Indicators**  
✅ **Real-Time Cycle Tracking** (Hilbert Transform)  

**Best suited for:**
- Swing Trading (1W-3M)
- Crypto (halving cycles)
- Seasonality strategies
- Trend-Noise separation

**Not suited for:**
- Daytrading/Scalping
- Markets without clear cycles
- Short time series (< 200 data points)

---

**The module is now ready for professional cycle-based trading strategies!** 📊🔄
