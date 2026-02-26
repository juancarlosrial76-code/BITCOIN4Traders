"""
BITCOIN4Traders - Quick Start Example
======================================
This example demonstrates how to use the core mathematical models
for quantitative trading.
"""

import sys
from pathlib import Path

# Add project root to path so src package is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from src.math_tools import (
    HurstExponent,
    SpectralAnalyzer,
    quick_hurst_check,
)

# Generate synthetic price data
np.random.seed(42)
n_samples = 1000
dates = pd.date_range("2023-01-01", periods=n_samples, freq="H")

# Simulate mean-reverting price series using cumulative returns
returns = np.random.randn(n_samples) * 0.02
prices = 100 * np.exp(np.cumsum(returns))  # Log-normal price path

print("=" * 60)
print("BITCOIN4Traders - Mathematical Models Demo")
print("=" * 60)

# 1. Hurst Exponent Analysis
print("\n1. Hurst Exponent Analysis")
print("-" * 40)
hurst = HurstExponent(max_lag=100)
hurst_value = hurst.calculate(prices, method="dfa")  # Detrended Fluctuation Analysis
print(f"   Hurst Exponent: {hurst_value:.3f}")
if hurst_value < 0.45:
    print("   → Mean-reverting series (H < 0.5)")
elif hurst_value > 0.55:
    print("   → Trending series (H > 0.5)")
else:
    print("   → Random walk (H ≈ 0.5)")

# Quick check function
print("\n2. Quick Hurst Check")
print("-" * 40)
result = quick_hurst_check(pd.Series(prices))
print(f"   Hurst: {result['hurst_exponent']:.3f}")
print(f"   Regime: {result['regime']}")
print(
    f"   Strategy: {result['strategy']}"
)  # Recommended trading strategy for this regime

# 3. Spectral Analysis
print("\n3. Spectral Analysis - Dominant Cycles")
print("-" * 40)
spectral = SpectralAnalyzer()
freqs, power = spectral.compute_fft(prices)  # Compute power spectrum via FFT
dominant = spectral.find_dominant_cycles(prices, n_cycles=3)  # Find top 3 cycles

print(f"   Found {len(dominant)} dominant cycles:")
for i, cycle in enumerate(dominant[:3], 1):
    print(f"   Cycle {i}: Period={cycle['period']:.1f}, Power={cycle['power']:.3f}")

# Extract trend
print("\n4. Trend Extraction (Low-pass Filter)")
print("-" * 40)
trend = spectral.extract_trend(prices, smoothness=0.95)  # 0.95 = strong smoothing
print(f"   Original Std: {np.std(prices):.3f}")
print(f"   Trend Std: {np.std(trend):.3f}")  # Should be lower (noise removed)

print("\n" + "=" * 60)
print("Demo completed successfully!")
print("See docs/ for more detailed examples and guides.")
print("=" * 60)
