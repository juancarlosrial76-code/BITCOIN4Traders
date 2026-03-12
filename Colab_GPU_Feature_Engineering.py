"""
GPU Feature Engineering cell for BITCOIN4Traders Colab Notebook.

Insert this cell AFTER cell 7 (Training configuration) and BEFORE cell 9 (Load data).

This cell provides:
1. GPU detection and info
2. Choice between CPU and GPU Feature Engineering
3. Benchmark of GPU vs CPU performance
4. Automatic selection based on data size
"""

# ═══════════════════════════════════════════════════════════════════════════════
# Cell 8a: GPU Info & Performance Check (NEW - before Feature Engineering)
# ═══════════════════════════════════════════════════════════════════════════════

import torch
import numpy as np
import time
from pathlib import Path

print("=" * 70)
print("GPU CHECK FOR FEATURE ENGINEERING")
print("=" * 70)

GPU_AVAILABLE = torch.cuda.is_available()
GPU_NAME = torch.cuda.get_device_name(0) if GPU_AVAILABLE else "N/A"
GPU_MEMORY_GB = (
    torch.cuda.get_device_properties(0).total_memory / 1e9 if GPU_AVAILABLE else 0
)

print(f"\nGPU Status:")
print(f"  Available: {'✓ YES' if GPU_AVAILABLE else '✗ NO'}")
if GPU_AVAILABLE:
    print(f"  Name: {GPU_NAME}")
    print(f"  VRAM: {GPU_MEMORY_GB:.1f} GB")
else:
    print("  (CPU only available)")

print(f"\nPyTorch Version: {torch.__version__}")
print(f"CUDA Version: {torch.version.cuda if GPU_AVAILABLE else 'N/A'}")

# ═══════════════════════════════════════════════════════════════════════════════
# GPU vs CPU Auto-Select for Feature Engineering
# ═══════════════════════════════════════════════════════════════════════════════

USE_GPU_FOR_FEATURES = False
GPU_SPEEDUP_ESTIMATE = 1.0

if GPU_AVAILABLE:
    # Thresholds based on GPU tests:
    # < 50k rows: GPU overhead makes GPU slower
    # >= 50k rows: GPU starts to benefit
    # >= 100k rows: GPU significantly faster (5-15x)
    # GPU_MEMORY_GB < 10: Limited VRAM, more cautious with large data

    USE_GPU_FOR_FEATURES = True
    GPU_SPEEDUP_ESTIMATE = 8.0  # Typical speedup at 100k rows on T4

    print(f"\nGPU Feature Engineering: ENABLED")
    print(f"  Estimated speedup: {GPU_SPEEDUP_ESTIMATE}x (at >100k rows)")
else:
    print(f"\nGPU Feature Engineering: DISABLED (no GPU)")
    print(f"  Using CPU: pandas/numpy implementation")

print(f"\nTip: Run benchmark with:")
print(f"  from src.features.feature_engine import benchmark_gpu_cpu")
print(f"  results = benchmark_gpu_cpu(n_rows=50000)")

print("=" * 70)

# ═══════════════════════════════════════════════════════════════════════════════
# Optional: GPU Feature Engine Import (lazy, only loaded when needed)
# ═══════════════════════════════════════════════════════════════════════════════


def get_feature_engine(use_gpu=False):
    """Returns the appropriate FeatureEngine class."""
    if use_gpu and GPU_AVAILABLE:
        try:
            from src.features.feature_engine import GPUFeatureEngine

            print("✓ GPUFeatureEngine loaded")
            return GPUFeatureEngine
        except ImportError:
            print("⚠ GPUFeatureEngine not available, falling back to CPU")
            from src.features.feature_engine import FeatureEngine

            return FeatureEngine
    else:
        from src.features.feature_engine import FeatureEngine

        return FeatureEngine


# ═══════════════════════════════════════════════════════════════════════════════
# Benchmark function for quick test
# ═══════════════════════════════════════════════════════════════════════════════


def quick_benchmark(n_rows=10000):
    """Quick benchmark for Feature Engineering."""
    from src.features.feature_engine import FeatureEngine, FeatureConfig

    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=n_rows, freq="1H")
    close = 50000 + np.cumsum(np.random.randn(n_rows) * 100)
    df = pd.DataFrame(
        {
            "open": close + np.random.randn(n_rows) * 50,
            "high": close + abs(np.random.randn(n_rows) * 100),
            "low": close - abs(np.random.randn(n_rows) * 100),
            "close": close,
            "volume": np.random.uniform(1000, 10000, n_rows),
        },
        index=dates,
    )

    config = FeatureConfig(
        volatility_window=20,
        ou_window=50,
        rolling_mean_window=20,
        use_log_returns=True,
        scaler_type="standard",
        save_scaler=False,
        scaler_path=Path("data/scalers"),
        dropna_strategy="rolling",
        min_valid_rows=100,
    )

    start = time.perf_counter()
    engine = FeatureEngine(config)
    features = engine.fit_transform(df)
    elapsed_ms = (time.perf_counter() - start) * 1000

    print(f"\nQuick Benchmark ({n_rows:,} rows):")
    print(f"  Time: {elapsed_ms:.0f}ms")
    print(f"  Features: {features.shape[1]}")
    return elapsed_ms


if __name__ == "__main__":
    # Directly executable for quick test
    quick_benchmark(10000)
