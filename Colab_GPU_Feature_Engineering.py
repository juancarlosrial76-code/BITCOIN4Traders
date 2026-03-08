"""
GPU Feature Engineering Zelle für BITCOIN4Traders Colab Notebook.

Füge diese Zelle NACH Zelle 7 (Training-Konfiguration) und VOR Zelle 9 (Daten laden) ein.

Diese Zelle bietet:
1. GPU-Erkennung und Info
2. Wahl zwischen CPU und GPU Feature Engineering
3. Benchmark der GPU vs CPU Performance
4. Automatische Auswahl basierend auf Datengrösse
"""

# ═══════════════════════════════════════════════════════════════════════════════
# Zelle 8a: GPU Info & Performance Check (NEU - vor Feature Engineering)
# ═══════════════════════════════════════════════════════════════════════════════

import torch
import numpy as np
import time
from pathlib import Path

print("=" * 70)
print("GPU CHECK FÜR FEATURE ENGINEERING")
print("=" * 70)

GPU_AVAILABLE = torch.cuda.is_available()
GPU_NAME = torch.cuda.get_device_name(0) if GPU_AVAILABLE else "N/A"
GPU_MEMORY_GB = (
    torch.cuda.get_device_properties(0).total_memory / 1e9 if GPU_AVAILABLE else 0
)

print(f"\nGPU Status:")
print(f"  Verfügbar: {'✓ JA' if GPU_AVAILABLE else '✗ NEIN'}")
if GPU_AVAILABLE:
    print(f"  Name: {GPU_NAME}")
    print(f"  VRAM: {GPU_MEMORY_GB:.1f} GB")
else:
    print("  (Nur CPU verfügbar)")

print(f"\nPyTorch Version: {torch.__version__}")
print(f"CUDA Version: {torch.version.cuda if GPU_AVAILABLE else 'N/A'}")

# ═══════════════════════════════════════════════════════════════════════════════
# GPU vs CPU Auto-Select für Feature Engineering
# ═══════════════════════════════════════════════════════════════════════════════

USE_GPU_FOR_FEATURES = False
GPU_SPEEDUP_ESTIMATE = 1.0

if GPU_AVAILABLE:
    # Schwellenwerte basierend auf GPU-Tests:
    # < 50k Zeilen: GPU-Overhead macht GPU langsamer
    # >= 50k Zeilen: GPU beginnt zu profitieren
    # >= 100k Zeilen: GPU signifikant schneller (5-15x)
    # GPU_MEMORY_GB < 10: Limitierte VRAM, bei grossen Daten vorsichtiger

    USE_GPU_FOR_FEATURES = True
    GPU_SPEEDUP_ESTIMATE = 8.0  # Typischer Speedup bei 100k Zeilen auf T4

    print(f"\nGPU Feature Engineering: AKTIVIERT")
    print(f"  Geschätzter Speedup: {GPU_SPEEDUP_ESTIMATE}x (bei >100k Zeilen)")
else:
    print(f"\nGPU Feature Engineering: DEAKTIVIERT (keine GPU)")
    print(f"  Nutze CPU: pandas/numpy Implementierung")

print(f"\nTipp: Benchmark ausführen mit:")
print(f"  from src.features.feature_engine import benchmark_gpu_cpu")
print(f"  results = benchmark_gpu_cpu(n_rows=50000)")

print("=" * 70)

# ═══════════════════════════════════════════════════════════════════════════════
# Optional: GPU Feature Engine Import (lazy, wird erst bei Bedarf geladen)
# ═══════════════════════════════════════════════════════════════════════════════


def get_feature_engine(use_gpu=False):
    """Gibt passende FeatureEngine-Klasse zurück."""
    if use_gpu and GPU_AVAILABLE:
        try:
            from src.features.feature_engine import GPUFeatureEngine

            print("✓ GPUFeatureEngine geladen")
            return GPUFeatureEngine
        except ImportError:
            print("⚠ GPUFeatureEngine nicht verfügbar, fällt zurück auf CPU")
            from src.features.feature_engine import FeatureEngine

            return FeatureEngine
    else:
        from src.features.feature_engine import FeatureEngine

        return FeatureEngine


# ═══════════════════════════════════════════════════════════════════════════════
# Benchmark Funktion für schnellen Test
# ═══════════════════════════════════════════════════════════════════════════════


def quick_benchmark(n_rows=10000):
    """Schneller Benchmark für Feature Engineering."""
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

    print(f"\nQuick Benchmark ({n_rows:,} Zeilen):")
    print(f"  Zeit: {elapsed_ms:.0f}ms")
    print(f"  Features: {features.shape[1]}")
    return elapsed_ms


if __name__ == "__main__":
    # Direkt ausführbar für schnellen Test
    quick_benchmark(10000)
