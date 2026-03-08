#!/usr/bin/env python3
"""
Test-Skript für GPU Feature Engine.

Usage:
    python test_gpu_feature_engine.py --benchmark --rows 50000
    python test_gpu_feature_engine.py --verify --rows 10000
    python test_gpu_feature_engine.py --all
"""

import sys
import time
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.features.feature_engine import (
    FeatureEngine,
    FeatureConfig,
    benchmark_gpu_cpu,
    verify_gpu_correctness,
    is_gpu_available,
    get_gpu_info,
    GPUFeatureEngine,
)


def create_test_data(n_rows: int = 10000, seed: int = 42) -> pd.DataFrame:
    """Erstellt synthetische OHLCV Test-Daten."""
    np.random.seed(seed)

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

    return df


def test_basic_functionality():
    """Testet grundlegende Funktionalität von FeatureEngine."""
    print("\n" + "=" * 60)
    print("TEST: Basic Functionality")
    print("=" * 60)

    df = create_test_data(n_rows=2000)

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

    engine = FeatureEngine(config)

    train_size = int(0.7 * len(df))
    train_df = df.iloc[:train_size]
    test_df = df.iloc[train_size:]

    print(f"Train: {len(train_df)} rows, Test: {len(test_df)} rows")

    train_features = engine.fit_transform(train_df)
    test_features = engine.transform(test_df)

    print(f"Train features: {train_features.shape}")
    print(f"Test features: {test_features.shape}")

    nan_count = train_features.isnull().sum().sum()
    if nan_count == 0:
        print("✓ Keine NaN-Werte")
    else:
        print(f"✗ {nan_count} NaN-Werte gefunden")

    print(f"Feature columns: {list(train_features.columns)}")

    return True


def test_gpu_info():
    """Zeigt GPU-Informationen an."""
    print("\n" + "=" * 60)
    print("TEST: GPU Info")
    print("=" * 60)

    available = is_gpu_available()
    print(f"GPU verfügbar: {'✓ JA' if available else '✗ NEIN'}")

    if available:
        info = get_gpu_info()
        print(f"  Name: {info['name']}")
        print(f"  VRAM: {info['memory_total_gb']:.1f} GB")
        print(f"  Compute Capability: {info['compute_cap']}")
    else:
        print("  Keine NVIDIA GPU gefunden (nur CPU verfügbar)")

    return True


def test_gpu_benchmark(rows: int = 50000, runs: int = 3):
    """Führt GPU vs CPU Benchmark durch."""
    print("\n" + "=" * 60)
    print(f"TEST: GPU Benchmark ({rows:,} Zeilen, {runs} Runs)")
    print("=" * 60)

    results = benchmark_gpu_cpu(n_rows=rows, n_runs=runs)

    print("\nZusammenfassung:")
    print(f"  CPU Zeit: {results['cpu_time_ms']:.1f}±{results['cpu_std_ms']:.1f}ms")

    if "gpu_time_ms" in results:
        print(f"  GPU Zeit: {results['gpu_time_ms']:.1f}±{results['gpu_std_ms']:.1f}ms")
        print(f"  Speedup: {results['speedup']:.1f}x")

        if results["speedup"] > 1.0:
            print(f"  ✓ GPU ist {results['speedup']:.1f}x schneller!")
        else:
            print(f"  ⚠ GPU ist langsamer (Overhead für diese Grösse)")
    else:
        print("  GPU nicht verfügbar")

    return results


def test_gpu_correctness(rows: int = 10000, tolerance: float = 1e-3):
    """Verifiziert GPU vs CPU Korrektheit."""
    print("\n" + "=" * 60)
    print(f"TEST: GPU Correctness ({rows:,} Zeilen, Toleranz: {tolerance})")
    print("=" * 60)

    if not is_gpu_available():
        print("✗ GPU nicht verfügbar - Überspringe Test")
        return {"passed": False, "reason": "GPU not available"}

    results = verify_gpu_correctness(n_rows=rows, tolerance=tolerance)

    if results["passed"]:
        print("✓ Alle Tests bestanden! GPU-Berechnung ist korrekt.")
    else:
        print("✗ Tests fehlgeschlagen!")

    return results


def test_performance_scaling():
    """Testet Performance bei verschiedenen Datenmengen."""
    print("\n" + "=" * 60)
    print("TEST: Performance Scaling")
    print("=" * 60)

    sizes = [1000, 5000, 10000, 25000, 50000]
    results = []

    if not is_gpu_available():
        print("⚠ GPU nicht verfügbar - nur CPU-Messung")

    for size in sizes:
        df = create_test_data(n_rows=size, seed=42)

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
        _ = engine.fit_transform(df)
        cpu_time = (time.perf_counter() - start) * 1000

        results.append(
            {
                "rows": size,
                "cpu_ms": cpu_time,
            }
        )

        print(f"  {size:>6,} Zeilen: {cpu_time:>8.1f}ms")

    return results


def run_all_tests(args):
    """Führt alle Tests aus."""
    print("\n" + "#" * 60)
    print("# GPU FEATURE ENGINE TESTS")
    print("#" * 60)

    all_passed = True

    test_gpu_info()
    test_basic_functionality()

    if args.benchmark or args.all:
        rows = getattr(args, "rows", 50000)
        test_gpu_benchmark(rows=rows)

    if args.verify or args.all:
        rows = getattr(args, "rows", 10000)
        result = test_gpu_correctness(rows=rows)
        if not result.get("passed", True):
            all_passed = False

    test_performance_scaling()

    print("\n" + "=" * 60)
    if all_passed:
        print("✓ ALLE TESTS ABGESCHLOSSEN")
    else:
        print("⚠ EINIGE TESTS MIT WARNUNGEN")
    print("=" * 60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="GPU Feature Engine Tests")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark")
    parser.add_argument(
        "--verify", action="store_true", help="Run correctness verification"
    )
    parser.add_argument("--all", action="store_true", help="Run all tests")
    parser.add_argument(
        "--rows", type=int, default=50000, help="Number of rows for tests"
    )
    parser.add_argument(
        "--quick", action="store_true", help="Quick test with 1000 rows"
    )

    args = parser.parse_args()

    if args.quick:
        args.rows = 1000
        args.all = True

    if not any([args.benchmark, args.verify, args.all]):
        args.all = True

    run_all_tests(args)
