#!/usr/bin/env python3
"""
Test script for stress scenarios.

Usage:
    python test_stress_scenarios.py
    python test_stress_scenarios.py --visualize flash_crash
    python test_stress_scenarios.py --benchmark
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from src.data_generation.stress_scenarios import (
    generate_stress_dataset,
    StressDataGenerator,
    visualize_stress_scenario,
)


def test_all_scenarios():
    """Tests all stress scenarios."""
    print("\n" + "=" * 70)
    print("STRESS SCENARIOS TEST")
    print("=" * 70)

    generator = StressDataGenerator(seed=42)

    scenarios = [
        ("Flash Crash", "flash_crash"),
        ("Volatility Spike", "volatility_spike"),
        ("Bear Market", "bear_market"),
        ("Whipsaw", "whipsaw"),
        ("Liquidity Crisis", "liquidity_crisis"),
        ("Black Swan", "black_swan"),
    ]

    results = []

    for name, scenario in scenarios:
        df = generator.generate(scenario, length=2000)

        close = df["close"]
        max_dd = ((close / close.cummax()) - 1).min()
        volatility = close.pct_change().std()
        total_return = (close.iloc[-1] / close.iloc[0]) - 1

        results.append(
            {
                "name": name,
                "scenario": scenario,
                "bars": len(df),
                "max_dd": max_dd,
                "volatility": volatility,
                "total_return": total_return,
            }
        )

        print(f"\n{name}:")
        print(f"  Bars:        {len(df):,}")
        print(f"  Start:       {close.iloc[0]:.2f}")
        print(f"  End:         {close.iloc[-1]:.2f}")
        print(f"  Max Drawdown: {max_dd:7.2%}")
        print(f"  Volatility:  {volatility:.4f}")
        print(f"  Total Return: {total_return:7.2%}")

        assert len(df) == 2000, f"Expected 2000 bars, got {len(df)}"
        assert close.notna().all(), "NaN values found in close price"
        assert (close > 0).all(), "Non-positive prices found"

    print("\n" + "-" * 70)
    print("Summary:")
    print("-" * 70)
    for r in results:
        print(f"  {r['name']:18s}: DD={r['max_dd']:7.2%}, Vol={r['volatility']:.4f}")

    return results


def test_mixed_dataset():
    """Tests the mixed stress dataset."""
    print("\n" + "=" * 70)
    print("MIXED STRESS DATASET TEST")
    print("=" * 70)

    df = generate_stress_dataset("all", length=6000)

    print(f"\nTotal Bars: {len(df)}")
    print(f"Scenarios: {df['scenario'].nunique()}")
    print(f"\nScenario Distribution:")
    print(df["scenario"].value_counts())

    assert df["scenario"].nunique() == 6, "Expected 6 scenarios"
    assert len(df) == 6000, f"Expected 6000 bars, got {len(df)}"

    return df


def test_curriculum():
    """Tests the curriculum system."""
    print("\n" + "=" * 70)
    print("CURRICULUM TEST")
    print("=" * 70)

    generator = StressDataGenerator(seed=42)
    curriculum = generator.generate_curriculum(base_length=500)

    print(f"\nCurriculum Lengths:")
    for name, df in curriculum.items():
        print(f"  {name}: {len(df)} bars")
        max_dd = ((df["close"] / df["close"].cummax()) - 1).min()
        print(f"    Max DD: {max_dd:7.2%}")

    return curriculum


def benchmark_scenarios():
    """Benchmark for all scenarios."""
    print("\n" + "=" * 70)
    print("BENCHMARK")
    print("=" * 70)

    sizes = [1000, 5000, 10000]
    scenarios = ["flash_crash", "volatility_spike", "bear_market"]

    for size in sizes:
        print(f"\nSize: {size:,} bars")
        for scenario in scenarios:
            import time

            start = time.perf_counter()
            df = generate_stress_dataset(scenario, length=size)
            elapsed = (time.perf_counter() - start) * 1000
            print(f"  {scenario:18s}: {elapsed:6.0f}ms")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Stress Scenarios Test")
    parser.add_argument("--visualize", type=str, help="Visualize specific scenario")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark")
    parser.add_argument("--curriculum", action="store_true", help="Test curriculum")

    args = parser.parse_args()

    print("\n" + "#" * 70)
    print("# STRESS SCENARIOS TEST SUITE")
    print("#" * 70)

    if args.visualize:
        print(f"\nVisualizing: {args.visualize}")
        df = generate_stress_dataset(args.visualize, length=2000)
        visualize_stress_scenario(df, args.visualize)
    elif args.benchmark:
        benchmark_scenarios()
    elif args.curriculum:
        test_curriculum()
    else:
        test_all_scenarios()
        test_mixed_dataset()

    print("\n" + "=" * 70)
    print("✓ ALL TESTS PASSED")
    print("=" * 70)
