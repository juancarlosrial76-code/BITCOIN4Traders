"""
Stress-Test Data Generator für Trading-RL.

Generiert synthetische Markt-Szenarien für Robustness-Training:
- Black Swan Events
- Volatilitätsspikes
- Flash Crashes
- Long Bear Markets
- Extreme Whipsaws
- Liquidity Crises

Diese Szenarien trainieren den Agent, auch in Stress-Perioden
korrekte Entscheidungen zu treffen.

Usage:
    from data_generation.stress_scenarios import generate_stress_dataset

    # Szenario: Flash Crash
    df = generate_stress_dataset(scenario="flash_crash", length=5000)

    # Szenario: Volatility Spike
    df = generate_stress_dataset(scenario="volatility_spike", length=5000)

    # Alle Szenarien kombiniert
    df = generate_stress_dataset(scenario="all", length=10000)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from loguru import logger


@dataclass
class StressConfig:
    """Konfiguration für Stress-Szenarien."""

    base_volatility: float = 0.02
    crash_probability: float = 0.001
    spike_probability: float = 0.005
    min_length: int = 1000
    seed: int = 42


def set_seed(seed: int):
    """Setzt den Random Seed für reproduzierbare Ergebnisse."""
    np.random.seed(seed)


def generate_brownian_motion(
    n: int, mu: float = 0.0, sigma: float = 0.02, seed: int = 42
) -> np.ndarray:
    """
    Generates Brownian motion (Geometric Brownian Motion returns).

    Args:
        n: Number of points
        mu: Drift (daily return)
        sigma: Volatility (daily)
        seed: Random seed

    Returns:
        Array of log returns
    """
    np.random.seed(seed)
    return np.random.normal(mu, sigma, n)


def generate_flash_crash(
    length: int = 5000,
    crash_depth: float = 0.30,
    crash_duration: int = 50,
    recovery_rate: float = 0.001,  # Very small daily recovery
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generiert ein Flash-Crash Szenario.

    Szenario: Plötzlicher starker Drop mit schneller Erholung.
    Ziel: Agent lernt, nicht in Panik zu verkaufen.

    Args:
        length: Gesamtlänge des Datensatzes
        crash_depth: Maximaler Drop (30% = 0.30)
        crash_duration: Wie schnell der Crash passiert (Bars)
        recovery_rate: Erholungsgeschwindigkeit
        seed: Random seed

    Returns:
        DataFrame mit OHLCV Daten
    """
    set_seed(seed)

    base_returns = generate_brownian_motion(length, mu=0.0, sigma=0.01)

    crash_start = int(length * 0.4)
    crash_end = crash_start + crash_duration

    for i in range(crash_start, crash_end):
        progress = (i - crash_start) / crash_duration
        crash_impact = -crash_depth * np.sin(progress * np.pi / 2)
        base_returns[i] = crash_impact / crash_duration

    for i in range(crash_end, min(crash_end + 200, length)):
        base_returns[i] = recovery_rate  # Very small recovery per bar

    close = 50000 * np.exp(np.cumsum(base_returns))

    high = close * (1 + np.abs(np.random.normal(0.005, 0.01, length)))
    low = close * (1 - np.abs(np.random.normal(0.005, 0.01, length)))
    low = np.minimum(low, close)
    high = np.maximum(high, close)

    volume = np.random.uniform(1000, 5000, length)
    volume[crash_start:crash_end] *= 5  # Volume Spike während Crash

    open_prices = close.copy()
    open_prices[1:] = close[:-1]
    open_prices[0] = close[0]

    df = pd.DataFrame(
        {
            "open": open_prices,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )
    df.index = pd.date_range("2024-01-01", periods=length, freq="1H")

    return df


def generate_volatility_spike(
    length: int = 5000,
    volatility_multiplier: float = 5.0,
    spike_duration: int = 100,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generiert ein High-Volatility Szenario.

    Szenario: Mehrmalige hohe Volatilität ohne klare Richtung.
    Ziel: Agent lernt, in unruhigen Märkten Positionen zu reduzieren.

    Args:
        length: Gesamtlänge des Datensatzes
        volatility_multiplier: Wie viel höher die Volatilität ist
        spike_duration: Länge des High-Volatility Bereichs
        seed: Random seed

    Returns:
        DataFrame mit OHLCV Daten
    """
    set_seed(seed)

    base_vol = 0.01
    spike_vol = base_vol * volatility_multiplier

    returns = np.zeros(length)
    phases = [
        (0, int(length * 0.3), base_vol, 0.0001),
        (int(length * 0.3), int(length * 0.3) + spike_duration, spike_vol, 0.0),
        (int(length * 0.3) + spike_duration, int(length * 0.7), base_vol, 0.0001),
        (int(length * 0.7), int(length * 0.7) + 150, spike_vol, 0.0),
        (int(length * 0.7) + 150, length, base_vol, 0.0001),
    ]

    for start, end, vol, mu in phases:
        returns[start:end] = np.random.normal(mu, vol, end - start)

    close = 50000 * np.exp(np.cumsum(returns))

    high = close * (1 + np.abs(np.random.normal(0.01, 0.02, length)))
    low = close * (1 - np.abs(np.random.normal(0.01, 0.02, length)))
    low = np.minimum(low, close)
    high = np.maximum(high, close)

    volume = np.random.uniform(1000, 5000, length)
    volume[phases[1][0] : phases[1][1]] *= 3
    volume[phases[3][0] : phases[3][1]] *= 3

    open_prices = close.copy()
    open_prices[1:] = close[:-1]
    open_prices[0] = close[0]

    df = pd.DataFrame(
        {
            "open": open_prices,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )
    df.index = pd.date_range("2024-01-01", periods=length, freq="1H")

    return df


def generate_bear_market(
    length: int = 5000,
    total_drop: float = 0.50,
    drawdown_peaks: int = 3,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generiert einen langen Bärenmarkt.

    Szenario: Stetiger Abwärtstrend mit lokalen Erholungen.
    Ziel: Agent lernt, Short-Positionen zu halten und nicht zu früh zu kaufen.

    Args:
        length: Gesamtlänge des Datensatzes
        total_drop: Gesamter Drop über den Zeitraum (50% = 0.50)
        drawdown_peaks: Anzahl der lokalen Erholungen
        seed: Random seed

    Returns:
        DataFrame mit OHLCV Daten
    """
    set_seed(seed)

    daily_drop = total_drop / length

    returns = np.full(length, -daily_drop)

    recovery_points = np.linspace(int(length * 0.2), int(length * 0.9), drawdown_peaks)
    for rp in recovery_points:
        rp = int(rp)
        recovery_length = 30
        if rp + recovery_length < length:
            for i in range(recovery_length):
                if rp + i < length:
                    returns[rp + i] = 0.01 + np.random.normal(0, 0.01)

    close = 50000 * np.exp(np.cumsum(returns))

    high = close * (1 + np.abs(np.random.normal(0.015, 0.02, length)))
    low = close * (1 - np.abs(np.random.normal(0.015, 0.02, length)))
    low = np.minimum(low, close)
    high = np.maximum(high, close)

    volume = np.random.uniform(1000, 5000, length)
    volume = volume * 1.5  # Mehr Volume in Bärenmärkten

    open_prices = close.copy()
    open_prices[1:] = close[:-1]
    open_prices[0] = close[0]

    df = pd.DataFrame(
        {
            "open": open_prices,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )
    df.index = pd.date_range("2024-01-01", periods=length, freq="1H")

    return df


def generate_whipsaw_market(
    length: int = 5000, amplitude: float = 0.10, frequency: float = 0.1, seed: int = 42
) -> pd.DataFrame:
    """
    Generiert ein extremes Whipsaw-Szenario.

    Szenario: Ständige Richtungswechsel ohne klare Trends.
    Ziel: Agent lernt, nicht überzu handeln (Overtrading zu vermeiden).

    Args:
        length: Gesamtlänge des Datensatzes
        amplitude: Stärke der Swings (10% = 0.10)
        frequency: Wie häufig die Richtungswechsel sind
        seed: Random seed

    Returns:
        DataFrame mit OHLCV Daten
    """
    set_seed(seed)

    t = np.arange(length)
    trend = amplitude * np.sin(2 * np.pi * frequency * t / length)
    noise = np.random.normal(0, 0.005, length)

    returns = np.gradient(trend) + noise

    close = 50000 * np.exp(np.cumsum(returns))

    high = close * (1 + np.abs(np.random.normal(0.01, 0.015, length)))
    low = close * (1 - np.abs(np.random.normal(0.01, 0.015, length)))
    low = np.minimum(low, close)
    high = np.maximum(high, close)

    volume = np.random.uniform(1000, 5000, length)
    volume = volume * 2  # Hohes Volume bei Whipsaws

    open_prices = close.copy()
    open_prices[1:] = close[:-1]
    open_prices[0] = close[0]

    df = pd.DataFrame(
        {
            "open": open_prices,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )
    df.index = pd.date_range("2024-01-01", periods=length, freq="1H")

    return df


def generate_liquidity_crisis(
    length: int = 5000,
    spread_widen: float = 0.05,
    volume_drop: float = 0.7,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generiert ein Liquiditäts-Krise Szenario.

    Szenario: Spreads weiten sich aus, Volume bricht ein.
    Ziel: Agent lernt, in illiquiden Märkten vorsichtiger zu handeln.

    Args:
        length: Gesamtlänge des Datensatzes
        spread_widen: Wie stark sich Spreads weiten
        volume_drop: Um wie viel Volume sinkt
        seed: Random seed

    Returns:
        DataFrame mit OHLCV Daten
    """
    set_seed(seed)

    base_returns = generate_brownian_motion(length, mu=0.0, sigma=0.015)

    crisis_start = int(length * 0.3)
    crisis_end = int(length * 0.7)

    for i in range(crisis_start, crisis_end):
        if i < length:
            base_returns[i] *= 0.5  # Weniger Bewegung in Krise

    close = 50000 * np.exp(np.cumsum(base_returns))

    spread = 0.002 + np.where(
        np.arange(length) >= crisis_start,
        np.where(np.arange(length) < crisis_end, spread_widen, 0.002),
        0.002,
    )

    high = close * (1 + spread * np.random.uniform(0.5, 1.5, length))
    low = close * (1 - spread * np.random.uniform(0.5, 1.5, length))

    volume_base = np.random.uniform(1000, 5000, length)
    volume = np.where(
        np.arange(length) >= crisis_start,
        np.where(
            np.arange(length) < crisis_end, volume_base * volume_drop, volume_base
        ),
        volume_base,
    )

    open_prices = close.copy()
    open_prices[1:] = close[:-1]
    open_prices[0] = close[0]

    df = pd.DataFrame(
        {
            "open": open_prices,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )
    df.index = pd.date_range("2024-01-01", periods=length, freq="1H")

    return df


def generate_black_swan(
    length: int = 5000,
    swan_probability: float = 0.001,
    swan_impact: float = 0.20,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generiert ein Black-Swan Event Szenario.

    Szenario: Seltene, extreme Moves (20%+ in kurzer Zeit).
    Ziel: Agent lernt, mit extremen Adverse Moves umzugehen.

    Args:
        length: Gesamtlänge des Datensatzes
        swan_probability: Wahrscheinlichkeit eines Black Swan pro Bar
        swan_impact: Maximaler Impact eines Black Swan (20% = 0.20)
        seed: Random seed

    Returns:
        DataFrame mit OHLCV Daten
    """
    set_seed(seed)

    base_returns = generate_brownian_motion(length, mu=0.0001, sigma=0.01)

    for i in range(length):
        if np.random.random() < swan_probability:
            direction = np.random.choice([-1, 1])
            impact = swan_impact * np.random.uniform(0.5, 1.0)
            base_returns[i] = direction * impact

            if i + 1 < length:
                base_returns[i + 1] = direction * impact * 0.5

    close = 50000 * np.exp(np.cumsum(base_returns))

    high = close * (1 + np.abs(np.random.normal(0.01, 0.02, length)))
    low = close * (1 - np.abs(np.random.normal(0.01, 0.02, length)))
    low = np.minimum(low, close)
    high = np.maximum(high, close)

    volume = np.random.uniform(1000, 5000, length)
    volume = volume * 3  # Extreme Volume bei Black Swans

    open_prices = close.copy()
    open_prices[1:] = close[:-1]
    open_prices[0] = close[0]

    df = pd.DataFrame(
        {
            "open": open_prices,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )
    df.index = pd.date_range("2024-01-01", periods=length, freq="1H")

    return df


def generate_mixed_stress_dataset(
    scenarios: Optional[List[str]] = None,
    length_per_scenario: int = 2000,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generiert einen gemischten Stress-Datensatz.

    Kombiniert mehrere Szenarien für umfassendes Stress-Training.

    Args:
        scenarios: Liste der Szenarien ('flash_crash', 'volatility_spike', etc.)
                   Wenn None, werden alle verwendet.
        length_per_scenario: Länge pro Szenario
        seed: Random seed

    Returns:
        DataFrame mit allen Szenarien kombiniert
    """
    if scenarios is None:
        scenarios = [
            "flash_crash",
            "volatility_spike",
            "bear_market",
            "whipsaw",
            "liquidity_crisis",
            "black_swan",
        ]

    all_data = []

    scenario_funcs = {
        "flash_crash": generate_flash_crash,
        "volatility_spike": generate_volatility_spike,
        "bear_market": generate_bear_market,
        "whipsaw": generate_whipsaw_market,
        "liquidity_crisis": generate_liquidity_crisis,
        "black_swan": generate_black_swan,
    }

    for i, scenario in enumerate(scenarios):
        if scenario in scenario_funcs:
            df = scenario_funcs[scenario](length=length_per_scenario, seed=seed + i)
            df["scenario"] = scenario
            all_data.append(df)
            logger.info(f"Generiert: {scenario} ({len(df)} bars)")

    combined = pd.concat(all_data, ignore_index=True)

    return combined


def generate_stress_dataset(
    scenario: str = "all", length: int = 10000, **kwargs
) -> pd.DataFrame:
    """
    Hauptfunktion zur Generierung von Stress-Szenarien.

    Args:
        scenario: Name des Szenarios oder 'all' für alle
                  Optionen: flash_crash, volatility_spike, bear_market,
                           whipsaw, liquidity_crisis, black_swan, all
        length: Gesamtlänge des Datensatzes
        **kwargs: Zusätzliche Parameter für spezifische Szenarien

    Returns:
        DataFrame mit OHLCV Daten und 'scenario' Spalte
    """
    scenarios_map = {
        "flash_crash": generate_flash_crash,
        "volatility_spike": generate_volatility_spike,
        "bear_market": generate_bear_market,
        "whipsaw": generate_whipsaw_market,
        "liquidity_crisis": generate_liquidity_crisis,
        "black_swan": generate_black_swan,
    }

    if scenario == "all":
        n_scenarios = len(scenarios_map)
        length_per = max(1000, length // n_scenarios)
        df = generate_mixed_stress_dataset(
            scenarios=list(scenarios_map.keys()),
            length_per_scenario=length_per,
            seed=kwargs.get("seed", 42),
        )
    elif scenario in scenarios_map:
        df = scenarios_map[scenario](length=length, **kwargs)
    else:
        raise ValueError(f"Unbekanntes Szenario: {scenario}")

    logger.success(f"Stress-Datensatz generiert: {len(df)} bars")

    return df


class StressDataGenerator:
    """
    Klasse zur Generierung von Stress-Trainingsdaten.

    Bietet einfachen Zugang zu allen Stress-Szenarien.

    Usage:
        generator = StressDataGenerator()

        # Einzelnes Szenario
        df = generator.generate('flash_crash', length=5000)

        # Alle Szenarien
        df = generator.generate_all(length=15000)

        # Training direkt
        features = engineer.fit_transform(df)
    """

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.config = StressConfig(seed=seed)

    def generate(self, scenario: str, length: int = 5000, **kwargs) -> pd.DataFrame:
        """Generiert ein spezifisches Szenario."""
        return generate_stress_dataset(
            scenario=scenario,
            length=length,
            seed=kwargs.get("seed", self.seed),
            **kwargs,
        )

    def generate_all(self, length: int = 15000) -> pd.DataFrame:
        """Generiert alle Szenarien kombiniert."""
        return generate_stress_dataset(scenario="all", length=length, seed=self.seed)

    def generate_curriculum(
        self, base_length: int = 3000, curriculum: Optional[List[str]] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Generiert ein Curriculum von einfachen zu schweren Szenarien.

        Args:
            base_length: Basis-Länge pro Szenario
            curriculum: Reihenfolge der Szenarien

        Returns:
            Dictionary mit Szenario-Namen als Keys
        """
        if curriculum is None:
            curriculum = [
                "volatility_spike",  # Einfach: nur hohe Volatilität
                "whipsaw",  # Mittelmässig: Richtungswechsel
                "bear_market",  # Schwer: Abwärtstrend
                "flash_crash",  # Schwerer: Plötzliche Drops
                "liquidity_crisis",  # Sehr schwer: Illiquidität
                "black_swan",  # Am schwersten: Extrem-Events
            ]

        result = {}
        for scenario in curriculum:
            df = self.generate(scenario, length=base_length)
            result[scenario] = df
            logger.info(f"Curriculum: {scenario} ({len(df)} bars)")

        return result


def visualize_stress_scenario(df: pd.DataFrame, title: str = "Stress Scenario"):
    """
    Visualisiert ein Stress-Szenario (für Debugging/Entwicklung).
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(14, 10))

    axes[0].plot(df["close"], label="Close", linewidth=0.8)
    axes[0].set_ylabel("Price")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title(f"{title} - Price")

    returns = df["close"].pct_change().fillna(0)
    axes[1].plot(returns, label="Returns", linewidth=0.5, alpha=0.7)
    axes[1].axhline(y=returns.std() * 3, color="r", linestyle="--", label="3σ")
    axes[1].axhline(y=-returns.std() * 3, color="r", linestyle="--")
    axes[1].set_ylabel("Returns")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_title(f"{title} - Returns (Vol={returns.std():.4f})")

    axes[2].plot(df["volume"], label="Volume", linewidth=0.5, alpha=0.7)
    axes[2].set_ylabel("Volume")
    axes[2].set_xlabel("Time")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    axes[2].set_title(f"{title} - Volume")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    print("=" * 70)
    print("STRESS SCENARIO GENERATOR - TEST")
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

    for name, scenario in scenarios:
        print(f"\n{name}:")
        df = generator.generate(scenario, length=2000)
        print(f"  Bars: {len(df)}")
        print(f"  Start: {df['close'].iloc[0]:.2f}")
        print(f"  End: {df['close'].iloc[-1]:.2f}")
        print(f"  Max Drawdown: {((df['close'] / df['close'].cummax()) - 1).min():.2%}")
        print(f"  Volatility: {df['close'].pct_change().std():.4f}")

    print("\n" + "=" * 70)
    print("MIXED STRESS DATASET:")
    df = generator.generate_all(length=12000)
    print(f"  Total Bars: {len(df)}")
    print(f"  Scenarios: {df['scenario'].unique().tolist()}")
    print("=" * 70)
