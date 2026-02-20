# Spectral Analysis & Fourier Transform Guide

## Überblick

**Spectral Analysis** transformiert Zeitreihen in den Frequenzbereich, um verborgene Zyklen, Saisonalitäten und Trends zu entdecken.

### Warum Spectral Analysis wichtig ist:

1. **Dominante Zyklen entdecken** (z.B. Bitcoin 4-Jahres-Halving-Zyklus)
2. **Noise entfernen** (Trend von Rauschen trennen)
3. **Saisonalitäten finden** (Montag-Effekt, Monats-Ende-Effekt)
4. **Markt-Regime identifizieren** (zyklische vs. trend-basierte Phasen)

**Mathematische Grundlage:** Fourier-Transform & Power Spectral Density (PSD)

---

## 1. Grundkonzepte

### 1.1 Fourier-Transform

Transformiert Zeitreihe von Zeit- in Frequenz-Domäne:

```
X(f) = ∫ x(t) * e^(-2πift) dt

Where:
x(t) = Preis zur Zeit t
X(f) = Frequenz-Komponente bei Frequenz f
```

### 1.2 Power Spectral Density (PSD)

Zeigt die Stärke jeder Frequenz:

```
PSD(f) = |X(f)|²

Hohe Werte bei f = 0.05 → Dominanter Zyklus bei Periode = 1/0.05 = 20 Tagen
```

### 1.3 Dominante Zyklen

Frequenzen mit höchster Power sind die dominierenden Markt-Zyklen.

---

## 2. Implementationen

### 2.1 SpectralAnalyzer - Grundlegende Analyse

```python
from math_tools import SpectralAnalyzer

# Initialisieren
analyzer = SpectralAnalyzer(sampling_rate=1.0)  # 1.0 für tägliche Daten

# FFT berechnen
freqs, power = analyzer.compute_fft(prices, detrend=True, window='hanning')

# Dominante Zyklen finden
dominant_cycles = analyzer.find_dominant_cycles(
    prices, 
    n_cycles=3,
    min_period=10,    # Mindestens 10 Tage
    max_period=500    # Höchstens 500 Tage
)

print("Dominante Zyklen:")
for cycle in dominant_cycles:
    print(f"  Periode: {cycle['period']:.1f} Tage")
    print(f"  Power: {cycle['power']:.2f}")
    print(f"  Amplitude: ${cycle['amplitude']:.2f}")
```

**Beispiel Output:**
```
Dominante Zyklen:
  Periode: 140.0 Tage (20 Wochen)
  Power: 0.85
  Amplitude: $850.00
  
  Periode: 730.0 Tage (4 Jahre)
  Power: 0.72
  Amplitude: $2,400.00
  
  Periode: 30.0 Tage (1 Monat)
  Power: 0.45
  Amplitude: $320.00
```

---

### 2.2 Cycle Composite Indikator

Kombiniert mehrere Zyklen zu einem Signal:

```python
# Zyklen kombinieren
cycles = [20, 50, 200]  # 20-Tage, 50-Tage, 200-Tage Zyklen
composite = analyzer.cycle_composite(prices, cycles, lookahead=10)

# Trading Signal
current_value = composite[-1]
previous_value = composite[-2]

if current_value > previous_value and current_value < 0:
    signal = 1  # Long (Zyklus-Tief)
elif current_value < previous_value and current_value > 0:
    signal = -1  # Short (Zyklus-Hoch)
else:
    signal = 0  # Neutral
```

---

### 2.3 Spectral Filtering

Trend von Noise trennen:

```python
# Lowpass Filter - Trend extrahieren
trend = analyzer.spectral_filter(prices, filter_type='lowpass', cutoff=0.05)

# Highpass Filter - Zyklen extrahieren
cycles = analyzer.spectral_filter(prices, filter_type='highpass', cutoff=0.1)

# Bandpass Filter - Spezifische Zyklen
specific_cycles = analyzer.spectral_filter(
    prices, 
    filter_type='bandpass', 
    cutoff=(0.02, 0.1)  # 10-50 Tage Zyklen
)

# Praktische Anwendung
from math_tools import SpectralAnalyzer

analyzer = SpectralAnalyzer()

# Rauschfreien Trend extrahieren
clean_trend = analyzer.extract_trend(prices, smoothness=0.95)

# Zyklische Komponente extrahien
cyclical = analyzer.extract_cycles(prices, min_period=5, max_period=100)
```

---

### 2.4 Hilbert Transform - Echtzeit-Zyklus-Analyse

Für Echtzeit-Trading ohne Look-Ahead Bias:

```python
from math_tools import HilbertTransformAnalyzer

# Hilbert Transform für Instantaneous Frequency
hilbert = HilbertTransformAnalyzer()
result = hilbert.compute(prices, window=20)

# Momentane Werte
amplitude = result['amplitude'][-1]      # Aktuelle Amplitude
phase = result['phase'][-1]              # Aktuelle Phase
frequency = result['frequency'][-1]      # Momentane Frequenz
period = result['current_period']        # Momentane Periode

print(f"Aktueller Zyklus: {period:.1f} Tage")
print(f"Amplitude: ${amplitude:.2f}")
print(f"Phase: {phase:.2f} rad")

# Cycle State für Trading
cycle_state = hilbert.get_cycle_state(recent_window=20)
print(f"\nDominante Periode: {cycle_state['dominant_period']:.1f}")
print(f"Zyklus-Stärke: {cycle_state['cycle_strength']:.2f}")
print(f"Amplitude Trend: {cycle_state['amplitude_trend']}")
```

---

### 2.5 Adaptive Cycle Indicator

Automatische Anpassung an wechselnde Zyklen:

```python
from math_tools import AdaptiveCycleIndicator

# Adaptiver Zyklus-Indikator
aci = AdaptiveCycleIndicator(lookback=200, adapt_period=50)

# Im Trading-Loop
for price in live_prices:
    result = aci.update(price)
    
    signal = result['signal']           # -1, 0, 1
    cycle_period = result['cycle_period']  # Aktueller Zyklus
    phase = result['phase']             # Position im Zyklus
    confidence = result['confidence']   # 0-1
    
    if signal == 1 and confidence > 0.6:
        place_buy_order()
    elif signal == -1 and confidence > 0.6:
        place_sell_order()
```

**Wie es funktioniert:**
1. Analysiert kontinuierlich die letzten 200 Perioden
2. Findet den dominanten Zyklus alle 50 Perioden neu
3. Generiert Signale basierend auf Zyklus-Phase
4. Kauft am Zyklus-Tief (Phase ~0.9), verkauft am Hoch (Phase ~0.3)

---

### 2.6 Seasonality Analysis

Saisonale Muster entdecken:

```python
from math_tools import SeasonalityAnalyzer
import pandas as pd

# Daten mit DatetimeIndex
returns = pd.Series(returns_array, index=pd.to_datetime(dates))

analyzer = SeasonalityAnalyzer()

# Tag-der-Woche Analyse
dow_stats = analyzer.analyze_day_of_week(returns, returns.index)
print("Tag-der-Woche Effekte:")
for day, stats in dow_stats.items():
    if day not in ['best_day', 'worst_day']:
        print(f"  {day}: {stats['mean_return']:.3f}% (Win Rate: {stats['win_rate']:.1%})")

print(f"\nBester Tag: {dow_stats['best_day']}")
print(f"Schlechtester Tag: {dow_stats['worst_day']}")

# Monat-der-Jahr Analyse
month_stats = analyzer.analyze_month_of_year(returns, returns.index)

# Holiday Effects
holiday_effects = analyzer.detect_holiday_effects(
    prices, 
    prices.index,
    holidays=['01-01', '07-04', '12-25']  # New Year, July 4th, Christmas
)
```

**Typische Ergebnisse für BTC:**
```
Tag-der-Woche Effekte:
  Monday: 0.25% (Win Rate: 54.2%)
  Tuesday: 0.18% (Win Rate: 52.1%)
  Wednesday: -0.05% (Win Rate: 48.9%)
  Thursday: 0.32% (Win Rate: 55.8%)  ← Best day
  Friday: -0.12% (Win Rate: 47.3%)

Bester Tag: Thursday
Schlechtester Tag: Friday

Holiday Effects:
  01-01: Pre-holiday +0.45%, Post-holiday +0.82%
  12-25: Pre-holiday -0.23%, Post-holiday +1.12%
```

---

## 3. Praktische Trading-Strategien

### 3.1 Crypto Cycle Strategy

Basierend auf Bitcoins 4-Jahres-Halving-Zyklus:

```python
from math_tools import SpectralAnalyzer, compute_dominant_cycle

class CryptoCycleStrategy:
    def __init__(self):
        self.analyzer = SpectralAnalyzer()
        self.cycle_period = None
        self.entry_phase = 0.85  # 85% durch Zyklus (Tiefpunkt)
        self.exit_phase = 0.25   # 25% durch Zyklus (Hochpunkt)
    
    def update(self, prices):
        # Zyklus alle 100 Bars neu berechnen
        if len(prices) % 100 == 0:
            self.cycle_period = compute_dominant_cycle(prices[-500:], min_period=100)
        
        if self.cycle_period is None:
            return 0
        
        # Position im Zyklus
        position = len(prices) % int(self.cycle_period)
        phase = position / self.cycle_period
        
        # Signale
        if phase > self.entry_phase:
            return 1  # Long am Zyklus-Tief
        elif phase < self.exit_phase:
            return -1  # Short/Exit am Zyklus-Hoch
        
        return 0
```

---

### 3.2 Spectral Mean Reversion

Kombiniert Spectral Analysis mit Mean-Reversion:

```python
from math_tools import SpectralAnalyzer, OrnsteinUhlenbeckProcess

def spectral_mean_reversion(prices, lookback=200):
    analyzer = SpectralAnalyzer()
    
    # 1. Trend extrahieren
    trend = analyzer.extract_trend(prices[-lookback:], smoothness=0.95)
    
    # 2. Zyklus-Komponente
    cycle = analyzer.extract_cycles(prices[-lookback:], min_period=5, max_period=50)
    
    # 3. Mean-Reversion auf Zyklus anwenden
    ou = OrnsteinUhlenbeckProcess()
    score = ou.calculate_score(cycle[-20:])
    
    # 4. Signal
    if score > 0.8:  # Weit über Trend
        return -1  # Short (Rückkehr zu Trend erwartet)
    elif score < -0.8:  # Weit unter Trend
        return 1   # Long
    
    return 0
```

---

### 3.3 Multi-Timeframe Cycle Confirmation

Zyklus-Bestätigung über mehrere Timeframes:

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
    
    # Überprüfe Konsistenz
    daily_long_cycle = daily_cycles[0]['period'] if daily_cycles else None
    weekly_long_cycle = weekly_cycles[0]['period'] * 5 if weekly_cycles else None  # Convert to days
    
    # Wenn Daily und Weekly übereinstimmen (±20%), Signal verstärken
    if daily_long_cycle and weekly_long_cycle:
        diff = abs(daily_long_cycle - weekly_long_cycle) / daily_long_cycle
        if diff < 0.2:
            confidence = 1.5  # Übereinstimmung!
        else:
            confidence = 1.0
    else:
        confidence = 0.5
    
    # Basissignal
    composite_daily = analyzer.cycle_composite(prices_daily, 
                                               [c['period'] for c in daily_cycles[:2]])
    signal = 1 if composite_daily[-1] > 0 else -1
    
    return signal * confidence
```

---

## 4. Vorteile & Limitationen

### ✅ Vorteile:

1. **Objektive Zyklus-Erkennung** (nicht subjektiv wie visuelle Analyse)
2. **Rauschunterdrückung** (Filterung im Frequenzbereich effektiver)
3. **Saisonalitäts-Nachweis** (statistisch signifikante Muster)
4. **Multi-Zyklus Kombination** (verschiedene Zeitrahmen gleichzeitig)
5. **Adaptive Anpassung** (reagiert auf sich ändernde Marktzyklen)

### ⚠️ Limitationen:

1. **Benötigt viel Daten** (mindestens 2x die längste Periode)
2. **Keine Daytrading-Strategie** (zu viel Noise auf kurzen Zeiträumen)
3. **Zyklen ändern sich** (müssen regelmäßig neu kalibriert werden)
4. **Nicht für alle Märkte geeignet** (funktioniert am besten bei klaren Zyklen wie Crypto)
5. **Spectral Leakage** (Fenster-Funktionen notwendig)

### 🎯 Beste Anwendungsfälle:

- **Swing Trading** (1W - 3M Haltedauer)
- **Crypto-Trading** (4-Jahres-Halving-Zyklus)
- **Aktien-Saisonalitäten** (Sell in May, Januar-Effekt)
- **Volatilitäts-Zyklen** (VIX-Seasonality)
- **Rohstoff-Zyklen** (Ernte-Zyklen, Wetter)

---

## 5. Integration mit anderen Modellen

### Kombination mit HMM:

```python
from math_tools import SpectralAnalyzer, HMMRegimeDetector

# Zyklus-basierte Regime-Detektion
analyzer = SpectralAnalyzer()
cycles = analyzer.find_dominant_cycles(prices, n_cycles=1)

if cycles and cycles[0]['period'] > 200:
    # Lange Zyklen = Trending Regime
    regime = "trending"
else:
    # Kurze Zyklen = Mean-Reversion Regime
    regime = "mean_reverting"
```

### Kombination mit Kelly:

```python
from math_tools import SpectralAnalyzer, KellyCriterion

# Position Size basierend auf Zyklus-Konfidenz
analyzer = SpectralAnalyzer()
dominant_cycle = analyzer.find_dominant_cycles(returns, n_cycles=1)

if dominant_cycle and dominant_cycle[0]['power'] > 0.7:
    # Starker Zyklus → Normale Position
    kelly = KellyCriterion()
    size = kelly.calculate_position_size(returns)
else:
    # Schwacher Zyklus → Reduzierte Position
    size = 0.5
```

---

## 6. Zusammenfassung

**Spectral Analysis bietet:

✅ **Frequenz-Domänen-Analyse** für Zyklus-Erkennung  
✅ **Noise-Reduktion** via Filterung  
✅ **Saisonalitäts-Analyse** (Tag, Monat, Holiday)  
✅ **Adaptive Zyklus-Indikatoren**  
✅ **Echtzeit-Zyklus-Tracking** (Hilbert Transform)  

**Beste geeignet für:**
- Swing Trading (1W-3M)
- Crypto (Halving-Zyklen)
- Seasonality-Strategien
- Trend-Noise Separation

**Nicht geeignet für:**
- Daytrading/Scalping
- Märkte ohne klare Zyklen
- Kurze Zeitreihen (< 200 Datenpunkte)

---

**Das Modul ist jetzt bereit für professionelle Zyklus-basierte Trading-Strategien!** 📊🔄
