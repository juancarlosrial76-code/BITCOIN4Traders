# Training Experiment Results

## Experiment Tracking - Phase 7 Model Training

### Ziel: Bidirektionaler Handel (Long + Short)

---

## Experiment 1: Standard Training (entropy=0.08)
- **Datum**: 2026-02-25
- **Ergebnis**: ❌ FAILED - 100% Short Action 1
- **Iterations**: 200
- **Analyse**: Policy-Kollaps

---

## Experiment 2: Erhöhte Entropy (entropy=0.20)
- **Datum**: 2026-02-25
- **Ergebnis**: ❌ FAILED - 100% Short Action 1
- **Iterations**: 52
- **Analyse**: Entropy-Erhöhung nicht ausreichend

---

## Experiment 3: Hohe Entropy (entropy=0.50)
- **Datum**: 2026-02-26
- **Ergebnis**: ❌ FAILED - 100% Short Action 1
- **Iterations**: 500
- **Analyse**: Fundamentaleres Problem - NICHT durch Entropy lösbar
- **Schlussfolgerung**: Wechsel zu Option 3 (ohne Adversary)

---

## Experiment 4: Training ohne Adversary
- **Ansatz**: adversary_start_iteration=999999, adversary_strength=0.0
- **Erwartung**: Agent lernt eigenständig ohne Adversary-Manipulation
- **Status**: GESTARTET
- **Config**: 
  - entropy_coef: 0.50
  - n_iterations: 300

---

## Experiment 5: Reward-System Overhaul (geplant)
- **Ansatz**: Reward-Gewichte ändern
- **Mögliche Änderungen**:
  - drawdown weight erhöhen
  - transaction cost erhöhen
  - Symmetrische Belohnung für Long/Short
- **Status**: AUSSTEHEND

---

## Training Konfiguration (aktuell)
```yaml
trader:
  entropy_coef: 0.50
  actor_lr: 1.0e-4
  n_epochs: 5
  batch_size: 128
training:
  n_iterations: 500
  steps_per_iteration: 1024
```
