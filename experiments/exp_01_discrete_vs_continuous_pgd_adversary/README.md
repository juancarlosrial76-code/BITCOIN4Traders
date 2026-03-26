# Experiment: Discrete vs Continuous (PGD) Adversarial Training

## Experiment Overview

**Experiment ID:** `exp_01_discrete_vs_continuous_pgd_adversary`  
**Title:** Discrete vs Continuous (PGD) Adversarial Training - Profit Comparison  
**Version:** 2.0 (Extended to 3 approaches)  
**Date:** 2026-03-13  
**Status:** Ready for Execution

---

## Research Question

**Does Continuous (PGD) adversarial training produce more profitable traders than Discrete adversarial training?**

**Extended Question:** Is more flexibility in discrete actions sufficient, or is continuous (PGD) truly necessary?

---

## Three Approaches Tested

| #   | Approach             | Description                                                         | Actions           |
| --- | -------------------- | ------------------------------------------------------------------- | ----------------- |
| 1   | **Discrete-4**       | Current implementation (4 fixed actions: noise, bias, invert, none) | 4 discrete        |
| 2   | **Discrete-16**      | Extended discrete with 16 actions (different strengths)             | 16 discrete       |
| 3   | **Continuous (PGD)** | Gaussian policy with L2-norm clipping, perturbation per feature     | Continuous vector |

---

## Hypothesis & Interpretation

| If Result                | Interpretation                                  |
| ------------------------ | ----------------------------------------------- |
| Discrete-16 ≈ Continuous | Flexibility is the key, not continuous space    |
| Continuous > Discrete-16 | Continuous space provides fundamental advantage |
| All similar              | Adversarial training method doesn't matter much |

---

## Experiment Design

### Phase 1: Training (3 traders)

| Parameter          | Value                |
| ------------------ | -------------------- |
| Iterations         | 300 per trader       |
| Adversary Start    | Iteration 50         |
| Adversary Strength | 0.1                  |
| PGD Epsilon        | 0.02 (2% max)        |
| Batch Size         | 512 (GPU) / 64 (CPU) |
| Hidden Dimension   | 256                  |
| Parallel Envs      | 16 (GPU)             |

### Phase 2: Comparison

- Compare final profits
- Determine winner
- Generate recommendation

---

## Metric: PROFIT (Only)

| Metric          | Description                            |
| --------------- | -------------------------------------- |
| Training Profit | Final cumulative profit after training |

**Winner:** The approach with highest profit.

---

## Files

| File               | Description         |
| ------------------ | ------------------- |
| `experiment.ipynb` | Main Colab notebook |
| `README.md`        | This documentation  |

---

## Execution Instructions

### 1. Open in Google Colab

```
experiments/exp_01_discrete_vs_continuous_pgd_adversary/experiment.ipynb
```

### 2. Select GPU Runtime

- Runtime → Change runtime type → T4 GPU

### 3. Run All Cells

The notebook will:

1. Load BTC data (2022-2024)
2. Compute features
3. Train Discrete-4 adversary trader
4. Train Discrete-16 adversary trader
5. Train Continuous (PGD) adversary trader
6. Compare profits
7. Generate interpretation & recommendation
8. Save results to Google Drive

### 4. Expected Duration

- Training: 3-6 hours total (1-2h per approach)
- Comparison: 5 minutes

---

## Results Template

After experiment completion:

```
═══════════════════════════════════════════════════
RESULTS
═══════════════════════════════════════════════════
Rank | Approach      | Profit     | Time (min)
-----|---------------|------------|------------
  1  | ???          | ???        | ???
  2  | ???          | ???        | ???
  3  | ???          | ???        | ???

🏆 WINNER: ???

═══════════════════════════════════════════════════
INTERPRETATION
═══════════════════════════════════════════════════
- ???

═══════════════════════════════════════════════════
RECOMMENDATION
═══════════════════════════════════════════════════
- ???

═══════════════════════════════════════════════════
NEXT STEPS
═══════════════════════════════════════════════════
- Archive results
- Implement winning approach
- Run paper trading (optional)
```

---

## Technical Details

### Discrete-4 (Baseline)

- 4 discrete actions: noise, trend bias, signal inversion, none
- Standard PPO with categorical distribution
- Same as current implementation

### Discrete-16 (Extended)

- 16 discrete actions with different strengths:
  - Actions 0-3: Noise (strength 0.1, 0.2, 0.3, 0.5)
  - Actions 4-7: Trend bias (different strengths)
  - Actions 8-11: Signal inversion (different # features)
  - Actions 12-15: No modification

### Continuous (PGD)

- Gaussian policy: outputs mean perturbation vector
- L2-norm clipping: `||perturbation|| <= epsilon` (2%)
- One perturbation value per feature
- More fine-grained control

---

## Notes

- All 3 traders use **identical** hyperparameters except for adversary type
- Results are saved to Google Drive: `MyDrive/BITCOIN4Traders/experiments/`
- Version 2.0 extends original to answer: "Is more flexibility in discrete enough?"

---

## Author

BITCOIN4Traders Research Team  
**Experiment ID:** `exp_01_discrete_vs_continuous_pgd_adversary`  
**Version:** 2.0
