# Phase 7 Training Experiments - Documentation

## Goal: Bidirectional Trading (Long + Short)

---

## Experiment Summary

### Experiment 1: Standard Training
- **Date**: 2026-02-25
- **Config**: entropy_coef=0.08
- **Iterations**: 200
- **Result**: ❌ FAILED - 100% Short (Action 1)
- **Analysis**: Policy collapse

---

### Experiment 2: Increased Entropy (0.20)
- **Date**: 2026-02-25
- **Config**: entropy_coef=0.20
- **Iterations**: 52
- **Result**: ❌ FAILED - 100% Short (Action 1)
- **Analysis**: Entropy increase not sufficient

---

### Experiment 3: High Entropy (0.50)
- **Date**: 2026-02-26
- **Config**: entropy_coef=0.50
- **Iterations**: 500
- **Result**: ❌ FAILED - 100% Short (Action 1)
- **Analysis**: Fundamental problem - NOT solvable by entropy alone
- **Conclusion**: Switching to Option 3 (without Adversary)

---

### Experiment 4: Training WITHOUT Adversary
- **Date**: 2026-02-27
- **Config**: 
  - adversary_start_iteration: 999999 (disabled)
  - adversary_strength: 0.0
  - entropy_coef: 0.50
- **Iterations**: 500
- **Result**: ❌ FAILED - 100% Long (Action 5)
- **Analysis**: Model now tends to Long instead of Short - training works but has market bias
- **Conclusion**: Problem is in Reward System

---

### Experiment 5: Symmetric Reward System
- **Date**: 2026-02-28
- **Result**: ❌ FAILED - 100% Neutral (Action 2)
- **Analysis**: Agent learned to stay neutral to avoid risk

---

### Experiment 6: Force Trading Reward (NEW)
- **Date**: 2026-02-28
- **Config Changes**:
  - Added "position_bonus": Reward for having ANY position (penalty for neutral)
  - Added "position_change": Reward for changing positions
  - Reduced transaction_cost: -0.5 → -0.2
  - Increased return weight: 1.0 → 2.0
- **Goal**: Force agent to take positions (not neutral)
- **Status**: IN PROGRESS

---

## Current Configuration

```yaml
training:
  n_iterations: 500
  steps_per_iteration: 1024
  save_frequency: 25
  adversary_start_iteration: 999999  # DISABLED
  adversary_strength: 0.0

trader:
  entropy_coef: 0.50
  actor_lr: 1.0e-4
  n_epochs: 5
  batch_size: 128
```

---

## Key Findings

1. **Training Works**: Model learns a direction (was Short, now Long)
2. **Bias Exists**: Model tends to one direction consistently
3. **Root Cause**: Likely in Reward System (favors one direction)

---

## Next Steps (Planned)

1. Continue training to 500 iterations
2. If still unbalanced: Modify Reward System
3. Consider symmetric reward for Long/Short

---

## Data Storage Notes

- Training creates large log files (~400MB per run)
- Need efficient storage: SQLite or CSV for metrics only
- Keep: rewards, action distribution, iteration count
- Discard: verbose debug logs

---

## Paper Trading Verification

After successful training:
1. Load model into Phase 7 engine
2. Run paper trading
3. Verify both Long and Short trades executed
4. Monitor PnL for both directions
