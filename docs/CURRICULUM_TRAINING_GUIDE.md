# Curriculum Training for RL-Based Trading Agents
## Solving the Kelly-Exploration Deadlock

**System:** BITCOIN4Traders — Adversarial PPO Trading Agent
**Date:** 2026-03-26
**Affected files:**
- `auto_12h_train.py`
- `src/risk/risk_manager.py`
- `src/math_tools/kelly_criterion.py`

---

## 1. Problem Statement

### 1.1 The Kelly-Exploration Deadlock

Reinforcement learning agents require exploration to learn: they must take actions, observe outcomes, and accumulate enough data for value-function and policy estimates to converge. The PPO agent in this system begins each training run with no trade history, which creates a fundamental conflict with the Kelly Criterion position sizer.

The Kelly Criterion computes the optimal fraction of capital to risk on a bet:

```
f* = (p·b − q) / b
```

Where:
- `p` = win probability
- `q` = 1 − p (loss probability)
- `b` = average win / average loss (win/loss ratio)

With no trade history, the system initialises with a win rate near 0% and a win/loss ratio estimated from noise. A representative early-training scenario (observed in logs) yields:

| Parameter | Value |
|-----------|-------|
| Win rate `p` | ≈ 0.15 |
| Win/loss ratio `b` | ≈ 5.577 |
| Kelly fraction | `(0.15 × 5.577 − 0.85) / 5.577 ≈ −0.0024` |
| Resulting position size | **0%** |

Kelly correctly returns zero or negative: there is no demonstrated positive edge. However, the consequence for training is catastrophic: with 0% position size, the agent never executes trades in the environment, never receives reward signals from trade outcomes, and cannot learn. The agent is stuck — it cannot demonstrate edge without trading, and Kelly will not allow trading until edge is demonstrated.

This is the **Kelly-Exploration Deadlock**.

### 1.2 Insufficient Training Iterations

PPO requires extensive environment interaction for stable convergence. The prior configuration ran only **10 iterations** per training subprocess call. For a complex financial environment with:
- Continuous price data
- Multi-dimensional feature space
- Sparse reward signal (trades are infrequent)
- Non-stationary market dynamics

10 iterations represent less than 1% of the interaction budget typically needed for initial policy improvement. No learning is observable at this scale.

---

## 2. Solution: Three-Phase Curriculum Training

The implemented solution separates the training lifecycle into distinct phases, each with appropriate risk parameters. This mirrors the curriculum learning approach used in leading RL research (e.g., OpenAI Five, AlphaZero, DeepMind's safety-constrained RL work).

```
Phase 1: Exploration          Phase 2: Refinement
─────────────────────    →    ──────────────────────
Kelly: OFF (bypass)           Kelly: ON (adaptive floor)
Entropy: high                 Entropy: standard
Iterations: 300/run           Iterations: 300/run
Trigger: start                Trigger: win_rate > 18%
```

### 2.1 Phase 1 — Unconstrained Exploration

Kelly Criterion is disabled for the duration of Phase 1 training. The `TRAINING_MODE=1` environment variable, set by `auto_12h_train.py` before each subprocess call, propagates through to `RiskManager`, which skips Kelly Stage 2 entirely in `validate_position_size()`.

This allows the agent to:
- Execute trades and accumulate a trade history
- Receive diverse reward signals (wins, losses, neutral)
- Learn accurate value function estimates via GAE
- Develop an initial policy that distinguishes profitable from unprofitable market conditions

Position sizing during Phase 1 is governed only by the hard cap (`max_position_size = 25%`), preventing catastrophic single-trade losses while permitting meaningful position sizes.

### 2.2 Phase 1 → Phase 2 Transition

The system monitors win rate from each training run's stdout (log line format: `Win Rate: X.X%`). When a rolling win rate exceeds **18%** for a completed training run, the system automatically transitions to Phase 2:

```python
if phase == 1 and last_win_rate > 0.18:
    phase = 2
    training_mode = False  # re-enables Kelly
```

The 18% threshold was chosen to be:
- Above random chance (33% for 3-action space, but actions cluster — realistically ~15–20% for profitable trades)
- Below the target operating win rate (typically 40–60% in profitable configurations)
- Low enough that the transition happens after genuine learning, not statistical noise

### 2.3 Phase 2 — Adaptive Kelly with Safety Net

Once Phase 2 is active, Kelly Criterion is re-enabled. An adaptive safety net in `dynamic_kelly()` prevents complete position blocking if Kelly would still be negative:

```python
if raw_kelly_f < 0 and TRAINING_MODE == "1":
    adaptive_floor = max(0.02 * (1 - win_rate), 0.005)
    return capital * adaptive_floor
```

The adaptive floor has two key properties:

1. **Self-correcting**: The floor = `2% × loss_rate` shrinks automatically as the win rate improves. At 20% win rate, floor ≈ 1.6%. At 40% win rate, floor ≈ 1.2%. At 60% win rate, floor ≈ 0.8%.

2. **Minimum guarantee**: Never falls below 0.5%, ensuring some market interaction continues during the transition period.

In production (no `TRAINING_MODE` env var), this safety net is inactive. Kelly returns 0 on negative EV as designed.

---

## 3. Implementation Details

### 3.1 Environment Variable Protocol

The training mode state is communicated via the `TRAINING_MODE` OS environment variable. This approach was chosen over config file injection because:

- `auto_12h_train.py` launches `train.py` as an isolated subprocess
- The subprocess inherits the environment of its parent
- No file I/O or config parsing overhead is added to the hot path
- The setting is process-scoped and cannot accidentally persist to live trading

```python
# auto_12h_train.py — sets for subprocess
env["TRAINING_MODE"] = "1" if training_mode else "0"

# risk_manager.py — reads at RiskManager init time
if os.environ.get("TRAINING_MODE", "0") == "1":
    self.config.training_mode = True
```

### 3.2 Iteration Count

The per-call iteration count was increased from **10 to 300**. The subprocess timeout was increased from 1200s to 7200s (2 hours) to accommodate this.

Within the 12-hour training window, the outer loop in `auto_12h_train.py` will execute approximately 4–6 complete 300-iteration runs. This provides:
- ~1200–1800 total PPO iterations in 12 hours
- Multiple full training runs with saved checkpoints between them
- Progressive improvement tracked via `get_best_return()`

### 3.3 Emergency Stop

A safety check prevents infinite Phase 1 loops in degenerate cases:

```python
if iteration > 10 and last_win_rate < 0.005 and phase == 1:
    log("EMERGENCY STOP: Win Rate at 0% despite training mode — check win rate calculation")
    break
```

If the win rate cannot exceed 0.5% after 10 rounds (3000 iterations), there is a structural problem with the environment, data, or reward function that training cannot overcome. Manual investigation is required.

---

## 4. Mathematical Background

### 4.1 Why Kelly Blocks at Low Win Rates

The Kelly fraction for a 3-outcome trading environment (buy/hold/sell) can be approximated as a binary bet where a "win" is a profitable closed trade. For a newly initialised agent:

- The policy is near-uniform random → ~33% of actions are "correct" directionally
- But correct direction does not guarantee profit (due to fees, spreads, timing)
- Effective win rate in early training: typically 10–20%
- With win/loss ratio b ≈ 5 (large wins are rare but agent discovers them randomly):

```
f* = (0.15 × 5 − 0.85) / 5 = (0.75 − 0.85) / 5 = −0.02
```

Any negative Kelly maps to 0% position size via `np.clip(kelly_f, 0.0, max_position)`.

### 4.2 Why 300 Iterations is the Right Scale

PPO convergence requires the critic (value function) to provide accurate advantage estimates. This requires:
- **Warm-up**: ~50–100 iterations for the value function to move from random initialisation toward a useful baseline
- **Initial policy improvement**: ~100–200 iterations for the actor to find the first exploitable patterns
- **Stabilisation**: ~200–500 iterations before the policy entropy meaningfully decreases

300 iterations per run sits in the "initial policy improvement" window — enough to show measurable progress, not so long that a single run monopolises the 12-hour window.

### 4.3 The Entropy Coefficient

The PPO agent uses `entropy_coef = 0.05` (already configured in `PPOConfig`). This entropy bonus in the loss function:

```
L = L_clip − c1 × L_value + c2 × H(π)
```

encourages the policy to maintain high entropy (exploration diversity) during training. Combined with the Kelly bypass, this ensures Phase 1 is genuinely exploratory rather than collapsing prematurely to a deterministic sub-optimal policy.

---

## 5. Expected Training Trajectory

| Phase | Round | Win Rate | Kelly Status | Expected Behaviour |
|-------|-------|----------|--------------|-------------------|
| 1 | 1–3 | 0–10% | Bypassed | Agent explores, first reward signals accumulated |
| 1 | 4–8 | 10–18% | Bypassed | Value function stabilises, policy begins improving |
| 2 | 9–15 | 18–35% | Adaptive floor | Kelly permits small positions, reward signal reinforces edge |
| 2 | 16+ | 35%+ | Standard Kelly | Full position sizing, performance optimisation |

---

## 6. Monitoring

During training runs, observe these signals in `logs/training/12h_auto.log`:

| Signal | Healthy | Warning |
|--------|---------|---------|
| `Win Rate` | Increases run-over-run in Phase 1 | Flat at 0% → check reward function |
| `Phase 2` transition log | Appears within 5–10 rounds | Never appears → increase iterations or check data |
| `Mean Return` | Trending upward | Oscillating without trend → reduce learning rate |
| Kelly bypass log line | Present every round in Phase 1 | Absent → TRAINING_MODE env var not propagating |

To verify `TRAINING_MODE` propagation manually:

```bash
TRAINING_MODE=1 python3 -c "
from src.risk.risk_manager import RiskConfig, RiskManager
rm = RiskManager(RiskConfig(), initial_capital=10000)
print('training_mode active:', rm.config.training_mode)
"
```

---

## 7. Safety Boundaries

The following production safeguards remain **active during all training phases**:

| Safeguard | Mechanism | Training Impact |
|-----------|-----------|-----------------|
| Hard position cap | `max_position_size = 25%` | Single trade cannot exceed 25% of capital |
| Session drawdown limit | `max_drawdown_per_session = 2%` | Circuit breaker fires if session loses >2% |
| VaR-based position cap | Stage 2b in `validate_position_size()` | Tail-risk positions are still capped |
| Max consecutive losses | `max_consecutive_losses = 5` | Streak of losses halts trading |

Kelly bypass does **not** disable these controls. Training mode only affects the dynamic position sizing optimisation layer.
