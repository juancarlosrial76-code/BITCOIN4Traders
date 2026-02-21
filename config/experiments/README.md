# Experiment Configurations

Place your personal experiment configs here.
These files are gitignored – only you see them locally.

## How to create your own experiment:

```bash
# Copy the base config
cp ../base/realistic_env.yaml my_experiment_01.yaml

# Edit your copy
nano my_experiment_01.yaml

# Run training with your config
python train.py --config config/experiments/my_experiment_01.yaml
```

## Naming convention:

```
config/experiments/
  alice_high_risk.yaml         # higher drawdown tolerance
  bob_conservative.yaml        # small positions, tight stops
  team_baseline_v2.yaml        # shared baseline (commit this one!)
```

## What to experiment with:

| Parameter | Base value | Try |
|---|---|---|
| `max_drawdown` | 0.70 | 0.50 – 0.90 |
| `max_position_size` | 0.30 | 0.10 – 0.50 |
| `transaction_costs.taker_fee_bps` | 5 | 3 – 10 |
| `slippage.enabled` | false | true |
| `reward.components[drawdown].weight` | -0.5 | -0.1 – -1.0 |

## Important:
- **Never** auto-modify configs from auto_train.py or auto_12h_train.py
- The base config in `config/base/` is the shared team baseline
- Commit experiment configs only if you want to share them with the team
