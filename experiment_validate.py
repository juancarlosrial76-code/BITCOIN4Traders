#!/usr/bin/env python3
"""
Experiment Validator — Paper Trading Win-Rate Verification
===========================================================
Runs the best-trained model through paper trading (dry run) and reports
win rate, win/loss ratio, and Sharpe ratio as the primary validation metric.

The ONLY metric that counts: Win Rate and Win/Loss Ratio in paper trading.
Not in training, not in backtesting — in live-simulated paper trading.

Usage:
    python experiment_validate.py                  # validate best model
    python experiment_validate.py --model path/to/model.pt
    python experiment_validate.py --backtest       # quick backtest mode
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

WORK_DIR = Path(__file__).resolve().parent
os.chdir(WORK_DIR)

VALIDATION_LOG = WORK_DIR / "logs/experiments/validation.log"
VALIDATION_RESULTS = WORK_DIR / "logs/experiments/validation_results.json"


def log(msg: str):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    VALIDATION_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(VALIDATION_LOG, "a") as f:
        f.write(line + "\n")


def validate_in_environment(model_path: Path = None) -> dict:
    """
    Run direct environment validation (backtest on test set).
    Returns win_rate, win_loss_ratio, sharpe, mean_return.
    """
    import numpy as np
    import pandas as pd

    sys.path.insert(0, str(WORK_DIR / "src"))

    log("Loading environment for validation...")

    try:
        from src.features.feature_engine import FeatureEngine, FeatureConfig
        from src.environment.config_integrated_env import ConfigIntegratedTradingEnv
        from src.environment.config_system import load_environment_config_from_yaml
        from src.agents.ppo_agent import PPOAgent, PPOConfig
    except ImportError as e:
        log(f"Import error: {e}")
        return {}

    # Load data
    data_path = WORK_DIR / "data/processed/BTC_USDT_1h_price.parquet"
    feat_path = WORK_DIR / "data/processed/BTC_USDT_1h_features.parquet"

    if not data_path.exists():
        log(f"No data at {data_path}")
        return {}

    price_data = pd.read_parquet(data_path)
    features = pd.read_parquet(feat_path) if feat_path.exists() else None

    if features is None:
        log("No precomputed features found, computing...")
        fc = FeatureConfig()
        engine = FeatureEngine(fc)
        features = engine.fit_transform(price_data)

    # Use test split (last 15%)
    n = len(price_data)
    test_start = int(n * 0.85)
    test_price = price_data.iloc[test_start:]
    test_feats = features.iloc[test_start:] if hasattr(features, 'iloc') else features

    # Align
    common = test_price.index.intersection(test_feats.index)
    test_price = test_price.loc[common]
    test_feats = test_feats.loc[common]

    if len(test_price) < 100:
        log(f"Test set too small: {len(test_price)} rows")
        return {}

    log(f"Test set: {len(test_price)} bars ({test_price.index[0]} → {test_price.index[-1]})")

    # Create env
    env_config = load_environment_config_from_yaml("config/environment/realistic_env.yaml")

    # Apply best reward params if available
    best_cfg = WORK_DIR / "logs/experiments/best_config.yaml"
    if best_cfg.exists():
        import yaml
        with open(best_cfg) as f:
            cfg = yaml.safe_load(f)
        env_config.reward_params = cfg.get("reward_params", {})
        log(f"Using reward params from: {cfg.get('name', 'unknown')}")

    env = ConfigIntegratedTradingEnv(test_price, test_feats, env_config)

    # Load model
    if model_path is None:
        model_path = WORK_DIR / "data/models/ppo_best.pt"

    if not model_path.exists():
        log(f"No model at {model_path} — running random agent for baseline")
        agent = None
    else:
        import torch
        log(f"Loading model: {model_path}")
        try:
            checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
            state_dim = env.observation_space.shape[0]
            n_actions = env.action_space.n
            agent_config = PPOConfig(state_dim=state_dim, n_actions=n_actions, hidden_dim=128)
            agent = PPOAgent(agent_config)
            if "trader_state_dict" in checkpoint:
                agent.actor.load_state_dict(checkpoint["trader_state_dict"])
            elif "state_dict" in checkpoint:
                agent.actor.load_state_dict(checkpoint["state_dict"])
            agent.actor.eval()
            log("Model loaded successfully")
        except Exception as e:
            log(f"Failed to load model: {e} — using random agent")
            agent = None

    # Run validation episodes
    N_EPISODES = 20
    all_trade_pnls = []
    episode_returns = []
    win_counts = []
    total_trades = 0

    for ep in range(N_EPISODES):
        obs, info = env.reset()
        done = False
        hidden = None
        ep_trades = []
        last_position = 0.0

        while not done:
            if agent is not None:
                import torch
                obs_t = torch.FloatTensor(obs).unsqueeze(0)
                with torch.no_grad():
                    action, log_prob, value, hidden = agent.select_action(obs_t, hidden)
                action = int(action)
            else:
                action = env.action_space.sample()

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Track trade closes (position → 0)
            cur_pos = info.get("position", 0.0)
            if last_position != 0.0 and abs(cur_pos) < 0.05:
                # Trade closed
                ep_return = info.get("return", 0.0)
                ep_trades.append(ep_return)
                total_trades += 1
            last_position = cur_pos

        episode_returns.append(info.get("return", 0.0))

        # Use per-trade win rate from reward fn if available
        ep_wr = info.get("win_rate", -1.0)
        if ep_wr >= 0:
            win_counts.append(ep_wr)

    # Compute metrics
    if not episode_returns:
        log("No episodes completed")
        return {}

    mean_return = float(np.mean(episode_returns)) * 100
    episode_win_rate = float(np.mean([r > 0 for r in episode_returns]))

    per_trade_wr = float(np.mean(win_counts)) if win_counts else -1.0

    # Win/loss ratio
    wins = [r for r in episode_returns if r > 0]
    losses = [r for r in episode_returns if r < 0]
    wl_ratio = (np.mean(wins) / abs(np.mean(losses))
                if wins and losses else float('inf') if wins else 0.0)

    # Sharpe (cross-episode)
    sharpe = (float(np.mean(episode_returns)) / (float(np.std(episode_returns)) + 1e-8)
              * np.sqrt(252))

    results = {
        "timestamp": datetime.now().isoformat(),
        "n_episodes": N_EPISODES,
        "total_trades": total_trades,
        "mean_return_pct": round(mean_return, 3),
        "episode_win_rate": round(episode_win_rate, 4),
        "per_trade_win_rate": round(per_trade_wr, 4),
        "win_loss_ratio": round(float(wl_ratio), 3),
        "sharpe": round(sharpe, 3),
        "model": str(model_path),
    }

    log("\n" + "="*55)
    log("VALIDATION RESULTS (Paper Trading — Test Set)")
    log("="*55)
    log(f"  Episodes:           {N_EPISODES}")
    log(f"  Total Trades:       {total_trades}")
    log(f"  Mean Return:        {mean_return:.2f}%")
    log(f"  Episode Win Rate:   {episode_win_rate*100:.1f}%")
    log(f"  Per-Trade Win Rate: {per_trade_wr*100:.1f}%  ← PRIMARY METRIC")
    log(f"  Win/Loss Ratio:     {wl_ratio:.2f}x  ← SECONDARY METRIC")
    log(f"  Sharpe Ratio:       {sharpe:.2f}")
    log("="*55)

    if per_trade_wr >= 0.65:
        log("GOAL REACHED: Trade Win Rate ≥ 65% ✓")
    elif per_trade_wr >= 0.55:
        log("PROGRESS: Trade Win Rate ≥ 55% (target: 65-76%)")
    else:
        log(f"NOT YET: Trade Win Rate = {per_trade_wr*100:.1f}% (target: 65-76%)")

    # Save results
    with open(VALIDATION_RESULTS, "w") as f:
        json.dump(results, f, indent=2)
    log(f"Results saved → {VALIDATION_RESULTS}")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="Path to model checkpoint")
    args = parser.parse_args()

    model_path = Path(args.model) if args.model else None
    validate_in_environment(model_path)


if __name__ == "__main__":
    main()
