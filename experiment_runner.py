#!/usr/bin/env python3
"""
Experiment Runner — Win-Rate Hyperparameter Search
====================================================
Runs N short training experiments with different WinRateAwareReward parameters,
records results, picks the best config, then optionally starts a full training run.

Usage:
    python experiment_runner.py                        # run all 5 experiments
    python experiment_runner.py --exp exp_B_balanced   # run one experiment
    python experiment_runner.py --full-run             # run best config for 24h

Results are saved to logs/experiments/results.json
Best config is written to logs/experiments/best_config.yaml
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

WORK_DIR = Path(__file__).resolve().parent
os.chdir(WORK_DIR)

LOG_DIR = WORK_DIR / "logs/experiments"
LOG_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_FILE = LOG_DIR / "results.json"
BEST_CONFIG_FILE = LOG_DIR / "best_config.yaml"


def log(msg: str):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_DIR / "runner.log", "a") as f:
        f.write(line + "\n")


def load_experiment_configs() -> list[dict]:
    config_dir = WORK_DIR / "experiment_configs"
    configs = []
    for path in sorted(config_dir.glob("exp_*.yaml")):
        with open(path) as f:
            cfg = yaml.safe_load(f)
        cfg["_path"] = str(path)
        configs.append(cfg)
    return configs


def run_experiment(exp_cfg: dict) -> dict:
    """Run a single experiment and return metrics."""
    name = exp_cfg["name"]
    iterations = exp_cfg.get("iterations", 50)
    reward_params = exp_cfg.get("reward_params", {})

    log(f"\n{'='*50}")
    log(f"Starting: {name}")
    log(f"Params: {reward_params}")
    log(f"Iterations: {iterations}")

    log_path = LOG_DIR / f"{name}_{datetime.now().strftime('%H%M%S')}.log"

    env = os.environ.copy()
    env["PYTHONPATH"] = str(WORK_DIR / "src")
    env["TRAINING_MODE"] = "1"
    env["EXPERIMENT_NAME"] = name
    # Pass reward params as JSON env var — train.py reads this below
    env["REWARD_PARAMS"] = json.dumps(reward_params)

    cmd = [
        "python3", "train.py",
        "--device", "cpu",
        "--iterations", str(iterations),
    ]

    t0 = time.time()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=7200,
            env=env,
            cwd=WORK_DIR,
        )
        elapsed = time.time() - t0

        with open(log_path, "w") as f:
            f.write(result.stdout)
            f.write("\nSTDERR:\n")
            f.write(result.stderr)

        metrics = parse_metrics(result.stdout, name)
        metrics["elapsed_s"] = round(elapsed)
        metrics["success"] = result.returncode == 0
        metrics["log"] = str(log_path)

        log(f"  Win Rate:      {metrics.get('win_rate', -1)*100:.1f}%")
        log(f"  Trade WR:      {metrics.get('trade_win_rate', -1)*100:.1f}%")
        log(f"  Mean Return:   {metrics.get('mean_return', -999):.2f}%")
        log(f"  Sharpe:        {metrics.get('sharpe', 0):.2f}")
        log(f"  Elapsed:       {elapsed/60:.1f} min")

        return metrics

    except subprocess.TimeoutExpired:
        log(f"  TIMEOUT after {7200}s")
        return {"name": name, "success": False, "error": "timeout"}
    except Exception as e:
        log(f"  ERROR: {e}")
        return {"name": name, "success": False, "error": str(e)}


def parse_metrics(stdout: str, name: str) -> dict:
    """Parse training output for key metrics."""
    metrics = {"name": name, "win_rate": -1.0, "trade_win_rate": -1.0,
               "mean_return": -999.0, "sharpe": 0.0, "calmar": 0.0}

    last_wr = -1.0
    last_twr = -1.0
    last_return = -999.0
    last_sharpe = 0.0
    last_calmar = 0.0

    for line in stdout.splitlines():
        # Episode win rate (fraction of profitable episodes)
        if "Win Rate:" in line:
            try:
                val = float(line.split("Win Rate:")[1].split("%")[0].strip())
                last_wr = val / 100.0
            except Exception:
                pass

        # Per-trade win rate (from WinRateAwareReward)
        if "Trade WR" in line:
            try:
                val = float(line.split("Trade WR")[1].split("%")[0].strip())
                last_twr = val / 100.0
            except Exception:
                pass

        if "Mean Return:" in line and "%" in line:
            try:
                val = float(line.split("Mean Return:")[1].split("%")[0].strip())
                last_return = val
            except Exception:
                pass

        if "Episode Sharpe:" in line:
            try:
                val = float(line.split("Episode Sharpe:")[1].strip())
                last_sharpe = val
            except Exception:
                pass

        if "Calmar Ratio:" in line:
            try:
                val = float(line.split("Calmar Ratio:")[1].strip())
                last_calmar = val
            except Exception:
                pass

    metrics["win_rate"] = last_wr
    metrics["trade_win_rate"] = last_twr
    metrics["mean_return"] = last_return
    metrics["sharpe"] = last_sharpe
    metrics["calmar"] = last_calmar
    return metrics


def score_experiment(metrics: dict) -> float:
    """
    Composite score to rank experiments.
    Primary: trade_win_rate (most important for our goal)
    Secondary: mean_return and sharpe
    Tertiary: episode win_rate
    """
    if not metrics.get("success", True):
        return -999.0

    twr = metrics.get("trade_win_rate", -1.0)
    wr = metrics.get("win_rate", -1.0)
    ret = metrics.get("mean_return", -999.0)
    sharpe = metrics.get("sharpe", 0.0)

    # Use trade_win_rate if available, else fall back to episode win_rate
    primary_wr = twr if twr >= 0 else wr

    score = (
        primary_wr * 3.0        # 3x weight on win rate
        + max(ret / 100.0, -1.0) * 1.0  # return (scaled, floored)
        + sharpe * 0.5           # Sharpe bonus
    )
    return score


def save_results(all_results: list[dict]):
    with open(RESULTS_FILE, "w") as f:
        json.dump(all_results, f, indent=2)
    log(f"Results saved → {RESULTS_FILE}")


def pick_best(all_results: list[dict]) -> dict:
    scored = [(score_experiment(r), r) for r in all_results if r.get("success", False)]
    if not scored:
        log("No successful experiments!")
        return {}
    scored.sort(key=lambda x: x[0], reverse=True)
    best_score, best = scored[0]
    log(f"\nBest: {best['name']} (score={best_score:.3f})")
    return best


def print_table(all_results: list[dict]):
    log("\n" + "="*70)
    log("EXPERIMENT RESULTS")
    log("="*70)
    log(f"{'Name':<25} {'TradeWR':>8} {'WR':>8} {'Return':>10} {'Sharpe':>8} {'Score':>8}")
    log("-"*70)
    for r in sorted(all_results, key=lambda x: score_experiment(x), reverse=True):
        twr = r.get("trade_win_rate", -1.0)
        wr = r.get("win_rate", -1.0)
        ret = r.get("mean_return", -999.0)
        sharpe = r.get("sharpe", 0.0)
        sc = score_experiment(r)
        log(f"{r['name']:<25} {twr*100:>7.1f}% {wr*100:>7.1f}% {ret:>9.2f}% {sharpe:>8.2f} {sc:>8.3f}")
    log("="*70)


def write_best_config(best_result: dict, exp_configs: list[dict]):
    """Write the best experiment's reward_params to best_config.yaml."""
    if not best_result:
        return
    name = best_result["name"]
    for cfg in exp_configs:
        if cfg["name"] == name:
            with open(BEST_CONFIG_FILE, "w") as f:
                yaml.dump(cfg, f, default_flow_style=False)
            log(f"Best config written → {BEST_CONFIG_FILE}")
            return


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", help="Run single experiment by name")
    parser.add_argument("--full-run", action="store_true",
                        help="After experiments, start 24h training with best config")
    parser.add_argument("--skip-training", action="store_true",
                        help="Skip training, just show existing results")
    args = parser.parse_args()

    exp_configs = load_experiment_configs()
    log(f"Found {len(exp_configs)} experiment configs")

    if args.skip_training:
        if RESULTS_FILE.exists():
            with open(RESULTS_FILE) as f:
                all_results = json.load(f)
            print_table(all_results)
        else:
            log("No results file found.")
        return

    if args.exp:
        exp_configs = [c for c in exp_configs if c["name"] == args.exp]
        if not exp_configs:
            log(f"Experiment '{args.exp}' not found.")
            sys.exit(1)

    all_results = []
    for exp_cfg in exp_configs:
        metrics = run_experiment(exp_cfg)
        metrics["reward_params"] = exp_cfg.get("reward_params", {})
        all_results.append(metrics)
        save_results(all_results)  # save incrementally

    print_table(all_results)
    best = pick_best(all_results)
    write_best_config(best, exp_configs)

    if args.full_run and best:
        log(f"\nStarting 24h full training with best config: {best['name']}")
        best_params = best.get("reward_params", {})
        env = os.environ.copy()
        env["PYTHONPATH"] = str(WORK_DIR / "src")
        env["TRAINING_MODE"] = "1"
        env["REWARD_PARAMS"] = json.dumps(best_params)
        subprocess.Popen(
            ["python3", "auto_12h_train.py"],
            env=env, cwd=WORK_DIR,
            stdout=open(LOG_DIR / f"full_train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log", "w"),
            stderr=subprocess.STDOUT,
        )
        log("Full training started in background.")


if __name__ == "__main__":
    main()
