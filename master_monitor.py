#!/usr/bin/env python3
"""
Master Monitor — Overnight Training & Paper Trading Supervisor
===============================================================
Monitors all running processes, experiment logs, paper trading logs.
Runs autonomously and reports status every CHECK_INTERVAL seconds.

Usage:
    python master_monitor.py              # run indefinitely
    python master_monitor.py --once       # single status check
    python master_monitor.py --summary    # final report only
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

WORK_DIR = Path(__file__).resolve().parent
LOG_DIR = WORK_DIR / "logs"
MONITOR_LOG = LOG_DIR / "master_monitor.log"
CHECK_INTERVAL = 900  # 15 minutes


def ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(msg: str):
    line = f"[{ts()}] {msg}"
    print(line, flush=True)
    MONITOR_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(MONITOR_LOG, "a") as f:
        f.write(line + "\n")


def pgrep(pattern: str) -> list[int]:
    """Return list of PIDs matching pattern."""
    try:
        out = subprocess.check_output(
            ["pgrep", "-f", pattern], text=True, stderr=subprocess.DEVNULL
        )
        return [int(p) for p in out.strip().splitlines() if p.strip()]
    except subprocess.CalledProcessError:
        return []


def tail_log(path: Path, n: int = 30) -> str:
    if not path.exists():
        return "(no log)"
    try:
        result = subprocess.run(
            ["tail", "-n", str(n), str(path)],
            capture_output=True, text=True,
        )
        return result.stdout
    except Exception:
        return "(read error)"


def get_latest_log(pattern: str) -> Path | None:
    logs = sorted(Path(WORK_DIR / "logs").rglob(pattern), key=lambda p: p.stat().st_mtime)
    return logs[-1] if logs else None


def parse_metric(text: str, key: str) -> str:
    """Extract last occurrence of 'Key: VALUE' from text."""
    matches = re.findall(rf"{re.escape(key)}[:\s]+([^\n]+)", text)
    return matches[-1].strip() if matches else "N/A"


def check_training() -> dict:
    pids = pgrep("auto_12h_train") + pgrep("train.py")
    status = {
        "alive": len(pids) > 0,
        "pids": pids,
        "win_rate": "N/A",
        "trade_win_rate": "N/A",
        "mean_return": "N/A",
        "iteration": "N/A",
        "phase": "N/A",
    }

    # Find latest training log
    log_path = get_latest_log("12h_*.log") or get_latest_log("train_*.log")
    if log_path:
        content = tail_log(log_path, 80)
        status["log_path"] = str(log_path)
        status["win_rate"] = parse_metric(content, "Win Rate")
        status["trade_win_rate"] = parse_metric(content, "Trade WR")
        status["mean_return"] = parse_metric(content, "Mean Return")
        status["iteration"] = parse_metric(content, "Round")
        status["phase"] = parse_metric(content, "Phase")

    return status


def check_paper_trading() -> dict:
    pids = pgrep("run.py")
    status = {
        "alive": len(pids) > 0,
        "pids": pids,
        "ticks": "N/A",
        "errors": 0,
    }

    log_path = get_latest_log("paper_*.log")
    if log_path:
        content = tail_log(log_path, 40)
        status["log_path"] = str(log_path)
        status["ticks"] = parse_metric(content, "tick")
        errors = content.lower().count("error")
        status["errors"] = errors

    return status


def check_experiments() -> dict:
    results_file = WORK_DIR / "logs/experiments/results.json"
    if not results_file.exists():
        return {"running": False, "results": []}

    with open(results_file) as f:
        results = json.load(f)

    exp_pids = pgrep("experiment_runner")
    return {
        "running": len(exp_pids) > 0,
        "n_completed": len(results),
        "results": results,
    }


def check_model() -> dict:
    model_path = WORK_DIR / "data/models/ppo_best.pt"
    champ_path = WORK_DIR / "data/models/champion.json"

    if not model_path.exists():
        return {"saved": False}

    mtime = datetime.fromtimestamp(model_path.stat().st_mtime)
    info = {
        "saved": True,
        "mtime": mtime.strftime("%Y-%m-%d %H:%M:%S"),
        "size_kb": model_path.stat().st_size // 1024,
    }

    if champ_path.exists():
        try:
            with open(champ_path) as f:
                champ = json.load(f)
            info["champion_return"] = champ.get("best_return", "N/A")
        except Exception:
            pass

    return info


def restart_training():
    """Restart training if it died."""
    log("Restarting training...")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(WORK_DIR / "src")
    env["TRAINING_MODE"] = "1"

    # Use best reward params if available
    best_cfg = WORK_DIR / "logs/experiments/best_config.yaml"
    if best_cfg.exists():
        import yaml
        with open(best_cfg) as f:
            cfg = yaml.safe_load(f)
        env["REWARD_PARAMS"] = json.dumps(cfg.get("reward_params", {}))
        log(f"Using best config: {cfg.get('name', 'unknown')}")

    ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_out = open(WORK_DIR / f"logs/training/auto_restart_{ts_str}.log", "w")
    p = subprocess.Popen(
        ["python3", "auto_12h_train.py"],
        env=env, cwd=WORK_DIR,
        stdout=log_out, stderr=subprocess.STDOUT,
    )
    log(f"Training restarted, PID={p.pid}")


def print_status(train: dict, paper: dict, model: dict, exps: dict):
    log("\n" + "="*60)
    log("MASTER MONITOR STATUS")
    log("="*60)

    # Training
    train_status = "ALIVE" if train["alive"] else "DEAD"
    log(f"Training:     {train_status} | PIDs: {train['pids']}")
    log(f"  Win Rate:   {train['win_rate']}")
    log(f"  Trade WR:   {train['trade_win_rate']}")
    log(f"  Return:     {train['mean_return']}")
    log(f"  Round/Phase:{train['iteration']} / {train['phase']}")

    # Paper trading
    paper_status = "ALIVE" if paper["alive"] else "DEAD"
    log(f"Paper Trade:  {paper_status} | PIDs: {paper['pids']}")
    log(f"  Ticks:      {paper['ticks']}")
    if paper["errors"] > 0:
        log(f"  ERRORS:     {paper['errors']} error lines in log!")

    # Model
    if model["saved"]:
        log(f"Model:        SAVED ({model['mtime']}, {model['size_kb']}KB)")
        if "champion_return" in model:
            log(f"  Best Return:{model['champion_return']}")
    else:
        log("Model:        NOT YET SAVED")

    # Experiments
    if exps["n_completed"] > 0:
        log(f"Experiments:  {exps['n_completed']} completed")
        if exps["results"]:
            best = max(exps["results"], key=lambda r: r.get("trade_win_rate", -1))
            log(f"  Best so far:{best['name']} "
                f"(Trade WR={best.get('trade_win_rate', -1)*100:.1f}%)")

    log("="*60)


def generate_final_report():
    """Write final report to session log."""
    train = check_training()
    paper = check_paper_trading()
    model = check_model()
    exps = check_experiments()

    session_dir = Path("/home/hp17/Tradingbot/BITCOIN4Traders-DE-DOKU/sessions")
    if session_dir.exists():
        import subprocess as sp
        files = sorted(session_dir.glob("session_*.md"), key=lambda p: p.stat().st_mtime)
        if files:
            session_file = files[-1]
            report = f"""
## Master Monitor Final Report — {ts()}

### Training
- Status: {"ALIVE" if train["alive"] else "DEAD"}
- Win Rate: {train['win_rate']}
- Per-Trade Win Rate: {train['trade_win_rate']}
- Mean Return: {train['mean_return']}

### Paper Trading
- Status: {"ALIVE" if paper["alive"] else "DEAD"}
- Ticks: {paper['ticks']}
- Errors: {paper.get('errors', 0)}

### Model
- Saved: {model['saved']}
- Timestamp: {model.get('mtime', 'N/A')}
- Best Return: {model.get('champion_return', 'N/A')}

### Experiments
- Completed: {exps.get('n_completed', 0)}
"""
            with open(session_file, "a") as f:
                f.write(report)
            log(f"Final report written to {session_file}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true", help="Single check then exit")
    parser.add_argument("--summary", action="store_true", help="Write final report")
    args = parser.parse_args()

    log("Master Monitor started")

    if args.summary:
        generate_final_report()
        return

    while True:
        train = check_training()
        paper = check_paper_trading()
        model = check_model()
        exps = check_experiments()

        print_status(train, paper, model, exps)

        # Auto-restart training if dead and not in experiment mode
        if not train["alive"] and not exps["running"]:
            log("WARNING: Training not running. Auto-restarting...")
            restart_training()

        if args.once:
            break

        log(f"Next check in {CHECK_INTERVAL//60} minutes...")
        time.sleep(CHECK_INTERVAL)


if __name__ == "__main__":
    main()
