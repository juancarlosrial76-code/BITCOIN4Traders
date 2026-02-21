#!/usr/bin/env python3
"""
12-Hour automated training with monitoring and automatic error recovery.

Differences from auto_train.py:
- Longer runtime (12h vs 8h)
- Targets the tr2win/complete_trading_system working directory
- Runs fix_common_errors() at startup and after each failure
"""

import subprocess
import time
import sys
import os
from datetime import datetime
from pathlib import Path
import signal

MAX_RUNTIME = 12 * 60 * 60  # 12 Stunden
CHECK_INTERVAL = 300  # Alle 5 Minuten
LOG_FILE = Path("logs/training/12h_auto.log")
ERROR_FILE = Path("logs/training/12h_errors.log")

# Target working directory for the tr2win system
WORK_DIR = Path("/home/hp17/Tradingbot/tr2win/complete_trading_system")
os.chdir(WORK_DIR)


def log(msg):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {msg}"
    print(line)
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


def log_error(msg):
    with open(ERROR_FILE, "a") as f:
        f.write(f"[{datetime.now()}] {msg}\n")


def get_best_return():
    """Parse all 12h training log files and return the best observed return."""
    log_dir = WORK_DIR / "logs/training"
    log_files = sorted(log_dir.glob("12h_*.log"), key=lambda x: x.stat().st_mtime)

    best = -999
    for lf in log_files:
        try:
            with open(lf) as f:
                for line in f:
                    if "Mean Return:" in line and "%" in line:
                        try:
                            val = float(
                                line.split("Mean Return:")[1].split("%")[0].strip()
                            )
                            if val > best:
                                best = val
                        except:
                            pass
        except:
            pass
    return best


def fix_common_errors():
    """Automatically fix known configuration issues that block training."""
    log("🔧 Checking for known errors...")

    # Check and patch the environment config if values are too restrictive
    config_file = WORK_DIR / "config/environment/realistic_env.yaml"
    if config_file.exists():
        with open(config_file) as f:
            content = f.read()

        # Relax max_position_size if it is too conservative
        if "max_position_size: 0.10" in content:
            content = content.replace(
                "max_position_size: 0.10", "max_position_size: 0.30"
            )
            log("  → max_position set to 30%")

        # Relax max_drawdown to give the agent more learning room
        if "max_drawdown: 0.70" in content:
            content = content.replace("max_drawdown: 0.70", "max_drawdown: 0.80")
            log("  → max_drawdown set to 80%")

        with open(config_file, "w") as f:
            f.write(content)

    log("✅ Error check complete")


def run_training():
    """Launch a single training run as a subprocess and return (success, stdout, stderr)."""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(WORK_DIR / "src")

    cmd = ["python", "train.py", "--device", "cpu", "--iterations", "100"]

    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=600, env=env, cwd=WORK_DIR
    )

    return result.returncode == 0, result.stdout, result.stderr


def main():
    log("=" * 60)
    log("🚀 12-HOUR TRAINING STARTED")
    log("=" * 60)

    start_time = time.time()
    iteration = 0
    last_best = get_best_return()

    # Fix any known issues before the first run
    fix_common_errors()

    while time.time() - start_time < MAX_RUNTIME:
        iteration += 1
        elapsed = time.time() - start_time
        remaining = MAX_RUNTIME - elapsed

        log(f"\n{'=' * 40}")
        log(f"Round {iteration} | Remaining: {remaining / 3600:.1f}h")
        log(f"{'=' * 40}")

        try:
            success, stdout, stderr = run_training()

            if not success:
                log_error(f"Training failed: {stderr[:500]}")
                fix_common_errors()
                time.sleep(30)
                continue

            # Check for improvement
            current_best = get_best_return()
            log(f"📊 Current: Best={current_best:.2f}%")

            if current_best > last_best + 1.0:
                log(f"✅ PROGRESS! {last_best:.2f}% → {current_best:.2f}%")
                last_best = current_best
            else:
                log(f"⏳ No improvement")

        except subprocess.TimeoutExpired:
            log_error("Timeout - restarting")
            fix_common_errors()
        except Exception as e:
            log_error(f"Exception: {e}")
            fix_common_errors()

        # Periodic checkpoint reminder every 10 rounds
        if iteration % 10 == 0:
            log(f"💾 Checkpoint reminder at round {iteration}")

    # Final summary
    final_best = get_best_return()
    log("\n" + "=" * 60)
    log("🏁 12-HOUR TRAINING COMPLETE")
    log(f"Final Best Return: {final_best:.2f}%")
    log("=" * 60)


if __name__ == "__main__":
    main()
