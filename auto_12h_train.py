#!/usr/bin/env python3
"""
12-Hour automated training with monitoring and automatic error recovery.

Differences from auto_train.py:
- Longer runtime (12h vs 8h)
- Targets the BITCOIN4Traders working directory (Local Master)
- Runs fix_common_errors() at startup and after each failure
- Designed for Linux-PC Local Master node (no time limits)
"""

import subprocess
import time
import sys
import os
from datetime import datetime
from pathlib import Path
import signal

MAX_RUNTIME = 24 * 60 * 60  # 24 hours
CHECK_INTERVAL = 300  # Every 5 minutes

# Automatically determine the directory of this file as WORK_DIR
# This makes the script work regardless of the calling directory
WORK_DIR = Path(__file__).resolve().parent
os.chdir(WORK_DIR)

LOG_FILE = WORK_DIR / "logs/training/12h_auto.log"
ERROR_FILE = WORK_DIR / "logs/training/12h_errors.log"


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
    """Parse training log files and return the best observed return."""
    log_dir = WORK_DIR / "logs/training"
    # Match both auto_12h_*.log and train_*.log (evaluator writes to train_*.log)
    log_files = sorted(
        list(log_dir.glob("auto_12h_*.log")) + list(log_dir.glob("train_*.log")),
        key=lambda x: x.stat().st_mtime,
    )[-20:]  # Only check the 20 most recent files

    best = -999
    for lf in log_files:
        try:
            with open(lf) as f:
                for line in f:
                    # Match both "Mean Return:" and "Weighted Return:"
                    if ("Mean Return:" in line or "Weighted Return:" in line) and "%" in line:
                        try:
                            key = "Mean Return:" if "Mean Return:" in line else "Weighted Return:"
                            val = float(line.split(key)[1].split("%")[0].strip())
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


def parse_win_rate(stdout: str) -> float:
    """Extract the most recent win rate from training stdout or log files."""
    last_rate = -1.0

    # First try stdout (may be empty if loguru writes to file only)
    for line in stdout.splitlines():
        if "Win Rate:" in line or "Trade WR" in line:
            try:
                if "Trade WR" in line:
                    val = float(line.split("Trade WR")[1].split("%")[0].strip())
                else:
                    val = float(line.split("Win Rate:")[1].split("%")[0].strip())
                last_rate = val / 100.0
            except Exception:
                pass

    # Fallback: read from the most recent train_*.log (loguru writes there)
    if last_rate < 0:
        log_dir = WORK_DIR / "logs/training"
        recent_logs = sorted(log_dir.glob("train_*.log"), key=lambda x: x.stat().st_mtime)
        if recent_logs:
            try:
                with open(recent_logs[-1]) as f:
                    for line in f:
                        if "Win Rate:" in line or "Trade WR" in line:
                            try:
                                if "Trade WR" in line:
                                    val = float(line.split("Trade WR")[1].split("%")[0].strip())
                                else:
                                    val = float(line.split("Win Rate:")[1].split("%")[0].strip())
                                last_rate = val / 100.0
                            except Exception:
                                pass
            except Exception:
                pass

    return last_rate


def run_training(training_mode: bool = True):
    """Launch a single training run as a subprocess and return (success, stdout, stderr)."""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(WORK_DIR / "src")
    env["TRAINING_MODE"] = "1" if training_mode else "0"

    cmd = ["python3", "train.py", "--device", "cpu", "--iterations", "10"]

    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=7200, env=env, cwd=WORK_DIR
    )

    return result.returncode == 0, result.stdout, result.stderr


def main():
    log("=" * 60)
    log("🚀 12-HOUR TRAINING STARTED")
    log("=" * 60)

    start_time = time.time()
    iteration = 0
    last_best = get_best_return()
    last_win_rate = -1.0  # -1.0 = never measured (avoids false NOTSTOP on parse failure)
    # Phase 1: training_mode=True (Kelly bypassed, full exploration)
    # Phase 2: training_mode=False (adaptive Kelly re-enabled), triggered by win_rate > 18%
    training_mode = True
    phase = 1

    # Fix any known issues before the first run
    fix_common_errors()

    while time.time() - start_time < MAX_RUNTIME:
        iteration += 1
        elapsed = time.time() - start_time
        remaining = MAX_RUNTIME - elapsed

        log(f"\n{'=' * 40}")
        log(f"Round {iteration} | Phase {phase} | Remaining: {remaining / 3600:.1f}h")
        log(f"{'=' * 40}")

        try:
            success, stdout, stderr = run_training(training_mode=training_mode)

            if not success:
                log_error(f"Training failed: {stderr[:500]}")
                fix_common_errors()
                time.sleep(30)
                continue

            # Parse win rate from this run's output
            run_win_rate = parse_win_rate(stdout)
            if run_win_rate >= 0:
                last_win_rate = run_win_rate
                log(f"📈 Win Rate: {last_win_rate:.1%}")

            # Emergency stop: Kelly still at 0% after 10 rounds despite training mode
            # Only trigger if we actually measured a win rate (last_win_rate >= 0)
            if iteration > 10 and 0 <= last_win_rate < 0.005 and phase == 1:
                log("⛔ NOTSTOP: Win Rate bleibt bei 0% trotz Training-Mode - bitte Win-Rate-Berechnung prüfen")
                break

            # Phase transition: switch to production Kelly once win rate is stable
            if phase == 1 and last_win_rate > 0.18:  # last_win_rate is -1 if never measured (won't trigger)
                phase = 2
                training_mode = False
                log(f"🎯 PHASE 2: Win-Rate {last_win_rate:.1%} > 18% - aktiviere adaptives Kelly")

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
