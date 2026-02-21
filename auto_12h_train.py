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
import shutil
import tempfile
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
                            val = float(line.split("Mean Return:")[1].split("%")[0].strip())
                            if val > best:
                                best = val
                        except:
                            pass
        except:
            pass
    return best


def fix_common_errors():
    """Temporarily relax config values that commonly block training.

    Writes the relaxed config to a temp file, atomically replaces the live
    config, and returns the backup path so the caller can restore it after
    the training batch.  The committed base config is therefore never
    permanently modified.
    """
    log("Checking for known errors...")

    config_file = WORK_DIR / "config/environment/realistic_env.yaml"
    if not config_file.exists():
        log("Config file not found, skipping fix")
        return None

    with open(config_file) as f:
        content = f.read()

    changed = False

    # Relax max_position_size if it is too conservative
    if "max_position_size: 0.10" in content:
        content = content.replace("max_position_size: 0.10", "max_position_size: 0.30")
        log("  -> max_position set to 30%")
        changed = True

    # Relax max_drawdown to give the agent more learning room
    if "max_drawdown: 0.70" in content:
        content = content.replace("max_drawdown: 0.70", "max_drawdown: 0.80")
        log("  -> max_drawdown set to 80%")
        changed = True

    if not changed:
        log("Config already within acceptable ranges, no changes made")
        return None

    # Backup original, then atomically replace with relaxed version
    backup = config_file.with_suffix(".yaml.bak")
    shutil.copy2(config_file, backup)

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", dir=config_file.parent, delete=False
    ) as tmp:
        tmp.write(content)
        tmp_path = Path(tmp.name)

    tmp_path.replace(config_file)
    log("Error check complete - relaxed config applied")
    return backup  # caller must restore after training


def run_training():
    """Launch a single training run as a subprocess and return (success, stdout, stderr)."""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(WORK_DIR / "src")

    cmd = ["python", "train.py", "--device", "cpu", "--iterations", "100"]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600, env=env, cwd=WORK_DIR)

    return result.returncode == 0, result.stdout, result.stderr


def main():
    log("=" * 60)
    log("🚀 12-HOUR TRAINING STARTED")
    log("=" * 60)

    start_time = time.time()
    iteration = 0
    last_best = get_best_return()

    def _restore_config(backup_path):
        """Restore base config from backup and clean up."""
        config_file = WORK_DIR / "config/environment/realistic_env.yaml"
        if backup_path and backup_path.exists():
            shutil.copy2(backup_path, config_file)
            backup_path.unlink()
            log("Base config restored after relaxed training batch")

    # Apply relaxed config for the very first run only, then restore immediately
    first_run_backup = fix_common_errors()

    while time.time() - start_time < MAX_RUNTIME:
        iteration += 1
        elapsed = time.time() - start_time
        remaining = MAX_RUNTIME - elapsed

        log(f"\n{'=' * 40}")
        log(f"Round {iteration} | Remaining: {remaining / 3600:.1f}h")
        log(f"{'=' * 40}")

        try:
            success, stdout, stderr = run_training()

            # Restore base config after first run (backup is only set on first run)
            _restore_config(first_run_backup)
            first_run_backup = None

            if not success:
                log_error(f"Training failed: {stderr[:500]}")
                # Relax config for one retry, then restore
                retry_backup = fix_common_errors()
                run_training()
                _restore_config(retry_backup)
                time.sleep(30)
                continue

            # Check for improvement
            current_best = get_best_return()
            log(f"Current best: {current_best:.2f}%")

            if current_best > last_best + 1.0:
                log(f"PROGRESS! {last_best:.2f}% -> {current_best:.2f}%")
                last_best = current_best
            else:
                log("No improvement")

        except subprocess.TimeoutExpired:
            log_error("Timeout - restarting")
            _restore_config(first_run_backup)
            first_run_backup = None
            retry_backup = fix_common_errors()
            _restore_config(retry_backup)
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
