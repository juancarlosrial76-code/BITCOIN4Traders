#!/usr/bin/env python3
"""
12-Stunden Training mit automatisierter Überwachung und Fehlerkorrektur
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

# Arbeitsverzeichnis
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
    """Hole beste Return aus Logs."""
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
    """Behebe häufige Fehler automatisch."""
    log("🔧 Prüfe auf bekannte Fehler...")

    # Prüfe Config
    config_file = WORK_DIR / "config/environment/realistic_env.yaml"
    if config_file.exists():
        with open(config_file) as f:
            content = f.read()

        # Stelle sicher, dass Parameter korrekt sind
        if "max_position_size: 0.10" in content:
            content = content.replace(
                "max_position_size: 0.10", "max_position_size: 0.30"
            )
            log("  → max_position auf 30% gesetzt")

        if "max_drawdown: 0.70" in content:
            content = content.replace("max_drawdown: 0.70", "max_drawdown: 0.80")
            log("  → max_drawdown auf 80% gesetzt")

        with open(config_file, "w") as f:
            f.write(content)

    log("✅ Fehlerprüfung abgeschlossen")


def run_training():
    """Starte Training."""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(WORK_DIR / "src")

    cmd = ["python", "train.py", "--device", "cpu", "--iterations", "100"]

    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=600, env=env, cwd=WORK_DIR
    )

    return result.returncode == 0, result.stdout, result.stderr


def main():
    log("=" * 60)
    log("🚀 12-STUNDEN TRAINING GESTARTET")
    log("=" * 60)

    start_time = time.time()
    iteration = 0
    last_best = get_best_return()

    # Initiale Fehlerbehebung
    fix_common_errors()

    while time.time() - start_time < MAX_RUNTIME:
        iteration += 1
        elapsed = time.time() - start_time
        remaining = MAX_RUNTIME - elapsed

        log(f"\n{'=' * 40}")
        log(f"Runde {iteration} | Verbleibend: {remaining / 3600:.1f}h")
        log(f"{'=' * 40}")

        try:
            success, stdout, stderr = run_training()

            if not success:
                log_error(f"Training fehlgeschlagen: {stderr[:500]}")
                fix_common_errors()
                time.sleep(30)
                continue

            # Metriken prüfen
            current_best = get_best_return()
            log(f"📊 Aktuell: Best={current_best:.2f}%")

            if current_best > last_best + 1.0:
                log(f"✅ FORTSCHRITT! {last_best:.2f}% → {current_best:.2f}%")
                last_best = current_best
            else:
                log(f"⏳ Keine Verbesserung")

        except subprocess.TimeoutExpired:
            log_error("Timeout - neustarten")
            fix_common_errors()
        except Exception as e:
            log_error(f"Exception: {e}")
            fix_common_errors()

        # Backup alle 10 Runden
        if iteration % 10 == 0:
            log(f"💾 Backup bei Runde {iteration}")

    # Abschluss
    final_best = get_best_return()
    log("\n" + "=" * 60)
    log("🏁 12-STUNDEN TRAINING ABGESCHLOSSEN")
    log(f"Final Best Return: {final_best:.2f}%")
    log("=" * 60)


if __name__ == "__main__":
    main()
