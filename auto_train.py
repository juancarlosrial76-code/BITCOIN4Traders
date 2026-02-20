#!/usr/bin/env python3
"""
8-Stunden Training mit automatischer Überwachung und Selbstoptimierung
- Alle 5 Min: Fortschritt prüfen
- Nach 1h ohne Verbesserung: Parameter anpassen
"""

import subprocess
import time
import sys
import json
from datetime import datetime
from pathlib import Path

MAX_RUNTIME = 8 * 60 * 60  # 8 Stunden
CHECK_INTERVAL = 300  # Alle 5 Minuten
NO_PROGRESS_LIMIT = 12  # 1 Stunde (12 x 5 Min)

LOG_FILE = Path("logs/training/auto_training.log")
ERROR_FILE = Path("logs/training/auto_errors.log")


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


def get_latest_metrics():
    """Hole neueste Trainingsmetriken."""
    log_dir = Path("logs/training")
    log_files = sorted(log_dir.glob("train_*.log"), key=lambda x: x.stat().st_mtime)

    if not log_files:
        return None

    latest = log_files[-1]
    returns = []
    lengths = []

    with open(latest) as f:
        for line in f:
            if "Mean Return:" in line and "%" in line:
                try:
                    val = float(line.split("Mean Return:")[1].split("%")[0].strip())
                    returns.append(val)
                except:
                    pass
            if "Mean Length:" in line:
                try:
                    val = float(line.split("Mean Length:")[1].strip())
                    lengths.append(val)
                except:
                    pass

    if returns:
        return {
            "latest_return": returns[-1],
            "best_return": max(returns),
            "latest_length": lengths[-1] if lengths else 0,
            "iteration_count": len(returns),
        }
    return None


def adjust_parameters():
    """Passe Parameter an wenn kein Fortschritt."""
    log("⚠️ Kein Fortschritt seit 1h - Optimiere Parameter...")

    config_file = Path("config/environment/realistic_env.yaml")

    # Lese aktuelle Config
    with open(config_file) as f:
        content = f.read()

    # Erhöhe max_drawdown für mehr Lernspielraum
    if "max_drawdown: 0.70" in content:
        content = content.replace("max_drawdown: 0.70", "max_drawdown: 0.80")
        log("  → max_drawdown erhöht auf 80%")

    # Reduziere Reward-Penalty für mehr Exploration
    if "weight: -0.5" in content:
        content = content.replace("weight: -0.5", "weight: -0.2")
        log("  → drawdown penalty reduziert")

    if "weight: -0.3" in content:
        content = content.replace("weight: -0.3", "weight: -0.1")
        log("  → transaction cost penalty reduziert")

    # Erhöhe max_position_size
    if "max_position_size: 0.10" in content:
        content = content.replace("max_position_size: 0.10", "max_position_size: 0.20")
        log("  → max_position auf 20% erhöht")

    with open(config_file, "w") as f:
        f.write(content)

    log("✅ Parameter angepasst - Training wird fortgesetzt")


def run_training_batch(iterations=50):
    """Führe Training für eine Batch aus."""
    try:
        result = subprocess.run(
            ["python", "train.py", "--device", "cpu", "--iterations", str(iterations)],
            capture_output=True,
            text=True,
            timeout=600,
            env={"PYTHONPATH": f"{Path.cwd()}/src"},
        )
        return result.returncode == 0
    except Exception as e:
        log_error(f"Training error: {e}")
        return False


def main():
    log("=" * 60)
    log("🚀 START: 8-Stunden Training mit Auto-Überwachung")
    log("=" * 60)

    start_time = time.time()
    no_progress_count = 0
    last_best_return = -999

    # Initiale Metrics
    initial_metrics = get_latest_metrics()
    if initial_metrics:
        last_best_return = initial_metrics.get("best_return", -999)
        log(f"Start-Metriken: Best Return = {last_best_return:.2f}%")

    while time.time() - start_time < MAX_RUNTIME:
        remaining = MAX_RUNTIME - (time.time() - start_time)
        log(
            f"\n--- Verbleibend: {remaining / 3600:.1f}h | No-Progress: {no_progress_count}/{NO_PROGRESS_LIMIT} ---"
        )

        # Training starten
        success = run_training_batch(iterations=30)

        if not success:
            log_error("Training failed, retrying...")
            time.sleep(10)
            continue

        # Metriken prüfen nach jeder Runde
        time.sleep(5)  # Kurze Pause für Log-Schreiben
        metrics = get_latest_metrics()

        if metrics:
            current_best = metrics.get("best_return", -999)
            log(
                f"📊 Aktuell: Best={current_best:.2f}%, Latest={metrics.get('latest_return', 0):.2f}%, Length={metrics.get('latest_length', 0):.0f}"
            )

            # Prüfe Verbesserung
            if current_best > last_best_return + 1.0:  # >1% Verbesserung
                log(f"✅ FORTSCHRITT! {last_best_return:.2f}% → {current_best:.2f}%")
                last_best_return = current_best
                no_progress_count = 0
            else:
                no_progress_count += 1
                log(f"⏳ Kein Fortschritt ({no_progress_count}x)")

                # Nach 1h keine Verbesserung → optimieren
                if no_progress_count >= NO_PROGRESS_LIMIT:
                    adjust_parameters()
                    last_best_return = -999  # Reset für neue Runde
                    no_progress_count = 0

        # Prüfe ob Zeit fast um
        if remaining < 600:  # <10 Min
            break

    # Abschluss
    final_metrics = get_latest_metrics()
    log("\n" + "=" * 60)
    log("🏁 TRAINING ABGESCHLOSSEN")
    log("=" * 60)
    if final_metrics:
        log(f"Final Results:")
        log(f"  - Best Return: {final_metrics.get('best_return', 0):.2f}%")
        log(f"  - Latest Return: {final_metrics.get('latest_return', 0):.2f}%")
        log(f"  - Total Iterations: {final_metrics.get('iteration_count', 0)}")


if __name__ == "__main__":
    main()
