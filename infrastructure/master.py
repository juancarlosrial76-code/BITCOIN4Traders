#!/usr/bin/env python3
"""
Master Orchestrator - BITCOIN4Traders Zero-Cost Infrastructure
================================================================
The central conductor. Coordinates all 4 pillars:

  Pillar 1: Linux-PC     = Brain    (local training, 24/7)
  Pillar 2: Google Drive = Memory   (models, logs, signals)
  Pillar 3: Colab GPU    = Muscle   (heavy training, free)
  Pillar 4: GitHub       = Backup   (versioning, recovery)

Tasks (cyclic):
  Every 60 min:  Drive-Sync + Colab-Heartbeat check
  Every 6 hrs:   Upload logs + GitHub-Push
  On error:      Telegram alert + restart signal
  Continuously:  Monitor local training

Start:
  python3 infrastructure/master.py

As service (persistent):
  sudo systemctl start bitcoin4traders
"""

import sys
import os
import time
import json
import signal
import threading
from datetime import datetime, timedelta
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "infrastructure" / "drive"))
sys.path.insert(0, str(BASE_DIR / "infrastructure" / "colab"))
sys.path.insert(0, str(BASE_DIR / "infrastructure" / "monitor"))

# Graceful Shutdown
_shutdown = threading.Event()


def _handle_signal(sig, frame):
    print(f"\n[MASTER] Signal {sig} received - shutting down...")
    _shutdown.set()


signal.signal(signal.SIGTERM, _handle_signal)
signal.signal(signal.SIGINT, _handle_signal)


# ─── Schedule ────────────────────────────────────────────────────────────────


class Scheduler:
    """Simple scheduler without external dependencies."""

    def __init__(self):
        self._tasks: dict[str, dict] = {}

    def every(self, seconds: int, name: str, func, *args, **kwargs):
        self._tasks[name] = {
            "interval": seconds,
            "func": func,
            "args": args,
            "kwargs": kwargs,
            "last_run": None,
            "next_run": datetime.now(),  # Execute immediately on start
        }

    def run_pending(self):
        now = datetime.now()
        for name, task in self._tasks.items():
            if now >= task["next_run"]:
                try:
                    print(f"\n[{now.strftime('%H:%M:%S')}] Task: {name}")
                    task["func"](*task["args"], **task["kwargs"])
                    task["last_run"] = now
                    task["next_run"] = now + timedelta(seconds=task["interval"])
                except Exception as e:
                    print(f"  [ERROR] Task '{name}': {e}")
                    try:
                        from alert_manager import warn

                        warn(f"Task error '{name}': {e}", source="master")
                    except Exception:
                        pass


# ─── Tasks ───────────────────────────────────────────────────────────────────


def task_drive_sync():
    """Drive-Sync: Champion + Heartbeat."""
    try:
        from drive_manager import run_sync

        run_sync("up")  # Linux-PC -> Drive
    except ImportError:
        print(
            "  [SKIP] drive_manager not installed - pip install google-api-python-client"
        )
    except Exception as e:
        _alert(f"Drive-Sync failed: {e}", "warning")


def task_colab_watchdog():
    """Check Colab status."""
    try:
        from colab_watchdog import run_watchdog
        from alert_manager import critical

        run_watchdog(alert_callback=lambda msg: critical(msg, source="colab_watchdog"))
    except ImportError:
        print("  [SKIP] colab_watchdog - dependencies missing")
    except Exception as e:
        _alert(f"Colab-Watchdog error: {e}", "warning")


def task_github_backup():
    """Push champion to GitHub via Git."""
    import subprocess

    try:
        result = subprocess.run(
            [str(BASE_DIR / "sync_champion.sh")],
            capture_output=True,
            text=True,
            cwd=str(BASE_DIR),
            timeout=120,
        )
        if result.returncode == 0:
            print("  [GITHUB] Sync successful")
        else:
            print(f"  [GITHUB] Sync error: {result.stderr[:200]}")
    except subprocess.TimeoutExpired:
        _alert("GitHub-Sync timeout", "warning")
    except Exception as e:
        _alert(f"GitHub-Sync error: {e}", "warning")


def task_local_training_check():
    """Check whether local training is still running."""
    import subprocess

    try:
        # Check if training process is running
        result = subprocess.run(
            ["pgrep", "-f", "auto_12h_train.py"], capture_output=True
        )
        if result.returncode != 0:
            _alert("Local training has stopped!", "warning")
            # Optional: automatically restart training
            # subprocess.Popen(["python3", str(BASE_DIR / "auto_12h_train.py")])
    except Exception as e:
        print(f"  [CHECK] Training check error: {e}")


def _start_listener():
    """Start the error listener in a background thread."""
    try:
        listener_path = BASE_DIR / "infrastructure" / "monitor" / "listener.py"
        if not listener_path.exists():
            print("  [LISTENER] listener.py not found")
            return

        import importlib.util

        spec = importlib.util.spec_from_file_location("listener", listener_path)
        module = importlib.util.module_from_spec(spec)  # type: ignore
        spec.loader.exec_module(module)  # type: ignore

        t = threading.Thread(target=module.run, daemon=True, name="error_listener")
        t.start()
        print("  [LISTENER] Error receiver started (background)")
    except ImportError:
        print("  [LISTENER] Flask missing: pip install flask")
    except Exception as e:
        print(f"  [LISTENER] Start failed: {e}")


def task_status_report():
    """Daily status report."""
    status = {
        "timestamp": datetime.now().isoformat(),
        "uptime": _get_uptime(),
        "champion": _read_champion_meta(),
        "disk_free_gb": _get_disk_free(),
    }
    status_file = BASE_DIR / "logs" / "infrastructure_status.json"
    status_file.parent.mkdir(parents=True, exist_ok=True)
    with open(status_file, "w") as f:
        json.dump(status, f, indent=2)

    report = (
        f"Daily Status Report\n"
        f"Uptime: {status['uptime']}\n"
        f"Disk free: {status['disk_free_gb']:.1f} GB\n"
        f"Champion: {status['champion'].get('name', 'unknown')}\n"
        f"Sharpe: {status['champion'].get('sharpe', '?')}"
    )
    _alert(report, "info")


# ─── Helper functions ────────────────────────────────────────────────────────


def _alert(message: str, severity: str = "info"):
    try:
        from alert_manager import alert

        alert(message, severity, source="master")  # type: ignore
    except Exception:
        print(f"  [{severity.upper()}] {message}")


def _get_uptime() -> str:
    try:
        with open("/proc/uptime") as f:
            seconds = float(f.read().split()[0])
        hours = int(seconds // 3600)
        return f"{hours}h"
    except Exception:
        return "unknown"


def _get_disk_free() -> float:
    import shutil

    try:
        usage = shutil.disk_usage(str(BASE_DIR))
        return usage.free / (1024**3)
    except Exception:
        return 0.0


def _read_champion_meta() -> dict:
    meta_file = BASE_DIR / "data" / "cache" / "multiverse_champion_meta.json"
    if meta_file.exists():
        try:
            with open(meta_file) as f:
                return json.load(f)
        except Exception:
            pass
    return {}


# ─── Main program ──────────────────────────────────────────────────────────


def main():
    print("=" * 60)
    print("  BITCOIN4Traders - Master Orchestrator")
    print(f"  Started: {datetime.now().strftime('%d.%m.%Y %H:%M:%S')}")
    print("=" * 60)

    _alert("Master Orchestrator started", "info")

    scheduler = Scheduler()

    # Configure schedule
    scheduler.every(
        seconds=60 * 60,  # Every 60 minutes
        name="drive_sync",
        func=task_drive_sync,
    )
    scheduler.every(
        seconds=60 * 60,  # Every 60 minutes
        name="colab_watchdog",
        func=task_colab_watchdog,
    )
    scheduler.every(
        seconds=60 * 15,  # Every 15 minutes
        name="training_check",
        func=task_local_training_check,
    )
    scheduler.every(
        seconds=60 * 60 * 6,  # Every 6 hours
        name="github_backup",
        func=task_github_backup,
    )
    scheduler.every(
        seconds=60 * 60 * 24,  # Once per day
        name="status_report",
        func=task_status_report,
    )

    print("\nSchedule:")
    print("  Every 15 min : Training check")
    print("  Every 60 min : Drive-Sync + Colab-Watchdog")
    print("  Every 6 hrs  : GitHub-Backup")
    print("  Every 24 hrs : Status report")
    print("\nCtrl+C to exit\n")

    # Start error listener in background thread
    _start_listener()

    # Main loop
    while not _shutdown.is_set():
        scheduler.run_pending()
        time.sleep(30)  # Check for pending tasks every 30 seconds

    _alert("Master Orchestrator stopped", "info")
    print("[MASTER] Clean shutdown.")


if __name__ == "__main__":
    main()
