#!/usr/bin/env python3
"""
Colab Watchdog - Intelligent Keepalive & Auto-Recovery
=========================================================
Problem: Colab kills sessions after 90 min of INACTIVITY (not time).
Solution: Trigger a real compute task every 60 min, not just a ping.

Strategy:
  1. Write heartbeat file to Google Drive
  2. Colab reads heartbeat and knows: "Linux-PC is still alive"
  3. Colab writes status file -> Linux-PC knows: "Colab is still alive"
  4. If Colab status > 120 min old -> send alert + restart signal

Cost: 0 EUR (only Drive API, no Selenium/browser needed)
"""

import json
import time
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Adjust path
BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE_DIR / "infrastructure" / "drive"))

from drive_manager import get_drive_service, load_drive_config, find_file_in_folder

# ─── Configuration ────────────────────────────────────────────────────────────

CHECK_INTERVAL_SEC = 60 * 60  # Check every hour
COLAB_TIMEOUT_SEC = 60 * 90  # 90 min without status = Colab dead
HEARTBEAT_FILE = "/tmp/linux_heartbeat.json"
STATUS_FILE = "/tmp/colab_status.json"

# ─── Read Colab status ────────────────────────────────────────────────────────


def read_colab_status(service, config: dict) -> dict | None:
    """
    Reads the status file that Colab writes to Drive.
    Colab notebook must periodically upload status for this to work.
    """
    folder_id = config.get("champion_folder_id", "")
    if not folder_id:
        return None

    file_id = find_file_in_folder(service, folder_id, "colab_status.json")
    if not file_id:
        return None

    from googleapiclient.http import MediaIoBaseDownload
    import io

    try:
        request = service.files().get_media(fileId=file_id)
        buf = io.BytesIO()
        downloader = MediaIoBaseDownload(buf, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()
        buf.seek(0)
        return json.loads(buf.read().decode())
    except Exception as e:
        print(f"  [WATCHDOG] Reading Colab status failed: {e}")
        return None


def is_colab_alive(status: dict | None) -> bool:
    """Checks whether Colab status is recent enough."""
    if not status:
        return False

    try:
        last_seen = datetime.fromisoformat(status["timestamp"])
        age = (datetime.now() - last_seen).total_seconds()
        return age < COLAB_TIMEOUT_SEC
    except Exception:
        return False


# ─── Restart signal ──────────────────────────────────────────────────────────


def send_restart_signal(service, config: dict):
    """
    Writes a 'restart_requested.json' to Drive.
    The Colab notebook checks this file and restarts training.
    """
    folder_id = config.get("champion_folder_id", "")
    if not folder_id:
        return

    signal_data = {
        "action": "restart_training",
        "reason": "colab_timeout",
        "requested_at": datetime.now().isoformat(),
        "requested_by": os.uname().nodename,
    }

    signal_file = Path("/tmp/restart_requested.json")
    with open(signal_file, "w") as f:
        json.dump(signal_data, f)

    from drive_manager import upload_file

    upload_file(service, signal_file, folder_id, "Restart signal from Linux-PC")
    print(f"  [WATCHDOG] Restart signal sent: {datetime.now().isoformat()}")


def clear_restart_signal(service, config: dict):
    """Deletes the restart signal after Colab has processed it."""
    folder_id = config.get("champion_folder_id", "")
    if not folder_id:
        return

    from drive_manager import find_file_in_folder

    file_id = find_file_in_folder(service, folder_id, "restart_requested.json")
    if file_id:
        service.files().delete(fileId=file_id).execute()
        print("  [WATCHDOG] Restart signal deleted")


# ─── Write heartbeat ──────────────────────────────────────────────────────────


def write_linux_heartbeat(service, config: dict):
    """
    Writes heartbeat from Linux-PC to Drive.
    Colab reads this file -> knows that Linux-PC is still there.
    """
    folder_id = config.get("champion_folder_id", "")
    if not folder_id:
        return

    heartbeat = {
        "timestamp": datetime.now().isoformat(),
        "host": os.uname().nodename,
        "next_check": (
            datetime.now() + timedelta(seconds=CHECK_INTERVAL_SEC)
        ).isoformat(),
        "status": "monitoring",
    }

    hb_file = Path(HEARTBEAT_FILE)
    with open(hb_file, "w") as f:
        json.dump(heartbeat, f)

    from drive_manager import upload_file

    upload_file(service, hb_file, folder_id, "Linux-PC Heartbeat")


# ─── Main loop ────────────────────────────────────────────────────────────────


def run_watchdog(alert_callback=None):
    """
    Main watchdog loop.
    alert_callback(message) is called on problems (e.g. Telegram).
    """
    print(f"\n[WATCHDOG] Started - checking every {CHECK_INTERVAL_SEC // 60} minutes")
    print(f"[WATCHDOG] Colab timeout: {COLAB_TIMEOUT_SEC // 60} minutes")

    consecutive_failures = 0
    restart_sent = False

    while True:
        try:
            service = get_drive_service()
            config = load_drive_config()

            # 1. Write own heartbeat
            write_linux_heartbeat(service, config)

            # 2. Check Colab status
            status = read_colab_status(service, config)
            colab_alive = is_colab_alive(status)

            if colab_alive:
                consecutive_failures = 0
                restart_sent = False

                training_step = status.get("training_step", "?")
                reward = status.get("last_reward", "?")
                print(f"[WATCHDOG] Colab OK | Step: {training_step} | Reward: {reward}")

                # Delete restart signal if still present
                clear_restart_signal(service, config)

            else:
                consecutive_failures += 1
                age_min = "?"
                if status:
                    try:
                        last = datetime.fromisoformat(status["timestamp"])
                        age_min = int((datetime.now() - last).total_seconds() / 60)
                    except Exception:
                        pass

                print(
                    f"[WATCHDOG] Colab INACTIVE (age: {age_min} min, "
                    f"attempts: {consecutive_failures})"
                )

                # On first failure: send restart signal
                if consecutive_failures == 1 and not restart_sent:
                    send_restart_signal(service, config)
                    restart_sent = True

                    msg = (
                        f"Colab session inactive ({age_min} min)!\n"
                        f"Restart signal has been sent.\n"
                        f"Please check Colab notebook manually if "
                        f"no restart occurs."
                    )
                    if alert_callback:
                        alert_callback(msg)

                # After 3 failed attempts: strong alert
                if consecutive_failures >= 3:
                    msg = (
                        f"CRITICAL: Colab has been dead for {age_min} min!\n"
                        f"Manual intervention required.\n"
                        f"URL: https://colab.research.google.com"
                    )
                    if alert_callback:
                        alert_callback(msg)
                    consecutive_failures = 0  # Reset to avoid spam

        except Exception as e:
            print(f"[WATCHDOG] Error: {e}")
            if alert_callback:
                alert_callback(f"Watchdog error: {e}")

        # Next check
        next_check = datetime.now() + timedelta(seconds=CHECK_INTERVAL_SEC)
        print(f"[WATCHDOG] Next check: {next_check.strftime('%H:%M:%S')}\n")
        time.sleep(CHECK_INTERVAL_SEC)


if __name__ == "__main__":
    run_watchdog()
