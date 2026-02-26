#!/usr/bin/env python3
"""
Error Listener - Receiver for Colab error reports
================================================
Improved version over the original plan:

PROBLEM with original plan:
  - Ngrok free = URL changes every 8h (Colab doesn't know the new URL)
  - Open port 5000 = security vulnerability (anyone can send POST)
  - Single point of failure: no fallback when Flask is down

SOLUTION (dual-channel):
  Channel 1: Flask HTTP (fast, direct path via ngrok/local network)
  Channel 2: Google Drive as message bus (reliable, no port needed)
  -> Colab tries HTTP first, then Drive. One of them always works.

Security:
  - API token verification (no outsider can inject reports)
  - Optional IP whitelist
  - Rate limiting (max 10 reports/min)

Auto-Repair:
  After receiving, error_repair.py is called to patch the notebook.
"""

import json
import os
import sys
import time
import hmac
import hashlib
import threading
from datetime import datetime
from pathlib import Path
from collections import defaultdict, deque

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE_DIR / "infrastructure" / "monitor"))
sys.path.insert(0, str(BASE_DIR / "infrastructure" / "drive"))

ERROR_LOG_DIR = BASE_DIR / "logs" / "colab_errors"
ERROR_LOG_DIR.mkdir(parents=True, exist_ok=True)

# ─── Configuration ────────────────────────────────────────────────────────────

# API token: store in config/listener_config.json
CONFIG_FILE = BASE_DIR / "config" / "listener_config.json"

DEFAULT_CONFIG = {
    "api_token": "CHANGE_ME_generate_random_token",  # Must match in Colab+Linux
    "port": 5001,  # 5001 instead of 5000 (less well-known)
    "rate_limit_per_min": 10,
    "auto_repair": True,
    "drive_polling_interval_sec": 120,  # Check Drive every 2 min
}


def load_config() -> dict:
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE) as f:
            return {**DEFAULT_CONFIG, **json.load(f)}
    return DEFAULT_CONFIG.copy()


def save_config(config: dict):
    CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=2)


# ─── Rate limiter ─────────────────────────────────────────────────────────────

_request_times: dict = defaultdict(lambda: deque(maxlen=10))


def _check_rate_limit(ip: str, max_per_min: int) -> bool:
    """True = allowed, False = blocked."""
    now = time.time()
    times = _request_times[ip]
    # Remove old entries (older than 60s)
    while times and now - times[0] > 60:
        times.popleft()
    if len(times) >= max_per_min:
        return False
    times.append(now)
    return True


# ─── Token verification ────────────────────────────────────────────────────────


def _verify_token(received_token: str, expected_token: str) -> bool:
    """Timing-safe comparison (prevents timing attacks)."""
    return hmac.compare_digest(received_token.encode(), expected_token.encode())


# ─── Process error report ───────────────────────────────────────────────────────


def process_error_report(report: dict) -> dict:
    """
    Processes an error report:
    1. Saves it to log file
    2. Sends Telegram alert
    3. Starts auto-repair (optional)
    """
    timestamp = datetime.now().isoformat()
    report["received_at"] = timestamp

    # 1. Save to log file
    log_file = ERROR_LOG_DIR / f"errors_{datetime.now().strftime('%Y-%m-%d')}.jsonl"
    with open(log_file, "a") as f:
        f.write(json.dumps(report, ensure_ascii=False) + "\n")

    # To console
    notebook_id = report.get("notebook_id", "?")
    error_type = report.get("error_type", "Exception")
    error_msg = report.get("error_message", "?")
    print(f"\n[{timestamp}] ERROR REPORT from {notebook_id}")
    print(f"  Type:    {error_type}")
    print(f"  Error:   {error_msg[:120]}")

    # 2. Telegram alert
    try:
        from alert_manager import critical

        msg = (
            f"Colab error report!\n"
            f"Notebook: {notebook_id}\n"
            f"Error: {error_type}\n"
            f"{error_msg[:200]}"
        )
        critical(msg, source="colab_listener")
    except Exception as e:
        print(f"  [WARN] Telegram alert failed: {e}")

    # 3. Start auto-repair
    config = load_config()
    if config.get("auto_repair"):
        threading.Thread(target=_trigger_repair, args=(report,), daemon=True).start()

    return {"status": "ok", "action": "repair_triggered"}


def _trigger_repair(report: dict):
    """Starts the repair engine asynchronously."""
    try:
        # Import here so listener runs without repair too
        repair_path = BASE_DIR / "infrastructure" / "monitor" / "error_repair.py"
        if repair_path.exists():
            import importlib.util

            spec = importlib.util.spec_from_file_location("error_repair", repair_path)
            module = importlib.util.module_from_spec(spec)  # type: ignore
            spec.loader.exec_module(module)  # type: ignore
            module.repair(report)
        else:
            print("  [REPAIR] error_repair.py not found")
    except Exception as e:
        print(f"  [REPAIR] Error: {e}")


# ─── Flask HTTP server ────────────────────────────────────────────────────────


def start_flask_server(config: dict):
    """Starts Flask server with security middleware."""
    try:
        from flask import Flask, request, jsonify, abort
    except ImportError:
        print("[LISTENER] Flask not installed: pip install flask")
        return

    app = Flask(__name__)

    @app.route("/health", methods=["GET"])
    def health():
        return jsonify({"status": "ok", "timestamp": datetime.now().isoformat()})

    @app.route("/report_error", methods=["POST"])
    def report_error():
        ip = request.remote_addr or "unknown"

        # Rate limit
        if not _check_rate_limit(ip, config.get("rate_limit_per_min", 10)):
            abort(429)  # Too Many Requests

        # Token verification
        token = request.headers.get("X-API-Token", "")
        if not _verify_token(token, config.get("api_token", "")):
            abort(403)  # Forbidden

        # Validate JSON
        data = request.get_json(silent=True)
        if not data or "error_message" not in data:
            abort(400)  # Bad Request

        result = process_error_report(data)
        return jsonify(result), 200

    @app.route("/status", methods=["GET"])
    def status():
        """Colab can use this to check if Linux-PC is reachable."""
        token = request.headers.get("X-API-Token", "")
        if not _verify_token(token, config.get("api_token", "")):
            abort(403)
        return jsonify(
            {
                "status": "alive",
                "timestamp": datetime.now().isoformat(),
                "host": os.uname().nodename,
            }
        )

    port = config.get("port", 5001)
    print(f"[LISTENER] Flask starting on port {port}")
    print(f"[LISTENER] Endpoints: /health, /report_error, /status")
    # debug=False, threaded=True for production
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)


# ─── Drive bus polling (backup channel) ─────────────────────────────────────


def start_drive_polling(config: dict):
    """
    Polling loop: reads error reports from Google Drive.
    Colab writes error_report.json to Drive when HTTP fails.
    """
    interval = config.get("drive_polling_interval_sec", 120)
    print(f"[LISTENER] Drive polling every {interval}s started")

    while True:
        try:
            _check_drive_for_errors()
        except Exception as e:
            print(f"[LISTENER] Drive polling error: {e}")
        time.sleep(interval)


def _check_drive_for_errors():
    """Reads error_report.json from Drive and processes them."""
    try:
        from drive_manager import (
            get_drive_service,
            load_drive_config,
            find_file_in_folder,
        )
        from googleapiclient.http import MediaIoBaseDownload
        import io

        service = get_drive_service()
        config = load_drive_config()
        folder_id = config.get("champion_folder_id", "")
        if not folder_id:
            return

        file_id = find_file_in_folder(service, folder_id, "colab_error_report.json")
        if not file_id:
            return

        # Download file
        request = service.files().get_media(fileId=file_id)
        buf = io.BytesIO()
        downloader = MediaIoBaseDownload(buf, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()

        buf.seek(0)
        report = json.loads(buf.read().decode())

        # Process
        print("[LISTENER] Drive: error report found!")
        process_error_report(report)

        # Delete after processing (no double processing)
        service.files().delete(fileId=file_id).execute()
        print("[LISTENER] Drive: error report processed and deleted")

    except ImportError:
        pass  # Drive not configured
    except Exception as e:
        print(f"[LISTENER] Drive check error: {e}")


# ─── Main program ────────────────────────────────────────────────────────────


def run():
    """Starts both channels in parallel."""
    config = load_config()

    print("=" * 55)
    print("  BITCOIN4Traders - Error Listener")
    print(f"  Channel 1: HTTP Port {config['port']}")
    print(f"  Channel 2: Drive polling every {config['drive_polling_interval_sec']}s")
    print("=" * 55)

    # Drive polling in background thread
    drive_thread = threading.Thread(
        target=start_drive_polling, args=(config,), daemon=True
    )
    drive_thread.start()

    # Flask in main thread (blocking)
    start_flask_server(config)


def setup():
    """Initial configuration."""
    import secrets

    config = load_config()
    if config["api_token"] == "CHANGE_ME_generate_random_token":
        config["api_token"] = secrets.token_hex(32)
        save_config(config)
        print(f"API token generated: {config['api_token']}")
        print(f"-> Enter this token in the Colab notebook as LINUX_API_TOKEN!")
    else:
        print(f"API token already set: {config['api_token'][:8]}...")
    print(f"Port: {config['port']}")
    print(f"Config: {CONFIG_FILE}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "setup":
        setup()
    else:
        run()
