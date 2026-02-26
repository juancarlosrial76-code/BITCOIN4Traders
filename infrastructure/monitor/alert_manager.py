#!/usr/bin/env python3
"""
Alert Manager - Intelligent Error Handling & Notifications
=================================================================
Central error handling for the entire infrastructure.

Channels (all free):
  1. Telegram Bot  - Push notification to mobile phone
  2. Log file      - Always available, even offline
  3. Desktop popup - When Linux-PC is actively used

Strategy:
  - Deduplication: same error reported at most every 30 min
  - Severity levels: INFO / WARNING / CRITICAL
  - Automatic log rotation after 7 days
"""

import json
import os
import time
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from typing import Literal

# ─── Configuration ────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).resolve().parents[2]
LOG_DIR = BASE_DIR / "logs" / "alerts"
CONFIG_FILE = BASE_DIR / "config" / "alert_config.json"

Severity = Literal["info", "warning", "critical"]

DEFAULT_CONFIG = {
    "telegram": {
        "enabled": False,
        "bot_token": "",  # From @BotFather
        "chat_id": "",  # Your chat ID
    },
    "desktop_notify": True,
    "min_interval_sec": {
        "info": 3600,  # INFO max. 1x per hour
        "warning": 1800,  # WARNING max. 1x per 30 min
        "critical": 300,  # CRITICAL max. 1x per 5 min
    },
    "log_retention_days": 7,
}

# ─── Deduplication ───────────────────────────────────────────────────────────

_last_sent: dict[str, datetime] = {}


def _should_send(key: str, severity: Severity, config: dict) -> bool:
    """Prevents spam: do not send the same message too often."""
    min_sec = config.get("min_interval_sec", {}).get(severity, 1800)
    last = _last_sent.get(key)
    if last is None or (datetime.now() - last).total_seconds() > min_sec:
        _last_sent[key] = datetime.now()
        return True
    return False


# ─── Telegram ─────────────────────────────────────────────────────────────────


def _send_telegram(message: str, config: dict) -> bool:
    """Sends Telegram message. Returns True on success."""
    tg = config.get("telegram", {})
    if not tg.get("enabled") or not tg.get("bot_token") or not tg.get("chat_id"):
        return False

    try:
        import urllib.request
        import urllib.parse

        token = tg["bot_token"]
        chat_id = tg["chat_id"]
        text = urllib.parse.quote(message)
        url = f"https://api.telegram.org/bot{token}/sendMessage?chat_id={chat_id}&text={text}&parse_mode=HTML"

        with urllib.request.urlopen(url, timeout=10) as response:
            result = json.loads(response.read())
            return result.get("ok", False)

    except Exception as e:
        _log_to_file(f"Telegram error: {e}", "warning")
        return False


# ─── Desktop notification ─────────────────────────────────────────────────────


def _send_desktop(title: str, message: str):
    """Shows desktop notification (Linux: notify-send)."""
    try:
        subprocess.run(
            ["notify-send", "-u", "normal", "-t", "5000", title, message],
            check=False,
            capture_output=True,
        )
    except FileNotFoundError:
        pass  # notify-send not installed


# ─── Logging ──────────────────────────────────────────────────────────────────


def _log_to_file(message: str, severity: Severity):
    """Writes alert to log file."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOG_DIR / f"alerts_{datetime.now().strftime('%Y-%m-%d')}.log"

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] [{severity.upper():8s}] {message}\n"

    with open(log_file, "a") as f:
        f.write(line)

    # Delete old logs
    _rotate_logs()


def _rotate_logs(retention_days: int = 7):
    """Deletes log files older than retention_days."""
    if not LOG_DIR.exists():
        return
    cutoff = datetime.now() - timedelta(days=retention_days)
    for log_file in LOG_DIR.glob("alerts_*.log"):
        try:
            date_str = log_file.stem.replace("alerts_", "")
            file_date = datetime.strptime(date_str, "%Y-%m-%d")
            if file_date < cutoff:
                log_file.unlink()
        except Exception:
            pass


# ─── Load configuration ──────────────────────────────────────────────────────


def load_config() -> dict:
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE) as f:
            loaded = json.load(f)
        # Merge with defaults
        config = DEFAULT_CONFIG.copy()
        config.update(loaded)
        return config
    return DEFAULT_CONFIG.copy()


def save_config(config: dict):
    CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=2)


# ─── Main interface ──────────────────────────────────────────────────────────


def alert(message: str, severity: Severity = "warning", source: str = "system"):
    """
    Central alert call.

    Example:
        alert("Training stopped", severity="critical", source="colab_watchdog")
        alert("Champion saved", severity="info", source="drive_manager")
    """
    config = load_config()

    # Deduplication key
    key = f"{source}:{message[:50]}"

    # Always log
    full_message = f"[{source}] {message}"
    _log_to_file(full_message, severity)

    # Console
    icons = {"info": "ℹ", "warning": "⚠", "critical": "🔴"}
    print(f"  {icons.get(severity, '•')} [{severity.upper()}] {full_message}")

    # External channels only if interval has elapsed
    if not _should_send(key, severity, config):
        return

    # Telegram
    emoji = {"info": "ℹ️", "warning": "⚠️", "critical": "🚨"}
    tg_msg = (
        f"{emoji.get(severity, '')} <b>BITCOIN4Traders</b>\n"
        f"<b>{severity.upper()}</b> | {source}\n\n"
        f"{message}\n\n"
        f"<i>{datetime.now().strftime('%d.%m.%Y %H:%M:%S')}</i>"
    )
    _send_telegram(tg_msg, config)

    # Desktop popup (only for warning/critical)
    if severity in ("warning", "critical") and config.get("desktop_notify"):
        _send_desktop(f"BITCOIN4Traders - {severity.upper()}", message)


def info(message: str, source: str = "system"):
    alert(message, "info", source)


def warn(message: str, source: str = "system"):
    alert(message, "warning", source)


def critical(message: str, source: str = "system"):
    alert(message, "critical", source)


# ─── Setup ────────────────────────────────────────────────────────────────────


def setup_telegram():
    """Interactive setup for Telegram bot."""
    print("=" * 50)
    print("  Telegram Alert Setup")
    print("=" * 50)
    print()
    print("1. Open Telegram, search for @BotFather")
    print("2. Send /newbot, follow the instructions")
    print("3. Copy the bot token")
    print()

    config = load_config()
    token = input("Bot token: ").strip()
    print()
    print("Now send a message to your bot in Telegram.")
    input("Then press Enter...")

    # Automatically detect chat ID
    try:
        import urllib.request

        url = f"https://api.telegram.org/bot{token}/getUpdates"
        with urllib.request.urlopen(url, timeout=10) as resp:
            data = json.loads(resp.read())
        updates = data.get("result", [])
        if updates:
            chat_id = str(updates[-1]["message"]["chat"]["id"])
            print(f"Chat ID found: {chat_id}")
        else:
            chat_id = input("Enter chat ID manually: ").strip()
    except Exception as e:
        print(f"Auto-detection failed: {e}")
        chat_id = input("Enter chat ID manually: ").strip()

    config["telegram"] = {
        "enabled": True,
        "bot_token": token,
        "chat_id": chat_id,
    }
    save_config(config)

    # Send test message
    success = _send_telegram("Test message from BITCOIN4Traders!", config)
    if success:
        print("Telegram configured and test successful!")
    else:
        print("Error sending - please check token and chat ID")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "setup":
        setup_telegram()
    else:
        # Test
        info("System started", source="alert_manager")
        warn("Test warning", source="alert_manager")
        critical("Test critical", source="alert_manager")
        print("Alerts were sent - check log:", LOG_DIR)
