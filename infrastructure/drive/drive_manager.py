#!/usr/bin/env python3
"""
Drive Manager - Google Drive Sync for BITCOIN4Traders
======================================================
Replaces pydrive2 (outdated) with google-api-python-client (current, stable).

Tasks:
  - Synchronize champion model bidirectionally
  - Upload training logs
  - Update Colab notebook
  - Write heartbeat file (proof that bot is alive)

Cost: 0 EUR (Google Drive API = free up to 1 billion requests/day)
"""

import json
import os
import time
from datetime import datetime
from pathlib import Path

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload
import io

# ─── Configuration ────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).resolve().parents[2]
CREDENTIALS_FILE = BASE_DIR / "config" / "gdrive_credentials.json"
CACHE_DIR = BASE_DIR / "data" / "cache"
LOGS_DIR = BASE_DIR / "logs"

SCOPES = ["https://www.googleapis.com/auth/drive"]

# Google Drive folder IDs (copy once from Drive URL)
# Format: https://drive.google.com/drive/folders/THIS_ID_HERE
DRIVE_CONFIG = {
    "champion_folder_id": "",  # <-- Folder ID for models
    "logs_folder_id": "",  # <-- Folder ID for logs
    "notebook_file_id": "",  # <-- File ID of the Colab notebook
}

CONFIG_FILE = BASE_DIR / "config" / "drive_config.json"


# ─── Drive client ─────────────────────────────────────────────────────────────


def get_drive_service():
    """Creates authenticated Google Drive service without browser."""
    if not CREDENTIALS_FILE.exists():
        raise FileNotFoundError(
            f"Service Account credentials not found: {CREDENTIALS_FILE}\n"
            f"Please place credentials.json from Google Cloud Console there."
        )

    creds = service_account.Credentials.from_service_account_file(
        str(CREDENTIALS_FILE), scopes=SCOPES
    )
    service = build("drive", "v3", credentials=creds, cache_discovery=False)
    return service


def load_drive_config() -> dict:
    """Loads Drive folder IDs from configuration file."""
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE) as f:
            return json.load(f)
    return DRIVE_CONFIG.copy()


def save_drive_config(config: dict):
    """Saves Drive folder IDs."""
    CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=2)


# ─── File operations ────────────────────────────────────────────────────────────


def find_file_in_folder(service, folder_id: str, filename: str) -> str | None:
    """Searches for a file in a Drive folder, returns file ID or None."""
    query = f"name='{filename}' and '{folder_id}' in parents and trashed=false"
    result = service.files().list(q=query, fields="files(id, name)").execute()
    files = result.get("files", [])
    return files[0]["id"] if files else None


def upload_file(
    service, local_path: Path, folder_id: str, description: str = ""
) -> str:
    """
    Uploads or updates a file (no duplicates).
    Returns file ID.
    """
    filename = local_path.name
    mime_type = _guess_mime(local_path)

    # Search for existing file
    existing_id = find_file_in_folder(service, folder_id, filename)

    media = MediaFileUpload(str(local_path), mimetype=mime_type, resumable=True)

    if existing_id:
        # Update (no new file, no Drive clutter)
        file = service.files().update(fileId=existing_id, media_body=media).execute()
        print(f"  [DRIVE] Updated: {filename} ({_size_str(local_path)})")
    else:
        # Create new
        metadata = {
            "name": filename,
            "parents": [folder_id],
            "description": description,
        }
        file = (
            service.files()
            .create(body=metadata, media_body=media, fields="id")
            .execute()
        )
        print(f"  [DRIVE] Uploaded: {filename} ({_size_str(local_path)})")

    return file.get("id", "")


def download_file(service, file_id: str, local_path: Path):
    """Downloads a file from Drive."""
    request = service.files().get_media(fileId=file_id)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    with open(local_path, "wb") as f:
        downloader = MediaIoBaseDownload(f, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()
    print(f"  [DRIVE] Downloaded: {local_path.name}")


def write_heartbeat(service, folder_id: str):
    """Writes heartbeat file to Drive - proof that Linux-PC is alive."""
    heartbeat = {
        "timestamp": datetime.now().isoformat(),
        "status": "alive",
        "host": os.uname().nodename,
        "uptime": _get_uptime(),
    }
    hb_file = Path("/tmp/heartbeat.json")
    with open(hb_file, "w") as f:
        json.dump(heartbeat, f, indent=2)

    upload_file(service, hb_file, folder_id, "Linux-PC Heartbeat")
    print(f"  [DRIVE] Heartbeat: {heartbeat['timestamp']}")


# ─── Champion sync ────────────────────────────────────────────────────────────


def sync_champion_to_drive(service, config: dict) -> bool:
    """
    Uploads the best champion from Linux-PC -> Google Drive.
    Colab can then load it from there.
    """
    folder_id = config.get("champion_folder_id", "")
    if not folder_id:
        print("  [DRIVE] champion_folder_id not configured - skipped")
        return False

    champion_pkl = CACHE_DIR / "multiverse_champion.pkl"
    champion_meta = CACHE_DIR / "multiverse_champion_meta.json"

    uploaded = False
    for f in [champion_pkl, champion_meta]:
        if f.exists():
            upload_file(
                service,
                f,
                folder_id,
                f"BITCOIN4Traders Champion - {datetime.now().strftime('%Y-%m-%d')}",
            )
            uploaded = True
        else:
            print(f"  [DRIVE] {f.name} not found - skipped")

    return uploaded


def sync_champion_from_drive(service, config: dict) -> bool:
    """
    Downloads champion from Google Drive -> Linux-PC.
    Useful when Colab has trained a better model.
    """
    folder_id = config.get("champion_folder_id", "")
    if not folder_id:
        return False

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    for filename in ["multiverse_champion.pkl", "multiverse_champion_meta.json"]:
        file_id = find_file_in_folder(service, folder_id, filename)
        if file_id:
            local_path = CACHE_DIR / filename
            # Only download if Drive version is newer
            download_file(service, file_id, local_path)
        else:
            print(f"  [DRIVE] {filename} not on Drive - skipped")

    return True


def upload_logs(service, config: dict):
    """Uploads the latest log files (only last 24h)."""
    folder_id = config.get("logs_folder_id", "")
    if not folder_id:
        return

    now = time.time()
    log_count = 0

    for log_file in LOGS_DIR.rglob("*.log"):
        # Only logs from the last 24 hours
        if now - log_file.stat().st_mtime < 86400:
            upload_file(service, log_file, folder_id)
            log_count += 1

    print(f"  [DRIVE] {log_count} log files uploaded")


# ─── Helper functions ──────────────────────────────────────────────────────────


def _guess_mime(path: Path) -> str:
    suffix = path.suffix.lower()
    return {
        ".json": "application/json",
        ".pkl": "application/octet-stream",
        ".log": "text/plain",
        ".ipynb": "application/json",
        ".txt": "text/plain",
    }.get(suffix, "application/octet-stream")


def _size_str(path: Path) -> str:
    size = path.stat().st_size
    if size < 1024:
        return f"{size}B"
    elif size < 1024**2:
        return f"{size / 1024:.1f}KB"
    else:
        return f"{size / 1024**2:.1f}MB"


def _get_uptime() -> str:
    try:
        with open("/proc/uptime") as f:
            seconds = float(f.read().split()[0])
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"
    except Exception:
        return "unknown"


# ─── Main function ────────────────────────────────────────────────────────────


def run_sync(direction: str = "both"):
    """
    Performs a complete Drive sync.
    direction: "up" | "down" | "both"
    """
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Drive-Sync started ({direction})")

    config = load_drive_config()
    service = get_drive_service()

    folder_id = config.get("champion_folder_id", "")
    if folder_id:
        write_heartbeat(service, folder_id)

    if direction in ("up", "both"):
        sync_champion_to_drive(service, config)
        upload_logs(service, config)

    if direction in ("down", "both"):
        sync_champion_from_drive(service, config)

    print(f"[{datetime.now().strftime('%H:%M:%S')}] Drive-Sync completed\n")


def setup_interactive():
    """
    Interactive setup assistant for initial configuration.
    """
    print("=" * 60)
    print("  Google Drive Setup - BITCOIN4Traders")
    print("=" * 60)
    print()
    print("Go to: https://drive.google.com")
    print("Create 2 folders: 'BTC4T_Champions' and 'BTC4T_Logs'")
    print("Share both folders with the service account email (write permissions)")
    print()

    config = load_drive_config()

    config["champion_folder_id"] = input("Champion folder ID (from URL): ").strip()
    config["logs_folder_id"] = input("Logs folder ID (from URL): ").strip()
    config["notebook_file_id"] = input(
        "Colab notebook ID (optional, Enter=skip): "
    ).strip()

    save_drive_config(config)
    print("\nConfiguration saved to:", CONFIG_FILE)

    # Test connection
    print("\nTesting connection...")
    try:
        service = get_drive_service()
        write_heartbeat(service, config["champion_folder_id"])
        print("Connection successful!")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "setup":
        setup_interactive()
    else:
        run_sync("both")
