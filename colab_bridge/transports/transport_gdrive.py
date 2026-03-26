"""
Transport Option 3: Google Drive as message bus
================================================

Latency  : 2–15 seconds  (Drive sync interval)
Cost     : $0 (15 GB free)
Accounts : Google account (already available for Colab)
Reliable : High (Google infrastructure)
Pros     : No installation needed, already available in Colab,
           already in project (drive_manager.py), persistent audit trail
Cons     : 2–15s latency (not suitable for 1m bars),
           Drive API rate limits (300 req/min)

Architecture:
┌─────────────────────────────────────────────────────────┐
│  LOCAL                         COLAB                    │
│                                                         │
│  drive_manager.py               google.colab.drive      │
│  writes JSON files              reads JSON files        │
│       │                              │                  │
│       └──── Google Drive ────────────┘                  │
│                                                         │
│  Structure on Drive:                                    │
│  bt4t/                                                  │
│    market/                                              │
│      BTCUSDT_latest.json   ← Local writes               │
│    signals/                                             │
│      latest.json           ← Colab writes               │
│    health/                                              │
│      heartbeat.json        ← Colab writes               │
│    control/                                             │
│      cmd.json              ← Local writes               │
│      ack.json              ← Colab writes               │
└─────────────────────────────────────────────────────────┘

Recommendation: Use only for 1h+ timeframes.
For 15m bars: latency is borderline.
For 1m bars: not suitable.

Installation:
  Local (for writing to Drive):
    pip install google-auth google-auth-oauthlib google-api-python-client

  Colab: Already available (google.colab.drive)

Authentication:
  Local: OAuth2 Credentials (one-time, saves token in ~/.config/bt4t/)
  Colab: drive.mount('/content/drive') — already integrated
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional

try:
    from loguru import logger
except ImportError:
    import logging

    logger = logging.getLogger("transport_gdrive")

from colab_bridge.transport_base import TransportBase

# ── Path configuration ────────────────────────────────────────────────────────
# Local (gdrive_manager syncs this directory):
LOCAL_SYNC_DIR = Path(os.getenv("BT4T_DRIVE_SYNC_DIR", "data/drive_sync"))

# In Colab (after drive.mount):
COLAB_DRIVE_DIR = Path("/content/drive/MyDrive/BITCOIN4Traders/bt4t_bus")

# Channel → filename mapping
CHANNEL_FILES = {
    "bt4t:market:BTCUSDT": "market/BTCUSDT_latest.json",
    "bt4t:market:ETHUSDT": "market/ETHUSDT_latest.json",
    "bt4t:signals": "signals/latest.json",
    "bt4t:portfolio:state": "portfolio/state.json",
    "bt4t:health": "health/heartbeat.json",
    "bt4t:control:cmd": "control/cmd.json",
    "bt4t:control:ack": "control/ack.json",
}

POLL_INTERVAL_S = 3.0  # How often to check for new files
STALENESS_S = 60.0  # Message older than N seconds → ignore


class DriveTransportLocal(TransportBase):
    """
    Local side: Reads/writes JSON files to a local directory
    that is synchronized by Google Drive (drive_manager.py).

    Usage:
        transport = DriveTransportLocal(
            sync_dir="data/drive_sync",   # Synced by drive_manager.py
        )
        await transport.connect()
        await transport.publish("bt4t:market:BTCUSDT", market_data)
        await transport.subscribe("bt4t:signals", on_signal)
    """

    def __init__(
        self,
        sync_dir: str | Path = LOCAL_SYNC_DIR,
        poll_interval_s: float = POLL_INTERVAL_S,
        use_drive_api: bool = False,  # True: write directly via Drive API
    ):
        self.sync_dir = Path(sync_dir)
        self.poll_interval = poll_interval_s
        self.use_drive_api = use_drive_api
        self._callbacks: Dict[str, List[Callable]] = defaultdict(list)
        self._poll_tasks: Dict[str, asyncio.Task] = {}
        self._file_mtimes: Dict[str, float] = {}
        self._running = False

    @property
    def name(self) -> str:
        return "GoogleDrive(Local)"

    @property
    def latency_class(self) -> str:
        return "seconds"  # 2–15s

    async def connect(self) -> None:
        """Creates directory structure."""
        self._running = True
        for subdir in ["market", "signals", "portfolio", "health", "control"]:
            (self.sync_dir / subdir).mkdir(parents=True, exist_ok=True)
        logger.success(f"[Drive/Local] Sync directory: {self.sync_dir}")
        logger.info("[Drive/Local] Make sure drive_manager.py is running!")

    async def disconnect(self) -> None:
        self._running = False
        for task in self._poll_tasks.values():
            task.cancel()
        logger.info("[Drive/Local] Disconnected")

    async def publish(self, channel: str, payload: dict) -> None:
        """
        Writes payload as JSON file to sync_dir.
        Drive Manager syncs the file to Google Drive.
        """
        rel_path = CHANNEL_FILES.get(channel)
        if not rel_path:
            rel_path = f"misc/{channel.replace(':', '_')}.json"

        file_path = self.sync_dir / rel_path
        file_path.parent.mkdir(parents=True, exist_ok=True)

        # Add timestamp for staleness check
        payload["_written_at"] = datetime.now(timezone.utc).isoformat()
        payload["_channel"] = channel

        # Atomic write (temp file → rename)
        tmp = file_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, default=str, indent=2), encoding="utf-8")
        tmp.rename(file_path)

        logger.debug(f"[Drive/Local] WRITE {file_path}")

    async def subscribe(self, channel: str, callback: Callable[[dict], None]) -> None:
        """Watches file for changes (file mtime polling)."""
        self._callbacks[channel].append(callback)
        if channel not in self._poll_tasks:
            self._poll_tasks[channel] = asyncio.create_task(self._watch_file(channel))
        logger.debug(f"[Drive/Local] Watching: {channel}")

    async def _watch_file(self, channel: str) -> None:
        """Polls file mtime and triggers callback on change."""
        rel_path = CHANNEL_FILES.get(channel, f"misc/{channel.replace(':', '_')}.json")
        file_path = self.sync_dir / rel_path

        while self._running:
            try:
                if file_path.exists():
                    mtime = file_path.stat().st_mtime
                    last_mtime = self._file_mtimes.get(channel, 0.0)

                    if mtime > last_mtime:
                        self._file_mtimes[channel] = mtime
                        payload = json.loads(file_path.read_text(encoding="utf-8"))

                        # Staleness check
                        written_at_str = payload.get("_written_at", "")
                        if written_at_str:
                            written_at = datetime.fromisoformat(
                                written_at_str.replace("Z", "+00:00")
                            )
                            age_s = (
                                datetime.now(timezone.utc) - written_at
                            ).total_seconds()
                            if age_s > STALENESS_S:
                                await asyncio.sleep(self.poll_interval)
                                continue

                        for cb in self._callbacks.get(channel, []):
                            cb(payload)
                        logger.debug(f"[Drive/Local] UPDATE {channel}")

            except Exception as e:
                logger.warning(f"[Drive/Local] Watch error {channel}: {e}")

            await asyncio.sleep(self.poll_interval)


class DriveTransportColab(TransportBase):
    """
    Colab side: Reads/writes files directly on Google Drive
    (after drive.mount('/content/drive')).

    No HTTP proxy needed — Drive is mounted directly.

    Usage in Colab:
        from google.colab import drive
        drive.mount('/content/drive')

        from colab_bridge.transports.transport_gdrive import DriveTransportColab
        transport = DriveTransportColab()
        await transport.connect()
        await transport.subscribe("bt4t:market:BTCUSDT", on_market_data)
        await transport.publish("bt4t:signals", signal)
    """

    def __init__(
        self,
        drive_dir: str | Path = COLAB_DRIVE_DIR,
        poll_interval_s: float = POLL_INTERVAL_S,
    ):
        self.drive_dir = Path(drive_dir)
        self.poll_interval = poll_interval_s
        self._callbacks: Dict[str, List[Callable]] = defaultdict(list)
        self._poll_tasks: Dict[str, asyncio.Task] = {}
        self._file_mtimes: Dict[str, float] = {}
        self._running = False

    @property
    def name(self) -> str:
        return "GoogleDrive(Colab)"

    @property
    def latency_class(self) -> str:
        return "seconds"

    async def connect(self) -> None:
        """Checks Drive mountpoint and creates directory structure."""
        self._running = True
        if not Path("/content/drive").exists():
            logger.warning(
                "[Drive/Colab] /content/drive not mounted!\n"
                "Run first: from google.colab import drive; drive.mount('/content/drive')"
            )

        for subdir in ["market", "signals", "portfolio", "health", "control"]:
            (self.drive_dir / subdir).mkdir(parents=True, exist_ok=True)

        logger.success(f"[Drive/Colab] Drive directory: {self.drive_dir}")

    async def disconnect(self) -> None:
        self._running = False
        for task in self._poll_tasks.values():
            task.cancel()

    async def publish(self, channel: str, payload: dict) -> None:
        """Writes to Drive file."""
        rel_path = CHANNEL_FILES.get(channel, f"misc/{channel.replace(':', '_')}.json")
        file_path = self.drive_dir / rel_path
        file_path.parent.mkdir(parents=True, exist_ok=True)

        payload["_written_at"] = datetime.now(timezone.utc).isoformat()
        payload["_channel"] = channel

        tmp = file_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, default=str, indent=2), encoding="utf-8")
        tmp.rename(file_path)
        logger.debug(f"[Drive/Colab] WRITE {file_path}")

    async def subscribe(self, channel: str, callback: Callable[[dict], None]) -> None:
        self._callbacks[channel].append(callback)
        if channel not in self._poll_tasks:
            self._poll_tasks[channel] = asyncio.create_task(self._watch_file(channel))

    async def _watch_file(self, channel: str) -> None:
        """Identical to DriveTransportLocal._watch_file."""
        rel_path = CHANNEL_FILES.get(channel, f"misc/{channel.replace(':', '_')}.json")
        file_path = self.drive_dir / rel_path

        while self._running:
            try:
                if file_path.exists():
                    mtime = file_path.stat().st_mtime
                    last_mtime = self._file_mtimes.get(channel, 0.0)

                    if mtime > last_mtime:
                        self._file_mtimes[channel] = mtime
                        payload = json.loads(file_path.read_text(encoding="utf-8"))

                        written_at_str = payload.get("_written_at", "")
                        if written_at_str:
                            written_at = datetime.fromisoformat(
                                written_at_str.replace("Z", "+00:00")
                            )
                            age_s = (
                                datetime.now(timezone.utc) - written_at
                            ).total_seconds()
                            if age_s > STALENESS_S:
                                await asyncio.sleep(self.poll_interval)
                                continue

                        for cb in self._callbacks.get(channel, []):
                            cb(payload)
                        logger.debug(f"[Drive/Colab] UPDATE {channel}")

            except Exception as e:
                logger.warning(f"[Drive/Colab] Watch error {channel}: {e}")

            await asyncio.sleep(self.poll_interval)
