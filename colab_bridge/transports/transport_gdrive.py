"""
Transport Option 3: Google Drive als Nachrichtenbus
====================================================

Latenz    : 2–15 Sekunden  (Drive-Sync-Intervall)
Kosten    : $0 (15 GB kostenlos)
Accounts  : Google-Account (bereits für Colab vorhanden)
Zuverlässig: Hoch (Google-Infrastruktur)
Vorteile  : Keine Installation, in Colab bereits verfügbar,
            bereits im Projekt (drive_manager.py), persistenter Audit-Trail
Nachteile : 2–15s Latenz (nicht für 1m-Bars geeignet),
            Drive-API Rate-Limits (300 req/min)

Architektur:
┌─────────────────────────────────────────────────────────┐
│  LOKAL                         COLAB                    │
│                                                         │
│  drive_manager.py               google.colab.drive      │
│  schreibt JSON-Dateien          liest JSON-Dateien      │
│       │                              │                  │
│       └──── Google Drive ────────────┘                  │
│                                                         │
│  Struktur auf Drive:                                    │
│  bt4t/                                                  │
│    market/                                              │
│      BTCUSDT_latest.json   ← Lokal schreibt             │
│    signals/                                             │
│      latest.json           ← Colab schreibt             │
│    health/                                              │
│      heartbeat.json        ← Colab schreibt             │
│    control/                                             │
│      cmd.json              ← Lokal schreibt             │
│      ack.json              ← Colab schreibt             │
└─────────────────────────────────────────────────────────┘

Empfehlung: Nur für 1h+ Timeframes verwenden.
Für 15m-Bars: Latenz grenzwertig.
Für 1m-Bars: Nicht geeignet.

Installation:
  Lokal (für Schreiben auf Drive):
    pip install google-auth google-auth-oauthlib google-api-python-client

  Colab: Bereits vorhanden (google.colab.drive)

Authentifizierung:
  Lokal: OAuth2 Credentials (einmalig, speichert Token in ~/.config/bt4t/)
  Colab: drive.mount('/content/drive') — bereits integriert
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

# ── Pfad-Konfiguration ────────────────────────────────────────────────────────
# Lokal (gdrive_manager synct diesen Ordner):
LOCAL_SYNC_DIR = Path(os.getenv("BT4T_DRIVE_SYNC_DIR", "data/drive_sync"))

# In Colab (nach drive.mount):
COLAB_DRIVE_DIR = Path("/content/drive/MyDrive/BITCOIN4Traders/bt4t_bus")

# Kanal → Dateiname Mapping
CHANNEL_FILES = {
    "bt4t:market:BTCUSDT": "market/BTCUSDT_latest.json",
    "bt4t:market:ETHUSDT": "market/ETHUSDT_latest.json",
    "bt4t:signals": "signals/latest.json",
    "bt4t:portfolio:state": "portfolio/state.json",
    "bt4t:health": "health/heartbeat.json",
    "bt4t:control:cmd": "control/cmd.json",
    "bt4t:control:ack": "control/ack.json",
}

POLL_INTERVAL_S = 3.0  # Wie oft auf neue Dateien prüfen
STALENESS_S = 60.0  # Nachricht älter als N Sekunden → ignorieren


class DriveTransportLocal(TransportBase):
    """
    Lokale Seite: Schreibt/liest JSON-Dateien in einen lokalen Ordner,
    der von Google Drive (drive_manager.py) synchronisiert wird.

    Verwendung:
        transport = DriveTransportLocal(
            sync_dir="data/drive_sync",   # Wird von drive_manager.py synct
        )
        await transport.connect()
        await transport.publish("bt4t:market:BTCUSDT", market_data)
        await transport.subscribe("bt4t:signals", on_signal)
    """

    def __init__(
        self,
        sync_dir: str | Path = LOCAL_SYNC_DIR,
        poll_interval_s: float = POLL_INTERVAL_S,
        use_drive_api: bool = False,  # True: direkt via Drive API schreiben
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
        """Erstellt Verzeichnisstruktur."""
        self._running = True
        for subdir in ["market", "signals", "portfolio", "health", "control"]:
            (self.sync_dir / subdir).mkdir(parents=True, exist_ok=True)
        logger.success(f"[Drive/Local] Sync-Verzeichnis: {self.sync_dir}")
        logger.info("[Drive/Local] Stelle sicher dass drive_manager.py läuft!")

    async def disconnect(self) -> None:
        self._running = False
        for task in self._poll_tasks.values():
            task.cancel()
        logger.info("[Drive/Local] Getrennt")

    async def publish(self, channel: str, payload: dict) -> None:
        """
        Schreibt payload als JSON-Datei in sync_dir.
        Drive-Manager synct die Datei zu Google Drive.
        """
        rel_path = CHANNEL_FILES.get(channel)
        if not rel_path:
            rel_path = f"misc/{channel.replace(':', '_')}.json"

        file_path = self.sync_dir / rel_path
        file_path.parent.mkdir(parents=True, exist_ok=True)

        # Timestamp hinzufügen für Staleness-Check
        payload["_written_at"] = datetime.now(timezone.utc).isoformat()
        payload["_channel"] = channel

        # Atomares Schreiben (temp file → rename)
        tmp = file_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, default=str, indent=2), encoding="utf-8")
        tmp.rename(file_path)

        logger.debug(f"[Drive/Local] WRITE {file_path}")

    async def subscribe(self, channel: str, callback: Callable[[dict], None]) -> None:
        """Überwacht Datei auf Änderungen (file mtime polling)."""
        self._callbacks[channel].append(callback)
        if channel not in self._poll_tasks:
            self._poll_tasks[channel] = asyncio.create_task(self._watch_file(channel))
        logger.debug(f"[Drive/Local] Überwache: {channel}")

    async def _watch_file(self, channel: str) -> None:
        """Pollt Datei-mtime und triggert Callback bei Änderung."""
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

                        # Staleness-Check
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
                logger.warning(f"[Drive/Local] Watch Fehler {channel}: {e}")

            await asyncio.sleep(self.poll_interval)


class DriveTransportColab(TransportBase):
    """
    Colab-Seite: Liest/schreibt Dateien direkt auf Google Drive
    (nach drive.mount('/content/drive')).

    Kein HTTP-Proxy nötig — Drive ist direkt gemountet.

    Verwendung in Colab:
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
        """Prüft Drive-Mountpoint und erstellt Verzeichnisstruktur."""
        self._running = True
        if not Path("/content/drive").exists():
            logger.warning(
                "[Drive/Colab] /content/drive nicht gemountet!\n"
                "Führe zuerst aus: from google.colab import drive; drive.mount('/content/drive')"
            )

        for subdir in ["market", "signals", "portfolio", "health", "control"]:
            (self.drive_dir / subdir).mkdir(parents=True, exist_ok=True)

        logger.success(f"[Drive/Colab] Drive-Verzeichnis: {self.drive_dir}")

    async def disconnect(self) -> None:
        self._running = False
        for task in self._poll_tasks.values():
            task.cancel()

    async def publish(self, channel: str, payload: dict) -> None:
        """Schreibt auf Drive-Datei."""
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
        """Identisch wie DriveTransportLocal._watch_file."""
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
                logger.warning(f"[Drive/Colab] Watch Fehler {channel}: {e}")

            await asyncio.sleep(self.poll_interval)
