"""
transport_base.py — Gemeinsames Transport-Interface
====================================================
Alle 4 Kommunikations-Optionen implementieren dieses Interface.
Dadurch kann Module A und Module B den Transport jederzeit wechseln
ohne den Handelslogik-Code zu ändern.

Kanal-Konventionen (identisch für alle Transporte):
  bt4t:market:BTCUSDT   — Marktdaten Lokal → Colab
  bt4t:signals          — Handelssignale Colab → Lokal
  bt4t:portfolio:state  — Portfolio-State Lokal → Colab
  bt4t:health           — Heartbeat Colab → Lokal
  bt4t:control:cmd      — Steuerbefehle Lokal → Colab
  bt4t:control:ack      — Bestätigung Colab → Lokal
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional

# ── Kanal-Konstanten ──────────────────────────────────────────────────────────
CH_MARKET = "bt4t:market:{symbol}"
CH_SIGNALS = "bt4t:signals"
CH_PORTFOLIO = "bt4t:portfolio:state"
CH_HEALTH = "bt4t:health"
CH_CONTROL = "bt4t:control:cmd"
CH_ACK = "bt4t:control:ack"


class TransportBase(ABC):
    """
    Abstrakte Basisklasse für alle Kommunikations-Transporte.

    Jeder Transport implementiert:
      connect()     — Verbindung aufbauen
      disconnect()  — Verbindung sauber trennen
      publish()     — Nachricht auf Kanal senden
      subscribe()   — Callback für eingehende Nachrichten registrieren
    """

    @abstractmethod
    async def connect(self) -> None:
        """Verbindung aufbauen. Wirft Exception bei Fehler."""
        ...

    @abstractmethod
    async def disconnect(self) -> None:
        """Verbindung sauber trennen."""
        ...

    @abstractmethod
    async def publish(self, channel: str, payload: dict) -> None:
        """
        Nachricht auf Kanal publishen.

        Parameters
        ----------
        channel : str   Kanalname, z.B. 'bt4t:signals'
        payload : dict  JSON-serialisierbares Dict
        """
        ...

    @abstractmethod
    async def subscribe(self, channel: str, callback: Callable[[dict], None]) -> None:
        """
        Callback für eingehende Nachrichten auf Kanal registrieren.

        Der Callback wird mit dem deserialisierten payload-Dict aufgerufen.

        Parameters
        ----------
        channel  : str      Kanalname
        callback : callable Funktion(payload: dict) -> None
        """
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Name des Transports für Logging."""
        ...

    @property
    @abstractmethod
    def latency_class(self) -> str:
        """Erwartete Latenzklasse: 'ms', 'sub-second', 'seconds', 'minutes'"""
        ...

    # ── Hilfsmethoden ─────────────────────────────────────────────────────────

    @staticmethod
    def encode(payload: dict) -> str:
        return json.dumps(payload, default=str)

    @staticmethod
    def decode(data: str | bytes | dict) -> dict:
        if isinstance(data, dict):
            return data
        if isinstance(data, bytes):
            data = data.decode("utf-8")
        return json.loads(data)
