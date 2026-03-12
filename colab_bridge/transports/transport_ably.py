"""
Transport Option 4: Ably Pub/Sub (externer Dienst)
===================================================

Latenz    : 50–150 ms  (Ably Global Edge Network)
Kosten    : $0 bis 6 Mio. Nachrichten/Monat (Free Tier)
            @ 1 Msg/30s = ~86.400 Msg/Monat → Free Tier reicht
Accounts  : Ably-Account nötig (https://ably.com → kostenlos)
Zuverlässig: Sehr hoch (globales CDN, 99.999% SLA)
Vorteile  : Kein Tunnel nötig, beide Seiten verbinden nach außen,
            Nachrichtenpuffer (letzte 100 Msgs), WebSocket-basiert
Nachteile : Externer Drittanbieter, Account nötig, Internetabhängig

Free Tier Limits:
  6 Mio. Nachrichten/Monat
  100 gleichzeitige Verbindungen
  Nachrichten-History: letzte 100 pro Kanal
  Kein SLA im Free Tier (nur paid)

Account erstellen:
  1. https://ably.com → Sign Up (kostenlos)
  2. Dashboard → Create App → API Keys
  3. Root Key kopieren (hat alle Permissions)
  4. In .env: ABLY_API_KEY=dein_key:dein_secret

Installation:
  pip install ably   (bereits installiert)

Achtung: Free Tier Rate-Limit = 250 Requests/Sekunde.
Bei 30s Poll-Intervall und 5 Kanälen = 0.16 req/s → weit unter Limit.

Dieser Transport ist die einzige Option die OHNE Cloudflare Tunnel auskommt.
Beide Seiten (lokal + Colab) verbinden sich direkt zum Ably-Server.
"""

from __future__ import annotations

import asyncio
import json
from collections import defaultdict
from typing import Callable, Dict, List, Optional

try:
    from ably import AblyRealtime

    _ABLY_OK = True
except ImportError:
    _ABLY_OK = False

try:
    from loguru import logger
except ImportError:
    import logging

    logger = logging.getLogger("transport_ably")

from colab_bridge.transport_base import TransportBase


class AblyTransport(TransportBase):
    """
    Ably Pub/Sub Transport.

    Identisch für BEIDE Seiten (lokal und Colab).
    Beide Seiten verwenden denselben API Key.

    Datenpfad:
      Lokal → Ably → Colab  (Marktdaten, ~50–150ms)
      Colab → Ably → Lokal  (Signale, ~50–150ms)

    Verwendung (lokal UND Colab identisch):
        import os
        transport = AblyTransport(api_key=os.getenv("ABLY_API_KEY"))
        await transport.connect()
        await transport.subscribe("bt4t:signals", on_signal)
        await transport.publish("bt4t:market:BTCUSDT", market_data)
    """

    def __init__(self, api_key: str = ""):
        if not _ABLY_OK:
            raise ImportError("pip install ably\nAccount: https://ably.com (kostenlos)")
        import os

        self.api_key = api_key or os.getenv("ABLY_API_KEY", "")
        if not self.api_key:
            raise ValueError(
                "ABLY_API_KEY fehlt!\n"
                "1. https://ably.com → Sign Up\n"
                "2. Dashboard → App → API Keys → Root Key kopieren\n"
                "3. In .env: ABLY_API_KEY=xxxxx:yyyyy"
            )
        self._ably: Optional[AblyRealtime] = None
        self._callbacks: Dict[str, List[Callable]] = defaultdict(list)

    @property
    def name(self) -> str:
        return "Ably"

    @property
    def latency_class(self) -> str:
        return "ms"  # 50–150ms

    async def connect(self) -> None:
        self._ably = AblyRealtime(self.api_key)
        await self._ably.connection.once_async("connected")
        logger.success(f"[Ably] Verbunden | Key: {self.api_key[:15]}...")

    async def disconnect(self) -> None:
        if self._ably:
            await self._ably.close()
        logger.info("[Ably] Getrennt")

    async def publish(self, channel: str, payload: dict) -> None:
        ch = self._ably.channels.get(channel)
        await ch.publish("update", self.encode(payload))
        logger.debug(f"[Ably] PUBLISH {channel}")

    async def subscribe(self, channel: str, callback: Callable[[dict], None]) -> None:
        def _wrapper(message):
            try:
                data = self.decode(message.data)
                callback(data)
            except Exception as e:
                logger.warning(f"[Ably] Decode Fehler: {e}")

        ch = self._ably.channels.get(channel)
        await ch.subscribe(_wrapper)
        logger.debug(f"[Ably] Abonniert: {channel}")
