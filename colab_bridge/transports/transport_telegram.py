"""
Transport Option 2: Telegram Bot API als Nachrichtenbus
========================================================

Latenz    : 200–800 ms  (Telegram-Server Roundtrip)
Kosten    : $0 (Telegram ist kostenlos)
Accounts  : Telegram Bot bereits im Projekt vorhanden (.env)
Zuverlässig: Sehr hoch (Telegram ist hochverfügbar)
Vorteile  : Bereits integriert, kein neuer Account, funktioniert überall
            Human-readable Messages, manuelle Kontrolle möglich
Nachteile : 200–800ms Latenz, Rate-Limit (30 Msg/s an einen Chat),
            Nur für 1h+ Timeframes sinnvoll, nicht für Sub-Sekunden-Trading

Architektur:
┌─────────────────────────────────────────────────────────┐
│  LOKAL                         COLAB                    │
│                                                         │
│  Telegram Bot                  Telegram Bot             │
│  sendet Marktdaten             pollt Updates            │
│  an Chat-ID                    (getUpdates API)         │
│       │                              │                  │
│       └──── Telegram Server ─────────┘                  │
│                                                         │
│  Kanal-Kodierung: JSON in Nachricht                     │
│  Format: #bt4t:signals {"action":"BUY","conf":0.8,...}  │
└─────────────────────────────────────────────────────────┘

Voraussetzung:
  TELEGRAM_BOT_TOKEN und TELEGRAM_CHAT_ID in .env setzen
  (Im Projekt bereits vorhanden, aber aktuell leer)

  Bot erstellen: t.me/BotFather → /newbot
  Chat-ID finden: t.me/userinfobot

Installation:
  pip install httpx  (bereits installiert)

Rate-Limits:
  Telegram erlaubt max. 30 Nachrichten/Sekunde an einen Chat.
  Für 1h-Bars: 1 Msg/30min → völlig unkritisch.
  Für 1m-Bars: könnte Rate-Limit treffen bei vielen Kanälen.

Sicherheit:
  Nachrichten sind Klartext auf Telegram-Servern.
  Für Signale mit kleinen Positionsgrößen akzeptabel.
  Keine API-Keys oder sensible Daten senden.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from collections import defaultdict
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional, Set

try:
    import httpx

    _HTTPX_OK = True
except ImportError:
    _HTTPX_OK = False

try:
    from loguru import logger
except ImportError:
    import logging

    logger = logging.getLogger("transport_telegram")

from colab_bridge.transport_base import TransportBase

# ── Nachrichtenformat ─────────────────────────────────────────────────────────
# Alle Nachrichten haben dieses Format:
# #bt4t:{channel} {json_payload}
#
# Beispiel:
# #bt4t:signals {"action":"BUY","confidence":0.82,"timestamp_utc":"..."}
#
# Das # am Anfang macht es zu einem Hashtag → einfaches Filtern.

MSG_PREFIX = "#bt4t:"
MAX_MSG_LEN = 4096  # Telegram Limit
POLL_INTERVAL_S = 2.0  # Update-Polling-Intervall
RATE_LIMIT_S = 0.1  # Min. 100ms zwischen Nachrichten (10 Msg/s max)


class TelegramTransport(TransportBase):
    """
    Telegram Bot API als bidirektionaler Nachrichtenbus.

    Funktioniert für BEIDE Seiten (lokal und Colab) identisch.
    Beide Seiten benutzen denselben Bot-Token und Chat-ID.

    Sender:  sendet JSON-Nachricht mit Kanal-Hashtag
    Empfänger: pollt getUpdates und filtert nach Hashtag

    Verwendung (lokal UND in Colab identisch):
        transport = TelegramTransport(
            bot_token="8512...",
            chat_id="2028041322",
        )
        await transport.connect()
        await transport.subscribe("bt4t:signals", on_signal)
        await transport.publish("bt4t:market:BTCUSDT", market_data)
    """

    def __init__(
        self,
        bot_token: str = "",
        chat_id: str = "",
        poll_interval_s: float = POLL_INTERVAL_S,
    ):
        if not _HTTPX_OK:
            raise ImportError("pip install httpx")

        # Aus .env laden falls nicht angegeben
        self.bot_token = bot_token or os.getenv("TELEGRAM_BOT_TOKEN", "")
        self.chat_id = chat_id or os.getenv("TELEGRAM_CHAT_ID", "")

        if not self.bot_token:
            raise ValueError(
                "TELEGRAM_BOT_TOKEN fehlt!\n"
                "1. Bot erstellen: t.me/BotFather → /newbot\n"
                "2. In .env: TELEGRAM_BOT_TOKEN=dein_token\n"
                "3. TELEGRAM_CHAT_ID=deine_chat_id"
            )

        self._base_url = f"https://api.telegram.org/bot{self.bot_token}"
        self.poll_interval = poll_interval_s

        self._client: Optional[httpx.AsyncClient] = None
        self._callbacks: Dict[str, List[Callable]] = defaultdict(list)
        self._update_id: int = 0  # Letzter verarbeiteter Update-ID
        self._seen_ids: Set[int] = set()  # Deduplizierung
        self._poll_task: Optional[asyncio.Task] = None
        self._running = False
        self._last_send_time: float = 0.0

    @property
    def name(self) -> str:
        return "TelegramBot"

    @property
    def latency_class(self) -> str:
        return "sub-second"  # 200–800ms

    async def connect(self) -> None:
        """Verbindung testen und Poll-Loop starten."""
        self._client = httpx.AsyncClient(timeout=15.0)
        # Bot-Info abrufen (Verbindungstest)
        resp = await self._client.get(f"{self._base_url}/getMe")
        data = resp.json()
        if not data.get("ok"):
            raise ConnectionError(f"Telegram Bot Fehler: {data.get('description')}")
        bot_name = data["result"]["username"]
        self._running = True
        self._poll_task = asyncio.create_task(self._poll_loop())
        logger.success(f"[Telegram] Verbunden als @{bot_name} | Chat: {self.chat_id}")

    async def disconnect(self) -> None:
        self._running = False
        if self._poll_task:
            self._poll_task.cancel()
        if self._client:
            await self._client.aclose()
        logger.info("[Telegram] Getrennt")

    async def publish(self, channel: str, payload: dict) -> None:
        """
        Sendet Nachricht als Telegram-Nachricht.

        Format: #bt4t:{channel} {json}
        Lange Nachrichten werden gekürzt (Marktdaten ohne close_60 Array).
        """
        # close_60 Array kürzen (zu lang für Telegram)
        compact = {k: v for k, v in payload.items() if k != "close_60"}
        compact["_ch"] = channel  # Kanal im Payload für Deduplizierung

        msg_text = f"{MSG_PREFIX}{channel} {json.dumps(compact, default=str)}"

        # Nachricht kürzen wenn nötig
        if len(msg_text) > MAX_MSG_LEN:
            msg_text = msg_text[: MAX_MSG_LEN - 10] + "...[CUT]"

        # Rate-Limiting: min. 100ms zwischen Nachrichten
        elapsed = time.time() - self._last_send_time
        if elapsed < RATE_LIMIT_S:
            await asyncio.sleep(RATE_LIMIT_S - elapsed)

        resp = await self._client.post(
            f"{self._base_url}/sendMessage",
            json={
                "chat_id": self.chat_id,
                "text": msg_text,
                "disable_notification": True,  # Kein Ton
            },
        )
        self._last_send_time = time.time()

        if resp.status_code != 200:
            logger.warning(f"[Telegram] sendMessage Fehler: {resp.text[:200]}")
        else:
            logger.debug(f"[Telegram] PUBLISH {channel} ({len(msg_text)} chars)")

    async def subscribe(self, channel: str, callback: Callable[[dict], None]) -> None:
        """Registriert Callback für eingehende Nachrichten auf Kanal."""
        self._callbacks[channel].append(callback)
        logger.debug(f"[Telegram] Abonniert: {channel}")

    # ── Poll-Loop ─────────────────────────────────────────────────────────────

    async def _poll_loop(self) -> None:
        """
        Pollt Telegram getUpdates und dispatcht Nachrichten.

        getUpdates long-polling: Telegram wartet bis zu 30s auf neue Updates.
        Effektiv: sofortige Benachrichtigung bei neuer Nachricht.
        """
        while self._running:
            try:
                resp = await self._client.get(
                    f"{self._base_url}/getUpdates",
                    params={
                        "offset": self._update_id + 1,
                        "timeout": 20,  # Long-polling: 20s warten
                        "allowed_updates": json.dumps(["message"]),
                    },
                    timeout=25.0,
                )
                data = resp.json()
                if not data.get("ok"):
                    await asyncio.sleep(self.poll_interval)
                    continue

                for update in data.get("result", []):
                    self._update_id = max(self._update_id, update["update_id"])
                    await self._process_update(update)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f"[Telegram] Poll Fehler: {e}")
                await asyncio.sleep(self.poll_interval)

    async def _process_update(self, update: dict) -> None:
        """Verarbeitet ein Telegram-Update und dispatcht Callbacks."""
        msg = update.get("message", {})
        text = msg.get("text", "")

        if not text.startswith(MSG_PREFIX):
            return  # Keine bt4t-Nachricht

        # Deduplizierung
        update_id = update["update_id"]
        if update_id in self._seen_ids:
            return
        self._seen_ids.add(update_id)
        if len(self._seen_ids) > 500:
            self._seen_ids = set(list(self._seen_ids)[-200:])

        # Kanal und Payload extrahieren
        try:
            # Format: "#bt4t:channel {json}"
            without_prefix = text[len(MSG_PREFIX) :]  # "channel {json}"
            space_idx = without_prefix.index(" ")
            channel = without_prefix[:space_idx]
            json_str = without_prefix[space_idx + 1 :]

            payload = json.loads(json_str)

            # Callbacks aufrufen
            for cb in self._callbacks.get(channel, []):
                cb(payload)

            logger.debug(f"[Telegram] RECEIVED {channel}")

        except (ValueError, json.JSONDecodeError) as e:
            logger.debug(f"[Telegram] Parse-Fehler: {e} | text={text[:100]}")

    # ── Hilfsmethoden ─────────────────────────────────────────────────────────

    async def send_alert(self, text: str, level: str = "INFO") -> None:
        """
        Sendet eine menschenlesbare Alert-Nachricht (nicht als bt4t-Kanal).
        Für manuelle Benachrichtigungen und Kontrolle.
        """
        emoji = {"INFO": "ℹ️", "WARNING": "⚠️", "CRITICAL": "🚨", "SUCCESS": "✅"}.get(
            level, "📊"
        )
        await self._client.post(
            f"{self._base_url}/sendMessage",
            json={
                "chat_id": self.chat_id,
                "text": f"{emoji} *BT4T [{level}]*\n{text}",
                "parse_mode": "Markdown",
            },
        )

    @staticmethod
    def from_env() -> "TelegramTransport":
        """Erstellt Transport aus .env Variablen."""
        return TelegramTransport(
            bot_token=os.getenv("TELEGRAM_BOT_TOKEN", ""),
            chat_id=os.getenv("TELEGRAM_CHAT_ID", ""),
        )
