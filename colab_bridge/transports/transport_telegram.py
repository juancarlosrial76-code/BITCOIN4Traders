"""
Transport Option 2: Telegram Bot API as message bus
====================================================

Latency  : 200–800 ms  (Telegram server round trip)
Cost     : $0 (Telegram is free)
Accounts : Telegram bot already in project (.env)
Reliable : Very high (Telegram is highly available)
Pros     : Already integrated, no new account needed, works everywhere,
           human-readable messages, manual control possible
Cons     : 200–800ms latency, rate limit (30 msg/s to one chat),
           only useful for 1h+ timeframes, not for sub-second trading

Architecture:
┌─────────────────────────────────────────────────────────┐
│  LOCAL                         COLAB                    │
│                                                         │
│  Telegram Bot                  Telegram Bot             │
│  sends market data             polls updates            │
│  to chat ID                    (getUpdates API)         │
│       │                              │                  │
│       └──── Telegram Server ─────────┘                  │
│                                                         │
│  Channel encoding: JSON in message                      │
│  Format: #bt4t:signals {"action":"BUY","conf":0.8,...}  │
└─────────────────────────────────────────────────────────┘

Prerequisites:
  Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in .env
  (Already in project, but currently empty)

  Create bot: t.me/BotFather → /newbot
  Find chat ID: t.me/userinfobot

Installation:
  pip install httpx  (already installed)

Rate limits:
  Telegram allows max. 30 messages/second to one chat.
  For 1h bars: 1 msg/30min → completely uncritical.
  For 1m bars: could hit rate limit with many channels.

Security:
  Messages are plain text on Telegram servers.
  Acceptable for signals with small position sizes.
  Do not send API keys or sensitive data.
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

# ── Message format ────────────────────────────────────────────────────────────
# All messages have this format:
# #bt4t:{channel} {json_payload}
#
# Example:
# #bt4t:signals {"action":"BUY","confidence":0.82,"timestamp_utc":"..."}
#
# The # at the start makes it a hashtag → easy filtering.

MSG_PREFIX = "#bt4t:"
MAX_MSG_LEN = 4096  # Telegram limit
POLL_INTERVAL_S = 2.0  # Update polling interval
RATE_LIMIT_S = 0.1  # Min. 100ms between messages (10 msg/s max)


class TelegramTransport(TransportBase):
    """
    Telegram Bot API as bidirectional message bus.

    Works identically for BOTH sides (local and Colab).
    Both sides use the same bot token and chat ID.

    Sender:   sends JSON message with channel hashtag
    Receiver: polls getUpdates and filters by hashtag

    Usage (local AND in Colab identical):
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

        # Load from .env if not provided
        self.bot_token = bot_token or os.getenv("TELEGRAM_BOT_TOKEN", "")
        self.chat_id = chat_id or os.getenv("TELEGRAM_CHAT_ID", "")

        if not self.bot_token:
            raise ValueError(
                "TELEGRAM_BOT_TOKEN missing!\n"
                "1. Create bot: t.me/BotFather → /newbot\n"
                "2. In .env: TELEGRAM_BOT_TOKEN=your_token\n"
                "3. TELEGRAM_CHAT_ID=your_chat_id"
            )

        self._base_url = f"https://api.telegram.org/bot{self.bot_token}"
        self.poll_interval = poll_interval_s

        self._client: Optional[httpx.AsyncClient] = None
        self._callbacks: Dict[str, List[Callable]] = defaultdict(list)
        self._update_id: int = 0  # Last processed update ID
        self._seen_ids: Set[int] = set()  # Deduplication
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
        """Test connection and start poll loop."""
        self._client = httpx.AsyncClient(timeout=15.0)
        # Fetch bot info (connection test)
        resp = await self._client.get(f"{self._base_url}/getMe")
        data = resp.json()
        if not data.get("ok"):
            raise ConnectionError(f"Telegram Bot error: {data.get('description')}")
        bot_name = data["result"]["username"]
        self._running = True
        self._poll_task = asyncio.create_task(self._poll_loop())
        logger.success(f"[Telegram] Connected as @{bot_name} | Chat: {self.chat_id}")

    async def disconnect(self) -> None:
        self._running = False
        if self._poll_task:
            self._poll_task.cancel()
        if self._client:
            await self._client.aclose()
        logger.info("[Telegram] Disconnected")

    async def publish(self, channel: str, payload: dict) -> None:
        """
        Sends message as Telegram message.

        Format: #bt4t:{channel} {json}
        Long messages are truncated (market data without close_60 array).
        """
        # Truncate close_60 array (too long for Telegram)
        compact = {k: v for k, v in payload.items() if k != "close_60"}
        compact["_ch"] = channel  # Channel in payload for deduplication

        msg_text = f"{MSG_PREFIX}{channel} {json.dumps(compact, default=str)}"

        # Truncate message if needed
        if len(msg_text) > MAX_MSG_LEN:
            msg_text = msg_text[: MAX_MSG_LEN - 10] + "...[CUT]"

        # Rate limiting: min. 100ms between messages
        elapsed = time.time() - self._last_send_time
        if elapsed < RATE_LIMIT_S:
            await asyncio.sleep(RATE_LIMIT_S - elapsed)

        resp = await self._client.post(
            f"{self._base_url}/sendMessage",
            json={
                "chat_id": self.chat_id,
                "text": msg_text,
                "disable_notification": True,  # Silent
            },
        )
        self._last_send_time = time.time()

        if resp.status_code != 200:
            logger.warning(f"[Telegram] sendMessage error: {resp.text[:200]}")
        else:
            logger.debug(f"[Telegram] PUBLISH {channel} ({len(msg_text)} chars)")

    async def subscribe(self, channel: str, callback: Callable[[dict], None]) -> None:
        """Registers callback for incoming messages on channel."""
        self._callbacks[channel].append(callback)
        logger.debug(f"[Telegram] Subscribed: {channel}")

    # ── Poll loop ─────────────────────────────────────────────────────────────

    async def _poll_loop(self) -> None:
        """
        Polls Telegram getUpdates and dispatches messages.

        getUpdates long-polling: Telegram waits up to 30s for new updates.
        Effectively: immediate notification on new message.
        """
        while self._running:
            try:
                resp = await self._client.get(
                    f"{self._base_url}/getUpdates",
                    params={
                        "offset": self._update_id + 1,
                        "timeout": 20,  # Long-polling: wait 20s
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
                logger.warning(f"[Telegram] Poll error: {e}")
                await asyncio.sleep(self.poll_interval)

    async def _process_update(self, update: dict) -> None:
        """Processes a Telegram update and dispatches callbacks."""
        msg = update.get("message", {})
        text = msg.get("text", "")

        if not text.startswith(MSG_PREFIX):
            return  # Not a bt4t message

        # Deduplication
        update_id = update["update_id"]
        if update_id in self._seen_ids:
            return
        self._seen_ids.add(update_id)
        if len(self._seen_ids) > 500:
            self._seen_ids = set(list(self._seen_ids)[-200:])

        # Extract channel and payload
        try:
            # Format: "#bt4t:channel {json}"
            without_prefix = text[len(MSG_PREFIX) :]  # "channel {json}"
            space_idx = without_prefix.index(" ")
            channel = without_prefix[:space_idx]
            json_str = without_prefix[space_idx + 1 :]

            payload = json.loads(json_str)

            # Call callbacks
            for cb in self._callbacks.get(channel, []):
                cb(payload)

            logger.debug(f"[Telegram] RECEIVED {channel}")

        except (ValueError, json.JSONDecodeError) as e:
            logger.debug(f"[Telegram] Parse error: {e} | text={text[:100]}")

    # ── Helper methods ────────────────────────────────────────────────────────

    async def send_alert(self, text: str, level: str = "INFO") -> None:
        """
        Sends a human-readable alert message (not as a bt4t channel).
        For manual notifications and control.
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
        """Creates transport from .env variables."""
        return TelegramTransport(
            bot_token=os.getenv("TELEGRAM_BOT_TOKEN", ""),
            chat_id=os.getenv("TELEGRAM_CHAT_ID", ""),
        )
