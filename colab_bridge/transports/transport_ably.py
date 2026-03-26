"""
Transport Option 4: Ably Pub/Sub (external service)
====================================================

Latency  : 50–150 ms  (Ably Global Edge Network)
Cost     : $0 up to 6 million messages/month (Free Tier)
           @ 1 Msg/30s = ~86,400 Msg/month → Free Tier is sufficient
Accounts : Ably account required (https://ably.com → free)
Reliable : Very high (global CDN, 99.999% SLA)
Pros     : No tunnel needed, both sides connect outward,
           message buffer (last 100 msgs), WebSocket-based
Cons     : External third-party, account required, internet-dependent

Free Tier Limits:
  6 million messages/month
  100 concurrent connections
  Message history: last 100 per channel
  No SLA in Free Tier (only paid)

Create account:
  1. https://ably.com → Sign Up (free)
  2. Dashboard → Create App → API Keys
  3. Copy Root Key (has all permissions)
  4. In .env: ABLY_API_KEY=your_key:your_secret

Installation:
  pip install ably   (already installed)

Note: Free Tier rate limit = 250 requests/second.
At 30s poll interval and 5 channels = 0.16 req/s → well below limit.

This transport is the only option that works WITHOUT a Cloudflare Tunnel.
Both sides (local + Colab) connect directly to the Ably server.
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

    Identical for BOTH sides (local and Colab).
    Both sides use the same API key.

    Data path:
      Local → Ably → Colab  (market data, ~50–150ms)
      Colab → Ably → Local  (signals, ~50–150ms)

    Usage (local AND Colab identical):
        import os
        transport = AblyTransport(api_key=os.getenv("ABLY_API_KEY"))
        await transport.connect()
        await transport.subscribe("bt4t:signals", on_signal)
        await transport.publish("bt4t:market:BTCUSDT", market_data)
    """

    def __init__(self, api_key: str = ""):
        if not _ABLY_OK:
            raise ImportError("pip install ably\nAccount: https://ably.com (free)")
        import os

        # Get API key from Secrets Manager (with fallback to environment)
        from src.config import get_ably_key

        secrets_key = get_ably_key()
        self.api_key = api_key or secrets_key or os.getenv("ABLY_API_KEY", "")
        if not self.api_key:
            raise ValueError(
                "ABLY_API_KEY missing!\n"
                "1. https://ably.com → Sign Up\n"
                "2. Dashboard → App → API Keys → Copy Root Key\n"
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
        logger.success(f"[Ably] Connected | Key: {self.api_key[:15]}...")

    async def disconnect(self) -> None:
        if self._ably:
            await self._ably.close()
        logger.info("[Ably] Disconnected")

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
                logger.warning(f"[Ably] Decode error: {e}")

        ch = self._ably.channels.get(channel)
        await ch.subscribe(_wrapper)
        logger.debug(f"[Ably] Subscribed: {channel}")
