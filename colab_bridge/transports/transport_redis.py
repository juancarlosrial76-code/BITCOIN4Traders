"""
Transport Option 1: Redis Pub/Sub + Cloudflare Tunnel
======================================================

Latency  : 30–150 ms  (Redis ~1ms local + Tunnel overhead ~30–100ms)
Cost     : $0 (Redis local + Cloudflare Tunnel free)
Accounts : No external accounts needed
Reliable : Very high (Redis in-memory, CF stable)
Pros     : Fastest option, no third-party, full control
Cons     : Redis must run locally, Cloudflare Tunnel must be active

Architecture:
┌─────────────────────────────────────────────────────────┐
│  LOCAL                         COLAB                    │
│                                                         │
│  Redis Server :6379             HTTP Client             │
│       │                              │                  │
│  Cloudflare Tunnel   ◄──────────────┘                  │
│  (local port 6379                                       │
│   as TCP or HTTP proxy)                                 │
│                                                         │
│  Channels = Redis Pub/Sub Topics                        │
│  bt4t:signals, bt4t:market:BTCUSDT, ...                 │
└─────────────────────────────────────────────────────────┘

IMPORTANT: Redis does not support native TCP tunneling over HTTP.
Therefore Redis is exposed via a minimal HTTP proxy (FastAPI),
which Cloudflare then tunnels. Colab communicates with the HTTP proxy.

HTTP Proxy Endpoints (local, reachable via Cloudflare):
  POST /publish         { channel, payload }  → Redis PUBLISH
  GET  /subscribe/{ch}  Server-Sent Events    → Redis SUBSCRIBE
  GET  /poll/{ch}       Last N messages       → Redis LRANGE (simpler for Colab)

Installation:
  Local (one-time):
    sudo apt install redis-server   # or: brew install redis
    pip install redis aioredis fastapi uvicorn httpx

  Colab:
    !pip install httpx

Start Cloudflare Tunnel (separate terminal):
  ./cloudflared tunnel --url http://localhost:8766
"""

from __future__ import annotations

import asyncio
import json
import time
from collections import defaultdict
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional

try:
    import redis.asyncio as aioredis

    _REDIS_OK = True
except ImportError:
    try:
        import aioredis

        _REDIS_OK = True
    except ImportError:
        _REDIS_OK = False

try:
    import httpx

    _HTTPX_OK = True
except ImportError:
    _HTTPX_OK = False

try:
    from fastapi import FastAPI, Request
    from fastapi.responses import StreamingResponse
    import uvicorn

    _FASTAPI_OK = True
except ImportError:
    _FASTAPI_OK = False

try:
    from loguru import logger
except ImportError:
    import logging

    logger = logging.getLogger("transport_redis")

from colab_bridge.transport_base import TransportBase

PROXY_PORT = 8766  # HTTP proxy port (Cloudflare tunnels this)


# ══════════════════════════════════════════════════════════════════════════════
# LOCAL: Redis Publisher + HTTP Proxy Server
# ══════════════════════════════════════════════════════════════════════════════


class RedisTransportLocal(TransportBase):
    """
    Local side: Redis Pub/Sub + FastAPI HTTP proxy.

    Redis PUBLISH sends to local subscribers and stores
    messages in a Redis list (for Colab polling).
    FastAPI exposes Redis via HTTP for Colab.

    Usage:
        transport = RedisTransportLocal(redis_url="redis://localhost:6379")
        await transport.connect()
        await transport.publish("bt4t:signals", {"action": "BUY", ...})
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        proxy_port: int = PROXY_PORT,
        list_maxlen: int = 200,  # How many messages to buffer per channel
    ):
        if not _REDIS_OK:
            raise ImportError("pip install redis  (or: pip install aioredis)")
        self.redis_url = redis_url
        self.proxy_port = proxy_port
        self.list_maxlen = list_maxlen
        self._redis: Optional[aioredis.Redis] = None
        self._pubsub = None
        self._callbacks: Dict[str, List[Callable]] = defaultdict(list)
        self._listener_task: Optional[asyncio.Task] = None
        self._proxy_server: Optional[uvicorn.Server] = None

    @property
    def name(self) -> str:
        return "Redis+CloudflareTunnel"

    @property
    def latency_class(self) -> str:
        return "ms"  # 30–150ms total latency

    async def connect(self) -> None:
        """Connects to Redis and starts HTTP proxy."""
        self._redis = aioredis.from_url(self.redis_url, decode_responses=True)
        await self._redis.ping()
        logger.success(f"[Redis] Connected: {self.redis_url}")

        # Start HTTP proxy (Colab communicates through this)
        if _FASTAPI_OK:
            await self._start_proxy()

    async def disconnect(self) -> None:
        if self._listener_task:
            self._listener_task.cancel()
        if self._redis:
            await self._redis.aclose()
        logger.info("[Redis] Disconnected")

    async def publish(self, channel: str, payload: dict) -> None:
        """
        Publishes to Redis channel AND stores in Redis list (for Colab polling).

        Two parallel mechanisms:
          1. Redis PUBLISH  → for local subscribers (immediate)
          2. Redis LPUSH    → for Colab HTTP poll (buffered, FIFO)
        """
        msg = self.encode(payload)
        # 1. Redis Pub/Sub (for local async subscribers)
        await self._redis.publish(channel, msg)
        # 2. Redis List (FIFO queue for Colab polling)
        list_key = f"bt4t:queue:{channel}"
        await self._redis.lpush(list_key, msg)
        await self._redis.ltrim(list_key, 0, self.list_maxlen - 1)  # Enforce max length
        logger.debug(f"[Redis] PUBLISH {channel} ({len(msg)} bytes)")

    async def subscribe(self, channel: str, callback: Callable[[dict], None]) -> None:
        """Registers callback for Redis Pub/Sub."""
        self._callbacks[channel].append(callback)
        if self._pubsub is None:
            self._pubsub = self._redis.pubsub()
        await self._pubsub.subscribe(channel)
        if self._listener_task is None or self._listener_task.done():
            self._listener_task = asyncio.create_task(self._listener_loop())
        logger.debug(f"[Redis] Subscribed: {channel}")

    async def _listener_loop(self) -> None:
        """Listens to Redis Pub/Sub and dispatches callbacks."""
        async for message in self._pubsub.listen():
            if message["type"] != "message":
                continue
            channel = message["channel"]
            try:
                payload = self.decode(message["data"])
                for cb in self._callbacks.get(channel, []):
                    cb(payload)
            except Exception as e:
                logger.warning(f"[Redis] Listener error: {e}")

    # ── HTTP proxy for Colab ──────────────────────────────────────────────────

    async def _start_proxy(self) -> None:
        """
        Starts a minimal FastAPI HTTP proxy.
        Cloudflare Tunnel exposes this to Colab.

        Endpoints:
          POST /publish              Colab → publish locally
          GET  /poll/{channel}       Colab polls new messages
          GET  /health               Health check
        """
        app = FastAPI(title="BT4T Redis Proxy")
        redis_ref = self._redis
        list_maxlen = self.list_maxlen

        @app.get("/health")
        async def health():
            return {"status": "ok", "transport": "redis"}

        @app.post("/publish")
        async def proxy_publish(request: Request):
            """Colab sends messages → published to Redis."""
            body = await request.json()
            channel = body.get("channel", "")
            payload = body.get("payload", {})
            if not channel:
                return {"error": "channel required"}
            msg = json.dumps(payload, default=str)
            await redis_ref.publish(channel, msg)
            list_key = f"bt4t:queue:{channel}"
            await redis_ref.lpush(list_key, msg)
            await redis_ref.ltrim(list_key, 0, list_maxlen - 1)
            return {"status": "ok", "channel": channel}

        @app.get("/poll/{channel:path}")
        async def proxy_poll(channel: str, n: int = 10):
            """
            Colab polls new messages from a channel.

            Returns the last n messages (newest first).
            Recommended poll interval: 1–5 seconds.
            """
            list_key = f"bt4t:queue:{channel}"
            raw_list = await redis_ref.lrange(list_key, 0, n - 1)
            messages = []
            for raw in raw_list:
                try:
                    messages.append(json.loads(raw))
                except Exception:
                    pass
            return {"channel": channel, "count": len(messages), "messages": messages}

        @app.delete("/poll/{channel:path}")
        async def proxy_ack(channel: str, n: int = 1):
            """Colab acknowledges n messages as processed (removes them from queue)."""
            list_key = f"bt4t:queue:{channel}"
            for _ in range(n):
                await redis_ref.rpop(list_key)
            return {"status": "ok", "acked": n}

        config = uvicorn.Config(
            app, host="0.0.0.0", port=self.proxy_port, log_level="warning"
        )
        server = uvicorn.Server(config)
        asyncio.create_task(server.serve())
        logger.success(f"[Redis] HTTP proxy started on ::{self.proxy_port}")
        logger.info(
            f"[Redis] Cloudflare Tunnel: ./cloudflared tunnel --url http://localhost:{self.proxy_port}"
        )


# ══════════════════════════════════════════════════════════════════════════════
# COLAB: HTTP Poll Client (no Redis client needed)
# ══════════════════════════════════════════════════════════════════════════════


class RedisTransportColab(TransportBase):
    """
    Colab side: Polls the local Redis HTTP proxy via Cloudflare Tunnel.

    No Redis client needed in Colab — only httpx.
    Latency = poll interval + network overhead (~30–150ms at 1s polling).

    Usage in Colab:
        transport = RedisTransportColab(
            proxy_url="https://abc.trycloudflare.com",
            poll_interval_s=1.0,
        )
        await transport.connect()
        await transport.subscribe("bt4t:market:BTCUSDT", on_market_data)
        await transport.publish("bt4t:signals", {"action": "BUY", ...})
    """

    def __init__(
        self,
        proxy_url: str,  # Cloudflare Tunnel URL, e.g. https://abc.trycloudflare.com
        poll_interval_s: float = 1.0,
    ):
        if not _HTTPX_OK:
            raise ImportError("pip install httpx")
        self.proxy_url = proxy_url.rstrip("/")
        self.poll_interval = poll_interval_s
        self._client: Optional[httpx.AsyncClient] = None
        self._callbacks: Dict[str, List[Callable]] = defaultdict(list)
        self._poll_tasks: Dict[str, asyncio.Task] = {}
        self._running = False

    @property
    def name(self) -> str:
        return "RedisProxy(Colab)"

    @property
    def latency_class(self) -> str:
        return "ms"

    async def connect(self) -> None:
        self._client = httpx.AsyncClient(timeout=10.0)
        # Health check
        resp = await self._client.get(f"{self.proxy_url}/health")
        resp.raise_for_status()
        self._running = True
        logger.success(f"[Redis/Colab] Connected to proxy: {self.proxy_url}")

    async def disconnect(self) -> None:
        self._running = False
        for task in self._poll_tasks.values():
            task.cancel()
        if self._client:
            await self._client.aclose()

    async def publish(self, channel: str, payload: dict) -> None:
        """Sends message via HTTP proxy → Redis → local subscriber."""
        resp = await self._client.post(
            f"{self.proxy_url}/publish",
            json={"channel": channel, "payload": payload},
        )
        resp.raise_for_status()
        logger.debug(f"[Redis/Colab] PUBLISH {channel}")

    async def subscribe(self, channel: str, callback: Callable[[dict], None]) -> None:
        """Starts poll loop for channel."""
        self._callbacks[channel].append(callback)
        if channel not in self._poll_tasks:
            self._poll_tasks[channel] = asyncio.create_task(self._poll_loop(channel))
        logger.debug(f"[Redis/Colab] Subscribed: {channel}")

    async def _poll_loop(self, channel: str) -> None:
        """Polls new messages and dispatches callbacks."""
        encoded_ch = channel.replace(":", "%3A")
        last_seen: set = set()  # Deduplication

        while self._running:
            try:
                resp = await self._client.get(
                    f"{self.proxy_url}/poll/{encoded_ch}",
                    params={"n": 5},
                )
                if resp.status_code == 200:
                    data = resp.json()
                    messages = data.get("messages", [])
                    # Process newest message (index 0 = newest)
                    for msg in reversed(messages):
                        msg_id = msg.get("timestamp", str(msg))
                        if msg_id not in last_seen:
                            last_seen.add(msg_id)
                            if len(last_seen) > 100:
                                # Remove old IDs
                                last_seen = set(list(last_seen)[-50:])
                            for cb in self._callbacks.get(channel, []):
                                cb(msg)
            except Exception as e:
                logger.warning(f"[Redis/Colab] Poll error {channel}: {e}")

            await asyncio.sleep(self.poll_interval)
