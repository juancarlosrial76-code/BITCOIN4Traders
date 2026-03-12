"""
Transport Option 1: Redis Pub/Sub + Cloudflare Tunnel
======================================================

Latenz    : 30–150 ms  (Redis ~1ms lokal + Tunnel-Overhead ~30–100ms)
Kosten    : $0 (Redis lokal + Cloudflare Tunnel kostenlos)
Accounts  : Keine externen Accounts nötig
Zuverlässig: Sehr hoch (Redis in-memory, CF stabil)
Vorteile  : Schnellste Option, kein Drittanbieter, volle Kontrolle
Nachteile : Redis muss lokal laufen, Cloudflare Tunnel muss aktiv sein

Architektur:
┌─────────────────────────────────────────────────────────┐
│  LOKAL                         COLAB                    │
│                                                         │
│  Redis Server :6379             HTTP-Client             │
│       │                              │                  │
│  Cloudflare Tunnel   ◄──────────────┘                  │
│  (lokaler Port 6379                                     │
│   als TCP oder HTTP-Proxy)                              │
│                                                         │
│  Channels = Redis Pub/Sub Topics                        │
│  bt4t:signals, bt4t:market:BTCUSDT, ...                 │
└─────────────────────────────────────────────────────────┘

WICHTIG: Redis unterstützt kein natives TCP-Tunneling über HTTP.
Daher wird Redis über einen minimalen HTTP-Proxy (FastAPI) exponiert,
den Cloudflare dann tunnelt. Colab spricht mit dem HTTP-Proxy.

HTTP-Proxy Endpoints (lokal, via Cloudflare erreichbar):
  POST /publish         { channel, payload }  → Redis PUBLISH
  GET  /subscribe/{ch}  Server-Sent Events    → Redis SUBSCRIBE
  GET  /poll/{ch}       Letzte N Nachrichten  → Redis LRANGE (einfacher für Colab)

Installation:
  Lokal (einmalig):
    sudo apt install redis-server   # oder: brew install redis
    pip install redis aioredis fastapi uvicorn httpx

  Colab:
    !pip install httpx

Cloudflare Tunnel starten (separates Terminal):
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

PROXY_PORT = 8766  # HTTP-Proxy Port (Cloudflare tunnelt diesen)


# ══════════════════════════════════════════════════════════════════════════════
# LOKAL: Redis Publisher + HTTP-Proxy Server
# ══════════════════════════════════════════════════════════════════════════════


class RedisTransportLocal(TransportBase):
    """
    Lokale Seite: Redis Pub/Sub + FastAPI HTTP-Proxy.

    Redis PUBLISH sendet an lokale Subscriber und speichert
    Nachrichten in einer Redis-List (für Colab-Polling).
    FastAPI exponiert Redis über HTTP für Colab.

    Verwendung:
        transport = RedisTransportLocal(redis_url="redis://localhost:6379")
        await transport.connect()
        await transport.publish("bt4t:signals", {"action": "BUY", ...})
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        proxy_port: int = PROXY_PORT,
        list_maxlen: int = 200,  # Wie viele Nachrichten pro Kanal gepuffert werden
    ):
        if not _REDIS_OK:
            raise ImportError("pip install redis  (oder: pip install aioredis)")
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
        return "ms"  # 30–150ms Gesamtlatenz

    async def connect(self) -> None:
        """Verbindet mit Redis und startet HTTP-Proxy."""
        self._redis = aioredis.from_url(self.redis_url, decode_responses=True)
        await self._redis.ping()
        logger.success(f"[Redis] Verbunden: {self.redis_url}")

        # HTTP-Proxy starten (Colab spricht hierüber)
        if _FASTAPI_OK:
            await self._start_proxy()

    async def disconnect(self) -> None:
        if self._listener_task:
            self._listener_task.cancel()
        if self._redis:
            await self._redis.aclose()
        logger.info("[Redis] Getrennt")

    async def publish(self, channel: str, payload: dict) -> None:
        """
        Publisht auf Redis-Kanal UND speichert in Redis-List (für Colab-Polling).

        Zwei Mechanismen parallel:
          1. Redis PUBLISH  → für lokale Subscriber (sofort)
          2. Redis LPUSH    → für Colab HTTP-Poll (gepuffert, FIFO)
        """
        msg = self.encode(payload)
        # 1. Redis Pub/Sub (für lokale async-Subscriber)
        await self._redis.publish(channel, msg)
        # 2. Redis List (FIFO-Queue für Colab-Polling)
        list_key = f"bt4t:queue:{channel}"
        await self._redis.lpush(list_key, msg)
        await self._redis.ltrim(list_key, 0, self.list_maxlen - 1)  # Maxlänge einhalten
        logger.debug(f"[Redis] PUBLISH {channel} ({len(msg)} bytes)")

    async def subscribe(self, channel: str, callback: Callable[[dict], None]) -> None:
        """Registriert Callback für Redis Pub/Sub."""
        self._callbacks[channel].append(callback)
        if self._pubsub is None:
            self._pubsub = self._redis.pubsub()
        await self._pubsub.subscribe(channel)
        if self._listener_task is None or self._listener_task.done():
            self._listener_task = asyncio.create_task(self._listener_loop())
        logger.debug(f"[Redis] Abonniert: {channel}")

    async def _listener_loop(self) -> None:
        """Hört auf Redis Pub/Sub und dispatcht Callbacks."""
        async for message in self._pubsub.listen():
            if message["type"] != "message":
                continue
            channel = message["channel"]
            try:
                payload = self.decode(message["data"])
                for cb in self._callbacks.get(channel, []):
                    cb(payload)
            except Exception as e:
                logger.warning(f"[Redis] Listener Fehler: {e}")

    # ── HTTP-Proxy für Colab ──────────────────────────────────────────────────

    async def _start_proxy(self) -> None:
        """
        Startet einen minimalen FastAPI HTTP-Proxy.
        Cloudflare Tunnel exponiert diesen für Colab.

        Endpoints:
          POST /publish              Colab → lokal publishen
          GET  /poll/{channel}       Colab pollt neue Nachrichten
          GET  /health               Healthcheck
        """
        app = FastAPI(title="BT4T Redis Proxy")
        redis_ref = self._redis
        list_maxlen = self.list_maxlen

        @app.get("/health")
        async def health():
            return {"status": "ok", "transport": "redis"}

        @app.post("/publish")
        async def proxy_publish(request: Request):
            """Colab sendet Nachrichten → werden in Redis gepublisht."""
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
            Colab pollt neue Nachrichten von einem Kanal.

            Gibt die letzten n Nachrichten zurück (neueste zuerst).
            Empfohlenes Poll-Intervall: 1–5 Sekunden.
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
            """Colab bestätigt n Nachrichten als verarbeitet (löscht sie aus Queue)."""
            list_key = f"bt4t:queue:{channel}"
            for _ in range(n):
                await redis_ref.rpop(list_key)
            return {"status": "ok", "acked": n}

        config = uvicorn.Config(
            app, host="0.0.0.0", port=self.proxy_port, log_level="warning"
        )
        server = uvicorn.Server(config)
        asyncio.create_task(server.serve())
        logger.success(f"[Redis] HTTP-Proxy gestartet auf ::{self.proxy_port}")
        logger.info(
            f"[Redis] Cloudflare Tunnel: ./cloudflared tunnel --url http://localhost:{self.proxy_port}"
        )


# ══════════════════════════════════════════════════════════════════════════════
# COLAB: HTTP-Poll Client (kein Redis-Client nötig)
# ══════════════════════════════════════════════════════════════════════════════


class RedisTransportColab(TransportBase):
    """
    Colab-Seite: Pollt den lokalen Redis HTTP-Proxy via Cloudflare Tunnel.

    Kein Redis-Client nötig in Colab — nur httpx.
    Latenz = Poll-Intervall + Netzwerk-Overhead (~30–150ms bei 1s-Polling).

    Verwendung in Colab:
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
        proxy_url: str,  # Cloudflare Tunnel URL, z.B. https://abc.trycloudflare.com
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
        # Healthcheck
        resp = await self._client.get(f"{self.proxy_url}/health")
        resp.raise_for_status()
        self._running = True
        logger.success(f"[Redis/Colab] Verbunden mit Proxy: {self.proxy_url}")

    async def disconnect(self) -> None:
        self._running = False
        for task in self._poll_tasks.values():
            task.cancel()
        if self._client:
            await self._client.aclose()

    async def publish(self, channel: str, payload: dict) -> None:
        """Sendet Nachricht über HTTP-Proxy → Redis → lokaler Subscriber."""
        resp = await self._client.post(
            f"{self.proxy_url}/publish",
            json={"channel": channel, "payload": payload},
        )
        resp.raise_for_status()
        logger.debug(f"[Redis/Colab] PUBLISH {channel}")

    async def subscribe(self, channel: str, callback: Callable[[dict], None]) -> None:
        """Startet Poll-Loop für Kanal."""
        self._callbacks[channel].append(callback)
        if channel not in self._poll_tasks:
            self._poll_tasks[channel] = asyncio.create_task(self._poll_loop(channel))
        logger.debug(f"[Redis/Colab] Abonniert: {channel}")

    async def _poll_loop(self, channel: str) -> None:
        """Pollt neue Nachrichten und dispatcht Callbacks."""
        encoded_ch = channel.replace(":", "%3A")
        last_seen: set = set()  # Deduplizierung

        while self._running:
            try:
                resp = await self._client.get(
                    f"{self.proxy_url}/poll/{encoded_ch}",
                    params={"n": 5},
                )
                if resp.status_code == 200:
                    data = resp.json()
                    messages = data.get("messages", [])
                    # Neueste Nachricht verarbeiten (Index 0 = neueste)
                    for msg in reversed(messages):
                        msg_id = msg.get("timestamp", str(msg))
                        if msg_id not in last_seen:
                            last_seen.add(msg_id)
                            if len(last_seen) > 100:
                                # Alte IDs entfernen
                                last_seen = set(list(last_seen)[-50:])
                            for cb in self._callbacks.get(channel, []):
                                cb(msg)
            except Exception as e:
                logger.warning(f"[Redis/Colab] Poll Fehler {channel}: {e}")

            await asyncio.sleep(self.poll_interval)
