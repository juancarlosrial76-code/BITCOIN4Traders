"""
PHASE 7 – BINANCE WEBSOCKET CONNECTOR
======================================
Production-grade connector with:
  - Automatic reconnect (exponential backoff, max 10 attempts)
  - Heartbeat / ping-pong monitoring
  - Subscription management (orderbook, trades, user data stream)
  - Thread-safe message dispatch
  - Structured error taxonomy
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import time
import urllib.parse
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Coroutine, Dict, List, Optional, Set

import aiohttp

logger = logging.getLogger("phase7.ws_connector")


# ─────────────────────────────────────────────
#  Error Taxonomy
# ─────────────────────────────────────────────


class ConnectorError(Exception):
    """Base class for all connector errors."""


class AuthenticationError(ConnectorError):
    """Invalid API key or signature."""


class RateLimitError(ConnectorError):
    """429 / 418 from Binance – back off immediately."""


class SubscriptionError(ConnectorError):
    """Failed to subscribe/unsubscribe a stream."""


class ConnectionLostError(ConnectorError):
    """WebSocket connection unexpectedly closed."""


# ─────────────────────────────────────────────
#  Connection State Machine
# ─────────────────────────────────────────────


class ConnState(Enum):
    DISCONNECTED = auto()
    CONNECTING = auto()
    CONNECTED = auto()
    RECONNECTING = auto()
    CLOSING = auto()
    CLOSED = auto()


# ─────────────────────────────────────────────
#  Reconnect Config
# ─────────────────────────────────────────────


@dataclass
class ReconnectPolicy:
    max_attempts: int = 10
    base_delay_s: float = 1.0
    max_delay_s: float = 60.0
    backoff_factor: float = 2.0
    jitter: bool = True  # add ±20 % random jitter

    def delay(self, attempt: int) -> float:
        import random

        d = min(
            self.base_delay_s * (self.backoff_factor**attempt), self.max_delay_s
        )  # Exponential backoff capped at max_delay_s
        if self.jitter:
            d *= random.uniform(
                0.8, 1.2
            )  # Add ±20% jitter to prevent thundering-herd reconnects
        return d


# ─────────────────────────────────────────────
#  Message Handler Registry
# ─────────────────────────────────────────────

Handler = Callable[[dict], Coroutine]


class HandlerRegistry:
    def __init__(self):
        self._handlers: Dict[str, List[Handler]] = {}

    def register(self, stream: str, handler: Handler) -> None:
        self._handlers.setdefault(stream, []).append(handler)
        logger.debug("Registered handler for stream '%s'", stream)

    def unregister(self, stream: str, handler: Handler) -> None:
        if stream in self._handlers:
            self._handlers[stream].remove(handler)

    async def dispatch(self, stream: str, data: dict) -> None:
        for h in self._handlers.get(
            stream, []
        ):  # Iterate all registered handlers for this stream
            try:
                await h(data)  # Await each async handler coroutine
            except Exception as exc:
                logger.error(
                    "Handler for '%s' raised: %s", stream, exc, exc_info=True
                )  # Isolate handler errors


# ─────────────────────────────────────────────
#  Core Connector
# ─────────────────────────────────────────────


class BinanceWSConnector:
    """
    Single multiplexed WebSocket to Binance combined stream endpoint.
    Manages: connection lifecycle, subscriptions, reconnect, heartbeat.
    """

    _BASE_URL = "wss://stream.binance.com:9443/stream"
    _REST_BASE = "https://api.binance.com"
    _LISTEN_KEY_REFRESH_INTERVAL = 1_800  # 30 min in seconds

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        reconnect_policy: Optional[ReconnectPolicy] = None,
        paper_trading: bool = False,
    ):
        self._api_key = api_key
        self._api_secret = api_secret
        self._policy = reconnect_policy or ReconnectPolicy()
        self._paper = paper_trading

        self._state = ConnState.DISCONNECTED
        self._ws: Optional[aiohttp.ClientWebSocketResponse] = None
        self._session: Optional[aiohttp.ClientSession] = None

        self._subscriptions: Set[str] = set()
        self._registry = HandlerRegistry()

        self._reconnect_attempts = 0
        self._listen_key: Optional[str] = None
        self._req_id = 0

        self._recv_task: Optional[asyncio.Task] = None
        self._hb_task: Optional[asyncio.Task] = None
        self._lk_task: Optional[asyncio.Task] = None

    # ── Properties ──────────────────────────

    @property
    def state(self) -> ConnState:
        return self._state

    @property
    def is_connected(self) -> bool:
        return self._state == ConnState.CONNECTED

    # ── Public API ──────────────────────────

    async def connect(self) -> None:
        """Open connection and start background tasks."""
        if self._state not in (ConnState.DISCONNECTED, ConnState.RECONNECTING):
            return
        self._state = ConnState.CONNECTING
        self._session = aiohttp.ClientSession()
        await self._do_connect()

    async def disconnect(self) -> None:
        """Graceful shutdown."""
        self._state = ConnState.CLOSING
        await self._cancel_background_tasks()
        if self._ws and not self._ws.closed:
            await self._ws.close()
        if self._session and not self._session.closed:
            await self._session.close()
        self._state = ConnState.CLOSED
        logger.info("WebSocket connector closed cleanly.")

    def on(self, stream: str, handler: Handler) -> None:
        """Register an async callback for a specific stream."""
        self._registry.register(stream, handler)

    async def subscribe(self, streams: List[str]) -> None:
        """Subscribe to additional streams at runtime."""
        new = [s for s in streams if s not in self._subscriptions]
        if not new:
            return
        await self._send_request("SUBSCRIBE", new)
        self._subscriptions.update(new)
        logger.info("Subscribed: %s", new)

    async def unsubscribe(self, streams: List[str]) -> None:
        """Unsubscribe from streams."""
        existing = [s for s in streams if s in self._subscriptions]
        if not existing:
            return
        await self._send_request("UNSUBSCRIBE", existing)
        self._subscriptions -= set(existing)
        logger.info("Unsubscribed: %s", existing)

    # ── Private: Connection ─────────────────

    async def _do_connect(self) -> None:
        url = self._BASE_URL
        if self._subscriptions:
            params = "/".join(
                sorted(self._subscriptions)
            )  # Build combined stream URL from sorted subscriptions
            url = f"{url}?streams={params}"
        try:
            self._ws = await self._session.ws_connect(
                url,
                heartbeat=20,  # aiohttp sends WebSocket ping frame every 20 s to keep connection alive
                receive_timeout=30,  # Raise error if no message received for 30 s
            )
            self._state = ConnState.CONNECTED
            self._reconnect_attempts = 0  # Reset counter on successful connection
            logger.info(
                "WebSocket connected (%s streams active)", len(self._subscriptions)
            )

            # (Re-)start background tasks
            self._recv_task = asyncio.create_task(
                self._recv_loop(), name="ws_recv"
            )  # Incoming message loop
            self._hb_task = asyncio.create_task(
                self._heartbeat_loop(), name="ws_hb"
            )  # Connectivity watchdog

            # User data stream
            await self._start_user_stream()

        except Exception as exc:
            logger.error("Connection attempt failed: %s", exc)
            await self._handle_reconnect()

    async def _handle_reconnect(self) -> None:
        if self._state == ConnState.CLOSING:
            return
        if self._reconnect_attempts >= self._policy.max_attempts:
            self._state = ConnState.CLOSED
            logger.critical(
                "Max reconnect attempts (%d) reached. Giving up.",
                self._policy.max_attempts,
            )
            raise ConnectionLostError("Exceeded max reconnect attempts.")

        self._state = ConnState.RECONNECTING
        delay = self._policy.delay(self._reconnect_attempts)
        self._reconnect_attempts += 1
        logger.warning(
            "Reconnecting in %.1f s (attempt %d/%d)…",
            delay,
            self._reconnect_attempts,
            self._policy.max_attempts,
        )
        await asyncio.sleep(delay)
        await self._do_connect()

    # ── Private: Receive Loop ───────────────

    async def _recv_loop(self) -> None:
        assert self._ws is not None
        try:
            async for msg in self._ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    await self._handle_message(msg.data)
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    logger.error("WS error frame: %s", msg.data)
                    break
                elif msg.type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.CLOSING):
                    logger.warning("WS closed by remote (code=%s).", msg.data)
                    break
        except asyncio.CancelledError:
            return
        except Exception as exc:
            logger.error("Receive loop crashed: %s", exc, exc_info=True)
        finally:
            if self._state not in (ConnState.CLOSING, ConnState.CLOSED):
                await self._handle_reconnect()

    async def _handle_message(self, raw: str) -> None:
        try:
            data = json.loads(raw)  # Parse JSON WebSocket frame
        except json.JSONDecodeError:
            logger.warning("Non-JSON message received: %s", raw[:200])
            return

        # Binance combined stream wraps messages as {"stream": "...", "data": {...}}
        stream = data.get("stream")  # Stream name from combined stream wrapper
        payload = data.get(
            "data", data
        )  # Unwrap payload; fall back to raw data if not wrapped

        if stream:
            await self._registry.dispatch(
                stream, payload
            )  # Route to registered stream handler
        elif "e" in payload:
            # User data stream events have no 'stream' wrapper — dispatch by event type
            await self._registry.dispatch(
                payload["e"], payload
            )  # e.g., "executionReport", "outboundAccountPosition"

    # ── Private: Heartbeat ──────────────────

    async def _heartbeat_loop(self) -> None:
        """Log connectivity every 60 s and detect stale connections."""
        while self._state == ConnState.CONNECTED:
            try:
                await asyncio.sleep(60)
                logger.debug(
                    "Heartbeat OK – state=%s, subs=%d",
                    self._state,
                    len(self._subscriptions),
                )
            except asyncio.CancelledError:
                return

    # ── Private: User Data Stream ───────────

    async def _start_user_stream(self) -> None:
        if self._paper:
            logger.info("[PAPER] User data stream skipped (paper-trading mode).")
            return
        try:
            self._listen_key = await self._create_listen_key()
            await self.subscribe([self._listen_key])
            self._lk_task = asyncio.create_task(
                self._keepalive_listen_key_loop(), name="lk_keepalive"
            )
            logger.info(
                "User data stream started (listenKey=...%s)", self._listen_key[-8:]
            )
        except Exception as exc:
            logger.error("User data stream failed to start: %s", exc)

    async def _keepalive_listen_key_loop(self) -> None:
        while self._state == ConnState.CONNECTED:
            try:
                await asyncio.sleep(self._LISTEN_KEY_REFRESH_INTERVAL)
                await self._refresh_listen_key()
                logger.debug("ListenKey refreshed.")
            except asyncio.CancelledError:
                return
            except Exception as exc:
                logger.error("ListenKey refresh failed: %s", exc)

    # ── Private: REST helpers ───────────────

    def _sign(self, params: dict) -> str:
        query = urllib.parse.urlencode(params)  # Serialize params to URL query string
        return hmac.new(
            self._api_secret.encode(),
            query.encode(),
            hashlib.sha256,  # HMAC-SHA256 with API secret as key
        ).hexdigest()  # Return hex-encoded signature for Binance request authentication

    async def _rest_post(
        self, path: str, params: Optional[dict] = None, signed: bool = False
    ) -> dict:
        params = params or {}
        headers = {"X-MBX-APIKEY": self._api_key}  # Binance requires API key in header
        if signed:
            params["timestamp"] = int(
                time.time() * 1000
            )  # Millisecond timestamp required for signed endpoints
            params["signature"] = self._sign(
                params
            )  # HMAC-SHA256 signature appended as query param
        async with self._session.post(
            f"{self._REST_BASE}{path}", params=params, headers=headers
        ) as resp:
            body = await resp.json()
            if resp.status == 401:
                raise AuthenticationError(body)  # Invalid API key or signature
            if resp.status in (429, 418):
                raise RateLimitError(body)  # 429=rate limit, 418=IP ban (too many 429s)
            if resp.status >= 400:
                raise ConnectorError(
                    f"REST {resp.status}: {body}"
                )  # Other client/server errors
            return body

    async def _rest_put(self, path: str, params: Optional[dict] = None) -> dict:
        params = params or {}
        headers = {"X-MBX-APIKEY": self._api_key}
        async with self._session.put(
            f"{self._REST_BASE}{path}", params=params, headers=headers
        ) as resp:
            return await resp.json()

    async def _create_listen_key(self) -> str:
        data = await self._rest_post("/api/v3/userDataStream")
        return data["listenKey"]

    async def _refresh_listen_key(self) -> None:
        if self._listen_key:
            await self._rest_put(
                "/api/v3/userDataStream", {"listenKey": self._listen_key}
            )

    # ── Private: WS Request ─────────────────

    async def _send_request(self, method: str, params: List[str]) -> None:
        if not self._ws or self._ws.closed:
            raise ConnectionLostError("Cannot send – WebSocket not open.")
        self._req_id += 1  # Auto-incrementing request ID for correlating responses
        payload = {
            "method": method,
            "params": params,
            "id": self._req_id,
        }  # Binance WebSocket API format
        await self._ws.send_str(
            json.dumps(payload)
        )  # Send JSON-encoded subscription/unsubscription request

    async def _cancel_background_tasks(self) -> None:
        for task in (self._recv_task, self._hb_task, self._lk_task):
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass
