import os
import sys
import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Depends, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from contextlib import asynccontextmanager
import random

from backend.api import trading, config, analytics, models, system, login
from backend.api.login import get_current_user
from src.config import get_binance_credentials


class ConnectionManager:
    """
    Thread-safe WebSocket connection manager.

    WS-001: asyncio.Lock protects active_connections from concurrent mutation.
    WS-002: broadcast() collects dead connections and removes them after iteration.
    WS-003: heartbeat() sends server-initiated ping every 20 s to detect stale clients.
    """

    def __init__(self):
        self.active_connections: list[WebSocket] = []
        self._lock = asyncio.Lock()  # WS-001: protects list mutation

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        async with self._lock:
            self.active_connections.append(websocket)

    async def disconnect(self, websocket: WebSocket):
        async with self._lock:
            if websocket in self.active_connections:
                self.active_connections.remove(websocket)

    async def send_message(self, message: dict, websocket: WebSocket):
        await websocket.send_json(message)

    async def broadcast(self, message: dict):
        """Send message to all connections; remove dead ones (WS-002)."""
        dead: list[WebSocket] = []
        async with self._lock:
            snapshot = list(self.active_connections)
        for connection in snapshot:
            try:
                await connection.send_json(message)
            except Exception:
                dead.append(connection)
        if dead:
            async with self._lock:
                for ws in dead:
                    if ws in self.active_connections:
                        self.active_connections.remove(ws)

    async def heartbeat(self, interval: int = 20):
        """WS-003: Send server-initiated ping every `interval` seconds to detect stale clients."""
        while True:
            await asyncio.sleep(interval)
            async with self._lock:
                snapshot = list(self.active_connections)
            dead: list[WebSocket] = []
            for ws in snapshot:
                try:
                    await ws.send_json({"type": "ping"})
                except Exception:
                    dead.append(ws)
            if dead:
                async with self._lock:
                    for ws in dead:
                        if ws in self.active_connections:
                            self.active_connections.remove(ws)


manager = ConnectionManager()
binance_connector = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global binance_connector

    try:
        from src.connectors.binance_connector import BinanceConnector

        api_key, api_secret = get_binance_credentials()
        testnet = os.getenv("BINANCE_TESTNET", "true").lower() == "true"

        if api_key and api_secret:
            binance_connector = BinanceConnector(
                api_key=api_key, api_secret=api_secret, testnet=testnet
            )
            logger.info("Binance connector initialized (testnet=%s)", testnet)
        else:
            logger.warning("BINANCE_API_KEY/SECRET not set, using mock price data")
    except ImportError:
        logger.warning("Binance connector not available, using mock price data")
    except Exception as e:
        logger.error("Failed to initialize Binance: %s", e)

    asyncio.create_task(price_stream())
    asyncio.create_task(manager.heartbeat(interval=20))  # WS-003: keep-alive
    yield

    if binance_connector:
        try:
            binance_connector.close()
        except Exception:
            pass


async def price_stream():
    """Broadcasts live BTC price every second via WebSocket to all clients."""
    base_price = 43000.0

    while True:
        await asyncio.sleep(1)

        price = base_price

        if binance_connector:
            try:
                # BUG FIX: correct method name is get_current_price()
                price = binance_connector.get_current_price("BTCUSDT") or base_price
                base_price = price
            except Exception:
                base_price += random.uniform(-0.5, 0.5)
                price = base_price
        else:
            base_price += random.uniform(-0.5, 0.5)
            price = base_price

        await manager.broadcast(
            {
                "type": "price_update",
                "data": {
                    "symbol": "BTCUSDT",
                    "price": round(price, 2),
                    "timestamp": datetime.now().isoformat(),
                },
            }
        )


def _get_cors_origins() -> list:
    """Load allowed CORS origins from environment or use dev defaults."""
    env_origins = os.getenv("CORS_ORIGINS", "")
    if env_origins:
        return [o.strip() for o in env_origins.split(",") if o.strip()]
    # Dev defaults (localhost only)
    return [
        "http://localhost:5173",
        "http://localhost:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:3000",
    ]


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add standard security headers to every response."""

    async def dispatch(self, request: Request, call_next):
        response: Response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "geolocation=(), microphone=()"
        # HSTS only in production (HTTPS required)
        if os.getenv("ENVIRONMENT") == "production":
            response.headers["Strict-Transport-Security"] = (
                "max-age=31536000; includeSubDomains"
            )
        return response


app = FastAPI(
    title="BITCOIN4Traders API",
    version="1.0.0",
    lifespan=lifespan,
    swagger_ui_parameters={"persistAuthorization": True},
)

app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=_get_cors_origins(),
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
)

# ── API v1 routes ─────────────────────────────────────────────────────────────
# All routes are versioned under /api/v1/ for forward-compatibility.
# Legacy /api/<resource> paths are kept as aliases for existing clients.

# Public routes (no auth required)
app.include_router(login.router, prefix="/api/v1/auth", tags=["auth"])
app.include_router(login.router, prefix="/api/auth", tags=["auth"])   # legacy alias

# Protected routes (JWT required)
app.include_router(
    trading.router,
    prefix="/api/v1/trading",
    tags=["trading"],
    dependencies=[Depends(get_current_user)],
)
app.include_router(
    trading.router,
    prefix="/api/trading",
    tags=["trading"],
    dependencies=[Depends(get_current_user)],
)
app.include_router(
    config.router,
    prefix="/api/v1/config",
    tags=["config"],
    dependencies=[Depends(get_current_user)],
)
app.include_router(
    config.router,
    prefix="/api/config",
    tags=["config"],
    dependencies=[Depends(get_current_user)],
)
app.include_router(
    analytics.router,
    prefix="/api/v1/analytics",
    tags=["analytics"],
    dependencies=[Depends(get_current_user)],
)
app.include_router(
    analytics.router,
    prefix="/api/analytics",
    tags=["analytics"],
    dependencies=[Depends(get_current_user)],
)
app.include_router(
    models.router,
    prefix="/api/v1/models",
    tags=["models"],
    dependencies=[Depends(get_current_user)],
)
app.include_router(
    models.router,
    prefix="/api/models",
    tags=["models"],
    dependencies=[Depends(get_current_user)],
)
app.include_router(
    system.router,
    prefix="/api/v1/system",
    tags=["system"],
    dependencies=[Depends(get_current_user)],
)
app.include_router(
    system.router,
    prefix="/api/system",
    tags=["system"],
    dependencies=[Depends(get_current_user)],
)


@app.get("/", tags=["root"])
async def root():
    return {"message": "BITCOIN4Traders API", "version": "2.0.0", "docs": "/docs"}


@app.get("/api/v1/status", tags=["root"])
@app.get("/api/status", tags=["root"])
async def get_status():
    binance_status = "connected" if binance_connector else "mock"
    return {
        "status": "running",
        "timestamp": datetime.now().isoformat(),
        "version": "2.0.0",
        "binance": binance_status,
    }


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint — requires JWT token as subprotocol header or query param: /ws?token=<jwt>"""
    from backend.api.login import get_current_user, SECRET_KEY, ALGORITHM
    from jose import JWTError, jwt as jose_jwt

    # Read token from subprotocol header (safer than query param — not logged in URLs)
    token = None
    subprotocols = websocket.headers.get("sec-websocket-protocol", "")
    for part in subprotocols.split(","):
        part = part.strip()
        if part.startswith("token."):
            token = part[len("token."):]
            break
    # Fallback to query param for backwards compatibility
    if not token:
        token = websocket.query_params.get("token")

    # Validate token before accepting connection
    if not token:
        await websocket.close(code=1008, reason="Missing token")
        return
    try:
        payload = jose_jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub") or ""
        if not username:
            raise JWTError("No subject")
    except JWTError:
        await websocket.close(code=1008, reason="Invalid token")
        return

    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            if message.get("type") == "ping":
                await manager.send_message({"type": "pong"}, websocket)
            elif message.get("type") == "subscribe":
                await manager.send_message(
                    {"type": "subscribed", "symbols": message.get("symbols", [])},
                    websocket,
                )
    except WebSocketDisconnect:
        await manager.disconnect(websocket)
