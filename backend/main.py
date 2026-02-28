import os
import sys
import asyncio
import json
from datetime import datetime
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Depends
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import random

from backend.api import trading, config, analytics, models, system, login
from backend.api.login import get_current_user


class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def send_message(self, message: dict, websocket: WebSocket):
        await websocket.send_json(message)

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception:
                pass


manager = ConnectionManager()
binance_connector = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global binance_connector

    try:
        from src.connectors.binance_connector import BinanceConnector

        api_key = os.getenv("BINANCE_API_KEY")
        api_secret = os.getenv("BINANCE_API_SECRET")
        testnet = os.getenv("BINANCE_TESTNET", "true").lower() == "true"

        if api_key and api_secret:
            binance_connector = BinanceConnector(
                api_key=api_key, api_secret=api_secret, testnet=testnet
            )
            print(f"Binance connector initialized (testnet={testnet})")
        else:
            print("BINANCE_API_KEY/SECRET not set, using mock price data")
    except ImportError:
        print("Binance connector not available, using mock price data")
    except Exception as e:
        print(f"Failed to initialize Binance: {e}")

    asyncio.create_task(price_stream())
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


app = FastAPI(
    title="BITCOIN4Traders API",
    version="1.0.0",
    lifespan=lifespan,
    # Swagger UI requires auth header
    swagger_ui_parameters={"persistAuthorization": True},
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Public routes (no auth required)
app.include_router(login.router, prefix="/api/auth", tags=["auth"])

# Protected routes (JWT required)
app.include_router(
    trading.router,
    prefix="/api/trading",
    tags=["trading"],
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
    prefix="/api/analytics",
    tags=["analytics"],
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
    prefix="/api/system",
    tags=["system"],
    dependencies=[Depends(get_current_user)],
)


@app.get("/", tags=["root"])
async def root():
    return {"message": "BITCOIN4Traders API", "version": "1.0.0", "docs": "/docs"}


@app.get("/api/status", tags=["root"])
async def get_status():
    binance_status = "connected" if binance_connector else "mock"
    return {
        "status": "running",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",
        "binance": binance_status,
    }


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
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
        manager.disconnect(websocket)
