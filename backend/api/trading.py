"""
trading.py — Live-Trading mit Darwin-Engine Champion für Signale.

- Echte Binance-Orders wenn API-Keys gesetzt
- Darwin-Champion liefert BUY/SELL/HOLD Signale
- Paper-Trading Modus wenn kein API-Key
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

router = APIRouter()

CHAMPION_PKL = PROJECT_ROOT / "data/cache/multiverse_champion.pkl"
CHAMPION_META = PROJECT_ROOT / "data/cache/multiverse_champion_meta.json"
CONFIG_FILE = PROJECT_ROOT / "data/cache/bot_config.json"

binance_connector = None
_champion_cache = None  # gecachter Champion-Bot


def get_binance_connector():
    global binance_connector
    if binance_connector is None:
        try:
            from src.connectors.binance_connector import BinanceConnector

            api_key = os.getenv("BINANCE_API_KEY")
            api_secret = os.getenv("BINANCE_API_SECRET")
            testnet = os.getenv("BINANCE_TESTNET", "true").lower() == "true"
            if api_key and api_secret:
                binance_connector = BinanceConnector(
                    api_key=api_key, api_secret=api_secret, testnet=testnet
                )
        except Exception as e:
            print(f"Binance-Connector Fehler: {e}")
    return binance_connector


def _get_champion():
    """Lädt Darwin-Champion (mit In-Memory Cache)."""
    global _champion_cache
    if _champion_cache is not None:
        return _champion_cache
    if CHAMPION_PKL.exists():
        try:
            from darwin_engine import ChampionPersistence

            _champion_cache = ChampionPersistence.load(
                str(CHAMPION_PKL),
                str(CHAMPION_META) if CHAMPION_META.exists() else None,
            )
        except Exception as e:
            print(f"Champion laden fehlgeschlagen: {e}")
    return _champion_cache


def _get_champion_signal() -> dict:
    """
    Fragt den Champion nach dem aktuellen Signal.
    Gibt {'signal': 1|0|-1, 'label': 'LONG'|'FLAT'|'SHORT', 'champion': name} zurück.
    """
    champion = _get_champion()
    if champion is None:
        return {
            "signal": 0,
            "label": "FLAT",
            "champion": None,
            "reason": "kein Champion",
        }

    try:
        from darwin_engine import generate_synthetic_btc

        try:
            from darwin_engine import load_live_data

            df = load_live_data(symbol="BTC/USDT", timeframe="1h", limit=200)
        except Exception:
            df = generate_synthetic_btc(
                n_bars=200, seed=int(datetime.now().timestamp()) % 1000
            )

        signals = champion.compute_signals(df["close"].values)
        last_signal = int(signals[-1])
        label = {1: "LONG", -1: "SHORT", 0: "FLAT"}.get(last_signal, "FLAT")
        return {
            "signal": last_signal,
            "label": label,
            "champion": champion.name,
            "reason": "darwin_engine",
        }
    except Exception as exc:
        return {
            "signal": 0,
            "label": "FLAT",
            "champion": getattr(champion, "name", None),
            "reason": str(exc),
        }


class OrderRequest(BaseModel):
    symbol: str
    side: str
    order_type: str
    quantity: float
    price: float | None = None


class TradingConfig(BaseModel):
    maxPositionSize: float = 0.1
    stopLoss: float = 0.02
    takeProfit: float = 0.05
    riskPerTrade: float = 0.02
    leverage: int = 1


trading_state = {
    "is_running": False,
    "current_position": 0.0,
    "unrealized_pnl": 0.0,
    "config": TradingConfig(),
}


@router.get("/status")
async def get_status():
    """Gibt Trading-Status + aktuelles Champion-Signal zurück."""
    connector = get_binance_connector()
    position_value = trading_state["current_position"]
    unrealized = trading_state["unrealized_pnl"]

    if connector:
        try:
            position = connector.get_position("BTCUSDT")
            if position:
                position_value = float(position.get("positionAmt", 0))
                unrealized = float(position.get("unrealizedProfit", 0))
        except Exception:
            pass

    signal_info = _get_champion_signal()

    return {
        "is_running": trading_state["is_running"],
        "current_position": position_value,
        "unrealized_pnl": unrealized,
        "timestamp": datetime.now().isoformat(),
        "champion_signal": signal_info["label"],
        "champion_name": signal_info.get("champion"),
        "mode": "live" if connector else "paper",
    }


@router.post("/start")
async def start_trading():
    trading_state["is_running"] = True
    signal = _get_champion_signal()
    return {
        "status": "started",
        "timestamp": datetime.now().isoformat(),
        "champion": signal.get("champion"),
        "initial_signal": signal["label"],
        "mode": "live" if get_binance_connector() else "paper",
    }


@router.post("/stop")
async def stop_trading():
    trading_state["is_running"] = False
    return {"status": "stopped", "timestamp": datetime.now().isoformat()}


@router.get("/signal")
async def get_signal():
    """Gibt das aktuelle Champion-Signal zurück."""
    return _get_champion_signal()


@router.post("/order")
async def place_order(order: OrderRequest):
    connector = get_binance_connector()

    if connector and order.order_type.lower() == "market":
        try:
            result = connector.place_market_order(
                symbol=order.symbol,
                side=order.side.upper(),
                quantity=order.quantity,
            )
            return {
                "order_id": result.get("orderId", f"ORD-{datetime.now().timestamp()}"),
                "symbol": order.symbol,
                "side": order.side,
                "status": result.get("status", "FILLED"),
                "quantity": order.quantity,
                "price": float(result.get("price", 0)),
                "timestamp": datetime.now().isoformat(),
                "mode": "live",
            }
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

    # Paper-Trading Fallback
    return {
        "order_id": f"PAPER-{int(datetime.now().timestamp())}",
        "symbol": order.symbol,
        "side": order.side,
        "status": "paper_filled",
        "quantity": order.quantity,
        "price": order.price or 43000.0,
        "timestamp": datetime.now().isoformat(),
        "mode": "paper",
    }


@router.get("/orders")
async def get_orders():
    connector = get_binance_connector()
    if connector:
        try:
            orders = connector.get_open_orders("BTCUSDT")
            return [
                {
                    "id": str(o.get("orderId", "")),
                    "symbol": o.get("symbol", "BTCUSDT"),
                    "side": o.get("side", ""),
                    "status": o.get("status", ""),
                    "quantity": float(o.get("origQty", 0)),
                    "price": float(o.get("price", 0)),
                    "timestamp": o.get("time", ""),
                }
                for o in orders[:10]
            ]
        except Exception:
            pass
    return []


@router.get("/balance")
async def get_balance():
    connector = get_binance_connector()
    if connector:
        try:
            return {"balance": connector.get_account_balance(), "mode": "live"}
        except Exception as e:
            return {"balance": {}, "mode": "live", "error": str(e)}
    return {"balance": {"USDT": 10000.0, "BTC": 0.0}, "mode": "paper"}


@router.get("/config")
async def get_config():
    return trading_state["config"]


@router.put("/config")
async def update_config(config: TradingConfig):
    trading_state["config"] = config
    return {"status": "updated", "config": config}
