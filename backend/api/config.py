"""
config.py — Bot/Risk/Data Konfiguration mit JSON-Persistenz.

Einstellungen werden in data/cache/bot_config.json gespeichert
und überleben Server-Neustarts.
"""

import json
import sys
from pathlib import Path

from fastapi import APIRouter
from pydantic import BaseModel

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

router = APIRouter()

CONFIG_FILE = PROJECT_ROOT / "data/cache/bot_config.json"

# Default-Konfiguration
_DEFAULTS = {
    "bot": {
        "symbol": "BTCUSDT",
        "timeframe": "1h",
        "maxPositions": 3,
        "agentType": "DarwinBot",
        "modelPath": "data/cache/multiverse_champion.pkl",
    },
    "risk": {
        "maxDrawdown": 0.2,
        "stopLoss": 0.02,
        "takeProfit": 0.05,
        "positionSizePercent": 0.1,
    },
    "data": {
        "dataSource": "binance",
        "startDate": "2023-01-01",
        "endDate": "2026-01-01",
        "trainTestSplit": 0.8,
    },
}


class BotConfig(BaseModel):
    symbol: str = "BTCUSDT"
    timeframe: str = "1h"
    maxPositions: int = 3
    agentType: str = "DarwinBot"
    modelPath: str = "data/cache/multiverse_champion.pkl"


class RiskConfig(BaseModel):
    maxDrawdown: float = 0.2
    stopLoss: float = 0.02
    takeProfit: float = 0.05
    positionSizePercent: float = 0.1


class DataConfig(BaseModel):
    dataSource: str = "binance"
    startDate: str = "2023-01-01"
    endDate: str = "2026-01-01"
    trainTestSplit: float = 0.8


def _load_config() -> dict:
    """Lädt Konfiguration aus Datei, Defaults als Fallback."""
    if CONFIG_FILE.exists():
        try:
            saved = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
            # Deep merge: saved überschreibt defaults
            result = {}
            for section in ("bot", "risk", "data"):
                result[section] = {**_DEFAULTS[section], **saved.get(section, {})}
            return result
        except Exception:
            pass
    return {k: dict(v) for k, v in _DEFAULTS.items()}


def _save_config(cfg: dict) -> None:
    """Speichert Konfiguration als JSON."""
    CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_FILE.write_text(json.dumps(cfg, indent=2), encoding="utf-8")


@router.get("/")
async def get_config():
    return _load_config()


@router.get("/bot")
async def get_bot_config():
    return _load_config()["bot"]


@router.put("/bot")
async def update_bot_config(config: BotConfig):
    cfg = _load_config()
    cfg["bot"] = config.model_dump()
    _save_config(cfg)
    return {"status": "updated", "config": cfg["bot"]}


@router.get("/risk")
async def get_risk_config():
    return _load_config()["risk"]


@router.put("/risk")
async def update_risk_config(config: RiskConfig):
    cfg = _load_config()
    cfg["risk"] = config.model_dump()
    _save_config(cfg)
    return {"status": "updated", "config": cfg["risk"]}


@router.get("/data")
async def get_data_config():
    return _load_config()["data"]


@router.put("/data")
async def update_data_config(config: DataConfig):
    cfg = _load_config()
    cfg["data"] = config.model_dump()
    _save_config(cfg)
    return {"status": "updated", "config": cfg["data"]}
