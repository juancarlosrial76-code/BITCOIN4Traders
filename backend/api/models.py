"""
models.py — Echte RL-Modell-Verwaltung aus darwin_engine.

Liest Champion .pkl/.json aus data/cache/ und PPO-Modelle aus data/models/.
Training wird als Hintergrund-Task gestartet.
"""

import json
import sys
import os
import asyncio
from pathlib import Path
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

router = APIRouter()

CHAMPION_PKL = PROJECT_ROOT / "data/cache/multiverse_champion.pkl"
CHAMPION_META = PROJECT_ROOT / "data/cache/multiverse_champion_meta.json"
PPO_MODEL = PROJECT_ROOT / "data/models/ppo_best.pt"
PPO_META = PROJECT_ROOT / "data/models/champion.json"
TRAINING_LOG = PROJECT_ROOT / "data/cache/training_log.json"

# Trainings-Status (in-memory, pro Prozess)
_training_status = {"running": False, "started_at": None, "log": []}


def _file_size_mb(path: Path) -> str:
    if path.exists():
        mb = path.stat().st_size / (1024 * 1024)
        return f"{mb:.1f} MB"
    return "0 MB"


def _load_champion_entry() -> Optional[dict]:
    """Liest Champion-Metadaten aus .json."""
    if CHAMPION_META.exists():
        try:
            meta = json.loads(CHAMPION_META.read_text(encoding="utf-8"))
            return {
                "id": 1,
                "name": meta.get("name", "champion"),
                "type": meta.get("strategy_type", "BollingerScout"),
                "created": meta.get("saved_at", "unknown")[:10]
                if meta.get("saved_at")
                else "unknown",
                "size": _file_size_mb(CHAMPION_PKL),
                "status": "active",
                "sharpe": round(meta.get("sharpe", 0), 2),
                "source": "darwin_engine",
            }
        except Exception:
            pass
    return None


def _load_ppo_entry() -> Optional[dict]:
    """Liest PPO-Modell-Metadaten aus champion.json."""
    if PPO_META.exists():
        try:
            meta = json.loads(PPO_META.read_text(encoding="utf-8"))
            return {
                "id": 2,
                "name": meta.get("name", "ppo_best"),
                "type": "PPO",
                "created": meta.get("saved_at", "unknown")[:10]
                if meta.get("saved_at")
                else "unknown",
                "size": _file_size_mb(PPO_MODEL),
                "status": "trained",
                "sharpe": round(meta.get("sharpe", 0), 2),
                "source": "reinforcement_learning",
            }
        except Exception:
            pass
    return None


def _run_training_background():
    """Führt Darwin-Evolution im Hintergrund aus."""
    global _training_status
    _training_status["running"] = True
    _training_status["started_at"] = datetime.now().isoformat()
    _training_status["log"] = ["Training gestartet..."]

    try:
        from darwin_engine import (
            run_multiverse,
            generate_synthetic_btc,
            TelegramNotifier,
        )
        from dotenv import load_dotenv

        load_dotenv()

        _training_status["log"].append("Daten werden geladen...")
        try:
            from darwin_engine import load_live_data

            df = load_live_data(symbol="BTC/USDT", timeframe="1h", limit=1000)
            _training_status["log"].append(f"Echte Binance-Daten: {len(df)} Bars")
        except Exception:
            df = generate_synthetic_btc(n_bars=2000, seed=42)
            _training_status["log"].append(
                "Synthetische Daten (Binance nicht erreichbar)"
            )

        _training_status["log"].append("Evolution läuft (5 Generationen)...")
        champion = run_multiverse(
            symbol="BTC/USDT",
            timeframe="1h",
            n_bars=len(df),
            generations=5,
            pop_size=12,
            n_mc_scenarios=20,
            save_dir=str(PROJECT_ROOT / "data/cache"),
        )

        if champion:
            msg = f"Champion: {champion.name}"
            _training_status["log"].append(msg)
            # Telegram-Benachrichtigung
            try:
                notifier = TelegramNotifier.from_env()
                notifier.send(f"<b>Training abgeschlossen</b>\n{msg}")
            except Exception:
                pass

        # Log speichern
        TRAINING_LOG.parent.mkdir(parents=True, exist_ok=True)
        TRAINING_LOG.write_text(
            json.dumps(
                {
                    "completed_at": datetime.now().isoformat(),
                    "champion": champion.name if champion else None,
                    "log": _training_status["log"],
                }
            ),
            encoding="utf-8",
        )

    except Exception as exc:
        _training_status["log"].append(f"FEHLER: {exc}")
    finally:
        _training_status["running"] = False


@router.get("/")
async def get_models():
    """Gibt alle vorhandenen Modelle zurück (Champion + PPO)."""
    models = []
    champ = _load_champion_entry()
    if champ:
        models.append(champ)
    ppo = _load_ppo_entry()
    if ppo:
        models.append(ppo)

    if not models:
        # Zeige Info-Eintrag wenn noch kein Training durchgeführt
        models.append(
            {
                "id": 0,
                "name": "Kein Modell vorhanden",
                "type": "-",
                "created": "-",
                "size": "0 MB",
                "status": "not_trained",
                "sharpe": 0,
                "source": "none",
            }
        )
    return models


@router.post("/train")
async def train_model(background_tasks: BackgroundTasks):
    """Startet Darwin-Evolution als Hintergrund-Task."""
    if _training_status["running"]:
        return {
            "status": "already_running",
            "started_at": _training_status["started_at"],
            "message": "Training läuft bereits",
        }
    background_tasks.add_task(_run_training_background)
    return {
        "status": "started",
        "message": "Darwin-Evolution gestartet (5 Generationen)",
        "timestamp": datetime.now().isoformat(),
    }


@router.get("/training/status")
async def get_training_status():
    """Gibt aktuellen Trainings-Status zurück."""
    if TRAINING_LOG.exists():
        try:
            last = json.loads(TRAINING_LOG.read_text(encoding="utf-8"))
        except Exception:
            last = {}
    else:
        last = {}

    return {
        "is_running": _training_status["running"],
        "started_at": _training_status["started_at"],
        "log": _training_status["log"][-20:],  # letzte 20 Zeilen
        "last_run": last,
    }


@router.get("/training/history")
async def get_training_history():
    """Gibt Trainings-Historie aus Log-Datei zurück."""
    history = []

    if TRAINING_LOG.exists():
        try:
            entry = json.loads(TRAINING_LOG.read_text(encoding="utf-8"))
            history.append(
                {
                    "model": entry.get("champion", "unknown"),
                    "start": entry.get("completed_at", "")[:16].replace("T", " "),
                    "end": entry.get("completed_at", "")[:16].replace("T", " "),
                    "status": "completed",
                    "result": f"Champion: {entry.get('champion', 'n/a')}",
                }
            )
        except Exception:
            pass

    if CHAMPION_META.exists():
        try:
            meta = json.loads(CHAMPION_META.read_text(encoding="utf-8"))
            saved = meta.get("saved_at", "")[:16].replace("T", " ")
            history.insert(
                0,
                {
                    "model": meta.get("name", "champion"),
                    "start": saved,
                    "end": saved,
                    "status": "active",
                    "result": f"Sharpe: {meta.get('sharpe', 0):.2f}",
                },
            )
        except Exception:
            pass

    return (
        history
        if history
        else [
            {
                "model": "Kein Training",
                "start": "-",
                "end": "-",
                "status": "none",
                "result": "-",
            }
        ]
    )


# WICHTIG: /{model_id} NACH allen statischen Routen
@router.get("/{model_id}")
async def get_model(model_id: int):
    models = await get_models()
    for m in models:
        if m["id"] == model_id:
            return m
    raise HTTPException(status_code=404, detail=f"Modell {model_id} nicht gefunden")


@router.delete("/{model_id}")
async def delete_model(model_id: int):
    if model_id == 1 and CHAMPION_PKL.exists():
        CHAMPION_PKL.unlink(missing_ok=True)
        CHAMPION_META.unlink(missing_ok=True)
        return {"status": "deleted", "model_id": model_id}
    if model_id == 2 and PPO_MODEL.exists():
        PPO_MODEL.unlink(missing_ok=True)
        PPO_META.unlink(missing_ok=True)
        return {"status": "deleted", "model_id": model_id}
    raise HTTPException(status_code=404, detail=f"Modell {model_id} nicht gefunden")
