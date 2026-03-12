"""
Module B — Colab RL-Inferenz-Engine mit Ably-Bridge
====================================================
Läuft in Google Colab (oder vast.ai / RunPod).

Aufgaben:
  1. Marktdaten von Lokal abonnieren (bt4t/market/BTCUSDT)
  2. Observation-Buffer aufbauen (rolling window)
  3. RL-Modell laden (Google Drive oder lokal)
  4. Inferenz: Observation → BUY/SELL/HOLD + Confidence
  5. Signal → Ably → Lokal publishen (bt4t/signals)
  6. Heartbeat publishen (bt4t/health) alle 10s
  7. Befehle vom Lokal empfangen und ausführen

Setup in Colab (erste Zelle):
    !pip install ably ccxt loguru
    !pip install torch  # oder stable-baselines3

Verwendung (Colab-Zelle):
    import sys
    sys.path.insert(0, '/content/drive/MyDrive/BITCOIN4Traders')

    from colab_bridge.module_b_colab import ModuleB
    import asyncio

    engine = ModuleB(
        ably_key="your_ably_key",
        model_path="/content/drive/MyDrive/BITCOIN4Traders/data/models/curriculum_test/best_model.pth",
    )
    await engine.run()  # In Colab: asyncio.get_event_loop().run_until_complete(engine.run())

Umgebungsvariablen:
    ABLY_API_KEY=your_ably_root_key
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional

import numpy as np

# ── Ably (Pflicht) ────────────────────────────────────────────────────────────
try:
    from ably import AblyRealtime

    _ABLY_OK = True
except ImportError:
    _ABLY_OK = False
    print("FEHLER: pip install ably")

# ── Logging (loguru oder print-Fallback) ──────────────────────────────────────
try:
    from loguru import logger

    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | <cyan>{message}</cyan>",
        level="INFO",
        colorize=True,
    )
    try:
        logger.add(
            "logs/module_b.log", rotation="20 MB", retention="7 days", level="DEBUG"
        )
    except Exception:
        pass  # In Colab kein logs/-Verzeichnis
except ImportError:
    import logging

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
    )
    logger = logging.getLogger("module_b")

# ── Kanal-Namen (identisch zu Module A) ──────────────────────────────────────
CH_MARKET = "bt4t:market:{symbol}"
CH_SIGNALS = "bt4t:signals"
CH_PORTFOLIO = "bt4t:portfolio:state"
CH_HEALTH = "bt4t:health"
CH_CONTROL = "bt4t:control:cmd"
CH_ACK = "bt4t:control:ack"

# ── Konfiguration ─────────────────────────────────────────────────────────────
OBS_WINDOW = 60  # Anzahl historischer Bars für RL-Observation
MIN_CONFIDENCE = 0.55  # Unter dieser Grenze → HOLD (kein Signal)
HEARTBEAT_INTERVAL_S = 10.0  # Heartbeat-Intervall in Sekunden
CONTROL_POLL_S = 5.0  # Wie oft auf Befehle prüfen


# ── Modell-Adapter ────────────────────────────────────────────────────────────


class ModelAdapter:
    """
    Abstraktionsschicht für verschiedene Modell-Typen.

    Unterstützt:
      - Darwin-Champion (darwin_engine.py)
      - PyTorch PPO/DRL Model (.pth Checkpoint)
      - Stable-Baselines3 Model
      - Fallback: RSI-basiertes Signal (kein Modell nötig)
    """

    def __init__(self, model_path: Optional[str] = None, model_type: str = "auto"):
        self.model = None
        self.model_type = model_type
        self.model_path = model_path
        self.model_version = "unknown"
        self._load(model_path, model_type)

    def _load(self, path: Optional[str], model_type: str):
        """Lädt Modell aus Pfad. Fallback auf RSI wenn kein Modell."""

        # ── 1. Darwin-Champion (.pkl) ─────────────────────────────────────────
        if path and path.endswith(".pkl"):
            try:
                import pickle

                with open(path, "rb") as f:
                    self.model = pickle.load(f)
                self.model_type = "darwin"
                self.model_version = getattr(self.model, "name", "darwin_champion")
                logger.success(f"Darwin-Champion geladen: {self.model_version}")
                return
            except Exception as e:
                logger.warning(f"Darwin-Champion laden fehlgeschlagen: {e}")

        # ── 2. PyTorch Checkpoint (.pth) ──────────────────────────────────────
        if path and path.endswith(".pth"):
            try:
                import torch

                checkpoint = torch.load(path, map_location="cpu")
                self.model = checkpoint
                self.model_type = "pytorch"
                self.model_version = f"pth:{Path(path).stem}"
                logger.success(f"PyTorch Checkpoint geladen: {path}")
                return
            except Exception as e:
                logger.warning(f"PyTorch Checkpoint laden fehlgeschlagen: {e}")

        # ── 3. Stable-Baselines3 ──────────────────────────────────────────────
        if path and (path.endswith(".zip") or "sb3" in (path or "").lower()):
            try:
                from stable_baselines3 import PPO

                self.model = PPO.load(path)
                self.model_type = "sb3"
                self.model_version = f"sb3:{Path(path).stem}"
                logger.success(f"SB3-Modell geladen: {path}")
                return
            except Exception as e:
                logger.warning(f"SB3-Modell laden fehlgeschlagen: {e}")

        # ── 4. Fallback: RSI-Signal ───────────────────────────────────────────
        logger.warning("Kein Modell geladen — verwende RSI-Fallback-Signal")
        self.model_type = "rsi_fallback"
        self.model_version = "rsi_fallback_v1"

    def predict(self, close_array: np.ndarray, features: dict) -> tuple[str, float]:
        """
        Gibt (action, confidence) zurück.
        action: 'BUY' | 'SELL' | 'HOLD'
        confidence: 0.0–1.0
        """
        try:
            if self.model_type == "darwin":
                return self._predict_darwin(close_array)
            elif self.model_type == "sb3":
                return self._predict_sb3(close_array, features)
            elif self.model_type == "pytorch":
                return self._predict_pytorch(close_array, features)
        except Exception as e:
            logger.warning(f"Inferenz-Fehler ({self.model_type}): {e}")

        # Immer Fallback
        return self._predict_rsi(close_array)

    def _predict_darwin(self, close: np.ndarray) -> tuple[str, float]:
        """Darwin-Champion Inferenz."""
        sigs = self.model.compute_signals(close)
        raw = int(np.sign(sigs[-1])) if len(sigs) else 0
        # Konfidenz: wie stark das Signal relativ zu den letzten 10 Bars ist
        recent = np.abs(sigs[-10:]).mean() if len(sigs) >= 10 else 0.5
        confidence = float(np.clip(recent + 0.5, 0.0, 1.0))
        action = {1: "BUY", -1: "SELL", 0: "HOLD"}.get(raw, "HOLD")
        return action, confidence

    def _predict_rsi(self, close: np.ndarray) -> tuple[str, float]:
        """RSI-basiertes Fallback-Signal."""
        if len(close) < 14:
            return "HOLD", 0.5
        delta = np.diff(close)
        gain = np.where(delta > 0, delta, 0.0)
        loss = np.where(delta < 0, -delta, 0.0)
        avg_gain = np.convolve(gain, np.ones(14) / 14, mode="valid")[-1]
        avg_loss = np.convolve(loss, np.ones(14) / 14, mode="valid")[-1]
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        if rsi < 30:
            return "BUY", float(0.5 + (30 - rsi) / 60)  # Stärker unter 30
        elif rsi > 70:
            return "SELL", float(0.5 + (rsi - 70) / 60)  # Stärker über 70
        return "HOLD", float(0.5 - abs(rsi - 50) / 100)

    def _predict_sb3(self, close: np.ndarray, features: dict) -> tuple[str, float]:
        """Stable-Baselines3 Inferenz."""
        # Observation: normalisierte Returns (letzten 60 Bars)
        obs = self._build_obs(close, features)
        action_idx, _states = self.model.predict(obs, deterministic=True)
        # Diskrete Action-Space: 0=SELL, 1=HOLD, 2=BUY (typisch für PPO)
        action_map = {0: "SELL", 1: "HOLD", 2: "BUY"}
        action = action_map.get(int(action_idx), "HOLD")
        # SB3 hat keine direkte Konfidenz → Wert aus Policy-Logits ableiten
        try:
            obs_tensor = self.model.policy.obs_to_tensor(obs)[0]
            with __import__("torch").no_grad():
                dist = self.model.policy.get_distribution(obs_tensor)
                probs = dist.distribution.probs[0].cpu().numpy()
            confidence = float(probs.max())
        except Exception:
            confidence = 0.6
        return action, confidence

    def _predict_pytorch(self, close: np.ndarray, features: dict) -> tuple[str, float]:
        """
        Generischer PyTorch-Checkpoint.
        Erwartet checkpoint-Dict mit 'actor_state_dict' oder 'model_state_dict'.
        Fällt auf RSI-Fallback zurück wenn Architektur unbekannt.
        """
        logger.debug(
            "PyTorch-Checkpoint: direkter Aufruf nicht implementiert → RSI-Fallback"
        )
        return self._predict_rsi(close)

    def _build_obs(self, close: np.ndarray, features: dict) -> np.ndarray:
        """Baut Observation-Vektor für RL-Modell."""
        n = min(len(close), OBS_WINDOW)
        close_w = close[-n:]
        # Normalisierte Log-Returns
        if len(close_w) >= 2:
            log_ret = np.diff(np.log(close_w + 1e-8))
        else:
            log_ret = np.zeros(OBS_WINDOW - 1)
        # Padding
        if len(log_ret) < OBS_WINDOW - 1:
            log_ret = np.pad(log_ret, (OBS_WINDOW - 1 - len(log_ret), 0))
        # Zusätzliche skalare Features
        scalars = np.array(
            [
                features.get("rsi14", 50.0) / 100.0 - 0.5,
                features.get("bb_pct", 0.5) - 0.5,
                features.get("macd", 0.0) / (close[-1] + 1e-8) * 1000,
                features.get("vol_ratio", 1.0) - 1.0,
                features.get("return_1h", 0.0) * 100,
                features.get("return_4h", 0.0) * 100,
            ],
            dtype=np.float32,
        )
        obs = np.concatenate([log_ret.astype(np.float32), scalars])
        return obs

    def reload(self, new_path: Optional[str] = None):
        """Lädt Modell neu (nach RELOAD_MODEL-Befehl)."""
        path = new_path or self.model_path
        logger.info(f"Modell wird neu geladen: {path}")
        self._load(path, "auto")


# ── Haupt-Engine ──────────────────────────────────────────────────────────────


class ModuleB:
    """
    Colab RL-Inferenz-Engine.

    Empfängt Marktdaten, führt RL-Inferenz durch,
    publisht Signale zurück an den lokalen Rechner.
    """

    def __init__(
        self,
        ably_key: str,
        model_path: Optional[str] = None,
        symbol: str = "BTCUSDT",
        min_confidence: float = MIN_CONFIDENCE,
    ):
        if not _ABLY_OK:
            raise ImportError("pip install ably")

        self.ably_key = ably_key
        self.symbol = symbol
        self.min_confidence = min_confidence

        # Observation Buffer (rolling window)
        self._obs_buffer: Deque[dict] = deque(maxlen=OBS_WINDOW + 10)

        # Modell
        self.model_adapter = ModelAdapter(model_path)

        # State
        self._running = False
        self._paused = False
        self._ably: Optional[AblyRealtime] = None
        self._last_market_ts: float = 0.0
        self._inference_count = 0
        self._signal_count = 0
        self._portfolio_state: dict = {}

        logger.success(
            f"Module B bereit | Symbol: {symbol} | Modell: {self.model_adapter.model_version}"
        )

    # ── Ably-Verbindung ───────────────────────────────────────────────────────

    async def _connect_ably(self):
        self._ably = AblyRealtime(self.ably_key)
        await self._ably.connection.once_async("connected")
        logger.success("Ably verbunden (Colab)")

        # Marktdaten abonnieren
        ch_name = CH_MARKET.format(symbol=self.symbol)
        ch_market = self._ably.channels.get(ch_name)
        await ch_market.subscribe(self._on_market_data)
        logger.info(f"Abonniert: {ch_name}")

        # Portfolio-State abonnieren (vom Lokal)
        ch_portfolio = self._ably.channels.get(CH_PORTFOLIO)
        await ch_portfolio.subscribe(self._on_portfolio_state)

        # Befehle abonnieren
        ch_ctrl = self._ably.channels.get(CH_CONTROL)
        await ch_ctrl.subscribe(self._on_command)
        logger.info(f"Abonniert: {CH_CONTROL}")

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def _on_market_data(self, message):
        """Empfängt Marktdaten und triggert Inferenz."""
        try:
            data = (
                json.loads(message.data)
                if isinstance(message.data, str)
                else message.data
            )
            self._obs_buffer.append(data)
            self._last_market_ts = time.time()

            # Inferenz nur wenn genug Daten
            if len(self._obs_buffer) >= min(20, OBS_WINDOW // 3):
                asyncio.create_task(self._run_inference(data))

        except Exception as e:
            logger.warning(f"Marktdaten-Parsing Fehler: {e}")

    def _on_portfolio_state(self, message):
        """Empfängt Portfolio-State vom lokalen Rechner."""
        try:
            data = (
                json.loads(message.data)
                if isinstance(message.data, str)
                else message.data
            )
            self._portfolio_state = data
        except Exception:
            pass

    def _on_command(self, message):
        """Empfängt Steuerbefehle vom lokalen Rechner."""
        try:
            data = (
                json.loads(message.data)
                if isinstance(message.data, str)
                else message.data
            )
            cmd = data.get("cmd", "")
            params = data.get("params", {})
            logger.info(f"Befehl empfangen: {cmd}")
            asyncio.create_task(self._execute_command(cmd, params))
        except Exception as e:
            logger.warning(f"Command-Parsing Fehler: {e}")

    # ── Inferenz ──────────────────────────────────────────────────────────────

    async def _run_inference(self, latest_features: dict):
        """Führt RL-Inferenz durch und publisht Signal."""
        if self._paused:
            return

        t0 = time.time()
        try:
            # Close-Array aus Buffer zusammenbauen
            close_60 = latest_features.get("close_60", [])
            if close_60:
                close_arr = np.array(close_60, dtype=np.float64)
            else:
                # Aus Buffer rekonstruieren
                close_arr = np.array(
                    [d.get("close", 0.0) for d in self._obs_buffer if "close" in d],
                    dtype=np.float64,
                )

            if len(close_arr) < 10:
                return

            # Inferenz
            action, confidence = self.model_adapter.predict(close_arr, latest_features)
            self._inference_count += 1
            latency_ms = (time.time() - t0) * 1000

            # Logging
            logger.debug(
                f"Inferenz #{self._inference_count}: {action} "
                f"conf={confidence:.3f} latency={latency_ms:.1f}ms"
            )

            # Signal publishen wenn Konfidenz ausreichend
            if action != "HOLD" and confidence >= self.min_confidence:
                await self._publish_signal(action, confidence, latency_ms)
            else:
                logger.debug(
                    f"Signal unterdrückt: {action} conf={confidence:.3f} "
                    f"(min={self.min_confidence})"
                )

        except Exception as e:
            logger.error(f"Inferenz-Fehler: {e}", exc_info=True)

    async def _publish_signal(self, action: str, confidence: float, latency_ms: float):
        """Publisht Handelssignal auf Ably → Lokal."""
        self._signal_count += 1
        payload = {
            "action": action,  # BUY | SELL | HOLD
            "symbol": self.symbol,
            "confidence": round(confidence, 4),
            "model_version": self.model_adapter.model_version,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "inference_latency_ms": round(latency_ms, 1),
            "signal_n": self._signal_count,
            # Portfolio-Kontext (vom Lokal empfangen)
            "portfolio_equity": self._portfolio_state.get("equity", 0.0),
            "portfolio_return_pct": self._portfolio_state.get("return_pct", 0.0),
        }
        channel = self._ably.channels.get(CH_SIGNALS)
        await channel.publish("signal", json.dumps(payload))
        logger.success(
            f"SIGNAL GESENDET: {action} | conf={confidence:.3f} | "
            f"model={self.model_adapter.model_version}"
        )

    # ── Heartbeat ─────────────────────────────────────────────────────────────

    async def _heartbeat_loop(self):
        """Publisht alle N Sekunden einen Heartbeat."""
        while self._running:
            try:
                payload = {
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "model_loaded": self.model_adapter.model is not None,
                    "model_version": self.model_adapter.model_version,
                    "inference_count": self._inference_count,
                    "signal_count": self._signal_count,
                    "obs_buffer_size": len(self._obs_buffer),
                    "paused": self._paused,
                    "last_market_data_age_s": round(
                        time.time() - self._last_market_ts, 1
                    ),
                    "status": "COLAB_READY" if not self._paused else "COLAB_PAUSED",
                }
                channel = self._ably.channels.get(CH_HEALTH)
                await channel.publish("heartbeat", json.dumps(payload))
                logger.debug(f"Heartbeat gesendet | inferences={self._inference_count}")
            except Exception as e:
                logger.warning(f"Heartbeat Fehler: {e}")

            await asyncio.sleep(HEARTBEAT_INTERVAL_S)

    # ── Befehls-Handler ───────────────────────────────────────────────────────

    async def _execute_command(self, cmd: str, params: dict):
        """Führt Steuerbefehl aus und sendet ACK zurück."""
        status = "OK"
        msg = ""

        try:
            if cmd == "PAUSE_INFERENCE":
                self._paused = True
                logger.warning("Inferenz PAUSIERT (Remote-Befehl)")
                msg = "Inference paused"

            elif cmd == "RESUME":
                self._paused = False
                logger.success("Inferenz FORTGESETZT (Remote-Befehl)")
                msg = "Inference resumed"

            elif cmd == "RELOAD_MODEL":
                new_path = params.get("model_path")
                self.model_adapter.reload(new_path)
                msg = f"Model reloaded: {self.model_adapter.model_version}"
                logger.success(msg)

            elif cmd == "SHUTDOWN":
                logger.warning("SHUTDOWN-Befehl empfangen")
                msg = "Shutting down"
                self._running = False

            elif cmd == "STATUS":
                msg = (
                    f"model={self.model_adapter.model_version}, "
                    f"inferences={self._inference_count}, "
                    f"signals={self._signal_count}, "
                    f"paused={self._paused}"
                )

            else:
                status = "UNKNOWN_CMD"
                msg = f"Unbekannter Befehl: {cmd}"
                logger.warning(msg)

        except Exception as e:
            status = "ERROR"
            msg = str(e)
            logger.error(f"Befehl {cmd} fehlgeschlagen: {e}")

        # ACK zurücksenden
        ack_payload = {
            "cmd": cmd,
            "status": status,
            "msg": msg,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        }
        channel = self._ably.channels.get(CH_ACK)
        await channel.publish("ack", json.dumps(ack_payload))

    # ── Status ────────────────────────────────────────────────────────────────

    def _print_status(self):
        age = time.time() - self._last_market_ts
        logger.info(
            f"Status | inferences={self._inference_count} | "
            f"signals={self._signal_count} | "
            f"buffer={len(self._obs_buffer)}/{OBS_WINDOW} | "
            f"data_age={age:.0f}s | paused={self._paused}"
        )

    # ── Haupt-Loop ────────────────────────────────────────────────────────────

    async def run(self):
        """Startet Module B. Blockiert bis SHUTDOWN oder Ctrl+C."""
        await self._connect_ably()
        self._running = True

        logger.success("=" * 60)
        logger.success("  Module B gestartet — Colab RL-Inferenz-Engine")
        logger.success(f"  Modell    : {self.model_adapter.model_version}")
        logger.success(f"  Symbol    : {self.symbol}")
        logger.success(f"  Abonniert : {CH_MARKET.format(symbol=self.symbol)}")
        logger.success(f"  Publisht  : {CH_SIGNALS}")
        logger.success("  Warte auf Marktdaten von Lokal...")
        logger.success("=" * 60)

        # Heartbeat-Task starten
        heartbeat_task = asyncio.create_task(self._heartbeat_loop())

        try:
            tick = 0
            while self._running:
                await asyncio.sleep(30.0)
                if tick % 4 == 0:  # alle 2 Minuten
                    self._print_status()
                tick += 1

        except asyncio.CancelledError:
            pass
        except KeyboardInterrupt:
            logger.info("Strg+C — beende Module B")
        finally:
            heartbeat_task.cancel()
            try:
                await heartbeat_task
            except asyncio.CancelledError:
                pass
            await self._shutdown()

    async def _shutdown(self):
        self._running = False
        logger.info("Module B: Beende Ably-Verbindung...")
        if self._ably:
            await self._ably.close()
        logger.success(
            f"Module B beendet | "
            f"Inferenzen={self._inference_count} | "
            f"Signale={self._signal_count}"
        )


# ── Colab-Hilfsfunktionen ─────────────────────────────────────────────────────


def colab_setup(drive_mount: bool = True) -> str:
    """
    Richtet Colab-Umgebung ein.

    Mountet Google Drive und gibt Projekt-Pfad zurück.
    Aufruf: project_path = colab_setup()

    Returns:
        str: Pfad zum Projektverzeichnis auf Drive
    """
    if drive_mount:
        try:
            from google.colab import drive

            drive.mount("/content/drive")
            logger.success("Google Drive gemountet: /content/drive")
        except ImportError:
            logger.warning("Nicht in Colab — Drive-Mount übersprungen")

    # Typischer Pfad
    project_path = "/content/drive/MyDrive/BITCOIN4Traders"
    if not Path(project_path).exists():
        project_path = "/content/BITCOIN4Traders"
        logger.warning(f"Drive-Pfad nicht gefunden — verwende: {project_path}")

    sys.path.insert(0, project_path)
    logger.success(f"Projekt-Pfad: {project_path}")
    return project_path


def find_champion_on_drive(project_path: str) -> Optional[str]:
    """Sucht den Darwin-Champion im Drive-Verzeichnis."""
    paths = [
        f"{project_path}/data/cache/multiverse_champion.pkl",
        f"{project_path}/data/models/champion.pkl",
    ]
    for p in paths:
        if Path(p).exists():
            logger.success(f"Champion gefunden: {p}")
            return p
    logger.warning("Kein Champion-Modell gefunden")
    return None


# ── Direktstart (lokal / vast.ai) ─────────────────────────────────────────────


async def main():
    import argparse

    parser = argparse.ArgumentParser(description="Module B — Colab RL-Inferenz-Engine")
    parser.add_argument(
        "--ably-key", default=os.getenv("ABLY_API_KEY", ""), help="Ably API Key"
    )
    parser.add_argument(
        "--model", default=None, help="Pfad zum Modell (.pkl/.pth/.zip)"
    )
    parser.add_argument(
        "--symbol", default="BTCUSDT", help="Symbol (Standard: BTCUSDT)"
    )
    parser.add_argument(
        "--min-conf", type=float, default=MIN_CONFIDENCE, help="Mindest-Konfidenz"
    )
    args = parser.parse_args()

    if not args.ably_key:
        logger.error(
            "Kein Ably API Key! "
            "Setze ABLY_API_KEY in .env oder --ably-key <key>\n"
            "Kostenloser Key: https://ably.com"
        )
        sys.exit(1)

    engine = ModuleB(
        ably_key=args.ably_key,
        model_path=args.model,
        symbol=args.symbol,
        min_confidence=args.min_conf,
    )
    await engine.run()


if __name__ == "__main__":
    asyncio.run(main())
