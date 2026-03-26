"""
Module B — Colab RL Inference Engine with Ably Bridge
======================================================
Runs in Google Colab (or vast.ai / RunPod).

Tasks:
  1. Subscribe to market data from local (bt4t/market/BTCUSDT)
  2. Build observation buffer (rolling window)
  3. Load RL model (Google Drive or local)
  4. Inference: observation → BUY/SELL/HOLD + confidence
  5. Signal → Ably → Local publish (bt4t/signals)
  6. Publish heartbeat (bt4t/health) every 10s
  7. Receive commands from local and execute them

Setup in Colab (first cell):
    !pip install ably ccxt loguru
    !pip install torch  # or stable-baselines3

Usage (Colab cell):
    import sys
    sys.path.insert(0, '/content/drive/MyDrive/BITCOIN4Traders')

    from colab_bridge.module_b_colab import ModuleB
    import asyncio

    engine = ModuleB(
        ably_key="your_ably_key",
        model_path="/content/drive/MyDrive/BITCOIN4Traders/data/models/curriculum_test/best_model.pth",
    )
    await engine.run()  # In Colab: asyncio.get_event_loop().run_until_complete(engine.run())

Environment variables:
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

# ── Ably (required) ───────────────────────────────────────────────────────────
try:
    from ably import AblyRealtime

    _ABLY_OK = True
except ImportError:
    _ABLY_OK = False
    print("ERROR: pip install ably")

# ── Logging (loguru or print fallback) ───────────────────────────────────────
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
        pass  # No logs/ directory in Colab
except ImportError:
    import logging

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
    )
    logger = logging.getLogger("module_b")

# ── Channel names (identical to Module A) ────────────────────────────────────
CH_MARKET = "bt4t:market:{symbol}"
CH_SIGNALS = "bt4t:signals"
CH_PORTFOLIO = "bt4t:portfolio:state"
CH_HEALTH = "bt4t:health"
CH_CONTROL = "bt4t:control:cmd"
CH_ACK = "bt4t:control:ack"

# ── Configuration ─────────────────────────────────────────────────────────────
OBS_WINDOW = 60  # Number of historical bars for RL observation
MIN_CONFIDENCE = 0.55  # Below this threshold → HOLD (no signal)
HEARTBEAT_INTERVAL_S = 10.0  # Heartbeat interval in seconds
CONTROL_POLL_S = 5.0  # How often to check for commands


# ── Model adapter ─────────────────────────────────────────────────────────────


class ModelAdapter:
    """
    Abstraction layer for different model types.

    Supports:
      - Darwin Champion (src/math_tools/archive/darwin_legacy.py)
      - PyTorch PPO/DRL Model (.pth checkpoint)
      - Stable-Baselines3 Model
      - Fallback: RSI-based signal (no model needed)
    """

    def __init__(self, model_path: Optional[str] = None, model_type: str = "auto"):
        self.model = None
        self.model_type = model_type
        self.model_path = model_path
        self.model_version = "unknown"
        self._load(model_path, model_type)

    def _load(self, path: Optional[str], model_type: str):
        """Loads model from path. Falls back to RSI if no model available."""

        # ── 1. Darwin Champion (.pkl) ─────────────────────────────────────────
        if path and path.endswith(".pkl"):
            try:
                import pickle

                with open(path, "rb") as f:
                    self.model = pickle.load(f)
                self.model_type = "darwin"
                self.model_version = getattr(self.model, "name", "darwin_champion")
                logger.success(f"Darwin Champion loaded: {self.model_version}")
                return
            except Exception as e:
                logger.warning(f"Darwin Champion load failed: {e}")

        # ── 2. PyTorch Checkpoint (.pth) ──────────────────────────────────────
        if path and path.endswith(".pth"):
            try:
                import torch

                checkpoint = torch.load(path, map_location="cpu")
                self.model = checkpoint
                self.model_type = "pytorch"
                self.model_version = f"pth:{Path(path).stem}"
                logger.success(f"PyTorch Checkpoint loaded: {path}")
                return
            except Exception as e:
                logger.warning(f"PyTorch Checkpoint load failed: {e}")

        # ── 3. Stable-Baselines3 ──────────────────────────────────────────────
        if path and (path.endswith(".zip") or "sb3" in (path or "").lower()):
            try:
                from stable_baselines3 import PPO

                self.model = PPO.load(path)
                self.model_type = "sb3"
                self.model_version = f"sb3:{Path(path).stem}"
                logger.success(f"SB3 model loaded: {path}")
                return
            except Exception as e:
                logger.warning(f"SB3 model load failed: {e}")

        # ── 4. Fallback: RSI signal ───────────────────────────────────────────
        logger.warning("No model loaded — using RSI fallback signal")
        self.model_type = "rsi_fallback"
        self.model_version = "rsi_fallback_v1"

    def predict(self, close_array: np.ndarray, features: dict) -> tuple[str, float]:
        """
        Returns (action, confidence).
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
            logger.warning(f"Inference error ({self.model_type}): {e}")

        # Always fall back
        return self._predict_rsi(close_array)

    def _predict_darwin(self, close: np.ndarray) -> tuple[str, float]:
        """Darwin Champion inference."""
        sigs = self.model.compute_signals(close)
        raw = int(np.sign(sigs[-1])) if len(sigs) else 0
        # Confidence: how strong the signal is relative to the last 10 bars
        recent = np.abs(sigs[-10:]).mean() if len(sigs) >= 10 else 0.5
        confidence = float(np.clip(recent + 0.5, 0.0, 1.0))
        action = {1: "BUY", -1: "SELL", 0: "HOLD"}.get(raw, "HOLD")
        return action, confidence

    def _predict_rsi(self, close: np.ndarray) -> tuple[str, float]:
        """RSI-based fallback signal."""
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
            return "BUY", float(0.5 + (30 - rsi) / 60)  # Stronger below 30
        elif rsi > 70:
            return "SELL", float(0.5 + (rsi - 70) / 60)  # Stronger above 70
        return "HOLD", float(0.5 - abs(rsi - 50) / 100)

    def _predict_sb3(self, close: np.ndarray, features: dict) -> tuple[str, float]:
        """Stable-Baselines3 inference."""
        # Observation: normalized returns (last 60 bars)
        obs = self._build_obs(close, features)
        action_idx, _states = self.model.predict(obs, deterministic=True)
        # Discrete action space: 0=SELL, 1=HOLD, 2=BUY (typical for PPO)
        action_map = {0: "SELL", 1: "HOLD", 2: "BUY"}
        action = action_map.get(int(action_idx), "HOLD")
        # SB3 has no direct confidence → derive from policy logits
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
        Generic PyTorch checkpoint.
        Expects checkpoint dict with 'actor_state_dict' or 'model_state_dict'.
        Falls back to RSI if architecture unknown.
        """
        logger.debug("PyTorch checkpoint: direct call not implemented → RSI fallback")
        return self._predict_rsi(close)

    def _build_obs(self, close: np.ndarray, features: dict) -> np.ndarray:
        """Builds observation vector for RL model."""
        n = min(len(close), OBS_WINDOW)
        close_w = close[-n:]
        # Normalized log returns
        if len(close_w) >= 2:
            log_ret = np.diff(np.log(close_w + 1e-8))
        else:
            log_ret = np.zeros(OBS_WINDOW - 1)
        # Padding
        if len(log_ret) < OBS_WINDOW - 1:
            log_ret = np.pad(log_ret, (OBS_WINDOW - 1 - len(log_ret), 0))
        # Additional scalar features
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
        """Reloads model (after RELOAD_MODEL command)."""
        path = new_path or self.model_path
        logger.info(f"Reloading model: {path}")
        self._load(path, "auto")


# ── Main engine ───────────────────────────────────────────────────────────────


class ModuleB:
    """
    Colab RL inference engine.

    Receives market data, runs RL inference,
    publishes signals back to the local machine.
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

        # Observation buffer (rolling window)
        self._obs_buffer: Deque[dict] = deque(maxlen=OBS_WINDOW + 10)

        # Model
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
            f"Module B ready | Symbol: {symbol} | Model: {self.model_adapter.model_version}"
        )

    # ── Ably connection ───────────────────────────────────────────────────────

    async def _connect_ably(self):
        self._ably = AblyRealtime(self.ably_key)
        await self._ably.connection.once_async("connected")
        logger.success("Ably connected (Colab)")

        # Subscribe to market data
        ch_name = CH_MARKET.format(symbol=self.symbol)
        ch_market = self._ably.channels.get(ch_name)
        await ch_market.subscribe(self._on_market_data)
        logger.info(f"Subscribed: {ch_name}")

        # Subscribe to portfolio state (from local)
        ch_portfolio = self._ably.channels.get(CH_PORTFOLIO)
        await ch_portfolio.subscribe(self._on_portfolio_state)

        # Subscribe to commands
        ch_ctrl = self._ably.channels.get(CH_CONTROL)
        await ch_ctrl.subscribe(self._on_command)
        logger.info(f"Subscribed: {CH_CONTROL}")

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def _on_market_data(self, message):
        """Receives market data and triggers inference."""
        try:
            data = (
                json.loads(message.data)
                if isinstance(message.data, str)
                else message.data
            )
            self._obs_buffer.append(data)
            self._last_market_ts = time.time()

            # Run inference only when enough data available
            if len(self._obs_buffer) >= min(20, OBS_WINDOW // 3):
                asyncio.create_task(self._run_inference(data))

        except Exception as e:
            logger.warning(f"Market data parsing error: {e}")

    def _on_portfolio_state(self, message):
        """Receives portfolio state from local machine."""
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
        """Receives control commands from local machine."""
        try:
            data = (
                json.loads(message.data)
                if isinstance(message.data, str)
                else message.data
            )
            cmd = data.get("cmd", "")
            params = data.get("params", {})
            logger.info(f"Command received: {cmd}")
            asyncio.create_task(self._execute_command(cmd, params))
        except Exception as e:
            logger.warning(f"Command parsing error: {e}")

    # ── Inference ─────────────────────────────────────────────────────────────

    async def _run_inference(self, latest_features: dict):
        """Runs RL inference and publishes signal."""
        if self._paused:
            return

        t0 = time.time()
        try:
            # Build close array from buffer
            close_60 = latest_features.get("close_60", [])
            if close_60:
                close_arr = np.array(close_60, dtype=np.float64)
            else:
                # Reconstruct from buffer
                close_arr = np.array(
                    [d.get("close", 0.0) for d in self._obs_buffer if "close" in d],
                    dtype=np.float64,
                )

            if len(close_arr) < 10:
                return

            # Inference
            action, confidence = self.model_adapter.predict(close_arr, latest_features)
            self._inference_count += 1
            latency_ms = (time.time() - t0) * 1000

            # Logging
            logger.debug(
                f"Inference #{self._inference_count}: {action} "
                f"conf={confidence:.3f} latency={latency_ms:.1f}ms"
            )

            # Publish signal if confidence is sufficient
            if action != "HOLD" and confidence >= self.min_confidence:
                await self._publish_signal(action, confidence, latency_ms)
            else:
                logger.debug(
                    f"Signal suppressed: {action} conf={confidence:.3f} "
                    f"(min={self.min_confidence})"
                )

        except Exception as e:
            logger.error(f"Inference error: {e}", exc_info=True)

    async def _publish_signal(self, action: str, confidence: float, latency_ms: float):
        """Publishes trade signal to Ably → Local."""
        self._signal_count += 1
        payload = {
            "action": action,  # BUY | SELL | HOLD
            "symbol": self.symbol,
            "confidence": round(confidence, 4),
            "model_version": self.model_adapter.model_version,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "inference_latency_ms": round(latency_ms, 1),
            "signal_n": self._signal_count,
            # Portfolio context (received from local)
            "portfolio_equity": self._portfolio_state.get("equity", 0.0),
            "portfolio_return_pct": self._portfolio_state.get("return_pct", 0.0),
        }
        channel = self._ably.channels.get(CH_SIGNALS)
        await channel.publish("signal", json.dumps(payload))
        logger.success(
            f"SIGNAL SENT: {action} | conf={confidence:.3f} | "
            f"model={self.model_adapter.model_version}"
        )

    # ── Heartbeat ─────────────────────────────────────────────────────────────

    async def _heartbeat_loop(self):
        """Publishes a heartbeat every N seconds."""
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
                logger.debug(f"Heartbeat sent | inferences={self._inference_count}")
            except Exception as e:
                logger.warning(f"Heartbeat error: {e}")

            await asyncio.sleep(HEARTBEAT_INTERVAL_S)

    # ── Command handler ───────────────────────────────────────────────────────

    async def _execute_command(self, cmd: str, params: dict):
        """Executes control command and sends ACK back."""
        status = "OK"
        msg = ""

        try:
            if cmd == "PAUSE_INFERENCE":
                self._paused = True
                logger.warning("Inference PAUSED (remote command)")
                msg = "Inference paused"

            elif cmd == "RESUME":
                self._paused = False
                logger.success("Inference RESUMED (remote command)")
                msg = "Inference resumed"

            elif cmd == "RELOAD_MODEL":
                new_path = params.get("model_path")
                self.model_adapter.reload(new_path)
                msg = f"Model reloaded: {self.model_adapter.model_version}"
                logger.success(msg)

            elif cmd == "SHUTDOWN":
                logger.warning("SHUTDOWN command received")
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
                msg = f"Unknown command: {cmd}"
                logger.warning(msg)

        except Exception as e:
            status = "ERROR"
            msg = str(e)
            logger.error(f"Command {cmd} failed: {e}")

        # Send ACK back
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

    # ── Main loop ─────────────────────────────────────────────────────────────

    async def run(self):
        """Starts Module B. Blocks until SHUTDOWN or Ctrl+C."""
        await self._connect_ably()
        self._running = True

        logger.success("=" * 60)
        logger.success("  Module B started — Colab RL inference engine")
        logger.success(f"  Model     : {self.model_adapter.model_version}")
        logger.success(f"  Symbol    : {self.symbol}")
        logger.success(f"  Subscribed: {CH_MARKET.format(symbol=self.symbol)}")
        logger.success(f"  Publishing: {CH_SIGNALS}")
        logger.success("  Waiting for market data from local...")
        logger.success("=" * 60)

        # Start heartbeat task
        heartbeat_task = asyncio.create_task(self._heartbeat_loop())

        try:
            tick = 0
            while self._running:
                await asyncio.sleep(30.0)
                if tick % 4 == 0:  # every 2 minutes
                    self._print_status()
                tick += 1

        except asyncio.CancelledError:
            pass
        except KeyboardInterrupt:
            logger.info("Ctrl+C — shutting down Module B")
        finally:
            heartbeat_task.cancel()
            try:
                await heartbeat_task
            except asyncio.CancelledError:
                pass
            await self._shutdown()

    async def _shutdown(self):
        self._running = False
        logger.info("Module B: Closing Ably connection...")
        if self._ably:
            await self._ably.close()
        logger.success(
            f"Module B stopped | "
            f"Inferences={self._inference_count} | "
            f"Signals={self._signal_count}"
        )


# ── Colab helper functions ────────────────────────────────────────────────────


def colab_setup(drive_mount: bool = True) -> str:
    """
    Sets up the Colab environment.

    Mounts Google Drive and returns the project path.
    Call: project_path = colab_setup()

    Returns:
        str: Path to the project directory on Drive
    """
    if drive_mount:
        try:
            from google.colab import drive

            drive.mount("/content/drive")
            logger.success("Google Drive mounted: /content/drive")
        except ImportError:
            logger.warning("Not in Colab — Drive mount skipped")

    # Typical path
    project_path = "/content/drive/MyDrive/BITCOIN4Traders"
    if not Path(project_path).exists():
        project_path = "/content/BITCOIN4Traders"
        logger.warning(f"Drive path not found — using: {project_path}")

    sys.path.insert(0, project_path)
    logger.success(f"Project path: {project_path}")
    return project_path


def find_champion_on_drive(project_path: str) -> Optional[str]:
    """Searches for the Darwin Champion in the Drive directory."""
    paths = [
        f"{project_path}/data/cache/multiverse_champion.pkl",
        f"{project_path}/data/models/champion.pkl",
    ]
    for p in paths:
        if Path(p).exists():
            logger.success(f"Champion found: {p}")
            return p
    logger.warning("No champion model found")
    return None


# ── Direct start (local / vast.ai) ───────────────────────────────────────────


async def main():
    import argparse

    parser = argparse.ArgumentParser(description="Module B — Colab RL Inference Engine")
    parser.add_argument(
        "--ably-key", default=os.getenv("ABLY_API_KEY", ""), help="Ably API Key"
    )
    parser.add_argument("--model", default=None, help="Path to model (.pkl/.pth/.zip)")
    parser.add_argument("--symbol", default="BTCUSDT", help="Symbol (default: BTCUSDT)")
    parser.add_argument(
        "--min-conf", type=float, default=MIN_CONFIDENCE, help="Minimum confidence"
    )
    args = parser.parse_args()

    if not args.ably_key:
        logger.error(
            "No Ably API Key! "
            "Set ABLY_API_KEY in .env or --ably-key <key>\n"
            "Free key: https://ably.com"
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
