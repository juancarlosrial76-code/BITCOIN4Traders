"""
BITCOIN4Traders — Main Entry Point
====================================
Wires all components together and starts live operation.
Reads config from config/phase7.yaml.

Automatic model loading:
  - Looks for data/models/ppo_best.pt + data/models/champion.json
  - Falls back to StubAgent if model not yet available
  - champion.json is written by Colab_3 after successful training

Usage:
    python run.py                          # Live mode
    python run.py --dry_run               # Paper-trading, no real orders
    python run.py --config path/to.yaml   # Custom config
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import signal
import sys
from collections import deque
from decimal import Decimal
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml

# ── Project imports ──────────────────────────────────────────────────
from src.connectors.binance_ws_connector import ReconnectPolicy
from src.execution.live_engine import EngineConfig, LiveExecutionEngine
from src.monitoring.monitor import EngineMonitor, TelegramNotifier

logger = logging.getLogger("run")

# ── Paths ────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "data" / "models" / "ppo_best.pt"
CHAMPION_PATH = ROOT / "data" / "models" / "champion.json"
SCALER_PATH = ROOT / "data" / "scalers"


# ─────────────────────────────────────────────────────────────────────
#  Live PPO Agent
#  Wraps PPOAgent to expose the predict(features) -> int interface
#  that LiveExecutionEngine expects.
#  Actions: -1 = short, 0 = flat, +1 = long
# ─────────────────────────────────────────────────────────────────────


class LivePPOAgent:
    """
    Production wrapper around PPOAgent.

    - Loads model from ppo_best.pt using metadata from champion.json
    - Maintains GRU hidden state across ticks (stateful inference)
    - Maps raw PPO action index to trading signal: -1 / 0 / +1
    - Runs deterministically (no exploration noise)
    """

    # PPO was trained with 3 actions: 0=flat, 1=long, 2=short
    _ACTION_MAP = {0: 0, 1: 1, 2: -1}

    def __init__(self, model_path: Path, champion_path: Path):
        import torch
        from src.agents.ppo_agent import PPOAgent, PPOConfig

        # Load champion metadata
        if not champion_path.exists():
            raise FileNotFoundError(f"champion.json not found: {champion_path}")
        with open(champion_path) as f:
            meta = json.load(f)

        logger.info(
            "Loading PPO model | trained: %s | iter: %s | state_dim: %s",
            meta.get("trained_at", "?"),
            meta.get("n_iterations", "?"),
            meta.get("state_dim", "?"),
        )

        device_str = meta.get("device", "cpu")
        # Force CPU for live inference — GPU not needed, avoids VRAM dependency
        self._device = torch.device("cpu")

        cfg = PPOConfig(
            state_dim=meta["state_dim"],
            hidden_dim=meta.get("hidden_dim", 256),
            n_actions=meta.get("n_actions", 3),
            use_recurrent=True,
            rnn_type="GRU",
        )

        self._agent = PPOAgent(cfg, device=str(self._device))
        self._agent.load(str(model_path))
        self._agent.actor.eval()
        self._agent.critic.eval()

        self._hidden = None  # GRU hidden state — reset on each episode boundary
        logger.info("PPO model loaded from %s", model_path)

    def predict(self, features: np.ndarray) -> int:
        """
        Given a feature vector, return trading signal.

        Parameters
        ----------
        features : np.ndarray of shape (state_dim,)

        Returns
        -------
        int: -1 (short), 0 (flat), +1 (long)
        """
        action, _log_prob, _value, self._hidden = self._agent.select_action(
            features,
            hidden=self._hidden,
            deterministic=True,
        )
        return self._ACTION_MAP.get(int(action), 0)

    def reset_hidden(self):
        """Reset GRU hidden state (call at start of new episode / reconnect)."""
        self._hidden = None


# ─────────────────────────────────────────────────────────────────────
#  Live Feature Engine
#  Maintains a rolling OHLCV buffer and uses the fitted FeatureEngine
#  scaler to transform single ticks into feature vectors.
# ─────────────────────────────────────────────────────────────────────


class LiveFeatureEngine:
    """
    Production wrapper around FeatureEngine for tick-by-tick inference.

    Strategy:
    - Keeps a rolling deque of the last BUFFER_SIZE synthetic OHLCV rows
    - Each tick: append new row (open=high=low=close=price, volume=0)
    - When buffer is full: call FeatureEngine.transform(buffer_df)
    - Return the last row as the feature vector for the agent

    The scaler must have been saved by Colab_1 via FeatureEngine.save_scaler().
    If scaler is missing, falls back to zero vector (stub mode).
    """

    BUFFER_SIZE = 120  # Need >= 50 bars for OU + volatility warmup

    def __init__(self, scaler_path: Path, state_dim: int):
        from src.features.feature_engine import FeatureEngine, FeatureConfig

        self._state_dim = state_dim
        self._buffers: dict[str, deque] = {}  # one buffer per symbol

        cfg = FeatureConfig(
            volatility_window=20,
            ou_window=50,
            rolling_mean_window=20,
            use_log_returns=True,
            scaler_type="standard",
            save_scaler=False,
            scaler_path=scaler_path,
            dropna_strategy="rolling",
            min_valid_rows=1,
        )
        self._engine = FeatureEngine(cfg)

        # Load fitted scaler from disk
        scaler_file = scaler_path / "feature_scaler.pkl"
        if scaler_file.exists():
            self._engine.load_scaler()
            logger.info("Feature scaler loaded from %s", scaler_path)
        else:
            logger.warning(
                "Scaler not found at %s — features will be UNSCALED. "
                "Run Colab_1 with save_scaler=True first.",
                scaler_file,
            )

    def transform_single(self, symbol: str, price: float) -> Optional[np.ndarray]:
        """
        Append price tick to buffer and return feature vector.

        Returns None if buffer not yet full (warmup period).
        Returns np.ndarray of shape (state_dim,) when ready.
        """
        if symbol not in self._buffers:
            self._buffers[symbol] = deque(maxlen=self.BUFFER_SIZE)

        buf = self._buffers[symbol]
        ts = pd.Timestamp.utcnow()
        buf.append(
            {
                "timestamp": ts,
                "open": price,
                "high": price,
                "low": price,
                "close": price,
                "volume": 0.0,
            }
        )

        if len(buf) < self.BUFFER_SIZE:
            return None  # Not enough history yet

        df = pd.DataFrame(list(buf)).set_index("timestamp")
        df = df.astype("float32")

        try:
            feat_df = self._engine.transform(df)
        except Exception as exc:
            logger.warning("Feature transform failed: %s", exc)
            return None

        if feat_df.empty:
            return None

        row = feat_df.iloc[-1].values.astype(np.float32)

        # Pad or truncate to expected state_dim
        if len(row) < self._state_dim:
            row = np.pad(row, (0, self._state_dim - len(row)))
        elif len(row) > self._state_dim:
            row = row[: self._state_dim]

        return row


# ─────────────────────────────────────────────────────────────────────
#  Stub fallbacks (used when model is not yet trained)
# ─────────────────────────────────────────────────────────────────────


class StubFeatureEngine:
    """Zero-vector stub — active until Colab_1 scaler is available."""

    def __init__(self, state_dim: int = 64):
        self._state_dim = state_dim

    def transform_single(self, symbol: str, price: float):
        return np.zeros(self._state_dim, dtype=np.float32)


class StubAgent:
    """Hold-only stub — active until ppo_best.pt is available."""

    def predict(self, features) -> int:
        return 0  # Always flat — safe default


# ─────────────────────────────────────────────────────────────────────
#  Agent + Feature engine factory
#  Tries real model first, falls back to stubs with clear log messages
# ─────────────────────────────────────────────────────────────────────


def build_agent_and_features(cfg: dict):
    """
    Build agent and feature engine.

    Priority:
    1. Real PPOAgent + LiveFeatureEngine  (if ppo_best.pt + champion.json exist)
    2. StubAgent + StubFeatureEngine      (safe fallback, logs warning)
    """
    model_ready = MODEL_PATH.exists() and CHAMPION_PATH.exists()

    if model_ready:
        try:
            with open(CHAMPION_PATH) as f:
                meta = json.load(f)
            state_dim = meta.get("state_dim", 64)

            agent = LivePPOAgent(MODEL_PATH, CHAMPION_PATH)
            feature_engine = LiveFeatureEngine(SCALER_PATH, state_dim)
            logger.info("=== LIVE MODE: Real PPO agent loaded ===")
            return agent, feature_engine

        except Exception as exc:
            logger.error("Failed to load PPO model (%s) — falling back to stub", exc)

    # Fallback
    logger.warning("=== STUB MODE: PPO model not available yet ===")
    if not MODEL_PATH.exists():
        logger.warning("  Missing: %s", MODEL_PATH)
        logger.warning("  Run Colab_3 to train and push ppo_best.pt")
    if not CHAMPION_PATH.exists():
        logger.warning("  Missing: %s", CHAMPION_PATH)

    state_dim = cfg.get("state_dim", 64)
    return StubAgent(), StubFeatureEngine(state_dim)


# ─────────────────────────────────────────────────────────────────────
#  Config loader
# ─────────────────────────────────────────────────────────────────────


def load_config(path: str) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    # Override with env vars (12-factor app style)
    cfg.setdefault("api_key", os.environ.get("BINANCE_API_KEY", ""))
    cfg.setdefault("api_secret", os.environ.get("BINANCE_API_SECRET", ""))
    cfg.setdefault("telegram_token", os.environ.get("TELEGRAM_BOT_TOKEN", ""))
    cfg.setdefault("telegram_chat_id", os.environ.get("TELEGRAM_CHAT_ID", ""))
    return cfg


# ─────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────


async def main(config_path: str, dry_run: bool = False) -> None:
    cfg = load_config(config_path)

    # ── Logging ──────────────────────────────────────────────────────
    log_level = cfg.get("log_level", "INFO")
    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s │ %(levelname)-8s │ %(name)s │ %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(cfg.get("log_file", "logs/run.log")),
        ],
    )

    if dry_run:
        logger.warning("═══ DRY-RUN MODE — No real orders will be placed ═══")

    # ── Engine Config ─────────────────────────────────────────────────
    reconnect = ReconnectPolicy(
        max_attempts=cfg.get("reconnect_max_attempts", 10),
        base_delay_s=cfg.get("reconnect_base_delay", 1.0),
        max_delay_s=cfg.get("reconnect_max_delay", 60.0),
        backoff_factor=cfg.get("reconnect_backoff", 2.0),
    )

    engine_cfg = EngineConfig(
        symbols=cfg["symbols"],
        api_key=cfg["api_key"] if not dry_run else "test",
        api_secret=cfg["api_secret"] if not dry_run else "test",
        max_position_usd=Decimal(str(cfg.get("max_position_usd", 10_000))),
        max_order_usd=Decimal(str(cfg.get("max_order_usd", 2_000))),
        circuit_breaker_pct=Decimal(str(cfg.get("circuit_breaker_pct", 0.02))),
        daily_loss_limit_usd=Decimal(str(cfg.get("daily_loss_limit_usd", 500))),
        use_limit_orders=cfg.get("use_limit_orders", True),
        limit_order_offset_bps=cfg.get("limit_order_offset_bps", 2),
        order_timeout_s=cfg.get("order_timeout_s", 30.0),
        reconnect_policy=reconnect,
    )

    # ── Monitoring ────────────────────────────────────────────────────
    telegram: Optional[TelegramNotifier] = None
    if cfg.get("telegram_token") and cfg.get("telegram_chat_id"):
        telegram = TelegramNotifier(cfg["telegram_token"], cfg["telegram_chat_id"])

    monitor = EngineMonitor(
        telegram=telegram,
        dashboard_interval_s=cfg.get("dashboard_interval_s", 30),
    )

    # ── Agent & Feature Engine ────────────────────────────────────────
    agent, feature_engine = build_agent_and_features(cfg)

    # ── Engine ───────────────────────────────────────────────────────
    engine = LiveExecutionEngine(engine_cfg, agent, feature_engine)

    # ── Graceful shutdown on SIGINT/SIGTERM ───────────────────────────
    loop = asyncio.get_running_loop()

    def _shutdown_handler():
        logger.info("Signal received — requesting graceful shutdown...")
        asyncio.create_task(engine.stop(reason="SIGTERM/SIGINT"))

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _shutdown_handler)

    # ── Start ─────────────────────────────────────────────────────────
    await monitor.start()
    try:
        await engine.start()
    except Exception as exc:
        logger.critical("Engine crashed: %s", exc, exc_info=True)
        monitor.on_error(f"Engine crashed: {exc}")
    finally:
        await monitor.stop()


# ─────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="BITCOIN4Traders — Live Execution Engine"
    )
    parser.add_argument(
        "--config", default="config/phase7.yaml", help="Path to YAML config file"
    )
    parser.add_argument(
        "--dry_run", action="store_true", help="Paper-trading mode — no real orders"
    )
    args = parser.parse_args()

    if not Path(args.config).exists():
        print(f"Config not found: {args.config}")
        print("Create it with: cp config/phase7.yaml.template config/phase7.yaml")
        sys.exit(1)

    # Ensure logs directory exists
    Path("logs").mkdir(exist_ok=True)

    asyncio.run(main(args.config, args.dry_run))
