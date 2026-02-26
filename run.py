"""
PHASE 7 – MAIN ENTRY POINT
============================
Wires all components together and starts live operation.
Reads config from config/phase7.yaml (Hydra-compatible).

Usage:
    python -m phase7.run --config config/phase7.yaml
    python -m phase7.run --dry_run          # Paper-trading mode
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import signal
import sys
from decimal import Decimal
from pathlib import Path
from typing import Optional

import yaml

# ── Phase 7 imports ─────────────────────────
from src.connectors.binance_ws_connector import ReconnectPolicy
from src.execution.live_engine import EngineConfig, LiveExecutionEngine
from src.monitoring.monitor import EngineMonitor, TelegramNotifier

# ── Phase 5 / Feature Engine stubs ──────────
# In production, import from previous phases:
#   from src.agents.ppo_agent import PPOAgent
#   from src.features.feature_engine import FeatureEngine

logger = logging.getLogger("phase7.main")


# ─────────────────────────────────────────────
#  Stub Agent & Feature Engine for standalone test
# ─────────────────────────────────────────────


class StubFeatureEngine:
    """Minimal stub – replace with real Phase 1 FeatureEngine."""

    def transform_single(self, symbol: str, price: float):
        import numpy as np

        return np.zeros(64, dtype=np.float32)


class StubAgent:
    """Minimal stub – replace with real Phase 5 PPOAgent."""

    def predict(self, features) -> int:
        import random

        return random.choice([-1, 0, 0, 0, 1])


# ─────────────────────────────────────────────
#  Config Loader
# ─────────────────────────────────────────────


def load_config(path: str) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    # Override with env vars (12-factor app style)
    cfg.setdefault("api_key", os.environ.get("BINANCE_API_KEY", ""))
    cfg.setdefault("api_secret", os.environ.get("BINANCE_API_SECRET", ""))
    cfg.setdefault("telegram_token", os.environ.get("TELEGRAM_BOT_TOKEN", ""))
    cfg.setdefault("telegram_chat_id", os.environ.get("TELEGRAM_CHAT_ID", ""))
    return cfg


# ─────────────────────────────────────────────
#  Bootstrap
# ─────────────────────────────────────────────


async def main(config_path: str, dry_run: bool = False) -> None:
    cfg = load_config(config_path)

    # ── Logging ─────────────────────────────
    log_level = cfg.get("log_level", "INFO")
    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s │ %(levelname)-8s │ %(name)s │ %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(cfg.get("log_file", "phase7.log")),
        ],
    )

    if dry_run:
        logger.warning("═══ DRY-RUN MODE – No real orders will be placed ═══")

    # ── Engine Config ────────────────────────
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

    # ── Monitoring ───────────────────────────
    telegram: Optional[TelegramNotifier] = None
    if cfg.get("telegram_token") and cfg.get("telegram_chat_id"):
        telegram = TelegramNotifier(cfg["telegram_token"], cfg["telegram_chat_id"])

    monitor = EngineMonitor(
        telegram=telegram,
        dashboard_interval_s=cfg.get("dashboard_interval_s", 30),
    )

    # ── Agent & Features ─────────────────────
    feature_engine = StubFeatureEngine()
    agent = StubAgent()

    # Replace stubs with real implementations:
    # from src.agents.ppo_agent import PPOAgent
    # from src.features.feature_engine import FeatureEngine
    # agent = PPOAgent.load(cfg["model_checkpoint"])
    # feature_engine = FeatureEngine.load(cfg["feature_scaler_path"])

    # ── Engine ──────────────────────────────
    engine = LiveExecutionEngine(engine_cfg, agent, feature_engine)

    # ── Graceful shutdown on SIGINT/SIGTERM ──
    loop = asyncio.get_running_loop()

    def _shutdown_handler():
        logger.info("Signal received – requesting graceful shutdown…")
        asyncio.create_task(engine.stop(reason="SIGTERM/SIGINT"))

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _shutdown_handler)

    # ── Start everything ─────────────────────
    await monitor.start()
    try:
        await engine.start()
    except Exception as exc:
        logger.critical("Engine crashed: %s", exc, exc_info=True)
        monitor.on_error(f"Engine crashed: {exc}")
    finally:
        await monitor.stop()


# ─────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 7 – Live Execution Engine")
    parser.add_argument(
        "--config", default="config/phase7.yaml", help="Path to YAML config file"
    )
    parser.add_argument(
        "--dry_run", action="store_true", help="Paper-trading mode – no real orders"
    )
    args = parser.parse_args()

    if not Path(args.config).exists():
        print(f"❌ Config not found: {args.config}")
        print("   Create it with: cp config/phase7.yaml.template config/phase7.yaml")
        sys.exit(1)

    asyncio.run(main(args.config, args.dry_run))
