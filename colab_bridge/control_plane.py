"""
Control Plane — Cloudflare Tunnel + FastAPI Control
====================================================
Two parts:

  ControlServer (local):
    FastAPI server that listens locally on port 8765.
    Cloudflare Tunnel makes it accessible via a permanent URL.
    Module A and external tools send commands here.

  ControlClient (Colab):
    Polls the ControlServer every N seconds for commands.
    Executes received commands in Module B.
    Sends status updates back.

Architecture:
    Colab --[HTTP poll]--> Cloudflare Tunnel ---> localhost:8765 (ControlServer)
    Dashboard/CLI          Local                  Local

Prerequisites:
  pip install fastapi uvicorn httpx

Set up Cloudflare Tunnel (one-time, free):
    1. curl -fsSL https://pkg.cloudflare.com/cloudflare-main.gpg | sudo tee /usr/share/keyrings/cloudflare-main.gpg > /dev/null
    2. echo 'deb [signed-by=/usr/share/keyrings/cloudflare-main.gpg] https://pkg.cloudflare.com/cloudflared any main' | sudo tee /etc/apt/sources.list.d/cloudflared.list
    3. sudo apt-get update && sudo apt-get install cloudflared
    Or: wget https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -O cloudflared && chmod +x cloudflared

    Then start (in separate terminal):
    ./cloudflared tunnel --url http://localhost:8765

    Outputs a URL like: https://abc-def.trycloudflare.com
    Set this URL in Colab as CONTROL_SERVER_URL.

Environment variables:
    CONTROL_API_TOKEN=your_secret_token   # Shared secret for auth
    CONTROL_SERVER_URL=https://abc-def.trycloudflare.com  # Set in Colab

Usage:
    # Terminal 1: Cloudflare Tunnel
    ./cloudflared tunnel --url http://localhost:8765

    # Terminal 2: Start Control Server
    python colab_bridge/control_plane.py server

    # In Colab: Start Control Client (together with Module B)
    from colab_bridge.control_plane import ControlClient
    client = ControlClient(server_url=CONTROL_SERVER_URL, module_b=engine)
    asyncio.create_task(client.run())
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# ── FastAPI (for Control Server only) ────────────────────────────────────────
try:
    from fastapi import FastAPI, HTTPException, Header, Request
    from fastapi.responses import JSONResponse
    import uvicorn

    _FASTAPI_OK = True
except ImportError:
    _FASTAPI_OK = False

# ── httpx (for Control Client) ───────────────────────────────────────────────
try:
    import httpx

    _HTTPX_OK = True
except ImportError:
    _HTTPX_OK = False

try:
    from loguru import logger
except ImportError:
    import logging

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("control_plane")

# ── Configuration ─────────────────────────────────────────────────────────────
DEFAULT_PORT = 8765


# Get token from Secrets Manager (lazy import to avoid circular dependencies)
def _get_default_token() -> str:
    try:
        from src.config import get_control_token

        token = get_control_token()
        if token:
            return token
    except Exception:
        pass
    return os.getenv("CONTROL_API_TOKEN", "bt4t-secret-token")


DEFAULT_TOKEN = _get_default_token()
CONTROL_POLL_S = 5.0  # How often Colab requests commands
COMMAND_EXPIRY_S = 60.0  # Commands older than N seconds → discard


# ══════════════════════════════════════════════════════════════════════════════
# CONTROL SERVER (local)
# ══════════════════════════════════════════════════════════════════════════════


class ControlServer:
    """
    FastAPI Control Server — runs locally, accessible via Cloudflare Tunnel.

    Endpoints:
      GET  /status          → System status
      GET  /positions       → Current portfolio
      GET  /colab/command   → Next command for Colab (Colab polls here)
      POST /colab/command   → Send command to Colab
      POST /colab/status    → Colab reports status here
      GET  /health          → Health check
    """

    def __init__(
        self,
        port: int = DEFAULT_PORT,
        token: str = DEFAULT_TOKEN,
        module_a=None,  # Reference to ModuleA (optional, for portfolio state)
    ):
        if not _FASTAPI_OK:
            raise ImportError("pip install fastapi uvicorn")

        self.port = port
        self.token = token
        self.module_a = module_a

        # Command queue: commands waiting for Colab
        self._command_queue: List[dict] = []
        # Last status report from Colab
        self._colab_status: dict = {}
        # System start time
        self._start_time = time.time()

        self.app = FastAPI(
            title="BITCOIN4Traders Control API",
            description="Control for local paper trader and Colab RL engine",
            version="1.0.0",
        )
        self._register_routes()
        logger.success(f"ControlServer initialized | Port: {port}")

    def _check_auth(self, authorization: Optional[str]) -> bool:
        """Checks Bearer token."""
        if not authorization:
            return False
        parts = authorization.split(" ")
        return len(parts) == 2 and parts[1] == self.token

    def _register_routes(self):
        app = self.app

        # ── GET /health ───────────────────────────────────────────────────────
        @app.get("/health")
        async def health():
            return {"status": "ok", "timestamp": datetime.now(timezone.utc).isoformat()}

        # ── GET /status ───────────────────────────────────────────────────────
        @app.get("/status")
        async def status(authorization: Optional[str] = Header(None)):
            if not self._check_auth(authorization):
                raise HTTPException(status_code=401, detail="Unauthorized")

            uptime_s = time.time() - self._start_time
            return {
                "status": "running",
                "uptime_s": round(uptime_s, 0),
                "command_queue_length": len(self._command_queue),
                "colab_status": self._colab_status,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

        # ── GET /positions ────────────────────────────────────────────────────
        @app.get("/positions")
        async def positions(authorization: Optional[str] = Header(None)):
            if not self._check_auth(authorization):
                raise HTTPException(status_code=401, detail="Unauthorized")

            if self.module_a and hasattr(self.module_a, "portfolio"):
                price = getattr(self.module_a, "_last_price", 0.0)
                return self.module_a.portfolio.state_dict(price)
            return {"error": "module_a not connected"}

        # ── GET /colab/command ────────────────────────────────────────────────
        @app.get("/colab/command")
        async def get_colab_command(authorization: Optional[str] = Header(None)):
            """
            Colab polls this endpoint every N seconds.
            Returns the next command (or {"cmd": "NONE"}).
            """
            if not self._check_auth(authorization):
                raise HTTPException(status_code=401, detail="Unauthorized")

            # Remove expired commands
            now = time.time()
            self._command_queue = [
                c
                for c in self._command_queue
                if now - c.get("_queued_at", now) < COMMAND_EXPIRY_S
            ]

            if self._command_queue:
                cmd = self._command_queue.pop(0)
                cmd.pop("_queued_at", None)
                logger.info(f"Command delivered to Colab: {cmd.get('cmd')}")
                return cmd

            return {"cmd": "NONE"}

        # ── POST /colab/command ───────────────────────────────────────────────
        @app.post("/colab/command")
        async def post_colab_command(
            request: Request, authorization: Optional[str] = Header(None)
        ):
            """
            Sends a command to Colab (stored in queue).

            Body: {"cmd": "PAUSE_INFERENCE|RESUME|RELOAD_MODEL|SHUTDOWN|STATUS",
                   "params": {...}}
            """
            if not self._check_auth(authorization):
                raise HTTPException(status_code=401, detail="Unauthorized")

            body = await request.json()
            cmd = body.get("cmd", "")
            valid_cmds = {
                "PAUSE_INFERENCE",
                "RESUME",
                "RELOAD_MODEL",
                "SHUTDOWN",
                "STATUS",
            }
            if cmd not in valid_cmds:
                raise HTTPException(status_code=400, detail=f"Invalid command: {cmd}")

            command = {
                "cmd": cmd,
                "params": body.get("params", {}),
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "_queued_at": time.time(),
            }
            self._command_queue.append(command)
            logger.info(
                f"Command queued: {cmd} (queue length: {len(self._command_queue)})"
            )
            return {"status": "queued", "cmd": cmd}

        # ── POST /colab/status ────────────────────────────────────────────────
        @app.post("/colab/status")
        async def post_colab_status(
            request: Request, authorization: Optional[str] = Header(None)
        ):
            """Colab reports its status here."""
            if not self._check_auth(authorization):
                raise HTTPException(status_code=401, detail="Unauthorized")

            body = await request.json()
            body["_received_at"] = datetime.now(timezone.utc).isoformat()
            self._colab_status = body
            return {"status": "ok"}

        # ── POST /trading/pause ───────────────────────────────────────────────
        @app.post("/trading/pause")
        async def pause_trading(authorization: Optional[str] = Header(None)):
            """Pauses locally + sends PAUSE_INFERENCE to Colab."""
            if not self._check_auth(authorization):
                raise HTTPException(status_code=401, detail="Unauthorized")

            if self.module_a:
                self.module_a.portfolio.pause()
            self._command_queue.append(
                {
                    "cmd": "PAUSE_INFERENCE",
                    "params": {},
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "_queued_at": time.time(),
                }
            )
            return {
                "status": "paused",
                "msg": "Trading paused locally + Colab PAUSE_INFERENCE queued",
            }

        # ── POST /trading/resume ──────────────────────────────────────────────
        @app.post("/trading/resume")
        async def resume_trading(authorization: Optional[str] = Header(None)):
            """Resumes trading locally + sends RESUME to Colab."""
            if not self._check_auth(authorization):
                raise HTTPException(status_code=401, detail="Unauthorized")

            if self.module_a:
                self.module_a.portfolio.resume()
            self._command_queue.append(
                {
                    "cmd": "RESUME",
                    "params": {},
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "_queued_at": time.time(),
                }
            )
            return {"status": "resumed"}

    async def start(self):
        """Starts the FastAPI server asynchronously."""
        config = uvicorn.Config(
            self.app,
            host="0.0.0.0",
            port=self.port,
            log_level="warning",
        )
        server = uvicorn.Server(config)
        logger.success(f"Control Server started on http://0.0.0.0:{self.port}")
        logger.info(
            "Cloudflare Tunnel: ./cloudflared tunnel --url http://localhost:{self.port}"
        )
        await server.serve()


# ══════════════════════════════════════════════════════════════════════════════
# CONTROL CLIENT (Colab)
# ══════════════════════════════════════════════════════════════════════════════


class ControlClient:
    """
    Control Client — runs in Colab alongside Module B.

    Polls the ControlServer every N seconds for commands.
    Executes commands in Module B.
    Sends status updates back.
    """

    def __init__(
        self,
        server_url: str,
        module_b,  # Reference to ModuleB instance
        token: str = DEFAULT_TOKEN,
        poll_interval_s: float = CONTROL_POLL_S,
    ):
        if not _HTTPX_OK:
            raise ImportError("pip install httpx")

        # Clean up URL
        self.server_url = server_url.rstrip("/")
        self.module_b = module_b
        self.token = token
        self.poll_interval = poll_interval_s
        self._running = False
        self._headers = {"Authorization": f"Bearer {token}"}
        logger.success(f"ControlClient ready | Server: {server_url}")

    async def run(self):
        """Starts the poll loop. Run as asyncio.Task."""
        self._running = True
        logger.info("ControlClient: Starting poll loop...")

        async with httpx.AsyncClient(timeout=10.0) as client:
            while self._running:
                try:
                    await self._poll_and_execute(client)
                    await self._report_status(client)
                except Exception as e:
                    logger.warning(f"ControlClient poll error: {e}")

                await asyncio.sleep(self.poll_interval)

    async def _poll_and_execute(self, client: "httpx.AsyncClient"):
        """Requests next command and executes it."""
        resp = await client.get(
            f"{self.server_url}/colab/command",
            headers=self._headers,
        )
        if resp.status_code != 200:
            logger.warning(f"Poll error HTTP {resp.status_code}")
            return

        data = resp.json()
        cmd = data.get("cmd", "NONE")

        if cmd == "NONE":
            return  # No command — normal

        logger.info(f"Command received: {cmd}")
        params = data.get("params", {})
        await self.module_b._execute_command(cmd, params)

    async def _report_status(self, client: "httpx.AsyncClient"):
        """Sends current Module B status to the ControlServer."""
        try:
            mb = self.module_b
            status = {
                "model_version": mb.model_adapter.model_version,
                "inference_count": mb._inference_count,
                "signal_count": mb._signal_count,
                "obs_buffer_size": len(mb._obs_buffer),
                "paused": mb._paused,
                "running": mb._running,
                "last_market_data_age_s": round(time.time() - mb._last_market_ts, 1),
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            }
            await client.post(
                f"{self.server_url}/colab/status",
                headers=self._headers,
                json=status,
            )
        except Exception:
            pass  # Status report is not critical

    def stop(self):
        self._running = False


# ══════════════════════════════════════════════════════════════════════════════
# CLOUDFLARE TUNNEL HELPER
# ══════════════════════════════════════════════════════════════════════════════


async def start_cloudflare_tunnel(port: int = DEFAULT_PORT) -> Optional[str]:
    """
    Starts a cloudflared tunnel and returns the public URL.

    Prerequisite: cloudflared must be installed.
    Download: https://github.com/cloudflare/cloudflared/releases

    Returns:
        str | None: Public URL (e.g. https://abc.trycloudflare.com) or None
    """
    import subprocess
    import re

    logger.info("Starting Cloudflare Tunnel...")

    proc = await asyncio.create_subprocess_exec(
        "cloudflared",
        "tunnel",
        "--url",
        f"http://localhost:{port}",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    # Read URL from stderr (cloudflared outputs it there)
    url = None
    for _ in range(60):  # Wait max 60s
        line = await asyncio.wait_for(proc.stderr.readline(), timeout=2.0)
        text = line.decode("utf-8", errors="replace")
        match = re.search(r"https://[a-z0-9\-]+\.trycloudflare\.com", text)
        if match:
            url = match.group(0)
            break

    if url:
        logger.success(f"Cloudflare Tunnel active: {url}")
        logger.info(f"Set in Colab: CONTROL_SERVER_URL = '{url}'")
    else:
        logger.warning("Cloudflare Tunnel URL not found — timeout")

    return url


# ══════════════════════════════════════════════════════════════════════════════
# FULL START HELPER (local)
# ══════════════════════════════════════════════════════════════════════════════


async def start_full_local_stack(
    ably_key: str,
    capital: float = 10_000.0,
    symbol: str = "BTC/USDT",
    timeframe: str = "1h",
    exchange_id: str = "binance",
    poll_interval_s: float = 30.0,
    api_token: str = DEFAULT_TOKEN,
    start_tunnel: bool = True,
):
    """
    Starts the full local stack:
      1. Module A (market data + paper order executor)
      2. Control Server (FastAPI)
      3. Cloudflare Tunnel (optional)

    Usage:
        import asyncio
        from colab_bridge.control_plane import start_full_local_stack

        asyncio.run(start_full_local_stack(
            ably_key="your_ably_key",
            capital=10_000.0,
        ))
    """
    from colab_bridge.module_a_local import ModuleA

    # Create Module A
    module_a = ModuleA(
        ably_key=ably_key,
        symbol=symbol,
        timeframe=timeframe,
        exchange_id=exchange_id,
        poll_interval_s=poll_interval_s,
        initial_capital=capital,
    )

    # Create Control Server
    server = ControlServer(
        port=DEFAULT_PORT,
        token=api_token,
        module_a=module_a,
    )

    logger.success("=" * 65)
    logger.success("  BITCOIN4Traders — Local Stack")
    logger.success(f"  Module A  : {symbol} | {exchange_id} | ${capital:,.0f}")
    logger.success(f"  Control   : http://localhost:{DEFAULT_PORT}")
    logger.success("=" * 65)

    # Start tasks in parallel
    tasks = [
        asyncio.create_task(module_a.run(), name="module_a"),
        asyncio.create_task(server.start(), name="control_server"),
    ]

    if start_tunnel:
        tunnel_url = await start_cloudflare_tunnel(DEFAULT_PORT)
        if tunnel_url:
            logger.success(f"\nColab configuration:")
            logger.success(f"  CONTROL_SERVER_URL = '{tunnel_url}'")
            logger.success(f"  CONTROL_API_TOKEN  = '{api_token[:8]}...' (full token in config file)")
            logger.success(f"  ABLY_API_KEY       = 'your_key'")

    try:
        await asyncio.gather(*tasks)
    except KeyboardInterrupt:
        for t in tasks:
            t.cancel()


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════


async def _cli_server():
    """Starts only the Control Server (without Module A)."""
    port = int(os.getenv("CONTROL_PORT", DEFAULT_PORT))
    token = os.getenv("CONTROL_API_TOKEN", DEFAULT_TOKEN)
    server = ControlServer(port=port, token=token)
    await server.start()


async def _cli_tunnel():
    """Starts only the Cloudflare Tunnel."""
    port = int(os.getenv("CONTROL_PORT", DEFAULT_PORT))
    url = await start_cloudflare_tunnel(port)
    if url:
        print(f"\nCloudflare Tunnel URL: {url}")
        print(f"Set in .env: CONTROL_SERVER_URL={url}")
        # Tunnel runs in background — wait for Ctrl+C
        try:
            await asyncio.sleep(86400)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Control Plane — BITCOIN4Traders")
    sub = parser.add_subparsers(dest="cmd")

    sub.add_parser("server", help="Start Control Server")
    sub.add_parser("tunnel", help="Start Cloudflare Tunnel")

    full = sub.add_parser("full", help="Full local stack (Module A + Server + Tunnel)")
    full.add_argument(
        "--ably-key", default=os.getenv("ABLY_API_KEY", ""), required=False
    )
    full.add_argument("--capital", type=float, default=10_000.0)
    full.add_argument("--symbol", default="BTC/USDT")
    full.add_argument("--exchange", default="binance")
    full.add_argument("--interval", type=float, default=30.0)
    full.add_argument("--no-tunnel", action="store_true")

    args = parser.parse_args()

    if args.cmd == "server":
        asyncio.run(_cli_server())
    elif args.cmd == "tunnel":
        asyncio.run(_cli_tunnel())
    elif args.cmd == "full":
        if not args.ably_key:
            print("ERROR: set --ably-key or ABLY_API_KEY")
            sys.exit(1)
        asyncio.run(
            start_full_local_stack(
                ably_key=args.ably_key,
                capital=args.capital,
                symbol=args.symbol,
                exchange_id=args.exchange,
                poll_interval_s=args.interval,
                start_tunnel=not args.no_tunnel,
            )
        )
    else:
        parser.print_help()
