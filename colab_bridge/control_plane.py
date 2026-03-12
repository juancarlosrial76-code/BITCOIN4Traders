"""
Control-Plane — Cloudflare Tunnel + FastAPI Steuerung
======================================================
Zwei Teile:

  ControlServer (lokal):
    FastAPI-Server der lokal auf Port 8765 lauscht.
    Cloudflare Tunnel macht ihn unter einer permanenten URL erreichbar.
    Module A und externe Tools senden Befehle hierher.

  ControlClient (Colab):
    Pollt alle N Sekunden den ControlServer nach Befehlen.
    Führt empfangene Befehle in Module B aus.
    Sendet Status-Updates zurück.

Architektur:
    Colab --[HTTP poll]--> Cloudflare Tunnel ---> localhost:8765 (ControlServer)
    Dashboard/CLI          Lokal                  Lokal

Voraussetzungen:
  pip install fastapi uvicorn httpx

Cloudflare Tunnel einrichten (einmalig, kostenlos):
    1. curl -fsSL https://pkg.cloudflare.com/cloudflare-main.gpg | sudo tee /usr/share/keyrings/cloudflare-main.gpg > /dev/null
    2. echo 'deb [signed-by=/usr/share/keyrings/cloudflare-main.gpg] https://pkg.cloudflare.com/cloudflared any main' | sudo tee /etc/apt/sources.list.d/cloudflared.list
    3. sudo apt-get update && sudo apt-get install cloudflared
    Oder: wget https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -O cloudflared && chmod +x cloudflared

    Dann starten (in separatem Terminal):
    ./cloudflared tunnel --url http://localhost:8765

    Gibt eine URL aus wie: https://abc-def.trycloudflare.com
    Diese URL in Colab als CONTROL_SERVER_URL setzen.

Umgebungsvariablen:
    CONTROL_API_TOKEN=your_secret_token   # Shared secret für Auth
    CONTROL_SERVER_URL=https://abc-def.trycloudflare.com  # In Colab setzen

Verwendung:
    # Terminal 1: Cloudflare Tunnel
    ./cloudflared tunnel --url http://localhost:8765

    # Terminal 2: Control Server starten
    python colab_bridge/control_plane.py server

    # In Colab: Control Client starten (zusammen mit Module B)
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

# ── FastAPI (nur für Control Server) ─────────────────────────────────────────
try:
    from fastapi import FastAPI, HTTPException, Header, Request
    from fastapi.responses import JSONResponse
    import uvicorn

    _FASTAPI_OK = True
except ImportError:
    _FASTAPI_OK = False

# ── httpx (für Control Client) ───────────────────────────────────────────────
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

# ── Konfiguration ─────────────────────────────────────────────────────────────
DEFAULT_PORT = 8765
DEFAULT_TOKEN = os.getenv("CONTROL_API_TOKEN", "bt4t-secret-token")
CONTROL_POLL_S = 5.0  # Wie oft Colab nach Befehlen fragt
COMMAND_EXPIRY_S = 60.0  # Befehle älter als N Sekunden → verwerfen


# ══════════════════════════════════════════════════════════════════════════════
# CONTROL SERVER (lokal)
# ══════════════════════════════════════════════════════════════════════════════


class ControlServer:
    """
    FastAPI Control Server — läuft lokal, erreichbar via Cloudflare Tunnel.

    Endpoints:
      GET  /status          → System-Status
      GET  /positions       → Aktuelles Portfolio
      GET  /colab/command   → Nächster Befehl für Colab (Colab pollt hier)
      POST /colab/command   → Befehl an Colab senden
      POST /colab/status    → Colab reportet Status hierher
      GET  /health          → Healthcheck
    """

    def __init__(
        self,
        port: int = DEFAULT_PORT,
        token: str = DEFAULT_TOKEN,
        module_a=None,  # Referenz auf ModuleA (optional, für Portfolio-State)
    ):
        if not _FASTAPI_OK:
            raise ImportError("pip install fastapi uvicorn")

        self.port = port
        self.token = token
        self.module_a = module_a

        # Command Queue: Befehle die auf Colab warten
        self._command_queue: List[dict] = []
        # Letzte Status-Meldung von Colab
        self._colab_status: dict = {}
        # System-Start Zeit
        self._start_time = time.time()

        self.app = FastAPI(
            title="BITCOIN4Traders Control API",
            description="Steuerung für lokalen Paper Trader und Colab RL-Engine",
            version="1.0.0",
        )
        self._register_routes()
        logger.success(f"ControlServer initialisiert | Port: {port}")

    def _check_auth(self, authorization: Optional[str]) -> bool:
        """Prüft Bearer Token."""
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
            Colab pollt diesen Endpoint alle N Sekunden.
            Gibt den nächsten Befehl zurück (oder {"cmd": "NONE"}).
            """
            if not self._check_auth(authorization):
                raise HTTPException(status_code=401, detail="Unauthorized")

            # Abgelaufene Befehle entfernen
            now = time.time()
            self._command_queue = [
                c
                for c in self._command_queue
                if now - c.get("_queued_at", now) < COMMAND_EXPIRY_S
            ]

            if self._command_queue:
                cmd = self._command_queue.pop(0)
                cmd.pop("_queued_at", None)
                logger.info(f"Befehl an Colab ausgeliefert: {cmd.get('cmd')}")
                return cmd

            return {"cmd": "NONE"}

        # ── POST /colab/command ───────────────────────────────────────────────
        @app.post("/colab/command")
        async def post_colab_command(
            request: Request, authorization: Optional[str] = Header(None)
        ):
            """
            Sendet einen Befehl an Colab (wird in Queue gespeichert).

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
                raise HTTPException(status_code=400, detail=f"Ungültiger Befehl: {cmd}")

            command = {
                "cmd": cmd,
                "params": body.get("params", {}),
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "_queued_at": time.time(),
            }
            self._command_queue.append(command)
            logger.info(
                f"Befehl in Queue: {cmd} (Queue-Länge: {len(self._command_queue)})"
            )
            return {"status": "queued", "cmd": cmd}

        # ── POST /colab/status ────────────────────────────────────────────────
        @app.post("/colab/status")
        async def post_colab_status(
            request: Request, authorization: Optional[str] = Header(None)
        ):
            """Colab reportet seinen Status hierher."""
            if not self._check_auth(authorization):
                raise HTTPException(status_code=401, detail="Unauthorized")

            body = await request.json()
            body["_received_at"] = datetime.now(timezone.utc).isoformat()
            self._colab_status = body
            return {"status": "ok"}

        # ── POST /trading/pause ───────────────────────────────────────────────
        @app.post("/trading/pause")
        async def pause_trading(authorization: Optional[str] = Header(None)):
            """Pausiert lokal + sendet PAUSE_INFERENCE an Colab."""
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
                "msg": "Trading lokal pausiert + Colab PAUSE_INFERENCE in Queue",
            }

        # ── POST /trading/resume ──────────────────────────────────────────────
        @app.post("/trading/resume")
        async def resume_trading(authorization: Optional[str] = Header(None)):
            """Setzt Trading fort lokal + RESUME an Colab."""
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
        """Startet den FastAPI-Server asynchron."""
        config = uvicorn.Config(
            self.app,
            host="0.0.0.0",
            port=self.port,
            log_level="warning",
        )
        server = uvicorn.Server(config)
        logger.success(f"Control Server gestartet auf http://0.0.0.0:{self.port}")
        logger.info(
            "Cloudflare Tunnel: ./cloudflared tunnel --url http://localhost:{self.port}"
        )
        await server.serve()


# ══════════════════════════════════════════════════════════════════════════════
# CONTROL CLIENT (Colab)
# ══════════════════════════════════════════════════════════════════════════════


class ControlClient:
    """
    Control Client — läuft in Colab neben Module B.

    Pollt alle N Sekunden den ControlServer nach Befehlen.
    Führt Befehle in Module B aus.
    Sendet Status-Updates zurück.
    """

    def __init__(
        self,
        server_url: str,
        module_b,  # Referenz auf ModuleB-Instanz
        token: str = DEFAULT_TOKEN,
        poll_interval_s: float = CONTROL_POLL_S,
    ):
        if not _HTTPX_OK:
            raise ImportError("pip install httpx")

        # URL bereinigen
        self.server_url = server_url.rstrip("/")
        self.module_b = module_b
        self.token = token
        self.poll_interval = poll_interval_s
        self._running = False
        self._headers = {"Authorization": f"Bearer {token}"}
        logger.success(f"ControlClient bereit | Server: {server_url}")

    async def run(self):
        """Startet den Poll-Loop. Als asyncio.Task ausführen."""
        self._running = True
        logger.info("ControlClient: Starte Poll-Loop...")

        async with httpx.AsyncClient(timeout=10.0) as client:
            while self._running:
                try:
                    await self._poll_and_execute(client)
                    await self._report_status(client)
                except Exception as e:
                    logger.warning(f"ControlClient Poll-Fehler: {e}")

                await asyncio.sleep(self.poll_interval)

    async def _poll_and_execute(self, client: "httpx.AsyncClient"):
        """Fragt nach nächstem Befehl und führt ihn aus."""
        resp = await client.get(
            f"{self.server_url}/colab/command",
            headers=self._headers,
        )
        if resp.status_code != 200:
            logger.warning(f"Poll Fehler HTTP {resp.status_code}")
            return

        data = resp.json()
        cmd = data.get("cmd", "NONE")

        if cmd == "NONE":
            return  # Kein Befehl — normal

        logger.info(f"Befehl empfangen: {cmd}")
        params = data.get("params", {})
        await self.module_b._execute_command(cmd, params)

    async def _report_status(self, client: "httpx.AsyncClient"):
        """Sendet aktuellen Module-B-Status an den ControlServer."""
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
            pass  # Status-Report ist nicht kritisch

    def stop(self):
        self._running = False


# ══════════════════════════════════════════════════════════════════════════════
# CLOUDFLARE TUNNEL HELPER
# ══════════════════════════════════════════════════════════════════════════════


async def start_cloudflare_tunnel(port: int = DEFAULT_PORT) -> Optional[str]:
    """
    Startet cloudflared Tunnel und gibt die öffentliche URL zurück.

    Voraussetzung: cloudflared muss installiert sein.
    Download: https://github.com/cloudflare/cloudflared/releases

    Returns:
        str | None: Öffentliche URL (z.B. https://abc.trycloudflare.com) oder None
    """
    import subprocess
    import re

    logger.info("Starte Cloudflare Tunnel...")

    proc = await asyncio.create_subprocess_exec(
        "cloudflared",
        "tunnel",
        "--url",
        f"http://localhost:{port}",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    # URL aus stderr lesen (cloudflared gibt sie dort aus)
    url = None
    for _ in range(60):  # Max 60s warten
        line = await asyncio.wait_for(proc.stderr.readline(), timeout=2.0)
        text = line.decode("utf-8", errors="replace")
        match = re.search(r"https://[a-z0-9\-]+\.trycloudflare\.com", text)
        if match:
            url = match.group(0)
            break

    if url:
        logger.success(f"Cloudflare Tunnel aktiv: {url}")
        logger.info(f"In Colab setzen: CONTROL_SERVER_URL = '{url}'")
    else:
        logger.warning("Cloudflare Tunnel URL nicht gefunden — Zeitüberschreitung")

    return url


# ══════════════════════════════════════════════════════════════════════════════
# VOLLSTÄNDIGER START-HELPER (lokal)
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
    Startet den vollständigen lokalen Stack:
      1. Module A (Marktdaten + Paper-Order-Executor)
      2. Control Server (FastAPI)
      3. Cloudflare Tunnel (optional)

    Verwendung:
        import asyncio
        from colab_bridge.control_plane import start_full_local_stack

        asyncio.run(start_full_local_stack(
            ably_key="your_ably_key",
            capital=10_000.0,
        ))
    """
    from colab_bridge.module_a_local import ModuleA

    # Module A erstellen
    module_a = ModuleA(
        ably_key=ably_key,
        symbol=symbol,
        timeframe=timeframe,
        exchange_id=exchange_id,
        poll_interval_s=poll_interval_s,
        initial_capital=capital,
    )

    # Control Server erstellen
    server = ControlServer(
        port=DEFAULT_PORT,
        token=api_token,
        module_a=module_a,
    )

    logger.success("=" * 65)
    logger.success("  BITCOIN4Traders — Lokaler Stack")
    logger.success(f"  Module A  : {symbol} | {exchange_id} | ${capital:,.0f}")
    logger.success(f"  Control   : http://localhost:{DEFAULT_PORT}")
    logger.success("=" * 65)

    # Tasks parallel starten
    tasks = [
        asyncio.create_task(module_a.run(), name="module_a"),
        asyncio.create_task(server.start(), name="control_server"),
    ]

    if start_tunnel:
        tunnel_url = await start_cloudflare_tunnel(DEFAULT_PORT)
        if tunnel_url:
            logger.success(f"\nColab-Konfiguration:")
            logger.success(f"  CONTROL_SERVER_URL = '{tunnel_url}'")
            logger.success(f"  CONTROL_API_TOKEN  = '{api_token}'")
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
    """Startet nur den Control Server (ohne Module A)."""
    port = int(os.getenv("CONTROL_PORT", DEFAULT_PORT))
    token = os.getenv("CONTROL_API_TOKEN", DEFAULT_TOKEN)
    server = ControlServer(port=port, token=token)
    await server.start()


async def _cli_tunnel():
    """Startet nur den Cloudflare Tunnel."""
    port = int(os.getenv("CONTROL_PORT", DEFAULT_PORT))
    url = await start_cloudflare_tunnel(port)
    if url:
        print(f"\nCloudflare Tunnel URL: {url}")
        print(f"In .env setzen: CONTROL_SERVER_URL={url}")
        # Tunnel läuft im Hintergrund — warte auf Ctrl+C
        try:
            await asyncio.sleep(86400)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Control Plane — BITCOIN4Traders")
    sub = parser.add_subparsers(dest="cmd")

    sub.add_parser("server", help="Control Server starten")
    sub.add_parser("tunnel", help="Cloudflare Tunnel starten")

    full = sub.add_parser(
        "full", help="Vollständiger lokaler Stack (Module A + Server + Tunnel)"
    )
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
            print("FEHLER: --ably-key oder ABLY_API_KEY setzen")
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
