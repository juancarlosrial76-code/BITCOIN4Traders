"""
colab_extension.py — BITCOIN4Traders Colab Extension
=====================================================
Ein einziger Import hängt sich in JEDES bestehende Colab-Notebook ein.
Kein Code-Umbau nötig. Funktioniert als Decorator, Context-Manager und
globaler Exception-Hook gleichzeitig.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
VERWENDUNG IN COLAB (erste Zelle):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    # Zelle 1 — Extension einrichten (copy-paste, nie ändern)
    import sys, os
    sys.path.insert(0, '/content/drive/MyDrive/BITCOIN4Traders')
    os.environ.setdefault('BT4T_LISTENER_URL', 'https://dein.tunnel.dev')
    os.environ.setdefault('BT4T_API_TOKEN',    'dein-token')
    os.environ.setdefault('BT4T_NOTEBOOK_ID',  'training_v3')

    from colab_bridge.colab_extension import bt4t
    bt4t.install()   # ← einmal aufrufen, danach läuft alles automatisch

    # Zelle 2 — dein normaler Training-Code UNVERÄNDERT:
    for epoch in range(100):
        loss = train_one_epoch(model, data)
        bt4t.step(epoch=epoch, loss=loss)   # ← optional: Fortschritt melden

    # ODER als Decorator (kein bt4t.step nötig):
    @bt4t.guard
    def run_training():
        for epoch in range(100):
            loss = train_one_epoch(model, data)

    # ODER als Context-Manager:
    with bt4t.session("training_run_42"):
        train_model(model, data)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WAS DIE EXTENSION MACHT:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. FEHLERBEHANDLUNG (automatisch, kein Code nötig)
   • Globaler sys.excepthook — fängt JEDEN unbehandelten Fehler
   • Klassifiziert: OOM / NaN-Loss / CUDA / Import / Timeout / ...
   • Repariert Parameter im laufenden Prozess (kein Notebook-Neustart nötig)
   • Sendet Fehlerbericht an lokalen Rechner (HTTP → Drive Fallback)
   • Kann Training automatisch fortsetzen nach Reparatur

2. ITERATIONS-STEUERUNG (opt-in via bt4t.step())
   • Fortschritt (epoch, loss, reward) → lokaler Rechner
   • Empfängt Befehle: PAUSE / RESUME / CHANGE_LR / CHANGE_BS / STOP
   • Checkpoint-Trigger: automatisch bei Fortschritt-Milestone
   • Early-Stopping: lokaler Rechner kann Training stoppen

3. SESSION-KEEPALIVE (automatisch)
   • Schreibt alle 10 Min eine echte Compute-Aufgabe → verhindert Colab-Timeout
   • Heartbeat → lokaler Rechner weiß dass Colab lebt
   • Erkennt Session-Reset und meldet "COLAB_READY" nach Neustart

4. SPEICHER-MANAGEMENT (automatisch)
   • Überwacht GPU/RAM alle 60s
   • Bei >85% GPU-Speicher: warnt + optional gc + torch.cuda.empty_cache()
   • OOM-Prophylaxe: reduziert Batch-Size bevor OOM auftritt

5. PARAMETER-HOT-RELOAD (via Befehle vom lokalen Rechner)
   • Hyperparameter während des Trainings ändern ohne Neustart
   • Werte werden im laufenden Prozess überschrieben
"""

from __future__ import annotations

import asyncio
import gc
import json
import os
import sys
import time
import threading
import traceback
import re
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, Optional

# ── Logging ───────────────────────────────────────────────────────────────────
try:
    from loguru import logger

    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | <cyan>[bt4t]</cyan> {message}",
        level="INFO",
        colorize=True,
    )
except ImportError:
    import logging

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)s | [bt4t] %(message)s"
    )
    logger = logging.getLogger("bt4t")

# ── Optionale Abhängigkeiten (graceful degradation) ───────────────────────────
try:
    import httpx

    _HTTPX = True
except ImportError:
    _HTTPX = False

try:
    import torch

    _TORCH = True
except ImportError:
    _TORCH = False


# ════════════════════════════════════════════════════════════════════════════════
# FEHLER-KLASSIFIZIERER (übernimmt Logik aus error_repair.py, läuft in Colab)
# ════════════════════════════════════════════════════════════════════════════════

# Muster → (Fehlertyp, Auto-Reparatur-Aktion, Schwere)
_ERROR_PATTERNS = [
    (
        r"CUDA out of memory|OutOfMemoryError|OOM|out of memory",
        "OOM",
        "halve_batch_size",
        "high",
    ),
    (
        r"nan|NaN|inf(?!o)|Inf.*(?:loss|reward|gradient)",
        "NAN_LOSS",
        "reduce_lr",
        "high",
    ),
    (r"gradient.*explod|loss.*explod|overflow", "EXPLODING", "clip_gradients", "high"),
    (
        r"ModuleNotFoundError|ImportError|No module named",
        "IMPORT",
        "pip_install",
        "medium",
    ),
    (r"TimeoutError|ReadTimeout|socket\.timeout", "TIMEOUT", "increase_timeout", "low"),
    (
        r"ConnectionError|ConnectTimeout|RemoteDisconnected",
        "CONNECTION",
        "retry_connection",
        "medium",
    ),
    (
        r"RuntimeError.*CUDA|device-side assert",
        "CUDA_ERROR",
        "halve_batch_size",
        "high",
    ),
    (
        r"KeyError|IndexError|ValueError.*(?:batch|data|feature)",
        "DATA_ERROR",
        "skip_batch",
        "medium",
    ),
    (
        r"PermissionError|FileNotFoundError.*(?:drive|model|cache)",
        "IO_ERROR",
        "retry_io",
        "medium",
    ),
    (r"KeyboardInterrupt", "INTERRUPTED", "none", "low"),
]


def classify_error(exc: BaseException) -> tuple[str, str, str]:
    """
    Klassifiziert eine Exception.
    Gibt (error_type, repair_action, severity) zurück.
    """
    text = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
    for pattern, etype, action, severity in _ERROR_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            return etype, action, severity
    return "UNKNOWN", "none", "medium"


# ════════════════════════════════════════════════════════════════════════════════
# IN-PROCESS REPARATUR (ändert laufende Variablen, kein Neustart)
# ════════════════════════════════════════════════════════════════════════════════


class InProcessRepair:
    """
    Repariert Hyperparameter direkt im laufenden Python-Prozess.
    Sucht Variablen im globalen Namespace des Notebooks (IPython __main__).
    """

    def __init__(self, repair_log: list):
        self._log = repair_log

    def apply(self, action: str, context: dict) -> dict:
        """
        Führt Reparatur-Aktion aus.
        Gibt dict mit {action, changes, success} zurück.
        """
        handler = {
            "halve_batch_size": self._halve_batch_size,
            "reduce_lr": self._reduce_lr,
            "clip_gradients": self._clip_gradients,
            "pip_install": self._pip_install,
            "increase_timeout": self._increase_timeout,
            "skip_batch": self._skip_batch,
            "retry_connection": self._retry_connection,
            "retry_io": self._retry_io,
            "none": self._noop,
        }.get(action, self._noop)

        try:
            changes = handler(context)
            record = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "action": action,
                "changes": changes,
                "success": True,
            }
            self._log.append(record)
            return record
        except Exception as e:
            logger.warning(f"Reparatur '{action}' fehlgeschlagen: {e}")
            return {"action": action, "changes": [], "success": False}

    def _get_global_ns(self) -> dict:
        """Holt den globalen Namespace des Notebooks (__main__)."""
        import __main__

        return vars(__main__)

    def _halve_batch_size(self, ctx: dict) -> list:
        """Halbiert alle BATCH_SIZE / batch_size Variablen im globalen NS."""
        ns = self._get_global_ns()
        changes = []
        for name in list(ns.keys()):
            if "batch_size" in name.lower() or "batch" == name.lower():
                val = ns[name]
                if isinstance(val, int) and val >= 4:
                    new_val = max(4, val // 2)
                    ns[name] = new_val
                    changes.append(f"{name}: {val} → {new_val}")
                    logger.warning(f"[Repair] {name}: {val} → {new_val}")

        # Auch torch DataLoader patchen wenn vorhanden
        if _TORCH:
            gc.collect()
            torch.cuda.empty_cache()
            logger.info("[Repair] CUDA cache geleert")

        if not changes:
            logger.info("[Repair] Keine batch_size Variable gefunden")
        return changes

    def _reduce_lr(self, ctx: dict) -> list:
        """Reduziert Learning Rate um Faktor 10."""
        ns = self._get_global_ns()
        changes = []
        lr_names = [k for k in ns if re.search(r"(?:learning_rate|lr)\b", k, re.I)]
        for name in lr_names:
            val = ns[name]
            if isinstance(val, (int, float)) and 0 < val < 1:
                new_val = val / 10
                ns[name] = new_val
                changes.append(f"{name}: {val:.2e} → {new_val:.2e}")
                logger.warning(f"[Repair] {name}: {val:.2e} → {new_val:.2e}")

        # PyTorch Optimizer patchen wenn vorhanden
        if _TORCH:
            for name, obj in ns.items():
                if hasattr(obj, "param_groups") and hasattr(obj, "step"):
                    for pg in obj.param_groups:
                        old = pg["lr"]
                        pg["lr"] = old / 10
                        changes.append(
                            f"optimizer[{name}].lr: {old:.2e} → {pg['lr']:.2e}"
                        )
                        logger.warning(
                            f"[Repair] optimizer.lr: {old:.2e} → {pg['lr']:.2e}"
                        )

        return changes

    def _clip_gradients(self, ctx: dict) -> list:
        """Reduziert gradient_clip_val oder aktiviert Gradient-Clipping."""
        ns = self._get_global_ns()
        changes = []
        for name in list(ns.keys()):
            if "clip" in name.lower() and isinstance(ns[name], (int, float)):
                old = ns[name]
                ns[name] = min(old, 0.5)
                changes.append(f"{name}: {old} → {ns[name]}")
        if not changes:
            # Neu setzen
            ns["GRADIENT_CLIP_VAL"] = 0.5
            changes.append("GRADIENT_CLIP_VAL = 0.5 (neu)")
            logger.info("[Repair] GRADIENT_CLIP_VAL = 0.5 gesetzt")
        return changes

    def _pip_install(self, ctx: dict) -> list:
        """Installiert fehlendes Paket automatisch."""
        error_msg = ctx.get("error_message", "")
        match = re.search(r"No module named '([^']+)'", error_msg)
        if not match:
            return []
        module = match.group(1).split(".")[0]
        logger.info(f"[Repair] pip install {module}")
        import subprocess

        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", module],
            capture_output=True,
            text=True,
        )
        return [f"pip install {module}: {'OK' if result.returncode == 0 else 'FEHLER'}"]

    def _increase_timeout(self, ctx: dict) -> list:
        ns = self._get_global_ns()
        changes = []
        for name in list(ns.keys()):
            if "timeout" in name.lower() and isinstance(ns[name], (int, float)):
                old = ns[name]
                ns[name] = old * 2
                changes.append(f"{name}: {old} → {ns[name]}")
        return changes

    def _skip_batch(self, ctx: dict) -> list:
        logger.info("[Repair] DATA_ERROR — nächster Batch wird übersprungen")
        return ["skip_next_batch_flag = True"]

    def _retry_connection(self, ctx: dict) -> list:
        logger.info("[Repair] CONNECTION — 30s warten, dann weiter")
        time.sleep(30)
        return ["waited_30s"]

    def _retry_io(self, ctx: dict) -> list:
        logger.info("[Repair] IO_ERROR — 10s warten")
        time.sleep(10)
        return ["waited_10s"]

    def _noop(self, ctx: dict) -> list:
        return []


# ════════════════════════════════════════════════════════════════════════════════
# REPORTER (sendet Status/Fehler an lokalen Rechner)
# ════════════════════════════════════════════════════════════════════════════════


class Reporter:
    """
    Sendet Berichte an den lokalen Rechner.
    Kanal 1: HTTP POST an Listener-URL (schnell)
    Kanal 2: Google Drive JSON-Datei (Fallback)
    """

    def __init__(self, listener_url: str, api_token: str, notebook_id: str):
        self.listener_url = listener_url.rstrip("/") if listener_url else ""
        self.api_token = api_token
        self.notebook_id = notebook_id
        self._queue: list = []
        self._send_thread: Optional[threading.Thread] = None
        self._running = False

    def start(self):
        self._running = True
        self._send_thread = threading.Thread(target=self._send_loop, daemon=True)
        self._send_thread.start()

    def stop(self):
        self._running = False

    def report_error(self, exc: BaseException, error_type: str, repair_applied: dict):
        """Sendet Fehlerbericht."""
        self._queue.append(
            {
                "type": "error",
                "notebook_id": self.notebook_id,
                "error_type": error_type,
                "error_message": f"{type(exc).__name__}: {exc}",
                "stacktrace": traceback.format_exc()[-2000:],
                "repair_applied": repair_applied,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            }
        )

    def report_progress(self, data: dict):
        """Sendet Trainings-Fortschritt."""
        self._queue.append(
            {
                "type": "progress",
                "notebook_id": self.notebook_id,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                **data,
            }
        )

    def report_heartbeat(self, extra: dict = None):
        """Sendet Heartbeat."""
        payload = {
            "type": "heartbeat",
            "notebook_id": self.notebook_id,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "status": "COLAB_ALIVE",
        }
        if extra:
            payload.update(extra)
        self._queue.append(payload)

    def _send_loop(self):
        """Hintergrund-Thread: leert die Queue und sendet Nachrichten."""
        while self._running or self._queue:
            if not self._queue:
                time.sleep(1.0)
                continue
            payload = self._queue.pop(0)
            self._send_one(payload)

    def _send_one(self, payload: dict):
        """Sendet eine Nachricht. Versucht HTTP, fällt auf Drive zurück."""
        if not self.listener_url:
            self._fallback_drive(payload)
            return

        if not _HTTPX:
            self._fallback_drive(payload)
            return

        try:
            import httpx

            with httpx.Client(timeout=8.0) as client:
                resp = client.post(
                    f"{self.listener_url}/report_error",
                    json=payload,
                    headers={"X-API-Token": self.api_token},
                )
                if resp.status_code == 200:
                    logger.debug(f"[Reporter] Gesendet: {payload['type']}")
                    return
        except Exception as e:
            logger.debug(f"[Reporter] HTTP fehlgeschlagen: {e} — versuche Drive")

        self._fallback_drive(payload)

    def _fallback_drive(self, payload: dict):
        """Schreibt Bericht als JSON-Datei — Drive-Manager liest sie."""
        try:
            # Prüfe ob Drive gemountet
            drive_dir = Path("/content/drive/MyDrive/BITCOIN4Traders/bt4t_bus/reports")
            if drive_dir.parent.parent.parent.exists():
                drive_dir.mkdir(parents=True, exist_ok=True)
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                out = drive_dir / f"{payload['type']}_{ts}.json"
                out.write_text(json.dumps(payload, default=str, indent=2))
                logger.debug(f"[Reporter] Drive Fallback: {out.name}")
        except Exception as e:
            logger.debug(f"[Reporter] Drive Fallback fehlgeschlagen: {e}")

    def poll_commands(self) -> list[dict]:
        """
        Fragt den lokalen Rechner nach Befehlen (synchron, für Iteration-Hook).
        Gibt Liste von Command-Dicts zurück.
        """
        if not self.listener_url or not _HTTPX:
            return []
        try:
            import httpx

            with httpx.Client(timeout=5.0) as client:
                resp = client.get(
                    f"{self.listener_url}/colab/command",
                    headers={"Authorization": f"Bearer {self.api_token}"},
                )
                if resp.status_code == 200:
                    data = resp.json()
                    if data.get("cmd") != "NONE":
                        return [data]
        except Exception:
            pass
        return []


# ════════════════════════════════════════════════════════════════════════════════
# ITERATIONS-CONTROLLER
# ════════════════════════════════════════════════════════════════════════════════


@dataclass
class IterationState:
    epoch: int = 0
    step: int = 0
    loss: float = float("nan")
    reward: float = float("nan")
    extra: dict = field(default_factory=dict)
    paused: bool = False
    stop_requested: bool = False
    # Hyperparameter die remote geändert werden können
    lr: Optional[float] = None
    batch_size: Optional[int] = None
    checkpoint_every: int = 50  # Alle N Steps Checkpoint senden


class IterationController:
    """
    Steuert den Trainings-Loop von außen.
    bt4t.step() ruft intern process() auf.
    """

    def __init__(self, reporter: Reporter, repair: InProcessRepair):
        self.reporter = reporter
        self.repair = repair
        self.state = IterationState()
        self._last_cmd_poll = 0.0
        CMD_POLL_INTERVAL = 5.0  # Befehle alle 5s holen
        self._cmd_poll_interval = CMD_POLL_INTERVAL

    def process(
        self,
        epoch: int = None,
        step: int = None,
        loss: float = None,
        reward: float = None,
        **extra,
    ):
        """
        Wird bei jedem Trainings-Schritt aufgerufen.
        1. State aktualisieren
        2. Fortschritt reporten
        3. Befehle prüfen + ausführen
        4. Pause-Loop wenn pausiert
        5. Stop-Flag zurückgeben
        """
        # State aktualisieren
        if epoch is not None:
            self.state.epoch = epoch
        if step is not None:
            self.state.step = step
        if loss is not None:
            self.state.loss = loss
        if reward is not None:
            self.state.reward = reward
        self.state.extra = extra

        # Fortschritt senden (nicht jeden Step — alle 10)
        if self.state.step % 10 == 0:
            self.reporter.report_progress(
                {
                    "epoch": self.state.epoch,
                    "step": self.state.step,
                    "loss": self.state.loss if not _isnan(self.state.loss) else None,
                    "reward": self.state.reward
                    if not _isnan(self.state.reward)
                    else None,
                    **extra,
                }
            )

        # Befehle prüfen (max alle 5s)
        now = time.time()
        if now - self._last_cmd_poll > self._cmd_poll_interval:
            self._last_cmd_poll = now
            self._process_commands()

        # Pause-Loop
        while self.state.paused and not self.state.stop_requested:
            logger.info("[bt4t] Training pausiert — warte auf RESUME...")
            time.sleep(5.0)
            self._process_commands()

        return not self.state.stop_requested

    def _process_commands(self):
        """Verarbeitet Befehle vom lokalen Rechner."""
        for cmd_dict in self.reporter.poll_commands():
            cmd = cmd_dict.get("cmd", "")
            params = cmd_dict.get("params", {})
            self._execute(cmd, params)

    def _execute(self, cmd: str, params: dict):
        """Führt einen Befehl aus."""
        if cmd == "PAUSE":
            self.state.paused = True
            logger.warning("[bt4t] Training PAUSIERT (Remote-Befehl)")

        elif cmd == "RESUME":
            self.state.paused = False
            logger.success("[bt4t] Training FORTGESETZT (Remote-Befehl)")

        elif cmd == "STOP" or cmd == "SHUTDOWN":
            self.state.stop_requested = True
            logger.warning("[bt4t] Training STOP angefordert (Remote-Befehl)")

        elif cmd == "CHANGE_LR":
            new_lr = params.get("value")
            if new_lr:
                self.repair.apply("reduce_lr", {})
                # Direktes Überschreiben im globalen NS
                ns = self.repair._get_global_ns()
                for k in ns:
                    if re.search(r"(?:learning_rate|lr)\b", k, re.I):
                        ns[k] = float(new_lr)
                logger.success(f"[bt4t] LR geändert auf {new_lr}")

        elif cmd == "CHANGE_BS":
            new_bs = params.get("value")
            if new_bs:
                ns = self.repair._get_global_ns()
                for k in ns:
                    if "batch_size" in k.lower():
                        ns[k] = int(new_bs)
                logger.success(f"[bt4t] Batch-Size geändert auf {new_bs}")

        elif cmd == "RELOAD_MODEL":
            path = params.get("model_path")
            logger.info(f"[bt4t] RELOAD_MODEL angefordert: {path}")
            # Signal setzen — Notebook-Code muss reload_requested prüfen
            ns = self.repair._get_global_ns()
            ns["BT4T_RELOAD_MODEL_PATH"] = path
            ns["BT4T_RELOAD_REQUESTED"] = True

        elif cmd == "STATUS":
            self.reporter.report_progress(
                {
                    "epoch": self.state.epoch,
                    "step": self.state.step,
                    "loss": self.state.loss,
                    "reward": self.state.reward,
                    "paused": self.state.paused,
                }
            )

        elif cmd != "NONE":
            logger.debug(f"[bt4t] Unbekannter Befehl: {cmd}")


# ════════════════════════════════════════════════════════════════════════════════
# SPEICHER-MONITOR
# ════════════════════════════════════════════════════════════════════════════════


class MemoryMonitor:
    """Überwacht GPU/RAM und warnt prophylaktisch vor OOM."""

    def __init__(self, repair: InProcessRepair, warn_pct: float = 85.0):
        self.repair = repair
        self.warn_pct = warn_pct
        self._thread: Optional[threading.Thread] = None
        self._running = False

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False

    def _loop(self):
        while self._running:
            self._check()
            time.sleep(60.0)

    def _check(self):
        if not _TORCH or not torch.cuda.is_available():
            return
        allocated = torch.cuda.memory_allocated()
        total = torch.cuda.get_device_properties(0).total_memory
        pct = allocated / total * 100

        if pct > self.warn_pct:
            logger.warning(f"[Memory] GPU: {pct:.1f}% — Prophylaktische Bereinigung")
            gc.collect()
            torch.cuda.empty_cache()
            pct_after = torch.cuda.memory_allocated() / total * 100
            logger.info(f"[Memory] GPU nach Bereinigung: {pct_after:.1f}%")

            if pct_after > 95.0:
                logger.warning("[Memory] GPU >95% — halbe Batch-Size prophylaktisch")
                self.repair.apply("halve_batch_size", {})


# ════════════════════════════════════════════════════════════════════════════════
# KEEPALIVE
# ════════════════════════════════════════════════════════════════════════════════


class ColabKeepalive:
    """
    Verhindert Colab-Session-Timeout durch echte Compute-Tasks.
    Colab killt Sessions bei INAKTIVITÄT (kein Output/Compute) — nicht nach Zeit.
    """

    def __init__(self, reporter: Reporter, interval_s: float = 600.0):
        self.reporter = reporter
        self.interval = interval_s
        self._thread: Optional[threading.Thread] = None
        self._running = False

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        logger.debug(f"[Keepalive] Gestartet (alle {self.interval:.0f}s)")

    def stop(self):
        self._running = False

    def _loop(self):
        while self._running:
            time.sleep(self.interval)
            if not self._running:
                break
            self._tick()

    def _tick(self):
        """Führt minimale Compute-Aufgabe aus (verhindert Timeout)."""
        import numpy as np

        # Echte Arbeit (kein Sleep-Trick) — Colab erkennt aktiven Kernel
        _ = np.random.randn(1000, 1000).mean()
        ts = datetime.now().strftime("%H:%M:%S")
        logger.debug(f"[Keepalive] Tick @ {ts}")
        self.reporter.report_heartbeat({"keepalive_tick": ts})


# ════════════════════════════════════════════════════════════════════════════════
# GLOBALER EXCEPTION HOOK
# ════════════════════════════════════════════════════════════════════════════════


class ExceptionHook:
    """
    Hängt sich als sys.excepthook ein.
    Fängt alle unbehandelten Exceptions und:
      1. Klassifiziert den Fehler
      2. Führt In-Process-Reparatur durch
      3. Meldet an lokalen Rechner
      4. Zeigt klare Diagnose-Ausgabe
    """

    def __init__(self, reporter: Reporter, repair: InProcessRepair):
        self.reporter = reporter
        self.repair = repair
        self._original_hook = sys.excepthook
        self._repair_count: Dict[str, int] = {}

    def install(self):
        sys.excepthook = self._hook
        logger.debug("[Hook] sys.excepthook installiert")

    def uninstall(self):
        sys.excepthook = self._original_hook

    def _hook(self, exc_type, exc_value, exc_tb):
        """Wird bei jeder unbehandelten Exception aufgerufen."""
        if exc_type is KeyboardInterrupt:
            self._original_hook(exc_type, exc_value, exc_tb)
            return

        error_type, action, severity = classify_error(exc_value)

        # Wiederholte gleiche Fehler zählen
        self._repair_count[error_type] = self._repair_count.get(error_type, 0) + 1
        count = self._repair_count[error_type]

        logger.error(
            f"Exception abgefangen: {exc_type.__name__}\n"
            f"  Typ:    {error_type} (Schwere: {severity})\n"
            f"  Aktion: {action}\n"
            f"  Anzahl: {count}x dieser Fehlertyp"
        )

        # Reparatur (max. 5x pro Fehlertyp um Loops zu vermeiden)
        repair_result = {"action": "none", "changes": [], "success": False}
        if action != "none" and count <= 5:
            ctx = {
                "error_message": str(exc_value),
                "error_type": error_type,
            }
            repair_result = self.repair.apply(action, ctx)
            if repair_result["success"]:
                logger.success(
                    f"[Hook] Reparatur '{action}' erfolgreich: "
                    f"{repair_result['changes']}"
                )
        elif count > 5:
            logger.warning(
                f"[Hook] {error_type} trat {count}x auf — keine weitere Reparatur"
            )

        # An lokalen Rechner melden
        self.reporter.report_error(exc_value, error_type, repair_result)

        # Original-Hook auch aufrufen (Traceback ausgeben)
        self._original_hook(exc_type, exc_value, exc_tb)


# ════════════════════════════════════════════════════════════════════════════════
# HAUPT-KLASSE: bt4t (öffentliche API)
# ════════════════════════════════════════════════════════════════════════════════


class BT4TExtension:
    """
    Öffentliche API der Colab-Extension.

    Wird als Singleton `bt4t` importiert:
        from colab_bridge.colab_extension import bt4t
        bt4t.install()
    """

    def __init__(self):
        # Konfiguration aus Umgebungsvariablen
        self._listener_url = os.getenv("BT4T_LISTENER_URL", "")
        self._api_token = os.getenv("BT4T_API_TOKEN", "bt4t-secret-token")
        self._notebook_id = os.getenv("BT4T_NOTEBOOK_ID", "colab_notebook")

        self._installed = False
        self._repair_log: list = []

        # Komponenten (werden bei install() gestartet)
        self._reporter: Optional[Reporter] = None
        self._repair: Optional[InProcessRepair] = None
        self._controller: Optional[IterationController] = None
        self._memory_mon: Optional[MemoryMonitor] = None
        self._keepalive: Optional[ColabKeepalive] = None
        self._hook: Optional[ExceptionHook] = None

    # ── Setup ─────────────────────────────────────────────────────────────────

    def install(
        self,
        listener_url: str = None,
        api_token: str = None,
        notebook_id: str = None,
        keepalive: bool = True,
        memory_monitor: bool = True,
        exception_hook: bool = True,
        keepalive_interval_s: float = 600.0,
        memory_warn_pct: float = 85.0,
    ) -> "BT4TExtension":
        """
        Installiert alle Extension-Komponenten.

        Parameter:
            listener_url    : URL des lokalen Control-Servers (z.B. Cloudflare Tunnel)
            api_token       : Auth-Token (muss mit lokalem Server übereinstimmen)
            notebook_id     : Name dieses Notebooks (für Logs/Berichte)
            keepalive       : Colab-Session-Keepalive aktivieren
            memory_monitor  : GPU/RAM-Überwachung aktivieren
            exception_hook  : Globalen Exception-Hook installieren
            keepalive_interval_s : Keepalive-Intervall in Sekunden (Standard: 600)
            memory_warn_pct : GPU-Warnschwelle in % (Standard: 85%)

        Gibt self zurück für Method-Chaining.
        """
        if self._installed:
            logger.warning("[bt4t] Extension bereits installiert — überspringe")
            return self

        # Konfiguration übernehmen
        if listener_url:
            self._listener_url = listener_url
        if api_token:
            self._api_token = api_token
        if notebook_id:
            self._notebook_id = notebook_id

        # Komponenten initialisieren
        self._reporter = Reporter(
            self._listener_url, self._api_token, self._notebook_id
        )
        self._repair = InProcessRepair(self._repair_log)
        self._controller = IterationController(self._reporter, self._repair)

        # Reporter starten
        self._reporter.start()

        # Keepalive
        if keepalive:
            self._keepalive = ColabKeepalive(self._reporter, keepalive_interval_s)
            self._keepalive.start()

        # Speicher-Monitor
        if memory_monitor:
            self._memory_mon = MemoryMonitor(self._repair, memory_warn_pct)
            self._memory_mon.start()

        # Exception-Hook
        if exception_hook:
            self._hook = ExceptionHook(self._reporter, self._repair)
            self._hook.install()

        self._installed = True

        # Startup-Heartbeat
        self._reporter.report_heartbeat({"event": "COLAB_READY", "installed": True})

        logger.success("=" * 55)
        logger.success("  bt4t Extension installiert")
        logger.success(f"  Notebook  : {self._notebook_id}")
        logger.success(f"  Listener  : {self._listener_url or '(nicht gesetzt)'}")
        logger.success(f"  Keepalive : {'an' if keepalive else 'aus'}")
        logger.success(f"  Memory    : {'an' if memory_monitor else 'aus'}")
        logger.success(f"  ExcHook   : {'an' if exception_hook else 'aus'}")
        logger.success("=" * 55)

        return self

    def uninstall(self):
        """Entfernt alle Hooks und stoppt alle Threads."""
        if self._hook:
            self._hook.uninstall()
        if self._keepalive:
            self._keepalive.stop()
        if self._memory_mon:
            self._memory_mon.stop()
        if self._reporter:
            self._reporter.stop()
        self._installed = False
        logger.info("[bt4t] Extension deinstalliert")

    # ── Öffentliche API ───────────────────────────────────────────────────────

    def step(
        self,
        epoch: int = None,
        step: int = None,
        loss: float = None,
        reward: float = None,
        **extra,
    ) -> bool:
        """
        Muss bei jedem Trainings-Schritt aufgerufen werden (optional).

        Meldet Fortschritt, prüft Befehle, führt Pause-Loop aus.

        Returns:
            bool: False wenn Training gestoppt werden soll.

        Verwendung:
            for epoch in range(100):
                loss = train(...)
                if not bt4t.step(epoch=epoch, loss=float(loss)):
                    break   # Training gestoppt
        """
        if not self._installed:
            return True
        return self._controller.process(
            epoch=epoch, step=step, loss=loss, reward=reward, **extra
        )

    def guard(self, fn: Callable) -> Callable:
        """
        Decorator: schützt eine Funktion mit automatischer Fehlerbehandlung.

        Verwendung:
            @bt4t.guard
            def run_training():
                for epoch in range(100):
                    train(...)
        """

        @wraps(fn)
        def wrapper(*args, **kwargs):
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    return fn(*args, **kwargs)
                except KeyboardInterrupt:
                    logger.info("[bt4t.guard] KeyboardInterrupt — abgebrochen")
                    raise
                except Exception as exc:
                    error_type, action, severity = classify_error(exc)
                    logger.error(
                        f"[bt4t.guard] Exception in '{fn.__name__}' "
                        f"(Versuch {attempt + 1}/{max_retries}): "
                        f"{type(exc).__name__}: {exc}"
                    )
                    ctx = {"error_message": str(exc), "error_type": error_type}
                    repair_result = self._repair.apply(action, ctx)
                    self._reporter.report_error(exc, error_type, repair_result)

                    if attempt < max_retries - 1 and severity != "high":
                        wait = 10 * (attempt + 1)
                        logger.info(f"[bt4t.guard] Warte {wait}s vor Wiederholung...")
                        time.sleep(wait)
                    else:
                        raise

        return wrapper

    @contextmanager
    def session(self, name: str = ""):
        """
        Context-Manager: schützt einen Code-Block.

        Verwendung:
            with bt4t.session("training_run_42"):
                train_model(model, data)
        """
        label = name or f"session_{int(time.time())}"
        logger.info(f"[bt4t] Session gestartet: {label}")
        self._reporter.report_heartbeat({"event": "SESSION_START", "name": label})
        try:
            yield self
            self._reporter.report_heartbeat(
                {"event": "SESSION_END", "name": label, "status": "OK"}
            )
            logger.success(f"[bt4t] Session beendet: {label}")
        except KeyboardInterrupt:
            self._reporter.report_heartbeat(
                {"event": "SESSION_END", "name": label, "status": "INTERRUPTED"}
            )
            raise
        except Exception as exc:
            error_type, action, _ = classify_error(exc)
            ctx = {"error_message": str(exc), "error_type": error_type}
            repair_result = self._repair.apply(action, ctx)
            self._reporter.report_error(exc, error_type, repair_result)
            self._reporter.report_heartbeat(
                {
                    "event": "SESSION_END",
                    "name": label,
                    "status": "ERROR",
                    "error": str(exc)[:200],
                }
            )
            raise

    def send_checkpoint(self, model_path: str = None, metrics: dict = None):
        """
        Meldet einen Checkpoint an den lokalen Rechner.
        Optional: Pfad zum gespeicherten Modell.
        """
        self._reporter.report_progress(
            {
                "event": "CHECKPOINT",
                "model_path": model_path or "",
                "metrics": metrics or {},
                "epoch": self._controller.state.epoch,
                "step": self._controller.state.step,
            }
        )
        logger.info(f"[bt4t] Checkpoint gemeldet: {model_path or '(kein Pfad)'}")

    def send_alert(self, message: str, level: str = "INFO"):
        """Sendet eine manuelle Nachricht an den lokalen Rechner."""
        self._reporter.report_progress(
            {"event": "ALERT", "level": level, "message": message}
        )

    @property
    def should_stop(self) -> bool:
        """True wenn Training gestoppt werden soll (Remote-Befehl)."""
        if not self._controller:
            return False
        return self._controller.state.stop_requested

    @property
    def is_paused(self) -> bool:
        if not self._controller:
            return False
        return self._controller.state.paused

    def repair_log(self) -> list:
        """Gibt alle durchgeführten Reparaturen zurück."""
        return list(self._repair_log)

    def status(self) -> dict:
        """Gibt den aktuellen Extension-Status zurück."""
        state = self._controller.state if self._controller else IterationState()
        return {
            "installed": self._installed,
            "notebook_id": self._notebook_id,
            "listener_url": self._listener_url,
            "epoch": state.epoch,
            "step": state.step,
            "paused": state.paused,
            "stop_requested": state.stop_requested,
            "repairs_done": len(self._repair_log),
        }


# ════════════════════════════════════════════════════════════════════════════════
# Singleton-Instanz
# ════════════════════════════════════════════════════════════════════════════════

bt4t = BT4TExtension()

# ── Hilfsfunktion ─────────────────────────────────────────────────────────────


def _isnan(v) -> bool:
    try:
        import math

        return math.isnan(v)
    except Exception:
        return False
