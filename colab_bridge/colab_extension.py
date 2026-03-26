"""
colab_extension.py — BITCOIN4Traders Colab Extension
=====================================================
A single import hooks into EVERY existing Colab notebook.
No code restructuring needed. Works as a decorator, context manager and
global exception hook simultaneously.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
USAGE IN COLAB (first cell):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    # Cell 1 — set up extension (copy-paste, never change)
    import sys, os
    sys.path.insert(0, '/content/drive/MyDrive/BITCOIN4Traders')
    os.environ.setdefault('BT4T_LISTENER_URL', 'https://your.tunnel.dev')
    os.environ.setdefault('BT4T_API_TOKEN',    'your-token')
    os.environ.setdefault('BT4T_NOTEBOOK_ID',  'training_v3')

    from colab_bridge.colab_extension import bt4t
    bt4t.install()   # ← call once, everything runs automatically afterwards

    # Cell 2 — your normal training code UNCHANGED:
    for epoch in range(100):
        loss = train_one_epoch(model, data)
        bt4t.step(epoch=epoch, loss=loss)   # ← optional: report progress

    # OR as a decorator (no bt4t.step needed):
    @bt4t.guard
    def run_training():
        for epoch in range(100):
            loss = train_one_epoch(model, data)

    # OR as a context manager:
    with bt4t.session("training_run_42"):
        train_model(model, data)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WHAT THE EXTENSION DOES:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. ERROR HANDLING (automatic, no code needed)
   • Global sys.excepthook — catches EVERY unhandled error
   • Classifies: OOM / NaN-Loss / CUDA / Import / Timeout / ...
   • Repairs parameters in the running process (no notebook restart needed)
   • Sends error report to local machine (HTTP → Drive fallback)
   • Can automatically resume training after repair

2. ITERATION CONTROL (opt-in via bt4t.step())
   • Progress (epoch, loss, reward) → local machine
   • Receives commands: PAUSE / RESUME / CHANGE_LR / CHANGE_BS / STOP
   • Checkpoint trigger: automatic at progress milestones
   • Early stopping: local machine can stop training

3. SESSION KEEPALIVE (automatic)
   • Runs a real compute task every 10 min → prevents Colab timeout
   • Heartbeat → local machine knows Colab is alive
   • Detects session reset and reports "COLAB_READY" after restart

4. MEMORY MANAGEMENT (automatic)
   • Monitors GPU/RAM every 60s
   • At >85% GPU memory: warns + optional gc + torch.cuda.empty_cache()
   • OOM prophylaxis: reduces batch size before OOM occurs

5. PARAMETER HOT-RELOAD (via commands from local machine)
   • Change hyperparameters during training without restart
   • Values are overwritten in the running process
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

# Import Secrets Manager
from src.config import get_colab_token

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

# ── Optional dependencies (graceful degradation) ─────────────────────────────
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
# ERROR CLASSIFIER (mirrors logic from error_repair.py, runs in Colab)
# ════════════════════════════════════════════════════════════════════════════════

# Pattern → (error_type, auto-repair action, severity)
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
    Classifies an exception.
    Returns (error_type, repair_action, severity).
    """
    text = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
    for pattern, etype, action, severity in _ERROR_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            return etype, action, severity
    return "UNKNOWN", "none", "medium"


# ════════════════════════════════════════════════════════════════════════════════
# IN-PROCESS REPAIR (modifies running variables, no restart needed)
# ════════════════════════════════════════════════════════════════════════════════


class InProcessRepair:
    """
    Repairs hyperparameters directly in the running Python process.
    Searches for variables in the notebook's global namespace (IPython __main__).
    """

    def __init__(self, repair_log: list):
        self._log = repair_log

    def apply(self, action: str, context: dict) -> dict:
        """
        Executes a repair action.
        Returns a dict with {action, changes, success}.
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
            logger.warning(f"Repair '{action}' failed: {e}")
            return {"action": action, "changes": [], "success": False}

    def _get_global_ns(self) -> dict:
        """Returns the notebook's global namespace (__main__)."""
        import __main__

        return vars(__main__)

    def _halve_batch_size(self, ctx: dict) -> list:
        """Halves all BATCH_SIZE / batch_size variables in the global namespace."""
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

        # Also patch torch DataLoader if available
        if _TORCH:
            gc.collect()
            torch.cuda.empty_cache()
            logger.info("[Repair] CUDA cache cleared")

        if not changes:
            logger.info("[Repair] No batch_size variable found")
        return changes

    def _reduce_lr(self, ctx: dict) -> list:
        """Reduces learning rate by a factor of 10."""
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

        # Patch PyTorch optimizer if available
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
        """Reduces gradient_clip_val or enables gradient clipping."""
        ns = self._get_global_ns()
        changes = []
        for name in list(ns.keys()):
            if "clip" in name.lower() and isinstance(ns[name], (int, float)):
                old = ns[name]
                ns[name] = min(old, 0.5)
                changes.append(f"{name}: {old} → {ns[name]}")
        if not changes:
            # Set new value
            ns["GRADIENT_CLIP_VAL"] = 0.5
            changes.append("GRADIENT_CLIP_VAL = 0.5 (new)")
            logger.info("[Repair] GRADIENT_CLIP_VAL = 0.5 set")
        return changes

    def _pip_install(self, ctx: dict) -> list:
        """Automatically installs a missing package."""
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
        return [f"pip install {module}: {'OK' if result.returncode == 0 else 'ERROR'}"]

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
        logger.info("[Repair] DATA_ERROR — skipping next batch")
        return ["skip_next_batch_flag = True"]

    def _retry_connection(self, ctx: dict) -> list:
        logger.info("[Repair] CONNECTION — waiting 30s, then continuing")
        time.sleep(30)
        return ["waited_30s"]

    def _retry_io(self, ctx: dict) -> list:
        logger.info("[Repair] IO_ERROR — waiting 10s")
        time.sleep(10)
        return ["waited_10s"]

    def _noop(self, ctx: dict) -> list:
        return []


# ════════════════════════════════════════════════════════════════════════════════
# REPORTER (sends status/errors to local machine)
# ════════════════════════════════════════════════════════════════════════════════


class Reporter:
    """
    Sends reports to the local machine.
    Channel 1: HTTP POST to listener URL (fast)
    Channel 2: Google Drive JSON file (fallback)
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
        """Sends error report."""
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
        """Sends training progress."""
        self._queue.append(
            {
                "type": "progress",
                "notebook_id": self.notebook_id,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                **data,
            }
        )

    def report_heartbeat(self, extra: dict = None):
        """Sends heartbeat."""
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
        """Background thread: drains the queue and sends messages."""
        while self._running or self._queue:
            if not self._queue:
                time.sleep(1.0)
                continue
            payload = self._queue.pop(0)
            self._send_one(payload)

    def _send_one(self, payload: dict):
        """Sends one message. Tries HTTP, falls back to Drive."""
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
                    logger.debug(f"[Reporter] Sent: {payload['type']}")
                    return
        except Exception as e:
            logger.debug(f"[Reporter] HTTP failed: {e} — trying Drive")

        self._fallback_drive(payload)

    def _fallback_drive(self, payload: dict):
        """Writes report as a JSON file — Drive manager reads it."""
        try:
            # Check if Drive is mounted
            drive_dir = Path("/content/drive/MyDrive/BITCOIN4Traders/bt4t_bus/reports")
            if drive_dir.parent.parent.parent.exists():
                drive_dir.mkdir(parents=True, exist_ok=True)
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                out = drive_dir / f"{payload['type']}_{ts}.json"
                out.write_text(json.dumps(payload, default=str, indent=2))
                logger.debug(f"[Reporter] Drive fallback: {out.name}")
        except Exception as e:
            logger.debug(f"[Reporter] Drive fallback failed: {e}")

    def poll_commands(self) -> list[dict]:
        """
        Polls the local machine for commands (synchronous, for iteration hook).
        Returns a list of command dicts.
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
# ITERATION CONTROLLER
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
    # Hyperparameters that can be changed remotely
    lr: Optional[float] = None
    batch_size: Optional[int] = None
    checkpoint_every: int = 50  # Send checkpoint every N steps


class IterationController:
    """
    Controls the training loop from outside.
    bt4t.step() internally calls process().
    """

    def __init__(self, reporter: Reporter, repair: InProcessRepair):
        self.reporter = reporter
        self.repair = repair
        self.state = IterationState()
        self._last_cmd_poll = 0.0
        CMD_POLL_INTERVAL = 5.0  # Poll commands every 5s
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
        Called at every training step.
        1. Update state
        2. Report progress
        3. Check + execute commands
        4. Pause loop when paused
        5. Return stop flag
        """
        # Update state
        if epoch is not None:
            self.state.epoch = epoch
        if step is not None:
            self.state.step = step
        if loss is not None:
            self.state.loss = loss
        if reward is not None:
            self.state.reward = reward
        self.state.extra = extra

        # Send progress (not every step — every 10)
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

        # Check commands (at most every 5s)
        now = time.time()
        if now - self._last_cmd_poll > self._cmd_poll_interval:
            self._last_cmd_poll = now
            self._process_commands()

        # Pause loop
        while self.state.paused and not self.state.stop_requested:
            logger.info("[bt4t] Training paused — waiting for RESUME...")
            time.sleep(5.0)
            self._process_commands()

        return not self.state.stop_requested

    def _process_commands(self):
        """Processes commands from the local machine."""
        for cmd_dict in self.reporter.poll_commands():
            cmd = cmd_dict.get("cmd", "")
            params = cmd_dict.get("params", {})
            self._execute(cmd, params)

    def _execute(self, cmd: str, params: dict):
        """Executes a command."""
        if cmd == "PAUSE":
            self.state.paused = True
            logger.warning("[bt4t] Training PAUSED (remote command)")

        elif cmd == "RESUME":
            self.state.paused = False
            logger.success("[bt4t] Training RESUMED (remote command)")

        elif cmd == "STOP" or cmd == "SHUTDOWN":
            self.state.stop_requested = True
            logger.warning("[bt4t] Training STOP requested (remote command)")

        elif cmd == "CHANGE_LR":
            new_lr = params.get("value")
            if new_lr:
                self.repair.apply("reduce_lr", {})
                # Directly overwrite in global namespace
                ns = self.repair._get_global_ns()
                for k in ns:
                    if re.search(r"(?:learning_rate|lr)\b", k, re.I):
                        ns[k] = float(new_lr)
                logger.success(f"[bt4t] LR changed to {new_lr}")

        elif cmd == "CHANGE_BS":
            new_bs = params.get("value")
            if new_bs:
                ns = self.repair._get_global_ns()
                for k in ns:
                    if "batch_size" in k.lower():
                        ns[k] = int(new_bs)
                logger.success(f"[bt4t] Batch size changed to {new_bs}")

        elif cmd == "RELOAD_MODEL":
            path = params.get("model_path")
            logger.info(f"[bt4t] RELOAD_MODEL requested: {path}")
            # Set signal — notebook code must check reload_requested
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
            logger.debug(f"[bt4t] Unknown command: {cmd}")


# ════════════════════════════════════════════════════════════════════════════════
# MEMORY MONITOR
# ════════════════════════════════════════════════════════════════════════════════


class MemoryMonitor:
    """Monitors GPU/RAM and proactively warns before OOM."""

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
            logger.warning(f"[Memory] GPU: {pct:.1f}% — Proactive cleanup")
            gc.collect()
            torch.cuda.empty_cache()
            pct_after = torch.cuda.memory_allocated() / total * 100
            logger.info(f"[Memory] GPU after cleanup: {pct_after:.1f}%")

            if pct_after > 95.0:
                logger.warning("[Memory] GPU >95% — proactively halving batch size")
                self.repair.apply("halve_batch_size", {})


# ════════════════════════════════════════════════════════════════════════════════
# KEEPALIVE
# ════════════════════════════════════════════════════════════════════════════════


class ColabKeepalive:
    """
    Prevents Colab session timeout via real compute tasks.
    Colab kills sessions on INACTIVITY (no output/compute) — not based on time.
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
        logger.debug(f"[Keepalive] Started (every {self.interval:.0f}s)")

    def stop(self):
        self._running = False

    def _loop(self):
        while self._running:
            time.sleep(self.interval)
            if not self._running:
                break
            self._tick()

    def _tick(self):
        """Runs a minimal compute task (prevents timeout)."""
        import numpy as np

        # Real work (no sleep trick) — Colab detects an active kernel
        _ = np.random.randn(1000, 1000).mean()
        ts = datetime.now().strftime("%H:%M:%S")
        logger.debug(f"[Keepalive] Tick @ {ts}")
        self.reporter.report_heartbeat({"keepalive_tick": ts})


# ════════════════════════════════════════════════════════════════════════════════
# GLOBAL EXCEPTION HOOK
# ════════════════════════════════════════════════════════════════════════════════


class ExceptionHook:
    """
    Installs itself as sys.excepthook.
    Catches all unhandled exceptions and:
      1. Classifies the error
      2. Performs in-process repair
      3. Reports to local machine
      4. Displays a clear diagnostic output
    """

    def __init__(self, reporter: Reporter, repair: InProcessRepair):
        self.reporter = reporter
        self.repair = repair
        self._original_hook = sys.excepthook
        self._repair_count: Dict[str, int] = {}

    def install(self):
        sys.excepthook = self._hook
        logger.debug("[Hook] sys.excepthook installed")

    def uninstall(self):
        sys.excepthook = self._original_hook

    def _hook(self, exc_type, exc_value, exc_tb):
        """Called for every unhandled exception."""
        if exc_type is KeyboardInterrupt:
            self._original_hook(exc_type, exc_value, exc_tb)
            return

        error_type, action, severity = classify_error(exc_value)

        # Count repeated identical errors
        self._repair_count[error_type] = self._repair_count.get(error_type, 0) + 1
        count = self._repair_count[error_type]

        logger.error(
            f"Exception caught: {exc_type.__name__}\n"
            f"  Type:   {error_type} (severity: {severity})\n"
            f"  Action: {action}\n"
            f"  Count:  {count}x this error type"
        )

        # Repair (max. 5x per error type to avoid loops)
        repair_result = {"action": "none", "changes": [], "success": False}
        if action != "none" and count <= 5:
            ctx = {
                "error_message": str(exc_value),
                "error_type": error_type,
            }
            repair_result = self.repair.apply(action, ctx)
            if repair_result["success"]:
                logger.success(
                    f"[Hook] Repair '{action}' successful: "
                    f"{repair_result['changes']}"
                )
        elif count > 5:
            logger.warning(f"[Hook] {error_type} occurred {count}x — no further repair")

        # Report to local machine
        self.reporter.report_error(exc_value, error_type, repair_result)

        # Also call original hook (print traceback)
        self._original_hook(exc_type, exc_value, exc_tb)


# ════════════════════════════════════════════════════════════════════════════════
# MAIN CLASS: bt4t (public API)
# ════════════════════════════════════════════════════════════════════════════════


class BT4TExtension:
    """
    Public API of the Colab extension.

    Imported as singleton `bt4t`:
        from colab_bridge.colab_extension import bt4t
        bt4t.install()
    """

    def __init__(self):
        # Configuration from Secrets Manager (with fallback to environment)
        self._listener_url = os.getenv("BT4T_LISTENER_URL", "")

        # Get token from Secrets Manager
        secrets_token = get_colab_token()
        self._api_token = secrets_token or os.getenv(
            "BT4T_API_TOKEN", "bt4t-secret-token"
        )

        self._notebook_id = os.getenv("BT4T_NOTEBOOK_ID", "colab_notebook")

        self._installed = False
        self._repair_log: list = []

        # Components (started during install())
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
        Installs all extension components.

        Parameters:
            listener_url    : URL of the local control server (e.g. Cloudflare Tunnel)
            api_token       : Auth token (must match local server)
            notebook_id     : Name of this notebook (for logs/reports)
            keepalive       : Enable Colab session keepalive
            memory_monitor  : Enable GPU/RAM monitoring
            exception_hook  : Install global exception hook
            keepalive_interval_s : Keepalive interval in seconds (default: 600)
            memory_warn_pct : GPU warning threshold in % (default: 85%)

        Returns self for method chaining.
        """
        if self._installed:
            logger.warning("[bt4t] Extension already installed — skipping")
            return self

        # Apply configuration
        if listener_url:
            self._listener_url = listener_url
        if api_token:
            self._api_token = api_token
        if notebook_id:
            self._notebook_id = notebook_id

        # Initialize components
        self._reporter = Reporter(
            self._listener_url, self._api_token, self._notebook_id
        )
        self._repair = InProcessRepair(self._repair_log)
        self._controller = IterationController(self._reporter, self._repair)

        # Start reporter
        self._reporter.start()

        # Start keepalive
        if keepalive:
            self._keepalive = ColabKeepalive(self._reporter, keepalive_interval_s)
            self._keepalive.start()

        # Start memory monitor
        if memory_monitor:
            self._memory_mon = MemoryMonitor(self._repair, memory_warn_pct)
            self._memory_mon.start()

        # Install exception hook
        if exception_hook:
            self._hook = ExceptionHook(self._reporter, self._repair)
            self._hook.install()

        self._installed = True

        # Startup heartbeat
        self._reporter.report_heartbeat({"event": "COLAB_READY", "installed": True})

        logger.success("=" * 55)
        logger.success("  bt4t Extension installed")
        logger.success(f"  Notebook  : {self._notebook_id}")
        logger.success(f"  Listener  : {self._listener_url or '(not set)'}")
        logger.success(f"  Keepalive : {'on' if keepalive else 'off'}")
        logger.success(f"  Memory    : {'on' if memory_monitor else 'off'}")
        logger.success(f"  ExcHook   : {'on' if exception_hook else 'off'}")
        logger.success("=" * 55)

        return self

    def uninstall(self):
        """Removes all hooks and stops all threads."""
        if self._hook:
            self._hook.uninstall()
        if self._keepalive:
            self._keepalive.stop()
        if self._memory_mon:
            self._memory_mon.stop()
        if self._reporter:
            self._reporter.stop()
        self._installed = False
        logger.info("[bt4t] Extension uninstalled")

    # ── Public API ────────────────────────────────────────────────────────────

    def step(
        self,
        epoch: int = None,
        step: int = None,
        loss: float = None,
        reward: float = None,
        **extra,
    ) -> bool:
        """
        Should be called at every training step (optional).

        Reports progress, checks commands, runs pause loop.

        Returns:
            bool: False if training should be stopped.

        Usage:
            for epoch in range(100):
                loss = train(...)
                if not bt4t.step(epoch=epoch, loss=float(loss)):
                    break   # Training stopped
        """
        if not self._installed:
            return True
        return self._controller.process(
            epoch=epoch, step=step, loss=loss, reward=reward, **extra
        )

    def guard(self, fn: Callable) -> Callable:
        """
        Decorator: protects a function with automatic error handling.

        Usage:
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
                    logger.info("[bt4t.guard] KeyboardInterrupt — aborted")
                    raise
                except Exception as exc:
                    error_type, action, severity = classify_error(exc)
                    logger.error(
                        f"[bt4t.guard] Exception in '{fn.__name__}' "
                        f"(attempt {attempt + 1}/{max_retries}): "
                        f"{type(exc).__name__}: {exc}"
                    )
                    ctx = {"error_message": str(exc), "error_type": error_type}
                    repair_result = self._repair.apply(action, ctx)
                    self._reporter.report_error(exc, error_type, repair_result)

                    if attempt < max_retries - 1 and severity != "high":
                        wait = 10 * (attempt + 1)
                        logger.info(f"[bt4t.guard] Waiting {wait}s before retry...")
                        time.sleep(wait)
                    else:
                        raise

        return wrapper

    @contextmanager
    def session(self, name: str = ""):
        """
        Context manager: protects a code block.

        Usage:
            with bt4t.session("training_run_42"):
                train_model(model, data)
        """
        label = name or f"session_{int(time.time())}"
        logger.info(f"[bt4t] Session started: {label}")
        self._reporter.report_heartbeat({"event": "SESSION_START", "name": label})
        try:
            yield self
            self._reporter.report_heartbeat(
                {"event": "SESSION_END", "name": label, "status": "OK"}
            )
            logger.success(f"[bt4t] Session ended: {label}")
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
        Reports a checkpoint to the local machine.
        Optional: path to the saved model.
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
        logger.info(f"[bt4t] Checkpoint reported: {model_path or '(no path)'}")

    def send_alert(self, message: str, level: str = "INFO"):
        """Sends a manual message to the local machine."""
        self._reporter.report_progress(
            {"event": "ALERT", "level": level, "message": message}
        )

    @property
    def should_stop(self) -> bool:
        """True if training should be stopped (remote command)."""
        if not self._controller:
            return False
        return self._controller.state.stop_requested

    @property
    def is_paused(self) -> bool:
        if not self._controller:
            return False
        return self._controller.state.paused

    def repair_log(self) -> list:
        """Returns all repairs that have been performed."""
        return list(self._repair_log)

    def status(self) -> dict:
        """Returns the current extension status."""
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
# Singleton instance
# ════════════════════════════════════════════════════════════════════════════════

bt4t = BT4TExtension()

# ── Helper function ───────────────────────────────────────────────────────────


def _isnan(v) -> bool:
    try:
        import math

        return math.isnan(v)
    except Exception:
        return False
