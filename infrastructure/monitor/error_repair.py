#!/usr/bin/env python3
"""
Auto-Repair Engine - Self-healing for the Colab notebook
==========================================================
Receives error reports and patches the notebook automatically.

Error mapping (known errors -> automatic solution):
  OutOfMemoryError      -> halve batch_size, shorten sequence length
  CUDA out of memory    -> halve batch_size + clear cache
  ConnectionError       -> send restart signal
  TimeoutError          -> increase timeout parameter
  ModuleNotFoundError   -> write pip install command into notebook
  NaN/Inf in loss       -> reduce learning_rate by 10x
  KeyboardInterrupt     -> ignore (normal behavior)
  SessionCrashed        -> trigger complete restart

Strategy:
  1. Classify error
  2. Select appropriate repair recipe
  3. Patch notebook JSON (change parameters)
  4. Upload patched notebook to Drive
  5. Send restart signal
  6. Keep repair log
"""

import json
import re
import sys
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE_DIR / "infrastructure" / "drive"))

REPAIR_LOG = BASE_DIR / "logs" / "repairs.jsonl"
NOTEBOOK_LOCAL = BASE_DIR / "BITCOIN4Traders_Colab.ipynb"
REPAIR_LOG.parent.mkdir(parents=True, exist_ok=True)

# ─── Error classification ────────────────────────────────────────────────────

ERROR_PATTERNS = [
    # (regex_pattern, error_type, severity)
    (r"(CUDA out of memory|OutOfMemoryError|OOM|out of memory)", "OOM", "high"),
    (r"(ConnectionError|ConnectTimeout|RemoteDisconnected)", "CONNECTION", "medium"),
    (r"(ModuleNotFoundError|ImportError|No module named)", "IMPORT", "medium"),
    (r"(nan|NaN|inf|Inf).*(loss|reward|gradient)", "NAN_LOSS", "high"),
    (r"(TimeoutError|ReadTimeout|socket.timeout)", "TIMEOUT", "low"),
    (r"(RuntimeError.*CUDA|device-side assert)", "CUDA_ERROR", "high"),
    (r"(KeyError|IndexError|ValueError).*(batch|data|feature)", "DATA_ERROR", "medium"),
    (r"(PermissionError|FileNotFoundError).*(drive|model|cache)", "IO_ERROR", "medium"),
    (r"(gradient.*explod|loss.*explod|overflow)", "EXPLODING", "high"),
]


def classify_error(error_message: str, stacktrace: str = "") -> str:
    """Classifies errors by patterns."""
    combined = (error_message + " " + stacktrace).lower()
    for pattern, error_type, _ in ERROR_PATTERNS:
        if re.search(pattern, combined, re.IGNORECASE):
            return error_type
    return "UNKNOWN"


# ─── Repair recipes ────────────────────────────────────────────────────────────


def _patch_notebook_parameter(
    nb: dict, old_value: str, new_value: str, description: str
) -> tuple[dict, int]:
    """
    Replaces a parameter value in all code cells of the notebook.
    Returns (modified notebook, number of changes).
    """
    changes = 0
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        source = cell.get("source", [])
        new_source = []
        for line in source:
            if old_value in line and not line.strip().startswith("#"):
                new_line = line.replace(old_value, new_value)
                new_source.append(new_line)
                changes += 1
                print(
                    f"  [PATCH] {description}: '{line.strip()}' → '{new_line.strip()}'"
                )
            else:
                new_source.append(line)
        cell["source"] = new_source
    return nb, changes


def repair_oom(nb: dict, report: dict) -> dict:
    """
    OOM (Out of Memory): halve batch_size.
    Strategy: finds all BATCH_SIZE = X and halves X.
    """
    print("[REPAIR] OOM detected - halving batch_size")

    # Find all batch_size values and halve them
    changes_total = 0
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        source = cell.get("source", [])
        new_source = []
        for line in source:
            # Match: BATCH_SIZE = 64, batch_size=128, etc.
            match = re.match(r"^(\s*(?:BATCH_SIZE|batch_size)\s*=\s*)(\d+)(.*)", line)
            if match:
                old_val = int(match.group(2))
                new_val = max(8, old_val // 2)  # Minimum: 8
                new_line = f"{match.group(1)}{new_val}{match.group(3)}"
                new_source.append(new_line)
                changes_total += 1
                print(f"  [PATCH] batch_size: {old_val} → {new_val}")
            else:
                new_source.append(line)
        cell["source"] = new_source

    if changes_total == 0:
        print("  [PATCH] No batch_size parameter found - restarting without patch")

    return nb


def repair_nan_loss(nb: dict, report: dict) -> dict:
    """NaN in loss: reduce learning rate by 10x."""
    print("[REPAIR] NaN loss detected - reducing learning_rate")

    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        source = cell.get("source", [])
        new_source = []
        for line in source:
            match = re.match(
                r"^(\s*(?:LEARNING_RATE|learning_rate|lr)\s*=\s*)([\d.e-]+)(.*)", line
            )
            if match:
                try:
                    old_val = float(match.group(2))
                    new_val = old_val / 10
                    new_line = f"{match.group(1)}{new_val:.2e}{match.group(3)}"
                    new_source.append(new_line)
                    print(f"  [PATCH] learning_rate: {old_val:.2e} → {new_val:.2e}")
                except ValueError:
                    new_source.append(line)
            else:
                new_source.append(line)
        cell["source"] = new_source

    return nb


def repair_exploding_gradient(nb: dict, report: dict) -> dict:
    """Exploding gradient: increase gradient clipping or reduce LR."""
    print("[REPAIR] Exploding gradient - activating/increasing gradient clipping")

    # Tighten gradient_clip_val
    nb, changes = _patch_notebook_parameter(
        nb,
        "gradient_clip_val=0.5",
        "gradient_clip_val=0.1",
        "tighten gradient_clip_val",
    )
    if changes == 0:
        # LR as fallback
        nb = repair_nan_loss(nb, report)

    return nb


def repair_import_error(nb: dict, report: dict) -> dict:
    """Missing library: insert pip install command into first code cell."""
    error_msg = report.get("error_message", "")

    # Extract module name
    match = re.search(r"No module named '([^']+)'", error_msg)
    if not match:
        return nb

    module = match.group(1).split(".")[0]  # Only main package
    print(f"[REPAIR] Missing module: {module} - adding pip install")

    # Insert into first code cell
    for cell in nb.get("cells", []):
        if cell.get("cell_type") == "code":
            source = cell.get("source", [])
            install_line = f"!pip install -q {module}  # Auto-Repair\n"
            if install_line not in source:
                cell["source"] = [install_line] + source
            break

    return nb


def repair_timeout(nb: dict, report: dict) -> dict:
    """Timeout: increase timeout parameter."""
    print("[REPAIR] Timeout - increasing REQUEST_TIMEOUT")
    nb, _ = _patch_notebook_parameter(
        nb, "REQUEST_TIMEOUT = 30", "REQUEST_TIMEOUT = 60", "increase request timeout"
    )
    return nb


# Dispatch table: error_type -> repair function
REPAIR_FUNCTIONS = {
    "OOM": repair_oom,
    "CUDA_ERROR": repair_oom,  # Same treatment
    "NAN_LOSS": repair_nan_loss,
    "EXPLODING": repair_exploding_gradient,
    "IMPORT": repair_import_error,
    "TIMEOUT": repair_timeout,
    # CONNECTION and IO_ERROR: restart only, no notebook patch needed
}


# ─── Main function ────────────────────────────────────────────────────────────


def repair(report: dict) -> bool:
    """
    Main function: receives error report, patches notebook, uploads.
    Returns True if repair was successful.
    """
    error_msg = report.get("error_message", "")
    stacktrace = report.get("stacktrace", "")
    notebook_id = report.get("notebook_id", "unknown")

    error_type = classify_error(error_msg, stacktrace)
    print(f"\n[REPAIR] Error type: {error_type} | Notebook: {notebook_id}")

    # Repair log record
    repair_record = {
        "timestamp": datetime.now().isoformat(),
        "notebook_id": notebook_id,
        "error_type": error_type,
        "error_message": error_msg[:200],
        "patch_applied": False,
        "restart_sent": False,
    }

    # Load notebook
    if not NOTEBOOK_LOCAL.exists():
        print(f"[REPAIR] Notebook not found: {NOTEBOOK_LOCAL}")
        _log_repair(repair_record)
        return False

    with open(NOTEBOOK_LOCAL) as f:
        nb = json.load(f)

    # Backup original
    backup = NOTEBOOK_LOCAL.with_suffix(".ipynb.backup")
    shutil.copy(NOTEBOOK_LOCAL, backup)

    # Select repair function
    repair_fn = REPAIR_FUNCTIONS.get(error_type)
    patched = False

    if repair_fn:
        try:
            nb = repair_fn(nb, report)
            # Save patched notebook locally
            with open(NOTEBOOK_LOCAL, "w") as f:
                json.dump(nb, f, indent=1, ensure_ascii=False)
            patched = True
            repair_record["patch_applied"] = True
            print(f"[REPAIR] Notebook patched: {NOTEBOOK_LOCAL}")
        except Exception as e:
            print(f"[REPAIR] Patch error: {e}")
            # Restore original
            shutil.copy(backup, NOTEBOOK_LOCAL)
    else:
        print(f"[REPAIR] No patch recipe for {error_type} - restart only")

    # Upload patched notebook to Drive
    if patched:
        _upload_patched_notebook()

    # Send restart signal (always)
    _send_restart_via_drive(error_type, repair_record)
    repair_record["restart_sent"] = True

    _log_repair(repair_record)

    result_msg = (
        "Notebook patched + restart signal sent" if patched else "Restart signal sent"
    )
    print(f"[REPAIR] Completed: {result_msg}")
    return True


def _upload_patched_notebook():
    """Uploads the patched notebook to Drive."""
    try:
        from drive_manager import get_drive_service, load_drive_config, upload_file

        service = get_drive_service()
        config = load_drive_config()
        folder_id = config.get("champion_folder_id", "")
        if folder_id:
            upload_file(service, NOTEBOOK_LOCAL, folder_id, "Auto-Repair Patch")
            print("[REPAIR] Patched notebook uploaded to Drive")
    except Exception as e:
        print(f"[REPAIR] Drive upload failed: {e}")


def _send_restart_via_drive(error_type: str, repair_record: dict):
    """Sends restart signal via Drive."""
    try:
        import tempfile
        from drive_manager import get_drive_service, load_drive_config, upload_file

        service = get_drive_service()
        config = load_drive_config()
        folder_id = config.get("champion_folder_id", "")
        if not folder_id:
            return

        signal = {
            "action": "restart_training",
            "reason": error_type,
            "repair_applied": repair_record.get("patch_applied", False),
            "requested_at": datetime.now().isoformat(),
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(signal, f)
            tmp_path = Path(f.name)

        upload_file(service, tmp_path, folder_id, "Restart signal")
        tmp_path.unlink()
    except Exception as e:
        print(f"[REPAIR] Restart signal failed: {e}")


def _log_repair(record: dict):
    """Writes repair log record."""
    with open(REPAIR_LOG, "a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    # Test with simulated OOM error
    test_report = {
        "notebook_id": "test",
        "error_message": "CUDA out of memory. Tried to allocate 2.00 GiB",
        "stacktrace": "RuntimeError: CUDA out of memory",
    }
    repair(test_report)
