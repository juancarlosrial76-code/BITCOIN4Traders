# BITCOIN4Traders — Infrastructure Documentation

Complete reference for the zero-cost distributed infrastructure that keeps the system running 24/7.

---

## Table of Contents

1. [Infrastructure Overview](#1-infrastructure-overview)
2. [master.py — Master Orchestrator](#2-masterpy--master-orchestrator)
3. [drive_manager.py — Google Drive Sync](#3-drive_managerpy--google-drive-sync)
4. [colab_watchdog.py — Colab Keepalive & Auto-Recovery](#4-colab_watchdogpy--colab-keepalive--auto-recovery)
5. [listener.py — Error Receiver](#5-listenerpy--error-receiver)
6. [error_repair.py — Auto-Repair Engine](#6-error_repairpy--auto-repair-engine)
7. [alert_manager.py — Notification System](#7-alert_managerpy--notification-system)
8. [Systemd Service](#8-systemd-service)
9. [install.sh — Setup Guide](#9-installsh--setup-guide)
10. [End-to-End Self-Healing Flow](#10-end-to-end-self-healing-flow)

---

## 1. Infrastructure Overview

The infrastructure directory contains the complete "operating system" for the BITCOIN4Traders project. It coordinates the four pillars (Linux PC, Google Drive, Colab GPU, GitHub) without any paid services.

```
infrastructure/
├── master.py                    # Central orchestrator — start here
├── install.sh                   # One-time setup script
├── drive/
│   └── drive_manager.py         # Google Drive sync (bidirectional)
├── colab/
│   └── colab_watchdog.py        # Colab session monitoring & keepalive
├── monitor/
│   ├── listener.py              # Flask HTTP receiver (dual-channel)
│   ├── error_repair.py          # Automated notebook patching engine
│   └── alert_manager.py        # Telegram / log / desktop notifications
└── systemd/
    └── bitcoin4traders.service  # systemd unit for persistent operation
```

**Entry point for production:** `python3 infrastructure/master.py`  
**Preferred entry point for 24/7 operation:** `sudo systemctl start bitcoin4traders`

---

## 2. master.py — Master Orchestrator

**File:** `infrastructure/master.py`  
**Role:** The central conductor. Coordinates all four infrastructure pillars via a lightweight internal scheduler.

### Scheduler

`master.py` implements its own dependency-free `Scheduler` class rather than using APScheduler or cron. This keeps the installation minimal and the failure surface small.

```python
class Scheduler:
    def every(self, seconds, name, func, *args, **kwargs)
    def run_pending()     # called every 30 seconds in main loop
```

Tasks are registered with a name, an interval in seconds, and a callable. On the first run, every task executes immediately (next_run = now). After each run, next_run = now + interval.

**Error handling:** If a task raises an exception, the exception is caught, printed, and an alert is sent via `alert_manager.warn()`. The scheduler continues — a single failing task never kills the process.

### Scheduled Tasks

| Task name | Interval | Function | Description |
|---|---|---|---|
| `drive_sync` | Every 60 min | `task_drive_sync()` | Upload champion model + logs to Google Drive |
| `colab_watchdog` | Every 60 min | `task_colab_watchdog()` | Check if Colab is still alive, send restart signal if not |
| `training_check` | Every 15 min | `task_local_training_check()` | Verify local training process (`auto_12h_train.py`) is running |
| `github_backup` | Every 6 hours | `task_github_backup()` | Push champion metadata + code to GitHub via `sync_champion.sh` |
| `status_report` | Every 24 hours | `task_status_report()` | Generate daily status JSON + send summary to Telegram |

### Task Implementation Details

#### `task_drive_sync()`

Calls `DriveManager.run_sync("up")` to push the following files to Google Drive:
- Champion model weights (`data/cache/*.pth`)
- Champion metadata (`data/cache/multiverse_champion_meta.json`)
- Latest training logs (`logs/`)
- Heartbeat file (timestamp proving the Linux PC is alive)

On failure: sends a WARNING alert. Does not halt other tasks.

#### `task_colab_watchdog()`

Calls `run_watchdog(alert_callback=...)`. If Colab status is stale (> 90 minutes), sends a CRITICAL alert and writes a restart signal to Drive.

#### `task_local_training_check()`

Uses `subprocess.run(["pgrep", "-f", "auto_12h_train.py"])`. If the process is not found (returncode != 0), sends a WARNING alert. Optional auto-restart is commented out by default — the operator can uncomment if desired.

#### `task_github_backup()`

Calls `sync_champion.sh` with a 120-second timeout. This shell script does a `git add + commit + push` for champion model metadata. The timeout prevents a network hang from blocking the scheduler loop.

#### `task_status_report()`

Writes `logs/infrastructure_status.json` with:
- Current timestamp
- System uptime (read from `/proc/uptime`)
- Champion metadata (name, Sharpe ratio)
- Free disk space (via `shutil.disk_usage`)

Then sends the summary as an INFO alert (Telegram + log file).

### Background Services

The listener (Flask HTTP server) is started in a background daemon thread by `_start_listener()` before the main scheduler loop begins. The listener runs independently and does not block the scheduler.

### Graceful Shutdown

Signal handlers for `SIGTERM` and `SIGINT` set a `threading.Event` (`_shutdown`). The main loop checks this event every 30 seconds and exits cleanly, allowing the current task to complete before shutdown. The systemd unit therefore receives a clean exit.

### Main Loop

```python
while not _shutdown.is_set():
    scheduler.run_pending()
    time.sleep(30)   # Poll interval: 30 seconds
```

---

## 3. drive_manager.py — Google Drive Sync

**File:** `infrastructure/drive/drive_manager.py`  
**Role:** Bidirectional synchronisation between the Linux PC and Google Drive using the official `google-api-python-client`.

**Cost:** Free. The Google Drive API allows up to 1 billion requests per day with no charge.

### Authentication

Uses a **Service Account** (not OAuth). This is the correct choice for server-side automation:
- No browser interaction required
- Token never expires
- No user approval needed

```
config/gdrive_credentials.json   ← Service Account key from Google Cloud Console
```

```python
def get_drive_service():
    creds = service_account.Credentials.from_service_account_file(
        CREDENTIALS_FILE, scopes=["https://www.googleapis.com/auth/drive"]
    )
    return build("drive", "v3", credentials=creds, cache_discovery=False)
```

### Configuration

Drive folder IDs are stored in `config/drive_config.json`:

```json
{
  "champion_folder_id": "1AbC...",   // Google Drive folder for models
  "logs_folder_id": "2XyZ...",       // Google Drive folder for logs
  "notebook_file_id": "3PqR..."      // File ID of the Colab notebook
}
```

Folder IDs are extracted from the Drive folder URL:  
`https://drive.google.com/drive/folders/THIS_ID_HERE`

### File Operations

#### `upload_file(service, local_path, folder_id, description)`

Uploads a file and **updates it in place** if it already exists. This prevents Drive clutter — there is never more than one copy of any file.

Internally:
1. Calls `find_file_in_folder()` to check for existing file
2. If exists: calls `files().update()` (modifies existing file ID)
3. If not exists: calls `files().create()` with parent folder metadata
4. Uses `resumable=True` for large model files (handles network interruptions)

#### `download_file(service, file_id, local_path)`

Downloads a file using `MediaIoBaseDownload` with chunked streaming. Handles multi-chunk downloads for large files.

#### `find_file_in_folder(service, folder_id, filename)`

Queries Drive for a file by name within a folder. Returns the file ID or None.  
Query format: `name='{name}' and '{folder_id}' in parents and trashed=false`

### Sync Operations

#### `run_sync(direction)`

- `direction="up"`: Linux PC → Drive
  - Uploads champion model
  - Uploads latest logs
  - Writes heartbeat file
- `direction="down"`: Drive → Linux PC
  - Downloads latest champion model (from Colab training)
  - Downloads updated notebook

#### `write_heartbeat(service, config)`

Creates `/tmp/linux_heartbeat.json` with current timestamp, hostname, and champion metadata, then uploads it to the champion folder. Colab reads this file to confirm the Linux PC is alive.

### MIME Type Handling

```python
def _guess_mime(path: Path) -> str:
    # .ipynb → application/json
    # .pth / .pkl → application/octet-stream
    # .json → application/json
    # .log → text/plain
```

---

## 4. colab_watchdog.py — Colab Keepalive & Auto-Recovery

**File:** `infrastructure/colab/colab_watchdog.py`  
**Role:** Monitor whether the Colab session is active. If Colab goes silent, trigger recovery.

### The Core Problem

Google Colab kills sessions after **90 minutes of inactivity** (not 90 minutes of total runtime). The key insight is that "inactivity" means no user interaction — but a running training loop counts as activity. Simply pinging Colab with a network request is not enough; the session must be doing real computation.

The solution is a bidirectional heartbeat system via Google Drive:
- Linux PC writes `linux_heartbeat.json` → Drive (every 60 min via drive_manager)
- Colab training loop writes `colab_status.json` → Drive (every N training iterations)
- Watchdog reads `colab_status.json` and checks how old it is

### Configuration

```python
CHECK_INTERVAL_SEC = 60 * 60   # Check every hour
COLAB_TIMEOUT_SEC  = 60 * 90   # 90 min without status = Colab dead
```

### Status Reading

```python
def read_colab_status(service, config) -> dict | None:
    """Downloads colab_status.json from Drive and returns parsed JSON."""
```

The Colab notebook must write and upload this file periodically:
```python
# In the Colab notebook (every N iterations):
status = {
    "timestamp": datetime.now().isoformat(),
    "iteration": current_iteration,
    "loss": current_loss,
    "gpu": torch.cuda.memory_allocated() / 1e9
}
upload_file(service, status_file, folder_id, "Colab training status")
```

### Liveness Check

```python
def is_colab_alive(status: dict | None) -> bool:
    if not status:
        return False
    last_seen = datetime.fromisoformat(status["timestamp"])
    age_seconds = (datetime.now() - last_seen).total_seconds()
    return age_seconds < COLAB_TIMEOUT_SEC   # < 90 minutes
```

### Recovery Actions

When `is_colab_alive()` returns False:

1. **Send CRITICAL alert** via AlertManager (Telegram + log)
2. **Write restart signal to Drive**

```python
def send_restart_signal(service, config):
    signal_data = {
        "action": "restart_training",
        "reason": "colab_timeout",
        "requested_at": datetime.now().isoformat(),
        "requested_by": os.uname().nodename,
    }
    # Writes restart_requested.json to Drive champion folder
```

3. **Colab notebook checks for this file** at the start of each new session and resumes training automatically.
4. After Colab processes the signal, `clear_restart_signal()` deletes the file from Drive.

### `run_watchdog(alert_callback)`

Main entry point called by master.py:
```python
def run_watchdog(alert_callback=None):
    service = get_drive_service()
    config  = load_drive_config()
    status  = read_colab_status(service, config)
    
    if not is_colab_alive(status):
        msg = "Colab appears offline - sending restart signal"
        if alert_callback:
            alert_callback(msg)
        send_restart_signal(service, config)
```

---

## 5. listener.py — Error Receiver

**File:** `infrastructure/monitor/listener.py`  
**Role:** Receive error reports from the Colab notebook via two independent channels, then trigger auto-repair.

### Why Dual-Channel?

**Problem with the naive approach (single HTTP endpoint):**
1. Ngrok free tier changes the public URL every 8 hours → Colab does not know the new URL
2. An open port is a security vulnerability — anyone who discovers it can inject fake error reports

**Solution — Dual Channel:**

```
Channel 1: Flask HTTP (Port 5001)
  + Fast (sub-second delivery)
  + Works when Colab can reach the Linux PC directly or via ngrok
  - Requires network connectivity + valid URL

Channel 2: Google Drive as message bus
  + Always works (Drive is universally accessible from Colab)
  + No open port required
  - 2-minute polling delay (checked every 120 seconds)

Colab strategy:
  try HTTP POST → if fails → write error_report.json to Drive
  Linux PC polls both channels: Flask receives HTTP directly,
  background thread polls Drive for error_report.json
```

### Configuration

Stored in `config/listener_config.json`:

```json
{
  "api_token": "abc123...",            // Must match in Colab + Linux PC
  "port": 5001,                        // Port 5001 (less well-known than 5000)
  "rate_limit_per_min": 10,            // Per-IP rate limit
  "auto_repair": true,                 // Trigger repair automatically
  "drive_polling_interval_sec": 120    // Drive fallback check interval
}
```

**Token generation:**
```bash
python3 -c "import secrets; print(secrets.token_hex(32))"
```

### Security

#### Rate Limiting

Per-IP sliding window: tracks the last 10 request timestamps per IP. Rejects requests (HTTP 429) if more than `rate_limit_per_min` requests have arrived in the last 60 seconds.

```python
_request_times: dict = defaultdict(lambda: deque(maxlen=10))

def _check_rate_limit(ip: str, max_per_min: int) -> bool:
    now = time.time()
    times = _request_times[ip]
    # Remove entries older than 60 seconds
    while times and now - times[0] > 60:
        times.popleft()
    if len(times) >= max_per_min:
        return False
    times.append(now)
    return True
```

#### Token Verification

Uses `hmac.compare_digest` for timing-safe comparison. This prevents timing attacks where an attacker could determine the correct token character by character by measuring response time.

```python
def _verify_token(received: str, expected: str) -> bool:
    return hmac.compare_digest(received.encode(), expected.encode())
```

### Flask HTTP Server

Runs on port 5001 (configurable). Endpoints:

| Endpoint | Method | Auth | Description |
|---|---|---|---|
| `/health` | GET | None | Liveness probe — returns `{"status": "ok", "timestamp": ...}` |
| `/report_error` | POST | X-API-Token header | Receive error report JSON |
| `/status` | GET | X-API-Token header | Colab connectivity check |

**`/report_error` request body:**
```json
{
  "notebook_id": "BITCOIN4Traders_Colab_v3",
  "error_type": "OutOfMemoryError",
  "error_message": "CUDA out of memory. Tried to allocate 2.00 GiB",
  "stacktrace": "...",
  "iteration": 347,
  "timestamp": "2026-02-23T14:22:01"
}
```

**Processing pipeline for each received report:**
1. Rate limit check (→ 429 if exceeded)
2. Token verification (→ 403 if invalid)
3. JSON validation (→ 400 if malformed)
4. `process_error_report(data)`:
   a. Append to daily JSONL log file (`logs/colab_errors/errors_YYYY-MM-DD.jsonl`)
   b. Send CRITICAL alert via AlertManager (Telegram + log)
   c. Spawn background thread → `error_repair.repair(report)`

### Drive Fallback Channel

A background thread polls Drive every 120 seconds for `error_report.json` in the champion folder. When found:
1. Downloads and parses the file
2. Processes it identically to an HTTP report
3. Deletes the file from Drive (so it is not processed again)

```python
def _poll_drive_fallback(config):
    while True:
        time.sleep(config["drive_polling_interval_sec"])
        # check for error_report.json in Drive
        # if found: process_error_report(data) + delete file
```

### `run()` Entry Point

Called by master.py in a daemon thread:
1. Load config
2. Start Drive polling thread
3. Start Flask server (blocking)

---

## 6. error_repair.py — Auto-Repair Engine

**File:** `infrastructure/monitor/error_repair.py`  
**Role:** Receive an error report, classify the error, patch the Colab notebook JSON to fix the root cause, upload the patched notebook, and send a restart signal.

### Error Classification

```python
ERROR_PATTERNS = [
    (r"(CUDA out of memory|OutOfMemoryError|OOM|out of memory)",          "OOM",        "high"),
    (r"(ConnectionError|ConnectTimeout|RemoteDisconnected)",               "CONNECTION",  "medium"),
    (r"(ModuleNotFoundError|ImportError|No module named)",                 "IMPORT",      "medium"),
    (r"(nan|NaN|inf|Inf).*(loss|reward|gradient)",                        "NAN_LOSS",    "high"),
    (r"(TimeoutError|ReadTimeout|socket.timeout)",                         "TIMEOUT",     "low"),
    (r"(RuntimeError.*CUDA|device-side assert)",                           "CUDA_ERROR",  "high"),
    (r"(KeyError|IndexError|ValueError).*(batch|data|feature)",            "DATA_ERROR",  "medium"),
    (r"(PermissionError|FileNotFoundError).*(drive|model|cache)",          "IO_ERROR",    "medium"),
    (r"(gradient.*explod|loss.*explod|overflow)",                          "EXPLODING",   "high"),
]
```

`classify_error(error_message, stacktrace)` applies patterns in order and returns the first match, or "UNKNOWN" if no pattern matches.

### Notebook Patching Mechanism

The Colab notebook is a JSON file (`.ipynb` format). Code cells are stored as arrays of strings:

```json
{
  "cells": [
    {
      "cell_type": "code",
      "source": ["BATCH_SIZE = 64\n", "LEARNING_RATE = 3e-4\n", ...]
    }
  ]
}
```

The `_patch_notebook_parameter(nb, old_value, new_value, description)` function iterates over all code cells and replaces `old_value` with `new_value` in every line (skipping comment lines). Returns the modified notebook and the number of changes made.

### Repair Recipes

The engine has **five active repair recipes** plus two restart-only error types:

#### Recipe 1: OOM (Out of Memory)

**Trigger:** `CUDA out of memory`, `OutOfMemoryError`, `OOM`  
**Root cause:** Batch size or sequence length is too large for the available GPU VRAM.

**Action:**
1. Find every line matching `BATCH_SIZE = <N>` or `batch_size = <N>` (case-insensitive regex)
2. Halve the value: new_value = max(8, old_value // 2)
3. Minimum batch size is 8 (training breaks below this)
4. Also applied to `CUDA_ERROR` (same root cause)

```
BATCH_SIZE = 64  →  BATCH_SIZE = 32
batch_size=128   →  batch_size=64
```

#### Recipe 2: NaN Loss

**Trigger:** `nan` / `NaN` / `inf` in loss or reward  
**Root cause:** Learning rate is too high, causing exploding gradients and numerical instability.

**Action:**
1. Find every line matching `LEARNING_RATE = <X>`, `learning_rate = <X>`, or `lr = <X>`
2. Divide the value by 10
3. Reformat as scientific notation (e.g., `3e-4` → `3e-05`)

```
LEARNING_RATE = 3e-4  →  LEARNING_RATE = 3.0e-05
```

#### Recipe 3: Exploding Gradient

**Trigger:** `gradient.*explod`, `loss.*explod`, `overflow`  
**Root cause:** Gradients are growing unbounded — gradient clipping is too loose.

**Action:**
1. Find `gradient_clip_val=0.5` and replace with `gradient_clip_val=0.1`
2. If no `gradient_clip_val` parameter found: fall back to `repair_nan_loss()` (reduce LR)

#### Recipe 4: Import Error (Missing Module)

**Trigger:** `ModuleNotFoundError`, `No module named '<X>'`  
**Root cause:** A required package is not installed in the Colab environment.

**Action:**
1. Extract module name from error message using regex
2. Insert `!pip install -q <module>  # Auto-Repair` as the first line of the first code cell
3. On next session start, pip will install the missing dependency automatically

#### Recipe 5: Timeout

**Trigger:** `TimeoutError`, `ReadTimeout`, `socket.timeout`  
**Root cause:** Network request timeout (to exchange API or Drive) is too short.

**Action:**
1. Find `REQUEST_TIMEOUT = 30`
2. Replace with `REQUEST_TIMEOUT = 60`

### Restart-Only Error Types

For `CONNECTION` and `IO_ERROR`, no notebook patch is needed — the error is transient. The engine skips patching and directly sends a restart signal.

### Main `repair(report)` Function

Full repair pipeline:

```
1. classify_error(error_message, stacktrace) → error_type
2. Load BITCOIN4Traders_Colab.ipynb as JSON
3. Create backup: BITCOIN4Traders_Colab.ipynb.backup
4. Dispatch to repair_fn = REPAIR_FUNCTIONS[error_type]
5. repair_fn(nb, report) → patched notebook dict
6. Write patched notebook back to BITCOIN4Traders_Colab.ipynb
7. Upload patched notebook to Drive (overwrites existing)
8. Send restart signal via Drive
9. Log repair record to logs/repairs.jsonl
10. Return True if successful, False otherwise
```

### Repair Log

Every repair attempt is recorded in `logs/repairs.jsonl` (newline-delimited JSON):

```json
{"timestamp": "2026-02-23T14:22:05", "notebook_id": "...", "error_type": "OOM",
 "error_message": "CUDA out of memory...", "patch_applied": true, "restart_sent": true}
```

---

## 7. alert_manager.py — Notification System

**File:** `infrastructure/monitor/alert_manager.py`  
**Role:** Centralised alert routing with deduplication, three severity levels, and three delivery channels.

### Three Channels

| Channel | Library | When used |
|---|---|---|
| Telegram Bot | `urllib.request` (stdlib only) | When enabled and configured |
| Log file | Built-in `open()` | Always (never fails) |
| Desktop popup | `notify-send` (Linux) | When `desktop_notify: true` |

### Three Severity Levels

| Severity | Minimum resend interval | Description |
|---|---|---|
| `info` | 60 min | Routine status updates |
| `warning` | 30 min | Degraded state, non-critical |
| `critical` | 5 min | Action required immediately |

### Public API

```python
def alert(message: str, severity: Severity = "info", source: str = "unknown")
def warn(message: str, source: str = "unknown")     # → alert(severity="warning")
def critical(message: str, source: str = "unknown") # → alert(severity="critical")
```

### Deduplication

Prevents alert spam. The same message (identified by a hash of the message text + source) is sent at most once per `min_interval_sec[severity]`.

```python
_last_sent: dict[str, datetime] = {}

def _should_send(key: str, severity: Severity, config: dict) -> bool:
    min_sec = config["min_interval_sec"][severity]   # e.g., 1800 for WARNING
    last = _last_sent.get(key)
    if last is None or (datetime.now() - last).total_seconds() > min_sec:
        _last_sent[key] = datetime.now()
        return True
    return False
```

### Telegram Setup

Configure `config/alert_config.json`:

```json
{
  "telegram": {
    "enabled": true,
    "bot_token": "123456:ABC-DEF...",
    "chat_id": "987654321"
  },
  "desktop_notify": true,
  "min_interval_sec": {
    "info": 3600,
    "warning": 1800,
    "critical": 300
  },
  "log_retention_days": 7
}
```

**How to get Telegram credentials:**
1. Send `/newbot` to `@BotFather` → get `bot_token`
2. Send any message to your bot → visit `https://api.telegram.org/bot<TOKEN>/getUpdates` → find `chat.id`

### Telegram Sending

Uses only the Python standard library (`urllib.request`) — no `requests` package required:

```python
url = f"https://api.telegram.org/bot{token}/sendMessage"
     f"?chat_id={chat_id}&text={urllib.parse.quote(message)}&parse_mode=HTML"
urllib.request.urlopen(url, timeout=10)
```

HTML parse_mode allows bold (`<b>`) and code (`<code>`) formatting in Telegram messages.

### Log Rotation

Log files are stored in `logs/alerts/alerts_YYYY-MM-DD.log`. Files older than `log_retention_days` (default: 7) are automatically deleted on each write.

### Interactive Setup

Running `python3 infrastructure/monitor/alert_manager.py setup` starts an interactive prompt:
1. Enter Telegram bot token
2. Enter chat ID
3. Send test message
4. Saves config to `config/alert_config.json`

---

## 8. Systemd Service

**File:** `infrastructure/systemd/bitcoin4traders.service`

The systemd unit ensures the Master Orchestrator runs continuously, survives reboots, and restarts automatically after crashes.

### Unit File

```ini
[Unit]
Description=BITCOIN4Traders Master Orchestrator
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=hp17
WorkingDirectory=/home/hp17/Tradingbot/Quantrivo/BITCOIN4Traders
ExecStart=/usr/bin/python3 infrastructure/master.py
Restart=on-failure
RestartSec=60
# Give up after 3 crashes within 10 min (no endless loop)
StartLimitIntervalSec=600
StartLimitBurst=3
StandardOutput=journal
StandardError=journal
SyslogIdentifier=bitcoin4traders
Environment=PYTHONUNBUFFERED=1
Environment=HOME=/home/hp17

[Install]
WantedBy=multi-user.target
```

### Key Service Directives

| Directive | Value | Effect |
|---|---|---|
| `After=network-online.target` | — | Wait for network before starting |
| `Restart=on-failure` | — | Restart only on non-zero exit, not on clean shutdown |
| `RestartSec=60` | 60s | Wait 60 seconds before each restart attempt |
| `StartLimitIntervalSec=600` | 600s | Rate limit window for restarts |
| `StartLimitBurst=3` | 3 | Max 3 restarts in 600 seconds, then give up |
| `PYTHONUNBUFFERED=1` | — | Flush stdout immediately → real-time journald logs |

### Management Commands

```bash
# Install (run once)
sudo cp infrastructure/systemd/bitcoin4traders.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable bitcoin4traders    # auto-start on boot
sudo systemctl start bitcoin4traders

# Daily operations
sudo systemctl status bitcoin4traders    # current state
journalctl -u bitcoin4traders -f         # follow live logs
journalctl -u bitcoin4traders --since "1 hour ago"   # recent logs
sudo systemctl restart bitcoin4traders   # manual restart
sudo systemctl stop bitcoin4traders      # graceful stop (SIGTERM → clean shutdown)
```

### Restart Behaviour

If master.py crashes:
- systemd waits 60 seconds (`RestartSec=60`)
- Restarts the process
- If it crashes again within 10 minutes, after 3 total attempts, systemd gives up
- AlertManager sends a CRITICAL alert before shutdown (if master.py had time to do so)

---

## 9. install.sh — Setup Guide

**File:** `infrastructure/install.sh`

One-time setup script. Run once after cloning the repository.

### Prerequisites

Before running the script:
1. Clone the repository to the target machine
2. Obtain `credentials.json` from Google Cloud Console (Service Account key)
3. Place it at `config/gdrive_credentials.json`
4. Ensure GitHub remote is configured (`git remote -v`)

### Usage

```bash
chmod +x infrastructure/install.sh
./infrastructure/install.sh
```

### Steps Performed

#### Step 1: Install Python Packages

```bash
pip3 install --quiet --upgrade \
    google-api-python-client \
    google-auth \
    google-auth-httplib2 \
    flask \
    requests
```

These are the only infrastructure-specific packages. All other project dependencies (`torch`, `gymnasium`, `ccxt`, etc.) are listed in `requirements.txt`.

#### Step 2: Create Directory Structure

```
logs/alerts/
logs/infrastructure/
config/
data/cache/
```

#### Step 3: Verify Google Credentials

Checks for `config/gdrive_credentials.json`. If missing, prints step-by-step instructions:
1. Go to Google Cloud Console
2. Enable Google Drive API
3. Create Service Account → generate JSON key
4. Share Drive folder with service account email

#### Step 4: Configure Google Drive Folder IDs

Runs `python3 infrastructure/drive/drive_manager.py setup` if credentials are present. This interactive prompt asks for Drive folder IDs and saves them to `config/drive_config.json`.

#### Step 5: Configure Telegram Alerts

Interactive: asks whether to configure Telegram now.  
If yes: runs `python3 infrastructure/monitor/alert_manager.py setup`.  
If no: can be run later manually.

#### Step 6: Install Systemd Service

Interactive: asks whether to install as a persistent service.  
If yes: copies service file, runs `daemon-reload`, `enable`, `start`.

#### Step 7: Update `.gitignore`

Automatically adds sensitive files to `.gitignore`:
```
config/gdrive_credentials.json
config/alert_config.json
config/drive_config.json
logs/alerts/
logs/infrastructure/
/tmp/
```

**Important:** Never commit credentials or configuration files containing tokens to git.

### Post-Setup: Colab Configuration

After running `install.sh`, configure the Colab notebook variables:

```python
# At the top of BITCOIN4Traders_Colab.ipynb
LINUX_IP          = "your-ip-or-ngrok-url"   # Where listener.py runs
LINUX_API_TOKEN   = "token-from-listener"      # From listener.py setup
DRIVE_FOLDER_ID   = "folder-id-from-drive"    # From drive_manager.py setup
```

Generate the API token:
```bash
python3 infrastructure/monitor/listener.py setup
```

---

## 10. End-to-End Self-Healing Flow

This section traces the complete lifecycle of an error, from detection in Colab to autonomous recovery.

### Normal Operation (no errors)

```
Linux PC (every 30s)                     Google Drive                  Colab
═══════════════════════════════════════════════════════════════════════════════
Master scheduler runs                         │                          │
  │                                           │                          │
  ├─ Every 60 min: run_sync("up") ────────▶  linux_heartbeat.json       │
  │                                           │                          │
  ├─ Every 60 min: run_watchdog() ──── reads ◀─── colab_status.json ◀── Colab writes
  │   is_colab_alive() = TRUE                 │    every 50 iterations   │
  │   → no action needed                      │                          │
  │                                           │                          │
  └─ Every 6h: github_backup()               │                          │
```

### Scenario 1: Colab session dies (timeout / manual stop)

```
Time 0:00  Colab last wrote colab_status.json
Time 0:30  Colab session killed by Google (or manually stopped)
Time 1:30  Linux PC: run_watchdog() checks colab_status.json
           age = 90 min = COLAB_TIMEOUT_SEC
           is_colab_alive() = FALSE
           │
           ├─ AlertManager.critical("Colab appears offline")
           │   → Telegram push notification to phone
           │   → Written to logs/alerts/alerts_YYYY-MM-DD.log
           │
           └─ send_restart_signal() → writes restart_requested.json to Drive

User starts new Colab session (manually or via scheduled trigger):
  Notebook cell at startup: checks for restart_requested.json
  → Finds it
  → Resumes training from last checkpoint
  → Deletes restart_requested.json (clear_restart_signal)
  → Writes new colab_status.json
```

### Scenario 2: Training error in Colab (OOM crash)

```
Colab: CUDA out of memory at iteration 347
  │
  ├─ Catch exception in notebook error handler
  │
  ├─ Try HTTP POST to http://LINUX_IP:5001/report_error
  │   headers: {"X-API-Token": "abc123..."}
  │   body: {"error_type": "OutOfMemoryError", "error_message": "...", ...}
  │
  │   [If HTTP succeeds]:
  │   Linux listener.py receives POST
  │     → Rate limit check: OK (first request in 60s)
  │     → Token verification: OK
  │     → process_error_report(data)
  │         → Append to logs/colab_errors/errors_2026-02-23.jsonl
  │         → AlertManager.critical("Colab error: OOM")
  │         → Spawn thread: error_repair.repair(report)
  │
  │   [If HTTP fails (no tunnel / network issue)]:
  │   Colab falls back: write error_report.json to Drive
  │
  │   Linux listener background thread (runs every 120s):
  │     → Finds error_report.json in Drive
  │     → process_error_report(data) [same path as HTTP]
  │     → Deletes error_report.json from Drive
  │
  └─ error_repair.repair(report):
       │
       ├─ classify_error("CUDA out of memory...") = "OOM"
       ├─ Load BITCOIN4Traders_Colab.ipynb (JSON)
       ├─ Create backup: BITCOIN4Traders_Colab.ipynb.backup
       ├─ repair_oom(nb, report):
       │     Find "BATCH_SIZE = 64" → replace with "BATCH_SIZE = 32"
       │     Find "batch_size=128" → replace with "batch_size=64"
       │     2 changes made
       ├─ Write patched notebook back to disk
       ├─ upload_file(service, notebook, folder_id) → Drive
       ├─ send_restart_signal() → restart_requested.json to Drive
       └─ Log to logs/repairs.jsonl: {"error_type": "OOM", "patch_applied": true}

Next Colab session:
  → Downloads patched notebook from Drive (smaller batch_size)
  → Reads restart_requested.json → resumes training
  → Clears restart signal
  → Training continues with new parameters
```

### Scenario 3: Cascading failure (multiple error types)

If the same error type recurs after a repair:
- A second OOM after the first repair will halve the batch size again (e.g., 32 → 16)
- A second NaN loss will reduce LR by another 10× (e.g., 3e-5 → 3e-6)
- The minimum batch size of 8 is enforced as a hard floor
- All repairs are logged with the iteration number for diagnosis

If an UNKNOWN error type is received:
- No notebook patch is applied
- A restart signal is sent anyway (often resolves transient issues)
- A CRITICAL alert is sent to Telegram with the full error message
- The operator must review `logs/repairs.jsonl` and address manually

### Alert Flow Summary

```
Any infrastructure event
       │
       ▼
alert_manager.alert(message, severity, source)
       │
       ├─ _should_send(key, severity) ?  → deduplication check
       │   NO: skip (too recent)
       │   YES: proceed
       │
       ├─ _log_to_file(message, severity)    ← always happens
       │   → logs/alerts/alerts_YYYY-MM-DD.log
       │
       ├─ _send_telegram(message, config)    ← if enabled
       │   → POST to Telegram Bot API
       │
       └─ _send_desktop(title, message)      ← if desktop_notify=true
           → subprocess: notify-send
```

### Recovery Coverage Matrix

| Failure Mode | Detected By | Recovery Action |
|---|---|---|
| Colab session timeout | colab_watchdog.py (60 min check) | Restart signal via Drive |
| OOM during training | error_repair.py | Halve batch_size, restart |
| NaN/Inf loss | error_repair.py | Reduce learning rate × 10, restart |
| Exploding gradients | error_repair.py | Tighten gradient clip, restart |
| Missing Python module | error_repair.py | Inject pip install, restart |
| Request timeout | error_repair.py | Increase timeout parameter, restart |
| Connection error | error_repair.py | Restart (transient) |
| Local training stopped | task_local_training_check() | WARNING alert |
| Drive sync failed | task_drive_sync() | WARNING alert |
| GitHub backup failed | task_github_backup() | WARNING alert |
| Master orchestrator crash | systemd (Restart=on-failure) | Auto-restart after 60s |
| Linux PC reboot | systemd (WantedBy=multi-user.target) | Auto-start on boot |
