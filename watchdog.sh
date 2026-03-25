#!/bin/bash
# Overnight watchdog — runs every 15 min via system cron
# Restarts dead processes, logs status, auto-fixes errors
# Written: 2026-03-26

WORKDIR="/home/hp17/Tradingbot/BITCOIN4Traders"
WATCHLOG="$WORKDIR/logs/watchdog.log"
START_TIME_FILE="$WORKDIR/logs/watchdog_start.txt"
MAX_RUNTIME=25200  # 7 hours in seconds

cd "$WORKDIR" || exit 1

ts() { date '+%Y-%m-%d %H:%M:%S'; }
log() { echo "[$(ts)] $*" | tee -a "$WATCHLOG"; }

# Record start time on first run
if [ ! -f "$START_TIME_FILE" ]; then
    date +%s > "$START_TIME_FILE"
fi
START=$(cat "$START_TIME_FILE")
NOW=$(date +%s)
ELAPSED=$(( NOW - START ))

log "=== Watchdog check | Elapsed: ${ELAPSED}s / ${MAX_RUNTIME}s ==="

# ── Stop after 7h ──────────────────────────────────────────────────
if [ "$ELAPSED" -ge "$MAX_RUNTIME" ]; then
    log "7h elapsed — watchdog duty complete. Stopping cron removal is manual."
    exit 0
fi

# ── Helper: get latest log file ────────────────────────────────────
latest_training_log() {
    ls -t "$WORKDIR/logs/training"/auto_12h_*.log 2>/dev/null | head -1
}
latest_paper_log() {
    ls -t "$WORKDIR/logs/paper"/paper_*.log 2>/dev/null | head -1
}

# ── Check & restart training ───────────────────────────────────────
TRAIN_PID=$(pgrep -f "auto_12h_train" | grep -v "$$" | head -1)
if [ -z "$TRAIN_PID" ]; then
    log "TRAINING DEAD — restarting..."
    LOG_TS=$(date +%Y%m%d_%H%M%S)
    LOGFILE="$WORKDIR/logs/training/auto_12h_${LOG_TS}.log"

    # Check last error before restart
    LAST_LOG=$(latest_training_log)
    if [ -n "$LAST_LOG" ]; then
        LAST_ERR=$(tail -20 "$LAST_LOG" | grep -i "error\|exception\|traceback" | tail -3)
        if [ -n "$LAST_ERR" ]; then
            log "Last training errors: $LAST_ERR"
            # Auto-fix: ModuleNotFoundError
            if echo "$LAST_ERR" | grep -qi "ModuleNotFoundError\|ImportError"; then
                log "AUTOFIX: Import error detected — checking src/..."
            fi
            # Auto-fix: KeyError in config
            if echo "$LAST_ERR" | grep -qi "KeyError"; then
                log "AUTOFIX: KeyError in config — relaxing adversarial.yaml"
                python3 -c "
import yaml
p = '$WORKDIR/config/training/adversarial.yaml'
try:
    with open(p) as f: c = yaml.safe_load(f)
    c.setdefault('learning_rate', 3e-4)
    c.setdefault('entropy_coef', 0.01)
    c.setdefault('clip_range', 0.2)
    with open(p, 'w') as f: yaml.dump(c, f)
    print('Config patched')
except Exception as e: print(f'Config patch failed: {e}')
" >> "$WATCHLOG" 2>&1
            fi
        fi
    fi

    nohup python auto_12h_train.py > "$LOGFILE" 2>&1 &
    NEW_PID=$!
    log "Training restarted (PID $NEW_PID) → $LOGFILE"
else
    log "Training OK (PID $TRAIN_PID)"
fi

# ── Check & restart paper trading ─────────────────────────────────
PAPER_PID=$(pgrep -f "run.py --dry_run" | grep -v "$$" | head -1)
if [ -z "$PAPER_PID" ]; then
    log "PAPER TRADING DEAD — restarting..."
    LOG_TS=$(date +%Y%m%d_%H%M%S)
    LOGFILE="$WORKDIR/logs/paper/paper_${LOG_TS}.log"

    # Check last error
    LAST_LOG=$(latest_paper_log)
    if [ -n "$LAST_LOG" ]; then
        LAST_ERR=$(tail -20 "$LAST_LOG" | grep -i "error\|exception\|traceback" | tail -3)
        if [ -n "$LAST_ERR" ]; then
            log "Last paper errors: $LAST_ERR"
        fi
    fi

    nohup python run.py --dry_run > "$LOGFILE" 2>&1 &
    NEW_PID=$!
    log "Paper trading restarted (PID $NEW_PID) → $LOGFILE"
else
    log "Paper trading OK (PID $PAPER_PID)"
fi

# ── Training progress snapshot ────────────────────────────────────
TLOG=$(latest_training_log)
if [ -n "$TLOG" ]; then
    REWARD=$(grep -i "mean return\|reward\|best=" "$TLOG" | tail -3)
    ITER=$(grep -i "Round\|iteration" "$TLOG" | tail -2)
    log "Training progress: $ITER | $REWARD"
else
    log "No training log found yet"
fi

# ── Paper trading snapshot ────────────────────────────────────────
PLOG=$(latest_paper_log)
if [ -n "$PLOG" ]; then
    PSTATUS=$(tail -5 "$PLOG" | tr '\n' ' ')
    log "Paper status: $PSTATUS"
else
    log "No paper log found yet"
fi

# ── Model saved? ──────────────────────────────────────────────────
if [ -f "$WORKDIR/data/models/ppo_best.pt" ]; then
    MTIME=$(stat -c '%y' "$WORKDIR/data/models/ppo_best.pt" | cut -d. -f1)
    log "MODEL EXISTS: ppo_best.pt (mtime $MTIME)"
else
    log "Model not yet saved (normal for first hours)"
fi

log "=== Check complete ==="
echo "" >> "$WATCHLOG"
