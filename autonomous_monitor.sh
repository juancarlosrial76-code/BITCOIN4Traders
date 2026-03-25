#!/bin/bash
# Autonomous overnight monitor — runs every 15 min via system cron
# Full autonomy: fix, restart, commit — no human input needed

WORKDIR="/home/hp17/Tradingbot/BITCOIN4Traders"
WATCHLOG="$WORKDIR/logs/watchdog.log"
cd "$WORKDIR" || exit 1

ts() { date '+%Y-%m-%d %H:%M:%S'; }
log() { echo "[$(ts)] $*" >> "$WATCHLOG"; }

log "=== AUTONOMOUS CHECK START ==="

# ── 1. Training alive? ──────────────────────────────────────────────
TRAIN_PID=$(pgrep -f auto_12h_train | head -1)
if [ -z "$TRAIN_PID" ]; then
    log "TRAINING DEAD — restarting"
    # Check last error before restart
    LAST_ERR=$(tail -30 logs/training/12h_auto.log 2>/dev/null | grep -i "timeout\|error\|exception" | tail -3)
    log "Last error: $LAST_ERR"

    # If repeated timeouts, reduce iterations further
    TIMEOUT_COUNT=$(grep -c "Timeout" logs/training/12h_errors.log 2>/dev/null || echo 0)
    if [ "$TIMEOUT_COUNT" -gt 5 ]; then
        # Reduce to 10 iterations
        python3 -c "
import re
with open('auto_12h_train.py') as f: c = f.read()
c = re.sub(r'\"--iterations\", \"\d+\"', '\"--iterations\", \"10\"', c)
with open('auto_12h_train.py', 'w') as f: f.write(c)
print('Reduced to 10 iterations')
" >> "$WATCHLOG" 2>&1
        git add auto_12h_train.py && git commit -m "fix: reduce iterations to 10 due to repeated timeouts" >> "$WATCHLOG" 2>&1
    fi

    LOG_TS=$(date +%Y%m%d_%H%M%S)
    nohup python auto_12h_train.py > "logs/training/auto_12h_${LOG_TS}.log" 2>&1 &
    log "Training restarted PID=$!"
else
    log "Training OK PID=$TRAIN_PID"
fi

# ── 2. Paper trading alive? ─────────────────────────────────────────
PAPER_PID=$(pgrep -f "run.py --dry_run" | head -1)
if [ -z "$PAPER_PID" ]; then
    log "PAPER DEAD — restarting"
    LOG_TS=$(date +%Y%m%d_%H%M%S)
    nohup python run.py --dry_run > "logs/paper/paper_${LOG_TS}.log" 2>&1 &
    log "Paper restarted PID=$!"
else
    log "Paper OK PID=$PAPER_PID"
fi

# ── 3. Training progress ────────────────────────────────────────────
LAST_LOG_LINES=$(tail -50 logs/training/12h_auto.log 2>/dev/null)
ROUND=$(echo "$LAST_LOG_LINES" | grep "Round " | tail -1)
REWARD=$(echo "$LAST_LOG_LINES" | grep -i "mean return\|reward\|best=" | tail -1)
TIMEOUT_NOW=$(echo "$LAST_LOG_LINES" | grep -c "Timeout")
log "Training progress: $ROUND | $REWARD | timeouts_this_check=$TIMEOUT_NOW"

# ── 4. Model saved? ─────────────────────────────────────────────────
if [ -f "$WORKDIR/data/models/ppo_best.pt" ]; then
    MTIME=$(stat -c '%y' "$WORKDIR/data/models/ppo_best.pt" | cut -d. -f1)
    log "MODEL EXISTS: ppo_best.pt mtime=$MTIME"
else
    log "Model not yet saved"
fi

# ── 5. Paper trading status ─────────────────────────────────────────
PLOG=$(ls -t logs/paper/paper_*.log 2>/dev/null | head -1)
if [ -n "$PLOG" ]; then
    TICKS=$(tail -30 "$PLOG" | grep "Ticks:" | tail -1 | grep -o "Ticks: *[0-9]*")
    log "Paper status: $TICKS"
fi

log "=== AUTONOMOUS CHECK END ==="
echo "" >> "$WATCHLOG"
