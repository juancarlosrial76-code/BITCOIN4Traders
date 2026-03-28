#!/bin/bash
# Autonomous overnight monitor — runs every 15 min via system cron
# Goal: maximum daily profit validated by paper trading
# Guarantees: exactly 1 training process, exactly 1 paper trading process
#             auto-deploy new model when training improves

WORKDIR="/home/hp17/Tradingbot/BITCOIN4Traders"
WATCHLOG="$WORKDIR/logs/watchdog.log"
LOCK="/tmp/btc4t_monitor.lock"

cd "$WORKDIR" || exit 1

ts()  { date '+%Y-%m-%d %H:%M:%S'; }
log() { echo "[$(ts)] $*" >> "$WATCHLOG"; }

# ── Lock: prevent overlapping cron runs ─────────────────────────────
if [ -f "$LOCK" ]; then
    LOCK_AGE=$(( $(date +%s) - $(stat -c %Y "$LOCK") ))
    if [ "$LOCK_AGE" -lt 840 ]; then   # 14 min — cron fires every 15
        log "SKIP: previous run still active (lock age ${LOCK_AGE}s)"
        exit 0
    fi
    log "WARN: stale lock (age ${LOCK_AGE}s), removing"
fi
echo $$ > "$LOCK"
trap 'rm -f "$LOCK"' EXIT

log "=== MONITOR START ==="

# ── 1. Kill duplicate training processes ────────────────────────────
TRAIN_PIDS=$(pgrep -f "auto_12h_train" | sort -n)
TRAIN_COUNT=$(echo "$TRAIN_PIDS" | grep -c "[0-9]")
if [ "$TRAIN_COUNT" -gt 1 ]; then
    log "WARN: $TRAIN_COUNT auto_12h processes — killing extras"
    # Keep only the newest (last PID), kill the rest
    NEWEST=$(echo "$TRAIN_PIDS" | tail -1)
    echo "$TRAIN_PIDS" | grep -v "^${NEWEST}$" | xargs -r kill -9 2>/dev/null
    log "Kept PID=$NEWEST, killed others"
fi

# ── 2. Training alive? ───────────────────────────────────────────────
TRAIN_PID=$(pgrep -f "auto_12h_train" | head -1)
if [ -z "$TRAIN_PID" ]; then
    log "TRAINING DEAD — restarting"
    LOG_TS=$(date +%Y%m%d_%H%M%S)
    TRAIN_LOG="$WORKDIR/logs/training/auto_12h_${LOG_TS}.log"
    nohup /home/hp17/miniconda3/bin/python3 auto_12h_train.py > "$TRAIN_LOG" 2>&1 &
    TRAIN_PID=$!
    log "Training restarted PID=$TRAIN_PID log=$TRAIN_LOG"
else
    log "Training OK PID=$TRAIN_PID"
fi

# ── 3. Paper trading alive? ──────────────────────────────────────────
PAPER_PID=$(pgrep -f "run.py --dry_run" | head -1)
if [ -z "$PAPER_PID" ]; then
    log "PAPER DEAD — restarting"
    LOG_TS=$(date +%Y%m%d_%H%M%S)
    PAPER_LOG="$WORKDIR/logs/paper/paper_${LOG_TS}.log"
    nohup /home/hp17/miniconda3/bin/python3 run.py --dry_run > "$PAPER_LOG" 2>&1 &
    PAPER_PID=$!
    log "Paper restarted PID=$PAPER_PID log=$PAPER_LOG"
else
    log "Paper OK PID=$PAPER_PID"
fi

# ── 4. Training progress (correct log path) ──────────────────────────
CURRENT_TRAIN_LOG=$(ls -t "$WORKDIR/logs/training/auto_12h_"*.log 2>/dev/null | head -1)
if [ -n "$CURRENT_TRAIN_LOG" ]; then
    ROUND=$(grep "Round " "$CURRENT_TRAIN_LOG" 2>/dev/null | tail -1)
    BEST=$(grep -E "Best=|best_return|Best Return" "$CURRENT_TRAIN_LOG" 2>/dev/null | tail -1)
    WR=$(grep -E "Win Rate:|Trade WR|Weighted Return" "$CURRENT_TRAIN_LOG" 2>/dev/null | tail -1)
    log "Training: $ROUND"
    log "Best return: $BEST"
    log "Win/Return: $WR"
else
    log "Training: no log found yet"
fi

# ── 5. Auto-deploy: nur wenn automl_loop NICHT läuft ────────────────
# Wenn automl_loop aktiv ist, entscheidet er selbst wann deployed wird
# (nur wenn Backtest besser). Blindes Deploy zerstört die Evaluation.
AUTOML_RUNNING=$(pgrep -f "automl_loop.py" | head -1)
if [ -n "$AUTOML_RUNNING" ]; then
    log "Auto-deploy SKIP: automl_loop läuft (PID=$AUTOML_RUNNING) — automl entscheidet"
else
    TRAINER_MODEL="$WORKDIR/data/models/adversarial/best_model_trader.pth"
    DEPLOYED_MODEL="$WORKDIR/data/models/ppo_best.pt"
    BEST_SCORE_FILE="$WORKDIR/logs/automl/best_params.json"

    if [ -f "$TRAINER_MODEL" ]; then
        TRAINER_MTIME=$(stat -c %Y "$TRAINER_MODEL")
        DEPLOYED_MTIME=$([ -f "$DEPLOYED_MODEL" ] && stat -c %Y "$DEPLOYED_MODEL" || echo 0)

        if [ "$TRAINER_MTIME" -gt "$DEPLOYED_MTIME" ]; then
            log "NEW MODEL detected — auto-deploying (automl not running)"
            /home/hp17/miniconda3/bin/python3 deploy_model.py --restart >> "$WATCHLOG" 2>&1
            log "Auto-deploy complete"
        else
            MTIME_H=$(stat -c '%y' "$DEPLOYED_MODEL" | cut -d. -f1)
            log "Model up-to-date: ppo_best.pt at $MTIME_H"
        fi
    fi
fi

# ── 6. Paper trading P&L (daily profit = the only metric that counts) ─
PLOG=$(ls -t "$WORKDIR/logs/paper/paper_"*.log 2>/dev/null | head -1)
if [ -n "$PLOG" ]; then
    REALIZED=$(grep "Realized:" "$PLOG" | tail -1 | grep -oP 'Realized: \$[-0-9.]+' || echo "?")
    TOTAL_PNL=$(grep "Total PnL:" "$PLOG" | tail -1 | grep -oP 'Total PnL: \+?\$[-0-9.]+' || echo "?")
    ORDERS=$(grep "Orders:" "$PLOG" | tail -1 | grep -oP 'Orders:\s+[0-9]+' || echo "?")
    TICKS=$(grep "Ticks:" "$PLOG" | tail -1 | grep -oP 'Ticks:\s+[0-9]+' || echo "?")
    log "Paper: $TICKS | $ORDERS | $REALIZED | $TOTAL_PNL"
else
    log "Paper: no log found"
fi

log "=== MONITOR END ==="
echo "" >> "$WATCHLOG"
