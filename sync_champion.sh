#!/bin/bash
# =============================================================================
# sync_champion.sh - Local Master Champion Synchronization
# =============================================================================
# Purpose: Synchronizes the locally trained champion to GitHub so that
#          GitHub Actions (backup bot) always uses the best champion.
#
# Architecture:
#   Linux PC (Local Master) --> GitHub --> GitHub Actions (Backup Bot)
#                                     --> Google Colab (AI Lab)
#
# Usage:
#   ./sync_champion.sh                     # Normal sync
#   ./sync_champion.sh --force             # Sync even without changes
#   ./sync_champion.sh --dry-run           # Check only, do not push
#
# Automation (Crontab):
#   crontab -e
#   # Sync every 6 hours:
#   0 */6 * * * /home/hp17/Tradingbot/Quantrivo/BITCOIN4Traders/sync_champion.sh >> /home/hp17/Tradingbot/Quantrivo/BITCOIN4Traders/logs/sync.log 2>&1
# =============================================================================

set -euo pipefail

# --- Configuration -----------------------------------------------------------
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CACHE_DIR="${REPO_DIR}/data/cache"
LOG_FILE="${REPO_DIR}/logs/sync.log"
FORCE=false
DRY_RUN=false

# --- Parse arguments ---------------------------------------------------------
for arg in "$@"; do
    case $arg in
        --force)   FORCE=true ;;
        --dry-run) DRY_RUN=true ;;
    esac
done

# --- Logging -----------------------------------------------------------------
mkdir -p "${REPO_DIR}/logs"
timestamp() { date '+%Y-%m-%d %H:%M:%S'; }
log() { echo "[$(timestamp)] $*" | tee -a "${LOG_FILE}"; }

log "======================================================"
log "  Champion Sync started (Local Master -> GitHub)"
log "======================================================"

# --- Check network connectivity ----------------------------------------------
log "Checking network..."
if ! ping -c 1 -W 5 github.com &>/dev/null; then
    log "ERROR: No network. Sync aborted."
    exit 1
fi
log "Network OK"

# --- Change to repo directory -------------------------------------------------
cd "${REPO_DIR}"

# --- Check for champion files -------------------------------------------------
CHAMPION_PKL="${CACHE_DIR}/multiverse_champion.pkl"
CHAMPION_META="${CACHE_DIR}/multiverse_champion_meta.json"

# Skip sync if no champion files exist yet
if [ ! -f "${CHAMPION_PKL}" ] && [ ! -f "${CHAMPION_META}" ]; then
    log "WARNING: No champion files found in ${CACHE_DIR}."
    log "Sync skipped."
    exit 0
fi

# --- Check git status ---------------------------------------------------------
log "Checking git status..."

# Count unstaged changes in the data/cache/ directory
CHANGED_FILES=$(git status --porcelain data/cache/ 2>/dev/null | wc -l)

# Skip if nothing changed and --force not set
if [ "${CHANGED_FILES}" -eq 0 ] && [ "${FORCE}" = false ]; then
    log "No changes in champion cache. Sync not needed."
    log "  (Use --force to force a sync)"
    exit 0
fi

log "Changed files: ${CHANGED_FILES}"

# --- Print champion metadata --------------------------------------------------
if [ -f "${CHAMPION_META}" ]; then
    # Extract champion name and Sharpe ratio from JSON metadata
    CHAMPION_NAME=$(python3 -c "import json; d=json.load(open('${CHAMPION_META}')); print(d.get('name','unknown'))" 2>/dev/null || echo "unknown")
    CHAMPION_SHARPE=$(python3 -c "import json; d=json.load(open('${CHAMPION_META}')); print(round(d.get('sharpe',0),3))" 2>/dev/null || echo "?")
    log "Champion: ${CHAMPION_NAME} (Sharpe: ${CHAMPION_SHARPE})"
fi

# --- Dry-run mode: print what would be pushed, then exit ---------------------
if [ "${DRY_RUN}" = true ]; then
    log "[DRY-RUN] Files that would be pushed:"
    git status --porcelain data/cache/
    log "[DRY-RUN] No actual push performed."
    exit 0
fi

# --- Stage, commit and push champion files ------------------------------------
log "Staging champion files..."

# Only stage champion cache files (no secrets or temporary files)
git add data/cache/multiverse_champion.pkl 2>/dev/null || true
git add data/cache/multiverse_champion_meta.json 2>/dev/null || true
git add data/cache/heartbeat.txt 2>/dev/null || true

# Verify there is something staged before committing
STAGED=$(git diff --cached --name-only | wc -l)
if [ "${STAGED}" -eq 0 ]; then
    log "Nothing to commit (files unchanged)."
    exit 0
fi

# Build a descriptive commit message including champion stats
COMMIT_MSG="Auto-Sync: Local Master Champion Update $(date '+%Y-%m-%d %H:%M')"
if [ -f "${CHAMPION_META}" ]; then
    COMMIT_MSG="Auto-Sync: ${CHAMPION_NAME} (Sharpe=${CHAMPION_SHARPE}) - $(date '+%Y-%m-%d %H:%M')"
fi

log "Commit: ${COMMIT_MSG}"
git commit -m "${COMMIT_MSG}"

log "Pushing to GitHub..."
git push origin main

log "Champion successfully synchronized!"
log "GitHub Actions (Backup Bot) will now use the new champion."
log "======================================================"
