#!/bin/bash
# =============================================================================
# setup_local_master.sh - Configure Linux PC as Local Master
# =============================================================================
# This Linux PC becomes the "Local Master" in the 3-pillar architecture:
#
#   Linux PC (Local Master)  --> Primary bot, no limits, local SQLite DB
#   GitHub Actions           --> Backup bot on power failure
#   Google Colab             --> AI lab with GPU
#
# Run once:
#   chmod +x setup_local_master.sh
#   ./setup_local_master.sh
#
# Requirements:
#   - Ubuntu/Debian Linux
#   - sudo privileges
#   - Internet connection
# =============================================================================

set -e

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="python3"
PIP_BIN="pip3"

# ANSI color codes for terminal output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log()    { echo -e "${GREEN}[OK]${NC} $*"; }
warn()   { echo -e "${YELLOW}[WARN]${NC} $*"; }
error()  { echo -e "${RED}[ERROR]${NC} $*"; }
header() { echo -e "\n${BLUE}=== $* ===${NC}"; }

header "BITCOIN4Traders - Local Master Setup"
echo "Directory: ${REPO_DIR}"
echo "Date:      $(date)"
echo ""

# =============================================================================
# 1. Install system packages
# =============================================================================
header "1. System Packages (Node.js, PM2, Python Tools)"

sudo apt-get update -qq

# Python3 and pip
if ! command -v python3 &>/dev/null; then
    sudo apt-get install -y python3 python3-pip python3-venv
    log "Python3 installed"
else
    log "Python3 already present: $(python3 --version)"
fi

# Node.js for PM2 (LTS version via NodeSource)
if ! command -v node &>/dev/null; then
    echo "Installing Node.js..."
    curl -fsSL https://deb.nodesource.com/setup_lts.x | sudo -E bash -
    sudo apt-get install -y nodejs
    log "Node.js installed: $(node --version)"
else
    log "Node.js already present: $(node --version)"
fi

# PM2 (process manager for persistent background processes)
if ! command -v pm2 &>/dev/null; then
    sudo npm install -g pm2
    log "PM2 installed: $(pm2 --version)"
else
    log "PM2 already present: $(pm2 --version)"
fi

# Git (should already be installed)
if ! command -v git &>/dev/null; then
    sudo apt-get install -y git
    log "Git installed"
else
    log "Git already present: $(git --version)"
fi

# htop for monitoring (recommended in the architecture plan)
if ! command -v htop &>/dev/null; then
    sudo apt-get install -y htop
    log "htop installed"
fi

# SQLite3 CLI (for manual database queries)
if ! command -v sqlite3 &>/dev/null; then
    sudo apt-get install -y sqlite3
    log "sqlite3 installed"
fi

# =============================================================================
# 2. Install Python dependencies
# =============================================================================
header "2. Installing Python Dependencies"

cd "${REPO_DIR}"

# Create virtual environment (optional - keeps system Python clean)
if [ ! -d "${REPO_DIR}/.venv" ]; then
    ${PYTHON_BIN} -m venv "${REPO_DIR}/.venv"
    log "Virtual environment created: ${REPO_DIR}/.venv"
fi

# Activate the virtual environment
source "${REPO_DIR}/.venv/bin/activate"
PYTHON_BIN="${REPO_DIR}/.venv/bin/python3"
PIP_BIN="${REPO_DIR}/.venv/bin/pip"

# Base dependencies for data handling and connectivity
${PIP_BIN} install --upgrade pip --quiet
${PIP_BIN} install --quiet \
    numpy pandas ccxt pyarrow numba joblib tqdm loguru \
    python-dotenv pyyaml requests

log "Base dependencies installed"

# Training dependencies (for local evolution runs)
${PIP_BIN} install --quiet \
    torch scikit-learn scipy hmmlearn ta yfinance \
    matplotlib seaborn plotly

log "Training dependencies installed"

# SQLite / ORM (for local database)
${PIP_BIN} install --quiet sqlalchemy

log "SQLAlchemy (SQLite backend) installed"

# =============================================================================
# 3. Create directory structure
# =============================================================================
header "3. Creating Directories"

mkdir -p "${REPO_DIR}/data/cache"
mkdir -p "${REPO_DIR}/data/models/adversarial"
mkdir -p "${REPO_DIR}/data/sqlite"
mkdir -p "${REPO_DIR}/logs/training"
mkdir -p "${REPO_DIR}/logs/pm2"
mkdir -p "${REPO_DIR}/logs/sync"

log "Directories created"

# =============================================================================
# 4. Check .env configuration file
# =============================================================================
header "4. Checking .env Configuration"

ENV_FILE="${REPO_DIR}/.env"

if [ ! -f "${ENV_FILE}" ]; then
    warn ".env file missing! Creating template..."
    cat > "${ENV_FILE}" << 'ENVEOF'
# =============================================================================
# BITCOIN4Traders - Environment Variables
# WARNING: NEVER commit this file to the git repository!
# =============================================================================

# --- Telegram Bot (for signal notifications) ----------------------------------
# Create a bot via @BotFather on Telegram
TELEGRAM_BOT_TOKEN=YOUR_TELEGRAM_TOKEN_HERE
TELEGRAM_CHAT_ID=YOUR_CHAT_ID_HERE

# --- Binance API (for live data and trading) ----------------------------------
# Create API keys at https://www.binance.com/en/my/settings/api-management
# Permissions: "Spot & Margin Trading" read access, no Withdraw!
BINANCE_API_KEY=YOUR_BINANCE_API_KEY_HERE
BINANCE_API_SECRET=YOUR_BINANCE_API_SECRET_HERE

# --- Local SQLite Database ----------------------------------------------------
# Path to local SQLite file (created automatically)
SQLITE_DB_PATH=/home/hp17/Tradingbot/Quantrivo/BITCOIN4Traders/data/sqlite/trading.db

# --- GitHub (for champion sync) -----------------------------------------------
# Personal Access Token with 'repo' permission
# Create at https://github.com/settings/tokens
GITHUB_TOKEN=YOUR_GITHUB_TOKEN_HERE
GITHUB_USER=juancarlosrial76-code

# --- Local Mode ---------------------------------------------------------------
LOCAL_MASTER=true
ENVEOF
    chmod 600 "${ENV_FILE}"  # Restrict file permissions to owner only
    warn "Please fill ${ENV_FILE} with your actual credentials!"
else
    log ".env file present"
    # Check if Binance API key has been configured
    if grep -q "BINANCE_API_KEY=YOUR" "${ENV_FILE}" || grep -q "BINANCE_API_KEY=$" "${ENV_FILE}"; then
        warn "Binance API key missing in .env - only public endpoints available"
    else
        log "Binance API key: configured"
    fi
fi

# =============================================================================
# 5. Initialize SQLite database
# =============================================================================
header "5. Initializing Local SQLite Database"

SQLITE_INIT="${REPO_DIR}/data/sqlite/init_db.py"
cat > "${SQLITE_INIT}" << 'PYEOF'
#!/usr/bin/env python3
"""Initializes the local SQLite database for the Local Master."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Determine database path from environment variable or default
DB_PATH = os.environ.get(
    "SQLITE_DB_PATH",
    str(Path(__file__).parent / "trading.db")
)

# Build SQLAlchemy-compatible SQLite URL
DATABASE_URL = f"sqlite:///{DB_PATH}"

try:
    from sqlalchemy import create_engine, text
    engine = create_engine(DATABASE_URL, echo=False)

    # Create tables (mirrors src/data/database.py schema but for SQLite)
    with engine.connect() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS market_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                timestamp DATETIME NOT NULL,
                open REAL, high REAL, low REAL, close REAL, volume REAL,
                timeframe TEXT DEFAULT '1h'
            )
        """))
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_market_symbol_ts
            ON market_data (symbol, timestamp)
        """))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trade_id TEXT UNIQUE,
                symbol TEXT, side TEXT, order_type TEXT,
                quantity REAL, price REAL, total_value REAL,
                fee REAL, exchange TEXT, strategy TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS champion_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                name TEXT, strategy TEXT,
                sharpe REAL, calmar REAL, sortino REAL,
                profit_factor REAL, total_return REAL,
                source TEXT DEFAULT 'local_master'
            )
        """))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS portfolio_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                total_value REAL, cash REAL, exposure REAL,
                daily_pnl REAL, total_pnl REAL,
                sharpe_ratio REAL, max_drawdown REAL
            )
        """))
        conn.commit()

    print(f"SQLite database initialized: {DB_PATH}")
    print(f"Tables: market_data, trades, champion_history, portfolio_snapshots")

except Exception as e:
    print(f"ERROR: {e}")
    sys.exit(1)
PYEOF

${PYTHON_BIN} "${SQLITE_INIT}" && log "SQLite database ready" || warn "SQLite init failed (non-critical)"

# =============================================================================
# 6. Set up crontab for champion sync
# =============================================================================
header "6. Setting Up Crontab (Champion Sync every 6h)"

SYNC_SCRIPT="${REPO_DIR}/sync_champion.sh"
CRON_JOB="0 */6 * * * ${SYNC_SCRIPT} >> ${REPO_DIR}/logs/sync/champion_sync.log 2>&1"

# Check if cron job already exists to avoid duplicates
if crontab -l 2>/dev/null | grep -q "sync_champion.sh"; then
    log "Cron job already present"
else
    # Append new job to existing crontab
    (crontab -l 2>/dev/null; echo "${CRON_JOB}") | crontab -
    log "Cron job added: ${CRON_JOB}"
fi

# =============================================================================
# 7. Configure and start PM2
# =============================================================================
header "7. Configuring PM2 Processes"

# Start PM2 with the ecosystem config
cd "${REPO_DIR}"

# Start PM2 using the virtual Python interpreter
VENV_PYTHON="${REPO_DIR}/.venv/bin/python3"

pm2 start ecosystem.config.js --env production 2>/dev/null || true

# Instruct user to configure PM2 autostart (requires root for systemd)
echo ""
warn "IMPORTANT: PM2 Autostart Configuration"
echo "Run the following commands as root (pm2 startup prints the exact command):"
echo ""
echo "  pm2 startup"
echo "  sudo <COMMAND_PRINTED_BY_PM2>"
echo "  pm2 save"
echo ""

# =============================================================================
# 8. Network robustness: router restart wait script
# =============================================================================
header "8. Network Check Script"

NETCHECK_SCRIPT="${REPO_DIR}/wait_for_network.sh"
cat > "${NETCHECK_SCRIPT}" << 'NETEOF'
#!/bin/bash
# Waits until network is available (e.g., after router restart).
# Called by PM2 processes at startup.
MAX_WAIT=300  # Maximum wait: 5 minutes
INTERVAL=5    # Poll interval in seconds
ELAPSED=0
TARGET="8.8.8.8"  # Google DNS as reachability check target

echo "[$(date)] Waiting for network..."
while ! ping -c 1 -W 3 "${TARGET}" &>/dev/null; do
    if [ "${ELAPSED}" -ge "${MAX_WAIT}" ]; then
        echo "[$(date)] ERROR: Network not available after ${MAX_WAIT}s"
        exit 1
    fi
    echo "[$(date)] No network - waiting ${INTERVAL}s... (${ELAPSED}/${MAX_WAIT}s)"
    sleep "${INTERVAL}"
    ELAPSED=$((ELAPSED + INTERVAL))
done
echo "[$(date)] Network available! (after ${ELAPSED}s)"
NETEOF
chmod +x "${NETCHECK_SCRIPT}"
log "Network check script: ${NETCHECK_SCRIPT}"

# =============================================================================
# 9. Final summary report
# =============================================================================
header "Setup complete!"

echo ""
echo "┌─────────────────────────────────────────────────────┐"
echo "│           Local Master Status                        │"
echo "├─────────────────────────────────────────────────────┤"
echo "│  Role: Linux PC as Primary Bot (Local Master)        │"
echo "├─────────────────────────────────────────────────────┤"
printf "│  Python:  %-43s│\n" "$(${PYTHON_BIN} --version 2>&1)"
printf "│  PM2:     %-43s│\n" "$(pm2 --version 2>/dev/null | head -1)"
printf "│  Node:    %-43s│\n" "$(node --version 2>/dev/null)"
echo "├─────────────────────────────────────────────────────┤"
echo "│  PM2 Processes:                                      │"
echo "│    btc-signal-check  - Hourly signal check           │"
echo "│    btc-evolution     - 12h evolution training        │"
echo "│    btc-champion-sync - Git sync every 6h             │"
echo "├─────────────────────────────────────────────────────┤"
echo "│  Next Steps:                                         │"
echo "│  1. Fill .env with real API keys                     │"
echo "│  2. pm2 startup (configure autostart)                │"
echo "│  3. pm2 save                                         │"
echo "│  4. pm2 status  (check processes)                    │"
echo "│  5. pm2 logs    (live logs)                          │"
echo "│  6. htop        (CPU/RAM monitor)                    │"
echo "├─────────────────────────────────────────────────────┤"
echo "│  3-Pillar Architecture:                              │"
echo "│  [Linux PC] --> [GitHub Actions] --> [Colab GPU]     │"
echo "└─────────────────────────────────────────────────────┘"
echo ""
