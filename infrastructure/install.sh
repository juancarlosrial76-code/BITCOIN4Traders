#!/bin/bash
# =============================================================================
# install.sh - Set up infrastructure (run once)
# =============================================================================
# Installs all dependencies and configures services.
# Cost: 0 EUR
#
# Prerequisites:
#   - credentials.json in config/ (from Google Cloud Console)
#   - GitHub remote already configured
#
# Usage:
#   chmod +x infrastructure/install.sh
#   ./infrastructure/install.sh
# =============================================================================

set -e

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INFRA_DIR="${REPO_DIR}/infrastructure"

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

ok()     { echo -e "${GREEN}[OK]${NC} $*"; }
warn()   { echo -e "${YELLOW}[WARN]${NC} $*"; }
err()    { echo -e "${RED}[ERROR]${NC} $*"; }
header() { echo -e "\n${BLUE}━━━ $* ━━━${NC}"; }

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  BITCOIN4Traders - Infrastructure Setup"
echo "  Zero-Cost Cloud Infrastructure"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# ─── 1. Python packages ────────────────────────────────────────────────────────
header "1. Install Python packages"

pip3 install --quiet --upgrade \
    google-api-python-client \
    google-auth \
    google-auth-httplib2 \
    flask \
    requests

ok "Python packages installed (google-api, flask, requests)"

# ─── 2. Directories ────────────────────────────────────────────────────────────
header "2. Create directories"

mkdir -p "${REPO_DIR}/logs/alerts"
mkdir -p "${REPO_DIR}/logs/infrastructure"
mkdir -p "${REPO_DIR}/config"
mkdir -p "${REPO_DIR}/data/cache"

ok "Directories created"

# ─── 3. Check credentials.json ──────────────────────────────────────────────────
header "3. Check Google credentials"

CREDS="${REPO_DIR}/config/gdrive_credentials.json"
if [ -f "${CREDS}" ]; then
    ok "credentials.json found: ${CREDS}"
else
    warn "credentials.json NOT found!"
    echo ""
    echo "  How to get the file:"
    echo "  1. https://console.cloud.google.com"
    echo "  2. Create project (or select existing one)"
    echo "  3. APIs & Services -> Library -> Enable 'Google Drive API'"
    echo "  4. IAM & Admin -> Service Accounts -> New Service Account"
    echo "  5. Create key (JSON) -> download"
    echo "  6. Rename file to: ${CREDS}"
    echo ""
    echo "  Then share the Drive folder:"
    echo "  - Create folder in Drive"
    echo "  - Right-click -> Share -> Enter service account email"
    echo ""
fi

# ─── 4. Drive configuration ──────────────────────────────────────────────────────
header "4. Configure Google Drive"

DRIVE_CONFIG="${REPO_DIR}/config/drive_config.json"
if [ -f "${DRIVE_CONFIG}" ]; then
    ok "drive_config.json already exists"
else
    if [ -f "${CREDS}" ]; then
        python3 "${INFRA_DIR}/drive/drive_manager.py" setup
    else
        warn "Skipped (credentials.json missing)"
    fi
fi

# ─── 5. Alert configuration ──────────────────────────────────────────────────────
header "5. Configure Telegram alerts"

ALERT_CONFIG="${REPO_DIR}/config/alert_config.json"
if [ -f "${ALERT_CONFIG}" ]; then
    ok "alert_config.json already exists"
else
    read -r -p "Configure Telegram bot now? [y/N] " answer
    if [[ "${answer}" =~ ^[yY]$ ]]; then
        python3 "${INFRA_DIR}/monitor/alert_manager.py" setup
    else
        warn "Skipped - later: python3 infrastructure/monitor/alert_manager.py setup"
    fi
fi

# ─── 6. Systemd service ──────────────────────────────────────────────────────────
header "6. Install systemd service"

SERVICE_SRC="${INFRA_DIR}/systemd/bitcoin4traders.service"
SERVICE_DST="/etc/systemd/system/bitcoin4traders.service"

if [ -f "${SERVICE_SRC}" ]; then
    read -r -p "Install as systemd service (persistent operation)? [y/N] " answer
    if [[ "${answer}" =~ ^[yY]$ ]]; then
        sudo cp "${SERVICE_SRC}" "${SERVICE_DST}"
        sudo systemctl daemon-reload
        sudo systemctl enable bitcoin4traders
        sudo systemctl start bitcoin4traders
        ok "Service installed and started"
        echo "  Status: sudo systemctl status bitcoin4traders"
        echo "  Logs:   journalctl -u bitcoin4traders -f"
    else
        warn "Skipped - start manually: python3 infrastructure/master.py"
    fi
fi

# ─── 7. Update .gitignore ─────────────────────────────────────────────────────────
header "7. Update .gitignore (protect secrets)"

GITIGNORE="${REPO_DIR}/.gitignore"
SECRETS_ENTRIES=(
    "config/gdrive_credentials.json"
    "config/alert_config.json"
    "config/drive_config.json"
    "logs/alerts/"
    "logs/infrastructure/"
    "/tmp/"
)

for entry in "${SECRETS_ENTRIES[@]}"; do
    if ! grep -qF "${entry}" "${GITIGNORE}" 2>/dev/null; then
        echo "${entry}" >> "${GITIGNORE}"
        ok "gitignore: ${entry}"
    fi
done

# ─── Completion ───────────────────────────────────────────────────────────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Setup complete!"
echo ""
echo "  Next steps:"
if [ ! -f "${CREDS}" ]; then
    echo "  1. Place credentials.json in config/"
    echo "  2. python3 infrastructure/drive/drive_manager.py setup"
    echo "  3. python3 infrastructure/monitor/alert_manager.py setup"
fi
echo ""
  echo "  Generate API token for Colab:"
  echo "    python3 infrastructure/monitor/listener.py setup"
  echo ""
  echo "  Start manually:"
  echo "    python3 infrastructure/master.py"
  echo ""
  echo "  As service:"
  echo "    sudo systemctl start bitcoin4traders"
  echo "    journalctl -u bitcoin4traders -f"
  echo ""
  echo "  Configure Colab notebook:"
  echo "    LINUX_IP     = 'your-ip-or-ngrok-url'"
  echo "    LINUX_API_TOKEN = 'token-from-listener-setup'"
  echo "    DRIVE_FOLDER_ID = 'folder-id-from-drive'"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
