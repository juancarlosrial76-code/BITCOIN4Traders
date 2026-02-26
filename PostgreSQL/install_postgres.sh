#!/bin/bash

# ============================================================
# PostgreSQL Installation Script for Linux (Ubuntu/Debian)
# ============================================================

set -e

# Colours for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Configuration – adjust here
DB_NAME="myapp_db"
DB_USER="myapp_user"
DB_PASSWORD="$(openssl rand -base64 16 | tr -dc 'a-zA-Z0-9' | head -c 20)"
DB_PORT=5432

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  PostgreSQL Installation & Setup${NC}"
echo -e "${GREEN}========================================${NC}"

# -----------------------------------------------
# 1. Update system packages
# -----------------------------------------------
echo -e "\n${YELLOW}[1/6] Updating system packages...${NC}"
sudo apt-get update -qq

# -----------------------------------------------
# 2. Install PostgreSQL
# -----------------------------------------------
echo -e "\n${YELLOW}[2/6] Installing PostgreSQL...${NC}"
sudo apt-get install -y postgresql postgresql-contrib

# -----------------------------------------------
# 3. Start & enable PostgreSQL service
# -----------------------------------------------
echo -e "\n${YELLOW}[3/6] Starting and enabling service...${NC}"
sudo systemctl enable postgresql
sudo systemctl start postgresql

# Wait briefly until PostgreSQL is ready
sleep 2

# -----------------------------------------------
# 4. Create database & user
# -----------------------------------------------
echo -e "\n${YELLOW}[4/6] Creating database and user...${NC}"

sudo -u postgres psql <<EOF
-- Create user (if not already exists)
DO \$\$
BEGIN
  IF NOT EXISTS (SELECT FROM pg_catalog.pg_user WHERE usename = '${DB_USER}') THEN
    CREATE USER ${DB_USER} WITH PASSWORD '${DB_PASSWORD}';
  ELSE
    ALTER USER ${DB_USER} WITH PASSWORD '${DB_PASSWORD}';
  END IF;
END
\$\$;

-- Create database (if not already exists)
SELECT 'CREATE DATABASE ${DB_NAME} OWNER ${DB_USER}'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = '${DB_NAME}')\gexec

-- Grant privileges
GRANT ALL PRIVILEGES ON DATABASE ${DB_NAME} TO ${DB_USER};
EOF

# -----------------------------------------------
# 5. pg_hba.conf – allow local connections
# -----------------------------------------------
echo -e "\n${YELLOW}[5/6] Adjusting configuration (pg_hba.conf)...${NC}"

PG_VERSION=$(psql --version | awk '{print $3}' | cut -d. -f1)
PG_HBA="/etc/postgresql/${PG_VERSION}/main/pg_hba.conf"

# Check if entry already exists
if ! sudo grep -q "host.*${DB_NAME}.*${DB_USER}" "$PG_HBA"; then
  echo "host    ${DB_NAME}    ${DB_USER}    127.0.0.1/32    md5" | sudo tee -a "$PG_HBA" > /dev/null
  echo "host    ${DB_NAME}    ${DB_USER}    ::1/128         md5" | sudo tee -a "$PG_HBA" > /dev/null
  sudo systemctl reload postgresql
fi

# -----------------------------------------------
# 6. Test connection
# -----------------------------------------------
echo -e "\n${YELLOW}[6/6] Testing connection...${NC}"

export PGPASSWORD="${DB_PASSWORD}"
if psql -h 127.0.0.1 -U "${DB_USER}" -d "${DB_NAME}" -c "SELECT version();" > /dev/null 2>&1; then
  echo -e "${GREEN}✓ Connection successful!${NC}"
else
  echo -e "${RED}✗ Connection test failed. Check logs: sudo journalctl -u postgresql${NC}"
  exit 1
fi
unset PGPASSWORD

# -----------------------------------------------
# Print & save connection details
# -----------------------------------------------
CONN_STRING="postgresql://${DB_USER}:${DB_PASSWORD}@127.0.0.1:${DB_PORT}/${DB_NAME}"

echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}  ✓ Installation complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "  Host:      ${YELLOW}127.0.0.1${NC}"
echo -e "  Port:      ${YELLOW}${DB_PORT}${NC}"
echo -e "  Database:  ${YELLOW}${DB_NAME}${NC}"
echo -e "  User:      ${YELLOW}${DB_USER}${NC}"
echo -e "  Password:  ${YELLOW}${DB_PASSWORD}${NC}"
echo ""
echo -e "  Connection String:"
echo -e "  ${YELLOW}${CONN_STRING}${NC}"
echo ""

# Save connection details to file
CREDS_FILE="$(pwd)/postgres_credentials.txt"
cat > "$CREDS_FILE" <<CREDS
# PostgreSQL connection details
# Created: $(date)

HOST=127.0.0.1
PORT=${DB_PORT}
DATABASE=${DB_NAME}
USER=${DB_USER}
PASSWORD=${DB_PASSWORD}

CONNECTION_STRING=${CONN_STRING}

# psql command:
# PGPASSWORD=${DB_PASSWORD} psql -h 127.0.0.1 -U ${DB_USER} -d ${DB_NAME}
CREDS

chmod 600 "$CREDS_FILE"
echo -e "  Connection details saved to: ${YELLOW}${CREDS_FILE}${NC}"
echo -e "${GREEN}========================================${NC}"
