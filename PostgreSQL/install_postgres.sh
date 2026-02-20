#!/bin/bash

# ============================================================
# PostgreSQL Installation Script für Linux (Ubuntu/Debian)
# ============================================================

set -e

# Farben für Ausgabe
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Konfiguration – hier anpassen
DB_NAME="myapp_db"
DB_USER="myapp_user"
DB_PASSWORD="$(openssl rand -base64 16 | tr -dc 'a-zA-Z0-9' | head -c 20)"
DB_PORT=5432

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  PostgreSQL Installation & Setup${NC}"
echo -e "${GREEN}========================================${NC}"

# -----------------------------------------------
# 1. System-Pakete aktualisieren
# -----------------------------------------------
echo -e "\n${YELLOW}[1/6] System-Pakete aktualisieren...${NC}"
sudo apt-get update -qq

# -----------------------------------------------
# 2. PostgreSQL installieren
# -----------------------------------------------
echo -e "\n${YELLOW}[2/6] PostgreSQL installieren...${NC}"
sudo apt-get install -y postgresql postgresql-contrib

# -----------------------------------------------
# 3. PostgreSQL-Dienst starten & aktivieren
# -----------------------------------------------
echo -e "\n${YELLOW}[3/6] Dienst starten und aktivieren...${NC}"
sudo systemctl enable postgresql
sudo systemctl start postgresql

# Kurz warten bis PostgreSQL bereit ist
sleep 2

# -----------------------------------------------
# 4. Datenbank & Benutzer anlegen
# -----------------------------------------------
echo -e "\n${YELLOW}[4/6] Datenbank und Benutzer anlegen...${NC}"

sudo -u postgres psql <<EOF
-- Benutzer anlegen (falls noch nicht vorhanden)
DO \$\$
BEGIN
  IF NOT EXISTS (SELECT FROM pg_catalog.pg_user WHERE usename = '${DB_USER}') THEN
    CREATE USER ${DB_USER} WITH PASSWORD '${DB_PASSWORD}';
  ELSE
    ALTER USER ${DB_USER} WITH PASSWORD '${DB_PASSWORD}';
  END IF;
END
\$\$;

-- Datenbank anlegen (falls noch nicht vorhanden)
SELECT 'CREATE DATABASE ${DB_NAME} OWNER ${DB_USER}'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = '${DB_NAME}')\gexec

-- Berechtigungen vergeben
GRANT ALL PRIVILEGES ON DATABASE ${DB_NAME} TO ${DB_USER};
EOF

# -----------------------------------------------
# 5. pg_hba.conf – lokale Verbindung erlauben
# -----------------------------------------------
echo -e "\n${YELLOW}[5/6] Konfiguration anpassen (pg_hba.conf)...${NC}"

PG_VERSION=$(psql --version | awk '{print $3}' | cut -d. -f1)
PG_HBA="/etc/postgresql/${PG_VERSION}/main/pg_hba.conf"

# Prüfen ob Eintrag bereits existiert
if ! sudo grep -q "host.*${DB_NAME}.*${DB_USER}" "$PG_HBA"; then
  echo "host    ${DB_NAME}    ${DB_USER}    127.0.0.1/32    md5" | sudo tee -a "$PG_HBA" > /dev/null
  echo "host    ${DB_NAME}    ${DB_USER}    ::1/128         md5" | sudo tee -a "$PG_HBA" > /dev/null
  sudo systemctl reload postgresql
fi

# -----------------------------------------------
# 6. Verbindung testen
# -----------------------------------------------
echo -e "\n${YELLOW}[6/6] Verbindung testen...${NC}"

export PGPASSWORD="${DB_PASSWORD}"
if psql -h 127.0.0.1 -U "${DB_USER}" -d "${DB_NAME}" -c "SELECT version();" > /dev/null 2>&1; then
  echo -e "${GREEN}✓ Verbindung erfolgreich!${NC}"
else
  echo -e "${RED}✗ Verbindungstest fehlgeschlagen. Bitte Logs prüfen: sudo journalctl -u postgresql${NC}"
  exit 1
fi
unset PGPASSWORD

# -----------------------------------------------
# Verbindungsdaten ausgeben & speichern
# -----------------------------------------------
CONN_STRING="postgresql://${DB_USER}:${DB_PASSWORD}@127.0.0.1:${DB_PORT}/${DB_NAME}"

echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}  ✓ Installation abgeschlossen!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "  Host:         ${YELLOW}127.0.0.1${NC}"
echo -e "  Port:         ${YELLOW}${DB_PORT}${NC}"
echo -e "  Datenbank:    ${YELLOW}${DB_NAME}${NC}"
echo -e "  Benutzer:     ${YELLOW}${DB_USER}${NC}"
echo -e "  Passwort:     ${YELLOW}${DB_PASSWORD}${NC}"
echo ""
echo -e "  Connection String:"
echo -e "  ${YELLOW}${CONN_STRING}${NC}"
echo ""

# Verbindungsdaten in Datei speichern
CREDS_FILE="$(pwd)/postgres_credentials.txt"
cat > "$CREDS_FILE" <<CREDS
# PostgreSQL Verbindungsdaten
# Erstellt am: $(date)

HOST=127.0.0.1
PORT=${DB_PORT}
DATABASE=${DB_NAME}
USER=${DB_USER}
PASSWORD=${DB_PASSWORD}

CONNECTION_STRING=${CONN_STRING}

# psql Befehl:
# PGPASSWORD=${DB_PASSWORD} psql -h 127.0.0.1 -U ${DB_USER} -d ${DB_NAME}
CREDS

chmod 600 "$CREDS_FILE"
echo -e "  Verbindungsdaten gespeichert in: ${YELLOW}${CREDS_FILE}${NC}"
echo -e "${GREEN}========================================${NC}"
