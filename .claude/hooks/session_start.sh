#!/bin/bash
# Session-Start Hook: Erstellt automatisch eine neue Session-Datei
# bei jedem Start von Claude Code / OpenCode

DOKU_DIR="/home/hp17/Tradingbot/BITCOIN4Traders-DE-DOKU"
SESSIONS_DIR="$DOKU_DIR/sessions"

# Sicherstellen dass das Verzeichnis existiert
mkdir -p "$SESSIONS_DIR"

# Datum und Zeit fuer den Dateinamen
TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
DATE_READABLE=$(date +"%d.%m.%Y %H:%M:%S")

# Session-ID aus dem Input lesen
SESSION_INPUT=$(cat)
SESSION_ID=$(echo "$SESSION_INPUT" | jq -r '.session_id // "unknown"' 2>/dev/null || echo "unknown")
SOURCE=$(echo "$SESSION_INPUT" | jq -r '.source // "startup"' 2>/dev/null || echo "startup")
WORKDIR=$(echo "$SESSION_INPUT" | jq -r '.cwd // "unknown"' 2>/dev/null || echo "unknown")

# Dateiname erstellen
SESSION_FILE="$SESSIONS_DIR/session_${TIMESTAMP}.md"

# Session-Datei erstellen
cat > "$SESSION_FILE" << EOF
# OpenCode Session - $DATE_READABLE

## Session Info
- **Session ID:** $SESSION_ID
- **Gestartet:** $DATE_READABLE
- **Quelle:** $SOURCE
- **Arbeitsverzeichnis:** $WORKDIR
- **Projekt:** BITCOIN4Traders

## Kontext
Diese Datei wurde automatisch beim Start von OpenCode erstellt.

---

## Gesprach / Notizen

<!-- Hier werden Gesprache und wichtige Notizen aus dieser Session gespeichert -->

EOF

# Transcript-Pfad speichern fuer spaetere Nutzung
TRANSCRIPT_PATH=$(echo "$SESSION_INPUT" | jq -r '.transcript_path // ""' 2>/dev/null || echo "")
if [ -n "$TRANSCRIPT_PATH" ]; then
    echo "- **Transcript:** \`$TRANSCRIPT_PATH\`" >> "$SESSION_FILE"
fi

# In globales Log eintragen
echo "$DATE_READABLE | Session: $SESSION_ID | $SOURCE | $WORKDIR" >> "$SESSIONS_DIR/session_log.txt"

# Rueckmeldung an Claude (wird als Kontext angezeigt)
jq -n \
  --arg ctx "Session-Datei wurde erstellt: $SESSION_FILE. Bitte dokumentiere wichtige Punkte dieses Gesprachs in dieser Datei am Ende der Session." \
  '{
    hookSpecificOutput: {
      hookEventName: "SessionStart",
      additionalContext: $ctx
    }
  }'

exit 0
