#!/bin/bash
# Session-End Hook: Speichert den Transcript am Ende der Session

DOKU_DIR="/home/hp17/Tradingbot/BITCOIN4Traders-DE-DOKU"
SESSIONS_DIR="$DOKU_DIR/sessions"

mkdir -p "$SESSIONS_DIR"

# Input lesen
SESSION_INPUT=$(cat)
TRANSCRIPT_PATH=$(echo "$SESSION_INPUT" | jq -r '.transcript_path // ""' 2>/dev/null || echo "")
SESSION_ID=$(echo "$SESSION_INPUT" | jq -r '.session_id // "unknown"' 2>/dev/null || echo "unknown")

# Aktuelle Session-Datei finden (die neueste)
LATEST_SESSION=$(ls -t "$SESSIONS_DIR"/session_*.md 2>/dev/null | head -1)

if [ -n "$TRANSCRIPT_PATH" ] && [ -f "$TRANSCRIPT_PATH" ] && [ -n "$LATEST_SESSION" ]; then
    # Transcript auslesen und in Markdown konvertieren
    echo "" >> "$LATEST_SESSION"
    echo "---" >> "$LATEST_SESSION"
    echo "" >> "$LATEST_SESSION"
    echo "## Session Ende: $(date +"%d.%m.%Y %H:%M:%S")" >> "$LATEST_SESSION"
    echo "" >> "$LATEST_SESSION"
    echo "### Gesprach-Zusammenfassung (aus Transcript)" >> "$LATEST_SESSION"
    echo "" >> "$LATEST_SESSION"
    
    # Nachrichten aus dem Transcript extrahieren
    if command -v jq &>/dev/null; then
        # Versuche JSONL zu lesen (eine JSON-Zeile pro Eintrag)
        while IFS= read -r line; do
            ROLE=$(echo "$line" | jq -r '.role // ""' 2>/dev/null)
            CONTENT=$(echo "$line" | jq -r '.content // ""' 2>/dev/null)
            TYPE=$(echo "$line" | jq -r '.type // ""' 2>/dev/null)
            
            if [ "$TYPE" = "user" ] || [ "$ROLE" = "user" ]; then
                echo "**User:** $CONTENT" >> "$LATEST_SESSION"
                echo "" >> "$LATEST_SESSION"
            elif [ "$TYPE" = "assistant" ] || [ "$ROLE" = "assistant" ]; then
                # Nur Text-Inhalte (keine Tool-Calls)
                TEXT=$(echo "$line" | jq -r '.content | if type == "array" then .[] | select(.type == "text") | .text else . end' 2>/dev/null | head -5)
                if [ -n "$TEXT" ]; then
                    echo "**Claude:** $TEXT" >> "$LATEST_SESSION"
                    echo "" >> "$LATEST_SESSION"
                fi
            fi
        done < "$TRANSCRIPT_PATH"
    fi
    
    echo "" >> "$LATEST_SESSION"
    echo "*(Vollstaendiger Transcript: \`$TRANSCRIPT_PATH\`)*" >> "$LATEST_SESSION"
fi

exit 0
