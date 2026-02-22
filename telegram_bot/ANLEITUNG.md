# Telegram Bot Einrichtung — BITCOIN4Traders

Komplette Schritt-für-Schritt Anleitung.
Dauer: ca. 5 Minuten. Kosten: 0,00 €.

---

## SCHRITT 1 — Telegram installieren

Falls noch nicht vorhanden:
- **Handy:** App Store / Play Store → "Telegram" suchen → installieren
- **PC:** https://desktop.telegram.org → herunterladen und installieren

Konto erstellen mit deiner Handynummer (falls noch kein Konto vorhanden).

---

## SCHRITT 2 — Neuen Bot erstellen (via @BotFather)

1. Telegram öffnen
2. Oben in der Suchleiste **@BotFather** eintippen und auswählen
   (blaues Häkchen = offizieller Bot von Telegram)
3. Auf **START** klicken
4. Schreibe: `/newbot`
5. BotFather fragt: **"Wie soll dein Bot heißen?"**
   Antworte z.B.: `BITCOIN4Traders Signal Bot`
6. BotFather fragt: **"Wie soll der Username sein?"** (muss auf `bot` enden)
   Antworte z.B.: `bitcoin4traders_signal_bot`
7. BotFather antwortet mit deinem **Token**, z.B.:
   ```
   7123456789:AAHdqTcvCH1vGWJxfSeofSs0K67lxyFouHo
   ```
   → Diesen Token kopieren und sicher aufbewahren (wie ein Passwort)

---

## SCHRITT 3 — Deine Chat-ID herausfinden

Du brauchst die Chat-ID damit der Bot weiß, wohin er Nachrichten schicken soll.

**Methode A — via @userinfobot (einfachste Methode):**
1. Suche in Telegram nach **@userinfobot**
2. Klick **START**
3. Der Bot antwortet sofort mit deiner ID, z.B.:
   ```
   Id: 123456789
   ```
4. Diese Zahl ist deine Chat-ID → kopieren

**Methode B — via Browser (falls Methode A nicht klappt):**
1. Deinen eigenen Bot (aus Schritt 2) öffnen und einmal **/start** schicken
2. Im Browser diese URL aufrufen (Token einsetzen):
   ```
   https://api.telegram.org/bot<DEIN_TOKEN>/getUpdates
   ```
   Beispiel:
   ```
   https://api.telegram.org/bot7123456789:AAHdqTcvCH1vGWJxfSeofSs0K67lxyFouHo/getUpdates
   ```
3. Im JSON nach `"chat":{"id":` suchen → die Zahl dahinter ist deine Chat-ID

---

## SCHRITT 4 — Bot testen

Ersetze TOKEN und CHAT_ID und öffne diese URL im Browser:

```
https://api.telegram.org/bot<TOKEN>/sendMessage?chat_id=<CHAT_ID>&text=Hallo+BITCOIN4Traders!
```

Wenn auf deinem Handy eine Nachricht von deinem Bot erscheint → alles funktioniert.

---

## SCHRITT 5 — Secrets in GitHub hinterlegen

1. Gehe zu:
   ```
   https://github.com/juancarlosrial76-code/BITCOIN4Traders/settings/secrets/actions
   ```
2. Klick **New repository secret** — füge nacheinander hinzu:

   | Name | Wert |
   |------|------|
   | `TELEGRAM_BOT_TOKEN` | `7123456789:AAHdqTcvCH1vGWJxfSeofSs0K67lxyFouHo` |
   | `TELEGRAM_CHAT_ID` | `123456789` |

3. Jeweils auf **Add secret** klicken

---

## SCHRITT 6 — Lokale .env Datei anlegen (für Tests auf dem Server)

Erstelle die Datei `/home/hp17/Tradingbot/Quantrivo/BITCOIN4Traders/.env`:

```
TELEGRAM_BOT_TOKEN=7123456789:AAHdqTcvCH1vGWJxfSeofSs0K67lxyFouHo
TELEGRAM_CHAT_ID=123456789
```

**Wichtig:** Diese Datei ist bereits in `.gitignore` eingetragen und wird
nie auf GitHub hochgeladen.

---

## SCHRITT 7 — Test aus Python

```python
from darwin_engine import TelegramNotifier

# Aus .env laden (einmalig am Anfang des Skripts)
from dotenv import load_dotenv
load_dotenv()

notifier = TelegramNotifier.from_env()
notifier.send("Test erfolgreich! BITCOIN4Traders Bot ist aktiv.")
```

Wenn die Nachricht ankommt — fertig. Der Bot ist einsatzbereit.

---

## Nachrichten die der Bot automatisch schickt

| Ereignis | Nachricht |
|----------|-----------|
| Stündlicher Signal-Check (GitHub) | `LONG / SHORT / FLAT` mit Preis und Uhrzeit |
| Neuer Champion nach Evolution (Colab) | Name, MV-Score, Survival-Rate, Worst-DD |
| Heartbeat-Ausfall (>90 Min kein Signal) | Warnung: "GitHub Actions ausgefallen" |
| Fehler bei Datenabruf | Alert mit Fehlermeldung |

---

## Optionaler Schritt — Gruppe oder Kanal statt privater Chat

Falls du die Signale in eine Telegram-Gruppe oder einen Kanal senden willst:

1. Gruppe/Kanal erstellen
2. Deinen Bot als **Admin** zur Gruppe hinzufügen
3. Eine Nachricht in die Gruppe schreiben
4. Chat-ID der Gruppe herausfinden via:
   ```
   https://api.telegram.org/bot<TOKEN>/getUpdates
   ```
   Gruppen-IDs beginnen mit einem Minus-Zeichen, z.B. `-1001234567890`
5. Diese Gruppen-ID als `TELEGRAM_CHAT_ID` verwenden

---

## Zusammenfassung der Werte die du brauchst

```
TELEGRAM_BOT_TOKEN  =  <von @BotFather>      z.B. 7123456789:AAH...
TELEGRAM_CHAT_ID    =  <von @userinfobot>    z.B. 123456789
```

Diese beiden Werte an zwei Stellen eintragen:
1. GitHub Secrets (für automatischen Betrieb)
2. Lokale .env Datei (für manuelle Tests)
