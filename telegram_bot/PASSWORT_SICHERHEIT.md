# Sichere Passwort- und Token-Verwaltung

Professionelle Lösung für Linux/Ubuntu — lokal, verschlüsselt, kostenlos.

---

## OPTION A — `pass` (Empfohlen für dieses Projekt)

`pass` ist der Standard-Passwortmanager für Linux.
Verschlüsselt alles mit GPG auf deiner Festplatte. Kein Cloud-Zwang.

### Installation

```bash
sudo apt install pass
```

### Einmalige Einrichtung (GPG-Schlüssel + pass init)

```bash
# 1. GPG-Schlüssel erstellen (einmalig)
gpg --full-generate-key
```

Wähle:
- Key-Typ: `(1) RSA and RSA`
- Key-Größe: `4096`
- Gültigkeit: `0` (kein Ablauf)
- Name: dein Name
- E-Mail: deine E-Mail
- Passwort: starkes Masterpasswort (das einzige das du dir merken musst)

```bash
# 2. GPG Key-ID herausfinden
gpg --list-keys
# Zeigt z.B.:
# pub   rsa4096 2026-02-22 [SC]
#       AB12CD34EF56GH78IJ90KL12MN34OP56QR78ST90   <-- das ist die Key-ID
# uid   [ultimate] Dein Name <deine@email.de>

# 3. pass mit diesem Schlüssel initialisieren
pass init AB12CD34EF56GH78IJ90KL12MN34OP56QR78ST90
```

### Tokens und Passwörter speichern

```bash
# Telegram Bot Token speichern
pass insert bitcoin4traders/telegram/bot_token
# -> Eingabe: dein Token  (wird nicht angezeigt)

# Telegram Chat-ID speichern
pass insert bitcoin4traders/telegram/chat_id
# -> Eingabe: deine Chat-ID

# Binance API Key speichern
pass insert bitcoin4traders/binance/api_key
pass insert bitcoin4traders/binance/api_secret

# GitHub Personal Access Token
pass insert bitcoin4traders/github/pat
```

### Tokens abrufen

```bash
# Anzeigen (öffnet kurz ein Fenster für GPG-Passwort)
pass bitcoin4traders/telegram/bot_token

# Direkt in Umgebungsvariable laden
export TELEGRAM_BOT_TOKEN=$(pass bitcoin4traders/telegram/bot_token)
export TELEGRAM_CHAT_ID=$(pass bitcoin4traders/telegram/chat_id)

# Dann Python starten
python3 telegram_bot/test_telegram.py
```

### Alle gespeicherten Einträge anzeigen

```bash
pass
# Zeigt die Struktur:
# Password Store
# └── bitcoin4traders
#     ├── binance
#     │   ├── api_key
#     │   └── api_secret
#     ├── github
#     │   └── pat
#     └── telegram
#         ├── bot_token
#         └── chat_id
```

### Backup des verschlüsselten Stores

```bash
# Der gesamte Store liegt hier:
ls ~/.password-store/

# Backup auf externe Festplatte:
cp -r ~/.password-store/ /media/usb-stick/password-store-backup/

# Oder komprimiert:
tar czf password-store-backup.tar.gz ~/.password-store/
```

---

## OPTION B — `.env` Datei (Einfach, für Entwicklung)

Für schnelle lokale Tests ohne pass-Installation.

```bash
# Datei erstellen
nano /home/hp17/Tradingbot/Quantrivo/BITCOIN4Traders/.env
```

Inhalt:
```
TELEGRAM_BOT_TOKEN=7123456789:AAHdqTcvCH1vGWJxfSeofSs0K67lxyFouHo
TELEGRAM_CHAT_ID=123456789
BINANCE_API_KEY=dein_api_key
BINANCE_API_SECRET=dein_api_secret
```

Datei absichern (nur du kannst sie lesen):
```bash
chmod 600 /home/hp17/Tradingbot/Quantrivo/BITCOIN4Traders/.env
```

**Wichtig:** `.env` ist in `.gitignore` eingetragen → wird NIE auf GitHub hochgeladen.

Prüfen:
```bash
grep ".env" /home/hp17/Tradingbot/Quantrivo/BITCOIN4Traders/.gitignore
```

---

## OPTION C — KeePassXC (Grafische Oberfläche)

Falls du eine App mit GUI bevorzugst.

```bash
sudo apt install keepassxc
```

- Erstelle eine `.kdbx` Datenbank mit Masterpasswort
- Speichere alle Tokens darin
- Für Skripte: KeePassXC hat eine CLI (`keepassxc-cli`)

```bash
# Token aus KeePassXC in Variable laden
export TELEGRAM_BOT_TOKEN=$(keepassxc-cli show -a Password vault.kdbx "bitcoin4traders/telegram_token")
```

---

## Was NIEMALS tun

| Verboten | Warum |
|----------|-------|
| Token direkt in `.py` Datei schreiben | Landet im Git-Verlauf, für immer sichtbar |
| Token in Chat/E-Mail schreiben | Kann abgefangen werden |
| Token in README oder Kommentaren | Öffentlich auf GitHub |
| Dasselbe Passwort mehrfach verwenden | Ein Leak kompromittiert alles |
| Tokens ohne Ablaufdatum erstellen | Bei Diebstahl ewig gültig |

---

## Empfehlung für dieses Projekt

```
Alltag (Entwicklung)  →  .env Datei  (chmod 600)
Langzeit-Archiv       →  pass        (GPG-verschlüsselt)
GitHub Actions        →  GitHub Secrets (eingebaut, sicher)
```

Alle drei Methoden sind bereits im Projekt integriert:
- `.env` → wird von `dotenv` in `test_telegram.py` geladen
- `pass` → Tokens per `$(pass ...)` als env vars exportieren
- GitHub Secrets → werden im Workflow als `${{ secrets.NAME }}` eingesetzt
