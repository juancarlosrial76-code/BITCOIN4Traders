# BITCOIN4Traders — Transport-Guide
## Kommunikation Lokal ↔ Colab/Cloud

---

## Vergleichstabelle

| | Option 1: Redis | Option 2: Telegram | Option 3: Google Drive | Option 4: Ably |
|---|---|---|---|---|
| **Latenz** | 30–150 ms | 200–800 ms | 2–15 Sekunden | 50–150 ms |
| **Kosten** | $0 | $0 | $0 | $0 (Free Tier) |
| **Externer Account** | Nein | Telegram (vorhanden) | Google (vorhanden) | Ably (neu) |
| **Setup-Aufwand** | Mittel | Gering | Gering | Gering |
| **Infrastruktur lokal** | Redis + CF Tunnel | Nichts | drive_manager.py | Nichts |
| **Für Timeframe** | 1m–1h | 1h+ | 1h+ | 1m–1h |
| **Zuverlässigkeit** | Sehr hoch | Sehr hoch | Hoch | Sehr hoch |
| **Offline-Puffer** | Ja (Redis List) | Nein | Ja (Datei) | Ja (100 Msgs) |
| **Menschenlesbar** | Nein | Ja | Ja (JSON) | Nein |
| **Datei** | `transport_redis.py` | `transport_telegram.py` | `transport_gdrive.py` | `transport_ably.py` |

---

## Option 1: Redis + Cloudflare Tunnel
**Empfohlen für: Sub-Sekunden bis 1m Timeframes**

### Wie es funktioniert
```
Exchange API → Lokal (Redis) → HTTP-Proxy :8766 → Cloudflare Tunnel → Colab (httpx poll)
Colab (Inferenz) → HTTP-Proxy → Redis PUBLISH → Lokal (Callback)
```

Redis wird lokal betrieben. Colab spricht via HTTP mit einem FastAPI-Proxy,
der von Cloudflare Tunnel erreichbar gemacht wird. Kein Redis-Client in Colab nötig.

### Einrichtung

**1. Redis installieren (lokal, einmalig):**
```bash
sudo apt install redis-server    # Ubuntu/Debian
# oder
brew install redis               # macOS

redis-server --daemonize yes     # Starten
redis-cli ping                   # Test → PONG
```

**2. Cloudflare Tunnel installieren:**
```bash
# Ubuntu/Debian:
curl -fsSL https://pkg.cloudflare.com/cloudflare-main.gpg | \
  sudo tee /usr/share/keyrings/cloudflare-main.gpg > /dev/null
echo 'deb [signed-by=/usr/share/keyrings/cloudflare-main.gpg] \
  https://pkg.cloudflare.com/cloudflared any main' | \
  sudo tee /etc/apt/sources.list.d/cloudflared.list
sudo apt-get update && sudo apt-get install cloudflared

# Oder direkt:
wget -O cloudflared https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64
chmod +x cloudflared
```

**3. Python-Pakete:**
```bash
pip install redis fastapi uvicorn httpx
```

**4. Starten:**
```bash
# Terminal 1 — Redis
redis-server

# Terminal 2 — Cloudflare Tunnel
./cloudflared tunnel --url http://localhost:8766
# Gibt aus: https://abc-xyz.trycloudflare.com  ← diese URL merken!

# Terminal 3 — Module A (lokal) mit Redis Transport
python colab_bridge/module_a_local.py --transport redis
```

**5. In Colab:**
```python
!pip install httpx

import sys
sys.path.insert(0, '/content/drive/MyDrive/BITCOIN4Traders')

from colab_bridge.transports import get_transport
transport = get_transport("redis", side="colab",
                          proxy_url="https://abc-xyz.trycloudflare.com")

from colab_bridge.module_b_colab import ModuleB
engine = ModuleB(transport=transport, model_path="...")
await engine.run()
```

### Latenz-Details
- Redis PUBLISH lokal: ~1ms
- Cloudflare Tunnel Overhead: ~30–100ms (je nach Standort)
- Colab HTTP-Poll Intervall: 1s (einstellbar)
- **Effektive Signallatenz: 30–150ms** (bei 1s-Polling)

---

## Option 2: Telegram Bot API
**Empfohlen für: 1h+ Timeframes, bereits im Projekt integriert**

### Wie es funktioniert
```
Lokal → Telegram API → Telegram Server → Colab (getUpdates long-poll)
```
Beide Seiten benutzen denselben Bot-Token. Nachrichten werden als
Telegram-Nachrichten mit Hashtag-Kodierung gesendet:
`#bt4t:signals {"action":"BUY","confidence":0.82,...}`

### Einrichtung

**1. Telegram Bot erstellen (falls nicht vorhanden):**
```
1. t.me/BotFather → /newbot
2. Name und Username vergeben
3. Token speichern: 8512251150:AAFXc_...
```

**2. Chat-ID ermitteln:**
```
1. t.me/userinfobot → gibt deine Chat-ID aus
   ODER: Bot starten → t.me/getidsbot → Chat-ID
```

**3. In .env aktivieren:**
```bash
# .env
TELEGRAM_BOT_TOKEN=8512251150:AAFXc_7dGvRXmnEKGv9_9bXYSCxuPijPen8
TELEGRAM_CHAT_ID=2028041322
```
*(Diese Werte sind bereits im .env vorhanden, nur # entfernen)*

**4. Starten (lokal):**
```python
from colab_bridge.transports import get_transport
transport = get_transport("telegram")  # liest aus .env
```

**5. In Colab:**
```python
import os
os.environ["TELEGRAM_BOT_TOKEN"] = "dein_token"
os.environ["TELEGRAM_CHAT_ID"] = "deine_chat_id"

from colab_bridge.transports import get_transport
transport = get_transport("telegram")
```

### Besonderheiten
- **Long-Polling**: Telegram getUpdates wartet bis zu 20s → effektiv Echtzeit-Benachrichtigung
- **Rate-Limit**: 30 Msg/s an einen Chat (bei 30s-Intervall unkritisch)
- **Manuell kontrollierbar**: Du kannst `/pause`, `/resume` direkt im Telegram-Chat schicken
- **Sicherheit**: Nachrichten sind auf Telegram-Servern — keine API-Keys oder Kontostände senden

---

## Option 3: Google Drive
**Empfohlen für: 1h+ Timeframes, wenn Colab bereits Drive nutzt**

### Wie es funktioniert
```
Lokal → Datei schreiben → drive_manager.py synct → Google Drive → Colab liest Datei
```
Jeder Kanal entspricht einer JSON-Datei auf Drive:
```
bt4t_bus/
  market/BTCUSDT_latest.json    ← Lokal schreibt, Colab liest
  signals/latest.json           ← Colab schreibt, Lokal liest
  health/heartbeat.json         ← Colab schreibt
  control/cmd.json              ← Lokal schreibt
```

### Einrichtung

**1. Lokal — drive_manager.py konfigurieren:**
```bash
# drive_manager.py ist bereits im Projekt (infrastructure/drive/drive_manager.py)
# Sync-Verzeichnis in .env:
BT4T_DRIVE_SYNC_DIR=data/drive_sync
```

**2. Starten (lokal):**
```python
from colab_bridge.transports import get_transport
transport = get_transport("gdrive", side="local",
                          sync_dir="data/drive_sync")
```

**3. In Colab:**
```python
from google.colab import drive
drive.mount('/content/drive')

from colab_bridge.transports import get_transport
transport = get_transport("gdrive", side="colab",
    drive_dir="/content/drive/MyDrive/BITCOIN4Traders/bt4t_bus")
```

### Latenz-Details
- Drive-Sync Intervall: 2–15s (abhängig von Google's Sync-Frequenz)
- **Nur für 1h-Bars geeignet** (15m-Bars grenzwertig)

---

## Option 4: Ably
**Empfohlen für: Einfachste Einrichtung ohne eigene Infrastruktur**

### Wie es funktioniert
```
Lokal → Ably Server (WebSocket) ← Colab
```
Beide Seiten verbinden sich direkt zum Ably-Server.
Kein Tunnel, kein Proxy, keine lokale Infrastruktur nötig.

### Einrichtung

**1. Ably Account erstellen:**
```
1. https://ably.com → Sign Up (kostenlos)
2. Dashboard → Create App → API Keys
3. Root Key kopieren: xxxxx:yyyyy
```

**2. In .env:**
```bash
ABLY_API_KEY=xxxxx:yyyyy
```

**3. Python-Paket:**
```bash
pip install ably   # bereits installiert
```

**4. Starten (lokal + Colab identisch):**
```python
import os
from colab_bridge.transports import get_transport
transport = get_transport("ably", api_key=os.getenv("ABLY_API_KEY"))
```

### Free Tier Limits
- 6 Mio. Nachrichten/Monat
- Bei 30s Poll-Intervall, 5 Kanäle: ~86.400 Msg/Monat → **weit unter Limit**
- Nachrichten-History: letzte 100 pro Kanal (für Session-Recovery)

---

## Transport austauschen (ohne Code-Änderungen)

Alle Transporte implementieren dasselbe Interface (`TransportBase`).
Module A und Module B akzeptieren jeden Transport:

```python
from colab_bridge.transports import get_transport
from colab_bridge.module_a_local import ModuleA

# Einfach Transport wechseln:
transport = get_transport("redis")       # Option 1
# transport = get_transport("telegram")  # Option 2
# transport = get_transport("gdrive")    # Option 3
# transport = get_transport("ably")      # Option 4

# Module A und B bekommen den Transport übergeben
engine = ModuleA(transport=transport, ...)
```

---

## Empfehlung nach Use-Case

| Use-Case | Empfohlener Transport | Warum |
|---|---|---|
| 1h-Bars, schnell starten | **Telegram** | Bereits im Projekt, kein Setup |
| 15m-Bars, kein Account | **Redis** | Schnellste Option, volle Kontrolle |
| 5m–1m-Bars | **Redis** | Einzige Option unter 500ms |
| Colab nutzt schon Drive | **Google Drive** | Keine neue Infrastruktur |
| Einfachste Einrichtung | **Ably** | Kein Tunnel, nur API-Key |
| Backup/Fallback | **Telegram** | Immer verfügbar |

---

## Dateistruktur

```
colab_bridge/
├── __init__.py
├── transport_base.py          # Gemeinsames Interface (ABC)
├── module_a_local.py          # Lokale Ausführungs-Engine
├── module_b_colab.py          # Colab RL-Inferenz-Engine
├── control_plane.py           # FastAPI Control Server + Client
├── TRANSPORT_GUIDE.md         # Diese Datei
└── transports/
    ├── __init__.py            # Factory-Funktion get_transport()
    ├── transport_redis.py     # Option 1: Redis + Cloudflare Tunnel
    ├── transport_telegram.py  # Option 2: Telegram Bot API
    ├── transport_gdrive.py    # Option 3: Google Drive
    └── transport_ably.py      # Option 4: Ably Pub/Sub
```
