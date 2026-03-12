# colab_bridge — Vollständige Dokumentation

**Version:** 1.0.0  
**Projekt:** BITCOIN4Traders (Reinforcement Learning Trading Bot)  
**Zweck:** Kommunikations-Bridge zwischen lokalem Rechner (Paper-Trading-Executor) und Google Colab / Cloud (RL-Modell-Inferenz)

---

## Inhaltsverzeichnis

1. [Systemübersicht](#1-systemübersicht)
2. [Architektur-Diagramm](#2-architektur-diagramm)
3. [Verzeichnisstruktur](#3-verzeichnisstruktur)
4. [Kanal-Schema](#4-kanal-schema)
5. [Schnellstart (5 Minuten)](#5-schnellstart-5-minuten)
6. [Modul-Referenz](#6-modul-referenz)
   - 6.1 [transport_base.py — Abstraktes Interface](#61-transport_basepy--abstraktes-interface)
   - 6.2 [module_a_local.py — Lokale Engine](#62-module_a_localpy--lokale-engine)
   - 6.3 [module_b_colab.py — Colab RL-Engine](#63-module_b_colabpy--colab-rl-engine)
   - 6.4 [control_plane.py — Steuerungsschicht](#64-control_planepy--steuerungsschicht)
   - 6.5 [colab_extension.py — Colab Extension](#65-colab_extensionpy--colab-extension)
   - 6.6 [transports/ — Alle 4 Transport-Optionen](#66-transports--alle-4-transport-optionen)
7. [Colab Extension — Detailreferenz](#7-colab-extension--detailreferenz)
   - 7.1 [classify_error()](#71-classify_error)
   - 7.2 [InProcessRepair](#72-inprocessrepair)
   - 7.3 [Reporter](#73-reporter)
   - 7.4 [IterationController](#74-iterationcontroller)
   - 7.5 [MemoryMonitor](#75-memorymonitor)
   - 7.6 [ColabKeepalive](#76-colabkeepalive)
   - 7.7 [ExceptionHook](#77-exceptionhook)
   - 7.8 [BT4TExtension / bt4t (öffentliche API)](#78-bt4textension--bt4t-öffentliche-api)
8. [Befehls-Referenz](#8-befehls-referenz)
9. [Fehlerbehandlungs-Tabelle](#9-fehlerbehandlungs-tabelle)
10. [Umgebungsvariablen](#10-umgebungsvariablen)
11. [Abhängigkeiten](#11-abhängigkeiten)
12. [Copy-Paste Colab-Zellen](#12-copy-paste-colab-zellen)
13. [Troubleshooting](#13-troubleshooting)

---

## 1. Systemübersicht

Das `colab_bridge`-System löst ein fundamentales Problem beim RL-Trading: Das trainierte Modell läuft in der Cloud (Google Colab / GPU-Server), der eigentliche Paper-Trader soll aber lokal mit echten Marktdaten arbeiten und Trades ausführen.

**Das System besteht aus drei Schichten:**

| Schicht | Komponente | Läuft auf |
|---------|-----------|-----------|
| **Ausführung** | Module A (`module_a_local.py`) | Lokaler Rechner (24/7) |
| **Inferenz** | Module B (`module_b_colab.py`) | Google Colab / Cloud |
| **Steuerung** | Control Plane (`control_plane.py`) | Lokal + Colab |
| **Extension** | `colab_extension.py` | Ausschließlich Colab |

**Datenfluss (vereinfacht):**

```
Lokaler Rechner                    Google Colab
────────────────                   ────────────
Binance OHLCV
    ↓
Feature-Berechnung
    ↓
Module A  ──── bt4t:market ────→  Module B
              (Ably/Redis/…)          ↓
                                   RL-Inferenz
                                      ↓
Module A  ←─── bt4t:signals ────  Signal (BUY/SELL/HOLD)
    ↓
Paper-Order ausführen
    ↓
Portfolio-State ─── bt4t:portfolio → Module B (Kontext)
```

---

## 2. Architektur-Diagramm

```
┌─────────────────────────────────────────────────────────────────────┐
│  LOKALER RECHNER                                                     │
│                                                                      │
│  ┌──────────────────┐      ┌─────────────────────────────────────┐  │
│  │    Module A       │      │        Control Server               │  │
│  │ (module_a_local)  │      │       (FastAPI :8765)               │  │
│  │                  │      │                                     │  │
│  │ - Binance OHLCV  │      │  GET  /colab/command  ← Colab pollt │  │
│  │ - RSI/BB/MACD    │      │  POST /colab/command  → Befehle     │  │
│  │ - LocalPortfolio │      │  POST /colab/status   ← Status      │  │
│  │ - Signal-Check   │      │  GET  /positions      → Portfolio   │  │
│  │ - Paper-Orders   │      │  POST /trading/pause               │  │
│  └────────┬─────────┘      └──────────────┬────────────────────┘  │
│           │                               │                         │
│           │  Ably / Redis / Telegram /    │  Cloudflare Tunnel      │
│           │  Google Drive                 │  (HTTPS öffentl. URL)   │
└───────────┼───────────────────────────────┼─────────────────────────┘
            │                               │
            │         INTERNET              │
            │                               │
┌───────────┼───────────────────────────────┼─────────────────────────┐
│  GOOGLE COLAB                             │                          │
│           │                               │ HTTP Poll (alle 5s)      │
│  ┌────────┴──────────┐      ┌─────────────┴──────────────────────┐  │
│  │    Module B        │      │       Control Client               │  │
│  │ (module_b_colab)   │      │   (Colab-Seite der ControlPlane)  │  │
│  │                   │      │                                    │  │
│  │ - ModelAdapter    │      │  - pollt Befehle alle 5s          │  │
│  │   Darwin/.pth/SB3 │      │  - führt PAUSE/RESUME/etc. aus    │  │
│  │ - Obs-Buffer(60)  │      │  - sendet Status zurück            │  │
│  │ - RL-Inferenz     │      │                                    │  │
│  │ - Heartbeat (10s) │      └────────────────────────────────────┘  │
│  └───────────────────┘                                              │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │          colab_extension (bt4t Singleton)                     │   │
│  │                                                              │   │
│  │  Reporter        → HTTP POST → Drive Fallback                │   │
│  │  InProcessRepair → batch_size / LR / gradient_clip           │   │
│  │  IterationCtrl   → PAUSE/RESUME/CHANGE_LR/CHANGE_BS          │   │
│  │  MemoryMonitor   → GPU-Check alle 60s                        │   │
│  │  ColabKeepalive  → numpy compute alle 600s                   │   │
│  │  ExceptionHook   → sys.excepthook (global)                   │   │
│  └──────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 3. Verzeichnisstruktur

```
colab_bridge/
│
├── __init__.py              # Paket-Init, Version 1.0.0
├── transport_base.py        # Abstraktes Interface (ABC) für alle Transporte
├── module_a_local.py        # Lokale Ausführungs-Engine
├── module_b_colab.py        # Colab RL-Inferenz-Engine
├── control_plane.py         # FastAPI ControlServer + ControlClient
├── colab_extension.py       # Colab Extension (bt4t Singleton)
│
├── DOKUMENTATION.md         # Diese Datei
├── TRANSPORT_GUIDE.md       # Transport-Vergleich und Setup
│
└── transports/
    ├── __init__.py           # Factory get_transport()
    ├── transport_ably.py     # Option 1: Ably Pub/Sub (50–150ms)
    ├── transport_redis.py    # Option 2: Redis + Cloudflare (30–150ms)
    ├── transport_telegram.py # Option 3: Telegram Bot (200–800ms)
    └── transport_gdrive.py   # Option 4: Google Drive (2–15s)
```

---

## 4. Kanal-Schema

Alle Transporte verwenden identische Kanal-Namen (definiert in `transport_base.py`):

| Konstante | Kanalname | Richtung | Inhalt |
|-----------|-----------|----------|--------|
| `CH_MARKET` | `bt4t:market:{symbol}` | Lokal → Colab | OHLCV + Features (RSI/BB/MACD) |
| `CH_SIGNALS` | `bt4t:signals` | Colab → Lokal | Handelssignal (BUY/SELL/HOLD + Confidence) |
| `CH_PORTFOLIO` | `bt4t:portfolio:state` | Lokal → Colab | Portfolio-State (Equity, Position, P&L) |
| `CH_HEALTH` | `bt4t:health` | Colab → Lokal | Heartbeat (alle 10s) |
| `CH_CONTROL` | `bt4t:control:cmd` | Lokal → Colab | Steuerbefehle (PAUSE/RESUME/RELOAD/…) |
| `CH_ACK` | `bt4t:control:ack` | Colab → Lokal | Bestätigung für Befehle |

### Signal-Payload (`bt4t:signals`)

```json
{
  "action": "BUY",
  "symbol": "BTCUSDT",
  "confidence": 0.7823,
  "model_version": "MV_Gen4_BB_breakout_p11_std1.91_9",
  "timestamp_utc": "2024-01-15T10:30:00+00:00",
  "inference_latency_ms": 12.3,
  "signal_n": 42,
  "portfolio_equity": 10500.00,
  "portfolio_return_pct": 5.0
}
```

### Market-Payload (`bt4t:market:BTCUSDT`)

```json
{
  "timestamp": "2024-01-15T10:30:00+00:00",
  "symbol": "BTCUSDT",
  "open": 42000.0, "high": 42500.0, "low": 41800.0,
  "close": 42300.0, "volume": 1234.5,
  "rsi14": 58.3,
  "bb_pct": 0.62,
  "bb_upper": 43000.0, "bb_lower": 41000.0,
  "macd": 120.5,
  "vol_ratio": 1.3,
  "return_1h": 0.0071, "return_4h": 0.0152, "return_24h": 0.0234,
  "sma20": 42100.0,
  "close_60": [41500.0, 41600.0, ..., 42300.0]
}
```

---

## 5. Schnellstart (5 Minuten)

### Schritt 1: Ably Account erstellen (kostenlos)

```
1. https://ably.com → "Sign up for free"
2. Create App → "BITCOIN4Traders"
3. API Keys → Root Key kopieren
4. In .env eintragen: ABLY_API_KEY=your_root_key_here
```

### Schritt 2: Lokalen Stack starten

```bash
# Terminal 1: Cloudflare Tunnel (optional, für Control-Plane)
./cloudflared tunnel --url http://localhost:8765

# Terminal 2: Vollständiger lokaler Stack
python colab_bridge/control_plane.py full \
    --ably-key $ABLY_API_KEY \
    --capital 10000

# ODER: Nur Module A (ohne Control Server)
python colab_bridge/module_a_local.py \
    --ably-key $ABLY_API_KEY \
    --symbol BTC/USDT \
    --interval 30
```

### Schritt 3: Google Colab einrichten

Erste Zelle im Colab-Notebook (copy-paste, einmalig):

```python
# ─── Zelle 1: Setup ───────────────────────────────────────────
import sys, os
from google.colab import drive
drive.mount('/content/drive')

sys.path.insert(0, '/content/drive/MyDrive/BITCOIN4Traders')

# Konfiguration (URL aus Terminal 1 kopieren)
os.environ['BT4T_LISTENER_URL'] = 'https://abc-def.trycloudflare.com'
os.environ['BT4T_API_TOKEN']    = 'bt4t-secret-token'
os.environ['BT4T_NOTEBOOK_ID']  = 'training_v1'
os.environ['ABLY_API_KEY']      = 'your_ably_key'

# bt4t Extension laden
from colab_bridge.colab_extension import bt4t
bt4t.install()

# Module B starten
from colab_bridge.module_b_colab import ModuleB
import asyncio

engine = ModuleB(
    ably_key=os.environ['ABLY_API_KEY'],
    model_path='/content/drive/MyDrive/BITCOIN4Traders/data/cache/multiverse_champion.pkl',
)
await engine.run()
```

---

## 6. Modul-Referenz

### 6.1 `transport_base.py` — Abstraktes Interface

Definiert das gemeinsame Interface für alle 4 Transporte und die Kanal-Konstanten.

#### Klasse: `TransportBase` (ABC)

Jeder Transport muss diese 5 Methoden implementieren:

| Methode | Signatur | Beschreibung |
|---------|---------|--------------|
| `connect` | `async connect() -> None` | Verbindung aufbauen. Wirft Exception bei Fehler. |
| `disconnect` | `async disconnect() -> None` | Verbindung sauber trennen. |
| `publish` | `async publish(channel: str, payload: dict) -> None` | Nachricht auf Kanal senden. |
| `subscribe` | `async subscribe(channel: str, callback: Callable) -> None` | Callback für eingehende Nachrichten registrieren. |
| `name` | `@property -> str` | Name des Transports für Logging. |
| `latency_class` | `@property -> str` | Latenzklasse: `'ms'` / `'sub-second'` / `'seconds'` |

#### Hilfsmethoden (statisch):

```python
TransportBase.encode(payload: dict) -> str     # dict → JSON-String
TransportBase.decode(data: str|bytes|dict) -> dict  # JSON → dict
```

---

### 6.2 `module_a_local.py` — Lokale Engine

Läuft auf dem lokalen Rechner (24/7). Verbindet Exchange mit Colab über Ably.

#### Klasse: `LocalPaperPortfolio`

In-Memory Paper-Portfolio. Keine externe Datenbank nötig.

```python
portfolio = LocalPaperPortfolio(
    initial_capital=10_000.0,  # Startkapital in USDT
    fee_rate=0.001             # 0.1% Handelsgebühr
)
```

| Methode | Beschreibung |
|---------|-------------|
| `execute(side, price, confidence)` | Simuliert Market-Order. Gibt Trade-Dict zurück oder `None`. |
| `current_equity(price)` | Berechnet aktuellen Gesamtwert (Cash + Position). |
| `state_dict(price)` | Gibt vollständigen Portfolio-State als Dict zurück. |
| `pause()` / `resume()` | Stoppt/startet Trading (bei Heartbeat-Timeout). |

**Trade-Dict (Rückgabe von `execute()`):**
```python
{
    "timestamp": "2024-01-15T10:30:00+00:00",
    "action": "BUY",    # BUY / SELL / SHORT / COVER
    "side": "buy",
    "qty": 0.0023,      # BTC-Menge
    "price": 42315.0,   # Fill-Preis (inkl. 0.03% Slippage)
    "fee": 0.09,
    "pnl": 0.0,         # Realisierter P&L (0 bei Eröffnung)
    "confidence": 0.78  # Signal-Konfidenz
}
```

#### Funktion: `compute_features(df: pd.DataFrame) -> dict`

Berechnet Trading-Features aus OHLCV DataFrame (200 Bars empfohlen).

**Berechnete Features:**

| Feature | Beschreibung | Formel |
|---------|-------------|--------|
| `rsi14` | RSI mit 14 Perioden | Wilder's RSI |
| `bb_pct` | Position in Bollinger Band (0–1) | `(close - lower) / (upper - lower)` |
| `bb_upper` / `bb_lower` | Bollinger Bands (20, 2σ) | SMA20 ± 2×STD20 |
| `macd` | MACD Linie | EMA12 − EMA26 |
| `vol_ratio` | Volumen relativ zum 20er Durchschnitt | `vol / mean(vol[-20:])` |
| `return_1h` | 1-Bar Return | `close[-1]/close[-2] - 1` |
| `return_4h` | 4-Bar Return | `close[-1]/close[-5] - 1` |
| `return_24h` | 24-Bar Return | `close[-1]/close[-25] - 1` |
| `sma20` | Einfacher Durchschnitt (20) | `mean(close[-20:])` |
| `close_60` | Letzte 60 Close-Werte | Array für RL-Observation |

#### Klasse: `ModuleA`

```python
engine = ModuleA(
    ably_key="your_key",          # ABLY_API_KEY
    symbol="BTC/USDT",            # CCXT-Format
    timeframe="1h",               # OHLCV-Timeframe
    exchange_id="binance",        # CCXT Exchange-ID
    poll_interval_s=30.0,         # Sekunden zwischen OHLCV-Abfragen
    initial_capital=10_000.0,     # Startkapital USDT
)
await engine.run()                # Blockiert bis Ctrl+C
```

**Interner Loop (alle `poll_interval_s`):**
1. OHLCV via CCXT holen (200 Bars)
2. `compute_features()` aufrufen
3. Features auf `bt4t:market:BTCUSDT` publishen
4. Alle 5 Ticks: Portfolio-State auf `bt4t:portfolio:state` publishen
5. Heartbeat-Alter prüfen (Timeout: 90s)
6. Alle 10 Ticks: Status-Log ausgeben

**Signal-Validierung (automatisch in `_on_signal()`):**
- Staleness-Check: Signal älter als 10s → verwerfen
- Confidence-Check: Unter 0.55 → verwerfen

**Befehle an Colab senden:**
```python
await engine.send_command("PAUSE_INFERENCE")
await engine.send_command("RELOAD_MODEL", {"model_path": "/path/to/model.pkl"})
await engine.send_command("SHUTDOWN")
```

**CLI:**
```bash
python colab_bridge/module_a_local.py [--symbol BTC/USDT] [--timeframe 1h]
    [--exchange binance] [--interval 30] [--capital 10000] [--ably-key KEY]
```

---

### 6.3 `module_b_colab.py` — Colab RL-Engine

Läuft in Google Colab. Empfängt Marktdaten, führt RL-Inferenz durch.

#### Klasse: `ModelAdapter`

Abstraktionsschicht die 4 Modell-Typen unterstützt:

```python
adapter = ModelAdapter(
    model_path="/path/to/model.pkl",  # oder .pth, .zip, None
    model_type="auto"                  # auto / darwin / sb3 / pytorch / rsi_fallback
)
action, confidence = adapter.predict(close_array, features_dict)
```

**Lade-Priorität (auto):**

| Reihenfolge | Dateiendung | Modell-Typ | Beschreibung |
|-------------|-------------|-----------|--------------|
| 1 | `.pkl` | `darwin` | Darwin-Champion aus Multiverse-Training |
| 2 | `.pth` | `pytorch` | PyTorch Checkpoint (fällt auf RSI zurück) |
| 3 | `.zip` oder `sb3` im Pfad | `sb3` | Stable-Baselines3 PPO |
| 4 | kein Pfad / Laden fehlgeschlagen | `rsi_fallback` | RSI-Signal (kein Modell nötig) |

**Darwin-Inferenz:**
```python
# intern: model.compute_signals(close_array)
# Konfidenz = Clip(abs(sigs[-10:]).mean() + 0.5, 0, 1)
# Signal = sign(sigs[-1]) → 1=BUY, -1=SELL, 0=HOLD
```

**RSI-Fallback-Inferenz:**
```python
# RSI < 30 → BUY  mit confidence = 0.5 + (30-rsi)/60
# RSI > 70 → SELL mit confidence = 0.5 + (rsi-70)/60
# sonst    → HOLD
```

**Modell neu laden:**
```python
adapter.reload("/path/to/new_model.pkl")  # oder None = gleicher Pfad
```

#### Klasse: `ModuleB`

```python
engine = ModuleB(
    ably_key="your_key",
    model_path="/content/drive/MyDrive/.../multiverse_champion.pkl",
    symbol="BTCUSDT",          # Ohne Slash (Ably-Kanal-Format)
    min_confidence=0.55,       # Signale unter dieser Schwelle → HOLD
)
await engine.run()             # Blockiert bis SHUTDOWN oder Ctrl+C
```

**Observation-Buffer:**
- Rolling Window der letzten 70 Marktdaten-Updates (`deque(maxlen=70)`)
- Inferenz startet wenn Buffer ≥ 20 Einträge hat
- RL-Observation = letzten 60 Bars + 6 skalare Features

**Heartbeat (automatisch alle 10s):**
```json
{
  "model_loaded": true,
  "model_version": "MV_Gen4_BB...",
  "inference_count": 142,
  "signal_count": 23,
  "obs_buffer_size": 70,
  "paused": false,
  "last_market_data_age_s": 8.2,
  "status": "COLAB_READY"
}
```

**Empfangene Befehle (von `bt4t:control:cmd`):**

| Befehl | Effekt |
|--------|--------|
| `PAUSE_INFERENCE` | Stoppt RL-Inferenz (keine Signale) |
| `RESUME` | Setzt Inferenz fort |
| `RELOAD_MODEL` | Lädt Modell neu (`params.model_path`) |
| `SHUTDOWN` | Beendet Module B |
| `STATUS` | Sendet Status-ACK |

**Hilfsfunktionen:**
```python
from colab_bridge.module_b_colab import colab_setup, find_champion_on_drive

project_path = colab_setup(drive_mount=True)   # Mountet Drive, gibt Pfad zurück
champion = find_champion_on_drive(project_path) # Sucht .pkl Datei
```

**CLI (für vast.ai / lokale GPU):**
```bash
python colab_bridge/module_b_colab.py \
    --ably-key $ABLY_API_KEY \
    --model data/cache/multiverse_champion.pkl \
    --symbol BTCUSDT \
    --min-conf 0.55
```

---

### 6.4 `control_plane.py` — Steuerungsschicht

#### Klasse: `ControlServer`

FastAPI-Server der lokal läuft und via Cloudflare Tunnel erreichbar ist.

```python
server = ControlServer(
    port=8765,
    token="bt4t-secret-token",  # CONTROL_API_TOKEN
    module_a=engine,             # Optional: ModuleA-Referenz für Portfolio-State
)
await server.start()
```

**API-Endpunkte:**

| Method | Endpunkt | Auth | Beschreibung |
|--------|---------|------|--------------|
| `GET` | `/health` | nein | Healthcheck |
| `GET` | `/status` | Bearer | Uptime, Queue-Länge, Colab-Status |
| `GET` | `/positions` | Bearer | Aktueller Portfolio-State |
| `GET` | `/colab/command` | Bearer | Nächster Befehl für Colab (Colab pollt hier) |
| `POST` | `/colab/command` | Bearer | Befehl an Colab senden |
| `POST` | `/colab/status` | Bearer | Colab reportet Status |
| `POST` | `/trading/pause` | Bearer | Portfolio + Inferenz pausieren |
| `POST` | `/trading/resume` | Bearer | Portfolio + Inferenz fortsetzen |

**Befehl an Colab senden (HTTP):**
```bash
curl -X POST http://localhost:8765/colab/command \
  -H "Authorization: Bearer bt4t-secret-token" \
  -H "Content-Type: application/json" \
  -d '{"cmd": "PAUSE_INFERENCE", "params": {}}'
```

**Gültige Commands für `POST /colab/command`:**
`PAUSE_INFERENCE` | `RESUME` | `RELOAD_MODEL` | `SHUTDOWN` | `STATUS`

**Befehl-Expiry:** Befehle die länger als 60s in der Queue warten werden automatisch verworfen.

#### Klasse: `ControlClient`

Läuft in Colab neben Module B. Pollt ControlServer alle 5s.

```python
client = ControlClient(
    server_url="https://abc-def.trycloudflare.com",
    module_b=engine,
    token="bt4t-secret-token",
    poll_interval_s=5.0,
)
asyncio.create_task(client.run())  # Als async Task starten
```

#### Funktion: `start_cloudflare_tunnel(port) -> Optional[str]`

Startet `cloudflared`-Prozess und gibt die öffentliche HTTPS-URL zurück.

```python
url = await start_cloudflare_tunnel(port=8765)
# → "https://abc-def.trycloudflare.com"
```

Voraussetzung: `cloudflared` muss installiert sein.

#### Funktion: `start_full_local_stack()`

Startet Module A + ControlServer + Cloudflare Tunnel in einem Aufruf:

```python
await start_full_local_stack(
    ably_key="your_key",
    capital=10_000.0,
    symbol="BTC/USDT",
    timeframe="1h",
    exchange_id="binance",
    poll_interval_s=30.0,
    api_token="bt4t-secret-token",
    start_tunnel=True,
)
```

**CLI:**
```bash
python colab_bridge/control_plane.py server   # Nur Control Server
python colab_bridge/control_plane.py tunnel   # Nur Cloudflare Tunnel
python colab_bridge/control_plane.py full \   # Alles zusammen
    --ably-key $ABLY_API_KEY \
    --capital 10000 --symbol BTC/USDT
```

---

### 6.5 `colab_extension.py` — Colab Extension

Die Extension hängt sich mit **einem einzigen Import** in jedes bestehende Colab-Notebook ein. Kein Code-Umbau nötig.

**Singleton-Import:**
```python
from colab_bridge.colab_extension import bt4t
```

Das Objekt `bt4t` ist eine globale Singleton-Instanz der `BT4TExtension`-Klasse.

Die detaillierte Referenz aller Klassen findet sich in [Kapitel 7](#7-colab-extension--detailreferenz).

---

### 6.6 `transports/` — Alle 4 Transport-Optionen

#### Factory-Funktion: `get_transport()`

```python
from colab_bridge.transports import get_transport

# Ably
t = get_transport("ably", api_key="root:xxx")

# Redis
t = get_transport("redis", side="local")    # auf lokalem Rechner
t = get_transport("redis", side="colab",    # in Colab
    redis_url="https://tunnel.dev/redis")

# Telegram
t = get_transport("telegram",
    bot_token="123:ABC",
    chat_id="-100123456")

# Google Drive
t = get_transport("gdrive", side="local")
t = get_transport("gdrive", side="colab")
```

#### Vergleich der 4 Optionen:

| Option | Latenz | Kosten | Konto | Beste für |
|--------|--------|--------|-------|-----------|
| **Ably** | 50–150ms | Free Tier: 100 Conn | Ja (Ably) | Primärtransport |
| **Redis** | 30–150ms | Kostenlos | Nein | Kein externer Account |
| **Telegram** | 200–800ms | Kostenlos | Bot-Token | Bereits in .env |
| **Google Drive** | 2–15s | Kostenlos | Google | Fallback / Debug |

Ausführliche Setup-Anleitungen: [TRANSPORT_GUIDE.md](TRANSPORT_GUIDE.md)

---

## 7. Colab Extension — Detailreferenz

### 7.1 `classify_error()`

```python
error_type, repair_action, severity = classify_error(exc)
```

Klassifiziert eine Exception anhand von Regex-Mustern.

**Erkannte Fehlertypen:**

| Fehlertyp | Erkennungs-Pattern | Aktion | Schwere |
|-----------|-------------------|--------|---------|
| `OOM` | `CUDA out of memory`, `OutOfMemoryError`, `OOM` | `halve_batch_size` | high |
| `NAN_LOSS` | `nan`, `NaN`, `inf.*loss/reward/gradient` | `reduce_lr` | high |
| `EXPLODING` | `gradient.*explod`, `loss.*explod`, `overflow` | `clip_gradients` | high |
| `IMPORT` | `ModuleNotFoundError`, `No module named` | `pip_install` | medium |
| `TIMEOUT` | `TimeoutError`, `ReadTimeout`, `socket.timeout` | `increase_timeout` | low |
| `CONNECTION` | `ConnectionError`, `ConnectTimeout` | `retry_connection` | medium |
| `CUDA_ERROR` | `RuntimeError.*CUDA`, `device-side assert` | `halve_batch_size` | high |
| `DATA_ERROR` | `KeyError`, `IndexError`, `ValueError.*batch/data` | `skip_batch` | medium |
| `IO_ERROR` | `PermissionError`, `FileNotFoundError.*drive/model` | `retry_io` | medium |
| `INTERRUPTED` | `KeyboardInterrupt` | `none` | low |
| `UNKNOWN` | (kein Muster passt) | `none` | medium |

**Beispiel:**
```python
try:
    train(model, data)
except Exception as exc:
    error_type, action, severity = classify_error(exc)
    # error_type = "OOM", action = "halve_batch_size", severity = "high"
```

---

### 7.2 `InProcessRepair`

Repariert Hyperparameter direkt im laufenden Python-Prozess ohne Notebook-Neustart.  
Sucht Variablen im globalen Namespace (`__main__`).

```python
repair = InProcessRepair(repair_log=[])
result = repair.apply("halve_batch_size", context={})
# result = {"action": "halve_batch_size", "changes": ["BATCH_SIZE: 32 → 16"], "success": True}
```

**Methode `apply(action, context) -> dict`:**

| Aktion | Was wird geändert |
|--------|-------------------|
| `halve_batch_size` | Halbiert alle `batch_size`-Variablen im globalen NS. Leert CUDA-Cache. |
| `reduce_lr` | Reduziert `learning_rate`/`lr`-Variablen um Faktor 10. Patcht auch PyTorch-Optimizer. |
| `clip_gradients` | Setzt `gradient_clip_val` auf min(aktuell, 0.5). Neu: `GRADIENT_CLIP_VAL = 0.5`. |
| `pip_install` | Installiert fehlendes Paket aus Fehlermeldung via `pip`. |
| `increase_timeout` | Verdoppelt alle `timeout`-Variablen im globalen NS. |
| `skip_batch` | Loggt Warnung (Signal für Notebook-Code). |
| `retry_connection` | Wartet 30s. |
| `retry_io` | Wartet 10s. |
| `none` | Keine Aktion. |

**Rückgabe-Dict:**
```python
{
    "timestamp": "2024-01-15T10:30:00+00:00",
    "action": "halve_batch_size",
    "changes": ["BATCH_SIZE: 32 → 16", "optimizer[opt].lr: 1.00e-03 → 1.00e-04"],
    "success": True
}
```

**Wichtig:** Die Reparatur ist nur wirksam wenn die Variablen im globalen Namespace existieren. Lokale Variablen in Funktionen werden nicht erfasst.

---

### 7.3 `Reporter`

Sendet Berichte an den lokalen Rechner. Dual-Kanal: HTTP POST → Google Drive Fallback.

```python
reporter = Reporter(
    listener_url="https://tunnel.trycloudflare.com",
    api_token="bt4t-secret-token",
    notebook_id="training_v1"
)
reporter.start()   # Hintergrund-Send-Thread starten
reporter.stop()    # Thread stoppen
```

**Öffentliche Methoden:**

| Methode | Beschreibung |
|---------|-------------|
| `report_error(exc, error_type, repair_applied)` | Sendet Fehlerbericht mit Stacktrace |
| `report_progress(data: dict)` | Sendet Trainings-Fortschritt (Epoch, Loss, Reward) |
| `report_heartbeat(extra={})` | Sendet Heartbeat mit Status `COLAB_ALIVE` |
| `poll_commands() -> list[dict]` | Fragt lokalen Rechner synchron nach Befehlen |

**Sendemechanismus:**
1. Alle Berichte werden in interne Queue gestellt
2. Hintergrund-Thread (`_send_loop`) leert Queue kontinuierlich
3. Jede Nachricht: HTTP POST an `{listener_url}/report_error`
4. Bei HTTP-Fehler: JSON-Datei in `/content/drive/MyDrive/BITCOIN4Traders/bt4t_bus/reports/`

**Befehle pollen (`poll_commands()`):**
```python
commands = reporter.poll_commands()
# → [{"cmd": "PAUSE", "params": {}}]   oder []
```
Fragt `GET {listener_url}/colab/command` (max. 5s Timeout).

---

### 7.4 `IterationController`

Steuert den Trainings-Loop von außen. Wird intern von `bt4t.step()` aufgerufen.

```python
controller = IterationController(reporter=reporter, repair=repair)
should_continue = controller.process(epoch=5, step=100, loss=0.42, reward=1.5)
```

**Methode `process(...) -> bool`:**

Wird bei jedem Trainings-Schritt aufgerufen. Gibt `False` zurück wenn Training gestoppt werden soll.

Interne Schritte:
1. State aktualisieren (epoch, step, loss, reward)
2. Alle 10 Steps: Fortschritt via Reporter senden
3. Alle 5s: Befehle pollen + ausführen
4. Wenn pausiert: Warte-Loop (prüft alle 5s ob RESUME kommt)
5. Return: `not stop_requested`

**`IterationState` Felder:**

```python
state.epoch            # Aktueller Epoch
state.step             # Aktueller Step
state.loss             # Letzter Loss-Wert
state.reward           # Letzter Reward-Wert
state.paused           # True wenn PAUSE-Befehl empfangen
state.stop_requested   # True wenn STOP-Befehl empfangen
state.lr               # Remote-gesetzte Learning Rate
state.batch_size       # Remote-gesetzte Batch Size
state.checkpoint_every # Checkpoint alle N Steps (Standard: 50)
```

**Verarbeitete Befehle:**

| Befehl | Effekt auf State |
|--------|-----------------|
| `PAUSE` | `paused = True` → Warte-Loop |
| `RESUME` | `paused = False` → Warte-Loop verlassen |
| `STOP` / `SHUTDOWN` | `stop_requested = True` → Loop gibt False zurück |
| `CHANGE_LR` | Setzt LR im globalen NS + PyTorch-Optimizer |
| `CHANGE_BS` | Setzt `batch_size` im globalen NS |
| `RELOAD_MODEL` | Setzt `BT4T_RELOAD_REQUESTED = True` + `BT4T_RELOAD_MODEL_PATH` im globalen NS |
| `STATUS` | Sendet aktuellen State als Fortschritts-Bericht |

---

### 7.5 `MemoryMonitor`

Überwacht GPU-Speicher alle 60s und führt prophylaktische Bereinigung durch.

```python
monitor = MemoryMonitor(
    repair=repair,
    warn_pct=85.0    # Warnschwelle in % (Standard: 85%)
)
monitor.start()   # Hintergrund-Thread starten
monitor.stop()    # Thread stoppen
```

**Automatisches Verhalten:**
- `GPU > 85%`: `gc.collect()` + `torch.cuda.empty_cache()` → Log-Ausgabe
- `GPU > 95%` nach Bereinigung: `repair.apply("halve_batch_size", {})` → Batch-Size halbieren

Funktioniert nur wenn `torch` installiert und GPU verfügbar ist. Andernfalls: No-op.

---

### 7.6 `ColabKeepalive`

Verhindert Colab-Session-Timeout durch echte Compute-Aufgaben.

```python
keepalive = ColabKeepalive(
    reporter=reporter,
    interval_s=600.0    # Alle 10 Minuten (Standard)
)
keepalive.start()
keepalive.stop()
```

**Mechanismus:**
- Alle `interval_s` Sekunden: `numpy.random.randn(1000, 1000).mean()` (echte Compute-Aufgabe)
- Schreibt kein Sleep-Trick — Colab erkennt aktiven Kernel durch GPU-CPU-Aktivität
- Sendet Heartbeat mit `{"keepalive_tick": "HH:MM:SS"}`

**Hinweis:** Colab killt Sessions bei **Inaktivität** (kein Output/Compute), nicht nach einer festen Zeit. Die numpy-Berechnung simuliert echte Aktivität.

---

### 7.7 `ExceptionHook`

Globaler `sys.excepthook` — fängt alle unbehandelten Exceptions.

```python
hook = ExceptionHook(reporter=reporter, repair=repair)
hook.install()    # sys.excepthook überschreiben
hook.uninstall()  # Original-Hook wiederherstellen
```

**Ablauf bei Exception:**
1. `KeyboardInterrupt` → an Original-Hook weitergeben (keine Behandlung)
2. `classify_error(exc)` aufrufen → Fehlertyp + Aktion + Schwere
3. Reparatur-Zähler für diesen Fehlertyp erhöhen
4. Wenn `action != "none"` und `count <= 5`: `repair.apply(action, ctx)` aufrufen
5. Wenn `count > 5`: Warnung "keine weitere Reparatur" (verhindert Endlos-Loop)
6. `reporter.report_error(exc, error_type, repair_result)` aufrufen
7. Original-Hook aufrufen (Traceback im Notebook ausgeben)

---

### 7.8 `BT4TExtension` / `bt4t` (öffentliche API)

Das Singleton `bt4t` ist die einzige öffentliche Schnittstelle der Extension.

```python
from colab_bridge.colab_extension import bt4t
```

#### `bt4t.install()` — Extension einrichten

```python
bt4t.install(
    listener_url="https://abc-def.trycloudflare.com",  # Optional (auch via ENV)
    api_token="bt4t-secret-token",    # Optional
    notebook_id="training_v1",        # Optional
    keepalive=True,                   # Colab-Keepalive aktivieren
    memory_monitor=True,              # GPU-Überwachung aktivieren
    exception_hook=True,              # Globalen Exception-Hook installieren
    keepalive_interval_s=600.0,       # Keepalive-Intervall
    memory_warn_pct=85.0,             # GPU-Warnschwelle %
)
```

- Gibt `self` zurück (Method-Chaining möglich)
- Idempotent: zweiter Aufruf loggt Warnung und tut nichts
- Sendet `COLAB_READY` Heartbeat nach erfolgreicher Installation
- Liest Konfiguration aus Umgebungsvariablen (Override möglich via Parameter)

**Ausgabe nach `install()`:**
```
═══════════════════════════════════════════════════════
  bt4t Extension installiert
  Notebook  : training_v1
  Listener  : https://abc-def.trycloudflare.com
  Keepalive : an
  Memory    : an
  ExcHook   : an
═══════════════════════════════════════════════════════
```

#### `bt4t.step()` — Trainings-Schritt melden

```python
should_continue = bt4t.step(
    epoch=5,
    step=100,
    loss=0.42,
    reward=1.5,
    # beliebige weitere kwargs werden als "extra" mitgesendet
    accuracy=0.94,
)
# False wenn STOP-Befehl empfangen

# Typische Verwendung:
for epoch in range(100):
    loss = train_one_epoch(model, data)
    if not bt4t.step(epoch=epoch, loss=float(loss)):
        break
```

Wenn Extension nicht installiert (`bt4t.install()` nicht aufgerufen): gibt immer `True` zurück.

#### `bt4t.guard` — Decorator

```python
@bt4t.guard
def run_training():
    for epoch in range(100):
        loss = train(model, data)
        bt4t.step(epoch=epoch, loss=loss)
```

- Max. 3 Versuche bei Fehler
- Klassifiziert Fehler + führt Reparatur aus
- Sendet Fehlerbericht
- Bei `high`-Schwere oder letztem Versuch: re-raise
- `KeyboardInterrupt` wird immer sofort weitergegeben

#### `bt4t.session()` — Context-Manager

```python
with bt4t.session("experiment_42"):
    train_model(model, data)
```

- Sendet `SESSION_START` Heartbeat beim Eintritt
- Sendet `SESSION_END` mit Status `OK` / `ERROR` / `INTERRUPTED` beim Verlassen
- Bei Exception: Fehler klassifizieren + reparieren + melden, dann re-raise

#### `bt4t.send_checkpoint()` — Checkpoint melden

```python
bt4t.send_checkpoint(
    model_path="/content/drive/.../model_ep50.pth",
    metrics={"loss": 0.42, "reward": 1.5, "sharpe": 1.2}
)
```

Sendet `CHECKPOINT`-Event mit aktuellem Epoch/Step und optionalen Metriken.

#### `bt4t.send_alert()` — Manuelle Nachricht

```python
bt4t.send_alert("Epoch 50 abgeschlossen", level="INFO")
bt4t.send_alert("WARNUNG: Loss divergiert!", level="WARNING")
```

#### `bt4t.status()` — Aktuellen Status abfragen

```python
status = bt4t.status()
# {
#   "installed": True,
#   "notebook_id": "training_v1",
#   "listener_url": "https://...",
#   "epoch": 42,
#   "step": 1234,
#   "paused": False,
#   "stop_requested": False,
#   "repairs_done": 2
# }
```

#### `bt4t.repair_log()` — Reparatur-Protokoll

```python
repairs = bt4t.repair_log()
# [
#   {"timestamp": "...", "action": "halve_batch_size",
#    "changes": ["BATCH_SIZE: 32 → 16"], "success": True},
#   ...
# ]
```

#### `bt4t.should_stop` / `bt4t.is_paused` — Eigenschaften

```python
if bt4t.should_stop:
    print("Training soll gestoppt werden")
if bt4t.is_paused:
    print("Training ist pausiert")
```

#### `bt4t.uninstall()` — Extension entfernen

```python
bt4t.uninstall()  # Alle Hooks entfernen, alle Threads stoppen
```

---

## 8. Befehls-Referenz

### Befehle von Lokal → Colab (via `bt4t:control:cmd` oder `POST /colab/command`)

| Befehl | Parameter | Verarbeitet von | Beschreibung |
|--------|-----------|----------------|--------------|
| `PAUSE_INFERENCE` | – | ModuleB | Stoppt RL-Inferenz in Module B |
| `RESUME` | – | ModuleB + IterationCtrl | Setzt Inferenz + Training fort |
| `RELOAD_MODEL` | `model_path: str` | ModuleB + IterationCtrl | Lädt neues Modell |
| `SHUTDOWN` | – | ModuleB | Beendet Module B sauber |
| `STATUS` | – | ModuleB + IterationCtrl | Fordert Status-Bericht an |
| `PAUSE` | – | IterationController | Pausiert Training-Loop |
| `STOP` | – | IterationController | Stoppt Training-Loop (gibt `False` bei `step()`) |
| `CHANGE_LR` | `value: float` | IterationController | Ändert Learning Rate live |
| `CHANGE_BS` | `value: int` | IterationController | Ändert Batch Size live |

### Befehle von Colab → Lokal (via `bt4t:control:ack`)

| Event | Inhalt | Beschreibung |
|-------|--------|--------------|
| ACK | `{cmd, status, msg}` | Bestätigung für jeden empfangenen Befehl |

### Events von Colab → Lokal (via Reporter HTTP POST)

| Typ | Felder | Beschreibung |
|-----|--------|--------------|
| `heartbeat` | `{status, notebook_id, timestamp_utc}` | Lebenszeichen (auch: `COLAB_READY`, `SESSION_START`) |
| `progress` | `{epoch, step, loss, reward}` | Trainings-Fortschritt (alle 10 Steps) |
| `error` | `{error_type, error_message, stacktrace, repair_applied}` | Fehlerbericht |
| `CHECKPOINT` | `{model_path, metrics, epoch, step}` | Checkpoint-Meldung |
| `ALERT` | `{level, message}` | Manuelle Nachricht |

---

## 9. Fehlerbehandlungs-Tabelle

| Fehler tritt auf | Erkannt als | Automatische Aktion | Ergebnis |
|-----------------|-------------|---------------------|---------|
| GPU-Speicher voll | `OOM` | `halve_batch_size` | BATCH_SIZE /2, CUDA cache geleert |
| Loss ist NaN | `NAN_LOSS` | `reduce_lr` | LR /10, Optimizer-LR angepasst |
| Gradient-Explosion | `EXPLODING` | `clip_gradients` | GRADIENT_CLIP_VAL = 0.5 |
| Fehlende Bibliothek | `IMPORT` | `pip_install` | Automatisches pip install |
| Netzwerk-Timeout | `TIMEOUT` | `increase_timeout` | Timeout-Variablen ×2 |
| Verbindungsfehler | `CONNECTION` | `retry_connection` | 30s warten |
| CUDA Runtime-Fehler | `CUDA_ERROR` | `halve_batch_size` | Wie OOM |
| Falsche Daten | `DATA_ERROR` | `skip_batch` | Flag setzen |
| Drive nicht erreichbar | `IO_ERROR` | `retry_io` | 10s warten |
| Gleicher Fehler >5x | Beliebig | Keine weitere Reparatur | Endlos-Loop vermieden |

**Dreistufige Eskalation:**
1. **ExceptionHook** fängt unbehandelte Exceptions (automatisch)
2. **`bt4t.guard`-Decorator** versucht bis zu 3x (opt-in)
3. **`bt4t.session`-Context-Manager** fängt + meldet (opt-in)

---

## 10. Umgebungsvariablen

### Für die Colab Extension (`colab_extension.py`)

| Variable | Standard | Beschreibung |
|---------|---------|--------------|
| `BT4T_LISTENER_URL` | `""` | HTTPS-URL des lokalen Control Servers (Cloudflare Tunnel) |
| `BT4T_API_TOKEN` | `"bt4t-secret-token"` | Shared Secret (muss mit ControlServer übereinstimmen) |
| `BT4T_NOTEBOOK_ID` | `"colab_notebook"` | Name des Notebooks für Logs/Berichte |

### Für das gesamte System

| Variable | Datei | Beschreibung |
|---------|-------|--------------|
| `ABLY_API_KEY` | `.env` | Ably Root API Key (kostenlos) |
| `CONTROL_API_TOKEN` | `.env` | Bearer Token für ControlServer-Auth |
| `CONTROL_PORT` | `.env` | ControlServer-Port (Standard: 8765) |
| `CONTROL_SERVER_URL` | Colab ENV | URL des Cloudflare Tunnels (in Colab setzen) |
| `TELEGRAM_BOT_TOKEN` | `.env` | Bot-Token für Telegram-Transport (bereits in .env) |
| `TELEGRAM_CHAT_ID` | `.env` | Telegram Chat-ID (bereits in .env, auskommentiert) |

---

## 11. Abhängigkeiten

### Pflicht (lokal)
```bash
pip install ably ccxt numpy pandas loguru
```

### Pflicht (Colab, Module B)
```bash
pip install ably loguru numpy
```

### Optional (für jeweilige Features)

| Paket | Feature |
|-------|---------|
| `fastapi uvicorn` | ControlServer |
| `httpx` | ControlClient + Reporter HTTP |
| `torch` | PyTorch-Modelle + GPU-Monitor |
| `stable-baselines3` | SB3-Modelle |
| `redis` | Redis-Transport |
| `cloudflared` | Cloudflare Tunnel (CLI-Tool, kein Python-Paket) |

### Graceful Degradation

Alle optionalen Pakete werden mit `try/except ImportError` behandelt. Das System läuft auch ohne sie — nur die entsprechenden Features sind deaktiviert.

---

## 12. Copy-Paste Colab-Zellen

### Zelle 1: Einmalig ausführen (Setup)

```python
# ─── bt4t Extension Setup ──────────────────────────────────────────
import sys, os
from google.colab import drive
drive.mount('/content/drive')
sys.path.insert(0, '/content/drive/MyDrive/BITCOIN4Traders')

# Konfiguration setzen (URL aus lokalem Terminal kopieren)
os.environ.setdefault('BT4T_LISTENER_URL', 'https://DEINE_URL.trycloudflare.com')
os.environ.setdefault('BT4T_API_TOKEN',    'bt4t-secret-token')
os.environ.setdefault('BT4T_NOTEBOOK_ID',  'training_v1')
os.environ.setdefault('ABLY_API_KEY',      'DEIN_ABLY_KEY')

from colab_bridge.colab_extension import bt4t
bt4t.install()
print("Setup abgeschlossen:", bt4t.status())
```

### Zelle 2: Module B starten

```python
# ─── Module B (RL-Inferenz) starten ────────────────────────────────
from colab_bridge.module_b_colab import ModuleB, find_champion_on_drive

champion_path = find_champion_on_drive('/content/drive/MyDrive/BITCOIN4Traders')

engine = ModuleB(
    ably_key=os.environ['ABLY_API_KEY'],
    model_path=champion_path,
    symbol='BTCUSDT',
    min_confidence=0.55,
)
await engine.run()
```

### Zelle 3: RL-Training mit bt4t.step() (Muster)

```python
# ─── Training mit Extension-Überwachung ────────────────────────────
from colab_bridge.colab_extension import bt4t

# Normaler Training-Code — nur bt4t.step() hinzufügen
for epoch in range(1000):
    loss = train_one_epoch(model, optimizer, data)
    reward = evaluate(model, env)

    # bt4t.step() gibt False zurück wenn STOP-Befehl empfangen
    if not bt4t.step(epoch=epoch, loss=float(loss), reward=float(reward)):
        print("Training vom lokalen Rechner gestoppt")
        break

    # Checkpoint alle 100 Epochs
    if epoch % 100 == 0:
        save_path = f"/content/drive/MyDrive/.../model_ep{epoch}.pth"
        torch.save(model.state_dict(), save_path)
        bt4t.send_checkpoint(save_path, {"loss": loss, "reward": reward})
```

### Zelle 4: Training mit Decorator

```python
# ─── Training mit bt4t.guard Decorator ────────────────────────────
from colab_bridge.colab_extension import bt4t

@bt4t.guard   # Automatischer Retry + Fehlerbehandlung
def run_full_training():
    for epoch in range(1000):
        loss = train_one_epoch(model, optimizer, data)
        bt4t.step(epoch=epoch, loss=float(loss))

run_full_training()
```

### Zelle 5: Training mit Context-Manager

```python
# ─── Training mit Context-Manager ─────────────────────────────────
from colab_bridge.colab_extension import bt4t

with bt4t.session("experiment_v3_bollinger"):
    for epoch in range(1000):
        loss = train_one_epoch(model, optimizer, data)
        if not bt4t.step(epoch=epoch, loss=float(loss)):
            break
```

### Zelle 6: Befehl vom Colab aus senden

```python
# ─── Status abrufen ────────────────────────────────────────────────
print(bt4t.status())
print("Reparaturen:", bt4t.repair_log())

# Manuelle Warnung senden
bt4t.send_alert("Epoch 500 erreicht — bitte Metriken prüfen", level="INFO")
```

---

## 13. Troubleshooting

### "Kein Ably API Key"

```
FEHLER: Kein Ably API Key!
```

**Lösung:** `ABLY_API_KEY` setzen:
```bash
# In .env:
ABLY_API_KEY=root.XXXXX:YYYYY

# In Colab:
os.environ['ABLY_API_KEY'] = 'root.XXXXX:YYYYY'
```

---

### Colab empfängt keine Marktdaten

**Symptom:** `obs_buffer_size` bleibt 0, `last_market_data_age_s` steigt.

**Checklist:**
1. Läuft Module A lokal? `python colab_bridge/module_a_local.py`
2. Gleicher Ably-Key auf beiden Seiten?
3. Symbol-Format korrekt? Module A: `BTC/USDT`, Module B: `BTCUSDT`
4. Ably-Verbindung aktiv? Siehe `connected` Log-Meldung

---

### Lokaler Rechner empfängt keine Signale

**Symptom:** Kein "Signal empfangen" im Module-A-Log.

**Checklist:**
1. Läuft Module B in Colab?
2. Reicht Confidence-Schwelle? Aktuell: 0.55 (senken mit `--min-conf 0.4`)
3. Reicht Observation-Buffer? Inferenz startet erst bei ≥ 20 Einträgen
4. Ably-Kanal korrekt? `bt4t:signals`

---

### Trading ist pausiert obwohl Colab läuft

**Symptom:** "Portfolio: Trading PAUSIERT (kein Colab-Heartbeat)"

**Ursache:** Kein Heartbeat von Colab in den letzten 90 Sekunden.

**Lösung:**
```python
# In Colab: Heartbeat-Intervall prüfen (Standard 10s)
# Module B läuft? await engine.run() ausführen

# Manuell fortsetzen (HTTP):
curl -X POST http://localhost:8765/trading/resume \
  -H "Authorization: Bearer bt4t-secret-token"
```

---

### bt4t Extension sendet keine Berichte

**Symptom:** "HTTP fehlgeschlagen" + "Drive Fallback"

**Checklist:**
1. `BT4T_LISTENER_URL` gesetzt und korrekt?
2. Cloudflare Tunnel läuft lokal? (`./cloudflared tunnel --url http://localhost:8765`)
3. ControlServer läuft? (`python colab_bridge/control_plane.py server`)
4. Falls Drive-Fallback genügt: Meldungen in `/content/drive/MyDrive/BITCOIN4Traders/bt4t_bus/reports/`

---

### OOM trotz MemoryMonitor

**Symptom:** Training bricht mit OOM ab obwohl Monitor läuft.

**Ursache:** Monitor prüft nur alle 60s. OOM kann dazwischen auftreten.

**Lösung:** `bt4t.guard` oder `bt4t.session` zusätzlich verwenden — der ExceptionHook fängt OOM und halbiert `batch_size` im laufenden Prozess.

---

### "Extension bereits installiert"

```
WARNING | [bt4t] Extension bereits installiert — überspringe
```

Kein Fehler. `bt4t.install()` ist idempotent — mehrfache Aufrufe sind sicher.

---

### Reparatur schlägt fehl ("keine batch_size Variable gefunden")

**Ursache:** Batch-Size-Variable hat unüblichen Namen oder ist eine lokale Variable.

**Lösung:** Variable umbenennen:
```python
# Statt:
bs = 32
# Verwende:
BATCH_SIZE = 32   # oder batch_size = 32
```

Oder manuell nach Reparatur anpassen:
```python
# Nach InProcessRepair: prüfe ob Änderung gewünscht
print(bt4t.repair_log())
```

---

*Dokumentation generiert: 2026-03-12 | BITCOIN4Traders v1.0.0*
