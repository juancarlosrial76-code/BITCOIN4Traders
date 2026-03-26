# BITCOIN4Traders — Transport Guide

## Communication Local ↔ Colab/Cloud

---

## Comparison Table

|                          | Option 1: Redis      | Option 2: Telegram      | Option 3: Google Drive | Option 4: Ably      |
| ------------------------ | -------------------- | ----------------------- | ---------------------- | ------------------- |
| **Latency**              | 30–150 ms            | 200–800 ms              | 2–15 seconds           | 50–150 ms           |
| **Cost**                 | $0                   | $0                      | $0                     | $0 (Free Tier)      |
| **External Account**     | No                   | Telegram (existing)     | Google (existing)      | Ably (new)          |
| **Setup Effort**         | Medium               | Low                     | Low                    | Low                 |
| **Local Infrastructure** | Redis + CF Tunnel    | Nothing                 | drive_manager.py       | Nothing             |
| **For Timeframe**        | 1m–1h                | 1h+                     | 1h+                    | 1m–1h               |
| **Reliability**          | Very high            | Very high               | High                   | Very high           |
| **Offline Buffer**       | Yes (Redis List)     | No                      | Yes (File)             | Yes (100 Msgs)      |
| **Human Readable**       | No                   | Yes                     | Yes (JSON)             | No                  |
| **File**                 | `transport_redis.py` | `transport_telegram.py` | `transport_gdrive.py`  | `transport_ably.py` |

---

## Option 1: Redis + Cloudflare Tunnel

**Recommended for: Sub-second to 1m timeframes**

### How it works

```
Exchange API → Local (Redis) → HTTP Proxy :8766 → Cloudflare Tunnel → Colab (httpx poll)
Colab (Inference) → HTTP Proxy → Redis PUBLISH → Local (Callback)
```

Redis runs locally. Colab communicates via HTTP with a FastAPI proxy,
which is made accessible by Cloudflare Tunnel. No Redis client needed in Colab.

### Setup

**1. Install Redis (local, one-time):**

```bash
sudo apt install redis-server    # Ubuntu/Debian
# or
brew install redis               # macOS

redis-server --daemonize yes     # Start
redis-cli ping                   # Test → PONG
```

**2. Install Cloudflare Tunnel:**

```bash
# Ubuntu/Debian:
curl -fsSL https://pkg.cloudflare.com/cloudflare-main.gpg | \
  sudo tee /usr/share/keyrings/cloudflare-main.gpg > /dev/null
echo 'deb [signed-by=/usr/share/keyrings/cloudflare-main.gpg] \
  https://pkg.cloudflare.com/cloudflared any main' | \
  sudo tee /etc/apt/sources.list.d/cloudflared.list
sudo apt-get update && sudo apt-get install cloudflared

# Or directly:
wget -O cloudflared https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64
chmod +x cloudflared
```

**3. Python packages:**

```bash
pip install redis fastapi uvicorn httpx
```

**4. Start:**

```bash
# Terminal 1 — Redis
redis-server

# Terminal 2 — Cloudflare Tunnel
./cloudflared tunnel --url http://localhost:8766
# Outputs: https://abc-xyz.trycloudflare.com  ← save this URL!

# Terminal 3 — Module A (local) with Redis transport
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

### Latency Details

- Redis PUBLISH local: ~1ms
- Cloudflare Tunnel overhead: ~30–100ms (depending on location)
- Colab HTTP poll interval: 1s (configurable)
- **Effective signal latency: 30–150ms** (at 1s polling)

---

## Option 2: Telegram Bot API

**Recommended for: 1h+ timeframes, already integrated in project**

### How it works

```
Local → Telegram API → Telegram Server → Colab (getUpdates long-poll)
```

Both sides use the same bot token. Messages are sent as
Telegram messages with hashtag encoding:
`#bt4t:signals {"action":"BUY","confidence":0.82,...}`

### Setup

**1. Create Telegram Bot (if not already done):**

```
1. t.me/BotFather → /newbot
2. Choose name and username
3. Save token: 8512251150:AAFXc_...
```

**2. Get chat ID:**

```
1. t.me/userinfobot → gives your chat ID
   OR: Start bot → t.me/getidsbot → chat ID
```

**3. Activate in .env:**

```bash
# .env
TELEGRAM_BOT_TOKEN=8512251150:AAFXc_7dGvRXmnEKGv9_9bXYSCxuPijPen8
TELEGRAM_CHAT_ID=2028041322
```

_(These values are already in .env, just remove the #)_

**4. Start (local):**

```python
from colab_bridge.transports import get_transport
transport = get_transport("telegram")  # reads from .env
```

**5. In Colab:**

```python
import os
os.environ["TELEGRAM_BOT_TOKEN"] = "your_token"
os.environ["TELEGRAM_CHAT_ID"] = "your_chat_id"

from colab_bridge.transports import get_transport
transport = get_transport("telegram")
```

### Characteristics

- **Long-Polling**: Telegram getUpdates waits up to 20s → effectively real-time notification
- **Rate limit**: 30 msg/s to one chat (uncritical at 30s interval)
- **Manual control**: You can send `/pause`, `/resume` directly in the Telegram chat
- **Security**: Messages are on Telegram servers — don't send API keys or account balances

---

## Option 3: Google Drive

**Recommended for: 1h+ timeframes, when Colab already uses Drive**

### How it works

```
Local → Write file → drive_manager.py syncs → Google Drive → Colab reads file
```

Each channel corresponds to a JSON file on Drive:

```
bt4t_bus/
  market/BTCUSDT_latest.json    ← Local writes, Colab reads
  signals/latest.json           ← Colab writes, Local reads
  health/heartbeat.json         ← Colab writes
  control/cmd.json              ← Local writes
```

### Setup

**1. Local — configure drive_manager.py:**

```bash
# drive_manager.py is already in the project (infrastructure/drive/drive_manager.py)
# Sync directory in .env:
BT4T_DRIVE_SYNC_DIR=data/drive_sync
```

**2. Start (local):**

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

### Latency Details

- Drive sync interval: 2–15s (depending on Google's sync frequency)
- **Only suitable for 1h bars** (15m bars borderline)

---

## Option 4: Ably

**Recommended for: Easiest setup without own infrastructure**

### How it works

```
Local → Ably Server (WebSocket) ← Colab
```

Both sides connect directly to the Ably server.
No tunnel, no proxy, no local infrastructure needed.

### Setup

**1. Create Ably account:**

```
1. https://ably.com → Sign Up (free)
2. Dashboard → Create App → API Keys
3. Copy Root Key: xxxxx:yyyyy
```

**2. In .env:**

```bash
ABLY_API_KEY=xxxxx:yyyyy
```

**3. Python package:**

```bash
pip install ably   # already installed
```

**4. Start (local + Colab identical):**

```python
import os
from colab_bridge.transports import get_transport
transport = get_transport("ably", api_key=os.getenv("ABLY_API_KEY"))
```

### Free Tier Limits

- 6 million messages/month
- At 30s poll interval, 5 channels: ~86,400 msg/month → **well below limit**
- Message history: last 100 per channel (for session recovery)

---

## Switching Transports (without code changes)

All transports implement the same interface (`TransportBase`).
Module A and Module B accept any transport:

```python
from colab_bridge.transports import get_transport
from colab_bridge.module_a_local import ModuleA

# Simply switch transport:
transport = get_transport("redis")       # Option 1
# transport = get_transport("telegram")  # Option 2
# transport = get_transport("gdrive")    # Option 3
# transport = get_transport("ably")      # Option 4

# Module A and B receive the transport
engine = ModuleA(transport=transport, ...)
```

---

## Recommendation by Use Case

| Use Case                 | Recommended Transport | Why                          |
| ------------------------ | --------------------- | ---------------------------- |
| 1h bars, quick start     | **Telegram**          | Already in project, no setup |
| 15m bars, no account     | **Redis**             | Fastest option, full control |
| 5m–1m bars               | **Redis**             | Only option under 500ms      |
| Colab already uses Drive | **Google Drive**      | No new infrastructure        |
| Simplest setup           | **Ably**              | No tunnel, just API key      |
| Backup/fallback          | **Telegram**          | Always available             |

---

## File Structure

```
colab_bridge/
├── __init__.py
├── transport_base.py          # Shared interface (ABC)
├── module_a_local.py          # Local execution engine
├── module_b_colab.py          # Colab RL inference engine
├── control_plane.py           # FastAPI Control Server + Client
├── TRANSPORT_GUIDE.md         # This file
└── transports/
    ├── __init__.py            # Factory function get_transport()
    ├── transport_redis.py     # Option 1: Redis + Cloudflare Tunnel
    ├── transport_telegram.py  # Option 2: Telegram Bot API
    ├── transport_gdrive.py    # Option 3: Google Drive
    └── transport_ably.py      # Option 4: Ably Pub/Sub
```
