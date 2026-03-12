"""
colab_bridge — Kommunikations-Bridge zwischen lokalem Rechner und Colab/Cloud
=============================================================================

Installieren:
    pip install colab-rl-bridge               # Basis
    pip install colab-rl-bridge[local]        # + ably + ccxt (lokaler Rechner)
    pip install colab-rl-bridge[server]       # + fastapi + uvicorn (Control-Server)
    pip install colab-rl-bridge[all]          # Alles

Schnellstart (Colab):
    from colab_bridge.colab_extension import bt4t
    bt4t.install()

Schnellstart (lokal):
    python -m colab_bridge.module_a_local --ably-key $ABLY_API_KEY

Dokumentation:
    colab_bridge/DOKUMENTATION.md
    https://github.com/juancarlosrial76-code/BITCOIN4Traders/blob/main/colab_bridge/DOKUMENTATION.md

Enthält:
  module_a_local   : Lokale Ausführungs-Engine (Marktdaten + Paper-Order)
  module_b_colab   : Colab RL-Inferenz-Engine (Modell + Signal-Publisher)
  control_plane    : FastAPI Control Server + Colab Control Client
  colab_extension  : Colab Extension (bt4t Singleton)
  transport_base   : Abstraktes Transport-Interface
  transports/      : 4 Transport-Optionen (Ably, Redis, Telegram, Google Drive)
"""

__version__ = "1.0.0"
__author__ = "juancarlosrial76-code"
__license__ = "MIT"

# Öffentliche API — verfügbar nach: from colab_bridge import ...
from colab_bridge.transport_base import (
    TransportBase,
    CH_MARKET,
    CH_SIGNALS,
    CH_PORTFOLIO,
    CH_HEALTH,
    CH_CONTROL,
    CH_ACK,
)
from colab_bridge.colab_extension import bt4t, BT4TExtension, classify_error

__all__ = [
    # Singleton
    "bt4t",
    # Klassen
    "BT4TExtension",
    "TransportBase",
    # Funktionen
    "classify_error",
    # Kanal-Konstanten
    "CH_MARKET",
    "CH_SIGNALS",
    "CH_PORTFOLIO",
    "CH_HEALTH",
    "CH_CONTROL",
    "CH_ACK",
]
