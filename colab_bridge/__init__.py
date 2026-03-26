"""
colab_bridge — Communication bridge between local machine and Colab/Cloud
=========================================================================

Install:
    pip install colab-rl-bridge               # Base
    pip install colab-rl-bridge[local]        # + ably + ccxt (local machine)
    pip install colab-rl-bridge[server]       # + fastapi + uvicorn (Control Server)
    pip install colab-rl-bridge[all]          # Everything

Quick start (Colab):
    from colab_bridge.colab_extension import bt4t
    bt4t.install()

Quick start (local):
    python -m colab_bridge.module_a_local --ably-key $ABLY_API_KEY

Documentation:
    colab_bridge/DOKUMENTATION.md
    https://github.com/juancarlosrial76-code/BITCOIN4Traders/blob/main/colab_bridge/DOKUMENTATION.md

Contains:
  module_a_local   : Local execution engine (market data + paper orders)
  module_b_colab   : Colab RL inference engine (model + signal publisher)
  control_plane    : FastAPI Control Server + Colab Control Client
  colab_extension  : Colab Extension (bt4t singleton)
  transport_base   : Abstract transport interface
  transports/      : 4 transport options (Ably, Redis, Telegram, Google Drive)
"""

__version__ = "1.0.0"
__author__ = "juancarlosrial76-code"
__license__ = "MIT"

# Public API — available via: from colab_bridge import ...
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
    # Classes
    "BT4TExtension",
    "TransportBase",
    # Functions
    "classify_error",
    # Channel constants
    "CH_MARKET",
    "CH_SIGNALS",
    "CH_PORTFOLIO",
    "CH_HEALTH",
    "CH_CONTROL",
    "CH_ACK",
]
