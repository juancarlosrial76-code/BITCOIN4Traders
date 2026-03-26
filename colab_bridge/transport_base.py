"""
transport_base.py — Shared transport interface
===============================================
All 4 communication options implement this interface.
This allows Module A and Module B to switch the transport at any time
without changing the trading logic code.

Channel conventions (identical for all transports):
  bt4t:market:BTCUSDT   — Market data Local → Colab
  bt4t:signals          — Trade signals Colab → Local
  bt4t:portfolio:state  — Portfolio state Local → Colab
  bt4t:health           — Heartbeat Colab → Local
  bt4t:control:cmd      — Control commands Local → Colab
  bt4t:control:ack      — Acknowledgement Colab → Local
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional

# ── Channel constants ─────────────────────────────────────────────────────────
CH_MARKET = "bt4t:market:{symbol}"
CH_SIGNALS = "bt4t:signals"
CH_PORTFOLIO = "bt4t:portfolio:state"
CH_HEALTH = "bt4t:health"
CH_CONTROL = "bt4t:control:cmd"
CH_ACK = "bt4t:control:ack"


class TransportBase(ABC):
    """
    Abstract base class for all communication transports.

    Each transport implements:
      connect()     — Establish connection
      disconnect()  — Cleanly disconnect
      publish()     — Send message on channel
      subscribe()   — Register callback for incoming messages
    """

    @abstractmethod
    async def connect(self) -> None:
        """Establish connection. Raises exception on error."""
        ...

    @abstractmethod
    async def disconnect(self) -> None:
        """Cleanly disconnect."""
        ...

    @abstractmethod
    async def publish(self, channel: str, payload: dict) -> None:
        """
        Publish message on channel.

        Parameters
        ----------
        channel : str   Channel name, e.g. 'bt4t:signals'
        payload : dict  JSON-serializable dict
        """
        ...

    @abstractmethod
    async def subscribe(self, channel: str, callback: Callable[[dict], None]) -> None:
        """
        Register callback for incoming messages on channel.

        The callback is called with the deserialized payload dict.

        Parameters
        ----------
        channel  : str      Channel name
        callback : callable Function(payload: dict) -> None
        """
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the transport for logging."""
        ...

    @property
    @abstractmethod
    def latency_class(self) -> str:
        """Expected latency class: 'ms', 'sub-second', 'seconds', 'minutes'"""
        ...

    # ── Helper methods ────────────────────────────────────────────────────────

    @staticmethod
    def encode(payload: dict) -> str:
        return json.dumps(payload, default=str)

    @staticmethod
    def decode(data: str | bytes | dict) -> dict:
        if isinstance(data, dict):
            return data
        if isinstance(data, bytes):
            data = data.decode("utf-8")
        return json.loads(data)
