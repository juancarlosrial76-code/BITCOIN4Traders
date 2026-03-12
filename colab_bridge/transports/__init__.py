"""
transports/ — Alle 4 Kommunikations-Transporte

Option 1: transport_redis    Redis + Cloudflare Tunnel  (30–150ms, keine externen Accounts)
Option 2: transport_telegram  Telegram Bot API           (200–800ms, bereits integriert)
Option 3: transport_gdrive    Google Drive               (2–15s, bereits im Projekt)
Option 4: transport_ably      Ably Pub/Sub               (50–150ms, Account nötig)
"""

from colab_bridge.transports.transport_ably import AblyTransport
from colab_bridge.transports.transport_telegram import TelegramTransport
from colab_bridge.transports.transport_gdrive import (
    DriveTransportLocal,
    DriveTransportColab,
)
from colab_bridge.transports.transport_redis import (
    RedisTransportLocal,
    RedisTransportColab,
)

__all__ = [
    "AblyTransport",
    "TelegramTransport",
    "DriveTransportLocal",
    "DriveTransportColab",
    "RedisTransportLocal",
    "RedisTransportColab",
]


def get_transport(option: str, side: str = "local", **kwargs):
    """
    Factory-Funktion: Gibt den passenden Transport zurück.

    Parameters
    ----------
    option : str   'redis' | 'telegram' | 'gdrive' | 'ably'
    side   : str   'local' | 'colab' (nur für redis und gdrive relevant)
    **kwargs       Werden an den Transport-Konstruktor weitergegeben

    Beispiel:
        t = get_transport("ably", api_key="xxx")
        t = get_transport("redis", side="local")
        t = get_transport("telegram")
        t = get_transport("gdrive", side="colab")
    """
    option = option.lower()
    side = side.lower()

    if option == "ably":
        return AblyTransport(**kwargs)
    elif option == "telegram":
        return TelegramTransport(**kwargs)
    elif option == "gdrive":
        if side == "colab":
            return DriveTransportColab(**kwargs)
        return DriveTransportLocal(**kwargs)
    elif option == "redis":
        if side == "colab":
            return RedisTransportColab(**kwargs)
        return RedisTransportLocal(**kwargs)
    else:
        raise ValueError(
            f"Unbekannte Transport-Option: '{option}'. Wähle: redis, telegram, gdrive, ably"
        )
