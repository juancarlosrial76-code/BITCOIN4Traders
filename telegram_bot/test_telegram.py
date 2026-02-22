"""
Telegram Bot Test-Skript
========================
Führe dieses Skript aus, sobald du Token und Chat-ID hast.

Verwendung:
    python3 telegram_bot/test_telegram.py

Oder mit manuellen Werten:
    TELEGRAM_BOT_TOKEN=123:ABC TELEGRAM_CHAT_ID=456 python3 telegram_bot/test_telegram.py
"""

import os
import sys

# Projektroot zum Python-Pfad hinzufügen
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# .env laden falls vorhanden (pip install python-dotenv)
try:
    from dotenv import load_dotenv

    load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
    print("  .env Datei geladen")
except ImportError:
    pass  # python-dotenv nicht installiert - env vars direkt nutzen

from darwin_engine import TelegramNotifier


def main():
    print("=" * 55)
    print("  BITCOIN4Traders — Telegram Bot Test")
    print("=" * 55)

    token = os.getenv("TELEGRAM_BOT_TOKEN", "")
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "")

    # Prüfung
    if not token:
        print("\nFEHLER: TELEGRAM_BOT_TOKEN nicht gesetzt.")
        print("  Entweder in .env Datei eintragen oder:")
        print("  export TELEGRAM_BOT_TOKEN=<dein_token>")
        sys.exit(1)

    if not chat_id:
        print("\nFEHLER: TELEGRAM_CHAT_ID nicht gesetzt.")
        print("  Entweder in .env Datei eintragen oder:")
        print("  export TELEGRAM_CHAT_ID=<deine_chat_id>")
        sys.exit(1)

    print(f"\n  Token   : {token[:10]}...{token[-4:]}")
    print(f"  Chat-ID : {chat_id}")

    notifier = TelegramNotifier(token=token, chat_id=chat_id)

    # Test 1: Einfache Textnachricht
    print("\n[Test 1] Sende einfache Nachricht...")
    ok = notifier.send(
        "BITCOIN4Traders Bot ist aktiv!\nVerbindung erfolgreich getestet."
    )
    print(f"  Ergebnis: {'OK' if ok else 'FEHLER'}")

    # Test 2: Signal-Nachricht (wie im echten Betrieb)
    print("\n[Test 2] Sende Signal-Nachricht...")
    ok = notifier.send_signal(
        champion_name="RSI_p14_l30_u70",
        signal=1,  # 1 = LONG, -1 = SHORT, 0 = FLAT
        price=45_230.50,
        environment="LOCAL",
    )
    print(f"  Ergebnis: {'OK' if ok else 'FEHLER'}")

    # Test 3: Champion-Update (nach Evolution)
    print("\n[Test 3] Sende Champion-Update...")
    ok = notifier.send_champion_update(
        champion_name="MV_Gen8_MACD_f10_s24_sig7",
        mv_score=12.45,
        survival_rate=0.94,
        worst_dd=0.12,
    )
    print(f"  Ergebnis: {'OK' if ok else 'FEHLER'}")

    # Test 4: Warnung / Alert
    print("\n[Test 4] Sende Warnung...")
    ok = notifier.send_alert(
        title="Test-Warnung",
        message="Dies ist ein Test-Alert. Im echten Betrieb erscheint\n"
        "hier z.B.: GitHub Actions Heartbeat ausgefallen.",
    )
    print(f"  Ergebnis: {'OK' if ok else 'FEHLER'}")

    print("\n" + "=" * 55)
    print("  Alle Tests abgeschlossen.")
    print("  Prüfe dein Telegram ob 4 Nachrichten ankamen.")
    print("=" * 55)


if __name__ == "__main__":
    main()
