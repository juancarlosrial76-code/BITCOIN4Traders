"""
Live Feature Engine – Phase 5
===============================
Berechnet Features aus Live-Tick-Daten in Echtzeit.

Hält einen rollenden OHLCV-Puffer pro Symbol (lookback=100 Stunden).
Sobald genug History vorhanden ist, liefert transform_single() einen
Feature-Vektor der denselben 34 Dimensionen entspricht wie im Training
(Phase 3: 26 Standard + 8 Multi-TF).

Wichtig: kein Scaler-Fit auf Live-Daten (würde Data Leakage erzeugen).
Stattdessen: z-score Normierung mit rollenden Statistiken.
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("phase7.live_features")

# Minimale History bevor Features berechnet werden können
MIN_HISTORY = 30  # 30 Kerzen (reduziert für schnellere Feature-Berechnung)


class LiveFeatureEngine:
    """
    Echtzeit-Feature-Engine für Live- und Paper-Trading.
    Drop-in Ersatz für StubFeatureEngine in run.py.
    """

    def __init__(self, lookback: int = 120):
        """
        Parameters
        ----------
        lookback : int
            Maximale OHLCV-History pro Symbol (in Ticks/Stunden)
        """
        self._lookback = lookback
        # Puffer: symbol → deque von (timestamp, open, high, low, close, volume)
        self._buffers: Dict[str, deque] = {}
        # Letzte berechnete "Stunden-Kerze" pro Symbol
        self._candles: Dict[str, Dict] = {}
        self._tick_count: Dict[str, int] = {}
        logger.info(f"[LiveFeature] Engine initialisiert (lookback={lookback})")

    def on_price_update(self, symbol: str, price: float, volume: float = 1.0) -> None:
        """
        Tick-Update aufnehmen.
        Im Paper-Trading: jeder Bid/Ask-Tick zählt als Preis-Update.
        """
        if symbol not in self._buffers:
            self._buffers[symbol] = deque(maxlen=self._lookback)
            self._tick_count[symbol] = 0

        self._tick_count[symbol] += 1

        # Jede 10. Tick als synthetische "Kerze" hinzufügen
        # (Live: normalerweise echte 1h-Kerzen, hier synthetisch für Paper-Mode)
        buf = self._buffers[symbol]
        if len(buf) == 0 or self._tick_count[symbol] % 10 == 0:
            buf.append(
                {
                    "open": price,
                    "high": price,
                    "low": price,
                    "close": price,
                    "volume": volume,
                }
            )
        else:
            # Kerze aktualisieren
            c = buf[-1]
            buf[-1] = {
                "open": c["open"],
                "high": max(c["high"], price),
                "low": min(c["low"], price),
                "close": price,
                "volume": c["volume"] + volume,
            }

    def transform_single(self, symbol: str, price: float) -> Optional[np.ndarray]:
        """
        Berechnet den Feature-Vektor für den aktuellen Zeitpunkt.

        Returns None wenn noch nicht genug History vorhanden.
        Returns np.ndarray der Form (34,) sobald MIN_HISTORY erreicht.
        """
        # Puffer aktualisieren
        self.on_price_update(symbol, price)

        buf = self._buffers.get(symbol)
        if buf is None or len(buf) < MIN_HISTORY:
            return None

        # DataFrame aus Puffer
        df = pd.DataFrame(list(buf))
        close = df["close"]
        vol = df["volume"]

        feats = {}

        # ── Standard-Features (26) ────────────────────────────────────────────

        # 1. Log Return
        feats["log_ret"] = np.log(close.iloc[-1] / close.iloc[-2] + 1e-9)

        # 2. Volatilität (20, 50)
        log_r = np.log(close / close.shift(1) + 1e-9)
        feats["volatility_20"] = float(log_r.rolling(20).std().iloc[-1] * np.sqrt(252 * 24))
        feats["volatility_50"] = float(
            log_r.rolling(min(50, len(df))).std().iloc[-1] * np.sqrt(252 * 24)
        )

        # 3. Rolling Mean / Std
        feats["rolling_mean"] = float(close.rolling(20).mean().iloc[-1])
        feats["rolling_std"] = float(close.rolling(20).std().iloc[-1])

        # 4. RSI-14
        feats["rsi_14"] = float(self._rsi(close, 14).iloc[-1])

        # 5. MACD
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        sig = macd.ewm(span=9, adjust=False).mean()
        feats["macd"] = float(macd.iloc[-1])
        feats["macd_signal"] = float(sig.iloc[-1])
        feats["macd_hist"] = float((macd - sig).iloc[-1])

        # 6. Bollinger Bands
        bb_mid = close.rolling(20).mean()
        bb_std = close.rolling(20).std()
        bb_up = bb_mid + 2 * bb_std
        bb_lo = bb_mid - 2 * bb_std
        band_w = (bb_up - bb_lo) / (close + 1e-9)
        feats["bb_width"] = float(band_w.iloc[-1])
        feats["bb_position"] = float(((close - bb_lo) / (bb_up - bb_lo + 1e-9)).iloc[-1])

        # 7. OU-Score (Ornstein-Uhlenbeck Mean-Reversion)
        mu = close.rolling(20).mean().iloc[-1]
        sd = close.rolling(20).std().iloc[-1]
        feats["ou_score"] = float((close.iloc[-1] - mu) / (sd + 1e-9))

        # Fehlende Standard-Features auf 0 auffüllen (Observation-Space = 26)
        standard_keys = [
            "log_ret",
            "volatility_20",
            "volatility_50",
            "rolling_mean",
            "rolling_std",
            "rsi_14",
            "macd",
            "macd_signal",
            "macd_hist",
            "bb_width",
            "bb_position",
            "ou_score",
        ]
        # Weitere 14 Werte (Portfolio-State etc.) → 0 im Paper-Mode
        std_vec = np.array([feats.get(k, 0.0) for k in standard_keys], dtype=np.float32)
        pad_vec = np.zeros(14, dtype=np.float32)  # Portfolio + Risk-State
        standard_26 = np.concatenate([std_vec, pad_vec])  # 26-dim

        # ── Multi-TF Features (8) ──────────────────────────────────────────────
        # 4h: verwende die letzten 4 Kerzen als "4h-Kerze"
        mtf_vec = np.zeros(8, dtype=np.float32)
        if len(df) >= 8:
            # 4h-Block: letzte 4 Candles
            c4 = close.iloc[-4:]
            ema_f4 = c4.ewm(span=3, adjust=False).mean()
            trend_4h = float(np.sign(ema_f4.iloc[-1] - ema_f4.iloc[0]))
            rsi_4h = float(self._rsi(close, 4).iloc[-1])
            mh_4h = float(
                (close.ewm(4, adjust=False).mean() - close.ewm(8, adjust=False).mean()).iloc[-1]
            )
            vol_4h = float(log_r.rolling(4).std().iloc[-1] * np.sqrt(252 * 24))

            # 1d-Block: letzte 24 Candles
            n1d = min(24, len(df))
            c1d = close.iloc[-n1d:]
            ema_f1d = c1d.ewm(span=6, adjust=False).mean()
            trend_1d = float(np.sign(ema_f1d.iloc[-1] - ema_f1d.iloc[0]))
            rsi_1d = float(self._rsi(close, min(14, n1d)).iloc[-1])
            mh_1d = float(
                (close.ewm(12, adjust=False).mean() - close.ewm(24, adjust=False).mean()).iloc[-1]
            )
            vol_1d = float(log_r.rolling(min(24, len(df))).std().iloc[-1] * np.sqrt(252 * 24))

            mtf_vec = np.array(
                [
                    trend_4h,
                    rsi_4h,
                    mh_4h,
                    vol_4h,
                    trend_1d,
                    rsi_1d,
                    mh_1d,
                    vol_1d,
                ],
                dtype=np.float32,
            )

        # ── Kombination: 26 + 8 = 34 Features ────────────────────────────────
        feature_vec = np.concatenate([standard_26, mtf_vec])

        # NaN/Inf absichern
        feature_vec = np.nan_to_num(feature_vec, nan=0.0, posinf=5.0, neginf=-5.0)
        feature_vec = np.clip(feature_vec, -10.0, 10.0)

        return feature_vec

    @staticmethod
    def _rsi(series: pd.Series, window: int) -> pd.Series:
        delta = series.diff()
        gain = delta.clip(lower=0).rolling(window).mean()
        loss = (-delta.clip(upper=0)).rolling(window).mean()
        rs = gain / (loss + 1e-9)
        return 100 - 100 / (1 + rs)
