"""
Multi-Timeframe Analysis Module
================================
SOTA feature: Aggregates signals across multiple timeframes
for robust trading decisions.

Professional traders use multi-timeframe analysis to:
1. Identify major trends (higher timeframe)
2. Time entries (lower timeframe)
3. Confirm signals (multiple timeframes align)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from loguru import logger


class Timeframe(Enum):
    """Standard trading timeframes."""

    M1 = "1m"  # 1 minute
    M5 = "5m"  # 5 minutes
    M15 = "15m"  # 15 minutes
    M30 = "30m"  # 30 minutes
    H1 = "1h"  # 1 hour
    H4 = "4h"  # 4 hours
    D1 = "1d"  # Daily
    W1 = "1w"  # Weekly


@dataclass
class SignalStrength:
    """Signal strength across timeframes."""

    timeframe: Timeframe
    direction: int  # -1, 0, 1
    strength: float  # 0.0 to 1.0
    confidence: float  # 0.0 to 1.0


class MultiTimeframeAnalyzer:
    """
    Multi-timeframe signal aggregator.

    Implements the "Top-Down Analysis" approach used by professional traders:
    - Weekly: Market structure and major trends
    - Daily: Trade direction (bias)
    - 4H: Key levels and zones
    - 1H: Entry/exit timing
    """

    def __init__(self, timeframes: List[Timeframe] = None):
        """
        Initialize multi-timeframe analyzer.

        Args:
            timeframes: List of timeframes to analyze
        """
        self.timeframes = timeframes or [Timeframe.H4, Timeframe.H1, Timeframe.M15]
        self.signals: Dict[Timeframe, SignalStrength] = {}
        logger.info(
            f"MultiTimeframeAnalyzer initialized: {[tf.value for tf in self.timeframes]}"
        )

    def calculate_trend_alignment(self, data: Dict[Timeframe, pd.DataFrame]) -> Dict:
        """
        Calculate trend alignment across timeframes.

        Returns:
            Dictionary with alignment metrics
        """
        signals = []

        for tf in self.timeframes:
            if tf not in data:
                continue

            df = data[tf]
            signal = self._analyze_single_timeframe(df, tf)
            signals.append(signal)

        # Calculate alignment
        if not signals:
            return {"alignment": 0.0, "direction": 0, "confidence": 0.0}

        directions = [s.direction for s in signals]
        strengths = [s.strength for s in signals]

        # Alignment: how many timeframes agree
        bull_count = sum(1 for d in directions if d > 0)
        bear_count = sum(1 for d in directions if d < 0)
        neutral_count = sum(1 for d in directions if d == 0)

        total = len(directions)
        alignment = max(bull_count, bear_count) / total if total > 0 else 0

        # Consensus direction
        if bull_count > bear_count and bull_count > neutral_count:
            consensus_direction = 1
        elif bear_count > bull_count and bear_count > neutral_count:
            consensus_direction = -1
        else:
            consensus_direction = 0

        # Weighted confidence
        avg_strength = np.mean(strengths) if strengths else 0
        confidence = alignment * avg_strength

        return {
            "alignment": alignment,
            "direction": consensus_direction,
            "confidence": confidence,
            "bull_count": bull_count,
            "bear_count": bear_count,
            "signals": signals,
        }

    def _analyze_single_timeframe(
        self, df: pd.DataFrame, tf: Timeframe
    ) -> SignalStrength:
        """Analyze a single timeframe for trend direction and strength."""
        # Calculate EMAs
        ema_fast = df["close"].ewm(span=12).mean().iloc[-1]
        ema_slow = df["close"].ewm(span=26).mean().iloc[-1]

        # Calculate RSI
        delta = df["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = (100 - (100 / (1 + rs))).iloc[-1]

        # MACD
        ema_12 = df["close"].ewm(span=12).mean()
        ema_26 = df["close"].ewm(span=26).mean()
        macd_line = ema_12 - ema_26
        signal_line = macd_line.ewm(span=9).mean()
        macd_hist = macd_line - signal_line

        # Determine direction
        direction = 0
        if ema_fast > ema_slow and macd_hist.iloc[-1] > 0:
            direction = 1
        elif ema_fast < ema_slow and macd_hist.iloc[-1] < 0:
            direction = -1

        # Calculate strength (0 to 1)
        ema_distance = abs(ema_fast - ema_slow) / df["close"].iloc[-1]
        macd_strength = abs(macd_hist.iloc[-1]) / df["close"].std()
        rsi_strength = abs(rsi - 50) / 50

        strength = min(1.0, (ema_distance * 10 + macd_strength + rsi_strength) / 3)

        # Confidence based on data quality
        confidence = 1.0 - (df["close"].isna().sum() / len(df))

        return SignalStrength(
            timeframe=tf, direction=direction, strength=strength, confidence=confidence
        )

    def get_entry_signal(self, alignment: Dict) -> Tuple[int, float]:
        """
        Get entry signal based on multi-timeframe alignment.

        Returns:
            (direction, confidence): direction (-1, 0, 1), confidence (0-1)
        """
        if alignment["confidence"] > 0.7 and alignment["alignment"] > 0.66:
            # Strong consensus across timeframes
            return alignment["direction"], alignment["confidence"]
        elif alignment["confidence"] > 0.5 and alignment["alignment"] > 0.5:
            # Moderate consensus
            return alignment["direction"], alignment["confidence"] * 0.7
        else:
            # No clear signal
            return 0, 0.0

    def calculate_confluence_zones(
        self, data: Dict[Timeframe, pd.DataFrame]
    ) -> List[Dict]:
        """
        Calculate support/resistance confluence zones across timeframes.

        Professional traders look for levels that hold across multiple
        timeframes - these are stronger support/resistance areas.
        """
        all_levels = []

        for tf, df in data.items():
            if len(df) < 50:
                continue

            # Pivot highs and lows
            highs = df["high"].values
            lows = df["low"].values

            # Find local maxima/minima
            for i in range(2, len(highs) - 2):
                if (
                    highs[i] > highs[i - 1]
                    and highs[i] > highs[i - 2]
                    and highs[i] > highs[i + 1]
                    and highs[i] > highs[i + 2]
                ):
                    all_levels.append(
                        {
                            "price": highs[i],
                            "type": "resistance",
                            "timeframe": tf,
                            "strength": 1,
                        }
                    )

                if (
                    lows[i] < lows[i - 1]
                    and lows[i] < lows[i - 2]
                    and lows[i] < lows[i + 1]
                    and lows[i] < lows[i + 2]
                ):
                    all_levels.append(
                        {
                            "price": lows[i],
                            "type": "support",
                            "timeframe": tf,
                            "strength": 1,
                        }
                    )

        # Cluster levels that are close together (confluence)
        confluence_zones = []
        tolerance = 0.005  # 0.5%

        for level in all_levels:
            price = level["price"]

            # Find existing zone
            found = False
            for zone in confluence_zones:
                if abs(zone["price"] - price) / price < tolerance:
                    zone["strength"] += level["strength"]
                    zone["hits"] += 1
                    zone["timeframes"].add(level["timeframe"])
                    found = True
                    break

            if not found:
                confluence_zones.append(
                    {
                        "price": price,
                        "type": level["type"],
                        "strength": level["strength"],
                        "hits": 1,
                        "timeframes": {level["timeframe"]},
                    }
                )

        # Sort by strength
        confluence_zones.sort(key=lambda x: x["strength"], reverse=True)

        return confluence_zones[:10]  # Top 10 strongest zones


class MarketStructureAnalyzer:
    """
    Analyzes market structure (trend, ranges, breakouts).

    SOTA Feature: Detects market structure shifts that precede
    major moves, allowing early positioning.
    """

    def __init__(self):
        self.structure_history = []
        logger.info("MarketStructureAnalyzer initialized")

    def detect_structure(self, df: pd.DataFrame) -> Dict:
        """
        Detect current market structure.

        Returns structure type and characteristics.
        """
        # Calculate swing highs and lows
        highs = df["high"].values
        lows = df["low"].values
        closes = df["close"].values

        # Find swing points
        swing_highs = []
        swing_lows = []

        for i in range(5, len(highs) - 5):
            # Swing high
            if highs[i] == max(highs[i - 5 : i + 6]):
                swing_highs.append((i, highs[i]))

            # Swing low
            if lows[i] == min(lows[i - 5 : i + 6]):
                swing_lows.append((i, lows[i]))

        # Analyze structure
        structure = {
            "type": "unknown",
            "trend_strength": 0.0,
            "swing_highs": swing_highs,
            "swing_lows": swing_lows,
            "higher_highs": False,
            "higher_lows": False,
            "lower_highs": False,
            "lower_lows": False,
        }

        if len(swing_highs) >= 2 and len(swing_lows) >= 2:
            # Check for trend
            hh = swing_highs[-1][1] > swing_highs[-2][1]
            hl = swing_lows[-1][1] > swing_lows[-2][1]
            lh = swing_highs[-1][1] < swing_highs[-2][1]
            ll = swing_lows[-1][1] < swing_lows[-2][1]

            structure["higher_highs"] = hh
            structure["higher_lows"] = hl
            structure["lower_highs"] = lh
            structure["lower_lows"] = ll

            if hh and hl:
                structure["type"] = "uptrend"
                structure["trend_strength"] = 1.0
            elif lh and ll:
                structure["type"] = "downtrend"
                structure["trend_strength"] = 1.0
            elif hh and ll:
                structure["type"] = "expansion"
                structure["trend_strength"] = 0.8
            elif lh and hl:
                structure["type"] = "contraction"
                structure["trend_strength"] = 0.5
            else:
                structure["type"] = "ranging"
                structure["trend_strength"] = 0.3

        self.structure_history.append(structure)

        return structure

    def detect_breakout(self, df: pd.DataFrame, lookback: int = 20) -> Optional[Dict]:
        """
        Detect breakout from consolidation.

        Returns breakout information or None if no breakout.
        """
        if len(df) < lookback + 5:
            return None

        recent = df.iloc[-lookback:]
        current = df.iloc[-1]

        # Calculate range
        range_high = recent["high"].max()
        range_low = recent["low"].min()
        range_size = range_high - range_low

        # Check for breakout
        if current["close"] > range_high:
            return {
                "type": "breakout_up",
                "price": current["close"],
                "resistance": range_high,
                "support": range_low,
                "strength": (current["close"] - range_high) / range_size,
            }
        elif current["close"] < range_low:
            return {
                "type": "breakout_down",
                "price": current["close"],
                "support": range_low,
                "resistance": range_high,
                "strength": (range_low - current["close"]) / range_size,
            }

        return None

    def calculate_fibonacci_levels(self, df: pd.DataFrame) -> Dict:
        """
        Calculate Fibonacci retracement levels from recent swing.

        Professional traders use fib levels for entries and targets.
        """
        if len(df) < 20:
            return {}

        # Find recent swing high and low
        recent_high = df["high"].iloc[-20:].max()
        recent_low = df["low"].iloc[-20:].min()

        diff = recent_high - recent_low

        levels = {
            "high": recent_high,
            "low": recent_low,
            "0.0": recent_high,
            "0.236": recent_high - 0.236 * diff,
            "0.382": recent_high - 0.382 * diff,
            "0.5": recent_high - 0.5 * diff,
            "0.618": recent_high - 0.618 * diff,
            "0.786": recent_high - 0.786 * diff,
            "1.0": recent_low,
        }

        return levels


def create_multi_timeframe_features(
    df: pd.DataFrame, timeframes: List[str]
) -> pd.DataFrame:
    """
    Create features by aggregating data from multiple timeframes.

    This is a SOTA technique used by quantitative funds to capture
    patterns across different time horizons.
    """
    features = pd.DataFrame(index=df.index)

    # Resample to different timeframes and create features
    for tf in timeframes:
        resampled = (
            df.resample(tf)
            .agg(
                {
                    "open": "first",
                    "high": "max",
                    "low": "min",
                    "close": "last",
                    "volume": "sum",
                }
            )
            .dropna()
        )

        # Calculate features on resampled data
        features[f"return_{tf}"] = df["close"].pct_change()
        features[f"volatility_{tf}"] = df["close"].rolling(window=20).std()
        features[f"trend_{tf}"] = np.where(
            df["close"] > df["close"].shift(5),
            1,
            np.where(df["close"] < df["close"].shift(5), -1, 0),
        )

    return features
