"""
Multi-Timeframe Analysis Module
================================

SOTA (State-of-the-Art) feature engineering module that aggregates trading
signals across multiple timeframes for robust trading decisions.

This module implements the "Top-Down Analysis" approach used by professional
traders and quantitative funds worldwide. The key insight is that trends and
signals should be confirmed across multiple timeframes to increase probability
of success.

Why Multi-Timeframe Analysis?
------------------------------
Professional traders use multi-timeframe analysis because:
1. IDENTIFY MAJOR TRENDS: Higher timeframes (daily, weekly) show the overall trend
2. TIME ENTRIES: Lower timeframes (1h, 15m) provide better entry/exit timing
3. CONFIRM SIGNALS: When multiple timeframes agree, signal reliability increases
4. AVOID FALSE SIGNALS: A signal that contradicts the higher timeframe trend
   is more likely to fail

Typical Analysis Framework:
---------------------------
- Weekly (W1): Market structure, major trend direction
- Daily (D1): Trade direction bias
- 4-Hour (H4): Key support/resistance levels, trade setup
- 1-Hour (H1): Entry/exit timing, stop placement
- 15-Minute (M15): Precise entry triggers

Key Components:
--------------
1. MultiTimeframeAnalyzer: Aggregates signals from multiple timeframes
2. MarketStructureAnalyzer: Detects trends, ranges, and breakouts
3. Support/Resistance Confluence: Finds strong price levels

Academic References:
--------------------
- "Trading in the Zone" by Mark Douglas (multi-timeframe psychology)
- "Technical Analysis of the Financial Markets" by John Murphy
- "The New Trading for a Living" by Dr. Alexander Elder

Author: BITCOIN4Traders Team
Version: 1.0.0
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from loguru import logger


class Timeframe(Enum):
    """
    Standard trading timeframes with their string representations.

    These timeframes correspond to common chart intervals used in
    technical analysis. The choice of timeframe depends on trading
    style:
    - Scalping: M1, M5
    - Day trading: M15, M30, H1
    - Swing trading: H4, D1
    - Position trading: W1, MN

    Attributes:
        value: String representation used by trading platforms (e.g., '1h')

    Example:
        >>> tf = Timeframe.H1
        >>> print(tf.value)
        '1h'
    """

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
    """
    Signal strength metrics for a single timeframe analysis.

    This dataclass encapsulates the directional signal, strength (0-1),
    and confidence (0-1) for a particular timeframe.

    Attributes:
        timeframe: The Timeframe enum value this signal represents
        direction: Trading direction (-1 = bearish, 0 = neutral, 1 = bullish)
        strength: Signal strength from 0.0 (weak) to 1.0 (strong)
        confidence: Confidence in the signal from 0.0 (low) to 1.0 (high)
                   Based on data quality (absence of missing values)

    Example:
        >>> signal = SignalStrength(
        ...     timeframe=Timeframe.H1,
        ...     direction=1,
        ...     strength=0.75,
        ...     confidence=0.9
        ... )
        >>> print(f"Bullish signal on {signal.timeframe.value}")
    """

    timeframe: Timeframe
    direction: int  # -1, 0, 1
    strength: float  # 0.0 to 1.0
    confidence: float  # 0.0 to 1.0


class MultiTimeframeAnalyzer:
    """
    Multi-timeframe signal aggregator for robust trading decisions.

    This class implements the "Top-Down Analysis" approach used by professional
    traders and quantitative funds. It analyzes price action across multiple
    timeframes and combines signals to produce more reliable trading signals.

    The Analysis Process:
    ---------------------
    1. For each timeframe, compute technical indicators (EMA, RSI, MACD)
    2. Determine direction based on EMA crossover and MACD histogram
    3. Calculate strength based on EMA distance, MACD magnitude, RSI extremity
    4. Calculate alignment: how many timeframes agree on direction
    5. Compute confidence: product of alignment and average strength

    Signal Generation Logic:
    ------------------------
    - BULLISH (direction=1): Fast EMA > Slow EMA AND MACD histogram > 0
    - BEARISH (direction=-1): Fast EMA < Slow EMA AND MACD histogram < 0
    - NEUTRAL (direction=0): Otherwise

    Entry Signal Rules:
    --------------------
    - STRONG CONSENSUS: >2/3 timeframes agree AND confidence > 0.7 → TAKE TRADE
    - MODERATE CONSENSUS: >1/2 timeframes agree → TAKE WITH 30% DISCOUNT
    - NO CONSENSUS: Stand aside, wait for better setup

    Attributes:
        timeframes: List of Timeframe enums to analyze
        signals: Dictionary mapping Timeframe to SignalStrength

    Usage:
        >>> analyzer = MultiTimeframeAnalyzer([Timeframe.H4, Timeframe.H1, Timeframe.M15])
        >>> alignment = analyzer.calculate_trend_alignment(data_dict)
        >>> direction, confidence = analyzer.get_entry_signal(alignment)
        >>> if direction != 0:
        ...     print(f"Signal: {'BUY' if direction == 1 else 'SELL'} (confidence: {confidence:.2f})")

    References:
        - Elder, A. (1993): "Trading for a Living" - Triple Screen system
        - Murphy, J. (1999): "Technical Analysis of the Financial Markets"
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
        Calculate trend alignment across multiple timeframes.

        This is the core method that analyzes each timeframe and determines
        how well they agree on market direction.

        The method:
        1. Analyzes each timeframe individually using _analyze_single_timeframe()
        2. Computes alignment: fraction of timeframes agreeing with dominant direction
        3. Computes consensus direction: bullish, bearish, or neutral
        4. Computes confidence: alignment × average strength

        Alignment Score:
        ----------------
        - 1.0 (100%): All timeframes agree
        - 0.66 (2/3): Two of three timeframes agree
        - 0.5 (1/2): Split decision
        - 0.0: Complete disagreement

        Parameters:
        -----------
        data : Dict[Timeframe, pd.DataFrame]
            Dictionary mapping Timeframe to OHLCV DataFrame.
            Each DataFrame must have 'close' and preferably 'high', 'low' columns.

        Returns:
        --------
        alignment_metrics : Dict
            Dictionary containing:
            - alignment: Float 0-1, fraction of TFs agreeing
            - direction: Int (-1, 0, 1), consensus direction
            - confidence: Float 0-1, alignment × avg strength
            - bull_count: Number of bullish timeframes
            - bear_count: Number of bearish timeframes
            - signals: List of SignalStrength objects for each TF

        Example:
            >>> data = {
            ...     Timeframe.H4: df_h4,
            ...     Timeframe.H1: df_h1,
            ...     Timeframe.M15: df_m15,
            ... }
            >>> result = analyzer.calculate_trend_alignment(data)
            >>> print(f"Alignment: {result['alignment']:.0%}")
            >>> print(f"Direction: {result['direction']} (1=bull, -1=bear)")
            >>> print(f"Confidence: {result['confidence']:.2f}")
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

        # Alignment: fraction of timeframes agreeing with the dominant direction
        bull_count = sum(1 for d in directions if d > 0)  # number of bullish TF signals
        bear_count = sum(1 for d in directions if d < 0)  # number of bearish TF signals
        neutral_count = sum(
            1 for d in directions if d == 0
        )  # number of neutral TF signals

        total = len(directions)
        alignment = (
            max(bull_count, bear_count) / total if total > 0 else 0
        )  # 1.0 = unanimous, 0.5 = split

        # Consensus direction
        if bull_count > bear_count and bull_count > neutral_count:
            consensus_direction = 1
        elif bear_count > bull_count and bear_count > neutral_count:
            consensus_direction = -1
        else:
            consensus_direction = 0

        # Weighted confidence: product of directional agreement × average signal strength
        avg_strength = (
            np.mean(strengths) if strengths else 0
        )  # mean per-TF strength (0–1)
        confidence = (
            alignment * avg_strength
        )  # high only when TFs agree AND individually strong

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
        # Calculate EMAs: fast (12-period) and slow (26-period) exponential moving averages
        ema_fast = (
            df["close"].ewm(span=12).mean().iloc[-1]
        )  # span=12 → α = 2/(12+1) ≈ 0.154
        ema_slow = (
            df["close"].ewm(span=26).mean().iloc[-1]
        )  # span=26 → α = 2/(26+1) ≈ 0.074

        # Calculate RSI (Wilder's method, 14-period)
        delta = df["close"].diff()  # bar-to-bar price changes
        gain = (
            (delta.where(delta > 0, 0)).rolling(window=14).mean()
        )  # average gain over 14 bars
        loss = (
            (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        )  # average loss (positive value)
        rs = gain / loss  # relative strength: ratio of avg gain to avg loss
        rsi = (100 - (100 / (1 + rs))).iloc[
            -1
        ]  # RSI = 100 - 100/(1+RS); oscillates 0–100

        # MACD: momentum indicator using difference between fast and slow EMA
        ema_12 = df["close"].ewm(span=12).mean()
        ema_26 = df["close"].ewm(span=26).mean()
        macd_line = ema_12 - ema_26  # MACD line: measures trend momentum
        signal_line = macd_line.ewm(
            span=9
        ).mean()  # 9-period EMA of MACD; acts as trigger line
        macd_hist = (
            macd_line - signal_line
        )  # histogram: positive → bullish momentum, negative → bearish

        # Determine direction: require both EMA crossover AND MACD histogram confirmation
        direction = 0
        if ema_fast > ema_slow and macd_hist.iloc[-1] > 0:
            direction = 1  # bullish: price trending up and momentum accelerating
        elif ema_fast < ema_slow and macd_hist.iloc[-1] < 0:
            direction = -1  # bearish: price trending down and momentum decelerating

        # Calculate signal strength (0 to 1): composite of EMA spread, MACD magnitude, RSI extremity
        ema_distance = (
            abs(ema_fast - ema_slow) / df["close"].iloc[-1]
        )  # normalized EMA gap (% of price)
        macd_strength = (
            abs(macd_hist.iloc[-1]) / df["close"].std()
        )  # MACD histogram normalized by price volatility
        rsi_strength = (
            abs(rsi - 50) / 50
        )  # distance from RSI midpoint (50); 0 = neutral, 1 = extreme overbought/sold

        strength = min(
            1.0, (ema_distance * 10 + macd_strength + rsi_strength) / 3
        )  # equal-weight average, capped at 1.0

        # Confidence based on data quality: fraction of non-NaN bars
        confidence = 1.0 - (
            df["close"].isna().sum() / len(df)
        )  # 1.0 = no missing data, 0.0 = all missing

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
            # Strong consensus: >2/3 of TFs agree AND composite confidence >0.7
            return alignment["direction"], alignment["confidence"]
        elif alignment["confidence"] > 0.5 and alignment["alignment"] > 0.5:
            # Moderate consensus: majority agree but weaker — discount confidence by 30%
            return alignment["direction"], alignment["confidence"] * 0.7
        else:
            # No clear signal: mixed or weak — stand aside
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
        tolerance = (
            0.005  # 0.5% price band: levels within this range are merged into one zone
        )

        for level in all_levels:
            price = level["price"]

            # Find existing zone within tolerance band
            found = False
            for zone in confluence_zones:
                if (
                    abs(zone["price"] - price) / price < tolerance
                ):  # relative distance < 0.5%
                    zone["strength"] += level["strength"]  # accumulate strength score
                    zone["hits"] += 1  # count how many raw levels merged here
                    zone["timeframes"].add(
                        level["timeframe"]
                    )  # track which TFs contributed
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
    Analyzes market structure including trend, ranges, and breakouts.

    This class implements state-of-the-art market structure detection used by
    professional traders to identify:
    - Trend direction and strength (uptrend, downtrend)
    - Consolidation phases (ranging, contraction)
    - Trend exhaustion (expansion patterns)
    - Breakout opportunities

    Market Structure Types:
    -----------------------
    1. UPTREND: Higher highs (HH) AND higher lows (HL)
       - Price making new highs while respecting previous lows
       - Bullish bias

    2. DOWNTREND: Lower highs (LH) AND lower lows (LL)
       - Price making new lows while failing at previous highs
       - Bearish bias

    3. CONSOLIDATION/RANGING: Price trapped between support and resistance
       - No clear directional bias
       - Wait for breakout

    4. EXPANSION: HH AND LL (volatile trend)
       - Strong momentum but may be exhausting

    5. CONTRACTION: LH AND HL (tightening range)
       - Often precedes breakout (squeeze pattern)

    Swing Detection:
    ---------------
    Uses 5-bar swing identification to find local maxima and minima:
    - Swing high: Highest high in 5-bar window
    - Swing low: Lowest low in 5-bar window

    Attributes:
        structure_history: List of detected structures over time

    Usage:
        >>> analyzer = MarketStructureAnalyzer()
        >>> structure = analyzer.detect_structure(df)
        >>> print(f"Structure: {structure['type']}")
        >>> print(f"Trend strength: {structure['trend_strength']}")
        >>>
        >>> # Check for breakout
        >>> breakout = analyzer.detect_breakout(df)
        >>> if breakout:
        ...     print(f"Breakout: {breakout['type']}")

    References:
        - "Trading in the Zone" by Mark Douglas
        - "The Inner Circle" by Michael Mauboussin
        - ICT (Inner Circle Trader) methodology
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

        diff = recent_high - recent_low  # full swing range

        # Fibonacci retracement levels: derived from the Golden Ratio (φ ≈ 1.618)
        levels = {
            "high": recent_high,
            "low": recent_low,
            "0.0": recent_high,  # 0% retracement = top of swing
            "0.236": recent_high - 0.236 * diff,  # 23.6% = φ^-3 ≈ shallow pullback
            "0.382": recent_high - 0.382 * diff,  # 38.2% = 1 - φ^-1 ≈ common entry zone
            "0.5": recent_high
            - 0.5 * diff,  # 50% = midpoint (not a true Fib, but widely used)
            "0.618": recent_high
            - 0.618 * diff,  # 61.8% = φ^-1 ≈ "golden ratio" — strongest Fib level
            "0.786": recent_high - 0.786 * diff,  # 78.6% = √0.618 ≈ deep retracement
            "1.0": recent_low,  # 100% retracement = bottom of swing
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

        # Calculate features on resampled data (applied to raw df, aligned to original index)
        features[f"return_{tf}"] = df[
            "close"
        ].pct_change()  # bar-to-bar return at base TF
        features[f"volatility_{tf}"] = (
            df["close"].rolling(window=20).std()
        )  # 20-bar realized vol proxy
        features[f"trend_{tf}"] = np.where(
            df["close"]
            > df["close"].shift(5),  # price higher than 5 bars ago → uptrend
            1,
            np.where(
                df["close"] < df["close"].shift(5), -1, 0
            ),  # lower → downtrend, equal → neutral
        )

    return features
