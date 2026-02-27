"""
Market Microstructure Features
===============================

SOTA (State-of-the-Art) module for order flow and market microstructure analysis.

Professional quant firms and high-frequency traders use microstructure features to:
1. DETECT INFORMED TRADING: Identify when large traders are actively buying/selling
2. PREDICT SHORT-TERM PRICE MOVEMENTS: Order flow often precedes price
3. OPTIMIZE EXECUTION TIMING: Time entries when liquidity is favorable
4. DETECT MARKET MANIPULATION: Identify iceberg orders, spoofing
5. MEASURE LIQUIDITY: Understand true market depth and impact

What is Market Microstructure?
------------------------------
Microstructure is the study of the process and outcomes of exchanging assets.
Unlike traditional technical analysis (which looks at prices over time),
microstructure examines:
- Order book dynamics (bid/ask spread, depth)
- Trade sequencing (who is trading, in what order)
- Transaction costs and execution quality
- Information arrival and price discovery

Key Concepts:
-------------
1. VPIN (Volume-Synchronized Probability of Informed Trading):
   Measures order flow toxicity - when informed traders are active, volatility
   tends to increase. Derived from the classic PIN model.

2. ORDER IMBALANCE: Difference between buy and sell volume at bid/ask levels.
   Positive imbalance = more buying pressure = potential price increase.

3. TOXIC FLOW: Order flow that moves prices against the market maker.
   Detected by trade sequencing patterns.

4. KYLE'S LAMBDA: Price impact coefficient. Measures how much prices move
   in response to order flow. High lambda = low liquidity.

5. AMIHUD ILLIQUIDITY: Price impact per dollar of trading. Used to measure
   market liquidity costs.

Academic References:
-------------------
- Easley, Lopez de Prado, O'Hara (2012): "VPIN and the Flash Crash"
- Kyle (1985): "Continuous Auctions and Insider Trading"
- Amihud (2002): "Illiquidity and Stock Returns"

Author: BITCOIN4Traders Team
Version: 1.0.0
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from numba import jit
from loguru import logger


@dataclass
class OrderFlowMetrics:
    """
    Comprehensive order flow metrics from tick-level trade data.

    This dataclass encapsulates all key order flow measurements used to
    understand the balance between buying and selling pressure in the market.

    Attributes:
        buy_volume: Total volume of buy-initiated trades (in base currency)
        sell_volume: Total volume of sell-initiated trades (in base currency)
        buy_pressure: Volume-weighted dollar value at bid levels
        sell_pressure: Volume-weighted dollar value at ask levels
        order_imbalance: Normalized difference between bid/ask volumes (-1 to +1)
        trade_intensity: Ratio of trades to book updates (high = active market)
        toxic_flow_score: Probability that informed traders are active (0-1)

    Interpretation:
        - Positive order_imbalance: More buying pressure
        - Negative order_imbalance: More selling pressure
        - High toxic_flow_score: Informed traders likely present, expect volatility
        - High trade_intensity: Active market, good for execution

    Example:
        >>> metrics = analyzer.calculate_order_flow_metrics(
        ...     bid_prices, ask_prices, bid_volumes, ask_volumes,
        ...     trade_prices, trade_volumes
        ... )
        >>> print(f"Order imbalance: {metrics.order_imbalance:.3f}")
        >>> print(f"Toxic flow: {metrics.toxic_flow_score:.3f}")
    """

    buy_volume: float
    sell_volume: float
    buy_pressure: float
    sell_pressure: float
    order_imbalance: float
    trade_intensity: float
    toxic_flow_score: float


class MicrostructureAnalyzer:
    """
    Analyzes market microstructure from tick-level data.

    This class implements VPIN (Volume-Synchronized Probability of Informed
    Trading) and other microstructure metrics used by high-frequency traders
    and quantitative funds.

    VPIN (Volume-Synchronized Probability of Informed Trading):
    -----------------------------------------------------------
    VPIN measures the probability that informed traders are active in the
    market. When informed traders (those with private information) are active:
    - Order flow becomes more one-sided
    - Volatility tends to increase
    - Market makers widen spreads

    VPIN is calculated by:
    1. Classifying trades as buys or sells using the tick rule
    2. Creating volume buckets of equal size
    3. Computing order imbalance in each bucket
    4. Averaging the imbalances

    High VPIN (>0.5) historically preceded market disruptions including
    the 2010 Flash Crash.

    Attributes:
        bucket_size: Number of trades per volume bucket for VPIN (default: 50)
        trade_history: List of recent trades
        vpin_history: Historical VPIN values

    Usage:
        >>> analyzer = MicrostructureAnalyzer(bucket_size=50)
        >>> vpin = analyzer.calculate_vpin(trades_df)
        >>> print(f"VPIN: {vpin:.3f}")
        >>>
        >>> # Calculate comprehensive metrics
        >>> metrics = analyzer.calculate_order_flow_metrics(
        ...     bid_prices, ask_prices, bid_volumes, ask_volumes,
        ...     trade_prices, trade_volumes
        ... )
        >>> print(f"Order imbalance: {metrics.order_imbalance:.3f}")

    References:
        - Easley, D., Lopez de Prado, M., O'Hara, M. (2012)
        - "The VPIN Theory of Flash Crashes"
    """

    def __init__(self, bucket_size: int = 50):
        """
        Initialize microstructure analyzer.

        Args:
            bucket_size: Number of trades per volume bucket for VPIN
        """
        self.bucket_size = bucket_size
        self.trade_history = []
        self.vpin_history = []
        logger.info(f"MicrostructureAnalyzer initialized: bucket_size={bucket_size}")

    def calculate_vpin(self, trades: pd.DataFrame) -> float:
        """
        Calculate Volume-Synchronized Probability of Informed Trading.

        VPIN detects when informed traders are active by measuring order flow
        toxicity. High VPIN predicts increased volatility and potential
        market dislocations.

        The Algorithm:
        --------------
        1. Classify each trade as buy or sell using the tick rule:
           - Price up from previous tick → Buy
           - Price down from previous tick → Sell
           - Price unchanged → Same as previous

        2. Create volume buckets of equal size (bucket_size trades each)

        3. Calculate order imbalance for each bucket:
           Imbalance = |Buy Volume - Sell Volume| / Total Volume

        4. VPIN = Average of all bucket imbalances

        Interpretation:
        ---------------
        - VPIN < 0.3: Low informed trading, stable conditions
        - VPIN 0.3-0.5: Moderate activity
        - VPIN > 0.5: High informed trading, expect volatility

        Parameters:
        -----------
        trades : pd.DataFrame
            Trade data with columns: price, volume (optional, defaults to 1)

        Returns:
        --------
        vpin : float
            VPIN value between 0 and 1
            Returns 0.0 if insufficient data (< bucket_size trades)

        Example:
            >>> trades = pd.DataFrame({
            ...     'price': [50000, 50010, 50020, 50015, 50025],
            ...     'volume': [1.5, 2.0, 1.0, 0.5, 3.0]
            ... })
            >>> vpin = analyzer.calculate_vpin(trades)
            >>> print(f"VPIN: {vpin:.3f}")

        Reference:
            Easley, Lopez de Prado, O'Hara (2012)
            "Volume Synchronized Probability of Informed Trading"
        """
        if len(trades) < self.bucket_size:
            return 0.0

        # Classify trades as buy or sell (using tick rule)
        trades = trades.copy()
        trades["sign"] = self._classify_trades(trades["price"].values)

        # Create volume buckets
        trades["volume"] = trades.get("volume", 1.0)
        trades["cumulative_volume"] = trades["volume"].cumsum()

        # Calculate buy/sell volume per bucket
        n_buckets = int(trades["cumulative_volume"].iloc[-1] / self.bucket_size)
        if n_buckets < 2:
            return 0.0

        bucket_buys = []
        bucket_sells = []

        for i in range(n_buckets):
            start_vol = i * self.bucket_size
            end_vol = (i + 1) * self.bucket_size

            bucket_trades = trades[
                (trades["cumulative_volume"] >= start_vol)
                & (trades["cumulative_volume"] < end_vol)
            ]

            buys = bucket_trades[bucket_trades["sign"] > 0]["volume"].sum()
            sells = bucket_trades[bucket_trades["sign"] < 0]["volume"].sum()

            bucket_buys.append(buys)
            bucket_sells.append(sells)

        # Calculate order imbalance for each bucket
        imbalances = []
        for buys, sells in zip(bucket_buys, bucket_sells):
            total = buys + sells
            if total > 0:
                imbalance = abs(buys - sells) / total
                imbalances.append(imbalance)

        # VPIN is average imbalance
        vpin = np.mean(imbalances) if imbalances else 0.0
        self.vpin_history.append(vpin)

        return vpin

    def _classify_trades(self, prices: np.ndarray) -> np.ndarray:
        """Classify trades as buys (+1) or sells (-1) using tick rule."""
        signs = np.zeros(len(prices))

        for i in range(1, len(prices)):
            if prices[i] > prices[i - 1]:
                signs[i] = 1  # Buy
            elif prices[i] < prices[i - 1]:
                signs[i] = -1  # Sell
            else:
                signs[i] = signs[i - 1]  # Same as previous

        return signs

    def calculate_order_flow_metrics(
        self,
        bid_prices: np.ndarray,
        ask_prices: np.ndarray,
        bid_volumes: np.ndarray,
        ask_volumes: np.ndarray,
        trade_prices: np.ndarray,
        trade_volumes: np.ndarray,
    ) -> OrderFlowMetrics:
        """
        Calculate comprehensive order flow metrics.
        """
        # Buy/Sell pressure: volume-weighted dollar value at bid/ask levels
        buy_pressure = np.sum(
            bid_volumes * bid_prices
        )  # Total bid-side liquidity in USD
        sell_pressure = np.sum(
            ask_volumes * ask_prices
        )  # Total ask-side liquidity in USD

        # Order imbalance: (bid_vol - ask_vol) / total_vol  (-1=fully ask-side, +1=fully bid-side)
        total_bid_vol = np.sum(bid_volumes)
        total_ask_vol = np.sum(ask_volumes)
        order_imbalance = (total_bid_vol - total_ask_vol) / (
            total_bid_vol + total_ask_vol + 1e-10  # Epsilon prevents division by zero
        )

        # Classify trades
        trade_signs = self._classify_trades(trade_prices)

        buy_volume = np.sum(trade_volumes[trade_signs > 0])
        sell_volume = np.sum(trade_volumes[trade_signs < 0])

        # Trade intensity
        trade_intensity = len(trade_prices) / max(len(bid_prices), 1)

        # Toxic flow score (informed trading detection)
        toxic_flow = self._calculate_toxic_flow(
            trade_prices, trade_volumes, trade_signs
        )

        return OrderFlowMetrics(
            buy_volume=buy_volume,
            sell_volume=sell_volume,
            buy_pressure=buy_pressure,
            sell_pressure=sell_pressure,
            order_imbalance=order_imbalance,
            trade_intensity=trade_intensity,
            toxic_flow_score=toxic_flow,
        )

    def _calculate_toxic_flow(
        self, prices: np.ndarray, volumes: np.ndarray, signs: np.ndarray
    ) -> float:
        """
        Calculate toxic flow score using trade sequencing.

        Detects when large orders are executed in sequence,
        indicating informed trading.
        """
        if len(prices) < 10:
            return 0.0

        # Calculate volume-weighted price autocorrelation
        vw_returns = np.diff(prices) * volumes[1:]

        if len(vw_returns) < 5:
            return 0.0

        # Calculate rolling correlation of signed volume
        window = min(20, len(vw_returns) // 2)
        correlations = []

        for i in range(window, len(vw_returns)):
            corr = np.corrcoef(signs[i - window : i], volumes[i - window : i])[0, 1]
            if not np.isnan(corr):
                correlations.append(abs(corr))

        toxic_score = np.mean(correlations) if correlations else 0.0
        return min(1.0, toxic_score)

    def detect_iceberg_orders(
        self, trades: pd.DataFrame, volume_threshold: float = 0.1
    ) -> List[Dict]:
        """
        Detect potential iceberg orders.

        Iceberg orders are large orders split into smaller pieces
        to hide their true size. Detecting them gives an edge.
        """
        iceberg_candidates = []

        if len(trades) < 10:
            return iceberg_candidates

        # Group trades by timestamp (if available)
        if "timestamp" in trades.columns:
            trades["time_bucket"] = trades["timestamp"].dt.floor("1min")
            grouped = trades.groupby("time_bucket")

            for time_bucket, group in grouped:
                if len(group) < 3:
                    continue

                volumes = group["volume"].values
                prices = group["price"].values

                # Check for similar-sized trades
                mean_vol = np.mean(volumes)
                std_vol = np.std(volumes)

                # Iceberg pattern: similar sizes, similar prices
                if std_vol / mean_vol < volume_threshold:
                    total_vol = np.sum(volumes)
                    price_range = np.max(prices) - np.min(prices)

                    if total_vol > mean_vol * 5:  # Significant total volume
                        iceberg_candidates.append(
                            {
                                "time": time_bucket,
                                "n_trades": len(group),
                                "avg_volume": mean_vol,
                                "total_volume": total_vol,
                                "price_range": price_range,
                                "confidence": 1.0 - (std_vol / mean_vol),
                            }
                        )

        return iceberg_candidates


@jit(nopython=True, cache=True)
def calculate_spread_metrics(
    bids: np.ndarray, asks: np.ndarray, bid_vols: np.ndarray, ask_vols: np.ndarray
) -> Tuple[float, float, float]:
    """
    Calculate spread metrics (Numba-accelerated).

    Returns:
        (quoted_spread, effective_spread, realized_spread)
    """
    # Quoted spread: (ask - bid) / mid  (relative cost of crossing the spread)
    best_bid = bids[0]
    best_ask = asks[0]
    mid_price = (best_bid + best_ask) / 2
    quoted_spread = (best_ask - best_bid) / mid_price

    # Volume-weighted mid: accounts for asymmetric liquidity at best bid/ask
    bid_weight = bid_vols[0] / (bid_vols[0] + ask_vols[0] + 1e-10)
    ask_weight = ask_vols[0] / (bid_vols[0] + ask_vols[0] + 1e-10)

    weighted_mid = (
        best_bid * bid_weight + best_ask * ask_weight
    )  # Volume-weighted fair value
    effective_spread = (
        abs(weighted_mid - mid_price) / mid_price
    )  # Deviation of weighted mid

    # Realized spread (simplified): portion of quoted spread retained by market maker
    realized_spread = quoted_spread * 0.5

    return quoted_spread, effective_spread, realized_spread


class LiquidityAnalyzer:
    """
    Analyzes market liquidity conditions for execution optimization.

    Liquidity analysis is critical for:
    1. POSITION SIZING: Larger positions require more liquid markets
    2. EXECUTION STRATEGY: Liquid markets → market orders; illiquid → limit orders
    3. SLIPPAGE ESTIMATION: Predict execution costs
    4. RISK MANAGEMENT: Illiquid markets have higher risk

    Key Metrics:
    ------------
    1. KYLE'S LAMBDA: Price impact coefficient
       - Measures how much prices move per unit of order flow
       - High lambda = low liquidity = high price impact

    2. AMIHUD ILLIQUIDITY RATIO: |return| / dollar volume
       - Measures price impact per dollar traded
       - Higher = more illiquid

    3. MARKET DEPTH: Volume available at price levels
       - Cumulative bid/ask volume
       - Depth imbalance

    4. LIQUIDITY SCORE: Composite 0-1 score
       - Combines volume, spread, and depth metrics

    Attributes:
        liquidity_history: Historical liquidity measurements

    Usage:
        >>> analyzer = LiquidityAnalyzer()
        >>>
        >>> # Calculate Kyle's Lambda
        >>> kyle_lambda = analyzer.calculate_kyle_lambda(returns, signed_volume)
        >>> print(f"Kyle's Lambda: {kyle_lambda:.6f}")
        >>>
        >>> # Calculate Amihud Illiquidity
        >>> amihud = analyzer.calculate_amihud_illiquidity(returns, volumes)
        >>> print(f"Amihud Ratio: {amihud:.4f}")
        >>>
        >>> # Estimate market depth
        >>> depth = analyzer.estimate_market_depth(bid_prices, ask_prices,
        ...                                          bid_volumes, ask_volumes)
        >>> print(f"Depth imbalance: {depth['depth_imbalance']:.3f}")

    References:
        - Kyle, A. (1985): "Continuous Auctions and Insider Trading"
        - Amihud, Y. (2002): "Illiquidity and Stock Returns"
    """

    def __init__(self):
        self.liquidity_history = []
        logger.info("LiquidityAnalyzer initialized")

    def calculate_kyle_lambda(
        self, price_changes: np.ndarray, signed_volumes: np.ndarray
    ) -> float:
        """
        Calculate Kyle's Lambda (price impact coefficient).

        Measures how much prices move in response to order flow.
        High lambda = low liquidity = high impact.

        Reference: Kyle (1985)
        """
        if len(price_changes) < 10:
            return 0.0

        # Regress price changes on signed volumes
        X = signed_volumes.reshape(-1, 1)
        y = price_changes

        # Simple OLS: λ = Σ(x*y) / Σ(x²)  (price impact per unit of signed order flow)
        lambda_coef = np.sum(X * y) / (np.sum(X**2) + 1e-10)

        return lambda_coef

    def calculate_amihud_illiquidity(
        self, returns: np.ndarray, volumes: np.ndarray
    ) -> float:
        """
        Calculate Amihud illiquidity ratio.

        Measures price impact per dollar of volume.
        Higher = more illiquid.
        """
        if len(returns) != len(volumes) or len(returns) == 0:
            return 0.0

        # Amihud ratio: |return| / dollar_volume  (price impact per dollar traded)
        ratios = np.abs(returns) / (volumes + 1e-10)
        amihud = (
            np.mean(ratios) * 1e6
        )  # Scale by 10^6 for readability (very small numbers)

        return amihud

    def estimate_market_depth(
        self,
        bid_prices: np.ndarray,
        ask_prices: np.ndarray,
        bid_volumes: np.ndarray,
        ask_volumes: np.ndarray,
        price_levels: int = 5,
    ) -> Dict:
        """
        Estimate market depth at multiple price levels.
        """
        depth = {
            "bid_depth": [],
            "ask_depth": [],
            "cumulative_bid_vol": [],
            "cumulative_ask_vol": [],
        }

        cum_bid_vol = 0
        cum_ask_vol = 0

        for i in range(min(price_levels, len(bid_prices))):
            cum_bid_vol += bid_volumes[i]
            cum_ask_vol += ask_volumes[i]

            depth["bid_depth"].append(bid_volumes[i])
            depth["ask_depth"].append(ask_volumes[i])
            depth["cumulative_bid_vol"].append(cum_bid_vol)
            depth["cumulative_ask_vol"].append(cum_ask_vol)

        depth["total_bid_depth"] = cum_bid_vol
        depth["total_ask_depth"] = cum_ask_vol
        depth["depth_imbalance"] = (cum_bid_vol - cum_ask_vol) / (
            cum_bid_vol + cum_ask_vol + 1e-10
        )

        return depth

    def calculate_liquidity_score(self, df: pd.DataFrame) -> float:
        """
        Calculate overall liquidity score (0-1, higher = more liquid).
        """
        if len(df) < 20:
            return 0.5

        # Components
        volume_score = min(
            1.0, df["volume"].mean() / df["volume"].rolling(20).mean().mean()
        )
        spread_score = 1.0 - min(
            1.0, df.get("spread", 0.001) * 100
        )  # Assume spread column
        depth_score = min(1.0, df.get("depth", 100) / 1000)

        # Weighted average
        liquidity_score = volume_score * 0.4 + spread_score * 0.4 + depth_score * 0.2

        self.liquidity_history.append(liquidity_score)

        return liquidity_score


def create_microstructure_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create comprehensive microstructure features.

    Professional-grade feature engineering for tick data.
    """
    features = pd.DataFrame(index=df.index)

    # Price impact
    features["price_impact"] = df["close"].diff().abs() / (df["volume"] + 1e-10)

    # Trade intensity
    features["trade_intensity"] = df["volume"] / df["volume"].rolling(20).mean()

    # Order flow imbalance (if bid/ask data available)
    if "bid_volume" in df.columns and "ask_volume" in df.columns:
        features["flow_imbalance"] = (df["bid_volume"] - df["ask_volume"]) / (
            df["bid_volume"] + df["ask_volume"] + 1e-10
        )

    # Volatility clustering
    features["realized_vol"] = df["close"].rolling(20).std()
    features["vol_of_vol"] = features["realized_vol"].rolling(20).std()

    # Tick rule (trade direction)
    features["tick_direction"] = np.where(
        df["close"] > df["close"].shift(1),
        1,
        np.where(df["close"] < df["close"].shift(1), -1, 0),
    )

    # Signed volume
    features["signed_volume"] = features["tick_direction"] * df["volume"]

    # Volume-weighted price
    features["vwap"] = (df["close"] * df["volume"]).rolling(20).sum() / df[
        "volume"
    ].rolling(20).sum()
    features["vwap_deviation"] = (df["close"] - features["vwap"]) / features["vwap"]

    # Roll's implicit spread measure: S = 2 * sqrt(-Cov(ΔP_t, ΔP_{t-1}))
    # Negative autocovariance of price changes implies bid-ask bounce
    if len(df) > 2:
        price_changes = df["close"].diff().dropna()
        cov = price_changes.cov(price_changes.shift(1))  # First-order autocovariance
        features["roll_spread"] = (
            np.sqrt(-4 * cov) if cov < 0 else 0
        )  # Only valid if cov < 0

    return features
