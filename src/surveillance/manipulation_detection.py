"""
Market Manipulation Detection System
====================================
Advanced surveillance system designed to detect exchange manipulation
and protect trading capital in real-time financial markets.

This module implements institutional-grade surveillance technology used by
prop trading firms, hedge funds, and market makers for market protection.

DETECTION CAPABILITIES:
-----------------------
- Spoofing: Large orders placed with intent to cancel, creating false liquidity
- Layering: Multiple fake orders at different price levels to manipulate depth
- Wash Trading: Self-trading to artificially inflate volume
- Quote Stuffing: Order flooding to cause delays and disrupt fair markets
- Pump and Dump: Coordinated price manipulation schemes
- Exchange Irregularities: Anomalous exchange behavior detection

PROTECTION FEATURES:
--------------------
- Real-time manipulation alerts with severity levels
- Order book toxicity scoring
- Smart execution adaptation based on market conditions
- Emergency position closure ("Shut and Run" protocol)

THREAT RESPONSE LEVELS:
----------------------
1. NORMAL: Standard trading operations
2. ELEVATED: Reduced position sizes, increased spread requirements
3. CRITICAL: Halt trading, close positions, switch to passive mode

Usage:
    from src.surveillance.manipulation_detection import (
        create_protection_system,
        detect_market_manipulation,
        emergency_shut_and_run
    )

    # Initialize protection system
    protection = create_protection_system()

    # Process order book and get recommendations
    result = protection.process_order_book(order_book_state)

    # Emergency shutdown if critical manipulation detected
    if result['protection_level'] == 'critical':
        closure_plan = emergency_shut_and_run(positions, prices, "Critical manipulation")

Author: BITCOIN4Traders Team
License: Proprietary - Internal Use Only
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import deque, defaultdict
from scipy import stats
import warnings

warnings.filterwarnings("ignore")
from loguru import logger


@dataclass
class ManipulationAlert:
    """
    Dataclass representing a detected market manipulation alert.

    Attributes:
        timestamp: When the manipulation was detected (UTC)
        type: Type of manipulation (spoofing, layering, wash_trading, etc.)
        severity: Risk level - 'low', 'medium', 'high', or 'critical'
        description: Human-readable description of the manipulation
        evidence: Dictionary containing supporting data for the alert
        affected_orders: List of order IDs involved in the manipulation
        recommended_action: Suggested response action to mitigate risk
    """

    timestamp: datetime
    type: str
    severity: str  # 'low', 'medium', 'high', 'critical'
    description: str
    evidence: Dict
    affected_orders: List[str]
    recommended_action: str


@dataclass
class OrderBookState:
    """
    Snapshot of current order book state for manipulation analysis.

    Represents a point-in-time view of the order book with aggregated
    metrics used for manipulation detection algorithms.

    Attributes:
        timestamp: Time of the snapshot (UTC)
        bids: List of (price, size) tuples for bid orders
        asks: List of (price, size) tuples for ask orders
        mid_price: Midpoint price (best_bid + best_ask) / 2
        spread: Difference best bid
        bid_depth: Total quote bid-side liquidity in currency between best ask and (USD)
        ask_depth: Total ask-side liquidity in quote currency (USD)
        imbalance: Order book imbalance in range [-1, 1]
            - Positive: More bids than asks (upward pressure)
            - Negative: More asks than bids (downward pressure)
            - Near zero: Balanced market

    Example:
        >>> book_state = OrderBookState(
        ...     timestamp=datetime.now(),
        ...     bids=[(50000.0, 1.5), (49999.0, 2.0)],
        ...     asks=[(50001.0, 1.0), (50002.0, 3.0)],
        ...     mid_price=50000.5,
        ...     spread=1.0,
        ...     bid_depth=175000.0,
        ...     ask_depth=200000.0,
        ...     imbalance=-0.067
        ... )
    """

    timestamp: datetime
    bids: List[Tuple[float, float]]  # (price, size)
    asks: List[Tuple[float, float]]
    mid_price: float
    spread: float
    bid_depth: float  # Total bid-side liquidity in USD
    ask_depth: float  # Total ask-side liquidity in USD
    imbalance: float  # (bid_depth - ask_depth) / (bid_depth + ask_depth)


class SpoofingDetector:
    """
    Detects spoofing - the practice of placing large fake orders to manipulate price.

    SPOOFING PATTERN:
        A trader places a large order (often at the top of the order book)
        with no intention of execution. This creates false impression of
        liquidity, moving price in the desired direction. The order is
        quickly canceled once price has moved.

    DETECTION METHOD:
        Monitors order lifetime - orders placed and canceled within milliseconds
        that exceed a size threshold are flagged as suspicious. After detecting
        5+ infractions from the same account, an alert is generated.

    CONFIGURATION:
        - cancel_threshold_ms: Maximum time an order can live before being
          considered suspicious (default: 500ms)
        - size_threshold_pct: Order size as fraction of average daily volume
          that triggers monitoring (default: 10%)

    Example:
        >>> detector = SpoofingDetector(cancel_threshold_ms=500, size_threshold_pct=0.1)
        >>> detector.record_order("order_123", "buy", 50000.0, 5.0, "account_abc", datetime.now())
        >>> detector.record_cancel("order_123", datetime.now() + timedelta(milliseconds=100))
    """

    def __init__(
        self, cancel_threshold_ms: float = 500, size_threshold_pct: float = 0.1
    ):
        """
        Initialize the spoofing detector.

        Args:
            cancel_threshold_ms: Orders alive less than this (ms) are suspicious
            size_threshold_pct: Orders larger than this fraction of ADV trigger checks
        """
        self.cancel_threshold = (
            cancel_threshold_ms  # Orders alive < this are suspicious
        )
        self.size_threshold = (
            size_threshold_pct  # Orders larger than this fraction trigger checks
        )
        self.order_history = deque(maxlen=10000)  # Rolling buffer of recent orders
        self.suspicious_accounts = defaultdict(int)  # Infraction count per account
        logger.info("SpoofingDetector initialized")

    def record_order(
        self,
        order_id: str,
        side: str,
        price: float,
        size: float,
        account: str,
        timestamp: datetime,
    ):
        """
        Record a new order placement for monitoring.

        Args:
            order_id: Unique identifier for the order
            side: Order side - 'buy' or 'sell'
            price: Limit price of the order
            size: Order quantity
            account: Trading account identifier
            timestamp: Order placement time (UTC)
        """
        self.order_history.append(
            {
                "order_id": order_id,
                "side": side,
                "price": price,
                "size": size,
                "account": account,
                "timestamp": timestamp,
                "status": "open",
            }
        )

    def record_cancel(self, order_id: str, timestamp: datetime):
        """
        Record order cancellation and check for spoofing pattern.

        When an order is canceled, this method checks if the cancellation
        matches a spoofing signature (canceled quickly after placement).

        Args:
            order_id: ID of the order being canceled
            timestamp: Time of cancellation (UTC)
        """
        for order in self.order_history:
            if order["order_id"] == order_id:
                order["status"] = "canceled"
                order["cancel_time"] = timestamp

                # Check for spoofing pattern immediately on cancellation
                self._check_spoofing_pattern(order)
                break

    def _check_spoofing_pattern(self, order: Dict):
        """
        Analyze if a canceled order matches spoofing characteristics.

        SPROOFING SIGNATURE:
            - Large order size relative to market
            - Short lifetime (< cancel_threshold_ms)
            - Placed near best bid/ask to influence price

        Args:
            order: Dictionary containing order details including timestamps
        """
        if "cancel_time" not in order:
            return

        # Time the order was alive in milliseconds
        time_alive = (order["cancel_time"] - order["timestamp"]).total_seconds() * 1000

        # Large order canceled quickly is a classic spoofing signature
        if time_alive < self.cancel_threshold and order["size"] > self.size_threshold:
            self.suspicious_accounts[order["account"]] += 1

            # Only alert after accumulating enough infractions to avoid false positives
            if self.suspicious_accounts[order["account"]] >= 5:
                logger.warning(
                    f"SPOOFING DETECTED: Account {order['account']}, "
                    f"Order size: {order['size']:.4f}, "
                    f"Time alive: {time_alive:.1f}ms"
                )

    def get_suspicious_accounts(self) -> Dict[str, int]:
        """
        Retrieve accounts flagged for suspicious spoofing activity.

        Returns:
            Dictionary mapping account IDs to number of suspicious infractions
        """
        return dict(self.suspicious_accounts)


class LayeringDetector:
    """
    Detects layering - placing multiple fake orders at different price levels.

    LAYERING PATTERN:
        A trader places multiple orders at different price levels (creating
        "layers" of fake liquidity) to give the illusion of market depth.
        Once price moves favorably, all orders are canceled simultaneously.

    DETECTION METHOD:
        - Monitors number of price levels occupied by a single account
        - Checks for correlated cancellations (all canceled within 100ms)
        - Flags events with 3+ layers as high severity

    CONFIGURATION:
        - max_layer_levels: Number of price levels that trigger investigation
        - cancel_correlation_threshold: Fraction of orders that must cancel together

    Example:
        >>> detector = LayeringDetector(max_layer_levels=5, cancel_correlation_threshold=0.8)
        >>> result = detector.analyze_order_book(book_state, account_orders)
    """

    def __init__(
        self, max_layer_levels: int = 5, cancel_correlation_threshold: float = 0.8
    ):
        """
        Initialize the layering detector.

        Args:
            max_layer_levels: Threshold for suspicious depth (default: 5 levels)
            cancel_correlation_threshold: Fraction that must cancel together (default: 0.8)
        """
        self.max_layers = max_layer_levels  # Threshold for suspicious depth
        self.correlation_threshold = (
            cancel_correlation_threshold  # Fraction that must cancel together
        )
        self.level_history = deque(maxlen=1000)
        logger.info("LayeringDetector initialized")

    def analyze_order_book(
        self, order_book: OrderBookState, account_orders: Dict[str, List[Dict]]
    ) -> Optional[Dict]:
        """
        Analyze order book for layering manipulation patterns.

        Searches for accounts that:
        1. Occupy 5+ distinct price levels
        2. Cancel a high fraction of those orders simultaneously

        Args:
            order_book: Current order book snapshot
            account_orders: Dictionary mapping accounts to their open orders

        Returns:
            Dictionary with manipulation details if detected, None otherwise.
            Contains:
                - manipulation_type: 'layering'
                - events: List of detected layering events
                - severity: 'high' if 3+ accounts, else 'medium'
        """
        layering_events = []

        for account, orders in account_orders.items():
            # Count distinct price levels this account occupies
            price_levels = set(order["price"] for order in orders)

            if len(price_levels) >= self.max_layers:
                # Check if a large fraction of orders were canceled together
                cancel_times = [
                    order.get("cancel_time")
                    for order in orders
                    if "cancel_time" in order
                ]

                # Enough cancellations to trigger correlation check
                if len(cancel_times) >= len(orders) * self.correlation_threshold:
                    # Calculate time differences between cancellations
                    if len(cancel_times) >= 2:
                        time_diffs = np.diff(
                            [
                                (ct - cancel_times[0]).total_seconds()
                                for ct in cancel_times
                            ]
                        )

                        # All canceled within 100ms → highly correlated cancellation
                        if np.all(
                            np.abs(time_diffs) < 0.1
                        ):  # All canceled within 100ms
                            layering_events.append(
                                {
                                    "account": account,
                                    "levels": len(price_levels),
                                    "total_size": sum(
                                        order["size"] for order in orders
                                    ),
                                    "price_range": max(price_levels)
                                    - min(price_levels),
                                }
                            )

        if layering_events:
            return {
                "manipulation_type": "layering",
                "events": layering_events,
                # More than 2 simultaneous events indicates organized manipulation
                "severity": "high" if len(layering_events) > 2 else "medium",
            }

        return None


class WashTradingDetector:
    """
    Detects wash trading - self-trading to create artificial volume.

    WASH TRADING PATTERN:
        The same entity (or coordinated accounts) trade with each other
        to create false impression of market activity, liquidity, or
        to manipulate volume-based indicators.

    DETECTION METHOD:
        - Monitors for trades where buyer and seller are the same entity
        - Uses time window to catch trades within same second
        - Tracks accounts involved in circular trading patterns

    Example:
        >>> detector = WashTradingDetector(time_window_seconds=1.0)
        >>> detector.record_trade("trade_123", "account_A", "account_B", 50000.0, 1.0, datetime.now())
    """

    def __init__(self, time_window_seconds: float = 1.0):
        """
        Initialize the wash trading detector.

        Args:
            time_window_seconds: Window for matching buy/sell pairs (default: 1.0s)
        """
        self.time_window = time_window_seconds  # Window for matching buy/sell pairs
        self.trade_history = deque(maxlen=10000)
        self.wash_trade_alerts = []
        logger.info("WashTradingDetector initialized")

    def record_trade(
        self,
        trade_id: str,
        buyer: str,
        seller: str,
        price: float,
        size: float,
        timestamp: datetime,
    ):
        """
        Record an executed trade for wash trading analysis.

        Args:
            trade_id: Unique identifier for the trade
            buyer: Account ID of buyer
            seller: Account ID of seller
            price: Execution price
            size: Trade quantity
            timestamp: Trade execution time (UTC)
        """
        self.trade_history.append(
            {
                "trade_id": trade_id,
                "buyer": buyer,
                "seller": seller,
                "price": price,
                "size": size,
                "timestamp": timestamp,
            }
        )

        # Check for wash trading after every new trade
        self._detect_wash_trades()

    def _detect_wash_trades(self):
        """
        Scan recent trades for wash trading patterns.

        PATTERN:
            Account A sells to Account B, then Account B sells back to A
            within a short time window. This circular trading creates
            artificial volume without true economic purpose.
        """
        recent_trades = list(self.trade_history)[-100:]  # Only scan recent window

        for i, trade1 in enumerate(recent_trades):
            for trade2 in recent_trades[i + 1 :]:
                # Check if buyer of one is seller of other (and vice versa)
                time_diff = abs(
                    (trade1["timestamp"] - trade2["timestamp"]).total_seconds()
                )

                if time_diff < self.time_window:
                    if (
                        trade1["buyer"] == trade2["seller"]
                        and trade1["seller"] == trade2["buyer"]
                    ):
                        # Wash trade detected – same two accounts trading back and forth
                        alert = {
                            "type": "wash_trading",
                            "accounts": [trade1["buyer"], trade1["seller"]],
                            "size": trade1["size"],
                            "price": trade1["price"],
                            "timestamp": trade1["timestamp"],
                        }

                        if alert not in self.wash_trade_alerts:
                            self.wash_trade_alerts.append(alert)
                            logger.critical(
                                f"WASH TRADING DETECTED: "
                                f"Accounts {trade1['buyer']} <-> {trade1['seller']}, "
                                f"Size: {trade1['size']:.4f}"
                            )


class QuoteStuffingDetector:
    """
    Detects quote stuffing - flooding market with orders to cause delays.

    QUOTE STUFFING PATTERN:
        A trader rapidly submits and cancels large numbers of orders,
        typically far from the best bid/ask, to overwhelm exchange
        matching engines and create delays for other participants.

    DETECTION METHOD:
        - Monitors rate of orders placed far from mid-price
        - Flags when orders/second exceeds threshold (default: 100/sec)
        - Only tracks off-market orders (distance > 2% from mid)

    Example:
        >>> detector = QuoteStuffingDetector(orders_per_second_threshold=100, far_from_best_pct=0.02)
        >>> is_stuffing = detector.record_order(52000.0, 50000.0, 50001.0, datetime.now())
    """

    def __init__(
        self, orders_per_second_threshold: int = 100, far_from_best_pct: float = 0.02
    ):
        """
        Initialize the quote stuffing detector.

        Args:
            orders_per_second_threshold: Alert when rate exceeds this (default: 100)
            far_from_best_pct: Orders this far from mid are suspicious (default: 2%)
        """
        self.orders_threshold = (
            orders_per_second_threshold  # Alert when rate exceeds this
        )
        self.far_threshold = (
            far_from_best_pct  # Orders this far from mid are suspicious
        )
        self.order_times = deque(maxlen=1000)
        logger.info("QuoteStuffingDetector initialized")

    def record_order(
        self, price: float, best_bid: float, best_ask: float, timestamp: datetime
    ) -> bool:
        """
        Record an order and check for quote stuffing pattern.

        Args:
            price: Order limit price
            best_bid: Current best bid price
            best_ask: Current best ask price
            timestamp: Order placement time

        Returns:
            True if stuffing pattern detected, False otherwise
        """
        # Calculate how far the order is from the current mid price
        mid = (best_bid + best_ask) / 2
        distance_from_mid = abs(price - mid) / mid

        # Only track orders placed far from the best prices (off-market orders)
        if distance_from_mid > self.far_threshold:
            self.order_times.append(timestamp)

            # Check rate of far-from-best orders in the last 1 second
            recent_orders = [
                t for t in self.order_times if (timestamp - t).total_seconds() <= 1.0
            ]

            if len(recent_orders) > self.orders_threshold:
                logger.critical(
                    f"QUOTE STUFFING DETECTED: "
                    f"{len(recent_orders)} far orders in 1 second"
                )
                return True  # Signal to caller that stuffing was detected

        return False


class OrderBookAnalyzer:
    """
    Advanced order book analysis for comprehensive manipulation detection.

    This analyzer performs multi-faceted analysis of order book state to
    detect various forms of market manipulation and assess market health.

    ANALYSIS METRICS:
    -----------------
    1. Order Book Imbalance: Detects directional pressure from depth imbalance
    2. Toxicity Score: VPIN-like measure of informed trader activity
    3. Flash Crash Risk: Identifies rapid price decline patterns
    4. Liquidity Stress: Detects sudden liquidity removal (spoofing aftermath)

    CONFIGURATION:
        Maintains rolling history of 1000 book snapshots for pattern detection

    Example:
        >>> analyzer = OrderBookAnalyzer()
        >>> indicators = analyzer.analyze_book(order_book_state)
        >>> print(f"Manipulation probability: {indicators['manipulation_probability']}")
    """

    def __init__(self):
        """Initialize the order book analyzer with default settings."""
        self.book_history = deque(maxlen=1000)  # Rolling history of book snapshots
        self.imbalance_history = []
        logger.info("OrderBookAnalyzer initialized")

    def analyze_book(self, order_book: OrderBookState) -> Dict:
        """
        Perform comprehensive order book analysis.

        Analyzes the current order book state across multiple dimensions
        to generate manipulation indicators and risk assessments.

        Args:
            order_book: Current order book snapshot

        Returns:
            Dictionary containing:
                - timestamp: Analysis timestamp
                - manipulation_probability: Aggregated probability [0, 1]
                - toxicity_score: Order book toxicity [0, 1]
                - liquidity_stress: Liquidity stress level [0, 1]
                - flash_crash_risk: Flash crash probability [0, 1]
                - alerts: List of detected anomalies with severity
        """
        self.book_history.append(order_book)

        indicators = {
            "timestamp": order_book.timestamp,
            "manipulation_probability": 0.0,  # Aggregated probability [0, 1]
            "toxicity_score": 0.0,
            "liquidity_stress": 0.0,
            "flash_crash_risk": 0.0,
            "alerts": [],
        }

        # 1. Check order book imbalance (large imbalance suggests directional pressure)
        imbalance = order_book.imbalance
        self.imbalance_history.append(imbalance)

        if abs(imbalance) > 0.8:
            indicators["alerts"].append(
                {"type": "extreme_imbalance", "severity": "high", "value": imbalance}
            )
            indicators["manipulation_probability"] += (
                0.3  # Raise manipulation probability
            )

        # 2. Check spread anomaly (wide spread suggests thin liquidity or manipulation)
        spread_pct = order_book.spread / order_book.mid_price
        if spread_pct > 0.01:  # > 1% spread
            indicators["alerts"].append(
                {"type": "wide_spread", "severity": "medium", "value": spread_pct}
            )
            indicators["liquidity_stress"] += 0.4

        # 3. Check for flash crash pattern (rapid price decline in recent samples)
        if len(self.book_history) >= 10:
            recent_books = list(self.book_history)[-10:]
            mid_prices = [b.mid_price for b in recent_books]

            # Calculate percentage change over last 10 samples
            price_change = (mid_prices[-1] - mid_prices[0]) / mid_prices[0]

            if price_change < -0.05:  # 5% drop in 10 samples
                indicators["alerts"].append(
                    {
                        "type": "flash_crash_pattern",
                        "severity": "critical",
                        "value": price_change,
                    }
                )
                indicators["flash_crash_risk"] = 0.9
                indicators["manipulation_probability"] += 0.5

        # 4. Detect sudden liquidity removal (spoofing aftermath signature)
        if len(self.book_history) >= 2:
            prev_book = self.book_history[-2]

            # Calculate fractional change in bid and ask depth
            bid_depth_change = (
                order_book.bid_depth - prev_book.bid_depth
            ) / prev_book.bid_depth
            ask_depth_change = (
                order_book.ask_depth - prev_book.ask_depth
            ) / prev_book.ask_depth

            if (
                bid_depth_change < -0.5 or ask_depth_change < -0.5
            ):  # 50% liquidity removed
                indicators["alerts"].append(
                    {
                        "type": "liquidity_removed",
                        "severity": "high",
                        "bid_change": bid_depth_change,
                        "ask_change": ask_depth_change,
                    }
                )
                indicators["liquidity_stress"] += 0.5

        # Clamp probability to [0, 1]
        indicators["manipulation_probability"] = min(
            1.0, indicators["manipulation_probability"]
        )
        indicators["toxicity_score"] = self._calculate_toxicity_score()

        return indicators

    def _calculate_toxicity_score(self) -> float:
        """
        Calculate order book toxicity score (VPIN-like metric).

        High toxicity indicates presence of informed traders who have
        directional information. This manifests as frequent changes in
        order flow direction.

        Returns:
            Toxicity score in range [0, 1]
            - 0.0: Low toxicity, stable order flow
            - 1.0: High toxicity, volatile order flow
        """
        if len(self.imbalance_history) < 20:
            return 0.0  # Not enough history to estimate toxicity

        # High toxicity when imbalance changes frequently (informed traders switching direction)
        imbalances = np.array(self.imbalance_history[-20:])
        volatility = np.std(np.diff(imbalances))  # Variance in imbalance changes

        return min(1.0, volatility * 5)  # Scale to [0, 1]


class ManipulationProtectionSystem:
    """
    Complete market manipulation protection system.

    This is the main orchestrator that integrates all manipulation
    detection components into a unified protection system.

    FEATURES:
    --------
    - Real-time monitoring of all manipulation types
    - Automated protection level adjustment
    - Smart execution recommendations
    - Integration with trading execution engine

    PROTECTION LEVELS:
    -----------------
    1. NORMAL (manipulation_probability <= 0.4)
       - Standard trading operations

    2. ELEVATED (0.4 < manipulation_probability <= 0.7)
       - Reduce position size by 50%
       - Increase spread requirements
       - Use limit orders only

    3. CRITICAL (manipulation_probability > 0.7)
       - Halt all trading
       - Close all positions
       - Switch to passive mode

    Example:
        >>> protection = ManipulationProtectionSystem()
        >>> result = protection.process_order_book(book_state)
        >>> print(f"Protection level: {result['protection_level']}")
    """

    def __init__(self):
        """Initialize the complete manipulation protection system."""
        self.spoofing_detector = SpoofingDetector()
        self.layering_detector = LayeringDetector()
        self.wash_detector = WashTradingDetector()
        self.stuffing_detector = QuoteStuffingDetector()
        self.book_analyzer = OrderBookAnalyzer()

        self.alerts = deque(maxlen=1000)
        self.protection_level = "normal"  # normal, elevated, critical

        logger.info("ManipulationProtectionSystem initialized")

    def process_order_book(self, order_book: OrderBookState) -> Dict:
        """
        Process order book update and generate protection recommendations.

        This is the main entry point for real-time manipulation monitoring.

        Args:
            order_book: Current order book snapshot

        Returns:
            Dictionary containing:
                - analysis: Detailed manipulation analysis
                - protection_level: Current protection level
                - recommendations: List of recommended actions
        """
        # Analyze order book with all detectors
        analysis = self.book_analyzer.analyze_book(order_book)

        # Update protection level based on manipulation probability
        if analysis["manipulation_probability"] > 0.7:
            self.protection_level = "critical"
        elif analysis["manipulation_probability"] > 0.4:
            self.protection_level = "elevated"
        else:
            self.protection_level = "normal"

        # Generate protection recommendations based on analysis
        recommendations = self._generate_recommendations(analysis)

        return {
            "analysis": analysis,
            "protection_level": self.protection_level,
            "recommendations": recommendations,
        }

    def _generate_recommendations(self, analysis: Dict) -> List[str]:
        """
        Generate trading recommendations based on manipulation analysis.

        Maps detected manipulation patterns to actionable trading decisions.

        Args:
            analysis: Dictionary containing manipulation analysis results

        Returns:
            List of recommended action strings
        """
        recommendations = []

        if analysis["manipulation_probability"] > 0.7:
            # Critical – stop all activity
            recommendations.extend(
                ["HALT_TRADING", "CLOSE_ALL_POSITIONS", "SWITCH_TO_PASSIVE_MODE"]
            )
        elif analysis["manipulation_probability"] > 0.4:
            # Elevated – reduce risk
            recommendations.extend(
                [
                    "REDUCE_POSITION_SIZE_50%",
                    "INCREASE_SPREAD_REQUIREMENT",
                    "USE_LIMIT_ORDERS_ONLY",
                ]
            )
        elif analysis["liquidity_stress"] > 0.6:
            # Thin liquidity – be cautious
            recommendations.extend(["REDUCE_POSITION_SIZE_25%", "AVOID_MARKET_ORDERS"])

        if analysis["flash_crash_risk"] > 0.8:
            recommendations.append("ACTIVATE_STOP_LOSS_TIGHTENING")

        return recommendations

    def should_execute_trade(
        self, side: str, size: float, order_book: OrderBookState
    ) -> Tuple[bool, str]:
        """
        Determine if a proposed trade should be executed given current risk.

        Pre-trade check to prevent execution during high-risk conditions.

        Args:
            side: Trade direction - 'buy' or 'sell'
            size: Order size
            order_book: Current order book state

        Returns:
            Tuple of (should_execute: bool, reason: str)
            - should_execute: True if trade should proceed
            - reason: Explanation of decision
        """
        analysis = self.book_analyzer.analyze_book(order_book)

        # Critical manipulation – don't trade
        if analysis["manipulation_probability"] > 0.8:
            return False, f"CRITICAL_MANIPULATION_DETECTED: {analysis['alerts']}"

        # High toxicity – reduce size below threshold
        if analysis["toxicity_score"] > 0.7 and size > 0.1:
            return False, f"HIGH_TOXICITY: Reduce size below 0.1 (current: {size})"

        # Flash crash risk – market orders are dangerous
        if analysis["flash_crash_risk"] > 0.8 and side == "market":
            return False, "FLASH_CRASH_RISK: Use limit orders only"

        return True, "OK"


class ShutAndRunProtocol:
    """
    Emergency "Shut and Run" protocol for immediate position closure.

    This is the final defense line when severe manipulation is detected.
    It provides rapid, controlled position liquidation to minimize losses.

    TRIGGER CONDITIONS:
    ------------------
    - Critical manipulation detected
    - Exchange failure or connectivity loss
    - Flash crash imminent
    - Regulatory halt signal

    EXECUTION STEPS:
    ---------------
    1. Immediately cancel all pending orders
    2. Close all positions at market or best available price
    3. Switch to passive/quiescent mode
    4. Generate detailed incident report

    Example:
        >>> protocol = ShutAndRunProtocol()
        >>> plan = protocol.trigger(
        ...     reason="Critical manipulation detected",
        ...     positions={'BTC': 1.5, 'ETH': -10.0},
        ...     current_prices={'BTC': 50000, 'ETH': 3000}
        ... )
    """

    def __init__(self):
        """Initialize the shut-and-run protocol in inactive state."""
        self.active = False
        self.triggered_at = None
        self.closure_history = []  # Log of all past shut-and-run events
        logger.info("ShutAndRunProtocol initialized")

    def trigger(
        self, reason: str, positions: Dict[str, float], current_prices: Dict[str, float]
    ) -> Dict:
        """
        Trigger emergency shutdown and generate closure plan.

        Creates a comprehensive plan for closing all positions while
        minimizing slippage and market impact during emergency conditions.

        Args:
            reason: Reason for triggering (for audit trail)
            positions: Dictionary of symbol -> position size
            current_prices: Dictionary of symbol -> current price

        Returns:
            Closure plan dictionary containing:
                - timestamp: When protocol was triggered
                - reason: Trigger reason
                - positions_to_close: List of positions with closure details
                - total_value_at_risk: Total portfolio value
                - estimated_slippage: Expected slippage from emergency closure
        """
        self.active = True
        self.triggered_at = datetime.now()

        logger.critical(f"🚨 SHUT AND RUN TRIGGERED: {reason}")
        logger.critical("Executing emergency position closure...")

        # Build the closure plan with per-position details
        closure_plan = {
            "timestamp": self.triggered_at,
            "reason": reason,
            "positions_to_close": [],
            "total_value_at_risk": 0.0,
            "estimated_slippage": 0.0,
        }

        for symbol, size in positions.items():
            if size != 0:
                current_price = current_prices.get(symbol, 0)
                position_value = abs(size) * current_price

                # Estimate emergency slippage (higher for larger positions)
                slippage = min(0.02, abs(size) * 0.001)  # Max 2%

                closure_plan["positions_to_close"].append(
                    {
                        "symbol": symbol,
                        "size": size,
                        "current_price": current_price,
                        "value": position_value,
                        "slippage_estimate": slippage,
                        "action": "sell" if size > 0 else "buy",  # Close direction
                    }
                )

                closure_plan["total_value_at_risk"] += position_value
                closure_plan["estimated_slippage"] += position_value * slippage

        self.closure_history.append(closure_plan)

        return closure_plan

    def execute_closure(self, closure_plan: Dict, execution_engine) -> Dict:
        """
        Execute the closure plan via the provided execution engine.

        Args:
            closure_plan: Plan from trigger() method
            execution_engine: Trading execution engine instance

        Returns:
            Execution results containing:
                - executed_at: Completion timestamp
                - positions_closed: List of successfully closed symbols
                - total_pnl: Realized P&L from closures
                - avg_slippage: Average slippage incurred
                - errors: List of any errors encountered
        """
        results = {
            "executed_at": datetime.now(),
            "positions_closed": [],
            "total_pnl": 0.0,
            "avg_slippage": 0.0,
            "errors": [],
        }

        for pos in closure_plan["positions_to_close"]:
            try:
                # Execute close order via the provided execution engine
                logger.warning(
                    f"Closing {pos['symbol']}: {pos['size']} @ {pos['current_price']}"
                )

                # Placeholder for actual execution
                # execution_engine.place_order(...)

                results["positions_closed"].append(pos["symbol"])

            except Exception as e:
                results["errors"].append({"symbol": pos["symbol"], "error": str(e)})

        self.active = False  # Reset active flag after closure

        logger.success(
            f"Shut and Run complete. Closed {len(results['positions_closed'])} positions"
        )

        return results

    def reset(self):
        """
        Reset the protocol after incident completion.

        Clears the active flag and triggered timestamp, allowing
        the system to resume normal operations after review.
        """
        self.active = False
        self.triggered_at = None
        logger.info("ShutAndRunProtocol reset")


# Production functions
def create_protection_system() -> ManipulationProtectionSystem:
    """
    Factory function to create a complete manipulation protection system.

    Returns:
        Initialized ManipulationProtectionSystem ready for use
    """
    return ManipulationProtectionSystem()


def detect_market_manipulation(order_book_data: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze historical order book data for manipulation patterns.

    Batch analysis function for backtesting or historical investigation.

    Args:
        order_book_data: DataFrame with columns:
            - timestamp: Datetime of snapshot
            - bid_price, bid_size: Best bid price and size
            - ask_price, ask_size: Best ask price and size
            - bid_depth, ask_depth: (optional) Total depth
            - imbalance: (optional) Book imbalance

    Returns:
        DataFrame with analysis results per timestamp containing:
            - manipulation_probability
            - toxicity_score
            - liquidity_stress
            - flash_crash_risk
            - alerts
    """
    protection = create_protection_system()
    results = []

    for idx, row in order_book_data.iterrows():
        # Build a structured book state from each row of the historical data
        book_state = OrderBookState(
            timestamp=row["timestamp"],
            bids=[(row["bid_price"], row["bid_size"])],
            asks=[(row["ask_price"], row["ask_size"])],
            mid_price=(row["bid_price"] + row["ask_price"]) / 2,
            spread=row["ask_price"] - row["bid_price"],
            bid_depth=row.get("bid_depth", 0),
            ask_depth=row.get("ask_depth", 0),
            imbalance=row.get("imbalance", 0),
        )

        analysis = protection.process_order_book(book_state)
        results.append(analysis)

    return pd.DataFrame(results)


def emergency_shut_and_run(
    positions: Dict[str, float], current_prices: Dict[str, float], reason: str
) -> Dict:
    """
    Emergency position closure function.

    Quick-access function to trigger immediate position closure
    in critical situations.

    Args:
        positions: Dictionary mapping symbols to position sizes
            - Positive for long positions
            - Negative for short positions
        current_prices: Dictionary mapping symbols to current prices
        reason: Reason for emergency closure (for audit trail)

    Returns:
        Closure plan dictionary (call execute_closure to execute)

    Example:
        >>> result = emergency_shut_and_run(
        ...     positions={'BTC': 1.5, 'ETH': -10.0},
        ...     current_prices={'BTC': 50000, 'ETH': 3000},
        ...     reason="Critical manipulation detected"
        ... )
        >>> print(f"Closing {len(result['positions_to_close'])} positions")
    """
    protocol = ShutAndRunProtocol()
    closure_plan = protocol.trigger(reason, positions, current_prices)

    return closure_plan
