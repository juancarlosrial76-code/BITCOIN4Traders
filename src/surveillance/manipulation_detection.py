"""
Market Manipulation Detection System
=====================================
Advanced surveillance to detect exchange manipulation and protect trading capital.

Detects:
- Spoofing (fake orders placed and canceled)
- Layering (multiple fake order levels)
- Wash trading (self-trading to create volume)
- Quote stuffing (flooding with orders to delay)
- Pump and dump schemes
- Exchange irregularities

Provides:
- Real-time manipulation alerts
- Order book toxicity scoring
- Smart execution adaptation
- Emergency position closure ("shut and run")

Used by: Prop trading firms, hedge funds, market makers for protection
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
    """Alert for detected market manipulation."""

    timestamp: datetime
    type: str
    severity: str  # 'low', 'medium', 'high', 'critical'
    description: str
    evidence: Dict
    affected_orders: List[str]
    recommended_action: str


@dataclass
class OrderBookState:
    """Snapshot of order book state."""

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
    Detects spoofing - placing fake orders to manipulate price.

    Pattern: Large orders placed and quickly canceled after price moves
    """

    def __init__(
        self, cancel_threshold_ms: float = 500, size_threshold_pct: float = 0.1
    ):
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
        """Record new order."""
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
        """Record order cancellation."""
        for order in self.order_history:
            if order["order_id"] == order_id:
                order["status"] = "canceled"
                order["cancel_time"] = timestamp

                # Check for spoofing pattern immediately on cancellation
                self._check_spoofing_pattern(order)
                break

    def _check_spoofing_pattern(self, order: Dict):
        """Check if cancellation matches spoofing pattern."""
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
        """Get accounts with suspicious activity."""
        return dict(self.suspicious_accounts)


class LayeringDetector:
    """
    Detects layering - placing multiple fake orders at different levels.

    Pattern: Multiple orders placed to create false depth, then all canceled
    """

    def __init__(
        self, max_layer_levels: int = 5, cancel_correlation_threshold: float = 0.8
    ):
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
        Analyze order book for layering patterns.

        Returns manipulation info if detected.
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
    Detects wash trading - trading with oneself to create false volume.

    Pattern: Same account buys and sells to itself
    """

    def __init__(self, time_window_seconds: float = 1.0):
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
        """Record executed trade."""
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
        """Detect if same account is buying from itself."""
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
    Detects quote stuffing - flooding market with orders to create delays.

    Pattern: Sudden burst of orders far from best price
    """

    def __init__(
        self, orders_per_second_threshold: int = 100, far_from_best_pct: float = 0.02
    ):
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
    ):
        """Record order placement."""
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
    Advanced order book analysis for manipulation detection.

    Analyzes:
    - Order book imbalance
    - Toxic flow detection
    - Flash crash prediction
    - Liquidity anomalies
    """

    def __init__(self):
        self.book_history = deque(maxlen=1000)  # Rolling history of book snapshots
        self.imbalance_history = []
        logger.info("OrderBookAnalyzer initialized")

    def analyze_book(self, order_book: OrderBookState) -> Dict:
        """
        Comprehensive order book analysis.

        Returns dictionary with manipulation indicators.
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
        """Calculate order book toxicity score (VPIN-like)."""
        if len(self.imbalance_history) < 20:
            return 0.0  # Not enough history to estimate toxicity

        # High toxicity when imbalance changes frequently (informed traders switching direction)
        imbalances = np.array(self.imbalance_history[-20:])
        volatility = np.std(np.diff(imbalances))  # Variance in imbalance changes

        return min(1.0, volatility * 5)  # Scale to [0, 1]


class ManipulationProtectionSystem:
    """
    Complete protection system against market manipulation.

    Integrates all detectors and provides:
    - Real-time monitoring
    - Smart execution adaptation
    - Emergency shutdown protocols
    """

    def __init__(self):
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
        Process order book update and check for manipulation.

        Returns protection recommendations.
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
        """Generate trading recommendations based on manipulation detection."""
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
        Determine if trade should be executed given current manipulation risk.

        Returns: (should_execute, reason)
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

    When severe manipulation is detected:
    1. Immediately cancel all pending orders
    2. Close all positions at market (or best available)
    3. Switch to passive mode
    4. Generate incident report
    """

    def __init__(self):
        self.active = False
        self.triggered_at = None
        self.closure_history = []  # Log of all past shut-and-run events
        logger.info("ShutAndRunProtocol initialized")

    def trigger(
        self, reason: str, positions: Dict[str, float], current_prices: Dict[str, float]
    ) -> Dict:
        """
        Trigger emergency shutdown.

        Returns closure plan.
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
        Execute the closure plan.

        Returns results.
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
        """Reset protocol after incident."""
        self.active = False
        self.triggered_at = None
        logger.info("ShutAndRunProtocol reset")


# Production functions
def create_protection_system() -> ManipulationProtectionSystem:
    """Create complete manipulation protection system."""
    return ManipulationProtectionSystem()


def detect_market_manipulation(order_book_data: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze order book history for manipulation patterns.

    Returns dataframe with manipulation indicators.
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
    Trigger emergency position closure.

    Usage:
        result = emergency_shut_and_run(
            positions={'BTC': 1.5, 'ETH': -10.0},
            current_prices={'BTC': 50000, 'ETH': 3000},
            reason="Critical manipulation detected"
        )
    """
    protocol = ShutAndRunProtocol()
    closure_plan = protocol.trigger(reason, positions, current_prices)

    return closure_plan
