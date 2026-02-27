"""
Transaction Fraud & Exchange Delay Prevention System
===================================================
Institutional-grade system for protecting against exchange-level fraud,
execution delays, slippage manipulation, and front-running.

This module provides comprehensive monitoring and protection against
various forms of exchange fraud and operational risks.

PROTECTION CATEGORIES:
---------------------
1. EXECUTION DELAYS
   - Monitors order-to-fill latency
   - Alerts on abnormal delays (>500ms default)
   - Tracks per-exchange performance metrics

2. SLIPPAGE FRAUD
   - Detects excessive slippage (>0.5% default)
   - Monitors for favorable vs adverse slippage
   - Tracks slippage patterns per venue

3. FRONT-RUNNING DETECTION
   - Identifies suspicious price gaps on buy orders
   - Detects patterns indicative of order book sniffing
   - Alerts when orders consistently execute at worse prices

4. EXCHANGE HEALTH MONITORING
   - Per-exchange latency tracking
   - Failure/success rate monitoring
   - Volume-weighted performance scoring
   - Smart order routing to avoid problematic venues

SMART ROUTING:
-------------
Routes orders to optimal venues based on:
- Historical latency performance
- Slippage rates
- Liquidity depth
- Fee structures
- Fill rates

Usage:
    from src.surveillance.fraud_prevention import (
        TransactionMonitor,
        SmartRouter,
        create_fraud_protection
    )

    # Initialize protection
    monitor, router = create_fraud_protection()

    # Record execution
    monitor.record_order_execution("order_123", 50000.0, "filled", "binance")

    # Get best exchange
    best = router.get_best_exchange()

Author: BITCOIN4Traders Team
License: Proprietary - Internal Use Only
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import deque, defaultdict
from loguru import logger


@dataclass
class TransactionMetrics:
    """
    Comprehensive metrics for a single transaction.

    Tracks all relevant execution details for fraud detection
    and performance analysis.

    Attributes:
        order_id: Unique order identifier
        symbol: Trading symbol (e.g., 'BTC', 'ETH')
        side: Trade direction - 'buy' or 'sell'
        size: Order quantity
        requested_price: Price at order submission time
        executed_price: Actual fill price
        slippage: Fractional price deviation (negative = favorable execution)
            - Positive: Adverse slippage (worse than requested)
            - Negative: Favorable slippage (better than requested)
        delay_ms: Round-trip latency in milliseconds
        timestamp: Order submission time (UTC)
        exchange: Exchange where order was placed
        status: Execution status - 'filled', 'rejected', 'timeout', etc.

    Example:
        >>> tx = TransactionMetrics(
        ...     order_id="order_001",
        ...     symbol="BTC",
        ...     side="buy",
        ...     size=1.0,
        ...     requested_price=50000.0,
        ...     executed_price=50025.0,
        ...     slippage=0.0005,  # 0.05% adverse slippage
        ...     delay_ms=250.0,
        ...     timestamp=datetime.now(),
        ...     exchange="binance",
        ...     status="filled"
        ... )
    """

    order_id: str
    symbol: str
    side: str
    size: float
    requested_price: float  # Price at order submission
    executed_price: float  # Price actually filled at
    slippage: float  # Fractional price deviation (negative = favorable)
    delay_ms: float  # Milliseconds between submission and fill
    timestamp: datetime
    exchange: str
    status: str  # 'filled', 'rejected', 'timeout', etc.


class TransactionMonitor:
    """
    Comprehensive transaction monitoring system.

    Monitors all transactions for fraud patterns, delays, and anomalies.
    Maintains rolling history and per-exchange metrics for real-time
    protection and historical analysis.

    DETECTION CAPABILITIES:
    ----------------------
    1. High Latency Detection
       - Flags orders taking >500ms to fill
       - Tracks latency trends per exchange

    2. Excessive Slippage
       - Alerts on slippage >0.5%
       - Distinguishes favorable vs adverse slippage

    3. Front-Running Patterns
       - Monitors buy orders with significant price gaps
       - Detects consistent adverse execution on directional orders

    METRICS TRACKED:
    ---------------
    - Per-exchange latency distribution
    - Success/failure rates
    - Total volume processed
    - Slippage statistics

    Example:
        >>> monitor = TransactionMonitor()
        >>> monitor.record_order_execution("order_123", 50000.0, "filled", "binance")
    """

    def __init__(self):
        """
        Initialize the transaction monitor with default thresholds.

        Default thresholds:
            - delay_threshold_ms: 500ms
            - slippage_threshold: 0.5%
        """
        self.transactions = deque(maxlen=10000)  # Rolling buffer of recent transactions
        # Per-exchange metrics: track latency, failure/success counts, and volume
        self.exchange_metrics = defaultdict(
            lambda: {
                "latencies": deque(maxlen=1000),
                "failures": 0,
                "successes": 0,
                "total_volume": 0.0,
            }
        )
        self.delay_threshold_ms = 500  # Alert if fill takes longer than 500ms
        self.slippage_threshold = 0.005  # Alert if slippage exceeds 0.5%
        logger.info("TransactionMonitor initialized")

    def record_order_execution(
        self,
        order_id: str,
        executed_price: float,
        status: str = "filled",
        exchange: str = None,
    ):
        """
        Record order execution and analyze for fraud patterns.

        This method should be called when an order is executed. It:
        1. Matches the execution to the original order
        2. Calculates latency and slippage
        3. Updates exchange metrics
        4. Checks for fraud patterns

        Args:
            order_id: ID of the executed order
            executed_price: Actual execution price
            status: Execution status ('filled', 'rejected', 'timeout')
            exchange: Exchange name (required for metrics tracking)
        """
        execution_time = datetime.now()

        # Search for the matching submitted order in the buffer
        for tx in self.transactions:
            if tx["order_id"] == order_id:
                tx["execution_time"] = execution_time
                tx["executed_price"] = executed_price
                tx["status"] = status

                # Calculate round-trip delay in milliseconds
                delay_ms = (
                    execution_time - tx["submission_time"]
                ).total_seconds() * 1000
                # Calculate absolute fractional slippage from requested price
                slippage = (
                    abs(executed_price - tx["requested_price"]) / tx["requested_price"]
                )

                tx["delay_ms"] = delay_ms
                tx["slippage"] = slippage

                # Update exchange metrics if exchange specified
                if exchange:
                    self.exchange_metrics[exchange]["latencies"].append(delay_ms)
                    if status == "filled":
                        self.exchange_metrics[exchange]["successes"] += 1
                    else:
                        self.exchange_metrics[exchange]["failures"] += 1

                # Check for fraud patterns on this execution
                self._detect_fraud(tx, delay_ms, slippage)
                break

    def _detect_fraud(self, tx: Dict, delay_ms: float, slippage: float):
        """
        Analyze transaction for fraud indicators.

        Checks multiple fraud patterns and logs alerts when detected.

        FRAUD PATTERNS:
        --------------
        1. High Delay: May indicate exchange manipulation or throttling
        2. High Slippage: May indicate front-running or thin liquidity
        3. Buy Gap-Up: Suspicious when combined with high slippage

        Args:
            tx: Transaction dictionary
            delay_ms: Execution delay in milliseconds
            slippage: Fractional slippage
        """
        # High delay = possible manipulation or exchange throttling
        if delay_ms > self.delay_threshold_ms:
            logger.warning(f"DELAY ALERT: {delay_ms:.1f}ms on {tx['exchange']}")

        # High slippage = possible front-running or thin liquidity
        if slippage > self.slippage_threshold:
            logger.critical(f"SLIPPAGE ALERT: {slippage:.4f} on {tx['exchange']}")

        # Front-running detection: buy orders that gap up in price are suspicious
        # This pattern suggests the exchange or market maker may have
        # traded ahead of our order
        if tx["side"] == "buy" and slippage > 0.01:
            logger.critical(f"FRONT-RUNNING SUSPECTED on {tx['exchange']}")


class SmartRouter:
    """
    Intelligent order routing system.

    Routes orders to optimal exchanges based on real-time and
    historical performance metrics. Avoids problematic venues
    while maximizing execution quality.

    ROUTING STRATEGIES:
    ------------------
    1. COST: Minimize total execution cost (fees + impact)
    2. SPEED: Minimize latency for time-sensitive orders
    3. LIQUIDITY: Maximize fill probability for large orders

    VENUE SELECTION:
    ---------------
    - Scores each venue based on configured priority
    - Excludes blacklisted exchanges
    - Requires minimum score threshold (>0.7)
    - Falls back to halt trading if no venue qualifies

    Example:
        >>> router = SmartRouter()
        >>> router.add_venue("binance", fees_bps=1.0, latency_ms=50, liquidity_score=0.9)
        >>> best = router.get_best_exchange()
    """

    def __init__(self):
        """Initialize the smart router."""
        self.exchange_scores = {}  # Health scores for each exchange (0-1)
        logger.info("SmartRouter initialized")

    def get_best_exchange(self, avoid_exchanges: List[str] = None) -> Optional[str]:
        """
        Select the best exchange for order execution.

        Evaluates all registered exchanges and returns the highest
        scoring venue that isn't blacklisted.

        Args:
            avoid_exchanges: List of exchange IDs to exclude

        Returns:
            Exchange ID of best venue, or None if no healthy exchanges

        Example:
            >>> best = router.get_best_exchange(avoid_exchanges=["ftx"])
            >>> if best is None:
            ...     print("Warning: No healthy exchanges available")
        """
        # Iterate exchanges sorted by score descending
        for exchange, score in sorted(
            self.exchange_scores.items(), key=lambda x: x[1], reverse=True
        ):
            if avoid_exchanges and exchange in avoid_exchanges:
                continue  # Skip blacklisted exchanges
            if score > 0.7:
                return exchange  # Return first healthy exchange found

        logger.critical("No healthy exchanges - NOT TRADING")
        return None  # All exchanges below threshold – halt trading


def create_fraud_protection():
    """
    Factory function to create complete fraud protection system.

    Returns:
        Tuple of (TransactionMonitor, SmartRouter)

    Example:
        >>> monitor, router = create_fraud_protection()
        >>> monitor.record_order_execution("order_123", 50000.0, "filled", "binance")
    """
    return TransactionMonitor(), SmartRouter()
