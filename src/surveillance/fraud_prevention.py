"""
Transaction Fraud & Delay Prevention System
===========================================
Protects against exchange delays, slippage fraud, and front-running
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
    """Metrics for a single transaction."""

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
    """Monitors all transactions for fraud, delays, and anomalies."""

    def __init__(self):
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
        """Record order execution and check for anomalies."""
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

                # Check for fraud patterns on this execution
                self._detect_fraud(tx, delay_ms, slippage)
                break

    def _detect_fraud(self, tx: Dict, delay_ms: float, slippage: float):
        """Detect fraud patterns."""
        # High delay = possible manipulation or exchange throttling
        if delay_ms > self.delay_threshold_ms:
            logger.warning(f"DELAY ALERT: {delay_ms:.1f}ms on {tx['exchange']}")

        # High slippage = possible front-running or thin liquidity
        if slippage > self.slippage_threshold:
            logger.critical(f"SLIPPAGE ALERT: {slippage:.4f} on {tx['exchange']}")

        # Front-running detection: buy orders that gap up in price are suspicious
        if tx["side"] == "buy" and slippage > 0.01:
            logger.critical(f"FRONT-RUNNING SUSPECTED on {tx['exchange']}")


class SmartRouter:
    """Routes orders to best exchange avoiding manipulation."""

    def __init__(self):
        self.exchange_scores = {}  # Health scores for each exchange (0-1)
        logger.info("SmartRouter initialized")

    def get_best_exchange(self, avoid_exchanges: List[str] = None) -> Optional[str]:
        """Get best exchange avoiding problematic ones."""
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


# Production function
def create_fraud_protection():
    """Create complete fraud protection system."""
    return TransactionMonitor(), SmartRouter()
