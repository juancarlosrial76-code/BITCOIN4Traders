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
    requested_price: float
    executed_price: float
    slippage: float
    delay_ms: float
    timestamp: datetime
    exchange: str
    status: str


class TransactionMonitor:
    """Monitors all transactions for fraud, delays, and anomalies."""
    
    def __init__(self):
        self.transactions = deque(maxlen=10000)
        self.exchange_metrics = defaultdict(lambda: {
            'latencies': deque(maxlen=1000),
            'failures': 0,
            'successes': 0,
            'total_volume': 0.0
        })
        self.delay_threshold_ms = 500
        self.slippage_threshold = 0.005
        logger.info("TransactionMonitor initialized")
    
    def record_order_execution(self, order_id: str, executed_price: float, 
                              status: str = 'filled', exchange: str = None):
        """Record order execution and check for anomalies."""
        execution_time = datetime.now()
        
        for tx in self.transactions:
            if tx['order_id'] == order_id:
                tx['execution_time'] = execution_time
                tx['executed_price'] = executed_price
                tx['status'] = status
                
                delay_ms = (execution_time - tx['submission_time']).total_seconds() * 1000
                slippage = abs(executed_price - tx['requested_price']) / tx['requested_price']
                
                tx['delay_ms'] = delay_ms
                tx['slippage'] = slippage
                
                # Check for fraud
                self._detect_fraud(tx, delay_ms, slippage)
                break
    
    def _detect_fraud(self, tx: Dict, delay_ms: float, slippage: float):
        """Detect fraud patterns."""
        # High delay = possible manipulation
        if delay_ms > self.delay_threshold_ms:
            logger.warning(f"DELAY ALERT: {delay_ms:.1f}ms on {tx['exchange']}")
        
        # High slippage = possible front-running
        if slippage > self.slippage_threshold:
            logger.critical(f"SLIPPAGE ALERT: {slippage:.4f} on {tx['exchange']}")
        
        # Front-running detection
        if tx['side'] == 'buy' and slippage > 0.01:
            logger.critical(f"FRONT-RUNNING SUSPECTED on {tx['exchange']}")


class SmartRouter:
    """Routes orders to best exchange avoiding manipulation."""
    
    def __init__(self):
        self.exchange_scores = {}
        logger.info("SmartRouter initialized")
    
    def get_best_exchange(self, avoid_exchanges: List[str] = None) -> Optional[str]:
        """Get best exchange avoiding problematic ones."""
        for exchange, score in sorted(self.exchange_scores.items(), 
                                     key=lambda x: x[1], reverse=True):
            if avoid_exchanges and exchange in avoid_exchanges:
                continue
            if score > 0.7:
                return exchange
        
        logger.critical("No healthy exchanges - NOT TRADING")
        return None


# Production function
def create_fraud_protection():
    """Create complete fraud protection system."""
    return TransactionMonitor(), SmartRouter()
