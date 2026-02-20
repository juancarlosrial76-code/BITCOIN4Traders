"""
Live Data Quality Monitor
==========================

Real-time monitoring of data quality for live trading systems.

Features:
- Continuous quality monitoring
- Real-time alerts for data degradation
- Automatic source switching
- Quality trend analysis
- Live comparison between sources

Use Cases:
- Production trading systems
- Multi-source data feeds
- Quality-based source selection
- Data pipeline monitoring
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque
import threading
import time
from enum import Enum
from loguru import logger


class DataQualityAlert(Enum):
    """Types of data quality alerts."""

    QUALITY_DEGRADATION = "quality_degradation"
    MISSING_DATA_SPIKE = "missing_data_spike"
    OUTLIER_DETECTED = "outlier_detected"
    STALE_DATA = "stale_data"
    SOURCE_DISCREPANCY = "source_discrepancy"
    LIQUIDITY_DROP = "liquidity_drop"


@dataclass
class QualityAlert:
    """Data quality alert."""

    alert_type: DataQualityAlert
    severity: str  # 'low', 'medium', 'high', 'critical'
    message: str
    timestamp: datetime
    source: str
    metric_value: float
    threshold: float

    def to_dict(self) -> Dict:
        return {
            "type": self.alert_type.value,
            "severity": self.severity,
            "message": self.message,
            "timestamp": self.timestamp.isoformat(),
            "source": self.source,
            "metric_value": self.metric_value,
            "threshold": self.threshold,
        }


@dataclass
class QualitySnapshot:
    """Quality snapshot at a point in time."""

    timestamp: datetime
    source: str
    overall_score: float
    grade: str
    metrics: Dict[str, float]

    def to_dict(self) -> Dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "source": self.source,
            "overall_score": self.overall_score,
            "grade": self.grade,
            "metrics": self.metrics,
        }


class LiveQualityMonitor:
    """
    Live monitoring of data quality for trading data feeds.

    Continuously monitors data quality and triggers alerts
    when quality degrades below thresholds.
    """

    def __init__(
        self,
        check_interval: int = 60,  # seconds
        history_window: int = 100,  # number of snapshots to keep
        quality_threshold: float = 70.0,  # minimum acceptable quality
    ):
        """
        Initialize live quality monitor.

        Args:
            check_interval: How often to check quality (seconds)
            history_window: Number of quality snapshots to retain
            quality_threshold: Minimum acceptable quality score
        """
        self.check_interval = check_interval
        self.history_window = history_window
        self.quality_threshold = quality_threshold

        # Data sources
        self.sources: Dict[str, pd.DataFrame] = {}
        self.source_configs: Dict[str, Dict] = {}

        # Quality history
        self.quality_history: Dict[str, deque] = {}

        # Alerts
        self.alerts: deque = deque(maxlen=1000)
        self.alert_handlers: List[Callable] = []

        # Monitoring state
        self.is_monitoring = False
        self.monitor_thread = None

        # Current best source
        self.current_best_source: Optional[str] = None

        logger.info(
            f"LiveQualityMonitor initialized (check_interval={check_interval}s)"
        )

    def add_source(self, name: str, df: pd.DataFrame, config: Optional[Dict] = None):
        """Add a data source to monitor."""
        self.sources[name] = df
        self.source_configs[name] = config or {}
        self.quality_history[name] = deque(maxlen=self.history_window)

        # Initial quality check
        self._check_source_quality(name)

        logger.info(f"Added data source: {name}")

    def update_source(self, name: str, new_data: pd.DataFrame):
        """Update data for a source (e.g., new tick)."""
        if name not in self.sources:
            logger.warning(f"Source {name} not found, adding new")
            self.add_source(name, new_data)
            return

        # Append new data
        self.sources[name] = pd.concat(
            [self.sources[name], new_data], ignore_index=True
        )

        # Keep only recent data (last 10000 rows)
        if len(self.sources[name]) > 10000:
            self.sources[name] = self.sources[name].tail(10000)

        # Check quality
        self._check_source_quality(name)

    def _check_source_quality(self, source_name: str):
        """Check quality for a specific source."""
        from data_quality.assessor import DataQualityAssessor

        df = self.sources[source_name]

        try:
            assessor = DataQualityAssessor(df, source_name)
            metrics = assessor.assess()

            # Create snapshot
            snapshot = QualitySnapshot(
                timestamp=datetime.now(),
                source=source_name,
                overall_score=metrics.overall_score,
                grade=metrics.quality_grade,
                metrics={
                    "completeness": metrics.completeness_score,
                    "consistency": metrics.consistency_score,
                    "accuracy": metrics.accuracy_score,
                    "freshness": metrics.freshness_score,
                },
            )

            # Store in history
            self.quality_history[source_name].append(snapshot)

            # Check for alerts
            self._evaluate_alerts(source_name, metrics)

            # Update best source
            self._update_best_source()

        except Exception as e:
            logger.error(f"Error checking quality for {source_name}: {e}")

    def _evaluate_alerts(self, source_name: str, metrics):
        """Evaluate and trigger alerts."""
        alerts_triggered = []

        # Quality degradation
        if metrics.overall_score < self.quality_threshold:
            alerts_triggered.append(
                QualityAlert(
                    alert_type=DataQualityAlert.QUALITY_DEGRADATION,
                    severity="high" if metrics.overall_score < 50 else "medium",
                    message=f"Quality score {metrics.overall_score:.1f} below threshold {self.quality_threshold}",
                    timestamp=datetime.now(),
                    source=source_name,
                    metric_value=metrics.overall_score,
                    threshold=self.quality_threshold,
                )
            )

        # Missing data spike
        if metrics.missing_values_pct > 5.0:
            alerts_triggered.append(
                QualityAlert(
                    alert_type=DataQualityAlert.MISSING_DATA_SPIKE,
                    severity="critical" if metrics.missing_values_pct > 20 else "high",
                    message=f"Missing data spike: {metrics.missing_values_pct:.2f}%",
                    timestamp=datetime.now(),
                    source=source_name,
                    metric_value=metrics.missing_values_pct,
                    threshold=5.0,
                )
            )

        # Outliers
        if metrics.outlier_pct > 5.0:
            alerts_triggered.append(
                QualityAlert(
                    alert_type=DataQualityAlert.OUTLIER_DETECTED,
                    severity="high" if metrics.outlier_pct > 10 else "medium",
                    message=f"High outlier percentage: {metrics.outlier_pct:.2f}%",
                    timestamp=datetime.now(),
                    source=source_name,
                    metric_value=metrics.outlier_pct,
                    threshold=5.0,
                )
            )

        # Stale data
        if metrics.freshness_score < 50:
            alerts_triggered.append(
                QualityAlert(
                    alert_type=DataQualityAlert.STALE_DATA,
                    severity="high",
                    message=f"Data is {metrics.data_age_hours:.1f} hours old",
                    timestamp=datetime.now(),
                    source=source_name,
                    metric_value=metrics.data_age_hours,
                    threshold=24.0,
                )
            )

        # Store and trigger alerts
        for alert in alerts_triggered:
            self.alerts.append(alert)
            self._trigger_alert_handlers(alert)
            logger.warning(f"Quality Alert: {alert.message}")

    def _trigger_alert_handlers(self, alert: QualityAlert):
        """Trigger all registered alert handlers."""
        for handler in self.alert_handlers:
            try:
                handler(alert)
            except Exception as e:
                logger.error(f"Error in alert handler: {e}")

    def _update_best_source(self):
        """Update the current best source based on quality."""
        if not self.quality_history:
            return

        best_score = -1
        best_source = None

        for source_name, history in self.quality_history.items():
            if history:
                latest = history[-1]
                if latest.overall_score > best_score:
                    best_score = latest.overall_score
                    best_source = source_name

        if best_source != self.current_best_source:
            old_source = self.current_best_source
            self.current_best_source = best_source

            if old_source is not None:
                logger.info(
                    f"Best source changed: {old_source} -> {best_source} "
                    f"(score: {best_score:.1f})"
                )
            else:
                logger.info(
                    f"Initial best source: {best_source} (score: {best_score:.1f})"
                )

    def start_monitoring(self):
        """Start continuous quality monitoring."""
        if self.is_monitoring:
            logger.warning("Monitoring already started")
            return

        self.is_monitoring = True
        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop, daemon=True
        )
        self.monitor_thread.start()

        logger.info("Live quality monitoring started")

    def stop_monitoring(self):
        """Stop continuous quality monitoring."""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)

        logger.info("Live quality monitoring stopped")

    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.is_monitoring:
            try:
                # Check all sources
                for source_name in self.sources.keys():
                    self._check_source_quality(source_name)

                # Wait for next check
                time.sleep(self.check_interval)

            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(5)  # Wait before retry

    def register_alert_handler(self, handler: Callable):
        """Register a handler for quality alerts."""
        self.alert_handlers.append(handler)
        logger.info(f"Registered alert handler: {handler.__name__}")

    def get_current_quality(
        self, source_name: Optional[str] = None
    ) -> Optional[QualitySnapshot]:
        """Get current quality snapshot."""
        if source_name is None:
            source_name = self.current_best_source

        if source_name and source_name in self.quality_history:
            history = self.quality_history[source_name]
            if history:
                return history[-1]

        return None

    def get_quality_trend(self, source_name: str, window: int = 10) -> Dict:
        """Get quality trend for a source."""
        if source_name not in self.quality_history:
            return {"error": "Source not found"}

        history = list(self.quality_history[source_name])[-window:]

        if not history:
            return {"error": "No quality history"}

        scores = [snap.overall_score for snap in history]

        return {
            "source": source_name,
            "window": len(scores),
            "current": scores[-1],
            "mean": np.mean(scores),
            "trend": "improving" if scores[-1] > scores[0] else "degrading",
            "trend_strength": abs(scores[-1] - scores[0]),
            "volatility": np.std(scores),
        }

    def get_alerts(
        self,
        severity: Optional[str] = None,
        source: Optional[str] = None,
        since: Optional[datetime] = None,
    ) -> List[QualityAlert]:
        """Get filtered alerts."""
        filtered = list(self.alerts)

        if severity:
            filtered = [a for a in filtered if a.severity == severity]

        if source:
            filtered = [a for a in filtered if a.source == source]

        if since:
            filtered = [a for a in filtered if a.timestamp >= since]

        return filtered

    def get_best_source(self) -> Optional[str]:
        """Get current best data source."""
        return self.current_best_source

    def should_switch_source(self) -> Tuple[bool, Optional[str]]:
        """
        Determine if we should switch to a different source.

        Returns:
            Tuple of (should_switch, new_source_name)
        """
        if not self.current_best_source:
            return False, None

        current_quality = self.get_current_quality(self.current_best_source)

        if not current_quality:
            return False, None

        # If current source quality is below threshold, find alternative
        if current_quality.overall_score < self.quality_threshold:
            best_alternative = None
            best_score = 0

            for source_name, history in self.quality_history.items():
                if source_name != self.current_best_source and history:
                    score = history[-1].overall_score
                    if score > best_score and score >= self.quality_threshold:
                        best_score = score
                        best_alternative = source_name

            if best_alternative:
                return True, best_alternative

        return False, None

    def print_status(self):
        """Print current monitoring status."""
        print(f"\n{'=' * 70}")
        print(f"  LIVE DATA QUALITY MONITOR STATUS")
        print(f"{'=' * 70}")
        print(f"  Monitoring: {'Active' if self.is_monitoring else 'Stopped'}")
        print(f"  Check Interval: {self.check_interval}s")
        print(f"  Quality Threshold: {self.quality_threshold}")
        print(f"  Sources Monitored: {len(self.sources)}")
        print(f"  Current Best Source: {self.current_best_source or 'None'}")

        print(f"\n  SOURCE QUALITIES:")
        for source_name, history in self.quality_history.items():
            if history:
                latest = history[-1]
                status = "✅" if latest.overall_score >= self.quality_threshold else "⚠️"
                print(
                    f"     {status} {source_name}: {latest.overall_score:.1f}/100 (Grade {latest.grade})"
                )

        recent_alerts = [
            a for a in self.alerts if a.timestamp > datetime.now() - timedelta(hours=1)
        ]
        print(f"\n  RECENT ALERTS (last hour): {len(recent_alerts)}")

        if recent_alerts:
            for alert in recent_alerts[-5:]:  # Show last 5
                print(
                    f"     [{alert.severity.upper()}] {alert.alert_type.value}: {alert.message}"
                )

        print(f"\n{'=' * 70}\n")


class DynamicSourceSelector:
    """
    Dynamically select the best data source based on quality.

    Automatically switches between sources when quality degrades.
    """

    def __init__(
        self,
        monitor: LiveQualityMonitor,
        switch_threshold: float = 60.0,
        cooldown_period: int = 300,  # 5 minutes
    ):
        """
        Initialize dynamic source selector.

        Args:
            monitor: LiveQualityMonitor instance
            switch_threshold: Quality threshold for switching
            cooldown_period: Minimum time between switches (seconds)
        """
        self.monitor = monitor
        self.switch_threshold = switch_threshold
        self.cooldown_period = cooldown_period

        self.current_source: Optional[str] = None
        self.last_switch_time: Optional[datetime] = None
        self.switch_count = 0

        # Register alert handler
        self.monitor.register_alert_handler(self._on_quality_alert)

    def select_source(self) -> Optional[str]:
        """Select best source based on current quality."""
        # Check cooldown
        if self.last_switch_time:
            elapsed = (datetime.now() - self.last_switch_time).total_seconds()
            if elapsed < self.cooldown_period:
                logger.debug(f"In cooldown period ({elapsed:.0f}s remaining)")
                return self.current_source

        # Get recommendation from monitor
        should_switch, new_source = self.monitor.should_switch_source()

        if should_switch and new_source:
            self._switch_source(new_source)
        elif not self.current_source:
            # Initial selection
            best = self.monitor.get_best_source()
            if best:
                self._switch_source(best)

        return self.current_source

    def _switch_source(self, new_source: str):
        """Switch to a new data source."""
        old_source = self.current_source
        self.current_source = new_source
        self.last_switch_time = datetime.now()
        self.switch_count += 1

        logger.info(
            f"DynamicSourceSelector: Switched from {old_source} to {new_source} "
            f"(switch #{self.switch_count})"
        )

    def _on_quality_alert(self, alert: QualityAlert):
        """Handle quality alerts."""
        if alert.severity in ["high", "critical"]:
            # Trigger immediate re-evaluation
            new_source = self.select_source()
            if new_source != self.current_source:
                logger.warning(
                    f"Emergency source switch due to alert: {alert.alert_type.value}"
                )

    def get_data(self) -> Optional[pd.DataFrame]:
        """Get data from current best source."""
        source = self.select_source()

        if source and source in self.monitor.sources:
            return self.monitor.sources[source]

        return None

    def get_stats(self) -> Dict:
        """Get selector statistics."""
        return {
            "current_source": self.current_source,
            "total_switches": self.switch_count,
            "last_switch": self.last_switch_time.isoformat()
            if self.last_switch_time
            else None,
            "switch_threshold": self.switch_threshold,
            "cooldown_period": self.cooldown_period,
            "uptime": (datetime.now() - self.last_switch_time).total_seconds()
            if self.last_switch_time
            else 0,
        }


# Example usage and testing
if __name__ == "__main__":
    # Create sample data
    dates = pd.date_range(start="2024-01-01", periods=1000, freq="1h")

    # Source 1: High quality
    df1 = pd.DataFrame(
        {
            "timestamp": dates,
            "open": np.random.randn(1000).cumsum() + 50000,
            "high": np.random.randn(1000).cumsum() + 50100,
            "low": np.random.randn(1000).cumsum() + 49900,
            "close": np.random.randn(1000).cumsum() + 50000,
            "volume": np.random.randint(1000, 10000, 1000),
        }
    )

    # Source 2: Lower quality (with gaps and outliers)
    df2 = df1.copy()
    # Add some missing values
    df2.loc[100:110, "close"] = np.nan
    # Add outliers
    df2.loc[200, "close"] = df2.loc[200, "close"] * 1.5

    # Create monitor
    monitor = LiveQualityMonitor(check_interval=10)

    # Add sources
    monitor.add_source("Binance_Primary", df1)
    monitor.add_source("Binance_Backup", df2)

    # Print initial status
    monitor.print_status()

    # Start monitoring
    monitor.start_monitoring()

    # Run for a bit
    time.sleep(30)

    # Check status
    monitor.print_status()

    # Stop
    monitor.stop_monitoring()
