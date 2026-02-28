"""
Efficient Training Metrics Logger
=================================
Speichert nur wichtige Metriken in CSV statt riesiger Text-Logs.

Usage:
    from src.training.metrics_logger import MetricsLogger
    logger = MetricsLogger('data/training_metrics.csv')
    logger.log(iteration=10, mean_reward=123.45, action_dist=[10,20,30,15,10,5,10])
"""

import csv
import json
import gzip
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import os


class MetricsLogger:
    """Lightweight metrics logger - stores only essential data."""

    def __init__(self, filepath: str, compress: bool = True):
        self.filepath = Path(filepath)
        self.compress = compress
        self.filepath.parent.mkdir(parents=True, exist_ok=True)

        # CSV Header
        self.fields = [
            "timestamp",
            "iteration",
            "mean_reward",
            "std_reward",
            "episode_count",
            "action_0",
            "action_1",
            "action_2",
            "action_3",
            "action_4",
            "action_5",
            "action_6",
            "policy_loss",
            "value_loss",
            "entropy",
        ]

        # Create file if not exists
        if not self.filepath.exists():
            with open(self.filepath, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self.fields)
                writer.writeheader()

    def log(self, iteration: int, mean_reward: float, action_dist: List[int], **kwargs):
        """Log training metrics."""
        row = {
            "timestamp": datetime.now().isoformat(),
            "iteration": iteration,
            "mean_reward": f"{mean_reward:.4f}",
            "std_reward": f"{kwargs.get('std_reward', 0):.4f}",
            "episode_count": kwargs.get("episode_count", 0),
            "action_0": action_dist[0] if len(action_dist) > 0 else 0,
            "action_1": action_dist[1] if len(action_dist) > 1 else 0,
            "action_2": action_dist[2] if len(action_dist) > 2 else 0,
            "action_3": action_dist[3] if len(action_dist) > 3 else 0,
            "action_4": action_dist[4] if len(action_dist) > 4 else 0,
            "action_5": action_dist[5] if len(action_dist) > 5 else 0,
            "action_6": action_dist[6] if len(action_dist) > 6 else 0,
            "policy_loss": f"{kwargs.get('policy_loss', 0):.6f}",
            "value_loss": f"{kwargs.get('value_loss', 0):.6f}",
            "entropy": f"{kwargs.get('entropy', 0):.6f}",
        }

        with open(self.filepath, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fields)
            writer.writerow(row)

    def compress_old_logs(self, max_size_mb: int = 100):
        """Compress old logs if file exceeds max size."""
        if not self.filepath.exists():
            return

        size_mb = self.filepath.stat().st_size / (1024 * 1024)
        if size_mb > max_size_mb:
            # Compress to gzip
            compressed = str(self.filepath) + ".gz"
            with open(self.filepath, "rb") as f_in:
                with gzip.open(compressed, "wb") as f_out:
                    f_out.writelines(f_in)

            # Start new file
            self.filepath.unlink()
            with open(self.filepath, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self.fields)
                writer.writeheader()


class CompactJSONLogger:
    """Logs events as compact JSON lines - much smaller than text logs."""

    def __init__(self, filepath: str):
        self.filepath = Path(filepath)
        self.filepath.parent.mkdir(parents=True, exist_ok=True)

    def log_event(self, event_type: str, data: Dict):
        """Log single event as JSON line."""
        line = json.dumps({"ts": datetime.now().isoformat(), "type": event_type, "data": data})
        with open(self.filepath, "a") as f:
            f.write(line + "\n")


if __name__ == "__main__":
    # Test
    logger = MetricsLogger("data/test_metrics.csv")
    logger.log(
        iteration=10, mean_reward=123.45, action_dist=[15, 25, 10, 20, 15, 10, 5], entropy=0.5
    )
    print("Metrics logger test OK")
