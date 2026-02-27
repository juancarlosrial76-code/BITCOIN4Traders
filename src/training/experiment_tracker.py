"""
Experiment Tracker
=================
Lightweight experiment registry that logs metadata to a JSON file.

This module provides a simple, file-based experiment tracking system designed
for teams running reinforcement learning experiments. It requires no external
database or service - all data is stored in a human-readable JSON file.

Key Features:
- Simple API: start(), log_metrics(), finish() workflow
- Human-readable: Registry stored as formatted JSON
- Team-friendly: Multiple users can run experiments simultaneously
- Queryable: Filter by name, find best experiments by metric
- Lightweight: No external dependencies beyond Python standard library

Data Structure:
--------------
Each experiment entry contains:
    - id:          Unique 8-character UUID
    - name:        Short descriptive name (e.g., "ppo_v3")
    - user:        Hostname of machine that ran the experiment
    - started_at:  ISO-8601 UTC timestamp
    - finished_at: ISO-8601 timestamp (set on finish())
    - status:      "running" | "success" | "failed" | "cancelled"
    - params:      Dict of hyperparameters/configuration
    - metrics:     List of {iteration, timestamp, **values} dicts
    - summary:     Final results dict (set on finish())

File Location:
-------------
Default: logs/experiments/registry.json
Configure via registry_path parameter.

Usage:
------
    # Start tracking
    tracker = ExperimentTracker()
    exp_id = tracker.start("ppo_baseline", {"lr": 3e-4, "gamma": 0.99})

    # Log training progress
    for iteration in range(500):
        mean_return = train_step()
        tracker.log_metrics(exp_id, iteration=iteration, mean_return=mean_return)

    # Mark complete
    tracker.finish(exp_id, status="success", best_return=15.3)

Querying:
--------
    # List all experiments
    tracker.list_experiments()

    # Filter by name
    tracker.list_experiments(name_filter="ppo")

    # Find best by metric
    tracker.best(metric="best_return", name_filter="ppo")

    # Get full details
    tracker.get(exp_id)

CLI Usage:
---------
    python -m src.training.experiment_tracker           # List all
    python -m src.training.experiment_tracker best     # Show best
    python -m src.training.experiment_tracker best sharpe  # Best by sharpe

Why Not TensorBoard/W&B?
------------------------
- No account required
- Works offline
- No data leaves your machine
- Simple grep/analysis with standard tools
- Perfect for quick local experiments
"""

import json
import uuid
import socket
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

DEFAULT_REGISTRY_PATH = Path("logs/experiments/registry.json")


class ExperimentTracker:
    """
    File-based Experiment Registry.

    A lightweight experiment tracking system that stores all metadata in a
    JSON file. Designed for teams to run and compare experiments without
    external services.

    The tracker provides a simple workflow:
    1. start(): Create a new experiment entry
    2. log_metrics(): Record progress during training
    3. finish(): Mark experiment complete with final results

    Data is automatically persisted to disk after each operation,
    ensuring no data is lost if training is interrupted.

    Attributes:
        registry_path (Path): Location of the JSON registry file.

    Args:
        registry_path (Path): Custom path for the registry file.
            Default: logs/experiments/registry.json

    Thread Safety:
    --------------
    This class is NOT thread-safe. If multiple processes write to the
    same registry file simultaneously, data may be corrupted. Use separate
    registry files per process or implement file locking for concurrent access.

    Example:
        Basic training loop with tracking:
        >>> tracker = ExperimentTracker()
        >>>
        >>> # Start experiment
        >>> exp_id = tracker.start(
        ...     name="ppo_v3_baseline",
        ...     params={"lr": 3e-4, "gamma": 0.99, "clip_eps": 0.2}
        ... )
        >>>
        >>> # Training loop
        >>> for episode in range(500):
        ...     metrics = train_agent()  # Your training logic
        ...     tracker.log_metrics(
        ...         exp_id,
        ...         episode=episode,
        ...         mean_return=metrics["return"],
        ...         sharpe=metrics["sharpe"],
        ...         loss=metrics["loss"]
        ...     )
        >>>
        >>> # Mark complete
        >>> tracker.finish(
        ...     exp_id,
        ...     status="success",
        ...     best_return=metrics["best_return"],
        ...     final_sharpe=metrics["sharpe"]
        ... )

    Querying experiments:
        >>> # Find all ppo experiments
        >>> ppo_exps = tracker.list_experiments(name_filter="ppo")
        >>>
        >>> # Get best by return
        >>> best = tracker.best(metric="best_return")
        >>> print(f"Best: {best['name']} = {best['summary']['best_return']}")
    """

    def __init__(self, registry_path: Path = DEFAULT_REGISTRY_PATH) -> None:
        self.registry_path = registry_path
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load(self) -> list[dict]:
        """Load all experiments from the registry file."""
        if not self.registry_path.exists():
            return []
        with open(self.registry_path) as f:
            return json.load(f)

    def _save(self, experiments: list[dict]) -> None:
        """Persist experiments to disk (pretty-printed for readability)."""
        with open(self.registry_path, "w") as f:
            json.dump(experiments, f, indent=2, default=str)

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self, name: str, params: Optional[dict[str, Any]] = None) -> str:
        """
        Register a new experiment and return its unique ID.

        Parameters
        ----------
        name:   short descriptive name, e.g. "ppo_baseline" or "adversarial_v2"
        params: dict of hyperparameters / config values to record
        """
        exp_id = str(uuid.uuid4())[:8]  # short 8-char ID for readability
        entry = {
            "id": exp_id,
            "name": name,
            "user": socket.gethostname(),
            "started_at": self._now(),
            "finished_at": None,
            "status": "running",
            "params": params or {},
            "metrics": [],
            "summary": {},
        }
        experiments = self._load()
        experiments.append(entry)
        self._save(experiments)
        return exp_id

    def log_metrics(self, exp_id: str, iteration: int, **values: float) -> None:
        """
        Append a metrics snapshot to an experiment.

        Parameters
        ----------
        exp_id:    ID returned by start()
        iteration: training iteration / step number
        **values:  any scalar metrics, e.g. mean_return=12.5, sharpe=1.2
        """
        experiments = self._load()
        for exp in experiments:
            if exp["id"] == exp_id:
                exp["metrics"].append(
                    {"iteration": iteration, "timestamp": self._now(), **values}
                )
                break
        self._save(experiments)

    def finish(
        self,
        exp_id: str,
        status: str = "success",
        **summary: float,
    ) -> None:
        """
        Mark an experiment as finished and record final summary metrics.

        Parameters
        ----------
        exp_id:   ID returned by start()
        status:   "success" | "failed" | "cancelled"
        **summary: final scalar results, e.g. best_return=15.3, episodes=500
        """
        experiments = self._load()
        for exp in experiments:
            if exp["id"] == exp_id:
                exp["finished_at"] = self._now()
                exp["status"] = status
                exp["summary"] = summary
                break
        self._save(experiments)

    def list_experiments(self, name_filter: Optional[str] = None) -> list[dict]:
        """
        Return all experiments, optionally filtered by name substring.

        Returns a list of experiment dicts (without the full metrics history
        to keep output compact).
        """
        experiments = self._load()
        if name_filter:
            experiments = [e for e in experiments if name_filter in e["name"]]
        # Strip per-iteration metrics from the listing view
        return [{k: v for k, v in e.items() if k != "metrics"} for e in experiments]

    def get(self, exp_id: str) -> Optional[dict]:
        """Return the full experiment record (including metrics) for a given ID."""
        for exp in self._load():
            if exp["id"] == exp_id:
                return exp
        return None

    def best(
        self, metric: str = "best_return", name_filter: Optional[str] = None
    ) -> Optional[dict]:
        """
        Return the experiment with the highest value of `metric` in its summary.

        Parameters
        ----------
        metric:      key to compare in experiment["summary"]
        name_filter: optional name substring to restrict the search
        """
        candidates = [
            e
            for e in self._load()
            if metric in e.get("summary", {})
            and (name_filter is None or name_filter in e["name"])
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda e: e["summary"][metric])


# ---------------------------------------------------------------------------
# Simple CLI for quick inspection (python -m src.training.experiment_tracker)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    tracker = ExperimentTracker()

    if len(sys.argv) == 1:
        # List all experiments
        rows = tracker.list_experiments()
        if not rows:
            print("No experiments recorded yet.")
        for row in rows:
            status = row.get("status", "?")
            summary = row.get("summary", {})
            best = summary.get("best_return", "-")
            print(
                f"[{row['id']}] {row['name']:<30} status={status:<10} best_return={best}"
            )
    elif sys.argv[1] == "best":
        metric = sys.argv[2] if len(sys.argv) > 2 else "best_return"
        best = tracker.best(metric)
        if best:
            print(
                json.dumps({k: v for k, v in best.items() if k != "metrics"}, indent=2)
            )
        else:
            print(f"No experiments with metric '{metric}' found.")
