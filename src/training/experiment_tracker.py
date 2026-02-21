"""
Experiment Tracker
==================
Lightweight model registry that logs experiment metadata to a JSON file.

Designed for team use: multiple people can run experiments and compare results
without a database or external service.

Usage:
    tracker = ExperimentTracker()
    exp_id = tracker.start("ppo_v3", {"lr": 3e-4, "gamma": 0.99})
    tracker.log_metrics(exp_id, iteration=100, mean_return=12.5, sharpe=1.2)
    tracker.finish(exp_id, status="success", best_return=15.3)

Results are stored in logs/experiments/registry.json (human-readable).
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
    File-based experiment registry.

    Each experiment entry contains:
    - id:          unique UUID
    - name:        short human-readable name (e.g. "ppo_v3")
    - user:        hostname of machine that ran the experiment
    - started_at:  ISO-8601 UTC timestamp
    - finished_at: ISO-8601 UTC timestamp (set on finish())
    - status:      "running" | "success" | "failed" | "cancelled"
    - params:      dict of hyperparameters / config values
    - metrics:     list of {iteration, timestamp, **values} dicts
    - summary:     dict of final scalar results (set on finish())
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
                exp["metrics"].append({"iteration": iteration, "timestamp": self._now(), **values})
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
            if metric in e.get("summary", {}) and (name_filter is None or name_filter in e["name"])
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
            print(f"[{row['id']}] {row['name']:<30} status={status:<10} best_return={best}")
    elif sys.argv[1] == "best":
        metric = sys.argv[2] if len(sys.argv) > 2 else "best_return"
        best = tracker.best(metric)
        if best:
            print(json.dumps({k: v for k, v in best.items() if k != "metrics"}, indent=2))
        else:
            print(f"No experiments with metric '{metric}' found.")
