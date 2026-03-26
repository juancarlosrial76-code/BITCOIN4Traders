"""
RunLogger – Unified Experiment Logging
=======================================
Single interface that writes training metrics to all available backends:

  1. MLflow       – if mlflow is installed (``pip install mlflow``)
  2. TensorBoard  – if tensorboard is installed (``pip install tensorboard``)
  3. ExperimentTracker – always (file-based JSON registry, zero dependencies)

All three backends are optional.  If a backend is unavailable or raises an
error, RunLogger logs a warning and continues without it.  This guarantees
that training never fails because of a missing logging library.

Usage – AdversarialTrainer integration:
---------------------------------------
    from training.run_logger import RunLogger

    logger_run = RunLogger(
        run_name="ppo_curriculum_phase1",
        params={"phase": 1, "n_iter": 200, "allowed_actions": "[3,4,5,6]"},
        mlflow_uri="mlruns",          # or "http://localhost:5000"
        tb_log_dir="logs/tb",
    )

    # Training loop
    for iteration in range(n_iterations):
        metrics = trainer.train_step()
        logger_run.log(iteration, **metrics)

    # End of run
    logger_run.finish(status="success", best_return=0.05)

Usage – Curriculum integration:
--------------------------------
    logger_run = RunLogger.for_curriculum(phase=1, allowed_actions=[3,4,5,6])
    ...
    logger_run.finish(status="success", **result.to_dict())

Author: BITCOIN4Traders Team
Version: 1.0.0
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Optional

from loguru import logger as _loguru

# ── Optional backend imports ───────────────────────────────────────────────

try:
    import mlflow
    import mlflow.pytorch

    _HAS_MLFLOW = True
except ImportError:
    _HAS_MLFLOW = False

try:
    from torch.utils.tensorboard import SummaryWriter

    _HAS_TB = True
except ImportError:
    try:
        from tensorboard.summary.writer.event_file_writer import EventFileWriter  # noqa
        from torch.utils.tensorboard import SummaryWriter

        _HAS_TB = True
    except ImportError:
        _HAS_TB = False

# Always available (zero extra dependencies)
from src.training.experiment_tracker import ExperimentTracker


class RunLogger:
    """
    Unified logging facade for training runs.

    Backends activated automatically based on installed packages:
    - MLflow:          requires ``mlflow``
    - TensorBoard:     requires ``tensorboard`` + ``torch``
    - ExperimentTracker: always active (JSON file, no deps)

    Parameters
    ----------
    run_name : str
        Human-readable name for this run, e.g. "curriculum_phase1".
    params : dict
        Hyperparameters / config to record at the start of the run.
    mlflow_uri : str or None
        MLflow tracking URI.  "mlruns" → local file store.
        None → disable MLflow even if installed.
    mlflow_experiment : str
        MLflow experiment name (created if it doesn't exist).
    tb_log_dir : str
        Root directory for TensorBoard event files.
        Run-specific sub-directory is created automatically.
    registry_path : str or None
        Path for the ExperimentTracker JSON registry.
        None → default ("logs/experiments/registry.json").
    """

    def __init__(
        self,
        run_name: str,
        params: Optional[dict[str, Any]] = None,
        mlflow_uri: Optional[str] = "mlruns",
        mlflow_experiment: str = "BITCOIN4Traders",
        tb_log_dir: str = "logs/tensorboard",
        registry_path: Optional[str] = None,
    ) -> None:
        self.run_name = run_name
        self.params = params or {}
        self._start_time = time.time()

        # ── 1. ExperimentTracker (always) ─────────────────────────────────
        reg = Path(registry_path) if registry_path else None
        self._tracker = ExperimentTracker(reg) if reg else ExperimentTracker()
        self._exp_id = self._tracker.start(run_name, self.params)
        _loguru.info(f"[RunLogger] ExperimentTracker started: id={self._exp_id}")

        # ── 2. MLflow ─────────────────────────────────────────────────────
        self._mlflow_run = None
        if _HAS_MLFLOW and mlflow_uri is not None:
            try:
                mlflow.set_tracking_uri(mlflow_uri)
                mlflow.set_experiment(mlflow_experiment)
                self._mlflow_run = mlflow.start_run(run_name=run_name)
                if self.params:
                    mlflow.log_params(self._flatten(self.params))
                _loguru.info(
                    f"[RunLogger] MLflow active: experiment='{mlflow_experiment}'"
                    f"  run_id={self._mlflow_run.info.run_id}"
                )
            except Exception as exc:
                _loguru.warning(f"[RunLogger] MLflow init failed (ignored): {exc}")
                self._mlflow_run = None

        # ── 3. TensorBoard ────────────────────────────────────────────────
        self._tb: Optional[Any] = None
        if _HAS_TB:
            try:
                tb_dir = Path(tb_log_dir) / run_name
                tb_dir.mkdir(parents=True, exist_ok=True)
                self._tb = SummaryWriter(log_dir=str(tb_dir))
                _loguru.info(f"[RunLogger] TensorBoard active: {tb_dir}")
            except Exception as exc:
                _loguru.warning(f"[RunLogger] TensorBoard init failed (ignored): {exc}")
                self._tb = None

        _loguru.info(
            f"[RunLogger] backends: "
            f"ExperimentTracker=ON  "
            f"MLflow={'ON' if self._mlflow_run else 'OFF'}  "
            f"TensorBoard={'ON' if self._tb else 'OFF'}"
        )

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def log(self, step: int, **metrics: float) -> None:
        """
        Log a scalar metrics snapshot at the given training step.

        Parameters
        ----------
        step : int
            Current iteration / global step number.
        **metrics : float
            Any scalar values, e.g. mean_return=0.05, actor_loss=0.12.
        """
        # ExperimentTracker
        try:
            self._tracker.log_metrics(self._exp_id, iteration=step, **metrics)
        except Exception as exc:
            _loguru.debug(f"[RunLogger] ExperimentTracker log error: {exc}")

        # MLflow
        if self._mlflow_run is not None:
            try:
                mlflow.log_metrics(self._flatten(metrics), step=step)
            except Exception as exc:
                _loguru.debug(f"[RunLogger] MLflow log error: {exc}")

        # TensorBoard
        if self._tb is not None:
            try:
                for key, val in metrics.items():
                    self._tb.add_scalar(key, float(val), global_step=step)
            except Exception as exc:
                _loguru.debug(f"[RunLogger] TensorBoard log error: {exc}")

    def log_model(self, model: Any, artifact_path: str = "trader") -> None:
        """
        Log a PyTorch model artifact to MLflow (no-op if MLflow unavailable).

        Parameters
        ----------
        model : torch.nn.Module
            Model to log.
        artifact_path : str
            Sub-path inside the MLflow run artifacts directory.
        """
        if self._mlflow_run is not None and _HAS_MLFLOW:
            try:
                mlflow.pytorch.log_model(model, artifact_path)
                _loguru.info(
                    f"[RunLogger] MLflow model artifact logged: {artifact_path}"
                )
            except Exception as exc:
                _loguru.warning(f"[RunLogger] MLflow log_model failed: {exc}")

    def log_figure(self, tag: str, figure: Any, step: int = 0) -> None:
        """
        Log a matplotlib figure to TensorBoard (no-op if unavailable).

        Parameters
        ----------
        tag : str
            Identifier string shown in TensorBoard.
        figure : matplotlib.figure.Figure
            The figure to log.
        step : int
            Global step.
        """
        if self._tb is not None:
            try:
                self._tb.add_figure(tag, figure, global_step=step)
            except Exception as exc:
                _loguru.debug(f"[RunLogger] TensorBoard add_figure error: {exc}")

    def finish(self, status: str = "success", **summary: float) -> None:
        """
        Finalise the run and flush all backends.

        Parameters
        ----------
        status : str
            "success" | "failed" | "cancelled"
        **summary : float
            Final scalar results, e.g. best_return=0.05, decision_matrix_score=71.
        """
        elapsed = time.time() - self._start_time
        summary["elapsed_seconds"] = round(elapsed, 1)

        # ExperimentTracker
        try:
            self._tracker.finish(self._exp_id, status=status, **summary)
        except Exception as exc:
            _loguru.debug(f"[RunLogger] ExperimentTracker finish error: {exc}")

        # MLflow
        if self._mlflow_run is not None and _HAS_MLFLOW:
            try:
                mlflow.log_metrics(self._flatten(summary), step=0)
                mlflow.set_tag("status", status)
                mlflow.end_run(status="FINISHED" if status == "success" else "FAILED")
            except Exception as exc:
                _loguru.debug(f"[RunLogger] MLflow finish error: {exc}")

        # TensorBoard
        if self._tb is not None:
            try:
                self._tb.flush()
                self._tb.close()
            except Exception as exc:
                _loguru.debug(f"[RunLogger] TensorBoard close error: {exc}")

        _loguru.info(
            f"[RunLogger] Run '{self.run_name}' finished: "
            f"status={status}  elapsed={elapsed:.0f}s"
        )

    def __enter__(self) -> "RunLogger":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        status = "failed" if exc_type is not None else "success"
        self.finish(status=status)

    # ------------------------------------------------------------------ #
    # Convenience constructors                                             #
    # ------------------------------------------------------------------ #

    @classmethod
    def for_curriculum(
        cls,
        phase: int,
        allowed_actions: Optional[list],
        n_iterations: int = 0,
        device: str = "cpu",
        **kwargs,
    ) -> "RunLogger":
        """
        Build a RunLogger pre-configured for a curriculum training phase.

        Parameters
        ----------
        phase : int
            Curriculum phase number (1, 2, or 3).
        allowed_actions : list or None
            Action mask for this phase (None = all actions).
        n_iterations : int
            Planned iterations for this phase.
        device : str
            Training device ("cpu" / "cuda").
        **kwargs
            Passed through to RunLogger.__init__().

        Returns
        -------
        RunLogger instance with phase-specific run name and params.
        """
        phase_labels = {1: "long_only", 2: "short_only", 3: "full_space"}
        label = phase_labels.get(phase, f"phase{phase}")
        run_name = f"curriculum_phase{phase}_{label}"

        params = {
            "curriculum_phase": phase,
            "phase_label": label,
            "allowed_actions": str(allowed_actions),
            "n_iterations": n_iterations,
            "device": device,
        }
        return cls(run_name=run_name, params=params, **kwargs)

    @classmethod
    def for_adversarial(
        cls,
        n_iterations: int,
        steps_per_iteration: int,
        device: str = "cpu",
        **kwargs,
    ) -> "RunLogger":
        """
        Build a RunLogger pre-configured for standard adversarial training.

        Parameters
        ----------
        n_iterations : int
            Total training iterations.
        steps_per_iteration : int
            Environment steps per iteration.
        device : str
            Training device.
        **kwargs
            Passed through to RunLogger.__init__().
        """
        params = {
            "n_iterations": n_iterations,
            "steps_per_iteration": steps_per_iteration,
            "device": device,
        }
        return cls(
            run_name=f"adversarial_{n_iterations}iter",
            params=params,
            **kwargs,
        )

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _flatten(d: dict, prefix: str = "") -> dict:
        """
        Flatten a nested dict to dot-notation keys for MLflow compatibility.

        Example:
            {"risk": {"sharpe": 1.2}} → {"risk.sharpe": 1.2}
        """
        out: dict = {}
        for k, v in d.items():
            key = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                out.update(RunLogger._flatten(v, key))
            else:
                # MLflow only accepts numeric or string values
                try:
                    out[key] = float(v)
                except (TypeError, ValueError):
                    out[key] = str(v)
        return out

    @staticmethod
    def backends_available() -> dict[str, bool]:
        """Return a dict showing which logging backends are available."""
        return {
            "mlflow": _HAS_MLFLOW,
            "tensorboard": _HAS_TB,
            "experiment_tracker": True,
        }
