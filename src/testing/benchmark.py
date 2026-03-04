"""
Model Benchmark Framework
=========================
Evaluates trained RL models on standardized metrics to support
objective model selection after each training run.

Decision matrix score:
    score = 0.4 * profit_score
          + 0.3 * balance_score
          + 0.2 * (100 - drawdown_score)
          + 0.1 * (100 / speed_score)

Metrics:
---------
- profit_score   : mean daily PnL relative to initial equity (%)
- balance_score  : action balance (deviation from uniform distribution)
- drawdown_score : maximum drawdown (%)
- speed_score    : training steps per second

Usage:
------
    from src.testing.benchmark import ModelBenchmark

    bench = ModelBenchmark(env, model)
    result = bench.run(n_episodes=5)
    print(result.summary())

Author: BITCOIN4Traders Team
Version: 1.0.0
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger("benchmark")


@dataclass
class BenchmarkResult:
    """
    Stores all metrics collected during a benchmark run.

    Attributes:
        n_episodes     : number of episodes evaluated
        episode_returns: total return per episode (fraction of initial equity)
        episode_lengths: number of steps per episode
        action_counts  : how many times each action was chosen (across all episodes)
        max_drawdowns  : maximum drawdown per episode (fraction, 0-1)
        steps_per_sec  : throughput measured during the run
        extra          : arbitrary extra metrics (e.g. win_rate, sharpe)
    """

    n_episodes: int = 0
    episode_returns: List[float] = field(default_factory=list)
    episode_lengths: List[int] = field(default_factory=list)
    action_counts: Dict[int, int] = field(default_factory=dict)
    max_drawdowns: List[float] = field(default_factory=list)
    steps_per_sec: float = 0.0
    extra: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------ #
    # Derived metrics                                                       #
    # ------------------------------------------------------------------ #

    @property
    def mean_return(self) -> float:
        """Mean episode return (fraction of initial equity)."""
        return float(np.mean(self.episode_returns)) if self.episode_returns else 0.0

    @property
    def std_return(self) -> float:
        """Standard deviation of episode returns."""
        return float(np.std(self.episode_returns)) if self.episode_returns else 0.0

    @property
    def mean_length(self) -> float:
        """Mean episode length in steps."""
        return float(np.mean(self.episode_lengths)) if self.episode_lengths else 0.0

    @property
    def mean_max_drawdown(self) -> float:
        """Mean maximum drawdown across episodes (fraction 0-1)."""
        return float(np.mean(self.max_drawdowns)) if self.max_drawdowns else 0.0

    @property
    def action_balance_score(self) -> float:
        """
        Action balance score (0-100).

        100 = perfectly uniform distribution across all actions.
        0   = all steps use a single action (fully degenerate policy).

        Uses normalised entropy: H / H_max  mapped to 0-100.
        """
        if not self.action_counts:
            return 0.0
        counts = np.array(list(self.action_counts.values()), dtype=float)
        total = counts.sum()
        if total == 0:
            return 0.0
        probs = counts / total
        # Remove zeros to avoid log(0)
        probs = probs[probs > 0]
        entropy = -float(np.sum(probs * np.log(probs)))
        max_entropy = float(np.log(len(self.action_counts)))
        if max_entropy == 0:
            return 100.0
        return round(100.0 * entropy / max_entropy, 2)

    @property
    def profit_score(self) -> float:
        """
        Profit score (0-100).

        Maps mean_return to a 0-100 scale.
        +5% return → 100,  0% → 50,  -5% → 0  (linearly clamped).
        """
        # ±5 % range mapped to 0-100
        score = 50.0 + self.mean_return * 1000.0  # 1% → +10 points
        return float(np.clip(score, 0.0, 100.0))

    @property
    def drawdown_score(self) -> float:
        """
        Drawdown score (0-100).  Lower is better (used as cost in matrix).

        0% drawdown → 0,  50%+ drawdown → 100.
        """
        return float(np.clip(self.mean_max_drawdown * 200.0, 0.0, 100.0))

    @property
    def decision_matrix_score(self) -> float:
        """
        Composite decision-matrix score (0-100).  Higher is better.

        score = 0.4 * profit_score
              + 0.3 * balance_score
              + 0.2 * (100 - drawdown_score)
              + 0.1 * speed_bonus

        speed_bonus: 100 / max(steps_per_sec, 1), clamped 0-100.
        """
        speed_bonus = float(np.clip(100.0 / max(self.steps_per_sec, 1.0), 0.0, 100.0))
        score = (
            0.4 * self.profit_score
            + 0.3 * self.action_balance_score
            + 0.2 * (100.0 - self.drawdown_score)
            + 0.1 * speed_bonus
        )
        return round(score, 2)

    # ------------------------------------------------------------------ #
    # Reporting                                                            #
    # ------------------------------------------------------------------ #

    def summary(self) -> str:
        """Return a human-readable summary string."""
        lines = [
            "=" * 55,
            "  MODEL BENCHMARK RESULT",
            "=" * 55,
            f"  Episodes evaluated  : {self.n_episodes}",
            f"  Mean episode length : {self.mean_length:.0f} steps",
            "",
            "  --- Returns ---",
            f"  Mean return         : {self.mean_return * 100:+.2f} %",
            f"  Std  return         : {self.std_return * 100:.2f} %",
            f"  Profit score        : {self.profit_score:.1f} / 100",
            "",
            "  --- Risk ---",
            f"  Mean max drawdown   : {self.mean_max_drawdown * 100:.2f} %",
            f"  Drawdown score      : {self.drawdown_score:.1f} / 100 (lower=better)",
            "",
            "  --- Policy health ---",
            f"  Action balance score: {self.action_balance_score:.1f} / 100",
            f"  Action distribution : {self._action_dist_str()}",
            "",
            "  --- Speed ---",
            f"  Steps / second      : {self.steps_per_sec:.1f}",
            "",
            f"  DECISION MATRIX SCORE: {self.decision_matrix_score:.1f} / 100",
            "=" * 55,
        ]
        return "\n".join(lines)

    def _action_dist_str(self) -> str:
        """Format action counts as a compact percentage string."""
        if not self.action_counts:
            return "n/a"
        total = sum(self.action_counts.values())
        if total == 0:
            return "n/a"
        parts = []
        for a in sorted(self.action_counts.keys()):
            pct = 100.0 * self.action_counts[a] / total
            parts.append(f"a{a}:{pct:.0f}%")
        return "  ".join(parts)

    def to_dict(self) -> dict:
        """Serialise key metrics to a flat dictionary (e.g. for MLflow logging)."""
        return {
            "n_episodes": self.n_episodes,
            "mean_return_pct": round(self.mean_return * 100, 4),
            "std_return_pct": round(self.std_return * 100, 4),
            "mean_max_drawdown_pct": round(self.mean_max_drawdown * 100, 4),
            "action_balance_score": self.action_balance_score,
            "profit_score": self.profit_score,
            "drawdown_score": self.drawdown_score,
            "steps_per_sec": round(self.steps_per_sec, 2),
            "decision_matrix_score": self.decision_matrix_score,
            **self.extra,
        }


class ModelBenchmark:
    """
    Run a trained model through one or more evaluation episodes and
    collect standardised metrics.

    The model must expose a `predict(obs)` method that returns an
    integer action (compatible with Stable-Baselines3 / custom agents).
    The environment must follow the standard Gymnasium API.

    Parameters:
    -----------
    env   : gym.Env-compatible environment instance.
    model : object with `.predict(obs) -> (action, state)` or
            `.predict(obs) -> action` interface.
    n_actions : int
        Total number of discrete actions in the action space.
        Used to initialise action count buckets.  Default: 7.
    deterministic : bool
        If True, pass `deterministic=True` to model.predict() when
        the model supports it (SB3-style).  Default: True.

    Example:
    --------
        bench = ModelBenchmark(env, ppo_agent, n_actions=7)
        result = bench.run(n_episodes=5)
        logger.info(result.summary())
    """

    # Human-readable action labels (matches config_integrated_env.py POSITION_SIZES)
    ACTION_LABELS = {
        0: "Short100%",
        1: "Short50%",
        2: "Neutral",
        3: "Long33%",
        4: "Long50%",
        5: "Long75%",
        6: "Long100%",
    }

    def __init__(
        self,
        env: Any,
        model: Any,
        n_actions: int = 7,
        deterministic: bool = True,
    ) -> None:
        self.env = env
        self.model = model
        self.n_actions = n_actions
        self.deterministic = deterministic
        # Per-episode recurrent hidden state (set in _run_episode)
        self._hidden: Any = None

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def run(self, n_episodes: int = 5, seed: Optional[int] = None) -> BenchmarkResult:
        """
        Execute n_episodes evaluation episodes and return metrics.

        Parameters:
        -----------
        n_episodes : int
            Number of full episodes to run (default: 5).
        seed : int or None
            Optional RNG seed for reproducibility.

        Returns:
        --------
        BenchmarkResult with all collected metrics.
        """
        result = BenchmarkResult(n_episodes=n_episodes)
        result.action_counts = {a: 0 for a in range(self.n_actions)}

        total_steps = 0
        t0 = time.perf_counter()

        for ep in range(n_episodes):
            ep_result = self._run_episode(seed=seed)

            result.episode_returns.append(ep_result["return"])
            result.episode_lengths.append(ep_result["length"])
            result.max_drawdowns.append(ep_result["max_drawdown"])
            total_steps += ep_result["length"]

            for a, cnt in ep_result["action_counts"].items():
                result.action_counts[a] = result.action_counts.get(a, 0) + cnt

            logger.debug(
                "Episode %d/%d: return=%.3f%%  length=%d  dd=%.2f%%",
                ep + 1,
                n_episodes,
                ep_result["return"] * 100,
                ep_result["length"],
                ep_result["max_drawdown"] * 100,
            )

        elapsed = time.perf_counter() - t0
        result.steps_per_sec = total_steps / max(elapsed, 1e-6)

        logger.info("Benchmark finished: %d steps in %.1fs", total_steps, elapsed)
        return result

    def compare(
        self,
        models: Dict[str, Any],
        n_episodes: int = 5,
        seed: Optional[int] = None,
    ) -> Dict[str, BenchmarkResult]:
        """
        Benchmark multiple models on the same environment and return
        a result dict keyed by model name.

        Parameters:
        -----------
        models : dict
            Mapping of name → model object.
        n_episodes : int
            Episodes per model.
        seed : int or None
            Shared seed for all runs.

        Returns:
        --------
        dict of name → BenchmarkResult
        """
        results: Dict[str, BenchmarkResult] = {}
        original_model = self.model

        for name, mdl in models.items():
            logger.info("Benchmarking model: %s", name)
            self.model = mdl
            results[name] = self.run(n_episodes=n_episodes, seed=seed)

        self.model = original_model
        return results

    @staticmethod
    def leaderboard(results: Dict[str, BenchmarkResult]) -> str:
        """
        Format a comparison table sorted by decision-matrix score.

        Parameters:
        -----------
        results : dict
            Output of compare().

        Returns:
        --------
        str: Formatted leaderboard table.
        """
        rows = sorted(
            results.items(),
            key=lambda kv: kv[1].decision_matrix_score,
            reverse=True,
        )
        lines = [
            "-" * 70,
            f"{'Rank':<5} {'Model':<25} {'Score':>6} {'Return':>8} {'MaxDD':>8} {'Balance':>9}",
            "-" * 70,
        ]
        for rank, (name, res) in enumerate(rows, 1):
            lines.append(
                f"{rank:<5} {name:<25} {res.decision_matrix_score:>6.1f}"
                f" {res.mean_return * 100:>+7.2f}%"
                f" {res.mean_max_drawdown * 100:>7.2f}%"
                f" {res.action_balance_score:>8.1f}"
            )
        lines.append("-" * 70)
        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _run_episode(self, seed: Optional[int] = None) -> dict:
        """
        Execute a single evaluation episode.

        Returns a dict with keys:
            return, length, max_drawdown, action_counts
        """
        reset_kwargs: dict = {}
        if seed is not None:
            reset_kwargs["seed"] = seed

        obs, _ = self._safe_reset(**reset_kwargs)

        # Reset recurrent hidden state for PPOAgent-style models
        self._hidden = self._get_initial_hidden()

        initial_equity = self._get_equity()
        peak_equity = initial_equity
        min_equity = initial_equity

        action_counts: Dict[int, int] = {a: 0 for a in range(self.n_actions)}
        step_count = 0
        terminated = False
        truncated = False

        while not (terminated or truncated):
            action = self._predict(obs)
            action_int = int(action)
            action_counts[action_int] = action_counts.get(action_int, 0) + 1

            obs, _reward, terminated, truncated, _info = self.env.step(action_int)
            step_count += 1

            equity = self._get_equity()
            peak_equity = max(peak_equity, equity)
            min_equity = min(min_equity, equity)

        final_equity = self._get_equity()

        # Return as fraction of initial equity
        ep_return = (final_equity - initial_equity) / max(initial_equity, 1e-8)

        # Max drawdown from peak (worst intra-episode)
        max_dd = (peak_equity - min_equity) / max(peak_equity, 1e-8)

        return {
            "return": ep_return,
            "length": step_count,
            "max_drawdown": max_dd,
            "action_counts": action_counts,
        }

    def _get_initial_hidden(self) -> Any:
        """Return initial hidden state for recurrent models, or None."""
        if hasattr(self.model, "get_initial_hidden_state"):
            try:
                return self.model.get_initial_hidden_state()
            except Exception:
                pass
        return None

    def _predict(self, obs: Any) -> int:
        """
        Call model with graceful fallback for different agent APIs.

        Supports:
        - PPOAgent (BITCOIN4Traders): select_action(obs, hidden, deterministic)
          → (action, log_prob, value, hidden)
        - SB3-style: model.predict(obs, deterministic=True) → (action, state)
        - Simple:    model.predict(obs) → action (int or array)
        """
        # ── PPOAgent (select_action API) ──────────────────────────────────
        if hasattr(self.model, "select_action"):
            try:
                action, _log_prob, _value, self._hidden = self.model.select_action(
                    obs,
                    hidden=self._hidden,
                    deterministic=self.deterministic,
                )
                if hasattr(action, "item"):
                    action = action.item()
                return int(action)
            except Exception:
                pass  # Fall through to generic path

        # ── Generic predict() API (SB3, custom) ───────────────────────────
        try:
            result = self.model.predict(obs, deterministic=self.deterministic)
            if isinstance(result, tuple):
                action = result[0]
            else:
                action = result
        except TypeError:
            result = self.model.predict(obs)
            action = result[0] if isinstance(result, tuple) else result

        if hasattr(action, "item"):
            action = action.item()
        return int(action)

    def _safe_reset(self, **kwargs) -> tuple:
        """
        Call env.reset() with graceful fallback.

        Gymnasium API returns (obs, info).
        Older Gym API returns just obs.
        """
        result = self.env.reset(**kwargs)
        if isinstance(result, tuple) and len(result) == 2:
            return result  # (obs, info)
        return result, {}  # wrap legacy API

    def _get_equity(self) -> float:
        """
        Try to read current equity from the environment.

        Checks common attribute names used across BITCOIN4Traders envs.
        Falls back to 1.0 if no equity attribute is found.
        """
        for attr in ("equity", "_equity", "portfolio_value", "balance"):
            val = getattr(self.env, attr, None)
            if val is not None:
                try:
                    return float(val)
                except (TypeError, ValueError):
                    pass
        return 1.0
