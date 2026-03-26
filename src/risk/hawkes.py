"""
Hawkes Point Process — Order Flow Toxicity
==========================================
Self-exciting point process to model clustering of large trades.
High conditional intensity λ(t) signals informed trading / potential crash.

The conditional intensity function:

    λ(t) = μ + Σᵢ α·sᵢ·exp(-β·(t - tᵢ))   for all past events tᵢ < t

where
    μ  = baseline (unconditional) arrival rate
    α  = excitation amplitude per unit event size
    sᵢ = size weight of event i  (large trades excite more)
    β  = exponential decay rate

Stationarity condition:  α/β < 1  (branching ratio < 1)

Reference:
    Hawkes, A. G. (1971). "Spectra of some self-exciting and mutually
    exciting point processes." Biometrika 58(1), 83–90.

    Bacry, E., Mastromatteo, I., Muzy, J.-F. (2015). "Hawkes processes
    in finance." Market Microstructure and Liquidity 1(01).
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional

from loguru import logger


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class HawkesConfig:
    """
    Hyperparameters for a univariate Hawkes process.

    Attributes
    ----------
    mu : float
        Baseline (background) intensity — the unconditional arrival rate
        of events in the absence of any recent activity.  Must be > 0.
    alpha : float
        Excitation amplitude.  Each event temporarily raises λ(t) by
        alpha * size.  For stationarity, must satisfy alpha < beta.
    beta : float
        Exponential decay rate.  Controls how quickly the excitation of
        each past event fades.  Larger beta → faster forgetting.
    toxicity_threshold : float
        λ(t)/μ ratio above which the market is considered "toxic".
        A ratio of 2.0 means the current intensity is twice the baseline,
        indicating significant trade clustering (informed order flow).
    window_size : int
        Maximum number of past events retained in memory.  Older events
        are pruned to keep memory bounded; their contribution to λ(t) is
        negligible once exp(-β·Δt) ≈ 0.
    """

    mu: float = 0.1
    alpha: float = 0.5
    beta: float = 1.0
    toxicity_threshold: float = 2.0
    window_size: int = 100

    def __post_init__(self) -> None:
        if self.mu <= 0:
            raise ValueError(f"mu must be > 0, got {self.mu}")
        if self.alpha < 0:
            raise ValueError(f"alpha must be >= 0, got {self.alpha}")
        if self.beta <= 0:
            raise ValueError(f"beta must be > 0, got {self.beta}")
        if self.toxicity_threshold <= 1.0:
            raise ValueError(
                f"toxicity_threshold must be > 1.0, got {self.toxicity_threshold}"
            )
        if self.window_size < 1:
            raise ValueError(f"window_size must be >= 1, got {self.window_size}")

        br = self.alpha / self.beta
        if br >= 1.0:
            logger.warning(
                f"HawkesConfig: branching ratio α/β = {br:.4f} >= 1.0 — "
                "process is non-stationary (super-critical).  "
                "Consider reducing alpha or increasing beta."
            )


# ---------------------------------------------------------------------------
# Core Process
# ---------------------------------------------------------------------------


class HawkesProcess:
    """
    Univariate Hawkes self-exciting point process.

    Models the conditional intensity of large-trade arrivals:

        λ(t) = μ + Σᵢ α·sᵢ·exp(-β·(t - tᵢ))

    Usage
    -----
    >>> cfg = HawkesConfig(mu=0.1, alpha=0.5, beta=1.0)
    >>> hp  = HawkesProcess(cfg)
    >>> hp.add_event(timestamp=1.0, size=1.0)
    >>> hp.add_event(timestamp=1.5, size=2.5)   # large trade — more excitation
    >>> lam = hp.compute_intensity(t=2.0)
    >>> print(f"λ(2.0) = {lam:.4f}")
    >>> print(f"toxic  = {hp.is_toxic(t=2.0)}")
    """

    def __init__(self, config: Optional[HawkesConfig] = None) -> None:
        self.config: HawkesConfig = config if config is not None else HawkesConfig()

        # Parallel lists: one entry per event
        self._event_times: List[float] = []  # tᵢ
        self._event_sizes: List[float] = []  # sᵢ  (size weights)

        # Snapshot of λ(t) each time it is queried
        self._intensity_history: List[float] = []

        logger.debug(
            f"HawkesProcess initialised | μ={self.config.mu}  "
            f"α={self.config.alpha}  β={self.config.beta}  "
            f"n*={self.config.alpha / self.config.beta:.4f}"
        )

    # ------------------------------------------------------------------
    # Event ingestion
    # ------------------------------------------------------------------

    def add_event(self, timestamp: float, size: float = 1.0) -> None:
        """
        Record a new event (e.g. a large trade).

        Parameters
        ----------
        timestamp : float
            Arrival time of the event.  Must be monotonically increasing
            for intensity calculations to be physically meaningful, but
            the class does not enforce ordering (use-case flexibility).
        size : float
            Relative size weight of the event.  Values > 1.0 amplify
            the α contribution so that unusually large trades excite
            future activity more strongly.  Must be > 0.
        """
        if size <= 0:
            raise ValueError(f"Event size must be > 0, got {size}")

        self._event_times.append(float(timestamp))
        self._event_sizes.append(float(size))

        # Prune oldest events beyond window_size to keep memory bounded.
        # Events are kept in insertion order; oldest = index 0.
        if len(self._event_times) > self.config.window_size:
            excess = len(self._event_times) - self.config.window_size
            del self._event_times[:excess]
            del self._event_sizes[:excess]

    # ------------------------------------------------------------------
    # Intensity computation
    # ------------------------------------------------------------------

    def compute_intensity(self, t: float) -> float:
        """
        Evaluate the conditional intensity λ(t).

        λ(t) = μ + Σ_{tᵢ < t} α · sᵢ · exp(-β · (t - tᵢ))

        Parameters
        ----------
        t : float
            The evaluation time.

        Returns
        -------
        float
            Conditional intensity ≥ μ.
        """
        mu, alpha, beta = self.config.mu, self.config.alpha, self.config.beta

        excitation = 0.0
        for ti, si in zip(self._event_times, self._event_sizes):
            dt = t - ti
            if dt > 0.0:  # only past events contribute
                excitation += alpha * si * np.exp(-beta * dt)

        lam = mu + excitation
        self._intensity_history.append(lam)
        return lam

    # ------------------------------------------------------------------
    # Derived quantities
    # ------------------------------------------------------------------

    def is_toxic(self, t: float) -> bool:
        """
        Return True if the current market is in a "toxic" state.

        Toxicity is defined as λ(t)/μ > toxicity_threshold, meaning the
        conditional arrival rate is significantly above baseline — a signal
        of informed-trader clustering or impending volatility.

        Parameters
        ----------
        t : float
            Evaluation time (passed to compute_intensity).

        Returns
        -------
        bool
        """
        lam = self.compute_intensity(t)
        ratio = lam / self.config.mu
        return bool(ratio > self.config.toxicity_threshold)

    def branching_ratio(self) -> float:
        """
        Compute the theoretical branching ratio n* = α / β.

        The branching ratio is the expected number of offspring events
        generated by a single parent event.  For the process to be
        stationary (mean-reverting to μ), n* must be < 1.

        Returns
        -------
        float
            n* = α / β.  Warns via logger if >= 1.0.
        """
        n_star = self.config.alpha / self.config.beta
        if n_star >= 1.0:
            logger.warning(
                f"Branching ratio n* = {n_star:.4f} >= 1.0 — "
                "process is non-stationary.  Intensity may diverge."
            )
        return n_star

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear all event history and intensity snapshots."""
        self._event_times.clear()
        self._event_sizes.clear()
        self._intensity_history.clear()
        logger.debug("HawkesProcess reset — all event history cleared.")

    def get_intensity_series(self) -> np.ndarray:
        """
        Return the recorded intensity snapshots as a NumPy array.

        Each call to compute_intensity() or is_toxic() appends a value.

        Returns
        -------
        np.ndarray, dtype=float64
        """
        return np.array(self._intensity_history, dtype=float)

    def __repr__(self) -> str:
        return (
            f"HawkesProcess("
            f"μ={self.config.mu}, α={self.config.alpha}, β={self.config.beta}, "
            f"n*={self.branching_ratio():.3f}, "
            f"events={len(self._event_times)})"
        )


# ---------------------------------------------------------------------------
# Integration helper
# ---------------------------------------------------------------------------


def hawkes_from_trades(
    prices: np.ndarray,
    volumes: np.ndarray,
    threshold_vol_mult: float = 2.0,
    config: Optional[HawkesConfig] = None,
) -> HawkesProcess:
    """
    Construct and populate a HawkesProcess from a trade tape.

    Only trades with volume > threshold_vol_mult * mean(volumes) are
    ingested as events.  The size weight fed to add_event() is
    volume / mean_volume, so abnormally large trades excite more.

    Parameters
    ----------
    prices : np.ndarray, shape (N,)
        Trade prices (used for future extension; not consumed here directly
        but kept for API consistency with microstructure helpers).
    volumes : np.ndarray, shape (N,)
        Per-trade volumes.
    threshold_vol_mult : float
        Multiplier on the mean volume to set the "large trade" threshold.
        Default 2.0 → events are trades with volume > 2× average.
    config : HawkesConfig, optional
        Process configuration.  A default HawkesConfig() is used if None.

    Returns
    -------
    HawkesProcess
        Process instance populated with large-trade events, ready for
        is_toxic() / compute_intensity() queries.

    Raises
    ------
    ValueError
        If prices and volumes have different lengths or are empty.

    Example
    -------
    >>> prices  = np.array([50_000.0] * 200)
    >>> volumes = np.random.exponential(scale=1.0, size=200)
    >>> hp = hawkes_from_trades(prices, volumes, threshold_vol_mult=2.0)
    >>> print(hp.is_toxic(t=float(len(volumes))))
    """
    prices = np.asarray(prices, dtype=float)
    volumes = np.asarray(volumes, dtype=float)

    if prices.shape != volumes.shape:
        raise ValueError(
            f"prices and volumes must have the same shape: "
            f"{prices.shape} vs {volumes.shape}"
        )
    if volumes.size == 0:
        raise ValueError("prices / volumes arrays must not be empty.")

    hp = HawkesProcess(config=config)

    mean_vol = float(np.mean(volumes))
    if mean_vol == 0.0:
        logger.warning("hawkes_from_trades: mean volume is 0 — no events ingested.")
        return hp

    threshold = threshold_vol_mult * mean_vol
    n_events = 0

    for idx, (price, vol) in enumerate(zip(prices, volumes)):
        if vol > threshold:
            # Use the trade index as a proxy for time (uniform spacing assumed).
            # Callers with real timestamps should use add_event() directly.
            size_weight = vol / mean_vol
            hp.add_event(timestamp=float(idx), size=size_weight)
            n_events += 1

    logger.info(
        f"hawkes_from_trades: {n_events}/{len(volumes)} trades ingested as events "
        f"(threshold = {threshold_vol_mult:.1f}× mean vol = {threshold:.4f})"
    )
    return hp
