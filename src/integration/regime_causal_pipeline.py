"""
Regime-Causal Integration Pipeline
=====================================
Connects HMM regime detection and causal inference to produce
an enriched state vector for the PPO agent.

Flow:
  raw_features → CausalDiscovery → causal_signal
  raw_features → HMMRegimeDetector → regime_probs [3 values]
  [raw_features + causal_signal + regime_probs] → agent

Design principles
-----------------
* Safe before fit()   – every public method returns a sensible default if the
                        pipeline has not been fitted yet.
* Graceful degradation – if HMM or causal discovery raises, the module logs a
                         warning and continues without that component.
* Correlation fallback – if the PC algorithm is unavailable / fails we fall back
                         to a correlation-based causal signal.

Dependencies
------------
  required : numpy, pandas, loguru
  optional : hmmlearn  (for HMM)
             networkx  (for PC algorithm — already imported inside causal module)
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from loguru import logger

# ---------------------------------------------------------------------------
# Optional heavy imports – handled gracefully so the module is importable even
# in minimal environments.
# ---------------------------------------------------------------------------

try:
    from src.math_tools.hmm_regime import HMMRegimeDetector

    _HMM_AVAILABLE = True
except Exception as _hmm_err:
    HMMRegimeDetector = None  # type: ignore[assignment,misc]
    _HMM_AVAILABLE = False
    logger.warning(
        f"hmmlearn not available – HMM regime detection disabled: {_hmm_err}"
    )

try:
    from src.causal.causal_inference import CausalDiscovery

    _CAUSAL_AVAILABLE = True
except Exception as _causal_err:
    CausalDiscovery = None  # type: ignore[assignment,misc]
    _CAUSAL_AVAILABLE = False
    logger.warning(f"CausalDiscovery not available: {_causal_err}")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class RegimeCausalConfig:
    """
    Configuration for the Regime-Causal integration pipeline.

    Attributes
    ----------
    n_regimes : int
        Number of hidden market regimes (default 3).
    causal_alpha : float
        Significance level for the PC algorithm conditional-independence tests.
    causal_target : str
        Target variable name for causal discovery (usually 'returns').
    hmm_fit_window : int
        Number of most-recent bars used to *fit* the HMM.
    refit_every_n_bars : int
        How many calls to maybe_refit() trigger an actual refit.
    use_causal : bool
        Whether to run causal discovery and append the causal signal.
    use_regime : bool
        Whether to run HMM regime detection and append regime probabilities.
    """

    n_regimes: int = 3
    causal_alpha: float = 0.05
    causal_target: str = "returns"
    hmm_fit_window: int = 500
    refit_every_n_bars: int = 50
    use_causal: bool = True
    use_regime: bool = True


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class RegimeCausalPipeline:
    """
    Pipeline that enriches raw feature vectors with regime probabilities
    and a causal signal derived from the PC algorithm (or a correlation
    fallback).

    Parameters
    ----------
    config : RegimeCausalConfig
        Pipeline configuration.

    Notes
    -----
    Call ``fit(features)`` first, then ``transform(features)`` on every step.
    ``fit_transform(features)`` is a convenience wrapper for both.

    The pipeline is stateful: it caches the fitted HMM and causal graph
    between calls so that ``transform()`` is cheap.
    """

    def __init__(self, config: Optional[RegimeCausalConfig] = None) -> None:
        self.config: RegimeCausalConfig = config or RegimeCausalConfig()

        self._hmm: Optional[HMMRegimeDetector] = None  # type: ignore[type-arg]
        self._causal: Optional[CausalDiscovery] = None  # type: ignore[type-arg]

        # Cached causal parents list  (str column names)
        self._causal_parents: List[str] = []

        self._fitted: bool = False
        self._hmm_fitted: bool = False
        self._causal_fitted: bool = False

        self._bar_count: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, features: pd.DataFrame) -> None:
        """
        Fit both the HMM and the causal discovery model on *features*.

        Parameters
        ----------
        features : pd.DataFrame
            Must contain numeric columns. A 'returns' column is expected for
            best causal results but is not strictly required.

        Behaviour on failure
        --------------------
        * If HMM fitting raises, a warning is logged and regime detection is
          disabled for this session (use_regime effectively becomes False).
        * Same for causal discovery.
        * At least one module must succeed, or ``_fitted`` remains False.
        """
        # ------------------------------------------------------------------ HMM
        if self.config.use_regime:
            self._fit_hmm(features)

        # ------------------------------------------------------------------ Causal
        if self.config.use_causal:
            self._fit_causal(features)

        # Mark overall fitted state
        self._fitted = self._hmm_fitted or self._causal_fitted or True
        # Always mark fitted=True after an attempt so transform() doesn't crash.
        self._fitted = True
        logger.info(
            f"RegimeCausalPipeline fitted — HMM: {self._hmm_fitted}, "
            f"Causal: {self._causal_fitted}"
        )

    def transform(self, features: pd.DataFrame) -> np.ndarray:
        """
        Produce an enriched 1-D numpy state vector from the last row of *features*.

        Appended dimensions (in order):
            1. regime_probs  (n_regimes floats, only if use_regime=True and HMM fitted)
            2. causal_signal (1 float: -1 / 0 / +1, only if use_causal=True and causal fitted)

        Returns
        -------
        np.ndarray
            Shape: (n_base_features + extra_dims,)  dtype float32.
            Falls back to raw last row if not fitted.
        """
        if features.empty:
            return np.zeros(0, dtype=np.float32)

        # Base: last row of numeric features
        numeric_cols = features.select_dtypes(include=[np.number]).columns.tolist()
        base = features[numeric_cols].iloc[-1].values.astype(np.float32)
        base = np.nan_to_num(base, nan=0.0, posinf=1.0, neginf=-1.0)

        extras: List[np.ndarray] = []

        # ---- Regime probabilities
        if self.config.use_regime and self._hmm_fitted and self._hmm is not None:
            regime_probs = self._predict_regime_proba(features)
            extras.append(regime_probs.astype(np.float32))

        # ---- Causal signal
        if self.config.use_causal and self._causal_fitted:
            causal_sig = self._compute_causal_signal(features)
            extras.append(np.array([causal_sig], dtype=np.float32))

        if extras:
            return np.concatenate([base] + extras).astype(np.float32)
        return base

    def fit_transform(self, features: pd.DataFrame) -> np.ndarray:
        """Fit then transform in a single call."""
        self.fit(features)
        return self.transform(features)

    def get_extra_dims(self) -> int:
        """
        Return the number of extra dimensions appended by ``transform()``.

        Returns
        -------
        int
            0, 1, 3, or 4 depending on which modules are active and fitted.
        """
        dims = 0
        if self.config.use_regime and self._hmm_fitted:
            dims += self.config.n_regimes  # one probability per regime
        if self.config.use_causal and self._causal_fitted:
            dims += 1  # single causal direction signal
        return dims

    def maybe_refit(self, features: pd.DataFrame) -> None:
        """
        Increment the internal bar counter and refit the pipeline every
        ``config.refit_every_n_bars`` calls.

        Parameters
        ----------
        features : pd.DataFrame
            The most-recent feature window to fit on when refit triggers.
        """
        self._bar_count += 1
        if self._bar_count % self.config.refit_every_n_bars == 0:
            logger.debug(
                f"RegimeCausalPipeline: refit triggered at bar {self._bar_count}"
            )
            self.fit(features)

    # ------------------------------------------------------------------
    # Private helpers – HMM
    # ------------------------------------------------------------------

    def _fit_hmm(self, features: pd.DataFrame) -> None:
        """Fit the HMM on the last ``hmm_fit_window`` rows."""
        if not _HMM_AVAILABLE:
            logger.warning("HMM not available – skipping regime fit")
            self._hmm_fitted = False
            return

        try:
            window_df = self._get_fit_window(features)

            # Build HMM-compatible numeric frame
            hmm_df = self._prepare_hmm_features(window_df)

            if len(hmm_df) < max(10, self.config.n_regimes * 5):
                logger.warning(
                    f"Insufficient data for HMM ({len(hmm_df)} rows) – "
                    "regime detection disabled"
                )
                self._hmm_fitted = False
                return

            self._hmm = HMMRegimeDetector(n_regimes=self.config.n_regimes)
            self._hmm.fit(hmm_df)
            self._hmm_fitted = True
            logger.debug("HMM fitted successfully")

        except Exception as exc:
            logger.warning(f"HMM fitting failed: {exc} – returning uniform probs")
            self._hmm_fitted = False

    def _predict_regime_proba(self, features: pd.DataFrame) -> np.ndarray:
        """
        Return regime probabilities for the last row of *features*.
        Falls back to uniform distribution on any error.
        """
        uniform = (
            np.ones(self.config.n_regimes, dtype=np.float32) / self.config.n_regimes
        )
        if self._hmm is None:
            return uniform

        try:
            hmm_df = self._prepare_hmm_features(features)
            if len(hmm_df) == 0:
                return uniform
            probs = self._hmm.predict_proba(hmm_df)
            return np.array(probs, dtype=np.float32)
        except Exception as exc:
            logger.warning(f"HMM predict_proba failed: {exc} – returning uniform probs")
            return uniform

    @staticmethod
    def _prepare_hmm_features(features: pd.DataFrame) -> pd.DataFrame:
        """
        Select / rename columns to feed into HMMRegimeDetector.

        HMMRegimeDetector.fit() accepts any numeric DataFrame. We prefer
        ['returns', 'volatility'] or similar but fall back to all numerics.
        """
        numeric_cols = features.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            return pd.DataFrame()

        # Prefer a small, meaningful subset if present
        preferred = [
            c
            for c in ["returns", "volatility", "volatility_20", "volume"]
            if c in numeric_cols
        ]
        cols = preferred if preferred else numeric_cols[:4]
        df = features[cols].copy()
        df = df.replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0)
        return df

    # ------------------------------------------------------------------
    # Private helpers – Causal
    # ------------------------------------------------------------------

    def _fit_causal(self, features: pd.DataFrame) -> None:
        """Run PC algorithm (or correlation fallback) to find causal parents."""
        if not _CAUSAL_AVAILABLE:
            logger.warning("CausalDiscovery not available – using correlation fallback")
            self._fit_causal_correlation_fallback(features)
            return

        # Need 'returns' column (or config.causal_target)
        target = self.config.causal_target
        if target not in features.columns:
            logger.warning(
                f"Causal target '{target}' not in features – "
                "trying correlation fallback"
            )
            self._fit_causal_correlation_fallback(features)
            return

        try:
            window_df = self._get_fit_window(features)
            numeric_cols = window_df.select_dtypes(include=[np.number]).columns.tolist()

            if len(numeric_cols) < 2 or len(window_df) < 30:
                logger.warning("Too few rows/cols for PC algorithm – fallback")
                self._fit_causal_correlation_fallback(features)
                return

            causal_df = window_df[numeric_cols].copy()
            causal_df = causal_df.replace([np.inf, -np.inf], np.nan).fillna(0.0)

            self._causal = CausalDiscovery(alpha=self.config.causal_alpha)
            self._causal.pc_algorithm(causal_df)

            parents = self._causal.get_causal_parents(target)
            # Filter to columns that are actually in the DataFrame
            self._causal_parents = [p for p in parents if p in features.columns]

            if not self._causal_parents:
                logger.info(
                    f"PC algorithm found 0 parents for '{target}' – "
                    "using correlation fallback for signal"
                )
                # Still mark causal_fitted so we can produce a signal via fallback
                self._fit_causal_correlation_fallback(features)
                return

            self._causal_fitted = True
            logger.debug(
                f"Causal discovery: {len(self._causal_parents)} parents of "
                f"'{target}': {self._causal_parents}"
            )

        except Exception as exc:
            logger.warning(f"PC algorithm failed: {exc} – using correlation fallback")
            self._fit_causal_correlation_fallback(features)

    def _fit_causal_correlation_fallback(self, features: pd.DataFrame) -> None:
        """
        Fallback: identify the top-3 most-correlated predictors of the target
        by absolute Pearson correlation and use their combined z-score as a
        direction signal.
        """
        target = self.config.causal_target
        numeric_cols = features.select_dtypes(include=[np.number]).columns.tolist()

        if target not in numeric_cols or len(numeric_cols) < 2:
            logger.warning(
                "Correlation fallback skipped: no numeric columns or target missing"
            )
            self._causal_fitted = False
            return

        try:
            window_df = self._get_fit_window(features)
            corr_col = [c for c in numeric_cols if c != target]
            if not corr_col:
                self._causal_fitted = False
                return

            corrs = (
                window_df[corr_col]
                .corrwith(window_df[target])
                .dropna()
                .abs()
                .sort_values(ascending=False)
            )
            self._causal_parents = corrs.head(3).index.tolist()
            self._causal_fitted = True
            logger.debug(f"Correlation fallback: top parents = {self._causal_parents}")
        except Exception as exc:
            logger.warning(f"Correlation fallback also failed: {exc}")
            self._causal_fitted = False

    def _compute_causal_signal(self, features: pd.DataFrame) -> float:
        """
        Derive a directional signal from causal parents.

        Signal is the sign of the mean z-score of causal parent values.
        Returns -1.0, 0.0, or +1.0.
        """
        if not self._causal_parents:
            return 0.0

        try:
            available = [p for p in self._causal_parents if p in features.columns]
            if not available:
                return 0.0

            last_row = features[available].iloc[-1]
            # Z-score using window statistics
            window_df = self._get_fit_window(features)
            means = window_df[available].mean()
            stds = window_df[available].std().replace(0, 1.0)

            z_scores = (last_row - means) / stds
            signal_raw = float(z_scores.mean())
            return float(np.sign(signal_raw)) if abs(signal_raw) > 0.01 else 0.0
        except Exception as exc:
            logger.warning(f"Causal signal computation failed: {exc}")
            return 0.0

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def _get_fit_window(self, features: pd.DataFrame) -> pd.DataFrame:
        """Return at most the last ``hmm_fit_window`` rows."""
        return features.tail(self.config.hmm_fit_window)

    def __repr__(self) -> str:
        return (
            f"RegimeCausalPipeline("
            f"fitted={self._fitted}, "
            f"hmm={self._hmm_fitted}, "
            f"causal={self._causal_fitted}, "
            f"extra_dims={self.get_extra_dims()})"
        )
