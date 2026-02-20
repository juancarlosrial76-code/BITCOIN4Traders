"""
ANTI-BIAS FRAMEWORK – Walk-Forward CV
======================================
Purged Walk-Forward Cross-Validation mit Purging + Embargo.
Verhindert Lookahead-Bias beim Training.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Generator, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("antibias.validation")


@dataclass
class WalkForwardConfig:
    """Konfiguration für Walk-Forward Cross-Validation."""

    n_splits: int = 5
    test_pct: float = 0.20
    embargo_pct: float = 0.01
    holdout_pct: float = 0.15
    min_train_pct: float = 0.30
    feature_lookback: int = 100
    purge: bool = True
    expanding_window: bool = True


@dataclass
class FoldSplit:
    """Ergebnis eines Walk-Forward Splits."""

    fold_id: int
    train_idx: np.ndarray
    test_idx: np.ndarray
    embargo_idx: np.ndarray
    purged_count: int = 0

    @property
    def train_size(self) -> int:
        return len(self.train_idx)

    @property
    def test_size(self) -> int:
        return len(self.test_idx)

    def __repr__(self) -> str:
        return (
            f"Fold {self.fold_id}: train={self.train_size} "
            f"test={self.test_size} "
            f"embargo={len(self.embargo_idx)} "
            f"purged={self.purged_count}"
        )


class PurgedWalkForwardCV:
    """
    Purged Walk-Forward CV mit Purging + Embargo.

    Usage:
        cv = PurgedWalkForwardCV(config)
        folds, holdout = cv.split(n_samples=len(df))

        for fold in folds:
            X_train = X[fold.train_idx]
            X_test = X[fold.test_idx]
    """

    def __init__(self, config: Optional[WalkForwardConfig] = None):
        self.config = config or WalkForwardConfig()

    def split(
        self,
        n_samples: int,
        timestamps: Optional[np.ndarray] = None,
    ) -> Tuple[List[FoldSplit], np.ndarray]:
        """
        Erzeugt Walk-Forward Splits.

        Returns:
            (folds, holdout_idx)
        """
        cfg = self.config
        n = n_samples

        holdout_n = int(n * cfg.holdout_pct)
        working_n = n - holdout_n
        holdout_idx = np.arange(working_n, n)

        embargo_n = max(1, int(n * cfg.embargo_pct))
        test_n = int(working_n * cfg.test_pct)

        folds: List[FoldSplit] = []

        for fold_id in range(cfg.n_splits):
            test_end = working_n - fold_id * (test_n + embargo_n)
            test_start = test_end - test_n

            if test_start < int(working_n * cfg.min_train_pct):
                logger.warning(
                    "Fold %d: test_start=%d zu früh, übersprungen.", fold_id, test_start
                )
                break

            embargo_start = test_start - embargo_n
            embargo_idx = np.arange(embargo_start, test_start)

            if cfg.expanding_window:
                train_start = 0
            else:
                window_size = embargo_start - embargo_n
                train_start = max(0, embargo_start - window_size)

            raw_train_idx = np.arange(train_start, embargo_start)

            if cfg.purge:
                train_idx, purged = self._purge(
                    raw_train_idx, test_start, cfg.feature_lookback
                )
            else:
                train_idx = raw_train_idx
                purged = 0

            test_idx = np.arange(test_start, test_end)

            fold = FoldSplit(
                fold_id=fold_id,
                train_idx=train_idx,
                test_idx=test_idx,
                embargo_idx=embargo_idx,
                purged_count=purged,
            )
            folds.append(fold)

            logger.info(
                "Fold %d: train=[%d..%d] embargo=[%d..%d] test=[%d..%d] purged=%d",
                fold_id,
                train_idx[0] if len(train_idx) else -1,
                train_idx[-1] if len(train_idx) else -1,
                embargo_idx[0] if len(embargo_idx) else -1,
                embargo_idx[-1] if len(embargo_idx) else -1,
                test_start,
                test_end - 1,
                purged,
            )

        folds = sorted(folds, key=lambda f: f.test_idx[0])

        logger.info(
            "WalkForward splits: %d folds | holdout=%d bars (%.1f%% of data)",
            len(folds),
            len(holdout_idx),
            cfg.holdout_pct * 100,
        )
        return folds, holdout_idx

    @staticmethod
    def _purge(
        train_idx: np.ndarray,
        test_start: int,
        feature_lookback: int,
    ) -> Tuple[np.ndarray, int]:
        """
        Entfernt Train-Samples deren Feature-Fenster in die Test-Zone ragt.
        """
        cutoff = test_start - feature_lookback
        purge_mask = train_idx >= cutoff
        n_purged = int(purge_mask.sum())
        clean_idx = train_idx[~purge_mask]
        return clean_idx, n_purged


class PurgedScaler:
    """
    Scaler der garantiert:
      1. fit() nur auf Trainingsdaten
      2. transform() auf Test-Daten verwendet gespeicherte Train-Statistiken
      3. Live-Inference verwendet denselben gespeicherten State

    Verhindert den häufigsten Lookahead-Bug!
    """

    def __init__(self, method: str = "zscore", clip: float = 5.0):
        assert method in ("zscore", "minmax", "robust")
        self.method = method
        self.clip = clip
        self._fitted = False
        self._loc: Optional[np.ndarray] = None
        self._scale: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> "PurgedScaler":
        """Fit NUR auf Trainingsdaten."""
        if self.method == "zscore":
            self._loc = X.mean(axis=0)
            self._scale = X.std(axis=0) + 1e-8
        elif self.method == "minmax":
            self._loc = X.min(axis=0)
            self._scale = (X.max(axis=0) - X.min(axis=0)) + 1e-8
        elif self.method == "robust":
            self._loc = np.median(X, axis=0)
            q75, q25 = np.percentile(X, [75, 25], axis=0)
            self._scale = (q75 - q25) + 1e-8
        self._fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform mit gespeicherten Train-Statistiken."""
        if not self._fitted:
            raise RuntimeError(
                "PurgedScaler.fit() muss vor transform() aufgerufen werden."
            )
        scaled = (X - self._loc) / self._scale
        return np.clip(scaled, -self.clip, self.clip)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Nur für Trainingsdaten – fit + transform in einem."""
        return self.fit(X).transform(X)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Not fitted.")
        return X * self._scale + self._loc

    def save(self, path: str) -> None:
        np.savez(path, loc=self._loc, scale=self._scale, method=np.array([self.method]))
        logger.info("Scaler saved → %s", path)

    @classmethod
    def load(cls, path: str) -> "PurgedScaler":
        data = np.load(path, allow_pickle=True)
        scaler = cls(method=str(data["method"][0]))
        scaler._loc = data["loc"]
        scaler._scale = data["scale"]
        scaler._fitted = True
        return scaler


class LeakDetector:
    """
    Automatischer Test ob Lookahead-Leak vorhanden ist.
    """

    def __init__(self, significance_threshold: float = 0.52):
        self.threshold = significance_threshold
        self._results: dict = {}

    def check_feature_future_correlation(
        self,
        X: np.ndarray,
        future_returns: np.ndarray,
        feature_names: Optional[list] = None,
        lag: int = 1,
    ) -> dict:
        """
        Berechnet Korrelation jedes Features mit future_returns[+lag].
        Hohe Korrelation (>0.3) deutet auf Lookahead hin.
        """
        n = len(X)
        if len(future_returns) != n:
            raise ValueError("X und future_returns müssen gleiche Länge haben.")

        results = {}
        for i in range(X.shape[1]):
            feat_name = feature_names[i] if feature_names else f"feature_{i}"
            x_slice = X[: n - lag, i]
            y_slice = future_returns[lag:]
            corr = float(np.corrcoef(x_slice, y_slice)[0, 1])
            results[feat_name] = corr

            if abs(corr) > 0.3:
                logger.warning(
                    "POTENTIELLER LOOKAHEAD: Feature '%s' korreliert %.3f mit future_return[+%d]",
                    feat_name,
                    corr,
                    lag,
                )

        self._results["feature_future_corr"] = results
        high_corr = {k: v for k, v in results.items() if abs(v) > 0.3}
        if high_corr:
            logger.error(
                "LEAK VERDACHT: %d Features mit |corr| > 0.3: %s",
                len(high_corr),
                high_corr,
            )
        else:
            logger.info("Leak-Check OK: kein Feature mit |corr| > 0.3 zur Zukunft.")

        return results

    def report(self) -> str:
        lines = ["═" * 50, "  LEAK DETECTION REPORT", "═" * 50]
        if "feature_future_corr" in self._results:
            high = {
                k: v
                for k, v in self._results["feature_future_corr"].items()
                if abs(v) > 0.15
            }
            lines.append(f"High-correlation features (|r|>0.15): {len(high)}")
            for k, v in sorted(high.items(), key=lambda x: -abs(x[1])):
                lines.append(f"  {k:<30} r={v:+.4f}")
        return "\n".join(lines)
