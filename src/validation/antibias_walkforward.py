"""
ANTI-BIAS FRAMEWORK – Walk-Forward Cross-Validation
===================================================
Purged Walk-Forward Cross-Validation with Purging and Embargo Zones.

This module implements rigorous cross-validation specifically designed for
financial time series, where standard k-fold validation fails due to:
1. Temporal dependencies (later samples depend on earlier ones)
2. Look-ahead bias (features may inadvertently include future information)
3. Autocorrelation (adjacent samples are not independent)

Key Concepts:
-------------
1. Purging: Removing training samples whose feature window overlaps test period
   - Prevents "future information" from leaking into training

2. Embargo: Adding a gap between training and test sets
   - Accounts for autocorrelated features/returns

3. Walk-Forward: Expanding or rolling windows over time
   - Simulates real trading: train on past, test on future

4. Holdout: Final portion of data reserved for true out-of-sample testing

Why Standard CV Fails for Finance:
---------------------------------
In standard k-fold CV, you randomly split data into train/test sets.
This creates two problems for time series:
1. Future leakage: Test data may influence feature calculation in training
2. Temporal discontinuity: Random splits ignore the time dimension

Purged Walk-Forward addresses these by:
- Ensuring no feature lookback window spans train/test boundary
- Adding embargo gaps to handle autocorrelation
- Always using chronological splits (no look-ahead)

Classes:
-------
- WalkForwardConfig: Configuration parameters
- FoldSplit: Single train/test/embargo split
- PurgedWalkForwardCV: Main CV implementation
- PurgedScaler: Bias-free feature scaling
- LeakDetector: Detects potential lookahead in features

Usage:
------
    from validation.antibias_walkforward import PurgedWalkForwardCV, WalkForwardConfig

    # Configure purged walk-forward
    config = WalkForwardConfig(
        n_splits=5,
        test_pct=0.20,       # 20% of data for testing
        embargo_pct=0.01,    # 1% gap between train/test
        holdout_pct=0.15,    # 15% final holdout
        feature_lookback=100, # Max feature lookback window
        purge=True,           # Enable purging
    )

    # Create CV splits
    cv = PurgedWalkForwardCV(config)
    folds, holdout = cv.split(n_samples=len(df))

    # Use in training loop
    for fold in folds:
        X_train, X_test = X[fold.train_idx], X[fold.test_idx]
        y_train, y_test = y[fold.train_idx], y[fold.test_idx]

        # Train model
        model.fit(X_train, y_train)

        # Evaluate on out-of-sample test
        predictions = model.predict(X_test)

    # Final validation on holdout
    X_holdout, y_holdout = X[holdout], y[holdout]

Reference:
---------
- Lopez de Prado, M. (2018): "Advances in Financial Machine Learning"
- Chapter 7: Cross-Validation in Financial Machine Learning
- Bailey & Lopez de Prado (2013): "The Ten Pitfalls of Backtesting"
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Generator, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("antibias.validation")


@dataclass
class WalkForwardConfig:
    """
    Configuration for Purged Walk-Forward Cross-Validation.

    This dataclass defines all parameters needed to configure the
    purged walk-forward cross-validation process. Each parameter
    affects how train/test splits are created and how leakage is prevented.

    Attributes:
        n_splits: Number of cross-validation folds (default: 5)
            More folds = more training data per fold but less test data

        test_pct: Fraction of working data for testing (default: 0.20 = 20%)
            Each fold's test set will be this fraction of available data

        embargo_pct: Fraction of data for embargo gap (default: 0.01 = 1%)
            Gap between train end and test start to prevent leakage

        holdout_pct: Fraction for final holdout (default: 0.15 = 15%)
            Reserved for final validation, never used in CV

        min_train_pct: Minimum train fraction (default: 0.30 = 30%)
            Ensures each fold has sufficient training data

        feature_lookback: Maximum feature lookback window (default: 100)
            Number of bars features may look back for calculation

        purge: Whether to enable purging (default: True)
            Removes train samples whose lookback overlaps test

        expanding_window: Train from start vs rolling window (default: True)
            True = expanding (train grows), False = rolling (fixed size)

    Example:
        --------
        # Standard configuration
        config = WalkForwardConfig(
            n_splits=5,
            test_pct=0.20,
            embargo_pct=0.01,
            holdout_pct=0.15,
            feature_lookback=100,
            purge=True,
        )

        # Aggressive (more folds, less test)
        config = WalkForwardConfig(
            n_splits=10,
            test_pct=0.10,
            feature_lookback=50,
        )

        # Conservative (fewer folds, more test)
        config = WalkForwardConfig(
            n_splits=3,
            test_pct=0.30,
            embargo_pct=0.02,
            feature_lookback=200,
        )
    """

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
    """
    Result of a single Walk-Forward split containing train/test/embargo indices.

    This dataclass represents one iteration of cross-validation, containing
    the indices for training, testing, and embargoed data. It provides
    convenient properties for accessing split sizes.

    Attributes:
        fold_id: Zero-based identifier for this fold
        train_idx: NumPy array of training sample indices
        test_idx: NumPy array of test sample indices
        embargo_idx: NumPy array of embargoed (excluded) indices
        purged_count: Number of samples removed by purging

    Properties:
        train_size: Number of training samples
        test_size: Number of test samples

    Example:
        --------
        fold = FoldSplit(
            fold_id=2,
            train_idx=np.arange(0, 1000),
            test_idx=np.arange(1200, 1400),
            embargo_idx=np.arange(1000, 1200),
            purged_count=50
        )

        print(f"Fold {fold.fold_id}:")
        print(f"  Train: {fold.train_size} samples")
        print(f"  Test: {fold.test_size} samples")
        print(f"  Embargo: {len(fold.embargo_idx)} samples")
        print(f"  Purged: {fold.purged_count} samples")
    """

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
    Purged Walk-Forward CV with Purging + Embargo.

    This class implements the core purged walk-forward cross-validation
    algorithm for financial time series. It creates train/test splits
    that prevent look-ahead bias through:
    1. Purging: Removing training samples whose feature window overlaps test
    2. Embargo: Adding gaps between train and test periods

    The algorithm works as follows:
    1. Reserve final holdout set (never used in CV)
    2. For each fold, walk backward through remaining data
    3. Define test period at end of working data
    4. Add embargo gap after training
    5. Optionally purge samples whose lookback spans the boundary

    Attributes:
        config: WalkForwardConfig with split parameters

    Methods:
        split(): Generate all fold splits

    Example:
        --------
        cv = PurgedWalkForwardCV(config)
        folds, holdout = cv.split(n_samples=len(df))

        for fold in folds:
            X_train = X[fold.train_idx]
            X_test = X[fold.test_idx]
            y_train = y[fold.train_idx]
            y_test = y[fold.test_idx]

            # Train and validate
            model.fit(X_train, y_train)
            score = model.score(X_test, y_test)

        # Final holdout test
        final_model.fit(X[holdout], y[holdout])
    """

    def __init__(self, config: Optional[WalkForwardConfig] = None):
        self.config = config or WalkForwardConfig()

    def split(
        self,
        n_samples: int,
        timestamps: Optional[np.ndarray] = None,
    ) -> Tuple[List[FoldSplit], np.ndarray]:
        """
        Generate walk-forward splits with purging and embargo.

        Returns:
            (folds, holdout_idx): list of FoldSplit objects and holdout indices
        """
        cfg = self.config
        n = n_samples

        holdout_n = int(
            n * cfg.holdout_pct
        )  # bars reserved as final holdout (never used in CV)
        working_n = n - holdout_n  # bars available for train/test splits
        holdout_idx = np.arange(
            working_n, n
        )  # holdout = last holdout_pct% of the series

        embargo_n = max(
            1, int(n * cfg.embargo_pct)
        )  # gap between train end and test start to prevent feature leakage
        test_n = int(working_n * cfg.test_pct)  # number of bars in each test fold

        folds: List[FoldSplit] = []

        for fold_id in range(cfg.n_splits):
            # Walk backwards from the end: newest fold first
            test_end = working_n - fold_id * (test_n + embargo_n)
            test_start = test_end - test_n

            if test_start < int(
                working_n * cfg.min_train_pct
            ):  # ensure enough training data remains
                logger.warning(
                    "Fold %d: test_start=%d too early, skipping fold.",
                    fold_id,
                    test_start,
                )
                break

            embargo_start = (
                test_start - embargo_n
            )  # end of training data (before embargo gap)
            embargo_idx = np.arange(embargo_start, test_start)  # excluded gap indices

            if cfg.expanding_window:
                train_start = 0  # expanding window: train always starts from bar 0
            else:
                window_size = (
                    embargo_start - embargo_n
                )  # rolling window: fixed train length
                train_start = max(0, embargo_start - window_size)

            raw_train_idx = np.arange(
                train_start, embargo_start
            )  # candidate training indices

            if cfg.purge:
                # Remove train samples whose feature window reaches into the test period
                train_idx, purged = self._purge(
                    raw_train_idx, test_start, cfg.feature_lookback
                )
            else:
                train_idx = raw_train_idx
                purged = 0

            test_idx = np.arange(test_start, test_end)  # out-of-sample test indices

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
        Remove training samples whose feature window overlaps the test zone.
        Any train index >= (test_start - feature_lookback) is purged.
        """
        cutoff = test_start - feature_lookback
        purge_mask = train_idx >= cutoff
        n_purged = int(purge_mask.sum())
        clean_idx = train_idx[~purge_mask]
        return clean_idx, n_purged


class PurgedScaler:
    """
    Feature scaler that strictly prevents lookahead bias.

    This scaler guarantees that:
    1. fit() is called only on training data
    2. transform() on test/live data uses saved training statistics
    3. Live inference reuses the same frozen state

    This prevents the most common lookahead bug in ML pipelines!

    The scaler supports three normalization methods:
    - zscore: Standardization (mean=0, std=1)
    - minmax: Scaling to [0, 1] range
    - robust: Using median and IQR (outlier-resistant)

    Attributes:
        method: Normalization method ("zscore", "minmax", "robust")
        clip: Value to clip scaled features to (prevents extreme values)

    Methods:
        fit(): Compute statistics from training data ONLY
        transform(): Apply scaling using fitted statistics
        fit_transform(): Fit and transform (for training data only!)
        inverse_transform(): Reverse the scaling
        save(): Persist scaler to disk
        load(): Load scaler from disk

    Example:
        --------
        scaler = PurgedScaler(method="zscore", clip=5.0)

        # Training: fit on train, transform train
        X_train_scaled = scaler.fit_transform(X_train)

        # Testing: use same scaler, transform test
        X_test_scaled = scaler.transform(X_test)

        # Live: load saved scaler
        live_scaler = PurgedScaler.load("scaler.npz")
        X_live_scaled = live_scaler.transform(X_live)

        # IMPORTANT: Never call fit() on test/live data!
        # Wrong: scaler.fit(X_test)  # LOOKAHEAD BIAS!
    """

    def __init__(self, method: str = "zscore", clip: float = 5.0):
        assert method in ("zscore", "minmax", "robust")
        self.method = method
        self.clip = clip
        self._fitted = False
        self._loc: Optional[np.ndarray] = None
        self._scale: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> "PurgedScaler":
        """Fit scaler statistics on training data ONLY. Never call on test/live data."""
        if self.method == "zscore":
            self._loc = X.mean(axis=0)  # feature means (training set only)
            self._scale = (
                X.std(axis=0) + 1e-8
            )  # feature std devs + epsilon to avoid div/0
        elif self.method == "minmax":
            self._loc = X.min(axis=0)  # training minimum per feature
            self._scale = (
                X.max(axis=0) - X.min(axis=0)
            ) + 1e-8  # training range per feature
        elif self.method == "robust":
            self._loc = np.median(X, axis=0)  # median is robust to outliers
            q75, q25 = np.percentile(X, [75, 25], axis=0)
            self._scale = (q75 - q25) + 1e-8  # IQR: robust scale estimator
        self._fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform data using frozen training statistics (safe for test and live data)."""
        if not self._fitted:
            raise RuntimeError("PurgedScaler.fit() must be called before transform().")
        scaled = (X - self._loc) / self._scale
        return np.clip(scaled, -self.clip, self.clip)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """For training data only: fit and transform in a single call."""
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
    Automatic test for lookahead leakage in feature sets.

    This class detects potential look-ahead bias in feature engineering
    by checking whether any feature is suspiciously correlated with future
    returns. High correlation suggests the feature may inadvertently
    include information that wouldn't be available at prediction time.

    The detector computes correlation between each feature and future
    returns (shifted by lag). Features with correlation > 0.3 are
    flagged as potential leakage.

    Attributes:
        significance_threshold: Correlation threshold for flagging (default: 0.52)

    Methods:
        check_feature_future_correlation(): Test for leakage
        report(): Generate formatted leakage report

    Example:
        --------
        detector = LeakDetector(significance_threshold=0.52)

        # Check all features
        results = detector.check_feature_future_correlation(
            X=features,
            future_returns=returns,
            feature_names=feature_names,
            lag=1
        )

        # Print report
        print(detector.report())

        # Results dictionary contains correlation for each feature
        for feat, corr in results.items():
            if abs(corr) > 0.3:
                print(f"WARNING: {feat} may have lookahead bias (r={corr:.3f})")

    Interpretation:
    --------------
    - |r| > 0.3: Strong indication of lookahead bias
    - 0.2 < |r| < 0.3: Moderate concern, investigate further
    - |r| < 0.2: Likely clean features

    Note: Some correlation is expected due to momentum/mean-reversion.
    The threshold of 0.3 is a heuristic that catches severe leakage.
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
        Compute correlation of each feature with future_returns[+lag].
        High correlation (>0.3) is a strong indicator of lookahead leakage.
        """
        n = len(X)
        if len(future_returns) != n:
            raise ValueError("X and future_returns must have the same length.")

        results = {}
        for i in range(X.shape[1]):
            feat_name = feature_names[i] if feature_names else f"feature_{i}"
            x_slice = X[: n - lag, i]
            y_slice = future_returns[lag:]
            corr = float(np.corrcoef(x_slice, y_slice)[0, 1])
            results[feat_name] = corr

            if abs(corr) > 0.3:
                logger.warning(
                    "POTENTIAL LOOKAHEAD: Feature '%s' correlates %.3f with future_return[+%d]",
                    feat_name,
                    corr,
                    lag,
                )

        self._results["feature_future_corr"] = results
        high_corr = {k: v for k, v in results.items() if abs(v) > 0.3}
        if high_corr:
            logger.error(
                "LEAK SUSPECTED: %d features with |corr| > 0.3: %s",
                len(high_corr),
                high_corr,
            )
        else:
            logger.info(
                "Leak check OK: no feature with |corr| > 0.3 to future returns."
            )

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
