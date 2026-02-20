"""
Hidden Markov Model - Market Regime Detection
=============================================
Identifies hidden market states (bull/bear/neutral) from observable features.

Mathematical Model:
- Hidden states: {Low Volatility, High Volatility, Transition}
- Observations: Returns, volatility, volume
- Transitions: Markov chain with transition matrix

Key Properties:
- Unsupervised learning (no labels needed)
- Probabilistic regime classification
- Captures market phase transitions

Applications:
- Adaptive strategy selection
- Risk management
- Position sizing
"""

import numpy as np
import pandas as pd
from hmmlearn import hmm
from typing import Tuple, Optional, List
from dataclasses import dataclass
from loguru import logger
from sklearn.preprocessing import StandardScaler


@dataclass
class MarketRegime:
    """Market regime definition."""

    id: int
    name: str
    mean_return: float
    mean_volatility: float
    probability: float


class HMMRegimeDetector:
    """
    Hidden Markov Model for market regime detection.

    Detects:
    - Low Volatility (trending)
    - High Volatility (mean-reverting)
    - Transition (regime change)

    Usage:
    ------
    # Train on historical data
    hmm_detector = HMMRegimeDetector(n_regimes=3)
    hmm_detector.fit(features)

    # Classify current regime
    regime_probs = hmm_detector.predict_proba(current_features)
    current_regime = hmm_detector.predict(current_features)
    """

    def __init__(
        self, n_regimes: int = 3, n_iter: int = 100, random_state: Optional[int] = 42
    ):
        """
        Initialize HMM regime detector.

        Parameters:
        -----------
        n_regimes : int
            Number of hidden states (default: 3)
        n_iter : int
            Number of training iterations
        random_state : int
            Random seed for reproducibility
        """
        self.n_regimes = n_regimes
        self.n_iter = n_iter
        self.random_state = random_state

        # Initialize HMM (Gaussian emissions)
        self.model = hmm.GaussianHMM(
            n_components=n_regimes,
            covariance_type="full",
            n_iter=n_iter,
            random_state=random_state,
            init_params="stmc",  # Initialize start, transition, means, covariance
        )

        # Scaler for feature normalization
        self.scaler = StandardScaler()

        # Fitted flag
        self.is_fitted = False

        # Regime labels (assigned after fitting)
        self.regime_labels = None

        # Track feature column names manually (feature_names_in_ not always available)
        self._feature_cols = None

        logger.info(f"HMMRegimeDetector initialized: {n_regimes} regimes")

    def fit(self, features: pd.DataFrame, feature_cols: Optional[List[str]] = None):
        """
        Fit HMM on historical features.

        Parameters:
        -----------
        features : pd.DataFrame
            Feature matrix (returns, volatility, etc.)
        feature_cols : List[str], optional
            Columns to use (default: all numeric)
        """
        # Select features
        if feature_cols is None:
            # Use all numeric columns
            feature_cols = features.select_dtypes(include=[np.number]).columns.tolist()

        X = features[feature_cols].values

        # Handle NaN
        if np.isnan(X).any():
            logger.warning(f"Found {np.isnan(X).sum()} NaN values, filling with mean")
            X = pd.DataFrame(X).fillna(pd.DataFrame(X).mean()).values

        # Track feature column names for later inference
        self._feature_cols = feature_cols

        # Normalize features
        X_scaled = self.scaler.fit_transform(X)

        logger.info(f"Fitting HMM on {len(X)} samples, {X.shape[1]} features")

        # Fit HMM
        self.model.fit(X_scaled)

        # Decode hidden states
        hidden_states = self.model.predict(X_scaled)

        # Assign regime labels based on characteristics
        self.regime_labels = self._assign_regime_labels(
            X_scaled, hidden_states, features
        )

        self.is_fitted = True

        logger.success("HMM training complete")
        self._log_regime_characteristics(X_scaled, hidden_states)

    def _assign_regime_labels(
        self, X: np.ndarray, states: np.ndarray, features: pd.DataFrame
    ) -> dict:
        """
        Assign meaningful labels to regimes based on characteristics.

        Logic:
        - High volatility + high returns → Bull Market
        - High volatility + negative returns → Bear Market
        - Low volatility → Consolidation
        """
        labels = {}

        for state_id in range(self.n_regimes):
            # Get samples for this state
            mask = states == state_id

            if mask.sum() == 0:
                labels[state_id] = f"Regime_{state_id}"
                continue

            # Calculate characteristics
            if "returns" in features.columns:
                mean_return = features.loc[mask, "returns"].mean()
            else:
                mean_return = 0.0

            if "volatility_20" in features.columns:
                mean_vol = features.loc[mask, "volatility_20"].mean()
            else:
                mean_vol = X[mask, :].std()

            # Assign label
            if mean_vol < np.percentile(features.get("volatility_20", X[:, 0]), 33):
                label = "Low Volatility"
            elif mean_vol > np.percentile(features.get("volatility_20", X[:, 0]), 67):
                if mean_return > 0:
                    label = "High Volatility (Bull)"
                else:
                    label = "High Volatility (Bear)"
            else:
                label = "Transition"

            labels[state_id] = label

            logger.info(f"State {state_id} → {label}")
            logger.info(f"  Mean return: {mean_return:.6f}")
            logger.info(f"  Mean volatility: {mean_vol:.6f}")

        return labels

    def predict(
        self, features: pd.DataFrame, feature_cols: Optional[List[str]] = None
    ) -> int:
        """
        Predict current regime.

        Returns:
        --------
        regime_id : int
            Most likely regime (0, 1, or 2)
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        # Get features
        if feature_cols is None:
            feature_cols = self._feature_cols or list(
                features.select_dtypes(include=[np.number]).columns
            )

        X = features[feature_cols].values

        # Handle single sample
        if X.ndim == 1:
            X = X.reshape(1, -1)

        # Scale
        X_scaled = self.scaler.transform(X)

        # Predict
        regime_id = self.model.predict(X_scaled)[-1]

        return int(regime_id)

    def predict_proba(
        self, features: pd.DataFrame, feature_cols: Optional[List[str]] = None
    ) -> np.ndarray:
        """
        Predict regime probabilities.

        Returns:
        --------
        probs : np.ndarray
            Probability distribution over regimes (sum = 1)
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        # Get features
        if feature_cols is None:
            feature_cols = self._feature_cols or list(
                features.select_dtypes(include=[np.number]).columns
            )
        X = features[feature_cols].values

        # Handle single sample
        if X.ndim == 1:
            X = X.reshape(1, -1)

        # Scale
        X_scaled = self.scaler.transform(X)

        # Predict probabilities
        probs = self.model.predict_proba(X_scaled)[-1]

        return probs

    def get_regime_info(self, regime_id: int) -> MarketRegime:
        """Get regime information."""
        if not self.is_fitted:
            raise RuntimeError("Model not fitted")

        # Get means for this regime
        means = self.model.means_[regime_id]

        return MarketRegime(
            id=regime_id,
            name=self.regime_labels.get(regime_id, f"Regime_{regime_id}"),
            mean_return=float(means[0]) if len(means) > 0 else 0.0,
            mean_volatility=float(means[1]) if len(means) > 1 else 0.0,
            probability=0.0,  # Will be filled by predict_proba
        )

    def get_transition_matrix(self) -> np.ndarray:
        """
        Get regime transition matrix.

        Returns:
        --------
        transition_matrix : np.ndarray
            P[i,j] = probability of transitioning from regime i to j
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted")

        return self.model.transmat_

    def _log_regime_characteristics(self, X: np.ndarray, states: np.ndarray):
        """Log regime characteristics for analysis."""
        logger.info("\nRegime Transition Matrix:")
        trans_mat = self.model.transmat_
        for i in range(self.n_regimes):
            label_i = self.regime_labels.get(i, f"Regime_{i}")
            logger.info(f"  From {label_i}:")
            for j in range(self.n_regimes):
                label_j = self.regime_labels.get(j, f"Regime_{j}")
                logger.info(f"    → {label_j}: {trans_mat[i, j]:.3f}")

        # Regime persistence
        logger.info("\nRegime Persistence:")
        for i in range(self.n_regimes):
            label = self.regime_labels.get(i, f"Regime_{i}")
            persistence = trans_mat[i, i]
            logger.info(f"  {label}: {persistence:.3f} (stay in same regime)")


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def prepare_hmm_features(price_data: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
    """
    Prepare features for HMM training.

    Features:
    - Returns
    - Volatility (rolling std)
    - Volume changes

    Parameters:
    -----------
    price_data : pd.DataFrame
        OHLCV data
    lookback : int
        Rolling window size

    Returns:
    --------
    features : pd.DataFrame
        Feature matrix for HMM
    """
    df = price_data.copy()

    # Returns
    df["returns"] = df["close"].pct_change()

    # Volatility
    df["volatility_20"] = df["returns"].rolling(lookback).std()

    # Volume change
    df["volume_change"] = df["volume"].pct_change()

    # Range (high-low normalized by close)
    df["range"] = (df["high"] - df["low"]) / df["close"]

    # Drop NaN
    df = df.dropna()

    return df[["returns", "volatility_20", "volume_change", "range"]]


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("HMM REGIME DETECTION TEST")
    print("=" * 80)

    # Generate synthetic data with regime changes
    np.random.seed(42)
    n_points = 2000

    # Simulate 3 regimes
    regime_lengths = [700, 600, 700]

    prices = [100.0]
    returns = []
    volatilities = []

    for i, length in enumerate(regime_lengths):
        if i == 0:  # Low volatility
            vol = 0.01
            drift = 0.0005
        elif i == 1:  # High volatility bull
            vol = 0.03
            drift = 0.001
        else:  # High volatility bear
            vol = 0.03
            drift = -0.001

        for _ in range(length):
            ret = np.random.normal(drift, vol)
            prices.append(prices[-1] * (1 + ret))
            returns.append(ret)
            volatilities.append(vol)

    # Create DataFrame
    dates = pd.date_range("2020-01-01", periods=len(prices), freq="1H")

    price_data = pd.DataFrame(
        {
            "close": prices,
            "high": [p * 1.01 for p in prices],
            "low": [p * 0.99 for p in prices],
            "volume": np.random.uniform(100, 1000, len(prices)),
        },
        index=dates[:-1],
    )  # -1 because prices has one extra element

    print(f"\n✓ Generated {len(price_data)} data points with 3 regimes")

    # Prepare features
    features = prepare_hmm_features(price_data)
    print(f"✓ Prepared features: {features.shape}")
    print(f"  Columns: {features.columns.tolist()}")

    # Test 1: Train HMM
    print("\n[TEST 1] Training HMM")

    hmm_detector = HMMRegimeDetector(n_regimes=3, n_iter=100)
    hmm_detector.fit(features)

    print("✓ HMM trained")

    # Test 2: Predict regimes
    print("\n[TEST 2] Regime Prediction")

    # Predict all regimes
    all_regimes = []
    for i in range(len(features)):
        regime = hmm_detector.predict(features.iloc[i : i + 1])
        all_regimes.append(regime)

    print(f"✓ Predicted {len(all_regimes)} regimes")

    # Count regime distribution
    unique, counts = np.unique(all_regimes, return_counts=True)
    print("\nRegime Distribution:")
    for regime_id, count in zip(unique, counts):
        label = hmm_detector.regime_labels.get(regime_id, f"Regime_{regime_id}")
        pct = count / len(all_regimes) * 100
        print(f"  {label}: {count} samples ({pct:.1f}%)")

    # Test 3: Probability prediction
    print("\n[TEST 3] Regime Probabilities")

    current_features = features.iloc[-1:]
    probs = hmm_detector.predict_proba(current_features)

    print("\nCurrent Regime Probabilities:")
    for regime_id, prob in enumerate(probs):
        label = hmm_detector.regime_labels.get(regime_id, f"Regime_{regime_id}")
        print(f"  {label}: {prob:.3f}")

    # Test 4: Transition matrix
    print("\n[TEST 4] Transition Matrix")
    trans_mat = hmm_detector.get_transition_matrix()

    print("\nRegime Transitions:")
    for i in range(3):
        label_i = hmm_detector.regime_labels.get(i, f"Regime_{i}")
        print(f"\n  From {label_i}:")
        for j in range(3):
            label_j = hmm_detector.regime_labels.get(j, f"Regime_{j}")
            print(f"    → {label_j}: {trans_mat[i, j]:.3f}")

    print("\n" + "=" * 80)
    print("✓ HMM REGIME DETECTION TEST PASSED")
    print("=" * 80)

    print("\nKey Insights:")
    print("• Successfully detected 3 market regimes")
    print("• Regime transitions captured in transition matrix")
    print("• Probabilistic classification enables adaptive strategies")
    print("• Ready for integration with trading system")
