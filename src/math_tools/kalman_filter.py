"""
Kalman Filter for Financial Time Series
========================================
Optimal state estimation for noisy price data.

Used for:
- Trend extraction from noisy prices
- Signal smoothing
- State estimation (true price vs. observed price)
- Adaptive parameter estimation
- Dynamic hedge ratio calculation

Mathematical Basis:
- State Space Model
- Optimal Recursive Estimation
- Minimum Mean Square Error
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass
from scipy.linalg import inv
import warnings

warnings.filterwarnings("ignore")


@dataclass
class KalmanFilterConfig:
    """Configuration for Kalman Filter."""

    # Process noise (how much the state can change)
    Q: float = 0.001  # Process variance

    # Measurement noise (observation uncertainty)
    R: float = 0.1  # Measurement variance

    # Initial state estimate
    x0: float = 0.0  # Initial state
    P0: float = 1.0  # Initial error covariance

    # Transition matrix (state evolution)
    A: float = 1.0  # State transition coefficient

    # Observation matrix (how we observe state)
    H: float = 1.0  # Observation coefficient


class KalmanFilter1D:
    """
    1-Dimensional Kalman Filter for price smoothing.

    Extracts the true underlying trend from noisy price observations.

    Mathematical Model:
        State equation: x_k = A * x_{k-1} + w_k
        Observation: z_k = H * x_k + v_k

    Where:
        x_k: True state at time k
        z_k: Observed price at time k
        w_k: Process noise ~ N(0, Q)
        v_k: Measurement noise ~ N(0, R)
    """

    def __init__(self, config: Optional[KalmanFilterConfig] = None):
        """Initialize Kalman Filter."""
        self.config = config or KalmanFilterConfig()

        # State estimate
        self.x = self.config.x0

        # Error covariance
        self.P = self.config.P0

        # History for analysis
        self.state_history = []
        self.prediction_history = []
        self.error_history = []
        self.kalman_gain_history = []

    def predict(self) -> float:
        """
        Prediction step.

        Returns:
            Predicted state
        """
        # Predict state: x_k|k-1 = A * x_{k-1}|k-1
        self.x = self.config.A * self.x

        # Predict covariance: P_k|k-1 = A * P_{k-1}|k-1 * A' + Q
        self.P = self.config.A * self.P * self.config.A + self.config.Q

        self.prediction_history.append(self.x)
        return self.x

    def update(self, measurement: float) -> float:
        """
        Update step with new observation.

        Args:
            measurement: Observed price

        Returns:
            Updated state estimate
        """
        # Predict first
        self.predict()

        # Innovation (measurement residual): y = z - H*x
        innovation = measurement - self.config.H * self.x

        # Innovation covariance: S = H*P*H' + R
        S = self.config.H * self.P * self.config.H + self.config.R

        # Kalman gain: K = P*H' / S
        K = self.P * self.config.H / S

        # Update state: x_k|k = x_k|k-1 + K*y
        self.x = self.x + K * innovation

        # Update covariance: P_k|k = (1 - K*H) * P_k|k-1
        self.P = (1 - K * self.config.H) * self.P

        # Store history
        self.state_history.append(self.x)
        self.error_history.append(self.P)
        self.kalman_gain_history.append(K)

        return self.x

    def filter_series(self, prices: np.ndarray) -> np.ndarray:
        """
        Apply Kalman filter to entire price series.

        Args:
            prices: Array of observed prices

        Returns:
            Array of filtered (smoothed) prices
        """
        filtered = []
        self.reset()

        for price in prices:
            smoothed = self.update(price)
            filtered.append(smoothed)

        return np.array(filtered)

    def get_trend_strength(self) -> float:
        """
        Get current trend strength based on Kalman gain.

        Returns:
            Trend strength (0-1, higher = stronger trend)
        """
        if not self.kalman_gain_history:
            return 0.5

        recent_gain = np.mean(self.kalman_gain_history[-10:])
        # Higher gain = more trust in measurements = stronger trend
        return min(1.0, recent_gain * 2)

    def reset(self):
        """Reset filter to initial state."""
        self.x = self.config.x0
        self.P = self.config.P0
        self.state_history = []
        self.prediction_history = []
        self.error_history = []
        self.kalman_gain_history = []


class KalmanFilterPairs:
    """
    Kalman Filter for pairs trading.

    Estimates dynamic hedge ratio and spread between two assets.

    Mathematical Model:
        Spread = Asset1 - beta * Asset2 - alpha

    Where beta (hedge ratio) and alpha are dynamically estimated.
    """

    def __init__(self, delta: float = 1e-4, R: float = 0.001):
        """
        Initialize Kalman Filter for pairs.

        Args:
            delta: Transition covariance (higher = more adaptive)
            R: Observation noise
        """
        # State vector: [alpha, beta]^T
        self.x = np.array([0.0, 1.0])

        # State covariance
        self.P = np.eye(2)

        # Transition matrix (identity for random walk)
        self.A = np.eye(2)

        # Process noise covariance
        self.Q = np.eye(2) * delta

        # Observation noise
        self.R = R

        # History
        self.alpha_history = []
        self.beta_history = []
        self.spread_history = []

    def update(
        self, asset1_price: float, asset2_price: float
    ) -> Tuple[float, float, float]:
        """
        Update with new prices.

        Args:
            asset1_price: Price of first asset
            asset2_price: Price of second asset

        Returns:
            (alpha, beta, spread)
        """
        # Observation matrix: H = [1, asset2_price]
        H = np.array([1.0, asset2_price])

        # Prediction
        x_pred = self.A @ self.x
        P_pred = self.A @ self.P @ self.A.T + self.Q

        # Innovation
        y = asset1_price - H @ x_pred

        # Innovation covariance
        S = H @ P_pred @ H.T + self.R

        # Kalman gain
        K = P_pred @ H.T / S

        # Update
        self.x = x_pred + K * y
        self.P = (np.eye(2) - np.outer(K, H)) @ P_pred

        alpha, beta = self.x
        spread = y  # Innovation is the spread

        # Store history
        self.alpha_history.append(alpha)
        self.beta_history.append(beta)
        self.spread_history.append(spread)

        return alpha, beta, spread

    def get_hedge_ratio(self) -> float:
        """Get current hedge ratio (beta)."""
        return self.x[1]

    def get_spread(self, asset1: float, asset2: float) -> float:
        """Calculate current spread."""
        alpha, beta = self.x
        return asset1 - beta * asset2 - alpha


class KalmanTrendDetector:
    """
    Advanced trend detection using Kalman Filter.

    Estimates position, velocity, and acceleration of price.
    """

    def __init__(self, process_noise: float = 0.001, measurement_noise: float = 0.1):
        """
        Initialize trend detector.

        Args:
            process_noise: Process noise covariance
            measurement_noise: Measurement noise covariance
        """
        # State: [position, velocity]^T
        self.x = np.array([0.0, 0.0])

        # State covariance
        self.P = np.eye(2)

        # Transition matrix (constant velocity model)
        self.A = np.array(
            [
                [1.0, 1.0],  # pos = pos + vel
                [0.0, 1.0],  # vel = vel
            ]
        )

        # Process noise
        self.Q = np.eye(2) * process_noise

        # Measurement matrix (observe position only)
        self.H = np.array([[1.0, 0.0]])

        # Measurement noise
        self.R = measurement_noise

        # History
        self.position_history = []
        self.velocity_history = []
        self.acceleration_history = []

    def update(self, price: float) -> Dict:
        """
        Update with new price.

        Args:
            price: Current price

        Returns:
            Dictionary with position, velocity, acceleration
        """
        # Prediction
        x_pred = self.A @ self.x
        P_pred = self.A @ self.P @ self.A.T + self.Q

        # Innovation
        y = price - self.H @ x_pred

        # Innovation covariance
        S = self.H @ P_pred @ self.H.T + self.R

        # Kalman gain
        K = P_pred @ self.H.T @ inv(np.array([[S]]))
        K = K.flatten()

        # Update
        self.x = x_pred + K * y
        self.P = (np.eye(2) - np.outer(K, self.H.flatten())) @ P_pred

        # Calculate acceleration (change in velocity)
        if len(self.velocity_history) > 0:
            acceleration = self.x[1] - self.velocity_history[-1]
        else:
            acceleration = 0.0

        # Store history
        self.position_history.append(self.x[0])
        self.velocity_history.append(self.x[1])
        self.acceleration_history.append(acceleration)

        return {
            "position": self.x[0],
            "velocity": self.x[1],
            "acceleration": acceleration,
            "trend_strength": abs(self.x[1]),
            "trend_direction": np.sign(self.x[1]),
        }

    def get_signal(self) -> int:
        """
        Generate trading signal based on velocity.

        Returns:
            -1: Sell (negative velocity)
             0: Neutral (velocity near zero)
             1: Buy (positive velocity)
        """
        if len(self.velocity_history) < 2:
            return 0

        velocity = self.x[1]

        if velocity > 0.01:
            return 1  # Buy
        elif velocity < -0.01:
            return -1  # Sell
        else:
            return 0  # Neutral


# Utility functions
def apply_kalman_smoothing(
    prices: np.ndarray, Q: float = 0.001, R: float = 0.1
) -> np.ndarray:
    """
    Convenience function to apply Kalman smoothing.

    Args:
        prices: Raw price series
        Q: Process noise
        R: Measurement noise

    Returns:
        Smoothed price series
    """
    config = KalmanFilterConfig(Q=Q, R=R)
    kf = KalmanFilter1D(config)
    return kf.filter_series(prices)


def detect_price_jumps(prices: np.ndarray, threshold: float = 3.0) -> np.ndarray:
    """
    Detect price jumps using Kalman Filter residuals.

    Args:
        prices: Price series
        threshold: Z-score threshold for jump detection

    Returns:
        Boolean array indicating jumps
    """
    config = KalmanFilterConfig(Q=0.001, R=0.1)
    kf = KalmanFilter1D(config)

    residuals = []
    for price in prices:
        pred = kf.predict()
        kf.update(price)
        residual = price - pred
        residuals.append(residual)

    residuals = np.array(residuals)
    z_scores = np.abs((residuals - np.mean(residuals)) / np.std(residuals))

    return z_scores > threshold


def calculate_dynamic_beta(
    asset1: pd.Series, asset2: pd.Series, delta: float = 1e-4
) -> pd.Series:
    """
    Calculate dynamic hedge ratio using Kalman Filter.

    Args:
        asset1: First asset prices
        asset2: Second asset prices
        delta: Adaptation rate

    Returns:
        Series of dynamic beta values
    """
    kf = KalmanFilterPairs(delta=delta)
    betas = []

    for p1, p2 in zip(asset1, asset2):
        _, beta, _ = kf.update(p1, p2)
        betas.append(beta)

    return pd.Series(betas, index=asset1.index)
