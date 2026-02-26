"""
Kalman Filter for Financial Time Series
=======================================
Optimal state estimation for noisy price data.

Mathematical Foundation:
------------------------
The Kalman Filter is an optimal recursive estimator for linear dynamic systems
with Gaussian noise. It provides the minimum mean square error (MMSE) estimate.

State-Space Model:
    State Equation:  x_k = A·x_{k-1} + w_k    (how state evolves)
    Observation:     z_k = H·x_k + v_k        (how we observe state)

    where:
        x_k: State vector at time k (hidden, we want to estimate)
        z_k: Observation at time k (what we measure)
        A: State transition matrix
        H: Observation matrix
        w_k: Process noise ~ N(0, Q)
        v_k: Measurement noise ~ N(0, R)

Kalman Filter Equations:
----------------------
Prediction Step:
    x̂_k|k-1 = A·x̂_{k-1|k-1}          # State prediction
    P_k|k-1 = A·P_{k-1|k-1·A' + Q      # Covariance prediction

Update Step:
    Innovation:     y_k = z_k - H·x̂_k|k-1      # Measurement residual
    Innovation Cov: S_k = H·P_k|k-1·H' + R       # Residual uncertainty
    Kalman Gain:    K_k = P_k|k-1·H'·S_k^{-1}    # Optimal weighting
    State Update:   x̂_k|k = x̂_k|k-1 + K_k·y_k   # State correction
    Cov Update:     P_k|k = (I - K_k·H)·P_k|k-1  # Cov correction

Interpretation:
    - K → 0: Trust prediction more (high R or low Q)
    - K → 1/H: Trust observation more (low R or high Q)

Properties:
    - Optimal for linear Gaussian systems
    - Recursive (computationally efficient)
    - Provides state estimate + uncertainty

Extended Kalman Filter (EKF):
    For nonlinear systems: linearize around current estimate

    x_k = f(x_{k-1}) + w_k
    z_k = h(x_k) + v_k

    where f() and h() can be nonlinear

Unscented Kalman Filter (UKF):
    More accurate for nonlinear systems
    Uses sigma points to propagate distribution

Used for:
- Trend extraction from noisy prices
- Signal smoothing
- State estimation (true price vs. observed price)
- Adaptive parameter estimation
- Dynamic hedge ratio calculation
- Pairs trading

References:
- Kalman, R.E. (1960) "A New Approach to Linear Filtering and Prediction Problems"
- Welch, G. & Bishop, G. (2006) "An Introduction to the Kalman Filter"
- Harvey, A.C. (1990) "Forecasting, Structural Time Series Models and the Kalman Filter"

Dependencies:
- numpy: Numerical computations
- pandas: Time series handling
- scipy.linalg: Linear algebra
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
    """
    Configuration for Kalman Filter.

    This dataclass holds the hyperparameters that control the Kalman filter's
    behavior. The choice of parameters determines the trade-off between
    responsiveness (adaptation speed) and smoothness (noise rejection).

    Parameters:
    ----------
    Q : float
        Process noise variance (default: 0.001).

        Represents uncertainty in how the state evolves.
        Higher Q → state can change more → more responsive/adaptive
        Lower Q → state is more stable → smoother output

        Interpretation for price filtering:
        - Q = 0.0001: Very stable, slow adaptation
        - Q = 0.001: Moderate (default)
        - Q = 0.01: Fast adaptation, follows prices closely

    R : float
        Measurement noise variance (default: 0.1).

        Represents uncertainty in observations.
        Higher R → observations are noisy → trust predictions more
        Lower R → observations are accurate → follow observations closely

        Interpretation:
        - R = 0.01: Very noisy observations
        - R = 0.1: Moderate (default)
        - R = 1.0: Clean observations, fast follow

    x0 : float
        Initial state estimate (default: 0.0).

        Starting point for the filter. For price data,
        typically set to first observed price.

    P0 : float
        Initial error covariance (default: 1.0).

        Uncertainty in initial state estimate.
        High P0 = uncertain initial state → filter learns quickly
        Low P0 = certain initial state → filter is slow to adapt

    A : float
        State transition coefficient (default: 1.0).

        For random walk: A = 1 (state persists)
        For mean reversion: A < 1 (state reverts to baseline)
        For trending: A > 1 (state extrapolates)

    H : float
        Observation matrix coefficient (default: 1.0).

        Relates state to observation.
        For direct price observation: H = 1

    Example:
    -------
    >>> # Smooth price with moderate filtering
    >>> config = KalmanFilterConfig(Q=0.001, R=0.1)
    >>>
    >>> # Fast adaptation (for volatile markets)
    >>> config = KalmanFilterConfig(Q=0.01, R=0.5)
    >>>
    >>> # Slow, smooth output (for stable markets)
    >>> config = KalmanFilterConfig(Q=0.0001, R=0.01)
    """

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

    Mathematical Background:
    -----------------------
    State-Space Model (1D):
        State:      x_k = A·x_{k-1} + w_k
        Observation: z_k = H·x_k + v_k

    For price smoothing:
        x_k: True (unobserved) price at time k
        z_k: Observed (noisy) price at time k
        A = 1: Random walk model for true price
        H = 1: We observe the true price with noise

    Kalman Gain Interpretation:
        K = P / (P + R)

        where:
            P: Predicted state variance (uncertainty in estimate)
            R: Measurement noise variance (noise in observations)

        - High K (≈1): Trust observations, fast adaptation
        - Low K (≈0): Trust prediction, slow adaptation

    Filtering vs Smoothing:
        - Filtering: Estimate x_k using observations up to time k
        - Smoothing: Estimate x_k using all observations (offline)

    Parameters:
    -----------
    config : KalmanFilterConfig, optional
        Configuration object with Q, R, x0, P0, A, H parameters.
        If None, uses default configuration.

    KalmanFilterConfig Parameters:
    ------------------------------
    Q : float
        Process noise variance (default: 0.001).
        Higher = state can change more = more adaptive.
        Typical range: 0.0001 - 0.01

    R : float
        Measurement noise variance (default: 0.1).
        Higher = observations are noisy = smoother output.
        Typical range: 0.01 - 1.0

    x0 : float
        Initial state estimate (default: 0.0)

    P0 : float
        Initial error covariance (default: 1.0)
        Uncertainty in initial state

    A : float
        State transition coefficient (default: 1.0)
        For random walk: A = 1

    H : float
        Observation coefficient (default: 1.0)
        For direct observation: H = 1

    Attributes:
    ----------
    x : float
        Current state estimate

    P : float
        Current error covariance

    state_history : list
        Historical state estimates

    prediction_history : list
        Historical predictions

    error_history : list
        Historical error covariances

    kalman_gain_history : list
        Historical Kalman gains

    Methods:
    --------
    predict()
        Prediction step (propagate state forward)

    update(measurement)
        Update step with new observation

    filter_series(prices)
        Apply filter to entire price series

    get_trend_strength()
        Calculate trend strength from Kalman gain

    reset()
        Reset filter to initial state

    Returns:
    --------
    Float: Filtered/smoothed state estimate

    Example:
    -------
    >>> import numpy as np
    >>>
    >>> # Generate noisy signal
    >>> np.random.seed(42)
    >>> true = np.cumsum(np.random.randn(100) * 0.1)
    >>> observed = true + np.random.randn(100) * 0.5
    >>>
    >>> # Apply Kalman filter
    >>> kf = KalmanFilter1D()
    >>> filtered = kf.filter_series(observed)
    >>>
    >>> # Get trend strength
    >>> strength = kf.get_trend_strength()
    >>> print(f"Trend strength: {strength:.3f}")
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
        Prediction step (Time Update).

        Propagates the state estimate forward in time using the state
        transition model. Also propagates the uncertainty (covariance).

        Mathematical Formulation:
        ------------------------
        State Prediction:
            x̂_k|k-1 = A × x̂_{k-1|k-1}

            where:
                x̂_k|k-1: Predicted state (at time k given observations up to k-1)
                A: State transition matrix
                x̂_{k-1|k-1}: Previous updated state estimate

        For 1D case with A=1: x̂_k|k-1 = x̂_{k-1}

        Covariance Prediction:
            P_k|k-1 = A × P_{k-1|k-1} × A' + Q

            where:
                P_k|k-1: Predicted error covariance
                P_{k-1|k-1}: Previous error covariance
                Q: Process noise covariance

        The +Q term adds uncertainty because the state evolves with noise.

        Returns:
            Predicted state value
        """
        # Predict state: x_k|k-1 = A * x_{k-1|k-1}
        # For random walk (A=1), this just carries forward the previous estimate
        self.x = self.config.A * self.x

        # Predict covariance: P_k|k-1 = A * P_{k-1|k-1} * A' + Q
        # The +Q accounts for process noise - uncertainty in state evolution
        # As time passes, uncertainty grows unless corrected by observations
        self.P = self.config.A * self.P * self.config.A + self.config.Q

        self.prediction_history.append(self.x)
        return self.x

    def update(self, measurement: float) -> float:
        """
        Update step (Measurement Update).

        Incorporates a new observation to improve the state estimate.
        This is where the "filtering" happens - combining prediction
        with observation optimally.

        Mathematical Formulation:
        ------------------------
        Innovation (Measurement Residual):
            y_k = z_k - H × x̂_k|k-1

            where:
                y_k: Innovation (actual - predicted observation)
                z_k: Actual observation (measurement)
                H: Observation matrix
                x̂_k|k-1: Predicted state

        The innovation tells us how wrong our prediction was.

        Innovation Covariance:
            S_k = H × P_k|k-1 × H' + R

            where:
                S_k: Total uncertainty in the measurement space
                R: Measurement noise variance

        This combines prediction uncertainty with observation noise.

        Kalman Gain (Optimal Weighting):
            K_k = P_k|k-1 × H' / S_k

            where:
                K_k: How much to weight the innovation

        Properties:
            - K → 0: Trust prediction more (high R or low P)
            - K → 1/H: Trust observation more (low R or high P)

        State Update:
            x̂_k|k = x̂_k|k-1 + K_k × y_k

            New estimate = Old prediction + (gain × innovation)

        Covariance Update:
            P_k|k = (I - K_k × H) × P_k|k-1

            After observation, uncertainty decreases (unless K=0)

        Args:
            measurement: Observed value z_k

        Returns:
            Updated state estimate x̂_k|k
        """
        # First, get the prediction (must predict before update)
        self.predict()

        # Innovation (measurement residual): y = z - H*x
        # This is how far off our prediction was from the actual observation
        innovation = measurement - self.config.H * self.x

        # Innovation covariance: S = H*P*H' + R
        # Total uncertainty = prediction uncertainty + measurement noise
        S = self.config.H * self.P * self.config.H + self.config.R

        # Kalman gain: K = P*H' / S
        # Optimal weighting between prediction and observation
        # Higher gain = trust observation more (adapt faster)
        K = self.P * self.config.H / S

        # Update state: x = x + K*y
        # Pull the state estimate toward the observation
        self.x = self.x + K * innovation

        # Update covariance: P = (1 - K*H)*P
        # Uncertainty decreases after incorporating observation
        # Using Joseph form: (I-KH)P(I-KH)' + KRK' for numerical stability
        self.P = (1 - K * self.config.H) * self.P

        # Store history for analysis
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
    This is the key innovation for pairs trading - hedge ratios that
    adapt to changing market conditions.

    Mathematical Background:
    -----------------------
    State-Space Model for Pairs:

        State: x_k = [α_k, β_k]ᵀ
               α_k: Intercept (spread adjustment)
               β_k: Hedge ratio

        State evolves as random walk:
            x_k = x_{k-1} + w_k,  w_k ~ N(0, Q)

        Observation:
            asset1_k = α_k + β_k × asset2_k + v_k
            z_k     = H_k × x_k + v_k

            where H_k = [1, asset2_k]

    The innovation (residual):
        y_k = asset1_k - (α_{k-1} + β_{k-1} × asset2_k)
            = Spread - Expected Spread

    This innovation is the trading signal:
        - Large positive: Asset1 overvalued relative to Asset2
        - Large negative: Asset1 undervalued relative to Asset2

    Advantages over Static Regression:
    ---------------------------------
    1. Adaptive: Hedge ratio adjusts to regime changes
    2. Smooth: Doesn't jump wildly with each new data point
    3. Probabilistic: Provides uncertainty estimates
    4. Real-time: Can update with streaming prices

    Parameters:
    -----------
    delta : float
        Transition covariance (default: 1e-4).

        Controls how fast the hedge ratio can change.
        Higher = more adaptive = faster adjustment
        Lower = more stable = slower adjustment

        Typical range: 1e-5 to 1e-3

    R : float
        Observation noise (default: 0.001).

        Measurement uncertainty in the spread.
        Higher = more noise in observations = smoother hedge ratio
        Lower = observations are accurate = faster adaptation

    Attributes:
    ----------
    x : np.ndarray
        Current state vector [alpha, beta]

    P : np.ndarray
        State covariance matrix (2x2)

    A : np.ndarray
        Transition matrix (2x2 identity for random walk)

    Q : np.ndarray
        Process noise covariance

    R : float
        Observation noise variance

    alpha_history : list
        Historical intercept values

    beta_history : list
        Historical hedge ratio values

    spread_history : list
        Historical spread values

    Methods:
    --------
    update(asset1_price, asset2_price)
        Update hedge ratio with new prices

    get_hedge_ratio()
        Get current hedge ratio (beta)

    get_spread(asset1, asset2)
        Calculate current spread

    Returns:
    --------
    Tuple (alpha, beta, spread):
        alpha: Intercept
        beta: Dynamic hedge ratio
        spread: Current spread (innovation)

    Example:
    -------
    >>> import numpy as np
    >>>
    >>> # Simulate cointegrated pair
    >>> np.random.seed(42)
    >>> asset2 = np.cumsum(np.random.randn(100))
    >>> asset1 = 2 * asset1 + 1 + np.cumsum(np.random.randn(100) * 0.1)
    >>>
    >>> # Track with Kalman filter
    >>> kf = KalmanFilterPairs(delta=1e-4, R=0.001)
    >>>
    >>> for p1, p2 in zip(asset1, asset2):
    ...     alpha, beta, spread = kf.update(p1, p2)
    >>>
    >>> print(f"Final hedge ratio: {kf.get_hedge_ratio():.4f}")
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
        # Observation matrix: H = [1, asset2_price]  (observes alpha + beta*asset2 = asset1)
        H = np.array([1.0, asset2_price])

        # Prediction step: propagate state and covariance forward
        x_pred = (
            self.A @ self.x
        )  # State prediction: [alpha, beta] unchanged (random walk)
        P_pred = (
            self.A @ self.P @ self.A.T + self.Q
        )  # Covariance grows by process noise

        # Innovation: how much asset1 deviates from predicted value
        y = asset1_price - H @ x_pred

        # Innovation covariance: uncertainty in the observation
        S = H @ P_pred @ H.T + self.R

        # Kalman gain: optimal weight for the innovation correction
        K = P_pred @ H.T / S

        # Update state and covariance with new observation
        self.x = x_pred + K * y
        self.P = (
            np.eye(2) - np.outer(K, H)
        ) @ P_pred  # Joseph form for numerical stability

        alpha, beta = self.x
        spread = (
            y  # Innovation is the residual spread (stationary signal for pairs trading)
        )

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
    This implements a constant-velocity model of price motion.

    Mathematical Background:
    -----------------------
    State Vector:
        x_k = [position_k, velocity_k]ᵀ

    State Equation (Constant Velocity Model):
        position_k = position_{k-1} + velocity_{k-1} × dt
        velocity_k = velocity_{k-1}

        Matrix form:
        x_k = A × x_{k-1} + w_k

        A = |1  1|
            |0  1|

        This means: position changes by velocity each step

    Physical Interpretation:
    -----------------------
    Position: The "true" price (filtered/smoothed)
    Velocity: Rate of price change (trend direction + strength)
              - Positive velocity = uptrend
              - Negative velocity = downtrend
              - Zero velocity = ranging
    Acceleration: Change in velocity (trend acceleration)

    Signal Generation:
    -----------------
    Velocity > threshold → Buy signal
    Velocity < -threshold → Sell signal
    |Velocity| ≈ 0 → Neutral

    Advantages over Moving Averages:
    --------------------------------
    1. Adaptive: Speed of adaptation varies with uncertainty
    2. Smooth: Less whipsaw than simple moving averages
    3. Continuous: Provides gradient (velocity), not just levels
    4. Responsive: Adjusts quickly to trend changes

    Parameters:
    -----------
    process_noise : float
        Process noise covariance (default: 0.001).

        Uncertainty in the constant-velocity model.
        Higher = price can change direction more easily
        Lower = price maintains momentum longer

    measurement_noise : float
        Measurement noise covariance (default: 0.1).

        Noise in observed prices.
        Higher = smoother output, slower to react
        Lower = faster reaction, more noise

    Attributes:
    ----------
    x : np.ndarray
        Current state [position, velocity]

    P : np.ndarray
        State covariance matrix

    A : np.ndarray
        Transition matrix (constant velocity model)

    Q : np.ndarray
        Process noise covariance

    H : np.ndarray
        Observation matrix (observe position only)

    R : float
        Measurement noise

    position_history : list
        Historical filtered positions

    velocity_history : list
        Historical velocities (trends)

    acceleration_history : list
        Historical accelerations

    Methods:
    --------
    update(price)
        Update with new price observation

    get_signal()
        Generate trading signal based on velocity

    Returns:
    --------
    Dictionary with:
        - position: Filtered price
        - velocity: Trend speed
        - acceleration: Trend acceleration
        - trend_strength: Absolute velocity
        - trend_direction: Sign of velocity

    Example:
    -------
    >>> import numpy as np
    >>>
    >>> # Generate trending price
    >>> np.random.seed(42)
    >>> trend = np.cumsum(np.random.randn(100) * 0.1) + np.linspace(0, 10, 100)
    >>>
    >>> # Detect trend
    >>> detector = KalmanTrendDetector()
    >>>
    >>> for price in trend:
    ...     state = detector.update(price)
    >>>
    >>> signal = detector.get_signal()
    >>> print(f"Signal: {signal} (1=buy, -1=sell, 0=neutral)")
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
