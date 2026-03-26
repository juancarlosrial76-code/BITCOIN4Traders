from typing import Optional

import numpy as np


class VPINCalculator:
    """
    Computes Volume-Synchronized Probability of Informed Trading (VPIN).
    Used as an early warning signal for market toxicity and impending
    high-volatility events.

    Reference: Easley, De Prado, O'Hara (2012)
    """

    def __init__(self, volume_bucket_size: float = 1000.0, sample_length: int = 50):
        self.volume_bucket_size = volume_bucket_size
        self.sample_length = sample_length
        self.price_history: list[float] = []
        self.volume_history: list[float] = []
        self.buy_volume_buckets: list[float] = []
        self.sell_volume_buckets: list[float] = []
        self.vpin_history: list[float] = []

    # ------------------------------------------------------------------
    # Bulk Volume Classification
    # ------------------------------------------------------------------

    def classify_trade(self, price: float, prev_price: float) -> bool:
        """
        Bulk Volume Classification (BVC) heuristic.

        Returns True (buy-initiated) if the price moved up relative to the
        previous price, False (sell-initiated) otherwise.  Equal prices are
        classified as sell-initiated (conservative / tie-breaking rule).
        """
        return price > prev_price

    # ------------------------------------------------------------------
    # Core update
    # ------------------------------------------------------------------

    def update(
        self,
        price: float,
        volume: float,
        is_buy_initiated: Optional[bool] = None,
    ) -> float:
        """
        Processes a new trade/tick and returns the updated VPIN metric.

        Parameters
        ----------
        price : float
            Trade price.
        volume : float
            Trade volume.
        is_buy_initiated : bool, optional
            Explicit trade direction flag.  When omitted, BVC is applied
            automatically using the last known price.
        """
        # Resolve direction via BVC when not supplied explicitly
        if is_buy_initiated is None:
            if self.price_history:
                is_buy_initiated = self.classify_trade(price, self.price_history[-1])
            else:
                # No prior price available – default to buy
                is_buy_initiated = True

        self.price_history.append(price)
        self.volume_history.append(volume)

        # Accumulate volume into directional buckets
        if is_buy_initiated:
            self.buy_volume_buckets.append(volume)
            self.sell_volume_buckets.append(0.0)
        else:
            self.buy_volume_buckets.append(0.0)
            self.sell_volume_buckets.append(volume)

        # Enforce rolling window
        if len(self.buy_volume_buckets) > self.sample_length:
            self.buy_volume_buckets.pop(0)
            self.sell_volume_buckets.pop(0)
            self.price_history.pop(0)
            self.volume_history.pop(0)

        return self.compute_vpin()

    # ------------------------------------------------------------------
    # VPIN computation
    # ------------------------------------------------------------------

    def compute_vpin(self) -> float:
        """Calculates VPIN = |V_buy - V_sell| / V_total in the current window."""
        if len(self.buy_volume_buckets) < self.sample_length // 2:
            return 0.0  # Not enough data yet

        sum_buy_vol = sum(self.buy_volume_buckets)
        sum_sell_vol = sum(self.sell_volume_buckets)
        total_vol = sum_buy_vol + sum_sell_vol

        if total_vol == 0:
            return 0.0

        vpin = abs(sum_buy_vol - sum_sell_vol) / total_vol
        self.vpin_history.append(vpin)
        return vpin

    # ------------------------------------------------------------------
    # Toxicity check
    # ------------------------------------------------------------------

    def is_market_toxic(self, threshold: float = 0.8) -> bool:
        """Returns True if the most recent VPIN exceeds *threshold*."""
        if not self.vpin_history:
            return False
        return self.vpin_history[-1] > threshold

    # ------------------------------------------------------------------
    # History accessor
    # ------------------------------------------------------------------

    def get_vpin_series(self) -> np.ndarray:
        """Returns the full VPIN history as a NumPy array."""
        return np.array(self.vpin_history, dtype=float)
