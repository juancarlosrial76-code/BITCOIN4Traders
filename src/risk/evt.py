from typing import Dict

import numpy as np
import scipy.stats as stats
from loguru import logger


class EVTRiskManager:
    """
    Extreme Value Theory (EVT) Risk Manager.

    Uses the Peak-Over-Threshold (POT) method with a Generalized Pareto
    Distribution (GPD, location fixed at 0) to estimate the probability
    and magnitude of extreme tail events in a streaming return series.
    """

    def __init__(
        self,
        history_window: int = 500,
        threshold_quantile: float = 0.95,
    ) -> None:
        self.history_window: int = history_window
        self.threshold_quantile: float = threshold_quantile
        self.returns_history: list[float] = []

    # ------------------------------------------------------------------
    # Data ingestion
    # ------------------------------------------------------------------

    def add_return(self, ret: float) -> None:
        """Adds a new return observation and enforces the rolling window."""
        self.returns_history.append(ret)
        if len(self.returns_history) > self.history_window:
            self.returns_history.pop(0)

    # ------------------------------------------------------------------
    # Convenience wrapper
    # ------------------------------------------------------------------

    def update_and_check(self, ret: float) -> bool:
        """
        Adds *ret* to history and returns True when the market is in a
        critical tail-risk regime (``is_critical`` flag from
        :meth:`compute_evt_risk_metrics`).
        """
        self.add_return(ret)
        metrics = self.compute_evt_risk_metrics()
        return bool(metrics.get("is_critical", False))

    # ------------------------------------------------------------------
    # Risk metrics
    # ------------------------------------------------------------------

    def compute_evt_risk_metrics(self) -> Dict[str, float]:
        """
        Fits GPD to the negative tail (losses) and estimates:

        - **VaR_99** – Value at Risk at 99 % confidence
        - **ES_99**  – Expected Shortfall (CVaR) at 99 % confidence
        - **shape_param** – GPD shape parameter ξ (only on successful fit)
        - **is_critical** – True when ES_99 > 5 % single-step loss

        Falls back to empirical quantiles when the GPD fit fails or when
        there are insufficient data points.
        """
        if len(self.returns_history) < 100:
            return {"VaR_99": 0.0, "ES_99": 0.0, "is_critical": False}

        returns = np.array(self.returns_history)

        # Work on the loss distribution (right tail of negated returns)
        losses = -returns
        losses = losses[losses > 0]  # Keep only negative return days

        if len(losses) < 20:
            return {"VaR_99": 0.0, "ES_99": 0.0, "is_critical": False}

        threshold = np.quantile(losses, self.threshold_quantile)
        exceedances = losses[losses > threshold] - threshold

        if len(exceedances) < 5:
            # Not enough tail data – fall back to historical VaR / ES
            var_99 = float(np.quantile(losses, 0.99))
            tail = losses[losses >= var_99]
            es_99 = float(np.mean(tail)) if len(tail) > 0 else var_99
            return {"VaR_99": var_99, "ES_99": es_99, "is_critical": False}

        try:
            # Fit GPD with location fixed at 0 (standard POT formulation)
            shape, loc, scale = stats.genpareto.fit(exceedances, floc=0)

            n = len(losses)
            n_u = len(exceedances)
            p = 0.01  # Complement of 99 % confidence level

            # EVT VaR formula (Pickands–Balkema–de Haan)
            if shape != 0:
                var_99 = threshold + (scale / shape) * (((n / n_u) * p) ** (-shape) - 1)
            else:
                # Exponential tail (shape → 0)
                var_99 = threshold - scale * np.log((n / n_u) * p)

            # McNeil & Frey Expected Shortfall formula
            es_99 = (var_99 + scale - shape * threshold) / (1 - shape)

            is_critical: bool = es_99 > 0.05  # 5 % single-step loss threshold

            return {
                "VaR_99": float(var_99),
                "ES_99": float(es_99),
                "shape_param": float(shape),
                "is_critical": is_critical,
            }

        except Exception as e:
            logger.warning(
                f"EVT GPD fit failed: {e}. Falling back to empirical quantiles.",
                exc_info=True,
            )
            var_99 = float(np.quantile(losses, 0.99))
            tail = losses[losses >= var_99]
            es_99 = float(np.mean(tail)) if len(tail) > 0 else var_99
            return {"VaR_99": var_99, "ES_99": es_99, "is_critical": False}
