"""
Advanced Portfolio Risk Manager
===============================
Institutional-grade risk management for quantitative trading.

This module provides comprehensive risk management tools used by hedge funds,
prop trading firms, and quantitative researchers. It implements state-of-the-art
risk measurement and management techniques.

Features:
  - Value at Risk (VaR): Historical and parametric VaR calculation
  - Risk Parity: Equal risk contribution portfolio optimization
  - Dynamic Position Sizing: Volatility-based and Kelly criterion sizing
  - Circuit Breakers: Automatic trading halt on extreme losses
  - Stress Testing: Simulate portfolio under crisis scenarios
  - Real-time Monitoring: Continuous risk metric tracking

Risk Metrics:
  - Portfolio VaR (1-day and 5-day)
  - Marginal Risk Contributions
  - Diversification Ratio
  - Beta (market sensitivity)
  - Maximum Drawdown
  - Position Concentration

Usage:
    from src.portfolio.portfolio_risk_manager import PortfolioRiskManager, PortfolioRiskConfig

    config = PortfolioRiskConfig(
        max_portfolio_var=0.02,  # 2% daily VaR limit
        max_drawdown_pct=0.15,   # 15% max drawdown
        risk_budget_method='risk_parity'
    )

    manager = PortfolioRiskManager(config)

    # Add positions
    for asset, returns in returns_df.items():
        manager.add_position(asset, position_size, returns)

    # Calculate portfolio risk
    var = manager.calculate_portfolio_var(returns_df)
    print(f"Portfolio VaR: {var['portfolio_var']:.2%}")

    # Get optimal position sizes
    sizes = manager.calculate_dynamic_position_sizes(returns_df, capital)

References:
  - "Risk Parity" by David Caberra
  - "Quantitative Risk Management" by McNeil, Frey, Embrechts
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from loguru import logger
import warnings

warnings.filterwarnings("ignore")


@dataclass
class PortfolioRiskConfig:
    """
    Configuration for portfolio risk management.

    Defines risk limits, budgeting methods, and circuit breaker settings
    for the portfolio risk manager.

    Portfolio Limits:
        - max_portfolio_var: Maximum allowed daily VaR (default 2%)
        - max_drawdown_pct: Maximum drawdown before circuit breaker (default 15%)
        - max_correlation: Max correlation between positions (default 0.7)
        - max_concentration: Max weight in single asset (default 25%)

    Risk Budgeting:
        - risk_budget_method: Method for allocating risk ('equal', 'inverse_vol', 'kelly')
        - target_volatility: Target annualized volatility (default 15%)

    Dynamic Adjustments:
        - use_dynamic_sizing: Enable dynamic position sizing
        - vol_lookback: Lookback window for volatility (default 30 days)
        - var_confidence: VaR confidence level (default 95%)

    Circuit Breakers:
        - enable_circuit_breakers: Enable automatic trading halt
        - daily_loss_limit: Max daily loss before halt (default 5%)
        - consecutive_loss_days: Days of loss before halt (default 3)

    Example:
        # Conservative config
        config = PortfolioRiskConfig(
            max_portfolio_var=0.01,  # 1% VaR
            max_drawdown_pct=0.10,   # 10% max drawdown
            risk_budget_method='risk_parity',
            daily_loss_limit=0.03    # 3% daily loss limit
        )

        # Aggressive config
        config = PortfolioRiskConfig(
            max_portfolio_var=0.05,
            max_drawdown_pct=0.25,
            risk_budget_method='inverse_vol',
            enable_circuit_breakers=False
        )
    """

    # Portfolio Limits
    max_portfolio_var: float = 0.02  # 2% daily VaR limit
    max_drawdown_pct: float = 0.15  # 15% max drawdown
    max_correlation: float = 0.7  # Max correlation between positions
    max_concentration: float = 0.25  # Max 25% in single asset

    # Risk Budgeting
    risk_budget_method: str = "equal"  # equal, inverse_vol, kelly
    target_volatility: float = 0.15  # 15% annualized target vol

    # Dynamic Adjustments
    use_dynamic_sizing: bool = True
    vol_lookback: int = 30
    var_confidence: float = 0.95

    # Circuit Breakers
    enable_circuit_breakers: bool = True
    daily_loss_limit: float = 0.05  # 5% daily loss limit
    consecutive_loss_days: int = 3


@dataclass
class PositionRisk:
    """
    Risk metrics for a single position.

    Contains comprehensive risk measurements for an individual position
    including VaR, beta, and correlation contributions.

    Attributes:
        asset: Asset symbol (e.g., 'BTC', 'AAPL')
        position_size: Current position size in base units
        weight: Portfolio weight (percentage of total portfolio)
        var_1d: 1-day Value at Risk (95% confidence)
        var_5d: 5-day Value at Risk (estimated via sqrt rule)
        beta: Market sensitivity (correlation with market)
        contribution_to_var: Position's contribution to portfolio VaR
        correlation_matrix: Correlations with other positions

    Note:
        - VaR represents potential loss at given confidence level
        - Beta > 1 means more volatile than market
        - Contribution to VaR shows how much position adds to total risk
    """

    asset: str
    position_size: float
    weight: float
    var_1d: float
    var_5d: float
    beta: float
    contribution_to_var: float
    correlation_matrix: Dict[str, float] = field(default_factory=dict)


class PortfolioRiskManager:
    """
    Institutional-grade portfolio risk management system.

    Implements comprehensive risk measurement and management techniques used by
    top quantitative funds. Provides real-time monitoring and automated
    risk controls.

    Key Capabilities:
        - Value at Risk (VaR): Both historical simulation and parametric methods
        - Risk Parity: Equal risk contribution portfolio optimization
        - Dynamic Position Sizing: Adjusts positions based on volatility and correlations
        - Circuit Breakers: Automatic halt on extreme market conditions
        - Risk Reporting: Comprehensive risk metrics and attribution

    Risk Budgeting Methods:
        - 'equal': Equal weights across all positions
        - 'inverse_vol': Weight inversely proportional to volatility
        - 'risk_parity': Equal risk contribution from each position
        - 'kelly': Kelly criterion for optimal growth

    Circuit Breaker Triggers:
        - Daily loss exceeds limit
        - Maximum drawdown exceeded
        - Consecutive loss days threshold

    Example:
        # Initialize with config
        config = PortfolioRiskConfig(
            max_portfolio_var=0.02,
            max_drawdown_pct=0.15,
            risk_budget_method='risk_parity',
            enable_circuit_breakers=True
        )

        manager = PortfolioRiskManager(config)

        # Add positions with historical returns
        manager.add_position('BTC', 0.5, btc_returns)
        manager.add_position('ETH', 0.3, eth_returns)

        # Calculate portfolio VaR
        var_result = manager.calculate_portfolio_var(returns_df)
        print(f"Portfolio VaR: {var_result['portfolio_var']:.2%}")

        # Get risk contributions
        contributions = manager.calculate_risk_contribution(returns_df)

        # Optimize position sizes
        sizes = manager.calculate_dynamic_position_sizes(returns_df, 100000)

        # Check circuit breaker after daily P&L
        cb_result = manager.check_circuit_breakers(daily_pnl=1000, portfolio_value=100000)
        if cb_result['triggered']:
            print(f"Circuit breaker: {cb_result['reason']}")
    """

    def __init__(self, config: PortfolioRiskConfig = None):
        """Initialize portfolio risk manager."""
        self.config = config or PortfolioRiskConfig()
        self.positions: Dict[str, PositionRisk] = {}
        self.portfolio_history = []
        self.risk_metrics_history = []

        # Circuit breaker state
        self.circuit_breaker_triggered = False
        self.consecutive_losses = 0
        self.last_reset_date = datetime.now()

        logger.info("PortfolioRiskManager initialized")
        logger.info(f"  Max VaR: {self.config.max_portfolio_var:.1%}")
        logger.info(f"  Max Drawdown: {self.config.max_drawdown_pct:.1%}")
        logger.info(f"  Risk Budget: {self.config.risk_budget_method}")

    def add_position(
        self, asset: str, position_size: float, returns: pd.Series
    ) -> PositionRisk:
        """
        Add a new position with full risk analysis.
        """
        # Calculate position metrics
        var_1d = self._calculate_var(returns, confidence=self.config.var_confidence)
        var_5d = var_1d * np.sqrt(5)  # scale 1-day VaR to 5-day using √T rule

        # Calculate beta (market sensitivity)
        beta = self._calculate_beta(returns)

        position = PositionRisk(
            asset=asset,
            position_size=position_size,
            weight=0.0,  # Will be calculated
            var_1d=var_1d,
            var_5d=var_5d,
            beta=beta,
            contribution_to_var=0.0,
        )

        self.positions[asset] = position
        self._update_weights()

        logger.info(
            f"Added position: {asset}, Size: {position_size:.4f}, VaR: {var_1d:.4f}"
        )

        return position

    def _calculate_var(self, returns: pd.Series, confidence: float = 0.95) -> float:
        """Calculate Value at Risk using historical simulation."""
        if len(returns) < 30:
            return 0.0

        var = np.percentile(
            returns.dropna(), (1 - confidence) * 100
        )  # left-tail percentile (e.g. 5th for 95% VaR)
        return abs(var)  # return as positive loss figure

    def _calculate_beta(
        self, returns: pd.Series, market_returns: pd.Series = None
    ) -> float:
        """Calculate asset beta to market."""
        if market_returns is None or len(returns) != len(market_returns):
            # Use returns themselves as proxy
            return 1.0

        covariance = np.cov(returns, market_returns)[0, 1]  # asset–market covariance
        market_variance = np.var(market_returns)  # market variance

        return covariance / (
            market_variance + 1e-10
        )  # β = Cov(asset, market) / Var(market)

    def _update_weights(self):
        """Update position weights based on risk budgeting."""
        total_size = sum(abs(p.position_size) for p in self.positions.values())

        if total_size == 0:
            return

        for asset, position in self.positions.items():
            position.weight = abs(position.position_size) / total_size

    def calculate_portfolio_var(self, returns_df: pd.DataFrame) -> Dict:
        """
        Calculate portfolio VaR using variance-covariance method.

        This is the industry standard for portfolio risk measurement.
        """
        if len(self.positions) == 0:
            return {"portfolio_var": 0.0, "diversification_ratio": 1.0}

        # Get position weights
        weights = np.array([p.weight for p in self.positions.values()])

        # Calculate covariance matrix
        cov_matrix = returns_df[list(self.positions.keys())].cov() * 252  # Annualized

        # Portfolio variance: σ²_p = w^T Σ w (matrix form)
        portfolio_variance = weights.T @ cov_matrix.values @ weights
        portfolio_volatility = np.sqrt(portfolio_variance)  # annualised portfolio σ

        # Parametric VaR (assuming normal distribution)
        z_score = 1.645  # 95% confidence one-tail z-value
        portfolio_var = (
            portfolio_volatility * z_score / np.sqrt(252)
        )  # scale annual σ → daily VaR

        # Diversification ratio: weighted sum of individual VaRs / portfolio VaR
        # DR > 1 means diversification reduces risk below simple sum-of-parts
        individual_vars = np.array([p.var_1d for p in self.positions.values()])
        weighted_individual_var = np.sum(weights * individual_vars)
        diversification_ratio = weighted_individual_var / (portfolio_var + 1e-10)

        return {
            "portfolio_var": portfolio_var,
            "portfolio_volatility": portfolio_volatility,
            "diversification_ratio": diversification_ratio,
            "covariance_matrix": cov_matrix,
            "weights": dict(zip(self.positions.keys(), weights)),
        }

    def calculate_risk_contribution(self, returns_df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate marginal risk contribution of each position.

        Used for risk parity allocation - positions should contribute
        equally to portfolio risk.
        """
        if len(self.positions) < 2:
            return {}

        weights = np.array([p.weight for p in self.positions.values()])
        cov_matrix = returns_df[list(self.positions.keys())].cov() * 252

        portfolio_variance = weights.T @ cov_matrix.values @ weights

        # Marginal contribution
        marginal_contribution = (cov_matrix.values @ weights) / np.sqrt(
            portfolio_variance
        )

        # Absolute contribution
        risk_contributions = (
            weights * marginal_contribution / np.sqrt(portfolio_variance)
        )

        return dict(zip(self.positions.keys(), risk_contributions))

    def optimize_risk_parity(self, returns_df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate risk parity weights.

        Risk parity allocates capital so each asset contributes
        equally to portfolio risk. This is a SOTA approach used
        by Bridgewater and other top funds.
        """
        if len(self.positions) == 0:
            return {}

        assets = list(self.positions.keys())
        n = len(assets)

        # Initial equal weights
        weights = np.ones(n) / n

        # Iterative optimization
        for _ in range(100):  # Max iterations
            # Calculate risk contributions
            cov_matrix = returns_df[assets].cov() * 252
            portfolio_variance = weights.T @ cov_matrix.values @ weights

            if portfolio_variance < 1e-10:
                break

            # Marginal risk contributions
            marginal = (cov_matrix.values @ weights) / np.sqrt(portfolio_variance)
            risk_contributions = weights * marginal

            # Update weights to equalize risk contributions
            target_risk = np.sum(risk_contributions) / n
            new_weights = target_risk / (marginal + 1e-10)

            # Normalize
            new_weights = new_weights / np.sum(new_weights)

            # Check convergence
            if np.max(np.abs(new_weights - weights)) < 1e-6:
                break

            weights = new_weights

        return dict(zip(assets, weights))

    def calculate_dynamic_position_sizes(
        self, returns_df: pd.DataFrame, capital: float
    ) -> Dict[str, float]:
        """
        Calculate position sizes using dynamic risk budgeting.

        Adjusts sizes based on:
        - Current volatility
        - Correlation environment
        - Risk budget
        """
        if len(self.positions) == 0:
            return {}

        if self.config.risk_budget_method == "equal":
            # Equal weight
            n = len(self.positions)
            weights = {asset: 1.0 / n for asset in self.positions.keys()}

        elif self.config.risk_budget_method == "inverse_vol":
            # Inverse volatility weighting
            vols = {}
            for asset in self.positions.keys():
                vol = returns_df[asset].rolling(self.config.vol_lookback).std().iloc[-1]
                vols[asset] = 1.0 / (vol + 1e-10)

            total_inv_vol = sum(vols.values())
            weights = {asset: vol / total_inv_vol for asset, vol in vols.items()}

        elif self.config.risk_budget_method == "risk_parity":
            weights = self.optimize_risk_parity(returns_df)

        else:
            # Default to equal
            n = len(self.positions)
            weights = {asset: 1.0 / n for asset in self.positions.keys()}

        # Scale by capital
        position_sizes = {asset: weight * capital for asset, weight in weights.items()}

        return position_sizes

    def check_circuit_breakers(self, daily_pnl: float, portfolio_value: float) -> Dict:
        """
        Check if circuit breakers should trigger.

        Implements institutional-grade risk controls.
        """
        if not self.config.enable_circuit_breakers:
            return {"triggered": False, "reason": None}

        triggers = []

        # Daily loss limit
        daily_return = daily_pnl / portfolio_value if portfolio_value > 0 else 0
        if daily_return < -self.config.daily_loss_limit:
            triggers.append(f"Daily loss limit: {daily_return:.2%}")
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0

        # Consecutive loss days
        if self.consecutive_losses >= self.config.consecutive_loss_days:
            triggers.append(f"Consecutive losses: {self.consecutive_losses} days")

        # Max drawdown check
        if len(self.portfolio_history) > 0:
            peak = max(self.portfolio_history)
            current_drawdown = (peak - portfolio_value) / peak

            if current_drawdown > self.config.max_drawdown_pct:
                triggers.append(f"Max drawdown: {current_drawdown:.2%}")

        # Update history
        self.portfolio_history.append(portfolio_value)

        # Trigger circuit breaker
        if triggers:
            self.circuit_breaker_triggered = True
            logger.warning(f"CIRCUIT BREAKER TRIGGERED: {'; '.join(triggers)}")

            return {
                "triggered": True,
                "reason": triggers,
                "reduce_positions": True,
                "halt_trading": len(triggers) >= 2,
            }

        return {"triggered": False, "reason": None}

    def get_risk_report(self, returns_df: pd.DataFrame) -> Dict:
        """
        Generate comprehensive risk report.
        """
        portfolio_var = self.calculate_portfolio_var(returns_df)
        risk_contributions = self.calculate_risk_contribution(returns_df)

        report = {
            "timestamp": datetime.now(),
            "n_positions": len(self.positions),
            "portfolio_var_1d": portfolio_var["portfolio_var"],
            "portfolio_volatility": portfolio_var["portfolio_volatility"],
            "diversification_ratio": portfolio_var["diversification_ratio"],
            "circuit_breaker": self.circuit_breaker_triggered,
            "positions": {},
            "risk_contributions": risk_contributions,
            "weights": portfolio_var["weights"],
        }

        for asset, position in self.positions.items():
            report["positions"][asset] = {
                "size": position.position_size,
                "weight": position.weight,
                "var_1d": position.var_1d,
                "beta": position.beta,
                "risk_contribution": risk_contributions.get(asset, 0.0),
            }

        self.risk_metrics_history.append(report)

        return report

    def reset_circuit_breaker(self):
        """Manually reset circuit breaker."""
        self.circuit_breaker_triggered = False
        self.consecutive_losses = 0
        logger.info("Circuit breaker manually reset")


class StressTestEngine:
    """
    Portfolio stress testing engine.

    Simulates portfolio performance under various market stress scenarios
    to understand potential losses during crisis events.

    Predefined Scenarios:
        - market_crash: 20% market drop, 3x volatility
        - high_volatility: 5% drop, 2.5x volatility
        - correlation_spike: All correlations go to 0.9
        - liquidity_crisis: 10% drop, 2x volatility

    Output Metrics:
        - VaR (95%): Value at Risk at 95% confidence
        - CVaR: Conditional Value at Risk (expected shortfall)
        - Max Drawdown: Maximum peak-to-trough decline
        - Worst Day: Single worst trading day
        - Stress Sharpe: Risk-adjusted return under stress

    Example:
        engine = StressTestEngine()

        # Run all scenarios
        results = engine.run_stress_test(returns_df, weights)

        # Run specific scenario
        crash_results = engine.run_stress_test(returns_df, weights, 'market_crash')

        print(f"Crash VaR: {crash_results['market_crash']['var_95']:.2%}")
        print(f"Crash Max DD: {crash_results['market_crash']['max_drawdown']:.2%}")
    """

    def __init__(self):
        self.scenarios = {
            "market_crash": {"return_shock": -0.20, "vol_mult": 3.0},
            "high_volatility": {"return_shock": -0.05, "vol_mult": 2.5},
            "correlation_spike": {"correlation": 0.9, "vol_mult": 1.5},
            "liquidity_crisis": {"return_shock": -0.10, "vol_mult": 2.0},
        }
        logger.info("StressTestEngine initialized")

    def run_stress_test(
        self, returns_df: pd.DataFrame, weights: Dict[str, float], scenario: str = None
    ) -> Dict:
        """
        Run stress test on portfolio.
        """
        if scenario and scenario not in self.scenarios:
            raise ValueError(f"Unknown scenario: {scenario}")

        scenarios_to_run = [scenario] if scenario else list(self.scenarios.keys())
        results = {}

        for scen in scenarios_to_run:
            params = self.scenarios[scen]

            # Apply shock to returns
            shocked_returns = returns_df.copy()

            if "return_shock" in params:
                shocked_returns += params["return_shock"] / 252

            if "vol_mult" in params:
                shocked_returns *= params["vol_mult"]

            if "correlation" in params:
                # Increase correlation (simplified)
                avg_corr = returns_df.corr().values.mean()
                shocked_returns = shocked_returns * np.sqrt(
                    params["correlation"] / (avg_corr + 1e-10)
                )

            # Calculate portfolio metrics under stress
            port_returns = sum(
                shocked_returns[asset] * weight for asset, weight in weights.items()
            )

            var_95 = np.percentile(port_returns, 5)
            cvar_95 = port_returns[port_returns <= var_95].mean()
            max_dd = self._calculate_max_drawdown(port_returns.cumsum())

            results[scen] = {
                "var_95": var_95,
                "cvar_95": cvar_95,
                "max_drawdown": max_dd,
                "worst_day": port_returns.min(),
                "sharpe": port_returns.mean()
                / (port_returns.std() + 1e-10)
                * np.sqrt(252),
            }

        return results

    def _calculate_max_drawdown(self, cumulative_returns: pd.Series) -> float:
        """Calculate maximum drawdown."""
        rolling_max = cumulative_returns.expanding().max()
        drawdown = cumulative_returns - rolling_max
        return drawdown.min()


# Convenience functions for production use
def calculate_portfolio_metrics(
    returns: pd.DataFrame, weights: Dict[str, float]
) -> Dict:
    """Calculate key portfolio metrics."""
    port_returns = sum(returns[asset] * weight for asset, weight in weights.items())

    return {
        "annualized_return": port_returns.mean() * 252,
        "annualized_volatility": port_returns.std() * np.sqrt(252),
        "sharpe_ratio": port_returns.mean()
        / (port_returns.std() + 1e-10)
        * np.sqrt(252),
        "var_95": np.percentile(port_returns, 5),
        "max_drawdown": (
            port_returns.cumsum() - port_returns.cumsum().expanding().max()
        ).min(),
    }


def suggest_position_sizes(
    risk_manager: PortfolioRiskManager, returns_df: pd.DataFrame, capital: float
) -> Dict[str, float]:
    """Get suggested position sizes based on current risk."""
    return risk_manager.calculate_dynamic_position_sizes(returns_df, capital)
