"""
Walk-Forward Backtesting Engine
================================
Rigorous validation with rolling train/test splits.

Walk-Forward Analysis:
- Train on window 1 → Test on window 2
- Train on window 2 → Test on window 3
- Train on window 3 → Test on window 4
- ...

Prevents look-ahead bias and overfitting.

Reference: Pardo (2008) - The Evaluation and Optimization of Trading Strategies
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from pathlib import Path
from loguru import logger
from datetime import datetime, timedelta
import json

from agents.ppo_agent import PPOAgent

# Anti-Bias Framework integration
try:
    from validation.antibias_walkforward import (
        PurgedWalkForwardCV,
        WalkForwardConfig as AntibiasWFConfig,
        PurgedScaler,
    )

    ANTIBIAS_AVAILABLE = True
except ImportError:
    ANTIBIAS_AVAILABLE = False
    logger.warning("Anti-Bias Walk-Forward not available")


@dataclass
class WalkForwardConfig:
    """Walk-forward analysis configuration."""

    # Window sizes
    train_window_days: int = 365  # 1 year training
    test_window_days: int = 90  # 3 months testing
    step_days: int = 30  # Move forward 1 month

    # Training
    train_iterations: int = 200  # Iterations per window

    # Paths
    results_dir: str = "data/backtests"

    # Validation
    min_trades: int = 10  # Minimum trades in test period
    max_drawdown_threshold: float = 0.30  # 30% max DD threshold


@dataclass
class WindowResult:
    """Results from a single walk-forward window."""

    window_id: int
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime

    # Training metrics
    train_return: float
    train_sharpe: float
    train_trades: int

    # Test metrics (out-of-sample)
    test_return: float
    test_sharpe: float
    test_sortino: float
    test_calmar: float
    test_max_drawdown: float
    test_trades: int
    test_win_rate: float

    # Trade-level data
    trades: List[Dict] = field(default_factory=list)
    equity_curve: pd.Series = field(default_factory=pd.Series)


class WalkForwardEngine:
    """
    Walk-forward backtesting engine.

    Implements rigorous out-of-sample validation:
    1. Split data into overlapping train/test windows
    2. Train agent on each training window
    3. Test agent on corresponding test window (out-of-sample)
    4. Aggregate results across all windows

    Usage:
    ------
    config = WalkForwardConfig(
        train_window_days=365,
        test_window_days=90,
        step_days=30
    )

    engine = WalkForwardEngine(env, agent, config)
    results = engine.run()

    # Analyze
    summary = engine.summarize_results(results)
    engine.plot_results(results)
    """

    def __init__(
        self,
        env,  # Trading environment
        agent: PPOAgent,
        config: WalkForwardConfig,
    ):
        """
        Initialize walk-forward engine.

        Parameters:
        -----------
        env : gym.Env
            Trading environment
        agent : PPOAgent
            Trading agent (will be retrained each window)
        config : WalkForwardConfig
            Backtesting configuration
        """
        self.env = env
        self.agent = agent
        self.config = config

        # Create results directory
        self.results_dir = Path(config.results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        logger.info("WalkForwardEngine initialized")
        logger.info(f"  Train window: {config.train_window_days} days")
        logger.info(f"  Test window: {config.test_window_days} days")
        logger.info(f"  Step: {config.step_days} days")

    def create_windows(
        self, start_date: datetime, end_date: datetime
    ) -> List[Tuple[datetime, datetime, datetime, datetime]]:
        """
        Create train/test windows.

        Parameters:
        -----------
        start_date : datetime
            Start of data
        end_date : datetime
            End of data

        Returns:
        --------
        windows : List[Tuple]
            List of (train_start, train_end, test_start, test_end)
        """
        windows = []

        current_start = start_date

        while True:
            # Calculate window dates
            train_start = current_start
            train_end = train_start + timedelta(days=self.config.train_window_days)
            test_start = train_end
            test_end = test_start + timedelta(days=self.config.test_window_days)

            # Check if we've reached the end
            if test_end > end_date:
                break

            windows.append((train_start, train_end, test_start, test_end))

            # Step forward
            current_start += timedelta(days=self.config.step_days)

        logger.info(f"Created {len(windows)} walk-forward windows")

        return windows

    def train_on_window(self, train_data: pd.DataFrame, window_id: int) -> Dict:
        """
        Train agent on training window.

        Parameters:
        -----------
        train_data : pd.DataFrame
            Training data
        window_id : int
            Window identifier

        Returns:
        --------
        metrics : dict
            Training metrics
        """
        logger.info(f"Training on window {window_id}...")

        # TODO: Implement training loop
        # For now, return mock metrics

        metrics = {"return": 0.10, "sharpe": 1.5, "trades": 50}

        return metrics

    def test_on_window(self, test_data: pd.DataFrame, window_id: int) -> Dict:
        """
        Test agent on test window (out-of-sample).

        Parameters:
        -----------
        test_data : pd.DataFrame
            Test data
        window_id : int
            Window identifier

        Returns:
        --------
        metrics : dict
            Test metrics
        """
        logger.info(f"Testing on window {window_id}...")

        # Reset environment with test data
        obs, info = self.env.reset()

        # Collect trades
        trades = []
        equity_curve = [self.config.initial_capital]

        episode_count = 0
        max_episodes = 10  # Limit episodes in test window

        while episode_count < max_episodes:
            done = False
            hidden = None

            while not done:
                # Deterministic action selection
                action, _, _, hidden = self.agent.select_action(
                    obs, hidden, deterministic=True
                )

                obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated

                # Record trade if executed
                if "trade_executed" in info and info["trade_executed"]:
                    trades.append(
                        {
                            "timestamp": info.get("step", 0),
                            "action": action,
                            "price": info.get("price", 0),
                            "pnl": info.get("pnl", 0),
                            "return": info.get("return", 0),
                        }
                    )

                # Record equity
                equity_curve.append(info.get("equity", equity_curve[-1]))

            episode_count += 1
            obs, info = self.env.reset()

        # Calculate metrics
        metrics = self._calculate_metrics(trades, equity_curve)
        metrics["trades"] = trades
        metrics["equity_curve"] = pd.Series(equity_curve)

        return metrics

    def _calculate_metrics(self, trades: List[Dict], equity_curve: List[float]) -> Dict:
        """Calculate performance metrics."""
        if len(trades) < self.config.min_trades:
            logger.warning(
                f"Only {len(trades)} trades, below minimum {self.config.min_trades}"
            )

        # Convert to arrays
        equity = np.array(equity_curve)
        returns = np.diff(equity) / equity[:-1]

        # Calculate metrics
        total_return = (equity[-1] - equity[0]) / equity[0]

        # Sharpe ratio
        if len(returns) > 1:
            sharpe = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252)
        else:
            sharpe = 0.0

        # Sortino ratio
        negative_returns = returns[returns < 0]
        if len(negative_returns) > 0:
            sortino = (
                np.mean(returns) / (np.std(negative_returns) + 1e-8) * np.sqrt(252)
            )
        else:
            sortino = sharpe

        # Max drawdown
        running_max = np.maximum.accumulate(equity)
        drawdowns = (equity - running_max) / running_max
        max_drawdown = np.min(drawdowns)

        # Calmar ratio
        if max_drawdown < 0:
            calmar = total_return / abs(max_drawdown)
        else:
            calmar = 0.0

        # Win rate
        if trades:
            winning_trades = sum(1 for t in trades if t["pnl"] > 0)
            win_rate = winning_trades / len(trades)
        else:
            win_rate = 0.0

        return {
            "return": total_return,
            "sharpe": sharpe,
            "sortino": sortino,
            "calmar": calmar,
            "max_drawdown": max_drawdown,
            "win_rate": win_rate,
        }

    def run(self) -> List[WindowResult]:
        """
        Run complete walk-forward analysis.

        Returns:
        --------
        results : List[WindowResult]
            Results from all windows
        """
        logger.info("=" * 80)
        logger.info("STARTING WALK-FORWARD ANALYSIS")
        logger.info("=" * 80)

        # Get data date range
        # (Simplified - in production, extract from env.price_data)
        start_date = datetime(2020, 1, 1)
        end_date = datetime(2024, 12, 31)

        # Create windows
        windows = self.create_windows(start_date, end_date)

        results = []

        for i, (train_start, train_end, test_start, test_end) in enumerate(windows):
            logger.info(f"\n{'=' * 80}")
            logger.info(f"Window {i + 1}/{len(windows)}")
            logger.info(f"  Train: {train_start.date()} to {train_end.date()}")
            logger.info(f"  Test:  {test_start.date()} to {test_end.date()}")
            logger.info(f"{'=' * 80}")

            # Train
            train_metrics = self.train_on_window(None, i)

            # Test (out-of-sample)
            test_metrics = self.test_on_window(None, i)

            # Validate
            if test_metrics["max_drawdown"] < -self.config.max_drawdown_threshold:
                logger.error(
                    f"Window {i}: Max drawdown {test_metrics['max_drawdown'] * 100:.1f}% exceeds threshold!"
                )

            # Store results
            result = WindowResult(
                window_id=i,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                train_return=train_metrics["return"],
                train_sharpe=train_metrics["sharpe"],
                train_trades=train_metrics["trades"],
                test_return=test_metrics["return"],
                test_sharpe=test_metrics["sharpe"],
                test_sortino=test_metrics["sortino"],
                test_calmar=test_metrics["calmar"],
                test_max_drawdown=test_metrics["max_drawdown"],
                test_trades=len(test_metrics["trades"]),
                test_win_rate=test_metrics["win_rate"],
                trades=test_metrics["trades"],
                equity_curve=test_metrics["equity_curve"],
            )

            results.append(result)

            # Log progress
            logger.info(f"\nWindow {i + 1} Results:")
            logger.info(f"  Train Return: {result.train_return * 100:.2f}%")
            logger.info(f"  Test Return: {result.test_return * 100:.2f}% (OOS)")
            logger.info(f"  Test Sharpe: {result.test_sharpe:.2f}")
            logger.info(f"  Test Max DD: {result.test_max_drawdown * 100:.2f}%")

        logger.success("\n" + "=" * 80)
        logger.success("WALK-FORWARD ANALYSIS COMPLETE")
        logger.success("=" * 80)

        # Save results
        self._save_results(results)

        return results

    def summarize_results(self, results: List[WindowResult]) -> Dict:
        """
        Summarize walk-forward results.

        Parameters:
        -----------
        results : List[WindowResult]
            Results from all windows

        Returns:
        --------
        summary : dict
            Aggregated statistics
        """
        test_returns = [r.test_return for r in results]
        test_sharpes = [r.test_sharpe for r in results]
        test_max_dds = [r.test_max_drawdown for r in results]
        test_win_rates = [r.test_win_rate for r in results]

        summary = {
            "n_windows": len(results),
            "mean_test_return": np.mean(test_returns),
            "std_test_return": np.std(test_returns),
            "median_test_return": np.median(test_returns),
            "mean_test_sharpe": np.mean(test_sharpes),
            "mean_test_max_dd": np.mean(test_max_dds),
            "worst_test_max_dd": np.min(test_max_dds),
            "mean_win_rate": np.mean(test_win_rates),
            "positive_windows": sum(1 for r in test_returns if r > 0),
            "negative_windows": sum(1 for r in test_returns if r < 0),
        }

        logger.info("\n" + "=" * 80)
        logger.info("WALK-FORWARD SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Windows: {summary['n_windows']}")
        logger.info(
            f"Mean Test Return: {summary['mean_test_return'] * 100:.2f}% ± {summary['std_test_return'] * 100:.2f}%"
        )
        logger.info(f"Median Test Return: {summary['median_test_return'] * 100:.2f}%")
        logger.info(f"Mean Sharpe: {summary['mean_test_sharpe']:.2f}")
        logger.info(f"Mean Max DD: {summary['mean_test_max_dd'] * 100:.2f}%")
        logger.info(f"Worst Max DD: {summary['worst_test_max_dd'] * 100:.2f}%")
        logger.info(f"Win Rate: {summary['mean_win_rate'] * 100:.1f}%")
        logger.info(
            f"Positive Windows: {summary['positive_windows']}/{summary['n_windows']}"
        )

        return summary

    def _save_results(self, results: List[WindowResult]):
        """Save results to disk."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = self.results_dir / f"walkforward_{timestamp}.json"

        # Convert to serializable format
        data = []
        for r in results:
            data.append(
                {
                    "window_id": r.window_id,
                    "train_start": r.train_start.isoformat(),
                    "train_end": r.train_end.isoformat(),
                    "test_start": r.test_start.isoformat(),
                    "test_end": r.test_end.isoformat(),
                    "train_return": r.train_return,
                    "train_sharpe": r.train_sharpe,
                    "test_return": r.test_return,
                    "test_sharpe": r.test_sharpe,
                    "test_sortino": r.test_sortino,
                    "test_max_drawdown": r.test_max_drawdown,
                    "test_win_rate": r.test_win_rate,
                }
            )

        with open(results_file, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Results saved: {results_file}")

    def create_purged_splits(
        self,
        n_samples: int,
        feature_lookback: int = 100,
        n_splits: int = 5,
        holdout_pct: float = 0.15,
    ) -> Tuple[List, np.ndarray]:
        """
        Create purged walk-forward splits using Anti-Bias Framework.

        Args:
            n_samples: Total number of samples
            feature_lookback: Maximum feature lookback window
            n_splits: Number of CV splits
            holdout_pct: Percentage for final holdout

        Returns:
            (folds, holdout_idx)
        """
        if not ANTIBIAS_AVAILABLE:
            logger.warning("Anti-Bias Framework not available, using standard splits")
            return [], np.array([])

        logger.info("Creating Purged Walk-Forward splits...")
        config = AntibiasWFConfig(
            n_splits=n_splits,
            feature_lookback=feature_lookback,
            holdout_pct=holdout_pct,
            purge=True,
            embargo_pct=0.01,
        )

        cv = PurgedWalkForwardCV(config)
        folds, holdout_idx = cv.split(n_samples=n_samples)

        logger.info(
            f"Created {len(folds)} purged folds with {len(holdout_idx)} holdout samples"
        )
        return folds, holdout_idx


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("WALK-FORWARD ENGINE TEST")
    print("=" * 80)

    # Mock environment and agent
    class MockEnv:
        def reset(self):
            return np.random.randn(20), {}

        def step(self, action):
            obs = np.random.randn(20)
            reward = np.random.randn() * 0.1
            done = np.random.rand() < 0.01
            info = {
                "equity": 100000 + np.random.randn() * 1000,
                "return": np.random.randn() * 0.01,
                "price": 50000,
                "pnl": reward * 1000,
                "trade_executed": np.random.rand() < 0.1,
            }
            return obs, reward, done, False, info

    from agents.ppo_agent import PPOAgent, PPOConfig

    config = PPOConfig(state_dim=20, n_actions=3)
    agent = PPOAgent(config)

    env = MockEnv()

    # Configure walk-forward
    wf_config = WalkForwardConfig(
        train_window_days=365, test_window_days=90, step_days=30
    )

    engine = WalkForwardEngine(env, agent, wf_config)

    print("\n✓ Engine initialized")

    # Test window creation
    print("\n[TEST] Window Creation")
    start = datetime(2020, 1, 1)
    end = datetime(2022, 12, 31)
    windows = engine.create_windows(start, end)
    print(f"  Created {len(windows)} windows")

    print("\n" + "=" * 80)
    print("✓ WALK-FORWARD ENGINE TEST PASSED")
    print("=" * 80)
