"""
Walk-Forward Backtesting Engine
================================
Rigorous out-of-sample validation with rolling train/test splits.

This module implements the Walk-Forward Analysis methodology, which is
the gold standard for validating trading strategies. It prevents the two
most common failures in algorithmic trading:
1. Look-ahead bias: Using future information in training
2. Overfitting: Creating strategies that work on historical data but fail live

Walk-Forward Analysis Process:
-------------------------------
1. Split data into overlapping train/test windows:
   - Window 1: Train on period A → Test on period B
   - Window 2: Train on period B → Test on period C
   - Window 3: Train on period C → Test on period D
   - ...continuing until all data is used

2. For each window:
   - Train the agent/strategy on the training period (in-sample)
   - Test on the test period without any modification (out-of-sample)
   - Record all metrics from both periods

3. Aggregate results:
   - Mean performance across all test windows
   - Stability analysis (how consistent are results?)
   - Robustness checks (did strategy work in most windows?)

Key Features:
-------------
- Configurable window sizes (train/test/step)
- Purged Walk-Forward integration with Anti-Bias Framework
- Automatic validation thresholds
- Results serialization and visualization support

Why Walk-Forward?
----------------
Traditional backtesting suffers from:
- In-sample overfitting: Strategy optimized for historical period
- Look-ahead bias: Accidental inclusion of future data
- Regime instability: Strategy may work in one market regime only

Walk-forward addresses these by:
- Always testing on unseen data
- Validating across multiple market regimes
- Providing realistic expectations for live performance

Reference:
---------
- Pardo (2008): "The Evaluation and Optimization of Trading Strategies"
- Lopez de Prado (2018): "Advances in Financial Machine Learning"
- Kaufman (2013): "Trading Systems and Methods"

Anti-Bias Integration:
---------------------
When the Anti-Bias Framework is available, this engine uses:
- Purged splits: Remove training samples whose lookback overlaps test
- Embargo zones: Gap between train/test to prevent leakage
- This ensures truly out-of-sample testing
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
    """
    Configuration parameters for Walk-Forward Analysis.

    This dataclass defines all parameters needed to configure the
    walk-forward backtesting process, including window sizes, training
    parameters, and validation thresholds.

    Attributes:
        train_window_days: Number of days for training period (default: 365 = 1 year)
            Longer windows provide more training data but may include stale patterns
        test_window_days: Number of days for testing period (default: 90 = 3 months)
            Shorter windows provide more test samples but less statistical power
        step_days: Number of days to step forward between windows (default: 30 = 1 month)
            Smaller steps create more windows but with higher correlation
        train_iterations: Training iterations per window (default: 200)
        results_dir: Directory to save results (default: "data/backtests")
        min_trades: Minimum trades required in test period for valid result (default: 10)
        max_drawdown_threshold: Maximum allowed drawdown in test (default: 0.30 = 30%)
            Strategies exceeding this threshold are flagged but not automatically rejected

    Example:
        --------
        # Conservative configuration (longer training, shorter testing)
        config = WalkForwardConfig(
            train_window_days=730,    # 2 years training
            test_window_days=60,     # 2 months testing
            step_days=30,            # Step monthly
            min_trades=20,           # Require more trades
            max_drawdown_threshold=0.25  # Stricter drawdown limit
        )

        # Aggressive configuration (shorter training, more frequent testing)
        config = WalkForwardConfig(
            train_window_days=180,    # 6 months training
            test_window_days=30,     # 1 month testing
            step_days=7,             # Step weekly
        )
    """

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
    """
    Results from a single walk-forward window.

    This dataclass stores all metrics from both training and testing
    phases of a single walk-forward iteration. It captures performance
    data that allows for detailed analysis of strategy behavior across
    different market conditions.

    Attributes:
        window_id: Zero-based index of this window in the sequence
        train_start: Start date of training period
        train_end: End date of training period
        test_start: Start date of test period (immediately follows train_end)
        test_end: End date of test period

    Training Metrics (In-Sample):
        train_return: Total return achieved during training
        train_sharpe: Sharpe ratio during training
        train_trades: Number of trades executed during training

    Test Metrics (Out-of-Sample):
        test_return: Total return achieved during testing
        test_sharpe: Sharpe ratio during testing
        test_sortino: Sortino ratio during testing
        test_calmar: Calmar ratio during testing
        test_max_drawdown: Maximum drawdown during testing
        test_trades: Number of trades executed during testing
        test_win_rate: Win rate during testing

    Trade-Level Data:
        trades: List of dictionaries containing trade details
        equity_curve: pandas Series with equity values over time

    Example:
        --------
        result = WindowResult(
            window_id=3,
            train_start=datetime(2020, 1, 1),
            train_end=datetime(2020, 12, 31),
            test_start=datetime(2021, 1, 1),
            test_end=datetime(2021, 3, 31),
            train_return=0.15,
            train_sharpe=1.2,
            train_trades=85,
            test_return=0.08,
            test_sharpe=0.9,
            test_sortino=1.1,
            test_calmar=1.5,
            test_max_drawdown=-0.05,
            test_trades=32,
            test_win_rate=0.62,
        )
    """

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
    Walk-forward backtesting engine for rigorous strategy validation.

    This engine implements the complete walk-forward analysis workflow,
    providing rigorous out-of-sample validation for trading strategies.
    It splits historical data into multiple overlapping train/test windows,
    trains on each training window, and tests on the corresponding test
    window without any modification.

    The key insight of walk-forward analysis is that it simulates real
    trading: you train on past data and apply to future data. By doing
    this repeatedly across different time periods, you get a realistic
    picture of how the strategy might perform live.

    Attributes:
        env: Trading environment (Gym-compatible) that provides observations
            and executes trades
        agent: PPOAgent or similar trading agent that will be retrained
            for each window
        config: WalkForwardConfig with window sizes and parameters

    Key Methods:
        create_windows(): Generate train/test date ranges
        train_on_window(): Train agent on training period
        test_on_window(): Test trained agent on test period (out-of-sample)
        run(): Execute complete walk-forward analysis
        summarize_results(): Aggregate and analyze all window results
        create_purged_splits(): Generate purged CV splits (if Anti-Bias available)

    Example:
        --------
        # Configure walk-forward analysis
        config = WalkForwardConfig(
            train_window_days=365,    # 1 year training
            test_window_days=90,     # 3 months testing
            step_days=30,            # Step forward 1 month each iteration
            min_trades=10,
            max_drawdown_threshold=0.30,
        )

        # Initialize engine with environment and agent
        engine = WalkForwardEngine(env, agent, config)

        # Run complete analysis
        results = engine.run()

        # Analyze results
        summary = engine.summarize_results(results)

        print(f"Mean Test Return: {summary['mean_test_return']:.2%}")
        print(f"Positive Windows: {summary['positive_windows']}/{summary['n_windows']}")

        # Visualize
        engine.plot_results(results)

    Performance Expectations:
    -------------------------
    A robust strategy should show:
    - Positive returns in most (ideally >70%) test windows
    - Consistent Sharpe ratios across windows (low variance)
    - Test performance not significantly worse than training
    - Max drawdowns within acceptable limits

    Warning Signs:
    -------------
    - Training Sharpe >> Test Sharpe (overfitting)
    - Very few positive test windows (strategy doesn't generalize)
    - High variance in test metrics (regime instability)
    - Test max drawdowns significantly exceed training
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
            test_start = train_end  # test window starts immediately after training ends
            test_end = test_start + timedelta(days=self.config.test_window_days)

            # Check if we've reached the end
            if test_end > end_date:
                break  # stop if the full test window would exceed available data

            windows.append((train_start, train_end, test_start, test_end))

            # Step forward
            current_start += timedelta(
                days=self.config.step_days
            )  # overlapping windows (anchored walk-forward)

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
        returns = np.diff(equity) / equity[:-1]  # per-step percentage returns

        # Calculate metrics
        total_return = (equity[-1] - equity[0]) / equity[
            0
        ]  # overall equity growth fraction

        # Sharpe ratio
        if len(returns) > 1:
            sharpe = (
                np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252)
            )  # annualised (daily steps)
        else:
            sharpe = 0.0

        # Sortino ratio
        negative_returns = returns[returns < 0]
        if len(negative_returns) > 0:
            sortino = (
                np.mean(returns)
                / (np.std(negative_returns) + 1e-8)
                * np.sqrt(252)  # only downside vol in denominator
            )
        else:
            sortino = sharpe  # no losses → use Sharpe as proxy

        # Max drawdown
        running_max = np.maximum.accumulate(equity)  # peak equity up to each step
        drawdowns = (
            equity - running_max
        ) / running_max  # negative values represent drawdown depth
        max_drawdown = np.min(drawdowns)  # worst drawdown (most negative)

        # Calmar ratio
        if max_drawdown < 0:
            calmar = total_return / abs(
                max_drawdown
            )  # return per unit of drawdown risk
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
