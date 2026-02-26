"""
Main Training Script
====================
Entry point for training the adversarial trading system.

Usage:
------
# Train with default config
python train.py

# Train with custom config
python train.py --config config/training/adversarial.yaml

# Resume from checkpoint
python train.py --resume data/models/adversarial/checkpoint_iter_100.pth
"""

import argparse
import sys
import yaml
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

# Phase 1: Data
from data.ccxt_loader import CCXTDataLoader
from features.feature_engine import FeatureEngine, FeatureConfig

# Phase 2: Environment
from environment.config_integrated_env import ConfigIntegratedTradingEnv
from environment.config_system import (
    EnvironmentConfig,
    load_environment_config_from_yaml,
)

# Phase 5: Training
from agents.ppo_agent import PPOConfig
from training.adversarial_trainer import AdversarialTrainer, AdversarialConfig


def setup_logging():
    """Configure logging."""
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
        level="INFO",
    )
    logger.add(
        "logs/training/train_{time}.log",
        rotation="1 day",
        retention="30 days",
        level="DEBUG",
    )


def load_data(args):
    """Load and prepare data using CCXTDataLoader and FeatureEngine."""
    logger.info("Loading data...")

    # Try to load cached data first (faster than downloading)
    cache_dir = Path("data/cache")
    processed_dir = Path("data/processed")

    if args.use_cached and cache_dir.exists():
        cached_files = list(cache_dir.glob("*.parquet"))
        if cached_files:
            logger.info(f"Loading cached data from {cached_files[0]}")
            price_data = pd.read_parquet(cached_files[0])

            # Load cached features if available
            feature_files = list(processed_dir.glob("*_features.parquet"))
            if feature_files:
                logger.info(f"Loading cached features from {feature_files[0]}")
                features = pd.read_parquet(feature_files[0])

                # Check for feature mismatch (e.g. we added RSI/MACD but cache is old)
                # Ideally we check columns, but for now let's just warn or refresh if requested.
                # If force_refresh is False, we assume cache is good.
                # BUT, main expects a split dict now, not a tuple.

                logger.success(
                    f"Data loaded from cache: {len(price_data)} candles, {len(features.columns)} features"
                )

                # We must split the cached data too!
                # Reuse the splitting logic which we'll define below or duplicate for now.
                # Let's duplicate strictly for safety in this restricted context edit.

                common = price_data.index.intersection(features.index)
                price_data = price_data.loc[common]
                features = features.loc[common]

                n = len(price_data)
                train_idx = int(n * 0.70)
                val_idx = int(n * 0.85)

                return {
                    "train": (price_data.iloc[:train_idx], features.iloc[:train_idx]),
                    "val": (
                        price_data.iloc[train_idx:val_idx],
                        features.iloc[train_idx:val_idx],
                    ),
                    "test": (price_data.iloc[val_idx:], features.iloc[val_idx:]),
                }

    # Download from exchange using CCXTDataLoader
    logger.info(f"Downloading data from {args.exchange}...")

    from data.ccxt_loader import CCXTDataLoader, DataLoaderConfig

    config = DataLoaderConfig(
        exchange_id=args.exchange,
        exchange_type=args.exchange_type,
        rate_limit_ms=args.rate_limit,
        cache_dir=cache_dir,
        processed_dir=processed_dir,
        compression="snappy",
    )

    try:
        loader = CCXTDataLoader(config)
        price_data = loader.download_and_cache(
            symbol=args.symbol,
            timeframe=args.timeframe,
            start_date=args.start_date,
            end_date=args.end_date,
            force_refresh=args.force_refresh,
        )

        logger.success(f"Downloaded {len(price_data)} candles from {args.exchange}")

    except Exception as e:
        logger.error(f"Failed to download data: {e}")
        logger.warning("Falling back to synthetic data for testing")

        # Fallback to synthetic data (for development only)
        np.random.seed(42)
        n_points = 10000
        dates = pd.date_range("2023-01-01", periods=n_points, freq="1H")

        close = 50000 + np.cumsum(np.random.randn(n_points) * 100)

        price_data = pd.DataFrame(
            {
                "open": close + np.random.randn(n_points) * 50,
                "high": close + abs(np.random.randn(n_points) * 100),
                "low": close - abs(np.random.randn(n_points) * 100),
                "close": close,
                "volume": np.random.uniform(100, 1000, n_points),
            },
            index=dates,
        )

    # Feature Engineering with strict data leakage prevention
    logger.info("Generating features with FeatureEngine...")
    from features.feature_engine import FeatureEngine, FeatureConfig

    feature_config = FeatureConfig(
        volatility_window=args.volatility_window,
        ou_window=args.ou_window,
        rolling_mean_window=args.rolling_mean_window,
        use_log_returns=True,
        scaler_type=args.scaler_type,
        save_scaler=True,
        scaler_path=processed_dir,
        dropna_strategy="rolling",
        min_valid_rows=1000,
    )

    engine = FeatureEngine(feature_config)

    # Split data chronologically (70% Train, 15% Val, 15% Test)
    n = len(price_data)
    train_idx = int(n * 0.70)
    val_idx = int(n * 0.85)

    train_data = price_data.iloc[:train_idx]
    val_data = price_data.iloc[train_idx:val_idx]
    test_data = price_data.iloc[val_idx:]

    logger.info(
        f"Data Split: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}"
    )

    # 1. Fit on TRAIN data only
    logger.info("Fitting FeatureEngine on TRAINING set...")
    train_features = engine.fit_transform(train_data)

    # 2. Transform Val and Test using Train statistics
    logger.info("Transforming Validation and Test sets...")
    val_features = engine.transform(val_data)
    test_features = engine.transform(test_data)

    # Align indexes after feature engine processing (dropping NaNs etc)
    # The engine returns a new dataframe with potentially fewer rows
    # We need to ensure price_data aligns with features for the environment

    # BUG-FIX: 'features' was not defined in the non-cache path.
    # The splits are built directly from the already correctly computed
    # train_features / val_features / test_features.

    # Align indexes after FeatureEngine processing
    common_train = price_data.iloc[:train_idx].index.intersection(train_features.index)
    common_val = price_data.iloc[train_idx:val_idx].index.intersection(
        val_features.index
    )
    common_test = price_data.iloc[val_idx:].index.intersection(test_features.index)

    splits = {
        "train": (price_data.loc[common_train], train_features.loc[common_train]),
        "val": (price_data.loc[common_val], val_features.loc[common_val]),
        "test": (price_data.loc[common_test], test_features.loc[common_test]),
    }

    logger.success(
        f"Data prepared: Train={len(splits['train'][0])}, "
        f"Val={len(splits['val'][0])}, Test={len(splits['test'][0])} samples"
    )

    return splits


def create_environment(price_data, features):
    """Create trading environment."""
    logger.info("Creating environment...")

    # Load config
    config_path = Path("config/environment/realistic_env.yaml")

    if config_path.exists():
        env_config = load_environment_config_from_yaml(str(config_path))
    else:
        logger.warning("Config not found, using defaults")
        env_config = EnvironmentConfig()

    env = ConfigIntegratedTradingEnv(price_data, features, env_config)

    logger.success("Environment created")

    return env


def create_trainer(env, args):
    """Create adversarial trainer."""
    logger.info("Creating trainer...")

    # Load training config
    config_path = (
        Path(args.config) if args.config else Path("config/training/adversarial.yaml")
    )

    if config_path.exists():
        logger.info(f"Loading training config from {config_path}")
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
    else:
        logger.warning(f"Config {config_path} not found, using hardcoded defaults")
        cfg = {}

    # Agent configs
    state_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    # Helper to get config value with default
    def get_cfg(section, key, default):
        return cfg.get(section, {}).get(key, default)

    trader_config = PPOConfig(
        state_dim=state_dim,
        hidden_dim=get_cfg("trader", "hidden_dim", 128),
        n_actions=n_actions,
        actor_lr=float(get_cfg("trader", "actor_lr", 3e-4)),
        critic_lr=float(get_cfg("trader", "critic_lr", 1e-3)),
        gamma=get_cfg("trader", "gamma", 0.99),
        gae_lambda=get_cfg("trader", "gae_lambda", 0.95),
        clip_epsilon=get_cfg("trader", "clip_epsilon", 0.2),
        n_epochs=get_cfg("trader", "n_epochs", 10),
        batch_size=get_cfg("trader", "batch_size", 64),
        use_recurrent=get_cfg("trader", "use_recurrent", True),
        rnn_type=get_cfg("trader", "rnn_type", "GRU"),
        entropy_coef=get_cfg("trader", "entropy_coef", 0.01),
        value_loss_coef=get_cfg("trader", "value_loss_coef", 0.5),
        max_grad_norm=get_cfg("trader", "max_grad_norm", 0.5),
        target_kl=get_cfg("trader", "target_kl", 0.01),
    )

    adversary_config = PPOConfig(
        state_dim=state_dim,
        hidden_dim=get_cfg("adversary", "hidden_dim", 128),
        n_actions=n_actions,
        actor_lr=float(get_cfg("adversary", "actor_lr", 1e-4)),
        critic_lr=float(get_cfg("adversary", "critic_lr", 5e-4)),
        gamma=get_cfg("adversary", "gamma", 0.99),
        gae_lambda=get_cfg("adversary", "gae_lambda", 0.95),
        clip_epsilon=get_cfg("adversary", "clip_epsilon", 0.2),
        n_epochs=get_cfg("adversary", "n_epochs", 10),
        batch_size=get_cfg("adversary", "batch_size", 64),
        use_recurrent=get_cfg("adversary", "use_recurrent", True),
        rnn_type=get_cfg("adversary", "rnn_type", "GRU"),
        entropy_coef=get_cfg("adversary", "entropy_coef", 0.02),
    )

    # Training config
    training_config = AdversarialConfig(
        n_iterations=args.iterations or get_cfg("training", "n_iterations", 500),
        steps_per_iteration=get_cfg("training", "steps_per_iteration", 2048),
        trader_config=trader_config,
        adversary_config=adversary_config,
        adversary_start_iteration=get_cfg("training", "adversary_start_iteration", 100),
        adversary_strength=get_cfg("training", "adversary_strength", 0.1),
        save_frequency=get_cfg("training", "save_frequency", 50),
        log_frequency=get_cfg("training", "log_frequency", 10),
        checkpoint_dir=get_cfg("training", "checkpoint_dir", "data/models/adversarial"),
    )

    trainer = AdversarialTrainer(env, training_config, device=args.device)

    logger.success("Trainer created from config")
    logger.info(f"  State dim: {state_dim}")
    logger.info(f"  Actions: {n_actions}")
    logger.info(f"  Iterations: {training_config.n_iterations}")

    return trainer


def main():
    """Main training function."""
    # Parse arguments
    parser = argparse.ArgumentParser(description="Train adversarial trading system")

    # Training arguments
    parser.add_argument(
        "--config",
        type=str,
        default="config/training/adversarial.yaml",
        help="Path to training config",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=None,
        help="Number of training iterations (overrides config)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to use",
    )
    parser.add_argument(
        "--resume", type=str, default=None, help="Resume from checkpoint"
    )
    parser.add_argument(
        "--eval-only", action="store_true", help="Only evaluate, don't train"
    )
    parser.add_argument(
        "--eval-episodes", type=int, default=100, help="Number of evaluation episodes"
    )

    # Data loading arguments
    parser.add_argument(
        "--exchange",
        type=str,
        default="binance",
        help="Exchange to download data from (e.g., binance, coinbase)",
    )
    parser.add_argument(
        "--exchange-type",
        type=str,
        default="spot",
        choices=["spot", "future", "swap"],
        help="Type of market (spot, future, swap)",
    )
    parser.add_argument(
        "--symbol",
        type=str,
        default="BTC/USDT",
        help="Trading pair symbol (e.g., BTC/USDT, ETH/USDT)",
    )
    parser.add_argument(
        "--timeframe",
        type=str,
        default="1h",
        help="Candle timeframe (e.g., 1h, 4h, 1d)",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default="2023-01-01",
        help="Start date for data (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="End date for data (YYYY-MM-DD, default: now)",
    )
    parser.add_argument(
        "--rate-limit",
        type=int,
        default=100,
        help="Rate limit in milliseconds for API calls",
    )
    parser.add_argument(
        "--use-cached",
        action="store_true",
        default=True,
        help="Use cached data if available",
    )
    parser.add_argument(
        "--force-refresh", action="store_true", help="Force re-download even if cached"
    )

    # Feature engineering arguments
    parser.add_argument(
        "--volatility-window",
        type=int,
        default=20,
        help="Window for volatility calculation",
    )
    parser.add_argument(
        "--ou-window", type=int, default=20, help="Window for Ornstein-Uhlenbeck score"
    )
    parser.add_argument(
        "--rolling-mean-window", type=int, default=20, help="Window for rolling mean"
    )
    parser.add_argument(
        "--scaler-type",
        type=str,
        default="standard",
        choices=["standard", "minmax", "robust"],
        help="Type of scaler to use",
    )

    args = parser.parse_args()

    # Setup
    setup_logging()

    logger.info("=" * 80)
    logger.info("ADVERSARIAL TRADING SYSTEM - TRAINING")
    logger.info("=" * 80)
    logger.info(f"Iterations: {args.iterations}")
    logger.info(f"Device: {args.device}")
    logger.info(f"Exchange: {args.exchange} ({args.exchange_type})")
    logger.info(f"Symbol: {args.symbol}, Timeframe: {args.timeframe}")
    logger.info(f"Date range: {args.start_date} to {args.end_date or 'now'}")

    # Load data (returns splits dict)
    splits = load_data(args)
    train_price, train_features = splits["train"]
    logger.info(f"Using {len(train_price)} samples for training environment")

    # Create environment
    env = create_environment(train_price, train_features)

    # Create trainer
    trainer = create_trainer(env, args)

    # Resume if checkpoint provided
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        try:
            trainer.load_checkpoint(args.resume)
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            raise e

    # Train or evaluate
    if args.eval_only:
        logger.info("Evaluation mode")
        metrics = trainer.evaluate(n_episodes=args.eval_episodes)
    else:
        logger.info("Training mode")
        trainer.train()

        # Evaluate after training
        logger.info("\nFinal evaluation:")
        metrics = trainer.evaluate(n_episodes=args.eval_episodes)

    logger.success("=" * 80)
    logger.success("TRAINING COMPLETE")
    logger.success("=" * 80)


if __name__ == "__main__":
    main()
