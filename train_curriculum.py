"""
Curriculum Training Script
==========================
3-phase curriculum training for BITCOIN4Traders PPO agent.

Problem being solved:
---------------------
A naive PPO agent converges to a 100% Short policy because the reward
signal has a market-direction bias on historical bear data.  Curriculum
Learning breaks this by constraining the action space in early phases so
the agent must first master directional trading before being exposed to
the full, unmasked action space.

Curriculum Phases:
------------------
  Phase 1 – Long-only  (actions 3,4,5,6):
      Agent learns to manage long positions.  Regime-Aware Reward already
      penalises holding longs during bear regimes, so the agent is pushed
      toward timing entry/exit rather than pure buy-and-hold.
      Default: 200 iterations.

  Phase 2 – Short-only (actions 0,1,2):
      Agent learns short-selling mechanics (cost, drawdown risk).
      Including Neutral (2) so the agent can hedge by going flat.
      Default: 200 iterations.

  Phase 3 – Full space (actions 0-6):
      All masks removed; agent must balance both directions.
      Regime-Aware Reward provides alignment bonus/penalty.
      Default: 400 iterations.

After each phase a ModelBenchmark evaluation is run and results are
printed.  The best checkpoint (highest decision-matrix score) is saved to
`data/models/curriculum/best_model.pth`.

Usage:
------
  # Default run (800 total iterations)
  python train_curriculum.py

  # Custom iteration counts
  python train_curriculum.py --phase1 100 --phase2 100 --phase3 200

  # Load cached data, skip download
  python train_curriculum.py --use-cached

  # GPU training
  python train_curriculum.py --device cuda

  # Resume from a phase checkpoint
  python train_curriculum.py --resume data/models/curriculum/phase2_final.pth --start-phase 3

Author: BITCOIN4Traders Team
Version: 1.0.0
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# ── path bootstrap ────────────────────────────────────────────────────────────
# Ensure src/ is on sys.path so bare-package imports work when running
# this script directly (python train_curriculum.py), identical to train.py.
_SRC = Path(__file__).parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import numpy as np
import pandas as pd
from loguru import logger

# ── project imports ───────────────────────────────────────────────────────────
from environment.config_integrated_env import ConfigIntegratedTradingEnv
from environment.config_system import (
    EnvironmentConfig,
    load_environment_config_from_yaml,
)
from agents.ppo_agent import PPOAgent, PPOConfig
from training.adversarial_trainer import AdversarialTrainer, AdversarialConfig
from testing.benchmark import ModelBenchmark, BenchmarkResult


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

# Action buckets for each curriculum phase
PHASE_ACTIONS = {
    1: [3, 4, 5, 6],  # Long-only:  Long33%, Long50%, Long75%, Long100%
    2: [0, 1, 2],  # Short-only: Short100%, Short50%, Neutral
    3: None,  # Full space: all 7 actions unrestricted
}

PHASE_NAMES = {
    1: "Phase 1 – Long-only",
    2: "Phase 2 – Short-only",
    3: "Phase 3 – Full action space",
}

CHECKPOINT_DIR = Path("data/models/curriculum")


# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────


def setup_logging() -> None:
    """Configure loguru for console + file output."""
    logger.remove()
    logger.add(
        sys.stderr,
        format=(
            "<green>{time:HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan> – <level>{message}</level>"
        ),
        level="INFO",
    )
    log_path = Path("logs/curriculum")
    log_path.mkdir(parents=True, exist_ok=True)
    logger.add(
        log_path / "curriculum_{time}.log",
        rotation="50 MB",
        retention="14 days",
        level="DEBUG",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Data loading  (reuses logic from train.py)
# ─────────────────────────────────────────────────────────────────────────────


def load_data(args: argparse.Namespace) -> dict:
    """
    Load price data and compute features.

    Returns a dict with keys 'train', 'val', 'test', each a tuple
    (price_data: DataFrame, features: DataFrame).
    """
    cache_dir = Path("data/cache")
    processed_dir = Path("data/processed")

    # ── Try cached data first ─────────────────────────────────────────────
    if args.use_cached and cache_dir.exists():
        cached = sorted(
            cache_dir.glob("*.parquet"), key=lambda p: p.stat().st_mtime, reverse=True
        )
        feat_cached = (
            sorted(
                processed_dir.glob("*_features.parquet"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            if processed_dir.exists()
            else []
        )

        if cached and feat_cached:
            logger.info(f"Loading cached price data: {cached[0]}")
            price_data = pd.read_parquet(cached[0])
            logger.info(f"Loading cached features:   {feat_cached[0]}")
            features = pd.read_parquet(feat_cached[0])

            common = price_data.index.intersection(features.index)
            price_data = price_data.loc[common]
            features = features.loc[common]

            n = len(price_data)
            i1, i2 = int(n * 0.70), int(n * 0.85)
            splits = {
                "train": (price_data.iloc[:i1], features.iloc[:i1]),
                "val": (price_data.iloc[i1:i2], features.iloc[i1:i2]),
                "test": (price_data.iloc[i2:], features.iloc[i2:]),
            }
            logger.success(
                f"Cache loaded: train={i1}, val={i2 - i1}, test={n - i2} rows"
            )
            return splits

    # ── Download from exchange ────────────────────────────────────────────
    logger.info(f"Downloading {args.symbol} {args.timeframe} from {args.exchange}…")
    try:
        from data.ccxt_loader import CCXTDataLoader, DataLoaderConfig

        loader_cfg = DataLoaderConfig(
            exchange_id=args.exchange,
            exchange_type=args.exchange_type,
            rate_limit_ms=args.rate_limit,
            cache_dir=cache_dir,
            processed_dir=processed_dir,
        )
        loader = CCXTDataLoader(loader_cfg)
        price_data = loader.download_and_cache(
            symbol=args.symbol,
            timeframe=args.timeframe,
            start_date=args.start_date,
            end_date=args.end_date,
            force_refresh=args.force_refresh,
        )
        logger.success(f"Downloaded {len(price_data)} candles")
    except Exception as exc:
        logger.warning(f"Download failed ({exc}), using synthetic fallback data")
        price_data = _synthetic_data(n=10_000)

    # ── Feature engineering ───────────────────────────────────────────────
    logger.info("Computing features with FeatureEngine…")
    from features.feature_engine import FeatureEngine, FeatureConfig

    feat_cfg = FeatureConfig(
        volatility_window=args.volatility_window,
        ou_window=args.ou_window,
        rolling_mean_window=args.rolling_mean_window,
        use_log_returns=True,
        scaler_type=args.scaler_type,
        save_scaler=True,
        scaler_path=processed_dir,
        dropna_strategy="rolling",
        min_valid_rows=500,
    )
    engine = FeatureEngine(feat_cfg)

    n = len(price_data)
    i1, i2 = int(n * 0.70), int(n * 0.85)

    train_p = price_data.iloc[:i1]
    val_p = price_data.iloc[i1:i2]
    test_p = price_data.iloc[i2:]

    train_f = engine.fit_transform(train_p)
    val_f = engine.transform(val_p)
    test_f = engine.transform(test_p)

    # Align indexes (feature engine may drop NaN rows)
    c_tr = train_p.index.intersection(train_f.index)
    c_va = val_p.index.intersection(val_f.index)
    c_te = test_p.index.intersection(test_f.index)

    splits = {
        "train": (train_p.loc[c_tr], train_f.loc[c_tr]),
        "val": (val_p.loc[c_va], val_f.loc[c_va]),
        "test": (test_p.loc[c_te], test_f.loc[c_te]),
    }
    logger.success(
        f"Features ready: train={len(c_tr)}, val={len(c_va)}, test={len(c_te)}"
    )
    return splits


def _synthetic_data(n: int = 10_000, seed: int = 42) -> pd.DataFrame:
    """Minimal synthetic OHLCV for offline development / CI."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2022-01-01", periods=n, freq="1h")
    close = 40_000 + np.cumsum(rng.normal(0, 100, n))
    noise = rng.normal(0, 50, n)
    return pd.DataFrame(
        {
            "open": close + noise,
            "high": close + np.abs(noise) + rng.uniform(0, 200, n),
            "low": close - np.abs(noise) - rng.uniform(0, 200, n),
            "close": close,
            "volume": rng.uniform(100, 2000, n),
        },
        index=dates,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Environment factory
# ─────────────────────────────────────────────────────────────────────────────


def make_env(
    price_data: pd.DataFrame, features: pd.DataFrame
) -> ConfigIntegratedTradingEnv:
    """Load YAML config and construct the trading environment."""
    cfg_path = Path("config/environment/realistic_env.yaml")
    if cfg_path.exists():
        env_config = load_environment_config_from_yaml(str(cfg_path))
    else:
        logger.warning(
            "realistic_env.yaml not found – using EnvironmentConfig defaults"
        )
        env_config = EnvironmentConfig()

    return ConfigIntegratedTradingEnv(
        price_data=price_data,
        features=features,
        config=env_config,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Trainer factory
# ─────────────────────────────────────────────────────────────────────────────


def make_trainer(
    env: ConfigIntegratedTradingEnv,
    n_iterations: int,
    device: str,
    checkpoint_subdir: str = "curriculum",
) -> AdversarialTrainer:
    """Build an AdversarialTrainer wired to the given env."""
    state_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    trader_cfg = PPOConfig(
        state_dim=state_dim,
        hidden_dim=128,
        n_actions=n_actions,
        actor_lr=3e-4,
        critic_lr=1e-3,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        n_epochs=10,
        batch_size=64,
        use_recurrent=True,
        rnn_type="GRU",
        entropy_coef=0.01,
        value_loss_coef=0.5,
        max_grad_norm=0.5,
        target_kl=0.01,
    )
    adversary_cfg = PPOConfig(
        state_dim=state_dim,
        hidden_dim=128,
        n_actions=n_actions,
        actor_lr=1e-4,
        critic_lr=5e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        n_epochs=10,
        batch_size=64,
        use_recurrent=True,
        rnn_type="GRU",
        entropy_coef=0.02,
    )
    training_cfg = AdversarialConfig(
        n_iterations=n_iterations,
        steps_per_iteration=2048,
        trader_config=trader_cfg,
        adversary_config=adversary_cfg,
        adversary_start_iteration=max(50, n_iterations // 4),
        adversary_strength=0.1,
        save_frequency=max(10, n_iterations // 10),
        log_frequency=10,
        checkpoint_dir=str(CHECKPOINT_DIR / checkpoint_subdir),
    )
    return AdversarialTrainer(env, training_cfg, device=device)


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark helper
# ─────────────────────────────────────────────────────────────────────────────


def run_benchmark(
    trainer: AdversarialTrainer,
    val_price: pd.DataFrame,
    val_features: pd.DataFrame,
    phase: int,
    n_episodes: int = 3,
) -> BenchmarkResult:
    """Evaluate the trainer's Trader on the validation set."""
    val_env = make_env(val_price, val_features)
    val_env.set_allowed_actions(PHASE_ACTIONS[phase])

    bench = ModelBenchmark(val_env, trainer.trader, n_actions=7)
    result = bench.run(n_episodes=n_episodes)
    logger.info(f"\n{result.summary()}")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Phase runner
# ─────────────────────────────────────────────────────────────────────────────


def run_phase(
    phase: int,
    n_iterations: int,
    train_price: pd.DataFrame,
    train_features: pd.DataFrame,
    val_price: pd.DataFrame,
    val_features: pd.DataFrame,
    device: str,
    resume_path: Optional[str] = None,
    prev_trainer: Optional[AdversarialTrainer] = None,
) -> tuple[AdversarialTrainer, BenchmarkResult]:
    """
    Run one curriculum phase.

    If prev_trainer is provided, the Trader weights are transferred as a
    warm-start (hot-start).  This preserves learned policy across phases.

    Returns (trainer, benchmark_result).
    """
    logger.info("=" * 60)
    logger.info(f"  {PHASE_NAMES[phase]}  ({n_iterations} iterations)")
    allowed = PHASE_ACTIONS[phase]
    logger.info(f"  Allowed actions: {allowed if allowed is not None else 'all (0-6)'}")
    logger.info("=" * 60)

    # Build env with phase mask
    env = make_env(train_price, train_features)
    env.set_allowed_actions(allowed)

    # Build trainer
    trainer = make_trainer(
        env,
        n_iterations=n_iterations,
        device=device,
        checkpoint_subdir=f"phase{phase}",
    )

    # Transfer weights from previous phase (warm-start)
    if prev_trainer is not None:
        logger.info(f"  Warm-start: transferring Trader weights from Phase {phase - 1}")
        trainer.trader.actor.load_state_dict(prev_trainer.trader.actor.state_dict())
        trainer.trader.critic.load_state_dict(prev_trainer.trader.critic.state_dict())

    # Resume from explicit checkpoint
    if resume_path and phase == 1:
        logger.info(f"  Resuming from: {resume_path}")
        try:
            trainer.load_checkpoint(resume_path)
        except Exception as exc:
            logger.error(f"  Failed to load checkpoint: {exc}")

    t0 = time.perf_counter()
    trainer.train()
    elapsed = time.perf_counter() - t0
    logger.info(f"  Phase {phase} training done in {elapsed / 60:.1f} min")

    # Save phase checkpoint
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = CHECKPOINT_DIR / f"phase{phase}_final.pth"
    trainer.save_checkpoint(str(ckpt_path))
    logger.info(f"  Checkpoint saved: {ckpt_path}")

    # Benchmark on validation set
    logger.info(f"  Benchmarking Phase {phase} on validation set…")
    result = run_benchmark(trainer, val_price, val_features, phase=phase)

    return trainer, result


# ─────────────────────────────────────────────────────────────────────────────
# Final evaluation on test set
# ─────────────────────────────────────────────────────────────────────────────


def final_evaluation(
    trainer: AdversarialTrainer,
    test_price: pd.DataFrame,
    test_features: pd.DataFrame,
    n_episodes: int = 5,
) -> BenchmarkResult:
    """Run final out-of-sample evaluation (full action space, no mask)."""
    logger.info("=" * 60)
    logger.info("  FINAL OUT-OF-SAMPLE EVALUATION (Test set)")
    logger.info("=" * 60)

    test_env = make_env(test_price, test_features)
    # Phase 3 = no mask
    test_env.set_allowed_actions(None)

    bench = ModelBenchmark(test_env, trainer.trader, n_actions=7)
    result = bench.run(n_episodes=n_episodes)
    logger.info(f"\n{result.summary()}")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="3-phase curriculum training for BITCOIN4Traders",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Curriculum schedule
    g = p.add_argument_group("Curriculum schedule")
    g.add_argument(
        "--phase1",
        type=int,
        default=200,
        metavar="ITER",
        help="Iterations for Phase 1 (Long-only)",
    )
    g.add_argument(
        "--phase2",
        type=int,
        default=200,
        metavar="ITER",
        help="Iterations for Phase 2 (Short-only)",
    )
    g.add_argument(
        "--phase3",
        type=int,
        default=400,
        metavar="ITER",
        help="Iterations for Phase 3 (Full space)",
    )
    g.add_argument(
        "--start-phase",
        type=int,
        default=1,
        choices=[1, 2, 3],
        help="Start from this phase (needs --resume for phase 2/3)",
    )
    g.add_argument(
        "--eval-episodes",
        type=int,
        default=3,
        help="Validation episodes per phase benchmark",
    )
    g.add_argument(
        "--final-episodes",
        type=int,
        default=5,
        help="Episodes for final test-set evaluation",
    )

    # Hardware
    g2 = p.add_argument_group("Hardware")
    g2.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Torch device",
    )

    # Checkpoint
    g3 = p.add_argument_group("Checkpoint")
    g3.add_argument(
        "--resume",
        type=str,
        default=None,
        metavar="PATH",
        help="Resume a trainer checkpoint (used with --start-phase)",
    )

    # Data
    g4 = p.add_argument_group("Data")
    g4.add_argument("--exchange", type=str, default="binance")
    g4.add_argument(
        "--exchange-type", type=str, default="spot", choices=["spot", "future", "swap"]
    )
    g4.add_argument("--symbol", type=str, default="BTC/USDT")
    g4.add_argument("--timeframe", type=str, default="1h")
    g4.add_argument("--start-date", type=str, default="2022-01-01")
    g4.add_argument("--end-date", type=str, default=None)
    g4.add_argument("--rate-limit", type=int, default=100)
    g4.add_argument("--use-cached", action="store_true", default=True)
    g4.add_argument("--force-refresh", action="store_true")

    # Feature engineering
    g5 = p.add_argument_group("Feature engineering")
    g5.add_argument("--volatility-window", type=int, default=20)
    g5.add_argument("--ou-window", type=int, default=20)
    g5.add_argument("--rolling-mean-window", type=int, default=20)
    g5.add_argument(
        "--scaler-type",
        type=str,
        default="standard",
        choices=["standard", "minmax", "robust"],
    )

    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


def main() -> None:
    args = parse_args()
    setup_logging()

    logger.info("=" * 60)
    logger.info("  BITCOIN4Traders – Curriculum Training")
    logger.info(f"  Phases: {args.phase1} / {args.phase2} / {args.phase3} iterations")
    logger.info(f"  Device: {args.device}  |  Start phase: {args.start_phase}")
    logger.info("=" * 60)

    # ── Data ──────────────────────────────────────────────────────────────
    splits = load_data(args)
    train_price, train_features = splits["train"]
    val_price, val_features = splits["val"]
    test_price, test_features = splits["test"]

    phase_iters = {1: args.phase1, 2: args.phase2, 3: args.phase3}
    phase_results: dict[int, BenchmarkResult] = {}
    best_score = -1.0
    best_ckpt: Optional[Path] = None
    prev_trainer: Optional[AdversarialTrainer] = None

    # ── Phase loop ────────────────────────────────────────────────────────
    for phase in range(args.start_phase, 4):
        n_iter = phase_iters[phase]

        # Only pass resume_path for the very first phase we run
        resume = args.resume if phase == args.start_phase else None

        trainer, result = run_phase(
            phase=phase,
            n_iterations=n_iter,
            train_price=train_price,
            train_features=train_features,
            val_price=val_price,
            val_features=val_features,
            device=args.device,
            resume_path=resume,
            prev_trainer=prev_trainer,
        )

        phase_results[phase] = result
        prev_trainer = trainer

        # Track best model
        score = result.decision_matrix_score
        if score > best_score:
            best_score = score
            best_ckpt = CHECKPOINT_DIR / f"phase{phase}_final.pth"
            best_dest = CHECKPOINT_DIR / "best_model.pth"
            CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
            import shutil

            shutil.copy2(str(best_ckpt), str(best_dest))
            logger.info(f"  New best model (score={score:.1f}) saved to {best_dest}")

    # ── Final OOS evaluation ──────────────────────────────────────────────
    if prev_trainer is not None:
        oos_result = final_evaluation(
            prev_trainer,
            test_price,
            test_features,
            n_episodes=args.final_episodes,
        )
    else:
        logger.warning("No trainer available for final evaluation (start-phase > 3?)")
        return

    # ── Summary ───────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("  CURRICULUM TRAINING COMPLETE – Phase scores")
    logger.info("=" * 60)
    for ph, res in sorted(phase_results.items()):
        logger.info(
            f"  {PHASE_NAMES[ph]:<35}  score={res.decision_matrix_score:.1f}"
            f"  return={res.mean_return * 100:+.2f}%"
            f"  dd={res.mean_max_drawdown * 100:.1f}%"
        )
    logger.info(
        f"  Final OOS                               score={oos_result.decision_matrix_score:.1f}"
        f"  return={oos_result.mean_return * 100:+.2f}%"
        f"  dd={oos_result.mean_max_drawdown * 100:.1f}%"
    )
    logger.info(f"\n  Best model: {best_dest}  (val score={best_score:.1f})")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
