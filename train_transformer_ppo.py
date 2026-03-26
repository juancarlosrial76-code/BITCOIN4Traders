"""
Transformer-PPO Training Script
================================
Dedicated entry point for training with the TransformerBackbone instead of GRU/LSTM.

Supports both operating modes:

  Option A  — Backbone swap only (transformer_seq_len = 1)
    Each step is fed as a single-step "sequence" [batch, 1, state_dim].
    The transformer attends across the MLP-projected features of the current
    observation.  Drop-in replacement for GRU with roughly the same compute.
    Use this for quick experimentation.

  Option B  — Full sequence context (transformer_seq_len > 1, default: 16)
    The agent maintains a sliding window of the last K raw states per episode.
    At each step the window [1, K, state_dim] is fed to the transformer so it
    can attend across real temporal history.  Call agent.reset_sequence_window()
    at every episode reset.  This is where the transformer's inductive bias
    (global attention vs. GRU's local hidden state) pays off.

Usage
-----
# Option A — backbone swap, seq_len=1
python train_transformer_ppo.py --mode A

# Option B — full sequence context, window of 16 steps
python train_transformer_ppo.py --mode B --seq-len 16

# Custom architecture
python train_transformer_ppo.py --mode B --seq-len 32 --nhead 8 --d-model 256 --layers 4

# Use GPU
python train_transformer_ppo.py --mode B --device cuda

# Full adversarial loop (Transformer vs Adversary)
python train_transformer_ppo.py --mode B --adversarial --iterations 500

Configuration precedence
------------------------
  CLI flags > config/training/adversarial_transformer.yaml > hardcoded defaults

The script writes to  data/models/transformer_{A|B}/  by default.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from loguru import logger

from src.agents.ppo_agent import PPOAgent, PPOConfig
from src.data.ccxt_loader import CCXTDataLoader
from src.environment.config_integrated_env import ConfigIntegratedTradingEnv
from src.environment.config_system import (
    EnvironmentConfig,
    load_environment_config_from_yaml,
)
from src.features.feature_engine import FeatureConfig, FeatureEngine
from src.training.adversarial_trainer import AdversarialConfig, AdversarialTrainer


# ──────────────────────────────────────────────────────────────────────────────
# Defaults
# ──────────────────────────────────────────────────────────────────────────────

DEFAULTS = {
    # Transformer architecture
    "hidden_dim": 128,
    "transformer_nhead": 4,
    "transformer_d_model": 0,  # 0 → use hidden_dim
    "rnn_layers": 2,  # transformer depth (num_encoder_layers)
    "dropout": 0.1,
    # PPO
    "actor_lr": 5e-5,  # lower than GRU — transformers are more sensitive
    "critic_lr": 2e-4,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_epsilon": 0.2,
    "n_epochs": 10,
    "batch_size": 64,
    "entropy_coef": 0.05,
    "value_loss_coef": 0.5,
    "max_grad_norm": 0.5,
    "target_kl": 0.02,  # slightly looser for transformer warmup
    "lr_decay_gamma": 0.995,
    # Training loop
    "n_iterations": 300,
    "steps_per_iteration": 2048,
    "save_frequency": 50,
    "log_frequency": 10,
    "adversary_start_iteration": 100,
    "adversary_strength": 0.1,
}


# ──────────────────────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────────────────────


def load_data(args) -> dict:
    """Load and return train/val/test splits."""
    logger.info("Loading data…")
    cache_dir = Path("data/cache")

    cached = sorted(
        cache_dir.glob("*.parquet"), key=lambda p: p.stat().st_size, reverse=True
    )
    if cached:
        price_data = pd.read_parquet(cached[0])
        logger.info(f"Using cached file: {cached[0].name}  ({len(price_data)} rows)")
    else:
        loader = CCXTDataLoader(exchange=args.exchange, symbol=args.symbol)
        price_data = loader.fetch_ohlcv(timeframe=args.timeframe, since=args.start_date)

    feature_config = FeatureConfig(
        volatility_window=20,
        ou_window=50,
        rolling_mean_window=20,
        use_log_returns=True,
        scaler_type="standard",
        save_scaler=False,
        scaler_path=Path("data/scalers"),
        dropna_strategy="rolling",
        min_valid_rows=1000,
    )
    engine = FeatureEngine(config=feature_config)

    n = len(price_data)
    t0, t1 = int(n * 0.70), int(n * 0.85)

    train_price = price_data.iloc[:t0]
    val_price = price_data.iloc[t0:t1]
    test_price = price_data.iloc[t1:]

    # Fit on TRAIN only to prevent data leakage
    train_features = engine.fit_transform(train_price)
    val_features = engine.transform(val_price)
    test_features = engine.transform(test_price)

    splits = {
        "train": (train_price, train_features),
        "val": (val_price, val_features),
        "test": (test_price, test_features),
    }
    logger.success(
        f"Split: train={t0}  val={t1-t0}  test={n-t1}  "
        f"features={train_features.shape[1]}"
    )
    return splits


# ──────────────────────────────────────────────────────────────────────────────
# Environment
# ──────────────────────────────────────────────────────────────────────────────


def make_env(
    price_data: pd.DataFrame, features: pd.DataFrame
) -> ConfigIntegratedTradingEnv:
    cfg_path = Path("config/environment/realistic_env.yaml")
    env_cfg = (
        load_environment_config_from_yaml(str(cfg_path))
        if cfg_path.exists()
        else EnvironmentConfig()
    )
    return ConfigIntegratedTradingEnv(price_data, features, env_cfg)


# ──────────────────────────────────────────────────────────────────────────────
# Build PPOConfig for transformer
# ──────────────────────────────────────────────────────────────────────────────


def build_transformer_config(
    state_dim: int,
    n_actions: int,
    args,
    cfg: dict,
    mode: str,
) -> PPOConfig:
    """Construct a PPOConfig with the transformer backbone active."""

    def g(key):
        """Get value from YAML cfg → CLI → defaults (precedence: CLI > YAML > defaults)."""
        cli_val = getattr(args, key.replace("-", "_"), None)
        if cli_val is not None:
            return cli_val
        return cfg.get(key, DEFAULTS.get(key))

    seq_len = args.seq_len if mode == "B" else 1

    config = PPOConfig(
        state_dim=state_dim,
        n_actions=n_actions,
        hidden_dim=g("hidden_dim"),
        # Backbone
        use_transformer=True,
        use_recurrent=False,
        rnn_layers=g("rnn_layers"),  # reused as num_transformer_layers
        dropout=g("dropout"),
        use_layer_norm=True,
        # Transformer-specific
        transformer_nhead=g("transformer_nhead"),
        transformer_d_model=g("transformer_d_model"),
        transformer_seq_len=seq_len,
        # PPO
        actor_lr=float(g("actor_lr")),
        critic_lr=float(g("critic_lr")),
        gamma=g("gamma"),
        gae_lambda=g("gae_lambda"),
        clip_epsilon=g("clip_epsilon"),
        n_epochs=g("n_epochs"),
        batch_size=g("batch_size"),
        entropy_coef=g("entropy_coef"),
        value_loss_coef=g("value_loss_coef"),
        max_grad_norm=g("max_grad_norm"),
        target_kl=g("target_kl"),
        use_lr_decay=True,
        lr_decay_gamma=g("lr_decay_gamma"),
        use_amp=True,
    )
    return config


# ──────────────────────────────────────────────────────────────────────────────
# Simple standalone rollout loop (non-adversarial)
# ──────────────────────────────────────────────────────────────────────────────


def run_standalone(
    agent: PPOAgent,
    env: ConfigIntegratedTradingEnv,
    args,
    cfg: dict,
    checkpoint_dir: Path,
    mode: str,
) -> None:
    """
    Collect trajectories with PPOAgent and call agent.train() — no adversary.

    This is the simplest path: one agent, one environment, standard PPO updates.
    """
    n_iterations = args.iterations or cfg.get("n_iterations", DEFAULTS["n_iterations"])
    steps_per_iter = cfg.get("steps_per_iteration", DEFAULTS["steps_per_iteration"])
    save_freq = cfg.get("save_frequency", DEFAULTS["save_frequency"])
    log_freq = cfg.get("log_frequency", DEFAULTS["log_frequency"])

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    agent.reset_buffers(capacity=steps_per_iter)

    state, _ = env.reset()
    if mode == "B":
        agent.reset_sequence_window()

    total_steps = 0
    episode_reward = 0.0
    episode_rewards = []

    for iteration in range(1, n_iterations + 1):
        # ── Collect trajectory ────────────────────────────────────────────────
        for _ in range(steps_per_iter):
            action, log_prob, value, _ = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.store_transition(state, action, reward, log_prob, value, done)
            episode_reward += reward
            total_steps += 1

            if done:
                episode_rewards.append(episode_reward)
                episode_reward = 0.0
                state, _ = env.reset()
                if mode == "B":
                    agent.reset_sequence_window()  # ← Option B: zero-fill window
            else:
                state = next_state

        # ── PPO update ────────────────────────────────────────────────────────
        # Bootstrap value for the last incomplete episode
        with torch.no_grad():
            _, _, bootstrap_value, _ = agent.select_action(state, deterministic=True)

        stats = agent.train(next_value=bootstrap_value)
        agent.reset_buffers(capacity=steps_per_iter)

        # ── Logging ───────────────────────────────────────────────────────────
        if iteration % log_freq == 0:
            mean_ep_r = np.mean(episode_rewards[-20:]) if episode_rewards else 0.0
            logger.info(
                f"[Iter {iteration:>4}/{n_iterations}]  "
                f"steps={total_steps:>7}  "
                f"ep_reward={mean_ep_r:+.3f}  "
                f"actor_loss={stats.get('actor_loss', 0):.4f}  "
                f"critic_loss={stats.get('critic_loss', 0):.4f}  "
                f"entropy={stats.get('entropy', 0):.4f}"
            )

        # ── Checkpoint ────────────────────────────────────────────────────────
        if iteration % save_freq == 0:
            ckpt = checkpoint_dir / f"checkpoint_iter_{iteration:04d}.pth"
            agent.save_checkpoint(str(ckpt))
            logger.info(f"  Saved checkpoint → {ckpt}")

    logger.success(
        f"Training complete.  Total steps={total_steps}  "
        f"Mean reward (last 20 ep)={np.mean(episode_rewards[-20:]) if episode_rewards else 0:.4f}"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Adversarial loop
# ──────────────────────────────────────────────────────────────────────────────


def run_adversarial(
    trader_config: PPOConfig,
    env: ConfigIntegratedTradingEnv,
    args,
    cfg: dict,
    checkpoint_dir: Path,
) -> None:
    """Run the full AdversarialTrainer loop with a Transformer trader."""
    from src.training.adversarial_trainer import AdversarialConfig, AdversarialTrainer

    # Adversary keeps GRU (simpler, faster, fewer params to avoid dominating)
    adversary_config = PPOConfig(
        state_dim=trader_config.state_dim,
        n_actions=trader_config.n_actions,
        hidden_dim=cfg.get("hidden_dim", DEFAULTS["hidden_dim"]),
        use_recurrent=True,
        rnn_type="GRU",
        actor_lr=float(cfg.get("adversary_actor_lr", 5e-5)),
        critic_lr=float(cfg.get("adversary_critic_lr", 2e-4)),
        gamma=cfg.get("gamma", DEFAULTS["gamma"]),
        gae_lambda=cfg.get("gae_lambda", DEFAULTS["gae_lambda"]),
        clip_epsilon=cfg.get("clip_epsilon", DEFAULTS["clip_epsilon"]),
        n_epochs=cfg.get("n_epochs", DEFAULTS["n_epochs"]),
        batch_size=cfg.get("batch_size", DEFAULTS["batch_size"]),
        entropy_coef=cfg.get("adversary_entropy_coef", 0.05),
    )

    training_config = AdversarialConfig(
        n_iterations=args.iterations
        or cfg.get("n_iterations", DEFAULTS["n_iterations"]),
        steps_per_iteration=cfg.get(
            "steps_per_iteration", DEFAULTS["steps_per_iteration"]
        ),
        trader_config=trader_config,
        adversary_config=adversary_config,
        adversary_start_iteration=cfg.get(
            "adversary_start_iteration", DEFAULTS["adversary_start_iteration"]
        ),
        adversary_strength=cfg.get(
            "adversary_strength", DEFAULTS["adversary_strength"]
        ),
        save_frequency=cfg.get("save_frequency", DEFAULTS["save_frequency"]),
        log_frequency=cfg.get("log_frequency", DEFAULTS["log_frequency"]),
        checkpoint_dir=str(checkpoint_dir),
    )

    trainer = AdversarialTrainer(env, training_config, device=args.device)
    trainer.train()


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train PPO with Transformer backbone (Option A or B)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Mode
    p.add_argument(
        "--mode",
        choices=["A", "B"],
        default="B",
        help=(
            "A = backbone-swap only (seq_len=1, Option A). "
            "B = sliding-window sequence context (Option B, requires --seq-len)."
        ),
    )
    p.add_argument(
        "--seq-len",
        type=int,
        default=16,
        help="Sliding-window length K for Option B (ignored in mode A).",
    )

    # Architecture
    p.add_argument(
        "--hidden-dim",
        type=int,
        default=None,
        help="MLP hidden dim / transformer d_model",
    )
    p.add_argument(
        "--nhead",
        type=int,
        default=None,
        dest="transformer_nhead",
        help="Number of attention heads (must divide d-model evenly)",
    )
    p.add_argument(
        "--d-model",
        type=int,
        default=None,
        dest="transformer_d_model",
        help="Transformer internal width (0 = same as hidden-dim)",
    )
    p.add_argument(
        "--layers",
        type=int,
        default=None,
        dest="rnn_layers",
        help="Number of transformer encoder layers",
    )
    p.add_argument("--dropout", type=float, default=None)

    # Training
    p.add_argument("--iterations", type=int, default=None, help="Override n_iterations")
    p.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cpu", "cuda"],
    )
    p.add_argument(
        "--adversarial",
        action="store_true",
        help="Use full AdversarialTrainer instead of standalone PPO loop",
    )
    p.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config (overrides built-in defaults)",
    )
    p.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="Directory for saving checkpoints",
    )

    # Data
    p.add_argument("--exchange", type=str, default="binance")
    p.add_argument("--symbol", type=str, default="BTC/USDT")
    p.add_argument("--timeframe", type=str, default="1h")
    p.add_argument("--start-date", type=str, default="2020-01-01")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ── Logging ───────────────────────────────────────────────────────────────
    logger.remove()
    logger.add(
        sys.stderr,
        level="INFO",
        format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | <level>{message}</level>",
    )
    log_path = Path("logs/training")
    log_path.mkdir(parents=True, exist_ok=True)
    logger.add(log_path / "transformer_{time}.log", rotation="1 day", level="DEBUG")

    mode = args.mode
    logger.info("=" * 70)
    logger.info(f"TRANSFORMER-PPO TRAINING  — Mode {mode}")
    if mode == "A":
        logger.info("  Option A: backbone swap, seq_len=1 (no temporal window)")
    else:
        logger.info(f"  Option B: sliding-window context, K={args.seq_len} steps")
    logger.info(f"  Device: {args.device}")
    logger.info("=" * 70)

    # ── Load YAML config ──────────────────────────────────────────────────────
    cfg_path = (
        Path(args.config)
        if args.config
        else Path("config/training/adversarial_transformer.yaml")
    )
    cfg: dict = {}
    if cfg_path.exists():
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f) or {}
        logger.info(f"Loaded config: {cfg_path}")
    else:
        logger.info("No YAML config found — using built-in defaults")

    # ── Data ──────────────────────────────────────────────────────────────────
    splits = load_data(args)
    train_price, train_features = splits["train"]

    # ── Environment ───────────────────────────────────────────────────────────
    env = make_env(train_price, train_features)
    state_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n
    logger.info(f"Environment: state_dim={state_dim}  n_actions={n_actions}")

    # Validate nhead divides d_model/hidden_dim evenly
    hidden_dim = args.hidden_dim or cfg.get("hidden_dim", DEFAULTS["hidden_dim"])
    d_model = (
        args.transformer_d_model or cfg.get("transformer_d_model", 0)
    ) or hidden_dim
    nhead = args.transformer_nhead or cfg.get(
        "transformer_nhead", DEFAULTS["transformer_nhead"]
    )
    if d_model % nhead != 0:
        logger.error(
            f"transformer_nhead={nhead} does not divide d_model={d_model} evenly. "
            f"Choose nhead from: {[h for h in [1,2,4,8,16] if d_model % h == 0]}"
        )
        sys.exit(1)

    # ── Build config ──────────────────────────────────────────────────────────
    trader_config = build_transformer_config(state_dim, n_actions, args, cfg, mode)

    logger.info(f"PPOConfig summary:")
    logger.info(
        f"  hidden_dim={trader_config.hidden_dim}  "
        f"nhead={trader_config.transformer_nhead}  "
        f"d_model={trader_config.transformer_d_model or trader_config.hidden_dim}  "
        f"layers={trader_config.rnn_layers}  "
        f"seq_len={trader_config.transformer_seq_len}"
    )
    logger.info(
        f"  actor_lr={trader_config.actor_lr}  critic_lr={trader_config.critic_lr}"
    )

    # ── Checkpoint directory ──────────────────────────────────────────────────
    default_ckpt = f"data/models/transformer_{mode.lower()}"
    checkpoint_dir = Path(
        args.checkpoint_dir or cfg.get("checkpoint_dir", default_ckpt)
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    if args.adversarial:
        logger.info("Running adversarial training loop…")
        run_adversarial(trader_config, env, args, cfg, checkpoint_dir)
    else:
        logger.info("Running standalone PPO training loop…")
        agent = PPOAgent(trader_config, device=args.device)
        run_standalone(agent, env, args, cfg, checkpoint_dir, mode)


if __name__ == "__main__":
    main()
