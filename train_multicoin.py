"""
Multi-Coin Training Script – Phase 1
======================================
Trainiert den PPO-Agent auf BTC + ETH + BNB kombiniert.
Daten werden chronologisch aneinandergehängt (Coin-Shuffling),
so dass der Agent marktübergreifende Muster lernt.

Vergleich: Baseline (BTC only) vs. Multi-Coin
"""

import sys
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from loguru import logger

sys.path.insert(0, "src")

from data.ccxt_loader import CCXTDataLoader, DataLoaderConfig
from features.feature_engine import FeatureEngine, FeatureConfig
from environment.config_integrated_env import ConfigIntegratedTradingEnv
from environment.config_system import load_environment_config_from_yaml
from agents.ppo_agent import PPOConfig
from training.adversarial_trainer import AdversarialTrainer, AdversarialConfig
import yaml


# ── Konfiguration ──────────────────────────────────────────────────────────────

SYMBOLS = ["BTC/USDT", "ETH/USDT", "BNB/USDT"]
TIMEFRAME = "1h"
START = "2023-01-01"
ITERS = 500
DEVICE = "cpu"
SAVE_DIR = "data/models/multicoin"

BASELINE = {
    "mean_return": 0.0855,
    "sharpe": 1.12,
    "win_rate": 0.77,
    "max_dd": 0.0843,
}


# ── Daten laden & Features ────────────────────────────────────────────────────


def load_and_combine(symbols: list) -> tuple:
    """
    Lädt alle Coins, berechnet Features auf jedem einzeln (kein Leakage),
    hängt Trainings-Splits aneinander. Val/Test bleibt getrennt pro Coin.
    """
    loader_cfg = DataLoaderConfig(
        exchange_id="binance",
        exchange_type="spot",
        rate_limit_ms=100,
        cache_dir=Path("data/cache"),
        processed_dir=Path("data/processed"),
    )
    loader = CCXTDataLoader(loader_cfg)

    feat_cfg = FeatureConfig(
        volatility_window=20,
        ou_window=20,
        rolling_mean_window=20,
        use_log_returns=True,
        scaler_type="standard",
        save_scaler=False,
        scaler_path=Path("data/processed"),
        dropna_strategy="rolling",
        min_valid_rows=1000,
    )

    train_prices_all, train_feats_all = [], []
    val_data, test_data = {}, {}

    for sym in symbols:
        logger.info(f"Verarbeite {sym}...")
        price = loader.download_and_cache(sym, TIMEFRAME, START)

        n = len(price)
        ti = int(n * 0.70)
        vi = int(n * 0.85)

        train_p = price.iloc[:ti]
        val_p = price.iloc[ti:vi]
        test_p = price.iloc[vi:]

        eng = FeatureEngine(feat_cfg)
        train_f = eng.fit_transform(train_p)
        val_f = eng.transform(val_p)
        test_f = eng.transform(test_p)

        # Align Indexes
        ci = train_p.index.intersection(train_f.index)
        train_prices_all.append(train_p.loc[ci])
        train_feats_all.append(train_f.loc[ci])

        ci_v = val_p.index.intersection(val_f.index)
        ci_t = test_p.index.intersection(test_f.index)
        val_data[sym] = (val_p.loc[ci_v], val_f.loc[ci_v])
        test_data[sym] = (test_p.loc[ci_t], test_f.loc[ci_t])

    # Trainings-Daten kombinieren: alle Coins aneinanderhängen
    combined_price = pd.concat(train_prices_all, ignore_index=True)
    combined_feats = pd.concat(train_feats_all, ignore_index=True)

    logger.info(
        f"Kombinierter Trainingsdatensatz: {len(combined_price)} Schritte "
        f"({len(symbols)} Coins × ~{len(combined_price)//len(symbols)} Steps)"
    )

    return combined_price, combined_feats, val_data, test_data


# ── Training ──────────────────────────────────────────────────────────────────


def train(price: pd.DataFrame, feats: pd.DataFrame, save_dir: str) -> dict:
    env_config = load_environment_config_from_yaml("config/environment/realistic_env.yaml")
    env = ConfigIntegratedTradingEnv(price, feats, env_config)

    with open("config/training/adversarial.yaml") as f:
        cfg = yaml.safe_load(f)

    def gc(s, k, d):
        return cfg.get(s, {}).get(k, d)

    state_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    trader_cfg = PPOConfig(
        state_dim=state_dim,
        n_actions=n_actions,
        hidden_dim=gc("trader", "hidden_dim", 128),
        actor_lr=float(gc("trader", "actor_lr", 1e-4)),
        critic_lr=float(gc("trader", "critic_lr", 5e-4)),
        gamma=gc("trader", "gamma", 0.99),
        gae_lambda=gc("trader", "gae_lambda", 0.95),
        clip_epsilon=gc("trader", "clip_epsilon", 0.2),
        n_epochs=gc("trader", "n_epochs", 10),
        batch_size=gc("trader", "batch_size", 64),
        entropy_coef=gc("trader", "entropy_coef", 0.08),
        use_recurrent=True,
        rnn_type="GRU",
        target_kl=gc("trader", "target_kl", 0.015),
        use_lr_decay=True,
        lr_decay_gamma=0.995,
    )
    adv_cfg = PPOConfig(
        state_dim=state_dim,
        n_actions=n_actions,
        hidden_dim=gc("adversary", "hidden_dim", 128),
        actor_lr=float(gc("adversary", "actor_lr", 5e-5)),
        critic_lr=float(gc("adversary", "critic_lr", 2e-4)),
        entropy_coef=gc("adversary", "entropy_coef", 0.05),
        use_recurrent=True,
        rnn_type="GRU",
    )

    train_cfg = AdversarialConfig(
        n_iterations=ITERS,
        steps_per_iteration=2048,
        trader_config=trader_cfg,
        adversary_config=adv_cfg,
        adversary_start_iteration=100,
        adversary_strength=0.1,
        save_frequency=50,
        log_frequency=10,
        checkpoint_dir=save_dir,
    )

    trainer = AdversarialTrainer(env, train_cfg, device=DEVICE)
    trainer.train()

    metrics = trainer.evaluate(n_episodes=100)
    return metrics, trainer


# ── Vergleich ─────────────────────────────────────────────────────────────────


def compare(baseline: dict, multicoin: dict):
    logger.info("\n" + "═" * 55)
    logger.info("  VERGLEICH: BTC only  vs.  Multi-Coin (BTC+ETH+BNB)")
    logger.info("═" * 55)
    logger.info(f"  {'Metrik':<20} {'Baseline':>12} {'Multi-Coin':>12} {'Δ':>8}")
    logger.info("─" * 55)

    metrics = [
        ("Mean Return", "mean_return", "{:+.2%}", True),
        ("Sharpe Ratio", "mean_sharpe", "{:.2f}", True),
        ("Win Rate", "win_rate", "{:.1%}", True),
        ("Max Drawdown", "mean_max_dd", "{:.2%}", False),
    ]

    better = 0
    for label, key, fmt, higher_is_better in metrics:
        b = baseline.get(key, 0)
        m = multicoin.get(key, 0)
        delta = m - b
        sign = "✓" if (delta > 0) == higher_is_better else "✗"
        if (delta > 0) == higher_is_better:
            better += 1
        logger.info(f"  {label:<20} {fmt.format(b):>12} {fmt.format(m):>12}  {sign} {delta:+.3f}")

    logger.info("═" * 55)
    verdict = "BESSER" if better >= 3 else ("NEUTRAL" if better >= 2 else "SCHLECHTER")
    logger.info(f"  FAZIT: Multi-Coin ist {verdict} ({better}/4 Metriken verbessert)")
    logger.info("═" * 55)
    return verdict


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logger.remove()
    logger.add(
        sys.stderr,
        level="INFO",
        format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | <level>{message}</level>",
    )
    logger.add(
        f"logs/training/multicoin_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.log", level="DEBUG"
    )

    Path("logs/training").mkdir(parents=True, exist_ok=True)
    Path(SAVE_DIR).mkdir(parents=True, exist_ok=True)

    logger.info("════ PHASE 1: MULTI-COIN TRAINING ════")
    logger.info(f"Coins: {SYMBOLS}")
    logger.info(f"Iterationen: {ITERS}")

    # Daten laden
    price, feats, val_data, test_data = load_and_combine(SYMBOLS)

    # Training
    logger.info("Starte Training...")
    multicoin_metrics, trainer = train(price, feats, SAVE_DIR)

    # Ergebnisse speichern
    results = {
        "baseline": BASELINE,
        "multicoin": multicoin_metrics,
    }
    with open(f"{SAVE_DIR}/comparison.json", "w") as f:
        json.dump(results, f, indent=2)

    # Vergleich ausgeben
    verdict = compare(BASELINE, multicoin_metrics)

    logger.info(f"\nModell gespeichert: {SAVE_DIR}/")
    logger.info(f"Fazit für nächste Phase: Multi-Coin ist {verdict}")
