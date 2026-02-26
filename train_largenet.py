"""
Large Network Training Script – Phase 3
=========================================
Base:   Multi-Coin (BTC+ETH+BNB) – Phase 1 winner
Change: hidden_dim 128 -> 256  (double GRU capacity)
Bonus:  Multi-TF features re-enabled (Phase 2 failed due to network being too small)

Comparison against: Multi-Coin Baseline
  Mean Return: +19.17%  |  Sharpe: 2.60  |  WinRate: 79%  |  MaxDD: 5.32%
"""

import sys, json, yaml
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


BASELINE_MULTICOIN = {
    "mean_return": 0.1917,
    "mean_sharpe": 2.60,
    "win_rate": 0.79,
    "mean_max_dd": 0.0532,
}

SYMBOLS = ["BTC/USDT", "ETH/USDT", "BNB/USDT"]
TIMEFRAME = "1h"
START = "2023-01-01"
ITERS = 500
DEVICE = "cpu"
SAVE_DIR = "data/models/largenet"
HIDDEN = 256  # <── the only difference from Phase 1


# ── Multi-TF Features (same logic as Phase 2) ────────────────────────────────


def _rsi(s, w=14):
    d = s.diff()
    g = d.clip(lower=0).rolling(w).mean()
    l = (-d.clip(upper=0)).rolling(w).mean()
    return 100 - 100 / (1 + g / (l + 1e-9))


def _macd_hist(s):
    m = s.ewm(span=12, adjust=False).mean() - s.ewm(span=26, adjust=False).mean()
    return m - m.ewm(span=9, adjust=False).mean()


def add_multitf_features(df):
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    for label, rule in [("4h", "4h"), ("1d", "1d")]:
        r = (
            df.resample(rule)
            .agg(
                {
                    "open": "first",
                    "high": "max",
                    "low": "min",
                    "close": "last",
                    "volume": "sum",
                }
            )
            .dropna()
        )
        close = r["close"]
        ema_f = close.ewm(span=12, adjust=False).mean()
        ema_s = close.ewm(span=26, adjust=False).mean()
        tf_df = pd.DataFrame(
            {
                f"trend_{label}": np.sign(ema_f - ema_s),
                f"rsi_{label}": _rsi(close),
                f"macd_hist_{label}": _macd_hist(close),
                f"volatility_{label}": close.pct_change().rolling(20).std()
                * np.sqrt(252),
            },
            index=r.index,
        ).reindex(df.index, method="ffill")
        df = pd.concat([df, tf_df], axis=1)
    return df


# ── Load data ────────────────────────────────────────────────────────────────


def load_and_prepare():
    loader = CCXTDataLoader(
        DataLoaderConfig(
            exchange_id="binance",
            exchange_type="spot",
            rate_limit_ms=100,
            cache_dir=Path("data/cache"),
            processed_dir=Path("data/processed"),
        )
    )
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

    prices_all, feats_all = [], []
    ohlcv = ["open", "high", "low", "close", "volume"]

    for sym in SYMBOLS:
        logger.info(f"Loading {sym}...")
        raw = loader.download_and_cache(sym, TIMEFRAME, START)
        mtf = add_multitf_features(raw)
        n = len(mtf)
        ti = int(n * 0.70)
        train_p = mtf.iloc[:ti]

        eng = FeatureEngine(feat_cfg)
        std_f = eng.fit_transform(train_p[ohlcv])
        mtf_cols = [c for c in train_p.columns if c not in ohlcv]
        ci = train_p.index.intersection(std_f.index)
        combined = pd.concat(
            [std_f.loc[ci], train_p.loc[ci, mtf_cols].fillna(0)], axis=1
        )

        prices_all.append(train_p[ohlcv].loc[ci])
        feats_all.append(combined)
        logger.info(f"  {sym}: {len(combined)} steps, {len(combined.columns)} features")

    price = pd.concat(prices_all, ignore_index=True)
    feats = pd.concat(feats_all, ignore_index=True).fillna(0)
    logger.info(f"Total: {len(price)} steps, {len(feats.columns)} features")
    return price, feats


# ── Training ──────────────────────────────────────────────────────────────────


def train(price, feats):
    env_config = load_environment_config_from_yaml(
        "config/environment/realistic_env.yaml"
    )
    env = ConfigIntegratedTradingEnv(price, feats, env_config)

    with open("config/training/adversarial.yaml") as f:
        cfg = yaml.safe_load(f)

    def gc(s, k, d):
        return cfg.get(s, {}).get(k, d)

    sd = env.observation_space.shape[0]
    na = env.action_space.n
    logger.info(f"Network: hidden_dim={HIDDEN}  |  state_dim={sd}  |  actions={na}")

    trader_cfg = PPOConfig(
        state_dim=sd,
        n_actions=na,
        hidden_dim=HIDDEN,  # ← 256 instead of 128
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
        state_dim=sd,
        n_actions=na,
        hidden_dim=HIDDEN,  # ← 256 instead of 128
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
        checkpoint_dir=SAVE_DIR,
    )

    trainer = AdversarialTrainer(env, train_cfg, device=DEVICE)
    trainer.train()
    return trainer.evaluate(n_episodes=100)


# ── Comparison ────────────────────────────────────────────────────────────────


def compare(baseline, largenet):
    logger.info("\n" + "═" * 62)
    logger.info("  COMPARISON: Multi-Coin (hidden=128)  vs.  Large Net (hidden=256)")
    logger.info("═" * 62)
    logger.info(f"  {'Metric':<22} {'h=128':>12} {'h=256':>12} {'Δ':>8}")
    logger.info("─" * 62)
    checks = [
        ("Mean Return", "mean_return", "{:+.2%}", True),
        ("Sharpe Ratio", "mean_sharpe", "{:.2f}", True),
        ("Win Rate", "win_rate", "{:.1%}", True),
        ("Max Drawdown", "mean_max_dd", "{:.2%}", False),
    ]
    better = 0
    for label, key, fmt, hib in checks:
        b = baseline.get(key, 0)
        m = largenet.get(key, 0)
        d = m - b
        ok = (d > 0) == hib
        if ok:
            better += 1
        logger.info(
            f"  {label:<22} {fmt.format(b):>12} {fmt.format(m):>12}  "
            f"{'✓' if ok else '✗'} {d:+.3f}"
        )
    logger.info("═" * 62)
    v = "BETTER" if better >= 3 else ("NEUTRAL" if better >= 2 else "WORSE")
    logger.info(f"  RESULT: Large Net is {v} ({better}/4 metrics improved)")
    logger.info("═" * 62)
    return v


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logger.remove()
    logger.add(
        sys.stderr,
        level="INFO",
        format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | <level>{message}</level>",
    )
    logger.add(
        f"logs/training/largenet_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.log",
        level="DEBUG",
    )

    Path(SAVE_DIR).mkdir(parents=True, exist_ok=True)

    logger.info("════ PHASE 3: LARGE NETWORK TRAINING ════")
    logger.info(f"hidden_dim: 128 → {HIDDEN}  |  With Multi-TF Features")

    price, feats = load_and_prepare()
    metrics = train(price, feats)

    with open(f"{SAVE_DIR}/comparison.json", "w") as f:
        json.dump(
            {"baseline_multicoin": BASELINE_MULTICOIN, "largenet": metrics}, f, indent=2
        )

    verdict = compare(BASELINE_MULTICOIN, metrics)
    logger.info(f"Model: {SAVE_DIR}/")
    logger.info(
        f"Next phase: {'Phase 4 (Curriculum Learning)' if verdict != 'WORSE' else 'hidden_dim=128 remains base'}"
    )
