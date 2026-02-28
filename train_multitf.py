"""
Multi-Timeframe Training Script – Phase 2
==========================================
Baut auf dem Multi-Coin Ergebnis (Phase 1) auf.
Fügt 4h und 1d Signale als zusätzliche Features hinzu,
berechnet aus den vorhandenen 1h-Daten per Resampling.

Neue Features (6 zusätzlich, kein Leakage):
  4h: trend_4h, rsi_4h, macd_hist_4h
  1d: trend_1d, rsi_1d, volatility_1d

Vergleich gegen: Multi-Coin Baseline
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


# ── Baseline aus Phase 1 ───────────────────────────────────────────────────────

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
SAVE_DIR = "data/models/multitf"


# ── Multi-Timeframe Feature-Berechnung ────────────────────────────────────────


def _rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))


def _macd_hist(series: pd.Series) -> pd.Series:
    ema12 = series.ewm(span=12, adjust=False).mean()
    ema26 = series.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    sig = macd.ewm(span=9, adjust=False).mean()
    return macd - sig


def add_multitf_features(price_1h: pd.DataFrame) -> pd.DataFrame:
    """
    Resampelt 1h-Daten auf 4h und 1d, berechnet Trend/RSI/MACD,
    merged zurück auf 1h (forward-fill → kein Leakage).
    """
    df = price_1h.copy()
    df.index = pd.to_datetime(df.index)

    for label, rule in [("4h", "4h"), ("1d", "1d")]:
        resampled = (
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

        close = resampled["close"]

        # EMA-Trend: +1 bullish, -1 bearish, 0 neutral
        ema_f = close.ewm(span=12, adjust=False).mean()
        ema_s = close.ewm(span=26, adjust=False).mean()
        trend = np.sign(ema_f - ema_s)

        rsi = _rsi(close, 14)
        mhist = _macd_hist(close)
        vol = close.pct_change().rolling(20).std() * np.sqrt(252)

        tf_df = pd.DataFrame(
            {
                f"trend_{label}": trend,
                f"rsi_{label}": rsi,
                f"macd_hist_{label}": mhist,
                f"volatility_{label}": vol,
            },
            index=resampled.index,
        )

        # Reindex auf 1h, forward-fill (nur vergangene Werte → kein Leakage)
        tf_df = tf_df.reindex(df.index, method="ffill")
        df = pd.concat([df, tf_df], axis=1)

    return df


# ── Daten + Features ──────────────────────────────────────────────────────────


def load_and_prepare(symbols):
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

    for sym in symbols:
        logger.info(f"Verarbeite {sym} mit Multi-TF Features...")
        price_raw = loader.download_and_cache(sym, TIMEFRAME, START)

        # Multi-TF Features hinzufügen VOR dem Split (Resampling braucht volle History)
        price_mtf = add_multitf_features(price_raw)

        n = len(price_mtf)
        ti = int(n * 0.70)

        train_p = price_mtf.iloc[:ti]

        # FeatureEngine auf den OHLCV-Teil fitten (Standard-Features)
        eng = FeatureEngine(feat_cfg)
        ohlcv_cols = ["open", "high", "low", "close", "volume"]
        train_ohlcv = train_p[ohlcv_cols]
        train_std = eng.fit_transform(train_ohlcv)

        # Multi-TF Spalten dazuhängen (bereits forward-filled, kein Leakage)
        mtf_cols = [c for c in train_p.columns if c not in ohlcv_cols]
        ci = train_p.index.intersection(train_std.index)
        mtf_part = train_p.loc[ci, mtf_cols].fillna(0)

        train_combined = pd.concat([train_std.loc[ci], mtf_part], axis=1)

        train_prices_all.append(train_p[ohlcv_cols].loc[ci])
        train_feats_all.append(train_combined)

        logger.info(
            f"  {sym}: {len(train_combined)} Schritte, "
            f"{len(train_combined.columns)} Features "
            f"(+{len(mtf_cols)} Multi-TF)"
        )

    combined_price = pd.concat(train_prices_all, ignore_index=True)
    combined_feats = pd.concat(train_feats_all, ignore_index=True)
    combined_feats = combined_feats.fillna(0)

    logger.info(
        f"Gesamtdatensatz: {len(combined_price)} Schritte, "
        f"{len(combined_feats.columns)} Features"
    )
    return combined_price, combined_feats


# ── Training ──────────────────────────────────────────────────────────────────


def train(price, feats, save_dir):
    env_config = load_environment_config_from_yaml("config/environment/realistic_env.yaml")
    env = ConfigIntegratedTradingEnv(price, feats, env_config)

    with open("config/training/adversarial.yaml") as f:
        cfg = yaml.safe_load(f)

    def gc(s, k, d):
        return cfg.get(s, {}).get(k, d)

    state_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n
    logger.info(f"Observation Space: {state_dim} Features (inkl. Multi-TF)")

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
    return trainer.evaluate(n_episodes=100), trainer


# ── Vergleich ─────────────────────────────────────────────────────────────────


def compare(baseline, multitf):
    logger.info("\n" + "═" * 60)
    logger.info("  VERGLEICH: Multi-Coin  vs.  Multi-Coin + Multi-Timeframe")
    logger.info("═" * 60)
    logger.info(f"  {'Metrik':<22} {'Multi-Coin':>12} {'+ Multi-TF':>12} {'Δ':>8}")
    logger.info("─" * 60)

    checks = [
        ("Mean Return", "mean_return", "{:+.2%}", True),
        ("Sharpe Ratio", "mean_sharpe", "{:.2f}", True),
        ("Win Rate", "win_rate", "{:.1%}", True),
        ("Max Drawdown", "mean_max_dd", "{:.2%}", False),
    ]
    better = 0
    for label, key, fmt, hib in checks:
        b = baseline.get(key, 0)
        m = multitf.get(key, 0)
        d = m - b
        ok = (d > 0) == hib
        if ok:
            better += 1
        logger.info(
            f"  {label:<22} {fmt.format(b):>12} {fmt.format(m):>12}  "
            f"{'✓' if ok else '✗'} {d:+.3f}"
        )

    logger.info("═" * 60)
    verdict = "BESSER" if better >= 3 else ("NEUTRAL" if better >= 2 else "SCHLECHTER")
    logger.info(f"  FAZIT: Multi-TF ist {verdict} ({better}/4 Metriken verbessert)")
    logger.info("═" * 60)
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
        f"logs/training/multitf_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.log", level="DEBUG"
    )

    Path(SAVE_DIR).mkdir(parents=True, exist_ok=True)

    logger.info("════ PHASE 2: MULTI-TIMEFRAME TRAINING ════")
    logger.info(f"Coins: {SYMBOLS}  |  Timeframes: 1h + 4h + 1d")
    logger.info(f"Iterationen: {ITERS}")

    price, feats = load_and_prepare(SYMBOLS)
    metrics, _ = train(price, feats, SAVE_DIR)

    with open(f"{SAVE_DIR}/comparison.json", "w") as f:
        json.dump({"baseline_multicoin": BASELINE_MULTICOIN, "multitf": metrics}, f, indent=2)

    verdict = compare(BASELINE_MULTICOIN, metrics)
    logger.info(f"\nModell gespeichert: {SAVE_DIR}/")
    logger.info(
        f"Nächste Phase: {'Phase 3 (Größeres Netz)' if verdict != 'SCHLECHTER' else 'Bleibt bei Multi-Coin'}"
    )
