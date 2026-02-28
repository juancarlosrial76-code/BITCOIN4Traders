"""
Curriculum Learning – Phase 4
===============================
Basis:  Large Net (hidden=256, Multi-TF, Multi-Coin) – Phase 3 Gewinner
Idee:   Agent lernt schrittweise schwierigere Marktbedingungen,
        statt sofort mit vollem Rauschen konfrontiert zu werden.

Curriculum (3 Stufen):
  Stufe 1 (Iter   1-150): nur Low-Volatility Perioden  → Agent lernt Grundstrategie
  Stufe 2 (Iter 151-300): Low + High-Volatility        → Agent lernt Risikomanagement
  Stufe 3 (Iter 301-500): alle Perioden + Adversary    → Robustheit

Vergleich gegen: Large Net Phase 3
  Mean Return: +68.75%  |  Sharpe: 2.84  |  WinRate: 86%  |  MaxDD: 10.70%
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
from environment.config_system import load_environment_config_from_yaml, EnvironmentConfig
from agents.ppo_agent import PPOConfig
from training.adversarial_trainer import AdversarialTrainer, AdversarialConfig


BASELINE_LARGENET = {
    "mean_return": 0.6875,
    "mean_sharpe": 2.84,
    "win_rate": 0.86,
    "mean_max_dd": 0.1070,
}

SYMBOLS = ["BTC/USDT", "ETH/USDT", "BNB/USDT"]
START = "2023-01-01"
DEVICE = "cpu"
HIDDEN = 256
SAVE_DIR = "data/models/curriculum"


# ── Multi-TF Features (gleich wie Phase 3) ─────────────────────────────────────


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
            .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
            .dropna()
        )
        c = r["close"]
        tf = pd.DataFrame(
            {
                f"trend_{label}": np.sign(
                    c.ewm(12, adjust=False).mean() - c.ewm(26, adjust=False).mean()
                ),
                f"rsi_{label}": _rsi(c),
                f"macd_hist_{label}": _macd_hist(c),
                f"volatility_{label}": c.pct_change().rolling(20).std() * np.sqrt(252),
            },
            index=r.index,
        ).reindex(df.index, method="ffill")
        df = pd.concat([df, tf], axis=1)
    return df


# ── Curriculum: Daten nach Volatilität filtern ────────────────────────────────


def filter_by_volatility(price_df, feat_df, level="low"):
    """
    Filtert Datenpunkte nach Markt-Volatilität.
    level: 'low'  → unteres 40%-Quantil der rollenden Volatilität
           'high' → oberes 40%-Quantil
           'all'  → alles
    """
    if level == "all":
        return price_df, feat_df

    # Rollende 20h-Volatilität aus den Features (bereits berechnet)
    vol_col = "volatility_20" if "volatility_20" in feat_df.columns else feat_df.columns[1]
    vol = feat_df[vol_col].fillna(0)

    if level == "low":
        mask = vol <= vol.quantile(0.40)
    else:  # high
        mask = vol >= vol.quantile(0.60)

    idx = feat_df.index[mask]
    logger.info(
        f"  Curriculum '{level}': {len(idx):,} / {len(feat_df):,} Schritte "
        f"({len(idx)/len(feat_df)*100:.0f}%)"
    )
    return price_df.loc[idx].reset_index(drop=True), feat_df.loc[idx].reset_index(drop=True)


# ── Daten vorbereiten ─────────────────────────────────────────────────────────


def load_all():
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
    ohlcv = ["open", "high", "low", "close", "volume"]
    prices_all, feats_all = [], []

    for sym in SYMBOLS:
        raw = loader.download_and_cache(sym, "1h", START)
        mtf = add_multitf_features(raw)
        ti = int(len(mtf) * 0.70)
        tp = mtf.iloc[:ti]
        eng = FeatureEngine(feat_cfg)
        sf = eng.fit_transform(tp[ohlcv])
        mc = [c for c in tp.columns if c not in ohlcv]
        ci = tp.index.intersection(sf.index)
        cf = pd.concat([sf.loc[ci], tp.loc[ci, mc].fillna(0)], axis=1)
        prices_all.append(tp[ohlcv].loc[ci])
        feats_all.append(cf)

    price = pd.concat(prices_all, ignore_index=True)
    feats = pd.concat(feats_all, ignore_index=True).fillna(0)
    logger.info(f"Gesamt: {len(price):,} Schritte, {feats.shape[1]} Features")
    return price, feats


# ── Trainer aufbauen ──────────────────────────────────────────────────────────


def make_trainer(price, feats, n_iters, adversary_start, save_dir, resume_trainer=None):
    env_config = load_environment_config_from_yaml("config/environment/realistic_env.yaml")
    env = ConfigIntegratedTradingEnv(price, feats, env_config)

    with open("config/training/adversarial.yaml") as f:
        cfg = yaml.safe_load(f)

    def gc(s, k, d):
        return cfg.get(s, {}).get(k, d)

    sd = env.observation_space.shape[0]
    na = env.action_space.n

    trader_cfg = PPOConfig(
        state_dim=sd,
        n_actions=na,
        hidden_dim=HIDDEN,
        actor_lr=float(gc("trader", "actor_lr", 1e-4)),
        critic_lr=float(gc("trader", "critic_lr", 5e-4)),
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        n_epochs=10,
        batch_size=64,
        entropy_coef=0.08,
        use_recurrent=True,
        rnn_type="GRU",
        target_kl=0.015,
        use_lr_decay=True,
        lr_decay_gamma=0.995,
    )
    adv_cfg = PPOConfig(
        state_dim=sd,
        n_actions=na,
        hidden_dim=HIDDEN,
        actor_lr=5e-5,
        critic_lr=2e-4,
        entropy_coef=0.05,
        use_recurrent=True,
        rnn_type="GRU",
    )
    train_cfg = AdversarialConfig(
        n_iterations=n_iters,
        steps_per_iteration=2048,
        trader_config=trader_cfg,
        adversary_config=adv_cfg,
        adversary_start_iteration=adversary_start,
        adversary_strength=0.1,
        save_frequency=50,
        log_frequency=10,
        checkpoint_dir=save_dir,
    )

    trainer = AdversarialTrainer(env, train_cfg, device=DEVICE)

    # Gewichte aus vorheriger Stufe übernehmen
    if resume_trainer is not None:
        try:
            trainer.trader.actor.load_state_dict(resume_trainer.trader.actor.state_dict())
            trainer.trader.critic.load_state_dict(resume_trainer.trader.critic.state_dict())
            trainer.adversary.actor.load_state_dict(resume_trainer.adversary.actor.state_dict())
            trainer.adversary.critic.load_state_dict(resume_trainer.adversary.critic.state_dict())
            logger.info("  Gewichte aus vorheriger Curriculum-Stufe übernommen ✓")
        except Exception as e:
            logger.warning(f"  Gewichte-Transfer fehlgeschlagen: {e}")

    return trainer


# ── Vergleich ─────────────────────────────────────────────────────────────────


def compare(baseline, curriculum):
    logger.info("\n" + "═" * 65)
    logger.info("  VERGLEICH: Large Net (h=256)  vs.  Curriculum (h=256)")
    logger.info("═" * 65)
    checks = [
        ("Mean Return", "mean_return", "{:+.2%}", True),
        ("Sharpe Ratio", "mean_sharpe", "{:.2f}", True),
        ("Win Rate", "win_rate", "{:.1%}", True),
        ("Max Drawdown", "mean_max_dd", "{:.2%}", False),
    ]
    better = 0
    for label, key, fmt, hib in checks:
        b = baseline.get(key, 0)
        m = curriculum.get(key, 0)
        d = m - b
        ok = (d > 0) == hib
        if ok:
            better += 1
        logger.info(
            f"  {label:<22} {fmt.format(b):>12} {fmt.format(m):>12}  "
            f"{'✓' if ok else '✗'} {d:+.3f}"
        )
    logger.info("═" * 65)
    v = "BESSER" if better >= 3 else ("NEUTRAL" if better >= 2 else "SCHLECHTER")
    logger.info(f"  FAZIT: Curriculum ist {v} ({better}/4 Metriken verbessert)")
    logger.info("═" * 65)
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
        f"logs/training/curriculum_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.log",
        level="DEBUG",
    )
    Path(SAVE_DIR).mkdir(parents=True, exist_ok=True)

    logger.info("════ PHASE 4: CURRICULUM LEARNING ════")
    logger.info("Stufe 1 (150 Iter): Low-Volatility  → Grundstrategie")
    logger.info("Stufe 2 (150 Iter): Low+High-Vol    → Risikomanagement")
    logger.info("Stufe 3 (200 Iter): Alles+Adversary → Robustheit")

    # Daten einmal laden
    price_all, feats_all = load_all()

    # ── Stufe 1: Low-Volatility ────────────────────────────────────────────────
    logger.info("\n─── STUFE 1: Low-Volatility (Iter 1-150) ───")
    p1, f1 = filter_by_volatility(price_all, feats_all, "low")
    trainer1 = make_trainer(
        p1, f1, n_iters=150, adversary_start=9999, save_dir=f"{SAVE_DIR}/stage1"
    )
    trainer1.train()
    logger.info("Stufe 1 abgeschlossen ✓")

    # ── Stufe 2: Low + High Volatility ────────────────────────────────────────
    logger.info("\n─── STUFE 2: Low+High-Volatility (Iter 151-300) ───")
    p2, f2 = filter_by_volatility(price_all, feats_all, "all")
    # Nur low+high: obere 60% und untere 40% = alles außer mittlere 20%
    vol = (
        feats_all["volatility_20"].fillna(0)
        if "volatility_20" in feats_all.columns
        else feats_all.iloc[:, 1].fillna(0)
    )
    mask = (vol <= vol.quantile(0.40)) | (vol >= vol.quantile(0.60))
    p2 = price_all.loc[feats_all.index[mask]].reset_index(drop=True)
    f2 = feats_all.loc[feats_all.index[mask]].reset_index(drop=True)
    logger.info(f"  Stufe 2: {len(p2):,} Schritte (Low+High, ohne Mitte)")
    trainer2 = make_trainer(
        p2,
        f2,
        n_iters=150,
        adversary_start=9999,
        save_dir=f"{SAVE_DIR}/stage2",
        resume_trainer=trainer1,
    )
    trainer2.train()
    logger.info("Stufe 2 abgeschlossen ✓")

    # ── Stufe 3: Volle Daten + Adversary ──────────────────────────────────────
    logger.info("\n─── STUFE 3: Alle Daten + Adversary (Iter 301-500) ───")
    trainer3 = make_trainer(
        price_all,
        feats_all,
        n_iters=200,
        adversary_start=0,
        save_dir=f"{SAVE_DIR}/stage3",
        resume_trainer=trainer2,
    )
    trainer3.train()
    logger.info("Stufe 3 abgeschlossen ✓")

    # ── Finale Evaluation ──────────────────────────────────────────────────────
    logger.info("\nFinale Evaluation (100 Episoden)...")
    metrics = trainer3.evaluate(n_episodes=100)

    with open(f"{SAVE_DIR}/comparison.json", "w") as f:
        json.dump({"baseline_largenet": BASELINE_LARGENET, "curriculum": metrics}, f, indent=2)

    verdict = compare(BASELINE_LARGENET, metrics)

    # Bestes Modell in Haupt-Ordner kopieren
    import shutil

    best_src = Path(f"{SAVE_DIR}/stage3")
    best_dst = Path("data/models/best")
    if verdict != "SCHLECHTER":
        shutil.copytree(best_src, best_dst, dirs_exist_ok=True)
        logger.info(f"Bestes Modell gespeichert: data/models/best/")
    else:
        shutil.copytree(Path("data/models/largenet"), best_dst, dirs_exist_ok=True)
        logger.info("Large Net bleibt bestes Modell → data/models/best/")

    logger.info(f"\n→ Nächster Schritt: Phase 5 – Paper Trading mit data/models/best/")
