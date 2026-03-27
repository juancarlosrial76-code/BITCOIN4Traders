#!/usr/bin/env python3
"""
automl_loop.py — Autoresearch Loop für maximalen Tages-Gewinn
=============================================================
Inspiriert von karpathy/autoresearch:
  - Fixer Zeitbudget pro Experiment
  - Echtes Paper Trading Profit als Metrik (nicht Training-Return)
  - Automatische Parametervorschläge basierend auf Geschichte
  - Läuft die ganze Nacht autonom

Unterschied zu experiment_runner.py:
  experiment_runner: feste Configs, einmalig, kein Feedback
  automl_loop:       adaptiv, echtes Paper Trading misst Erfolg,
                     jedes Experiment lernt vom vorherigen

Usage:
    python3 automl_loop.py                 # läuft bis Strg+C
    python3 automl_loop.py --hours 8       # läuft 8 Stunden
    python3 automl_loop.py --iterations 5  # max 5 Experimente
    python3 automl_loop.py --status        # zeigt bisherige Ergebnisse
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

WORK_DIR = Path(__file__).resolve().parent
LOG_DIR  = WORK_DIR / "logs" / "automl"
RESULTS  = LOG_DIR / "results.json"
BEST     = LOG_DIR / "best_params.json"

LOG_DIR.mkdir(parents=True, exist_ok=True)


# ── Suchraum (aus automl_program.md) ────────────────────────────────
PARAM_SPACE = {
    "lambda_cost":    (0.5,  5.0),
    "lambda_draw":    (0.5,  4.0),
    "lambda_regime":  (0.2,  1.5),
    "win_bonus":      (0.1,  1.0),
    "loss_penalty":   (0.2,  2.0),
}

# Startpunkt (letzte bekannte gute Werte)
DEFAULT_PARAMS = {
    "lambda_cost":   2.0,
    "lambda_draw":   2.0,
    "lambda_regime": 0.5,
    "win_bonus":     0.3,
    "loss_penalty":  0.5,
}

TRAINING_ITERATIONS = 10    # pro Experiment
PAPER_MEASURE_SECS  = 1800  # 30 Min Paper Trading messen
PAPER_STARTUP_SECS  = 120   # 2 Min Warmup bevor Messung startet


def log(msg: str):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    main_log = LOG_DIR / "automl_run.log"
    with open(main_log, "a") as f:
        f.write(line + "\n")


def load_results() -> list[dict]:
    if RESULTS.exists():
        return json.loads(RESULTS.read_text())
    return []


def save_results(results: list[dict]):
    RESULTS.write_text(json.dumps(results, indent=2))


def load_best_params() -> dict:
    """Load best params from file, fallback to DEFAULT_PARAMS."""
    if BEST.exists():
        return json.loads(BEST.read_text())
    # Also check experiment_runner best config
    exp_best = WORK_DIR / "logs" / "experiments" / "best_config.yaml"
    if exp_best.exists():
        try:
            import yaml
            cfg = yaml.safe_load(exp_best.read_text())
            if "reward_params" in cfg:
                return cfg["reward_params"]
        except Exception:
            pass
    return DEFAULT_PARAMS.copy()


def save_best_params(params: dict, score: float):
    BEST.write_text(json.dumps({"params": params, "score": score,
                                "saved_at": datetime.now(timezone.utc).isoformat()}, indent=2))


def propose_params(history: list[dict]) -> dict:
    """
    Schlägt nächste Reward-Parameter vor.
    Strategie:
      - Erste 3 Experimente: gezielte Varianten (hohe/niedrige Kostenstrafe)
      - Danach: Gauß-Perturbation um bisherige beste Parameter
      - Wenn letzter Run besser als vorletzter: aggressiver (Exploitation)
      - Sonst: breitere Exploration
    """
    n = len(history)

    # Gezielte Startkandidaten für erste Runden
    if n == 0:
        return DEFAULT_PARAMS.copy()
    if n == 1:
        return {**DEFAULT_PARAMS, "lambda_cost": 3.5, "lambda_draw": 1.5}
    if n == 2:
        return {**DEFAULT_PARAMS, "lambda_cost": 1.0, "lambda_draw": 3.0, "win_bonus": 0.5}

    # Ab Runde 3: Lerne aus Geschichte
    best_result = max(history, key=lambda r: r.get("paper_pnl_30min", -9999))
    best_params = best_result["params"]

    # Wie gut war der letzte Run vs vorletzter?
    last_score = history[-1].get("paper_pnl_30min", -9999)
    prev_score = history[-2].get("paper_pnl_30min", -9999) if len(history) >= 2 else -9999
    improving = last_score > prev_score

    # Perturbation-Breite: kleiner wenn wir verbessern (Exploitation), größer sonst
    sigma_factor = 0.15 if improving else 0.30

    new_params = {}
    for key, (lo, hi) in PARAM_SPACE.items():
        center = best_params.get(key, DEFAULT_PARAMS.get(key, (lo+hi)/2))
        sigma  = (hi - lo) * sigma_factor
        val    = center + random.gauss(0, sigma)
        val    = max(lo, min(hi, val))   # Clip to search space
        val    = round(val, 3)
        new_params[key] = val

    return new_params


def run_training(params: dict) -> dict:
    """Training mit gegebenen Params, gibt Metriken zurück."""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(WORK_DIR / "src")
    env["TRAINING_MODE"] = "1"
    env["REWARD_PARAMS"] = json.dumps(params)

    log_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_log = LOG_DIR / f"train_{log_ts}.log"

    log(f"  Training {TRAINING_ITERATIONS} iter | params: {params}")

    train_log_dir = WORK_DIR / "logs" / "training"
    existing = set(train_log_dir.glob("train_*.log")) if train_log_dir.exists() else set()

    t0 = time.time()
    result = subprocess.run(
        ["python3", "train.py", "--device", "cpu",
         "--iterations", str(TRAINING_ITERATIONS)],
        capture_output=True, text=True,
        timeout=7200, env=env, cwd=WORK_DIR,
    )
    elapsed = time.time() - t0

    # Find the loguru log from this run
    new_logs = sorted(
        set(train_log_dir.glob("train_*.log")) - existing,
        key=lambda p: p.stat().st_mtime,
    )
    train_text = new_logs[-1].read_text(errors="replace") if new_logs else result.stderr

    # Parse training metrics
    train_return = -999.0
    sharpe = 0.0
    trade_wr = -1.0
    for line in train_text.splitlines():
        if ("Weighted Return:" in line or "Mean Return:" in line) and "%" in line:
            try:
                key = "Weighted Return:" if "Weighted Return:" in line else "Mean Return:"
                train_return = float(line.split(key)[1].split("%")[0].strip())
            except Exception:
                pass
        if "Episode Sharpe:" in line:
            try:
                sharpe = float(line.split("Episode Sharpe:")[1].strip())
            except Exception:
                pass
        if "Trade WR" in line and "%" in line:
            try:
                trade_wr = float(line.split("Trade WR")[1].split("%")[0].strip()) / 100.0
            except Exception:
                pass

    log(f"  Training done in {elapsed/60:.1f}min | return={train_return:.1f}% sharpe={sharpe:.2f} tradeWR={trade_wr*100:.1f}%")
    return {
        "success": result.returncode == 0,
        "train_return": train_return,
        "sharpe": sharpe,
        "trade_wr": trade_wr,
        "elapsed_s": round(elapsed),
        "log": str(run_log),
    }


def deploy_and_measure(params: dict) -> float:
    """
    Deploy bestes Modell → starte Paper Trading neu → messe 30 Min.
    Gibt realized PnL nach 30 Min zurück (echtes Geld-Metrik).
    """
    log("  Deploying model to paper trading...")
    deploy_result = subprocess.run(
        ["python3", "deploy_model.py", "--restart"],
        capture_output=True, text=True, cwd=WORK_DIR,
    )
    if deploy_result.returncode != 0:
        log(f"  Deploy failed: {deploy_result.stderr[:200]}")
        return -9999.0

    log(f"  Warmup {PAPER_STARTUP_SECS}s...")
    time.sleep(PAPER_STARTUP_SECS)

    # Snapshot PnL vor Messung
    pnl_start = read_paper_pnl()
    log(f"  Measure start: realized=${pnl_start:.2f}")

    log(f"  Measuring {PAPER_MEASURE_SECS//60} min...")
    time.sleep(PAPER_MEASURE_SECS)

    pnl_end = read_paper_pnl()
    pnl_delta = pnl_end - pnl_start
    daily_est = pnl_delta * 48   # 30 Min × 48 = 24h Hochrechnung

    log(f"  Measure end: realized=${pnl_end:.2f} | delta=${pnl_delta:.2f} | daily_est=${daily_est:.2f}")
    return pnl_delta


INITIAL_CAPITAL = 10_000.0

def read_paper_pnl() -> float:
    """
    Liest aktuellen Total PnL (Realized + Unrealized) aus Paper Trading Log.
    Gibt PnL relativ zum Startkapital zurück (positiv = Gewinn).
    Total PnL in log = aktueller Portfolio-Wert, nicht delta.
    """
    paper_logs = sorted(
        (WORK_DIR / "logs" / "paper").glob("paper_*.log"),
        key=lambda p: p.stat().st_mtime,
    )
    if not paper_logs:
        return 0.0
    try:
        text = paper_logs[-1].read_text(errors="replace")
        total_pnl_raw = None
        realized = 0.0
        for line in text.splitlines():
            # "Total PnL: +$9823.69" — portfolio value (not delta)
            if "Total PnL:" in line and "$" in line:
                try:
                    val = float(
                        line.split("Total PnL:")[1].replace("+","").split("$")[1].split()[0].replace(",","")
                    )
                    total_pnl_raw = val
                except Exception:
                    pass
            if "Realized:" in line and "$" in line:
                try:
                    realized = float(line.split("Realized:")[1].split("$")[1].split()[0].replace(",",""))
                except Exception:
                    pass
        if total_pnl_raw is not None:
            return total_pnl_raw - INITIAL_CAPITAL  # delta vs start capital
        return realized
    except Exception:
        return 0.0


def print_status(results: list[dict]):
    if not results:
        print("No results yet.")
        return
    print("\n" + "="*80)
    print("AUTOML RESULTS — sortiert nach Paper Trading PnL (30 min)")
    print("="*80)
    print(f"{'#':>3} {'TrainRet':>9} {'Sharpe':>7} {'PnL30m':>8} {'DailyEst':>10}  Params")
    print("-"*80)
    for i, r in enumerate(sorted(results, key=lambda x: x.get("paper_pnl_30min", -9999), reverse=True)):
        pnl = r.get("paper_pnl_30min", -9999)
        daily = pnl * 48
        tr = r.get("train_metrics", {}).get("train_return", -999)
        sh = r.get("train_metrics", {}).get("sharpe", 0)
        p = r.get("params", {})
        params_str = f"lc={p.get('lambda_cost','?')} ld={p.get('lambda_draw','?')} lr={p.get('lambda_regime','?')} wb={p.get('win_bonus','?')}"
        print(f"{i+1:>3} {tr:>8.1f}% {sh:>7.2f} {pnl:>+7.2f}$ {daily:>+9.2f}$  {params_str}")
    print("="*80)

    best = max(results, key=lambda x: x.get("paper_pnl_30min", -9999))
    print(f"\nBeste Parameter (PnL={best.get('paper_pnl_30min',0):+.2f}$):")
    for k, v in best.get("params", {}).items():
        print(f"  {k}: {v}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hours",      type=float, default=24.0, help="Max Laufzeit in Stunden")
    parser.add_argument("--iterations", type=int,   default=999,  help="Max Anzahl Experimente")
    parser.add_argument("--status",     action="store_true",      help="Nur Ergebnisse anzeigen")
    parser.add_argument("--no-measure", action="store_true",      help="Paper Trading Messung überspringen (nur Training)")
    args = parser.parse_args()

    results = load_results()

    if args.status:
        print_status(results)
        return

    log("=" * 60)
    log("AUTOML LOOP GESTARTET")
    log(f"Ziel: max Tages-Gewinn | Budget: {args.hours}h | Max: {args.iterations} Experimente")
    log("=" * 60)

    start_time = time.time()
    max_secs = args.hours * 3600
    iteration = 0
    best_score = max((r.get("paper_pnl_30min", -9999) for r in results), default=-9999)

    while (time.time() - start_time) < max_secs and iteration < args.iterations:
        iteration += 1
        elapsed_h = (time.time() - start_time) / 3600
        remaining_h = (max_secs - (time.time() - start_time)) / 3600
        log(f"\n{'='*60}")
        log(f"EXPERIMENT {iteration} | Elapsed: {elapsed_h:.1f}h | Remaining: {remaining_h:.1f}h")
        log(f"{'='*60}")

        # 1. Neue Parameter vorschlagen
        params = propose_params(results)
        log(f"Params: {params}")

        # 2. Training
        try:
            train_metrics = run_training(params)
        except subprocess.TimeoutExpired:
            log("  Training timeout — skip")
            continue
        except Exception as e:
            log(f"  Training error: {e} — skip")
            continue

        if not train_metrics["success"]:
            log("  Training failed — skip")
            continue

        # 3. Deploy + Messen
        if args.no_measure:
            pnl_30min = 0.0
            log("  --no-measure: skipping paper trading measurement")
        else:
            pnl_30min = deploy_and_measure(params)

        # 4. Ergebnis speichern
        result = {
            "iteration": iteration,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "params": params,
            "train_metrics": train_metrics,
            "paper_pnl_30min": pnl_30min,
            "daily_est_usd": round(pnl_30min * 48, 2),
        }
        results.append(result)
        save_results(results)

        # 5. Bester bisher?
        if pnl_30min > best_score:
            best_score = pnl_30min
            save_best_params(params, pnl_30min)
            log(f"  ★ NEW BEST: PnL30m=${pnl_30min:+.2f} | DailyEst=${pnl_30min*48:+.2f}")
        else:
            log(f"  PnL30m=${pnl_30min:+.2f} (best=${best_score:+.2f})")

        print_status(results)

    log("\n" + "="*60)
    log("AUTOML LOOP BEENDET")
    log(f"Experimente: {iteration} | Beste PnL30m: ${best_score:+.2f} | DailyEst: ${best_score*48:+.2f}")
    log("="*60)
    print_status(results)


if __name__ == "__main__":
    main()
