#!/usr/bin/env python3
"""
automl_loop.py — Autoresearch Loop für maximalen Tages-Gewinn
=============================================================
Architektur (2-Phasen):

  Phase 1 — Parallel Training + Backtest-Filter (schnell, ~20 Min pro Runde)
    ├── K=3 Param-Sets gleichzeitig trainieren
    ├── Jedes Modell auf 236 Tage Out-of-Sample Daten backtesten
    └── Top 1 Kandidat deployen

  Phase 2 — Paper Trading läuft kontinuierlich im Hintergrund
    └── Validierung über den Tag — keine erzwungene 30-Min-Messung

Warum kein langes Paper Trading Messen:
  30 Min Messung: Signal/Noise = 0.05x (fast reines Rauschen)
  Backtest 236 Tage: deterministisch, schnell, kein Rauschen

Metriken (Priorität):
  1. Backtest Daily Return % auf Test-Set (Tages-Gewinn nach Kosten)
  2. Backtest Sharpe Ratio
  3. Training Return (nur als Proxy)

Usage:
    python3 automl_loop.py                 # läuft bis Strg+C
    python3 automl_loop.py --hours 16      # 16 Stunden
    python3 automl_loop.py --parallel 3    # 3 parallele Trainings (default)
    python3 automl_loop.py --status        # Ergebnisse anzeigen
    python3 automl_loop.py --no-backtest   # nur Training, kein Backtest
"""

from __future__ import annotations

import argparse
import concurrent.futures
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


# ── Suchraum ─────────────────────────────────────────────────────────
PARAM_SPACE = {
    "lambda_cost":    (0.5,  5.0),
    "lambda_draw":    (0.5,  4.0),
    "lambda_regime":  (0.2,  1.5),
    "win_bonus":      (0.1,  1.0),
    "loss_penalty":   (0.2,  2.0),
}

DEFAULT_PARAMS = {
    "lambda_cost":   2.0,
    "lambda_draw":   2.0,
    "lambda_regime": 0.5,
    "win_bonus":     0.3,
    "loss_penalty":  0.5,
}

TRAINING_ITERATIONS = 10     # pro Kandidat pro Runde
PARALLEL_K          = 1      # auf CPU: sequentiell (parallel teilt Kerne = kein Gewinn)
MODEL_DIR           = WORK_DIR / "data" / "models" / "automl_candidates"
ADV_DIR             = WORK_DIR / "data" / "models" / "adversarial"


def log(msg: str):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_DIR / "automl_run.log", "a") as f:
        f.write(line + "\n")


def load_results() -> list[dict]:
    return json.loads(RESULTS.read_text()) if RESULTS.exists() else []


def save_results(results: list[dict]):
    RESULTS.write_text(json.dumps(results, indent=2))


def load_best_params() -> dict:
    if BEST.exists():
        data = json.loads(BEST.read_text())
        return data.get("params", DEFAULT_PARAMS.copy())
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


def save_best(params: dict, score: float, backtest: dict):
    BEST.write_text(json.dumps({
        "params": params,
        "backtest_daily_return_pct": score,
        "backtest": backtest,
        "saved_at": datetime.now(timezone.utc).isoformat(),
    }, indent=2))


# ── Parametervorschläge ──────────────────────────────────────────────

def propose_batch(history: list[dict], k: int) -> list[dict]:
    """
    Schlägt k verschiedene Param-Sets vor.
    Erste Runden: gezielte Eckpunkte des Suchraums.
    Danach: adaptiv um beste bekannte Parameter.
    """
    n = len(history)
    candidates = []

    # Runde 0: Default + 2 Eckpunkte
    if n == 0:
        candidates.append(DEFAULT_PARAMS.copy())
        candidates.append({**DEFAULT_PARAMS, "lambda_cost": 4.0, "lambda_draw": 1.0})
        candidates.append({**DEFAULT_PARAMS, "lambda_cost": 1.0, "lambda_draw": 3.5, "win_bonus": 0.6})
        return candidates[:k]

    # Runde 1: weitere Eckpunkte
    if n <= k:
        candidates.append({**DEFAULT_PARAMS, "lambda_cost": 3.0, "lambda_regime": 1.2})
        candidates.append({**DEFAULT_PARAMS, "lambda_cost": 2.5, "loss_penalty": 1.5, "win_bonus": 0.2})
        candidates.append({**DEFAULT_PARAMS, "lambda_cost": 1.5, "lambda_draw": 2.5, "lambda_regime": 0.8})

    # Ab Runde 2: lerne aus Geschichte
    best = max(history, key=lambda r: r.get("backtest_daily_return_pct", -999))
    best_params = best.get("params", DEFAULT_PARAMS)

    # Wie stark verbessern wir uns?
    scores = [r.get("backtest_daily_return_pct", -999) for r in history[-3:]]
    improving = len(scores) >= 2 and scores[-1] > scores[-2]
    sigma_factor = 0.12 if improving else 0.25

    while len(candidates) < k:
        new_params = {}
        for key, (lo, hi) in PARAM_SPACE.items():
            center = best_params.get(key, DEFAULT_PARAMS.get(key, (lo + hi) / 2))
            sigma  = (hi - lo) * sigma_factor
            val    = center + random.gauss(0, sigma)
            new_params[key] = round(max(lo, min(hi, val)), 3)
        # Sicherstellen dass dieser Vorschlag einzigartig ist
        if not any(_similar(new_params, c) for c in candidates):
            candidates.append(new_params)

    return candidates[:k]


def _similar(a: dict, b: dict, tol: float = 0.1) -> bool:
    """Zwei Param-Sets sind ähnlich wenn alle Werte innerhalb tol liegen."""
    for k in PARAM_SPACE:
        lo, hi = PARAM_SPACE[k]
        range_ = hi - lo
        if abs(a.get(k, 0) - b.get(k, 0)) > tol * range_:
            return False
    return True


# ── Training (läuft parallel) ────────────────────────────────────────

def train_candidate(params: dict, slot: int) -> dict:
    """
    Trainiert ein Modell mit gegebenen Params.
    Nutzt Standard-Checkpoint-Dir (data/models/adversarial/).
    Nach Training: kopiert best_model_trader.pth → automl_candidates/slot_{slot}/
    """
    import shutil
    model_out = MODEL_DIR / f"slot_{slot}"
    model_out.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["PYTHONPATH"] = str(WORK_DIR / "src")
    env["TRAINING_MODE"] = "1"
    env["REWARD_PARAMS"] = json.dumps(params)

    train_log_dir = WORK_DIR / "logs" / "training"
    existing = set(train_log_dir.glob("train_*.log")) if train_log_dir.exists() else set()

    t0 = time.time()
    result = subprocess.run(
        ["python3", "train.py", "--device", "cpu",
         "--iterations", str(TRAINING_ITERATIONS)],
        capture_output=True, text=True, timeout=3600,
        env=env, cwd=WORK_DIR,
    )
    elapsed = time.time() - t0

    # Modell sichern bevor nächster Kandidat es überschreibt
    src_trader = ADV_DIR / "best_model_trader.pth"
    if src_trader.exists():
        shutil.copy2(src_trader, model_out / "best_model_trader.pth")

    # Lese loguru-Log
    new_logs = sorted(
        set(train_log_dir.glob("train_*.log")) - existing,
        key=lambda p: p.stat().st_mtime,
    )
    train_text = new_logs[-1].read_text(errors="replace") if new_logs else result.stderr

    train_return = -999.0
    sharpe = 0.0
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

    return {
        "slot": slot,
        "params": params,
        "success": result.returncode == 0,
        "train_return": train_return,
        "train_sharpe": sharpe,
        "elapsed_s": round(elapsed),
        "model_dir": str(model_out),
    }


# ── Backtest-Evaluation ──────────────────────────────────────────────

def backtest_model(model_dir: Path, params: dict) -> dict:
    """
    Backtest auf den letzten 15% der Daten (236 Tage out-of-sample).
    Gibt daily_return_pct, sharpe, max_drawdown zurück.
    """
    trader_path = model_dir / "best_model_trader.pth"
    if not trader_path.exists():
        # Fallback: letzter Checkpoint
        checkpoints = sorted(model_dir.glob("checkpoint_iter_*_trader.pth"))
        if checkpoints:
            trader_path = checkpoints[-1]
        else:
            return {"success": False, "daily_return_pct": -999.0}

    env = os.environ.copy()
    env["PYTHONPATH"] = str(WORK_DIR / "src")
    env["REWARD_PARAMS"] = json.dumps(params)
    env["BACKTEST_MODEL_PATH"] = str(trader_path)

    t0 = time.time()
    result = subprocess.run(
        ["python3", "experiment_validate.py", "--backtest",
         "--model", str(trader_path)],
        capture_output=True, text=True, timeout=600,
        env=env, cwd=WORK_DIR,
    )
    elapsed = time.time() - t0

    output = result.stdout + result.stderr
    daily_return = -999.0
    sharpe = 0.0
    max_dd = 0.0

    for line in output.splitlines():
        if ("Weighted Return:" in line or "Mean Return:" in line) and "%" in line:
            try:
                key = "Weighted Return:" if "Weighted Return:" in line else "Mean Return:"
                val = float(line.split(key)[1].split("%")[0].strip())
                # Umrechnen: episode return → daily return
                # 236 Tage Test, jede Episode ~50-100 Stunden → ~N Episoden
                # Einfacher Proxy: return / 236 = täglicher Anteil
                daily_return = val / 236.0
            except Exception:
                pass
        if "Episode Sharpe:" in line:
            try:
                sharpe = float(line.split("Episode Sharpe:")[1].strip())
            except Exception:
                pass
        if "Max DD:" in line or "Max Drawdown:" in line or "Mean Max DD:" in line:
            try:
                dd_str = line.split(":")[-1].strip().replace("%","")
                max_dd = float(dd_str)
            except Exception:
                pass

    return {
        "success": result.returncode == 0,
        "daily_return_pct": daily_return,
        "sharpe": sharpe,
        "max_drawdown_pct": max_dd,
        "elapsed_s": round(elapsed),
    }


def score(backtest: dict, train: dict) -> float:
    """Composite Score: Tages-Return dominiert, Sharpe als Tiebreaker."""
    if not backtest.get("success", False):
        return -999.0
    daily = backtest.get("daily_return_pct", -999)
    sharpe = backtest.get("sharpe", 0)
    dd = backtest.get("max_drawdown_pct", 100)
    # Tages-Return 3x, Sharpe 1x, Drawdown-Penalty
    return daily * 3.0 + sharpe * 1.0 - max(dd - 5.0, 0) * 0.1


# ── Status-Ausgabe ───────────────────────────────────────────────────

def print_status(results: list[dict]):
    if not results:
        print("Keine Ergebnisse bisher.")
        return
    print("\n" + "="*90)
    print("AUTOML ERGEBNISSE — sortiert nach Backtest Tages-Return")
    print("="*90)
    print(f"{'#':>3} {'DailyRet':>9} {'Sharpe':>7} {'MaxDD':>7} {'TrnRet':>8} {'Score':>7}  Params")
    print("-"*90)
    for i, r in enumerate(sorted(results, key=lambda x: x.get("score", -999), reverse=True)[:15]):
        bt   = r.get("backtest", {})
        tr   = r.get("train_return", -999)
        sc   = r.get("score", -999)
        dr   = bt.get("daily_return_pct", -999)
        sh   = bt.get("sharpe", 0)
        dd   = bt.get("max_drawdown_pct", 0)
        p    = r.get("params", {})
        ps   = f"lc={p.get('lambda_cost','?')} ld={p.get('lambda_draw','?')} lr={p.get('lambda_regime','?')} wb={p.get('win_bonus','?')} lp={p.get('loss_penalty','?')}"
        print(f"{i+1:>3} {dr:>8.3f}% {sh:>7.2f} {dd:>6.1f}% {tr:>7.1f}% {sc:>7.3f}  {ps}")
    print("="*90)

    best = max(results, key=lambda x: x.get("score", -999))
    bt = best.get("backtest", {})
    daily = bt.get("daily_return_pct", 0)
    print(f"\nBeste Parameter (daily={daily:.3f}%/Tag ≈ ${daily*100:.2f}/Tag auf $10k):")
    for k, v in best.get("params", {}).items():
        print(f"  {k}: {v}")


# ── Hauptloop ────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hours",       type=float, default=24.0)
    parser.add_argument("--iterations",  type=int,   default=999)
    parser.add_argument("--parallel",    type=int,   default=PARALLEL_K)
    parser.add_argument("--status",      action="store_true")
    parser.add_argument("--no-backtest", action="store_true")
    args = parser.parse_args()

    results = load_results()

    if args.status:
        print_status(results)
        return

    k = args.parallel
    log("=" * 60)
    log(f"AUTOML LOOP | {args.hours}h | {k} parallele Trainings")
    log(f"Metrik: Backtest Tages-Return auf 236 Tage Out-of-Sample Daten")
    log("=" * 60)

    start_time = time.time()
    max_secs   = args.hours * 3600
    round_num  = 0
    best_score = max((r.get("score", -999) for r in results), default=-999)

    while (time.time() - start_time) < max_secs and round_num < args.iterations:
        round_num += 1
        elapsed_h   = (time.time() - start_time) / 3600
        remaining_h = (max_secs - (time.time() - start_time)) / 3600

        log(f"\n{'='*60}")
        log(f"RUNDE {round_num} | {elapsed_h:.1f}h vergangen | {remaining_h:.1f}h verbleibend")
        log(f"{'='*60}")

        # 1. K Kandidaten vorschlagen
        batch = propose_batch(results, k)
        log(f"Teste {len(batch)} Kandidaten parallel:")
        for i, p in enumerate(batch):
            log(f"  Slot {i}: lc={p['lambda_cost']} ld={p['lambda_draw']} lr={p['lambda_regime']} wb={p['win_bonus']} lp={p['loss_penalty']}")

        # 2. Parallel trainieren
        log(f"Starte {k} Trainings parallel ({TRAINING_ITERATIONS} iter je)...")
        train_results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=k) as ex:
            futures = {ex.submit(train_candidate, p, i): i for i, p in enumerate(batch)}
            for fut in concurrent.futures.as_completed(futures):
                try:
                    tr = fut.result()
                    train_results.append(tr)
                    log(f"  Slot {tr['slot']} fertig: return={tr['train_return']:.1f}% in {tr['elapsed_s']//60}min")
                except Exception as e:
                    log(f"  Training-Error: {e}")

        if not train_results:
            log("Alle Trainings fehlgeschlagen — weiter")
            continue

        # 3. Backtest für jeden Kandidaten
        if not args.no_backtest:
            log("Backteste alle Kandidaten auf 236 Tage Out-of-Sample...")
            round_results = []
            for tr in train_results:
                if not tr["success"]:
                    continue
                bt = backtest_model(Path(tr["model_dir"]), tr["params"])
                sc = score(bt, tr)
                result_entry = {
                    "round": round_num,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "params": tr["params"],
                    "train_return": tr["train_return"],
                    "backtest": bt,
                    "score": sc,
                }
                round_results.append(result_entry)
                log(f"  Slot {tr['slot']}: daily={bt.get('daily_return_pct',-999):.3f}%/Tag "
                    f"sharpe={bt.get('sharpe',0):.2f} score={sc:.3f}")

            if not round_results:
                log("Kein Backtest erfolgreich")
                continue

            # 4. Besten Kandidaten dieser Runde deployen
            best_this_round = max(round_results, key=lambda r: r["score"])
            bt_best = best_this_round["backtest"]

            results.extend(round_results)
            save_results(results)

            if best_this_round["score"] > best_score:
                best_score = best_this_round["score"]
                save_best(best_this_round["params"], bt_best.get("daily_return_pct", -999), bt_best)

                log(f"\n★ NEUES BEST: daily={bt_best.get('daily_return_pct',0):.3f}%/Tag "
                    f"≈ ${bt_best.get('daily_return_pct',0)*100:.2f}/Tag auf $10k")
                log("  Deploye bestes Modell zu Paper Trading...")

                # Bestes Modell aus dem richtigen Slot deployen
                best_slot = next(t for t in train_results if t["params"] == best_this_round["params"])
                best_model_src = Path(best_slot["model_dir"]) / "best_model_trader.pth"
                if best_model_src.exists():
                    import shutil
                    dst = WORK_DIR / "data" / "models" / "adversarial" / "best_model_trader.pth"
                    shutil.copy2(best_model_src, dst)
                    subprocess.run(
                        ["python3", "deploy_model.py", "--restart"],
                        cwd=WORK_DIR, capture_output=True,
                    )
                    log("  Deploy abgeschlossen — Paper Trading läuft mit neuem Modell")
            else:
                results.extend(round_results)
                save_results(results)
                log(f"  Kein Verbesserung (best={best_score:.3f})")

        else:
            # Kein Backtest: direkt besten nach Training-Return deployen
            best_train = max(train_results, key=lambda r: r.get("train_return", -999))
            if best_train["success"]:
                result_entry = {
                    "round": round_num,
                    "params": best_train["params"],
                    "train_return": best_train["train_return"],
                    "score": best_train["train_return"],
                }
                results.append(result_entry)
                save_results(results)

        print_status(results)
        log("")

    log("\n" + "="*60)
    log("AUTOML FERTIG")
    log(f"Runden: {round_num} | Bester Score: {best_score:.3f}")
    log("="*60)
    print_status(results)


if __name__ == "__main__":
    main()
