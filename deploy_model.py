#!/usr/bin/env python3
"""
deploy_model.py — Brücke Training → Paper Trading
===================================================
Kopiert das beste Adversarial-Trainer-Modell nach data/models/ppo_best.pt
und erstellt champion.json damit run.py es laden kann.

Startet Paper Trading danach automatisch neu (wenn --restart angegeben).

Usage:
    python deploy_model.py                        # deploy + kein restart
    python deploy_model.py --restart              # deploy + paper trading restart
    python deploy_model.py --check                # nur Status prüfen
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
SRC_TRADER = ROOT / "data" / "models" / "adversarial" / "best_model_trader.pth"
DST_MODEL  = ROOT / "data" / "models" / "ppo_best.pt"
CHAMPION   = ROOT / "data" / "models" / "champion.json"
SCALER_SRC = ROOT / "data" / "processed" / "feature_scaler.joblib"
SCALER_DST = ROOT / "data" / "scalers"


def check_status():
    print("=== Deploy Status ===")
    print(f"Source model  : {'✓' if SRC_TRADER.exists() else '✗'} {SRC_TRADER}")
    print(f"Deployed model: {'✓' if DST_MODEL.exists() else '✗'} {DST_MODEL}")
    print(f"Champion JSON : {'✓' if CHAMPION.exists() else '✗'} {CHAMPION}")
    print(f"Scaler src    : {'✓' if SCALER_SRC.exists() else '✗'} {SCALER_SRC}")
    scaler_dst = SCALER_DST / "feature_scaler.pkl"
    print(f"Scaler dst    : {'✓' if scaler_dst.exists() else '✗'} {scaler_dst}")
    if CHAMPION.exists():
        meta = json.loads(CHAMPION.read_text())
        print(f"\nDeployed model info:")
        for k, v in meta.items():
            print(f"  {k}: {v}")


def deploy():
    if not SRC_TRADER.exists():
        print(f"ERROR: Source model not found: {SRC_TRADER}")
        sys.exit(1)

    # Load model to extract config
    ckpt = torch.load(SRC_TRADER, weights_only=False, map_location="cpu")
    cfg = ckpt["config"]
    state_dim  = int(cfg.state_dim)
    hidden_dim = int(cfg.hidden_dim)
    n_actions  = int(cfg.n_actions)

    # Copy model
    DST_MODEL.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(SRC_TRADER, DST_MODEL)
    print(f"✓ Model deployed: {SRC_TRADER.name} → {DST_MODEL}")

    # Get training metrics from main checkpoint
    main_ckpt_path = ROOT / "data" / "models" / "adversarial" / "best_model.pth"
    best_return = 0.0
    n_iter = 0
    if main_ckpt_path.exists():
        main = torch.load(main_ckpt_path, weights_only=False, map_location="cpu")
        best_return = float(main.get("best_return", 0.0))
        n_iter = int(main.get("iteration", 0))

    # Write champion.json
    meta = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "state_dim": state_dim,
        "hidden_dim": hidden_dim,
        "n_actions": n_actions,
        "n_iterations": n_iter,
        "best_return": round(best_return * 100, 2),
        "device": "cpu",
        "source": str(SRC_TRADER),
    }
    CHAMPION.write_text(json.dumps(meta, indent=2))
    print(f"✓ champion.json written: state_dim={state_dim} n_actions={n_actions} iter={n_iter}")

    # Deploy scaler — run.py expects data/scalers/feature_scaler.pkl
    # but training saves data/processed/feature_scaler.joblib (same content, different name)
    SCALER_DST.mkdir(parents=True, exist_ok=True)
    scaler_dst = SCALER_DST / "feature_scaler.joblib"  # load_scaler() looks for .joblib first
    if SCALER_SRC.exists():
        shutil.copy2(SCALER_SRC, scaler_dst)
        print(f"✓ Scaler deployed: feature_scaler.joblib → {scaler_dst}")
    else:
        print(f"⚠ Scaler not found at {SCALER_SRC} — paper trading will use unscaled features")

    print("\n✓ Deploy complete.")
    return meta


def restart_paper_trading():
    """Kill existing run.py and start fresh so it picks up the new model."""
    import subprocess

    # Find and kill existing paper trading
    result = subprocess.run(
        ["pgrep", "-f", "run.py --dry_run"],
        capture_output=True, text=True
    )
    if result.stdout.strip():
        for pid in result.stdout.strip().split():
            try:
                os.kill(int(pid), signal.SIGTERM)
                print(f"✓ Killed old paper trading PID={pid}")
            except ProcessLookupError:
                pass

    import time
    time.sleep(2)

    log_path = ROOT / "logs" / "paper" / f"paper_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    proc = subprocess.Popen(
        [sys.executable, "run.py", "--dry_run"],
        cwd=ROOT,
        stdout=open(log_path, "w"),
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    print(f"✓ Paper trading restarted PID={proc.pid} log={log_path}")
    return proc.pid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--restart", action="store_true", help="Restart paper trading after deploy")
    parser.add_argument("--check",   action="store_true", help="Only show status, no deploy")
    args = parser.parse_args()

    if args.check:
        check_status()
        return

    meta = deploy()

    if args.restart:
        pid = restart_paper_trading()
        print(f"\nPaper trading live with model: state_dim={meta['state_dim']} n_actions={meta['n_actions']}")
    else:
        print("\nRun with --restart to also restart paper trading.")


if __name__ == "__main__":
    main()
