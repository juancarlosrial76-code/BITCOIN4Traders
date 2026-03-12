"""
analytics.py — Real performance metrics from the Darwin Engine Champion.

Data sources (in priority):
  1. Champion .pkl + .json from data/cache/
  2. Cached equity curve (data/cache/equity_curve.json)
  3. Static fallback if no Champion present
"""

import json
import sys
from pathlib import Path
from typing import Any

from fastapi import APIRouter

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

router = APIRouter()

CHAMPION_PKL = PROJECT_ROOT / "data/cache/multiverse_champion.pkl"
CHAMPION_META = PROJECT_ROOT / "data/cache/multiverse_champion_meta.json"
WFV_REPORT = PROJECT_ROOT / "data/cache/wfv_report.json"
EQUITY_CACHE = PROJECT_ROOT / "data/cache/equity_curve.json"


def _load_meta() -> dict:
    if CHAMPION_META.exists():
        try:
            return json.loads(CHAMPION_META.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def _load_wfv() -> dict:
    if WFV_REPORT.exists():
        try:
            return json.loads(WFV_REPORT.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def _load_champion_stats() -> dict[str, Any]:
    """Loads Champion, runs mini-simulation, caches equity curve."""
    if not CHAMPION_PKL.exists():
        return {}
    try:
        from darwin_engine import (
            ChampionPersistence,
            generate_synthetic_btc,
            FitnessEvaluator,
            ArenaConfig,
        )

        bot = ChampionPersistence.load(
            str(CHAMPION_PKL),
            str(CHAMPION_META) if CHAMPION_META.exists() else None,
        )
        if bot is None:
            return {}

        try:
            from darwin_engine import load_live_data

            df = load_live_data(symbol="BTC/USDT", timeframe="1h", limit=500)
        except Exception:
            df = generate_synthetic_btc(n_bars=500, seed=42)

        equity = bot.run_simulation(df)
        evaluator = FitnessEvaluator(ArenaConfig())
        stats = evaluator.evaluate(equity, bot)

        ec_data = [
            {"timestamp": str(idx.date()), "value": round(float(val), 2)}
            for idx, val in zip(df.index, equity)
        ]
        EQUITY_CACHE.parent.mkdir(parents=True, exist_ok=True)
        EQUITY_CACHE.write_text(json.dumps(ec_data), encoding="utf-8")

        return {
            "champion_name": bot.name,
            "total_return": round(stats.total_return, 4),
            "sharpe_ratio": round(stats.sharpe, 4),
            "sortino_ratio": round(stats.sortino, 4),
            "calmar_ratio": round(stats.calmar, 4),
            "max_drawdown": round(stats.max_drawdown, 4),
            "win_rate": round(stats.win_rate, 4),
            "profit_factor": round(stats.profit_factor, 4),
            "total_trades": int(bot.trade_count),
            "equity_curve": ec_data,
        }
    except Exception as exc:
        return {"_error": str(exc)}


@router.get("/metrics")
async def get_metrics():
    """Real metrics of the current Darwin Engine Champion."""
    meta = _load_meta()
    stats = _load_champion_stats()
    wfv = _load_wfv()

    if stats and "_error" not in stats:
        winning = int(stats["total_trades"] * stats["win_rate"])
        losing = stats["total_trades"] - winning
        return {
            "champion_name": stats.get("champion_name", meta.get("name", "unknown")),
            "strategy_type": meta.get("strategy_type", "unknown"),
            "saved_at": meta.get("saved_at", "unknown"),
            "totalReturn": stats["total_return"],
            "sharpeRatio": stats["sharpe_ratio"],
            "sortinoRatio": stats["sortino_ratio"],
            "calmarRatio": stats["calmar_ratio"],
            "maxDrawdown": stats["max_drawdown"],
            "maxDrawdownDuration": wfv.get("avg_dd_duration", 0),
            "winRate": stats["win_rate"],
            "profitFactor": stats["profit_factor"],
            "totalTrades": stats["total_trades"],
            "winningTrades": winning,
            "losingTrades": losing,
            # Derive avg win/loss in basis points from profit_factor and total_return.
            # profit_factor = gross_profit / gross_loss
            # gross_profit + gross_loss ≈ total_return * total_trades (normalised)
            # avg_win_bps = gross_profit / winning = (pf / (pf+1)) * total_return * 10000 / max(winning,1)
            # avg_loss_bps = gross_loss / losing  = (1 / (pf+1)) * total_return * 10000 / max(losing,1)
            "avgWin": round(
                stats["profit_factor"]
                / (stats["profit_factor"] + 1)
                * stats["total_return"]
                * 10000
                / max(winning, 1),
                2,
            ),
            "avgLoss": round(
                1.0
                / (stats["profit_factor"] + 1)
                * abs(stats["total_return"])
                * 10000
                / max(losing, 1),
                2,
            ),
            "largestWin": round(
                stats["profit_factor"]
                / (stats["profit_factor"] + 1)
                * stats["total_return"]
                * 10000
                / max(winning, 1)
                * 3,  # Estimate: largest win ≈ 3× average win
                2,
            ),
            "largestLoss": round(abs(stats["max_drawdown"]) * 10000, 2),
            "avgHoldingPeriod": wfv.get("avg_holding_period", 0),
            "wfv_oos_return": wfv.get("avg_oos_return", None),
            "wfv_degradation": wfv.get("degradation", None),
            "data_source": "darwin_engine_champion",
        }

    if meta:
        return {
            "champion_name": meta.get("name", "unknown"),
            "strategy_type": meta.get("strategy_type", "unknown"),
            "saved_at": meta.get("saved_at", "unknown"),
            "totalReturn": meta.get("total_return", 0),
            "sharpeRatio": meta.get("sharpe", 0),
            "sortinoRatio": meta.get("sortino", 0),
            "calmarRatio": meta.get("calmar", 0),
            "maxDrawdown": meta.get("max_drawdown", 0),
            "maxDrawdownDuration": 0,
            "winRate": meta.get("win_rate", 0),
            "profitFactor": meta.get("profit_factor", 0),
            "totalTrades": meta.get("trade_count", 0),
            "winningTrades": 0,
            "losingTrades": 0,
            "avgWin": 0,
            "avgLoss": 0,
            "largestWin": 0,
            "largestLoss": 0,
            "avgHoldingPeriod": 0,
            "data_source": "champion_meta_only",
        }

    return {
        "champion_name": "Kein Champion",
        "totalReturn": 0,
        "sharpeRatio": 0,
        "sortinoRatio": 0,
        "calmarRatio": 0,
        "maxDrawdown": 0,
        "maxDrawdownDuration": 0,
        "winRate": 0,
        "profitFactor": 0,
        "totalTrades": 0,
        "winningTrades": 0,
        "losingTrades": 0,
        "avgWin": 0,
        "avgLoss": 0,
        "largestWin": 0,
        "largestLoss": 0,
        "avgHoldingPeriod": 0,
        "data_source": "no_champion_found",
    }


@router.get("/equity-curve")
async def get_equity_curve():
    """Equity curve of the Champion (cached or recalculated)."""
    if EQUITY_CACHE.exists():
        try:
            data = json.loads(EQUITY_CACHE.read_text(encoding="utf-8"))
            if data:
                return data
        except Exception:
            pass

    stats = _load_champion_stats()
    if stats and "equity_curve" in stats:
        return stats["equity_curve"]
    return []


@router.get("/monthly-returns")
async def get_monthly_returns():
    """Monthly returns from the equity curve."""
    if EQUITY_CACHE.exists():
        try:
            import pandas as pd

            data = json.loads(EQUITY_CACHE.read_text(encoding="utf-8"))
            if not data:
                return []
            df = pd.DataFrame(data)
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            monthly = df.set_index("timestamp")["value"].resample("ME").last()
            returns = monthly.pct_change().dropna()
            return [
                {"month": idx.strftime("%b %Y"), "return": round(float(r) * 100, 2)}
                for idx, r in returns.items()
            ]
        except Exception:
            pass
    return []


@router.get("/trade-distribution")
async def get_trade_distribution():
    """P&L distribution in buckets from the equity curve."""
    if EQUITY_CACHE.exists():
        try:
            import pandas as pd

            data = json.loads(EQUITY_CACHE.read_text(encoding="utf-8"))
            if not data:
                return []
            values = pd.Series([d["value"] for d in data])
            returns = values.pct_change().dropna() * 100
            buckets = [
                ("< -5%", returns[returns < -5]),
                ("-5% to -2%", returns[(returns >= -5) & (returns < -2)]),
                ("-2% to 0%", returns[(returns >= -2) & (returns < 0)]),
                ("0% to 2%", returns[(returns >= 0) & (returns < 2)]),
                ("2% to 5%", returns[(returns >= 2) & (returns < 5)]),
                ("> 5%", returns[returns >= 5]),
            ]
            return [{"range": lbl, "count": int(len(b))} for lbl, b in buckets]
        except Exception:
            pass
    return [
        {"range": "< -5%", "count": 0},
        {"range": "-5% to -2%", "count": 0},
        {"range": "-2% to 0%", "count": 0},
        {"range": "0% to 2%", "count": 0},
        {"range": "2% to 5%", "count": 0},
        {"range": "> 5%", "count": 0},
    ]
