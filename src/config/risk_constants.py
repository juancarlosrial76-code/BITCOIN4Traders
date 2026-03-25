"""
Risk Constants — Single source of truth for all drawdown / risk limits.
=======================================================================

Three contexts with intentionally different limits:

1. LIVE TRADING (circuit breaker)
   - Controlled by RiskConfig.max_drawdown_per_session (default 2%)
   - Controlled by EngineConfig.circuit_breaker_pct (default 2%)
   - These are the SAME limit via two different config objects.
     TODO: merge into a single live-trading config.
   - Tight because real money is at stake.

2. TRAINING ENVIRONMENT (episode termination)
   - Controlled by EnvConfig.max_drawdown (default 15–20%)
   - Deliberately wider so the agent can explore adverse scenarios
     without terminating every episode too early.
   - realistic_trading_env.py uses 15% (tighter variant).
   - config_system.py uses 20% (standard training default).

3. ARCHIVE / RESEARCH (darwin_legacy.py)
   - Uses 20% — research context, not production.

Summary table
-------------
Context                       | Limit | File
------------------------------|-------|-------------------------------
Live circuit breaker          |  2%   | risk_manager.py, live_engine.py
Realistic env termination     | 15%   | realistic_trading_env.py
Standard training termination | 20%   | config_system.py
"""

# ── Live Trading ─────────────────────────────────────────────────────────────
LIVE_MAX_DRAWDOWN: float = 0.02        # 2%  — circuit breaker in production
LIVE_DAILY_LOSS_LIMIT_USD: float = 500  # $500 — daily loss hard stop

# ── Training Environment ─────────────────────────────────────────────────────
TRAIN_MAX_DRAWDOWN: float = 0.15       # 15% — episode termination during training
TRAIN_MAX_DRAWDOWN_RELAXED: float = 0.20  # 20% — used in config_system.py default

# ── Min Capital Threshold ─────────────────────────────────────────────────────
LIVE_MIN_CAPITAL_FRACTION: float = 0.30  # 30% of initial — below this: reduce sizing
