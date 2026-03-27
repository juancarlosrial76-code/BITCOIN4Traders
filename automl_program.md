# AutoML Program — BITCOIN4Traders
## Ziel
**Maximaler Tages-Gewinn**, gemessen in Paper Trading (Dry Run).
Das ist die einzige Metrik die zählt. Training-Return ist nur ein Proxy.

## Was optimiert wird
Die Reward-Parameter von `WinRateAwareReward`:
- `lambda_cost`    — Strafe für Transaktionskosten (verhindert Overtrading)
- `lambda_draw`    — Strafe für Drawdown
- `lambda_regime`  — Bonus für Regime-Alignment
- `win_bonus`      — Bonus wenn Trade gewinnt (skaliert mit PnL)
- `loss_penalty`   — Strafe wenn Trade verliert (skaliert mit PnL)

## Suchraum
```
lambda_cost:   [0.5 .. 5.0]   — hoch = weniger Trades, mehr Netto-Gewinn
lambda_draw:   [0.5 .. 4.0]   — hoch = konservativeres Risiko
lambda_regime: [0.2 .. 1.5]   — Regime-Signal Gewicht
win_bonus:     [0.1 .. 1.0]   — klein halten damit Return-Signal dominiert
loss_penalty:  [0.2 .. 2.0]   — Verlust-Asymmetrie
```

## Iterations-Budget
- Training: 10 Iterationen pro Experiment (~8-12 Min)
- Paper Trading Messung: 30 Min nach Deploy
- Overnight: ~14 Experimente in 8h

## Erfolgs-Kriterium
Paper Trading Realized PnL > $0 nach 30 Min Messung
Tages-Hochrechnung: realized_pnl_30min × 48 = geschätzter Tages-Gewinn

## Strategie für nächste Parameter
1. Beste bisherige Parameter als Ausgangspunkt
2. Gauß-Perturbation (σ = 20% des Wertebereichs)
3. Niemals probierte Richtung bevorzugen (Exploration)
4. Wenn letzter Run besser: aggressiver in gleicher Richtung (Exploitation)
