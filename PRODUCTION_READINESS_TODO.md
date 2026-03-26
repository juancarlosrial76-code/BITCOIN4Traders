# PRODUCTION READINESS TODO — BITCOIN4Traders

> **Erstellt:** 2026-03-26
> **Basis:** FINAL_PRODUCTION_READINESS_REVIEW.md
> **Aufwand Total:** ~18h

---

## PHASE 1 — SOFORT (blocking für Production)

- [x] F-030: `aiohttp>=3.9.0` zu requirements.txt hinzufügen
- [x] F-025: Docker non-root User (`backend/Dockerfile`)
- [x] F-026: Docker HEALTHCHECK (`backend/Dockerfile`)
- [x] F-005: PM2 REPO Pfad korrigieren (`ecosystem.config.js:22`)
- [x] F-006: PM2 Paper Trading Prozess hinzufügen (`ecosystem.config.js`)
- [x] F-007: PM2 Log Rotation hinzufügen (`ecosystem.config.js`)
- [x] F-001: watchdog.sh 7h Limit entfernen
- [x] F-002: watchdog.sh Live Trading Monitoring hinzufügen
- [x] F-017: backend/main.py print() → logger (3 Stellen)
- [x] F-018: rate_limit.py IP-Pruning gegen Memory Leak
- [x] F-022: PositionsTable.tsx API-Mapping implementieren

## PHASE 2 — DIESE WOCHE

- [ ] F-010: binance_ws_connector.py ListenKey Refresh 1800→1200s
- [ ] F-031: requirements.txt torch Upper Bound `<3.0.0`
- [ ] F-019: /api/trading/status Endpoint Auth hinzufügen
- [ ] F-020: price_stream() asyncio.wait_for Timeout
- [ ] F-015: binance_ws_connector.py Reconnect Backoff

## PHASE 3 — NÄCHSTE WOCHE

- [ ] F-028: Tests PPO Agent + Live Engine (Task #37)
- [ ] F-029: Logging konsolidieren (loguru überall)
- [ ] F-023: Error Boundary in Frontend
- [ ] F-043: CHANGELOG.md erstellen
- [ ] F-044: DEPLOYMENT_GUIDE.md erstellen
- [ ] F-045: TROUBLESHOOTING.md erstellen
- [ ] F-046: API_REFERENCE.md erstellen

## PHASE 4 — NIEDRIGE PRIORITÄT

- [ ] F-032: Unbenutzte packages aus requirements.txt entfernen
- [ ] F-033: pickle.load Validierung
- [ ] F-037/F-038: Test Files nach /tests/ verschieben
- [ ] F-039: build/ in .gitignore
- [ ] F-041: Module Docstrings (9 Dateien)
- [ ] F-042: Type Hints vervollständigen (35 Funktionen)
- [ ] F-021: global → app.state Refactor (Backend)

---

## STATUS NACH SESSION

| Phase | Items | Erledigt |
|-------|-------|---------|
| Phase 1 (Sofort) | 11 | 11 ✅ |
| Phase 2 (Diese Woche) | 5 | 0 |
| Phase 3 (Nächste Woche) | 7 | 0 |
| Phase 4 (Niedrig) | 7 | 0 |
