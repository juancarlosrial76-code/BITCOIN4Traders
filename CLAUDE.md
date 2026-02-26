# BITCOIN4Traders - Project Context

## Documentation System
All conversation logs and session notes are automatically saved in:
`/home/hp17/Tradingbot/BITCOIN4Traders-DE-DOKU/sessions/`

## Rule: At the End of Each Session
Create a brief summary of the conversation in the current session file under:
`/home/hp17/Tradingbot/BITCOIN4Traders-DE-DOKU/sessions/`

Use the following bash command:
```bash
ls -t /home/hp17/Tradingbot/BITCOIN4Traders-DE-DOKU/sessions/session_*.md | head -1
```
to find the most recent session file, and add the key points there.

## Project Overview
- **Project:** BITCOIN4Traders (Reinforcement Learning Trading Bot)
- **Language:** English (conversations), Python (code)
- **Documentation:** `/home/hp17/Tradingbot/BITCOIN4Traders-DE-DOKU/`

## Documentation Structure
```
BITCOIN4Traders-DE-DOKU/
├── 00_Einstieg/          # Getting started, overview
├── 01_Daten/             # Data acquisition & processing
├── 02_Features/          # Feature Engineering
├── 03_Environment/       # Trading Environment (Gym)
├── 04_Agenten_Training/  # RL agent training
├── 05_Mathematik_Risiko/ # Risk management & mathematics
├── 06_Backtesting/       # Backtesting & validation
├── 07_Monitoring/        # Live monitoring & execution
├── 08_Bereits_Erledigt/  # Completed tasks
├── 09_Docs_Improvement/  # Documentation improvements
├── telegram_bot/         # Telegram Bot docs
└── sessions/             # Automatic session logs
```
