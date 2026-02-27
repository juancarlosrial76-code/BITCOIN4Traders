# BITCOIN4Traders - Developer Guide

**Version:** 2.0 (February 2026)  
**Status:** SOTA (State of the Art)

---

## Table of Contents

1. [Quick Start](#1-quick-start)
2. [System Architecture](#2-system-architecture)
3. [Training Pipeline](#3-training-pipeline)
4. [Module Overview](#4-module-overview)
5. [Live Trading](#5-live-trading)
6. [Dependencies](#6-dependencies)
7. [For Contributors](#7-for-contributors)

---

## 1. Quick Start

### Prerequisites

| Requirement | Training (Colab) | Live Trading (Server) |
|-------------|------------------|----------------------|
| GPU | T4 (15GB VRAM) | Not required |
| RAM | 12+ GB | 4+ GB |
| Storage | 50+ GB | 20+ GB |
| OS | Colab (Linux) | Ubuntu/Debian |

### Installation

```bash
# Clone repository
git clone https://github.com/juancarlosrial76-code/BITCOIN4Traders.git
cd BITCOIN4Traders

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux

# Install dependencies
pip install -r requirements.txt

# Test
python -m pytest tests/ -v
```

### Training Pipeline (Colab)

| Notebook | Purpose |
|----------|---------|
| `Colab_1_Daten.ipynb` | Load data + compute features |
| `Colab_2_Features.ipynb` | Feature optimization (optional) |
| `Colab_3_PPO_Training.ipynb` | PPO training on GPU |

### Live Trading

```bash
# Configure
cp config/base/config.yaml config/production/config.yaml
# Edit config.yaml

# Run
python run.py
```

---

## 2. System Architecture

### 7 Main Components

```
┌─────────────────────────────────────────────────────────────┐
│                    BITCOIN4Traders Pipeline                │
└─────────────────────────────────────────────────────────────┘

  [1. DATA]    [2. FEATURES]   [3. ENVIRONMENT]
      │              │                │
      ▼              ▼                ▼
  ccxt_loader   feature_engine  realistic_trading_env
  data_manager  multi_tf        order_book
               microstructure   slippage

                       │
                       ▼
  [7. MONITOR]  [6. BACKTEST]   [4. AGENTS]
       │              │                │
       ▼              ▼                ▼
  monitor.py    performance     ppo_agent.py
  production    walkforward     drl_agents
```

### Data Flow

```
Training (Colab):
  Binance API → CCXT → Parquet → FeatureEngine → PPO → Drive

Live Trading (Server):
  Binance API → Environment → Agent → Order Manager → Binance
```

---

## 3. Training Pipeline

### Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `N_ITERATIONS` | 500 | Training iterations |
| `STEPS_PER_ITER` | 2048 | Steps per iteration |
| `BATCH_SIZE` | 256 (GPU) / 64 (CPU) | Batch size |
| `HIDDEN_DIM` | 256 (GPU) / 128 (CPU) | GRU hidden size |
| `ACTOR_LR` | 1e-4 | Actor learning rate |
| `CRITIC_LR` | 3e-4 | Critic learning rate |
| `GAMMA` | 0.99 | Discount factor |
| `GAE_LAMBDA` | 0.95 | GAE lambda |
| `CLIP_EPSILON` | 0.2 | PPO clipping |
| `ENTROPY_COEF` | 0.08 | Entropy bonus |
| `ADVERSARY_START` | 100 | Adversary starts iteration 100 |
| `ADVERSARY_STRENGTH` | 0.1 | 10% weight |

### Anti-Bias Framework

Prevents overfitting in backtesting:

| Method | Description |
|--------|-------------|
| **CPCV** | Combinatorial Purged Cross-Validation |
| **Purged Walk-Forward** | Walk-Forward with embargo |
| **Deflated Sharpe** | Sharpes corrected for multiple testing |
| **Transaction Costs** | Realistic 5 bps |

---

## 4. Module Overview

### Core Modules

| Directory | Purpose | Key Files |
|-----------|---------|-----------|
| `src/data/` | Data loading | ccxt_loader, data_manager |
| `src/features/` | Feature engineering | feature_engine, multi_tf |
| `src/environment/` | RL Environment | realistic_trading_env, order_book |
| `src/agents/` | RL Agents | ppo_agent, drl_agents |
| `src/training/` | Training loop | adversarial_trainer |
| `src/math_tools/` | Math models | kelly, garch, hurst |
| `src/risk/` | Risk management | risk_manager |
| `src/backtesting/` | Validation | performance_calculator, antibias_validator |

### Key Classes

```python
# Environment
from src.environment.realistic_trading_env import RealisticTradingEnv, TradingEnvConfig
env = RealisticTradingEnv(price_data, features, config)

# Agent
from src.agents.ppo_agent import PPOAgent, PPOConfig
agent = PPOAgent(config, device="cuda")

# Risk
from src.risk.risk_manager import RiskManager, RiskConfig
risk_mgr = RiskManager(config, initial_capital=100000)

# Features
from src.features.feature_engine import FeatureEngine, FeatureConfig
engine = FeatureEngine(config)
features = engine.fit_transform(price_data)
```

---

## 5. Live Trading

### Architecture

```
run.py
  │
  ├── binance_connector.py    (API)
  ├── feature_engine.py       (Features)
  ├── realistic_trading_env.py (Environment)
  ├── ppo_agent.py            (Agent)
  ├── order_manager.py        (Execution)
  └── risk_manager.py         (Risk Control)
```

### Configuration

Create `config/production/config.yaml`:

```yaml
environment:
  initial_capital: 100000
  transaction_cost_bps: 5.0
  max_position_size: 0.25

risk:
  max_drawdown: 0.15
  circuit_breaker: true

agent:
  model_path: data/models/ppo_best.pt
```

### Running

```bash
# With PM2
pm2 start ecosystem.config.js

# Direct
python run.py
```

---

## 6. Dependencies

### Python Packages

| Package | Version | Purpose |
|---------|---------|---------|
| `torch` | >=2.0 | Deep Learning |
| `gymnasium` | >=0.29 | RL Environment |
| `numpy` | >=1.24 | Numerics |
| `pandas` | >=2.0 | DataFrames |
| `scikit-learn` | >=1.3 | ML Utils |
| `ccxt` | >=4.0 | Exchange API |
| `loguru` | >=0.7 | Logging |
| `pyyaml` | >=6.0 | Config |
| `numba` | >=0.58 | JIT |
| `scipy` | >=1.11 | Statistics |

### Environment Variables

```bash
# .env
BINANCE_API_KEY=your_key
BINANCE_API_SECRET=your_secret
TELEGRAM_BOT_TOKEN=your_token
DATABASE_URL=postgresql://...
```

---

## 7. For Contributors

### Code Style

```bash
# Format
black src/
isort src/

# Lint
ruff src/

# Tests
pytest tests/ -v
```

### Key Files

| File | Purpose |
|------|---------|
| `run.py` | Main entry for live trading |
| `train.py` | Training script |
| `risk_engine.py` | Risk engine |
| `darwin_engine.py` | Strategy evolution |

### Module Structure

```
src/
├── agents/        # RL Agents (PPO, DQN, etc.)
├── backtesting/   # Performance, validation
├── data/          # Data loading
├── environment/   # Gymnasium environment
├── features/      # Feature engineering
├── math_tools/    # Mathematical models
├── risk/          # Risk management
├── training/      # Training loops
└── ...
```

---

## SOTA Methods

### Reinforcement Learning

- **PPO** with GRU
- **Adversarial Training** (prevents overfitting)
- **GAE** (Generalized Advantage Estimation)

### Feature Engineering

- Multi-timeframe analysis
- Order flow (VPIN)
- Regime detection (HMM)
- Spectral analysis (FFT)

### Risk Management

- Kelly Criterion
- Value at Risk (VaR)
- Circuit Breaker
- Anti-Bias Framework

---

**Last Updated:** 2026-02-27  
**Maintained by:** Juan Carlos Rial
