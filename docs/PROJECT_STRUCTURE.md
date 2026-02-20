# BITCOIN4Traders - Detailed Project Structure

This document provides a comprehensive overview of the BITCOIN4Traders project structure and module purposes.

## 📁 Directory Structure

```
BITCOIN4Traders/
├── src/                          # Core source code
│   ├── __init__.py              # Package initialization
│   │
│   ├── data/                     # Data Infrastructure (Phase 1)
│   │   ├── __init__.py
│   │   ├── ccxt_loader.py       # Exchange data loading via CCXT
│   │   └── data_manager.py      # Data caching and management
│   │
│   ├── features/                 # Feature Engineering (Phase 1)
│   │   ├── __init__.py
│   │   └── feature_engine.py    # Technical indicators and features
│   │
│   ├── environment/              # Trading Environment (Phase 2)
│   │   ├── __init__.py
│   │   ├── realistic_trading_env.py  # Main Gym environment
│   │   ├── config_system.py     # YAML configuration system
│   │   ├── config_integrated_env.py  # Config-integrated environment
│   │   ├── order_book.py        # Order book simulation
│   │   ├── slippage_model.py    # Realistic slippage models
│   │   └── position_actions.py  # Position management actions
│   │
│   ├── math_tools/               # Mathematical Core (Phase 3)
│   │   ├── __init__.py
│   │   ├── ornstein_uhlenbeck.py   # Mean-reversion scoring
│   │   ├── hmm_regime.py        # Hidden Markov Model regimes
│   │   └── kelly_criterion.py   # Optimal position sizing
│   │
│   ├── risk/                     # Risk Management (Phase 4)
│   │   ├── __init__.py
│   │   ├── risk_manager.py      # Circuit breaker, position limits
│   │   └── risk_metrics_logger.py  # Risk tracking and logging
│   │
│   ├── agents/                   # RL Agents (Phase 5)
│   │   ├── __init__.py
│   │   └── ppo_agent.py         # Proximal Policy Optimization
│   │
│   ├── training/                 # Training Infrastructure (Phase 5)
│   │   ├── __init__.py
│   │   └── adversarial_trainer.py  # Self-play adversarial training
│   │
│   ├── backtesting/              # Backtesting (Phase 6)
│   │   ├── __init__.py
│   │   ├── walkforward_engine.py   # Walk-forward validation
│   │   ├── performance_calculator.py  # Performance metrics
│   │   └── visualizer.py        # Visualization and reporting
│   │
│   ├── validation/               # Anti-Bias: Validation
│   │   ├── __init__.py
│   │   └── antibias_walkforward.py  # Purged Walk-Forward CV
│   │
│   ├── costs/                    # Anti-Bias: Transaction Costs
│   │   ├── __init__.py
│   │   └── antibias_costs.py    # Realistic cost models
│   │
│   ├── reward/                   # Anti-Bias: Reward Functions
│   │   ├── __init__.py
│   │   └── antibias_rewards.py  # Risk-adjusted rewards
│   │
│   ├── evaluation/               # Anti-Bias: Statistical Evaluation
│   │   ├── __init__.py
│   │   └── antibias_validator.py  # CPCV, Permutation, DSR, MTRL
│   │
│   ├── orders/                   # Order Management
│   │   └── order_manager.py
│   │
│   ├── execution/                # Live Trading Execution
│   │   └── live_engine.py
│   │
│   ├── connectors/               # Exchange Connectors
│   │   └── binance_ws_connector.py  # Binance WebSocket
│   │
│   └── monitoring/               # System Monitoring
│       └── monitor.py
│
├── config/                       # Configuration Files
│   ├── environment/             # Environment configurations
│   │   └── realistic_env.yaml
│   ├── training/                # Training configurations
│   │   └── adversarial.yaml
│   └── phase7.yaml              # Phase 7 configuration
│
├── tests/                        # Unit Tests
│   ├── conftest.py              # Pytest configuration
│   ├── test_antibias_integration.py  # Anti-bias framework tests
│   ├── test_integration.py      # Integration tests
│   ├── test_phase2_environment.py    # Environment tests
│   ├── test_phase4_risk_management.py  # Risk management tests
│   └── test_phase5_adversarial_training.py  # Training tests
│
├── docs/                         # Documentation
│   ├── ANTIBIAS_INTEGRATION.md  # Anti-bias framework guide
│   ├── PHASE1_REPORT.txt        # Phase 1 documentation
│   ├── PHASE2_REPORT.txt        # Phase 2 documentation
│   ├── PHASE3_REPORT.txt        # Phase 3 documentation
│   ├── PHASE3_FUNCTION_SUMMARY.txt  # Function summary
│   ├── PHASE4_REPORT.txt        # Phase 4 documentation
│   ├── PHASE5_REPORT.txt        # Phase 5 documentation
│   ├── PHASE6_REPORT.txt        # Phase 6 documentation
│   └── PROJECT_STRUCTURE.md     # This file
│
├── train.py                      # Main training script
├── run.py                        # Main execution script
├── auto_train.py                 # Automated training (8h)
├── auto_12h_train.py             # Automated training (12h)
├── requirements.txt              # Python dependencies
├── setup.py                      # Package installation
├── .gitignore                    # Git ignore rules
├── LICENSE                       # MIT License
├── README.md                     # Main documentation
└── CONTRIBUTING.md               # Contribution guidelines
```

## 📦 Module Descriptions

### Phase 1: Data Infrastructure

#### `src/data/`
- **ccxt_loader.py**: Downloads market data from exchanges (Binance, etc.) using CCXT library
- **data_manager.py**: Caches data locally, handles data updates and validation

#### `src/features/`
- **feature_engine.py**: Generates 50+ technical indicators (RSI, MACD, Bollinger Bands, etc.)

### Phase 2: Market Simulation

#### `src/environment/`
- **realistic_trading_env.py**: Main Gymnasium environment with realistic market simulation
- **config_system.py**: YAML-based configuration management
- **order_book.py**: Simulates Level 2 order book with bid/ask spreads
- **slippage_model.py**: Models price impact and slippage based on order size

### Phase 3: Mathematical Core

#### `src/math_tools/`
- **ornstein_uhlenbeck.py**: Mean-reversion process for statistical arbitrage signals
- **hmm_regime.py**: Hidden Markov Model for market regime detection (Bull/Bear/Neutral)
- **kelly_criterion.py**: Optimal position sizing based on edge and variance

### Phase 4: Risk Management

#### `src/risk/`
- **risk_manager.py**: Implements circuit breakers, position limits, and risk checks
- **risk_metrics_logger.py**: Tracks Sharpe, Sortino, Calmar ratios, drawdowns

### Phase 5: Adversarial RL

#### `src/agents/`
- **ppo_agent.py**: Proximal Policy Optimization agent with actor-critic architecture

#### `src/training/`
- **adversarial_trainer.py**: Self-play training where agent competes against adversary

### Phase 6: Backtesting

#### `src/backtesting/`
- **walkforward_engine.py**: Walk-forward analysis to prevent overfitting
- **performance_calculator.py**: Computes 25+ performance metrics
- **visualizer.py**: Generates professional trading reports

### Phase 7: Anti-Bias Framework

#### `src/validation/`
- **antibias_walkforward.py**: Purged Walk-Forward CV with embargo periods and leak detection

#### `src/costs/`
- **antibias_costs.py**: Realistic transaction costs (fees, spread, slippage, funding)

#### `src/reward/`
- **antibias_rewards.py**: Risk-adjusted reward functions (Sharpe, Cost-Aware, Regime-Aware)

#### `src/evaluation/`
- **antibias_validator.py**: Statistical validation suite (CPCV, Permutation Test, DSR, MTRL)

## 🔧 Configuration Files

### `config/environment/realistic_env.yaml`
Environment configuration including:
- Initial capital
- Transaction costs
- Slippage parameters
- Order book settings
- Reward configuration

### `config/training/adversarial.yaml`
Training configuration including:
- Learning rates
- Batch sizes
- Network architecture
- Adversarial training parameters

## 🧪 Test Files

### `tests/test_antibias_integration.py`
Tests for the anti-bias framework:
- Purged Walk-Forward CV
- Transaction costs
- Reward functions
- Statistical validators

### `tests/test_phase2_environment.py`
Tests for trading environment:
- Environment initialization
- Step functionality
- Reward calculation
- Order execution

### `tests/test_phase4_risk_management.py`
Tests for risk management:
- Circuit breaker
- Position sizing
- Risk metrics

### `tests/test_phase5_adversarial_training.py`
Tests for adversarial training:
- PPO agent
- Training loop
- Checkpoint saving/loading

## 📜 Documentation Files

### `docs/ANTIBIAS_INTEGRATION.md`
Comprehensive guide for using the anti-bias framework:
- Purged Walk-Forward CV usage
- Transaction cost models
- Reward function selection
- Statistical validation

### `docs/PHASE*_REPORT.txt`
Detailed reports for each development phase with:
- Implementation details
- Performance benchmarks
- Usage examples

## 🚀 Execution Scripts

### `train.py`
Main training script with CLI arguments:
```bash
python train.py --iterations 500 --device cuda
```

### `run.py`
Main execution script for running trained models

### `auto_train.py`
Automated training with duration parameter:
```bash
python auto_train.py --duration 8h
```

### `auto_12h_train.py`
Extended automated training (12 hours)

## 📋 Dependencies

### `requirements.txt`
Key dependencies:
- PyTorch (deep learning)
- Gymnasium (RL environments)
- CCXT (exchange connectivity)
- NumPy/Pandas (data processing)
- Numba (performance optimization)

### `setup.py`
Package installation configuration for `pip install -e .`

## 🔒 Important Files

### `.gitignore`
Excludes from version control:
- Python cache files (`__pycache__`, `*.pyc`)
- Model weights (`*.pth`)
- Logs (`logs/`)
- Data files (`data/`)
- Virtual environment (`venv/`)

### `LICENSE`
MIT License - Open source with attribution

## 📝 Notes

1. **No `__pycache__`**: All cache directories are excluded from the mirror
2. **No model weights**: Trained models are not included (regenerated via training)
3. **No logs**: Log files are excluded
4. **Clean structure**: Only source code, configs, tests, and docs
5. **English documentation**: All documentation is in English

## 🎯 Usage Flow

1. **Data Loading**: `src/data/ccxt_loader.py` → Load market data
2. **Feature Engineering**: `src/features/feature_engine.py` → Generate indicators
3. **Environment**: `src/environment/realistic_trading_env.py` → Create trading env
4. **Training**: `src/training/adversarial_trainer.py` → Train PPO agent
5. **Validation**: `src/evaluation/antibias_validator.py` → Validate with anti-bias
6. **Backtesting**: `src/backtesting/walkforward_engine.py` → Walk-forward analysis
7. **Live Trading**: `src/execution/live_engine.py` → Execute live trades

---

**Last Updated:** 2026-02-18  
**Version:** 1.0.0
