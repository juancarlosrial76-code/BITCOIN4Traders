# BITCOIN4Traders — Colab Notebooks

> Reinforcement Learning Trading Bot for BTC/USDT using PPO + GRU

---

## Table of Contents

- [Architecture](#architecture)
- [Notebooks](#notebooks)
- [Prerequisites](#prerequisites)
- [Costs](#costs)
- [Quick Start](#quick-start)
- [Setup](#setup)
- [Notebook Details](#notebook-details)
- [Colab Shortcuts](#colab-shortcuts)
- [Troubleshooting](#troubleshooting)
- [FAQ](#faq)
- [Changelog](#changelog)
- [Links](#links)

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Google Drive                         │
│  MyDrive/BITCOIN4Traders/                               │
│  ├── data/                                              │
│  │   ├── BTC_USDT_1h_raw.parquet      ← from Notebook 1 │
│  │   ├── train_feat.parquet           ← from Notebook 1 │
│  │   └── scaler.pkl                   ← from Notebook 1 │
│  └── models/                                            │
│      ├── multiverse_champion.pkl      ← from Notebook 2 │
│      └── checkpoint_iter_*.pth        ← from Notebook 3 │
└─────────────────────────────────────────────────────────┘
        │              │              │
        ▼              ▼              ▼
   ┌─────────┐   ┌──────────┐   ┌──────────┐
   │ 1: Data │──▶│2: Evolve │──▶│3: PPO    │
   │  (CPU)  │   │  (CPU)   │   │  (GPU)   │
   └─────────┘   └──────────┘   └──────────┘
        │
        ▼
   ┌──────────┐   ┌──────────┐
   │ All-in-1 │   │ Tune     │
   │ Train    │   │ (GPU)    │
   └──────────┘   └──────────┘
```

---

## Notebooks

| # | File | Purpose | Runtime | Duration |
|---|------|---------|---------|----------|
| 1 | `Colab_1_Daten.ipynb` | Download BTC data, compute features | **None (CPU)** | ~5 min |
| 2 | `Colab_2_Evolution.ipynb` | Genetic evolution of strategy champion | **None (CPU)** | ~15 min |
| 3 | `Colab_3_PPO_Training.ipynb` | PPO + Adversarial RL training | **T4 GPU** | ~60 min |
| 4 | `BITCOIN4Traders_Colab.ipynb` | All-in-one (data + training + eval) | **T4 GPU** | ~90 min |
| 5 | `B4T_Tune.ipynb` | Autonomous hyperparameter optimization | **T4 GPU** | ~120 min |

---

## Prerequisites

| Requirement | Details |
|-------------|---------|
| **Google Account** | For Colab + Drive |
| **Google Drive** | Min 2 GB free space |
| **GitHub Account** | For cloning the repo |
| **BTC Exchange** | Binance, KuCoin, or Yahoo Finance (no API key needed for public data) |
| **Browser** | Chrome, Firefox, Edge (recommended) |
| **API Keys** | Optional: `GEMINI_API_KEY` or `ANTHROPIC_API_KEY` for tuning |

**Before your first run:**
1. Create a Google Drive folder: `MyDrive/BITCOIN4Traders/`
2. Clone or fork the GitHub repo
3. (Optional) Add API keys to Colab Secrets (🔑 icon)

---

## Costs

### Google Colab Pricing

| Plan | GPU | RAM | Timeout | Cost |
|------|-----|-----|---------|------|
| **Free** | T4 (sometimes) | 12 GB | ~90 min | $0/month |
| **Pro** | T4 (guaranteed) | 25 GB | 24 hours | $10/month |
| **Pro+** | T4/V100/A100 | 52 GB | 24 hours | $50/month |
| **Pay-as-you-go** | A100 | 80 GB | 24 hours | ~$1.50/hour |

### Recommended Setup

| Use Case | Plan | Notes |
|----------|------|-------|
| **Learning / Testing** | Free | GPU not always available |
| **Regular Training** | Pro ($10/mo) | Guaranteed T4, no interruptions |
| **Heavy Tuning** | Pro+ ($50/mo) | A100 for faster experiments |

### API Costs (Optional — for B4T_Tune.ipynb)

| Provider | Model | Cost per Experiment | 20 Experiments |
|----------|-------|---------------------|----------------|
| Gemini | gemini-2.0-flash | Free | $0 |
| Claude | claude-sonnet-4-6 | ~$0.10 | ~$2 |

### Estimated Monthly Costs

| Scenario | Colab | API | Total |
|----------|-------|-----|-------|
| Split Pipeline (3 notebooks) | Free | $0 | **$0** |
| Split Pipeline (Pro) | $10 | $0 | **$10** |
| All-in-One (Pro) | $10 | $0 | **$10** |
| Tuning with Gemini (Pro) | $10 | $0 | **$10** |
| Tuning with Claude (Pro) | $10 | $2 | **$12** |

---

## Quick Start

### Option A: Split Pipeline (Recommended)

Run notebooks **in order**. Each notebook reads from Drive and writes results back.

```
Step 1: Colab_1_Daten.ipynb          (CPU only, 5 min)
         ↓ writes data to Drive
Step 2: Colab_2_Evolution.ipynb      (CPU only, 15 min)
         ↓ writes champion to Drive
Step 3: Colab_3_PPO_Training.ipynb   (GPU required, 60 min)
         ↓ writes model to Drive
```

**Why split?** A single notebook doing everything consumes ~12 GB RAM after 1 hour. Splitting ensures each task starts in a fresh session with clean memory.

### Option B: All-in-One

```
BITCOIN4Traders_Colab.ipynb          (GPU required, ~90 min)
```

Runs everything in one session: data → features → training → evaluation.

### Option C: Hyperparameter Tuning

```
B4T_Tune.ipynb                       (GPU required, ~120 min)
```

Automatically searches for optimal PPO parameters using:
- **Grid Search** — no API key needed
- **Gemini** — free tier, requires `GEMINI_API_KEY` in Colab Secrets
- **Claude** — paid, requires `ANTHROPIC_API_KEY` in Colab Secrets

---

## Setup

### 1. Google Colab

Go to [colab.research.google.com](https://colab.research.google.com) and open each notebook.

### 2. Runtime Type

| Notebook | Runtime |
|----------|---------|
| `Colab_1_Daten.ipynb` | None (CPU) |
| `Colab_2_Evolution.ipynb` | None (CPU) |
| `Colab_3_PPO_Training.ipynb` | T4 GPU |
| `BITCOIN4Traders_Colab.ipynb` | T4 GPU |
| `B4T_Tune.ipynb` | T4 GPU |

**How to set:** Runtime → Change runtime type → Hardware accelerator → GPU (T4)

### 3. Google Drive

All notebooks mount Google Drive at startup. Data persists across sessions:

```
MyDrive/BITCOIN4Traders/
├── data/                          # Raw & processed data
│   ├── BTC_USDT_1h_raw.parquet
│   ├── train_feat.parquet
│   ├── val_feat.parquet
│   └── test_feat.parquet
├── models/                        # Checkpoints & champions
│   ├── checkpoint_iter_100.pth
│   ├── checkpoint_iter_200.pth
│   ├── best_model_trader.pth
│   └── multiverse_champion.pkl
└── logs/                          # Training logs
```

### 4. API Keys (Optional)

For `B4T_Tune.ipynb` LLM-guided tuning:

1. Click the 🔑 icon in Colab's left sidebar
2. Add secrets:
   - `GEMINI_API_KEY` — for Gemini (free)
   - `ANTHROPIC_API_KEY` — for Claude (paid)

---

## Notebook Details

### Colab_1_Daten.ipynb — Data Preparation

**What it does:**
- Downloads BTC/USDT 1h OHLCV data from Binance or Yahoo Finance
- Computes technical indicators (RSI, EMA, ATR, VWAP, etc.)
- Fits and saves the feature scaler
- Saves everything to Google Drive

**Output on Drive:**
```
MyDrive/BITCOIN4Traders/data/
├── BTC_USDT_1h_raw.parquet        # Raw price data
├── train_feat.parquet             # Training features
├── val_feat.parquet               # Validation features
├── test_feat.parquet              # Test features
└── scaler.pkl                     # StandardScaler for features
```

---

### Colab_2_Evolution.ipynb — Darwin Evolution

**What it does:**
- Loads raw data from Drive (written by Notebook 1)
- Runs `darwin_engine.run_multiverse()` — genetic evolution
- Tests strategies across 155+ scenarios:
  - Original BTC data
  - Flash crash (30% drop)
  - Slow bear market
  - Sideways chop
  - Parabolic bull run
  - 50 Monte Carlo simulations per regime
- Elimination rule: Drawdown > 20% in ANY scenario → disqualified
- Saves champion genotype to Drive

**Output on Drive:**
```
MyDrive/BITCOIN4Traders/data/
├── multiverse_champion.pkl        # Champion genotype
└── multiverse_champion_meta.json  # Champion metadata
```

---

### Colab_3_PPO_Training.ipynb — PPO Training

**What it does:**
- Loads precomputed features from Drive (written by Notebook 1)
- Creates trading environment (VecEnv with 16 parallel envs)
- Trains PPO agent with GRU network:
  - 500 iterations
  - 256 hidden dimensions
  - 512 batch size
  - Mixed precision (AMP) for speed
- Auto-saves every 10 minutes (Colab timeout protection)
- Supports curriculum training (3-phase anti-bias)
- Evaluates on validation set

**Output on Drive:**
```
MyDrive/BITCOIN4Traders/models/
├── checkpoint_iter_100.pth
├── checkpoint_iter_200.pth
├── best_model_trader.pth
└── ppo_best.pt                    # For live trading
```

---

### BITCOIN4Traders_Colab.ipynb — All-in-One

**What it does:**
- Everything from Notebooks 1-3 in one session
- Plus: Evaluation, Multiverse Evolution, Risk Engine, Plotly Dashboard
- Best for: Quick experiments, single-session runs

**Cells:**
| Cell | Purpose |
|------|---------|
| 0b | Error handler setup |
| 1 | GPU check |
| 2 | Google Drive mount |
| 3 | Repository clone |
| 4 | Package installation |
| 5 | Python path setup |
| 6 | Load cached data (optional) |
| 7 | Training configuration |
| 8 | Download BTC data |
| 9 | Feature engineering |
| 10 | Trading environment |
| 11 | PPO trainer creation |
| 12 | Load checkpoint (optional) |
| 13 | Auto-save setup |
| 14 | Start training |
| 14b | Curriculum training (optional) |
| 15 | Evaluation |
| 16 | List Drive checkpoints |
| 17 | Memory utilities |
| 18 | Multiverse evolution |
| 19 | Risk engine |
| 20 | Plotly dashboard |

---

### B4T_Tune.ipynb — Hyperparameter Optimization

**What it does:**
- Runs baseline training (50 iterations)
- Autonomous loop: patch YAML → train → compare → keep/revert
- 20 experiments maximum
- Keeps only changes that improve Sharpe Ratio

**Search options:**

| Option | API Key | Model | Intelligence |
|--------|---------|-------|--------------|
| A: Grid Search | None | — | Fixed grid |
| B: Gemini | `GEMINI_API_KEY` | gemini-2.0-flash | LLM-guided |
| C: Claude | `ANTHROPIC_API_KEY` | claude-sonnet-4-6 | LLM-guided (best) |

**Searchable parameters:**
- `trader.actor_lr` — learning rate (1e-6 to 1e-2)
- `trader.entropy_coef` — entropy coefficient (0.001 to 0.5)
- `trader.hidden_dim` — network size (64, 128, 256, 512)
- `trader.clip_epsilon` — PPO clip (0.05 to 0.5)
- `trader.n_epochs` — training epochs (1 to 30)
- `trader.batch_size` — batch size (32, 64, 128)
- `trader.target_kl` — KL divergence target (0.005 to 0.05)
- `adversary.actor_lr` — adversary learning rate
- `adversary.entropy_coef` — adversary entropy
- `training.adversary_strength` — adversarial weight (0.0 to 1.0)

**Output on Drive:**
```
MyDrive/BITCOIN4Traders_results/
├── results_YYYYMMDD_HHMMSS.tsv
├── adversarial_champion_YYYYMMDD_HHMMSS.yaml
├── champion_run_YYYYMMDD_HHMMSS.log
└── champion_model_YYYYMMDD_HHMMSS_checkpoint.pth
```

---

## Colab Shortcuts

| Shortcut | Action |
|----------|--------|
| `Shift + Enter` | Run cell and move to next |
| `Ctrl + Enter` | Run cell and stay |
| `Alt + Enter` | Run cell and insert new below |
| `Ctrl + M + H` | Show all shortcuts |
| `Ctrl + M + D` | Delete cell |
| `Ctrl + M + A` | Insert cell above |
| `Ctrl + M + B` | Insert cell below |
| `Ctrl + M + Y` | Change to code cell |
| `Ctrl + M + M` | Change to markdown cell |
| `Ctrl + S` | Save notebook |
| `Ctrl + F9` | Run all cells |
| `Ctrl + M + Z` | Undo cell deletion |

---

## Troubleshooting

### "No GPU detected"
Go to Runtime → Change runtime type → Hardware accelerator → GPU (T4)

### "Out of memory" (CUDA OOM)
- Reduce `BATCH_SIZE` in the configuration cell
- Reduce `N_ENVS` (number of parallel environments)
- Restart runtime: Runtime → Restart runtime

### "Colab disconnected" (timeout)
- Colab disconnects after ~90 minutes of inactivity
- Auto-save protects against data loss (saves every 10 min)
- Click on the notebook tab at least once per hour

### "Checkpoint not loading" (shape mismatch)
- `HIDDEN_DIM` changed between training runs
- Solution: Use the new `HIDDEN_DIM`, training starts fresh (correct behavior)

### "Download failed" (rate limit)
- Binance rate limit: add `time.sleep(0.1)` between requests
- Yahoo Finance: limited to ~730 hours of 1h data

### "Drive mount failed"
- Try: Runtime → Restart runtime
- Then re-run the Drive mount cell
- Make sure you're logged into the correct Google account

### "pip install fails"
- Restart runtime before re-installing
- Check internet connection
- Try: `!pip install --upgrade pip`

---

## FAQ

### Q: Do I need a GPU?
**A:** Only for Notebooks 3, 4, and 5. Notebooks 1 and 2 run on CPU.

### Q: How much does it cost?
**A:** Google Colab Free works for learning. For regular use, Colab Pro ($10/month) is recommended. See [Costs](#costs) section.

### Q: Can I run this on my local machine?
**A:** Yes, but you need a CUDA-capable GPU (GTX 1060+ recommended). The notebooks are optimized for Colab's T4 GPU.

### Q: How long does training take?
**A:** ~60 min for 500 iterations on T4 GPU. Tuning adds ~120 min.

### Q: What happens if Colab disconnects?
**A:** Auto-save runs every 10 minutes. Your latest checkpoint is saved to Drive. Just re-run from the checkpoint loading cell.

### Q: Can I use a different cryptocurrency?
**A:** Yes, change `SYMBOL` in the configuration cell. Supported: BTC/USDT, ETH/USDT, SOL/USDT, etc.

### Q: Is this financial advice?
**A:** No. This is an educational project. Do not use for real trading without thorough testing.

### Q: What's the difference between the split pipeline and all-in-one?
**A:** Split = 3 separate notebooks (cleaner memory). All-in-one = everything in one session (faster but more RAM).

### Q: Do I need API keys?
**A:** Only for `B4T_Tune.ipynb` with Option B (Gemini) or Option C (Claude). Option A (Grid Search) needs no API keys.

### Q: How do I check if my model is training correctly?
**A:** Look for these patterns:
- Iterations 1-50: Return fluctuates wildly (agent explores)
- Iterations 50-200: Return stabilizes, slow improvement
- Iterations 200+: Convergence, return should become positive

---

## Changelog

### v2.9.1 (2026-03-09)
- Fixed reward always 0.0: `STEPS_PER_ENV` was smaller than `max_steps`
- Fixed HMM crash: Inf values in Volume-Features not caught
- Fixed GRU learning: Hidden State was shifted by 1 step
- Fixed curriculum training: Action-Masking was never actually set
- Fixed VecEnv losing episode return: Info after Auto-Reset was overwritten
- Fixed Kelly calculated too often: Now only on trade events
- Removed emergency CPU-only fix, restored GPU path

### v2.9.0 (2026-03-01)
- Added GPU Feature Engineering (conv1d, JIT, float16)
- Added Mixed Precision (AMP) training (~1.7x speedup)
- Added VecEnv with 16 parallel environments
- Added Auto-Save every 10 minutes (Colab timeout protection)
- Added Curriculum Training (3-phase anti-bias)

### v2.8.0 (2026-02-15)
- Added Darwin Evolution Engine
- Added Multiverse Validation (155+ scenarios)
- Added Risk Engine (1%-Regel, Kelly, ATR-Stop-Loss)
- Added Telegram notifications

### v2.7.0 (2026-02-01)
- Initial Colab notebook release
- PPO + GRU agent
- Adversarial training (Trader vs Adversary)

---

## Links

| Resource | URL |
|----------|-----|
| **GitHub** | [github.com/juancarlosrial76-code/BITCOIN4Traders](https://github.com/juancarlosrial76-code/BITCOIN4Traders) |
| **Google Colab** | [colab.research.google.com](https://colab.research.google.com) |
| **PyTorch Docs** | [pytorch.org/docs](https://pytorch.org/docs/) |
| **Gymnasium Docs** | [gymnasium.farama.org](https://gymnasium.farama.org/) |
| **Stable Baselines3** | [sb3-contrib.readthedocs.io](https://sb3-contrib.readthedocs.io/) |
| **CCXT Docs** | [docs.ccxt.com](https://docs.ccxt.com/) |
| **Yahoo Finance** | [finance.yahoo.com](https://finance.yahoo.com/) |
| **PPO Paper** | [arxiv.org/abs/1707.06347](https://arxiv.org/abs/1707.06347) |

---

## Architecture Summary

```
┌──────────────────────────────────────────────────────────────┐
│                      BITCOIN4Traders                         │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    │
│  │  Data Layer  │───▶│ Evolution    │───▶│ PPO Training │    │
│  │  (Notebook 1)│    │ (Notebook 2) │    │ (Notebook 3) │    │
│  │              │    │              │    │              │    │
│  │ • CCXT/Yahoo │    │ • Darwin     │    │ • GRU Network│    │
│  │ • Features   │    │ • 155+ Scen. │    │ • VecEnv 16x │    │
│  │ • Scaler     │    │ • Champion   │    │ • AMP/Mixed  │    │
│  └──────────────┘    └──────────────┘    └──────────────┘    │
│         │                   │                   │            │
│         └───────────────────┼───────────────────┘            │
│                             ▼                                │
│                    ┌─────────────────┐                       │
│                    │  Google Drive   │                       │
│                    │  (Persistent)   │                       │
│                    └─────────────────┘                       │
│                             │                                │
│         ┌───────────────────┼───────────────────┐            │
│         ▼                   ▼                   ▼            │
│  ┌─────────────┐    ┌──────────────┐    ┌──────────────┐     │
│  │ All-in-One  │    │ Hyperparam   │    │ Risk Engine  │     │
│  │ (Notebook 4)│    │ Tune (NB 5)  │    │ + Live Trade │     │
│  └─────────────┘    └──────────────┘    └──────────────┘     │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

---

## Version

- **Current:** v2.9.1 (2026-03-09)
- **Python:** 3.10+
- **PyTorch:** 2.x
- **Colab Runtime:** T4 GPU (15 GB VRAM)
