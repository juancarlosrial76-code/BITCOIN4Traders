"""
Adversarial Training System
===========================
Self-play mechanism where Trader and Adversary improve each other.

This module implements an adversarial training framework for developing
robust trading agents. Instead of training in isolation, the Trader agent
must contend with an intelligent Adversary that learns to create challenging
market conditions.

Architecture:
-------------
- Trader Agent: Maximizes profit in the trading environment
- Adversary Agent: Learns to create difficult market scenarios
- Self-Play: Alternating training pushes both agents to improve

The Adversary doesn't just add random noise - it learns to create realistic
but difficult scenarios that expose the Trader's weaknesses. This leads to
more robust strategies that perform well under adverse conditions.

Adversary Strategies:
---------------------
The adversary can apply the following modifications to the observation space:
- Action 0 (Volatility): Adds Gaussian noise to simulate volatile markets
- Action 1 (Trend Bias): Injects systematic bias into price features
- Action 2 (Signal Inversion): Flips signs of random features
- Action 3 (No-op): Observes without interference

Adversary Reward:
-----------------
The adversary receives rewards based on:
- Zero-sum component: 50% of trader's loss
- Difficulty bonus: Reward for increasing market difficulty
- Success bonus: Extra reward when trader loses money

This creates a competitive co-evolution where both agents improve.

Reference:
---------
Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017).
Proximal Policy Optimization Algorithms. arXiv:1707.06347.

Usage Example:
-------------
    from src.training.adversarial_trainer import AdversarialTrainer, AdversarialConfig
    from src.agents.ppo_agent import PPOConfig

    # Configure agents
    trader_config = PPOConfig(state_dim=20, n_actions=3)
    adversary_config = PPOConfig(state_dim=20, n_actions=4)

    config = AdversarialConfig(
        n_iterations=500,
        steps_per_iteration=2048,
        trader_config=trader_config,
        adversary_config=adversary_config,
        adversary_start_iteration=100,  # Warm-up period
        adversary_strength=0.1,
    )

    # Initialize trainer
    trainer = AdversarialTrainer(env, config)

    # Train
    trainer.train()

    # Evaluate
    metrics = trainer.evaluate(n_episodes=100)
    print(f"Mean Return: {metrics['mean_return']*100:.2f}%")

Memory Management:
------------------
The trainer includes configurable memory management:
- History trimming: Limits stored metrics to prevent unbounded growth
- GPU memory: Clears CUDA cache periodically
- Adversary buffers: Optional clearing after each training iteration

Configure via config/memory_management.yaml
"""

import gc
import os
import numpy as np
import torch
from typing import Dict, Tuple, Optional
from dataclasses import dataclass, field
from pathlib import Path
from loguru import logger
import json
from datetime import datetime

try:
    import yaml

    _YAML_OK = True
except ImportError:
    _YAML_OK = False

from src.agents.ppo_agent import PPOAgent, PPOConfig

# Optional unified logging (MLflow + TensorBoard + ExperimentTracker)
try:
    from training.run_logger import RunLogger

    _HAS_RUN_LOGGER = True
except ImportError:
    _HAS_RUN_LOGGER = False


def _load_mem_cfg() -> dict:
    """Load memory_management.yaml (graceful fallback)."""
    try:
        cfg_path = Path("config/memory_management.yaml")
        if cfg_path.exists() and _YAML_OK:
            with open(cfg_path) as f:
                return yaml.safe_load(f) or {}
    except Exception as e:
        logger.warning(f"Adversarial training step failed loading memory_management.yaml: {e}")
        # Continue with empty config — all memory settings will use defaults
    return {}


@dataclass
class AdversarialConfig:
    """
    Configuration for Adversarial Training.

    This dataclass contains all hyperparameters and settings for the
    adversarial training system, including agent configurations, training
    schedules, and checkpointing options.

    Attributes:
        n_iterations (int): Total number of training iterations. Default: 500.
        steps_per_iteration (int): Environment steps per iteration. Default: 2048.

    Agent Configuration:
        trader_config (PPOConfig): Configuration for the trader agent.
        adversary_config (PPOConfig): Configuration for the adversary agent.

    Adversarial Settings:
        adversary_start_iteration (int): When to start adversary training (warm-up).
            Default: 100. The trader trains alone for the first 100 iterations.
        adversary_strength (float): Scaling factor for adversary modifications [0, 1].
            Higher values = more severe market manipulations. Default: 0.1.

    Checkpointing:
        save_frequency (int): Save checkpoint every N iterations. Default: 50.
        checkpoint_dir (str): Directory to save model checkpoints. Default: "data/models/adversarial".

    Logging:
        log_frequency (int): Log detailed metrics every N iterations. Default: 10.
        tensorboard (bool): Whether to use TensorBoard for logging. Default: True.

    Example:
        >>> from src.agents.ppo_agent import PPOConfig
        >>>
        >>> trader_config = PPOConfig(state_dim=20, n_actions=3, gamma=0.99)
        >>> adversary_config = PPOConfig(state_dim=20, n_actions=4)
        >>>
        >>> config = AdversarialConfig(
        ...     n_iterations=500,
        ...     steps_per_iteration=2048,
        ...     trader_config=trader_config,
        ...     adversary_config=adversary_config,
        ...     adversary_start_iteration=100,
        ...     adversary_strength=0.1,
        ...     save_frequency=50,
        ...     log_frequency=10,
        ... )
    """

    # Training
    n_iterations: int = 500
    steps_per_iteration: int = 2048

    # Agent configs
    trader_config: PPOConfig = field(default_factory=PPOConfig)
    adversary_config: PPOConfig = field(default_factory=PPOConfig)

    # Adversarial
    adversary_start_iteration: int = 100  # Warm-up period
    adversary_strength: float = 0.1  # How much adversary affects environment

    # Checkpointing
    save_frequency: int = 50
    checkpoint_dir: str = "data/models/adversarial"

    # Logging
    log_frequency: int = 10
    tensorboard: bool = True


class AdversarialTrainer:
    """
    Adversarial Training System for Robust Trading Agents.

    This trainer implements a self-play mechanism where a Trader agent and an
    Adversary agent co-evolve to improve each other. The Adversary learns to
    create challenging market conditions that expose the Trader's weaknesses,
    leading to more robust trading strategies.

    Training Workflow:
    -----------------
    1. Warm-up Phase (iterations < adversary_start_iteration):
       - Trader collects experience in normal market conditions
       - Trader learns from its own experience
       - Adversary is inactive

    2. Adversarial Phase (iterations >= adversary_start_iteration):
       - Trader selects actions and receives (potentially modified) observations
       - Adversary observes the same state and selects a modification action
       - Modified observations are fed to the Trader
       - Both agents receive rewards based on their objectives
       - Both agents update their policies

    3. Evaluation:
       - Disable adversary for deterministic evaluation
       - Run multiple episodes and collect metrics

    Key Methods:
    -----------
    - train(): Main training loop
    - collect_trajectories(): Gather experience from environment
    - train_trader(): Update Trader policy
    - train_adversary(): Update Adversary policy
    - evaluate(): Evaluate trained Trader
    - save_checkpoint()/load_checkpoint(): Model persistence

    Attributes:
        env: The trading environment (Gym-style).
        config (AdversarialConfig): Training configuration.
        device (str): Computation device.
        trader (PPOAgent): The main trading agent.
        adversary (PPOAgent): The adversary agent.
        iteration (int): Current training iteration.
        total_steps (int): Total environment steps taken.
        history (dict): Training history with metrics.

    Args:
        env: Trading environment with Gym-style interface (reset(), step()).
        config (AdversarialConfig): Training configuration.
        device (str): Device for computation. Default: "cpu".

    Environment Interface:
    ----------------------
    The environment must implement:
        - reset() -> observation, info
        - step(action) -> observation, reward, terminated, truncated, info

    The info dict should contain:
        - "return": Total episode return
        - "risk_metrics": Dict with "sharpe_ratio" and "max_drawdown"

    Example:
        >>> config = AdversarialConfig(n_iterations=500)
        >>> trainer = AdversarialTrainer(env, config)
        >>>
        >>> # Train for specified iterations
        >>> trainer.train()
        >>>
        >>> # Evaluate the trained trader (adversary disabled)
        >>> metrics = trainer.evaluate(n_episodes=100)
        >>> print(f"Sharpe: {metrics['mean_sharpe']:.2f}")
        >>>
        >>> # Save/Load
        >>> trainer.save_checkpoint("data/models/adversarial/checkpoint.pth")
        >>> trainer.load_checkpoint("data/models/adversarial/checkpoint.pth")
    """

    def __init__(
        self,
        env,  # Trading environment
        config: AdversarialConfig,
        device: str = "cpu",
        run_logger=None,  # Optional RunLogger instance
    ):
        """
        Initialize adversarial trainer.

        Parameters:
        -----------
        env : gym.Env
            Trading environment (ConfigIntegratedTradingEnv)
        config : AdversarialConfig
            Training configuration
        device : str
            'cpu' or 'cuda'
        run_logger : RunLogger or None
            Optional unified logger (MLflow + TensorBoard + ExperimentTracker).
            If None, only loguru console/file logging is used.
        """
        self.env = env
        self.config = config
        self.device = device
        self._run_logger = run_logger  # may be None

        # Create agents
        self.trader = PPOAgent(config.trader_config, device)
        self.adversary = PPOAgent(config.adversary_config, device)

        # Training state
        self.iteration = 0
        self.total_steps = 0

        # Load memory configuration
        _mem = _load_mem_cfg()
        self._max_history = _mem.get("history", {}).get("max_entries", 200)
        self._clear_adv_buf = _mem.get("adversary_buffer", {}).get(
            "clear_after_train", True
        )
        self._cuda_every = _mem.get("cuda", {}).get(
            "empty_cache_every_n_iterations", 10
        )
        self._ipython_every = _mem.get("ipython", {}).get(
            "reset_every_n_iterations", 50
        )
        self._reset_out = _mem.get("ipython", {}).get("reset_output_cache", True)

        # Metrics history (limited to max_history entries - no unlimited growth)
        self.history = {
            "trader_rewards": [],
            "trader_returns": [],
            "trader_sharpe": [],
            "adversary_rewards": [],
            "adversary_success": [],
            "episodes": [],
        }

        # Adversary state tracking (deleted after each collect_trajectories)
        self.adversary_states = []
        self.adversary_actions = []
        self.adversary_log_probs = []
        self.adversary_values = []
        self.adversary_rewards = []
        self.adversary_dones = []

        # Setup directories
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        logger.info("AdversarialTrainer initialized")
        logger.info(f"  Iterations: {config.n_iterations}")
        logger.info(f"  Steps/iteration: {config.steps_per_iteration}")
        logger.info(
            f"  Adversary starts at: iteration {config.adversary_start_iteration}"
        )

    def collect_trajectories(self, n_steps: int, use_adversary: bool = False) -> Dict:
        """
        Collect trajectories from environment with adversarial modifications.

        Parameters:
        -----------
        n_steps : int
            Number of steps to collect
        use_adversary : bool
            Whether to use adversary to modify environment

        Returns:
        --------
        metrics : dict
            Episode metrics including adversary performance
        """
        episode_rewards = []
        episode_returns = []
        episode_lengths = []
        episode_trade_winrates = []  # per-trade win rate from WinRateAwareReward

        # Track adversary performance
        adversary_episode_rewards = []
        adversary_challenges = []

        # Reset hidden states
        trader_hidden = None
        adversary_hidden = None

        # Pre-allocate trader buffer with exact capacity (P0-B)
        self.trader.reset_buffers(capacity=n_steps)

        obs, info = self.env.reset()
        episode_reward = 0
        episode_length = 0
        adversary_episode_reward = 0  # Per-episode adversary accumulator (avoids slice bug)

        # Clear adversary buffers for this collection
        self.adversary_states = []
        self.adversary_actions = []
        self.adversary_log_probs = []
        self.adversary_values = []
        self.adversary_rewards = []
        self.adversary_dones = []
        self.adversary_hiddens = []  # New buffer

        done = False

        for step in range(n_steps):
            # Trader selects action
            # Store hidden state used for this step
            current_trader_hidden = trader_hidden
            action, log_prob, value, trader_hidden = self.trader.select_action(
                obs, trader_hidden
            )

            # Adversary modifies environment (if active)
            adversary_reward = 0.0
            challenge_info: dict = {}  # default: no challenge applied
            if (
                use_adversary
                and self.iteration >= self.config.adversary_start_iteration
            ):
                # Adversary observes current state and selects modification
                current_adversary_hidden = adversary_hidden
                adv_action, adv_log_prob, adv_value, adversary_hidden = (
                    self.adversary.select_action(obs, adversary_hidden)
                )

                # Apply adversary modification to environment
                modified_obs, challenge_info = self._apply_adversary_modification(
                    obs, adv_action, info
                )

                # Store adversary transition (will compute reward after step)
                self.adversary_states.append(obs.copy())
                self.adversary_actions.append(adv_action)
                self.adversary_log_probs.append(adv_log_prob)
                self.adversary_values.append(adv_value)
                self.adversary_hiddens.append(current_adversary_hidden)
                adversary_challenges.append(challenge_info)

                # Use modified observation for trader
                obs_input = modified_obs
            else:
                obs_input = obs

            # Step environment
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated

            # Compute adversary reward: adversary wins when trader loses
            if (
                use_adversary
                and self.iteration >= self.config.adversary_start_iteration
            ):
                # Base reward: adversary gains 50% of trader's loss (zero-sum component)
                adversary_reward = -reward * 0.5

                # Additional reward for increasing volatility/difficulty
                if "volatility_increase" in challenge_info:
                    adversary_reward += (
                        challenge_info["volatility_increase"] * 0.1
                    )  # Reward adversary for injecting harder noise

                # Extra bonus if trader loses money (encourages adversary to find weaknesses)
                if reward < 0:
                    adversary_reward += (
                        abs(reward) * 0.3
                    )  # 30% of trader loss as adversary bonus

                self.adversary_rewards.append(adversary_reward)
                self.adversary_dones.append(done)
                adversary_episode_reward += adversary_reward

            # Store transition for trader
            self.trader.store_transition(
                obs_input,
                action,
                reward,
                log_prob,
                value,
                done,
                hidden=current_trader_hidden,
            )

            episode_reward += reward
            episode_length += 1
            self.total_steps += 1

            # Handle episode end
            if done:
                episode_rewards.append(episode_reward)
                episode_returns.append(info.get("return", 0.0))
                episode_lengths.append(episode_length)
                wr = info.get("win_rate", -1.0)
                if wr >= 0:
                    episode_trade_winrates.append(wr)

                # Track adversary episode reward using per-episode accumulator
                # (avoids wrong-slice bug when multiple episodes occur in one iteration)
                if (
                    use_adversary
                    and self.iteration >= self.config.adversary_start_iteration
                ):
                    adversary_episode_rewards.append(adversary_episode_reward)

                obs, info = self.env.reset()
                episode_reward = 0
                episode_length = 0
                adversary_episode_reward = 0  # Reset per-episode accumulator

                # Reset hidden states on episode completion.
                # TODO (BUG-HIGH): GRU starts cold (hidden=None) even though the
                # environment positions current_step after a `lookback_window` warmup.
                # The agent cannot build temporal context for the lookback features.
                # Fix: expose env._features_np[step-lookback:step] and run the GRU
                # over those observations (no env.step calls) to warm up hidden state.
                # This requires adding `env.get_warmup_obs()` to config_integrated_env.
                trader_hidden = None
                adversary_hidden = None
            else:
                obs = next_obs

        if not done:
            _, _, next_value, _ = self.trader.select_action(obs, trader_hidden)
            if (
                use_adversary
                and self.iteration >= self.config.adversary_start_iteration
            ):
                _, _, adv_next_value, _ = self.adversary.select_action(
                    obs, adversary_hidden
                )
            else:
                adv_next_value = 0.0
        else:
            next_value = 0.0
            adv_next_value = 0.0

        # Length-weighted return: longer episodes contribute proportionally more.
        # Avoids survivorship bias from short profitable episodes.
        _total_steps = sum(episode_lengths) or 1
        weighted_return = (
            sum(r * l for r, l in zip(episode_returns, episode_lengths)) / _total_steps
            if episode_returns else 0.0
        )

        return {
            "episode_rewards": episode_rewards,
            "episode_returns": episode_returns,
            "episode_lengths": episode_lengths,
            "mean_reward": np.mean(episode_rewards) if episode_rewards else 0.0,
            "mean_return": np.mean(episode_returns) if episode_returns else 0.0,
            "weighted_return": weighted_return,
            "mean_length": np.mean(episode_lengths) if episode_lengths else 0.0,
            "trade_win_rate": np.mean(episode_trade_winrates) if episode_trade_winrates else -1.0,
            "next_value": next_value,
            "adversary_next_value": adv_next_value,
            "adversary_episode_rewards": adversary_episode_rewards,
            "mean_adversary_reward": np.mean(adversary_episode_rewards)
            if adversary_episode_rewards
            else 0.0,
        }

    # ------------------------------------------------------------------
    # Vectorised trajectory collection (GPU-optimised)
    # ------------------------------------------------------------------

    def collect_trajectories_vec(
        self,
        vec_env,  # VecTradingEnv
        steps_per_env: int,
        use_adversary: bool = False,
    ) -> Dict:
        """
        Collect trajectories with GPU-CPU Double-Buffering Pipeline.

        Problem (old): GPU forward pass and CPU env.step() run SEQUENTIALLY.
        GPU waits while CPU steps → T4 GPU utilization ~10-20%.

        Solution (new): CUDA Stream Overlap — CPU and GPU run PARALLEL:

            Step t:   GPU computes forward(obs_t)
                      CPU steps envs with actions_{t-1}  ← simultaneously!
            Step t+1: GPU computes forward(obs_{t+1})
                      CPU steps envs with actions_t       ← simultaneously!

        Implementation via torch.cuda.Stream:
        - compute_stream: GPU forward pass (select_action_batch)
        - transfer_stream: Non-blocking obs copy to GPU
        - CPU env.step() runs in ThreadPoolExecutor parallel to GPU stream

        Expected GPU utilization: 10-20% → 60-80% on T4.

        Parameters
        ----------
        vec_env : VecTradingEnv
        steps_per_env : int

        Returns
        -------
        metrics : dict
        """
        n_envs = vec_env.n_envs
        use_cuda = torch.cuda.is_available()

        # Per-env episode tracking
        ep_rewards: list = [[] for _ in range(n_envs)]
        ep_returns: list = [[] for _ in range(n_envs)]
        ep_lengths: list = [[] for _ in range(n_envs)]
        cur_reward = np.zeros(n_envs, dtype=np.float32)
        cur_length = np.zeros(n_envs, dtype=np.int32)

        self.trader.reset_buffers(capacity=steps_per_env * n_envs)
        trader_hidden = self.trader.get_initial_hidden_state(batch_size=n_envs)

        # Adversary init for VecEnv mode
        if use_adversary:
            self.adversary.reset_buffers(capacity=steps_per_env * n_envs)
            adversary_hidden = self.adversary.get_initial_hidden_state(batch_size=n_envs)
            self.adversary_states = []
            self.adversary_actions = []
            self.adversary_log_probs = []
            self.adversary_values = []
            self.adversary_rewards = []
            self.adversary_dones = []
            self.adversary_hiddens = []
        else:
            adversary_hidden = None

        obs = vec_env.reset()  # (N, state_dim)

        # ── CUDA Streams for Overlap ──────────────────────────────────────────
        # compute_stream: GPU forward pass
        # transfer_stream: non-blocking H2D Transfer (obs CPU→GPU)
        if use_cuda:
            compute_stream = torch.cuda.Stream()
            transfer_stream = torch.cuda.Stream()
        else:
            compute_stream = None
            transfer_stream = None

        # Pinned Memory Buffer for fast H2D Transfer (non-blocking)
        # pinned memory enables DMA transfer without CPU involvement
        if use_cuda:
            obs_pinned = torch.empty(
                (n_envs, obs.shape[1]), dtype=torch.float32, pin_memory=True
            )
        else:
            obs_pinned = None

        # ── Double-Buffering State ─────────────────────────────────────────
        # We hold onto the results from the previous GPU step
        # so we can save them while the GPU is already computing the next one
        prev_actions = None
        prev_log_probs = None
        prev_values = None
        prev_hidden = None
        prev_obs = None
        # Adversary double-buffering (same pattern as trader)
        prev_adv_actions = None
        prev_adv_log_probs = None
        prev_adv_values = None
        prev_adv_hidden = None

        # Futures for async env.step() (CPU runs parallel to GPU)
        from concurrent.futures import Future

        step_future: Optional[Future] = None
        step_result = None  # (next_obs, rewards, dones, infos) vom letzten step

        def _submit_step(actions_np):
            """Start env.step() async - runs parallel to GPU forward."""
            return vec_env._executor.submit(lambda: vec_env.step(actions_np))

        # ── Rollout Loop with Pipeline ────────────────────────────────────────
        for step_idx in range(steps_per_env):
            # ── Remember hidden state BEFORE the forward pass ───────────────────
            # IMPORTANT: prev_hidden must be h_{t-1} (input to forward computation),
            # not h_t (output). That's why we set it HERE, before trader_hidden is
            # updated by forward() to h_t. Without this fix, the wrong (shifted by 1)
            # hidden state would be used for BPTT during training.
            prev_hidden = trader_hidden

            # ── Adversary: select modification for all N envs ────────────
            # Runs synchronously on CPU (cheap compared to trader GPU forward).
            # Must happen BEFORE the trader sees obs so the modification reaches
            # the trader's policy network.
            if use_adversary:
                prev_adv_hidden = adversary_hidden
                adv_actions_t, adv_lp_t, adv_val_t, adversary_hidden = (
                    self.adversary.select_action_batch(obs, adversary_hidden)
                )
                obs_for_trader = self._apply_adversary_modification_batch(obs, adv_actions_t)
            else:
                obs_for_trader = obs

            # ── GPU forward pass (non-blocking) ──────────────────────────
            if use_cuda and compute_stream is not None and obs_pinned is not None:
                with torch.cuda.stream(compute_stream):
                    # Non-blocking H2D Transfer
                    obs_pinned.copy_(torch.from_numpy(obs_for_trader), non_blocking=True)
                    obs_gpu = obs_pinned.to(self.trader.device, non_blocking=True)
                    # Forward pass on compute_stream
                    actions, log_probs, values, trader_hidden = (
                        self.trader.select_action_batch(obs_gpu, trader_hidden)
                    )
            else:
                # CPU-Fallback: synchron
                actions, log_probs, values, trader_hidden = (
                    self.trader.select_action_batch(obs_for_trader, trader_hidden)
                )

            # ── CPU env.step() PARALLEL to next GPU forward ──────────────
            # Wait for previous step_future (if present)
            if step_future is not None:
                next_obs, rewards, dones, infos = step_future.result()
            else:
                next_obs, rewards, dones = None, None, None
                infos = [{} for _ in range(n_envs)]

            # ── GPU Sync: ensure forward pass is finished ───────
            if use_cuda and compute_stream is not None:
                compute_stream.synchronize()

            # actions is now ready → start env.step() async
            # select_action_batch() already returns numpy int64
            step_future = _submit_step(actions)

            # ── Save Transitions from the PREVIOUS step ──────────────
            # (prev_* has the forward pass from t-1, step_result has rewards from t-1)
            # prev_hidden = h_{t-1} (gesetzt VOR dem forward pass oben — korrekt!)
            if prev_actions is not None and next_obs is not None:
                # ADV-5: Build hidden list for all N envs, then batch-write.
                # store_transitions_batch() uses a single numpy slice assignment
                # for states/actions/rewards/log_probs/values/dones instead of
                # N individual scalar writes → ~10-15% faster per collect step.
                if prev_hidden is not None:
                    if isinstance(prev_hidden, tuple):
                        _hiddens_list = [
                            tuple(h[:, i : i + 1, :].detach() for h in prev_hidden)
                            for i in range(n_envs)
                        ]
                    else:
                        _hiddens_list = [
                            prev_hidden[:, i : i + 1, :].detach() for i in range(n_envs)
                        ]
                else:
                    _hiddens_list = [None] * n_envs

                self.trader.store_transitions_batch(
                    states=prev_obs,  # (N, state_dim)
                    actions=prev_actions,  # (N,)
                    rewards=rewards,  # (N,)
                    log_probs=prev_log_probs,  # (N,)
                    values=prev_values,  # (N,)
                    dones=dones,  # (N,)
                    hiddens_batch=_hiddens_list,
                )

                cur_reward += rewards
                cur_length += 1
                self.total_steps += n_envs

                # ── Save adversary transitions for this step ──────
                if use_adversary and prev_adv_actions is not None:
                    # Adversary reward: wins when trader loses
                    adv_rewards_batch = -rewards * 0.5
                    if prev_adv_hidden is not None:
                        if isinstance(prev_adv_hidden, tuple):
                            _adv_hiddens = [
                                tuple(h[:, i:i+1, :].detach() for h in prev_adv_hidden)
                                for i in range(n_envs)
                            ]
                        else:
                            _adv_hiddens = [
                                prev_adv_hidden[:, i:i+1, :].detach()
                                for i in range(n_envs)
                            ]
                    else:
                        _adv_hiddens = [None] * n_envs

                    for i in range(n_envs):
                        self.adversary_states.append(prev_obs[i])
                        self.adversary_actions.append(int(prev_adv_actions[i]))
                        self.adversary_log_probs.append(float(prev_adv_log_probs[i]))
                        self.adversary_values.append(float(prev_adv_values[i]))
                        self.adversary_rewards.append(float(adv_rewards_batch[i]))
                        self.adversary_dones.append(bool(dones[i]))
                        self.adversary_hiddens.append(_adv_hiddens[i])

                for i in range(n_envs):
                    if dones[i]:
                        ep_rewards[i].append(float(cur_reward[i]))
                        ep_returns[i].append(float(infos[i].get("return", 0.0)))
                        ep_lengths[i].append(int(cur_length[i]))
                        cur_reward[i] = 0.0
                        cur_length[i] = 0
                        if trader_hidden is not None:
                            if isinstance(trader_hidden, tuple):
                                trader_hidden[0][:, i, :].zero_()
                                trader_hidden[1][:, i, :].zero_()
                            else:
                                trader_hidden[:, i, :].zero_()
                        if use_adversary and adversary_hidden is not None:
                            if isinstance(adversary_hidden, tuple):
                                adversary_hidden[0][:, i, :].zero_()
                                adversary_hidden[1][:, i, :].zero_()
                            else:
                                adversary_hidden[:, i, :].zero_()

            # ── Shift state one step forward ──────────────────────
            prev_obs = obs.copy()
            prev_actions = actions
            prev_log_probs = log_probs
            prev_values = values
            # NOT: prev_hidden = trader_hidden (moved to the beginning of the loop)
            if use_adversary:
                prev_adv_actions = adv_actions_t
                prev_adv_log_probs = adv_lp_t
                prev_adv_values = adv_val_t
                # prev_adv_hidden already set at top of loop

            if next_obs is not None:
                obs = next_obs

        # ── Resolve last step_future and save last step ──────────────────
        if step_future is not None and prev_actions is not None:
            next_obs, rewards, dones, infos = step_future.result()
            # ADV-5: same batch-write pattern as the main loop
            if prev_hidden is not None:
                if isinstance(prev_hidden, tuple):
                    _hiddens_list = [
                        tuple(h[:, i : i + 1, :].detach() for h in prev_hidden)
                        for i in range(n_envs)
                    ]
                else:
                    _hiddens_list = [
                        prev_hidden[:, i : i + 1, :].detach() for i in range(n_envs)
                    ]
            else:
                _hiddens_list = [None] * n_envs

            self.trader.store_transitions_batch(
                states=prev_obs,
                actions=prev_actions,
                rewards=rewards,
                log_probs=prev_log_probs,
                values=prev_values,
                dones=dones,
                hiddens_batch=_hiddens_list,
            )

            cur_reward += rewards
            cur_length += 1
            self.total_steps += n_envs

            # Last-step adversary transitions
            if use_adversary and prev_adv_actions is not None:
                adv_rewards_batch = -rewards * 0.5
                if prev_adv_hidden is not None:
                    if isinstance(prev_adv_hidden, tuple):
                        _adv_hiddens = [
                            tuple(h[:, i:i+1, :].detach() for h in prev_adv_hidden)
                            for i in range(n_envs)
                        ]
                    else:
                        _adv_hiddens = [
                            prev_adv_hidden[:, i:i+1, :].detach()
                            for i in range(n_envs)
                        ]
                else:
                    _adv_hiddens = [None] * n_envs
                for i in range(n_envs):
                    self.adversary_states.append(prev_obs[i])
                    self.adversary_actions.append(int(prev_adv_actions[i]))
                    self.adversary_log_probs.append(float(prev_adv_log_probs[i]))
                    self.adversary_values.append(float(prev_adv_values[i]))
                    self.adversary_rewards.append(float(adv_rewards_batch[i]))
                    self.adversary_dones.append(bool(dones[i]))
                    self.adversary_hiddens.append(_adv_hiddens[i])

            for i in range(n_envs):
                if dones[i]:
                    ep_rewards[i].append(float(cur_reward[i]))
                    ep_returns[i].append(float(infos[i].get("return", 0.0)))
                    ep_lengths[i].append(int(cur_length[i]))
                    cur_reward[i] = 0.0
                    cur_length[i] = 0

            obs = next_obs

        # Bootstrap values
        _, _, bootstrap_values, _ = self.trader.select_action_batch(obs, trader_hidden)
        next_value = float(bootstrap_values.mean())

        if use_adversary:
            _, _, adv_bootstrap, _ = self.adversary.select_action_batch(obs, adversary_hidden)
            adversary_next_value = float(adv_bootstrap.mean())
            mean_adv_reward = float(np.mean(self.adversary_rewards)) if self.adversary_rewards else 0.0
        else:
            adversary_next_value = 0.0
            mean_adv_reward = 0.0

        all_ep_rewards = [r for env_r in ep_rewards for r in env_r]
        all_ep_returns = [r for env_r in ep_returns for r in env_r]
        all_ep_lengths = [l for env_l in ep_lengths for l in env_l]

        return {
            "episode_rewards": all_ep_rewards,
            "episode_returns": all_ep_returns,
            "episode_lengths": all_ep_lengths,
            "mean_reward": float(np.mean(all_ep_rewards)) if all_ep_rewards else 0.0,
            "mean_return": float(np.mean(all_ep_returns)) if all_ep_returns else 0.0,
            "mean_length": float(np.mean(all_ep_lengths)) if all_ep_lengths else 0.0,
            "next_value": next_value,
            "adversary_next_value": adversary_next_value,
            "adversary_episode_rewards": list(self.adversary_rewards) if use_adversary else [],
            "mean_adversary_reward": mean_adv_reward,
        }

    def _apply_adversary_modification(
        self, obs: np.ndarray, adv_action: int, env_info: Dict
    ) -> tuple[np.ndarray, Dict]:
        """
        Apply adversary's modification to the observation/environment.

        Adversary actions:
        0: Increase volatility (noisy observations)
        1: Add trend bias (misleading trend signals)
        2: Invert signals (confusing patterns)
        3: No modification

        Parameters:
        -----------
        obs : np.ndarray
            Original observation
        adv_action : int
            Adversary's chosen action
        env_info : dict
            Current environment info

        Returns:
        --------
        modified_obs : np.ndarray
            Modified observation
        challenge_info : dict
            Info about the challenge applied
        """
        modified_obs = obs.copy()
        challenge_info = {}

        strength = self.config.adversary_strength

        if adv_action == 0:
            # Action 0: Add Gaussian noise to all features (simulates volatile/noisy market)
            noise = (
                np.random.randn(len(obs)) * strength * 0.5
            )  # Scale noise by adversary strength
            modified_obs += noise
            challenge_info["type"] = "volatility_increase"
            challenge_info["volatility_increase"] = np.mean(
                np.abs(noise)
            )  # Track average noise magnitude

        elif adv_action == 1:
            # Action 1: Inject systematic bias into price-related features (simulates misleading trends)
            n_features = min(
                5, len(obs) // 2
            )  # Target first 5 features (likely price/return signals)
            bias = np.random.randn() * strength  # Random directional bias
            modified_obs[:n_features] += bias
            challenge_info["type"] = "trend_bias"
            challenge_info["bias_magnitude"] = abs(bias)

        elif adv_action == 2:
            # Action 2: Flip signs of random features (simulates confusing/inverted signals)
            n_invert = max(
                1, int(len(obs) * strength * 0.3)
            )  # Number of features to invert
            invert_indices = np.random.choice(
                len(obs), n_invert, replace=False
            )  # Random feature subset
            modified_obs[invert_indices] *= -1  # Flip sign to confuse the trader
            challenge_info["type"] = "signal_inversion"
            challenge_info["n_inverted"] = n_invert

        else:  # adv_action == 3
            # Action 3: No modification (adversary chooses to observe without interfering)
            challenge_info["type"] = "none"
            challenge_info["volatility_increase"] = 0.0

        # Clip modified observation to prevent extreme values that could destabilize training
        modified_obs = np.clip(modified_obs, -10.0, 10.0)

        return modified_obs, challenge_info

    def _apply_adversary_modification_batch(
        self, obs: np.ndarray, adv_actions: np.ndarray
    ) -> np.ndarray:
        """
        Vectorised adversary modification for VecEnv: applies per-env action to each row.

        Parameters
        ----------
        obs : np.ndarray  shape (N, D)
        adv_actions : np.ndarray  shape (N,)  int64

        Returns
        -------
        modified_obs : np.ndarray  shape (N, D)
        """
        modified = obs.copy()
        strength = self.config.adversary_strength
        N, D = modified.shape

        for i in range(N):
            a = int(adv_actions[i])
            if a == 0:
                modified[i] += np.random.randn(D) * strength * 0.5
            elif a == 1:
                n_feat = min(5, D // 2)
                modified[i, :n_feat] += np.random.randn() * strength
            elif a == 2:
                n_inv = max(1, int(D * strength * 0.3))
                idx = np.random.choice(D, n_inv, replace=False)
                modified[i, idx] *= -1
            # a == 3: no modification

        return np.clip(modified, -10.0, 10.0)

    def train_trader(self, next_value: float) -> Dict:
        """Train trader agent."""
        stats = self.trader.train(next_value)
        return stats

    def train_adversary(self, next_value: float = 0.0) -> Dict:
        """
        Train adversary agent using collected trajectories.

        Adversary learns to create scenarios that:
        1. Are realistic (maintain market dynamics)
        2. Challenge the trader (lower trader's performance)
        3. Expose trader's weaknesses

        Parameters:
        -----------
        next_value : float
            Value estimate for final state (for GAE)

        Returns:
        --------
        stats : dict
            Training statistics
        """
        # Check if we have adversary data to train on
        if len(self.adversary_states) == 0:
            return {
                "adversary_loss": 0.0,
                "adversary_success_rate": 0.0,
                "adversary_episodes": 0,
            }

        # Store transitions in adversary's buffer
        for i in range(len(self.adversary_states)):
            self.adversary.store_transition(
                state=self.adversary_states[i],
                action=self.adversary_actions[i],
                reward=self.adversary_rewards[i],
                log_prob=self.adversary_log_probs[i],
                value=self.adversary_values[i],
                done=self.adversary_dones[i],
                hidden=self.adversary_hiddens[i],
            )

        # Train adversary using PPO
        stats = self.adversary.train(next_value)

        # Calculate success rate: how often adversary made trader lose money
        n_challenges = len(self.adversary_rewards)
        n_successful = sum(1 for r in self.adversary_rewards if r > 0)
        success_rate = n_successful / n_challenges if n_challenges > 0 else 0.0

        # Store history (limited - no unlimited growth)
        self.history["adversary_rewards"].append(np.mean(self.adversary_rewards))
        self.history["adversary_success"].append(success_rate)
        self._trim_history()

        result = {
            "adversary_loss": stats.get("actor_loss", 0.0),
            "adversary_critic_loss": stats.get("critic_loss", 0.0),
            "adversary_entropy": stats.get("entropy", 0.0),
            "adversary_success_rate": success_rate,
            "adversary_episodes": len(self.adversary_rewards),
            "mean_adversary_reward": np.mean(self.adversary_rewards),
        }

        # Free RAM: Clear adversary buffer (configurable)
        if self._clear_adv_buf:
            self.adversary_states.clear()
            self.adversary_actions.clear()
            self.adversary_log_probs.clear()
            self.adversary_values.clear()
            self.adversary_rewards.clear()
            self.adversary_dones.clear()
            if hasattr(self, "adversary_hiddens"):
                self.adversary_hiddens.clear()
            gc.collect()

        return result

    def train(self, vec_env=None, steps_per_env: Optional[int] = None):
        """
        Main training loop.

        Alternates between:
        1. Trader improvement
        2. Adversary learning (after warm-up)
        3. Evaluation
        4. Checkpointing

        Parameters
        ----------
        vec_env : VecTradingEnv or None
            If provided, uses collect_trajectories_vec() for GPU-optimised
            trajectory collection (single batched forward-pass per step).
            Adversarial training is NOT supported with VecEnv — adversary stays
            inactive when vec_env is passed.
        steps_per_env : int or None
            Steps per sub-environment per iteration when using vec_env.
            Defaults to config.steps_per_iteration // vec_env.n_envs.
        """
        logger.info("Starting adversarial training...")
        _use_vec = vec_env is not None
        # _steps_per_env: always defined before the loop so Pyright doesn't complain
        _n_envs: int = vec_env.n_envs if (_use_vec and vec_env is not None) else 1
        _steps_per_env: int = (
            (steps_per_env or (self.config.steps_per_iteration // _n_envs))
            if _use_vec
            else self.config.steps_per_iteration
        )
        if _use_vec and vec_env is not None:
            logger.info(
                f"VecEnv mode: {vec_env.n_envs} envs x {_steps_per_env} steps "
                f"= {vec_env.n_envs * _steps_per_env} steps/iter | adversary ENABLED"
            )

        for iteration in range(self.config.n_iterations):
            self.iteration = iteration

            # Collect trajectories
            use_adversary = iteration >= self.config.adversary_start_iteration

            # debug level: goes to log file only, not to Colab stdout
            logger.debug(f"\n{'=' * 80}")
            logger.debug(f"Iteration {iteration + 1}/{self.config.n_iterations}")
            logger.debug(f"Adversary active: {use_adversary}")

            if _use_vec:
                traj_metrics = self.collect_trajectories_vec(vec_env, _steps_per_env, use_adversary=use_adversary)
            else:
                traj_metrics = self.collect_trajectories(
                    self.config.steps_per_iteration, use_adversary=use_adversary
                )

            # Train trader
            trader_stats = self.train_trader(traj_metrics["next_value"])

            # Train adversary (after warm-up)
            if use_adversary:
                adversary_stats = self.train_adversary(
                    traj_metrics.get("adversary_next_value", 0.0)
                )
            else:
                adversary_stats = {}

            # Log metrics
            if iteration % self.config.log_frequency == 0:
                self._log_iteration(
                    iteration, traj_metrics, trader_stats, adversary_stats
                )

            # Checkpoint: use length-weighted return for best-model tracking
            if iteration % self.config.save_frequency == 0:
                mean_ret = traj_metrics.get("weighted_return", traj_metrics.get("mean_return", 0))
                self._save_checkpoint(iteration, mean_ret)

            # Store history (limited to max_history)
            self.history["trader_rewards"].append(traj_metrics["mean_reward"])
            self.history["trader_returns"].append(traj_metrics.get("weighted_return", traj_metrics["mean_return"]))
            self.history["episodes"].append(len(traj_metrics["episode_rewards"]))
            self._trim_history()

            # ── Free RAM + GPU Memory ──────────────────────────────
            if iteration % self._cuda_every == 0:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

            # ── Clear IPython Output Cache (Colab) ───────────────────────
            if self._reset_out and iteration % self._ipython_every == 0:
                try:
                    from IPython import get_ipython

                    ip = get_ipython()
                    if ip is not None:
                        ip.run_line_magic("reset_selective", "-f _i")
                except Exception as e:
                    logger.warning(f"IPython output cache clear failed at iteration {iteration}: {e}")
                    # Non-critical — training continues regardless

        logger.success("Training complete!")
        self._save_final_checkpoint()

    def evaluate(self, n_episodes: int = 100) -> Dict:
        """
        Evaluate trained trader.

        Parameters:
        -----------
        n_episodes : int
            Number of evaluation episodes

        Returns:
        --------
        metrics : dict
            Evaluation metrics
        """
        logger.info(f"Evaluating trader on {n_episodes} episodes...")

        episode_returns = []
        episode_lengths = []
        episode_sharpes = []
        episode_max_dds = []

        for ep in range(n_episodes):
            obs, info = self.env.reset()
            done = False
            ep_steps = 0

            hidden = None
            while not done:
                # Deterministic action selection
                action, _, _, hidden = self.trader.select_action(
                    obs, hidden, deterministic=True
                )
                obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                ep_steps += 1

            # Collect metrics
            episode_returns.append(info.get("return", 0.0))
            episode_lengths.append(ep_steps)

            if "risk_metrics" in info:
                sharpe = info["risk_metrics"].get("sharpe_ratio", 0.0)
                max_dd = info["risk_metrics"].get("max_drawdown", 0.0)
            else:
                sharpe = 0.0
                max_dd = 0.0

            episode_sharpes.append(sharpe)
            episode_max_dds.append(max_dd)

        # Length-weighted mean return: longer episodes count proportionally more
        total_steps = sum(episode_lengths) or 1
        weighted_return = sum(
            r * l for r, l in zip(episode_returns, episode_lengths)
        ) / total_steps
        # Sharpe of episode returns (cross-episode consistency)
        ep_sharpe_ratio = (
            np.mean(episode_returns) / (np.std(episode_returns) + 1e-8)
            if len(episode_returns) > 1 else 0.0
        )
        # Calmar ratio (return / max drawdown)
        mean_max_dd = np.mean(episode_max_dds) if episode_max_dds else 0.0
        calmar = weighted_return / (abs(mean_max_dd) + 1e-8)

        metrics = {
            "mean_return": np.mean(episode_returns),            # unweighted (legacy)
            "weighted_return": weighted_return,                 # length-weighted (correct)
            "std_return": np.std(episode_returns),
            "episode_sharpe": ep_sharpe_ratio,                  # cross-episode Sharpe
            "mean_sharpe": np.mean(episode_sharpes),            # in-episode Sharpe
            "mean_max_dd": mean_max_dd,
            "calmar_ratio": calmar,
            "win_rate": np.mean([r > 0 for r in episode_returns]),
            "mean_episode_length": np.mean(episode_lengths),
        }

        logger.info("\nEvaluation Results:")
        logger.info(f"  Weighted Return: {metrics['weighted_return'] * 100:.2f}%  (mean: {metrics['mean_return'] * 100:.2f}%)")
        logger.info(f"  Std Return:  {metrics['std_return'] * 100:.2f}%")
        logger.info(f"  Episode Sharpe: {metrics['episode_sharpe']:.2f}")
        logger.info(f"  Mean In-Ep Sharpe: {metrics['mean_sharpe']:.2f}")
        logger.info(f"  Calmar Ratio: {metrics['calmar_ratio']:.2f}")
        logger.info(f"  Mean Max DD: {metrics['mean_max_dd'] * 100:.2f}%")
        logger.info(f"  Win Rate: {metrics['win_rate'] * 100:.1f}%")

        return metrics

    def _trim_history(self):
        """
        Limits all history lists to self._max_history entries.
        Prevents unlimited RAM growth over 500+ iterations.
        0 = no limit (for local machines).
        """
        if self._max_history <= 0:
            return
        for key in self.history:
            lst = self.history[key]
            if len(lst) > self._max_history:
                # Keep only the last max_history entries
                self.history[key] = lst[-self._max_history :]

    def _log_iteration(
        self,
        iteration: int,
        traj_metrics: Dict,
        trader_stats: Dict,
        adversary_stats: Dict,
    ):
        """Log iteration metrics (debug = file only, not Colab stdout)."""
        logger.debug(f"\nIteration {iteration + 1} Results:")
        logger.debug(f"  Episodes: {len(traj_metrics['episode_rewards'])}")
        logger.debug(f"  Mean Reward: {traj_metrics['mean_reward']:.4f}")
        logger.debug(f"  Mean Return: {traj_metrics['mean_return'] * 100:.2f}%")
        logger.debug(f"  Mean Length: {traj_metrics['mean_length']:.0f}")

        if trader_stats:
            logger.debug(f"\nTrader Training:")
            logger.debug(f"  Actor Loss: {trader_stats['actor_loss']:.4f}")
            logger.debug(f"  Critic Loss: {trader_stats['critic_loss']:.4f}")
            logger.debug(f"  Entropy: {trader_stats['entropy']:.4f}")
            logger.debug(f"  KL: {trader_stats['mean_kl']:.4f}")

        if adversary_stats:
            logger.debug(f"\nAdversary Training:")
            logger.debug(f"  Loss: {adversary_stats.get('adversary_loss', 0):.4f}")
            logger.debug(
                f"  Success: {adversary_stats.get('adversary_success_rate', 0) * 100:.1f}%"
            )
        # Per-trade win rate (from WinRateAwareReward rolling tracker)
        twr = traj_metrics.get("trade_win_rate", -1.0)
        if twr >= 0:
            logger.debug(f"  Per-Trade Win Rate: {twr * 100:.1f}%")

        # One-line WARNING summary visible on console every log_frequency iterations
        twr_str = f" | Trade WR {twr * 100:.1f}%" if twr >= 0 else ""
        logger.warning(
            f"Iter {iteration + 1} | "
            f"Return {traj_metrics['mean_return'] * 100:.1f}% | "
            f"Reward {traj_metrics['mean_reward']:.3f}"
            f"{twr_str}"
        )

        # ── RunLogger: unified MLflow + TensorBoard + ExperimentTracker ──
        if self._run_logger is not None:
            metrics: Dict = {
                "trader/mean_return": traj_metrics["mean_return"],
                "trader/mean_reward": traj_metrics["mean_reward"],
                "trader/mean_length": traj_metrics["mean_length"],
            }
            if trader_stats:
                metrics.update(
                    {
                        "trader/actor_loss": trader_stats.get("actor_loss", 0.0),
                        "trader/critic_loss": trader_stats.get("critic_loss", 0.0),
                        "trader/entropy": trader_stats.get("entropy", 0.0),
                        "trader/kl": trader_stats.get("mean_kl", 0.0),
                    }
                )
            if adversary_stats:
                metrics.update(
                    {
                        "adversary/loss": adversary_stats.get("adversary_loss", 0.0),
                        "adversary/success_rate": adversary_stats.get(
                            "adversary_success_rate", 0.0
                        ),
                        "adversary/mean_reward": adversary_stats.get(
                            "mean_adversary_reward", 0.0
                        ),
                    }
                )
            self._run_logger.log(iteration, **metrics)

    def _save_checkpoint(self, iteration: int, mean_return: Optional[float] = None):
        """Save training checkpoint only if it's better than previous best."""
        # Track best return
        if not hasattr(self, "_best_return"):
            self._best_return = -999.0

        # Only save if this is the best model so far
        should_save = (mean_return is None) or (mean_return > self._best_return)

        if should_save and mean_return is not None:
            self._best_return = mean_return
            logger.info(f"🏆 New best model! Return: {mean_return:.2f}%")

        # Always save checkpoint for recovery (with iteration number)
        checkpoint_path = self.checkpoint_dir / f"checkpoint_iter_{iteration}.pth"
        self.trader.save(str(checkpoint_path).replace(".pth", "_trader.pth"))
        self.adversary.save(str(checkpoint_path).replace(".pth", "_adversary.pth"))

        state = {
            "iteration": iteration,
            "total_steps": self.total_steps,
            "history": self.history,
            "config": self.config,
            "best_return": self._best_return,
        }

        torch.save(state, checkpoint_path)

        # Only save as "best_model" if it's actually better
        if should_save and mean_return is not None:
            best_path = self.checkpoint_dir / "best_model.pth"
            self.trader.save(str(best_path).replace(".pth", "_trader.pth"))
            self.adversary.save(str(best_path).replace(".pth", "_adversary.pth"))
            torch.save(state, best_path)
            logger.info(f"✅ Best model saved: {best_path}")

    def _save_final_checkpoint(self):
        """Save final trained models only if better than best."""
        final_path = self.checkpoint_dir / "final_model.pth"

        self.trader.save(str(final_path).replace(".pth", "_trader.pth"))
        self.adversary.save(str(final_path).replace(".pth", "_adversary.pth"))

        # Save history
        history_path = self.checkpoint_dir / "training_history.json"
        with open(history_path, "w") as f:
            json.dump(self.history, f, indent=2)

        # Create a final checkpoint file that includes metadata
        # so load_checkpoint can work similarly to regular checkpoints
        state = {
            "iteration": self.iteration,
            "total_steps": self.total_steps,
            "history": self.history,
            "config": self.config,
        }
        torch.save(state, final_path)

        logger.success(f"Final models saved: {final_path}")

    def load_checkpoint(self, path: str):
        """Load training checkpoint."""
        logger.info(f"Loading checkpoint from: {path}")

        # PyTorch >= 2.6: weights_only=True is default, fails if
        # dataclasses (PPOConfig, AdversarialConfig) are stored in the checkpoint.
        # weights_only=False is safe since checkpoints come from our own training.
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        # Load agents
        # We need to reconstruct the paths for trader and adversary weights
        # The main checkpoint stores training state, but agents save their own weights separately
        # in the save() method implementation above:
        # self.trader.save(str(checkpoint_path).replace(".pth", "_trader.pth"))

        trader_path = str(path).replace(".pth", "_trader.pth")
        adversary_path = str(path).replace(".pth", "_adversary.pth")

        # Load sidecar files (trader_path / adversary_path).
        # On first startup without previous checkpoint these files are missing - this is normal.
        # FileNotFoundError is logged as INFO, not WARNING.
        _trader_ok = False
        _adversary_ok = False
        if os.path.exists(trader_path):
            try:
                self.trader.load(trader_path)
                _trader_ok = True
            except Exception as e:
                logger.warning(f"Error loading trader weights: {e}")
        else:
            logger.info(
                f"Trader sidecar not present (OK on first startup): {trader_path}"
            )

        if os.path.exists(adversary_path):
            try:
                self.adversary.load(adversary_path)
                _adversary_ok = True
            except Exception as e:
                logger.warning(f"Error loading adversary weights: {e}")
        else:
            logger.info(
                f"Adversary sidecar not present (OK on first startup): {adversary_path}"
            )

        if not _trader_ok or not _adversary_ok:
            logger.info(
                "Sidecar files incomplete - starting with fresh weights "
                "(training progress from checkpoint will be loaded)."
            )

        # Load training state
        if "iteration" in checkpoint:
            self.iteration = checkpoint["iteration"]
        if "total_steps" in checkpoint:
            self.total_steps = checkpoint["total_steps"]
        if "history" in checkpoint:
            self.history = checkpoint["history"]

        logger.success(f"Resumed from iteration {self.iteration}")


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("ADVERSARIAL TRAINING SYSTEM TEST")
    print("=" * 80)

    # Mock environment for testing
    class MockEnv:
        def __init__(self):
            self.observation_space = type("", (), {"shape": (20,)})()
            self.action_space = type("", (), {"n": 3})()

        def reset(self):
            return np.random.randn(20), {}

        def step(self, action):
            obs = np.random.randn(20)
            reward = np.random.randn()
            done = np.random.rand() < 0.01
            info = {
                "return": 0.05,
                "risk_metrics": {"sharpe_ratio": 1.5, "max_drawdown": 0.1},
            }
            return obs, reward, done, False, info

    # Configure
    trader_config = PPOConfig(state_dim=20, n_actions=3)
    adversary_config = PPOConfig(state_dim=20, n_actions=3)

    config = AdversarialConfig(
        n_iterations=10,
        steps_per_iteration=100,
        trader_config=trader_config,
        adversary_config=adversary_config,
        adversary_start_iteration=5,
        log_frequency=2,
        save_frequency=5,
    )

    env = MockEnv()
    trainer = AdversarialTrainer(env, config)

    print("\n✓ Trainer initialized")

    # Test trajectory collection
    print("\n[TEST] Trajectory Collection")
    metrics = trainer.collect_trajectories(100, use_adversary=False)
    print(f"  Episodes: {len(metrics['episode_rewards'])}")
    print(f"  Mean reward: {metrics['mean_reward']:.4f}")

    # Test training step
    print("\n[TEST] Training Step")
    stats = trainer.train_trader(metrics["next_value"])
    print(f"  Actor loss: {stats['actor_loss']:.4f}")
    print(f"  Critic loss: {stats['critic_loss']:.4f}")

    print("\n" + "=" * 80)
    print("✓ ADVERSARIAL TRAINING SYSTEM TEST PASSED")
    print("=" * 80)
