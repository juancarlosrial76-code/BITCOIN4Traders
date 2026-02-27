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

from agents.ppo_agent import PPOAgent, PPOConfig


def _load_mem_cfg() -> dict:
    """Laedt memory_management.yaml (graceful fallback)."""
    try:
        cfg_path = Path("config/memory_management.yaml")
        if cfg_path.exists() and _YAML_OK:
            with open(cfg_path) as f:
                return yaml.safe_load(f) or {}
    except Exception:
        pass
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
        """
        self.env = env
        self.config = config
        self.device = device

        # Create agents
        self.trader = PPOAgent(config.trader_config, device)
        self.adversary = PPOAgent(config.adversary_config, device)

        # Training state
        self.iteration = 0
        self.total_steps = 0

        # Memory-Konfiguration laden
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

        # Metrics history (begrenzt auf max_history Eintraege - kein unbegrenztes Wachstum)
        self.history = {
            "trader_rewards": [],
            "trader_returns": [],
            "trader_sharpe": [],
            "adversary_rewards": [],
            "adversary_success": [],
            "episodes": [],
        }

        # Adversary state tracking (werden nach jedem collect_trajectories geloescht)
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

        # Track adversary performance
        adversary_episode_rewards = []
        adversary_challenges = []

        # Reset hidden states
        trader_hidden = None
        adversary_hidden = None

        obs, info = self.env.reset()
        episode_reward = 0
        episode_length = 0

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

                # Track adversary episode reward
                if (
                    use_adversary
                    and self.iteration >= self.config.adversary_start_iteration
                ):
                    adversary_episode_rewards.append(
                        sum(self.adversary_rewards[-episode_length:])
                    )

                obs, info = self.env.reset()
                episode_reward = 0
                episode_length = 0

                # Reset hidden states on episode completion
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

        return {
            "episode_rewards": episode_rewards,
            "episode_returns": episode_returns,
            "episode_lengths": episode_lengths,
            "mean_reward": np.mean(episode_rewards) if episode_rewards else 0.0,
            "mean_return": np.mean(episode_returns) if episode_returns else 0.0,
            "mean_length": np.mean(episode_lengths) if episode_lengths else 0.0,
            "next_value": next_value,
            "adversary_next_value": adv_next_value,
            "adversary_episode_rewards": adversary_episode_rewards,
            "mean_adversary_reward": np.mean(adversary_episode_rewards)
            if adversary_episode_rewards
            else 0.0,
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

        # Store history (begrenzt - kein unbegrenztes Wachstum)
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

        # RAM freigeben: Adversary-Puffer loeschen (konfigurierbar)
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

    def train(self):
        """
        Main training loop.

        Alternates between:
        1. Trader improvement
        2. Adversary learning (after warm-up)
        3. Evaluation
        4. Checkpointing
        """
        logger.info("Starting adversarial training...")

        for iteration in range(self.config.n_iterations):
            self.iteration = iteration

            # Collect trajectories
            use_adversary = iteration >= self.config.adversary_start_iteration

            logger.info(f"\n{'=' * 80}")
            logger.info(f"Iteration {iteration + 1}/{self.config.n_iterations}")
            logger.info(f"Adversary active: {use_adversary}")

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

            # Checkpoint
            if iteration % self.config.save_frequency == 0:
                mean_ret = traj_metrics.get("mean_return", 0)
                self._save_checkpoint(iteration, mean_ret)

            # Store history (begrenzt auf max_history)
            self.history["trader_rewards"].append(traj_metrics["mean_reward"])
            self.history["trader_returns"].append(traj_metrics["mean_return"])
            self.history["episodes"].append(len(traj_metrics["episode_rewards"]))
            self._trim_history()

            # ── RAM + GPU-Speicher freigeben ──────────────────────────────
            if iteration % self._cuda_every == 0:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

            # ── IPython Output-Cache leeren (Colab) ───────────────────────
            if self._reset_out and iteration % self._ipython_every == 0:
                try:
                    from IPython import get_ipython

                    ip = get_ipython()
                    if ip is not None:
                        ip.run_line_magic("reset_selective", "-f _i")
                except Exception:
                    pass

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
        episode_sharpes = []
        episode_max_dds = []

        for ep in range(n_episodes):
            obs, info = self.env.reset()
            done = False

            hidden = None
            while not done:
                # Deterministic action selection
                action, _, _, hidden = self.trader.select_action(
                    obs, hidden, deterministic=True
                )
                obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated

            # Collect metrics
            episode_returns.append(info.get("return", 0.0))

            if "risk_metrics" in info:
                sharpe = info["risk_metrics"].get("sharpe_ratio", 0.0)
                max_dd = info["risk_metrics"].get("max_drawdown", 0.0)
            else:
                sharpe = 0.0
                max_dd = 0.0

            episode_sharpes.append(sharpe)
            episode_max_dds.append(max_dd)

        metrics = {
            "mean_return": np.mean(episode_returns),
            "std_return": np.std(episode_returns),
            "mean_sharpe": np.mean(episode_sharpes),
            "mean_max_dd": np.mean(episode_max_dds),
            "win_rate": np.mean([r > 0 for r in episode_returns]),
        }

        logger.info("\nEvaluation Results:")
        logger.info(f"  Mean Return: {metrics['mean_return'] * 100:.2f}%")
        logger.info(f"  Std Return: {metrics['std_return'] * 100:.2f}%")
        logger.info(f"  Mean Sharpe: {metrics['mean_sharpe']:.2f}")
        logger.info(f"  Mean Max DD: {metrics['mean_max_dd'] * 100:.2f}%")
        logger.info(f"  Win Rate: {metrics['win_rate'] * 100:.1f}%")

        return metrics

    def _trim_history(self):
        """
        Begrenzt alle history-Listen auf self._max_history Eintraege.
        Verhindert unbegrenztes RAM-Wachstum ueber 500+ Iterationen.
        0 = kein Limit (fuer lokale Maschinen).
        """
        if self._max_history <= 0:
            return
        for key in self.history:
            lst = self.history[key]
            if len(lst) > self._max_history:
                # Behalte nur die letzten max_history Eintraege
                self.history[key] = lst[-self._max_history :]

    def _log_iteration(
        self,
        iteration: int,
        traj_metrics: Dict,
        trader_stats: Dict,
        adversary_stats: Dict,
    ):
        """Log iteration metrics."""
        logger.info(f"\nIteration {iteration + 1} Results:")
        logger.info(f"  Episodes: {len(traj_metrics['episode_rewards'])}")
        logger.info(f"  Mean Reward: {traj_metrics['mean_reward']:.4f}")
        logger.info(f"  Mean Return: {traj_metrics['mean_return'] * 100:.2f}%")
        logger.info(f"  Mean Length: {traj_metrics['mean_length']:.0f}")

        if trader_stats:
            logger.info(f"\nTrader Training:")
            logger.info(f"  Actor Loss: {trader_stats['actor_loss']:.4f}")
            logger.info(f"  Critic Loss: {trader_stats['critic_loss']:.4f}")
            logger.info(f"  Entropy: {trader_stats['entropy']:.4f}")
            logger.info(f"  KL: {trader_stats['mean_kl']:.4f}")

        if adversary_stats:
            logger.info(f"\nAdversary Training:")
            logger.info(f"  Loss: {adversary_stats.get('adversary_loss', 0):.4f}")
            logger.info(
                f"  Success: {adversary_stats.get('adversary_success_rate', 0) * 100:.1f}%"
            )

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

        checkpoint = torch.load(path, map_location=self.device)

        # Load agents
        # We need to reconstruct the paths for trader and adversary weights
        # The main checkpoint stores training state, but agents save their own weights separately
        # in the save() method implementation above:
        # self.trader.save(str(checkpoint_path).replace(".pth", "_trader.pth"))

        trader_path = str(path).replace(".pth", "_trader.pth")
        adversary_path = str(path).replace(".pth", "_adversary.pth")

        try:
            self.trader.load(trader_path)
            self.adversary.load(adversary_path)
        except Exception as e:
            logger.warning(f"Could not load agent weights from sidecar files: {e}")
            logger.warning(
                "Attempting to load from main checkpoint if embedded (legacy support)..."
            )
            # If we ever decide to embed weights in the main file

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
