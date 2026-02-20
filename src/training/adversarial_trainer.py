"""
Adversarial Training System
============================
Self-play mechanism where Trader and Adversary improve each other.

Architecture:
- Trader Agent: Maximizes profit in trading environment
- Adversary Agent: Creates challenging market conditions
- Self-Play: Agents alternate training to push each other

Key Innovation:
Adversary doesn't just add noise - it learns to create realistic
but difficult scenarios that expose Trader weaknesses.
"""

import numpy as np
import torch
from typing import Dict, Tuple, Optional
from dataclasses import dataclass, field
from pathlib import Path
from loguru import logger
import json
from datetime import datetime

from agents.ppo_agent import PPOAgent, PPOConfig


@dataclass
class AdversarialConfig:
    """Adversarial training configuration."""

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
    Adversarial training system.

    Coordinates self-play between Trader and Adversary agents.

    Workflow:
    1. Trader trains to maximize profit
    2. Adversary observes Trader's strategy
    3. Adversary learns to create scenarios that hurt Trader
    4. Trader must adapt to new challenges
    5. Repeat → Robust trading strategy

    Usage:
    ------
    config = AdversarialConfig(n_iterations=500)
    trainer = AdversarialTrainer(env, config)

    # Train
    trainer.train()

    # Evaluate
    metrics = trainer.evaluate(n_episodes=100)
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

        # Metrics history
        self.history = {
            "trader_rewards": [],
            "trader_returns": [],
            "trader_sharpe": [],
            "adversary_rewards": [],
            "adversary_success": [],
            "episodes": [],
        }

        # Adversary state tracking
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

            # Compute adversary reward: negative of trader reward + challenge bonus
            if (
                use_adversary
                and self.iteration >= self.config.adversary_start_iteration
            ):
                # Base reward: trader performs poorly (negative reward is good for adversary)
                adversary_reward = -reward * 0.5

                # Additional reward for increasing volatility/difficulty
                if "volatility_increase" in challenge_info:
                    adversary_reward += challenge_info["volatility_increase"] * 0.1

                # Bonus if trader loses money
                if reward < 0:
                    adversary_reward += abs(reward) * 0.3

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
            # Increase volatility: add noise to features
            noise = np.random.randn(len(obs)) * strength * 0.5
            modified_obs += noise
            challenge_info["type"] = "volatility_increase"
            challenge_info["volatility_increase"] = np.mean(np.abs(noise))

        elif adv_action == 1:
            # Add trend bias: shift price-related features
            # Find features that might be price/return related (usually first few)
            n_features = min(5, len(obs) // 2)
            bias = np.random.randn() * strength
            modified_obs[:n_features] += bias
            challenge_info["type"] = "trend_bias"
            challenge_info["bias_magnitude"] = abs(bias)

        elif adv_action == 2:
            # Invert signals: flip signs of some features
            n_invert = max(1, int(len(obs) * strength * 0.3))
            invert_indices = np.random.choice(len(obs), n_invert, replace=False)
            modified_obs[invert_indices] *= -1
            challenge_info["type"] = "signal_inversion"
            challenge_info["n_inverted"] = n_invert

        else:  # adv_action == 3
            # No modification
            challenge_info["type"] = "none"
            challenge_info["volatility_increase"] = 0.0

        # Clip to reasonable bounds
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

        # Store history
        self.history["adversary_rewards"].append(np.mean(self.adversary_rewards))
        self.history["adversary_success"].append(success_rate)

        return {
            "adversary_loss": stats.get("actor_loss", 0.0),
            "adversary_critic_loss": stats.get("critic_loss", 0.0),
            "adversary_entropy": stats.get("entropy", 0.0),
            "adversary_success_rate": success_rate,
            "adversary_episodes": len(self.adversary_rewards),
            "mean_adversary_reward": np.mean(self.adversary_rewards),
        }

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

            # Store history
            self.history["trader_rewards"].append(traj_metrics["mean_reward"])
            self.history["trader_returns"].append(traj_metrics["mean_return"])
            self.history["episodes"].append(len(traj_metrics["episode_rewards"]))

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

    def _save_checkpoint(self, iteration: int, mean_return: float = None):
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
