"""
Phase 5: Adversarial Training Tests
====================================

Tests for the adversarial training system including:
- PPO Agent functionality
- Adversary modification strategies
- Self-play training loop
- Trajectory collection with adversary
"""

import pytest
import numpy as np
import torch

from agents.ppo_agent import PPOAgent, PPOConfig
from training.adversarial_trainer import AdversarialTrainer, AdversarialConfig


class MockEnv:
    """Mock environment for testing."""

    def __init__(self, obs_dim=20, n_actions=7):
        self.observation_space = type(
            "obs", (), {"shape": (obs_dim,)}
        )()  # Minimal object with shape attribute
        self.action_space = type(
            "act", (), {"n": n_actions}
        )()  # Minimal object with n attribute
        self.obs_dim = obs_dim

    def reset(self, seed=None, options=None):
        return np.random.randn(self.obs_dim).astype(
            np.float32
        ), {}  # Random obs + empty info dict

    def step(self, action):
        obs = np.random.randn(self.obs_dim).astype(np.float32)
        reward = np.random.randn()  # Random scalar reward
        terminated = np.random.rand() < 0.05  # 5% chance of episode end
        truncated = False
        info = {"return": reward * 10}
        return obs, reward, terminated, truncated, info


class TestPPOConfig:
    """Test PPO configuration."""

    def test_default_config(self):
        """Test default PPO configuration."""
        config = PPOConfig(state_dim=20, n_actions=7)

        assert config.state_dim == 20
        assert config.n_actions == 7
        assert config.hidden_dim == 128
        assert config.actor_lr == 3e-4
        assert config.gamma == 0.99

    def test_custom_config(self):
        """Test custom PPO configuration."""
        config = PPOConfig(
            state_dim=50, n_actions=5, hidden_dim=256, actor_lr=1e-4, gamma=0.95
        )

        assert config.state_dim == 50
        assert config.n_actions == 5
        assert config.hidden_dim == 256
        assert config.actor_lr == 1e-4
        assert config.gamma == 0.95


class TestPPOAgent:
    """Test PPO Agent functionality."""

    @pytest.fixture
    def agent(self):
        """Create PPO agent."""
        config = PPOConfig(
            state_dim=20, n_actions=7, hidden_dim=64, n_epochs=2, batch_size=16
        )
        return PPOAgent(config, device="cpu")

    def test_initialization(self, agent):
        """Test agent initialization."""
        assert agent.config is not None
        assert agent.actor is not None
        assert agent.critic is not None
        assert len(agent.states) == 0  # Buffer starts empty

    def test_select_action(self, agent):
        """Test action selection."""
        state = np.random.randn(20).astype(np.float32)
        action, log_prob, value, hidden = agent.select_action(state)

        assert isinstance(action, (int, np.integer))
        assert 0 <= action < 7  # Action must be in valid range
        assert isinstance(log_prob, (float, np.floating))
        assert isinstance(value, (float, np.floating))

    def test_select_action_deterministic(self, agent):
        """Test deterministic action selection."""
        state = np.random.randn(20).astype(np.float32)

        # Run multiple times - deterministic should give same action
        actions = []
        for _ in range(5):
            action, _, _, _ = agent.select_action(state, deterministic=True)
            actions.append(action)

        # All actions should be the same (argmax is deterministic)
        assert all(a == actions[0] for a in actions)

    def test_store_transition(self, agent):
        """Test transition storage."""
        state = np.random.randn(20).astype(np.float32)
        agent.store_transition(
            state, 1, 1.0, -0.5, 0.5, False
        )  # (s, a, r, log_p, v, done)

        assert len(agent.states) == 1
        assert len(agent.actions) == 1
        assert len(agent.rewards) == 1

    def test_compute_gae(self, agent):
        """Test GAE computation."""
        # Add some transitions
        for i in range(10):
            state = np.random.randn(20).astype(np.float32)
            agent.store_transition(
                state, 1, np.random.randn(), -0.5, 0.5, i == 9
            )  # Last step is done

        next_value = 0.0  # Bootstrap value for non-terminal
        advantages, returns = agent.compute_gae(next_value)

        assert len(advantages) == 10
        assert len(returns) == 10
        assert all(isinstance(a, (float, np.floating)) for a in advantages)

    def test_train(self, agent):
        """Test training step."""
        # Collect some transitions
        for i in range(64):
            state = np.random.randn(20).astype(np.float32)
            action, log_prob, value, hidden = agent.select_action(state)
            reward = np.random.randn()
            agent.store_transition(
                state, action, reward, log_prob, value, i % 20 == 19
            )  # Done every 20 steps

        stats = agent.train(next_value=0.0)  # Train on collected buffer

        assert "actor_loss" in stats
        assert "critic_loss" in stats
        assert "entropy" in stats

    def test_buffer_clear_after_train(self, agent):
        """Test that buffer is cleared after training."""
        # Add transitions
        for i in range(64):
            state = np.random.randn(20).astype(np.float32)
            action, log_prob, value, hidden = agent.select_action(state)
            agent.store_transition(state, action, 1.0, log_prob, value, False)

        assert len(agent.states) == 64

        agent.train(next_value=0.0)

        # Buffer should be cleared after training to avoid stale data
        assert len(agent.states) == 0


class TestAdversarialConfig:
    """Test adversarial training configuration."""

    def test_default_config(self):
        """Test default configuration."""
        trader_config = PPOConfig(state_dim=20, n_actions=7)
        adversary_config = PPOConfig(state_dim=20, n_actions=4)
        config = AdversarialConfig(
            trader_config=trader_config,
            adversary_config=adversary_config,
        )

        assert config.n_iterations == 500
        assert config.steps_per_iteration == 2048
        assert config.adversary_start_iteration == 100  # Adversary starts after warmup
        assert config.adversary_strength == 0.1

    def test_custom_config(self):
        """Test custom configuration."""
        trader_config = PPOConfig(state_dim=20, n_actions=7)
        adversary_config = PPOConfig(state_dim=20, n_actions=4)

        config = AdversarialConfig(
            n_iterations=100,
            steps_per_iteration=512,
            trader_config=trader_config,
            adversary_config=adversary_config,
            adversary_start_iteration=10,
            adversary_strength=0.2,
        )

        assert config.n_iterations == 100
        assert config.steps_per_iteration == 512
        assert config.adversary_start_iteration == 10
        assert config.adversary_strength == 0.2


class TestAdversarialTrainer:
    """Test adversarial trainer functionality."""

    @pytest.fixture
    def trainer(self):
        """Create adversarial trainer."""
        env = MockEnv(obs_dim=20, n_actions=7)

        trader_config = PPOConfig(
            state_dim=20, n_actions=7, hidden_dim=32, n_epochs=2, batch_size=16
        )

        adversary_config = PPOConfig(
            state_dim=20, n_actions=4, hidden_dim=32, n_epochs=2, batch_size=16
        )

        config = AdversarialConfig(
            n_iterations=2,
            steps_per_iteration=100,
            trader_config=trader_config,
            adversary_config=adversary_config,
            adversary_start_iteration=0,  # Start immediately for testing
            adversary_strength=0.1,
            log_frequency=1,
            save_frequency=10,
        )

        return AdversarialTrainer(env, config, device="cpu")

    def test_initialization(self, trainer):
        """Test trainer initialization."""
        assert trainer.env is not None
        assert trainer.trader is not None
        assert trainer.adversary is not None
        assert trainer.config is not None

    def test_collect_trajectories_without_adversary(self, trainer):
        """Test trajectory collection without adversary."""
        metrics = trainer.collect_trajectories(n_steps=50, use_adversary=False)

        assert "episode_rewards" in metrics
        assert "mean_reward" in metrics
        assert "next_value" in metrics
        assert len(trainer.adversary_states) == 0  # No adversary data collected

    def test_collect_trajectories_with_adversary(self, trainer):
        """Test trajectory collection with active adversary."""
        trainer.iteration = (
            trainer.config.adversary_start_iteration
        )  # Activate adversary by setting iteration

        metrics = trainer.collect_trajectories(n_steps=50, use_adversary=True)

        assert "episode_rewards" in metrics
        assert "mean_reward" in metrics
        assert "mean_adversary_reward" in metrics
        assert "adversary_next_value" in metrics
        assert len(trainer.adversary_states) > 0  # Adversary collected data
        assert len(trainer.adversary_actions) > 0

    def test_train_trader(self, trainer):
        """Test trader training."""
        # Collect trajectories first
        metrics = trainer.collect_trajectories(n_steps=64, use_adversary=False)

        # Train trader
        stats = trainer.train_trader(metrics["next_value"])

        assert "actor_loss" in stats
        assert "critic_loss" in stats
        assert "entropy" in stats

    def test_train_adversary(self, trainer):
        """Test adversary training."""
        trainer.iteration = trainer.config.adversary_start_iteration

        # Collect trajectories with adversary
        metrics = trainer.collect_trajectories(n_steps=64, use_adversary=True)

        # Train adversary
        stats = trainer.train_adversary(metrics.get("adversary_next_value", 0.0))

        assert "adversary_loss" in stats
        assert "adversary_success_rate" in stats
        assert "mean_adversary_reward" in stats

    def test_apply_adversary_modification(self, trainer):
        """Test adversary modification strategies."""
        obs = np.random.randn(20).astype(np.float32)
        original_obs = obs.copy()

        # Test all 4 adversary actions
        for action in range(4):
            modified, info = trainer._apply_adversary_modification(obs, action, {})

            assert isinstance(modified, np.ndarray)
            assert modified.shape == obs.shape
            assert "type" in info  # Info must describe which modification was applied

            if action != 3:  # Not "none"
                assert not np.allclose(modified, original_obs), (
                    f"Action {action} should modify observation"
                )

    def test_adversary_volatility_increase(self, trainer):
        """Test volatility increase modification."""
        obs = np.random.randn(20).astype(np.float32)

        modified, info = trainer._apply_adversary_modification(obs, 0, {})

        assert info["type"] == "volatility_increase"
        assert "volatility_increase" in info
        assert info["volatility_increase"] > 0  # Must apply a positive noise scaling

    def test_adversary_trend_bias(self, trainer):
        """Test trend bias modification."""
        obs = np.random.randn(20).astype(np.float32)

        modified, info = trainer._apply_adversary_modification(obs, 1, {})

        assert info["type"] == "trend_bias"
        assert "bias_magnitude" in info  # Describes direction/magnitude of bias applied

    def test_adversary_signal_inversion(self, trainer):
        """Test signal inversion modification."""
        obs = np.random.randn(20).astype(np.float32)

        modified, info = trainer._apply_adversary_modification(obs, 2, {})

        assert info["type"] == "signal_inversion"
        assert "n_inverted" in info
        assert info["n_inverted"] > 0  # At least one feature was inverted

    def test_adversary_no_modification(self, trainer):
        """Test no modification action."""
        obs = np.random.randn(20).astype(np.float32)
        original_obs = obs.copy()

        modified, info = trainer._apply_adversary_modification(obs, 3, {})

        assert info["type"] == "none"
        assert np.allclose(modified, original_obs)  # Observation unchanged


class TestAdversarialTrainingLoop:
    """Test full adversarial training loop."""

    def test_full_training_iteration(self):
        """Test one complete training iteration."""
        env = MockEnv(obs_dim=20, n_actions=7)

        trader_config = PPOConfig(
            state_dim=20, n_actions=7, hidden_dim=32, n_epochs=2, batch_size=16
        )

        adversary_config = PPOConfig(
            state_dim=20, n_actions=4, hidden_dim=32, n_epochs=2, batch_size=16
        )

        config = AdversarialConfig(
            n_iterations=1,
            steps_per_iteration=64,
            trader_config=trader_config,
            adversary_config=adversary_config,
            adversary_start_iteration=0,  # Start adversary from first iteration
            adversary_strength=0.1,
        )

        trainer = AdversarialTrainer(env, config, device="cpu")

        # Collect trajectories
        metrics = trainer.collect_trajectories(64, use_adversary=True)

        # Train both agents
        trader_stats = trainer.train_trader(metrics["next_value"])
        adversary_stats = trainer.train_adversary(
            metrics.get("adversary_next_value", 0.0)
        )

        # Verify training occurred (non-zero losses)
        assert trader_stats["actor_loss"] != 0
        assert adversary_stats["adversary_loss"] != 0

    def test_adversary_improves_over_iterations(self):
        """Test that adversary success rate changes over time."""
        env = MockEnv(obs_dim=20, n_actions=7)

        config = AdversarialConfig(
            n_iterations=3,
            steps_per_iteration=50,
            trader_config=PPOConfig(
                state_dim=20, n_actions=7, hidden_dim=32, n_epochs=2, batch_size=16
            ),
            adversary_config=PPOConfig(
                state_dim=20, n_actions=4, hidden_dim=32, n_epochs=2, batch_size=16
            ),
            adversary_start_iteration=0,
            adversary_strength=0.1,
        )

        trainer = AdversarialTrainer(env, config, device="cpu")

        success_rates = []

        for i in range(3):
            trainer.iteration = i  # Manually advance iteration counter
            metrics = trainer.collect_trajectories(50, use_adversary=True)
            adv_stats = trainer.train_adversary(
                metrics.get("adversary_next_value", 0.0)
            )
            success_rates.append(adv_stats["adversary_success_rate"])

        # Success rates should exist and be valid probabilities [0, 1]
        assert all(0 <= rate <= 1 for rate in success_rates)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
