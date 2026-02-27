"""
Ensemble Methods for DRL Trading Agents
========================================
Combine multiple Deep Reinforcement Learning agents for robust trading decisions.

This module provides various ensemble techniques to improve trading system
reliability and performance by combining multiple DRL agents. Ensemble methods
reduce variance, improve stability, and can adapt to changing market conditions.

Ensemble Methods Implemented:
    1. Voting Ensemble: Majority vote from discrete-action agents
       - Best for classification-style trading decisions
       - Simple and robust baseline

    2. Weighted Ensemble: Weighted average of agent actions
       - Weights can be static or dynamically adjusted
       - Supports both discrete and continuous actions

    3. Stacking Ensemble: Meta-learner on agent outputs
      ns optimal combination of agent - Lear predictions
       - More sophisticated than simple averaging

    4. Bagging Ensemble: Bootstrap aggregating
       - Random sampling with replacement
       - Reduces overfitting variance

Additional Features:
    - DynamicEnsemble: Regime-aware agent switching
    - ModelSelector: Best model selection based on validation
    - Adaptive weight updating based on performance

Architecture:
    AgentEnsemble
    ├── Voting Ensemble
    ├── Weighted Ensemble
    ├── Stacking Ensemble
    └── Bagging Ensemble

    DynamicEnsemble (regime-based switching)

    ModelSelector (best agent selection)

Usage:
    from src.ensemble.ensemble_agents import AgentEnsemble, EnsembleConfig, create_ensemble

    # Create ensemble from trained agents
    config = EnsembleConfig(
        method="voting",
        window_size=10,
        temperature=1.0
    )
    ensemble = AgentEnsemble(agents=[agent1, agent2, agent3], config=config)

    # Get prediction
    action = ensemble.predict(state, deterministic=True)

    # Update weights based on performance
    ensemble.update_weights({0: 0.5, 1: 0.8, 2: 0.3})

Dependencies:
    - numpy: Numerical operations
    - torch: Neural network agents
    - scipy: Statistical functions

Note:
    All agents must produce compatible action outputs. For mixed agent types,
    ensure they either all have select_action() method or can be called as
    PyTorch models with forward() returning action tensors.
"""

import numpy as np
import torch
from typing import List, Dict, Optional, Callable
from dataclasses import dataclass
from collections import deque


@dataclass
class EnsembleConfig:
    """Configuration for ensemble methods."""

    method: str = "voting"  # voting, weighted, stacking, bagging
    weights: Optional[np.ndarray] = None  # For weighted ensemble
    window_size: int = 10  # For online performance tracking
    temperature: float = 1.0  # For softmax weighting


class AgentEnsemble:
    """
    Ensemble of DRL agents for robust trading predictions.

    Combines multiple trained Deep Reinforcement Learning agents to reduce
    prediction variance and improve stability. The ensemble maintains rolling
    performance history for adaptive weighting.

    Attributes:
        agents: List of trained DRL agent instances
        config: EnsembleConfiguration with method and parameters
        n_agents: Number of agents in ensemble
        performance_history: Rolling reward history per agent
        weights: Current ensemble weights (updated adaptively if configured)

    Ensemble Methods:
        voting: Majority vote (for discrete actions)
            - Each agent votes for an action
            - Most-voted action wins
            - Best for classification-style decisions

        weighted: Weighted average (for continuous/discrete actions)
            - Actions weighted by performance-based weights
            - Supports multi-dimensional action spaces

        stacking: Meta-learner combination
            - Concatenates all agent predictions
            - Applies weighted sum via meta-learner
            - Can learn non-linear combinations

        bagging: Bootstrap aggregating
            - Randomly samples agents with replacement
            - Averages predictions from sampled agents
            - Reduces overfitting variance

    Adaptive Weighting:
        Weights can be updated based on recent performance using update_weights().
        Uses Sharpe-like ratio (mean/std) for risk-adjusted scoring.
        Temperature parameter controls softmax sharpness.

    Example:
        >>> config = EnsembleConfig(method="voting", window_size=20)
        >>> ensemble = AgentEnsemble(agents=[a1, a2, a3], config=config)
        >>>
        >>> # Get prediction
        >>> action = ensemble.predict(state, deterministic=True)
        >>>
        >>> # Update weights based on episode rewards
        >>> ensemble.update_weights({0: 0.5, 1: 0.8, 2: 0.3})
        >>>
        >>> # Evaluate all agents
        >>> performances = ensemble.evaluate_agents(env, n_episodes=10)

    Note:
        Agents must either have a select_action(state, deterministic) method
        or be callable PyTorch models returning action tensors.
    """

    def __init__(self, agents: List, config: EnsembleConfig):
        """
        Initialize ensemble.

        Args:
            agents: List of trained DRL agents
            config: Ensemble configuration
        """
        self.agents = agents
        self.config = config
        self.n_agents = len(agents)

        # Performance tracking for adaptive weighting
        self.performance_history = {
            i: deque(maxlen=config.window_size)
            for i in range(self.n_agents)  # rolling reward history per agent
        }
        self.weights = (
            config.weights
            if config.weights is not None
            else np.ones(self.n_agents)
            / self.n_agents  # equal weights by default (uniform ensemble)
        )

    def predict(self, state: np.ndarray, deterministic: bool = True) -> np.ndarray:
        """
        Get ensemble prediction.

        Args:
            state: Current state
            deterministic: Whether to use deterministic policies

        Returns:
            Ensemble action
        """
        if self.config.method == "voting":
            return self._voting_predict(state, deterministic)
        elif self.config.method == "weighted":
            return self._weighted_predict(state, deterministic)
        elif self.config.method == "stacking":
            return self._stacking_predict(state)
        elif self.config.method == "bagging":
            return self._bagging_predict(state, deterministic)
        else:
            raise ValueError(f"Unknown ensemble method: {self.config.method}")

    def _voting_predict(self, state: np.ndarray, deterministic: bool) -> np.ndarray:
        """Voting ensemble (for discrete actions)."""
        votes = []
        for agent in self.agents:
            if hasattr(agent, "select_action"):
                # check if the agent's select_action supports the deterministic kwarg
                if "deterministic" in agent.select_action.__code__.co_varnames:
                    action = agent.select_action(state, deterministic=deterministic)
                else:
                    action = agent.select_action(state)
            else:
                # Assume PyTorch model with forward() returning action logits
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).unsqueeze(
                        0
                    )  # add batch dimension
                    action_probs = agent(state_tensor)
                    action = action_probs.argmax().item()  # greedy action
            votes.append(action)

        # Majority vote: most-voted action wins
        return np.bincount(votes).argmax()

    def _weighted_predict(self, state: np.ndarray, deterministic: bool) -> np.ndarray:
        """Weighted ensemble (for continuous actions)."""
        actions = []
        for agent in self.agents:
            if hasattr(agent, "select_action"):
                if "deterministic" in agent.select_action.__code__.co_varnames:
                    action = agent.select_action(state, deterministic=deterministic)
                else:
                    action = agent.select_action(state)
            else:
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).unsqueeze(0)
                    action = agent(state_tensor).cpu().numpy()[0]
            actions.append(action)

        actions = np.array(actions)

        # Weighted average
        if len(actions.shape) == 1:
            return np.average(actions, weights=self.weights)
        else:
            # Multi-dimensional actions
            return np.average(actions, axis=0, weights=self.weights)

    def _stacking_predict(self, state: np.ndarray) -> np.ndarray:
        """Stacking ensemble with meta-learner."""
        # Get predictions from all agents
        predictions = []
        for agent in self.agents:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                if hasattr(agent, "actor"):
                    pred = agent.actor(state_tensor).cpu().numpy()[0]
                else:
                    pred = agent(state_tensor).cpu().numpy()[0]
                predictions.append(pred)

        predictions = np.concatenate(predictions)

        # Meta-learner (simple weighted sum, can be replaced with trained model)
        return np.dot(predictions, self.weights)

    def _bagging_predict(self, state: np.ndarray, deterministic: bool) -> np.ndarray:
        """Bootstrap aggregating."""
        # Randomly sample agents with replacement
        indices = np.random.choice(self.n_agents, size=self.n_agents, replace=True)

        actions = []
        for idx in indices:
            agent = self.agents[idx]
            if hasattr(agent, "select_action"):
                action = agent.select_action(state, deterministic=deterministic)
            else:
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).unsqueeze(0)
                    action = agent(state_tensor).cpu().numpy()[0]
            actions.append(action)

        return np.mean(actions, axis=0)

    def update_weights(self, rewards: Dict[int, float]):
        """
        Update ensemble weights based on agent performance.

        Args:
            rewards: Dictionary of {agent_idx: reward}
        """
        for idx, reward in rewards.items():
            self.performance_history[idx].append(reward)  # rolling buffer per agent

        # Calculate performance scores
        scores = []
        for i in range(self.n_agents):
            if len(self.performance_history[i]) > 0:
                # Sharpe-like score: mean return / return std (higher is better)
                returns = np.array(list(self.performance_history[i]))
                if np.std(returns) > 0:
                    score = np.mean(returns) / np.std(returns)  # risk-adjusted score
                else:
                    score = np.mean(returns)  # no volatility → use raw mean
            else:
                score = 0.0  # no history yet → neutral score
            scores.append(score)

        scores = np.array(scores)

        # Softmax weighting: better-performing agents get exponentially higher weight
        exp_scores = np.exp(
            scores / self.config.temperature
        )  # temperature controls sharpness
        self.weights = exp_scores / np.sum(exp_scores)  # normalise to sum to 1

    def evaluate_agents(self, env, n_episodes: int = 10) -> Dict[int, float]:
        """
        Evaluate all agents in environment.

        Args:
            env: Trading environment
            n_episodes: Number of episodes per agent

        Returns:
            Dictionary of agent performances
        """
        performances = {}

        for idx, agent in enumerate(self.agents):
            total_reward = 0.0
            for _ in range(n_episodes):
                state, _ = env.reset()
                done = False
                episode_reward = 0.0

                while not done:
                    if hasattr(agent, "select_action"):
                        action = agent.select_action(state, deterministic=True)
                    else:
                        with torch.no_grad():
                            state_tensor = torch.FloatTensor(state).unsqueeze(0)
                            action = agent(state_tensor).cpu().numpy()[0]

                    state, reward, terminated, truncated, _ = env.step(action)
                    done = terminated or truncated
                    episode_reward += reward

                total_reward += episode_reward

            performances[idx] = total_reward / n_episodes

        return performances


class DynamicEnsemble(AgentEnsemble):
    """
    Dynamic ensemble that switches between agents based on market regime.
    """

    def __init__(self, agents: List, regime_detector, agent_regime_map: Dict[int, str]):
        """
        Initialize dynamic ensemble.

        Args:
            agents: List of agents
            regime_detector: Market regime detector (e.g., HMM)
            agent_regime_map: Map of {agent_idx: regime_name}
        """
        super().__init__(agents, EnsembleConfig(method="weighted"))
        self.regime_detector = regime_detector
        self.agent_regime_map = agent_regime_map

    def predict(self, state: np.ndarray, **kwargs) -> np.ndarray:
        """Predict based on current market regime."""
        # Detect current regime
        current_regime = self.regime_detector.predict(state)

        # Find best agent for this regime
        best_agent_idx = None
        best_score = -np.inf

        for idx, regime in self.agent_regime_map.items():
            if regime == current_regime:
                # Evaluate this agent's recent performance
                if len(self.performance_history[idx]) > 0:
                    score = np.mean(list(self.performance_history[idx]))
                    if score > best_score:
                        best_score = score
                        best_agent_idx = idx

        # If no specific agent found, use ensemble
        if best_agent_idx is None:
            return super().predict(state, **kwargs)

        # Use best agent for this regime
        agent = self.agents[best_agent_idx]
        if hasattr(agent, "select_action"):
            return agent.select_action(state, deterministic=True)
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                return agent(state_tensor).cpu().numpy()[0]


class ModelSelector:
    """
    Select best model from ensemble based on validation performance.
    """

    def __init__(self, metric: str = "sharpe"):
        """
        Initialize model selector.

        Args:
            metric: Metric to use for selection (sharpe, return, sortino)
        """
        self.metric = metric
        self.best_agent_idx = None
        self.performances = {}

    def select_best(self, agents: List, env, n_episodes: int = 5) -> int:
        """
        Select best agent based on validation performance.

        Args:
            agents: List of agents
            env: Validation environment
            n_episodes: Number of validation episodes

        Returns:
            Index of best agent
        """
        best_score = -np.inf

        for idx, agent in enumerate(agents):
            scores = []

            for _ in range(n_episodes):
                state, _ = env.reset()
                done = False
                rewards = []

                while not done:
                    if hasattr(agent, "select_action"):
                        action = agent.select_action(state, deterministic=True)
                    else:
                        with torch.no_grad():
                            state_tensor = torch.FloatTensor(state).unsqueeze(0)
                            action = agent(state_tensor).cpu().numpy()[0]

                    state, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated
                    rewards.append(reward)

                # Calculate metric
                if self.metric == "sharpe":
                    if len(rewards) > 1 and np.std(rewards) > 0:
                        score = np.mean(rewards) / np.std(rewards)
                    else:
                        score = 0.0
                elif self.metric == "return":
                    score = np.sum(rewards)
                else:
                    score = np.mean(rewards)

                scores.append(score)

            avg_score = np.mean(scores)
            self.performances[idx] = avg_score

            if avg_score > best_score:
                best_score = avg_score
                self.best_agent_idx = idx

        return self.best_agent_idx

    def get_best_agent(self, agents: List):
        """Get the best agent."""
        if self.best_agent_idx is None:
            raise ValueError("Must call select_best() first")
        return agents[self.best_agent_idx]


# Factory function
def create_ensemble(agents: List, method: str = "voting", **kwargs) -> AgentEnsemble:
    """
    Create ensemble of agents.

    Args:
        agents: List of agents
        method: Ensemble method
        **kwargs: Additional arguments

    Returns:
        Ensemble instance
    """
    config = EnsembleConfig(method=method, **kwargs)
    return AgentEnsemble(agents, config)
