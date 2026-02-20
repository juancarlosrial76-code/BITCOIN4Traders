"""
Ensemble Methods for DRL Agents
================================
Combine multiple DRL agents for robust trading decisions.

Methods:
1. Voting Ensemble: Majority vote from agents
2. Weighted Ensemble: Weighted average of agent actions
3. Stacking Ensemble: Meta-learner on agent outputs
4. Bagging Ensemble: Bootstrap aggregating
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
    Ensemble of DRL agents for robust trading.

    Reduces variance and improves stability by combining
    multiple agents with different architectures or seeds.
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
            i: deque(maxlen=config.window_size) for i in range(self.n_agents)
        }
        self.weights = (
            config.weights
            if config.weights is not None
            else np.ones(self.n_agents) / self.n_agents
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
                if "deterministic" in agent.select_action.__code__.co_varnames:
                    action = agent.select_action(state, deterministic=deterministic)
                else:
                    action = agent.select_action(state)
            else:
                # Assume PyTorch model
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).unsqueeze(0)
                    action_probs = agent(state_tensor)
                    action = action_probs.argmax().item()
            votes.append(action)

        # Majority vote
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
            self.performance_history[idx].append(reward)

        # Calculate performance scores
        scores = []
        for i in range(self.n_agents):
            if len(self.performance_history[i]) > 0:
                # Sharpe-like score
                returns = np.array(list(self.performance_history[i]))
                if np.std(returns) > 0:
                    score = np.mean(returns) / np.std(returns)
                else:
                    score = np.mean(returns)
            else:
                score = 0.0
            scores.append(score)

        scores = np.array(scores)

        # Softmax weighting
        exp_scores = np.exp(scores / self.config.temperature)
        self.weights = exp_scores / np.sum(exp_scores)

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
