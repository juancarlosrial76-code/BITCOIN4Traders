"""
DRL Agents Module - Extended Algorithms
========================================
Additional DRL algorithms beyond PPO:
- DQN (Deep Q-Network)
- DDPG (Deep Deterministic Policy Gradient)
- SAC (Soft Actor-Critic)
- A2C (Advantage Actor-Critic)
- TD3 (Twin Delayed Deep Deterministic Policy Gradient)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Tuple, Dict, Optional, List
import copy


class DQNAgent:
    """
    Deep Q-Network (DQN) Agent.

    For discrete action spaces. Uses experience replay and target network.
    Suitable for trading with discrete actions (buy/sell/hold).
    """

    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        learning_rate: float = 1e-3,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        buffer_size: int = 100000,
        batch_size: int = 64,
        target_update: int = 1000,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update = target_update
        self.device = device

        # Q-Networks
        self.q_network = self._build_network().to(device)
        self.target_network = copy.deepcopy(self.q_network)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)

        # Replay buffer
        self.buffer = ReplayBuffer(buffer_size)
        self.steps = 0

    def _build_network(self) -> nn.Module:
        """Build Q-Network."""
        return nn.Sequential(
            nn.Linear(self.state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, self.n_actions),
        )

    def select_action(self, state: np.ndarray, epsilon: Optional[float] = None) -> int:
        """Select action using epsilon-greedy policy."""
        eps = epsilon if epsilon is not None else self.epsilon

        if np.random.random() < eps:
            return np.random.randint(self.n_actions)

        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            return q_values.argmax().item()

    def store_transition(self, state, action, reward, next_state, done):
        """Store transition in replay buffer."""
        self.buffer.push(state, action, reward, next_state, done)

    def train(self) -> Optional[float]:
        """Train the agent."""
        if len(self.buffer) < self.batch_size:
            return None

        # Sample batch
        states, actions, rewards, next_states, dones = self.buffer.sample(
            self.batch_size
        )

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # Current Q values
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze()

        # Target Q values
        with torch.no_grad():
            next_q = self.target_network(next_states).max(1)[0]
            target_q = rewards + (1 - dones) * self.gamma * next_q

        # Loss and update
        loss = nn.MSELoss()(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update target network
        self.steps += 1
        if self.steps % self.target_update == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

        return loss.item()

    def save(self, path: str):
        """Save model."""
        torch.save(
            {
                "q_network": self.q_network.state_dict(),
                "target_network": self.target_network.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "epsilon": self.epsilon,
                "steps": self.steps,
            },
            path,
        )

    def load(self, path: str):
        """Load model."""
        checkpoint = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint["q_network"])
        self.target_network.load_state_dict(checkpoint["target_network"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.epsilon = checkpoint["epsilon"]
        self.steps = checkpoint["steps"]


class ActorNetwork(nn.Module):
    """Actor network for continuous action spaces."""

    def __init__(self, state_dim: int, action_dim: int, max_action: float = 1.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 400),
            nn.ReLU(),
            nn.Linear(400, 300),
            nn.ReLU(),
            nn.Linear(300, action_dim),
            nn.Tanh(),
        )
        self.max_action = max_action

    def forward(self, state):
        return self.max_action * self.net(state)


class CriticNetwork(nn.Module):
    """Critic network for Q-value estimation."""

    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 400),
            nn.ReLU(),
            nn.Linear(400, 300),
            nn.ReLU(),
            nn.Linear(300, 1),
        )

    def forward(self, state, action):
        return self.net(torch.cat([state, action], dim=1))


class DDPGAgent:
    """
    Deep Deterministic Policy Gradient (DDPG) Agent.

    For continuous action spaces. Uses actor-critic architecture
    with target networks and soft updates.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        max_action: float = 1.0,
        learning_rate: float = 1e-3,
        gamma: float = 0.99,
        tau: float = 0.005,
        buffer_size: int = 1000000,
        batch_size: int = 64,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.device = device

        # Actor
        self.actor = ActorNetwork(state_dim, action_dim, max_action).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate)

        # Critic
        self.critic = CriticNetwork(state_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=learning_rate)

        # Replay buffer
        self.buffer = ReplayBuffer(buffer_size)

    def select_action(self, state: np.ndarray, noise: float = 0.1) -> np.ndarray:
        """Select action with exploration noise."""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action = self.actor(state_tensor).cpu().numpy()[0]

        # Add exploration noise
        if noise > 0:
            action += np.random.normal(0, noise, size=self.action_dim)
            action = np.clip(action, -self.max_action, self.max_action)

        return action

    def store_transition(self, state, action, reward, next_state, done):
        """Store transition."""
        self.buffer.push(state, action, reward, next_state, done)

    def train(self) -> Tuple[Optional[float], Optional[float]]:
        """Train actor and critic."""
        if len(self.buffer) < self.batch_size:
            return None, None

        # Sample batch
        states, actions, rewards, next_states, dones = self.buffer.sample(
            self.batch_size
        )

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # Update Critic
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            target_q = self.critic_target(next_states, next_actions)
            target_q = rewards + (1 - dones) * self.gamma * target_q

        current_q = self.critic(states, actions)
        critic_loss = nn.MSELoss()(current_q, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update Actor
        actor_actions = self.actor(states)
        actor_loss = -self.critic(states, actor_actions).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Soft update targets
        self._soft_update(self.actor, self.actor_target)
        self._soft_update(self.critic, self.critic_target)

        return critic_loss.item(), actor_loss.item()

    def _soft_update(self, source, target):
        """Soft update target network."""
        for param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )


class SACAgent:
    """
    Soft Actor-Critic (SAC) Agent.

    Maximum entropy RL algorithm for continuous action spaces.
    Better sample efficiency than DDPG.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        max_action: float = 1.0,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha: float = 0.2,
        buffer_size: int = 1000000,
        batch_size: int = 256,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.batch_size = batch_size
        self.device = device

        # Actor (Gaussian policy)
        self.actor = GaussianActor(state_dim, action_dim, max_action).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate)

        # Critics (double Q-learning)
        self.critic1 = CriticNetwork(state_dim, action_dim).to(device)
        self.critic2 = CriticNetwork(state_dim, action_dim).to(device)
        self.critic1_target = copy.deepcopy(self.critic1)
        self.critic2_target = copy.deepcopy(self.critic2)

        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=learning_rate)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=learning_rate)

        # Replay buffer
        self.buffer = ReplayBuffer(buffer_size)

    def select_action(self, state: np.ndarray, evaluate: bool = False) -> np.ndarray:
        """Select action from policy."""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            if evaluate:
                _, _, action = self.actor.sample(state_tensor)
            else:
                action, _, _ = self.actor.sample(state_tensor)
            return action.cpu().numpy()[0]

    def store_transition(self, state, action, reward, next_state, done):
        """Store transition."""
        self.buffer.push(state, action, reward, next_state, done)

    def train(self) -> Dict[str, float]:
        """Train SAC agent."""
        if len(self.buffer) < self.batch_size:
            return {}

        # Sample batch
        states, actions, rewards, next_states, dones = self.buffer.sample(
            self.batch_size
        )

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # Update Critics
        with torch.no_grad():
            next_actions, next_log_probs, _ = self.actor.sample(next_states)
            target_q1 = self.critic1_target(next_states, next_actions)
            target_q2 = self.critic2_target(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_probs
            target_q = rewards + (1 - dones) * self.gamma * target_q

        current_q1 = self.critic1(states, actions)
        current_q2 = self.critic2(states, actions)

        critic1_loss = nn.MSELoss()(current_q1, target_q)
        critic2_loss = nn.MSELoss()(current_q2, target_q)

        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()

        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()

        # Update Actor
        new_actions, log_probs, _ = self.actor.sample(states)
        q1_new = self.critic1(states, new_actions)
        q2_new = self.critic2(states, new_actions)
        q_new = torch.min(q1_new, q2_new)

        actor_loss = (self.alpha * log_probs - q_new).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Soft update targets
        self._soft_update(self.critic1, self.critic1_target)
        self._soft_update(self.critic2, self.critic2_target)

        return {
            "critic1_loss": critic1_loss.item(),
            "critic2_loss": critic2_loss.item(),
            "actor_loss": actor_loss.item(),
        }

    def _soft_update(self, source, target):
        """Soft update target network."""
        for param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )


class GaussianActor(nn.Module):
    """Gaussian policy for SAC."""

    def __init__(self, state_dim: int, action_dim: int, max_action: float = 1.0):
        super().__init__()
        self.max_action = max_action

        self.net = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU()
        )

        self.mean = nn.Linear(256, action_dim)
        self.log_std = nn.Linear(256, action_dim)

    def forward(self, state):
        x = self.net(state)
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, -20, 2)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()
        action = torch.tanh(x_t) * self.max_action

        # Compute log probability
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(
            self.max_action * (1 - action.pow(2) / self.max_action**2) + 1e-6
        )
        log_prob = log_prob.sum(1, keepdim=True)

        return action, log_prob, mean


class ReplayBuffer:
    """Experience Replay Buffer."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        """Add transition to buffer."""
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int):
        """Sample random batch."""
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, next_states, dones = zip(
            *[self.buffer[i] for i in indices]
        )

        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones),
        )

    def __len__(self):
        return len(self.buffer)


class A2CAgent:
    """
    Advantage Actor-Critic (A2C) Agent.

    Synchronous version of A3C. Uses on-policy updates.
    Good for discrete and continuous action spaces.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        discrete: bool = True,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        entropy_coef: float = 0.01,
        value_loss_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.discrete = discrete
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.max_grad_norm = max_grad_norm
        self.device = device

        # Actor-Critic network
        self.actor_critic = ActorCriticNetwork(state_dim, action_dim, discrete).to(
            device
        )
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=learning_rate)

    def select_action(self, state: np.ndarray, deterministic: bool = False):
        """Select action from policy."""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

            if self.discrete:
                logits, _ = self.actor_critic(state_tensor)
                dist = torch.distributions.Categorical(logits=logits)

                if deterministic:
                    action = logits.argmax().item()
                else:
                    action = dist.sample().item()

                log_prob = dist.log_prob(torch.tensor(action))
                return action, log_prob.item()
            else:
                mean, log_std, _ = self.actor_critic(state_tensor)
                std = log_std.exp()

                if deterministic:
                    action = mean
                else:
                    dist = torch.distributions.Normal(mean, std)
                    action = dist.sample()

                log_prob = -0.5 * (
                    ((action - mean) / (std + 1e-8)) ** 2
                    + 2 * log_std
                    + np.log(2 * np.pi)
                )
                log_prob = log_prob.sum(dim=-1)

                return action.cpu().numpy()[0], log_prob.item()

    def compute_returns(
        self, rewards: List[float], dones: List[bool], next_value: float
    ):
        """Compute discounted returns."""
        returns = []
        R = next_value

        for reward, done in zip(reversed(rewards), reversed(dones)):
            if done:
                R = 0
            R = reward + self.gamma * R
            returns.insert(0, R)

        return torch.FloatTensor(returns).to(self.device)

    def train(self, states, actions, rewards, dones, next_state):
        """Train using collected rollout."""
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = (
            torch.LongTensor(actions).to(self.device)
            if self.discrete
            else torch.FloatTensor(actions).to(self.device)
        )

        # Compute returns
        with torch.no_grad():
            if self.discrete:
                _, next_value = self.actor_critic(
                    torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
                )
            else:
                _, _, next_value = self.actor_critic(
                    torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
                )
            next_value = next_value.item()

        returns = self.compute_returns(rewards, dones, next_value)

        # Forward pass
        if self.discrete:
            logits, values = self.actor_critic(states)
            dist = torch.distributions.Categorical(logits=logits)
            log_probs = dist.log_prob(actions)
            entropy = dist.entropy()
        else:
            mean, log_std, values = self.actor_critic(states)
            std = log_std.exp()
            dist = torch.distributions.Normal(mean, std)
            log_probs = dist.log_prob(actions).sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1)

        # Compute advantages
        advantages = returns - values.squeeze()

        # Losses
        actor_loss = -(log_probs * advantages.detach()).mean()
        critic_loss = advantages.pow(2).mean()
        entropy_loss = entropy.mean()

        total_loss = (
            actor_loss
            + self.value_loss_coef * critic_loss
            - self.entropy_coef * entropy_loss
        )

        # Update
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.actor_critic.parameters(), self.max_grad_norm
        )
        self.optimizer.step()

        return {
            "total_loss": total_loss.item(),
            "actor_loss": actor_loss.item(),
            "critic_loss": critic_loss.item(),
            "entropy": entropy_loss.item(),
        }

    def save(self, path: str):
        """Save model."""
        torch.save(
            {
                "model": self.actor_critic.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            },
            path,
        )

    def load(self, path: str):
        """Load model."""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor_critic.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])


class ActorCriticNetwork(nn.Module):
    """Combined Actor-Critic network for A2C."""

    def __init__(self, state_dim: int, action_dim: int, discrete: bool = True):
        super().__init__()
        self.discrete = discrete

        # Shared layers
        self.shared = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU()
        )

        # Actor head
        if discrete:
            self.actor = nn.Linear(256, action_dim)
        else:
            self.actor_mean = nn.Linear(256, action_dim)
            self.actor_log_std = nn.Parameter(torch.zeros(action_dim))

        # Critic head
        self.critic = nn.Linear(256, 1)

    def forward(self, state):
        x = self.shared(state)
        value = self.critic(x)

        if self.discrete:
            logits = self.actor(x)
            return logits, value
        else:
            mean = self.actor_mean(x)
            log_std = self.actor_log_std.expand_as(mean)
            return mean, log_std, value


class TD3Agent:
    """
    Twin Delayed Deep Deterministic Policy Gradient (TD3).

    Improvement over DDPG with:
    - Twin critics to reduce overestimation
    - Delayed policy updates
    - Target policy smoothing
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        max_action: float = 1.0,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        policy_noise: float = 0.2,
        noise_clip: float = 0.5,
        policy_freq: int = 2,
        buffer_size: int = 1000000,
        batch_size: int = 256,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.gamma = gamma
        self.tau = tau
        self.policy_noise = policy_noise * max_action
        self.noise_clip = noise_clip * max_action
        self.policy_freq = policy_freq
        self.batch_size = batch_size
        self.device = device

        # Actor
        self.actor = ActorNetwork(state_dim, action_dim, max_action).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate)

        # Twin Critics
        self.critic1 = CriticNetwork(state_dim, action_dim).to(device)
        self.critic2 = CriticNetwork(state_dim, action_dim).to(device)
        self.critic1_target = copy.deepcopy(self.critic1)
        self.critic2_target = copy.deepcopy(self.critic2)

        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=learning_rate)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=learning_rate)

        # Replay buffer
        self.buffer = ReplayBuffer(buffer_size)
        self.total_it = 0

    def select_action(self, state: np.ndarray, noise: float = 0.1) -> np.ndarray:
        """Select action with exploration noise."""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action = self.actor(state_tensor).cpu().numpy()[0]

        # Add exploration noise
        if noise > 0:
            action += np.random.normal(0, noise, size=self.action_dim)
            action = np.clip(action, -self.max_action, self.max_action)

        return action

    def store_transition(self, state, action, reward, next_state, done):
        """Store transition."""
        self.buffer.push(state, action, reward, next_state, done)

    def train(self) -> Dict[str, float]:
        """Train TD3 agent."""
        if len(self.buffer) < self.batch_size:
            return {}

        self.total_it += 1

        # Sample batch
        states, actions, rewards, next_states, dones = self.buffer.sample(
            self.batch_size
        )

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # Update Critics
        with torch.no_grad():
            # Target policy smoothing
            noise = (torch.randn_like(actions) * self.policy_noise).clamp(
                -self.noise_clip, self.noise_clip
            )
            next_actions = (self.actor_target(next_states) + noise).clamp(
                -self.max_action, self.max_action
            )

            # Twin target Q-values
            target_q1 = self.critic1_target(next_states, next_actions)
            target_q2 = self.critic2_target(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2)
            target_q = rewards + (1 - dones) * self.gamma * target_q

        # Current Q-values
        current_q1 = self.critic1(states, actions)
        current_q2 = self.critic2(states, actions)

        # Critic losses
        critic1_loss = nn.MSELoss()(current_q1, target_q)
        critic2_loss = nn.MSELoss()(current_q2, target_q)

        # Update critics
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()

        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()

        # Delayed policy updates
        actor_loss = torch.tensor(0.0)
        if self.total_it % self.policy_freq == 0:
            # Update Actor
            actor_loss = -self.critic1(states, self.actor(states)).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Soft update targets
            self._soft_update(self.actor, self.actor_target)
            self._soft_update(self.critic1, self.critic1_target)
            self._soft_update(self.critic2, self.critic2_target)

        return {
            "critic1_loss": critic1_loss.item(),
            "critic2_loss": critic2_loss.item(),
            "actor_loss": actor_loss.item()
            if isinstance(actor_loss, torch.Tensor)
            else 0.0,
        }

    def _soft_update(self, source, target):
        """Soft update target network."""
        for param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )

    def save(self, path: str):
        """Save model."""
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic1": self.critic1.state_dict(),
                "critic2": self.critic2.state_dict(),
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "critic1_optimizer": self.critic1_optimizer.state_dict(),
                "critic2_optimizer": self.critic2_optimizer.state_dict(),
            },
            path,
        )

    def load(self, path: str):
        """Load model."""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic1.load_state_dict(checkpoint["critic1"])
        self.critic2.load_state_dict(checkpoint["critic2"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        self.critic1_optimizer.load_state_dict(checkpoint["critic1_optimizer"])
        self.critic2_optimizer.load_state_dict(checkpoint["critic2_optimizer"])

        self.actor_target = copy.deepcopy(self.actor)
        self.critic1_target = copy.deepcopy(self.critic1)
        self.critic2_target = copy.deepcopy(self.critic2)


# Factory function to create agents
def create_agent(
    algorithm: str, state_dim: int, action_dim: int, discrete: bool = True, **kwargs
):
    """
    Factory function to create DRL agents.

    Args:
        algorithm: One of ['ppo', 'dqn', 'ddpg', 'sac', 'a2c', 'td3']
        state_dim: Dimension of state space
        action_dim: Dimension of action space (n_actions for discrete)
        discrete: Whether action space is discrete
        **kwargs: Additional arguments for specific algorithm

    Returns:
        Agent instance
    """
    algorithm = algorithm.lower()

    if algorithm == "dqn":
        assert discrete, "DQN requires discrete action space"
        return DQNAgent(state_dim, action_dim, **kwargs)
    elif algorithm == "ddpg":
        assert not discrete, "DDPG requires continuous action space"
        return DDPGAgent(state_dim, action_dim, **kwargs)
    elif algorithm == "sac":
        assert not discrete, "SAC requires continuous action space"
        return SACAgent(state_dim, action_dim, **kwargs)
    elif algorithm == "a2c":
        return A2CAgent(state_dim, action_dim, discrete, **kwargs)
    elif algorithm == "td3":
        assert not discrete, "TD3 requires continuous action space"
        return TD3Agent(state_dim, action_dim, **kwargs)
    elif algorithm == "ppo":
        # Import existing PPO
        from src.agents.ppo_agent import PPOAgent, PPOConfig

        config = PPOConfig(state_dim=state_dim, n_actions=action_dim, **kwargs)
        return PPOAgent(config)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
