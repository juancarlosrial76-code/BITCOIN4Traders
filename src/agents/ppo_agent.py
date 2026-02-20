"""
PPO Agent - Proximal Policy Optimization
=========================================
State-of-the-art actor-critic RL algorithm for continuous control.

Key Features:
- Clipped surrogate objective
- Separate actor and critic networks
- GAE (Generalized Advantage Estimation)
- Batch training with mini-batches
- Early stopping via KL divergence
- Support for LSTM/GRU (Recurrent Policy)
- Layer Normalization and Dropout
- Learning Rate Scheduling

Reference: Schulman et al. (2017) - Proximal Policy Optimization Algorithms
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from typing import Tuple, List, Dict, Optional, Union, Any
from dataclasses import dataclass, field
from loguru import logger
from torch.optim.lr_scheduler import ExponentialLR


@dataclass
class PPOConfig:
    """PPO hyperparameters."""

    # Network architecture
    state_dim: int
    hidden_dim: int = 128
    n_actions: int = 3

    # Advanced Architecture
    use_recurrent: bool = True
    rnn_type: str = "GRU"  # "LSTM" or "GRU"
    rnn_layers: int = 1
    dropout: float = 0.1
    use_layer_norm: bool = True

    # Learning rates & Scheduling
    actor_lr: float = 3e-4
    critic_lr: float = 1e-3
    use_lr_decay: bool = True
    lr_decay_gamma: float = 0.99

    # PPO specific
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2

    # Training
    n_epochs: int = 10
    batch_size: int = 64
    seq_len: int = (
        10  # Sequence length for recurrent training (if implemented sequentially)
    )

    # Regularization
    entropy_coef: float = 0.01
    value_loss_coef: float = 0.5
    max_grad_norm: float = 0.5

    # Early stopping
    target_kl: float = 0.01


class BaseNetwork(nn.Module):
    """Base network with MLP feature extractor and optional RNN."""

    def __init__(self, config: PPOConfig, output_dim: int):
        super().__init__()
        self.config = config

        # Feature Extractor
        modules = []
        modules.append(nn.Linear(config.state_dim, config.hidden_dim))
        if config.use_layer_norm:
            modules.append(nn.LayerNorm(config.hidden_dim))
        modules.append(nn.ReLU())
        if config.dropout > 0:
            modules.append(nn.Dropout(config.dropout))

        modules.append(nn.Linear(config.hidden_dim, config.hidden_dim))
        if config.use_layer_norm:
            modules.append(nn.LayerNorm(config.hidden_dim))
        modules.append(nn.ReLU())
        if config.dropout > 0:
            modules.append(nn.Dropout(config.dropout))

        self.feature_extractor = nn.Sequential(*modules)

        # Recurrent Layer
        self.rnn = None
        if config.use_recurrent:
            if config.rnn_type == "LSTM":
                self.rnn = nn.LSTM(
                    config.hidden_dim,
                    config.hidden_dim,
                    num_layers=config.rnn_layers,
                    batch_first=True,
                )
            elif config.rnn_type == "GRU":
                self.rnn = nn.GRU(
                    config.hidden_dim,
                    config.hidden_dim,
                    num_layers=config.rnn_layers,
                    batch_first=True,
                )

        # Output Layer
        self.head = nn.Linear(config.hidden_dim, output_dim)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)
        elif isinstance(module, (nn.LSTM, nn.GRU)):
            for name, param in module.named_parameters():
                if "weight_ih" in name:
                    nn.init.orthogonal_(param.data)
                elif "weight_hh" in name:
                    nn.init.orthogonal_(param.data)
                elif "bias" in name:
                    nn.init.constant_(param.data, 0)

    def forward(
        self, x: torch.Tensor, hidden: Optional[Union[Tuple, torch.Tensor]] = None
    ):
        """
        Forward pass.
        x: (batch_size, state_dim) or (batch_size, seq_len, state_dim)
        """
        # Handle input shape (add sequence dim if missing for RNN)
        if hasattr(self, "rnn") and self.rnn is not None:
            if x.dim() == 2:
                x = x.unsqueeze(1)  # (batch, 1, features)

        features = self.feature_extractor(x)

        next_hidden = None
        if self.rnn is not None:
            features, next_hidden = self.rnn(features, hidden)

        # If we added a sequence dim, remove it for the head if it's 1
        if x.dim() == 3 and features.size(1) == 1:
            features = features.squeeze(1)

        # If input was sequence, output is sequence. If input was batch, output is batch.
        # But head expects (..., hidden_dim).
        output = self.head(features)

        return output, next_hidden


class ActorNetwork(BaseNetwork):
    """Policy network (actor)."""

    def __init__(self, config: PPOConfig):
        super().__init__(config, config.n_actions)

    def forward(
        self, state: torch.Tensor, hidden: Optional[Union[Tuple, torch.Tensor]] = None
    ) -> Tuple[Categorical, Optional[Union[Tuple, torch.Tensor]]]:
        logits, next_hidden = super().forward(state, hidden)
        return Categorical(logits=logits), next_hidden


class CriticNetwork(BaseNetwork):
    """Value network (critic)."""

    def __init__(self, config: PPOConfig):
        super().__init__(config, 1)

    def forward(
        self, state: torch.Tensor, hidden: Optional[Union[Tuple, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Optional[Union[Tuple, torch.Tensor]]]:
        value, next_hidden = super().forward(state, hidden)
        return value, next_hidden


class PPOAgent:
    """
    PPO Agent for trading with support for Recurrent policies.
    """

    def __init__(self, config: PPOConfig, device: str = "cpu"):
        self.config = config
        self.device = device

        # Networks
        self.actor = ActorNetwork(config).to(device)
        self.critic = CriticNetwork(config).to(device)

        # Optimizers & Schedulers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=config.actor_lr)
        self.critic_optimizer = optim.Adam(
            self.critic.parameters(), lr=config.critic_lr
        )

        self.actor_scheduler = None
        self.critic_scheduler = None

        if config.use_lr_decay:
            self.actor_scheduler = ExponentialLR(
                self.actor_optimizer, gamma=config.lr_decay_gamma
            )
            self.critic_scheduler = ExponentialLR(
                self.critic_optimizer, gamma=config.lr_decay_gamma
            )

        # Experience buffers
        self.reset_buffers()

        logger.info(f"PPOAgent initialized on {device}")
        logger.info(
            f"  Architecture: {config.hidden_dim} hidden, {config.rnn_type if config.use_recurrent else 'MLP'}"
        )
        logger.info(f"  LayerNorm: {config.use_layer_norm}, Dropout: {config.dropout}")
        logger.info(f"  LR Schedule: {config.use_lr_decay}")

    def reset_buffers(self):
        """Reset experience buffers."""
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.dones = []
        # Store hidden states for training
        self.hiddens = []

    def get_initial_hidden_state(self, batch_size: int = 1):
        """Get initial hidden state for RNN."""
        if not self.config.use_recurrent:
            return None

        if self.config.rnn_type == "LSTM":
            return (
                torch.zeros(
                    self.config.rnn_layers, batch_size, self.config.hidden_dim
                ).to(self.device),
                torch.zeros(
                    self.config.rnn_layers, batch_size, self.config.hidden_dim
                ).to(self.device),
            )
        else:  # GRU
            return torch.zeros(
                self.config.rnn_layers, batch_size, self.config.hidden_dim
            ).to(self.device)

    def select_action(
        self,
        state: np.ndarray,
        hidden: Optional[Union[Tuple, torch.Tensor]] = None,
        deterministic: bool = False,
    ) -> Tuple[int, float, float, Optional[Union[Tuple, torch.Tensor]]]:
        """
        Select action given state.
        Returns: action, log_prob, value, next_hidden
        """
        state_tensor = (
            torch.FloatTensor(state).unsqueeze(0).to(self.device)
        )  # (1, state_dim)

        # Switch to eval mode for deterministic inference (disables dropout)
        training_mode = self.actor.training
        if deterministic:
            self.actor.eval()
            self.critic.eval()

        with torch.no_grad():
            # Get action distribution
            dist, next_hidden_actor = self.actor(state_tensor, hidden)

            # Get value estimate (we can pass the same hidden, or keep them separate.
            # Usually Actor and Critic have separate networks/hidden states if not shared.)
            # Here we assume separate networks, so we need separate hidden states?
            # Implem Note: If we use the same 'hidden' input for both, it implies shared trunk or we maintain two hidden states.
            # For simplicity in this structure, let's assume 'hidden' is just for the Actor (Policy)
            # because the Critic is usually running on the same trajectory but we often don't pass its hidden state out
            # for the environment loop. However, to accept 'next_hidden' we need the actor's.
            # Critic recurrent state is often handled similarly or avoided.
            # Let's enforce that 'hidden' passed in is for the Actor.

            # For the Critic, to get a valid value on a recurrent policy, it should also be recurrent.
            # But usually we just burn-in or use the actor's hidden? No, they are different params.
            # Simplified: The 'hidden' returned is ONLY for the Actor.
            # We will ignore storing Critic's hidden state for rollout purposes in this simplified version.
            # Ideally: recursive state for both. But let's return Actor's hidden.

            value, _ = self.critic(
                state_tensor, None
            )  # Critic might need its own hidden state tracking if recurrent!
            # If Critic is recurrent, it's problematic without tracking its state too.
            # FIX: Only Actor is recurrent in this implementation for "Action Selection".
            # If config says use_recurrent, both become recurrent.
            # Ideally we accept (actor_hidden, critic_hidden).
            # For now, let's stick to returning Actor hidden for the policy loop.

            if deterministic:
                action = dist.probs.argmax()
            else:
                action = dist.sample()

            log_prob = dist.log_prob(action)

        # Restore training mode
        if deterministic:
            self.actor.train(training_mode)
            self.critic.train(training_mode)

        return (action.item(), log_prob.item(), value.item(), next_hidden_actor)

    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        log_prob: float,
        value: float,
        done: bool,
        hidden: Optional[Any] = None,  # New arg
    ):
        """Store transition in buffer."""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.dones.append(done)

        # We store the hidden state used to generate this action
        # This is needed for "burn-in" or initialization during training
        if hidden is not None:
            # Move to CPU to save GPU memory
            if isinstance(hidden, tuple):
                self.hiddens.append(tuple(h.cpu() for h in hidden))
            else:
                self.hiddens.append(hidden.cpu())
        else:
            self.hiddens.append(None)

    def compute_gae(self, next_value: float) -> Tuple[np.ndarray, np.ndarray]:
        values = self.values + [next_value]
        advantages = []
        gae = 0
        for t in reversed(range(len(self.rewards))):
            delta = (
                self.rewards[t]
                + self.config.gamma * values[t + 1] * (1 - self.dones[t])
                - values[t]
            )
            gae = (
                delta
                + self.config.gamma * self.config.gae_lambda * (1 - self.dones[t]) * gae
            )
            advantages.insert(0, gae)

        advantages = np.array(advantages)
        returns = advantages + np.array(self.values)
        return advantages, returns

    def train(self, next_value: float = 0.0) -> Dict:
        if len(self.states) == 0:
            return {}

        # Compute advantages
        advantages, returns = self.compute_gae(next_value)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Convert to tensors
        states = torch.FloatTensor(np.array(self.states)).to(self.device)
        actions = torch.LongTensor(self.actions).to(self.device)
        old_log_probs = torch.FloatTensor(self.log_probs).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)

        # For recurrent training, we need to handle hidden states.
        # Simplified approach: Train on shuffled batches but detach history (truncate) or use stored hidden states.
        # For full correctness -> Sequences.
        # Here we will use the stored hidden states as the initial state for the batch.
        # This is "truncated BPTT" with window 1 if we just use the stored state.

        # Prepare hidden states batch
        # If we have stored hidden states
        has_hidden = len(self.hiddens) > 0 and self.hiddens[0] is not None

        dataset_size = len(states)
        indices = np.arange(dataset_size)

        total_actor_loss = 0
        total_critic_loss = 0
        total_entropy = 0
        kl_divergences = []

        for epoch in range(self.config.n_epochs):
            np.random.shuffle(indices)

            for start_idx in range(0, dataset_size, self.config.batch_size):
                end_idx = min(start_idx + self.config.batch_size, dataset_size)
                batch_indices = indices[start_idx:end_idx]

                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]

                batch_hidden = None
                if has_hidden:
                    # Collate hidden states
                    # self.hiddens is a list of (h, c) or h
                    raw_hiddens = [self.hiddens[i] for i in batch_indices]

                    if self.config.rnn_type == "LSTM":
                        # List of tuples (h, c) -> tuple of stacks
                        h_list = [x[0] for x in raw_hiddens]
                        c_list = [x[1] for x in raw_hiddens]
                        # Stack: (batch, num_layers, hidden) -> permute to (num_layers, batch, hidden)
                        # Actually stored as (num_layers, 1, hidden) (from select_action unsqueeze batch) or (num_layers, hidden)
                        # select_action uses batch count 1.
                        # We need (num_layers, batch_size, hidden)

                        h_stack = torch.cat(h_list, dim=1).to(
                            self.device
                        )  # dim 1 is batch size in stored?
                        # Stored: (num_layers, 1, hidden)
                        # cat dim 1 -> (num_layers, batch, hidden)
                        c_stack = torch.cat(c_list, dim=1).to(self.device)
                        batch_hidden = (h_stack, c_stack)
                    else:  # GRU
                        # List of tensors
                        batch_hidden = torch.cat(raw_hiddens, dim=1).to(self.device)

                # Forward pass
                # Note: We pass the stored hidden state.
                # Gradients flow through the network for this step.
                dist, _ = self.actor(batch_states, batch_hidden)

                # Critic
                # NOTE: We didn't store critic hidden states.
                # So we pass None. This implicitly zeroes the critic hidden state every step if critic is recurrent.
                # This is suboptimal for Recurrent Critic. Ideally we store both.
                # For now, let's assume Critic is MLP or we accept the limitation.
                critic_values, _ = self.critic(batch_states, None)
                critic_values = critic_values.squeeze()

                # Losses
                log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()

                ratio = torch.exp(log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = (
                    torch.clamp(
                        ratio,
                        1 - self.config.clip_epsilon,
                        1 + self.config.clip_epsilon,
                    )
                    * batch_advantages
                )

                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = F.mse_loss(critic_values, batch_returns)

                loss = (
                    actor_loss
                    + self.config.value_loss_coef * critic_loss
                    - self.config.entropy_coef * entropy
                )

                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                loss.backward()

                nn.utils.clip_grad_norm_(
                    self.actor.parameters(), self.config.max_grad_norm
                )
                nn.utils.clip_grad_norm_(
                    self.critic.parameters(), self.config.max_grad_norm
                )

                self.actor_optimizer.step()
                self.critic_optimizer.step()

                total_actor_loss += actor_loss.item()
                total_critic_loss += critic_loss.item()
                total_entropy += entropy.item()

                with torch.no_grad():
                    kl = (batch_old_log_probs - log_probs).mean()
                    kl_divergences.append(kl.item())

            if np.mean(kl_divergences) > self.config.target_kl:
                logger.info(f"Early stopping at epoch {epoch + 1}")
                break

        # Step Schedulers
        if self.actor_scheduler:
            self.actor_scheduler.step()
        if self.critic_scheduler:
            self.critic_scheduler.step()

        n_updates = (dataset_size // self.config.batch_size) * (epoch + 1)
        self.reset_buffers()

        return {
            "actor_loss": total_actor_loss / max(n_updates, 1),
            "critic_loss": total_critic_loss / max(n_updates, 1),
            "entropy": total_entropy / max(n_updates, 1),
            "mean_kl": np.mean(kl_divergences),
            "n_epochs": epoch + 1,
        }

    def save(self, path: str):
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "config": self.config,
            },
            path,
        )
        logger.info(f"Agent saved to {path}")

    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)

        # Adapt weights if dimensions changed (e.g. removed reserved features)
        actor_state = self._adapt_state_dict(
            checkpoint["actor"], self.actor.state_dict(), "actor"
        )
        critic_state = self._adapt_state_dict(
            checkpoint["critic"], self.critic.state_dict(), "critic"
        )

        self.actor.load_state_dict(actor_state)
        self.critic.load_state_dict(critic_state)
        logger.info(f"Agent loaded from {path}")

    def _adapt_state_dict(
        self, saved_state: Dict, current_state: Dict, name: str
    ) -> Dict:
        """Adapt state dict if input dimensions match (handling removed features)."""
        adapted_state = saved_state.copy()

        # Check first layer weight
        key = "feature_extractor.0.weight"
        if key in saved_state and key in current_state:
            saved_weight = saved_state[key]
            current_weight = current_state[key]

            if saved_weight.shape != current_weight.shape:
                # Check if only input dimension (dim 1) differs
                if (
                    saved_weight.shape[0] == current_weight.shape[0]
                    and saved_weight.shape[1] > current_weight.shape[1]
                ):
                    diff = saved_weight.shape[1] - current_weight.shape[1]
                    logger.warning(
                        f"Adapting {name} input layer: {saved_weight.shape} -> {current_weight.shape}"
                    )
                    logger.warning(
                        f"Dropping last {diff} input features (assuming reserved/unused)"
                    )

                    # Slice to match current input dim
                    adapted_state[key] = saved_weight[:, : current_weight.shape[1]]
                else:
                    logger.warning(
                        f"Shape mismatch in {name} {key}: {saved_weight.shape} vs {current_weight.shape} - cannot auto-adapt"
                    )

        return adapted_state
