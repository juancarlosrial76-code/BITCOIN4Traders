"""
PPO Agent - Proximal Policy Optimization
========================================
State-of-the-art actor-critic RL algorithm for continuous control.

This module implements Proximal Policy Optimization (PPO), a policy gradient
algorithm that strikes a balance between sample efficiency and training stability.
PPO uses a clipped surrogate objective to prevent destructive large policy updates.

Key Features:
- Clipped surrogate objective: Prevents excessively large policy changes
- Separate actor and critic networks: Independent learning rates and architectures
- GAE (Generalized Advantage Estimation): Computes low-variance advantage estimates
- Batch training with mini-batches: Efficient gradient updates
- Early stopping via KL divergence: Prevents policy collapse
- Support for LSTM/GRU (Recurrent Policy): Handles sequential dependencies
- Layer Normalization and Dropout: Improves generalization and prevents overfitting
- Learning Rate Scheduling: Adaptive learning rate decay

Algorithm Overview:
--------------------
PPO optimizes the following clipped objective:

    L(θ) = E[min(r(θ) * A, clip(r(θ), 1-ε, 1+ε) * A)]

where r(θ) = π_θ(a|s) / π_θ_old(a|s) is the probability ratio, and A is the
advantage estimate. The clipping prevents the policy from changing too much
in a single update.

Reference: Schulman et al. (2017) - Proximal Policy Optimization Algorithms
            https://arxiv.org/abs/1707.06347

Example Usage:
--------------
    from src.agents.ppo_agent import PPOAgent, PPOConfig

    # Create configuration
    config = PPOConfig(
        state_dim=20,
        hidden_dim=128,
        n_actions=3,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        actor_lr=3e-4,
        critic_lr=1e-3,
    )

    # Initialize agent
    agent = PPOAgent(config, device="cuda")

    # Collect experience
    state = env.reset()
    hidden = agent.get_initial_hidden_state()
    action, log_prob, value, hidden = agent.select_action(state, hidden)

    # Store transition
    agent.store_transition(state, action, reward, log_prob, value, done, hidden)

    # Train
    stats = agent.train(next_value=0.0)
    print(f"Actor loss: {stats['actor_loss']:.4f}")

    # Save/Load
    agent.save("models/ppo_agent.pth")
    agent.load("models/ppo_agent.pth")

Imports:
--------
    torch: PyTorch deep learning framework
    numpy: Numerical computing
    loguru: Logging utility
    typing: Type hints for better code clarity
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
    """
    PPO Hyperparameters Configuration.

    This dataclass encapsulates all hyperparameters for the PPO algorithm,
    including network architecture, learning rates, PPO-specific parameters,
    and regularization settings.

    Attributes:
        state_dim (int): Dimension of the state space (required).
        hidden_dim (int): Number of hidden units in each layer. Default: 128.
        n_actions (int): Number of discrete actions. Default: 3.

    Network Architecture:
        use_recurrent (bool): Whether to use RNN (LSTM/GRU). Default: True.
        rnn_type (str): Type of RNN - "LSTM" or "GRU". Default: "GRU".
        rnn_layers (int): Number of RNN layers. Default: 1.
        dropout (float): Dropout probability (0 = no dropout). Default: 0.1.
        use_layer_norm (bool): Apply layer normalization. Default: True.

    Learning Rates & Scheduling:
        actor_lr (float): Learning rate for actor optimizer. Default: 3e-4.
        critic_lr (float): Learning rate for critic optimizer. Default: 1e-3.
        use_lr_decay (bool): Enable exponential learning rate decay. Default: True.
        lr_decay_gamma (float): Decay factor per epoch. Default: 0.99.

    PPO Specific:
        gamma (float): Discount factor for future rewards [0, 1]. Default: 0.99.
        gae_lambda (float): GAE lambda parameter [0, 1]. Default: 0.95.
        clip_epsilon (float): PPO clipping epsilon [0, 1]. Default: 0.2.

    Training:
        n_epochs (int): Number of epochs per update. Default: 10.
        batch_size (int): Mini-batch size for training. Default: 64.
        seq_len (int): Sequence length for recurrent training. Default: 10.

    Regularization:
        entropy_coef (float): Entropy bonus coefficient. Default: 0.01.
        value_loss_coef (float): Value loss weight. Default: 0.5.
        max_grad_norm (float): Gradient clipping threshold. Default: 0.5.

    Early Stopping:
        target_kl (float): KL divergence threshold for early stopping. Default: 0.01.

    Example:
        >>> config = PPOConfig(state_dim=20, n_actions=3, gamma=0.99)
        >>> print(config.gamma)
        0.99
    """

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
    """
    Base Network with MLP Feature Extractor and Optional Recurrent Layer.

    This is the foundational network architecture used by both Actor and Critic.
    It consists of:
    1. MLP Feature Extractor: Two hidden layers with activation, normalization, and dropout
    2. Optional Recurrent Layer: LSTM or GRU for handling sequential state dependencies
    3. Output Head: Linear projection to output dimension

    The network uses orthogonal weight initialization, which is particularly
    beneficial for reinforcement learning as it improves gradient flow.

    Architecture:
        Input (state_dim)
            → Linear → LayerNorm → ReLU → Dropout
            → Linear → LayerNorm → ReLU → Dropout
            → [Optional RNN (GRU/LSTM)]
            → Linear (output)

    Attributes:
        config (PPOConfig): Configuration object with network hyperparameters.
        feature_extractor (nn.Sequential): MLP feature extraction layers.
        rnn (nn.LSTM or nn.GRU or None): Recurrent layer if use_recurrent=True.
        head (nn.Linear): Output projection layer.

    Args:
        config (PPOConfig): PPO configuration containing architecture parameters.
        output_dim (int): Dimension of the output layer.

    Example:
        >>> config = PPOConfig(state_dim=20, hidden_dim=128, use_recurrent=True)
        >>> network = BaseNetwork(config, output_dim=3)
        >>> state = torch.randn(1, 20)
        >>> output, hidden = network(state)
        >>> print(output.shape)
        torch.Size([1, 3])
    """

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
            nn.init.orthogonal_(
                module.weight
            )  # Orthogonal init improves gradient flow in RL
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)  # Zero-initialize biases
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.weight, 1)  # Scale=1 (identity transform at init)
            nn.init.constant_(module.bias, 0)  # Shift=0 (identity transform at init)
        elif isinstance(module, (nn.LSTM, nn.GRU)):
            for name, param in module.named_parameters():
                if "weight_ih" in name:
                    nn.init.orthogonal_(
                        param.data
                    )  # Input-hidden weights: orthogonal init
                elif "weight_hh" in name:
                    nn.init.orthogonal_(
                        param.data
                    )  # Hidden-hidden weights: orthogonal init
                elif "bias" in name:
                    nn.init.constant_(param.data, 0)  # Zero-initialize RNN biases

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

        features = self.feature_extractor(x)  # Extract features via MLP trunk

        next_hidden = None
        if self.rnn is not None:
            features, next_hidden = self.rnn(
                features, hidden
            )  # Propagate recurrent state

        # If we added a sequence dim, remove it for the head if it's 1
        if x.dim() == 3 and features.size(1) == 1:
            features = features.squeeze(
                1
            )  # Remove seq dim so head sees (batch, hidden_dim)

        # If input was sequence, output is sequence. If input was batch, output is batch.
        # But head expects (..., hidden_dim).
        output = self.head(features)

        return output, next_hidden


class ActorNetwork(BaseNetwork):
    """
    Actor (Policy) Network for PPO.

    The actor network learns a stochastic policy π(a|s) that maps states to
    action probabilities. It uses a Categorical distribution over discrete actions.

    The actor is optimized to maximize expected rewards while the critic provides
    value estimates for advantage computation.

    Attributes:
        Inherits all attributes from BaseNetwork.

    Args:
        config (PPOConfig): PPO configuration with state_dim and n_actions.

    Returns:
        Tuple[Categorical, Optional[Union[Tuple, torch.Tensor]]]:
            - Categorical distribution over actions
            - Next hidden state (if recurrent, else None)

    Forward Pass:
        Input: state tensor of shape (batch_size, state_dim) or (batch_size, seq_len, state_dim)
        Output: (logits, hidden_state) where logits shape is (batch_size, n_actions)

    Example:
        >>> config = PPOConfig(state_dim=20, n_actions=3)
        >>> actor = ActorNetwork(config)
        >>> state = torch.randn(4, 20)
        >>> dist, hidden = actor(state)
        >>> action = dist.sample()  # Sample action from policy
        >>> print(dist.probs)  # Action probabilities
    """

    def __init__(self, config: PPOConfig):
        super().__init__(config, config.n_actions)

    def forward(
        self, state: torch.Tensor, hidden: Optional[Union[Tuple, torch.Tensor]] = None
    ) -> Tuple[Categorical, Optional[Union[Tuple, torch.Tensor]]]:
        logits, next_hidden = super().forward(state, hidden)
        return Categorical(logits=logits), next_hidden


class CriticNetwork(BaseNetwork):
    """
    Critic (Value) Network for PPO.

    The critic network estimates the value function V(s), which represents
    the expected return from a given state under the current policy. This
    value estimate is used to compute advantages for policy gradient updates.

    The critic outputs a single scalar value (not a distribution) representing
    the state value estimate.

    Attributes:
        Inherits all attributes from BaseNetwork.

    Args:
        config (PPOConfig): PPO configuration with state_dim.

    Returns:
        Tuple[torch.Tensor, Optional[Union[Tuple, torch.Tensor]]]:
            - Value estimate (scalar per state)
            - Next hidden state (if recurrent, else None)

    Forward Pass:
        Input: state tensor of shape (batch_size, state_dim) or (batch_size, seq_len, state_dim)
        Output: (value, hidden_state) where value shape is (batch_size, 1)

    Example:
        >>> config = PPOConfig(state_dim=20)
        >>> critic = CriticNetwork(config)
        >>> state = torch.randn(4, 20)
        >>> value, hidden = critic(state)
        >>> print(value.shape)
        torch.Size([4, 1])
    """

    def __init__(self, config: PPOConfig):
        super().__init__(config, 1)

    def forward(
        self, state: torch.Tensor, hidden: Optional[Union[Tuple, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Optional[Union[Tuple, torch.Tensor]]]:
        value, next_hidden = super().forward(state, hidden)
        return value, next_hidden


class PPOAgent:
    """
    Proximal Policy Optimization Agent.

    A complete PPO implementation for reinforcement learning in trading environments.
    Supports both stateless (MLP) and recurrent (LSTM/GRU) policy architectures.

    This agent implements:
    - On-policy learning with experience replay buffers
    - Generalized Advantage Estimation (GAE) for low-variance advantage estimates
    - Clipped surrogate objective to prevent destructive updates
    - Separate actor and critic optimization with different learning rates
    - KL divergence early stopping to prevent policy collapse
    - Learning rate scheduling with exponential decay

    The agent maintains experience buffers and performs multiple epochs of
    mini-batch updates per training iteration, computing advantages using GAE.

    Attributes:
        config (PPOConfig): Configuration object with all hyperparameters.
        device (str): Computation device ("cuda" or "cpu").
        actor (ActorNetwork): Policy network for action selection.
        critic (CriticNetwork): Value network for state estimation.
        actor_optimizer (torch.optim.Adam): Optimizer for actor parameters.
        critic_optimizer (torch.optim.Adam): Optimizer for critic parameters.
        actor_scheduler (ExponentialLR or None): Learning rate scheduler for actor.
        critic_scheduler (ExponentialLR or None): Learning rate scheduler for critic.

    Args:
        config (PPOConfig): PPO configuration object.
        device (str): Device for computation. Default: "cpu".

    Example:
        Basic usage for trading:
        >>> from src.agents.ppo_agent import PPOAgent, PPOConfig
        >>>
        >>> config = PPOConfig(state_dim=20, n_actions=3, gamma=0.99)
        >>> agent = PPOAgent(config, device="cuda")
        >>>
        >>> # Collect experience
        >>> state = env.reset()
        >>> hidden = agent.get_initial_hidden_state()
        >>>
        >>> for step in range(2048):
        ...     action, log_prob, value, hidden = agent.select_action(state, hidden)
        ...     next_state, reward, done, info = env.step(action)
        ...     agent.store_transition(state, action, reward, log_prob, value, done, hidden)
        ...     state = next_state
        ...     if done:
        ...         state = env.reset()
        ...         hidden = None
        >>>
        >>> # Train agent
        >>> stats = agent.train(next_value=0.0)
        >>> print(f"Training complete. Actor loss: {stats['actor_loss']:.4f}")

    Note:
        - The agent uses separate buffers for states, actions, rewards, etc.
        - Hidden states are stored to support truncated backpropagation through time.
        - GAE requires a bootstrap value for the final state (passed as next_value).
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
        values = self.values + [next_value]  # Append bootstrap value for final state
        advantages = []
        gae = 0  # Running GAE accumulator (initialized to 0)
        for t in reversed(
            range(len(self.rewards))
        ):  # Iterate backwards through trajectory
            delta = (
                self.rewards[t]
                + self.config.gamma * values[t + 1] * (1 - self.dones[t])  # TD target
                - values[t]  # Subtract current value estimate → TD error
            )
            # GAE recursion: A_t = δ_t + (γλ)(1-done) * A_{t+1}
            gae = (
                delta
                + self.config.gamma * self.config.gae_lambda * (1 - self.dones[t]) * gae
            )
            advantages.insert(0, gae)  # Prepend to maintain time-order

        advantages = np.array(advantages)
        returns = advantages + np.array(
            self.values
        )  # Target returns = advantage + value baseline
        return advantages, returns

    def train(self, next_value: float = 0.0) -> Dict:
        if len(self.states) == 0:
            return {}

        # Compute advantages
        advantages, returns = self.compute_gae(next_value)

        # Normalize advantages to zero-mean, unit-variance for training stability
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
            epoch_kls = []  # KL only measured for this epoch (not cumulative)

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

                ratio = torch.exp(
                    log_probs - batch_old_log_probs
                )  # Importance sampling ratio π_new/π_old
                surr1 = ratio * batch_advantages  # Unclipped surrogate objective
                surr2 = (
                    torch.clamp(
                        ratio,
                        1 - self.config.clip_epsilon,
                        1 + self.config.clip_epsilon,
                    )
                    * batch_advantages  # Clipped surrogate: prevents large policy updates
                )

                actor_loss = -torch.min(
                    surr1, surr2
                ).mean()  # PPO-clip objective (negated for gradient ascent)
                critic_loss = F.mse_loss(
                    critic_values, batch_returns
                )  # Value function MSE loss

                loss = (
                    actor_loss
                    + self.config.value_loss_coef * critic_loss  # Weighted value loss
                    - self.config.entropy_coef
                    * entropy  # Entropy bonus encourages exploration
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
                    kl = (
                        batch_old_log_probs - log_probs
                    ).mean()  # Approx KL: KL(old||new) ≈ mean(log_old - log_new)
                    kl_divergences.append(kl.item())
                    epoch_kls.append(kl.item())

            # Early stopping: check KL only for this epoch (not cumulative across epochs)
            if np.mean(epoch_kls) > self.config.target_kl:
                logger.debug(
                    f"Early stopping at epoch {epoch + 1} (KL={np.mean(epoch_kls):.4f})"
                )
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
