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

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from loguru import logger
from torch.distributions import Categorical
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.networks.transformer_net import TransformerBackbone


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
    use_transformer: bool = False  # If True, overrides RNN with TransformerBackbone
    rnn_layers: int = 1
    dropout: float = 0.1
    use_layer_norm: bool = True

    # Transformer-specific knobs (only used when use_transformer=True)
    transformer_nhead: int = 4  # attention heads — must divide transformer_d_model
    transformer_d_model: int = 0  # 0 = use hidden_dim (most common); set >0 to decouple
    transformer_seq_len: int = 1  # sliding-window depth fed to transformer:
    #   1  → Option A: each step is seq_len=1 (backbone sees only current features)
    #   >1 → Option B: the last `transformer_seq_len` states are stacked and fed as a
    #                  real sequence so the transformer attends across time

    # Learning rates & Scheduling
    actor_lr: float = 3e-4
    critic_lr: float = 1e-3
    use_lr_decay: bool = True
    lr_decay_gamma: float = 0.99       # kept for backwards-compat (unused)
    lr_decay_steps: int = 3000         # CosineAnnealingLR T_max (full cycle)

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

    # Self-Imitation Learning (SIL)
    use_sil: bool = True
    sil_update_ratio: int = 4  # Number of SIL updates per PPO update
    sil_batch_size: int = 512
    sil_capacity: int = 100000
    sil_value_weight: float = 0.01

    # GPU optimizations (only active on CUDA, ignored on CPU)
    use_amp: bool = True  # Mixed Precision (float16 forward, float32 grads)
    use_compile: bool = False  # torch.compile() — PyTorch 2.x, ~20-40% faster
    # Disabled by default (incompatible with GRU on
    # older CUDA; enable when PyTorch >= 2.1)

    # ── Feature 5: Dual-Head Actor ────────────────────────────────────────────
    # If True: Actor has two heads sharing the same GRU backbone:
    #   Head 1 (direction_head): 3 logits  → {Short, Neutral, Long}
    #   Head 2 (sizing_head):    3 logits  → {Full, Half, Quarter}
    # Together they span a 3×3=9 action space, mapped to the
    # original 7 actions (some combinations merged).
    # Advantage: The agent learns direction and size separately → better generalization.
    # If False: classic single-head actor (n_actions logits).
    use_dual_head: bool = False  # Default off; enable via PPOConfig(use_dual_head=True)


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

        # Recurrent or Transformer Layer
        self.rnn = None
        self.transformer = None

        if getattr(config, "use_transformer", False):
            # Resolve d_model: use explicit transformer_d_model if set, else hidden_dim
            _d_model = (
                config.transformer_d_model
                if getattr(config, "transformer_d_model", 0) > 0
                else config.hidden_dim
            )
            # If d_model != hidden_dim the MLP trunk output needs a projection;
            # TransformerBackbone handles that via its own input_projection layer.
            self.transformer = TransformerBackbone(
                input_dim=config.hidden_dim,
                d_model=_d_model,
                nhead=getattr(config, "transformer_nhead", 4),
                num_layers=config.rnn_layers,
                dropout=config.dropout,
            )
            # Store resolved d_model so forward() can reference it if needed
            self._transformer_d_model = _d_model
        elif config.use_recurrent:
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

        # Output Layer — input dim depends on backbone:
        #   Transformer with custom d_model → d_model (may differ from hidden_dim)
        #   GRU / LSTM / Transformer with default d_model → hidden_dim
        _head_in = getattr(self, "_transformer_d_model", config.hidden_dim)
        self.head = nn.Linear(_head_in, output_dim)

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
        elif isinstance(module, nn.TransformerEncoderLayer):
            for name, param in module.named_parameters():
                if param.dim() > 1:
                    nn.init.xavier_uniform_(param)

    def forward(
        self, x: torch.Tensor, hidden: Optional[Union[Tuple, torch.Tensor]] = None
    ):
        """
        Forward pass.
        x: (batch_size, state_dim) or (batch_size, seq_len, state_dim)
        """
        # NaN-Guard on Input: prevents NaN propagation through LayerNorm/GRU.
        # Cause: some envs can produce NaN observations (e.g., empty
        # price history at episode start). A NaN in GRU input infects the
        # entire hidden state and all subsequent steps.
        x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)

        # NaN-Guard on Hidden State: if a previous step produced NaN
        # (e.g., due to gradient explosion), the hidden state also gets
        # infected. Here we sanitize it before it goes into GRU.
        if hidden is not None:
            if isinstance(hidden, tuple):
                # LSTM: (h, c) sanitize both
                hidden = tuple(
                    torch.nan_to_num(h, nan=0.0, posinf=1.0, neginf=-1.0)
                    for h in hidden
                )
            else:
                hidden = torch.nan_to_num(hidden, nan=0.0, posinf=1.0, neginf=-1.0)

        # Handle input shape (add sequence dim if missing for RNN)
        if hasattr(self, "rnn") and self.rnn is not None:
            if x.dim() == 2:
                x = x.unsqueeze(1)  # (batch, 1, features)

        features = self.feature_extractor(x)  # Extract features via MLP trunk

        next_hidden = None
        if self.transformer is not None:
            # Transformer expects sequence [batch, seq_len, hidden_dim]
            if features.dim() == 2:
                features = features.unsqueeze(1)
            features = self.transformer(features)  # output is [batch, hidden_dim]
        elif self.rnn is not None:
            features, next_hidden = self.rnn(
                features, hidden
            )  # Propagate recurrent state

        # If we added a sequence dim, remove it for the head if it's 1
        if x.dim() == 3 and features.dim() == 3 and features.size(1) == 1:
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
        # Cast to float32: AMP autocast produces float16 logits, Categorical
        # validate_args then fails (IndependentConstraint). The float32 cast
        # fixes this without losing AMP performance benefits.
        logits = logits.float()
        # Final NaN-Guard: if despite input sanitization NaNs emerge from LayerNorm
        # or GRU (e.g., gradient explosion), replace with 0.0
        # (uniform distribution over all actions — safe fallback).
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            logits = torch.nan_to_num(logits, nan=0.0, posinf=10.0, neginf=-10.0)
        return Categorical(logits=logits), next_hidden


class DualHeadActorNetwork(nn.Module):
    """
    Feature 5: Dual-Head Actor Network.

    Splits the action space into two orthogonal decisions:
      - Direction Head (3 logits): Short | Neutral | Long
      - Sizing Head   (3 logits): Full (100%) | Half (50%) | Quarter (33%)

    Both heads share the same GRU backbone (feature_extractor + rnn).
    The combination results in a 3x3=9 combined action space, which is
    mapped via DIRECTION_SIZE_TO_ACTION to the 7 original actions.

    Mapping:
        Short+Full    -> Action 0  (Short 100%)
        Short+Half    -> Action 1  (Short 50%)
        Short+Quarter -> Action 1  (Short 50%, merge)
        Neutral+*     -> Action 2  (Neutral, ignore size)
        Long+Quarter  -> Action 3  (Long 33%)
        Long+Half     -> Action 4  (Long 50%)
        Long+Full     -> Action 6  (Long 100%)
        Long+*        -> Action 5  (Long 75%, catch-all)

    Advantage over Single-Head:
    - Agent learns direction and size SEPARATELY -> more training signal
    - Generalizes better: e.g., "Long learned" + "Full learned" = immediately usable
    - Fewer actions per head -> faster convergence
    """

    # Mapping: (direction_idx, size_idx) → original action (0-6)
    # direction: 0=Short, 1=Neutral, 2=Long
    # size:      0=Full,  1=Half,    2=Quarter
    DIRECTION_SIZE_TO_ACTION = {
        (0, 0): 0,  # Short+Full     → Action 0
        (0, 1): 1,  # Short+Half     → Action 1
        (0, 2): 1,  # Short+Quarter  → Action 1 (merge)
        (1, 0): 2,  # Neutral+Full   → Action 2
        (1, 1): 2,  # Neutral+Half   → Action 2
        (1, 2): 2,  # Neutral+Quarter→ Action 2
        (2, 0): 6,  # Long+Full      → Action 6
        (2, 1): 4,  # Long+Half      → Action 4
        (2, 2): 3,  # Long+Quarter   → Action 3
    }

    def __init__(self, config: "PPOConfig"):
        super().__init__()
        self.config = config

        # Shared backbone (identical to BaseNetwork without head)
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

        # Shared recurrent layer
        self.rnn = None
        if config.use_recurrent:
            self.rnn = nn.GRU(
                config.hidden_dim,
                config.hidden_dim,
                num_layers=config.rnn_layers,
                batch_first=True,
            )

        # Two separate output heads
        self.direction_head = nn.Linear(config.hidden_dim, 3)  # Short/Neutral/Long
        self.sizing_head = nn.Linear(config.hidden_dim, 3)  # Full/Half/Quarter

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.GRU):
            for name, param in module.named_parameters():
                if "weight" in name:
                    nn.init.orthogonal_(param.data)
                elif "bias" in name:
                    nn.init.constant_(param.data, 0)

    def forward(
        self,
        state: torch.Tensor,
        hidden: Optional[Union[Tuple, torch.Tensor]] = None,
    ) -> Tuple[Categorical, Optional[torch.Tensor]]:
        """
        Forward pass returning a Categorical over the 7 original actions.

        Internally samples direction and sizing independently, then maps
        the joint probability to the original 7-action space.
        """
        x = state
        if self.rnn is not None and x.dim() == 2:
            x = x.unsqueeze(1)

        features = self.feature_extractor(x)

        next_hidden = None
        if self.rnn is not None:
            features, next_hidden = self.rnn(features, hidden)
            if features.size(1) == 1:
                features = features.squeeze(1)

        dir_logits = self.direction_head(features).float()  # (B, 3)
        siz_logits = self.sizing_head(features).float()  # (B, 3)

        dir_probs = torch.softmax(dir_logits, dim=-1)  # (B, 3)
        siz_probs = torch.softmax(siz_logits, dim=-1)  # (B, 3)

        # Joint probability: outer product → (B, 3, 3) → flatten → (B, 9)
        joint = torch.bmm(dir_probs.unsqueeze(2), siz_probs.unsqueeze(1)).view(
            dir_probs.size(0), 9
        )

        # Map joint 9 → original 7 actions by summing probabilities
        n_actions = 7
        device = joint.device
        action_probs = torch.zeros(joint.size(0), n_actions, device=device)
        for (d_idx, s_idx), a_idx in self.DIRECTION_SIZE_TO_ACTION.items():
            joint_idx = d_idx * 3 + s_idx
            action_probs[:, a_idx] += joint[:, joint_idx]

        # Add tiny epsilon to prevent log(0) in entropy calculation
        action_probs = action_probs + 1e-8
        action_probs = action_probs / action_probs.sum(dim=-1, keepdim=True)

        return Categorical(probs=action_probs), next_hidden


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

        # ── Feature 5: Dual-Head Actor ─────────────────────────────────────────
        # If use_dual_head=True: DualHeadActorNetwork (Direction x Sizing)
        # Otherwise: classic single-head ActorNetwork
        if getattr(config, "use_dual_head", False):
            self.actor = DualHeadActorNetwork(config).to(device)
            logger.info("  Actor: DualHeadActorNetwork (Direction × Sizing)")
        else:
            self.actor = ActorNetwork(config).to(device)
        self.critic = CriticNetwork(config).to(device)

        # GPU Optimization: torch.compile()
        # Compiles the computation graph once -> ~20-40% faster on T4/A100.
        # Only useful on CUDA with PyTorch >= 2.1. First call is slightly slower
        # (warmup), then permanently faster.
        _on_cuda = device.startswith("cuda") or device == "cuda"
        if config.use_compile and _on_cuda:
            try:
                self.actor = torch.compile(self.actor, mode="reduce-overhead")
                self.critic = torch.compile(self.critic, mode="reduce-overhead")
                logger.info("  torch.compile() active (mode='reduce-overhead')")
            except Exception as e:
                logger.warning(f"  torch.compile() failed, continuing without: {e}")

        # GPU Optimization: AMP GradScaler
        # GradScaler scales the loss to avoid numerical underflows in float16.
        # Automatically disabled if not on CUDA or use_amp=False.
        self._amp_enabled = config.use_amp and _on_cuda
        self._scaler = torch.amp.GradScaler("cuda", enabled=self._amp_enabled)
        if self._amp_enabled:
            logger.info("  Mixed Precision (AMP) active — forward pass in float16")

        # Optimizers & Schedulers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=config.actor_lr)
        self.critic_optimizer = optim.Adam(
            self.critic.parameters(), lr=config.critic_lr
        )

        self.actor_scheduler = None
        self.critic_scheduler = None

        if config.use_lr_decay:
            # CosineAnnealingLR: LR follows a cosine curve from lr → eta_min
            # over T_max steps, then restarts.  Unlike ExponentialLR(0.99) which
            # reaches LR≈0 after ~500 steps and stays there, cosine annealing
            # stays in a useful range for the full training run and cycles back.
            self.actor_scheduler = CosineAnnealingLR(
                self.actor_optimizer,
                T_max=config.lr_decay_steps,
                eta_min=config.actor_lr * 0.01,  # floor at 1% of initial LR
            )
            self.critic_scheduler = CosineAnnealingLR(
                self.critic_optimizer,
                T_max=config.lr_decay_steps,
                eta_min=config.critic_lr * 0.01,
            )

        # ── Option B: Sliding-window sequence buffer ─────────────────────────
        # When use_transformer=True and transformer_seq_len > 1, we maintain a
        # per-episode ring buffer of the last K raw states.  select_action()
        # stacks them into [1, K, state_dim] and feeds the full window to the
        # transformer so it attends across real temporal context instead of just
        # the current step.
        #
        # Key design decisions:
        #  - The ring buffer lives on CPU (numpy). Tensor conversion happens inside
        #    select_action() so the GPU transfer is one contiguous block.
        #  - On episode reset the buffer is zero-filled (padding = zero state).
        #    The causal mask ensures the transformer ignores the padded prefix.
        #  - When transformer_seq_len == 1 (Option A) this buffer is never created
        #    and there is zero overhead on the GRU path.
        self._seq_len = getattr(config, "transformer_seq_len", 1)
        self._use_seq_window = (
            getattr(config, "use_transformer", False) and self._seq_len > 1
        )
        if self._use_seq_window:
            # Ring buffer: [seq_len, state_dim]  — oldest at index 0, newest at -1
            self._state_window: np.ndarray = np.zeros(
                (self._seq_len, config.state_dim), dtype=np.float32
            )
            logger.info(
                f"  Transformer Option B active — sliding window K={self._seq_len}"
            )
        else:
            # Allocate a dummy 1-row buffer so the type is always ndarray.
            # _use_seq_window=False means this buffer is never accessed at runtime.
            self._state_window: np.ndarray = np.zeros(  # type: ignore[no-redef]
                (1, config.state_dim), dtype=np.float32
            )

        # Self-Imitation Learning Buffer
        # Stores past high-reward transitions continuously across epochs
        self.use_sil = config.use_sil
        if self.use_sil:
            self.sil_buffer = {
                "states": np.zeros(
                    (config.sil_capacity, config.state_dim), dtype=np.float32
                ),
                "actions": np.zeros(config.sil_capacity, dtype=np.int64),
                "returns": np.zeros(config.sil_capacity, dtype=np.float32),
                "ptr": 0,
                "size": 0,
            }

        # Experience buffers
        self.reset_buffers()

        logger.info(f"PPOAgent initialized on {device}")
        logger.info(
            f"  Architecture: {config.hidden_dim} hidden, {config.rnn_type if config.use_recurrent else 'MLP'}"
        )
        logger.info(f"  LayerNorm: {config.use_layer_norm}, Dropout: {config.dropout}")
        logger.info(f"  LR Schedule: {config.use_lr_decay}")

    def reset_sequence_window(self) -> None:
        """Reset the sliding-window state buffer to zero-padding.

        Must be called at the start of every episode when transformer_seq_len > 1
        (Option B).  The causal mask in TransformerBackbone ensures that zero-padded
        prefix positions do not pollute the final-position output.

        No-op when the window buffer is disabled (Option A / GRU mode).
        """
        if self._state_window is not None:
            self._state_window[:] = 0.0

    def reset_buffers(self, capacity: int = 0):
        """Reset experience buffers.

        P0-B: Pre-allocate fixed-size numpy arrays for scalar/vector fields.
        Avoids O(N) Python list traversal when converting to tensors at train time.
        'capacity' should be steps_per_iteration (or steps_per_env * n_envs).
        Falls back to dynamic lists when capacity is unknown (capacity=0).
        """
        self._buf_capacity = capacity
        self._buf_ptr = 0  # write pointer

        if capacity > 0:
            state_dim = self.config.state_dim
            self._states_np = np.empty((capacity, state_dim), dtype=np.float32)
            self._actions_np = np.empty(capacity, dtype=np.int64)
            self._rewards_np = np.empty(capacity, dtype=np.float32)
            self._logprobs_np = np.empty(capacity, dtype=np.float32)
            self._values_np = np.empty(capacity, dtype=np.float32)
            self._dones_np = np.empty(capacity, dtype=np.float32)
            # hiddens stay as list (tensors of variable type/shape)
            self.hiddens = []
            # Keep legacy list aliases as views (used by compute_gae)
            self.states = self._states_np  # slice later
            self.actions = self._actions_np
            self.rewards = self._rewards_np
            self.log_probs = self._logprobs_np
            self.values = self._values_np
            self.dones = self._dones_np
        else:
            # Fallback: dynamic Python lists (same API, used when capacity unknown)
            self.states = []
            self.actions = []
            self.rewards = []
            self.log_probs = []
            self.values = []
            self.dones = []
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
        # NaN-Guard: replace NaN/Inf in observations before they go into the network.
        state = np.nan_to_num(
            np.asarray(state, dtype=np.float32), nan=0.0, posinf=1.0, neginf=-1.0
        )

        if self._use_seq_window:
            # Option B: maintain a sliding window of the last K states.
            # Shift all rows left by 1, write new state at the end.
            # _state_window is always a valid ndarray when _use_seq_window is True.
            self._state_window[:-1] = self._state_window[1:]  # type: ignore[index]
            self._state_window[-1] = state  # type: ignore[index]
            # Feed [1, K, state_dim] so TransformerBackbone sees real sequences
            state_tensor = (
                torch.FloatTensor(self._state_window).unsqueeze(0).to(self.device)
            )  # (1, K, state_dim)
        else:
            state_tensor = (
                torch.FloatTensor(state).unsqueeze(0).to(self.device)
            )  # (1, state_dim)

        # Switch to eval mode for deterministic inference (disables dropout)
        training_mode = self.actor.training
        if deterministic:
            self.actor.eval()
            self.critic.eval()

        with torch.no_grad():
            # AMP: Inference runs in float16 on GPU (~1.7x faster on T4/A100).
            # torch.amp.autocast replaces the deprecated torch.cuda.amp.autocast.
            # On CPU: device_type="cpu" with enabled=False → no overhead.
            _amp_device = "cuda" if self._amp_enabled else "cpu"
            with torch.amp.autocast(device_type=_amp_device, enabled=self._amp_enabled):
                # Actor forward pass: delivers action distribution + next hidden state
                dist, next_hidden_actor = self.actor(state_tensor, hidden)

                # Critic forward pass: gets the same hidden state as the actor.
                value, _ = self.critic(state_tensor, hidden)

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

    def select_action_batch(
        self,
        states: np.ndarray,
        hidden: Optional[Union[Tuple, torch.Tensor]] = None,
        deterministic: bool = False,
    ) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray, Optional[Union[Tuple, torch.Tensor]]
    ]:
        """
        Select actions for a batch of N observations in a single GPU forward-pass.

        This is the vectorised counterpart of select_action() and is used together
        with VecTradingEnv to maximise GPU utilisation during trajectory collection.

        Parameters
        ----------
        states : np.ndarray  shape (N, state_dim)
            Batch of observations, one per environment.
        hidden : optional
            Recurrent hidden state with leading batch dimension N.
            For GRU:  tensor shape (rnn_layers, N, hidden_dim)
            For LSTM: tuple of two tensors with the same shape
            Pass None (or the return value of get_initial_hidden_state(batch_size=N))
            at the start of each episode.
        deterministic : bool
            If True uses argmax instead of sampling (for evaluation).

        Returns
        -------
        actions   : np.ndarray  shape (N,)   int   — one action per env
        log_probs : np.ndarray  shape (N,)   float — log π(a|s) per env
        values    : np.ndarray  shape (N,)   float — V(s) per env
        next_hidden : same type/shape as hidden — updated recurrent state
        """
        # (N, state_dim) -> GPU
        # NaN-Guard: replace NaN/Inf in observations before they go into the network.
        # Cause: Feature engine produces NaNs on very small datasets (< lookback)
        # that propagate through GRU and set all logits to NaN.
        # states can be a CUDA tensor (obs_gpu from collect_trajectories_vec) --
        # np.nan_to_num then throws TypeError. Therefore: handle tensor path separately.
        if isinstance(states, torch.Tensor):
            state_tensor = states.to(dtype=torch.float32, device=self.device)
            state_tensor = torch.nan_to_num(
                state_tensor, nan=0.0, posinf=1.0, neginf=-1.0
            )
        else:
            states = np.nan_to_num(states, nan=0.0, posinf=1.0, neginf=-1.0)
            state_tensor = torch.FloatTensor(states).to(self.device)  # (N, state_dim)

        training_mode = self.actor.training
        if deterministic:
            self.actor.eval()
            self.critic.eval()

        with torch.no_grad():
            # AMP: float16 on GPU (~1.7x faster on T4/A100), no overhead on CPU
            _amp_device = "cuda" if self._amp_enabled else "cpu"
            with torch.amp.autocast(device_type=_amp_device, enabled=self._amp_enabled):
                dist, next_hidden_actor = self.actor(
                    state_tensor, hidden
                )  # batch forward
                value_tensor, _ = self.critic(state_tensor, hidden)  # (N, 1)

            if deterministic:
                actions_t = dist.probs.argmax(dim=-1)  # (N,)
            else:
                actions_t = dist.sample()  # (N,)

            log_probs_t = dist.log_prob(actions_t)  # (N,)

        if deterministic:
            self.actor.train(training_mode)
            self.critic.train(training_mode)

        return (
            actions_t.cpu().numpy().astype(np.int64),  # (N,)
            log_probs_t.cpu().numpy().astype(np.float32),  # (N,)
            value_tensor.squeeze(-1).cpu().numpy().astype(np.float32),  # (N,)
            next_hidden_actor,
        )

    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        log_prob: float,
        value: float,
        done: bool,
        hidden: Optional[Any] = None,
    ):
        """Store transition in buffer.

        P0-B: Writes directly into pre-allocated numpy arrays (no list.append
        overhead, no object allocation per step). Falls back to list.append
        when pre-allocation was not used (capacity=0 at reset_buffers call).
        """
        if self._buf_capacity > 0:
            i = self._buf_ptr
            self._states_np[i] = state
            self._actions_np[i] = action
            self._rewards_np[i] = reward
            self._logprobs_np[i] = log_prob
            self._values_np[i] = value
            self._dones_np[i] = float(done)
            self._buf_ptr += 1
        else:
            self.states.append(state)
            self.actions.append(action)
            self.rewards.append(reward)
            self.log_probs.append(log_prob)
            self.values.append(value)
            self.dones.append(float(done))

        # Hidden states stay as list (tensors can't easily go into a pre-alloc buffer)
        if hidden is not None:
            if isinstance(hidden, tuple):
                self.hiddens.append(tuple(h.cpu() for h in hidden))
            else:
                self.hiddens.append(hidden.cpu())
        else:
            self.hiddens.append(None)

    def store_transitions_batch(
        self,
        states: np.ndarray,  # (N, state_dim)
        actions: np.ndarray,  # (N,) int64
        rewards: np.ndarray,  # (N,) float32
        log_probs: np.ndarray,  # (N,) float32
        values: np.ndarray,  # (N,) float32
        dones: np.ndarray,  # (N,) bool/float
        hiddens_batch: list,  # list of N hidden tensors (or list of Nones)
    ):
        """ADV-5: Batch-write N env transitions in a single numpy slice assignment.

        Replaces N individual store_transition() calls with one bulk write into
        the pre-allocated buffers. Reduces Python overhead from O(N×6) scalar
        writes to O(6) slice assignments, which is ~10-15% faster per collect step
        at N=32 envs.

        Falls back to the scalar path if pre-allocation was not used.
        """
        N = len(states)
        if self._buf_capacity > 0:
            ptr = self._buf_ptr
            end = ptr + N
            self._states_np[ptr:end] = states
            self._actions_np[ptr:end] = actions
            self._rewards_np[ptr:end] = rewards.astype(np.float32)
            self._logprobs_np[ptr:end] = log_probs.astype(np.float32)
            self._values_np[ptr:end] = values.astype(np.float32)
            self._dones_np[ptr:end] = dones.astype(np.float32)
            self._buf_ptr = end
        else:
            for i in range(N):
                self.states.append(states[i])
                self.actions.append(int(actions[i]))
                self.rewards.append(float(rewards[i]))
                self.log_probs.append(float(log_probs[i]))
                self.values.append(float(values[i]))
                self.dones.append(float(dones[i]))

        # Hiddens stay as a list of tensors — per-env .cpu() detach
        for h in hiddens_batch:
            if h is None:
                self.hiddens.append(None)
            elif isinstance(h, tuple):
                self.hiddens.append(tuple(x.cpu() for x in h))
            else:
                self.hiddens.append(h.cpu())

    def compute_gae(self, next_value: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Generalized Advantage Estimation (GAE) and returns.

        GAE provides a bias-variance tradeoff in advantage estimation:
        - lambda=0: High bias, low variance (TD(0))
        - lambda=1: Low bias, high variance (Monte Carlo)

        The algorithm works backwards through the trajectory, computing
        advantage estimates that incorporate multi-step returns while
        maintaining computational efficiency.

        Args:
            next_value (float): Bootstrap value for final state (0 if episode ended).

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - advantages: Normalized advantage estimates
                - returns: Target values for value function (advantage + baseline)
        """
        # Determine filled length (pre-alloc path vs list path)
        T = self._buf_ptr if self._buf_capacity > 0 else len(self.rewards)

        # Build contiguous float32 arrays (zero-copy if pre-alloc, one-copy if list)
        if self._buf_capacity > 0:
            rewards_arr = self._rewards_np[:T]
            values_arr_partial = self._values_np[:T]
            dones_arr = self._dones_np[:T]
            values_arr = np.append(values_arr_partial, next_value).astype(np.float32)
        else:
            rewards_arr = np.array(self.rewards, dtype=np.float32)
            dones_arr = np.array(self.dones, dtype=np.float32)
            values_arr = np.array(self.values + [next_value], dtype=np.float32)

        gamma = self.config.gamma
        lam = self.config.gae_lambda

        # PPO-1: Vectorized GAE via scipy.signal.lfilter scan.
        #
        # Standard backward loop:
        #   delta[t] = r[t] + γ·V[t+1]·(1-done[t]) - V[t]
        #   A[t]     = delta[t] + γλ·(1-done[t])·A[t+1]
        #
        # When there are NO episode boundaries (dones == 0) this is a simple
        # IIR filter:  A[t] = delta[t] + c·A[t+1],  c = γλ
        # → lfilter([1], [1, -c], deltas[::-1])[::-1]  (O(T) C-level loop)
        #
        # With episode boundaries (done[t]=1) the continuation coefficient
        # drops to 0 at those steps. We handle this by splitting each segment
        # between done=1 points and running lfilter independently on each.
        # For typical episodes this is equivalent to the scalar loop but runs
        # ~5-10x faster on T=16384 because lfilter is a compiled C routine.
        from scipy.signal import lfilter  # lazy import — already installed on Colab

        deltas = (
            rewards_arr + gamma * values_arr[1:] * (1.0 - dones_arr) - values_arr[:T]
        )  # shape (T,), float32

        # Clip deltas BEFORE lfilter to prevent IIR blow-up on long sequences.
        # With c = γλ ≈ 0.94 and T = 16384, a single delta of 1e4 accumulates to
        # ~1e4 / (1 - 0.94) ≈ 1.7e5 — within float32 range.  But delta = 1e6
        # (from a diverged critic that wasn't fully caught by nan_to_num) would
        # push the filter output toward float32 overflow (>3.4e38).  Clipping to
        # ±10 is safe: rewards are normalised/clipped earlier in the pipeline,
        # and value estimates for a well-behaved critic stay in the same range.
        _DELTA_CLIP = 10.0
        deltas = np.clip(deltas, -_DELTA_CLIP, _DELTA_CLIP)

        advantages = np.empty(T, dtype=np.float32)
        c = float(gamma * lam)

        # Find episode boundaries: indices where done[t] == 1
        done_idxs = np.where(dones_arr)[0]  # may be empty

        if len(done_idxs) == 0:
            # No boundaries → single lfilter pass (fastest path)
            # lfilter solves:  y[n] - c*y[n-1] = x[n]  in forward direction.
            # Reversing makes it a backwards recurrence A[t] = delta[t] + c*A[t+1].
            rev_adv = lfilter([1.0], [1.0, -c], deltas[::-1])
            advantages[:] = rev_adv[::-1]
        else:
            # Segment [start, end] (inclusive) between done boundaries.
            # At a done=1 step the bootstrap is 0, so each segment starts fresh.
            boundaries = np.concatenate(([-1], done_idxs, [T - 1]))
            for i in range(len(boundaries) - 1):
                start = int(boundaries[i]) + 1
                end = int(boundaries[i + 1]) + 1  # exclusive
                seg = deltas[start:end]
                if len(seg) == 0:
                    continue
                rev_adv = lfilter([1.0], [1.0, -c], seg[::-1])
                advantages[start:end] = rev_adv[::-1]

        # Returns = advantage + value baseline (target for V(s))
        returns = advantages + values_arr[:T]

        # NaN/Inf guard: a diverged critic (V → ±∞) or extreme rewards produce
        # non-finite deltas that silently corrupt the entire training step.
        # Zero out any non-finite entries so those transitions contribute no
        # gradient rather than poisoning the whole batch.
        if not np.isfinite(advantages).all():
            n_bad = (~np.isfinite(advantages)).sum()
            logger.warning(f"GAE: {n_bad}/{len(advantages)} non-finite advantages — zeroing")
            advantages = np.nan_to_num(advantages, nan=0.0, posinf=0.0, neginf=0.0)
        if not np.isfinite(returns).all():
            n_bad = (~np.isfinite(returns)).sum()
            logger.warning(f"GAE: {n_bad}/{len(returns)} non-finite returns — zeroing")
            returns = np.nan_to_num(returns, nan=0.0, posinf=0.0, neginf=0.0)

        return advantages, returns

    def train(self, next_value: float = 0.0) -> Dict:
        """
        Perform one PPO update using collected experience.

        This method implements the core PPO training loop:
        1. Compute advantages using GAE
        2. Normalize advantages for stability
        3. Multiple epochs of mini-batch updates
        4. Compute clipped surrogate objective
        5. Update actor and critic networks
        6. Optional early stopping based on KL divergence

        The clipped surrogate objective prevents destructive large policy updates
        by limiting how much the policy can change in each update.

        Args:
            next_value (float): Bootstrap value for the final state in the buffer.
                Should be 0 if the episode ended, otherwise the estimated value
                of the current state. Default: 0.0.

        Returns:
            Dict: Training statistics including:
                - actor_loss: Mean actor loss for this update
                - critic_loss: Mean critic loss for this update
                - entropy: Mean policy entropy
                - mean_kl: Mean KL divergence between old and new policy
                - n_epochs: Number of epochs trained
        """
        # Determine actual number of stored transitions
        # (pre-alloc path: _buf_ptr; list path: len)
        _n = self._buf_ptr if self._buf_capacity > 0 else len(self.states)
        if _n == 0:
            return {}

        # Get contiguous views of only the filled portion
        if self._buf_capacity > 0:
            _states_raw = self._states_np[:_n]
            _actions_raw = self._actions_np[:_n]
            _logprobs_raw = self._logprobs_np[:_n]
        else:
            _states_raw = np.array(self.states, dtype=np.float32)
            _actions_raw = np.array(self.actions, dtype=np.int64)
            _logprobs_raw = np.array(self.log_probs, dtype=np.float32)

        # Step 1: Compute Generalized Advantage Estimation
        advantages, returns = self.compute_gae(next_value)

        # Step 2: Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        # Guard: std≈0 (constant rewards + perfect critic) → normalized adv ≈ 0/ε → safe,
        # but if compute_gae produced NaN that slipped through, clamp it here.
        advantages = np.nan_to_num(advantages, nan=0.0, posinf=0.0, neginf=0.0)
        returns = np.nan_to_num(returns, nan=0.0, posinf=0.0, neginf=0.0)

        # Store good trajectories in SIL buffer
        if self.use_sil:
            self._add_to_sil_buffer(_states_raw, _actions_raw, returns)

        # NaN-Guard on states: prevents NaN logits when feature engineering
        # produces NaN on small datasets (< lookback window).
        _states_raw = np.nan_to_num(_states_raw, nan=0.0, posinf=1.0, neginf=-1.0)

        # Convert to tensors — with Pinned Memory for faster CPU->GPU transfer.
        # When pre-alloc is active: numpy arrays are already contiguous float32 ->
        # no np.array() copy overhead anymore. torch.from_numpy() references the same
        # memory (zero-copy), pin_memory() then pins in-place.
        _pin = self._amp_enabled

        def _to_tensor_f32(arr: np.ndarray) -> torch.Tensor:
            t = torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32))
            return t.pin_memory() if _pin else t

        def _to_tensor_i64(arr: np.ndarray) -> torch.Tensor:
            t = torch.from_numpy(np.ascontiguousarray(arr, dtype=np.int64))
            return t.pin_memory() if _pin else t

        states = _to_tensor_f32(_states_raw).to(self.device, non_blocking=True)
        actions = _to_tensor_i64(_actions_raw).to(self.device, non_blocking=True)
        old_log_probs = _to_tensor_f32(_logprobs_raw).to(self.device, non_blocking=True)
        advantages_t = _to_tensor_f32(advantages).to(self.device, non_blocking=True)
        returns_t = _to_tensor_f32(returns).to(self.device, non_blocking=True)
        advantages = advantages_t
        returns = returns_t

        # Old value estimates V_old(s_t) — needed for clipped value loss.
        # Already stored in the rollout buffer (_values_np) at sample time.
        # nan_to_num: a diverged critic can write NaN into the buffer; clamp to 0
        # so the clipped value loss degrades to an unclipped MSE for those steps.
        if self._buf_capacity > 0:
            _values_raw = self._values_np[:_n]
        else:
            _values_raw = np.array(self.values[:_n], dtype=np.float32)
        _values_raw = np.nan_to_num(_values_raw, nan=0.0, posinf=0.0, neginf=0.0)
        old_values = _to_tensor_f32(_values_raw).to(self.device, non_blocking=True)

        # P1-B + PPO-3: Pre-move all hidden states to GPU ONCE and stack into a
        # single tensor before the epoch loop.
        #
        # P1-B benefit: The old code called .to(device) inside the mini-batch loop
        # → up to (n_epochs × dataset_size // batch_size) = 10×256 = 2560 PCIe
        # transfers per training update. Doing it once here cuts that to 1 transfer.
        #
        # PPO-3 benefit: Each mini-batch previously called torch.cat(gpu_hiddens, dim=1)
        # on a Python list of T=16384 individual tensors → 2560 torch.cat calls per
        # update. Pre-stacking into _h_stacked = (n_layers, T, hidden_dim) lets each
        # mini-batch use a simple index slice: _h_stacked[:, batch_indices, :].
        # This eliminates all per-batch cat overhead.
        has_hidden = len(self.hiddens) > 0 and self.hiddens[0] is not None
        _h_stacked = None  # GRU: (n_layers, T, hidden_dim)
        _hc_stacked = None  # LSTM: tuple ((n_layers,T,h), (n_layers,T,c))
        if has_hidden:
            if self.config.rnn_type == "LSTM":
                # Each hidden: (h=(n_layers,1,h_dim), c=(n_layers,1,h_dim))
                h_list = [hid[0] for hid in self.hiddens[:_n]]
                c_list = [hid[1] for hid in self.hiddens[:_n]]
                # Stack along batch (dim=1): (n_layers, T, h_dim)
                _h_all = torch.cat(h_list, dim=1).to(self.device, non_blocking=True)
                _c_all = torch.cat(c_list, dim=1).to(self.device, non_blocking=True)
                _hc_stacked = (_h_all, _c_all)
            else:  # GRU — each hidden: (n_layers, 1, hidden_dim)
                h_list = [hid for hid in self.hiddens[:_n]]
                # Stack along batch dim → (n_layers, T, hidden_dim)
                _h_stacked = torch.cat(h_list, dim=1).to(self.device, non_blocking=True)

        dataset_size = len(states)
        indices = np.arange(dataset_size)

        total_actor_loss = 0
        total_critic_loss = 0
        total_entropy = 0
        kl_divergences = []

        for epoch in range(self.config.n_epochs):
            # Fix #5: Recurrent PPO needs SEQUENTIAL mini-batches.
            # Shuffling destroys the time-series order and makes GRU-BPTT
            # meaningless (gradient flows through random states).
            # For MLP policies (use_recurrent=False) shuffling is still OK.
            if not self.config.use_recurrent:
                np.random.shuffle(indices)
            # For GRU/LSTM: indices stay in temporal order (0,1,...,T)
            epoch_kls = []  # KL only measured for this epoch (not cumulative)

            for start_idx in range(0, dataset_size, self.config.batch_size):
                end_idx = min(start_idx + self.config.batch_size, dataset_size)
                batch_indices = indices[start_idx:end_idx]

                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                batch_old_values = old_values[batch_indices]

                batch_hidden = None
                if has_hidden:
                    # PPO-3: Slice from pre-stacked tensor instead of calling torch.cat
                    # per batch. _h_stacked is (n_layers, T, hidden_dim); slicing
                    # batch_indices along dim=1 is a zero-copy view → no allocation.
                    if self.config.rnn_type == "LSTM":
                        batch_hidden = (
                            _hc_stacked[0][:, batch_indices, :],
                            _hc_stacked[1][:, batch_indices, :],
                        )
                    else:  # GRU
                        batch_hidden = _h_stacked[:, batch_indices, :]

                # ── Mixed Precision forward pass ─────────────────────────────
                # autocast: Actor/Critic GRU + Linear run in float16 → half
                # VRAM usage, ~1.7× faster matmul on Tensor Cores (T4/A100).
                # On CPU, autocast automatically falls back to float32 (no overhead).
                _amp_device = "cuda" if self._amp_enabled else "cpu"
                with torch.amp.autocast(
                    device_type=_amp_device, enabled=self._amp_enabled
                ):
                    # Actor forward pass with saved hidden state
                    dist, _ = self.actor(batch_states, batch_hidden)

                    # Fix #4 (Training): Critic receives the same hidden state as Actor.
                    # No more None — Critic now has access to the same temporal context.
                    critic_values, _ = self.critic(batch_states, batch_hidden)
                    critic_values = critic_values.squeeze(-1)

                    # Compute losses for this mini-batch
                    # --------------------------------------

                    # Log probability of taken actions under current policy.
                    # .float() forces fp32 even under AMP: dist.log_prob() returns
                    # fp16 on CUDA with autocast, but batch_old_log_probs is fp32.
                    # Keeping both in fp32 prevents precision loss in the log-ratio
                    # that would artificially inflate importance-sampling divergence.
                    log_probs = dist.log_prob(batch_actions).float()

                    # Policy entropy - measures stochasticity of the policy
                    # Higher entropy = more exploration
                    entropy = dist.entropy().mean()

                    # Importance sampling ratio: π_θ(a|s) / π_θ_old(a|s)
                    #
                    # CRITICAL FIX — clamp log-ratio BEFORE exp() to prevent
                    # float32 overflow and gradient explosion.
                    #
                    # Explosion path (without clamp):
                    #   old_log_prob = -0.2  (action was likely under old policy)
                    #   new_log_prob = -8.0  (policy diverged, now action is unlikely)
                    #   log_ratio = +7.8  →  ratio = exp(7.8) ≈ 2440
                    #   surr1 = 2440 × (−adv)  →  huge negative value
                    #   surr2 = clamp(2440, 0.8, 1.2) × (−adv) = 1.2 × (−adv)
                    #   min(surr1, surr2) = surr1  →  giant loss  →  gradient explosion
                    #
                    # With clip_ε=0.2, any |log_ratio| >> ln(1.2)≈0.18 is already
                    # clipped in surr2. Clamping to [-10, +10] is safe: exp(10)≈22k
                    # still allows surr2 to dominate for large ratios while preventing
                    # the pathological surr1 path above.
                    log_ratio = (log_probs - batch_old_log_probs.float()).clamp(-10.0, 10.0)
                    ratio = torch.exp(log_ratio)

                    # PPO Clipped Surrogate Objective
                    # --------------------------------
                    # surr1: Unclipped objective (can lead to large updates)
                    # surr2: Clipped objective (constrained within [1-ε, 1+ε])
                    # Taking the minimum prevents overly large policy changes
                    surr1 = ratio * batch_advantages
                    surr2 = (
                        torch.clamp(
                            ratio,
                            1 - self.config.clip_epsilon,
                            1 + self.config.clip_epsilon,
                        )
                        * batch_advantages
                    )

                    # Actor loss: negative of clipped objective (we maximize, so minimize negative)
                    # The min() ensures we take the less aggressive update
                    actor_loss = -torch.min(surr1, surr2).mean()

                    # Clipped value loss (PPO paper appendix / CleanRL standard):
                    # Prevents the critic from making large jumps in a single update.
                    #
                    # Without clipping: if V_old ≈ 0 and Return ≈ 100, MSE = 10000 →
                    # enormous gradient → critic overshoots, destabilises actor.
                    #
                    # With clipping: the critic update is bounded by clip_epsilon per
                    # step, same as the actor. The max() selects the more conservative
                    # (larger) loss so the critic still improves, just not explosively.
                    v_clipped = batch_old_values + (
                        critic_values - batch_old_values
                    ).clamp(-self.config.clip_epsilon, self.config.clip_epsilon)
                    critic_loss = torch.max(
                        F.mse_loss(critic_values, batch_returns),
                        F.mse_loss(v_clipped, batch_returns),
                    )

                    # Total loss: weighted combination of all components
                    # -actor_loss: We want to maximize this (via minimization)
                    # +critic_loss: Weighted value function loss
                    # -entropy: Entropy bonus (subtracted to maximize entropy)
                    loss = (
                        actor_loss
                        + self.config.value_loss_coef * critic_loss
                        - self.config.entropy_coef * entropy
                    )

                # ── NaN / Inf guard ──────────────────────────────────────────
                # Despite the log-ratio clamp above, AMP scaler overflow or
                # pathological advantage values can still produce non-finite loss.
                # Propagating NaN through .backward() silently corrupts all weights,
                # so we skip the update and zero the grads instead.
                if not torch.isfinite(loss):
                    logger.warning(
                        f"Non-finite PPO loss ({loss.item():.4g}) — skipping batch"
                    )
                    self.actor_optimizer.zero_grad()
                    self.critic_optimizer.zero_grad()
                    continue

                # ── Backward + Gradient Clipping with GradScaler ────────────
                # GradScaler scales the loss to prevent float16 underflow,
                # then automatically unscales before grad-clip and optimizer.step().
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                self._scaler.scale(loss).backward()

                # Grad-Clip: unscale first, then clip (order matters!)
                self._scaler.unscale_(self.actor_optimizer)
                self._scaler.unscale_(self.critic_optimizer)
                nn.utils.clip_grad_norm_(
                    self.actor.parameters(), self.config.max_grad_norm
                )
                nn.utils.clip_grad_norm_(
                    self.critic.parameters(), self.config.max_grad_norm
                )

                self._scaler.step(self.actor_optimizer)
                self._scaler.step(self.critic_optimizer)
                self._scaler.update()  # Scale-Faktor anpassen (auto)

                total_actor_loss += actor_loss.item()
                total_critic_loss += critic_loss.item()
                total_entropy += entropy.item()

                with torch.no_grad():
                    # K3 estimator (Schulman 2017 / SpinningUp):
                    #   KL(old ‖ new) ≈ mean[(ratio − 1) − log_ratio]
                    # Properties:
                    #   • Always ≥ 0  (unlike the 1st-order mean(log_old − log_new)
                    #     which can be negative and mask genuine policy divergence)
                    #   • O(Δ²) accuracy vs O(Δ) for the linear approximation
                    #   • Uses log_ratio already clamped above — consistent estimate
                    # Ref: http://joschu.net/blog/kl-approx.html
                    approx_kl = ((ratio.detach() - 1) - log_ratio.detach()).mean()
                    kl_divergences.append(approx_kl.item())
                    epoch_kls.append(approx_kl.item())

            # Early stopping: check KL only for this epoch (not cumulative across epochs)
            if np.mean(epoch_kls) > self.config.target_kl:
                logger.debug(
                    f"Early stopping at epoch {epoch + 1} (KL={np.mean(epoch_kls):.4f})"
                )
                break

        # Self-Imitation Learning Update
        sil_loss_val = 0.0
        if self.use_sil and self.sil_buffer["size"] >= self.config.sil_batch_size:
            sil_loss_val = self._update_sil()

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
            "mean_kl": float(np.mean(kl_divergences)) if kl_divergences else 0.0,
            "sil_loss": sil_loss_val,
            "n_epochs": epoch + 1,
        }

    def _add_to_sil_buffer(
        self, states: np.ndarray, actions: np.ndarray, returns: np.ndarray
    ):
        """Adds experiences to the SIL replay buffer."""
        n = len(states)
        cap = self.config.sil_capacity

        # We only want to store transitions that were genuinely good (R > V)
        # However, for simplicity and to avoid too much memory copy, we add all
        # and filter during the SIL update step.
        ptr = self.sil_buffer["ptr"]

        if ptr + n <= cap:
            self.sil_buffer["states"][ptr : ptr + n] = states
            self.sil_buffer["actions"][ptr : ptr + n] = actions
            self.sil_buffer["returns"][ptr : ptr + n] = returns
            self.sil_buffer["ptr"] = (ptr + n) % cap
            self.sil_buffer["size"] = min(self.sil_buffer["size"] + n, cap)
        else:
            first_part = cap - ptr
            self.sil_buffer["states"][ptr:cap] = states[:first_part]
            self.sil_buffer["actions"][ptr:cap] = actions[:first_part]
            self.sil_buffer["returns"][ptr:cap] = returns[:first_part]

            second_part = n - first_part
            self.sil_buffer["states"][0:second_part] = states[first_part:]
            self.sil_buffer["actions"][0:second_part] = actions[first_part:]
            self.sil_buffer["returns"][0:second_part] = returns[first_part:]

            self.sil_buffer["ptr"] = second_part
            self.sil_buffer["size"] = cap

    def _update_sil(self) -> float:
        """
        Self-Imitation Learning update step.
        Imitates past decisions that resulted in higher returns than the current value estimate.
        L_sil = -log_prob(a|s) * max(0, R - V(s)) + 1/2 * max(0, R - V(s))^2
        """
        total_sil_loss = 0.0
        updates = 0

        # We do multiple SIL updates per PPO update
        for _ in range(self.config.sil_update_ratio):
            # Sample random batch
            idx = np.random.randint(
                0, self.sil_buffer["size"], size=self.config.sil_batch_size
            )

            b_states = torch.FloatTensor(self.sil_buffer["states"][idx]).to(
                self.device, non_blocking=True
            )
            b_actions = torch.LongTensor(self.sil_buffer["actions"][idx]).to(
                self.device, non_blocking=True
            )
            b_returns = torch.FloatTensor(self.sil_buffer["returns"][idx]).to(
                self.device, non_blocking=True
            )

            _amp_device = "cuda" if self._amp_enabled else "cpu"
            with torch.amp.autocast(device_type=_amp_device, enabled=self._amp_enabled):
                dist, _ = self.actor(
                    b_states, None
                )  # SIL does not currently support BPTT for RNNs, context is forgotten
                values, _ = self.critic(b_states, None)
                values = values.squeeze()

                # Compute advantages: R - V(s)
                adv = b_returns - values

                # We only imitate if the return was better than the value estimate
                clipped_adv = torch.clamp(adv, min=0.0)

                # Check if we have any valid transitions to imitate
                if clipped_adv.max().item() <= 0:
                    continue

                log_probs = dist.log_prob(b_actions)

                # Actor loss: cross entropy weighted by advantage (only positive adv)
                actor_loss = -(log_probs * clipped_adv).mean()

                # Critic loss: update value function towards return (only for positive adv)
                critic_loss = 0.5 * (clipped_adv**2).mean()

                loss = actor_loss + self.config.sil_value_weight * critic_loss

                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()

                self._scaler.scale(loss).backward()
                self._scaler.unscale_(self.actor_optimizer)
                self._scaler.unscale_(self.critic_optimizer)

                nn.utils.clip_grad_norm_(
                    self.actor.parameters(), self.config.max_grad_norm
                )
                nn.utils.clip_grad_norm_(
                    self.critic.parameters(), self.config.max_grad_norm
                )

                self._scaler.step(self.actor_optimizer)
                self._scaler.step(self.critic_optimizer)
                self._scaler.update()

                total_sil_loss += loss.item()
                updates += 1

        return total_sil_loss / max(updates, 1)

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
        # PyTorch >= 2.6: weights_only=False required because PPOConfig
        # is stored as a dataclass in the checkpoint (own source = safe).
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

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
        """Adapt checkpoint weights to current model architecture.

        Handles ONE case cleanly: input feature dimension changed (same hidden_dim).
        Example: feature count went from 30 to 26 — slice the input layer.

        For hidden_dim changes (e.g. 128→256): returns current_state unchanged
        (fresh orthogonal init). Transfer learning from a badly-trained smaller
        model is counterproductive — the old weights encode wrong gradients and
        would require extra iterations to unlearn.
        """
        adapted_state = saved_state.copy()

        key = "feature_extractor.0.weight"
        if key in saved_state and key in current_state:
            saved_weight = saved_state[key]
            current_weight = current_state[key]

            if saved_weight.shape == current_weight.shape:
                # Exact match — nothing to do
                return adapted_state

            # Case: same hidden_dim, different input features (feature engineering change)
            if (
                saved_weight.shape[0] == current_weight.shape[0]
                and saved_weight.shape[1] != current_weight.shape[1]
            ):
                if saved_weight.shape[1] > current_weight.shape[1]:
                    diff = saved_weight.shape[1] - current_weight.shape[1]
                    logger.warning(
                        f"Checkpoint {name}: input features {saved_weight.shape[1]}"
                        f"→{current_weight.shape[1]} (dropping last {diff} features)"
                    )
                    adapted_state[key] = saved_weight[:, : current_weight.shape[1]]
                else:
                    logger.warning(
                        f"Checkpoint {name}: input features {saved_weight.shape[1]}"
                        f"→{current_weight.shape[1]} — cannot adapt, using fresh init"
                    )
                    return current_state.copy()
                return adapted_state

            # Case: hidden_dim changed — do NOT transfer, start fresh.
            # Old weights were trained with wrong gradients (broken reward signal,
            # HMM crash, etc.) — copying them into the larger model would hurt more
            # than a clean orthogonal start.
            logger.warning(
                f"Checkpoint {name}: hidden_dim mismatch "
                f"{saved_weight.shape[0]}→{current_weight.shape[0]}. "
                f"Starting fresh (orthogonal init). Old checkpoint is incompatible."
            )
            return current_state.copy()

        return adapted_state
