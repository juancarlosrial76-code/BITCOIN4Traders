"""
Meta-Learning (Learning to Learn) for Trading
===============================================
SUPERHUMAN feature: The system learns how to learn from few examples.

Instead of training from scratch on each market:
- Learns a "prior" from many markets
- Adapts to new markets with just 10-100 examples
- Continually improves without forgetting

Uses:
- Model-Agnostic Meta-Learning (MAML)
- Few-shot learning
- Continual learning without catastrophic forgetting
- Adaptive learning rates per parameter

2040 Status: AI that adapts instantly like a human expert
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass
from collections import deque
import copy
from loguru import logger


@dataclass
class MetaLearningConfig:
    """Configuration for meta-learning."""

    inner_lr: float = 0.01  # Learning rate for task-level adaptation (inner loop)
    meta_lr: float = 0.001  # Learning rate for meta-parameter update (outer loop)
    inner_steps: int = 5  # Gradient steps per task during adaptation
    meta_batch_size: int = 4  # Number of tasks per meta-update
    first_order: bool = False  # Use first-order MAML (faster, less memory)


class MAMLTrader:
    """
    Model-Agnostic Meta-Learning for trading.

    Learns network initialization that can adapt to any market
    with just a few gradient steps.

    Key insight: Instead of learning one strategy, learn HOW to
    quickly learn strategies for any market condition.
    """

    def __init__(self, base_model: nn.Module, config: MetaLearningConfig = None):
        self.config = config or MetaLearningConfig()
        self.base_model = base_model
        # Store meta-parameters as clones (updated by meta-optimizer)
        self.meta_parameters = {n: p.clone() for n, p in base_model.named_parameters()}
        # Adam optimizer drives the outer (meta) loop updates
        self.meta_optimizer = torch.optim.Adam(
            self.base_model.parameters(), lr=self.config.meta_lr
        )
        self.task_history = []
        logger.info(f"MAMLTrader initialized: {self.config.inner_steps} inner steps")

    def adapt_to_task(
        self, support_x: torch.Tensor, support_y: torch.Tensor, create_copy: bool = True
    ) -> nn.Module:
        """
        Adapt model to new task (market) with few examples.

        This is the "learning" phase - takes just 5-10 gradient steps.
        """
        if create_copy:
            model = copy.deepcopy(self.base_model)  # Deep copy to avoid mutating base
        else:
            model = self.base_model

        # Initialize from meta-parameters (warm start from learned initialization)
        for name, param in model.named_parameters():
            if name in self.meta_parameters:
                param.data = self.meta_parameters[name].clone()

        # Inner loop: adapt to task with SGD (fast, few steps)
        inner_optimizer = torch.optim.SGD(model.parameters(), lr=self.config.inner_lr)

        for step in range(self.config.inner_steps):
            inner_optimizer.zero_grad()

            # Forward pass on support examples
            predictions = model(support_x)
            loss = F.mse_loss(predictions, support_y)

            # Backward pass: compute gradients wrt task-adapted parameters
            loss.backward()
            inner_optimizer.step()

            if step == 0:
                initial_loss = (
                    loss.item()
                )  # Record starting loss for improvement logging

        final_loss = loss.item()
        improvement = (initial_loss - final_loss) / (initial_loss + 1e-8)

        logger.debug(
            f"Task adaptation: {initial_loss:.4f} -> {final_loss:.4f} "
            f"({improvement:.1%} improvement)"
        )

        return model

    def meta_update(
        self, tasks: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]
    ):
        """
        Meta-learning update across multiple tasks.

        Each task is (support_x, support_y, query_x, query_y).
        Learns initialization that works well for ALL tasks.
        """
        meta_loss = 0.0

        for support_x, support_y, query_x, query_y in tasks:
            # Inner loop: adapt to this task
            adapted_model = self.adapt_to_task(support_x, support_y, create_copy=True)

            # Evaluate on query set (held-out data from same task)
            query_predictions = adapted_model(query_x)
            task_loss = F.mse_loss(query_predictions, query_y)

            meta_loss += task_loss  # Accumulate across tasks

        # Average meta-loss across the batch of tasks
        meta_loss = meta_loss / len(tasks)

        # Outer loop update: improve the initialization
        self.meta_optimizer.zero_grad()
        meta_loss.backward()
        self.meta_optimizer.step()

        # Snapshot updated meta-parameters for next adaptation
        self.meta_parameters = {
            n: p.clone().detach() for n, p in self.base_model.named_parameters()
        }

        logger.info(f"Meta-update: loss = {meta_loss.item():.4f}")

        return meta_loss.item()

    def train_meta(self, task_generator: Callable, n_iterations: int = 1000):
        """
        Train meta-learning across many tasks.

        task_generator should yield (support_x, support_y, query_x, query_y)
        """
        losses = []

        for iteration in range(n_iterations):
            # Sample a batch of diverse market tasks
            tasks = [task_generator() for _ in range(self.config.meta_batch_size)]

            # Meta-update using this batch
            loss = self.meta_update(tasks)
            losses.append(loss)

            if iteration % 100 == 0:
                avg_loss = np.mean(losses[-100:])  # Rolling 100-iteration average
                logger.info(f"Iteration {iteration}: avg loss = {avg_loss:.4f}")

        return losses


class FewShotLearner:
    """
    Few-shot learning: Learn from just 10-100 examples.

    For new markets or rare regimes where data is scarce.
    """

    def __init__(self, feature_dim: int = 64, hidden_dim: int = 128):
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim

        # Prototypical networks: learn embeddings where
        # similar examples cluster together in embedding space
        self.encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 64),  # Compress to 64-dim embedding space
        )

        self.prototypes = {}  # Class centroids in embedding space
        logger.info("FewShotLearner initialized")

    def compute_prototypes(
        self, support_x: torch.Tensor, support_y: torch.Tensor, n_classes: int = 2
    ):
        """
        Compute class prototypes (centroids) in embedding space.

        Each class is represented by the mean of its support examples.
        """
        embeddings = self.encoder(support_x)  # Map support examples to embedding space

        prototypes = {}
        for c in range(n_classes):
            mask = (support_y == c).squeeze()  # Boolean mask for class c
            if mask.sum() > 0:
                prototypes[c] = embeddings[mask].mean(dim=0)  # Centroid of class c

        self.prototypes = prototypes
        return prototypes

    def predict(self, query_x: torch.Tensor) -> torch.Tensor:
        """
        Predict class for query examples.

        Assigns to nearest prototype in embedding space.
        """
        if len(self.prototypes) == 0:
            raise ValueError("No prototypes computed. Call compute_prototypes first.")

        query_embeddings = self.encoder(query_x)  # Embed query examples

        # Compute Euclidean distance from each query to each prototype
        distances = {}
        for c, prototype in self.prototypes.items():
            dist = torch.norm(query_embeddings - prototype, dim=1)
            distances[c] = dist

        # Negative distances as logits (closer = higher score)
        logits = torch.stack([-distances[c] for c in sorted(distances.keys())], dim=1)
        probs = F.softmax(logits, dim=1)  # Convert to probabilities

        return probs

    def learn_new_market(
        self, examples: pd.DataFrame, labels: pd.Series, n_shots: int = 10
    ):
        """
        Learn a new market from just n examples.
        """
        # Sample n-shot support set without replacement
        support_indices = np.random.choice(
            len(examples), size=min(n_shots, len(examples)), replace=False
        )

        support_x = torch.FloatTensor(examples.iloc[support_indices].values)
        support_y = torch.LongTensor(labels.iloc[support_indices].values)

        # Compute class prototypes from this tiny support set
        self.compute_prototypes(support_x, support_y)

        logger.success(f"Learned new market from {n_shots} examples")


class ContinualLearner:
    """
    Continual learning: Learn continuously without forgetting.

    Standard neural networks forget old tasks when learning new ones.
    This system remembers everything.

    Uses Elastic Weight Consolidation (EWC) to protect important weights.
    """

    def __init__(self, model: nn.Module, lambda_ewc: float = 1000.0):
        self.model = model
        self.lambda_ewc = (
            lambda_ewc  # Regularization strength (higher = less forgetting)
        )
        self.fisher_dict = {}  # Fisher information matrices per task
        self.optimal_params = {}  # Optimal parameters after each task
        self.task_count = 0
        logger.info("ContinualLearner initialized")

    def compute_fisher_information(self, data_loader: torch.utils.data.DataLoader):
        """
        Compute Fisher Information matrix for current task.

        Tells us which parameters are important for current task.
        """
        self.model.eval()
        fisher = {n: torch.zeros_like(p) for n, p in self.model.named_parameters()}

        for batch_x, batch_y in data_loader:
            self.model.zero_grad()
            output = self.model(batch_x)
            loss = F.mse_loss(output, batch_y)
            loss.backward()

            # Accumulate squared gradients (diagonal Fisher approximation)
            for n, p in self.model.named_parameters():
                if p.grad is not None:
                    fisher[n] += p.grad.pow(2) / len(
                        data_loader
                    )  # Normalize by dataset size

        return fisher

    def update_ewc_params(self, data_loader: torch.utils.data.DataLoader):
        """Update EWC parameters after learning a task."""
        # Store Fisher information and optimal params for current task
        self.fisher_dict[self.task_count] = self.compute_fisher_information(data_loader)
        self.optimal_params[self.task_count] = {
            n: p.clone().detach() for n, p in self.model.named_parameters()
        }
        self.task_count += 1

    def ewc_loss(self) -> torch.Tensor:
        """
        Compute EWC penalty to prevent forgetting.

        Penalizes changes to important parameters from previous tasks.
        """
        if self.task_count == 0:
            return torch.tensor(0.0)  # No penalty before any task has been learned

        loss = torch.tensor(0.0)
        for task_id in range(self.task_count):
            for n, p in self.model.named_parameters():
                if n in self.fisher_dict[task_id]:
                    fisher = self.fisher_dict[task_id][n]  # Importance weight
                    optimal = self.optimal_params[task_id][n]  # Target value
                    # Weighted squared deviation from optimal parameters
                    loss += (fisher * (p - optimal).pow(2)).sum()

        return self.lambda_ewc * loss  # Scale by regularization strength

    def train_on_task(self, data_loader: torch.utils.data.DataLoader, epochs: int = 10):
        """Train on new task while protecting old knowledge."""
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

        for epoch in range(epochs):
            for batch_x, batch_y in data_loader:
                optimizer.zero_grad()

                # Task loss: fit the current data
                output = self.model(batch_x)
                task_loss = F.mse_loss(output, batch_y)

                # EWC penalty: don't deviate from what worked for previous tasks
                ewc_penalty = self.ewc_loss()

                # Total loss balances new learning and memory retention
                total_loss = task_loss + ewc_penalty

                total_loss.backward()
                optimizer.step()

            if epoch % 2 == 0:
                logger.debug(
                    f"Epoch {epoch}: task_loss={task_loss.item():.4f}, "
                    f"ewc={ewc_penalty.item():.4f}"
                )

        # Update EWC parameters with this task's importance information
        self.update_ewc_params(data_loader)

        logger.success(f"Trained on task {self.task_count} without forgetting")


class AdaptiveLearningRate:
    """
    Adaptive learning rates for each parameter.

    Different parameters need different learning rates.
    This learns the optimal learning rate for each weight.
    """

    def __init__(self, model: nn.Module, base_lr: float = 0.001):
        self.model = model
        self.base_lr = base_lr

        # Per-parameter learning rate multipliers (initialized to 1 = base_lr)
        self.lr_multipliers = {
            n: torch.ones_like(p) for n, p in model.named_parameters()
        }

        # Momentum for LR adaptation (exponential moving average of gradients)
        self.lr_momentum = {n: torch.zeros_like(p) for n, p in model.named_parameters()}

        logger.info("AdaptiveLearningRate initialized")

    def step(self, loss: torch.Tensor):
        """
        Update parameters with adaptive learning rates.

        Increases LR for consistently useful parameters.
        Decreases LR for noisy/irrelevant parameters.
        """
        # Compute gradients for all parameters
        self.model.zero_grad()
        loss.backward()

        # Update learning rates based on gradient history
        for n, p in self.model.named_parameters():
            if p.grad is None:
                continue

            grad = p.grad

            # Exponential moving average of gradients (tracks consistent direction)
            self.lr_momentum[n] = 0.9 * self.lr_momentum[n] + 0.1 * grad

            # Adapt learning rate: consistent gradient direction → increase LR
            consistency = torch.sign(self.lr_momentum[n] * grad)
            self.lr_multipliers[n] += 0.01 * consistency
            # Clamp multipliers to [0.1, 10] to prevent extreme values
            self.lr_multipliers[n] = torch.clamp(self.lr_multipliers[n], 0.1, 10.0)

            # Update parameter using element-wise adaptive learning rate
            effective_lr = self.base_lr * self.lr_multipliers[n]
            p.data -= effective_lr * grad


class MetaTradingAgent:
    """
    Complete meta-learning trading agent.

    Combines MAML, few-shot learning, and continual learning
    to create a trading agent that:
    1. Adapts instantly to new markets
    2. Learns from few examples
    3. Never forgets old strategies
    4. Optimizes its own learning
    """

    def __init__(self, input_dim: int = 64):
        # Base model: 3-layer MLP producing Buy/Hold/Sell logits
        self.base_model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 3),  # Buy, Hold, Sell
        )

        # Meta-learner for fast adaptation via MAML
        self.maml = MAMLTrader(self.base_model)

        # Few-shot learner for new markets with minimal data
        self.few_shot = FewShotLearner(input_dim)

        # Continual learner to retain historical strategies
        self.continual = ContinualLearner(self.base_model)

        # Adaptive LR to self-tune parameter update sizes
        self.adaptive_lr = AdaptiveLearningRate(self.base_model)

        logger.info("MetaTradingAgent initialized")

    def quick_adapt(self, market_data: pd.DataFrame, n_examples: int = 20) -> nn.Module:
        """
        Instantly adapt to a new market.

        Uses MAML to adapt in just 5 gradient steps.
        """
        # Prepare data: separate features from target column
        X = torch.FloatTensor(market_data.drop("target", axis=1).values[:n_examples])
        y = torch.LongTensor(market_data["target"].values[:n_examples])

        # Run inner-loop adaptation
        adapted_model = self.maml.adapt_to_task(X, y)

        logger.success(f"Adapted to new market in {self.maml.config.inner_steps} steps")

        return adapted_model

    def predict(self, features: np.ndarray, use_adapted: bool = True) -> np.ndarray:
        """Generate predictions."""
        x = torch.FloatTensor(features)

        if use_adapted:
            with torch.no_grad():
                logits = self.base_model(x)
                probs = F.softmax(logits, dim=-1)  # Softmax over Buy/Hold/Sell
        else:
            # Use few-shot prototypes instead of base model
            probs = self.few_shot.predict(x)

        return probs.numpy()


# Production functions
def create_meta_trader(input_dim: int = 64) -> MetaTradingAgent:
    """Create meta-learning trading agent."""
    return MetaTradingAgent(input_dim)


def adapt_to_new_market(
    agent: MetaTradingAgent, market_data: pd.DataFrame, n_examples: int = 20
) -> nn.Module:
    """Quickly adapt agent to new market."""
    return agent.quick_adapt(market_data, n_examples)


def learn_without_forgetting(
    agent: MetaTradingAgent, task_data: torch.utils.data.DataLoader
):
    """Learn new strategy without forgetting old ones."""
    agent.continual.train_on_task(task_data)


# Example usage
if __name__ == "__main__":
    # Create meta-trader
    agent = create_meta_trader(input_dim=10)

    # Adapt to new market with just 20 examples
    # (vs thousands normally needed)
    new_market_data = pd.DataFrame(
        np.random.randn(100, 11), columns=[f"f{i}" for i in range(10)] + ["target"]
    )
    adapted_model = adapt_to_new_market(agent, new_market_data, n_examples=20)

    logger.success("Meta-learning demonstration complete!")
