"""
Elastic Weight Consolidation (EWC)
====================================
Prevents catastrophic forgetting during online learning by penalizing
large changes to important weights (measured by Fisher Information).

The EWC loss term added to the agent's training objective is:

    L_ewc = λ/2 · Σᵢ Fᵢ · (θᵢ - θ*ᵢ)²

Where:
    θ*ᵢ  = "anchor" parameter values captured before the online update
    Fᵢ   = diagonal Fisher Information (importance of parameter i)
    λ    = regularization strength (higher → stronger protection of old knowledge)

Fisher Information is approximated via squared gradients of the log-likelihood
(empirical Fisher):

    Fᵢ ≈ E[(∂ log π(a|s) / ∂θᵢ)²]

This is computed by a single forward+backward pass over a representative
batch of recent experiences, using the current policy as the reference
distribution.

Reference:
    Kirkpatrick et al. (2017) "Overcoming catastrophic forgetting in neural
    networks" — https://arxiv.org/abs/1612.00796
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple

from loguru import logger


class EWCRegularizer:
    """
    Elastic Weight Consolidation regularizer.

    Captures a parameter "anchor" (θ*) and diagonal Fisher Information (F)
    for a given model, then provides an EWC penalty loss that resists drift
    away from the anchor during subsequent gradient updates.

    Usage pattern
    -------------
    1. After the FIRST successful online update (model has learned task A):
           ewc.compute_fisher(model, batch)   # sets anchor θ* and Fisher F

    2. During each subsequent update, add the EWC penalty to the total loss:
           loss_total = loss_ppo + ewc.ewc_loss(model)
           loss_total.backward()

    3. After each subsequent successful update, refresh the anchor:
           ewc.update_anchor(model)           # slide θ* forward to prevent
                                               # over-constraining old anchors

    Args:
        model      : The nn.Module whose weights will be regularized.
                     Only parameters that require gradients are tracked.
        lambda_ewc : Regularization strength λ.  Sensible range: 100–1000.
                     Higher values → stronger protection, slower adaptation.
                     Default 400.0 is a reasonable starting point for PPO.
    """

    def __init__(self, model: nn.Module, lambda_ewc: float = 400.0) -> None:
        self._lambda: float = lambda_ewc

        # θ* — snapshot of parameters at anchor time
        self._anchor_params: Dict[str, torch.Tensor] = {}

        # Fᵢ — diagonal Fisher Information for each parameter tensor
        self._fisher: Dict[str, torch.Tensor] = {}

        logger.info(
            f"EWCRegularizer created | lambda_ewc={lambda_ewc} | "
            f"initialized=False (call compute_fisher() after first update)"
        )

    # ------------------------------------------------------------------
    # Fisher estimation & anchor capture
    # ------------------------------------------------------------------

    def compute_fisher(
        self,
        model: nn.Module,
        batch: Tuple[torch.Tensor, torch.Tensor],
        n_samples: int = 200,
    ) -> None:
        """Estimate diagonal Fisher Information and capture the anchor.

        Uses the empirical Fisher approximation:

            F_i ≈ (1/N) Σ_n [ (∂ log π(aₙ | sₙ) / ∂θᵢ)² ]

        A single forward+backward pass over *batch* (or up to *n_samples*
        rows from it) is sufficient for the diagonal approximation.

        After this call:
            - ``self._fisher``        contains per-parameter importance tensors.
            - ``self._anchor_params`` contains a detached copy of all params.

        Args:
            model    : The current (post-first-update) model. Must have an
                       ``actor`` attribute with a forward that returns
                       (distribution, hidden) — same interface as PPOAgent.actor.
                       If model has no ``actor`` attribute we fall back to
                       treating ``model`` itself as the distribution producer.
            batch    : Tuple of (states, actions) torch tensors, both on the
                       same device as the model.  States shape: (N, state_dim).
                       Actions shape: (N,) — discrete action indices.
            n_samples: Maximum number of transitions used for Fisher estimation.
                       Larger → more accurate but slower.  Default: 200.
        """
        states, actions = batch

        # Clamp to n_samples
        if states.shape[0] > n_samples:
            idx = torch.randperm(states.shape[0], device=states.device)[:n_samples]
            states = states[idx]
            actions = actions[idx]

        # Accumulate squared gradients of log π(a|s) w.r.t. all trainable params
        fisher_acc: Dict[str, torch.Tensor] = {}
        for name, p in model.named_parameters():
            if p.requires_grad:
                fisher_acc[name] = torch.zeros_like(p.data)

        model.train()

        # Determine which sub-module produces the action distribution
        actor = getattr(model, "actor", model)

        for i in range(states.shape[0]):
            s = states[i].unsqueeze(0)  # (1, state_dim)
            a = actions[i]  # scalar index

            # Zero existing gradients in the model
            model.zero_grad()

            # Forward pass — actor returns (distribution, hidden)
            try:
                dist, _ = actor(s, None)
            except TypeError:
                # Fallback: actor takes only state (no hidden)
                dist = actor(s)

            # Log-likelihood of the action taken
            log_prob = dist.log_prob(a)

            # Backward to get per-parameter gradients of log π
            log_prob.backward()

            # Accumulate squared gradients
            for name, p in model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    fisher_acc[name] += p.grad.data.clone() ** 2

        # Normalise by number of samples
        n = float(states.shape[0])
        self._fisher = {name: v / n for name, v in fisher_acc.items()}

        # Capture anchor θ* — detached copies, moved to CPU to save GPU memory
        self._anchor_params = {
            name: p.data.detach().clone()
            for name, p in model.named_parameters()
            if p.requires_grad
        }

        model.zero_grad()

        logger.info(
            f"EWC anchor set | n_samples={int(n)} | "
            f"tracked_params={len(self._anchor_params)} | "
            f"mean_fisher={float(torch.stack([f.mean() for f in self._fisher.values()]).mean()):.6f}"
        )

    # ------------------------------------------------------------------
    # EWC penalty loss
    # ------------------------------------------------------------------

    def ewc_loss(self, model: nn.Module) -> torch.Tensor:
        """Compute the EWC regularization penalty.

        L_ewc = λ/2 · Σᵢ Fᵢ · (θᵢ - θ*ᵢ)²

        Returns a scalar tensor that can be added to any PyTorch loss and
        back-propagated.  If the anchor has not been set yet (first update),
        returns a zero tensor so training proceeds without regularization.

        Args:
            model: The model currently being trained.  Must be the same
                   architecture as the one used in compute_fisher().

        Returns:
            Scalar torch.Tensor — the EWC penalty (differentiable).
        """
        if not self.is_initialized():
            # No anchor yet — return differentiable zero on the right device
            device = next(model.parameters()).device
            return torch.tensor(0.0, device=device, requires_grad=False)

        device = next(model.parameters()).device
        penalty = torch.tensor(0.0, device=device)

        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if name not in self._anchor_params or name not in self._fisher:
                continue

            anchor = self._anchor_params[name].to(device)
            fisher = self._fisher[name].to(device)

            # Weighted squared deviation from anchor
            penalty = penalty + (fisher * (p - anchor) ** 2).sum()

        return (self._lambda / 2.0) * penalty

    # ------------------------------------------------------------------
    # State queries & anchor management
    # ------------------------------------------------------------------

    def is_initialized(self) -> bool:
        """Return True if both the anchor and Fisher have been computed.

        This is False before the first call to compute_fisher() and True
        for all subsequent calls.
        """
        return bool(self._anchor_params) and bool(self._fisher)

    def update_anchor(self, model: nn.Module) -> None:
        """Slide the anchor θ* forward to the current model parameters.

        Call this *after* each successful online update to prevent the
        regularizer from constraining the model too tightly to a distant
        historical anchor (which would eventually stall adaptation).

        This does NOT recompute the Fisher — it only refreshes θ*.  If you
        want to recompute Fisher (more expensive) call compute_fisher() again.

        Args:
            model: The freshly updated model.
        """
        if not self.is_initialized():
            logger.warning(
                "EWC.update_anchor() called before compute_fisher(). "
                "Call compute_fisher() first to set the initial anchor."
            )
            return

        self._anchor_params = {
            name: p.data.detach().clone()
            for name, p in model.named_parameters()
            if p.requires_grad
        }
        logger.debug("EWC anchor updated to current model parameters.")

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def get_forgetting_risk(self, model: nn.Module) -> float:
        """Compute mean Fisher-weighted parameter drift from the anchor.

        This is a scalar diagnostic indicating how much the current model
        has drifted from the anchor in Fisher-importance-weighted space.
        Higher values indicate higher risk of catastrophic forgetting.

        Defined as:
            risk = mean_i [ Fᵢ · (θᵢ - θ*ᵢ)² ]

        Unlike ewc_loss(), this is not multiplied by λ and is not
        differentiable — it is for monitoring/logging only.

        Args:
            model: The current model to evaluate.

        Returns:
            Float in [0, ∞).  Zero means no drift from the anchor.
        """
        if not self.is_initialized():
            return 0.0

        device = next(model.parameters()).device
        weighted_drifts = []

        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if name not in self._anchor_params or name not in self._fisher:
                continue

            anchor = self._anchor_params[name].to(device)
            fisher = self._fisher[name].to(device)

            drift = (fisher * (p.data - anchor) ** 2).mean().item()
            weighted_drifts.append(drift)

        if not weighted_drifts:
            return 0.0

        return float(sum(weighted_drifts) / len(weighted_drifts))

    def __repr__(self) -> str:
        return (
            f"EWCRegularizer("
            f"lambda={self._lambda}, "
            f"initialized={self.is_initialized()}, "
            f"tracked_params={len(self._anchor_params)})"
        )
