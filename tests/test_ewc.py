"""
Tests for Elastic Weight Consolidation (EWC)
=============================================
Covers EWCRegularizer in isolation (no PPOAgent dependency) so that
tests are fast, hermetic, and GPU-optional.

Test inventory
--------------
1. test_ewc_zero_loss_before_anchor_set
   EWC returns 0 before compute_fisher() is called.

2. test_is_initialized_false_before_compute_fisher
   is_initialized() is False before compute_fisher().

3. test_is_initialized_true_after_compute_fisher
   is_initialized() is True after compute_fisher().

4. test_ewc_loss_zero_when_params_equal_anchor
   After anchor is set, ewc_loss == 0 when model params haven't changed.

5. test_ewc_loss_positive_when_params_drift
   After params change, ewc_loss > 0.

6. test_high_fisher_gives_higher_penalty_for_same_drift
   Same drift, but higher Fisher on that weight → larger penalty.

7. test_ewc_loss_is_differentiable
   ewc_loss().backward() succeeds without RuntimeError.

8. test_update_anchor_slides_forward
   After update_anchor(), a further param change produces a smaller penalty
   than the original drift measured from the old anchor.

9. test_get_forgetting_risk_zero_before_init
   get_forgetting_risk() returns 0.0 before anchor is set.

10. test_get_forgetting_risk_positive_after_drift
    get_forgetting_risk() > 0 after params drift from anchor.

11. test_ewc_lambda_scales_loss
    Doubling lambda_ewc doubles the ewc_loss value.

12. test_compute_fisher_with_distribution_model
    compute_fisher works with a Categorical-output actor mock.

13. test_continuous_learning_manager_ewc_disabled
    ContinuousLearningManager(use_ewc=False) leaves self.ewc as None.

14. test_continuous_learning_manager_ewc_enabled
    ContinuousLearningManager(use_ewc=True) creates EWCRegularizer.

15. test_get_forgetting_risk_manager_before_init
    ContinuousLearningManager.get_forgetting_risk() returns 0.0 before
    the first online update sets the anchor.
"""

import asyncio
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.distributions import Categorical

from src.training.ewc import EWCRegularizer


# ─────────────────────────────────────────────────────────────────────────────
# Helpers / fixtures
# ─────────────────────────────────────────────────────────────────────────────


class TinyMLP(nn.Module):
    """Minimal 2-layer MLP used as a stand-in for the actor network.

    forward(state, hidden) → (Categorical distribution, None)

    This mirrors the PPOAgent.actor interface used in EWCRegularizer.
    """

    def __init__(self, in_dim: int = 4, hidden: int = 8, n_actions: int = 3) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, n_actions)

    def forward(self, x: torch.Tensor, hidden=None):
        logits = self.fc2(torch.relu(self.fc1(x)))
        dist = Categorical(logits=logits)
        return dist, None


class ModelWithActor(nn.Module):
    """Wraps TinyMLP as an ``actor`` attribute so EWCRegularizer can find it.

    This mirrors how PPOAgent exposes ``self.actor``.
    """

    def __init__(self, in_dim: int = 4, hidden: int = 8, n_actions: int = 3) -> None:
        super().__init__()
        self.actor = TinyMLP(in_dim, hidden, n_actions)

    def named_parameters(self, prefix="", recurse=True):
        # Expose only actor params (simulates the real agent)
        return self.actor.named_parameters(prefix=prefix, recurse=recurse)


def _make_batch(n: int = 32, state_dim: int = 4, n_actions: int = 3):
    """Return (states, actions) tensors."""
    states = torch.randn(n, state_dim)
    actions = torch.randint(0, n_actions, (n,))
    return states, actions


def _make_ewc_with_anchor(model, lambda_ewc=400.0, n_samples=32):
    """Return an EWCRegularizer that already has an anchor set."""
    ewc = EWCRegularizer(model, lambda_ewc=lambda_ewc)
    batch = _make_batch()
    ewc.compute_fisher(model, batch, n_samples=n_samples)
    return ewc


# ─────────────────────────────────────────────────────────────────────────────
# Tests — EWCRegularizer in isolation
# ─────────────────────────────────────────────────────────────────────────────


def test_ewc_zero_loss_before_anchor_set():
    """EWC should return a zero tensor before compute_fisher() is called."""
    model = TinyMLP()
    ewc = EWCRegularizer(model, lambda_ewc=400.0)

    loss = ewc.ewc_loss(model)

    assert isinstance(loss, torch.Tensor), "ewc_loss must return a torch.Tensor"
    assert float(loss.item()) == pytest.approx(
        0.0
    ), f"Expected 0.0 before anchor is set, got {loss.item()}"


def test_is_initialized_false_before_compute_fisher():
    """is_initialized() must be False before compute_fisher() is called."""
    model = TinyMLP()
    ewc = EWCRegularizer(model)
    assert ewc.is_initialized() is False


def test_is_initialized_true_after_compute_fisher():
    """is_initialized() must be True after compute_fisher() is called."""
    model = TinyMLP()
    ewc = _make_ewc_with_anchor(model)
    assert ewc.is_initialized() is True


def test_ewc_loss_zero_when_params_equal_anchor():
    """
    After anchor is set, if the model parameters haven't changed, ewc_loss
    must be exactly zero (anchor == current params → no penalty).
    """
    model = TinyMLP()
    ewc = _make_ewc_with_anchor(model)

    loss = ewc.ewc_loss(model)

    assert float(loss.item()) == pytest.approx(
        0.0, abs=1e-7
    ), f"Expected 0.0 when params equal anchor, got {loss.item()}"


def test_ewc_loss_positive_when_params_drift():
    """
    After params are perturbed away from the anchor, ewc_loss must be > 0.
    """
    model = TinyMLP()
    ewc = _make_ewc_with_anchor(model)

    # Perturb all parameters by a fixed offset
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.ones_like(p) * 0.5)

    loss = ewc.ewc_loss(model)

    assert (
        float(loss.item()) > 0.0
    ), f"Expected positive ewc_loss after param drift, got {loss.item()}"


def test_high_fisher_gives_higher_penalty_for_same_drift():
    """
    Higher Fisher weights on a parameter → larger penalty for identical drift.

    We test this by:
    1. Computing EWC on a model, then manually inflating Fisher for one layer.
    2. Comparing the resulting ewc_loss against a baseline with lower Fisher.
    """
    model_low = TinyMLP()
    model_high = TinyMLP()

    # Give both models identical parameter values
    with torch.no_grad():
        for p_h, p_l in zip(model_high.parameters(), model_low.parameters()):
            p_h.copy_(p_l)

    ewc_low = _make_ewc_with_anchor(model_low)
    ewc_high = _make_ewc_with_anchor(model_high)

    # Manually set Fisher 10× higher for model_high
    for name in ewc_high._fisher:
        ewc_high._fisher[name] = ewc_high._fisher[name] * 10.0

    # Apply the same drift to both models
    drift = 0.3
    with torch.no_grad():
        for p_l, p_h in zip(model_low.parameters(), model_high.parameters()):
            p_l.add_(torch.ones_like(p_l) * drift)
            p_h.add_(torch.ones_like(p_h) * drift)

    loss_low = float(ewc_low.ewc_loss(model_low).item())
    loss_high = float(ewc_high.ewc_loss(model_high).item())

    assert loss_high > loss_low, (
        f"Higher Fisher should produce higher penalty. "
        f"loss_low={loss_low:.4f}, loss_high={loss_high:.4f}"
    )


def test_ewc_loss_is_differentiable():
    """ewc_loss() must be differentiable so it can be added to PPO loss."""
    model = TinyMLP()
    ewc = _make_ewc_with_anchor(model)

    # Perturb to make loss non-zero
    with torch.no_grad():
        for p in model.parameters():
            p.add_(0.1)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    optimizer.zero_grad()

    loss = ewc.ewc_loss(model)

    # Should not raise
    try:
        loss.backward()
    except RuntimeError as e:
        pytest.fail(f"ewc_loss.backward() raised RuntimeError: {e}")

    # At least one grad should be non-None and non-zero
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads, "No gradients were computed via ewc_loss.backward()"


def test_update_anchor_slides_forward():
    """
    After update_anchor(), the new ewc_loss should reflect the NEW anchor.

    Workflow:
    1. Set anchor at position A.
    2. Drift model to position B → big ewc_loss.
    3. Call update_anchor() → anchor is now at B.
    4. Drift model to C (same magnitude from B) → ewc_loss matches the
       original drift from A to B (same Fisher, same drift size).
    5. Verify ewc_loss at C from new anchor ≈ ewc_loss at B from old anchor.
    """
    model = TinyMLP()
    ewc = _make_ewc_with_anchor(model)

    # Drift A → B
    with torch.no_grad():
        for p in model.parameters():
            p.add_(0.5)

    loss_at_B_from_A = float(ewc.ewc_loss(model).item())
    assert loss_at_B_from_A > 0.0

    # Slide anchor to B
    ewc.update_anchor(model)

    # ewc_loss at B from new anchor B should be ≈ 0
    loss_at_B_from_B = float(ewc.ewc_loss(model).item())
    assert loss_at_B_from_B == pytest.approx(0.0, abs=1e-6), (
        f"After update_anchor, ewc_loss at current position should be 0, "
        f"got {loss_at_B_from_B}"
    )

    # Drift B → C (same drift magnitude)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(0.5)

    loss_at_C_from_B = float(ewc.ewc_loss(model).item())

    # Both drifts are the same magnitude with the same Fisher → losses should match
    assert loss_at_C_from_B == pytest.approx(loss_at_B_from_A, rel=1e-4), (
        f"Same-magnitude drift should produce same ewc_loss. "
        f"loss_at_B_from_A={loss_at_B_from_A:.6f}, "
        f"loss_at_C_from_B={loss_at_C_from_B:.6f}"
    )


def test_get_forgetting_risk_zero_before_init():
    """get_forgetting_risk() must return 0.0 when anchor has not been set."""
    model = TinyMLP()
    ewc = EWCRegularizer(model)
    assert ewc.get_forgetting_risk(model) == pytest.approx(0.0)


def test_get_forgetting_risk_positive_after_drift():
    """get_forgetting_risk() must be > 0 after model drifts from anchor."""
    model = TinyMLP()
    ewc = _make_ewc_with_anchor(model)

    # No drift yet → risk is 0
    assert ewc.get_forgetting_risk(model) == pytest.approx(0.0, abs=1e-8)

    # Perturb params
    with torch.no_grad():
        for p in model.parameters():
            p.add_(0.5)

    risk = ewc.get_forgetting_risk(model)
    assert risk > 0.0, f"Expected positive forgetting risk after drift, got {risk}"


def test_ewc_lambda_scales_loss():
    """Doubling lambda_ewc should double the ewc_loss value."""
    model_a = TinyMLP()
    model_b = TinyMLP()

    # Give both models identical initial parameters
    with torch.no_grad():
        for pa, pb in zip(model_a.parameters(), model_b.parameters()):
            pb.copy_(pa)

    ewc_a = EWCRegularizer(model_a, lambda_ewc=200.0)
    ewc_b = EWCRegularizer(model_b, lambda_ewc=400.0)

    # Use the same batch for both
    batch = _make_batch()
    ewc_a.compute_fisher(model_a, batch, n_samples=len(batch[0]))
    ewc_b.compute_fisher(model_b, batch, n_samples=len(batch[0]))

    # Copy Fisher and anchors from ewc_a to ewc_b so they're identical
    ewc_b._fisher = {k: v.clone() for k, v in ewc_a._fisher.items()}
    ewc_b._anchor_params = {k: v.clone() for k, v in ewc_a._anchor_params.items()}

    # Apply the same drift to both models
    with torch.no_grad():
        for pa, pb in zip(model_a.parameters(), model_b.parameters()):
            pa.add_(0.3)
            pb.add_(0.3)

    loss_a = float(ewc_a.ewc_loss(model_a).item())
    loss_b = float(ewc_b.ewc_loss(model_b).item())

    assert loss_b == pytest.approx(loss_a * 2.0, rel=1e-5), (
        f"lambda=400 should give 2× the loss of lambda=200. "
        f"loss_a={loss_a:.6f}, loss_b={loss_b:.6f}"
    )


def test_compute_fisher_with_distribution_model():
    """compute_fisher() should work with a model that has an 'actor' attribute."""
    model = ModelWithActor()
    ewc = EWCRegularizer(model, lambda_ewc=100.0)

    batch = _make_batch(n=20)
    ewc.compute_fisher(model, batch, n_samples=20)

    assert ewc.is_initialized(), "Should be initialized after compute_fisher"
    assert len(ewc._fisher) > 0, "Fisher dict should be non-empty"
    assert len(ewc._anchor_params) > 0, "Anchor dict should be non-empty"

    # All Fisher values should be non-negative (squared gradients)
    for name, f in ewc._fisher.items():
        assert (f >= 0).all(), f"Fisher values for {name} should be >= 0"


# ─────────────────────────────────────────────────────────────────────────────
# Tests — ContinuousLearningManager EWC integration (lightweight, no I/O)
# ─────────────────────────────────────────────────────────────────────────────


def _make_mock_agent():
    """Return a minimal mock PPOAgent sufficient for CLM construction."""
    agent = MagicMock()
    agent.device = torch.device("cpu")
    agent.config = {}

    # actor mock: returns (Categorical, None) — mirrors the real PPOAgent.actor
    actor_net = TinyMLP(in_dim=4, hidden=8, n_actions=3)
    agent.actor = actor_net

    # critic mock: returns (values_tensor, None)
    def mock_critic(x, h):
        return torch.zeros(x.shape[0], 1), None

    agent.critic = MagicMock(side_effect=mock_critic)
    agent.train = MagicMock(return_value={})
    agent.reset_buffers = MagicMock()
    agent.store_transitions_batch = MagicMock()

    # named_parameters delegates to the actor so EWCRegularizer can iterate
    agent.named_parameters = actor_net.named_parameters
    agent.parameters = actor_net.parameters

    return agent


def test_continuous_learning_manager_ewc_disabled(tmp_path):
    """With use_ewc=False, the manager's ewc attribute should be None."""
    from src.training.continuous_learning import ContinuousLearningManager

    agent = _make_mock_agent()
    mgr = ContinuousLearningManager(
        agent=agent,
        checkpoint_dir=str(tmp_path),
        update_interval_steps=128,
        use_ewc=False,
    )

    assert mgr.ewc is None, "ewc should be None when use_ewc=False"


def test_continuous_learning_manager_ewc_enabled(tmp_path):
    """With use_ewc=True (default), the manager's ewc attribute should be an EWCRegularizer."""
    from src.training.continuous_learning import ContinuousLearningManager

    agent = _make_mock_agent()
    mgr = ContinuousLearningManager(
        agent=agent,
        checkpoint_dir=str(tmp_path),
        update_interval_steps=128,
        use_ewc=True,
        ewc_lambda=400.0,
    )

    assert isinstance(
        mgr.ewc, EWCRegularizer
    ), f"ewc should be EWCRegularizer, got {type(mgr.ewc)}"
    assert (
        not mgr.ewc.is_initialized()
    ), "EWC should NOT be initialized before the first online update"


def test_get_forgetting_risk_manager_before_init(tmp_path):
    """
    ContinuousLearningManager.get_forgetting_risk() must return 0.0 before
    the first online update sets the EWC anchor.
    """
    from src.training.continuous_learning import ContinuousLearningManager

    agent = _make_mock_agent()
    mgr = ContinuousLearningManager(
        agent=agent,
        checkpoint_dir=str(tmp_path),
        update_interval_steps=128,
        use_ewc=True,
    )

    risk = mgr.get_forgetting_risk()
    assert risk == pytest.approx(
        0.0
    ), f"Expected 0.0 before first online update, got {risk}"
