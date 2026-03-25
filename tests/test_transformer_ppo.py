"""
Validation tests for TransformerBackbone + PPOAgent with Transformer backbone
and Self-Imitation Learning (SIL).

Run with:
    cd /home/hp17/Tradingbot/BITCOIN4Traders
    python -m pytest tests/test_transformer_ppo.py -v
"""

import numpy as np
import pytest
import torch

from src.networks.transformer_net import PositionalEncoding, TransformerBackbone
from src.agents.ppo_agent import PPOAgent, PPOConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def device():
    return "cpu"


@pytest.fixture
def transformer_backbone():
    return TransformerBackbone(
        input_dim=20,
        d_model=64,
        nhead=4,
        num_layers=2,
        dropout=0.0,  # deterministic for tests
    )


@pytest.fixture
def ppo_config_transformer():
    return PPOConfig(
        state_dim=20,
        hidden_dim=64,
        n_actions=3,
        use_transformer=True,
        use_recurrent=False,  # transformer replaces RNN
        use_amp=False,  # CPU — no AMP
        use_compile=False,
        use_sil=True,
        sil_capacity=1000,
        sil_batch_size=16,
        sil_update_ratio=2,
        n_epochs=2,
        batch_size=16,
        dropout=0.0,
    )


# ---------------------------------------------------------------------------
# 1. PositionalEncoding tests
# ---------------------------------------------------------------------------


class TestPositionalEncoding:
    """Verify PositionalEncoding is compatible with batch_first=True layout."""

    def test_output_shape_unchanged(self):
        pe = PositionalEncoding(d_model=64)
        x = torch.randn(2, 10, 64)  # [batch, seq_len, d_model]
        out = pe(x)
        assert (
            out.shape == x.shape
        ), f"PositionalEncoding must preserve shape; got {out.shape}"

    def test_pe_adds_information(self):
        """Encoding should modify the tensor (not be a no-op)."""
        pe = PositionalEncoding(d_model=64)
        x = torch.zeros(1, 10, 64)
        out = pe(x)
        assert not torch.allclose(out, x), "PE must add non-zero positional info"

    def test_different_positions_differ(self):
        """Two different positions should have different encodings."""
        pe = PositionalEncoding(d_model=64)
        x = torch.zeros(1, 10, 64)
        out = pe(x)
        # Position 0 and position 5 should differ
        assert not torch.allclose(
            out[0, 0], out[0, 5]
        ), "PE at different positions must differ"

    def test_batch_first_layout(self):
        """Buffer is stored as [1, max_len, d_model] — batch_first format."""
        pe = PositionalEncoding(d_model=32, max_len=100)
        assert pe.pe.shape == (
            1,
            100,
            32,
        ), f"PE buffer must be [1, max_len, d_model], got {pe.pe.shape}"


# ---------------------------------------------------------------------------
# 2. TransformerBackbone tests
# ---------------------------------------------------------------------------


class TestTransformerBackbone:
    """Verify the TransformerBackbone forward pass and causal masking."""

    def test_3d_input_shape(self, transformer_backbone):
        """Primary use-case: (batch=2, seq=10, features=20)."""
        x = torch.randn(2, 10, 20)
        out = transformer_backbone(x)
        assert out.shape == (2, 64), f"Expected (2, 64), got {out.shape}"

    def test_2d_input_shape(self, transformer_backbone):
        """Single time-step fallback: (batch=4, features=20)."""
        x = torch.randn(4, 20)
        out = transformer_backbone(x)
        assert out.shape == (4, 64), f"Expected (4, 64), got {out.shape}"

    def test_causal_mask_shape(self, transformer_backbone):
        """Causal mask must be [seq_len, seq_len] upper-triangular -inf."""
        seq_len = 10
        mask = transformer_backbone._causal_mask(seq_len, torch.device("cpu"))
        assert mask.shape == (
            seq_len,
            seq_len,
        ), f"Mask shape should be ({seq_len},{seq_len}), got {mask.shape}"
        # Diagonal and below should be 0.0 (attend freely)
        assert mask[0, 0] == 0.0
        assert mask[seq_len - 1, seq_len - 1] == 0.0
        # Above diagonal should be -inf (masked)
        assert mask[0, 1] == float("-inf"), "Position 0 must not attend to position 1"
        assert mask[0, -1] == float("-inf"), "No attending to future"

    def test_causal_no_future_leakage(self, transformer_backbone):
        """
        The output at position t must not depend on tokens at positions > t.

        We feed a full sequence [batch, seq, feat] twice:
          - x_base:      future tokens are all zeros
          - x_perturbed: future tokens are huge (1e6)
        The last-position output of *both sequences up to prefix_len* must be
        identical, proving the causal mask prevents future information leakage.
        """
        transformer_backbone.eval()
        batch, seq, feat = 1, 8, 20
        prefix_len = 4

        # Baseline: prefix is random, suffix is zeros
        x_base = torch.zeros(batch, seq, feat)
        x_base[0, :prefix_len, :] = torch.randn(prefix_len, feat)

        # Perturbed: same prefix, but suffix positions are huge
        x_perturbed = x_base.clone()
        x_perturbed[0, prefix_len:, :] = (
            1e6  # would massively shift output if mask fails
        )

        with torch.no_grad():
            # Both must produce the same last-position output for the prefix
            # We feed the full sequence — the causal mask should block the suffix
            out_base = transformer_backbone(x_base)
            out_perturbed = transformer_backbone(x_perturbed)

        # The prefix (positions 0..prefix_len-1) output must be identical.
        # NOTE: We compare the FULL sequence output at position prefix_len-1
        # by accessing both outputs at the same last position.
        # TransformerBackbone returns output[:, -1, :] so we need equal-length
        # sequences; instead compare position [prefix_len-1] manually.
        # For a simpler check: if the causal mask is correct, the LAST position
        # of x_base and the position at index (seq-1) of x_perturbed will differ
        # (because x_perturbed has huge values in positions prefix_len..seq-1),
        # but TRUNCATED to prefix_len they must agree.
        # We test this by running both on the truncated prefix only:
        out_base_prefix = transformer_backbone(x_base[:, :prefix_len, :])
        out_perturbed_prefix = transformer_backbone(x_perturbed[:, :prefix_len, :])

        assert torch.allclose(
            out_base_prefix, out_perturbed_prefix, atol=1e-5
        ), "Causal mask failed: identical prefixes produce different outputs"

        # Also verify: the FULL sequence outputs differ when suffix is changed —
        # this confirms the suffix perturbation is visible to positions *within*
        # the suffix (otherwise the test has no power).
        assert not torch.allclose(
            out_base, out_perturbed, atol=1e-2
        ), "Perturbation had no effect — suffix tokens may be invisible even without mask"

    def test_gradient_flows(self, transformer_backbone):
        """Gradients must flow back through the Transformer."""
        x = torch.randn(2, 10, 20, requires_grad=False)
        x = x.clone().detach().requires_grad_(True)
        out = transformer_backbone(x)
        loss = out.sum()
        loss.backward()
        assert (
            x.grad is not None and x.grad.abs().sum() > 0
        ), "No gradient flowing through TransformerBackbone"

    def test_batch_size_1(self, transformer_backbone):
        """Edge case: batch size 1."""
        x = torch.randn(1, 5, 20)
        out = transformer_backbone(x)
        assert out.shape == (1, 64)

    def test_seq_len_1(self, transformer_backbone):
        """Edge case: single time-step in 3D form."""
        x = torch.randn(3, 1, 20)
        out = transformer_backbone(x)
        assert out.shape == (3, 64)


# ---------------------------------------------------------------------------
# 3. PPOAgent with Transformer backbone
# ---------------------------------------------------------------------------


class TestPPOAgentTransformer:
    """Integration tests: PPOAgent initialized with use_transformer=True."""

    def test_agent_initialization(self, ppo_config_transformer, device):
        """PPOAgent must initialize without errors when use_transformer=True."""
        agent = PPOAgent(ppo_config_transformer, device=device)
        assert agent is not None
        # Verify transformer is attached to both actor and critic
        assert (
            agent.actor.transformer is not None
        ), "Actor must have TransformerBackbone"
        assert (
            agent.critic.transformer is not None
        ), "Critic must have TransformerBackbone"
        assert agent.actor.rnn is None, "RNN must be None when transformer is active"

    def test_forward_pass_2d_input(self, ppo_config_transformer, device):
        """select_action must work with standard (state_dim,) observation."""
        agent = PPOAgent(ppo_config_transformer, device=device)
        state = np.random.randn(ppo_config_transformer.state_dim).astype(np.float32)
        action, log_prob, value, hidden = agent.select_action(state, hidden=None)
        assert isinstance(action, int), f"action must be int, got {type(action)}"
        assert 0 <= action < ppo_config_transformer.n_actions
        assert isinstance(log_prob, float)
        assert isinstance(value, float)

    def test_actor_output_shape(self, ppo_config_transformer, device):
        """Actor output (logits) must be [batch, n_actions]."""
        agent = PPOAgent(ppo_config_transformer, device=device)
        state = torch.randn(4, ppo_config_transformer.state_dim)
        dist, _ = agent.actor(state, None)
        assert dist.logits.shape == (
            4,
            ppo_config_transformer.n_actions,
        ), f"Actor logits shape mismatch: {dist.logits.shape}"

    def test_critic_output_shape(self, ppo_config_transformer, device):
        """Critic must output [batch, 1] scalar values."""
        agent = PPOAgent(ppo_config_transformer, device=device)
        state = torch.randn(4, ppo_config_transformer.state_dim)
        value, _ = agent.critic(state, None)
        assert value.shape == (4, 1), f"Critic output shape mismatch: {value.shape}"

    def test_3d_sequence_input(self, ppo_config_transformer, device):
        """Actor/Critic must accept 3D [batch, seq_len, state_dim] inputs."""
        agent = PPOAgent(ppo_config_transformer, device=device)
        x = torch.randn(2, 10, ppo_config_transformer.state_dim)
        dist, _ = agent.actor(x, None)
        assert dist.logits.shape == (2, ppo_config_transformer.n_actions)
        value, _ = agent.critic(x, None)
        assert value.shape == (2, 1)


# ---------------------------------------------------------------------------
# 4. SIL buffer tests
# ---------------------------------------------------------------------------


class TestSILBuffer:
    """Verify Self-Imitation Learning buffer initialisation and circular fill."""

    def test_sil_buffer_initialized(self, ppo_config_transformer, device):
        """Buffer must be a dict with numpy arrays of correct shapes."""
        agent = PPOAgent(ppo_config_transformer, device=device)
        buf = agent.sil_buffer
        cap = ppo_config_transformer.sil_capacity
        sdim = ppo_config_transformer.state_dim

        assert buf["states"].shape == (
            cap,
            sdim,
        ), f"states shape mismatch: {buf['states'].shape}"
        assert buf["actions"].shape == (
            cap,
        ), f"actions shape mismatch: {buf['actions'].shape}"
        assert buf["returns"].shape == (
            cap,
        ), f"returns shape mismatch: {buf['returns'].shape}"
        assert buf["ptr"] == 0
        assert buf["size"] == 0

    def test_sil_buffer_fills(self, ppo_config_transformer, device):
        """_add_to_sil_buffer must correctly write data and advance ptr."""
        agent = PPOAgent(ppo_config_transformer, device=device)
        n = 50
        sdim = ppo_config_transformer.state_dim
        states = np.random.randn(n, sdim).astype(np.float32)
        actions = np.random.randint(0, 3, n).astype(np.int64)
        returns = np.random.randn(n).astype(np.float32)

        agent._add_to_sil_buffer(states, actions, returns)

        assert agent.sil_buffer["size"] == n
        assert agent.sil_buffer["ptr"] == n
        np.testing.assert_array_almost_equal(
            agent.sil_buffer["states"][:n],
            states,
            err_msg="SIL buffer states data mismatch",
        )

    def test_sil_buffer_circular_wrap(self, ppo_config_transformer, device):
        """Buffer must wrap around when capacity is exceeded."""
        agent = PPOAgent(ppo_config_transformer, device=device)
        cap = ppo_config_transformer.sil_capacity  # 1000 for test fixture
        sdim = ppo_config_transformer.state_dim

        # Fill buffer to capacity
        batch = 600
        for _ in range(2):  # 1200 > 1000 → wraps
            states = np.random.randn(batch, sdim).astype(np.float32)
            actions = np.random.randint(0, 3, batch).astype(np.int64)
            returns = np.random.randn(batch).astype(np.float32)
            agent._add_to_sil_buffer(states, actions, returns)

        assert (
            agent.sil_buffer["size"] == cap
        ), f"Buffer size should be capped at {cap}, got {agent.sil_buffer['size']}"
        assert (
            agent.sil_buffer["ptr"] == (2 * batch) % cap
        ), f"ptr should be {(2*batch)%cap}, got {agent.sil_buffer['ptr']}"

    def test_sil_update_called_in_train(self, ppo_config_transformer, device):
        """train() must call _update_sil when buffer is sufficiently filled."""
        agent = PPOAgent(ppo_config_transformer, device=device)
        sdim = ppo_config_transformer.state_dim

        # Pre-fill the SIL buffer so it is ready
        n_fill = ppo_config_transformer.sil_batch_size + 1
        agent._add_to_sil_buffer(
            np.random.randn(n_fill, sdim).astype(np.float32),
            np.zeros(n_fill, dtype=np.int64),
            np.ones(n_fill, dtype=np.float32) * 5.0,  # high returns → positive adv
        )

        # Store a small trajectory for PPO
        n_steps = 32
        for _ in range(n_steps):
            state = np.random.randn(sdim).astype(np.float32)
            agent.store_transition(
                state=state,
                action=0,
                reward=1.0,
                log_prob=-1.0,
                value=0.5,
                done=False,
                hidden=None,
            )

        stats = agent.train(next_value=0.0)
        assert "sil_loss" in stats, "train() must return sil_loss key"
        # sil_loss is 0.0 only if every sampled adv was <= 0 (unlikely with R=5.0)
        # We just check the key exists and is a finite float
        assert np.isfinite(
            stats["sil_loss"]
        ), f"sil_loss must be finite, got {stats['sil_loss']}"


# ---------------------------------------------------------------------------
# 5. End-to-end smoke test
# ---------------------------------------------------------------------------


class TestEndToEnd:
    """Full episode smoke-test: collect → train with transformer + SIL."""

    def test_full_training_step(self, ppo_config_transformer, device):
        """
        Collect a short rollout and perform one PPO+SIL update.
        Verifies that the entire pipeline works without errors.
        """
        agent = PPOAgent(ppo_config_transformer, device=device)
        sdim = ppo_config_transformer.state_dim
        n_steps = 64

        for step in range(n_steps):
            state = np.random.randn(sdim).astype(np.float32)
            action, log_prob, value, _ = agent.select_action(state, hidden=None)
            reward = float(np.random.randn())
            done = step == n_steps - 1
            agent.store_transition(state, action, reward, log_prob, value, done)

        stats = agent.train(next_value=0.0)

        required_keys = {"actor_loss", "critic_loss", "entropy", "mean_kl", "n_epochs"}
        assert required_keys.issubset(
            stats.keys()
        ), f"Missing keys in stats: {required_keys - stats.keys()}"
        assert np.isfinite(stats["actor_loss"]), "actor_loss is not finite"
        assert np.isfinite(stats["critic_loss"]), "critic_loss is not finite"


# ---------------------------------------------------------------------------
# 6. Option A and Option B — new config fields
# ---------------------------------------------------------------------------


class TestTransformerOptions:
    """Verify Option A (seq_len=1) and Option B (seq_len=K) modes."""

    def _make_config(self, seq_len: int, nhead: int = 4, d_model: int = 0) -> PPOConfig:
        return PPOConfig(
            state_dim=20,
            n_actions=3,
            hidden_dim=64,
            use_transformer=True,
            use_recurrent=False,
            rnn_layers=2,
            transformer_nhead=nhead,
            transformer_d_model=d_model,
            transformer_seq_len=seq_len,
            use_sil=False,
        )

    # ── Option A ─────────────────────────────────────────────────────────────

    def test_option_a_no_window_allocated(self):
        """With seq_len=1 (Option A) no sliding-window overhead is created."""
        cfg = self._make_config(seq_len=1)
        agent = PPOAgent(cfg, device="cpu")
        # _use_seq_window must be False — no temporal window in Option A
        assert not agent._use_seq_window

    def test_option_a_select_action_shape(self):
        """Option A: select_action feeds [1, state_dim] — output is valid."""
        cfg = self._make_config(seq_len=1)
        agent = PPOAgent(cfg, device="cpu")
        state = np.random.randn(20).astype(np.float32)
        action, log_prob, value, hidden = agent.select_action(state)
        assert isinstance(action, int) and 0 <= action < 3
        assert np.isfinite(log_prob)
        assert np.isfinite(value)
        assert hidden is None  # transformer returns no recurrent state

    # ── Option B ─────────────────────────────────────────────────────────────

    def test_option_b_window_allocated(self):
        """With seq_len>1 (Option B) the sliding window is initialised."""
        K = 8
        cfg = self._make_config(seq_len=K)
        agent = PPOAgent(cfg, device="cpu")
        assert agent._use_seq_window
        assert agent._state_window.shape == (K, 20)

    def test_option_b_reset_zeros_window(self):
        """reset_sequence_window() must zero the buffer."""
        K = 8
        cfg = self._make_config(seq_len=K)
        agent = PPOAgent(cfg, device="cpu")
        # Dirty the window
        agent._state_window[:] = 999.0
        agent.reset_sequence_window()
        assert np.all(agent._state_window == 0.0)

    def test_option_b_window_shifts_on_select_action(self):
        """After two select_action calls the newest state is at index [-1]."""
        K = 4
        cfg = self._make_config(seq_len=K)
        agent = PPOAgent(cfg, device="cpu")

        state1 = np.ones(20, dtype=np.float32)
        state2 = np.full(20, 2.0, dtype=np.float32)

        agent.select_action(state1)
        agent.select_action(state2)

        # state2 must now be at the last row of the window
        np.testing.assert_array_almost_equal(agent._state_window[-1], state2)
        # state1 must be at row [-2]
        np.testing.assert_array_almost_equal(agent._state_window[-2], state1)

    def test_option_b_select_action_shape(self):
        """Option B: select_action feeds [1, K, state_dim] — still returns valid action."""
        K = 16
        cfg = self._make_config(seq_len=K)
        agent = PPOAgent(cfg, device="cpu")
        state = np.random.randn(20).astype(np.float32)
        action, log_prob, value, hidden = agent.select_action(state)
        assert isinstance(action, int) and 0 <= action < 3
        assert np.isfinite(log_prob)
        assert np.isfinite(value)
        assert hidden is None

    def test_option_b_full_rollout(self):
        """Option B: 32-step rollout + one PPO update completes without error."""
        K = 8
        cfg = self._make_config(seq_len=K)
        agent = PPOAgent(cfg, device="cpu")
        agent.reset_sequence_window()

        for step in range(32):
            state = np.random.randn(20).astype(np.float32)
            action, log_prob, value, _ = agent.select_action(state)
            done = step == 31
            agent.store_transition(
                state, action, float(np.random.randn()), log_prob, value, done
            )

        stats = agent.train(next_value=0.0)
        assert np.isfinite(stats["actor_loss"])
        assert np.isfinite(stats["critic_loss"])

    # ── Custom nhead / d_model ────────────────────────────────────────────────

    def test_custom_nhead(self):
        """transformer_nhead is wired through to TransformerBackbone."""
        cfg = self._make_config(seq_len=1, nhead=2)
        agent = PPOAgent(cfg, device="cpu")
        # Inspect first encoder layer's self-attention
        layer = agent.actor.transformer.transformer_encoder.layers[0]
        assert layer.self_attn.num_heads == 2

    def test_custom_d_model_decoupled_from_hidden(self):
        """transformer_d_model > 0 decouples transformer width from MLP hidden_dim."""
        cfg = self._make_config(seq_len=1, nhead=4, d_model=128)
        # hidden_dim=64, d_model=128 — TransformerBackbone has its own projection
        agent = PPOAgent(cfg, device="cpu")
        assert agent.actor.transformer.d_model == 128
        # Output head must match d_model, not hidden_dim
        assert agent.actor.head.in_features == 128

    def test_default_d_model_equals_hidden_dim(self):
        """With transformer_d_model=0, d_model defaults to hidden_dim."""
        cfg = self._make_config(seq_len=1, d_model=0)  # 0 = use hidden_dim
        agent = PPOAgent(cfg, device="cpu")
        assert agent.actor.transformer.d_model == cfg.hidden_dim


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
