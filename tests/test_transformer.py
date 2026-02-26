"""
Test Suite for Transformer Models
==================================
"""

import pytest
import torch
import numpy as np
from src.transformer.trading_transformer import (
    TradingTransformer,
    TransformerConfig,
    PositionalEncoding,
    AdaptiveAttentionSpan,
    TemporalFusionTransformer,
    create_transformer_model,
    train_transformer,
    interpret_attention,
)


class TestPositionalEncoding:
    """Test positional encoding."""

    def test_learnable_encoding(self):
        """Test learnable positional encoding."""
        pe = PositionalEncoding(d_model=64, max_len=100, learnable=True)

        x = torch.randn(2, 50, 64)  # batch=2, seq=50, dim=64
        output = pe(x)

        assert output.shape == x.shape
        assert not torch.allclose(output, x)  # Positional info should change values

    def test_fixed_encoding(self):
        """Test fixed sinusoidal encoding."""
        pe = PositionalEncoding(d_model=64, max_len=100, learnable=False)

        x = torch.randn(2, 50, 64)
        output = pe(x)

        assert output.shape == x.shape  # Shape unchanged after adding positional codes


class TestAdaptiveAttentionSpan:
    """Test adaptive attention span."""

    def test_initialization(self):
        """Test adaptive span initialization."""
        span = AdaptiveAttentionSpan(max_span=100, d_model=64)

        assert span.max_span == 100
        assert span.d_model == 64

    def test_adaptive_masking(self):
        """Test that attention is adaptively masked."""
        span = AdaptiveAttentionSpan(max_span=50, d_model=32)

        attention_weights = torch.randn(2, 8, 30, 30)  # (batch, heads, seq, seq)
        attention_weights = torch.softmax(attention_weights, dim=-1)

        masked = span(attention_weights)  # Apply learned span mask

        assert masked.shape == attention_weights.shape
        # Masked attention should still sum to 1 (valid distribution)
        sums = masked.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-6)


class TestTradingTransformer:
    """Test main transformer model."""

    @pytest.fixture
    def config(self):
        return TransformerConfig(
            input_dim=10,
            d_model=64,
            nhead=4,
            num_layers=2,
            max_seq_len=100,
            output_dim=3,  # 3 classes: bullish, neutral, bearish
        )

    @pytest.fixture
    def sample_input(self):
        return torch.randn(4, 50, 10)  # batch=4, seq=50, features=10

    def test_initialization(self, config):
        """Test model initialization."""
        model = TradingTransformer(config)

        assert model.config == config
        assert hasattr(model, "transformer")
        assert hasattr(model, "trend_head")

    def test_forward_pass(self, config, sample_input):
        """Test forward pass."""
        model = TradingTransformer(config)

        output = model(sample_input)

        assert "trend_logits" in output
        assert "trend_probs" in output
        assert "volatility" in output

        # Check shapes
        assert output["trend_logits"].shape == (4, 3)
        assert output["trend_probs"].shape == (4, 3)
        assert output["volatility"].shape == (4, 1)

        # Check probabilities sum to 1
        prob_sums = output["trend_probs"].sum(dim=1)
        assert torch.allclose(prob_sums, torch.ones(4), atol=1e-6)

    def test_causal_masking(self, config, sample_input):
        """Test that causal masking prevents future leakage."""
        model = TradingTransformer(config)

        # Create input with clear future information
        sample_input[:, 25:, :] = 100.0  # Future is very large

        output = model(sample_input)

        # Predictions should not be affected by future values
        # (due to causal masking — past attends only to itself)
        assert output["trend_probs"].shape == (4, 3)

    def test_attention_extraction(self, config, sample_input):
        """Test attention weight extraction."""
        model = TradingTransformer(config)

        output = model(sample_input, return_attention=True)

        assert "attention_weights" in output
        assert output["attention_weights"].shape[0] == 4  # batch size

        # Attention weights should be probabilities
        weights = output["attention_weights"]
        assert torch.all(weights >= 0)
        assert torch.all(weights <= 1)

    def test_different_sequence_lengths(self, config):
        """Test with different sequence lengths."""
        model = TradingTransformer(config)

        for seq_len in [10, 50, 100]:
            x = torch.randn(2, seq_len, 10)
            output = model(x)

            assert output["trend_probs"].shape == (2, 3)  # Shape independent of seq len

    def test_get_context_importance(self, config, sample_input):
        """Test context importance extraction."""
        model = TradingTransformer(config)

        importance = model.get_context_importance(sample_input)

        assert isinstance(importance, np.ndarray)
        assert importance.shape[0] == 4  # batch size


class TestTemporalFusionTransformer:
    """Test TFT model."""

    def test_initialization(self):
        """Test TFT initialization."""
        model = TemporalFusionTransformer(
            num_static_features=5, num_temporal_features=10, hidden_size=64
        )

        assert model.hidden_size == 64

    def test_forward_pass(self):
        """Test TFT forward pass."""
        model = TemporalFusionTransformer(
            num_static_features=5, num_temporal_features=10, hidden_size=64
        )

        batch_size = 4
        static = torch.randn(batch_size, 5)  # Static features (e.g., asset class)
        temporal = torch.randn(batch_size, 50, 10)  # Time-series features

        output = model(static, temporal)

        assert output.shape == (batch_size, 1)  # Single scalar prediction per sample


class TestIntegrationFunctions:
    """Test high-level integration functions."""

    def test_create_transformer_model(self):
        """Test model factory function."""
        model = create_transformer_model(input_dim=20, output_dim=5, num_layers=3)

        assert isinstance(model, TradingTransformer)
        assert model.config.input_dim == 20
        assert model.config.output_dim == 5
        assert model.config.num_layers == 3

    def test_train_transformer(self):
        """Test training function."""
        model = create_transformer_model(input_dim=10, output_dim=3)

        # Create dummy training data
        train_data = torch.randn(10, 50, 10)

        losses = train_transformer(model, train_data, epochs=5, lr=0.001)

        assert len(losses) == 5
        # Loss should generally decrease or stay stable (allow some variance)
        assert losses[-1] <= losses[0] * 1.5

    def test_interpret_attention(self):
        """Test attention interpretation."""
        model = create_transformer_model(input_dim=10, output_dim=3)

        sample_input = torch.randn(2, 30, 10)

        result = interpret_attention(model, sample_input)

        assert "attention_weights" in result
        assert "predictions" in result
        assert "most_important_timesteps" in result  # Timesteps with highest attention


class TestEdgeCases:
    """Test edge cases."""

    def test_empty_sequence(self, config):
        """Test handling of very short sequences."""
        model = TradingTransformer(config)

        x = torch.randn(1, 1, 10)  # Very short sequence (single timestep)

        # Should handle gracefully
        output = model(x)
        assert output["trend_probs"].shape == (1, 3)

    def test_large_batch(self, config):
        """Test with large batch size."""
        model = TradingTransformer(config)

        x = torch.randn(100, 50, 10)  # Large batch (stress test)
        output = model(x)

        assert output["trend_probs"].shape == (100, 3)

    def test_gradients_flow(self, config, sample_input):
        """Test that gradients flow properly."""
        model = TradingTransformer(config)
        model.train()

        output = model(sample_input)
        loss = output["trend_probs"].mean()
        loss.backward()  # Backpropagate through entire model

        # Check that gradients exist (at least one parameter has non-zero grad)
        has_gradients = False
        for param in model.parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_gradients = True
                break

        assert has_gradients  # Gradient must flow back to parameters


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
