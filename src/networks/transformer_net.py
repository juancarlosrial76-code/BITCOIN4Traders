import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding compatible with batch_first=True tensors.

    The positional encoding tensor is stored in batch_first layout:
        pe shape: [1, max_len, d_model]

    This means x (shape [batch, seq_len, d_model]) can be added directly
    without any permutation, which is required when the TransformerEncoder
    uses batch_first=True.
    """

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        # Build standard sinusoidal table in batch_first format: [1, max_len, d_model]
        position = torch.arange(max_len).unsqueeze(1)  # [max_len, 1]
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )  # [d_model/2]
        pe = torch.zeros(1, max_len, d_model)  # [1, max_len, d_model]
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape [batch_size, seq_len, d_model]  (batch_first=True)
        Returns:
            Tensor of same shape with positional information added.
        """
        # pe[:, :seq_len, :] broadcasts over the batch dimension
        x = x + self.pe[:, : x.size(1), :]
        return x


class TransformerBackbone(nn.Module):
    """
    Causally-masked Transformer feature extractor for RL agents.

    Uses batch_first=True throughout and applies a causal (upper-triangular)
    attention mask so that position t can only attend to positions <= t.
    This prevents future information leakage — essential for online RL where
    the agent processes the sequence left-to-right at inference time.

    Input shapes accepted:
        2D: [batch_size, input_dim]          — treated as a single time-step
        3D: [batch_size, seq_len, input_dim] — full sequence

    Output shape:
        [batch_size, d_model] — representation of the *last* sequence position
        (causally valid: encodes only information up to and including t=seq_len-1)
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model

        # Linear projection: input_dim → d_model
        self.input_projection = nn.Linear(input_dim, d_model)

        # Positional encoding in batch_first layout
        self.pos_encoder = PositionalEncoding(d_model)

        # Transformer encoder with batch_first=True so tensors stay in
        # [batch, seq, d_model] layout throughout — no permute() needed.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)

    def _causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Generate a causal (autoregressive) attention mask.

        Returns an additive mask of shape [seq_len, seq_len] where future
        positions are set to -inf so softmax zeroes them out.  This is the
        same convention as nn.Transformer.generate_square_subsequent_mask().
        """
        # Upper-triangular True → positions to MASK (attend to future)
        mask = torch.triu(
            torch.ones(seq_len, seq_len, device=device), diagonal=1
        ).bool()
        # Convert bool mask to additive float mask (-inf for masked positions)
        additive_mask = torch.zeros(seq_len, seq_len, device=device)
        additive_mask = additive_mask.masked_fill(mask, float("-inf"))
        return additive_mask

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with causal masking.

        Args:
            src: [batch_size, input_dim]           — single step (2D)
                 [batch_size, seq_len, input_dim]  — sequence   (3D)
        Returns:
            [batch_size, d_model] — last-position context vector.
        """
        # Handle 2D input (single time-step) — treat as seq_len=1
        if src.dim() == 2:
            src = src.unsqueeze(1)  # [batch, 1, input_dim]

        # Project to d_model and scale (standard Transformer convention)
        src = self.input_projection(src) * math.sqrt(self.d_model)

        # Add positional encodings (batch_first: [batch, seq_len, d_model])
        src = self.pos_encoder(src)

        # Build causal mask for this sequence length
        seq_len = src.size(1)
        causal_mask = self._causal_mask(seq_len, src.device)

        # Transformer encoder with causal masking
        output = self.transformer_encoder(src, mask=causal_mask)
        # output shape: [batch_size, seq_len, d_model]

        # Return the last position — causally valid representation of the full
        # sequence up to t=seq_len-1 (identical to how GPT/decoder models work).
        return output[:, -1, :]  # [batch_size, d_model]
