"""
Temporal Convolutional Network (TCN) for Flow Matching velocity field.

The TCN serves as the velocity field v_θ(x_t, t | context) that predicts
the vector field for the ODE solver. It is conditioned on:
  - x_t: the noisy sample at time t  (prediction_horizon, 1)
  - t:   the flow time ∈ [0, 1]
  - context: the past observations   (context_length, 1)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal positional embedding for the flow time t ∈ [0, 1]."""

    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: (batch_size,) or (batch_size, 1) flow time values in [0, 1].

        Returns:
            (batch_size, embed_dim) sinusoidal embedding.
        """
        t = t.view(-1)
        half_dim = self.embed_dim // 2
        emb_scale = math.log(10000.0) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device, dtype=t.dtype) * -emb_scale)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb


class CausalConv1d(nn.Module):
    """1D causal convolution with left-padding."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int = 1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size, dilation=dilation
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, (self.padding, 0))
        return self.conv(x)


class TCNResidualBlock(nn.Module):
    """
    A single TCN residual block with:
      - Two causal dilated convolutions
      - Weight normalization
      - Dropout
      - Residual connection (with 1x1 conv if channel mismatch)
      - FiLM conditioning from time embedding
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float = 0.1,
        time_embed_dim: int = 64,
    ):
        super().__init__()

        self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size, dilation)
        nn.utils.parametrizations.weight_norm(self.conv1.conv, name="weight")
        self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size, dilation)
        nn.utils.parametrizations.weight_norm(self.conv2.conv, name="weight")
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

        # FiLM: affine modulation from time embedding
        self.time_proj = nn.Linear(time_embed_dim, out_channels * 2)

        # Residual connection
        self.residual = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, channels, seq_len)
            t_emb: (batch, time_embed_dim) time embedding

        Returns:
            (batch, out_channels, seq_len)
        """
        residual = self.residual(x)

        out = self.conv1(x)
        out = self.activation(out)
        out = self.dropout(out)

        out = self.conv2(out)

        # FiLM conditioning: scale and shift
        film = self.time_proj(t_emb)  # (batch, 2 * out_channels)
        gamma, beta = film.chunk(2, dim=-1)  # each (batch, out_channels)
        out = out * (1 + gamma.unsqueeze(-1)) + beta.unsqueeze(-1)

        out = self.activation(out)
        out = self.dropout(out)

        return out + residual


class TCNVelocityField(nn.Module):
    """
    TCN-based velocity field for Flow Matching.

    Input:  x_t (noisy prediction) concatenated with context
    Output: predicted velocity v_θ of shape (batch, prediction_horizon, 1)

    Architecture:
        context + x_t  -->  [TCN blocks with dilated causal convs]  -->  velocity
        t  -->  [sinusoidal embedding]  -->  FiLM conditioning into each block
    """

    def __init__(
        self,
        in_channels: int = 1,
        hidden_channels: int = 128,
        num_layers: int = 6,
        kernel_size: int = 3,
        dropout: float = 0.1,
        time_embed_dim: int = 64,
        context_length: int = 256,
        prediction_horizon: int = 64,
    ):
        super().__init__()
        self.context_length = context_length
        self.prediction_horizon = prediction_horizon

        # Time embedding
        self.time_embed = SinusoidalTimeEmbedding(time_embed_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_embed_dim, time_embed_dim * 2),
            nn.GELU(),
            nn.Linear(time_embed_dim * 2, time_embed_dim),
        )

        # Input projection
        self.input_proj = nn.Conv1d(in_channels, hidden_channels, 1)

        # TCN blocks with exponentially increasing dilation
        self.blocks = nn.ModuleList()
        for i in range(num_layers):
            dilation = 2 ** i
            self.blocks.append(
                TCNResidualBlock(
                    in_channels=hidden_channels,
                    out_channels=hidden_channels,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                    time_embed_dim=time_embed_dim,
                )
            )

        # Output projection: map back to velocity in data space
        self.output_proj = nn.Sequential(
            nn.Conv1d(hidden_channels, hidden_channels // 2, 1),
            nn.GELU(),
            nn.Conv1d(hidden_channels // 2, in_channels, 1),
        )

    def forward(
        self, x_t: torch.Tensor, t: torch.Tensor, context: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x_t:     (batch, prediction_horizon, 1) noisy sample at time t
            t:       (batch,) flow time values in [0, 1]
            context: (batch, context_length, 1) past observations

        Returns:
            velocity: (batch, prediction_horizon, 1) predicted velocity field
        """
        # Time embedding
        t_emb = self.time_embed(t)
        t_emb = self.time_mlp(t_emb)  # (batch, time_embed_dim)

        # Concatenate context and x_t along time dimension
        # Shape: (batch, context_length + prediction_horizon, 1)
        combined = torch.cat([context, x_t], dim=1)

        # Transpose to (batch, channels, seq_len) for Conv1d
        combined = combined.permute(0, 2, 1)

        # Input projection
        h = self.input_proj(combined)

        # TCN blocks
        for block in self.blocks:
            h = block(h, t_emb)

        # Output projection
        h = self.output_proj(h)

        # Take only the prediction horizon part
        h = h[:, :, -self.prediction_horizon :]

        # Transpose back to (batch, prediction_horizon, 1)
        velocity = h.permute(0, 2, 1)

        return velocity
