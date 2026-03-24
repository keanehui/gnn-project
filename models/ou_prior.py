"""
Ornstein-Uhlenbeck (OU) Process Prior & Optimal Transport Coupling.

The OU process is defined by the SDE:
    dx_t = θ(μ - x_t) dt + σ dW_t

where:
    θ = mean reversion speed
    μ = long-term mean
    σ = volatility

The stationary distribution is N(μ, σ² / (2θ)).

This module provides:
    1. Sampling from the OU process (for informed prior generation)
    2. Mini-batch Optimal Transport coupling between prior and data samples
"""

import torch
import numpy as np
from typing import Optional, Tuple


class OUPrior:
    """
    Ornstein-Uhlenbeck process sampler.

    Generates temporally correlated noise samples that are more informative
    than isotropic Gaussian noise for time-series Flow Matching.
    """

    def __init__(
        self,
        theta: float = 1.0,
        mu: float = 0.0,
        sigma: float = 0.5,
        n_steps: int = 100,
    ):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.n_steps = n_steps

    @property
    def stationary_std(self) -> float:
        """Standard deviation of the stationary distribution."""
        return self.sigma / np.sqrt(2 * self.theta)

    def sample(
        self,
        batch_size: int,
        seq_len: int,
        channels: int = 1,
        device: torch.device = torch.device("cpu"),
    ) -> torch.Tensor:
        """
        Sample paths from the OU process using Euler-Maruyama discretization.

        Args:
            batch_size: Number of independent paths.
            seq_len: Length of each path.
            channels: Number of channels (1 for univariate).
            device: Target device.

        Returns:
            (batch_size, seq_len, channels) tensor of OU process samples.
        """
        dt = 1.0 / self.n_steps  # Time step

        # Initialize from stationary distribution
        x = torch.randn(batch_size, channels, device=device) * self.stationary_std + self.mu

        # We need to generate seq_len output points
        # Use sub-stepping: each output point uses n_steps/seq_len sub-steps
        steps_per_output = max(1, self.n_steps // seq_len)
        outputs = []

        for i in range(seq_len):
            for _ in range(steps_per_output):
                dW = torch.randn_like(x) * np.sqrt(dt)
                x = x + self.theta * (self.mu - x) * dt + self.sigma * dW
            outputs.append(x.clone())

        # Stack: (batch_size, seq_len, channels)
        result = torch.stack(outputs, dim=1)
        return result

    def sample_stationary(
        self,
        batch_size: int,
        seq_len: int,
        channels: int = 1,
        device: torch.device = torch.device("cpu"),
    ) -> torch.Tensor:
        """
        Sample from the OU stationary distribution (independent across time).
        Faster than full process sampling but loses temporal correlations.

        Returns:
            (batch_size, seq_len, channels) tensor.
        """
        std = self.stationary_std
        return torch.randn(batch_size, seq_len, channels, device=device) * std + self.mu


def compute_ot_coupling(
    x0: torch.Tensor,
    x1: torch.Tensor,
    reg: float = 0.05,
    max_iter: int = 100,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute mini-batch Optimal Transport coupling between prior and data samples.

    Uses the Sinkhorn algorithm (via the POT library) to find an approximately
    optimal assignment between source (prior) and target (data) samples.

    Args:
        x0: (batch_size, seq_len, channels) source/prior samples.
        x1: (batch_size, seq_len, channels) target/data samples.
        reg: Entropic regularization parameter for Sinkhorn.
        max_iter: Maximum Sinkhorn iterations.

    Returns:
        (x0_coupled, x1_coupled): Reordered tensors according to the OT plan.
        x1 is always returned as-is; x0 is permuted to match.
    """
    try:
        import ot
    except ImportError:
        # Fallback: random coupling (no OT)
        print("[WARNING] POT not installed. Using random coupling instead of OT.")
        return x0, x1

    batch_size = x0.shape[0]

    # Flatten spatial dimensions for cost computation
    x0_flat = x0.reshape(batch_size, -1).detach().cpu().numpy()
    x1_flat = x1.reshape(batch_size, -1).detach().cpu().numpy()

    # Cost matrix: squared Euclidean distance
    M = ot.dist(x0_flat, x1_flat, metric="sqeuclidean")

    # Uniform marginals
    a = np.ones(batch_size) / batch_size
    b = np.ones(batch_size) / batch_size

    # Sinkhorn OT plan
    plan = ot.sinkhorn(a, b, M, reg=reg, numItermax=max_iter)

    # Use the OT plan to find the best permutation (argmax per row)
    permutation = plan.argmax(axis=1)

    # Reorder x0 according to the permutation
    x0_coupled = x0[torch.from_numpy(permutation).long().to(x0.device)]

    return x0_coupled, x1
