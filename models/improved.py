"""
Improved Flow Matching with Ornstein-Uhlenbeck Prior and Optimal Transport.

Main Author: Min-Han Yeh

This module implements the improved variant of Conditional Flow Matching
that replaces the isotropic Gaussian prior with a structured OU process
prior and uses mini-batch Optimal Transport coupling to simplify the
transport paths.

Key differences from the baseline:
    - x_0 ~ OU process        (informed/structured prior)
    - OT coupling between x_0 and x_1 to simplify transport paths
    - Hypothesized to require fewer NFEs for equivalent accuracy

Reference: Tong et al. (2023), "Improving and Generalizing Flow-Based
           Generative Models with Minibatch Optimal Transport"
"""

import torch
import torch.nn as nn
from typing import Dict

from models.tcn import TCNVelocityField
from models.ou_prior import OUPrior, compute_ot_coupling


class ImprovedFlowMatching(nn.Module):
    """
    Flow Matching with Ornstein-Uhlenbeck process prior and OT coupling.

    Key differences from baseline:
        - x_0 ~ OU process        (informed/structured prior)
        - OT coupling between x_0 and x_1 to simplify transport paths
        - Same interpolation and velocity computation

    The OU prior generates temporally correlated noise that is structurally
    closer to the data distribution, hypothetically requiring fewer NFEs.
    """

    def __init__(self, config: Dict):
        super().__init__()
        tcn_cfg = config["tcn"]
        data_cfg = config["data"]
        ou_cfg = config["ou_prior"]
        ot_cfg = config["optimal_transport"]

        self.prediction_horizon = data_cfg["prediction_horizon"]
        self.use_ot = ot_cfg["enabled"]
        self.ot_reg = ot_cfg["reg"]
        self.ot_max_iter = ot_cfg["max_iter"]

        # OU prior sampler
        self.ou_prior = OUPrior(
            theta=ou_cfg["theta"],
            mu=ou_cfg["mu"],
            sigma=ou_cfg["sigma"],
            n_steps=ou_cfg["n_steps"],
        )

        # Same TCN velocity network
        self.velocity_net = TCNVelocityField(
            in_channels=tcn_cfg["in_channels"],
            hidden_channels=tcn_cfg["hidden_channels"],
            num_layers=tcn_cfg["num_layers"],
            kernel_size=tcn_cfg["kernel_size"],
            dropout=tcn_cfg["dropout"],
            time_embed_dim=tcn_cfg["time_embed_dim"],
            context_length=data_cfg["context_length"],
            prediction_horizon=data_cfg["prediction_horizon"],
        )

    def compute_loss(
        self, context: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the CFM loss with OU prior and OT coupling.

        Args:
            context: (batch, context_length, 1)
            target:  (batch, prediction_horizon, 1)

        Returns:
            Scalar loss.
        """
        batch_size = target.shape[0]
        device = target.device

        # Sample from OU prior instead of Gaussian
        x_0 = self.ou_prior.sample(
            batch_size=batch_size,
            seq_len=self.prediction_horizon,
            channels=1,
            device=device,
        )

        # Apply OT coupling
        if self.use_ot:
            x_0, target_coupled = compute_ot_coupling(
                x_0, target, reg=self.ot_reg, max_iter=self.ot_max_iter
            )
        else:
            target_coupled = target

        # Sample time uniformly
        t = torch.rand(batch_size, device=device)

        # Linear interpolation
        t_expanded = t[:, None, None]
        x_t = (1 - t_expanded) * x_0 + t_expanded * target_coupled

        # Target velocity: u_t = x_1 - x_0
        target_velocity = target_coupled - x_0

        # Predicted velocity
        pred_velocity = self.velocity_net(x_t, t, context)

        # L2 loss
        loss = torch.mean((pred_velocity - target_velocity) ** 2)
        return loss

    @torch.no_grad()
    def sample(
        self,
        context: torch.Tensor,
        nfe: int = 4,
        return_trajectory: bool = False,
    ) -> torch.Tensor:
        """
        Generate predictions by solving ODE from OU prior.

        Args:
            context: (batch, context_length, 1)
            nfe: Number of Euler steps (default lower than baseline).
            return_trajectory: If True, return all intermediate steps.

        Returns:
            predictions or trajectory.
        """
        batch_size = context.shape[0]
        device = context.device

        # Start from OU prior (not Gaussian noise)
        x = self.ou_prior.sample(
            batch_size=batch_size,
            seq_len=self.prediction_horizon,
            channels=1,
            device=device,
        )

        dt = 1.0 / nfe
        trajectory = [x.clone()] if return_trajectory else None

        for i in range(nfe):
            t = torch.full((batch_size,), i * dt, device=device)
            v = self.velocity_net(x, t, context)
            x = x + v * dt
            if return_trajectory:
                trajectory.append(x.clone())

        if return_trajectory:
            return torch.stack(trajectory, dim=0)
        return x
