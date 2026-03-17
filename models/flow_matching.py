"""
Baseline Flow Matching for Time-Series Forecasting.

Main Author: Ka Wong Hui

Implements the standard Conditional Flow Matching (CFM) with an isotropic
Gaussian prior N(0, I). This establishes the accuracy baseline that requires
many NFEs. The improved variant (OU prior + OT) is in models/improved.py.

    L = E_{t, x_0, x_1} [ ||v_θ(x_t, t | context) - u_t(x_t | x_0, x_1)||² ]

where:
    u_t(x_t | x_0, x_1) = x_1 - x_0   (conditional velocity)
    x_t = (1 - t) x_0 + t x_1          (linear interpolation)
    x_0 ~ N(0, I)                       (Gaussian prior)

Reference: Lipman et al. (2023), "Flow Matching for Generative Modeling"
"""

import torch
import torch.nn as nn
from typing import Dict

from models.tcn import TCNVelocityField


class BaselineFlowMatching(nn.Module):
    """
    Flow Matching with isotropic Gaussian prior N(0, I).

    This is the standard Conditional Flow Matching setup where:
        - x_0 ~ N(0, I)           (source/noise)
        - x_1 ~ p_data            (target/data)
        - x_t = (1-t)·x_0 + t·x_1 (linear interpolation)
        - u_t = x_1 - x_0         (conditional velocity)
    """

    def __init__(self, config: Dict):
        super().__init__()
        tcn_cfg = config["tcn"]
        data_cfg = config["data"]

        self.prediction_horizon = data_cfg["prediction_horizon"]

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
        Compute the Conditional Flow Matching loss.

        Args:
            context: (batch, context_length, 1) past observations.
            target:  (batch, prediction_horizon, 1) future observations (x_1).

        Returns:
            Scalar loss value.
        """
        batch_size = target.shape[0]
        device = target.device

        # Sample source from Gaussian prior
        x_0 = torch.randn_like(target)

        # Sample time uniformly
        t = torch.rand(batch_size, device=device)

        # Linear interpolation: x_t = (1 - t) * x_0 + t * x_1
        t_expanded = t[:, None, None]  # (batch, 1, 1)
        x_t = (1 - t_expanded) * x_0 + t_expanded * target

        # Target velocity: u_t = x_1 - x_0
        target_velocity = target - x_0

        # Predicted velocity
        pred_velocity = self.velocity_net(x_t, t, context)

        # L2 loss
        loss = torch.mean((pred_velocity - target_velocity) ** 2)
        return loss

    @torch.no_grad()
    def sample(
        self,
        context: torch.Tensor,
        nfe: int = 16,
        return_trajectory: bool = False,
    ) -> torch.Tensor:
        """
        Generate predictions by solving the ODE from t=0 to t=1.

        Args:
            context: (batch, context_length, 1) past observations.
            nfe: Number of function evaluations (Euler steps).
            return_trajectory: If True, return all intermediate steps.

        Returns:
            predictions: (batch, prediction_horizon, 1)
            or trajectory: (nfe+1, batch, prediction_horizon, 1) if return_trajectory
        """
        batch_size = context.shape[0]
        device = context.device

        # Start from Gaussian noise
        x = torch.randn(batch_size, self.prediction_horizon, 1, device=device)

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


def build_model(config: Dict, model_type: str = "baseline") -> nn.Module:
    """
    Factory function to create the appropriate Flow Matching model.

    Args:
        config: Full configuration dictionary.
        model_type: 'baseline' or 'improved'.

    Returns:
        Flow Matching model.
    """
    if model_type == "baseline":
        return BaselineFlowMatching(config)
    elif model_type == "improved":
        from models.improved import ImprovedFlowMatching
        return ImprovedFlowMatching(config)
    else:
        raise ValueError(f"Unknown model type: {model_type}. Use 'baseline' or 'improved'.")
