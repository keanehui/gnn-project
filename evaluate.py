"""
Main evaluation entry point and metrics/visualization library.

Usage:
    python evaluate.py --checkpoint outputs/baseline/model/best.pt --model baseline --nfe 16
    python evaluate.py --checkpoint outputs/improved/model/best.pt --model improved --nfe 4
"""

import argparse
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server/GPU machines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

from data.dataset import create_dataloaders
from models.flow_matching import build_model

# Set style for plots
sns.set_theme(style="whitegrid", font_scale=1.2)
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.family": "serif",
})

# ==============================================================================
# Metrics
# ==============================================================================

def mae(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """Mean Absolute Error."""
    return torch.mean(torch.abs(predictions - targets)).item()


def rmse(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """Root Mean Square Error."""
    return torch.sqrt(torch.mean((predictions - targets) ** 2)).item()


def crps_gaussian(
    predictions: torch.Tensor,  # (n_samples, batch, horizon, channels)
    targets: torch.Tensor,      # (batch, horizon, channels)
) -> float:
    """
    Empirical CRPS (Continuous Ranked Probability Score).
    Lower CRPS = better probabilistic forecast.
    CRPS = E|X - y| - 0.5 * E|X - X'|
    """
    n_samples = predictions.shape[0]

    # E|X - y|: average absolute error across samples
    abs_errors = torch.mean(torch.abs(predictions - targets.unsqueeze(0)), dim=0)
    term1 = torch.mean(abs_errors).item()

    # E|X - X'|: expected absolute difference between pairs of samples
    if n_samples > 1:
        diffs = 0.0
        count = 0
        for i in range(min(n_samples, 20)):  # Limit pairs for efficiency
            for j in range(i + 1, min(n_samples, 20)):
                diffs += torch.mean(torch.abs(predictions[i] - predictions[j])).item()
                count += 1
        term2 = diffs / max(count, 1)
    else:
        term2 = 0.0

    return term1 - 0.5 * term2


@torch.no_grad()
def measure_latency(
    model: nn.Module,
    context: torch.Tensor,
    nfe: int,
    warmup_iters: int = 10,
    measure_iters: int = 100,
) -> Dict[str, float]:
    """Measure inference latency with proper GPU synchronization."""
    model.eval()
    device = next(model.parameters()).device

    # Warmup
    for _ in range(warmup_iters):
        _ = model.sample(context, nfe=nfe)

    # Measure
    latencies = []
    for _ in range(measure_iters):
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()

        _ = model.sample(context, nfe=nfe)

        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) * 1000  # ms

        latencies.append(elapsed)

    latencies = np.array(latencies)
    return {
        "mean_ms": float(np.mean(latencies)),
        "std_ms": float(np.std(latencies)),
        "min_ms": float(np.min(latencies)),
        "max_ms": float(np.max(latencies)),
    }


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    nfe: int = 16,
    n_samples: int = 10,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, float]:
    """
    Full evaluation of a Flow Matching model.
    Generates multiple stochastic samples per test case to compute MAE, RMSE, CRPS.
    """
    model.eval()

    all_mae = []
    all_rmse = []
    all_crps = []

    for context, target in test_loader:
        context = context.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        samples = []
        for _ in range(n_samples):
            pred = model.sample(context, nfe=nfe)
            samples.append(pred)

        samples = torch.stack(samples, dim=0)  # (n_samples, batch, horizon, channels)

        # Use the mean prediction for MAE/RMSE
        mean_pred = samples.mean(dim=0)
        
        all_mae.append(mae(mean_pred, target))
        all_rmse.append(rmse(mean_pred, target))
        all_crps.append(crps_gaussian(samples, target))

    return {
        "mae": float(np.mean(all_mae)),
        "rmse": float(np.mean(all_rmse)),
        "crps": float(np.mean(all_crps)),
    }

# ==============================================================================
# Visualization
# ==============================================================================

def plot_nfe_accuracy(
    results_df: pd.DataFrame,
    save_path: str,
    metric: str = "mae",
) -> None:
    """Plot NFE vs. accuracy (MAE or RMSE) for both models."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))

    for model_name in results_df["model"].unique():
        df = results_df[results_df["model"] == model_name]
        ax.plot(
            df["nfe"], df[metric], marker="o", linewidth=2, markersize=8, label=model_name,
        )

    ax.set_xlabel("Number of Function Evaluations (NFE)")
    ax.set_ylabel(metric.upper())
    ax.set_title(f"NFE vs. {metric.upper()} — Baseline vs. Improved (OU + OT)")
    ax.legend()
    ax.set_xscale("log", base=2)
    ax.set_xticks(results_df["nfe"].unique())
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved plot: {save_path}")


def plot_nfe_latency(
    results_df: pd.DataFrame,
    save_path: str,
) -> None:
    """Plot NFE vs. inference latency for both models."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))

    for model_name in results_df["model"].unique():
        df = results_df[results_df["model"] == model_name]
        ax.plot(
            df["nfe"], df["latency_mean_ms"], marker="s", linewidth=2, markersize=8, label=model_name,
        )
        if "latency_std_ms" in df.columns:
            ax.fill_between(
                df["nfe"],
                df["latency_mean_ms"] - df["latency_std_ms"],
                df["latency_mean_ms"] + df["latency_std_ms"],
                alpha=0.15,
            )

    ax.set_xlabel("Number of Function Evaluations (NFE)")
    ax.set_ylabel("Inference Latency (ms)")
    ax.set_title("NFE vs. Inference Latency")
    ax.legend()
    ax.set_xscale("log", base=2)
    ax.set_xticks(results_df["nfe"].unique())
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.axhline(y=1.0, color="red", linestyle="--", alpha=0.5, label="1 ms threshold")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved plot: {save_path}")


def plot_nfe_accuracy_latency(
    results_df: pd.DataFrame,
    save_path: str,
    metric: str = "mae",
) -> None:
    """Combined dual-axis plot: NFE vs Accuracy and NFE vs Latency."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()

    colors = {"Baseline": "#1f77b4", "Improved (OU+OT)": "#ff7f0e"}
    nfe_values = sorted(results_df["nfe"].unique())

    for model_name in results_df["model"].unique():
        df = results_df[results_df["model"] == model_name]
        color = colors.get(model_name, "#333333")

        ax1.plot(df["nfe"], df[metric], marker="o", linewidth=2, markersize=8, color=color, label=f"{model_name} ({metric.upper()})")
        ax2.plot(df["nfe"], df["latency_mean_ms"], marker="s", linewidth=2, markersize=8, linestyle="--", color=color, alpha=0.6, label=f"{model_name} (Latency)")

    ax1.set_xlabel("Number of Function Evaluations (NFE)")
    ax1.set_ylabel(f"{metric.upper()}", color="#1f77b4")
    ax2.set_ylabel("Latency (ms)", color="#ff7f0e")
    ax1.set_xscale("log", base=2)
    ax1.set_xticks(nfe_values)
    ax1.get_xaxis().set_major_formatter(plt.ScalarFormatter())

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
    ax1.set_title(f"Accuracy-Latency Trade-off: Baseline vs. Improved (OU + OT)")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved plot: {save_path}")


def plot_sample_predictions(
    context: np.ndarray,
    target: np.ndarray,
    predictions: np.ndarray,
    save_path: str,
    n_samples: int = 5,
    title: str = "Sample Predictions",
) -> None:
    """Plot context + ground truth vs. predicted trajectories."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 4))

    ctx_len = len(context)
    horizon = len(target)
    t_ctx = np.arange(ctx_len)
    t_pred = np.arange(ctx_len, ctx_len + horizon)

    ax.plot(t_ctx, context, color="black", linewidth=1.5, label="Context")
    ax.plot(t_pred, target, color="green", linewidth=2, label="Ground Truth")

    for i in range(min(n_samples, predictions.shape[0])):
        alpha = 0.3 if n_samples > 1 else 0.8
        label = "Predictions" if i == 0 else None
        ax.plot(t_pred, predictions[i], color="blue", alpha=alpha, linewidth=1, label=label)

    if predictions.shape[0] > 1:
        mean_pred = predictions.mean(axis=0)
        ax.plot(t_pred, mean_pred, color="red", linewidth=2, linestyle="--", label="Mean Prediction")

    ax.axvline(x=ctx_len, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Amplitude")
    ax.set_title(title)
    ax.legend(loc="upper left")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved plot: {save_path}")


def plot_training_loss(
    train_losses: List[float],
    val_losses: Optional[List[float]] = None,
    save_path: str = "training_loss.png",
) -> None:
    """Plot training (and optionally validation) loss curves."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(train_losses, label="Train Loss", linewidth=2)
    if val_losses:
        ax.plot(val_losses, label="Val Loss", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training Progress")
    ax.legend()
    ax.set_yscale("log")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved plot: {save_path}")


# ==============================================================================
# CLI Entry Point
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained Flow Matching model"
    )
    parser.add_argument(
        "--config", type=str, default="config.yaml",
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--model", type=str, default="baseline", choices=["baseline", "improved"],
        help="Model variant",
    )
    parser.add_argument(
        "--nfe", type=int, default=16,
        help="Number of function evaluations for ODE solver",
    )
    parser.add_argument(
        "--n_samples", type=int, default=10,
        help="Number of stochastic samples per test case",
    )
    parser.add_argument(
        "--plot", action="store_true",
        help="Generate sample prediction plots",
    )
    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # The new output structure is outputs/<model>/evaluate/
    eval_dir = os.path.join(config["training"].get("output_dir", "outputs"), args.model, "evaluate")
    os.makedirs(eval_dir, exist_ok=True)

    # Load data
    _, _, test_loader, dataset = create_dataloaders(config)

    # Load model
    model = build_model(config, args.model)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {args.model} ({n_params:,} params)")
    print(f"NFE: {args.nfe}")
    print(f"Samples per test case: {args.n_samples}")
    print(f"Evaluation output dir: {eval_dir}")

    # Evaluate
    print("\nEvaluating...")
    metrics_res = evaluate_model(
        model, test_loader, nfe=args.nfe,
        n_samples=args.n_samples, device=device,
    )

    print(f"\n{'='*40}")
    print(f"Results ({args.model}, NFE={args.nfe}):")
    print(f"  MAE:  {metrics_res['mae']:.6f}")
    print(f"  RMSE: {metrics_res['rmse']:.6f}")
    print(f"  CRPS: {metrics_res['crps']:.6f}")

    # Latency
    context_batch, _ = next(iter(test_loader))
    context_batch = context_batch.to(device)

    latency = measure_latency(model, context_batch, nfe=args.nfe)
    print(f"  Latency: {latency['mean_ms']:.3f} ± {latency['std_ms']:.3f} ms")
    print(f"{'='*40}")

    # Sample plots
    if args.plot:
        context, target = next(iter(test_loader))
        context = context.to(device)
        target = target.to(device)

        ctx = context[0:1]
        tgt = target[0]

        preds = []
        for _ in range(20):
            pred = model.sample(ctx, nfe=args.nfe)
            preds.append(pred[0].cpu().numpy().squeeze())
        preds = np.array(preds)

        plot_path = os.path.join(eval_dir, f"eval_nfe{args.nfe}.png")
        plot_sample_predictions(
            context=ctx[0].cpu().numpy().squeeze(),
            target=tgt.cpu().numpy().squeeze(),
            predictions=preds,
            save_path=plot_path,
            title=f"{args.model} (NFE={args.nfe})",
        )


if __name__ == "__main__":
    main()
