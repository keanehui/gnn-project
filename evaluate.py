"""
Main evaluation entry point and metrics/visualization library.

Usage:
    python evaluate.py --checkpoint outputs/baseline/model/best.pt --model baseline --nfe 16
    python evaluate.py --checkpoint outputs/improved/model/best.pt --model improved --nfe 4
"""

import argparse
import json
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
from utils.checkpoints import load_model_state

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
def collect_predictions(
    model: nn.Module,
    test_loader: DataLoader,
    nfe: int,
    n_samples: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Collect contexts, targets, and stochastic predictions for the full test split."""
    model.eval()

    all_contexts = []
    all_targets = []
    all_samples = []

    for context, target in test_loader:
        context = context.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        batch_samples = []
        for _ in range(n_samples):
            batch_samples.append(model.sample(context, nfe=nfe).cpu())

        all_contexts.append(context.cpu())
        all_targets.append(target.cpu())
        all_samples.append(torch.stack(batch_samples, dim=0))

    contexts = torch.cat(all_contexts, dim=0)
    targets = torch.cat(all_targets, dim=0)
    samples = torch.cat(all_samples, dim=1)
    return contexts, targets, samples


def summarize_prediction_metrics(
    samples: torch.Tensor,
    targets: torch.Tensor,
    dataset,
) -> Tuple[Dict[str, float], Dict[str, float], pd.DataFrame]:
    """Compute aggregate and horizon-wise metrics in normalized and original scales."""
    mean_pred = samples.mean(dim=0)

    samples_original = dataset.inverse_transform(samples)
    targets_original = dataset.inverse_transform(targets)
    mean_pred_original = samples_original.mean(dim=0)

    lower = torch.quantile(samples, 0.1, dim=0)
    upper = torch.quantile(samples, 0.9, dim=0)
    lower_original = torch.quantile(samples_original, 0.1, dim=0)
    upper_original = torch.quantile(samples_original, 0.9, dim=0)

    summary_normalized = {
        "mae": mae(mean_pred, targets),
        "rmse": rmse(mean_pred, targets),
        "crps": crps_gaussian(samples, targets),
        "p10_p90_coverage": float(((targets >= lower) & (targets <= upper)).float().mean().item()),
        "mean_predictive_std": float(samples.std(dim=0).mean().item()),
        "mean_interval_width_p10_p90": float((upper - lower).mean().item()),
    }
    summary_original = {
        "mae": mae(mean_pred_original, targets_original),
        "rmse": rmse(mean_pred_original, targets_original),
        "crps": crps_gaussian(samples_original, targets_original),
        "mean_predictive_std": float(samples_original.std(dim=0).mean().item()),
        "mean_interval_width_p10_p90": float((upper_original - lower_original).mean().item()),
    }

    mean_pred_np = mean_pred.squeeze(-1).numpy()
    targets_np = targets.squeeze(-1).numpy()
    mean_pred_original_np = mean_pred_original.squeeze(-1).numpy()
    targets_original_np = targets_original.squeeze(-1).numpy()
    pred_std_np = samples.std(dim=0).squeeze(-1).numpy()
    pred_std_original_np = samples_original.std(dim=0).squeeze(-1).numpy()
    coverage_np = (((targets >= lower) & (targets <= upper)).float().mean(dim=0).squeeze(-1).numpy())
    interval_width_np = (upper - lower).squeeze(-1).numpy()
    interval_width_original_np = (upper_original - lower_original).squeeze(-1).numpy()

    horizon_steps = np.arange(1, targets.shape[1] + 1)
    per_horizon_df = pd.DataFrame(
        {
            "horizon_step": horizon_steps,
            "mae_normalized": np.mean(np.abs(mean_pred_np - targets_np), axis=0),
            "rmse_normalized": np.sqrt(np.mean((mean_pred_np - targets_np) ** 2, axis=0)),
            "bias_normalized": np.mean(mean_pred_np - targets_np, axis=0),
            "predictive_std_normalized": np.mean(pred_std_np, axis=0),
            "interval_width_p10_p90_normalized": np.mean(interval_width_np, axis=0),
            "coverage_p10_p90": coverage_np,
            "mae_original": np.mean(np.abs(mean_pred_original_np - targets_original_np), axis=0),
            "rmse_original": np.sqrt(
                np.mean((mean_pred_original_np - targets_original_np) ** 2, axis=0)
            ),
            "bias_original": np.mean(mean_pred_original_np - targets_original_np, axis=0),
            "predictive_std_original": np.mean(pred_std_original_np, axis=0),
            "interval_width_p10_p90_original": np.mean(interval_width_original_np, axis=0),
        }
    )

    return summary_normalized, summary_original, per_horizon_df


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
    ax.set_ylabel("Single-Sample Inference Latency (ms)")
    ax.set_title("NFE vs. Single-Sample Inference Latency")
    ax.set_xscale("log", base=2)
    ax.set_xticks(results_df["nfe"].unique())
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.axhline(y=1.0, color="red", linestyle="--", alpha=0.5, label="1 ms threshold")
    ax.legend()

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
    ax2.set_ylabel("Single-Sample Latency (ms)", color="#ff7f0e")
    ax1.set_xscale("log", base=2)
    ax1.set_xticks(nfe_values)
    ax1.get_xaxis().set_major_formatter(plt.ScalarFormatter())

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
    ax1.set_title("Accuracy-Latency Trade-off: Baseline vs. Improved (single-sample)")

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
    train_epochs: List[int],
    train_losses: List[float],
    val_epochs: Optional[List[int]] = None,
    val_losses: Optional[List[float]] = None,
    save_path: str = "training_loss.png",
) -> None:
    """Plot training (and optionally validation) loss curves.

    Uses explicit epoch arrays so that sparse validation losses
    (recorded every N epochs) are plotted at the correct x-positions.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(train_epochs, train_losses, label="Train Loss", linewidth=2)
    if val_losses and val_epochs:
        ax.plot(val_epochs, val_losses, marker="o", markersize=3,
                label="Val Loss", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training Progress")
    ax.legend()
    ax.set_yscale("log")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved plot: {save_path}")


def select_representative_examples(
    samples: torch.Tensor,
    targets: torch.Tensor,
    n_examples: int = 3,
) -> List[int]:
    """Pick low-, mid-, and high-error examples for report-ready visualization."""
    mean_pred = samples.mean(dim=0)
    sample_mae = torch.mean(torch.abs(mean_pred - targets), dim=(1, 2))
    sorted_indices = torch.argsort(sample_mae)
    if len(sorted_indices) == 0:
        return []

    percentiles = np.linspace(0, len(sorted_indices) - 1, num=min(n_examples, len(sorted_indices)))
    chosen = sorted_indices[torch.tensor(percentiles.round().astype(int))]
    return chosen.tolist()


def plot_forecast_examples(
    contexts: torch.Tensor,
    targets: torch.Tensor,
    predictions: torch.Tensor,
    save_path: str,
    example_indices: List[int],
    title: str,
) -> None:
    """Plot several representative forecast examples with uncertainty bands."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    n_examples = max(len(example_indices), 1)
    fig, axes = plt.subplots(n_examples, 1, figsize=(12, 3.5 * n_examples), squeeze=False)

    for row, sample_idx in enumerate(example_indices):
        ax = axes[row, 0]
        context = contexts[sample_idx].numpy().squeeze()
        target = targets[sample_idx].numpy().squeeze()
        preds = predictions[:, sample_idx].numpy().squeeze(-1)

        ctx_len = len(context)
        horizon = len(target)
        t_ctx = np.arange(ctx_len)
        t_pred = np.arange(ctx_len, ctx_len + horizon)

        mean_pred = preds.mean(axis=0)
        lower = np.quantile(preds, 0.1, axis=0)
        upper = np.quantile(preds, 0.9, axis=0)

        ax.plot(t_ctx, context, color="black", linewidth=1.4, label="Context")
        ax.plot(t_pred, target, color="green", linewidth=2, label="Ground Truth")
        ax.fill_between(
            t_pred,
            lower,
            upper,
            color="#4c78a8",
            alpha=0.20,
            label="P10-P90",
        )

        for pred in preds[: min(5, len(preds))]:
            ax.plot(t_pred, pred, color="#4c78a8", alpha=0.18, linewidth=1)

        ax.plot(t_pred, mean_pred, color="#d62728", linewidth=2, linestyle="--", label="Mean Prediction")
        ax.axvline(x=ctx_len, color="gray", linestyle=":", alpha=0.6)
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Amplitude")
        ax.set_title(f"Example {row + 1} (test index {sample_idx})")
        ax.legend(loc="upper left")

    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved plot: {save_path}")


def plot_horizon_error(per_horizon_df: pd.DataFrame, save_path: str) -> None:
    """Plot error growth as the forecast horizon increases."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(
        per_horizon_df["horizon_step"],
        per_horizon_df["mae_normalized"],
        label="MAE",
        linewidth=2,
    )
    ax.plot(
        per_horizon_df["horizon_step"],
        per_horizon_df["rmse_normalized"],
        label="RMSE",
        linewidth=2,
    )
    ax.set_xlabel("Forecast Horizon Step")
    ax.set_ylabel("Normalized Error")
    ax.set_title("Forecast Error vs. Horizon")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved plot: {save_path}")


def plot_horizon_uncertainty(per_horizon_df: pd.DataFrame, save_path: str) -> None:
    """Plot predictive uncertainty and interval coverage across the horizon."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax2 = ax1.twinx()

    ax1.plot(
        per_horizon_df["horizon_step"],
        per_horizon_df["predictive_std_normalized"],
        color="#1f77b4",
        linewidth=2,
        label="Predictive Std",
    )
    ax2.plot(
        per_horizon_df["horizon_step"],
        per_horizon_df["coverage_p10_p90"],
        color="#ff7f0e",
        linewidth=2,
        linestyle="--",
        label="P10-P90 Coverage",
    )

    ax1.set_xlabel("Forecast Horizon Step")
    ax1.set_ylabel("Normalized Predictive Std", color="#1f77b4")
    ax2.set_ylabel("Coverage", color="#ff7f0e")
    ax2.set_ylim(0.0, 1.0)
    ax1.set_title("Predictive Uncertainty vs. Horizon")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

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
        help="Deprecated: plots are now saved by default unless --skip_plots is used.",
    )
    parser.add_argument(
        "--skip_plots", action="store_true",
        help="Skip saving evaluation plots",
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
    load_model_state(model, ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {args.model} ({n_params:,} params)")
    print(f"NFE: {args.nfe}")
    print(f"Samples per test case: {args.n_samples}")
    print(f"Evaluation output dir: {eval_dir}")

    # Evaluate
    print("\nEvaluating...")
    contexts, targets, samples = collect_predictions(
        model=model,
        test_loader=test_loader,
        nfe=args.nfe,
        n_samples=args.n_samples,
        device=device,
    )
    metrics_res, metrics_original, per_horizon_df = summarize_prediction_metrics(
        samples=samples,
        targets=targets,
        dataset=dataset,
    )

    print(f"\n{'='*40}")
    print(f"Results ({args.model}, NFE={args.nfe}):")
    print(f"  MAE:  {metrics_res['mae']:.6f}")
    print(f"  RMSE: {metrics_res['rmse']:.6f}")
    print(f"  CRPS: {metrics_res['crps']:.6f}")
    print(f"  P10-P90 coverage: {metrics_res['p10_p90_coverage']:.4f}")

    # Latency
    context_batch = contexts[:1].to(device)

    latency = measure_latency(model, context_batch, nfe=args.nfe)
    print(f"  Latency: {latency['mean_ms']:.3f} ± {latency['std_ms']:.3f} ms")
    print(f"{'='*40}")

    # Save metrics to JSON for report writing
    metrics_output = {
        "model": args.model,
        "checkpoint": args.checkpoint,
        "nfe": args.nfe,
        "n_samples": args.n_samples,
        "parameter_count": n_params,
        "metrics_normalized": metrics_res,
        "metrics_original_scale": metrics_original,
        "latency_ms": {
            **latency,
            "batch_size": 1,
        },
        "dataset": {
            "context_length": config["data"]["context_length"],
            "prediction_horizon": config["data"]["prediction_horizon"],
            "test_windows": len(test_loader.dataset),
            "normalization_mean": dataset.mean,
            "normalization_std": dataset.std,
        },
    }
    metrics_json_path = os.path.join(eval_dir, f"metrics_nfe{args.nfe}.json")
    with open(metrics_json_path, "w") as f:
        json.dump(metrics_output, f, indent=2)
    print(f"Metrics saved to: {metrics_json_path}")

    per_horizon_path = os.path.join(eval_dir, f"per_horizon_metrics_nfe{args.nfe}.csv")
    per_horizon_df.to_csv(per_horizon_path, index=False)
    print(f"Per-horizon metrics saved to: {per_horizon_path}")

    # Sample plots
    generate_plots = not args.skip_plots
    if generate_plots:
        example_indices = select_representative_examples(samples, targets, n_examples=3)

        plot_forecast_examples(
            contexts=dataset.inverse_transform(contexts),
            targets=dataset.inverse_transform(targets),
            predictions=dataset.inverse_transform(samples),
            save_path=os.path.join(eval_dir, f"forecast_examples_nfe{args.nfe}.png"),
            example_indices=example_indices,
            title=f"{args.model} Forecast Examples (NFE={args.nfe})",
        )

        plot_path = os.path.join(eval_dir, f"eval_nfe{args.nfe}.png")
        idx = example_indices[min(1, len(example_indices) - 1)] if example_indices else 0
        plot_sample_predictions(
            context=dataset.inverse_transform(contexts[idx]).numpy().squeeze(),
            target=dataset.inverse_transform(targets[idx]).numpy().squeeze(),
            predictions=dataset.inverse_transform(samples[:, idx]).numpy().squeeze(-1),
            save_path=plot_path,
            title=f"{args.model} (NFE={args.nfe})",
        )

        plot_horizon_error(
            per_horizon_df=per_horizon_df,
            save_path=os.path.join(eval_dir, f"error_by_horizon_nfe{args.nfe}.png"),
        )
        plot_horizon_uncertainty(
            per_horizon_df=per_horizon_df,
            save_path=os.path.join(eval_dir, f"uncertainty_by_horizon_nfe{args.nfe}.png"),
        )

        # Plot training loss curves from CSV log if available
        model_dir = os.path.join(config["training"].get("output_dir", "outputs"), args.model, "model")
        csv_log_path = os.path.join(model_dir, "training_log.csv")
        if os.path.exists(csv_log_path):
            log_df = pd.read_csv(csv_log_path)

            # Train losses: every row has a value
            train_mask = log_df["train_loss"].notna()
            train_epochs = log_df.loc[train_mask, "epoch"].astype(int).tolist()
            train_losses = log_df.loc[train_mask, "train_loss"].tolist()

            # Val losses: only present every eval_every epochs
            val_epochs = None
            val_losses = None
            if "val_loss" in log_df.columns:
                val_mask = log_df["val_loss"].notna() & (log_df["val_loss"] != "")
                if val_mask.any():
                    val_epochs = log_df.loc[val_mask, "epoch"].astype(int).tolist()
                    val_losses = pd.to_numeric(log_df.loc[val_mask, "val_loss"]).tolist()

            plot_training_loss(
                train_epochs=train_epochs,
                train_losses=train_losses,
                val_epochs=val_epochs,
                val_losses=val_losses,
                save_path=os.path.join(eval_dir, "training_loss.png"),
            )
        else:
            print(f"Training log not found at {csv_log_path}, skipping loss curve plot.")


if __name__ == "__main__":
    main()
