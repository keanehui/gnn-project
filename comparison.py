"""
NFE Ablation Study Runner.

Evaluates trained Baseline and Improved Flow Matching models across
a gradient of NFE steps (e.g., 2, 4, 8, 16, 32) and measures:
    - MAE / RMSE (accuracy)
    - Wall-clock inference latency (ms)
    - CRPS (probabilistic calibration)

Outputs a CSV results table and generates comparison plots.
"""

import argparse
import os
import sys
from typing import Dict

import numpy as np
import pandas as pd
import torch
import yaml


from data.dataset import create_dataloaders
from models.flow_matching import build_model
from evaluate import (
    crps_gaussian,
    mae,
    measure_latency,
    plot_nfe_accuracy,
    plot_nfe_accuracy_latency,
    plot_nfe_latency,
    plot_sample_predictions,
    rmse,
)


def load_trained_model(
    checkpoint_path: str, config: Dict, model_type: str, device: torch.device
) -> torch.nn.Module:
    """Load a trained model from checkpoint."""
    model = build_model(config, model_type)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()
    print(f"Loaded {model_type} model from {checkpoint_path}")
    return model


def run_ablation(
    config: Dict,
    checkpoint_baseline: str,
    checkpoint_improved: str,
    device: torch.device,
    results_dir: str,
) -> pd.DataFrame:
    """
    Run the NFE ablation study for both models.

    Returns:
        DataFrame with columns:
            model, nfe, mae, rmse, crps, latency_mean_ms, latency_std_ms
    """
    ablation_cfg = config["ablation"]
    nfe_steps = ablation_cfg["nfe_steps"]
    # results_dir is now passed in from main
    # os.makedirs(results_dir, exist_ok=True) # This is now handled in main

    # Load data
    _, _, test_loader, dataset = create_dataloaders(config)

    # Load models
    baseline_model = load_trained_model(checkpoint_baseline, config, "baseline", device)
    improved_model = load_trained_model(checkpoint_improved, config, "improved", device)

    models = {
        "Baseline": baseline_model,
        "Improved (OU+OT)": improved_model,
    }
    param_counts = {
        "Baseline": sum(p.numel() for p in baseline_model.parameters() if p.requires_grad),
        "Improved (OU+OT)": sum(p.numel() for p in improved_model.parameters() if p.requires_grad),
    }

    # Get a representative context batch for latency measurement
    context_batch, _ = next(iter(test_loader))
    context_batch = context_batch.to(device)

    results = []

    for model_name, model in models.items():
        print(f"\n{'='*60}")
        print(f"Evaluating: {model_name}")
        print(f"{'='*60}")

        for nfe in nfe_steps:
            print(f"\n  NFE = {nfe}:")

            # Generate samples for metrics
            all_preds = []
            all_targets = []
            for context, target in test_loader:
                context = context.to(device)
                target = target.to(device)
                # Generate multiple samples per context for CRPS
                preds_batch = []
                for _ in range(ablation_cfg["n_samples_for_metrics"]):
                    preds_batch.append(model.sample(context, nfe=nfe).cpu().numpy())
                all_preds.append(np.stack(preds_batch, axis=1)) # (batch_size, n_samples, seq_len, 1)
                all_targets.append(target.cpu().numpy())

            all_preds = np.concatenate(all_preds, axis=0) # (num_total_samples, n_samples, seq_len, 1)
            all_targets = np.concatenate(all_targets, axis=0) # (num_total_samples, seq_len, 1)

            # Accuracy metrics
            mean_preds = all_preds.mean(axis=1) # (num_total_samples, seq_len, 1)
            mae_val = mae(mean_preds, all_targets)
            rmse_val = rmse(mean_preds, all_targets)
            crps_val = crps_gaussian(all_preds, all_targets)

            print(f"    MAE:  {mae_val:.6f}")
            print(f"    RMSE: {rmse_val:.6f}")
            print(f"    CRPS: {crps_val:.6f}")

            # Latency
            latency = measure_latency(
                model, context_batch, nfe=nfe,
                warmup_iters=ablation_cfg["latency_warmup_iters"],
                measure_iters=ablation_cfg["latency_measure_iters"],
            )
            print(f"    Latency: {latency['mean_ms']:.3f} ± {latency['std_ms']:.3f} ms")

            results.append({
                "model": model_name,
                "params": param_counts[model_name],
                "nfe": nfe,
                "mae": mae_val,
                "rmse": rmse_val,
                "crps": crps_val,
                "latency_mean_ms": latency["mean_ms"],
                "latency_std_ms": latency["std_ms"],
                "latency_min_ms": latency["min_ms"],
                "latency_max_ms": latency["max_ms"],
            })

    df = pd.DataFrame(results)

    # Save CSV
    csv_path = os.path.join(results_dir, "nfe_ablation_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")

    return df


def generate_sample_plots(
    config: Dict,
    checkpoint_baseline: str,
    checkpoint_improved: str,
    device: torch.device,
    results_dir: str,
    nfe_baseline: int = 16,
    nfe_improved: int = 4,
) -> None:
    """Generate sample prediction visualizations."""
    # results_dir is now passed in from main
    _, _, test_loader, dataset = create_dataloaders(config)

    baseline_model = load_trained_model(checkpoint_baseline, config, "baseline", device)
    improved_model = load_trained_model(checkpoint_improved, config, "improved", device)

    # Get one test batch
    context, target = next(iter(test_loader))
    context = context.to(device)
    target = target.to(device)

    # Take first sample from batch
    ctx = context[0:1]
    tgt = target[0]

    for model_name, model, nfe in [
        ("Baseline", baseline_model, nfe_baseline),
        ("Improved", improved_model, nfe_improved),
    ]:
        # Generate multiple samples
        preds = []
        for _ in range(20):
            pred = model.sample(ctx, nfe=nfe)
            preds.append(pred[0].cpu().numpy().squeeze())
        preds = np.array(preds)

        plot_sample_predictions(
            context=ctx[0].cpu().numpy().squeeze(),
            target=tgt.cpu().numpy().squeeze(),
            predictions=preds,
            save_path=os.path.join(results_dir, f"sample_predictions_{model_name.lower()}.png"),
            n_samples=10,
            title=f"{model_name} Model (NFE={nfe}): Sample Predictions",
        )


def main():
    parser = argparse.ArgumentParser(description="NFE Ablation Study")
    parser.add_argument(
        "--config", type=str, default="config.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--checkpoint_baseline", type=str, required=True,
        help="Path to trained baseline model checkpoint",
    )
    parser.add_argument(
        "--checkpoint_improved", type=str, required=True,
        help="Path to trained improved model checkpoint",
    )
    parser.add_argument(
        "--skip_plots", action="store_true",
        help="Skip generating plots",
    )
    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Set up ablation results directory
    results_dir = os.path.join(config["training"].get("output_dir", "outputs"), "ablation")
    os.makedirs(results_dir, exist_ok=True)

    # Run ablation
    results_df = run_ablation(
        config, args.checkpoint_baseline, args.checkpoint_improved, device, results_dir
    )

    if not args.skip_plots:
        # results_dir is already defined above
        # Generate NFE comparison plots
        plot_nfe_accuracy(results_df, os.path.join(results_dir, "nfe_vs_mae.png"), metric="mae")
        plot_nfe_accuracy(results_df, os.path.join(results_dir, "nfe_vs_rmse.png"), metric="rmse")
        plot_nfe_latency(results_df, os.path.join(results_dir, "nfe_vs_latency.png"))
        plot_nfe_accuracy_latency(
            results_df, os.path.join(results_dir, "nfe_accuracy_latency.png"), metric="mae"
        )

        # Sample prediction plots
        generate_sample_plots(
            config, args.checkpoint_baseline, args.checkpoint_improved, device
        )

    # Print summary table
    print("\n" + "=" * 80)
    print("NFE ABLATION RESULTS SUMMARY")
    print("=" * 80)
    print(results_df.to_string(index=False, float_format="%.4f"))


if __name__ == "__main__":
    main()
