"""
Training entry point and pipeline for Flow Matching models.

Usage:
    python train.py --model baseline
    python train.py --model improved
    python train.py --model baseline --resume outputs/baseline/model/best.pt

Features:
    - Automatic GPU detection and mixed precision (torch.cuda.amp)
    - Cosine annealing LR scheduler with warmup
    - Checkpoint save/load with early stopping
    - CSV logging for training curves
    - Gradient clipping
"""

import argparse
import csv
import os
import random
import sys
import time
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataset import create_dataloaders
from models.flow_matching import build_model


# ==============================================================================
# Trainer
# ==============================================================================

class Trainer:
    """Main training loop for Flow Matching models."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict,
        model_type: str = "baseline",
    ):
        self.config = config
        self.train_cfg = config["training"]
        self.model_type = model_type

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        if self.device.type == "cuda":
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
            print(f"  Memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

        self.model = model.to(self.device)

        # PyTorch 2.0+ Compiler for significant speedups
        if hasattr(torch, "compile") and self.device.type == "cuda":
            print("  Optimizing model with torch.compile()...")
            # mode="reduce-overhead" is great for fast ODE solvers
            self.model = torch.compile(self.model, mode="reduce-overhead")

        self.train_loader = train_loader
        self.val_loader = val_loader

        # Optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.train_cfg["learning_rate"],
            weight_decay=self.train_cfg["weight_decay"],
        )

        # LR Scheduler: warmup + cosine annealing
        warmup_epochs = self.train_cfg.get("warmup_epochs", 5)
        cosine_epochs = self.train_cfg.get("cosine_T_max", 150)

        if self.train_cfg.get("scheduler") == "cosine" and cosine_epochs > warmup_epochs:
            warmup_scheduler = LinearLR(
                self.optimizer,
                start_factor=0.01,
                end_factor=1.0,
                total_iters=warmup_epochs,
            )
            cosine_scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=cosine_epochs - warmup_epochs,
                eta_min=1e-6,
            )
            self.scheduler = SequentialLR(
                self.optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[warmup_epochs],
            )
        else:
            self.scheduler = None

        # Mixed precision
        self.use_amp = self.train_cfg.get("mixed_precision", True) and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda") if self.use_amp else None

        # Checkpoint directory
        self.ckpt_dir = os.path.join(self.train_cfg["output_dir"], model_type, "model")
        os.makedirs(self.ckpt_dir, exist_ok=True)

        # Training state
        self.start_epoch = 0
        self.global_step = 0
        self.best_val_loss = float("inf")
        self.epochs_without_improvement = 0
        self.patience = self.train_cfg.get("early_stopping_patience", 30)

        # CSV log for training curves
        self.csv_path = os.path.join(self.ckpt_dir, "training_log.csv")
        with open(self.csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_loss", "val_loss", "lr", "best_val_loss"])

    def train(self, resume_from: Optional[str] = None) -> None:
        """Run the full training loop."""
        if resume_from:
            self.load_checkpoint(resume_from)

        total_epochs = self.train_cfg["epochs"]
        grad_clip = self.train_cfg.get("grad_clip_norm", 1.0)

        print(f"\n{'='*60}")
        print(f"Training {self.model_type} model for {total_epochs} epochs")
        print(f"  Batch size: {self.train_cfg['batch_size']}")
        print(f"  Learning rate: {self.train_cfg['learning_rate']}")
        print(f"  Mixed precision: {self.use_amp}")
        print(f"{'='*60}\n")

        for epoch in range(self.start_epoch, total_epochs):
            # --- Training ---
            self.model.train()
            train_loss_sum = 0.0
            n_batches = 0

            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{total_epochs}")
            for context, target in pbar:
                context = context.to(self.device, non_blocking=True)
                target = target.to(self.device, non_blocking=True)

                self.optimizer.zero_grad()

                if self.use_amp:
                    with torch.amp.autocast("cuda"):
                        loss = self.model.compute_loss(context, target)
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss = self.model.compute_loss(context, target)
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
                    self.optimizer.step()

                train_loss_sum += loss.item()
                n_batches += 1
                self.global_step += 1

                pbar.set_postfix(loss=f"{loss.item():.4f}")

            avg_train_loss = train_loss_sum / max(n_batches, 1)

            if self.scheduler is not None:
                self.scheduler.step()

            # --- Validation ---
            if (epoch + 1) % self.train_cfg.get("eval_every", 5) == 0:
                val_loss = self.validate()

                print(
                    f"  Epoch {epoch+1}: train_loss={avg_train_loss:.4f}, "
                    f"val_loss={val_loss:.4f}, "
                    f"lr={self.optimizer.param_groups[0]['lr']:.2e}"
                )

                # Save best model
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.epochs_without_improvement = 0
                    self.save_checkpoint(epoch, "best.pt")
                else:
                    self.epochs_without_improvement += self.train_cfg.get("eval_every", 5)

                # Log to CSV
                self._log_csv(epoch + 1, avg_train_loss, val_loss)

                # Early stopping check
                if self.epochs_without_improvement >= self.patience:
                    print(
                        f"\n  Early stopping at epoch {epoch+1}: "
                        f"no improvement for {self.epochs_without_improvement} epochs."
                    )
                    break
            else:
                print(
                    f"  Epoch {epoch+1}: train_loss={avg_train_loss:.4f}, "
                    f"lr={self.optimizer.param_groups[0]['lr']:.2e}, "
                    f"best_val={self.best_val_loss:.4f}"
                )
                # Log to CSV (val_loss not evaluated this epoch)
                self._log_csv(epoch + 1, avg_train_loss, None)

        print(f"\nTraining complete. Best val loss: {self.best_val_loss:.4f}")
        print(f"Training log saved to: {self.csv_path}")

    def _log_csv(self, epoch: int, train_loss: float, val_loss: float = None) -> None:
        """Append one row to the training CSV log."""
        lr = self.optimizer.param_groups[0]["lr"]
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                f"{train_loss:.6f}",
                f"{val_loss:.6f}" if val_loss is not None else "",
                f"{lr:.2e}",
                f"{self.best_val_loss:.6f}",
            ])

    @torch.no_grad()
    def validate(self) -> float:
        """Run validation and return average loss."""
        self.model.eval()
        val_loss_sum = 0.0
        n_batches = 0

        for context, target in self.val_loader:
            context = context.to(self.device, non_blocking=True)
            target = target.to(self.device, non_blocking=True)

            if self.use_amp:
                with torch.amp.autocast("cuda"):
                    loss = self.model.compute_loss(context, target)
            else:
                loss = self.model.compute_loss(context, target)

            val_loss_sum += loss.item()
            n_batches += 1

        return val_loss_sum / max(n_batches, 1)

    def save_checkpoint(self, epoch: int, filename: str) -> None:
        """Save model checkpoint."""
        path = os.path.join(self.ckpt_dir, filename)
        torch.save(
            {
                "epoch": epoch,
                "global_step": self.global_step,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": (
                    self.scheduler.state_dict() if self.scheduler else None
                ),
                "best_val_loss": self.best_val_loss,
                "config": self.config,
                "model_type": self.model_type,
            },
            path,
        )
        print(f"  Checkpoint saved: {path}")

    def load_checkpoint(self, path: str) -> None:
        """Load model checkpoint."""
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if self.scheduler and ckpt.get("scheduler_state_dict"):
            self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        self.start_epoch = ckpt["epoch"] + 1
        self.global_step = ckpt["global_step"]
        self.best_val_loss = ckpt.get("best_val_loss", float("inf"))
        print(f"Resumed from epoch {self.start_epoch} (step {self.global_step})")


# ==============================================================================
# CLI Entry Point
# ==============================================================================

def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # Disable exact determinism in favor of cuDNN autotuner speed (benchmark=True)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        
        # Enable TensorFloat-32 (TF32) for massive matmul speedups on Ampere GPUs (RTX 3090)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


def main():
    parser = argparse.ArgumentParser(
        description="Train Flow Matching model for time-series forecasting"
    )
    parser.add_argument(
        "--config", type=str, default="config.yaml",
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--model", type=str, default="baseline", choices=["baseline", "improved"],
        help="Model variant: 'baseline' (Gaussian prior) or 'improved' (OU prior + OT)",
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to checkpoint to resume training from",
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Override number of training epochs",
    )
    parser.add_argument(
        "--batch_size", type=int, default=None,
        help="Override batch size",
    )
    parser.add_argument(
        "--lr", type=float, default=None,
        help="Override learning rate",
    )
    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Apply CLI overrides
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    if args.batch_size is not None:
        config["training"]["batch_size"] = args.batch_size
    if args.lr is not None:
        config["training"]["learning_rate"] = args.lr

    # Set seed
    set_seed(config["training"]["seed"])

    # Enforce NVIDIA GPU requirement
    if not torch.cuda.is_available() or torch.version.cuda is None:
        print("Error: An NVIDIA GPU is required for training. Aborting.")
        sys.exit(1)

    # Print configuration
    print("=" * 60)
    print(f"Flow Matching — {args.model.upper()} Model")
    print("=" * 60)
    print(f"Config: {args.config}")
    print(f"Epochs: {config['training']['epochs']}")
    print(f"Batch size: {config['training']['batch_size']}")
    print(f"Learning rate: {config['training']['learning_rate']}")
    print(f"Context length: {config['data']['context_length']}")
    print(f"Prediction horizon: {config['data']['prediction_horizon']}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("=" * 60)

    # Create data loaders
    print("\nLoading data...")
    train_loader, val_loader, test_loader, dataset = create_dataloaders(config)
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches:   {len(val_loader)}")
    print(f"  Test batches:  {len(test_loader)}")

    # Build model
    print(f"\nBuilding {args.model} model...")
    model = build_model(config, args.model)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {n_params:,}")

    # Train
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        model_type=args.model,
    )
    trainer.train(resume_from=args.resume)


if __name__ == "__main__":
    main()
