"""
PyTorch Dataset for CWRU Bearing Data.

Loads .mat files, applies sliding-window segmentation to create
(context, target) pairs for time-series forecasting, and provides
train/val/test splits.
"""

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from scipy.io import loadmat
from torch.utils.data import Dataset, DataLoader, Subset

# Mapping from fault type to the key pattern in .mat files
# CWRU .mat files store the accelerometer signal under keys like 'X097_DE_time'
FAULT_TO_MAT_KEY = {
    "normal": "X097_DE_time",
    "IR007": "X105_DE_time",
    "B007": "X118_DE_time",
    "OR007": "X130_DE_time",
}


def load_cwru_signal(filepath: str, fault_type: str) -> np.ndarray:
    """
    Load a single CWRU .mat file and extract the Drive End accelerometer signal.

    Args:
        filepath: Path to the .mat file.
        fault_type: One of 'normal', 'IR007', 'B007', 'OR007'.

    Returns:
        1D numpy array of the vibration signal.
    """
    mat = loadmat(filepath)
    key = FAULT_TO_MAT_KEY[fault_type]

    if key not in mat:
        # Try to find the DE key automatically
        de_keys = [k for k in mat.keys() if "DE_time" in k]
        if de_keys:
            key = de_keys[0]
        else:
            raise KeyError(
                f"Could not find Drive End signal in {filepath}. "
                f"Available keys: {[k for k in mat.keys() if not k.startswith('__')]}"
            )

    signal = mat[key].squeeze()
    return signal.astype(np.float32)


class CWRUDataset(Dataset):
    """
    CWRU Bearing Dataset for time-series forecasting.

    Each sample consists of:
        - context: (context_length, 1) tensor of past observations
        - target:  (prediction_horizon, 1) tensor of future observations

    The dataset concatenates signals from multiple fault conditions.
    """

    def __init__(
        self,
        data_dir: str = "data/raw",
        fault_types: Optional[List[str]] = None,
        context_length: int = 256,
        prediction_horizon: int = 64,
        stride: int = 64,
    ):
        super().__init__()
        self.context_length = context_length
        self.prediction_horizon = prediction_horizon
        self.stride = stride

        if fault_types is None:
            fault_types = ["normal", "IR007", "B007", "OR007"]

        # Load and concatenate all signals
        signals = []
        self.signal_boundaries = []  # Track where each signal starts/ends
        offset = 0

        from data.download import CWRU_FILES

        for fault in fault_types:
            info = CWRU_FILES[fault]
            filepath = os.path.join(data_dir, info["filename"])
            if not os.path.exists(filepath):
                print(f"[WARNING] {filepath} not found. Skipping {fault}.")
                continue

            sig = load_cwru_signal(filepath, fault)
            self.signal_boundaries.append((offset, offset + len(sig), fault))
            signals.append(sig)
            offset += len(sig)

        if not signals:
            raise FileNotFoundError(
                f"No data files found in {data_dir}. Run `python data/download.py` first."
            )

        # Concatenate and normalize (z-score)
        self.raw_signal = np.concatenate(signals)
        self.mean = self.raw_signal.mean()
        self.std = self.raw_signal.std()
        self.signal = (self.raw_signal - self.mean) / (self.std + 1e-8)

        # Create sliding window indices
        window_size = context_length + prediction_horizon
        self.indices = []

        # Create windows within each signal (don't cross signal boundaries)
        for start, end, fault in self.signal_boundaries:
            for i in range(start, end - window_size + 1, stride):
                self.indices.append(i)

        print(
            f"CWRUDataset: {len(self.indices)} samples, "
            f"context={context_length}, horizon={prediction_horizon}, "
            f"signal_length={len(self.signal)}"
        )

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        start = self.indices[idx]
        context = self.signal[start : start + self.context_length]
        target = self.signal[
            start + self.context_length : start + self.context_length + self.prediction_horizon
        ]

        # Shape: (seq_len, 1)
        context = torch.from_numpy(context).unsqueeze(-1)
        target = torch.from_numpy(target).unsqueeze(-1)
        return context, target

    def inverse_transform(self, x: torch.Tensor) -> torch.Tensor:
        """Undo z-score normalization."""
        return x * self.std + self.mean


def create_dataloaders(
    config: Dict,
) -> Tuple[DataLoader, DataLoader, DataLoader, "CWRUDataset"]:
    """
    Create train/val/test DataLoaders from configuration.

    Each fault condition is split independently (70/15/15) so that
    all fault types are represented in every split. This avoids the
    problem of a global sequential cut leaving some fault types only
    in val/test.

    Args:
        config: Full configuration dictionary.

    Returns:
        (train_loader, val_loader, test_loader, dataset)
    """
    data_cfg = config["data"]
    train_cfg = config["training"]

    dataset = CWRUDataset(
        data_dir=data_cfg["dataset_dir"],
        fault_types=data_cfg.get("fault_types"),
        context_length=data_cfg["context_length"],
        prediction_horizon=data_cfg["prediction_horizon"],
        stride=data_cfg["stride"],
    )

    train_ratio = data_cfg["train_ratio"]
    val_ratio = data_cfg["val_ratio"]

    # Split within each fault condition independently so all fault types
    # are represented in every split. dataset.signal_boundaries holds
    # (sig_start, sig_end, fault) in absolute signal-array coordinates.
    window_size = data_cfg["context_length"] + data_cfg["prediction_horizon"]

    train_indices, val_indices, test_indices = [], [], []

    for sig_start, sig_end, fault in dataset.signal_boundaries:
        # Collect dataset-level positions (i.e. indices into dataset.indices[])
        # whose absolute start coordinate belongs to this fault's signal.
        fault_window_positions = [
            pos for pos, abs_start in enumerate(dataset.indices)
            if sig_start <= abs_start < sig_end - window_size + 1
        ]

        n = len(fault_window_positions)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        # Sequential split within this fault's windows (preserves temporal order)
        train_indices.extend(fault_window_positions[:n_train])
        val_indices.extend(fault_window_positions[n_train: n_train + n_val])
        test_indices.extend(fault_window_positions[n_train + n_val:])
        print(
            f"  {fault:8s}: {n} windows -> "
            f"train={n_train}, val={n_val}, test={n - n_train - n_val}"
        )

    print(
        f"Total — train: {len(train_indices)}, "
        f"val: {len(val_indices)}, test: {len(test_indices)}"
    )

    train_loader = DataLoader(
        Subset(dataset, train_indices),
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=train_cfg["num_workers"],
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        Subset(dataset, val_indices),
        batch_size=train_cfg["batch_size"],
        shuffle=False,
        num_workers=train_cfg["num_workers"],
        pin_memory=True,
    )
    test_loader = DataLoader(
        Subset(dataset, test_indices),
        batch_size=train_cfg["batch_size"],
        shuffle=False,
        num_workers=train_cfg["num_workers"],
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader, dataset
