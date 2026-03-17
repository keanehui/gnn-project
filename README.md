# Flow Matching for Predictive Time-Series Forecasting

**Overcoming Latency in High-Frequency Systems**

> Course Project — Generative Neural Networks for the Sciences (WS 2025/26)  
> Team: Ka Wong Hui, Min-Han Yeh

## Overview

This project implements a **Flow Matching** framework for multi-step time-series forecasting on high-frequency vibration data. We compare two variants:

| Model | Prior | Description |
|---|---|---|
| **Baseline** | Gaussian N(0, I) | Standard isotropic prior — high accuracy but many NFEs needed |
| **Improved** | OU Process + OT | Ornstein-Uhlenbeck prior with Optimal Transport coupling — fewer NFEs |

The key hypothesis: the OU-prior model at **4 NFEs** achieves competitive accuracy to the Baseline at **16 NFEs**, resolving the inference latency bottleneck.

## Project Structure

```
├── config.yaml                   # All hyperparameters
├── data/
│   ├── download.py               # Download CWRU bearing dataset
│   └── dataset.py                # PyTorch Dataset + preprocessing
├── models/
│   ├── tcn.py                    # TCN velocity field network
│   ├── flow_matching.py          # Flow Matching (Baseline + Improved)
│   └── ou_prior.py               # OU prior + OT coupling
├── comparison.py                 # NFE ablation study
├── train.py                      # Training pipeline (Trainer, AMP, checkpoints, CSV logging)
└── evaluate.py                   # Unified evaluation, metrics, and visualization

Outputs generated during training/evaluation:
outputs/
├── baseline/
│   ├── model/                    # best.pt and training_log.csv
│   └── evaluate/                 # evaluation plots and metrics
├── improved/
│   ├── model/
│   └── evaluate/
└── ablation/                     # Ablation study CSVs and plots
```

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

---

### Part 1: Data Preparation & Baseline Model Training (Ka Wong Hui)

#### 2. Download data
- **Reads:** Internet source (CWRU bearing dataset URL).
- **Outputs:** Raw and processed datasets saved to the `data/cwru_data/` and `data/processed/` directories.

```bash
python3 data/download.py
```

#### 3. Train Baseline Model
- **Reads:** Configuration from `config.yaml` and the datasets from `data/processed/`.
- **Outputs:** Model checkpoints (e.g., `best.pt`) and training logs saved to `outputs/baseline/model/`.

**Foreground Training (Terminal stays open):**
```bash
python3 train.py --model baseline
```

**Background Training (Safe to disconnect):**
```bash
./run_training_background.sh baseline
```
*To view the live progress, run:* `tail -f baseline_training.log`

---

### Part 2: Improved Model Training, Evaluation & Ablation Study (Min-Han Yeh)

#### 4. Train Improved Model
- **Reads:** Configuration from `config.yaml` and the datasets from `data/processed/`.
- **Outputs:** Model checkpoints (e.g., `best.pt`) and training logs saved to `outputs/improved/model/`.

**Foreground Training:**
```bash
python3 train.py --model improved
```

**Background Training (Safe to disconnect):**
```bash
./run_training_background.sh improved
```
*To view the live progress, run:* `tail -f improved_training.log`

#### 5. Evaluate Models
- **Reads:** Trained model checkpoints from Part 1 (`outputs/baseline/model/best.pt`, `outputs/improved/model/best.pt`) and `config.yaml`.
- **Outputs:** Evaluation plots and metrics in `outputs/baseline/evaluate/` and `outputs/improved/evaluate/`.

```bash
python3 evaluate.py --checkpoint outputs/baseline/model/best.pt --model baseline --nfe 16 --plot
python3 evaluate.py --checkpoint outputs/improved/model/best.pt --model improved --nfe 4 --plot
```

#### 6. NFE Ablation Study
- **Reads:** Trained model checkpoints for both models, the dataset, and `config.yaml`.
- **Outputs:** CSV results table and NFE comparison plots saved to `outputs/ablation/`.

```bash
python3 comparison.py \
    --checkpoint_baseline outputs/baseline/model/best.pt \
    --checkpoint_improved outputs/improved/model/best.pt
```

## Configuration

All hyperparameters are centralized in `config.yaml`. Key settings:

| Parameter | Default | Description |
|---|---|---|
| `context_length` | 256 | Past timesteps as input |
| `prediction_horizon` | 64 | Future timesteps to predict |
| `tcn.hidden_channels` | 128 | TCN hidden dimension |
| `tcn.num_layers` | 6 | Number of TCN blocks |
| `ou_prior.theta` | 1.0 | OU mean reversion speed |
| `training.epochs` | 200 | Training epochs |
| `training.batch_size` | 128 | Batch size |
| `training.mixed_precision` | true | Use AMP on NVIDIA GPUs |

## Dataset

**CWRU Bearing Fault Dataset** — 12,000 samples/second Drive End accelerometer data from Case Western Reserve University. Includes normal operation and three fault conditions (inner race, ball, outer race at 0.007").

## GPU Training

The project is designed for **NVIDIA GPU** training with:
- Automatic CUDA detection
- Mixed precision (`torch.cuda.amp`) for ~2x speedup
- Gradient clipping and cosine annealing LR

## References

- Lipman et al. (2023). *Flow Matching for Generative Modeling*
- Tong et al. (2023). *Improving and Generalizing Flow-Based Generative Models with Minibatch Optimal Transport*
- Kollovieh et al. *TSFlow*
- CWRU Bearing Data Center: https://engineering.case.edu/bearingdatacenter
