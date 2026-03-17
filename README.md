# Flow Matching for Predictive Time-Series Forecasting

**Overcoming Latency in High-Frequency Systems**

> Course Project — Generative Neural Networks for the Sciences (WS 2025/26)  
> Team: Ka Wong Hui, Min-Han Yeh

## Overview

This project implements a **Flow Matching** framework for multi-step time-series forecasting on high-frequency vibration data. We compare two variants:

| Model        | Prior            | Description                                                           |
| ------------ | ---------------- | --------------------------------------------------------------------- |
| **Baseline** | Gaussian N(0, I) | Standard isotropic prior — high accuracy but many NFEs needed         |
| **Improved** | OU Process + OT  | Ornstein-Uhlenbeck prior with Optimal Transport coupling — fewer NFEs |

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

## Run with Google Colab

Use this workflow if you want to run everything in Colab with a GPU.

### 1. Create a new Colab notebook

- Open Colab and create a notebook.
- Go to Runtime -> Change runtime type.
- Set Hardware accelerator to GPU.

### 2. Get the project into Colab

Option A (recommended): clone from GitHub

```bash
!git clone https://github.com/keanehui/gnn-project.git

%cd gnn-project
```

Option B: upload a ZIP of this project, then unzip

```bash
!unzip gnn-project.zip
%cd gnn-project
```

### 3. Install dependencies

```bash
!pip install -r requirements.txt
```

### 4. Download dataset

```bash
!python3 data/download.py
```

This downloads the CWRU .mat files into `data/raw`.

### 5. Train models

Train baseline:

```bash
!python3 train.py --model baseline
```

Train improved:

```bash
!python3 train.py --model improved
```

Model checkpoints and logs are saved under `outputs/baseline/model` and `outputs/improved/model`.

### 6. Evaluate models

```bash
!python3 evaluate.py --checkpoint outputs/baseline/model/best.pt --model baseline --nfe 16 --plot
!python3 evaluate.py --checkpoint outputs/improved/model/best.pt --model improved --nfe 4 --plot
```

Evaluation figures and metrics are saved under each model folder in `outputs/.../evaluate`.

### 7. Run NFE ablation

```bash
!python3 comparison.py \
    --checkpoint_baseline outputs/baseline/model/best.pt \
    --checkpoint_improved outputs/improved/model/best.pt
```

Ablation CSVs and plots are saved in `outputs/ablation`.

### 8. (Optional) Save outputs to Google Drive

Mount Drive:

```python
from google.colab import drive
drive.mount('/content/drive')
```

Copy outputs:

```bash
!cp -r outputs /content/drive/MyDrive/gnn-project-outputs
```

### Colab Notes

- If Colab disconnects, rerun setup cells and continue training with:

```bash
!python3 train.py --model baseline --resume outputs/baseline/model/best.pt
!python3 train.py --model improved --resume outputs/improved/model/best.pt
```

- Background training scripts are not needed in Colab.

## Quick Start (Local)

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

---

### Part 2: Improved Model Training, Evaluation & Ablation Study (Min-Han Yeh)

#### 4. Train Improved Model

- **Reads:** Configuration from `config.yaml` and the datasets from `data/processed/`.
- **Outputs:** Model checkpoints (e.g., `best.pt`) and training logs saved to `outputs/improved/model/`.

**Foreground Training:**

```bash
python3 train.py --model improved
```

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

| Parameter                  | Default | Description                 |
| -------------------------- | ------- | --------------------------- |
| `context_length`           | 256     | Past timesteps as input     |
| `prediction_horizon`       | 64      | Future timesteps to predict |
| `tcn.hidden_channels`      | 128     | TCN hidden dimension        |
| `tcn.num_layers`           | 6       | Number of TCN blocks        |
| `ou_prior.theta`           | 1.0     | OU mean reversion speed     |
| `training.epochs`          | 200     | Training epochs             |
| `training.batch_size`      | 128     | Batch size                  |
| `training.mixed_precision` | true    | Use AMP on NVIDIA GPUs      |

## Dataset

**CWRU Bearing Fault Dataset** — 12,000 samples/second Drive End accelerometer data from Case Western Reserve University. Includes normal operation and three fault conditions (inner race, ball, outer race at 0.007").

## GPU Training

The project is designed for **NVIDIA GPU** training with:

- Automatic CUDA detection
- Mixed precision (`torch.cuda.amp`) for ~2x speedup
- Gradient clipping and cosine annealing LR

## References

- Lipman et al. (2023). _Flow Matching for Generative Modeling_
- Tong et al. (2023). _Improving and Generalizing Flow-Based Generative Models with Minibatch Optimal Transport_
- Kollovieh et al. _TSFlow_
- CWRU Bearing Data Center: https://engineering.case.edu/bearingdatacenter
