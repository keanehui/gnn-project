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
│   ├── flow_matching.py          # Baseline Flow Matching + model factory
│   ├── improved.py               # Improved Flow Matching (OU prior + OT)
│   └── ou_prior.py               # OU prior sampler + OT coupling
├── comparison.py                 # NFE ablation study
├── train.py                      # Training pipeline (Trainer, AMP, checkpoints, CSV logging)
└── evaluate.py                   # Unified evaluation, metrics, and visualization

Outputs generated during training/evaluation:
outputs/
├── baseline/
│   ├── model/                    # best.pt, latest.pt, and training_log.csv
│   └── evaluate/                 # report-ready metrics tables and plots
├── improved/
│   ├── model/
│   └── evaluate/
└── ablation/                     # NFE ablation CSVs and comparison plots
```

## Team Workflow

Follow the steps below in order.
These instructions are written so that each teammate can complete their part even if they did not originally design the whole project.

### Step 1. Both Teammates Do the Same Setup

First, both teammates must use the same code version and the same `config.yaml`.

Second, if using Google Colab, create a notebook and enable GPU:
Runtime -> Change runtime type -> Hardware accelerator -> GPU

The current config is suitable for a Colab NVIDIA L4 as-is:

- keep `training.batch_size: 128`
- keep `training.mixed_precision: true`
- keep `training.enable_compile: true`
- do not worry about `training.num_workers`; the code automatically forces it to `0` on Colab for stability

Third, get the project into Colab:

```bash
!git clone https://github.com/keanehui/gnn-project.git
%cd gnn-project
```

Fourth, install dependencies:

```bash
!python3 -m pip install -r requirements.txt
```

Fifth, download the dataset:

```bash
!python3 data/download.py
```

This should create:

- `data/raw/97.mat`
- `data/raw/105.mat`
- `data/raw/118.mat`
- `data/raw/130.mat`

If working locally instead of Colab, use the same commands without the `!`.
When installing packages, prefer `python3 -m pip install -r requirements.txt` so the packages go into the same Python interpreter used by `python3 train.py` and `python3 evaluate.py`.
If `which python3` shows `/usr/bin/python3` on macOS, your shell is using the system Python instead of your Conda/Python environment. In that case, either fix your shell setup first or run the project with your Conda interpreter path explicitly, for example `/opt/miniconda3/bin/python3`.
All command blocks below are written in terminal style; if you run them inside a Colab cell, add `!` in front of each command.

### Step 2. Both Teammates Follow the Same Training Rule

During training:

- only checkpoints and loss logs should be saved
- no plots should be saved
- all training outputs must stay under `outputs/<model>/model/`

After training:

- `evaluate.py` creates model-specific plots
- `comparison.py` creates the final comparison plots

### Step 3. Recommended 2-Half Split

If you want the workflow to split into two clean halves, use this division:

- **Half A: Ka Wong Hui**
  - finish everything related to the baseline model
  - hand over the full baseline output folder
  - start writing the baseline-related report sections immediately
- **Half B: Min-Han Yeh**
  - finish everything related to the improved model
  - run the final cross-model comparison after receiving the baseline folder
  - hand back the comparison outputs for the final report

This is the simplest split if Ka should be able to finish his execution work early and stop waiting.

### Step 4. Half A: Ka Wong Hui Does Only the Baseline Package

If you are Ka Wong Hui, do these steps in this exact order.

First, train the baseline model:

```bash
python3 train.py --model baseline
```

Second, make sure these files now exist:

- `outputs/baseline/model/best.pt`
- `outputs/baseline/model/latest.pt`
- `outputs/baseline/model/training_log.csv`

Third, evaluate the baseline model:

```bash
python3 evaluate.py --checkpoint outputs/baseline/model/best.pt --model baseline --nfe 16 --n_samples 20
```

Fourth, make sure these files now exist:

- `outputs/baseline/evaluate/metrics_nfe16.json`
- `outputs/baseline/evaluate/per_horizon_metrics_nfe16.csv`
- `outputs/baseline/evaluate/eval_nfe16.png`
- `outputs/baseline/evaluate/forecast_examples_nfe16.png`
- `outputs/baseline/evaluate/error_by_horizon_nfe16.png`
- `outputs/baseline/evaluate/uncertainty_by_horizon_nfe16.png`
- `outputs/baseline/evaluate/training_loss.png`

Fifth, give Min-Han the entire folder:

- `outputs/baseline/`

Do not only send screenshots.
The safest rule is: if you are unsure what to send, send the whole `outputs/baseline/` folder.

Sixth, after you send `outputs/baseline/`, your execution half is done.
You do not need to wait before starting your report writing.

Seventh, start writing these parts of the report:

- Section 1: Introduction
- Section 2.1: Application Domain
- Section 2.2: Data Description
- your baseline-related part of Section 2.3: Theory and Methodology
- your baseline-related part of Section 2.4: Related Work
- your baseline-related part of Section 3.1: Specialized Implementation
- Section 3.2: Project Planning and Resource Allocation
- your baseline-related part of Section 3.4: Success Metrics
- Section 4.1: Experimental Setup
- your baseline-related part of Section 4.2: Objective Results
- your baseline-related part of Section 4.3: Discussion

Eighth, later, when Min-Han sends back the final comparison outputs, add those shared figures only where needed.

Ka is finished with the execution half once `outputs/baseline/` is complete and has been handed to Min-Han.

### Step 5. Half B: Min-Han Yeh Does the Improved Package and Final Comparison

If you are Min-Han Yeh, do these steps in this exact order.

First, train the improved model:

```bash
python3 train.py --model improved
```

Second, make sure these files now exist:

- `outputs/improved/model/best.pt`
- `outputs/improved/model/latest.pt`
- `outputs/improved/model/training_log.csv`

Third, evaluate the improved model:

```bash
python3 evaluate.py --checkpoint outputs/improved/model/best.pt --model improved --nfe 4 --n_samples 20
```

Fourth, make sure these files now exist:

- `outputs/improved/evaluate/metrics_nfe4.json`
- `outputs/improved/evaluate/per_horizon_metrics_nfe4.csv`
- `outputs/improved/evaluate/eval_nfe4.png`
- `outputs/improved/evaluate/forecast_examples_nfe4.png`
- `outputs/improved/evaluate/error_by_horizon_nfe4.png`
- `outputs/improved/evaluate/uncertainty_by_horizon_nfe4.png`
- `outputs/improved/evaluate/training_loss.png`

Fifth, receive this folder from Ka:

- `outputs/baseline/`

Sixth, place `outputs/baseline/` and `outputs/improved/` inside the same project.

Seventh, run the final ablation study:

```bash
python3 comparison.py \
    --checkpoint_baseline outputs/baseline/model/best.pt \
    --checkpoint_improved outputs/improved/model/best.pt
```

Eighth, make sure these files now exist:

- `outputs/ablation/nfe_ablation_results.csv`
- `outputs/ablation/nfe_vs_mae.png`
- `outputs/ablation/nfe_vs_rmse.png`
- `outputs/ablation/nfe_vs_crps.png`
- `outputs/ablation/nfe_vs_latency.png`
- `outputs/ablation/nfe_accuracy_latency.png`
- `outputs/ablation/sample_predictions_baseline.png`
- `outputs/ablation/sample_predictions_improved.png`

Ninth, give Ka the folder:

- `outputs/ablation/`

Tenth, if Ka wants the full improved package too for archiving, also send:

- `outputs/improved/`

Min-Han is finished with the execution half once `outputs/improved/` and `outputs/ablation/` are complete and have been handed back as needed.

### Step 6. Exactly What Each Person Must Send

To avoid confusion, use this simple exchange rule:

- Ka sends `outputs/baseline/` to Min-Han
- Min-Han sends `outputs/ablation/` back to Ka
- Min-Han may also send `outputs/improved/` back to Ka for a full shared archive

If you are not sure which individual files are needed, do not try to guess.
Send the full folder.

### Step 7. Simple Ways to Exchange the Folders

If the two teammates work on different Colab notebooks or different machines, use one of these methods.

Option A: copy the full `outputs/` folder to Google Drive

```python
from google.colab import drive
drive.mount('/content/drive')
```

```bash
!cp -r outputs /content/drive/MyDrive/gnn-project-outputs
```

Option B: zip the folder and send it

```bash
zip -r baseline_outputs.zip outputs/baseline
zip -r improved_outputs.zip outputs/improved
zip -r ablation_outputs.zip outputs/ablation
```

### Step 8. If Colab Disconnects During Training

Rerun the setup cells, then resume from the latest checkpoint:

```bash
python3 train.py --model baseline --resume outputs/baseline/model/latest.pt
python3 train.py --model improved --resume outputs/improved/model/latest.pt
```

### Step 9. What Both Teammates Should Have Before Writing the Report

Before writing the report:

- Ka must have:
  - `outputs/baseline/`
  - `outputs/ablation/`
- Min-Han must have:
  - `outputs/improved/`
  - `outputs/baseline/`
  - `outputs/ablation/`

If Ka also wants a full local copy of everything, Min-Han can send `outputs/improved/` back too, but that is optional for the simplified two-half execution split.

### Step 10. Recommended Report Split So the Total Workload Stays Fair

The execution split above is intentionally asymmetric: Ka finishes early, and Min-Han does the final combined comparison work.
To keep the overall project fair, the report should be split so that both teammates still have enough material for about 4000 words each.

If you follow the suggested report structure shown in the course template, use this ownership split.

**Ka Wong Hui should pick up these report parts**

- `1.1 Context and Motivation`
- `1.2 Research Questions`
- `1.3 Objectives and Limitations`
- `2.1 Application Domain: Mechanical Vibration Data Analysis`
- `2.2 Data Description`
- `2.3.1 Theory and Methodology for the baseline forecasting formulation and standard flow matching`
- `2.4.1 Related Work for forecasting, TCNs, and baseline flow matching`
- `3.1.1 Specialized Implementation: preprocessing, sliding windows, normalization, TCN backbone, baseline model`
- `3.2 Project Planning and Resource Allocation`
- `3.2.1 Subgoals and Milestones`
- `3.2.2 Estimated vs. Actual Effort`
- `3.4.1 Success Metrics for baseline accuracy, uncertainty, and qualitative forecast evaluation`
- `4.1 Experimental Setup`
- `4.2.1 Objective Results for the baseline model`
- `4.3.1 Discussion of baseline behavior, strengths, and limitations`

**Min-Han Yeh should pick up these report parts**

- `2.3.2 Theory and Methodology for the OU-prior + OT improvements`
- `2.4.2 Related Work for OU priors, optimal transport, and efficiency-oriented generative modeling`
- `3.1.2 Specialized Implementation: improved OU-prior + OT model`
- `3.3 Teamwork and Collaboration`
- `3.4.2 Success Metrics for latency targets, NFE efficiency, and final comparison criteria`
- `4.2.2 Objective Results for the improved model and final comparison`
- `4.3.2 Discussion of the accuracy-latency trade-off and improved-vs-baseline comparison`
- `5 Conclusions and Outlook`
- `5.1 Summary of Findings`
- `5.2 Future Work`

This is the recommended split for both fairness and convenience:

- Ka can start writing immediately after finishing `outputs/baseline/`, without waiting for the improved model
- Min-Han owns the sections that depend on the final improved-model results and ablation outputs
- both teammates still get a full mix of background, methods, experiments, and discussion material
- both teammates should have enough material to reach about 4000 words
- Ka now also owns a clearly technical theory-and-metrics slice, so his part is not just introductory or organizational

To make this split clean, do not keep `2.3`, `2.4`, `3.1`, `3.4`, `4.2`, and `4.3` as single undivided blocks.
Instead, split them into the sub-subsections listed above and mark the main author for each one.

### Step 11. Explain the Workflow Change Honestly in the Report

The original proposal assigned the NFE ablation experiments to Ka.
The simplified two-half workflow instead lets Min-Han run the final combined ablation after receiving the baseline outputs.

This is acceptable, but the report should explain it clearly in the project-management section:

- the original plan
- what changed
- why it changed
- how the new split reduced waiting and improved parallel progress
- why the overall workload remained fair after balancing the report ownership

Before submitting the report, both teammates should also confirm that:

- each report section has a clearly marked main author
- the report includes planning vs actual effort
- the report explains how teamwork improved the project
- the experiments and plots directly support the main hypothesis:
  improved model at 4 NFEs vs baseline at 16 NFEs

## Configuration

All hyperparameters are centralized in `config.yaml`. Key settings:

| Parameter                  | Default | Description                 |
| -------------------------- | ------- | --------------------------- |
| `data.context_length`      | 256     | Past timesteps as input     |
| `data.prediction_horizon`  | 64      | Future timesteps to predict |
| `tcn.hidden_channels`      | 256     | TCN hidden dimension        |
| `tcn.num_layers`           | 6       | Number of TCN blocks        |
| `ou_prior.theta`           | 1.0     | OU mean reversion speed     |
| `training.epochs`          | 500     | Max training epochs         |
| `training.batch_size`      | 128     | Batch size                  |
| `training.mixed_precision` | true    | Use AMP on NVIDIA GPUs      |
| `training.enable_compile`  | true    | Use `torch.compile()` when available |
| `training.num_workers`     | 4       | Local loader workers; Colab auto-uses `0` |
| `ablation.num_eval_samples` | 64     | Monte Carlo samples per NFE in final comparison |

For a Colab NVIDIA L4, the checked recommendation is to keep the defaults above unchanged for the first full run.

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
