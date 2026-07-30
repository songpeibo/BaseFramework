<div align="center">

# BaseFramework

### A Minimal and Reproducible PyTorch Framework for Computer Vision Research

A lightweight research codebase for organizing, running, and recording computer vision experiments.

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python\&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch\&logoColor=white)](https://pytorch.org/)
[![Last Commit](https://img.shields.io/github/last-commit/songpeibo/BaseFramework)](https://github.com/songpeibo/BaseFramework/commits/main)
[![Stars](https://img.shields.io/github/stars/songpeibo/BaseFramework?style=social)](https://github.com/songpeibo/BaseFramework)

[Overview](#overview) ·
[Highlights](#highlights) ·
[Quick Start](#quick-start) ·
[Project Structure](#project-structure) ·
[Configuration](#configuration) ·
[Adaptation](#adapting-to-a-new-project)

</div>

---

## Overview

**BaseFramework** is a compact PyTorch project template for computer vision research.

It provides a consistent workflow for:

* experiment configuration;
* dataset loading and preprocessing;
* model development;
* training, evaluation, and inference;
* checkpoint and metric management;
* environment and command recording;
* result visualization and experiment tracking.

The repository is designed to be copied and adapted for an individual research project rather than extended into a large multi-project platform.

The current implementation includes a small hyperspectral image classification example. This example is used to verify the complete research workflow and is not intended as a state-of-the-art benchmark implementation.

<!--
Optional: add a framework overview figure to docs/assets/overview.png
and uncomment the following block.

<p align="center">
  <img src="docs/assets/overview.png" width="95%" alt="BaseFramework overview">
</p>
-->

## Highlights

### Clean research structure

The codebase separates task-specific implementations, reusable utilities, executable workflows, inspection tools, documentation, and generated experiment outputs.

### Configuration-driven experiments

Common settings are defined in `configs/base.yaml`. Dataset- or experiment-specific configuration files override only the required fields through recursive configuration merging.

### Reproducible experiment records

Each training run records its merged configuration, launch command, environment information, checkpoints, logs, metrics, figures, and evaluation results.

### Complete research workflow

Training, evaluation, and inference use independent entry points while sharing the same configuration and experiment directory conventions.

### Easy project adaptation

The included hyperspectral classification example can be replaced with image fusion, reconstruction, segmentation, detection, photometric stereo, neural fields, or other computer vision tasks without redesigning the entire repository.

## Workflow

```text
Configuration
     │
     ▼
Dataset and DataLoader
     │
     ▼
Model ── Loss ── Optimizer
     │
     ▼
Training and Validation
     │
     ├── Checkpoints
     ├── Logs and metrics
     ├── Environment information
     └── Training figures
     │
     ▼
Evaluation and Inference
     │
     ├── Quantitative results
     ├── Prediction files
     └── Visualizations
```

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/songpeibo/BaseFramework.git
cd BaseFramework
```

### 2. Create an environment

```bash
conda create -n baseframework python=3.10 -y
conda activate baseframework
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

For a CUDA environment, install the PyTorch build compatible with the local CUDA version before installing the remaining dependencies.

### 3. Prepare the demonstration dataset

Place the Pavia University demonstration file at:

```text
data/raw/PaviaU.mat
```

The default configuration expects the following MATLAB keys:

```yaml
data_key: "Y"
label_key: "groundTruth"
```

When using files with different keys or filenames, update `configs/paviau.yaml` accordingly.

The expected data directory is:

```text
data/
├── raw/
│   └── PaviaU.mat
└── processed/
```

### 4. Check the project

Verify the required project structure:

```bash
python tools/check_project.py
```

Inspect the final configuration after merging `base.yaml` with the experiment-specific configuration:

```bash
python tools/inspect_config.py --config paviau.yaml
```

### 5. Train

```bash
python train.py --config paviau.yaml
```

The corresponding shell script can also be used:

```bash
bash scripts/train.sh
```

### 6. Evaluate

Evaluate the best checkpoint:

```bash
python eval.py --config paviau.yaml
```

Evaluate another checkpoint:

```bash
python eval.py \
    --config paviau.yaml \
    --checkpoint last.pth
```

Alternatively:

```bash
bash scripts/eval.sh
```

### 7. Run inference

```bash
python infer.py \
    --config paviau.yaml \
    --checkpoint best.pth
```

Specify a custom output path when needed:

```bash
python infer.py \
    --config paviau.yaml \
    --checkpoint best.pth \
    --output experiments/paviau_default/results/prediction.npy
```

Alternatively:

```bash
bash scripts/infer.sh
```

## Project Structure

```text
BaseFramework/
├── configs/                  # Base and experiment-specific configurations
│   ├── __init__.py
│   ├── base.yaml
│   ├── paviau.yaml
│   └── chikusei.yaml
│
├── data/                     # Dataset loading and preprocessing
│   ├── raw/
│   ├── processed/
│   ├── dataset.py
│   ├── dataloader.py
│   └── preprocess.py
│
├── models/                   # Proposed models and reusable components
│   ├── baselines/
│   ├── __init__.py
│   ├── components.py
│   └── net.py
│
├── utils/                    # Stable project utilities
│   ├── __init__.py
│   ├── logger.py
│   ├── losses.py
│   ├── metrics.py
│   ├── misc.py
│   └── visualize.py
│
├── scripts/                  # Reproducible execution scripts
│   ├── train.sh
│   ├── eval.sh
│   └── infer.sh
│
├── tools/                    # Inspection and debugging tools
│   ├── README.md
│   ├── check_project.py
│   └── inspect_config.py
│
├── docs/                     # Experiment protocols and research records
│   ├── experiment_log.md
│   └── protocol.md
│
├── experiments/              # Generated experiment outputs
├── train.py                  # Training entry point
├── eval.py                   # Evaluation entry point
├── infer.py                  # Inference entry point
├── requirements.txt
└── README.md
```

## Configuration

The framework uses a base configuration together with lightweight experiment-specific overrides.

### Base configuration

`configs/base.yaml` defines shared options such as:

```yaml
project:
  name: "base_framework"
  task: "classification_demo"
  seed: 42
  device: "cuda"

data:
  raw_dir: "data/raw"
  processed_dir: "data/processed"
  patch_size: 9

model:
  name: "hsi_net"
  hidden_dim: 128
  dropout: 0.3

train:
  epochs: 100
  batch_size: 64
  lr: 0.001
  scheduler: "cosine"
  early_stopping_patience: 15
  grad_clip: 1.0
  amp: true

experiment:
  exp_name: "default"
  save_dir: "experiments"
```

### Experiment-specific configuration

A dataset configuration only needs to override fields that differ from the base configuration:

```yaml
dataset:
  name: "paviau"
  mat_file: "PaviaU.mat"
  data_key: "Y"
  label_key: "groundTruth"
  in_channels: 103
  num_classes: 9

model:
  in_channels: 103
  num_classes: 9

experiment:
  exp_name: "paviau_default"
```

The final configuration is created by recursively merging the selected file with `configs/base.yaml`.

Inspect the merged result with:

```bash
python tools/inspect_config.py --config paviau.yaml
```

## Experiment Outputs

Each experiment is saved in an independent directory:

```text
experiments/<exp_name>/
├── config.yaml
├── command.txt
├── env.txt
├── checkpoints/
│   ├── best.pth
│   └── last.pth
├── logs/
│   ├── train.log
│   └── metrics.jsonl
├── results/
├── figures/
└── tables/
```

The generated files serve different purposes:

| File or directory    | Description                                             |
| -------------------- | ------------------------------------------------------- |
| `config.yaml`        | Complete merged configuration used for the run          |
| `command.txt`        | Command that launched the experiment                    |
| `env.txt`            | Runtime and environment information                     |
| `checkpoints/`       | Best and latest model states                            |
| `logs/train.log`     | Human-readable training log                             |
| `logs/metrics.jsonl` | Machine-readable per-epoch metrics                      |
| `results/`           | Evaluation and inference outputs                        |
| `figures/`           | Training curves, confusion matrices, and visual results |
| `tables/`            | Exported quantitative results                           |

Generated datasets, checkpoints, and experiment outputs are excluded from version control by default.

## Evaluation Outputs

The demonstration evaluation pipeline reports:

* overall accuracy;
* average accuracy;
* Cohen's kappa;
* F1 score;
* confusion matrix.

Evaluation results are written to:

```text
experiments/<exp_name>/results/eval_metrics.json
```

The confusion matrix is stored in the experiment figure directory.

Inference produces a full prediction map and saves it as:

```text
experiments/<exp_name>/results/prediction.npy
```

## Reproducibility

For a paper experiment, the following information should remain fixed and recorded:

* configuration file;
* dataset version and preprocessing protocol;
* random seed;
* training, validation, and test split;
* model and loss definitions;
* evaluation protocol;
* software environment;
* checkpoint selection rule;
* final launch command.

Important experiments should also be summarized in:

```text
docs/experiment_log.md
```

A recommended record is:

```markdown
## YYYY-MM-DD — Experiment name

### Setup

- Configuration:
- Dataset:
- Model:
- Seed:
- Checkpoint:
- Evaluation protocol:

### Results

- Metric 1:
- Metric 2:

### Observations

- Main finding:
- Failure case:
- Next action:
```

Dataset-independent evaluation rules should be documented in:

```text
docs/protocol.md
```

## Adapting to a New Project

BaseFramework provides stable experiment infrastructure while allowing task-specific components to be replaced.

### 1. Define the experiment

Create a new configuration file:

```text
configs/my_experiment.yaml
```

Specify only the fields that differ from `configs/base.yaml`.

### 2. Implement the data pipeline

Modify:

```text
data/dataset.py
data/dataloader.py
data/preprocess.py
```

The data loader should return the inputs and targets required by the task.

### 3. Implement the method

Place the main model in:

```text
models/net.py
```

Reusable network blocks can be placed in:

```text
models/components.py
```

Reference methods can be placed in:

```text
models/baselines/
```

### 4. Implement task-specific utilities

Modify the relevant files under `utils/`:

```text
utils/losses.py
utils/metrics.py
utils/visualize.py
```

These files should contain stable functions used by the main workflow.

### 5. Update the entry points

Adapt the task-specific logic in:

```text
train.py
eval.py
infer.py
```

Keep their responsibilities separate:

* `train.py` performs optimization and validation;
* `eval.py` performs quantitative evaluation;
* `infer.py` generates predictions or reconstructed outputs.

### 6. Add reproducible commands

Store stable commands under:

```text
scripts/
```

Use `tools/` for inspection, debugging, conversion, or validation scripts that are not required by the core workflow.

## Design Conventions

The repository follows several intentionally simple conventions:

* one repository corresponds to one research project;
* task code remains visible at the project root;
* stable utilities are placed in `utils/`;
* reproducible workflows are placed in `scripts/`;
* inspection tools are placed in `tools/`;
* generated outputs are placed in `experiments/`;
* configurations are placed in `configs/`;
* important experimental decisions are recorded in `docs/`.

The framework does not require a `src/` directory. Losses, metrics, and visualization functions remain under `utils/` unless they become substantial research modules in their own right.

## Current Scope

The included hyperspectral classification implementation demonstrates the framework workflow.

It is intended to verify:

* data loading;
* configuration merging;
* training and validation;
* checkpoint management;
* metric computation;
* experiment logging;
* evaluation;
* full-image inference;
* result export.

For a new research project, replace the demonstration-specific dataset, model, loss, metric, and visualization implementations while retaining the experiment-management structure.

## Contributing

Issues and pull requests are welcome.

When contributing, please keep changes focused and ensure that:

* stable functions are not placed in `tools/`;
* generated datasets and experiment outputs are not committed;
* new experiments include reproducible configurations or scripts;
* task-specific changes do not silently alter existing protocols;
* documentation is updated when the project workflow changes.
