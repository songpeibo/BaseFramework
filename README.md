<div align="center">

# BaseFramework

### A Lightweight PyTorch Template for Computer Vision Research

A compact and reproducible codebase for developing, training, and evaluating computer vision models.

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python\&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch\&logoColor=white)](https://pytorch.org/)
[![Last Commit](https://img.shields.io/github/last-commit/songpeibo/BaseFramework)](https://github.com/songpeibo/BaseFramework/commits/main)

[Overview](#overview) ·
[Quick Start](#quick-start) ·
[Structure](#project-structure) ·
[Adaptation](#adapting-the-framework)

</div>

---

## Overview

**BaseFramework** is a minimal PyTorch template for independent computer vision research projects.

It provides:

* configuration-driven experiments;
* unified training, evaluation, and inference entry points;
* dataset and model interfaces;
* checkpoint and metric management;
* experiment logging and environment recording;
* reusable losses, metrics, and visualization utilities.

The repository currently includes a small hyperspectral image classification example to demonstrate the complete workflow. It is intended as a project template rather than a benchmark implementation.

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/songpeibo/BaseFramework.git
cd BaseFramework
```

### 2. Install dependencies

```bash
conda create -n baseframework python=3.10 -y
conda activate baseframework

pip install -r requirements.txt
```

Install the PyTorch build compatible with your CUDA environment when using a GPU.

### 3. Prepare the demonstration data

Place the Pavia University data at:

```text
data/raw/PaviaU.mat
```

The default configuration expects:

```yaml
data_key: "Y"
label_key: "groundTruth"
```

Update `configs/paviau.yaml` when using a file with different keys.

### 4. Inspect the project

Check the project structure:

```bash
python tools/check_project.py
```

Inspect the merged configuration:

```bash
python tools/inspect_config.py --config paviau.yaml
```

### 5. Train

```bash
python train.py --config paviau.yaml
```

or:

```bash
bash scripts/train.sh
```

### 6. Evaluate

```bash
python eval.py --config paviau.yaml
```

To evaluate a specific checkpoint:

```bash
python eval.py \
    --config paviau.yaml \
    --checkpoint last.pth
```

### 7. Run inference

```bash
python infer.py \
    --config paviau.yaml \
    --checkpoint best.pth
```

The prediction map is saved to:

```text
experiments/<exp_name>/results/prediction.npy
```

## Project Structure

```text
BaseFramework/
├── configs/          # Base and experiment configurations
├── data/             # Dataset loading and preprocessing
├── models/           # Proposed models and baselines
├── utils/            # Losses, metrics, logging, and visualization
├── scripts/          # Reproducible running commands
├── tools/            # Inspection and debugging utilities
├── docs/             # Protocols and experiment records
├── experiments/      # Generated experiment outputs
├── train.py
├── eval.py
├── infer.py
└── requirements.txt
```

The project follows several simple conventions:

* `configs/` stores experiment settings;
* `models/` contains the proposed method and reusable model components;
* `utils/` contains stable project utilities;
* `scripts/` contains reproducible commands;
* `tools/` contains temporary inspection or debugging tools;
* `experiments/` contains generated outputs.

## Configuration

Shared settings are defined in:

```text
configs/base.yaml
```

Dataset- or experiment-specific files override only the required fields:

```text
configs/paviau.yaml
configs/chikusei.yaml
```

Configuration files are recursively merged before an experiment starts.

## Experiment Outputs

Each run creates an independent directory:

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
└── figures/
```

The framework records the effective configuration, launch command, environment information, checkpoints, logs, metrics, and visual results.

## Adapting the Framework

To use BaseFramework for a new project:

1. Create an experiment configuration under `configs/`.
2. Implement the dataset interface under `data/`.
3. Replace or extend the model under `models/`.
4. Add task-specific losses and metrics under `utils/`.
5. Adapt `train.py`, `eval.py`, and `infer.py` when necessary.
6. Add stable execution commands under `scripts/`.

The framework can be adapted to tasks such as:

* image classification;
* image fusion;
* image reconstruction;
* segmentation;
* detection;
* photometric stereo;
* neural-field reconstruction.

Detailed experiment protocols and research records should be maintained under `docs/`.

## Reproducibility

For important experiments, record:

* the configuration;
* dataset and preprocessing protocol;
* random seed;
* software environment;
* checkpoint selection rule;
* evaluation protocol;
* final command and results.

Experiment notes can be maintained in:

```text
docs/experiment_log.md
```

## Contributing

Issues and pull requests are welcome. Please avoid committing datasets, checkpoints, cache files, and generated experiment outputs.
