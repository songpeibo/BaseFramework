# Base Framework

A lightweight research-oriented Python framework for academic experiments.

This repository is designed as a clean project template for independent research papers. Each paper project should keep its own copy of this structure instead of being merged into a large multi-project platform.

The framework follows a simple principle:

* no `src/` directory;
* no top-level `losses/`, `metrics/`, or `visualization/` directories;
* stable project utilities are placed in `utils/`;
* temporary debugging and inspection scripts are placed in `tools/`;
* reproducible running commands are placed in `scripts/`;
* experiment outputs are saved under `experiments/`.

The current implementation contains a small hyperspectral classification demo. It is only used to verify the framework workflow. For HSI-MSI fusion, neural-field reconstruction, photometric stereo, or other research tasks, the dataset, model, loss, and metric implementations can be replaced while keeping the same project structure.

## Project structure

```text
base_framework/
├── configs/
│   ├── base.yaml
│   ├── paviau.yaml
│   └── chikusei.yaml
│
├── data/
│   ├── raw/
│   ├── processed/
│   ├── dataset.py
│   ├── dataloader.py
│   └── preprocess.py
│
├── models/
│   ├── __init__.py
│   ├── net.py
│   ├── components.py
│   └── baselines/
│
├── utils/
│   ├── __init__.py
│   ├── logger.py
│   ├── misc.py
│   ├── losses.py
│   ├── metrics.py
│   └── visualize.py
│
├── scripts/
│   ├── train.sh
│   ├── eval.sh
│   └── infer.sh
│
├── tools/
│   ├── README.md
│   ├── check_project.py
│   └── inspect_config.py
│
├── docs/
│   ├── experiment_log.md
│   └── protocol.md
│
├── experiments/
│   └── .gitkeep
│
├── train.py
├── eval.py
├── infer.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Directory conventions

### `configs/`

Configuration files are stored here.

`base.yaml` defines common options, while dataset- or experiment-specific files override only the necessary fields. Configuration loading uses recursive deep merge, so fields in `base.yaml` are not accidentally removed when loading a specific config.

Example:

```bash
python tools/inspect_config.py --config paviau.yaml
```

### `data/`

Dataset loading and preprocessing code are stored here.

Raw datasets should be placed under `data/raw/`, and processed data can be placed under `data/processed/`. Large data files are ignored by Git by default.

The current demo uses a hyperspectral classification-style dataset. For a new paper project, replace `data/dataset.py` and `data/dataloader.py` according to the task.

### `models/`

Model definitions are stored here.

Use this directory for the proposed method, common model components, and optional baselines. Avoid placing temporary test models here. Temporary model checks should go to `tools/`.

### `utils/`

Stable project utilities are stored here.

This directory may include:

* configuration helpers;
* logging utilities;
* random seed control;
* checkpoint saving and loading;
* loss functions;
* evaluation metrics;
* visualization functions;
* experiment directory helpers.

Losses, metrics, and visualization functions are intentionally kept in `utils/` instead of being split into top-level directories. They are considered stable project utilities rather than independent research modules.

### `scripts/`

Reproducible running scripts are stored here.

Scripts in this directory should correspond to stable workflows, such as training, evaluation, inference, and result export. They should not contain temporary debugging logic.

### `tools/`

Temporary tools are stored here.

This directory is for one-off checks, debugging scripts, inspection scripts, and lightweight utilities. Files in `tools/` should not be required by the main training, evaluation, or inference pipeline.

In principle, deleting `tools/` should not break the core project workflow.

### `experiments/`

All experiment outputs are saved here.

Each experiment should create a separate directory:

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

Experiment outputs are ignored by Git by default.

## Installation

Create an environment and install the required packages:

```bash
pip install -r requirements.txt
```

If PyTorch is not installed by the requirement file or needs a specific CUDA version, install the correct PyTorch build first according to the server environment.

## Basic checks

Before running experiments, check the project structure:

```bash
python tools/check_project.py
```

Inspect the merged configuration:

```bash
python tools/inspect_config.py --config paviau.yaml
```

The configuration check should confirm that important fields such as `model.name`, `experiment.save_dir`, `experiment.exp_name`, and `train.epochs` are preserved after deep merging.

## Training

Run training with:

```bash
python train.py --config paviau.yaml
```

Or use the script:

```bash
bash scripts/train.sh
```

During training, the framework saves:

* the merged configuration;
* the command used to launch the run;
* environment information;
* checkpoints;
* logs;
* per-epoch metrics in JSONL format.

## Evaluation

Run evaluation with:

```bash
python eval.py --config paviau.yaml
```

Or use the script:

```bash
bash scripts/eval.sh
```

Evaluation results are saved under the corresponding experiment directory.

## Inference

Run inference with:

```bash
python infer.py --config paviau.yaml --checkpoint best.pth
```

Or use the script:

```bash
bash scripts/infer.sh
```

Inference outputs are saved under `results/` by default.

## Experiment record

Important experiments should be recorded in:

```text
docs/experiment_log.md
```

A recommended format is:

```text
## YYYY-MM-DD Experiment name

Config:
- model:
- dataset:
- seed:
- protocol:

Result:
- metric 1:
- metric 2:

Notes:
-
```

This file is used to keep track of meaningful experiments, failed attempts, and final results for paper writing.

## Adapting this framework to a new paper

When starting a new paper project, copy this framework and modify only the task-related parts.

Usually, the files that need task-specific changes are:

```text
configs/
data/dataset.py
data/dataloader.py
models/
utils/losses.py
utils/metrics.py
utils/visualize.py
train.py
eval.py
infer.py
```

The following conventions should remain unchanged unless there is a strong reason:

```text
utils/        stable project utilities
tools/        temporary tools
scripts/      reproducible commands
experiments/  generated experiment outputs
configs/      experiment configurations
```

Do not add a `src/` directory.
Do not create top-level `losses/`, `metrics/`, or `visualization/` directories.
Do not place stable project functions in `tools/`.

## Git policy

The repository should track code, configurations, scripts, and documentation.

The repository should not track:

* datasets;
* checkpoints;
* generated experiment results;
* large `.mat`, `.npy`, `.npz`, `.h5`, or `.hdf5` files;
* cache files;
* temporary outputs.

Use `experiments/` for generated results and summarize important findings in `docs/experiment_log.md`.
