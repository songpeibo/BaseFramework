"""DataLoader construction."""

from typing import Any

import torch
from torch.utils.data import DataLoader

from data.dataset import HyperspectralDataset, load_mat_dataset, split_indices


def build_dataloaders(config: dict[str, Any]) -> dict[str, DataLoader]:
    """Build train/val/test DataLoaders from configuration.

    Args:
        config: Merged configuration dictionary.

    Returns:
        Dictionary with keys ``train``, ``val``, ``test``.
    """
    data_cfg = config["data"]
    ds_cfg = config["dataset"]
    train_cfg = config["train"]

    raw_dir = data_cfg["raw_dir"]
    mat_path = f"{raw_dir}/{ds_cfg['mat_file']}"

    data, labels = load_mat_dataset(
        mat_path,
        data_key=ds_cfg["data_key"],
        label_key=ds_cfg["label_key"],
    )

    train_idx, val_idx, test_idx = split_indices(
        labels,
        train_ratio=data_cfg["train_ratio"],
        val_ratio=data_cfg["val_ratio"],
        seed=config["project"]["seed"],
    )

    common_kwargs = {
        "data": data,
        "labels": labels,
        "patch_size": data_cfg["patch_size"],
        "normalize": data_cfg["normalize"],
    }

    datasets = {
        "train": HyperspectralDataset(indices=train_idx, **common_kwargs),
        "val": HyperspectralDataset(indices=val_idx, **common_kwargs),
        "test": HyperspectralDataset(indices=test_idx, **common_kwargs),
    }

    loaders = {}
    for split, dataset in datasets.items():
        batch_size = (
            train_cfg["batch_size"] if split == "train" else config["test"]["batch_size"]
        )
        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=data_cfg["num_workers"],
            pin_memory=torch.cuda.is_available(),
            drop_last=(split == "train"),
        )

    return loaders
