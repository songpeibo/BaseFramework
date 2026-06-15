"""Data preprocessing script.

Usage:
    python -m data.preprocess --config paviau.yaml
"""

import argparse
import pickle
from pathlib import Path

from configs import load_config
from data.dataset import load_mat_dataset, split_indices


def preprocess(config_name: str) -> None:
    """Load raw .mat data, split indices, and save to processed directory."""
    config = load_config(config_name)
    data_cfg = config["data"]
    ds_cfg = config["dataset"]

    raw_path = Path(data_cfg["raw_dir"]) / ds_cfg["mat_file"]
    processed_dir = Path(data_cfg["processed_dir"]) / ds_cfg["name"]
    processed_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {raw_path} ...")
    data, labels = load_mat_dataset(
        raw_path,
        data_key=ds_cfg["data_key"],
        label_key=ds_cfg["label_key"],
    )

    train_idx, val_idx, test_idx = split_indices(
        labels,
        train_ratio=data_cfg["train_ratio"],
        val_ratio=data_cfg["val_ratio"],
        seed=config["project"]["seed"],
    )

    splits = {"train": train_idx, "val": val_idx, "test": test_idx}
    for split_name, indices in splits.items():
        save_path = processed_dir / f"{split_name}_indices.pkl"
        with open(save_path, "wb") as f:
            pickle.dump(indices, f)
        print(f"  {split_name}: {len(indices)} samples -> {save_path}")

    meta = {
        "data_shape": data.shape,
        "num_classes": ds_cfg["num_classes"],
        "in_channels": ds_cfg["in_channels"],
    }
    with open(processed_dir / "meta.pkl", "wb") as f:
        pickle.dump(meta, f)

    print(f"Preprocessing complete. Output: {processed_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess hyperspectral dataset")
    parser.add_argument(
        "--config",
        type=str,
        default="paviau.yaml",
        help="Dataset config file name",
    )
    args = parser.parse_args()
    preprocess(args.config)


if __name__ == "__main__":
    main()
