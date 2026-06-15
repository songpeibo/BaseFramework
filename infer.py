"""Inference entry point.

Usage:
    python infer.py --config paviau.yaml --checkpoint best.pth
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from configs import load_config
from data.dataset import HyperspectralDataset, load_mat_dataset
from models.net import build_model
from utils.misc import get_device, get_exp_dir, load_checkpoint


@torch.no_grad()
def predict_full_image(
    model: torch.nn.Module,
    data: np.ndarray,
    labels: np.ndarray,
    patch_size: int,
    device: torch.device,
    batch_size: int = 128,
) -> np.ndarray:
    """Run sliding-window inference on the full hyperspectral image."""
    model.eval()
    h, w, _ = data.shape
    pred_map = np.zeros((h, w), dtype=np.int64)

    valid_mask = labels > 0
    indices = np.argwhere(valid_mask)

    dataset = HyperspectralDataset(
        data=data, labels=labels, patch_size=patch_size,
        indices=indices, normalize=True,
    )
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=0,
    )

    offset = 0
    for patches, _ in loader:
        patches = patches.to(device)
        outputs = model(patches)
        preds = outputs.argmax(dim=1).cpu().numpy()
        batch_len = len(preds)
        for i in range(batch_len):
            row, col = indices[offset + i]
            pred_map[row, col] = preds[i] + 1
        offset += batch_len

    return pred_map


def main() -> None:
    parser = argparse.ArgumentParser(description="Run inference with a trained model")
    parser.add_argument("--config", type=str, default="paviau.yaml")
    parser.add_argument("--checkpoint", type=str, default="best.pth")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    device = get_device(config["project"]["device"])
    data_cfg = config["data"]
    ds_cfg = config["dataset"]

    exp_dir = get_exp_dir(config)
    ckpt_path = exp_dir / "checkpoints" / args.checkpoint

    data, labels = load_mat_dataset(
        f"{data_cfg['raw_dir']}/{ds_cfg['mat_file']}",
        data_key=ds_cfg["data_key"],
        label_key=ds_cfg["label_key"],
    )

    model = build_model(config).to(device)
    load_checkpoint(model, ckpt_path)

    print(f"Running inference on {ds_cfg['name']} ...")
    pred_map = predict_full_image(
        model, data, labels,
        patch_size=data_cfg["patch_size"],
        device=device,
        batch_size=config["test"]["batch_size"],
    )

    output_path = args.output or str(exp_dir / "results" / "prediction.npy")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, pred_map)
    print(f"Prediction saved to {output_path}")


if __name__ == "__main__":
    main()
