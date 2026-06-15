"""Evaluation entry point.

Usage:
    python eval.py --config paviau.yaml
"""

import argparse
import json

import numpy as np
import torch

from configs import load_config
from data.dataloader import build_dataloaders
from models.net import build_model
from utils.logger import Logger
from utils.metrics import compute_metrics
from utils.misc import get_device, get_exp_dir, load_checkpoint
from utils.visualize import plot_confusion_matrix


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    num_classes: int,
) -> dict:
    """Run evaluation on a data loader."""
    model.eval()
    all_preds, all_labels = [], []

    for patches, labels in loader:
        patches = patches.to(device)
        outputs = model(patches)
        preds = outputs.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    return compute_metrics(
        y_true=np.array(all_labels),
        y_pred=np.array(all_preds),
        num_classes=num_classes,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained model")
    parser.add_argument("--config", type=str, default="paviau.yaml")
    parser.add_argument("--checkpoint", type=str, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    device = get_device(config["project"]["device"])

    exp_dir = get_exp_dir(config)
    results_dir = exp_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    ckpt_name = args.checkpoint or config["test"]["checkpoint"]
    ckpt_path = exp_dir / "checkpoints" / ckpt_name

    logger = Logger(exp_dir / "logs", name="eval")
    logger.info(f"Loading checkpoint: {ckpt_path}")

    loaders = build_dataloaders(config)
    model = build_model(config).to(device)
    load_checkpoint(model, ckpt_path)

    metrics = evaluate_model(
        model, loaders["test"], device, config["model"]["num_classes"],
    )

    logger.info(f"Eval OA:    {metrics['OA']:.4f}")
    logger.info(f"Eval AA:    {metrics['AA']:.4f}")
    logger.info(f"Eval Kappa: {metrics['Kappa']:.4f}")
    logger.info(f"Eval F1:    {metrics['F1']:.4f}")

    with open(results_dir / "eval_metrics.json", "w", encoding="utf-8") as f:
        json.dump(
            {k: v for k, v in metrics.items() if k != "confusion_matrix"},
            f, indent=2,
        )

    plot_confusion_matrix(
        np.array(metrics["confusion_matrix"]),
        save_path=results_dir / "confusion_matrix.png",
        title=f"Eval Confusion Matrix ({config['dataset']['name']})",
    )
    logger.info(f"Results saved to {results_dir}")


if __name__ == "__main__":
    main()
