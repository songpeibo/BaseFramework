"""Training entry point.

Usage:
    python train.py --config paviau.yaml
"""

import argparse

import numpy as np
import torch
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnneLR, StepLR
from tqdm import tqdm

from configs import load_config, save_config
from data.dataloader import build_dataloaders
from models.net import build_model
from utils.logger import Logger, log_config
from utils.losses import build_loss
from utils.metrics import compute_metrics
from utils.misc import (
    append_jsonl,
    count_parameters,
    ensure_exp_subdirs,
    get_device,
    get_exp_dir,
    save_checkpoint,
    save_command,
    save_env_info,
    set_seed,
)
from utils.visualize import plot_training_curve


def train_one_epoch(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: GradScaler | None,
    grad_clip: float,
    epoch: int,
    total_epochs: int,
) -> float:
    """Train for one epoch with a tqdm progress bar."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch}/{total_epochs} [train]", leave=False)
    for patches, labels in pbar:
        patches = patches.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        if scaler is not None:
            with autocast():
                outputs = model(patches)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(patches)
            loss = criterion(outputs, labels)
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        total_loss += loss.item()
        num_batches += 1
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / max(num_batches, 1)


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    num_classes: int,
    epoch: int,
    total_epochs: int,
    split: str = "val",
) -> tuple[float, dict[str, float]]:
    """Evaluate model with a tqdm progress bar."""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    all_preds, all_labels = [], []

    pbar = tqdm(loader, desc=f"Epoch {epoch}/{total_epochs} [{split}]", leave=False)
    for patches, labels in pbar:
        patches = patches.to(device)
        labels = labels.to(device)
        outputs = model(patches)
        loss = criterion(outputs, labels)
        total_loss += loss.item()
        num_batches += 1

        preds = outputs.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    avg_loss = total_loss / max(num_batches, 1)
    metrics = compute_metrics(
        y_true=np.array(all_labels),
        y_pred=np.array(all_preds),
        num_classes=num_classes,
    )
    return avg_loss, metrics


def build_scheduler(optimizer, config):
    """Build learning rate scheduler from config."""
    name = config["train"]["scheduler"]
    epochs = config["train"]["epochs"]
    if name == "cosine":
        return CosineAnneLR(optimizer, T_max=epochs)
    if name == "step":
        return StepLR(optimizer, step_size=epochs // 3, gamma=0.1)
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a research model")
    parser.add_argument("--config", type=str, default="paviau.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config["project"]["seed"])
    device = get_device(config["project"]["device"])

    exp_dir = get_exp_dir(config)
    subdirs = ensure_exp_subdirs(exp_dir)

    save_config(config, exp_dir / "config.yaml")
    save_command(exp_dir / "command.txt")
    save_env_info(exp_dir / "env.txt")

    logger = Logger(subdirs["logs"])
    log_config(logger, config)
    logger.info(f"project.name: {config['project']['name']}")
    logger.info(f"project.task: {config['project']['task']}")
    logger.info(f"dataset.name: {config['dataset']['name']}")
    logger.info(f"model.name: {config['model']['name']}")
    logger.info(f"seed: {config['project']['seed']}")
    logger.info(f"device: {device}")
    logger.info(f"experiment dir: {exp_dir}")

    loaders = build_dataloaders(config)
    model = build_model(config).to(device)
    num_params = count_parameters(model)
    logger.info(f"total trainable parameters: {num_params:,}")

    criterion = build_loss(config)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["train"]["lr"],
        weight_decay=config["train"]["weight_decay"],
    )
    scheduler = build_scheduler(optimizer, config)
    use_amp = config["train"]["amp"] and device.type == "cuda"
    scaler = GradScaler() if use_amp else None

    metrics_jsonl = subdirs["logs"] / "metrics.jsonl"
    best_oa = 0.0
    patience_counter = 0
    train_losses, val_losses = [], []
    total_epochs = config["train"]["epochs"]

    for epoch in range(1, total_epochs + 1):
        train_loss = train_one_epoch(
            model, loaders["train"], criterion, optimizer, device,
            scaler, config["train"]["grad_clip"], epoch, total_epochs,
        )
        val_loss, val_metrics = evaluate(
            model, loaders["val"], criterion, device,
            config["model"]["num_classes"], epoch, total_epochs,
        )
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if scheduler is not None:
            scheduler.step()

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_OA": val_metrics["OA"],
            "val_AA": val_metrics["AA"],
            "val_Kappa": val_metrics["Kappa"],
            "val_F1": val_metrics["F1"],
        }
        append_jsonl(record, metrics_jsonl)

        logger.info(
            f"Epoch [{epoch}/{total_epochs}] "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Val OA: {val_metrics['OA']:.4f} | Kappa: {val_metrics['Kappa']:.4f}"
        )

        if val_metrics["OA"] > best_oa:
            best_oa = val_metrics["OA"]
            patience_counter = 0
            save_checkpoint(
                model, optimizer, epoch, val_metrics,
                subdirs["checkpoints"] / "best.pth",
            )
            logger.info(f"  -> New best model saved (OA={best_oa:.4f})")
        else:
            patience_counter += 1

        if patience_counter >= config["train"]["early_stopping_patience"]:
            logger.info(f"Early stopping at epoch {epoch}")
            break

    save_checkpoint(
        model, optimizer, epoch, val_metrics,
        subdirs["checkpoints"] / "last.pth",
    )
    plot_training_curve(
        train_losses, val_losses,
        subdirs["figures"] / "train_curve.png",
    )
    logger.info(f"Training finished. Best OA: {best_oa:.4f}")


if __name__ == "__main__":
    main()
