"""Evaluation metrics for the project demo.

This is a stable project utility file. For HSI-MSI fusion projects, replace
with PSNR, SSIM, SAM, RMSE, ERGAS and related metrics.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int,
) -> dict[str, float]:
    """Compute classification metrics.

    Returns:
        Dictionary with OA, AA, Kappa, F1, and per-class accuracy.
    """
    oa = accuracy_score(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    per_class_acc = cm.diagonal() / (cm.sum(axis=1) + 1e-8)
    aa = per_class_acc.mean()

    return {
        "OA": float(oa),
        "AA": float(aa),
        "Kappa": float(kappa),
        "F1": float(f1),
        "per_class_acc": per_class_acc.tolist(),
        "confusion_matrix": cm.tolist(),
    }
