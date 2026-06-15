"""Hyperspectral image dataset."""

from pathlib import Path
from typing import Any

import numpy as np
import scipy.io as sio
import torch
from torch.utils.data import Dataset


class HyperspectralDataset(Dataset):
    """Patch-based hyperspectral image classification dataset."""

    def __init__(
        self,
        data: np.ndarray,
        labels: np.ndarray,
        patch_size: int = 9,
        indices: np.ndarray | None = None,
        normalize: bool = True,
    ) -> None:
        """
        Args:
            data: Hyperspectral cube of shape (H, W, C).
            labels: Ground-truth label map of shape (H, W).
            patch_size: Spatial patch size (must be odd).
            indices: Optional array of flat pixel indices to include.
            normalize: Whether to apply per-band z-score normalization.
        """
        self.data = data.astype(np.float32)
        self.labels = labels.astype(np.int64)
        self.patch_size = patch_size
        self.pad = patch_size // 2
        self.normalize = normalize

        if indices is None:
            valid_mask = labels > 0
            self.indices = np.argwhere(valid_mask)
        else:
            self.indices = indices

        if self.normalize:
            self._normalize_data()

    def _normalize_data(self) -> None:
        mean = self.data.mean(axis=(0, 1), keepdims=True)
        std = self.data.std(axis=(0, 1), keepdims=True) + 1e-8
        self.data = (self.data - mean) / std

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        row, col = self.indices[idx]
        label = self.labels[row, col] - 1  # 0-indexed

        padded = np.pad(
            self.data,
            ((self.pad, self.pad), (self.pad, self.pad), (0, 0)),
            mode="reflect",
        )
        patch = padded[
            row : row + self.patch_size,
            col : col + self.patch_size,
            :,
        ]
        # (H, W, C) -> (C, H, W)
        patch = torch.from_numpy(patch.transpose(2, 0, 1))
        label = torch.tensor(label, dtype=torch.long)
        return patch, label


def load_mat_dataset(
    mat_path: str | Path,
    data_key: str,
    label_key: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Load hyperspectral data and labels from a .mat file.

    Returns:
        Tuple of (data, labels) with shapes (H, W, C) and (H, W).
    """
    mat_path = Path(mat_path)
    if not mat_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {mat_path}")

    mat = sio.loadmat(str(mat_path))
    data = np.asarray(mat[data_key], dtype=np.float32)
    labels = np.asarray(mat[label_key], dtype=np.int64)

    if data.ndim == 3 and data.shape[0] < data.shape[2]:
        # Some .mat files store (C, H, W) — convert to (H, W, C)
        data = data.transpose(1, 2, 0)

    return data, labels


def split_indices(
    labels: np.ndarray,
    train_ratio: float = 0.1,
    val_ratio: float = 0.05,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stratified split of labeled pixel indices into train/val/test."""
    rng = np.random.default_rng(seed)
    valid_mask = labels > 0
    all_indices = np.argwhere(valid_mask)

    train_indices, val_indices, test_indices = [], [], []

    unique_classes = np.unique(labels[valid_mask])
    for cls in unique_classes:
        cls_mask = labels[all_indices[:, 0], all_indices[:, 1]] == cls
        cls_indices = all_indices[cls_mask]
        rng.shuffle(cls_indices)

        n = len(cls_indices)
        n_train = max(1, int(n * train_ratio))
        n_val = max(1, int(n * val_ratio))

        train_indices.append(cls_indices[:n_train])
        val_indices.append(cls_indices[n_train : n_train + n_val])
        test_indices.append(cls_indices[n_train + n_val :])

    return (
        np.vstack(train_indices),
        np.vstack(val_indices),
        np.vstack(test_indices),
    )
