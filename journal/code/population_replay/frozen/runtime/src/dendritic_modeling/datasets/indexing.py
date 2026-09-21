"""Dataset adapters for robust slice and batch-style indexing."""

from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset, TensorDataset


def _normalize_dataset_index(idx: Any, length: int) -> tuple[Any, bool]:
    """Normalize scalar/slice/batch indices into scalar int or list[int]."""
    if isinstance(idx, slice):
        return list(range(*idx.indices(length))), False

    if isinstance(idx, torch.Tensor):
        if idx.ndim == 0:
            return int(idx.item()), True
        return [int(i) for i in idx.detach().cpu().reshape(-1).tolist()], False

    if isinstance(idx, np.ndarray):
        if idx.ndim == 0:
            return int(idx.item()), True
        return [int(i) for i in idx.reshape(-1).tolist()], False

    if isinstance(idx, Sequence) and not isinstance(idx, (str, bytes)):
        return [int(i) for i in idx], False

    return int(idx), True


class SliceSafeDataset(Dataset):
    """
    Dataset adapter that adds robust list/slice/tensor indexing support.

    Scalar access is delegated unchanged. Batch-style indexing is materialized by
    stacking scalar reads, which keeps behavior consistent for wrapped datasets.
    """

    def __init__(self, base_dataset: Dataset):
        super().__init__()
        self.base_dataset = base_dataset

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        normalized_idx, is_scalar = _normalize_dataset_index(
            idx, len(self.base_dataset)
        )

        if is_scalar:
            return self.base_dataset[normalized_idx]

        samples = [self.base_dataset[i] for i in normalized_idx]
        if len(samples) == 0:
            return torch.empty((0,)), torch.empty((0,), dtype=torch.long)

        xs, ys = zip(*samples)
        xs = [x if torch.is_tensor(x) else torch.as_tensor(x) for x in xs]
        ys = [y if torch.is_tensor(y) else torch.as_tensor(y) for y in ys]
        return torch.stack(xs, dim=0), torch.stack(ys, dim=0)


def _ensure_slice_safe_dataset(dataset: Dataset) -> Dataset:
    if isinstance(dataset, (TensorDataset, SliceSafeDataset)):
        return dataset
    return SliceSafeDataset(dataset)


def _ensure_slice_safe_triplet(
    train_ds: Dataset, valid_ds: Dataset, test_ds: Dataset
) -> tuple[Dataset, Dataset, Dataset]:
    return (
        _ensure_slice_safe_dataset(train_ds),
        _ensure_slice_safe_dataset(valid_ds),
        _ensure_slice_safe_dataset(test_ds),
    )


def _dataset_mapping_to_triplet(
    data: dict[str, Dataset],
    *,
    valid_fallback_to_test: bool = False,
    slice_safe: bool = False,
) -> tuple[Dataset, Dataset, Dataset]:
    """Convert a dataset mapping into a train/valid/test triplet."""
    train_ds = data["train"]
    if valid_fallback_to_test:
        valid_ds = data["valid"] if "valid" in data else data["test"]
    else:
        valid_ds = data["valid"]
    test_ds = data["test"]

    if slice_safe:
        return _ensure_slice_safe_triplet(train_ds, valid_ds, test_ds)
    return train_ds, valid_ds, test_ds


__all__ = [
    "SliceSafeDataset",
    "_dataset_mapping_to_triplet",
    "_ensure_slice_safe_dataset",
    "_ensure_slice_safe_triplet",
    "_normalize_dataset_index",
]
