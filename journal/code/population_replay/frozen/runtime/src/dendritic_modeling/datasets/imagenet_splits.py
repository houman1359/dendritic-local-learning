"""Subsetting and split helpers for ImageNet datasets."""

import logging

import torch
from torch.utils.data import Dataset, Subset, TensorDataset

from .paths import split_generator

_logger = logging.getLogger(__name__)

IMAGENET_FLATTEN_MATERIALIZATION_LIMIT = 50_000


def _class_subset_indices(
    dataset: Dataset,
    max_per_class: int | None,
    split_name: str,
    *,
    seed: int,
) -> list[int] | None:
    """Return indices for a fixed number of samples per class."""
    if max_per_class is None:
        return None
    if max_per_class <= 0:
        raise ValueError(
            f"{split_name}_samples_per_class must be positive, got {max_per_class}"
        )

    targets = getattr(dataset, "targets", None)
    if targets is None:
        _logger.warning(
            "Dataset %s has no targets attribute; cannot apply %s_samples_per_class=%s",
            type(dataset).__name__,
            split_name,
            max_per_class,
        )
        return None

    per_class_indices: dict[int, list[int]] = {}
    for idx, target in enumerate(targets):
        per_class_indices.setdefault(int(target), []).append(idx)

    generator = split_generator(seed)
    selected_indices = []
    for label in sorted(per_class_indices):
        indices = per_class_indices[label]
        order = torch.randperm(len(indices), generator=generator).tolist()
        selected_indices.extend(indices[i] for i in order[:max_per_class])
    selected_indices.sort()

    _logger.info(
        "Subsetting %s split to %d samples across %d classes (%d per class cap)",
        split_name,
        len(selected_indices),
        len(per_class_indices),
        max_per_class,
    )
    return selected_indices


def _subset_by_indices(dataset: Dataset, indices: list[int] | None) -> Dataset:
    return Subset(dataset, indices) if indices is not None else dataset


def _split_train_valid_pair(
    train_dataset: Dataset,
    train_eval_dataset: Dataset,
    *,
    train_valid_split: float,
    split_seed: int,
) -> tuple[Dataset, Dataset]:
    train_size = int(len(train_dataset) * train_valid_split)
    valid_size = len(train_dataset) - train_size
    indices = torch.randperm(
        len(train_dataset), generator=split_generator(split_seed)
    ).tolist()
    train_indices = indices[:train_size]
    valid_indices = indices[train_size : train_size + valid_size]
    return Subset(train_dataset, train_indices), Subset(
        train_eval_dataset, valid_indices
    )


def _apply_imagenet_class_subsets(
    *,
    train_dataset: Dataset,
    train_eval_dataset: Dataset,
    test_dataset: Dataset,
    train_samples_per_class: int | None,
    val_samples_per_class: int | None,
    split_seed: int,
) -> tuple[Dataset, Dataset, Dataset]:
    """Apply deterministic per-class train/test subsetting."""
    train_indices = _class_subset_indices(
        train_dataset,
        train_samples_per_class,
        split_name="train",
        seed=int(split_seed),
    )
    test_indices = _class_subset_indices(
        test_dataset,
        val_samples_per_class,
        split_name="val",
        seed=int(split_seed) + 1,
    )
    return (
        _subset_by_indices(train_dataset, train_indices),
        _subset_by_indices(train_eval_dataset, train_indices),
        _subset_by_indices(test_dataset, test_indices),
    )


def _split_imagenet_train_valid_or_use_test(
    *,
    train_dataset: Dataset,
    train_eval_dataset: Dataset,
    test_dataset: Dataset,
    train_valid_split: float,
    split_seed: int,
) -> tuple[Dataset, Dataset]:
    """Split train/validation when possible, otherwise reuse the test dataset."""
    if hasattr(train_dataset, "__len__"):
        return _split_train_valid_pair(
            train_dataset,
            train_eval_dataset,
            train_valid_split=train_valid_split,
            split_seed=split_seed,
        )
    return train_dataset, test_dataset


def _materialize_flattened_tensor_dataset(
    dataset: Dataset,
    *,
    max_samples: int = IMAGENET_FLATTEN_MATERIALIZATION_LIMIT,
) -> Dataset:
    """Convert an indexable image dataset into a capped flattened TensorDataset."""
    if not hasattr(dataset, "__getitem__"):
        return dataset

    data_list = []
    label_list = []
    subset_size = min(len(dataset), max_samples)
    for i in range(subset_size):
        data, label = dataset[i]
        if isinstance(data, torch.Tensor) and len(data.shape) > 1:
            data = data.view(-1)
        data_list.append(data)
        label_list.append(label)

    if not data_list:
        return dataset

    data_tensor = torch.stack(data_list)
    label_tensor = torch.tensor(label_list)
    return TensorDataset(data_tensor, label_tensor)


def _maybe_materialize_flattened_imagenet_triplet(
    train_dataset: Dataset,
    valid_dataset: Dataset,
    test_dataset: Dataset,
    *,
    flatten: bool,
) -> tuple[Dataset, Dataset, Dataset]:
    """Materialize flattened datasets only when explicitly requested."""
    if not flatten:
        return train_dataset, valid_dataset, test_dataset
    return (
        _materialize_flattened_tensor_dataset(train_dataset),
        _materialize_flattened_tensor_dataset(valid_dataset),
        _materialize_flattened_tensor_dataset(test_dataset),
    )


def _prepare_imagenet_dataset_triplet(
    *,
    train_dataset: Dataset,
    train_eval_dataset: Dataset,
    test_dataset: Dataset,
    train_valid_split: float,
    train_samples_per_class: int | None,
    val_samples_per_class: int | None,
    split_seed: int,
    flatten: bool,
) -> tuple[Dataset, Dataset, Dataset]:
    """Apply ImageNet subsetting, validation split, and optional flattening."""
    train_dataset, train_eval_dataset, test_dataset = _apply_imagenet_class_subsets(
        train_dataset=train_dataset,
        train_eval_dataset=train_eval_dataset,
        test_dataset=test_dataset,
        train_samples_per_class=train_samples_per_class,
        val_samples_per_class=val_samples_per_class,
        split_seed=split_seed,
    )
    train_dataset, valid_dataset = _split_imagenet_train_valid_or_use_test(
        train_dataset=train_dataset,
        train_eval_dataset=train_eval_dataset,
        test_dataset=test_dataset,
        train_valid_split=train_valid_split,
        split_seed=split_seed,
    )
    return _maybe_materialize_flattened_imagenet_triplet(
        train_dataset,
        valid_dataset,
        test_dataset,
        flatten=flatten,
    )


__all__ = [
    "IMAGENET_FLATTEN_MATERIALIZATION_LIMIT",
    "_apply_imagenet_class_subsets",
    "_class_subset_indices",
    "_materialize_flattened_tensor_dataset",
    "_maybe_materialize_flattened_imagenet_triplet",
    "_prepare_imagenet_dataset_triplet",
    "_split_imagenet_train_valid_or_use_test",
    "_split_train_valid_pair",
    "_subset_by_indices",
]
