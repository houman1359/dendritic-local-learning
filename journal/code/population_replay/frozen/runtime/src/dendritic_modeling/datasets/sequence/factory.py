"""Factory for sequence dataset train/valid/test triplets."""

from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import Subset, random_split

from dendritic_modeling.datasets.registry import build_registered_dataset
from dendritic_modeling.datasets.sequence.builders import (
    get_registered_sequence_dataset_names,
    get_sequence_dataset_builder,
    has_sequence_dataset_builder,
    register_sequence_dataset_builder,
    unregister_sequence_dataset_builder,
)


def _split_train_valid(
    train_full: Any,
    test_ds: Any,
    *,
    train_valid_split: float,
    split_seed: int,
) -> tuple[Any, Any, Any]:
    if bool(getattr(train_full, "is_counterfactual_paired", False)):
        if not 0.0 <= train_valid_split <= 1.0:
            raise ValueError("train_valid_split must lie in [0, 1]")
        pair_ids = getattr(train_full, "pair_ids", None)
        if not isinstance(pair_ids, torch.Tensor) or pair_ids.ndim != 1:
            raise ValueError(
                "counterfactually paired datasets must expose one-dimensional pair_ids"
            )
        if pair_ids.numel() != len(train_full):
            raise ValueError("pair_ids must have one entry per dataset sample")
        unique_pairs, counts = torch.unique(
            pair_ids.detach().cpu(), sorted=True, return_counts=True
        )
        if unique_pairs.numel() == 0 or not torch.all(counts == 2):
            raise ValueError("every counterfactual pair_id must occur exactly twice")

        n_pairs = int(unique_pairs.numel())
        n_train_pairs = int(n_pairs * train_valid_split)
        if 0.0 < train_valid_split < 1.0:
            if n_pairs == 1:
                n_train_pairs = 1
            else:
                n_train_pairs = min(max(n_train_pairs, 1), n_pairs - 1)
        permutation = torch.randperm(
            n_pairs, generator=torch.Generator().manual_seed(split_seed)
        )
        train_pair_ids = unique_pairs[permutation[:n_train_pairs]]
        train_mask = torch.isin(pair_ids.detach().cpu(), train_pair_ids)
        train_indices = torch.nonzero(train_mask, as_tuple=False).flatten().tolist()
        valid_indices = torch.nonzero(~train_mask, as_tuple=False).flatten().tolist()
        train_ds = Subset(train_full, train_indices)
        if valid_indices:
            valid_ds = Subset(train_full, valid_indices)
        else:
            valid_ds = test_ds
        return train_ds, valid_ds, test_ds

    n_train = int(len(train_full) * train_valid_split)
    n_valid = len(train_full) - n_train
    if n_valid > 0:
        train_ds, valid_ds = random_split(
            train_full,
            [n_train, n_valid],
            generator=torch.Generator().manual_seed(split_seed),
        )
    else:
        train_ds = train_full
        valid_ds = test_ds
    return train_ds, valid_ds, test_ds


def _registered_sequence_dataset_task_config(
    *,
    dataset_name: str,
    data_path: str | None,
    train_valid_split: float,
    kwargs: dict[str, Any],
) -> dict[str, Any]:
    """Build the config payload used by external sequence dataset registries."""
    return {
        "dataset": dataset_name,
        "data_path": data_path,
        "train_valid_split": train_valid_split,
        "parameters": dict(kwargs),
    }


def get_sequence_datasets(
    dataset_name: str,
    data_path: str | None = None,
    train_valid_split: float = 0.9,
    **kwargs: Any,
) -> tuple[Any, Any, Any]:
    """Factory function for sequence datasets.

    Args:
        dataset_name: Name of the dataset.
        data_path: Path for datasets that need file storage.
        train_valid_split: Fraction of training data to use for training.
        **kwargs: Dataset-specific parameters.

    Returns:
        Tuple of (train_ds, valid_ds, test_ds).
    """
    registered = build_registered_dataset(
        dataset_name,
        _registered_sequence_dataset_task_config(
            dataset_name=dataset_name,
            data_path=data_path,
            train_valid_split=train_valid_split,
            kwargs=kwargs,
        ),
    )
    if registered is not None:
        return registered

    builder = get_sequence_dataset_builder(dataset_name)

    build = builder(dataset_name, data_path, kwargs)
    if build.split_datasets is not None:
        return build.split_datasets
    if build.train_full is None or build.test_ds is None:
        raise RuntimeError(
            f"Sequence dataset builder returned no datasets: {dataset_name}"
        )

    split_seed = int(kwargs.get("split_seed", kwargs.get("seed", 42)) or 0)
    return _split_train_valid(
        build.train_full,
        build.test_ds,
        train_valid_split=train_valid_split,
        split_seed=split_seed,
    )


__all__ = [
    "get_registered_sequence_dataset_names",
    "get_sequence_dataset_builder",
    "get_sequence_datasets",
    "has_sequence_dataset_builder",
    "register_sequence_dataset_builder",
    "unregister_sequence_dataset_builder",
]
