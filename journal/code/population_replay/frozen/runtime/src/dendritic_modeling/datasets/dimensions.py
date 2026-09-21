"""Canonical input shapes for built-in vision datasets."""

from __future__ import annotations

from math import prod

DATASET_CLASS_COUNTS: dict[str, int] = {
    "mnist": 10,
    "cifar10": 10,
    "cifar100": 100,
    "imagenet": 1000,
}
DATASET_FLAT_INPUT_SHAPES: dict[str, tuple[int, ...]] = {
    "mnist": (784,),
    "fashion_mnist": (784,),
    "cifar10": (3072,),
    "cifar100": (3072,),
    "imagenet": (150528,),
}
DATASET_IMAGE_INPUT_SHAPES: dict[str, tuple[int, ...]] = {
    "mnist": (1, 28, 28),
    "fashion_mnist": (1, 28, 28),
    "cifar10": (3, 32, 32),
    "cifar100": (3, 32, 32),
    "imagenet": (3, 224, 224),
}


def dataset_input_shape(
    dataset_name: str | None,
    *,
    flatten: bool = True,
    default_input_dim: int | None = None,
) -> tuple[int, ...] | None:
    """Return the canonical input shape for a built-in dataset."""
    if not flatten and dataset_name in DATASET_IMAGE_INPUT_SHAPES:
        return DATASET_IMAGE_INPUT_SHAPES[dataset_name]

    shape = DATASET_FLAT_INPUT_SHAPES.get(str(dataset_name))
    if shape is not None:
        return shape
    if default_input_dim is None:
        return None
    return (int(default_input_dim),)


def flat_dataset_input_dim(
    dataset_name: str | None,
    *,
    default_input_dim: int,
) -> int:
    """Return the flattened input dimension for a built-in dataset."""
    shape = dataset_input_shape(
        dataset_name,
        flatten=True,
        default_input_dim=default_input_dim,
    )
    if shape is None:
        return int(default_input_dim)
    return int(prod(shape))


def dataset_class_count(
    dataset_name: str | None,
    *,
    default_class_count: int | None = None,
) -> int | None:
    """Return the canonical class count for a built-in classification dataset."""
    return DATASET_CLASS_COUNTS.get(dataset_name, default_class_count)


__all__ = [
    "DATASET_CLASS_COUNTS",
    "DATASET_FLAT_INPUT_SHAPES",
    "DATASET_IMAGE_INPUT_SHAPES",
    "dataset_class_count",
    "dataset_input_shape",
    "flat_dataset_input_dim",
]
