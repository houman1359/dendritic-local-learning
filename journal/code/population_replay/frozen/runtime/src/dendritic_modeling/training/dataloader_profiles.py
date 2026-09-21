"""Dataset-profile helpers for DataLoader tuning."""

from __future__ import annotations

from dataclasses import dataclass

from torch.utils.data import ConcatDataset, Dataset, Subset, TensorDataset

try:
    from torchvision.datasets.folder import DatasetFolder as TorchvisionDatasetFolder
except Exception:  # pragma: no cover - torchvision is an optional runtime dependency
    TorchvisionDatasetFolder = None

_TORCHVISION_IMAGE_DATASET_NAMES = {
    "DatasetFolder",
    "ImageFolder",
    "ImageNet",
}


@dataclass(frozen=True)
class DatasetLoaderProfile:
    """Storage-backed dataset properties used for DataLoader tuning."""

    type_name: str
    is_torchvision_image_dataset: bool
    prefers_main_process_loading: bool


def _unwrap_dataset_adapter(dataset: Dataset, seen: set[int]) -> Dataset:
    """Unwrap single-child dataset adapters."""
    while id(dataset) not in seen:
        seen.add(id(dataset))
        if isinstance(dataset, Subset):
            dataset = dataset.dataset
            continue
        base = getattr(dataset, "base_dataset", None)
        if isinstance(base, Dataset):
            dataset = base
            continue
        wrapped = getattr(dataset, "dataset", None)
        if isinstance(wrapped, Dataset):
            dataset = wrapped
            continue
        break
    return dataset


def base_dataset(dataset: Dataset) -> Dataset:
    """Unwrap common dataset adapters to the storage-backed base dataset."""
    return _unwrap_dataset_adapter(dataset, set())


def _leaf_datasets(dataset: Dataset) -> list[Dataset]:
    """Return storage-backed leaves for composed datasets."""
    resolved_dataset = base_dataset(dataset)
    if isinstance(resolved_dataset, ConcatDataset):
        leaves = []
        for child in resolved_dataset.datasets:
            leaves.extend(_leaf_datasets(child))
        return leaves
    return [resolved_dataset]


def _is_torchvision_image_dataset_leaf(dataset: Dataset) -> bool:
    if TorchvisionDatasetFolder is not None and isinstance(
        dataset,
        TorchvisionDatasetFolder,
    ):
        return True

    dataset_cls = type(dataset)
    return (
        dataset_cls.__name__ in _TORCHVISION_IMAGE_DATASET_NAMES
        and dataset_cls.__module__.startswith("torchvision.datasets")
    )


def _module_qualified_type_name(dataset: Dataset) -> str:
    dataset_cls = type(dataset)
    return f"{dataset_cls.__module__}.{dataset_cls.__name__}"


def _prefers_main_process_loading_for_leaves(leaves: list[Dataset]) -> bool:
    return bool(leaves) and all(isinstance(leaf, TensorDataset) for leaf in leaves)


def is_torchvision_image_dataset(dataset: Dataset) -> bool:
    """Return True for torchvision image datasets that are usually I/O-bound."""
    return dataset_loader_profile(dataset).is_torchvision_image_dataset


def dataset_type_name(dataset: Dataset) -> str:
    """Return the module-qualified type name of the storage-backed dataset."""
    return dataset_loader_profile(dataset).type_name


def prefers_main_process_loading(dataset: Dataset) -> bool:
    """Return True for in-memory tensor datasets where workers add overhead."""
    return dataset_loader_profile(dataset).prefers_main_process_loading


def dataset_loader_profile(dataset: Dataset) -> DatasetLoaderProfile:
    """Return the dataset properties used by DataLoader tuning."""
    resolved_dataset = base_dataset(dataset)
    leaves = _leaf_datasets(dataset)
    return DatasetLoaderProfile(
        type_name=_module_qualified_type_name(resolved_dataset),
        is_torchvision_image_dataset=any(
            _is_torchvision_image_dataset_leaf(leaf) for leaf in leaves
        ),
        prefers_main_process_loading=_prefers_main_process_loading_for_leaves(leaves),
    )


__all__ = [
    "DatasetLoaderProfile",
    "base_dataset",
    "dataset_loader_profile",
    "dataset_type_name",
    "is_torchvision_image_dataset",
    "prefers_main_process_loading",
]
