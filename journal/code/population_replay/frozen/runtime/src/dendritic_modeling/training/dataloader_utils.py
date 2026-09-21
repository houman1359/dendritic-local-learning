"""Shared DataLoader tuning helpers for training entrypoints."""

from __future__ import annotations

import inspect
import random
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from dendritic_modeling.training.dataloader_profiles import (
    DatasetLoaderProfile,
    dataset_loader_profile,
    dataset_type_name,
    prefers_main_process_loading,
)
from dendritic_modeling.training.dataloader_resources import (
    auto_worker_cpu_budget,
    visible_cpu_count,
)

_IMAGE_DATASET_PREFETCH_FACTOR = 4
_CUDA_WORKER_CAP = 16
_CPU_WORKER_CAP = 8
_CPU_RESERVED_MAIN_PROCESS_WORKERS = 1
_DATALOADER_SEED_MODULUS = (1 << 63) - 1
_DATALOADER_STREAM_STRIDE = 1_000_003


def seed_dataloader_worker(_worker_id: int) -> None:
    """Seed Python and NumPy from PyTorch's deterministic worker seed."""
    worker_seed = int(torch.initial_seed() % (2**32))
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def seeded_dataloader_kwargs(
    seed: int,
    *,
    stream: int = 0,
    rank: int = 0,
) -> dict[str, Any]:
    """Return an independent generator plus a picklable worker seed hook."""
    resolved_seed = (
        int(seed) + int(stream) * _DATALOADER_STREAM_STRIDE + int(rank)
    ) % _DATALOADER_SEED_MODULUS
    generator = torch.Generator(device="cpu")
    generator.manual_seed(resolved_seed)
    return {
        "generator": generator,
        "worker_init_fn": seed_dataloader_worker,
    }


def dataloader_supports_kwarg(name: str) -> bool:
    """Return whether the installed PyTorch DataLoader accepts ``name``."""
    return name in inspect.signature(DataLoader).parameters


@dataclass(frozen=True)
class DataLoaderTuning:
    """Resolved DataLoader knobs after applying auto defaults and validation."""

    num_workers: int
    pin_memory: bool
    persistent_workers: bool
    prefetch_factor: int | None
    in_order: bool | None
    available_cpus: int
    auto_num_workers: bool

    def as_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
            "persistent_workers": self.persistent_workers,
        }
        if self.prefetch_factor is not None:
            kwargs["prefetch_factor"] = self.prefetch_factor
        if self.in_order is not None and dataloader_supports_kwarg("in_order"):
            kwargs["in_order"] = self.in_order
        return kwargs

    def summary(self, dataset: Dataset) -> str:
        """Return a compact human-readable summary of resolved loader tuning."""
        dataset_name = dataset_type_name(dataset)
        return (
            f"dataset={dataset_name}, num_workers={self.num_workers}, "
            f"pin_memory={self.pin_memory}, "
            f"persistent_workers={self.persistent_workers}, "
            f"prefetch_factor={self.prefetch_factor}, "
            f"in_order={self.in_order}, "
            f"available_cpus={self.available_cpus}, "
            f"auto_num_workers={self.auto_num_workers}"
        )


def auto_num_workers_for_dataset(
    dataset: Dataset,
    *,
    cuda_available: bool | None = None,
    available_cpus: int | None = None,
) -> int:
    """Choose a conservative worker count for a training dataset."""
    return _auto_num_workers_for_profile(
        dataset_loader_profile(dataset),
        cuda_available=cuda_available,
        available_cpus=available_cpus,
    )


def _auto_num_workers_for_profile(
    dataset_profile: DatasetLoaderProfile,
    *,
    cuda_available: bool | None,
    available_cpus: int | None,
) -> int:
    if dataset_profile.prefers_main_process_loading:
        return 0

    cpus = visible_cpu_count() if available_cpus is None else max(1, available_cpus)
    worker_cpu_budget = auto_worker_cpu_budget(cpus)
    use_cuda = torch.cuda.is_available() if cuda_available is None else cuda_available
    if use_cuda:
        return min(_CUDA_WORKER_CAP, worker_cpu_budget)
    return min(
        _CPU_WORKER_CAP,
        max(0, worker_cpu_budget - _CPU_RESERVED_MAIN_PROCESS_WORKERS),
    )


def _resolve_num_workers(
    dataset_profile: DatasetLoaderProfile,
    num_workers: int | None,
    *,
    available_cpus: int,
) -> tuple[int, bool]:
    auto_num_workers = num_workers is None
    if auto_num_workers:
        return (
            _auto_num_workers_for_profile(
                dataset_profile,
                cuda_available=torch.cuda.is_available(),
                available_cpus=available_cpus,
            ),
            True,
        )

    resolved_workers = max(0, int(num_workers))
    return min(resolved_workers, available_cpus), False


def _resolve_device_type(device: torch.device | str | None) -> str:
    """Resolve the device type used for DataLoader pin-memory defaults."""

    if device is not None:
        return torch.device(device).type
    return "cuda" if torch.cuda.is_available() else "cpu"


def _resolve_pin_memory(
    *,
    device: torch.device | str | None,
    pin_memory: bool | None,
) -> bool:
    device_type = _resolve_device_type(device)
    return bool(pin_memory) if pin_memory is not None else device_type == "cuda"


def _resolve_persistent_workers(
    persistent_workers: bool | None,
    *,
    num_workers: int,
) -> bool:
    if persistent_workers is None:
        return num_workers > 0

    resolved_persistent = bool(persistent_workers)
    if resolved_persistent and num_workers == 0:
        raise ValueError("persistent_workers=True requires num_workers > 0")
    return resolved_persistent


def _resolve_prefetch_factor(
    prefetch_factor: int | None,
    *,
    dataset_profile: DatasetLoaderProfile,
    num_workers: int,
) -> int | None:
    if prefetch_factor is None:
        if num_workers > 0 and dataset_profile.is_torchvision_image_dataset:
            return _IMAGE_DATASET_PREFETCH_FACTOR
        return None
    if num_workers == 0:
        raise ValueError("prefetch_factor requires num_workers > 0")
    return max(1, int(prefetch_factor))


def _resolve_in_order(
    in_order: bool | None,
    *,
    dataset_profile: DatasetLoaderProfile | None = None,
    num_workers: int = 0,
) -> bool | None:
    if in_order is not None:
        return bool(in_order)
    if (
        dataset_profile is not None
        and num_workers > 0
        and dataset_profile.is_torchvision_image_dataset
    ):
        return False
    return None


def resolve_dataloader_tuning(
    dataset: Dataset,
    *,
    device: torch.device | str | None = None,
    num_workers: int | None = None,
    pin_memory: bool | None = None,
    persistent_workers: bool | None = None,
    prefetch_factor: int | None = None,
    in_order: bool | None = None,
) -> DataLoaderTuning:
    """Resolve DataLoader worker/pinning/prefetch settings.

    Tensor-backed datasets default to single-process loading because worker
    startup and IPC usually dominate. Lazy image/file-backed datasets, including
    ImageNet ``ImageFolder``/``ImageNet`` subsets, keep multi-worker loading.
    """
    dataset_profile = dataset_loader_profile(dataset)
    available_cpus = visible_cpu_count()
    resolved_workers, auto_num_workers = _resolve_num_workers(
        dataset_profile,
        num_workers,
        available_cpus=available_cpus,
    )
    resolved_pin_memory = _resolve_pin_memory(
        device=device,
        pin_memory=pin_memory,
    )
    resolved_persistent = _resolve_persistent_workers(
        persistent_workers,
        num_workers=resolved_workers,
    )
    resolved_prefetch = _resolve_prefetch_factor(
        prefetch_factor,
        dataset_profile=dataset_profile,
        num_workers=resolved_workers,
    )
    resolved_in_order = _resolve_in_order(
        in_order,
        dataset_profile=dataset_profile,
        num_workers=resolved_workers,
    )

    return DataLoaderTuning(
        num_workers=resolved_workers,
        pin_memory=resolved_pin_memory,
        persistent_workers=resolved_persistent,
        prefetch_factor=resolved_prefetch,
        in_order=resolved_in_order,
        available_cpus=available_cpus,
        auto_num_workers=auto_num_workers,
    )


def resolve_dataloader_tuning_from_config(
    dataset: Dataset,
    config: Any,
    *,
    device: torch.device | str | None = None,
    default_num_workers: int | None = None,
) -> DataLoaderTuning:
    """Resolve DataLoader tuning from a config-like object."""
    num_workers = getattr(config, "num_workers", None)
    if num_workers is None:
        num_workers = default_num_workers
    return resolve_dataloader_tuning(
        dataset,
        device=device,
        num_workers=num_workers,
        pin_memory=getattr(config, "pin_memory", None),
        persistent_workers=getattr(config, "persistent_workers", None),
        prefetch_factor=getattr(config, "prefetch_factor", None),
        in_order=getattr(config, "in_order", None),
    )


def dataloader_kwargs_from_config(
    dataset: Dataset,
    config: Any,
    *,
    device: torch.device | str | None = None,
) -> dict[str, Any]:
    """Return validated DataLoader keyword arguments from a config-like object."""
    return resolve_dataloader_tuning_from_config(
        dataset,
        config,
        device=device,
    ).as_kwargs()


__all__ = [
    "DataLoaderTuning",
    "auto_num_workers_for_dataset",
    "dataloader_kwargs_from_config",
    "dataloader_supports_kwarg",
    "prefers_main_process_loading",
    "resolve_dataloader_tuning",
    "resolve_dataloader_tuning_from_config",
    "seed_dataloader_worker",
    "seeded_dataloader_kwargs",
    "visible_cpu_count",
]
