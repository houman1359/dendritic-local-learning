"""
Standard dataset loading utilities for dendritic modeling.

This module contains functions for loading and preprocessing standard datasets
like MNIST, CIFAR, and ImageNet, as well as synthetic datasets for theoretical validation.
"""

import importlib
import logging
import os
import pickle
import shutil
from collections.abc import Callable
from contextlib import contextmanager
from typing import Any, Optional

import torch
from torch.utils.data import Dataset, TensorDataset
from torchvision import datasets, transforms

from .imagenet import (
    IMAGENET_FLATTEN_MATERIALIZATION_LIMIT as IMAGENET_FLATTEN_MATERIALIZATION_LIMIT,
    _apply_imagenet_class_subsets as _apply_imagenet_class_subsets,
    _apply_imagenet_local_override as _apply_imagenet_local_override,
    _build_imagenet_transforms as _build_imagenet_transforms,
    _class_subset_indices as _class_subset_indices,
    _copy_dataset_with_transform as _copy_dataset_with_transform,
    _is_official_imagenet_root as _is_official_imagenet_root,
    _load_imagenet_dataset_triplet as _load_imagenet_dataset_triplet,
    _materialize_flattened_tensor_dataset as _materialize_flattened_tensor_dataset,
    _maybe_materialize_flattened_imagenet_triplet as _maybe_materialize_flattened_imagenet_triplet,
    _normalized_imagenet_loader_backend as _normalized_imagenet_loader_backend,
    _normalized_imagenet_preset_name as _normalized_imagenet_preset_name,
    _prepare_imagenet_dataset_triplet as _prepare_imagenet_dataset_triplet,
    _resolve_imagefolder_roots as _resolve_imagefolder_roots,
    _resolve_imagenet_data_location as _resolve_imagenet_data_location,
    _split_imagenet_train_valid_or_use_test as _split_imagenet_train_valid_or_use_test,
    _split_train_valid_pair as _split_train_valid_pair,
    _subset_by_indices as _subset_by_indices,
    load_imagenet_as_datasets,
)
from .indexing import (
    SliceSafeDataset as SliceSafeDataset,
    _dataset_mapping_to_triplet as _dataset_mapping_to_triplet,
    _ensure_slice_safe_dataset as _ensure_slice_safe_dataset,
    _ensure_slice_safe_triplet as _ensure_slice_safe_triplet,
    _normalize_dataset_index as _normalize_dataset_index,
)
from .paired import (
    PairedArithmeticDataset as PairedArithmeticDataset,
    _build_iterative_modulo10_dataset as _build_iterative_modulo10_dataset,
    _build_paired_arithmetic_dataset as _build_paired_arithmetic_dataset,
    _make_pair_indices as _make_pair_indices,
)
from .paths import get_data_directory, split_generator as _split_generator
from .poisson import PoissonGeneratorDataset as PoissonGeneratorDataset
from .registry import build_registered_dataset
from .synthetic_datasets import get_synthetic_datasets
from .unified_options import (
    _EXPERIMENTAL_DATASET_IMPORT_CANDIDATES,
    _THEORETICAL_SYNTHETIC_DATASET_NAMES,
    _DatasetTriplet,
    _get_config_value as _imported_get_config_value,
    _resolve_unified_dataset_options,
    _UnifiedDatasetOptions,
)

_ds_logger = logging.getLogger(__name__)
_get_config_value = _imported_get_config_value


try:
    import fcntl
except ImportError:  # pragma: no cover - only relevant on non-POSIX systems
    fcntl = None


def _apply_label_noise(
    dataset: TensorDataset,
    noise_rate: float,
    num_classes: int,
    seed: int = 0,
) -> TensorDataset:
    """Randomly flip a fraction of training labels to random classes.

    Args:
        dataset: TensorDataset with (inputs, labels).
        noise_rate: Fraction of labels to randomize (0 = clean, 1 = fully random).
        num_classes: Number of classes for random label replacement.
        seed: Random seed for reproducibility.

    Returns:
        New TensorDataset with corrupted labels.
    """
    if noise_rate <= 0.0:
        return dataset
    x, y = dataset.tensors
    n = len(y)
    n_noisy = int(n * noise_rate)
    rng = torch.Generator().manual_seed(seed)
    indices = torch.randperm(n, generator=rng)[:n_noisy]
    noisy_labels = torch.randint(
        0, num_classes, (n_noisy,), generator=rng, dtype=y.dtype
    )
    y_noisy = y.clone()
    y_noisy[indices] = noisy_labels
    actual_flipped = int((y_noisy[indices] != y[indices]).sum().item())
    _ds_logger.info(
        f"Label noise: flipped {actual_flipped}/{n} labels "
        f"(requested rate={noise_rate:.1%}, effective={actual_flipped / n:.1%})"
    )
    return TensorDataset(x, y_noisy)


def _infer_num_classes(dataset: TensorDataset) -> int:
    """Infer number of classes from dataset labels.

    Returns 0 for empty datasets. Raises ValueError if labels are
    not integer-typed (float labels are ambiguous for classification).
    """
    _, y = dataset.tensors
    if y.numel() == 0:
        return 0
    if y.is_floating_point():
        raise ValueError(
            f"Cannot infer num_classes from float labels (dtype={y.dtype}). "
            "Label noise requires integer classification labels."
        )
    return int(y.max().item()) + 1


def _maybe_apply_label_noise_to_triplet(
    train_ds: Dataset,
    valid_ds: Dataset,
    test_ds: Dataset,
    *,
    label_noise_rate: float,
    label_noise_seed: int,
) -> tuple[Dataset, Dataset, Dataset]:
    """Apply label noise to the training TensorDataset only when configured."""
    if label_noise_rate > 0.0 and isinstance(train_ds, TensorDataset):
        num_classes = _infer_num_classes(train_ds)
        if num_classes > 0:
            train_ds = _apply_label_noise(
                train_ds,
                label_noise_rate,
                num_classes,
                seed=label_noise_seed,
            )
    return train_ds, valid_ds, test_ds


def _dataset_to_tensors(
    dataset: Dataset,
    *,
    flatten: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Materialize a finite dataset using its declared length.

    Direct iteration over torch Dataset objects relies on ``__getitem__`` raising
    ``IndexError`` out of range. Some wrappers and test doubles do not enforce
    that contract, so explicit indexing by ``len(dataset)`` is safer.
    """
    data = []
    labels = []
    for idx in range(len(dataset)):
        img, label = dataset[idx]
        if not torch.is_tensor(img):
            img = torch.as_tensor(img)
        if not torch.is_tensor(label):
            label = torch.as_tensor(label)
        data.append(img)
        labels.append(label)

    data_tensor = torch.stack(data)
    label_tensor = torch.stack(labels)
    if flatten:
        data_tensor = data_tensor.view(data_tensor.size(0), -1)
    return data_tensor, label_tensor


@contextmanager
def _dataset_download_lock(root: str):
    """Serialize dataset repair/download work for a shared cache root."""
    if fcntl is None:
        yield
        return

    os.makedirs(root, exist_ok=True)
    lock_path = os.path.join(root, ".download.lock")
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def _is_recoverable_torchvision_error(exc: BaseException) -> bool:
    """Return True for missing/corrupted-cache errors that merit a clean redownload."""
    if isinstance(exc, (AssertionError, EOFError, pickle.UnpicklingError)):
        return True

    if not isinstance(exc, (RuntimeError, OSError, ValueError)):
        return False

    message = str(exc).lower()
    recoverable_patterns = (
        "dataset not found",
        "file not found or corrupted",
        "pickle data was truncated",
        "truncated",
        "corrupt",
        "corrupted",
        "invalid magic number",
        "pascalvincent",
        "no such file",
    )
    return any(pattern in message for pattern in recoverable_patterns)


def _clear_dataset_cache(root: str) -> None:
    """Remove cached dataset payload while preserving the download lock file."""
    if not os.path.isdir(root):
        return

    for entry in os.listdir(root):
        if entry == ".download.lock":
            continue
        entry_path = os.path.join(root, entry)
        try:
            if os.path.isdir(entry_path):
                shutil.rmtree(entry_path)
            else:
                os.remove(entry_path)
        except FileNotFoundError:
            continue


def _load_torchvision_train_test_pair(dataset_cls, *, root, transform, dataset_name):
    """Load a torchvision dataset pair with cache repair and serialized download."""
    with _dataset_download_lock(root):
        try:
            train_dataset = dataset_cls(
                root=root, train=True, download=False, transform=transform
            )
            test_dataset = dataset_cls(
                root=root, train=False, download=False, transform=transform
            )
            return train_dataset, test_dataset
        except Exception as exc:
            if not _is_recoverable_torchvision_error(exc):
                raise

            _ds_logger.warning(
                "Recovering %s cache at %s after load failure: %s",
                dataset_name,
                root,
                exc,
            )
            _clear_dataset_cache(root)
            train_dataset = dataset_cls(
                root=root, train=True, download=True, transform=transform
            )
            test_dataset = dataset_cls(
                root=root, train=False, download=True, transform=transform
            )
            return train_dataset, test_dataset


def _build_to_tensor_transform(
    *,
    normalize: bool,
    mean: tuple[float, ...],
    std: tuple[float, ...],
):
    """Build the common torchvision ToTensor plus optional Normalize transform."""
    transform_steps = [transforms.ToTensor()]
    if normalize:
        transform_steps.append(transforms.Normalize(mean, std))
    return transforms.Compose(transform_steps)


def _split_train_valid_dataset(
    train_dataset: Dataset,
    *,
    train_valid_split: float,
    split_seed: int,
) -> tuple[Dataset, Dataset]:
    train_size = int(len(train_dataset) * train_valid_split)
    valid_size = len(train_dataset) - train_size
    return torch.utils.data.random_split(
        train_dataset,
        [train_size, valid_size],
        generator=_split_generator(split_seed),
    )


def _materialize_mnist_style_tensor_triplet(
    train_dataset: Dataset,
    test_dataset: Dataset,
    *,
    train_valid_split: float,
    flatten: bool,
    split_seed: int,
) -> tuple[TensorDataset, TensorDataset, TensorDataset]:
    """Split and materialize MNIST-style torchvision datasets as TensorDatasets."""
    train_dataset, valid_dataset = _split_train_valid_dataset(
        train_dataset,
        train_valid_split=train_valid_split,
        split_seed=split_seed,
    )

    x_train, y_train = _dataset_to_tensors(train_dataset)

    if len(valid_dataset) > 0:
        x_valid, y_valid = _dataset_to_tensors(valid_dataset)
    else:
        sample_img, _ = train_dataset[0]
        x_valid = torch.empty(0, *sample_img.shape)
        y_valid = torch.empty(0, dtype=torch.long)

    x_test, y_test = _dataset_to_tensors(test_dataset)

    if flatten:
        x_train = x_train.view(x_train.shape[0], -1)
        if len(x_valid) > 0:
            x_valid = x_valid.view(x_valid.shape[0], -1)
        else:
            x_valid = torch.empty(0, x_train.shape[1])
        x_test = x_test.view(x_test.shape[0], -1)

    return (
        torch.utils.data.TensorDataset(x_train, y_train),
        torch.utils.data.TensorDataset(x_valid, y_valid),
        torch.utils.data.TensorDataset(x_test, y_test),
    )


def _load_mnist_style_as_datasets(
    *,
    dataset_key: str,
    dataset_cls: Callable[..., Dataset],
    dataset_name: str,
    mean: tuple[float, ...],
    std: tuple[float, ...],
    train_valid_split: float,
    flatten: bool,
    normalize: bool,
    task_data_path: str | None,
    split_seed: int,
) -> tuple[TensorDataset, TensorDataset, TensorDataset]:
    data_dir = get_data_directory(dataset_key, task_data_path)
    transform = _build_to_tensor_transform(
        normalize=normalize,
        mean=mean,
        std=std,
    )

    train_dataset, test_dataset = _load_torchvision_train_test_pair(
        dataset_cls,
        root=data_dir,
        transform=transform,
        dataset_name=dataset_name,
    )

    return _materialize_mnist_style_tensor_triplet(
        train_dataset,
        test_dataset,
        train_valid_split=train_valid_split,
        flatten=flatten,
        split_seed=split_seed,
    )


def load_mnist_as_datasets(
    train_valid_split=0.8,
    flatten=False,
    normalize=False,
    task_data_path=None,
    split_seed: int = 0,
):
    """
    Load the MNIST dataset and split into train, validation, and test sets.

    Args:
        train_valid_split (float): Proportion of training data to use for training (vs validation)
        flatten (bool): Whether to flatten the images
        normalize (bool): Whether to normalize the images
        task_data_path (str, optional): Path from task.data_path config

    Returns:
        tuple: (train_dataset, valid_dataset, test_dataset)
    """
    return _load_mnist_style_as_datasets(
        dataset_key="mnist",
        dataset_cls=datasets.MNIST,
        dataset_name="MNIST",
        mean=(0.1307,),
        std=(0.3081,),
        train_valid_split=train_valid_split,
        flatten=flatten,
        normalize=normalize,
        task_data_path=task_data_path,
        split_seed=split_seed,
    )


def load_fashion_mnist_as_datasets(
    train_valid_split=0.8,
    flatten=False,
    normalize=False,
    task_data_path=None,
    split_seed: int = 0,
):
    """
    Load the Fashion-MNIST dataset and split into train, validation, and test sets.

    Args:
        train_valid_split (float): Proportion of training data to use for training (vs validation)
        flatten (bool): Whether to flatten the images
        normalize (bool): Whether to normalize the images
        task_data_path (str, optional): Path from task.data_path config

    Returns:
        tuple: (train_dataset, valid_dataset, test_dataset)
    """
    return _load_mnist_style_as_datasets(
        dataset_key="fashion_mnist",
        dataset_cls=datasets.FashionMNIST,
        dataset_name="FashionMNIST",
        mean=(0.2860,),
        std=(0.3530,),
        train_valid_split=train_valid_split,
        flatten=flatten,
        normalize=normalize,
        task_data_path=task_data_path,
        split_seed=split_seed,
    )


def load_mnist_modulo10(
    shuffle_iterations=2,
    train_valid_split=1,
    flatten=False,
    normalize=False,
    task_data_path=None,
    split_seed: int = 0,
    pair_seed: int = 0,
):
    """
    Load MNIST dataset with modulo 10 operation on labels.

    Args:
        shuffle_iterations (int): Number of shuffle iterations
        train_valid_split (float): Proportion of training data to use for training
        flatten (bool): Whether to flatten the images
        normalize (bool): Whether to normalize the images
        task_data_path (str, optional): Path from task.data_path config

    Returns:
        dict: Dictionary containing train, valid, and test datasets
    """
    train_ds, valid_ds, test_ds = load_mnist_as_datasets(
        train_valid_split=train_valid_split,
        flatten=flatten,
        normalize=normalize,
        task_data_path=task_data_path,
        split_seed=split_seed,
    )
    data = {"train": train_ds, "valid": valid_ds, "test": test_ds}

    effective_pair_seed = int(pair_seed) + 1009 * max(1, int(shuffle_iterations))
    for split_idx, (key, dataset) in enumerate(data.items()):
        data[key] = _build_iterative_modulo10_dataset(
            dataset,
            shuffle_iterations=shuffle_iterations,
            pair_seed=effective_pair_seed + split_idx,
        )

    return data


def _materialize_cifar_tensor_triplet(
    train_dataset: Dataset,
    test_dataset: Dataset,
    *,
    train_valid_split: float,
    flatten: bool,
    split_seed: int,
) -> tuple[TensorDataset, TensorDataset, TensorDataset]:
    """Split and materialize CIFAR-style datasets while preserving clone-valid semantics."""
    if train_valid_split < 1.0:
        train_dataset, valid_dataset = _split_train_valid_dataset(
            train_dataset,
            train_valid_split=train_valid_split,
            split_seed=split_seed,
        )

        x_train, y_train = _dataset_to_tensors(train_dataset, flatten=flatten)
        x_valid, y_valid = _dataset_to_tensors(valid_dataset, flatten=flatten)
        x_test, y_test = _dataset_to_tensors(test_dataset, flatten=flatten)

        return (
            torch.utils.data.TensorDataset(x_train, y_train),
            torch.utils.data.TensorDataset(x_valid, y_valid),
            torch.utils.data.TensorDataset(x_test, y_test),
        )

    x_train, y_train = _dataset_to_tensors(train_dataset, flatten=flatten)
    x_test, y_test = _dataset_to_tensors(test_dataset, flatten=flatten)
    return (
        torch.utils.data.TensorDataset(x_train, y_train),
        torch.utils.data.TensorDataset(x_train.clone(), y_train.clone()),
        torch.utils.data.TensorDataset(x_test, y_test),
    )


def _materialize_cifar10_tensor_triplet(
    train_dataset: Dataset,
    test_dataset: Dataset,
    *,
    train_valid_split: float,
    flatten: bool,
    split_seed: int,
) -> tuple[TensorDataset, TensorDataset, TensorDataset]:
    """Compatibility wrapper for the CIFAR-style materializer."""
    return _materialize_cifar_tensor_triplet(
        train_dataset,
        test_dataset,
        train_valid_split=train_valid_split,
        flatten=flatten,
        split_seed=split_seed,
    )


def load_cifar10_as_datasets(
    train_valid_split=1.0,
    flatten=False,
    normalize=False,
    task_data_path=None,
    split_seed: int = 0,
):
    """
    Load the CIFAR-10 dataset and split into train, validation, and test sets.

    Args:
        train_valid_split (float): Proportion of training data to use for training (vs validation)
        flatten (bool): Whether to flatten the images
        normalize (bool): Whether to normalize the images
        task_data_path (str, optional): Path from task.data_path config

    Returns:
        tuple: (train_dataset, valid_dataset, test_dataset)
    """
    transform = _build_to_tensor_transform(
        normalize=normalize,
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2470, 0.2435, 0.2616),
    )

    cifar_dir = get_data_directory("cifar", task_data_path)
    train_dataset, test_dataset = _load_torchvision_train_test_pair(
        datasets.CIFAR10,
        root=cifar_dir,
        transform=transform,
        dataset_name="CIFAR10",
    )

    return _materialize_cifar_tensor_triplet(
        train_dataset,
        test_dataset,
        train_valid_split=train_valid_split,
        flatten=flatten,
        split_seed=split_seed,
    )


def load_cifar100_as_datasets(
    train_valid_split=1.0,
    flatten=False,
    normalize=False,
    task_data_path=None,
    split_seed: int = 0,
):
    """
    Load the CIFAR-100 dataset and split into train, validation, and test sets.

    Args:
        train_valid_split (float): Proportion of training data to use for training (vs validation)
        flatten (bool): Whether to flatten the images
        normalize (bool): Whether to normalize the images
        task_data_path (str, optional): Path from task.data_path config

    Returns:
        tuple: (train_dataset, valid_dataset, test_dataset)
    """
    transform = _build_to_tensor_transform(
        normalize=normalize,
        mean=(0.5071, 0.4867, 0.4408),
        std=(0.2675, 0.2565, 0.2761),
    )

    cifar_dir = get_data_directory("cifar", task_data_path)
    train_dataset, test_dataset = _load_torchvision_train_test_pair(
        datasets.CIFAR100,
        root=cifar_dir,
        transform=transform,
        dataset_name="CIFAR100",
    )

    return _materialize_cifar_tensor_triplet(
        train_dataset,
        test_dataset,
        train_valid_split=train_valid_split,
        flatten=flatten,
        split_seed=split_seed,
    )


def _build_paired_arithmetic_triplet(
    train_ds: TensorDataset,
    valid_ds: TensorDataset,
    test_ds: TensorDataset,
    *,
    shuffle_iterations: int,
    task_mode: str,
    flatten: bool,
    include_context: bool,
    pair_seed: int,
    context_seed: int = 0,
) -> dict[str, Dataset]:
    """Build paired arithmetic train/validation/test datasets with split seeds."""
    datasets_by_split = {
        "train": train_ds,
        "valid": valid_ds,
        "test": test_ds,
    }
    return {
        split_name: _build_paired_arithmetic_dataset(
            dataset,
            shuffle_iterations=shuffle_iterations,
            task_mode=task_mode,
            flatten=flatten,
            include_context=include_context,
            pair_seed=pair_seed + split_idx,
            context_seed=context_seed + split_idx,
        )
        for split_idx, (split_name, dataset) in enumerate(datasets_by_split.items())
    }


def load_cifar10_modulo10(
    shuffle_iterations=1,
    train_valid_split=1.0,
    flatten=True,
    normalize=False,
    task_data_path=None,
    pair_seed: int = 0,
    split_seed: int = 0,
):
    """Load paired CIFAR-10 where the label is the sum of two digits mod 10."""
    train_ds, valid_ds, test_ds = load_cifar10_as_datasets(
        train_valid_split=train_valid_split,
        flatten=False,
        normalize=normalize,
        task_data_path=task_data_path,
        split_seed=split_seed,
    )

    return _build_paired_arithmetic_triplet(
        train_ds,
        valid_ds,
        test_ds,
        shuffle_iterations=shuffle_iterations,
        task_mode="sum_mod10",
        flatten=flatten,
        include_context=False,
        pair_seed=pair_seed,
    )


def load_double_cifar10_contextual_mod10(
    shuffle_iterations=1,
    train_valid_split=1.0,
    flatten=True,
    normalize=False,
    task_data_path=None,
    pair_seed: int = 0,
    context_seed: int = 123,
    split_seed: int = 0,
):
    """Load paired CIFAR-10 with a context-controlled sum-or-difference label."""
    train_ds, valid_ds, test_ds = load_cifar10_as_datasets(
        train_valid_split=train_valid_split,
        flatten=False,
        normalize=normalize,
        task_data_path=task_data_path,
        split_seed=split_seed,
    )

    return _build_paired_arithmetic_triplet(
        train_ds,
        valid_ds,
        test_ds,
        shuffle_iterations=shuffle_iterations,
        task_mode="context_sum_diff",
        flatten=flatten,
        include_context=True,
        pair_seed=pair_seed,
        context_seed=context_seed,
    )


def load_double_mnist_contextual_mod10(
    shuffle_iterations=1,
    train_valid_split=1.0,
    flatten=True,
    normalize=False,
    task_data_path=None,
    pair_seed: int = 0,
    context_seed: int = 123,
    split_seed: int = 0,
):
    """Load paired MNIST with a context-controlled sum-or-difference label."""
    train_ds, valid_ds, test_ds = load_mnist_as_datasets(
        train_valid_split=train_valid_split,
        flatten=False,
        normalize=normalize,
        task_data_path=task_data_path,
        split_seed=split_seed,
    )

    return _build_paired_arithmetic_triplet(
        train_ds,
        valid_ds,
        test_ds,
        shuffle_iterations=shuffle_iterations,
        task_mode="context_sum_diff",
        flatten=flatten,
        include_context=True,
        pair_seed=pair_seed,
        context_seed=context_seed,
    )


def _wrap_poisson_generator_dataset(
    dataset: Dataset,
    *,
    multiplicative_gain: bool,
    fixed_gain_factor: Optional[float],
    max_gain_factor: Optional[float],
    uniform_gain: bool,
    gain_sampling: str,
    poisson_sampling: bool,
    stimulus_duration: float,
    max_gain_tau_ratio: Optional[float],
) -> PoissonGeneratorDataset:
    """Wrap one tensor-like dataset with the Poisson generator."""
    return PoissonGeneratorDataset(
        *dataset[:],
        multiplicative_gain=multiplicative_gain,
        fixed_gain_factor=fixed_gain_factor,
        max_gain_factor=max_gain_factor,
        uniform_gain=uniform_gain,
        gain_sampling=gain_sampling,
        poisson_sampling=poisson_sampling,
        stimulus_duration=stimulus_duration,
        max_gain_tau_ratio=max_gain_tau_ratio,
    )


def _wrap_poisson_generator_triplet(
    train_ds: Dataset,
    valid_ds: Dataset,
    test_ds: Dataset,
    *,
    multiplicative_gain: bool,
    fixed_gain_factor: Optional[float],
    max_gain_factor: Optional[float],
    uniform_gain: bool,
    poisson_sampling: bool,
    stimulus_duration: float,
    max_gain_tau_ratio: Optional[float],
    gain_sampling: str = "log_uniform",
) -> tuple[PoissonGeneratorDataset, PoissonGeneratorDataset, PoissonGeneratorDataset]:
    """Wrap train/validation/test datasets with identical Poisson settings."""
    kwargs = {
        "multiplicative_gain": multiplicative_gain,
        "fixed_gain_factor": fixed_gain_factor,
        "max_gain_factor": max_gain_factor,
        "uniform_gain": uniform_gain,
        "gain_sampling": gain_sampling,
        "poisson_sampling": poisson_sampling,
        "stimulus_duration": stimulus_duration,
        "max_gain_tau_ratio": max_gain_tau_ratio,
    }
    return (
        _wrap_poisson_generator_dataset(train_ds, **kwargs),
        _wrap_poisson_generator_dataset(valid_ds, **kwargs),
        _wrap_poisson_generator_dataset(test_ds, **kwargs),
    )


def load_poisson_generator_datasets(
    base_dataset: str,
    multiplicative_gain: bool = True,
    fixed_gain_factor: Optional[float] = None,
    max_gain_factor: Optional[float] = None,
    uniform_gain: bool = True,
    gain_sampling: str = "log_uniform",
    poisson_sampling: bool = True,
    stimulus_duration: float = 1.0,
    max_gain_tau_ratio: Optional[float] = None,
    train_valid_split: float = 0.8,
    flatten: bool = False,
    task_data_path: Optional[str] = None,
    split_seed: int = 0,
):
    """
    Generate a dataset of Poisson-distributed samples from a base dataset.

    Args:
        base_dataset (str): The base dataset to sample from
        stimulus_duration (float): The stimulus duration for the Poisson distribution
        fixed_gain_factor (float): The fixed gain factor for the Poisson distribution
        max_gain_factor (float): The maximum gain factor for the Poisson distribution
        max_gain_tau_ratio (float): The maximum gain tau ratio for the Poisson distribution
        uniform_gain (bool): Whether one gain is shared across each sample
        gain_sampling (str): "log_uniform" (default) or "linear_uniform"
        train_valid_split (float): The proportion of samples to use for training vs validation
        flatten (bool): Whether to flatten the samples
        task_data_path (str, optional): Path from task.data_path config

    Returns:
        tuple: (train_dataset, valid_dataset, test_dataset)
    """
    # Load the base dataset
    if base_dataset == "mnist":
        train_ds, valid_ds, test_ds = load_mnist_as_datasets(
            train_valid_split=train_valid_split,
            flatten=flatten,
            normalize=False,
            task_data_path=task_data_path,
            split_seed=split_seed,
        )
    elif base_dataset == "cifar10":
        train_ds, valid_ds, test_ds = load_cifar10_as_datasets(
            train_valid_split=train_valid_split,
            flatten=flatten,
            normalize=False,
            task_data_path=task_data_path,
            split_seed=split_seed,
        )
    else:
        raise ValueError(f"Unknown base dataset: {base_dataset}")

    return _wrap_poisson_generator_triplet(
        train_ds,
        valid_ds,
        test_ds,
        multiplicative_gain=multiplicative_gain,
        fixed_gain_factor=fixed_gain_factor,
        max_gain_factor=max_gain_factor,
        uniform_gain=uniform_gain,
        gain_sampling=gain_sampling,
        poisson_sampling=poisson_sampling,
        stimulus_duration=stimulus_duration,
        max_gain_tau_ratio=max_gain_tau_ratio,
    )


_STANDARD_VISION_DATASET_LOADERS: dict[str, str | Callable[..., _DatasetTriplet]] = {
    "mnist": "load_mnist_as_datasets",
    "fashion_mnist": "load_fashion_mnist_as_datasets",
    "cifar10": "load_cifar10_as_datasets",
    "cifar100": "load_cifar100_as_datasets",
}


def _resolve_standard_vision_dataset_loader(
    dataset_name: str,
) -> Callable[..., _DatasetTriplet] | None:
    """Resolve a standard vision loader without freezing monkeypatch targets."""
    loader = _STANDARD_VISION_DATASET_LOADERS.get(dataset_name)
    if isinstance(loader, str):
        return globals()[loader]
    return loader


def _load_standard_vision_dataset_triplet(
    *,
    dataset_name: str,
    train_valid_split: float,
    flatten: bool,
    normalize: bool,
    task_data_path: str | None,
    split_seed: int,
    label_noise_rate: float,
    label_noise_seed: int,
) -> tuple[Dataset, Dataset, Dataset] | None:
    """Load MNIST/Fashion-MNIST/CIFAR-10 with shared post-processing."""
    loader = _resolve_standard_vision_dataset_loader(dataset_name)
    if loader is None:
        return None

    train_ds, valid_ds, test_ds = loader(
        train_valid_split=train_valid_split,
        flatten=flatten,
        normalize=normalize,
        task_data_path=task_data_path,
        split_seed=split_seed,
    )
    return _maybe_apply_label_noise_to_triplet(
        train_ds,
        valid_ds,
        test_ds,
        label_noise_rate=label_noise_rate,
        label_noise_seed=label_noise_seed,
    )


def _load_sequence_dataset_triplet(
    *,
    dataset_name: str,
    data_path: str | None,
    train_valid_split: float,
    parameters: dict[str, Any],
) -> tuple[Dataset, Dataset, Dataset]:
    """Load a sequence dataset through the sequence dataset factory."""
    from dendritic_modeling.datasets.sequence_datasets import get_sequence_datasets

    return get_sequence_datasets(
        dataset_name=dataset_name,
        data_path=data_path,
        train_valid_split=train_valid_split,
        **parameters,
    )


def _has_sequence_dataset_builder(dataset_name: str) -> bool:
    """Return whether the live sequence builder registry can load this dataset."""
    from dendritic_modeling.datasets.sequence_datasets import (
        has_sequence_dataset_builder,
    )

    return has_sequence_dataset_builder(dataset_name)


def _load_theoretical_synthetic_dataset_triplet(
    *,
    dataset_name: str,
    parameters: dict[str, Any],
) -> tuple[Dataset, Dataset, Dataset]:
    """Load built-in theoretical synthetic datasets by injecting the type key."""
    dataset_config = parameters.copy()
    dataset_config["type"] = dataset_name
    return get_synthetic_datasets(dataset_config)


def _experimental_dataset_import_path_candidates() -> list[str]:
    """Return historical experiments-package import roots in legacy order."""
    repo_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )
    experiments_root = os.path.join(repo_root, "experiments")
    return [
        experiments_root,
        os.path.join(experiments_root, "experiments"),
    ]


def _ensure_experimental_dataset_import_paths(path_candidates: list[str]) -> None:
    """Add existing experiments-package roots to ``sys.path`` with legacy insertion."""
    import sys

    for candidate_path in path_candidates:
        if os.path.isdir(candidate_path) and candidate_path not in sys.path:
            sys.path.insert(0, candidate_path)


def _resolve_experimental_dataset_loader() -> tuple[Callable | None, Exception | None]:
    """Resolve the optional experiments dataset loader plus the final import error."""
    last_error: Exception | None = None
    get_experimental_datasets = None
    for module_name in _EXPERIMENTAL_DATASET_IMPORT_CANDIDATES:
        try:
            module = importlib.import_module(module_name)
            get_experimental_datasets = module.get_unified_datasets
            break
        except Exception as e:  # pragma: no cover - fallback path
            last_error = e
    return get_experimental_datasets, last_error


def _load_experimental_dataset_triplet(
    *,
    dataset_name: str,
    task_cfg: Any,
) -> tuple[Dataset, Dataset, Dataset]:
    """Load a dataset from the optional experiments package fallback."""
    # For synthetic/experimental datasets, delegate to experiments synthetic datasets.
    # Keep this robust to both layouts:
    #   <repo>/experiments/experiments/data_generation/synthetic_datasets.py
    #   <repo>/experiments/data_generation/synthetic_datasets.py
    # Ensure both candidate roots are importable regardless launch cwd/PYTHONPATH.
    _ensure_experimental_dataset_import_paths(
        _experimental_dataset_import_path_candidates()
    )

    get_experimental_datasets, last_error = _resolve_experimental_dataset_loader()

    if get_experimental_datasets is None:
        raise ValueError(
            f"Unknown dataset name: {dataset_name}. Synthetic datasets not available: {last_error}"
        )

    train_ds, valid_ds, test_ds = get_experimental_datasets(task_cfg, task_cfg)
    return _ensure_slice_safe_triplet(train_ds, valid_ds, test_ds)


def _load_poisson_generator_from_options(
    options: _UnifiedDatasetOptions,
) -> _DatasetTriplet:
    return load_poisson_generator_datasets(
        base_dataset=options.base_dataset,
        multiplicative_gain=options.multiplicative_gain,
        fixed_gain_factor=options.fixed_gain_factor,
        max_gain_factor=options.max_gain_factor,
        uniform_gain=options.uniform_gain,
        gain_sampling=options.gain_sampling,
        poisson_sampling=options.poisson_sampling,
        stimulus_duration=options.stimulus_duration,
        max_gain_tau_ratio=options.max_gain_tau_ratio,
        train_valid_split=options.train_valid_split,
        flatten=options.flatten,
        task_data_path=options.task_data_path,
        split_seed=options.split_seed,
    )


def _load_imagenet_from_options(options: _UnifiedDatasetOptions) -> _DatasetTriplet:
    p = options.parameters
    return load_imagenet_as_datasets(
        train_valid_split=options.train_valid_split,
        flatten=options.flatten,
        normalize=options.normalize,
        task_data_path=options.task_data_path,
        train_samples_per_class=p.get("train_samples_per_class"),
        val_samples_per_class=p.get("val_samples_per_class"),
        split_seed=options.split_seed,
        transform_preset=p.get("transform_preset", "legacy_resize"),
        allow_flatten_materialize=bool(p.get("allow_flatten_materialize", False)),
        loader_backend=p.get("loader_backend", "auto"),
    )


def _standard_dataset_option_kwargs(options: _UnifiedDatasetOptions) -> dict[str, Any]:
    """Return common standard-dataset loader kwargs from unified options."""
    return {
        "train_valid_split": options.train_valid_split,
        "flatten": options.flatten,
        "normalize": options.normalize,
        "task_data_path": options.task_data_path,
        "split_seed": options.split_seed,
    }


def _paired_arithmetic_option_kwargs(
    options: _UnifiedDatasetOptions,
    *,
    default_shuffle_iterations: int,
    include_context: bool,
) -> dict[str, Any]:
    """Return paired arithmetic loader kwargs from unified options."""
    p = options.parameters
    kwargs = {
        **_standard_dataset_option_kwargs(options),
        "shuffle_iterations": p.get("shuffle_iterations", default_shuffle_iterations),
        "pair_seed": p.get("pair_seed", 0),
    }
    if include_context:
        kwargs["context_seed"] = p.get("context_seed", 123)
    return kwargs


def _load_paired_arithmetic_from_options(
    options: _UnifiedDatasetOptions,
    loader: Callable[..., dict[str, Dataset]],
    *,
    default_shuffle_iterations: int,
    include_context: bool,
    valid_fallback_to_test: bool = False,
    slice_safe: bool = False,
) -> _DatasetTriplet:
    data = loader(
        **_paired_arithmetic_option_kwargs(
            options,
            default_shuffle_iterations=default_shuffle_iterations,
            include_context=include_context,
        )
    )
    return _dataset_mapping_to_triplet(
        data,
        valid_fallback_to_test=valid_fallback_to_test,
        slice_safe=slice_safe,
    )


def _load_mnist_modulo10_from_options(
    options: _UnifiedDatasetOptions,
) -> _DatasetTriplet:
    return _load_paired_arithmetic_from_options(
        options,
        load_mnist_modulo10,
        default_shuffle_iterations=2,
        include_context=False,
        valid_fallback_to_test=True,
    )


def _load_cifar10_modulo10_from_options(
    options: _UnifiedDatasetOptions,
) -> _DatasetTriplet:
    return _load_paired_arithmetic_from_options(
        options,
        load_cifar10_modulo10,
        default_shuffle_iterations=1,
        include_context=False,
        slice_safe=True,
    )


def _load_double_cifar10_contextual_mod10_from_options(
    options: _UnifiedDatasetOptions,
) -> _DatasetTriplet:
    return _load_paired_arithmetic_from_options(
        options,
        load_double_cifar10_contextual_mod10,
        default_shuffle_iterations=1,
        include_context=True,
        slice_safe=True,
    )


def _load_double_mnist_contextual_mod10_from_options(
    options: _UnifiedDatasetOptions,
) -> _DatasetTriplet:
    return _load_paired_arithmetic_from_options(
        options,
        load_double_mnist_contextual_mod10,
        default_shuffle_iterations=1,
        include_context=True,
        slice_safe=True,
    )


_SPECIAL_DATASET_LOADERS: dict[
    str, Callable[[_UnifiedDatasetOptions], _DatasetTriplet]
] = {
    "poisson_generator": _load_poisson_generator_from_options,
    "imagenet": _load_imagenet_from_options,
    "mnist_modulo10": _load_mnist_modulo10_from_options,
    "cifar10_modulo10": _load_cifar10_modulo10_from_options,
    "double_cifar10_contextual_mod10": (
        _load_double_cifar10_contextual_mod10_from_options
    ),
    "double_mnist_contextual_mod10": _load_double_mnist_contextual_mod10_from_options,
}


def _load_special_dataset_triplet(
    options: _UnifiedDatasetOptions,
) -> _DatasetTriplet | None:
    dataset_name = options.dataset_name
    p = options.parameters

    loader = _SPECIAL_DATASET_LOADERS.get(dataset_name)
    if loader is not None:
        return loader(options)

    if dataset_name in _THEORETICAL_SYNTHETIC_DATASET_NAMES:
        return _load_theoretical_synthetic_dataset_triplet(
            dataset_name=dataset_name,
            parameters=p,
        )

    if _has_sequence_dataset_builder(dataset_name):
        return _load_sequence_dataset_triplet(
            dataset_name=dataset_name,
            data_path=options.task_data_path,
            train_valid_split=options.train_valid_split,
            parameters=p,
        )

    return None


def get_unified_datasets(task_cfg):
    """
    Get datasets based on configuration.

    Args:
        task_cfg: Task configuration (must include train_valid_split)

    Returns:
        tuple: (train_dataset, valid_dataset, test_dataset)
    """
    options = _resolve_unified_dataset_options(task_cfg)
    dataset_name = options.dataset_name
    task_data_path = options.task_data_path

    registered = build_registered_dataset(dataset_name, task_cfg)
    if registered is not None:
        return registered

    # Handle standard datasets directly
    standard_vision_datasets = _load_standard_vision_dataset_triplet(
        dataset_name=dataset_name,
        train_valid_split=options.train_valid_split,
        flatten=options.flatten,
        normalize=options.normalize,
        task_data_path=task_data_path,
        split_seed=options.split_seed,
        label_noise_rate=options.label_noise_rate,
        label_noise_seed=options.label_noise_seed,
    )
    if standard_vision_datasets is not None:
        return standard_vision_datasets

    special_datasets = _load_special_dataset_triplet(options)
    if special_datasets is not None:
        return special_datasets

    return _load_experimental_dataset_triplet(
        dataset_name=dataset_name,
        task_cfg=task_cfg,
    )
