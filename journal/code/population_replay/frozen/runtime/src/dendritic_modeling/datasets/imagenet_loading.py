"""ImageNet dataset construction helpers."""

import hashlib
import logging
import os
import pickle
import tempfile
from copy import copy

from torch.utils.data import Dataset
from torchvision import datasets

from .imagenet_paths import _normalized_imagenet_loader_backend

_logger = logging.getLogger(__name__)

_INDEX_CACHE_ENV = "DENDRITIC_IMAGEFOLDER_CACHE"


class _IndexCachedImageFolder(datasets.ImageFolder):
    """ImageFolder that reads a prebuilt (classes, samples) index.

    Scanning the ~1.28M-file ImageNet train tree over NFS at every job start
    is slow and has stalled outright on cold nodes; the class/sample index is
    static, so it is scanned once and reused.
    """

    def __init__(self, root: str, transform, cached_index: dict):
        self._cached_index = cached_index
        super().__init__(root=root, transform=transform)

    def find_classes(self, directory):
        return self._cached_index["classes"], self._cached_index["class_to_idx"]

    def make_dataset(
        self, directory, class_to_idx, extensions=None, is_valid_file=None, **kwargs
    ):
        return [
            (os.path.join(self.root, relpath), idx)
            for relpath, idx in self._cached_index["samples"]
        ]


def _index_cache_path(root: str) -> str | None:
    cache_dir = os.environ.get(_INDEX_CACHE_ENV)
    if not cache_dir:
        return None
    key = hashlib.sha256(os.path.realpath(root).encode()).hexdigest()[:16]
    return os.path.join(cache_dir, f"imagefolder_index_{key}.pkl")


def _imagefolder_with_index_cache(root: str, transform) -> Dataset:
    """Build an ImageFolder, bypassing the directory scan via a cached index.

    Opt-in through the ``DENDRITIC_IMAGEFOLDER_CACHE`` environment variable
    (a directory); unset means stock behavior. A missing cache entry is built
    by one full scan and written atomically for subsequent jobs.
    """
    cache_path = _index_cache_path(root)
    if cache_path is None:
        return datasets.ImageFolder(root=root, transform=transform)
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as handle:
            cached_index = pickle.load(handle)
        _logger.info(
            "ImageFolder index cache hit: %s (%d samples)",
            cache_path,
            len(cached_index["samples"]),
        )
        return _IndexCachedImageFolder(
            root=root, transform=transform, cached_index=cached_index
        )

    dataset = datasets.ImageFolder(root=root, transform=transform)
    index = {
        "classes": dataset.classes,
        "class_to_idx": dataset.class_to_idx,
        "samples": [
            (os.path.relpath(path, root), idx) for path, idx in dataset.samples
        ],
    }
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=os.path.dirname(cache_path), suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as handle:
            pickle.dump(index, handle)
        os.replace(tmp_path, cache_path)
    except BaseException:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise
    _logger.info(
        "ImageFolder index cache written: %s (%d samples)",
        cache_path,
        len(index["samples"]),
    )
    return dataset


def _copy_dataset_with_transform(dataset: Dataset, transform) -> Dataset:
    """Return a shallow dataset view with the same samples and a new transform."""
    dataset_view = copy(dataset)
    dataset_view.transform = transform

    transforms_obj = getattr(dataset_view, "transforms", None)
    if transforms_obj is not None and hasattr(transforms_obj, "transform"):
        dataset_view.transforms = copy(transforms_obj)
        dataset_view.transforms.transform = transform

    return dataset_view


def _load_imagefolder_dataset_triplet(
    *,
    train_root: str,
    val_root: str,
    train_transform,
    eval_transform,
) -> tuple[Dataset, Dataset, Dataset]:
    if not train_root or not val_root:
        raise RuntimeError(
            "ImageFolder ImageNet loading requires both train and val "
            "class-directory roots."
        )

    _logger.info(
        "Loading ImageNet-style folder tree via ImageFolder: train=%s val=%s",
        train_root,
        val_root,
    )
    train_dataset = _imagefolder_with_index_cache(train_root, train_transform)
    return (
        train_dataset,
        _copy_dataset_with_transform(train_dataset, eval_transform),
        _imagefolder_with_index_cache(val_root, eval_transform),
    )


def _load_torchvision_imagenet_dataset_triplet(
    *,
    imagenet_dir: str,
    train_transform,
    eval_transform,
) -> tuple[Dataset, Dataset, Dataset]:
    train_dataset = datasets.ImageNet(
        root=imagenet_dir, split="train", transform=train_transform
    )
    return (
        train_dataset,
        _copy_dataset_with_transform(train_dataset, eval_transform),
        datasets.ImageNet(root=imagenet_dir, split="val", transform=eval_transform),
    )


def _validate_imagefolder_backend_roots(
    backend: str,
    train_root: str | None,
    val_root: str | None,
) -> None:
    if backend == "imagefolder" and (not train_root or not val_root):
        raise RuntimeError(
            "ImageNet loader_backend='imagefolder' requires train/val "
            "class-directory roots."
        )


def _load_imagenet_dataset_triplet(
    *,
    imagenet_dir: str,
    train_root: str | None,
    val_root: str | None,
    train_transform,
    eval_transform,
    loader_backend: str | None = "auto",
) -> tuple[Dataset, Dataset, Dataset]:
    """Load ImageNet train/eval/test datasets from ImageFolder or ImageNet roots."""
    backend = _normalized_imagenet_loader_backend(loader_backend)
    _validate_imagefolder_backend_roots(backend, train_root, val_root)
    if train_root is not None:
        return _load_imagefolder_dataset_triplet(
            train_root=train_root,
            val_root=val_root,
            train_transform=train_transform,
            eval_transform=eval_transform,
        )
    return _load_torchvision_imagenet_dataset_triplet(
        imagenet_dir=imagenet_dir,
        train_transform=train_transform,
        eval_transform=eval_transform,
    )


__all__ = [
    "_copy_dataset_with_transform",
    "_load_imagenet_dataset_triplet",
]
