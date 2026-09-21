"""ImageNet dataset loading and preprocessing helpers."""

from torchvision import datasets as datasets, transforms as transforms

from .imagenet_loading import (
    _copy_dataset_with_transform,
    _load_imagenet_dataset_triplet,
)
from .imagenet_paths import (
    _apply_imagenet_local_override,
    _is_official_imagenet_root,
    _normalized_imagenet_loader_backend,
    _resolve_imagefolder_roots,
    _resolve_imagenet_data_location,
)
from .imagenet_splits import (
    IMAGENET_FLATTEN_MATERIALIZATION_LIMIT,
    _apply_imagenet_class_subsets,
    _class_subset_indices,
    _materialize_flattened_tensor_dataset,
    _maybe_materialize_flattened_imagenet_triplet,
    _prepare_imagenet_dataset_triplet,
    _split_imagenet_train_valid_or_use_test,
    _split_train_valid_pair,
    _subset_by_indices,
)
from .imagenet_transforms import (
    _append_imagenet_normalize,
    _build_imagenet_transforms,
    _normalized_imagenet_preset_name,
)


def _format_imagenet_load_error(
    *,
    task_data_path: str | None,
    imagenet_dir: str,
    train_root: str | None,
    val_root: str | None,
    loader_backend: str,
) -> str:
    """Return a loader error that includes the resolved ImageNet location."""
    return (
        "Could not load ImageNet. Set data.base_dir or IMAGENET_LOCAL_DIR "
        "to a valid ImageNet root. "
        f"Resolved loader_backend={loader_backend!r}, "
        f"task_data_path={task_data_path!r}, "
        f"imagenet_dir={imagenet_dir!r}, "
        f"train_root={train_root!r}, val_root={val_root!r}. "
        "No synthetic or FakeData fallback is used."
    )


def load_imagenet_as_datasets(
    train_valid_split=0.8,
    flatten=False,
    normalize=False,
    task_data_path=None,
    train_samples_per_class=None,
    val_samples_per_class=None,
    split_seed: int = 0,
    transform_preset: str = "legacy_resize",
    allow_flatten_materialize: bool = False,
    loader_backend: str = "auto",
):
    """
    Load the ImageNet dataset and split into train, validation, and test sets.

    If the environment variable ``IMAGENET_LOCAL_DIR`` is set and points to a
    valid directory, it overrides *task_data_path*.  This allows SLURM scripts
    to copy ImageNet to node-local NVMe (``$SLURM_TMPDIR``) for faster I/O
    without editing the YAML config.

    Args:
        train_valid_split (float): Proportion of training data to use for training (vs validation)
        flatten (bool): Whether to flatten the images
        normalize (bool): Whether to normalize the images
        task_data_path (str, optional): Path from task.data_path config
        train_samples_per_class (int, optional): Limit training samples per class.
        val_samples_per_class (int, optional): Limit validation/test samples per class.
        transform_preset (str): "legacy_resize" preserves the historical
            Resize(224,224) transform; "pretrained" uses train/eval transforms
            appropriate for pretrained torchvision backbones.
        allow_flatten_materialize (bool): Explicit opt-in for the memory-heavy
            ImageNet flatten path.
        loader_backend (str): "auto" preserves the historical official-root
            preference; "imagefolder" forces train/val class-directory loading;
            "torchvision" forces torchvision.datasets.ImageNet.

    Returns:
        tuple: (train_dataset, valid_dataset, test_dataset)
    """
    if flatten and not allow_flatten_materialize:
        raise ValueError(
            "ImageNet with flatten=True materializes up to 50k 224x224x3 "
            "samples and can consume tens of GB. Use flatten=False for "
            "AlexNet/pretrained runs, or set "
            "data.dataset_params.imagenet.allow_flatten_materialize=true "
            "for an explicit debugging-only opt-in."
        )

    loader_backend = _normalized_imagenet_loader_backend(loader_backend)
    imagenet_dir, train_root, val_root = _resolve_imagenet_data_location(
        task_data_path,
        loader_backend=loader_backend,
    )

    train_transform, eval_transform = _build_imagenet_transforms(
        normalize=normalize,
        transform_preset=transform_preset,
    )

    try:
        train_dataset, train_eval_dataset, test_dataset = (
            _load_imagenet_dataset_triplet(
                imagenet_dir=imagenet_dir,
                train_root=train_root,
                val_root=val_root,
                train_transform=train_transform,
                eval_transform=eval_transform,
                loader_backend=loader_backend,
            )
        )
    except Exception as e:
        raise RuntimeError(
            _format_imagenet_load_error(
                task_data_path=task_data_path,
                imagenet_dir=imagenet_dir,
                train_root=train_root,
                val_root=val_root,
                loader_backend=loader_backend,
            )
        ) from e

    return _prepare_imagenet_dataset_triplet(
        train_dataset=train_dataset,
        train_eval_dataset=train_eval_dataset,
        test_dataset=test_dataset,
        train_valid_split=train_valid_split,
        train_samples_per_class=train_samples_per_class,
        val_samples_per_class=val_samples_per_class,
        split_seed=split_seed,
        flatten=flatten,
    )


__all__ = [
    "IMAGENET_FLATTEN_MATERIALIZATION_LIMIT",
    "_append_imagenet_normalize",
    "_apply_imagenet_class_subsets",
    "_apply_imagenet_local_override",
    "_build_imagenet_transforms",
    "_class_subset_indices",
    "_copy_dataset_with_transform",
    "_format_imagenet_load_error",
    "_is_official_imagenet_root",
    "_load_imagenet_dataset_triplet",
    "_materialize_flattened_tensor_dataset",
    "_maybe_materialize_flattened_imagenet_triplet",
    "_normalized_imagenet_loader_backend",
    "_normalized_imagenet_preset_name",
    "_prepare_imagenet_dataset_triplet",
    "_resolve_imagefolder_roots",
    "_resolve_imagenet_data_location",
    "_split_imagenet_train_valid_or_use_test",
    "_split_train_valid_pair",
    "_subset_by_indices",
    "load_imagenet_as_datasets",
]
