"""Path and backend resolution helpers for ImageNet datasets."""

import logging
import os

from .paths import get_data_directory

_logger = logging.getLogger(__name__)


def _imagefolder_root_candidates(root: str) -> list[tuple[str, str]]:
    return [
        (os.path.join(root, "train"), os.path.join(root, "val")),
        (os.path.join(root, "ILSVRC2012_img_train"), os.path.join(root, "val")),
    ]


def _resolve_imagefolder_roots(root: str | None) -> tuple[str | None, str | None]:
    """Return ``(train_root, val_root)`` for ImageFolder-style ImageNet layouts."""
    if not root or not os.path.isdir(root):
        return None, None

    for train_root, val_root in _imagefolder_root_candidates(root):
        if os.path.isdir(train_root) and os.path.isdir(val_root):
            return train_root, val_root
    return None, None


def _normalized_imagenet_loader_backend(loader_backend: str | None) -> str:
    """Normalize ImageNet loader backend aliases."""
    backend = str(loader_backend or "auto").strip().lower().replace("-", "_")
    aliases = {
        "auto": "auto",
        "default": "auto",
        "torchvision": "torchvision",
        "imagenet": "torchvision",
        "official": "torchvision",
        "image_folder": "imagefolder",
        "imagefolder": "imagefolder",
        "folder": "imagefolder",
    }
    if backend not in aliases:
        raise ValueError(
            "Unknown ImageNet loader_backend "
            f"{loader_backend!r}; expected 'auto', 'torchvision', or 'imagefolder'."
        )
    return aliases[backend]


def _is_official_imagenet_root(root: str | None) -> bool:
    """Return True when a root looks like a torchvision ImageNet root."""
    if not root or not os.path.isdir(root):
        return False

    required_any = (
        os.path.exists(os.path.join(root, "meta.bin"))
        or os.path.exists(os.path.join(root, "ILSVRC2012_devkit_t12.tar.gz"))
        or os.path.isdir(os.path.join(root, "ILSVRC2012_devkit_t12"))
    )
    has_train = os.path.isdir(
        os.path.join(root, "ILSVRC2012_img_train")
    ) or os.path.isdir(os.path.join(root, "train"))
    has_val = os.path.isdir(os.path.join(root, "val")) or os.path.isdir(
        os.path.join(root, "ILSVRC2012_img_val")
    )
    return required_any and has_train and has_val


def _apply_imagenet_local_override(task_data_path: str | None) -> str | None:
    """Apply ``IMAGENET_LOCAL_DIR`` when it points to an existing directory."""
    local_override = os.environ.get("IMAGENET_LOCAL_DIR")
    if local_override and os.path.isdir(local_override):
        _logger.info(
            "IMAGENET_LOCAL_DIR set — using local data path: %s (was: %s)",
            local_override,
            task_data_path,
        )
        return local_override
    return task_data_path


def _resolve_imagenet_data_location(
    task_data_path: str | None,
    loader_backend: str | None = "auto",
) -> tuple[str, str | None, str | None]:
    """Resolve ImageNet root plus optional ImageFolder train/val roots."""
    backend = _normalized_imagenet_loader_backend(loader_backend)
    task_data_path = _apply_imagenet_local_override(task_data_path)

    train_root, val_root = None, None
    if backend == "torchvision" and task_data_path and task_data_path.strip():
        return str(task_data_path), train_root, val_root

    if backend == "imagefolder":
        train_root, val_root = _resolve_imagefolder_roots(task_data_path)
        if train_root is not None:
            return str(task_data_path), train_root, val_root
    elif _is_official_imagenet_root(task_data_path):
        return str(task_data_path), train_root, val_root

    if backend != "torchvision":
        train_root, val_root = _resolve_imagefolder_roots(task_data_path)
    if train_root is not None:
        return str(task_data_path), train_root, val_root

    imagenet_dir = get_data_directory("imagenet", task_data_path)
    if backend == "imagefolder":
        train_root, val_root = _resolve_imagefolder_roots(imagenet_dir)
        return imagenet_dir, train_root, val_root

    if _is_official_imagenet_root(imagenet_dir):
        return imagenet_dir, None, None

    if backend != "torchvision":
        train_root, val_root = _resolve_imagefolder_roots(imagenet_dir)
    return imagenet_dir, train_root, val_root


__all__ = [
    "_apply_imagenet_local_override",
    "_is_official_imagenet_root",
    "_normalized_imagenet_loader_backend",
    "_resolve_imagefolder_roots",
    "_resolve_imagenet_data_location",
]
