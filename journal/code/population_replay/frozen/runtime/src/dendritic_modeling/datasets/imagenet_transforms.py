"""Transform builders for ImageNet datasets."""

from typing import Any

from torchvision import transforms


def _append_imagenet_normalize(tf_list: list, *, normalize: bool) -> list:
    """Append ImageNet normalization to a transform list when requested."""
    if normalize:
        tf_list.append(
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            )
        )
    return tf_list


def _normalized_imagenet_preset_name(transform_preset: str | None) -> str:
    return str(transform_preset or "legacy_resize").strip().lower().replace("-", "_")


def _legacy_resize_transform_steps() -> list:
    return [transforms.Resize((224, 224)), transforms.ToTensor()]


def _pretrained_train_transform_steps() -> list:
    return [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]


def _pretrained_eval_transform_steps() -> list:
    return [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ]


def _build_imagenet_transforms(
    *,
    normalize: bool,
    transform_preset: str,
) -> tuple[Any, Any]:
    """Build train/eval transforms for ImageNet without changing preset semantics."""
    preset = _normalized_imagenet_preset_name(transform_preset)
    if preset in {"legacy", "legacy_resize", "resize", "resize_224"}:
        transform = transforms.Compose(
            _append_imagenet_normalize(
                _legacy_resize_transform_steps(),
                normalize=normalize,
            )
        )
        return transform, transform

    if preset in {"pretrained", "torchvision", "imagenet", "standard"}:
        train_transform = transforms.Compose(
            _append_imagenet_normalize(
                _pretrained_train_transform_steps(),
                normalize=normalize,
            )
        )
        eval_transform = transforms.Compose(
            _append_imagenet_normalize(
                _pretrained_eval_transform_steps(),
                normalize=normalize,
            )
        )
        return train_transform, eval_transform

    raise ValueError(
        "Unknown ImageNet transform_preset "
        f"{transform_preset!r}; expected 'legacy_resize' or 'pretrained'."
    )


__all__ = [
    "_append_imagenet_normalize",
    "_build_imagenet_transforms",
    "_normalized_imagenet_preset_name",
]
