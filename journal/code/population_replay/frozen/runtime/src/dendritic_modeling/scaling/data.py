"""Deterministic, nested data sets for architecture scaling experiments.

Synthetic targets are fixed polynomial functions, independent of student models.
Changing a model's parameter budget does not change a task or its examples.
"""

from __future__ import annotations

import hashlib
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import TensorDataset

DATA_VERSION = "parameter-scaling-data-v1"


def canonical_hash(value: Any) -> str:
    """Hash JSON values without depending on dictionary insertion order."""
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def derived_seed(seed: int, purpose: str) -> int:
    return int(canonical_hash([int(seed), purpose])[:15], 16)


def _tensor_hash(*tensors: torch.Tensor) -> str:
    digest = hashlib.sha256()
    for tensor in tensors:
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(str((array.shape, array.dtype.str)).encode())
        digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


@dataclass
class DatasetBundle:
    train: TensorDataset
    validation: TensorDataset
    test: TensorDataset | None
    identity: dict[str, Any]
    sample_ids: dict[str, list[str]]


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or int(value) != value or int(value) < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _task_options(spec: dict[str, Any]) -> dict[str, Any]:
    complexity = spec.get("task_complexity", 8)
    options = (
        dict(complexity) if isinstance(complexity, dict) else {"terms": complexity}
    )
    unknown = set(options) - {
        "terms",
        "interaction_strength",
        "gain_log_std",
        "gain_target",
    }
    if unknown:
        raise ValueError(f"Unknown task_complexity options: {sorted(unknown)}")
    options["terms"] = _positive_int(options.get("terms", 8), "task_complexity.terms")
    options.setdefault("interaction_strength", 0.6)
    options.setdefault("gain_log_std", 0.7)
    options.setdefault("gain_target", "invariant")
    if options["gain_target"] not in {"invariant", "amplitude"}:
        raise ValueError("gain_target must be invariant or amplitude")
    for name in ("interaction_strength", "gain_log_std"):
        if not np.isfinite(options[name]) or options[name] < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    return options


def _synthetic_target(
    x: torch.Tensor,
    *,
    dataset: str,
    output_dim: int,
    target_seed: int,
    options: dict[str, Any],
) -> torch.Tensor:
    """A fixed local polynomial teacher; global controls rotate its coordinates."""
    generator = torch.Generator().manual_seed(
        derived_seed(target_seed, "polynomial-target")
    )
    target_x = x
    if dataset == "synthetic_gain":
        # The final input is a positive reference carrying the same nuisance gain.
        target_x = x[:, :-1]
        if options["gain_target"] == "invariant":
            target_x = target_x / x[:, -1:]
    if dataset == "synthetic_global":
        rotation_generator = torch.Generator().manual_seed(
            derived_seed(target_seed, "global-rotation")
        )
        rotation, _ = torch.linalg.qr(
            torch.randn(
                target_x.shape[1], target_x.shape[1], generator=rotation_generator
            )
        )
        target_x = target_x @ rotation

    dim = target_x.shape[1]
    terms = options["terms"]
    starts = torch.randint(dim, (terms,), generator=generator)
    a, b, c = (target_x[:, (starts + offset) % dim] for offset in range(3))
    # Gaussian polynomial features have controlled scale. This is not a copy of
    # any student architecture, and the same target is used at every model size.
    strength = float(options["interaction_strength"])
    features = torch.cat((a, strength * a * b, 0.5 * strength * a * b * c), dim=1)
    coefficients = torch.randn(features.shape[1], output_dim, generator=generator)
    logits = features @ coefficients / features.shape[1] ** 0.5
    return logits.argmax(dim=1)


def _synthetic_split(
    *,
    split: str,
    size: int,
    input_dim: int,
    output_dim: int,
    dataset: str,
    data_seed: int,
    target_seed: int,
    options: dict[str, Any],
) -> TensorDataset:
    # NumPy's normal generator is prefix stable as the number of rows changes;
    # torch.randn may change its normal-generation algorithm at small sizes.
    generator = np.random.default_rng(derived_seed(data_seed, f"examples-{split}"))
    x = torch.from_numpy(
        generator.standard_normal((size, input_dim)).astype(np.float32)
    )
    if dataset == "synthetic_gain":
        gain_generator = np.random.default_rng(derived_seed(data_seed, f"gain-{split}"))
        log_gain = gain_generator.normal(0.0, options["gain_log_std"], (size, 1))
        gain = torch.from_numpy(np.exp(log_gain).astype(np.float32))
        x[:, :-1] *= gain
        x[:, -1:] = gain
    y = _synthetic_target(
        x,
        dataset=dataset,
        output_dim=output_dim,
        target_seed=target_seed,
        options=options,
    )
    return TensorDataset(x, y)


def _cifar_directory(root: Path, *, download: bool) -> Path:
    candidates = [
        root / "cifar-10-batches-py",
        root,
        root / "data" / "cifar-10-batches-py",
    ]
    for candidate in candidates:
        if (candidate / "data_batch_1").is_file():
            return candidate
    if download:
        from torchvision.datasets import CIFAR10

        CIFAR10(root=str(root), train=True, download=True)
        return root / "cifar-10-batches-py"
    raise FileNotFoundError(
        f"No local CIFAR-10 Python batches found under {root}. "
        "Set data.root to the existing data directory; downloading is disabled."
    )


def _read_cifar_batches(
    directory: Path, *, train: bool
) -> tuple[np.ndarray, np.ndarray]:
    filenames = [f"data_batch_{i}" for i in range(1, 6)] if train else ["test_batch"]
    arrays, labels = [], []
    for filename in filenames:
        # Official CIFAR Python batches are trusted local dataset artifacts.
        with (directory / filename).open("rb") as handle:
            batch = pickle.load(handle, encoding="bytes")
        arrays.append(np.asarray(batch[b"data"], dtype=np.uint8))
        labels.append(np.asarray(batch[b"labels"], dtype=np.int64))
    data, targets = np.concatenate(arrays), np.concatenate(labels)
    if data.ndim != 2 or data.shape[1] != 3072 or len(data) != len(targets):
        raise ValueError("Invalid CIFAR-10 batch dimensions")
    return data, targets


def build_datasets(
    spec: dict[str, Any], model_spec: dict[str, Any], *, include_test: bool = False
) -> DatasetBundle:
    """Build disjoint splits, with a shared prefix for smaller training budgets.

    Test examples are only materialized when explicitly requested. The data and
    target seeds are independent of the model initialization seed.
    """
    dataset = str(spec.get("dataset", "synthetic_composition"))
    input_dim = _positive_int(model_spec["input_dim"], "model.input_dim")
    output_dim = _positive_int(model_spec["output_dim"], "model.output_dim")
    if output_dim < 2:
        raise ValueError("Classification tasks require output_dim >= 2")
    sizes = {
        "train": _positive_int(spec.get("train_size", 1024), "train_size"),
        "validation": _positive_int(
            spec.get("validation_size", 512), "validation_size"
        ),
        "test": _positive_int(spec.get("test_size", 512), "test_size"),
    }
    data_seed, target_seed = int(spec.get("data_seed", 1729)), int(
        spec.get("target_seed", 2718)
    )
    splits: dict[str, TensorDataset] = {}
    sample_ids: dict[str, list[str]] = {}
    identity: dict[str, Any] = {
        "version": DATA_VERSION,
        "dataset": dataset,
        "data_seed": data_seed,
        "target_seed": target_seed,
        "input_dim": input_dim,
        "output_dim": output_dim,
        "declared_sizes": sizes,
        "test_materialized": bool(include_test),
    }
    requested_splits = (
        ("train", "validation", "test") if include_test else ("train", "validation")
    )
    if dataset in {"synthetic_composition", "synthetic_global", "synthetic_gain"}:
        if input_dim < 4:
            raise ValueError("Synthetic polynomial tasks require input_dim >= 4")
        options = _task_options(spec)
        identity["task_options"] = options
        identity["input_transform"] = (
            "raw Gaussian coordinates; gain task adds shared gain/reference"
        )
        target_definition = {
            key: identity[key]
            for key in ("version", "dataset", "target_seed", "input_dim", "output_dim")
        }
        target_definition["polynomial"] = {
            key: options[key] for key in ("terms", "interaction_strength")
        }
        if dataset == "synthetic_gain":
            target_definition["gain_target"] = options["gain_target"]
        identity["target_sha256"] = canonical_hash(target_definition)
        for split in requested_splits:
            splits[split] = _synthetic_split(
                split=split,
                size=sizes[split],
                input_dim=input_dim,
                output_dim=output_dim,
                dataset=dataset,
                data_seed=data_seed,
                target_seed=target_seed,
                options=options,
            )
            sample_ids[split] = [
                f"{dataset}:{data_seed}:{split}:{i}" for i in range(sizes[split])
            ]
    elif dataset == "cifar10":
        if input_dim != 3072 or output_dim != 10:
            raise ValueError("CIFAR-10 requires input_dim=3072 and output_dim=10")
        directory = _cifar_directory(
            Path(spec.get("root", "data/cifar")),
            download=bool(spec.get("download", False)),
        )
        data, labels = _read_cifar_batches(directory, train=True)
        if sizes["train"] + sizes["validation"] > len(data):
            raise ValueError(
                "Requested CIFAR-10 training/validation splits exceed available training examples"
            )
        permutation = np.random.default_rng(
            derived_seed(data_seed, "cifar-split")
        ).permutation(len(data))
        val_size = sizes["validation"]
        indices = {
            "validation": permutation[:val_size],
            "train": permutation[val_size : val_size + sizes["train"]],
        }
        for split in ("train", "validation"):
            index = indices[split]
            x = torch.from_numpy(data[index]).float().div_(127.5).sub_(1.0)
            y = torch.from_numpy(labels[index])
            splits[split] = TensorDataset(x, y)
            sample_ids[split] = [f"cifar10:official_train:{int(i)}" for i in index]
        if include_test:
            test_data, test_labels = _read_cifar_batches(directory, train=False)
            if sizes["test"] > len(test_data):
                raise ValueError(
                    "Requested test_size exceeds the official CIFAR-10 test set"
                )
            # The official test set is separate from both model-selection splits.
            index = np.random.default_rng(
                derived_seed(data_seed, "cifar-test")
            ).permutation(len(test_data))[: sizes["test"]]
            splits["test"] = TensorDataset(
                torch.from_numpy(test_data[index]).float().div_(127.5).sub_(1.0),
                torch.from_numpy(test_labels[index]),
            )
            sample_ids["test"] = [f"cifar10:official_test:{int(i)}" for i in index]
        identity["input_transform"] = "CHW flatten; float32 x/127.5-1; no augmentation"
        identity["root"] = str(directory.resolve())
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    identity["splits"] = {
        split: {
            "size": len(data),
            "sample_ids_sha256": canonical_hash(sample_ids[split]),
            "content_sha256": _tensor_hash(*data.tensors),
            "class_counts": torch.bincount(
                data.tensors[1], minlength=output_dim
            ).tolist(),
        }
        for split, data in splits.items()
    }
    identity["sha256"] = canonical_hash(identity)
    return DatasetBundle(
        splits["train"], splits["validation"], splits.get("test"), identity, sample_ids
    )
