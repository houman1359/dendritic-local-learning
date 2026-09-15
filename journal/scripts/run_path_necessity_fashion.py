#!/usr/bin/env python3
"""Run and aggregate the frozen Fashion-MNIST path-necessity experiment.

The experiment varies whether inactive branch-local views support or conflict
with the class of a context-selected view.  Exact path transport updates only
the selected branch.  A neuron-shared coordinate applies the same somatic
teaching signal to every branch and therefore fails once incompatible branch
updates dominate.  Raw runs and logs live on kempner_project_b; aggregation
writes only compact, publication-facing source tables into the journal tree.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy.stats import wilcoxon
from torchvision.datasets import FashionMNIST


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "configs"
    / "path_necessity"
    / "credit_conflict_fashion_confirmatory.json"
)
SOURCE = ROOT / "source_data" / "path_necessity_fashion"

CONDITIONS = [
    "neuron_shared_k1",
    "correct_path",
    "within_neuron_deranged",
    "backpropagation",
    "gated_point_emulation",
]
EXACT_EQUIVALENTS = [
    "correct_path",
    "backpropagation",
    "gated_point_emulation",
]


@dataclass(frozen=True)
class FeaturePools:
    train: tuple[np.ndarray, np.ndarray]
    test: tuple[np.ndarray, np.ndarray]
    feature_dim: int
    provenance: dict[str, Any]


@dataclass(frozen=True)
class BaseTrials:
    context: np.ndarray
    label: np.ndarray
    selected: np.ndarray
    opposite: np.ndarray
    uniforms: np.ndarray
    selected_source_label: np.ndarray
    opposite_source_label: np.ndarray


def load_config() -> dict[str, Any]:
    with CONFIG.open() as handle:
        return json.load(handle)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _raw_dataset_hashes(root: Path) -> dict[str, str]:
    raw = root / "FashionMNIST" / "raw"
    required = [
        "train-images-idx3-ubyte",
        "train-labels-idx1-ubyte",
        "t10k-images-idx3-ubyte",
        "t10k-labels-idx1-ubyte",
    ]
    missing = [name for name in required if not (raw / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Fashion-MNIST cache is incomplete: {missing}")
    return {name: sha256(raw / name) for name in required}


def _pool_features(images: torch.Tensor, side: int) -> torch.Tensor:
    if images.ndim != 3 or images.shape[1:] != (28, 28):
        raise ValueError(f"unexpected Fashion-MNIST shape {tuple(images.shape)}")
    if 28 % side:
        raise ValueError("average_pool_side must divide 28")
    values = images.to(dtype=torch.float32).unsqueeze(1) / 255.0
    values = torch.nn.functional.avg_pool2d(values, kernel_size=side)
    return values.squeeze(1).reshape(len(values), -1)


def load_feature_pools(cfg: dict[str, Any]) -> FeaturePools:
    data_cfg = cfg["dataset"]
    root = Path(data_cfg["root"])
    train_raw = FashionMNIST(root=str(root), train=True, download=False)
    test_raw = FashionMNIST(root=str(root), train=False, download=False)
    class_a, class_b = (int(value) for value in data_cfg["classes"])
    side = int(data_cfg["average_pool_side"])

    train_features = _pool_features(train_raw.data, side)
    test_features = _pool_features(test_raw.data, side)
    train_keep = (train_raw.targets == class_a) | (train_raw.targets == class_b)
    test_keep = (test_raw.targets == class_a) | (test_raw.targets == class_b)
    train_labels = (train_raw.targets[train_keep] == class_b).to(torch.int64)
    test_labels = (test_raw.targets[test_keep] == class_b).to(torch.int64)
    train_features = train_features[train_keep]
    test_features = test_features[test_keep]

    mean = train_features.mean(dim=0)
    std = train_features.std(dim=0).clamp_min(1e-4)
    train_features = (train_features - mean) / std
    test_features = (test_features - mean) / std
    if bool(data_cfg["append_bias"]):
        train_features = torch.cat(
            [train_features, torch.ones(len(train_features), 1)], dim=1
        )
        test_features = torch.cat(
            [test_features, torch.ones(len(test_features), 1)], dim=1
        )

    train = (
        train_features.numpy().astype(np.float32, copy=False),
        train_labels.numpy(),
    )
    test = (
        test_features.numpy().astype(np.float32, copy=False),
        test_labels.numpy(),
    )
    counts_train = np.bincount(train[1], minlength=2)
    counts_test = np.bincount(test[1], minlength=2)
    if not np.array_equal(counts_train, [6000, 6000]):
        raise ValueError(f"unexpected binary train counts {counts_train.tolist()}")
    if not np.array_equal(counts_test, [1000, 1000]):
        raise ValueError(f"unexpected binary test counts {counts_test.tolist()}")

    provenance = {
        "dataset": "FashionMNIST",
        "classes": [class_a, class_b],
        "train_binary_counts": counts_train.tolist(),
        "test_binary_counts": counts_test.tolist(),
        "raw_sha256": _raw_dataset_hashes(root),
        "average_pool_side": side,
        "feature_dim": int(train[0].shape[1]),
    }
    return FeaturePools(
        train=train,
        test=test,
        feature_dim=int(train[0].shape[1]),
        provenance=provenance,
    )


def balanced_context_labels(
    rng: np.random.Generator, n: int, branches: int
) -> tuple[np.ndarray, np.ndarray]:
    cells = np.asarray(
        [(context, label) for context in range(branches) for label in (0, 1)],
        dtype=np.int64,
    )
    if n % len(cells):
        raise ValueError(f"{n=} must be divisible by 2B={len(cells)}")
    values = np.tile(cells, (n // len(cells), 1))
    rng.shuffle(values)
    return values[:, 0], values[:, 1]


def make_base_trials(
    rng: np.random.Generator,
    n: int,
    branches: int,
    features: np.ndarray,
    labels: np.ndarray,
) -> BaseTrials:
    context, target = balanced_context_labels(rng, n, branches)
    pools = [np.flatnonzero(labels == value) for value in (0, 1)]

    selected = np.empty((n, features.shape[1]), dtype=np.float32)
    opposite = np.empty((n, branches, features.shape[1]), dtype=np.float32)
    selected_source_label = target.copy()
    opposite_source_label = np.repeat((1 - target)[:, None], branches, axis=1)
    for label in (0, 1):
        rows = np.flatnonzero(target == label)
        selected_draws = rng.choice(pools[label], size=len(rows), replace=True)
        selected[rows] = features[selected_draws]
        opposite_draws = rng.choice(
            pools[1 - label], size=(len(rows), branches), replace=True
        )
        opposite[rows] = features[opposite_draws]

    return BaseTrials(
        context=context,
        label=target.astype(np.float32),
        selected=selected,
        opposite=opposite,
        uniforms=rng.random((n, branches)),
        selected_source_label=selected_source_label,
        opposite_source_label=opposite_source_label,
    )


def materialize_trials(
    base: BaseTrials, conflict_probability: float
) -> tuple[np.ndarray, np.ndarray]:
    branches = base.opposite.shape[1]
    values = np.repeat(base.selected[:, None, :], branches, axis=1)
    mask = base.uniforms < float(conflict_probability)
    mask[np.arange(len(mask)), base.context] = False
    values[mask] = base.opposite[mask]
    return values, mask


def route_matrices(branches: int) -> np.ndarray:
    exact = np.eye(branches, dtype=np.float32)
    shared = np.ones((branches, branches), dtype=np.float32) / float(branches)
    deranged = np.roll(exact, shift=1, axis=1)
    return np.stack([shared, exact, deranged, exact, exact], axis=0)


def theoretical_relative_signals(
    branches: int, conflict_probability: float
) -> tuple[float, float]:
    """Return shared and deranged useful signal relative to correct routing."""
    alpha = float(conflict_probability)
    shared = 1.0 - (2.0 * (branches - 1) / branches) * alpha
    deranged = 1.0 - 2.0 * alpha
    return shared, deranged


def _torch_device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return device


def logits(weights: torch.Tensor, values: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
    active = values[torch.arange(len(context), device=context.device), context]
    selected_weights = weights[:, context, :]
    return torch.einsum("mnd,nd->mn", selected_weights, active)


def analytic_gradients(
    weights: torch.Tensor,
    values: torch.Tensor,
    context: torch.Tensor,
    labels: torch.Tensor,
    routes: torch.Tensor,
) -> torch.Tensor:
    delta = torch.sigmoid(logits(weights, values, context)) - labels[None, :]
    route_batch = routes[:, context, :]
    return torch.einsum("mn,mnb,nbd->mbd", delta, route_batch, values) / len(labels)


def loss_accuracy(
    weights: torch.Tensor,
    values: torch.Tensor,
    context: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    prediction = logits(weights, values, context)
    loss = (
        torch.nn.functional.softplus(prediction) - labels[None, :] * prediction
    ).mean(dim=1)
    accuracy = ((prediction >= 0) == (labels[None, :] >= 0.5)).float().mean(dim=1)
    return loss, accuracy


def autograd_exact_difference(
    initial: torch.Tensor,
    values: torch.Tensor,
    context: torch.Tensor,
    labels: torch.Tensor,
) -> float:
    candidate = initial.detach().clone().requires_grad_(True)
    active = values[torch.arange(len(context), device=context.device), context]
    prediction = torch.sum(candidate[context] * active, dim=1)
    loss = (
        torch.nn.functional.softplus(prediction) - labels * prediction
    ).mean()
    (automatic,) = torch.autograd.grad(loss, candidate)
    exact_route = torch.eye(
        candidate.shape[0], dtype=candidate.dtype, device=candidate.device
    )[None, ...]
    analytic = analytic_gradients(
        candidate[None, ...], values, context, labels, exact_route
    )[0]
    return float(torch.max(torch.abs(automatic - analytic)).item())


def gradient_geometry(gradients: torch.Tensor, exact_index: int = 1) -> list[dict[str, float]]:
    exact = gradients[exact_index].reshape(-1)
    exact_norm_sq = torch.dot(exact, exact).clamp_min(1e-30)
    rows: list[dict[str, float]] = []
    for gradient in gradients:
        flat = gradient.reshape(-1)
        norm_sq = torch.dot(flat, flat).clamp_min(1e-30)
        dot = torch.dot(exact, flat)
        cosine = dot / torch.sqrt(exact_norm_sq * norm_sq)
        rows.append(
            {
                "initial_gradient_cosine": float(cosine.item()),
                "initial_gradient_capture": float((cosine * cosine).item()),
                "initial_signed_utility": float((dot / exact_norm_sq).item()),
            }
        )
    return rows


def train_models(
    initial: torch.Tensor,
    train_values: torch.Tensor,
    train_context: torch.Tensor,
    train_labels: torch.Tensor,
    routes: torch.Tensor,
    cfg: dict[str, Any],
) -> torch.Tensor:
    weights = initial[None, ...].repeat(len(CONDITIONS), 1, 1)
    learning_rate = float(cfg["training"]["learning_rate"])
    for _ in range(int(cfg["training"]["epochs"])):
        weights -= learning_rate * analytic_gradients(
            weights, train_values, train_context, train_labels, routes
        )
    return weights


def task_audit(
    base: BaseTrials,
    conflict_mask: np.ndarray,
    conflict_probability: float,
    branches: int,
) -> dict[str, Any]:
    counts = np.zeros((branches, 2), dtype=int)
    for context, label in zip(base.context, base.label.astype(int)):
        counts[int(context), int(label)] += 1
    inactive = np.ones_like(conflict_mask, dtype=bool)
    inactive[np.arange(len(inactive)), base.context] = False
    realized = float(conflict_mask[inactive].mean())
    return {
        "context_label_count_min": int(counts.min()),
        "context_label_count_max": int(counts.max()),
        "context_label_count_range": int(np.ptp(counts)),
        "context_label_correlation": float(
            np.corrcoef(base.context, base.label)[0, 1]
        ),
        "selected_label_mismatch_rate": float(
            np.mean(base.selected_source_label != base.label.astype(int))
        ),
        "opposite_label_match_rate": float(
            np.mean(
                base.opposite_source_label
                == base.label.astype(int)[:, None]
            )
        ),
        "target_conflict_probability": float(conflict_probability),
        "realized_inactive_conflict_probability": realized,
        "conflict_probability_abs_error": abs(realized - conflict_probability),
    }


def run_seed(
    seed: int,
    cfg: dict[str, Any],
    pools: FeaturePools,
    device: torch.device,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = False
    rows: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    maximum_autograd_difference = 0.0
    maximum_equivalent_difference = 0.0
    all_nested = True
    all_finite = True

    for branches in (int(value) for value in cfg["task"]["branches"]):
        rng = np.random.default_rng(seed + 10_000 * branches)
        train_base = make_base_trials(
            rng,
            int(cfg["dataset"]["train_examples"]),
            branches,
            pools.train[0],
            pools.train[1],
        )
        test_base = make_base_trials(
            rng,
            int(cfg["dataset"]["test_examples"]),
            branches,
            pools.test[0],
            pools.test[1],
        )
        initial_rng = np.random.default_rng(seed + 70_000 + branches)
        initial_np = initial_rng.normal(
            scale=float(cfg["training"]["initialization_sd"]),
            size=(branches, pools.feature_dim),
        ).astype(np.float32)
        initial = torch.as_tensor(initial_np, device=device)
        routes = torch.as_tensor(route_matrices(branches), device=device)
        train_context = torch.as_tensor(train_base.context, device=device)
        train_labels = torch.as_tensor(train_base.label, device=device)
        test_context = torch.as_tensor(test_base.context, device=device)
        test_labels = torch.as_tensor(test_base.label, device=device)
        previous_train_mask: np.ndarray | None = None
        previous_test_mask: np.ndarray | None = None

        for alpha in (float(value) for value in cfg["task"]["conflict_probability"]):
            train_np, train_mask = materialize_trials(train_base, alpha)
            test_np, test_mask = materialize_trials(test_base, alpha)
            if previous_train_mask is not None:
                all_nested &= bool(np.all(~previous_train_mask | train_mask))
                all_nested &= bool(np.all(~previous_test_mask | test_mask))
            previous_train_mask = train_mask
            previous_test_mask = test_mask

            train_values = torch.as_tensor(train_np, device=device)
            test_values = torch.as_tensor(test_np, device=device)
            initial_all = initial[None, ...].repeat(len(CONDITIONS), 1, 1)
            initial_gradients = analytic_gradients(
                initial_all,
                train_values,
                train_context,
                train_labels,
                routes,
            )
            geometry = gradient_geometry(initial_gradients)
            autograd_difference = autograd_exact_difference(
                initial, train_values, train_context, train_labels
            )
            maximum_autograd_difference = max(
                maximum_autograd_difference, autograd_difference
            )
            fitted = train_models(
                initial,
                train_values,
                train_context,
                train_labels,
                routes,
                cfg,
            )
            train_loss, train_accuracy = loss_accuracy(
                fitted, train_values, train_context, train_labels
            )
            test_loss, test_accuracy = loss_accuracy(
                fitted, test_values, test_context, test_labels
            )
            endpoint = torch.stack([test_loss, test_accuracy], dim=1)
            exact_reference = endpoint[CONDITIONS.index("correct_path")]
            for equivalent in EXACT_EQUIVALENTS:
                difference = torch.max(
                    torch.abs(endpoint[CONDITIONS.index(equivalent)] - exact_reference)
                )
                maximum_equivalent_difference = max(
                    maximum_equivalent_difference, float(difference.item())
                )
            all_finite &= bool(
                torch.isfinite(fitted).all()
                and torch.isfinite(train_loss).all()
                and torch.isfinite(test_loss).all()
            )

            theoretical_shared, theoretical_deranged = theoretical_relative_signals(
                branches, alpha
            )
            for index, condition in enumerate(CONDITIONS):
                rows.append(
                    {
                        "seed": seed,
                        "branches": branches,
                        "conflict_probability": alpha,
                        "condition": condition,
                        "train_loss": float(train_loss[index].item()),
                        "train_accuracy": float(train_accuracy[index].item()),
                        "test_loss": float(test_loss[index].item()),
                        "test_accuracy": float(test_accuracy[index].item()),
                        **geometry[index],
                        "theoretical_shared_relative_signal": theoretical_shared,
                        "theoretical_deranged_relative_signal": theoretical_deranged,
                        "feedback_rank": 1 if condition == "neuron_shared_k1" else branches,
                        "feedback_nonzeros_per_example": (
                            branches if condition == "neuron_shared_k1" else 1
                        ),
                        "forward_parameters": int(branches * pools.feature_dim),
                        "active_forward_branches_per_example": 1,
                        "simultaneously_driven_branches_per_example": branches,
                    }
                )

            train_audit = task_audit(train_base, train_mask, alpha, branches)
            test_audit = task_audit(test_base, test_mask, alpha, branches)
            audits.append(
                {
                    "seed": seed,
                    "branches": branches,
                    "conflict_probability": alpha,
                    **{f"train_{key}": value for key, value in train_audit.items()},
                    **{f"test_{key}": value for key, value in test_audit.items()},
                    "exact_autograd_gradient_max_abs_difference": autograd_difference,
                    "fallback_count": 0,
                }
            )

    metadata = {
        "seed": seed,
        "device": str(device),
        "gpu": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else None
        ),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "python": platform.python_version(),
        "config_sha256": sha256(CONFIG),
        "script_sha256": sha256(Path(__file__).resolve()),
        "maximum_exact_autograd_gradient_abs_difference": maximum_autograd_difference,
        "maximum_exact_equivalent_endpoint_difference": maximum_equivalent_difference,
        "nested_conflict_masks": all_nested,
        "all_finite": all_finite,
        "dataset": pools.provenance,
    }
    return pd.DataFrame(rows), pd.DataFrame(audits), metadata


def verify_canary(
    audit: pd.DataFrame, metadata: dict[str, Any], cfg: dict[str, Any]
) -> dict[str, Any]:
    gates = cfg["canary_gates"]
    maximum_count_range = int(
        max(audit.train_context_label_count_range.max(), audit.test_context_label_count_range.max())
    )
    maximum_selected_mismatch = float(
        max(
            audit.train_selected_label_mismatch_rate.max(),
            audit.test_selected_label_mismatch_rate.max(),
        )
    )
    maximum_conflict_error = float(
        max(
            audit.train_conflict_probability_abs_error.max(),
            audit.test_conflict_probability_abs_error.max(),
        )
    )
    result = {
        "maximum_context_label_count_range": maximum_count_range,
        "maximum_selected_label_mismatch_rate": maximum_selected_mismatch,
        "maximum_conflict_rate_absolute_error": maximum_conflict_error,
        "nested_conflict_masks": bool(metadata["nested_conflict_masks"]),
        "maximum_exact_autograd_gradient_abs_difference": float(
            metadata["maximum_exact_autograd_gradient_abs_difference"]
        ),
        "maximum_exact_equivalent_endpoint_difference": float(
            metadata["maximum_exact_equivalent_endpoint_difference"]
        ),
        "all_finite": bool(metadata["all_finite"]),
    }
    result["passed"] = bool(
        maximum_count_range
        <= int(gates["maximum_context_label_count_range"])
        and maximum_selected_mismatch
        <= float(gates["maximum_selected_label_mismatch_rate"])
        and maximum_conflict_error
        <= float(gates["maximum_conflict_rate_absolute_error"])
        and result["nested_conflict_masks"]
        == bool(gates["require_nested_conflict_masks"])
        and result["maximum_exact_autograd_gradient_abs_difference"]
        <= float(gates["maximum_exact_autograd_gradient_abs_difference"])
        and result["maximum_exact_equivalent_endpoint_difference"]
        <= float(gates["maximum_exact_equivalent_endpoint_difference"])
        and (result["all_finite"] or not bool(gates["require_all_finite"]))
    )
    return result


def write_seed_output(
    phase: str,
    seed: int,
    outcomes: pd.DataFrame,
    audit: pd.DataFrame,
    metadata: dict[str, Any],
    cfg: dict[str, Any],
) -> Path:
    output = Path(cfg["output_root"]) / phase
    output.mkdir(parents=True, exist_ok=True)
    outcomes.to_csv(output / f"seed_{seed}_outcomes.csv", index=False, float_format="%.10g")
    audit.to_csv(output / f"seed_{seed}_audit.csv", index=False, float_format="%.10g")
    if phase == "canary":
        metadata["canary_gates"] = verify_canary(audit, metadata, cfg)
    with (output / f"seed_{seed}_metadata.json").open("w") as handle:
        json.dump(metadata, handle, indent=2)
        handle.write("\n")
    return output


def bootstrap_mean(
    values: np.ndarray, *, seed: int, draws: int
) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(draws, len(values)), replace=True).mean(axis=1)
    low, high = np.quantile(samples, [0.025, 0.975])
    return float(values.mean()), float(low), float(high)


def paired_test(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    if np.allclose(values, 0):
        return 1.0
    return float(wilcoxon(values, alternative="two-sided", zero_method="wilcox").pvalue)


def aggregate(cfg: dict[str, Any]) -> dict[str, Any]:
    raw = Path(cfg["output_root"]) / "confirmatory"
    seeds = [int(value) for value in cfg["confirmatory_seeds"]]
    outcome_paths = [raw / f"seed_{seed}_outcomes.csv" for seed in seeds]
    audit_paths = [raw / f"seed_{seed}_audit.csv" for seed in seeds]
    metadata_paths = [raw / f"seed_{seed}_metadata.json" for seed in seeds]
    missing = [
        str(path)
        for path in outcome_paths + audit_paths + metadata_paths
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(f"confirmatory output is incomplete: {missing[:8]}")
    outcomes = pd.concat([pd.read_csv(path) for path in outcome_paths], ignore_index=True)
    audits = pd.concat([pd.read_csv(path) for path in audit_paths], ignore_index=True)
    metadata = [json.loads(path.read_text()) for path in metadata_paths]
    expected_rows = (
        len(seeds)
        * len(cfg["task"]["branches"])
        * len(cfg["task"]["conflict_probability"])
        * len(CONDITIONS)
    )
    if len(outcomes) != expected_rows:
        raise ValueError(f"outcome rows {len(outcomes)} != expected {expected_rows}")
    if outcomes.seed.nunique() != len(seeds):
        raise ValueError("seed count mismatch")

    draws = int(cfg["analysis"]["bootstrap_draws"])
    summary_rows: list[dict[str, Any]] = []
    metrics = [
        "test_accuracy",
        "test_loss",
        "initial_gradient_cosine",
        "initial_gradient_capture",
        "initial_signed_utility",
    ]
    for group_index, (keys, part) in enumerate(
        outcomes.groupby(["branches", "conflict_probability", "condition"], sort=True)
    ):
        row: dict[str, Any] = dict(
            zip(["branches", "conflict_probability", "condition"], keys)
        )
        row["n_seeds"] = int(part.seed.nunique())
        for metric_index, metric in enumerate(metrics):
            mean, low, high = bootstrap_mean(
                part[metric].to_numpy(float),
                seed=710_000 + 20 * group_index + metric_index,
                draws=draws,
            )
            row[f"mean_{metric}"] = mean
            row[f"ci95_low_{metric}"] = low
            row[f"ci95_high_{metric}"] = high
        summary_rows.append(row)

    contrast_rows: list[dict[str, Any]] = []
    comparisons = [
        ("correct_path", "neuron_shared_k1", "correct - shared"),
        ("correct_path", "within_neuron_deranged", "correct - deranged"),
        ("correct_path", "backpropagation", "correct - BP"),
        ("correct_path", "gated_point_emulation", "correct - gated point"),
    ]
    for branches in cfg["task"]["branches"]:
        for alpha_index, alpha in enumerate(cfg["task"]["conflict_probability"]):
            part = outcomes[
                outcomes.branches.eq(int(branches))
                & np.isclose(outcomes.conflict_probability, float(alpha))
            ]
            for comparison_index, (left, right, label) in enumerate(comparisons):
                for metric_index, metric in enumerate(["test_accuracy", "test_loss"]):
                    wide = part.pivot(index="seed", columns="condition", values=metric)
                    values = (wide[left] - wide[right]).to_numpy(float)
                    mean, low, high = bootstrap_mean(
                        values,
                        seed=(
                            730_000
                            + 10_000 * int(branches)
                            + 100 * alpha_index
                            + 10 * comparison_index
                            + metric_index
                        ),
                        draws=draws,
                    )
                    contrast_rows.append(
                        {
                            "branches": int(branches),
                            "conflict_probability": float(alpha),
                            "contrast": label,
                            "left_condition": left,
                            "right_condition": right,
                            "endpoint": metric,
                            "n_pairs": len(values),
                            "mean_difference": mean,
                            "ci95_low": low,
                            "ci95_high": high,
                            "positive_pairs": int(np.sum(values > 0)),
                            "ties": int(np.sum(np.isclose(values, 0))),
                            "wilcoxon_p_two_sided": paired_test(values),
                        }
                    )

    interaction_rows: list[dict[str, Any]] = []
    boundary_rows: list[dict[str, Any]] = []
    for branches in (int(value) for value in cfg["task"]["branches"]):
        part = outcomes[outcomes.branches.eq(branches)]
        for seed in seeds:
            seed_part = part[part.seed.eq(seed)]
            accuracy = seed_part.pivot(
                index="conflict_probability", columns="condition", values="test_accuracy"
            ).sort_index()
            advantage = accuracy.correct_path - accuracy.neuron_shared_k1
            slope = float(np.polyfit(accuracy.index.to_numpy(float), advantage, 1)[0])
            interaction_rows.append(
                {"seed": seed, "branches": branches, "accuracy_interaction_slope": slope}
            )
            shared_rows = seed_part[seed_part.condition.eq("neuron_shared_k1")].sort_values(
                "conflict_probability"
            )
            utility_nonpositive = shared_rows[
                shared_rows.initial_signed_utility.le(0)
            ]
            accuracy_at_chance = shared_rows[shared_rows.test_accuracy.le(0.5)]
            boundary_rows.append(
                {
                    "seed": seed,
                    "branches": branches,
                    "predicted_boundary": branches / (2.0 * (branches - 1)),
                    "first_nonpositive_utility_dose": (
                        float(utility_nonpositive.conflict_probability.iloc[0])
                        if len(utility_nonpositive)
                        else math.nan
                    ),
                    "first_at_or_below_chance_accuracy_dose": (
                        float(accuracy_at_chance.conflict_probability.iloc[0])
                        if len(accuracy_at_chance)
                        else math.nan
                    ),
                }
            )

    interaction = pd.DataFrame(interaction_rows)
    interaction_summary: list[dict[str, Any]] = []
    for branches, part in interaction.groupby("branches", sort=True):
        values = part.accuracy_interaction_slope.to_numpy(float)
        mean, low, high = bootstrap_mean(
            values, seed=760_000 + int(branches), draws=draws
        )
        interaction_summary.append(
            {
                "branches": int(branches),
                "n_seeds": len(values),
                "mean_accuracy_interaction_slope": mean,
                "ci95_low": low,
                "ci95_high": high,
                "positive_pairs": int(np.sum(values > 0)),
                "wilcoxon_p_two_sided": paired_test(values),
            }
        )

    summary = pd.DataFrame(summary_rows)
    contrasts = pd.DataFrame(contrast_rows)
    boundaries = pd.DataFrame(boundary_rows)
    interaction_summary_frame = pd.DataFrame(interaction_summary)

    interpretation = cfg["interpretation_gates"]
    alpha_zero = contrasts[
        contrasts.contrast.eq("correct - shared")
        & contrasts.endpoint.eq("test_accuracy")
        & np.isclose(contrasts.conflict_probability, 0.0)
    ]
    alpha_one = contrasts[
        contrasts.contrast.eq("correct - shared")
        & contrasts.endpoint.eq("test_accuracy")
        & np.isclose(contrasts.conflict_probability, 1.0)
    ]
    deranged_one = contrasts[
        contrasts.contrast.eq("correct - deranged")
        & contrasts.endpoint.eq("test_accuracy")
        & np.isclose(contrasts.conflict_probability, 1.0)
    ]
    minimum_positive = float(
        interpretation["minimum_positive_seedwise_interaction_fraction"]
    )
    equivalence_margin = float(
        cfg["analysis"]["equivalence_margin_accuracy_at_zero_conflict"]
    )
    exact_equivalence = max(
        float(item["maximum_exact_equivalent_endpoint_difference"])
        for item in metadata
    )
    gate_summary = {
        "complete_seed_count": int(outcomes.seed.nunique()),
        "all_finite": bool(all(bool(item["all_finite"]) for item in metadata)),
        "maximum_exact_autograd_gradient_abs_difference": max(
            float(item["maximum_exact_autograd_gradient_abs_difference"])
            for item in metadata
        ),
        "maximum_exact_equivalent_endpoint_difference": exact_equivalence,
        "all_interaction_positive_fraction_pass": bool(
            np.all(
                interaction_summary_frame.positive_pairs
                / interaction_summary_frame.n_seeds
                >= minimum_positive
            )
        ),
        "all_full_conflict_effects_pass": bool(
            np.all(
                alpha_one.mean_difference
                >= float(
                    interpretation[
                        "minimum_correct_minus_shared_accuracy_at_full_conflict"
                    ]
                )
            )
        ),
        "all_zero_conflict_effects_within_margin": bool(
            np.all(np.abs(alpha_zero.mean_difference) <= equivalence_margin)
        ),
        "all_full_conflict_derangement_effects_positive": bool(
            np.all(deranged_one.mean_difference > 0)
        ),
        "exact_equivalence_checks_pass": bool(
            exact_equivalence
            <= float(
                cfg["canary_gates"]["maximum_exact_equivalent_endpoint_difference"]
            )
        ),
    }
    gate_summary["conditional_path_necessity_supported"] = bool(
        gate_summary["complete_seed_count"] == len(seeds)
        and gate_summary["all_finite"]
        and gate_summary["all_interaction_positive_fraction_pass"]
        and gate_summary["all_full_conflict_effects_pass"]
        and gate_summary["all_zero_conflict_effects_within_margin"]
        and gate_summary["all_full_conflict_derangement_effects_positive"]
        and gate_summary["exact_equivalence_checks_pass"]
    )

    SOURCE.mkdir(parents=True, exist_ok=True)
    outcomes.to_csv(SOURCE / "seed_outcomes.csv", index=False, float_format="%.10g")
    audits.to_csv(SOURCE / "mechanism_audit.csv", index=False, float_format="%.10g")
    summary.to_csv(SOURCE / "condition_summary.csv", index=False, float_format="%.10g")
    contrasts.to_csv(SOURCE / "paired_contrasts.csv", index=False, float_format="%.10g")
    interaction.to_csv(SOURCE / "seedwise_interactions.csv", index=False, float_format="%.10g")
    interaction_summary_frame.to_csv(
        SOURCE / "interaction_summary.csv", index=False, float_format="%.10g"
    )
    boundaries.to_csv(SOURCE / "boundary_by_seed.csv", index=False, float_format="%.10g")
    with (SOURCE / "audit.json").open("w") as handle:
        json.dump(gate_summary, handle, indent=2)
        handle.write("\n")

    report_lines = [
        "# Fashion-MNIST path-necessity credit-conflict result",
        "",
        f"Paired confirmatory seeds: {len(seeds)}.",
        f"Interpretation gate passed: {gate_summary['conditional_path_necessity_supported']}.",
        "",
        "## Seed-wise route-by-conflict interactions",
        "",
    ]
    for row in interaction_summary:
        report_lines.append(
            f"- B={row['branches']}: mean slope {row['mean_accuracy_interaction_slope']:.4f} "
            f"({row['ci95_low']:.4f} to {row['ci95_high']:.4f}); "
            f"{row['positive_pairs']}/{row['n_seeds']} positive."
        )
    report_lines.extend(["", "## Endpoint contrasts", ""])
    for branches in cfg["task"]["branches"]:
        zero = alpha_zero[alpha_zero.branches.eq(int(branches))].iloc[0]
        one = alpha_one[alpha_one.branches.eq(int(branches))].iloc[0]
        report_lines.append(
            f"- B={int(branches)}: correct-shared accuracy at alpha=0: "
            f"{zero.mean_difference:.4f} ({zero.ci95_low:.4f} to {zero.ci95_high:.4f}); "
            f"at alpha=1: {one.mean_difference:.4f} "
            f"({one.ci95_low:.4f} to {one.ci95_high:.4f})."
        )
    report_lines.extend(
        [
            "",
            "This controlled family tests conditional need for a path address. ",
            "The grouped-point equality check prevents interpreting a positive result as a dendrite-exclusive advantage.",
            "",
        ]
    )
    (SOURCE / "RESULTS.md").write_text("\n".join(report_lines))
    return gate_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase", choices=["canary", "confirmatory", "aggregate"], required=True
    )
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config()
    if args.phase == "aggregate":
        result = aggregate(cfg)
        print(json.dumps(result, indent=2))
        return

    allowed = (
        [int(cfg["canary_seed"])]
        if args.phase == "canary"
        else [int(value) for value in cfg["confirmatory_seeds"]]
    )
    seed = allowed[0] if args.seed is None and len(allowed) == 1 else args.seed
    if seed is None or int(seed) not in allowed:
        raise SystemExit(f"--seed must be one of {allowed}")
    device = _torch_device(args.device)
    pools = load_feature_pools(cfg)
    outcomes, audit, metadata = run_seed(int(seed), cfg, pools, device)
    output = write_seed_output(
        args.phase, int(seed), outcomes, audit, metadata, cfg
    )
    print(f"wrote {output}")
    if args.phase == "canary":
        gates = metadata.get("canary_gates")
        # write_seed_output mutates metadata before serialization.
        gates = verify_canary(audit, metadata, cfg)
        print(json.dumps(gates, indent=2))
        if not gates["passed"]:
            raise SystemExit("canary gates failed")


if __name__ == "__main__":
    main()
