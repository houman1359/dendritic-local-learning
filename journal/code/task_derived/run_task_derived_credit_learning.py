#!/usr/bin/env python3
"""Task-derived credit capture and projected local learning on MICrONS pairs.

This analysis uses the directly connected, functionally imaged presynaptic
partners of reconstructed targets as learning sites.  A small branch neuron is
fit to predict the postsynaptic response.  Exact branch-error fields are then
projected through anatomy-defined feedback dictionaries, and the projected
errors are used to train matched models from scratch.

The projection coefficients are optimal least-squares coefficients.  Thus this
is a capacity/sufficiency experiment, not a claim that the coefficient encoder
is biologically available.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from morphology_tree_helpers import ancestry_matrix, parent_map


PROJECT = Path(__file__).resolve().parents[2]
DEFAULT_EXTRACTS = PROJECT / "reproduced_results" / "microns_functional_partner_responses"
DEFAULT_SEGMENTS = PROJECT / "reproduced_results" / "microns_morphology_credit" / "segment_metrics.csv"
DEFAULT_OUTDIR = PROJECT / "reproduced_results" / "microns_task_derived_credit_learning"
METHODS = [
    "exact backprop",
    "dense PCA oracle",
    "morphology-aware paths",
    "random nonempty paths",
    "depth-only bins",
    "shuffled ancestry",
    "scalar broadcast",
]
COLORS = {
    "exact backprop": "#222222",
    "dense PCA oracle": "#e69f00",
    "morphology-aware paths": "#26828e",
    "random nonempty paths": "#8c8c8c",
    "depth-only bins": "#7a5195",
    "shuffled ancestry": "#cc79a7",
    "scalar broadcast": "#56b4e9",
}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def stable_rng(seed: int, root_id: int, stream: int, replicate: int = 0) -> np.random.Generator:
    root_id = int(root_id)
    entropy = [
        int(seed) & 0xFFFFFFFF,
        root_id & 0xFFFFFFFF,
        (root_id >> 32) & 0xFFFFFFFF,
        int(stream) & 0xFFFFFFFF,
        int(replicate) & 0xFFFFFFFF,
    ]
    return np.random.default_rng(np.random.SeedSequence(entropy))


def projector(dictionary: np.ndarray, tolerance: float = 1e-10) -> np.ndarray:
    """Orthogonal projector onto a possibly rank-deficient dictionary."""

    dictionary = np.asarray(dictionary, dtype=float)
    if dictionary.ndim != 2 or dictionary.shape[0] == 0 or dictionary.shape[1] == 0:
        return np.zeros((dictionary.shape[0], dictionary.shape[0]), dtype=float)
    u, singular, _ = np.linalg.svd(dictionary, full_matrices=False)
    if not len(singular) or singular[0] <= 0:
        return np.zeros((dictionary.shape[0], dictionary.shape[0]), dtype=float)
    keep = singular > float(tolerance) * singular[0]
    basis = u[:, keep]
    return basis @ basis.T


def capture(delta: np.ndarray, projection: np.ndarray) -> float:
    delta = np.asarray(delta, dtype=float)
    reconstruction = delta @ projection
    denominator = float(np.sum(delta * delta))
    if denominator <= 0:
        return float("nan")
    return float(np.clip(1.0 - np.sum((delta - reconstruction) ** 2) / denominator, 0.0, 1.0))


def grouped_split(stimulus_ids: np.ndarray, rng: np.random.Generator, test_fraction: float) -> tuple[np.ndarray, np.ndarray]:
    unique = np.unique(stimulus_ids)
    shuffled = rng.permutation(unique)
    n_test = max(1, int(round(float(test_fraction) * len(unique))))
    test_ids = set(shuffled[:n_test].tolist())
    test = np.asarray([value in test_ids for value in stimulus_ids], dtype=bool)
    return ~test, test


def load_target(extract_dir: Path) -> dict[str, Any]:
    protocol = np.load(extract_dir / "microns_dandi_trial_protocol.npz")
    mapping = pd.read_csv(extract_dir / "microns_dandi_trial_unit_mapping.csv")
    contacts = pd.read_csv(extract_dir / "functional_topology" / "functional_contacts.csv")
    manifest = pd.read_csv(extract_dir / "extraction_manifest.csv")
    responses = np.asarray(protocol["responses"], dtype=float)
    partner_columns = np.flatnonzero(mapping["role"].eq("presynaptic_partner").to_numpy())
    target_columns = np.flatnonzero(mapping["role"].eq("postsynaptic_target").to_numpy())
    if len(target_columns) != 1:
        raise ValueError(f"expected one postsynaptic target, found {len(target_columns)}")
    if len(partner_columns) != len(contacts):
        raise ValueError("partner/contact count mismatch")
    order = contacts["unit_index"].to_numpy(dtype=int)
    x = responses[:, partner_columns][:, order]
    y = responses[:, target_columns[0]]
    return {
        "root_id": int(manifest["target_root_id"].iloc[0]),
        "nucleus_id": int(manifest["target_nucleus_id"].iloc[0]),
        "x": x,
        "y": y,
        "stimulus_ids": np.asarray(protocol["stimulus_ids"], dtype=int),
        "contacts": contacts,
    }


def standardize(x: np.ndarray, train: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = np.nanmean(x[train], axis=0, keepdims=True)
    scale = np.nanstd(x[train], axis=0, ddof=1, keepdims=True)
    scale = np.where(np.isfinite(scale) & (scale > 1e-8), scale, 1.0)
    return np.nan_to_num((x - mean) / scale), mean, scale


def remap_labels(values: np.ndarray) -> np.ndarray:
    unique = {value: index for index, value in enumerate(np.unique(values))}
    return np.asarray([unique[value] for value in values], dtype=int)


def anatomy_dictionaries(
    contacts: pd.DataFrame,
    segments: pd.DataFrame,
    channels: int,
    rng: np.random.Generator,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    _, parents, _ = parent_map(segments)
    sites = contacts["segment_id"].astype(int).tolist()
    inhibitory = segments.loc[segments["I_size"] > 0, "segment_id"].astype(int).tolist()
    lookup = segments.set_index("segment_id")
    ancestry = ancestry_matrix(sites, inhibitory, parents)
    beta = (
        lookup.loc[inhibitory, "g_i"].to_numpy(dtype=float)
        / np.maximum(lookup.loc[inhibitory, "g_total"].to_numpy(dtype=float), 1e-12)
    )
    kernel = ancestry * beta[None, :]
    candidates = np.flatnonzero(np.any(np.abs(kernel) > 0, axis=0))
    if not len(candidates):
        raise ValueError("no nonempty inhibitory routes reach observed sites")
    used_channels = min(int(channels), len(sites), len(candidates))
    leverage = beta * ancestry.mean(axis=0)
    selected = candidates[np.argsort(-leverage[candidates])[:used_channels]]
    morphology = kernel[:, selected]
    random_selected = rng.choice(candidates, size=used_channels, replace=False)
    random_paths = kernel[:, random_selected]
    shuffled = morphology.copy()
    for column in range(shuffled.shape[1]):
        shuffled[:, column] = rng.permutation(shuffled[:, column])
    if used_channels == 1:
        depth = np.ones((len(sites), 1), dtype=float)
    else:
        path_depth = contacts["path_um"].to_numpy(dtype=float)
        edges = np.unique(np.quantile(path_depth, np.linspace(0, 1, used_channels + 1)))
        labels = np.digitize(path_depth, edges[1:-1], right=True)
        depth = np.eye(int(labels.max()) + 1, dtype=float)[labels]
    dictionaries = {
        "morphology-aware paths": morphology,
        "random nonempty paths": random_paths,
        "depth-only bins": depth,
        "shuffled ancestry": shuffled,
        "scalar broadcast": np.ones((len(sites), 1), dtype=float),
    }
    metadata = {
        "requested_channels": int(channels),
        "used_channels": int(used_channels),
        "n_candidate_routes": int(len(candidates)),
        "selected_inhibitory_segments": [int(inhibitory[index]) for index in selected],
        "random_inhibitory_segments": [int(inhibitory[index]) for index in random_selected],
        "dictionary_ranks": {key: int(np.linalg.matrix_rank(value)) for key, value in dictionaries.items()},
        "dictionary_nonzeros": {key: int(np.count_nonzero(value)) for key, value in dictionaries.items()},
    }
    return dictionaries, metadata


@dataclass
class ModelState:
    w: np.ndarray
    v: np.ndarray
    bias: float


def initialize_model(n_sites: int, n_branches: int, rng: np.random.Generator) -> ModelState:
    return ModelState(
        w=rng.normal(0.0, 0.12 / np.sqrt(max(n_sites, 1)), size=n_sites),
        v=rng.normal(0.0, 0.25 / np.sqrt(max(n_branches, 1)), size=n_branches),
        bias=0.0,
    )


def activate(drive: np.ndarray, activation: str) -> tuple[np.ndarray, np.ndarray]:
    if activation == "tanh":
        hidden = np.tanh(drive)
        return hidden, 1.0 - hidden * hidden
    if activation == "linear":
        return drive, np.ones_like(drive)
    raise ValueError(f"unknown activation: {activation}")


def forward(
    x: np.ndarray, state: ModelState, branch_ids: np.ndarray, activation: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_branches = int(branch_ids.max()) + 1
    drive = np.zeros((len(x), n_branches), dtype=float)
    for branch in range(n_branches):
        selected = branch_ids == branch
        drive[:, branch] = x[:, selected] @ state.w[selected]
    hidden, derivative = activate(drive, activation)
    prediction = hidden @ state.v + state.bias
    return prediction, drive, hidden, derivative


def error_field(
    x: np.ndarray,
    y: np.ndarray,
    state: ModelState,
    branch_ids: np.ndarray,
    activation: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    prediction, _, _, derivative = forward(x, state, branch_ids, activation)
    error = prediction - y
    branch_delta = error[:, None] * state.v[None, :] * derivative
    site_delta = branch_delta[:, branch_ids]
    return prediction, error, site_delta


def train_model(
    x: np.ndarray,
    y: np.ndarray,
    branch_ids: np.ndarray,
    initial: ModelState,
    projection: np.ndarray | None,
    steps: int,
    learning_rate: float,
    weight_decay: float,
    activation: str,
) -> tuple[ModelState, list[float]]:
    state = ModelState(initial.w.copy(), initial.v.copy(), float(initial.bias))
    moment1 = {"w": np.zeros_like(state.w), "v": np.zeros_like(state.v), "bias": 0.0}
    moment2 = {"w": np.zeros_like(state.w), "v": np.zeros_like(state.v), "bias": 0.0}
    history: list[float] = []
    for step in range(1, int(steps) + 1):
        prediction, _, hidden, derivative = forward(x, state, branch_ids, activation)
        error = prediction - y
        branch_delta = error[:, None] * state.v[None, :] * derivative
        site_delta = branch_delta[:, branch_ids]
        routed_delta = site_delta if projection is None else site_delta @ projection
        gradients = {
            "w": np.mean(x * routed_delta, axis=0) + float(weight_decay) * state.w,
            "v": hidden.T @ error / len(x) + float(weight_decay) * state.v,
            "bias": float(np.mean(error)),
        }
        for name, gradient in gradients.items():
            moment1[name] = 0.9 * moment1[name] + 0.1 * gradient
            moment2[name] = 0.999 * moment2[name] + 0.001 * gradient * gradient
            mhat = moment1[name] / (1.0 - 0.9**step)
            vhat = moment2[name] / (1.0 - 0.999**step)
            update = float(learning_rate) * mhat / (np.sqrt(vhat) + 1e-8)
            if name == "bias":
                state.bias -= float(update)
            else:
                setattr(state, name, getattr(state, name) - update)
        if step == 1 or step % 25 == 0 or step == int(steps):
            objective = 0.5 * float(np.mean(error * error)) + 0.5 * float(weight_decay) * (
                float(np.sum(state.w * state.w)) + float(np.sum(state.v * state.v))
            )
            history.append(objective)
    return state, history


def paired_summary(frame: pd.DataFrame, metric: str, control: str, seed: int) -> dict[str, Any]:
    means = frame.groupby(["target_root_id", "method"], as_index=False)[metric].mean()
    wide = means.pivot(index="target_root_id", columns="method", values=metric).dropna()
    values = (wide["morphology-aware paths"] - wide[control]).to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    boot = rng.choice(values, size=(20_000, len(values)), replace=True).mean(axis=1)
    test = stats.wilcoxon(values, alternative="two-sided") if len(values) >= 2 and not np.allclose(values, 0) else None
    return {
        "contrast": f"morphology-aware paths minus {control}",
        "n_targets": int(len(values)),
        "mean_difference": float(np.mean(values)),
        "target_bootstrap_ci95": [float(x) for x in np.quantile(boot, [0.025, 0.975])],
        "morphology_lower_count": int(np.sum(values < 0)),
        "morphology_higher_count": int(np.sum(values > 0)),
        "wilcoxon_two_sided_p": float(test.pvalue) if test is not None else float("nan"),
    }


def clustered_capture_utility_association(
    frame: pd.DataFrame,
    seed: int,
    n_bootstrap: int = 20_000,
    n_permutations: int = 20_000,
) -> dict[str, Any]:
    """Test the within-target capture--utility association.

    Target means are removed from both quantities before computing Spearman's
    rho.  The null reassigns the learning outcomes to method labels separately
    within each target and recomputes that same centered statistic.  Thus the
    permutation cannot exchange observations across biological target cells,
    and its reference statistic is not the pooled correlation.
    """

    selected = (
        frame[~frame["method"].isin(["exact backprop", "dense PCA oracle"])]
        .dropna(subset=["heldout_credit_capture", "heldout_normalized_mse"])
        .sort_values(["target_root_id", "method"])
        .reset_index(drop=True)
    )
    if selected.empty:
        raise ValueError("no non-oracle target-by-method observations remain")
    if selected.duplicated(["target_root_id", "method"]).any():
        raise ValueError("expected one mean per target and method")

    method_sets = selected.groupby("target_root_id")["method"].agg(lambda x: tuple(sorted(x)))
    if method_sets.nunique() != 1:
        raise ValueError("every target must contain the same structural feedback methods")

    centered = selected.copy()
    metrics = ["heldout_credit_capture", "heldout_normalized_mse"]
    for metric in metrics:
        centered[metric] -= centered.groupby("target_root_id")[metric].transform("mean")
    capture_centered = centered[metrics[0]].to_numpy(dtype=float)
    mse_centered = centered[metrics[1]].to_numpy(dtype=float)
    within_target = float(stats.spearmanr(capture_centered, mse_centered).statistic)
    pooled_descriptive = float(
        stats.spearmanr(
            selected["heldout_credit_capture"], selected["heldout_normalized_mse"]
        ).statistic
    )

    targets = centered["target_root_id"].drop_duplicates().to_numpy()
    group_indices = {
        target: np.flatnonzero(centered["target_root_id"].to_numpy() == target)
        for target in targets
    }
    group_lengths = {len(group_indices[target]) for target in targets}
    if len(group_lengths) != 1:
        raise ValueError("target groups must have equal method counts")
    group_index_matrix = np.stack([group_indices[target] for target in targets], axis=0)
    rng = np.random.default_rng(seed)

    def rowwise_spearman(first: np.ndarray, second: np.ndarray) -> np.ndarray:
        first_rank = stats.rankdata(first, axis=1)
        second_rank = stats.rankdata(second, axis=1)
        first_rank -= first_rank.mean(axis=1, keepdims=True)
        second_rank -= second_rank.mean(axis=1, keepdims=True)
        denominator = np.linalg.norm(first_rank, axis=1) * np.linalg.norm(second_rank, axis=1)
        return np.sum(first_rank * second_rank, axis=1) / np.maximum(denominator, 1e-15)

    target_draws = rng.integers(
        0,
        len(targets),
        size=(int(n_bootstrap), len(targets)),
    )
    bootstrap_indices = group_index_matrix[target_draws].reshape(int(n_bootstrap), -1)
    boot = rowwise_spearman(
        capture_centered[bootstrap_indices],
        mse_centered[bootstrap_indices],
    )

    # Draw a distinct permutation of method positions within every target and
    # Monte Carlo row.  Vectorizing the rank correlations keeps the full
    # 20,000-draw test inexpensive while preserving the exact stratification.
    methods_per_target = group_index_matrix.shape[1]
    permutation_order = rng.random(
        (int(n_permutations), len(targets), methods_per_target)
    ).argsort(axis=2)
    target_mse = mse_centered[group_index_matrix]
    permuted_target_mse = np.take_along_axis(
        np.broadcast_to(target_mse, permutation_order.shape),
        permutation_order,
        axis=2,
    )
    permuted_mse = permuted_target_mse.reshape(int(n_permutations), -1)
    capture_for_null = np.broadcast_to(capture_centered, permuted_mse.shape)
    null = rowwise_spearman(capture_for_null, permuted_mse)
    valid_null = null[np.isfinite(null)]
    if not len(valid_null):
        raise ValueError("all target-stratified permutation statistics are undefined")
    extreme = int(np.sum(np.abs(valid_null) >= abs(within_target)))
    return {
        "unit": "target-by-method means excluding exact and PCA oracles",
        "n_target_method_observations": int(len(selected)),
        "n_target_clusters": int(len(targets)),
        "methods": list(method_sets.iloc[0]),
        "pooled_spearman_rho_descriptive": pooled_descriptive,
        "within_target_centered_spearman_rho": within_target,
        "within_target_centered_target_bootstrap_ci95": [
            float(x) for x in np.nanquantile(boot, [0.025, 0.975])
        ],
        "within_target_centered_method_label_permutation_two_sided_p": float(
            (1 + extreme) / (1 + len(valid_null))
        ),
        "permutation_extreme_count": extreme,
        "valid_permutations": int(len(valid_null)),
        "note": (
            "Both variables are centered within target. The target-stratified null reassigns learning "
            "outcomes among method labels within each target and compares the centered statistic with "
            "the observed centered rho. The pooled rho is descriptive and is not tested."
        ),
    }


def run_target(
    data: dict[str, Any],
    all_segments: pd.DataFrame,
    seed: int,
    replicate: int,
    channels: int,
    test_fraction: float,
    steps: int,
    learning_rate: float,
    weight_decay: float,
    activation: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    root_id = int(data["root_id"])
    split_rng = stable_rng(seed, root_id, stream=1, replicate=replicate)
    train, test = grouped_split(data["stimulus_ids"], split_rng, test_fraction)
    x, _, _ = standardize(data["x"], train)
    y_matrix, _, _ = standardize(data["y"][:, None], train)
    y = y_matrix[:, 0]
    branches = remap_labels(data["contacts"]["major_branch"].to_numpy(dtype=int))
    segments = all_segments[all_segments["root_id"] == root_id].copy()
    dictionaries, metadata = anatomy_dictionaries(
        data["contacts"],
        segments,
        channels,
        stable_rng(seed, root_id, stream=2, replicate=replicate),
    )
    initial = initialize_model(
        x.shape[1],
        int(branches.max()) + 1,
        stable_rng(seed, root_id, stream=3, replicate=replicate),
    )
    exact, exact_history = train_model(
        x[train], y[train], branches, initial, None, steps, learning_rate, weight_decay, activation
    )
    _, _, train_delta = error_field(x[train], y[train], exact, branches, activation)
    _, _, test_delta = error_field(x[test], y[test], exact, branches, activation)
    _, _, vh = np.linalg.svd(train_delta, full_matrices=False)
    pca_channels = min(metadata["used_channels"], vh.shape[0])
    dictionaries["dense PCA oracle"] = vh[:pca_channels].T
    projections = {key: projector(value) for key, value in dictionaries.items()}
    baseline = float(np.mean((y[test] - np.mean(y[train])) ** 2))
    rows: list[dict[str, Any]] = []
    method_states: dict[str, ModelState] = {"exact backprop": exact}
    histories: dict[str, list[float]] = {"exact backprop": exact_history}
    for method in METHODS:
        if method == "exact backprop":
            state = exact
            projection = np.eye(x.shape[1])
        else:
            projection = projections[method]
            state, histories[method] = train_model(
                x[train],
                y[train],
                branches,
                initial,
                projection,
                steps,
                learning_rate,
                weight_decay,
                activation,
            )
            method_states[method] = state
        prediction_train, _, _ = error_field(x[train], y[train], state, branches, activation)
        prediction_test, _, _ = error_field(x[test], y[test], state, branches, activation)
        train_mse = float(np.mean((prediction_train - y[train]) ** 2))
        test_mse = float(np.mean((prediction_test - y[test]) ** 2))
        rows.append(
            {
                "target_root_id": root_id,
                "target_nucleus_id": int(data["nucleus_id"]),
                "replicate": int(replicate),
                "method": method,
                "n_trials": int(len(y)),
                "n_train_trials": int(np.sum(train)),
                "n_test_trials": int(np.sum(test)),
                "n_stimuli": int(len(np.unique(data["stimulus_ids"]))),
                "n_sites": int(x.shape[1]),
                "n_major_branches": int(branches.max() + 1),
                "requested_channels": int(channels),
                "used_channels": int(metadata["used_channels"]),
                "dictionary_rank": int(x.shape[1] if method == "exact backprop" else np.linalg.matrix_rank(dictionaries[method])),
                "heldout_credit_capture": float(1.0 if method == "exact backprop" else capture(test_delta, projection)),
                "train_mse": train_mse,
                "heldout_mse": test_mse,
                "heldout_normalized_mse": float(test_mse / max(baseline, 1e-12)),
                "final_training_objective": float(histories[method][-1]),
            }
        )
    metadata.update(
        {
            "target_root_id": root_id,
            "target_nucleus_id": int(data["nucleus_id"]),
            "replicate": int(replicate),
            "n_train_trials": int(np.sum(train)),
            "n_test_trials": int(np.sum(test)),
        }
    )
    return rows, metadata


def make_figure(frame: pd.DataFrame, outdir: Path) -> None:
    averaged = frame.groupby(["target_root_id", "method"], as_index=False).agg(
        heldout_credit_capture=("heldout_credit_capture", "mean"),
        heldout_normalized_mse=("heldout_normalized_mse", "mean"),
    )
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.8))
    for axis, metric, ylabel, title in [
        (axes[0], "heldout_credit_capture", "held-out credit capture", "A  Task-derived credit"),
        (axes[1], "heldout_normalized_mse", "held-out normalized MSE", "B  Local-learning outcome"),
    ]:
        for index, method in enumerate(METHODS):
            values = averaged.loc[averaged["method"] == method, metric].to_numpy(dtype=float)
            jitter = np.linspace(-0.08, 0.08, len(values))
            axis.scatter(np.full(len(values), index) + jitter, values, s=22, alpha=0.68, color=COLORS[method])
            axis.plot(index, np.mean(values), marker="D", ms=5, color="black")
        axis.set_xticks(
            range(len(METHODS)),
            ["exact", "PCA", "morph.", "random", "depth", "shuffle", "scalar"],
            rotation=35,
            ha="right",
        )
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        axis.spines[["top", "right"]].set_visible(False)
    nonexact = averaged[~averaged["method"].isin(["exact backprop", "dense PCA oracle"])].copy()
    for method in METHODS[2:]:
        selected = nonexact[nonexact["method"] == method]
        axes[2].scatter(
            selected["heldout_credit_capture"],
            selected["heldout_normalized_mse"],
            label=method,
            s=28,
            alpha=0.72,
            color=COLORS[method],
        )
    axes[2].set_xlabel("held-out credit capture")
    axes[2].set_ylabel("held-out normalized MSE")
    axes[2].set_title("C  Capture predicts utility?")
    axes[2].spines[["top", "right"]].set_visible(False)
    axes[2].legend(frameon=False, fontsize=7)
    fig.suptitle("Task-derived credit routing on real MICrONS connected-input cohorts", weight="bold", fontsize=11)
    fig.tight_layout()
    fig.savefig(outdir / "microns_task_derived_credit_learning.png", dpi=280, bbox_inches="tight")
    fig.savefig(outdir / "microns_task_derived_credit_learning.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--extract-root", type=Path, default=DEFAULT_EXTRACTS)
    parser.add_argument("--segments", type=Path, default=DEFAULT_SEGMENTS)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--replicates", type=int, default=10)
    parser.add_argument("--channels", type=int, default=4)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--learning-rate", type=float, default=0.015)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--activation", choices=["tanh", "linear"], default="tanh")
    parser.add_argument("--min-repeat-reliability", type=float, default=-np.inf)
    parser.add_argument("--manual-only", action="store_true")
    parser.add_argument("--reuse-runs", action="store_true")
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    errors: list[dict[str, str]] = []
    if args.reuse_runs:
        previous_summary_path = args.outdir / "summary.json"
        if previous_summary_path.is_file():
            previous_summary = json.loads(previous_summary_path.read_text(encoding="utf-8"))
            errors = list(previous_summary.get("errors", []))
        frame = pd.read_csv(args.outdir / "target_method_runs.csv")
    else:
        all_segments = pd.read_csv(args.segments)
        rows: list[dict[str, Any]] = []
        metadata: list[dict[str, Any]] = []
        for extract_dir in sorted(args.extract_root.glob("target*_automatic_conservative")):
            try:
                data = load_target(extract_dir)
                keep = (
                    data["contacts"]["repeat_reliability"].to_numpy(dtype=float)
                    >= float(args.min_repeat_reliability)
                )
                if args.manual_only:
                    keep &= data["contacts"]["manual_match"].astype(bool).to_numpy()
                data["x"] = data["x"][:, keep]
                data["contacts"] = data["contacts"].loc[keep].reset_index(drop=True)
                if data["x"].shape[1] < 2:
                    raise ValueError("fewer than two connected inputs pass the requested filter")
                for replicate in range(int(args.replicates)):
                    target_rows, target_metadata = run_target(
                        data,
                        all_segments,
                        args.seed,
                        replicate,
                        args.channels,
                        args.test_fraction,
                        args.steps,
                        args.learning_rate,
                        args.weight_decay,
                        args.activation,
                    )
                    rows.extend(target_rows)
                    metadata.append(target_metadata)
            except Exception as exc:
                errors.append({"extract_dir": str(extract_dir), "error": f"{type(exc).__name__}: {exc}"})
        frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError(f"no targets completed: {errors}")
    if not args.reuse_runs:
        frame.to_csv(args.outdir / "target_method_runs.csv", index=False)
        pd.DataFrame(metadata).to_json(args.outdir / "dictionary_metadata.jsonl", orient="records", lines=True)
    target_means = frame.groupby(["target_root_id", "target_nucleus_id", "method"], as_index=False).agg(
        heldout_credit_capture=("heldout_credit_capture", "mean"),
        heldout_credit_capture_sd=("heldout_credit_capture", "std"),
        heldout_normalized_mse=("heldout_normalized_mse", "mean"),
        heldout_normalized_mse_sd=("heldout_normalized_mse", "std"),
        train_mse=("train_mse", "mean"),
    )
    target_means.to_csv(args.outdir / "target_method_means.csv", index=False)
    capture_test = paired_summary(frame, "heldout_credit_capture", "shuffled ancestry", args.seed + 1)
    learning_test = paired_summary(frame, "heldout_normalized_mse", "shuffled ancestry", args.seed + 2)
    association = clustered_capture_utility_association(target_means, args.seed + 3)
    method_means = {
        method: {
            "heldout_credit_capture": float(group["heldout_credit_capture"].mean()),
            "heldout_normalized_mse": float(group["heldout_normalized_mse"].mean()),
        }
        for method, group in target_means.groupby("method")
    }
    summary = {
        "analysis": "task-derived credit capture and projected local learning",
        "status": "completed" if not errors else "completed_with_exclusions",
        "interpretation_level": "oracle channel-capacity and sufficiency test on a real response-prediction objective",
        "primary_budget_channels": int(args.channels),
        "n_targets": int(frame["target_root_id"].nunique()),
        "n_replicates_per_target": int(args.replicates),
        "n_target_method_runs": int(len(frame)),
        "split": "held-out stimulus identities",
        "method_target_means": method_means,
        "primary_credit_capture_contrast": capture_test,
        "primary_learning_contrast": learning_test,
        "capture_utility_association": association,
        "errors": errors,
        "parameters": {
            "seed": int(args.seed),
            "steps": int(args.steps),
            "learning_rate": float(args.learning_rate),
            "weight_decay": float(args.weight_decay),
            "test_fraction": float(args.test_fraction),
            "min_repeat_reliability": float(args.min_repeat_reliability)
            if np.isfinite(args.min_repeat_reliability)
            else None,
            "manual_only": bool(args.manual_only),
            "activation": args.activation,
        },
    }
    write_json(args.outdir / "summary.json", summary)
    lines = [
        "# Task-derived credit learning",
        "",
        f"Completed {summary['n_targets']} target neurons with {args.replicates} fixed-seed replicates each.",
        "The split holds out complete stimulus identities. Target neurons are the replication units.",
        "",
        "## Method means",
        "",
        "| method | credit capture | normalized held-out MSE |",
        "|---|---:|---:|",
    ]
    for method in METHODS:
        value = method_means[method]
        lines.append(f"| {method} | {value['heldout_credit_capture']:.4f} | {value['heldout_normalized_mse']:.4f} |")
    lines.extend(
        [
            "",
            "## Frozen primary contrasts",
            "",
            f"- Credit capture, morphology minus shuffled ancestry: {capture_test['mean_difference']:.4f}, "
            f"95% target bootstrap CI [{capture_test['target_bootstrap_ci95'][0]:.4f}, {capture_test['target_bootstrap_ci95'][1]:.4f}], "
            f"Wilcoxon p={capture_test['wilcoxon_two_sided_p']:.4g}.",
            f"- Normalized held-out MSE, morphology minus shuffled ancestry: {learning_test['mean_difference']:.4f}, "
            f"95% target bootstrap CI [{learning_test['target_bootstrap_ci95'][0]:.4f}, {learning_test['target_bootstrap_ci95'][1]:.4f}], "
            f"Wilcoxon p={learning_test['wilcoxon_two_sided_p']:.4g}.",
            f"- Within-target centered capture/utility Spearman rho="
            f"{association['within_target_centered_spearman_rho']:.3f}, target-cluster bootstrap "
            f"95% CI [{association['within_target_centered_target_bootstrap_ci95'][0]:.3f}, "
            f"{association['within_target_centered_target_bootstrap_ci95'][1]:.3f}], "
            f"target-stratified method-label permutation p="
            f"{association['within_target_centered_method_label_permutation_two_sided_p']:.4g}.",
            "",
            "## Boundary",
            "",
            "Optimal projection coefficients make this an upper-bound test. It does not yet show that a biological circuit can encode those coefficients.",
        ]
    )
    (args.outdir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    make_figure(frame, args.outdir)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
