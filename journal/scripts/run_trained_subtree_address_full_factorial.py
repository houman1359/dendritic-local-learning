#!/usr/bin/env python3
"""Run the frozen full bandwidth-by-representation subtree-address factorial.

Eight balanced contexts select eight task streams assigned to leaves of a
binary hierarchy.  The output error is scalar, while the exact update is
context/leaf specific.  K-route feedback fields interpolate between one shared
coordinate and exact leaf transport.  Point, flat and grouped controls pay for
the same explicit route fields; they are deliberately implementation-
equivalent controls, whereas the rewired tree breaks task/topology alignment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "configs"
    / "trained_subtree_address"
    / "full_factorial_confirmatory.json"
)
SOURCE = ROOT / "source_data" / "trained_subtree_address_full_factorial"
CANARY = ROOT / "analysis" / "trained_subtree_address_full_factorial_canary.json"


@dataclass(frozen=True)
class Dataset:
    x: np.ndarray  # trial x task stream x feature
    context: np.ndarray
    label: np.ndarray


@dataclass(frozen=True)
class Condition:
    architecture: str
    architecture_index: int
    family: str
    budget_k: int
    condition_id: str


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def array_digest(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values).view(np.uint8)).hexdigest()


def sigmoid(values: np.ndarray) -> np.ndarray:
    positive = values >= 0
    result = np.empty_like(values, dtype=float)
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exponent = np.exp(values[~positive])
    result[~positive] = exponent / (1.0 + exponent)
    return result


def tree_relation(left: int, right: int) -> str:
    if left // 2 == right // 2:
        return "sibling"
    if left // 4 == right // 4:
        return "same_half"
    return "opposite_half"


def make_dataset(
    rng: np.random.Generator,
    n: int,
    teachers: np.ndarray,
    cfg: dict,
    *,
    allowed_contexts: list[int] | None = None,
) -> Dataset:
    task = cfg["task"]
    contexts = (
        list(range(int(task["contexts"])))
        if allowed_contexts is None
        else [int(value) for value in allowed_contexts]
    )
    cells = 2 * len(contexts)
    if n % cells:
        raise ValueError(f"dataset size {n} is not divisible by {cells}")
    context = np.repeat(np.asarray(contexts, dtype=int), 2)
    label = np.tile(np.asarray([0, 1], dtype=int), len(contexts))
    context = np.tile(context, n // cells)
    label = np.tile(label, n // cells)
    order = rng.permutation(n)
    context = context[order]
    label = label[order]
    sign = 2.0 * label - 1.0
    n_streams, n_features = teachers.shape
    x = np.empty((n, n_streams, n_features), dtype=float)
    distractor = task["distractor_signal_by_tree_distance"]
    for stream in range(n_streams):
        amplitudes = np.empty(n, dtype=float)
        selected = context == stream
        amplitudes[selected] = float(task["selected_signal"])
        for active in np.unique(context[~selected]):
            relation = tree_relation(stream, int(active))
            amplitudes[(context == active) & ~selected] = -float(distractor[relation])
        x[:, stream, :] = (
            amplitudes[:, None] * sign[:, None] * teachers[stream][None, :]
            + float(task["feature_noise_sd"])
            * rng.normal(size=(n, n_features))
        )
    return Dataset(x=x, context=context, label=label.astype(float))


def task_audit(data: Dataset, n_contexts: int) -> dict[str, float | bool]:
    counts = np.zeros((n_contexts, 2), dtype=int)
    for context, label in zip(data.context, data.label.astype(int)):
        counts[int(context), int(label)] += 1
    return {
        "balanced_context_label_cells": bool(np.ptp(counts) == 0),
        "context_label_correlation": float(np.corrcoef(data.context, data.label)[0, 1]),
        "minimum_cell_count": int(counts.min()),
        "maximum_cell_count": int(counts.max()),
    }


def architecture_permutations(seed: int, architectures: list[str], n: int) -> np.ndarray:
    result = []
    for index, architecture in enumerate(architectures):
        if architecture == "degree_depth_matched_rewired_tree":
            rng = np.random.default_rng(seed + 820_000 + index)
            permutation = rng.permutation(n)
            if np.all(permutation == np.arange(n)):
                permutation = np.roll(permutation, 1)
        else:
            permutation = np.arange(n)
        result.append(permutation)
    return np.asarray(result, dtype=int)


def conditions(cfg: dict) -> list[Condition]:
    result: list[Condition] = []
    for architecture_index, architecture in enumerate(cfg["architectures"]):
        for budget in cfg["feedback_budgets"]:
            for family in cfg["feedback_families"]:
                result.append(
                    Condition(
                        architecture=architecture,
                        architecture_index=architecture_index,
                        family=family,
                        budget_k=int(budget),
                        condition_id=f"{architecture}__{family}__k{budget}",
                    )
                )
        for family, budget in (
            ("neuron_indexed_shared", 1),
            ("exact_compartment_transport", n_contexts(cfg)),
            ("backpropagation", n_contexts(cfg)),
        ):
            result.append(
                Condition(
                    architecture=architecture,
                    architecture_index=architecture_index,
                    family=family,
                    budget_k=int(budget),
                    condition_id=f"{architecture}__{family}__k{budget}",
                )
            )
    return result


def n_contexts(cfg: dict) -> int:
    return int(cfg["task"]["contexts"])


def normalized_rows(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return np.divide(values, norms, out=np.zeros_like(values), where=norms > 1e-15)


def exact_routes(permutation: np.ndarray) -> np.ndarray:
    n = len(permutation)
    routes = np.zeros((n, n), dtype=float)
    routes[np.arange(n), permutation] = 1.0
    return routes


def grouped_routes(permutation: np.ndarray, budget: int, mode: str, rng: np.random.Generator) -> np.ndarray:
    n = len(permutation)
    if n % budget:
        raise ValueError("feedback budget must divide the number of leaves")
    block_group = np.arange(n) // (n // budget)
    if mode == "depth_interleaved_bins":
        block_group = np.arange(n) % budget
    elif mode == "random_sparse_matched":
        shuffled = rng.permutation(n)
        block_group = np.empty(n, dtype=int)
        block_group[shuffled] = np.repeat(np.arange(budget), n // budget)
    routes = np.zeros((n, n), dtype=float)
    if mode == "within_neuron_route_derangement" and budget > 1:
        assigned = np.roll(np.arange(budget), 1)
    else:
        assigned = np.arange(budget)
    for context in range(n):
        active_block = int(permutation[context])
        group = int(block_group[active_block])
        target_group = int(assigned[group])
        routes[context, block_group == target_group] = 1.0
    return normalized_rows(routes)


def random_rank_routes(
    permutation: np.ndarray, budget: int, rng: np.random.Generator
) -> np.ndarray:
    n = len(permutation)
    basis, _ = np.linalg.qr(rng.normal(size=(n, budget)))
    projection = basis[:, :budget] @ basis[:, :budget].T
    return normalized_rows(exact_routes(permutation) @ projection)


def learned_rank_routes(
    permutation: np.ndarray,
    budget: int,
    data: Dataset,
    initial_task_weights: np.ndarray,
) -> np.ndarray:
    exact = exact_routes(permutation)
    logits = np.sum(initial_task_weights[data.context] * data.x[np.arange(len(data.label)), data.context], axis=1)
    delta = sigmoid(logits) - data.label
    coefficients = delta[:, None] * exact[data.context]
    covariance = coefficients.T @ coefficients / len(coefficients)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    basis = eigenvectors[:, np.argsort(eigenvalues)[::-1][:budget]]
    projection = basis @ basis.T
    return normalized_rows(exact @ projection)


def route_matrices(
    cfg: dict,
    seed: int,
    condition_list: list[Condition],
    permutations: np.ndarray,
    train_data: Dataset,
    initial_task_weights: np.ndarray,
) -> np.ndarray:
    matrices = []
    family_index = {
        family: index
        for index, family in enumerate(
            cfg["feedback_families"] + cfg["reference_conditions"]
        )
    }
    for condition in condition_list:
        permutation = permutations[condition.architecture_index]
        # Stochastic controls are paired across implementation-equivalent
        # representations. Architecture enters only through its permutation.
        rng = np.random.default_rng(
            seed
            + 930_000
            + 1001 * family_index[condition.family]
            + 17 * condition.budget_k
        )
        if condition.family in {"exact_compartment_transport", "backpropagation"}:
            routes = exact_routes(permutation)
        elif condition.family == "neuron_indexed_shared":
            routes = normalized_rows(np.ones((n_contexts(cfg), n_contexts(cfg))))
        elif condition.family in {
            "correct_ancestry_subtrees",
            "within_neuron_route_derangement",
            "depth_interleaved_bins",
            "random_sparse_matched",
        }:
            routes = grouped_routes(
                permutation, condition.budget_k, condition.family, rng
            )
        elif condition.family == "random_rank_k":
            routes = random_rank_routes(permutation, condition.budget_k, rng)
        elif condition.family == "learned_rank_k_upper_bound":
            routes = learned_rank_routes(
                permutation,
                condition.budget_k,
                train_data,
                initial_task_weights,
            )
        else:
            raise ValueError(condition.family)
        matrices.append(routes)
    return np.asarray(matrices, dtype=float)


def initial_weights(
    initial_task_weights: np.ndarray,
    condition_list: list[Condition],
    permutations: np.ndarray,
) -> np.ndarray:
    values = []
    for condition in condition_list:
        permutation = permutations[condition.architecture_index]
        block_weights = np.empty_like(initial_task_weights)
        block_weights[permutation] = initial_task_weights
        values.append(block_weights)
    return np.asarray(values, dtype=float)


def block_inputs(data: Dataset, permutation: np.ndarray) -> np.ndarray:
    inverse = np.argsort(permutation)
    return data.x[:, inverse, :]


def logits_all(
    weights: np.ndarray,
    data: Dataset,
    condition_list: list[Condition],
    permutations: np.ndarray,
) -> np.ndarray:
    model_indices = np.arange(len(condition_list))[:, None]
    active_blocks = np.stack(
        [permutations[c.architecture_index][data.context] for c in condition_list],
        axis=0,
    )
    selected_weights = weights[model_indices, active_blocks]
    selected_inputs = data.x[np.arange(len(data.label)), data.context]
    return np.einsum("mbf,bf->mb", selected_weights, selected_inputs, optimize=True)


def gradients_all(
    weights: np.ndarray,
    data: Dataset,
    routes: np.ndarray,
    condition_list: list[Condition],
    permutations: np.ndarray,
    context_scale: float,
) -> np.ndarray:
    delta = sigmoid(logits_all(weights, data, condition_list, permutations)) - data.label[None, :]
    route_batch = routes[:, data.context, :]
    coefficients = delta[:, :, None] * route_batch
    gradients = np.empty_like(weights)
    for architecture_index in range(len(permutations)):
        selected = np.asarray(
            [i for i, c in enumerate(condition_list) if c.architecture_index == architecture_index],
            dtype=int,
        )
        x_blocks = block_inputs(data, permutations[architecture_index])
        gradients[selected] = (
            context_scale
            * np.einsum(
                "mbl,blf->mlf",
                coefficients[selected],
                x_blocks,
                optimize=True,
            )
            / len(data.label)
        )
    return gradients


def train_all(
    initial: np.ndarray,
    data: Dataset,
    routes: np.ndarray,
    condition_list: list[Condition],
    permutations: np.ndarray,
    cfg: dict,
    *,
    epochs: int,
) -> np.ndarray:
    weights = initial.copy()
    for _ in range(int(epochs)):
        weights -= float(cfg["training"]["learning_rate"]) * gradients_all(
            weights,
            data,
            routes,
            condition_list,
            permutations,
            float(cfg["training"]["gradient_context_rescaling"]),
        )
    return weights


def loss_accuracy(logits: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    losses = np.mean(np.logaddexp(0.0, logits) - labels[None, :] * logits, axis=1)
    accuracy = np.mean((logits >= 0) == (labels[None, :] >= 0.5), axis=1)
    return losses, accuracy


def gradient_geometry(
    data: Dataset,
    routes: np.ndarray,
    condition_list: list[Condition],
    permutations: np.ndarray,
    initial: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    logits = logits_all(initial, data, condition_list, permutations)
    delta = sigmoid(logits) - data.label[None, :]
    cosines = np.empty(len(condition_list), dtype=float)
    captures = np.empty(len(condition_list), dtype=float)
    for index, condition in enumerate(condition_list):
        x_blocks = block_inputs(data, permutations[condition.architecture_index])
        energy = delta[index, :, None] ** 2 * np.sum(x_blocks * x_blocks, axis=2)
        exact = exact_routes(permutations[condition.architecture_index])[data.context]
        approximate = routes[index, data.context]
        dot = float(np.sum(energy * exact * approximate))
        exact_norm = float(np.sum(energy * exact * exact))
        approximate_norm = float(np.sum(energy * approximate * approximate))
        cosine = dot / max(np.sqrt(exact_norm * approximate_norm), 1e-30)
        cosines[index] = cosine
        captures[index] = cosine * cosine
    return cosines, captures


def one_step_progress(
    data: Dataset,
    routes: np.ndarray,
    condition_list: list[Condition],
    permutations: np.ndarray,
    initial: np.ndarray,
    cfg: dict,
) -> np.ndarray:
    base_loss, _ = loss_accuracy(
        logits_all(initial, data, condition_list, permutations), data.label
    )
    gradients = gradients_all(
        initial,
        data,
        routes,
        condition_list,
        permutations,
        float(cfg["training"]["gradient_context_rescaling"]),
    )
    exact_by_architecture = {}
    for architecture_index in range(len(permutations)):
        reference = next(
            i
            for i, condition in enumerate(condition_list)
            if condition.architecture_index == architecture_index
            and condition.family == "exact_compartment_transport"
        )
        exact_by_architecture[architecture_index] = gradients[reference]
    normalized = gradients.copy()
    for index, condition in enumerate(condition_list):
        exact = exact_by_architecture[condition.architecture_index]
        normalized[index] *= np.linalg.norm(exact) / max(np.linalg.norm(gradients[index]), 1e-30)
    step = 0.25
    exact_step = initial.copy()
    for index, condition in enumerate(condition_list):
        exact_step[index] -= step * exact_by_architecture[condition.architecture_index]
    approximate_step = initial - step * normalized
    exact_loss, _ = loss_accuracy(
        logits_all(exact_step, data, condition_list, permutations), data.label
    )
    approximate_loss, _ = loss_accuracy(
        logits_all(approximate_step, data, condition_list, permutations), data.label
    )
    denominator = base_loss - exact_loss
    return np.divide(
        base_loss - approximate_loss,
        denominator,
        out=np.full_like(base_loss, np.nan),
        where=denominator > 1e-15,
    )


def run_seed(seed: int, cfg: dict) -> tuple[list[dict], list[dict]]:
    rng = np.random.default_rng(seed)
    n = n_contexts(cfg)
    d = int(cfg["task"]["features_per_stream"])
    teachers = rng.normal(size=(n, d))
    teachers /= np.linalg.norm(teachers, axis=1, keepdims=True)
    train_data = make_dataset(
        rng, int(cfg["task"]["train_examples"]), teachers, cfg
    )
    test_data = make_dataset(
        rng, int(cfg["task"]["test_examples"]), teachers, cfg
    )
    switch_data = make_dataset(
        rng,
        int(cfg["task"]["switch_examples"]),
        teachers,
        cfg,
        allowed_contexts=cfg["task"]["switch_contexts"],
    )
    retained_mask = np.isin(test_data.context, cfg["task"]["retained_contexts"])
    retained_data = Dataset(
        x=test_data.x[retained_mask],
        context=test_data.context[retained_mask],
        label=test_data.label[retained_mask],
    )
    condition_list = conditions(cfg)
    permutations = architecture_permutations(seed, cfg["architectures"], n)
    initial_task = rng.normal(
        scale=float(cfg["training"]["initialization_sd"]), size=(n, d)
    )
    initial = initial_weights(initial_task, condition_list, permutations)
    routes = route_matrices(
        cfg,
        seed,
        condition_list,
        permutations,
        train_data,
        initial_task,
    )
    cosine, capture = gradient_geometry(
        test_data, routes, condition_list, permutations, initial
    )
    progress = one_step_progress(
        test_data, routes, condition_list, permutations, initial, cfg
    )
    fitted = train_all(
        initial,
        train_data,
        routes,
        condition_list,
        permutations,
        cfg,
        epochs=int(cfg["training"]["epochs"]),
    )
    train_loss, train_accuracy = loss_accuracy(
        logits_all(fitted, train_data, condition_list, permutations), train_data.label
    )
    test_loss, test_accuracy = loss_accuracy(
        logits_all(fitted, test_data, condition_list, permutations), test_data.label
    )
    retained_loss, retained_accuracy = loss_accuracy(
        logits_all(fitted, retained_data, condition_list, permutations), retained_data.label
    )
    adapted = train_all(
        fitted,
        switch_data,
        routes,
        condition_list,
        permutations,
        cfg,
        epochs=int(cfg["training"]["switch_epochs"]),
    )
    _, retained_after = loss_accuracy(
        logits_all(adapted, retained_data, condition_list, permutations), retained_data.label
    )
    train_audit = task_audit(train_data, n)
    test_audit = task_audit(test_data, n)
    rows: list[dict] = []
    ledgers: list[dict] = []
    for index, condition in enumerate(condition_list):
        route = routes[index]
        rows.append(
            {
                "seed": seed,
                "condition_id": condition.condition_id,
                "architecture": condition.architecture,
                "feedback_family": condition.family,
                "budget_k": condition.budget_k,
                "train_loss": float(train_loss[index]),
                "train_accuracy": float(train_accuracy[index]),
                "heldout_loss": float(test_loss[index]),
                "heldout_accuracy": float(test_accuracy[index]),
                "retained_context_loss_before_switch": float(retained_loss[index]),
                "retained_context_accuracy_before_switch": float(retained_accuracy[index]),
                "retained_context_accuracy_after_switch": float(retained_after[index]),
                "retained_context_forgetting": float(
                    retained_accuracy[index] - retained_after[index]
                ),
                "initial_gradient_cosine": float(cosine[index]),
                "initial_gradient_capture": float(capture[index]),
                "norm_matched_one_step_progress": float(progress[index]),
            }
        )
        ledgers.append(
            {
                "seed": seed,
                "condition_id": condition.condition_id,
                "architecture": condition.architecture,
                "feedback_family": condition.family,
                "budget_k": condition.budget_k,
                "forward_parameters": int(n * d),
                "active_input_contacts": int(n * d),
                "forward_nonlinearities": 1,
                "feedback_channels": int(condition.budget_k),
                "feedback_nonzeros": int(np.count_nonzero(np.abs(route) > 1e-14)),
                "route_sha256": array_digest(route),
                "architecture_permutation": json.dumps(
                    permutations[condition.architecture_index].tolist()
                ),
                "implementation_equivalent_control": bool(
                    condition.architecture
                    in {
                        "point_neuron_explicit_gating",
                        "flat_compartment",
                        "grouped_point_subunits",
                    }
                ),
                "train_balanced": train_audit["balanced_context_label_cells"],
                "test_balanced": test_audit["balanced_context_label_cells"],
                "train_context_label_correlation": train_audit[
                    "context_label_correlation"
                ],
                "test_context_label_correlation": test_audit[
                    "context_label_correlation"
                ],
                "fallback_count": 0,
                "scalar_replacement_rate": 0.0,
            }
        )
    return rows, ledgers


def bootstrap(values: np.ndarray, seed: int, draws: int) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = rng.choice(values, size=(int(draws), len(values)), replace=True).mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(values.mean()), float(low), float(high)


def summaries(outcomes: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics = [
        "heldout_accuracy",
        "heldout_loss",
        "retained_context_forgetting",
        "initial_gradient_capture",
        "initial_gradient_cosine",
        "norm_matched_one_step_progress",
    ]
    summary_rows: list[dict] = []
    for group_index, (keys, part) in enumerate(
        outcomes.groupby(["architecture", "feedback_family", "budget_k"], sort=True)
    ):
        row = dict(zip(["architecture", "feedback_family", "budget_k"], keys))
        row["n_seeds"] = int(part.seed.nunique())
        for metric_index, metric in enumerate(metrics):
            mean, low, high = bootstrap(
                part[metric].to_numpy(float),
                120_000 + 20 * group_index + metric_index,
                int(cfg["analysis"]["bootstrap_draws"]),
            )
            row[f"mean_{metric}"] = mean
            row[f"ci95_low_{metric}"] = low
            row[f"ci95_high_{metric}"] = high
        summary_rows.append(row)

    contrast_rows: list[dict] = []
    endpoints = ["heldout_accuracy", "heldout_loss", "retained_context_forgetting"]
    controls = [
        "within_neuron_route_derangement",
        "depth_interleaved_bins",
        "random_sparse_matched",
        "random_rank_k",
    ]
    for architecture_index, architecture in enumerate(cfg["architectures"]):
        arch = outcomes[outcomes.architecture.eq(architecture)]
        for budget in cfg["feedback_budgets"]:
            correct = arch[
                arch.feedback_family.eq("correct_ancestry_subtrees")
                & arch.budget_k.eq(int(budget))
            ]
            control_wide = arch[
                arch.feedback_family.isin(controls) & arch.budget_k.eq(int(budget))
            ].pivot(index="seed", columns="feedback_family", values="heldout_accuracy")
            oracle_best = control_wide.max(axis=1)
            comparisons: list[tuple[str, pd.Series]] = []
            for control in controls:
                values = arch[
                    arch.feedback_family.eq(control) & arch.budget_k.eq(int(budget))
                ].set_index("seed")["heldout_accuracy"]
                comparisons.append((control, values))
            comparisons.append(("best_matched_nonanatomical_oracle", oracle_best))
            for endpoint_index, endpoint in enumerate(endpoints):
                left = correct.set_index("seed")[endpoint]
                endpoint_comparisons = comparisons
                if endpoint != "heldout_accuracy":
                    endpoint_comparisons = [
                        (
                            control,
                            arch[
                                arch.feedback_family.eq(control)
                                & arch.budget_k.eq(int(budget))
                            ].set_index("seed")[endpoint],
                        )
                        for control in controls
                    ]
                for control_index, (control, right) in enumerate(endpoint_comparisons):
                    joined = pd.concat([left.rename("left"), right.rename("right")], axis=1).dropna()
                    values = (joined.left - joined.right).to_numpy(float)
                    mean, low, high = bootstrap(
                        values,
                        140_000
                        + 1000 * architecture_index
                        + 100 * int(budget)
                        + 10 * endpoint_index
                        + control_index,
                        int(cfg["analysis"]["bootstrap_draws"]),
                    )
                    pvalue = 1.0 if np.allclose(values, 0) else float(
                        wilcoxon(values, alternative="two-sided", zero_method="wilcox").pvalue
                    )
                    contrast_rows.append(
                        {
                            "architecture": architecture,
                            "budget_k": int(budget),
                            "contrast": f"correct - {control}",
                            "endpoint": endpoint,
                            "n_pairs": len(values),
                            "mean_difference": mean,
                            "ci95_low": low,
                            "ci95_high": high,
                            "wins": int(np.sum(values > 0)),
                            "ties": int(np.sum(np.isclose(values, 0))),
                            "wilcoxon_p_two_sided": pvalue,
                        }
                    )
    return pd.DataFrame(summary_rows), pd.DataFrame(contrast_rows)


def canary_gates(outcomes: pd.DataFrame, ledgers: pd.DataFrame, cfg: dict) -> dict:
    exact = outcomes.feedback_family.eq("exact_compartment_transport")
    backprop = outcomes.feedback_family.eq("backpropagation")
    exact_rows = outcomes[exact].set_index(["seed", "architecture"])
    bp_rows = outcomes[backprop].set_index(["seed", "architecture"])
    joined = exact_rows[["heldout_accuracy", "heldout_loss"]].join(
        bp_rows[["heldout_accuracy", "heldout_loss"]],
        lsuffix="_exact",
        rsuffix="_bp",
    )
    maximum_endpoint_difference = float(
        np.max(
            np.abs(
                joined[
                    [
                        "heldout_accuracy_exact",
                        "heldout_loss_exact",
                    ]
                ].to_numpy()
                - joined[["heldout_accuracy_bp", "heldout_loss_bp"]].to_numpy()
            )
        )
    )
    shared_capture = float(
        outcomes[outcomes.feedback_family.eq("neuron_indexed_shared")][
            "initial_gradient_capture"
        ].max()
    )
    exact_capture = float(outcomes[exact]["initial_gradient_capture"].min())
    equivalent = outcomes[
        outcomes.architecture.isin(
            [
                "dendritic_tree",
                "point_neuron_explicit_gating",
                "flat_compartment",
                "grouped_point_subunits",
            ]
        )
    ].pivot_table(
        index=["seed", "feedback_family", "budget_k"],
        columns="architecture",
        values=["heldout_accuracy", "heldout_loss"],
    )
    equivalence_difference = float(
        max(
            (
                equivalent[endpoint].max(axis=1)
                - equivalent[endpoint].min(axis=1)
            ).max()
            for endpoint in ("heldout_accuracy", "heldout_loss")
        )
    )
    gate = cfg["gates"]
    payload = {
        "n_models": int(len(outcomes)),
        "all_finite": bool(
            np.isfinite(outcomes.select_dtypes(include=[np.number])).all().all()
        ),
        "all_balanced": bool(ledgers.train_balanced.all() and ledgers.test_balanced.all()),
        "maximum_absolute_context_label_correlation": float(
            ledgers[
                ["train_context_label_correlation", "test_context_label_correlation"]
            ].abs().to_numpy().max()
        ),
        "maximum_neuron_shared_capture": shared_capture,
        "minimum_exact_capture": exact_capture,
        "maximum_exact_bp_endpoint_difference": maximum_endpoint_difference,
        "maximum_implementation_equivalence_endpoint_difference": equivalence_difference,
        "maximum_fallback_count": int(ledgers.fallback_count.max()),
    }
    payload["passed"] = bool(
        payload["all_finite"]
        and payload["all_balanced"]
        and payload["maximum_absolute_context_label_correlation"]
        <= float(gate["maximum_context_label_correlation"])
        and payload["maximum_neuron_shared_capture"]
        <= float(gate["maximum_neuron_shared_capture"])
        and payload["minimum_exact_capture"] >= float(gate["minimum_exact_capture"])
        and payload["maximum_exact_bp_endpoint_difference"]
        <= float(gate["maximum_exact_bp_endpoint_difference"])
        and payload["maximum_implementation_equivalence_endpoint_difference"]
        <= float(gate["maximum_implementation_equivalence_endpoint_difference"])
        and payload["maximum_fallback_count"] <= int(gate["maximum_fallback_count"])
    )
    return payload


def execute(seeds: list[int], cfg: dict, workers: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    if workers <= 1 or len(seeds) == 1:
        results = [run_seed(seed, cfg) for seed in seeds]
    else:
        with ProcessPoolExecutor(max_workers=min(workers, len(seeds))) as pool:
            results = list(pool.map(run_seed, seeds, [cfg] * len(seeds)))
    outcome_rows = [row for outcome, _ in results for row in outcome]
    ledger_rows = [row for _, ledger in results for row in ledger]
    return pd.DataFrame(outcome_rows), pd.DataFrame(ledger_rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["canary", "confirmatory"], required=True)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    if args.phase == "canary":
        outcomes, ledgers = execute([int(cfg["canary_seed"])], cfg, args.workers)
        gates = canary_gates(outcomes, ledgers, cfg)
        payload = {
            "phase": "canary",
            "config_sha256": digest(CONFIG),
            "script_sha256": digest(Path(__file__).resolve()),
            "artifact_and_numerical_gates": gates,
            "outcomes_are_not_confirmatory": True,
        }
        CANARY.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(payload, indent=2))
        if not gates["passed"]:
            raise SystemExit("canary failed")
        return

    if not CANARY.is_file():
        raise SystemExit("passing canary record required")
    canary = json.loads(CANARY.read_text(encoding="utf-8"))
    if not canary["artifact_and_numerical_gates"]["passed"]:
        raise SystemExit("canary did not pass")
    if canary["config_sha256"] != digest(CONFIG) or canary["script_sha256"] != digest(Path(__file__).resolve()):
        raise SystemExit("code or configuration changed after the canary")
    seeds = [int(value) for value in cfg["confirmatory_seeds"]]
    outcomes, ledgers = execute(seeds, cfg, args.workers)
    expected = len(seeds) * len(conditions(cfg))
    if len(outcomes) != expected or outcomes.duplicated(["seed", "condition_id"]).any():
        raise SystemExit(f"incomplete factorial: expected {expected}, found {len(outcomes)}")
    if not np.isfinite(outcomes.select_dtypes(include=[np.number])).all().all():
        raise SystemExit("non-finite confirmatory outcome")
    SOURCE.mkdir(parents=True, exist_ok=True)
    condition_summary, contrasts = summaries(outcomes, cfg)
    outcomes.to_csv(SOURCE / "seed_outcomes.csv", index=False, float_format="%.10g")
    ledgers.to_csv(SOURCE / "mechanism_ledger.csv", index=False, float_format="%.10g")
    condition_summary.to_csv(SOURCE / "condition_summary.csv", index=False, float_format="%.10g")
    contrasts.to_csv(SOURCE / "paired_contrasts.csv", index=False, float_format="%.10g")
    equivalence = (
        outcomes[
            outcomes.architecture.isin(
                [
                    "dendritic_tree",
                    "point_neuron_explicit_gating",
                    "flat_compartment",
                    "grouped_point_subunits",
                ]
            )
        ]
        .pivot_table(
            index=["seed", "feedback_family", "budget_k"],
            columns="architecture",
            values="heldout_accuracy",
        )
    )
    maximum_equivalence_difference = float(
        (equivalence.max(axis=1) - equivalence.min(axis=1)).max()
    )
    metadata = {
        "study": cfg["study"],
        "status": "complete_confirmatory",
        "scope_boundary": cfg["scope_boundary"],
        "config_sha256": digest(CONFIG),
        "script_sha256": digest(Path(__file__).resolve()),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "n_seeds": len(seeds),
        "n_conditions_per_seed": len(conditions(cfg)),
        "n_model_fits": len(outcomes),
        "n_architectures": int(outcomes.architecture.nunique()),
        "feedback_budgets": cfg["feedback_budgets"],
        "all_balanced": bool(ledgers.train_balanced.all() and ledgers.test_balanced.all()),
        "all_fallback_counts_zero": bool((ledgers.fallback_count == 0).all()),
        "maximum_implementation_equivalence_accuracy_difference": maximum_equivalence_difference,
    }
    (SOURCE / "summary.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
