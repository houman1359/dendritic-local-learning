#!/usr/bin/env python3
"""Run the frozen phase-1 trained within-neuron address experiment.

The task presents two active sibling streams. Context selects the stream that
controls the somatic output, while the other stream carries an anti-correlated
distractor. Thus the somatic error can be the same while the exact update must
be routed to different sibling subtrees. The primary comparison changes only
the within-neuron route map: correct versus fixed derangement at K=2.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from journal_style import (
    COLORS,
    ERR_CAPSIZE,
    FIG_W,
    LW_DATA,
    LW_ERR,
    PT_LEGEND,
    PT_SMALL,
    apply_neurips_style,
    audit_layout,
    audit_text_over_data,
    clean_legend,
    panel_title,
    style_axis,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "trained_subtree_address" / "phase1_confirmatory.json"
SOURCE = ROOT / "source_data" / "trained_subtree_address"
FIGURES = ROOT / "figures" / "generated"
CANARY = ROOT / "analysis" / "trained_subtree_address_phase1_canary.json"

CONDITION_ORDER = [
    "neuron_shared_k1",
    "depth_shared_k1",
    "correct_subtree_k2",
    "within_neuron_deranged_k2",
    "random_dense_rank2",
    "exact_transport",
    "backpropagation",
    "gated_point_emulation",
]

LABELS = {
    "neuron_shared_k1": "neuron shared",
    "depth_shared_k1": "depth shared",
    "correct_subtree_k2": "correct subtree",
    "within_neuron_deranged_k2": "deranged subtree",
    "random_dense_rank2": "random rank-2",
    "exact_transport": "exact transport",
    "backpropagation": "backpropagation",
    "gated_point_emulation": "gated point emulation",
}

COLORS_BY_CONDITION = {
    "neuron_shared_k1": COLORS["per_soma"],
    "depth_shared_k1": COLORS["scalar"],
    "correct_subtree_k2": COLORS["shunting"],
    "within_neuron_deranged_k2": COLORS["mute"],
    "random_dense_rank2": COLORS["additive"],
    "exact_transport": COLORS["oracle"],
    "backpropagation": COLORS["bp"],
    "gated_point_emulation": COLORS["inh"],
}


@dataclass(frozen=True)
class Dataset:
    x_a: np.ndarray
    x_b: np.ndarray
    context: np.ndarray
    label: np.ndarray


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sigmoid(values: np.ndarray) -> np.ndarray:
    positive = values >= 0
    result = np.empty_like(values, dtype=float)
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exp_values = np.exp(values[~positive])
    result[~positive] = exp_values / (1.0 + exp_values)
    return result


def make_dataset(
    rng: np.random.Generator,
    n: int,
    teacher_a: np.ndarray,
    teacher_b: np.ndarray,
    cfg: dict,
    *,
    context_only: int | None = None,
) -> Dataset:
    if context_only is None:
        if n % 4:
            raise ValueError("Balanced datasets require n divisible by four")
        context = np.tile(np.array([0, 0, 1, 1], dtype=int), n // 4)
        label = np.tile(np.array([0, 1, 0, 1], dtype=int), n // 4)
    else:
        if n % 2:
            raise ValueError("Single-context datasets require an even n")
        context = np.full(n, int(context_only), dtype=int)
        label = np.tile(np.array([0, 1], dtype=int), n // 2)
    order = rng.permutation(n)
    context = context[order]
    label = label[order]
    sign_y = 2.0 * label - 1.0
    selected = float(cfg["task"]["selected_signal"])
    distractor = float(cfg["task"]["distractor_signal"])
    noise = float(cfg["task"]["feature_noise_sd"])
    signal_a = np.where(context == 0, selected, -distractor) * sign_y
    signal_b = np.where(context == 1, selected, -distractor) * sign_y
    x_a = signal_a[:, None] * teacher_a[None, :] + noise * rng.normal(
        size=(n, teacher_a.size)
    )
    x_b = signal_b[:, None] * teacher_b[None, :] + noise * rng.normal(
        size=(n, teacher_b.size)
    )
    return Dataset(x_a=x_a, x_b=x_b, context=context, label=label.astype(float))


def subset(data: Dataset, indices: np.ndarray) -> Dataset:
    return Dataset(
        x_a=data.x_a[indices],
        x_b=data.x_b[indices],
        context=data.context[indices],
        label=data.label[indices],
    )


def logits(weights: np.ndarray, data: Dataset) -> np.ndarray:
    branch_a = data.x_a @ weights[0]
    branch_b = data.x_b @ weights[1]
    return np.where(data.context == 0, branch_a, branch_b)


def loss_accuracy(weights: np.ndarray, data: Dataset) -> tuple[float, float]:
    z = logits(weights, data)
    loss = np.mean(np.logaddexp(0.0, z) - data.label * z)
    accuracy = np.mean((z >= 0) == (data.label >= 0.5))
    return float(loss), float(accuracy)


def route_matrix(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed + 71_331)
    theta = rng.uniform(np.pi / 4.0, 3.0 * np.pi / 4.0)
    return np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]],
        dtype=float,
    )


def coefficients(
    condition: str,
    data: Dataset,
    delta: np.ndarray,
    random_route: np.ndarray,
) -> np.ndarray:
    exact = np.column_stack([data.context == 0, data.context == 1]).astype(float)
    if condition in {
        "correct_subtree_k2",
        "exact_transport",
        "backpropagation",
        "gated_point_emulation",
    }:
        route = exact
    elif condition in {"neuron_shared_k1", "depth_shared_k1"}:
        route = np.ones_like(exact)
    elif condition == "within_neuron_deranged_k2":
        route = exact[:, ::-1]
    elif condition == "random_dense_rank2":
        route = exact @ random_route.T
    else:
        raise ValueError(condition)
    return delta[:, None] * route


def gradient(
    weights: np.ndarray,
    data: Dataset,
    condition: str,
    random_route: np.ndarray,
) -> np.ndarray:
    delta = sigmoid(logits(weights, data)) - data.label
    coeff = coefficients(condition, data, delta, random_route)
    return np.stack(
        [
            np.mean(coeff[:, 0, None] * data.x_a, axis=0),
            np.mean(coeff[:, 1, None] * data.x_b, axis=0),
        ]
    )


def per_example_gradients(
    weights: np.ndarray,
    data: Dataset,
    condition: str,
    random_route: np.ndarray,
) -> np.ndarray:
    delta = sigmoid(logits(weights, data)) - data.label
    coeff = coefficients(condition, data, delta, random_route)
    return np.concatenate(
        [coeff[:, 0, None] * data.x_a, coeff[:, 1, None] * data.x_b], axis=1
    )


def gradient_geometry(
    weights: np.ndarray,
    data: Dataset,
    condition: str,
    random_route: np.ndarray,
) -> tuple[float, float]:
    exact = per_example_gradients(weights, data, "exact_transport", random_route)
    approx = per_example_gradients(weights, data, condition, random_route)
    dot = float(np.sum(exact * approx))
    exact_norm_sq = float(np.sum(exact * exact))
    approx_norm_sq = float(np.sum(approx * approx))
    if exact_norm_sq <= 0 or approx_norm_sq <= 0:
        return 0.0, 0.0
    cosine = dot / np.sqrt(exact_norm_sq * approx_norm_sq)
    capture = dot * dot / (exact_norm_sq * approx_norm_sq)
    return float(cosine), float(capture)


def one_step_progress(
    weights: np.ndarray,
    data: Dataset,
    condition: str,
    random_route: np.ndarray,
    step: float = 0.25,
) -> float:
    base_loss, _ = loss_accuracy(weights, data)
    exact = gradient(weights, data, "exact_transport", random_route)
    approximate = gradient(weights, data, condition, random_route)
    exact_norm = float(np.linalg.norm(exact))
    approx_norm = float(np.linalg.norm(approximate))
    if approx_norm <= 1e-15 or exact_norm <= 1e-15:
        return 0.0
    approximate = approximate * (exact_norm / approx_norm)
    exact_loss, _ = loss_accuracy(weights - step * exact, data)
    approximate_loss, _ = loss_accuracy(weights - step * approximate, data)
    denominator = base_loss - exact_loss
    if denominator <= 1e-15:
        return float("nan")
    return float((base_loss - approximate_loss) / denominator)


def train(
    initial: np.ndarray,
    data: Dataset,
    condition: str,
    random_route: np.ndarray,
    batches: list[np.ndarray],
    learning_rate: float,
) -> np.ndarray:
    weights = initial.copy()
    for indices in batches:
        weights -= learning_rate * gradient(
            weights, subset(data, indices), condition, random_route
        )
    return weights


def batch_schedule(
    seed: int, n: int, epochs: int, batch_size: int, *, salt: int
) -> list[np.ndarray]:
    rng = np.random.default_rng(seed + salt)
    batches: list[np.ndarray] = []
    for _ in range(epochs):
        order = rng.permutation(n)
        batches.extend(
            order[start : start + batch_size] for start in range(0, n, batch_size)
        )
    return batches


def task_audit(data: Dataset) -> dict[str, float | bool]:
    table = pd.crosstab(data.context, data.label)
    balanced = table.shape == (2, 2) and table.to_numpy().ptp() == 0
    context_label_corr = float(np.corrcoef(data.context, data.label)[0, 1])
    return {
        "balanced_context_label_cells": bool(balanced),
        "context_label_correlation": context_label_corr,
    }


def run_seed(seed: int, cfg: dict) -> tuple[list[dict], list[dict], list[dict]]:
    rng = np.random.default_rng(seed)
    d = int(cfg["task"]["features_per_stream"])
    teacher_a = rng.normal(size=d)
    teacher_b = rng.normal(size=d)
    teacher_a /= np.linalg.norm(teacher_a)
    teacher_b /= np.linalg.norm(teacher_b)
    train_data = make_dataset(
        rng, int(cfg["task"]["train_examples"]), teacher_a, teacher_b, cfg
    )
    test_data = make_dataset(
        rng, int(cfg["task"]["test_examples"]), teacher_a, teacher_b, cfg
    )
    switch_data = make_dataset(
        rng,
        int(cfg["task"]["switch_examples"]),
        teacher_a,
        teacher_b,
        cfg,
        context_only=1,
    )
    context_zero = subset(test_data, np.flatnonzero(test_data.context == 0))
    init = rng.normal(scale=float(cfg["training"]["initialization_sd"]), size=(2, d))
    route = route_matrix(seed)
    schedule = batch_schedule(
        seed,
        len(train_data.label),
        int(cfg["training"]["epochs"]),
        int(cfg["training"]["batch_size"]),
        salt=90_001,
    )
    switch_schedule = batch_schedule(
        seed,
        len(switch_data.label),
        int(cfg["training"]["switch_epochs"]),
        int(cfg["training"]["batch_size"]),
        salt=90_101,
    )
    train_audit = task_audit(train_data)
    test_audit = task_audit(test_data)
    outcomes: list[dict] = []
    gradients: list[dict] = []
    ledgers: list[dict] = []
    for condition in CONDITION_ORDER:
        init_cosine, init_capture = gradient_geometry(init, test_data, condition, route)
        init_progress = one_step_progress(init, test_data, condition, route)
        fitted = train(
            init,
            train_data,
            condition,
            route,
            schedule,
            float(cfg["training"]["learning_rate"]),
        )
        train_loss, train_accuracy = loss_accuracy(fitted, train_data)
        test_loss, test_accuracy = loss_accuracy(fitted, test_data)
        context0_loss, context0_accuracy = loss_accuracy(fitted, context_zero)
        trained_cosine, trained_capture = gradient_geometry(
            fitted, test_data, condition, route
        )
        adapted = train(
            fitted,
            switch_data,
            condition,
            route,
            switch_schedule,
            float(cfg["training"]["learning_rate"]),
        )
        _, context0_after = loss_accuracy(adapted, context_zero)
        feedback_channels = {
            "neuron_shared_k1": 1,
            "depth_shared_k1": 1,
            "correct_subtree_k2": 2,
            "within_neuron_deranged_k2": 2,
            "random_dense_rank2": 2,
            "exact_transport": 2,
            "backpropagation": 2,
            "gated_point_emulation": 1,
        }[condition]
        feedback_nonzeros = 4 if condition == "random_dense_rank2" else 2
        outcomes.append(
            {
                "seed": seed,
                "condition": condition,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "test_loss": test_loss,
                "test_accuracy": test_accuracy,
                "context0_loss_before_switch": context0_loss,
                "context0_accuracy_before_switch": context0_accuracy,
                "context0_accuracy_after_switch": context0_after,
                "context_switch_forgetting": context0_accuracy - context0_after,
                "forward_parameters": int(init.size),
                "active_input_contacts": int(2 * d),
                "forward_branch_groups": 2,
                "feedback_channels": feedback_channels,
                "feedback_nonzeros": feedback_nonzeros,
            }
        )
        gradients.append(
            {
                "seed": seed,
                "condition": condition,
                "initial_gradient_cosine": init_cosine,
                "initial_gradient_scaled_capture": init_capture,
                "initial_norm_matched_one_step_progress": init_progress,
                "trained_gradient_cosine": trained_cosine,
                "trained_gradient_scaled_capture": trained_capture,
            }
        )
        ledgers.append(
            {
                "seed": seed,
                "condition": condition,
                "train_balanced": train_audit["balanced_context_label_cells"],
                "test_balanced": test_audit["balanced_context_label_cells"],
                "train_context_label_correlation": train_audit[
                    "context_label_correlation"
                ],
                "test_context_label_correlation": test_audit[
                    "context_label_correlation"
                ],
                "route_matrix_00": float(route[0, 0]),
                "route_matrix_01": float(route[0, 1]),
                "route_matrix_10": float(route[1, 0]),
                "route_matrix_11": float(route[1, 1]),
                "fallback_count": 0,
                "scalar_replacement_rate": 0.0,
            }
        )
    return outcomes, gradients, ledgers


def bootstrap_ci(values: np.ndarray, seed: int) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = rng.choice(values, size=(20_000, values.size), replace=True).mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(values.mean()), float(low), float(high)


def paired_contrasts(outcomes: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    comparisons = [
        ("correct_subtree_k2", "within_neuron_deranged_k2", "correct - deranged"),
        ("correct_subtree_k2", "neuron_shared_k1", "correct - neuron shared"),
        ("correct_subtree_k2", "depth_shared_k1", "correct - depth shared"),
        ("correct_subtree_k2", "random_dense_rank2", "correct - random rank-2"),
        ("correct_subtree_k2", "gated_point_emulation", "correct - gated point"),
        ("exact_transport", "backpropagation", "exact - backpropagation"),
    ]
    endpoints = ["test_accuracy", "test_loss", "context_switch_forgetting"]
    for contrast_index, (left, right, label) in enumerate(comparisons):
        for endpoint_index, endpoint in enumerate(endpoints):
            left_rows = outcomes[outcomes.condition.eq(left)][["seed", endpoint]]
            right_rows = outcomes[outcomes.condition.eq(right)][["seed", endpoint]]
            paired = left_rows.merge(
                right_rows, on="seed", suffixes=("_left", "_right"), validate="one_to_one"
            )
            values = (
                paired[f"{endpoint}_left"] - paired[f"{endpoint}_right"]
            ).to_numpy(float)
            mean, low, high = bootstrap_ci(
                values, seed=20_000 + 10 * contrast_index + endpoint_index
            )
            p_value = 1.0 if np.allclose(values, 0) else float(
                wilcoxon(values, zero_method="wilcox", alternative="two-sided").pvalue
            )
            rows.append(
                {
                    "contrast": label,
                    "left_condition": left,
                    "right_condition": right,
                    "endpoint": endpoint,
                    "n_pairs": len(values),
                    "mean_difference": mean,
                    "ci95_low": low,
                    "ci95_high": high,
                    "wins": int(np.sum(values > 0)),
                    "ties": int(np.sum(np.isclose(values, 0))),
                    "wilcoxon_p_two_sided": p_value,
                }
            )
    return pd.DataFrame(rows)


def condition_summary(outcomes: pd.DataFrame, gradients: pd.DataFrame) -> pd.DataFrame:
    merged = outcomes.merge(gradients, on=["seed", "condition"], validate="one_to_one")
    rows: list[dict] = []
    for index, condition in enumerate(CONDITION_ORDER):
        part = merged[merged.condition.eq(condition)]
        row: dict[str, object] = {"condition": condition, "n_seeds": len(part)}
        for metric_index, metric in enumerate(
            [
                "test_accuracy",
                "test_loss",
                "context_switch_forgetting",
                "initial_gradient_cosine",
                "initial_gradient_scaled_capture",
                "initial_norm_matched_one_step_progress",
            ]
        ):
            mean, low, high = bootstrap_ci(
                part[metric].to_numpy(float), seed=30_000 + 20 * index + metric_index
            )
            row[f"mean_{metric}"] = mean
            row[f"ci95_low_{metric}"] = low
            row[f"ci95_high_{metric}"] = high
        rows.append(row)
    return pd.DataFrame(rows)


def verify_gates(
    outcomes: pd.DataFrame, gradients: pd.DataFrame, cfg: dict
) -> dict[str, float | bool]:
    gates = cfg["gates"]
    by_condition = outcomes.set_index("condition")
    by_gradient = gradients.set_index("condition")
    exact_bp_accuracy_difference = abs(
        float(by_condition.loc["exact_transport", "test_accuracy"])
        - float(by_condition.loc["backpropagation", "test_accuracy"])
    )
    exact_bp_weighted_metrics = max(
        exact_bp_accuracy_difference,
        abs(
            float(by_condition.loc["exact_transport", "test_loss"])
            - float(by_condition.loc["backpropagation", "test_loss"])
        ),
    )
    result = {
        "neuron_shared_capture": float(
            by_gradient.loc[
                "neuron_shared_k1", "initial_gradient_scaled_capture"
            ]
        ),
        "correct_capture": float(
            by_gradient.loc[
                "correct_subtree_k2", "initial_gradient_scaled_capture"
            ]
        ),
        "correct_test_accuracy": float(
            by_condition.loc["correct_subtree_k2", "test_accuracy"]
        ),
        "exact_bp_max_endpoint_difference": exact_bp_weighted_metrics,
    }
    result["passed"] = bool(
        result["neuron_shared_capture"]
        < float(gates["maximum_neuron_shared_capture"])
        and result["correct_capture"] >= float(gates["minimum_correct_capture"])
        and result["correct_test_accuracy"]
        >= float(gates["minimum_correct_test_accuracy"])
        and result["exact_bp_max_endpoint_difference"]
        <= float(gates["maximum_exact_bp_difference"])
    )
    return result


def save_confirmatory(
    outcomes: pd.DataFrame,
    gradients: pd.DataFrame,
    ledgers: pd.DataFrame,
    cfg: dict,
) -> None:
    SOURCE.mkdir(parents=True, exist_ok=True)
    outcomes.to_csv(SOURCE / "seed_outcomes.csv", index=False, float_format="%.10g")
    gradients.to_csv(SOURCE / "gradient_audit.csv", index=False, float_format="%.10g")
    ledgers.to_csv(SOURCE / "mechanism_ledger.csv", index=False, float_format="%.10g")
    contrasts = paired_contrasts(outcomes)
    summary = condition_summary(outcomes, gradients)
    contrasts.to_csv(SOURCE / "paired_contrasts.csv", index=False, float_format="%.10g")
    summary.to_csv(SOURCE / "condition_summary.csv", index=False, float_format="%.10g")
    metadata = {
        "study": cfg["study"],
        "status": "complete_phase1_confirmatory",
        "scope_boundary": cfg["scope_boundary"],
        "config_sha256": sha256(CONFIG),
        "script_sha256": sha256(Path(__file__).resolve()),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "n_seeds": int(outcomes.seed.nunique()),
        "n_conditions": int(outcomes.condition.nunique()),
        "all_balanced": bool(ledgers.train_balanced.all() and ledgers.test_balanced.all()),
        "all_fallback_counts_zero": bool((ledgers.fallback_count == 0).all()),
        "exact_bp_max_test_accuracy_difference": float(
            outcomes.pivot(index="seed", columns="condition", values="test_accuracy")
            .eval("exact_transport - backpropagation")
            .abs()
            .max()
        ),
    }
    (SOURCE / "summary.json").write_text(json.dumps(metadata, indent=2) + "\n")


def plot_confirmatory(outcomes: pd.DataFrame, gradients: pd.DataFrame) -> None:
    apply_neurips_style()
    merged = outcomes.merge(gradients, on=["seed", "condition"], validate="one_to_one")
    displayed = [
        "neuron_shared_k1",
        "correct_subtree_k2",
        "within_neuron_deranged_k2",
        "random_dense_rank2",
        "exact_transport",
        "gated_point_emulation",
    ]
    short_labels = [
        "neuron\nshared",
        "correct\nsubtree",
        "deranged\nsubtree",
        "random\nrank-2",
        "exact",
        "gated point\nemulation",
    ]
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(FIG_W, 2.85),
        gridspec_kw={
            "left": 0.085,
            "right": 0.985,
            "bottom": 0.245,
            "top": 0.84,
            "wspace": 0.52,
        },
    )
    ax_a, ax_b, ax_c = axes
    for index, condition in enumerate(displayed):
        values = merged[merged.condition.eq(condition)].test_accuracy.to_numpy(float)
        jitter = np.linspace(-0.055, 0.055, values.size)
        color = COLORS_BY_CONDITION[condition]
        ax_a.scatter(
            index + jitter,
            values,
            s=11,
            color=color,
            alpha=0.55,
            edgecolors="none",
        )
        mean, low, high = bootstrap_ci(values, seed=40_000 + index)
        ax_a.errorbar(
            index,
            mean,
            yerr=[[mean - low], [high - mean]],
            marker="D",
            markerfacecolor="white",
            markeredgecolor=color,
            color=color,
            ms=4.2,
            lw=LW_ERR,
            capsize=ERR_CAPSIZE,
            zorder=5,
        )
    ax_a.set_xticks(range(len(displayed)), short_labels, rotation=24, ha="right")
    ax_a.tick_params(axis="x", labelsize=PT_SMALL)
    ax_a.set_ylabel("held-out accuracy")
    ax_a.set_ylim(0.0, 1.03)
    panel_title(ax_a, "A", "Trained subtree address")
    style_axis(ax_a, grid="y")

    for condition in displayed:
        part = merged[merged.condition.eq(condition)]
        ax_b.scatter(
            part.initial_gradient_scaled_capture,
            part.initial_norm_matched_one_step_progress,
            s=15,
            color=COLORS_BY_CONDITION[condition],
            alpha=0.62,
            edgecolors="none",
            label=LABELS[condition],
        )
    ax_b.axhline(0, color=COLORS["mute"], ls="--", lw=0.8)
    ax_b.set_xlabel("exact-gradient capture")
    ax_b.set_ylabel("norm-matched one-step progress")
    panel_title(ax_b, "B", "Geometry predicts progress")
    style_axis(ax_b)

    switch_display = [
        "correct_subtree_k2",
        "neuron_shared_k1",
        "within_neuron_deranged_k2",
        "random_dense_rank2",
    ]
    switch_labels = ["correct", "neuron\nshared", "deranged", "random"]
    for index, condition in enumerate(switch_display):
        values = merged[
            merged.condition.eq(condition)
        ].context_switch_forgetting.to_numpy(float)
        jitter = np.linspace(-0.05, 0.05, values.size)
        color = COLORS_BY_CONDITION[condition]
        ax_c.scatter(
            index + jitter,
            values,
            s=11,
            color=color,
            alpha=0.55,
            edgecolors="none",
        )
        mean, low, high = bootstrap_ci(values, seed=41_000 + index)
        ax_c.errorbar(
            index,
            mean,
            yerr=[[mean - low], [high - mean]],
            color=color,
            marker="D",
            markerfacecolor="white",
            ms=4.2,
            lw=LW_ERR,
            capsize=ERR_CAPSIZE,
        )
    ax_c.axhline(0, color=COLORS["mute"], ls="--", lw=0.8)
    ax_c.set_xticks(range(len(switch_display)), switch_labels)
    ax_c.tick_params(axis="x", labelsize=PT_SMALL)
    ax_c.set_ylabel("context-0 forgetting")
    panel_title(ax_c, "C", "Context-switch interference")
    style_axis(ax_c, grid="y")

    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.canvas.draw()
    audit_layout(fig, "fig_trained_subtree_address")
    audit_text_over_data(fig, "fig_trained_subtree_address")
    fig.savefig(
        FIGURES / "fig_trained_subtree_address.pdf",
        metadata={"CreationDate": None, "ModDate": None},
    )
    fig.savefig(FIGURES / "fig_trained_subtree_address.png", dpi=600)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["canary", "confirmatory"], required=True)
    args = parser.parse_args()
    cfg = json.loads(CONFIG.read_text())
    seeds = (
        [int(cfg["canary_seed"])]
        if args.phase == "canary"
        else [int(seed) for seed in cfg["confirmatory_seeds"]]
    )
    outcome_rows: list[dict] = []
    gradient_rows: list[dict] = []
    ledger_rows: list[dict] = []
    for seed in seeds:
        outcomes, gradients, ledgers = run_seed(seed, cfg)
        outcome_rows.extend(outcomes)
        gradient_rows.extend(gradients)
        ledger_rows.extend(ledgers)
    outcomes = pd.DataFrame(outcome_rows)
    gradients = pd.DataFrame(gradient_rows)
    ledgers = pd.DataFrame(ledger_rows)
    if args.phase == "canary":
        gate = verify_gates(outcomes, gradients, cfg)
        payload = {
            "phase": "canary",
            "seed": seeds[0],
            "config_sha256": sha256(CONFIG),
            "script_sha256": sha256(Path(__file__).resolve()),
            "artifact_and_nondegeneracy_gates": gate,
            "outcomes_are_not_confirmatory": True,
        }
        CANARY.write_text(json.dumps(payload, indent=2) + "\n")
        print(json.dumps(payload, indent=2))
        if not gate["passed"]:
            raise SystemExit("Canary gate failed; confirmatory execution is blocked")
        return
    if not CANARY.exists():
        raise SystemExit("Run and pass the frozen canary before confirmatory execution")
    canary = json.loads(CANARY.read_text())
    if not canary["artifact_and_nondegeneracy_gates"]["passed"]:
        raise SystemExit("Recorded canary did not pass")
    if canary["config_sha256"] != sha256(CONFIG):
        raise SystemExit("Configuration changed after canary")
    if canary["script_sha256"] != sha256(Path(__file__).resolve()):
        raise SystemExit("Script changed after canary")
    save_confirmatory(outcomes, gradients, ledgers, cfg)
    plot_confirmatory(outcomes, gradients)
    print(
        f"complete: {outcomes.seed.nunique()} seeds, "
        f"{outcomes.condition.nunique()} conditions, {len(outcomes)} runs"
    )


if __name__ == "__main__":
    main()
