#!/usr/bin/env python3
"""Reanalyse the frozen 2,700-fit address factorial as credit operators.

This is a deterministic secondary analysis.  It reconstructs the exact frozen
datasets and route matrices from their archived seeds, estimates minibatch
gradient noise at initialization, and reports signal retention, noise
admission, spectral capture, conditioning, and the smoothness-bound utility.
It does not alter or rerun the confirmatory training outcomes.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "trained_subtree_address" / "full_factorial_confirmatory.json"
OUTCOMES = ROOT / "source_data" / "trained_subtree_address_full_factorial" / "seed_outcomes.csv"
OUTPUT = ROOT / "source_data" / "credit_phase_existing"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


FACTORIAL = load_module(
    "frozen_subtree_factorial",
    ROOT / "scripts" / "run_trained_subtree_address_full_factorial.py",
)
PHASE = load_module("credit_phase", ROOT / "code" / "theory" / "credit_phase.py")


def subset(data, indices: np.ndarray):
    return FACTORIAL.Dataset(
        x=data.x[indices], context=data.context[indices], label=data.label[indices]
    )


def reconstruct(seed: int, cfg: dict):
    rng = np.random.default_rng(seed)
    n = FACTORIAL.n_contexts(cfg)
    d = int(cfg["task"]["features_per_stream"])
    teachers = rng.normal(size=(n, d))
    teachers /= np.linalg.norm(teachers, axis=1, keepdims=True)
    train = FACTORIAL.make_dataset(
        rng, int(cfg["task"]["train_examples"]), teachers, cfg
    )
    # Advance through exactly the same frozen RNG sequence used by run_seed.
    FACTORIAL.make_dataset(rng, int(cfg["task"]["test_examples"]), teachers, cfg)
    FACTORIAL.make_dataset(
        rng,
        int(cfg["task"]["switch_examples"]),
        teachers,
        cfg,
        allowed_contexts=cfg["task"]["switch_contexts"],
    )
    condition_list = FACTORIAL.conditions(cfg)
    permutations = FACTORIAL.architecture_permutations(
        seed, cfg["architectures"], n
    )
    initial_task = rng.normal(
        scale=float(cfg["training"]["initialization_sd"]), size=(n, d)
    )
    initial = FACTORIAL.initial_weights(initial_task, condition_list, permutations)
    routes = FACTORIAL.route_matrices(
        cfg, seed, condition_list, permutations, train, initial_task
    )
    return train, condition_list, permutations, initial_task, initial, routes


def objective_smoothness(data, initial_task: np.ndarray, context_scale: float) -> float:
    n = len(data.label)
    maximum = 0.0
    for context in np.unique(data.context):
        mask = data.context == context
        x = data.x[mask, int(context), :]
        logits = x @ initial_task[int(context)]
        p = FACTORIAL.sigmoid(logits)
        weighted = x * np.sqrt(p * (1.0 - p))[:, None]
        hessian = float(context_scale) * (weighted.T @ weighted) / n
        maximum = max(maximum, float(np.linalg.eigvalsh(hessian).max()))
    return maximum


def route_projector(route: np.ndarray) -> np.ndarray:
    basis, singular, _ = np.linalg.svd(route.T, full_matrices=False)
    keep = singular > 1e-12
    return basis[:, keep] @ basis[:, keep].T


def route_dictionary(route: np.ndarray) -> np.ndarray:
    unique = np.unique(np.round(route, decimals=12), axis=0)
    return unique.T


def bootstrap(values: np.ndarray, seed: int, draws: int = 20_000) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    sampled = rng.choice(values, size=(draws, len(values)), replace=True).mean(axis=1)
    low, high = np.quantile(sampled, [0.025, 0.975])
    return float(values.mean()), float(low), float(high)


def block_bootstrap_correlations(
    frame: pd.DataFrame, draws: int, seed: int
) -> dict[str, tuple[float, float, float]]:
    """Seed-block intervals for the three descriptive rank correlations."""

    seeds = np.asarray(sorted(frame.seed.unique()), dtype=int)
    rng = np.random.default_rng(seed)
    block_indices = {
        int(value): np.flatnonzero(frame.seed.to_numpy(int) == int(value))
        for value in seeds
    }
    utility = frame.maximum_guaranteed_decrease.to_numpy(float)
    capture = frame.initial_gradient_capture.to_numpy(float)
    progress = frame.norm_matched_one_step_progress.to_numpy(float)
    accuracy = frame.heldout_accuracy.to_numpy(float)
    values = {"utility_progress": [], "capture_progress": [], "utility_accuracy": []}
    for _ in range(int(draws)):
        sampled = rng.choice(seeds, size=len(seeds), replace=True)
        indices = np.concatenate([block_indices[int(value)] for value in sampled])
        values["utility_progress"].append(
            spearmanr(utility[indices], progress[indices]).statistic
        )
        values["capture_progress"].append(
            spearmanr(capture[indices], progress[indices]).statistic
        )
        values["utility_accuracy"].append(
            spearmanr(utility[indices], accuracy[indices]).statistic
        )
    result = {}
    for key, draws_for_key in values.items():
        array = np.asarray(draws_for_key, dtype=float)
        low, high = np.quantile(array, [0.025, 0.975])
        result[key] = (float(np.mean(array)), float(low), float(high))
    difference = np.asarray(values["utility_progress"]) - np.asarray(
        values["capture_progress"]
    )
    low, high = np.quantile(difference, [0.025, 0.975])
    result["utility_minus_capture_progress"] = (
        float(np.mean(difference)),
        float(low),
        float(high),
    )
    return result


def analyse_seed(seed: int, cfg: dict, n_batches: int, batch_size: int):
    train, conditions, permutations, initial_task, initial, routes = reconstruct(seed, cfg)
    context_scale = float(cfg["training"]["gradient_context_rescaling"])
    full = FACTORIAL.gradients_all(
        initial, train, routes, conditions, permutations, context_scale
    )
    exact_index = {
        architecture_index: next(
            index
            for index, condition in enumerate(conditions)
            if condition.architecture_index == architecture_index
            and condition.family == "exact_compartment_transport"
        )
        for architecture_index in range(len(permutations))
    }
    rng = np.random.default_rng(seed + 1_700_000)
    batch_gradients = np.empty((n_batches,) + full.shape, dtype=float)
    for draw in range(n_batches):
        indices = rng.integers(0, len(train.label), size=batch_size)
        batch_gradients[draw] = FACTORIAL.gradients_all(
            initial,
            subset(train, indices),
            routes,
            conditions,
            permutations,
            context_scale,
        )
    noise = np.mean(
        np.sum(
            (batch_gradients - full[None, ...]) ** 2,
            axis=tuple(range(2, batch_gradients.ndim)),
        ),
        axis=0,
    )
    smoothness = objective_smoothness(train, initial_task, context_scale)

    # Exact coefficient covariance at the shared frozen initialization.
    logits = np.sum(
        initial_task[train.context]
        * train.x[np.arange(len(train.label)), train.context],
        axis=1,
    )
    delta = FACTORIAL.sigmoid(logits) - train.label
    exact_coefficients = np.zeros((len(train.label), len(initial_task)), dtype=float)
    exact_coefficients[np.arange(len(train.label)), train.context] = delta
    coefficient_covariance = exact_coefficients.T @ exact_coefficients / len(train.label)
    eigenvalues = np.linalg.eigvalsh(coefficient_covariance)[::-1]

    rows = []
    for index, condition in enumerate(conditions):
        exact = full[exact_index[condition.architecture_index]].ravel()
        approximate = full[index].ravel()
        exact_noise = float(noise[exact_index[condition.architecture_index]])
        admitted_noise = float(noise[index])
        retained = float(exact @ approximate)
        second = float(approximate @ approximate + admitted_noise)
        eta_star = retained / (smoothness * second) if retained > 0 and second > 0 else 0.0
        utility = retained * retained / (2.0 * smoothness * second) if retained > 0 and second > 0 else 0.0
        projector = route_projector(routes[index])
        rank = int(round(np.trace(projector)))
        spectral = PHASE.spectral_capture(projector, coefficient_covariance)
        pca_capture = float(eigenvalues[:rank].sum() / eigenvalues.sum())
        diagnostics = PHASE.route_gram_diagnostics(
            route_dictionary(routes[index]), np.ones(len(initial_task))
        )
        rows.append(
            {
                "seed": seed,
                "condition_id": condition.condition_id,
                "architecture": condition.architecture,
                "feedback_family": condition.family,
                "budget_k": condition.budget_k,
                "route_rank": rank,
                "objective_smoothness": smoothness,
                "retained_signal_inner_product": retained,
                "retained_signal_fraction": retained / max(float(exact @ exact), 1e-30),
                "operator_gradient_norm_sq": float(approximate @ approximate),
                "admitted_gradient_noise": admitted_noise,
                "exact_gradient_noise": exact_noise,
                "noise_retention_fraction": admitted_noise / max(exact_noise, 1e-30),
                "gradient_estimator_mse": float(np.sum((exact - approximate) ** 2) + admitted_noise),
                "full_stochastic_gradient_mse": exact_noise,
                "projected_estimator_better": bool(
                    np.sum((exact - approximate) ** 2) + admitted_noise < exact_noise
                ),
                "optimal_bound_step": eta_star,
                "maximum_guaranteed_decrease": utility,
                "spectral_capture": spectral,
                "pca_rank_upper_bound": pca_capture,
                "morphology_regret": pca_capture - spectral,
                "route_gram_condition": diagnostics["condition_number"],
                "route_coherence": diagnostics["coherence"],
            }
        )

    # Reliability quantities are calculated from the exact method only and
    # use leaf/task blocks as the disjoint branch groups.
    exact_reference = exact_index[0]
    exact_full = full[exact_reference]
    exact_batches = batch_gradients[:, exact_reference]
    reliability = []
    signal_energy = np.sum(exact_full * exact_full, axis=1)
    noise_energy = np.mean(
        np.sum((exact_batches - exact_full[None, ...]) ** 2, axis=2), axis=0
    )
    gains = PHASE.reliability_shrinkage(signal_energy, noise_energy)
    for branch in range(len(signal_energy)):
        reliability.append(
            {
                "seed": seed,
                "branch": branch,
                "signal_energy": float(signal_energy[branch]),
                "noise_energy": float(noise_energy[branch]),
                "signal_to_noise": float(
                    signal_energy[branch] / max(noise_energy[branch], 1e-30)
                ),
                "optimal_reliability_gain": float(gains[branch]),
            }
        )
    spectra = [
        {
            "seed": seed,
            "mode": mode + 1,
            "eigenvalue": float(value),
            "energy_fraction": float(value / eigenvalues.sum()),
        }
        for mode, value in enumerate(eigenvalues)
    ]
    return rows, reliability, spectra


def summarize(metrics: pd.DataFrame, outcomes: pd.DataFrame, cfg: dict) -> dict:
    merged = metrics.merge(
        outcomes,
        on=["seed", "condition_id", "architecture", "feedback_family", "budget_k"],
        validate="one_to_one",
    )
    primary = merged[merged.architecture.eq("dendritic_tree")].copy()
    finite = primary[
        np.isfinite(primary.maximum_guaranteed_decrease)
        & np.isfinite(primary.norm_matched_one_step_progress)
    ]
    utility_progress = spearmanr(
        finite.maximum_guaranteed_decrease,
        finite.norm_matched_one_step_progress,
    )
    utility_accuracy = spearmanr(
        finite.maximum_guaranteed_decrease, finite.heldout_accuracy
    )
    capture_progress = spearmanr(
        finite.initial_gradient_capture, finite.norm_matched_one_step_progress
    )
    block = block_bootstrap_correlations(
        finite, int(cfg["analysis"]["bootstrap_draws"]), 879_101
    )
    summary_rows = []
    for group_index, (keys, part) in enumerate(
        primary.groupby(["feedback_family", "budget_k"], sort=True)
    ):
        family, budget = keys
        row = {
            "feedback_family": family,
            "budget_k": int(budget),
            "n_seeds": int(part.seed.nunique()),
        }
        for metric_index, metric in enumerate(
            [
                "spectral_capture",
                "morphology_regret",
                "retained_signal_fraction",
                "noise_retention_fraction",
                "maximum_guaranteed_decrease",
                "route_gram_condition",
                "route_coherence",
            ]
        ):
            mean, low, high = bootstrap(
                part[metric].to_numpy(float),
                880_000 + 100 * group_index + metric_index,
                int(cfg["analysis"]["bootstrap_draws"]),
            )
            row[f"mean_{metric}"] = mean
            row[f"ci95_low_{metric}"] = low
            row[f"ci95_high_{metric}"] = high
        row["projected_estimator_better_fraction"] = float(
            part.projected_estimator_better.mean()
        )
        summary_rows.append(row)
    pd.DataFrame(summary_rows).to_csv(
        OUTPUT / "condition_summary.csv", index=False, float_format="%.10g"
    )
    return {
        "status": "complete_secondary_analysis",
        "frozen_factorial_models": int(len(outcomes)),
        "reanalysed_seeds": int(metrics.seed.nunique()),
        "minibatches_per_seed": int(metrics.minibatches_per_seed.iloc[0]),
        "minibatch_size": int(metrics.minibatch_size.iloc[0]),
        "dendritic_tree_rows": int(len(primary)),
        "spearman_utility_vs_one_step_progress": float(utility_progress.statistic),
        "seed_block_ci95_utility_vs_one_step_progress": list(
            block["utility_progress"][1:]
        ),
        "spearman_capture_vs_one_step_progress": float(capture_progress.statistic),
        "seed_block_ci95_capture_vs_one_step_progress": list(
            block["capture_progress"][1:]
        ),
        "seed_block_mean_utility_minus_capture_rho": block[
            "utility_minus_capture_progress"
        ][0],
        "seed_block_ci95_utility_minus_capture_rho": list(
            block["utility_minus_capture_progress"][1:]
        ),
        "spearman_utility_vs_final_accuracy": float(utility_accuracy.statistic),
        "seed_block_ci95_utility_vs_final_accuracy": list(
            block["utility_accuracy"][1:]
        ),
        "scope_boundary": (
            "The noise and smoothness quantities are initialization diagnostics on the "
            "frozen synthetic factorial. They explain immediate stochastic geometry and "
            "do not by themselves establish a final-accuracy or biological-noise advantage."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--minibatches", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    outcomes = pd.read_csv(OUTCOMES)
    expected_seeds = [int(value) for value in cfg["confirmatory_seeds"]]
    if sorted(outcomes.seed.unique().tolist()) != expected_seeds:
        raise SystemExit("frozen outcome seeds do not match the confirmatory contract")
    all_rows, all_reliability, all_spectra = [], [], []
    for seed in expected_seeds:
        rows, reliability, spectra = analyse_seed(
            seed, cfg, args.minibatches, args.batch_size
        )
        all_rows.extend(rows)
        all_reliability.extend(reliability)
        all_spectra.extend(spectra)
    metrics = pd.DataFrame(all_rows)
    metrics["minibatches_per_seed"] = int(args.minibatches)
    metrics["minibatch_size"] = int(args.batch_size)
    expected = len(outcomes)
    if len(metrics) != expected:
        raise SystemExit(f"expected {expected} operator rows, found {len(metrics)}")
    if metrics.duplicated(["seed", "condition_id"]).any():
        raise SystemExit("duplicate operator rows")
    if not np.isfinite(metrics.select_dtypes(include=[np.number])).all().all():
        raise SystemExit("non-finite operator metric")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(OUTPUT / "operator_metrics.csv", index=False, float_format="%.10g")
    pd.DataFrame(all_reliability).to_csv(
        OUTPUT / "branch_reliability.csv", index=False, float_format="%.10g"
    )
    pd.DataFrame(all_spectra).to_csv(
        OUTPUT / "task_spectra.csv", index=False, float_format="%.10g"
    )
    summary = summarize(metrics, outcomes, cfg)
    (OUTPUT / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    report = f"""# Credit-phase reanalysis of the frozen subtree factorial

- Frozen trained fits: {summary['frozen_factorial_models']:,}
- Reanalysed paired seeds: {summary['reanalysed_seeds']}
- Minibatches per seed: {summary['minibatches_per_seed']} at batch size {summary['minibatch_size']}
- Utility versus norm-matched one-step progress: Spearman rho = {summary['spearman_utility_vs_one_step_progress']:.3f}
- Capture versus norm-matched one-step progress: Spearman rho = {summary['spearman_capture_vs_one_step_progress']:.3f}
- Utility versus final held-out accuracy: Spearman rho = {summary['spearman_utility_vs_final_accuracy']:.3f}

The smoothness-bound utility combines retained population-gradient signal,
finite-step gain cost, and admitted minibatch-gradient noise. Spectral capture
and morphology regret are computed from the exact task-coefficient covariance.
The analysis is diagnostic at the frozen initialization and does not turn a
same-model local update into an optimizer that uniformly exceeds exact
full-batch backpropagation.
"""
    (OUTPUT / "report.md").write_text(report, encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
