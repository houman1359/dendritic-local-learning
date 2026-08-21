#!/usr/bin/env python3
"""Measure address and coefficient residuals in the trained route factorial.

This is a post-hoc diagnostic of the already frozen 2,700-fit experiment.  It
reconstructs every seed from the unchanged, hash-matched simulation and checks
the reconstructed endpoints against the archived source data before computing
the weighted projection decomposition at initialization and after training.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon


ROOT = Path(__file__).resolve().parents[1]
SIMULATION_PATH = ROOT / "scripts" / "run_trained_subtree_address_full_factorial.py"
CONFIG_PATH = ROOT / "configs" / "trained_subtree_address" / "full_factorial_confirmatory.json"
ARCHIVED = ROOT / "source_data" / "trained_subtree_address_full_factorial"
OUTPUT = ROOT / "source_data" / "trained_partition_residual"


def _load_simulation():
    spec = importlib.util.spec_from_file_location("trained_subtree_factorial", SIMULATION_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not import {SIMULATION_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


SIM = _load_simulation()


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def weighted_field_decomposition(
    exact: np.ndarray,
    route: np.ndarray,
    weights: np.ndarray,
    delta: np.ndarray,
) -> dict[str, float]:
    """Project each example's exact field onto its available route.

    One teaching coefficient may be chosen for the active route on each
    example.  ``route`` therefore specifies the address/support constraint;
    the coefficient actually used by the simulation remains ``delta``.
    """

    exact = np.asarray(exact, dtype=float)
    route = np.asarray(route, dtype=float)
    weights = np.asarray(weights, dtype=float)
    delta = np.asarray(delta, dtype=float)
    if not (exact.shape == route.shape == weights.shape):
        raise ValueError("exact, route and weights must have the same shape")
    if delta.shape != (len(exact),):
        raise ValueError("delta must contain one value per example")
    if np.any(weights <= 0):
        raise ValueError("coordinate weights must be strictly positive")

    coefficient = delta[:, None] * exact
    numerator = np.sum(weights * coefficient * route, axis=1)
    denominator = np.sum(weights * route * route, axis=1)
    optimum = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator > 1e-30,
    )
    projected = optimum[:, None] * route
    supplied = delta[:, None] * route
    address = float(np.sum(weights * (coefficient - projected) ** 2))
    estimation = float(np.sum(weights * (projected - supplied) ** 2))
    total = float(np.sum(weights * (coefficient - supplied) ** 2))
    energy = float(np.sum(weights * coefficient * coefficient))
    return {
        "exact_field_energy": energy,
        "address_residual_energy": address,
        "coefficient_residual_energy": estimation,
        "total_field_error": total,
        "address_capture": 1.0 - address / max(energy, 1e-30),
        "realized_field_score": 1.0 - total / max(energy, 1e-30),
        "decomposition_error": abs(address + estimation - total),
    }


def _state_rows(
    seed: int,
    state_name: str,
    weights: np.ndarray,
    data,
    routes: np.ndarray,
    condition_list: list,
    permutations: np.ndarray,
) -> list[dict]:
    logits = SIM.logits_all(weights, data, condition_list, permutations)
    deltas = SIM.sigmoid(logits) - data.label[None, :]
    rows: list[dict] = []
    for index, condition in enumerate(condition_list):
        permutation = permutations[condition.architecture_index]
        inputs = SIM.block_inputs(data, permutation)
        eligibility_energy = np.sum(inputs * inputs, axis=2) + 1e-15
        exact = SIM.exact_routes(permutation)[data.context]
        route = routes[index, data.context]
        values = weighted_field_decomposition(
            exact,
            route,
            eligibility_energy,
            deltas[index],
        )
        rows.append(
            {
                "seed": seed,
                "state": state_name,
                "condition_id": condition.condition_id,
                "architecture": condition.architecture,
                "feedback_family": condition.family,
                "budget_k": condition.budget_k,
                **values,
            }
        )
    return rows


def reconstruct_seed(seed: int, cfg: dict) -> tuple[list[dict], pd.DataFrame]:
    rng = np.random.default_rng(seed)
    n = SIM.n_contexts(cfg)
    d = int(cfg["task"]["features_per_stream"])
    teachers = rng.normal(size=(n, d))
    teachers /= np.linalg.norm(teachers, axis=1, keepdims=True)
    train_data = SIM.make_dataset(rng, int(cfg["task"]["train_examples"]), teachers, cfg)
    test_data = SIM.make_dataset(rng, int(cfg["task"]["test_examples"]), teachers, cfg)
    # Advance the generator exactly as the frozen simulation did before
    # drawing the common initialization.
    SIM.make_dataset(
        rng,
        int(cfg["task"]["switch_examples"]),
        teachers,
        cfg,
        allowed_contexts=cfg["task"]["switch_contexts"],
    )
    condition_list = SIM.conditions(cfg)
    permutations = SIM.architecture_permutations(seed, cfg["architectures"], n)
    initial_task = rng.normal(
        scale=float(cfg["training"]["initialization_sd"]), size=(n, d)
    )
    initial = SIM.initial_weights(initial_task, condition_list, permutations)
    routes = SIM.route_matrices(
        cfg, seed, condition_list, permutations, train_data, initial_task
    )
    fitted = SIM.train_all(
        initial,
        train_data,
        routes,
        condition_list,
        permutations,
        cfg,
        epochs=int(cfg["training"]["epochs"]),
    )
    loss, accuracy = SIM.loss_accuracy(
        SIM.logits_all(fitted, test_data, condition_list, permutations),
        test_data.label,
    )
    endpoints = pd.DataFrame(
        {
            "seed": seed,
            "condition_id": [condition.condition_id for condition in condition_list],
            "reconstructed_heldout_loss": loss,
            "reconstructed_heldout_accuracy": accuracy,
        }
    )
    rows = _state_rows(
        seed, "initialization", initial, test_data, routes, condition_list, permutations
    )
    rows.extend(
        _state_rows(
            seed, "trained", fitted, test_data, routes, condition_list, permutations
        )
    )
    return rows, endpoints


def bootstrap(values: np.ndarray, seed: int, draws: int) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    sampled = rng.choice(values, size=(int(draws), len(values)), replace=True).mean(axis=1)
    low, high = np.quantile(sampled, [0.025, 0.975])
    return float(values.mean()), float(low), float(high)


def summarize(frame: pd.DataFrame, draws: int) -> pd.DataFrame:
    rows = []
    metrics = [
        "address_capture",
        "coefficient_residual_energy",
        "realized_field_score",
    ]
    keys = ["state", "architecture", "feedback_family", "budget_k"]
    for group_index, (key, part) in enumerate(frame.groupby(keys, sort=True)):
        row = dict(zip(keys, key))
        row["n_seeds"] = int(part.seed.nunique())
        for metric_index, metric in enumerate(metrics):
            mean, low, high = bootstrap(
                part[metric].to_numpy(float),
                5_810_000 + 100 * group_index + metric_index,
                draws,
            )
            row[f"mean_{metric}"] = mean
            row[f"ci95_low_{metric}"] = low
            row[f"ci95_high_{metric}"] = high
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    archived_summary = json.loads((ARCHIVED / "summary.json").read_text(encoding="utf-8"))
    hashes = {
        "simulation_script_sha256": digest(SIMULATION_PATH),
        "config_sha256": digest(CONFIG_PATH),
    }
    if hashes["simulation_script_sha256"] != archived_summary["script_sha256"]:
        raise SystemExit("frozen simulation script no longer matches archived hash")
    if hashes["config_sha256"] != archived_summary["config_sha256"]:
        raise SystemExit("frozen configuration no longer matches archived hash")

    all_rows: list[dict] = []
    endpoints: list[pd.DataFrame] = []
    for seed in cfg["confirmatory_seeds"]:
        rows, reconstructed = reconstruct_seed(int(seed), cfg)
        all_rows.extend(rows)
        endpoints.append(reconstructed)
    frame = pd.DataFrame(all_rows)
    reconstructed = pd.concat(endpoints, ignore_index=True)
    archived = pd.read_csv(ARCHIVED / "seed_outcomes.csv")
    comparison = archived.merge(
        reconstructed,
        on=["seed", "condition_id"],
        validate="one_to_one",
    )
    maximum_loss_error = float(
        np.max(np.abs(comparison.heldout_loss - comparison.reconstructed_heldout_loss))
    )
    maximum_accuracy_error = float(
        np.max(
            np.abs(
                comparison.heldout_accuracy - comparison.reconstructed_heldout_accuracy
            )
        )
    )
    maximum_decomposition_error = float(frame.decomposition_error.max())
    gates = {
        "expected_rows": 2 * len(archived),
        "observed_rows": len(frame),
        "maximum_endpoint_loss_error": maximum_loss_error,
        "maximum_endpoint_accuracy_error": maximum_accuracy_error,
        "maximum_pythagorean_decomposition_error": maximum_decomposition_error,
    }
    gates["passed"] = bool(
        len(frame) == 2 * len(archived)
        # Archived endpoints were rounded with ``%.10g``.  These tolerances
        # are one order tighter than the largest possible source-table unit.
        and maximum_loss_error <= 1e-8
        and maximum_accuracy_error <= 1e-9
        and maximum_decomposition_error <= 1e-8
    )
    if not gates["passed"]:
        raise SystemExit(f"reconstruction or decomposition gate failed: {gates}")

    merged = frame.merge(
        archived[["seed", "condition_id", "heldout_accuracy"]],
        on=["seed", "condition_id"],
        validate="many_to_one",
    )
    trained = merged[merged.state.eq("trained")]
    seed_condition = trained.groupby(
        ["seed", "feedback_family", "budget_k"], as_index=False
    ).agg(address_capture=("address_capture", "mean"), heldout_accuracy=("heldout_accuracy", "mean"))
    restricted_families = {
        "correct_ancestry_subtrees",
        "within_neuron_route_derangement",
        "depth_interleaved_bins",
        "random_sparse_matched",
        "random_rank_k",
        "learned_rank_k_upper_bound",
    }
    correlation_rows = []
    for subset_name, subset in {
        "all_conditions": seed_condition,
        "restricted_k_lt_8": seed_condition[
            seed_condition.feedback_family.isin(restricted_families)
            & seed_condition.budget_k.lt(8)
        ],
    }.items():
        for seed, part in subset.groupby("seed", sort=True):
            correlation_rows.append(
                {
                    "subset": subset_name,
                    "seed": int(seed),
                    "n_conditions": len(part),
                    "spearman_rho": float(
                        spearmanr(
                            part.address_capture.to_numpy(float),
                            part.heldout_accuracy.to_numpy(float),
                        ).statistic
                    ),
                }
            )
    seed_correlations = pd.DataFrame(correlation_rows)
    association = {}
    for subset_name, part in seed_correlations.groupby("subset", sort=True):
        values = part.spearman_rho.to_numpy(float)
        mean, low, high = bootstrap(
            values,
            5_890_000 + len(association),
            int(cfg["analysis"]["bootstrap_draws"]),
        )
        association[subset_name] = {
            "independent_unit": "simulation seed",
            "n_seeds": len(values),
            "mean_within_seed_spearman_rho": mean,
            "ci95_low": low,
            "ci95_high": high,
            "positive_seeds": int(np.sum(values > 0)),
            "wilcoxon_p_two_sided": float(
                wilcoxon(values, alternative="two-sided").pvalue
            ),
            "scope": "post-hoc association; not an independent prediction test",
        }

    OUTPUT.mkdir(parents=True, exist_ok=True)
    merged.to_csv(OUTPUT / "seed_state_residuals.csv", index=False, float_format="%.10g")
    summarize(frame, int(cfg["analysis"]["bootstrap_draws"])).to_csv(
        OUTPUT / "condition_summary.csv", index=False, float_format="%.10g"
    )
    comparison[[
        "seed",
        "condition_id",
        "heldout_loss",
        "reconstructed_heldout_loss",
        "heldout_accuracy",
        "reconstructed_heldout_accuracy",
    ]].to_csv(OUTPUT / "reconstruction_audit.csv", index=False, float_format="%.10g")
    seed_correlations.to_csv(
        OUTPUT / "seed_level_associations.csv", index=False, float_format="%.10g"
    )
    payload = {
        "status": "complete_posthoc_diagnostic",
        "n_seeds": int(frame.seed.nunique()),
        "n_trained_fits_reconstructed": len(archived),
        "hashes": hashes,
        "gates": gates,
        "trained_capture_accuracy_association": association,
        "claim_boundary": (
            "This deterministic reconstruction measures per-example weighted address "
            "residuals in the existing synthetic factorial. It is a post-hoc mechanism "
            "diagnostic, not a new confirmatory task or evidence for a dendrite-exclusive operation."
        ),
    }
    (OUTPUT / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
