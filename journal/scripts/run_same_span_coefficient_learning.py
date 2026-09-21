#!/usr/bin/env python3
"""Run the frozen same-span noisy route-coefficient learning experiment."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "credit_phase_theory" / "same_span_learning_confirmatory.json"
CANARY = ROOT / "analysis" / "same_span_coefficient_learning_canary.json"
OUTPUT = ROOT / "source_data" / "same_span_coefficient_learning"


def load_phase_module():
    path = ROOT / "code" / "theory" / "credit_phase.py"
    spec = importlib.util.spec_from_file_location("credit_phase_same_span", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PHASE = load_phase_module()


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def raw_nested_dictionary(n: int, max_depth: int) -> np.ndarray:
    columns = []
    for level in range(max_depth + 1):
        width = n // (2**level)
        for block in range(2**level):
            column = np.zeros(n, dtype=float)
            column[block * width : (block + 1) * width] = 1.0
            columns.append(column)
    return np.column_stack(columns)


def dictionaries(cfg: dict) -> dict[str, np.ndarray]:
    n = int(cfg["n_leaves"])
    depth = int(cfg["route_depth"])
    basis, levels = PHASE.tree_haar_basis(n)
    haar = basis[:, levels <= depth]
    raw = raw_nested_dictionary(n, depth)
    log_range = float(cfg["scaled_nested_log_range"])
    scales = np.exp(np.linspace(-log_range, log_range, raw.shape[1]))
    scaled = raw @ np.diag(scales)
    return {
        "tree_haar": haar,
        "raw_nested_indicators": raw,
        "static_gain_scaled_nested": scaled,
    }


def train_one(
    dictionary: np.ndarray,
    target: np.ndarray,
    observations: np.ndarray,
    optimizer: str,
    checkpoints: set[int],
    step_fraction: float,
) -> tuple[list[dict], dict[int, np.ndarray]]:
    gram = dictionary.T @ dictionary
    positive = np.linalg.eigvalsh(gram)
    positive = positive[positive > 1e-12]
    lambda_max = float(positive.max())
    condition = float(positive.max() / positive.min())
    preconditioner = np.linalg.pinv(gram, rcond=1e-12)
    coefficients = np.zeros(dictionary.shape[1], dtype=float)
    rows = []
    fields = {}
    for iteration, observation in enumerate(observations, start=1):
        field = dictionary @ coefficients
        gradient = dictionary.T @ (field - observation)
        if optimizer == "vanilla_local":
            coefficients -= (step_fraction / lambda_max) * gradient
        elif optimizer == "gram_preconditioned_control":
            coefficients -= step_fraction * (preconditioner @ gradient)
        else:
            raise ValueError(f"unknown optimizer {optimizer}")
        if iteration in checkpoints:
            field = dictionary @ coefficients
            fields[iteration] = field.copy()
            rows.append(
                {
                    "checkpoint": iteration,
                    "population_loss": 0.5 * float(np.sum((field - target) ** 2)),
                    "field_estimation_error": float(np.sum((field - target) ** 2)),
                    "coefficient_norm": float(np.linalg.norm(coefficients)),
                    "gram_condition_number": condition,
                    "gram_lambda_max": lambda_max,
                }
            )
    return rows, fields


def run_seed(seed: int, cfg: dict) -> tuple[list[dict], dict[str, float]]:
    rng = np.random.default_rng(seed)
    route_dictionaries = dictionaries(cfg)
    reference = route_dictionaries["tree_haar"]
    target_coefficients = rng.normal(size=reference.shape[1])
    target = reference @ target_coefficients
    target /= max(np.linalg.norm(target), 1e-30)
    weights = np.ones(len(target))
    projectors = {
        name: PHASE.weighted_projector(dictionary, weights)
        for name, dictionary in route_dictionaries.items()
    }
    reference_projector = projectors["tree_haar"]
    maximum_projector_difference = max(
        float(np.max(np.abs(projector - reference_projector)))
        for projector in projectors.values()
    )
    maximum_address_residual = max(
        float(np.sum((target - projector @ target) ** 2))
        for projector in projectors.values()
    )

    rows = []
    maximum_preconditioned_difference = 0.0
    maximum_haar_optimizer_difference = 0.0
    checkpoints = {int(value) for value in cfg["checkpoints"]}
    for sample_size in cfg["effective_sample_sizes"]:
        n_eff = int(sample_size)
        observation_rng = np.random.default_rng(seed + 10_000_000 + n_eff)
        noises = observation_rng.normal(
            scale=float(cfg["observation_noise_sd"]) / np.sqrt(n_eff),
            size=(int(cfg["iterations"]), len(target)),
        )
        observations = target[None, :] + noises
        fields_by_condition = {}
        for parameterization in cfg["parameterizations"]:
            dictionary = route_dictionaries[parameterization]
            diagnostics = PHASE.route_gram_diagnostics(dictionary, weights)
            for optimizer in cfg["optimizers"]:
                condition_rows, fields = train_one(
                    dictionary,
                    target,
                    observations,
                    optimizer,
                    checkpoints,
                    float(cfg["normalized_step_fraction"]),
                )
                fields_by_condition[(parameterization, optimizer)] = fields
                for row in condition_rows:
                    rows.append(
                        {
                            "seed": seed,
                            "effective_sample_size": n_eff,
                            "parameterization": parameterization,
                            "optimizer": optimizer,
                            "route_rank": int(round(diagnostics["rank"])),
                            "address_residual": float(
                                np.sum(
                                    (
                                        target
                                        - projectors[parameterization] @ target
                                    )
                                    ** 2
                                )
                            ),
                            **row,
                        }
                    )
        for checkpoint in checkpoints:
            preconditioned = [
                fields_by_condition[(name, "gram_preconditioned_control")][checkpoint]
                for name in cfg["parameterizations"]
            ]
            for field in preconditioned[1:]:
                maximum_preconditioned_difference = max(
                    maximum_preconditioned_difference,
                    float(np.max(np.abs(field - preconditioned[0]))),
                )
            maximum_haar_optimizer_difference = max(
                maximum_haar_optimizer_difference,
                float(
                    np.max(
                        np.abs(
                            fields_by_condition[("tree_haar", "vanilla_local")][checkpoint]
                            - fields_by_condition[(
                                "tree_haar", "gram_preconditioned_control"
                            )][checkpoint]
                        )
                    )
                ),
            )
    gates = {
        "maximum_projector_difference": maximum_projector_difference,
        "maximum_initial_address_residual": maximum_address_residual,
        "maximum_preconditioned_trajectory_difference": maximum_preconditioned_difference,
        "maximum_haar_vanilla_vs_preconditioned_difference": maximum_haar_optimizer_difference,
    }
    return rows, gates


def bootstrap(values: np.ndarray, seed: int, draws: int) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = rng.choice(values, size=(draws, len(values)), replace=True).mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(values.mean()), float(low), float(high)


def summarize(frame: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    rows = []
    draws = int(cfg["analysis"]["bootstrap_draws"])
    groups = ["effective_sample_size", "checkpoint", "parameterization", "optimizer"]
    for group_index, (keys, part) in enumerate(frame.groupby(groups, sort=True)):
        row = dict(zip(groups, keys))
        row["n_seeds"] = int(part.seed.nunique())
        row["gram_condition_number"] = float(part.gram_condition_number.iloc[0])
        for metric_index, metric in enumerate(["population_loss", "field_estimation_error"]):
            mean, low, high = bootstrap(
                part[metric].to_numpy(float),
                10_200_000 + 100 * group_index + metric_index,
                draws,
            )
            row[f"mean_{metric}"] = mean
            row[f"ci95_low_{metric}"] = low
            row[f"ci95_high_{metric}"] = high
        rows.append(row)
    return pd.DataFrame(rows)


def paired(
    frame: pd.DataFrame,
    sample_size: int,
    left: tuple[str, str],
    right: tuple[str, str],
    cfg: dict,
    seed: int,
) -> dict:
    final = frame[
        frame.effective_sample_size.eq(sample_size)
        & frame.checkpoint.eq(int(cfg["iterations"]))
    ].copy()
    final["condition"] = final.parameterization + " / " + final.optimizer
    wide = final.pivot(index="seed", columns="condition", values="population_loss")
    left_name = " / ".join(left)
    right_name = " / ".join(right)
    values = (wide[left_name] - wide[right_name]).to_numpy(float)
    mean, low, high = bootstrap(values, seed, int(cfg["analysis"]["bootstrap_draws"]))
    return {
        "effective_sample_size": sample_size,
        "left_minus_right": f"{left_name} - {right_name}",
        "n_pairs": len(values),
        "mean_loss_difference": mean,
        "ci95_low": low,
        "ci95_high": high,
        "positive_pairs": int(np.sum(values > 0)),
        "ties": int(np.sum(np.isclose(values, 0))),
        "wilcoxon_p_two_sided": 1.0
        if np.allclose(values, 0)
        else float(wilcoxon(values, zero_method="wilcox").pvalue),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["canary", "confirmatory"], required=True)
    args = parser.parse_args()
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    seeds = [int(value) for value in cfg[f"{args.phase}_seeds"]]
    all_rows, all_gates = [], []
    for seed in seeds:
        rows, gates = run_seed(seed, cfg)
        all_rows.extend(rows)
        all_gates.append(gates)
    frame = pd.DataFrame(all_rows)
    aggregate_gates = {
        key: float(max(gate[key] for gate in all_gates))
        for key in all_gates[0]
    }
    limits = cfg["gates"]
    aggregate_gates["passed"] = all(
        aggregate_gates[key] <= float(limits[key]) for key in limits
    )
    hashes = {
        "config_sha256": digest(CONFIG),
        "script_sha256": digest(Path(__file__).resolve()),
        "reference_sha256": digest(ROOT / "code" / "theory" / "credit_phase.py"),
    }
    if args.phase == "canary":
        payload = {
            "phase": "artifact_only_canary",
            **hashes,
            "numerical_gates": aggregate_gates,
            "outcomes_are_not_confirmatory": True,
        }
        CANARY.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(payload, indent=2))
        if not aggregate_gates["passed"]:
            raise SystemExit("canary failed")
        return
    if not CANARY.is_file():
        raise SystemExit("passing canary required")
    canary = json.loads(CANARY.read_text(encoding="utf-8"))
    if not canary["numerical_gates"]["passed"] or any(
        canary[key] != value for key, value in hashes.items()
    ):
        raise SystemExit("canary failed or code/configuration/reference changed")
    expected = (
        len(seeds)
        * len(cfg["effective_sample_sizes"])
        * len(cfg["parameterizations"])
        * len(cfg["optimizers"])
        * len(cfg["checkpoints"])
    )
    if len(frame) != expected or frame.duplicated(
        ["seed", "effective_sample_size", "parameterization", "optimizer", "checkpoint"]
    ).any():
        raise SystemExit(f"incomplete same-span factorial: {len(frame)} != {expected}")
    if not np.isfinite(frame.select_dtypes(include=[np.number])).all().all():
        raise SystemExit("non-finite outcome")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUTPUT / "seed_trajectories.csv", index=False, float_format="%.10g")
    summarize(frame, cfg).to_csv(
        OUTPUT / "condition_summary.csv", index=False, float_format="%.10g"
    )
    contrasts = []
    for index, sample_size in enumerate(cfg["effective_sample_sizes"]):
        for comparison_index, (left, right) in enumerate(
            [
                (("raw_nested_indicators", "vanilla_local"), ("tree_haar", "vanilla_local")),
                (("static_gain_scaled_nested", "vanilla_local"), ("tree_haar", "vanilla_local")),
                (("static_gain_scaled_nested", "vanilla_local"), ("static_gain_scaled_nested", "gram_preconditioned_control")),
            ]
        ):
            contrasts.append(
                paired(
                    frame,
                    int(sample_size),
                    left,
                    right,
                    cfg,
                    10_300_000 + 100 * index + comparison_index,
                )
            )
    pd.DataFrame(contrasts).to_csv(
        OUTPUT / "paired_contrasts.csv", index=False, float_format="%.10g"
    )
    summary = {
        "study": cfg["study"],
        "status": "complete_confirmatory",
        "n_seeds": len(seeds),
        "row_count": len(frame),
        "python": platform.python_version(),
        "numpy": np.__version__,
        **hashes,
        "numerical_gates": aggregate_gates,
        "scope_boundary": cfg["scope_boundary"],
    }
    (OUTPUT / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
