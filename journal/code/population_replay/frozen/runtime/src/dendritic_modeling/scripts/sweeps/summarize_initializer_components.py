#!/usr/bin/env python3
"""Summarize a validation-only analytical/occupancy initializer factorial."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

POLICY_COMPONENTS = {
    "analytical": ("analytical", "analytical"),
    "analytical_slope_occupancy_center": ("analytical", "occupancy"),
    "occupancy_slope_analytical_center": ("occupancy", "analytical"),
    "occupancy_quantile": ("occupancy", "occupancy"),
}
DATA_DRIVEN_POLICIES = set(POLICY_COMPONENTS) - {"analytical"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def nested(document: dict[str, Any], *keys: str) -> Any:
    current: Any = document
    for key in keys:
        current = current[key]
    return current


def _condition_metadata(
    config: dict[str, Any], layer: dict[str, Any]
) -> dict[str, Any]:
    dataset = str(nested(config, "data", "dataset_name"))
    morphology = json.dumps(
        layer["populations"][0]["branch_factors"], separators=(",", ":")
    )
    session = ""
    running_state = ""
    if dataset == "stringer_v1":
        params = nested(config, "data", "dataset_params", "stringer_v1")
        session = str(params["session_id"])
        running_state = str(bool(params["running_residualize"])).lower()
    condition = "|".join(
        value for value in (dataset, morphology, session, running_state) if value != ""
    )
    return {
        "dataset": dataset,
        "morphology": morphology,
        "session_id": session,
        "running_residualize": running_state,
        "condition": condition,
    }


def load_row(sweep_dir: Path, index: int) -> dict[str, Any] | None:
    config_path = sweep_dir / "configs" / f"unified_config_{index}.yaml"
    result_dir = sweep_dir / "results" / f"config_{index}"
    performance_path = result_dir / "performance" / "final.json"
    training_path = result_dir / "training_summary.json"
    init_gate_path = result_dir / "init_gate_stats.json"
    required = (config_path, performance_path, training_path, init_gate_path)
    if not all(path.is_file() for path in required):
        return None

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    performance = json.loads(performance_path.read_text(encoding="utf-8"))
    training = json.loads(training_path.read_text(encoding="utf-8"))
    layer = nested(config, "model", "core", "population_network", "layers")[0]
    defaults = layer["population_defaults"]
    policy = str(defaults["reactivation_init_policy"])
    if policy not in POLICY_COMPONENTS:
        raise ValueError(f"config {index} has non-factorial policy {policy!r}")

    calibration_path = result_dir / "reactivation_calibration.json"
    post_gate_path = result_dir / "post_calibration_gate_stats.json"
    calibration: dict[str, Any] | None = None
    if policy in DATA_DRIVEN_POLICIES:
        if not calibration_path.is_file() or not post_gate_path.is_file():
            return None
        calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
        if calibration.get("policies") != [policy]:
            raise ValueError(f"config {index} executed the wrong policy")
        layers = calibration.get("layers", {})
        if not layers:
            raise ValueError(f"config {index} has no calibrated layers")
        if policy in {
            "analytical_slope_occupancy_center",
            "occupancy_slope_analytical_center",
        } and {stats.get("component_policy") for stats in layers.values()} != {policy}:
            raise ValueError(f"config {index} has inconsistent component diagnostics")
    else:
        if calibration_path.exists() or post_gate_path.exists():
            raise ValueError(f"analytical config {index} unexpectedly calibrated")
        layers = {}

    valid_accuracy = float(performance["accuracy"]["valid"])
    valid_loglik = float(performance["categorical_loglikelihood"]["valid"])
    if not np.isfinite(valid_accuracy) or not np.isfinite(valid_loglik):
        raise ValueError(f"config {index} has a non-finite validation endpoint")
    slope_source, center_source = POLICY_COMPONENTS[policy]
    row = {
        "config_id": index,
        "seed": int(nested(config, "experiment", "seed")),
        "mechanism": "shunting" if defaults["use_shunting"] else "additive",
        "policy": policy,
        "slope_source": slope_source,
        "center_source": center_source,
        "valid_accuracy": valid_accuracy,
        "valid_categorical_loglikelihood": valid_loglik,
        "best_epoch": int(training["best_epoch"]),
        "calibration_converged": bool(
            calibration.get("converged", False) if calibration else True
        ),
        "calibration_iterations": int(
            calibration.get("iterations_completed", 0) if calibration else 0
        ),
        "calibration_reverted_layers": int(
            sum(bool(stats.get("calibration_reverted")) for stats in layers.values())
        ),
        "occupancy_fit_reverted_layers": int(
            sum(bool(stats.get("occupancy_fit_reverted")) for stats in layers.values())
        ),
        "config_sha256": sha256(config_path),
        "performance_sha256": sha256(performance_path),
        "init_gate_sha256": sha256(init_gate_path),
        "calibration_sha256": sha256(calibration_path) if calibration else "",
        "post_gate_sha256": sha256(post_gate_path) if calibration else "",
    }
    row.update(_condition_metadata(config, layer))
    return row


def summarize(runs: pd.DataFrame) -> pd.DataFrame:
    keys = ["mechanism", "policy", "slope_source", "center_source"]
    result = (
        runs.groupby(keys, as_index=False)
        .agg(
            valid_accuracy_mean=("valid_accuracy", "mean"),
            valid_accuracy_sd=("valid_accuracy", "std"),
            valid_loglik_mean=("valid_categorical_loglikelihood", "mean"),
            valid_loglik_sd=("valid_categorical_loglikelihood", "std"),
            n_runs=("config_id", "size"),
            n_seeds=("seed", "nunique"),
            n_conditions=("condition", "nunique"),
            calibration_convergence_rate=("calibration_converged", "mean"),
            calibration_iterations_max=("calibration_iterations", "max"),
            calibration_reverted_layers=("calibration_reverted_layers", "sum"),
            occupancy_fit_reverted_layers=("occupancy_fit_reverted_layers", "sum"),
        )
        .sort_values(["mechanism", "policy"])
        .reset_index(drop=True)
    )
    result["valid_accuracy_sem"] = result["valid_accuracy_sd"] / np.sqrt(
        result["n_runs"]
    )
    return result


def select_per_mechanism(summary: pd.DataFrame) -> pd.DataFrame:
    ordered = summary.sort_values(
        ["mechanism", "valid_accuracy_mean", "valid_loglik_mean", "policy"],
        ascending=[True, False, False, True],
    )
    return ordered.groupby("mechanism", as_index=False).head(1).reset_index(drop=True)


def select_shared_policy(summary: pd.DataFrame) -> pd.DataFrame:
    result = (
        summary.groupby("policy", as_index=False)
        .agg(
            valid_accuracy_mean=("valid_accuracy_mean", "mean"),
            valid_loglik_mean=("valid_loglik_mean", "mean"),
            n_mechanisms=("mechanism", "nunique"),
        )
        .sort_values(
            ["valid_accuracy_mean", "valid_loglik_mean", "policy"],
            ascending=[False, False, True],
        )
        .reset_index(drop=True)
    )
    return result.head(1)


def component_effects(summary: pd.DataFrame) -> pd.DataFrame:
    """Return center, slope, and interaction contrasts for each mechanism."""
    policies = {
        "aa": "analytical",
        "ao": "analytical_slope_occupancy_center",
        "oa": "occupancy_slope_analytical_center",
        "oo": "occupancy_quantile",
    }
    rows = []
    for mechanism, group in summary.groupby("mechanism"):
        indexed = group.set_index("policy")
        if set(indexed.index) != set(policies.values()):
            raise ValueError(f"incomplete component factorial for {mechanism}")
        for metric, stem in (
            ("valid_accuracy_mean", "valid_accuracy"),
            ("valid_loglik_mean", "valid_loglik"),
        ):
            value = {
                key: float(indexed.loc[policy, metric])
                for key, policy in policies.items()
            }
            rows.append(
                {
                    "mechanism": mechanism,
                    "metric": stem,
                    "center_effect_at_analytical_slope": value["ao"] - value["aa"],
                    "center_effect_at_occupancy_slope": value["oo"] - value["oa"],
                    "slope_effect_at_analytical_center": value["oa"] - value["aa"],
                    "slope_effect_at_occupancy_center": value["oo"] - value["ao"],
                    "center_by_slope_interaction": value["oo"]
                    - value["oa"]
                    - value["ao"]
                    + value["aa"],
                }
            )
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stem", required=True)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()

    sweep_dir = args.sweep_dir.resolve()
    manifest_path = sweep_dir / "frozen_sweep_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = int(manifest["expected_config_count"])
    rows = [row for index in range(expected) if (row := load_row(sweep_dir, index))]
    if len(rows) != expected and not args.allow_incomplete:
        raise RuntimeError(f"incomplete sweep: found {len(rows)}/{expected} results")
    if not rows:
        raise RuntimeError("no completed results")

    runs = pd.DataFrame(rows).sort_values("config_id").reset_index(drop=True)
    recipes = summarize(runs)
    selected = select_per_mechanism(recipes)
    shared = select_shared_policy(recipes)
    effects = (
        component_effects(recipes)
        if set(runs["policy"]) == set(POLICY_COMPONENTS)
        else pd.DataFrame()
    )
    print(recipes.to_string(index=False))
    print("\nPer-mechanism validation selections")
    print(selected.to_string(index=False))
    print("\nSingle shared-policy validation selection")
    print(shared.to_string(index=False))
    if not effects.empty:
        print("\nCenter/slope validation contrasts")
        print(effects.to_string(index=False))

    if len(rows) != expected:
        return 0
    if not bool(runs["calibration_converged"].all()):
        raise RuntimeError("complete sweep contains non-converged calibration cells")
    counts = runs.groupby(["mechanism", "policy"]).size()
    if counts.nunique() != 1 or len(counts) != 8:
        raise RuntimeError("complete sweep is unbalanced across mechanism/policy cells")

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    tables = {
        "validation_runs": runs,
        "validation_recipes": recipes,
        "selected_per_mechanism": selected,
        "selected_shared_policy": shared,
        "component_effects": effects,
    }
    paths = {}
    for name, table in tables.items():
        path = output / f"{args.stem}_{name}.csv"
        table.to_csv(path, index=False)
        paths[name] = path
    provenance = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "selector": str(Path(__file__).resolve()),
        "selector_sha256": sha256(Path(__file__).resolve()),
        "sweep_dir": str(sweep_dir),
        "frozen_manifest_sha256": sha256(manifest_path),
        "expected_results": expected,
        "complete_results": len(runs),
        "test_metrics_used_or_written": False,
        "outputs": {
            path.name: {"sha256": sha256(path), "rows": len(tables[name])}
            for name, path in paths.items()
        },
    }
    (output / f"{args.stem}_provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
