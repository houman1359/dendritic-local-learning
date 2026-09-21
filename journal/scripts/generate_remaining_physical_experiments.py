#!/usr/bin/env python3
"""Generate the grouped-point and second-hierarchy physical-depth sweeps."""

from __future__ import annotations

import argparse
import copy
import importlib.util
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parents[1]
REFERENCE_GENERATOR = ROOT / "scripts" / "generate_nonlinear_physical_depth_confirmatory.py"
OUTPUT = ROOT / "configs" / "remaining_physical_experiments"
RUNS = Path("remaining_physical_experiment_runs")
H3_SEEDS = list(range(10200, 10210))
H2_SEEDS = list(range(10300, 10310))
H2_MORPHOLOGIES = [[8], [4, 1]]
H2_INVENTORY = [4, 4]
H2_RANGES = {
    "aligned": [[0, 32], [32, 64]],
    "rewired_tree": [[32, 64], [0, 32]],
}


def _reference_module():
    spec = importlib.util.spec_from_file_location(
        "physical_depth_confirmatory", REFERENCE_GENERATOR
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {REFERENCE_GENERATOR}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _reference_config(regime: str, strategy: str = "standard") -> dict[str, Any]:
    module = _reference_module()
    _path, content = module._one(  # noqa: SLF001 - shared frozen recipe
        regime=regime,
        strategy=strategy,
        mechanism="shunting",
    )
    import yaml

    value = yaml.safe_load(content)
    if not isinstance(value, dict):
        raise TypeError("Expected generated YAML mapping")
    return value


def _population(config: dict[str, Any]) -> dict[str, Any]:
    return config["base_config"]["model"]["core"]["population_network"][
        "layers"
    ][0]["populations"][0]


def _pathways(config: dict[str, Any]) -> dict[str, Any]:
    return config["base_config"]["model"]["core"]["population_network"][
        "layers"
    ][0]["population_defaults"]["structured_connectivity"]["pathways"]


def _set_seeds(config: dict[str, Any], seeds: list[int]) -> None:
    config["experiment_settings"] = {
        "seeds_per_condition": len(seeds),
        "base_seed": seeds[0],
        "use_array_jobs": True,
    }


def _finish(
    config: dict[str, Any],
    *,
    run_name: str,
    expected: int,
    experiment: str,
    seeds: list[int],
) -> str:
    # Keep the archived recipes portable. The sweep launcher resolves this
    # path from the journal working directory; cluster runs may override it.
    config["output_dir"] = RUNS.as_posix()
    config["base_config"]["outputs"]["run_name"] = run_name
    config["outputs"]["run_name"] = run_name
    config["slurm_config"]["run_name"] = run_name
    config["slurm_config"]["max_concurrent_jobs"] = 10
    config["sweep_contract"] = {
        "status": "prospective_remaining_reviewer_experiment",
        "frozen_date": "2026-08-13",
        "outcomes_unopened_at_freeze": True,
        "reference_h3_endpoints_already_observed": True,
        "no_hyperparameter_retuning": True,
        "experiment": experiment,
        "seeds": seeds,
        "primary_inferential_unit": "training_seed",
        "expected_config_count": expected,
    }
    return OmegaConf.to_yaml(OmegaConf.create(config))


def _h3_grouped_point(regime: str) -> str:
    config = _reference_config(regime)
    _set_seeds(config, H3_SEEDS)
    population = _population(config)
    population.setdefault("population", {})["cross_level_mode"] = "parallel_readout"
    return _finish(
        config,
        run_name=f"journal_remaining_h3_{regime}_grouped_point_bp",
        expected=30,
        experiment=f"h3_grouped_point__{regime}",
        seeds=H3_SEEDS,
    )


def _h2_config(
    *,
    regime: str,
    strategy: str,
    grouped_point: bool,
) -> str:
    config = _reference_config(regime, strategy=strategy)
    _set_seeds(config, H2_SEEDS)
    dataset = config["base_config"]["data"]["dataset_params"][
        "hierarchical_gain_load"
    ]
    dataset["n_levels"] = 2
    for pathway in ("ee", "ie"):
        spec = _pathways(config)[pathway]
        spec["inventory_counts"] = copy.deepcopy(H2_INVENTORY)
        spec["feature_ranges"] = copy.deepcopy(H2_RANGES[regime])

    population = _population(config)
    population["branch_factors"] = copy.deepcopy(H2_MORPHOLOGIES[0])
    if grouped_point:
        population.setdefault("population", {})[
            "cross_level_mode"
        ] = "parallel_readout"

    sweep: dict[str, Any] = {
        "model.core.population_network.layers.0.populations.0.branch_factors": (
            copy.deepcopy(H2_MORPHOLOGIES)
        )
    }
    conditions = len(H2_MORPHOLOGIES)
    if strategy == "local_ca":
        sweep["training.main.learning_strategy_config.error_broadcast_mode"] = [
            "per_soma_shared",
            "path_transport",
        ]
        conditions *= 2
    config["sweep_config"] = sweep

    architecture = "grouped_point" if grouped_point else "serial"
    method = "bp" if strategy == "standard" else "local3f"
    return _finish(
        config,
        run_name=f"journal_remaining_h2_{regime}_{architecture}_{method}",
        expected=conditions * len(H2_SEEDS),
        experiment=f"h2_{architecture}_{method}__{regime}",
        seeds=H2_SEEDS,
    )


def render() -> dict[Path, str]:
    rendered: dict[Path, str] = {}
    for regime in ("aligned", "rewired_tree"):
        rendered[OUTPUT / f"h3_{regime}_grouped_point_bp.yaml"] = (
            _h3_grouped_point(regime)
        )
        rendered[OUTPUT / f"h2_{regime}_serial_bp.yaml"] = _h2_config(
            regime=regime,
            strategy="standard",
            grouped_point=False,
        )
        rendered[OUTPUT / f"h2_{regime}_grouped_point_bp.yaml"] = _h2_config(
            regime=regime,
            strategy="standard",
            grouped_point=True,
        )
        rendered[OUTPUT / f"h2_{regime}_serial_local3f.yaml"] = _h2_config(
            regime=regime,
            strategy="local_ca",
            grouped_point=False,
        )
    return rendered


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    expected = render()
    if args.check:
        mismatches = [
            path
            for path, content in expected.items()
            if not path.is_file() or path.read_text(encoding="utf-8") != content
        ]
        if mismatches:
            raise SystemExit(
                "Generated config mismatch: " + ", ".join(map(str, mismatches))
            )
        return
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for path, content in expected.items():
        path.write_text(content, encoding="utf-8")
        print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
