#!/usr/bin/env python3
"""Generate the frozen H=4 physical-depth by hierarchy experiment."""

from __future__ import annotations

import argparse
import copy
import importlib.util
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parents[1]
REFERENCE_GENERATOR = ROOT / "scripts" / "generate_nonlinear_physical_depth_confirmatory.py"
OUTPUT = ROOT / "configs" / "physical_depth_h4_factorial"
RUNS = Path("physical_depth_h4_runs")
SEEDS = list(range(10400, 10410))
MORPHOLOGIES = [[8], [2, 3], [2, 1, 2], [1, 1, 2, 2]]
INVENTORY = [4, 2, 1, 1]
RANGES = {
    "aligned": [[0, 16], [16, 32], [32, 48], [48, 64]],
    "rewired_tree": [[48, 64], [32, 48], [16, 32], [0, 16]],
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


def _reference_config(
    *, regime: str, strategy: str = "standard", mechanism: str = "shunting"
) -> dict[str, Any]:
    module = _reference_module()
    _path, content = module._one(  # noqa: SLF001 - frozen shared recipe
        regime=regime,
        strategy=strategy,
        mechanism=mechanism,
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


def _finish(
    config: dict[str, Any], *, run_name: str, expected: int, experiment: str
) -> str:
    config["output_dir"] = RUNS.as_posix()
    config["experiment_settings"] = {
        "seeds_per_condition": len(SEEDS),
        "base_seed": SEEDS[0],
        "use_array_jobs": True,
    }
    config["base_config"]["outputs"]["run_name"] = run_name
    config["outputs"]["run_name"] = run_name
    config["slurm_config"]["run_name"] = run_name
    config["slurm_config"]["max_concurrent_jobs"] = 10
    config["sweep_contract"] = {
        "status": "prospective_h4_factorial",
        "frozen_date": "2026-08-19",
        "outcomes_unopened_at_freeze": True,
        "no_hyperparameter_retuning": True,
        "experiment": experiment,
        "seeds": SEEDS,
        "primary_inferential_unit": "training_seed",
        "expected_config_count": expected,
    }
    return OmegaConf.to_yaml(OmegaConf.create(config))


def _config(
    *,
    regime: str,
    strategy: str,
    grouped_point: bool = False,
    mechanism: str = "shunting",
) -> str:
    config = _reference_config(
        regime=regime,
        strategy=strategy,
        mechanism=mechanism,
    )
    dataset = config["base_config"]["data"]["dataset_params"][
        "hierarchical_gain_load"
    ]
    dataset["n_levels"] = 4
    for pathway in ("ee", "ie"):
        spec = _pathways(config)[pathway]
        spec["inventory_counts"] = copy.deepcopy(INVENTORY)
        spec["feature_ranges"] = copy.deepcopy(RANGES[regime])

    population = _population(config)
    population["branch_factors"] = copy.deepcopy(MORPHOLOGIES[0])
    if grouped_point:
        population.setdefault("population", {})[
            "cross_level_mode"
        ] = "parallel_readout"

    sweep: dict[str, Any] = {
        "model.core.population_network.layers.0.populations.0.branch_factors": (
            copy.deepcopy(MORPHOLOGIES)
        )
    }
    conditions = len(MORPHOLOGIES)
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
        run_name=(
            f"journal_h4_{regime}_{architecture}_{mechanism}_{method}"
        ),
        expected=conditions * len(SEEDS),
        experiment=(
            f"h4_{regime}_{architecture}_{mechanism}_{method}"
        ),
    )


def render() -> dict[Path, str]:
    specs = [
        ("aligned", "standard", False, "shunting"),
        ("rewired_tree", "standard", False, "shunting"),
        ("aligned", "standard", True, "shunting"),
        ("rewired_tree", "standard", True, "shunting"),
        ("aligned", "local_ca", False, "shunting"),
        ("rewired_tree", "local_ca", False, "shunting"),
        ("aligned", "standard", False, "additive"),
    ]
    rendered: dict[Path, str] = {}
    for regime, strategy, grouped_point, mechanism in specs:
        architecture = "grouped_point" if grouped_point else "serial"
        method = "bp" if strategy == "standard" else "local3f"
        path = OUTPUT / (
            f"h4_{regime}_{architecture}_{mechanism}_{method}.yaml"
        )
        rendered[path] = _config(
            regime=regime,
            strategy=strategy,
            grouped_point=grouped_point,
            mechanism=mechanism,
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

