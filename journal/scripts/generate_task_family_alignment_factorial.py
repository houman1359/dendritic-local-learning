#!/usr/bin/env python3
"""Generate the frozen fixed-depth task-family by alignment factorial."""

from __future__ import annotations

import argparse
import copy
import importlib.util
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "scripts" / "generate_nonlinear_physical_depth_confirmatory.py"
OUTPUT = ROOT / "configs" / "task_family_alignment"
PROJECT_B_ROOT = Path(
    "/n/holylfs06/LABS/kempner_project_b/Lab/dendritic/HS/LOCAL_LEARNING/"
    "journal_extension_20260820"
)
RUNS = PROJECT_B_ROOT / "sweep_runs" / "task_family_alignment"
SEEDS = list(range(10500, 10510))
ALPHAS = [0.0, 0.5, 1.0]
MORPHOLOGY = [2, 1, 2]

FAMILIES: dict[str, dict[str, Any]] = {
    "nested_factor": {
        "nuisance_layout": "factorized_sensors",
        "gain_structure": "hierarchical",
        "signal_mode": "e_only",
        "signal_profile": "distal_only",
        "i_signal_delta": 0.0,
    },
    "flat_factor": {
        "nuisance_layout": "factorized_sensors",
        "gain_structure": "flat",
        "signal_mode": "e_only",
        "signal_profile": "distal_only",
        "i_signal_delta": 0.0,
    },
    "local_ratio": {
        "nuisance_layout": "paired_cumulative",
        "gain_structure": "hierarchical",
        "signal_mode": "e_only",
        "signal_profile": "distal_only",
        "i_signal_delta": 0.0,
    },
}


def _reference_module():
    spec = importlib.util.spec_from_file_location("physical_reference", REFERENCE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {REFERENCE}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _base(*, strategy: str) -> dict[str, Any]:
    module = _reference_module()
    _path, content = module._one(  # noqa: SLF001 - frozen shared recipe
        regime="aligned",
        strategy=strategy,
        mechanism="shunting",
    )
    import yaml

    config = yaml.safe_load(content)
    if not isinstance(config, dict):
        raise TypeError("Expected generated YAML mapping")
    config["output_dir"] = RUNS.as_posix()
    config["experiment_settings"] = {
        "seeds_per_condition": len(SEEDS),
        "base_seed": SEEDS[0],
        "use_array_jobs": True,
    }
    config["base_config"]["wandb"]["use_wandb"] = False
    config["slurm_config"].update(
        {
            "account": "kempner_bsabatini_lab",
            "partition": "kempner_h100_priority",
            "qos": "kemp_gpu16_id38",
            "time": "02:00:00",
            "max_concurrent_jobs": 4,
            # Keep the publication config portable. The exact pinned runtime
            # used for the frozen execution is recorded in the execution log.
            "conda_env_path": "",
        }
    )
    population = config["base_config"]["model"]["core"][
        "population_network"
    ]["layers"][0]["populations"][0]
    population["branch_factors"] = copy.deepcopy(MORPHOLOGY)
    return config


def _one(*, family: str, architecture: str, credit: str) -> str:
    if family not in FAMILIES:
        raise ValueError(family)
    if architecture not in {"serial", "grouped_point"}:
        raise ValueError(architecture)
    if credit not in {"bp", "local3f"}:
        raise ValueError(credit)

    strategy = "standard" if credit == "bp" else "local_ca"
    config = _base(strategy=strategy)
    dataset = config["base_config"]["data"]["dataset_params"][
        "hierarchical_gain_load"
    ]
    dataset.update(copy.deepcopy(FAMILIES[family]))

    population = config["base_config"]["model"]["core"][
        "population_network"
    ]["layers"][0]["populations"][0]
    if architecture == "grouped_point":
        population.setdefault("population", {})[
            "cross_level_mode"
        ] = "parallel_readout"
    else:
        population.setdefault("population", {}).pop("cross_level_mode", None)

    if credit == "local3f":
        learning = config["base_config"]["training"]["main"][
            "learning_strategy_config"
        ]
        learning["error_broadcast_mode"] = "path_transport"

    alpha_key = (
        "data.dataset_params.hierarchical_gain_load.sensor_alignment_alpha"
    )
    config["sweep_config"] = {alpha_key: ALPHAS}
    run_name = f"journal_taskfamily_{family}_{architecture}_{credit}"
    config["base_config"]["outputs"]["run_name"] = run_name
    config["outputs"]["run_name"] = run_name
    config["slurm_config"]["run_name"] = run_name
    config["sweep_contract"] = {
        "status": "prospective_fixed_depth_task_family_factorial",
        "frozen_date": "2026-08-20",
        "outcomes_unopened_at_freeze": True,
        "no_hyperparameter_retuning": True,
        "external_tracking": "disabled; no W&B import in pinned runtime",
        "storage_filesystem": "kempner_project_b on holylfs06",
        "family": family,
        "architecture": architecture,
        "credit": credit,
        "fixed_depth": 3,
        "fixed_branch_factors": MORPHOLOGY,
        "alignment_alpha": ALPHAS,
        "seeds": SEEDS,
        "primary_inferential_unit": "training_seed",
        "expected_config_count": len(ALPHAS) * len(SEEDS),
    }
    return OmegaConf.to_yaml(OmegaConf.create(config))


def render() -> dict[Path, str]:
    rendered: dict[Path, str] = {}
    for family in FAMILIES:
        for architecture in ("serial", "grouped_point"):
            for credit in ("bp", "local3f"):
                path = OUTPUT / f"{family}_{architecture}_{credit}.yaml"
                rendered[path] = _one(
                    family=family,
                    architecture=architecture,
                    credit=credit,
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
