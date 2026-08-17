#!/usr/bin/env python3
"""Generate inhibition-dose and fixed-topology prospective sweep configs."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from generate_prospective_learning_sweeps import (
    FEEDBACK_MODES,
    JOURNAL_ROOT,
    _common_config,
)


CONFIG_DIR = JOURNAL_ROOT / "configs" / "prospective_shunting_topology"
INHIBITION_DOSES = [[0], [5], [10], [20], [40]]
CANARY_DOSES = [INHIBITION_DOSES[0], INHIBITION_DOSES[-1]]


def _set_expected(config: dict[str, Any], conditions: int, phase: str) -> None:
    seeds = 10 if phase == "confirmatory" else 1
    config["experiment_settings"]["seeds_per_condition"] = seeds
    config["sweep_contract"] = {"expected_config_count": conditions * seeds}
    config["slurm_config"]["max_concurrent_jobs"] = 1


def _inhibition_config(
    *, core: str, phase: str, strategy: str
) -> dict[str, Any]:
    config = _common_config(
        core=core,
        task="noise",
        phase=phase,
        strategy=strategy,
    )
    local = strategy == "local_ca"
    doses = INHIBITION_DOSES if phase == "confirmatory" else CANARY_DOSES
    feedback = FEEDBACK_MODES if phase == "confirmatory" else [
        "per_soma",
        "path_transport",
    ]
    sweep: dict[str, Any] = {
        "model.core.connectivity.ie_synapses_per_branch_per_layer": doses,
    }
    if local:
        sweep["training.main.learning_strategy_config.error_broadcast_mode"] = feedback
    config["sweep_config"] = sweep
    _set_expected(config, len(doses) * (len(feedback) if local else 1), phase)
    base = config["base_config"]
    base["model"]["core"]["architecture"]["excitatory_branch_factors"] = [2, 2, 2]
    label = "local3f" if local else "backprop"
    base["outputs"]["run_name"] = (
        f"journal_{phase}_inhibition_dose_noise_{core}_{label}"
    )
    return config


def _fixed_indexed_sparsity() -> dict[str, Any]:
    return {
        "init_method": "xavier_normal",
        "noise_level": 0.0,
        "type": "indexed",
        "weight_norm_order": None,
        "gamma": 1.0,
        "gradient_scaling": "none",
        "indexed": {
            "seed": None,
            "index_dtype": "int32",
            "output_chunk_size": 2048,
            "workspace_mb": 128,
            "cache_transformed_weights": False,
            "recompute_backward": False,
            "persistent_indices": True,
            "init_mode": "per_rank",
        },
    }


def _spatial_config(
    *, core: str, task: str, phase: str, strategy: str
) -> dict[str, Any]:
    config = _common_config(
        core=core,
        task=task,
        phase=phase,
        strategy=strategy,
    )
    local = strategy == "local_ca"
    feedback = FEEDBACK_MODES if phase == "confirmatory" else ["per_soma_shared"]
    sweep: dict[str, Any] = {
        "model.core.connectivity.structured.enabled": [False, True],
    }
    if local:
        sweep["training.main.learning_strategy_config.error_broadcast_mode"] = feedback
    config["sweep_config"] = sweep
    _set_expected(config, 2 * (len(feedback) if local else 1), phase)

    base = config["base_config"]
    core_config = base["model"]["core"]
    core_config["architecture"]["excitatory_branch_factors"] = [2, 2, 2, 2]
    core_config["connectivity"].update(
        {
            "ee_synapses_per_branch_per_layer": [21],
            "ie_synapses_per_branch_per_layer": [11],
            "structured": {
                "enabled": False,
                "method": "spatial_morphology",
                "seed": 42,
                "spatial": {
                    "input_shape": [1, 28, 28],
                    "split_axes": ["height", "width", "height", "width"],
                },
            },
        }
    )
    core_config["sparsity"] = _fixed_indexed_sparsity()
    label = "local3f" if local else "backprop"
    base["outputs"]["run_name"] = (
        f"journal_{phase}_spatial_topology_{task}_{core}_{label}"
    )
    return config


def _routing_config(*, core: str, task: str, phase: str) -> dict[str, Any]:
    config = _common_config(
        core=core,
        task=task,
        phase=phase,
        strategy="local_ca",
    )
    depths = (
        [[2, 2], [2, 2, 2, 2]]
        if phase == "confirmatory"
        else [[2, 2, 2, 2]]
    )
    config["sweep_config"] = {
        "model.core.architecture.excitatory_branch_factors": depths,
        "training.main.learning_strategy_config.error_broadcast_mode": [
            "per_soma_shared",
            "per_soma_shuffled",
        ],
    }
    _set_expected(config, len(depths) * 2, phase)
    local = config["base_config"]["training"]["main"][
        "learning_strategy_config"
    ]
    local["broadcast_seed"] = 20260802
    config["base_config"]["outputs"]["run_name"] = (
        f"journal_{phase}_ancestry_routing_{task}_{core}_local3f"
    )
    return config


def render() -> dict[Path, str]:
    rendered: dict[Path, str] = {}
    for phase in ("canary", "confirmatory"):
        for core in ("shunting", "additive"):
            for strategy in ("local_ca", "backprop"):
                label = "local3f" if strategy == "local_ca" else "backprop"
                inhibition = _inhibition_config(
                    core=core,
                    phase=phase,
                    strategy=strategy,
                )
                path = CONFIG_DIR / f"{phase}_inhibition_noise_{core}_{label}.yaml"
                rendered[path] = OmegaConf.to_yaml(OmegaConf.create(inhibition))

                for task in ("mnist", "noise"):
                    spatial = _spatial_config(
                        core=core,
                        task=task,
                        phase=phase,
                        strategy=strategy,
                    )
                    path = CONFIG_DIR / (
                        f"{phase}_spatial_{task}_{core}_{label}.yaml"
                    )
                    rendered[path] = OmegaConf.to_yaml(OmegaConf.create(spatial))

                    if strategy == "local_ca":
                        routing = _routing_config(
                            core=core,
                            task=task,
                            phase=phase,
                        )
                        path = CONFIG_DIR / (
                            f"{phase}_routing_{task}_{core}_local3f.yaml"
                        )
                        rendered[path] = OmegaConf.to_yaml(OmegaConf.create(routing))
    return rendered


def generate() -> list[Path]:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    rendered = render()
    for path, text in rendered.items():
        path.write_text(text)
    return list(rendered)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check:
        mismatches = [
            path
            for path, text in render().items()
            if not path.exists() or path.read_text() != text
        ]
        if mismatches:
            raise SystemExit("Generated config mismatch: " + ", ".join(map(str, mismatches)))
        return
    for path in generate():
        print(path.relative_to(JOURNAL_ROOT))


if __name__ == "__main__":
    main()
