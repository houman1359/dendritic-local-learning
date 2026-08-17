#!/usr/bin/env python3
"""Generate fixed-leaf, approximately fixed-contact dendritic-depth sweeps."""

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
from generate_shunting_topology_sweeps import _fixed_indexed_sparsity


CONFIG_DIR = JOURNAL_ROOT / "configs" / "prospective_fixed_budget_depth"
SHAPES = {
    "d1_16": ([16], 40, 20),
    "d2_4x4": ([4, 4], 32, 16),
    "d3_2x2x4": ([2, 2, 4], 29, 15),
    "d4_2x2x2x2": ([2, 2, 2, 2], 21, 11),
}


def _config(*, core: str, phase: str, strategy: str, shape_name: str) -> dict[str, Any]:
    config = _common_config(
        core=core,
        task="noise",
        phase=phase,
        strategy=strategy,
    )
    branch_factors, excitatory, inhibitory = SHAPES[shape_name]
    local = strategy == "local_ca"
    feedback = (
        FEEDBACK_MODES
        if phase == "confirmatory"
        else [
            "per_soma",
            "path_transport",
        ]
    )
    seeds = 10 if phase == "confirmatory" else 1

    config["experiment_settings"]["seeds_per_condition"] = seeds
    config["slurm_config"]["max_concurrent_jobs"] = 1
    config["sweep_config"] = {} if local else {"training.main.common.epochs": [180]}
    if local:
        config["sweep_config"][
            "training.main.learning_strategy_config.error_broadcast_mode"
        ] = feedback
    config["sweep_contract"] = {
        "expected_config_count": seeds * (len(feedback) if local else 1)
    }

    model_core = config["base_config"]["model"]["core"]
    model_core["architecture"]["excitatory_branch_factors"] = branch_factors
    model_core["connectivity"].update(
        {
            "ee_synapses_per_branch_per_layer": [excitatory],
            "ie_synapses_per_branch_per_layer": [inhibitory],
            "structured": {"enabled": False},
        }
    )
    model_core["sparsity"] = _fixed_indexed_sparsity()
    label = "local3f" if local else "backprop"
    config["base_config"]["outputs"][
        "run_name"
    ] = f"journal_{phase}_fixed_budget_depth_{shape_name}_{core}_{label}"
    return config


def render() -> dict[Path, str]:
    rendered: dict[Path, str] = {}
    for phase in ("canary", "confirmatory"):
        for core in ("shunting", "additive"):
            for strategy in ("local_ca", "backprop"):
                label = "local3f" if strategy == "local_ca" else "backprop"
                for shape_name in SHAPES:
                    config = _config(
                        core=core,
                        phase=phase,
                        strategy=strategy,
                        shape_name=shape_name,
                    )
                    path = CONFIG_DIR / (f"{phase}_{shape_name}_{core}_{label}.yaml")
                    rendered[path] = OmegaConf.to_yaml(OmegaConf.create(config))
    return rendered


def generate() -> list[Path]:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    rendered = render()
    for path, content in rendered.items():
        path.write_text(content)
    return list(rendered)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check:
        mismatches = [
            path
            for path, content in render().items()
            if not path.exists() or path.read_text() != content
        ]
        if mismatches:
            raise SystemExit(
                "Generated config mismatch: " + ", ".join(map(str, mismatches))
            )
        return
    for path in generate():
        print(path.relative_to(JOURNAL_ROOT))


if __name__ == "__main__":
    main()
