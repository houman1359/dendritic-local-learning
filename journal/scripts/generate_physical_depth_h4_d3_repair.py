#!/usr/bin/env python3
"""Generate the frozen construction repair for the H=4 D3 conditions."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from typing import Any

import yaml
from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parents[1]
H4_GENERATOR = ROOT / "scripts" / "generate_physical_depth_h4_factorial.py"
OUTPUT = ROOT / "configs" / "physical_depth_h4_d3_repair"
RUNS = Path("physical_depth_h4_d3_repair_runs")
MORPHOLOGY = [2, 1, 2]
TIER_GROUPS = [[0], [1], [2, 3]]
SOURCE_COMMIT = "e90fb9896daa95df4aad4c0d92acc0cd65bcd750"


def _h4_module():
    spec = importlib.util.spec_from_file_location("physical_depth_h4", H4_GENERATOR)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {H4_GENERATOR}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _repair_config(
    *,
    regime: str,
    strategy: str,
    grouped_point: bool = False,
    mechanism: str = "shunting",
) -> str:
    h4 = _h4_module()
    rendered = h4._config(
        regime=regime,
        strategy=strategy,
        grouped_point=grouped_point,
        mechanism=mechanism,
    )
    config: dict[str, Any] = yaml.safe_load(rendered)
    architecture = "grouped_point" if grouped_point else "serial"
    method = "bp" if strategy == "standard" else "local3f"
    run_name = f"journal_h4_d3repair_{regime}_{architecture}_{mechanism}_{method}"

    config["output_dir"] = RUNS.as_posix()
    config["slurm_config"]["account"] = "kempner_dev"
    config["slurm_config"]["partition"] = "kempner_rtx"
    config["slurm_config"]["max_concurrent_jobs"] = 10
    config["slurm_config"]["run_name"] = run_name
    config["base_config"]["outputs"]["run_name"] = run_name
    config["outputs"]["run_name"] = run_name

    population = config["base_config"]["model"]["core"]["population_network"]["layers"][
        0
    ]
    for pathway in ("ee", "ie"):
        population["population_defaults"]["structured_connectivity"]["pathways"][
            pathway
        ]["tier_groups"] = TIER_GROUPS

    sweep = {
        "model.core.population_network.layers.0.populations.0.branch_factors": [
            MORPHOLOGY
        ]
    }
    conditions = 1
    if strategy == "local_ca":
        sweep["training.main.learning_strategy_config.error_broadcast_mode"] = [
            "per_soma_shared",
            "path_transport",
        ]
        conditions = 2
    config["sweep_config"] = sweep
    config["sweep_contract"].update(
        {
            "status": "prospective_h4_d3_construction_repair",
            "frozen_date": "2026-08-20",
            "source_commit": SOURCE_COMMIT,
            "original_outcomes_unopened": True,
            "repair_scope": "failed_D3_indices_only",
            "tier_groups": TIER_GROUPS,
            "expected_config_count": conditions * 10,
        }
    )
    return OmegaConf.to_yaml(OmegaConf.create(config))


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
            f"h4_d3repair_{regime}_{architecture}_{mechanism}_{method}.yaml"
        )
        rendered[path] = _repair_config(
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
