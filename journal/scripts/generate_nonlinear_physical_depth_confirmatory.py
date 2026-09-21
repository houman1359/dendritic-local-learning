#!/usr/bin/env python3
"""Generate the fresh-seed nonlinear physical-depth confirmatory sweeps."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf


JOURNAL_ROOT = Path(__file__).resolve().parents[1]
PILOT_GENERATOR = JOURNAL_ROOT / "scripts" / "generate_nonlinear_physical_depth_sweeps.py"
OUTPUT = JOURNAL_ROOT / "configs" / "nonlinear_physical_depth" / "confirmatory"
CONFIRMATORY_SEEDS = list(range(10200, 10210))


def _pilot_module():
    spec = importlib.util.spec_from_file_location("physical_depth_pilot", PILOT_GENERATOR)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {PILOT_GENERATOR}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _set_selected_point(config: dict[str, Any]) -> None:
    config["experiment_settings"] = {
        "seeds_per_condition": len(CONFIRMATORY_SEEDS),
        "base_seed": CONFIRMATORY_SEEDS[0],
        "use_array_jobs": True,
    }
    dataset = config["base_config"]["data"]["dataset_params"][
        "hierarchical_gain_load"
    ]
    dataset.update(
        {
            "e_signal_delta": 0.80,
            "train_gain_sigma": 0.25,
            "test_gain_sigma": 0.25,
            "valid_split_mode": "train",
        }
    )
    defaults = config["base_config"]["model"]["core"]["population_network"][
        "layers"
    ][0]["population_defaults"]
    defaults["initial_child_conductance"] = 16.0
    config["slurm_config"]["max_concurrent_jobs"] = 8
    config["slurm_config"]["time"] = "02:00:00"


def _one(*, regime: str, strategy: str, mechanism: str) -> tuple[Path, str]:
    pilot = _pilot_module()
    config = pilot._base(  # noqa: SLF001 - shared journal generator contract
        strategy=strategy,
        regime=regime,
        seeds=CONFIRMATORY_SEEDS,
        test_gain_sigma=0.25,
    )
    _set_selected_point(config)
    if mechanism not in {"shunting", "additive"}:
        raise ValueError(mechanism)
    defaults = config["base_config"]["model"]["core"]["population_network"][
        "layers"
    ][0]["population_defaults"]
    defaults["use_shunting"] = mechanism == "shunting"
    defaults["use_additive_normalization"] = False
    defaults["additive_mode"] = "raw"

    morphologies = pilot.MORPHOLOGIES
    sweep: dict[str, Any] = {
        "model.core.population_network.layers.0.populations.0.branch_factors": (
            morphologies
        )
    }
    conditions = len(morphologies)
    if strategy == "local_ca":
        sweep["training.main.learning_strategy_config.error_broadcast_mode"] = (
            pilot.FEEDBACK_MODES
        )
        conditions *= len(pilot.FEEDBACK_MODES)
    config["sweep_config"] = sweep
    config["sweep_contract"] = {
        "expected_config_count": conditions * len(CONFIRMATORY_SEEDS),
        "status": "confirmatory_fresh_seed",
        "selected_from_exploratory_signal": 0.80,
        "regime": regime,
        "mechanism": mechanism,
        "strategy": strategy,
    }
    method = "bp" if strategy == "standard" else "local3f"
    run_name = f"journal_confirmatory_physical_depth_{regime}_{mechanism}_{method}"
    config["base_config"]["outputs"]["run_name"] = run_name
    config["outputs"]["run_name"] = run_name
    path = OUTPUT / f"{regime}_{mechanism}_{method}.yaml"
    return path, OmegaConf.to_yaml(OmegaConf.create(config))


def render() -> dict[Path, str]:
    specs = [
        ("aligned", "standard", "shunting"),
        ("zero_alignment", "standard", "shunting"),
        ("sensor_shuffled", "standard", "shunting"),
        ("rewired_tree", "standard", "shunting"),
        ("aligned", "standard", "additive"),
        ("aligned", "local_ca", "shunting"),
        ("rewired_tree", "local_ca", "shunting"),
    ]
    return dict(_one(regime=r, strategy=s, mechanism=m) for r, s, m in specs)


def generate() -> list[Path]:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rendered = render()
    for path, content in rendered.items():
        path.write_text(content)
    return list(rendered)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    expected = render()
    if args.check:
        mismatches = [
            path
            for path, content in expected.items()
            if not path.exists() or path.read_text() != content
        ]
        if mismatches:
            raise SystemExit("Generated config mismatch: " + ", ".join(map(str, mismatches)))
        return
    for path in generate():
        print(path.relative_to(JOURNAL_ROOT))


if __name__ == "__main__":
    main()
