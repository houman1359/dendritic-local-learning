#!/usr/bin/env python3
"""Generate the frozen intermediate-alignment physical-depth sweep."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "scripts" / "generate_nonlinear_physical_depth_confirmatory.py"
OUTPUT = ROOT / "configs" / "physical_alignment_dose" / "intermediate_bp.yaml"
ALPHAS = [0.25, 0.50, 0.75]


def _reference_module():
    spec = importlib.util.spec_from_file_location("physical_depth_reference", REFERENCE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {REFERENCE}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def render() -> str:
    module = _reference_module()
    _path, content = module._one(  # noqa: SLF001 - frozen shared recipe
        regime="aligned", strategy="standard", mechanism="shunting"
    )
    config: dict[str, Any] = OmegaConf.to_container(
        OmegaConf.create(content), resolve=True
    )
    config["output_dir"] = str(ROOT / "physical_alignment_dose_runs")
    alpha_key = (
        "data.dataset_params.hierarchical_gain_load.sensor_alignment_alpha"
    )
    config["sweep_config"][alpha_key] = ALPHAS
    run_name = "journal_physical_alignment_dose_intermediate_bp"
    config["base_config"]["outputs"]["run_name"] = run_name
    config["outputs"]["run_name"] = run_name
    config["slurm_config"]["run_name"] = run_name
    config["sweep_contract"] = {
        "status": "prospective_endpoint_interpolation",
        "frozen_date": "2026-08-13",
        "endpoint_outcomes_already_observed": True,
        "intermediate_outcomes_unopened_at_freeze": True,
        "alignment_alpha": ALPHAS,
        "paired_seeds": list(range(10200, 10210)),
        "expected_config_count": 90,
    }
    return OmegaConf.to_yaml(OmegaConf.create(config))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    content = render()
    if args.check:
        if not OUTPUT.is_file() or OUTPUT.read_text(encoding="utf-8") != content:
            raise SystemExit(f"Generated config mismatch: {OUTPUT}")
        return
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(content, encoding="utf-8")
    print(OUTPUT.relative_to(ROOT))


if __name__ == "__main__":
    main()
