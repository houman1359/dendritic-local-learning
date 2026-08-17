#!/usr/bin/env python3
"""Generate frozen point--dendrite and BP--local-credit control sweeps."""

from __future__ import annotations

import argparse
import copy
import importlib.util
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parents[1]
CONFIRMATORY_GENERATOR = ROOT / "scripts" / "generate_nonlinear_physical_depth_confirmatory.py"
OUTPUT = ROOT / "configs" / "point_dendrite_credit_controls"


def _reference_module():
    spec = importlib.util.spec_from_file_location(
        "physical_depth_confirmatory", CONFIRMATORY_GENERATOR
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {CONFIRMATORY_GENERATOR}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _as_mapping(content: str) -> dict[str, Any]:
    import yaml

    value = yaml.safe_load(content)
    if not isinstance(value, dict):
        raise TypeError("Expected generated YAML mapping")
    return value


def _reference_config(regime: str, strategy: str = "standard") -> dict[str, Any]:
    reference = _reference_module()
    _path, content = reference._one(  # noqa: SLF001 - frozen shared recipe
        regime=regime,
        strategy=strategy,
        mechanism="shunting",
    )
    return _as_mapping(content)


def _population(config: dict[str, Any]) -> dict[str, Any]:
    return config["base_config"]["model"]["core"]["population_network"][
        "layers"
    ][0]["populations"][0]


def _finish(
    config: dict[str, Any],
    *,
    run_name: str,
    expected: int,
    condition: str,
    status: str = "prospective_reviewer_requested_extension",
    primary_outcomes_unopened: bool = True,
) -> str:
    config["output_dir"] = str(ROOT / "point_dendrite_credit_runs")
    config["base_config"]["outputs"]["run_name"] = run_name
    config["outputs"]["run_name"] = run_name
    config["slurm_config"]["run_name"] = run_name
    config["sweep_contract"] = {
        "status": status,
        "frozen_date": "2026-08-13",
        "reference_seeds_already_observed": True,
        "primary_control_outcomes_unopened_at_freeze": primary_outcomes_unopened,
        "this_condition_outcomes_unopened_at_freeze": True,
        "condition": condition,
        "expected_config_count": expected,
    }
    return OmegaConf.to_yaml(OmegaConf.create(config))


def _star(regime: str) -> str:
    config = _reference_config(regime)
    population = _population(config)
    population.setdefault("population", {})["cross_level_mode"] = "all_active_star"
    return _finish(
        config,
        run_name=f"journal_point_credit_{regime}_all_active_star_bp",
        expected=30,
        condition=f"all_active_star__{regime}",
    )


def _soma_broadcast(regime: str, *, matched_optimizer: bool = False) -> str:
    config = _reference_config(regime)
    population = _population(config)
    population.setdefault("population", {})[
        "autograd_credit_mode"
    ] = "soma_broadcast"
    suffix = "soma_broadcast_bp"
    condition = f"soma_broadcast_autograd__{regime}"
    status = "prospective_reviewer_requested_extension"
    primary_unopened = True
    if matched_optimizer:
        local_reference = _reference_config(regime, strategy="local_ca")
        standard_main = config["base_config"]["training"]["main"]
        local_main = local_reference["base_config"]["training"]["main"]
        standard_main["common"]["param_groups"] = copy.deepcopy(
            local_main["common"]["param_groups"]
        )
        standard_main["optimizer"] = copy.deepcopy(local_main["optimizer"])
        suffix = "soma_broadcast_matched_optimizer"
        condition = f"soma_broadcast_autograd_localca_optimizer__{regime}"
        status = "posthoc_optimizer_matching_diagnostic"
        primary_unopened = False
    return _finish(
        config,
        run_name=f"journal_point_credit_{regime}_{suffix}",
        expected=30,
        condition=condition,
        status=status,
        primary_outcomes_unopened=primary_unopened,
    )


def _matched_mlp(match_mode: str) -> str:
    if match_mode not in {"active", "total"}:
        raise ValueError(match_mode)
    config = _reference_config("aligned")
    core = config["base_config"]["model"]["core"]
    core["type"] = f"{match_mode}_param_mlp"
    _population(config)["branch_factors"] = [2, 1, 2]
    config["sweep_config"] = {
        "model.core.population_network.layers.0.populations.0.branch_factors": [
            [2, 1, 2]
        ]
    }
    return _finish(
        config,
        run_name=f"journal_point_credit_aligned_{match_mode}_param_mlp_bp",
        expected=10,
        condition=f"unstructured_{match_mode}_parameter_matched_point_mlp",
    )


def render() -> dict[Path, str]:
    return {
        OUTPUT / "aligned_all_active_star_bp.yaml": _star("aligned"),
        OUTPUT / "rewired_tree_all_active_star_bp.yaml": _star("rewired_tree"),
        OUTPUT / "aligned_soma_broadcast_bp.yaml": _soma_broadcast("aligned"),
        OUTPUT / "rewired_tree_soma_broadcast_bp.yaml": _soma_broadcast(
            "rewired_tree"
        ),
        OUTPUT / "aligned_soma_broadcast_matched_optimizer.yaml": _soma_broadcast(
            "aligned", matched_optimizer=True
        ),
        OUTPUT
        / "rewired_tree_soma_broadcast_matched_optimizer.yaml": _soma_broadcast(
            "rewired_tree", matched_optimizer=True
        ),
        OUTPUT / "aligned_active_param_mlp_bp.yaml": _matched_mlp("active"),
        OUTPUT / "aligned_total_param_mlp_bp.yaml": _matched_mlp("total"),
    }


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
            raise SystemExit("Generated config mismatch: " + ", ".join(map(str, mismatches)))
        return
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for path, content in expected.items():
        path.write_text(content, encoding="utf-8")
        print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
