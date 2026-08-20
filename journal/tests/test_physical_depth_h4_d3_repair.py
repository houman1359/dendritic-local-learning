from __future__ import annotations

import importlib.util
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
GENERATOR = ROOT / "scripts" / "generate_physical_depth_h4_d3_repair.py"


def _module():
    spec = importlib.util.spec_from_file_location("h4_d3_repair", GENERATOR)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {GENERATOR}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_h4_d3_repair_is_limited_to_failed_conditions() -> None:
    module = _module()
    rendered = module.render()
    assert len(rendered) == 7
    assert (
        sum(
            yaml.safe_load(content)["sweep_contract"]["expected_config_count"]
            for content in rendered.values()
        )
        == 90
    )
    for content in rendered.values():
        config = yaml.safe_load(content)
        sweep = config["sweep_config"]
        assert sweep[
            "model.core.population_network.layers.0.populations.0.branch_factors"
        ] == [[2, 1, 2]]
        assert config["slurm_config"]["account"] == "kempner_dev"
        assert config["slurm_config"]["partition"] == "kempner_rtx"
        assert config["sweep_contract"]["original_outcomes_unopened"] is True


def test_h4_d3_repair_uses_explicit_ordered_tier_partition() -> None:
    module = _module()
    for content in module.render().values():
        config = yaml.safe_load(content)
        defaults = config["base_config"]["model"]["core"]["population_network"][
            "layers"
        ][0]["population_defaults"]
        for pathway in ("ee", "ie"):
            spec = defaults["structured_connectivity"]["pathways"][pathway]
            assert spec["inventory_counts"] == [4, 2, 1, 1]
            assert spec["tier_groups"] == [[0], [1], [2, 3]]
