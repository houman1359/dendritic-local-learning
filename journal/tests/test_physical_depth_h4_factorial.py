from __future__ import annotations

import importlib.util
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
GENERATOR = ROOT / "scripts" / "generate_physical_depth_h4_factorial.py"


def _module():
    spec = importlib.util.spec_from_file_location("h4_generator", GENERATOR)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_h4_factorial_matches_frozen_contract() -> None:
    module = _module()
    rendered = module.render()
    assert len(rendered) == 7

    expected_total = 0
    for content in rendered.values():
        config = yaml.safe_load(content)
        contract = config["sweep_contract"]
        expected_total += int(contract["expected_config_count"])
        assert contract["seeds"] == list(range(10400, 10410))
        assert contract["primary_inferential_unit"] == "training_seed"
        assert config["base_config"]["data"]["dataset_params"][
            "hierarchical_gain_load"
        ]["n_levels"] == 4
        pathways = config["base_config"]["model"]["core"][
            "population_network"
        ]["layers"][0]["population_defaults"]["structured_connectivity"][
            "pathways"
        ]
        for pathway in ("ee", "ie"):
            assert pathways[pathway]["inventory_counts"] == [4, 2, 1, 1]
        morphologies = config["sweep_config"][
            "model.core.population_network.layers.0.populations.0.branch_factors"
        ]
        assert morphologies == [[8], [2, 3], [2, 1, 2], [1, 1, 2, 2]]
        assert [
            sum(__import__("math").prod(shape[:i]) for i in range(1, len(shape) + 1))
            for shape in morphologies
        ] == [8, 8, 8, 8]

    assert expected_total == 360

