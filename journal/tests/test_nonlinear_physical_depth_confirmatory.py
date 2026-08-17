from __future__ import annotations

import importlib.util
from pathlib import Path

from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "generate_nonlinear_physical_depth_confirmatory.py"


def _module():
    spec = importlib.util.spec_from_file_location("physical_depth_confirmatory", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_confirmatory_design_has_fresh_seeds_and_selected_point() -> None:
    module = _module()
    rendered = module.render()
    assert len(rendered) == 7
    for path, content in rendered.items():
        config = OmegaConf.create(content)
        assert config.experiment_settings.base_seed == 10200
        assert config.experiment_settings.seeds_per_condition == 10
        assert config.sweep_contract.status == "confirmatory_fresh_seed"
        assert config.sweep_contract.selected_from_exploratory_signal == 0.80
        data = config.base_config.data.dataset_params.hierarchical_gain_load
        assert data.e_signal_delta == 0.80
        assert data.train_gain_sigma == data.test_gain_sigma == 0.25
        defaults = config.base_config.model.core.population_network.layers[0].population_defaults
        assert defaults.initial_child_conductance == 16.0
        assert defaults.reactivate is False
        assert config.base_config.analysis.compartment_statistics.enabled is False
        expected = 60 if path.stem.endswith("local3f") else 30
        assert config.sweep_contract.expected_config_count == expected


def test_confirmatory_controls_are_complete_and_resource_matched() -> None:
    module = _module()
    rendered = module.render()
    stems = {path.stem for path in rendered}
    assert stems == {
        "aligned_shunting_bp",
        "zero_alignment_shunting_bp",
        "sensor_shuffled_shunting_bp",
        "rewired_tree_shunting_bp",
        "aligned_additive_bp",
        "aligned_shunting_local3f",
        "rewired_tree_shunting_local3f",
    }
    for content in rendered.values():
        config = OmegaConf.create(content)
        assert list(
            config.sweep_config[
                "model.core.population_network.layers.0.populations.0.branch_factors"
            ]
        ) == [[8], [2, 3], [2, 1, 2]]
        defaults = config.base_config.model.core.population_network.layers[0].population_defaults
        assert defaults.ff_excitatory_synapses == 16
        assert defaults.ff_inhibitory_synapses == 12
        assert list(defaults.structured_connectivity.pathways.ee.inventory_counts) == [4, 2, 2]
