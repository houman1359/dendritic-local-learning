from __future__ import annotations

import importlib.util
from pathlib import Path

from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "generate_nonlinear_physical_depth_sweeps.py"


def _module():
    spec = importlib.util.spec_from_file_location("physical_depth_generator", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_physical_depth_canary_is_exact_resource_and_positive_conductance() -> None:
    module = _module()
    rendered = module.render()
    assert len(rendered) == 8
    for path, content in rendered.items():
        config = OmegaConf.create(content)
        assert config.experiment_settings.seeds_per_condition == 2
        assert config.experiment_settings.base_seed == 10100
        assert list(
            config.sweep_config[
                "model.core.population_network.layers.0.populations.0.branch_factors"
            ]
        ) == [[8], [2, 3], [2, 1, 2]]
        defaults = config.base_config.model.core.population_network.layers[0].population_defaults
        assert defaults.use_shunting is True
        assert defaults.reactivate is False
        assert defaults.dbl_init_method == "mechanism_neutral"
        assert defaults.adaptive_initialization is False
        assert defaults.initial_child_conductance == 16.0
        dataset = config.base_config.data.dataset_params.hierarchical_gain_load
        assert dataset.e_signal_delta == 0.24
        assert dataset.private_gain_sigma == 0.0
        assert dataset.load_mean == 0.0
        assert config.base_config.analysis.compartment_statistics.enabled is False
        expected = 12 if path.stem.endswith("local3f") else 6
        assert config.sweep_contract.expected_config_count == expected


def test_rewired_control_reverses_equal_resource_feature_inventory() -> None:
    module = _module()
    aligned = OmegaConf.create(
        module.render()[module.OUTPUT / "canary_aligned_bp.yaml"]
    )
    rewired = OmegaConf.create(
        module.render()[module.OUTPUT / "canary_rewired_tree_bp.yaml"]
    )
    aligned_layer = aligned.base_config.model.core.population_network.layers[0]
    rewired_layer = rewired.base_config.model.core.population_network.layers[0]
    for pathway in ("ee", "ie"):
        assert list(
            aligned_layer.population_defaults.structured_connectivity.pathways[
                pathway
            ].feature_ranges
        ) == [[0, 21], [21, 43], [43, 64]]
        assert list(
            rewired_layer.population_defaults.structured_connectivity.pathways[
                pathway
            ].feature_ranges
        ) == [[43, 64], [21, 43], [0, 21]]
    assert (
        aligned.base_config.data.dataset_params.hierarchical_gain_load
        == rewired.base_config.data.dataset_params.hierarchical_gain_load
    )


def test_accessibility_ladder_is_separate_and_exploratory() -> None:
    module = _module()
    rendered = module.render_accessibility()
    assert len(rendered) == 1
    config = OmegaConf.create(next(iter(rendered.values())))
    assert config.experiment_settings.seeds_per_condition == 2
    assert config.experiment_settings.base_seed == 10110
    assert config.sweep_contract.status == "exploratory_post_canary"
    assert config.sweep_contract.expected_config_count == 18
    assert list(
        config.sweep_config[
            "data.dataset_params.hierarchical_gain_load.test_gain_sigma"
        ]
    ) == [0.25, 0.5, 0.8]
    assert list(
        config.sweep_config[
            "model.core.population_network.layers.0.populations.0.branch_factors"
        ]
    ) == [[8], [2, 3], [2, 1, 2]]


def test_coupling_ladder_is_exact_resource_and_exploratory() -> None:
    module = _module()
    rendered = module.render_coupling_accessibility()
    assert len(rendered) == 1
    config = OmegaConf.create(next(iter(rendered.values())))
    assert config.experiment_settings.seeds_per_condition == 2
    assert config.experiment_settings.base_seed == 10120
    assert config.sweep_contract.status == "exploratory_post_accessibility"
    assert config.sweep_contract.expected_config_count == 24
    assert list(
        config.sweep_config[
            "model.core.population_network.layers.0.population_defaults."
            "initial_child_conductance"
        ]
    ) == [1.0, 4.0, 16.0, 64.0]
    dataset = config.base_config.data.dataset_params.hierarchical_gain_load
    assert dataset.train_gain_sigma == 0.25
    assert dataset.test_gain_sigma == 0.25


def test_signal_ladder_keeps_stable_coupling_and_gain_fixed() -> None:
    module = _module()
    rendered = module.render_signal_accessibility()
    assert len(rendered) == 1
    config = OmegaConf.create(next(iter(rendered.values())))
    assert config.experiment_settings.base_seed == 10130
    assert config.sweep_contract.status == "exploratory_post_coupling"
    assert config.sweep_contract.expected_config_count == 24
    assert config.sweep_contract.fixed_initial_child_conductance == 16.0
    assert list(
        config.sweep_config[
            "data.dataset_params.hierarchical_gain_load.e_signal_delta"
        ]
    ) == [0.24, 0.36, 0.48, 0.72]
    defaults = config.base_config.model.core.population_network.layers[0].population_defaults
    assert defaults.initial_child_conductance == 16.0


def test_boundary_resolution_is_one_fixed_post_ladder_point() -> None:
    module = _module()
    rendered = module.render_boundary_resolution()
    assert len(rendered) == 1
    config = OmegaConf.create(next(iter(rendered.values())))
    assert config.experiment_settings.base_seed == 10140
    assert config.sweep_contract.status == "exploratory_boundary_resolution"
    assert config.sweep_contract.expected_config_count == 6
    assert config.sweep_contract.fixed_signal_delta == 0.80
    assert config.sweep_contract.no_further_signal_search is True
    dataset = config.base_config.data.dataset_params.hierarchical_gain_load
    assert dataset.e_signal_delta == 0.80
    assert dataset.train_gain_sigma == dataset.test_gain_sigma == 0.25
