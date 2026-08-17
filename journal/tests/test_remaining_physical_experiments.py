from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from omegaconf import OmegaConf
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "generate_remaining_physical_experiments.py"


def _module():
    spec = importlib.util.spec_from_file_location("remaining_physical", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_remaining_matrix_is_complete_and_frozen() -> None:
    module = _module()
    rendered = module.render()
    assert len(rendered) == 8
    assert sum(
        int(OmegaConf.create(content).sweep_contract.expected_config_count)
        for content in rendered.values()
    ) == 220
    assert {path.stem for path in rendered} == {
        "h3_aligned_grouped_point_bp",
        "h3_rewired_tree_grouped_point_bp",
        "h2_aligned_serial_bp",
        "h2_rewired_tree_serial_bp",
        "h2_aligned_grouped_point_bp",
        "h2_rewired_tree_grouped_point_bp",
        "h2_aligned_serial_local3f",
        "h2_rewired_tree_serial_local3f",
    }
    for content in rendered.values():
        config = OmegaConf.create(content)
        assert config.sweep_contract.status == (
            "prospective_remaining_reviewer_experiment"
        )
        assert config.sweep_contract.outcomes_unopened_at_freeze is True
        assert config.sweep_contract.no_hyperparameter_retuning is True
        assert config.sweep_contract.primary_inferential_unit == "training_seed"
        assert config.output_dir == "remaining_physical_experiment_runs"


def test_h3_grouped_point_inherits_the_frozen_physical_cohort() -> None:
    module = _module()
    for regime in ("aligned", "rewired_tree"):
        content = module.render()[
            module.OUTPUT / f"h3_{regime}_grouped_point_bp.yaml"
        ]
        config = OmegaConf.create(content)
        assert config.experiment_settings.base_seed == 10200
        assert config.experiment_settings.seeds_per_condition == 10
        assert list(
            config.sweep_config[
                "model.core.population_network.layers.0.populations.0.branch_factors"
            ]
        ) == [[8], [2, 3], [2, 1, 2]]
        population = config.base_config.model.core.population_network.layers[0].populations[0]
        assert population.population.cross_level_mode == "parallel_readout"
        dataset = config.base_config.data.dataset_params.hierarchical_gain_load
        assert dataset.n_levels == 3
        assert dataset.e_signal_delta == 0.80


def test_h2_changes_only_hierarchy_and_exact_resource_inventory() -> None:
    module = _module()
    for path, content in module.render().items():
        if not path.stem.startswith("h2_"):
            continue
        config = OmegaConf.create(content)
        assert config.experiment_settings.base_seed == 10300
        assert config.experiment_settings.seeds_per_condition == 10
        assert list(
            config.sweep_config[
                "model.core.population_network.layers.0.populations.0.branch_factors"
            ]
        ) == [[8], [4, 1]]
        dataset = config.base_config.data.dataset_params.hierarchical_gain_load
        assert dataset.n_levels == 2
        assert dataset.e_signal_delta == 0.80
        assert dataset.train_gain_sigma == dataset.test_gain_sigma == 0.25
        defaults = config.base_config.model.core.population_network.layers[0].population_defaults
        assert defaults.initial_child_conductance == 16.0
        for pathway in ("ee", "ie"):
            spec = defaults.structured_connectivity.pathways[pathway]
            assert list(spec.inventory_counts) == [4, 4]
            expected = (
                [[0, 32], [32, 64]]
                if "_aligned_" in path.stem
                else [[32, 64], [0, 32]]
            )
            assert list(spec.feature_ranges) == expected


def test_completed_remaining_matrix_passes_all_artifact_gates() -> None:
    source = ROOT / "source_data" / "remaining_physical_experiments"
    audit = json.loads((source / "audit.json").read_text())
    assert audit["status"] == "complete_pass"
    assert audit["expected_rows"] == audit["observed_rows"] == 220
    assert audit["missing_count"] == 0
    assert audit["duplicate_seed_conditions"] == 0
    assert audit["all_metrics_finite"] is True
    assert audit["fallback_mentions"] == 0
    assert audit["nonfinite_alerts"] == 0
    assert audit["resource_gate"] is True


def test_primary_remaining_contrasts_pass_the_frozen_claim_gate() -> None:
    contrasts = pd.read_csv(
        ROOT
        / "source_data"
        / "remaining_physical_experiments"
        / "paired_contrasts.csv"
    ).set_index("contrast")
    primary = {
        "h3_serial_minus_grouped__aligned__d3",
        "h3_serial_grouped_alignment_interaction__d3",
        "h2_depth__serial_bp__aligned",
        "h2_alignment_interaction__serial_bp",
        "h2_serial_minus_grouped__aligned__d2",
        "h2_serial_grouped_alignment_interaction__d2",
        "h2_depth__shared_local__aligned",
        "h2_alignment_interaction__shared_local",
        "h2_depth__path_local__aligned",
        "h2_alignment_interaction__path_local",
    }
    assert contrasts.loc[list(primary), "positive_claim_gate"].all()
    assert (contrasts.loc[list(primary), "positive_pairs"] == 10).all()
    assert not bool(
        contrasts.loc[
            "h2_alignment_interaction__grouped_bp", "positive_claim_gate"
        ]
    )
