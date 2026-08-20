from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


JOURNAL = Path(__file__).resolve().parents[1]
GENERATOR = JOURNAL / "scripts" / "generate_task_family_alignment_factorial.py"


def _module():
    spec = importlib.util.spec_from_file_location("task_family_generator", GENERATOR)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _analysis_module():
    path = JOURNAL / "scripts" / "analyze_task_family_alignment_factorial.py"
    spec = importlib.util.spec_from_file_location("task_family_analysis", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_factorial_is_fixed_depth_complete_and_resource_matched() -> None:
    module = _module()
    rendered = module.render()
    assert len(rendered) == 12
    seen = set()
    for content in rendered.values():
        config = yaml.safe_load(content)
        contract = config["sweep_contract"]
        dataset = config["base_config"]["data"]["dataset_params"][
            "hierarchical_gain_load"
        ]
        population = config["base_config"]["model"]["core"][
            "population_network"
        ]["layers"][0]["populations"][0]
        assert population["branch_factors"] == [2, 1, 2]
        assert list(config["sweep_config"].values()) == [[0.0, 0.5, 1.0]]
        assert contract["expected_config_count"] == 30
        assert contract["seeds"] == list(range(10500, 10510))
        assert dataset["signal_profile"] == "distal_only"
        assert dataset["i_signal_delta"] == 0.0
        assert config["base_config"]["wandb"]["use_wandb"] is False
        assert "kempner_project_b" in config["output_dir"]
        assert contract["external_tracking"].startswith("disabled")
        seen.add(
            (contract["family"], contract["architecture"], contract["credit"])
        )
    assert len(seen) == 12


def test_task_families_change_composition_without_changing_operating_point() -> None:
    module = _module()
    configs = [yaml.safe_load(value) for value in module.render().values()]
    datasets = [
        value["base_config"]["data"]["dataset_params"]["hierarchical_gain_load"]
        for value in configs
    ]
    invariant = (
        "n_samples",
        "stream_dim",
        "n_levels",
        "e_signal_delta",
        "train_gain_sigma",
        "test_gain_sigma",
        "independent_noise_std",
    )
    for field in invariant:
        assert len({dataset[field] for dataset in datasets}) == 1
    pairs = {
        (dataset["nuisance_layout"], dataset["gain_structure"])
        for dataset in datasets
    }
    assert pairs == {
        ("factorized_sensors", "hierarchical"),
        ("factorized_sensors", "flat"),
        ("paired_cumulative", "hierarchical"),
    }


def test_paired_interactions_preserve_seed_and_contrast_sign() -> None:
    module = _analysis_module()
    alignment_slopes = {
        ("nested_factor", "bp"): 0.10,
        ("nested_factor", "local3f"): 0.08,
        ("flat_factor", "bp"): 0.01,
        ("flat_factor", "local3f"): 0.00,
        ("local_ratio", "bp"): -0.02,
        ("local_ratio", "local3f"): -0.01,
    }
    rows = []
    for family in module.FAMILIES:
        for credit in module.CREDITS:
            for alpha in module.ALPHAS:
                for seed in module.SEEDS:
                    grouped = 0.50 + 1e-4 * (seed - module.SEEDS[0])
                    serial = grouped + alignment_slopes[(family, credit)] * alpha
                    for architecture, accuracy in (
                        ("grouped_point", grouped),
                        ("serial", serial),
                    ):
                        rows.append(
                            {
                                "family": family,
                                "architecture": architecture,
                                "credit": credit,
                                "alignment_alpha": alpha,
                                "seed": seed,
                                "test_accuracy": accuracy,
                            }
                        )
    _summary, _effects, contrasts = module.summarize(pd.DataFrame(rows))
    nested_bp = contrasts[
        contrasts.estimand.eq("alignment_interaction")
        & contrasts.family.eq("nested_factor")
        & contrasts.credit.eq("bp")
    ].iloc[0]
    specificity = contrasts[
        contrasts.estimand.eq("task_specificity_interaction")
        & contrasts.credit.eq("bp")
        & contrasts.comparison.eq("nested_minus_flat_factor")
    ].iloc[0]
    credit = contrasts[
        contrasts.estimand.eq("credit_interaction")
        & contrasts.family.eq("nested_factor")
        & contrasts.comparison.eq("alpha_1.0")
    ].iloc[0]
    assert np.isclose(nested_bp.mean_difference, 0.10)
    assert np.isclose(specificity.mean_difference, 0.09)
    assert np.isclose(credit.mean_difference, -0.02)
    assert nested_bp.positive_pairs == len(module.SEEDS)
