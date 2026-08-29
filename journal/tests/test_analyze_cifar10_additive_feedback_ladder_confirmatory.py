from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT = (
    Path(__file__).parents[1]
    / "scripts"
    / "analyze_cifar10_additive_feedback_ladder_confirmatory.py"
)
SPEC = importlib.util.spec_from_file_location(
    "analyze_cifar10_additive_feedback_ladder_confirmatory", SCRIPT
)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _passing_frame(
    bp_offset: float = 0.0,
    exact_gain: float = 0.02,
    baseline: float = 0.43,
) -> pd.DataFrame:
    rows = []
    for offset, seed in enumerate(MODULE.EXPECTED_SEEDS):
        seed_baseline = baseline + 0.0002 * offset
        scalar = seed_baseline
        neuron = seed_baseline + 0.05 + 0.0002 * ((offset % 3) - 1)
        exact = neuron + exact_gain + 0.0002 * ((offset % 4) - 1.5)
        bp = exact + bp_offset + 0.001 * (1 if offset % 2 else -1)
        for feedback, value in zip(
            MODULE.CONDITION_ORDER, (scalar, neuron, exact, bp)
        ):
            rows.append(
                {
                    "feedback": feedback,
                    "seed": seed,
                    "test_accuracy": value,
                }
            )
    return pd.DataFrame(rows)


def test_exact_sign_flip_enumerates_all_assignments():
    values = np.array([1.0, 1.0])
    assert np.isclose(MODULE.exact_sign_flip_p(values, "greater"), 0.25)
    assert np.isclose(MODULE.exact_sign_flip_p(values, "two-sided"), 0.50)


def test_holm_adjustment_is_monotone_in_rank():
    adjusted = MODULE.holm_adjust({"a": 0.01, "b": 0.03, "c": 0.20})
    assert np.isclose(adjusted["a"], 0.03)
    assert np.isclose(adjusted["b"], 0.06)
    assert np.isclose(adjusted["c"], 0.20)


def test_path_promotion_is_separate_from_bp_equivalence_claim():
    audit = {"integrity_valid": True, "convergence_valid": True}
    _, _, passing = MODULE.summarize(_passing_frame(), audit)
    assert passing["main_promotion"]
    assert passing["path_resolution_main_promotion"]
    assert passing["exact_path_recovers_bp_claim"]
    assert passing["matched_bp_adequacy_passes"]
    assert passing["neuron_minus_scalar_positive_control_passes"]
    assert passing["exact_minus_neuron_hierarchical_superiority_passes"]
    assert passing["exact_vs_bp_equivalence"]["equivalent"]

    _, _, non_equivalent = MODULE.summarize(_passing_frame(bp_offset=0.02), audit)
    assert non_equivalent["main_promotion"]
    assert non_equivalent["path_resolution_main_promotion"]
    assert not non_equivalent["exact_path_recovers_bp_claim"]
    assert not non_equivalent["exact_vs_bp_equivalence"]["equivalent"]


def test_inadequate_bp_blocks_cross_dataset_promotion():
    audit = {"integrity_valid": True, "convergence_valid": True}
    _, _, decision = MODULE.summarize(_passing_frame(baseline=0.30), audit)
    assert not decision["matched_bp_adequacy_passes"]
    assert not decision["bandwidth_cross_dataset_promotion"]
    assert not decision["path_resolution_main_promotion"]
    assert not decision["exact_path_recovers_bp_claim"]


def test_sub_one_point_exact_gain_cannot_pass_promotion():
    audit = {"integrity_valid": True, "convergence_valid": True}
    _, _, decision = MODULE.summarize(_passing_frame(exact_gain=0.009), audit)
    assert not decision["main_promotion"]
    assert not decision["promotion_gate"]["primary_mean_at_least_1pp"]


def test_convergence_audit_flags_right_censoring(tmp_path):
    summary = {
        "train_losses": np.linspace(2.0, 1.0, 20).tolist(),
        "valid_losses": np.linspace(2.0, 1.0, 20).tolist(),
        "best_epoch": 20,
        "best_loss": 1.0,
    }
    (tmp_path / "training_summary.json").write_text(json.dumps(summary))
    resolved = {"training": {"main": {"common": {"epochs": 20, "patience": 5}}}}
    record = MODULE.convergence_record(tmp_path, resolved, "exact path")
    assert record["valid"]
    assert record["right_censored"]
    assert record["final10_validation_slope"] < 0


def test_empirical_calibration_requires_five_converged_unreverted_layers(tmp_path):
    layers = {
        f"core_network.layers.0.excitatory_cells.branch_layers.{index}": {
            "m_applied": 1.0 + index,
            "b_applied": 0.1 * index,
            "calibration_reverted": False,
        }
        for index in range(5)
    }
    (tmp_path / "reactivation_calibration.json").write_text(
        json.dumps(
            {
                "requested": True,
                "policies": ["empirical"],
                "mode": "per-layer",
                "n_batches": 3,
                "converged": True,
                "layers": layers,
            }
        )
    )
    (tmp_path / "init_gate_stats.json").write_text("{}")
    post_modules = {
        f"core_network.layers.0.excitatory_cells.branch_layers.{index}.reactivation": {
            "class": "ParametricTanh",
            "m": {"n": 10, "mean": 1.0 + index},
            "b": {"n": 10, "mean": 0.1 * index},
        }
        for index in range(5)
    }
    # Reverse insertion order to ensure the post-calibration signature is
    # structural rather than dependent on local-vs-BP construction order.
    (tmp_path / "post_calibration_gate_stats.json").write_text(
        json.dumps({"reactivation_modules": dict(reversed(list(post_modules.items())))})
    )
    valid = MODULE.calibration_record(tmp_path)
    assert valid["valid"]
    assert valid["n_layers"] == 5

    layers[
        "core_network.layers.0.excitatory_cells.branch_layers.0"
    ]["calibration_reverted"] = True
    (tmp_path / "reactivation_calibration.json").write_text(
        json.dumps(
            {
                "requested": True,
                "policies": ["empirical"],
                "mode": "per-layer",
                "n_batches": 3,
                "converged": True,
                "layers": layers,
            }
        )
    )
    invalid = MODULE.calibration_record(tmp_path)
    assert not invalid["valid"]
    assert invalid["reverted"]


def test_resolved_field_audit_catches_wrong_additive_operator():
    config = {
        "model": {
            "core": {
                "morphology": {
                    "use_additive_normalization": True,
                }
            }
        }
    }
    errors = MODULE.field_errors(
        config,
        {"model.core.morphology.use_additive_normalization": False},
    )
    assert errors
    assert "use_additive_normalization" in errors[0]


def test_input_mode1_semantics_reject_explicit_inhibitory_population(tmp_path):
    metadata = {"strategy": "standard", "broadcast": None}
    config = {
        "experiment": {"seed": 10800},
        "model": {
            "core": {
                "transfer": {"input_mode1_build_inhibitory_population": True}
            }
        },
        "training": {
            "main": {"strategy": "standard", "learning_strategy_config": None}
        },
        "outputs": {"results_dir": str(tmp_path)},
    }
    errors = MODULE.audit_resolved_config(config, "bp", metadata, tmp_path)
    assert any("without an explicit inhibitory-cell population" in error for error in errors)


def test_active_optimizer_semantics_do_not_treat_inert_group_lrs_as_active():
    config = {
        "training": {
            "main": {
                "common": {
                    "weight_decay_rate": 0.01,
                    "param_groups": {
                        "split_params": False,
                        "lr": 0.001,
                        "blocklinear_lr": 0.0001,
                        "reactivation_lr": 0.0001,
                        "topk_lr": 0.001,
                        "decoder_lr": 0.001,
                    },
                },
                "optimizer": {"name": "adam", "weight_decay": 0.0},
            }
        }
    }
    semantics = MODULE.active_optimizer_semantics(config)
    assert semantics["valid"]
    assert semantics["active_group_count"] == 1
    assert np.isclose(semantics["active_learning_rate"], 0.001)
    assert np.isclose(semantics["sparse_active_weight_maintenance_rate"], 0.01)
    assert np.isclose(semantics["adam_weight_decay"], 0.0)
    assert np.isclose(
        semantics["inactive_stored_group_learning_rates"]["reactivation"],
        0.0001,
    )


def test_matched_bp_requires_explicit_null_local_learning_config(tmp_path):
    metadata = {"strategy": "standard", "broadcast": None}
    explicit_null = {
        "experiment": {"seed": 10800},
        "training": {"main": {"strategy": "standard", "learning_strategy_config": None}},
        "outputs": {"results_dir": str(tmp_path)},
    }
    errors = MODULE.audit_resolved_config(
        explicit_null, "bp", metadata, tmp_path
    )
    assert not any("matched BP" in error for error in errors)

    missing = {
        "experiment": {"seed": 10800},
        "training": {"main": {"strategy": "standard"}},
        "outputs": {"results_dir": str(tmp_path)},
    }
    errors = MODULE.audit_resolved_config(missing, "bp", metadata, tmp_path)
    assert any("explicitly as null" in error for error in errors)


def test_condition_name_accepts_native_sweep_variant():
    generated = {
        "_sweep_variant": "cifar10_raw_additive_exact_path_confirmatory"
    }
    name, metadata = MODULE.condition_from_config(generated)
    assert name == "cifar10_raw_additive_exact_path_confirmatory"
    assert metadata["label"] == "exact path"
