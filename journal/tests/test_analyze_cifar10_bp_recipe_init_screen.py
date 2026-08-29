from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd


SCRIPT = (
    Path(__file__).parents[1]
    / "scripts"
    / "analyze_cifar10_bp_recipe_init_screen.py"
)
SPEC = importlib.util.spec_from_file_location(
    "analyze_cifar10_bp_recipe_init_screen", SCRIPT
)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _cell(
    architecture,
    policy,
    recipe,
    validation,
    test,
    sd=0.01,
    eligible=True,
    gate_signature=None,
):
    return {
        "architecture": architecture,
        "policy": policy,
        "recipe": recipe,
        "variant": f"{architecture}_{policy}_{recipe}",
        "eligible": eligible,
        "mean_validation_accuracy": validation,
        "sd_validation_accuracy": sd,
        "mean_test_accuracy": test,
        "applied_gate_signatures": gate_signature or policy,
    }


def _write_gate_stats(path: Path, m: float, b: float):
    path.write_text(
        json.dumps(
            {
                "reactivation_modules": {
                    "layer.reactivation": {
                        "m": {"mean": m},
                        "b": {"mean": b},
                    }
                }
            }
        )
    )


def _write_calibration(
    result_dir: Path,
    policy: str,
    m: float,
    b: float,
    *,
    converged: bool = True,
    reverted: bool = False,
    reason: str = "none",
):
    (result_dir / "reactivation_calibration.json").write_text(
        json.dumps(
            {
                "policies": [policy],
                "converged": converged,
                "iterations_completed": 3,
                "layers": {
                    "layer": {
                        "m_applied": m,
                        "b_applied": b,
                        "calibration_reverted": reverted,
                        "calibration_revert_reason": reason,
                    }
                },
            }
        )
    )
    _write_gate_stats(result_dir / "post_calibration_gate_stats.json", m, b)


def test_selection_uses_near_tie_sd_then_frozen_priorities():
    frame = pd.DataFrame(
        [
            _cell("additive", "fixed", "archived", 0.500, 0.46, sd=0.012),
            _cell("additive", "analytical", "local_matched", 0.504, 0.10, sd=0.010),
            _cell("shunting", "fixed", "archived", 0.520, 0.50, sd=0.011),
            _cell("shunting", "empirical", "archived", 0.521, 0.51, sd=0.011),
        ]
    )
    selected, decision = MODULE.select_by_validation(frame)
    winners = selected.set_index("architecture").variant.to_dict()
    # A 0.4-point validation near-tie is resolved by lower validation SD, even
    # though its test value is deliberately much worse.
    assert winners["additive"] == "additive_analytical_local_matched"
    # Remaining ties favor the archived recipe and then fixed initialization.
    assert winners["shunting"] == "shunting_fixed_archived"
    assert not decision["launch_fresh_feedback_ladder"]


def test_additive_gate_requires_both_absolute_and_relative_thresholds():
    base = [
        _cell("shunting", "fixed", "archived", 0.50, 0.50),
    ]
    low_absolute = pd.DataFrame(
        [_cell("additive", "fixed", "archived", 0.50, 0.449), *base]
    )
    _, decision = MODULE.select_by_validation(low_absolute)
    assert not decision["launch_fresh_feedback_ladder"]

    passing = pd.DataFrame(
        [_cell("additive", "fixed", "archived", 0.50, 0.46), *base]
    )
    _, decision = MODULE.select_by_validation(passing)
    assert decision["launch_fresh_feedback_ladder"]


def test_selection_collapses_requested_policies_with_the_same_applied_gate():
    frame = pd.DataFrame(
        [
            _cell(
                "additive", "analytical", "archived", 0.48, 0.48,
                gate_signature="m=0.1,b=0",
            ),
            _cell(
                "additive", "occupancy_quantile", "archived", 0.49, 0.49,
                gate_signature="m=0.1,b=0",
            ),
            _cell("shunting", "fixed", "archived", 0.50, 0.50),
        ]
    )
    selected, decision = MODULE.select_by_validation(frame)
    winner = selected.set_index("architecture").loc["additive", "variant"]
    assert winner == "additive_analytical_archived"
    duplicates = decision["selection"]["additive"]["collapsed_effective_gate_groups"]
    assert duplicates == [[
        "additive_analytical_archived",
        "additive_occupancy_quantile_archived",
    ]]


def test_empirical_reversion_is_ineligible_but_occupancy_reversion_is_allowed(tmp_path):
    result_dir = tmp_path / "result"
    result_dir.mkdir()
    _write_gate_stats(result_dir / "init_gate_stats.json", 0.1, 0.0)
    _write_calibration(
        result_dir,
        "empirical",
        0.1,
        0.0,
        reverted=True,
        reason="safety",
    )
    empirical = MODULE.calibration_record(result_dir, "empirical")
    _write_calibration(
        result_dir,
        "occupancy_quantile",
        0.1,
        0.0,
        reverted=True,
        reason="safety",
    )
    occupancy = MODULE.calibration_record(result_dir, "occupancy_quantile")
    assert not empirical["calibration_valid"]
    assert occupancy["calibration_valid"]
    assert "safety reversion" in occupancy["applied_gate"]


def test_empirical_label_and_signature_use_post_calibration_snapshot(tmp_path):
    result_dir = tmp_path / "result"
    result_dir.mkdir()
    _write_gate_stats(result_dir / "init_gate_stats.json", 0.1, 0.0)
    _write_calibration(result_dir, "empirical", 21.7, 1.5)

    record = MODULE.calibration_record(result_dir, "empirical")

    assert record["calibration_valid"]
    assert record["calibration_converged"]
    assert record["calibration_source"] == "post-calibration gate snapshot"
    assert "21.7" in record["applied_gate"]
    assert "b=1.5" in record["applied_gate"]
    assert "m=21.700000,b=1.500000" in record["applied_gate_signature"]
    assert "m=0.100000" not in record["applied_gate_signature"]


def test_nonconverged_or_post_mismatched_calibration_is_ineligible(tmp_path):
    result_dir = tmp_path / "result"
    result_dir.mkdir()
    _write_gate_stats(result_dir / "init_gate_stats.json", 0.1, 0.0)
    _write_calibration(
        result_dir, "empirical", 2.0, 1.0, converged=False
    )
    nonconverged = MODULE.calibration_record(result_dir, "empirical")
    assert not nonconverged["calibration_valid"]
    assert "did not converge" in nonconverged["calibration_reason"]

    _write_calibration(result_dir, "empirical", 2.0, 1.0)
    _write_gate_stats(result_dir / "post_calibration_gate_stats.json", 3.0, 1.0)
    mismatched = MODULE.calibration_record(result_dir, "empirical")
    assert not mismatched["calibration_valid"]
    assert "mismatch" in mismatched["calibration_reason"]


def test_analytical_uses_post_build_gate_without_calibration_artifacts(tmp_path):
    result_dir = tmp_path / "result"
    result_dir.mkdir()
    _write_gate_stats(result_dir / "init_gate_stats.json", 0.1, 0.0)

    record = MODULE.calibration_record(result_dir, "analytical")

    assert record["calibration_valid"]
    assert record["calibration_iterations"] == 0
    assert record["calibration_source"] == "post-build fixed/analytical gate"
    assert "m=0.100000,b=0.000000" in record["applied_gate_signature"]


def test_empirical_is_not_collapsed_with_reverted_occupancy(tmp_path):
    empirical_dir = tmp_path / "empirical"
    occupancy_dir = tmp_path / "occupancy"
    analytical_dir = tmp_path / "analytical"
    for result_dir in (empirical_dir, occupancy_dir, analytical_dir):
        result_dir.mkdir()
        _write_gate_stats(result_dir / "init_gate_stats.json", 0.1, 0.0)
    _write_calibration(empirical_dir, "empirical", 21.7, 1.5)
    _write_calibration(
        occupancy_dir,
        "occupancy_quantile",
        0.1,
        0.0,
        reverted=True,
        reason="would_exceed_max_m",
    )
    empirical = MODULE.calibration_record(empirical_dir, "empirical")
    occupancy = MODULE.calibration_record(occupancy_dir, "occupancy_quantile")
    analytical = MODULE.calibration_record(analytical_dir, "analytical")
    assert empirical["applied_gate_signature"] != occupancy["applied_gate_signature"]
    assert analytical["applied_gate_signature"] == occupancy["applied_gate_signature"]

    frame = pd.DataFrame(
        [
            _cell(
                "additive", "empirical", "archived", 0.50, 0.50,
                gate_signature=empirical["applied_gate_signature"],
            ),
            _cell(
                "additive", "analytical", "archived", 0.48, 0.48,
                gate_signature=analytical["applied_gate_signature"],
            ),
            _cell(
                "additive", "occupancy_quantile", "archived", 0.49, 0.49,
                gate_signature=occupancy["applied_gate_signature"],
            ),
            _cell("shunting", "fixed", "archived", 0.50, 0.50),
        ]
    )
    selected, decision = MODULE.select_by_validation(frame)
    assert (
        selected.set_index("architecture").loc["additive", "variant"]
        == "additive_empirical_archived"
    )
    duplicates = decision["selection"]["additive"][
        "collapsed_effective_gate_groups"
    ]
    assert duplicates == [[
        "additive_analytical_archived",
        "additive_occupancy_quantile_archived",
    ]]
