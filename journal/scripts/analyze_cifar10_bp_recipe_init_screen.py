#!/usr/bin/env python3
"""Audit the frozen CIFAR-10 BP recipe/initialization calibration screen.

Selection is based only on validation accuracy.  Test accuracy is read only
after all 48 expected runs are complete, and is used solely for the frozen
decision gate described in ``analysis/CIFAR10_BP_RECIPE_INITIALIZATION_AUDIT_20260828.md``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from omegaconf import OmegaConf


EXPECTED_SEEDS = (10700, 10701, 10702)
ARCHITECTURES = ("additive", "shunting")
POLICIES = ("fixed", "analytical", "empirical", "occupancy_quantile")
DATA_DRIVEN_POLICIES = frozenset({"empirical", "occupancy_quantile"})
RECIPES = ("archived", "local_matched")
EXPECTED_VARIANTS = tuple(
    f"{architecture}_{policy}_{recipe}"
    for architecture in ARCHITECTURES
    for policy in POLICIES
    for recipe in RECIPES
)
PROVISIONAL_ADDITIVE_BP_MEAN = 0.38976
MIN_ACCEPTABLE_ADDITIVE_BP_MEAN = 0.45
MIN_IMPROVEMENT_OVER_PROVISIONAL = 0.03
NEAR_TIE_TOLERANCE = 0.005


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_variant(variant: str) -> tuple[str, str, str]:
    match = re.fullmatch(
        r"(additive|shunting)_(fixed|analytical|empirical|occupancy_quantile)_"
        r"(archived|local_matched)",
        variant,
    )
    if match is None:
        raise ValueError(f"unexpected sweep variant: {variant}")
    return match.group(1), match.group(2), match.group(3)


def config_index(config) -> int:
    return int(str(config._sweep_config_id).rsplit("_", 1)[-1])


def _unique_rounded(values: list[float]) -> str:
    unique = sorted({round(float(value), 6) for value in values})
    return ";".join(f"{value:g}" for value in unique)


def _gate_stats(path: Path) -> dict[str, Any]:
    """Read one gate snapshot and return a stable, layer-resolved signature."""
    if not path.is_file():
        raise ValueError(f"missing {path.name}")
    payload = json.loads(path.read_text())
    modules = payload.get("reactivation_modules", {})
    if not modules:
        raise ValueError(f"no reactivation modules in {path.name}")

    layer_values: dict[str, tuple[float, float]] = {}
    for name, record in sorted(modules.items()):
        try:
            m_value = float(record["m"]["mean"])
            b_value = float(record["b"]["mean"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"invalid gate record {name!r} in {path.name}") from error
        if not np.isfinite(m_value) or not np.isfinite(b_value):
            raise ValueError(f"non-finite gate record {name!r} in {path.name}")
        # Calibration diagnostics name a branch layer; gate snapshots append
        # ``.reactivation``.  Normalize that suffix so the two artifacts can be
        # compared layer by layer.
        layer_name = name.removesuffix(".reactivation")
        layer_values[layer_name] = (m_value, b_value)

    m_values = [value[0] for value in layer_values.values()]
    b_values = [value[1] for value in layer_values.values()]
    # Six decimal places are far tighter than the recorded float32 precision
    # needed to identify an effective gate, while treating 0.099999994 and
    # 0.100000001 as the same applied value.
    signature = "|".join(
        f"{name}:m={m_value:.6f},b={b_value:.6f}"
        for name, (m_value, b_value) in layer_values.items()
    )
    return {
        "layer_values": layer_values,
        "summary": f"m={_unique_rounded(m_values)},b={_unique_rounded(b_values)}",
        "signature": signature,
        "sha256": sha256(path),
    }


def _post_calibration_matches(
    layers: dict[str, Any], post_stats: dict[str, Any]
) -> tuple[bool, str]:
    post_values = post_stats["layer_values"]
    if set(layers) != set(post_values):
        return False, "calibration/post-calibration layer sets differ"
    for name, record in layers.items():
        try:
            expected_m = float(record["m_applied"])
            expected_b = float(record["b_applied"])
        except (KeyError, TypeError, ValueError):
            return False, f"missing applied gate values for {name}"
        observed_m, observed_b = post_values[name]
        if not (
            np.isclose(expected_m, observed_m, rtol=1e-6, atol=1e-6)
            and np.isclose(expected_b, observed_b, rtol=1e-6, atol=1e-6)
        ):
            return False, f"post-calibration gate mismatch for {name}"
    return True, ""


def calibration_record(result_dir: Path, requested_policy: str) -> dict:
    """Return the requested and actually applied gate calibration."""
    stats_path = result_dir / "init_gate_stats.json"
    try:
        pre_stats = _gate_stats(stats_path)
    except (ValueError, json.JSONDecodeError) as error:
        return {
            "calibration_valid": False,
            "calibration_reverted": False,
            "calibration_reason": str(error),
            "calibration_converged": False,
            "calibration_iterations": 0,
            "calibration_reverted_layers": 0,
            "calibration_source": "invalid",
            "applied_gate": "unknown",
            "applied_gate_signature": "unknown",
            "gate_stats_sha256": "",
            "calibration_sha256": "",
            "post_gate_stats_sha256": "",
        }

    calibration_path = result_dir / "reactivation_calibration.json"
    post_stats_path = result_dir / "post_calibration_gate_stats.json"
    if requested_policy not in DATA_DRIVEN_POLICIES:
        unexpected = [
            path.name for path in (calibration_path, post_stats_path) if path.exists()
        ]
        valid = not unexpected
        return {
            "calibration_valid": valid,
            "calibration_reverted": False,
            "calibration_reason": (
                "unexpected data-driven artifacts: " + ", ".join(unexpected)
                if unexpected
                else ""
            ),
            "calibration_converged": True,
            "calibration_iterations": 0,
            "calibration_reverted_layers": 0,
            "calibration_source": "post-build fixed/analytical gate",
            "applied_gate": f"{requested_policy} ({pre_stats['summary']})",
            "applied_gate_signature": pre_stats["signature"],
            "gate_stats_sha256": pre_stats["sha256"],
            "calibration_sha256": "",
            "post_gate_stats_sha256": "",
        }

    if not calibration_path.is_file() or not post_stats_path.is_file():
        return {
            "calibration_valid": False,
            "calibration_reverted": False,
            "calibration_reason": "missing data-driven calibration artifacts",
            "calibration_converged": False,
            "calibration_iterations": 0,
            "calibration_reverted_layers": 0,
            "calibration_source": "invalid",
            "applied_gate": "unknown",
            "applied_gate_signature": "unknown",
            "gate_stats_sha256": pre_stats["sha256"],
            "calibration_sha256": "",
            "post_gate_stats_sha256": "",
        }

    try:
        calibration = json.loads(calibration_path.read_text())
        post_stats = _gate_stats(post_stats_path)
    except (ValueError, json.JSONDecodeError) as error:
        return {
            "calibration_valid": False,
            "calibration_reverted": False,
            "calibration_reason": str(error),
            "calibration_converged": False,
            "calibration_iterations": 0,
            "calibration_reverted_layers": 0,
            "calibration_source": "invalid",
            "applied_gate": "unknown",
            "applied_gate_signature": "unknown",
            "gate_stats_sha256": pre_stats["sha256"],
            "calibration_sha256": sha256(calibration_path),
            "post_gate_stats_sha256": (
                sha256(post_stats_path) if post_stats_path.is_file() else ""
            ),
        }

    layers = calibration.get("layers", {})
    policies = calibration.get("policies", [])
    converged = bool(calibration.get("converged", False))
    iterations = int(calibration.get("iterations_completed", 0))
    reverted_layers = [
        record for record in layers.values() if bool(record.get("calibration_reverted"))
    ]
    reasons = sorted(
        {
            str(record.get("calibration_revert_reason", "unspecified"))
            for record in reverted_layers
        }
    )
    reverted = bool(reverted_layers)
    undocumented_reversion = any(
        str(record.get("calibration_revert_reason", "")).strip().lower()
        in {"", "none", "unspecified"}
        for record in reverted_layers
    )
    post_matches, post_reason = _post_calibration_matches(layers, post_stats)
    policy_matches = policies == [requested_policy]
    # The frozen contract excludes reverted empirical fits. Occupancy fits may
    # remain eligible after a documented safety reversion and are named by the
    # post-calibration gate actually used for training.
    valid = bool(
        layers
        and converged
        and iterations > 0
        and policy_matches
        and post_matches
        and not undocumented_reversion
        and not (requested_policy == "empirical" and reverted)
    )
    invalid_reasons = []
    if not layers:
        invalid_reasons.append("no calibrated layers")
    if not converged:
        invalid_reasons.append("calibration did not converge")
    if iterations <= 0:
        invalid_reasons.append("invalid calibration iteration count")
    if not policy_matches:
        invalid_reasons.append(
            f"calibration policies {policies!r} do not match {requested_policy!r}"
        )
    if not post_matches:
        invalid_reasons.append(post_reason)
    if undocumented_reversion:
        invalid_reasons.append("reverted calibration lacks a documented reason")
    if requested_policy == "empirical" and reverted:
        invalid_reasons.append("empirical calibration reverted")
    if reverted:
        label = f"{requested_policy} safety reversion ({post_stats['summary']})"
    else:
        label = f"{requested_policy} calibrated ({post_stats['summary']})"
    return {
        "calibration_valid": valid,
        "calibration_reverted": reverted,
        "calibration_reason": ";".join(invalid_reasons or reasons),
        "calibration_converged": converged,
        "calibration_iterations": iterations,
        "calibration_reverted_layers": len(reverted_layers),
        "calibration_source": "post-calibration gate snapshot",
        "applied_gate": label,
        "applied_gate_signature": post_stats["signature"],
        "gate_stats_sha256": pre_stats["sha256"],
        "calibration_sha256": sha256(calibration_path),
        "post_gate_stats_sha256": post_stats["sha256"],
    }


def resolved_behavior_record(
    result_dir: Path, architecture: str, requested_policy: str, recipe: str
) -> dict[str, Any]:
    """Validate the resolved operator, initialization, and optimizer recipe."""
    path = result_dir / "config.json"
    if not path.is_file():
        raise ValueError("missing resolved config.json")
    payload = json.loads(path.read_text())
    try:
        core = payload["model"]["core"]
        implementation = core["implementation"]
        reactivation = core["reactivation"]
        morphology = core["morphology"]
        common = payload["training"]["main"]["common"]
        split_params = bool(common["param_groups"]["split_params"])
    except (KeyError, TypeError) as error:
        raise ValueError("resolved config is missing audited fields") from error

    expected_type = {
        "additive": "dendritic_additive",
        "shunting": "dendritic_shunting",
    }[architecture]
    expected_recipe = {
        "archived": {
            "epochs": 200,
            "patience": 40,
            "weight_decay_rate": 0.01,
            "split_params": False,
        },
        "local_matched": {
            "epochs": 400,
            "patience": 50,
            "weight_decay_rate": 0.0,
            "split_params": True,
        },
    }[recipe]
    observed_recipe = {
        "epochs": int(common["epochs"]),
        "patience": int(common["patience"]),
        "weight_decay_rate": float(common["weight_decay_rate"]),
        "split_params": split_params,
    }
    errors = []
    if str(core["type"]) != expected_type:
        errors.append(f"core type {core['type']!r} != {expected_type!r}")
    if str(reactivation["init_policy"]) != requested_policy:
        errors.append(
            f"resolved init policy {reactivation['init_policy']!r} "
            f"!= {requested_policy!r}"
        )
    if architecture == "additive" and bool(
        morphology.get("use_additive_normalization", False)
    ):
        errors.append("additive screen unexpectedly used normalized integration")
    if observed_recipe != expected_recipe:
        errors.append(
            f"resolved recipe {observed_recipe!r} != {expected_recipe!r}"
        )

    adaptive = bool(implementation["adaptive_initialization"])
    return {
        "resolved_behavior_valid": not errors,
        "resolved_behavior_reason": ";".join(errors),
        "adaptive_initialization": adaptive,
        "adaptive_initialization_policy": str(
            implementation["adaptive_initialization_policy"]
        ),
        "resolved_core_type": str(core["type"]),
        "resolved_init_policy": str(reactivation["init_policy"]),
        "resolved_epochs": observed_recipe["epochs"],
        "resolved_patience": observed_recipe["patience"],
        "resolved_weight_decay": observed_recipe["weight_decay_rate"],
        "resolved_split_params": observed_recipe["split_params"],
        "resolved_config_sha256": sha256(path),
    }


def training_record(result_dir: Path, configured_epochs: int) -> dict[str, Any]:
    """Validate the selected checkpoint trace without changing eligibility at cap."""
    path = result_dir / "training_summary.json"
    if not path.is_file():
        raise ValueError("missing training_summary.json")
    payload = json.loads(path.read_text())
    try:
        best_epoch = int(payload["best_epoch"])
        valid_losses = np.asarray(payload["valid_losses"], dtype=float)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("invalid training_summary.json") from error
    epochs_completed = len(valid_losses)
    valid = bool(
        1 <= best_epoch <= epochs_completed <= configured_epochs
        and epochs_completed > 0
        and np.isfinite(valid_losses).all()
    )
    if not valid:
        raise ValueError(
            "invalid best-epoch/validation-loss trace: "
            f"best={best_epoch}, completed={epochs_completed}, "
            f"configured={configured_epochs}"
        )
    return {
        "best_epoch": best_epoch,
        "epochs_completed": epochs_completed,
        "training_hit_epoch_cap": epochs_completed == configured_epochs,
        "training_summary_sha256": sha256(path),
    }


def collect(sweep_root: Path) -> tuple[pd.DataFrame, dict]:
    rows: list[dict] = []
    seen: set[tuple[str, int]] = set()
    unexpected: list[str] = []
    invalid: list[str] = []

    for config_path in sorted((sweep_root / "configs").glob("unified_config_*.yaml")):
        config = OmegaConf.load(config_path)
        variant = str(config.get("_sweep_variant", ""))
        try:
            architecture, policy, recipe = parse_variant(variant)
        except ValueError as error:
            unexpected.append(f"{config_path.name}: {error}")
            continue
        seed = int(config.experiment.seed)
        index = config_index(config)
        key = (variant, seed)
        if key in seen:
            invalid.append(f"duplicate {variant}/seed-{seed}")
            continue
        seen.add(key)

        result_dir = sweep_root / "results" / f"config_{index}"
        final_path = result_dir / "performance" / "final.json"
        checkpoint_path = result_dir / "main_network" / "standard_best_model.pt"
        if not final_path.is_file():
            continue
        try:
            payload = json.loads(final_path.read_text())
            validation_accuracy = float(payload["accuracy"]["valid"])
            test_accuracy = float(payload["accuracy"]["test"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            invalid.append(f"{final_path}: {error}")
            continue
        if not all(
            np.isfinite(value) and 0.0 <= value <= 1.0
            for value in (validation_accuracy, test_accuracy)
        ):
            invalid.append(f"{final_path}: non-finite or out-of-range accuracy")
            continue

        try:
            behavior = resolved_behavior_record(
                result_dir, architecture, policy, recipe
            )
            training = training_record(result_dir, behavior["resolved_epochs"])
        except (ValueError, json.JSONDecodeError) as error:
            invalid.append(f"{result_dir}: {error}")
            continue
        if not behavior["resolved_behavior_valid"]:
            invalid.append(
                f"{result_dir}: {behavior['resolved_behavior_reason']}"
            )
            continue
        calibration = calibration_record(result_dir, policy)
        run_eligible = bool(
            checkpoint_path.is_file()
            and calibration["calibration_valid"]
            and behavior["resolved_behavior_valid"]
        )
        if not checkpoint_path.is_file():
            invalid.append(f"{result_dir}: missing best checkpoint")
        rows.append(
            {
                "architecture": architecture,
                "policy": policy,
                "recipe": recipe,
                "variant": variant,
                "seed": seed,
                "config_index": index,
                "validation_accuracy": validation_accuracy,
                "test_accuracy": test_accuracy,
                "run_eligible": run_eligible,
                **behavior,
                **training,
                **calibration,
                "config_sha256": sha256(config_path),
                "result_sha256": sha256(final_path),
                "checkpoint_sha256": sha256(checkpoint_path)
                if checkpoint_path.is_file()
                else "",
            }
        )

    expected = {(variant, seed) for variant in EXPECTED_VARIANTS for seed in EXPECTED_SEEDS}
    missing = sorted(expected - seen)
    completed = {(row["variant"], row["seed"]) for row in rows}
    missing_results = sorted(expected - completed)
    audit = {
        "status": "complete_and_validated"
        if not (missing or missing_results or unexpected or invalid)
        else "incomplete_or_invalid",
        "n_expected": len(expected),
        "n_configs_seen": len(seen),
        "n_results_complete": len(rows),
        "missing_configs": [f"{variant}/seed-{seed}" for variant, seed in missing],
        "missing_results": [
            f"{variant}/seed-{seed}" for variant, seed in missing_results
        ],
        "unexpected": unexpected,
        "invalid": invalid,
    }
    frame = pd.DataFrame(rows)
    if len(frame):
        frame = frame.sort_values(
            ["architecture", "policy", "recipe", "seed"]
        ).reset_index(drop=True)
    return frame, audit


def summarize_cells(frame: pd.DataFrame) -> pd.DataFrame:
    records: list[dict] = []
    for (architecture, policy, recipe, variant), part in frame.groupby(
        ["architecture", "policy", "recipe", "variant"], sort=True
    ):
        complete = set(part.seed) == set(EXPECTED_SEEDS) and len(part) == len(EXPECTED_SEEDS)
        eligible = bool(complete and part.run_eligible.all())
        records.append(
            {
                "architecture": architecture,
                "policy": policy,
                "recipe": recipe,
                "variant": variant,
                "n_seeds": len(part),
                "eligible": eligible,
                "mean_validation_accuracy": float(part.validation_accuracy.mean()),
                "sd_validation_accuracy": float(part.validation_accuracy.std(ddof=1)),
                "mean_test_accuracy": float(part.test_accuracy.mean()),
                "sd_test_accuracy": float(part.test_accuracy.std(ddof=1)),
                "adaptive_initialization": bool(part.adaptive_initialization.all()),
                "adaptive_initialization_policies": " | ".join(
                    sorted(set(part.adaptive_initialization_policy))
                ),
                "all_calibrations_converged": bool(
                    part.calibration_converged.all()
                ),
                "max_calibration_iterations": int(
                    part.calibration_iterations.max()
                ),
                "calibration_reverted_layers": int(
                    part.calibration_reverted_layers.sum()
                ),
                "any_calibration_reverted": bool(part.calibration_reverted.any()),
                "calibration_sources": " | ".join(
                    sorted(set(part.calibration_source))
                ),
                "best_epoch_min": int(part.best_epoch.min()),
                "best_epoch_max": int(part.best_epoch.max()),
                "runs_hitting_epoch_cap": int(part.training_hit_epoch_cap.sum()),
                "applied_gates": " | ".join(sorted(set(part.applied_gate))),
                "applied_gate_signatures": " | ".join(
                    sorted(set(part.applied_gate_signature))
                ),
            }
        )
    return pd.DataFrame(records)


def select_by_validation(summary: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Apply the frozen selection rule without using test accuracy."""
    selected_rows: list[pd.Series] = []
    selection_notes: dict[str, dict] = {}
    for architecture in ARCHITECTURES:
        eligible = summary[
            summary.architecture.eq(architecture) & summary.eligible
        ].copy()
        if eligible.empty:
            selection_notes[architecture] = {"selected": None, "reason": "no eligible cell"}
            continue
        eligible["policy_priority"] = eligible.policy.map(
            {"fixed": 0, "analytical": 1, "empirical": 2, "occupancy_quantile": 3}
        )
        effective_duplicate_groups: list[list[str]] = []
        if "applied_gate_signatures" in eligible:
            effective_key = ["recipe", "applied_gate_signatures"]
            for _, group in eligible.groupby(effective_key, sort=True):
                if len(group) > 1:
                    effective_duplicate_groups.append(sorted(group.variant.tolist()))
            # A reverted occupancy arm and its analytical fallback are the same
            # current-code gate, not two independent candidates.
            eligible = (
                eligible.sort_values(["policy_priority", "variant"])
                .drop_duplicates(effective_key, keep="first")
                .copy()
            )

        best_mean = float(eligible.mean_validation_accuracy.max())
        near_best = eligible[
            eligible.mean_validation_accuracy >= best_mean - NEAR_TIE_TOLERANCE
        ].copy()
        near_best["recipe_priority"] = near_best.recipe.map(
            {"archived": 0, "local_matched": 1}
        )
        winner = near_best.sort_values(
            [
                "sd_validation_accuracy",
                "recipe_priority",
                "policy_priority",
                "variant",
            ],
            ascending=True,
        ).iloc[0]
        selected_rows.append(winner)
        selection_notes[architecture] = {
            "selected": str(winner.variant),
            "best_mean_validation_accuracy": best_mean,
            "n_cells_within_0.5pp": len(near_best),
            "selection_endpoint": "validation accuracy only",
            "collapsed_effective_gate_groups": effective_duplicate_groups,
        }

    selected = pd.DataFrame(selected_rows)
    if len(selected):
        selected = selected.drop(
            columns=["recipe_priority", "policy_priority"], errors="ignore"
        ).reset_index(drop=True)
    additive = selected[selected.architecture.eq("additive")]
    if additive.empty:
        passes = False
        additive_test = None
    else:
        additive_test = float(additive.iloc[0].mean_test_accuracy)
        passes = bool(
            additive_test >= MIN_ACCEPTABLE_ADDITIVE_BP_MEAN
            and additive_test - PROVISIONAL_ADDITIVE_BP_MEAN
            >= MIN_IMPROVEMENT_OVER_PROVISIONAL
        )
    decision = {
        "selection": selection_notes,
        "selected_additive_mean_test_accuracy": additive_test,
        "provisional_additive_bp_mean": PROVISIONAL_ADDITIVE_BP_MEAN,
        "minimum_acceptable_additive_mean": MIN_ACCEPTABLE_ADDITIVE_BP_MEAN,
        "minimum_improvement_over_provisional": MIN_IMPROVEMENT_OVER_PROVISIONAL,
        "launch_fresh_feedback_ladder": passes,
        "if_gate_fails": "exclude provisional CIFAR-10 ladder from the manuscript",
    }
    return selected, decision


def execution_identity(sweep_root: Path) -> dict:
    launcher = sweep_root / "jobs" / "run_array_sweep.sh"
    text = launcher.read_text()

    def value(name: str) -> str:
        match = re.search(
            rf'^{name}=(?:"([^"]+)"|([^\s]+))$', text, flags=re.MULTILINE
        )
        if match is None:
            raise RuntimeError(f"missing {name} in {launcher}")
        return match.group(1) or match.group(2)

    manifest_candidates = (
        sweep_root / "frozen_sweep_manifest.json",
        sweep_root / "original_manifest.yaml",
        sweep_root / "original_config.yaml",
    )
    manifest = next((path for path in manifest_candidates if path.is_file()), None)
    if manifest is None:
        raise RuntimeError(f"missing frozen manifest in {sweep_root}")
    return {
        "source_worktree": value("REPOSITORY_ROOT"),
        "source_commit": value("EXPECTED_REPOSITORY_HEAD"),
        "source_tracked_diff_sha256": value("EXPECTED_TRACKED_DIFF_SHA256"),
        "launcher_sha256": sha256(launcher),
        "manifest_sha256": sha256(manifest),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("sweep_root", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()

    frame, audit = collect(args.sweep_root)
    if audit["status"] != "complete_and_validated" and not args.allow_incomplete:
        raise RuntimeError(json.dumps(audit, indent=2))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_dir / "seed_outcomes.csv", index=False)
    record = {
        "audit": audit,
        "sweep_root": str(args.sweep_root),
        "execution_identity": execution_identity(args.sweep_root),
        "selection_frozen_before_outcome_inspection": True,
        "analyzer_correction": {
            "version": 2,
            "selection_algorithm_changed": False,
            "reason": (
                "Version 1 read pre-calibration init_gate_stats.json when "
                "labeling data-driven policies. Version 2 reads and validates "
                "reactivation_calibration.json plus "
                "post_calibration_gate_stats.json, audits resolved initialization "
                "and training behavior, and only collapses cells with matching "
                "post-calibration gate signatures."
            ),
            "analyzer_sha256": sha256(Path(__file__)),
        },
    }
    if audit["status"] == "complete_and_validated":
        summary = summarize_cells(frame)
        selected, decision = select_by_validation(summary)
        summary.to_csv(args.output_dir / "cell_summary.csv", index=False)
        selected.to_csv(args.output_dir / "selected_cells.csv", index=False)
        record["decision"] = decision
    (args.output_dir / "summary.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
