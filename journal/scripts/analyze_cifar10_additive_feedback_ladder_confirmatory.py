#!/usr/bin/env python3
"""Audit and analyze the frozen CIFAR-10 raw-additive feedback ladder.

The analyzer refuses to treat incomplete, misconfigured, or provenance-invalid
cohorts as confirmatory.  Statistical decisions are evaluated only after all 80
paired runs are present.  See
``analysis/CIFAR10_ADDITIVE_FEEDBACK_LADDER_CONFIRMATORY_20260828.md``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from scipy import stats


EXPECTED_SEEDS = tuple(range(10800, 10820))
EXPECTED_SOURCE_COMMIT = "e516c7fec3169253ff8c14bc5f4ab1325469e4f5"
EMPTY_TRACKED_DIFF_SHA256 = hashlib.sha256(b"").hexdigest()
PROJECT_B_ROOT = Path(
    "/n/holylfs06/LABS/kempner_project_b/Lab/dendritic/HS/LOCAL_LEARNING/"
    "journal_extension_20260828"
)
PROJECT_B_STORAGE_ROOT = Path(
    "/n/holylfs06/LABS/kempner_project_b/Lab/dendritic/HS/LOCAL_LEARNING"
)
SELECTED_SCREEN_SUMMARY = (
    PROJECT_B_ROOT
    / "analysis/cifar10_bp_recipe_init_screen_20260828_v2_postcalibration/summary.json"
)
EXPECTED_SCREEN_ANALYZER_CORRECTION_VERSION = 2
EXPECTED_SCREEN_ANALYZER_SHA256 = (
    "e938929556831cdf51323750cdf2e45588c2eaa36a1073552c47deafa27bb255"
)
SELECTED_SCREEN_CONFIG = (
    PROJECT_B_ROOT
    / "sweep_runs/cifar10_bp_recipe_init_screen/"
    "journal_cifar10_bp_recipe_init_screen_20260828101020/results/config_12/config.json"
)

CONDITIONS = {
    "cifar10_raw_additive_strict_scalar_confirmatory": {
        "label": "strict scalar",
        "strategy": "local_ca",
        "broadcast": "scalar",
    },
    "cifar10_raw_additive_neuron_specific_confirmatory": {
        "label": "neuron specific",
        "strategy": "local_ca",
        "broadcast": "per_soma_shared",
    },
    "cifar10_raw_additive_exact_path_confirmatory": {
        "label": "exact path",
        "strategy": "local_ca",
        "broadcast": "path_transport",
    },
    "cifar10_raw_additive_matched_bp_confirmatory": {
        "label": "backpropagation",
        "strategy": "standard",
        "broadcast": None,
    },
}
CONDITION_ORDER = (
    "strict scalar",
    "neuron specific",
    "exact path",
    "backpropagation",
)
SUPERIORITY_CONTRASTS = (
    ("neuron specific minus strict scalar", "neuron specific", "strict scalar"),
    ("exact path minus neuron specific", "exact path", "neuron specific"),
)
EQUIVALENCE_CONTRAST = (
    "exact path minus backpropagation",
    "exact path",
    "backpropagation",
)
EQUIVALENCE_MARGIN = 0.01
BP_ADEQUACY_THRESHOLD = 0.45
ALPHA = 0.05
EXPECTED_GENERATOR = (
    "dendritic_modeling.scripts.sweeps.unified.sweep_types.unified_sweep."
    "UnifiedSweepGenerator"
)
EXPECTED_INPUT_YAML_SHA256 = (
    "94512fdf74b345415457dbbee8dfa496b5c82bcb509348141e2bc2cfac6253e6"
)
EXPECTED_MANIFEST_SHA256 = (
    "97a35b333c8d74885298cffc2fc4f0ebcc48c8ee55b8d13f45a0474e9ca52222"
)
EXPECTED_LAUNCHER_SHA256 = (
    "c5df7886a1d530a046e6dd7f357b624f31854455cc94e576c5c4504c9dd0b1af"
)


EXPECTED_RESOLVED_FIELDS = {
    "experiment.train_valid_split": 0.9,
    "experiment.enable_profiling": False,
    "experiment.enable_hooks": True,
    "experiment.deterministic": True,
    "experiment.allow_tf32": False,
    "data.dataset_name": "cifar10",
    "data.processing.flatten": True,
    "data.processing.normalize": False,
    "model.task": "classification",
    "model.encoder.type": "identity",
    "model.encoder.params.input_dim": 3072,
    "model.core.type": "dendritic_additive",
    "model.core.architecture.excitatory_layer_sizes": [20],
    "model.core.architecture.inhibitory_layer_sizes": [20],
    "model.core.architecture.excitatory_branch_factors": [3, 3, 3, 3],
    "model.core.architecture.inhibitory_branch_factors": [1],
    "model.core.architecture.inhibitory_network_type": "dendritic",
    "model.core.connectivity.ee_synapses_per_branch_per_layer": [25],
    "model.core.connectivity.ei_synapses_per_branch_per_layer": [25],
    "model.core.connectivity.ie_synapses_per_branch_per_layer": [25],
    "model.core.connectivity.ii_synapses_per_branch_per_layer": [0],
    "model.core.transfer.input_mode": 1,
    "model.core.transfer.independent_pathways": False,
    "model.core.transfer.output_activation": "relu",
    "model.core.transfer.inhibitory_mode": "first",
    "model.core.morphology.somatic_synapses": False,
    "model.core.morphology.use_shunting": False,
    "model.core.morphology.use_additive_normalization": False,
    "model.core.morphology.additive_mode": "raw",
    "model.core.morphology.weight_transform": "softplus",
    "model.core.implementation.adaptive_initialization": True,
    "model.core.implementation.adaptive_initialization_policy": (
        "preserve_shunting_center"
    ),
    "model.core.implementation.adaptive_target_conductance": 5.0,
    "model.core.implementation.initial_child_conductance": 1.0,
    "model.core.reactivation.enabled": True,
    "model.core.reactivation.type": "param_tanh",
    "model.core.reactivation.init_policy": "empirical",
    "model.core.reactivation.init_m": 1.5,
    "model.core.reactivation.init_b": 0.5,
    "model.core.reactivation.calibration_min_quantile_width": 0.001,
    "model.core.reactivation.calibration_max_m": 50.0,
    "model.core.reactivation.calibration_revert_on_invalid": True,
    "model.decoder.type": "MLP",
    "model.decoder.params.input_dim": 20,
    "model.decoder.params.hidden_dims": [32, 16],
    "model.decoder.params.activation": "relu",
    "model.decoder.params.output_dim": 10,
    "training.main.common.epochs": 200,
    "training.main.common.batch_size": 256,
    "training.main.common.shuffle": True,
    "training.main.common.grad_clip_value": 5.0,
    "training.main.common.loss_function": "cat_nll",
    "training.main.common.early_stopping": True,
    "training.main.common.patience": 40,
    "training.main.common.load_best_state_dict": True,
    "training.main.common.use_amp": False,
    "training.main.common.lr_schedule": "none",
    "training.main.common.lr_warmup_epochs": 0,
    "training.main.common.reactivation_initialization_batch_size": 256,
    "training.main.common.reactivation_initialization_num_batches": 3,
    "training.main.common.weight_decay_rate": 0.01,
    "training.main.common.recalibrate_reactivation_every": 0,
    "training.main.common.reactivation_update_mode": "backprop",
    "training.main.common.param_groups.lr": 0.001,
    "training.main.common.param_groups.split_params": False,
    "training.main.common.param_groups.topk_lr": 0.001,
    "training.main.common.param_groups.blocklinear_lr": 0.0001,
    "training.main.common.param_groups.reactivation_lr": 0.0001,
    "training.main.common.param_groups.decoder_lr": 0.001,
    "training.main.optimizer.name": "adam",
    "training.main.optimizer.lr": 0.001,
    "training.main.optimizer.weight_decay": 0.0,
    "wandb.use_wandb": False,
}

LOCAL_EXPECTED_FIELDS = {
    "training.main.learning_strategy_config.rule_variant": "5f",
    "training.main.learning_strategy_config.error_mode": "auto",
    "training.main.learning_strategy_config.decoder_update_mode": "backprop",
    "training.main.learning_strategy_config.update_reactivation": True,
    "training.main.learning_strategy_config.update_inactive_weights": False,
    "training.main.learning_strategy_config.normalize_by_batch": True,
    "training.main.learning_strategy_config.three_factor.dynamics_mode": "auto",
    "training.main.learning_strategy_config.three_factor.use_conductance_scaling": True,
    "training.main.learning_strategy_config.three_factor.use_driving_force": True,
    "training.main.learning_strategy_config.three_factor.additive_gain_mode": "none",
    "training.main.learning_strategy_config.morphology_aware.use_path_propagation": False,
    "training.main.learning_strategy_config.morphology_aware.morphology_modulator_mode": "none",
    "training.main.learning_strategy_config.morphology_aware.use_dendritic_normalization": False,
}

_MISSING = object()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def nested_get(mapping: dict, dotted_path: str) -> Any:
    value: Any = mapping
    for component in dotted_path.split("."):
        if not isinstance(value, dict) or component not in value:
            return _MISSING
        value = value[component]
    return value


def values_equal(observed: Any, expected: Any) -> bool:
    if observed is _MISSING:
        return False
    if isinstance(expected, float):
        try:
            return bool(math.isclose(float(observed), expected, rel_tol=0, abs_tol=1e-12))
        except (TypeError, ValueError):
            return False
    return observed == expected


def field_errors(config: dict, expected: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for path, target in expected.items():
        observed = nested_get(config, path)
        if not values_equal(observed, target):
            rendered = "<missing>" if observed is _MISSING else repr(observed)
            errors.append(f"{path}: expected {target!r}, observed {rendered}")
    return errors


def config_index(config: Any) -> int:
    return int(str(config._sweep_config_id).rsplit("_", 1)[-1])


def condition_from_config(config: Any) -> tuple[str, dict]:
    group = str(
        config.get("_seed_repeat_group", "") or config.get("_sweep_variant", "")
    )
    if group not in CONDITIONS:
        raise ValueError(f"unexpected condition {group!r}")
    return group, CONDITIONS[group]


def selected_screen_evidence() -> dict:
    record = {
        "summary_path": str(SELECTED_SCREEN_SUMMARY),
        "selected_bp_config_path": str(SELECTED_SCREEN_CONFIG),
        "valid": False,
        "errors": [],
    }
    if not SELECTED_SCREEN_SUMMARY.is_file():
        record["errors"].append("missing frozen screen summary")
        return record
    if not SELECTED_SCREEN_CONFIG.is_file():
        record["errors"].append("missing selected screen resolved config")
        return record
    summary = json.loads(SELECTED_SCREEN_SUMMARY.read_text())
    selected = nested_get(summary, "decision.selection.additive.selected")
    source_commit = nested_get(summary, "execution_identity.source_commit")
    correction_version = nested_get(summary, "analyzer_correction.version")
    correction_sha = nested_get(summary, "analyzer_correction.analyzer_sha256")
    if selected != "additive_empirical_archived":
        record["errors"].append(f"selected cell changed: {selected!r}")
    if source_commit != EXPECTED_SOURCE_COMMIT:
        record["errors"].append(f"screen source changed: {source_commit!r}")
    if correction_version != EXPECTED_SCREEN_ANALYZER_CORRECTION_VERSION:
        record["errors"].append(
            f"screen analyzer correction version changed: {correction_version!r}"
        )
    if correction_sha != EXPECTED_SCREEN_ANALYZER_SHA256:
        record["errors"].append(
            f"screen analyzer SHA changed: {correction_sha!r}"
        )
    reference = json.loads(SELECTED_SCREEN_CONFIG.read_text())
    reference_errors = field_errors(reference, EXPECTED_RESOLVED_FIELDS)
    record["errors"].extend(f"selected BP reference: {item}" for item in reference_errors)
    record.update(
        {
            "summary_sha256": sha256(SELECTED_SCREEN_SUMMARY),
            "selected_bp_config_sha256": sha256(SELECTED_SCREEN_CONFIG),
            "selected_cell": selected if selected is not _MISSING else None,
            "source_commit": source_commit if source_commit is not _MISSING else None,
            "analyzer_correction_version": (
                correction_version if correction_version is not _MISSING else None
            ),
            "analyzer_sha256": correction_sha if correction_sha is not _MISSING else None,
        }
    )
    record["valid"] = not record["errors"]
    return record


def execution_identity(sweep_root: Path) -> dict:
    launcher = sweep_root / "jobs" / "run_array_sweep.sh"
    manifest_path = sweep_root / "frozen_sweep_manifest.json"
    errors: list[str] = []
    if not launcher.is_file():
        return {"valid": False, "errors": [f"missing {launcher}"]}
    text = launcher.read_text()

    def shell_value(name: str) -> str | None:
        match = re.search(
            rf'^{name}=(?:"([^"]+)"|([^\s]+))$', text, flags=re.MULTILINE
        )
        return None if match is None else (match.group(1) or match.group(2))

    source_root = shell_value("REPOSITORY_ROOT")
    source_commit = shell_value("EXPECTED_REPOSITORY_HEAD")
    tracked_diff = shell_value("EXPECTED_TRACKED_DIFF_SHA256")
    output_dir = shell_value("OUTPUT_DIR")
    if source_commit != EXPECTED_SOURCE_COMMIT:
        errors.append(f"source commit {source_commit!r} != {EXPECTED_SOURCE_COMMIT}")
    if tracked_diff != EMPTY_TRACKED_DIFF_SHA256:
        errors.append(f"tracked diff is not empty: {tracked_diff!r}")
    if source_root is None or not str(source_root).startswith(
        str(PROJECT_B_STORAGE_ROOT)
    ):
        errors.append(f"source worktree is not on project-B storage: {source_root!r}")
    if output_dir is None or not str(output_dir).startswith(str(PROJECT_B_ROOT)):
        errors.append(f"output directory is not on project-B storage: {output_dir!r}")
    required_fragments = (
        "#SBATCH --partition=kempner_h100_priority",
        "#SBATCH --account=kempner_bsabatini_lab",
        "#SBATCH --array=0-79%8",
        "Mount canary",
    )
    for fragment in required_fragments:
        if fragment not in text:
            errors.append(f"launcher missing {fragment!r}")
    launcher_sha = sha256(launcher)
    if launcher_sha != EXPECTED_LAUNCHER_SHA256:
        errors.append(f"launcher SHA changed: {launcher_sha}")

    manifest: dict[str, Any] = {}
    manifest_sha = None
    if not manifest_path.is_file():
        errors.append("missing frozen_sweep_manifest.json")
    else:
        manifest_sha = sha256(manifest_path)
        if manifest_sha != EXPECTED_MANIFEST_SHA256:
            errors.append(f"frozen manifest SHA changed: {manifest_sha}")
        try:
            manifest = json.loads(manifest_path.read_text())
        except json.JSONDecodeError as error:
            errors.append(f"malformed frozen manifest: {error}")

    if manifest:
        if manifest.get("expected_config_count") != 80:
            errors.append(
                "frozen manifest expected_config_count is "
                f"{manifest.get('expected_config_count')!r}, not 80"
            )
        if manifest.get("generation_mode") != "generate_only":
            errors.append(
                f"unexpected generation mode {manifest.get('generation_mode')!r}"
            )
        if manifest.get("generator") != EXPECTED_GENERATOR:
            errors.append(f"unexpected generator {manifest.get('generator')!r}")
        scheduler = manifest.get("scheduler_profile", {})
        if scheduler.get("account") != "kempner_bsabatini_lab":
            errors.append(f"unexpected manifest account {scheduler.get('account')!r}")
        if scheduler.get("partition") != "kempner_h100_priority":
            errors.append(
                f"unexpected manifest partition {scheduler.get('partition')!r}"
            )
        source_git = manifest.get("source_identity", {}).get("git", {})
        if source_git.get("commit") != EXPECTED_SOURCE_COMMIT:
            errors.append(f"manifest source commit changed: {source_git.get('commit')!r}")
        if source_git.get("tracked_diff_sha256") != EMPTY_TRACKED_DIFF_SHA256:
            errors.append("manifest records a non-empty tracked source diff")
        if source_git.get("tracked_worktree_dirty") is not False:
            errors.append("manifest records a dirty tracked source worktree")

        original = manifest.get("original_yaml", {})
        if original.get("sha256") != EXPECTED_INPUT_YAML_SHA256:
            errors.append(f"input YAML SHA changed: {original.get('sha256')!r}")
        original_path = Path(str(original.get("path", "")))
        if not original_path.is_file():
            errors.append(f"missing frozen input YAML {original_path}")
        elif sha256(original_path) != EXPECTED_INPUT_YAML_SHA256:
            errors.append("current input YAML no longer matches its frozen SHA")

        resolved = manifest.get("resolved_original_yaml", {})
        resolved_path = sweep_root / str(resolved.get("path", ""))
        if not resolved_path.is_file():
            errors.append(f"missing resolved sweep YAML {resolved_path}")
        elif sha256(resolved_path) != resolved.get("sha256"):
            errors.append("resolved sweep YAML hash does not match the manifest")

        generated = manifest.get("generated_configs", [])
        indices = [
            item.get("index")
            for item in generated
            if isinstance(item, dict) and isinstance(item.get("index"), int)
        ]
        if len(generated) != 80 or sorted(indices) != list(range(80)):
            errors.append("manifest does not enumerate exactly generated indices 0--79")
        for item in generated:
            if not isinstance(item, dict):
                errors.append("manifest contains a malformed generated-config record")
                continue
            generated_path = sweep_root / str(item.get("path", ""))
            if not generated_path.is_file():
                errors.append(f"missing generated config {generated_path}")
            elif sha256(generated_path) != item.get("sha256"):
                errors.append(f"generated config hash changed: {generated_path.name}")

        source_root_path = Path(str(source_root)) if source_root else None
        for item in manifest.get("source_identity", {}).get("files", []):
            if not isinstance(item, dict):
                errors.append("manifest contains a malformed source-file record")
                continue
            declared_path = str(item.get("path", ""))
            if declared_path.startswith("repo:") and source_root_path is not None:
                source_path = source_root_path / declared_path.removeprefix("repo:")
            else:
                source_path = Path(declared_path)
            if not source_path.is_file():
                errors.append(f"missing frozen source file {declared_path}")
            elif sha256(source_path) != item.get("sha256"):
                errors.append(f"frozen source hash changed: {declared_path}")
    return {
        "valid": not errors,
        "errors": errors,
        "source_worktree": source_root,
        "source_commit": source_commit,
        "source_tracked_diff_sha256": tracked_diff,
        "output_dir": output_dir,
        "launcher_sha256": launcher_sha,
        "manifest_path": str(manifest_path) if manifest_path.is_file() else None,
        "manifest_sha256": manifest_sha,
    }


def calibration_record(result_dir: Path) -> dict:
    path = result_dir / "reactivation_calibration.json"
    init_path = result_dir / "init_gate_stats.json"
    post_path = result_dir / "post_calibration_gate_stats.json"
    errors: list[str] = []
    if not path.is_file():
        return {"valid": False, "errors": ["missing reactivation_calibration.json"]}
    if not init_path.is_file():
        errors.append("missing init_gate_stats.json")
    if not post_path.is_file():
        errors.append("missing post_calibration_gate_stats.json")
    payload = json.loads(path.read_text())
    if not bool(payload.get("requested")):
        errors.append("empirical calibration was not requested")
    if "empirical" not in payload.get("policies", []):
        errors.append(f"calibration policies are {payload.get('policies')!r}")
    if not bool(payload.get("converged")):
        errors.append("empirical calibration did not converge")
    layers = payload.get("layers", {})
    if len(layers) != 5:
        errors.append(f"expected five calibrated layers, observed {len(layers)}")
    signature_layers: dict[str, dict[str, float]] = {}
    for name, layer in sorted(layers.items()):
        if bool(layer.get("calibration_reverted")):
            errors.append(f"{name}: calibration reverted")
        values: dict[str, float] = {}
        for key in ("m_applied", "b_applied"):
            try:
                value = float(layer[key])
            except (KeyError, TypeError, ValueError):
                errors.append(f"{name}: missing or invalid {key}")
                continue
            if not np.isfinite(value):
                errors.append(f"{name}: non-finite {key}")
            values[key] = value
        signature_layers[name] = values
    fit_signature = {
        "mode": payload.get("mode"),
        "n_batches": payload.get("n_batches"),
        "layers": signature_layers,
    }
    post_signature: list[dict[str, Any]] = []
    if post_path.is_file():
        post_payload = json.loads(post_path.read_text())
        post_modules = post_payload.get("reactivation_modules", {})
        if len(post_modules) != 5:
            errors.append(
                "post-calibration statistics contain "
                f"{len(post_modules)} modules rather than five"
            )
        normalized_modules: dict[int, dict[str, Any]] = {}
        for module_name, module in post_modules.items():
            match = re.search(r"branch_layers\.(\d+)\.reactivation$", module_name)
            if match is None:
                errors.append(f"cannot normalize post-calibration module {module_name!r}")
                continue
            depth = int(match.group(1))
            if depth in normalized_modules:
                errors.append(f"duplicate post-calibration depth {depth}")
                continue
            try:
                record = {
                    "depth": depth,
                    "class": str(module["class"]),
                    "n": int(module["m"]["n"]),
                    # Rounding makes the signature insensitive to harmless JSON
                    # serialization and construction-order differences while
                    # preserving sub-ppm gate discrepancies.
                    "m_mean": round(float(module["m"]["mean"]), 8),
                    "b_mean": round(float(module["b"]["mean"]), 8),
                }
            except (KeyError, TypeError, ValueError) as error:
                errors.append(f"{module_name}: malformed post-calibration stats ({error})")
                continue
            if not all(np.isfinite(record[key]) for key in ("m_mean", "b_mean")):
                errors.append(f"{module_name}: non-finite post-calibration gate")
            normalized_modules[depth] = record
        post_signature = [normalized_modules[key] for key in sorted(normalized_modules)]

        fit_by_depth: dict[int, dict[str, float]] = {}
        for layer_name, values in signature_layers.items():
            match = re.search(r"branch_layers\.(\d+)$", layer_name)
            if match is None:
                errors.append(f"cannot normalize calibration layer {layer_name!r}")
                continue
            fit_by_depth[int(match.group(1))] = values
        for record in post_signature:
            fit = fit_by_depth.get(int(record["depth"]))
            if fit is None:
                errors.append(
                    f"post-calibration depth {record['depth']} has no fitted calibration"
                )
                continue
            if not np.isclose(record["m_mean"], fit.get("m_applied", np.nan), rtol=1e-6, atol=1e-7):
                errors.append(f"depth {record['depth']}: applied m disagrees with fit")
            if not np.isclose(record["b_mean"], fit.get("b_applied", np.nan), rtol=1e-6, atol=1e-7):
                errors.append(f"depth {record['depth']}: applied b disagrees with fit")
    return {
        "valid": not errors,
        "errors": errors,
        "calibration_sha256": sha256(path),
        "calibration_fit_signature_sha256": canonical_sha256(fit_signature),
        "calibration_signature_sha256": canonical_sha256(post_signature),
        "n_layers": len(layers),
        "converged": bool(payload.get("converged")),
        "reverted": any(bool(layer.get("calibration_reverted")) for layer in layers.values()),
        "init_gate_stats_sha256": sha256(init_path) if init_path.is_file() else "",
        "post_calibration_gate_stats_sha256": sha256(post_path) if post_path.is_file() else "",
    }


def convergence_record(result_dir: Path, resolved: dict, condition: str) -> dict:
    path = result_dir / "training_summary.json"
    errors: list[str] = []
    flags: list[str] = []
    if not path.is_file():
        return {
            "valid": False,
            "errors": ["missing training_summary.json"],
            "right_censored": True,
            "flags": ["missing trajectory"],
        }
    payload = json.loads(path.read_text())
    try:
        train = np.asarray(payload["train_losses"], dtype=float)
        valid = np.asarray(payload["valid_losses"], dtype=float)
        best_epoch = int(payload["best_epoch"])
        best_loss = float(payload["best_loss"])
    except (KeyError, TypeError, ValueError) as error:
        return {
            "valid": False,
            "errors": [f"invalid training summary: {error}"],
            "right_censored": True,
            "flags": ["invalid trajectory"],
        }
    max_epochs = int(nested_get(resolved, "training.main.common.epochs"))
    patience = int(nested_get(resolved, "training.main.common.patience"))
    if len(train) != len(valid) or not 1 <= len(valid) <= max_epochs:
        errors.append(
            f"trajectory lengths train={len(train)}, valid={len(valid)}, max={max_epochs}"
        )
    if not (np.isfinite(train).all() and np.isfinite(valid).all()):
        errors.append("trajectory contains non-finite losses")
    if not np.isfinite(best_loss):
        errors.append("best loss is non-finite")
    if len(valid):
        minimum = float(valid.min())
        minimum_epoch = int(valid.argmin()) + 1
        tolerance = max(1e-6, 1e-5 * abs(minimum))
        if not math.isclose(best_loss, minimum, rel_tol=0, abs_tol=tolerance):
            errors.append(f"best loss {best_loss:.9g} != validation minimum {minimum:.9g}")
        if best_epoch != minimum_epoch:
            errors.append(f"best epoch {best_epoch} != minimum epoch {minimum_epoch}")
    if not 1 <= best_epoch <= max(len(valid), 1):
        errors.append(f"best epoch {best_epoch} outside recorded trajectory")

    reached_limit = len(valid) == max_epochs
    final_slope = float("nan")
    if len(valid) >= 10 and np.isfinite(valid[-10:]).all():
        final_slope = float(np.polyfit(np.arange(10, dtype=float), valid[-10:], 1)[0])
    best_in_final_five = bool(len(valid) and best_epoch > len(valid) - 5)
    right_censored = bool(
        reached_limit
        and (best_in_final_five or (np.isfinite(final_slope) and final_slope < 0.0))
    )
    if right_censored:
        flags.append(
            f"{condition}: reached epoch limit with best_in_final_five="
            f"{best_in_final_five}, final10_slope={final_slope:.6g}"
        )
    if len(valid) < max_epochs and len(valid) - best_epoch < patience - 1:
        errors.append(
            "early-stopped trajectory is shorter than the configured patience after best epoch"
        )
    return {
        "valid": not errors,
        "errors": errors,
        "flags": flags,
        "right_censored": right_censored,
        "n_epochs_recorded": len(valid),
        "best_epoch": best_epoch,
        "best_validation_loss": best_loss,
        "final10_validation_slope": final_slope,
        "training_summary_sha256": sha256(path),
    }


def pairing_projection(resolved: dict) -> dict:
    return {
        "experiment": resolved.get("experiment"),
        "data": resolved.get("data"),
        "model": resolved.get("model"),
        "training_common": nested_get(resolved, "training.main.common"),
        "optimizer": nested_get(resolved, "training.main.optimizer"),
        "regularization": nested_get(resolved, "training.main.regularization"),
    }


def active_optimizer_semantics(resolved: dict) -> dict:
    """Resolve the optimizer fields that are active when parameters are unsplit."""
    split = nested_get(resolved, "training.main.common.param_groups.split_params")
    active_lr = nested_get(resolved, "training.main.common.param_groups.lr")
    sparse_weight_maintenance_rate = nested_get(
        resolved, "training.main.common.weight_decay_rate"
    )
    adam_weight_decay = nested_get(resolved, "training.main.optimizer.weight_decay")
    name = nested_get(resolved, "training.main.optimizer.name")
    return {
        "valid": bool(
            split is False
            and values_equal(active_lr, 0.001)
            and values_equal(sparse_weight_maintenance_rate, 0.01)
            and values_equal(adam_weight_decay, 0.0)
            and name == "adam"
        ),
        "optimizer": None if name is _MISSING else name,
        "split_params": None if split is _MISSING else split,
        "active_group_count": 1 if split is False else None,
        "active_learning_rate": None if active_lr is _MISSING else active_lr,
        "sparse_active_weight_maintenance_rate": (
            None
            if sparse_weight_maintenance_rate is _MISSING
            else sparse_weight_maintenance_rate
        ),
        "adam_weight_decay": (
            None if adam_weight_decay is _MISSING else adam_weight_decay
        ),
        "inactive_stored_group_learning_rates": {
            "topk": nested_get(
                resolved, "training.main.common.param_groups.topk_lr"
            ),
            "blocklinear": nested_get(
                resolved, "training.main.common.param_groups.blocklinear_lr"
            ),
            "reactivation": nested_get(
                resolved, "training.main.common.param_groups.reactivation_lr"
            ),
            "decoder": nested_get(
                resolved, "training.main.common.param_groups.decoder_lr"
            ),
        },
    }


def audit_resolved_config(
    resolved: dict,
    condition: str,
    metadata: dict,
    result_dir: Path,
) -> list[str]:
    errors = field_errors(resolved, EXPECTED_RESOLVED_FIELDS)
    explicit_i_flag = nested_get(
        resolved,
        "model.core.transfer.input_mode1_build_inhibitory_population",
    )
    if explicit_i_flag not in (_MISSING, False):
        errors.append(
            "input_mode=1 must retain the legacy direct signed-input "
            "architecture without an explicit inhibitory-cell population"
        )
    optimizer = active_optimizer_semantics(resolved)
    if not optimizer["valid"]:
        errors.append(f"active optimizer semantics are invalid: {optimizer}")
    strategy = nested_get(resolved, "training.main.strategy")
    if strategy != metadata["strategy"]:
        errors.append(
            f"training.main.strategy: expected {metadata['strategy']!r}, observed {strategy!r}"
        )
    local = nested_get(resolved, "training.main.learning_strategy_config")
    if metadata["strategy"] == "standard":
        if local is _MISSING:
            errors.append(
                "matched BP is missing training.main.learning_strategy_config; "
                "the resolved schema must record it explicitly as null"
            )
        elif local is not None:
            errors.append("matched BP unexpectedly has a local-learning configuration")
    else:
        errors.extend(field_errors(resolved, LOCAL_EXPECTED_FIELDS))
        mode = nested_get(
            resolved, "training.main.learning_strategy_config.error_broadcast_mode"
        )
        if mode != metadata["broadcast"]:
            errors.append(
                f"feedback mode: expected {metadata['broadcast']!r}, observed {mode!r}"
            )
    output_dir = nested_get(resolved, "outputs.results_dir")
    if output_dir is _MISSING or Path(str(output_dir)) != result_dir:
        errors.append(f"resolved output path {output_dir!r} != {result_dir}")
    seed = int(nested_get(resolved, "experiment.seed"))
    for name in (
        "dataset_seed",
        "split_seed",
        "model_seed",
        "topology_seed",
        "loader_seed",
        "evaluation_seed",
        "probe_seed",
    ):
        observed = nested_get(resolved, f"experiment.{name}")
        if observed != seed:
            errors.append(f"experiment.{name}: expected paired seed {seed}, observed {observed!r}")
    return [f"{condition}: {error}" for error in errors]


def collect(sweep_root: Path) -> tuple[pd.DataFrame, dict]:
    rows: list[dict] = []
    seen_configs: set[tuple[str, int]] = set()
    completed: set[tuple[str, int]] = set()
    unexpected: list[str] = []
    invalid: list[str] = []
    convergence_flags: list[str] = []
    projections: dict[int, dict[str, str]] = {seed: {} for seed in EXPECTED_SEEDS}
    calibrations: dict[int, dict[str, str]] = {seed: {} for seed in EXPECTED_SEEDS}

    config_paths = sorted((sweep_root / "configs").glob("*.yaml"))
    for config_path in config_paths:
        if config_path.name in {"metadata.yaml", "sweep_metadata.yaml"}:
            continue
        generated = OmegaConf.load(config_path)
        try:
            condition, metadata = condition_from_config(generated)
            seed = int(generated.experiment.seed)
            index = config_index(generated)
        except (ValueError, TypeError, AttributeError) as error:
            unexpected.append(f"{config_path.name}: {error}")
            continue
        key = (condition, seed)
        if key in seen_configs:
            invalid.append(f"duplicate generated config {condition}/seed-{seed}")
            continue
        seen_configs.add(key)
        if seed not in EXPECTED_SEEDS:
            unexpected.append(f"{condition}/seed-{seed}")
            continue

        result_dir = sweep_root / "results" / f"config_{index}"
        final_path = result_dir / "performance" / "final.json"
        resolved_path = result_dir / "config.json"
        checkpoint = result_dir / "main_network" / (
            "standard_best_model.pt"
            if metadata["strategy"] == "standard"
            else "local_learning_best_model.pt"
        )
        if not final_path.is_file() or not resolved_path.is_file():
            continue
        completed.add(key)
        if not checkpoint.is_file():
            invalid.append(f"{condition}/seed-{seed}: missing validation-best checkpoint")
        if not (result_dir / "final_model.pt").is_file():
            invalid.append(f"{condition}/seed-{seed}: missing restored final model")
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
        resolved = json.loads(resolved_path.read_text())
        invalid.extend(audit_resolved_config(resolved, condition, metadata, result_dir))
        projection_hash = canonical_sha256(pairing_projection(resolved))
        projections[seed][condition] = projection_hash

        calibration = calibration_record(result_dir)
        if not calibration["valid"]:
            invalid.extend(
                f"{condition}/seed-{seed}: {error}" for error in calibration["errors"]
            )
        calibrations[seed][condition] = calibration.get(
            "calibration_signature_sha256", ""
        )
        convergence = convergence_record(result_dir, resolved, condition)
        if not convergence["valid"]:
            invalid.extend(
                f"{condition}/seed-{seed}: {error}" for error in convergence["errors"]
            )
        convergence_flags.extend(
            f"seed-{seed}: {flag}" for flag in convergence.get("flags", [])
        )
        optimizer = active_optimizer_semantics(resolved)
        rows.append(
            {
                "condition": condition,
                "feedback": metadata["label"],
                "seed": seed,
                "config_index": index,
                "validation_accuracy": validation_accuracy,
                "test_accuracy": test_accuracy,
                "n_epochs_recorded": convergence.get("n_epochs_recorded"),
                "best_epoch": convergence.get("best_epoch"),
                "best_validation_loss": convergence.get("best_validation_loss"),
                "final10_validation_slope": convergence.get(
                    "final10_validation_slope"
                ),
                "right_censored": convergence.get("right_censored", True),
                "active_optimizer_group_count": optimizer["active_group_count"],
                "active_learning_rate": optimizer["active_learning_rate"],
                "sparse_active_weight_maintenance_rate": optimizer[
                    "sparse_active_weight_maintenance_rate"
                ],
                "adam_weight_decay": optimizer["adam_weight_decay"],
                "calibration_converged": calibration.get("converged", False),
                "calibration_reverted": calibration.get("reverted", True),
                "pairing_projection_sha256": projection_hash,
                "calibration_signature_sha256": calibration.get(
                    "calibration_signature_sha256", ""
                ),
                "generated_config_sha256": sha256(config_path),
                "resolved_config_sha256": sha256(resolved_path),
                "result_sha256": sha256(final_path),
                "checkpoint_sha256": sha256(checkpoint) if checkpoint.is_file() else "",
                "training_summary_sha256": convergence.get(
                    "training_summary_sha256", ""
                ),
                "calibration_sha256": calibration.get("calibration_sha256", ""),
            }
        )

    expected = {(condition, seed) for condition in CONDITIONS for seed in EXPECTED_SEEDS}
    missing_configs = sorted(expected - seen_configs)
    missing_results = sorted(expected - completed)
    for seed in EXPECTED_SEEDS:
        seed_projections = projections[seed]
        if len(seed_projections) == len(CONDITIONS) and len(set(seed_projections.values())) != 1:
            invalid.append(
                f"seed-{seed}: model/data/common-training projections differ across conditions"
            )
        seed_calibrations = calibrations[seed]
        if len(seed_calibrations) == len(CONDITIONS) and len(set(seed_calibrations.values())) != 1:
            invalid.append(f"seed-{seed}: empirical calibration differs across conditions")

    provenance = execution_identity(sweep_root)
    if not provenance.get("valid", False):
        invalid.extend(f"provenance: {error}" for error in provenance.get("errors", []))
    selection = selected_screen_evidence()
    if not selection.get("valid", False):
        invalid.extend(
            f"selection evidence: {error}" for error in selection.get("errors", [])
        )
    integrity_valid = not (missing_configs or missing_results or unexpected or invalid)
    if not integrity_valid:
        status = "incomplete_or_invalid"
    elif convergence_flags:
        status = "complete_with_convergence_flags"
    else:
        status = "complete_and_validated"
    audit = {
        "status": status,
        "integrity_valid": integrity_valid,
        "convergence_valid": not convergence_flags,
        "n_expected": len(expected),
        "n_configs_seen": len(seen_configs),
        "n_results_complete": len(completed),
        "missing_configs": [f"{condition}/seed-{seed}" for condition, seed in missing_configs],
        "missing_results": [f"{condition}/seed-{seed}" for condition, seed in missing_results],
        "unexpected": unexpected,
        "invalid": invalid,
        "convergence_flags": convergence_flags,
        "execution_identity": provenance,
        "selection_evidence": selection,
    }
    frame = pd.DataFrame(rows)
    if len(frame):
        frame = frame.sort_values(["seed", "feedback"]).reset_index(drop=True)
    return frame, audit


def t_interval(values: np.ndarray, confidence: float = 0.95) -> dict:
    values = np.asarray(values, dtype=float)
    n = len(values)
    mean = float(values.mean())
    sd = float(values.std(ddof=1))
    sem = sd / math.sqrt(n)
    critical = float(stats.t.ppf((1.0 + confidence) / 2.0, df=n - 1))
    half_width = critical * sem
    return {
        "n": n,
        "mean": mean,
        "sd": sd,
        "ci_low": mean - half_width,
        "ci_high": mean + half_width,
        "confidence": confidence,
    }


def one_sample_t_p(values: np.ndarray, alternative: str) -> float:
    values = np.asarray(values, dtype=float)
    if np.allclose(values, values[0], rtol=0, atol=1e-15):
        mean = float(values[0])
        if alternative == "greater":
            return 0.0 if mean > 0 else 1.0
        if alternative == "less":
            return 0.0 if mean < 0 else 1.0
        return 0.0 if mean != 0 else 1.0
    return float(stats.ttest_1samp(values, 0.0, alternative=alternative).pvalue)


def exact_sign_flip_p(values: np.ndarray, alternative: str = "greater") -> float:
    """Exact paired randomization P value over all sign assignments."""
    values = np.asarray(values, dtype=float)
    n = len(values)
    if n > 24:
        raise ValueError("exact sign-flip enumeration is limited to n <= 24")
    observed = float(values.sum())
    count = 0
    total = 1 << n
    bit_positions = np.arange(n, dtype=np.uint64)
    tolerance = 1e-14 * max(1.0, abs(observed))
    for start in range(0, total, 16_384):
        stop = min(start + 16_384, total)
        masks = np.arange(start, stop, dtype=np.uint64)[:, None]
        signs = 1.0 - 2.0 * ((masks >> bit_positions) & 1).astype(float)
        randomized = signs @ values
        if alternative == "greater":
            count += int(np.count_nonzero(randomized >= observed - tolerance))
        elif alternative == "less":
            count += int(np.count_nonzero(randomized <= observed + tolerance))
        elif alternative == "two-sided":
            count += int(
                np.count_nonzero(np.abs(randomized) >= abs(observed) - tolerance)
            )
        else:
            raise ValueError(f"unknown alternative {alternative!r}")
    return count / total


def holm_adjust(p_values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(p_values.items(), key=lambda item: (item[1], item[0]))
    adjusted: dict[str, float] = {}
    running = 0.0
    m = len(ordered)
    for rank, (name, p_value) in enumerate(ordered):
        running = max(running, (m - rank) * float(p_value))
        adjusted[name] = min(1.0, running)
    return adjusted


def tost_equivalence(values: np.ndarray, margin: float = EQUIVALENCE_MARGIN) -> dict:
    values = np.asarray(values, dtype=float)
    interval = t_interval(values, confidence=0.90)
    n = len(values)
    mean = float(values.mean())
    sd = float(values.std(ddof=1))
    sem = sd / math.sqrt(n)
    if sem == 0.0:
        p_lower = 0.0 if mean > -margin else 1.0
        p_upper = 0.0 if mean < margin else 1.0
    else:
        p_lower = float(stats.t.sf((mean + margin) / sem, df=n - 1))
        p_upper = float(stats.t.cdf((mean - margin) / sem, df=n - 1))
    return {
        "margin": margin,
        "p_lower": p_lower,
        "p_upper": p_upper,
        "p_tost": max(p_lower, p_upper),
        "ci90_low": interval["ci_low"],
        "ci90_high": interval["ci_high"],
        "equivalent": bool(
            p_lower < ALPHA
            and p_upper < ALPHA
            and interval["ci_low"] > -margin
            and interval["ci_high"] < margin
        ),
    }


def summarize(frame: pd.DataFrame, audit: dict) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    wide = frame.pivot(index="seed", columns="feedback", values="test_accuracy")
    wide = wide.loc[list(EXPECTED_SEEDS), list(CONDITION_ORDER)]
    condition_rows: list[dict] = []
    for feedback in CONDITION_ORDER:
        interval = t_interval(wide[feedback].to_numpy(float))
        condition_rows.append(
            {
                "feedback": feedback,
                "n_seeds": interval["n"],
                "mean_test_accuracy": interval["mean"],
                "sd_test_accuracy": interval["sd"],
                "ci95_low_test_accuracy": interval["ci_low"],
                "ci95_high_test_accuracy": interval["ci_high"],
            }
        )

    contrast_data: dict[str, np.ndarray] = {}
    for name, left, right in (*SUPERIORITY_CONTRASTS, EQUIVALENCE_CONTRAST):
        contrast_data[name] = (wide[left] - wide[right]).to_numpy(float)
    directional_p = {
        name: one_sample_t_p(contrast_data[name], alternative="greater")
        for name, _, _ in SUPERIORITY_CONTRASTS
    }
    holm_p = holm_adjust(directional_p)

    contrast_rows: list[dict] = []
    for name, left, right in (*SUPERIORITY_CONTRASTS, EQUIVALENCE_CONTRAST):
        differences = contrast_data[name]
        interval = t_interval(differences)
        is_superiority = name in directional_p
        contrast_rows.append(
            {
                "contrast": name,
                "left": left,
                "right": right,
                "n_seeds": len(differences),
                "mean_difference": interval["mean"],
                "sd_difference": interval["sd"],
                "ci95_low_difference": interval["ci_low"],
                "ci95_high_difference": interval["ci_high"],
                "seeds_positive": int((differences > 0).sum()),
                "paired_t_p_greater": directional_p.get(name, np.nan),
                "paired_t_p_two_sided": one_sample_t_p(
                    differences, alternative="two-sided"
                ),
                "holm_p_greater": holm_p.get(name, np.nan),
                "exact_sign_flip_p_greater": exact_sign_flip_p(
                    differences, alternative="greater"
                ),
                "exact_sign_flip_p_two_sided": exact_sign_flip_p(
                    differences, alternative="two-sided"
                ),
                "confirmatory_superiority_family": is_superiority,
                "seed_differences": ";".join(f"{value:.9f}" for value in differences),
            }
        )

    primary_name = "exact path minus neuron specific"
    control_name = "neuron specific minus strict scalar"
    primary = next(row for row in contrast_rows if row["contrast"] == primary_name)
    control = next(row for row in contrast_rows if row["contrast"] == control_name)
    equivalence_values = contrast_data[EQUIVALENCE_CONTRAST[0]]
    equivalence = tost_equivalence(equivalence_values)
    control_passes = bool(
        control["ci95_low_difference"] > 0
        and control["holm_p_greater"] < ALPHA
        and control["exact_sign_flip_p_greater"] < ALPHA
    )
    primary_passes = bool(
        control_passes
        and primary["mean_difference"] >= 0.01
        and primary["ci95_low_difference"] > 0
        and primary["seeds_positive"] >= 16
        and primary["holm_p_greater"] < ALPHA
        and primary["exact_sign_flip_p_greater"] < ALPHA
    )
    audit_passes = bool(
        audit.get("integrity_valid") and audit.get("convergence_valid")
    )
    bp_mean = float(wide["backpropagation"].mean())
    bp_adequate = bool(bp_mean >= BP_ADEQUACY_THRESHOLD)
    bandwidth_promotion = bool(audit_passes and bp_adequate and control_passes)
    path_resolution_promotion = bool(audit_passes and bp_adequate and primary_passes)
    bp_recovery_claim = bool(path_resolution_promotion and equivalence["equivalent"])
    decision = {
        "main_promotion": path_resolution_promotion,
        "bandwidth_cross_dataset_promotion": bandwidth_promotion,
        "path_resolution_main_promotion": path_resolution_promotion,
        "exact_path_recovers_bp_claim": bp_recovery_claim,
        "audit_passes": audit_passes,
        "matched_bp_mean_test_accuracy": bp_mean,
        "matched_bp_adequacy_threshold": BP_ADEQUACY_THRESHOLD,
        "matched_bp_adequacy_passes": bp_adequate,
        "neuron_minus_scalar_positive_control_passes": control_passes,
        "exact_minus_neuron_hierarchical_superiority_passes": primary_passes,
        "exact_vs_bp_equivalence": equivalence,
        "promotion_gate": {
            "primary_mean_at_least_1pp": bool(primary["mean_difference"] >= 0.01),
            "primary_ci95_above_zero": bool(primary["ci95_low_difference"] > 0),
            "primary_positive_at_least_16_of_20": bool(
                primary["seeds_positive"] >= 16
            ),
            "superiority_holm_and_hierarchy": bool(
                control_passes and primary["holm_p_greater"] < ALPHA
            ),
            "sign_flip_sensitivity": bool(
                primary["exact_sign_flip_p_greater"] < ALPHA
            ),
            "formal_equivalence_with_bp_within_1pp": equivalence["equivalent"],
        },
        "if_gate_fails": (
            "Use only the claims whose separate audit, BP-adequacy, bandwidth, "
            "path-resolution, and BP-equivalence gates pass; otherwise retain "
            "the cohort as a compact Supplementary/talk-backup boundary result."
        ),
    }
    return pd.DataFrame(condition_rows), pd.DataFrame(contrast_rows), decision


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("sweep_root", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()

    frame, audit = collect(args.sweep_root)
    if not audit["integrity_valid"] and not args.allow_incomplete:
        raise RuntimeError(json.dumps(audit, indent=2))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_dir / "seed_outcomes.csv", index=False)
    record = {
        "audit": audit,
        "sweep_root": str(args.sweep_root),
        "contract_frozen_before_confirmatory_outcomes": True,
    }
    if audit["integrity_valid"]:
        conditions, contrasts, decision = summarize(frame, audit)
        conditions.to_csv(args.output_dir / "condition_summary.csv", index=False)
        contrasts.to_csv(args.output_dir / "paired_contrasts.csv", index=False)
        record["decision"] = decision
    (args.output_dir / "summary.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
