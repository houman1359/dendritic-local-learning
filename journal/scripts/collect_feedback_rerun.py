#!/usr/bin/env python3
"""Validate and summarize the clean 15-seed feedback rerun.

The script is deliberately non-destructive.  It requires the complete 60-run
cohort, checks the frozen seed-by-feedback design and expected architecture
policies, and writes a staged table and audit summary.  Publication source
data are replaced only after the staged output has been inspected.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = ROOT.parents[2]
DEFAULT_SHUNTING = ROOT / "reproduction_runs" / "journal_feedback_definition_shunting_3f_mnist_15seed_20260731150354"
DEFAULT_ADDITIVE = ROOT / "reproduction_runs" / "journal_feedback_definition_additive_3f_mnist_15seed_20260731150354"
DEFAULT_OUTDIR = ROOT / "analysis" / "feedback_rerun_validation"
EXPECTED_SEEDS = tuple(range(42, 57))
MODE_TO_LABEL = {"per_soma": "scalar_fallback", "per_soma_shared": "ancestry_shared"}
EXPECTED_POLICY = {"dendritic_shunting": "analytical", "dendritic_additive": "occupancy_quantile"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def nested(config: dict[str, Any], *keys: str) -> Any:
    value: Any = config
    for key in keys:
        value = value[key]
    return value


def scientific_signature(config: dict[str, Any]) -> dict[str, Any]:
    """Fields that must be constant within an architecture after design axes."""

    return {
        "dataset": nested(config, "data", "dataset_name"),
        "processing": nested(config, "data", "processing"),
        "core_type": nested(config, "model", "core", "type"),
        "architecture": nested(config, "model", "core", "architecture"),
        "connectivity": nested(config, "model", "core", "connectivity"),
        "transfer": nested(config, "model", "core", "transfer"),
        "morphology": nested(config, "model", "core", "morphology"),
        "reactivation": nested(config, "model", "core", "reactivation"),
        "decoder": nested(config, "model", "decoder"),
        "training_common": nested(config, "training", "main", "common"),
        "local_rule": {
            key: value
            for key, value in nested(config, "training", "main", "learning_strategy_config").items()
            if key != "error_broadcast_mode"
        },
    }


def load_architecture(run_dir: Path, expected_core: str) -> tuple[list[dict[str, Any]], list[str]]:
    config_paths = sorted(
        (run_dir / "configs").glob("unified_config_*.yaml"),
        key=lambda path: int(path.stem.rsplit("_", 1)[1]),
    )
    errors: list[str] = []
    if len(config_paths) != 30:
        errors.append(f"{run_dir.name}: expected 30 resolved configurations, found {len(config_paths)}")

    rows: list[dict[str, Any]] = []
    signatures: list[dict[str, Any]] = []
    for config_path in config_paths:
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        config_id = config_path.stem.replace("unified_", "")
        seed = int(nested(config, "experiment", "seed"))
        core = str(nested(config, "model", "core", "type"))
        mode = str(nested(config, "training", "main", "learning_strategy_config", "error_broadcast_mode"))
        rule = str(nested(config, "training", "main", "learning_strategy_config", "rule_variant"))
        decoder_mode = str(nested(config, "training", "main", "learning_strategy_config", "decoder_update_mode"))
        policy = str(nested(config, "model", "core", "reactivation", "init_policy"))
        result_dir = REPOSITORY / str(nested(config, "outputs", "results_dir"))
        # Resolved paths are repository-relative; ROOT is the journal project.
        if not result_dir.exists():
            result_dir = run_dir / "results" / config_id
        performance = result_dir / "performance" / "final.json"
        checkpoint = result_dir / "final_model.pt"
        if core != expected_core:
            errors.append(f"{config_path.name}: core {core!r}, expected {expected_core!r}")
        if mode not in MODE_TO_LABEL:
            errors.append(f"{config_path.name}: unexpected feedback mode {mode!r}")
        if rule.lower() != "3f":
            errors.append(f"{config_path.name}: rule {rule!r}, expected '3f'")
        if decoder_mode != "local":
            errors.append(f"{config_path.name}: decoder mode {decoder_mode!r}, expected 'local'")
        expected_policy = EXPECTED_POLICY.get(expected_core)
        if policy != expected_policy:
            errors.append(f"{config_path.name}: init policy {policy!r}, expected {expected_policy!r}")
        if seed not in EXPECTED_SEEDS:
            errors.append(f"{config_path.name}: seed {seed} outside frozen 42--56 cohort")
        if not performance.exists():
            errors.append(f"{config_path.name}: missing {performance}")
            continue
        if not checkpoint.exists():
            errors.append(f"{config_path.name}: missing {checkpoint}")
            continue
        result = json.loads(performance.read_text(encoding="utf-8"))
        accuracy = float(result["accuracy"]["test"])
        if not np.isfinite(accuracy) or not 0.0 <= accuracy <= 1.0:
            errors.append(f"{config_path.name}: invalid test accuracy {accuracy}")
            continue
        rows.append(
            {
                "feedback": MODE_TO_LABEL[mode],
                "run_dir": str(result_dir.relative_to(ROOT)),
                "seed": seed,
                "network_type": core,
                "rule_variant": rule.lower(),
                "broadcast_mode": mode,
                "decoder_update_mode": decoder_mode,
                "init_policy": policy,
                "test_accuracy": accuracy,
                "config_sha256": sha256(config_path),
                "result_sha256": sha256(performance),
                "checkpoint_sha256": sha256(checkpoint),
            }
        )
        signatures.append(scientific_signature(config))

    if signatures:
        reference = json.dumps(signatures[0], sort_keys=True)
        for index, signature in enumerate(signatures[1:], start=1):
            if json.dumps(signature, sort_keys=True) != reference:
                errors.append(
                    f"{run_dir.name}: scientific configuration differs at resolved configuration {index} "
                    "after removing seed, feedback mode, and output path"
                )
    return rows, errors


def paired_summary(frame: pd.DataFrame, network_type: str) -> dict[str, Any]:
    part = frame.loc[frame["network_type"].eq(network_type)]
    wide = part.pivot(index="seed", columns="feedback", values="test_accuracy").sort_index()
    expected_columns = {"scalar_fallback", "ancestry_shared"}
    if set(wide.columns) != expected_columns or tuple(wide.index) != EXPECTED_SEEDS:
        raise ValueError(f"{network_type}: incomplete paired design")
    difference = (wide["ancestry_shared"] - wide["scalar_fallback"]).to_numpy(dtype=float)
    test = stats.wilcoxon(difference, alternative="two-sided", method="auto")
    return {
        "network_type": network_type,
        "n_paired_seeds": int(len(wide)),
        "scalar_fallback_mean": float(wide["scalar_fallback"].mean()),
        "scalar_fallback_sd": float(wide["scalar_fallback"].std(ddof=1)),
        "ancestry_shared_mean": float(wide["ancestry_shared"].mean()),
        "ancestry_shared_sd": float(wide["ancestry_shared"].std(ddof=1)),
        "mean_paired_improvement": float(difference.mean()),
        "sd_paired_improvement": float(difference.std(ddof=1)),
        "pairs_improved": int((difference > 0).sum()),
        "pairs_tied": int((difference == 0).sum()),
        "wilcoxon_two_sided_p": float(test.pvalue),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shunting", type=Path, default=DEFAULT_SHUNTING)
    parser.add_argument("--additive", type=Path, default=DEFAULT_ADDITIVE)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    args = parser.parse_args()

    shunting_rows, shunting_errors = load_architecture(args.shunting, "dendritic_shunting")
    additive_rows, additive_errors = load_architecture(args.additive, "dendritic_additive")
    errors = shunting_errors + additive_errors
    frame = pd.DataFrame(shunting_rows + additive_rows)
    if errors:
        raise RuntimeError("Clean feedback rerun is not complete or invalid:\n- " + "\n- ".join(errors))
    if len(frame) != 60 or frame.duplicated(["network_type", "feedback", "seed"]).any():
        raise RuntimeError("Expected exactly 60 unique architecture-by-feedback-by-seed results")

    frame = frame.sort_values(["network_type", "feedback", "seed"]).reset_index(drop=True)
    summaries = [paired_summary(frame, network) for network in sorted(frame["network_type"].unique())]
    old_path = ROOT / "source_data" / "figure2" / "feedback_accuracy_runs.csv"
    old = pd.read_csv(old_path)
    old_summary = [paired_summary(old, network) for network in sorted(old["network_type"].unique())]
    payload = {
        "status": "complete_and_validated",
        "n_rows": int(len(frame)),
        "expected_seeds": list(EXPECTED_SEEDS),
        "new_cohort": summaries,
        "previous_mixed_archive_cohort": old_summary,
        "source_runs": {
            "shunting": str(args.shunting.relative_to(ROOT)),
            "additive": str(args.additive.relative_to(ROOT)),
        },
        "replacement_policy": (
            "The clean current-code cohort replaces the mixed-provenance table regardless of whether "
            "its effect is larger or smaller. Publication source data are not overwritten by this script."
        ),
    }
    args.outdir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.outdir / "feedback_accuracy_runs_clean.csv", index=False)
    (args.outdir / "validation_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
