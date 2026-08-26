#!/usr/bin/env python3
"""Audit the current-source matched MNIST ladder and freeze publication tables.

Scalar, neuron-specific and exact-path feedback were re-executed from one
source commit with matched architecture-specific specifications. Publication
tables are written only when all 90 architecture-by-feedback-by-seed outcomes
and checkpoints are present and valid. The earlier 60-run coordinate cohort is
loaded only to preserve an explicit comparison audit; it is not plotted.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


JOURNAL = Path(__file__).resolve().parents[1]
OLD_SOURCE = JOURNAL / "source_data" / "figure2" / "feedback_accuracy_runs.csv"
OUTPUT = JOURNAL / "source_data" / "mnist_feedback_ladder"
PROJECT_RUN_ROOT = Path(
    "/n/holylfs06/LABS/kempner_project_b/Lab/dendritic/HS/LOCAL_LEARNING/"
    "journal_extension_20260820/sweep_runs/mnist_feedback_ladder"
)
EXPECTED_SEEDS = tuple(range(42, 57))
CORE_LABEL = {
    "dendritic_shunting": "shunting",
    "dendritic_additive": "additive",
}
MODE_LABEL = {
    "per_soma": "scalar broadcast",
    "per_soma_shared": "neuron specific",
    "path_transport": "exact path",
}
OLD_FEEDBACK_LABEL = {
    "scalar_fallback": "scalar broadcast",
    "ancestry_shared": "neuron specific",
}
ORDER = ("scalar broadcast", "neuron specific", "exact path")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def config_index(path: Path) -> int:
    return int(path.stem.rsplit("_", 1)[1])


def bootstrap_mean(
    values: np.ndarray, seed: int, draws: int = 50_000
) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    sampled = rng.choice(values, size=(draws, len(values)), replace=True).mean(axis=1)
    low, high = np.quantile(sampled, [0.025, 0.975])
    return float(values.mean()), float(low), float(high)


def exact_sign_flip_p(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    observed = abs(float(values.mean()))
    null = np.asarray(
        [
            np.mean(values * np.asarray(signs, dtype=float))
            for signs in itertools.product((-1.0, 1.0), repeat=len(values))
        ]
    )
    return float(np.mean(np.abs(null) >= observed - 1e-15))


def load_existing() -> pd.DataFrame:
    frame = pd.read_csv(OLD_SOURCE)
    required = {
        "feedback",
        "run_dir",
        "seed",
        "network_type",
        "test_accuracy",
        "config_sha256",
        "result_sha256",
        "checkpoint_sha256",
    }
    missing_columns = sorted(required - set(frame.columns))
    if missing_columns:
        raise RuntimeError(f"existing source lacks columns: {missing_columns}")
    if len(frame) != 60:
        raise RuntimeError(f"expected 60 existing rows, found {len(frame)}")
    if set(frame.feedback) != set(OLD_FEEDBACK_LABEL):
        raise RuntimeError("existing source has an unexpected feedback design")
    if set(frame.network_type) != set(CORE_LABEL):
        raise RuntimeError("existing source has an unexpected architecture design")
    if set(map(int, frame.seed)) != set(EXPECTED_SEEDS):
        raise RuntimeError("existing source has an unexpected seed design")
    if frame.duplicated(["network_type", "feedback", "seed"]).any():
        raise RuntimeError("existing source contains duplicate design cells")
    accuracy = frame.test_accuracy.to_numpy(float)
    if not np.isfinite(accuracy).all() or not np.logical_and(accuracy >= 0, accuracy <= 1).all():
        raise RuntimeError("existing source contains invalid accuracies")
    return pd.DataFrame(
        {
            "architecture": frame.network_type.map(CORE_LABEL),
            "core": frame.network_type,
            "seed": frame.seed.astype(int),
            "feedback": frame.feedback.map(OLD_FEEDBACK_LABEL),
            "broadcast_mode": frame.broadcast_mode,
            "test_accuracy": frame.test_accuracy.astype(float),
            "run_dir": frame.run_dir,
            "config_index": pd.NA,
            "config_sha256": frame.config_sha256,
            "result_sha256": frame.result_sha256,
            "checkpoint_sha256": frame.checkpoint_sha256,
            "cohort": "validated_2026-07-31_coordinate_cohort",
        }
    )


def collect_run(
    run: Path,
    expected_core: str,
    expected_modes: tuple[str, ...],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not run.is_dir():
        raise FileNotFoundError(run)
    manifest_path = run / "frozen_sweep_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_count = len(EXPECTED_SEEDS) * len(expected_modes)
    if int(manifest.get("expected_config_count", -1)) != expected_count:
        raise RuntimeError(
            f"{run}: frozen manifest does not declare {expected_count} configs"
        )
    scheduler = manifest.get("scheduler_profile", {})
    if scheduler.get("account") != "kempner_bsabatini_lab":
        raise RuntimeError(f"{run}: unexpected scheduler account")
    if scheduler.get("partition") != "kempner_h100_priority":
        raise RuntimeError(f"{run}: unexpected scheduler partition")

    config_paths = sorted(
        (run / "configs").glob("unified_config_*.yaml"), key=config_index
    )
    if len(config_paths) != expected_count:
        raise RuntimeError(
            f"{run}: expected {expected_count} resolved configs, found {len(config_paths)}"
        )

    rows: list[dict[str, Any]] = []
    signatures: list[str] = []
    errors: list[str] = []
    for config_path in config_paths:
        index = config_index(config_path)
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        seed = int(config["experiment"]["seed"])
        core = str(config["model"]["core"]["type"])
        local = config["training"]["main"]["learning_strategy_config"]
        mode = str(local["error_broadcast_mode"])
        rule = str(local["rule_variant"])
        decoder_mode = str(local["decoder_update_mode"])
        init_policy = str(config["model"]["core"]["reactivation"]["init_policy"])
        expected_policy = (
            "analytical" if expected_core == "dendritic_shunting" else "occupancy_quantile"
        )
        checks = {
            "seed": seed in EXPECTED_SEEDS,
            "core": core == expected_core,
            "dataset": config["data"]["dataset_name"] == "mnist",
            "branches": config["model"]["core"]["architecture"]["excitatory_branch_factors"]
            == [3, 3],
            "mode": mode in expected_modes,
            "rule": rule.lower() == "3f",
            "decoder": decoder_mode == "local",
            "init_policy": init_policy == expected_policy,
            "epochs": int(config["training"]["main"]["common"]["epochs"]) == 180,
            "wandb": config["wandb"]["use_wandb"] is False,
        }
        failed = [name for name, passed in checks.items() if not passed]
        if failed:
            errors.append(f"{config_path.name}: failed {', '.join(failed)}")

        signature = json.loads(json.dumps(config))
        signature["_sweep_config_id"] = "<sweep_config_id>"
        signature["experiment"]["seed"] = "<seed>"
        signature["outputs"]["results_dir"] = "<results_dir>"
        signature["outputs"]["run_name"] = "<run_name>"
        signature["training"]["main"]["learning_strategy_config"][
            "error_broadcast_mode"
        ] = "<feedback_mode>"
        signatures.append(json.dumps(signature, sort_keys=True))

        result_dir = run / "results" / f"config_{index}"
        final_path = result_dir / "performance" / "final.json"
        checkpoint = result_dir / "final_model.pt"
        if not final_path.is_file() or not checkpoint.is_file():
            errors.append(f"{config_path.name}: missing final metric or checkpoint")
            continue
        final = json.loads(final_path.read_text(encoding="utf-8"))
        accuracy = float(final["accuracy"]["test"])
        if not np.isfinite(accuracy) or not 0 <= accuracy <= 1:
            errors.append(f"{config_path.name}: invalid accuracy {accuracy}")
            continue
        rows.append(
            {
                "architecture": CORE_LABEL[core],
                "core": core,
                "seed": seed,
                "feedback": MODE_LABEL[mode],
                "broadcast_mode": mode,
                "test_accuracy": accuracy,
                "run_dir": str(run),
                "config_index": index,
                "config_sha256": sha256(config_path),
                "result_sha256": sha256(final_path),
                "checkpoint_sha256": sha256(checkpoint),
                "cohort": "matched_current_source_ladder_2026-08-25",
            }
        )

    # After replacing seed and output location, every resolved scientific
    # configuration within one architecture must be identical.
    if len(set(signatures)) != 1:
        errors.append("resolved configurations differ beyond seed/output location")
    if errors:
        raise RuntimeError(f"invalid or incomplete ladder run {run}:\n- " + "\n- ".join(errors))
    frame = pd.DataFrame(rows)
    if len(frame) != expected_count or set(frame.seed) != set(EXPECTED_SEEDS):
        raise RuntimeError(f"{run}: incomplete seed design after collection")
    if set(frame.broadcast_mode) != set(expected_modes):
        raise RuntimeError(f"{run}: incomplete feedback-mode design after collection")
    return frame, {
        "run_dir": str(run),
        "manifest_sha256": sha256(manifest_path),
        "scientific_signature_sha256": hashlib.sha256(
            signatures[0].encode("utf-8")
        ).hexdigest(),
        "source_identity": manifest.get("source_identity", {}),
        "scheduler_profile": scheduler,
    }


def summarize(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows: list[dict[str, Any]] = []
    contrast_rows: list[dict[str, Any]] = []
    contrasts = (
        ("neuron specific", "scalar broadcast"),
        ("exact path", "neuron specific"),
        ("exact path", "scalar broadcast"),
    )
    for architecture_index, architecture in enumerate(("shunting", "additive")):
        wide = (
            frame[frame.architecture.eq(architecture)]
            .pivot(index="seed", columns="feedback", values="test_accuracy")
            .loc[list(EXPECTED_SEEDS), list(ORDER)]
        )
        for feedback_index, feedback in enumerate(ORDER):
            values = wide[feedback].to_numpy(float)
            mean, low, high = bootstrap_mean(
                values, 250_000 + 10 * architecture_index + feedback_index
            )
            summary_rows.append(
                {
                    "architecture": architecture,
                    "feedback": feedback,
                    "n_seeds": len(values),
                    "mean_accuracy": mean,
                    "sd_accuracy": float(values.std(ddof=1)),
                    "ci95_low": low,
                    "ci95_high": high,
                }
            )
        for contrast_index, (high_name, low_name) in enumerate(contrasts):
            values = (wide[high_name] - wide[low_name]).to_numpy(float)
            mean, low, high = bootstrap_mean(
                values, 251_000 + 10 * architecture_index + contrast_index
            )
            contrast_rows.append(
                {
                    "architecture": architecture,
                    "contrast": f"{high_name} - {low_name}",
                    "n_seeds": len(values),
                    "mean_difference": mean,
                    "ci95_low": low,
                    "ci95_high": high,
                    "positive_seeds": int(np.sum(values > 0)),
                    "negative_seeds": int(np.sum(values < 0)),
                    "tied_seeds": int(np.sum(values == 0)),
                    "exact_two_sided_sign_flip_p": exact_sign_flip_p(values),
                }
            )
    return pd.DataFrame(summary_rows), pd.DataFrame(contrast_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--coordinate-shunting",
        type=Path,
        default=None,
        help="scalar/neuron-specific shunting run directory",
    )
    parser.add_argument(
        "--coordinate-additive",
        type=Path,
        default=None,
        help="scalar/neuron-specific additive run directory",
    )
    parser.add_argument(
        "--shunting",
        type=Path,
        default=None,
        help="exact-path shunting run directory (default: unique matching run)",
    )
    parser.add_argument(
        "--additive",
        type=Path,
        default=None,
        help="exact-path additive run directory (default: unique matching run)",
    )
    args = parser.parse_args()

    def resolve(explicit: Path | None, pattern: str) -> Path:
        if explicit is not None:
            return explicit.resolve()
        matches = sorted(PROJECT_RUN_ROOT.glob(pattern))
        if len(matches) != 1:
            raise RuntimeError(f"expected one run matching {pattern!r}, found {len(matches)}")
        return matches[0]

    coordinate_shunting_run = resolve(
        args.coordinate_shunting,
        "journal_mnist_feedback_ladder_coordinates_shunting_15seed_*",
    )
    coordinate_additive_run = resolve(
        args.coordinate_additive,
        "journal_mnist_feedback_ladder_coordinates_additive_15seed_*",
    )
    shunting_run = resolve(
        args.shunting, "journal_mnist_feedback_ladder_exact_path_shunting_15seed_*"
    )
    additive_run = resolve(
        args.additive, "journal_mnist_feedback_ladder_exact_path_additive_15seed_*"
    )
    historical = load_existing()
    coordinate_modes = ("per_soma", "per_soma_shared")
    coordinate_shunting, coordinate_shunting_record = collect_run(
        coordinate_shunting_run, "dendritic_shunting", coordinate_modes
    )
    coordinate_additive, coordinate_additive_record = collect_run(
        coordinate_additive_run, "dendritic_additive", coordinate_modes
    )
    shunting, shunting_record = collect_run(
        shunting_run, "dendritic_shunting", ("path_transport",)
    )
    additive, additive_record = collect_run(
        additive_run, "dendritic_additive", ("path_transport",)
    )
    for architecture, coordinate_record, exact_record in (
        ("shunting", coordinate_shunting_record, shunting_record),
        ("additive", coordinate_additive_record, additive_record),
    ):
        if (
            coordinate_record["scientific_signature_sha256"]
            != exact_record["scientific_signature_sha256"]
        ):
            raise RuntimeError(
                f"{architecture}: coordinate and exact-path scientific signatures differ"
            )
    run_records = [
        coordinate_shunting_record,
        coordinate_additive_record,
        shunting_record,
        additive_record,
    ]
    source_versions = {
        (
            record["source_identity"].get("git", {}).get("commit"),
            record["source_identity"].get("git", {}).get("tracked_diff_sha256"),
            record["source_identity"].get("python_version"),
        )
        for record in run_records
    }
    if len(source_versions) != 1:
        raise RuntimeError("the four ladder arrays do not share one source environment")
    frame = pd.concat(
        [coordinate_shunting, coordinate_additive, shunting, additive],
        ignore_index=True,
    )
    if len(frame) != 90 or frame.duplicated(["architecture", "feedback", "seed"]).any():
        raise RuntimeError("combined ladder is not a complete 90-cell paired design")
    summary, contrasts = summarize(frame)
    comparison = frame[frame.feedback.ne("exact path")].merge(
        historical[["architecture", "seed", "feedback", "test_accuracy"]],
        on=["architecture", "seed", "feedback"],
        how="inner",
        validate="one_to_one",
        suffixes=("_current", "_historical"),
    )
    if len(comparison) != 60:
        raise RuntimeError("current-to-historical coordinate comparison is incomplete")
    comparison["accuracy_difference"] = (
        comparison.test_accuracy_current - comparison.test_accuracy_historical
    )
    historical_concordance = []
    for (architecture, feedback), part in comparison.groupby(
        ["architecture", "feedback"], sort=True
    ):
        differences = part.accuracy_difference.to_numpy(float)
        historical_concordance.append(
            {
                "architecture": architecture,
                "feedback": feedback,
                "n_paired_seeds": len(differences),
                "mean_current_minus_historical": float(differences.mean()),
                "maximum_absolute_difference": float(np.max(np.abs(differences))),
            }
        )

    OUTPUT.mkdir(parents=True, exist_ok=True)
    frame.sort_values(["architecture", "seed", "feedback"]).to_csv(
        OUTPUT / "seed_outcomes.csv", index=False
    )
    summary.to_csv(OUTPUT / "condition_summary.csv", index=False)
    contrasts.to_csv(OUTPUT / "paired_contrasts.csv", index=False)
    report = {
        "status": "complete_and_validated",
        "n_historical_rows_used_for_comparison_only": len(historical),
        "n_current_coordinate_rows": len(coordinate_shunting) + len(coordinate_additive),
        "n_current_exact_path_rows": len(shunting) + len(additive),
        "n_combined_rows": len(frame),
        "seeds": list(EXPECTED_SEEDS),
        "existing_source": {
            "path": str(OLD_SOURCE),
            "sha256": sha256(OLD_SOURCE),
        },
        "historical_coordinate_concordance": historical_concordance,
        "source_environment": list(source_versions)[0],
        "current_source_runs": run_records,
        "scheduler_overrides": {
            "coordinate_shunting_tasks_6_29": {
                "job_id": "41843834",
                "partition": "kempner_requeue",
                "qos": "normal",
            },
            "coordinate_additive_tasks_0_5": {
                "job_id": "41842076",
                "partition": "kempner_eng",
                "qos": "normal",
            },
            "coordinate_additive_tasks_6_29": {
                "job_id": "41843847",
                "partition": "kempner_requeue",
                "qos": "normal",
            },
        },
    }
    (OUTPUT / "audit.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    print("\nPaired contrasts (percentage points):")
    printable = contrasts.copy()
    for column in ("mean_difference", "ci95_low", "ci95_high"):
        printable[column] *= 100
    print(printable.to_string(index=False))


if __name__ == "__main__":
    main()
