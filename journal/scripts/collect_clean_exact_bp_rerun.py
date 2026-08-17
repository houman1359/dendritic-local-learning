#!/usr/bin/env python3
"""Audit and summarize the clean exact-transport versus backpropagation rerun."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.stats import wilcoxon


JOURNAL = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = JOURNAL.parents[2]
ROOT = JOURNAL / "analysis" / "clean_exact_bp"
RESULTS = ROOT / "results"
OUT = JOURNAL / "source_data" / "clean_exact_bp"

COHORTS = {
    "bp_mnist_additive": "clean_bp_mnist_additive_20260810124345",
    "bp_mnist_shunting": "clean_bp_mnist_shunting_20260810124408",
    "bp_noise_additive": "clean_bp_noise_additive_20260810124424",
    "bp_noise_shunting": "clean_bp_noise_shunting_20260810135859",
    "exact_mnist_additive": "clean_exact_mnist_additive_20260810124504",
    "exact_mnist_shunting": "clean_exact_mnist_shunting_20260810124612",
    "exact_noise_additive": "clean_exact_noise_additive_20260810124630",
    "exact_noise_shunting": "clean_exact_noise_shunting_20260810135823",
}
EXPECTED_ROOT_COMMIT = "74792ca91724066148f36c758191744a315d62b3"
EXPECTED_EXPERIMENTS_COMMIT = "4f3612a52756c3691e5bd269c6465c02f240c1e8"
CRITICAL_PATTERNS = {
    "traceback": re.compile(r"Traceback \(most recent call last\):"),
    "nonfinite": re.compile(r"\b(?:nan|NaN|inf|Inf)\b"),
    "oom": re.compile(r"CUDA out of memory|OutOfMemoryError"),
    "transport_shape_fallback": re.compile(
        r"(?:Path transport|Path propagation) shape mismatch|"
        r"Falling back to scalar averaging"
    ),
}
RECOVERABLE_STREAMING_MARKER = "retrying with streaming evaluation"
RECOVERABLE_STREAMING_LINES = (
    "Error computing metric in _compute_",
    "retrying with streaming evaluation",
    "Streaming evaluation for ",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def bootstrap_ci(values: np.ndarray, seed: int, n_boot: int = 20_000) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), size=(n_boot, len(values)))].mean(axis=1)
    return tuple(float(x) for x in np.quantile(means, [0.025, 0.975]))


def latest_scheduler_text(generated: Path, index: int) -> str:
    attempts: dict[int, list[Path]] = defaultdict(list)
    pattern = re.compile(rf"^(?:output|error)_(\d+)_{index}\.(?:out|err)$")
    roots = [generated]
    manifest_path = generated / "frozen_sweep_manifest.json"
    if manifest_path.is_file():
        manifest = load_json(manifest_path)
        repository_name = manifest.get("source_identity", {}).get("repository_name")
        if repository_name:
            clean_root = REPOSITORY_ROOT.parent / str(repository_name)
            try:
                mirror = clean_root / generated.relative_to(REPOSITORY_ROOT)
            except ValueError:
                mirror = None
            if mirror is not None and mirror != generated:
                roots.append(mirror)
    for root in roots:
        for path in (root / "jobs").glob(f"*_*_{index}.*"):
            match = pattern.match(path.name)
            if match:
                attempts[int(match.group(1))].append(path)
    if not attempts:
        return ""
    return "\n".join(
        path.read_text(encoding="utf-8", errors="replace")
        for path in attempts[max(attempts)]
    )


def checkpoint_issues(path: Path, expected_depth: int) -> tuple[list[str], int]:
    issues: list[str] = []
    state = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(state, dict):
        return ["checkpoint_not_state_dict"], 0
    tensor_count = 0
    coupling_depths: set[int] = set()
    excitation_depths: set[int] = set()
    for name, value in state.items():
        if not isinstance(value, torch.Tensor):
            continue
        tensor_count += 1
        if value.is_floating_point() and not torch.isfinite(value).all():
            issues.append(f"nonfinite_checkpoint:{name}")
        match = re.search(r"branch_layers\.(\d+)\.branches_to_output\.log_weight$", name)
        if match:
            coupling_depths.add(int(match.group(1)))
        match = re.search(r"branch_layers\.(\d+)\.branch_excitation\.pre_w$", name)
        if match:
            excitation_depths.add(int(match.group(1)))
    if len(coupling_depths) < expected_depth:
        issues.append(f"missing_coupling_stages:{len(coupling_depths)}/{expected_depth}")
    if len(excitation_depths) < expected_depth:
        issues.append(f"missing_excitatory_stages:{len(excitation_depths)}/{expected_depth}")
    return issues, tensor_count


def audit_manifest(name: str, generated: Path) -> list[str]:
    issues: list[str] = []
    path = generated / "frozen_sweep_manifest.json"
    if not path.is_file():
        return [f"{name}:missing_manifest"]
    manifest = load_json(path)
    if int(manifest.get("expected_config_count", -1)) != 40:
        issues.append(f"{name}:manifest_expected_count")
    git_identity = manifest.get("source_identity", {}).get("git", {})
    if git_identity.get("commit") != EXPECTED_ROOT_COMMIT:
        issues.append(f"{name}:source_commit")
    if git_identity.get("tracked_worktree_dirty") is not False:
        issues.append(f"{name}:tracked_worktree_dirty")
    return issues


def audit_clean_checkouts() -> list[str]:
    """Verify the detached training source and nested dataset repository."""

    first_generated = ROOT / "generated" / next(iter(COHORTS.values()))
    manifest = load_json(first_generated / "frozen_sweep_manifest.json")
    repository_name = manifest.get("source_identity", {}).get("repository_name")
    if not repository_name:
        return ["clean_checkout:missing_repository_name"]
    clean_root = REPOSITORY_ROOT.parent / str(repository_name)

    def git(*args: str, cwd: Path) -> str:
        return subprocess.check_output(
            ["git", "-C", str(cwd), *args], text=True
        ).strip()

    issues: list[str] = []
    if git("rev-parse", "HEAD", cwd=clean_root) != EXPECTED_ROOT_COMMIT:
        issues.append("clean_checkout:root_commit")
    if git("status", "--porcelain=v1", "--untracked-files=no", cwd=clean_root):
        issues.append("clean_checkout:root_tracked_dirty")
    experiments = clean_root / "experiments"
    if git("rev-parse", "HEAD", cwd=experiments) != EXPECTED_EXPERIMENTS_COMMIT:
        issues.append("clean_checkout:experiments_commit")
    if git("status", "--porcelain=v1", "--untracked-files=no", cwd=experiments):
        issues.append("clean_checkout:experiments_tracked_dirty")
    return issues


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"No rows for {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    issues: list[str] = audit_clean_checkouts()
    rows: list[dict[str, Any]] = []
    for cohort, generated_name in COHORTS.items():
        generated = ROOT / "generated" / generated_name
        issues.extend(audit_manifest(cohort, generated))
        method = "exact_transport" if cohort.startswith("exact_") else "backpropagation"
        expected_dataset = "mnist" if "_mnist_" in cohort else "noise_resilience"
        expected_core = "additive" if cohort.endswith("_additive") else "shunting"
        for index in range(40):
            run = RESULTS / cohort / f"config_{index}"
            required = [
                run / "config.json",
                run / "resolved_seeds.json",
                run / "training_summary.json",
                run / "performance" / "final.json",
                run / "final_model.pt",
                run / "train.log",
                run / "dendritic_modeling.log",
            ]
            missing = [path.name for path in required if not path.is_file()]
            if missing:
                issues.append(f"{cohort}/config_{index}:missing:{','.join(missing)}")
                continue
            config = load_json(run / "config.json")
            final = load_json(run / "performance" / "final.json")
            seed = int(config["experiment"]["seed"])
            dataset = str(config["data"]["dataset_name"])
            core_type = str(config["model"]["core"]["type"])
            core = "shunting" if bool(config["model"]["core"]["morphology"]["use_shunting"]) else "additive"
            depth = len(config["model"]["core"]["architecture"]["excitatory_branch_factors"])
            strategy = str(config["training"]["main"]["strategy"])
            feedback = str(
                config["training"]["main"].get("learning_strategy_config", {}).get(
                    "error_broadcast_mode", "backpropagation"
                )
            )
            if dataset != expected_dataset or core != expected_core:
                issues.append(f"{cohort}/config_{index}:condition_mismatch")
            transfer_activation = config["model"]["core"]["transfer"].get(
                "output_activation"
            )
            if expected_dataset == "noise_resilience" and expected_core == "shunting":
                if transfer_activation != "relu":
                    issues.append(f"{cohort}/config_{index}:signed_shunting_input")
            if method == "exact_transport" and (strategy != "local_ca" or feedback != "path_transport"):
                issues.append(f"{cohort}/config_{index}:not_exact_transport")
            if method == "backpropagation" and strategy != "standard":
                issues.append(f"{cohort}/config_{index}:not_backpropagation")
            accuracy = float(final["accuracy"]["test"])
            log_likelihood = float(final["categorical_loglikelihood"]["test"])
            if not (math.isfinite(accuracy) and math.isfinite(log_likelihood)):
                issues.append(f"{cohort}/config_{index}:nonfinite_final")
            log_text = "\n".join(
                [
                    (run / "train.log").read_text(encoding="utf-8", errors="replace"),
                    (run / "dendritic_modeling.log").read_text(encoding="utf-8", errors="replace"),
                    latest_scheduler_text(generated, index),
                ]
            )
            used_streaming_evaluation = RECOVERABLE_STREAMING_MARKER in log_text
            audit_log_text = log_text
            if used_streaming_evaluation:
                # Full-dataset metric materialization can exceed device memory;
                # runtime.py then recomputes the same metric batchwise.  Remove
                # only that explicitly paired warning/retry sequence.  Any
                # training OOM, scheduler OOM or unpaired non-finite event
                # remains visible to the critical-pattern audit below.
                audit_log_text = "\n".join(
                    line
                    for line in log_text.splitlines()
                    if not any(marker in line for marker in RECOVERABLE_STREAMING_LINES)
                )
            matches = [
                label
                for label, pattern in CRITICAL_PATTERNS.items()
                if pattern.search(audit_log_text)
            ]
            if matches:
                issues.append(f"{cohort}/config_{index}:log:{','.join(matches)}")
            checkpoint_findings, tensor_count = checkpoint_issues(run / "final_model.pt", depth)
            issues.extend(f"{cohort}/config_{index}:{item}" for item in checkpoint_findings)
            rows.append(
                {
                    "cohort": cohort,
                    "config_index": index,
                    "method": method,
                    "dataset": dataset,
                    "core": core,
                    "core_type": core_type,
                    "depth": depth,
                    "seed": seed,
                    "test_accuracy": accuracy,
                    "test_categorical_loglikelihood": log_likelihood,
                    "best_epoch": int(load_json(run / "training_summary.json")["best_epoch"]),
                    "checkpoint_tensor_count": tensor_count,
                    "checkpoint_bytes": (run / "final_model.pt").stat().st_size,
                    "checkpoint_sha256": sha256(run / "final_model.pt"),
                    "config_sha256": sha256(run / "config.json"),
                    "latest_log_critical_count": len(matches),
                    "used_streaming_evaluation": used_streaming_evaluation,
                }
            )
            if len(rows) % 20 == 0:
                print(
                    f"Audited {len(rows)}/320 checkpoints "
                    f"({cohort}, config {index})",
                    flush=True,
                )

    expected_keys = {
        (method, dataset, core, depth, seed)
        for method in ("exact_transport", "backpropagation")
        for dataset in ("mnist", "noise_resilience")
        for core in ("additive", "shunting")
        for depth in range(1, 5)
        for seed in range(42, 52)
    }
    actual_keys = {
        (row["method"], row["dataset"], row["core"], row["depth"], row["seed"])
        for row in rows
    }
    if actual_keys != expected_keys:
        issues.append(
            f"factorial_key_mismatch:missing={len(expected_keys - actual_keys)}:extra={len(actual_keys - expected_keys)}"
        )
    if len(rows) != 320:
        issues.append(f"row_count:{len(rows)}/320")
    if issues:
        raise RuntimeError("Clean exact/BP audit failed:\n" + "\n".join(issues[:100]))

    OUT.mkdir(parents=True, exist_ok=True)
    rows.sort(key=lambda row: (row["dataset"], row["core"], row["depth"], row["seed"], row["method"]))
    write_csv(OUT / "run_outcomes.csv", rows)

    indexed = {
        (row["method"], row["dataset"], row["core"], row["depth"], row["seed"]): row
        for row in rows
    }
    paired: list[dict[str, Any]] = []
    for dataset in ("mnist", "noise_resilience"):
        for core in ("additive", "shunting"):
            for depth in range(1, 5):
                for seed in range(42, 52):
                    exact = indexed[("exact_transport", dataset, core, depth, seed)]
                    bp = indexed[("backpropagation", dataset, core, depth, seed)]
                    paired.append(
                        {
                            "dataset": dataset,
                            "core": core,
                            "depth": depth,
                            "seed": seed,
                            "exact_accuracy": exact["test_accuracy"],
                            "backpropagation_accuracy": bp["test_accuracy"],
                            "exact_minus_backpropagation_accuracy": exact["test_accuracy"] - bp["test_accuracy"],
                            "absolute_accuracy_difference": abs(exact["test_accuracy"] - bp["test_accuracy"]),
                        }
                    )
    write_csv(OUT / "paired_differences.csv", paired)

    condition_rows: list[dict[str, Any]] = []
    for key in sorted({(r["method"], r["dataset"], r["core"], r["depth"]) for r in rows}):
        method, dataset, core, depth = key
        values = np.array(
            [r["test_accuracy"] for r in rows if (r["method"], r["dataset"], r["core"], r["depth"]) == key]
        )
        low, high = bootstrap_ci(values, seed=9000 + depth + (0 if method == "exact_transport" else 100))
        condition_rows.append(
            {
                "method": method,
                "dataset": dataset,
                "core": core,
                "depth": depth,
                "n_seeds": len(values),
                "mean_test_accuracy": float(values.mean()),
                "sample_sd_test_accuracy": float(values.std(ddof=1)),
                "ci95_low": low,
                "ci95_high": high,
            }
        )
    write_csv(OUT / "condition_summary.csv", condition_rows)

    contrast_rows: list[dict[str, Any]] = []
    for dataset in ("mnist", "noise_resilience"):
        for core in ("additive", "shunting"):
            for depth_label, selected_depths in [(str(d), [d]) for d in range(1, 5)] + [("all_depths_seed_mean", [1, 2, 3, 4])]:
                seed_values = []
                exact_seed_values = []
                bp_seed_values = []
                for seed in range(42, 52):
                    selected = [
                        row
                        for row in paired
                        if row["dataset"] == dataset
                        and row["core"] == core
                        and row["seed"] == seed
                        and row["depth"] in selected_depths
                    ]
                    exact_mean = float(np.mean([row["exact_accuracy"] for row in selected]))
                    bp_mean = float(np.mean([row["backpropagation_accuracy"] for row in selected]))
                    exact_seed_values.append(exact_mean)
                    bp_seed_values.append(bp_mean)
                    seed_values.append(exact_mean - bp_mean)
                array = np.array(seed_values)
                low, high = bootstrap_ci(array, seed=12000 + len(contrast_rows))
                try:
                    p_value = float(wilcoxon(array, alternative="two-sided", method="auto").pvalue)
                except ValueError:
                    p_value = 1.0
                contrast_rows.append(
                    {
                        "dataset": dataset,
                        "core": core,
                        "depth": depth_label,
                        "n_paired_seeds": len(array),
                        "mean_exact_accuracy": float(np.mean(exact_seed_values)),
                        "mean_backpropagation_accuracy": float(np.mean(bp_seed_values)),
                        "mean_exact_minus_backpropagation_accuracy": float(array.mean()),
                        "sample_sd_exact_minus_backpropagation_accuracy": float(array.std(ddof=1)),
                        "ci95_low": low,
                        "ci95_high": high,
                        "positive_seeds": int((array > 0).sum()),
                        "negative_seeds": int((array < 0).sum()),
                        "ties": int((array == 0).sum()),
                        "wilcoxon_two_sided_p": p_value,
                        "maximum_absolute_seed_difference": float(np.max(np.abs(array))),
                    }
                )
    write_csv(OUT / "exact_bp_contrasts.csv", contrast_rows)

    all_differences = np.array([row["exact_minus_backpropagation_accuracy"] for row in paired])
    summary = {
        "study": "clean_exact_transport_backpropagation_rerun",
        "status": "complete_and_audited",
        "root_source_commit": EXPECTED_ROOT_COMMIT,
        "experiments_dataset_source_commit": EXPECTED_EXPERIMENTS_COMMIT,
        "n_runs": len(rows),
        "n_paired_conditions": len(paired),
        "seeds": list(range(42, 52)),
        "depths": [1, 2, 3, 4],
        "datasets": ["mnist", "noise_resilience"],
        "cores": ["additive", "shunting"],
        "all_required_artifacts_present": True,
        "all_checkpoints_finite_and_stage_complete": True,
        "all_final_metrics_finite": True,
        "critical_log_matches": 0,
        "runs_using_audited_streaming_evaluation": int(
            sum(bool(row["used_streaming_evaluation"]) for row in rows)
        ),
        "transport_shape_fallbacks": 0,
        "mean_exact_minus_backpropagation_accuracy_across_160_pairs": float(all_differences.mean()),
        "maximum_absolute_run_pair_accuracy_difference": float(np.max(np.abs(all_differences))),
        "scope_boundary": "Exact path transport is an information-oracle implementation check on the same artificial architectures, tasks, seeds and training budget as backpropagation; it is not a claim that biological circuits compute the oracle.",
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Clean exact-transport versus backpropagation rerun",
        "",
        f"Status: **{summary['status']}**",
        "",
        f"- {len(rows)} complete runs and {len(paired)} exact/BP pairs.",
        "- Ten paired seeds, four depths, two tasks and two forward cores.",
        "- No missing artifacts, nonfinite checkpoint tensors, critical log matches or transport-shape fallbacks.",
        f"- {summary['runs_using_audited_streaming_evaluation']} runs used the logged batchwise metric fallback after full-dataset materialization exceeded device memory; all accepted final metrics were finite.",
        f"- Mean exact-minus-BP accuracy across all run pairs: {all_differences.mean():.8f}.",
        f"- Maximum absolute paired accuracy difference: {np.max(np.abs(all_differences)):.8f}.",
        "",
        "Primary seed-level contrasts (depth averaged):",
        "",
    ]
    for row in contrast_rows:
        if row["depth"] != "all_depths_seed_mean":
            continue
        lines.append(
            f"- {row['dataset']} / {row['core']}: exact {row['mean_exact_accuracy']:.6f}, "
            f"BP {row['mean_backpropagation_accuracy']:.6f}, difference "
            f"{row['mean_exact_minus_backpropagation_accuracy']:.6f} "
            f"[{row['ci95_low']:.6f}, {row['ci95_high']:.6f}], "
            f"{row['positive_seeds']}/10 positive, P={row['wilcoxon_two_sided_p']:.6g}."
        )
    (ROOT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
