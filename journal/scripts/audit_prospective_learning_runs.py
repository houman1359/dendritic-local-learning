#!/usr/bin/env python3
"""Audit and collect prospective local-learning depth experiments."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf


JOURNAL_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = JOURNAL_ROOT.parents[2]
RUN_ROOT = JOURNAL_ROOT / "prospective_runs"
TRANSPORT_SOURCE = (
    WORKSPACE_ROOT / "src/dendritic_modeling/training/strategies/local_learning_parts/"
    "local_learning_broadcast_transport_mixin.py"
)
CRITICAL_LOG_PATTERNS = {
    "traceback": re.compile(r"Traceback \(most recent call last\):"),
    "nan": re.compile(r"\b(?:nan|NaN)\b"),
    "oom": re.compile(r"CUDA out of memory|OutOfMemoryError"),
    "transport_shape_fallback": re.compile(
        r"(?:Path transport|Path propagation) shape mismatch|"
        r"Falling back to scalar averaging"
    ),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> Any:
    with path.open() as handle:
        return json.load(handle)


def _source_identity_digest(identity: dict[str, Any]) -> str:
    payload = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _current_source_mismatches(identity: dict[str, Any]) -> list[str]:
    mismatches: list[str] = []
    for record in identity.get("files", []):
        recorded_path = str(record["path"])
        if recorded_path.startswith("repo:"):
            path = WORKSPACE_ROOT / recorded_path.removeprefix("repo:")
        else:
            path = Path(recorded_path)
        if not path.exists():
            mismatches.append(f"missing:{recorded_path}")
        elif _sha256(path) != str(record["sha256"]):
            mismatches.append(f"hash:{recorded_path}")
    return mismatches


def _verified_source_equivalence_mismatches(
    identity: dict[str, Any], report_path: Path
) -> list[str]:
    """Return mismatches not covered by a still-current equivalence audit."""
    if not report_path.exists():
        return [f"missing_equivalence_report:{report_path}"]
    report = _load_json(report_path)
    if report.get("status") != "verified":
        return ["equivalence_report_not_verified"]
    identity_sha = _source_identity_digest(identity)
    accepted = set(report.get("accepted_frozen_identity_sha256", []))
    if identity_sha not in accepted:
        return [f"identity_not_in_equivalence_report:{identity_sha}"]

    exceptions = report.get("current_source_exceptions", {})
    mismatches: list[str] = []
    for record in identity.get("files", []):
        recorded_path = str(record["path"])
        if recorded_path.startswith("repo:"):
            path = WORKSPACE_ROOT / recorded_path.removeprefix("repo:")
        else:
            path = Path(recorded_path)
        if not path.exists():
            mismatches.append(f"missing:{recorded_path}")
            continue
        current_sha = _sha256(path)
        frozen_sha = str(record["sha256"])
        if current_sha == frozen_sha:
            continue
        exception = exceptions.get(recorded_path)
        if not isinstance(exception, dict):
            mismatches.append(f"unverified_hash:{recorded_path}")
            continue
        if (
            exception.get("frozen_sha256") != frozen_sha
            or exception.get("current_sha256") != current_sha
            or exception.get("normalized_sha256") != frozen_sha
        ):
            mismatches.append(f"stale_equivalence:{recorded_path}")
    return mismatches


def _latest_run_dirs(phase: str, study: str) -> list[Path]:
    by_prefix: dict[str, Path] = {}
    patterns = (
        [f"journal_{phase}_*_depth_feedback_*"]
        if study == "primary"
        else (
            [f"journal_{phase}_fixed_budget_depth_*"]
            if study == "fixed_budget"
            else [
                f"journal_{phase}_inhibition_dose_*",
                f"journal_{phase}_spatial_topology_*",
                f"journal_{phase}_ancestry_routing_*",
            ]
        )
    )
    candidates = [path for pattern in patterns for path in RUN_ROOT.glob(pattern)]
    for path in sorted(candidates):
        if not path.is_dir():
            continue
        prefix = re.sub(r"_\d{14}$", "", path.name)
        current = by_prefix.get(prefix)
        if current is None or path.name > current.name:
            by_prefix[prefix] = path
    return sorted(by_prefix.values())


def _scan_logs(run_dir: Path, index: int) -> tuple[list[str], float | None]:
    """Inspect the most recent scheduler attempt for one frozen config.

    A config can be resubmitted after an infrastructure-only failure.  Slurm's
    ``%A_%a`` filenames retain every attempt, so pooling all matching files
    would make a successful retry inherit an earlier node failure.  We group
    logs by array-job identifier and audit only the newest attempt.  The result
    log is still included because it records application-level diagnostics for
    the shared result directory.
    """
    attempts: dict[int, list[Path]] = {}
    pattern = re.compile(rf"^(?:output|error)_(\d+)_{index}\.(?:out|err)$")
    for path in list((run_dir / "jobs").glob(f"output_*_{index}.out")) + list(
        (run_dir / "jobs").glob(f"error_*_{index}.err")
    ):
        match = pattern.match(path.name)
        if match:
            attempts.setdefault(int(match.group(1)), []).append(path)

    texts: list[str] = []
    duration = None
    latest_paths = attempts[max(attempts)] if attempts else []
    for path in latest_paths:
        text = path.read_text(errors="replace")
        texts.append(text)
        match = re.search(r"Total duration:\s*(\d+)\s*seconds", text)
        if match:
            duration = float(match.group(1))
    result_log = run_dir / "results" / f"config_{index}" / "dendritic_modeling.log"
    if result_log.exists():
        texts.append(result_log.read_text(errors="replace"))
    combined = "\n".join(texts)
    return [
        name
        for name, pattern in CRITICAL_LOG_PATTERNS.items()
        if pattern.search(combined)
    ], duration


def _checkpoint_audit(path: Path, expected_depth: int) -> tuple[list[str], int]:
    issues: list[str] = []
    state = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(state, dict):
        return ["checkpoint_not_state_dict"], 0

    tensor_count = 0
    for name, value in state.items():
        if not isinstance(value, torch.Tensor):
            continue
        tensor_count += 1
        if value.is_floating_point() and not torch.isfinite(value).all():
            issues.append(f"nonfinite_checkpoint:{name}")

    coupling_depths = set()
    excitation_depths = set()
    for name, value in state.items():
        match = re.search(
            r"branch_layers\.(\d+)\.branches_to_output\.log_weight$", name
        )
        if match and isinstance(value, torch.Tensor):
            coupling_depths.add(int(match.group(1)))
            if value.numel() == 0 or float(value.detach().abs().median()) <= 1e-8:
                issues.append(f"degenerate_coupling:{name}")
        match = re.search(r"branch_layers\.(\d+)\.branch_excitation\.pre_w$", name)
        if match and isinstance(value, torch.Tensor):
            excitation_depths.add(int(match.group(1)))

    if len(coupling_depths) < expected_depth:
        issues.append(
            f"missing_coupling_stages:{len(coupling_depths)}/{expected_depth}"
        )
    if len(excitation_depths) < expected_depth:
        issues.append(
            f"missing_excitatory_stages:{len(excitation_depths)}/{expected_depth}"
        )
    return issues, tensor_count


def _config_metadata(config: Any) -> dict[str, Any]:
    branch_factors = list(config.model.core.architecture.excitatory_branch_factors)
    cumulative_width = 1
    non_somatic_compartments = 0
    for factor in branch_factors:
        cumulative_width *= int(factor)
        non_somatic_compartments += cumulative_width
    strategy = str(config.training.main.strategy)
    feedback = str(config.training.main.learning_strategy_config.error_broadcast_mode)
    inhibitory = list(config.model.core.connectivity.ie_synapses_per_branch_per_layer)
    structured = OmegaConf.select(config, "model.core.connectivity.structured.enabled")
    sparsity_type = OmegaConf.select(config, "model.core.sparsity.type")
    return {
        "task": str(config.data.dataset_name),
        "core": str(config.model.core.type),
        "strategy": strategy,
        "feedback": feedback if strategy == "local_ca" else "backprop",
        "depth": len(branch_factors),
        "branch_factors": "x".join(map(str, branch_factors)),
        "non_somatic_compartments_per_soma": non_somatic_compartments,
        "inhibitory_synapses_per_branch": int(inhibitory[0]),
        "topology": (
            "spatial"
            if structured is True
            else "random" if str(sparsity_type).lower() == "indexed" else "standard"
        ),
        "routing": (
            "correct"
            if feedback == "per_soma_shared"
            else "shuffled" if feedback == "per_soma_shuffled" else "not_applicable"
        ),
        "seed": int(config.experiment.seed),
        "init_policy": str(config.model.core.reactivation.init_policy),
    }


def audit_run(run_dir: Path) -> list[dict[str, Any]]:
    manifest_path = run_dir / "frozen_sweep_manifest.json"
    manifest = _load_json(manifest_path)
    rows: list[dict[str, Any]] = []
    for record in manifest["generated_configs"]:
        index = int(record["index"])
        config_path = run_dir / record["path"]
        config = OmegaConf.load(config_path)
        metadata = _config_metadata(config)
        issues: list[str] = []
        if _sha256(config_path) != record["sha256"]:
            issues.append("config_hash_mismatch")

        result_dir = run_dir / "results" / f"config_{index}"
        required = {
            "checkpoint": result_dir / "final_model.pt",
            "performance": result_dir / "performance" / "final.json",
            "training": result_dir / "training_summary.json",
            "resources": result_dir / "model_resources.json",
            "resolved_seeds": result_dir / "resolved_seeds.json",
            "result_config": result_dir / "config.json",
        }
        missing = [name for name, path in required.items() if not path.exists()]
        issues.extend(f"missing:{name}" for name in missing)

        log_issues, duration = _scan_logs(run_dir, index)
        issues.extend(log_issues)
        if duration is None:
            issues.append("missing:duration")
        row: dict[str, Any] = {
            "run_dir": str(run_dir.relative_to(JOURNAL_ROOT)),
            "config_index": index,
            **metadata,
            "duration_seconds": duration,
            "test_accuracy": None,
            "valid_accuracy": None,
            "best_epoch": None,
            "best_loss": None,
            "epochs_recorded": None,
            "total_parameters": None,
            "active_synapses": None,
            "cuda_peak_memory_allocated_bytes": None,
            "cuda_peak_memory_reserved_bytes": None,
            "checkpoint_bytes": None,
            "checkpoint_tensors": None,
        }

        if not missing:
            performance = _load_json(required["performance"])
            training = _load_json(required["training"])
            resources = _load_json(required["resources"])
            result_config = _load_json(required["result_config"])

            result_meta = _config_metadata(OmegaConf.create(result_config))
            if result_meta != metadata:
                issues.append("result_config_mismatch")

            test_accuracy = float(performance["accuracy"]["test"])
            valid_accuracy = float(performance["accuracy"]["valid"])
            train_losses = [float(value) for value in training["train_losses"]]
            valid_losses = [float(value) for value in training["valid_losses"]]
            numeric = [test_accuracy, valid_accuracy, *train_losses, *valid_losses]
            if not all(math.isfinite(value) for value in numeric):
                issues.append("nonfinite_metric")
            if not (0.0 <= test_accuracy <= 1.0 and 0.0 <= valid_accuracy <= 1.0):
                issues.append("invalid_accuracy")
            if not train_losses or min(train_losses) >= 0.99 * train_losses[0]:
                issues.append("no_training_loss_progress")
            expected_epochs = int(config.training.main.common.epochs)
            if (
                len(train_losses) != expected_epochs
                or len(valid_losses) != expected_epochs
            ):
                issues.append(
                    f"incomplete_history:{len(train_losses)}/{expected_epochs}"
                )

            checkpoint_issues, tensor_count = _checkpoint_audit(
                required["checkpoint"], metadata["depth"]
            )
            issues.extend(checkpoint_issues)
            row.update(
                {
                    "test_accuracy": test_accuracy,
                    "valid_accuracy": valid_accuracy,
                    "best_epoch": int(training["best_epoch"]),
                    "best_loss": float(training["best_loss"]),
                    "epochs_recorded": len(train_losses),
                    "total_parameters": int(resources["total_parameters"]),
                    "active_synapses": int(resources["active_synapses"]),
                    "cuda_peak_memory_allocated_bytes": resources.get(
                        "cuda_peak_memory_allocated_bytes"
                    ),
                    "cuda_peak_memory_reserved_bytes": resources.get(
                        "cuda_peak_memory_reserved_bytes"
                    ),
                    "checkpoint_bytes": required["checkpoint"].stat().st_size,
                    "checkpoint_tensors": tensor_count,
                }
            )

        row["status"] = (
            "pass"
            if not issues
            else (
                "incomplete"
                if all(issue.startswith("missing:") for issue in issues)
                else "fail"
            )
        )
        row["issues"] = ";".join(issues)
        rows.append(row)
    return rows


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_report(
    rows: list[dict[str, Any]],
    path: Path,
    phase: str,
    study: str,
    source_identity_sha: str,
) -> None:
    counts = {
        status: sum(row["status"] == status for row in rows)
        for status in ("pass", "incomplete", "fail")
    }
    lines = [
        f"# Prospective {study} {phase} audit",
        "",
        f"Transport source SHA-256: `{_sha256(TRANSPORT_SOURCE)}`",
        f"Frozen source-identity SHA-256: `{source_identity_sha}`",
        "",
        f"Runs: {len(rows)}; pass: {counts['pass']}; incomplete: {counts['incomplete']}; fail: {counts['fail']}.",
        "",
        "| Task | Core | Rule | Feedback | Topology | I/branch | Depth | Seed | Test accuracy | Status | Issues |",
        "|---|---|---|---|---|---:|---:|---:|---:|---|---|",
    ]
    for row in sorted(
        rows,
        key=lambda value: (
            value["task"],
            value["core"],
            value["strategy"],
            value["feedback"],
            value["depth"],
            value["seed"],
        ),
    ):
        accuracy = "" if row["test_accuracy"] is None else f"{row['test_accuracy']:.4f}"
        lines.append(
            f"| {row['task']} | {row['core']} | {row['strategy']} | {row['feedback']} | "
            f"{row['topology']} | {row['inhibitory_synapses_per_branch']} | "
            f"{row['depth']} | {row['seed']} | {accuracy} | {row['status']} | {row['issues']} |"
        )
    lines.extend(
        [
            "",
            "The canary cohort is a software/stability gate and is not scientific evidence.",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("canary", "confirmatory"), default="canary")
    parser.add_argument(
        "--study",
        choices=("primary", "followup", "fixed_budget"),
        default="primary",
    )
    parser.add_argument("--allow-running", action="store_true")
    parser.add_argument(
        "--require-current-source",
        action="store_true",
        help="fail if a frozen source file differs from the current worktree",
    )
    parser.add_argument(
        "--verified-source-equivalence",
        type=Path,
        help=(
            "accept only behavior-neutral source differences recorded by a "
            "still-current source-equivalence audit"
        ),
    )
    args = parser.parse_args()

    run_dirs = _latest_run_dirs(args.phase, args.study)
    expected_arrays = 8 if args.study == "primary" else 16
    if len(run_dirs) != expected_arrays:
        raise SystemExit(
            f"Expected {expected_arrays} latest {args.study} {args.phase} arrays, "
            f"found {len(run_dirs)}"
        )
    manifests = [
        _load_json(run_dir / "frozen_sweep_manifest.json") for run_dir in run_dirs
    ]
    source_identities = [manifest["source_identity"] for manifest in manifests]
    source_digests = {
        _source_identity_digest(identity) for identity in source_identities
    }
    if len(source_digests) != 1:
        raise SystemExit(
            f"Expected one frozen source identity across arrays, found {len(source_digests)}"
        )
    source_identity_sha = next(iter(source_digests))
    if args.require_current_source:
        if args.verified_source_equivalence is not None:
            mismatches = _verified_source_equivalence_mismatches(
                source_identities[0], args.verified_source_equivalence
            )
        else:
            mismatches = _current_source_mismatches(source_identities[0])
        if mismatches:
            raise SystemExit(
                "Current source differs from the frozen experiment identity: "
                + ", ".join(mismatches)
            )
    rows = [row for run_dir in run_dirs for row in audit_run(run_dir)]
    expected_by_study = {
        ("primary", "canary"): 32,
        ("primary", "confirmatory"): 640,
        ("followup", "canary"): 36,
        ("followup", "confirmatory"): 880,
        ("fixed_budget", "canary"): 24,
        ("fixed_budget", "confirmatory"): 320,
    }
    expected = expected_by_study[(args.study, args.phase)]
    if len(rows) != expected:
        raise SystemExit(f"Expected {expected} configs, found {len(rows)}")

    stem = f"prospective_{args.study}_{args.phase}_audit"
    _write_csv(rows, JOURNAL_ROOT / "analysis" / f"{stem}.csv")
    _write_report(
        rows,
        JOURNAL_ROOT / "analysis" / f"{stem}.md",
        args.phase,
        args.study,
        source_identity_sha,
    )
    failures = [row for row in rows if row["status"] == "fail"]
    incomplete = [row for row in rows if row["status"] == "incomplete"]
    print(
        f"{args.phase}: {len(rows)} runs, {len(failures)} failed, {len(incomplete)} incomplete"
    )
    if failures or (incomplete and not args.allow_running):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
