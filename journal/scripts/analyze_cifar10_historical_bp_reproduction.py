#!/usr/bin/env python3
"""Audit the pinned-source historical CIFAR-10 BP reproduction."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf


EXPECTED_SEEDS = tuple(range(42, 47))
ARCHIVED_MEANS = {"normalized additive": 0.48300, "shunting": 0.49516}
COMPATIBILITY_TOLERANCE = 0.02


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def config_index(config) -> int:
    return int(str(config._sweep_config_id).rsplit("_", 1)[-1])


def architecture_from_type(core_type: str) -> str:
    mapping = {
        "dendritic_additive": "normalized additive",
        "dendritic_shunting": "shunting",
    }
    if core_type not in mapping:
        raise ValueError(f"unexpected historical core type: {core_type}")
    return mapping[core_type]


def bootstrap_mean(values: np.ndarray, seed: int) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    generator = np.random.default_rng(seed)
    draws = generator.choice(values, size=(100_000, len(values)), replace=True).mean(axis=1)
    low, high = np.quantile(draws, [0.025, 0.975])
    return float(values.mean()), float(low), float(high)


def audit_resolved_config(result_dir: Path, architecture: str) -> tuple[dict, str]:
    path = result_dir / "config.json"
    if not path.is_file():
        raise ValueError("missing resolved config.json")
    payload = json.loads(path.read_text())
    core = payload["model"]["core"]
    morphology = core["morphology"]
    implementation = core["implementation"]
    reactivation = core["reactivation"]
    use_shunting = bool(morphology["use_shunting"])
    use_normalization = bool(morphology["use_additive_normalization"])
    adaptive = bool(implementation["adaptive_initialization"])
    if adaptive:
        raise ValueError("historical adaptive_initialization was not false")
    if architecture == "normalized additive":
        if use_shunting or not use_normalization:
            raise ValueError("historical additive alias did not resolve to normalization")
        if str(reactivation["init_policy"]) != "fixed":
            raise ValueError("historical additive gate did not resolve to fixed")
        if not np.isclose(float(reactivation["init_m"]), 0.1, atol=1e-12):
            raise ValueError("historical additive init_m did not resolve to 0.1")
        if not np.isclose(float(reactivation["init_b"]), 0.0, atol=1e-12):
            raise ValueError("historical additive init_b did not resolve to 0")
    elif not use_shunting:
        raise ValueError("historical shunting alias did not resolve to shunting")
    return {
        "resolved_use_shunting": use_shunting,
        "resolved_use_additive_normalization": use_normalization,
        "resolved_adaptive_initialization": adaptive,
        "resolved_reactivation_policy": str(reactivation["init_policy"]),
        "resolved_reactivation_m": float(reactivation["init_m"]),
        "resolved_reactivation_b": float(reactivation["init_b"]),
    }, sha256(path)


def collect(sweep_root: Path) -> tuple[pd.DataFrame, dict]:
    rows: list[dict] = []
    seen: set[tuple[str, int]] = set()
    unexpected: list[str] = []
    invalid: list[str] = []
    for config_path in sorted((sweep_root / "configs").glob("unified_config_*.yaml")):
        config = OmegaConf.load(config_path)
        try:
            architecture = architecture_from_type(str(config.model.core.type))
        except ValueError as error:
            unexpected.append(f"{config_path.name}: {error}")
            continue
        seed = int(config.experiment.seed)
        key = (architecture, seed)
        if key in seen:
            invalid.append(f"duplicate {architecture}/seed-{seed}")
            continue
        seen.add(key)
        index = config_index(config)
        result_dir = sweep_root / "results" / f"config_{index}"
        final_path = result_dir / "performance" / "final.json"
        checkpoint_path = result_dir / "main_network" / "standard_best_model.pt"
        if not final_path.is_file():
            continue
        try:
            payload = json.loads(final_path.read_text())
            validation_accuracy = float(payload["accuracy"]["valid"])
            test_accuracy = float(payload["accuracy"]["test"])
            if not all(
                np.isfinite(value) and 0.0 <= value <= 1.0
                for value in (validation_accuracy, test_accuracy)
            ):
                raise ValueError("non-finite or out-of-range accuracy")
            if not checkpoint_path.is_file():
                raise ValueError("missing best checkpoint")
            resolved, resolved_hash = audit_resolved_config(result_dir, architecture)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            invalid.append(f"{result_dir}: {error}")
            continue
        rows.append(
            {
                "architecture": architecture,
                "seed": seed,
                "config_index": index,
                "validation_accuracy": validation_accuracy,
                "test_accuracy": test_accuracy,
                **resolved,
                "config_sha256": sha256(config_path),
                "resolved_config_sha256": resolved_hash,
                "result_sha256": sha256(final_path),
                "checkpoint_sha256": sha256(checkpoint_path),
            }
        )

    expected = {
        (architecture, seed)
        for architecture in ARCHIVED_MEANS
        for seed in EXPECTED_SEEDS
    }
    completed = {(row["architecture"], row["seed"]) for row in rows}
    missing_configs = sorted(expected - seen)
    missing_results = sorted(expected - completed)
    audit = {
        "status": "complete_and_validated"
        if not (missing_configs or missing_results or unexpected or invalid)
        else "incomplete_or_invalid",
        "n_expected": len(expected),
        "n_configs_seen": len(seen),
        "n_results_complete": len(rows),
        "missing_configs": [f"{a}/seed-{s}" for a, s in missing_configs],
        "missing_results": [f"{a}/seed-{s}" for a, s in missing_results],
        "unexpected": unexpected,
        "invalid": invalid,
    }
    frame = pd.DataFrame(rows)
    if len(frame):
        frame = frame.sort_values(["architecture", "seed"]).reset_index(drop=True)
    return frame, audit


def summarize(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    rows: list[dict] = []
    for index, (architecture, archived_mean) in enumerate(ARCHIVED_MEANS.items()):
        values = frame[frame.architecture.eq(architecture)].sort_values("seed")
        test = values.test_accuracy.to_numpy(float)
        mean, low, high = bootstrap_mean(test, 43_000 + index)
        rows.append(
            {
                "architecture": architecture,
                "n_seeds": len(test),
                "mean_test_accuracy": mean,
                "sd_test_accuracy": float(test.std(ddof=1)),
                "ci95_low_test_accuracy": low,
                "ci95_high_test_accuracy": high,
                "archived_mean_test_accuracy": archived_mean,
                "new_minus_archived": mean - archived_mean,
                "within_two_pp": bool(abs(mean - archived_mean) <= COMPATIBILITY_TOLERANCE),
            }
        )
    wide = frame.pivot(index="seed", columns="architecture", values="test_accuracy")
    wide = wide.loc[list(EXPECTED_SEEDS)]
    difference = (wide["shunting"] - wide["normalized additive"]).to_numpy(float)
    mean, low, high = bootstrap_mean(difference, 43_100)
    contrast = pd.DataFrame(
        [
            {
                "contrast": "shunting minus normalized additive",
                "n_seeds": len(difference),
                "mean_paired_difference": mean,
                "ci95_low_paired_difference": low,
                "ci95_high_paired_difference": high,
                "seeds_positive": int((difference > 0).sum()),
                "seed_differences": ";".join(f"{value:.8f}" for value in difference),
            }
        ]
    )
    summary = pd.DataFrame(rows)
    decision = {
        "compatible_reproduction": bool(summary.within_two_pp.all()),
        "tolerance": "absolute mean difference <= 0.02 for each architecture",
        "if_failed": "report source/runtime discrepancy; do not tune",
    }
    return summary, contrast, decision


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

    manifest = sweep_root / "frozen_historical_manifest.json"
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
        "analysis_frozen_before_outcome_inspection": True,
    }
    if audit["status"] == "complete_and_validated":
        summary, contrast, decision = summarize(frame)
        summary.to_csv(args.output_dir / "condition_summary.csv", index=False)
        contrast.to_csv(args.output_dir / "paired_contrast.csv", index=False)
        record["decision"] = decision
    (args.output_dir / "summary.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
