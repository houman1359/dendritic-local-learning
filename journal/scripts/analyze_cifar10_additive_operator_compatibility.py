#!/usr/bin/env python3
"""Audit the frozen CIFAR-10 raw/normalized additive compatibility factorial."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from scipy import stats


EXPECTED_SEEDS = tuple(range(42, 47))
CONDITIONS = {
    "raw_no_adaptive": ("raw additive", False),
    "raw_adaptive": ("raw additive", True),
    "normalized_no_adaptive": ("normalized additive", False),
    "normalized_adaptive": ("normalized additive", True),
}
CONTRASTS = (
    (
        "normalized minus raw, no adaptive scaling",
        "normalized_no_adaptive",
        "raw_no_adaptive",
    ),
    (
        "normalized minus raw, adaptive scaling",
        "normalized_adaptive",
        "raw_adaptive",
    ),
    (
        "adaptive minus non-adaptive, raw",
        "raw_adaptive",
        "raw_no_adaptive",
    ),
    (
        "adaptive minus non-adaptive, normalized",
        "normalized_adaptive",
        "normalized_no_adaptive",
    ),
)
HISTORICAL_NORMALIZED_ADDITIVE_BP = 0.4830
ELIGIBILITY_THRESHOLD = 0.45


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def config_index(config) -> int:
    return int(str(config._sweep_config_id).rsplit("_", 1)[-1])


def bootstrap_mean(values: np.ndarray, seed: int) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    generator = np.random.default_rng(seed)
    draws = generator.choice(values, size=(100_000, len(values)), replace=True).mean(axis=1)
    low, high = np.quantile(draws, [0.025, 0.975])
    return float(values.mean()), float(low), float(high)


def t_mean_ci(values: np.ndarray, confidence: float = 0.95) -> tuple[float, float]:
    """Small-sample Student-t interval for a mean or paired difference."""
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return float("nan"), float("nan")
    mean = float(values.mean())
    sem = float(stats.sem(values))
    half_width = float(stats.t.ppf((1.0 + confidence) / 2.0, len(values) - 1) * sem)
    return mean - half_width, mean + half_width


def exact_sign_flip_p(differences: np.ndarray) -> float:
    """Two-sided paired randomization P value over all sign assignments."""
    differences = np.asarray(differences, dtype=float)
    observed = abs(float(differences.mean()))
    null_means = np.asarray(
        [
            np.mean(differences * np.asarray(signs, dtype=float))
            for signs in itertools.product((-1.0, 1.0), repeat=len(differences))
        ]
    )
    return float(np.mean(np.abs(null_means) >= observed - 1e-15))


def _read_applied_gate(result_dir: Path) -> tuple[float, float, str]:
    path = result_dir / "init_gate_stats.json"
    if not path.is_file():
        raise ValueError("missing init_gate_stats.json")
    payload = json.loads(path.read_text())
    modules = payload.get("reactivation_modules", {})
    if not modules:
        raise ValueError("no reactivation modules in init_gate_stats.json")
    m_values = np.asarray([record["m"]["mean"] for record in modules.values()], float)
    b_values = np.asarray([record["b"]["mean"] for record in modules.values()], float)
    if not np.allclose(m_values, 0.1, rtol=0, atol=2e-7):
        raise ValueError(f"applied m differs from frozen 0.1: {m_values.tolist()}")
    if not np.allclose(b_values, 0.0, rtol=0, atol=2e-7):
        raise ValueError(f"applied b differs from frozen 0: {b_values.tolist()}")
    return float(m_values.mean()), float(b_values.mean()), sha256(path)


def _audit_resolved_config(
    result_dir: Path, operator: str, adaptive_expected: bool
) -> tuple[bool, bool, str]:
    path = result_dir / "config.json"
    if not path.is_file():
        raise ValueError("missing resolved config.json")
    payload = json.loads(path.read_text())
    core = payload["model"]["core"]
    morphology = core["morphology"]
    implementation = core["implementation"]
    normalized_observed = bool(morphology["use_additive_normalization"])
    normalized_expected = operator == "normalized additive"
    adaptive_observed = bool(implementation["adaptive_initialization"])
    if normalized_observed != normalized_expected:
        raise ValueError(
            f"operator mismatch: expected normalized={normalized_expected}, "
            f"observed {normalized_observed}"
        )
    if adaptive_observed != adaptive_expected:
        raise ValueError(
            f"adaptive mismatch: expected {adaptive_expected}, observed {adaptive_observed}"
        )
    return normalized_observed, adaptive_observed, sha256(path)


def collect(sweep_root: Path) -> tuple[pd.DataFrame, dict]:
    rows: list[dict] = []
    seen: set[tuple[str, int]] = set()
    unexpected: list[str] = []
    invalid: list[str] = []
    for config_path in sorted((sweep_root / "configs").glob("unified_config_*.yaml")):
        config = OmegaConf.load(config_path)
        variant = str(config.get("_sweep_variant", ""))
        if variant not in CONDITIONS:
            unexpected.append(f"{config_path.name}: {variant}")
            continue
        seed = int(config.experiment.seed)
        key = (variant, seed)
        if key in seen:
            invalid.append(f"duplicate {variant}/seed-{seed}")
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
            operator, adaptive_expected = CONDITIONS[variant]
            normalized, adaptive, resolved_hash = _audit_resolved_config(
                result_dir, operator, adaptive_expected
            )
            gate_m, gate_b, gate_hash = _read_applied_gate(result_dir)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            invalid.append(f"{result_dir}: {error}")
            continue
        rows.append(
            {
                "variant": variant,
                "operator": operator,
                "adaptive_initialization": adaptive,
                "normalized_observed": normalized,
                "seed": seed,
                "config_index": index,
                "validation_accuracy": validation_accuracy,
                "test_accuracy": test_accuracy,
                "gate_m": gate_m,
                "gate_b": gate_b,
                "config_sha256": sha256(config_path),
                "resolved_config_sha256": resolved_hash,
                "gate_stats_sha256": gate_hash,
                "result_sha256": sha256(final_path),
                "checkpoint_sha256": sha256(checkpoint_path),
            }
        )

    expected = {(variant, seed) for variant in CONDITIONS for seed in EXPECTED_SEEDS}
    completed = {(row["variant"], row["seed"]) for row in rows}
    missing_configs = sorted(expected - seen)
    missing_results = sorted(expected - completed)
    audit = {
        "status": "complete_and_validated"
        if not (missing_configs or missing_results or unexpected or invalid)
        else "incomplete_or_invalid",
        "n_expected": len(expected),
        "n_configs_seen": len(seen),
        "n_results_complete": len(rows),
        "missing_configs": [f"{v}/seed-{s}" for v, s in missing_configs],
        "missing_results": [f"{v}/seed-{s}" for v, s in missing_results],
        "unexpected": unexpected,
        "invalid": invalid,
    }
    frame = pd.DataFrame(rows)
    if len(frame):
        frame = frame.sort_values(["variant", "seed"]).reset_index(drop=True)
    return frame, audit


def summarize(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    summary_rows: list[dict] = []
    wide = frame.pivot(index="seed", columns="variant", values="test_accuracy")
    wide = wide.loc[list(EXPECTED_SEEDS), list(CONDITIONS)]
    for index, variant in enumerate(CONDITIONS):
        values = wide[variant].to_numpy(float)
        mean, low, high = bootstrap_mean(values, 42_000 + index)
        t_low, t_high = t_mean_ci(values)
        operator, adaptive = CONDITIONS[variant]
        summary_rows.append(
            {
                "variant": variant,
                "operator": operator,
                "adaptive_initialization": adaptive,
                "n_seeds": len(values),
                "mean_test_accuracy": mean,
                "sd_test_accuracy": float(values.std(ddof=1)),
                "ci95_low_test_accuracy": low,
                "ci95_high_test_accuracy": high,
                "t_ci95_low_test_accuracy": t_low,
                "t_ci95_high_test_accuracy": t_high,
            }
        )

    contrast_rows: list[dict] = []
    for index, (name, left, right) in enumerate(CONTRASTS):
        differences = (wide[left] - wide[right]).to_numpy(float)
        mean, low, high = bootstrap_mean(differences, 42_100 + index)
        t_low, t_high = t_mean_ci(differences)
        contrast_rows.append(
            {
                "contrast": name,
                "left": left,
                "right": right,
                "n_seeds": len(differences),
                "mean_paired_difference": mean,
                "ci95_low_paired_difference": low,
                "ci95_high_paired_difference": high,
                "t_ci95_low_paired_difference": t_low,
                "t_ci95_high_paired_difference": t_high,
                "exact_sign_flip_p": exact_sign_flip_p(differences),
                "seeds_positive": int((differences > 0).sum()),
                "seed_differences": ";".join(f"{value:.8f}" for value in differences),
            }
        )

    # Difference in differences: whether adaptive initialization changes the
    # normalized-minus-raw operator contrast.
    interaction = (
        wide["normalized_adaptive"]
        - wide["normalized_no_adaptive"]
        - wide["raw_adaptive"]
        + wide["raw_no_adaptive"]
    ).to_numpy(float)
    mean, low, high = bootstrap_mean(interaction, 42_200)
    t_low, t_high = t_mean_ci(interaction)
    contrast_rows.append(
        {
            "contrast": "operator by adaptive-initialization interaction",
            "left": "(normalized_adaptive-normalized_no_adaptive)",
            "right": "(raw_adaptive-raw_no_adaptive)",
            "n_seeds": len(interaction),
            "mean_paired_difference": mean,
            "ci95_low_paired_difference": low,
            "ci95_high_paired_difference": high,
            "t_ci95_low_paired_difference": t_low,
            "t_ci95_high_paired_difference": t_high,
            "exact_sign_flip_p": exact_sign_flip_p(interaction),
            "seeds_positive": int((interaction > 0).sum()),
            "seed_differences": ";".join(f"{value:.8f}" for value in interaction),
        }
    )

    normalized_no_adaptive = float(wide["normalized_no_adaptive"].mean())
    eligible = bool(normalized_no_adaptive >= ELIGIBILITY_THRESHOLD)
    decision = {
        "historical_context_mean": HISTORICAL_NORMALIZED_ADDITIVE_BP,
        "current_normalized_no_adaptive_mean": normalized_no_adaptive,
        "eligibility_threshold_met": eligible,
        "eligible_for_separately_frozen_normalized_additive_feedback_pilot": eligible,
        "historical_compatibility_assessed_here": False,
        "rule": (
            "eligibility requires >=45% mean BP accuracy for the explicit normalized-"
            "additive, non-adaptive historical configuration"
        ),
        "compatibility_note": (
            "Historical compatibility is determined by the separately pinned "
            "source reproduction, not by this point-mean eligibility gate."
        ),
    }
    return pd.DataFrame(summary_rows), pd.DataFrame(contrast_rows), decision


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

    manifest = sweep_root / "frozen_sweep_manifest.json"
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
        summary, contrasts, decision = summarize(frame)
        summary.to_csv(args.output_dir / "condition_summary.csv", index=False)
        contrasts.to_csv(args.output_dir / "paired_contrasts.csv", index=False)
        record["decision"] = decision
    (args.output_dir / "summary.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
