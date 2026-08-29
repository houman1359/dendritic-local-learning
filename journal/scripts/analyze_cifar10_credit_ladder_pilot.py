#!/usr/bin/env python3
"""Audit and summarize the paired CIFAR-10 credit-resolution pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf


EXPECTED_SEEDS = tuple(range(10600, 10605))
CONDITIONS = {
    "cifar10_additive_strict_scalar": ("additive", "strict scalar"),
    "cifar10_additive_neuron_specific": ("additive", "neuron specific"),
    "cifar10_additive_exact_path": ("additive", "exact path"),
    "cifar10_additive_matched_bp": ("additive", "backpropagation"),
    "cifar10_shunting_strict_scalar": ("shunting", "strict scalar"),
    "cifar10_shunting_neuron_specific": ("shunting", "neuron specific"),
    "cifar10_shunting_exact_path": ("shunting", "exact path"),
    "cifar10_shunting_matched_bp": ("shunting", "backpropagation"),
}
FEEDBACK_ORDER = (
    "strict scalar",
    "neuron specific",
    "exact path",
    "backpropagation",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bootstrap_mean(values: np.ndarray, seed: int) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    generator = np.random.default_rng(seed)
    draws = generator.choice(values, size=(100_000, len(values)), replace=True).mean(axis=1)
    low, high = np.quantile(draws, [0.025, 0.975])
    return float(values.mean()), float(low), float(high)


def config_index(config) -> int:
    value = str(config._sweep_config_id)
    return int(value.rsplit("_", 1)[-1])


def collect(sweep_root: Path) -> tuple[pd.DataFrame, dict]:
    rows: list[dict] = []
    seen: set[tuple[str, int]] = set()
    unexpected: list[str] = []
    invalid: list[str] = []

    for config_path in sorted((sweep_root / "configs").glob("*.yaml")):
        if config_path.name == "metadata.yaml":
            continue
        config = OmegaConf.load(config_path)
        group = str(config.get("_seed_repeat_group", ""))
        if group not in CONDITIONS:
            unexpected.append(f"{config_path.name}: {group}")
            continue
        architecture, feedback = CONDITIONS[group]
        seed = int(config.experiment.seed)
        index = config_index(config)
        final_path = sweep_root / "results" / f"config_{index}" / "performance" / "final.json"
        if not final_path.is_file():
            continue
        payload = json.loads(final_path.read_text())
        accuracy = float(payload["accuracy"]["test"])
        if not np.isfinite(accuracy) or not 0.0 <= accuracy <= 1.0:
            invalid.append(f"{final_path}: {accuracy}")
            continue
        key = (group, seed)
        if key in seen:
            invalid.append(f"duplicate {group}/seed-{seed}")
            continue
        seen.add(key)
        rows.append(
            {
                "architecture": architecture,
                "feedback": feedback,
                "condition": group,
                "seed": seed,
                "test_accuracy": accuracy,
                "config_index": index,
                "config_sha256": sha256(config_path),
                "result_sha256": sha256(final_path),
            }
        )

    expected = {(condition, seed) for condition in CONDITIONS for seed in EXPECTED_SEEDS}
    missing = sorted(expected - seen)
    audit = {
        "status": "complete_and_validated"
        if not (missing or unexpected or invalid)
        else "incomplete_or_invalid",
        "n_expected": len(expected),
        "n_complete": len(rows),
        "missing": [f"{condition}/seed-{seed}" for condition, seed in missing],
        "unexpected": unexpected,
        "invalid": invalid,
    }
    frame = pd.DataFrame(rows)
    if len(frame):
        frame = frame.sort_values(["architecture", "seed", "feedback"]).reset_index(drop=True)
    return frame, audit


def execution_identity(sweep_root: Path) -> dict:
    launcher = sweep_root / "jobs" / "run_array_sweep.sh"
    text = launcher.read_text()

    def value(name: str) -> str:
        match = re.search(
            rf'^{name}=(?:"([^"]+)"|([^\s]+))$',
            text,
            flags=re.MULTILINE,
        )
        if match is None:
            raise RuntimeError(f"missing {name} in {launcher}")
        return match.group(1) or match.group(2)

    manifest = sweep_root / "original_manifest.yaml"
    return {
        "source_worktree": value("REPOSITORY_ROOT"),
        "source_commit": value("EXPECTED_REPOSITORY_HEAD"),
        "source_tracked_diff_sha256": value("EXPECTED_TRACKED_DIFF_SHA256"),
        "launcher_sha256": sha256(launcher),
        "manifest_sha256": sha256(manifest),
    }


def summarize(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    summary_rows: list[dict] = []
    contrast_rows: list[dict] = []
    gate_by_architecture: dict[str, dict] = {}
    for architecture_index, architecture in enumerate(("additive", "shunting")):
        part = frame[frame.architecture.eq(architecture)]
        wide = part.pivot(index="seed", columns="feedback", values="test_accuracy")
        wide = wide.loc[list(EXPECTED_SEEDS), list(FEEDBACK_ORDER)]
        for feedback_index, feedback in enumerate(FEEDBACK_ORDER):
            mean, low, high = bootstrap_mean(
                wide[feedback].to_numpy(float),
                seed=10_600_000 + 10 * architecture_index + feedback_index,
            )
            summary_rows.append(
                {
                    "architecture": architecture,
                    "feedback": feedback,
                    "n_seeds": len(wide),
                    "mean_test_accuracy": mean,
                    "ci95_low_test_accuracy": low,
                    "ci95_high_test_accuracy": high,
                }
            )

        contrasts = (
            ("neuron specific minus strict scalar", "neuron specific", "strict scalar"),
            ("exact path minus neuron specific", "exact path", "neuron specific"),
            ("exact path minus backpropagation", "exact path", "backpropagation"),
        )
        for contrast_index, (name, left, right) in enumerate(contrasts):
            differences = (wide[left] - wide[right]).to_numpy(float)
            mean, low, high = bootstrap_mean(
                differences,
                seed=10_601_000 + 20 * architecture_index + contrast_index,
            )
            contrast_rows.append(
                {
                    "architecture": architecture,
                    "contrast": name,
                    "n_seeds": len(differences),
                    "mean_difference": mean,
                    "ci95_low_difference": low,
                    "ci95_high_difference": high,
                    "seeds_positive": int((differences > 0).sum()),
                    "seed_differences": ";".join(f"{value:.8f}" for value in differences),
                }
            )

        primary = (wide["exact path"] - wide["neuron specific"]).to_numpy(float)
        mean_primary = float(primary.mean())
        seeds_positive = int((primary > 0).sum())
        gate_by_architecture[architecture] = {
            "mean_exact_minus_neuron_specific": mean_primary,
            "seeds_exact_above_neuron_specific": seeds_positive,
            "passes": bool(mean_primary >= 0.01 and seeds_positive >= 4),
        }

    decision = {
        "eligible_for_fresh_ten_seed_confirmation": bool(
            any(record["passes"] for record in gate_by_architecture.values())
        ),
        "gate_by_architecture": gate_by_architecture,
        "gate": "mean exact-path advantage >= 0.01 and positive in >=4/5 paired seeds",
    }
    return pd.DataFrame(summary_rows), pd.DataFrame(contrast_rows), decision


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
