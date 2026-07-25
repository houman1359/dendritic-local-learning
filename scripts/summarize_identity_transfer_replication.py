#!/usr/bin/env python3
"""Summarize identity-transfer 3F gradient and learning replications."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon


METRICS = [
    "test_accuracy",
    "branch_numel_weighted_cosine",
    "branch_macro_cosine",
    "branch_concatenated_cosine",
    "branch_local_exact_norm_ratio",
    "soma_exact_energy_fraction",
]


def _split_from_path(path: Path) -> str | None:
    name = path.parent.name.lower()
    for split in ("train", "valid", "test"):
        if name == split or name.endswith(f"_{split}"):
            return split
    return None


def _read_diagnostics(roots: list[Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for root in roots:
        for path in sorted(root.rglob("branch_gradient_checkpoint_summary.csv")):
            split = _split_from_path(path)
            if split is None:
                continue
            frame = pd.read_csv(path)
            frame = frame[
                frame["condition"] == "approx_direct_code_per_soma"
            ].copy()
            frame["split"] = split
            frames.append(frame)
    if not frames:
        raise FileNotFoundError("No identity-transfer branch summaries found.")
    details = pd.concat(frames, ignore_index=True).drop_duplicates(
        ["seed", "network_type", "split"],
        keep="last",
    )
    metric_columns = [
        "branch_numel_weighted_cosine",
        "branch_macro_cosine",
        "branch_concatenated_cosine",
        "branch_local_exact_norm_ratio",
        "soma_exact_energy_fraction",
    ]
    return details.groupby(["seed", "network_type"], as_index=False)[
        metric_columns
    ].mean()


def _read_performance(roots: list[Path]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for root in roots:
        for run_dir in sorted((root / "results").glob("config_*")):
            config_path = run_dir / "config.json"
            result_path = run_dir / "performance" / "final.json"
            if not config_path.exists() or not result_path.exists():
                continue
            config = json.loads(config_path.read_text(encoding="utf-8"))
            result = json.loads(result_path.read_text(encoding="utf-8"))
            rows.append(
                {
                    "seed": int(config["experiment"]["seed"]),
                    "network_type": str(config["model"]["core"]["type"]),
                    "test_accuracy": float(result["accuracy"]["test"]),
                }
            )
    if not rows:
        raise FileNotFoundError("No completed identity-transfer runs found.")
    return pd.DataFrame(rows).drop_duplicates(
        ["seed", "network_type"],
        keep="last",
    )


def _paired_tests(seed_level: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for cohort, frame in seed_level.groupby("cohort", sort=False):
        for metric in METRICS:
            pivot = frame.pivot(
                index="seed",
                columns="network_type",
                values=metric,
            ).dropna()
            required = {"dendritic_shunting", "dendritic_additive"}
            if not required.issubset(pivot.columns) or pivot.empty:
                continue
            shunting = pivot["dendritic_shunting"].to_numpy(dtype=float)
            additive = pivot["dendritic_additive"].to_numpy(dtype=float)
            try:
                signed_rank = float(wilcoxon(shunting, additive).pvalue)
            except ValueError:
                signed_rank = float("nan")
            rows.append(
                {
                    "cohort": cohort,
                    "metric": metric,
                    "n_pairs": int(len(pivot)),
                    "shunting_mean": float(np.mean(shunting)),
                    "additive_mean": float(np.mean(additive)),
                    "paired_difference_mean": float(np.mean(shunting - additive)),
                    "shunting_wins": int(np.sum(shunting > additive)),
                    "paired_t_pvalue": float(ttest_rel(shunting, additive).pvalue),
                    "wilcoxon_pvalue": signed_rank,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostic-root", type=Path, nargs="+", required=True)
    parser.add_argument("--sweep-root", type=Path, nargs="+", required=True)
    parser.add_argument("--extension-first-seed", type=int, default=47)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    seed_level = _read_diagnostics(args.diagnostic_root).merge(
        _read_performance(args.sweep_root),
        on=["seed", "network_type"],
        how="inner",
        validate="one_to_one",
    )
    seed_level["cohort"] = np.where(
        seed_level["seed"] >= args.extension_first_seed,
        "extension",
        "original",
    )
    combined = seed_level.copy()
    combined["cohort"] = "combined"
    with_cohorts = pd.concat([seed_level, combined], ignore_index=True)

    summary = (
        with_cohorts.groupby(["cohort", "network_type"], as_index=False)[METRICS]
        .agg(["count", "mean", "std", "sem"])
    )
    summary.columns = [
        "_".join(str(part) for part in column if str(part))
        if isinstance(column, tuple)
        else str(column)
        for column in summary.columns
    ]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    seed_level.to_csv(
        args.output_dir / "identity_transfer_seed.csv",
        index=False,
    )
    summary.to_csv(
        args.output_dir / "identity_transfer_summary.csv",
        index=False,
    )
    _paired_tests(with_cohorts).to_csv(
        args.output_dir / "identity_transfer_paired.csv",
        index=False,
    )
    print(f"Saved identity-transfer replication summary to {args.output_dir}")


if __name__ == "__main__":
    main()
