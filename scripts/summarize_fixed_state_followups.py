#!/usr/bin/env python3
"""Summarize the fixed-state factor and norm-matched one-step diagnostics.

The inputs are the checkpoint-level outputs of
``measure_fixed_state_factorial.py`` and the long-form output of
``measure_norm_matched_one_step.py``.  Outputs remain seed-level wherever
possible so every aggregate reported in the manuscript can be reconstructed.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon


def _paired_pvalues(left: np.ndarray, right: np.ndarray) -> tuple[float, float]:
    paired_t = float(ttest_rel(left, right).pvalue)
    try:
        signed_rank = float(wilcoxon(left, right).pvalue)
    except ValueError:
        signed_rank = float("nan")
    return paired_t, signed_rank


def summarize_fixed_state(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = pd.read_csv(path)
    frame = frame[
        ~frame["condition"].str.endswith("_vs_reconstructed_exact")
    ].copy()
    keep = [
        "seed",
        "condition",
        "branch_numel_weighted_cosine",
        "branch_macro_cosine",
        "branch_concatenated_cosine",
        "branch_local_exact_norm_ratio",
    ]
    seed_level = frame[keep].sort_values(["condition", "seed"])
    summary = (
        seed_level.groupby("condition", as_index=False)
        .agg(
            n_seeds=("seed", "nunique"),
            weighted_cosine_mean=("branch_numel_weighted_cosine", "mean"),
            weighted_cosine_sem=("branch_numel_weighted_cosine", "sem"),
            concatenated_cosine_mean=("branch_concatenated_cosine", "mean"),
            concatenated_cosine_sem=("branch_concatenated_cosine", "sem"),
            norm_ratio_mean=("branch_local_exact_norm_ratio", "mean"),
            norm_ratio_sem=("branch_local_exact_norm_ratio", "sem"),
        )
        .sort_values("weighted_cosine_mean", ascending=False)
    )
    return seed_level, summary


def summarize_one_step(
    path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frame = pd.read_csv(path)
    pivot = (
        frame.pivot_table(
            index=["seed", "network_type", "relative_step"],
            columns="direction",
            values="loss_decrease",
            aggfunc="mean",
        )
        .reset_index()
        .rename_axis(columns=None)
    )
    pivot["norm_matched_fraction_of_exact"] = (
        pivot["local_norm_matched"] / pivot["exact"]
    )
    pivot["raw_fraction_of_exact"] = pivot["local_raw"] / pivot["exact"]
    seed_level = pivot.sort_values(["relative_step", "network_type", "seed"])

    summary = (
        seed_level.groupby(["network_type", "relative_step"], as_index=False)
        .agg(
            n_seeds=("seed", "nunique"),
            exact_loss_decrease_mean=("exact", "mean"),
            exact_loss_decrease_sem=("exact", "sem"),
            norm_matched_loss_decrease_mean=("local_norm_matched", "mean"),
            norm_matched_loss_decrease_sem=("local_norm_matched", "sem"),
            norm_matched_descent_count=(
                "local_norm_matched",
                lambda values: int((values > 0).sum()),
            ),
            norm_matched_fraction_of_exact_mean=(
                "norm_matched_fraction_of_exact",
                "mean",
            ),
            norm_matched_fraction_of_exact_sem=(
                "norm_matched_fraction_of_exact",
                "sem",
            ),
        )
        .sort_values(["relative_step", "network_type"])
    )

    tests: list[dict[str, float | int]] = []
    for relative_step, step_frame in seed_level.groupby("relative_step"):
        paired = step_frame.pivot(
            index="seed",
            columns="network_type",
            values="norm_matched_fraction_of_exact",
        ).dropna()
        required = {"dendritic_shunting", "dendritic_additive"}
        if not required.issubset(paired.columns) or paired.empty:
            continue
        shunting = paired["dendritic_shunting"].to_numpy(dtype=float)
        additive = paired["dendritic_additive"].to_numpy(dtype=float)
        paired_t, signed_rank = _paired_pvalues(shunting, additive)
        tests.append(
            {
                "relative_step": float(relative_step),
                "n_pairs": int(len(paired)),
                "shunting_mean": float(np.mean(shunting)),
                "additive_mean": float(np.mean(additive)),
                "paired_difference_mean": float(np.mean(shunting - additive)),
                "shunting_wins": int(np.sum(shunting > additive)),
                "paired_t_pvalue": paired_t,
                "wilcoxon_pvalue": signed_rank,
            }
        )
    return seed_level, summary, pd.DataFrame(tests)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixed-state-csv", type=Path, required=True)
    parser.add_argument("--one-step-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    fixed_seed, fixed_summary = summarize_fixed_state(args.fixed_state_csv)
    step_seed, step_summary, step_tests = summarize_one_step(args.one_step_csv)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fixed_seed.to_csv(args.output_dir / "fixed_state_factorial_seed.csv", index=False)
    fixed_summary.to_csv(
        args.output_dir / "fixed_state_factorial_summary.csv",
        index=False,
    )
    step_seed.to_csv(args.output_dir / "norm_matched_one_step_seed.csv", index=False)
    step_summary.to_csv(
        args.output_dir / "norm_matched_one_step_summary.csv",
        index=False,
    )
    step_tests.to_csv(
        args.output_dir / "norm_matched_one_step_paired.csv",
        index=False,
    )
    print(f"Saved fixed-state and one-step summaries to {args.output_dir}")


if __name__ == "__main__":
    main()
