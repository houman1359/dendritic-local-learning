#!/usr/bin/env python3
"""Recompute credit-phase contrasts at the declared independent-seed unit.

The original depth table contains four task depths per simulation seed. This
analysis retains those transparent rows but averages their paired contrasts
within seed before bootstrap resampling and hypothesis testing.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "source_data" / "credit_phase_theory" / "depth_training_seed.csv"
OUTPUT = ROOT / "source_data" / "credit_phase_theory"
BOOTSTRAP_DRAWS = 20_000
BOOTSTRAP_SEED = 2_740_002


def bootstrap_mean(values: np.ndarray) -> tuple[float, float, float]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    indices = rng.integers(0, len(values), size=(BOOTSTRAP_DRAWS, len(values)))
    draws = values[indices].mean(axis=1)
    return (
        float(values.mean()),
        float(np.quantile(draws, 0.025)),
        float(np.quantile(draws, 0.975)),
    )


def main() -> None:
    depth = pd.read_csv(SOURCE)
    aligned = depth[depth.method.eq("aligned_tree")]
    matched = aligned[aligned.task_depth.eq(aligned.model_depth)][
        ["seed", "task_depth", "final_population_loss"]
    ]
    best_mismatched = (
        aligned[~aligned.task_depth.eq(aligned.model_depth)]
        .groupby(["seed", "task_depth"], as_index=False)
        .final_population_loss.min()
        .rename(columns={"final_population_loss": "best_mismatched_loss"})
    )
    seed_task = matched.merge(
        best_mismatched,
        on=["seed", "task_depth"],
        validate="one_to_one",
    )
    # Positive values mean lower loss at the matched depth.
    seed_task["matched_loss_reduction"] = (
        seed_task.best_mismatched_loss - seed_task.final_population_loss
    )
    seed_level = (
        seed_task.groupby("seed", as_index=False)
        .matched_loss_reduction.mean()
        .sort_values("seed")
    )
    values = seed_level.matched_loss_reduction.to_numpy(float)
    mean, low, high = bootstrap_mean(values)
    result = {
        "analysis": "matched_depth_vs_best_mismatched_depth",
        "metric": "final_population_loss_reduction",
        "inferential_unit": "independent simulation seed",
        "task_depths_averaged_within_seed": 4,
        "n_seed_task_rows": int(len(seed_task)),
        "n_independent_seeds": int(len(seed_level)),
        "mean_difference": mean,
        "ci95_low": low,
        "ci95_high": high,
        "positive_seeds": int(np.sum(values > 0)),
        "ties": int(np.sum(np.isclose(values, 0))),
        "wilcoxon_p_two_sided": float(
            wilcoxon(values, zero_method="wilcox").pvalue
        ),
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "bootstrap_seed": BOOTSTRAP_SEED,
    }
    seed_task.to_csv(
        OUTPUT / "depth_matched_seed_task_contrasts.csv",
        index=False,
        float_format="%.10g",
    )
    seed_level.to_csv(
        OUTPUT / "depth_matched_seed_level_contrasts.csv",
        index=False,
        float_format="%.10g",
    )
    (OUTPUT / "depth_matched_seed_level_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
