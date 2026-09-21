#!/usr/bin/env python3
"""Analyze the frozen correct-versus-deranged ancestry-routing cohort.

This focused analysis is intentionally independent of the other prospective
follow-up families. It runs only when all 160 routing rows pass the artifact
audit, contain the frozen seeds 42--51, and form exactly 80 paired comparisons.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "analysis"
OUTPUT = ROOT / "source_data" / "prospective_routing_control"


def bootstrap_ci(values: np.ndarray, seed: int, n_boot: int = 20_000):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(n_boot, values.size), replace=True).mean(axis=1)
    low, high = np.quantile(draws, [0.025, 0.975])
    return float(values.mean()), float(low), float(high)


def benjamini_hochberg(values: pd.Series) -> pd.Series:
    p = values.to_numpy(dtype=float)
    order = np.argsort(p)
    ranked = p[order]
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    restored = np.empty_like(adjusted)
    restored[order] = np.clip(adjusted, 0.0, 1.0)
    return pd.Series(restored, index=values.index)


def validate(frame: pd.DataFrame) -> pd.DataFrame:
    required = {
        "run_dir", "status", "task", "core", "strategy", "feedback",
        "routing", "topology", "depth", "seed", "test_accuracy",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise SystemExit(f"Audit table is missing columns: {', '.join(missing)}")
    part = frame[frame.run_dir.str.contains("ancestry_routing", na=False)].copy()
    if len(part) != 160:
        raise SystemExit(f"Expected 160 frozen routing runs, found {len(part)}")
    if not part.status.eq("pass").all():
        raise SystemExit(
            f"Refusing partial analysis: {(part.status != 'pass').sum()} routing rows failed"
        )
    if set(part.seed.unique()) != set(range(42, 52)):
        raise SystemExit(f"Unexpected seeds: {sorted(part.seed.unique())}")
    if set(part.task.unique()) != {"mnist", "noise_resilience"}:
        raise SystemExit("Unexpected task set")
    if set(part.core.unique()) != {"dendritic_shunting", "dendritic_additive"}:
        raise SystemExit("Unexpected core set")
    if set(part.depth.unique()) != {2, 4}:
        raise SystemExit("Unexpected depth set")
    expected_modes = {
        ("per_soma_shared", "correct"),
        ("per_soma_shuffled", "shuffled"),
    }
    if set(zip(part.feedback, part.routing)) != expected_modes:
        raise SystemExit("Feedback and routing labels do not match the frozen design")
    key = ["task", "core", "depth", "routing", "seed"]
    if part.duplicated(key).any():
        raise SystemExit("Duplicate routing condition-by-seed rows")
    counts = part.groupby(["task", "core", "depth", "routing"]).seed.nunique()
    if not counts.eq(10).all():
        raise SystemExit(f"Unbalanced routing conditions:\n{counts.to_string()}")
    if not np.isfinite(part.test_accuracy.to_numpy(float)).all():
        raise SystemExit("Non-finite accuracy in routing cohort")
    return part


def summarize(part: pd.DataFrame):
    condition_rows = []
    paired_rows = []
    seed_rows = []
    counter = 0
    for (task, core, depth), group in part.groupby(["task", "core", "depth"]):
        pivot = group.pivot(index="seed", columns="routing", values="test_accuracy")
        if list(pivot.index) != list(range(42, 52)):
            raise SystemExit(f"Incomplete paired seeds for {task}, {core}, depth {depth}")
        difference = (pivot.correct - pivot.shuffled).to_numpy(float)
        mean, low, high = bootstrap_ci(difference, seed=counter)
        p_value = float(
            wilcoxon(difference, zero_method="wilcox", alternative="two-sided").pvalue
        )
        paired_rows.append(
            {
                "task": task,
                "core": core,
                "depth": int(depth),
                "n_pairs": len(difference),
                "correct_mean_accuracy": float(pivot.correct.mean()),
                "shuffled_mean_accuracy": float(pivot.shuffled.mean()),
                "mean_difference": mean,
                "ci95_low": low,
                "ci95_high": high,
                "positive_pairs": int((difference > 0).sum()),
                "wilcoxon_p_two_sided": p_value,
            }
        )
        for seed, row in pivot.iterrows():
            seed_rows.append(
                {
                    "task": task,
                    "core": core,
                    "depth": int(depth),
                    "seed": int(seed),
                    "correct_accuracy": float(row.correct),
                    "shuffled_accuracy": float(row.shuffled),
                    "correct_minus_shuffled": float(row.correct - row.shuffled),
                }
            )
        for routing in ("correct", "shuffled"):
            values = pivot[routing].to_numpy(float)
            value_mean, value_low, value_high = bootstrap_ci(values, seed=100 + counter)
            condition_rows.append(
                {
                    "task": task,
                    "core": core,
                    "depth": int(depth),
                    "routing": routing,
                    "n_seeds": len(values),
                    "mean_accuracy": value_mean,
                    "ci95_low": value_low,
                    "ci95_high": value_high,
                }
            )
        counter += 1
    contrasts = pd.DataFrame(paired_rows)
    contrasts["fdr_bh_across_eight"] = benjamini_hochberg(
        contrasts.wilcoxon_p_two_sided
    )
    return pd.DataFrame(condition_rows), contrasts, pd.DataFrame(seed_rows)


def report(contrasts: pd.DataFrame, seeds: pd.DataFrame) -> str:
    task_label = {"mnist": "MNIST", "noise_resilience": "noise task"}
    core_label = {
        "dendritic_shunting": "shunting",
        "dendritic_additive": "additive",
    }
    lines = [
        "# Bandwidth-matched ancestry-routing control",
        "",
        "> **Historical complete-cohort report; superseded for publication inference.**",
        "> The outcome-independent input-validity audit excludes the two signed-noise",
        "> positive-conductance shunting cells. Use",
        "> `source_data/prospective_input_validity/routing_valid_paired_contrasts.csv`",
        "> for the six retained contrasts and 58/60 publication count.",
        "",
        "All 160 frozen runs passed the artifact audit. Correct and deranged "
        "feedback have the same number and distribution of teaching coordinates; "
        "only their assignment to neuronal trees differs.",
        "",
    ]
    for row in contrasts.itertuples(index=False):
        lines.append(
            f"- {task_label[row.task]}, {core_label[row.core]}, depth {row.depth}: "
            f"correct routing improved accuracy by {100 * row.mean_difference:.2f} "
            f"percentage points (95% paired bootstrap CI "
            f"{100 * row.ci95_low:.2f} to {100 * row.ci95_high:.2f}; "
            f"{row.positive_pairs}/{row.n_pairs} positive pairs; exact Wilcoxon "
            f"P={row.wilcoxon_p_two_sided:.4g}; BH-adjusted P="
            f"{row.fdr_bh_across_eight:.4g})."
        )
    lines += [
        "",
        f"Across the eight conditions, {int((seeds.correct_minus_shuffled > 0).sum())}/"
        f"{len(seeds)} paired seeds favored the correct map. The effect is smaller "
        "than the previously reported scalar-to-neuron-indexed difference. The results "
        "therefore separate a large coordinate-bandwidth contribution from a smaller "
        "assignment-specific contribution of routing each coordinate to the correct "
        "neuronal tree.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--audit-csv",
        type=Path,
        default=ANALYSIS / "prospective_followup_confirmatory_audit.csv",
    )
    args = parser.parse_args()
    part = validate(pd.read_csv(args.audit_csv))
    conditions, contrasts, seeds = summarize(part)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    part.to_csv(OUTPUT / "audited_run_outcomes.csv", index=False)
    conditions.to_csv(OUTPUT / "condition_summary.csv", index=False)
    contrasts.to_csv(OUTPUT / "paired_contrasts.csv", index=False)
    seeds.to_csv(OUTPUT / "paired_seed_outcomes.csv", index=False)
    summary = {
        "n_runs": int(len(part)),
        "n_paired_seeds": int(len(seeds)),
        "positive_pairs": int((seeds.correct_minus_shuffled > 0).sum()),
        "mean_condition_effect_range_percentage_points": [
            float(100 * contrasts.mean_difference.min()),
            float(100 * contrasts.mean_difference.max()),
        ],
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (OUTPUT / "report.md").write_text(report(contrasts, seeds))


if __name__ == "__main__":
    main()
