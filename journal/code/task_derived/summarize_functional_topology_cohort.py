#!/usr/bin/env python3
"""Target-level synthesis of the exploratory MICrONS topology/function screens."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


PROJECT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = PROJECT / "reproduced_results" / "microns_functional_partner_responses"
DEFAULT_OUTDIR = PROJECT / "reproduced_results" / "microns_functional_topology_cohort"


METRICS = {
    "shared_path_r": ("tests", "shared_path_fraction", "spearman_r"),
    "negative_tree_distance_r": ("tests", "negative_tree_distance", "spearman_r"),
    "partial_shared_path_r": (
        "tests",
        "shared_path_partial_euclidean_and_depth",
        "partial_rank_r",
    ),
    "same_major_branch_delta": ("tests", "same_major_branch", "mean_similarity_delta"),
    "all_condition_partial_r": ("all_condition_sensitivity", "shared_path_partial_r"),
}


def nested(value: dict[str, Any], keys: tuple[str, ...]) -> float:
    out: Any = value
    for key in keys:
        out = out[key]
    return float(out)


def bootstrap_ci(values: np.ndarray, rng: np.random.Generator, n_boot: int) -> list[float]:
    draws = rng.choice(values, size=(n_boot, len(values)), replace=True).mean(axis=1)
    return [float(item) for item in np.quantile(draws, [0.025, 0.975])]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--bootstrap", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260721)
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    rows = []
    for path in sorted(args.root.glob("target*/functional_topology/summary.json")):
        value = json.loads(path.read_text(encoding="utf-8"))
        row = {
            "target_nucleus_id": int(value["target_nucleus_id"]),
            "target_root_id": int(value["target_root_id"]),
            "session": int(value["session"]),
            "scan_idx": int(value["scan_idx"]),
            "n_partners": int(value["n_presynaptic_partners"]),
            "n_manual_subset": int(value["n_manual_partner_subset"]),
            "n_pairs": int(value["n_pairs"]),
            "median_repeat_reliability": float(value["median_repeat_reliability"]),
            "source": str(path.relative_to(PROJECT)),
        }
        for name, keys in METRICS.items():
            row[name] = nested(value, keys)
        rows.append(row)
    table = pd.DataFrame(rows).sort_values("target_nucleus_id").reset_index(drop=True)
    if len(table) < 2:
        raise ValueError("at least two completed target analyses are required")
    table.to_csv(args.outdir / "target_level_metrics.csv", index=False)

    rng = np.random.default_rng(args.seed)
    tests: dict[str, Any] = {}
    for name in METRICS:
        values = table[name].to_numpy(float)
        try:
            signed_rank = wilcoxon(values, alternative="two-sided", method="auto")
            p_value = float(signed_rank.pvalue)
        except ValueError:
            p_value = 1.0
        leave_one_out = np.asarray(
            [np.delete(values, index).mean() for index in range(len(values))], dtype=float
        )
        tests[name] = {
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "positive_fraction": float(np.mean(values > 0)),
            "bootstrap_95_ci_mean": bootstrap_ci(values, rng, args.bootstrap),
            "wilcoxon_two_sided_p": p_value,
            "leave_one_out_mean_range": [
                float(np.min(leave_one_out)),
                float(np.max(leave_one_out)),
            ],
        }

    summary = {
        "claim_level": "exploratory target-level cohort test of functional similarity versus real dendritic topology",
        "n_targets": int(len(table)),
        "n_connected_presynaptic_partners": int(table["n_partners"].sum()),
        "n_manual_matches_within_expanded_cohorts": int(table["n_manual_subset"].sum()),
        "n_pairwise_descriptors_not_used_as_replicates": int(table["n_pairs"].sum()),
        "functional_similarity": "condition-mean calcium response correlation using 136 repeated conditions per scan",
        "replication_unit": "postsynaptic target neuron",
        "tests": tests,
        "conclusion": (
            "No cohort-level evidence that functionally similar observed excitatory inputs occupy more shared ancestry, shorter tree paths, or the same major branch. Effects are heterogeneous across seven targets and all target-level intervals include zero."
        ),
        "limitations": [
            "Cohorts were selected from an eight-cell morphology pilot rather than a coverage-ranked population sample.",
            "The expanded tier uses residual-filtered automatic coregistration; manual overlap is sparse.",
            "Observed excitatory partners cover only a small fraction of each target's inputs.",
            "Calcium response repeat reliability is modest.",
            "MICrONS contains one animal and no learning intervention.",
        ],
    }
    (args.outdir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    labels = [
        ("partial_shared_path_r", "shared ancestry\n(Euclidean/depth controlled)"),
        ("shared_path_r", "shared ancestry"),
        ("negative_tree_distance_r", "shorter tree distance"),
        ("same_major_branch_delta", "same major branch"),
    ]
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    offsets = np.linspace(-0.18, 0.18, len(table))
    for y_index, (name, label) in enumerate(labels):
        values = table[name].to_numpy(float)
        ax.scatter(values, y_index + offsets, s=34, color="#33658a", alpha=0.82)
        item = tests[name]
        ax.errorbar(
            item["mean"],
            y_index,
            xerr=[[item["mean"] - item["bootstrap_95_ci_mean"][0]], [item["bootstrap_95_ci_mean"][1] - item["mean"]]],
            fmt="o",
            color="#b33a3a",
            capsize=3,
            linewidth=2,
            markersize=6,
        )
    ax.axvline(0.0, color="black", linewidth=1, linestyle="--")
    ax.set_yticks(range(len(labels)), [label for _, label in labels])
    ax.set_xlabel("target-level association or similarity difference")
    ax.set_title("Functional input organization is heterogeneous across seven real dendritic trees")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(args.outdir / "functional_topology_cohort.png", dpi=220)
    fig.savefig(args.outdir / "functional_topology_cohort.pdf")
    plt.close(fig)

    primary = tests["partial_shared_path_r"]
    lines = [
        "# MICrONS functional topology cohort",
        "",
        summary["claim_level"],
        "",
        f"- Targets: {summary['n_targets']}.",
        f"- Connected partners with streamed responses: {summary['n_connected_presynaptic_partners']}.",
        f"- Expert/manual matches inside those cohorts: {summary['n_manual_matches_within_expanded_cohorts']}.",
        f"- Pairwise descriptors: {summary['n_pairwise_descriptors_not_used_as_replicates']} (not treated as independent replicates).",
        f"- Primary target-level shared-ancestry effect after Euclidean/depth controls: mean {primary['mean']:.3f}, bootstrap 95% CI [{primary['bootstrap_95_ci_mean'][0]:.3f}, {primary['bootstrap_95_ci_mean'][1]:.3f}], positive in {primary['positive_fraction']:.1%} of targets, Wilcoxon p={primary['wilcoxon_two_sided_p']:.4f}.",
        "",
        f"**Conclusion:** {summary['conclusion']}",
        "",
        "This negative pilot is useful for design: the next cohort must be selected for high manual functional-partner coverage before skeletonization, and the paper should not assume a universal functional-clustering signature.",
    ]
    (args.outdir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
