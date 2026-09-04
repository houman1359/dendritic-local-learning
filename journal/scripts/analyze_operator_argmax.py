#!/usr/bin/env python3
"""Constructive argmax-U(M) exhibit for the route-dictionary atlas.

Deterministic secondary analysis; no training outcomes are altered or rerun.

The credit-operator reanalysis evaluates, at the frozen initialization, the
one-step smoothness-bound utility U(M) (``maximum_guaranteed_decrease``) for
every route family M at every feedback bandwidth K.  If U(M) is a usable
design criterion, then for a parameterized family the bandwidth that
maximizes the seed-mean utility should be the bandwidth at which the trained
factorial actually learns best.  This script tests that constructively:

For every (architecture, feedback_family) pair present in BOTH frozen files
with at least 2 distinct budget_k values (this excludes backpropagation,
exact_compartment_transport, and neuron_indexed_shared, each pinned to a
single K):

- predicted_best_k        = argmax_K seed-mean maximum_guaranteed_decrease
                            (operator_metrics.csv, 20 seeds per cell);
- realized_best_k_one_step = argmax_K mean_norm_matched_one_step_progress
                            (condition_summary.csv, trained factorial);
- realized_best_k_accuracy = argmax_K mean_heldout_accuracy.

Ties in an argmax are broken toward the smallest K (rows are sorted by
budget_k before idxmax), which is deterministic and conservative for the
prediction; a family whose utility is constant over the whole grid (the
within-neuron route derangement, where the bound guarantees no decrease at
any K) is flagged with utility_tie_across_grid and its Spearman rho is
undefined (NaN).  Per family we also record the Spearman rho, across the K
grid, between seed-mean U(M) and each realized metric.  Overall concordance
is the fraction of families where the predicted argmax equals each realized
argmax exactly, and where it lands within one rung of the sorted K grid.

Inputs (read-only, frozen):
- source_data/credit_phase_existing/operator_metrics.csv
- source_data/trained_subtree_address_full_factorial/condition_summary.csv

Outputs:
- source_data/route_dictionary_atlas/argmax_summary.csv
- source_data/route_dictionary_atlas/argmax_report.json
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OPERATOR = ROOT / "source_data" / "credit_phase_existing" / "operator_metrics.csv"
OUTCOMES = (
    ROOT
    / "source_data"
    / "trained_subtree_address_full_factorial"
    / "condition_summary.csv"
)
OUTPUT = ROOT / "source_data" / "route_dictionary_atlas"

KEYS = ["architecture", "feedback_family", "budget_k"]
FAMILY = ["architecture", "feedback_family"]

METHOD = (
    "Per (architecture, feedback_family) with >=2 distinct budget_k in both "
    "frozen files: argmax over K of seed-mean maximum_guaranteed_decrease "
    "(initialization utility bound U(M)) compared with the argmax of "
    "mean_norm_matched_one_step_progress and of mean_heldout_accuracy in the "
    "trained factorial; ties broken toward smallest K."
)


def argmax_row(group: pd.DataFrame, column: str) -> pd.Series:
    """First-maximum row of ``column`` in a group pre-sorted by budget_k."""
    return group.loc[group[column].idxmax()]


def rung(k: int, grid: list[int]) -> int:
    return grid.index(k)


def main() -> None:
    operator = pd.read_csv(OPERATOR)
    outcomes = pd.read_csv(OUTCOMES)

    utility = (
        operator.groupby(KEYS, as_index=False)
        .agg(
            mean_maximum_guaranteed_decrease=("maximum_guaranteed_decrease", "mean"),
            n_seeds_operator=("seed", "nunique"),
        )
        .sort_values(KEYS)
    )
    merged = utility.merge(
        outcomes[
            KEYS + ["mean_norm_matched_one_step_progress", "mean_heldout_accuracy"]
        ],
        on=KEYS,
        how="inner",
        validate="one_to_one",
    ).sort_values(KEYS, kind="mergesort")

    rows = []
    for (arch, family), group in merged.groupby(FAMILY, sort=True):
        if group["budget_k"].nunique() < 2:
            continue
        grid = sorted(group["budget_k"].tolist())
        pred = argmax_row(group, "mean_maximum_guaranteed_decrease")
        real_step = argmax_row(group, "mean_norm_matched_one_step_progress")
        real_acc = argmax_row(group, "mean_heldout_accuracy")
        utility_varies = group["mean_maximum_guaranteed_decrease"].nunique() > 1

        def rho(column: str) -> float:
            if not utility_varies:
                return float("nan")
            return group["mean_maximum_guaranteed_decrease"].corr(
                group[column], method="spearman"
            )

        rows.append(
            {
                "architecture": arch,
                "feedback_family": family,
                "budget_grid": ",".join(str(k) for k in grid),
                "n_budgets": len(grid),
                "predicted_best_k": int(pred["budget_k"]),
                "utility_at_predicted_best": pred["mean_maximum_guaranteed_decrease"],
                "realized_best_k_one_step": int(real_step["budget_k"]),
                "one_step_progress_at_realized_best": real_step[
                    "mean_norm_matched_one_step_progress"
                ],
                "realized_best_k_accuracy": int(real_acc["budget_k"]),
                "accuracy_at_realized_best": real_acc["mean_heldout_accuracy"],
                "spearman_utility_vs_one_step": rho(
                    "mean_norm_matched_one_step_progress"
                ),
                "spearman_utility_vs_accuracy": rho("mean_heldout_accuracy"),
                "utility_tie_across_grid": not utility_varies,
                "match_one_step": int(pred["budget_k"]) == int(real_step["budget_k"]),
                "match_accuracy": int(pred["budget_k"]) == int(real_acc["budget_k"]),
                "within_one_rung_one_step": abs(
                    rung(int(pred["budget_k"]), grid)
                    - rung(int(real_step["budget_k"]), grid)
                )
                <= 1,
                "within_one_rung_accuracy": abs(
                    rung(int(pred["budget_k"]), grid)
                    - rung(int(real_acc["budget_k"]), grid)
                )
                <= 1,
            }
        )

    summary = pd.DataFrame(rows).sort_values(FAMILY, kind="mergesort")
    n_families = len(summary)
    grids = sorted(summary["budget_grid"].unique())

    report = {
        "method": METHOD,
        "n_families": int(n_families),
        "budget_k_grid": [int(k) for k in grids[0].split(",")]
        if len(grids) == 1
        else grids,
        "concordance": {
            "exact_one_step": float(summary["match_one_step"].mean()),
            "exact_accuracy": float(summary["match_accuracy"].mean()),
            "within_one_rung_one_step": float(
                summary["within_one_rung_one_step"].mean()
            ),
            "within_one_rung_accuracy": float(
                summary["within_one_rung_accuracy"].mean()
            ),
            "exact_one_step_count": int(summary["match_one_step"].sum()),
            "exact_accuracy_count": int(summary["match_accuracy"].sum()),
            "within_one_rung_one_step_count": int(
                summary["within_one_rung_one_step"].sum()
            ),
            "within_one_rung_accuracy_count": int(
                summary["within_one_rung_accuracy"].sum()
            ),
        },
        "n_families_utility_tie": int(summary["utility_tie_across_grid"].sum()),
        "inputs": {
            "operator_metrics": str(OPERATOR.relative_to(ROOT)),
            "condition_summary": str(OUTCOMES.relative_to(ROOT)),
        },
    }

    OUTPUT.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUTPUT / "argmax_summary.csv", index=False, float_format="%.10g")
    (OUTPUT / "argmax_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(summary.to_string(index=False))
    print(json.dumps(report["concordance"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
