#!/usr/bin/env python3
"""Summarize the branch-number-dependent path-necessity boundary."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "source_data" / "path_necessity_fashion"


def interpolated_chance_crossing(part: pd.DataFrame) -> float:
    """Return the first linear crossing of mean trained accuracy through 0.5."""

    required = {"conflict_probability", "mean_test_accuracy"}
    missing = required.difference(part.columns)
    if missing:
        raise ValueError(f"condition summary is missing {sorted(missing)}")
    ordered = part.sort_values("conflict_probability")
    x = ordered.conflict_probability.to_numpy(float)
    y = ordered.mean_test_accuracy.to_numpy(float)
    crossing = np.flatnonzero(y <= 0.5)
    if not len(crossing):
        return float("nan")
    index = int(crossing[0])
    if index == 0 or np.isclose(y[index], y[index - 1]):
        return float(x[index])
    return float(
        x[index - 1]
        + (0.5 - y[index - 1])
        * (x[index] - x[index - 1])
        / (y[index] - y[index - 1])
    )


def summarize_plotted_crossings(condition_summary: pd.DataFrame) -> pd.DataFrame:
    """Materialize the descriptive mean-curve crossings plotted in Fig. S29C."""

    required = {"branches", "condition", "conflict_probability", "mean_test_accuracy"}
    missing = required.difference(condition_summary.columns)
    if missing:
        raise ValueError(f"condition summary is missing {sorted(missing)}")
    shared = condition_summary[
        condition_summary.condition.eq("neuron_shared_k1")
    ]
    rows = []
    for branches, part in shared.groupby("branches", sort=True):
        branches = int(branches)
        predicted = branches / (2.0 * (branches - 1))
        trained = interpolated_chance_crossing(part)
        rows.append(
            {
                "branches": branches,
                "predicted_boundary": predicted,
                "trained_mean_curve_chance_crossing": trained,
                "trained_minus_predicted": trained - predicted,
                "chance_accuracy": 0.5,
                "interpolation": "first linear crossing of the plotted mean shared-credit accuracy curve",
            }
        )
    return pd.DataFrame(rows)


def summarize_boundaries(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    required = {
        "seed",
        "branches",
        "predicted_boundary",
        "first_nonpositive_utility_dose",
        "first_at_or_below_chance_accuracy_dose",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"boundary table is missing {sorted(missing)}")

    rows = []
    for branches, part in frame.groupby("branches", sort=True):
        row: dict[str, object] = {
            "branches": int(branches),
            "predicted_boundary": float(part.predicted_boundary.iloc[0]),
            "n_seeds": int(part.seed.nunique()),
        }
        for column, stem in (
            ("first_nonpositive_utility_dose", "utility_boundary"),
            ("first_at_or_below_chance_accuracy_dose", "accuracy_boundary"),
        ):
            observed = part[column].dropna().to_numpy(float)
            row[f"{stem}_observed_seeds"] = int(len(observed))
            row[f"mean_{stem}"] = float(np.mean(observed)) if len(observed) else np.nan
            row[f"min_{stem}"] = float(np.min(observed)) if len(observed) else np.nan
            row[f"max_{stem}"] = float(np.max(observed)) if len(observed) else np.nan
        rows.append(row)

    utility = frame.pivot(
        index="seed", columns="branches", values="first_nonpositive_utility_dose"
    )
    accuracy = frame.pivot(
        index="seed", columns="branches", values="first_at_or_below_chance_accuracy_dose"
    )
    # A missing B=2 crossing means it lies beyond the tested interval and is
    # therefore right-censored above chi=1. This preserves the predicted
    # ordering without pretending that 1.01 is an observed dose.
    utility_order = (utility.fillna(1.01)[8] < utility.fillna(1.01)[4]) & (
        utility.fillna(1.01)[4] < utility.fillna(1.01)[2]
    )
    accuracy_order = (accuracy.fillna(1.01)[8] < accuracy.fillna(1.01)[4]) & (
        accuracy.fillna(1.01)[4] < accuracy.fillna(1.01)[2]
    )
    result = {
        "ordered_prediction": "B=8 boundary < B=4 boundary < B=2 boundary",
        "utility_strict_order_pairs": int(utility_order.sum()),
        "utility_total_pairs": int(len(utility_order)),
        "utility_exact_sign_p_two_sided": float(
            binomtest(int(utility_order.sum()), len(utility_order), 0.5).pvalue
        ),
        "accuracy_strict_order_pairs": int(accuracy_order.sum()),
        "accuracy_total_pairs": int(len(accuracy_order)),
        "accuracy_exact_sign_p_two_sided": float(
            binomtest(int(accuracy_order.sum()), len(accuracy_order), 0.5).pvalue
        ),
        "right_censoring_rule": "missing boundary is treated as beyond chi=1 only for the ordering test",
    }
    return pd.DataFrame(rows), result


def main() -> None:
    frame = pd.read_csv(SOURCE / "boundary_by_seed.csv")
    summary, order = summarize_boundaries(frame)
    summary.to_csv(SOURCE / "boundary_summary.csv", index=False, float_format="%.10g")
    plotted = summarize_plotted_crossings(
        pd.read_csv(SOURCE / "condition_summary.csv")
    )
    plotted.to_csv(
        SOURCE / "plotted_crossings.csv", index=False, float_format="%.10g"
    )
    with (SOURCE / "boundary_order.json").open("w") as handle:
        json.dump(order, handle, indent=2)
        handle.write("\n")
    print(summary.to_string(index=False))
    print(plotted.to_string(index=False))
    print(json.dumps(order, indent=2))


if __name__ == "__main__":
    main()
