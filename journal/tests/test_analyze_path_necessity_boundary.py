from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "analyze_path_necessity_boundary.py"
)
SPEC = importlib.util.spec_from_file_location("path_boundary", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_boundary_summary_and_right_censored_order() -> None:
    rows = []
    for seed in (1, 2):
        rows.extend(
            [
                {
                    "seed": seed,
                    "branches": 2,
                    "predicted_boundary": 1.0,
                    "first_nonpositive_utility_dose": np.nan if seed == 1 else 1.0,
                    "first_at_or_below_chance_accuracy_dose": 1.0,
                },
                {
                    "seed": seed,
                    "branches": 4,
                    "predicted_boundary": 2 / 3,
                    "first_nonpositive_utility_dose": 0.75,
                    "first_at_or_below_chance_accuracy_dose": 0.75,
                },
                {
                    "seed": seed,
                    "branches": 8,
                    "predicted_boundary": 4 / 7,
                    "first_nonpositive_utility_dose": 0.6,
                    "first_at_or_below_chance_accuracy_dose": 0.6,
                },
            ]
        )
    summary, result = MODULE.summarize_boundaries(pd.DataFrame(rows))
    assert summary.branches.tolist() == [2, 4, 8]
    assert int(summary.loc[summary.branches.eq(2), "utility_boundary_observed_seeds"].iloc[0]) == 1
    assert result["utility_strict_order_pairs"] == 2
    assert result["accuracy_strict_order_pairs"] == 2


def test_plotted_crossing_is_materialized_from_the_mean_curve() -> None:
    frame = pd.DataFrame(
        {
            "branches": [4, 4, 4, 4],
            "condition": ["neuron_shared_k1"] * 4,
            "conflict_probability": [0.0, 0.5, 2 / 3, 0.75],
            "mean_test_accuracy": [0.8, 0.7, 0.5, 0.4],
        }
    )
    table = MODULE.summarize_plotted_crossings(frame)
    assert table.branches.tolist() == [4]
    assert np.isclose(table.predicted_boundary.iloc[0], 2 / 3)
    assert np.isclose(table.trained_mean_curve_chance_crossing.iloc[0], 2 / 3)
    assert table.interpolation.iloc[0].startswith("first linear crossing")
