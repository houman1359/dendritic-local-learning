from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd


SCRIPT = (
    Path(__file__).parents[1]
    / "scripts"
    / "analyze_cifar10_additive_operator_compatibility.py"
)
SPEC = importlib.util.spec_from_file_location(
    "analyze_cifar10_additive_operator_compatibility", SCRIPT
)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _frame(normalized_no_adaptive: float) -> pd.DataFrame:
    means = {
        "raw_no_adaptive": 0.40,
        "raw_adaptive": 0.42,
        "normalized_no_adaptive": normalized_no_adaptive,
        "normalized_adaptive": 0.47,
    }
    rows = []
    for variant, mean in means.items():
        operator, adaptive = MODULE.CONDITIONS[variant]
        for offset, seed in enumerate(MODULE.EXPECTED_SEEDS):
            rows.append(
                {
                    "variant": variant,
                    "operator": operator,
                    "adaptive_initialization": adaptive,
                    "seed": seed,
                    "test_accuracy": mean + (offset - 2) * 0.001,
                }
            )
    return pd.DataFrame(rows)


def test_summary_and_compatibility_gate():
    summary, contrasts, decision = MODULE.summarize(_frame(0.46))
    assert len(summary) == 4
    assert len(contrasts) == 5
    assert decision["eligibility_threshold_met"]
    assert not decision["historical_compatibility_assessed_here"]
    assert decision["eligible_for_separately_frozen_normalized_additive_feedback_pilot"]
    operator_effect = contrasts.set_index("contrast").loc[
        "normalized minus raw, no adaptive scaling", "mean_paired_difference"
    ]
    assert abs(operator_effect - 0.06) < 1e-12


def test_gate_fails_below_absolute_threshold():
    _, _, decision = MODULE.summarize(_frame(0.449))
    assert not decision["eligibility_threshold_met"]


def test_small_sample_interval_and_factorial_interaction_are_reported():
    _, contrasts, _ = MODULE.summarize(_frame(0.46))
    assert contrasts["t_ci95_low_paired_difference"].notna().all()
    assert contrasts["t_ci95_high_paired_difference"].notna().all()
    assert contrasts["exact_sign_flip_p"].between(0.0, 1.0).all()
    assert (
        contrasts["contrast"]
        == "operator by adaptive-initialization interaction"
    ).sum() == 1
