from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd


SCRIPT = (
    Path(__file__).parents[1]
    / "scripts"
    / "analyze_cifar10_historical_bp_reproduction.py"
)
SPEC = importlib.util.spec_from_file_location(
    "analyze_cifar10_historical_bp_reproduction", SCRIPT
)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _frame(additive: float, shunting: float) -> pd.DataFrame:
    rows = []
    for architecture, mean in (
        ("normalized additive", additive),
        ("shunting", shunting),
    ):
        for offset, seed in enumerate(MODULE.EXPECTED_SEEDS):
            rows.append(
                {
                    "architecture": architecture,
                    "seed": seed,
                    "test_accuracy": mean + (offset - 2) * 0.001,
                }
            )
    return pd.DataFrame(rows)


def test_compatible_reproduction_and_paired_contrast():
    summary, contrast, decision = MODULE.summarize(_frame(0.483, 0.495))
    assert decision["compatible_reproduction"]
    assert summary.within_two_pp.all()
    assert abs(contrast.iloc[0].mean_paired_difference - 0.012) < 1e-12


def test_reproduction_fails_outside_two_percentage_points():
    _, _, decision = MODULE.summarize(_frame(0.4629, 0.495))
    assert not decision["compatible_reproduction"]
