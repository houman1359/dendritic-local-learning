from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
ANALYZER = ROOT / "scripts" / "analyze_physical_depth_h4_factorial.py"


def _module():
    spec = importlib.util.spec_from_file_location("h4_analyzer", ANALYZER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _synthetic_frame() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    seeds = list(range(10400, 10410))
    conditions = [
        ("aligned", "serial_tree", "shunting", "full_bp"),
        ("rewired_tree", "serial_tree", "shunting", "full_bp"),
        ("aligned", "grouped_point", "shunting", "full_bp"),
        ("rewired_tree", "grouped_point", "shunting", "full_bp"),
        ("aligned", "serial_tree", "shunting", "local_shared"),
        ("rewired_tree", "serial_tree", "shunting", "local_shared"),
        ("aligned", "serial_tree", "shunting", "local_path"),
        ("rewired_tree", "serial_tree", "shunting", "local_path"),
        ("aligned", "serial_tree", "raw_additive", "full_bp"),
    ]
    offsets = {
        ("serial_tree", "shunting", "full_bp"): 0.06,
        ("grouped_point", "shunting", "full_bp"): 0.03,
        ("serial_tree", "shunting", "local_shared"): 0.02,
        ("serial_tree", "shunting", "local_path"): 0.04,
        ("serial_tree", "raw_additive", "full_bp"): 0.00,
    }
    for regime, architecture, mechanism, credit in conditions:
        for depth in (1, 2, 3, 4):
            for seed in seeds:
                slope = offsets[(architecture, mechanism, credit)]
                if regime == "rewired_tree":
                    slope *= 0.2
                rows.append(
                    {
                        "hierarchy": 4,
                        "regime": regime,
                        "architecture": architecture,
                        "mechanism": mechanism,
                        "credit": credit,
                        "depth": depth,
                        "seed": seed,
                        "test_accuracy": 0.55 + slope * depth + (seed - 10404.5) * 1e-4,
                    }
                )
    return pd.DataFrame(rows)


def test_build_contrasts_uses_ten_paired_seeds_and_primary_bh() -> None:
    module = _module()
    contrasts, by_seed = module.build_contrasts(_synthetic_frame())
    assert contrasts.n_seeds.eq(10).all()
    assert by_seed.seed.tolist() == list(range(10400, 10410))
    primary = contrasts[contrasts.family.eq("primary")]
    assert primary.bh_adjusted_p_primary_family.notna().all()
    assert primary.bh_adjusted_p_primary_family.between(0, 1).all()
    assert contrasts.loc[
        contrasts.contrast.eq("depth__serial_bp__aligned__d4_d3"), "mean_pp"
    ].iloc[0] == pytest.approx(6.0)


def test_bh_adjustment_is_monotone_in_sorted_p_values() -> None:
    module = _module()
    values = pd.Series([0.04, 0.001, 0.02, 0.2])
    adjusted = module._bh_adjust(values)
    order = np.argsort(values.to_numpy())
    assert np.all(np.diff(adjusted.to_numpy()[order]) >= 0)
    assert np.all(adjusted.to_numpy() >= values.to_numpy())
