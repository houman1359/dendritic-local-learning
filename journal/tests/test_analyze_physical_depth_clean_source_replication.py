from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
ANALYZER = ROOT / "scripts" / "analyze_physical_depth_clean_source_replication.py"


def _module():
    spec = importlib.util.spec_from_file_location("clean_source_analyzer", ANALYZER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _synthetic_frame() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    slopes = {
        ("serial_tree", "shunting", "full_bp"): 0.06,
        ("grouped_point", "shunting", "full_bp"): 0.01,
        ("serial_tree", "shunting", "local_shared"): 0.03,
        ("serial_tree", "shunting", "local_path"): 0.05,
        ("serial_tree", "raw_additive", "full_bp"): -0.01,
    }
    for hierarchy in (2, 3):
        conditions = [
            ("serial_tree", "shunting", "full_bp"),
            ("grouped_point", "shunting", "full_bp"),
            ("serial_tree", "shunting", "local_shared"),
            ("serial_tree", "shunting", "local_path"),
        ]
        if hierarchy == 3:
            conditions.append(("serial_tree", "raw_additive", "full_bp"))
        for regime in ("aligned", "rewired_tree"):
            for architecture, mechanism, credit in conditions:
                if mechanism == "raw_additive" and regime != "aligned":
                    continue
                for depth in range(1, hierarchy + 1):
                    for offset, seed in enumerate(range(19000, 19010)):
                        slope = slopes[(architecture, mechanism, credit)]
                        if regime == "rewired_tree":
                            slope *= -0.1
                        rows.append(
                            {
                                "hierarchy": hierarchy,
                                "regime": regime,
                                "architecture": architecture,
                                "mechanism": mechanism,
                                "credit": credit,
                                "depth": depth,
                                "seed": seed,
                                "test_accuracy": 0.55 + slope * depth + offset * 1e-5,
                            }
                        )
    return pd.DataFrame(rows)


def test_historical_table_has_all_430_declared_pairs() -> None:
    module = _module()
    frame = module.historical_frame()
    assert len(frame) == 430
    assert not frame.duplicated(module.PAIR_KEYS).any()


def test_clean_source_contrasts_are_seed_paired() -> None:
    module = _module()
    contrasts, by_seed = module.build_contrasts(_synthetic_frame())
    assert contrasts.n_seeds.eq(10).all()
    assert len(by_seed) == 10
    value = contrasts.loc[
        contrasts.contrast.eq("h3_depth__serial_bp__aligned"), "mean_pp"
    ].iloc[0]
    assert value == pytest.approx(12.0)
    assert contrasts.loc[
        contrasts.contrast.eq("h3_placement_interaction__serial_bp"),
        "positive_claim_gate",
    ].iloc[0]
