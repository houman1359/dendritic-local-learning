from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


JOURNAL = Path(__file__).resolve().parents[1]
SCRIPT = JOURNAL / "scripts" / "freeze_pinky_v185_cohort.py"


def _module():
    spec = importlib.util.spec_from_file_location("pinky_cohort", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_position_parser_ignores_broken_legacy_coordinate_columns() -> None:
    module = _module()
    assert module.parse_position("[103860  44385    891]") == (103860, 44385, 891)


def test_selection_is_deterministic_stratified_and_outcome_free() -> None:
    module = _module()
    rows = []
    for index in range(48):
        rows.append(
            {
                "id": index,
                "cell_type": "e" if index < 36 else "i",
                "pt_position": f"[{100 + index} {200 + index} {300 + index}]",
                "pt_root_id": 10_000 + index,
                "soma_x_nm": -1,
                "soma_y_nm": -1,
                "soma_z_nm": -1,
            }
        )
    frame = pd.DataFrame(rows)
    first = module.select(frame)
    second = module.select(frame.sample(frac=1, random_state=4))
    assert first["root_id"].tolist() == second["root_id"].tolist()
    assert len(first) == 12
    assert first["selection_stratum"].tolist() == list(range(12))
    assert "soma_x_nm" not in first.columns
