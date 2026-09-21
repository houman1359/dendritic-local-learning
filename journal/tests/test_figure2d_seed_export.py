from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


JOURNAL = Path(__file__).resolve().parents[1]
SCRIPT = JOURNAL / "scripts" / "export_figure2d_seed_data.py"
SPEC = importlib.util.spec_from_file_location("export_figure2d_seed_data", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
EXPORT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = EXPORT
SPEC.loader.exec_module(EXPORT)


def test_archived_figure2d_runs_reproduce_frozen_summaries() -> None:
    if not (
        EXPORT.DEFAULT_FACTORIAL_RESULTS.is_dir()
        and EXPORT.DEFAULT_BACKPROP_RESULTS.is_dir()
    ):
        pytest.skip(
            "optional machine-local NeurIPS run archive is not mounted; "
            "tracked publication tables are validated by the source-data tests"
        )
    rows = EXPORT.extract_rows(
        EXPORT.DEFAULT_FACTORIAL_RESULTS,
        EXPORT.DEFAULT_BACKPROP_RESULTS,
    )
    verification, maximum_difference = EXPORT.verify_rows(
        rows,
        EXPORT.DEFAULT_FACTORIAL_SUMMARY,
        EXPORT.DEFAULT_BACKPROP_SUMMARY,
    )

    assert len(rows) == 25
    assert len({row["run_id"] for row in rows}) == 25
    assert len(verification) == 5
    assert maximum_difference < 1e-15
