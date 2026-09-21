"""Publication packaging must preserve corrected figure coordinates and IDs."""
from __future__ import annotations

import csv
import sys
from collections import Counter
from pathlib import Path

import pytest

JOURNAL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(JOURNAL / "scripts"))

import build_nature_source_data as release


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


@pytest.mark.parametrize("figure", [1, 7, 8, 9])
def test_display_release_preserves_identifiers_and_panel_rows(tmp_path, figure):
    source = f"source_data/curated_publication/figure_{figure:02d}_plotted.csv"
    items = [item for item in release.FILES if item.source == source]
    assert items, f"Current Figure {figure} display data are absent from release"
    original = rows(JOURNAL / source)
    assert original
    for index, item in enumerate(items):
        destination = tmp_path / f"released_{index}.csv"
        release.copy_source_file(item, JOURNAL / source, destination)
        release.portable_text_copy(destination)
        packaged = rows(destination)
        assert len(packaged) == len(original)
        assert Counter(row["panel"] for row in packaged) == Counter(
            row["panel"] for row in original
        )
        for key in ("root_id", "seed", "seed_index"):
            if key in original[0]:
                assert [row[key] for row in packaged] == [row[key] for row in original]
