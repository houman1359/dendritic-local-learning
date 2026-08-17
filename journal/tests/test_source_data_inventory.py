from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path


JOURNAL = Path(__file__).resolve().parents[1]
SCRIPTS = JOURNAL / "scripts"
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location(
    "build_nature_source_data", SCRIPTS / "build_nature_source_data.py"
)
assert SPEC is not None and SPEC.loader is not None
builder = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = builder
SPEC.loader.exec_module(builder)


def test_source_data_inventory_matches_final_display_numbering() -> None:
    figures = {item.figure for item in builder.FILES}
    main_numbers = {
        int(match.group(1))
        for figure in figures
        if (match := re.fullmatch(r"Figure (\d+)", figure))
    }
    supplementary_numbers = {
        int(match.group(1))
        for figure in figures
        if (match := re.fullmatch(r"Supplementary Figure (\d+)", figure))
    }
    assert main_numbers == set(range(2, 9))  # Figure 1 is conceptual.
    assert supplementary_numbers == set(range(1, 18))


def test_source_data_destinations_are_unique_and_sources_exist() -> None:
    destinations = [item.destination for item in builder.FILES]
    assert len(destinations) == len(set(destinations))
    assert not [item.source for item in builder.FILES if not (JOURNAL / item.source).is_file()]
