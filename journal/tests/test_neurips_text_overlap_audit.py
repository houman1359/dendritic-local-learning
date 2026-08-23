from __future__ import annotations

import importlib.util
from pathlib import Path


JOURNAL_ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "audit_neurips_text_overlap",
    JOURNAL_ROOT / "scripts" / "audit_neurips_text_overlap.py",
)
assert SPEC is not None and SPEC.loader is not None
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def test_spaced_line_break_does_not_open_display_math() -> None:
    source = (
        r"Author One\\[4pt] Author Two. "
        r"This sentence must remain visible to the overlap audit. "
        r"\[x^2\] This sentence must remain too."
    )

    extracted = audit.prose(source)

    assert "Author Two" in extracted
    assert "This sentence must remain visible" in extracted
    assert "This sentence must remain too" in extracted
    assert "x^2" not in extracted
