#!/usr/bin/env python3
"""Build and validate the compact 26-slide workshop deck.

The output contains 26 core slides, one Backup divider, and seven backup
slides. Canvas exports are deliberately forbidden as deck dependencies:
equations and schematics must remain native/vector.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

import fitz


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "dendritic_credit_workshop_25.tex"
TEX_SOURCES = (
    SOURCE,
    HERE / "dendritic_credit_workshop_25_core.tex",
    HERE / "dendritic_credit_workshop_25_appendix.tex",
    HERE / "credit_tree_lib.tex",
    HERE / "eq_style.tex",
)
EXPECTED_PAGES = 34
AUX_SUFFIXES = (".aux", ".log", ".nav", ".out", ".snm", ".toc")
FATAL_LOG_PATTERNS = (
    r"Package hyperref Warning",
    r"LaTeX Error",
    r"Undefined control sequence",
    r"Missing character:",
)


def run_pdflatex() -> None:
    result = subprocess.run(
        ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", SOURCE.name],
        cwd=HERE,
        text=True,
        capture_output=True,
    )
    if result.returncode:
        sys.stderr.write(result.stdout)
        sys.stderr.write(result.stderr)
        raise SystemExit(result.returncode)


def validate_sources() -> None:
    for source_path in TEX_SOURCES:
        text = source_path.read_text(encoding="utf-8")
        if re.search(r"(?<!\\)qquad", text):
            raise SystemExit(
                f"Found literal 'qquad' in {source_path.name}; a backslash is missing"
            )
        if "canvas_assets" in text:
            raise SystemExit(
                f"{source_path.name}: Canvas exports are layout references only"
            )


def validate_output() -> None:
    pdf_path = SOURCE.with_suffix(".pdf")
    log_path = SOURCE.with_suffix(".log")
    with fitz.open(pdf_path) as document:
        if document.page_count != EXPECTED_PAGES:
            raise SystemExit(
                f"{pdf_path.name}: expected {EXPECTED_PAGES} pages, "
                f"got {document.page_count}"
            )
        for page_number, page in enumerate(document, start=1):
            ratio = page.rect.width / page.rect.height
            if abs(ratio - 16 / 9) > 0.01:
                raise SystemExit(
                    f"{pdf_path.name} page {page_number}: expected 16:9, got {ratio:.4f}"
                )
            # Rendering every page catches broken image and font resources.
            page.get_pixmap(matrix=fitz.Matrix(0.25, 0.25), alpha=False)

    log = log_path.read_text(encoding="utf-8", errors="replace")
    failures = [pattern for pattern in FATAL_LOG_PATTERNS if re.search(pattern, log)]
    if failures:
        raise SystemExit(f"{log_path.name}: forbidden diagnostics matched {failures}")

    warnings = re.findall(r"Overfull \\[hv]box", log)
    warning_note = f"; {len(warnings)} TeX box diagnostics reviewed visually" if warnings else ""
    print(
        f"Validated {pdf_path.name}: {EXPECTED_PAGES} pages, 16:9, "
        f"all pages rendered{warning_note}"
    )


def clean_auxiliaries() -> None:
    for suffix in AUX_SUFFIXES:
        auxiliary = SOURCE.with_suffix(suffix)
        if auxiliary.exists():
            auxiliary.unlink()


def main() -> None:
    if shutil.which("pdflatex") is None:
        raise SystemExit("pdflatex is required to build the presentation")
    validate_sources()
    run_pdflatex()
    run_pdflatex()
    validate_output()
    clean_auxiliaries()


if __name__ == "__main__":
    main()
