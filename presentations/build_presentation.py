#!/usr/bin/env python3
"""Build and validate the workshop deck (single self-contained output).

Compiles dendritic_credit_workshop.tex twice with pdflatex and enforces the
same quality gate as the legacy multi-deck package: exact page count, 16:9
aspect, and a layout-clean log (no overfull/underfull boxes, hyperref
warnings, or LaTeX errors).  Run build_pdf_assets.py first if the journal
figures changed.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

import fitz


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "dendritic_credit_workshop.tex"
TEX_SOURCES = (
    SOURCE,
    HERE / "dendritic_credit_workshop_core.tex",
    HERE / "dendritic_credit_workshop_appendix.tex",
    HERE / "credit_tree_lib.tex",
    HERE / "eq_style.tex",
)
EXPECTED_PAGES = 42
AUX_SUFFIXES = (".aux", ".log", ".nav", ".out", ".snm", ".toc")
FORBIDDEN_LOG_PATTERNS = (
    r"Overfull \\[hv]box",
    r"Underfull \\[hv]box",
    r"Package hyperref Warning",
    r"LaTeX Error",
    r"Undefined control sequence",
)


def run_pdflatex() -> None:
    command = [
        "pdflatex",
        "-interaction=nonstopmode",
        "-halt-on-error",
        SOURCE.name,
    ]
    result = subprocess.run(command, cwd=HERE, text=True, capture_output=True)
    if result.returncode:
        sys.stderr.write(result.stdout)
        sys.stderr.write(result.stderr)
        raise SystemExit(result.returncode)


def validate() -> None:
    pdf_path = SOURCE.with_suffix(".pdf")
    log_path = SOURCE.with_suffix(".log")
    with fitz.open(pdf_path) as document:
        if document.page_count != EXPECTED_PAGES:
            raise SystemExit(
                f"{pdf_path.name}: expected {EXPECTED_PAGES} pages, "
                f"got {document.page_count}"
            )
        page = document[0]
        ratio = page.rect.width / page.rect.height
        if abs(ratio - 16 / 9) > 0.01:
            raise SystemExit(f"{pdf_path.name}: expected 16:9, got {ratio:.4f}")
    log = log_path.read_text(encoding="utf-8", errors="replace")
    failures = [p for p in FORBIDDEN_LOG_PATTERNS if re.search(p, log)]
    if failures:
        raise SystemExit(f"{log_path.name}: forbidden warnings matched {failures}")
    print(f"Validated {pdf_path.name}: {EXPECTED_PAGES} pages, 16:9, clean layout log")


def main() -> None:
    if shutil.which("pdflatex") is None:
        raise SystemExit("pdflatex is required to build the presentation")
    for source_path in TEX_SOURCES:
        if re.search(r"(?<!\\)qquad", source_path.read_text(encoding="utf-8")):
            raise SystemExit(
                f"Found literal 'qquad' in {source_path.name}; a backslash is missing"
            )
    run_pdflatex()
    run_pdflatex()
    validate()
    for suffix in AUX_SUFFIXES:
        aux = SOURCE.with_suffix(suffix)
        if suffix != ".pdf" and aux.exists():
            aux.unlink()


if __name__ == "__main__":
    main()
