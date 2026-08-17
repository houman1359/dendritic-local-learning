#!/usr/bin/env python3
"""Audit citation keys across the main manuscript and Supplementary Information."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCES = (ROOT / "main.tex", ROOT / "supplementary" / "supplementary.tex")
BIB = ROOT / "references.bib"
REPORT = ROOT / "analysis" / "CITATION_AUDIT.md"


def citation_keys(text: str) -> set[str]:
    keys: set[str] = set()
    for match in re.finditer(r"\\cite(?:t|p|author|year)?\*?(?:\[[^]]*\])?\{([^{}]+)\}", text):
        keys.update(key.strip() for key in match.group(1).split(",") if key.strip())
    return keys


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    bib_keys = set(re.findall(r"(?m)^@\w+\s*\{\s*([^,\s]+)\s*,", BIB.read_text(encoding="utf-8")))
    source_keys: dict[str, set[str]] = {}
    for source in SOURCES:
        source_keys[str(source.relative_to(ROOT))] = citation_keys(source.read_text(encoding="utf-8"))
    cited = set().union(*source_keys.values())
    undefined = sorted(cited - bib_keys)
    unused = sorted(bib_keys - cited)

    lines = [
        "# Citation audit",
        "",
        f"- Bibliography entries: {len(bib_keys)}",
        f"- Unique cited keys: {len(cited)}",
        f"- Undefined cited keys: {len(undefined)}",
        f"- Unused bibliography entries: {len(unused)}",
        "",
        "## Undefined keys",
        "",
        *(f"- `{key}`" for key in undefined),
        *( ["- None."] if not undefined else [] ),
        "",
        "## Unused entries",
        "",
        *(f"- `{key}`" for key in unused),
        *( ["- None."] if not unused else [] ),
        "",
        "Unused entries are reported for editorial cleanup; only undefined cited keys are a strict build failure.",
    ]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Bibliography entries: {len(bib_keys)}")
    print(f"Unique cited keys: {len(cited)}")
    print(f"Undefined cited keys: {len(undefined)}")
    print(f"Unused bibliography entries: {len(unused)}")
    return 1 if args.strict and undefined else 0


if __name__ == "__main__":
    raise SystemExit(main())
