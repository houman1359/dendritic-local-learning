#!/usr/bin/env python3
"""Audit the manuscript against current Nature Communications Article guidance."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / "main.tex"
BIB = ROOT / "references.bib"
REPORT = ROOT / "analysis" / "NATURE_COMMUNICATIONS_FORMAT_AUDIT.md"


def remove_environment(text: str, environment: str) -> str:
    return re.sub(
        rf"\\begin\{{{environment}\}}.*?\\end\{{{environment}\}}",
        " ",
        text,
        flags=re.DOTALL,
    )


def prose_words(text: str) -> list[str]:
    text = re.sub(r"(?m)(?<!\\)%.*$", " ", text)
    for environment in (
        "figure", "figure*", "table", "table*", "equation", "align", "align*", "gather"
    ):
        text = remove_environment(text, environment)
    text = re.sub(r"\$.*?\$", " ", text, flags=re.DOTALL)
    text = re.sub(r"\\\[.*?\\\]", " ", text, flags=re.DOTALL)
    text = re.sub(r"\\(?:cite|citep|citet|ref|eqref|label)\{[^{}]*\}", " ", text)
    for _ in range(4):
        text = re.sub(
            r"\\(?:textbf|textit|emph|mathrm|operatorname)\{([^{}]*)\}", r"\1", text
        )
    text = re.sub(r"\\[A-Za-z@]+\*?(?:\[[^\]]*\])?", " ", text)
    text = text.replace("~", " ").replace("--", " ")
    text = re.sub(r"[{}_^&]", " ", text)
    return re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*", text)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    tex = MAIN.read_text(encoding="utf-8")
    abstract_match = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", tex, re.DOTALL)
    if abstract_match is None:
        raise SystemExit("Abstract not found")
    abstract_words = len(prose_words(abstract_match.group(1)))

    narrative_match = re.search(
        r"\\section\*?\{Introduction\}(.*?)\\section\*?\{Methods\}", tex, re.DOTALL
    )
    if narrative_match is None:
        raise SystemExit("Introduction-to-Methods narrative not found")
    main_words = len(prose_words(narrative_match.group(1)))
    figure_environments = len(
        re.findall(r"\\begin\{figure\*?\}", narrative_match.group(1))
    )
    continued_figures = len(
        re.findall(r"\\ContinuedFloat\b", narrative_match.group(1))
    )
    # ``ContinuedFloat`` adds panels to the preceding numbered display rather
    # than creating another display item.  Report both quantities so a large
    # journal figure is not mistakenly counted as several figures.
    main_figures = figure_environments - continued_figures
    references = len(re.findall(r"(?m)^@\w+\s*\{", BIB.read_text(encoding="utf-8")))

    title_match = re.search(r"\\title\{([^{}]*)\}", tex)
    if title_match is None:
        raise SystemExit("Title not found")
    title_words = len(prose_words(title_match.group(1)))

    checks = [
        ("Title", title_words, 15, "words", True),
        ("Abstract", abstract_words, 200, "words", True),
        # Nature Communications describes 5,000 words as approximate rather
        # than as a hard initial-submission ceiling. Report it, but do not make
        # --strict fail solely because a developed journal draft exceeds it.
        ("Introduction + Results + Discussion", main_words, 5000, "approximate words", False),
        # Keep the complete working scientific display during revision, as
        # requested by the authors. Consolidation to ten items is a pre-upload
        # editorial action rather than an active scientific-content gate.
        ("Main display items", main_figures, 10, "figures", False),
        ("References", references, 70, "entries (general guide)", True),
    ]
    failures = [
        (name, value, limit)
        for name, value, limit, _, enforced in checks
        if enforced and value > limit
    ]

    lines = [
        "# Nature Communications format audit",
        "",
        "Checked against the official Article guidance retrieved on 9 August 2026.",
        "The narrative count is mechanical and excludes figure/table environments,",
        "displayed mathematics, citations and cross-references; the portal count may differ.",
        f"The {main_figures} numbered figures occupy {figure_environments} figure environments; ",
        f"{continued_figures} environments are continued multi-panel displays.",
        "",
        "| Item | Current | Guidance | Status |",
        "|---|---:|---:|---|",
    ]
    for name, value, limit, unit, enforced in checks:
        status = "PASS" if value <= limit else ("NEEDS REVISION" if enforced else "ADVISORY")
        lines.append(f"| {name} | {value} {unit} | {limit} | {status} |")
    lines.extend(
        [
            "",
            "Official guidance:",
            "",
            "- https://www.nature.com/ncomms/submit/article",
            "- https://www.nature.com/ncomms/aims",
            "- https://www.nature.com/ncomms/submit/resources",
            "",
            "Nature Communications accepts flexible initial formatting and encourages a full",
            "submission rather than a presubmission enquiry.",
            "",
        ]
    )
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {REPORT.relative_to(ROOT)}")
    for name, value, limit, unit, _ in checks:
        print(f"{name}: {value} {unit} (guidance {limit})")
    return 1 if args.strict and failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
