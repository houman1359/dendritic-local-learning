#!/usr/bin/env python3
"""Audit the working manuscript against current Nature Neuroscience Article limits."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / "main.tex"
BIB = ROOT / "references.bib"
REPORT = ROOT / "analysis" / "NATURE_NEUROSCIENCE_FORMAT_AUDIT.md"


def remove_environment(text: str, environment: str) -> str:
    return re.sub(
        rf"\\begin\{{{environment}\}}.*?\\end\{{{environment}\}}",
        " ",
        text,
        flags=re.DOTALL,
    )


def prose_words(text: str) -> list[str]:
    text = re.sub(r"(?m)(?<!\\)%.*$", " ", text)
    for environment in ("figure", "figure*", "table", "table*", "equation", "align", "align*", "gather"):
        text = remove_environment(text, environment)
    text = re.sub(r"\$.*?\$", " ", text, flags=re.DOTALL)
    text = re.sub(r"\\\[.*?\\\]", " ", text, flags=re.DOTALL)
    text = re.sub(r"\\(?:cite|citep|citet|ref|eqref|label)\{[^{}]*\}", " ", text)
    for _ in range(4):
        text = re.sub(r"\\(?:textbf|textit|emph|mathrm|operatorname)\{([^{}]*)\}", r"\1", text)
    text = re.sub(r"\\[A-Za-z@]+\*?(?:\[[^\]]*\])?", " ", text)
    text = text.replace("~", " ").replace("--", " ")
    text = re.sub(r"[{}_^&]", " ", text)
    return re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*", text)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strict", action="store_true", help="Exit nonzero when a journal limit is exceeded.")
    args = parser.parse_args()

    tex = MAIN.read_text()
    abstract_match = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", tex, re.DOTALL)
    if abstract_match is None:
        raise SystemExit("Abstract not found")
    abstract_words = len(prose_words(abstract_match.group(1)))

    narrative_match = re.search(
        r"\\end\{abstract\}(.*?)\\section\*\{Methods\}",
        tex,
        re.DOTALL,
    )
    if narrative_match is None:
        raise SystemExit("Post-abstract-to-Methods narrative not found")
    main_words = len(prose_words(narrative_match.group(1)))
    main_figures = len(re.findall(r"\\begin\{figure\*?\}", narrative_match.group(1)))
    references = len(re.findall(r"(?m)^@\w+\s*\{", BIB.read_text()))
    introduction_heading = bool(re.search(r"\\section\*?\{Introduction\}", tex))

    checks = [
        ("Abstract", abstract_words, 150, "words"),
        ("Introduction + Results + Discussion", main_words, 4500, "approximate words"),
        ("Main display items", main_figures, 8, "figures"),
        ("References", references, 50, "entries (typical maximum)"),
    ]
    failures = [(name, value, limit) for name, value, limit, _ in checks if value > limit]
    if introduction_heading:
        failures.append(("Introduction heading", 1, 0))

    lines = [
        "# Nature Neuroscience format audit",
        "",
        "Checked against the journal's Article guidance retrieved on 9 August 2026.",
        "The narrative word count is mechanical and excludes LaTeX figure/table environments,",
        "displayed mathematics, citations and cross-references; the submission portal count may differ.",
        "",
        "| Item | Current | Limit | Status |",
        "|---|---:|---:|---|",
    ]
    for name, value, limit, unit in checks:
        status = "PASS" if value <= limit else "NEEDS REVISION"
        lines.append(f"| {name} | {value} {unit} | {limit} | {status} |")
    lines.append(
        "| Introduction heading | "
        f"{'present' if introduction_heading else 'absent'} | absent | "
        f"{'NEEDS REVISION' if introduction_heading else 'PASS'} |"
    )
    lines.extend(
        [
            "",
            "Official guidance:",
            "",
            "- https://www.nature.com/neuro/content",
            "- https://www.nature.com/neuro/submission-guidelines/about/aims",
            "- https://www.nature.com/neuro/submission-guidelines/initial-formatting",
            "- https://www.nature.com/neuro/submission-guidelines/presubmission-enquiries",
            "",
            "The working manuscript uses the completed eight-main-figure layout; the",
            "checkpoint-mechanism and v661 sensitivity displays are Supplementary Figures S9 and S10.",
            "",
        ]
    )
    REPORT.write_text("\n".join(lines))
    print(f"Wrote {REPORT.relative_to(ROOT)}")
    for name, value, limit, unit in checks:
        print(f"{name}: {value} {unit} (limit {limit})")
    return 1 if args.strict and failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
