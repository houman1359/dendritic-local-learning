#!/usr/bin/env python3
"""Document textual continuity between the NeurIPS and journal manuscripts."""

from __future__ import annotations

import argparse
import difflib
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
JOURNAL = ROOT / "main.tex"
NEURIPS = ROOT.parent / "neurips" / "local_credit_assignment_body.tex"
EXTENSION = ROOT / "submission" / "extension_statement.md"
REPORT = ROOT / "analysis" / "NEURIPS_TEXT_REUSE_AUDIT.md"


def prose(text: str) -> str:
    text = re.sub(r"(?m)(?<!\\)%.*$", " ", text)
    text = re.sub(
        r"\\begin\{(?:figure\*?|table\*?|equation|align\*?|gather\*?)\}.*?"
        r"\\end\{(?:figure\*?|table\*?|equation|align\*?|gather\*?)\}",
        " ",
        text,
        flags=re.DOTALL,
    )
    # A display-math opener is a single ``\[``.  Require that the backslash
    # is not itself preceded by another backslash so a LaTeX line break with
    # spacing (for example ``\\[4pt]`` in the author block) cannot consume
    # everything through the next genuine ``\]`` display-math closer.
    text = re.sub(
        r"\$.*?\$|(?<!\\)\\\[.*?\\\]", " ", text, flags=re.DOTALL
    )
    text = re.sub(r"\\(?:cite|citep|citet|ref|eqref|label)\{[^{}]*\}", " ", text)
    for _ in range(5):
        text = re.sub(r"\\[A-Za-z@]+\*?(?:\[[^\]]*\])?\{([^{}]*)\}", r"\1", text)
    text = re.sub(r"\\[A-Za-z@]+\*?(?:\[[^\]]*\])?", " ", text)
    return re.sub(r"\s+", " ", text.replace("~", " ")).strip()


def sentences(text: str) -> list[str]:
    return [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+(?=[A-Z])", prose(text))
        if len(normalize(sentence).split()) >= 12
    ]


def normalize(sentence: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", sentence.lower()))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    missing = [path for path in (JOURNAL, NEURIPS, EXTENSION) if not path.is_file()]
    if missing:
        raise SystemExit("Missing overlap-audit input: " + ", ".join(map(str, missing)))

    journal_text = JOURNAL.read_text(encoding="utf-8")
    neurips_text = NEURIPS.read_text(encoding="utf-8")
    journal_sentences = sentences(journal_text)
    neurips_sentences = sentences(neurips_text)
    neurips_normalized = {normalize(sentence): sentence for sentence in neurips_sentences}
    neurips_tokens = [normalize(sentence).split() for sentence in neurips_sentences]

    exact: list[str] = []
    near: list[tuple[float, str, str]] = []
    for journal_sentence in journal_sentences:
        normalized = normalize(journal_sentence)
        if normalized in neurips_normalized:
            exact.append(journal_sentence)
            continue
        best_score = 0.0
        best_sentence = ""
        journal_tokens = normalized.split()
        for neurips_sentence, candidate_tokens in zip(neurips_sentences, neurips_tokens):
            length_ratio = len(journal_tokens) / max(len(candidate_tokens), 1)
            if length_ratio < 0.55 or length_ratio > 1.8:
                continue
            score = difflib.SequenceMatcher(None, journal_tokens, candidate_tokens).ratio()
            if score > best_score:
                best_score = score
                best_sentence = neurips_sentence
        if best_score >= 0.82:
            near.append((best_score, journal_sentence, best_sentence))

    cited = "safaai2026localcredit" in journal_text
    disclosure_text = EXTENSION.read_text(encoding="utf-8").lower()
    disclosure_mentions_foundation = (
        "complete load-bearing foundation" in disclosure_text
        and "canonical, standalone" in disclosure_text
    )
    lines = [
        "# NeurIPS-to-journal text reuse audit",
        "",
        "This audit documents deliberate continuity with the earlier manuscript. Shared",
        "theory, definitions and framing are reused where precision benefits, while the",
        "Nature Communications Article discloses the foundation and makes its new evidence",
        "chain explicit. Counts are mechanical and are not a legal similarity assessment.",
        "",
        f"- Journal prose sentences checked: {len(journal_sentences)}",
        f"- NeurIPS prose sentences checked: {len(neurips_sentences)}",
        f"- Exact normalized sentence matches: {len(exact)}",
        f"- Near matches at similarity >= 0.82: {len(near)}",
        f"- Earlier work cited in the journal manuscript: {'yes' if cited else 'NO'}",
        f"- Related-work statement identifies the integrated foundation: {'yes' if disclosure_mentions_foundation else 'NO'}",
        "",
        "## Exact matches",
        "",
    ]
    lines.extend(f"- {sentence}" for sentence in exact[:30])
    if not exact:
        lines.append("- None.")
    lines.extend(["", "## Strongest near matches", ""])
    for score, journal_sentence, neurips_sentence in sorted(near, reverse=True)[:20]:
        lines.extend(
            [
                f"- Similarity {score:.3f}",
                f"  - Journal: {journal_sentence}",
                f"  - NeurIPS: {neurips_sentence}",
            ]
        )
    if not near:
        lines.append("- None.")
    lines.extend(
        [
            "",
            "## Editorial interpretation",
            "",
            "The shared conductance equations, exact eligibility--error factorization,",
            "path-gain definition and regular-tree controls are the disclosed foundation.",
            "The journal-specific claim depends on the prospective interventions, real-arbor",
            "route analyses, focal conductance controls, topology--task alignment tests and",
            "external animal-coordinate analysis itemized in `submission/extension_statement.md`.",
            "Re-run this audit against any accepted proceedings version before submission.",
            "",
        ]
    )
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {REPORT.relative_to(ROOT)}")
    print(f"Exact matches: {len(exact)}; near matches: {len(near)}")

    failed = not cited or not disclosure_mentions_foundation
    return 1 if args.strict and failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
