#!/usr/bin/env python3
"""Audit figure provenance and submission-critical manuscript invariants.

The audit is deliberately independent of the plotting code.  It verifies the
frozen panel-source manifest, resolves every ``\\includegraphics`` call in the
manuscript, checks that every numbered figure has provenance entries, scans
manuscript text for withdrawn metrics, and inspects PDF fonts when ``pdffonts``
is available.

Run from any directory::

    python scripts/audit_submission.py
    python scripts/audit_submission.py --strict-pending --json

The first form treats explicitly declared pending work as a warning.  The
second is the pre-submission gate and treats it as an error.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence


SCRIPT_PATH = Path(__file__).resolve()
JOURNAL_ROOT = SCRIPT_PATH.parent.parent
PROJECT_ROOT = JOURNAL_ROOT.parent
LEGACY_PROJECT_PREFIX = Path("drafts/dendritic-local-learning")
DEFAULT_MANIFEST = JOURNAL_ROOT / "source_data" / "provenance_manifest.tsv"
DEFAULT_MANUSCRIPT = JOURNAL_ROOT / "main.tex"

REQUIRED_MANIFEST_FIELDS = {
    "entry_id",
    "record_type",
    "figure",
    "panel",
    "status",
    "source_path",
    "sha256",
    "generator_path",
    "replication_unit",
    "notes",
}
READY_STATUSES = {"ready"}
PENDING_STATUSES = {"pending", "pending_rerender", "pending_analysis"}
ALLOWED_STATUSES = READY_STATUSES | PENDING_STATUSES
ALLOWED_RECORD_TYPES = {"panel_source", "figure_asset", "generator_snapshot"}

# These values arose from pooled or otherwise invalid scopes in an earlier
# analysis and must not re-enter a manuscript or submission document.
FORBIDDEN_METRICS = (
    "0.817",
    "0.536",
    "0.297",
    "0.364",
    "0.901",
    "0.934",
)

INCLUDEGRAPHICS_RE = re.compile(
    r"\\includegraphics\s*(?:\[[^\]]*\])?\s*\{([^}]+)\}", re.MULTILINE
)
FIGURE_LABEL_RE = re.compile(r"\\label\{fig:([^}]+)\}")
FIGURE_REF_RE = re.compile(r"\\(?:auto|page|eq)?ref\{fig:([^}]+)\}")
IF_FILE_EXISTS_RE = re.compile(r"\\IfFileExists\s*\{([^}]+)\}")
FIGURE_ENV_RE = re.compile(
    r"\\begin\{figure\}(?:\[[^]]*\])?(?P<body>.*?)\\end\{figure\}",
    re.DOTALL,
)


@dataclass(frozen=True)
class Finding:
    severity: str
    code: str
    message: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def strip_tex_comments(text: str) -> str:
    """Remove unescaped TeX comments while preserving line boundaries."""

    cleaned: list[str] = []
    for line in text.splitlines(keepends=True):
        comment_at = None
        for index, character in enumerate(line):
            if character != "%":
                continue
            preceding = 0
            cursor = index - 1
            while cursor >= 0 and line[cursor] == "\\":
                preceding += 1
                cursor -= 1
            if preceding % 2 == 0:
                comment_at = index
                break
        if comment_at is None:
            cleaned.append(line)
        elif line.endswith("\n"):
            cleaned.append(line[:comment_at] + "\n")
        else:
            cleaned.append(line[:comment_at])
    return "".join(cleaned)


def load_manifest(path: Path) -> tuple[list[dict[str, str]], list[Finding]]:
    findings: list[Finding] = []
    if not path.is_file():
        return [], [Finding("error", "manifest.missing", f"Missing manifest: {path}")]

    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        fields = set(reader.fieldnames or [])
        missing = sorted(REQUIRED_MANIFEST_FIELDS - fields)
        if missing:
            findings.append(
                Finding(
                    "error",
                    "manifest.columns",
                    "Manifest is missing columns: " + ", ".join(missing),
                )
            )
        rows = [dict(row) for row in reader]

    seen: set[str] = set()
    for row_number, row in enumerate(rows, start=2):
        entry_id = row.get("entry_id", "").strip()
        if not entry_id:
            findings.append(Finding("error", "manifest.entry_id", f"Row {row_number} has no entry_id"))
        elif entry_id in seen:
            findings.append(
                Finding("error", "manifest.duplicate", f"Duplicate entry_id {entry_id!r}")
            )
        seen.add(entry_id)

        status = row.get("status", "").strip()
        if status not in ALLOWED_STATUSES:
            findings.append(
                Finding(
                    "error",
                    "manifest.status",
                    f"{entry_id or 'row ' + str(row_number)} has invalid status {status!r}",
                )
            )
        record_type = row.get("record_type", "").strip()
        if record_type not in ALLOWED_RECORD_TYPES:
            findings.append(
                Finding(
                    "error",
                    "manifest.record_type",
                    f"{entry_id or 'row ' + str(row_number)} has invalid record_type {record_type!r}",
                )
            )
    return rows, findings


def resolve_repo_path(raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    # The manifest deliberately records repository-relative provenance paths
    # from the enclosing dendritic-modeling workspace.  Resolve those paths
    # equally in the canonical checkout and in a standalone/recovery worktree.
    try:
        relative = path.relative_to(LEGACY_PROJECT_PREFIX)
    except ValueError:
        relative = path
    return PROJECT_ROOT / relative


def audit_manifest_rows(
    rows: Sequence[dict[str, str]], *, strict_pending: bool
) -> list[Finding]:
    findings: list[Finding] = []
    for row in rows:
        entry_id = row.get("entry_id", "<unnamed>").strip() or "<unnamed>"
        status = row.get("status", "").strip()
        source_raw = row.get("source_path", "").strip()
        expected_hash = row.get("sha256", "").strip().lower()
        generator_raw = row.get("generator_path", "").strip()

        if status in PENDING_STATUSES:
            findings.append(
                Finding(
                    "error" if strict_pending else "warning",
                    "manifest.pending",
                    f"{entry_id}: {status}; {row.get('notes', '').strip()}",
                )
            )
            # Pending rows are declarations of unfinished work.  An existing
            # placeholder must not accidentally make them look complete.
            continue

        if not source_raw:
            findings.append(
                Finding("error", "source.unspecified", f"{entry_id}: ready row has no source_path")
            )
            continue
        source = resolve_repo_path(source_raw)
        if not source.is_file():
            findings.append(Finding("error", "source.missing", f"{entry_id}: missing {source_raw}"))
        elif not expected_hash:
            findings.append(
                Finding("error", "source.hash_missing", f"{entry_id}: ready source has no SHA256")
            )
        else:
            observed_hash = sha256_file(source)
            if observed_hash != expected_hash:
                findings.append(
                    Finding(
                        "error",
                        "source.hash_mismatch",
                        f"{entry_id}: {source_raw} changed; expected {expected_hash}, observed {observed_hash}",
                    )
                )

        if generator_raw and not resolve_repo_path(generator_raw).is_file():
            findings.append(
                Finding("error", "generator.missing", f"{entry_id}: missing {generator_raw}")
            )
    return findings


def graphic_candidates(raw_path: str, manuscript: Path) -> Iterable[Path]:
    path = Path(raw_path)
    bases = (manuscript.parent, JOURNAL_ROOT / "figures")
    extensions = ("", ".pdf", ".png", ".jpg", ".jpeg", ".eps") if not path.suffix else ("",)
    seen: set[Path] = set()
    for base in bases:
        candidate_base = path if path.is_absolute() else base / path
        for extension in extensions:
            candidate = Path(str(candidate_base) + extension)
            if candidate not in seen:
                seen.add(candidate)
                yield candidate


def resolve_graphic(raw_path: str, manuscript: Path) -> Path | None:
    return next((path for path in graphic_candidates(raw_path, manuscript) if path.is_file()), None)


def audit_manuscript(
    manuscript: Path, rows: Sequence[dict[str, str]]
) -> tuple[list[Finding], list[Path]]:
    if not manuscript.is_file():
        return [Finding("error", "manuscript.missing", f"Missing manuscript: {manuscript}")], []

    text = strip_tex_comments(manuscript.read_text(encoding="utf-8"))
    findings: list[Finding] = []
    graphics: list[Path] = []
    conditional_graphics = {Path(path).name for path in IF_FILE_EXISTS_RE.findall(text)}
    for raw_path in INCLUDEGRAPHICS_RE.findall(text):
        resolved = resolve_graphic(raw_path.strip(), manuscript)
        if resolved is None:
            findings.append(
                Finding(
                    "warning" if Path(raw_path.strip()).name in conditional_graphics else "error",
                    "figure.conditional_missing"
                    if Path(raw_path.strip()).name in conditional_graphics
                    else "figure.missing",
                    f"Cannot resolve \\includegraphics{{{raw_path.strip()}}}",
                )
            )
        elif resolved not in graphics:
            graphics.append(resolved)

    figure_environments: list[tuple[int, str, str | None]] = []
    figure_number = 0
    for match in FIGURE_ENV_RE.finditer(text):
        body = match.group("body")
        if r"\ContinuedFloat" not in body:
            figure_number += 1
        labels_in_body = FIGURE_LABEL_RE.findall(body)
        if len(labels_in_body) != 1:
            findings.append(
                Finding(
                    "error",
                    "figure.label_count",
                    f"Figure environment {figure_number} has {len(labels_in_body)} labels",
                )
            )
            continue
        graphics_in_body = INCLUDEGRAPHICS_RE.findall(body)
        figure_environments.append(
            (figure_number, labels_in_body[0], graphics_in_body[0].strip() if graphics_in_body else None)
        )
    # Small release fixtures and legacy fragments can contain a labelled
    # graphic without a surrounding figure environment.  Retain provenance
    # checking for those inputs; the production manuscript takes the stricter
    # environment-aware path above (including \ContinuedFloat numbering).
    if not figure_environments:
        orphan_labels = FIGURE_LABEL_RE.findall(text)
        orphan_graphics = [path.strip() for path in INCLUDEGRAPHICS_RE.findall(text)]
        for index, label in enumerate(orphan_labels, start=1):
            raw_graphic = orphan_graphics[index - 1] if index <= len(orphan_graphics) else None
            figure_environments.append((index, label, raw_graphic))
    labels = [label for _, label, _ in figure_environments]
    references = FIGURE_REF_RE.findall(text)
    supplementary_numbering = (
        manuscript.name.lower().startswith("supplementary")
        or manuscript.parent.name.lower() == "supplementary"
    )
    for figure_number, label, _ in figure_environments:
        literal_reference = re.search(
            rf"(?:Fig\.?|Figure)~?\s*{figure_number}\b", text, re.IGNORECASE
        )
        # Supplementary figures are normally cited from the separate main
        # manuscript, so absence of an internal self-citation is not a defect.
        if not supplementary_numbering and label not in references and literal_reference is None:
            findings.append(
                Finding("warning", "figure.unreferenced", f"Figure label fig:{label} is never cited")
            )

    manifest_figures = {row.get("figure", "").strip() for row in rows}
    asset_paths_by_figure: dict[str, set[Path]] = {}
    for row in rows:
        if row.get("record_type", "").strip() != "figure_asset":
            continue
        figure = row.get("figure", "").strip()
        source = row.get("source_path", "").strip()
        if figure and source:
            asset_paths_by_figure.setdefault(figure, set()).add(
                resolve_repo_path(source).resolve()
            )
    for figure_number, label, raw_graphic in figure_environments:
        expected = f"figS{figure_number}" if supplementary_numbering else f"fig{figure_number}"
        if expected not in manifest_figures:
            findings.append(
                Finding(
                    "error",
                    "provenance.figure_missing",
                    f"Figure fig:{label} has no {expected} entry in the provenance manifest",
                )
            )
        elif raw_graphic is not None:
            resolved_graphic = resolve_graphic(raw_graphic, manuscript)
            if resolved_graphic is None:
                continue
            rendered = resolved_graphic.resolve()
            if rendered not in asset_paths_by_figure.get(expected, set()):
                findings.append(
                    Finding(
                        "error",
                        "provenance.asset_mismatch",
                        f"Figure fig:{label} resolves to {rendered.name}, but {expected} "
                        "has no matching figure_asset row",
                    )
                )

    for metric in FORBIDDEN_METRICS:
        for match in re.finditer(re.escape(metric), text):
            line_number = text.count("\n", 0, match.start()) + 1
            findings.append(
                Finding(
                    "error",
                    "metric.withdrawn",
                    f"Withdrawn value {metric} occurs in {manuscript.name}:{line_number}",
                )
            )
    return findings, graphics


def audit_pdf_fonts(pdf_paths: Sequence[Path]) -> list[Finding]:
    pdffonts = shutil.which("pdffonts")
    if pdffonts is None:
        return [
            Finding(
                "warning",
                "pdf.pdffonts_unavailable",
                "pdffonts is unavailable; PDF font embedding was not checked",
            )
        ]

    findings: list[Finding] = []
    for pdf in pdf_paths:
        if pdf.suffix.lower() != ".pdf" or not pdf.is_file():
            continue
        completed = subprocess.run(
            [pdffonts, str(pdf)], capture_output=True, text=True, check=False
        )
        if completed.returncode:
            findings.append(
                Finding(
                    "error",
                    "pdf.inspect_failed",
                    f"pdffonts failed for {pdf}: {completed.stderr.strip()}",
                )
            )
            continue
        for line in completed.stdout.splitlines()[2:]:
            if not line.strip():
                continue
            if re.search(r"\bType\s+3\b", line):
                findings.append(
                    Finding("error", "pdf.type3", f"Type 3 font in {pdf.relative_to(JOURNAL_ROOT)}")
                )
                break
        if any(
            match.group(1) == "no"
            for line in completed.stdout.splitlines()[2:]
            if (match := re.search(r"\s(yes|no)\s+(?:yes|no)\s+(?:yes|no)\s+\d+\s+\d+\s*$", line))
        ):
            findings.append(
                Finding("error", "pdf.font_unembedded", f"Unembedded font in {pdf.relative_to(JOURNAL_ROOT)}")
            )
    return findings


def run_audit(
    manifest: Path = DEFAULT_MANIFEST,
    manuscript: Path = DEFAULT_MANUSCRIPT,
    *,
    strict_pending: bool = False,
    check_fonts: bool = True,
) -> tuple[list[Finding], dict[str, int]]:
    rows, findings = load_manifest(manifest)
    findings.extend(audit_manifest_rows(rows, strict_pending=strict_pending))
    manuscript_findings, graphics = audit_manuscript(manuscript, rows)
    findings.extend(manuscript_findings)
    if check_fonts:
        findings.extend(audit_pdf_fonts(graphics))
    counts = {
        "manifest_entries": len(rows),
        "graphics_resolved": len(graphics),
        "errors": sum(item.severity == "error" for item in findings),
        "warnings": sum(item.severity == "warning" for item in findings),
    }
    return findings, counts


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--manuscript", type=Path, default=DEFAULT_MANUSCRIPT)
    parser.add_argument(
        "--strict-pending",
        action="store_true",
        help="Treat declared pending panels or renders as submission-blocking errors.",
    )
    parser.add_argument("--skip-fonts", action="store_true")
    parser.add_argument("--json", action="store_true", dest="as_json")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    findings, counts = run_audit(
        args.manifest.resolve(),
        args.manuscript.resolve(),
        strict_pending=args.strict_pending,
        check_fonts=not args.skip_fonts,
    )
    if args.as_json:
        print(json.dumps({"summary": counts, "findings": [asdict(item) for item in findings]}, indent=2))
    else:
        for finding in findings:
            print(f"{finding.severity.upper():7s} {finding.code}: {finding.message}")
        print(
            "Audit summary: "
            f"{counts['manifest_entries']} manifest entries, "
            f"{counts['graphics_resolved']} graphics, "
            f"{counts['errors']} errors, {counts['warnings']} warnings"
        )
    return 1 if counts["errors"] else 0


if __name__ == "__main__":
    sys.exit(main())
