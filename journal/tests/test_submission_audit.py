from __future__ import annotations

import csv
import hashlib
import importlib.util
import sys
from pathlib import Path


JOURNAL_ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "audit_submission", JOURNAL_ROOT / "scripts" / "audit_submission.py"
)
assert SPEC is not None and SPEC.loader is not None
audit = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = audit
SPEC.loader.exec_module(audit)


FIELDS = [
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
]


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def _ready_row(source: Path, **updates: str) -> dict[str, str]:
    row = {
        "entry_id": "fig1.a",
        "record_type": "panel_source",
        "figure": "fig1",
        "panel": "a",
        "status": "ready",
        "source_path": str(source),
        "sha256": _digest(source),
        "generator_path": "",
        "replication_unit": "test fixture",
        "notes": "",
    }
    row.update(updates)
    return row


def test_strip_tex_comments_preserves_escaped_percent() -> None:
    text = "kept \\% value % removed\nnext\n"
    assert audit.strip_tex_comments(text) == "kept \\% value \nnext\n"


def test_clean_fixture_passes_without_font_inspection(tmp_path: Path) -> None:
    source = tmp_path / "source.csv"
    source.write_text("x,y\n1,2\n", encoding="utf-8")
    graphic = tmp_path / "figure.png"
    graphic.write_bytes(b"not-decoded-when-font-check-is-disabled")
    manifest = tmp_path / "manifest.tsv"
    _write_manifest(
        manifest,
        [
            _ready_row(source),
            _ready_row(
                graphic,
                entry_id="fig1.asset",
                record_type="figure_asset",
                panel="all",
            ),
        ],
    )
    manuscript = tmp_path / "main.tex"
    manuscript.write_text(
        "\\includegraphics{" + str(graphic) + "}\n"
        "See Fig.~\\ref{fig:test1}.\n"
        "\\label{fig:test1}\n",
        encoding="utf-8",
    )

    findings, counts = audit.run_audit(
        manifest, manuscript, strict_pending=False, check_fonts=False
    )
    assert counts["errors"] == 0, findings
    assert counts["graphics_resolved"] == 1


def test_rendered_graphic_must_match_figure_asset(tmp_path: Path) -> None:
    source = tmp_path / "source.csv"
    source.write_text("x,y\n1,2\n", encoding="utf-8")
    registered = tmp_path / "registered.png"
    registered.write_bytes(b"registered")
    rendered = tmp_path / "rendered.png"
    rendered.write_bytes(b"different")
    manifest = tmp_path / "manifest.tsv"
    _write_manifest(
        manifest,
        [
            _ready_row(source),
            _ready_row(
                registered,
                entry_id="fig1.asset",
                record_type="figure_asset",
                panel="all",
            ),
        ],
    )
    manuscript = tmp_path / "main.tex"
    manuscript.write_text(
        "\\includegraphics{" + str(rendered) + "}\n"
        "See Fig.~\\ref{fig:test1}.\n"
        "\\label{fig:test1}\n",
        encoding="utf-8",
    )

    findings, _ = audit.run_audit(
        manifest, manuscript, strict_pending=False, check_fonts=False
    )
    assert "provenance.asset_mismatch" in {finding.code for finding in findings}


def test_continued_float_reuses_number_for_asset_provenance(tmp_path: Path) -> None:
    source = tmp_path / "source.csv"
    source.write_text("x,y\n1,2\n", encoding="utf-8")
    first = tmp_path / "first.png"
    first.write_bytes(b"first")
    continuation = tmp_path / "continuation.png"
    continuation.write_bytes(b"continuation")
    manifest = tmp_path / "manifest.tsv"
    _write_manifest(
        manifest,
        [
            _ready_row(source),
            _ready_row(
                first,
                entry_id="fig1.asset.first",
                record_type="figure_asset",
                panel="a",
            ),
            _ready_row(
                continuation,
                entry_id="fig1.asset.continuation",
                record_type="figure_asset",
                panel="b",
            ),
        ],
    )
    manuscript = tmp_path / "main.tex"
    manuscript.write_text(
        "\\begin{figure}\\includegraphics{" + str(first) + "}"
        "\\label{fig:first}\\end{figure}\n"
        "\\begin{figure}\\ContinuedFloat\\includegraphics{" + str(continuation) + "}"
        "\\label{fig:continued}\\end{figure}\n"
        "See Fig.~\\ref{fig:first} and Fig.~\\ref{fig:continued}.\n",
        encoding="utf-8",
    )

    findings, _ = audit.run_audit(
        manifest, manuscript, strict_pending=False, check_fonts=False
    )
    assert not [
        finding
        for finding in findings
        if finding.code in {"provenance.figure_missing", "provenance.asset_mismatch"}
    ]


def test_hash_change_and_withdrawn_metric_are_errors(tmp_path: Path) -> None:
    source = tmp_path / "source.csv"
    source.write_text("before\n", encoding="utf-8")
    row = _ready_row(source)
    manifest = tmp_path / "manifest.tsv"
    _write_manifest(manifest, [row])
    source.write_text("after\n", encoding="utf-8")
    manuscript = tmp_path / "main.tex"
    manuscript.write_text(
        "See Fig.~\\ref{fig:test1}; value 0.817.\n\\label{fig:test1}\n",
        encoding="utf-8",
    )

    findings, _ = audit.run_audit(manifest, manuscript, check_fonts=False)
    codes = {finding.code for finding in findings}
    assert "source.hash_mismatch" in codes
    assert "metric.withdrawn" in codes


def test_pending_changes_from_warning_to_error_in_strict_mode(tmp_path: Path) -> None:
    source = tmp_path / "source.csv"
    source.write_text("x\n", encoding="utf-8")
    row = _ready_row(
        source,
        entry_id="fig1.asset",
        record_type="figure_asset",
        status="pending_rerender",
        source_path="",
        sha256="",
    )
    manifest = tmp_path / "manifest.tsv"
    _write_manifest(manifest, [row])

    rows, schema_findings = audit.load_manifest(manifest)
    assert not schema_findings
    development = audit.audit_manifest_rows(rows, strict_pending=False)
    release = audit.audit_manifest_rows(rows, strict_pending=True)
    assert [finding.severity for finding in development] == ["warning"]
    assert [finding.severity for finding in release] == ["error"]


def test_repository_manifest_has_no_missing_or_changed_ready_sources() -> None:
    rows, findings = audit.load_manifest(audit.DEFAULT_MANIFEST)
    findings.extend(audit.audit_manifest_rows(rows, strict_pending=False))
    fatal_codes = {
        "manifest.columns",
        "manifest.duplicate",
        "manifest.entry_id",
        "manifest.record_type",
        "manifest.status",
        "source.unspecified",
        "source.missing",
        "source.hash_missing",
        "source.hash_mismatch",
        "generator.missing",
    }
    failures = [finding for finding in findings if finding.code in fatal_codes]
    assert not failures
