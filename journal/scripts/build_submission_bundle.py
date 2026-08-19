#!/usr/bin/env python3
"""Build a non-destructive Nature Communications initial-submission bundle.

Only an explicit allowlist is copied. The builder refuses to overwrite an
existing directory or archive, rejects unresolved provenance hashes by
default, and scans copied text files for common private absolute paths.
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
import tarfile
import zipfile
from pathlib import Path, PurePosixPath


JOURNAL = Path(__file__).resolve().parents[1]
SUBMISSION = JOURNAL / "submission"

MAIN_FIGURES = (
    "main/figure_01.pdf",
    "main/figure_02.pdf",
    "main/figure_03.pdf",
    "main/figure_04.pdf",
    "main/figure_05.pdf",
    "main/figure_06.pdf",
    "main/figure_07.pdf",
    "main/figure_08.pdf",
)

SUPPLEMENTARY_FIGURES = (
    "supplementary/figure_S01_panels_A-E.pdf",
    "supplementary/figure_S02_panels_A-E.pdf",
    "supplementary/figure_S03_panels_A-D.pdf",
    "supplementary/figure_S04_panels_A-I.pdf",
    "supplementary/figure_S05_panels_A-H.pdf",
    "supplementary/figure_S06_panels_A-D.pdf",
    "supplementary/figure_S07_panels_A-D.pdf",
    "supplementary/figure_S08_panels_A-I.pdf",
    "supplementary/figure_S09_panels_A-D.pdf",
    "supplementary/figure_S10_panels_A-H.pdf",
    "supplementary/figure_S11_panels_A-C.pdf",
    "supplementary/figure_S12_panels_A-J.pdf",
    "supplementary/figure_S13_panels_A-F.pdf",
    "supplementary/figure_S14_panels_A-F.pdf",
    "supplementary/figure_S15_panels_A-D.pdf",
    "supplementary/figure_S16_panels_A-D.pdf",
    "supplementary/figure_S17_panels_A-D.pdf",
    "supplementary/figure_S18_panels_A-K.pdf",
    "supplementary/figure_S19_panels_A-I.pdf",
    "supplementary/figure_S20_panels_A-J.pdf",
    "supplementary/figure_S21_panels_A-I.pdf",
    "supplementary/figure_S22_panels_A-H.pdf",
)

FIGURES = MAIN_FIGURES + SUPPLEMENTARY_FIGURES

REQUIRED_FILES = {
    JOURNAL / "main.tex": Path("main.tex"),
    JOURNAL / "main.pdf": Path("main.pdf"),
    JOURNAL / "main_with_supplementary.pdf": Path("main_with_supplementary.pdf"),
    JOURNAL / "main.bbl": Path("main.bbl"),
    JOURNAL / "OVERLEAF_README.md": Path("OVERLEAF_README.md"),
    JOURNAL / "figures" / "README.md": Path("figures/README.md"),
    JOURNAL / "references.bib": Path("references.bib"),
    JOURNAL / "supplementary" / "supplementary.tex": Path("supplementary/supplementary.tex"),
    JOURNAL / "supplementary" / "supplementary.pdf": Path("supplementary/supplementary.pdf"),
    JOURNAL / "supplementary" / "supplementary.bbl": Path("supplementary/supplementary.bbl"),
    SUBMISSION / "Source_Data.zip": Path("Source_Data.zip"),
    SUBMISSION / "cover_letter.md": Path("submission_materials/cover_letter.md"),
    SUBMISSION / "editorial_summary.md": Path("submission_materials/editorial_summary.md"),
    SUBMISSION / "README.md": Path("submission_materials/README.md"),
    SUBMISSION / "extension_statement.md": Path("submission_materials/extension_statement.md"),
    SUBMISSION / "reporting_checklist.md": Path("submission_materials/reporting_checklist.md"),
    SUBMISSION / "AUTHOR_ACTIONS.md": Path("submission_materials/AUTHOR_ACTIONS.md"),
    SUBMISSION / "OFFICIAL_FORMS_REQUIRED.md": Path("submission_materials/OFFICIAL_FORMS_REQUIRED.md"),
    JOURNAL / "source_data" / "provenance_manifest.tsv": Path("manifests/source_data_provenance_manifest.tsv"),
    JOURNAL / "reproducibility" / "origin_manifest.tsv": Path("manifests/reproducibility_origin_manifest.tsv"),
}

SOFTWARE_CANDIDATES = (
    SUBMISSION / "Dendritic_credit_assignment_software.zip",
    SUBMISSION / "Software.zip",
    SUBMISSION / "Software_Archive.zip",
    SUBMISSION / "software_release.zip",
    SUBMISSION / "Software.tar.gz",
    SUBMISSION / "software_release.tar.gz",
)

TEXT_SUFFIXES = {".tex", ".bib", ".bbl", ".md", ".tsv", ".csv", ".json", ".txt"}
PRIVATE_PATH_PATTERNS = (
    re.compile(rb"/n/(?:home[^/]*/|holylabs/)", re.IGNORECASE),
    re.compile(rb"/Users/[A-Za-z0-9._-]+/"),
    re.compile(rb"file:///", re.IGNORECASE),
)
GRAPHIC_PATTERN = re.compile(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def repository_commit() -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(JOURNAL), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "not-recorded"


def verify_provenance(path: Path, allow_pending: bool) -> None:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if not rows:
        raise RuntimeError(f"Provenance manifest is empty: {path}")
    unresolved = [
        row.get("entry_id", "<unknown>")
        for row in rows
        if re.fullmatch(r"[0-9a-fA-F]{64}", row.get("sha256", "").strip()) is None
    ]
    if unresolved and not allow_pending:
        preview = ", ".join(unresolved[:8])
        suffix = " ..." if len(unresolved) > 8 else ""
        raise RuntimeError(
            f"Provenance contains {len(unresolved)} unresolved SHA-256 values "
            f"({preview}{suffix}). Refresh hashes before building the final bundle."
        )


def verify_figure_allowlist() -> None:
    """Require the source bundle to include every rendered manuscript figure."""
    referenced: set[str] = set()
    for manuscript in (
        JOURNAL / "main.tex",
        JOURNAL / "supplementary" / "supplementary.tex",
    ):
        text = manuscript.read_text(encoding="utf-8")
        for match in GRAPHIC_PATTERN.finditer(text):
            name = Path(match.group(1)).as_posix()
            if not Path(name).suffix:
                name += ".pdf"
            referenced.add(name)
    allowed = set(FIGURES)
    if referenced != allowed:
        missing = sorted(referenced - allowed)
        stale = sorted(allowed - referenced)
        details = []
        if missing:
            details.append("missing from allowlist: " + ", ".join(missing))
        if stale:
            details.append("not referenced by either manuscript: " + ", ".join(stale))
        raise RuntimeError("Figure allowlist is out of sync (" + "; ".join(details) + ")")


def verify_archive(path: Path) -> None:
    lower = path.name.lower()
    if lower.endswith(".zip"):
        with zipfile.ZipFile(path) as archive:
            for member in archive.infolist():
                name = PurePosixPath(member.filename)
                if name.is_absolute() or ".." in name.parts:
                    raise RuntimeError(f"Unsafe member {member.filename!r} in {path}")
            bad = archive.testzip()
        if bad is not None:
            raise RuntimeError(f"Corrupt member {bad!r} in {path}")
    elif lower.endswith(".tar.gz") or lower.endswith(".tgz"):
        with tarfile.open(path, "r:gz") as archive:
            for member in archive.getmembers():
                name = Path(member.name)
                if name.is_absolute() or ".." in name.parts:
                    raise RuntimeError(f"Unsafe member {member.name!r} in {path}")
    else:
        raise RuntimeError(f"Unsupported archive format: {path}")


def private_path_labels(data: bytes) -> list[str]:
    return [pattern.pattern.decode("ascii") for pattern in PRIVATE_PATH_PATTERNS if pattern.search(data)]


def scan_archive_private_paths(path: Path) -> None:
    """Inspect text-like members without recursively unpacking nested archives."""
    failures: list[str] = []
    lower = path.name.lower()
    if lower.endswith(".zip"):
        with zipfile.ZipFile(path) as archive:
            members = ((name, archive.read(name)) for name in archive.namelist() if not name.endswith("/"))
            for name, data in members:
                if Path(name).suffix.lower() not in TEXT_SUFFIXES:
                    continue
                labels = private_path_labels(data)
                if labels:
                    failures.append(f"{name}: {', '.join(labels)}")
    elif lower.endswith(".tar.gz") or lower.endswith(".tgz"):
        with tarfile.open(path, "r:gz") as archive:
            for member in archive.getmembers():
                if not member.isfile() or Path(member.name).suffix.lower() not in TEXT_SUFFIXES:
                    continue
                handle = archive.extractfile(member)
                data = b"" if handle is None else handle.read()
                labels = private_path_labels(data)
                if labels:
                    failures.append(f"{member.name}: {', '.join(labels)}")
    if failures:
        raise RuntimeError(
            f"Private absolute path scan failed inside {path}:\n- " + "\n- ".join(failures)
        )


def choose_software_archive(explicit: Path | None) -> Path | None:
    if explicit is None:
        return None
    path = explicit.expanduser().resolve()
    try:
        path.relative_to(JOURNAL)
    except ValueError as exc:
        raise RuntimeError("The software archive must be inside the journal project") from exc
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def copy_allowlist(output_dir: Path, software: Path | None) -> list[tuple[Path, Path]]:
    sources = dict(REQUIRED_FILES)
    for name in FIGURES:
        sources[JOURNAL / "figures" / name] = Path("figures") / name
    if software is not None:
        destination_name = "Software.zip" if software.name.lower().endswith(".zip") else "Software.tar.gz"
        sources[software] = Path(destination_name)

    missing = [str(path.relative_to(JOURNAL)) for path in sources if not path.is_file()]
    if missing:
        raise FileNotFoundError("Required bundle inputs are missing:\n- " + "\n- ".join(missing))

    copied: list[tuple[Path, Path]] = []
    for source, relative_destination in sorted(sources.items(), key=lambda item: str(item[1])):
        destination = output_dir / relative_destination
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        copied.append((source, relative_destination))
    return copied


def scan_private_paths(output_dir: Path, relative_paths: list[Path]) -> None:
    failures: list[str] = []
    for relative in relative_paths:
        path = output_dir / relative
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        labels = private_path_labels(path.read_bytes())
        if labels:
            failures.append(f"{relative}: {', '.join(labels)}")
    if failures:
        raise RuntimeError("Private absolute path scan failed:\n- " + "\n- ".join(failures))


def write_metadata(output_dir: Path, software: Path | None) -> None:
    worktree_clean = not subprocess.run(
        ["git", "-C", str(JOURNAL), "status", "--porcelain=v1"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    metadata = {
        "target_journal": "Nature Communications",
        "article_title": "When dendritic structure helps local credit assignment",
        "source_commit": repository_commit(),
        "source_worktree_clean": worktree_clean,
        "source_snapshot_method": "current explicit allowlist",
        "numbered_main_figure_count": 8,
        "main_figure_asset_count": len(MAIN_FIGURES),
        "supplementary_figure_count": len(SUPPLEMENTARY_FIGURES),
        "source_data_archive": "Source_Data.zip",
        "combined_reading_copy": "main_with_supplementary.pdf",
        "software_archive": None if software is None else ("Software.zip" if software.name.lower().endswith(".zip") else "Software.tar.gz"),
        "build_policy": "explicit allowlist; no overwrite; deterministic ZIP metadata",
    }
    destination = output_dir / "manifests" / "bundle_metadata.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_manifest(output_dir: Path) -> None:
    manifest = output_dir / "manifests" / "bundle_manifest.tsv"
    payload = sorted(
        path for path in output_dir.rglob("*")
        if path.is_file() and path != manifest
    )
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["bundle_path", "bytes", "sha256"])
        for path in payload:
            writer.writerow([path.relative_to(output_dir).as_posix(), path.stat().st_size, sha256(path)])


def write_readme(output_dir: Path, software: Path | None) -> None:
    software_line = (
        "The selected reviewer software archive is included."
        if software is not None
        else "No software archive was present; create it before treating this as a final upload bundle."
    )
    text = f"""# Nature Communications initial-submission bundle

Article: *When dendritic structure helps local credit assignment*

This directory was assembled from an explicit allowlist by `scripts/build_submission_bundle.py`. It contains the compiled and source manuscripts, a combined main-plus-supplementary reading copy, {len(MAIN_FIGURES)} main figure assets across eight numbered figures, {len(SUPPLEMENTARY_FIGURES)} supplementary figure PDFs, references, Source Data, submission documents and provenance manifests. {software_line}

`main_with_supplementary.pdf` contains the complete Article followed by the Supplementary Information. The separate `main.pdf` and `supplementary/supplementary.pdf` files are retained because the journal portal may request separate uploads.

`manifests/bundle_manifest.tsv` records the byte size and SHA-256 digest of every other file in this directory. The manifest does not hash itself. `manifests/bundle_metadata.json` records the source commit when available.

The files in `submission_materials/` include internal author checklists. Upload only the items requested by the journal portal. Official interactive reporting forms must be downloaded fresh and completed separately; see `submission_materials/OFFICIAL_FORMS_REQUIRED.md`.

The bundle builder excludes auxiliary LaTeX files, logs, scheduler scripts, local caches, checkpoints and arbitrary analysis reports. It also scans copied text for common private absolute paths. Passing these automated checks does not replace author review of disclosures, licensing, confidentiality or concurrent-submission status.
"""
    (output_dir / "README.md").write_text(text, encoding="utf-8")


def deterministic_zip(source_dir: Path, destination: Path) -> None:
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in sorted(p for p in source_dir.rglob("*") if p.is_file()):
            relative = Path(source_dir.name) / path.relative_to(source_dir)
            info = zipfile.ZipInfo(relative.as_posix(), date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, path.read_bytes(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
    verify_archive(destination)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SUBMISSION / "nature_communications_bundle",
        help="New bundle directory; must not already exist.",
    )
    parser.add_argument(
        "--archive",
        type=Path,
        default=SUBMISSION / "Nature_Communications_Submission.zip",
        help="New deterministic ZIP archive; must not already exist.",
    )
    parser.add_argument(
        "--software-archive",
        type=Path,
        default=None,
        help="Optional existing software ZIP or tar.gz inside this journal project.",
    )
    parser.add_argument(
        "--allow-pending-provenance",
        action="store_true",
        help="Development-only override; never use for a final submission bundle.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace only the default generated Nature Communications outputs.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    archive = args.archive.expanduser().resolve()
    digest_file = archive.with_suffix(archive.suffix + ".sha256")
    if args.force:
        default_dir = (SUBMISSION / "nature_communications_bundle").resolve()
        default_archive = (SUBMISSION / "Nature_Communications_Submission.zip").resolve()
        if output_dir != default_dir or archive != default_archive:
            raise RuntimeError("--force is restricted to the default generated outputs")
        if output_dir.exists():
            if not output_dir.is_dir():
                raise RuntimeError(f"Expected generated directory at {output_dir}")
            shutil.rmtree(output_dir)
        for path in (archive, digest_file):
            if path.exists():
                if not path.is_file():
                    raise RuntimeError(f"Expected generated file at {path}")
                path.unlink()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {output_dir}")
    if archive.exists():
        raise FileExistsError(f"Refusing to overwrite existing archive: {archive}")
    if digest_file.exists():
        raise FileExistsError(f"Refusing to overwrite existing digest: {digest_file}")
    if output_dir == JOURNAL or JOURNAL in output_dir.parents and output_dir.name in {"figures", "source_data", "submission"}:
        raise RuntimeError(f"Unsafe output directory: {output_dir}")

    provenance = JOURNAL / "source_data" / "provenance_manifest.tsv"
    verify_figure_allowlist()
    verify_provenance(provenance, args.allow_pending_provenance)
    verify_archive(SUBMISSION / "Source_Data.zip")
    scan_archive_private_paths(SUBMISSION / "Source_Data.zip")
    software = choose_software_archive(args.software_archive)
    if software is not None:
        verify_archive(software)
        scan_archive_private_paths(software)

    output_dir.mkdir(parents=True)
    try:
        copied = copy_allowlist(output_dir, software)
        write_metadata(output_dir, software)
        write_readme(output_dir, software)
        relative_paths = [relative for _, relative in copied]
        relative_paths.extend([Path("README.md"), Path("manifests/bundle_metadata.json")])
        scan_private_paths(output_dir, relative_paths)
        write_manifest(output_dir)
        deterministic_zip(output_dir, archive)
        digest_file.write_text(f"{sha256(archive)}  {archive.name}\n", encoding="utf-8")
    except Exception:
        # Preserve partial output for diagnosis; never delete or overwrite user data.
        raise

    print(f"Bundle directory: {output_dir}")
    print(f"Bundle archive:   {archive}")
    print(f"Archive SHA-256:  {sha256(archive)}")
    print(f"Digest file:     {digest_file}")
    if software is None:
        print("WARNING: no software archive was found or included", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
