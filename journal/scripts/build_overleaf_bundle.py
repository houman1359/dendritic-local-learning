#!/usr/bin/env python3
"""Build the minimal, canonical Overleaf source project."""

from __future__ import annotations

import argparse
import hashlib
import shutil
import zipfile
from pathlib import Path

from build_submission_bundle import (
    MAIN_FIGURES,
    SUPPLEMENTARY_FIGURES,
    verify_figure_allowlist,
)


ROOT = Path(__file__).resolve().parents[1]
SUBMISSION = ROOT / "submission"
DEFAULT_STAGE = SUBMISSION / "overleaf_project"
DEFAULT_ZIP = SUBMISSION / "Overleaf_Project.zip"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inputs() -> dict[Path, Path]:
    files = {
        ROOT / "main.tex": Path("main.tex"),
        ROOT / "main.bbl": Path("main.bbl"),
        ROOT / "references.bib": Path("references.bib"),
        ROOT / "OVERLEAF_README.md": Path("OVERLEAF_README.md"),
        ROOT / "figures" / "README.md": Path("figures/README.md"),
        ROOT / "supplementary" / "supplementary.tex": Path("supplementary/supplementary.tex"),
        ROOT / "supplementary" / "supplementary.bbl": Path("supplementary/supplementary.bbl"),
    }
    for relative in MAIN_FIGURES + SUPPLEMENTARY_FIGURES:
        files[ROOT / "figures" / relative] = Path("figures") / relative
    return files


def build(stage: Path, archive: Path, *, force: bool) -> None:
    verify_figure_allowlist()
    digest = archive.with_suffix(archive.suffix + ".sha256")
    if force:
        if stage != DEFAULT_STAGE.resolve() or archive != DEFAULT_ZIP.resolve():
            raise ValueError("--force is restricted to the default Overleaf outputs")
        if stage.exists():
            if not stage.is_dir():
                raise RuntimeError(f"Expected generated directory: {stage}")
            shutil.rmtree(stage)
        for path in (archive, digest):
            if path.exists():
                if not path.is_file():
                    raise RuntimeError(f"Expected generated file: {path}")
                path.unlink()
    if stage.exists() or archive.exists() or digest.exists():
        raise FileExistsError("Overleaf output exists; move it or use --force")

    sources = inputs()
    missing = [source for source in sources if not source.is_file()]
    if missing:
        raise FileNotFoundError("Missing Overleaf inputs:\n" + "\n".join(map(str, missing)))

    stage.mkdir(parents=True)
    for source, relative in sorted(sources.items(), key=lambda item: item[1].as_posix()):
        destination = stage / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        if sha256(source) != sha256(destination):
            raise RuntimeError(f"Copy verification failed: {relative}")

    # Overleaf expects the TeX project at ZIP root, with no enclosing bundle
    # directory and no software/source-data archives mixed into the editor.
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as handle:
        for path in sorted(p for p in stage.rglob("*") if p.is_file()):
            relative = path.relative_to(stage).as_posix()
            info = zipfile.ZipInfo(relative, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            handle.writestr(info, path.read_bytes(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
    if zipfile.ZipFile(archive).testzip() is not None:
        raise RuntimeError(f"Corrupt Overleaf archive: {archive}")
    digest.write_text(f"{sha256(archive)}  {archive.name}\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=Path, default=DEFAULT_STAGE)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ZIP)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    stage = args.stage.resolve()
    archive = args.archive.resolve()
    build(stage, archive, force=args.force)
    print(f"Overleaf directory: {stage}")
    print(f"Overleaf ZIP:       {archive}")
    print(f"Archive SHA-256:    {sha256(archive)}")


if __name__ == "__main__":
    main()
