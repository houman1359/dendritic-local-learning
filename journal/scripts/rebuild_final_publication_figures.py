#!/usr/bin/env python3
"""Build the publication figures from frozen tables and selected vector inputs.

The current entry point renders the restored main figures and the consolidated
supplement. It does not regenerate a sequence of superseded displays, invoke
training, or change any experimental outcome. Four unchanged main figures and
the selected supplementary source panels are authenticated vector inputs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
RESTORED_MAIN = (1, 4, 6, 7, 9)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_script(name: str, *arguments: str) -> None:
    print(f"Rendering {name}", flush=True)
    subprocess.run([sys.executable, str(ROOT / "scripts" / name), *arguments],
                   cwd=ROOT, check=True)


def verify_retained_main() -> None:
    manifest = json.loads((ROOT / "configs/figure_structure/retained_main_inputs.json").read_text())
    for asset in manifest["assets"]:
        path = ROOT / asset["path"]
        if not path.is_file() or sha256(path) != asset["sha256"]:
            raise ValueError(f"Retained publication input differs: {path}")


def export_display_tables() -> None:
    """Copy generated display summaries, without changing their source studies."""
    output = ROOT / "source_data/curated_publication"
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for number in RESTORED_MAIN:
        source = ROOT / "figures/provenance/structure_restoration_20260908" / f"figure_{number:02d}_plotted.csv"
        destination = output / source.name
        shutil.copyfile(source, destination)
        if sha256(source) != sha256(destination):
            raise RuntimeError(f"Display-table copy failed: {source}")
        records.append({"figure": number, "source": str(source.relative_to(ROOT)),
                        "path": str(destination.relative_to(ROOT)),
                        "sha256": sha256(destination),
                        "scope": "Display summaries from existing inputs; panel column identifies the selected data. Not an additional experiment."})
    (output / "manifest.json").write_text(json.dumps({"records": records}, indent=2) + "\n")
    (output / "README.md").write_text(
        "# Publication display tables\n\n"
        "These tables reproduce the values drawn in the current main figures. "
        "The panel column identifies the display; heterogeneous panels leave "
        "unrelated columns empty. Full numerical inputs and protocol records "
        "remain in their study directories. Figure 1C is an explicitly "
        "illustrative evaluation of the one-step bound, not a fitted result.\n\n"
        "Rebuild with `python scripts/rebuild_final_publication_figures.py`. "
        "Per-panel input hashes and rendering definitions are in "
        "`figures/provenance/structure_restoration_20260908/` and "
        "`scripts/credit_first_figures/build_restored_main.py`.\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-emit-main", action="store_true",
                        help="Render main components without replacing canonical main PDFs.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--supplement-only", action="store_true")
    mode.add_argument("--main-only", action="store_true")
    args = parser.parse_args()
    verify_retained_main()
    if not args.supplement_only:
        arguments = [] if args.no_emit_main else ["--emit-main"]
        run_script("credit_first_figures/build_restored_main.py", *arguments)
        export_display_tables()
    if not args.main_only:
        run_script("supplement_consolidation/build.py")
    from build_submission_bundle import MAIN_FIGURES, SUPPLEMENTARY_FIGURES, verify_figure_allowlist
    verify_figure_allowlist()
    for relative in MAIN_FIGURES + SUPPLEMENTARY_FIGURES:
        if not (ROOT / "figures" / relative).is_file():
            raise FileNotFoundError(relative)
    print(f"Publication figures ready: {len(MAIN_FIGURES)} main and "
          f"{len(SUPPLEMENTARY_FIGURES)} supplementary figures.", flush=True)


if __name__ == "__main__":
    main()
