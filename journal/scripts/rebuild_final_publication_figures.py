#!/usr/bin/env python3
"""Build the publication figures from frozen tables and selected vector inputs.

The entry point renders the current main figures and consolidated supplement.
It does not invoke training or change experimental outcomes. Figure 2 and the
selected supplementary source panels remain authenticated vector inputs.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
# The v2 overhaul (2026-09-10) production builders, one per main figure.
# build_restored_main.py keeps only Figures 4 and 6; Figures 1, 7 and 9 moved to
# their own builders and Figure 2 is no longer a retained input.
RESTORED_MAIN = (4, 7)
FOCUSED_MAIN = {
    1: "credit_first_figures/build_framework.py",
    2: "build_main_figure_04.py",
    3: "credit_first_figures/build_ancestry.py",
    5: "conductance_local_gate/figure.py",
    6: "review_completion/population_figure.py",
    8: "credit_first_figures/build_anatomy.py",
    9: "shunt_ancestry_gain/build_focused_main.py",
    10: "credit_first_figures/build_measured.py",
}
#: Builders with no --emit-main flag: the component they write is copied here.
COMPONENT_OUTPUT = {
    1: "figures/components/credit_first_figure_01.pdf",
    2: "figures/components/main_figure_04_native.pdf",
}
#: Figures whose builder writes its display table into figures/provenance/;
#: the emission step copies those into source_data/curated_publication/.
#: Figures 1, 2, 3, 5 and 9 write that file directly and are not copied here
#: (copying the stale provenance twin would overwrite the fresh table).
PROVENANCE_DIR = {
    4: "structure_restoration_20260908",
    7: "structure_restoration_20260908",
    8: "structure_restoration_20260908",
    9: "credit_clarity_20260908",
}


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
    for number in sorted(set(RESTORED_MAIN) | set(FOCUSED_MAIN)):
        destination = output / f"figure_{number:02d}_plotted.csv"
        if number in PROVENANCE_DIR:
            directory = PROVENANCE_DIR[number]
            internal = {7: 6, 8: 7}.get(number, number)
            source = ROOT / "figures/provenance" / directory / f"figure_{internal:02d}_plotted.csv"
            shutil.copyfile(source, destination)
            if sha256(source) != sha256(destination):
                raise RuntimeError(f"Display-table copy failed: {source}")
        else:
            # These builders emit directly into curated_publication. Do not
            # replace their fresh tables with older provenance-directory twins,
            # or drop their records when writing the aggregate manifest.
            source = ROOT / "scripts" / FOCUSED_MAIN[number]
            if not destination.is_file():
                raise FileNotFoundError(f"Direct display table is missing: {destination}")
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
        "remain in their study directories. Figure 1C shows the indicator "
        "supports of the illustrative K = 1, 3 and 12 dictionaries, not a "
        "fitted result.\n\n"
        "Rebuild with `python scripts/rebuild_final_publication_figures.py`. "
        "Per-panel input hashes and rendering definitions are in "
        "`figures/provenance/structure_restoration_20260908/` and "
        "`figures/provenance/credit_clarity_20260908/`. Promoted capture, "
        "coefficient and continuous-gate panels use already completed studies. "
        "Exploratory coefficient comparisons remain distinct from primary "
        "noisy-cue outcomes.\n\n"
        "Figure 1 panels C-G are emitted by "
        "`scripts/credit_first_figures/build_framework.py`; panels A and B "
        "are schematics and contribute no rows.\n\n"
        "Figure 2 panels C-H are emitted by `scripts/build_main_figure_04.py` "
        "from `source_data/path_necessity_fashion/` and "
        "`source_data/trained_subtree_address/`.\n")


def record_render_environment() -> None:
    """Record the executed rendering environment, separate from training."""
    versions = {package: importlib.metadata.version(package)
                for package in ("numpy", "pandas", "scipy", "matplotlib", "PyMuPDF")}
    fonts = {}
    for name in ("NimbusSans-Regular.otf", "NimbusSans-Bold.otf"):
        path = Path("/usr/share/fonts/urw-base35") / name
        fonts[name] = {"path": str(path), "sha256": sha256(path)}
    output = ROOT / "figures/provenance/publication_render_environment.json"
    output.write_text(json.dumps({
        "python": sys.version, "packages": versions, "fonts": fonts,
        "scope": "Executed figure rendering only; not an original training environment.",
    }, indent=2) + "\n")


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
        run_script("credit_first_figures/build_restored_main.py",
                   "--figures", "4", "6", *arguments)
        for number, script in sorted(FOCUSED_MAIN.items()):
            run_script(script, *([] if number in COMPONENT_OUTPUT else arguments))
        if not args.no_emit_main:
            for number, component in sorted(COMPONENT_OUTPUT.items()):
                destination = ROOT / f"figures/main/figure_{number:02d}.pdf"
                shutil.copyfile(ROOT / component, destination)
                print(f"Emitted figure_{number:02d}.pdf from {component}", flush=True)
        export_display_tables()
    if not args.main_only:
        for name in ("mnist_dictionary_geometry", "local_gate_controls",
                     "physical_architecture", "measured_transfer_geometry"):
            run_script(f"build_supplementary_figure_{name}_native.py")
        run_script("supplement_consolidation/refresh_native_assets.py")
        run_script("supplement_consolidation/build.py")
        run_script("review_completion/checkpoint_figure.py")
        run_script("review_completion/optional_extension_paper_figure.py")
    record_render_environment()
    from build_submission_bundle import MAIN_FIGURES, SUPPLEMENTARY_FIGURES, verify_figure_allowlist
    verify_figure_allowlist()
    for relative in MAIN_FIGURES + SUPPLEMENTARY_FIGURES:
        if not (ROOT / "figures" / relative).is_file():
            raise FileNotFoundError(relative)
    print(f"Publication figures ready: {len(MAIN_FIGURES)} main and "
          f"{len(SUPPLEMENTARY_FIGURES)} supplementary figures.", flush=True)


if __name__ == "__main__":
    main()
