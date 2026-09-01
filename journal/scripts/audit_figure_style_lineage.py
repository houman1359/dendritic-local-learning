#!/usr/bin/env python3
"""Audit exact NeurIPS figure inheritance and journal-wide style use."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NEURIPS_ROOT = ROOT.parent / "neurips"

# The canonical S01-S03 paths now carry journal-native redraws; the
# byte-identical NeurIPS/arXiv assets are archived under
# figures/supplementary/inherited/ and are what these lineage checks pin.
SHARED_NEURIPS_FILES = {
    "scripts/neurips_style.py": "scripts/neurips_style.py",
    "figures/supplementary/inherited/figure_S01_panels_A-E.pdf": "figures/fig3_mechanistic_evidence.pdf",
    "figures/supplementary/inherited/figure_S02_panels_A-E.pdf": "figures/fig4_competence_regime.pdf",
    "figures/supplementary/inherited/figure_S03_panels_A-D.pdf": "figures/fig5_rule_feedback_controls.pdf",
}

FROZEN_HASHES = {
    "scripts/neurips_style.py": "47b62db8656dd576f2c52a17cddb170a4be35f9a5fb7cacbe2cc2f646b76da46",
    "scripts/figure1_neurips_components.py": "f57439109e3d6e411b1a05abeab86d8c7a994eed6ce21215f3d2f978052e1926",
    "scripts/inherited_neurips/generate_neurips_figures.py": "d03020396906e4ea59db2464dcfd1ad3a33da213d9c7cc970f637cd55e94223d",
    "scripts/inherited_neurips/generate_revision_figures.py": "ded3e36ea4684443b4e5f5f6e7a181c50793f9a9653e209cf40d5ec920ad510b",
    "scripts/inherited_neurips/generate_theory_diagnostics_figures.py": "213610d30ebaebfd7e7417da7b3787221bc8a578bb2d4da9f9a9f82780aede80",
    "figures/supplementary/inherited/figure_S01_panels_A-E.pdf": "88c0fe5290fc4b07539af46388f1ba150ce09c30a3b236881b14f153869c327f",
    "figures/supplementary/inherited/figure_S02_panels_A-E.pdf": "90e4d2378aa8579eeb4f7d6bf47034ea31545730710ad4aee00c6df21cec031a",
    "figures/supplementary/inherited/figure_S03_panels_A-D.pdf": "b51c005792b7165a94f02d661820e1ae7df104304be440d21e1e3e59258431fc",
}

FORBIDDEN_PRODUCTION_TOKENS = (
    "savefig.bbox",
    "bbox_inches=",
    "tight_layout(",
)

NONCANONICAL_WIDTH_RE = re.compile(
    r"figsize\s*=\s*\(\s*(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)"
)

# Main figures are placed at the largest width that keeps the complete figure
# and its self-contained legend on one page.  They remain on the canonical
# NeurIPS-derived canvas; only the LaTeX placement scale changes.  A lower
# bound prevents a crowded sheet from being made illegibly small merely to
# pass the page-layout audit.
MIN_MAIN_TEXTWIDTH_SCALE = 0.85
TEXTWIDTH_SCALE_RE = re.compile(
    r"(?P<scale>(?:0(?:\.\d+)?|1(?:\.0+)?))?\\textwidth"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    failures: list[str] = []
    for relative, expected in FROZEN_HASHES.items():
        path = ROOT / relative
        if not path.is_file():
            failures.append(f"missing frozen file: {relative}")
            continue
        observed = sha256(path)
        if observed != expected:
            failures.append(
                f"frozen lineage mismatch: {relative}: {observed} != {expected}"
            )

    if NEURIPS_ROOT.is_dir():
        for journal_relative, neurips_relative in SHARED_NEURIPS_FILES.items():
            journal_path = ROOT / journal_relative
            neurips_path = NEURIPS_ROOT / neurips_relative
            if not neurips_path.is_file():
                failures.append(
                    f"missing sibling NeurIPS lineage file: {neurips_relative}"
                )
                continue
            if sha256(journal_path) != sha256(neurips_path):
                failures.append(
                    "journal/NeurIPS lineage divergence: "
                    f"{journal_relative} != ../neurips/{neurips_relative}"
                )

    production: list[Path] = []
    for path in sorted((ROOT / "scripts").glob("*.py")):
        text = path.read_text(encoding="utf-8")
        if "savefig" not in text and "save_figure" not in text:
            continue
        if path.name in {"neurips_style.py", "audit_figure_style_lineage.py"}:
            continue
        production.append(path)
        if not any(
            token in text
            for token in ("neurips_style import", "journal_style import")
        ):
            failures.append(f"production generator does not import shared style: {path.name}")
        for token in FORBIDDEN_PRODUCTION_TOKENS:
            if token in text:
                failures.append(f"noncanonical layout token {token!r}: {path.name}")
        if NONCANONICAL_WIDTH_RE.search(text):
            failures.append(
                f"production generator hard-codes a noncanonical figure width: {path.name}"
            )
        if not any(token in text for token in ("FIG_W", "MAIN_W", "grid_figure(")):
            failures.append(
                f"production generator does not use the canonical NeurIPS canvas: {path.name}"
            )

    included_assets: list[str] = []
    include_pattern = re.compile(
        r"\\includegraphics\[width=(?P<width>[^]]+)\]\{(?P<asset>[^}]+)\}"
    )
    for relative in ("main.tex", "supplementary/supplementary.tex"):
        text = (ROOT / relative).read_text(encoding="utf-8")
        for match in include_pattern.finditer(text):
            width = match.group("width")
            asset = match.group("asset")
            included_assets.append(asset)
            width_match = TEXTWIDTH_SCALE_RE.fullmatch(width)
            if width_match is None:
                failures.append(
                    f"non-textwidth LaTeX figure scale in {relative}: {asset} uses {width}"
                )
            else:
                scale_text = width_match.group("scale")
                scale = 1.0 if scale_text is None else float(scale_text)
                if relative == "main.tex":
                    if not MIN_MAIN_TEXTWIDTH_SCALE <= scale <= 1.0:
                        failures.append(
                            "main-figure placement outside the legibility range "
                            f"[{MIN_MAIN_TEXTWIDTH_SCALE:.2f}, 1.00] in {relative}: "
                            f"{asset} uses {width}"
                        )
                elif scale != 1.0:
                    failures.append(
                        f"supplementary figure is not full width in {relative}: "
                        f"{asset} uses {width}"
                    )
            pdf = ROOT / "figures" / asset
            if not pdf.is_file():
                failures.append(f"missing included PDF: {pdf.relative_to(ROOT)}")
            # Raster previews are disposable inspection artifacts and are
            # intentionally ignored by Git.  The publication-facing vector
            # PDF is the sole canonical figure asset.

    if failures:
        raise SystemExit("Figure-lineage audit failed:\n- " + "\n- ".join(failures))
    print(f"Figure-lineage audit passed: {len(FROZEN_HASHES)} frozen files, "
          f"{len(SHARED_NEURIPS_FILES)} live sibling matches, "
          f"{len(production)} production generators, "
          f"{len(included_assets)} bounded-width figure placements.")


if __name__ == "__main__":
    main()
