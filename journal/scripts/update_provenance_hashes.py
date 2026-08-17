#!/usr/bin/env python3
"""Refresh provenance hashes and enforce current figure-asset paths.

Figure numbers and panel assignments in the manifest are publication facts,
not historical generator names.  This script therefore changes only the
canonical path of each figure asset and the SHA-256 values of existing source
files.  It never silently renumbers figures, panels, or prose notes.
"""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path


JOURNAL_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = JOURNAL_ROOT.parents[2]
MANIFEST = JOURNAL_ROOT / "source_data" / "provenance_manifest.tsv"
PROJECT_PREFIX = "drafts/dendritic-local-learning/journal/"


CANONICAL_ASSETS = {
    "fig1.asset": "figures/main/figure_01_panels_A-E.pdf",
    "fig2.asset": "figures/main/figure_02_panels_A-F.pdf",
    "prospective.learning.asset": "figures/main/figure_02_panels_G-O.pdf",
    "fashion.asset": "figures/main/figure_02_panels_P-Q.pdf",
    "subtree.factorial.asset": "figures/main/figure_03_panels_A-F.pdf",
    "creditphase.asset": "figures/main/figure_04_panels_A-I.pdf",
    "physical.asset": "figures/main/figure_05_panels_A-F.pdf",
    "pointcredit.asset": "figures/main/figure_05_panels_G-L.pdf",
    "alignmentdose.asset": "figures/main/figure_05_panels_M-O.pdf",
    "remainingphysical.asset": "figures/main/figure_05_panels_P-U.pdf",
    "fig3.asset": "figures/main/figure_06_panels_A-J.pdf",
    "capturewire.asset": "figures/main/figure_06_panels_K-L.pdf",
    "fig4.asset": "figures/main/figure_07_panels_A-I.pdf",
    "extensions.asset": "figures/main/figure_07_panels_J-M.pdf",
    "extensions.fulltree.asset": "figures/main/figure_08_panels_I-J.pdf",
    "fig6.asset": "figures/main/figure_08_panels_A-H.pdf",
    "fig8.asset": "figures/main/figure_08_panels_K-N.pdf",
    "phaseplane.asset": "figures/main/figure_08_panel_O.pdf",
    "inherited.s1.asset": "figures/supplementary/figure_S01_panels_A-E.pdf",
    "inherited.s2.asset": "figures/supplementary/figure_S02_panels_A-E.pdf",
    "inherited.s3.asset": "figures/supplementary/figure_S03_panels_A-D.pdf",
    "regimes.asset": "figures/supplementary/figure_S04_panels_A-I.pdf",
    "fig7.asset": "figures/supplementary/figure_S05_panels_A-H.pdf",
    "interference.asset": "figures/supplementary/figure_S06_panels_A-D.pdf",
    "spatial.audit.asset": "figures/supplementary/figure_S07_panels_A-D.pdf",
    "prospective.fixed_budget.asset": "figures/supplementary/figure_S08_panels_A-I.pdf",
    "prospective.mechanism.asset": "figures/supplementary/figure_S09_panels_A-D.pdf",
    "fig5.asset": "figures/supplementary/figure_S10_panels_A-H.pdf",
    "focalmatrix.asset": "figures/supplementary/figure_S11_panels_A-C.pdf",
    "inhcensus.asset": "figures/supplementary/figure_S12_panels_A-J.pdf",
    "positive.asset": "figures/supplementary/figure_S13_panels_A-F.pdf",
    "same_span.asset": "figures/supplementary/figure_S14_panels_A-F.pdf",
    "physical_calibration.asset": "figures/supplementary/figure_S15_panels_A-D.pdf",
    "interior.asset": "figures/supplementary/figure_S16_panels_A-D.pdf",
    "animalcredit.asset": "figures/supplementary/figure_S17_panels_A-D.pdf",
}

# Correct stale prose left by an earlier numbering-only remap.  These notes
# are descriptive metadata, not independent scientific content.
CANONICAL_NOTES = {
    "fig3.e": "Direct-presynaptic-type-only sensitivity analysis shown in Fig. 6H.",
    "fig4.b": "Cell-level additive, depth-shuffled, and focal-shunt localization shown in Fig. 7C.",
    "fig4.c": "Dose response and focal-depth localization values shown in Fig. 7D--E.",
    "fig4.effects": "Descendant, sister, ancestor, and unrelated category effects shown in Fig. 7B.",
    "fig4.e.cell": "Cell-level exact factor-freeze values underlying the same-voltage driving-force and full-shunt contrast in Fig. 7H.",
    "fig4.e.site": "Site-level factor substitutions supporting the Fig. 7H cell summary and supplementary Shapley analysis.",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    with MANIFEST.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    if "source_path" not in fieldnames or "sha256" not in fieldnames:
        raise ValueError("manifest must contain source_path and sha256 columns")

    updated = 0
    seen_assets: set[str] = set()
    for row in rows:
        entry_id = row.get("entry_id", "")
        canonical = CANONICAL_ASSETS.get(entry_id)
        if canonical is not None:
            expected = PROJECT_PREFIX + canonical
            if row.get("source_path") != expected:
                row["source_path"] = expected
                updated += 1
            seen_assets.add(entry_id)

        canonical_note = CANONICAL_NOTES.get(entry_id)
        if canonical_note is not None and row.get("notes") != canonical_note:
            row["notes"] = canonical_note
            updated += 1

        source = REPOSITORY_ROOT / row["source_path"]
        if not source.is_file():
            raise FileNotFoundError(source)
        observed = sha256(source)
        if row.get("sha256") != observed:
            row["sha256"] = observed
            updated += 1

    missing_entries = sorted(set(CANONICAL_ASSETS) - seen_assets)
    if missing_entries:
        raise ValueError("canonical figure assets absent from manifest: " + ", ".join(missing_entries))

    with MANIFEST.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"updated {updated} provenance fields")


if __name__ == "__main__":
    main()
