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
PROJECT_ROOT = JOURNAL_ROOT.parent
MANIFEST = JOURNAL_ROOT / "source_data" / "provenance_manifest.tsv"
PROJECT_PREFIX = "drafts/dendritic-local-learning/journal/"
LEGACY_PROJECT_PREFIX = Path("drafts/dendritic-local-learning")


CANONICAL_ASSETS = {
    "fig1.asset": "figures/main/figure_01.pdf",
    "fig2.asset": "figures/main/figure_02.pdf",
    "prospective.learning.asset": "figures/main/figure_02.pdf",
    "fashion.asset": "figures/main/figure_02.pdf",
    "subtree.factorial.asset": "figures/main/figure_03.pdf",
    "creditphase.asset": "figures/main/figure_04.pdf",
    "physical.asset": "figures/main/figure_05.pdf",
    "pointcredit.asset": "figures/main/figure_05.pdf",
    "alignmentdose.asset": "figures/supplementary/figure_S18_panels_A-K.pdf",
    "remainingphysical.asset": "figures/supplementary/figure_S18_panels_A-K.pdf",
    "fig3.asset": "figures/main/figure_06.pdf",
    "capturewire.asset": "figures/main/figure_06.pdf",
    "fig4.asset": "figures/main/figure_07.pdf",
    "extensions.asset": "figures/main/figure_07.pdf",
    "extensions.fulltree.asset": "figures/main/figure_08.pdf",
    "fig6.asset": "figures/main/figure_08.pdf",
    "fig8.asset": "figures/main/figure_08.pdf",
    "phaseplane.asset": "figures/main/figure_08.pdf",
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

CANONICAL_ASSIGNMENTS = {
    "fig1.asset": ("fig1", "all"),
    "fig2.asset": ("fig2", "a-c"),
    "prospective.learning.asset": ("fig2", "d-f"),
    "fashion.asset": ("fig2", "g-h"),
    "subtree.factorial.asset": ("fig3", "all"),
    "creditphase.asset": ("fig4", "all"),
    "physical.asset": ("fig5", "a-f"),
    "pointcredit.asset": ("fig5", "g-j"),
    "fig3.asset": ("fig6", "a-f"),
    "capturewire.asset": ("fig6", "g-h"),
    "fig4.asset": ("fig7", "a-f"),
    "extensions.asset": ("fig7", "g-j"),
    "fig6.asset": ("fig8", "a-f"),
    "extensions.fulltree.asset": ("fig8", "g-h"),
    "fig8.asset": ("fig8", "i-j"),
    "phaseplane.asset": ("fig8", "k"),
    "alignmentdose.asset": ("figS18", "c-e"),
    "remainingphysical.asset": ("figS18", "f-k"),
}

NEW_DETAIL_ASSETS = {
    "prospective.detail.asset": {
        "figure": "figS19",
        "path": "figures/supplementary/figure_S19_panels_A-I.pdf",
        "generator": "scripts/assemble_compact_main_figures.py",
        "notes": "Detailed identity, ownership and route diagnostics underlying focused Fig. 2.",
    },
    "morphology.detail.asset": {
        "figure": "figS20",
        "path": "figures/supplementary/figure_S20_panels_A-J.pdf",
        "generator": "scripts/build_journal_figures.py",
        "notes": "Detailed morphology-route capacity and topology controls underlying focused Fig. 6.",
    },
    "focal.detail.asset": {
        "figure": "figS21",
        "path": "figures/supplementary/figure_S21_panels_A-I.pdf",
        "generator": "scripts/build_journal_figures.py",
        "notes": "Detailed focal-shunting controls and electrotonic calibration underlying focused Fig. 7.",
    },
    "measured.detail.asset": {
        "figure": "figS22",
        "path": "figures/supplementary/figure_S22_panels_A-H.pdf",
        "generator": "scripts/build_journal_figures.py",
        "notes": "Detailed measured-response topology and learning boundary underlying focused Fig. 8.",
    },
    "partition.residual.asset": {
        "figure": "figS23",
        "path": "figures/supplementary/figure_S23_panels_A-C.pdf",
        "generator": "scripts/build_trained_partition_residual_figure.py",
        "notes": "Hash-gated reconstruction by scripts/analyze_trained_partition_residual.py and seed-level capture--utility association.",
    },
    "adaptive.reliability.asset": {
        "figure": "figS24",
        "path": "figures/supplementary/figure_S24_panels_A-D.pdf",
        "generator": "scripts/build_adaptive_conductance_reliability_figure.py",
        "notes": "Fresh 50-seed adaptive local reliability experiment with global, shuffled, no-shunt, oracle and independent point-gate controls.",
    },
}


def resolve_project_path(raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    try:
        relative = path.relative_to(LEGACY_PROJECT_PREFIX)
    except ValueError:
        relative = path
    return PROJECT_ROOT / relative

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

        assignment = CANONICAL_ASSIGNMENTS.get(entry_id)
        if assignment is not None:
            figure, panel = assignment
            if row.get("figure") != figure:
                row["figure"] = figure
                updated += 1
            if row.get("panel") != panel:
                row["panel"] = panel
                updated += 1

        canonical_note = CANONICAL_NOTES.get(entry_id)
        if canonical_note is not None and row.get("notes") != canonical_note:
            row["notes"] = canonical_note
            updated += 1

        source = resolve_project_path(row["source_path"])
        if not source.is_file():
            raise FileNotFoundError(source)
        observed = sha256(source)
        if row.get("sha256") != observed:
            row["sha256"] = observed
            updated += 1

    existing_ids = {row.get("entry_id", "") for row in rows}
    for entry_id, detail in NEW_DETAIL_ASSETS.items():
        if entry_id in existing_ids:
            row = next(item for item in rows if item.get("entry_id") == entry_id)
            expected = {
                "figure": detail["figure"],
                "panel": "all",
                "status": "ready",
                "source_path": PROJECT_PREFIX + detail["path"],
                "generator_path": PROJECT_PREFIX + detail["generator"],
                "replication_unit": "not applicable",
                "notes": detail["notes"],
            }
            for field, value in expected.items():
                if row.get(field) != value:
                    row[field] = value
                    updated += 1
            continue
        relative_path = detail["path"]
        source_path = PROJECT_PREFIX + relative_path
        source = resolve_project_path(source_path)
        if not source.is_file():
            raise FileNotFoundError(source)
        rows.append(
            {
                "entry_id": entry_id,
                "record_type": "figure_asset",
                "figure": detail["figure"],
                "panel": "all",
                "status": "ready",
                "source_path": source_path,
                "sha256": sha256(source),
                "generator_path": PROJECT_PREFIX + detail["generator"],
                "replication_unit": "not applicable",
                "notes": detail["notes"],
            }
        )
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
