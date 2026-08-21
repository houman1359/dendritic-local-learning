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
    "fig3.asset": "figures/main/figure_07.pdf",
    "capturewire.asset": "figures/main/figure_07.pdf",
    "fig4.asset": "figures/main/figure_08.pdf",
    "extensions.asset": "figures/main/figure_08.pdf",
    "extensions.fulltree.asset": "figures/main/figure_09.pdf",
    "fig6.asset": "figures/main/figure_09.pdf",
    "phaseplane.asset": "figures/main/figure_04.pdf",
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
    "fig2.asset": ("fig2", "b-c"),
    "prospective.learning.asset": ("fig2", "d-f"),
    "fashion.asset": ("fig2", "g-h"),
    "subtree.factorial.asset": ("fig3", "all"),
    "creditphase.asset": ("fig4", "a-f"),
    "phaseplane.asset": ("fig4", "g"),
    "physical.asset": ("fig5", "b-e"),
    "pointcredit.asset": ("fig5", "f"),
    "fig3.asset": ("fig7", "b-d"),
    "capturewire.asset": ("fig7", "e-f"),
    "fig4.asset": ("fig8", "a-e"),
    "extensions.asset": ("fig8", "f-g"),
    "fig6.asset": ("fig9", "b-f"),
    "extensions.fulltree.asset": ("fig9", "g-h"),
    "alignmentdose.asset": ("figS18", "c-e"),
    "remainingphysical.asset": ("figS18", "f-k"),
}

CANONICAL_GENERATORS = {
    entry_id: "scripts/assemble_compact_main_figures.py"
    for entry_id in (
        "fig1.asset",
        "fig2.asset",
        "prospective.learning.asset",
        "fashion.asset",
        "subtree.factorial.asset",
        "creditphase.asset",
        "phaseplane.asset",
        "physical.asset",
        "pointcredit.asset",
        "fig3.asset",
        "capturewire.asset",
        "fig4.asset",
        "extensions.asset",
        "fig6.asset",
        "extensions.fulltree.asset",
    )
}

# This standalone controlled-alignment graphic was superseded by the focused
# main-panel summary plus Supplementary Figure S5.  Its numerical source rows
# remain registered; the obsolete publication asset row does not.
SUPERSEDED_ENTRIES = {"fig8.asset"}

NEW_DETAIL_ASSETS = {
    "physical.h4.asset": {
        "figure": "fig6",
        "path": "figures/main/figure_06.pdf",
        "generator": "scripts/assemble_compact_main_figures.py",
        "notes": "H4 depth-saturation test: exact backpropagation, point emulation, local credit, and mechanism controls.",
    },
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
        "notes": "Detailed morphology-route capacity and topology controls underlying focused Fig. 7.",
    },
    "focal.detail.asset": {
        "figure": "figS21",
        "path": "figures/supplementary/figure_S21_panels_A-I.pdf",
        "generator": "scripts/build_journal_figures.py",
        "notes": "Detailed focal-shunting controls and electrotonic calibration underlying focused Fig. 8.",
    },
    "measured.detail.asset": {
        "figure": "figS22",
        "path": "figures/supplementary/figure_S22_panels_A-H.pdf",
        "generator": "scripts/build_journal_figures.py",
        "notes": "Detailed measured-response topology and learning boundary underlying focused Fig. 9.",
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
    "irregular.wavelet.asset": {
        "figure": "figS25",
        "path": "figures/supplementary/figure_S25_panels_A-D.pdf",
        "generator": "scripts/build_irregular_tree_wavelet_figure.py",
        "notes": "Frozen 47-cell primary and eight-cell secondary irregular-tree Haar analysis with isotropic and column-permuted controls.",
    },
    "clean.physical.depth.asset": {
        "figure": "figS26",
        "path": "figures/supplementary/figure_S26_panels_A-D.pdf",
        "generator": "scripts/analyze_physical_depth_clean_source_replication.py",
        "notes": "Immutable-source same-seed replication of all 430 load-bearing H2/H3 physical-depth fits.",
    },
    "physical.diagnostics.asset": {
        "figure": "figS28",
        "path": "figures/supplementary/figure_S28_panels_A-B.pdf",
        "generator": "scripts/assemble_compact_main_figures.py",
        "notes": "Credit-coordinate ladder and backpropagation--local decomposition demoted from the focused main physical-depth figure.",
    },
}

NEW_PROVENANCE_ENTRIES = {
    "taskfamily.asset": {
        "record_type": "figure_asset",
        "figure": "fig6",
        "panel": "f-h",
        "path": "figures/main/figure_06.pdf",
        "generator": "scripts/assemble_compact_main_figures.py",
        "replication_unit": "paired independent training seed (n=10)",
        "notes": "Fixed-D3 architecture-by-task-family-by-alignment boundary under exact backpropagation and path-transport LocalCA.",
    },
    "taskfamily.outcomes": {
        "record_type": "panel_source",
        "figure": "fig6",
        "panel": "f-h",
        "path": "source_data/task_family_alignment/seed_outcomes.csv",
        "generator": "scripts/analyze_task_family_alignment_factorial.py",
        "replication_unit": "paired independent training seed (n=10)",
        "notes": "All 360 fixed-D3 task-family, architecture, credit-rule and alignment outcomes.",
    },
    "taskfamily.conditions": {
        "record_type": "panel_source",
        "figure": "fig6",
        "panel": "f-g",
        "path": "source_data/task_family_alignment/condition_summary.csv",
        "generator": "scripts/analyze_task_family_alignment_factorial.py",
        "replication_unit": "paired independent training seed (n=10)",
        "notes": "Condition means and paired-seed bootstrap intervals for the fixed-depth task-family boundary.",
    },
    "taskfamily.contrasts": {
        "record_type": "panel_source",
        "figure": "fig6",
        "panel": "h",
        "path": "source_data/task_family_alignment/paired_contrasts.csv",
        "generator": "scripts/analyze_task_family_alignment_factorial.py",
        "replication_unit": "paired independent training seed (n=10)",
        "notes": "Architecture-by-alignment interactions and task-family difference-in-differences.",
    },
    "taskfamily.audit": {
        "record_type": "panel_source",
        "figure": "fig6",
        "panel": "text",
        "path": "source_data/task_family_alignment/audit.json",
        "generator": "scripts/analyze_task_family_alignment_factorial.py",
        "replication_unit": "complete 360-fit audit",
        "notes": "Completeness, finite-metric, no-fallback, seed and exact-resource gates.",
    },
    "pinky.asset": {
        "record_type": "figure_asset",
        "figure": "figS27",
        "panel": "all",
        "path": "figures/supplementary/figure_S27_panels_A-D.pdf",
        "generator": "scripts/analyze_pinky_v185_replication.py",
        "replication_unit": "reconstructed cell (10 QC-passing of 12 selected; one second mouse)",
        "notes": "Independent-animal structural route-capacity replication; panel C is promoted as Fig. 7G.",
    },
    "pinky.cohort": {
        "record_type": "panel_source",
        "figure": "figS27",
        "panel": "a",
        "path": "source_data/pinky_v185_replication/cohort_manifest.csv",
        "generator": "scripts/freeze_pinky_v185_cohort.py",
        "replication_unit": "outcome-independent reconstructed-cell selection (n=12)",
        "notes": "Twelve equal-count y strata with cells nearest the global x/z medians in the MICrONS Pinky v185 volume.",
    },
    "pinky.curves": {
        "record_type": "panel_source",
        "figure": "fig7/figS27",
        "panel": "g/b-d",
        "path": "source_data/pinky_v185_replication/routing/feedback_compression_curves.csv",
        "generator": "scripts/analyze_pinky_v185_replication.py",
        "replication_unit": "QC-passing reconstructed cell (n=10; one second mouse)",
        "notes": "Cell-level capture curves for ancestry, random, depth-only, shuffled and dense routes.",
    },
    "pinky.contrasts": {
        "record_type": "panel_source",
        "figure": "fig7/figS27",
        "panel": "g/c",
        "path": "source_data/pinky_v185_replication/routing/k4_cross_animal_contrasts.csv",
        "generator": "scripts/analyze_pinky_v185_replication.py",
        "replication_unit": "reconstructed cell; animal is the biological unit (two mice)",
        "notes": "K=4 ancestry-route advantages shown separately for the original and second MICRONS animals.",
    },
    "pinky.summary": {
        "record_type": "panel_source",
        "figure": "figS27",
        "panel": "text",
        "path": "source_data/pinky_v185_replication/routing/summary.json",
        "generator": "scripts/analyze_pinky_v185_replication.py",
        "replication_unit": "12 selected cells, 10 QC-passing; one second mouse",
        "notes": "Frozen preprocessing, QC, route-capacity and cross-animal directional-replication summary.",
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
    "fig2.asset": "Final eight-panel identity, ownership and within-tree-address figure with shared vector schematic.",
    "fig3.e": "Direct-presynaptic-type-only sensitivity analysis shown in Supplementary Fig. 20H.",
    "fig4.b": "Cell-level additive, depth-shuffled, and focal-shunt localization shown in Fig. 8B.",
    "fig4.c": "Dose response and focal-depth localization values shown in Fig. 8B--C.",
    "fig4.effects": "Descendant, sister, ancestor, and unrelated category effects shown in Fig. 8B.",
    "fig4.e.cell": "Cell-level exact factor-freeze values underlying the same-voltage driving-force and full-shunt contrast in Fig. 8D.",
    "fig4.e.site": "Site-level factor substitutions supporting the Fig. 8D cell summary and supplementary Shapley analysis.",
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
        rows = [
            row for row in reader if row.get("entry_id", "") not in SUPERSEDED_ENTRIES
        ]
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

        generator = CANONICAL_GENERATORS.get(entry_id)
        if generator is not None:
            expected_generator = PROJECT_PREFIX + generator
            if row.get("generator_path") != expected_generator:
                row["generator_path"] = expected_generator
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

    existing_ids = {row.get("entry_id", "") for row in rows}
    for entry_id, detail in NEW_PROVENANCE_ENTRIES.items():
        source_path = PROJECT_PREFIX + detail["path"]
        source = resolve_project_path(source_path)
        if not source.is_file():
            raise FileNotFoundError(source)
        expected = {
            "record_type": detail["record_type"],
            "figure": detail["figure"],
            "panel": detail["panel"],
            "status": "ready",
            "source_path": source_path,
            "sha256": sha256(source),
            "generator_path": PROJECT_PREFIX + detail["generator"],
            "replication_unit": detail["replication_unit"],
            "notes": detail["notes"],
        }
        if entry_id in existing_ids:
            row = next(item for item in rows if item.get("entry_id") == entry_id)
            for field, value in expected.items():
                if row.get(field) != value:
                    row[field] = value
                    updated += 1
        else:
            rows.append({"entry_id": entry_id, **expected})
            updated += 1

    missing_entries = sorted(set(CANONICAL_ASSETS) - seen_assets)
    if missing_entries:
        raise ValueError(
            "canonical figure assets absent from manifest: "
            + ", ".join(missing_entries)
        )

    with MANIFEST.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fieldnames, delimiter="\t", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"updated {updated} provenance fields")


if __name__ == "__main__":
    main()
