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
    "prospective.learning.asset": "figures/supplementary/figure_S19_panels_A-I.pdf",
    "fashion.asset": "figures/main/figure_02.pdf",
    "subtree.factorial.asset": "figures/main/figure_05.pdf",
    "creditphase.asset": "figures/main/figure_03.pdf",
    "physical.asset": "figures/main/figure_06.pdf",
    "pointcredit.asset": "figures/main/figure_06.pdf",
    "alignmentdose.asset": "figures/supplementary/figure_S18_panels_A-K.pdf",
    "remainingphysical.asset": "figures/supplementary/figure_S18_panels_A-K.pdf",
    "fig3.asset": "figures/main/figure_07.pdf",
    "capturewire.asset": "figures/main/figure_07.pdf",
    "fig4.asset": "figures/main/figure_08.pdf",
    "extensions.asset": "figures/main/figure_08.pdf",
    "extensions.fulltree.asset": "figures/main/figure_09.pdf",
    "fig6.asset": "figures/main/figure_09.pdf",
    "phaseplane.asset": "figures/main/figure_03.pdf",
    "inherited.s1.asset": "figures/supplementary/figure_S01_panels_A-E.pdf",
    "inherited.s2.asset": "figures/supplementary/figure_S02_panels_A-E.pdf",
    "inherited.s3.asset": "figures/supplementary/figure_S03_panels_A-D.pdf",
    "regimes.asset": "figures/supplementary/figure_S04_panels_A-D.pdf",
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
    "animalcredit.asset": "figures/supplementary/figure_S17_panels_A-B.pdf",
}

CANONICAL_ASSIGNMENTS = {
    "fig1.asset": ("fig1", "all"),
    "fig2.asset": ("fig2", "all"),
    "prospective.learning.asset": ("figS19", "all"),
    "fashion.asset": ("fig2", "c,f"),
    "subtree.factorial.asset": ("fig5", "all"),
    "creditphase.asset": ("fig3", "a-f"),
    "phaseplane.asset": ("fig3", "g"),
    "physical.asset": ("fig6", "c"),
    "pointcredit.asset": ("fig6", "c"),
    "fig3.asset": ("fig7", "all"),
    "capturewire.asset": ("fig7", "e-f"),
    "fig4.asset": ("fig8", "a-d,f"),
    "extensions.asset": ("fig8", "e,g"),
    "fig6.asset": ("fig9", "a,c-g"),
    "extensions.fulltree.asset": ("fig9", "b-c,h"),
    "alignmentdose.asset": ("figS18", "c-e"),
    "remainingphysical.asset": ("figS18", "f-k"),
}

CANONICAL_GENERATORS = {
    entry_id: "scripts/assemble_compact_main_figures.py"
    for entry_id in (
        "animalcredit.asset",
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

# The regular-tree foundation sheets are redrawn by native journal builders
# from the frozen archived aggregations; the byte-identical originals live in
# figures/supplementary/inherited/.
CANONICAL_GENERATORS["regimes.asset"] = (
    "scripts/build_supplementary_figure_s04_native.py")
CANONICAL_GENERATORS["inherited.s1.asset"] = (
    "scripts/build_supplementary_figure_s01_native.py")
CANONICAL_GENERATORS["inherited.s2.asset"] = (
    "scripts/build_supplementary_figure_s02_native.py")
CANONICAL_GENERATORS["inherited.s3.asset"] = (
    "scripts/build_supplementary_figure_s03_native.py")

# This standalone controlled-alignment graphic was superseded by the focused
# main-panel summary plus Supplementary Figure S5.  Its numerical source rows
# remain registered; the obsolete publication asset row does not.
SUPERSEDED_ENTRIES = {"fig8.asset"}

NEW_DETAIL_ASSETS = {
    "branchconflict.asset": {
        "figure": "fig4",
        "path": "figures/main/figure_04.pdf",
        "generator": "scripts/build_main_figure_04.py",
        "notes": "Context-gated branch-conflict task: analytic shared-mode boundary, trained transitions and exact-route equivalence.",
    },
    "physical.h4.asset": {
        "figure": "fig6",
        "panel": "d-e",
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
        "path": "figures/supplementary/figure_S20_panels_A-B.pdf",
        "generator": "scripts/assemble_compact_main_figures.py",
        "notes": "Detailed morphology-route capacity and topology controls underlying focused Fig. 7.",
    },
    "focal.detail.asset": {
        "figure": "figS21",
        "path": "figures/supplementary/figure_S21_panels_A-D.pdf",
        "generator": "scripts/assemble_compact_main_figures.py",
        "notes": "Detailed focal-shunting controls and electrotonic calibration underlying focused Fig. 8.",
    },
    "measured.detail.asset": {
        "figure": "figS22",
        "path": "figures/supplementary/figure_S22_panels_A-J.pdf",
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
    "path.demand.asset": {
        "figure": "figS29",
        "path": "figures/supplementary/figure_S29_panels_A-C.pdf",
        "generator": "scripts/build_path_necessity_fashion_figure.py",
        "notes": "Controlled Fashion-MNIST branch-conflict task, analytic shared-mode boundary and trained transition.",
    },
}

NEW_PROVENANCE_ENTRIES = {
    "path.demand.outcomes": {
        "record_type": "panel_source",
        "figure": "fig4/figS29",
        "panel": "e/b-c",
        "path": "source_data/path_necessity_fashion/seed_outcomes.csv",
        "generator": "scripts/run_path_necessity_fashion.py",
        "replication_unit": "paired independent training seed (n=20 per branch count)",
        "notes": "All 2,400 confirmatory fits across B=2,4,8, eight conflict doses and five routing conditions.",
    },
    "path.demand.conditions": {
        "record_type": "panel_source",
        "figure": "fig4/figS29",
        "panel": "d,e/b-c",
        "path": "source_data/path_necessity_fashion/condition_summary.csv",
        "generator": "scripts/run_path_necessity_fashion.py",
        "replication_unit": "paired independent training seed (n=20 per branch count)",
        "notes": "Condition means and seed-bootstrap intervals across the frozen conflict doses.",
    },
    "path.demand.contrasts": {
        "record_type": "panel_source",
        "figure": "fig4/figS29",
        "panel": "text/text",
        "path": "source_data/path_necessity_fashion/paired_contrasts.csv",
        "generator": "scripts/run_path_necessity_fashion.py",
        "replication_unit": "paired independent training seed (n=20 per branch count)",
        "notes": "Correct-minus-shared and rank-matched derangement contrasts at every conflict dose.",
    },
    "path.demand.interactions": {
        "record_type": "panel_source",
        "figure": "fig4/figS29",
        "panel": "text/text",
        "path": "source_data/path_necessity_fashion/interaction_summary.csv",
        "generator": "scripts/run_path_necessity_fashion.py",
        "replication_unit": "paired independent training seed (n=20 per branch count)",
        "notes": "Seed-wise correct-minus-shared accuracy slopes on conflict dose.",
    },
    "path.demand.boundaries": {
        "record_type": "panel_source",
        "figure": "fig4/figS29",
        "panel": "c-d/b-c",
        "path": "source_data/path_necessity_fashion/boundary_summary.csv",
        "generator": "scripts/analyze_path_necessity_boundary.py",
        "replication_unit": "paired independent training seed (n=20 per branch count)",
        "notes": "Analytic boundaries and first nonpositive-utility or chance-level trained doses.",
    },
    "path.demand.seedwise.interactions": {
        "record_type": "panel_source",
        "figure": "fig4/figS29",
        "panel": "text/text",
        "path": "source_data/path_necessity_fashion/seedwise_interactions.csv",
        "generator": "scripts/run_path_necessity_fashion.py",
        "replication_unit": "paired independent training seed (n=20 per branch count)",
        "notes": "One prespecified correct-minus-shared accuracy slope on conflict dose per seed and branch count.",
    },
    "path.demand.seedwise.boundaries": {
        "record_type": "panel_source",
        "figure": "figS29",
        "panel": "c",
        "path": "source_data/path_necessity_fashion/boundary_by_seed.csv",
        "generator": "scripts/run_path_necessity_fashion.py",
        "replication_unit": "paired independent training seed (n=20 per branch count)",
        "notes": "First nonpositive initial utility and first chance-level trained dose for every seed.",
    },
    "path.demand.boundary.order": {
        "record_type": "panel_source",
        "figure": "figS29",
        "panel": "c",
        "path": "source_data/path_necessity_fashion/boundary_order.json",
        "generator": "scripts/analyze_path_necessity_boundary.py",
        "replication_unit": "paired independent training seed (n=20)",
        "notes": "Seed-wise strict ordering audit with prespecified right-censoring of unobserved B=2 crossings.",
    },
    "path.demand.plotted.crossings": {
        "record_type": "panel_source",
        "figure": "fig4/figS29",
        "panel": "f/c",
        "path": "source_data/path_necessity_fashion/plotted_crossings.csv",
        "generator": "scripts/analyze_path_necessity_boundary.py",
        "replication_unit": "descriptive mean curve over 20 independent seeds per branch count",
        "notes": "Linearly interpolated chance crossings shown in Supplementary Figure 29c.",
    },
    "path.demand.audit": {
        "record_type": "panel_source",
        "figure": "methods",
        "panel": "path-demand audit",
        "path": "source_data/path_necessity_fashion/audit.json",
        "generator": "scripts/run_path_necessity_fashion.py",
        "replication_unit": "complete 2,400-fit audit",
        "notes": "Finite-value, analytic-gradient, exact-equivalence and interpretation-gate audit.",
    },
    "mnist.ladder.outcomes": {
        "record_type": "panel_source",
        "figure": "fig2",
        "panel": "b",
        "path": "source_data/mnist_feedback_ladder/seed_outcomes.csv",
        "generator": "scripts/collect_mnist_feedback_ladder.py",
        "replication_unit": "paired independent training seed (n=15 per architecture)",
        "notes": "Complete 90-run MNIST strict-scalar, neuron-specific and exact-path feedback ladder.",
    },
    "mnist.ladder.conditions": {
        "record_type": "panel_source",
        "figure": "fig2",
        "panel": "b",
        "path": "source_data/mnist_feedback_ladder/condition_summary.csv",
        "generator": "scripts/collect_mnist_feedback_ladder.py",
        "replication_unit": "paired independent training seed (n=15 per architecture)",
        "notes": "Condition means and paired-seed bootstrap intervals for the strict-scalar MNIST ladder.",
    },
    "mnist.ladder.contrasts": {
        "record_type": "panel_source",
        "figure": "fig2",
        "panel": "text",
        "path": "source_data/mnist_feedback_ladder/paired_contrasts.csv",
        "generator": "scripts/collect_mnist_feedback_ladder.py",
        "replication_unit": "paired independent training seed (n=15 per architecture)",
        "notes": "Strict-scalar-to-neuron-specific and exact-transport paired contrasts for both MNIST architectures.",
    },
    "mnist.ladder.audit": {
        "record_type": "panel_source",
        "figure": "fig2",
        "panel": "text",
        "path": "source_data/mnist_feedback_ladder/audit.json",
        "generator": "scripts/collect_mnist_feedback_ladder.py",
        "replication_unit": "complete 120-fit current-cohort audit",
        "notes": "Completeness, finite-metric, checkpoint, scientific-signature, executable-source, source-environment and no-W&B gates for the 90 plotted fits plus 30 legacy implementation controls.",
    },
    "mnist.scalar.audit.rows": {
        "record_type": "panel_source",
        "figure": "methods",
        "panel": "strict-scalar implementation audit",
        "path": "source_data/mnist_feedback_ladder/strict_scalar_implementation_audit.csv",
        "generator": "scripts/collect_mnist_feedback_ladder.py",
        "replication_unit": "paired independent training seed (n=15 per architecture)",
        "notes": "Seed-paired strict scalar versus legacy matched-width/scalar-fallback implementation audit.",
    },
    "mnist.scalar.audit.contrasts": {
        "record_type": "panel_source",
        "figure": "methods",
        "panel": "strict-scalar implementation audit",
        "path": "source_data/mnist_feedback_ladder/strict_scalar_paired_contrasts.csv",
        "generator": "scripts/collect_mnist_feedback_ladder.py",
        "replication_unit": "paired independent training seed (n=15 per architecture)",
        "notes": "Frozen practical-equivalence analysis for strict scalar minus the legacy implementation.",
    },
    "taskfamily.asset": {
        "record_type": "figure_asset",
        "figure": "fig6",
        "panel": "f-g",
        "path": "figures/main/figure_06.pdf",
        "generator": "scripts/assemble_compact_main_figures.py",
        "replication_unit": "paired independent training seed (n=10)",
        "notes": "Fixed-D3 architecture-by-task-family-by-alignment boundary under exact backpropagation and path-transport LocalCA.",
    },
    "taskfamily.outcomes": {
        "record_type": "panel_source",
        "figure": "fig6",
        "panel": "f-g",
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
        "panel": "text",
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
        "path": "figures/supplementary/figure_S27_panels_A-C.pdf",
        "generator": "scripts/assemble_compact_main_figures.py",
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
        "figure": "figS27",
        "panel": "b-d",
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
    "fig2.asset": "Final six-panel feedback-resolution, ownership and transport-contrast figure.",
    "subtree.factorial.asset": "Final subtree-address bandwidth factorial with matched oracle and derangement controls.",
    "fig3.e": "Direct-presynaptic-type-only sensitivity analysis shown in Supplementary Fig. 20H.",
    "fig4.b": "Cell-level additive, depth-shuffled, and focal-shunt localization shown in Fig. 8B.",
    "fig4.c": "Dose response and focal-depth localization values shown in Fig. 8B--C.",
    "fig4.effects": "Descendant, sister, ancestor, and unrelated category effects shown in Fig. 8B.",
    "fig4.e.cell": "Cell-level exact factor-freeze values underlying the same-voltage driving-force and full-shunt contrast in Fig. 8D.",
    "fig4.e.site": "Site-level factor substitutions supporting the Fig. 8D cell summary and supplementary Shapley analysis.",
}

# The focused eight-panel Figure 2 replaced an older, much larger panel map.
# Keep each numerical record attached to where it is now displayed (or to the
# Methods/Supplement when it is no longer a main-figure result).  Entry IDs are
# deliberately stable so archived analyses remain traceable.
CANONICAL_METADATA = {
    "fig2.a": ("methods", "gradient check"),
    "fig2.b": ("figS2", "e"),
    "fig2.c": ("fig2", "d"),
    "fig2.d": ("figS3", "c"),
    "fig2.d.bp": ("figS3", "c"),
    "fig2.d.runs": ("figS3", "c"),
    "fig2.path_gain": ("fig2", "e"),
    "fig2.init.runs": ("methods", "initialization control"),
    "fig2.init.summary": ("methods", "initialization control"),
    "fig2.init.tests": ("methods", "initialization control"),
    "fig2.feedback_relevance.runs": ("methods", "feedback diagnostic"),
    "fig2.feedback_relevance.summary": ("methods", "feedback diagnostic"),
    "prospective.learning.seeds": ("figS19", "a-c"),
    "prospective.learning.conditions": ("figS19", "a-c"),
    "prospective.learning.contrasts": ("figS19", "a-c"),
    "prospective.routing.runs": ("fig2/figS19", "f/d"),
    "prospective.routing.contrasts": ("fig2/figS19", "f/d"),
    "prospective.dose.conditions": ("figS8", "a-d,g-i"),
    "prospective.dose.contrasts": ("figS8", "e-f"),
    "prospective.fixed_budget.contrasts": ("figS8/figS19", "e-f/f"),
    "subtree.phase1.outcomes": ("figS19", "g,i"),
    "subtree.phase1.gradients": ("figS19", "h"),
    "subtree.phase1.contrasts": ("figS19", "g,i"),
    "subtree.phase1.summary": ("figS19", "g-i"),
    "subtree.phase1.ledger": ("figS19", "text"),
    "subtree.phase1.reproducibility": ("figS19", "text"),
    "subtree.factorial.outcomes": ("fig3/fig5", "f/e,g"),
    "subtree.factorial.conditions": ("fig5", "e"),
    "subtree.factorial.contrasts": ("fig5", "f"),
    "subtree.factorial.ledger": ("fig5", "text"),
    "subtree.factorial.summary": ("fig5", "text"),
    "fashion.outcomes": ("fig2", "c,f"),
    "fashion.conditions": ("fig2", "c"),
    "fashion.contrasts": ("fig2", "f"),
    "fashion.audit": ("fig2", "text"),
    # 2026-08-28 panel-span ground-truthing: every row below was checked
    # against the panel its source is actually drawn in by the current
    # native builders (build_main_figure_0{3,5,6,7,8,9}.py and the S18-S22
    # assemblies).  Rows whose data feed no drawn panel are "text".
    # Old physical-depth Figure 5 panels M-U now live in Supplementary S18.
    "alignmentdose.outcomes": ("figS18", "c-d"),
    "alignmentdose.conditions": ("figS18", "c"),
    "alignmentdose.contrasts": ("figS18", "d-e"),
    "alignmentdose.audit": ("figS18", "text"),
    "remainingphysical.outcomes": ("figS18", "f-j"),
    "remainingphysical.conditions": ("figS18", "f-j"),
    "remainingphysical.contrasts": ("figS18", "h,k"),
    "remainingphysical.audit": ("figS18", "text"),
    # MICrONS anatomy stream: drawn homes are Figure 7 (and its S20 detail);
    # the capture-bound audit backs the topology-matched capacity note.
    "fig3.capture_bound": ("fig7", "text"),
    "fig3.a": ("fig7", "a"),
    "fig3.b-d": ("fig7", "d"),
    "fig3.cell": ("fig7", "text"),
    "fig3.e": ("figS20", "h"),
    "fig3.summary": ("fig7", "d"),
    "reciprocal.operator": ("fig7", "c"),
    "reciprocal.capture": ("fig7", "c"),
    # Physical-depth main figure 6: paired contrasts back prose only.
    "physical.contrasts": ("fig6", "text"),
    "pointcredit.outcomes": ("fig6", "c"),
    "pointcredit.conditions": ("fig6", "c"),
    "pointcredit.contrasts": ("fig6", "text"),
    # Focal-shunting stream: drawn in Figure 8 and its S21 detail.
    "fig4.b": ("fig8", "c"),
    "fig4.c": ("fig8", "c"),
    "fig4.effects": ("fig8", "b"),
    "fig4.d.primary": ("figS21", "f"),
    "fig4.d.scale01": ("figS21", "f"),
    "fig4.d.scale10": ("figS21", "f"),
    "fig4.d.irevm05": ("figS21", "f"),
    "fig4.d.irev0": ("figS21", "f"),
    "fig4.e.cell": ("fig8", "d"),
    "fig4.e.site": ("fig8", "d"),
    "fig4.e.payoffs": ("fig8", "d"),
    "fig4.e.bootstrap": ("fig8", "d"),
    "fig4.e.summary": ("fig8", "d"),
    "fig4.typed.effects": ("figS21", "g"),
    "fig4.typed.cells": ("figS21", "g"),
    "fig4.typed.sites": ("figS21", "g"),
    "fig4.typed.summary": ("figS21", "g"),
    "physical.focal.cells": ("fig8", "f"),
    "extensions.active.cells": ("fig8", "g"),
    "extensions.active.conditions": ("fig8", "e"),
    "extensions.active.contrasts": ("fig8", "g"),
    "extensions.active.acceptance": ("fig8", "text"),
    "extensions.active.summary": ("fig8", "text"),
    # Measured-response stream: drawn in Figure 9 with S22/S5 details;
    # the ch4 task tables also feed the fig3 phase-plane panel g.
    "fig6.a": ("fig9", "c"),
    "fig6.b-d": ("fig3/figS22", "g/d-f"),
    "fig6.e.ch1.targets": ("figS22", "h"),
    "fig6.e.ch2.targets": ("figS22", "h"),
    "fig6.e.ch8.targets": ("figS22", "h"),
    "fig6.e.ch1": ("figS22", "h"),
    "fig6.e.ch2": ("figS22", "h"),
    "fig6.e.ch4": ("figS22", "text"),
    "fig6.e.ch8": ("figS22", "h"),
    "fig6.functional_summary": ("fig9", "text"),
    "fig8.alignment.schematic": ("fig9", "d"),
    "fig8.alignment.curves": ("fig9", "e"),
    "fig8.alignment.cells": ("figS5", "g-h"),
    "fig8.alignment.dictionary": ("figS5", "a"),
    "fig8.alignment.summary": ("fig9", "d-e"),
    "fig8.animal.schematic": ("fig9", "f"),
    "functional.allscans.cells": ("fig9", "c"),
    "functional.allscans.scans": ("fig9", "c"),
    # S17 keeps only old panels A and D (relettered A-B); the common
    # signed-mode table backs the signed-mode prose and Fig. 9G caption.
    "animalcredit.modes": ("figS17", "text"),
    "animalcredit.neurons": ("figS17", "b"),
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

        metadata = CANONICAL_METADATA.get(entry_id)
        if metadata is not None:
            figure, panel = metadata
            if row.get("figure") != figure:
                row["figure"] = figure
                updated += 1
            if row.get("panel") != panel:
                row["panel"] = panel
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
                "panel": detail.get("panel", "all"),
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
                "panel": detail.get("panel", "all"),
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
