#!/usr/bin/env python3
"""Publish generated figure assets under unambiguous manuscript filenames.

Figure-building scripts write descriptive internal filenames to
``figures/generated``.  This script is the single mapping from those internal
artifacts to the exact figure/panel blocks compiled by LaTeX and uploaded to
Overleaf.  Keeping this map explicit prevents legacy names such as
``fig3_microns_topology`` from being mistaken for the manuscript's Figure 3
(that asset is now Figure 7).
"""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GENERATED = ROOT / "figures" / "generated"


# destination path relative to figures/ -> internal generated stem
FIGURE_MAP: dict[str, str] = {
    # Main-manuscript component blocks.  assemble_compact_main_figures.py
    # composes these into exactly one publication-facing PDF per figure.
    "main/figure_01_panels_A-E": "fig1_framework",
    "main/figure_02_panels_A-F": "fig2_feedback",
    "main/figure_02_panels_G-O": "fig_prospective_learning_benefits",
    "main/figure_02_panels_P-Q": "fig_fashion_feedback_ladder",
    "main/figure_03_panels_A-F": "fig_trained_subtree_full_factorial",
    "main/figure_04_panels_A-I": "fig_credit_phase_theory",
    "main/figure_05_panels_A-F": "fig_nonlinear_physical_depth",
    "main/figure_05_panels_G-L": "fig_point_dendrite_credit_controls",
    "main/figure_05_panels_M-O": "fig_physical_alignment_dose",
    "main/figure_05_panels_P-U": "fig_remaining_physical_crossovers",
    "main/figure_06_panels_A-D": "fig_physical_depth_h4_factorial",
    "main/figure_07_panels_A-J": "fig3_microns_topology",
    "main/figure_07_panels_K-L": "fig_capture_per_wire",
    "main/figure_08_panels_A-I": "fig4_focal_shunting",
    "main/figure_08_panels_J-M": "fig_active_focal_extension",
    "main/figure_09_panels_A-H": "fig5_alignment_boundary",
    "main/figure_09_panels_I-J": "fig_fulltree_boundary",
    "main/figure_09_panels_K-N": "fig8_alignment_controlled",
    "main/figure_09_panel_O": "fig_credit_phase_plane",
    # Shared vector teaching schematics used by the publication compositor.
    "components/schematic_fig2_ownership_address": "schematic_fig2_ownership_address",
    "components/schematic_fig3_route_resolution": "schematic_fig3_route_resolution",
    "components/schematic_fig4_credit_operator": "schematic_fig4_credit_operator",
    "components/schematic_fig5_physical_depth": "schematic_fig5_physical_depth",
    "components/schematic_fig6_generalization": "schematic_fig6_generalization",
    "components/schematic_fig7_anatomy_pipeline": "schematic_fig7_anatomy_pipeline",
    "components/schematic_fig8_focal_shunt": "schematic_fig8_focal_shunt",
    "components/schematic_fig9_alignment_boundary": "schematic_fig9_alignment_boundary",
    "components/animal_pairs_wide": "fig_animal_pairs_wide",
    "components/active_dose_main": "fig_active_dose_main",
    "components/physical_controls_main": "fig_physical_controls_main",
    "components/figure5_schematics": "fig_figure5_schematics",
    "components/figure5_architecture_schematic": "fig_figure5_architecture_schematic",
    # Final-aspect replacements for main panels that previously retained
    # diagnostic-style layouts or crowded teaching graphics.
    "components/main_phase_plane_clean": "fig_main_phase_plane_clean",
    "components/main_physical_task_schematic": "fig_main_physical_task_schematic",
    "components/main_figure6_redesigned": "fig_main_figure6_redesigned",
    "components/main_mapped_reconstruction": "fig_main_mapped_reconstruction",
    "components/main_ancestry_addresses": "fig_main_ancestry_addresses",
    "components/main_wire_efficiency": "fig_main_wire_efficiency",
    "components/main_cross_animal": "fig_main_cross_animal",
    "components/main_focal_schematic": "fig_main_focal_schematic",
    "components/main_structure_function_summary": "fig_main_structure_function_summary",
    # Supplementary Information: generated source blocks S1--S29.  The
    # compositor adds compact S18--S19 blocks and S28 from selected diagnostic
    # panels.
    "supplementary/figure_S01_panels_A-E": "fig3_mechanistic_evidence",
    "supplementary/figure_S02_panels_A-E": "fig4_competence_regime",
    "supplementary/figure_S03_panels_A-D": "fig5_rule_feedback_controls",
    "supplementary/figure_S04_panels_A-I": "fig3_regular_tree_regimes",
    "supplementary/figure_S05_panels_A-H": "fig6_alignment_controlled",
    "supplementary/figure_S06_panels_A-D": "fig_branch_interference_prediction",
    "supplementary/figure_S07_panels_A-D": "fig_spatial_topology_audit",
    "supplementary/figure_S08_panels_A-I": "fig_prospective_fixed_budget_depth",
    "supplementary/figure_S09_panels_A-D": "fig_prospective_credit_mechanism",
    "supplementary/figure_S10_panels_A-H": "fig_v661_robustness",
    "supplementary/figure_S11_panels_A-C": "fig_focal_selectivity_matrix",
    "supplementary/figure_S12_panels_A-J": "fig_microns_inhibitory_routes",
    "supplementary/figure_S13_panels_A-F": "fig_positive_conductance_reliability",
    "supplementary/figure_S14_panels_A-F": "fig_same_span_coefficient_learning",
    "supplementary/figure_S15_panels_A-D": "fig_supp_nonlinear_depth_calibration",
    "supplementary/figure_S16_panels_A-D": "fig_interior_optimum",
    "supplementary/figure_S17_panels_A-D": "fig_animal_credit_supplement",
    "supplementary/figure_S20_panels_A-J": "fig3_microns_topology_detailed",
    "supplementary/figure_S21_panels_A-I": "fig4_focal_shunting_detailed",
    "supplementary/figure_S22_panels_A-J": "fig5_alignment_boundary_detailed",
    "supplementary/figure_S23_panels_A-C": "fig_trained_partition_residual",
    "supplementary/figure_S24_panels_A-D": "fig_adaptive_conductance_reliability",
    "supplementary/figure_S25_panels_A-D": "fig_irregular_tree_wavelets",
    "supplementary/figure_S26_panels_A-D": "fig_physical_depth_clean_source_replication",
    # Source component retained for provenance, not compiled separately.
    "components/supplementary_animal_credit_component": "fig_francioni_signed_credit_validation",
    # Completed but currently superseded displays, retained for provenance.
    "archive_superseded/prospective_inhibition_dose": "fig_prospective_inhibition_dose",
    "archive_superseded/prospective_topology_routing": "fig_prospective_topology_routing",
    "archive_superseded/trained_subtree_address": "fig_trained_subtree_address",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    missing: list[Path] = []
    copied = 0
    for destination_stem, source_stem in FIGURE_MAP.items():
        # Vector PDF is the sole canonical format.  Raster previews are local
        # build products and are intentionally not copied into the repository.
        for suffix in (".pdf",):
            source = GENERATED / f"{source_stem}{suffix}"
            destination = ROOT / "figures" / f"{destination_stem}{suffix}"
            if not source.exists():
                # A clean checkout contains publication-facing assets but may
                # omit internal generated duplicates.  That is sufficient to
                # compile; after a generator is rerun, its new output replaces
                # the canonical copy here.
                if not destination.exists():
                    missing.append(source)
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists() and sha256(destination) == sha256(source):
                continue
            shutil.copy2(source, destination)
            copied += 1

    if missing:
        joined = "\n".join(f"  - {path.relative_to(ROOT)}" for path in missing)
        raise FileNotFoundError(f"Missing generated figure assets:\n{joined}")

    print(f"Canonical figure assets are current ({copied} files updated).")


if __name__ == "__main__":
    main()
