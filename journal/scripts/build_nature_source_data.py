#!/usr/bin/env python3
"""Build a submission-facing Nature Source Data package.

Only allow-listed files are copied. The script never modifies manuscript
source, figures, analysis outputs, or the panel-level provenance manifest.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import tempfile
import zipfile
from dataclasses import dataclass, replace
from datetime import date
from pathlib import Path

from export_figure2d_seed_data import read_rows as read_figure2d_rows
from export_figure2d_seed_data import verify_rows as verify_figure2d_rows


JOURNAL = Path(__file__).resolve().parents[1]
SUBMISSION = JOURNAL / "submission"
DEFAULT_STAGE = SUBMISSION / "nature_source_data"
DEFAULT_ZIP = SUBMISSION / "Source_Data.zip"

SANITIZED_DROP_COLUMNS = {
    "source_data/mnist_feedback_ladder/seed_outcomes.csv": {"run_dir"},
    "source_data/mnist_feedback_ladder/strict_scalar_implementation_audit.csv": {
        "run_dir_strict",
        "run_dir_legacy",
    },
    "source_data/inherited_neurips/error_field_decomposition_runs.csv": {"run_dir"},
    "source_data/functional_topology_all_scans/scan_metrics.csv": {"source_summary"},
    "source_data/physical_depth_h4_factorial/seed_outcomes.csv": {"run_dir"},
    "source_data/physical_depth_clean_source_replication/seed_outcomes.csv": {
        "run_dir"
    },
}

SANITIZED_JSON_PATHS = {
    "source_data/cifar10_additive_feedback_ladder_confirmatory/summary.json",
    "source_data/physical_depth_h4_factorial/audit.json",
    "source_data/physical_depth_clean_source_replication/audit.json",
    "source_data/task_family_alignment/audit.json",
}


@dataclass(frozen=True)
class SourceFile:
    figure: str
    panels: str
    source: str
    destination: str
    role: str
    independent_unit: str
    status: str
    notes: str


FILES = (
    SourceFile(
        "Figure 2",
        "b",
        "source_data/mnist_feedback_ladder/seed_outcomes.csv",
        "Figure_2/Fig2b_mnist_feedback_seed_outcomes.csv",
        "paired independent training-seed values",
        "paired training seed (n=15 per architecture)",
        "executable-source matched cohort",
        "Complete 90-run MNIST strict-scalar, neuron-specific and exact-path feedback ladder.",
    ),
    SourceFile(
        "Figure 2",
        "b",
        "source_data/mnist_feedback_ladder/condition_summary.csv",
        "Figure_2/Fig2b_mnist_feedback_condition_summary.csv",
        "condition means and paired-seed bootstrap intervals",
        "paired training seed (n=15 per architecture)",
        "executable-source matched cohort",
        "Condition summaries for the complete matched MNIST ladder.",
    ),
    SourceFile(
        "Figure 2",
        "b",
        "source_data/mnist_feedback_ladder/paired_contrasts.csv",
        "Figure_2/Fig2b_mnist_feedback_paired_contrasts.csv",
        "paired training-seed contrasts",
        "paired training seed (n=15 per architecture)",
        "executable-source matched cohort",
        "Neuron-identity and exact-transport contrasts for both MNIST architectures.",
    ),
    SourceFile(
        "Figure 2",
        "text",
        "source_data/mnist_feedback_ladder/strict_scalar_implementation_audit.csv",
        "Figure_2/Text_strict_scalar_implementation_audit.csv",
        "paired strict-scalar and legacy implementation outcomes",
        "paired training seed (n=15 per architecture)",
        "current matched implementation audit",
        "Seed-paired audit of the strict scalar used in the figure against the legacy scalar-fallback condition.",
    ),
    SourceFile(
        "Figure 2",
        "text",
        "source_data/mnist_feedback_ladder/strict_scalar_paired_contrasts.csv",
        "Figure_2/Text_strict_scalar_paired_contrasts.csv",
        "paired implementation contrasts and practical-equivalence decision",
        "paired training seed (n=15 per architecture)",
        "current matched implementation audit",
        "Strict-scalar minus legacy paired effects under the analysis rule frozen before outcome inspection.",
    ),
    SourceFile(
        "Methods",
        "gradient check",
        "source_data/figure2/exact_gradient_reconstruction_runs.csv",
        "Methods/exact_gradient_reconstruction_runs.csv",
        "independent diagnostic runs",
        "diagnostic run (n=100)",
        "current",
        "Exact analytic-gradient reconstruction against automatic differentiation.",
    ),
    SourceFile(
        "Supplementary Figure 2",
        "e",
        "source_data/figure2/feedback_accuracy_runs.csv",
        "Supplementary_Figure_2/SuppFig2e_legacy_feedback_accuracy_runs.csv",
        "independent training-seed values",
        "paired training seed (n=15 per architecture)",
        "current clean cohort",
        "Complete clean 15-seed cohort generated under the frozen current-code design and validated for configuration, checkpoints, and hashes.",
    ),
    SourceFile(
        "Figure 2",
        "d",
        "source_data/figure2/feedback_gradient_runs.csv",
        "Figure_2/Fig2d_feedback_gradient_runs.csv",
        "independent checkpoint/seed values",
        "paired trained checkpoint / seed (n=15 per architecture)",
        "current clean cohort",
        "Fixed-checkpoint diagnostics for both feedback fields at all 30 checkpoints from the validated clean training cohort.",
    ),
    SourceFile(
        "Supplementary Figure 3",
        "c",
        "source_data/figure2/exact_transport_and_backprop_runs.csv",
        "Supplementary_Figure_3/SuppFig3c_exact_transport_and_backprop_runs.csv",
        "independent training-seed values",
        "training seed (n=5 per condition)",
        "current",
        "Twenty exact-transport factorial runs and five same-architecture backpropagation references, exported directly from archived final.json files and verified against the frozen summaries.",
    ),
    SourceFile(
        "Supplementary Figure 3",
        "c",
        "source_data/figure2/exact_transport_factorial_summary.csv",
        "Supplementary_Figure_3/SuppFig3c_exact_transport_factorial_summary.csv",
        "derived condition summaries",
        "training seed (n=5 per condition)",
        "current derived summary",
        "Four exact-transport condition summaries verified against the included independent-seed table.",
    ),
    SourceFile(
        "Supplementary Figure 3",
        "c",
        "source_data/figure2/backprop_summary.csv",
        "Supplementary_Figure_3/SuppFig3c_backprop_summary.csv",
        "derived reference summary",
        "training seed (n=5)",
        "current derived summary",
        "Matched backpropagation summary verified against the included independent-seed table.",
    ),
    SourceFile(
        "Figure 2",
        "e",
        "source_data/figure2/path_gain_dispersion_ladder_runs.csv",
        "Figure_2/Fig2e_path_gain_dispersion_ladder_runs.csv",
        "independent checkpoint/seed values",
        "paired training seed (n=15 per architecture)",
        "current fixed-checkpoint diagnostic on the Figure 2 ladder",
        "Within-neuron transported-error CV at the exact-path checkpoints of the MNIST ladder; run, configuration, and checkpoint hashes are included.",
    ),
    SourceFile(
        "Supplementary Figure 1",
        "a",
        "source_data/figure2/path_gain_cv_runs.csv",
        "Supplementary_Figure_1/SuppFig1a_path_gain_cv_runs.csv",
        "independent checkpoint/seed values",
        "paired training seed (n=5 per architecture)",
        "current separate diagnostic cohort",
        "Path-gain coefficient of variation across compartments within examples at five inhibitory synapses per branch; run, configuration, and checkpoint hashes are included.",
    ),
    SourceFile(
        "Methods",
        "initialization control",
        "source_data/figure2/initialization_factorial_runs.csv",
        "Methods/initialization_factorial_runs.csv",
        "independent training-seed values",
        "paired training seed (n=15 per architecture and initialization policy)",
        "current archived factorial",
        "Complete two-architecture by two-initialization-policy factorial underlying the operating-point control reported in the main text.",
    ),
    SourceFile(
        "Methods",
        "initialization control",
        "source_data/figure2/initialization_factorial_summary.csv",
        "Methods/initialization_factorial_summary.csv",
        "derived condition summaries",
        "paired training seed (n=15 per architecture and initialization policy)",
        "current derived summary",
        "Condition means, sample standard deviations, standard errors, and sample counts derived from the included run-level table.",
    ),
    SourceFile(
        "Methods",
        "initialization control",
        "source_data/figure2/initialization_factorial_paired_tests.csv",
        "Methods/initialization_factorial_paired_tests.csv",
        "derived paired comparisons",
        "paired training seed (n=15)",
        "current derived support",
        "Paired architecture, policy, and interaction comparisons derived from the included run-level factorial.",
    ),
    SourceFile(
        "Figure 3",
        "text",
        "source_data/theory/credit_capture_bound_verification.json",
        "Figure_3/Text_credit_capture_bound_verification.json",
        "deterministic numerical validation",
        "simulated positive-definite quadratic loss (n=10,000)",
        "current",
        "Numerical validation of the projected-gradient descent guarantee and capture identity.",
    ),
    SourceFile(
        "Figure 3",
        "a-c",
        "source_data/figure3/segment_metrics.csv",
        "Figure_3/Fig3a-c_segment_metrics.csv",
        "displayed reconstruction coordinates and segment measurements",
        "biological cell; segments nested within the displayed cell",
        "current",
        "Compressed skeleton and route-domain measurements for the displayed reconstruction.",
    ),
    SourceFile(
        "Figure 3",
        "d-g",
        "source_data/figure3/routing_capacity_curves.csv.gz",
        "Figure_3/Fig3d-g_routing_capacity_curves.csv.gz",
        "cell-by-stream source values",
        "biological cell; Monte Carlo streams nested within cell (n=8 cells)",
        "current",
        "Field capture and wiring values across channel counts and control dictionaries.",
    ),
    SourceFile(
        "Figure 3",
        "a-g",
        "source_data/figure3/cell_metrics.csv",
        "Figure_3/Fig3a-g_cell_metrics.csv",
        "independent-cell values",
        "biological cell (n=8)",
        "current",
        "Cell-level anatomy and summary measurements.",
    ),
    SourceFile(
        "Figure 3",
        "h",
        "source_data/figure3/typed_only_compression_curves.csv",
        "Figure_3/Fig3h_typed_only_compression_curves.csv",
        "independent-cell values",
        "biological cell (n=8)",
        "current",
        "Sensitivity analysis restricted to directly typed presynaptic partners.",
    ),
    SourceFile(
        "Figure 3",
        "d-h",
        "source_data/figure3/routing_capacity_summary.json",
        "Figure_3/Fig3d-h_routing_capacity_summary.json",
        "derived statistical summary",
        "biological cell (n=8)",
        "current",
        "Frozen hierarchical estimates and intervals.",
    ),
    SourceFile(
        "Figure 4",
        "b",
        "source_data/figure4/category_effects.csv",
        "Figure_4/Fig4b_category_effects.csv",
        "site- and category-level values",
        "biological cell; focal sites nested within cell (n=8 cells)",
        "current",
        "Descendant, sister, ancestor, and unrelated gradient effects.",
    ),
    SourceFile(
        "Figure 4",
        "c",
        "source_data/figure4/cell_primary_contrasts.csv",
        "Figure_4/Fig4c_cell_primary_contrasts.csv",
        "independent-cell values",
        "biological cell (n=8)",
        "current",
        "Cell-level shunt, additive, and depth-stratified relation-reassignment localization values.",
    ),
    SourceFile(
        "Figure 4",
        "d-e",
        "source_data/figure4/focal_localization.csv",
        "Figure_4/Fig4d-e_focal_localization.csv",
        "site-level values",
        "biological cell; 101 focal sites nested within 8 cells",
        "current",
        "Perturbation dose response and per-site localization values.",
    ),
    SourceFile(
        "Figure 4",
        "g",
        "source_data/figure4/direct_typed_category_effects.csv",
        "Figure_4/Fig4g_direct_typed_category_effects.csv",
        "site- and category-level sensitivity values",
        "reconstructed cell; 55 focal sites nested within 8 cells",
        "current direct-type-only sensitivity",
        "Descendant, sister, ancestor, and unrelated effects after removing proxy-classified contacts.",
    ),
    SourceFile(
        "Figure 4",
        "g",
        "source_data/figure4/direct_typed_cell_primary_contrasts.csv",
        "Figure_4/Fig4g_direct_typed_cell_primary_contrasts.csv",
        "cell-level sensitivity values",
        "reconstructed cell (n=8; one animal)",
        "current direct-type-only sensitivity",
        "Cell-level shunt, additive, and reassigned-relation localization values after removing proxy-classified contacts.",
    ),
    SourceFile(
        "Figure 4",
        "g",
        "source_data/figure4/direct_typed_focal_localization.csv",
        "Figure_4/Fig4g_direct_typed_focal_localization.csv",
        "site-level sensitivity values",
        "reconstructed cell; 55 focal sites nested within 8 cells",
        "current direct-type-only sensitivity",
        "Per-site dose response after restricting both synaptic weighting and focal sites to directly typed inputs.",
    ),
    SourceFile(
        "Figure 4",
        "g",
        "source_data/figure4/direct_typed_summary.json",
        "Figure_4/Fig4g_direct_typed_summary.json",
        "derived sensitivity summary",
        "reconstructed cell (n=8; one animal)",
        "current direct-type-only sensitivity",
        "Frozen estimand, interval, sign count, numerical checks, and analysis parameters for the direct-type-only control.",
    ),
    SourceFile(
        "Figure 4",
        "f",
        "source_data/figure4/summary.json",
        "Figure_4/Fig4f_primary_summary.json",
        "derived statistical summary",
        "biological cell (n=8)",
        "current",
        "Primary conductance-scale and reversal-potential condition.",
    ),
    SourceFile(
        "Figure 4",
        "f",
        "source_data/figure4/scale0p1_summary.json",
        "Figure_4/Fig4f_conductance_scale_0p1_summary.json",
        "derived statistical summary",
        "biological cell (n=8)",
        "current",
        "Conductance-scale sensitivity.",
    ),
    SourceFile(
        "Figure 4",
        "f",
        "source_data/figure4/scale1p0_summary.json",
        "Figure_4/Fig4f_conductance_scale_1p0_summary.json",
        "derived statistical summary",
        "biological cell (n=8)",
        "current",
        "Conductance-scale sensitivity.",
    ),
    SourceFile(
        "Figure 4",
        "f",
        "source_data/figure4/irevm0p5_summary.json",
        "Figure_4/Fig4f_inhibitory_reversal_minus0p5_summary.json",
        "derived statistical summary",
        "biological cell (n=8)",
        "current",
        "Inhibitory-reversal-potential sensitivity.",
    ),
    SourceFile(
        "Figure 4",
        "f",
        "source_data/figure4/irev0_summary.json",
        "Figure_4/Fig4f_inhibitory_reversal_0_summary.json",
        "derived statistical summary",
        "biological cell (n=8)",
        "current",
        "Inhibitory-reversal-potential sensitivity.",
    ),
    SourceFile(
        "Figure 4",
        "h",
        "source_data/focal_decomposition/cell_shapley.csv",
        "Figure_4/Fig4h_cell_shapley.csv",
        "independent-cell values",
        "biological cell (n=8)",
        "current",
        "Cell-level Shapley allocations plotted in the panel.",
    ),
    SourceFile(
        "Figure 4",
        "h",
        "source_data/focal_decomposition/site_shapley.csv",
        "Figure_4/Fig4h_site_shapley.csv",
        "site-level support values",
        "biological cell; 101 focal sites nested within 8 cells",
        "current",
        "Site-level allocations underlying the independent-cell values.",
    ),
    SourceFile(
        "Figure 4",
        "h",
        "source_data/focal_decomposition/site_decomposition.csv",
        "Figure_4/Fig4h_site_factor_payoffs.csv",
        "site-level factor-substitution payoffs",
        "biological cell; 101 focal sites nested within 8 cells",
        "current",
        "Four model payoffs from which the two-factor Shapley allocation is computed.",
    ),
    SourceFile(
        "Figure 4",
        "h",
        "source_data/focal_decomposition/shapley_bootstrap_draws.csv.gz",
        "Figure_4/Fig4h_shapley_bootstrap_draws.csv.gz",
        "derived hierarchical bootstrap draws",
        "biological cell resampled first; sites resampled within selected cells",
        "current derived support",
        "Frozen draws underlying the displayed 95% intervals; not additional biological replicates.",
    ),
    SourceFile(
        "Figure 4",
        "h",
        "source_data/focal_decomposition/shapley_summary.json",
        "Figure_4/Fig4h_shapley_summary.json",
        "derived statistical summary",
        "biological cell (n=8)",
        "current",
        "Estimands, intervals, sign counts, and exact reconciliation checks.",
    ),
    SourceFile(
        "Figure 6",
        "a-c",
        "source_data/figure5/functional_target_metrics.csv",
        "Figure_6/Fig6a-c_functional_target_metrics.csv",
        "independent-target values",
        "target cell (n=7)",
        "current",
        "Functional-similarity and shared-ancestry relation after target-level controls.",
    ),
    SourceFile(
        "Figure 6",
        "a-c",
        "source_data/figure5/functional_summary.json",
        "Figure_6/Fig6a-c_functional_summary.json",
        "derived statistical summary",
        "target cell (n=7)",
        "current",
        "Frozen target-level partial-correlation summary.",
    ),
    SourceFile(
        "Figure 6",
        "d-g",
        "source_data/figure5/task_target_method_means_ch4.csv",
        "Figure_6/Fig6d-g_task_target_method_means_ch4.csv",
        "independent-target values",
        "target cell; ten fixed stimulus splits averaged within target (n=7)",
        "current",
        "Four-channel primary task-derived capture and learning values.",
    ),
    SourceFile(
        "Figure 6",
        "h",
        "source_data/figure5/task_target_method_means_ch1.csv",
        "Figure_6/Fig6h_task_target_method_means_ch1.csv",
        "independent-target values",
        "target cell; ten fixed stimulus splits averaged within target (n=7)",
        "current",
        "One-channel sensitivity values.",
    ),
    SourceFile(
        "Figure 6",
        "h",
        "source_data/figure5/task_target_method_means_ch2.csv",
        "Figure_6/Fig6h_task_target_method_means_ch2.csv",
        "independent-target values",
        "target cell; ten fixed stimulus splits averaged within target (n=7)",
        "current",
        "Two-channel sensitivity values.",
    ),
    SourceFile(
        "Figure 6",
        "h",
        "source_data/figure5/task_target_method_means_ch8.csv",
        "Figure_6/Fig6h_task_target_method_means_ch8.csv",
        "independent-target values",
        "target cell; ten fixed stimulus splits averaged within target (n=7)",
        "current",
        "Eight-channel sensitivity values.",
    ),
    *(
        SourceFile(
            "Figure 6",
            "h",
            f"source_data/figure5/task_summary_ch{channels}.json",
            f"Figure_6/Fig6h_task_summary_ch{channels}.json",
            "derived statistical summary",
            "target cell (n=7)",
            "current",
            f"Frozen {channels}-channel morphology-minus-shuffle summary.",
        )
        for channels in (1, 2, 4, 8)
    ),
    SourceFile(
        "Figure 5",
        "b-e",
        "source_data/microns_v661_replication/routing_capacity_20stream.csv.gz",
        "Figure_5/Fig5b-e_routing_capacity_20stream.csv.gz",
        "cell-by-stream source values",
        "biological cell; 20 Monte Carlo streams nested within cell (n=47 cells)",
        "current",
        "Frozen public-v661 routing analysis.",
    ),
    SourceFile(
        "Figure 5",
        "b-e",
        "source_data/microns_v661_replication/supp_figure_routing_curves.csv",
        "Figure_5/Fig5b-e_routing_curves_cell_means.csv",
        "independent-cell plotting values",
        "biological cell (n=47)",
        "current",
        "Monte Carlo streams averaged within cell and condition.",
    ),
    SourceFile(
        "Figure 5",
        "b-e",
        "source_data/microns_v661_replication/routing_capacity_20stream_summary.json",
        "Figure_5/Fig5b-e_routing_capacity_summary.json",
        "derived statistical summary",
        "biological cell (n=47)",
        "current",
        "Final 20-stream hierarchical estimates and comparisons.",
    ),
    SourceFile(
        "Figure 5",
        "f-g",
        "source_data/microns_v661_replication/supp_figure_focal_cells.csv",
        "Figure_5/Fig5f-g_focal_cells.csv",
        "independent-cell values",
        "biological cell (n=45 shunt/additive; n=40 topology shuffle)",
        "current",
        "Cell-averaged focal localization contrasts.",
    ),
    SourceFile(
        "Figure 5",
        "a-h",
        "source_data/microns_v661_replication/cohort_manifest.csv",
        "Figure_5/Fig5_cohort_manifest.csv",
        "cohort and source provenance",
        "candidate reconstructed cell (n=55)",
        "current",
        "All candidates, pilot-cohort exclusions, source URLs, and file hashes.",
    ),
    SourceFile(
        "Figure 5",
        "a-h",
        "source_data/microns_v661_replication/prospective_endpoint_exclusion_log.csv",
        "Figure_5/Fig5_endpoint_exclusion_log.csv",
        "prospective inclusion/exclusion record",
        "candidate reconstructed cell (n=55)",
        "current",
        "Endpoint-specific inclusion and exclusion decisions.",
    ),
)


def shifted_legacy_file(item: SourceFile) -> SourceFile:
    """Map the original seven-figure manuscript sources to the current order."""

    mapping = {
        "Figure 3": "Figure 7",
        "Figure 4": "Figure 8",
        "Figure 5": "Supplementary Figure 10",
        "Figure 6": "Figure 10",
        "Figure 7": "Supplementary Figure 5",
    }
    new_figure = mapping.get(item.figure)
    if new_figure is None:
        return item
    old_dir = item.figure.replace(" ", "_") + "/"
    new_dir = new_figure.replace(" ", "_") + "/"
    old_number = item.figure.split()[-1]
    new_number = new_figure.split()[-1]
    destination = item.destination.replace(old_dir, new_dir, 1)
    new_prefix = (
        f"SuppFig{new_number}"
        if new_figure.startswith("Supplementary")
        else f"Fig{new_number}"
    )
    destination = destination.replace(f"Fig{old_number}", new_prefix, 1)
    return replace(
        item,
        figure=new_figure,
        destination=destination,
    )


EXPANDED_FILES = (
    SourceFile(
        "Figure 3",
        "a",
        "source_data/regular_tree_regimes/competence_summary.csv",
        "Figure_3/Fig3a_task_competence.csv",
        "condition summary",
        "independent training seed (n=5 per condition)",
        "inherited validated control",
        "MNIST, Fashion-MNIST, and context-gating task controls.",
    ),
    SourceFile(
        "Figure 3",
        "b",
        "source_data/regular_tree_regimes/inhibition_dose_summary.csv",
        "Figure_3/Fig3b_inhibition_dose.csv",
        "condition summary",
        "independent training seed (n=5 per condition)",
        "inherited validated control",
        "Task-dependent inhibitory-synapse-count sweep.",
    ),
    SourceFile(
        "Figure 3",
        "c",
        "source_data/regular_tree_regimes/depth_scaling_summary.csv",
        "Figure_3/Fig3c_depth_scaling.csv",
        "condition summary",
        "independent training seed (n=5 per condition)",
        "inherited validated control",
        "Dendritic-depth stress under local learning and backpropagation.",
    ),
    SourceFile(
        "Figure 3",
        "d",
        "source_data/regular_tree_regimes/broadcast_noise_summary.csv",
        "Figure_3/Fig3d_broadcast_noise.csv",
        "condition summary",
        "independent training seed (n=5 per condition)",
        "inherited validated control",
        "Gaussian teaching-signal-noise stress.",
    ),
    SourceFile(
        "Figure 3",
        "e",
        "source_data/regular_tree_regimes/rule_family_summary_source.csv",
        "Figure_3/Fig3e_rule_family.csv",
        "condition summary",
        "independent training seed (n=3 per condition)",
        "inherited descriptive control",
        "Within-shunting three-, four-, and five-factor comparison.",
    ),
    SourceFile(
        "Figure 3",
        "f",
        "source_data/regular_tree_regimes/error_source_runs.csv",
        "Figure_3/Fig3f_error_source_runs.csv",
        "independent training-seed values",
        "independent training seed (n=3 per condition)",
        "inherited descriptive control",
        "Somatic teaching error versus an unmatched local quantity.",
    ),
    SourceFile(
        "Figure 3",
        "g",
        "source_data/regular_tree_regimes/exact_transport_summary.csv",
        "Figure_3/Fig3g_exact_transport_controls.csv",
        "condition summary",
        "independent training seed (n=5 per condition)",
        "inherited validated control",
        "Exact-transport mechanism controls.",
    ),
    SourceFile(
        "Figure 3",
        "g",
        "source_data/regular_tree_regimes/backprop_reference_summary.csv",
        "Figure_3/Fig3g_backprop_reference.csv",
        "reference summary",
        "independent training seed (n=5)",
        "inherited validated control",
        "Matched backpropagation reference.",
    ),
    SourceFile(
        "Figure 3",
        "g",
        "source_data/regular_tree_regimes/additive_controls_summary.csv",
        "Figure_3/Fig3g_additive_controls.csv",
        "condition summary",
        "independent training seed (n=5 per condition)",
        "inherited validated control",
        "Additive gain and normalization controls.",
    ),
    SourceFile(
        "Figure 3",
        "g",
        "source_data/regular_tree_regimes/reactivation_controls_summary.csv",
        "Figure_3/Fig3g_reactivation_controls.csv",
        "condition summary",
        "independent training seed (n=5 per condition)",
        "inherited validated control",
        "Identity and learned-reactivation controls.",
    ),
    SourceFile(
        "Figure 3",
        "h",
        "source_data/regular_tree_regimes/noise_feedback_ladder_summary.csv",
        "Figure_3/Fig3h_noise_feedback_ladder.csv",
        "condition summary",
        "independent training seed (n=5 per condition)",
        "inherited validated control",
        "Scalar, path-propagated, low-rank, and exact feedback.",
    ),
    SourceFile(
        "Figure 3",
        "i",
        "source_data/regular_tree_regimes/cifar10_control_ladder_runs.csv",
        "Figure_3/Fig3i_cifar10_control_runs.csv",
        "independent training-seed values",
        "independent training seed (n=5 per condition)",
        "inherited harder-data control",
        "Flattened CIFAR-10 mechanism control; not a competitive vision benchmark.",
    ),
    SourceFile(
        "Figure 5",
        "a",
        "source_data/microns_inhibitory_routes/cohort_overlap.csv",
        "Figure_5/Fig5a_cohort_overlap.csv",
        "frozen cohort join",
        "target cell (n=20)",
        "current",
        "Complete stable-ID overlap between the v661 reconstruction cohort and the v795 inhibitory census.",
    ),
    SourceFile(
        "Figure 5",
        "b-c,g-h",
        "source_data/microns_inhibitory_routes/mapped_inhibitory_contacts.csv",
        "Figure_5/Fig5b-c_g-h_mapped_inhibitory_contacts.csv",
        "mapped contact-level values",
        "target cell; contacts nested within 20 target cells",
        "current",
        "Mapped known inhibitory contacts, published presynaptic classes, descendant domains, and clump representatives.",
    ),
    SourceFile(
        "Figure 5",
        "d-f",
        "source_data/microns_inhibitory_routes/inhibitory_connection_topology.csv",
        "Figure_5/Fig5d-f_inhibitory_connection_topology.csv",
        "connection-level matched values",
        "target cell; 402 multi-clump connections nested within 20 target cells",
        "current",
        "Same-axon and matched-location tree distance, shared path, and descendant-domain overlap.",
    ),
    SourceFile(
        "Figure 5",
        "d-j",
        "source_data/microns_inhibitory_routes/target_metrics.csv",
        "Figure_5/Fig5d-j_target_metrics.csv",
        "independent-target values",
        "target cell (n=20)",
        "current",
        "Target-level values used for inference across topology, class, placement, and capacity endpoints.",
    ),
    SourceFile(
        "Figure 5",
        "i-j",
        "source_data/microns_inhibitory_routes/route_capture_curves.csv",
        "Figure_5/Fig5i-j_route_capture_curves.csv",
        "cell-by-budget route-capacity values",
        "target cell (n=20)",
        "current",
        "Actual inhibitory-site dictionary capacity and matched structural controls.",
    ),
    SourceFile(
        "Figure 5",
        "a-j",
        "source_data/microns_inhibitory_routes/summary.json",
        "Figure_5/Fig5_summary.json",
        "derived statistical summary",
        "target cell (n=20)",
        "current",
        "Frozen mapping, cohort, effect-size, interval, sign-count, and test summary.",
    ),
    SourceFile(
        "Figure 8",
        "b-d",
        "source_data/alignment_controlled/alignment_controlled_runs.csv.gz",
        "Figure_8/Fig8b-d_alignment_controlled_runs.csv.gz",
        "cell-by-stream source values",
        "reconstructed cell; Monte Carlo streams nested within cell (n=8 cells)",
        "current controlled sufficiency test",
        "Source values across imposed alignment, routing dictionary, and learning endpoint.",
    ),
    SourceFile(
        "Figure 8",
        "b-c",
        "source_data/alignment_controlled/alignment_controlled_curves.csv",
        "Figure_8/Fig8b-c_alignment_controlled_curves.csv",
        "derived plotting summary",
        "reconstructed cell (n=8)",
        "current controlled sufficiency test",
        "Displayed means and hierarchical intervals.",
    ),
    SourceFile(
        "Figure 8",
        "d",
        "source_data/alignment_controlled/cell_alignment_metrics.csv",
        "Figure_8/Fig8d_cell_alignment_metrics.csv",
        "independent-cell values",
        "reconstructed cell (n=8)",
        "current controlled sufficiency test",
        "Cell-level capture and 20-step learning progress.",
    ),
    SourceFile(
        "Figure 8",
        "a-d",
        "source_data/alignment_controlled/dictionary_audit.csv",
        "Figure_8/Fig8a-d_dictionary_audit.csv",
        "dictionary construction audit",
        "reconstructed cell; streams nested within cell",
        "current controlled sufficiency test",
        "Requested channels, effective rank, and nonzero counts.",
    ),
    SourceFile(
        "Figure 8",
        "a-d",
        "source_data/alignment_controlled/summary.json",
        "Figure_8/Fig8a-d_alignment_summary.json",
        "derived statistical summary",
        "reconstructed cell (n=8)",
        "current controlled sufficiency test",
        "Frozen design, endpoint contrasts, and scope boundaries.",
    ),
    SourceFile(
        "Figure 8",
        "f",
        "source_data/animal_learning_francioni/animal_signed_contrasts.csv",
        "Figure_8/Fig8f_animal_signed_contrasts.csv",
        "independent-animal values",
        "animal (n=6)",
        "current external reanalysis",
        "Animal-level P+ and P- contrasts extracted from the published source workbook.",
    ),
    SourceFile(
        "Figure 8",
        "g",
        "source_data/animal_learning_francioni/animal_common_signed_modes.csv",
        "Figure_8/Fig8g_common_signed_modes.csv",
        "animal-level mode decomposition",
        "animal (n=6)",
        "current external reanalysis",
        "Common and signed two-coordinate contrast modes.",
    ),
    SourceFile(
        "Figure 8",
        "h",
        "source_data/animal_learning_francioni/neuron_sd_residual_distributions.csv",
        "Figure_8/Fig8h_neuron_residual_distributions.csv",
        "descriptive neuron-level values",
        "animal is inferential unit; neurons nested within animals",
        "current external reanalysis",
        "Neuron-level values are shown descriptively and are not treated as independent replicates.",
    ),
    SourceFile(
        "Figure 8",
        "f-h",
        "source_data/animal_learning_francioni/summary.json",
        "Figure_8/Fig8f-h_animal_summary.json",
        "derived statistical summary",
        "animal (n=6)",
        "current external reanalysis",
        "Frozen effects, intervals, exact sign tests, mode energy, source sheet, DOI, and checksum.",
    ),
    SourceFile(
        "Figure 11",
        "b-c",
        "source_data/branch_interference/interference_surface.csv",
        "Figure_11/Fig11b-c_interference_surface.csv",
        "deterministic calculation grid",
        "route-overlap by leakage grid (101 by 101)",
        "current exact calculation",
        "Exact quadratic one-step interference and explicit vector verification.",
    ),
    SourceFile(
        "Figure 11",
        "b-c",
        "source_data/branch_interference/summary.json",
        "Figure_11/Fig11b-c_summary.json",
        "deterministic numerical summary",
        "route-overlap by leakage grid (101 by 101)",
        "current exact calculation",
        "Maximum verification error and parameter grid.",
    ),
)


def shifted_expanded_file(item: SourceFile) -> SourceFile:
    """Map pre-prospective expansion sources to the current figure order."""

    mapping = {
        "Figure 3": "Supplementary Figure 4",
        "Figure 5": "Supplementary Figure 12",
        "Figure 8": "Figure 11",
        "Figure 11": "Supplementary Figure 6",
    }
    new_figure = mapping.get(item.figure)
    if new_figure is None:
        return item
    old_dir = item.figure.replace(" ", "_") + "/"
    new_dir = new_figure.replace(" ", "_") + "/"
    old_number = item.figure.split()[-1]
    new_number = new_figure.split()[-1]
    destination = item.destination.replace(old_dir, new_dir, 1)
    new_prefix = (
        f"SuppFig{new_number}"
        if new_figure.startswith("Supplementary")
        else f"Fig{new_number}"
    )
    destination = destination.replace(f"Fig{old_number}", new_prefix, 1)
    if item.figure == "Figure 3":
        # The historical nine-panel regime sheet is now a four-panel boundary
        # figure. Six duplicated panels are supplied under the unchanged
        # Supplementary Figures 1--3 and are filtered when FILES is assembled.
        # Relabel the three retained historical panels to their current
        # positions; fresh raw-additive CIFAR-10 is appended as panel d below.
        panel_mapping = {"c": "a", "d": "b", "i": "c"}
        current_panel = panel_mapping[item.panels]
        destination = destination.replace(
            f"SuppFig4{item.panels}", f"SuppFig4{current_panel}", 1
        )
        return replace(
            item,
            figure=new_figure,
            panels=current_panel,
            destination=destination,
        )
    return replace(
        item,
        figure=new_figure,
        destination=destination,
    )


PROSPECTIVE_FILES = (
    SourceFile(
        "Figure 3",
        "a-c",
        "source_data/prospective_input_validity/central_valid_seed_outcomes.csv",
        "Figure_3/Fig3a-c_seed_outcomes.csv",
        "run-level held-out and resource values",
        "paired independent training seed (n=10 per condition)",
        "current complete prospective cohort",
        "All 480 input-valid training runs across three task--core combinations, depth, feedback, and matched backpropagation.",
    ),
    SourceFile(
        "Figure 3",
        "a-c",
        "source_data/prospective_input_validity/central_valid_condition_summary.csv",
        "Figure_3/Fig3a-c_condition_summary.csv",
        "derived condition summaries",
        "paired independent training seed (n=10 per condition)",
        "current derived summary",
        "Condition means and paired-seed bootstrap intervals plotted in Figure 3.",
    ),
    SourceFile(
        "Figure 3",
        "a-c",
        "source_data/prospective_input_validity/central_valid_paired_contrasts.csv",
        "Figure_3/Fig3a-c_paired_contrasts.csv",
        "prespecified paired contrasts",
        "paired independent training seed (n=10 per condition)",
        "current derived analysis",
        "Neuron-indexed, exact-transport, architecture, and feedback-interaction contrasts.",
    ),
    SourceFile(
        "Supplementary Figure 9",
        "a-d",
        "source_data/prospective_input_validity/mechanism_checkpoint_rows_valid.csv",
        "Supplementary_Figure_9/SuppFig9a-d_checkpoint_rows.csv",
        "checkpoint-by-field-by-step diagnostic values",
        "independent input-valid trained checkpoint (n=120)",
        "current complete prospective diagnostic",
        "All 2,400 validity-qualified rows; only non-somatic dendritic stages and their trainable branch parameters enter the geometry.",
    ),
    SourceFile(
        "Supplementary Figure 9",
        "d",
        "source_data/prospective_input_validity/mechanism_association_summary_valid.csv",
        "Supplementary_Figure_9/SuppFig9d_association_summary.csv",
        "checkpoint-clustered association summaries",
        "independent input-valid trained checkpoint (n=120)",
        "current derived analysis",
        "Spearman associations and checkpoint-bootstrap intervals at four relative step sizes.",
    ),
    SourceFile(
        "Supplementary Figure 9",
        "a-c",
        "source_data/prospective_input_validity/mechanism_feedback_summary_valid.csv",
        "Supplementary_Figure_9/SuppFig9a-c_feedback_summary.csv",
        "primary-step feedback summaries",
        "independent input-valid trained checkpoint (n=120)",
        "current derived summary",
        "Feedback-family means and descent counts at relative step 1e-5.",
    ),
    SourceFile(
        "Figure 3",
        "d",
        "source_data/prospective_input_validity/followup_publication_seed_outcomes.csv",
        "Figure_3/Fig3d_ownership_run_outcomes.csv",
        "run-level matched-bandwidth routing outcomes",
        "paired independent training seed (n=10 per condition)",
        "current complete prospective cohort",
        "Filter family=routing; all 120 input-valid correct-versus-fixed-deranged routing runs.",
    ),
    SourceFile(
        "Figure 3",
        "d",
        "source_data/prospective_input_validity/routing_valid_paired_contrasts.csv",
        "Figure_3/Fig3d_ownership_contrasts.csv",
        "prespecified paired routing contrasts",
        "paired independent training seed (n=10 per condition)",
        "current derived analysis",
        "Correct-minus-deranged effects at matched coordinate bandwidth.",
    ),
    SourceFile(
        "Figure 3",
        "e",
        "source_data/clean_exact_bp/run_outcomes.csv",
        "Figure_3/Fig3e_clean_exact_bp_run_outcomes.csv",
        "audited run-level exact-transport and backpropagation outcomes",
        "paired independent training seed (n=10 per task-core-depth condition)",
        "detached tracked-clean confirmatory audit",
        "All 320 accepted runs, resolved conditions, finite endpoints, checkpoint hashes and evaluation-path flags.",
    ),
    SourceFile(
        "Figure 3",
        "e",
        "source_data/clean_exact_bp/paired_differences.csv",
        "Figure_3/Fig3e_clean_exact_bp_paired_differences.csv",
        "seed-level exact-transport-minus-backpropagation differences",
        "paired independent training seed (n=10 per task-core-depth condition)",
        "current derived analysis",
        "All 160 exact/BP pairs before depth averaging.",
    ),
    SourceFile(
        "Figure 3",
        "e",
        "source_data/clean_exact_bp/condition_summary.csv",
        "Figure_3/Fig3e_clean_exact_bp_condition_summary.csv",
        "condition-level accuracy summaries",
        "paired independent training seed (n=10 per condition)",
        "current derived summary",
        "Mean, sample standard deviation and seed-bootstrap interval for each method, task, core and depth.",
    ),
    SourceFile(
        "Figure 3",
        "e",
        "source_data/clean_exact_bp/exact_bp_contrasts.csv",
        "Figure_3/Fig3e_clean_exact_bp_contrasts.csv",
        "depth-specific and seed-level depth-averaged exact/BP contrasts",
        "paired independent training seed (n=10 per task-core contrast)",
        "current derived analysis",
        "Panel e uses the four all-depths-seed-mean rows; depth-specific rows are retained.",
    ),
    SourceFile(
        "Figure 3",
        "text",
        "source_data/clean_exact_bp/summary.json",
        "Figure_3/Text_clean_exact_bp_audit_summary.json",
        "clean-run completeness and numerical audit summary",
        "complete 320-run audit",
        "detached tracked-clean confirmatory audit",
        "Source commits, acceptance gates, streaming-evaluation count and global paired-difference bounds.",
    ),
    SourceFile(
        "Figure 3",
        "f",
        "source_data/prospective_input_validity/followup_publication_condition_summary.csv",
        "Figure_3/Fig3f_fixed_budget_condition_summary.csv",
        "input-valid fixed-budget condition summaries",
        "paired independent training seed (n=10 per condition)",
        "current complete prospective cohort",
        "Filter family=fixed_budget; all 160 additive runs supply panel f.",
    ),
    SourceFile(
        "Figure 3",
        "f",
        "source_data/prospective_input_validity/followup_publication_paired_contrasts.csv",
        "Figure_3/Fig3f_fixed_budget_contrasts.csv",
        "prespecified paired fixed-budget contrasts",
        "paired independent training seed (n=10 per condition)",
        "current derived analysis",
        "Filter study=fixed_budget; input-valid additive depth-four-minus-depth-one contrasts.",
    ),
    SourceFile(
        "Figure 3",
        "g,i",
        "source_data/trained_subtree_address/seed_outcomes.csv",
        "Figure_3/Fig3g_i_subtree_seed_outcomes.csv",
        "run-level held-out, switch, and resource values",
        "paired independent confirmatory seed (n=10 per condition)",
        "current complete phase-1 confirmatory cohort",
        "All 80 frozen K=2 trained subtree-address runs across eight conditions.",
    ),
    SourceFile(
        "Figure 3",
        "h",
        "source_data/trained_subtree_address/gradient_audit.csv",
        "Figure_3/Fig3h_subtree_gradient_audit.csv",
        "seed-level gradient geometry and one-step values",
        "paired independent confirmatory seed (n=10 per condition)",
        "current complete phase-1 confirmatory cohort",
        "Eligibility-weighted capture, cosine, and norm-matched one-step progress.",
    ),
    SourceFile(
        "Figure 3",
        "g,i",
        "source_data/trained_subtree_address/paired_contrasts.csv",
        "Figure_3/Fig3g_i_subtree_paired_contrasts.csv",
        "prespecified paired phase-1 contrasts",
        "paired independent confirmatory seed (n=10)",
        "current derived analysis",
        "Correct-minus-deranged, shared, random, and gated-point comparisons.",
    ),
    SourceFile(
        "Figure 3",
        "g-i",
        "source_data/trained_subtree_address/condition_summary.csv",
        "Figure_3/Fig3g-i_subtree_condition_summary.csv",
        "derived phase-1 condition summaries",
        "paired independent confirmatory seed (n=10 per condition)",
        "current derived summary",
        "Means and seed-bootstrap intervals for every displayed condition.",
    ),
    SourceFile(
        "Figure 3",
        "text",
        "source_data/trained_subtree_address/mechanism_ledger.csv",
        "Figure_3/Text_subtree_mechanism_ledger.csv",
        "balance and route-map audit",
        "paired independent confirmatory seed (n=10 per condition)",
        "current complete audit",
        "Context-label balance, route matrices, and strict fallback counts.",
    ),
    SourceFile(
        "Figure 3",
        "text",
        "source_data/trained_subtree_address/summary.json",
        "Figure_3/Text_subtree_summary.json",
        "source and environment summary",
        "confirmatory experiment",
        "current reproducibility summary",
        "Frozen configuration and executable hashes, environment, and completeness gates.",
    ),
    SourceFile(
        "Supplementary Figure 11",
        "a-c",
        "source_data/focal_selectivity_phase1/site_outcomes.csv.gz",
        "Supplementary_Figure_11/SuppFig11a-c_site_outcomes.csv.gz",
        "site-by-regime intervention outcomes",
        "reconstructed cell; focal sites and regimes nested within cell (n=8 cells)",
        "current complete passive phase-1 matrix",
        "All 16,362 focal-site, dose, regime, and intervention rows.",
    ),
    SourceFile(
        "Supplementary Figure 11",
        "a-c",
        "source_data/focal_selectivity_phase1/cell_condition_metrics.csv",
        "Supplementary_Figure_11/SuppFig11a-c_cell_condition_metrics.csv",
        "cell-averaged condition values",
        "reconstructed cell (n=8)",
        "current complete passive phase-1 matrix",
        "Site outcomes averaged within cell before inference.",
    ),
    SourceFile(
        "Supplementary Figure 11",
        "a-c",
        "source_data/focal_selectivity_phase1/condition_summary.csv",
        "Supplementary_Figure_11/SuppFig11a-c_condition_summary.csv",
        "derived condition summaries",
        "reconstructed cell (n=8)",
        "current derived summary",
        "Cell means and cell-bootstrap intervals across the complete matrix.",
    ),
    SourceFile(
        "Supplementary Figure 11",
        "a",
        "source_data/focal_selectivity_phase1/paired_contrasts.csv",
        "Supplementary_Figure_11/SuppFig11a_paired_contrasts.csv",
        "shunt-minus-additive cell contrasts",
        "reconstructed cell (n=8)",
        "current derived analysis",
        "Conductance-specific contrasts for localization and signed endpoints.",
    ),
    SourceFile(
        "Supplementary Figure 11",
        "text",
        "source_data/focal_selectivity_phase1/summary.json",
        "Supplementary_Figure_11/Text_reproducibility_summary.json",
        "source and numerical audit",
        "confirmatory passive matrix",
        "current reproducibility summary",
        "Frozen hashes, matrix definiteness, somatic residual, environment, and scope boundary.",
    ),
    SourceFile(
        "Figure 4",
        "i-j",
        "source_data/reciprocal_routing/cell_method_capture.csv",
        "Figure_4/Fig4i-j_reciprocal_routing_capture.csv",
        "independent-generator cell-level capture",
        "biological cell (n=8)",
        "current mechanistic control",
        "Exact reciprocal-cable response operators evaluated with real and controlled route dictionaries.",
    ),
    SourceFile(
        "Figure 4",
        "i",
        "source_data/reciprocal_routing/operator_audit.csv",
        "Figure_4/Fig4i_reciprocal_operator_audit.csv",
        "exact response-operator audit",
        "biological cell (n=8; modeled operator)",
        "current mechanistic control",
        "Operator dimensions, conductance state, and numerical diagnostics.",
    ),
    SourceFile(
        "Figure 5",
        "d-f",
        "source_data/microns_inhibitory_routes/inhibitory_axon_metrics.csv",
        "Figure_5/Fig5d-f_presynaptic_axon_control.csv",
        "presynaptic-axon dependence control",
        "presynaptic inhibitory axon aggregate (n=92) within target",
        "current sensitivity analysis",
        "Repeated connections are averaged within presynaptic inhibitory axon before inference.",
    ),
    SourceFile(
        "Figure 5",
        "d-f",
        "source_data/microns_inhibitory_routes/inhibitory_3d_matched_target_metrics.csv",
        "Figure_5/Fig5d-f_joint_3d_path_control.csv",
        "joint geometry-matched target values",
        "reconstructed target cell (n=20; one mouse)",
        "current sensitivity analysis",
        "Joint path-distance and three-dimensional matching control.",
    ),
    SourceFile(
        "Figure 6",
        "i",
        "source_data/physical_cable_sensitivity/cell_primary_contrasts.csv",
        "Figure_6/Fig6i_physical_cable_contrasts.csv",
        "physical-unit cell-level sensitivity",
        "biological cell (n=8 or 45 by cohort)",
        "current sensitivity analysis",
        "Focal localization across axial and membrane-resistance settings.",
    ),
    SourceFile(
        "Supplementary Figure 7",
        "b-c",
        "source_data/spatial_topology_audit/owner_metrics.csv",
        "Supplementary_Figure_7/SuppFig7b-c_owner_connectivity_metrics.csv",
        "fixed-topology connectivity audit",
        "neuron-level topology map (n=2,560)",
        "current boundary control",
        "Unique input coverage, repeated contacts, branch overlap, and feature-degree dispersion.",
    ),
    SourceFile(
        "Supplementary Figure 7",
        "d",
        "source_data/spatial_topology_audit/task_feedback_effects.csv",
        "Supplementary_Figure_7/SuppFig7d_task_feedback_effects.csv",
        "paired spatial-minus-random learning contrasts",
        "paired independent training seed (n=10 per task and feedback)",
        "current boundary control",
        "Effects averaged within seed across input-valid cores: two for MNIST and additive only for the noise task.",
    ),
    SourceFile(
        "Supplementary Figure 8",
        "a-d,g-i",
        "source_data/prospective_input_validity/followup_publication_condition_summary.csv",
        "Supplementary_Figure_8/SuppFig8a-d_g-i_condition_summary.csv",
        "fixed-budget condition summaries",
        "paired independent training seed (n=10 per condition)",
        "current complete prospective cohort",
        "Fixed-terminal-branch and input-contact depth profiles and resource summaries; filter family=fixed_budget.",
    ),
    SourceFile(
        "Supplementary Figure 8",
        "e-f",
        "source_data/prospective_input_validity/followup_publication_paired_contrasts.csv",
        "Supplementary_Figure_8/SuppFig8e-f_paired_contrasts.csv",
        "prespecified fixed-budget contrasts",
        "paired independent training seed (n=10 per condition)",
        "current derived analysis",
        "Depth-four-minus-depth-one and local-minus-backpropagation contrasts; filter study=fixed_budget.",
    ),
    SourceFile(
        "Supplementary Figure 8",
        "a-i",
        "source_data/prospective_input_validity/followup_publication_seed_outcomes.csv",
        "Supplementary_Figure_8/SuppFig8a-i_seed_outcomes.csv",
        "complete follow-up run outcomes",
        "paired independent training seed (n=10 per condition)",
        "current complete prospective cohort",
        "All 160 input-valid additive fixed-budget outcomes; filter family=fixed_budget.",
    ),
)


# Frozen numerical inputs for the three unchanged NeurIPS/arXiv figure assets.
# These are deliberately packaged separately from the journal's expanded
# replots so a reader can reproduce the inherited panels from their original
# source tables without guessing which later analysis replaced them.
INHERITED_NEURIPS_FILES = (
    SourceFile(
        "Supplementary Figure 1",
        "a,d-e",
        "source_data/inherited_neurips/theory_diag_by_condition.csv",
        "Supplementary_Figure_1/SuppFig1a,d-e_theory_diagnostics.csv",
        "frozen original plotting table",
        "independent training seed or checkpoint",
        "unchanged NeurIPS/arXiv source",
        "Path-gain, feedback-fidelity, and noise-task summaries.",
    ),
    SourceFile(
        "Supplementary Figure 1",
        "a",
        "source_data/inherited_neurips/path_gain_cv_mnist_ni5_seed.csv",
        "Supplementary_Figure_1/SuppFig1a_path_gain_seed_values.csv",
        "frozen original paired-seed values",
        "paired independent training seed (n=5)",
        "unchanged NeurIPS/arXiv source",
        "Inset values for the path-gain coefficient of variation.",
    ),
    SourceFile(
        "Supplementary Figure 1",
        "b",
        "source_data/inherited_neurips/error_field_decomposition_runs.csv",
        "Supplementary_Figure_1/SuppFig1b_stage_resolved_feedback_fidelity.csv",
        "frozen original checkpoint values",
        "trained checkpoint (n=5 per architecture)",
        "release-sanitized NeurIPS/arXiv source",
        "Dendritic-stage matched-field cosine values; the machine-local run_dir column is omitted from the release copy.",
    ),
    SourceFile(
        "Supplementary Figure 1",
        "c",
        "source_data/inherited_neurips/inhibition_causality_runs.csv",
        "Supplementary_Figure_1/SuppFig1c_inhibition_interventions.csv",
        "frozen original intervention values",
        "independent training seed (n=5)",
        "unchanged NeurIPS/arXiv source",
        "Post-training inhibition interventions.",
    ),
    SourceFile(
        "Supplementary Figure 1",
        "e",
        "source_data/inherited_neurips/path_transport_upper_bound_summary.csv",
        "Supplementary_Figure_1/SuppFig1e_exact_transport_learning.csv",
        "frozen original condition summary",
        "independent training seed (n=5 per condition)",
        "unchanged NeurIPS/arXiv source",
        "Exact-transport learning control.",
    ),
    SourceFile(
        "Supplementary Figure 2",
        "a",
        "source_data/inherited_neurips/competence_summary_20260422.csv",
        "Supplementary_Figure_2/SuppFig2a_task_competence.csv",
        "frozen original condition summary",
        "independent training seed (n=5 per condition)",
        "unchanged NeurIPS/arXiv source",
        "MNIST and task competence values.",
    ),
    SourceFile(
        "Supplementary Figure 2",
        "a",
        "source_data/inherited_neurips/fashion_mnist_competence_summary.csv",
        "Supplementary_Figure_2/SuppFig2a_fashion_mnist.csv",
        "frozen original condition summary",
        "independent training seed (n=5 per condition)",
        "unchanged NeurIPS/arXiv source",
        "Fashion-MNIST competence values.",
    ),
    SourceFile(
        "Supplementary Figure 2",
        "a",
        "source_data/inherited_neurips/standard_ceiling_refresh_summary.csv",
        "Supplementary_Figure_2/SuppFig2a_backprop_reference.csv",
        "frozen original backpropagation summary",
        "independent training seed (n=5 per condition)",
        "unchanged NeurIPS/arXiv source",
        "Matched backpropagation references.",
    ),
    SourceFile(
        "Supplementary Figure 2",
        "a",
        "source_data/inherited_neurips/core_fair_tuning.csv",
        "Supplementary_Figure_2/SuppFig2a_core_fair_tuning.csv",
        "frozen original run table",
        "independent training seed",
        "unchanged NeurIPS/arXiv source",
        "Archived matched-core learning values.",
    ),
    SourceFile(
        "Supplementary Figure 2",
        "a",
        "source_data/inherited_neurips/phase2b_gap_closing.csv",
        "Supplementary_Figure_2/SuppFig2a_figure_ground.csv",
        "frozen original condition summary",
        "independent training seed",
        "unchanged NeurIPS/arXiv source",
        "Figure-ground MNIST control.",
    ),
    SourceFile(
        "Supplementary Figure 2",
        "b",
        "source_data/inherited_neurips/gradient_fidelity_vs_ie_summary.csv",
        "Supplementary_Figure_2/SuppFig2b_inhibition_dose.csv",
        "frozen original dose summary",
        "independent training seed (n=5 per condition)",
        "unchanged NeurIPS/arXiv source",
        "Inhibitory-dose learning values.",
    ),
    SourceFile(
        "Supplementary Figure 2",
        "c",
        "source_data/inherited_neurips/morphology_ie_regime_runs.csv",
        "Supplementary_Figure_2/SuppFig2c_morphology_inhibition_runs.csv",
        "frozen original run table",
        "independent training seed (n=3 per condition)",
        "unchanged NeurIPS/arXiv source",
        "Descriptive morphology-by-inhibition map.",
    ),
    SourceFile(
        "Supplementary Figure 2",
        "d",
        "source_data/inherited_neurips/revision_exact_transport_factorial_grouped.csv",
        "Supplementary_Figure_2/SuppFig2d_exact_transport_controls.csv",
        "frozen original grouped summary",
        "independent training seed (n=5 per condition)",
        "unchanged NeurIPS/arXiv source",
        "Exact-transport rule and decoder controls.",
    ),
    SourceFile(
        "Supplementary Figure 2",
        "d",
        "source_data/inherited_neurips/revision_exact_transport_bp_grouped.csv",
        "Supplementary_Figure_2/SuppFig2d_backprop_reference.csv",
        "frozen original grouped summary",
        "independent training seed (n=5)",
        "unchanged NeurIPS/arXiv source",
        "Backpropagation reference for panel d.",
    ),
    SourceFile(
        "Supplementary Figure 2",
        "d",
        "source_data/inherited_neurips/revision_additive_gain_norm_grouped.csv",
        "Supplementary_Figure_2/SuppFig2d_additive_controls.csv",
        "frozen original grouped summary",
        "independent training seed (n=5 per condition)",
        "unchanged NeurIPS/arXiv source",
        "Additive gain and normalization controls.",
    ),
    SourceFile(
        "Supplementary Figure 2",
        "d",
        "source_data/inherited_neurips/revision_reactivation_identity_grouped.csv",
        "Supplementary_Figure_2/SuppFig2d_reactivation_controls.csv",
        "frozen original grouped summary",
        "independent training seed (n=5 per condition)",
        "unchanged NeurIPS/arXiv source",
        "Learned versus identity reactivation control.",
    ),
    SourceFile(
        "Supplementary Figure 2",
        "e",
        "source_data/inherited_neurips/feedback_definition_details.csv",
        "Supplementary_Figure_2/SuppFig2e_neuron_indexed_feedback.csv",
        "frozen original run table",
        "paired independent training seed (n=15 per architecture)",
        "unchanged NeurIPS/arXiv source",
        "Matched-width versus neuron-indexed feedback.",
    ),
    SourceFile(
        "Supplementary Figure 3",
        "a",
        "source_data/inherited_neurips/core_fair_tuning.csv",
        "Supplementary_Figure_3/SuppFig3a_rule_family.csv",
        "frozen original run table",
        "independent training seed (n=3 per rule)",
        "unchanged NeurIPS/arXiv source",
        "Within-shunting rule-family comparison.",
    ),
    SourceFile(
        "Supplementary Figure 3",
        "b",
        "source_data/inherited_neurips/local_mismatch_recheck_runs.csv",
        "Supplementary_Figure_3/SuppFig3b_error_source.csv",
        "frozen original run table",
        "independent training seed (n=3 per condition)",
        "unchanged NeurIPS/arXiv source",
        "Somatic teaching error and unmatched local control.",
    ),
    SourceFile(
        "Supplementary Figure 3",
        "c",
        "source_data/inherited_neurips/revision_exact_transport_factorial_grouped.csv",
        "Supplementary_Figure_3/SuppFig3c_exact_transport_factorial.csv",
        "frozen original grouped summary",
        "independent training seed (n=5 per condition)",
        "unchanged NeurIPS/arXiv source",
        "Three- and five-factor exact-transport factorial.",
    ),
    SourceFile(
        "Supplementary Figure 3",
        "c",
        "source_data/inherited_neurips/revision_exact_transport_bp_grouped.csv",
        "Supplementary_Figure_3/SuppFig3c_backprop_reference.csv",
        "frozen original grouped summary",
        "independent training seed (n=5)",
        "unchanged NeurIPS/arXiv source",
        "Backpropagation reference for panel c.",
    ),
    SourceFile(
        "Supplementary Figure 3",
        "d",
        "source_data/inherited_neurips/noise_resilience_rank_bridge_summary.csv",
        "Supplementary_Figure_3/SuppFig3d_feedback_ladder.csv",
        "frozen original condition summary",
        "independent training seed (n=5 per condition)",
        "unchanged NeurIPS/arXiv source",
        "Noise-resilience feedback-structure ladder.",
    ),
)


EXTENSION_FILES = (
    SourceFile(
        "Figure 3",
        "text",
        "source_data/prospective_input_validity/historical_run_validity.csv",
        "Figure_3/Text_historical_input_validity_ledger.csv",
        "outcome-independent run-validity ledger",
        "training run (n=1,840)",
        "current validity audit",
        "Resolved input convention and publication inclusion for every historical prospective run.",
    ),
    SourceFile(
        "Figure 3",
        "text",
        "source_data/prospective_input_validity/summary.json",
        "Figure_3/Text_input_validity_summary.json",
        "validity-accounting summary",
        "historical prospective programme",
        "current validity audit",
        "Exclusion rule and retained counts; no outcome enters the rule.",
    ),
    SourceFile(
        "Figure 4",
        "a-f",
        "source_data/trained_subtree_address_full_factorial/seed_outcomes.csv",
        "Figure_4/Fig4a-f_seed_outcomes.csv",
        "complete run-level factorial outcomes",
        "paired independent training seed (n=20)",
        "current corrected confirmatory cohort",
        "All 2,700 fits after the stochastic-control pairing audit and complete rerun.",
    ),
    SourceFile(
        "Figure 4",
        "b,e-f",
        "source_data/trained_subtree_address_full_factorial/condition_summary.csv",
        "Figure_4/Fig4b,e-f_condition_summary.csv",
        "derived condition summaries",
        "paired independent training seed (n=20)",
        "current derived summary",
        "Held-out learning, capture, cosine and one-step summaries by representation, route family and budget.",
    ),
    SourceFile(
        "Figure 4",
        "c-d",
        "source_data/trained_subtree_address_full_factorial/paired_contrasts.csv",
        "Figure_4/Fig4c-d_paired_contrasts.csv",
        "paired seed contrasts",
        "paired independent training seed (n=20)",
        "current derived analysis",
        "Correct-ancestry contrasts against matched controls and the rewired tree.",
    ),
    SourceFile(
        "Figure 4",
        "a-f",
        "source_data/trained_subtree_address_full_factorial/mechanism_ledger.csv",
        "Figure_4/Fig4a-f_mechanism_ledger.csv",
        "resource and implementation audit",
        "paired independent training seed (n=20)",
        "current audit",
        "Parameter, contact, rank, routing and fallback checks for every condition.",
    ),
    SourceFile(
        "Figure 4",
        "a-f",
        "source_data/trained_subtree_address_full_factorial/summary.json",
        "Figure_4/Fig4a-f_summary.json",
        "frozen cohort summary",
        "paired independent training seed (n=20)",
        "current confirmatory summary",
        "Configuration and script hashes, fit counts, numerical gates and scope boundary.",
    ),
    SourceFile(
        "Figure 8",
        "b-d",
        "source_data/focal_selectivity_active_ensemble/cell_condition_metrics.csv",
        "Figure_8/Fig8b-d_active_cell_metrics.csv",
        "independent-cell endpoint values",
        "reconstructed cell (n=8; channel draws and sites nested)",
        "current active steady-state sensitivity",
        "Cell-level localization, direction and gradient-energy endpoints.",
    ),
    SourceFile(
        "Figure 8",
        "b-c",
        "source_data/focal_selectivity_active_ensemble/condition_summary.csv",
        "Figure_8/Fig8b-c_active_condition_summary.csv",
        "derived condition summaries",
        "reconstructed cell (n=8)",
        "current derived summary",
        "Dose curves and cell-bootstrap intervals for shunting and matched additive input.",
    ),
    SourceFile(
        "Figure 8",
        "c",
        "source_data/focal_selectivity_active_ensemble/paired_contrasts.csv",
        "Figure_8/Fig8c_active_paired_contrasts.csv",
        "paired cell contrasts",
        "reconstructed cell (n=8)",
        "current derived analysis",
        "Shunt-minus-additive localization by dose.",
    ),
    SourceFile(
        "Figure 8",
        "a-d",
        "source_data/focal_selectivity_active_ensemble/acceptance_ledger.csv",
        "Figure_8/Fig8a-d_active_acceptance_ledger.csv",
        "nonlinear-equilibrium acceptance audit",
        "reconstructed cell; channel draws nested within cell",
        "current numerical audit",
        "Attempt, acceptance, residual and Jacobian checks for all channel draws.",
    ),
    SourceFile(
        "Figure 8",
        "a-d",
        "source_data/focal_selectivity_active_ensemble/summary.json",
        "Figure_8/Fig8a-d_active_summary.json",
        "frozen cohort summary",
        "reconstructed cell (n=8)",
        "current active steady-state sensitivity",
        "Protocol hashes, row counts, numerical gates and scope boundary.",
    ),
    SourceFile(
        "Figure 8",
        "e-f",
        "source_data/fulltree_boundary/output/cell_method_means.csv",
        "Figure_8/Fig8e-f_full_tree_cell_method_means.csv",
        "independent-target method values",
        "target cell (n=7; scans and splits nested)",
        "current scan-complete full-tree analysis",
        "Cell-level held-out learning and common-checkpoint capture values.",
    ),
    SourceFile(
        "Figure 8",
        "e-f",
        "source_data/fulltree_boundary/output/scan_method_means.csv",
        "Figure_8/Fig8e-f_full_tree_scan_method_means.csv",
        "scan-level nested values",
        "target cell; 13 scans nested within 7 targets",
        "current scan-complete full-tree analysis",
        "Scan means before target-level averaging.",
    ),
    SourceFile(
        "Figure 8",
        "e-f",
        "source_data/fulltree_boundary/output/summary.json",
        "Figure_8/Fig8e-f_full_tree_summary.json",
        "derived statistical summary",
        "target cell (n=7)",
        "current scan-complete full-tree analysis",
        "Completeness, finite-difference gates, effects, intervals and paired tests.",
    ),
    SourceFile(
        "Figure 9",
        "text",
        "source_data/functional_topology_all_scans/cell_metrics.csv",
        "Figure_9/Text_all_scan_functional_cell_metrics.csv",
        "independent-target topology values",
        "target cell (n=7; scans nested)",
        "current deterministic scan-complete cohort",
        "Target-level structure--function statistics after averaging eligible scans.",
    ),
    SourceFile(
        "Figure 9",
        "text",
        "source_data/functional_topology_all_scans/scan_metrics.csv",
        "Figure_9/Text_all_scan_functional_scan_metrics.csv",
        "scan-level nested values",
        "target cell; 13 scans nested within 7 targets",
        "current deterministic scan-complete cohort",
        "All scan-specific topology statistics; the machine-local source-summary path is omitted from the release copy.",
    ),
    SourceFile(
        "Figure 9",
        "text",
        "source_data/functional_topology_all_scans/summary.json",
        "Figure_9/Text_all_scan_functional_summary.json",
        "derived statistical summary",
        "target cell (n=7)",
        "current deterministic scan-complete cohort",
        "Eligibility rule, effects, target-bootstrap intervals and scope boundary.",
    ),
)


def shifted_extension_file(item: SourceFile) -> SourceFile:
    """Insert the physical-depth display into the current main-figure order."""

    mapping = {"Figure 8": "Figure 9", "Figure 9": "Figure 10"}
    new_figure = mapping.get(item.figure)
    if new_figure is None:
        return item
    old_dir = item.figure.replace(" ", "_") + "/"
    new_dir = new_figure.replace(" ", "_") + "/"
    old_number = item.figure.split()[-1]
    new_number = new_figure.split()[-1]
    destination = item.destination.replace(old_dir, new_dir, 1)
    destination = destination.replace(f"Fig{old_number}", f"Fig{new_number}", 1)
    return replace(item, figure=new_figure, destination=destination)


NEW_CONFIRMATORY_FILES = (
    SourceFile(
        "Figure 5",
        "b",
        "source_data/credit_phase_theory/spectral_phase_seed.csv",
        "Figure_5/Fig5b_spectral_phase_seed.csv",
        "paired seed values",
        "paired independent simulation seed (n=50)",
        "frozen confirmatory theory experiment",
        "Task--tree alignment crossed with route rank and rank-matched controls.",
    ),
    SourceFile(
        "Figure 5",
        "c-d",
        "source_data/credit_phase_theory/depth_training_seed.csv",
        "Figure_5/Fig5c-d_depth_training_seed.csv",
        "paired seed values",
        "paired independent simulation seed (n=50)",
        "frozen confirmatory theory experiment",
        "Task hierarchy crossed with routed depth and full stochastic-gradient controls.",
    ),
    SourceFile(
        "Figure 5",
        "e",
        "source_data/credit_phase_theory/projection_phase_seed.csv",
        "Figure_5/Fig5e_projection_phase_seed.csv",
        "paired seed values",
        "paired independent simulation seed (n=50)",
        "frozen confirmatory theory experiment",
        "Signal-retention by noise-retention projection crossover.",
    ),
    SourceFile(
        "Figure 5",
        "f",
        "source_data/credit_phase_theory/reliability_phase_seed.csv",
        "Figure_5/Fig5f_reliability_phase_seed.csv",
        "paired seed values",
        "paired independent simulation seed (n=50)",
        "frozen confirmatory theory experiment",
        "Reliability-aligned, global, shuffled, reversed and point-gate branch gains.",
    ),
    SourceFile(
        "Figure 5",
        "g-h",
        "source_data/credit_phase_theory/same_span_diagnostics.csv",
        "Figure_5/Fig5g-h_same_span_diagnostics.csv",
        "deterministic diagnostics",
        "exact numerical diagnostic",
        "frozen confirmatory theory experiment",
        "Static nonzero route-gain span and Gram-conditioning audit.",
    ),
    SourceFile(
        "Figure 5",
        "i",
        "source_data/credit_phase_existing/operator_metrics.csv",
        "Figure_5/Fig5i_operator_metrics.csv",
        "checkpoint operator values",
        "independent training seed block (n=20)",
        "secondary frozen-factorial analysis",
        "Signal--noise--curvature utility and observed one-step progress.",
    ),
    SourceFile(
        "Figure 6",
        "c,e-f",
        "source_data/nonlinear_physical_depth_confirmatory/seed_outcomes.csv",
        "Figure_6/Fig6c,e-f_seed_outcomes.csv",
        "complete run-level outcomes",
        "paired independent training seed (n=10)",
        "fresh exact-resource confirmatory cohort",
        "All 270 valid BP, additive and LocalCA outcomes.",
    ),
    SourceFile(
        "Figure 6",
        "c,e-f",
        "source_data/nonlinear_physical_depth_confirmatory/condition_summary.csv",
        "Figure_6/Fig6c,e-f_condition_summary.csv",
        "derived condition summaries",
        "paired independent training seed (n=10)",
        "current derived analysis",
        "Depth-resolved condition means and seed-bootstrap intervals.",
    ),
    SourceFile(
        "Figure 6",
        "d-e",
        "source_data/nonlinear_physical_depth_confirmatory/paired_contrasts.csv",
        "Figure_6/Fig6d-e_paired_contrasts.csv",
        "paired seed contrasts",
        "paired independent training seed (n=10)",
        "current derived analysis",
        "Prespecified depth effects and aligned-minus-control interactions.",
    ),
    SourceFile(
        "Figure 6",
        "text",
        "source_data/nonlinear_physical_depth_confirmatory/audit.json",
        "Figure_6/Text_artifact_and_claim_audit.json",
        "artifact and claim audit",
        "confirmatory cohort",
        "current audit",
        "Completeness, exact-resource, finite-metric, accessibility and positive-claim gates.",
    ),
    SourceFile(
        "Figure 6",
        "text",
        "source_data/nonlinear_physical_depth_confirmatory/mechanism_gradient_summary.csv",
        "Figure_6/Text_checkpoint_gradient_summary.csv",
        "checkpoint gradient values",
        "paired trained checkpoint (n=10 per depth)",
        "post-training mechanism diagnostic",
        "Exact-path and shared-soma gradients compared with autograd on identical batches.",
    ),
    SourceFile(
        "Supplementary Figure 13",
        "b",
        "source_data/positive_conductance_reliability_step_consistent/branch_reliability.csv",
        "Supplementary_Figure_13/SuppFig13b_branch_reliability.csv",
        "branch-level source values",
        "paired independent simulation seed (n=50)",
        "fresh step-consistent confirmatory cohort",
        "Branch signal, noise, SNR and step-optimal attenuation.",
    ),
    SourceFile(
        "Supplementary Figure 13",
        "c-e",
        "source_data/positive_conductance_reliability_step_consistent/condition_summary.csv",
        "Supplementary_Figure_13/SuppFig13c-e_condition_summary.csv",
        "derived condition summaries",
        "paired independent simulation seed (n=50)",
        "current derived analysis",
        "One-step and final-loss summaries across nine controls.",
    ),
    SourceFile(
        "Supplementary Figure 13",
        "d,f",
        "source_data/positive_conductance_reliability_step_consistent/extended_contrasts.csv",
        "Supplementary_Figure_13/SuppFig13d,f_contrasts.csv",
        "paired seed contrasts",
        "paired independent simulation seed (n=50)",
        "current derived analysis",
        "Aligned-versus-control immediate and final-loss contrasts.",
    ),
    SourceFile(
        "Supplementary Figure 14",
        "c-e",
        "source_data/same_span_coefficient_learning/seed_trajectories.csv",
        "Supplementary_Figure_14/SuppFig14c-e_seed_trajectories.csv",
        "checkpoint trajectories",
        "paired independent target-and-noise seed (n=50)",
        "fresh confirmatory rate experiment",
        "Finite-data learning trajectories for identical-span coordinates.",
    ),
    SourceFile(
        "Supplementary Figure 14",
        "f",
        "source_data/same_span_coefficient_learning/paired_contrasts.csv",
        "Supplementary_Figure_14/SuppFig14f_paired_contrasts.csv",
        "paired seed contrasts",
        "paired independent target-and-noise seed (n=50)",
        "current derived analysis",
        "Nested-minus-Haar final-loss contrasts.",
    ),
    SourceFile(
        "Supplementary Figure 14",
        "f",
        "source_data/same_span_coefficient_learning/theory_predictions.csv",
        "Supplementary_Figure_14/SuppFig14f_theory_predictions.csv",
        "exact finite-time predictions",
        "paired independent target seed (n=50)",
        "current exact analysis",
        "Modal bias and accumulated-variance predictions.",
    ),
    SourceFile(
        "Supplementary Figure 15",
        "a",
        "source_data/nonlinear_physical_depth_canary/bp_aligned_seed_rows.csv",
        "Supplementary_Figure_15/SuppFig15a_negative_canary.csv",
        "exploratory seed values",
        "exploratory training seed (n=2)",
        "transparent calibration",
        "Negative severe-shift canary.",
    ),
    SourceFile(
        "Supplementary Figure 15",
        "b",
        "source_data/nonlinear_physical_depth_accessibility/bp_seed_rows.csv",
        "Supplementary_Figure_15/SuppFig15b_accessibility.csv",
        "exploratory seed values",
        "exploratory training seed (n=2 per cell)",
        "transparent calibration",
        "Unseen gain-shift accessibility ladder.",
    ),
    SourceFile(
        "Supplementary Figure 15",
        "c",
        "source_data/nonlinear_physical_depth_coupling/bp_seed_rows.csv",
        "Supplementary_Figure_15/SuppFig15c_coupling.csv",
        "exploratory seed values",
        "exploratory training seed (n=2 per cell)",
        "transparent calibration",
        "Child-coupling ladder.",
    ),
    SourceFile(
        "Supplementary Figure 15",
        "d",
        "source_data/nonlinear_physical_depth_signal/bp_seed_rows.csv",
        "Supplementary_Figure_15/SuppFig15d_signal_ladder.csv",
        "exploratory seed values",
        "exploratory training seed (n=2 per cell)",
        "transparent calibration",
        "Signal-contrast ladder.",
    ),
    SourceFile(
        "Supplementary Figure 15",
        "d",
        "source_data/nonlinear_physical_depth_boundary/bp_seed_rows.csv",
        "Supplementary_Figure_15/SuppFig15d_selected_boundary.csv",
        "exploratory seed values",
        "exploratory training seed (n=2)",
        "transparent calibration",
        "Separately run 0.80 boundary that selected the confirmatory point.",
    ),
)


FILES = (
    *PROSPECTIVE_FILES,
    *NEW_CONFIRMATORY_FILES,
    *(shifted_extension_file(item) for item in EXTENSION_FILES),
    *INHERITED_NEURIPS_FILES,
    *(
        shifted_expanded_file(item)
        for item in EXPANDED_FILES
        if item.figure != "Figure 3" or item.panels in {"c", "d", "i"}
    ),
    *(shifted_legacy_file(item) for item in FILES),
)

# Final theory-first consolidation additions.  These tables underlie the new
# cross-dataset bandwidth replication and the point-versus-dendrite,
# BP-versus-local, and alignment-dose continuations.  Keeping them as explicit
# allow-list entries ensures that the submission archive cannot silently lag
# the compiled nine-figure manuscript.
FILES += (
    SourceFile(
        "Supplementary Figure 4",
        "d",
        "source_data/cifar10_additive_feedback_ladder_confirmatory/seed_outcomes.csv",
        "Supplementary_Figure_4/SuppFig4d_CIFAR10_raw_additive_seed_outcomes.csv",
        "complete run-level outcomes",
        "paired independent training seed (n=20 per condition)",
        "fresh frozen confirmation",
        "All 80 strict-scalar, neuron-specific, exact-path and matched-backpropagation outcomes.",
    ),
    SourceFile(
        "Supplementary Figure 4",
        "d",
        "source_data/cifar10_additive_feedback_ladder_confirmatory/condition_summary.csv",
        "Supplementary_Figure_4/SuppFig4d_CIFAR10_raw_additive_condition_summary.csv",
        "condition summaries",
        "paired independent training seed (n=20)",
        "current derived analysis",
        "Condition means, standard deviations and Student-t confidence intervals.",
    ),
    SourceFile(
        "Supplementary Figure 4",
        "d",
        "source_data/cifar10_additive_feedback_ladder_confirmatory/paired_contrasts.csv",
        "Supplementary_Figure_4/SuppFig4d_CIFAR10_raw_additive_paired_contrasts.csv",
        "paired seed contrasts",
        "paired independent training seed (n=20)",
        "current derived analysis",
        "Holm-adjusted superiority tests, exact sign-flip sensitivities and seed differences.",
    ),
    SourceFile(
        "Supplementary Figure 4",
        "d",
        "source_data/cifar10_additive_feedback_ladder_confirmatory/summary.json",
        "Supplementary_Figure_4/SuppFig4d_CIFAR10_raw_additive_audit.json",
        "artifact and claim audit",
        "complete 80-fit paired cohort",
        "current frozen audit",
        "Completeness, source identity, calibration, convergence, adequacy, superiority and equivalence gates.",
    ),
    SourceFile(
        "Figure 2",
        "p-q",
        "source_data/fashion_feedback_ladder/seed_outcomes.csv",
        "Figure_2/Fig2p-q_Fashion_MNIST_seed_outcomes.csv",
        "complete run-level outcomes",
        "paired independent training seed (n=10 per architecture and rule)",
        "fresh preregistered replication",
        "All 60 Fashion-MNIST fits in the scalar, neuron-indexed, and exact-path feedback ladder.",
    ),
    SourceFile(
        "Figure 2",
        "p-q",
        "source_data/fashion_feedback_ladder/condition_summary.csv",
        "Figure_2/Fig2p-q_Fashion_MNIST_condition_summary.csv",
        "derived condition summaries",
        "paired independent training seed (n=10)",
        "current derived analysis",
        "Accuracy summaries for shunting and additive architectures.",
    ),
    SourceFile(
        "Figure 2",
        "p-q",
        "source_data/fashion_feedback_ladder/paired_contrasts.csv",
        "Figure_2/Fig2p-q_Fashion_MNIST_paired_contrasts.csv",
        "paired seed contrasts",
        "paired independent training seed (n=10)",
        "current derived analysis",
        "Neuron-indexed-minus-scalar and exact-path-minus-neuron-indexed contrasts.",
    ),
    SourceFile(
        "Figure 2",
        "p-q",
        "source_data/fashion_feedback_ladder/audit.json",
        "Figure_2/Fig2p-q_Fashion_MNIST_audit.json",
        "artifact and claim audit",
        "complete 60-fit cohort",
        "current audit",
        "Completeness, finite-metric, replication-gate, and source-hash checks.",
    ),
    SourceFile(
        "Figure 5",
        "g-l",
        "source_data/point_dendrite_credit_controls/combined_seed_outcomes.csv",
        "Figure_5/Fig5g-l_point_dendrite_seed_outcomes.csv",
        "complete run-level outcomes",
        "paired independent training seed (n=10 per condition)",
        "fresh exact-resource controls",
        "Serial-tree, grouped-star, matched-point, soma-broadcast, and LocalCA outcomes.",
    ),
    SourceFile(
        "Figure 5",
        "g-l",
        "source_data/point_dendrite_credit_controls/condition_summary.csv",
        "Figure_5/Fig5g-l_point_dendrite_condition_summary.csv",
        "derived condition summaries",
        "paired independent training seed (n=10)",
        "current derived analysis",
        "Condition means and seed-bootstrap intervals.",
    ),
    SourceFile(
        "Figure 5",
        "g-l",
        "source_data/point_dendrite_credit_controls/paired_contrasts.csv",
        "Figure_5/Fig5g-l_point_dendrite_paired_contrasts.csv",
        "paired seed contrasts",
        "paired independent training seed (n=10)",
        "current derived analysis",
        "Point-versus-serial and BP-versus-local credit-gap decompositions.",
    ),
    SourceFile(
        "Figure 5",
        "g-l",
        "source_data/point_dendrite_credit_controls/audit.json",
        "Figure_5/Fig5g-l_point_dendrite_audit.json",
        "artifact and claim audit",
        "complete point--dendrite control programme",
        "current audit",
        "Configuration, exact-resource, completeness, and source-hash checks.",
    ),
    SourceFile(
        "Figure 5",
        "m-o",
        "source_data/physical_alignment_dose/combined_seed_outcomes.csv",
        "Figure_5/Fig5m-o_alignment_dose_seed_outcomes.csv",
        "complete run-level outcomes",
        "paired independent training seed (n=10 per alignment and depth)",
        "prospective interpolation",
        "Ninety new intermediate-dose fits combined with the frozen endpoint cohorts.",
    ),
    SourceFile(
        "Figure 5",
        "m-o",
        "source_data/physical_alignment_dose/condition_summary.csv",
        "Figure_5/Fig5m-o_alignment_dose_condition_summary.csv",
        "derived condition summaries",
        "paired independent training seed (n=10)",
        "current derived analysis",
        "Depth effects across five task--tree alignment doses.",
    ),
    SourceFile(
        "Figure 5",
        "m-o",
        "source_data/physical_alignment_dose/paired_contrasts.csv",
        "Figure_5/Fig5m-o_alignment_dose_paired_contrasts.csv",
        "paired seed contrasts",
        "paired independent training seed (n=10)",
        "current derived analysis",
        "Depth effects, adjacent-dose increments, and within-seed slope statistics.",
    ),
    SourceFile(
        "Figure 5",
        "m-o",
        "source_data/physical_alignment_dose/audit.json",
        "Figure_5/Fig5m-o_alignment_dose_audit.json",
        "artifact and claim audit",
        "complete 90-fit interpolation cohort plus frozen endpoints",
        "current audit",
        "Completeness, source-equivalence, finite-metric, and preregistered-gate checks.",
    ),
    SourceFile(
        "Figure 5",
        "p-u",
        "source_data/remaining_physical_experiments/seed_outcomes_with_h3_reference.csv",
        "Figure_5/Fig5p-u_grouped_point_and_H2_seed_outcomes.csv",
        "complete run-level outcomes",
        "paired independent training seed (n=10 per condition)",
        "fresh frozen extension plus unchanged H3 references",
        "Literal grouped-point H3 controls and serial, grouped-point, shared-LocalCA, and path-LocalCA H2 outcomes.",
    ),
    SourceFile(
        "Figure 5",
        "p-u",
        "source_data/remaining_physical_experiments/condition_summary.csv",
        "Figure_5/Fig5p-u_grouped_point_and_H2_condition_summary.csv",
        "derived condition summaries",
        "paired independent training seed (n=10)",
        "current derived analysis",
        "Accuracy summaries and seed-bootstrap intervals for both hierarchy depths and placements.",
    ),
    SourceFile(
        "Figure 5",
        "p-u",
        "source_data/remaining_physical_experiments/paired_contrasts.csv",
        "Figure_5/Fig5p-u_grouped_point_and_H2_paired_contrasts.csv",
        "paired seed contrasts",
        "paired independent training seed (n=10)",
        "current derived analysis",
        "Serial-minus-grouped, depth, placement-interaction, and LocalCA contrasts retained regardless of sign.",
    ),
    SourceFile(
        "Figure 5",
        "p-u",
        "source_data/remaining_physical_experiments/audit.json",
        "Figure_5/Fig5p-u_grouped_point_and_H2_audit.json",
        "artifact and claim audit",
        "complete 220-fit cohort",
        "current audit",
        "Completeness, finite-metric, no-fallback, seed, resource, manifest, and claim-gate checks.",
    ),
    SourceFile(
        "Figure 6",
        "a-d",
        "source_data/physical_depth_h4_factorial/seed_outcomes.csv",
        "Figure_6/Fig6a-d_H4_seed_outcomes.csv",
        "complete run-level outcomes",
        "paired independent training seed (n=10 per condition)",
        "fresh frozen extension",
        "All 360 intended H4 outcomes, including explicit provenance for the 90 construction-only D3 replacements.",
    ),
    SourceFile(
        "Figure 6",
        "a-d",
        "source_data/physical_depth_h4_factorial/condition_summary.csv",
        "Figure_6/Fig6a-d_H4_condition_summary.csv",
        "derived condition summaries",
        "paired independent training seed (n=10)",
        "current derived analysis",
        "Depth-resolved H4 accuracy and seed-bootstrap intervals.",
    ),
    SourceFile(
        "Figure 6",
        "c",
        "source_data/physical_depth_h4_factorial/paired_contrasts.csv",
        "Figure_6/Fig6c_H4_paired_contrasts.csv",
        "paired seed contrasts",
        "paired independent training seed (n=10)",
        "current derived analysis",
        "Depth, placement, point-control, local-credit and mechanism contrasts with primary-family correction.",
    ),
    SourceFile(
        "Figure 6",
        "a-d",
        "source_data/physical_depth_h4_factorial/audit.json",
        "Figure_6/Fig6a-d_H4_audit.json",
        "artifact and claim audit",
        "complete 360-fit intended cohort",
        "current audit",
        "Completeness, immutable-source, D3 replacement, finite-metric, seed and exact-resource gates.",
    ),
    SourceFile(
        "Supplementary Figure 26",
        "a-d",
        "source_data/physical_depth_clean_source_replication/seed_outcomes.csv",
        "Supplementary_Figure_26/SuppFig26a-d_clean_source_seed_outcomes.csv",
        "complete run-level outcomes",
        "same paired training seeds as the historical H2/H3 cohorts",
        "immutable-source concordance replication",
        "All 430 clean-source outcomes.",
    ),
    SourceFile(
        "Supplementary Figure 26",
        "a-d",
        "source_data/physical_depth_clean_source_replication/historical_concordance.csv",
        "Supplementary_Figure_26/SuppFig26a-b_historical_concordance.csv",
        "same-seed source concordance",
        "paired training seed and semantic condition (n=430)",
        "current derived analysis",
        "Historical and clean-source outcome pairs with no outlier removal.",
    ),
    SourceFile(
        "Supplementary Figure 26",
        "c-d",
        "source_data/physical_depth_clean_source_replication/paired_contrasts.csv",
        "Supplementary_Figure_26/SuppFig26c-d_paired_contrasts.csv",
        "paired seed contrasts",
        "paired training seed (n=10)",
        "current derived analysis",
        "Recomputed H2/H3 depth, placement, point-control, local-credit and additive contrasts.",
    ),
    SourceFile(
        "Supplementary Figure 26",
        "a-d",
        "source_data/physical_depth_clean_source_replication/audit.json",
        "Supplementary_Figure_26/SuppFig26a-d_audit.json",
        "artifact and claim audit",
        "complete 430-fit rerun cohort",
        "current audit",
        "Immutable source, completeness, finite-metric, seed, resource and concordance gates.",
    ),
    SourceFile(
        "Supplementary Figure 16",
        "a-b",
        "source_data/interior_optimum/depth_curves.csv",
        "Supplementary_Figure_16/SuppFig16a-b_depth_curves.csv",
        "depth-phase curves",
        "paired independent simulation seed (n=50)",
        "current derived analysis",
        "Depth-resolved theory and trained outcomes across task hierarchy levels.",
    ),
    SourceFile(
        "Supplementary Figure 16",
        "c-d",
        "source_data/interior_optimum/depth_agreement_seed.csv",
        "Supplementary_Figure_16/SuppFig16c-d_depth_agreement_seed.csv",
        "paired seed agreement values",
        "paired independent simulation seed (n=50)",
        "current derived analysis",
        "Within-seed agreement between predicted and observed depth effects.",
    ),
    SourceFile(
        "Supplementary Figure 16",
        "e-f",
        "source_data/interior_optimum/factorial_contrast_seed.csv",
        "Supplementary_Figure_16/SuppFig16e-f_factorial_contrast_seed.csv",
        "paired seed contrasts",
        "paired independent training seed (n=20)",
        "current derived analysis",
        "Intermediate-budget contrasts from the complete subtree factorial.",
    ),
    SourceFile(
        "Supplementary Figure 16",
        "e-f",
        "source_data/interior_optimum/factorial_peak_seed.csv",
        "Supplementary_Figure_16/SuppFig16e-f_factorial_peak_seed.csv",
        "paired seed peak locations",
        "paired independent training seed (n=20)",
        "current derived analysis",
        "Seed-resolved locations of the empirical route-budget optimum.",
    ),
    SourceFile(
        "Supplementary Figure 16",
        "a-f",
        "source_data/interior_optimum/summary.json",
        "Supplementary_Figure_16/SuppFig16_summary.json",
        "derived statistical summary",
        "paired simulation or training seed",
        "current derived analysis",
        "Frozen crossover and peak-location summary.",
    ),
    SourceFile(
        "Supplementary Figure 17",
        "a-c",
        "source_data/capture_per_wire/cell_method_channel.csv",
        "Supplementary_Figure_17/SuppFig17a-c_cell_method_channel.csv",
        "cell-level capture and wiring values",
        "reconstructed cell (n=8; one mouse)",
        "current derived analysis",
        "Capture, dense-wiring fraction, and capture per unit wiring by route family.",
    ),
    SourceFile(
        "Supplementary Figure 17",
        "a-c",
        "source_data/capture_per_wire/summary.csv",
        "Supplementary_Figure_17/SuppFig17a-c_summary.csv",
        "derived cell-bootstrap summaries",
        "reconstructed cell (n=8; one mouse)",
        "current derived analysis",
        "Model-matched 14.2-fold ceiling and density-matched 2.7-fold anatomy-specific factor.",
    ),
    SourceFile(
        "Figure 9",
        "k",
        "source_data/credit_phase_plane/points.csv",
        "Figure_9/Fig9k_phase_plane_points.csv",
        "phase-plane source values",
        "independent simulation or training seed block",
        "current synthesis",
        "Alignment-by-bandwidth coordinates and observed outcomes used in the final phase-plane synthesis.",
    ),
)


def final_display_file(item: SourceFile) -> SourceFile:
    """Map legacy allow-list entries onto the compiled nine-figure layout."""

    source = item.source
    figure = item.figure
    panels = item.panels

    if source.startswith("source_data/microns_inhibitory_routes/"):
        figure = "Supplementary Figure 12"
    elif source.startswith("source_data/trained_subtree_address_full_factorial/"):
        figure = "Figure 3"
    elif source.startswith("source_data/trained_subtree_address/"):
        # The two-stream route diagnostics are retained in full as S19g--i;
        # the Results text identifies this as the predecessor to the continuous
        # branch-conflict task, but no two-stream panel remains in the main set.
        figure = "Supplementary Figure 19"
        panels = {"g,i": "g,i", "h": "h", "g-i": "g-i"}.get(panels, panels)
    elif source.startswith("source_data/prospective_input_validity/"):
        if item.figure == "Supplementary Figure 9":
            pass
        elif Path(source).name.startswith("central_valid_"):
            figure, panels = "Figure 2", "g"
        elif Path(source).name in {
            "followup_publication_seed_outcomes.csv",
            "routing_valid_paired_contrasts.csv",
        }:
            figure, panels = "Figure 2", "g"
        else:
            figure = "Supplementary Figure 19"
    elif source.startswith("source_data/fashion_feedback_ladder/"):
        figure, panels = "Figure 2", "c"
    elif source.startswith("source_data/clean_exact_bp/"):
        figure, panels = "Supplementary Figure 19", "e"
    elif source.startswith("source_data/credit_phase_plane/"):
        figure = "Figure 4"
        panels = "g"
    elif source.startswith("source_data/credit_phase_"):
        figure = "Figure 4"
    elif source.startswith("source_data/physical_alignment_dose/"):
        figure, panels = "Supplementary Figure 18", "c-e"
    elif source.startswith("source_data/remaining_physical_experiments/"):
        figure, panels = "Supplementary Figure 18", "f-k"
    elif source.startswith("source_data/nonlinear_physical_depth_confirmatory/"):
        figure, panels = "Figure 5", "c-f"
    elif source.startswith("source_data/point_dendrite_credit_controls/"):
        figure, panels = "Figure 5", "g-h"
    elif source.startswith("source_data/physical_depth_h4_factorial/"):
        figure = "Figure 6"
        panels = {"a-d": "a-d", "c": "c"}.get(panels, panels)
    elif source.startswith("source_data/task_family_alignment/"):
        figure = "Figure 6"
        panels = {"f-h": "e-g", "f-g": "e-f", "h": "g"}.get(
            panels, panels
        )
    elif source.startswith("source_data/theory/credit_capture"):
        figure = "Figure 4"
        panels = "a"
    elif source.startswith("source_data/reciprocal_routing/"):
        figure, panels = "Figure 7", "c"
    elif source.startswith("source_data/figure3/typed_only"):
        figure, panels = "Supplementary Figure 20", "h"
    elif source.startswith("source_data/figure3/"):
        figure = "Figure 7"
        panels = {"a-c": "a-b", "d-g": "d", "a-g": "a-b,d", "d-h": "d"}.get(
            panels, panels
        )
    elif source.startswith("source_data/figure4/direct_typed"):
        figure, panels = "Supplementary Figure 21", "g"
    elif source.startswith("source_data/figure4/"):
        if panels == "f":
            figure, panels = "Supplementary Figure 21", "f"
        else:
            figure = "Figure 8"
            panels = {"d-e": "c"}.get(panels, panels)
    elif source.startswith("source_data/focal_decomposition/"):
        figure, panels = "Figure 8", "d"
    elif source.startswith("source_data/physical_cable_sensitivity/"):
        figure, panels = "Figure 8", "e"
    elif source.startswith("source_data/focal_selectivity_active_ensemble/"):
        figure = "Figure 8"
        panels = {"a-d": "f-g", "b-d": "f-g", "b-c": "f-g", "c": "g"}.get(
            panels, panels
        )
    elif source.startswith("source_data/fulltree_boundary/"):
        figure = "Figure 9"
        panels = "f-g"
    elif source.startswith("source_data/figure5/"):
        if any(f"ch{value}" in source for value in (1, 2, 8)):
            figure, panels = "Supplementary Figure 22", "h"
        else:
            figure = "Figure 9"
            panels = {"a-c": "a", "d-g": "b-c", "h": "b-c"}.get(panels, panels)
    elif source.startswith("source_data/functional_topology_all_scans/"):
        figure, panels = "Supplementary Figure 22", "a-h"
    elif source.startswith("source_data/alignment_controlled/"):
        figure = "Figure 9"
        panels = {"a-d": "d", "b-d": "d", "b-c": "d", "d": "d"}.get(
            panels, panels
        )
    elif source.startswith("source_data/animal_learning_francioni/"):
        if "neuron_sd_residual" in source:
            figure, panels = "Supplementary Figure 17", "d"
        elif "common_signed_modes" in source:
            figure, panels = "Supplementary Figure 17", "c"
        else:
            figure, panels = "Figure 9", "e"
    elif source.startswith("source_data/pinky_v185_replication/") and item.figure == "Figure 7":
        figure, panels = "Figure 7", "g"
    elif source.startswith("source_data/capture_per_wire/"):
        figure = "Figure 7"
        panels = "e-f"

    if figure == item.figure and panels == item.panels:
        return item

    old_dir = item.figure.replace(" ", "_") + "/"
    new_dir = figure.replace(" ", "_") + "/"
    destination = item.destination.replace(old_dir, new_dir, 1)
    if figure.startswith("Supplementary Figure"):
        number = figure.split()[-1]
        destination = re.sub(
            r"^(?:Fig|SuppFig)\d+", f"SuppFig{number}", destination.split("/", 1)[1]
        )
        destination = new_dir + destination
    else:
        number = figure.split()[-1]
        destination = re.sub(
            r"^(?:Fig|SuppFig)\d+", f"Fig{number}", destination.split("/", 1)[1]
        )
        destination = new_dir + destination
        if panels != item.panels:
            destination = re.sub(
                rf"Fig{number}[^_]*_",
                f"Fig{number}{panels}_",
                destination,
            )
    if source.startswith("source_data/theory/credit_capture"):
        destination = "Figure_4/Fig4a_credit_capture_bound_verification.json"
    elif source.startswith("source_data/fulltree_boundary/"):
        destination = re.sub(r"Fig9[^_]*_", "Fig9f-g_", destination)
    elif source.startswith("source_data/alignment_controlled/"):
        destination = re.sub(r"Fig9[^_]*_", f"Fig9{panels}_", destination)
    elif source.startswith("source_data/animal_learning_francioni/"):
        prefix = (
            "SuppFig17d" if figure == "Supplementary Figure 17" else f"Fig9{panels}"
        )
        destination = re.sub(r"(?:Fig|SuppFig)\d+[^_]*_", f"{prefix}_", destination)
    elif source.startswith("source_data/capture_per_wire/"):
        destination = re.sub(r"(?:Fig|SuppFig)\d+[^_]*_", "Fig7e-f_", destination)
    elif source.startswith("source_data/credit_phase_plane/"):
        destination = "Figure_4/Fig4g_phase_plane_points.csv"
    return replace(item, figure=figure, panels=panels, destination=destination)


FILES = tuple(final_display_file(item) for item in FILES)

# Figure 4 is the continuous path-demand family; the
# existing subtree-factorial display is compressed into panels e--h.
_SUBTREE_PANEL_MAP = {
    "seed_outcomes.csv": "e-h",
    "condition_summary.csv": "f-h",
    "paired_contrasts.csv": "g-h",
    "mechanism_ledger.csv": "e-h",
    "summary.json": "e-h",
}
_remapped_files = []
for item in FILES:
    if (
        item.figure == "Figure 3"
        and item.source.startswith(
            "source_data/trained_subtree_address_full_factorial/"
        )
    ):
        panels = _SUBTREE_PANEL_MAP[Path(item.source).name]
        destination = re.sub(r"Fig3[^_]*_", f"Fig3{panels}_", item.destination)
        item = replace(item, panels=panels, destination=destination)
    _remapped_files.append(item)
FILES = tuple(_remapped_files)

# ── Renumber main-figure entries to the restructured figure order ────────
# The manuscript now runs: 1 framework, 2 feedback identity, 3 credit-operator
# theory, 4 branch conflict, 5 subtree routing, 6 physical depth, 7 anatomical
# routes, 8 focal shunting, 9 measured boundary.  Entries written under the
# older order are renumbered here by their SOURCE-DATA prefix, which names the
# experiment unambiguously; the destination folder and FigN filename prefix
# follow the corrected figure so no source-data folder describes the wrong
# experiment.
_PREFIX_TO_FIGURE = (
    ("source_data/theory/", "Figure 3"),
    ("source_data/credit_phase", "Figure 3"),
    ("source_data/path_necessity_fashion/", "Figure 4"),
    ("source_data/trained_subtree_address_full_factorial/", "Figure 5"),
    ("source_data/interior_optimum/", "Figure 5"),
    ("source_data/nonlinear_physical_depth", "Figure 6"),
    ("source_data/physical_depth_h4_factorial/", "Figure 6"),
    ("source_data/point_dendrite_credit_controls/", "Figure 6"),
    ("source_data/physical_alignment_dose/", "Figure 6"),
    ("source_data/figure3/", "Figure 7"),
    ("source_data/microns_inhibitory_routes/", "Figure 7"),
    ("source_data/capture_per_wire/", "Figure 7"),
    ("source_data/figure4/", "Figure 8"),
    ("source_data/focal_", "Figure 8"),
    ("source_data/physical_cable_sensitivity/", "Figure 8"),
    ("source_data/fulltree_boundary/", "Figure 9"),
    ("source_data/functional_topology_all_scans/", "Figure 9"),
    ("source_data/alignment_controlled/", "Figure 9"),
    ("source_data/animal_learning_francioni/", "Figure 9"),
)


def _renumber_main(item: SourceFile) -> SourceFile:
    if not item.figure.startswith("Figure "):
        return item
    for prefix, figure in _PREFIX_TO_FIGURE:
        if item.source.startswith(prefix):
            break
    else:
        return item
    if item.figure == figure:
        return item
    old_n = item.figure.split()[1]
    new_n = figure.split()[1]
    destination = item.destination
    if destination.startswith(f"Figure_{old_n}/"):
        destination = f"Figure_{new_n}/" + destination.split("/", 1)[1]
    destination = re.sub(rf"\bFig{old_n}(?=[a-z_\-])", f"Fig{new_n}",
                        destination)
    return replace(item, figure=figure, destination=destination)


FILES = tuple(_renumber_main(item) for item in FILES)

# The continuous path-demand family supersedes the two-stream endpoint in main
# Figure 2H and supplies the first four panels of the revised Figure 3.  The
# original two-stream source remains packaged with Supplementary Figure 19.
_PATH_DEMAND_FILES = (
    ("seed_outcomes.csv", "independent condition outcomes"),
    ("condition_summary.csv", "condition means and seed-bootstrap intervals"),
    ("paired_contrasts.csv", "paired route contrasts at every conflict dose"),
    ("interaction_summary.csv", "seed-wise route-by-conflict interaction summary"),
    ("seedwise_interactions.csv", "one primary route-by-conflict slope per independent seed"),
    ("boundary_summary.csv", "predicted and discrete empirical boundaries"),
    ("boundary_by_seed.csv", "seed-level utility and trained-accuracy boundary doses"),
    ("boundary_order.json", "seed-wise test of the predicted boundary ordering"),
    ("plotted_crossings.csv", "descriptive mean-curve crossings plotted in Supplementary Figure 29c"),
    ("mechanism_audit.csv", "task-balance and gradient implementation audit"),
    ("audit.json", "confirmatory interpretation-gate audit"),
)

for filename, role in _PATH_DEMAND_FILES:
    source = f"source_data/path_necessity_fashion/{filename}"
    suffix = Path(filename).stem
    common = dict(
        source=source,
        role=role,
        independent_unit="paired independent training seed (n=20 per branch count)",
        status="frozen confirmatory cohort",
        notes=(
            "Fashion-MNIST branch-conflict family with B=2,4,8 and eight "
            "nested conflict doses; exact/BP/gated-point equality was audited."
        ),
    )
    FILES += (
        SourceFile(
            figure="Figure 4",
            panels="c-f",
            destination=f"Figure_4/Fig4c-f_path_demand_{suffix}{Path(filename).suffix}",
            **common,
        ),
        SourceFile(
            figure="Supplementary Figure 29",
            panels="a-c",
            destination=(
                f"Supplementary_Figure_29/SuppFig29a-c_path_demand_"
                f"{suffix}{Path(filename).suffix}"
            ),
            **common,
        ),
    )


def duplicate_for_supplement(item: SourceFile, number: int, panels: str) -> SourceFile:
    """Reuse one numerical source for a detailed supplementary display."""

    basename = Path(item.destination).name
    suffix = basename.split("_", 1)[1] if "_" in basename else basename
    destination = f"Supplementary_Figure_{number}/SuppFig{number}{panels}_{suffix}"
    return replace(
        item,
        figure=f"Supplementary Figure {number}",
        panels=panels,
        destination=destination,
    )


# Focused main figures and full diagnostic SI figures intentionally reuse the
# same frozen tables.  Record both display destinations explicitly.
FILES += tuple(
    duplicate_for_supplement(item, 20, "a-j")
    for item in FILES
    if item.source.startswith(
        ("source_data/figure3/", "source_data/reciprocal_routing/")
    )
    and item.figure != "Supplementary Figure 20"
)
FILES += tuple(
    duplicate_for_supplement(item, 21, "a-i")
    for item in FILES
    if item.source.startswith(
        (
            "source_data/figure4/",
            "source_data/focal_decomposition/",
            "source_data/physical_cable_sensitivity/",
        )
    )
    and item.figure != "Supplementary Figure 21"
)
FILES += tuple(
    duplicate_for_supplement(item, 22, "a-h")
    for item in FILES
    if item.source.startswith(
        ("source_data/figure5/", "source_data/functional_topology_all_scans/")
    )
    and item.figure != "Supplementary Figure 22"
)
FILES += tuple(
    duplicate_for_supplement(item, 18, "a-b")
    for item in FILES
    if item.source.startswith("source_data/point_dendrite_credit_controls/")
    and item.figure != "Supplementary Figure 18"
)
FILES += tuple(
    duplicate_for_supplement(item, 28, "a-b")
    for item in FILES
    if item.source.startswith("source_data/point_dendrite_credit_controls/")
    and item.figure == "Figure 6"
)
FILES += tuple(
    duplicate_for_supplement(item, 19, "a-c")
    for item in FILES
    if item.source.startswith("source_data/prospective_input_validity/central_valid_")
    and item.figure == "Figure 2"
)
FILES += tuple(
    duplicate_for_supplement(item, 19, "d")
    for item in FILES
    if item.source.startswith("source_data/prospective_input_validity/")
    and Path(item.source).name in {
        "followup_publication_seed_outcomes.csv",
        "routing_valid_paired_contrasts.csv",
    }
    and item.figure == "Figure 2"
)

# Several detailed supplementary displays reuse cohorts that also support
# condensed main-figure panels.  Keep explicit supplementary destinations so
# the submission manifest maps every displayed panel rather than silently
# assigning the shared numerical table only to its main-figure use.
FILES += (
    SourceFile(
        "Supplementary Figure 5",
        "b-h",
        "source_data/alignment_controlled/alignment_controlled_runs.csv.gz",
        "Supplementary_Figure_5/SuppFig5b-h_alignment_controlled_runs.csv.gz",
        "cell-by-stream source values",
        "reconstructed cell; Monte Carlo streams nested within cell (n=8 cells)",
        "current controlled sufficiency test",
        "Source values across imposed alignment, routing dictionary, and learning endpoint.",
    ),
    SourceFile(
        "Supplementary Figure 5",
        "c-h",
        "source_data/alignment_controlled/alignment_controlled_curves.csv",
        "Supplementary_Figure_5/SuppFig5c-h_alignment_controlled_curves.csv",
        "derived plotting summary",
        "reconstructed cell (n=8)",
        "current controlled sufficiency test",
        "Displayed capture, one-step, iterative-progress, and route-control summaries.",
    ),
    SourceFile(
        "Supplementary Figure 5",
        "g-h",
        "source_data/alignment_controlled/cell_alignment_metrics.csv",
        "Supplementary_Figure_5/SuppFig5g-h_cell_alignment_metrics.csv",
        "independent-cell values",
        "reconstructed cell (n=8)",
        "current controlled sufficiency test",
        "Cell-level capture and learning-progress associations.",
    ),
    SourceFile(
        "Supplementary Figure 5",
        "a",
        "source_data/alignment_controlled/dictionary_audit.csv",
        "Supplementary_Figure_5/SuppFig5a_dictionary_audit.csv",
        "dictionary construction audit",
        "reconstructed cell; streams nested within cell",
        "current controlled sufficiency test",
        "Requested channels, effective rank, wiring, and coordinate counts.",
    ),
    SourceFile(
        "Supplementary Figure 5",
        "a-h",
        "source_data/alignment_controlled/summary.json",
        "Supplementary_Figure_5/SuppFig5_alignment_summary.json",
        "derived statistical summary",
        "reconstructed cell (n=8)",
        "current controlled sufficiency test",
        "Frozen design, endpoint contrasts, and scope boundaries.",
    ),
    SourceFile(
        "Supplementary Figure 8",
        "a-d,g-i",
        "source_data/prospective_input_validity/followup_publication_condition_summary.csv",
        "Supplementary_Figure_8/SuppFig8a-d_g-i_condition_summary.csv",
        "fixed-budget condition summaries",
        "paired independent training seed (n=10 per condition)",
        "current complete prospective cohort",
        "Fixed-terminal-branch and input-contact depth profiles and resource summaries.",
    ),
    SourceFile(
        "Supplementary Figure 8",
        "e-f",
        "source_data/prospective_input_validity/followup_publication_paired_contrasts.csv",
        "Supplementary_Figure_8/SuppFig8e-f_paired_contrasts.csv",
        "prespecified fixed-budget contrasts",
        "paired independent training seed (n=10 per condition)",
        "current derived analysis",
        "Depth-four-minus-depth-one and local-minus-backpropagation contrasts.",
    ),
    SourceFile(
        "Supplementary Figure 8",
        "a-i",
        "source_data/prospective_input_validity/followup_publication_seed_outcomes.csv",
        "Supplementary_Figure_8/SuppFig8a-i_seed_outcomes.csv",
        "complete follow-up run outcomes",
        "paired independent training seed (n=10 per condition)",
        "current complete prospective cohort",
        "All 160 input-valid additive fixed-budget outcomes.",
    ),
)


FILES += (
    SourceFile(
        "Supplementary Figure 23",
        "a-c",
        "source_data/trained_partition_residual/seed_state_residuals.csv",
        "Supplementary_Figure_23/SuppFig23a-c_seed_state_residuals.csv",
        "complete seed-by-condition residuals at initialization and training",
        "paired independent training seed (n=20)",
        "post-hoc deterministic reconstruction with frozen-script hash and endpoint gates",
        "All 2,700 trained route-factorial fits reconstructed at two states; includes address, coefficient and total residuals.",
    ),
    SourceFile(
        "Supplementary Figure 23",
        "a-c",
        "source_data/trained_partition_residual/condition_summary.csv",
        "Supplementary_Figure_23/SuppFig23a-c_condition_summary.csv",
        "derived condition summaries",
        "paired independent training seed (n=20)",
        "current derived analysis",
        "Condition means and seed-bootstrap intervals for partition capture and residual components.",
    ),
    SourceFile(
        "Supplementary Figure 23",
        "c",
        "source_data/trained_partition_residual/seed_level_associations.csv",
        "Supplementary_Figure_23/SuppFig23c_seed_level_associations.csv",
        "within-seed capture--accuracy associations",
        "paired independent training seed (n=20)",
        "current derived analysis",
        "Spearman correlations are computed within seed for all conditions and the K<8 restricted set.",
    ),
    SourceFile(
        "Supplementary Figure 23",
        "a-c",
        "source_data/trained_partition_residual/reconstruction_audit.csv",
        "Supplementary_Figure_23/SuppFig23_reconstruction_audit.csv",
        "archived-versus-reconstructed endpoint audit",
        "complete 2,700-fit cohort",
        "hash-gated numerical audit",
        "Reconstructed losses and accuracies paired to every archived seed--condition endpoint.",
    ),
    SourceFile(
        "Supplementary Figure 23",
        "a-c",
        "source_data/trained_partition_residual/summary.json",
        "Supplementary_Figure_23/SuppFig23_summary.json",
        "analysis and validity summary",
        "complete 2,700-fit cohort and n=20 independent seeds",
        "current audit",
        "Frozen hashes, endpoint agreement, Pythagorean closure and seed-level associations.",
    ),
)


FILES += (
    SourceFile(
        "Supplementary Figure 24",
        "b-d",
        "source_data/adaptive_conductance_reliability/seed_outcomes.csv",
        "Supplementary_Figure_24/SuppFig24b-d_seed_outcomes.csv",
        "complete seed-by-condition outcomes",
        "paired independent simulation seed (n=50)",
        "fresh prospectively frozen adaptive-reliability experiment",
        "All 1,050 outcomes across three heterogeneity levels and seven paired methods.",
    ),
    SourceFile(
        "Supplementary Figure 24",
        "b",
        "source_data/adaptive_conductance_reliability/branch_estimates.csv",
        "Supplementary_Figure_24/SuppFig24b_branch_estimates.csv",
        "branch-level oracle and adaptive gains",
        "paired independent simulation seed (n=50); branches nested within seed",
        "fresh prospectively frozen adaptive-reliability experiment",
        "Fixed initial-oracle and final adaptive gain estimates for every branch.",
    ),
    SourceFile(
        "Supplementary Figure 24",
        "c",
        "source_data/adaptive_conductance_reliability/condition_summary.csv",
        "Supplementary_Figure_24/SuppFig24c_condition_summary.csv",
        "derived condition summaries",
        "paired independent simulation seed (n=50)",
        "current derived analysis",
        "Means and seed-bootstrap intervals for one-step and final outcomes.",
    ),
    SourceFile(
        "Supplementary Figure 24",
        "d",
        "source_data/adaptive_conductance_reliability/paired_contrasts.csv",
        "Supplementary_Figure_24/SuppFig24d_paired_contrasts.csv",
        "paired high-heterogeneity contrasts",
        "paired independent simulation seed (n=50)",
        "frozen primary and boundary analysis",
        "Adaptive-local comparisons with global, shuffled, no-shunt, initial-oracle and point-gate controls.",
    ),
    SourceFile(
        "Supplementary Figure 24",
        "b",
        "source_data/adaptive_conductance_reliability/estimator_seed_summary.csv",
        "Supplementary_Figure_24/SuppFig24b_estimator_seed_summary.csv",
        "seed-level estimator diagnostics",
        "paired independent simulation seed (n=50)",
        "secondary descriptive analysis",
        "Within-seed gain-order correlation and mean absolute gain error.",
    ),
    SourceFile(
        "Supplementary Figure 24",
        "a-d",
        "source_data/adaptive_conductance_reliability/summary.json",
        "Supplementary_Figure_24/SuppFig24_summary.json",
        "analysis and validity summary",
        "complete 1,050-outcome cohort",
        "current audit",
        "Frozen hashes, numerical gates and claim boundary.",
    ),
)


FILES += (
    SourceFile(
        "Supplementary Figure 25",
        "b-d",
        "source_data/irregular_tree_wavelets/cell_scale_summary.csv",
        "Supplementary_Figure_25/SuppFig25b-d_cell_scale_summary.csv",
        "cell-level multiscale route-energy endpoints",
        "reconstructed cell (n=47 primary; n=8 secondary)",
        "prospectively frozen structural reanalysis",
        "Actual, isotropic and column-permuted route-energy summaries at coarse, intermediate and fine scales.",
    ),
    SourceFile(
        "Supplementary Figure 25",
        "b-c",
        "source_data/irregular_tree_wavelets/cohort_scale_summary.csv",
        "Supplementary_Figure_25/SuppFig25b-c_cohort_scale_summary.csv",
        "cohort-level multiscale summaries",
        "reconstructed cell",
        "current derived analysis",
        "Cell means and cell-bootstrap intervals for scale energy and enrichment.",
    ),
    SourceFile(
        "Supplementary Figure 25",
        "a-c",
        "source_data/irregular_tree_wavelets/mode_spectrum.csv",
        "Supplementary_Figure_25/SuppFig25a-c_mode_spectrum.csv",
        "complete irregular-tree wavelet spectrum",
        "wavelet mode nested within reconstructed cell",
        "current derived analysis",
        "Mode support, split balance, scale assignment and route-field energy for all 55 cells.",
    ),
    SourceFile(
        "Supplementary Figure 25",
        "a-d",
        "source_data/irregular_tree_wavelets/numerical_audit.csv",
        "Supplementary_Figure_25/SuppFig25_numerical_audit.csv",
        "basis and decomposition audit",
        "reconstructed cell (n=55)",
        "frozen numerical audit",
        "Full-rank, orthonormality, energy-decomposition and scale-count gates.",
    ),
    SourceFile(
        "Supplementary Figure 25",
        "a-d",
        "source_data/irregular_tree_wavelets/summary.json",
        "Supplementary_Figure_25/SuppFig25_summary.json",
        "analysis and inferential summary",
        "reconstructed cell (n=47 primary; n=8 secondary)",
        "current audit",
        "Input hashes, frozen settings, numerical gates, cell-level tests and claim boundaries.",
    ),
)


# Final fixed-depth task-family boundary and independent-animal structural
# replication. These are explicit so the source-data release cannot lag the
# compiled Figure 6, Figure 7 or Supplementary Figure 27.
FILES += (
    SourceFile(
        "Figure 6",
        "e-g",
        "source_data/task_family_alignment/seed_outcomes.csv",
        "Figure_6/Fig6e-g_task_family_seed_outcomes.csv",
        "complete run-level outcomes",
        "paired independent training seed (n=10 per condition)",
        "fresh fixed-D3 factorial",
        "All 360 fits across three task families, two architectures, two credit rules and three alignment levels.",
    ),
    SourceFile(
        "Figure 6",
        "e-f",
        "source_data/task_family_alignment/condition_summary.csv",
        "Figure_6/Fig6e-f_task_family_condition_summary.csv",
        "derived condition summaries",
        "paired independent training seed (n=10)",
        "current derived analysis",
        "Architecture effects with paired-seed bootstrap intervals.",
    ),
    SourceFile(
        "Figure 6",
        "g",
        "source_data/task_family_alignment/paired_contrasts.csv",
        "Figure_6/Fig6g_task_family_paired_contrasts.csv",
        "paired seed contrasts",
        "paired independent training seed (n=10)",
        "current derived analysis",
        "Architecture-by-alignment interactions and task-family difference-in-differences.",
    ),
    SourceFile(
        "Figure 6",
        "e-g",
        "source_data/task_family_alignment/audit.json",
        "Figure_6/Fig6e-g_task_family_audit.json",
        "artifact and claim audit",
        "complete 360-fit cohort",
        "current audit",
        "Completeness, finite-metric, no-fallback, seed and exact-resource gates.",
    ),
    SourceFile(
        "Figure 7",
        "g",
        "source_data/pinky_v185_replication/routing/k4_cross_animal_contrasts.csv",
        "Figure_7/Fig7g_independent_animal_K4_contrasts.csv",
        "cell-level cross-animal contrasts",
        "reconstructed cell; animal is the biological replication unit (two mice)",
        "independent-animal directional replication",
        "K=4 ancestry-route advantages shown separately for the original and Pinky v185 MICRONS animals.",
    ),
    SourceFile(
        "Supplementary Figure 27",
        "a",
        "source_data/pinky_v185_replication/cohort_manifest.csv",
        "Supplementary_Figure_27/SuppFig27a_cohort_manifest.csv",
        "outcome-independent cohort selection",
        "selected reconstructed cell (n=12; one second mouse)",
        "frozen before route-capacity analysis",
        "Equal-count y strata with selection nearest the global x/z medians.",
    ),
    SourceFile(
        "Supplementary Figure 27",
        "a-b,d",
        "source_data/pinky_v185_replication/routing/cell_metrics.csv",
        "Supplementary_Figure_27/SuppFig27a-b_d_cell_metrics.csv",
        "cell-level morphology, QC and route summaries",
        "QC-passing reconstructed cell (n=10; one second mouse)",
        "current independent-animal analysis",
        "Cell-level QC, wiring and four-channel capture values.",
    ),
    SourceFile(
        "Supplementary Figure 27",
        "b,d",
        "source_data/pinky_v185_replication/routing/feedback_compression_curves.csv",
        "Supplementary_Figure_27/SuppFig27b_d_feedback_compression_curves.csv",
        "cell-level route-capacity curves",
        "QC-passing reconstructed cell (n=10; one second mouse)",
        "current independent-animal analysis",
        "Capture and weighted residual across channel budgets for ancestry routes and controls.",
    ),
    SourceFile(
        "Supplementary Figure 27",
        "c",
        "source_data/pinky_v185_replication/routing/k4_cross_animal_contrasts.csv",
        "Supplementary_Figure_27/SuppFig27c_cross_animal_K4_contrasts.csv",
        "cell-level cross-animal contrasts",
        "reconstructed cell; animal is the biological replication unit (two mice)",
        "independent-animal directional replication",
        "K=4 ancestry-route advantages shown separately by animal; within-volume intervals are descriptive.",
    ),
    SourceFile(
        "Supplementary Figure 27",
        "a-d",
        "source_data/pinky_v185_replication/preparation_manifest.json",
        "Supplementary_Figure_27/SuppFig27_preparation_manifest.json",
        "preprocessing and archive audit",
        "12 outcome-independently selected cells",
        "current audit",
        "Archive checksums, component selection, skeletonization, radius estimation and direct-type QC provenance.",
    ),
    SourceFile(
        "Supplementary Figure 27",
        "a-d",
        "source_data/pinky_v185_replication/routing/summary.json",
        "Supplementary_Figure_27/SuppFig27_summary.json",
        "analysis and validity summary",
        "12 selected cells, 10 QC-passing; one second mouse",
        "current audit",
        "Frozen route settings, QC results, within-volume summaries and cross-animal direction checks.",
    ),
)

# ── Panel-span ground-truth corrections (2026-08-28) ─────────────────────
# Every main-figure entry above was audited against the panel its source is
# actually drawn in by the current native builders.  Entries whose data feed
# no drawn panel become Text_ companions; entries duplicated verbatim by a
# supplementary home are dropped; three files move to the figure that really
# draws them.  Keys are the pre-correction destinations, which are unique.
# None: drop the entry (a correct duplicate elsewhere already packages it).
_PANEL_CORRECTIONS = {
    # Figure 2 lost its old panels G-H; the two-stream data live in S19.
    "Figure_2/Fig2g_seed_outcomes.csv": None,
    "Figure_2/Fig2g_condition_summary.csv": None,
    "Figure_2/Fig2g_paired_contrasts.csv": None,
    "Figure_2/Fig2b_mnist_feedback_paired_contrasts.csv":
        (None, "g", "Figure_2/Fig2g_mnist_feedback_paired_contrasts.csv"),
    "Figure_2/Fig2c_Fashion_MNIST_condition_summary.csv":
        (None, "c", "Figure_2/Fig2c_Fashion_MNIST_condition_summary.csv"),
    "Figure_2/Fig2c_Fashion_MNIST_paired_contrasts.csv":
        (None, "text", "Figure_2/Text_Fashion_MNIST_paired_contrasts.csv"),
    "Figure_2/Fig2c_Fashion_MNIST_audit.json":
        (None, "text", "Figure_2/Text_Fashion_MNIST_audit.json"),
    # Figure 3 letters follow the native seven-panel builder.
    "Figure_3/Fig3c-d_depth_training_seed.csv":
        (None, "c", "Figure_3/Fig3c_depth_training_seed.csv"),
    "Figure_3/Fig3e_projection_phase_seed.csv":
        (None, "d", "Figure_3/Fig3d_projection_phase_seed.csv"),
    "Figure_3/Fig3f_reliability_phase_seed.csv":
        (None, "e", "Figure_3/Fig3e_reliability_phase_seed.csv"),
    "Figure_3/Fig3g-h_same_span_diagnostics.csv":
        (None, "text", "Figure_3/Text_same_span_diagnostics.csv"),
    "Figure_3/Fig3i_operator_metrics.csv":
        (None, "f", "Figure_3/Fig3f_operator_metrics.csv"),
    # The capture-bound audit backs the topology-matched capacity note
    # behind Figure 7, not any drawn Figure 3 panel.
    "Figure_3/Fig3a_credit_capture_bound_verification.json":
        ("Figure 7", "text",
         "Figure_7/Text_credit_capture_bound_verification.json"),
    # Figure 4: d,f draw condition summaries, e draws the paired seed
    # clouds and slopes, c-d mark the analytic boundaries.
    "Figure_4/Fig4c-f_path_demand_seed_outcomes.csv":
        (None, "e", "Figure_4/Fig4e_path_demand_seed_outcomes.csv"),
    # 2026-08-29: Figure 4's panels E and F were merged and the boundary test
    # promoted from Supplementary Fig. S29c, so the drawn homes moved.
    "Figure_4/Fig4c-f_path_demand_condition_summary.csv":
        (None, "d,e", "Figure_4/Fig4d,e_path_demand_condition_summary.csv"),
    "Figure_4/Fig4c-f_path_demand_paired_contrasts.csv":
        (None, "text", "Figure_4/Text_path_demand_paired_contrasts.csv"),
    "Figure_4/Fig4c-f_path_demand_interaction_summary.csv":
        (None, "text", "Figure_4/Text_path_demand_interaction_summary.csv"),
    "Figure_4/Fig4c-f_path_demand_seedwise_interactions.csv":
        (None, "text", "Figure_4/Text_path_demand_seedwise_interactions.csv"),
    "Figure_4/Fig4c-f_path_demand_boundary_summary.csv":
        (None, "c-d", "Figure_4/Fig4c-d_path_demand_boundary_summary.csv"),
    "Figure_4/Fig4c-f_path_demand_boundary_by_seed.csv": None,
    "Figure_4/Fig4c-f_path_demand_boundary_order.json": None,
    "Figure_4/Fig4c-f_path_demand_plotted_crossings.csv":
        (None, "f", "Figure_4/Fig4f_path_demand_plotted_crossings.csv"),
    "Figure_4/Fig4c-f_path_demand_mechanism_audit.csv":
        (None, "text", "Figure_4/Text_path_demand_mechanism_audit.csv"),
    "Figure_4/Fig4c-f_path_demand_audit.json":
        (None, "text", "Figure_4/Text_path_demand_audit.json"),
    # Supplementary Figure 29: panel a is a data-free schematic.
    "Supplementary_Figure_29/SuppFig29a-c_path_demand_seed_outcomes.csv":
        (None, "b-c",
         "Supplementary_Figure_29/SuppFig29b-c_path_demand_seed_outcomes.csv"),
    "Supplementary_Figure_29/SuppFig29a-c_path_demand_condition_summary.csv":
        (None, "b-c",
         "Supplementary_Figure_29/"
         "SuppFig29b-c_path_demand_condition_summary.csv"),
    "Supplementary_Figure_29/SuppFig29a-c_path_demand_paired_contrasts.csv":
        (None, "text",
         "Supplementary_Figure_29/Text_path_demand_paired_contrasts.csv"),
    "Supplementary_Figure_29/SuppFig29a-c_path_demand_interaction_summary.csv":
        (None, "text",
         "Supplementary_Figure_29/Text_path_demand_interaction_summary.csv"),
    "Supplementary_Figure_29/"
    "SuppFig29a-c_path_demand_seedwise_interactions.csv":
        (None, "text",
         "Supplementary_Figure_29/Text_path_demand_seedwise_interactions.csv"),
    "Supplementary_Figure_29/SuppFig29a-c_path_demand_boundary_summary.csv":
        (None, "b-c",
         "Supplementary_Figure_29/SuppFig29b-c_path_demand_boundary_summary.csv"),
    "Supplementary_Figure_29/SuppFig29a-c_path_demand_boundary_by_seed.csv":
        (None, "c",
         "Supplementary_Figure_29/SuppFig29c_path_demand_boundary_by_seed.csv"),
    "Supplementary_Figure_29/SuppFig29a-c_path_demand_boundary_order.json":
        (None, "c",
         "Supplementary_Figure_29/SuppFig29c_path_demand_boundary_order.json"),
    "Supplementary_Figure_29/SuppFig29a-c_path_demand_plotted_crossings.csv":
        (None, "c",
         "Supplementary_Figure_29/SuppFig29c_path_demand_plotted_crossings.csv"),
    "Supplementary_Figure_29/SuppFig29a-c_path_demand_mechanism_audit.csv":
        (None, "text",
         "Supplementary_Figure_29/Text_path_demand_mechanism_audit.csv"),
    "Supplementary_Figure_29/SuppFig29a-c_path_demand_audit.json":
        (None, "text", "Supplementary_Figure_29/Text_path_demand_audit.json"),
    # Figure 5: e bandwidth sweep, f route contrasts, g paired bootstrap.
    "Figure_5/Fig5e-h_seed_outcomes.csv":
        (None, "e,g", "Figure_5/Fig5e,g_seed_outcomes.csv"),
    "Figure_5/Fig5f-h_condition_summary.csv":
        (None, "e", "Figure_5/Fig5e_condition_summary.csv"),
    "Figure_5/Fig5g-h_paired_contrasts.csv":
        (None, "f", "Figure_5/Fig5f_paired_contrasts.csv"),
    "Figure_5/Fig5e-h_mechanism_ledger.csv":
        (None, "text", "Figure_5/Text_subtree_factorial_mechanism_ledger.csv"),
    "Figure_5/Fig5e-h_summary.json":
        (None, "text", "Figure_5/Text_subtree_factorial_summary.json"),
    # Figure 6: c is the H=3 boundary with point controls, d-e the H4
    # heatmaps, f-g the task-family boundary; contrasts back prose.
    "Figure_6/Fig6c-f_seed_outcomes.csv":
        (None, "c", "Figure_6/Fig6c_seed_outcomes.csv"),
    "Figure_6/Fig6c-f_condition_summary.csv":
        (None, "c", "Figure_6/Fig6c_condition_summary.csv"),
    "Figure_6/Fig6c-f_paired_contrasts.csv":
        (None, "text", "Figure_6/Text_H3_paired_contrasts.csv"),
    "Figure_6/Text_artifact_and_claim_audit.json": (None, "text", None),
    "Figure_6/Text_checkpoint_gradient_summary.csv": (None, "text", None),
    "Figure_6/Fig6g-h_point_dendrite_seed_outcomes.csv":
        (None, "c", "Figure_6/Fig6c_point_dendrite_seed_outcomes.csv"),
    "Figure_6/Fig6g-h_point_dendrite_condition_summary.csv":
        (None, "c", "Figure_6/Fig6c_point_dendrite_condition_summary.csv"),
    "Figure_6/Fig6g-h_point_dendrite_paired_contrasts.csv":
        (None, "text", "Figure_6/Text_point_dendrite_paired_contrasts.csv"),
    "Figure_6/Fig6g-h_point_dendrite_audit.json":
        (None, "text", "Figure_6/Text_point_dendrite_audit.json"),
    "Figure_6/Fig6a-d_H4_seed_outcomes.csv":
        (None, "d-e", "Figure_6/Fig6d-e_H4_seed_outcomes.csv"),
    "Figure_6/Fig6a-d_H4_condition_summary.csv":
        (None, "d", "Figure_6/Fig6d_H4_condition_summary.csv"),
    "Figure_6/Fig6c_H4_paired_contrasts.csv":
        (None, "text", "Figure_6/Text_H4_paired_contrasts.csv"),
    "Figure_6/Fig6a-d_H4_audit.json":
        (None, "text", "Figure_6/Text_H4_audit.json"),
    "Figure_6/Fig6e-g_task_family_seed_outcomes.csv":
        (None, "f-g", "Figure_6/Fig6f-g_task_family_seed_outcomes.csv"),
    "Figure_6/Fig6e-f_task_family_condition_summary.csv":
        (None, "f-g", "Figure_6/Fig6f-g_task_family_condition_summary.csv"),
    "Figure_6/Fig6g_task_family_paired_contrasts.csv":
        (None, "text", "Figure_6/Text_task_family_paired_contrasts.csv"),
    "Figure_6/Fig6e-g_task_family_audit.json":
        (None, "text", "Figure_6/Text_task_family_audit.json"),
    # Figure 7: only panel A draws segment metrics; the cell inventory
    # supports the cohort prose (full detail lives in S20).
    "Figure_7/Fig7a-b_segment_metrics.csv":
        (None, "a", "Figure_7/Fig7a_segment_metrics.csv"),
    "Figure_7/Fig7a-b,d_cell_metrics.csv":
        (None, "text", "Figure_7/Text_cell_metrics.csv"),
    # Figure 8: e active dose, f electrotonic boundary, g paired forest.
    "Figure_8/Fig8e_physical_cable_contrasts.csv":
        (None, "f", "Figure_8/Fig8f_physical_cable_contrasts.csv"),
    "Figure_8/Fig8f-g_active_cell_metrics.csv":
        (None, "g", "Figure_8/Fig8g_active_cell_metrics.csv"),
    "Figure_8/Fig8f-g_active_condition_summary.csv":
        (None, "e", "Figure_8/Fig8e_active_condition_summary.csv"),
    "Figure_8/Fig8f-g_active_acceptance_ledger.csv":
        (None, "text", "Figure_8/Text_active_acceptance_ledger.csv"),
    "Figure_8/Fig8f-g_active_summary.json":
        (None, "text", "Figure_8/Text_active_summary.json"),
    # Figure 9: a pipeline schematic, b-c full-tree and topology rows,
    # d alignment schematic, e controlled curves, f animal contrast
    # with its mode partition, g evidence ladder.
    "Figure_9/Fig9f-g_full_tree_cell_method_means.csv":
        (None, "b-c", "Figure_9/Fig9b-c_full_tree_cell_method_means.csv"),
    "Figure_9/Fig9f-g_full_tree_scan_method_means.csv":
        (None, "b-c", "Figure_9/Fig9b-c_full_tree_scan_method_means.csv"),
    "Figure_9/Fig9f-g_full_tree_summary.json":
        (None, "text", "Figure_9/Text_full_tree_summary.json"),
    "Figure_9/Fig9d_alignment_controlled_runs.csv.gz":
        (None, "e", "Figure_9/Fig9e_alignment_controlled_runs.csv.gz"),
    "Figure_9/Fig9d_alignment_controlled_curves.csv":
        (None, "e", "Figure_9/Fig9e_alignment_controlled_curves.csv"),
    "Figure_9/Fig9d_cell_alignment_metrics.csv":
        (None, "e", "Figure_9/Fig9e_cell_alignment_metrics.csv"),
    "Figure_9/Fig9d_dictionary_audit.csv":
        (None, "e", "Figure_9/Fig9e_dictionary_audit.csv"),
    "Figure_9/Fig9d_alignment_summary.json":
        (None, "e", "Figure_9/Fig9e_alignment_summary.json"),
    "Figure_9/Fig9e_animal_signed_contrasts.csv":
        (None, "f", "Figure_9/Fig9f_animal_signed_contrasts.csv"),
    "Figure_9/Fig9e_animal_summary.json":
        (None, "f", "Figure_9/Fig9f_animal_summary.json"),
    "Figure_9/Fig9a_functional_target_metrics.csv":
        (None, "c", "Figure_9/Fig9c_functional_target_metrics.csv"),
    "Figure_9/Fig9a_functional_summary.json":
        (None, "text", "Figure_9/Text_functional_summary.json"),
    # The ch4 task table is drawn in the Figure 3 phase plane (its ch1/2/8
    # companions, packaged with Supplementary Figure 22, set the dense-
    # oracle effective rank); the ch4 summary is S22 text support.
    "Figure_9/Fig9b-c_task_target_method_means_ch4.csv":
        ("Figure 3", "g", "Figure_3/Fig3g_task_target_method_means_ch4.csv"),
    "Figure_9/Fig9b-c_task_summary_ch4.json": None,
    # Supplementary Figure 19: panel letters follow the native sheet; the
    # S8-lettered duplicates of its panel-f files are dropped.
    "Supplementary_Figure_19/Text_clean_exact_bp_audit_summary.json":
        (None, "text", None),
    "Supplementary_Figure_19/SuppFig19a-d_g-i_condition_summary.csv": None,
    "Supplementary_Figure_19/SuppFig19e-f_paired_contrasts.csv": None,
    # Supplementary Figure 17 keeps only old panels A and D (relettered
    # A-B); the common-signed-modes table backs the signed-mode prose.
    "Supplementary_Figure_17/SuppFig17d_common_signed_modes.csv":
        (None, "text", "Supplementary_Figure_17/Text_common_signed_modes.csv"),
    "Supplementary_Figure_17/SuppFig17d_neuron_residual_distributions.csv":
        (None, "b",
         "Supplementary_Figure_17/SuppFig17b_neuron_residual_distributions.csv"),
}

_corrected_files = []
_correction_hits = set()
for item in FILES:
    correction = _PANEL_CORRECTIONS.get(item.destination, "keep")
    if correction != "keep":
        _correction_hits.add(item.destination)
    if correction is None:
        continue
    if correction != "keep":
        figure, panels, destination = correction
        item = replace(
            item,
            figure=figure if figure is not None else item.figure,
            panels=panels,
            destination=(destination if destination is not None
                         else item.destination),
        )
    _corrected_files.append(item)
_missing_corrections = set(_PANEL_CORRECTIONS) - _correction_hits
if _missing_corrections:
    raise RuntimeError(
        "panel corrections matched no entry: "
        + ", ".join(sorted(_missing_corrections))
    )
FILES = tuple(_corrected_files)


README = """# Source Data

This directory contains the numerical source data underlying every current
main and supplementary figure panel and each quantitative main-text control.
Files are grouped by
display item, and `manifest.tsv` maps each file to its panel, role, highest
independent unit, status, original journal-package source, byte size, and
SHA-256 digest.

Figure 1 is a programmatic conceptual schematic and has no numerical source
data. Its directory contains a short scope note.

The independent unit is a training seed for the artificial-tree learning
experiments, a reconstructed cell for within-volume anatomical and focal
analyses, and a postsynaptic target cell for the measured-response analyses.
The primary MICRONS analyses use one mouse; Supplementary Figure 27 adds a
second mouse for structural route capacity. Cells nested within a volume are
not independent biological replicates at the animal level, and two animals do
not support population-level animal inference.
Focal sites, channel draws, scans, stimulus splits, and Monte Carlo streams are
nested observations; they are not counted as independent biological
replicates.

Figure 2 contains the feedback-coordinate, ownership, input-validity and
Fashion-MNIST ladders. Figure 3 contains the stochastic credit-operator phase
tests and the alignment--bandwidth synthesis. Figure 4 is the continuous
branch-conflict path-demand family. Figure 5 is the 2,700-fit subtree-address
factorial. Figure 6 contains the H2--H4 physical-depth, point--dendrite,
BP--local-credit and task-family-by-alignment extensions. Figures
7--9 contain reconstructed-anatomy capacity, focal conductance,
measured-response, controlled-alignment and six-animal boundary tests; Figure
7 also shows the independent-animal directional replication of structural
route capacity.
Supplementary Figures 1--3 reproduce the three unchanged regular-tree figures
from the final NeurIPS/arXiv revision, and Supplementary Figure 4 is the
reduced four-panel regular-tree boundary sheet. Supplementary Figure 5 is a constructed
topology--task alignment control, Supplementary Figure 6 is an exact
deterministic interference calculation, and Supplementary Figure 7 audits the
fixed spatial-connectivity control. Supplementary Figure 8 is the fixed-budget
depth control, Supplementary Figure 9 is the complete checkpoint diagnostic,
Supplementary Figure 10 is the same-mouse v661 sensitivity cohort, and
Supplementary Figures 11--17 contain conductance, inhibitory, same-span,
calibration, interior-optimum and external-animal analyses. Supplementary
Figures 18--29 retain the expanded physical-depth, prospective routing,
morphology, focal-shunting, measured-response and trained partition-residual
diagnostics plus the adaptive conductance-reliability and irregular-tree
wavelet tests, clean-source physical-depth replication, independent-animal
structural replication, physical-depth diagnostics and the complete
branch-conflict boundary. Capture per wire
and the phase-plane synthesis are promoted to main Figures 7 and 3.

This is the submission-facing source-data package. Figure 2 panels b and c
contain the complete clean 15-seed feedback cohort and the corresponding
fixed-checkpoint diagnostics. Panel d includes all 25 independent-seed values
and its verified aggregate summaries. Run-level data and derived summaries for
the initialization-policy factorial and direct-type-only focal sensitivity are
also included. Figure 2 contains the 480 input-valid primary depth-by-feedback
outcomes, 120 input-valid bandwidth-matched routing outcomes, the detached
320-run exact/backpropagation audit, and 160 additive fixed-budget outcomes.
The complete 1,840-run historical input-validity ledger is supplied, including
the excluded inhibitory-dose family. The 120-checkpoint diagnostic is supplied
as Supplementary Figure 9; Figure 7 is the reconstructed-morphology routing
analysis, Figure 4 the branch-conflict family, Figure 5 the subtree-address
factorial and Figure 6 the physical-depth program.
Supplementary Figure 4 retains the historical depth, teaching-noise and
shunting-only CIFAR-10 boundaries together with the fresh raw-additive
CIFAR-10 ladder. The omitted historical panels are represented in
Supplementary Figures 1--3. No current figure panel or
quantitative main-text control has a known source-data completeness gap.

The public-v661 sensitivity cohort is disjoint from the original eight-cell
pilot by stable nucleus identifier, but both cohorts come from the same
MICrONS mouse. It is therefore a same-animal sensitivity analysis, not an
independent-animal replication.
"""


FIGURE_1_README = """# Figure 1

Figure 1 is a conceptual schematic generated programmatically from the theory
and analysis design. It contains no plotted numerical observations.
"""


FIGURE_2_README = """# Figure 2 source data

Panels a-c contain the run- or seed-level values plotted in the final figure.
Panels b and c use the complete clean 15-seed feedback cohort, which passed the
frozen completeness, scientific-configuration, checkpoint, and hash audits.
Panel c evaluates both feedback fields at every checkpoint in that cohort.

Panel d includes one row per independent seed:

- 20 exact-transport rows: 2 rule variants x 2 decoder modes x 5 seeds;
- 5 matched-backpropagation rows: 1 condition x 5 seeds.

`Fig2d_exact_transport_and_backprop_runs.csv` was exported directly from the
archived `performance/final.json` and resolved `config.json` files. It records
the run/config identifier, seed, rule, decoder mode, feedback mode, learning
method, architecture, dataset, and final test accuracy. Regrouping these 25
rows reproduces every reported mean, sample s.d., minimum, maximum, and n in
the two retained convenience summaries.

The `Fig2f_initialization_factorial_*` files support the 15-seed
architecture-by-initialization-policy control reported in the main text. The
run-level table is primary; the summary and paired-comparison tables are
derived from those 60 runs.

"""


FIGURE_3_README = """# Figure 3 source data

The publication-facing run-level table contains 480 validity-qualified runs
from the 640-run depth-by-feedback execution: both cores on MNIST and the
additive core on the signed synthetic task. The outcome-independent rule and
every historical row are included in the validity ledger. Panels a--c show the
retained depth-by-feedback cohort, panel d shows the 120-run retained matched-
bandwidth ownership control, panel e shows the detached 320-run exact/BP audit,
and panel f summarizes the 160-run additive fixed-budget depth control. The
historical inhibitory-dose family is excluded because its prespecified
cross-core endpoint depends on invalid signed-input shunting cells.
"""


FIGURE_4_README = """# Figure 4 source data

These files support the complete 2,700-fit subtree-address factorial. Training
seed is the independent unit. The four implementation-equivalent
representations receive identical routed fields; equality therefore bounds a
uniquely dendritic interpretation. The stochastic-control pairing failure from
the first execution was quarantined, corrected, and followed by a complete
rerun; only the corrected cohort is packaged.
"""


FIGURE_5_README = """# Figure 5 source data

These files support the mechanism-matched physical-depth experiment and its
point, grouped-star, coordinate and optimizer controls. Training seed is the
independent unit. Expanded alignment-dose and second-hierarchy controls are
Supplementary Figure 18.
"""


FIGURE_6_README = """# Figure 6 source data

These files support reconstructed-tree morphology, model-field and reciprocal-
cable route capacity, and capture per unit wiring. Reconstructed cell is the
independent unit; perturbation streams are nested within cell. All fields are
modeled rather than measured task gradients. Full diagnostics are
Supplementary Figure 20.
"""


FIGURE_7_README = """# Figure 7 source data

The panel files support the focal perturbation, dose response, parameter and
direct-type sensitivity, exact factor-freeze contrast, and physical cable
calibration. Focal sites are nested within reconstructed cells; all eight cells
come from one animal.
"""


SUPPLEMENTARY_FIGURE_1_README = """# Supplementary Figure 1 source data

This is the unchanged mechanistic figure from the final NeurIPS/arXiv
regular-tree study. The packaged files are the original plotting tables for
path-gain dispersion, stage-resolved feedback fidelity, inhibition
interventions, and exact-transport learning. The archived generator and PDF
asset are preserved byte for byte in the journal repository.
"""


FIGURE_9_README = """# Figure 9 source data

These files support the measured-visual-response boundary analysis, including
all 13 deterministically eligible scans. Target cells are the independent
units; scans and ten frozen stimulus splits are nested within target. The
analysis tests one response-derived objective and does not infer an endogenous
teaching signal.
"""


SUPPLEMENTARY_FIGURE_10_README = """# Supplementary Figure 10 source data

Panels b-e use the final 20-stream public-v661 routing analysis. The plotting
table averages Monte Carlo streams within each cell and condition; the
cell-by-stream table is included for full support. Panels f-g use cell-averaged
focal contrasts. Panels a and h use the cohort manifest. The cohort and endpoint-exclusion manifests document the
frozen sampling frame.

The older combined replication summary is deliberately not packaged because
its routing subsection predates the final 20-stream analysis. The included
`SuppFig10b-e_routing_capacity_summary.json` is the authoritative routing
summary, and `SuppFig10f-g_focal_cells.csv` supplies the focal panels directly.
"""


FIGURE_8_README = """# Figure 8 source data

Panels a-d support the seven-target measured-response boundary, panels e-f the
controlled alignment test, panels g-h complete-tree learning, panels i-j the
six-animal signed-coordinate analysis, and panel k the alignment-by-bandwidth
synthesis. Scans, splits and Monte Carlo fields are nested within target cell;
animal is the independent unit for panels i-j.
"""


FIGURE_10_README = """# Figure 10 source data

Panels a-d support the controlled alignment-sufficiency experiment on eight
fixed reconstructed morphologies. Monte Carlo streams are nested within cell;
the imposed quadratic gradients are a constructive computational test rather
than evidence that those gradients occurred in vivo. Panels e-h support the
external six-animal signed-credit reanalysis. Animal is the inferential unit;
neuron-level distributions in panel h are descriptive and remain nested within
animal.
"""


SUPPLEMENTARY_FIGURE_2_README = """# Supplementary Figure 2 source data

This is the unchanged matched-capacity and feedback-definition figure from the
final NeurIPS/arXiv regular-tree study. The packaged files are its original
task, dose, morphology, mechanism-control, and neuron-indexed-feedback tables.
"""


SUPPLEMENTARY_FIGURE_3_README = """# Supplementary Figure 3 source data

This is the unchanged rule-and-feedback-control figure from the final
NeurIPS/arXiv regular-tree study. The packaged files are the original
rule-family, error-source, exact-transport, and feedback-ladder tables.
"""


SUPPLEMENTARY_FIGURE_4_README = """# Supplementary Figure 4 source data

These files retain the historical depth, broadcast-noise and shunting-only
flattened-CIFAR-10 boundaries and the fresh 20-seed raw-additive CIFAR-10
feedback ladder. The first three panels contain five independent seeds per
condition. Exact transport is an information oracle. These harder-data panels
are transfer and mechanism controls, not competitive vision benchmarks.
"""


SUPPLEMENTARY_FIGURE_5_README = """# Supplementary Figure 5 source data

These files support the constructed topology--task alignment experiment on
eight fixed reconstructed morphologies. Monte Carlo streams are nested within
cell; the construction is a sufficiency test rather than biological evidence
that the imposed gradients occurred in vivo.
"""


SUPPLEMENTARY_FIGURE_6_README = """# Supplementary Figure 6 source data

These files support the exact static route-overlap and off-route-leakage
calculation. The 101 by 101 grid is deterministic and is not a temporal neural
simulation or an independent biological cohort.
"""


SUPPLEMENTARY_FIGURE_7_README = """# Supplementary Figure 7 source data

These files support the fixed spatial-topology boundary control. Both maps
have the same number of active contacts. The spatial map has greater unique
input coverage and no cross-branch collisions, and its learning effect
persists under backpropagation and on the randomly projected task.
"""


SUPPLEMENTARY_FIGURE_8_README = """# Supplementary Figure 8 source data

These files support the 160 input-valid additive runs from the 320-run fixed-
terminal-branch and input-contact-budget depth execution. Training seed is the inferential unit. The
design fixes 16 terminal branches and 960--968 active contacts per soma, but
does not match compartment or trainable-parameter count.
"""


SUPPLEMENTARY_FIGURE_9_README = """# Supplementary Figure 9 source data

The checkpoint table contains all 2,400 input-valid diagnostic rows from 120
independently trained backpropagation checkpoints, five feedback labels and four relative
step sizes. Only non-somatic dendritic stages enter the geometry. The plotted
operating point is 1e-5.
"""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def copy_source_file(item: SourceFile, source: Path, destination: Path) -> None:
    """Copy one allow-listed source, removing explicitly declared private columns."""
    if item.source in SANITIZED_JSON_PATHS:
        payload = json.loads(source.read_text(encoding="utf-8"))

        def redact(value):
            if isinstance(value, dict):
                return {key: redact(entry) for key, entry in value.items()}
            if isinstance(value, list):
                return [redact(entry) for entry in value]
            if isinstance(value, str) and value.startswith(
                ("/n/holylabs/", "/n/holylfs06/", "/n/home13/")
            ):
                return f"$RUNTIME_ARCHIVE/{Path(value).name}"
            return value

        destination.write_text(
            json.dumps(redact(payload), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return
    drop = SANITIZED_DROP_COLUMNS.get(item.source)
    if not drop:
        shutil.copy2(source, destination)
        if sha256(source) != sha256(destination):
            raise RuntimeError(f"Copy verification failed for {item.source}")
        return
    with source.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = [name for name in (reader.fieldnames or []) if name not in drop]
        rows = [{name: row[name] for name in fieldnames} for row in reader]
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_manifest(stage: Path, rows: list[dict[str, str]]) -> None:
    columns = (
        "figure",
        "panels",
        "file",
        "role",
        "independent_unit",
        "status",
        "original_source",
        "bytes",
        "sha256",
        "notes",
    )
    with (stage / "manifest.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=columns, delimiter="\t", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def write_display_readmes(stage: Path, rows: list[dict[str, str]]) -> None:
    """Write current, manifest-derived scope notes for every populated display."""
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        directory = Path(row["file"]).parts[0]
        grouped.setdefault(directory, []).append(row)
    for directory, items in grouped.items():
        figure = items[0]["figure"]
        panels = ", ".join(dict.fromkeys(item["panels"] for item in items))
        units = "; ".join(dict.fromkeys(item["independent_unit"] for item in items))
        statuses = "; ".join(dict.fromkeys(item["status"] for item in items))
        text = (
            f"# {figure} source data\n\n"
            f"This directory contains the numerical files for panel(s) {panels}. "
            "The package-level `manifest.tsv` records the role, original source, "
            "size and SHA-256 digest of every file.\n\n"
            f"Independent unit(s): {units}.\n\n"
            f"Evidence status: {statuses}.\n"
        )
        (stage / directory / "README.md").write_text(text, encoding="utf-8")


def audit_no_machine_local_paths(stage: Path) -> None:
    """Reject release text that exposes nonportable machine-local paths."""

    forbidden = ("/n/holylabs/", "/n/holylfs06/", "/n/home13/")
    findings: list[str] = []
    for path in stage.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in {
            ".csv",
            ".json",
            ".md",
            ".tsv",
            ".txt",
        }:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        if any(prefix in text for prefix in forbidden):
            findings.append(str(path.relative_to(stage)))
    if findings:
        raise RuntimeError(
            "Machine-local paths remain in Source Data release files:\n"
            + "\n".join(findings)
        )


def make_zip(stage: Path, output: Path) -> None:
    with zipfile.ZipFile(
        output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as archive:
        for path in sorted(stage.rglob("*")):
            if path.is_file():
                archive.write(path, Path("Source_Data") / path.relative_to(stage))


def build(stage: Path, zip_path: Path, *, force: bool = False) -> None:
    digest_path = zip_path.with_suffix(zip_path.suffix + ".sha256")
    missing = [item.source for item in FILES if not (JOURNAL / item.source).is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing allow-listed source files:\n" + "\n".join(missing)
        )
    figure2d_rows = read_figure2d_rows(
        JOURNAL / "source_data" / "figure2" / "exact_transport_and_backprop_runs.csv"
    )
    verify_figure2d_rows(
        figure2d_rows,
        JOURNAL / "source_data" / "figure2" / "exact_transport_factorial_summary.csv",
        JOURNAL / "source_data" / "figure2" / "backprop_summary.csv",
    )
    if force:
        if stage != DEFAULT_STAGE.resolve() or zip_path != DEFAULT_ZIP.resolve():
            raise ValueError("--force is restricted to the default generated outputs")
        if stage.exists():
            if not stage.is_dir():
                raise RuntimeError(f"Expected generated directory at {stage}")
            shutil.rmtree(stage)
        for path in (zip_path, digest_path):
            if path.exists():
                if not path.is_file():
                    raise RuntimeError(f"Expected generated file at {path}")
                path.unlink()
    if stage.exists() or zip_path.exists() or digest_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing output. Remove or move {stage}, {zip_path}, "
            f"and {digest_path}, or rerun the default build with --force."
        )

    SUBMISSION.mkdir(parents=True, exist_ok=True)
    # Stage on the node-local temporary filesystem.  Recursive cleanup of a
    # just-written tree on the shared NFS volume can race metadata propagation
    # and raise a spurious "directory not empty" after a valid package was
    # already produced.
    with tempfile.TemporaryDirectory(prefix="nature_source_data_") as temporary:
        work = Path(temporary) / "Source_Data"
        work.mkdir()
        rows: list[dict[str, str]] = []
        for item in FILES:
            source = JOURNAL / item.source
            destination = work / item.destination
            destination.parent.mkdir(parents=True, exist_ok=True)
            copy_source_file(item, source, destination)
            destination_hash = sha256(destination)
            rows.append(
                {
                    "figure": item.figure,
                    "panels": item.panels,
                    "file": item.destination,
                    "role": item.role,
                    "independent_unit": item.independent_unit,
                    "status": item.status,
                    "original_source": item.source,
                    "bytes": str(destination.stat().st_size),
                    "sha256": destination_hash,
                    "notes": item.notes,
                }
            )

        (work / "README.md").write_text(
            README + f"\nPackage generated: {date.today().isoformat()}.\n",
            encoding="utf-8",
        )
        (work / "Figure_1").mkdir(exist_ok=True)
        (work / "Figure_1" / "README.md").write_text(FIGURE_1_README, encoding="utf-8")
        # Generate every populated display note from the current manifest rows
        # so no inherited prose can describe a superseded panel assignment.
        write_display_readmes(work, rows)
        write_manifest(work, rows)
        audit_no_machine_local_paths(work)
        shutil.copytree(work, stage)
        make_zip(work, zip_path)
        digest_path.write_text(
            f"{sha256(zip_path)}  {zip_path.name}\n", encoding="utf-8"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=Path, default=DEFAULT_STAGE)
    parser.add_argument("--zip", dest="zip_path", type=Path, default=DEFAULT_ZIP)
    parser.add_argument(
        "--force",
        action="store_true",
        help="replace only the default generated Source Data outputs",
    )
    args = parser.parse_args()
    build(args.stage.resolve(), args.zip_path.resolve(), force=args.force)
    print(f"Wrote {args.stage.resolve()}")
    print(f"Wrote {args.zip_path.resolve()}")
    print(f"Packaged {len(FILES)} source files")


if __name__ == "__main__":
    main()
