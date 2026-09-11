#!/usr/bin/env python3
"""Rebuild corrected supplementary artwork from frozen analysis tables.

This entry point redraws figures only; it does not rerun training, change
analysis tables, or invoke the historical all-figure submission compositor.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd

import analyze_branch_credit_interference as interference
import analyze_spatial_topology_audit as spatial
import build_microns_v661_replication_figure as replication
from analyze_prospective_followup_results import plot_fixed_budget
from assemble_compact_main_figures import SUPP, compose, panel


ROOT = Path(__file__).resolve().parents[1]


def rebuild_valid_fixed_budget() -> None:
    """Draw S8 from the retained, input-qualified cohort only."""
    source = ROOT / "source_data/prospective_input_validity"
    summary = pd.read_csv(source / "followup_publication_condition_summary.csv")
    fixed = summary[summary.family.eq("fixed_budget")]
    if set(fixed.core) != {"dendritic_additive"}:
        raise ValueError("Expected the qualified fixed-budget additive cohort")
    plot_fixed_budget(
        summary,
        pd.read_csv(source / "followup_publication_paired_contrasts.csv"),
        pd.read_csv(source / "followup_publication_seed_outcomes.csv"),
    )


def main() -> None:
    rebuild_valid_fixed_budget()
    interference.make_figure(
        pd.read_csv(interference.DEFAULT_OUTDIR / "interference_surface.csv"),
        interference.DEFAULT_STEM,
        eta=0.2,
    )
    spatial.plot(
        pd.read_csv(spatial.OUT / "owner_metrics.csv"),
        pd.read_csv(spatial.OUT / "task_feedback_effects.csv"),
    )
    replication.render(
        pd.read_csv(replication.DATA / "supp_figure_routing_curves.csv"),
        pd.read_csv(replication.DATA / "supp_figure_focal_cells.csv"),
        json.loads((replication.DATA / "replication_summary.json").read_text()),
    )
    for source, target in (
        (interference.DEFAULT_STEM.with_suffix(".pdf"),
         "figure_S06_panels_A-D.pdf"),
        (spatial.FIGURES / "fig_spatial_topology_audit.pdf",
         "figure_S07_panels_A-D.pdf"),
        (replication.FIGURES / f"{replication.STEM}.pdf",
         "figure_S10_panels_A-H.pdf"),
        (ROOT / "figures/generated/fig_trained_partition_residual.pdf",
         "figure_S23_panels_A-C.pdf"),
    ):
        shutil.copyfile(source, SUPP / target)
    # S21 is owned exclusively by build_supplementary_figure_s21_native.py.



if __name__ == "__main__":
    main()
