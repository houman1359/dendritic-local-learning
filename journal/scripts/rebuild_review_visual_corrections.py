#!/usr/bin/env python3
"""Rebuild corrected supplementary artwork from frozen analysis tables.

This entry point redraws figures only; it does not rerun training, change
analysis tables, or invoke the historical all-figure submission compositor.
"""

from __future__ import annotations

import contextlib
import json
import shutil
from pathlib import Path

import pandas as pd
from matplotlib.figure import Figure

import figure_canvas
from journal_style import LW_EDGE, LW_HAIR

import analyze_branch_credit_interference as interference
import analyze_spatial_topology_audit as spatial
import build_microns_v661_replication_figure as replication
import build_trained_partition_residual_figure as partition
from analyze_prospective_followup_results import plot_fixed_budget
from assemble_compact_main_figures import SUPP, compose, panel


ROOT = Path(__file__).resolve().parents[1]


# ── house tokens for the three legacy renders this script republishes ─────
#
# S7 (spatial topology audit), S10 (v661 replication) and S23 (trained
# partition residual) were authored before the 2026-09-08 stroke scale, so
# their scaffolding still carries the pre-canvas widths: 0.8 pt spines and
# tick marks, 0.6 pt grid and zero rules, and 0.25/0.4/0.45/0.5 pt marker
# and patch edges, plus one 1.65 pt (LW_DATA + 0.4) mean bar in S23 C that
# trips the 1.35 pt open-stroke rule.  None of those is a line-weight token,
# so every curated sheet that pastes a crop of them inherits the note.
#
# The fix is presentation-only and is applied at save time, not at the call
# sites: the drawing code lives in shared upstream builders that this script
# only drives, so the weights are snapped on the finished figure.  Role-bearing
# scaffolding is taken to its own token first (spines and major ticks to
# LW_EDGE, minor ticks and grid to LW_HAIR) and everything else is snapped by
# :func:`figure_canvas.enforce_tokens`, which maps each remaining width to the
# nearest of (.55, .7, .85, .95, 1.25) and pulls any line at or above the
# 1.35 pt area-mark threshold back to LW_DATA.  Hairline < grid < data <
# emphasis therefore survives, and no coordinate, limit or condition moves.
# Type is left alone (``fonts=False``): the consolidated sheets cut these
# renders at frozen panel boxes, so text may not change size here.


def _panel_axes(fig) -> list:
    """Every Axes in the figure, inset children included.

    ``ax.inset_axes`` registers its child on the parent, not on the figure, so
    a bare ``fig.axes`` walk misses the difference inset of S23 A and leaves
    its 0.8 pt spines and ticks to the generic snap (which reads them as 0.85,
    the reference weight, instead of the spine weight).
    """
    found, stack = [], list(fig.axes)
    while stack:
        ax = stack.pop()
        if any(ax is seen for seen in found):
            continue
        found.append(ax)
        stack.extend(getattr(ax, "child_axes", []))
    return found


def _conform_strokes(fig) -> None:
    """Snap one finished figure's line weights onto the journal tokens."""
    token = figure_canvas.is_token_lw
    for ax in _panel_axes(fig):
        for spine in ax.spines.values():
            width = float(spine.get_linewidth() or 0.0)
            if width and not token(width):
                spine.set_linewidth(LW_EDGE)
        for axis in (ax.xaxis, ax.yaxis):
            for ticks, weight, which in (
                (axis.get_major_ticks(), LW_EDGE, "major"),
                (axis.get_minor_ticks(), LW_HAIR, "minor"),
            ):
                marks = [float(t.tick1line.get_markeredgewidth() or 0.0)
                         for t in ticks]
                if any(w and not token(w) for w in marks):
                    ax.tick_params(axis=axis.axis_name, which=which,
                                   width=weight)
            grid = [float(line.get_linewidth() or 0.0)
                    for line in axis.get_gridlines()]
            if any(w and not token(w) for w in grid):
                ax.tick_params(axis=axis.axis_name, which="major",
                               grid_linewidth=LW_HAIR)
    figure_canvas.enforce_tokens(fig, fonts=False, strokes=True)


@contextlib.contextmanager
def house_strokes():
    """Conform every figure saved inside the block; restore ``savefig`` after."""
    original = Figure.savefig

    def savefig(self, *args, **kwargs):
        _conform_strokes(self)
        return original(self, *args, **kwargs)

    Figure.savefig = savefig
    try:
        yield
    finally:
        Figure.savefig = original


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
    # The three legacy renders below are republished on the journal stroke
    # scale; the two above keep their own registered bytes untouched.
    with house_strokes():
        spatial.plot(
            pd.read_csv(spatial.OUT / "owner_metrics.csv"),
            pd.read_csv(spatial.OUT / "task_feedback_effects.csv"),
        )
        replication.render(
            pd.read_csv(replication.DATA / "supp_figure_routing_curves.csv"),
            pd.read_csv(replication.DATA / "supp_figure_focal_cells.csv"),
            json.loads((replication.DATA / "replication_summary.json").read_text()),
        )
        # S23 was previously copied from whatever figures/generated held; it is
        # redrawn here from the same frozen tables so the copied bytes carry the
        # token weights too.
        partition.main()
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
