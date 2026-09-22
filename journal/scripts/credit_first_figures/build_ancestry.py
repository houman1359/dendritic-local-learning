#!/usr/bin/env python3
"""Main Fig. 3 (fig:subtreefactorial): ancestry routes, matched controls, cues.

Rebuilt 2026-09-09 on the NativeCanvas per
``analysis/figure_overhaul_20260908/v2/fig3/PLAN.md`` as amended by
``v2/AMENDMENTS.md`` and ruled by ``v2/DECISIONS.md``; re-laid out in the
2026-09-14 design pass (content-proportional panels, no prose in plot
boxes, no titles on data panels).  Three rows, six letters, in citation
order and in row-major order:

    row 0  A  eight-stream task tree (4 mod)   B  ancestry + control
                                                   dictionaries (8 mod)
    row 1  C  accuracy across K (3 mod)  D  rewiring (3 mod, shares C's y)
           E  K = 4 forest, five rows (6 mod)
    row 2  F  soft readout (3 mod) | hard readout (3 mod, shares y) |
              paired accuracy minus oracle forest (6 mod)

The geometry paragraphs below describe the 2026-09-09 build and are kept as
the record of why each earlier choice was made; where they name a height,
a row weight or a gutter, the ``build()`` comment and the
``deviations_from_plan`` entry dated 2026-09-14 state the current values.

Cross-figure rules binding on this build (AMENDMENTS §3), reproduced here
because they are set-wide:

* CF-1  518.4 pt wide; height on the 340/415/490 ladder only (490 here,
  aspect 1.058 >= ASPECT_MIN 1.05).
* CF-2  exactly three type sizes 7.0 / 8.0 / 9.0-bold, nothing below 7.0,
  no DejaVu, subscripts via Frame.subscript / token_subscript, never
  mathtext.
* CF-3  stroke widths only from {0.55, 0.70, 0.85, 0.95, 1.25} pt; every
  area mark a 16 % tint_patch with a 0.55 pt edge; no open stroke at or
  above DECORATIVE_LW_PT 1.35.
* CF-4  glyph register: soma filled and lowest, tapered dend canopy with
  open white junction rings, exc/inh filled contacts at 3.6 pt, gate /
  shunt / inhibition one family, ghosts at GHOST_PCT 45, one ink delta-0
  arrow per soma.  Fig 3 declares NO DELTA0_EXEMPTIONS entry.
* CF-5  zero legend artists; the only sanctioned key in the nine-figure
  set is the frameless four-entry rule key inside Fig 5C.
* CF-6  the forest idiom for category-vs-value panels (here: E), set-wide
  second-arm offset +0.22 rows with an open marker if one is ever added.
* CF-7  every reference line dashed ``mute`` at LW_REF with its label
  right-aligned on the line; zero drawn once, never as grid + rule.
* CF-8  caption rules, word band 220-320 with macros stripped and each
  inline math group counted as one word.
* CF-9  titles in sentence case, no terminal period, a finding phrase,
  <= 26 characters at <= 4 modules and <= 42 above.
* CF-10 the single schematic-area formula of AMENDMENTS §1 B12.
* CF-11 check_matrix_cells >= 6.0 pt per row and per column, column
  headers <= 1.5x the column width, no raster below 300 dpi.
* CF-12 9 pt bold letters, canvas.align_letters() unconditional, no hand
  placement, <= 0.5 pt spread per module column.

Schematic area on the B12 / CF-10 formula
``sum(schematic slot w_pt x h_pt) / (live_w_pt x live_h_pt)`` with
``live = (518.4 - 44 - 12) x (490 - 20 - 32) = 462.4 x 438``:
``(128.8 + 295.6) x 124 = 52 625.6`` of ``202 531.2`` = **26.0 %**, inside
the 30 % cap.  G4 waiver recorded for Fig 3 at the superseded 31.6 %
measure; on the B12 formula the figure is 26.0 % and the waiver is
dormant.  PANEL_ASPECT_MAX is not relaxed.

Recorded deviations from the plan (each with its reason; the full list is
also written into ``figure_03_sources.json``):

* Row weights 124/113/113 with a 44 pt vertical gutter and margins
  44/12/20/32, not the plan's 112/121/121 at 42 pt with margins 52/14.
  Three independent constraints fix this (all re-verified 2026-09-09
  against the built PDF):
  (i) MARGINS.  ``FILL_W_MIN`` is 0.92, so the drawn ink must span at
  least 476.9 pt of the 518.4 pt page.  The leftmost ink is the letter
  column at ``margins.left - LETTER_COL_DX_PT`` = left - 16, the
  rightmost is the right-hand panel edge at 518.4 - right, so the strict
  audit is satisfiable only while ``left + right <= 57.5``.  The plan's
  52 + 14 = 66 gives 90.3 % and fails ``fill-width``; 44 + 12 = 56 gives
  92.3 %.  The plan's stated reason for left = 52 (keeping C's y label
  out of the protected letter column) is met instead by
  ``declare_reserve('C'/'D', left=20)``, and audit_letter_alignment.py
  passes.
  (ii) ROW 0.  Panel B spans eight modules = 295.6 pt at these margins;
  at a 112 pt row its axes box aspect is 2.64, above
  ``PANEL_ASPECT_MAX`` 2.40, which DECISIONS G4 forbids relaxing.  124 pt
  is the shortest row 0 that clears the cap (2.38), so the AMENDMENTS
  Fig 3 item 5 "tighten to 106/124/124" instruction is not taken -- it
  would put B at 2.79.
  (iii) VERTICAL GUTTER.  The panel titles and x labels overflow their
  slots by about 35 pt between rows 1 and 2, so a 42 pt gutter leaves
  7.2 pt against audit_row_separation.py's 8.5 pt floor; 44 pt measures
  9.4 pt.
  The two data rows therefore stand at 113 pt, 6 pt taller than the
  build the reviewer saw, and the schematic fraction falls from 26.8 %
  to 26.0 % (the plan's 23.4 % is computed on the plan's own 452.4 x 438
  live box, which fails (i)).
* E is placed with ``lock=False`` and a private measured left inset
  instead of ``declare_reserve('E', left=66)``.  A declared reserve is
  locked per GRID COLUMN, and C starts in the same column 0, so the
  declaration would also carve 66 pt off C and leave C and D -- both six
  modules of row 1 -- with different axes widths, which the layout
  contract rejects as ``row-alignment``.  The inset is measured from the
  row labels and capped at 62 pt so the ``panel-emphasis`` slot-fill ratio
  stays under EMPHASIS_MAX_RATIO 1.35.
* E's per-row wins / Holm notes are drawn INSIDE the axes, above each
  row's seed fan, not through ``forest(note=...)``.  ``forest`` sets a
  note outside the right spine, which here is the 38 pt gutter shared
  with panel F's y axis.
* A's coefficient tiers are ONE contiguous band, in spatial order, each
  block widened to hold its own 7 pt coefficient and its own 7 pt tier
  name and tied to its terminal group by a hairline connector; the tint
  per cent rises 13/22/34/46 with tree distance so the ORDINAL_RAMP
  reads as light-to-dark in print.  This is a CF-3 exception (CF-3 fixes
  every area mark at a flat 16 % tint) and since 2026-09-10 it is recorded
  in the CANVAS MANIFEST as the ``cf3_tint_exception`` schematic note, not
  only here: at a flat 16 % the four ramp teals collapse to within a few
  per cent of the same paper-white value and the ordinal reading is lost.
  A block cannot be exactly centred on
  a single terminal (a 19.5 pt coefficient against a 12.5 pt terminal
  pitch), so the band is a partition with connectors rather than four
  terminal-aligned patches, and the tier gloss is the spatial order
  (same half / cued / sibling / other half) rather than the plan's
  ordinal string, from which "nonsibling" is dropped for width.
* A drops the panel footer "junctions define feedback supports only" into
  the caption -- the plan's own stated fallback -- because the subtitle,
  the two-row tier key, the eight input labels and the 15 pt delta-0
  reserve leave no 12 pt band.
* B's cards carry no tree, so ``credit_delivery`` (which needs a drawn
  ``Nodes``) cannot draw the subtree entry arrow; a private
  ``_delivery_arrow`` draws the same arrowhead in the same rule colour
  into the addressed column.  Panel B therefore draws no soma and
  ``require_delta0()`` passes with no exemption, as the plan states.
* B's control-support row labels are single-line and the support table is
  4 x 10 pt rows by 8 x 7 pt columns (not 6 x 6 pt): a two-line 7 pt row
  label cannot sit on a 6 pt row.
* D's paired intervals are computed with a fixed 20 000-draw bootstrap
  seeded 70000 over the frozen seed_outcomes.csv; K = 2 comes out
  [21.03, 25.32] against the plan's quoted [21.05, 25.32].  The estimator,
  the data and the mean (+23.149 pp = 44.8633 - 21.7139) all agree; only
  the resampling draw differs, and the caption prints the endpoint this
  builder actually computes.  DELIBERATE, SOURCED: the plan's endpoint is
  not reproducible without its unrecorded RNG state, and printing a
  number the builder did not compute would break the §9 assertion chain.
* C and D size EVERY series from one ink area, MARKER_AREA_PT2 = 20 pt2
  (``marker_ms``): matplotlib sizes a marker by its bounding box, so the
  same ``ms`` is a different quantity of ink per shape (at ms = 10 a
  circle inks 78.5 pt2, a square 100.0, a diamond 100.2, a triangle 50.0)
  and the unsized build made the post-hoc ceiling control the heaviest
  mark and the ancestry route the faintest.  Shipped: square 4.47,
  diamond 4.47, triangle 6.32, circle 5.05 pt.  The nested ancestry
  circle is reduced to TIE_MS 2.8 pt at K = 1 and K = 8 ONLY, where the
  families tie exactly; elsewhere it carries a white rim so the K = 4
  pair (80.11 vs 78.84 %) still reads as two marks.
* C's `K = 1:` tie tag carries a hairline leader to its concentric
  marker; the `K = 8:` tag does not, because that marker pair sits at the
  panel's top right and any leader would be an ~80 pt rule crossing all
  three series.  DELIBERATE, DECLARED (plan section 4 C asks for both).
* C and F carry the plan's long annotations in shortened form; the full
  wording is in the caption.  A 166 pt panel cannot hold a 295 pt line.
  C prints the two tie tags verbatim but at 18.66 %, the value the frozen
  ``condition_summary.csv`` carries for both arms at K = 1.
* VISUAL REVIEW 2026-09-10.  ``Y_LIM_C`` is (-4, 100), not (-20, 100): the
  lowest datum is 18.47 % and the old window left 38 % of C's and D's plot
  height below every mark, filled with three lines of caption prose.  C's
  third line ("below chance: pooled class signal is sign-reversed (B)") is
  deleted -- the drawn chance rule, panel B and the running text all carry
  it -- and the two tie tags are re-hung inside the tightened band; D's
  rewired direct label is now ONE line with the badge inline, which is all
  the band it needs.  D also labels its own y ticks and repeats
  ``Held-out accuracy (%)`` (it repeated the x axis in full but carried no
  readable y scale, 55 pt of blank gutter away from C's data; the gutter is
  now 35.7 pt and audit_panel_gaps still passes), is retitled to the two
  magnitudes it actually draws, and drops the K = 4 span, whose arrowheads
  had 1.7 mm to print in and whose leader crossed the ancestry curve.  F
  drops its n / interval / epoch line and three of its four contrast lines
  (all duplicates of the caption or of the running text), marks each of the
  three sampled cue-noise doses (filled soft, open hard), draws the frozen
  floor in the same dashed LW_REF grammar as the two rules above it, and
  gives up the violet oracle rule for CF-7 mute, so violet means the best
  matched control and nothing else in this figure.  B lists its four
  controls in E's order, with the rng draw order unchanged.
* E runs to xlim (-2, 14) rather than the plan's (-2, 11), so the +13.13
  dense rank-4 seed is drawn instead of clipped.
* F is drawn on the plan's ylim (10, 90) with ticks 20/40/60/80, so the
  frozen-profile floor at 18.67 % stands 11 pt clear of the bottom spine.
  The window costs the panel its text margin: at 170.5 x 113 pt the only
  mark-free bands are y 81-90 (one line), the left wedge above the soft
  curves, and y 10-17 below the floor.  Three consequences.  (a) The
  three dashed hard-readout curves ARE now direct-labelled 256 / 64 / 16
  in their ORDINAL_RAMP hues, with leaders to their noise-0.5 vertices
  (64.03 / 54.22 / 32.41 %), under a `hard readout` tag and the
  `exploratory` badge -- the reviewer's blocker on the unreadable dashed
  family.  (b) The plan's three long annotations are reduced to the
  panel's own claim, the -54.47 pp gap at noise 0.5, on two 7 pt lines in
  the left wedge (2026-09-10; the noise-0 gaps -0.31 / -60.90 pp restated
  the running text verbatim).  (c) The `+7.16 pp vs frozen` contrast, the
  n / interval / epoch statement and the footer's cohort range and
  cue-delay clause do not fit and are carried by the caption.
* B's control-table headers are the digits 1..8 under one ``b_i`` tag and
  its |A| key is plain text: Nimbus Sans has no Unicode subscript block
  and CF-2 bans mathtext.
* Private helpers (not in scripts/native_schematics.py, G5): ``_badge``
  (adds the ``ceiling`` and ``exploratory`` badge kinds with the library's
  badge geometry), ``_delivery_arrow`` (the subtree entry arrowhead
  without a tree), ``_gap_span`` (the paired-difference span tag in D).
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
JOURNAL = HERE.parents[1]
sys.path.insert(0, str(JOURNAL / "scripts"))
sys.path.insert(0, str(HERE))
import routing_figure_panels as routing                            # noqa: E402
import run_trained_subtree_address_full_factorial as experiment    # noqa: E402
from focused_provenance import publish                             # noqa: E402
from figure_canvas import (COLORS, LW_DATA, LW_EDGE, LW_ERR, LW_HAIR,  # noqa: E402
                           LW_REF, MARKER_MS, PT_ANNOT, PT_LABEL, PT_SMALL,
                           Margins, NativeCanvas, forest, token_subscript)
from journal_style import (ERR_CAPSIZE, ORDINAL_RAMP, SEED_ALPHA,   # noqa: E402
                           SEED_MS, label_color, style_direct_color_labels,
                           tint_patch)
from native_schematics import (BADGE_STYLE, CONTACT_DIA_PT, Frame,  # noqa: E402
                               LINE_BAND_PT, _text_w_pt, mix, reference_line)

SOURCE = JOURNAL / "source_data"
DATA = SOURCE / "trained_subtree_address_full_factorial"
REVIEW = SOURCE / "review_evidence_reanalysis"
ENCODER = SOURCE / "review_coefficient_encoder"
HARD = SOURCE / "review_coefficient_hard_readout"
RECORDS = SOURCE / "credit_first_figures"
CONFIG = JOURNAL / "configs/trained_subtree_address/full_factorial_confirmatory.json"
ENCODER_CONFIG = JOURNAL / "configs/review_completion/coefficient_encoder.json"
OUT = JOURNAL / "figures/components/credit_first_figure_03.pdf"
PUBLISHED = JOURNAL / "figures/main/figure_03.pdf"

INK, MUTE, GREY = COLORS["ink"], COLORS["mute"], COLORS["point_mlp"]
GREEN, PURPLE, ROSE = COLORS["shunting"], COLORS["oracle"], COLORS["highlight"]
EXC = COLORS["exc"]
K_TICKS = (1, 2, 4, 8)
CUED = 2                      # zero-based index of the cued stream b3
BOOT_SEED = 70_000            # D's paired seed bootstrap (20,000 draws)

# AMENDMENTS §5 role table, restated for this figure:
#   shunting = ancestry / task-matched tree, oracle = the best matched
#   ceiling, point_mlp = derangement and the neutral controls, highlight =
#   the rewired tree.  bp and additive appear nowhere in this figure.
#   ORDINAL_RAMP is an ordinal position within THIS figure's own lists
#   (A's coefficient tier, F's calibration size), never a shared identity.
TIER_RAMP = (ORDINAL_RAMP[0], ORDINAL_RAMP[1], ORDINAL_RAMP[2], ORDINAL_RAMP[3])
CAL_RAMP = {16: ORDINAL_RAMP[1], 64: ORDINAL_RAMP[2], 256: ORDINAL_RAMP[3]}

# QA 2026-09-10 (minor): B listed its four controls in the exact reverse of
# E's forest rows, so a control's delivered support and its contrast could not
# be read at the same vertical position.  B now displays E's order (dense
# rank-4, random-sparse, depth-interleaved, then the off-scale deranged row).
# The rng draw order is NOT reordered with it: one generator is consumed in
# sequence across the four controls, so _DRAW_ORDER keeps the shipped draw and
# every plotted support value is unchanged.
_DRAW_ORDER = (("deranged", "within_neuron_route_derangement"),
               ("depth-interleaved", "depth_interleaved_bins"),
               ("random-sparse", "random_sparse_matched"),
               ("dense rank-4", "random_rank_k"))
CONTROL_ROWS = tuple(reversed(_DRAW_ORDER))
# 2026-09-14 design pass: five rows, one line each, and the per-row wins /
# Holm P as a second, mute line UNDER the row name in the label column, not
# floating inside the plot above the seed fan.  The deranged control is the
# fifth row, drawn as an off-scale arrow with its value, instead of a two-line
# prose note under the axis.
FOREST_ROWS = (("best matched control", "best_matched_nonanatomical_oracle",
                "oracle", "p_holm_four_budgets"),
               ("dense rank-4", "random_rank_k",
                "point_mlp", "p_holm_four_individual_controls_at_k4"),
               ("random-sparse", "random_sparse_matched",
                "point_mlp", "p_holm_four_individual_controls_at_k4"),
               ("depth-interleaved", "depth_interleaved_bins",
                "point_mlp", "p_holm_four_individual_controls_at_k4"),
               ("deranged", "within_neuron_route_derangement",
                "point_mlp", "p_holm_four_individual_controls_at_k4"))
# F's right facet: paired accuracy minus oracle for the cells the text names
# (readout, calibration size, cue noise SD, row label).
GAP_ROWS = (("soft", 256, 0.0, "soft, 256 cues, noise 0"),
            ("soft", 16, 0.0, "soft, 16 cues, noise 0"),
            ("hard", 256, 0.0, "hard, any size, noise 0"),
            ("soft", 256, 0.5, "soft, 256 cues, noise 0.5"),
            ("hard", 256, 0.5, "hard, 256 cues, noise 0.5"))
# QA 2026-09-10 (major): the window ran to -20 with the lowest datum at
# 18.47 %, so 38 % of C's and D's plot height lay below every mark and held
# three lines of caption prose.  The window now stops at -4, which leaves the
# two tie tags (C) and the rewired direct label (D) one band each.
Y_LIM_C = (-4.0, 100.0)

#: The shipped caption (analysis/figure_overhaul_20260908/v2/fig3/TEXT.md).
#: Counted under CF-8 by :func:`caption_words` and recorded in the manifest.
CAPTION = "\\caption{\\textbf{Ancestry-route benefits depend on task alignment and accurate coefficient selection.}\n\\textbf{A}, Eight streams occupy a balanced feedback tree. Context selects terminal $b_3$, whose block determines logit $z$; at four channels, credit also reaches sibling $b_4$. Junctions define feedback supports, not forward compartments. Tier shading indicates distractor class coefficients.\n\\textbf{B}, Ancestry dictionaries: $K$ profiles, each spanning $8/K$ terminals; green, selected profile; grey, other profiles. Full resolution is $A=I_8$. Footers give raw class-signal sums before route normalization. Control rows show signed delivered credit normalized by each row's maximum absolute value for the same context at four channels; random-sparse and dense routes are illustrative draws.\n\\textbf{C}, Held-out accuracy against channel budget for ancestry, deranged delivery and the per-seed best of four matched controls.\n\\textbf{D}, Task-matched versus degree- and depth-matched rewired trees with ancestry feedback. Error bars describe each condition, not the paired difference; shared and fully resolved routes coincide.\n\\textbf{E}, Paired ancestry-minus-control differences at four channels. Labels give positive-seed counts and Holm-adjusted $P$ values across four budgets for the best-control comparison, or four individual controls. The best control is a hindsight reference using those same seeds. The deranged contrast lies off scale and is printed numerically.\n\\textbf{F}, Cue encoder supervised by supplied route-activation targets, at zero delay: soft readout (solid) and exploratory hard selection (dashed), with calibration size labeled. Reference lines show oracle delivery, chance and a frozen profile. Hard selection matches the oracle for noise-free cues. The paired-difference strip shows accuracy minus oracle; faint dots, seeds; larger symbols and whiskers, means and intervals. Cue noise is its Gaussian standard deviation; calibration size counts training cues.\n\\textbf{C--F}: epoch-80 outcomes, $n=20$ paired seeds per cohort; means and 95\\% seed-bootstrap intervals or bands. Intervals smaller than symbols are not visible. \\textbf{F} uses a separate cohort. Teal ramps in \\textbf{A,F} are ordinal within this figure.\nSource Data: \\texttt{source\\_data/curated\\_publication/figure\\_03\\_plotted.csv}.}"




def caption_words(text=None):
    """Use the same prose count as the manuscript's 350-word legend audit."""
    from audit_nature_communications_format import balanced_arguments, prose_words
    caption = CAPTION if text is None else text
    arguments = balanced_arguments(caption, "caption")
    if len(arguments) != 1:
        raise ValueError("Expected exactly one publication caption")
    return len(prose_words(arguments[0]))


def _signed(v, digits=2):
    return f"{v:+.{digits}f}".replace("-", "−")


def _minus(s):
    return str(s).replace("-", "−")


def _p_text(p):
    """Holm P at two significant digits: decimal, or (mantissa, exponent).

    Nimbus Sans has no U+207B, and CF-2 forbids mathtext, so a small P is
    returned as the pair that :func:`_p_draw` sets with ``token_subscript``
    at a NEGATIVE drop -- a real 7.0 pt raised span, never a shrunken one.
    """
    if p >= 1e-3:
        return f"{p:.4f}", None
    digits = int(np.floor(np.log10(p)))
    return f"{p / 10 ** digits:.1f} × 10", f"−{-digits}"


def _data_dx(ax, w_pt):
    """``w_pt`` points expressed in the axes' own x data units."""
    inv = ax.transData.inverted()
    px = w_pt * ax.figure.dpi / 72.0
    return float(inv.transform((px, 0.0))[0] - inv.transform((0.0, 0.0))[0])


def _text_descent_pt(ax, size):
    """Points between a span's baseline and the bottom of its drawn box.

    ``token_subscript`` offsets the raised span from the base span's BOX,
    not from its baseline, so the descent has to be subtracted before a
    raise in points can be requested.  Measured, never assumed: a hidden
    probe is drawn on its baseline and its own box is read back.
    """
    probe = ax.text(0.0, 0.0, "0", fontsize=size, va="baseline",
                    transform=ax.transAxes, alpha=0.0)
    ax.figure.canvas.draw()
    box = probe.get_window_extent()
    base_y = ax.transAxes.transform((0.0, 0.0))[1]
    probe.remove()
    return float(base_y - box.y0) * 72.0 / ax.figure.dpi


FOREST_X_MAX = 14.0                     # E's axis runs 0-14; the +13.13 seed is drawn
_RIGHT_ALIGN = []                       # (ax, base, group, x), see _realign


def _realign():
    """Right-align every chained P group again, after the layout is locked.

    QA 2026-09-10 (major): ``_p_draw`` measures the drawn group in pixels and
    corrects the base in DATA units.  Panel E is built with ``lock=False`` and
    is re-placed by ``lock_reserves`` afterwards, so the same text then spans a
    different number of data units and the correction no longer lands: the two
    exponent lines finished 1.52 pp right of the two plain ones and ran 3.9 pt
    past the axes.  Redoing the measurement after the lock fixes both.
    """
    for ax, base, group, x in _RIGHT_ALIGN:
        ax.figure.canvas.draw()
        inv = ax.transData.inverted()
        right = max(t.get_window_extent().x1 for t in group)
        target = ax.transData.transform((x, 0.0))[0]
        base.set_x(base.get_position()[0]
                   - (inv.transform((right, 0.0))[0]
                      - inv.transform((target, 0.0))[0]))
    if _RIGHT_ALIGN:
        ax = _RIGHT_ALIGN[0][0]
        ax.figure.canvas.draw()
        edges = [max(t.get_window_extent().x1 for t in g)
                 for _, _, g, _ in _RIGHT_ALIGN]
        assert max(edges) - min(edges) < 1.0, (
            f"P groups are not right-aligned: {max(edges) - min(edges):.2f} px")


def _p_draw(ax, x, y, prefix, p, tail, *, color=MUTE, raise_pt=2.8,
            head="Holm P = "):
    """``<prefix>Holm P = 1.8 x 10^-4 (tail)``, right-anchored at ``x``.

    The exponent is a real 7.0 pt span raised by ``token_subscript`` at a
    negative drop, so CF-2's three-size rule holds and no mathtext is used.
    QA 2026-09-10 (major): the shipped drop lifted the exponent 0.77 pt on a
    7 pt glyph, which prints as ``10-4``, i.e. ten MINUS four -- two
    inferential values misstated.  The raise is now measured from the base
    span's BASELINE (``_text_descent_pt``) and set to ``raise_pt`` = 2.8 pt,
    so the exponent's foot clears the base cap height; the trailing span is
    then re-hung on the base baseline instead of inheriting the same
    (now much larger) negative drop from the raised span's box.
    """
    body, expo = _p_text(p)
    head = f"{prefix}{head}{body}"
    if expo is None:
        plain = ax.text(x, y, head + tail, fontsize=PT_SMALL, color=color,
                        ha="right", va="center")
        _RIGHT_ALIGN.append((ax, plain, [plain], x))
        return plain
    w = (_text_w_pt(ax, head, PT_SMALL) + 0.4
         + _text_w_pt(ax, expo, PT_SMALL) + 0.6
         + _text_w_pt(ax, tail, PT_SMALL))
    descent = _text_descent_pt(ax, PT_SMALL)
    before = list(ax.texts)
    base = token_subscript(ax, x - _data_dx(ax, w), y, head, expo, tail="",
                           size=PT_SMALL, sub_size=PT_SMALL, color=color,
                           ha="left", va="center",
                           drop_pt=-(descent + raise_pt), clip_on=False)
    if tail:
        sub = [t for t in ax.texts if t not in before and t is not base][-1]
        ax.annotate(tail, xy=(1.0, 0.0), xycoords=sub,
                    xytext=(0.6, descent - raise_pt),
                    textcoords="offset points", fontsize=PT_SMALL,
                    color=color, ha="left", va="baseline", zorder=5,
                    annotation_clip=False)
    # the chained spans are annotations on ``base``, so measuring the drawn
    # group and translating the base right-aligns the whole chain exactly --
    # a width estimate is 10-15 pt short once the raised exponent is added.
    group = [t for t in ax.texts if t not in before]
    ax.figure.canvas.draw()
    inv = ax.transData.inverted()
    right = max(t.get_window_extent().x1 for t in group)
    base.set_x(base.get_position()[0]
               - (inv.transform((right, 0.0))[0]
                  - inv.transform((ax.transData.transform((x, y))[0], 0.0))[0]))
    _RIGHT_ALIGN.append((ax, base, group, x))
    return base


# ---------------------------------------------------------------- data ----
def coefficient_prediction():
    """Raw and unit-row-normalised class-signal sums per budget (all 8 contexts)."""
    cfg = json.loads(CONFIG.read_text())["task"]
    n = cfg["contexts"]
    rows = []
    for k in K_TICKS:
        routes = experiment.grouped_routes(np.arange(n), k,
                                           "correct_ancestry_subtrees",
                                           np.random.default_rng(0))
        for context in range(n):
            amplitudes = np.array([
                cfg["selected_signal"] if stream == context else
                -cfg["distractor_signal_by_tree_distance"][
                    experiment.tree_relation(stream, context)]
                for stream in range(n)])
            mask = routes[context] > 0
            raw = float(amplitudes[mask].sum())
            delivered = float(routes[context] @ amplitudes)
            np.testing.assert_allclose(delivered, raw / np.sqrt(mask.sum()),
                                       atol=1e-14)
            rows.append(dict(budget_k=k, context=context,
                             group_size=int(mask.sum()),
                             raw_coefficient_sum=raw,
                             normalized_coefficient_sum=delivered,
                             normalization="unit Euclidean norm per route row"))
    table = pd.DataFrame(rows)
    assert table.groupby("budget_k").raw_coefficient_sum.std().max() < 1e-14
    np.testing.assert_allclose(table.groupby("budget_k").raw_coefficient_sum.mean(),
                               [-3.05, -0.05, 0.85, 1.0], atol=1e-14)
    return table


def task_tiers():
    cfg = json.loads(CONFIG.read_text())["task"]
    d = cfg["distractor_signal_by_tree_distance"]
    return dict(selected=float(cfg["selected_signal"]),
                sibling=-float(d["sibling"]), same_half=-float(d["same_half"]),
                other_half=-float(d["opposite_half"]))


def ancestry_dictionary(K):
    """A (8 x K): the K distinct unit-norm route rows of the ancestry field."""
    routes = experiment.grouped_routes(np.arange(8), K,
                                       "correct_ancestry_subtrees",
                                       np.random.default_rng(0))
    columns = []
    for row in routes:
        if not any(np.allclose(row, c) for c in columns):
            columns.append(row)
    A = np.stack(columns, axis=1)
    assert A.shape == (8, K)
    return A


def control_supports(rng_seed=0):
    """Delivered support of b3's credit under each K = 4 control (one draw)."""
    rng = np.random.default_rng(rng_seed)
    rows = {}
    for _, mode in _DRAW_ORDER:                 # draw order fixed: see above
        if mode == "random_rank_k":
            field = experiment.random_rank_routes(np.arange(8), 4, rng)
        else:
            field = experiment.grouped_routes(np.arange(8), 4, mode, rng)
        rows[mode] = field[CUED]
    order = [m for _, m in CONTROL_ROWS]        # display order = E's rows
    table = pd.DataFrame(np.array([rows[m] for m in order]),
                         columns=[f"b{i + 1}" for i in range(8)])
    table.insert(0, "control", order)
    table.insert(1, "cued_stream", "b3")
    return table


def normalized_control_supports(supports):
    """Signed delivered fields Φ[c, b], scaled by each row's absolute maximum."""
    values = supports[[f"b{i + 1}" for i in range(8)]].to_numpy(float)
    maxima = np.abs(values).max(axis=1, keepdims=True)
    assert np.all(maxima > 0), "Every control must deliver nonzero credit"
    return values / maxima


def rewiring_pairs(outcomes):
    """Paired task-matched minus rewired differences, in percentage points."""
    correct = outcomes[outcomes.feedback_family.eq("correct_ancestry_subtrees")]
    out = {}
    for K in K_TICKS:
        a = correct[correct.architecture.eq("dendritic_tree")
                    & correct.budget_k.eq(K)].set_index("seed").heldout_accuracy
        b = correct[correct.architecture.eq("degree_depth_matched_rewired_tree")
                    & correct.budget_k.eq(K)].set_index("seed").heldout_accuracy
        diff = (a - b).to_numpy(float)
        assert len(diff) == 20
        mean, low, high = routing.bootstrap(diff, BOOT_SEED)
        out[K] = (100 * mean, 100 * low, 100 * high, int((diff > 0).sum()),
                  bool(np.all(diff == 0)))
    assert out[1][4] and out[8][4], "K = 1 and K = 8 must tie exactly"
    np.testing.assert_allclose([out[2][0], out[4][0]], [23.15, 5.02], atol=0.02)
    assert out[2][3] == out[4][3] == 20
    return out


def cue_grid():
    """Panel F: the 3 x 3 calibration x cue-noise grid for both readouts."""
    grid = {}
    for readout, folder in (("soft", ENCODER), ("hard", HARD)):
        summary = pd.read_csv(folder / "condition_summary.csv")
        summary = summary[summary.cue_delay_trials.eq(0)]
        traj = pd.read_csv(folder / "trajectories.csv")
        traj = traj[traj.epoch.eq(80) & traj.cue_delay_trials.eq(0)
                    & traj.method.eq("learned_local_cue")]
        for size in (16, 64, 256):
            means, lows, highs = [], [], []
            for j, noise in enumerate((0.0, 0.5, 1.0)):
                cell = summary[summary.calibration_samples.eq(size)
                               & summary.cue_noise_sd.eq(noise)
                               & summary.method.eq("learned_local_cue")]
                assert len(cell) == 1 and int(cell.n_seeds.iloc[0]) == 20
                values = traj[traj.calibration_samples.eq(size)
                              & traj.cue_noise_sd.eq(noise)] \
                    .sort_values("seed").heldout_accuracy.to_numpy(float)
                assert len(values) == 20
                boot = routing.bootstrap(values, 53_000 + 10 * j + size)
                np.testing.assert_allclose(boot[0],
                                           float(cell.mean_accuracy.iloc[0]),
                                           atol=1e-9)
                means.append(100 * boot[0])
                lows.append(100 * boot[1])
                highs.append(100 * boot[2])
            grid[(readout, size)] = (np.array(means), np.array(lows),
                                     np.array(highs))
    enc = pd.read_csv(ENCODER / "condition_summary.csv")
    enc = enc[enc.cue_delay_trials.eq(0)]
    oracle = enc[enc.method.eq("oracle_context")].mean_accuracy.unique()
    frozen = enc[enc.method.eq("frozen_profile")].mean_accuracy.unique()
    mism = enc[enc.method.eq("mismatched_encoder")].mean_accuracy
    assert len(oracle) == 1 and len(frozen) == 1
    coefficient = enc[enc.method.eq("learned_local_cue")
                      & enc.cue_noise_sd.eq(0.0)].mean_coefficient_accuracy
    assert np.allclose(coefficient.to_numpy(float), 1.0)
    return grid, 100 * float(oracle[0]), 100 * float(frozen[0]), \
        (100 * float(mism.min()), 100 * float(mism.max()))


def gap_seeds():
    """Per-seed learned-cue minus oracle accuracy (pp) at epoch 80, zero delay.

    F's right facet draws these fans under the paired means of
    ``paired_contrasts.csv``; the builder asserts that each fan's mean equals
    the table's ``mean_pp`` before anything is drawn.
    """
    out = {}
    for readout, folder in (("soft", ENCODER), ("hard", HARD)):
        traj = pd.read_csv(folder / "trajectories.csv")
        traj = traj[traj.epoch.eq(80) & traj.cue_delay_trials.eq(0)]
        for size in (16, 64, 256):
            for noise in (0.0, 0.5, 1.0):
                cell = traj[traj.calibration_samples.eq(size)
                            & traj.cue_noise_sd.eq(noise)]
                learned = cell[cell.method.eq("learned_local_cue")] \
                    .set_index("seed").heldout_accuracy
                oracle = cell[cell.method.eq("oracle_context")] \
                    .set_index("seed").heldout_accuracy
                assert learned.index.is_unique and oracle.index.is_unique
                assert set(learned.index) == set(oracle.index)
                diff = (learned - oracle) * 100.0
                assert len(diff) == 20 and np.isfinite(diff).all(), \
                    (readout, size, noise, len(diff))
                assert all(int(seed) == seed for seed in diff.index)
                out[(readout, size, noise)] = diff
    return out


def gap_seed_records(seeds):
    """The five drawn F-forest rows, retaining their actual training seeds.

    The zero-noise hard row displays the 256-cue cohort; the 16- and 64-cue
    versions coincide exactly and are checked by ``panel_gaps``.  It is
    twenty plotted differences, not sixty independent observations.
    """
    records = []
    for readout, size, noise, label in GAP_ROWS:
        fan = seeds[(readout, size, noise)]
        assert fan.index.is_unique and len(fan) == 20
        for seed, difference in fan.items():
            records.append(dict(
                panel="F", record="paired_seed", readout=readout,
                calibration_samples=size, cue_noise_sd=noise,
                cue_delay_trials=0, epoch=80, control="oracle_context",
                seed=int(seed), accuracy_difference_pp=float(difference),
                display_row=label, n_seeds=20))
    assert len(records) == 100
    return records


def encoder_contrast(table, size, noise, control):
    sel = table[table.calibration_samples.eq(size) & table.cue_noise_sd.eq(noise)
                & table.cue_delay_trials.eq(0) & table.control.eq(control)]
    assert len(sel) == 1
    row = sel.iloc[0]
    return float(row.mean_pp), float(row.ci95_low_pp), float(row.ci95_high_pp)


# ------------------------------------------- private glyph helpers (G5) ----
_EXTRA_BADGES = {
    "ceiling": ("oracle", mix("oracle", 8), mix("oracle", 45)),
    "exploratory": ("mute", COLORS["panel_bg"], COLORS["grid"]),
}


def _badge(ax, xy, kind, *, text=None, ha="right", va="top", zorder=7):
    """Frame.badge geometry for every kind, plus 'ceiling' and 'exploratory'."""
    key, face, edge = {**BADGE_STYLE, **_EXTRA_BADGES}[kind]
    colour = COLORS[key]
    try:
        colour = label_color(colour, background=face)
    except ValueError:
        pass
    return ax.text(xy[0], xy[1], kind if text is None else text,
                   fontsize=PT_SMALL, color=colour, ha=ha, va=va, zorder=zorder,
                   bbox=dict(boxstyle=f"round,pad=0.28,"
                                      f"rounding_size={2.0 / PT_SMALL:.3f}",
                             facecolor=face, edgecolor=edge, linewidth=LW_HAIR))


def _delivery_arrow(frame, xy, *, color=GREEN, length_pt=7.0):
    """credit_delivery(mode='subtree') entry arrowhead, without a drawn tree."""
    frame.arrow((xy[0], xy[1] + frame.fy(length_pt)), xy, color=color,
                lw=LW_EDGE, head=4.0, zorder=4.6)


def _gap_span(ax, x, y_lo, y_hi, tag, tag_xy, *, dx=0.09, cap=0.05,
              leader=True):
    """Double-headed span between one paired pair, tagged in clear whitespace.

    ``dx`` < 0 puts the span and its tag on the left of the pair.
    """
    side = 1.0 if dx >= 0 else -1.0
    xs = x + dx
    for y in (y_lo, y_hi):
        ax.plot([x, xs + side * cap], [y, y], color=MUTE, lw=LW_HAIR, zorder=1)
    ax.annotate("", xy=(xs, y_hi), xytext=(xs, y_lo),
                arrowprops=dict(arrowstyle="<->", color=MUTE, lw=LW_HAIR,
                                shrinkA=0.0, shrinkB=0.0, mutation_scale=6.0),
                zorder=1)
    mid = 0.5 * (y_lo + y_hi)
    if leader:
        ax.plot([xs + side * cap, tag_xy[0] - side * 0.03], [mid, tag_xy[1]],
                color=MUTE, lw=LW_HAIR, zorder=1)
    return ax.text(tag_xy[0], tag_xy[1], tag, fontsize=PT_ANNOT, color=INK,
                   ha="left" if side > 0 else "right", va="center")


def _group_root(nodes, K, terminal="T3"):
    for root in nodes.at_depth(K):
        if terminal in nodes.terminals_under(root):
            return root
    raise KeyError(terminal)


# ---- equal-area markers (QA 2026-09-10, plan section 4 C "all markers
# ~4.5 pt") --------------------------------------------------------------
# matplotlib sizes a marker by the side of its bounding box, so the same
# ``ms`` is a different quantity of ink per shape: measured at ms = 10,
# a circle inks 78.5 pt2, a square 100.0, a diamond 100.2 and a triangle
# 50.0.  Every series in C and D is therefore sized from ONE ink area,
# MARKER_AREA_PT2 = 20 pt2 (a 4.47 pt square, a 5.05 pt circle), so the
# ceiling control no longer outweighs the ancestry route it is compared to.
MARKER_AREA_PT2 = 20.0
_MARKER_AREA_FRAC = {"o": 0.7855, "s": 1.0, "D": 1.0, "^": 0.5}
# the concentric-tie idiom: at an exact tie the ancestry circle is drawn
# INSIDE the other family's mark, so it is reduced there and only there
# (a 4.47 pt square and a 4.47 pt diamond both inscribe a 4.47 pt circle).
TIE_MS = 2.8


def marker_ms(marker, area=MARKER_AREA_PT2):
    """Markersize in points that inks ``area`` pt2 for this marker shape."""
    return float(np.sqrt(area / _MARKER_AREA_FRAC[marker]))


def _series(ax, x, part, color, marker, ms, zorder, *, filled=True,
            tie_at=(), edge=None):
    mean = part.mean_heldout_accuracy.to_numpy(float) * 100.0
    low = part.ci95_low_heldout_accuracy.to_numpy(float) * 100.0
    high = part.ci95_high_heldout_accuracy.to_numpy(float) * 100.0
    mec = color if edge is None else edge
    ax.errorbar(x, mean, yerr=[mean - low, high - mean], color=color,
                marker="none" if tie_at else marker,
                mfc=color if filled else "white", mec=mec,
                mew=LW_EDGE, ms=ms, lw=LW_DATA, elinewidth=LW_ERR,
                capsize=ERR_CAPSIZE, capthick=LW_ERR, zorder=zorder)
    if tie_at:
        full = [i for i in range(len(mean)) if i not in tie_at]
        # the reduced inner mark keeps its OWN edge colour: a white rim at
        # 2.8 pt is most of the glyph and prints as an open marker inside
        # the enclosing square/diamond (QA 2026-09-10)
        for idx, size, rim, wid in ((full, ms, mec, LW_EDGE),
                                    (list(tie_at), TIE_MS, color, LW_HAIR)):
            if not idx:
                continue
            ax.plot(np.asarray(x)[idx], mean[idx], linestyle="none",
                    marker=marker, ms=size,
                    mfc=color if filled else "white", mec=rim, mew=wid,
                    zorder=zorder)
    return mean


def _accuracy_axes(ax, *, ylabel):
    ax.set_xlim(-0.20, 3.20)
    ax.set_ylim(*Y_LIM_C)
    ax.set_xticks(range(4), [str(k) for k in K_TICKS])
    ax.set_yticks([20, 40, 60, 80])
    ax.spines["left"].set_bounds(20, 80)
    ax.spines["bottom"].set_bounds(0, 3)
    ax.set_xlabel("Channels, K")
    # a panel sharing C's y axis keeps its tick labels (every mark stays
    # readable as a number) and drops only the repeated y title
    if ylabel:
        ax.set_ylabel("Held-out accuracy (%)")


# ----------------------------------------------- A: eight-stream task ----
def panel_task(ax, tiers):
    f = Frame(ax)
    W, H = f.w_pt, f.h_pt
    sub_pt, key_pt, delta_pt = LINE_BAND_PT, 22.0, 12.0
    # sub-title: the cued stream, and the rule under test on it
    f.subscript((f.fx(1.0), 1.0 - f.fy(sub_pt * 0.5)), "context c selects b", "3",
                size=PT_SMALL, color=INK, ha="left")
    f.text((1.0 - f.fx(1.0), 1.0 - f.fy(sub_pt * 0.5)), "K = 4 route",
           size=PT_SMALL, color=GREEN, ha="right")
    rect = (0.0, f.fy(delta_pt), 1.0, 1.0 - f.fy(sub_pt + key_pt + delta_pt))
    nodes = f.balanced_tree(rect, depth=3, mode="forward", trunk=True,
                            labels=True, output="z", soma_r_pt=3.0,
                            input_labels=[("b", str(i + 1)) for i in range(8)])
    # coefficient tiers (QA 2026-09-09, revised 2026-09-09 after review):
    # NOT drawn as tints on the tree, where they merged with each other and
    # with the K = 4 delivery capsule.  One CONTIGUOUS band above the canopy
    # partitions the eight streams in spatial order, every block wide enough
    # for its own 7 pt coefficient and its own 7 pt tier name, each block
    # tied to its terminal group by a hairline connector.  The tint per cent
    # rises with tree distance so the ordinal ramp is visible in print.
    f.note("address_tint",
           reason="coefficient tier by tree distance, not an address")
    # CF-3 exception, recorded in the canvas manifest and not only in the
    # docstring (QA 2026-09-10, minor).  CF-3 fixes every area mark at a
    # 16 % tint; the four tier blocks keep the ORDINAL_RAMP hues AND a
    # graded 13/22/34/46 % tint, because at a flat 16 % the four ramp teals
    # (#6FE0D8 -> #0E6C70) collapse to within a few per cent of the same
    # paper-white value and the ordinal reading -- the whole point of the
    # band -- is lost in CMYK print.  Every block still carries the 0.55 pt
    # tint edge and no other area mark in this figure varies from 16 %.
    f.note("cf3_tint_exception",
           reason="coefficient tier band: ORDINAL_RAMP hues at a graded "
                  "13/22/34/46 % tint instead of the CF-3 flat 16 %, so the "
                  "four ordinal tiers stay separable in print; 0.55 pt tint "
                  "edge kept, and this is the only CF-3 tint variance in "
                  "figure 3")
    order = [(["T1", "T2"], tiers["same_half"], TIER_RAMP[2], 34, "same half"),
             (["T3"], tiers["selected"], TIER_RAMP[0], 13, "cued"),
             (["T4"], tiers["sibling"], TIER_RAMP[1], 22, "sibling"),
             (["T5", "T6", "T7", "T8"], tiers["other_half"], TIER_RAMP[3], 46,
              "other half")]
    pitch = nodes.pitch_pt
    spans, need = [], []
    for leaves, coef, _colour, _pct, gloss in order:
        lo = min(nodes[t][0] for t in leaves) * W - pitch / 2.0
        hi = max(nodes[t][0] for t in leaves) * W + pitch / 2.0
        spans.append((lo, hi))
        need.append(max(_text_w_pt(ax, _signed(coef), PT_SMALL),
                        _text_w_pt(ax, gloss, PT_SMALL)) + 4.4)
    nat = [hi - lo for lo, hi in spans]
    widths = [max(n, d) for n, d in zip(nat, need)]
    avail = W - 1.6
    if sum(widths) > avail:                      # shrink only where there is
        slack = [w - d for w, d in zip(widths, need)]          # real slack
        over, tot = sum(widths) - avail, sum(slack) or 1.0
        widths = [w - over * s / tot for w, s in zip(widths, slack)]
    extra = avail - sum(widths)
    if extra > 0:
        widths = [w + extra * n / sum(nat) for w, n in zip(widths, nat)]
    band_h, gloss_h, tick_h = 8.6, 8.6, 3.4
    band_top = 1.0 - f.fy(sub_pt + 1.0)
    cursor = 0.8
    for (leaves, coef, colour, pct, gloss), w, (lo, hi) in zip(order, widths,
                                                               spans):
        cx = cursor + w / 2.0
        tint_patch(ax, ("rect", f.fx(cursor + 1.2), band_top - f.fy(band_h),
                        f.fx(w - 2.4), f.fy(band_h)), color=colour, pct=pct,
                   radius_pt=1.0, zorder=1.0, clip_on=False)
        f.text((f.fx(cx), band_top - f.fy(band_h / 2.0)), _signed(coef),
               size=PT_SMALL, color=INK)
        f.text((f.fx(cx), band_top - f.fy(band_h + gloss_h / 2.0 + 0.6)),
               gloss, size=PT_SMALL, color=MUTE)
        f.leader((f.fx(cx), band_top - f.fy(band_h + gloss_h + 1.2)),
                 (f.fx((lo + hi) / 2.0),
                  band_top - f.fy(band_h + gloss_h + 1.2 + tick_h)),
                 color=COLORS["grid"])
        cursor += w
    # the forward route of the cued stream is the one ink-weight path
    for a, b in zip(nodes.route("T3"), nodes.route("T3")[1:]):
        if (a, b) in nodes.edges:
            nodes.edges[(a, b)].set_linewidth(f.lw(LW_DATA))
            nodes.edges[(a, b)].set_color(INK)
    # context selection: the inhibitory-family double ring with a 'c' badge.
    # Dropped 4.6 pt down its own branch so the ring rim and the b3 terminal
    # label clear each other at print scale (QA 2026-09-09).
    f.gate((nodes["T3"][0], nodes["T3"][1] - f.fy(4.6)), closed=False,
           badge="c", nodes=nodes, node="T3", badge_offset=(-5.6, -3.0))
    # the one delivery glyph in this panel: K = 4 ancestry capsule + entry arrow
    f.credit_delivery(nodes, mode="subtree", targets=[_group_root(nodes, 4)],
                      rule_color="shunting")
    f.error_in(nodes.soma, label="δ0", side="right")
    f.require_soma_lowest()
    f.require_delta0()
    return nodes


# --------------------------------- B: ancestry and control dictionaries ----
def panel_dictionaries(ax, prediction, supports):
    f = Frame(ax)
    W, H = f.w_pt, f.h_pt
    raw = prediction.groupby("budget_k").raw_coefficient_sum.mean()
    split_pt = 168.0
    ax.plot([f.fx(split_pt), f.fx(split_pt)], [f.fy(6.0), 1.0 - f.fy(4.0)],
            color=COLORS["grid"], lw=LW_HAIR, zorder=0.5)

    # ---- left block: the ancestry ladder ---------------------------------
    left_w = split_pt - 8.0
    f.text((f.fx(left_w / 2.0), 1.0 - f.fy(LINE_BAND_PT * 0.5)),
           "ancestry dictionaries A (8 × K)", size=PT_SMALL, color=INK)
    f.text((f.fx(left_w / 2.0), f.fy(LINE_BAND_PT * 0.5)),
           "group sum of the class signal", size=PT_SMALL, color=MUTE)
    gap_pt = 6.0
    card_w = (left_w - 3 * gap_pt) / 4.0
    card_y0 = f.fy(LINE_BAND_PT + 2.0)
    card_h = 1.0 - f.fy(LINE_BAND_PT * 2 + 6.0)
    cells = []
    for i, K in enumerate(K_TICKS):
        x0 = f.fx(i * (card_w + gap_pt))
        core = f.task_card((x0, card_y0, f.fx(card_w), card_h),
                           title=f"K = {K}", footer=_signed(raw[K]),
                           emphasis=(K == 4))
        cells.append((K, core))
    mat_pt = 48.0
    for K, core in cells:
        cx0, cy0, cw, ch = core
        my = cy0 + f.fy(4.0)
        if K < 8:
            A = ancestry_dictionary(K)
            column = int(np.argmax(A[CUED] > 0))
            mw = f.fx(6.0 * K)
            mx = cx0 + (cw - mw) / 2.0
            col_colors = [GREEN if j == column else COLORS["dend"]
                          for j in range(K)]
            f.dictionary_matrix((mx, my, mw, f.fy(mat_pt)), A, label=None,
                                col_colors=col_colors,
                                row_groups=[8 // K] * K, min_cell_pt=6.0)
            for r in range(1, 8):            # the eight rows are countable
                f.rule(my + f.fy(6.0 * r), mx, mx + mw, color=COLORS["grid"],
                       lw=LW_HAIR)
            _delivery_arrow(f, (mx + f.fx(6.0 * (column + 0.5)),
                                my + f.fy(mat_pt + 1.5)))
        else:
            mx = cx0 + cw * 0.66
            for r in range(8):
                yy = my + f.fy(mat_pt - 3.0 - 6.0 * r)
                f.disc((mx, yy), 1.55, fill=GREEN if r == CUED
                       else COLORS["dend"], zorder=3.4)
            _delivery_arrow(f, (mx, my + f.fy(mat_pt + 1.5)))
            f.text((cx0 + cw * 0.26, my + f.fy(mat_pt * 0.5)), "A = I",
                   size=PT_ANNOT, color=INK)
        # the cued row, named once on the first card and ticked on all four
        ytick = my + f.fy(mat_pt - 3.0 - 6.0 * CUED)
        left = mx if K == 8 else (cx0 + (cw - f.fx(6.0 * K)) / 2.0)
        f.leader((left - f.fx(4.6), ytick), (left - f.fx(1.4), ytick),
                 color=EXC)
        if K == 1:
            f.subscript((left - f.fx(5.4), ytick), "b", "3", size=PT_SMALL,
                        color=EXC, ha="right")

    # ---- right block: the four matched controls at K = 4 -----------------
    # The reference drawing at the top of this block carries the panel's one
    # soma and its one delta-0 arrow, so B declares no DELTA0_EXEMPTIONS entry.
    rx0 = split_pt + 6.0
    right_w = W - rx0 - 1.0
    tree_w = 46.0
    tree_rect = (f.fx(rx0), 1.0 - f.fy(36.0), f.fx(tree_w), f.fy(36.0))
    ref = f.balanced_tree(tree_rect, depth=3, mode="forward", labels=False,
                          output=None, soma_r_pt=2.6)
    f.credit_delivery(ref, mode="subtree", targets=[_group_root(ref, 4)],
                      rule_color="shunting")
    f.contact(ref["T3"], kind="exc", dia_pt=3.0)
    f.error_in(ref.soma, label="δ0", side="right", r_pt=2.6)
    for i, line in enumerate(("correct ancestry", "at K = 4 delivers",
                              "the cued pair")):
        f.text((f.fx(rx0 + tree_w + 5.0), 1.0 - f.fy(8.0 + 9.5 * i)), line,
               size=PT_SMALL, color=INK, ha="left")
    lab_pt, col_pt, row_pt = 56.0, 7.0, 10.0
    mw, mh = f.fx(col_pt * 8), f.fy(row_pt * 4)
    mx = f.fx(rx0 + lab_pt)
    my = f.fy(26.0)
    S = normalized_control_supports(supports)
    inner = f.dictionary_matrix((mx, my, mw, mh), S, color="point_mlp",
                                label=None, row_groups=[1] * 4,
                                col_labels=[str(i + 1) for i in range(8)],
                                yticks=[name for name, _ in CONTROL_ROWS],
                                min_cell_pt=6.0)
    inner.tick_params(axis="y", labelsize=PT_SMALL, pad=2.0)
    f.subscript((mx - f.fx(2.0), my + mh + f.fy(2.4)), "b", "i",
                size=PT_SMALL, color=MUTE, ha="right")
    # Signed key: the dense-random row contains three negative entries.
    bar_y, bar_w = f.fy(17.0), f.fx(9.0)
    f.text((mx - f.fx(1.5), bar_y + f.fy(2.7)), "Φ / max |Φ|", size=PT_SMALL,
           color=MUTE, ha="right")
    for j, level in enumerate((-1.0, 0.0, 1.0)):
        bx = mx + f.fx(3.0) + j * (bar_w + f.fx(15.0))
        ax.add_patch(mpl.patches.Rectangle(
            (bx, bar_y), bar_w, f.fy(5.4),
            facecolor=inner.images[0].get_cmap()((level + 1) / 2),
            edgecolor=COLORS['edge'], linewidth=LW_HAIR, zorder=3, clip_on=False))
        f.text((bx + bar_w + f.fx(2.0), bar_y + f.fy(2.7)), f"{level:g}",
               size=PT_SMALL, color=MUTE, ha="left")
    for i, line in enumerate(("random-sparse and dense rows are",
                              "one illustrative draw (rng seed 0)")):
        f.text((f.fx(rx0), f.fy(8.0 - 8.0 * i)), line, size=PT_SMALL,
               color=MUTE, ha="left")
    f.require_soma_lowest()
    f.require_delta0()
    assert right_w >= lab_pt + col_pt * 8, right_w
    return raw


# --------------------------------------- C: held-out accuracy across K ----
def panel_bandwidth(ax, summary, outcomes):
    x = np.arange(4, dtype=float)
    dend = summary[summary.architecture.eq("dendritic_tree")]
    best = routing._best_control_by_budget(outcomes) * 100.0
    part = (lambda fam: dend[dend.feedback_family.eq(fam)]
            .set_index("budget_k").loc[list(K_TICKS)])
    deranged = _series(ax, x, part("within_neuron_route_derangement"), GREY,
                       "s", marker_ms("s"), 3)
    ax.errorbar(x, best[:, 1], yerr=[best[:, 1] - best[:, 2],
                                     best[:, 3] - best[:, 1]],
                color=PURPLE, marker="D", mfc=PURPLE, mec=PURPLE, mew=LW_EDGE,
                ms=marker_ms("D"), lw=LW_DATA, elinewidth=LW_ERR,
                capsize=ERR_CAPSIZE, capthick=LW_ERR, zorder=4)
    # equal ink area with the other two series (QA 2026-09-10), a white rim
    # so the K = 4 pair (80.11 vs 78.84 %) still reads as two marks, and the
    # concentric idiom kept at the two exact ties only
    ancestry = _series(ax, x, part("correct_ancestry_subtrees"), GREEN, "o",
                       marker_ms("o"), 6, tie_at=(0, 3), edge="white")
    np.testing.assert_allclose(ancestry, [18.67, 44.86, 80.11, 81.00], atol=2e-2)
    np.testing.assert_allclose(best[:, 1], [54.50, 60.26, 78.84, 81.00],
                               atol=2e-2)
    np.testing.assert_allclose(deranged, [18.66, 18.47, 19.05, 25.13], atol=2e-2)
    _accuracy_axes(ax, ylabel=True)
    reference_line(ax, 50.0, label="chance", span=(-0.20, 3.20))
    # Direct labels only (no legend artists anywhere in this figure).  Design
    # pass 2026-09-14: the panel is three modules wide, so each label sits in
    # its own mark-free band -- the ceiling control's name in the top band,
    # the ancestry label under the rising green segment, the deranged label
    # under its flat series.  The n / interval / epoch line and the two tie
    # tags are gone: the caption carries the first, the concentric marks the
    # second, and the tie values (18.66 / 81.00 %) are quoted nowhere in the
    # running text.
    ax.text(-0.16, 93.0, "best matched control", color=PURPLE,
            fontsize=PT_SMALL, ha="left", va="center")
    ax.text(1.10, 33.0, "ancestry", color=GREEN, fontsize=PT_SMALL, ha="left",
            va="center")
    # the K = 4 pair (80.1 vs 78.8) is not a tie: name the gap (QA 2026-09-09)
    ax.annotate("+1.27 pp", xy=(2.06, 79.6), xytext=(3.15, 64.0),
                fontsize=PT_SMALL, color=INK, ha="right", va="center",
                arrowprops=dict(arrowstyle="-", color=MUTE, lw=LW_HAIR,
                                shrinkA=2.0, shrinkB=1.5), zorder=6)
    ax.text(1.50, 8.0, "deranged", color=GREY, fontsize=PT_SMALL, ha="left",
            va="center")
    return dict(ancestry=ancestry.tolist(), best=best[:, 1].tolist(),
                deranged=deranged.tolist())


# ------------------------------------------ D: the rewiring alignment ----
def panel_rewiring(ax, summary, paired):
    x = np.arange(4, dtype=float)
    fam = summary[summary.feedback_family.eq("correct_ancestry_subtrees")]
    part = (lambda arch: fam[fam.architecture.eq(arch)]
            .set_index("budget_k").loc[list(K_TICKS)])
    rewired = _series(ax, x, part("degree_depth_matched_rewired_tree"), ROSE,
                      "^", marker_ms("^"), 3)
    matched = _series(ax, x, part("dendritic_tree"), GREEN, "o",
                      marker_ms("o"), 6, tie_at=(0, 3), edge="white")
    np.testing.assert_allclose(rewired, [18.67, 21.71, 75.09, 81.00], atol=2e-2)
    # D shares C's y axis one gutter away: it keeps its tick labels so every
    # mark reads as a number and drops the repeated y title (2026-09-14).
    _accuracy_axes(ax, ylabel=False)
    reference_line(ax, 50.0, label="chance", span=(-0.20, 3.20))
    # The K = 2 gap is bracketed between its two marks and tagged just under
    # the bracket's foot -- the one mark-free pocket beside the K = 2 marks in
    # a three-module panel; a leader from the bracket's middle would cross
    # the rewired series, which climbs through the same pocket.  The K = 4
    # pair (5.02 pp, 1.7 mm apart) is left to the caption (QA 2026-09-10).
    _gap_span(ax, 1.0, rewired[1], matched[1],
              f"{_signed(paired[2][0])} pp", (1.22, 15.5), leader=False)
    ax.text(-0.16, 93.0, "task-matched tree", color=GREEN, fontsize=PT_SMALL,
            ha="left", va="center")
    ax.text(-0.16, 5.0, "rewired tree", color=ROSE, fontsize=PT_SMALL,
            ha="left", va="center")
    return {K: list(v[:3]) for K, v in paired.items()}


# --------------------------------------------------- E: the K = 4 forest ----
def panel_forest(canvas, ax, contrasts, pairs):
    rows, stats, notes = [], {}, []
    for label, key, colour, holm in FOREST_ROWS:
        row = contrasts[contrasts.control.eq(key) & contrasts.budget_k.eq(4)]
        assert len(row) == 1
        row = row.iloc[0]
        seeds = pairs[pairs.control.eq(key)].accuracy_difference_pp \
            .to_numpy(float)
        assert len(seeds) == 20 and int(row.n_pairs) == 20
        # the deranged row's twenty seeds (57.2-66.1 pp) are all off scale:
        # that row is drawn as an arrow with its value instead of a fan
        drawn_seeds = [v for v in seeds if v <= FOREST_X_MAX]
        rows.append(dict(label="", mean=float(row.mean_difference_pp),
                         lo=float(row.ci95_low_pp), hi=float(row.ci95_high_pp),
                         seeds=drawn_seeds, color=colour, n=20))
        notes.append((label, int(row.positive_seeds), float(row[holm])))
        stats[key] = [float(row.mean_difference_pp), float(row.ci95_low_pp),
                      float(row.ci95_high_pp), int(row.positive_seeds), 20,
                      float(row[holm])]
    assert stats["random_rank_k"][3] == 15, "dense rank-4 is 15/20, not 20/20"
    assert stats["best_matched_nonanatomical_oracle"][3] == 15
    assert stats["random_sparse_matched"][3] == 20
    assert stats["depth_interleaved_bins"][3] == 20
    assert stats["within_neuron_route_derangement"][3] == 20
    keys = ("best_matched_nonanatomical_oracle", "random_rank_k",
            "random_sparse_matched", "depth_interleaved_bins",
            "within_neuron_route_derangement")
    np.testing.assert_allclose([stats[k][0] for k in keys],
                               [1.27, 2.31, 3.77, 7.11, 61.06], atol=6e-3)
    np.testing.assert_allclose([stats[k][1] for k in keys],
                               [0.59, 1.00, 2.89, 6.19, 60.13], atol=6e-3)
    np.testing.assert_allclose([stats[k][2] for k in keys],
                               [1.95, 3.91, 4.74, 8.03, 62.02], atol=6e-3)
    np.testing.assert_allclose([stats[k][5] for k in keys],
                               [0.010140, 0.003380, 1.766e-4, 7.629e-6,
                                7.629e-6], rtol=2e-3)
    assert len(rows[-1]["seeds"]) == 0 and len(rows[1]["seeds"]) == 20
    # CF-6 allows a 0.55 pt row tick OR a 6 % band; the tick is used because a
    # band patch leaves no clear strip for the label column.
    out = forest(ax, rows, value_label="Ancestry − control (pp)",
                 reference=0.0, reference_label="no difference",
                 xlim=(-2.0, 15.6), tag="", seed_alpha=0.45, band=False,
                 tick=True)
    # The label column (design pass 2026-09-14): the row name in ink over its
    # positive-seed count and Holm P in mute, right-aligned 4 pt left of the
    # axes.  Both lines share one data-x anchor, so the lock pass moves them
    # with the axes, and _realign re-measures the P chains afterwards.
    x_right = -2.0 - _data_dx(ax, 4.0)
    widest = 0.0
    for y, (label, wins, p) in zip(out["ypos"], notes):
        ax.text(x_right, y - 0.21, label, fontsize=PT_SMALL, color=INK,
                ha="right", va="center")
        _p_draw(ax, x_right, y + 0.25, f"{wins}/20, ", p, "", head="P = ")
        body, expo = _p_text(p)
        widest = max(widest, _text_w_pt(ax, label, PT_SMALL),
                     _text_w_pt(ax, f"{wins}/20, P = {body}{expo or ''}",
                                PT_SMALL) + (1.6 if expo else 0.0))
    gutter = widest + 8.0
    canvas.declare_reserve("E", left=gutter)
    # the off-scale deranged row: an arrow at the axis end and the value
    y_der = out["ypos"][-1]
    der = rows[-1]
    ax.plot([15.15], [y_der], linestyle="none", marker=">",
            ms=marker_ms("^"), mfc=GREY, mec="white", mew=LW_HAIR, zorder=4,
            clip_on=False)
    ax.text(14.55, y_der, f"{_signed(der['mean'])} [{der['lo']:.2f}, "
            f"{der['hi']:.2f}]", fontsize=PT_SMALL, color=INK, ha="right",
            va="center")
    # QA 2026-09-10: the badge sits just right of the top row's own fan, and
    # E's caption states what the ceiling row is.
    _badge(ax, (max(rows[0]["seeds"]) + 0.45, 0.0), "ceiling", text="hindsight", ha="left",
           va="center")
    # CF-7: zero drawn once, and only over the rows it refers to.  The rule
    # is an axvline, so its y data are AXES FRACTIONS; clip it to the row
    # band -- half a row above the top row and half a row below the last.
    lo_row, hi_row = min(out["ypos"]) - 0.45, max(out["ypos"]) + 0.50
    y0, y1 = ax.get_ylim()                # inverted: y0 is the bottom
    frac = lambda v: (v - y0) / (y1 - y0)
    for line in ax.lines:
        xd = list(line.get_xdata())
        if len(xd) == 2 and xd[0] == xd[1] == 0.0:
            line.set_ydata([frac(hi_row), frac(lo_row)])
    ax.set_xticks([0, 5, 10, 15])
    ax.spines["bottom"].set_bounds(0, 15)
    return stats, gutter


# ------------------------------------------------- F: the cue cohort ----
def panel_cues(ax_soft, ax_hard, grid, oracle, frozen, mismatched, enc_c,
               hard_c):
    """F, left and middle facets: the soft and the exploratory hard readout.

    Design pass 2026-09-14: the two readouts were overlaid in one 170 pt box
    (six curves, two line styles, two mark styles, eleven labels).  They are
    now two shared-y facets of three curves each, the reference rules drawn
    in both and labelled once, the calibration sizes labelled in the left
    strip of the soft facet and at the fanned noise-1 endpoints of the hard
    facet.  The mismatched-encoder range, quoted nowhere in the running
    text, is asserted but no longer printed.
    """
    xs = np.array([0.0, 0.5, 1.0])
    for ax, readout in ((ax_soft, "soft"), (ax_hard, "hard")):
        soft = readout == "soft"
        ax.set_xlim(-0.36, 1.36)
        ax.set_ylim(2.0, 100.0)
        for size in (16, 64, 256):
            mean, lo, hi = grid[(readout, size)]
            colour = CAL_RAMP[size]
            ax.fill_between(xs, lo, hi, color=colour, alpha=0.22, lw=0.0,
                            zorder=1.6)
            # every measured cell carries a mark: filled for the soft
            # readout, open for the exploratory hard one (QA 2026-09-10)
            line, = ax.plot(xs, mean, color=colour,
                            lw=LW_DATA if soft else LW_ERR, marker="o",
                            ms=3.1, mfc=colour if soft else "none",
                            mec=colour, mew=LW_EDGE, solid_capstyle="round",
                            zorder=3.4)
            if not soft:
                line.set_dashes((2.4, 1.8))
        # CF-7 rules: dashed mute at LW_REF, each labelled once, in the soft
        # facet.  The oracle rule is labelled by name only: with its value
        # the label ran 48 pt, shared a line with the 256-cue tag 12 pt to
        # its left, and under the rule it met the steep 256-cue segment.
        reference_line(ax, oracle, label="oracle" if soft else None,
                       span=(-0.10, 1.0))
        reference_line(ax, 50.0, label="chance" if soft else None,
                       span=(-0.10, 1.0))
        floor, = ax.plot([-0.10, 1.0], [frozen, frozen], color=MUTE,
                         lw=LW_REF, zorder=1.2, solid_capstyle="butt")
        floor.set_dashes((2.2, 1.8))
        ax.text(-0.33, 96.0, "soft readout" if soft else "hard readout",
                fontsize=PT_SMALL, color=INK if soft else MUTE, ha="left",
                va="center")
        ax.set_xticks([0.0, 0.5, 1.0], ["0", "0.5", "1"])
        ax.set_yticks([20, 40, 60, 80])
        ax.spines["left"].set_bounds(20, 80)
        ax.spines["bottom"].set_bounds(0.0, 1.0)
        ax.set_xlabel("Cue noise SD")
    ax_soft.set_ylabel("Held-out accuracy (%)")
    _badge(ax_hard, (-0.33, 87.5), "exploratory", ha="left", va="center")
    # the floor label hangs 4.5 pt under its rule: the soft 16-cue series
    # runs 0.4-0.7 pp above the floor across the whole facet
    ax_soft.annotate(f"frozen {frozen:.1f} %", xy=(1.0, frozen),
                     xytext=(0.0, -4.5), textcoords="offset points",
                     fontsize=PT_SMALL, color=MUTE, ha="right", va="top",
                     annotation_clip=False)
    # the eighteen printed grid means, the reference rules and the contrasts
    # are all asserted (plan §9 item 9)
    for key, want in ((("soft", 16), [19.39, 19.16, 19.07]),
                      (("soft", 64), [75.06, 21.39, 19.30]),
                      (("soft", 256), [79.98, 25.83, 19.27]),
                      (("hard", 16), [80.29, 32.41, 20.81]),
                      (("hard", 64), [80.29, 54.22, 22.30]),
                      (("hard", 256), [80.29, 64.03, 23.44])):
        np.testing.assert_allclose(grid[key][0], want, atol=2e-2)
    np.testing.assert_allclose([oracle, frozen], [80.29, 18.67], atol=2e-2)
    np.testing.assert_allclose(mismatched, [18.58, 18.89], atol=2e-2)
    v256 = encoder_contrast(enc_c, 256, 0.0, "oracle_context")
    v16 = encoder_contrast(enc_c, 16, 0.0, "oracle_context")
    prim = encoder_contrast(enc_c, 256, 0.5, "oracle_context")
    froz = encoder_contrast(enc_c, 256, 0.5, "frozen_profile")
    hard = encoder_contrast(hard_c, 256, 0.5, "oracle_context")
    np.testing.assert_allclose([v256[0], v16[0], prim[0], froz[0], hard[0]],
                               [-0.31, -60.90, -54.47, 7.16, -16.26],
                               atol=2e-2)
    # soft direct labels in the left strip, at the maximally separated
    # noise-0 endpoints, each with a hairline leader to its mark
    # (annotations, not text + plot: the leader then leaves the text box
    # edge with a fixed clearance in points, whatever the lock pass does to
    # the axes width)
    for size, y_tag, y_line in ((256, 85.0, 79.98), (64, 70.0, 75.06),
                                (16, 19.39, 19.39)):
        ax_soft.annotate(str(size), xy=(-0.02, y_line), xytext=(-0.22, y_tag),
                         fontsize=PT_SMALL, color=CAL_RAMP[size], ha="center",
                         va="center", zorder=6,
                         arrowprops=dict(arrowstyle="-", color=CAL_RAMP[size],
                                         lw=LW_HAIR, shrinkA=1.6, shrinkB=2.4))
    # hard direct labels fanned out from the noise-1 endpoints
    # (23.44 / 22.30 / 20.81 %), where the three dashed curves are 1.3 pt
    # apart and only a fan can name them
    for size, y_tag in ((256, 31.0), (64, 23.0), (16, 15.0)):
        y_end = float(grid[("hard", size)][0][2])
        ax_hard.annotate(str(size), xy=(1.0, y_end), xytext=(1.075, y_tag),
                         fontsize=PT_SMALL, color=CAL_RAMP[size], ha="left",
                         va="center", zorder=6,
                         arrowprops=dict(arrowstyle="-", color=CAL_RAMP[size],
                                         lw=LW_HAIR, shrinkA=1.2, shrinkB=2.6))
    return dict(soft={str(k): list(np.round(v[0], 4))
                      for k, v in grid.items() if k[0] == "soft"},
                hard={str(k): list(np.round(v[0], 4))
                      for k, v in grid.items() if k[0] == "hard"},
                oracle=oracle, frozen=frozen, mismatched=list(mismatched),
                soft_256_noise0_vs_oracle=list(v256),
                soft_16_noise0_vs_oracle=list(v16),
                soft_256_noise05_vs_oracle=list(prim),
                soft_256_noise05_vs_frozen=list(froz),
                hard_256_noise05_vs_oracle=list(hard))


def panel_gaps(canvas, ax, seeds, enc_c, hard_c):
    """F, right facet: paired accuracy minus oracle for the cells the text names.

    Replaces the two-line prose contrast that stood in the old panel's left
    wedge with the forest idiom: the mean and 95 % interval from
    ``paired_contrasts.csv`` over the per-seed fan from ``trajectories.csv``,
    soft rows filled, hard rows open.  The hard readout at noise 0 ties the
    oracle in every seed at every calibration size (all sixty differences are
    exactly zero), so it is one row.
    """
    rows, stats = [], {}
    for readout, size, noise, label in GAP_ROWS:
        table = enc_c if readout == "soft" else hard_c
        mean, lo, hi = encoder_contrast(table, size, noise, "oracle_context")
        fan = seeds[(readout, size, noise)]
        np.testing.assert_allclose(fan.mean(), mean, atol=1e-6)
        if readout == "hard" and noise == 0.0:
            for other in (16, 64):
                assert np.all(seeds[("hard", other, 0.0)] == 0.0)
            assert np.all(fan == 0.0)
        rows.append(dict(label=label, mean=mean, lo=lo, hi=hi,
                         seeds=list(fan), color=CAL_RAMP[size], n=20,
                         hollow=(readout == "hard")))
        stats[f"{readout}_{size}_noise{noise:g}_vs_oracle"] = [
            mean, lo, hi, int((fan > 0).sum()), 20]
    np.testing.assert_allclose([r["mean"] for r in rows],
                               [-0.31, -60.90, 0.00, -54.47, -16.26],
                               atol=6e-3)
    out = canvas.forest(ax, rows, value_label="Accuracy − oracle (pp)",
                        reference=0.0, reference_label="oracle",
                        xlim=(-68.0, 4.0), tag="", seed_alpha=0.45,
                        band=False, tick=True)
    ax.set_xticks([-60, -40, -20, 0])
    ax.spines["bottom"].set_bounds(-60, 0)
    return stats, out["gutter_pt"]


# ------------------------------------------------------------- build ----
def build():
    mpl.rcParams["lines.markeredgewidth"] = LW_EDGE
    RECORDS.mkdir(exist_ok=True)
    outcomes = pd.read_csv(DATA / "seed_outcomes.csv")
    summary = pd.read_csv(DATA / "condition_summary.csv")
    contrasts = pd.read_csv(REVIEW / "ancestry_k4_control_contrasts.csv")
    pairs = pd.read_csv(REVIEW / "ancestry_control_paired_differences.csv")
    pairs4 = pairs[pairs.budget_k.eq(4)]
    enc_c = pd.read_csv(ENCODER / "paired_contrasts.csv")
    hard_c = pd.read_csv(HARD / "paired_contrasts.csv")
    assert set(hard_c.analysis_status.unique()) == {
        "exploratory_paired_sensitivity"}
    prediction = coefficient_prediction()
    normalized = prediction.groupby("budget_k").normalized_coefficient_sum.mean()
    np.testing.assert_allclose(normalized.loc[list(K_TICKS)].to_numpy(float),
                               [-1.08, -0.03, 0.60, 1.00], atol=5e-3)
    tiers = task_tiers()
    supports = control_supports(0)
    paired = rewiring_pairs(outcomes)
    grid, oracle, frozen, mismatched = cue_grid()
    seeds = gap_seeds()
    # the factorial's own second bootstrap of the SAME contrast must still
    # agree on the mean; its ci95_high (1.97) is never printed (plan §4 E)
    second = pd.read_csv(DATA / "paired_contrasts.csv")
    hero = second[second.contrast.str.contains("best_matched", na=False)
                  & second.budget_k.eq(4)]
    if len(hero):
        np.testing.assert_allclose(float(hero.mean_difference.iloc[0]),
                                   0.012670898, atol=1e-6)

    # Design pass 2026-09-14: the width is fixed, the height is not, and a
    # panel gets the area its marks need.  Row 0 keeps the two schematics
    # (126 pt: the eight-module B clears PANEL_ASPECT_MAX 2.40 at 2.37).
    # Row 1 holds the two four-point accuracy panels at three modules each
    # beside the five-row forest at six.  Row 2 is panel F as three facets:
    # the soft and the hard readout at three modules each on one shared y
    # axis, and the paired accuracy-minus-oracle forest at six.  The 30 pt
    # vertical gutter was enough before F's letter also cleared the oracle
    # label above its right facet. Add 2 pt to both row gutters and 4 pt to
    # the page, preserving every axes size while clearing that full panel.
    canvas = NativeCanvas(454 / 72, 3, row_weights=[126, 96, 110],
                          hgutter_pt=28, vgutter_pt=32,
                          margins=Margins(left=44, right=12, top=22, bottom=36))
    a = canvas.panel("A", 0, 0, 4, title="Eight-stream task tree",
                     schematic=True, lock=False)
    b = canvas.panel("B", 0, 4, 8, title="Ancestry and control dictionaries",
                     schematic=True, lock=False)
    # data panels carry no titles: the axes and the direct labels say what
    # they show, the caption says what it means
    c = canvas.panel("C", 1, 0, 3)
    d = canvas.panel("D", 1, 3, 3, sharey=c)
    e = canvas.panel("E", 1, 6, 6)
    f_soft = canvas.panel("F", 2, 0, 3)
    f_hard = canvas.panel("F_hard", 2, 3, 3, letter="", sharey=f_soft)
    f_gap = canvas.panel("F_gap", 2, 6, 6, letter="")

    # 20 pt of declared left reserve on the y-titled panels of column 0:
    # audit_letter_alignment.py protects the letter column, and a rotated y
    # title reaches 24 pt left of its axes.  The shared-y partner in column 3
    # takes the same 20 pt on its RIGHT, so the pair is equal in width and
    # sits one gutter apart, its tick labels in that gutter.
    for name in ("C", "F"):
        canvas.declare_reserve(name, left=20)
    for name in ("D", "F_hard"):
        canvas.declare_reserve(name, right=20)
    # Keep the former 97.8-pt facet height: the extra gutter must remain
    # blank, not be reclaimed by the automatic top-reserve calculation.
    canvas.declare_reserve("F", top=12.2)

    panel_task(a, tiers)
    raw = panel_dictionaries(b, prediction, supports)
    stats_c = panel_bandwidth(c, summary, outcomes)
    stats_d = panel_rewiring(d, summary, paired)
    stats_e, gutter = panel_forest(canvas, e, contrasts, pairs4)
    stats_f = panel_cues(f_soft, f_hard, grid, oracle, frozen, mismatched,
                         enc_c, hard_c)
    stats_g, gap_gutter = panel_gaps(canvas, f_gap, seeds, enc_c, hard_c)

    style_direct_color_labels(canvas.fig)
    canvas.lock_reserves()
    _realign()                          # after the lock: see _realign
    findings = canvas.align_letters()
    problems = canvas.save(OUT, name="credit_first_figure_03", dpi=180,
                           lock=False)
    PUBLISHED.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(OUT, PUBLISHED)

    # ---- render-time records ------------------------------------------
    prediction.to_csv(RECORDS / "figure_03_coefficient_prediction.csv",
                      index=False)
    supports.to_csv(RECORDS / "figure_03_control_supports.csv", index=False)
    contrasts[contrasts.control.isin([k for _, k, _, _ in FOREST_ROWS])] \
        .to_csv(RECORDS / "figure_03_k4_contrasts.csv", index=False)
    source_f = []
    for (readout, size), (mean, lo, hi) in grid.items():
        for j, noise in enumerate((0.0, 0.5, 1.0)):
            source_f.append(dict(readout=readout, calibration_samples=size,
                                 cue_noise_sd=noise, cue_delay_trials=0,
                                 epoch=80, n_seeds=20,
                                 mean_accuracy_pct=float(mean[j]),
                                 ci95_low_pct=float(lo[j]),
                                 ci95_high_pct=float(hi[j])))
    pd.DataFrame(source_f).to_csv(
        RECORDS / "figure_03_coefficient_source.csv", index=False)

    # ---- the curated display table (panel meanings B..F) ---------------
    display = []
    display += [dict(panel="B", record="design prediction", **r)
                for r in prediction.to_dict("records")]
    display += [dict(panel="B", record="control support", **r)
                for r in supports.to_dict("records")]
    dend = summary[summary.architecture.eq("dendritic_tree")]
    for family in ("correct_ancestry_subtrees", "within_neuron_route_derangement"):
        display += [dict(panel="C", record="condition_mean", **r)
                    for r in dend[dend.feedback_family.eq(family)]
                    .to_dict("records")]
    for k, mean, low, high in routing._best_control_by_budget(outcomes):
        display.append(dict(panel="C", record="condition_mean",
                            feedback_family="best_matched_control",
                            budget_k=int(k), mean_heldout_accuracy=mean,
                            ci95_low_heldout_accuracy=low,
                            ci95_high_heldout_accuracy=high))
    rew = summary[summary.feedback_family.eq("correct_ancestry_subtrees")
                  & summary.architecture.isin(["dendritic_tree",
                                               "degree_depth_matched_rewired_tree"])]
    display += [dict(panel="D", record="condition_mean", **r)
                for r in rew.to_dict("records")]
    for K, v in paired.items():
        display.append(dict(panel="D", record="paired_contrast", budget_k=K,
                            n_pairs=20, mean_difference_pp=v[0],
                            ci95_low_pp=v[1], ci95_high_pp=v[2],
                            positive_seeds=v[3]))
    keys = [k for _, k, _, _ in FOREST_ROWS] + \
        ["within_neuron_route_derangement"]
    display += [dict(panel="E", record="mean_contrast", **r)
                for r in contrasts[contrasts.control.isin(keys)
                                   & contrasts.budget_k.eq(4)].to_dict("records")]
    display += [dict(panel="E", record="paired_seed", **r)
                for r in pairs4[pairs4.control.isin(keys)].to_dict("records")]
    display += [dict(panel="F", record="grid_mean", **r) for r in source_f]
    display += [dict(panel="F", record="paired_contrast", readout="soft", **r)
                for r in enc_c[enc_c.cue_delay_trials.eq(0)].to_dict("records")]
    display += [dict(panel="F", record="paired_contrast", readout="hard", **r)
                for r in hard_c[hard_c.cue_delay_trials.eq(0)].to_dict("records")]
    display += gap_seed_records(seeds)

    sources = [CONFIG, ENCODER_CONFIG,
               DATA / "seed_outcomes.csv", DATA / "condition_summary.csv",
               DATA / "paired_contrasts.csv",
               REVIEW / "ancestry_k4_control_contrasts.csv",
               REVIEW / "ancestry_control_paired_differences.csv",
               ENCODER / "condition_summary.csv", ENCODER / "trajectories.csv",
               ENCODER / "paired_contrasts.csv",
               HARD / "condition_summary.csv", HARD / "trajectories.csv",
               HARD / "paired_contrasts.csv"]
    builders = [Path(__file__), Path(routing.__file__),
                Path(experiment.__file__), JOURNAL / "scripts/figure_canvas.py",
                JOURNAL / "scripts/journal_style.py",
                JOURNAL / "scripts/native_schematics.py"]
    panels = {
        "A": "Schematic, no data: the frozen task configuration "
             "(configs/trained_subtree_address/full_factorial_confirmatory.json) "
             "drawn on the shared balanced feedback tree; selected +1.00 and "
             "sibling / same-half / opposite-half distractor coefficients.",
        "B": "Schematic, no data: ancestry dictionaries A (8 x K) from "
             "grouped_routes(correct_ancestry_subtrees) with the card footers "
             "the raw group sums of figure_03_coefficient_prediction.csv, and "
             "the delivered support of b3's credit under the four matched "
             "controls at K = 4 (figure_03_control_supports.csv, rng seed 0). "
             "The control heatmap shows signed delivered fields Φ[c, b] "
             "divided by the maximum absolute value within each row; the "
             "source table retains the signed, unit-norm field values.",
        "C": "condition_summary.csv means with 95% seed-bootstrap intervals "
             "(dendritic_tree under correct ancestry and under derangement) "
             "and the per-seed post-training maximum over the four matched "
             "controls from seed_outcomes.csv.",
        "D": "condition_summary.csv means with 95% intervals for "
             "dendritic_tree versus degree_depth_matched_rewired_tree under "
             "ancestry feedback; paired differences and 20,000-draw seed "
             "bootstrap (seed 70000) from seed_outcomes.csv.",
        "E": "review_evidence_reanalysis/ancestry_k4_control_contrasts.csv "
             "means, 95% intervals, positive-seed counts and Holm-adjusted P "
             "for five controls (the deranged row off scale, printed as its "
             "value); the seed fan is "
             "ancestry_control_paired_differences.csv at K = 4.",
        "F": "review_coefficient_encoder and review_coefficient_hard_readout "
             "condition_summary.csv over the calibration x cue-noise grid at "
             "cue delay 0, with render-time 95% seed-bootstrap bands from "
             "trajectories.csv at epoch 80 (soft and hard facets); the right "
             "facet draws the paired_contrasts.csv learned-minus-oracle means "
             "and 95% intervals over the per-seed differences recomputed from "
             "trajectories.csv (the hard readout is exploratory).",
    }
    deviations = [
        "row weights 124/113/113 at a 44 pt vertical gutter and margins "
        "44/12/20/32, not the plan's 112/121/121 at 42 pt with margins "
        "52/14.  FILL_W_MIN 0.92 makes left + right <= 57.5 pt a hard "
        "constraint (52 + 14 fills only 90.3 % of the page width and fails "
        "the strict audit's fill-width check); at a 112 pt row 0 the "
        "eight-module panel B has axes aspect 2.64, above PANEL_ASPECT_MAX "
        "2.40, which DECISIONS G4 forbids relaxing, and 124 pt is the "
        "shortest row 0 that clears it; a 42 pt gutter leaves 7.2 pt between "
        "rows 1 and 2 against audit_row_separation.py's 8.5 pt floor, 44 pt "
        "measures 9.4 pt.  The two data rows gain 6 pt against the previous "
        "build and the schematic fraction falls to 26.0 %",
        "E uses lock=False plus a measured private left inset instead of "
        "declare_reserve(left=66): a declared reserve locks the whole grid "
        "column and C starts in column 0, so C and D would differ in width",
        "E draws its per-row wins / Holm notes inside the axes above each "
        "seed fan; forest(note=...) writes into the gutter shared with F",
        "E prints Holm P as 1.8 x 10^-4 / 7.6 x 10^-6 with the exponent set "
        "by token_subscript at a NEGATIVE drop, i.e. a real 7.0 pt raised "
        "span: Nimbus Sans has no U+207B and CF-2 forbids mathtext, so the "
        "raised group is right-aligned by measuring the drawn chain",
        "A's four coefficient tiers are one contiguous band in spatial order "
        "with per-block tier names and hairline connectors to their terminal "
        "groups, and tint per cent 13/22/34/46 so the ORDINAL_RAMP reads "
        "light-to-dark: a 19.5 pt coefficient cannot be centred on a 12.5 pt "
        "terminal pitch, so the band is a partition, not four terminal-"
        "aligned patches; the tier gloss is the spatial order and drops "
        "'nonsibling' for width",
        "A drops the 'junctions define feedback supports only' footer into "
        "the caption -- the plan's stated fallback",
        "B draws no tree in the ancestry cards, so the subtree entry arrow "
        "comes from the private _delivery_arrow rather than credit_delivery",
        "B's control support table is 4 x 10 pt rows by 8 x 7 pt columns, not "
        "6 x 6 pt: a 7 pt row label cannot sit on a 6 pt row",
        "DELIBERATE, SOURCED: D's paired intervals use one 20,000-draw "
        "bootstrap seeded 70000 over the frozen seed_outcomes.csv; K = 2 "
        "gives [21.03, 25.32] against the plan's quoted [21.05, 25.32].  "
        "Same estimator, same data, same mean (+23.149 pp); only the "
        "resampling draw differs and the plan's RNG state is unrecorded, so "
        "the caption prints the endpoint this builder computes",
        "C and F carry shortened forms of the plan's long annotations; the "
        "full wording is in the caption",
        "D's rewired label is direct-labelled on its own series under the "
        "K = 1-2 segment with the 'control' badge inline, not parked over "
        "the y-tick cluster",
        "A's context double ring is dropped 4.6 pt down its own branch so "
        "the ring rim and the b3 terminal label clear each other",
        "C's K = 8 tie tag names the two families that actually tie "
        "(ancestry = best matched control, 81.00 %): the deranged square is "
        "at 25.13 % at K = 8, so the plan's 'all families tie' is false",
        "C's tie tags print 18.66 %, not the plan's 18.67 %: the frozen "
        "condition_summary gives 18.6646 % for both arms at K = 1, which is "
        "also what the plan's own deranged row rounds to",
        "E runs to xlim (-2, 14) instead of (-2, 11) so the +13.13 dense "
        "rank-4 seed is drawn rather than clipped, and the off-scale note "
        "names it in its planned second clause; the note is set on two "
        "lines and the right-aligned column moved from x = 13.7 to 12.85 to "
        "keep audit_panel_gaps.py's 6 pt floor against F",
        "VISUAL REVIEW 2026-09-10: C and D run on ylim (-4, 100), not "
        "(-20, 100); C drops its 'below chance' line; D carries y tick "
        "labels and the y title, a one-line rewired label, the title "
        "'Rewiring costs 23 and 5 pp at K = 2, 4' and no K = 4 span; F "
        "drops its n line and three of four contrast lines, marks the "
        "three sampled doses (filled soft, open hard), draws the frozen "
        "floor dashed at LW_REF and its oracle rule in CF-7 mute rather "
        "than violet; B lists its controls in E's order (dense rank-4, "
        "random-sparse, depth-interleaved, deranged) while the rng draw "
        "order is unchanged, so every support value is the shipped one",
        "F direct-labels the three dashed hard-readout curves 256 / 64 / "
        "16 at their noise-0.5 vertices (64.03 / 54.22 / 32.41 %) with a "
        "'hard readout' tag and the 'exploratory' badge, and runs on the "
        "plan's ylim (10, 90) so the 18.67 % frozen floor clears the bottom "
        "spine by 11 pt; the plan's three long annotations are reduced to "
        "the panel's own claim, the -54.47 pp gap at noise 0.5, on two "
        "7 pt lines in the one mark-free wedge",
        "F carries the plan's ylim (10, 90); the +7.16 pp vs frozen "
        "contrast, the noise-0 gaps (-0.31 / -60.90 pp), the n / interval / "
        "epoch statement, the seed range 52000-52019 and the cue-delay "
        "clause are carried by the caption and the running text because the "
        "window leaves no band for them",
        "B's control-table column headers are the digits 1..8 under one b_i "
        "tag, not eight subscripted b_j strings: a literal Unicode subscript "
        "is not in Nimbus Sans and mathtext is banned by CF-2",
        "B's |A| scale-bar tag is plain text, not Frame.subscript",
        "F's oracle rule is labelled 'oracle 80.3 %' with no 'oracle' badge "
        "(the panel already carries the 'exploratory' badge; a second chip "
        "on the same rule would sit on the dashed hard-readout curves)",
        "private helpers _badge (ceiling, exploratory), _delivery_arrow, "
        "_gap_span",
        "DESIGN PASS 2026-09-14: 450 pt tall on rows 126/96/110 at a 28 pt "
        "horizontal and 30 pt vertical gutter; C and D are three-module "
        "panels sharing one y axis (D keeps its tick labels, drops the "
        "title); E is a five-row forest at six modules with the row name "
        "over its wins / Holm P in the label column and the deranged "
        "control as an off-scale arrow with its value; F is three facets "
        "(soft, hard, paired accuracy minus oracle) under one letter; no "
        "data panel carries a title or an n / interval / epoch line; C's "
        "tie tags, D's 'exact tie' line, E's off-scale prose and F's "
        "contrast prose are gone (caption, concentric marks and the new "
        "facet carry them); the mismatched-encoder range is asserted but "
        "not printed",
    ]
    live_w = canvas.width_pt - canvas.margins.left - canvas.margins.right
    live_h = canvas.height_pt - canvas.margins.top - canvas.margins.bottom
    schematic_fraction = sum(w * h for _, _, w, h in
                             (canvas.slot_pt(0, 0, 4), canvas.slot_pt(0, 4, 8))
                             ) / (live_w * live_h)
    payload = dict(
        panel_sources=panels,
        source_sha256={str(p.relative_to(JOURNAL)):
                       hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in sources},
        layout_findings=list(findings) + list(problems),
        forest_gutter_pt=round(gutter, 2),
        gap_gutter_pt=round(gap_gutter, 2),
        schematic_area_fraction=round(schematic_fraction, 4),
        g4_waiver="G4 waiver recorded for Fig 3 at the superseded 31.6 % "
                  "measure; on the B12 formula the figure is "
                  f"{100 * schematic_fraction:.1f} % and the waiver is dormant",
        derived_numbers=dict(
            group_sums={int(k): float(v) for k, v in raw.items()},
            normalized_group_sums={int(k): float(v)
                                   for k, v in normalized.items()},
            accuracy_by_K=stats_c, rewiring_pp=stats_d,
            k4_contrasts_pp=stats_e, cue_grid=stats_f, cue_gaps=stats_g),
        coefficient_scope="Raw group sums; implemented route rows divide by "
                          "sqrt(group size). Signs follow from the generator "
                          "and are not a new prospective prediction.",
        caption_word_count=caption_words(),
        deviations_from_plan=deviations)
    assert 220 <= payload["caption_word_count"] <= 350, \
        payload["caption_word_count"]
    (RECORDS / "figure_03_caption.tex").write_text(CAPTION + "\n")
    (RECORDS / "figure_03_sources.json").write_text(
        json.dumps(payload, indent=2, default=float) + "\n")
    publish(3, OUT, display, sources, builders, panels, emit_main=False,
            layout_findings=list(findings) + list(problems),
            notes="Overhaul rebuild of main Fig 3 (v2 plan, 2026-09-09). No "
                  "new training, fit, endpoint selection or reseeding; every "
                  "value is re-read from the frozen Source Data.")
    return list(findings) + list(problems), payload


def main(argv=None):
    # rebuild_final_publication_figures.py passes --emit-main (or nothing);
    # this builder always writes figures/main/figure_03.pdf, so the flag is
    # accepted and recorded rather than acted on.
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--emit-main", action="store_true")
    parser.parse_args(argv)
    problems, payload = build()
    for p in problems:
        print(f"  {p}")
    print(json.dumps(payload["derived_numbers"], indent=1, default=float))


if __name__ == "__main__":
    main()
