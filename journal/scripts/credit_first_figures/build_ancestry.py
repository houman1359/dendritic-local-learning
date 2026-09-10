#!/usr/bin/env python3
"""Main Fig. 3 (fig:subtreefactorial): ancestry routes, matched controls, cues.

Rebuilt 2026-09-09 on the NativeCanvas per
``analysis/figure_overhaul_20260908/v2/fig3/PLAN.md`` as amended by
``v2/AMENDMENTS.md`` and ruled by ``v2/DECISIONS.md``.  Three rows, six
panels, letters in citation order and in row-major order:

    row 0  A  eight-stream task tree (4 mod)   B  ancestry + control
                                                   dictionaries (8 mod)
    row 1  C  accuracy across K (6 mod)        D  rewiring control (6 mod)
    row 2  E  K = 4 forest (7 mod)             F  cue cohort grid (5 mod)

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
``(128.8 + 295.6) x 128 = 54 323.2`` of ``202 531.2`` = **26.8 %**, inside
the 30 % cap.  G4 waiver recorded for Fig 3 at the superseded 31.6 %
measure; on the B12 formula the figure is 26.8 % and the waiver is
dormant.  PANEL_ASPECT_MAX is not relaxed.

Recorded deviations from the plan (each with its reason; the full list is
also written into ``figure_03_sources.json``):

* Row weights 128/107/107 with a 48 pt vertical gutter and margins
  44/12/20/32, not the plan's 112/121/121 at 42 pt with margins 52/14.  Panel B is eight
  modules wide (288.9 pt); at a 112 pt row its axes box is 2.58 wide,
  above ``PANEL_ASPECT_MAX`` 2.40, which DECISIONS G4 forbids relaxing.
  128 pt clears the cap (2.31); the 48 pt
  gutter is what audit_row_separation.py's 8.5 pt floor needs between the
  x labels of row 1 and the letters of row 2, and it leaves the two data
  rows at 107 pt.  The left margin is 44 rather than 52 so that the
  figure fills 92 % of the canvas width, which the strict audit requires.
  The AMENDMENTS Fig 3 item 5 "tighten to 106/124/124" instruction is
  therefore not taken: it would put B at 2.73.
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
* E prints Holm P values in decimal (0.00018, 0.0000076) rather than
  1.8 x 10^-4 / 7.6 x 10^-6: Nimbus Sans has no U+207B superscript minus
  and CF-2 forbids mathtext.  The caption keeps the scientific form.
* A's coefficient tier labels are set in a two-row keyed band rather than
  over their own blocks: at the 12-16 pt terminal pitch a 17.6 pt
  coefficient string cannot sit over a one-terminal block without
  colliding with its neighbour.  The swatches key the tints, so the band
  is the partition label.
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
  seeded 70000; K = 2 comes out [21.03, 25.32] against the plan's quoted
  [21.05, 25.32] (same estimator, different draw).  The printed tags carry
  the means only.
* C and F carry the plan's long annotations in shortened form; the full
  wording is in the caption.  A 166 pt panel cannot hold a 295 pt line.
  C prints the two tie tags verbatim but at 18.66 %, the value the frozen
  ``condition_summary.csv`` carries for both arms at K = 1.
* E runs to xlim (-2, 14) rather than the plan's (-2, 11), so the +13.13
  dense rank-4 seed is drawn instead of clipped.
* F does not repeat 256 / 64 / 16 on the dashed hard-readout lines: they
  carry the same ORDINAL_RAMP hues as the labelled solid lines and the
  noise-0.5 points have no clearance for three more labels.  Its
  noise-free annotation names both oracle gaps on one line.
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
                           Margins, NativeCanvas, forest)
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

CONTROL_ROWS = (("deranged", "within_neuron_route_derangement"),
                ("depth-interleaved", "depth_interleaved_bins"),
                ("random-sparse", "random_sparse_matched"),
                ("dense rank-4", "random_rank_k"))
FOREST_ROWS = (("best matched\ncontrol", "best_matched_nonanatomical_oracle",
                "oracle", "p_holm_four_budgets", "four budgets"),
               ("dense rank-4", "random_rank_k",
                "point_mlp", "p_holm_four_individual_controls_at_k4",
                "four controls"),
               ("random-sparse", "random_sparse_matched",
                "point_mlp", "p_holm_four_individual_controls_at_k4", None),
               ("depth-\ninterleaved", "depth_interleaved_bins",
                "point_mlp", "p_holm_four_individual_controls_at_k4", None))
Y_LIM_C = (-20.0, 100.0)

#: The shipped caption (analysis/figure_overhaul_20260908/v2/fig3/TEXT.md).
#: Counted under CF-8 by :func:`caption_words` and recorded in the manifest.
CAPTION = r"""\caption{\textbf{Ancestry routes help only where the tree's partition matches the task, and only when the coefficients are supplied.}
\textbf{A}, Schematic, no data: eight streams enter the terminals $b_1$--$b_8$ of a balanced tree; context $c$ selects $b_3$, whose block alone sets the logit $z$, the somatic error $\delta_0$ enters the soma, and the $K=4$ ancestry capsule delivers it to $b_3$ and sibling $b_4$. Junctions define feedback supports only; teal tints give each stream's class coefficient ($+1.00$ cued, $-0.15$ sibling, $-0.45$ same-half nonsibling, $-0.75$ other half).
\textbf{B}, Schematic, no data: ancestry dictionaries $A$ ($8\times K$), the channel carrying $b_3$'s credit shaded, over raw class-signal sums ($-3.05$, $-0.05$, $+0.85$, $+1.00$; unit-row normalization preserves the signs, giving $-1.08$, $-0.03$, $+0.60$, $+1.00$), beside the delivered support of $b_3$'s credit under the four matched controls at $K=4$ (random-sparse and dense rows, one draw).
\textbf{C}, Held-out accuracy across budgets for the ancestry route, the deranged route and the best matched control (the per-seed maximum over the four controls, a ceiling); 20 paired seeds, per cent, means with 95\% seed-bootstrap intervals at epoch 80.
\textbf{D}, Task-matched versus degree- and depth-matched rewired tree under ancestry feedback: paired $+23.15$ [$21.03$, $25.32$] and $+5.02$ [$4.19$, $5.94$] percentage points at $K=2,4$, exact ties at $K=1,8$; 20 paired seeds, 95\% paired seed-bootstrap intervals at epoch 80.
\textbf{E}, Ancestry minus each control at $K=4$, intervals from \texttt{review\_evidence\_reanalysis/ancestry\_k4\_control\_contrasts.csv}, Holm-adjusted $P$ across four budgets (top row) or four controls; 20 paired seeds, percentage points, means with 95\% seed-bootstrap intervals at epoch 80.
\textbf{F}, Coefficient source across the calibration $\times$ cue-noise grid, soft readout solid, exploratory hard readout dashed, against the oracle ceiling and the frozen-profile floor, printed noise-free values being oracle gaps; 20 fresh seeds (52000--52019), per cent, means with 95\% seed-bootstrap bands at epoch 80.
Teal ramps in \textbf{A} and \textbf{F} are ordinal within this figure's own lists, not shared identities.
Source Data: \texttt{source\_data/curated\_publication/figure\_03\_plotted.csv}.}"""




def caption_words(text=None):
    """CF-8 word count: macros stripped, each inline math group one word."""
    import re
    t = (CAPTION if text is None else text).strip()
    t = t[len("\\caption{"):-1]
    t = re.sub(r"\$[^$]*\$", " MATH ", t)
    t = re.sub(r"\\text(bf|tt)\{([^}]*)\}", r" \2 ", t)
    t = re.sub(r"\\[a-zA-Z]+", " ", t)
    t = t.replace("\\%", "%").replace("--", " ")
    return len([w for w in re.split(r"\s+", t) if w.strip(" .,;:()[]{}")])


def _signed(v, digits=2):
    return f"{v:+.{digits}f}".replace("-", "−")


def _minus(s):
    return str(s).replace("-", "−")


def _p_text(p):
    """Holm P at two significant digits, without a superscript minus.

    Nimbus Sans has no U+207B, and CF-2 forbids mathtext, so the panel
    prints the decimal form and the caption keeps 1.8 x 10^-4 / 7.6 x 10^-6.
    """
    if p >= 1e-3:
        return f"{p:.4f}"
    digits = int(np.floor(np.log10(p)))
    return f"{p:.{1 - digits}f}"


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
    rows = []
    for _, mode in CONTROL_ROWS:
        if mode == "random_rank_k":
            field = experiment.random_rank_routes(np.arange(8), 4, rng)
        else:
            field = experiment.grouped_routes(np.arange(8), 4, mode, rng)
        rows.append(field[CUED])
    table = pd.DataFrame(np.array(rows), columns=[f"b{i + 1}" for i in range(8)])
    table.insert(0, "control", [m for _, m in CONTROL_ROWS])
    table.insert(1, "cued_stream", "b3")
    return table


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


def _gap_span(ax, x, y_lo, y_hi, tag, tag_xy, *, dx=0.09, cap=0.05):
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
    ax.plot([xs + side * cap, tag_xy[0] - side * 0.03], [mid, tag_xy[1]],
            color=MUTE, lw=LW_HAIR, zorder=1)
    return ax.text(tag_xy[0], tag_xy[1], tag, fontsize=PT_ANNOT, color=INK,
                   ha="left" if side > 0 else "right", va="center")


def _group_root(nodes, K, terminal="T3"):
    for root in nodes.at_depth(K):
        if terminal in nodes.terminals_under(root):
            return root
    raise KeyError(terminal)


def _series(ax, x, part, color, marker, ms, zorder, *, filled=True):
    mean = part.mean_heldout_accuracy.to_numpy(float) * 100.0
    low = part.ci95_low_heldout_accuracy.to_numpy(float) * 100.0
    high = part.ci95_high_heldout_accuracy.to_numpy(float) * 100.0
    ax.errorbar(x, mean, yerr=[mean - low, high - mean], color=color,
                marker=marker, mfc=color if filled else "white", mec=color,
                mew=LW_EDGE, ms=ms, lw=LW_DATA, elinewidth=LW_ERR,
                capsize=ERR_CAPSIZE, capthick=LW_ERR, zorder=zorder)
    return mean


def _accuracy_axes(ax, *, ylabel):
    ax.set_xlim(-0.20, 3.20)
    ax.set_ylim(*Y_LIM_C)
    ax.set_xticks(range(4), [str(k) for k in K_TICKS])
    ax.set_yticks([20, 40, 60, 80])
    ax.spines["left"].set_bounds(20, 80)
    ax.spines["bottom"].set_bounds(0, 3)
    ax.set_xlabel("Channels, K")
    if ylabel:
        ax.set_ylabel("Held-out accuracy (%)")
    else:
        ax.tick_params(axis="y", labelleft=False)


# ----------------------------------------------- A: eight-stream task ----
def panel_task(ax, tiers):
    f = Frame(ax)
    W, H = f.w_pt, f.h_pt
    sub_pt, key_pt, delta_pt = LINE_BAND_PT, 19.5, 12.0
    # sub-title: the cued stream, and the rule under test on it
    f.subscript((f.fx(1.0), 1.0 - f.fy(sub_pt * 0.5)), "context c selects b", "3",
                size=PT_SMALL, color=INK, ha="left")
    f.text((1.0 - f.fx(1.0), 1.0 - f.fy(sub_pt * 0.5)), "K = 4 route",
           size=PT_SMALL, color=GREEN, ha="right")
    rect = (0.0, f.fy(delta_pt), 1.0, 1.0 - f.fy(sub_pt + key_pt + delta_pt))
    nodes = f.balanced_tree(rect, depth=3, mode="forward", trunk=True,
                            labels=True, output="z", soma_r_pt=3.0,
                            input_labels=[("b", str(i + 1)) for i in range(8)])
    # coefficient tiers (QA 2026-09-09): NOT drawn as tints on the tree,
    # where they merged with each other and with the K = 4 delivery capsule.
    # Each tier is a 16 % ORDINAL_RAMP band in the key row above its own
    # leaves, carrying its coefficient, so tier and delivery never share a fill.
    f.note("address_tint",
           reason="coefficient tier by tree distance, not an address")
    # the two single-leaf bands (b3, b4) are staggered by 5.5 pt, because a
    # 7 pt coefficient is as wide as one leaf pitch
    blocks = [(["T3"], tiers["selected"], TIER_RAMP[0], 2, 5.5),
              (["T4"], tiers["sibling"], TIER_RAMP[1], 2, 0.0),
              (["T1", "T2"], tiers["same_half"], TIER_RAMP[2], 2, 0.0),
              (["T5", "T6", "T7", "T8"], tiers["other_half"], TIER_RAMP[3], 2,
               0.0)]
    band_y = 1.0 - f.fy(sub_pt + 14.0)
    half = f.fx(nodes.pitch_pt / 2.0 - 1.2)      # 2.4 pt between bands
    for leaves, coef, colour, digits, dy in blocks:
        xs = [nodes[t][0] for t in leaves]
        x0, x1 = min(xs) - half, max(xs) + half
        yb = band_y + f.fy(dy)
        tint_patch(ax, ("rect", x0, yb - f.fy(4.6), x1 - x0, f.fy(9.2)),
                   color=colour, pct=16, radius_pt=1.0, zorder=1.0,
                   clip_on=False)
        f.text(((x0 + x1) / 2.0, yb), _signed(coef, digits), size=PT_SMALL,
               color=INK)
    # the forward route of the cued stream is the one ink-weight path
    for a, b in zip(nodes.route("T3"), nodes.route("T3")[1:]):
        if (a, b) in nodes.edges:
            nodes.edges[(a, b)].set_linewidth(f.lw(LW_DATA))
            nodes.edges[(a, b)].set_color(INK)
    # context selection: the inhibitory-family double ring with a 'c' badge
    f.gate(nodes["T3"], closed=False, badge="c", nodes=nodes, node="T3",
           badge_offset=(-5.6, -3.6))
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
    S = np.abs(supports[[f"b{i + 1}" for i in range(8)]].to_numpy(float))
    S = S / S.max(axis=1, keepdims=True)
    inner = f.dictionary_matrix((mx, my, mw, mh), S, color="point_mlp",
                                label=None, row_groups=[1] * 4,
                                col_labels=[str(i + 1) for i in range(8)],
                                yticks=[name for name, _ in CONTROL_ROWS],
                                min_cell_pt=6.0)
    inner.tick_params(axis="y", labelsize=PT_SMALL, pad=2.0)
    f.subscript((mx - f.fx(2.0), my + mh + f.fy(2.4)), "b", "i",
                size=PT_SMALL, color=MUTE, ha="right")
    # 3-stop key for the grey levels of the dense rank-4 row
    bar_y, bar_w = f.fy(17.0), f.fx(9.0)
    f.text((mx - f.fx(1.5), bar_y + f.fy(2.7)), "|A|", size=PT_SMALL,
           color=MUTE, ha="right")
    for j, level in enumerate((0.0, 0.5, 1.0)):
        bx = mx + f.fx(3.0) + j * (bar_w + f.fx(15.0))
        tint_patch(ax, ("rect", bx, bar_y, bar_w, f.fy(5.4)), color="point_mlp",
                   pct=int(round(8 + 84 * level)), radius_pt=0.4, zorder=3,
                   clip_on=False)
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
                       "s", MARKER_MS + 0.6, 3)
    ax.errorbar(x, best[:, 1], yerr=[best[:, 1] - best[:, 2],
                                     best[:, 3] - best[:, 1]],
                color=PURPLE, marker="D", mfc=PURPLE, mec=PURPLE, mew=LW_EDGE,
                ms=MARKER_MS + 0.9, lw=LW_DATA, elinewidth=LW_ERR,
                capsize=ERR_CAPSIZE, capthick=LW_ERR, zorder=4)
    ancestry = _series(ax, x, part("correct_ancestry_subtrees"), GREEN, "o",
                       MARKER_MS - 1.6, 6)
    np.testing.assert_allclose(ancestry, [18.67, 44.86, 80.11, 81.00], atol=2e-2)
    np.testing.assert_allclose(best[:, 1], [54.50, 60.26, 78.84, 81.00],
                               atol=2e-2)
    np.testing.assert_allclose(deranged, [18.66, 18.47, 19.05, 25.13], atol=2e-2)
    _accuracy_axes(ax, ylabel=True)
    reference_line(ax, 50.0, label="chance", span=(-0.20, 3.20))
    # direct labels (no legend artists anywhere in this figure)
    ax.text(0.02, 84.0, "best matched control", color=PURPLE,
            fontsize=PT_SMALL, ha="left", va="center")
    _badge(ax, (1.24, 84.0), "ceiling", ha="left", va="center")
    ax.text(0.72, 30.0, "ancestry", color=GREEN, fontsize=PT_SMALL, ha="left",
            va="center")
    # the K = 4 pair (80.1 vs 78.8) is not a tie: name the gap (QA 2026-09-09)
    ax.annotate("+1.27 pp (E)", xy=(2.06, 79.6), xytext=(2.30, 68.0),
                fontsize=PT_SMALL, color=INK, ha="left", va="center",
                arrowprops=dict(arrowstyle="-", color=MUTE, lw=LW_HAIR,
                                shrinkA=0, shrinkB=1.5), zorder=6)
    ax.text(2.20, 31.0, "deranged", color=GREY, fontsize=PT_SMALL, ha="left",
            va="center")
    ax.text(3.16, 95.0, "n = 20 paired seeds; mean [95 % bootstrap]; epoch 80",
            fontsize=PT_SMALL, color=MUTE, ha="right", va="center")
    # the two concentric ties, named verbatim, and the sub-chance cause,
    # set under the data (plan §4 C)
    ax.text(-0.16, 8.0, f"K = 1: ancestry = deranged, {ancestry[0]:.2f} %",
            fontsize=PT_SMALL, color=MUTE, ha="left", va="center")
    ax.text(-0.16, -2.0, f"K = 8: all families tie, {ancestry[3]:.2f} %",
            fontsize=PT_SMALL, color=MUTE, ha="left", va="center")
    ax.text(-0.16, -12.0,
            "below chance: pooled class signal is sign-reversed (B)",
            fontsize=PT_SMALL, color=MUTE, ha="left", va="center")
    return dict(ancestry=ancestry.tolist(), best=best[:, 1].tolist(),
                deranged=deranged.tolist())


# ------------------------------------------ D: the rewiring alignment ----
def panel_rewiring(ax, summary, paired):
    x = np.arange(4, dtype=float)
    fam = summary[summary.feedback_family.eq("correct_ancestry_subtrees")]
    part = (lambda arch: fam[fam.architecture.eq(arch)]
            .set_index("budget_k").loc[list(K_TICKS)])
    rewired = _series(ax, x, part("degree_depth_matched_rewired_tree"), ROSE,
                      "^", MARKER_MS + 0.6, 3)
    matched = _series(ax, x, part("dendritic_tree"), GREEN, "o",
                      MARKER_MS - 1.6, 6)
    np.testing.assert_allclose(rewired, [18.67, 21.71, 75.09, 81.00], atol=2e-2)
    _accuracy_axes(ax, ylabel=False)
    reference_line(ax, 50.0, label="chance", span=(-0.20, 3.20))
    _gap_span(ax, 1.0, rewired[1], matched[1],
              f"{_signed(paired[2][0])} pp", (1.42, 33.0))
    _gap_span(ax, 2.0, rewired[2], matched[2],
              f"{_signed(paired[4][0])} pp", (1.62, 74.0), dx=-0.09)
    _badge(ax, (-0.05, 85.0), "control", ha="left", va="center")
    ax.text(-0.16, 76.0, "degree- and depth-", color=ROSE, fontsize=PT_SMALL,
            ha="left", va="center")
    ax.text(-0.16, 69.0, "matched rewiring", color=ROSE, fontsize=PT_SMALL,
            ha="left", va="center")
    ax.text(2.05, 62.0, "task-matched tree", color=GREEN, fontsize=PT_SMALL,
            ha="left", va="center")
    ax.text(-0.16, 8.0, "exact tie at K = 1 and at K = 8", color=MUTE,
            fontsize=PT_SMALL, ha="left", va="center")
    ax.text(3.16, 95.0, "n = 20 paired seeds; mean [95 % bootstrap]; epoch 80",
            fontsize=PT_SMALL, color=MUTE, ha="right", va="center")
    return {K: list(v[:3]) for K, v in paired.items()}


# --------------------------------------------------- E: the K = 4 forest ----
def panel_forest(canvas, ax, contrasts, pairs, *, cap_pt=62.0):
    rows, stats = [], {}
    for label, key, colour, holm, family in FOREST_ROWS:
        row = contrasts[contrasts.control.eq(key) & contrasts.budget_k.eq(4)]
        assert len(row) == 1
        row = row.iloc[0]
        seeds = pairs[pairs.control.eq(key)].accuracy_difference_pp \
            .to_numpy(float)
        assert len(seeds) == 20 and int(row.n_pairs) == 20
        note = (f"{int(row.positive_seeds)}/20, Holm P = "
                f"{_p_text(float(row[holm]))}")
        if family:
            note += f" ({family})"
        rows.append(dict(label=label, mean=float(row.mean_difference_pp),
                         lo=float(row.ci95_low_pp), hi=float(row.ci95_high_pp),
                         seeds=list(seeds), color=colour, n=20, note=note))
        stats[key] = [float(row.mean_difference_pp), float(row.ci95_low_pp),
                      float(row.ci95_high_pp), int(row.positive_seeds), 20,
                      float(row[holm])]
    assert stats["random_rank_k"][3] == 15, "dense rank-4 is 15/20, not 20/20"
    assert stats["best_matched_nonanatomical_oracle"][3] == 15
    assert stats["random_sparse_matched"][3] == 20
    assert stats["depth_interleaved_bins"][3] == 20
    np.testing.assert_allclose(
        [stats[k][0] for k in ("best_matched_nonanatomical_oracle",
                               "random_rank_k", "random_sparse_matched",
                               "depth_interleaved_bins")],
        [1.27, 2.31, 3.77, 7.11], atol=6e-3)
    np.testing.assert_allclose(
        [stats[k][1] for k in ("best_matched_nonanatomical_oracle",
                               "random_rank_k", "random_sparse_matched",
                               "depth_interleaved_bins")],
        [0.59, 1.00, 2.89, 6.19], atol=6e-3)
    np.testing.assert_allclose(
        [stats[k][2] for k in ("best_matched_nonanatomical_oracle",
                               "random_rank_k", "random_sparse_matched",
                               "depth_interleaved_bins")],
        [1.95, 3.91, 4.74, 8.03], atol=6e-3)
    np.testing.assert_allclose(
        [stats[k][5] for k in ("best_matched_nonanatomical_oracle",
                               "random_rank_k", "random_sparse_matched",
                               "depth_interleaved_bins")],
        [0.010140, 0.003380, 1.766e-4, 7.629e-6], rtol=2e-3)
    # the label gutter is measured, then applied as a PRIVATE inset (see the
    # module docstring): a declared reserve is locked per grid column and C
    # starts in the same column.
    widest = max(_text_w_pt(ax, line, PT_SMALL)
                 for r in rows for line in str(r["label"]).split("\n"))
    gutter = min(widest + 7.0, cap_pt)
    # the lock pass re-places the panel from its slot, so only the record is
    # touched here: calling ax.set_position() as well would be captured as a
    # manual nudge and applied twice.
    record = canvas._record_for("E")
    record["inset_pt"] = (gutter, 0.0, 0.0, 0.0)
    notes = [r.pop("note") for r in rows]
    # CF-6 allows a 0.55 pt row tick OR a 6 % band; the tick is used because a
    # band patch leaves no clear strip for the per-row note.
    out = forest(ax, rows, value_label="Ancestry − control (pp)",
                 reference=0.0, reference_label="no difference",
                 xlim=(-2.0, 14.0), tag="", seed_alpha=0.45, band=False,
                 tick=True)
    ax.set_ylim(4.62, -0.60)
    # QA 2026-09-09: 0.38 rows above the row it annotates (7.8 pt above its
    # marker, 12.7 pt below the previous row), so attribution is unambiguous
    for y, note in zip(out["ypos"], notes):
        ax.text(13.7, y - 0.42, note, fontsize=PT_SMALL, color=MUTE,
                ha="right", va="center")
    _badge(ax, (13.7, 0.0), "ceiling", ha="right", va="center")
    derange = contrasts[contrasts.control.eq("within_neuron_route_derangement")
                        & contrasts.budget_k.eq(4)].iloc[0]
    ax.text(-1.9, 3.85,
            f"off scale: derangement {_signed(derange.mean_difference_pp)} "
            f"[{derange.ci95_low_pp:.2f}, {derange.ci95_high_pp:.2f}] pp "
            f"(20/20)", fontsize=PT_SMALL, color=MUTE, ha="left", va="center")
    ax.text(13.7, 4.30,
            "n = 20 paired seeds; mean [95 % seed bootstrap]; epoch 80",
            fontsize=PT_SMALL, color=MUTE, ha="right", va="center")
    for line in ax.lines:                 # CF-7: zero drawn once, and only
        xd = list(line.get_xdata())       # over the rows it refers to
        if len(xd) == 2 and xd[0] == xd[1] == 0.0:
            line.set_ydata([0.20, 1.0])
    ax.set_xticks([0, 4, 8, 12])
    ax.spines["bottom"].set_bounds(0, 12)
    return stats, gutter


# ------------------------------------------------- F: the cue cohort ----
def panel_cues(ax, grid, oracle, frozen, mismatched, enc_c, hard_c):
    xs = np.array([0.0, 0.5, 1.0])
    ax.set_xlim(-0.30, 1.16)
    ax.set_ylim(-14.0, 112.0)
    for readout, dashes in (("soft", None), ("hard", (2.4, 1.8))):
        for size in (16, 64, 256):
            mean, lo, hi = grid[(readout, size)]
            colour = CAL_RAMP[size]
            ax.fill_between(xs, lo, hi, color=colour, alpha=0.22, lw=0.0,
                            zorder=1.6)
            line, = ax.plot(xs, mean, color=colour,
                            lw=LW_DATA if readout == "soft" else LW_ERR,
                            solid_capstyle="round", zorder=3.0)
            if dashes:
                line.set_dashes(dashes)
    # the eighteen printed grid means, the three reference rules and the
    # four contrast tags are all asserted (plan §9 item 9)
    for key, want in ((("soft", 16), [19.39, 19.16, 19.07]),
                      (("soft", 64), [75.06, 21.39, 19.30]),
                      (("soft", 256), [79.98, 25.83, 19.27]),
                      (("hard", 16), [80.29, 32.41, 20.81]),
                      (("hard", 64), [80.29, 54.22, 22.30]),
                      (("hard", 256), [80.29, 64.03, 23.44])):
        np.testing.assert_allclose(grid[key][0], want, atol=2e-2)
    np.testing.assert_allclose([oracle, frozen], [80.29, 18.67], atol=2e-2)
    np.testing.assert_allclose(mismatched, [18.58, 18.89], atol=2e-2)
    reference_line(ax, oracle, label=f"oracle {oracle:.1f} %",
                   color=PURPLE, span=(-0.30, 1.16))
    # CF-7 exception (recorded): the chance label sits at the LEFT end of
    # its rule, because the right end carries the two-line oracle-gap block
    # and the two would read as one stack (QA 2026-09-09)
    reference_line(ax, 50.0, label=None, span=(-0.30, 1.16))
    ax.text(-0.28, 52.0, "chance", fontsize=PT_SMALL, color=MUTE, ha="left",
            va="bottom", zorder=5)
    ax.plot([-0.30, 1.16], [frozen, frozen], color=MUTE, lw=LW_HAIR,
            zorder=1.2)
    # which family is which, and the calibration ramp keyed on the left ends
    ax.text(-0.28, 106.0, "soft readout, calibration cues (solid)",
            fontsize=PT_SMALL, color=INK, ha="left", va="center")
    ax.text(-0.28, 96.0, "hard readout, exploratory (dashed)",
            fontsize=PT_SMALL, color=MUTE, ha="left", va="center")
    _badge(ax, (1.16, 106.0), "exploratory", ha="right", va="center")
    v256 = encoder_contrast(enc_c, 256, 0.0, "oracle_context")
    v16 = encoder_contrast(enc_c, 16, 0.0, "oracle_context")
    prim = encoder_contrast(enc_c, 256, 0.5, "oracle_context")
    froz = encoder_contrast(enc_c, 256, 0.5, "frozen_profile")
    hard = encoder_contrast(hard_c, 256, 0.5, "oracle_context")
    np.testing.assert_allclose([v256[0], v16[0], prim[0], froz[0], hard[0]],
                               [-0.31, -60.90, -54.47, 7.16, -16.26],
                               atol=2e-2)
    # both noise-free oracle gaps: leakage is a small-calibration effect.
    # The line is capped at 120 pt so it clears the right-aligned
    # 'oracle 80.3 %' rule label 4 units below it.
    ax.text(-0.28, 86.0,
            f"noise 0: {_signed(v256[0])} pp (256), {_signed(v16[0])} (16)",
            fontsize=PT_SMALL, color=INK, ha="left", va="center")
    for size, y_tag, y_line in ((256, 73.0, 79.98), (64, 63.0, 75.06),
                                (16, 27.0, 19.39)):
        ax.text(-0.05, y_tag, str(size), color=CAL_RAMP[size],
                fontsize=PT_SMALL, ha="right", va="center")
        ax.plot([-0.04, 0.0], [y_tag, y_line], color=CAL_RAMP[size],
                lw=LW_HAIR, zorder=1)
    # the block sits left of the right edge and 4 pt above the chance label
    # so the two never read as one stack (QA 2026-09-09)
    ax.text(1.16, 72.0, f"{_signed(prim[0])} pp vs oracle", fontsize=PT_SMALL,
            color=INK, ha="right", va="center")
    ax.text(1.16, 64.0, f"{_signed(froz[0])} pp vs frozen", fontsize=PT_SMALL,
            color=INK, ha="right", va="center")
    # leader from the annotation block to the soft 256-cue vertex at noise
    # 0.5 (25.8 %), approaching from above-right (QA 2026-09-09)
    ax.plot([0.62, 0.515], [61.0, 28.6], color=MUTE, lw=LW_HAIR, zorder=1)
    ax.text(-0.28, 12.0, f"frozen {frozen:.1f} %; mismatched "
                         f"{mismatched[0]:.1f}–{mismatched[1]:.1f} %",
            fontsize=PT_SMALL, color=MUTE, ha="left", va="center")
    ax.set_xticks([0.0, 0.5, 1.0], ["0", "0.5", "1"])
    ax.set_yticks([20, 40, 60, 80])
    ax.spines["left"].set_bounds(20, 80)
    ax.spines["bottom"].set_bounds(0.0, 1.0)
    ax.set_xlabel("Cue noise SD")
    ax.set_ylabel("Held-out accuracy (%)")
    ax.text(1.16, 1.0, "n = 20 fresh seeds (52000–52019);",
            fontsize=PT_SMALL, color=MUTE, ha="right", va="center")
    ax.text(1.16, -8.5, "mean [95 % bootstrap]; epoch 80; cue delay 0",
            fontsize=PT_SMALL, color=MUTE, ha="right", va="center")
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
    # the factorial's own second bootstrap of the SAME contrast must still
    # agree on the mean; its ci95_high (1.97) is never printed (plan §4 E)
    second = pd.read_csv(DATA / "paired_contrasts.csv")
    hero = second[second.contrast.str.contains("best_matched", na=False)
                  & second.budget_k.eq(4)]
    if len(hero):
        np.testing.assert_allclose(float(hero.mean_difference.iloc[0]),
                                   0.012670898, atol=1e-6)

    canvas = NativeCanvas(490 / 72, 3, row_weights=[128, 107, 107],
                          hgutter_pt=38, vgutter_pt=48,
                          margins=Margins(left=44, right=12, top=20, bottom=32))
    a = canvas.panel("A", 0, 0, 4, title="Eight-stream task tree",
                     schematic=True, lock=False)
    b = canvas.panel("B", 0, 4, 8, title="Ancestry and control dictionaries",
                     schematic=True, lock=False)
    c = canvas.panel("C", 1, 0, 6, title="Ancestry wins only at K = 4")
    d = canvas.panel("D", 1, 6, 6, title="Rewiring removes the K = 2, 4 gain",
                     sharey=c)
    e = canvas.panel("E", 2, 0, 7, title="Ancestry minus each control, K = 4",
                     lock=False)
    fpan = canvas.panel("F", 2, 7, 5, title="Learned cues fall short of oracle",
                        lock=False)

    # 20 pt of declared left reserve on BOTH six-module panels of row 1:
    # audit_letter_alignment.py protects the 45.9 pt letter column, and C's
    # rotated y label reaches 22 pt left of its axes.  Declaring it on D as
    # well keeps the two six-module panels the same width, which the layout
    # contract requires.
    canvas.declare_reserve("C", left=20)
    canvas.declare_reserve("D", left=20)

    panel_task(a, tiers)
    raw = panel_dictionaries(b, prediction, supports)
    stats_c = panel_bandwidth(c, summary, outcomes)
    stats_d = panel_rewiring(d, summary, paired)
    stats_e, gutter = panel_forest(canvas, e, contrasts, pairs4)
    stats_f = panel_cues(fpan, grid, oracle, frozen, mismatched, enc_c, hard_c)

    style_direct_color_labels(canvas.fig)
    canvas.lock_reserves()
    findings = canvas.align_letters()
    problems = canvas.save(OUT, name="credit_first_figure_03", dpi=180,
                           lock=False)
    PUBLISHED.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(OUT, PUBLISHED)

    # ---- render-time records ------------------------------------------
    prediction.to_csv(RECORDS / "figure_03_coefficient_prediction.csv",
                      index=False)
    supports.to_csv(RECORDS / "figure_03_control_supports.csv", index=False)
    contrasts[contrasts.control.isin([k for _, k, _, _, _ in FOREST_ROWS])] \
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
    keys = [k for _, k, _, _, _ in FOREST_ROWS] + \
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
             "controls at K = 4 (figure_03_control_supports.csv, rng seed 0).",
        "C": "condition_summary.csv means with 95% seed-bootstrap intervals "
             "(dendritic_tree under correct ancestry and under derangement) "
             "and the per-seed post-training maximum over the four matched "
             "controls from seed_outcomes.csv.",
        "D": "condition_summary.csv means with 95% intervals for "
             "dendritic_tree versus degree_depth_matched_rewired_tree under "
             "ancestry feedback; paired differences and 20,000-draw seed "
             "bootstrap (seed 70000) from seed_outcomes.csv.",
        "E": "review_evidence_reanalysis/ancestry_k4_control_contrasts.csv "
             "means, 95% intervals, positive-seed counts and Holm-adjusted P; "
             "the seed fan is ancestry_control_paired_differences.csv at K = 4.",
        "F": "review_coefficient_encoder and review_coefficient_hard_readout "
             "condition_summary.csv over the calibration x cue-noise grid at "
             "cue delay 0, with render-time 95% seed-bootstrap bands from "
             "trajectories.csv at epoch 80; contrasts from both "
             "paired_contrasts.csv (the hard readout is exploratory).",
    }
    deviations = [
        "row weights 128/107/107 at a 48 pt vertical gutter and margins "
        "44/12/20/32, not 112/121/121 at 42 pt with margins 52/14 "
        "at 42 pt: at a 112 pt row 0 the eight-module panel B has axes aspect "
        "2.58, above PANEL_ASPECT_MAX 2.40, which DECISIONS G4 forbids "
        "relaxing, and a 42 pt gutter leaves only 6.5 pt between rows 1 and 2 "
        "against audit_row_separation.py's 8.5 pt floor",
        "E uses lock=False plus a measured private left inset instead of "
        "declare_reserve(left=66): a declared reserve locks the whole grid "
        "column and C starts in column 0, so C and D would differ in width",
        "E draws its per-row wins / Holm notes inside the axes above each "
        "seed fan; forest(note=...) writes into the gutter shared with F",
        "E prints Holm P in decimal (Nimbus Sans has no superscript minus and "
        "CF-2 forbids mathtext); the caption keeps the scientific form",
        "A sets the four coefficient tier labels in a keyed two-row band, not "
        "over their own blocks (a 17.6 pt coefficient string cannot sit over "
        "a one-terminal block at the terminal pitch)",
        "A drops the 'junctions define feedback supports only' footer into "
        "the caption -- the plan's stated fallback",
        "B draws no tree in the ancestry cards, so the subtree entry arrow "
        "comes from the private _delivery_arrow rather than credit_delivery",
        "B's control support table is 4 x 10 pt rows by 8 x 7 pt columns, not "
        "6 x 6 pt: a 7 pt row label cannot sit on a 6 pt row",
        "D's paired intervals use one 20,000-draw bootstrap seeded 70000; "
        "K = 2 gives [21.03, 25.32] against the plan's quoted [21.05, 25.32]",
        "C and F carry shortened forms of the plan's long annotations; the "
        "full wording is in the caption",
        "C's tie tags print 18.66 %, not the plan's 18.67 %: the frozen "
        "condition_summary gives 18.6646 % for both arms at K = 1, which is "
        "also what the plan's own deranged row rounds to",
        "E runs to xlim (-2, 14) instead of (-2, 11) so the +13.13 dense "
        "rank-4 seed is drawn rather than clipped; the off-scale note "
        "therefore names only the derangement contrast",
        "F does not repeat 256 / 64 / 16 on the dashed hard-readout lines: "
        "they carry the same three ORDINAL_RAMP hues as the labelled solid "
        "lines, and the noise-0.5 points have no 12 pt clearance for three "
        "more labels without a text-over-data collision",
        "F's noise-free annotation is one line, capped at 120 pt so it "
        "clears the right-aligned 'oracle 80.3 %' rule label four units "
        "below it; the caption says the printed values are oracle gaps",
        "B's control-table column headers are the digits 1..8 under one b_i "
        "tag, not eight subscripted b_j strings: a literal Unicode subscript "
        "is not in Nimbus Sans and mathtext is banned by CF-2",
        "B's |A| scale-bar tag is plain text, not Frame.subscript",
        "F's oracle rule is labelled 'oracle 80.3 %' with no 'oracle' badge "
        "(the panel already carries the 'exploratory' badge; a second chip "
        "on the same rule would sit on the dashed hard-readout curves)",
        "private helpers _badge (ceiling, exploratory), _delivery_arrow, "
        "_gap_span",
    ]
    schematic_fraction = (128.8 + 295.6) * 128.0 / (462.4 * 438.0)
    payload = dict(
        panel_sources=panels,
        source_sha256={str(p.relative_to(JOURNAL)):
                       hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in sources},
        layout_findings=list(findings) + list(problems),
        forest_gutter_pt=round(gutter, 2),
        schematic_area_fraction=round(schematic_fraction, 4),
        g4_waiver="G4 waiver recorded for Fig 3 at the superseded 31.6 % "
                  "measure; on the B12 formula the figure is "
                  f"{100 * schematic_fraction:.1f} % and the waiver is dormant",
        derived_numbers=dict(
            group_sums={int(k): float(v) for k, v in raw.items()},
            normalized_group_sums={int(k): float(v)
                                   for k, v in normalized.items()},
            accuracy_by_K=stats_c, rewiring_pp=stats_d,
            k4_contrasts_pp=stats_e, cue_grid=stats_f),
        coefficient_scope="Raw group sums; implemented route rows divide by "
                          "sqrt(group size). Signs follow from the generator "
                          "and are not a new prospective prediction.",
        caption_word_count=caption_words(),
        deviations_from_plan=deviations)
    assert 220 <= payload["caption_word_count"] <= 320, \
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
