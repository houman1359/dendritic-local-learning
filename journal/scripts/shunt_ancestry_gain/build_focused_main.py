#!/usr/bin/env python3
"""Main Figure 8 (``fig:focal``): focal shunts as ancestry-defined route gains.

Eight panels A--H on one 12-module :class:`NativeCanvas`, 518.4 x 490 pt,
rows 128 / 116 / 116 pt::

    A focal shunt vs matched injection (schematic, 6 mod)
    B one exact gain per ancestry block (schematic, 6 mod)
    C descendants change most   D contrast grows with dose   E adjoint replacement
    F opposite signed changes   G state sets selectivity     H background restores it

Built to ``analysis/figure_overhaul_20260908/v2/fig8/PLAN.md`` as amended by
``v2/AMENDMENTS.md`` and ruled by ``v2/DECISIONS.md``.  Nothing here recomputes
an estimate: every mean, interval and n comes from the frozen Source Data
tables through ``build_journal_figures.mean_ci`` at the archived seeds, and the
panel-B block gains are exact consequences of the frozen segment table and the
frozen unit dose (the identity ``q'/q = B eta`` is asserted at build time).

Cross-figure rules (AMENDMENTS section 3), with this figure's disposition
--------------------------------------------------------------------------
CF-1  canvas 518.4 x 490 pt, aspect 1.058, height on the 340/415/490 ladder.
CF-2  three type sizes only, 7.0 / 8.0 / 9.0-bold; no DejaVu; subscripts via
      ``Frame.subscript`` / ``token_subscript``, never mathtext.
CF-3  strokes 0.55/0.70/0.85/0.95/1.25 only; area marks are 16 % tints with a
      0.55 pt edge; no open stroke at or above DECORATIVE_LW_PT 1.35.
CF-4  glyph family: soma lowest and filled, one ink ``delta0`` arrow per soma
      (Fig 8 claims NO ``DELTA0_EXEMPTIONS`` entry -- both A cards and B's
      arbor draw one), shunt/inhibition is contact + badge, never a bar.
CF-5  zero legend boxes; the set's only sanctioned in-axes key is Fig 5C, and
      Fig 8 has none: every series is named by a direct label.
CF-6  forest idiom for C and F; the second arm is overplotted at +0.22 rows
      with an open marker.  CF-6 exception: Fig 8F overplots three additional
      arms per row at +0.14/+0.28/+0.42 instead of the set-wide +0.22, because
      each row carries four marks (shunt/injection x descendant/off-route).
CF-7  reference lines are dashed ``mute`` at LW_REF with the label
      right-aligned on the line.
CF-8  caption rules; the caption ships in v2/fig8/TEXT.md.
CF-9  sentence-case finding titles; ``Adjoint replacement`` (E) is one of the
      set's two recorded method-naming titles (the other is Fig 7F).
CF-10 schematic area = (210.2 + 210.2) x 128 / (457.4 x 442) = 26.6 %, cap
      30 %, no waiver.  The PLAN quotes 27.2 % on a 432 pt live height, i.e.
      on its bottom margin of 35 pt; this build carries bottom = 25 pt and
      vgutter = 41 pt instead (see the layout deviation below), so the live
      height is 442 pt and the fraction falls to 26.6 %.  The 29.6 % row
      basis stays the conservative bound.
CF-11 ``check_matrix_cells`` >= 6.0 pt for panel B's dictionary product.
CF-12 ``canvas.align_letters()`` is called unconditionally before ``save()``;
      no letter is hand-placed and no hand alignment loop exists.

Recorded waivers and deviations
-------------------------------
layout deviation from PLAN section "Canvas": the vertical gutter is 41 pt
and the bottom margin 25 pt, not 36 / 35.  The total is unchanged
(23 + 128 + 41 + 116 + 41 + 116 + 25 = 490) and every row height is the
plan's, but at vgutter 36 the r1|r2 boundary measures about 5 pt, below
``audit_row_separation``'s 8.5 pt (3 mm) floor, because rows 1 and 2 both
carry an x label under a 4-module axes.  At 41 pt it measures 10.1 pt.

waiver D3: row 2 (F signed forest / G state curve / H background rescue) is
three 4-module panels that share no axis; F is a signed log change, G and H a
shunt-minus-injection contrast on different manipulations, and the row is
column-locked.

deviation from specification D8: ``ORDINAL_RAMP`` is NOT used for the cohorts.
``ORDINAL_RAMP[3]`` is dE 8.6/5.0 from the grey injection series and
``ORDINAL_RAMP[1]`` dE 13.1/12.8 from ``shunting``; DECISIONS (Figure 8) rules
that cohorts are encoded by marker shape and fill -- filled diamond = initial
eight-cell cohort, open triangle = disjoint 45-cell calibration cohort.

deviation from PLAN section 4 F (2026-09-10 fix round): the two in-panel key
sentences of panel F are printed as ``shunt attenuates;`` / ``injection
enhances`` and ``filled = descendants,`` / ``open = off-route``, two adjacent
line pairs rather than the plan's ``shunt attenuates; matched injection
enhances``.  At 7.0 pt the full string is 86.8 pt wide in a 99.8 pt axes and
crosses the dashed zero rule and the near-zero marks of every row; the panel
names the injection in full nowhere else, and the caption's F sentence carries
``matched current injection``.  The row labels drop the plan's ``R_m 300``
string from the gutter text and set a chained ``R``+``m`` span per row instead
(the gutter cannot host mathtext and ``Rm`` beside panel H's ``R_m`` read as a
typo).  ``forest(tick=False)`` for F: the 6 % band already ties each label to
its row, and the alternative gutter hairline is drawn ON the left spine, where
it is the one data artist the grouped key block cannot clear.

deviation from PLAN section 3 B (2026-09-10 fix round): the five block gains
are set against the five bands of the ``q'/q`` column with NO leader.  A
3.2 pt mute hairline abutting a numeral at x-height is a minus sign at print
scale, so every attenuating positive multiplier read as a negative number.
The two value columns of ``dictionary_product`` are also recoloured off
``DIV_CMAP`` -- whose positive end is ``bp`` #932F1E, the exact-path/backprop
role colour -- onto an achromatic ``edge`` ramp, because no learning rule
appears in this figure and the gains are all positive.

deviation from PLAN section 3 B: the arbor is a real bipolar reconstruction,
so a few basal strokes fall below the soma disc; the in-plane rotation that
minimises that excursion is recorded as the ``arbor-orientation`` schematic
note.  The strokes are NOT clipped: the panel's claim is the exact ancestry
partition of this cell, and deleting members of the soma-side block to satisfy
a reading convention would misstate it.

deviation from ``journal_style.K_CYCLE`` for panel B's partition blocks:
``pathway`` (descendants) and ``local`` (sister blocks) are declared through
``require_address_tint`` so the reason is recorded in the canvas manifest.

Private helpers (library frozen for per-figure work, errata 7; DECISIONS G5)
----------------------------------------------------------------------------
``_arbor_blocks``     ``Frame.arbor(mode='blocks')`` does not exist upstream;
                      this is the ancestry partition drawn on the real cell,
                      with an in-plane rotation that puts the soma as low as
                      the reconstruction allows.
``_formula``          a multi-subscript chain (``Frame.subscript`` carries one
                      subscript per call).
``_tint``             ``pct`` % of a colour over white; the library exports no
                      mix helper.
``_crosses``/``_nearest``  leader anchoring for panel B's three block tags:
                      the landing point is the member of the tagged block
                      whose straight leader crosses the fewest strokes of
                      other blocks and lands farthest from them.
``place_cohort_markers``  panel G's cohort key glyph, deferred until after
                      ``lock_reserves()`` because the axes is resized between
                      the panel draw and the save.
``panel_adjoint`` bracket  the +0.079 paired contrast is q\u2032d\u2032
                      minus d\u2032; it is drawn as a bracket over exactly
                      those two columns because the number, printed loose,
                      read as the q\u2032 column (0.130) or as
                      q\u2032 minus inject (0.095, also the printed bound).

``_balance_slot_fill``  declares the per-column reserves that keep the
                      ``panel-emphasis`` slot-fill spread inside 1.35x once the
                      two forest panels have claimed their row-label gutter.
"""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
J = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(J / "scripts/credit_first_figures"))
sys.path.insert(0, str(J / "code/reconstructed_tree"))

import build_figure as previous                       # noqa: E402
from focused_provenance import publish                # noqa: E402
from figure_canvas import (NativeCanvas, Margins, COLORS, PT_BASE, PT_EMPH,   # noqa: E402
                           LW_HAIR, LW_EDGE, LW_REF, LW_ERR, LW_DATA,
                           MARKER_MS, SEED_MS, SEED_ALPHA, ERR_CAPSIZE,
                           token_subscript, forest)
from journal_style import (style_direct_color_labels, label_color)  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, to_rgb  # noqa: E402
from native_schematics import (Frame, reference_line, check_matrix_cells,     # noqa: E402
                               collapsed_row_note, require_address_tint,
                               CONTACT_DIA_PT, _text_w_pt)
from build_journal_figures import mean_ci             # noqa: E402
import build_main_figure_07 as f7                     # noqa: E402

ancestry_gains = previous.ancestry_gains
_formula = previous._formula
_lerp = previous._lerp

INK = COLORS["ink"]
MUTE = COLORS["mute"]
EDGE = COLORS["edge"]
SHUNT = COLORS["shunting"]        # focal shunt
INJECT = COLORS["point_mlp"]      # matched current injection (never additive)
BLOCK_DESC = COLORS["pathway"]    # descendant block, Fig 8B only
BLOCK_SIS = COLORS["local"]       # sister blocks, Fig 8B only
BLOCK_SOMA = COLORS["dend"]       # soma-side block

M_SHUNT, M_INJECT = "o", "s"
M_INITIAL, M_DISJOINT = "D", "^"
MEAN_MS = MARKER_MS + 1.2

MEDIAN_ROOT = previous.MEDIAN_ROOT          # 864691135409937097
FOCAL_SEGMENT = previous.FOCAL_SEGMENT      # 4227

CATEGORIES = ("descendant", "sister", "ancestor", "depth-matched unrelated",
              "unrelated")
CATEGORY_LABELS = ("descendant", "sister", "ancestor", "depth-\nmatched",
                   "unrelated")

# 2026-09-11 fix round: the pair used to run to 0.335 with D's highest drawn
# point at 0.230 and E's at 0.190, i.e. 30 % / 41 % of the ordinate empty and
# filled with prose.  The top now clears D's highest PER-CELL point (0.2572,
# drawn from this round on) and E's paired-contrast bracket, and nothing else.
LOCAL_YLIM = (-0.014, 0.286)
LOCAL_YTICKS = (0.0, 0.05, 0.10, 0.15, 0.20, 0.25)
LOCAL_TICKLABELS = ("0", "0.05", "0.10", "0.15", "0.20", "0.25")
LOCAL_LABEL = "Localization index (log units)"      # sentence case, 2026-09-23
CONTRAST_LABEL = "Shunt − injection (log units)"

# review pass 2026-09-23: cohort sizes and axial/leak ranges are in the legend
COHORTS = (("original_eight", "Initial", M_INITIAL, True),
           ("v661_disjoint", "Disjoint", M_DISJOINT, False))
RM_ORDER = ("Ra150_Rm300", "Ra150_Rm1000", "Ra150_Rm3000", "Ra150_Rm5000",
            "Ra150_Rm15000", "Ra150_Rm30000")


def _tint(color, pct):
    """``pct`` % of ``color`` over white (private: the library has no mix)."""
    r, g, b = to_rgb(color)
    k = pct / 100.0
    return (1.0 - k + k * r, 1.0 - k + k * g, 1.0 - k + k * b)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _num(fmt, value):
    return fmt.format(value).replace("-", "−")


def _rm(regime):
    return int(str(regime).split("Rm")[1])


def _rm_text(regime):
    return f"{_rm(regime):,}"


# ── frozen-table loaders (every column verified against the shipped CSV) ──
def load_categories():
    """Unit-dose relation effects, cell means of within-cell site medians."""
    table = pd.read_csv(J / "source_data/figure4/category_effects.csv")
    table = table[np.isclose(table.dose, 1.0)]
    out = {}
    for perturbation in ("focal shunt", "matched additive"):
        cells = table[table.perturbation.eq(perturbation)].groupby(
            ["root_id", "category"],
            as_index=False).median_abs_log_gradient_change.mean()
        for index, relation in enumerate(CATEGORIES):
            values = cells[cells.category.eq(relation)].set_index("root_id") \
                .median_abs_log_gradient_change.astype(float)
            assert values.size == 8, (perturbation, relation, values.size)
            out[(perturbation, relation)] = (
                values, mean_ci(values, seed=1610 + index))
    unrelated = out[("matched additive", "unrelated")][0]
    assert np.all(np.abs(unrelated) < 1e-9), \
        "injection 'unrelated' changes are no longer structurally zero"
    return out


def load_dose():
    """Localization against normalized dose, equal-weighted cell means."""
    focal = pd.read_csv(J / "source_data/figure4/focal_localization.csv")
    per_cell = focal.groupby(["root_id", "dose", "perturbation"],
                             as_index=False).localization_index.mean()
    out = {}
    for pindex, perturbation in enumerate(("matched additive", "focal shunt")):
        subset = per_cell[per_cell.perturbation.eq(perturbation)]
        rows = []
        for dose, group in subset.groupby("dose"):
            values = group.set_index("root_id").localization_index.astype(float)
            assert values.size == 8, (perturbation, dose, values.size)
            rows.append((float(dose),
                         *mean_ci(values, seed=1660 + 10 * pindex
                                  + int(dose * 4)), values))
        out[perturbation] = sorted(rows)
    return out


def load_shapley():
    """The four factor-substitution corners, paired within cell."""
    table = pd.read_csv(J / "source_data/focal_decomposition/cell_shapley.csv")
    table = table[table.estimand.eq(
        "full_shunt_minus_matched_additive")].sort_values("root_id")
    assert len(table) == 8, len(table)
    assert np.allclose(table.reference_localization,
                       table.matched_additive_localization)
    columns = ("matched_additive_localization",
               "driving_force_only_localization",
               "adjoint_only_localization",
               "full_shunt_localization")
    out = {c: table[c].to_numpy(float) for c in columns}
    out["root_id"] = table.root_id.to_numpy()
    out["stats"] = {c: mean_ci(out[c], seed=1710 + i)
                    for i, c in enumerate(columns)}
    paired = out["full_shunt_localization"] - \
        out["driving_force_only_localization"]
    out["replacement"] = mean_ci(paired, seed=1720)
    out["replacement_positive"] = int((paired > 0).sum())
    return out


def load_signed():
    """Signed log changes at the two physical calibrations, both cohorts."""
    root = J / "source_data/shunt_ancestry_gain/signed_calibration"
    summary = pd.read_csv(root / "signed_cohort_summary.csv")
    assert len(summary) == 16, len(summary)
    cells = pd.read_csv(root / "signed_cell_effects.csv")
    return summary, cells


def load_physical():
    """Shunt-minus-injection contrast against membrane resistance."""
    base = J / "source_data/physical_cable_sensitivity"
    physical = pd.read_csv(base / "cell_primary_contrasts.csv")
    ratio = pd.read_csv(base / "cell_electrotonic_ratios.csv")
    keys = physical[["cohort", "regime", "root_id"]].drop_duplicates()
    ratio = ratio.merge(keys, on=["cohort", "regime", "root_id"],
                        validate="one_to_one")
    assert len(ratio) == len(keys), (len(ratio), len(keys))
    median = ratio.groupby(["cohort", "regime"],
                           as_index=False).median_axial_to_leak_ratio.median()
    disjoint = median[median.cohort.eq("v661_disjoint")] \
        .sort_values("median_axial_to_leak_ratio") \
        .median_axial_to_leak_ratio.to_numpy(float)
    assert np.allclose(disjoint, [2.659141, 26.591409, 132.957044], atol=1e-4), \
        f"disjoint cohort medians are the 47-cell values: {disjoint}"
    out = {}
    for cohort, _label, _marker, _filled in COHORTS:
        subset = physical[physical.cohort.eq(cohort)
                          & physical.regime.str.startswith("Ra150_")]
        points = []
        for index, regime in enumerate(RM_ORDER):
            group = subset[subset.regime.eq(regime)]
            if group.empty:
                continue
            values = group.set_index("root_id").difference.astype(float)
            points.append((_rm(regime), *mean_ci(values, seed=1740 + index),
                           int(values.size), values))
        out[cohort] = points
    ranges = {c: (float(median[(median.cohort.eq(c))
                               & median.regime.str.startswith("Ra150_")]
                        .median_axial_to_leak_ratio.min()),
                  float(median[(median.cohort.eq(c))
                               & median.regime.str.startswith("Ra150_")]
                        .median_axial_to_leak_ratio.max()))
              for c, _l, _m, _f in COHORTS}
    return out, ranges


def load_background():
    """The background-conductance rescue at the standard calibration."""
    base = J / "source_data/focal_selectivity_phase1"
    paired = pd.read_csv(base / "paired_contrasts.csv")
    sel = paired[paired.membrane_resistance_ohm_cm2.eq(15000)
                 & paired.dose_scheme.eq("input_conductance_normalized")
                 & np.isclose(paired.dose_value, 1.0)
                 & paired.metric.eq("localization_index")] \
        .sort_values("background_leak_multiplier")
    assert list(sel.background_leak_multiplier) == [0, 1, 4], \
        list(sel.background_leak_multiplier)
    assert list(sel.cells_positive) == [4, 8, 8], list(sel.cells_positive)
    cells = pd.read_csv(base / "cell_condition_metrics.csv")
    cells = cells[cells.membrane_resistance_ohm_cm2.eq(15000)
                  & cells.dose_scheme.eq("input_conductance_normalized")
                  & np.isclose(cells.dose_value, 1.0)]
    wide = cells.pivot_table(index=["root_id", "background_leak_multiplier"],
                             columns="perturbation",
                             values="localization_index").reset_index()
    wide["difference"] = wide["focal shunt"] - wide["matched additive"]
    per_cell = {int(m): g.set_index("root_id").difference.astype(float)
                for m, g in wide.groupby("background_leak_multiplier")}
    for m in (0, 1, 4):
        assert per_cell[m].size == 8, (m, per_cell[m].size)
    frozen = sel.mean_shunt_minus_additive.to_numpy(float)
    assert np.allclose(frozen, [per_cell[m].mean() for m in (0, 1, 4)],
                       atol=1e-6)
    return sel, per_cell


# ── private glyph helper: the ancestry partition on the real arbor ───────
def _orient_arbor(positions, soma_id):
    """Rotate the SVD projection in plane so the soma sits as low as it can.

    The projection plane is fixed by ``morphology_geometry``; the rotation
    inside it is free, so it is chosen to minimise how far the reconstruction
    reaches below the soma.  This cell is bipolar (apical plus basal), so the
    soma cannot be the strictly lowest node of a faithful drawing; the
    residual basal excursion is recorded as a schematic note.
    """
    keys = list(positions)
    xy = np.asarray([positions[k] for k in keys], dtype=float)
    soma = np.asarray(positions[soma_id], dtype=float)
    best, best_cost = 0.0, None
    for degrees in range(0, 360, 1):
        theta = np.deg2rad(degrees)
        rot = np.array([[np.cos(theta), -np.sin(theta)],
                        [np.sin(theta), np.cos(theta)]])
        turned = (xy - soma) @ rot.T
        cost = float(np.clip(-turned[:, 1], 0.0, None).sum())
        if best_cost is None or cost < best_cost:
            best, best_cost = float(theta), cost
    rot = np.array([[np.cos(best), -np.sin(best)],
                    [np.sin(best), np.cos(best)]])
    turned = (xy - soma) @ rot.T
    below = int((turned[:, 1] < -1e-9).sum())
    return ({k: tuple(p) for k, p in zip(keys, turned)},
            np.degrees(best), below)


def _arbor_blocks(f, rect, gains, colours, weights):
    """Private ``Frame.arbor(mode='blocks')``: the partition on the real cell.

    Returns the placement map, the block anchors and the scale-bar geometry so
    the caller can lead its tags in from outside the canopy.
    """
    cell, positions, rows, parent, span_um = f7.morphology_geometry()
    assert int(cell.root_id.iloc[0]) == MEDIAN_ROOT, "median cell moved"
    assert len(rows) == 78, len(rows)
    soma_id = min(rows, key=lambda key: rows[key].topological_depth)
    positions, degrees, below = _orient_arbor(positions, soma_id)
    f.note("arbor-orientation", panel="B", rotation_deg=round(degrees, 1),
           nodes_below_soma=below,
           reason="real bipolar reconstruction: the in-plane rotation that "
                  "minimises the basal excursion below the soma")
    block, path = gains["block"], gains["path"]
    xy = np.asarray([positions[k] for k in rows], dtype=float)
    to_axes = f7.fit_isotropic(xy, rect, f.w_pt, f.h_pt, pad_pt=2.0)
    place = {key: tuple(to_axes(point)[0]) for key, point in positions.items()}
    order = list(reversed(path))                 # focal .. soma
    strokes = []
    for segment in rows:
        parent_id = parent[segment]
        if parent_id not in rows:
            continue
        home = block[segment]
        start, end = place[segment], place[parent_id]
        strokes.append((home, start, end))
        f.ax.plot([start[0], end[0]], [start[1], end[1]],
                  color=colours[home], lw=weights[home],
                  solid_capstyle="round", zorder=2.4)
    f.soma(place[soma_id])
    f.error_in(place[soma_id], side="left")
    f.shunt(place[FOCAL_SEGMENT], label=None)
    anchors = {}
    for node in order:
        members = [s for s in rows if block[s] == node]
        pts = np.asarray([place[s] for s in members], dtype=float)
        anchors[node] = (float(np.median(pts[:, 0])), float(pts[:, 1].min()),
                         float(pts[:, 1].max()))
    bar = to_axes(np.vstack([xy.min(axis=0),
                             xy.min(axis=0) + [50.0 / span_um, 0.0]]))
    return place, anchors, float(bar[1, 0] - bar[0, 0]), soma_id, strokes


# ── panel A: the two state-matched interventions ─────────────────────────
def panel_interventions(ax):
    """Focal shunt versus the current injection that matches its drive."""
    f = Frame(ax)
    # Review pass 2026-09-23: the card footers (`soma V restored`), the two
    # prose tag lines, the `control` badge and the `baseline focal current
    # matched` bracket are legend A's sentences; each card keeps its title,
    # its formula and its glyphs, and the cards take the full height.
    cards = f.split(2, axis="x", gap_pt=9.0, pad_pt=(0.0, 0.0, 0.0, 0.0))
    for cell, title, hero in ((cards[0], "focal shunt", True),
                              (cards[1], "matched injection", False)):
        core = f.task_card(cell, title=title, footer=None,
                           emphasis=hero, tone=None if hero else "control")
        body = Frame.inset(core, left=0.07, right=0.07, top=0.02)
        # 12 pt under the soma for the two arrows and the delta0 tag, 20 pt
        # over the canopy for the two-line intervention tag: both cards keep
        # identical geometry.
        body = (body[0], body[1] + f.fy(12.0), body[2],
                body[3] - f.fy(12.0 + 12.0))
        nodes = f.balanced_tree(body, depth=3, trunk=False, mode="forward",
                                output="z")
        site = _lerp(nodes["JL"], nodes["JLL"], 0.80)
        tag_y = body[1] + body[3] + f.fy(6.5)
        line_y = tag_y + f.fy(10.0)
        if hero:
            f.shunt(site, label=None)
            # the ACTIVE shunt contact keeps the inhibitory rim: a bare
            # filled disc is the excitatory glyph at 3.6 pt, so at print
            # scale the shunt site was indistinguishable from a synapse
            f.ax.plot([site[0]], [site[1]], marker="o", linestyle="none",
                      ms=f.ms(CONTACT_DIA_PT * 2.05), mfc="none",
                      mec=COLORS["inh"], mew=f.lw(LW_HAIR), zorder=4.6)
            # the badge sits INSIDE the card, in the clear band between the
            # shunted branch and the trunk (right-aligned off the contact it
            # ran 8 pt past the card's left frame and printed its "g" in the
            # panel gutter), and its leader runs the whole way from the
            # contact rim to the label's top edge
            tag = (body[0], site[1] - f.fy(11.0))
            f.subscript(tag, "g", "shunt", size=PT_BASE,
                        color=COLORS["inh"], ha="left", va="top")
            f.leader((site[0], site[1] - f.fy(3.9)),
                     (site[0] - f.fx(1.0), tag[1] + f.fy(0.8)), color=MUTE)
            f.fade(["JLL"], nodes=nodes)
            f.fade([(nodes["JL"], nodes["JLL"])])
            _formula(f, (body[0], tag_y),
                     [("γ", "i"), " = ", ("q", "i"), "(", ("E", "E"), " − ",
                      ("V", "i"), ")"], size=PT_BASE, color=INK, ha="left")
        else:
            f.contact(site, kind="inh", active=False)
            # the two card-2 tag lines are ordered like card 1's: the prose
            # line above, the formula below, so the arrow into the contact
            # leaves the LOWER line and no longer crosses the prose
            width = _formula(f, (body[0], tag_y),
                             ["κ(", ("E", "I"), " − ", ("V", "k"), ")"],
                             size=PT_BASE, color=MUTE, ha="left")
            f.arrow((body[0] + f.fx(width + 3.0), tag_y),
                    (site[0] - f.fx(1.6), site[1] + f.fy(2.2)),
                    color=MUTE, lw=LW_EDGE, head=3.4, rad=-0.18)
        f.error_in(nodes.soma, side="right")
        drive = (nodes.soma[0] - f.fx(15.0), nodes.soma[1] - f.fy(3.5))
        f.arrow(drive, (nodes.soma[0] - f.fx(3.0), nodes.soma[1]),
                color=MUTE, lw=LW_EDGE, head=3.4)
        f.subscript((drive[0] - f.fx(1.2), drive[1]), "I", "soma",
                    size=PT_BASE, color=MUTE, ha="right", va="center")
    f.require_soma_lowest()
    f.require_delta0()
    return ax


# ── panel B: one exact gain per ancestry block ───────────────────────────
def panel_gain_dictionary(ax, gains):
    """The ancestry partition of the median cell and its exact gain column."""
    f = Frame(ax)
    # Review pass 2026-09-23: the footer, the root identifier, the block
    # counts, the two formula lines, the collapse note and the kappa /
    # identity-residual line are legend B (or Source Data) material.
    band = 0.0
    path, order = gains["path"], list(reversed(gains["path"]))
    require_address_tint("pathway", where="Fig 8B arbor block")
    require_address_tint("local", where="Fig 8B arbor block")
    colours = {order[0]: BLOCK_DESC, order[1]: BLOCK_SIS,
               order[2]: BLOCK_SIS, order[3]: BLOCK_SIS,
               path[0]: BLOCK_SOMA}
    weights = {order[0]: LW_DATA, order[1]: LW_ERR, order[2]: LW_REF,
               order[3]: LW_EDGE, path[0]: LW_HAIR}
    top = 1.0 - f.fy(5.0)
    key_pt = 21.0
    arbor_rect = (0.0, f.fy(band + key_pt), 0.43,
                  1.0 - f.fy(band + key_pt + 12.0))
    place, anchors, bar_w, soma_id, strokes = _arbor_blocks(
        f, arbor_rect, gains, colours, weights)

    # the g_shunt badge is set FIRST, in the clear whitespace up-left of the
    # focal contact.  At the lower right it was 6 pt below the soma disc, so
    # its leader crossed the disc and read as naming the soma, and the sister
    # tag's leader ran through the label itself.  Its box is seeded into
    # ``drawn`` so every later leader treats it as ink and routes around it.
    drawn = []
    focal = place[FOCAL_SEGMENT]
    shunt_tag = (focal[0] - f.fx(15.6), focal[1] - f.fy(4.6))
    f.leader((shunt_tag[0] + f.fx(1.0), shunt_tag[1]),
             (focal[0] - f.fx(2.0), focal[1] - f.fy(0.5)), color=MUTE)
    f.subscript(shunt_tag, "g", "shunt", size=PT_BASE,
                color=COLORS["inh"], ha="right", va="center")
    _bx0, _bx1 = shunt_tag[0] - f.fx(22.5), shunt_tag[0] + f.fx(1.0)
    _by0, _by1 = shunt_tag[1] - f.fy(5.5), shunt_tag[1] + f.fy(5.5)
    _corners = ((_bx0, _by0), (_bx1, _by0), (_bx1, _by1), (_bx0, _by1))
    for _i in range(4):
        drawn.append((_corners[_i], _corners[(_i + 1) % 4]))

    # block tags: each one is set OUTSIDE the canopy and joined to the
    # NEAREST member of its own block, so no leader crosses the arbor (v1
    # led every tag to the block's median-x / minimum-y member, which drew
    # two diagonals across the whole projection and over the scale bar)
    block_of = gains["block"]
    members = {}
    for segment in gains["ids"]:
        members.setdefault(block_of[segment], []).append(place[segment])

    def _crosses(p0, p1, q0, q1):
        """True when the open segments p0p1 and q0q1 intersect."""
        def side(a, b, c):
            return ((b[0] - a[0]) * (c[1] - a[1])
                    - (b[1] - a[1]) * (c[0] - a[0]))
        d1, d2 = side(q0, q1, p0), side(q0, q1, p1)
        d3, d4 = side(p0, p1, q0), side(p0, p1, q1)
        return (d1 * d2 < 0.0) and (d3 * d4 < 0.0)

    def _nearest(nodes, origin):
        """The block member a straight leader reaches WITHOUT crossing ink.

        Nearest-member re-anchoring alone (v1's fix for the crossed leaders)
        landed two of the three tags inside the dense grey/orange tangle
        beside the soma, where no single member is identified.  The candidate
        is now scored by how many strokes of OTHER blocks its leader would
        cross, with length only as the tie-break, so each tag ends on a
        member of its own block that is isolated from the tag's point of
        view -- the plan's "enter each block from outside the canopy".
        """
        keep = set(nodes)
        foreign = [(a, b) for home, a, b in strokes if home not in keep] \
            + drawn
        best = None
        for node in nodes:
            for point in members[node]:
                shrunk = (origin[0] + 0.90 * (point[0] - origin[0]),
                          origin[1] + 0.90 * (point[1] - origin[1]))
                hits = sum(1 for a, b in foreign
                           if _crosses(origin, shrunk, a, b))
                length = np.hypot((point[0] - origin[0]) * f.w_pt,
                                  (point[1] - origin[1]) * f.h_pt)
                # how ISOLATED the landing point is: a tag that ends inside
                # the grey/orange tangle beside the soma identifies nothing
                clear = min([np.hypot((point[0] - q[0]) * f.w_pt,
                                      (point[1] - q[1]) * f.h_pt)
                             for a, b in foreign for q in (a, b)] or [20.0])
                score = 240.0 * hits + length - 6.0 * min(clear, 14.0)
                if best is None or score < best[0]:
                    best = (score, point)
        drawn.append((origin, tuple(best[1])))
        return tuple(best[1])

    desc_y = top
    # sister blocks ABOVE soma side: their landing points sit that way round,
    # so the reversed tag order made the two leaders cross each other
    tags = (("descendants", BLOCK_DESC, desc_y, [order[0]]),
            ("sister blocks", BLOCK_SIS,
             f.fy(band + key_pt - 9.5), list(order[1:4])),
            ("soma side", BLOCK_SOMA,
             f.fy(band + key_pt - 18.5), [path[0]]))
    placed = []
    for text, hue, y_tag, nodes in tags:
        artist = f.text((0.0, y_tag), text, size=PT_BASE,
                        color=label_color(hue), ha="left", va="center")
        placed.append((artist, y_tag, nodes))
    ax.figure.canvas.draw()
    # ONE turn x for all three leaders (the right edge of the longest tag):
    # letting each diagonal start at its own tag's edge ran the shorter tags'
    # leaders straight through the longer tags' text.  Each leader is
    # therefore an ELBOW -- a horizontal stub along its own tag's baseline out
    # to the shared turn, then the diagonal into the block.  Without the stub
    # the two shorter tags' leaders began 9.1 pt and 26.2 pt clear of their
    # own labels and read as two more arbor strokes hanging in the white
    # space under the canopy.
    turn_x = max(a.get_window_extent().transformed(ax.transData.inverted()).x1
                 for a, _y, _n in placed) + f.fx(4.0)
    for artist, y_tag, nodes in placed:
        own_x = artist.get_window_extent() \
            .transformed(ax.transData.inverted()).x1 + f.fx(2.0)
        start = (turn_x, y_tag)
        if own_x < turn_x - f.fx(0.4):
            f.leader((own_x, y_tag), start, color=MUTE)
        f.leader(start, _nearest(nodes, start), color=MUTE)

    # scale bar at the lower LEFT of the projection (PLAN section 3 B)
    x_bar = arbor_rect[0] + f.fx(1.0)
    y_bar = arbor_rect[1] + f.fy(2.0)
    f.ax.plot([x_bar, x_bar + bar_w], [y_bar, y_bar], color=INK, lw=LW_DATA,
              solid_capstyle="butt", zorder=7)
    f.text((x_bar + bar_w / 2.0, y_bar + f.fy(2.0)), "50 µm", size=PT_BASE,
           color=INK, va="bottom")

    # the gain product: B (78 x 5 indicators) x eta = q'/q
    ids = gains["ids"]
    block_of = gains["block"]
    column_of = {node: i for i, node in enumerate(order)}
    ordered = sorted(ids, key=lambda s: (column_of[block_of[s]], s))
    indicators = np.zeros((len(ordered), 5))
    for row, segment in enumerate(ordered):
        indicators[row, column_of[block_of[segment]]] = 1.0
    groups = [int(indicators[:, j].sum()) for j in range(5)]
    assert groups == [17, 5, 4, 6, 46], groups
    eta = np.asarray([gains["gain"][node] for node in order])
    hues = [BLOCK_DESC, BLOCK_SIS, BLOCK_SIS, BLOCK_SIS, BLOCK_SOMA]

    # Draw the baseline-state superscript with ordinary font glyphs, at
    # the 7 pt floor, rather than unsupported Unicode superscript brackets.
    prod_rect = (0.46, f.fy(band + 8.0), 0.40,
                 1.0 - f.fy(band + 8.0 + 16.0))
    check_matrix_cells(prod_rect[2] * f.w_pt, prod_rect[3] * f.h_pt, 5, 5,
                       where="Fig 8B dictionary_product")
    axes = f.dictionary_product(prod_rect, indicators, eta, cell_pt=9.0,
                                col_colors=hues, row_groups=groups,
                                captions=("B", "\u03b7", "q\u2032/q"),
                                numbers=False, collapse="auto")
    # The library defaults the two value columns to DIV_CMAP, whose positive
    # end is `bp` #932F1E -- the backprop / exact-path role colour, which has
    # no meaning in a passive-cable schematic.  Every gain here is a positive
    # multiplier in [0.958, 0.998], so a diverging map is wrong twice over.
    # Both columns are recoloured onto an achromatic edge ramp (private to
    # this builder; the library is not edited, errata #7).
    gain_cmap = LinearSegmentedColormap.from_list(
        "fig8_gain", ["white", _tint(COLORS["edge"], 55), COLORS["edge"]])
    for inner in axes[1:]:
        for image in inner.images:
            image.set_cmap(gain_cmap)
            image.set_clim(0.94, 1.005)
    # the library centres its collapse note on the B matrix; at 6 modules a
    # 37-character line centred there runs left over the arbor and the scale
    # bar, so it is re-anchored right, under the product (the string itself
    # is the library's, ``collapsed_row_note([17, 5, 4, 6, 46])``)
    expected = collapsed_row_note(groups)
    for artist in list(ax.texts):
        if artist.get_text() == expected:
            artist.remove()          # legend B gives the band sizes
    host, box = ax.get_position(), axes[2].get_position()
    x_right = (box.x1 - host.x0) / host.width
    y_top = (box.y1 - host.y0) / host.height
    y_bottom = (box.y0 - host.y0) / host.height
    # NO leaders: a 3.2 pt mute hairline abutting a numeral at x-height is a
    # minus sign at print scale, and every one of the five attenuating gains
    # read as negative.  The five bands of the q'/q column are already
    # vertically registered, so the values are simply set against them.
    for j in range(5):
        y = y_top - (j + 0.5) / 5.0 * (y_top - y_bottom)
        f.text((x_right + f.fx(2.8), y), f"{eta[j]:.3f}", size=PT_BASE,
               color=INK, ha="left", va="center")
    # Gamma_u printed on the gain column (AMENDMENTS B10)
    gamma_y = y_top + f.fy(7.0)
    f.subscript((x_right - f.fx(0.5), gamma_y), "\u0393", "u", size=PT_BASE,
                color=INK, ha="right", va="center")
    # "1e-17" is calculator notation; mathtext is banned (CF-2), so the
    # exponent is a second span chained RIGHT to LEFT off a right-aligned
    # tail, the same construction panel H uses for R_m.
    f.require_soma_lowest()
    f.require_delta0()
    return ax


# ── panel C: relation forest ─────────────────────────────────────────────
def panel_relations(canvas, ax, category):
    """Unit-dose change by relation to the site, shunt against injection."""
    rows = []
    for label, relation in zip(CATEGORY_LABELS, CATEGORIES):
        values, (mean, lo, hi) = category[("focal shunt", relation)]
        rows.append(dict(label=label, mean=mean, lo=lo, hi=hi,
                         seeds=list(values), n=8))
    # the reference label is drawn here, not by forest(): right-aligned above
    # the top spine it lands on the panel title, because the zero rule sits
    # 4 pt from the left edge and the centred title starts 8 pt from it
    out = forest(ax, rows, value_label="|Δ log |γ|| (log units)",
                 reference=0.0, reference_label="", color="shunting",
                 xlim=(-0.006, 0.158), tag="")
    # CF-7 wants the reference label right-aligned ON its line.  The rule is
    # 3.6 pt from the left spine, so the label necessarily runs back into the
    # row-label gutter; at the TOP of the rule it landed 2.8 pt above
    # `descendant` and the gutter read as a six-item stack whose first entry
    # was `no change`.  It moves to the BOTTOM of the rule instead, hard
    # against the abscissa as in G and H, where the nearest row label is
    # 13 pt away and the x tick row reads it as the axis annotation it is.
    # The 0.32 of extra ordinate below the last row is what opens that strip;
    # it costs 0.3 pt of the +0.22 overplot offset and nothing else.
    ax.set_ylim(4.62, -1.02)
    # ... and INSIDE the axes, right of the rule, as F, G and H set theirs.
    # Right-aligned in the gutter it was set in the same face and tone as
    # `unrelated` directly above it and read as a sixth, empty relation row.
    # (review pass 2026-09-23: the zero rule is unlabelled)
    ax.tick_params(axis="x", labelsize=PT_BASE, pad=0.8, length=2.0)
    ax.xaxis.labelpad = 0.5
    ypos = out["ypos"]
    for index, relation in enumerate(CATEGORIES):
        values, (mean, lo, hi) = category[("matched additive", relation)]
        y = ypos[index] + 0.22
        ax.plot(values, np.full(values.size, y), linestyle="none", marker="o",
                markersize=SEED_MS, markerfacecolor=INJECT,
                markeredgecolor="none", alpha=SEED_ALPHA, zorder=2.0)
        ax.plot([lo, hi], [y, y], color=INJECT, lw=LW_ERR, zorder=3.0,
                solid_capstyle="butt")
        for bound in (lo, hi):
            ax.plot([bound, bound], [y - 0.10, y + 0.10], color=INJECT,
                    lw=LW_ERR, zorder=3.0)
        ax.plot([mean], [y], linestyle="none", marker=M_INJECT,
                markersize=MEAN_MS, markerfacecolor="none",
                markeredgecolor=INJECT, markeredgewidth=LW_ERR, zorder=4.0)
    shunt_d = category[("focal shunt", "descendant")][1][0]
    inject_d = category[("matched additive", "descendant")][1][0]
    # review pass 2026-09-23: the two series names stack above the first
    # row, as in D; between rows the leader-labelled name sat on a band
    ax.text(0.156, -0.99, "focal shunt", fontsize=PT_BASE, color=SHUNT,
            ha="right", va="bottom", zorder=6)
    ax.text(0.156, -0.50, "matched injection", fontsize=PT_BASE,
            color=INJECT, ha="right", va="bottom", zorder=6)
    # Design pass 2026-09-14: the `shunt 3.1x injection` tag is gone (the
    # running text says the injection changed descendant gradients roughly
    # threefold less); the ratio is held to the source here instead.
    assert 2.9 < shunt_d / inject_d < 3.3, shunt_d / inject_d
    # 2026-09-11 fix round: EIGHT further lines of prose stood here -- the
    # three-line stat block (`n = 8 cells` / `101 focal sites` /
    # `mean [95 % CI]`), the two-line `injection off-route: zero by
    # construction` tag with its leader, and the three-line `45/45 cells
    # positive, Disjoint, 45 cells, Supplementary Fig. S30A` sign count.  The
    # first three are the caption's C--H tail verbatim; the leader of the
    # fourth landed on the DEPTH-MATCHED row, whose injection interval
    # [5.9e-5, 5.1e-4] excludes zero (only `unrelated` is zero by
    # construction, and the caption already says so); the last three describe
    # the 45-cell cohort, of which this panel plots no row at all.  Deleting
    # them is the whole of the fix: every remaining string here is a direct
    # series label or the panel's own finding.
    return out


# ── panel D: dose ────────────────────────────────────────────────────────
def panel_dose(ax, dose):
    """Localization against normalized dose; shared y with panel E."""
    for perturbation, colour, marker in (("matched additive", INJECT, M_INJECT),
                                         ("focal shunt", SHUNT, M_SHUNT)):
        rows = dose[perturbation]
        x = np.asarray([r[0] for r in rows])
        mean = np.asarray([r[1] for r in rows])
        lo = np.asarray([r[2] for r in rows])
        hi = np.asarray([r[3] for r in rows])
        # per-cell points (2026-09-11): the caption's C--H tail promises
        # `dots, cells` and D drew none, while the eight cell means per dose
        # sat unused in the Source Data table.  The fan is geometric in the
        # log2 abscissa so the jitter is the same width at every dose.
        for value, _m, _lo, _hi, cells in rows:
            ax.plot(value * 2.0 ** np.linspace(-0.075, 0.075, cells.size),
                    cells, linestyle="none", marker="o", ms=SEED_MS,
                    markerfacecolor=colour, markeredgecolor="none",
                    alpha=SEED_ALPHA, zorder=2.0)
        ax.errorbar(x, mean, yerr=[mean - lo, hi - mean], marker=marker,
                    ms=MEAN_MS, lw=LW_DATA, color=colour,
                    markerfacecolor="white", markeredgecolor=colour,
                    markeredgewidth=LW_ERR, elinewidth=LW_ERR,
                    capsize=ERR_CAPSIZE, zorder=3)
    ax.set_xscale("log", base=2)
    ax.set_xticks((0.25, 0.5, 1, 2), ("0.25", "0.5", "1", "2"))
    ax.minorticks_off()
    ax.set_xlim(0.215, 2.35)
    ax.set_ylim(*LOCAL_YLIM)
    ax.set_yticks(LOCAL_YTICKS, LOCAL_TICKLABELS)
    ax.tick_params(labelsize=PT_BASE, pad=0.8, length=2.0)
    ax.set_xlabel("Dose (× local conductance)", fontsize=PT_EMPH, color=INK,
                  labelpad=0.5)
    ax.set_ylabel(LOCAL_LABEL, fontsize=PT_EMPH, color=INK, labelpad=0.5)
    ax.text(0.03, 0.965, "focal shunt", color=SHUNT, fontsize=PT_BASE,
            transform=ax.transAxes, ha="left", va="top", zorder=6)
    ax.text(0.03, 0.880, "matched injection", color=INJECT, fontsize=PT_BASE,
            transform=ax.transAxes, ha="left", va="top", zorder=6)
    # `permissive normalized / passive parameters` stood here; it is the
    # caption's C sentence verbatim, and it occupied ordinate that now
    # carries the per-cell fan.
    return ax


# ── panel E: adjoint replacement ─────────────────────────────────────────
def panel_adjoint(ax, shapley):
    """The four factor-substitution corners, cells paired across d′ and q′d′."""
    columns = ("matched_additive_localization",
               "driving_force_only_localization",
               "adjoint_only_localization",
               "full_shunt_localization")
    colours = (INJECT, INJECT, SHUNT, SHUNT)
    pair = np.column_stack([shapley[columns[1]], shapley[columns[3]]])
    for row in pair:
        ax.plot((1, 3), row, color=MUTE, lw=LW_HAIR, alpha=0.5, zorder=1.6)
    for index, column in enumerate(columns):
        values = shapley[column]
        jitter = np.linspace(-0.16, 0.16, values.size)
        ax.plot(index + jitter, values, linestyle="none", marker="o",
                ms=SEED_MS, mfc=colours[index], mec="none", alpha=SEED_ALPHA,
                zorder=2.4)
        mean, lo, hi = shapley["stats"][column]
        ax.errorbar(index, mean, yerr=[[mean - lo], [hi - mean]],
                    marker=M_SHUNT, ms=MEAN_MS, color=colours[index],
                    markerfacecolor="white", markeredgecolor=colours[index],
                    markeredgewidth=LW_ERR, lw=LW_ERR, capsize=ERR_CAPSIZE,
                    zorder=5)
    ax.set_xticks(range(4), ("inject", "d′", "q′", "q′d′"))
    ax.set_xlim(-0.6, 3.6)
    ax.set_ylim(*LOCAL_YLIM)
    # D and E share one y axis, which means they share ONE major ticker: a
    # set_yticks(labels) here silently blanked D's tick labels too, so the
    # whole pair shipped with a bare axis.  E hides its own labels instead.
    # ...but E still has to PRINT them: with labelleft=False the panel
    # shipped with no numeral and no ordinate title at all, and its values
    # could only be read by counting unlabelled gridlines across D.  Sharing
    # the ticker is what forbids a second set_yticks here, not a second set
    # of tick LABELS, so the labels (and the shared axis title) come back.
    ax.tick_params(labelsize=PT_BASE, pad=0.8, length=2.0, labelleft=True)
    ax.set_ylabel(LOCAL_LABEL, fontsize=PT_EMPH, color=INK, labelpad=0.5)
    ax.set_xlabel("Substituted factor", fontsize=PT_EMPH, color=INK,
                  labelpad=0.5)
    mean, lo, hi = shapley["replacement"]
    # PLAN section 4 E asks for three SEPARATE items.  They were collapsed into one
    # seven-line stack in the top-left corner, which squeezed the four
    # columns right; each now sits where the plan puts it.
    # (i) `substitution, not / a decomposition` stood top-right: a methods
    # note, deleted 2026-09-11 with the rest of this panel's dead-band prose.
    # (ii) the printed contrast is now DRAWN, not merely printed.  +0.079 is
    # q′d′ minus d′ -- the paired grey lines already span exactly that pair,
    # straight across the q′ column -- but nothing said so, and beside a
    # panel whose q′ mean is 0.130 the number read as the q′ column itself
    # or as q′ minus inject (0.095, which is also the printed upper bound).
    # A paired-difference bracket over the two columns it spans fixes the
    # identification and keeps the statistic on the artwork.
    bracket_y = 0.209
    ax.plot([1, 3], [bracket_y, bracket_y], color=INK, lw=LW_HAIR,
            solid_capstyle="butt", zorder=6)
    for x_end in (1, 3):
        ax.plot([x_end, x_end], [bracket_y - 0.009, bracket_y], color=INK,
                lw=LW_HAIR, solid_capstyle="butt", zorder=6)
    # 2.00, not 2.15: at 2.15 the second line ended 1.7 pt and the first
    # 1.1 pt PAST the 506.4 pt live right edge, the same overhang already
    # corrected in H's x label.  2.00 is also the bracket's midpoint.
    # Design pass 2026-09-14: the bracket carries its value alone.  The
    # interval, the 8/8 count and the factor key (q adjoint, d driving
    # force, prime = post-shunt) are caption E's and the running text's own
    # words; the values are held to the source here instead.
    ax.text(2.00, 0.235, f"+{mean:.3f}", fontsize=PT_BASE, color=INK,
            ha="center", va="center", zorder=6)
    assert abs(mean - 0.079) < 5e-4 and abs(lo - 0.060) < 5e-4 \
        and abs(hi - 0.095) < 5e-4 and shapley["replacement_positive"] == 8
    # `shared y with D` stood at the lower right, inside the paired-line
    # bundle that is the densest ink in the panel.  E prints its own tick
    # labels now, so the note has nothing left to excuse.
    return ax


# ── panel F: signed change forest ────────────────────────────────────────
# "Rm 300" printed an unsubscripted R_m beside panel H's properly chained
# one; the gutter is 28 pt wide and cannot host a chained span, so the row
# labels name the manipulation in words and the in-panel line below says
# which quantity the numbers are.
# review pass 2026-09-23: two-line labels (cohort over R_m); the second line
# is left blank for the chained R_m span, and cohort sizes are in the legend
SIGNED_ROWS = (("original_eight", "Ra150_Rm300", "Initial\n ", "300"),
               ("original_eight", "Ra150_Rm15000", "Initial\n ", "15,000"),
               ("v661_disjoint", "Ra150_Rm300", "Disjoint\n ", "300"),
               ("v661_disjoint", "Ra150_Rm15000", "Disjoint\n ", "15,000"))


def signed_fill_key(ax):
    """A compact neutral key: fill names spatial support, not perturbation."""
    artists = []
    # Keep the key immediately above its own axes: at 1.07 it consumes the
    # preceding row's gutter on the current 444 pt publication canvas.
    key_y = 1.02
    for x, label, face in ((0.025, 'descendants', INK),
                            (0.57, 'off-route', 'white')):
        artists.extend(ax.plot([x], [key_y], transform=ax.transAxes,
                               marker='o', linestyle='none',
                               markersize=MARKER_MS * 0.8,
                               markerfacecolor=face, markeredgecolor=INK,
                               markeredgewidth=LW_HAIR, clip_on=False,
                               zorder=6))
        artists.append(ax.text(x + 0.04, key_y, label,
                               transform=ax.transAxes, fontsize=PT_BASE,
                               color=INK, ha='left', va='center',
                               clip_on=False, zorder=6))
    return artists


def panel_signed(canvas, ax, summary, cells):
    """Signed log change: the shunt attenuates, the matched injection adds."""
    def pick(cohort, regime, perturbation, category):
        row = summary[summary.cohort.eq(cohort) & summary.regime.eq(regime)
                      & summary.perturbation.eq(perturbation)
                      & summary.category.eq(category)]
        assert len(row) == 1, (cohort, regime, perturbation, category)
        row = row.iloc[0]
        return (float(row.mean_signed_log_change), float(row.ci95_low),
                float(row.ci95_high), int(row.n_cells))

    rows = []
    for cohort, regime, label, _rm in SIGNED_ROWS:
        mean, lo, hi, n = pick(cohort, regime, "focal shunt", "descendant")
        seeds = cells[cells.cohort.eq(cohort) & cells.regime.eq(regime)
                      & cells.perturbation.eq("focal shunt")
                      & cells.category.eq("descendant")] \
            .signed_log_change.to_numpy(float)
        assert seeds.size == n, (cohort, regime, seeds.size, n)
        rows.append(dict(label=label, mean=mean, lo=lo, hi=hi,
                         seeds=list(seeds), n=n))
    # tick=False: the 6 % band already ties each label to its row, and the
    # library draws the alternative gutter hairline ON the left spine, where
    # it is the one data artist the grouped key block cannot avoid.
    out = forest(ax, rows,
                 value_label="Signed \u0394 log |\u03b3| (log units)",
                 reference=0.0, reference_label="", color="shunting",
                 xlim=(-0.262, 0.098), tag="", tick=False)
    ax.tick_params(axis="x", labelsize=PT_BASE, pad=0.8, length=2.0)
    ax.xaxis.labelpad = 0.5
    ypos = out["ypos"]
    arms = (("focal shunt", "depth-matched unrelated", 0.14, SHUNT, M_SHUNT,
             "none"),
            ("matched additive", "descendant", 0.28, INJECT, M_INJECT,
             INJECT),
            ("matched additive", "depth-matched unrelated", 0.42, INJECT,
             M_INJECT, "none"))
    for index, (cohort, regime, _label, _rm) in enumerate(SIGNED_ROWS):
        for perturbation, category, dy, colour, marker, face in arms:
            mean, lo, hi, _n = pick(cohort, regime, perturbation, category)
            y = ypos[index] + dy
            ax.plot([lo, hi], [y, y], color=colour, lw=LW_ERR, zorder=3.0,
                    solid_capstyle="butt")
            ax.plot([mean], [y], linestyle="none", marker=marker,
                    markersize=MARKER_MS, markerfacecolor=face,
                    markeredgecolor=colour, markeredgewidth=LW_ERR,
                    zorder=4.0)
    ax.plot([-0.262, 0.098], [1.58, 1.58], color=EDGE, lw=LW_HAIR,
            zorder=1.2, solid_capstyle="butt")
    # each key sentence as one ADJACENT pair of lines in a clear strip; the
    # four clauses used to sit one per forest row, so neither sentence read
    # as a statement.  Both pairs stop short of the dashed zero rule.
    # the two sentence lines carry the perturbation key: the shunt clause is
    # set in the shunt hue and the injection clause in the injection hue, so
    # green circle and grey square are named on the panel without a legend.
    # "open = depth-matched off-route" is 98.3 pt and the clear strip left of
    # the zero rule is 73 pt, so the verbatim string wraps at its own hyphen
    # over three lines; the tie to panel C's `depth-matched` row is kept.
    # Keep the spatial fill key even though the former prose blocks are gone.
    # The caption specifies that off-route sites are depth-matched unrelated
    # sites, and separately identifies the two perturbation colours/shapes.
    signed_fill_key(ax)
    # the third label line is a properly chained R_m (the gutter cannot host
    # a mathtext span, and "Rm" beside panel H's R_m read as a typo)
    for index, (_cohort, _regime, _label, rm_value) in enumerate(SIGNED_ROWS):
        tail = ax.annotate(f" {rm_value}", xy=(0.0, ypos[index]),
                           xycoords=("axes fraction", "data"),
                           xytext=(-4.0, -4.4), textcoords="offset points",
                           ha="right", va="center", fontsize=PT_BASE,
                           color=INK, annotation_clip=False)
        sub = ax.annotate("m", xy=(0.0, 0.5), xycoords=tail,
                          xytext=(0.0, -1.6), textcoords="offset points",
                          ha="right", va="center", fontsize=PT_BASE,
                          color=INK, annotation_clip=False)
        ax.annotate("R", xy=(0.0, 0.5), xycoords=sub, xytext=(0.0, 1.6),
                    textcoords="offset points", ha="right", va="center",
                    fontsize=PT_BASE, color=INK, annotation_clip=False)
    # The dashed reference and its zero tick already specify no change;
    # another label would crowd the spatial key or the first cohort row.
    return out


# ── panel G: electrotonic state ──────────────────────────────────────────
def panel_state(ax, physical, ranges):
    """Shunt-minus-injection contrast against membrane resistance."""
    ax.set_xscale("log")
    ax.set_xlim(240.0, 40000.0)
    ax.set_ylim(-0.030, 0.098)
    drops = {}
    for cohort, label, marker, filled in COHORTS:
        points = physical[cohort]
        x = np.asarray([p[0] for p in points], dtype=float)
        mean = np.asarray([p[1] for p in points])
        lo = np.asarray([p[2] for p in points])
        hi = np.asarray([p[3] for p in points])
        ax.errorbar(x, mean, yerr=[mean - lo, hi - mean], marker=marker,
                    ms=MEAN_MS, lw=LW_DATA, color=INK,
                    markerfacecolor=INK if filled else "white",
                    markeredgecolor=INK, markeredgewidth=LW_ERR,
                    elinewidth=LW_ERR, capsize=ERR_CAPSIZE, zorder=3)
        by_rm = {int(p[0]): p[1] for p in points}
        drops[cohort] = abs(by_rm[300] / by_rm[15000])
    reference_line(ax, 0.0, axis="y", label=None)
    # 10,000 loses its major TICK as well as its label.  At 99.8 pt of axes
    # the "10,000" and "30,000" strings abut exactly (62.1--83.5 pt against
    # 83.5--104.9) and cannot both be set; keeping the bare tick left one
    # unlabelled major on a log abscissa, which is worse than four labelled
    # decades. 10,000 stays on the axis as a log minor tick.
    # 2026-09-11: the labelled major moves from 30,000 to 15,000.  The whole
    # decade 3,000--30,000 carried no labelled tick, and 15,000 -- the
    # calibration the inset, the caption, the main text and the whole of
    # panel H rest on -- had no tick at all, major or minor (a log minor
    # falls at 2,3..9 x a decade, never at 1.5).  30,000 keeps its minor.
    ax.set_xticks((300, 1000, 3000, 15000),
                  ("300", "1,000", "3,000", "15,000"))
    ax.set_yticks((0.0, 0.02, 0.04, 0.06, 0.08),
                  ("0", "0.02", "0.04", "0.06", "0.08"))
    ax.tick_params(labelsize=PT_BASE, pad=0.8, length=2.0)
    ax.tick_params(axis="x", which="minor", length=1.2)
    ax.set_xlabel("Membrane resistance (\u03a9 cm\u00b2)", fontsize=PT_EMPH,
                  color=INK, labelpad=0.5)
    # 2.6 pt of labelpad: at 0.5 the "(log units)" parentheses touched the
    # "0.08" tick numeral (0.1 pt of clearance at print scale)
    ax.set_ylabel(CONTRAST_LABEL, fontsize=PT_EMPH, color=INK, labelpad=2.6)
    # cohort key: shape and fill carry the cohort (DECISIONS, Figure 8), and
    # each cohort's median axial/leak range rides under its own name so the
    # ratio is an annotation and never a second abscissa (PLAN 0.6)
    # each cohort name carries ITS OWN marker glyph immediately before it:
    # without one, nothing on the page or in the caption said which series
    # was the filled diamond and which the open triangle.
    block = ((COHORTS[1], ranges["v661_disjoint"], 0.950, None),
             (COHORTS[0], ranges["original_eight"], 0.876, None))
    keyed = []
    for (_key, text, marker, filled), (rlo, rhi), y_name, y_range in block:
        name = ax.text(0.995, y_name, text, transform=ax.transAxes,
                       fontsize=PT_BASE, color=INK, ha="right", va="center",
                       zorder=6)
        # (the axial/leak range line moved to the legend, 2026-09-23)
        keyed.append((name, marker, filled, y_name))
    # Design pass 2026-09-14: the `>= 33x from 300 to 15,000` tag is gone
    # (the running text: `more than an order of magnitude`); held here.
    smallest = min(drops.values())
    assert smallest >= 10.0, smallest

    # 0.31 (not 0.28) lifts the inset clear of the source-region box it
    # magnifies: at 0.28 the frame overlapped the box by 0.0024 log units
    inset = ax.inset_axes([0.60, 0.31, 0.38, 0.27])
    inset.set_xscale("log")
    inset.set_xlim(2400.0, 40000.0)
    inset.set_ylim(-0.0062, 0.0082)
    for spine in inset.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(LW_EDGE)
        spine.set_color(EDGE)
    inset.set_facecolor("white")
    inset.patch.set_alpha(1.0)
    for cohort, _label, marker, filled in COHORTS:
        points = [p for p in physical[cohort] if abs(p[1]) < 0.006]
        if not points:
            continue
        x = np.asarray([p[0] for p in points], dtype=float)
        mean = np.asarray([p[1] for p in points])
        lo = np.asarray([p[2] for p in points])
        hi = np.asarray([p[3] for p in points])
        inset.errorbar(x, mean, yerr=[mean - lo, hi - mean], marker=marker,
                       ms=MARKER_MS - 1.6, lw=LW_HAIR, color=INK,
                       markerfacecolor=INK if filled else "white",
                       markeredgecolor=INK, markeredgewidth=LW_ERR,
                       elinewidth=LW_ERR, capsize=1.4, zorder=3)
        # ...and the intervals again ON TOP of the marks.  At 7,300 px per
        # log unit the disjoint interval at R_m 15,000 is 1.2e-3 tall, i.e.
        # smaller than the open triangle that carries its mean, so the panel
        # asserted an interval excluding zero that the reader could not see.
        # The caption's opposite-sign sentence rests on exactly this pair.
        # QA 2026-09-11 (major): with the stroke at zorder 6 the disjoint
        # cohort's open triangle came out SOLID, because its 3.0 pt marker has
        # a ~1 pt interior and the 2.6 pt interval plus its two caps covered
        # all of it -- a fifth, unkeyed marker in the one window the caption's
        # opposite-sign claim rests on.  The stroke now sits under the marker;
        # only the caps ride on top, wide enough to show past its edges.
        for xv, l, h in zip(x, lo, hi):
            inset.plot([xv, xv], [l, h], color=INK, lw=LW_ERR, zorder=2.5,
                       solid_capstyle="butt")
            for bound in (l, h):
                inset.plot([xv], [bound], marker="_", markersize=4.6,
                           markeredgecolor=INK, markeredgewidth=LW_ERR,
                           linestyle="none", zorder=6)
    inset.axhline(0.0, color=MUTE, lw=LW_REF, dashes=(2.2, 1.8), zorder=1)
    # The inset frame is ~38 x 31 pt: no 7 pt numeral row fits inside it, and
    # outside it the labels floated in the PARENT axes, where "0.005" sat
    # where the main ordinate reads 0.037 -- exactly the free-floating
    # interior numeral row PLAN section 0.6 exists to remove.  The inset therefore
    # keeps its zero rule and its marks and carries NO tick labels; its
    # window and its two values are stated in the caption, and one mute line
    # above the frame gives the ordinate half-range.
    inset.set_xticks(())
    inset.minorticks_off()
    # ...except ONE numeral.  Without it the frame's internal zero rule sits
    # where the PARENT ordinate reads 0.025, and the magnified marks read as
    # genuine positive contrasts of about 0.02.  "0" is set right-aligned
    # 2 pt outside the inset's left spine at the height of its own dashed
    # rule (CF-7), i.e. exactly where that inset's y tick label would be, so
    # it is read as the inset's and not as a free-floating parent numeral.
    # It is the only numeral the 38 x 31 pt frame has room for: the 3,000
    # whisker spans -0.0043 to +0.0068 and occupies the whole interior left
    # edge, so the half-range stays on the "inset +- 0.005" line above.
    # ...set as a REAL tick of the inset rather than a free annotation.  As
    # an annotation the numeral had no tick mark to own it and floated in the
    # parent axes, where the parent ordinate reads about 0.028.
    inset.set_yticks((0.0,), ("0",))
    inset.tick_params(axis="y", labelsize=PT_BASE, pad=1.0, length=1.6,
                      width=LW_HAIR, colors=MUTE)
    # The magnified window used to be a closed rectangle: every edge of a box
    # that CONTAINS the points it magnifies must cross them, and it cut both
    # cohort lines and the 8-cell whisker caps.  It is replaced by an open
    # LW_HAIR mute bracket under the window, below all ink in it, with 2.5 pt
    # end ticks turned up towards the region.
    bracket_y = -0.0175
    ax.plot([2400.0, 40000.0], [bracket_y, bracket_y], color=MUTE,
            lw=LW_HAIR, zorder=1.4, solid_capstyle="butt")
    for x_end in (2400.0, 40000.0):
        ax.plot([x_end, x_end], [bracket_y, bracket_y + 0.0028], color=MUTE,
                lw=LW_HAIR, zorder=1.4, solid_capstyle="butt")
    # PLAN section 4 G: "a LW_HAIR mute connector marks the inset's source region on
    # the main axes".  The bracket alone left the reader to guess that the
    # window under the abscissa and the frame at the upper right were the
    # same object, so the two ends of the bracket are tied to the two lower
    # corners of the frame.  zorder 0.9 puts both under the reference rule
    # and under every mark, so they read as background tie-lines.
    # ...from the RIGHT end only (2026-09-11).  The left connector ran from
    # the 2,400 end of the bracket to the inset's lower-left corner, which
    # put it straight through the 3,000 and 5,000 marks and their whiskers --
    # the one stretch of the abscissa where every mark of both cohorts is
    # within 0.002 of the rule.  The right connector rises at the far edge of
    # the axes, past the last sampled resistance, and crosses nothing; with
    # the bracket's two end ticks it still ties window to frame.
    to_axes = ax.transData + ax.transAxes.inverted()
    x_frac, y_frac = to_axes.transform((40000.0, bracket_y))
    ax.plot([x_frac, 0.98], [y_frac, 0.31], transform=ax.transAxes,
            color=MUTE, lw=LW_HAIR, zorder=0.9, solid_capstyle="butt")

    def place_cohort_markers():
        """Set each key marker 3.4 pt left of its own label.

        Deferred to after ``lock_reserves()`` / ``match_row_heights()``: the
        axes is resized between the panel draw and the save, so the label's
        width in axes fractions is only known at the end.
        """
        renderer = ax.figure.canvas.get_renderer()
        width_pt = ax.get_window_extent(renderer).width / ax.figure.dpi * 72.0
        for name, marker, filled, y_name in keyed:
            box = name.get_window_extent(renderer) \
                .transformed(ax.transAxes.inverted())
            ax.plot([box.x0 - 4.6 / width_pt], [y_name], marker=marker,
                    linestyle="none", transform=ax.transAxes, ms=MARKER_MS,
                    markerfacecolor=INK if filled else "white",
                    markeredgecolor=INK, markeredgewidth=LW_ERR,
                    clip_on=False, zorder=6)
    return drops, place_cohort_markers


# ── panel H: background-conductance rescue ───────────────────────────────
def panel_background(ax, sel, per_cell):
    """Distributed background conductance restores the contrast at 15,000."""
    x = np.arange(3)
    mean = sel.mean_shunt_minus_additive.to_numpy(float)
    lo = sel.ci95_low.to_numpy(float)
    hi = sel.ci95_high.to_numpy(float)
    positive = sel.cells_positive.to_numpy(int)
    for index, multiplier in enumerate((0, 1, 4)):
        values = per_cell[multiplier]
        jitter = np.linspace(-0.16, 0.16, values.size)
        ax.plot(index + jitter, values, linestyle="none", marker="o",
                ms=SEED_MS, mfc=INK, mec="none", alpha=SEED_ALPHA, zorder=2.4)
    # NO segment between the three means (2026-09-11).  The abscissa is
    # ordinal -- 0, 1 and 4 drawn at equal spacing under a quantitative axis
    # title -- so a joining line turned a 2.6-fold saturation (0.121 per leak
    # unit from 0 to 1, 0.046 per unit from 1 to 4) into a straight ramp.
    ax.errorbar(x, mean, yerr=[mean - lo, hi - mean], marker=M_INITIAL,
                ms=MEAN_MS, lw=LW_DATA, linestyle="none", color=INK,
                markerfacecolor=INK,
                markeredgecolor=INK, markeredgewidth=LW_ERR,
                elinewidth=LW_ERR, capsize=ERR_CAPSIZE, zorder=5)
    reference_line(ax, 0.0, axis="y", label=None)   # 2026-09-23
    for index, multiplier in enumerate((0, 1, 4)):
        ceiling = max(hi[index], per_cell[multiplier].max())
        ax.text(index, ceiling + 0.014, f"{positive[index]}/8",
                fontsize=PT_BASE, color=MUTE, ha="center", va="bottom",
                zorder=6)
    ax.set_xticks(x, ("0", "1", "4"))
    ax.set_xlim(-0.5, 2.5)
    # 2026-09-11: the -0.150 / 0.45 window existed to open a three-line
    # footer strip and a two-line sentence band.  Both blocks are gone (see
    # below), so the window closes onto the data: the lowest cell value is
    # -0.0445 and the highest sign-count label tops out at 0.309.
    ax.set_ylim(-0.062, 0.325)
    ax.set_yticks((0.0, 0.1, 0.2, 0.3), ("0", "0.1", "0.2", "0.3"))
    ax.tick_params(labelsize=PT_BASE, pad=0.8, length=2.0)
    ax.set_xlabel("Background leak (\u00d7 baseline)", fontsize=PT_EMPH,
                  color=INK, labelpad=0.5)
    # 103.7 pt of label centred on a ~99 pt axes overhung the 506.4 pt live
    # right edge by 1.5 pt; 0.478 pulls its right edge back inside
    ax.xaxis.label.set_x(0.478)
    ax.set_ylabel(CONTRAST_LABEL, fontsize=PT_EMPH, color=INK, labelpad=2.6)
    # Five lines of prose stood here and are all deleted (2026-09-11).  The
    # two-line sentence `contrast restored at the / standard calibration` and
    # the three-line footer `R_m = 15,000 Omega cm^2 / input-conductance- /
    # normalized dose 1` are together the caption's H sentence verbatim, and
    # the footer's chained subscript descended into the ascenders of the line
    # under it.  Between them they were the only reason this panel needed
    # 46 % of its ordinate empty.
    return ax


# ── the slot-fill balancer (private helper, DECISIONS G5) ────────────────
GUTTER_PT = 28.0        # one row-label gutter for every module column
LETTER_DX_PT = 24.0     # one letter offset for the whole figure (CF-12)


def _balance_slot_fill(canvas, target=1.35):
    """One left reserve for every module column, and the resulting spread.

    The layout contract requires every panel of one grid row that spans the
    same number of modules to keep the SAME axes width, so the row-label
    gutter the two forest panels claim out of module column 0 has to be given
    to columns 4, 6 and 8 as well: the gutter becomes a property of the grid
    instead of a property of one panel.  Returns the reserve and the measured
    ``panel-emphasis`` slot-fill spread.
    """
    def measure():
        manifest = canvas.manifest()
        margins = manifest["margins_pt"]
        hgutter = manifest["hgutter_pt"]
        vgutter = manifest["vgutter_pt"]
        cols = manifest["module_cols"]
        row_h = manifest["row_h_pt"]
        module_w = (manifest["width_pt"] - margins["left"] - margins["right"]
                    - (cols - 1) * hgutter) / cols

        def slot(rec):
            span = max(int(rec["colspan"]), 1)
            rows = max(int(rec.get("rowspan", 1)), 1)
            width = span * module_w + (span - 1) * hgutter
            height = sum(row_h[rec["row"]:rec["row"] + rows]) \
                + (rows - 1) * vgutter
            return width * height

        panels = manifest["panels"]
        return {rec["name"]: rec["w_pt"] * rec["h_pt"] / slot(rec)
                for rec in panels}

    reserve = max(GUTTER_PT,
                  max(lock[0] for lock in canvas._locks.values()))
    # the RIGHT side too (design pass 2026-09-14): at 30 pt gutters the y
    # decorations of D and E overrun the gutter, so the lock pass pads C's
    # and D's right edges by different amounts (4.7 and 2.6 pt) and the
    # three axes of a row come out three different widths.  Every data panel
    # takes the largest measured right pad, so C = D = E and F = G = H.
    right = max(lock[1] for name, lock in canvas._locks.items()
                if not canvas._record_for(name).get("schematic"))
    for rec in canvas._records:
        canvas.declare_reserve(rec["name"], left=reserve, right=right)
    canvas.lock_reserves()
    fills = measure()
    spread = max(fills.values()) / min(fills.values())
    return {"left_reserve_pt": round(reserve, 2),
            "slot_fill_spread": round(spread, 3),
            "slot_fill": {k: round(v, 3) for k, v in fills.items()}}


# ── the render-time display table ────────────────────────────────────────
def display_rows(category, dose, shapley, signed_summary, signed_cells,
                 physical, sel, per_cell, gains):
    rows = []
    for node, size, gain in zip(reversed(gains["path"]), [17, 5, 4, 6, 46],
                                [gains["gain"][n]
                                 for n in reversed(gains["path"])]):
        rows.append(dict(panel="B", record="derived_gain", block=int(node),
                         block_size=size, eta=gain, kappa=gains["kappa"],
                         root_id=MEDIAN_ROOT,
                         identity_residual=gains["residual"]))
    for (perturbation, relation), (values, (mean, lo, hi)) in category.items():
        rows.append(dict(panel="C", record="cohort_mean",
                         perturbation=perturbation, category=relation,
                         mean=mean, ci95_low=lo, ci95_high=hi, n_cells=8,
                         regime="normalized passive"))
        rows.extend(dict(panel="C", record="cell_value",
                         perturbation=perturbation, category=relation,
                         root_id=int(root), value=float(v))
                    for root, v in values.items())
    for perturbation, points in dose.items():
        for value, mean, lo, hi, cells in points:
            rows.append(dict(panel="D", record="cohort_mean",
                             perturbation=perturbation, dose=value, mean=mean,
                             ci95_low=lo, ci95_high=hi, n_cells=8))
            # Source Data completeness (2026-09-11): D, F and H drew marks,
            # fans or both whose per-cell values appeared in no row of
            # figure_08_plotted.csv, the file the caption names as the
            # figure's Source Data.  Every drawn point is now a row.
            rows.extend(dict(panel="D", record="cell_value",
                             perturbation=perturbation, dose=value,
                             root_id=int(root), value=float(v))
                        for root, v in cells.items())
    for column, (mean, lo, hi) in shapley["stats"].items():
        rows.append(dict(panel="E", record="cohort_mean", condition=column,
                         mean=mean, ci95_low=lo, ci95_high=hi, n_cells=8))
        rows.extend(dict(panel="E", record="cell_value", condition=column,
                         root_id=int(r), value=float(v))
                    for r, v in zip(shapley["root_id"], shapley[column]))
    mean, lo, hi = shapley["replacement"]
    rows.append(dict(panel="E", record="paired_contrast",
                     condition="q_prime_replacement", mean=mean, ci95_low=lo,
                     ci95_high=hi, n_cells=8,
                     cells_positive=shapley["replacement_positive"]))
    assert len(signed_summary) == 16, len(signed_summary)
    rows.extend(dict(panel="F", record="cohort_mean", **r)
                for r in signed_summary.to_dict("records"))
    drawn = {(c, r) for c, r, _l, _v in SIGNED_ROWS}
    arms = (("focal shunt", "descendant"),
            ("focal shunt", "depth-matched unrelated"),
            ("matched additive", "descendant"),
            ("matched additive", "depth-matched unrelated"))
    for cohort, regime in sorted(drawn):
        for perturbation, relation in arms:
            subset = signed_cells[
                signed_cells.cohort.eq(cohort) & signed_cells.regime.eq(regime)
                & signed_cells.perturbation.eq(perturbation)
                & signed_cells.category.eq(relation)]
            assert not subset.empty, (cohort, regime, perturbation, relation)
            rows.extend(dict(panel="F", record="cell_value", cohort=cohort,
                             regime=regime, perturbation=perturbation,
                             category=relation, root_id=int(r.root_id),
                             value=float(r.signed_log_change))
                        for r in subset.itertuples())
    for cohort, _label, _marker, _filled in COHORTS:
        for rm, mean, lo, hi, n, values in physical[cohort]:
            rows.append(dict(panel="G", record="cohort_mean", cohort=cohort,
                             membrane_resistance_ohm_cm2=rm, mean=mean,
                             ci95_low=lo, ci95_high=hi, n_cells=n))
            rows.extend(dict(panel="G", record="cell_value", cohort=cohort,
                             membrane_resistance_ohm_cm2=rm,
                             root_id=int(root), value=float(v))
                        for root, v in values.items())
    for record in sel.to_dict("records"):
        rows.append(dict(panel="H", record="cohort_mean",
                         background_leak_multiplier=record[
                             "background_leak_multiplier"],
                         mean=record["mean_shunt_minus_additive"],
                         ci95_low=record["ci95_low"],
                         ci95_high=record["ci95_high"],
                         n_cells=record["n_cells"],
                         cells_positive=record["cells_positive"]))
    for multiplier, values in sorted(per_cell.items()):
        rows.extend(dict(panel="H", record="cell_value",
                         background_leak_multiplier=int(multiplier),
                         root_id=int(root), value=float(v))
                    for root, v in values.items())
    # Convert identifiers before publish() constructs a heterogeneous frame;
    # empty IDs in cohort rows must not round the exact 64-bit cell IDs.
    from source_data_export import exact_id_table
    return exact_id_table(rows).to_dict("records")


def build(emit_main=True):
    gains = ancestry_gains()
    assert gains["n_descendants"] == 16, gains["n_descendants"]
    assert len(gains["ids"]) == 78, len(gains["ids"])
    assert gains["residual"] < 1e-15, gains["residual"]

    category = load_categories()
    dose = load_dose()
    shapley = load_shapley()
    signed_summary, signed_cells = load_signed()
    physical, ratio_ranges = load_physical()
    sel, per_cell = load_background()

    # cross-panel identity: D(dose 1) == E(full shunt) and E(inject)
    d_shunt = dict((r[0], r[1]) for r in dose["focal shunt"])
    d_inject = dict((r[0], r[1]) for r in dose["matched additive"])
    assert abs(d_shunt[1.0]
               - shapley["full_shunt_localization"].mean()) < 1e-12
    assert abs(d_inject[1.0]
               - shapley["matched_additive_localization"].mean()) < 1e-12

    # Design pass 2026-09-14: 23 + 124 + 30 + 104 + 30 + 108 + 25 = 444 pt at
    # 30 pt gutters (the lock pass carves ~9 pt off the top of row 2 for
    # row 1's x labels and the letters).  Data panels carry no titles, and
    # the key sentences of C, E, F and G are gone: caption C-H names every
    # series, marker and sign, and the running text quotes the values.
    # Review pass 2026-09-23: without titles, footers and prose tags the
    # schematic row needs 106 pt; the 18 pt go to the two data rows.
    canvas = NativeCanvas(444 / 72, 3, row_weights=[106, 113, 117],
                          hgutter_pt=30, vgutter_pt=30,
                          margins=Margins(left=49, right=12, top=23,
                                          bottom=25))
    canvas.letter_dx = LETTER_DX_PT
    a = canvas.panel("A", 0, 0, 6, schematic=True)     # no titles, 2026-09-23
    b = canvas.panel("B", 0, 6, 6, schematic=True)
    c = canvas.panel("C", 1, 0, 4)
    d = canvas.panel("D", 1, 4, 4, grid="y")
    e = canvas.panel("E", 1, 8, 4, grid="y", sharey=d)
    g_f = canvas.panel("F", 2, 0, 4)
    g = canvas.panel("G", 2, 4, 4, grid="y")
    h = canvas.panel("H", 2, 8, 4, grid="y")
    canvas.fig.canvas.draw()

    # data panels first: they set the labels the reserve lock measures
    panel_relations(canvas, c, category)
    panel_dose(d, dose)
    panel_adjoint(e, shapley)
    panel_signed(canvas, g_f, signed_summary, signed_cells)
    drops, place_cohort_markers = panel_state(g, physical, ratio_ranges)
    panel_background(h, sel, per_cell)
    canvas.lock_reserves()
    balance = _balance_slot_fill(canvas)
    canvas.match_row_heights(1)
    canvas.match_row_heights(2)
    canvas.lock_reserves()
    canvas.fig.canvas.draw()
    place_cohort_markers()

    panel_interventions(a)
    panel_gain_dictionary(b, gains)

    # Review pass 2026-09-23: the two regime titles (`Permissive normalized
    # model` over C-E, `Physical calibration` over F-H) sat in the letter
    # band above D and G; legend C-H names the regimes.
    style_direct_color_labels(canvas.fig)
    output = J / "figures/components/focused_main_08.pdf"
    findings = canvas.save(output, name="focused_main_08", dpi=400)
    if findings:
        print("layout findings: " + "; ".join(str(x) for x in findings))
    plt.close(canvas.fig)
    main = J / "figures/main/figure_09.pdf"
    main.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(output, main)

    sources = [J / "source_data" / folder / name for folder, name in [
        ("figure3", "segment_metrics.csv"),
        ("figure4", "category_effects.csv"),
        ("figure4", "focal_localization.csv"),
        ("focal_decomposition", "cell_shapley.csv"),
        ("focal_decomposition", "shapley_summary.json"),
        ("shunt_ancestry_gain/signed_calibration", "signed_cohort_summary.csv"),
        ("shunt_ancestry_gain/signed_calibration", "signed_cell_effects.csv"),
        ("shunt_ancestry_gain/signed_calibration", "signed_category_effects.csv"),
        ("shunt_ancestry_gain/signed_calibration", "validation.json"),
        ("physical_cable_sensitivity", "cell_primary_contrasts.csv"),
        ("physical_cable_sensitivity", "cell_electrotonic_ratios.csv"),
        ("physical_cable_sensitivity", "summary.json"),
        ("focal_selectivity_phase1", "paired_contrasts.csv"),
        ("focal_selectivity_phase1", "cell_condition_metrics.csv"),
        ("focal_selectivity_phase1", "summary.json")]]
    builders = [Path(__file__), Path(previous.__file__),
                J / "scripts/credit_first_figures/source_data_export.py",
                J / "scripts/build_main_figure_07.py",
                J / "scripts/build_main_figure_08.py",
                J / "scripts/build_journal_figures.py",
                J / "scripts/figure_canvas.py",
                J / "scripts/journal_style.py",
                J / "scripts/native_schematics.py"]
    panels = {
        "A": "Two state-matched interventions with identical tree geometry; "
             "shunt as an inhibitory contact, not a bar.",
        "B": "Exact ancestry partition and its derived per-block gain column "
             "on the median reconstructed cell.",
        "C": "Unit-dose relation selectivity as a forest; injection in "
             "neutral grey; off-route zeros annotated.",
        "D": "Normalized-dose contrast promoted out of the SI (old S48 / "
             "today's S28D; no SI copy kept); equal-weighted cell means.",
        "E": "Four-corner factor substitution (adjoint replacement), with the "
             "printed paired contrast and sign count.",
        "F": "Signed log changes at two physical calibrations, with the "
             "matched-injection arm restored.",
        "G": "Shunt-minus-injection contrast against membrane resistance, "
             "half width, with a near-zero inset.",
        "H": "Background-conductance rescue of the contrast at the standard "
             "calibration; new main panel."}
    notes = (
        "Eight-panel v2 rebuild of main Figure 8 on one NativeCanvas "
        "(518.4 x 490 pt, rows 128/116/116, schematic fraction 26.6 % of the "
        "live canvas on the AMENDMENTS B12 formula, 457.4 x 442 pt; the plan's "
        "27.2 % assumed a 35 pt bottom margin, this build uses 25 pt with a "
        "41 pt vertical gutter so the r1|r2 boundary clears the 8.5 pt floor). No new experiments and no change to any "
        "frozen source table: every estimate reuses the archived rows, "
        "estimators and bootstrap seeds. Panel D is the old S48 / S28D "
        "normalized-dose sheet promoted into the main figure, printed as "
        "equal-weighted cell means (dose 2 = 0.194, not the 0.203 pooled-site "
        "mean). Panel H is new artwork over the frozen "
        "focal_selectivity_phase1 table. Cohorts are encoded by marker shape "
        "and fill, not by ORDINAL_RAMP (DECISIONS, Figure 8). 2026-09-10 "
        "fix round: D and E share one y ticker, so E hides its own tick "
        "labels instead of blanking the pair's; panel B's five block gains "
        "lose their mute leaders (each read as a minus sign) and its two "
        "value columns leave DIV_CMAP's bp red for an achromatic edge ramp; "
        "panel C names the disjoint 45-cell cohort beside its 45/45 sign "
        "count; panel G draws each cohort's marker glyph before its direct "
        "label; panel A's active shunt keeps the inhibitory rim and a leader "
        "to its badge; panel F's key sentences are two adjacent line pairs "
        "and its rows carry a chained R_m. Visual-QA round (2026-09-10, "
        "second pass): A's g_shunt badge moved inside the hero card with a "
        "full-length leader; B's g_shunt badge moved to the clear band "
        "up-left of the focal contact (its box seeded into the leader "
        "obstacle set so no block tag's leader crosses it, and its own "
        "leader no longer passes under the soma); C's reference label "
        "right-aligned on the rule; E's corner stack split into the plan's "
        "three items (badge top-right, contrast centred over q'/q'd', key "
        "over the inject and d' columns); F restores the verbatim "
        "depth-matched off-route key and colours its two sentence lines by "
        "perturbation; G drops the inset's tick labels and replaces the "
        "closed source-region box with an open mute bracket under the "
        "window; H's sentence drops clear of the title and is reworded, its "
        "x label is pulled inside the live right edge, and its footer names "
        "the input-conductance normalization. Residual round (2026-09-10, "
        "third pass): B's three block tags gain an elbow leader that starts "
        "at each tag's OWN right edge; C's `no change` moves to the bottom of "
        "the zero rule (ordinate extended to 4.92) out of the row-label "
        "gutter and its sign count names the cohort by the AMENDMENTS 4c "
        "panel form `Disjoint, 45 cells`; E's printed contrast is centred at "
        "2.00 so both lines end inside the 506.4 pt live right edge; G drops "
        "the unlabelled 10,000 major tick, labels its inset's zero rule and "
        "ties the source-region bracket to the inset's lower corners with the "
        "plan's LW_HAIR mute connectors; H's ylim falls to -0.150 and its "
        "three footer lines rise 0.049 of the axes off the bottom spine. "
        "Visual-review round (2026-09-11, fourth pass): the figure loses "
        "sixteen lines of in-axes prose, every one of them either the "
        "caption verbatim or a description of a cohort the panel does not "
        "plot -- C's three-line stat block, its mis-anchored `injection "
        "off-route: zero by construction` tag (whose leader landed on the "
        "depth-matched row, whose injection interval excludes zero) and its "
        "three-line 45-cell sign count; D's `permissive normalized passive "
        "parameters`; E's `substitution, not a decomposition` and `shared y "
        "with D`; F's second `mean [95 % CI]`; H's two-line sentence and "
        "three-line footer. The ordinate then closes onto the data: D and E "
        "run to 0.286 instead of 0.335 (and D draws its eight per-cell "
        "points per dose, whose 0.2572 maximum sets the new top), H to "
        "-0.062/0.325 instead of -0.150/0.450. E prints its own y tick "
        "labels and axis title -- sharing D's ticker forbids a second "
        "set_yticks, not a second set of tick labels -- and its +0.079 is "
        "drawn as a paired-difference bracket over the d\u2032 and "
        "q\u2032d\u2032 columns it actually spans, the same pair the grey "
        "paired lines already join. H drops the segment joining its three "
        "means (the abscissa is ordinal, so the line hid a 2.6-fold "
        "saturation). G labels 15,000 instead of 30,000 (the calibration the "
        "inset, the caption and all of panel H rest on had no tick at all, "
        "major or minor), keeps only the right-hand inset connector (the "
        "left one crossed the 3,000 and 5,000 marks), shrinks the inset "
        "markers clear of the frame and redraws the inset intervals ON TOP "
        "of them, and sets the inset's zero as a real tick. Source Data "
        "completeness: every drawn per-cell point of D, F and H is now a "
        "cell_value row of figure_08_plotted.csv (D 64, F 424, H 24); G's "
        "183 were already there. "
        "Slot-fill "
        f"slot-fill balance: {balance}. Contrast drop from "
        f"R_m 300 to 15,000: {drops}.")
    record = publish(9, output, display_rows(category, dose, shapley,
                                             signed_summary, signed_cells,
                                             physical, sel, per_cell, gains),
                     sources, builders, panels, emit_main=emit_main,
                     layout_findings=[str(x) for x in findings], notes=notes)
    (J / "figures/provenance/credit_clarity_20260908"
     / "figure_08_layout.json").write_text(json.dumps(
         {"main_figure_sha256": sha(main), "slot_fill_balance": balance,
          "schematic_fraction": round(2 * 210.2 * 128.0 / (457.4 * 442.0), 4),
          "contrast_drop": {k: float(v) for k, v in drops.items()},
          "kappa_unit_dose": gains["kappa"],
          "identity_residual": gains["residual"],
          "block_sizes": [17, 5, 4, 6, 46],
          "block_gains": [float(gains["gain"][n])
                          for n in reversed(gains["path"])]},
         indent=2) + "\n")
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--emit-main", action="store_true", default=True)
    build(parser.parse_args().emit_main)
