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
CF-10 schematic area = (210.2 + 210.2) x 128 / (457.4 x 432) = 27.2 %, cap
      30 %, no waiver (the 29.6 % row-basis figure is the conservative bound).
CF-11 ``check_matrix_cells`` >= 6.0 pt for panel B's dictionary product.
CF-12 ``canvas.align_letters()`` is called unconditionally before ``save()``;
      no letter is hand-placed and no hand alignment loop exists.

Recorded waivers and deviations
-------------------------------
waiver D3: row 2 (F signed forest / G state curve / H background rescue) is
three 4-module panels that share no axis; F is a signed log change, G and H a
shunt-minus-injection contrast on different manipulations, and the row is
column-locked.

deviation from specification D8: ``ORDINAL_RAMP`` is NOT used for the cohorts.
``ORDINAL_RAMP[3]`` is dE 8.6/5.0 from the grey injection series and
``ORDINAL_RAMP[1]`` dE 13.1/12.8 from ``shunting``; DECISIONS (Figure 8) rules
that cohorts are encoded by marker shape and fill -- filled diamond = initial
eight-cell cohort, open triangle = disjoint 45-cell calibration cohort.

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
from native_schematics import (Frame, reference_line, check_matrix_cells,     # noqa: E402
                               collapsed_row_note, require_address_tint,
                               _text_w_pt)
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

LOCAL_YLIM = (-0.020, 0.248)
LOCAL_YTICKS = (0.0, 0.05, 0.10, 0.15, 0.20)
LOCAL_TICKLABELS = ("0", "0.05", "0.10", "0.15", "0.20")
LOCAL_LABEL = "localization index (log units)"
CONTRAST_LABEL = "shunt − injection (log units)"

COHORTS = (("original_eight", "Initial, 8 cells", M_INITIAL, True),
           ("v661_disjoint", "Disjoint, 45 cells", M_DISJOINT, False))
RM_ORDER = ("Ra150_Rm300", "Ra150_Rm1000", "Ra150_Rm3000", "Ra150_Rm5000",
            "Ra150_Rm15000", "Ra150_Rm30000")


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
            values = cells[cells.category.eq(relation)] \
                .median_abs_log_gradient_change.to_numpy(float)
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
            values = group.localization_index.to_numpy(float)
            assert values.size == 8, (perturbation, dose, values.size)
            rows.append((float(dose),
                         *mean_ci(values, seed=1660 + 10 * pindex
                                  + int(dose * 4))))
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
            values = group.difference.to_numpy(float)
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
    per_cell = {int(m): g.difference.to_numpy(float)
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
    for segment in rows:
        parent_id = parent[segment]
        if parent_id not in rows:
            continue
        home = block[segment]
        start, end = place[segment], place[parent_id]
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
    return place, anchors, float(bar[1, 0] - bar[0, 0]), soma_id


# ── panel A: the two state-matched interventions ─────────────────────────
def panel_interventions(ax):
    """Focal shunt versus the current injection that matches its drive."""
    f = Frame(ax)
    bracket_pt = 14.0
    cards = f.split(2, axis="x", gap_pt=9.0,
                    pad_pt=(0.0, 0.0, 0.0, bracket_pt))
    for cell, title, hero in ((cards[0], "focal shunt", True),
                              (cards[1], "matched injection", False)):
        core = f.task_card(cell, title=title, footer="soma V restored",
                           emphasis=hero, tone=None if hero else "control")
        body = Frame.inset(core, left=0.07, right=0.07, top=0.02)
        # 12 pt under the soma for the two arrows and the delta0 tag, 20 pt
        # over the canopy for the two-line intervention tag: both cards keep
        # identical geometry.
        body = (body[0], body[1] + f.fy(12.0), body[2],
                body[3] - f.fy(12.0 + 20.0))
        nodes = f.balanced_tree(body, depth=3, trunk=False, mode="forward",
                                output="z")
        site = _lerp(nodes["JL"], nodes["JLL"], 0.80)
        tag_y = body[1] + body[3] + f.fy(6.5)
        line_y = tag_y + f.fy(10.0)
        if hero:
            f.shunt(site, label=None)
            f.subscript((site[0] - f.fx(2.2), site[1] - f.fy(4.2)),
                        "g", "shunt", size=PT_BASE, color=COLORS["inh"],
                        ha="right", va="top")
            f.fade(["JLL"], nodes=nodes)
            f.fade([(nodes["JL"], nodes["JLL"])])
            f.text((body[0], line_y), "descendants attenuated",
                   size=PT_BASE, color=label_color(COLORS["inh"]),
                   ha="left", va="center")
            _formula(f, (body[0], tag_y),
                     [("γ", "i"), " = ", ("q", "i"), "(", ("E", "E"), " − ",
                      ("V", "i"), ")"], size=PT_BASE, color=INK, ha="left")
        else:
            f.contact(site, kind="inh", active=False)
            width = _formula(f, (body[0], line_y),
                             ["κ(", ("E", "I"), " − ", ("V", "k"), ")"],
                             size=PT_BASE, color=MUTE, ha="left")
            f.arrow((body[0] + f.fx(width + 3.0), line_y),
                    (site[0] - f.fx(1.6), site[1] + f.fy(2.2)),
                    color=MUTE, lw=LW_EDGE, head=3.4, rad=-0.18)
            f.text((body[0], tag_y), "no conductance change", size=PT_BASE,
                   color=MUTE, ha="left", va="center")
            f.badge((cell[0] + cell[2] - f.fx(2.0),
                     cell[1] + cell[3] - f.fy(1.5)), "control")
        f.error_in(nodes.soma, side="right")
        drive = (nodes.soma[0] - f.fx(15.0), nodes.soma[1] - f.fy(3.5))
        f.arrow(drive, (nodes.soma[0] - f.fx(3.0), nodes.soma[1]),
                color=MUTE, lw=LW_EDGE, head=3.4)
        f.subscript((drive[0] - f.fx(1.2), drive[1]), "I", "soma",
                    size=PT_BASE, color=MUTE, ha="right", va="center")
    y = f.fy(bracket_pt - 4.0)
    x0 = cards[0][0] + f.fx(5.0)
    x1 = cards[1][0] + cards[1][2] - f.fx(5.0)
    f.ax.plot([x0, x1], [y, y], color=MUTE, lw=LW_HAIR, zorder=2,
              solid_capstyle="butt")
    for x in (x0, x1):
        f.ax.plot([x, x], [y, y + f.fy(2.4)], color=MUTE, lw=LW_HAIR, zorder=2)
    f.text(((x0 + x1) / 2.0, y - f.fy(1.5)), "baseline focal current matched",
           size=PT_BASE, color=MUTE, va="top")
    f.require_soma_lowest()
    f.require_delta0()
    return ax


# ── panel B: one exact gain per ancestry block ───────────────────────────
def panel_gain_dictionary(ax, gains):
    """The ancestry partition of the median cell and its exact gain column."""
    f = Frame(ax)
    band = f.footer("driving forces may vary within a block")
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
                  1.0 - f.fy(band + key_pt + 24.0))
    place, anchors, bar_w, soma_id = _arbor_blocks(f, arbor_rect, gains,
                                                   colours, weights)
    f.text((0.0, top), f"root {MEDIAN_ROOT}", size=PT_BASE, color=MUTE,
           ha="left", va="center")
    f.subscript((place[FOCAL_SEGMENT][0] + f.fx(3.0),
                 place[FOCAL_SEGMENT][1] + f.fy(3.5)), "g", "shunt",
                size=PT_BASE, color=COLORS["inh"], ha="left", va="center")

    # block tags: the descendant tuft is named above the canopy, the two
    # proximal blocks below it, so no leader crosses the arbor
    desc_y = top - f.fy(9.5)
    f.text((0.0, desc_y), f"descendants ({gains['n_descendants']})",
           size=PT_BASE, color=label_color(BLOCK_DESC), ha="left",
           va="center")
    f.leader((f.fx(58.0), desc_y),
             (anchors[order[0]][0], anchors[order[0]][2]), color=MUTE)
    for index, (node, text, hue) in enumerate((
            (order[1], "sister blocks (3)", BLOCK_SIS),
            (path[0], "soma side", BLOCK_SOMA))):
        y_tag = f.fy(band + key_pt - 9.5 - 9.0 * index)
        f.text((0.0, y_tag), text, size=PT_BASE, color=label_color(hue),
               ha="left", va="center")
        f.leader((f.fx(58.0), y_tag), (anchors[node][0], anchors[node][1]),
                 color=MUTE)
    x_bar = arbor_rect[0] + arbor_rect[2] - f.fx(2.0) - bar_w
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

    _formula(f, (1.0, top), ["q\u2032 = diag(h) B \u03b7"], size=PT_BASE,
             ha="right")
    f.text((1.0, top - f.fy(9.5)), "h = baseline transfer", size=PT_BASE,
           color=MUTE, ha="right", va="center")
    prod_rect = (0.46, f.fy(band + 20.0), 0.40,
                 1.0 - f.fy(band + 20.0 + 34.0))
    check_matrix_cells(prod_rect[2] * f.w_pt, prod_rect[3] * f.h_pt, 5, 5,
                       where="Fig 8B dictionary_product")
    axes = f.dictionary_product(prod_rect, indicators, eta, cell_pt=7.0,
                                col_colors=hues, row_groups=groups,
                                captions=("B", "\u03b7", "q\u2032/q"),
                                numbers=False, collapse="auto")
    host, box = ax.get_position(), axes[2].get_position()
    x_right = (box.x1 - host.x0) / host.width
    y_top = (box.y1 - host.y0) / host.height
    y_bottom = (box.y0 - host.y0) / host.height
    for j in range(5):
        y = y_top - (j + 0.5) / 5.0 * (y_top - y_bottom)
        f.leader((x_right + f.fx(0.8), y), (x_right + f.fx(4.0), y),
                 color=MUTE)
        f.text((x_right + f.fx(5.2), y), f"{eta[j]:.3f}", size=PT_BASE,
               color=INK, ha="left", va="center")
    # Gamma_u printed on the gain column (AMENDMENTS B10)
    gamma_y = y_top + f.fy(7.0)
    f.subscript((x_right - f.fx(0.5), gamma_y), "\u0393", "u", size=PT_BASE,
                color=INK, ha="right", va="center")
    f.text((1.0, y_bottom - f.fy(23.0)),
           f"\u03ba = {gains['kappa']:.3f}; identity residual 1e\u221217",
           size=PT_BASE, color=MUTE, ha="right", va="top")
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
    out = forest(ax, rows, value_label="|Δ log |γ|| (log units)",
                 reference=0.0, reference_label="no change", color="shunting",
                 xlim=(-0.006, 0.158), tag="")
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
    ax.text(-0.005, -0.44, f"shunt {shunt_d / inject_d:.1f}\u00d7 injection",
            fontsize=PT_BASE, color=INK, ha="left", va="center", zorder=6)
    ax.text(0.156, -0.44, "focal shunt", fontsize=PT_BASE, color=SHUNT,
            ha="right", va="center", zorder=6)
    ax.annotate("matched injection", xy=(inject_d + 0.004, 0.28),
                xytext=(0.156, 0.62), textcoords="data", fontsize=PT_BASE,
                color=INJECT, ha="right", va="center", zorder=6,
                arrowprops=dict(arrowstyle="-", lw=LW_HAIR, color=MUTE,
                                shrinkA=1.5, shrinkB=1.5))
    for offset, text in enumerate(("n = 8 cells", "101 focal sites",
                                   "mean [95 % CI]")):
        ax.text(0.156, 1.00 + 0.33 * offset, text, fontsize=PT_BASE,
                color=MUTE, ha="right", va="center", zorder=6)
    ax.text(0.156, 2.86, "injection off-route:", fontsize=PT_BASE, color=MUTE,
            ha="right", va="center", zorder=6)
    ax.text(0.156, 3.19, "zero by construction", fontsize=PT_BASE, color=MUTE,
            ha="right", va="center", zorder=6)
    ax.plot([0.040, 0.0055], [3.19, 3.22], color=MUTE, lw=LW_HAIR,
            solid_capstyle="round", zorder=1.6)
    ax.text(0.156, 3.72, "45/45 cells positive,", fontsize=PT_BASE, color=MUTE,
            ha="right", va="center", zorder=6)
    ax.text(0.156, 4.05, "Supplementary Fig. S30A", fontsize=PT_BASE,
            color=MUTE, ha="right", va="center", zorder=6)
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
    ax.set_xlabel("dose (× local conductance)", fontsize=PT_EMPH, color=INK,
                  labelpad=0.5)
    ax.set_ylabel(LOCAL_LABEL, fontsize=PT_EMPH, color=INK, labelpad=0.5)
    ax.text(0.03, 0.965, "focal shunt", color=SHUNT, fontsize=PT_BASE,
            transform=ax.transAxes, ha="left", va="top", zorder=6)
    ax.text(0.03, 0.880, "matched injection", color=INJECT, fontsize=PT_BASE,
            transform=ax.transAxes, ha="left", va="top", zorder=6)
    ax.text(0.03, 0.735, "permissive normalized\npassive parameters",
            transform=ax.transAxes, ha="left", va="top", fontsize=PT_BASE,
            color=MUTE, linespacing=1.15, zorder=6)
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
    ax.set_yticks(LOCAL_YTICKS, [""] * len(LOCAL_YTICKS))
    ax.tick_params(labelsize=PT_BASE, pad=0.8, length=2.0)
    ax.set_xlabel("substituted factor", fontsize=PT_EMPH, color=INK,
                  labelpad=0.5)
    mean, lo, hi = shapley["replacement"]
    ax.text(1.5, 0.238, "q′ replacement", fontsize=PT_BASE, color=INK,
            ha="center", va="center", zorder=6)
    ax.text(1.5, 0.214,
            f"+{mean:.3f} [{lo:.3f}, {hi:.3f}], "
            f"{shapley['replacement_positive']}/8 cells",
            fontsize=PT_BASE, color=INK, ha="center", va="center", zorder=6)
    for offset, line in enumerate(("q adjoint,", "d driving force;",
                                   "′ = post-shunt")):
        ax.text(-0.52, 0.150 - 0.022 * offset, line, fontsize=PT_BASE,
                color=MUTE, ha="left", va="center", zorder=6)
    for offset, line in enumerate(("substitution, not", "a decomposition")):
        ax.text(-0.52, 0.196 - 0.022 * offset, line, fontsize=PT_BASE,
                color=MUTE, ha="left", va="center", zorder=6)
    ax.text(0.985, 0.008, "shared y with D", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=PT_BASE, color=MUTE, zorder=6)
    return ax


# ── panel F: signed change forest ────────────────────────────────────────
SIGNED_ROWS = (("original_eight", "Ra150_Rm300", "Initial,\n8 cells\nRm 300"),
               ("original_eight", "Ra150_Rm15000",
                "Initial,\n8 cells\nRm 15,000"),
               ("v661_disjoint", "Ra150_Rm300", "Disjoint,\n45 cells\nRm 300"),
               ("v661_disjoint", "Ra150_Rm15000",
                "Disjoint,\n45 cells\nRm 15,000"))


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
    for cohort, regime, label in SIGNED_ROWS:
        mean, lo, hi, n = pick(cohort, regime, "focal shunt", "descendant")
        seeds = cells[cells.cohort.eq(cohort) & cells.regime.eq(regime)
                      & cells.perturbation.eq("focal shunt")
                      & cells.category.eq("descendant")] \
            .signed_log_change.to_numpy(float)
        assert seeds.size == n, (cohort, regime, seeds.size, n)
        rows.append(dict(label=label, mean=mean, lo=lo, hi=hi,
                         seeds=list(seeds), n=n))
    out = forest(ax, rows,
                 value_label="signed \u0394 log |\u03b3| (log units)",
                 reference=0.0, reference_label="", color="shunting",
                 xlim=(-0.262, 0.098), tag="")
    ax.tick_params(axis="x", labelsize=PT_BASE, pad=0.8, length=2.0)
    ax.xaxis.labelpad = 0.5
    ypos = out["ypos"]
    arms = (("focal shunt", "depth-matched unrelated", 0.14, SHUNT, M_SHUNT,
             "none"),
            ("matched additive", "descendant", 0.28, INJECT, M_INJECT,
             INJECT),
            ("matched additive", "depth-matched unrelated", 0.42, INJECT,
             M_INJECT, "none"))
    for index, (cohort, regime, _label) in enumerate(SIGNED_ROWS):
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
    ax.text(-0.258, 0.38, "mean [95 % CI]", fontsize=PT_BASE, color=MUTE,
            ha="left", va="center", zorder=6)
    ax.text(-0.258, 0.68, "shunt attenuates;", fontsize=PT_BASE, color=INK,
            ha="left", va="center", zorder=6)
    ax.text(-0.258, 1.35, "injection enhances", fontsize=PT_BASE, color=INK,
            ha="left", va="center", zorder=6)
    ax.text(-0.258, 2.62, "filled = descendants", fontsize=PT_BASE, color=INK,
            ha="left", va="center", zorder=6)
    ax.text(-0.258, 3.30, "open = off-route", fontsize=PT_BASE, color=INK,
            ha="left", va="center", zorder=6)
    ax.text(-0.006, -0.46, "no change", fontsize=PT_BASE, color=MUTE,
            ha="right", va="center", zorder=6)
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
    ax.text(262.0, -0.0055, "no contrast", fontsize=PT_BASE, color=MUTE,
            ha="left", va="center", zorder=6)
    # 10,000 keeps its major tick but loses its label: at 99.8 pt of axes the
    # "10,000" and "30,000" strings abut (62.1--83.5 pt against 83.5--104.9).
    ax.set_xticks((300, 1000, 3000, 10000, 30000),
                  ("300", "1,000", "3,000", "", "30,000"))
    ax.set_yticks((0.0, 0.02, 0.04, 0.06, 0.08),
                  ("0", "0.02", "0.04", "0.06", "0.08"))
    ax.tick_params(labelsize=PT_BASE, pad=0.8, length=2.0)
    ax.tick_params(axis="x", which="minor", length=1.2)
    ax.set_xlabel("membrane resistance (\u03a9 cm\u00b2)", fontsize=PT_EMPH,
                  color=INK, labelpad=0.5)
    ax.set_ylabel(CONTRAST_LABEL, fontsize=PT_EMPH, color=INK, labelpad=0.5)
    # cohort key: shape and fill carry the cohort (DECISIONS, Figure 8), and
    # each cohort's median axial/leak range rides under its own name so the
    # ratio is an annotation and never a second abscissa (PLAN 0.6)
    block = ((COHORTS[1][1], ranges["v661_disjoint"], INK, 0.950, 0.876),
             (COHORTS[0][1], ranges["original_eight"], INK, 0.796, 0.722))
    for text, (rlo, rhi), colour, y_name, y_range in block:
        ax.text(0.995, y_name, text, transform=ax.transAxes, fontsize=PT_BASE,
                color=colour, ha="right", va="center", zorder=6)
        ax.text(0.995, y_range, f"axial/leak {rlo:.1f}\u2013{rhi:.0f}",
                transform=ax.transAxes, fontsize=PT_BASE, color=MUTE,
                ha="right", va="center", zorder=6)
    smallest = min(drops.values())
    ax.text(0.005, 0.045, f"\u2265 {np.floor(smallest):.0f}\u00d7 from 300 "
                          "to 15,000", transform=ax.transAxes,
            fontsize=PT_BASE, color=INK, ha="left", va="center", zorder=6)

    inset = ax.inset_axes([0.60, 0.28, 0.38, 0.30])
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
                       ms=MARKER_MS, lw=LW_HAIR, color=INK,
                       markerfacecolor=INK if filled else "white",
                       markeredgecolor=INK, markeredgewidth=LW_ERR,
                       elinewidth=LW_ERR, capsize=1.4, zorder=3)
    inset.axhline(0.0, color=MUTE, lw=LW_REF, dashes=(2.2, 1.8), zorder=1)
    inset.set_xticks((3000, 30000), ("3,000", "30,000"))
    inset.minorticks_off()
    # the inset sits over the region it magnifies, so its abscissa labels go
    # on top: below the frame they would land on the 45-cell 15,000 marker
    inset.set_yticks((0.0, 0.005), ("0", "0.005"))
    inset.tick_params(labelsize=PT_BASE, pad=1.2, length=1.8,
                      width=LW_EDGE, color=EDGE, labelcolor=INK,
                      labelbottom=False, labeltop=True, bottom=False,
                      top=True)
    # the magnified window, marked on the main axes directly under the inset
    ax.plot([2400, 40000, 40000, 2400, 2400],
            [-0.0062, -0.0062, 0.0082, 0.0082, -0.0062], color=MUTE,
            lw=LW_HAIR, zorder=1.4, solid_capstyle="butt")
    return drops


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
    ax.errorbar(x, mean, yerr=[mean - lo, hi - mean], marker=M_INITIAL,
                ms=MEAN_MS, lw=LW_DATA, color=INK, markerfacecolor=INK,
                markeredgecolor=INK, markeredgewidth=LW_ERR,
                elinewidth=LW_ERR, capsize=ERR_CAPSIZE, zorder=5)
    reference_line(ax, 0.0, axis="y", label="no contrast")
    for index, multiplier in enumerate((0, 1, 4)):
        ceiling = max(hi[index], per_cell[multiplier].max())
        ax.text(index, ceiling + 0.014, f"{positive[index]}/8",
                fontsize=PT_BASE, color=MUTE, ha="center", va="bottom",
                zorder=6)
    ax.set_xticks(x, ("0", "1", "4"))
    ax.set_xlim(-0.5, 2.5)
    ax.set_ylim(-0.075, 0.365)
    ax.set_yticks((0.0, 0.1, 0.2, 0.3), ("0", "0.1", "0.2", "0.3"))
    ax.tick_params(labelsize=PT_BASE, pad=0.8, length=2.0)
    ax.set_xlabel("background leak (\u00d7 baseline)", fontsize=PT_EMPH,
                  color=INK, labelpad=0.5)
    ax.set_ylabel(CONTRAST_LABEL, fontsize=PT_EMPH, color=INK, labelpad=0.5)
    ax.text(0.02, 0.968, "background conductance", transform=ax.transAxes,
            fontsize=PT_BASE, color=INK, ha="left", va="center", zorder=6)
    ax.text(0.02, 0.902, "restores the contrast", transform=ax.transAxes,
            fontsize=PT_BASE, color=INK, ha="left", va="center", zorder=6)
    ax.text(0.985, 0.095, "R = 15,000 \u03a9 cm\u00b2", transform=ax.transAxes,
            fontsize=PT_BASE, color=MUTE, ha="right", va="center", zorder=6)
    ax.text(0.985, 0.030, "normalized dose 1", transform=ax.transAxes,
            fontsize=PT_BASE, color=MUTE, ha="right", va="center", zorder=6)
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
    for rec in canvas._records:
        canvas.declare_reserve(rec["name"], left=reserve)
    canvas.lock_reserves()
    fills = measure()
    spread = max(fills.values()) / min(fills.values())
    return {"left_reserve_pt": round(reserve, 2),
            "slot_fill_spread": round(spread, 3),
            "slot_fill": {k: round(v, 3) for k, v in fills.items()}}


# ── the render-time display table ────────────────────────────────────────
def display_rows(category, dose, shapley, signed_summary, physical, sel,
                 gains):
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
                         value=float(v)) for v in values)
    for perturbation, points in dose.items():
        for value, mean, lo, hi in points:
            rows.append(dict(panel="D", record="cohort_mean",
                             perturbation=perturbation, dose=value, mean=mean,
                             ci95_low=lo, ci95_high=hi, n_cells=8))
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
    for cohort, _label, _marker, _filled in COHORTS:
        for rm, mean, lo, hi, n, values in physical[cohort]:
            rows.append(dict(panel="G", record="cohort_mean", cohort=cohort,
                             membrane_resistance_ohm_cm2=rm, mean=mean,
                             ci95_low=lo, ci95_high=hi, n_cells=n))
            rows.extend(dict(panel="G", record="cell_value", cohort=cohort,
                             membrane_resistance_ohm_cm2=rm, value=float(v))
                        for v in values)
    for record in sel.to_dict("records"):
        rows.append(dict(panel="H", record="cohort_mean",
                         background_leak_multiplier=record[
                             "background_leak_multiplier"],
                         mean=record["mean_shunt_minus_additive"],
                         ci95_low=record["ci95_low"],
                         ci95_high=record["ci95_high"],
                         n_cells=record["n_cells"],
                         cells_positive=record["cells_positive"]))
    return rows


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
    d_shunt = dict((v, m) for v, m, _lo, _hi in dose["focal shunt"])
    d_inject = dict((v, m) for v, m, _lo, _hi in dose["matched additive"])
    assert abs(d_shunt[1.0]
               - shapley["full_shunt_localization"].mean()) < 1e-12
    assert abs(d_inject[1.0]
               - shapley["matched_additive_localization"].mean()) < 1e-12

    canvas = NativeCanvas(490 / 72, 3, row_weights=[128, 116, 116],
                          hgutter_pt=37, vgutter_pt=41,
                          margins=Margins(left=49, right=12, top=23,
                                          bottom=25))
    canvas.letter_dx = LETTER_DX_PT
    a = canvas.panel("A", 0, 0, 6, schematic=True,
                     title="Focal shunt versus matched injection")
    b = canvas.panel("B", 0, 6, 6, schematic=True,
                     title="One exact gain per ancestry block")
    c = canvas.panel("C", 1, 0, 4, title="Descendants change most")
    d = canvas.panel("D", 1, 4, 4, grid="y", title="Contrast grows with dose")
    e = canvas.panel("E", 1, 8, 4, grid="y", title="Adjoint replacement",
                     sharey=d)
    g_f = canvas.panel("F", 2, 0, 4, title="Opposite signed changes")
    g = canvas.panel("G", 2, 4, 4, grid="y", title="State sets selectivity")
    h = canvas.panel("H", 2, 8, 4, grid="y", title="Background restores it")
    for ax in canvas.axes.values():
        # 3 pt of title pad plus the 5.4 pt letter clearance push the row-2
        # letters 14 pt above their axes and close the r1|r2 band to 6.7 pt
        ax.set_title(ax.get_title(), fontsize=PT_EMPH, color=INK, pad=1.0,
                     fontweight="normal")
    canvas.fig.canvas.draw()

    # data panels first: they set the labels the reserve lock measures
    panel_relations(canvas, c, category)
    panel_dose(d, dose)
    panel_adjoint(e, shapley)
    panel_signed(canvas, g_f, signed_summary, signed_cells)
    drops = panel_state(g, physical, ratio_ranges)
    panel_background(h, sel, per_cell)
    canvas.lock_reserves()
    balance = _balance_slot_fill(canvas)
    canvas.match_row_heights(1)
    canvas.match_row_heights(2)
    canvas.lock_reserves()
    canvas.fig.canvas.draw()

    panel_interventions(a)
    panel_gain_dictionary(b, gains)

    style_direct_color_labels(canvas.fig)
    output = J / "figures/components/focused_main_08.pdf"
    findings = canvas.save(output, name="focused_main_08", dpi=400)
    if findings:
        print("layout findings: " + "; ".join(str(x) for x in findings))
    plt.close(canvas.fig)
    main = J / "figures/main/figure_08.pdf"
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
        "(518.4 x 490 pt, rows 128/116/116, schematic fraction 27.2 % of the "
        "live canvas on the AMENDMENTS B12 formula; the 29.6 % row basis is "
        "the conservative bound). No new experiments and no change to any "
        "frozen source table: every estimate reuses the archived rows, "
        "estimators and bootstrap seeds. Panel D is the old S48 / S28D "
        "normalized-dose sheet promoted into the main figure, printed as "
        "equal-weighted cell means (dose 2 = 0.194, not the 0.203 pooled-site "
        "mean). Panel H is new artwork over the frozen "
        "focal_selectivity_phase1 table. Cohorts are encoded by marker shape "
        "and fill, not by ORDINAL_RAMP (DECISIONS, Figure 8). Slot-fill "
        f"slot-fill balance: {balance}. Contrast drop from "
        f"R_m 300 to 15,000: {drops}.")
    record = publish(8, output, display_rows(category, dose, shapley,
                                             signed_summary, physical, sel,
                                             gains),
                     sources, builders, panels, emit_main=emit_main,
                     layout_findings=[str(x) for x in findings], notes=notes)
    (J / "figures/provenance/credit_clarity_20260908"
     / "figure_08_layout.json").write_text(json.dumps(
         {"main_figure_sha256": sha(main), "slot_fill_balance": balance,
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
