#!/usr/bin/env python3
"""Main Fig. 3 (fig:subtreefactorial): hierarchical distractors and ancestry bandwidth.

Rebuilt 2026-09-08 on the NativeCanvas per analysis/figure_overhaul_20260908/
DESIGN_SPEC.md §3 (labels per SPEC_ERRATA.md).  Three rows, seven panels:

    row 0  A  eight-stream task (4 modules)      B  dictionary ladder K = 1..8 (8)
    row 1  C  accuracy across K (4)  D  rewiring (4, sharey)  E  coefficient source (4, sharey)
    row 2  F  matched controls at K = 4 (5, schematic)   G  K = 4 forest (7, 62-pt reserve)

Recorded deviations from the specification (each with its reason):

* Canvas 490 pt, rows 122/108/108, vgutter 48 (spec: 492, 124/116/116, 40).
  audit_row_separation.py needs >= 8.5 pt of blank band between rows; the
  x tick labels + x label of row 1 (about 25 pt) and the letter + title of
  row 2 (about 17 pt) leave < 3 pt at a 40-pt gutter.  Fig. 2 met the floor
  with the same 48-pt gutter, so this figure uses it too.  Row 0 keeps B
  inside the 2.40 panel-aspect ceiling (290 x 122 pt = 2.38).
* Left margin 52 (spec 46): with 46 the y label of C entered the 16.8-pt
  letter column that audit_letter_alignment.py protects.
* F is one ghost tree carrying the correct-ancestry delivery plus a
  4 x 8 measured-support table (rows = the four controls, columns = the
  eight streams) instead of four 38-pt cards.  A 38-pt card cannot hold a
  control name at PT_SMALL ("depth-interleaved" is 41 pt wrapped) beside
  an eight-terminal tree with a legible pitch; the support strips the
  spec asked for under each card are the table's rows.
* D prints the short paired-difference tags "+23.2 pp" / "+5.0 pp"; the
  95% intervals are in the caption (a 20-character interval tag has no
  clear whitespace inside a 4-module accuracy panel).
* Private helpers (helpers not present in scripts/native_schematics.py):
  ``_badge`` adds the ``ceiling`` and ``exploratory`` badge kinds with the
  library's badge geometry; ``_subtree_arrow`` draws the addressed-subtree
  entry arrow with ``Frame.credit_delivery``'s geometry but on capsules
  tinted from the D7 K-cycle (dend / soma / exc / mute) rather than the
  library's legacy shunting / additive / local / oracle cycle.

Column-lock waiver (D3): row 1 is three 4-module panels sharing the
held-out-accuracy axis (C carries the label, D and E hide their tick labels).
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import shutil
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
JOURNAL = HERE.parents[1]
sys.path.insert(0, str(JOURNAL / "scripts"))
import routing_figure_panels as routing                       # noqa: E402
import run_trained_subtree_address_full_factorial as experiment  # noqa: E402
from figure_canvas import (COLORS, LW_DATA, LW_EDGE, LW_ERR, LW_HAIR, LW_REF,  # noqa: E402
                           MARKER_MS, PT_ANNOT, PT_LABEL, PT_SMALL, PT_TICK,
                           Margins, NativeCanvas)
from journal_style import (ERR_CAPSIZE, SEED_ALPHA, SEED_MS, label_color,  # noqa: E402
                           style_direct_color_labels, wrap_ticklabels)
from native_schematics import (BADGE_STYLE, CONTACT_DIA_PT, Frame,  # noqa: E402
                               LINE_BAND_PT, _text_w_pt, mix, reference_line)

SOURCE = JOURNAL / "source_data"
DATA = SOURCE / "trained_subtree_address_full_factorial"
REVIEW = SOURCE / "review_evidence_reanalysis"
ENCODER = SOURCE / "review_coefficient_encoder"
HARD = SOURCE / "review_coefficient_hard_readout"
RECORDS = SOURCE / "credit_first_figures"
CONFIG = JOURNAL / "configs/trained_subtree_address/full_factorial_confirmatory.json"
OUT = JOURNAL / "figures/components/credit_first_figure_03.pdf"
PUBLISHED = JOURNAL / "figures/main/figure_03.pdf"

INK, MUTE, GREEN, PURPLE = COLORS["ink"], COLORS["mute"], COLORS["shunting"], COLORS["oracle"]
GREY = COLORS["point_mlp"]
K_TICKS = (1, 2, 4, 8)
CUED = 2                       # zero-based index of the cued stream c3
E_CELL = dict(calibration_samples=256, cue_noise_sd=0.5, cue_delay_trials=0, epoch=80)
E_ORDER = (("oracle", "oracle_context", "enc", "oracle"),
           ("soft", "learned_local_cue", "enc", "shunting"),
           ("hard*", "learned_local_cue", "hard", "shunting"),
           ("frozen", "frozen_profile", "enc", "per_soma"))
CONTROLS_F = (("deranged", "within_neuron_route_derangement"),
              ("depth-\ninterleaved", "depth_interleaved_bins"),
              ("random sparse", "random_sparse_matched"),
              ("dense rank-4", "random_rank_k"))
CONTROLS_G = (("best matched control", "best_matched_nonanatomical_oracle", "oracle"),
              ("dense rank-4", "random_rank_k", "shunting"),
              ("random sparse", "random_sparse_matched", "shunting"),
              ("depth interleaved", "depth_interleaved_bins", "shunting"))
Y_LIM = (0.10, 0.92)

# D7 K-cycle: four hues that are never data series; 62 % ink mixes for k >= 4.
# Reserved for B's group identity (and F's partition capsule) alone.
K_CYCLE = ("dend", "soma", "exc", "mute")
# D8 ordinal ramp for A's coefficient tiers (|0.15| < |0.45| < |0.75|).
# journal_style.py does not register ORDINAL_RAMP yet and a per-figure task
# may not edit it, so this builder defines the same three values locally,
# exactly as scripts/build_main_figure_04.py (Fig. 2) does.
ORDINAL_RAMP = (mix("edge", 45), mix("edge", 75), COLORS["edge"])
CAPSULE_TINT_PCT = 58          # tint of A's ordinal tier band (see _tier_bands)
TIER_BAND_W_PT = 5.6           # fixed width of an A tier band
LIFT_PT = 4.2                     # stream-label lift above its terminal (pt)
# (terminal, sideways nudge in pt): the four streams panel A argues about,
# each nudged along the canopy so its label hugs its own tip.
A_STREAM_LABELS = ((0, -2.2), (2, -2.4), (3, 3.6), (7, 2.2))


def _k_hue(k):
    key = K_CYCLE[k % 4]
    return COLORS[key] if k < 4 else mix(key, 62, "ink")


def _signed(v, digits=2):
    return f"{v:+.{digits}f}".replace("-", "−")


def _minus(s):
    return str(s).replace("-", "−")


# ── data ─────────────────────────────────────────────────────────────────
def coefficient_prediction():
    cfg = json.loads(CONFIG.read_text())["task"]
    n = cfg["contexts"]
    rows = []
    for k in K_TICKS:
        routes = experiment.grouped_routes(np.arange(n), k, "correct_ancestry_subtrees",
                                           np.random.default_rng(0))
        for context in range(n):
            amplitudes = np.array([cfg["selected_signal"] if stream == context else
                                   -cfg["distractor_signal_by_tree_distance"][
                                       experiment.tree_relation(stream, context)]
                                   for stream in range(n)])
            mask = routes[context] > 0
            raw = float(amplitudes[mask].sum())
            delivered = float(routes[context] @ amplitudes)
            np.testing.assert_allclose(delivered, raw / np.sqrt(mask.sum()), atol=1e-14)
            rows.append(dict(budget_k=k, context=context, group_size=int(mask.sum()),
                             raw_coefficient_sum=raw, normalized_coefficient_sum=delivered,
                             normalization="unit Euclidean norm per route row"))
    table = pd.DataFrame(rows)
    assert table.groupby("budget_k").raw_coefficient_sum.std().max() < 1e-14
    np.testing.assert_allclose(table.groupby("budget_k").raw_coefficient_sum.mean(),
                               [-3.05, -.05, .85, 1.0], atol=1e-14)
    return table


def task_tiers():
    """Per-tier class coefficients of the frozen task configuration."""
    cfg = json.loads(CONFIG.read_text())["task"]
    d = cfg["distractor_signal_by_tree_distance"]
    return dict(selected=float(cfg["selected_signal"]), sibling=-float(d["sibling"]),
                same_half=-float(d["same_half"]), other_half=-float(d["opposite_half"]))


def ancestry_dictionary(K):
    """A (8 x K): the K distinct unit-norm route rows of the correct ancestry field."""
    routes = experiment.grouped_routes(np.arange(8), K, "correct_ancestry_subtrees",
                                       np.random.default_rng(0))
    columns = []
    for row in routes:
        if not any(np.allclose(row, c) for c in columns):
            columns.append(row)
    A = np.stack(columns, axis=1)
    assert A.shape == (8, K)
    return A


def control_supports(rng_seed=0):
    """What each K = 4 control delivers when c3 is cued: one draw per random family."""
    rng = np.random.default_rng(rng_seed)
    rows = []
    for name, mode in CONTROLS_F:
        if mode == "random_rank_k":
            field = experiment.random_rank_routes(np.arange(8), 4, rng)
        else:
            field = experiment.grouped_routes(np.arange(8), 4, mode, rng)
        rows.append(field[CUED])
    table = pd.DataFrame(np.array(rows), columns=[f"c{i + 1}" for i in range(8)])
    table.insert(0, "control", [m for _, m in CONTROLS_F])
    table.insert(1, "cued_stream", "c3")
    return table


def coefficient_source_table(enc, hard):
    """Per-seed held-out accuracy of the four K = 4 coefficient sources (fresh cohort)."""
    out = []
    for label, method, which, colour in E_ORDER:
        t = enc if which == "enc" else hard
        sel = t[(t.method.eq(method))
                & np.all([t[k].eq(v) for k, v in E_CELL.items()], axis=0)]
        assert len(sel) == 20, (label, len(sel))
        for _, r in sel.sort_values("seed").iterrows():
            out.append(dict(source=label.rstrip("*"), method=method,
                            experiment="review_coefficient_hard_readout" if which == "hard"
                            else "review_coefficient_encoder", seed=int(r.seed),
                            heldout_accuracy=float(r.heldout_accuracy), **E_CELL))
    return pd.DataFrame(out)


def contrast_row(table, control):
    sel = table[table.control.eq(control)
                & np.all([table[k].eq(v) for k, v in E_CELL.items() if k != "epoch"], axis=0)]
    assert len(sel) == 1
    return sel.iloc[0]


# ── private glyph helpers (geometry copied from native_schematics) ────────
_EXTRA_BADGES = {
    "ceiling": ("oracle", mix("oracle", 8), mix("oracle", 45)),
    "exploratory": ("mute", COLORS["panel_bg"], COLORS["grid"]),
}


def _badge(ax, xy, kind, *, text=None, ha="right", va="top", transform=None,
           zorder=7):
    """Frame.badge geometry for every kind, plus D6 'ceiling' and 'exploratory'."""
    key, face, edge = {**BADGE_STYLE, **_EXTRA_BADGES}[kind]
    colour = COLORS[key]
    try:
        colour = label_color(colour, background=face)
    except ValueError:
        pass
    kw = {} if transform is None else {"transform": transform}
    return ax.text(xy[0], xy[1], kind if text is None else text, fontsize=PT_SMALL,
                   color=colour, ha=ha, va=va, zorder=zorder,
                   bbox=dict(boxstyle=f"round,pad=0.28,rounding_size={2.0 / PT_SMALL:.3f}",
                             facecolor=face, edgecolor=edge, linewidth=LW_HAIR), **kw)


def _tier_bands(frame, nodes, blocks, colors, *, pct=CAPSULE_TINT_PCT,
                width_pt=TIER_BAND_W_PT):
    """Frame.partition's capsule geometry at a fixed, narrower width.

    ``Frame.partition`` scales its capsule with the terminal pitch (5.6 pt
    for one terminal, 10 pt for four), so an ORDINAL_RAMP tint that is
    legible under the one-terminal sibling band turns the four-terminal
    other-half band into a grey slab.  Same chains, same zorder, one width.
    """
    for block, colour in zip(blocks, colors):
        members = set(block)
        chains = [[nodes[nodes.parent[n]], nodes[n]] for n in block
                  if nodes.parent.get(n) in members]
        if not chains:
            chains = [[nodes[block[0]], nodes[block[0]]]]
        frame._draw_chains(chains, mix(colour, pct), width_pt)


def _lerp(a, b, t):
    return (a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]))


def _subtree_arrow(frame, nodes, target, color):
    """Entry arrow into the addressed subtree root (credit_delivery geometry)."""
    par = nodes.parent.get(target)
    p0 = nodes[par] if par in nodes else nodes.soma
    frame.arrow(_lerp(p0, nodes[target], 0.40), _lerp(p0, nodes[target], 0.80),
                color=color, lw=LW_EDGE, head=4.0, zorder=4.6)


def _gap_span(ax, x, y_lo, y_hi, tag, tag_xy, *, ha="left", dx=0.07, cap=0.06):
    """Double-headed span between one paired pair, tagged in clear whitespace.

    The Fig. 6 C leader idiom applied to a gap: hairline caps on the two
    paired markers, a mute double arrow between them and a leader from the
    middle of that span to the PT_ANNOT tag, so the number annotates the
    difference and not either marker.
    """
    xs = x + dx
    for y in (y_lo, y_hi):
        ax.plot([x, xs + cap], [y, y], color=MUTE, lw=LW_HAIR, zorder=1)
    ax.annotate("", xy=(xs, y_hi), xytext=(xs, y_lo),
                arrowprops=dict(arrowstyle="<->", color=MUTE, lw=LW_HAIR,
                                shrinkA=0.0, shrinkB=0.0, mutation_scale=6.0),
                zorder=1)
    mid = 0.5 * (y_lo + y_hi)
    ax.plot([xs + cap, tag_xy[0] - 0.04], [mid, tag_xy[1]], color=MUTE,
            lw=LW_HAIR, zorder=1)
    return ax.text(tag_xy[0], tag_xy[1], tag, fontsize=PT_ANNOT, color=INK,
                   ha=ha, va="center")


def _group_root(nodes, K, terminal="T3"):
    """Root of the K-partition block that contains ``terminal``."""
    for root in nodes.at_depth(K):
        if terminal in nodes.terminals_under(root):
            return root
    raise KeyError(terminal)


def _fan(n, width=0.36, seed=7):
    return np.random.default_rng(seed).permutation(np.linspace(-width / 2, width / 2, n))


def _mean_marker(ax, x, values, color, marker="o", seed=0, ms=MARKER_MS, hollow_seeds=False,
                 fan_width=0.36):
    mean, low, high = routing.bootstrap(np.asarray(values, float), seed)
    ax.plot(x + _fan(len(values), fan_width, seed + 1), values, ls="none", marker="o",
            ms=SEED_MS, mfc="none" if hollow_seeds else color, mec=color if hollow_seeds
            else "none", mew=LW_EDGE if hollow_seeds else 0.0, alpha=SEED_ALPHA, zorder=2)
    ax.errorbar([x], [mean], yerr=[[mean - low], [high - mean]], fmt=marker, color=color,
                mfc="white", mec=color, mew=LW_EDGE, ms=ms, elinewidth=LW_ERR,
                capsize=ERR_CAPSIZE, capthick=LW_ERR, zorder=4)
    return mean, low, high


def _series(ax, x, part, color, marker, ms, zorder):
    mean = part.mean_heldout_accuracy.to_numpy(float)
    low = part.ci95_low_heldout_accuracy.to_numpy(float)
    high = part.ci95_high_heldout_accuracy.to_numpy(float)
    ax.errorbar(x, mean, yerr=[mean - low, high - mean], color=color, marker=marker,
                mfc="white", mec=color, mew=LW_EDGE, ms=ms, lw=LW_DATA, elinewidth=LW_ERR,
                capsize=ERR_CAPSIZE, capthick=LW_ERR, zorder=zorder)
    return mean


def _accuracy_axes(ax, *, ylabel):
    ax.set_xlim(-0.18, 3.18)
    ax.set_ylim(*Y_LIM)
    ax.set_xticks(range(4), [str(k) for k in K_TICKS])
    ax.set_yticks([0.2, 0.5, 0.8])
    ax.set_xlabel("Channels, K")
    if ylabel:
        ax.set_ylabel("Held-out accuracy")
    else:
        ax.tick_params(axis="y", labelleft=False)


# ── A: eight-stream hierarchical task ────────────────────────────────────
def panel_task(ax, tiers):
    f = Frame(ax)
    W, H = f.w_pt, f.h_pt
    sub_pt, rows_pt, foot_pt, delta_pt = LINE_BAND_PT, 19.0, LINE_BAND_PT, 12.0
    f.text((0.5, 1.0 - f.fy(sub_pt * 0.5)), "context selects the +sᵢ stream", size=PT_ANNOT)
    footer = "junctions define feedback supports only"
    if _text_w_pt(ax, footer, PT_SMALL) > W - 4.0:
        footer = "junctions define supports only"
    f.text((0.5, f.fy(foot_pt * 0.5)), footer, size=PT_SMALL, color=MUTE)
    head_pt = sub_pt + rows_pt + LIFT_PT + 8.0      # key band + label ring
    rect = (0.0, f.fy(foot_pt + delta_pt), 1.0,
            1.0 - f.fy(head_pt + foot_pt + delta_pt))
    nodes = f.balanced_tree(rect, mode="plain", output="z")
    # The three distractor tiers are ordinal factor levels of one coefficient
    # (0.15 < 0.45 < 0.75), so they take ORDINAL_RAMP (D8); K_CYCLE stays the
    # alphabet of B's group identity, one hgutter to the right.
    blocks = [["JLR", "T4"], ["JLL", "T1", "T2"],
              ["JR", "JRL", "JRR", "T5", "T6", "T7", "T8"]]
    _tier_bands(f, nodes, blocks, ORDINAL_RAMP)
    for a, b in zip(nodes.route("T3"), nodes.route("T3")[1:]):
        if (a, b) in nodes.edges:
            nodes.edges[(a, b)].set_linewidth(f.lw(LW_DATA))
    f.contact(nodes["T3"], kind="exc")
    f.soma(nodes.soma, output=True, label="z")
    f.error_in(nodes.soma, side="right")
    # The terminal pitch (about 10 pt) is below the one-label-per-terminal
    # floor for PT_SMALL: eight labels clear each other by only 2-4 pt and
    # read as one cluster.  Label the four streams the panel argues about --
    # the ends c1 and c8, the cued c3 and its sibling c4 -- with c4 lifted
    # into the second band so the cued pair parts; the tier key carries the
    # rest and F names every stream in its table.
    for i, push in A_STREAM_LABELS:
        f.text(f._off(nodes[nodes.terminals[i]], push, LIFT_PT),
               "c" + "₁₂₃₄₅₆₇₈"[i], size=PT_SMALL,
               color=COLORS["exc"] if i == CUED else INK)
    # tier key: name over swatch + coefficient; the swatch keys the capsules
    y_name = 1.0 - f.fy(sub_pt + 5.0)
    y_coef = 1.0 - f.fy(sub_pt + rows_pt - 5.0)
    entries = [("cued", _signed(tiers["selected"], 0), COLORS["exc"], "dot"),
               ("sibling", _signed(tiers["sibling"]), ORDINAL_RAMP[0], "bar"),
               ("same half", _signed(tiers["same_half"]), ORDINAL_RAMP[1], "bar"),
               ("other half", _signed(tiers["other_half"]), ORDINAL_RAMP[2], "bar")]
    sw_pt, sw_gap = 5.0, 2.0
    widths = [max(_text_w_pt(ax, n, PT_SMALL),
                  sw_pt + sw_gap + _text_w_pt(ax, c, PT_SMALL))
              for n, c, _, _ in entries]
    gap = (W - 4.0 - sum(widths)) / (len(entries) - 1)
    assert gap >= 3.0, gap
    x = 2.0
    for (name, coef, colour, kind), w in zip(entries, widths):
        f.text((f.fx(x + w / 2.0), y_name), name, size=PT_SMALL, color=INK)
        run = sw_pt + sw_gap + _text_w_pt(ax, coef, PT_SMALL)
        sx = x + (w - run) / 2.0
        if kind == "dot":
            ax.plot([f.fx(sx + sw_pt / 2.0)], [y_coef], ls="none", marker="o",
                    ms=CONTACT_DIA_PT, mfc=colour, mec="none", zorder=6)
        else:
            ax.add_patch(Rectangle((f.fx(sx), y_coef - f.fy(1.7)), f.fx(sw_pt),
                                   f.fy(3.4), facecolor=mix(colour, CAPSULE_TINT_PCT),
                                   edgecolor=COLORS["grid"], linewidth=LW_HAIR,
                                   zorder=6))
        f.text((f.fx(sx + sw_pt + sw_gap), y_coef), coef, size=PT_SMALL, color=INK,
               ha="left")
        x += w + gap
    return nodes


# ── B: ancestry dictionaries K = 1 .. 8 ──────────────────────────────────
def panel_dictionaries(ax, prediction):
    f = Frame(ax)
    sums = prediction.groupby("budget_k").raw_coefficient_sum.mean()
    cells = f.split(4, axis="x", gap_pt=6.0,
                    pad_pt=(0.0, 0.0, LINE_BAND_PT, 0.0))
    f.text((1.0, 1.0 - f.fy(LINE_BAND_PT * 0.5)),
           "matrix rows = site blocks in tree order", size=PT_SMALL, color=MUTE,
           ha="right")
    for cell, K in zip(cells, K_TICKS):
        core = f.task_card(cell, title=f"K = {K}", footer=f"group sum {_signed(sums[K])}",
                           emphasis=(K == 4))
        x0, y0, w, h = core
        mat_pt = 48.0
        tree_rect = (x0, y0 + f.fy(mat_pt + 4.0), w, h - f.fy(mat_pt + 4.0))
        site_colors = [_k_hue(k) for k in range(8)] if K == 8 else None
        nodes = f.balanced_tree(tree_rect, mode="plain", labels=False,
                                site_colors=site_colors)
        if K < 8:
            roots = nodes.at_depth(K)
            f.partition(nodes, [nodes.subtree(r) for r in roots],
                        colors=[K_CYCLE[k % 4] for k in range(len(roots))])
        _subtree_arrow(f, nodes, _group_root(nodes, K), GREEN)
        f.contact(nodes["T3"], kind="exc")
        mx, my = x0 + f.fx(6.0), y0 + f.fy(2.0)
        if K < 8:
            A = ancestry_dictionary(K)
            rect = (mx, my, f.fx(6.0 * K), f.fy(mat_pt))
            f.dictionary_matrix(rect, A, color="shunting", label=None,
                                row_groups=[8 // K] * K if K > 1 else None)
            for j in range(K):                       # column-header swatches
                ax.add_patch(Rectangle((mx + f.fx(6.0 * j + 1.0), my + f.fy(mat_pt + 1.2)),
                                       f.fx(4.0), f.fy(2.0), facecolor=_k_hue(j),
                                       edgecolor="none", zorder=3))
            tx = mx + f.fx(6.0 * K + 5.0)
            f.text((tx, my + f.fy(mat_pt * 0.5 + 4.5)), "A", size=PT_ANNOT, ha="left")
            f.text((tx, my + f.fy(mat_pt * 0.5 - 4.5)), f"(8 × {K})", size=PT_SMALL,
                   ha="left")
        else:
            f.text((x0 + w / 2.0, my + f.fy(mat_pt * 0.5 + 4.5)), "A = I", size=PT_ANNOT)
            f.text((x0 + w / 2.0, my + f.fy(mat_pt * 0.5 - 4.5)), "(8 × 8)", size=PT_SMALL)
    return sums


# ── C: held-out accuracy across K ────────────────────────────────────────
def panel_bandwidth(ax, summary, outcomes):
    x = np.arange(4, dtype=float)
    dend = summary[summary.architecture.eq("dendritic_tree")]
    best = routing._best_control_by_budget(outcomes)          # per-seed max of 4 controls
    part = lambda fam: dend[dend.feedback_family.eq(fam)].set_index("budget_k").loc[list(K_TICKS)]
    deranged = _series(ax, x, part("within_neuron_route_derangement"), GREY, "v",
                       MARKER_MS + 1.2, 3)
    ax.errorbar(x, best[:, 1], yerr=[best[:, 1] - best[:, 2], best[:, 3] - best[:, 1]],
                color=PURPLE, marker="^", mfc="white", mec=PURPLE, mew=LW_EDGE,
                ms=MARKER_MS + 1.6, lw=LW_DATA, elinewidth=LW_ERR, capsize=ERR_CAPSIZE,
                capthick=LW_ERR, zorder=2)
    ancestry = _series(ax, x, part("correct_ancestry_subtrees"), GREEN, "o",
                       MARKER_MS - 0.3, 5)
    np.testing.assert_allclose(ancestry, [0.187, 0.449, 0.801, 0.810], atol=2e-3)
    np.testing.assert_allclose(best[:, 1], [0.545, 0.603, 0.788, 0.810], atol=2e-3)
    np.testing.assert_allclose(deranged, [0.187, 0.185, 0.191, 0.251], atol=2e-3)
    _accuracy_axes(ax, ylabel=True)
    reference_line(ax, 0.5, label="chance", span=(-0.18, 3.18))
    ax.text(2.1, 0.86, "ancestry", color=GREEN, fontsize=PT_SMALL, ha="left", va="bottom")
    ax.text(0.02, 0.735, "best control", color=PURPLE, fontsize=PT_SMALL, ha="left",
            va="bottom")
    _badge(ax, (0.02, 0.86), "ceiling", ha="left", va="center")
    ax.text(3.05, 0.29, "deranged", color=GREY, fontsize=PT_SMALL, ha="right", va="bottom")
    return dict(ancestry=ancestry.tolist(), best=best[:, 1].tolist(), deranged=deranged.tolist())


# ── D: rewiring at K = 2, 4 ──────────────────────────────────────────────
def panel_rewiring(ax, summary, outcomes):
    x = np.arange(4, dtype=float)
    fam = summary[summary.feedback_family.eq("correct_ancestry_subtrees")]
    part = lambda arch: fam[fam.architecture.eq(arch)].set_index("budget_k").loc[list(K_TICKS)]
    rewired = _series(ax, x, part("degree_depth_matched_rewired_tree"), GREY, "^",
                      MARKER_MS + 1.4, 3)
    matched = _series(ax, x, part("dendritic_tree"), GREEN, "o", MARKER_MS - 0.3, 5)
    correct = outcomes[outcomes.feedback_family.eq("correct_ancestry_subtrees")]
    paired = {}
    for index, K in enumerate(K_TICKS):
        a = correct[correct.architecture.eq("dendritic_tree") & correct.budget_k.eq(K)] \
            .set_index("seed").heldout_accuracy
        b = correct[correct.architecture.eq("degree_depth_matched_rewired_tree")
                    & correct.budget_k.eq(K)].set_index("seed").heldout_accuracy
        diff = (a - b).to_numpy(float)
        mean, low, high = routing.bootstrap(diff, 70_000 + index)
        paired[K] = (100 * mean, 100 * low, 100 * high, bool(np.all(diff == 0)))
    assert paired[1][3] and paired[8][3], "K = 1 and K = 8 must tie exactly"
    np.testing.assert_allclose([paired[2][0], paired[4][0]], [23.2, 5.0], atol=0.1)
    _accuracy_axes(ax, ylabel=False)
    reference_line(ax, 0.5, label="chance", span=(-0.18, 3.18))
    # each tag annotates the vertical gap between the two paired markers
    _gap_span(ax, 1.0, rewired[1], matched[1], f"{_signed(paired[2][0])} pp",
              (1.36, 0.305))
    _gap_span(ax, 2.0, rewired[2], matched[2], f"{_signed(paired[4][0])} pp",
              (2.32, 0.862))
    ax.text(0.05, 0.865, "matched tree", color=GREEN, fontsize=PT_SMALL, ha="left",
            va="center")
    ax.text(2.1, 0.70, "rewired", color=GREY, fontsize=PT_SMALL, ha="left", va="center")
    _badge(ax, (2.1, 0.625), "control", ha="left", va="center")
    ax.text(3.12, 0.13, "ties at K = 1, 8", color=MUTE, fontsize=PT_SMALL, ha="right",
            va="bottom")
    return {K: v[:3] for K, v in paired.items()}


# ── E: coefficient source at K = 4 (fresh cohort) ────────────────────────
def panel_coefficients(ax, source, contrasts_enc, contrasts_hard):
    stats = {}
    for i, (label, method, which, colour) in enumerate(E_ORDER):
        vals = source[source.source.eq(label.rstrip("*"))].sort_values("seed") \
            .heldout_accuracy.to_numpy(float)
        stats[label] = _mean_marker(ax, i, vals, COLORS[colour], seed=90_000 + i,
                                    hollow_seeds=(which == "hard"))
    np.testing.assert_allclose([stats[k][0] for k in ("oracle", "soft", "hard*", "frozen")],
                               [0.803, 0.258, 0.640, 0.187], atol=2e-3)
    vs_oracle = contrast_row(contrasts_enc, "oracle_context")
    vs_frozen = contrast_row(contrasts_enc, "frozen_profile")
    hard_vs_oracle = contrast_row(contrasts_hard, "oracle_context")
    assert pd.isna(hard_vs_oracle.get("primary_family_holm_p", np.nan)) or \
        "analysis_status" in contrasts_hard.columns
    ax.set_xlim(-0.55, 3.45)
    ax.set_ylim(*Y_LIM)
    ax.set_xticks(range(4), [e[0] for e in E_ORDER])
    ax.set_yticks([0.2, 0.5, 0.8])
    ax.tick_params(axis="y", labelleft=False)
    ax.set_xlabel("Coefficient source")
    reference_line(ax, 0.5, label="chance", span=(-0.55, 3.45))
    _badge(ax, (0.0, 0.875), "oracle", ha="center", va="center")
    _badge(ax, (2.0, 0.765), "exploratory", ha="center", va="center")
    # both contrasts are of the soft estimator: they sit in the whitespace
    # below chance, right-aligned, with one leader onto the soft cluster
    span_x = ax.get_position().width * 518.4
    left = 3.42
    for y, name, row in ((0.445, "oracle", vs_oracle), (0.355, "frozen", vs_frozen)):
        tag = f"soft vs {name} {_signed(row.mean_pp)} pp"
        if _text_w_pt(ax, tag, PT_ANNOT) > 0.80 * span_x:
            tag = f"vs {name} {_signed(row.mean_pp)} pp"
        ax.text(3.42, y, tag, fontsize=PT_ANNOT, color=INK, ha="right", va="center")
        left = min(left, 3.42 - _text_w_pt(ax, tag, PT_ANNOT) / span_x * 4.0)
    ax.plot([left - 0.04, 1.02], [0.355, 0.300], color=MUTE, lw=LW_HAIR, zorder=1)
    ax.text(3.4, 0.875, "*outside the Holm family", color=MUTE, fontsize=PT_SMALL,
            ha="right", va="center")
    seeds = source.seed
    ax.text(3.4, 0.125, f"20 fresh seeds {seeds.min()}–{seeds.max()}", color=MUTE,
            fontsize=PT_SMALL, ha="right", va="center")
    return dict(means={k: v[0] for k, v in stats.items()},
                soft_minus_oracle=[float(vs_oracle.mean_pp), float(vs_oracle.ci95_low_pp),
                                   float(vs_oracle.ci95_high_pp),
                                   float(vs_oracle.primary_family_holm_p)],
                soft_minus_frozen=[float(vs_frozen.mean_pp), float(vs_frozen.ci95_low_pp),
                                   float(vs_frozen.ci95_high_pp),
                                   float(vs_frozen.primary_family_holm_p)],
                hard_minus_oracle=[float(hard_vs_oracle.mean_pp),
                                   float(hard_vs_oracle.ci95_low_pp),
                                   float(hard_vs_oracle.ci95_high_pp)])


# ── F: matched controls at K = 4 ─────────────────────────────────────────
def panel_controls(ax, supports):
    f = Frame(ax)
    W, H = f.w_pt, f.h_pt
    sub_pt = LINE_BAND_PT
    f.text((0.0, 1.0 - f.fy(sub_pt * 0.5)), "correct ancestry delivers c₃ + sibling c₄",
           size=PT_ANNOT, ha="left")
    body_top = 1.0 - f.fy(sub_pt + 2.0)
    # left: the ghost tree with the correct K = 4 delivery to c3's group
    tree_w = 54.0
    tree_rect = (0.0, f.fy(30.0), f.fx(tree_w), body_top - f.fy(30.0))
    nodes = f.balanced_tree(tree_rect, mode="plain", ghost=True, output="z",
                            input_labels=[("c", str(i + 1)) for i in range(8)])
    root = _group_root(nodes, 4)
    f.partition(nodes, [nodes.subtree(root)], colors=("soma",))
    _subtree_arrow(f, nodes, root, GREEN)
    for t in nodes.terminals_under(root):
        f.terminal(nodes[t], site_color=GREEN)
    f.contact(nodes["T3"], kind="exc")
    f.soma(nodes.soma, output=True, label="z")
    f.error_in(nodes.soma, side="right")
    # right: 4 x 8 support table, rows = controls, columns = streams
    n_rows, n_cols = len(CONTROLS_F), 8
    row_pt, col_pt = 15.0, 7.0
    label_pt = 46.0
    mx = f.fx(tree_w + 6.0 + label_pt)
    assert (tree_w + 8.0 + label_pt + col_pt * n_cols) <= W - 2.0
    mw, mh = f.fx(col_pt * n_cols), f.fy(row_pt * n_rows)
    head_pt, foot_pt = 10.0, 10.0
    body_pt = (body_top - 0.0) * H
    my = f.fy((body_pt - (mh * H + head_pt + foot_pt)) / 2.0 + foot_pt)
    A = np.abs(supports[[f"c{i + 1}" for i in range(8)]].to_numpy(float))
    A = A / A.max(axis=1, keepdims=True)
    A = np.where(A > 0, 0.25 + 0.75 * A, 0.0)   # print floor: graded cells survive
    inner = f.dictionary_matrix((mx, my, mw, mh), A, color="point_mlp", label=None,
                                row_groups=[1] * n_rows,
                                yticks=[name for name, _ in CONTROLS_F])
    inner.tick_params(axis="y", labelsize=PT_SMALL, pad=2.0)
    for j in range(n_cols):                       # every stream is named
        f.text((mx + f.fx(col_pt * (j + 0.5)), my + mh + f.fy(head_pt * 0.5)),
               str(j + 1), size=PT_SMALL,
               color=COLORS["exc"] if j == CUED else INK)
    f.text((mx - f.fx(2.0), my + mh + f.fy(head_pt * 0.5)), "c", size=PT_SMALL,
           color=MUTE, ha="right")
    _badge(ax, (mx - f.fx(label_pt) - f.fx(2.0), my + mh + f.fy(head_pt * 0.5 + 1.0)),
           "control", ha="left", va="center")
    f.text((1.0, f.fy(4.5)), "random, dense: one illustrative draw", size=PT_SMALL,
           color=MUTE, ha="right")
    return nodes


# ── G: K = 4 forest ──────────────────────────────────────────────────────
def panel_forest(ax, contrasts, pairs):
    rows = {}
    for y, (label, key, colour) in zip([3, 2, 1, 0], CONTROLS_G):
        row = contrasts[contrasts.control.eq(key)].iloc[0]
        values = pairs[pairs.control.eq(key)].accuracy_difference_pp.to_numpy(float)
        assert len(values) == 20
        c = COLORS[colour]
        ax.plot(values, y + _fan(len(values), 0.30, 11 + y), ls="none", marker="o",
                ms=SEED_MS, mfc=c, mec="none", alpha=SEED_ALPHA, zorder=2)
        ax.errorbar([row.mean_difference_pp], [y],
                    xerr=[[row.mean_difference_pp - row.ci95_low_pp],
                          [row.ci95_high_pp - row.mean_difference_pp]],
                    fmt="D", color=c, mfc="white", mec=c, mew=LW_EDGE, ms=MARKER_MS,
                    elinewidth=LW_ERR, capsize=ERR_CAPSIZE, capthick=LW_ERR, zorder=4)
        rows[key] = row
    hero = rows["best_matched_nonanatomical_oracle"]
    derangement = contrasts[contrasts.control.eq("within_neuron_route_derangement")].iloc[0]
    ax.set_xlim(-2.0, 14.0)
    ax.set_ylim(-0.55, 3.75)
    ax.set_xticks([0, 4, 8, 12])
    ax.set_yticks([3, 2, 1, 0], wrap_ticklabels([c[0] for c in CONTROLS_G], width=12))
    ax.tick_params(axis="y", length=0, labelsize=PT_SMALL)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Ancestry − control (percentage points)")
    reference_line(ax, 0.0, axis="x", label=None, span=(-0.55, 3.2))
    ax.text(-0.15, -0.52, "zero", fontsize=PT_SMALL, color=MUTE, rotation=90, ha="right",
            va="bottom")
    ax.text(13.9, 3.58, f"{_signed(hero.mean_difference_pp)} [{hero.ci95_low_pp:.2f}, "
            f"{hero.ci95_high_pp:.2f}] pp", fontsize=PT_ANNOT, color=INK, ha="right",
            va="center")
    ax.text(13.9, 3.14, f"{int(hero.positive_seeds)}/{int(hero.n_pairs)} seeds, "
            f"Holm P = {hero.p_holm_four_budgets:.4f}", fontsize=PT_SMALL, color=MUTE,
            ha="right", va="center")
    _badge(ax, (13.9, 2.66), "ceiling", ha="right", va="center")
    ax.text(13.9, -0.40, f"derangement {_signed(derangement.mean_difference_pp, 1)} off scale",
            fontsize=PT_SMALL, color=MUTE, ha="right", va="center")
    for key, r in rows.items():
        assert int(r.n_pairs) == 20, (key, r.n_pairs)
    return {k: [float(r.mean_difference_pp), float(r.ci95_low_pp), float(r.ci95_high_pp),
                int(r.positive_seeds), int(r.n_pairs)] for k, r in rows.items()}


# ── build ────────────────────────────────────────────────────────────────
def build():
    mpl.rcParams["lines.markeredgewidth"] = LW_EDGE
    RECORDS.mkdir(exist_ok=True)
    outcomes = pd.read_csv(DATA / "seed_outcomes.csv")
    summary = pd.read_csv(DATA / "condition_summary.csv")
    contrasts = pd.read_csv(REVIEW / "ancestry_k4_control_contrasts.csv")
    pairs = pd.read_csv(REVIEW / "ancestry_control_paired_differences.csv")
    pairs = pairs[pairs.budget_k.eq(4)]
    enc = pd.read_csv(ENCODER / "trajectories.csv")
    hard = pd.read_csv(HARD / "trajectories.csv")
    contrasts_enc = pd.read_csv(ENCODER / "paired_contrasts.csv")
    contrasts_hard = pd.read_csv(HARD / "paired_contrasts.csv")
    prediction = coefficient_prediction()
    normalized = prediction.groupby("budget_k").normalized_coefficient_sum.mean()
    tiers = task_tiers()
    supports = control_supports(0)
    source_e = coefficient_source_table(enc, hard)

    canvas = NativeCanvas(490 / 72, 3, row_weights=[122, 108, 108], hgutter_pt=38,
                          vgutter_pt=48, margins=Margins(left=52, right=14, top=22, bottom=34))
    a = canvas.panel("A", 0, 0, 4, title="Eight-stream hierarchical task", schematic=True,
                     lock=False)
    b = canvas.panel("B", 0, 4, 8, title="Ancestry dictionaries, K = 1 to 8", schematic=True,
                     lock=False)
    c = canvas.panel("C", 1, 0, 4, title="Ancestry wins only at K = 4")
    d = canvas.panel("D", 1, 4, 4, title="Rewiring costs at K = 2, 4", sharey=c)
    e = canvas.panel("E", 1, 8, 4, title="Learned cues fall short", sharey=c)
    fpan = canvas.panel("F", 2, 0, 5, title="Matched controls at K = 4", schematic=True,
                        lock=False)
    g = canvas.panel("G", 2, 5, 7, title="K = 4: ancestry exceeds each control")

    panel_task(a, tiers)
    sums = panel_dictionaries(b, prediction)
    stats_c = panel_bandwidth(c, summary, outcomes)
    stats_d = panel_rewiring(d, summary, outcomes)
    stats_e = panel_coefficients(e, source_e, contrasts_enc, contrasts_hard)
    panel_controls(fpan, supports)
    stats_g = panel_forest(g, contrasts, pairs)
    canvas.declare_reserve("G", left=62)
    style_direct_color_labels(canvas.fig)
    canvas.lock_reserves()                 # place G behind its reserve before aligning
    findings = canvas.align_letters()
    # lock=False: the reserves are already locked, and a second lock pass would
    # re-place every letter against its own panel's ink, undoing the shared
    # module-column x that align_letters gave B and D.
    problems = canvas.save(OUT, name="credit_first_figure_03", dpi=180, lock=False)
    PUBLISHED.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(OUT, PUBLISHED)

    # records and provenance
    prediction.to_csv(RECORDS / "figure_03_coefficient_prediction.csv", index=False)
    contrasts[contrasts.control.isin(["best_matched_nonanatomical_oracle", "random_rank_k",
                                      "random_sparse_matched", "depth_interleaved_bins"])] \
        .to_csv(RECORDS / "figure_03_k4_contrasts.csv", index=False)
    supports.to_csv(RECORDS / "figure_03_control_supports.csv", index=False)
    source_e.to_csv(RECORDS / "figure_03_coefficient_source.csv", index=False)
    files = [Path(__file__), Path(routing.__file__), Path(experiment.__file__), CONFIG,
             JOURNAL / "scripts/figure_canvas.py", JOURNAL / "scripts/journal_style.py",
             JOURNAL / "scripts/native_schematics.py",
             JOURNAL / "scripts/credit_tree_schematics.py",
             DATA / "seed_outcomes.csv", DATA / "condition_summary.csv",
             REVIEW / "ancestry_k4_control_contrasts.csv",
             REVIEW / "ancestry_control_paired_differences.csv",
             ENCODER / "trajectories.csv", ENCODER / "paired_contrasts.csv",
             HARD / "trajectories.csv", HARD / "paired_contrasts.csv"]
    mapping = {
        "A": "Schematic from the frozen task configuration (selected +1; sibling, same-half "
             "and opposite-half distractor coefficients) on the shared balanced tree",
        "B": "Ancestry dictionaries A (8 x K) from grouped_routes(correct_ancestry_subtrees); "
             "card footers are the raw group sums of figure_03_coefficient_prediction.csv, "
             "verified across all 8 contexts",
        "C": "condition_summary.csv means/95% intervals (dendritic_tree: correct ancestry, "
             "derangement) and the per-seed maximum of four matched controls from "
             "seed_outcomes.csv (routing_figure_panels._best_control_by_budget)",
        "D": "condition_summary.csv means/95% intervals for dendritic_tree versus "
             "degree_depth_matched_rewired_tree under correct ancestry; paired differences "
             "and seed-bootstrap intervals from seed_outcomes.csv (bootstrap seeds 70000+)",
        "E": "review_coefficient_encoder and review_coefficient_hard_readout trajectories at "
             "calibration 256, noise 0.5, delay 0, epoch 80 (figure_03_coefficient_source.csv); "
             "paired contrasts from both paired_contrasts.csv files; hard readout exploratory",
        "F": "Recipient supports of c3's credit at K = 4 from grouped_routes / "
             "random_rank_routes (figure_03_control_supports.csv, rng seed 0 for the random "
             "families); correct ancestry delivery drawn on the ghost tree",
        "G": "Frozen ancestry_k4_control_contrasts primary maximum-over-four comparator and "
             "individual controls with paired seed points; Holm adjustment across four budgets",
    }
    payload = dict(
        panel_sources=mapping,
        source_sha256={str(p.relative_to(JOURNAL)): hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in files},
        layout_findings=list(findings) + list(problems),
        derived_numbers=dict(group_sums={int(k): float(v) for k, v in sums.items()},
                             normalized_group_sums={int(k): float(v) for k, v in
                                                    normalized.items()},
                             accuracy_by_K=stats_c, rewiring_pp=stats_d, coefficient_source=stats_e,
                             k4_contrasts_pp=stats_g),
        coefficient_scope="Raw group sums; implemented route rows divide by sqrt(group size). "
                          "Signs follow from the generator and are not a newly prospective "
                          "accuracy prediction.",
        deviations_from_spec=[
            "canvas 490 pt, rows 122/108/108, vgutter 48 (row-separation audit floor)",
            "left margin 52 (letter-column audit)",
            "F drawn as one ghost tree plus a 4 x 8 support table instead of four 38-pt cards",
            "D difference tags print the mean only (as a span between the paired "
            "markers); intervals in the caption",
            "A draws its three tiers as ORDINAL_RAMP bands (D8) at a fixed 5.6 pt width "
            "(private _tier_bands), not 16 % K-cycle capsules: K_CYCLE is B's alphabet and "
            "a 16 % tint of an edge mix does not print",
            "A labels four streams (c1, c3, c4, c8): at the 10 pt terminal pitch eight "
            "PT_SMALL labels clear each other by 2-4 pt and read as one cluster",
            "F table cells carry a 0.25 shading floor so sub-maximal supports print",
            "private helpers _badge (ceiling, exploratory) and _subtree_arrow (D7 K-cycle)"])
    (RECORDS / "figure_03_sources.json").write_text(json.dumps(payload, indent=2) + "\n")
    def _iv(v, digits=2):
        return f"[{_minus(f'{v[1]:.{digits}f}')}, {_minus(f'{v[2]:.{digits}f}')}]"

    norm_txt = ", ".join(_signed(normalized[k], 3) for k in K_TICKS)
    raw_txt = ", ".join(_signed(sums[k]) for k in K_TICKS)
    hero_g = stats_g["best_matched_nonanatomical_oracle"]
    cnt = {k: f"{v[3]}/{v[4]}" for k, v in stats_g.items()}
    # the printed seed counts are read off positive_seeds / n_pairs, never typed:
    # dense rank-4 is 15/20 like the best-matched control, not 20/20
    assert cnt["random_sparse_matched"] == cnt["depth_interleaved_bins"], cnt
    ctrl_txt = (f"{_signed(stats_g['random_rank_k'][0])} pp over dense rank-4 "
                f"({cnt['random_rank_k']} seeds), "
                f"{_signed(stats_g['random_sparse_matched'][0])} and "
                f"{_signed(stats_g['depth_interleaved_bins'][0])} pp over random-sparse and "
                f"depth-interleaved ({cnt['depth_interleaved_bins']} each)")
    caption = (
        "**Hierarchical distractors and ancestry bandwidth.** "
        "**A**, Eight streams c1–c8 at the terminals of a balanced feedback tree (four "
        "labelled); the cued stream c3 (blue contact) carries +1 sᵢ, its sibling, same-half "
        f"and other-half streams {_signed(tiers['sibling'])}, {_signed(tiers['same_half'])} "
        f"and {_signed(tiers['other_half'])} sᵢ (grey tier bands, darkest most negative). "
        "Output z, somatic error δ0. "
        "**B**, Ancestry dictionaries A (8 × K): capsules mark the K subtree groups, the "
        "arrow the group receiving c3's credit; each drawn row is a site block in tree "
        f"order. Raw group sums {raw_txt}; implemented rows divide by the square root of "
        f"group size ({norm_txt}), a design consequence, not a discovered optimum. "
        "**C**, Held-out accuracy across K for ancestry, route derangement and the per-seed "
        "maximum of four matched controls taken after training (ceiling, not a learned "
        "selector). "
        "**D**, Matched versus degree/depth-matched rewired tree under ancestry feedback; "
        f"spans give paired differences {_signed(stats_d[2][0])} "
        f"{_iv(stats_d[2])} and {_signed(stats_d[4][0])} {_iv(stats_d[4])} pp at K = 2, 4; "
        "exact ties at K = 1, 8. "
        "**E**, K = 4 coefficient source in 20 fresh seeds (52000–52019): oracle context, "
        "learned soft estimator, frozen profile and the exploratory hard readout (hollow "
        f"seeds; outside the Holm family); soft − oracle {_signed(stats_e['soft_minus_oracle'][0])} "
        f"{_iv(stats_e['soft_minus_oracle'])} pp and soft − frozen "
        f"{_signed(stats_e['soft_minus_frozen'][0])} {_iv(stats_e['soft_minus_frozen'])} pp, "
        "Holm-adjusted. "
        "**F**, Recipients of c3's credit at K = 4: the ghost tree shows the correct ancestry "
        "delivery (c3 and sibling c4); rows are rank-matched controls, columns the eight "
        "streams, cell darkness the cued context's absolute delivered support normalised to "
        "each row's largest cell (random-sparse and dense rows: single draws). "
        f"**G**, Ancestry minus each control at K = 4: {_signed(hero_g[0])} {_iv(hero_g)} pp "
        f"over the best matched control ({hero_g[3]}/{hero_g[4]} seeds, Holm "
        f"P = {float(contrasts[contrasts.control.eq('best_matched_nonanatomical_oracle') & contrasts.budget_k.eq(4)].p_holm_four_budgets.iloc[0]):.4f}), "
        f"{ctrl_txt}; derangement +61.1 pp off scale. Means with 95% seed-bootstrap "
        "intervals; seeds as dots (E, G); dashed chance (C–E) and zero (G) lines; C, D, G "
        "share 20 paired seeds.\n")
    (RECORDS / "figure_03_caption.md").write_text(caption)
    return list(findings) + list(problems), payload


def main():
    problems, payload = build()
    for p in problems:
        print(f"  {p}")
    print(json.dumps(payload["derived_numbers"], indent=1))


if __name__ == "__main__":
    main()
