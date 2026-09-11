#!/usr/bin/env python3
"""Supplementary Fig. S2 -- competence, regime dependence and feedback
definition in the regular-tree foundation cohorts -- as ONE native canvas.

The inherited asset (``figures/supplementary/inherited/figure_S02_panels_A-E.pdf``)
is the ``fig4_competence_regime`` sheet of the archived NeurIPS generator
(``scripts/inherited_neurips/generate_neurips_figures.py``,
``figure4_competence_regime``, lines 1019-1497): a five-panel strip whose
2.63 aspect, 4.9-12 pt type ladder and eight stroke weights all fail the
journal contract.  This builder redraws the SAME five panels -- same letters,
same conditions, same numbers -- on one 12-module
:class:`figure_canvas.NativeCanvas` at 518.4 pt wide.

Frozen content, ported aggregation
----------------------------------
Every statistic is read from the journal's tracked copies of the exact CSVs
the archived generator read, and every aggregation below is a verbatim port
of the generator's own computation (line references in the panel functions):

* A  ``competence_summary_20260422.csv`` -- matched shunting-BP vs five-factor
     local shunting/additive on MNIST, Fashion-MNIST, figure-ground MNIST;
* B  ``gradient_fidelity_vs_ie_summary.csv`` -- inhibitory dose-response on
     MNIST (solid) and the noise task (dashed);
* C  ``morphology_ie_regime_runs.csv`` -- per-seed shunting-additive gap,
     merged on (branch_factors, depth, branch_product, ie, seed), grouped by
     depth x N_I (mean, ddof-1 s.d.);
* D  ``revision_exact_transport_bp_grouped.csv`` /
     ``revision_exact_transport_factorial_grouped.csv`` /
     ``revision_reactivation_identity_grouped.csv`` /
     ``revision_additive_gain_norm_grouped.csv`` -- the mechanism controls:
     transport (2), activation (2) and, since 2026-09-11, the COMPLETE
     normalized/raw x gain-mode factorial of the additive table (8 cells
     as four paired rows) instead of three hand-picked cells;
* E  ``feedback_definition_details.csv`` -- the fifteen-seed legacy
     matched-width/scalar-fallback cohort, paired per seed, with the BP and
     path-transport reference lines from the panel-D tables.  (The manifest's
     ``fig2.b`` row points figS2/e at ``figure2/feedback_accuracy_runs.csv``,
     but that is the newer journal cohort: it yields +6.62/+5.38, not the
     archived +6.40/+5.56, so this rebuild keeps the generator's own table.)

Restyling only: house palette semantics (shunting green o, additive blue s,
red-brown backprop, violet path-transport oracle), direct colour-word keys
instead of boxed legends, token type/strokes, and a 3+2 panel grid replacing
the 2.63-aspect strip.  In D the colour names the architecture exactly as
in A and the marker the variant inside it (house o / s = the configuration
A draws, triangle = the ablated variant: identity activation, raw additive).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from figure_canvas import (  # noqa: E402
    COLORS,
    ERR_CAPSIZE,
    LW_DATA,
    LW_EDGE,
    LW_ERR,
    LW_HAIR,
    LW_REF,
    MARKER_MS,
    PT_ANNOT,
    PT_LEGEND,
    PT_SMALL,
    Margins,
    NativeCanvas,
)

ROOT = SCRIPT_DIR.parent
DATA = ROOT / "source_data" / "inherited_neurips"
TARGET = ROOT / "figures" / "supplementary" / "figure_S02_panels_A-E.pdf"

SHUNT = COLORS["shunting"]
ADD = COLORS["additive"]
BP = COLORS["bp"]
ORACLE = COLORS["oracle"]
MUTE = COLORS["mute"]
INK = COLORS["ink"]

# The morphology panel contrasts two tree depths, not two mechanisms, so it
# borrows two house hues with no series semantics of their own on this
# sheet (violet = depth 2 as in the archived panel, amber = depth 3).
# 2026-09-11: ``COLORS["soma"]`` became the schematic yellow (#F2EC30), which
# printed depth 3 as a yellow line on white; the amber ``scalar`` slot is the
# violet/amber pair every other sheet already separates.
DEPTH_COLORS = {2: COLORS["oracle"], 3: COLORS["scalar"]}
DEPTH_MARKERS = {2: "o", 3: "^"}

XLABEL_NI = "inhibitory synapses per branch"


def _read(name):
    return pd.read_csv(DATA / name)


# ── panel A: multi-benchmark competence bars (generator 1038-1182) ───────
def panel_tasks(ax, report):
    competence = _read("competence_summary_20260422.csv")

    def _pick(dataset, network_type, strategy):
        sub = competence[
            (competence["dataset"] == dataset)
            & (competence["network_type"] == network_type)
            & (competence["strategy"] == strategy)
        ]
        return None if len(sub) == 0 else sub.iloc[0]

    datasets_info = []
    for dataset, label in [("mnist", "MNIST"), ("fashion_mnist", "F-MNIST"),
                           ("context_gating", "FG-MNIST")]:
        bp = _pick(dataset, "dendritic_shunting", "standard")
        shunt = _pick(dataset, "dendritic_shunting", "local_ca")
        add = _pick(dataset, "dendritic_additive", "local_ca")
        if bp is None or shunt is None:
            continue
        datasets_info.append((
            label,
            float(bp["test_accuracy_mean"]), float(bp["test_accuracy_std"]),
            float(shunt["test_accuracy_mean"]),
            float(shunt["test_accuracy_std"]),
            None if add is None else float(add["test_accuracy_mean"]),
            0 if add is None else float(add["test_accuracy_std"]),
        ))

    bar_w = 0.22
    for i, (name, bp_v, bp_e, sh_v, sh_e, ad_v, ad_e) in enumerate(
            datasets_info):
        ax.bar(i - bar_w, bp_v * 100, bar_w * 0.88, yerr=bp_e * 100,
               color=BP, edgecolor="white", lw=LW_EDGE,
               capsize=ERR_CAPSIZE, error_kw={"lw": LW_ERR})
        if sh_v is not None:
            ax.bar(i, sh_v * 100, bar_w * 0.88, yerr=sh_e * 100,
                   color=SHUNT, edgecolor="white", lw=LW_EDGE,
                   capsize=ERR_CAPSIZE, error_kw={"lw": LW_ERR})
        if ad_v is not None:
            ax.bar(i + bar_w, ad_v * 100, bar_w * 0.88, yerr=ad_e * 100,
                   color=ADD, edgecolor="white", lw=LW_EDGE,
                   capsize=ERR_CAPSIZE, error_kw={"lw": LW_ERR})
        report(f"A {name}: BP {bp_v * 100:.4f}+-{bp_e * 100:.4f}  "
               f"shunt {sh_v * 100:.4f}+-{sh_e * 100:.4f}  "
               f"add {ad_v * 100:.4f}+-{ad_e * 100:.4f}")

    ax.set_xticks(range(len(datasets_info)))
    ax.set_xticklabels([d[0] for d in datasets_info])
    ax.set_xlim(-0.62, len(datasets_info) - 1 + 0.62)
    # Accuracy cannot exceed 100 %, so the axis stops there; the key no
    # longer lives inside the axes (it used to need 14 pp of headroom).
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_ylabel("test accuracy (%)")
    # Direct colour-word key ABOVE the axes, on one left edge and one
    # line pitch, naming BOTH factors of every series: the red bar is the
    # same shunting architecture as the green bar, trained by
    # backpropagation instead of the local rule; blue changes the
    # architecture at the fixed local rule.  The band is reserved through
    # ``declare_reserve`` in :func:`build` so the row lock keeps A, B and C
    # on one axes top.
    # Bottom-up: the n / interval statement nearest the axes (the bars
    # leave no data-free room inside a 0-100 % axis), then the three series
    # in the left-to-right order of the bars, reading top-down.
    for k, (words, color) in enumerate([
            ("mean ± s.d., 5 seeds", MUTE),
            ("normalized additive, local rule", ADD),
            ("shunting, local rule", SHUNT),
            ("shunting, backprop", BP)]):
        ax.text(0.0, 1.0, words, color=color, fontsize=PT_LEGEND,
                ha="left", va="bottom",
                transform=ax.transAxes
                + _pt_offset(ax, 0.0, KEY_PAD_PT + k * KEY_PITCH_PT))
    return ax


KEY_PAD_PT = 3.0        # first key line sits this far above the axes top
KEY_PITCH_PT = 8.6      # line pitch of the stacked key (7 pt type)
KEY_LINES = 4
KEY_RESERVE_PT = KEY_PAD_PT + KEY_LINES * KEY_PITCH_PT + 2.0


def _pt_offset(ax, dx_pt, dy_pt):
    """A translation of ``(dx, dy)`` points in the axes' figure."""
    from matplotlib.transforms import ScaledTranslation
    return ScaledTranslation(dx_pt / 72.0, dy_pt / 72.0,
                             ax.figure.dpi_scale_trans)


# ── panel B: inhibitory dose-response (generator 1184-1231) ──────────────
def panel_inhibition(ax, report):
    ie_data = _read("gradient_fidelity_vs_ie_summary.csv")
    # House marker identity (shunting o, additive s) on both tasks; the
    # line style alone separates MNIST (solid) from the noise task (dashed).
    for ct, ds, color, ls in [
        ("dendritic_shunting", "mnist", SHUNT, "-"),
        ("dendritic_additive", "mnist", ADD, "-"),
        ("dendritic_additive", "noise_resilience", ADD, "--"),
    ]:
        sub = ie_data[(ie_data["network_type"] == ct)
                      & (ie_data["dataset"] == ds)].copy()
        sub = sub.sort_values("ie_value")
        if len(sub) == 0:
            continue
        marker = "o" if "shunting" in ct else "s"
        ax.errorbar(sub["ie_value"], sub["test_accuracy_mean"] * 100,
                    yerr=sub["test_accuracy_std"] * 100,
                    marker=marker, markersize=3.6, linewidth=LW_DATA,
                    elinewidth=LW_ERR, capsize=ERR_CAPSIZE, capthick=LW_ERR,
                    color=color, linestyle=ls,
                    markerfacecolor="white", markeredgecolor=color,
                    markeredgewidth=LW_ERR)
        report(f"B {ct}/{ds}: x={sub['ie_value'].tolist()} "
               f"y={[round(v, 4) for v in (sub['test_accuracy_mean'] * 100)]} "
               f"sd={[round(v, 4) for v in (sub['test_accuracy_std'] * 100)]}")
    ax.set_xlabel(XLABEL_NI)
    ax.set_ylabel("test accuracy (%)")
    ax.set_xlim(-2.5, 43.5)
    ax.set_xticks([0, 20, 40])
    ax.set_ylim(48, 97)
    ax.set_yticks([50, 60, 70, 80, 90])
    # Line-style key as direct labels; the colour key lives in panel A.
    ax.text(41.5, 94.3, "MNIST", color=MUTE, fontsize=PT_LEGEND,
            ha="right", va="center")
    ax.text(41.5, 78.6, "noise", color=MUTE, fontsize=PT_LEGEND,
            ha="right", va="center")
    return ax


# ── panel C: morphology-dependent operating regime (generator 1233-1275) ─
def panel_morphology(ax, report):
    morph_runs = _read("morphology_ie_regime_runs.csv")
    shunt = morph_runs[morph_runs["network_type"] == "dendritic_shunting"].copy()
    add = morph_runs[morph_runs["network_type"] == "dendritic_additive"].copy()
    merged = shunt.merge(
        add,
        on=["branch_factors", "depth", "branch_product", "ie", "seed"],
        suffixes=("_s", "_a"),
    )
    merged["gap"] = (merged["test_acc_s"] - merged["test_acc_a"]) * 100.0
    depth_gap = (
        merged.groupby(["depth", "ie"])["gap"]
        .agg(mean="mean", std="std")
        .reset_index()
        .sort_values(["depth", "ie"])
    )
    for depth, sub in depth_gap.groupby("depth"):
        color = DEPTH_COLORS[int(depth)]
        ax.errorbar(
            sub["ie"], sub["mean"], yerr=sub["std"].fillna(0.0),
            marker=DEPTH_MARKERS[int(depth)], markersize=MARKER_MS,
            linewidth=LW_DATA, color=color, elinewidth=LW_ERR,
            capsize=ERR_CAPSIZE, capthick=LW_ERR,
            markerfacecolor="white", markeredgecolor=color,
            markeredgewidth=LW_ERR,
        )
        report(f"C depth {int(depth)}: x={sub['ie'].tolist()} "
               f"gap={[round(v, 4) for v in sub['mean']]} "
               f"sd={[round(v, 4) for v in sub['std']]}")
    # The dashed zero reference and a y=0 gridline would print as one
    # mottled stroke, so the y grid is drawn here without its zero member.
    for yt in (10, 20):
        ax.axhline(yt, color=COLORS["grid"], lw=LW_HAIR, alpha=0.9, zorder=0)
    ax.axhline(0, color=MUTE, lw=LW_REF, ls="--", zorder=0.5)
    ax.set_xlim(-3, 43)
    ax.set_xticks([0, 20, 40])
    ax.set_ylim(-8, 29)
    ax.set_yticks([0, 10, 20])
    ax.set_xlabel(XLABEL_NI)
    ax.set_ylabel("shunting − normalized additive (pp)")
    # Colour words between the x=20 and x=40 whiskers, one under the violet
    # segment and one over the amber segment; the footnote sits in the
    # data-free lower right (2026-09-11: the shorter row put the old
    # positions on the whiskers).
    ax.text(30.0, 5.6, "depth 2", color=DEPTH_COLORS[2], fontsize=PT_LEGEND,
            ha="center", va="center")
    ax.text(30.0, 19.8, "depth 3", color=DEPTH_COLORS[3], fontsize=PT_LEGEND,
            ha="center", va="center")
    ax.text(.98, .04, "depth and resources change", transform=ax.transAxes,
            fontsize=PT_SMALL, color=MUTE, ha="right", va="bottom")
    return ax


# ── panel D: mechanism controls (generator 1277-1392) ────────────────────
def panel_controls(ax, report):
    revision_exact = _read("revision_exact_transport_factorial_grouped.csv")
    revision_bp = _read("revision_exact_transport_bp_grouped.csv")
    revision_additive = _read("revision_additive_gain_norm_grouped.csv")
    revision_reactivation = _read("revision_reactivation_identity_grouped.csv")

    def _one(df, mask):
        sub = df[mask]
        return None if len(sub) == 0 else sub.iloc[0]

    bp = revision_bp.iloc[0]
    transport = _one(
        revision_exact,
        (revision_exact["rule_variant"] == "5f")
        & (revision_exact["decoder_update_mode"] == "local"),
    )
    identity = _one(
        revision_reactivation,
        revision_reactivation["reactivation_enabled"] == False,  # noqa: E712
    )
    tanh = _one(
        revision_reactivation,
        revision_reactivation["reactivation_enabled"] == True,  # noqa: E712
    )

    # Colour names the architecture / reference exactly as panel A does
    # (red-brown backprop, green shunting, blue additive, violet exact
    # path); the marker names the variant inside a family: the house marker
    # (shunting o, additive s) is the configuration panel A also draws, a
    # triangle is the ablated variant (identity activation; raw additive
    # without normalization).  Every row label is set in ink as a y tick.
    # 2026-09-11: the additive group draws the COMPLETE 2 x 4 (normalized /
    # raw x gain mode) factorial of the source table as four paired rows,
    # instead of three hand-picked cells whose gain modes differed.
    GAIN_LABELS = [("none", "no gain"),
                   ("learned_gain", "learned gain"),
                   ("input_dependent", "input-dependent"),
                   ("running_stats", "running stats")]

    def _cell(core, mode):
        return _one(revision_additive,
                    (revision_additive["core"] == core)
                    & (revision_additive["additive_gain_mode"] == mode))

    # (kind, label, entries); entries = (name, row, colour, marker, dodge)
    lines = [
        ("hdr", "Transport", []),
        ("row", "exact path", [("exact path", transport, ORACLE, "o", 0.0)]),
        ("row", "BP", [("BP", bp, BP, "o", 0.0)]),
        ("gap", "", []),
        ("hdr", "Activation (shunting)", []),
        ("row", "tanh", [("tanh", tanh, SHUNT, "o", 0.0)]),
        ("row", "identity", [("identity", identity, SHUNT, "^", 0.0)]),
        ("gap", "", []),
        ("hdr", "Additive models, by gain mode", []),
    ]
    for mode, label in GAIN_LABELS:
        lines.append(("row", label, [
            (f"normalized/{mode}", _cell("normalized_additive", mode), ADD,
             "s", -0.15),
            (f"raw/{mode}", _cell("additive", mode), ADD, "^", 0.15)]))

    y = 0.0
    tick_pos, tick_lab = [], []
    for kind, label, entries in lines:
        if kind == "gap":
            y += 0.45
            continue
        if kind == "hdr":
            ax.text(0.01, y, label, transform=ax.get_yaxis_transform(),
                    ha="left", va="center", fontsize=PT_ANNOT, color=MUTE)
            y += 1.0
            continue
        tick_pos.append(y)
        tick_lab.append(label)
        for name, row, color, marker, dodge in entries:
            if row is None:
                continue
            mean = float(row["test_acc_mean"]) * 100.0
            std = float(row["test_acc_std"]) * 100.0
            ax.errorbar(mean, y + dodge, xerr=std, fmt=marker, ms=3.5,
                        color=color, lw=LW_ERR, capsize=ERR_CAPSIZE,
                        zorder=4)
            report(f"D {name}: {mean:.4f}+-{std:.4f}")
        y += 1.0
    n_lines = y
    ax.set_yticks(tick_pos)
    ax.set_yticklabels(tick_lab)
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(n_lines - 0.3, -0.55)          # first line at the top
    # Marker key for the paired additive rows, on the group's header line
    # (no data on that line), and the sheet's n / interval statement.
    hdr_y = [yy for (k, lab, _), yy in zip(
        lines, _line_positions(lines)) if lab.startswith("Additive")][0]
    for xf, marker, words in ((0.60, "s", "normalized"),
                              (0.86, "^", "raw")):
        ax.plot([xf], [hdr_y], marker=marker, ms=3.5, color=ADD, ls="none",
                transform=ax.get_yaxis_transform(), clip_on=False, zorder=4)
        ax.text(xf + 0.03, hdr_y, words, transform=ax.get_yaxis_transform(),
                ha="left", va="center", fontsize=PT_ANNOT, color=INK)
    ax.text(0.99, 0.03, "mean ± s.d., 5 seeds", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=PT_SMALL, color=MUTE)
    ax.set_xlim(86.2, 97.9)
    ax.set_xticks([88, 92, 96])
    ax.set_xlabel("MNIST test accuracy (%)")
    return ax


def _line_positions(lines):
    """The y of every entry of a ``lines`` list, mirroring the loop above."""
    out, y = [], 0.0
    for kind, _, _ in lines:
        out.append(y)
        y += 0.45 if kind == "gap" else 1.0
    return out


# ── panel E: neuron-specific feedback definition (generator 1394-1487) ───
def panel_feedback(ax, report):
    feedback_definition = _read("feedback_definition_details.csv")
    revision_exact = _read("revision_exact_transport_factorial_grouped.csv")
    revision_bp = _read("revision_exact_transport_bp_grouped.csv")

    feedback_order = ["scalar_fallback", "ancestry_shared"]
    x = np.arange(len(feedback_order), dtype=float)
    core_specs = [
        ("dendritic_shunting", SHUNT, "o", -0.035),
        ("dendritic_additive", ADD, "s", 0.035),
    ]
    for network_type, color, marker, offset in core_specs:
        core_rows = feedback_definition[
            feedback_definition["network_type"] == network_type
        ]
        pivot = core_rows.pivot(
            index="seed", columns="feedback", values="test_accuracy",
        ).dropna()
        paired = 100.0 * pivot[feedback_order].to_numpy(dtype=float)
        for row in paired:
            ax.plot(x + offset, row, color=color, alpha=0.24, lw=LW_HAIR,
                    zorder=1)
        means = paired.mean(axis=0)
        stds = paired.std(axis=0, ddof=1)
        ax.errorbar(x + offset, means, yerr=stds, color=color, marker=marker,
                    markersize=MARKER_MS, markerfacecolor="white",
                    markeredgewidth=LW_ERR, lw=LW_DATA, elinewidth=LW_ERR,
                    capsize=ERR_CAPSIZE, zorder=4)
        gain = means[1] - means[0]
        ax.text(-0.045,
                (means[0] + 1.6) if offset > 0 else (means[0] - 1.3),
                f"+{gain:.2f}", color=color, fontsize=PT_ANNOT,
                ha="center", va="center",
                bbox={"facecolor": "white", "edgecolor": "none",
                      "pad": 0.4, "alpha": 0.82})
        report(f"E {network_type}: n={paired.shape[0]} "
               f"means={[round(v, 4) for v in means]} "
               f"sd={[round(v, 4) for v in stds]} gain=+{gain:.2f}")

    bp_value = 100.0 * float(revision_bp.iloc[0]["test_acc_mean"])
    exact_row = revision_exact[
        (revision_exact["rule_variant"] == "3f")
        & (revision_exact["decoder_update_mode"] == "local")
    ].iloc[0]
    exact_value = 100.0 * float(exact_row["test_acc_mean"])
    ax.axhline(bp_value, color=BP, lw=LW_REF, ls="--", zorder=2)
    ax.axhline(exact_value, color=ORACLE, lw=LW_REF, ls=":", zorder=2)
    ax.text(1.40, bp_value - 0.12, "BP", color=BP, fontsize=PT_LEGEND,
            ha="right", va="top")
    ax.text(1.40, exact_value + 0.12, "exact path", color=ORACLE,
            fontsize=PT_LEGEND, ha="right", va="bottom")
    report(f"E reference lines: BP={bp_value:.4f} PT={exact_value:.4f}")
    ax.text(0.02, 0.97, "15/15 pairs", transform=ax.transAxes, ha="left",
            va="top", fontsize=PT_ANNOT, color=MUTE)
    ax.set_xticks(x)
    ax.set_xticklabels(["matched-width\nfallback", "neuron-\nspecific"])
    ax.set_xlim(-0.20, 1.42)
    ax.set_ylim(88.5, 98.2)
    ax.set_yticks([90, 94, 98])
    ax.set_ylabel("3F test accuracy (%)")
    # Direct colour words in the data-free lower right (no legend box).
    ax.text(1.02, 90.15, "shunting", color=SHUNT, fontsize=PT_LEGEND,
            ha="center", va="center")
    ax.text(1.02, 89.25, "additive (legacy)", color=ADD, fontsize=PT_LEGEND,
            ha="center", va="center")
    return ax


# ── the canvas ───────────────────────────────────────────────────────────
CANVAS_H_PT = 380.0     # aspect 518.4/380 = 1.36, inside the 1.05-1.55 band


def build(path=TARGET):
    lines = []

    def report(msg):
        lines.append(msg)

    canvas = NativeCanvas(CANVAS_H_PT / 72.0, 2, hgutter_pt=32.0,
                          vgutter_pt=38.0,
                          margins=Margins(left=62.0, right=12.0, top=16.0,
                                          bottom=26.0))
    # A and D carry no title: "Tasks" only restated the tick labels and
    # "Controls" the caption, and A's title band now holds its colour key.
    ax_a = canvas.panel("tasks", 0, 0, 4)
    ax_b = canvas.panel("inhibition", 0, 4, 4, title="Inhibition")
    ax_c = canvas.panel("morphology", 0, 8, 4, title="Morphology")
    ax_d = canvas.panel("controls", 1, 0, 6)
    ax_e = canvas.panel("feedback", 1, 6, 6, title="Feedback")
    canvas.declare_reserve("tasks", top=KEY_RESERVE_PT)

    panel_tasks(ax_a, report)
    panel_inhibition(ax_b, report)
    panel_morphology(ax_c, report)
    panel_controls(ax_d, report)
    panel_feedback(ax_e, report)

    problems = canvas.save(Path(path), name="figure_S02_panels_A-E",
                           png=False)
    print("\n-- plotted values (frozen check) --")
    for line in lines:
        print(" ", line)
    for problem in problems:
        print(f"    {problem}")
    return problems


if __name__ == "__main__":
    raise SystemExit(1 if build() else 0)
