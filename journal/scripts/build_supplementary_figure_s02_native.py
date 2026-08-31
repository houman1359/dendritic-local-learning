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
     ``revision_additive_gain_norm_grouped.csv`` -- the seven mechanism
     controls;
* E  ``feedback_definition_details.csv`` -- the fifteen-seed legacy
     matched-width/scalar-fallback cohort, paired per seed, with the BP and
     path-transport reference lines from the panel-D tables.  (The manifest's
     ``fig2.b`` row points figS2/e at ``figure2/feedback_accuracy_runs.csv``,
     but that is the newer journal cohort: it yields +6.62/+5.38, not the
     archived +6.40/+5.56, so this rebuild keeps the generator's own table.)

Restyling only: house palette semantics (shunting green o, additive blue s,
red-brown backprop, violet path-transport oracle), direct colour-word keys
instead of boxed legends, token type/strokes, and a 3+2 panel grid replacing
the 2.63-aspect strip.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.colors import to_rgb

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
# borrows the two house hues with no series semantics of their own on this
# sheet (violet = depth 2 as in the archived panel, orange = depth 3).
DEPTH_COLORS = {2: COLORS["oracle"], 3: COLORS["soma"]}
DEPTH_MARKERS = {2: "o", 3: "^"}

XLABEL_NI = "inhibitory synapses per branch"


def _tint(color, frac):
    """Blend ``color`` toward white by ``frac`` (lightness ladder for D)."""
    r, g, b = to_rgb(color)
    return (r + (1 - r) * frac, g + (1 - g) * frac, b + (1 - b) * frac)


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
    ax.set_xticklabels({"MNIST": "MN", "F-MNIST": "FMN",
                        "FG-MNIST": "FG"}[d[0]] for d in datasets_info)
    ax.set_xlim(-0.62, len(datasets_info) - 1 + 0.62)
    ax.set_ylim(0, 114)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_ylabel("test accuracy (%)")
    # Direct colour-word key, the sheet's palette definition (no legend box).
    ax.text(-0.55, 106.5, "backprop", color=BP, fontsize=PT_LEGEND,
            ha="left", va="center")
    ax.text(1.0, 106.5, "shunting", color=SHUNT, fontsize=PT_LEGEND,
            ha="center", va="center")
    ax.text(2.55, 109.0, "normalized\nadditive", color=ADD,
            fontsize=PT_LEGEND, ha="right", va="center", linespacing=0.95)
    return ax


# ── panel B: inhibitory dose-response (generator 1184-1231) ──────────────
def panel_inhibition(ax, report):
    ie_data = _read("gradient_fidelity_vs_ie_summary.csv")
    # House marker identity (shunting o, additive s) on both tasks; the
    # line style alone separates MNIST (solid) from the noise task (dashed).
    for ct, ds, color, ls in [
        ("dendritic_shunting", "mnist", SHUNT, "-"),
        ("dendritic_additive", "mnist", ADD, "-"),
        ("dendritic_shunting", "noise_resilience", SHUNT, "--"),
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
    ax.text(41.0, 5.2, "depth 2", color=DEPTH_COLORS[2], fontsize=PT_LEGEND,
            ha="right", va="center")
    ax.text(41.0, 17.6, "depth 3", color=DEPTH_COLORS[3], fontsize=PT_LEGEND,
            ha="right", va="center")
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
    add_none = _one(
        revision_additive,
        (revision_additive["core"] == "additive")
        & (revision_additive["additive_gain_mode"] == "none"),
    )
    gain = revision_additive[
        (revision_additive["core"] == "additive")
        & (revision_additive["additive_gain_mode"] != "none")
    ].sort_values("test_acc_mean", ascending=False)
    norm = revision_additive[
        revision_additive["core"] == "normalized_additive"
    ].sort_values("test_acc_mean", ascending=False)

    grouped = [
        (2.0, "Transport", [("BP", bp, BP), ("exact path", transport, ORACLE)]),
        (1.0, "Activation", [("identity", identity, _tint(SHUNT, 0.45)),
                             ("tanh", tanh, SHUNT)]),
        (0.0, "Additive models", [("raw", add_none, ADD),
                           ("gain-modified", None if gain.empty else gain.iloc[0],
                            _tint(ADD, 0.30)),
                           ("normalized", None if norm.empty else norm.iloc[0],
                            _tint(ADD, 0.55))]),
    ]
    height = 0.18
    x_base = 87.0
    for center, group_label, entries in grouped:
        entries = [(lab, row, col) for lab, row, col in entries
                   if row is not None]
        offsets = (np.arange(len(entries))
                   - (len(entries) - 1) / 2.0) * (height * 1.25)
        for offset, (label, row, color) in zip(offsets, entries):
            mean = float(row["test_acc_mean"]) * 100.0
            std = float(row["test_acc_std"]) * 100.0
            ypos = center + offset
            ax.barh(ypos, mean - x_base, left=x_base, height=height,
                    xerr=std, color=color, edgecolor="white", lw=LW_EDGE,
                    capsize=ERR_CAPSIZE, error_kw={"lw": LW_ERR})
            inside = (mean - x_base) > 3.2
            ax.text(x_base + 0.28 if inside else mean + std + 0.35, ypos,
                    label, ha="left", va="center", fontsize=PT_SMALL,
                    color="white" if inside else INK, zorder=6)
            report(f"D {label}: {mean:.4f}+-{std:.4f}")
        ax.text(0.015, center + 0.32, group_label,
                transform=ax.get_yaxis_transform(), ha="left", va="bottom",
                fontsize=PT_ANNOT, color=MUTE)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlim(87.0, 98.25)
    ax.set_xticks([88, 92, 96])
    ax.set_ylim(-0.55, 2.75)
    ax.set_xlabel("MNIST test accuracy (%)")
    return ax


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
    ax.text(1.02, 89.25, "normalized additive", color=ADD, fontsize=PT_LEGEND,
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
                          margins=Margins(left=40.0, right=12.0, top=16.0,
                                          bottom=26.0))
    ax_a = canvas.panel("tasks", 0, 0, 4, title="Tasks")
    ax_b = canvas.panel("inhibition", 0, 4, 4, title="Inhibition")
    ax_c = canvas.panel("morphology", 0, 8, 4, title="Morphology")
    ax_d = canvas.panel("controls", 1, 0, 6, title="Controls")
    ax_e = canvas.panel("feedback", 1, 6, 6, title="Feedback")

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
