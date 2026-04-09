#!/usr/bin/env python3
"""Generate revision figures for NeurIPS 2026 paper.

Produces:
  Main:
    fig5_mechanistic_evidence.pdf  (4 panels: conductance-stage path gains, exact-error fidelity, oracle learning, low-bandwidth)
    fig6_cue_routing.pdf          (3 panels: hard cue-routing diagnosis)
  Supplement:
    fig_s5_fa_dfa.pdf              (FA/DFA baseline comparison)
    fig_s6_cifar10.pdf             (corrected strong-family CIFAR-10 mechanism extension)
    fig_s7_additive_norm.pdf       (Additive + normalization control)

Usage:
    cd /n/holylabs/LABS/kempner_dev/Users/hsafaai/Code/dendritic-modeling
    python drafts/dendritic-local-learning/scripts/generate_revision_figures.py
"""

import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

from generate_cue_routing_figures import SUMMARY_CSV as CUE_SUMMARY_CSV
from generate_cue_routing_figures import build_figure as build_cue_routing_figure
from generate_5f_sensitivity_figure import (
    SUMMARY_CSV as FIVE_FACTOR_SUMMARY_CSV,
    build_figure as build_five_factor_sensitivity_figure,
)
from generate_theory_diagnostics_figures import (
    build_figure as build_theory_diagnostics_figure,
)
from summarize_cue_routing_results import summarize_runs as summarize_cue_routing_runs

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DRAFT_DIR = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(DRAFT_DIR, "data")
FIGURES_DIR = os.path.join(DRAFT_DIR, "figures")

# ---------------------------------------------------------------------------
# Style (unified NeurIPS style)
# ---------------------------------------------------------------------------
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from neurips_style import apply_neurips_style, COLORS, panel_label, style_axis
apply_neurips_style()

COLOR_SHUNTING = COLORS["shunting"]
COLOR_ADDITIVE = COLORS["additive"]
COLOR_BACKPROP = COLORS["bp"]
COLOR_POINT_MLP = COLORS["point_mlp"]
COLOR_LOW_RANK = COLORS["low_rank"]
COLOR_ORACLE = COLORS["oracle"]

W = 5.5  # NeurIPS single-column width
DPI = 300

CIFAR10_BP_SUMMARY_CSV = os.path.join(
    DRAFT_DIR, "analysis", "cifar10_compactei_depth4", "cifar10_compactei_depth4_grouped_summary.csv"
)
CIFAR10_LOCALCA_SUMMARY_CSV = os.path.join(
    DRAFT_DIR,
    "analysis",
    "cifar10_compactei_depth4_decoderfix_mechanism_5seed",
    "cifar10_compactei_depth4_decoderfix_mechanism_summary.csv",
)


def _panel(ax, label, x=-0.18, y=1.12):
    panel_label(ax, label, x=x, y=y)


def _save(fig, name):
    os.makedirs(FIGURES_DIR, exist_ok=True)
    for ext in ("pdf", "png"):
        p = os.path.join(FIGURES_DIR, f"{name}.{ext}")
        fig.savefig(p, dpi=DPI)
    print(f"  Saved: {name}.{{pdf,png}}")


def _csv(filename):
    path = os.path.join(DATA_DIR, filename)
    if not os.path.isfile(path):
        warnings.warn(f"CSV not found: {path}")
        return None
    return pd.read_csv(path)


def _csv_path(path):
    if not os.path.isfile(path):
        warnings.warn(f"CSV not found: {path}")
        return None
    return pd.read_csv(path)


# ===================================================================
# Figure 5 — Mechanistic Evidence (NEW main figure)
# ===================================================================
def figure5():
    """Mechanistic figure built from exact-error diagnostics and bandwidth sweep."""
    print("\n--- Figure 5: Mechanistic Evidence ---")
    fig = build_theory_diagnostics_figure()
    _save(fig, "fig5_mechanistic_evidence")
    plt.close(fig)


# ===================================================================
# Figure S5 — FA/DFA Baselines
# ===================================================================
def figure_s5():
    print("\n--- Figure S5: FA/DFA Baselines ---")

    fa_dfa = _csv("fa_dfa_results.csv")
    if fa_dfa is None:
        print("  SKIPPED: fa_dfa_results.csv not found")
        return

    fig, axes = plt.subplots(1, 2, figsize=(W, 2.8),
                             gridspec_kw={"wspace": 0.45})

    # ---- Panel A: Grouped bars by model ----
    ax = axes[0]
    _panel(ax, "A")

    models = ["dendritic_shunting", "dendritic_additive", "point_mlp"]
    model_labels = ["Shunting", "Additive", "Point MLP"]
    strategies = ["standard", "dfa", "fa"]
    strat_labels = ["Backprop", "DFA", "FA"]
    strat_colors = [COLOR_BACKPROP, "#E67E22", "#8E44AD"]

    x = np.arange(len(models))
    bw = 0.22

    for j, (strat, slabel, scolor) in enumerate(zip(strategies, strat_labels, strat_colors)):
        means, errs = [], []
        valid = []
        for model in models:
            sub = fa_dfa[(fa_dfa["model"] == model) & (fa_dfa["strategy"] == strat)]
            if len(sub) > 0:
                means.append(sub["test_accuracy"].mean() * 100)
                errs.append(sub["test_accuracy"].std() * 100)
                valid.append(True)
            else:
                means.append(0)
                errs.append(0)
                valid.append(False)
        for i in range(len(models)):
            if valid[i]:
                ax.bar(x[i] + (j - 1) * bw, means[i], bw * 0.88,
                       yerr=errs[i], color=scolor, edgecolor="white", lw=0.3,
                       capsize=1.5, error_kw={"lw": 0.5})
            else:
                # Mark as failed with X
                ax.text(x[i] + (j - 1) * bw, 5, "X", ha="center", va="bottom",
                        fontsize=8, color="red", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(model_labels)
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title("FA/DFA comparison (MNIST)")
    ax.set_ylim(0, 100)

    # Legend
    handles = [mpatches.Patch(color=c, label=l)
               for c, l in zip(strat_colors, strat_labels)]
    ax.legend(handles=handles, fontsize=5.5, loc="upper left",
              handlelength=1.0, handletextpad=0.3)

    # ---- Panel B: DFA advantage for shunting vs additive ----
    ax = axes[1]
    _panel(ax, "B")

    for model, color, label in [
        ("dendritic_shunting", COLOR_SHUNTING, "Shunting"),
        ("dendritic_additive", COLOR_ADDITIVE, "Additive"),
        ("point_mlp", COLOR_POINT_MLP, "Point MLP"),
    ]:
        std = fa_dfa[(fa_dfa["model"] == model) & (fa_dfa["strategy"] == "standard")]
        dfa = fa_dfa[(fa_dfa["model"] == model) & (fa_dfa["strategy"] == "dfa")]
        if len(std) and len(dfa):
            std_mean = std["test_accuracy"].mean() * 100
            dfa_mean = dfa["test_accuracy"].mean() * 100
            gap = std_mean - dfa_mean
            ax.barh(label, gap, color=color, edgecolor="white", lw=0.3, height=0.5)
            ax.text(gap + 0.5, label, f"{gap:.1f}pp", va="center", fontsize=5.5)

    ax.set_xlabel("Backprop - DFA gap (pp)")
    ax.set_title("DFA performance gap")
    ax.axvline(0, color="black", lw=0.4)

    fig.subplots_adjust(left=0.12, right=0.97, bottom=0.15, top=0.88,
                        wspace=0.50)
    _save(fig, "fig_s5_fa_dfa")
    plt.close(fig)


# ===================================================================
# Figure S3 — 5F Sensitivity
# ===================================================================
def figure_s3():
    print("\n--- Figure S3: 5F Sensitivity ---")
    if not os.path.isfile(FIVE_FACTOR_SUMMARY_CSV):
        print("  SKIPPED: five_factor_sensitivity_summary.csv not found")
        return
    fig = build_five_factor_sensitivity_figure(FIVE_FACTOR_SUMMARY_CSV)
    _save(fig, "fig_s3_5f_sensitivity")
    plt.close(fig)


# ===================================================================
# Figure S6 — CIFAR-10 Results
# ===================================================================
def figure_s6():
    print("\n--- Figure S6: CIFAR-10 Mechanism Extension ---")

    bp = _csv_path(CIFAR10_BP_SUMMARY_CSV)
    localca = _csv_path(CIFAR10_LOCALCA_SUMMARY_CSV)
    if bp is None or localca is None:
        print("  SKIPPED: corrected CIFAR summary CSVs not found")
        return

    fig, axes = plt.subplots(1, 2, figsize=(W, 2.55), gridspec_kw={"wspace": 0.42})

    def get_bp(model):
        sub = bp[(bp["strategy"] == "standard") & (bp["model_type"] == model)]
        row = sub.iloc[0]
        return row["mean_test_accuracy"] * 100, row["std_test_accuracy"] * 100

    def get_local(condition):
        sub = localca[localca["condition"] == condition]
        row = sub.iloc[0]
        return row["acc_test_mean"] * 100, row["acc_test_std"] * 100

    # Panel A: matched additive vs shunting ladder
    ax = axes[0]
    _panel(ax, "A")
    style_axis(ax, grid="y")

    categories = ["Standard", "Per-soma", "Path transport"]
    additive_vals = [
        get_bp("dendritic_additive"),
        get_local("cifar10_additive_5f_per_soma_bpdec_wd0"),
        get_local("cifar10_additive_5f_path_transport_bpdec_wd0"),
    ]
    shunting_vals = [
        get_bp("dendritic_shunting"),
        get_local("cifar10_shunting_5f_per_soma_bpdec_wd0"),
        get_local("cifar10_shunting_5f_path_transport_bpdec_wd0"),
    ]

    x = np.arange(len(categories))
    bw = 0.34
    ax.bar(
        x - bw / 2,
        [m for m, _ in additive_vals],
        yerr=[s for _, s in additive_vals],
        width=bw,
        color=COLOR_ADDITIVE,
        edgecolor="white",
        lw=0.4,
        capsize=2,
        error_kw={"lw": 0.6},
        label="Additive",
    )
    ax.bar(
        x + bw / 2,
        [m for m, _ in shunting_vals],
        yerr=[s for _, s in shunting_vals],
        width=bw,
        color=COLOR_SHUNTING,
        edgecolor="white",
        lw=0.4,
        capsize=2,
        error_kw={"lw": 0.6},
        label="Shunting",
    )
    for xi, (m, s) in zip(x - bw / 2, additive_vals):
        ax.text(xi, m + s + 0.9, f"{m:.1f}", ha="center", va="bottom", fontsize=5.6)
    for xi, (m, s) in zip(x + bw / 2, shunting_vals):
        ax.text(xi, m + s + 0.9, f"{m:.1f}", ha="center", va="bottom", fontsize=5.6)
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title("Corrected CIFAR-10 ladder")
    ax.set_ylim(0, 58)
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(0.00, 1.04),
        ncol=1,
        frameon=False,
        handlelength=1.0,
        columnspacing=0.8,
        handletextpad=0.4,
        borderaxespad=0.0,
    )

    # Panel B: shunting rank bridge in the strong family
    ax = axes[1]
    _panel(ax, "B")
    style_axis(ax, grid="y")

    shunt_standard_mean, _ = get_bp("dendritic_shunting")
    ladder = [
        ("Per-soma",) + get_local("cifar10_shunting_5f_per_soma_bpdec_wd0") + (COLOR_SHUNTING,),
        ("Low-rank\n$K=4$",) + get_local("cifar10_shunting_5f_low_rank4_bpdec_wd0") + (COLOR_LOW_RANK,),
        ("Path\ntransport",) + get_local("cifar10_shunting_5f_path_transport_bpdec_wd0") + (COLOR_ORACLE,),
    ]
    x2 = np.arange(len(ladder))
    bars = ax.bar(
        x2,
        [m for _, m, _, _ in ladder],
        yerr=[s for _, _, s, _ in ladder],
        color=[c for _, _, _, c in ladder],
        edgecolor="white",
        lw=0.4,
        width=0.62,
        capsize=2,
        error_kw={"lw": 0.6},
    )
    for rect, (_, m, s, _) in zip(bars, ladder):
        ax.text(rect.get_x() + rect.get_width()/2, m + s + 0.9, f"{m:.1f}",
                ha="center", va="bottom", fontsize=5.8)
    ax.axhline(shunt_standard_mean, color=COLOR_BACKPROP, lw=1.0, ls=(0, (4, 2)))
    ax.text(2.38, shunt_standard_mean + 0.8, f"Shunt. BP {shunt_standard_mean:.1f}",
            color=COLOR_BACKPROP, fontsize=5.8, ha="right", va="bottom")
    ax.set_xticks(x2)
    ax.set_xticklabels([label for label, *_ in ladder])
    ax.set_title("Shunting feedback bridge")
    ax.set_ylim(0, 55)

    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.18, top=0.90, wspace=0.42)
    _save(fig, "fig_s6_cifar10")
    _save(fig, "fig_cifar10_mechanism_extension")
    plt.close(fig)


# ===================================================================
# Figure S7 — Additive + Normalization Control
# ===================================================================
def figure_s7():
    print("\n--- Figure S7: Additive + Normalization Control ---")

    anorm = _csv("additive_norm_results.csv")
    lbw = _csv("low_bandwidth_results.csv")
    if anorm is None:
        print("  SKIPPED: additive_norm_results.csv not found")
        return

    fig, axes = plt.subplots(1, 2, figsize=(W, 2.8),
                             gridspec_kw={"wspace": 0.45})

    # ---- Panel A: Standard training ----
    ax = axes[0]
    _panel(ax, "A")

    conditions = []
    for norm in [False, True]:
        sub = anorm[(anorm["use_additive_normalization"] == norm) &
                    (anorm["strategy"] == "standard")]
        if len(sub):
            m = sub["test_accuracy"].mean() * 100
            s = sub["test_accuracy"].std() * 100
            label = "Add.+norm" if norm else "Additive"
            color = "#5B8AC4" if norm else COLOR_ADDITIVE
            conditions.append((label, m, s, color))

    x_pos = np.arange(len(conditions))
    for i, (label, m, s, color) in enumerate(conditions):
        ax.bar(i, m, yerr=s, color=color, edgecolor="white", lw=0.3,
               width=0.55, capsize=2, error_kw={"lw": 0.5})
        ax.text(i, m + s + 0.3, f"{m:.1f}%", ha="center", va="bottom",
                fontsize=5.5)

    ax.set_xticks(x_pos)
    ax.set_xticklabels([c[0] for c in conditions])
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title("Standard training (MNIST)")
    ax.set_ylim(85, 94)

    # ---- Panel B: Local learning ----
    ax = axes[1]
    _panel(ax, "B")

    conditions = []
    for norm in [False, True]:
        sub = anorm[(anorm["use_additive_normalization"] == norm) &
                    (anorm["strategy"] == "local_ca")]
        if len(sub):
            m = sub["test_accuracy"].mean() * 100
            s = sub["test_accuracy"].std() * 100
            label = "Add.+norm" if norm else "Additive"
            color = "#5B8AC4" if norm else COLOR_ADDITIVE
            conditions.append((label, m, s, color))

    x_pos = np.arange(len(conditions))
    for i, (label, m, s, color) in enumerate(conditions):
        ax.bar(i, m, yerr=s, color=color, edgecolor="white", lw=0.3,
               width=0.55, capsize=2, error_kw={"lw": 0.5})
        ax.text(i, m + s + 1.5, f"{m:.1f}%", ha="center", va="bottom",
                fontsize=5.5)

    # Shunting reference
    if lbw is not None:
        full = lbw[lbw["broadcast_bandwidth"] == "full"]
        if len(full):
            ref = full["test_accuracy"].mean() * 100
            ax.axhline(ref, color=COLOR_SHUNTING, lw=1.0, ls="--", alpha=0.7)
            ax.text(1.4, ref + 0.8, f"Shunting\n{ref:.1f}%",
                    fontsize=5, color=COLOR_SHUNTING, ha="center")

    ax.set_xticks(x_pos)
    ax.set_xticklabels([c[0] for c in conditions])
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title("Local learning (MNIST)")
    ax.set_ylim(30, 72)

    fig.subplots_adjust(left=0.12, right=0.97, bottom=0.15, top=0.88,
                        wspace=0.50)
    _save(fig, "fig_s7_additive_norm")
    _save(fig, "fig_additive_norm_control")
    plt.close(fig)


# ===================================================================
# Main
# ===================================================================
def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)

    figure5()
    summarize_cue_routing_runs([])
    build_cue_routing_figure(CUE_SUMMARY_CSV)
    figure_s3()
    figure_s5()
    figure_s6()
    figure_s7()

    print("\nDone! All revision figures generated.")


if __name__ == "__main__":
    main()
