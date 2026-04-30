#!/usr/bin/env python3
"""Generate publication-quality figures for the NeurIPS 2026 paper.

This script intentionally does NOT generate Figure 1.
The authoritative Figure 1 is produced by `generate_figure1_schematic.py`.

Produces the current manuscript figures used by the submission draft.

This script still contains a few legacy helper panels, but by default it writes
only the figures that are referenced by `local_credit_assignment_body.tex`.

  Main:
    fig2_competence_regime.pdf     (4 panels: competence, IE dose-response, shunting advantage, F-MNIST)
    fig3_gradient_fidelity.pdf     (4 panels: cosine, gradient norms, alignment dynamics, factorization)
    fig_additional_stress_tests.pdf
  Appendix:
    fig_s1_calibration.pdf         (2x2: capacity, rule ranking, decoder, broadcast)
    fig_s2_gradient_extended.pdf   (2x2: scale mismatch, noise IE detail, MNIST IE detail, FMNIST seeds)
    fig_s4_verification.pdf        (1x3: MNIST seeds, CG seeds, HSIC ablation)

Usage:
    cd /n/holylabs/LABS/kempner_dev/Users/hsafaai/Code/dendritic-modeling
    PYTHONPATH=src:$PYTHONPATH python drafts/dendritic-local-learning/scripts/generate_neurips_figures.py
"""

import os
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DRAFT_DIR = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(DRAFT_DIR, "data")
FIGURES_DIR = os.path.join(DRAFT_DIR, "figures")
ANALYSIS_DIR = os.path.join(DRAFT_DIR, "analysis")
COMPETENCE_SUMMARY_CSV = os.path.join(
    FIGURES_DIR, "data", "competence_summary_20260422.csv"
)
CIFAR10_BP_SUMMARY_CSV = os.path.join(
    ANALYSIS_DIR, "cifar10_compactei_depth4", "cifar10_compactei_depth4_grouped_summary.csv"
)
CIFAR10_LOCALCA_MECH_SUMMARY_CSV = os.path.join(
    ANALYSIS_DIR,
    "cifar10_compactei_depth4_decoderfix_mechanism_5seed",
    "cifar10_compactei_depth4_decoderfix_mechanism_summary.csv",
)
RANK_BRIDGE_NOISE_CSV = os.path.join(
    ANALYSIS_DIR,
    "rank_bridge_nonnegativeinput_fix",
    "noise_resilience_rank_bridge_summary.csv",
)
CUE_RANK_STRUCTURE_CSV = os.path.join(
    ANALYSIS_DIR,
    "rank_bridge_activation_corrected",
    "cue_routing_rank_structure_summary.csv",
)

BUNDLE = (
    "/n/holylfs06/LABS/kempner_project_b/Lab/dendritic/HS/LOCAL_LEARNING"
    "/analysis/publication_bundle_faircheck_plus_20260225"
)
LOCAL_MISMATCH_CSV = (
    "/n/holylfs06/LABS/kempner_project_b/Lab/dendritic/HS/LOCAL_LEARNING"
    "/analysis/local_mismatch_recheck_20260224_summary.csv"
)
FMNIST_SUMMARY_CSV = os.path.join(
    ANALYSIS_DIR, "fashion_mnist_competence_activation_corrected", "fashion_mnist_competence_summary.csv"
)
FMNIST_RUNS_CSV = os.path.join(
    ANALYSIS_DIR, "fashion_mnist_competence_activation_corrected", "fashion_mnist_competence_runs.csv"
)
STANDARD_CEILING_SUMMARY_CSV = os.path.join(
    ANALYSIS_DIR, "standard_ceiling_refresh_5seed", "standard_ceiling_refresh_summary.csv"
)
IE_PERF_SUMMARY_CSV = os.path.join(
    ANALYSIS_DIR,
    "gradient_fidelity_vs_ie_nonnegativeinput_fix",
    "gradient_fidelity_summary.csv",
)
THEORY_IE_SUMMARY_CSV = os.path.join(
    ANALYSIS_DIR,
    "theory_diag_gradient_fidelity_vs_ie_nonnegativeinput_fix_summary",
    "theory_diag_by_condition.csv",
)
THEORY_IE_RUNS_CSV = os.path.join(
    ANALYSIS_DIR,
    "theory_diag_gradient_fidelity_vs_ie_nonnegativeinput_fix",
    "run_summary.csv",
)
CORE_FAIR_TUNING_CSV = os.path.join(ANALYSIS_DIR, "core_fair_tuning.csv")
PHASE2B_GAP_CLOSING_CSV = os.path.join(ANALYSIS_DIR, "phase2b_gap_closing.csv")

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
COLOR_SHUNTING = "#18864B"
COLOR_ADDITIVE = "#2D5DA8"
COLOR_BACKPROP = "#666666"
COLOR_POINT_MLP = "#999999"
COLOR_NOISE = "#E67E22"
COLOR_FASHION = "#8E44AD"

EXC_COLOR = "#2166AC"
INH_COLOR = "#B2182B"
DEN_COLOR = "#4DAF4A"
SOMA_COLOR = "#FF7F00"
RULE3_COLOR = "#66C2A5"
RULE4_COLOR = "#FC8D62"
RULE5_COLOR = "#8DA0CB"

W = 5.5  # NeurIPS single-column width
DPI = 300


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from neurips_style import apply_neurips_style, style_axis  # noqa: E402


LABEL_MAP = {
    "dendritic_shunting": "Shunting",
    "dendritic_additive": "Additive",
    "dendritic_mlp": "Dendr. MLP",
    "point_mlp": "Point MLP",
}
DATASET_LABEL = {
    "mnist": "MNIST",
    "fashion_mnist": "F-MNIST",
    "context_gating": "Context\nGating",
    "noise_resilience": "Noise\nResil.",
    "info_shunting": "Info\nShunt.",
    "cifar10": "CIFAR-10",
}


def _panel(ax, label, x=-0.24, y=1.15):
    """Place bold panel label (A, B, C, ...) well above the axes to avoid title overlap."""
    ax.text(x, y, label, transform=ax.transAxes, fontsize=13.2,
            fontweight="bold", va="top", ha="left")


def _save(fig, name):
    os.makedirs(FIGURES_DIR, exist_ok=True)
    for ext in ("pdf", "png"):
        p = os.path.join(FIGURES_DIR, f"{name}.{ext}")
        fig.savefig(p, dpi=DPI)
    print(f"  Saved: {name}.{{pdf,png}}")


def _csv(filename, bundle=False):
    if bundle:
        path = os.path.join(BUNDLE, filename)
    else:
        path = os.path.join(DATA_DIR, filename)
    if not os.path.isfile(path):
        warnings.warn(f"CSV not found: {path}", stacklevel=2)
        return None
    return pd.read_csv(path)


def _csv_path(path):
    if not os.path.isfile(path):
        warnings.warn(f"CSV not found: {path}", stacklevel=2)
        return None
    return pd.read_csv(path)


# ===================================================================
# Legacy Figure 1 generator
# ===================================================================

def _draw_arrow(ax, x1, y1, x2, y2, color="k", lw=1.0, style="-|>"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops={"arrowstyle": style, "color": color, "lw": lw}, zorder=4)


def fig1_panel_a(ax):
    """Panel A: Dendritic architecture — forward + backward on single diagram."""
    ax.set_xlim(-1.0, 6.4)
    ax.set_ylim(-1.2, 4.2)
    ax.set_aspect("equal")
    ax.axis("off")

    # ── Compartment positions (spread out vertically) ──
    soma_x, soma_y = 4.8, 1.7
    px, py = 2.6, 1.7           # proximal branch
    d_pos = [(0.8, 3.2), (0.8, 0.2)]  # distal branches — well separated

    # ── Soma (larger) ──
    soma_r = 0.57
    ax.add_patch(plt.Circle((soma_x, soma_y), soma_r, fc=SOMA_COLOR, ec="k",
                             lw=1.2, alpha=0.45, zorder=5))
    ax.text(soma_x, soma_y, "soma", ha="center", va="center",
            fontsize=5, fontweight="bold", zorder=6)

    # ── Compartment boxes (rounded rectangles — proximal larger) ──
    comp_w, comp_h = 0.85, 0.55
    prox_w, prox_h = 1.60, 0.80
    # Draw each compartment with appropriate size
    for cx, cy, lbl in [(px, py, "proximal"),
                         (d_pos[0][0], d_pos[0][1], "distal"),
                         (d_pos[1][0], d_pos[1][1], "distal")]:
        bw = prox_w if lbl == "proximal" else comp_w
        bh = prox_h if lbl == "proximal" else comp_h
        box = FancyBboxPatch((cx - bw/2, cy - bh/2), bw, bh,
                              boxstyle="round,pad=0.05", fc=DEN_COLOR, ec="k",
                              lw=0.8, alpha=0.25, zorder=3)
        ax.add_patch(box)
        fs = 5
        ax.text(cx, cy, lbl, ha="center", va="center", fontsize=fs,
                fontweight="bold", color="#2E7D32", zorder=6)

    # ── Dendritic conductance arrows (green, forward flow) ──
    _draw_arrow(ax, px + prox_w/2 + 0.06, py, soma_x - soma_r - 0.02, soma_y,
                color=DEN_COLOR, lw=1.4)
    for dx, dy in d_pos:
        off = 0.14 if dy > py else -0.14
        _draw_arrow(ax, dx + comp_w/2 + 0.06, dy,
                    px - prox_w/2 - 0.06, py + off, color=DEN_COLOR, lw=1.2)

    # ── Excitatory synapses (triangles, blue) ──
    tri_size = 0.10
    for bx, by, is_prox in [(d_pos[0][0], d_pos[0][1], False),
                              (d_pos[1][0], d_pos[1][1], False),
                              (px, py, True)]:
        bw = prox_w if is_prox else comp_w
        for yo in [0.14, -0.14]:
            sx = bx - bw/2 - 0.38
            sy = by + yo
            tri = plt.Polygon(
                [(sx - tri_size, sy - tri_size*0.7),
                 (sx + tri_size, sy),
                 (sx - tri_size, sy + tri_size*0.7)],
                fc=EXC_COLOR, ec="k", lw=0.5, alpha=0.8, zorder=5)
            ax.add_patch(tri)
            _draw_arrow(ax, sx + tri_size + 0.02, sy,
                        bx - bw/2 - 0.06, by + yo * 0.3,
                        color=EXC_COLOR, lw=0.7)

    # ── Inhibitory synapses (circles, red) — one per branch, on top ──
    for bx, by, is_prox in [(d_pos[0][0], d_pos[0][1], False),
                              (d_pos[1][0], d_pos[1][1], False),
                              (px, py, True)]:
        bh = prox_h if is_prox else comp_h
        sx, sy = bx + 0.15, by + bh/2 + 0.28
        ax.add_patch(plt.Circle((sx, sy), 0.09, fc=INH_COLOR, ec="k",
                                 lw=0.5, alpha=0.8, zorder=5))
        _draw_arrow(ax, sx, sy - 0.09, bx + 0.15, by + bh/2 + 0.03,
                    color=INH_COLOR, lw=0.7)

    # ── Output ──
    _draw_arrow(ax, soma_x + soma_r + 0.02, soma_y, 5.8, soma_y, color="k", lw=1.4)
    ax.text(5.95, soma_y, "$\\hat{y}$", fontsize=9, va="center")

    # ── Broadcast error (dashed red arrows, backward) ──
    for bx, by, is_prox in [
        (px, py, True),
        (d_pos[0][0], d_pos[0][1], False),
        (d_pos[1][0], d_pos[1][1], False),
    ]:
        bw = prox_w if is_prox else comp_w
        ax.annotate("", xy=(bx + bw/2 + 0.08, by + 0.05),
                    xytext=(soma_x - soma_r - 0.02, soma_y + 0.12),
                    arrowprops={
                        "arrowstyle": "->",
                        "color": INH_COLOR,
                        "lw": 1.0,
                        "ls": (0, (4, 3)),
                    }, zorder=2)

    # ── Legend (with x^I and x^E as entries, moved lower) ──
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], marker=">", color=EXC_COLOR, markerfacecolor=EXC_COLOR,
               markersize=5, lw=0.7, label="$x^E$ (excitatory input)"),
        Line2D([0], [0], marker="o", color=INH_COLOR, markerfacecolor=INH_COLOR,
               markersize=4, lw=0.7, label="$x^I$ (inhibitory input)"),
        mpatches.Patch(fc=DEN_COLOR, ec="k", lw=0.4, alpha=0.25),
        Line2D([0], [0], color=INH_COLOR, lw=1.0, ls="--"),
    ]
    labels = ["$x^E$: excitatory input", "$x^I$: inhibitory input",
              "Dendritic compartment", "Error broadcast"]
    ax.legend(handles, labels, loc="lower left", fontsize=6.5,
              framealpha=0.95, handlelength=1.2, handletextpad=0.3,
              borderpad=0.3, labelspacing=0.3, bbox_to_anchor=(0.02, -0.18))


def fig1_panel_c(ax):
    """Panel B: Rule hierarchy (3F -> 4F -> 5F) — compact layout."""
    ax.set_xlim(-0.75, 4.3)
    ax.set_ylim(-0.1, 2.55)
    ax.axis("off")

    rules = [
        ("3F", 2.0, RULE3_COLOR,
         r"$\Delta g \propto x_j (E_j{-}V_n) \cdot \delta$"),
        ("4F", 1.15, RULE4_COLOR,
         r"$\Delta g \propto x_j (E_j{-}V_n) \cdot \delta \cdot \rho$"),
        ("5F", 0.3, RULE5_COLOR,
         r"$\Delta g \propto x_j (E_j{-}V_n) \cdot \delta \cdot \rho \cdot \phi$"),
    ]
    box_left = -0.20
    box_w = 4.2
    box_h = 0.65
    for name, yc, color, eq in rules:
        box = FancyBboxPatch(
            (box_left, yc - box_h / 2), box_w, box_h,
            boxstyle="round,pad=0.04",
            fc=color, ec="k", lw=0.5, alpha=0.12, zorder=2)
        ax.add_patch(box)
        # Label OUTSIDE the box on the left
        ax.text(box_left - 0.12, yc, name, ha="right", va="center",
                fontsize=9, fontweight="bold", color=color, zorder=5)
        ax.text(box_left + 0.12, yc, eq, ha="left", va="center",
                fontsize=6.5, zorder=5)

    # Arrows between rules (centered in box)
    arrow_x = box_left + box_w / 2
    for yt, yb in [(2.0 - box_h / 2 - 0.04, 1.15 + box_h / 2 + 0.04),
                    (1.15 - box_h / 2 - 0.04, 0.3 + box_h / 2 + 0.04)]:
        ax.annotate("", xy=(arrow_x, yb), xytext=(arrow_x, yt),
                    arrowprops={
                        "arrowstyle": "->",
                        "color": "gray",
                        "lw": 0.7,
                        "ls": "--",
                    })


def fig1_panel_learning_curves(ax):
    """Panel C: Learning curves — BP vs local for shunting & additive on MNIST."""
    csv_path = os.path.join(DATA_DIR, "learning_curves_fig1.csv")
    if not os.path.isfile(csv_path):
        warnings.warn(f"Learning curves CSV not found: {csv_path}", stacklevel=2)
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return

    df = pd.read_csv(csv_path)
    df = df[df["dataset"] == "mnist"]

    conditions = [
        ("standard",  "dendritic_shunting",  COLOR_SHUNTING, "-",  "Shunting BP"),
        ("local_ca",  "dendritic_shunting",  COLOR_SHUNTING, "--", "Shunting local"),
        ("standard",  "dendritic_additive",  COLOR_ADDITIVE, "-",  "Additive BP"),
        ("local_ca",  "dendritic_additive",  COLOR_ADDITIVE, "--", "Additive local"),
    ]

    for strategy, core, color, ls, label in conditions:
        sub = df[(df["strategy"] == strategy) & (df["core_type"] == core)]
        if sub.empty:
            continue
        grouped = sub.groupby("epoch")["test_acc"]
        mean = grouped.mean()
        sem = grouped.std() / np.sqrt(grouped.count())
        epochs = mean.index.values
        ax.plot(epochs, mean.values * 100, color=color, ls=ls, lw=1.2, label=label)
        ax.fill_between(epochs, (mean - sem).values * 100, (mean + sem).values * 100,
                         color=color, alpha=0.10)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)", labelpad=1)
    ax.legend(loc="upper center", fontsize=6.5, ncol=2, handlelength=1.5,
              columnspacing=0.8, bbox_to_anchor=(0.5, 1.0))
    ax.set_ylim(5, 100)
    ax.set_xlim(0, 200)
    ax.axhline(10, color="gray", ls=":", lw=0.5, zorder=0)  # chance level


def figure1_legacy():
    """Legacy Figure 1 generator kept only for reference.

    The submission figure is generated by `generate_figure1_schematic.py`.
    This function is intentionally not called from `main()` to avoid
    overwriting the authoritative 4-panel Figure 1.
    """
    print("\n--- Legacy Figure 1 (not used for submission) ---")
    fig = plt.figure(figsize=(W, 3.2))
    gs = fig.add_gridspec(1, 3, width_ratios=[0.95, 0.70, 0.85],
                          wspace=0.25, left=0.01, right=0.99,
                          top=0.88, bottom=0.02)

    ax_a = fig.add_subplot(gs[0])
    ax_b = fig.add_subplot(gs[1])
    ax_c = fig.add_subplot(gs[2])

    fig1_panel_a(ax_a)
    fig1_panel_c(ax_b)  # rule hierarchy
    fig1_panel_learning_curves(ax_c)

    # Place all panel labels at the same figure-level y position
    fig_top = 0.97
    for ax_i, lbl, xoff in [(ax_a, "A", -0.06), (ax_b, "B", -0.08),
                              (ax_c, "C", -0.05)]:
        # Convert axes-fraction x to figure coords
        bbox = ax_i.get_position()
        fig_x = bbox.x0 + xoff * bbox.width
        fig.text(fig_x, fig_top, lbl, fontsize=11, fontweight="bold",
                 va="top", ha="left")

    # Panel titles — all at the same figure y, derived from Panel C axes top
    fig.canvas.draw()  # force layout so positions are computed
    bbox_c = ax_c.get_position()
    title_y = bbox_c.y1 + 0.005  # just above Panel C axes top
    for ax_i, title in [(ax_a, "Dendritic network"),
                          (ax_b, "Local learning rules"),
                          (ax_c, "Learning dynamics (MNIST)")]:
        bbox = ax_i.get_position()
        fig_x = bbox.x0 + bbox.width / 2
        fig.text(fig_x, title_y, title, fontsize=7, ha="center", va="bottom")

    _save(fig, "fig1_model_and_credit")
    plt.close(fig)


# ===================================================================
# Figure 2b — Local Rule and Broadcast Design
# ===================================================================
def figure_rule_feedback_design():
    print("\n--- Figure: Local Rule & Feedback Design ---")
    fig, axes = plt.subplots(
        1,
        4,
        figsize=(7.35, 3.0),
        gridspec_kw={"wspace": 0.50, "width_ratios": [1.04, 1.06, 1.26, 1.10]},
    )
    axes = axes.ravel()

    # ---- Panel A: 3F/4F/5F rule family ----
    ax = axes[0]
    _panel(ax, "A")
    style_axis(ax, grid="y")

    rule_data = {
        "MNIST": {"3F": 62.2, "4F": 62.8, "5F": 91.6},
        "CG": {"3F": 39.6, "4F": 41.1, "5F": 78.9},
    }
    rules = ["3F", "4F", "5F"]
    rule_colors = [RULE3_COLOR, RULE4_COLOR, RULE5_COLOR]
    x = np.arange(len(rule_data))
    bw = 0.22
    for j, (rule, color) in enumerate(zip(rules, rule_colors)):
        vals = [rule_data[ds][rule] for ds in rule_data]
        ax.bar(
            x + (j - 1) * bw,
            vals,
            bw * 0.92,
            color=color,
            edgecolor="white",
            lw=0.65,
            label=rule,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(list(rule_data.keys()))
    ax.set_ylabel("Top-10 test (%)")
    ax.set_ylim(30, 100)
    ax.set_title("Eligibility rule")
    ax.legend(
        fontsize=9.2,
        loc="upper left",
        ncol=3,
        frameon=False,
        handlelength=0.9,
        columnspacing=0.55,
        handletextpad=0.25,
    )

    # ---- Panel B: broadcast definition on MNIST ----
    ax = axes[1]
    _panel(ax, "B")
    style_axis(ax, grid="y")

    broadcast_groups = ["Rank-1\nshared", "Local\nmismatch"]
    core_entries = [
        ("Shunt.", COLOR_SHUNTING, [91.2, 14.6], [0.5, 4.6]),
        ("Add.", COLOR_ADDITIVE, [89.4, 34.2], [0.7, 5.8]),
    ]
    x = np.arange(len(broadcast_groups))
    bw = 0.30
    for j, (label, color, vals, errs) in enumerate(core_entries):
        ax.bar(
            x + (j - 0.5) * bw,
            vals,
            bw * 0.92,
            yerr=errs,
            color=color,
            edgecolor="white",
            lw=0.65,
            capsize=2.5,
            error_kw={"lw": 1.0},
            label=label,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(broadcast_groups)
    ax.set_ylabel("MNIST test (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Error definition")
    ax.legend(fontsize=9.2, loc="upper right", frameon=False,
              handlelength=0.9, handletextpad=0.3)

    # ---- Panel C: rank/propagation ladder on noise resilience ----
    ax = axes[2]
    _panel(ax, "C")
    style_axis(ax, grid="y")

    rank_noise = _csv_path(RANK_BRIDGE_NOISE_CSV)
    if rank_noise is not None:
        plot_rows = [
            ("R1", "per_soma", 4, False, COLOR_SHUNTING),
            ("Path", "per_soma", 4, True, "#4DAF4A"),
            ("K2", "low_rank", 2, False, "#F39C12"),
            ("K4", "low_rank", 4, False, "#E67E22"),
            ("K8", "low_rank", 8, False, "#D35400"),
            ("Or.", "path_transport", 4, False, "#6C3483"),
        ]
        vals, errs, labels, colors = [], [], [], []
        for label, mode, rank, path_prop, color in plot_rows:
            row = rank_noise[
                (rank_noise["broadcast_mode"] == mode)
                & (rank_noise["broadcast_rank"] == rank)
                & (rank_noise["use_path_propagation"] == path_prop)
            ]
            if len(row) == 0:
                continue
            labels.append(label)
            vals.append(float(row.iloc[0]["test_accuracy_mean"]) * 100)
            errs.append(float(row.iloc[0]["test_accuracy_std"]) * 100)
            colors.append(color)
        xpos = np.arange(len(vals))
        ax.bar(xpos, vals, 0.68, yerr=errs, color=colors, edgecolor="white",
               lw=0.75, capsize=2.5, error_kw={"lw": 1.15})
        ax.set_xticks(xpos)
        ax.set_xticklabels(labels, fontsize=9.0)
        ax.set_ylim(35, 90)
    ax.set_ylabel("Noise test (%)")
    ax.set_title("Error propagation")

    # ---- Panel D: routed feedback rank and structure ----
    ax = axes[3]
    _panel(ax, "D")
    style_axis(ax, grid="y")

    cue_rank = _csv_path(CUE_RANK_STRUCTURE_CSV)
    if cue_rank is not None:
        cue_rows = [
            ("R1", "per_soma", 2, COLOR_SHUNTING),
            ("K1", "low_rank", 1, "#F5B041"),
            ("K2", "low_rank", 2, "#E67E22"),
            ("K4", "low_rank", 4, "#BA4A00"),
            ("PV", "pathway_vector", 2, "#1F77B4"),
        ]
        vals, errs, labels, colors = [], [], [], []
        for label, mode, rank, color in cue_rows:
            row = cue_rank[
                (cue_rank["broadcast_mode"] == mode)
                & (cue_rank["broadcast_rank"] == rank)
            ]
            if len(row) == 0:
                continue
            labels.append(label)
            vals.append(float(row.iloc[0]["test_accuracy_mean"]) * 100)
            errs.append(float(row.iloc[0]["test_accuracy_std"]) * 100)
            colors.append(color)
        xpos = np.arange(len(vals))
        ax.bar(xpos, vals, 0.68, yerr=errs, color=colors, edgecolor="white",
               lw=0.75, capsize=2.5, error_kw={"lw": 1.15})
        ax.set_xticks(xpos)
        ax.set_xticklabels(labels, fontsize=9.0)
        ax.set_ylim(55, 100)
    ax.set_ylabel("Cue-routing test (%)")
    ax.set_title("Routed task")

    fig.subplots_adjust(left=0.062, right=0.992, bottom=0.23, top=0.82, wspace=0.50)
    _save(fig, "fig4_rule_feedback_design")
    plt.close(fig)


# ===================================================================
# Figure 2 — Competence & Regime Dependence
# ===================================================================
def figure2():
    print("\n--- Figure 2: Competence & Regime Dependence ---")

    competence = _csv_path(COMPETENCE_SUMMARY_CSV)
    ceilings = _csv_path(STANDARD_CEILING_SUMMARY_CSV)
    core = _csv_path(CORE_FAIR_TUNING_CSV)
    p2b = _csv_path(PHASE2B_GAP_CLOSING_CSV)
    fmnist = _csv_path(FMNIST_SUMMARY_CSV)
    ie_data = _csv_path(IE_PERF_SUMMARY_CSV)

    fig, axes = plt.subplots(
        1,
        4,
        figsize=(7.35, 3.0),
        gridspec_kw={"wspace": 0.48, "width_ratios": [1.16, 1.18, 1.02, 1.00]},
    )
    axes = axes.ravel()

    # ---- Panel A: Multi-benchmark bars ----
    ax = axes[0]
    _panel(ax, "A")

    datasets_info = []

    if competence is not None:
        def _pick_competence(dataset, network_type, strategy):
            sub = competence[
                (competence["dataset"] == dataset)
                & (competence["network_type"] == network_type)
                & (competence["strategy"] == strategy)
            ]
            return None if len(sub) == 0 else sub.iloc[0]

        for dataset, label in [
            ("mnist", "MNIST"),
            ("fashion_mnist", "F-MNIST"),
            ("context_gating", "CG"),
        ]:
            bp = _pick_competence(dataset, "dendritic_shunting", "standard")
            shunt = _pick_competence(dataset, "dendritic_shunting", "local_ca")
            add = _pick_competence(dataset, "dendritic_additive", "local_ca")
            if bp is None or shunt is None:
                continue
            datasets_info.append(
                (
                    label,
                    float(bp["test_accuracy_mean"]),
                    float(bp["test_accuracy_std"]),
                    float(shunt["test_accuracy_mean"]),
                    float(shunt["test_accuracy_std"]),
                    None if add is None else float(add["test_accuracy_mean"]),
                    0 if add is None else float(add["test_accuracy_std"]),
                )
            )
    else:
        # MNIST
        bp_mnist = None
        if ceilings is not None:
            r = ceilings[(ceilings["dataset"] == "mnist") & (ceilings["network_type"] == "dendritic_shunting")]
            if len(r):
                bp_mnist = r.iloc[0]["test_accuracy_mean"]
        local_shunt_mnist, local_shunt_mnist_e = None, 0
        local_add_mnist, local_add_mnist_e = None, 0
        if core is not None:
            for nt, store in [("dendritic_shunting", "shunt"), ("dendritic_additive", "add")]:
                sub = core[(core["dataset"] == "mnist") & (core["network_type"] == nt) &
                           (core["rule_variant"] == "5f") & (core["error_broadcast_mode"] == "per_soma") &
                           (core["decoder_update_mode"] == "local")]
                if len(sub):
                    r = sub.iloc[0]
                    if store == "shunt":
                        local_shunt_mnist = r["test_accuracy_mean"]
                        local_shunt_mnist_e = r["test_accuracy_std"]
                    else:
                        local_add_mnist = r["test_accuracy_mean"]
                        local_add_mnist_e = r["test_accuracy_std"]
        if bp_mnist:
            datasets_info.append(("MNIST", bp_mnist, 0, local_shunt_mnist,
                                  local_shunt_mnist_e, local_add_mnist, local_add_mnist_e))

        # Fashion-MNIST
        if fmnist is not None:
            bp_s = fmnist[(fmnist["network_type"] == "dendritic_shunting") & (fmnist["strategy"] == "standard")]
            loc_s = fmnist[(fmnist["network_type"] == "dendritic_shunting") & (fmnist["strategy"] == "local_ca")]
            loc_a = fmnist[(fmnist["network_type"] == "dendritic_additive") & (fmnist["strategy"] == "local_ca")]
            if len(bp_s) and len(loc_s) and len(loc_a):
                datasets_info.append(("F-MNIST",
                                      bp_s.iloc[0]["test_accuracy_mean"],
                                      bp_s.iloc[0]["test_accuracy_std"],
                                      loc_s.iloc[0]["test_accuracy_mean"], loc_s.iloc[0]["test_accuracy_std"],
                                      loc_a.iloc[0]["test_accuracy_mean"], loc_a.iloc[0]["test_accuracy_std"]))

        # Context gating
        bp_cg = None
        if ceilings is not None:
            r = ceilings[(ceilings["dataset"] == "context_gating") & (ceilings["network_type"] == "dendritic_shunting")]
            if len(r):
                bp_cg = r.iloc[0]["test_accuracy_mean"]
        local_shunt_cg, local_shunt_cg_e = None, 0
        if p2b is not None:
            sub = p2b[(p2b["dataset"] == "context_gating") & p2b["hsic_enabled"] &
                      (p2b["hsic_weight"] == 0.01) & (p2b["error_broadcast_mode"] == "per_soma")]
            if len(sub):
                local_shunt_cg = sub.iloc[0]["test_accuracy_mean"]
                local_shunt_cg_e = sub.iloc[0]["test_accuracy_std"]
        if bp_cg and local_shunt_cg:
            datasets_info.append(("CG", bp_cg, 0, local_shunt_cg, local_shunt_cg_e, None, 0))

    # Plot grouped bars
    n_ds = len(datasets_info)
    x_base = np.arange(n_ds)
    bar_w = 0.22

    for i, (_ds_name, bp_val, bp_err, shunt_val, shunt_err, add_val, add_err) in enumerate(datasets_info):
        ax.bar(i - bar_w, bp_val * 100, bar_w * 0.88, yerr=bp_err * 100,
               color=COLOR_BACKPROP, edgecolor="white", lw=0.75,
               capsize=2.6, error_kw={"lw": 1.15})
        if shunt_val is not None:
            ax.bar(i, shunt_val * 100, bar_w * 0.88, yerr=shunt_err * 100,
                   color=COLOR_SHUNTING, edgecolor="white", lw=0.75,
                   capsize=2.6, error_kw={"lw": 1.15})
        if add_val is not None:
            ax.bar(i + bar_w, add_val * 100, bar_w * 0.88, yerr=add_err * 100,
                   color=COLOR_ADDITIVE, edgecolor="white", lw=0.75,
                   capsize=2.6, error_kw={"lw": 1.15})

    ax.set_xticks(x_base)
    short_dataset_labels = {"MNIST": "MN", "F-MNIST": "FMN", "CG": "CG"}
    ax.set_xticklabels([short_dataset_labels.get(d[0], d[0]) for d in datasets_info])
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title("Competence")

    # Legend
    legend_handles = [
        mpatches.Patch(color=COLOR_BACKPROP, label="Backprop"),
        mpatches.Patch(color=COLOR_SHUNTING, label="Shunt. (local)"),
        mpatches.Patch(color=COLOR_ADDITIVE, label="Add. (local)"),
    ]
    ax.legend(
        handles=legend_handles,
        fontsize=8.4,
        loc="upper right",
        bbox_to_anchor=(0.99, 0.99),
        ncol=1,
        frameon=True,
        framealpha=0.86,
        facecolor="white",
        edgecolor="0.85",
        handlelength=1.0,
        handletextpad=0.35,
        borderpad=0.25,
        labelspacing=0.25,
    )

    all_vals = [d[1]*100 for d in datasets_info if d[1]] + \
               [d[3]*100 for d in datasets_info if d[3]] + \
               [d[5]*100 for d in datasets_info if d[5]]
    if all_vals:
        ax.set_ylim(max(0, min(all_vals) - 6), max(all_vals) + 2)

    # ---- Panel B: IE dose-response ----
    ax = axes[1]
    _panel(ax, "B")

    if ie_data is not None:
        for ct, ds, color, ls, marker, ms in [
            ("dendritic_shunting", "mnist", COLOR_SHUNTING, "-", "o", 3),
            ("dendritic_additive", "mnist", COLOR_ADDITIVE, "-", "s", 3),
            ("dendritic_shunting", "noise_resilience", COLOR_SHUNTING, "--", "^", 3),
            ("dendritic_additive", "noise_resilience", COLOR_ADDITIVE, "--", "v", 3),
        ]:
            sub = ie_data[(ie_data["network_type"] == ct) & (ie_data["dataset"] == ds)].copy()
            sub = sub.sort_values("ie_value")
            if len(sub) == 0:
                continue
            short = "Sh" if "shunting" in ct else "Add"
            ds_short = "MN" if ds == "mnist" else "noise"
            ax.errorbar(sub["ie_value"], sub["test_accuracy_mean"] * 100,
                        yerr=sub["test_accuracy_std"] * 100,
                        marker=marker, markersize=ms + 1.5, linewidth=2.1, capsize=2.5,
                        color=color, linestyle=ls, label=f"{short}-{ds_short}",
                        capthick=1.0)

    ax.set_xlabel("$N_I$ per branch")
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title("Inhibition sweep")
    ax.legend(
        fontsize=8.4,
        loc="lower left",
        ncol=2,
        handlelength=1.0,
        handletextpad=0.25,
        columnspacing=0.35,
        borderpad=0.18,
        labelspacing=0.22,
        frameon=True,
        framealpha=0.86,
        facecolor="white",
        edgecolor="0.85",
    )
    ax.set_ylim(25, 100)

    # ---- Panel C: Shunting advantage ----
    ax = axes[2]
    _panel(ax, "C")

    if ie_data is not None:
        for ds, color, marker, lbl in [
            ("mnist", COLOR_SHUNTING, "o", "MNIST"),
            ("noise_resilience", COLOR_NOISE, "^", "Noise resil."),
        ]:
            shunt = ie_data[(ie_data["network_type"] == "dendritic_shunting") &
                            (ie_data["dataset"] == ds)].copy()
            add = ie_data[(ie_data["network_type"] == "dendritic_additive") &
                          (ie_data["dataset"] == ds)].copy()
            if len(shunt) == 0 or len(add) == 0:
                continue
            merged = pd.merge(shunt, add, on=["dataset", "ie_value"],
                              suffixes=("_s", "_a"))
            merged["delta"] = (merged["test_accuracy_mean_s"] - merged["test_accuracy_mean_a"]) * 100
            merged["delta_err"] = (
                np.sqrt(
                    merged["test_accuracy_std_s"].fillna(0) ** 2
                    + merged["test_accuracy_std_a"].fillna(0) ** 2
                )
                * 100
            )
            merged = merged.sort_values("ie_value")
            ax.errorbar(
                merged["ie_value"],
                merged["delta"],
                yerr=merged["delta_err"],
                marker=marker,
                markersize=6.2,
                linewidth=2.2,
                color=color,
                label=lbl,
                capsize=2,
                capthick=1.0,
            )

    ax.axhline(0, color="black", lw=1.0, ls="--")
    ax.set_xlabel("$N_I$ per branch")
    ax.set_ylabel("Shunting adv. (pp)")
    ax.set_title("Shunting gain")
    ax.legend(fontsize=8.4, loc="upper right", handlelength=1.0)

    # ---- Panel D: Rule-family comparison ----
    ax = axes[3]
    _panel(ax, "D")
    style_axis(ax, grid="y")

    rule_data = {
        "MNIST": {"3F": 62.2, "4F": 62.8, "5F": 91.6},
        "CG": {"3F": 39.6, "4F": 41.1, "5F": 78.9},
    }
    rules = ["3F", "4F", "5F"]
    rule_colors = [RULE3_COLOR, RULE4_COLOR, RULE5_COLOR]
    x = np.arange(len(rule_data))
    bw = 0.23
    for j, (rule, color) in enumerate(zip(rules, rule_colors)):
        vals = [rule_data[ds][rule] for ds in rule_data]
        ax.bar(
            x + (j - 1) * bw,
            vals,
            bw * 0.92,
            color=color,
            edgecolor="white",
            lw=0.75,
            label=rule,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(list(rule_data.keys()), fontsize=9.2)
    ax.set_ylabel("Top-10 test (%)")
    ax.set_title("Rule family")
    ax.set_ylim(30, 100)
    ax.legend(
        fontsize=8.4,
        loc="upper left",
        ncol=3,
        frameon=False,
        handlelength=0.9,
        columnspacing=0.45,
        handletextpad=0.22,
    )

    fig.subplots_adjust(left=0.062, right=0.992, bottom=0.24, top=0.82, wspace=0.48)
    _save(fig, "fig2_competence_regime")
    plt.close(fig)


# ===================================================================
# Figure 3 — Gradient Fidelity
# ===================================================================
def figure3():
    print("\n--- Figure 3: Gradient Fidelity ---")
    fig, axes = plt.subplots(
        1,
        4,
        figsize=(7.35, 3.0),
        gridspec_kw={"wspace": 0.46, "width_ratios": [1.00, 0.98, 1.06, 1.16]},
    )
    axes = axes.ravel()

    def _load_gradient_trajectory(rule_variant="5f"):
        """Load trajectory diagnostics and keep one rule variant.

        The trajectory files store one row per parameter tensor, epoch, and
        rule. Aggregated full-gradient norms are therefore computed as the
        Euclidean norm over tensor-wise norms: sqrt(sum_i ||g_i||^2).
        """
        gf_dir = os.path.join(DRAFT_DIR, "analysis", "gradient_fidelity")
        if not os.path.isdir(gf_dir):
            return None

        all_traj = []
        for cfg_name in sorted(os.listdir(gf_dir)):
            csv_path = os.path.join(gf_dir, cfg_name, "gradient_fidelity_trajectory.csv")
            if not os.path.isfile(csv_path):
                continue
            tdf = pd.read_csv(csv_path)
            if "rule_variant" in tdf.columns:
                tdf = tdf[tdf["rule_variant"] == rule_variant].copy()
            if tdf.empty:
                continue
            tdf["config"] = cfg_name

            def _layer_idx(pname):
                parts = pname.split(".")
                for i, p in enumerate(parts):
                    if p == "layers" and i + 1 < len(parts):
                        try:
                            return int(parts[i + 1])
                        except ValueError:
                            pass
                return -1

            tdf["layer_idx"] = tdf["parameter_name"].apply(_layer_idx)
            all_traj.append(tdf)

        if not all_traj:
            return None

        traj = pd.concat(all_traj, ignore_index=True)
        configs = sorted(traj["config"].unique())
        mid = len(configs) // 2
        core_map = {
            cfg: ("shunting" if i < mid else "additive")
            for i, cfg in enumerate(configs)
        }
        traj["core_type"] = traj["config"].map(core_map)
        return traj

    def _full_norm_rows(traj):
        if traj is None:
            return pd.DataFrame()
        rows = []
        for (cfg, core, epoch), sub in traj.groupby(["config", "core_type", "epoch"]):
            local_norm = float(np.sqrt(np.square(sub["local_grad_norm"]).sum()))
            bp_norm = float(np.sqrt(np.square(sub["backprop_grad_norm"]).sum()))
            wcos = float((sub["cosine_similarity"] * sub["numel"]).sum() / sub["numel"].sum())
            rows.append({
                "config": cfg,
                "core_type": core,
                "epoch": int(epoch),
                "local_grad_norm": local_norm,
                "backprop_grad_norm": bp_norm,
                "weighted_cosine": wcos,
            })
        return pd.DataFrame(rows)

    traj_5f = _load_gradient_trajectory("5f")
    norm_df = _full_norm_rows(traj_5f)
    if not norm_df.empty:
        norm_out = os.path.join(ANALYSIS_DIR, "gradient_fidelity", "gradient_norm_dynamics_summary.csv")
        norm_df.to_csv(norm_out, index=False)

    def _final_gradient_stats() -> pd.DataFrame:
        if norm_df.empty:
            return pd.DataFrame()
        final_rows = []
        for _cfg, sub in norm_df.groupby("config"):
            final_rows.append(sub.sort_values("epoch").iloc[-1])
        final = pd.DataFrame(final_rows)
        if final.empty:
            return pd.DataFrame()
        final["scale_mismatch"] = np.abs(
            np.log10(
                np.maximum(final["local_grad_norm"].to_numpy(dtype=float), 1e-12)
                / np.maximum(final["backprop_grad_norm"].to_numpy(dtype=float), 1e-12)
            )
        )
        return (
            final.groupby("core_type")
            .agg(
                n=("weighted_cosine", "count"),
                cosine_mean=("weighted_cosine", "mean"),
                cosine_std=("weighted_cosine", "std"),
                scale_mean=("scale_mismatch", "mean"),
                scale_std=("scale_mismatch", "std"),
            )
            .reset_index()
        )

    final_stats = _final_gradient_stats()

    # ---- Panel A: Exact factorization sanity ----
    ax = axes[0]
    _panel(ax, "A")
    style_axis(ax, grid="y")

    theory_runs = pd.read_csv(THEORY_IE_RUNS_CSV)
    rel_error = theory_runs["factorization_weighted_relative_l2"].to_numpy(dtype=float)
    scale_error = theory_runs["factorization_weighted_scale_mismatch"].to_numpy(dtype=float)
    rng = np.random.default_rng(7)
    for i, (vals, color) in enumerate([
        (rel_error, RULE3_COLOR),
        (scale_error, RULE5_COLOR),
    ]):
        jitter = rng.uniform(-0.08, 0.08, size=len(vals))
        ax.scatter(
            np.full_like(vals, i, dtype=float) + jitter,
            vals,
            s=32,
            color=color,
            alpha=0.72,
            edgecolor="white",
            linewidth=0.75,
            zorder=3,
        )
        ax.hlines(np.median(vals), i - 0.18, i + 0.18, color="black", lw=1.8, zorder=4)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["rel. $L_2$", "scale"], fontsize=9.2)
    ax.set_ylabel("Reconstruction error")
    ax.set_yscale("log")
    ax.set_ylim(1e-10, 1e-4)
    ax.set_title("Factorization")
    ax.grid(axis="y", alpha=0.24, linewidth=0.78)

    # ---- Panel B: Cosine similarity bars ----
    ax = axes[1]
    _panel(ax, "B")
    style_axis(ax, grid="y")

    core_order = ["shunting", "additive"]
    core_labels = ["Shunt.", "Add."]
    core_colors = [COLOR_SHUNTING, COLOR_ADDITIVE]
    if not final_stats.empty:
        means, errs, ns = [], [], []
        for core in core_order:
            row = final_stats[final_stats["core_type"] == core]
            if row.empty:
                means.append(np.nan)
                errs.append(0.0)
                ns.append(0)
            else:
                r = row.iloc[0]
                means.append(float(r["cosine_mean"]))
                errs.append(0.0 if pd.isna(r["cosine_std"]) else float(r["cosine_std"]))
                ns.append(int(r["n"]))
        x_pos = np.arange(len(core_order))
        bars = ax.bar(
            x_pos,
            means,
            yerr=errs,
            color=core_colors,
            edgecolor="white",
            lw=0.75,
            width=0.58,
            capsize=2.6,
            error_kw={"lw": 1.15},
        )
        ax.set_xticks(x_pos)
        ax.set_xticklabels(core_labels, fontsize=9.2)
        ax.set_ylabel("Cosine similarity")
        n_label = min(ns) if ns else 0
        ax.set_title(f"Alignment (n={n_label})")
        ax.set_ylim(-0.06, max(0.18, np.nanmax(np.asarray(means) + np.asarray(errs)) + 0.05))
        ax.axhline(0, color="black", lw=0.8, ls="--")
        for bar_rect, val in zip(bars, means):
            yo = 0.006 if val >= 0 else -0.014
            va = "bottom" if val >= 0 else "top"
            ax.text(
                bar_rect.get_x() + bar_rect.get_width() / 2,
                val + yo,
                f"{val:.3f}",
                ha="center",
                va=va,
                fontsize=8.1,
            )
    else:
        ax.text(0.5, 0.5, "No seeded alignment data", transform=ax.transAxes,
                ha="center", va="center", fontsize=8, color="red")

    # ---- Panel C: Scale mismatch ----
    ax = axes[2]
    _panel(ax, "C")
    style_axis(ax, grid="y")
    if not final_stats.empty:
        means, errs, ns = [], [], []
        for core in core_order:
            row = final_stats[final_stats["core_type"] == core]
            if row.empty:
                means.append(np.nan)
                errs.append(0.0)
                ns.append(0)
            else:
                r = row.iloc[0]
                means.append(float(r["scale_mean"]))
                errs.append(0.0 if pd.isna(r["scale_std"]) else float(r["scale_std"]))
                ns.append(int(r["n"]))
        x_pos = np.arange(len(core_order))
        bars = ax.bar(
            x_pos,
            means,
            yerr=errs,
            color=core_colors,
            edgecolor="white",
            lw=0.75,
            width=0.58,
            capsize=2.6,
            error_kw={"lw": 1.15},
        )
        ax.set_xticks(x_pos)
        ax.set_xticklabels(core_labels, fontsize=9.2)
        ax.set_ylabel(r"$|\log_{10}$ norm ratio$|$")
        n_label = min(ns) if ns else 0
        ax.set_title(f"Scale (n={n_label})")
        ax.set_ylim(0, max(0.65, np.nanmax(np.asarray(means) + np.asarray(errs)) + 0.08))
        for bar_rect, val in zip(bars, means):
            ax.text(
                bar_rect.get_x() + bar_rect.get_width() / 2,
                val + 0.025,
                f"{val:.3f}" if val < 0.2 else f"{val:.2f}",
                ha="center",
                va="bottom",
                fontsize=8.1,
            )
    else:
        ax.text(0.5, 0.5, "No seeded scale data", transform=ax.transAxes,
                ha="center", va="center", fontsize=8, color="red")

    # ---- Panel D: Nonzero-gradient norm dynamics ----
    ax = axes[3]
    _panel(ax, "D")
    style_axis(ax, grid="y")
    if not norm_df.empty:
        final_label_rows = []
        for core, core_label, color in [
            ("shunting", "Shunt.", COLOR_SHUNTING),
            ("additive", "Add.", COLOR_ADDITIVE),
        ]:
            sub = norm_df[norm_df["core_type"] == core]
            for metric, metric_label, linestyle in [
                ("local_grad_norm", "LocalCA", "-"),
                ("backprop_grad_norm", "BP", "--"),
            ]:
                grouped = (
                    sub.groupby("epoch")[metric]
                    .agg(["mean", "std"])
                    .reset_index()
                    .sort_values("epoch")
                )
                x = grouped["epoch"].to_numpy(dtype=float)
                y = grouped["mean"].to_numpy(dtype=float)
                err = grouped["std"].fillna(0.0).to_numpy(dtype=float)
                ax.plot(
                    x,
                    y,
                    color=color,
                    linestyle=linestyle,
                    lw=2.2,
                    marker="o" if linestyle == "-" else "s",
                    markersize=4.8,
                    label=f"{core_label} {metric_label}",
                    zorder=3,
                )
                ax.fill_between(
                    x,
                    np.maximum(y - err, 1e-8),
                    y + err,
                    color=color,
                    alpha=0.10 if linestyle == "-" else 0.06,
                    linewidth=0,
                    zorder=2,
                )
                if len(x):
                    final_label_rows.append(
                        (x[-1], y[-1], color, f"{core_label} {metric_label}")
                    )
        ax.set_yscale("log")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Full gradient norm")
        ax.set_title("Gradient norms")
        ax.set_xlim(-2, 57)
        for x_last, y_last, color, label in final_label_rows:
            ax.text(
                x_last + 1.2,
                y_last,
                label,
                color=color,
                fontsize=7.5,
                va="center",
                ha="left",
                fontweight="bold",
            )
    else:
        ax.text(0.5, 0.5, "No trajectory data found", transform=ax.transAxes,
                ha="center", va="center", fontsize=8, color="red")

    fig.subplots_adjust(left=0.062, right=0.992, bottom=0.25, top=0.82, wspace=0.46)
    _save(fig, "fig3_gradient_fidelity")
    plt.close(fig)


# ===================================================================
# Figure 4 — Scalability & Generalization
# ===================================================================
def figure4():
    print("\n--- Figure 4: Scalability & Generalization ---")

    depth = _csv("depth_scaling.csv", bundle=True)
    noise = _csv("noise_robustness.csv", bundle=True)
    fmnist = _csv_path(FMNIST_SUMMARY_CSV)

    fig, axes = plt.subplots(1, 3, figsize=(W * 1.9, 3.6),
                             gridspec_kw={"wspace": 0.52})

    # ---- Panel A: Depth scaling (LOCAL only — cleaner) ----
    ax = axes[0]
    _panel(ax, "A")

    if depth is not None:
        def _depth(bf_str):
            try:
                return len(bf_str.strip("[]").split(","))
            except Exception:
                return 1

        # Separate local and backprop, show local as solid + backprop as light reference
        for strat, ls, alpha_val in [("local_ca", "-", 1.0), ("standard", "--", 0.35)]:
            for nt in ["dendritic_shunting", "dendritic_additive"]:
                sub = depth[depth["network_type"] == nt].copy()
                if "strategy" in sub.columns:
                    sub = sub[sub["strategy"] == strat].copy()
                elif strat == "standard":
                    continue  # no strategy column = skip backprop
                if len(sub) == 0:
                    continue
                sub["depth"] = sub["branch_factors"].apply(_depth)
                agg = sub.groupby("depth").agg(
                    mean=("test_accuracy_mean", "mean"),
                    std=("test_accuracy_std", "mean")
                ).reset_index().sort_values("depth")
                color = COLOR_SHUNTING if "shunting" in nt else COLOR_ADDITIVE
                lbl_core = "Shunt." if "shunting" in nt else "Add."
                lbl_strat = "local" if strat == "local_ca" else "BP"
                ax.errorbar(agg["depth"], agg["mean"] * 100,
                            yerr=agg["std"] * 100,
                            marker="o", markersize=3, lw=1.0, capsize=1.5,
                            color=color, linestyle=ls, alpha=alpha_val,
                            label=f"{lbl_core} {lbl_strat}")

        ax.set_xlabel("Network depth")
        ax.set_ylabel("Test accuracy (%)")
        ax.set_title("Shunting degrades more gracefully with depth")
        ax.legend(fontsize=7, loc="best", handlelength=1.5,
                  handletextpad=0.3, ncol=1)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    # ---- Panel B: Noise robustness ----
    ax = axes[1]
    _panel(ax, "B")

    if noise is not None:
        for nt in ["dendritic_shunting", "dendritic_additive"]:
            sub = noise[noise["network_type"] == nt].copy()
            if len(sub) == 0:
                continue
            sub = sub.sort_values("error_noise_sigma")
            color = COLOR_SHUNTING if "shunting" in nt else COLOR_ADDITIVE
            ax.errorbar(sub["error_noise_sigma"], sub["test_accuracy_mean"] * 100,
                        yerr=sub["test_accuracy_std"] * 100,
                        marker="o", markersize=3, lw=1.0, capsize=1.5,
                        color=color, label=LABEL_MAP.get(nt, nt))

        ax.set_xlabel(r"Error noise $\sigma$")
        ax.set_ylabel("Test accuracy (%)")
        ax.set_title("Shunting tolerates noisy broadcasts")
        ax.legend(fontsize=7, handlelength=1.0)

    # ---- Panel C: Fashion-MNIST ----
    ax = axes[2]
    _panel(ax, "C")

    if fmnist is not None:
        conditions = []
        for ct in ["dendritic_shunting", "dendritic_additive"]:
            for strat in ["local_ca", "standard"]:
                sub = fmnist[(fmnist["network_type"] == ct) & (fmnist["strategy"] == strat)]
                if len(sub):
                    r = sub.iloc[0]
                    short_ct = "Shunt." if "shunting" in ct else "Add."
                    short_st = "Local" if strat == "local_ca" else "BP"
                    color = COLOR_SHUNTING if "shunting" in ct else COLOR_ADDITIVE
                    alpha = 1.0 if strat == "local_ca" else 0.40
                    conditions.append((f"{short_ct}\n{short_st}",
                                       r["test_accuracy_mean"] * 100,
                                       r["test_accuracy_std"] * 100,
                                       color, alpha))

        x = np.arange(len(conditions))
        for i, c in enumerate(conditions):
            ax.bar(i, c[1], yerr=c[2], color=c[3], alpha=c[4],
                   edgecolor="white", lw=0.3, width=0.55,
                   capsize=1.5, error_kw={"lw": 0.5})
        ax.set_xticks(x)
        ax.set_xticklabels([c[0] for c in conditions], fontsize=7.5)
        ax.set_ylabel("Test accuracy (%)")
        ax.set_title("The gain is modest on cleaner vision tasks")

        all_v = [c[1] for c in conditions]
        ax.set_ylim(max(0, min(all_v) - 4), max(all_v) + 3)

        for i, c in enumerate(conditions):
            ax.text(i, c[1] + c[2] + 0.4,
                    f"{c[1]:.1f}", ha="center", va="bottom", fontsize=7)

    fig.subplots_adjust(left=0.08, right=0.97, bottom=0.15, top=0.90,
                        wspace=0.55)
    _save(fig, "fig_additional_stress_tests")
    plt.close(fig)


# ===================================================================
# Appendix Figure S1 — Capacity Calibration
# ===================================================================
def figure_s1():
    print("\n--- Figure S1: Capacity Calibration ---")

    phase1 = _csv("phase1_best_standard.csv", bundle=True)
    core = _csv("core_fair_tuning.csv", bundle=True)
    mismatch_path = LOCAL_MISMATCH_CSV
    mismatch = pd.read_csv(mismatch_path) if os.path.isfile(mismatch_path) else None

    fig, axes = plt.subplots(2, 2, figsize=(W * 1.9, 7.0),
                             gridspec_kw={"wspace": 0.45, "hspace": 0.52})

    # ---- Panel A: Phase 1 capacity ceilings ----
    ax = axes[0, 0]
    _panel(ax, "A")

    if phase1 is not None:
        p1 = phase1.dropna(subset=["test_accuracy"]).copy()
        ds_order = [d for d in ["mnist", "context_gating", "cifar10", "info_shunting"]
                    if d in p1["dataset"].values]
        arch_order = ["dendritic_shunting", "dendritic_additive", "dendritic_mlp", "point_mlp"]
        arch_colors = {"dendritic_shunting": COLOR_SHUNTING,
                       "dendritic_additive": COLOR_ADDITIVE,
                       "dendritic_mlp": "#D4A017", "point_mlp": COLOR_POINT_MLP}
        n_arch = len(arch_order)
        bw = 0.8 / n_arch
        xb = np.arange(len(ds_order))
        for j, arch in enumerate(arch_order):
            vals = []
            for ds in ds_order:
                r = p1[(p1["dataset"] == ds) & (p1["network_type"] == arch)]
                vals.append(r.iloc[0]["test_accuracy"] * 100 if len(r) else 0)
            off = (j - (n_arch - 1)/2) * bw
            ax.bar(xb + off, vals, bw * 0.9, label=LABEL_MAP.get(arch, arch),
                   color=arch_colors.get(arch, "#999"), edgecolor="white", lw=0.2)
        ax.set_xticks(xb)
        ax.set_xticklabels([DATASET_LABEL.get(d, d) for d in ds_order], fontsize=7.5)
        ax.set_ylabel("Test accuracy (%)")
        ax.set_title("Backprop ceilings")
        ax.legend(fontsize=7, ncol=2, loc="lower left",
                  handlelength=1.0, handletextpad=0.3)

    # ---- Panel B: Rule family ranking ----
    ax = axes[0, 1]
    _panel(ax, "B")

    if core is not None:
        sub = core[(core["dataset"] == "mnist") & (core["error_broadcast_mode"] == "per_soma") &
                   (core["decoder_update_mode"] == "local")].copy()
        rules = ["3f", "4f", "5f"]
        nets = ["dendritic_shunting", "dendritic_additive"]
        bw = 0.3
        xb = np.arange(len(rules))
        for j, nt in enumerate(nets):
            vals, errs = [], []
            for rv in rules:
                r = sub[(sub["rule_variant"] == rv) & (sub["network_type"] == nt)]
                if len(r):
                    vals.append(r.iloc[0]["test_accuracy_mean"] * 100)
                    errs.append(r.iloc[0]["test_accuracy_std"] * 100)
                else:
                    vals.append(0)
                    errs.append(0)
            off = (j - 0.5) * bw
            color = COLOR_SHUNTING if "shunting" in nt else COLOR_ADDITIVE
            ax.bar(xb + off, vals, bw * 0.9, yerr=errs, color=color,
                   edgecolor="white", lw=0.2, capsize=1.5, error_kw={"lw": 0.5},
                   label=LABEL_MAP.get(nt, nt))
        ax.set_xticks(xb)
        ax.set_xticklabels([r.upper() for r in rules])
        ax.set_ylabel("Test accuracy (%)")
        ax.set_title("Rule ranking (MNIST)")
        ax.legend(fontsize=7)
        av = sub["test_accuracy_mean"].dropna() * 100
        if len(av):
            ax.set_ylim(max(0, av.min() - 6), av.max() + 3)

    # ---- Panel C: Decoder locality ----
    ax = axes[1, 0]
    _panel(ax, "C")

    if core is not None:
        sub = core[(core["dataset"] == "mnist") & (core["error_broadcast_mode"] == "per_soma") &
                   (core["rule_variant"] == "5f")].copy()
        dms = ["local", "backprop"]
        nets = ["dendritic_shunting", "dendritic_additive"]
        bw = 0.3
        xb = np.arange(len(dms))
        for j, nt in enumerate(nets):
            vals, errs = [], []
            for dm in dms:
                r = sub[(sub["decoder_update_mode"] == dm) & (sub["network_type"] == nt)]
                if len(r):
                    vals.append(r.iloc[0]["test_accuracy_mean"] * 100)
                    errs.append(r.iloc[0]["test_accuracy_std"] * 100)
                else:
                    vals.append(0)
                    errs.append(0)
            off = (j - 0.5) * bw
            color = COLOR_SHUNTING if "shunting" in nt else COLOR_ADDITIVE
            ax.bar(xb + off, vals, bw * 0.9, yerr=errs, color=color,
                   edgecolor="white", lw=0.2, capsize=1.5, error_kw={"lw": 0.5},
                   label=LABEL_MAP.get(nt, nt))
        ax.set_xticks(xb)
        ax.set_xticklabels(["Local", "Backprop"])
        ax.set_ylabel("Test accuracy (%)")
        ax.set_title("Decoder mode (5F, MNIST)")
        ax.legend(fontsize=7)
        av = sub["test_accuracy_mean"].dropna() * 100
        if len(av):
            ax.set_ylim(max(0, av.min() - 4), av.max() + 2)

    # ---- Panel D: Broadcast mode comparison (fixed x-labels) ----
    ax = axes[1, 1]
    _panel(ax, "D")

    if mismatch is not None:
        # Simplify: group by core x broadcast (ignoring decoder for cleaner plot)
        agg2 = mismatch.groupby(
            ["core_type", "error_broadcast_mode"]
        ).agg(test_mean=("test_acc", "mean"),
              test_std=("test_acc", "std")
        ).reset_index()
        conds = []
        for _, r in agg2.iterrows():
            eb = "per-soma" if r["error_broadcast_mode"] == "per_soma" else "local-mm"
            ct = "Shunt." if "shunting" in r["core_type"] else "Add."
            color = COLOR_SHUNTING if "shunting" in r["core_type"] else COLOR_ADDITIVE
            conds.append((f"{ct}\n{eb}", r["test_mean"]*100, r["test_std"]*100, color))
        conds.sort(key=lambda c: c[1], reverse=True)
        x = np.arange(len(conds))
        ax.bar(x, [c[1] for c in conds], yerr=[c[2] for c in conds],
               color=[c[3] for c in conds], edgecolor="white", lw=0.2,
               capsize=1.5, width=0.55, error_kw={"lw": 0.5})
        ax.set_xticks(x)
        ax.set_xticklabels([c[0] for c in conds], fontsize=7.5)
        ax.set_ylabel("Test accuracy (%)")
        ax.set_title("Broadcast mode (MNIST)")

    fig.subplots_adjust(left=0.08, right=0.97, bottom=0.06, top=0.94)
    _save(fig, "fig_s1_calibration")
    plt.close(fig)


# ===================================================================
# Appendix Figure S2 — Extended Gradient & IE Detail
# ===================================================================
def figure_s2():
    print("\n--- Figure S2: Extended Gradient & IE Detail ---")
    fig, axes = plt.subplots(2, 2, figsize=(W * 1.9, 7.0),
                             gridspec_kw={"wspace": 0.45, "hspace": 0.52})

    # Panel A: Scale mismatch bars (from Table 2)
    ax = axes[0, 0]
    _panel(ax, "A")

    conditions = [
        ("MNIST\nShunt.", 0.117, COLOR_SHUNTING),
        ("MNIST\nAdd.", 1.053, COLOR_ADDITIVE),
        ("CG\nShunt.", 0.036, COLOR_SHUNTING),
        ("CG\nAdd.", 2.154, COLOR_ADDITIVE),
    ]
    x = np.arange(len(conditions))
    bars = ax.bar(x, [c[1] for c in conditions], color=[c[2] for c in conditions],
                  edgecolor="white", lw=0.4, width=0.55)
    ax.set_xticks(x)
    ax.set_xticklabels([c[0] for c in conditions], fontsize=7.5)
    ax.set_ylabel("Scale mismatch\n(||local|| / ||BP||)")
    ax.set_title("Scale mismatch")
    ax.set_yscale("log")
    ax.axhline(1.0, color="black", lw=0.4, ls="--", label="Ideal (1.0)")
    ax.legend(fontsize=7)

    for bar_rect, c in zip(bars, conditions):
        ax.text(bar_rect.get_x() + bar_rect.get_width()/2, c[1] * 1.3,
                f"{c[1]:.3f}", ha="center", va="bottom", fontsize=7)

    # Panel B: Noise resilience IE detail with error bands
    ax = axes[0, 1]
    _panel(ax, "B")

    ie_data = _csv_path(IE_PERF_SUMMARY_CSV)
    if ie_data is not None:
        for ct, color, marker in [("dendritic_shunting", COLOR_SHUNTING, "o"),
                                   ("dendritic_additive", COLOR_ADDITIVE, "s")]:
            sub = ie_data[(ie_data["network_type"] == ct) &
                          (ie_data["dataset"] == "noise_resilience")].copy()
            sub = sub.sort_values("ie_value")
            if len(sub):
                ax.fill_between(sub["ie_value"],
                                (sub["test_accuracy_mean"] - sub["test_accuracy_std"]) * 100,
                                (sub["test_accuracy_mean"] + sub["test_accuracy_std"]) * 100,
                                alpha=0.15, color=color)
                ax.plot(sub["ie_value"], sub["test_accuracy_mean"] * 100,
                        marker=marker, markersize=3, lw=1.0, color=color,
                        label=LABEL_MAP.get(ct, ct))
        ax.set_xlabel("$N_I$")
        ax.set_ylabel("Test accuracy (%)")
        ax.set_title("Noise resilience ($N_I$ detail)")
        ax.legend(fontsize=7)

    # Panel C: MNIST N_I detail with error bands
    ax = axes[1, 0]
    _panel(ax, "C")

    if ie_data is not None:
        for ct, color, marker in [("dendritic_shunting", COLOR_SHUNTING, "o"),
                                   ("dendritic_additive", COLOR_ADDITIVE, "s")]:
            sub = ie_data[(ie_data["network_type"] == ct) &
                          (ie_data["dataset"] == "mnist")].copy()
            sub = sub.sort_values("ie_value")
            if len(sub):
                ax.fill_between(sub["ie_value"],
                                (sub["test_accuracy_mean"] - sub["test_accuracy_std"]) * 100,
                                (sub["test_accuracy_mean"] + sub["test_accuracy_std"]) * 100,
                                alpha=0.15, color=color)
                ax.plot(sub["ie_value"], sub["test_accuracy_mean"] * 100,
                        marker=marker, markersize=3, lw=1.0, color=color,
                        label=LABEL_MAP.get(ct, ct))
        ax.set_xlabel("$N_I$")
        ax.set_ylabel("Test accuracy (%)")
        ax.set_title("MNIST $N_I$ sweep (detail)")
        ax.legend(fontsize=7)

    # Panel D: Fashion-MNIST all seeds
    ax = axes[1, 1]
    _panel(ax, "D")

    fmnist_raw = _csv_path(FMNIST_RUNS_CSV)
    if fmnist_raw is not None:
        for ct, color in [("dendritic_shunting", COLOR_SHUNTING),
                           ("dendritic_additive", COLOR_ADDITIVE)]:
            for strat, marker in [("local_ca", "o"), ("standard", "s")]:
                sub = fmnist_raw[(fmnist_raw["network_type"] == ct) &
                                 (fmnist_raw["strategy"] == strat)]
                short_ct = "Shunt." if "shunting" in ct else "Add."
                short_st = "Local" if strat == "local_ca" else "BP"
                alpha = 0.9 if strat == "local_ca" else 0.45
                ax.scatter(sub["seed"], sub["test_accuracy"] * 100,
                           color=color, marker=marker, alpha=alpha, s=25,
                           label=f"{short_ct} {short_st}", edgecolors="white",
                           lw=0.3)
        ax.set_ylabel("Test accuracy (%)")
        ax.set_xlabel("Seed")
        ax.set_title("F-MNIST (all seeds)")
        ax.legend(fontsize=7, ncol=2, loc="lower right",
                  handlelength=1.0, handletextpad=0.3)

    fig.subplots_adjust(left=0.08, right=0.97, bottom=0.06, top=0.94)
    _save(fig, "fig_s2_gradient_extended")
    plt.close(fig)


# ===================================================================
# Appendix Figure S3 — Sandbox (copy existing)
# ===================================================================
def figure_s3():
    print("\n--- Figure S3: Sandbox ---")
    import shutil
    src = os.path.join(FIGURES_DIR, "fig_neurips_combined.pdf")
    dst = os.path.join(FIGURES_DIR, "fig_s3_sandbox.pdf")
    if os.path.isfile(src):
        shutil.copy2(src, dst)
        src_png = src.replace(".pdf", ".png")
        if os.path.isfile(src_png):
            shutil.copy2(src_png, dst.replace(".pdf", ".png"))
        print(f"  Copied: {src} -> {dst}")
    else:
        print(f"  WARNING: {src} not found, creating placeholder")
        fig, ax = plt.subplots(figsize=(W, 3))
        ax.text(0.5, 0.5, "Sandbox figure - see fig_neurips_combined",
                transform=ax.transAxes, ha="center", va="center", fontsize=12)
        ax.axis("off")
        _save(fig, "fig_s3_sandbox")
        plt.close(fig)


# ===================================================================
# Appendix Figure S4 — Verification & Reproducibility
# ===================================================================
def figure_s4():
    print("\n--- Figure S4: Verification & Reproducibility ---")

    verif = _csv("verification_seeds_summary.csv")
    p2b = _csv("phase2b_gap_closing.csv", bundle=True)

    fig, axes = plt.subplots(1, 3, figsize=(W * 1.9, 3.6),
                             gridspec_kw={"wspace": 0.52})

    # ---- Panel A: MNIST verification ----
    ax = axes[0]
    _panel(ax, "A")

    bars_data = []
    bars_data.append(("Seeds\n42-46", 91.39, 0.33, COLOR_SHUNTING, 1.0))
    if verif is not None:
        v = verif[(verif["core_type"] == "dendritic_shunting") & (verif["dataset_name"] == "mnist")]
        if len(v):
            bars_data.append(("Seeds\n47-49",
                              v.iloc[0]["test_acc_mean"] * 100,
                              v.iloc[0]["test_acc_std"] * 100,
                              COLOR_SHUNTING, 0.55))

    x = np.arange(len(bars_data))
    for i, b in enumerate(bars_data):
        ax.bar(i, b[1], yerr=b[2], color=b[3], alpha=b[4],
               edgecolor="white", lw=0.3, width=0.5, capsize=2, error_kw={"lw": 0.6})
    ax.set_xticks(x)
    ax.set_xticklabels([b[0] for b in bars_data], fontsize=7.5)
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title("MNIST verification")
    ax.set_ylim(85, 95)

    for i, b in enumerate(bars_data):
        ax.text(i, b[1] + b[2] + 0.3, f"{b[1]:.1f}$\\pm${b[2]:.1f}",
                ha="center", va="bottom", fontsize=7)

    # ---- Panel B: Context gating verification ----
    ax = axes[1]
    _panel(ax, "B")

    bars_data = []
    bars_data.append(("Seeds 42-46\n(+HSIC)", 80.26, 0.61, COLOR_SHUNTING, 1.0))
    if verif is not None:
        v = verif[(verif["core_type"] == "dendritic_shunting") & (verif["dataset_name"] == "context_gating")]
        if len(v):
            bars_data.append(("Seeds 47-49\n(no HSIC)",
                              v.iloc[0]["test_acc_mean"] * 100,
                              v.iloc[0]["test_acc_std"] * 100,
                              COLOR_SHUNTING, 0.55))

    x = np.arange(len(bars_data))
    for i, b in enumerate(bars_data):
        ax.bar(i, b[1], yerr=b[2], color=b[3], alpha=b[4],
               edgecolor="white", lw=0.3, width=0.5, capsize=2, error_kw={"lw": 0.6})
    ax.set_xticks(x)
    ax.set_xticklabels([b[0] for b in bars_data], fontsize=7.5)
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title("Context gating verif.")
    ax.set_ylim(60, 90)

    for i, b in enumerate(bars_data):
        ax.text(i, b[1] + b[2] + 0.5, f"{b[1]:.1f}$\\pm${b[2]:.1f}",
                ha="center", va="bottom", fontsize=7)

    # ---- Panel C: HSIC weight ablation ----
    ax = axes[2]
    _panel(ax, "C")

    if p2b is not None:
        cg = p2b[(p2b["dataset"] == "context_gating") &
                 (p2b["error_broadcast_mode"] == "per_soma")].copy()
        if len(cg):
            cg = cg.sort_values("hsic_weight")
            ax.errorbar(cg["hsic_weight"], cg["test_accuracy_mean"] * 100,
                        yerr=cg["test_accuracy_std"] * 100,
                        marker="o", markersize=4, lw=1.0, capsize=2,
                        color=COLOR_SHUNTING)
            ax.set_xlabel("HSIC weight")
            ax.set_ylabel("Test accuracy (%)")
            ax.set_title("CG: HSIC ablation")
            ax.set_xscale("symlog", linthresh=0.005)

    fig.subplots_adjust(left=0.08, right=0.97, bottom=0.15, top=0.90,
                        wspace=0.55)
    _save(fig, "fig_s4_verification")
    plt.close(fig)


# ===================================================================
# Analysis selection helpers
# ===================================================================
# Pin the analysis-summary directory date used for the submission. If you rerun
# a sweep and want the new result, update this constant (or set it to None to
# fall back to latest-matching).
ANALYSIS_DATE = "20260417"


def _latest_analysis_dir(prefix: str):
    """Find the pinned or most recent analysis subdir matching <prefix>_*.

    If ANALYSIS_DATE is set, prefer the exact `<prefix>_<ANALYSIS_DATE>` dir;
    otherwise fall back to the most recent match. This makes figure data
    provenance explicit and reproducible across reruns.
    """
    import glob
    if ANALYSIS_DATE is not None:
        pinned = os.path.join(ANALYSIS_DIR, f"{prefix}_{ANALYSIS_DATE}")
        if os.path.isdir(pinned):
            return pinned
    matches = sorted(glob.glob(os.path.join(ANALYSIS_DIR, f"{prefix}_*")))
    return matches[-1] if matches else None


def _analysis_csv(prefix: str, filename: str):
    ana_dir = _latest_analysis_dir(prefix)
    if ana_dir is None:
        return None
    path = os.path.join(ana_dir, filename)
    return path if os.path.isfile(path) else None


def _read_markdown_table(md_path: str, section_header: str) -> pd.DataFrame:
    """Parse a simple markdown table from a named section in a report note."""
    with open(md_path, encoding="utf-8") as handle:
        lines = handle.readlines()

    in_section = False
    table_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped == section_header:
            in_section = True
            continue
        if in_section and stripped.startswith("## "):
            break
        if in_section and stripped.startswith("|"):
            table_lines.append(stripped)

    if len(table_lines) < 3:
        raise ValueError(f"No markdown table found under section {section_header!r} in {md_path}")

    def parse_row(raw: str):
        return [cell.strip() for cell in raw.strip("|").split("|")]

    header = parse_row(table_lines[0])
    rows = []
    for raw in table_lines[2:]:
        cells = parse_row(raw)
        if len(cells) == len(header):
            rows.append(cells)
    return pd.DataFrame(rows, columns=header)


# ===================================================================
# Appendix Figure — Soma-On Extension
# ===================================================================
SOMA_EXTENSION_REPORT = os.path.join(
    ANALYSIS_DIR,
    "localca_finished_reruns_and_bm_comparison_20260415.md",
)


def figure_s_soma_extension():
    """Bar chart: soma-off vs soma-on across 4 paper-facing LocalCA claim families.

    Data source (by preference):
      1. Structured CSV at analysis/soma_extension_summary_<DATE>/soma_extension_summary.csv
         emitted by summarize_soma_extension() in summarize_neurips_new_sweeps.py.
      2. Fallback: the dated comparison markdown note that originally
         carried these numbers.
    """
    print("\n--- Supplementary: Soma Extension ---")

    csv_family_order = [
        "phase1_capacity", "claimA_shunting_regime",
        "claimB_morphology", "claimC_error_shaping",
    ]
    md_family_order = [
        "phase1_capacity_calibration", "phase3_claimA_shunting_regime_strong",
        "phase3_claimB_morphology_scaling", "phase3_claimC_error_shaping",
    ]
    family_labels_csv = {
        "phase1_capacity":        "Phase 1\n(Capacity)",
        "claimA_shunting_regime": "Claim A\n(Shunting\nRegime)",
        "claimB_morphology":      "Claim B\n(Morphology\nScaling)",
        "claimC_error_shaping":   "Claim C\n(Error\nShaping)",
    }
    family_labels_md = {
        "phase1_capacity_calibration":           "Phase 1\n(Capacity)",
        "phase3_claimA_shunting_regime_strong":  "Claim A\n(Shunting\nRegime)",
        "phase3_claimB_morphology_scaling":      "Claim B\n(Morphology\nScaling)",
        "phase3_claimC_error_shaping":           "Claim C\n(Error\nShaping)",
    }

    families = mean_no_soma = mean_soma = best_no_soma = best_soma = None

    # --- Preferred: structured CSV from summarize_soma_extension()
    ana_dir = _latest_analysis_dir("soma_extension_summary")
    if ana_dir is not None:
        csv_path = os.path.join(ana_dir, "soma_extension_summary.csv")
        if os.path.isfile(csv_path):
            print(f"  source: {csv_path}")
            df = pd.read_csv(csv_path)
            families = []
            mean_no_soma, mean_soma, best_no_soma, best_soma = [], [], [], []
            for fam in csv_family_order:
                sub = df[df["family"] == fam]
                off = sub[sub["soma"] == "off"]
                on  = sub[sub["soma"] == "on"]
                if len(off) and len(on):
                    families.append(family_labels_csv[fam])
                    mean_no_soma.append(float(off.iloc[0]["mean_test_accuracy"]))
                    mean_soma.append(float(on.iloc[0]["mean_test_accuracy"]))
                    best_no_soma.append(float(off.iloc[0]["best_test_accuracy"]))
                    best_soma.append(float(on.iloc[0]["best_test_accuracy"]))
            if not families:
                print("  WARNING: CSV present but no complete soma-off/soma-on pairs; falling back to markdown.")
                families = None

    # --- Fallback: original markdown comparison note
    if families is None:
        if not os.path.isfile(SOMA_EXTENSION_REPORT):
            print(f"  WARNING: neither CSV nor {SOMA_EXTENSION_REPORT} found; skipping figure.")
            return
        print(f"  source (fallback): {SOMA_EXTENSION_REPORT}")
        soma_df = _read_markdown_table(SOMA_EXTENSION_REPORT, "## 2. Soma-on extension")
        soma_df = soma_df.set_index("Family")
        families = [family_labels_md[key] for key in md_family_order]
        mean_no_soma = [float(soma_df.loc[key, "Safe non-soma mean"]) for key in md_family_order]
        mean_soma    = [float(soma_df.loc[key, "Safe soma mean"])     for key in md_family_order]
        best_no_soma = [float(soma_df.loc[key, "Safe best"])          for key in md_family_order]
        best_soma    = [float(soma_df.loc[key, "Soma best"])          for key in md_family_order]

    fig, axes = plt.subplots(1, 2, figsize=(W * 1.9, 4.0),
                             gridspec_kw={"wspace": 0.48})

    x = np.arange(len(families))
    bw = 0.38
    col_off = "#4A7CB5"   # steel blue — soma-off
    col_on  = "#1C8A57"   # deep green — soma-on

    panel_titles = [
        "(A)  Mean accuracy: soma off vs. soma on",
        "(B)  Best accuracy: soma off vs. soma on",
    ]

    for ax_idx, (ax, vals_off, vals_on, ylabel, ptitle) in enumerate(zip(
        axes,
        [mean_no_soma, best_no_soma],
        [mean_soma,    best_soma],
        ["Mean test accuracy (%)", "Best test accuracy (%)"],
        panel_titles,
    )):
        ax.bar(x - bw / 2, [v * 100 for v in vals_off], bw,
               color=col_off, alpha=0.92, edgecolor="white", lw=0.3,
               label="Soma off (baseline)")
        ax.bar(x + bw / 2, [v * 100 for v in vals_on],  bw,
               color=col_on,  alpha=0.92, edgecolor="white", lw=0.3,
               label="Soma on (extension)")

        # Annotate delta on top of soma-on bars
        for xi, (vo, vs) in enumerate(zip(vals_off, vals_on)):
            delta = (vs - vo) * 100
            bar_top = vs * 100 + 0.8
            ax.text(xi + bw / 2, bar_top, f"+{delta:.1f}" if delta >= 0 else f"{delta:.1f}",
                    ha="center", va="bottom", fontsize=6.5, color="#333333")

        ax.set_xticks(x)
        ax.set_xticklabels(families, fontsize=7.5)
        ax.set_ylabel(ylabel)
        ax.set_title(ptitle, fontsize=8.5, loc="left", pad=6)
        ax.set_ylim(0, 108)
        if ax_idx == 0:
            ax.legend(fontsize=7, handlelength=1.2, handletextpad=0.4,
                      loc="upper right", framealpha=0.9)
        style_axis(ax)

    fig.subplots_adjust(left=0.09, right=0.97, bottom=0.22, top=0.94)
    _save(fig, "fig_s_soma_extension")
    plt.close(fig)


# ===================================================================
# Appendix Figure — b,m Update Policy Comparison
# ===================================================================
def figure_s_bm_policy():
    """Grouped bar chart: learned-local vs quantile-maintained b,m on MNIST + CIFAR."""
    print("\n--- Supplementary: b,m Policy Comparison ---")

    bm_csv = _analysis_csv("bm_classification_summary", "bm_classification_grouped_summary.csv")
    bm_df = _csv_path(bm_csv)
    if bm_df is None:
        print("  WARNING: BM grouped CSV not found; skipping figure.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(W * 1.9, 3.8),
                             gridspec_kw={"wspace": 0.50})

    datasets = ["mnist", "cifar10"]
    ds_titles = {"mnist": "MNIST", "cifar10": "CIFAR-10"}

    # We want: for each dataset, show additive vs shunting, and for each core
    # show learned_local_bm vs quantile_bm (LocalCA rows), plus standard/backprop_bm ref
    col_learned = {"additive": "#2D5DA8", "shunting": "#18864B"}   # solid
    col_quantile = {"additive": "#7DAAD6", "shunting": "#6FC98D"}   # lighter version
    col_bp_ref   = "#999999"

    for ax_idx, (ax, ds) in enumerate(zip(axes, datasets)):
        _panel(ax, "AB"[ax_idx])

        sub = bm_df[bm_df["dataset"] == ds].copy()
        # Columns: dataset, training_strategy, core, morphology, bm_update_scheme,
        #          init_policy, completed_count, mean_train_accuracy, mean_valid_accuracy,
        #          mean_test_accuracy, best_config_name, best_test_accuracy

        # Aggregate across morphology and init_policy variants within (strategy, core, bm_scheme)
        agg = (
            sub.groupby(["training_strategy", "core", "bm_update_scheme"])["mean_test_accuracy"]
            .mean()
            .reset_index()
        )

        # Build bar groups: x = [add_learned, add_quantile, shunt_learned, shunt_quantile]
        # Plus a light horizontal reference line for standard backprop_bm
        group_labels = [
            "Add.\nLearned",
            "Add.\nQuantile",
            "Shunt.\nLearned",
            "Shunt.\nQuantile",
        ]
        group_vals = []
        group_cols = []

        ref_add_bp = agg[(agg["training_strategy"] == "standard") &
                         (agg["core"] == "additive") &
                         (agg["bm_update_scheme"] == "backprop_bm")]["mean_test_accuracy"]
        ref_shu_bp = agg[(agg["training_strategy"] == "standard") &
                         (agg["core"] == "shunting") &
                         (agg["bm_update_scheme"] == "backprop_bm")]["mean_test_accuracy"]

        for strategy, core, bm_key, color in [
            ("local_ca", "additive",  "learned_local_bm",  col_learned["additive"]),
            ("local_ca", "additive",  "quantile_bm",        col_quantile["additive"]),
            ("local_ca", "shunting",  "learned_local_bm",  col_learned["shunting"]),
            ("local_ca", "shunting",  "quantile_bm",        col_quantile["shunting"]),
        ]:
            row = agg[(agg["training_strategy"] == strategy) &
                      (agg["core"] == core) &
                      (agg["bm_update_scheme"] == bm_key)]
            group_vals.append(row.iloc[0]["mean_test_accuracy"] * 100 if len(row) else 0)
            group_cols.append(color)

        x = np.arange(len(group_labels))
        ax.bar(x, group_vals, 0.55,
               color=group_cols, alpha=0.92,
               edgecolor="white", lw=0.3)

        # Value annotations
        for xi, v in enumerate(group_vals):
            ax.text(xi, v + 0.3, f"{v:.1f}", ha="center", va="bottom", fontsize=6.5)

        # Reference lines for standard backprop
        if len(ref_add_bp):
            ax.axhline(ref_add_bp.iloc[0] * 100, xmin=0.0, xmax=0.5,
                       color=col_bp_ref, lw=1.1, ls="--", label="BP ceiling (add.)")
        if len(ref_shu_bp):
            ax.axhline(ref_shu_bp.iloc[0] * 100, xmin=0.5, xmax=1.0,
                       color=col_bp_ref, lw=1.1, ls=":", label="BP ceiling (shunt.)")

        ax.set_xticks(x)
        ax.set_xticklabels(group_labels, fontsize=7.5)
        ax.set_ylabel("Mean test accuracy (%)")
        ax.set_title(f"{ds_titles[ds]}: LocalCA b,m update policy",
                     fontsize=8.5, loc="left", pad=6)
        style_axis(ax)

        # y-range: include BP ceilings if they're higher
        all_vals = list(group_vals)
        if len(ref_add_bp):
            all_vals.append(ref_add_bp.iloc[0] * 100)
        if len(ref_shu_bp):
            all_vals.append(ref_shu_bp.iloc[0] * 100)
        ymin = max(0, min(all_vals) - 4)
        ymax = max(all_vals) + 4
        ax.set_ylim(ymin, ymax)

        # Compact per-axis legend (only show BP ceiling lines)
        ax.legend(fontsize=6.5, handlelength=1.2, loc="lower right",
                  framealpha=0.9, handletextpad=0.4)

    # Bottom legend for bar colors
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(color=col_learned["additive"], alpha=0.92, label="Learned local b,m (add.)"),
        Patch(color=col_quantile["additive"], alpha=0.92, label="Quantile-maintained b,m (add.)"),
        Patch(color=col_learned["shunting"], alpha=0.92, label="Learned local b,m (shunt.)"),
        Patch(color=col_quantile["shunting"], alpha=0.92, label="Quantile-maintained b,m (shunt.)"),
    ]
    fig.legend(handles=legend_handles, ncol=2, fontsize=7, loc="lower center",
               bbox_to_anchor=(0.5, -0.03), handlelength=1.1, framealpha=0.9)

    fig.subplots_adjust(left=0.09, right=0.97, bottom=0.22, top=0.94)
    _save(fig, "fig_s_bm_policy")
    plt.close(fig)


# ===================================================================
# Appendix Figure — Component Ablation
# ===================================================================
def figure_s_ablation():
    """Bar chart: per-component ablation for shunting and additive on MNIST LocalCA."""
    print("\n--- Supplementary: Component Ablation ---")

    ana_dir = _latest_analysis_dir("component_ablation_summary")
    if ana_dir is None:
        print("  WARNING: No component_ablation_summary dir found. Run summarize_neurips_new_sweeps.py first.")
        return
    csv_path = os.path.join(ana_dir, "ablation_grouped_summary.csv")
    if not os.path.isfile(csv_path):
        print(f"  WARNING: {csv_path} not found")
        return

    df = pd.read_csv(csv_path)

    # Plot order: full → no_quantile_init → no_learned_bm → no_reactivation
    #             → relu_reactivation → with_soma → bp_reference
    ordered_conditions = [
        ("full_config",          "Full\n(default)"),
        ("no_quantile_init",     "-Quantile\ninit"),
        ("no_learned_bm",        "-Learned\n(b,m)"),
        ("no_reactivation",      "-Reactiv.\n(identity)"),
        ("with_soma",            "+Soma\n(extension)"),
        ("bp_reference",         "Backprop\nref."),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(W * 1.95, 4.0),
                             gridspec_kw={"wspace": 0.42})

    cores = ["shunting", "additive"]
    core_titles = {"shunting": "Shunting (MNIST)", "additive": "Additive (MNIST)"}
    core_color = {"shunting": COLOR_SHUNTING, "additive": COLOR_ADDITIVE}

    panel_letters = ["A", "B"]
    for i, (ax, core) in enumerate(zip(axes, cores)):
        sub = df[df["core"] == core].set_index("condition")
        labels = []
        means = []
        stds = []
        bar_colors = []
        for cond_key, label in ordered_conditions:
            if cond_key not in sub.index:
                continue
            row = sub.loc[cond_key]
            labels.append(label)
            means.append(row["test_acc_mean"] * 100)
            stds.append((row.get("test_acc_std") or 0.0) * 100)
            if cond_key == "full_config":
                bar_colors.append(core_color[core])
            elif cond_key == "with_soma":
                bar_colors.append("#1C8A57")
            elif cond_key == "bp_reference":
                bar_colors.append(COLOR_POINT_MLP)
            else:
                bar_colors.append("#A3A3A3")

        x = np.arange(len(labels))
        ax.bar(x, means, 0.62, yerr=stds, capsize=2.5,
               color=bar_colors, alpha=0.92, edgecolor="white", lw=0.4,
               error_kw={"lw": 0.6})
        # Reference line at full_config value
        if "full_config" in sub.index:
            full_val = sub.loc["full_config", "test_acc_mean"] * 100
            ax.axhline(full_val, color=core_color[core], lw=0.7, ls=":", alpha=0.7)

        # Value annotations
        for xi, (m_, s_) in enumerate(zip(means, stds)):
            ax.text(xi, m_ + s_ + 0.4, f"{m_:.1f}",
                    ha="center", va="bottom", fontsize=6.5)

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=0, ha="center", fontsize=7)
        ax.set_ylabel("MNIST test accuracy (%)")
        ax.set_title(f"({panel_letters[i]})  {core_titles[core]}",
                     fontsize=8.5, loc="left", pad=6)
        if means:
            ax.set_ylim(max(0, min(means) - 3), max(means) + 3)
        style_axis(ax)

    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.22, top=0.93)
    _save(fig, "fig_s_ablation")
    plt.close(fig)


# ===================================================================
# Appendix Figure — Cue Routing + Soma extension
# ===================================================================
def figure_s_cue_routing_soma():
    """Bar chart: cue routing before vs. after soma extension.

    Soma-off baselines come from the corrected cue-routing summaries.
    Soma-on results come from the dedicated cue_routing_soma summary.
    """
    print("\n--- Supplementary: Cue Routing Soma ---")

    ana_dir = _latest_analysis_dir("cue_routing_soma_summary")
    if ana_dir is None:
        print("  WARNING: No cue_routing_soma_summary dir found. Run summarize_neurips_new_sweeps.py first.")
        return
    csv_path = os.path.join(ana_dir, "cue_routing_soma_grouped.csv")
    if not os.path.isfile(csv_path):
        print(f"  WARNING: {csv_path} not found")
        return

    df = pd.read_csv(csv_path).set_index("base_name")

    rank_csv = os.path.join(
        ANALYSIS_DIR,
        "rank_bridge_activation_corrected",
        "cue_routing_rank_structure_summary.csv",
    )
    if rank_csv is None:
        print("  WARNING: cue-routing rank summary not found; skipping figure.")
        return
    rank_df = pd.read_csv(rank_csv)
    per_soma_row = rank_df[(rank_df["broadcast_mode"] == "per_soma")]
    low_rank_row = rank_df[
        (rank_df["broadcast_mode"] == "low_rank") & (rank_df["broadcast_rank"] == 2)
    ]
    pathway_row = rank_df[(rank_df["broadcast_mode"] == "pathway_vector")]
    if per_soma_row.empty or low_rank_row.empty or pathway_row.empty:
        print("  WARNING: cue-routing rank summary missing expected rows; skipping figure.")
        return

    histo = {
        "shunting\nrank-1 LocalCA": (
            float(per_soma_row.iloc[0]["test_accuracy_mean"]),
            float(per_soma_row.iloc[0]["test_accuracy_std"]),
        ),
        "shunting\nlow-rank K=2": (
            float(low_rank_row.iloc[0]["test_accuracy_mean"]),
            float(low_rank_row.iloc[0]["test_accuracy_std"]),
        ),
        "shunting\npathway-vec.": (
            float(pathway_row.iloc[0]["test_accuracy_mean"]),
            float(pathway_row.iloc[0]["test_accuracy_std"]),
        ),
    }

    # New soma-on numbers
    soma_entries = [
        ("shunting fixed\nLocalCA + soma",
         df.loc["cue_hard_fixed_shunting_localca_soma"]
         if "cue_hard_fixed_shunting_localca_soma" in df.index else None),
        ("additive learned\nLocalCA + soma",
         df.loc["cue_hard_learned_additive_localca_soma"]
         if "cue_hard_learned_additive_localca_soma" in df.index else None),
        ("additive learned\nBP + soma",
         df.loc["cue_hard_learned_additive_standard_soma"]
         if "cue_hard_learned_additive_standard_soma" in df.index else None),
    ]

    fig, ax = plt.subplots(figsize=(W * 1.4, 4.0))

    labels = []
    means = []
    errs = []
    colors = []
    for lbl, (m, s) in histo.items():
        labels.append(lbl)
        means.append(m * 100)
        errs.append(s * 100)
        colors.append("#C2916E")  # soma-off baseline: muted orange-brown
    for lbl, row in soma_entries:
        if row is None:
            continue
        labels.append(lbl)
        means.append(row["test_acc_mean"] * 100)
        errs.append((row.get("test_acc_std") or 0.0) * 100)
        colors.append("#1C8A57")  # soma-on: deep green

    x = np.arange(len(labels))
    ax.bar(x, means, 0.6, yerr=errs, capsize=2.5, color=colors,
           alpha=0.92, edgecolor="white", lw=0.4,
           error_kw={"lw": 0.6})

    # Separator between soma-off and soma-on groups
    ax.axvline(len(histo) - 0.5, color="#777", lw=0.6, ls=":", alpha=0.6)
    ax.text(1, 104, "Prior (soma off)",
            ha="center", va="bottom", fontsize=7.5, color="#7B5C42", style="italic")
    ax.text(len(histo) + (len(soma_entries) - 1) / 2, 104, "Soma-on extension",
            ha="center", va="bottom", fontsize=7.5, color="#135C3A", style="italic")

    for xi, (m, e) in enumerate(zip(means, errs)):
        ax.text(xi, m + e + 0.7, f"{m:.1f}",
                ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7.5)
    ax.set_ylabel("Cue-routing test accuracy (%)")
    ax.set_title("Somatic inputs rescue cue-routing LocalCA",
                 fontsize=8.7, loc="left", pad=6)
    ax.set_ylim(60, 108)
    style_axis(ax)

    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.18, top=0.90)
    _save(fig, "fig_s_cue_routing_soma")
    plt.close(fig)


# ===================================================================
# Appendix Figure — CIFAR-10 soma-on extension
# ===================================================================
def figure_s_cifar10_soma_extension():
    """Two-panel appendix figure for the CIFAR-10 soma-on extension."""
    print("\n--- Supplementary: CIFAR-10 Soma Extension ---")

    on_csv = _analysis_csv("cifar10_depth4_soma_summary", "cifar10_depth4_soma_grouped.csv")
    on_df = _csv_path(on_csv)
    off_df = _csv_path(CIFAR10_LOCALCA_MECH_SUMMARY_CSV)
    bp_df = _csv_path(CIFAR10_BP_SUMMARY_CSV)
    if on_df is None or off_df is None or bp_df is None:
        print("  WARNING: CIFAR soma extension CSVs not found; skipping figure.")
        return

    def _on_value(core: str, mode: str):
        sub = on_df[
            on_df["base_name"].str.contains(core)
            & on_df["base_name"].str.contains(mode)
        ]
        if len(sub) == 0:
            return None
        row = sub.iloc[0]
        return float(row["test_acc_mean"]), float(row["test_acc_std"])

    def _off_value(core: str, mode: str):
        sub = off_df[off_df["model_type"] == f"dendritic_{core}"]
        if mode == "per_soma":
            sub = sub[sub["broadcast_mode"] == "per_soma"]
        elif mode == "path_transport":
            sub = sub[sub["broadcast_mode"] == "path_transport"]
        elif mode == "low_rank_k4":
            sub = sub[(sub["broadcast_mode"] == "low_rank") & (sub["rank_k"] == 4)]
        if len(sub) == 0:
            return None
        row = sub.iloc[0]
        return float(row["acc_test_mean"]), float(row["acc_test_std"])

    def _bp_value(core: str):
        sub = bp_df[
            (bp_df["family"] == "standard")
            & (bp_df["model_type"] == f"dendritic_{core}")
        ]
        if len(sub) == 0:
            return None
        row = sub.iloc[0]
        return float(row["mean_test_accuracy"]), float(row["std_test_accuracy"])

    fig, axes = plt.subplots(1, 2, figsize=(W * 1.95, 4.0),
                             gridspec_kw={"wspace": 0.42})

    # Panel A: absolute soma-on ladder
    ax = axes[0]
    _panel(ax, "A")

    modes = [
        ("per_soma", "Per-soma"),
        ("low_rank_k4", "Low-rank\n$K=4$"),
        ("path_transport", "Path\ntransport"),
    ]
    x = np.arange(len(modes))
    bw = 0.32
    add_vals = [_on_value("additive", mode) for mode, _ in modes]
    shunt_vals = [_on_value("shunting", mode) for mode, _ in modes]

    ax.bar(
        x - bw / 2,
        [v[0] * 100 for v in add_vals],
        bw,
        yerr=[v[1] * 100 for v in add_vals],
        color=COLOR_ADDITIVE,
        edgecolor="white",
        lw=0.3,
        capsize=1.8,
        error_kw={"lw": 0.6},
        label="Additive + soma",
    )
    ax.bar(
        x + bw / 2,
        [v[0] * 100 for v in shunt_vals],
        bw,
        yerr=[v[1] * 100 for v in shunt_vals],
        color=COLOR_SHUNTING,
        edgecolor="white",
        lw=0.3,
        capsize=1.8,
        error_kw={"lw": 0.6},
        label="Shunting + soma",
    )

    bp_add = _bp_value("additive")
    bp_shunt = _bp_value("shunting")
    if bp_add is not None:
        ax.axhline(bp_add[0] * 100, color=COLOR_ADDITIVE, lw=1.0, ls="--", alpha=0.75)
    if bp_shunt is not None:
        ax.axhline(bp_shunt[0] * 100, color=COLOR_SHUNTING, lw=1.0, ls="--", alpha=0.75)

    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in modes], fontsize=7.5)
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title("CIFAR-10 soma-on ladder", fontsize=8.8, loc="left", pad=6)
    ax.set_ylim(20, 53)
    ax.legend(fontsize=6.8, loc="upper center", bbox_to_anchor=(0.5, -0.10),
              ncol=2, framealpha=0.92, handlelength=1.1, handletextpad=0.4)
    style_axis(ax, grid="y")

    for xpos, val in zip(x - bw / 2, add_vals):
        ax.text(xpos, val[0] * 100 + val[1] * 100 + 0.7,
                f"{val[0] * 100:.1f}", ha="center", va="bottom", fontsize=6.2)
    for xpos, val in zip(x + bw / 2, shunt_vals):
        ax.text(xpos, val[0] * 100 + val[1] * 100 + 0.7,
                f"{val[0] * 100:.1f}", ha="center", va="bottom", fontsize=6.2)

    # Panel B: matched soma-off -> soma-on deltas
    ax = axes[1]
    _panel(ax, "B")

    paired = [
        ("Add.\nPer-soma", COLOR_ADDITIVE,
         _on_value("additive", "per_soma"), _off_value("additive", "per_soma")),
        ("Add.\nPath trans.", COLOR_ADDITIVE,
         _on_value("additive", "path_transport"), _off_value("additive", "path_transport")),
        ("Shunt.\nPer-soma", COLOR_SHUNTING,
         _on_value("shunting", "per_soma"), _off_value("shunting", "per_soma")),
        ("Shunt.\nLow-rank\n$K=4$", COLOR_SHUNTING,
         _on_value("shunting", "low_rank_k4"), _off_value("shunting", "low_rank_k4")),
        ("Shunt.\nPath trans.", COLOR_SHUNTING,
         _on_value("shunting", "path_transport"), _off_value("shunting", "path_transport")),
    ]
    labels, deltas, colors = [], [], []
    for label, color, on_val, off_val in paired:
        if on_val is None or off_val is None:
            continue
        labels.append(label)
        deltas.append((on_val[0] - off_val[0]) * 100)
        colors.append(color)

    xpos = np.arange(len(labels))
    bars = ax.bar(
        xpos,
        deltas,
        color=colors,
        edgecolor="white",
        lw=0.3,
        width=0.62,
    )
    ax.axhline(0, color="black", lw=0.5, ls="--")
    ax.set_xticks(xpos)
    ax.set_xticklabels(labels, fontsize=7.2)
    ax.set_ylabel(r"$\Delta$ test accuracy (pp)")
    ax.set_title("Matched shift from soma-off", fontsize=8.8, loc="left", pad=6)
    style_axis(ax, grid="y")
    ax.set_ylim(min(deltas) - 2.0, max(deltas) + 2.0)
    for rect, delta in zip(bars, deltas):
        yo = 0.35 if delta >= 0 else -0.45
        va = "bottom" if delta >= 0 else "top"
        ax.text(
            rect.get_x() + rect.get_width() / 2,
            delta + yo,
            f"{delta:+.1f}",
            ha="center",
            va=va,
            fontsize=6.3,
        )

    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.27, top=0.92, wspace=0.42)
    _save(fig, "fig_s_cifar10_soma_extension")
    plt.close(fig)


# ===================================================================
# Main Figure — Mechanism Summary scatter
# ===================================================================
def figure_mechanism_summary():
    """Two-panel scatter showing the causal chain:
        (A) path-gain CV -> per-soma cosine alignment
        (B) per-soma cosine alignment -> test accuracy
    across all conditions (additive/shunting x 5 IE values).

    This is the one-glance mechanism summary requested by the review.
    """
    print("\n--- Main: Mechanism Summary ---")

    if not os.path.isfile(THEORY_IE_SUMMARY_CSV):
        print(f"  WARNING: {THEORY_IE_SUMMARY_CSV} not found; skipping.")
        return
    df = pd.read_csv(THEORY_IE_SUMMARY_CSV)

    # Keep only the MNIST rows used for the theory-diag panel in Fig. 5, and the
    # noise-resilience rows, to show the chain holds across both datasets.
    ds_colors = {"mnist": "#4A7CB5", "noise_resilience": "#E67E22"}
    ds_titles = {"mnist": "MNIST", "noise_resilience": "Noise resil."}
    core_marker = {"dendritic_additive": "o", "dendritic_shunting": "s"}
    core_label  = {"dendritic_additive": "Additive", "dendritic_shunting": "Shunting"}

    fig, axes = plt.subplots(1, 2, figsize=(W * 1.9, 3.4),
                             gridspec_kw={"wspace": 0.32})

    # --- Panel A: path-gain CV vs per-soma cosine alignment
    axA = axes[0]
    for ds, ds_df in df.groupby("dataset"):
        if ds not in ds_colors:
            continue
        for core, sub in ds_df.groupby("network_type"):
            if core not in core_marker:
                continue
            axA.errorbar(
                sub["path_gain_cv_mean_mean"],
                sub["per_soma_weighted_cosine_mean"],
                xerr=sub["path_gain_cv_mean_std"],
                yerr=sub["per_soma_weighted_cosine_std"],
                fmt=core_marker[core],
                color=ds_colors[ds],
                markersize=6, alpha=0.85,
                capsize=2.5, lw=0.8, elinewidth=0.8,
                markeredgecolor="white", markeredgewidth=0.6,
                label=f"{ds_titles[ds]} {core_label[core]}",
            )
    axA.set_xlabel("Path-gain CV")
    axA.set_ylabel("Per-soma cosine alignment")
    axA.set_title("(A)  Concentrated path gains $\\to$ higher alignment",
                  fontsize=9, loc="left", pad=6)
    axA.legend(fontsize=6.5, handlelength=1.1, handletextpad=0.4,
               loc="upper right", framealpha=0.9, ncol=2)
    style_axis(axA)

    # --- Panel B: per-soma cosine alignment vs test accuracy
    axB = axes[1]
    for ds, ds_df in df.groupby("dataset"):
        if ds not in ds_colors:
            continue
        for core, sub in ds_df.groupby("network_type"):
            if core not in core_marker:
                continue
            axB.errorbar(
                sub["per_soma_weighted_cosine_mean"],
                sub["test_accuracy_mean"] * 100,
                xerr=sub["per_soma_weighted_cosine_std"],
                yerr=sub["test_accuracy_std"] * 100,
                fmt=core_marker[core],
                color=ds_colors[ds],
                markersize=6, alpha=0.85,
                capsize=2.5, lw=0.8, elinewidth=0.8,
                markeredgecolor="white", markeredgewidth=0.6,
                label=f"{ds_titles[ds]} {core_label[core]}",
            )
    axB.set_xlabel("Per-soma cosine alignment")
    axB.set_ylabel("Test accuracy (%)")
    axB.set_title("(B)  Higher alignment $\\to$ better learning",
                  fontsize=9, loc="left", pad=6)
    axB.legend(fontsize=6.5, handlelength=1.1, handletextpad=0.4,
               loc="lower right", framealpha=0.9, ncol=2)
    style_axis(axB)

    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.18, top=0.90)
    _save(fig, "fig_mechanism_summary")
    plt.close(fig)


# ===================================================================
# Appendix Figure — Weight distributions under soma-on
# ===================================================================
def figure_s_weight_dist_soma_comparison():
    """Bar chart comparing soma-off vs. soma-on excitatory-weight CV and mean
    across the 4 (core, strategy) cells on MNIST at depth [3,3].

    soma-off source: analysis/weight_dist_by_depth_summary_<DATE>/weight_dist_grouped_summary.csv
                     filtered to rows with depth == 'd33'
    soma-on  source: analysis/weight_dist_soma_summary_<DATE>/weight_dist_soma_grouped_summary.csv
    """
    print("\n--- Supplementary: Weight Distributions under Soma-On ---")

    off_dir = _latest_analysis_dir("weight_dist_by_depth_summary")
    on_dir = _latest_analysis_dir("weight_dist_soma_summary")
    if off_dir is None or on_dir is None:
        print("  WARNING: could not find both soma-off and soma-on weight-dist summaries; skipping.")
        return

    off_csv = os.path.join(off_dir, "weight_dist_grouped_summary.csv")
    on_csv = os.path.join(on_dir, "weight_dist_soma_grouped_summary.csv")
    if not (os.path.isfile(off_csv) and os.path.isfile(on_csv)):
        print(f"  WARNING: missing CSV ({off_csv}, {on_csv})")
        return

    off_df = pd.read_csv(off_csv)
    off_df = off_df[off_df["depth"] == "d33"].copy()
    on_df = pd.read_csv(on_csv).copy()

    cells = [
        ("shunting", "bp"), ("shunting", "localca"),
        ("additive", "bp"), ("additive", "localca"),
    ]
    cell_labels = {
        ("shunting", "bp"):      "Shunting\nBP",
        ("shunting", "localca"): "Shunting\nLocalCA",
        ("additive", "bp"):      "Additive\nBP",
        ("additive", "localca"): "Additive\nLocalCA",
    }

    def _cv(row) -> float:
        m = float(row["excitatory_weights_mean"])
        s = float(row["excitatory_weights_std"])
        return s / m if m > 0 else float("nan")

    def _acc(row) -> float:
        return 100.0 * float(row["test_accuracy"])

    off_cv, on_cv = [], []
    off_acc, on_acc = [], []
    for core, strat in cells:
        off_r = off_df[(off_df["core"] == core) & (off_df["strategy"] == strat)]
        on_r = on_df[(on_df["core"] == core) & (on_df["strategy"] == strat)]
        off_cv.append(_cv(off_r.iloc[0]) if len(off_r) else float("nan"))
        on_cv.append(_cv(on_r.iloc[0]) if len(on_r) else float("nan"))
        off_acc.append(_acc(off_r.iloc[0]) if len(off_r) else float("nan"))
        on_acc.append(_acc(on_r.iloc[0]) if len(on_r) else float("nan"))

    fig, axes = plt.subplots(1, 2, figsize=(W * 1.9, 3.6),
                             gridspec_kw={"wspace": 0.42})

    x = np.arange(len(cells))
    bw = 0.38
    col_off = "#4A7CB5"
    col_on = "#1C8A57"

    for ax_idx, (ax, yvals_off, yvals_on, ylabel, ptitle) in enumerate(zip(
        axes,
        [off_cv, off_acc],
        [on_cv, on_acc],
        ["Excitatory weight CV (std/mean)", "Test accuracy (%)"],
        ["CV under soma off vs. soma on",
         "Test accuracy under soma off vs. soma on"],
    )):
        ax.bar(x - bw / 2, yvals_off, bw, color=col_off, alpha=0.92,
               edgecolor="white", lw=0.3, label="Soma off")
        ax.bar(x + bw / 2, yvals_on, bw, color=col_on, alpha=0.92,
               edgecolor="white", lw=0.3, label="Soma on")
        for xi, (vo, vn) in enumerate(zip(yvals_off, yvals_on)):
            if np.isfinite(vn):
                ax.text(xi + bw / 2, vn + 0.01 * max(abs(vo or 0), abs(vn)),
                        f"{vn:.2f}" if ax_idx == 0 else f"{vn:.1f}",
                        ha="center", va="bottom", fontsize=6.5)
        ax.set_xticks(x)
        ax.set_xticklabels([cell_labels[c] for c in cells], fontsize=7.5)
        ax.set_ylabel(ylabel)
        ax.set_title(ptitle, fontsize=8.8, loc="left", pad=6)
        if ax_idx == 0:
            ax.legend(fontsize=7, handlelength=1.2, handletextpad=0.4,
                      loc="upper left", framealpha=0.88)
        style_axis(ax)

    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.16, top=0.93)
    _save(fig, "fig_s_weight_dist_soma_comparison")
    plt.close(fig)


# ===================================================================
# Appendix Figure - Weight Distributions x Depth x Strategy
# ===================================================================
def figure_s_weight_distributions():
    """Weight distribution comparison: BP vs LocalCA x additive vs shunting x 3 depths.

    Two panels:
      A) Excitatory weight mean ± std across conditions, grouped by depth.
      B) Coefficient of variation (std/mean) of excitatory weights vs depth.
    """
    print("\n--- Supplementary: Weight Distributions vs Depth ---")

    ana_dir = _latest_analysis_dir("weight_dist_by_depth_summary")
    if ana_dir is None:
        print("  WARNING: No weight_dist_by_depth_summary dir found. Run summarize_neurips_new_sweeps.py first.")
        return
    csv_path = os.path.join(ana_dir, "weight_dist_grouped_summary.csv")
    if not os.path.isfile(csv_path):
        print(f"  WARNING: {csv_path} not found")
        return

    df = pd.read_csv(csv_path)

    depths = ["d22", "d33", "d333"]
    depth_labels = {"d22": "[2,2]", "d33": "[3,3]", "d333": "[3,3,3]"}
    cores = ["shunting", "additive"]
    strategies = ["bp", "localca"]

    color_map = {
        ("shunting", "bp"):     COLOR_SHUNTING,
        ("shunting", "localca"): "#7BCFA0",
        ("additive", "bp"):     COLOR_ADDITIVE,
        ("additive", "localca"): "#88B0DC",
    }

    fig, axes = plt.subplots(1, 3, figsize=(W * 2.0, 3.6),
                             gridspec_kw={"wspace": 0.42})

    # Panel A: excitatory weight mean by depth, grouped bars
    ax = axes[0]
    n_groups = len(cores) * len(strategies)
    bw = 0.8 / n_groups
    x = np.arange(len(depths))
    for j, (core, strat) in enumerate([(c, s) for c in cores for s in strategies]):
        means = []
        stds = []
        for depth in depths:
            row = df[(df["core"] == core) & (df["strategy"] == strat) &
                     (df["depth"] == depth)]
            if len(row) == 0:
                means.append(np.nan)
                stds.append(0)
            else:
                means.append(row.iloc[0].get("excitatory_weights_mean", np.nan))
                stds.append(row.iloc[0].get("excitatory_weights_std", 0))
        off = (j - (n_groups - 1) / 2) * bw
        core_l = "Sh." if core == "shunting" else "Ad."
        strat_l = "BP" if strat == "bp" else "Loc."
        ax.bar(x + off, means, bw * 0.92,
               yerr=stds, capsize=1.5, error_kw={"lw": 0.4},
               color=color_map[(core, strat)], edgecolor="white", lw=0.3,
               label=f"{core_l} {strat_l}")
    ax.set_xticks(x)
    ax.set_xticklabels([depth_labels[d] for d in depths])
    ax.set_xlabel("Branch factors (depth)")
    ax.set_ylabel("Excitatory weight mean")
    ax.set_title("(A)  Excitatory weight mean vs depth", fontsize=8.5, loc="left", pad=6)
    ax.legend(fontsize=6.5, ncol=2, handlelength=1.0, handletextpad=0.3,
              loc="upper right", framealpha=0.9)
    style_axis(ax)

    # Panel B: coefficient of variation (std/mean) of excitatory weights
    ax = axes[1]
    for j, (core, strat) in enumerate([(c, s) for c in cores for s in strategies]):
        cvs = []
        for depth in depths:
            row = df[(df["core"] == core) & (df["strategy"] == strat) &
                     (df["depth"] == depth)]
            if len(row) == 0:
                cvs.append(np.nan)
            else:
                m = row.iloc[0].get("excitatory_weights_mean", np.nan)
                s = row.iloc[0].get("excitatory_weights_std", np.nan)
                cvs.append(s / m if (m and not np.isnan(m) and m > 0) else np.nan)
        off = (j - (n_groups - 1) / 2) * bw
        core_l = "Sh." if core == "shunting" else "Ad."
        strat_l = "BP" if strat == "bp" else "Loc."
        ax.bar(x + off, cvs, bw * 0.92,
               color=color_map[(core, strat)], edgecolor="white", lw=0.3,
               label=f"{core_l} {strat_l}")
    ax.set_xticks(x)
    ax.set_xticklabels([depth_labels[d] for d in depths])
    ax.set_xlabel("Branch factors (depth)")
    ax.set_ylabel("CV (std/mean)")
    ax.set_title("(B)  Excitatory weight CV vs depth", fontsize=8.5, loc="left", pad=6)
    ax.legend(fontsize=6.5, ncol=2, handlelength=1.0, handletextpad=0.3,
              loc="upper right", framealpha=0.9)
    style_axis(ax)

    # Panel C: test accuracy vs depth (sanity check that depth scaling holds)
    ax = axes[2]
    for j, (core, strat) in enumerate([(c, s) for c in cores for s in strategies]):
        accs = []
        for depth in depths:
            row = df[(df["core"] == core) & (df["strategy"] == strat) &
                     (df["depth"] == depth)]
            if len(row) == 0:
                accs.append(np.nan)
            else:
                accs.append(row.iloc[0].get("test_accuracy", np.nan) * 100)
        off = (j - (n_groups - 1) / 2) * bw
        core_l = "Sh." if core == "shunting" else "Ad."
        strat_l = "BP" if strat == "bp" else "Loc."
        ax.bar(x + off, accs, bw * 0.92,
               color=color_map[(core, strat)], edgecolor="white", lw=0.3,
               label=f"{core_l} {strat_l}")
    ax.set_xticks(x)
    ax.set_xticklabels([depth_labels[d] for d in depths])
    ax.set_xlabel("Branch factors (depth)")
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title("(C)  Test accuracy vs depth", fontsize=8.5, loc="left", pad=6)
    ax.legend(fontsize=6.5, ncol=2, handlelength=1.0, handletextpad=0.3,
              loc="lower left", framealpha=0.9)
    style_axis(ax)

    fig.subplots_adjust(left=0.06, right=0.98, bottom=0.18, top=0.92)
    _save(fig, "fig_s_weight_dist_by_depth")
    plt.close(fig)


# ===================================================================
# Main
# ===================================================================
def main():
    apply_neurips_style()
    print(f"Data dir: {DATA_DIR}")
    print(f"Figures dir: {FIGURES_DIR}")
    print(f"Bundle: {BUNDLE}")

    print("Skipping Figure 1 here; use generate_figure1_schematic.py for the submission figure.")
    figure2()
    figure_rule_feedback_design()
    figure3()
    figure4()
    figure_s1()
    figure_s2()
    figure_s4()
    # figure_mechanism_summary() is now panel D of fig5_mechanistic_evidence,
    # produced by generate_revision_figures.py (via generate_theory_diagnostics_figures).
    figure_s_soma_extension()
    figure_s_bm_policy()
    figure_s_ablation()
    figure_s_cue_routing_soma()
    figure_s_cifar10_soma_extension()
    figure_s_weight_distributions()
    figure_s_weight_dist_soma_comparison()

    print("\n" + "="*50)
    print("All figures generated successfully.")


if __name__ == "__main__":
    main()
