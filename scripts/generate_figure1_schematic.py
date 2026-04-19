#!/usr/bin/env python3
"""Generate Figure 1: compartmental dendritic neuron, shunting integration,
local-rule family, broadcast modes, and representative MNIST dynamics.

Outputs:
  - figures/fig1_model_and_credit.{pdf,png}

Layout (13 in wide × 7 in tall):

   ┌─────────────────────────────────────────────┐ ┌───────────────────┐
   │              Panel A                        │ │    Panel B        │
   │  Compartmental dendritic neuron schematic   │ │  Shunting rule    │
   │  (distal leaves → proximal → soma)          │ │   + equation      │
   └─────────────────────────────────────────────┘ └───────────────────┘
   ┌──────────────────┐ ┌──────────────────┐ ┌────────────────────────┐
   │    Panel C       │ │    Panel D       │ │       Panel E          │
   │  3F / 4F / 5F    │ │  Broadcast modes │ │  MNIST learning curves │
   └──────────────────┘ └──────────────────┘ └────────────────────────┘
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from neurips_style import (  # noqa: E402
    COLORS, apply_neurips_style, clean_schematic_axis, panel_label,
)

apply_neurips_style()

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.patches as mpatches  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402
from matplotlib.patheffects import withStroke  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# ── Shortcuts to palette ────────────────────────────────────────────────
EXC   = COLORS["exc"]
INH   = COLORS["inh"]
DEND  = COLORS["dend"]
SOMA  = COLORS["soma"]
INK   = COLORS["ink"]
MUTE  = COLORS["mute"]

# ── Paths ────────────────────────────────────────────────────────────────
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "figures"
SWEEP_ROOT = Path(
    "/n/holylfs06/LABS/kempner_project_b/Lab/dendritic/HS/LOCAL_LEARNING/sweep_runs"
)
FALLBACK_RUNS = {
    "bp_shunting": SWEEP_ROOT
    / "sweep_neurips_phase1_capacity_calibration_20260211171057/results/config_83",
    "local_shunting": SWEEP_ROOT
    / "sweep_neurips_phase2b_gap_closing_pilot_20260218161300/results/config_74",
    "local_additive": SWEEP_ROOT
    / "sweep_neurips_additive_dynamics_fairness_audit_20260223231355/results/config_1",
}
FIG1_FIXED100_PREFIX = "sweep_fig1_mnist_fixed100_"
FIG1_RUN_NAMES = {
    "bp_shunting": "fig1_mnist_shunting_bp_fixed100_s43",
    "rule_3f":     "fig1_mnist_shunting_3f_fixed100_s43",
    "rule_4f":     "fig1_mnist_shunting_4f_fixed100_s43",
    "rule_5f":     "fig1_mnist_shunting_localca_fixed100_s43",
}

CURVE_STYLES = {
    "bp_shunting":    dict(color="#185A33", ls="-",  label="Shunt. BP (oracle)"),
    "rule_5f":        dict(color=COLORS["rule_5f"], ls="-",  label="Shunt. 5F"),
    "rule_4f":        dict(color=COLORS["rule_4f"], ls="--", label="Shunt. 4F"),
    "rule_3f":        dict(color=COLORS["rule_3f"], ls=":",  label="Shunt. 3F"),
    "local_additive": dict(color=COLORS["additive"], ls="-.", label="Add. 5F"),
}


# ── Data loaders ─────────────────────────────────────────────────────────
def resolve_runs() -> dict[str, Path]:
    """Prefer the latest fixed-100-epoch Figure 1 sweep when available."""
    candidate_sweeps = sorted(
        (p for p in SWEEP_ROOT.glob(f"{FIG1_FIXED100_PREFIX}*") if p.is_dir()),
        reverse=True,
    )
    for sweep_dir in candidate_sweeps:
        resolved: dict[str, Path] = {}
        for config_dir in sorted((sweep_dir / "results").glob("config_*")):
            config_json = config_dir / "config.json"
            if not config_json.exists():
                continue
            with open(config_json) as fh:
                payload = json.load(fh)
            run_name = payload.get("outputs", {}).get("run_name")
            for key, expected in FIG1_RUN_NAMES.items():
                if run_name == expected:
                    resolved[key] = config_dir
        if set(resolved) == set(FIG1_RUN_NAMES):
            return resolved
    return FALLBACK_RUNS


def load_epoch_history(run_dir: Path) -> pd.DataFrame:
    perf_dir = run_dir / "performance" / "epochs"
    rows = []
    for epoch_file in sorted(
        perf_dir.glob("epoch*.json"),
        key=lambda p: int(p.stem.replace("epoch", "")),
    ):
        with open(epoch_file) as fh:
            payload = json.load(fh)
        rows.append({
            "epoch": int(epoch_file.stem.replace("epoch", "")),
            "train_accuracy": float(payload["accuracy"]["train"]),
            "test_accuracy":  float(payload["accuracy"]["test"]),
        })
    return pd.DataFrame(rows)


# ── Drawing primitives ──────────────────────────────────────────────────
def draw_branch_rect(ax, x, y, w, h, label=None, fc=DEND, alpha=0.85,
                     fontsize=7.2, ec=None, lw=0.6):
    if ec is None:
        ec = COLORS["edge"]
    box = FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.015,rounding_size=0.035",
        fc=fc, ec=ec, linewidth=lw, alpha=alpha, zorder=3,
    )
    ax.add_patch(box)
    if label is not None:
        ax.text(x, y, label, ha="center", va="center",
                fontsize=fontsize, fontweight="bold", color="white",
                zorder=6, path_effects=[withStroke(linewidth=0.8, foreground="#0A3A23")])


def draw_synapse_E(ax, x, y, size=0.055):
    c = plt.Circle((x, y), size, fc=EXC, ec="white",
                   linewidth=0.5, zorder=5)
    ax.add_patch(c)


def draw_synapse_I(ax, x, y, size=0.11):
    tri = mpatches.RegularPolygon(
        (x, y), numVertices=3, radius=size, orientation=np.pi,
        fc=INH, ec="white", linewidth=0.5, zorder=5,
    )
    ax.add_patch(tri)


def draw_wire(ax, x1, y1, x2, y2, color=DEND, lw=1.3, alpha=0.9, zorder=2):
    ax.plot([x1, x2], [y1, y2], color=color, lw=lw, alpha=alpha,
            zorder=zorder, solid_capstyle="round")


def draw_arrow(ax, x1, y1, x2, y2, color=INK, lw=1.1, style="-|>",
               mutation_scale=10, zorder=4, alpha=1.0):
    arrow = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle=style, mutation_scale=mutation_scale,
        color=color, lw=lw, alpha=alpha, zorder=zorder,
        shrinkA=0, shrinkB=0,
    )
    ax.add_patch(arrow)


# ── Panel A: compartmental dendritic neuron ─────────────────────────────
def panel_A(ax):
    clean_schematic_axis(ax)
    ax.set_xlim(-0.3, 7.8)
    ax.set_ylim(-0.35, 3.95)
    ax.set_title("Compartmental dendritic neuron: shunting E/I integration",
                 fontsize=10.0, pad=4, loc="left", x=0.00)

    # ── Geometry: 3 proximal branches, each with 3 distal leaves ──
    soma_xy = (6.55, 1.8)
    proximal_y = [3.05, 1.80, 0.55]
    proximal_x = 4.8
    distal_dx = 1.7  # horizontal offset of distal column from proximal
    distal_subspacing = 0.28

    # Draw the soma (large rounded circle)
    soma = plt.Circle(soma_xy, 0.36, fc=SOMA, ec=COLORS["edge"],
                      linewidth=1.0, alpha=0.9, zorder=5)
    ax.add_patch(soma)
    ax.text(soma_xy[0], soma_xy[1] + 0.02, "soma",
            ha="center", va="center", fontsize=7.8, fontweight="bold",
            color="white", zorder=7)
    ax.text(soma_xy[0], soma_xy[1] - 0.18, r"$V_{\mathrm{out}}$",
            ha="center", va="center", fontsize=8.0, color="white", zorder=7)

    # Proximal branches (3)
    for py in proximal_y:
        draw_branch_rect(ax, proximal_x, py, 0.80, 0.44, label=r"$V_{b_2}$",
                         fc=DEND, alpha=0.95, fontsize=8.5)
        # Wire from proximal branch → soma
        draw_wire(ax, proximal_x + 0.40, py,
                  soma_xy[0] - 0.36, soma_xy[1] + (py - soma_xy[1]) * 0.22,
                  color=DEND, lw=2.0, alpha=0.8, zorder=2)

    # Distal branches (3 per proximal)
    for idx_p, py in enumerate(proximal_y):
        distal_x = proximal_x - distal_dx
        distal_ys = [py + 0.50, py, py - 0.50]  # wider spacing
        for didx, dy in enumerate(distal_ys):
            draw_branch_rect(ax, distal_x, dy, 0.68, 0.36,
                             label=r"$V_{b_1}$", fc=DEND, alpha=0.85,
                             fontsize=7.8)
            # Wire distal → proximal
            draw_wire(ax, distal_x + 0.34, dy,
                      proximal_x - 0.40, py + (dy - py) * 0.25,
                      color=DEND, lw=1.3, alpha=0.7, zorder=2)

            # Synapses on each distal branch
            # 2 excitatory above/below-left, 1 inhibitory above
            sx_left = distal_x - 0.45
            for edy in [-0.11, 0.11]:
                draw_synapse_E(ax, sx_left, dy + edy)
                # thin wire from synapse to branch
                draw_wire(ax, sx_left + 0.04, dy + edy,
                          distal_x - 0.34, dy + edy * 0.5,
                          color=EXC, lw=0.5, alpha=0.6, zorder=1)
            draw_synapse_I(ax, distal_x, dy + 0.28)
            draw_wire(ax, distal_x, dy + 0.28 - 0.09,
                      distal_x, dy + 0.18,
                      color=INH, lw=0.5, alpha=0.6, zorder=1)

    # Also add I and E synapses onto proximal branches (biological realism)
    for py in proximal_y:
        draw_synapse_I(ax, proximal_x, py + 0.32)
        draw_wire(ax, proximal_x, py + 0.32 - 0.09,
                  proximal_x, py + 0.22, color=INH, lw=0.5, alpha=0.6, zorder=1)
        for ex in [-0.22, 0.22]:
            draw_synapse_E(ax, proximal_x + ex, py - 0.30)
            draw_wire(ax, proximal_x + ex, py - 0.30 + 0.055,
                      proximal_x + ex * 0.3, py - 0.22,
                      color=EXC, lw=0.5, alpha=0.6, zorder=1)

    # Input-labels legend (bottom-right corner of panel to avoid title overlap)
    leg_x0, leg_y0 = 6.0, 3.55
    draw_synapse_E(ax, leg_x0, leg_y0)
    ax.text(leg_x0 + 0.16, leg_y0, "excitatory ($E_j^E > 0$)",
            fontsize=7.2, color=EXC, va="center", fontweight="bold")
    draw_synapse_I(ax, leg_x0, leg_y0 - 0.28)
    ax.text(leg_x0 + 0.16, leg_y0 - 0.28, "inhibitory ($E_j^I = 0$)",
            fontsize=7.2, color=INH, va="center", fontweight="bold")

    # Output arrow from soma
    draw_arrow(ax, soma_xy[0] + 0.36, soma_xy[1], 7.55, soma_xy[1],
               color=INK, lw=1.3, mutation_scale=12)
    ax.text(7.60, soma_xy[1], "output",
            fontsize=8.2, va="center", ha="left", fontweight="bold")

    # ── Depth annotation (tree levels) ──
    for (x, label) in [(3.1, "distal\n(level 1)"),
                       (4.8, "proximal\n(level 2)"),
                       (6.55, "soma\n(level 3)")]:
        ax.text(x, -0.23, label, ha="center", va="top",
                fontsize=7.0, color=MUTE, style="italic")

    # ── Branch zoom + equation callout (bottom-right area) ──
    # drawn on the same axes but placed below the tree — removed to keep panel clean
    # (equation moved to Panel B)


# ── Panel B: shunting integration equation + mechanism ───────────────────
def panel_B(ax):
    clean_schematic_axis(ax)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("Shunting voltage integration",
                 fontsize=10.0, pad=4, loc="left", x=0.00)

    # Explanatory top text
    ax.text(0.03, 0.95,
            "Each branch aggregates its synapses\nvia a conductance equation:",
            fontsize=8.2, color=INK, va="top")

    # Core equation (rendered in mathtext — mathtext supports \frac, \sum, \mathrm)
    eq_box = FancyBboxPatch(
        (0.02, 0.50), 0.96, 0.30,
        boxstyle="round,pad=0.025,rounding_size=0.03",
        fc="#FAFBFC", ec=COLORS["edge"], lw=0.7, zorder=2,
    )
    ax.add_patch(eq_box)
    # Numerator / denominator on two lines for clarity
    ax.text(
        0.5, 0.725,
        r"$V_n \;=\; \frac{\sum_{j} g_j\, x_j\, E_j}{g_n^{\mathrm{tot}}}$",
        ha="center", va="center", fontsize=12.5, color=INK,
    )
    ax.text(
        0.5, 0.57,
        r"$g_n^{\mathrm{tot}} \;=\; g^{\mathrm{leak}} \;+\; "
        r"\sum_{j \in E} g_j\, x_j \;+\; \sum_{j \in I} g_j\, x_j$",
        ha="center", va="center", fontsize=9.8, color=INK,
    )

    # Annotation: shunting denominator
    ax.annotate(
        "shunting: inhibition raises $g_n^{\\mathrm{tot}}$ and lowers gain",
        xy=(0.5, 0.52), xytext=(0.5, 0.39),
        ha="center", va="top",
        fontsize=6.9, color=INH, fontweight="bold",
        arrowprops=dict(arrowstyle="-|>", color=INH, lw=1.0, shrinkA=0, shrinkB=2),
    )

    # Bottom comparison: shunting vs additive path gain histogram sketch
    # Draw two stylised histograms in a mini inset-like strip.
    hist_y0 = 0.04
    hist_h = 0.22
    hist_w = 0.42
    # Shunting (narrow) on the left
    _draw_gain_hist(ax, x0=0.03, y0=hist_y0, w=hist_w, h=hist_h,
                    color=COLORS["shunting"], spread=0.14,
                    title="shunting\nnarrow gain dist.")
    # Additive (broad) on the right
    _draw_gain_hist(ax, x0=0.55, y0=hist_y0, w=hist_w, h=hist_h,
                    color=COLORS["additive"], spread=0.42,
                    title="additive\nbroad gain dist.")


def _draw_gain_hist(ax, x0, y0, w, h, color, spread, title):
    """Draw a small stylised log-normal histogram as a patch group."""
    # Axis base
    ax.plot([x0, x0 + w], [y0, y0], color=COLORS["edge"], lw=0.6)
    # Samples around a log-scale mean
    rng = np.random.default_rng(int(spread * 100))
    samples = np.clip(rng.normal(0.5, spread, size=900), 0.02, 0.98)
    # Bin into 20 bins within [0, 1]
    bins = np.linspace(0, 1, 21)
    counts, _ = np.histogram(samples, bins=bins)
    counts = counts / counts.max()
    bar_w = (w / 20) * 0.88
    for i, c in enumerate(counts):
        bx = x0 + (i + 0.06) * (w / 20)
        ax.add_patch(plt.Rectangle(
            (bx, y0), bar_w, c * h * 0.95,
            fc=color, ec=color, linewidth=0.3, alpha=0.85, zorder=3,
        ))
    ax.text(x0 + w / 2, y0 + h + 0.01, title,
            ha="center", va="bottom",
            fontsize=6.5, color=color, fontweight="bold", linespacing=1.0)
    ax.text(x0 + w / 2, y0 - 0.03, "path gain",
            ha="center", va="top", fontsize=6.1, color=MUTE, style="italic")


# ── Panel C: local learning rule hierarchy ───────────────────────────────
def panel_C(ax):
    clean_schematic_axis(ax)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("Local rule family (3F → 4F → 5F)",
                 fontsize=10.0, pad=4, loc="left", x=0.00)

    rules = [
        dict(name="3F", color=COLORS["rule_3f"], y=0.78,
             equation=r"$\Delta w_j \;\propto\; x_j\,(E_j - V_n)\, e$",
             legend="presynaptic × post-local × broadcast"),
        dict(name="4F", color=COLORS["rule_4f"], y=0.50,
             equation=r"$\Delta w_j \;\propto\; \mathrm{3F}\;\cdot\;\rho_n$",
             legend=r"$\rho_n$: morphology modulation (voltage variance)"),
        dict(name="5F", color=COLORS["rule_5f"], y=0.22,
             equation=r"$\Delta w_j \;\propto\; \mathrm{4F}\;\cdot\;\phi_n$",
             legend=r"$\phi_n$: confidence / correctness factor"),
    ]
    for r in rules:
        # Strip with colored tag on left
        tag_w = 0.12
        FancyBboxPatch_ = FancyBboxPatch(
            (0.02, r["y"] - 0.10), tag_w, 0.20,
            boxstyle="round,pad=0.01,rounding_size=0.02",
            fc=r["color"], ec=r["color"], lw=0.6, zorder=3, alpha=0.95,
        )
        ax.add_patch(FancyBboxPatch_)
        ax.text(0.02 + tag_w / 2, r["y"], r["name"],
                ha="center", va="center", fontsize=11, fontweight="bold",
                color="white", zorder=4)
        # Equation + legend
        ax.text(0.17, r["y"] + 0.04, r["equation"],
                fontsize=9.0, color=INK, va="center", ha="left")
        ax.text(0.17, r["y"] - 0.08, r["legend"],
                fontsize=7.2, color=MUTE, va="center", ha="left",
                style="italic")

    # Arrows between levels
    for y_top, y_bot in [(0.68, 0.60), (0.40, 0.32)]:
        draw_arrow(ax, 0.08, y_top, 0.08, y_bot, color=MUTE,
                   lw=0.8, mutation_scale=8, style="-|>")

    # Small footer note
    ax.text(0.02, 0.04,
            "all factors are locally computable at synapse $j$\n"
            "only $e$ is broadcast from soma to dendrite",
            fontsize=7.0, color=MUTE, va="bottom", ha="left")


# ── Panel D: broadcast modes ─────────────────────────────────────────────
def panel_D(ax):
    clean_schematic_axis(ax)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("Broadcast channels for $e$",
                 fontsize=10.0, pad=4, loc="left", x=0.00)

    def mini_tree(x, y, scale=0.11, n_leaves=3):
        """Draw a small 1-level dendritic tree centered at (x, y)."""
        soma = (x + 1.6 * scale, y)
        hub = (x + 0.7 * scale, y)
        leaves = [(x - 0.4 * scale, y + (i - (n_leaves - 1) / 2) * scale * 1.5)
                  for i in range(n_leaves)]
        # Draw dendrite wires
        for leaf in leaves:
            ax.plot([leaf[0], hub[0]], [leaf[1], hub[1]],
                    color=DEND, lw=1.0, alpha=0.75, zorder=2)
        ax.plot([hub[0], soma[0]], [hub[1], soma[1]],
                color=DEND, lw=1.3, alpha=0.75, zorder=2)
        # Nodes
        for leaf in leaves:
            ax.add_patch(plt.Circle(leaf, 0.012, fc=DEND, ec="white",
                                    linewidth=0.3, alpha=0.9, zorder=3))
        ax.add_patch(plt.Circle(hub, 0.015, fc=DEND, ec="white",
                                linewidth=0.3, alpha=0.9, zorder=3))
        ax.add_patch(plt.Circle(soma, 0.022, fc=SOMA, ec="white",
                                linewidth=0.3, alpha=0.9, zorder=3))
        return soma, hub, leaves

    modes = [
        dict(label="scalar", sub="1 shared broadcast",
             y=0.82, color=COLORS["scalar"], kind="scalar"),
        dict(label="per-soma", sub="one $e_n$ per soma",
             y=0.59, color=COLORS["per_soma"], kind="per_soma"),
        dict(label="low-rank", sub="$K$ mixed channels",
             y=0.36, color=COLORS["low_rank"], kind="low_rank"),
        dict(label="pathway", sub="structured per-branch",
             y=0.13, color=COLORS["pathway"], kind="pathway"),
    ]
    for m in modes:
        # Text label on far left
        ax.text(0.04, m["y"] + 0.04, m["label"],
                fontsize=8.5, fontweight="bold", color=m["color"],
                ha="left", va="center")
        ax.text(0.04, m["y"] - 0.04, m["sub"],
                fontsize=6.8, color=MUTE, ha="left", va="center",
                style="italic")
        # Draw two mini-trees side by side (two somas)
        tree_x0, tree_x1 = 0.36, 0.62
        soma0, hub0, leaves0 = mini_tree(tree_x0, m["y"], scale=0.055)
        soma1, hub1, leaves1 = mini_tree(tree_x1, m["y"], scale=0.055)

        if m["kind"] == "scalar":
            # Single source → both somas
            src = (0.92, m["y"])
            ax.add_patch(plt.Circle(src, 0.015, fc=m["color"], ec="white",
                                    linewidth=0.4, alpha=0.95, zorder=5))
            ax.text(src[0], src[1] + 0.045, "$e$",
                    ha="center", va="bottom",
                    fontsize=8.5, color=m["color"], fontweight="bold")
            for soma_pt in [soma0, soma1]:
                draw_arrow(ax, src[0], src[1], soma_pt[0] + 0.02, soma_pt[1],
                           color=m["color"], lw=0.9, mutation_scale=7, alpha=0.85)
        elif m["kind"] == "per_soma":
            # Two sources, one per soma
            for si, soma_pt in enumerate([soma0, soma1]):
                sx = 0.88 + 0.04 * si
                src = (sx, m["y"] + 0.04 * (1 if si == 0 else -1))
                ax.add_patch(plt.Circle(src, 0.012, fc=m["color"], ec="white",
                                        linewidth=0.4, alpha=0.95, zorder=5))
                draw_arrow(ax, src[0], src[1], soma_pt[0] + 0.02, soma_pt[1],
                           color=m["color"], lw=0.9, mutation_scale=7, alpha=0.85)
            ax.text(0.92, m["y"] + 0.09, r"$e_{n}$",
                    ha="center", fontsize=8.0, color=m["color"], fontweight="bold")
        elif m["kind"] == "low_rank":
            # 2 channels mixed across both somas
            chs = [(0.90, m["y"] + 0.035), (0.90, m["y"] - 0.035)]
            for ch in chs:
                ax.add_patch(plt.Circle(ch, 0.012, fc=m["color"], ec="white",
                                        linewidth=0.4, alpha=0.95, zorder=5))
                for soma_pt in [soma0, soma1]:
                    draw_arrow(ax, ch[0], ch[1], soma_pt[0] + 0.02, soma_pt[1],
                               color=m["color"], lw=0.8, mutation_scale=6, alpha=0.7)
            ax.text(0.94, m["y"], "K=2", fontsize=7.0, color=m["color"],
                    fontweight="bold", va="center")
        elif m["kind"] == "pathway":
            # Each leaf / branch gets its own channel
            chs = [(0.88, m["y"] + 0.06), (0.92, m["y"]), (0.88, m["y"] - 0.06)]
            colors = ["#7C5AA6", "#C15A8A", "#B13138"]
            for ch, col in zip(chs, colors):
                ax.add_patch(plt.Circle(ch, 0.012, fc=col, ec="white",
                                        linewidth=0.4, alpha=0.95, zorder=5))
            # Connect distinct channels to distinct leaves
            for ch, col, leaf in zip(chs, colors, leaves0):
                draw_arrow(ax, ch[0], ch[1], leaf[0], leaf[1],
                           color=col, lw=0.9, mutation_scale=6, alpha=0.8)
            ax.text(0.96, m["y"], r"$e^{(p)}$", fontsize=7.5,
                    color=m["color"], fontweight="bold", va="center")

    ax.text(0.02, 0.02,
            "rank(broadcast): scalar 1 → per-soma N → low-rank K → pathway M",
            fontsize=6.8, color=MUTE, style="italic", va="bottom")


# ── Panel E: MNIST learning dynamics ─────────────────────────────────────
def panel_E(ax):
    runs = resolve_runs()
    plotted = []
    for key, style in CURVE_STYLES.items():
        run_dir = runs.get(key) if key in runs else FALLBACK_RUNS.get("local_additive")
        if run_dir is None or not run_dir.exists():
            continue
        try:
            history = load_epoch_history(run_dir)
        except Exception:
            continue
        if history.empty:
            continue
        ax.plot(
            history["epoch"],
            history["test_accuracy"] * 100,
            color=style["color"],
            ls=style["ls"],
            lw=1.55,
            label=style["label"],
            alpha=0.95,
        )
        plotted.append((style["label"], float(history["test_accuracy"].iloc[-1] * 100)))

    ax.axhline(10, color=MUTE, ls=":", lw=0.7, alpha=0.6, zorder=0)
    ax.text(98.5, 12.0, "chance", color=MUTE, fontsize=6.8, ha="right", va="bottom")
    ax.set_xlim(0, 100)
    ax.set_ylim(5, 100)
    ax.set_xlabel("epoch")
    ax.set_ylabel("MNIST test accuracy (%)")
    ax.grid(axis="y", linewidth=0.5, alpha=0.25, color=COLORS["grid"])
    ax.set_title("MNIST learning dynamics ($[3,3]$ shunting)",
                 fontsize=10.0, pad=4, loc="left", x=0.00)

    # Legend below-right, anchored outside the plotted curves' high-accuracy zone
    ax.legend(
        loc="lower right", fontsize=6.8, ncol=1,
        handlelength=1.8, handletextpad=0.5,
        labelspacing=0.25, borderaxespad=0.6,
        frameon=True, framealpha=0.92, facecolor="white",
        edgecolor="#DDDDDD",
    )


# ── Main ────────────────────────────────────────────────────────────────
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(13.0, 7.2))
    gs = fig.add_gridspec(
        2, 3,
        height_ratios=[1.08, 1.0],
        width_ratios=[1.0, 1.0, 1.0],
        hspace=0.32, wspace=0.22,
        left=0.04, right=0.985, top=0.93, bottom=0.07,
    )

    # Panel A spans columns 0-1 on row 0 (wider schematic)
    ax_A = fig.add_subplot(gs[0, :2])
    # Panel B takes column 2 on row 0 (shunting equation)
    ax_B = fig.add_subplot(gs[0, 2])
    # Row 1: three panels
    ax_C = fig.add_subplot(gs[1, 0])
    ax_D = fig.add_subplot(gs[1, 1])
    ax_E = fig.add_subplot(gs[1, 2])

    panel_A(ax_A)
    panel_B(ax_B)
    panel_C(ax_C)
    panel_D(ax_D)
    panel_E(ax_E)

    # Panel labels — outside upper-left of each
    for ax, lbl, x_off in [
        (ax_A, "A", -0.04),
        (ax_B, "B", -0.09),
        (ax_C, "C", -0.05),
        (ax_D, "D", -0.05),
        (ax_E, "E", -0.17),
    ]:
        panel_label(ax, lbl, x=x_off, y=1.10, fontsize=13)

    out_path = OUTPUT_DIR / "fig1_model_and_credit"
    fig.savefig(out_path.with_suffix(".pdf"))
    fig.savefig(out_path.with_suffix(".png"), dpi=300)
    print(f"Saved: {out_path}.{{pdf,png}}")
    plt.close(fig)


if __name__ == "__main__":
    main()
