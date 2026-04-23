#!/usr/bin/env python3
"""Generate Figure 1: model, credit assignment, and MNIST dynamics.

Outputs:
  - figures/fig1_model_and_credit.{pdf,png}

Design philosophy:
  - One clear idea per panel. No overlapping text or arrows.
  - Panel A: the dendritic neuron (anatomy + shunting).
  - Panel B: credit assignment contrast (backprop vs LocalCA) side-by-side.
  - Panel C: rule family (3F / 4F / 5F).
  - Panel D: broadcast modes.
  - Panel E: MNIST learning dynamics.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from neurips_style import (  # noqa: E402
    COLORS,
    apply_neurips_style,
    clean_schematic_axis,
    panel_label,
    style_axis,
)

apply_neurips_style()

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.patches as mpatches  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch  # noqa: E402
from matplotlib.patheffects import withStroke  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


# ── Palette shortcuts ────────────────────────────────────────────────────
EXC  = COLORS["exc"]
INH  = COLORS["inh"]
DEND = COLORS["dend"]
SOMA = COLORS["soma"]
INK  = COLORS["ink"]
MUTE = COLORS["mute"]
EDGE = COLORS["edge"]
EXACT = "#A03E36"        # backprop / exact credit
APPROX = "#D97D51"       # local-broadcast credit

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
    "bp_shunting":    dict(color="#185A33",              ls="-",  label="Shunt. BP (oracle)"),
    "rule_5f":        dict(color=COLORS["rule_5f"],       ls="-",  label="Shunt. 5F"),
    "rule_4f":        dict(color=COLORS["rule_4f"],       ls="--", label="Shunt. 4F"),
    "rule_3f":        dict(color=COLORS["rule_3f"],       ls=":",  label="Shunt. 3F"),
    "local_additive": dict(color=COLORS["additive"],      ls="-.", label="Add. 5F"),
}


# ── Data loaders ─────────────────────────────────────────────────────────
def resolve_runs() -> dict[str, Path]:
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
    for ep in sorted(perf_dir.glob("epoch*.json"),
                     key=lambda p: int(p.stem.replace("epoch", ""))):
        with open(ep) as fh:
            payload = json.load(fh)
        rows.append({
            "epoch":          int(ep.stem.replace("epoch", "")),
            "test_accuracy":  float(payload["accuracy"]["test"]),
        })
    return pd.DataFrame(rows)


# ── Drawing primitives ──────────────────────────────────────────────────
def draw_arrow(ax, x1, y1, x2, y2, color=INK, lw=1.1, style="-|>",
               mutation_scale=10, zorder=5, alpha=1.0, linestyle="-",
               connectionstyle="arc3"):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle=style, mutation_scale=mutation_scale,
        color=color, lw=lw, alpha=alpha, zorder=zorder,
        shrinkA=0, shrinkB=0, linestyle=linestyle,
        connectionstyle=connectionstyle,
    ))


def rounded_box(ax, cx, cy, w, h, fc="white", ec=EDGE, lw=0.8, radius=0.025,
                alpha=1.0, zorder=2):
    patch = FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle=f"round,pad=0.008,rounding_size={radius}",
        fc=fc, ec=ec, linewidth=lw, alpha=alpha, zorder=zorder,
    )
    ax.add_patch(patch)
    return patch


def draw_exc(ax, x, y, r=0.010):
    ax.add_patch(Circle((x, y), r, fc=EXC, ec="white", linewidth=0.4, zorder=7))


def draw_inh(ax, x, y, r=0.011):
    ax.add_patch(mpatches.RegularPolygon(
        (x, y), numVertices=3, radius=r * 1.45, orientation=np.pi,
        fc=INH, ec="white", linewidth=0.4, zorder=7,
    ))


def soma_circle(ax, xy, r=0.045, label=None, color=SOMA, label_color="white"):
    ax.add_patch(Circle(xy, r, fc=color, ec=EDGE, linewidth=0.9, zorder=5))
    if label is not None:
        ax.text(xy[0], xy[1], label, ha="center", va="center",
                fontsize=8.2, fontweight="bold", color=label_color, zorder=6)


def draw_wire(ax, x1, y1, x2, y2, color=DEND, lw=1.3, alpha=0.82, zorder=1):
    ax.plot([x1, x2], [y1, y2], color=color, lw=lw, alpha=alpha, zorder=zorder,
            solid_capstyle="round")


# ── Tree drawing helper (used in Panels A and B) ─────────────────────────
def draw_tree(ax, *, x_leaf, x_branch, x_soma, branch_ys=None, leaf_offsets=None,
              leaf_box_w=0.045, leaf_box_h=0.028, branch_box_w=0.060,
              branch_box_h=0.04, draw_synapses=True, label_only_center=True,
              soma_r=0.038):
    """Draw a [3,3] dendritic tree (3 proximal × 3 distal each).

    Returns dict with coordinates for later use by caller.
    """
    if branch_ys is None:
        branch_ys = np.array([0.78, 0.50, 0.22])
    if leaf_offsets is None:
        leaf_offsets = np.array([+0.10, 0.0, -0.10])

    leaf_ys = []
    for by in branch_ys:
        for off in leaf_offsets:
            leaf_ys.append(by + off)
    leaf_ys = np.array(leaf_ys)

    # Wires: leaf → branch
    for bi, by in enumerate(branch_ys):
        for off in leaf_offsets:
            ly = by + off
            draw_wire(ax, x_leaf + leaf_box_w / 2 + 0.003, ly,
                      x_branch - branch_box_w / 2 - 0.003, by + off * 0.22,
                      color=DEND, lw=1.05, alpha=0.72)

    # Wires: branch → soma
    for by in branch_ys:
        draw_wire(ax, x_branch + branch_box_w / 2 + 0.003, by,
                  x_soma - soma_r - 0.003, 0.5 + (by - 0.5) * 0.18,
                  color=DEND, lw=1.7, alpha=0.82)

    # Leaves (distal compartments)
    for ly in leaf_ys:
        rounded_box(ax, x_leaf, ly, leaf_box_w, leaf_box_h,
                    fc=DEND, ec=EDGE, alpha=0.92, lw=0.5)

    # Branches (proximal compartments)
    for by in branch_ys:
        rounded_box(ax, x_branch, by, branch_box_w, branch_box_h,
                    fc=DEND, ec=EDGE, alpha=1.0, lw=0.6)

    # Soma
    soma_circle(ax, (x_soma, 0.5), r=soma_r, label=None)

    # Synapses on distal leaves
    if draw_synapses:
        for ly in leaf_ys:
            draw_exc(ax, x_leaf - leaf_box_w / 2 - 0.012, ly + 0.006, r=0.0075)
            draw_exc(ax, x_leaf - leaf_box_w / 2 - 0.012, ly - 0.006, r=0.0075)
            draw_inh(ax, x_leaf,                          ly + leaf_box_h / 2 + 0.012, r=0.009)

    # Compartment labels (only center one, to avoid clutter)
    if label_only_center:
        ax.text(x_leaf,   leaf_ys[4], r"$V_n$",    ha="center", va="center",
                fontsize=7.4, color="white", fontweight="bold", zorder=9)
        ax.text(x_branch, branch_ys[1], r"$V_{p(n)}$", ha="center", va="center",
                fontsize=7.4, color="white", fontweight="bold", zorder=9)
        ax.text(x_soma, 0.5, r"$V_{\mathrm{out}}$", ha="center", va="center",
                fontsize=7.4, color="white", fontweight="bold", zorder=9)

    return dict(leaf_x=x_leaf, branch_x=x_branch, soma_x=x_soma,
                branch_ys=branch_ys, leaf_ys=leaf_ys, soma_r=soma_r,
                leaf_box_w=leaf_box_w, leaf_box_h=leaf_box_h,
                branch_box_w=branch_box_w, branch_box_h=branch_box_h)


# ── Panel A: dendritic neuron (anatomy + shunting) ──────────────────────
def panel_A(ax):
    clean_schematic_axis(ax)
    ax.set_aspect("auto")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("Dendritic neuron with shunting E/I integration",
                 fontsize=10.0, pad=6, loc="left", x=0.02)

    # Draw the tree centered, with generous margins
    tree = draw_tree(
        ax,
        x_leaf=0.32,
        x_branch=0.56,
        x_soma=0.76,
        branch_ys=np.array([0.74, 0.50, 0.26]),
        leaf_offsets=np.array([+0.09, 0.0, -0.09]),
        leaf_box_w=0.050, leaf_box_h=0.032,
        branch_box_w=0.065, branch_box_h=0.045,
        soma_r=0.042,
    )

    # Legend (top-left, well separated from tree)
    legend_x, legend_y = 0.02, 0.92
    ax.text(legend_x, legend_y, "external input", fontsize=7.6,
            color=INK, fontweight="bold", ha="left", va="center")
    # Excitatory marker
    draw_exc(ax, legend_x + 0.012, legend_y - 0.08, r=0.009)
    ax.text(legend_x + 0.032, legend_y - 0.08, r"excitatory ($E_j^E > 0$)",
            fontsize=7.3, color=EXC, va="center", fontweight="bold")
    # Inhibitory marker
    draw_inh(ax, legend_x + 0.012, legend_y - 0.15, r=0.010)
    ax.text(legend_x + 0.032, legend_y - 0.15, r"inhibitory ($E_j^I = 0$)",
            fontsize=7.3, color=INH, va="center", fontweight="bold")

    # Anatomy labels below the tree, well clear of any elements
    ax.text(tree["leaf_x"],   0.06, "distal",   ha="center", va="center",
            fontsize=7.4, color=MUTE, style="italic")
    ax.text(tree["branch_x"], 0.06, "proximal", ha="center", va="center",
            fontsize=7.4, color=MUTE, style="italic")
    ax.text(tree["soma_x"],   0.06, "soma",     ha="center", va="center",
            fontsize=7.4, color=MUTE, style="italic")
    ax.text(0.54, 0.015, r"tree $[3,3]$: 3 proximal $\times$ 3 distal branches",
            ha="center", va="center", fontsize=7.0, color=MUTE, style="italic")

    # Output arrow from soma
    draw_arrow(ax, tree["soma_x"] + tree["soma_r"], 0.5, 0.94, 0.5,
               color=INK, lw=1.3, mutation_scale=11)
    ax.text(0.95, 0.5, "output", fontsize=8.0, color=INK,
            va="center", ha="left", fontweight="bold")

    # Right-side callout: shunting equation in a clean boxed area
    eq_cx, eq_cy = 0.86, 0.84
    rounded_box(ax, eq_cx, eq_cy, 0.26, 0.12,
                fc="#FAFBFC", ec="#D7DCE2", lw=0.8, radius=0.03)
    ax.text(eq_cx, eq_cy + 0.018,
            r"$V_n = \frac{\sum_j g_j\,x_j\,E_j}{g_n^{\mathrm{tot}}}$",
            ha="center", va="center", fontsize=10.0, color=INK)
    ax.text(eq_cx, eq_cy - 0.04,
            "shunting: I in denominator",
            ha="center", va="center", fontsize=6.8, color=INH,
            fontweight="bold", style="italic")


# ── Panel B: credit-assignment contrast (BP vs LocalCA) ─────────────────
def panel_B(ax):
    clean_schematic_axis(ax)
    ax.set_aspect("auto")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("Credit assignment: backprop vs. LocalCA",
                 fontsize=10.0, pad=6, loc="left", x=0.02)

    # Two mini-trees side by side, far enough apart to avoid overlap.
    # We use a smaller tree to preserve whitespace.

    # ── Left subpanel: backprop (EXACT) ──
    tree_L = draw_tree(
        ax,
        x_leaf=0.08, x_branch=0.20, x_soma=0.32,
        branch_ys=np.array([0.70, 0.50, 0.30]),
        leaf_offsets=np.array([+0.065, 0.0, -0.065]),
        leaf_box_w=0.032, leaf_box_h=0.020,
        branch_box_w=0.040, branch_box_h=0.028,
        soma_r=0.028,
        draw_synapses=False,
        label_only_center=False,
    )

    # Header for left subpanel
    ax.text(0.20, 0.90, "Backprop (exact)", ha="center", va="center",
            fontsize=8.6, color=EXACT, fontweight="bold")

    # Per-compartment gradient arrows: each arrow comes from ABOVE the tree and
    # drops vertically onto its corresponding proximal branch. This avoids the
    # horizontal "arrow through the soma" artifact.
    for by in tree_L["branch_ys"]:
        # start a bit to the left of the branch, above the panel top
        top_y = by + 0.10
        draw_arrow(
            ax,
            tree_L["branch_x"], top_y,
            tree_L["branch_x"], by + tree_L["branch_box_h"] / 2 + 0.003,
            color=EXACT, lw=1.1, mutation_scale=9, linestyle="--",
            connectionstyle="arc3", alpha=0.9,
        )

    # label for the set of arrows
    ax.text(0.20, 0.85,
            r"$\partial L / \partial V_n$ per compartment",
            ha="center", va="center", fontsize=7.2, color=EXACT, fontweight="bold")

    # compartment-visible list (left side, below tree)
    ax.text(0.20, 0.12,
            "compartment-specific\nerror signals",
            ha="center", va="center", fontsize=6.8, color=EXACT,
            fontweight="bold", style="italic")

    # ── Right subpanel: LocalCA (broadcast) ──
    tree_R = draw_tree(
        ax,
        x_leaf=0.60, x_branch=0.72, x_soma=0.84,
        branch_ys=np.array([0.70, 0.50, 0.30]),
        leaf_offsets=np.array([+0.065, 0.0, -0.065]),
        leaf_box_w=0.032, leaf_box_h=0.020,
        branch_box_w=0.040, branch_box_h=0.028,
        soma_r=0.028,
        draw_synapses=False,
        label_only_center=False,
    )

    # Header
    ax.text(0.72, 0.90, "LocalCA (approximate)", ha="center", va="center",
            fontsize=8.6, color=APPROX, fontweight="bold")

    # Single broadcast source near soma → ONE arrow that hits all branches (scalar)
    src_R = (0.95, 0.50)
    ax.add_patch(Circle(src_R, 0.016, fc=APPROX, ec="white", linewidth=0.4, zorder=7))
    ax.text(src_R[0], src_R[1] + 0.05, r"$e_n$", ha="center", va="bottom",
            fontsize=8.5, color=APPROX, fontweight="bold")
    # Hub near the soma of tree_R
    hub = (0.88, 0.50)
    draw_arrow(ax, src_R[0] - 0.014, src_R[1], hub[0] + 0.006, hub[1],
               color=APPROX, lw=1.2, mutation_scale=9, alpha=0.9)
    for by in tree_R["branch_ys"]:
        draw_arrow(ax, hub[0] - 0.004, hub[1],
                   tree_R["branch_x"] + tree_R["branch_box_w"] / 2 + 0.005, by,
                   color=APPROX, lw=1.0, mutation_scale=8, alpha=0.85,
                   connectionstyle=f"arc3,rad={0.2 if by > 0.5 else (-0.2 if by < 0.5 else 0)}")

    # label
    ax.text(0.72, 0.12,
            "scalar broadcast\n$+$ local state $\\{x_j, V_n, E_j\\}$",
            ha="center", va="center", fontsize=6.8, color=APPROX,
            fontweight="bold", style="italic")

    # ── Center "vs." separator and key insight line at bottom ──
    ax.plot([0.485, 0.485], [0.18, 0.82], color="#D0D0D0", lw=0.6, ls=":")

    # Bottom summary line (whole panel width)
    ax.text(0.5, 0.015,
            "Shunting denominator concentrates path gains $\\Rightarrow$ scalar $e_n$ is enough",
            ha="center", va="center", fontsize=7.3, color=INK, fontweight="bold")


# ── Panel C: rule family ────────────────────────────────────────────────
def panel_C(ax):
    clean_schematic_axis(ax)
    ax.set_aspect("auto")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("Local rule family", fontsize=9.5, pad=6, loc="left", x=0.02)

    cards = [
        dict(name="3F", color=COLORS["rule_3f"], y=0.80,
             badge="exact", badge_fc="#E4F0EA", badge_ec=COLORS["shunting"],
             eq=r"$\Delta g_j \propto x_j\,R_n^{\mathrm{tot}}\,(E_j-V_n)\,e_n$",
             note="theorem-facing 3-factor"),
        dict(name="4F", color=COLORS["rule_4f"], y=0.50,
             badge="heuristic", badge_fc="#FBEFE8", badge_ec=COLORS["rule_4f"],
             eq=r"$\Delta g_j \propto \mathrm{3F} \cdot \rho_n$",
             note=r"adds $\rho_n$: morphology / variance"),
        dict(name="5F", color=COLORS["rule_5f"], y=0.20,
             badge="heuristic", badge_fc="#ECEEF7", badge_ec=COLORS["rule_5f"],
             eq=r"$\Delta g_j \propto \mathrm{4F} \cdot \phi_n$",
             note=r"adds $\phi_n$: confidence"),
    ]
    for c in cards:
        # Card
        rounded_box(ax, 0.52, c["y"], 0.92, 0.22,
                    fc="white", ec="#D7DCE2", lw=0.6, radius=0.025)
        # Colored name tag
        rounded_box(ax, 0.12, c["y"], 0.12, 0.12,
                    fc=c["color"], ec=c["color"], lw=0, radius=0.02)
        ax.text(0.12, c["y"], c["name"], ha="center", va="center",
                fontsize=12.5, color="white", fontweight="bold")
        # Badge (tier)
        rounded_box(ax, 0.89, c["y"] + 0.065, 0.12, 0.05,
                    fc=c["badge_fc"], ec=c["badge_ec"], lw=0.7, radius=0.016)
        ax.text(0.89, c["y"] + 0.065, c["badge"], ha="center", va="center",
                fontsize=6.2, color=c["badge_ec"], fontweight="bold")
        # Equation & note
        ax.text(0.22, c["y"] + 0.025, c["eq"],
                ha="left", va="center", fontsize=8.8, color=INK)
        ax.text(0.22, c["y"] - 0.05, c["note"],
                ha="left", va="center", fontsize=6.9, color=MUTE, style="italic")

    # Arrows between cards
    for y_top, y_bot in [(0.67, 0.62), (0.37, 0.32)]:
        draw_arrow(ax, 0.12, y_top, 0.12, y_bot, color=MUTE, lw=0.8,
                   mutation_scale=8, style="-|>")


# ── Panel D: broadcast modes ────────────────────────────────────────────
def panel_D(ax):
    clean_schematic_axis(ax)
    ax.set_aspect("auto")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("Broadcast channels for $e_n$", fontsize=9.5, pad=6, loc="left", x=0.02)

    rows = [
        dict(label="scalar",    sub="1 shared field",        color=COLORS["scalar"],   y=0.82, kind="scalar"),
        dict(label="per-soma",  sub="one $e_n$ per soma",    color=COLORS["per_soma"], y=0.59, kind="per_soma"),
        dict(label="low-rank",  sub="$K$ mixed channels",    color=COLORS["low_rank"], y=0.36, kind="low_rank"),
        dict(label="pathway",   sub="structured per-branch", color=COLORS["pathway"],  y=0.13, kind="pathway"),
    ]
    # Column positions (well-separated)
    SOURCE_X = 0.12
    CHAN_X   = 0.45
    TARGET_X = 0.78

    for r in rows:
        y = r["y"]
        # Label group on far left
        ax.text(0.03, y + 0.04, r["label"], ha="left", va="center",
                fontsize=8.8, color=r["color"], fontweight="bold")
        ax.text(0.03, y - 0.04, r["sub"], ha="left", va="center",
                fontsize=6.6, color=MUTE, style="italic")

        # Three target "compartments" (squares)
        targets = [(TARGET_X, y + 0.055), (TARGET_X, y), (TARGET_X, y - 0.055)]
        for tx, ty in targets:
            rounded_box(ax, tx, ty, 0.055, 0.033,
                        fc="#F4F8F5", ec=DEND, lw=0.45, radius=0.012)

        if r["kind"] == "scalar":
            # 1 source → 1 channel → all targets
            ax.add_patch(Circle((SOURCE_X + 0.05, y), 0.014,
                                fc=r["color"], ec="white", linewidth=0.4, zorder=7))
            for tx, ty in targets:
                draw_arrow(ax, SOURCE_X + 0.065, y, tx - 0.033, ty,
                           color=r["color"], lw=0.9, mutation_scale=7, alpha=0.85)

        elif r["kind"] == "per_soma":
            # one source per target (here 3 targets = 3 sources)
            for (tx, ty) in targets:
                sx = SOURCE_X + 0.05
                ax.add_patch(Circle((sx, ty), 0.010,
                                    fc=r["color"], ec="white", linewidth=0.35, zorder=7))
                draw_arrow(ax, sx + 0.011, ty, tx - 0.033, ty,
                           color=r["color"], lw=0.85, mutation_scale=6, alpha=0.85)

        elif r["kind"] == "low_rank":
            # 2 channels, all-to-all
            channels = [(CHAN_X, y + 0.035), (CHAN_X, y - 0.035)]
            for cx, cy in channels:
                ax.add_patch(Circle((cx, cy), 0.012,
                                    fc=r["color"], ec="white", linewidth=0.35, zorder=7))
                for tx, ty in targets:
                    draw_arrow(ax, cx + 0.013, cy, tx - 0.033, ty,
                               color=r["color"], lw=0.75, mutation_scale=6, alpha=0.7)
            # single source → each channel
            ax.add_patch(Circle((SOURCE_X + 0.05, y), 0.011,
                                fc=r["color"], ec="white", linewidth=0.35, zorder=7))
            for cx, cy in channels:
                draw_arrow(ax, SOURCE_X + 0.061, y, cx - 0.013, cy,
                           color=r["color"], lw=0.7, mutation_scale=6, alpha=0.85)
            ax.text(CHAN_X + 0.02, y, "K=2", ha="left", va="center",
                    fontsize=6.5, color=r["color"], fontweight="bold")

        else:  # pathway
            # 3 channels, each → exactly one target
            pathway_colors = ["#7C5AA6", "#C15A8A", "#D08C2F"]
            for (tx, ty), col in zip(targets, pathway_colors):
                cx = CHAN_X
                cy = ty
                ax.add_patch(Circle((cx, cy), 0.011,
                                    fc=col, ec="white", linewidth=0.35, zorder=7))
                draw_arrow(ax, cx + 0.012, cy, tx - 0.033, ty,
                           color=col, lw=0.85, mutation_scale=6, alpha=0.88)
                # single "origin" to each channel
                ax.add_patch(Circle((SOURCE_X + 0.05, ty), 0.008,
                                    fc=col, ec="white", linewidth=0.3, zorder=7))
                draw_arrow(ax, SOURCE_X + 0.058, ty, cx - 0.012, cy,
                           color=col, lw=0.7, mutation_scale=5, alpha=0.8)


# ── Panel E: MNIST learning dynamics ────────────────────────────────────
def panel_E(ax):
    runs = resolve_runs()
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
            color=style["color"], ls=style["ls"], lw=1.6,
            label=style["label"], alpha=0.96,
        )
    ax.axhline(10, color=MUTE, ls=":", lw=0.7, alpha=0.55, zorder=0)
    ax.text(98.5, 11.8, "chance", color=MUTE, fontsize=6.7, ha="right", va="bottom")
    ax.set_xlim(0, 100)
    ax.set_ylim(5, 100)
    ax.set_xlabel("epoch")
    ax.set_ylabel("MNIST test accuracy (%)")
    ax.set_title("MNIST learning dynamics", fontsize=9.5, pad=6, loc="left", x=0.02)
    style_axis(ax, grid="y")
    ax.legend(loc="lower right", fontsize=6.9, ncol=1,
              handlelength=1.7, handletextpad=0.5, labelspacing=0.3,
              borderaxespad=0.7, frameon=True, framealpha=0.92,
              facecolor="white", edgecolor="#DDDDDD")


# ── Main ────────────────────────────────────────────────────────────────
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(13.4, 8.0))
    gs = fig.add_gridspec(
        2, 3,
        height_ratios=[1.0, 1.02],
        width_ratios=[1.0, 1.0, 1.0],
        hspace=0.28, wspace=0.16,
        left=0.035, right=0.985, top=0.94, bottom=0.055,
    )

    ax_A = fig.add_subplot(gs[0, 0])       # anatomy
    ax_B = fig.add_subplot(gs[0, 1:])      # BP vs LocalCA contrast (wider)
    ax_C = fig.add_subplot(gs[1, 0])       # rule family
    ax_D = fig.add_subplot(gs[1, 1])       # broadcast
    ax_E = fig.add_subplot(gs[1, 2])       # MNIST curves

    panel_A(ax_A)
    panel_B(ax_B)
    panel_C(ax_C)
    panel_D(ax_D)
    panel_E(ax_E)

    for ax, lbl, x_off in [
        (ax_A, "A", -0.04),
        (ax_B, "B", -0.02),
        (ax_C, "C", -0.05),
        (ax_D, "D", -0.05),
        (ax_E, "E", -0.15),
    ]:
        panel_label(ax, lbl, x=x_off, y=1.12, fontsize=13)

    out = OUTPUT_DIR / "fig1_model_and_credit"
    fig.savefig(out.with_suffix(".pdf"))
    fig.savefig(out.with_suffix(".png"), dpi=300)
    print(f"Saved: {out}.{{pdf,png}}")
    plt.close(fig)


if __name__ == "__main__":
    main()
