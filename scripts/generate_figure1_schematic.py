#!/usr/bin/env python3
"""Generate Figure 1: model, computation, and local credit assignment.

Outputs:
  - figures/fig1_model_and_credit.{pdf,png}

The figure is intentionally built as an orientation figure:
  A. Network, exact credit, and local approximation
  B. Single-compartment conductance computation
  C. Local rule family
  D. Broadcast modes
  E. Representative MNIST learning dynamics
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


EXC = COLORS["exc"]
INH = COLORS["inh"]
DEND = COLORS["dend"]
SOMA = COLORS["soma"]
INK = COLORS["ink"]
MUTE = COLORS["mute"]
EDGE = COLORS["edge"]
PANEL_BG = "#FBFBFC"
EXACT = "#B13138"
APPROX = "#E88B69"

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
    "rule_3f": "fig1_mnist_shunting_3f_fixed100_s43",
    "rule_4f": "fig1_mnist_shunting_4f_fixed100_s43",
    "rule_5f": "fig1_mnist_shunting_localca_fixed100_s43",
}

CURVE_STYLES = {
    "bp_shunting": dict(color="#185A33", ls="-", label="Shunt. BP (oracle)"),
    "rule_5f": dict(color=COLORS["rule_5f"], ls="-", label="Shunt. 5F"),
    "rule_4f": dict(color=COLORS["rule_4f"], ls="--", label="Shunt. 4F"),
    "rule_3f": dict(color=COLORS["rule_3f"], ls=":", label="Shunt. 3F"),
    "local_additive": dict(color=COLORS["additive"], ls="-.", label="Add. 5F"),
}


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
        rows.append(
            {
                "epoch": int(epoch_file.stem.replace("epoch", "")),
                "train_accuracy": float(payload["accuracy"]["train"]),
                "test_accuracy": float(payload["accuracy"]["test"]),
            }
        )
    return pd.DataFrame(rows)


def draw_arrow(
    ax,
    x1,
    y1,
    x2,
    y2,
    color=INK,
    lw=1.2,
    style="-|>",
    mutation_scale=10,
    zorder=5,
    alpha=1.0,
    linestyle="-",
    connectionstyle="arc3",
):
    arrow = FancyArrowPatch(
        (x1, y1),
        (x2, y2),
        arrowstyle=style,
        mutation_scale=mutation_scale,
        color=color,
        lw=lw,
        alpha=alpha,
        zorder=zorder,
        shrinkA=0,
        shrinkB=0,
        linestyle=linestyle,
        connectionstyle=connectionstyle,
    )
    ax.add_patch(arrow)


def rounded_box(
    ax,
    x,
    y,
    w,
    h,
    fc="white",
    ec=EDGE,
    lw=0.8,
    radius=0.025,
    alpha=1.0,
):
    patch = FancyBboxPatch(
        (x - w / 2, y - h / 2),
        w,
        h,
        boxstyle=f"round,pad=0.01,rounding_size={radius}",
        fc=fc,
        ec=ec,
        linewidth=lw,
        alpha=alpha,
        zorder=2,
    )
    ax.add_patch(patch)
    return patch


def draw_synapse(ax, x, y, kind="exc", size=0.012):
    if kind == "exc":
        circ = Circle((x, y), size, fc=EXC, ec="white", linewidth=0.4, zorder=6)
        ax.add_patch(circ)
    else:
        tri = mpatches.RegularPolygon(
            (x, y),
            numVertices=3,
            radius=size * 1.55,
            orientation=np.pi,
            fc=INH,
            ec="white",
            linewidth=0.4,
            zorder=6,
        )
        ax.add_patch(tri)


def draw_tree(ax, x_leaf=0.17, x_branch=0.34, x_soma=0.51, y0=0.18, y1=0.82):
    """Return representative coordinates for the orientation tree."""
    branch_ys = np.array([0.72, 0.50, 0.28])
    leaf_ys = []
    for by in branch_ys:
        leaf_ys.extend([by + 0.08, by, by - 0.08])
    leaf_ys = np.array(leaf_ys)
    soma_xy = (x_soma, 0.50)

    # wires: leaf -> branch
    for bi, by in enumerate(branch_ys):
        group = leaf_ys[bi * 3 : (bi + 1) * 3]
        for ly in group:
            ax.plot(
                [x_leaf + 0.03, x_branch - 0.04],
                [ly, by + (ly - by) * 0.25],
                color=DEND,
                lw=1.2,
                alpha=0.78,
                zorder=1,
                solid_capstyle="round",
            )

    # wires: branch -> soma
    for by in branch_ys:
        ax.plot(
            [x_branch + 0.045, x_soma - 0.045],
            [by, 0.50 + (by - 0.50) * 0.22],
            color=DEND,
            lw=2.0,
            alpha=0.82,
            zorder=1,
            solid_capstyle="round",
        )

    # leaf compartments
    for ly in leaf_ys:
        rounded_box(ax, x_leaf, ly, 0.065, 0.042, fc=DEND, ec=EDGE, alpha=0.88)

    # branch compartments
    for by in branch_ys:
        rounded_box(ax, x_branch, by, 0.092, 0.060, fc=DEND, ec=EDGE, alpha=0.96)

    # soma
    soma = Circle(soma_xy, 0.045, fc=SOMA, ec=EDGE, linewidth=0.9, zorder=4)
    ax.add_patch(soma)

    # labels on representative nodes only
    ax.text(
        x_leaf,
        leaf_ys[4],
        r"$V_n$",
        ha="center",
        va="center",
        fontsize=8.0,
        color="white",
        fontweight="bold",
        zorder=7,
        path_effects=[withStroke(linewidth=0.8, foreground="#0A3A23")],
    )
    ax.text(
        x_branch,
        branch_ys[1],
        r"$V_{p(n)}$",
        ha="center",
        va="center",
        fontsize=8.0,
        color="white",
        fontweight="bold",
        zorder=7,
        path_effects=[withStroke(linewidth=0.8, foreground="#0A3A23")],
    )
    ax.text(
        soma_xy[0],
        soma_xy[1],
        r"$V_{\mathrm{out}}$",
        ha="center",
        va="center",
        fontsize=8.0,
        color="white",
        fontweight="bold",
        zorder=7,
    )

    # synapses on leaves
    for ly in leaf_ys:
        draw_synapse(ax, x_leaf - 0.048, ly - 0.014, "exc", 0.0085)
        draw_synapse(ax, x_leaf - 0.048, ly + 0.014, "exc", 0.0085)
        draw_synapse(ax, x_leaf, ly + 0.038, "inh", 0.0088)

    return {"leaf_x": x_leaf, "leaf_ys": leaf_ys, "branch_x": x_branch, "branch_ys": branch_ys, "soma": soma_xy}


def panel_A(ax):
    clean_schematic_axis(ax)
    ax.set_aspect("auto")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(
        "Model, exact credit, and LocalCA broadcast",
        fontsize=10.4,
        pad=4,
        loc="left",
        x=0.045,
    )

    # light panel strips
    rounded_box(ax, 0.27, 0.50, 0.48, 0.82, fc=PANEL_BG, ec="#E5E7EB", lw=0.7, radius=0.03)
    rounded_box(ax, 0.83, 0.50, 0.28, 0.82, fc=PANEL_BG, ec="#E5E7EB", lw=0.7, radius=0.03)

    # left: tree
    tree = draw_tree(ax)
    soma_xy = tree["soma"]
    branch_ys = tree["branch_ys"]
    leaf_ys = tree["leaf_ys"]
    x_leaf = tree["leaf_x"]
    x_branch = tree["branch_x"]

    # input bundles
    rounded_box(ax, 0.06, 0.64, 0.10, 0.08, fc="#EAF3FB", ec=EXC, lw=0.8)
    ax.text(0.06, 0.64, "input\n$x_j$", ha="center", va="center", fontsize=8.2, color=INK)
    rounded_box(ax, 0.06, 0.36, 0.10, 0.08, fc="#FBEAEC", ec=INH, lw=0.8)
    ax.text(0.06, 0.36, "inh.\n$E^I=0$", ha="center", va="center", fontsize=8.0, color=INK)
    draw_arrow(ax, 0.11, 0.64, x_leaf - 0.065, 0.64, color=EXC, lw=1.2, mutation_scale=10)
    draw_arrow(ax, 0.11, 0.36, x_leaf - 0.055, 0.36, color=INH, lw=1.1, mutation_scale=10)

    # output / loss
    rounded_box(ax, 0.69, 0.56, 0.09, 0.07, fc="#FFF2E8", ec=SOMA, lw=0.9)
    ax.text(0.69, 0.56, "decoder", ha="center", va="center", fontsize=8.0)
    rounded_box(ax, 0.81, 0.56, 0.08, 0.07, fc="#F6F1FA", ec=COLORS["oracle"], lw=0.9)
    ax.text(0.81, 0.56, "loss", ha="center", va="center", fontsize=8.2)
    draw_arrow(ax, soma_xy[0] + 0.045, soma_xy[1], 0.645, 0.56, color=INK, lw=1.3, mutation_scale=10)
    draw_arrow(ax, 0.735, 0.56, 0.77, 0.56, color=INK, lw=1.3, mutation_scale=10)
    ax.text(0.59, 0.57, "output", fontsize=7.8, color=MUTE, va="bottom")

    # exact top-down compartment-specific errors
    ax.text(
        0.73,
        0.86,
        "exact backprop:\ncompartment-specific $\\partial L/\\partial V_n$",
        ha="center",
        va="center",
        fontsize=7.6,
        color=EXACT,
        fontweight="bold",
    )
    hub_exact = (0.72, 0.78)
    draw_arrow(ax, 0.81, 0.60, hub_exact[0], hub_exact[1], color=EXACT, lw=1.0,
               mutation_scale=9, linestyle="--", alpha=0.95)
    for tgt in [(x_branch, branch_ys[0] + 0.045), (x_branch, branch_ys[1]), (x_leaf, leaf_ys[7])]:
        draw_arrow(ax, hub_exact[0], hub_exact[1], tgt[0] + 0.01, tgt[1] + 0.01,
                   color=EXACT, lw=0.95, mutation_scale=8, linestyle="--", alpha=0.8,
                   connectionstyle="arc3,rad=0.10")

    # localca broadcast from soma to tree
    ax.text(
        0.73,
        0.18,
        "LocalCA:\nshared low-bandwidth broadcast $e_n$",
        ha="center",
        va="center",
        fontsize=7.6,
        color=APPROX,
        fontweight="bold",
    )
    hub_local = (0.69, 0.26)
    draw_arrow(ax, soma_xy[0] + 0.01, soma_xy[1] - 0.045, hub_local[0], hub_local[1],
               color=APPROX, lw=1.1, mutation_scale=9, alpha=0.95)
    for tgt in [(x_branch, branch_ys[2] - 0.04), (x_branch, branch_ys[1] - 0.01), (x_leaf, leaf_ys[1] - 0.02)]:
        draw_arrow(ax, hub_local[0], hub_local[1], tgt[0] + 0.005, tgt[1],
                   color=APPROX, lw=0.95, mutation_scale=8, alpha=0.78,
                   connectionstyle="arc3,rad=-0.10")

    # local update callout
    rounded_box(
        ax, 0.83, 0.37, 0.255, 0.205, fc="white", ec="#D7DCE2", lw=0.8, radius=0.025
    )
    ax.text(
        0.83,
        0.445,
        "local synapse update",
        ha="center",
        va="center",
        fontsize=8.0,
        color=INK,
        fontweight="bold",
    )
    ax.text(
        0.83,
        0.39,
        r"$\Delta g_j \propto x_j\,R_n^{\mathrm{tot}}\,(E_j-V_n)\,e_n$",
        ha="center",
        va="center",
        fontsize=9.6,
        color=INK,
    )
    ax.text(0.715, 0.322, "local:", fontsize=7.0, color=MUTE, fontweight="bold")
    ax.text(0.768, 0.322, r"$x_j,\;V_n,\;E_j,\;R_n^{\mathrm{tot}}$", fontsize=6.8, color=MUTE)
    ax.text(0.715, 0.286, "non-local:", fontsize=7.0, color=MUTE, fontweight="bold")
    ax.text(0.807, 0.286, r"$\partial L/\partial V_n$ or $e_n$", fontsize=6.8, color=MUTE)

    # morphology labels
    ax.text(
        x_leaf,
        0.082,
        "distal\ncompartments",
        ha="center",
        va="top",
        fontsize=7.0,
        color=MUTE,
        style="italic",
    )
    ax.text(
        x_branch,
        0.082,
        "proximal\nbranches",
        ha="center",
        va="top",
        fontsize=7.0,
        color=MUTE,
        style="italic",
    )
    ax.text(
        soma_xy[0],
        0.082,
        "soma",
        ha="center",
        va="top",
        fontsize=7.0,
        color=MUTE,
        style="italic",
    )
    ax.text(0.27, 0.022, r"tree $[3,3]$", ha="center", va="bottom", fontsize=6.8, color=MUTE)


def _draw_hist(ax, x0, y0, w, h, color, spread, title):
    bins = np.linspace(0, 1, 15)
    rng = np.random.default_rng(int(spread * 1000))
    samples = np.clip(rng.normal(0.52, spread, size=500), 0.02, 0.98)
    counts, _ = np.histogram(samples, bins=bins)
    counts = counts / counts.max()
    ax.plot([x0, x0 + w], [y0, y0], color="#C9CED6", lw=0.6)
    for i, c in enumerate(counts):
        bw = w / len(counts) * 0.78
        bx = x0 + i * (w / len(counts)) + 0.01 * w
        ax.add_patch(plt.Rectangle((bx, y0), bw, c * h, fc=color, ec=color, alpha=0.86, lw=0.25))
    ax.text(
        x0 + w / 2,
        y0 + h + 0.018,
        title,
        ha="center",
        va="bottom",
        fontsize=6.4,
        color=color,
        fontweight="bold",
    )
    ax.text(x0 + w / 2, y0 - 0.04, "path gain", ha="center", va="top", fontsize=6.2, color=MUTE, style="italic")


def panel_B(ax):
    clean_schematic_axis(ax)
    ax.set_aspect("auto")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("Compartment computation", fontsize=8.9, pad=3, loc="left", x=0.13)

    rounded_box(ax, 0.51, 0.68, 0.20, 0.12, fc="#F8FBF9", ec=DEND, lw=0.8)
    ax.text(0.51, 0.68, r"compartment $n$", ha="center", va="center", fontsize=8.6, fontweight="bold")

    # inputs
    rounded_box(ax, 0.16, 0.78, 0.16, 0.09, fc="#EAF3FB", ec=EXC, lw=0.8)
    ax.text(0.16, 0.78, "exc. synapses\n$g_j x_j E_j$", ha="center", va="center", fontsize=7.4)
    rounded_box(ax, 0.16, 0.58, 0.16, 0.09, fc="#FBEAEC", ec=INH, lw=0.8)
    ax.text(0.16, 0.58, "inh. synapses\n$g_j x_j$", ha="center", va="center", fontsize=7.4)
    rounded_box(ax, 0.50, 0.44, 0.20, 0.09, fc="#EEF7F1", ec=DEND, lw=0.8)
    ax.text(0.50, 0.44, "child branches\n$g_j^{\\mathrm{den}}V_j$", ha="center", va="center", fontsize=7.4)
    rounded_box(ax, 0.84, 0.68, 0.16, 0.09, fc="#FFF2E8", ec=SOMA, lw=0.8)
    ax.text(0.84, 0.68, "to parent\n$V_n$", ha="center", va="center", fontsize=7.4)

    draw_arrow(ax, 0.24, 0.76, 0.40, 0.70, color=EXC, lw=1.2, mutation_scale=10)
    draw_arrow(ax, 0.24, 0.60, 0.40, 0.66, color=INH, lw=1.2, mutation_scale=10)
    draw_arrow(ax, 0.50, 0.485, 0.50, 0.62, color=DEND, lw=1.2, mutation_scale=10)
    draw_arrow(ax, 0.61, 0.68, 0.76, 0.68, color=INK, lw=1.2, mutation_scale=10)

    # equation box
    rounded_box(ax, 0.50, 0.24, 0.90, 0.18, fc="white", ec="#D7DCE2", lw=0.8)
    ax.text(0.50, 0.28, r"$V_n \;=\; \frac{\sum_j g_j x_j E_j \;+\; \sum_j g_j^{\mathrm{den}} V_j}{g_n^{\mathrm{tot}}}$",
            ha="center", va="center", fontsize=12.0, color=INK)
    ax.text(0.50, 0.18, r"$g_n^{\mathrm{tot}} \;=\; g^{\mathrm{leak}} + \sum_{j\in E} g_j x_j + \sum_{j\in I} g_j x_j + \sum_j g_j^{\mathrm{den}}$",
            ha="center", va="center", fontsize=8.5, color=INK)

    ax.text(
        0.50,
        0.112,
        "path-gain distributions",
        ha="center",
        va="center",
        fontsize=6.1,
        color=MUTE,
        style="italic",
    )

    _draw_hist(ax, 0.08, 0.01, 0.34, 0.075, COLORS["shunting"], 0.10, "shunting")
    _draw_hist(ax, 0.58, 0.01, 0.34, 0.075, COLORS["additive"], 0.26, "additive")


def panel_C(ax):
    clean_schematic_axis(ax)
    ax.set_aspect("auto")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("Rule family", fontsize=8.9, pad=3, loc="left", x=0.13)

    cards = [
        ("3F", COLORS["rule_3f"], "theorem-facing", "exact", r"$\Delta g_j \propto x_j\,R_n^{\mathrm{tot}}(E_j-V_n)\,e_n$",
         "eligibility × broadcast"),
        ("4F", COLORS["rule_4f"], r"adds $\rho_n$", "heuristic", r"$\Delta g_j \propto \mathrm{3F}\cdot\rho_n$",
         "morphology-aware modulation"),
        ("5F", COLORS["rule_5f"], r"adds $\phi_n$", "heuristic", r"$\Delta g_j \propto \mathrm{4F}\cdot\phi_n$",
         "confidence / correctness modulation"),
    ]
    ys = [0.78, 0.50, 0.22]
    for (name, color, sub, badge, eq, expl), y in zip(cards, ys):
        rounded_box(ax, 0.50, y, 0.94, 0.22, fc="white", ec="#D7DCE2", lw=0.75, radius=0.03)
        rounded_box(ax, 0.12, y, 0.16, 0.16, fc=color, ec=color, lw=0.6, radius=0.03)
        ax.text(0.12, y, name, ha="center", va="center", fontsize=12.0, color="white", fontweight="bold")

        badge_fc = "#EDF7F0" if badge == "exact" else "#FBF0EC"
        badge_ec = COLORS["shunting"] if badge == "exact" else COLORS["rule_4f"]
        rounded_box(ax, 0.80, y + 0.06, 0.13, 0.055, fc=badge_fc, ec=badge_ec, lw=0.7, radius=0.02)
        ax.text(0.80, y + 0.06, badge, ha="center", va="center", fontsize=6.5, color=badge_ec, fontweight="bold")

        ax.text(0.26, y + 0.062, sub, ha="left", va="center", fontsize=6.7, color=MUTE, fontweight="bold")
        ax.text(0.26, y + 0.03, eq, ha="left", va="center", fontsize=8.6, color=INK)
        ax.text(0.26, y - 0.055, expl, ha="left", va="center", fontsize=6.8, color=MUTE, style="italic")


def _broadcast_row(ax, y, label, sub, color, mode):
    ax.text(0.03, y + 0.028, label, fontsize=8.3, fontweight="bold", color=color, ha="left")
    ax.text(0.03, y - 0.032, sub, fontsize=6.7, color=MUTE, ha="left", style="italic")

    # source
    src = (0.30, y)
    ax.add_patch(Circle(src, 0.016, fc=color, ec="white", linewidth=0.4, zorder=6))
    ax.text(src[0], y + 0.055, r"$\delta_0$", ha="center", va="center", fontsize=7.0, color=color, fontweight="bold")

    # targets
    targets = [(0.62, y + 0.06), (0.62, y), (0.62, y - 0.06)]
    for tx, ty in targets:
        rounded_box(ax, tx, ty, 0.08, 0.04, fc="#F7FAF8", ec=DEND, lw=0.55, radius=0.02, alpha=0.95)

    if mode == "scalar":
        for tx, ty in targets:
            draw_arrow(ax, src[0] + 0.02, src[1], tx - 0.05, ty, color=color, lw=0.95, mutation_scale=7, alpha=0.85)
    elif mode == "per_soma":
        mids = [(0.44, y + 0.035), (0.44, y - 0.035)]
        for mid in mids:
            ax.add_patch(Circle(mid, 0.011, fc=color, ec="white", linewidth=0.35, zorder=6, alpha=0.92))
        for mid, tgt in zip(mids, [targets[0], targets[2]]):
            draw_arrow(ax, src[0] + 0.02, src[1], mid[0] - 0.012, mid[1], color=color, lw=0.85, mutation_scale=7, alpha=0.75)
            draw_arrow(ax, mid[0] + 0.012, mid[1], tgt[0] - 0.05, tgt[1], color=color, lw=0.85, mutation_scale=7, alpha=0.75)
        draw_arrow(ax, src[0] + 0.02, src[1], targets[1][0] - 0.05, targets[1][1], color=color, lw=0.85, mutation_scale=7, alpha=0.75)
    elif mode == "low_rank":
        chans = [(0.41, y + 0.05), (0.41, y - 0.05)]
        for ch in chans:
            ax.add_patch(Circle(ch, 0.011, fc=color, ec="white", linewidth=0.35, zorder=6, alpha=0.92))
            draw_arrow(ax, src[0] + 0.02, src[1], ch[0] - 0.012, ch[1], color=color, lw=0.85, mutation_scale=7, alpha=0.75)
            for tx, ty in targets:
                draw_arrow(ax, ch[0] + 0.012, ch[1], tx - 0.05, ty, color=color, lw=0.75, mutation_scale=6, alpha=0.62)
        ax.text(0.46, y + 0.067, r"$K$", fontsize=6.8, color=color, fontweight="bold")
    else:
        branch_colors = ["#7C5AA6", "#C15A8A", "#D08C2F"]
        for bc, tgt in zip(branch_colors, targets):
            mid = (0.43, tgt[1])
            ax.add_patch(Circle(mid, 0.0105, fc=bc, ec="white", linewidth=0.35, zorder=6, alpha=0.95))
            draw_arrow(ax, src[0] + 0.02, src[1], mid[0] - 0.011, mid[1], color=bc, lw=0.8, mutation_scale=7, alpha=0.78)
            draw_arrow(ax, mid[0] + 0.011, mid[1], tgt[0] - 0.05, tgt[1], color=bc, lw=0.8, mutation_scale=7, alpha=0.78)


def panel_D(ax):
    clean_schematic_axis(ax)
    ax.set_aspect("auto")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("Broadcast modes", fontsize=8.9, pad=3, loc="left", x=0.13)

    rows = [
        ("scalar", "one shared field", COLORS["scalar"], "scalar"),
        ("per-soma", "vector when widths match, else scalar fallback", COLORS["per_soma"], "per_soma"),
        ("low-rank", r"$K$ random channels", COLORS["low_rank"], "low_rank"),
        ("pathway", "structured branch-specific channels", COLORS["pathway"], "pathway"),
    ]
    ys = [0.82, 0.60, 0.38, 0.16]
    for (label, sub, color, mode), y in zip(rows, ys):
        _broadcast_row(ax, y, label, sub, color, mode)

    ax.text(0.36, 0.02, r"broadcast rank: $1 \rightarrow N \rightarrow K \rightarrow M$",
            fontsize=6.8, color=MUTE, style="italic")


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
            color=style["color"],
            ls=style["ls"],
            lw=1.6,
            label=style["label"],
            alpha=0.96,
        )

    ax.axhline(10, color=MUTE, ls=":", lw=0.7, alpha=0.55, zorder=0)
    ax.text(98.5, 11.8, "chance", color=MUTE, fontsize=6.7, ha="right", va="bottom")
    ax.set_xlim(0, 100)
    ax.set_ylim(5, 100)
    ax.set_xlabel("epoch")
    ax.set_ylabel("MNIST test accuracy (%)")
    ax.set_title("MNIST dynamics", fontsize=8.9, pad=3, loc="left", x=0.13)
    style_axis(ax, grid="y")
    ax.legend(
        loc="lower right",
        fontsize=6.7,
        ncol=1,
        handlelength=1.8,
        handletextpad=0.45,
        labelspacing=0.25,
        borderaxespad=0.5,
        frameon=True,
        framealpha=0.92,
        facecolor="white",
        edgecolor="#DDDDDD",
    )


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(13.2, 8.25))
    gs = fig.add_gridspec(
        2,
        4,
        height_ratios=[1.03, 1.17],
        width_ratios=[1.02, 1.02, 1.08, 1.18],
        hspace=0.28,
        wspace=0.26,
        left=0.04,
        right=0.985,
        top=0.93,
        bottom=0.07,
    )

    ax_A = fig.add_subplot(gs[0, :])
    ax_B = fig.add_subplot(gs[1, 0])
    ax_C = fig.add_subplot(gs[1, 1])
    ax_D = fig.add_subplot(gs[1, 2])
    ax_E = fig.add_subplot(gs[1, 3])

    panel_A(ax_A)
    panel_B(ax_B)
    panel_C(ax_C)
    panel_D(ax_D)
    panel_E(ax_E)

    panel_label(ax_A, "A", x=-0.015, y=1.055, fontsize=12.5)
    panel_label(ax_B, "B", x=-0.075, y=1.06, fontsize=12.5)
    panel_label(ax_C, "C", x=-0.075, y=1.06, fontsize=12.5)
    panel_label(ax_D, "D", x=-0.070, y=1.06, fontsize=12.5)
    panel_label(ax_E, "E", x=-0.080, y=1.06, fontsize=12.5)

    out_path = OUTPUT_DIR / "fig1_model_and_credit"
    fig.savefig(out_path.with_suffix(".pdf"))
    fig.savefig(out_path.with_suffix(".png"), dpi=300)
    print(f"Saved: {out_path}.{{pdf,png}}")
    plt.close(fig)


if __name__ == "__main__":
    main()
