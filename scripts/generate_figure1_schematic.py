#!/usr/bin/env python3
"""Generate Figure 1: model, rules, and representative MNIST learning curves.

Creates:
  - fig_model_schematic.{pdf,png}: legacy two-panel schematic
  - fig1_model_and_credit.{pdf,png}: four-panel publication figure

The publication figure replaces the stale sandbox learning curve with
representative current MNIST runs:
  - standard shunting backprop ceiling
  - shunting LocalCA (5F, per-soma)
  - additive LocalCA (5F, per-soma)
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd

# ── Style ────────────────────────────────────────────────────────────
EXC_COLOR = "#2166AC"      # Blue - excitatory
INH_COLOR = "#B2182B"      # Red - inhibitory
DEN_COLOR = "#4DAF4A"      # Green - dendritic
SOMA_COLOR = "#FF7F00"     # Orange - soma
RULE3_COLOR = "#66C2A5"    # Teal - 3F
RULE4_COLOR = "#FC8D62"    # Salmon - 4F
RULE5_COLOR = "#8DA0CB"    # Lavender - 5F

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
    "local_shunting": "fig1_mnist_shunting_localca_fixed100_s43",
    "local_additive": "fig1_mnist_additive_localca_fixed100_s43",
}
CURVE_COLORS = {
    "bp_shunting": "#1B7837",
    "local_shunting": "#4DAF4A",
    "local_additive": "#2166AC",
}
CURVE_LABELS = {
    "bp_shunting": "Shunting BP",
    "local_shunting": "Shunting 5F-per-soma",
    "local_additive": "Additive 5F-per-soma",
}


def resolve_runs() -> dict[str, Path]:
    """Prefer the latest fixed-100-epoch Figure 1 sweep when available."""
    candidate_sweeps = sorted(
        (
            path
            for path in SWEEP_ROOT.glob(f"{FIG1_FIXED100_PREFIX}*")
            if path.is_dir()
        ),
        reverse=True,
    )
    for sweep_dir in candidate_sweeps:
        resolved: dict[str, Path] = {}
        for config_dir in sorted((sweep_dir / "results").glob("config_*")):
            config_json = config_dir / "config.json"
            if not config_json.exists():
                continue
            with open(config_json) as handle:
                config_payload = json.load(handle)
            run_name = config_payload.get("outputs", {}).get("run_name")
            for key, expected in FIG1_RUN_NAMES.items():
                if run_name == expected:
                    resolved[key] = config_dir
        if set(resolved) == set(FIG1_RUN_NAMES):
            return resolved
    return FALLBACK_RUNS


def load_epoch_history(run_dir: Path) -> pd.DataFrame:
    """Load nested epoch JSONs into a flat dataframe."""
    perf_dir = run_dir / "performance" / "epochs"
    rows = []
    for epoch_file in sorted(
        perf_dir.glob("epoch*.json"),
        key=lambda path: int(path.stem.replace("epoch", "")),
    ):
        with open(epoch_file) as handle:
            payload = json.load(handle)
        rows.append(
            {
                "epoch": int(epoch_file.stem.replace("epoch", "")),
                "train_accuracy": float(payload["accuracy"]["train"]),
                "valid_accuracy": float(payload["accuracy"]["valid"]),
                "test_accuracy": float(payload["accuracy"]["test"]),
                # Stored quantity is log-likelihood, so negate it for a loss plot.
                "train_loss": -float(payload["categorical_loglikelihood"]["train"]),
                "valid_loss": -float(payload["categorical_loglikelihood"]["valid"]),
                "test_loss": -float(payload["categorical_loglikelihood"]["test"]),
            }
        )
    return pd.DataFrame(rows)


def draw_synapse(ax, x, y, color, size=0.07):
    """Draw a small filled circle representing a synapse."""
    circle = plt.Circle((x, y), size, fc=color, ec="k", linewidth=0.5, zorder=5)
    ax.add_patch(circle)


def draw_compartment(ax, x, y, w, h, label, color, fontsize=7):
    """Draw a rounded rectangle representing a dendritic compartment."""
    box = FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.02",
        fc=color, ec="k", linewidth=0.8, alpha=0.3, zorder=3,
    )
    ax.add_patch(box)
    ax.text(x, y, label, ha="center", va="center", fontsize=fontsize,
            fontweight="bold", zorder=6)


def draw_arrow(ax, x1, y1, x2, y2, color="k", style="-|>", lw=1.0):
    """Draw an arrow from (x1,y1) to (x2,y2)."""
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle=style, color=color, lw=lw),
        zorder=4,
    )


def panel_a(ax):
    """Panel A: Dendritic neuron architecture.

    Architecture (matching the code):
      - Each branch has BOTH excitatory AND inhibitory synapses
      - E synapses: TopKLinear with reversal E_exc > 0, contribute to both
        numerator (E_j * x_j * g_j) and denominator (x_j * g_j)
      - I synapses: TopKLinear with E_inh = 0, contribute only to
        denominator (shunting / divisive normalization)
      - Dendritic conductances (BlockLinear) connect branches to parent
    """
    ax.set_xlim(-0.8, 4.7)
    ax.set_ylim(-0.5, 3.7)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("(A) Compartmental dendritic neuron", fontsize=11,
                 fontweight="bold", pad=8)

    # ── Soma ──
    soma_x, soma_y = 3.9, 1.5
    soma = plt.Circle((soma_x, soma_y), 0.28, fc=SOMA_COLOR, ec="k",
                       linewidth=1.2, alpha=0.5, zorder=5)
    ax.add_patch(soma)
    ax.text(soma_x, soma_y, "Soma\n$V_{\\mathrm{out}}$", ha="center",
            va="center", fontsize=7, fontweight="bold", zorder=6)

    # ── Branch compartments ──
    # Layer 2 (proximal, closer to soma)
    b2_x, b2_y = 2.6, 1.5
    draw_compartment(ax, b2_x, b2_y, 0.7, 0.5, "$V_{b_2}$", DEN_COLOR,
                     fontsize=8)

    # Layer 1 (distal, two branches)
    b1_pos = [(1.0, 2.6), (1.0, 0.4)]
    for i, (bx, by) in enumerate(b1_pos):
        draw_compartment(ax, bx, by, 0.7, 0.5,
                         f"$V_{{b_1}}^{{({i+1})}}$", DEN_COLOR, fontsize=8)

    # ── Dendritic conductance arrows: branches → parent ──
    # Branch2 → Soma
    draw_arrow(ax, b2_x + 0.35, b2_y, soma_x - 0.28, soma_y,
               color=DEN_COLOR, lw=1.5)

    # Branch1 → Branch2
    for bx, by in b1_pos:
        dy = 0.15 if by > 1.5 else -0.15
        draw_arrow(ax, bx + 0.35, by, b2_x - 0.35, b2_y + dy,
                   color=DEN_COLOR, lw=1.2)

    # ── External input labels ──
    ax.text(-0.58, 2.0, "Excitatory\ninputs $x_j^E$", ha="center",
            va="center", fontsize=7, color=EXC_COLOR, fontweight="bold")
    ax.text(-0.58, 1.0, "Inhibitory\ninputs $x_j^I$", ha="center",
            va="center", fontsize=7, color=INH_COLOR, fontweight="bold")

    # ── Draw E and I synapses on EVERY branch ──
    # Each branch receives both E (blue) and I (red) synapses
    all_branches = b1_pos + [(b2_x, b2_y)]

    for bx, by in all_branches:
        # Excitatory synapses (2 per branch, on left/top side)
        e_offsets = [(-0.55, 0.12), (-0.55, -0.12)]
        for dx, dy in e_offsets:
            sx, sy = bx + dx, by + dy
            draw_synapse(ax, sx, sy, EXC_COLOR, size=0.06)
            draw_arrow(ax, sx + 0.06, sy, bx - 0.35, by + dy * 0.3,
                       color=EXC_COLOR, lw=0.7)

        # Inhibitory synapses (1 per branch, on right/bottom side)
        # Drawn slightly offset to distinguish from E
        i_offsets = [(0.0, 0.38)]
        for dx, dy in i_offsets:
            sx, sy = bx + dx, by + dy
            draw_synapse(ax, sx, sy, INH_COLOR, size=0.055)
            draw_arrow(ax, sx, sy - 0.055, bx, by + 0.25,
                       color=INH_COLOR, lw=0.7)

    # (input wiring lines omitted for clarity; labels indicate shared input)

    # ── Annotation: shunting mechanism ──
    # Small annotation near proximal branch
    ax.annotate(
        "shunting:\nI enters\ndenominator",
        xy=(b2_x + 0.05, b2_y + 0.38), xytext=(b2_x + 0.65, b2_y + 0.95),
        fontsize=5, color=INH_COLOR, ha="center",
        arrowprops=dict(arrowstyle="->", color=INH_COLOR, lw=0.6),
    )

    # ── Voltage equation ──
    eq_text = (
        r"$V_n = \frac{\sum_j E_j x_j g_j^{\mathrm{syn}} + \sum_j V_j g_j^{\mathrm{den}}}"
        r"{\sum_j x_j g_j^{\mathrm{syn}} + \sum_j g_j^{\mathrm{den}} + 1}$"
    )
    ax.text(2.0, -0.35, eq_text, ha="center", va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray",
                      alpha=0.9))

    # ── Output arrow ──
    draw_arrow(ax, soma_x + 0.28, soma_y, 4.5, soma_y, color="k", lw=1.5)
    ax.text(4.55, soma_y, "output", fontsize=7, va="center")

    # ── Legend ──
    legend_items = [
        mpatches.Patch(fc=EXC_COLOR, ec="k", label="Excitatory ($E_j > 0$)",
                       alpha=0.7),
        mpatches.Patch(fc=INH_COLOR, ec="k", label="Inhibitory ($E_j = 0$)",
                       alpha=0.7),
        mpatches.Patch(fc=DEN_COLOR, ec="k", label="Dendritic cond.",
                       alpha=0.7),
        mpatches.Patch(fc=SOMA_COLOR, ec="k", label="Soma", alpha=0.5),
    ]
    ax.legend(handles=legend_items, loc="upper right", fontsize=6.2,
              framealpha=0.9)


def panel_b(ax):
    """Panel B: Local learning rule hierarchy (3F -> 4F -> 5F)."""
    ax.set_xlim(0.0, 2.7)
    ax.set_ylim(0.0, 3.15)
    ax.axis("off")
    ax.set_title("(B) Local learning rule hierarchy", fontsize=11,
                 fontweight="bold", pad=8)

    rules = [
        ("3F", 2.42, RULE3_COLOR,
         r"$x(E{-}V)e$",
         "eligibility × broadcast"),
        ("4F", 1.52, RULE4_COLOR,
         r"$3F \times \rho$",
         "morphology modulation"),
        ("5F", 0.62, RULE5_COLOR,
         r"$4F \times \phi$",
         "confidence modulation"),
    ]

    for name, y_center, color, equation, description in rules:
        box = FancyBboxPatch(
            (0.08, y_center - 0.28), 2.35, 0.56,
            boxstyle="round,pad=0.05",
            fc=color, ec="k", linewidth=0.8, alpha=0.15, zorder=2,
        )
        ax.add_patch(box)
        tag = FancyBboxPatch(
            (0.15, y_center - 0.17), 0.42, 0.34,
            boxstyle="round,pad=0.03",
            fc=color, ec="k", linewidth=0.6, alpha=0.75, zorder=3,
        )
        ax.add_patch(tag)
        ax.text(0.36, y_center, name, ha="center", va="center",
                fontsize=11.0, fontweight="bold", color="white", zorder=5)
        ax.text(0.72, y_center + 0.07, equation, ha="left", va="center",
                fontsize=8.9, zorder=5)
        ax.text(0.72, y_center - 0.10, description, ha="left", va="center",
                fontsize=6.5, color="gray", style="italic", zorder=5)

    # Arrows: 3F → 4F → 5F
    for y_top, y_bot in [(2.08, 1.83), (1.18, 0.93)]:
        ax.annotate(
            "", xy=(0.36, y_bot), xytext=(0.36, y_top),
            arrowprops=dict(arrowstyle="->", color="gray", lw=0.9, ls="--"),
        )

    ax.text(0.08, 2.96, "base LocalCA family", fontsize=6.7, color="gray")


def panel_c_broadcast(ax):
    """Panel C: broadcast field schematic."""
    ax.set_xlim(0.0, 2.35)
    ax.set_ylim(0.0, 3.0)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("(C) Broadcast modes", fontsize=11, fontweight="bold", pad=8)

    def draw_mini_tree(x, y, scale=1.0):
        soma = (x + 0.44 * scale, y)
        hub = (x + 0.20 * scale, y)
        leaves = [
            (x - 0.06 * scale, y + 0.22 * scale),
            (x - 0.06 * scale, y),
            (x - 0.06 * scale, y - 0.22 * scale),
        ]
        ax.plot([hub[0], soma[0]], [hub[1], soma[1]], color=DEN_COLOR, lw=1.4, zorder=2)
        for leaf in leaves:
            ax.plot([leaf[0], hub[0]], [leaf[1], hub[1]], color=DEN_COLOR, lw=1.1, zorder=2)
            ax.add_patch(plt.Circle(leaf, 0.025 * scale, fc=DEN_COLOR, ec="k",
                                    linewidth=0.4, alpha=0.65, zorder=3))
        ax.add_patch(plt.Circle(hub, 0.028 * scale, fc=DEN_COLOR, ec="k",
                                linewidth=0.4, alpha=0.70, zorder=3))
        ax.add_patch(plt.Circle(soma, 0.045 * scale, fc=SOMA_COLOR, ec="k",
                                linewidth=0.5, alpha=0.75, zorder=3))
        return soma, hub, leaves

    def draw_mode(y, mode, title, subtitle):
        soma, hub, leaves = draw_mini_tree(1.05, y, scale=1.0)
        if mode == "scalar":
            source = (2.08, y)
            ax.add_patch(plt.Circle(source, 0.05, fc=INH_COLOR, ec="k",
                                    linewidth=0.4, alpha=0.85, zorder=4))
            ax.annotate(
                "", xy=hub, xytext=source,
                arrowprops=dict(arrowstyle="-|>", color=INH_COLOR, lw=1.6),
            )
            ax.text(source[0], y, r"$e$", ha="center", va="center",
                    fontsize=7.2, color="white", fontweight="bold")
        elif mode == "per_soma":
            starts = [(2.02, y + 0.18), (2.02, y), (2.02, y - 0.18)]
            vec_box = FancyBboxPatch(
                (1.93, y - 0.26), 0.18, 0.52,
                boxstyle="round,pad=0.02", fc=INH_COLOR, ec="k",
                linewidth=0.4, alpha=0.18, zorder=3,
            )
            ax.add_patch(vec_box)
            for start, leaf in zip(starts, leaves):
                ax.annotate(
                    "", xy=leaf, xytext=start,
                    arrowprops=dict(arrowstyle="-|>", color=INH_COLOR, lw=1.0),
                )
            ax.text(2.02, y, r"$\delta$", ha="center", va="center",
                    fontsize=7.2, color=INH_COLOR, fontweight="bold")
        elif mode == "structured":
            colors = ["#7B3294", "#C51B7D"]
            channel_pts = [(2.02, y + 0.14), (2.02, y - 0.14)]
            labels = [r"$c_1$", r"$c_2$"]
            for (cx, cy), color, label in zip(channel_pts, colors, labels):
                ax.add_patch(plt.Circle((cx, cy), 0.032, fc=color, ec="k",
                                        linewidth=0.4, alpha=0.8, zorder=4))
                ax.text(cx, cy, label, ha="center", va="center",
                        fontsize=6.2, color="white", fontweight="bold", zorder=5)
            for leaf in leaves[:2]:
                ax.annotate(
                    "", xy=leaf, xytext=channel_pts[0],
                    arrowprops=dict(arrowstyle="-|>", color=colors[0], lw=1.0),
                )
                ax.annotate(
                    "", xy=leaves[2], xytext=channel_pts[1],
                    arrowprops=dict(arrowstyle="-|>", color=colors[1], lw=1.0),
                )
        ax.text(0.05, y + 0.09, title, ha="left", va="center",
                fontsize=6.9, fontweight="bold")
        ax.text(0.05, y - 0.10, subtitle, ha="left", va="center",
                fontsize=5.8, color="gray")

    draw_mode(2.35, "scalar", "scalar", "one shared field")
    draw_mode(1.50, "per_soma", "per-soma", "vector from soma to all branches")
    draw_mode(0.65, "structured", "structured pathways", "separate channels for distinct branches")


def panel_d(ax, histories: dict[str, pd.DataFrame]):
    """Panel D: representative MNIST accuracy curves."""
    ax.set_title("(D) Representative MNIST accuracy", fontsize=11, fontweight="bold", pad=8)

    for key in ["bp_shunting", "local_shunting", "local_additive"]:
        df = histories[key]
        style = "--" if key == "bp_shunting" else "-"
        lw = 2.0 if key != "bp_shunting" else 1.8
        ax.plot(
            df["epoch"],
            100.0 * df["test_accuracy"],
            style,
            lw=lw,
            color=CURVE_COLORS[key],
            label=CURVE_LABELS[key],
        )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Test accuracy (%)")
    ax.set_ylim(0, 100)
    ax.set_xlim(1, max(int(histories[key]["epoch"].max()) for key in histories))
    ax.grid(alpha=0.18, linewidth=0.5)
    ax.legend(loc="lower right", fontsize=6.3, handlelength=2.2)


def panel_e(ax, histories: dict[str, pd.DataFrame]):
    """Panel E: representative MNIST loss curves."""
    ax.set_title("(E) Representative MNIST loss", fontsize=11, fontweight="bold", pad=8)

    for key in ["bp_shunting", "local_shunting", "local_additive"]:
        df = histories[key]
        style = "--" if key == "bp_shunting" else "-"
        lw = 2.0 if key != "bp_shunting" else 1.8
        ax.plot(
            df["epoch"],
            df["test_loss"],
            style,
            lw=lw,
            color=CURVE_COLORS[key],
            label=CURVE_LABELS[key],
        )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Test NLL")
    ax.set_ylim(bottom=0.0)
    ax.set_xlim(1, max(int(histories[key]["epoch"].max()) for key in histories))
    ax.grid(alpha=0.18, linewidth=0.5)
    ax.legend(loc="upper right", fontsize=6.3, handlelength=2.2)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Legacy two-panel schematic.
    fig, (ax_a, ax_b) = plt.subplots(
        1, 2, figsize=(13.4, 5.0),
        gridspec_kw={"width_ratios": [1.1, 1]},
    )

    panel_a(ax_a)
    panel_b(ax_b)

    fig.tight_layout(pad=1.5)

    out_path = OUTPUT_DIR / "fig_model_schematic"
    fig.savefig(out_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    print(f"Saved: {out_path}.{{png,pdf}}")
    plt.close(fig)

    # Publication figure used in the paper.
    runs = resolve_runs()
    histories = {name: load_epoch_history(path) for name, path in runs.items()}
    fig = plt.figure(figsize=(13.4, 8.8))
    gs = fig.add_gridspec(2, 12, height_ratios=[1.0, 0.9], hspace=0.34, wspace=0.14)
    ax_a = fig.add_subplot(gs[0, 0:6])
    ax_b = fig.add_subplot(gs[0, 6:9])
    ax_c = fig.add_subplot(gs[0, 9:12])
    ax_d = fig.add_subplot(gs[1, 0:6])
    ax_e = fig.add_subplot(gs[1, 6:12])

    panel_a(ax_a)
    panel_b(ax_b)
    panel_c_broadcast(ax_c)
    panel_d(ax_d, histories)
    panel_e(ax_e, histories)

    fig.subplots_adjust(left=0.04, right=0.985, top=0.965, bottom=0.07)
    out_path = OUTPUT_DIR / "fig1_model_and_credit"
    fig.savefig(out_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    print(f"Saved: {out_path}.{{png,pdf}}")
    plt.close(fig)


if __name__ == "__main__":
    main()
