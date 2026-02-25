#!/usr/bin/env python3
"""Generate revision figures for NeurIPS 2026 paper.

Produces:
  Main:
    fig5_mechanistic_evidence.pdf  (3 panels: R_tot, low-bandwidth, additive+norm)
  Supplement:
    fig_s5_fa_dfa.pdf              (FA/DFA baseline comparison)
    fig_s6_cifar10.pdf             (CIFAR-10 results)
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

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DRAFT_DIR = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(DRAFT_DIR, "data")
FIGURES_DIR = os.path.join(DRAFT_DIR, "figures")

# ---------------------------------------------------------------------------
# Style (matches generate_neurips_figures.py)
# ---------------------------------------------------------------------------
COLOR_SHUNTING = "#18864B"
COLOR_ADDITIVE = "#2D5DA8"
COLOR_BACKPROP = "#666666"
COLOR_POINT_MLP = "#999999"

W = 5.5  # NeurIPS single-column width
DPI = 300


def _setup_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 7,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "axes.titlepad": 6,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "legend.fontsize": 6,
        "figure.dpi": DPI,
        "savefig.dpi": DPI,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.major.pad": 2,
        "ytick.major.pad": 2,
    })


def _panel(ax, label, x=-0.18, y=1.12):
    ax.text(x, y, label, transform=ax.transAxes, fontsize=11,
            fontweight="bold", va="top", ha="left")


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


# ===================================================================
# Figure 5 — Mechanistic Evidence (NEW main figure)
# ===================================================================
def figure5():
    """Three panels: (A) R_tot distributions, (B) low-bandwidth degradation,
    (C) additive + normalization control."""
    print("\n--- Figure 5: Mechanistic Evidence ---")

    rtot = _csv("rtot_distributions.csv")
    lbw = _csv("low_bandwidth_results.csv")
    anorm = _csv("additive_norm_results.csv")

    fig, axes = plt.subplots(1, 3, figsize=(W, 2.6),
                             gridspec_kw={"wspace": 0.55})

    # ---- Panel A: R_tot distributions ----
    ax = axes[0]
    _panel(ax, "A")

    if rtot is not None:
        # Use only the first branch layer (outermost = receives external input)
        bl0 = rtot[rtot["layer_name"].str.contains("branch_layers.0")]
        for model_type, color, label in [
            ("dendritic_shunting", COLOR_SHUNTING, "Shunting"),
            ("dendritic_additive", COLOR_ADDITIVE, "Additive"),
        ]:
            sub = bl0[bl0["model_type"] == model_type]
            if len(sub) == 0:
                continue
            ie_vals = sorted(sub["ie_synapses"].unique())
            means = []
            stds = []
            for ie in ie_vals:
                s = sub[sub["ie_synapses"] == ie]
                means.append(s["R_tot_mean"].mean())
                stds.append(s["R_tot_mean"].std())
            ax.errorbar(ie_vals, means, yerr=stds,
                        marker="o", markersize=3, lw=1.2, capsize=2,
                        color=color, label=label, capthick=0.5)

        ax.set_xlabel("IE synapses per branch")
        ax.set_ylabel(r"$R_{\mathrm{tot}}$ (input resistance)")
        ax.set_title(r"$R_{\mathrm{tot}}$ vs. inhibition")
        ax.legend(fontsize=5.5, loc="upper right")
        ax.set_ylim(0, 1.15)

        # Add annotation for mechanism
        ax.annotate("Shunting\nreduces $R_{\\mathrm{tot}}$\n4x",
                     xy=(10, 0.25), fontsize=5, color=COLOR_SHUNTING,
                     ha="center", style="italic")
    else:
        ax.text(0.5, 0.5, "R_tot data\nnot found", transform=ax.transAxes,
                ha="center", va="center")

    # ---- Panel B: Low-bandwidth degradation curve ----
    ax = axes[1]
    _panel(ax, "B")

    if lbw is not None:
        # Create a bandwidth label for ordering
        def bw_label(row):
            bw = row["broadcast_bandwidth"]
            bits = row.get("broadcast_bits", 8)
            if bw == "full":
                return "Full"
            elif bw == "quantized":
                return f"Q{int(bits)}b"
            elif bw == "sign_only":
                return "Sign"
            elif bw == "sparse_topk":
                return "Top-30%"
            return bw

        def bw_bits_equiv(row):
            """Map bandwidth modes to effective bits for x-axis ordering."""
            bw = row["broadcast_bandwidth"]
            bits = row.get("broadcast_bits", 8)
            if bw == "full":
                return 32
            elif bw == "quantized":
                return int(bits)
            elif bw == "sign_only":
                return 1
            elif bw == "sparse_topk":
                return 0.3  # separate category
            return 16

        lbw["bw_label"] = lbw.apply(bw_label, axis=1)
        lbw["bw_bits"] = lbw.apply(bw_bits_equiv, axis=1)

        # Group and compute stats (exclude sparse for line plot)
        quant_modes = lbw[lbw["broadcast_bandwidth"].isin(["full", "quantized", "sign_only"])]
        grp = quant_modes.groupby(["bw_label", "bw_bits"]).agg(
            mean=("test_accuracy", "mean"),
            std=("test_accuracy", "std"),
            n=("test_accuracy", "count"),
        ).reset_index().sort_values("bw_bits")

        # Plot line: bits vs accuracy
        ax.errorbar(grp["bw_bits"], grp["mean"] * 100, yerr=grp["std"] * 100,
                     marker="o", markersize=4, lw=1.5, capsize=2,
                     color=COLOR_SHUNTING, capthick=0.5, zorder=5)

        # Label each point
        for _, row in grp.iterrows():
            offset = 2.5 if row["bw_bits"] < 16 else -2.5
            va = "bottom" if row["bw_bits"] < 16 else "top"
            ax.annotate(row["bw_label"],
                        xy=(row["bw_bits"], row["mean"] * 100),
                        xytext=(0, offset), textcoords="offset points",
                        fontsize=5, ha="center", va=va, color=COLOR_SHUNTING)

        # Add sparse top-k as separate point
        sparse = lbw[lbw["broadcast_bandwidth"] == "sparse_topk"]
        if len(sparse):
            sp_mean = sparse["test_accuracy"].mean() * 100
            sp_std = sparse["test_accuracy"].std() * 100
            ax.errorbar([0.5], [sp_mean], yerr=[sp_std],
                        marker="^", markersize=5, color="#E67E22",
                        capsize=2, capthick=0.5, zorder=5)
            ax.annotate("Top-30%", xy=(0.5, sp_mean),
                        xytext=(8, -5), textcoords="offset points",
                        fontsize=5, color="#E67E22")

        ax.set_xlabel("Effective bits per neuron")
        ax.set_ylabel("Test accuracy (%)")
        ax.set_title("Broadcast bandwidth")
        ax.set_xscale("symlog", linthresh=1)
        ax.set_xticks([1, 2, 4, 8, 32])
        ax.set_xticklabels(["1", "2", "4", "8", "32"])
        ax.set_ylim(25, 72)

        # Add horizontal reference line for full
        full_mean = lbw[lbw["broadcast_bandwidth"] == "full"]["test_accuracy"].mean() * 100
        ax.axhline(full_mean, color=COLOR_SHUNTING, lw=0.5, ls=":", alpha=0.5)
    else:
        ax.text(0.5, 0.5, "Low-BW data\nnot found", transform=ax.transAxes,
                ha="center", va="center")

    # ---- Panel C: Additive + normalization control ----
    ax = axes[2]
    _panel(ax, "C")

    if anorm is not None:
        # Bar chart: 4 conditions
        conditions = []
        for (norm, strat), g in anorm.groupby(["use_additive_normalization", "strategy"]):
            m = g["test_accuracy"].mean() * 100
            s = g["test_accuracy"].std() * 100
            if strat == "local_ca":
                label = "Add.+norm\nlocal" if norm else "Add.\nlocal"
                conditions.append((label, m, s, COLOR_ADDITIVE if not norm else "#5B8AC4", strat, norm))

        # Sort: additive first, then additive+norm
        conditions.sort(key=lambda x: (x[5], x[4]))

        # Also add shunting reference from low-bandwidth full baseline
        shunting_local_ref = None
        if lbw is not None:
            full = lbw[lbw["broadcast_bandwidth"] == "full"]
            if len(full):
                shunting_local_ref = (full["test_accuracy"].mean() * 100,
                                      full["test_accuracy"].std() * 100)

        x_pos = np.arange(len(conditions))
        bars = ax.bar(x_pos, [c[1] for c in conditions],
                      yerr=[c[2] for c in conditions],
                      color=[c[3] for c in conditions],
                      edgecolor="white", lw=0.4, width=0.55,
                      capsize=2, error_kw={"lw": 0.5})

        # Value labels
        for i, (label, m, s, color, strat, norm) in enumerate(conditions):
            ax.text(i, m + s + 1.5, f"{m:.1f}%", ha="center", va="bottom",
                    fontsize=5, color="black")

        # Shunting reference line
        if shunting_local_ref:
            ax.axhline(shunting_local_ref[0], color=COLOR_SHUNTING, lw=1.0, ls="--",
                       alpha=0.7, zorder=0)
            ax.text(len(conditions) - 0.5, shunting_local_ref[0] + 1,
                    f"Shunting\n{shunting_local_ref[0]:.1f}%",
                    fontsize=5, color=COLOR_SHUNTING, ha="center", va="bottom")

        ax.set_xticks(x_pos)
        ax.set_xticklabels([c[0] for c in conditions], fontsize=5.5)
        ax.set_ylabel("Test accuracy (%)")
        ax.set_title("Normalization control")
        ax.set_ylim(30, 75)
    else:
        ax.text(0.5, 0.5, "Additive norm\ndata not found", transform=ax.transAxes,
                ha="center", va="center")

    fig.subplots_adjust(left=0.10, right=0.97, bottom=0.18, top=0.86,
                        wspace=0.55)
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
# Figure S6 — CIFAR-10 Results
# ===================================================================
def figure_s6():
    print("\n--- Figure S6: CIFAR-10 Results ---")

    cifar = _csv("cifar10_results.csv")
    if cifar is None:
        print("  SKIPPED: cifar10_results.csv not found")
        return

    fig, ax = plt.subplots(1, 1, figsize=(W * 0.55, 2.8))
    _panel(ax, "A", x=-0.15)

    conditions = [
        ("Shunt.\nBP", "dendritic_shunting", "standard", COLOR_SHUNTING),
        ("Add.\nBP", "dendritic_additive", "standard", COLOR_ADDITIVE),
        ("Shunt.\nlocal", "dendritic_shunting", "local_ca", COLOR_SHUNTING),
        ("Add.\nlocal", "dendritic_additive", "local_ca", COLOR_ADDITIVE),
    ]

    x_pos = np.arange(len(conditions))
    for i, (label, model, strat, color) in enumerate(conditions):
        sub = cifar[(cifar["model"] == model) & (cifar["strategy"] == strat)]
        if len(sub):
            m = sub["test_accuracy"].mean() * 100
            s = sub["test_accuracy"].std() * 100
            alpha = 1.0 if strat == "local_ca" else 0.6
            ax.bar(i, m, yerr=s, color=color, alpha=alpha,
                   edgecolor="white", lw=0.3, width=0.55,
                   capsize=2, error_kw={"lw": 0.5})
            ax.text(i, m + s + 0.8, f"{m:.1f}%", ha="center", va="bottom",
                    fontsize=5.5)

    ax.set_xticks(x_pos)
    ax.set_xticklabels([c[0] for c in conditions], fontsize=6)
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title("CIFAR-10 (flattened)")
    ax.set_ylim(0, 50)

    # Add chance level
    ax.axhline(10, color="gray", lw=0.5, ls=":", alpha=0.5)
    ax.text(3.3, 10.5, "chance", fontsize=5, color="gray", va="bottom")

    fig.subplots_adjust(left=0.18, right=0.92, bottom=0.15, top=0.88)
    _save(fig, "fig_s6_cifar10")
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
    plt.close(fig)


# ===================================================================
# Main
# ===================================================================
def main():
    _setup_style()
    os.makedirs(FIGURES_DIR, exist_ok=True)

    figure5()
    figure_s5()
    figure_s6()
    figure_s7()

    print("\nDone! All revision figures generated.")


if __name__ == "__main__":
    main()
