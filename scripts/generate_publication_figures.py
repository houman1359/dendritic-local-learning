#!/usr/bin/env python3
"""
Generate publication-quality figures for NeurIPS 2026 dendritic local learning paper.

Produces:
  - Figure 2: fig_pub_competence_regime.pdf   (2 panels, double-column)
  - Figure 3: fig_pub_gradient_fidelity.pdf   (2 panels, double-column)
  - Appendix: fig_pub_appendix_combined.pdf   (6 panels, 2x3, double-column)

Usage:
    cd /n/holylabs/LABS/kempner_dev/Users/hsafaai/Code/dendritic-modeling
    PYTHONPATH=src:$PYTHONPATH python drafts/dendritic-local-learning/scripts/generate_publication_figures.py
"""

import os
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BUNDLE = (
    "/n/holylfs06/LABS/kempner_project_b/Lab/dendritic/HS/LOCAL_LEARNING"
    "/analysis/publication_bundle_faircheck_plus_20260225"
)
LOCAL_MISMATCH_CSV = (
    "/n/holylfs06/LABS/kempner_project_b/Lab/dendritic/HS/LOCAL_LEARNING"
    "/analysis/local_mismatch_recheck_20260224_summary.csv"
)
FIGURES_DIR = (
    "/n/holylabs/LABS/kempner_dev/Users/hsafaai/Code/dendritic-modeling"
    "/drafts/dendritic-local-learning/figures"
)

# ---------------------------------------------------------------------------
# Style constants
# ---------------------------------------------------------------------------
COLOR_SHUNTING = "#18864B"
COLOR_ADDITIVE = "#2D5DA8"
COLOR_BACKPROP = "#666666"
COLOR_POINT_MLP = "#888888"

NEURIPS_SINGLE_COL = 5.5   # inches
NEURIPS_DOUBLE_COL = 11.0  # inches
DPI = 300
FONT_SIZE_MIN = 8
FONT_SIZE_LABEL = 10
FONT_SIZE_PANEL = 12

LABEL_MAP = {
    "dendritic_shunting": "Shunting",
    "dendritic_additive": "Additive",
    "dendritic_mlp": "Dendritic MLP",
    "point_mlp": "Point MLP",
}

DATASET_LABEL = {
    "mnist": "MNIST",
    "context_gating": "Context Gating",
    "cifar10": "CIFAR-10",
    "info_shunting": "Info Shunting",
    "noise_resilience": "Noise Resilience",
    "orthonet": "OrthoNet",
}


def _setup_style():
    """Apply publication-quality matplotlib defaults."""
    plt.rcParams.update({
        "font.size": FONT_SIZE_MIN,
        "axes.labelsize": FONT_SIZE_LABEL,
        "axes.titlesize": FONT_SIZE_LABEL,
        "xtick.labelsize": FONT_SIZE_MIN,
        "ytick.labelsize": FONT_SIZE_MIN,
        "legend.fontsize": FONT_SIZE_MIN,
        "figure.dpi": DPI,
        "savefig.dpi": DPI,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "pdf.fonttype": 42,       # TrueType for editable text in PDFs
        "ps.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
    })


def _panel_label(ax, label, x=-0.12, y=1.08):
    """Add bold panel label (A), (B), ... to an axes."""
    ax.text(
        x, y, f"({label})",
        transform=ax.transAxes,
        fontsize=FONT_SIZE_PANEL,
        fontweight="bold",
        va="top",
        ha="left",
    )


def _save_fig(fig, name):
    """Save figure as both PDF and PNG."""
    os.makedirs(FIGURES_DIR, exist_ok=True)
    for ext in ("pdf", "png"):
        path = os.path.join(FIGURES_DIR, f"{name}.{ext}")
        fig.savefig(path, dpi=DPI)
        print(f"  Saved: {path}")


def _load_csv(filename, bundle=True):
    """Load a CSV from BUNDLE directory or from the given path."""
    if bundle:
        path = os.path.join(BUNDLE, filename)
    else:
        path = filename
    if not os.path.isfile(path):
        warnings.warn(f"CSV not found: {path}")
        return None
    return pd.read_csv(path)


# ===================================================================
# Figure 2 -- Competence regime
# ===================================================================
def figure2_competence_regime():
    """
    Figure 2: fig_pub_competence_regime.pdf (double-column, 2 panels)

    Panel A: Backprop ceiling vs local learning best (bar chart).
    Panel B: Shunting-minus-additive advantage from claimA (grouped bar).
    """
    print("\n--- Figure 2: Competence Regime ---")

    phase1 = _load_csv("phase1_best_standard.csv")
    core = _load_csv("core_fair_tuning.csv")
    claim_a = _load_csv("claimA_shunting_regime.csv")

    if phase1 is None or core is None or claim_a is None:
        print("  SKIPPED (missing data)")
        return

    fig, axes = plt.subplots(1, 2, figsize=(NEURIPS_DOUBLE_COL, 3.2))

    # ------ Panel A: Backprop ceiling vs local learning best ------
    ax = axes[0]
    _panel_label(ax, "A")

    # Backprop ceilings from phase1_best_standard (standard training)
    bp_data = {}
    for ds in ["mnist", "context_gating"]:
        row = phase1[
            (phase1["dataset"] == ds)
            & (phase1["network_type"] == "dendritic_shunting")
        ]
        if len(row) > 0:
            bp_data[ds] = row.iloc[0]["test_accuracy"]

    # Local learning best from core_fair_tuning: 5f, per_soma, local decoder
    local_data = {}
    for ds in ["mnist"]:
        for nt in ["dendritic_shunting", "dendritic_additive"]:
            sub = core[
                (core["dataset"] == ds)
                & (core["network_type"] == nt)
                & (core["rule_variant"] == "5f")
                & (core["error_broadcast_mode"] == "per_soma")
                & (core["decoder_update_mode"] == "local")
            ]
            if len(sub) > 0:
                row = sub.iloc[0]
                local_data[(ds, nt)] = (
                    row["test_accuracy_mean"],
                    row["test_accuracy_std"],
                )

    # Build bar groups
    # Group 1: MNIST -- backprop ceiling, shunting local, additive local
    # Group 2: Context Gating -- backprop ceiling only (no local data for CG)
    bar_labels = []
    bar_vals = []
    bar_errs = []
    bar_colors = []

    # MNIST group
    if "mnist" in bp_data:
        bar_labels.append("MNIST\nBackprop")
        bar_vals.append(bp_data["mnist"] * 100)
        bar_errs.append(0)
        bar_colors.append(COLOR_BACKPROP)

    if ("mnist", "dendritic_shunting") in local_data:
        m, s = local_data[("mnist", "dendritic_shunting")]
        bar_labels.append("MNIST\nShunting\n(local)")
        bar_vals.append(m * 100)
        bar_errs.append(s * 100)
        bar_colors.append(COLOR_SHUNTING)

    if ("mnist", "dendritic_additive") in local_data:
        m, s = local_data[("mnist", "dendritic_additive")]
        bar_labels.append("MNIST\nAdditive\n(local)")
        bar_vals.append(m * 100)
        bar_errs.append(s * 100)
        bar_colors.append(COLOR_ADDITIVE)

    # Context Gating group
    if "context_gating" in bp_data:
        bar_labels.append("CG\nBackprop")
        bar_vals.append(bp_data["context_gating"] * 100)
        bar_errs.append(0)
        bar_colors.append(COLOR_BACKPROP)

    x_pos = np.arange(len(bar_labels))
    bars = ax.bar(
        x_pos,
        bar_vals,
        yerr=bar_errs,
        color=bar_colors,
        edgecolor="white",
        linewidth=0.5,
        capsize=3,
        width=0.65,
        error_kw={"linewidth": 1.0},
    )
    ax.set_xticks(x_pos)
    ax.set_xticklabels(bar_labels, fontsize=FONT_SIZE_MIN)
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title("Backprop ceiling vs. local learning")

    # Add value labels on bars
    for bar_rect, val in zip(bars, bar_vals):
        ax.text(
            bar_rect.get_x() + bar_rect.get_width() / 2,
            bar_rect.get_height() + 0.8,
            f"{val:.1f}",
            ha="center",
            va="bottom",
            fontsize=7,
        )

    # Set y-axis range to zoom into relevant region
    y_min = max(0, min(bar_vals) - 8)
    ax.set_ylim(y_min, max(bar_vals) + 4)

    # ------ Panel B: Shunting advantage from claimA ------
    ax = axes[1]
    _panel_label(ax, "B")

    # Pivot: for each (dataset, ie_value), compute shunting - additive
    shunting = claim_a[claim_a["network_type"] == "dendritic_shunting"].copy()
    additive = claim_a[claim_a["network_type"] == "dendritic_additive"].copy()

    # Merge on dataset + ie_value
    merged = pd.merge(
        shunting,
        additive,
        on=["dataset", "ie_value"],
        suffixes=("_shunt", "_add"),
    )
    merged["delta"] = (
        merged["test_accuracy_mean_shunt"] - merged["test_accuracy_mean_add"]
    )
    merged["delta_pct"] = merged["delta"] * 100

    # Grouped bar chart: group by dataset, bars for each ie_value
    datasets = sorted(merged["dataset"].unique())
    ie_vals = sorted(merged["ie_value"].unique())
    # Only keep ie_vals that appear in both datasets
    ie_vals_common = [
        iv for iv in ie_vals
        if all(
            len(merged[(merged["dataset"] == ds) & (merged["ie_value"] == iv)]) > 0
            for ds in datasets
        )
    ]

    n_datasets = len(datasets)
    n_ie = len(ie_vals_common) if ie_vals_common else len(ie_vals)
    use_ie = ie_vals_common if ie_vals_common else ie_vals

    bar_width = 0.35
    x_base = np.arange(len(use_ie))

    for i, ds in enumerate(datasets):
        ds_data = merged[merged["dataset"] == ds]
        vals = []
        for iv in use_ie:
            row = ds_data[ds_data["ie_value"] == iv]
            if len(row) > 0:
                vals.append(row.iloc[0]["delta_pct"])
            else:
                vals.append(0)
        offset = (i - (n_datasets - 1) / 2) * bar_width
        color = COLOR_SHUNTING if "noise" in ds else "#2ca02c"
        ax.bar(
            x_base + offset,
            vals,
            bar_width * 0.9,
            label=DATASET_LABEL.get(ds, ds),
            color=color,
            alpha=0.7 + 0.3 * (i / max(1, n_datasets - 1)),
            edgecolor="white",
            linewidth=0.5,
        )

    ax.set_xticks(x_base)
    ax.set_xticklabels([str(iv) for iv in use_ie], fontsize=FONT_SIZE_MIN)
    ax.set_xlabel("I/E synapses per branch")
    ax.set_ylabel("Shunting advantage (pp)")
    ax.set_title("Shunting vs. additive accuracy gap")
    ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
    ax.legend(loc="best", fontsize=FONT_SIZE_MIN)

    fig.tight_layout(w_pad=3.0)
    _save_fig(fig, "fig_pub_competence_regime")
    plt.close(fig)


# ===================================================================
# Figure 3 -- Gradient fidelity
# ===================================================================
def figure3_gradient_fidelity():
    """
    Figure 3: fig_pub_gradient_fidelity.pdf (double-column, 2 panels)

    Panel A: Cosine similarity bars (hard-coded Table 3 values).
    Panel B: Scale mismatch bars (log scale y-axis).
    """
    print("\n--- Figure 3: Gradient Fidelity ---")

    # Hard-coded from Table 3
    conditions = [
        ("MNIST\nShunting", 0.202, 0.117, COLOR_SHUNTING),
        ("MNIST\nAdditive", 0.006, 1.053, COLOR_ADDITIVE),
        ("CG\nShunting", 0.108, 0.036, COLOR_SHUNTING),
        ("CG\nAdditive", -0.007, 2.154, COLOR_ADDITIVE),
    ]

    labels = [c[0] for c in conditions]
    cosines = [c[1] for c in conditions]
    mismatches = [c[2] for c in conditions]
    colors = [c[3] for c in conditions]

    fig, axes = plt.subplots(1, 2, figsize=(NEURIPS_DOUBLE_COL, 3.2))

    # ------ Panel A: Cosine similarity ------
    ax = axes[0]
    _panel_label(ax, "A")

    x_pos = np.arange(len(labels))
    bars = ax.bar(
        x_pos,
        cosines,
        color=colors,
        edgecolor="white",
        linewidth=0.5,
        width=0.6,
    )
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, fontsize=FONT_SIZE_MIN)
    ax.set_ylabel("Cosine similarity\n(local vs. backprop gradient)")
    ax.set_title("Gradient alignment")
    ax.axhline(0, color="black", linewidth=0.5, linestyle="--")

    # Value labels
    for bar_rect, val in zip(bars, cosines):
        y_offset = 0.005 if val >= 0 else -0.015
        va = "bottom" if val >= 0 else "top"
        ax.text(
            bar_rect.get_x() + bar_rect.get_width() / 2,
            val + y_offset,
            f"{val:.3f}",
            ha="center",
            va=va,
            fontsize=7,
        )

    # ------ Panel B: Scale mismatch (log scale) ------
    ax = axes[1]
    _panel_label(ax, "B")

    bars = ax.bar(
        x_pos,
        mismatches,
        color=colors,
        edgecolor="white",
        linewidth=0.5,
        width=0.6,
    )
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, fontsize=FONT_SIZE_MIN)
    ax.set_ylabel("Scale mismatch\n(||local|| / ||backprop||)")
    ax.set_title("Gradient scale mismatch")
    ax.set_yscale("log")
    ax.axhline(1.0, color="black", linewidth=0.5, linestyle="--", label="Ideal (1.0)")
    ax.legend(loc="upper left", fontsize=7)

    # Value labels
    for bar_rect, val in zip(bars, mismatches):
        ax.text(
            bar_rect.get_x() + bar_rect.get_width() / 2,
            val * 1.15,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=7,
        )

    fig.tight_layout(w_pad=3.0)
    _save_fig(fig, "fig_pub_gradient_fidelity")
    plt.close(fig)


# ===================================================================
# Appendix figure -- combined 2x3
# ===================================================================
def figure_appendix_combined():
    """
    Appendix figure: fig_pub_appendix_combined.pdf (double-column, 2x3 = 6 panels)

    Panel A: Phase 1 capacity bars (all architectures x datasets)
    Panel B: Local mismatch recheck bars
    Panel C: Depth scaling curves
    Panel D: Noise robustness curves
    Panel E: Rule family ranking (3F vs 4F vs 5F)
    Panel F: Decoder locality comparison
    """
    print("\n--- Appendix Figure: Combined ---")

    phase1 = _load_csv("phase1_best_standard.csv")
    mismatch = _load_csv(LOCAL_MISMATCH_CSV, bundle=False)
    depth = _load_csv("depth_scaling.csv")
    noise = _load_csv("noise_robustness.csv")
    core = _load_csv("core_fair_tuning.csv")

    fig, axes = plt.subplots(2, 3, figsize=(NEURIPS_DOUBLE_COL, 6.4))
    panel_labels = ["A", "B", "C", "D", "E", "F"]
    for i, lbl in enumerate(panel_labels):
        _panel_label(axes.flat[i], lbl)

    # ------ Panel A: Phase 1 capacity bars ------
    ax = axes[0, 0]
    if phase1 is not None:
        # Filter out orthonet (all NaN) and prepare
        p1 = phase1.dropna(subset=["test_accuracy"]).copy()
        datasets_order = [
            d for d in ["mnist", "context_gating", "cifar10", "info_shunting"]
            if d in p1["dataset"].values
        ]
        arch_order = ["dendritic_shunting", "dendritic_additive", "dendritic_mlp", "point_mlp"]
        arch_colors = {
            "dendritic_shunting": COLOR_SHUNTING,
            "dendritic_additive": COLOR_ADDITIVE,
            "dendritic_mlp": "#D4A017",
            "point_mlp": COLOR_POINT_MLP,
        }

        n_ds = len(datasets_order)
        n_arch = len(arch_order)
        bar_width = 0.8 / n_arch
        x_base = np.arange(n_ds)

        for j, arch in enumerate(arch_order):
            vals = []
            for ds in datasets_order:
                row = p1[(p1["dataset"] == ds) & (p1["network_type"] == arch)]
                if len(row) > 0:
                    vals.append(row.iloc[0]["test_accuracy"] * 100)
                else:
                    vals.append(0)
            offset = (j - (n_arch - 1) / 2) * bar_width
            ax.bar(
                x_base + offset,
                vals,
                bar_width * 0.9,
                label=LABEL_MAP.get(arch, arch),
                color=arch_colors.get(arch, "#999999"),
                edgecolor="white",
                linewidth=0.3,
            )

        ax.set_xticks(x_base)
        ax.set_xticklabels(
            [DATASET_LABEL.get(d, d) for d in datasets_order],
            fontsize=7,
            rotation=15,
            ha="right",
        )
        ax.set_ylabel("Test accuracy (%)")
        ax.set_title("Phase 1: Standard training ceiling")
        ax.legend(fontsize=6, loc="lower left", ncol=2)
    else:
        ax.text(0.5, 0.5, "Data not found", transform=ax.transAxes, ha="center")

    # ------ Panel B: Local mismatch recheck ------
    ax = axes[0, 1]
    if mismatch is not None:
        # Aggregate per (core_type, error_broadcast_mode, decoder_update_mode)
        agg = mismatch.groupby(
            ["core_type", "error_broadcast_mode", "decoder_update_mode"]
        ).agg(
            test_mean=("test_acc", "mean"),
            test_sem=("test_acc", lambda x: x.std() / np.sqrt(len(x))),
        ).reset_index()

        # Sort for display
        conditions_list = []
        for _, row in agg.iterrows():
            short_eb = "per_soma" if row["error_broadcast_mode"] == "per_soma" else "local_mm"
            short_du = "local" if row["decoder_update_mode"] == "local" else "bp"
            short_ct = "Shunt" if "shunting" in row["core_type"] else "Add"
            label = f"{short_ct}\n{short_eb}\n{short_du}"
            color = COLOR_SHUNTING if "shunting" in row["core_type"] else COLOR_ADDITIVE
            conditions_list.append((label, row["test_mean"] * 100, row["test_sem"] * 100, color))

        # Sort by accuracy descending
        conditions_list.sort(key=lambda c: c[1], reverse=True)

        x_pos = np.arange(len(conditions_list))
        ax.bar(
            x_pos,
            [c[1] for c in conditions_list],
            yerr=[c[2] for c in conditions_list],
            color=[c[3] for c in conditions_list],
            edgecolor="white",
            linewidth=0.3,
            capsize=2,
            width=0.65,
            error_kw={"linewidth": 0.8},
        )
        ax.set_xticks(x_pos)
        ax.set_xticklabels([c[0] for c in conditions_list], fontsize=6)
        ax.set_ylabel("Test accuracy (%)")
        ax.set_title("Local mismatch recheck")
    else:
        ax.text(0.5, 0.5, "Data not found", transform=ax.transAxes, ha="center")

    # ------ Panel C: Depth scaling ------
    ax = axes[0, 2]
    if depth is not None:
        for nt in ["dendritic_shunting", "dendritic_additive"]:
            sub = depth[depth["network_type"] == nt].copy()
            if len(sub) == 0:
                continue

            # Parse branch_factors to get depth (number of elements)
            def _parse_depth(bf_str):
                try:
                    items = bf_str.strip("[]").split(",")
                    return len(items)
                except Exception:
                    return 1

            sub = sub.copy()
            sub["depth"] = sub["branch_factors"].apply(_parse_depth)
            sub = sub.sort_values("depth")

            color = COLOR_SHUNTING if "shunting" in nt else COLOR_ADDITIVE
            ax.errorbar(
                sub["depth"],
                sub["test_accuracy_mean"] * 100,
                yerr=sub["test_accuracy_std"] * 100,
                marker="o",
                markersize=4,
                linewidth=1.5,
                capsize=3,
                label=LABEL_MAP.get(nt, nt),
                color=color,
            )

        ax.set_xlabel("Network depth (layers)")
        ax.set_ylabel("Test accuracy (%)")
        ax.set_title("Depth scaling")
        ax.legend(fontsize=7)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    else:
        ax.text(0.5, 0.5, "Data not found", transform=ax.transAxes, ha="center")

    # ------ Panel D: Noise robustness ------
    ax = axes[1, 0]
    if noise is not None:
        for nt in ["dendritic_shunting", "dendritic_additive"]:
            sub = noise[noise["network_type"] == nt].copy()
            if len(sub) == 0:
                continue
            sub = sub.sort_values("error_noise_sigma")
            color = COLOR_SHUNTING if "shunting" in nt else COLOR_ADDITIVE
            ax.errorbar(
                sub["error_noise_sigma"],
                sub["test_accuracy_mean"] * 100,
                yerr=sub["test_accuracy_std"] * 100,
                marker="o",
                markersize=4,
                linewidth=1.5,
                capsize=3,
                label=LABEL_MAP.get(nt, nt),
                color=color,
            )

        ax.set_xlabel("Error noise sigma")
        ax.set_ylabel("Test accuracy (%)")
        ax.set_title("Noise robustness")
        ax.legend(fontsize=7)
    else:
        ax.text(0.5, 0.5, "Data not found", transform=ax.transAxes, ha="center")

    # ------ Panel E: Rule family ranking (3F vs 4F vs 5F) ------
    ax = axes[1, 1]
    if core is not None:
        # Filter: mnist, per_soma, local decoder
        sub = core[
            (core["dataset"] == "mnist")
            & (core["error_broadcast_mode"] == "per_soma")
            & (core["decoder_update_mode"] == "local")
        ].copy()

        rule_order = ["3f", "4f", "5f"]
        net_types = ["dendritic_shunting", "dendritic_additive"]
        n_rules = len(rule_order)
        n_nets = len(net_types)
        bar_width = 0.35
        x_base = np.arange(n_rules)

        for j, nt in enumerate(net_types):
            vals = []
            errs = []
            for rv in rule_order:
                row = sub[(sub["rule_variant"] == rv) & (sub["network_type"] == nt)]
                if len(row) > 0:
                    r = row.iloc[0]
                    vals.append(r["test_accuracy_mean"] * 100)
                    errs.append(r["test_accuracy_std"] * 100)
                else:
                    vals.append(0)
                    errs.append(0)
            offset = (j - (n_nets - 1) / 2) * bar_width
            color = COLOR_SHUNTING if "shunting" in nt else COLOR_ADDITIVE
            ax.bar(
                x_base + offset,
                vals,
                bar_width * 0.9,
                yerr=errs,
                label=LABEL_MAP.get(nt, nt),
                color=color,
                edgecolor="white",
                linewidth=0.3,
                capsize=2,
                error_kw={"linewidth": 0.8},
            )

        ax.set_xticks(x_base)
        ax.set_xticklabels([rv.upper() for rv in rule_order])
        ax.set_ylabel("Test accuracy (%)")
        ax.set_title("Rule family ranking (MNIST)")
        ax.legend(fontsize=7)

        # Zoom y-axis
        all_vals = sub["test_accuracy_mean"].dropna() * 100
        if len(all_vals) > 0:
            ax.set_ylim(max(0, all_vals.min() - 5), all_vals.max() + 3)
    else:
        ax.text(0.5, 0.5, "Data not found", transform=ax.transAxes, ha="center")

    # ------ Panel F: Decoder locality comparison ------
    ax = axes[1, 2]
    if core is not None:
        # Filter: mnist, per_soma, 5f
        sub = core[
            (core["dataset"] == "mnist")
            & (core["error_broadcast_mode"] == "per_soma")
            & (core["rule_variant"] == "5f")
        ].copy()

        decoder_modes = ["local", "backprop"]
        net_types = ["dendritic_shunting", "dendritic_additive"]
        n_dm = len(decoder_modes)
        n_nets = len(net_types)
        bar_width = 0.35
        x_base = np.arange(n_dm)

        for j, nt in enumerate(net_types):
            vals = []
            errs = []
            for dm in decoder_modes:
                row = sub[
                    (sub["decoder_update_mode"] == dm) & (sub["network_type"] == nt)
                ]
                if len(row) > 0:
                    r = row.iloc[0]
                    vals.append(r["test_accuracy_mean"] * 100)
                    errs.append(r["test_accuracy_std"] * 100)
                else:
                    vals.append(0)
                    errs.append(0)
            offset = (j - (n_nets - 1) / 2) * bar_width
            color = COLOR_SHUNTING if "shunting" in nt else COLOR_ADDITIVE
            ax.bar(
                x_base + offset,
                vals,
                bar_width * 0.9,
                yerr=errs,
                label=LABEL_MAP.get(nt, nt),
                color=color,
                edgecolor="white",
                linewidth=0.3,
                capsize=2,
                error_kw={"linewidth": 0.8},
            )

        ax.set_xticks(x_base)
        ax.set_xticklabels(["Local decoder", "Backprop decoder"])
        ax.set_ylabel("Test accuracy (%)")
        ax.set_title("Decoder locality (5F, MNIST)")
        ax.legend(fontsize=7)

        # Zoom y-axis
        all_vals = sub["test_accuracy_mean"].dropna() * 100
        if len(all_vals) > 0:
            ax.set_ylim(max(0, all_vals.min() - 3), all_vals.max() + 2)
    else:
        ax.text(0.5, 0.5, "Data not found", transform=ax.transAxes, ha="center")

    fig.tight_layout(h_pad=3.0, w_pad=2.5)
    _save_fig(fig, "fig_pub_appendix_combined")
    plt.close(fig)


# ===================================================================
# Main
# ===================================================================
def main():
    _setup_style()
    print(f"Bundle dir: {BUNDLE}")
    print(f"Figures dir: {FIGURES_DIR}")

    figure2_competence_regime()
    figure3_gradient_fidelity()
    figure_appendix_combined()

    print("\nDone.")


if __name__ == "__main__":
    main()
