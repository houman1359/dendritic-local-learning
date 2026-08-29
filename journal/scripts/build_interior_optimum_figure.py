#!/usr/bin/env python3
"""Render the interior-optimum validation figure.

Panels (data from source_data/interior_optimum, produced by
scripts/analyze_interior_optimum.py):
A - mean final population loss versus routed depth D per task depth H
    (depth phase, aligned tree), argmin marked;
B - mean initialization utility U versus D per H, argmax marked;
C - per-pair predicted (argmax U) versus observed (argmin loss) best depth,
    counts over 200 seed x task-depth pairs;
D - factorial ancestry-minus-best-control contrasts across budget K in
    held-out accuracy and in initialization utility (best of the four
    published controls and best of the three implementable, non-oracle
    controls).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from journal_style import (
    COLORS,
    FIG_W,
    LW_DATA,
    LW_REF,
    MARKERS,
    PT_ANNOT,
    PT_LEGEND,
    SEQ_CMAP,
    add_colorbar,
    annotate_heatmap,
    apply_neurips_style,
    audit_layout,
    audit_text_over_data,
    panel_title,
    style_axis,
)

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "source_data" / "interior_optimum"
FIGURES = ROOT / "figures" / "generated"

# Task depth H is an ORDINAL series, so it takes the manuscript slate ramp.
# It previously wore four reserved condition hues (additive blue, routing
# green, shared amber, backprop red-brown), which put the whole reserved
# vocabulary on arbitrary depth levels.
H_COLORS = {
    1: "#9AA5B4",
    2: "#7C8899",
    3: "#5F6B7E",
    4: "#2E3947",
}


def main() -> None:
    apply_neurips_style()
    curves = pd.read_csv(SOURCE / "depth_curves.csv")
    agreement = pd.read_csv(SOURCE / "depth_agreement_seed.csv")
    per_k = pd.read_csv(SOURCE / "factorial_contrast_summary.csv")

    fig, axes = plt.subplots(
        # Author at the full 518.4 pt canvas: at 0.72 x FIG_W the sheet
        # was upscaled ~1.39x by \includegraphics[width=\textwidth],
        # so every glyph and stroke printed off-token.
        2, 2, figsize=(FIG_W, FIG_W * 0.78),
        gridspec_kw={"wspace": 0.46, "hspace": 0.62},
    )
    (ax_a, ax_b), (ax_c, ax_d) = axes

    # A/B - depth-phase loss and utility curves.
    for i, (h, grp) in enumerate(curves.groupby("task_depth")):
        grp = grp.sort_values("model_depth")
        color = H_COLORS[int(h)]
        marker = MARKERS[i % len(MARKERS)]
        ax_a.plot(
            grp["model_depth"],
            grp["mean_final_population_loss"],
            color=color,
            marker=marker,
            ms=3.4,
            lw=LW_DATA,
            label=f"$H={int(h)}$",
        )
        best = grp.loc[grp["mean_final_population_loss"].idxmin()]
        ax_a.plot(
            best["model_depth"],
            best["mean_final_population_loss"],
            marker="o",
            ms=8.5,
            mfc="none",
            mec=color,
            mew=1.0,
        )
        ax_b.plot(
            grp["model_depth"],
            grp["mean_utility"],
            color=color,
            marker=marker,
            ms=3.4,
            lw=LW_DATA,
        )
        top = grp.loc[grp["mean_utility"].idxmax()]
        ax_b.plot(
            top["model_depth"],
            top["mean_utility"],
            marker="o",
            ms=8.5,
            mfc="none",
            mec=color,
            mew=1.0,
        )
    ax_a.set_xlabel("routed depth $D$")
    ax_a.set_ylabel("mean final population loss")
    ax_a.set_xticks([1, 2, 3, 4])
    panel_title(ax_a, "A", "Trained loss across depth")
    style_axis(ax_a)
    ax_a.legend(frameon=False, fontsize=PT_LEGEND - 0.4, handlelength=1.3,
                loc="upper left", borderaxespad=0.2)

    ax_b.set_xlabel("routed depth $D$")
    ax_b.set_ylabel("initialization utility $U$")
    ax_b.set_xticks([1, 2, 3, 4])
    panel_title(ax_b, "B", "Bound utility across depth")
    style_axis(ax_b)

    # C - predicted vs observed best-depth counts.
    counts = np.zeros((4, 4))
    for _, row in agreement.iterrows():
        counts[int(row["predicted_best_depth"]) - 1, int(row["observed_best_depth"]) - 1] += 1
    im = ax_c.imshow(counts, cmap=SEQ_CMAP, origin="lower", aspect="equal")
    annotate_heatmap(ax_c, im, counts, fmt="{:.0f}")
    ax_c.set_xticks(range(4), [str(d) for d in range(1, 5)])
    ax_c.set_yticks(range(4), [str(d) for d in range(1, 5)])
    ax_c.set_xlabel("observed best $D$ (argmin loss)")
    ax_c.set_ylabel("predicted best $D$ (argmax $U$)")
    panel_title(ax_c, "C", "Optimum location per pair")
    add_colorbar(fig, ax_c, im, label="seed--task pairs")

    # D - factorial contrasts across budget K.
    per_k = per_k.sort_values("budget_k")
    ks = per_k["budget_k"].to_numpy()
    x = np.arange(len(ks))
    series = [
        (
            "mean_accuracy_contrast",
            ("accuracy_ci_low", "accuracy_ci_high"),
            # These three are endpoint metrics, not the backprop and oracle
            # conditions whose hues they used to borrow.
            COLORS["ink"],
            "o",
            "held-out accuracy",
        ),
        (
            "mean_utility_contrast",
            ("utility_ci_low", "utility_ci_high"),
            COLORS["per_soma"],
            "s",
            "utility $U$ (vs best of four)",
        ),
        (
            "mean_utility_contrast_non_oracle",
            ("utility_non_oracle_ci_low", "utility_non_oracle_ci_high"),
            COLORS["low_rank"],
            "^",
            "utility $U$ (vs non-oracle)",
        ),
    ]
    for col, (lo, hi), color, marker, label in series:
        y = per_k[col].to_numpy()
        yerr = np.vstack([y - per_k[lo].to_numpy(), per_k[hi].to_numpy() - y])
        ax_d.errorbar(
            x,
            y,
            yerr=yerr,
            color=color,
            marker=marker,
            ms=3.6,
            lw=LW_DATA,
            elinewidth=LW_REF,
            capsize=1.6,
            label=label,
        )
    ax_d.axhline(0, color=COLORS["mute"], ls=":", lw=LW_REF)
    ax_d.set_ylim(-0.56, 0.12)
    peak = per_k.loc[per_k["mean_accuracy_contrast"].idxmax()]
    ax_d.annotate(
        "trained peak",
        xy=(list(ks).index(int(peak["budget_k"])), peak["mean_accuracy_contrast"]),
        xytext=(0.9, 0.075),
        fontsize=PT_ANNOT,
        color=COLORS["bp"],
        arrowprops={"arrowstyle": "-", "lw": LW_REF, "color": COLORS["bp"]},
    )
    ax_d.set_xticks(x, [str(int(k)) for k in ks])
    ax_d.set_xlabel("feedback budget $K$")
    ax_d.set_ylabel("ancestry $-$ best control")
    panel_title(ax_d, "D", "Trained versus one-step contrast")
    style_axis(ax_d)
    handles, labels = ax_d.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center",
               bbox_to_anchor=(0.5, 0.005), ncol=3, frameon=False,
               fontsize=PT_LEGEND - 0.4, handlelength=1.3, columnspacing=1.0)

    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.subplots_adjust(left=0.10, right=0.90, top=0.92, bottom=0.145)
    fig.canvas.draw()
    audit_layout(fig, "fig_interior_optimum")
    audit_text_over_data(fig, "fig_interior_optimum")
    fig.savefig(FIGURES / "fig_interior_optimum.pdf",
                metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(FIGURES / "fig_interior_optimum.png", dpi=600)
    plt.close(fig)


if __name__ == "__main__":
    main()
