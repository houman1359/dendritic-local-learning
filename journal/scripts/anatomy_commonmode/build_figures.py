#!/usr/bin/env python3
"""Render common-mode analysis tables with the shared journal visual style."""
from pathlib import Path
import hashlib
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
JOURNAL = HERE.parents[1]
sys.path.insert(0, str(JOURNAL / "scripts"))
from journal_style import (COLORS, LW_DATA, LW_ERR, MARKER_MS, PANEL_LABEL_PT,
                           PT_SMALL, apply_neurips_style, style_axis)

OUT = JOURNAL / "source_data/anatomy_commonmode"
METHODS = ["common + ancestry", "common + random routes", "common + depth bins",
           "common + shuffled routes", "common + surrogate ancestry", "common-constrained SVD"]
LABELS = ["Ancestry", "Random routes", "Depth bins", "Shuffled routes", "Surrogate ancestry", "SVD oracle"]
HUES = [COLORS[k] for k in ["shunting", "local", "additive", "point_mlp", "bp", "oracle"]]
COHORTS = ["original8", "v661", "pinky"]
COHORT_LABELS = ["Original mouse: development", "Same mouse: disjoint cells", "Second mouse: Pinky"]


def axis(ax):
    style_axis(ax, grid="y")
    ax.spines[["top", "right"]].set_visible(False)


def main():
    apply_neurips_style()
    out = OUT / "figures"
    out.mkdir(exist_ok=True)
    tables = {c: pd.read_csv(OUT / c / "cell_method_summary.csv") for c in COHORTS}
    reports = {c: json.loads((OUT / c / "summary.json").read_text()) for c in COHORTS}
    fig = plt.figure(figsize=(7.2, 5.6))
    gs = fig.add_gridspec(2, 3, left=.085, right=.985, bottom=.13, top=.79, hspace=.45, wspace=.40)
    for col, cohort in enumerate(COHORTS):
        table = tables[cohort]
        for row, metric in enumerate(["total_capture", "residual_capture"]):
            ax = fig.add_subplot(gs[row, col])
            for method, color, marker in zip(METHODS, HUES, ["o", "s", "^", "D", "v", "P"]):
                part = table[table.method.eq(method)].groupby("channels")[metric].mean()
                ax.plot(part.index, part.values, color=color, marker=marker, ms=3.5,
                        lw=LW_DATA, markeredgecolor="white", markeredgewidth=.45)
            ax.set_xscale("log", base=2)
            ticks = sorted(table.channels.unique())
            ax.set_xticks(ticks, [str(k) for k in ticks])
            ax.set_xlim(.85, max(ticks)*1.18)
            ax.set_ylim(-.02, 1.02)
            ax.set_yticks([0, .25, .5, .75, 1.0])
            ax.set_xlabel("Total channels, K")
            if col == 0:
                ax.set_ylabel("Total energy captured" if row == 0 else "Spatial residual captured")
            if row == 0:
                ax.set_title(COHORT_LABELS[col], pad=11, fontsize=8.1)
            ax.text(-.22, 1.09, chr(65 + row*3 + col), transform=ax.transAxes,
                    fontsize=PANEL_LABEL_PT, fontweight="bold", va="bottom")
            axis(ax)
    legend = [Line2D([], [], color=c, marker=m, lw=LW_DATA, ms=4, label=l)
              for c, m, l in zip(HUES, ["o", "s", "^", "D", "v", "P"], LABELS)]
    fig.legend(handles=legend, loc="upper center", bbox_to_anchor=(.535,.965), ncol=3,
               frameon=False, columnspacing=1.4, handlelength=2)
    fig.text(.535,.845, "Every dictionary contains the same one-channel broadcast", ha="center", fontsize=8.2)
    fig.text(.085,.028, "Cell means; randomized controls average 200 draws per cell. K = 8: n = 8, 47, 8 cells.\n"
             "Available K varies with site count; cohorts are not pooled. Spatial residual removes the weighted broadcast.",
             fontsize=PT_SMALL, va="bottom")
    fig.savefig(out / "anatomy_commonmode_capacity.pdf")
    fig.savefig(out / "anatomy_commonmode_capacity.png", dpi=180)
    plt.close(fig)

    fig = plt.figure(figsize=(7.2, 6.7))
    gs = fig.add_gridspec(2, 1, left=.245, right=.975, bottom=.11, top=.885, hspace=.51,
                         height_ratios=[1, 1.15])
    ax = fig.add_subplot(gs[0,0])
    cohort_colors = [COLORS["shunting"], COLORS["additive"], COLORS["oracle"]]
    cohort_markers = ["o", "s", "^"]
    for ci, cohort in enumerate(COHORTS):
        contrasts = {v["control"]: v for v in reports[cohort]["comparisons"] if v["metric"] == "residual_capture"}
        for j, method in enumerate(METHODS[1:5]):
            item = contrasts[method]
            mean = item["mean_difference"]*100
            lo, hi = np.asarray(item["ci95"])*100
            y = 3-j + (1-ci)*.19
            ax.errorbar(mean, y, xerr=[[mean-lo],[hi-mean]], fmt=cohort_markers[ci],
                        color=cohort_colors[ci], ms=MARKER_MS, capsize=2, elinewidth=LW_ERR)
    ax.axvline(0, color=COLORS["mute"], lw=.85, ls="--")
    ax.set_yticks(range(4), LABELS[1:5][::-1])
    ax.set_ylim(-.5,3.5)
    ax.set_xlabel("Ancestry advantage in spatial-residual capture (percentage points)")
    ax.set_title("Common broadcast + seven spatial channels", loc="left", pad=11)
    axis(ax)
    ax.text(-.25, 1.11, "A", transform=ax.transAxes, fontsize=PANEL_LABEL_PT, fontweight="bold")
    ax2 = fig.add_subplot(gs[1,0])
    for ci, cohort in enumerate(COHORTS):
        focus = tables[cohort][tables[cohort].channels.eq(8)].groupby("method")
        for j, method in enumerate(METHODS):
            row = focus.get_group(method)
            ax2.scatter(row.wiring_density.mean()*100, j+(ci-1)*.19, s=26,
                        color=cohort_colors[ci], marker=cohort_markers[ci], edgecolors="white", linewidths=.45)
    ax2.set_yticks(range(6), LABELS)
    ax2.invert_yaxis()
    ax2.set_xlim(10,104)
    ax2.set_xticks([12.5,25,50,75,100], ["12.5", "25", "50", "75", "100"])
    ax2.set_xlabel("Nonzero coefficients (% of a dense K = 8 dictionary)")
    ax2.set_title("Equal channel count does not imply equal wiring", loc="left", pad=11)
    axis(ax2)
    ax2.text(-.25, 1.09, "B", transform=ax2.transAxes, fontsize=PANEL_LABEL_PT, fontweight="bold")
    handles = [Line2D([],[],color=c, marker=m,ls="none",label=l)
               for c,m,l in zip(cohort_colors,cohort_markers,["Development (8)","Disjoint cells (47)","Second mouse (8)"])]
    fig.legend(handles=handles,loc="upper center",bbox_to_anchor=(.535,.984),ncol=3,frameon=False,columnspacing=1.0)
    fig.text(.06,.017,"Paired cell means and 95% cell-bootstrap intervals; intervals describe within-cohort consistency.\n"
             "Ancestry and shuffled routes have identical nonzero counts. Actual ranks are reported in the source tables.",
             fontsize=PT_SMALL,va="bottom")
    fig.savefig(out / "anatomy_commonmode_controls.pdf")
    fig.savefig(out / "anatomy_commonmode_controls.png",dpi=180)
    plt.close(fig)
    manifest = {str(p.relative_to(JOURNAL)):hashlib.sha256(p.read_bytes()).hexdigest()
                for p in [Path(__file__), JOURNAL / "scripts/journal_style.py"] +
                [OUT / c / "cell_method_summary.csv" for c in COHORTS]}
    (out / "figure_inputs.json").write_text(json.dumps(manifest,indent=2)+"\n")


if __name__ == "__main__":
    main()
