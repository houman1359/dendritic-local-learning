"""Native S34: nested baselines and observed-input sensitivity on seven targets."""
from pathlib import Path
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from journal_style import (apply_neurips_style, COLORS, LW_DATA, LW_EDGE,
                           LW_ERR, LW_HAIR, LW_REF, PT_BASE, panel_title,
                           style_axis)
from neurips_style import FIG_W

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "source_data/review_response_baselines"
DEST = ROOT / "figures/supplementary/figure_S34_panels_A-D.pdf"

# Green is reserved for the topology-matched (ancestry) routes, the hue the
# frozen S31C uses for its ancestry row, so one hue keeps one meaning across the
# two neighbouring measured-response sheets; ridge is blue in every panel.
ANCESTRY_GREEN = COLORS["shunting"]
RIDGE, EXACT, GREY = COLORS["additive"], COLORS["bp"], COLORS["point_mlp"]
# Shared normalized-MSE range: A's x axis and B-D's y axes, with the
# training-mean predictor (normalized MSE = 1) as a dotted reference line.
NMSE_LIM = (.60, 1.015)
DOT = dict(s=8, alpha=.4, linewidths=0, zorder=2)


def _token_weights(fig):
    """Pin spine, tick and grid weights onto the journal line-weight tokens.

    ``journal_style.style_axis`` predates the 2026-09-08 token set: it still
    hard-sets 0.8 pt spines/ticks and 0.6 pt grid lines, neither of which is a
    line-weight token, so the strict canvas audit reports every panel.  This
    pass runs after the panels are final and re-states the same hierarchy in
    tokens -- spine and major tick to ``LW_EDGE`` (the edge weight the native
    canvas uses for exactly these marks), minor tick and grid line to
    ``LW_HAIR``.  Geometry, data and colour are untouched.
    """
    for ax in fig.get_axes():
        for spine in ax.spines.values():
            spine.set_linewidth(LW_EDGE)
        ax.tick_params(axis="both", which="major", width=LW_EDGE,
                       grid_linewidth=LW_HAIR)
        ax.tick_params(axis="both", which="minor", width=LW_HAIR,
                       grid_linewidth=LW_HAIR)
        for line in (*ax.get_xgridlines(), *ax.get_ygridlines()):
            line.set_linewidth(LW_HAIR)


def main():
    apply_neurips_style()
    summary = pd.read_csv(SOURCE / "condition_summary.csv").set_index("method")
    targets = pd.read_csv(SOURCE / "target_metrics.csv").pivot(
        index="target_root_id", columns="method", values="nmse")
    contrast = pd.read_csv(SOURCE / "paired_ridge_contrasts.csv").set_index("comparator")
    fig = plt.figure(figsize=(FIG_W, 6.6))
    grid = fig.add_gridspec(2, 2, left=.14, right=.98, bottom=.11, top=.92,
                           wspace=.48, hspace=.5)
    ag = grid[0, 0].subgridspec(2, 1, height_ratios=[1.4, 2.7], hspace=.6)
    a, ac = fig.add_subplot(ag[0]), fig.add_subplot(ag[1])
    b, c, d = [fig.add_subplot(grid[i, j]) for i, j in [(0, 1), (1, 0), (1, 1)]]

    # A, absolute held-out error of the linear baselines and the exact fit.
    # The training mean is 1 by construction and is drawn as a reference line;
    # the route surrogates are compared to ridge in the paired strip below and
    # keep their absolute rows in Supplementary Fig. S31C.
    methods = ["ols_all", "ridge_all", "archived_exact compartment error"]
    labels = ["Linear (OLS)", "Nested ridge", "Exact compartment\nerror"]
    colors = [GREY, RIDGE, EXACT]
    a.axvline(1, color=COLORS["mute"], lw=LW_EDGE, ls=":", zorder=1)
    for i, (method, color) in enumerate(zip(methods, colors)):
        row = summary.loc[method]
        a.scatter(targets[method], i + np.linspace(-.13, .13, len(targets)),
                  color=color, **DOT)
        a.errorbar(row.mean_nmse, i, xerr=[[row.mean_nmse-row.ci95_low],
                   [row.ci95_high-row.mean_nmse]], fmt="D", ms=3.5, color=color,
                   capsize=2, lw=LW_ERR, zorder=3)
    a.set(yticks=np.arange(len(methods)), yticklabels=labels, xlim=NMSE_LIM,
          ylim=(len(methods)-.5, -.5), xlabel="Held-out normalized MSE")
    a.tick_params(axis="y", labelsize=PT_BASE, length=0)
    panel_title(a, "A", "Prediction baselines")
    style_axis(a, grid="x")

    # Paired differences from nested ridge, one row per comparator, per target.
    comparators = ["ols_all", "ridge_raw", "archived_exact compartment error",
                   "archived_topology-matched routes",
                   "archived_random anatomical routes",
                   "archived_site-shuffled routes"]
    clabels = ["Linear (OLS)", "Ridge, raw inputs", "Exact compartment\nerror",
               "Topology routes", "Random routes", "Site-shuffled routes"]
    ccolors = [GREY, RIDGE, EXACT, ANCESTRY_GREEN, GREY, GREY]
    ac.axvline(0, color=COLORS["mute"], lw=LW_EDGE, ls=":", zorder=1)
    for i, (method, color) in enumerate(zip(comparators, ccolors)):
        delta = targets[method] - targets.ridge_all
        row = contrast.loc[method]
        mean = row.other_minus_allridge_mean_nmse
        assert abs(delta.mean() - mean) < 1e-9, method
        ac.scatter(delta, i + np.linspace(-.13, .13, len(delta)), color=color, **DOT)
        ac.errorbar(mean, i, xerr=[[mean-row.ci95_low], [row.ci95_high-mean]],
                    fmt="D", color=color, ms=3.5, capsize=2, lw=LW_ERR, zorder=3)
    ac.set(yticks=np.arange(len(comparators)), yticklabels=clabels,
           ylim=(len(comparators)-.5, -.5), xlim=(-.055, .13),
           xticks=[-.05, 0, .05, .1],
           xlabel="Difference from nested ridge (normalized MSE)")
    ac.tick_params(axis="y", labelsize=PT_BASE, length=0)
    ac.xaxis.label.set_size(PT_BASE)
    style_axis(ac, grid="x")

    def sequence(ax, keys, x, color, marker="o"):
        values = summary.loc[keys]
        for _, row in targets[keys].iterrows():
            ax.plot(x, row.to_numpy(), color=color, alpha=.4, lw=LW_EDGE, zorder=1)
            ax.scatter(x, row.to_numpy(), color=color, **DOT)
        ax.errorbar(x, values.mean_nmse, yerr=np.vstack([
            values.mean_nmse-values.ci95_low, values.ci95_high-values.mean_nmse]),
            fmt=marker+"-", color=color, ms=4, capsize=2, lw=LW_DATA, zorder=3)
        ax.set_ylim(*NMSE_LIM)
        ax.axhline(1, color=COLORS["mute"], lw=LW_EDGE, ls=":")
        ax.set_ylabel("Held-out normalized MSE")
        style_axis(ax, grid="y")

    keys = ["keep_0.25", "keep_0.5", "keep_0.75", "ridge_all"]
    sequence(b, keys, [0, 1, 2, 3], RIDGE)
    # Manual-only partners are a provenance restriction, not a retention dose:
    # the same ridge model (blue, square) sits beyond a separator, off the dose
    # axis.
    manual_x = 4.7
    row = summary.loc["manual_only"]
    b.axvline(3.85, color=COLORS["grid"], lw=LW_REF, zorder=1)
    b.scatter(manual_x + np.linspace(-.13, .13, len(targets)), targets.manual_only,
              color=RIDGE, **DOT)
    b.errorbar(manual_x, row.mean_nmse, yerr=[[row.mean_nmse-row.ci95_low],
               [row.ci95_high-row.mean_nmse]], fmt="s", color=RIDGE,
               capsize=2, ms=4, zorder=3)
    b.set_xticks([0, 1, 2, 3, manual_x], ["25%", "50%", "75%", "All", "Manual\nonly"])
    b.set_xlim(-.4, manual_x + .4)
    b.set_xlabel("Retained observed presynaptic inputs")
    b.tick_params(axis="x", labelsize=PT_BASE)
    panel_title(b, "B", "Observed-input sensitivity")
    sequence(c, ["ridge_all", "noise_0.25", "noise_0.5", "noise_1"],
             [0, .25, .5, 1], RIDGE)
    c.set_xticks([0, .25, .5, 1])
    c.set_xlabel("Added predictor noise (training feature SD)")
    c.xaxis.label.set_size(PT_BASE)
    panel_title(c, "C", "Measurement-noise sensitivity")
    keys = ["ridge_all", "training_reliability_ge_0", "training_reliability_ge_0.1",
            "training_reliability_ge_0.2"]
    sequence(d, keys, np.arange(4), RIDGE)
    labels = ["All", "≥0", "≥0.1", "≥0.2"]
    d.set_xticks(range(4), [f"{label}\n({summary.loc[key].mean_features:.1f})"
                          for label, key in zip(labels, keys)])
    d.set_xlabel("Training repeat reliability\n(mean retained partner count)")
    panel_title(d, "D", "Training-only partner filtering")
    DEST.parent.mkdir(parents=True, exist_ok=True)
    _token_weights(fig)
    fig.savefig(DEST, metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(SOURCE / "figure_S34_preview.png", dpi=180)
    (SOURCE / "figure_S34_provenance.json").write_text(json.dumps({
        "script": str(Path(__file__).relative_to(ROOT)),
        "sources": ["condition_summary.csv", "target_metrics.csv", "paired_ridge_contrasts.csv"],
        "statistical_unit": "seven targets; 20,000 target-bootstrap draws",
        "scope": "Only ridge models are refitted under observed-input perturbations; nonlinear references are archived fits."
    }, indent=2)+"\n")


if __name__ == "__main__":
    main()
