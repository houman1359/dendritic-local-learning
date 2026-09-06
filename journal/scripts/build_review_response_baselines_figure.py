"""Native S34: nested baselines and observed-input sensitivity on seven targets."""
from pathlib import Path
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from journal_style import apply_neurips_style, COLORS, panel_title, style_axis
from neurips_style import FIG_W

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "source_data/review_response_baselines"
DEST = ROOT / "figures/supplementary/figure_S34_panels_A-D.pdf"


def main():
    apply_neurips_style()
    summary = pd.read_csv(SOURCE / "condition_summary.csv").set_index("method")
    targets = pd.read_csv(SOURCE / "target_metrics.csv").pivot(
        index="target_root_id", columns="method", values="nmse")
    contrast = pd.read_csv(SOURCE / "paired_ridge_contrasts.csv").set_index("comparator")
    fig = plt.figure(figsize=(FIG_W, 6.6))
    grid = fig.add_gridspec(2, 2, left=.14, right=.98, bottom=.15, top=.92,
                           wspace=.48, hspace=.52)
    ag = grid[0, 0].subgridspec(2, 1, height_ratios=[3.2, 1.05], hspace=.73)
    a, ac = fig.add_subplot(ag[0]), fig.add_subplot(ag[1])
    b, c, d = [fig.add_subplot(grid[i, j]) for i, j in [(0, 1), (1, 0), (1, 1)]]
    methods = ["training_mean", "ols_all", "ridge_all", "archived_exact compartment error",
               "archived_topology-matched routes", "archived_random anatomical routes",
               "archived_site-shuffled routes"]
    labels = ["Train mean", "Linear (OLS)", "Nested ridge", "Exact nonlinear",
              "Subtree nonlinear", "Random nonlinear", "Shuffle nonlinear"]
    colors = [COLORS["mute"], COLORS["point_mlp"], COLORS["additive"], COLORS["bp"],
              COLORS["dend"], COLORS["point_mlp"], COLORS["mute"]]
    for i, (method, color) in enumerate(zip(methods, colors)):
        row = summary.loc[method]
        a.scatter(targets[method], i + np.linspace(-.11, .11, len(targets)),
                  s=8, color=color, alpha=.4, linewidths=0, zorder=2)
        a.errorbar(row.mean_nmse, i, xerr=[[row.mean_nmse-row.ci95_low],
                   [row.ci95_high-row.mean_nmse]], fmt="D", ms=3.5, color=color,
                   capsize=2, lw=.9, zorder=3)
    a.set(yticks=np.arange(len(methods)), yticklabels=labels, xlim=(.57, 1.035),
          ylim=(6.55, -.55), xlabel="Held-out normalized MSE")
    a.tick_params(axis="y", labelsize=6.8, length=0)
    panel_title(a, "A", "Prediction baselines")
    style_axis(a, grid="x")
    delta = targets[methods[3]] - targets.ridge_all
    row = contrast.loc[methods[3]]
    ac.axvline(0, color=COLORS["mute"], lw=.7, ls=":")
    ac.scatter(delta, np.linspace(-.12, .12, len(delta)), s=11, color=COLORS["bp"], alpha=.55)
    mean = row.other_minus_allridge_mean_nmse
    ac.errorbar(mean, 0, xerr=[[mean-row.ci95_low], [row.ci95_high-mean]], fmt="D",
                color=COLORS["bp"], ms=4, capsize=2, lw=1.1)
    ac.set(yticks=[], ylim=(-.4, .4), xlim=(-.055, .016),
           xlabel="Exact nonlinear − ridge (normalized MSE)")
    ac.xaxis.label.set_size(7.1)
    style_axis(ac, grid="x")

    def sequence(ax, keys, x, color, marker="o"):
        values = summary.loc[keys]
        for _, row in targets[keys].iterrows():
            ax.plot(x, row.to_numpy(), color=color, alpha=.15, lw=.65)
        ax.errorbar(x, values.mean_nmse, yerr=np.vstack([
            values.mean_nmse-values.ci95_low, values.ci95_high-values.mean_nmse]),
            fmt=marker+"-", color=color, ms=4, capsize=2, lw=1.25)
        ax.set_ylim(.63, 1.015)
        ax.axhline(1, color=COLORS["mute"], lw=.7, ls=":")
        ax.set_ylabel("Held-out normalized MSE")
        style_axis(ax, grid="y")

    keys = ["keep_0.25", "keep_0.5", "keep_0.75", "ridge_all"]
    sequence(b, keys, [0, 1, 2, 3], COLORS["additive"])
    row = summary.loc["manual_only"]
    b.scatter(4 + np.linspace(-.08, .08, len(targets)), targets.manual_only,
              s=10, color=COLORS["point_mlp"], alpha=.4)
    b.errorbar(4, row.mean_nmse, yerr=[[row.mean_nmse-row.ci95_low],
               [row.ci95_high-row.mean_nmse]], fmt="s", color=COLORS["point_mlp"],
               capsize=2, ms=4)
    b.set_xticks(range(5), ["25%", "50%", "75%", "All", "Manual\nonly"])
    b.set_xlabel("Retained observed presynaptic inputs")
    b.tick_params(axis="x", labelsize=7)
    panel_title(b, "B", "Observed-input sensitivity")
    sequence(c, ["ridge_all", "noise_0.25", "noise_0.5", "noise_1"],
             [0, .25, .5, 1], COLORS["additive"])
    c.set_xticks([0, .25, .5, 1])
    c.set_xlabel("Added predictor noise (training feature SD)")
    c.xaxis.label.set_size(7.5)
    panel_title(c, "C", "Measurement-noise sensitivity")
    keys = ["ridge_all", "training_reliability_ge_0", "training_reliability_ge_0.1",
            "training_reliability_ge_0.2"]
    sequence(d, keys, np.arange(4), COLORS["dend"])
    labels = ["All", "≥0", "≥0.1", "≥0.2"]
    d.set_xticks(range(4), [f"{label}\n({summary.loc[key].mean_features:.1f})"
                          for label, key in zip(labels, keys)])
    d.set_xlabel("Training repeat reliability\n(mean retained partner count)")
    panel_title(d, "D", "Training-only partner filtering")
    fig.text(.14, .025, "Seven targets; thin traces/dots: target means; diamonds/lines: means and 95% target-bootstrap intervals.",
             fontsize=7, color=COLORS["mute"])
    DEST.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(DEST)
    fig.savefig(SOURCE / "figure_S34_preview.png", dpi=180)
    (SOURCE / "figure_S34_provenance.json").write_text(json.dumps({
        "script": str(Path(__file__).relative_to(ROOT)),
        "sources": ["condition_summary.csv", "target_metrics.csv", "paired_ridge_contrasts.csv"],
        "statistical_unit": "seven targets; 20,000 target-bootstrap draws",
        "scope": "Only ridge models are refitted under observed-input perturbations; nonlinear references are archived fits."
    }, indent=2)+"\n")


if __name__ == "__main__":
    main()
