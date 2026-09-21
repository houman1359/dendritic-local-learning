#!/usr/bin/env python3
"""Main Figure 3: preserve the design prediction and expose coefficient delivery."""
import argparse
from pathlib import Path
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
J = HERE.parents[1]
sys.path.insert(0, str(HERE))
import build_ancestry as previous
from focused_provenance import publish
from figure_canvas import NativeCanvas, Margins, COLORS, PT_SMALL, PT_ANNOT, LW_REF, LW_ERR, LW_DATA
from journal_style import style_direct_color_labels


def encoder(ax, rows):
    conditions = [(256, .5), (16, 0.)]
    ax.axhline(0, color=COLORS["mute"], lw=LW_REF, ls="--", zorder=0)
    for readout, folder, color, marker in [
        ("Soft", "review_coefficient_encoder", COLORS["additive"], "o"),
        ("Hard*", "review_coefficient_hard_readout", COLORS["shunting"], "s"),
    ]:
        trajectory = pd.read_csv(J / "source_data" / folder / "trajectories.csv", float_precision="round_trip")
        contrast = pd.read_csv(J / "source_data" / folder / "paired_contrasts.csv", float_precision="round_trip")
        means = []
        for x, (size, noise) in enumerate(conditions):
            chosen = trajectory[(trajectory.epoch == 80) & (trajectory.calibration_samples == size)
                                & (trajectory.cue_noise_sd == noise) & (trajectory.cue_delay_trials == 0)]
            paired = chosen.pivot(index="seed", columns="method", values="heldout_accuracy")
            assert len(paired) == 20 and not paired.isna().any().any()
            assert set(paired.index) == set(range(52000, 52020))
            if size == 16 and noise == 0:
                selected = chosen[chosen.method.eq("learned_local_cue")]
                assert selected.coefficient_group_accuracy.eq(1.).all()
            values = 100 * (paired.learned_local_cue - paired.oracle_context)
            row = contrast[(contrast.calibration_samples == size) & (contrast.cue_noise_sd == noise)
                           & (contrast.cue_delay_trials == 0) & (contrast.control == "oracle_context")].iloc[0]
            np.testing.assert_allclose(values.mean(), row.mean_pp, atol=1e-12)
            if readout == "Hard*" and noise == 0:
                assert np.array_equal(values.to_numpy(), np.zeros(20))
            ax.scatter(x + np.linspace(-.045, .045, 20), values, s=7, color=color, alpha=.3, edgecolors="none")
            ax.errorbar(x, row.mean_pp, yerr=[[row.mean_pp-row.ci95_low_pp], [row.ci95_high_pp-row.mean_pp]],
                        marker=marker, color=color, mfc="white", ms=4, lw=LW_ERR, capsize=2, zorder=4)
            means.append(row.mean_pp)
            status = "primary" if readout == "Soft" and noise == .5 else "exploratory sensitivity"
            rows.append(dict(panel="D", record="mean_contrast", readout=readout.rstrip("*"),
                             calibration_samples=size, cue_noise_sd=noise, cue_delay_trials=0,
                             mean_pp=row.mean_pp, ci95_low_pp=row.ci95_low_pp, ci95_high_pp=row.ci95_high_pp,
                             analysis_status=status, n_seeds=20))
            for seed, value in values.items():
                rows.append(dict(panel="D", record="paired_seed", readout=readout.rstrip("*"),
                                 calibration_samples=size, cue_noise_sd=noise, cue_delay_trials=0,
                                 seed=int(seed), accuracy_difference_pp=value, analysis_status=status))
        ax.plot([0, 1], means, color=color, marker="", lw=LW_DATA, zorder=2)
    ax.text(.07, -43, "Soft", color=COLORS["additive"], fontsize=PT_SMALL)
    ax.text(.07, -7, "Hard (exploratory)", color=COLORS["shunting"], fontsize=PT_SMALL)
    ax.text(1.13, 1.5, "Oracle", color=COLORS["mute"], fontsize=PT_SMALL, ha="right")
    ax.text(.96, -62, "Correct route,\nleaky readout", fontsize=PT_SMALL, ha="right", va="top", color=COLORS["additive"])
    ax.set(xlim=(-.14, 1.17), ylim=(-79, 9), yticks=[-60, -40, -20, 0],
           xticks=[0, 1], xticklabels=["256 cues\nnoise SD 0.5", "16 cues\nno noise"],
           ylabel="Accuracy − oracle (pp)")
    ax.tick_params(axis="x", labelsize=PT_SMALL)


def build(emit_main=False):
    outcomes = pd.read_csv(previous.DATA / "seed_outcomes.csv")
    summary = pd.read_csv(previous.DATA / "condition_summary.csv")
    contrasts = pd.read_csv(previous.REVIEW / "ancestry_k4_control_contrasts.csv")
    pairs = pd.read_csv(previous.REVIEW / "ancestry_control_paired_differences.csv")
    pairs = pairs[pairs.budget_k.eq(4)]
    prediction = previous.coefficient_prediction()
    rows = [dict(panel="B", record="design prediction", **r) for r in prediction.to_dict("records")]
    canvas = NativeCanvas(572/72, 4, row_weights=[124, 120, 65, 99], hgutter_pt=38, vgutter_pt=35,
                          margins=Margins(left=39, right=13, top=24, bottom=34))
    a = canvas.panel("A", 0, 0, 7, title="Eight-context hierarchical task", schematic=True, lock=False)
    b = canvas.panel("B", 0, 7, 5, title="Coefficient sums set by task design")
    c = canvas.panel("C", 1, 0, 6, title="Learning across channel budgets", lock=False)
    d = canvas.panel("D", 1, 6, 6, title="Separate cue-learning cohort", lock=False)
    e = canvas.panel("E", 2, 0, 12, title="K = 4: ancestry exceeds the strongest matched control", lock=False)
    f = canvas.panel("F", 3, 0, 12, title="K = 4: ancestry versus individual controls")
    previous.original.hierarchical_task(a)
    previous.prediction_panel(b, prediction)
    dendritic = summary[summary.architecture.eq("dendritic_tree")]
    previous.routing.bandwidth_sweep_compact(c, outcomes, dendritic)
    c.axhline(.5, color=COLORS["mute"], lw=LW_REF, ls="--", zorder=0)
    c.set_ylabel("Held-out accuracy")
    for text in c.texts:
        if text.get_text() == "best control": text.set_text("best matched control"); text.set_position((.03, .70))
        elif text.get_text() == "correct": text.set_text("ancestry"); text.set_position((1.53, .843))
    c.text(3.12, .505, "chance", ha="right", va="bottom", fontsize=PT_SMALL, color=COLORS["mute"])
    for family in ["correct_ancestry_subtrees", "within_neuron_route_derangement"]:
        for r in dendritic[dendritic.feedback_family.eq(family)].to_dict("records"):
            rows.append(dict(panel="C", record="condition_mean", **r))
    for k, mean, low, high in previous.routing._best_control_by_budget(outcomes):
        rows.append(dict(panel="C", record="condition_mean", feedback_family="best_matched_control",
                         budget_k=int(k), mean_heldout_accuracy=mean, ci95_low_heldout_accuracy=low, ci95_high_heldout_accuracy=high))
    encoder(d, rows)
    previous.primary_effect(e, contrasts, pairs)
    previous.individual_effects(f, contrasts, pairs)
    for letter, controls in [("E", ["best_matched_nonanatomical_oracle"]),
                             ("F", ["random_rank_k", "random_sparse_matched", "depth_interleaved_bins"])]:
        rows.extend(dict(panel=letter, record="mean_contrast", **r) for r in contrasts[contrasts.control.isin(controls)].to_dict("records"))
        rows.extend(dict(panel=letter, record="paired_seed", **r) for r in pairs[pairs.control.isin(controls)].to_dict("records"))
    canvas.declare_reserve("F", left=80)
    style_direct_color_labels(canvas.fig)
    canvas.lock_reserves()
    right = [r["art"] for r in canvas._letters if r["letter"] in "BD"]
    x = min(r.get_position()[0] for r in right)
    for r in right: r.set_position((x, r.get_position()[1]))
    output = J / "figures/components/focused_main_03.pdf"
    findings = canvas.save(output, name="focused_main_03", dpi=180, lock=False)
    plt.close(canvas.fig)
    sources = [previous.CONFIG, previous.DATA/"seed_outcomes.csv", previous.DATA/"condition_summary.csv",
               previous.REVIEW/"ancestry_k4_control_contrasts.csv", previous.REVIEW/"ancestry_control_paired_differences.csv"]
    sources += [J/"source_data"/folder/name for folder in ["review_coefficient_encoder", "review_coefficient_hard_readout"]
                for name in ["trajectories.csv", "paired_contrasts.csv", "protocol_freeze.json"]]
    builders = [Path(__file__), Path(previous.__file__), Path(previous.original.__file__), Path(previous.routing.__file__),
                Path(previous.experiment.__file__), J/"scripts/figure_canvas.py", J/"scripts/journal_style.py"]
    panels = {"A": "Unchanged eight-stream task schematic; supplied context and teacher features.",
              "B": "Raw ancestry-group coefficient sums from the frozen generator; normalization unchanged.",
              "C": "Original matched-bandwidth accuracy and maximum-over-four-control summaries.",
              "D": "Separate coefficient-learning cohort, seeds 52000–52019, distinct from the ancestry cohort in C/E/F. Paired final accuracy minus oracle at noisy 256-cue and noiseless 16-cue conditions. Soft noisy is primary; hard readout and the noiseless comparison are exploratory; hard/soft reuse the same twenty coefficient-study seeds. Both receive supplied cues and four-route activation targets. Correct identification concerns the route group, not unique identification of every one of the eight contexts. Lines join two specified conditions with different calibration resources, not a continuous noise dose.",
              "E": "Primary K4 small effect with original intervals and twenty paired points.",
              "F": "Individual K4 controls on their own wider scale; original endpoints and intervals."}
    publish(3, output, rows, sources, builders, panels, emit_main=emit_main, layout_findings=findings,
            notes="Editorial promotion of retained encoder evidence; no new training, coefficient fit, endpoint selection or bootstrap. Original matched-versus-rewired panel remains in the archived native figure and complete Source Data.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--emit-main", action="store_true")
    build(parser.parse_args().emit_main)
