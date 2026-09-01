#!/usr/bin/env python3
"""Summarize the frozen prospective depth-by-feedback experiment.

The audit table is the only input.  This keeps statistical summaries and
figures downstream of the artifact, checkpoint, and configuration checks in
``audit_prospective_learning_runs.py``.  The script refuses to summarize a
partial or unbalanced confirmatory cohort.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from journal_style import (
    COLORS,
    ERR_CAPSIZE,
    FIG_W,
    LW_DATA,
    LW_ERR,
    LW_HAIR,
    LW_REF,
    MARKER_MS,
    MARKERS,
    PT_LEGEND,
    PT_SMALL,
    SEED_ALPHA,
    SEED_MS,
    apply_neurips_style,
    audit_layout,
    audit_text_over_data,
    clean_legend,
    panel_title,
    style_axis,
    wrap_ticklabels,
)


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "analysis"
SOURCE_DATA = ROOT / "source_data" / "prospective_learning"
FIGURES = ROOT / "figures" / "generated"

TASK_LABEL = {"mnist": "MNIST", "noise_resilience": "Noise resilience"}
CORE_LABEL = {
    "dendritic_shunting": "shunting",
    "dendritic_additive": "raw additive",
}
CORE_COLOR = {
    "dendritic_shunting": COLORS["shunting"],
    "dendritic_additive": COLORS["additive"],
}
# One marker per core everywhere a core is plotted (colour is never the only
# cue): shunting = circle, additive = square.
CORE_MARKER = {
    "dendritic_shunting": MARKERS[0],
    "dendritic_additive": MARKERS[1],
}
FEEDBACK_LABEL = {
    "per_soma": "Matched-width fallback",
    "per_soma_shared": "Neuron-specific",
    "path_transport": "Exact path",
    "backprop": "Backpropagation",
}

apply_neurips_style()


def bootstrap_ci(values: np.ndarray, *, seed: int, n_boot: int = 20_000):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    sampled = rng.choice(values, size=(n_boot, values.size), replace=True)
    means = sampled.mean(axis=1)
    lo, hi = np.quantile(means, [0.025, 0.975])
    return float(values.mean()), float(lo), float(hi)


def paired_test(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    if np.allclose(values, 0.0):
        return 1.0
    return float(
        wilcoxon(
            values, zero_method="wilcox", correction=False, alternative="two-sided"
        ).pvalue
    )


def validate(frame: pd.DataFrame, expected_seeds: int) -> pd.DataFrame:
    failed = frame[frame.status != "pass"]
    if not failed.empty:
        raise SystemExit(
            f"Refusing partial analysis: {len(failed)} rows did not pass the audit"
        )
    key = ["task", "core", "strategy", "feedback", "depth"]
    counts = frame.groupby(key, dropna=False).seed.nunique()
    bad = counts[counts != expected_seeds]
    if not bad.empty:
        raise SystemExit(f"Unbalanced seed counts:\n{bad.to_string()}")
    if frame.duplicated([*key, "seed"]).any():
        raise SystemExit("Duplicate condition-by-seed rows in audit table")
    return frame.copy()


def paired_values(
    frame: pd.DataFrame,
    left: dict[str, object],
    right: dict[str, object],
) -> pd.DataFrame:
    def select(spec: dict[str, object], label: str) -> pd.DataFrame:
        part = frame
        for column, value in spec.items():
            part = part[part[column] == value]
        return part[["seed", "test_accuracy"]].rename(columns={"test_accuracy": label})

    paired = select(left, "left").merge(
        select(right, "right"), on="seed", validate="one_to_one"
    )
    paired["difference"] = paired.left - paired.right
    return paired


def add_contrast(
    rows: list[dict[str, object]],
    *,
    family: str,
    task: str,
    core: str,
    depth: int,
    contrast: str,
    paired: pd.DataFrame,
    seed: int,
) -> None:
    values = paired.difference.to_numpy(dtype=float)
    mean, lo, hi = bootstrap_ci(values, seed=seed)
    rows.append(
        {
            "family": family,
            "task": task,
            "core": core,
            "depth": depth,
            "contrast": contrast,
            "n_pairs": len(values),
            "mean_difference": mean,
            "ci95_low": lo,
            "ci95_high": hi,
            "wins": int((values > 0).sum()),
            "ties": int(np.isclose(values, 0.0).sum()),
            "wilcoxon_p_two_sided": paired_test(values),
        }
    )


def contrasts(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    counter = 0
    tasks = sorted(frame.task.unique())
    cores = sorted(frame.core.unique())
    depths = sorted(frame.depth.unique())

    for task in tasks:
        for core in cores:
            for depth in depths:
                base = {"task": task, "core": core, "depth": depth}
                for left, right, name in (
                    ("per_soma_shared", "per_soma", "neuron-indexed - scalar"),
                    ("path_transport", "per_soma_shared", "exact - neuron-indexed"),
                ):
                    paired = paired_values(
                        frame,
                        {**base, "strategy": "local_ca", "feedback": left},
                        {**base, "strategy": "local_ca", "feedback": right},
                    )
                    add_contrast(
                        rows,
                        family="feedback",
                        task=task,
                        core=core,
                        depth=depth,
                        contrast=name,
                        paired=paired,
                        seed=counter,
                    )
                    counter += 1

                for feedback in ("per_soma", "per_soma_shared", "path_transport"):
                    paired = paired_values(
                        frame,
                        {**base, "strategy": "local_ca", "feedback": feedback},
                        {**base, "strategy": "standard", "feedback": "backprop"},
                    )
                    add_contrast(
                        rows,
                        family="local_minus_bp",
                        task=task,
                        core=core,
                        depth=depth,
                        contrast=f"{feedback} - backprop",
                        paired=paired,
                        seed=counter,
                    )
                    counter += 1

        for depth in depths:
            for feedback in ("per_soma", "per_soma_shared", "path_transport"):
                paired = paired_values(
                    frame,
                    {
                        "task": task,
                        "core": "dendritic_shunting",
                        "strategy": "local_ca",
                        "feedback": feedback,
                        "depth": depth,
                    },
                    {
                        "task": task,
                        "core": "dendritic_additive",
                        "strategy": "local_ca",
                        "feedback": feedback,
                        "depth": depth,
                    },
                )
                add_contrast(
                    rows,
                    family="core",
                    task=task,
                    core="shunting - additive",
                    depth=depth,
                    contrast=f"shunting - additive ({feedback})",
                    paired=paired,
                    seed=counter,
                )
                counter += 1

            scalar = paired_values(
                frame,
                {
                    "task": task,
                    "core": "dendritic_shunting",
                    "strategy": "local_ca",
                    "feedback": "per_soma",
                    "depth": depth,
                },
                {
                    "task": task,
                    "core": "dendritic_additive",
                    "strategy": "local_ca",
                    "feedback": "per_soma",
                    "depth": depth,
                },
            )
            exact = paired_values(
                frame,
                {
                    "task": task,
                    "core": "dendritic_shunting",
                    "strategy": "local_ca",
                    "feedback": "path_transport",
                    "depth": depth,
                },
                {
                    "task": task,
                    "core": "dendritic_additive",
                    "strategy": "local_ca",
                    "feedback": "path_transport",
                    "depth": depth,
                },
            )
            interaction = scalar[["seed", "difference"]].merge(
                exact[["seed", "difference"]], on="seed", suffixes=("_scalar", "_exact")
            )
            interaction["left"] = interaction.difference_scalar
            interaction["right"] = interaction.difference_exact
            interaction["difference"] = interaction.left - interaction.right
            add_contrast(
                rows,
                family="feedback_conditioning",
                task=task,
                core="shunting - additive",
                depth=depth,
                contrast="(shunting - additive) scalar - exact",
                paired=interaction,
                seed=counter,
            )
            counter += 1

    low, high = min(depths), max(depths)
    for task in tasks:
        for core in cores:
            for strategy, feedback in [
                ("standard", "backprop"),
                ("local_ca", "per_soma"),
                ("local_ca", "per_soma_shared"),
                ("local_ca", "path_transport"),
            ]:
                paired = paired_values(
                    frame,
                    {
                        "task": task,
                        "core": core,
                        "strategy": strategy,
                        "feedback": feedback,
                        "depth": high,
                    },
                    {
                        "task": task,
                        "core": core,
                        "strategy": strategy,
                        "feedback": feedback,
                        "depth": low,
                    },
                )
                add_contrast(
                    rows,
                    family="depth",
                    task=task,
                    core=core,
                    depth=high,
                    contrast=f"depth {high} - depth {low} ({feedback})",
                    paired=paired,
                    seed=counter,
                )
                counter += 1

    return pd.DataFrame(rows)


def condition_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    key = ["task", "core", "strategy", "feedback", "depth"]
    for values, part in frame.groupby(key, sort=True):
        mean, lo, hi = bootstrap_ci(part.test_accuracy.to_numpy(), seed=len(rows))
        rows.append(
            dict(
                zip(key, values),
                n_seeds=part.seed.nunique(),
                mean_accuracy=mean,
                ci95_low=lo,
                ci95_high=hi,
            )
        )
    return pd.DataFrame(rows)


def _plot_legacy(frame: pd.DataFrame, summary: pd.DataFrame) -> None:
    validity = ROOT / "source_data" / "prospective_input_validity"
    mechanism = pd.read_csv(validity / "mechanism_checkpoint_rows_valid.csv")
    association = pd.read_csv(validity / "mechanism_association_summary_valid.csv")
    routing = pd.read_csv(
        ROOT / "source_data" / "prospective_routing_control" / "paired_contrasts.csv"
    )
    dose = pd.read_csv(
        ROOT / "source_data" / "prospective_followup" / "condition_summary.csv"
    )
    dose = dose[dose.family.eq("inhibition")].copy()
    primary_step = 1e-5
    mechanism = mechanism[
        np.isclose(mechanism.relative_step, primary_step)
        & mechanism.feedback_family.isin(
            ["global_scalar_available", "ancestry_available", "exact_transport"]
        )
    ].copy()

    fig, axes = plt.subplots(
        3,
        4,
        figsize=(FIG_W, 7.35),
        gridspec_kw={
            "left": 0.075,
            "right": 0.985,
            "bottom": 0.065,
            "top": 0.955,
            "wspace": 0.62,
            "hspace": 0.66,
        },
    )
    feedbacks = ["per_soma", "per_soma_shared", "path_transport"]
    feedback_letters = (("A", "B", "C"), ("D", "E", "F"))
    for row, task in enumerate(("mnist", "noise_resilience")):
        for col, feedback in enumerate(feedbacks):
            ax = axes[row, col]
            for core in ("dendritic_shunting", "dendritic_additive"):
                part = summary[
                    (summary.task == task)
                    & (summary.core == core)
                    & (summary.strategy == "local_ca")
                    & (summary.feedback == feedback)
                ].sort_values("depth")
                x = part.depth.to_numpy(dtype=float)
                y = part.mean_accuracy.to_numpy(dtype=float)
                lo = part.ci95_low.to_numpy(dtype=float)
                hi = part.ci95_high.to_numpy(dtype=float)
                ax.errorbar(
                    x,
                    y,
                    yerr=np.vstack([y - lo, hi - y]),
                    color=CORE_COLOR[core],
                    marker="o",
                    ms=4.2,
                    lw=LW_DATA,
                    elinewidth=LW_ERR,
                    capsize=ERR_CAPSIZE,
                    label=CORE_LABEL[core],
                )
                bp = summary[
                    (summary.task == task)
                    & (summary.core == core)
                    & (summary.strategy == "standard")
                ].sort_values("depth")
                ax.plot(
                    bp.depth,
                    bp.mean_accuracy,
                    color=CORE_COLOR[core],
                    lw=0.85,
                    ls="--",
                    alpha=0.65,
                )
            short_feedback = {
                "per_soma": "matched-width fallback",
                "per_soma_shared": "neuron-specific",
                "path_transport": "exact path",
            }[feedback]
            panel_title(ax, feedback_letters[row][col], short_feedback.capitalize())
            ax.set_xticks([1, 2, 3, 4])
            ax.set_xlabel("dendritic stage count")
            if col == 0:
                ax.set_ylabel(
                    "MNIST accuracy" if task == "mnist" else "noise-task accuracy"
                )
            style_axis(ax)
            if row == 0 and col == 2:
                clean_legend(ax, fontsize=PT_LEGEND, loc="lower right")

        ax = axes[row, 3]
        for core in ("dendritic_shunting", "dendritic_additive"):
            part = routing[
                (routing.task == task) & (routing.core == core)
            ].sort_values("depth")
            x = part.depth.to_numpy(dtype=float)
            y = 100 * part.mean_difference.to_numpy(dtype=float)
            lo = 100 * part.ci95_low.to_numpy(dtype=float)
            hi = 100 * part.ci95_high.to_numpy(dtype=float)
            ax.errorbar(
                x,
                y,
                yerr=np.vstack([y - lo, hi - y]),
                color=CORE_COLOR[core],
                marker="o",
                ms=4.2,
                lw=LW_DATA,
                elinewidth=LW_ERR,
                capsize=ERR_CAPSIZE,
                label=CORE_LABEL[core],
            )
        ax.axhline(0, color=COLORS["mute"], lw=0.8, ls="--", zorder=0)
        panel_title(ax, ("G", "H")[row], "Routing map")
        ax.set_xticks([2, 4])
        ax.set_xlabel("dendritic stage count")
        ax.set_ylabel("routing benefit (pp)")
        style_axis(ax)

    dose_modes = [
        ("local_ca", "per_soma", "I", "Matched-width fallback"),
        ("local_ca", "per_soma_shared", "J", "Neuron-specific"),
        ("local_ca", "path_transport", "K", "Exact path"),
        ("standard", "backprop", "L", "Backpropagation"),
    ]
    for col, (strategy, feedback, letter, title) in enumerate(dose_modes):
        ax = axes[2, col]
        for core in ("dendritic_shunting", "dendritic_additive"):
            part = dose[
                (dose.core == core)
                & (dose.strategy == strategy)
                & (dose.feedback == feedback)
            ].sort_values("inhibitory_synapses_per_branch")
            x = part.inhibitory_synapses_per_branch.to_numpy(dtype=float)
            y = part.mean_accuracy.to_numpy(dtype=float)
            lo = part.ci95_low.to_numpy(dtype=float)
            hi = part.ci95_high.to_numpy(dtype=float)
            ax.errorbar(
                x,
                y,
                yerr=np.vstack([y - lo, hi - y]),
                color=CORE_COLOR[core],
                marker="o",
                ms=4.2,
                lw=LW_DATA,
                elinewidth=LW_ERR,
                capsize=ERR_CAPSIZE,
                label=CORE_LABEL[core],
            )
        panel_title(ax, letter, title)
        ax.set_xticks([0, 20, 40])
        ax.set_xlabel("inhibitory contacts")
        if col == 0:
            ax.set_ylabel("noise-task accuracy")
        style_axis(ax)

    superseded = ANALYSIS / "superseded"
    superseded.mkdir(parents=True, exist_ok=True)
    fig.canvas.draw()
    audit_layout(fig, "fig_prospective_learning_benefits_historical_excluded")
    audit_text_over_data(fig, "fig_prospective_learning_benefits_historical_excluded")
    fig.savefig(
        superseded / "fig_prospective_learning_benefits_historical_excluded.pdf",
        metadata={"CreationDate": None, "ModDate": None},
    )
    fig.savefig(
        superseded / "fig_prospective_learning_benefits_historical_excluded.png",
        dpi=600,
    )
    plt.close(fig)

    fig, mechanism_axes = plt.subplots(
        2,
        2,
        figsize=(FIG_W, 3.95),
        gridspec_kw={
            "left": 0.09,
            "right": 0.985,
            "bottom": 0.105,
            "top": 0.92,
            "wspace": 0.48,
            "hspace": 0.62,
        },
    )

    feedback_order = [
        "global_scalar_available",
        "ancestry_available",
        "exact_transport",
    ]
    feedback_labels = ["matched-width\nfallback", "neuron-\nspecific",
                       "exact\npath"]
    feedback_colors = [COLORS["scalar"], COLORS["per_soma"], COLORS["oracle"]]

    def distribution_panel(ax, metric: str, letter: str, title: str, ylabel: str, ylim):
        values = [
            mechanism.loc[mechanism.feedback_family.eq(family), metric].to_numpy(float)
            for family in feedback_order
        ]
        boxes = ax.boxplot(
            values,
            positions=np.arange(3),
            widths=0.55,
            whis=(5, 95),
            showfliers=False,
            patch_artist=True,
            medianprops={"color": COLORS["ink"], "linewidth": 1.15},
            whiskerprops={"color": COLORS["mute"], "linewidth": 0.8},
            capprops={"color": COLORS["mute"], "linewidth": 0.8},
        )
        for box, color, vals in zip(boxes["boxes"], feedback_colors, values):
            box.set_facecolor(color)
            box.set_alpha(0.72)
            box.set_edgecolor("white")
            mean, low, high = bootstrap_ci(vals, seed=700 + len(title))
            position = feedback_colors.index(color)
            ax.errorbar(
                position,
                mean,
                yerr=[[mean - low], [high - mean]],
                color=COLORS["ink"],
                marker="D",
                markerfacecolor="white",
                markersize=3.2,
                linewidth=1.0,
                capsize=2.0,
                zorder=5,
            )
        ax.set_xticks(range(3))
        ax.set_xticklabels(feedback_labels)
        ax.set_ylabel(ylabel)
        ax.set_ylim(*ylim)
        panel_title(ax, letter, title)
        style_axis(ax, grid="y")

    ax_g, ax_h, ax_i, ax_j = mechanism_axes.ravel()
    distribution_panel(
        ax_g,
        "gradient_cosine",
        "A",
        "Gradient direction",
        "gradient cosine",
        (-0.16, 1.08),
    )
    ax_g.axhline(0, color=COLORS["mute"], ls="--", lw=0.8, zorder=0)
    distribution_panel(
        ax_h,
        "gradient_scaled_capture",
        "B",
        "Gradient capture",
        "gradient capture",
        (-0.03, 1.07),
    )
    distribution_panel(
        ax_i,
        "norm_matched_fraction_of_exact",
        "C",
        "One-step learning",
        "one-step progress",
        (-1.58, 1.18),
    )
    descent = (
        mechanism.groupby("feedback_family").norm_matched_is_descent.sum().to_dict()
    )
    # The denominator is the checkpoint count that survives this panel's own
    # filter (primary relative step, input-valid cohort), not the unfiltered
    # collection total; hard-coding the latter understated every fraction.
    checkpoints = mechanism.groupby("feedback_family").size().to_dict()
    for index, family in enumerate(feedback_order):
        ax_i.text(
            index,
            -1.45,
            f"{int(descent[family])}/{int(checkpoints[family])} descent",
            ha="center",
            va="bottom",
            fontsize=PT_SMALL,
            color=COLORS["mute"],
        )

    for family, label, color in zip(
        feedback_order[:2], feedback_labels[:2], feedback_colors[:2]
    ):
        part = mechanism[mechanism.feedback_family.eq(family)]
        ax_j.scatter(
            part.gradient_cosine,
            part.norm_matched_fraction_of_exact,
            s=10,
            color=color,
            alpha=0.34,
            edgecolors="none",
            label=label,
        )
    row = association[
        np.isclose(association.relative_step, primary_step)
        & association.geometry_metric.eq("gradient_cosine")
    ].iloc[0]
    ax_j.text(
        0.04,
        0.95,
        rf"$\rho={row.spearman_rho:.2f}$ "
        rf"[{row.checkpoint_bootstrap_ci95_low:.2f}, {row.checkpoint_bootstrap_ci95_high:.2f}]",
        transform=ax_j.transAxes,
        ha="left",
        va="top",
        fontsize=7.0,
        color=COLORS["ink"],
    )
    ax_j.axhline(0, color=COLORS["mute"], ls="--", lw=0.8, zorder=0)
    ax_j.set_xlim(-0.16, 0.88)
    ax_j.set_ylim(-1.58, 1.18)
    ax_j.set_xlabel("cosine with exact gradient")
    ax_j.set_ylabel("relative one-step progress")
    panel_title(ax_j, "D", "Geometry predicts learning")
    style_axis(ax_j)
    clean_legend(ax_j, fontsize=PT_LEGEND, loc="lower right")

    fig.canvas.draw()
    audit_layout(fig, "fig_prospective_credit_mechanism")
    audit_text_over_data(fig, "fig_prospective_credit_mechanism")
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        FIGURES / "fig_prospective_credit_mechanism.pdf",
        metadata={"CreationDate": None, "ModDate": None},
    )
    fig.savefig(FIGURES / "fig_prospective_credit_mechanism.png", dpi=600)
    plt.close(fig)


def _plot_streamlined_main() -> None:
    """Render the main prospective figure as nine focused evidence panels."""

    valid_root = ROOT / "source_data" / "prospective_input_validity"
    contrast = pd.read_csv(valid_root / "central_valid_paired_contrasts.csv")
    routing = pd.read_csv(
        valid_root / "routing_valid_paired_contrasts.csv"
    )
    followup_contrast = pd.read_csv(
        valid_root / "followup_publication_paired_contrasts.csv"
    )
    clean_exact = pd.read_csv(
        ROOT / "source_data" / "clean_exact_bp" / "exact_bp_contrasts.csv"
    )
    subtree = pd.read_csv(
        ROOT / "source_data" / "trained_subtree_address" / "seed_outcomes.csv"
    ).merge(
        pd.read_csv(
            ROOT / "source_data" / "trained_subtree_address" / "gradient_audit.csv"
        ),
        on=["seed", "condition"],
        validate="one_to_one",
    )

    # Height grew and the top margin widened (0.31 in -> 0.46 in) so the
    # A/B/C panel letters, which rise above the panel titles, clear the canvas.
    # The 0.112 left margin fully holds panel G's two-line routing-condition
    # tick labels ('route derangement' is the widest, and clipped at the
    # canvas edge under the former 0.098 margin); the names match the subtree
    # full-factorial legend verbatim.
    fig, axes = plt.subplots(
        3,
        3,
        figsize=(FIG_W, 6.45),
        gridspec_kw={
            "left": 0.112,
            "right": 0.985,
            "bottom": 0.078,
            "top": 0.933,
            "wspace": 0.55,
            "hspace": 0.58,
        },
    )
    ax_a, ax_b, ax_c, ax_d, ax_e, ax_f, ax_g, ax_h, ax_i = axes.ravel()

    identity = contrast[
        contrast.family.eq("feedback")
        & contrast.contrast.eq("neuron-indexed - scalar")
    ]
    for ax, task, letter, title in [
        (ax_a, "mnist", "G", "MNIST neuron-specific gain"),
        (ax_b, "noise_resilience", "H", "Noise-task neuron-specific gain"),
    ]:
        for core in ("dendritic_shunting", "dendritic_additive"):
            part = identity[identity.task.eq(task) & identity.core.eq(core)].sort_values("depth")
            if part.empty:
                continue
            x = part.depth.to_numpy(float)
            y = 100 * part.mean_difference.to_numpy(float)
            lo = 100 * part.ci95_low.to_numpy(float)
            hi = 100 * part.ci95_high.to_numpy(float)
            ax.errorbar(
                x,
                y,
                yerr=np.vstack([y - lo, hi - y]),
                color=CORE_COLOR[core],
                marker=CORE_MARKER[core],
                ms=MARKER_MS,
                lw=LW_DATA,
                elinewidth=LW_ERR,
                capsize=ERR_CAPSIZE,
                label=CORE_LABEL[core],
            )
        ax.axhline(0, color=COLORS["mute"], ls="--", lw=LW_REF)
        ax.set_xticks([1, 2, 3, 4])
        ax.set_xlabel("dendritic stage count")
        ax.set_ylabel("accuracy gain (pp)")
        panel_title(ax, letter, title)
        style_axis(ax)
    # Two series only: direct colour labels at the well-separated left
    # endpoints replace a legend (shunting sits ~2 pp above additive at
    # depth 1).  This panel is the figure's core-colour statement; panel J
    # cross-references it.
    ax_a.set_ylim(-0.5, 8.6)
    ax_a.text(
        1.0,
        7.55,
        "shunting",
        color=CORE_COLOR["dendritic_shunting"],
        fontsize=PT_LEGEND,
        ha="left",
        va="bottom",
    )
    ax_a.text(
        1.0,
        4.4,
        "raw additive",
        color=CORE_COLOR["dendritic_additive"],
        fontsize=PT_LEGEND,
        ha="left",
        va="top",
    )
    # Panel H carries a single series; a direct colour label replaces a legend
    # and doubles as the in-panel colour key for the additive core.
    ax_b.text(
        1.05,
        7.6,
        "raw additive",
        color=CORE_COLOR["dendritic_additive"],
        fontsize=PT_LEGEND,
        ha="left",
        va="center",
    )

    exact = contrast[
        contrast.family.eq("local_minus_bp")
        & contrast.contrast.eq("path_transport - backprop")
    ]
    # Three series, three hues, three markers: the two MNIST series keep the
    # core colours/markers of panel A, while the noise-task additive series is
    # re-keyed to amber + triangle (with a dashed line as a redundant cue) so
    # no two series in this panel share a hue or a marker.  Each series carries
    # a 95% bootstrap CI, matching every other quantitative panel, and a small
    # x dodge keeps the caps from stacking.
    exact_specs = [
        ("mnist", "dendritic_shunting", CORE_COLOR["dendritic_shunting"],
         CORE_MARKER["dendritic_shunting"], "-", -0.10, "MNIST · shunting"),
        ("mnist", "dendritic_additive", CORE_COLOR["dendritic_additive"],
         CORE_MARKER["dendritic_additive"], "-", 0.0, "MNIST · raw additive"),
        ("noise_resilience", "dendritic_additive", COLORS["local"],
         MARKERS[2], "--", 0.10, "noise · raw additive"),
    ]
    for task, core, color, marker, linestyle, dodge, label in exact_specs:
        part = exact[exact.task.eq(task) & exact.core.eq(core)].sort_values("depth")
        if part.empty:
            continue
        x = part.depth.to_numpy(float) + dodge
        y = 100 * part.mean_difference.to_numpy(float)
        lo = 100 * part.ci95_low.to_numpy(float)
        hi = 100 * part.ci95_high.to_numpy(float)
        ax_c.errorbar(
            x,
            y,
            yerr=np.vstack([y - lo, hi - y]),
            color=color,
            marker=marker,
            ls=linestyle,
            ms=MARKER_MS - 0.6,
            lw=LW_DATA,
            elinewidth=LW_ERR,
            capsize=ERR_CAPSIZE,
            label=label,
        )
    ax_c.axhline(0, color=COLORS["mute"], ls="--", lw=LW_REF)
    ax_c.set_xticks([1, 2, 3, 4])
    ax_c.set_ylim(-0.30, 0.66)
    ax_c.set_yticks([-0.2, 0.0, 0.2, 0.4])
    ax_c.set_xlabel("dendritic stage count")
    ax_c.set_ylabel("exact path − BP (pp)")
    panel_title(ax_c, "I", "Reference exact path–BP")
    style_axis(ax_c)
    clean_legend(ax_c, fontsize=PT_LEGEND, loc="upper right", auto_clear=True)

    positions = []
    labels = []
    for task_index, task in enumerate(("mnist", "noise_resilience")):
        for depth_index, depth in enumerate((2, 4)):
            base = task_index * 2 + depth_index
            labels.append(f"{'MNIST' if task == 'mnist' else 'noise'} D{depth}")
            for core, offset in (("dendritic_shunting", -0.09), ("dendritic_additive", 0.09)):
                selected = routing[
                    routing.task.eq(task)
                    & routing.core.eq(core)
                    & routing.depth.eq(depth)
                ]
                if selected.empty:
                    continue
                row = selected.iloc[0]
                x = base + offset
                mean = 100 * float(row.mean_difference)
                low = 100 * float(row.ci95_low)
                high = 100 * float(row.ci95_high)
                ax_d.errorbar(
                    x,
                    mean,
                    yerr=[[mean - low], [high - mean]],
                    color=CORE_COLOR[core],
                    marker=CORE_MARKER[core],
                    ms=MARKER_MS,
                    lw=LW_ERR,
                    capsize=ERR_CAPSIZE,
                )
                positions.append(x)
    ax_d.axhline(0, color=COLORS["mute"], ls="--", lw=LW_REF)
    ax_d.set_xticks(range(4))
    ax_d.set_xticklabels(wrap_ticklabels(labels, width=6))
    ax_d.tick_params(axis="x", labelsize=PT_SMALL)
    ax_d.set_ylim(-0.12, 1.04)
    ax_d.set_yticks([0.0, 0.4, 0.8])
    # Two-line y label: the single-line form overhangs the panel's grid cell
    # and was truncated when this block is recomposed into Fig. 2/S19.
    ax_d.set_ylabel("correct routing\ngain (pp)")
    panel_title(ax_d, "J", "Matched routing")
    style_axis(ax_d)
    # This panel is recomposed under different letters in Fig. 2 and S19, so
    # a letter-based cross-reference cannot stay correct in both contexts:
    # name the two cores directly in their own hues instead.
    ax_d.text(
        0.03,
        0.99,
        "shunting",
        transform=ax_d.transAxes,
        color=CORE_COLOR["dendritic_shunting"],
        fontsize=PT_SMALL,
        ha="left",
        va="top",
    )
    ax_d.text(
        0.03,
        0.90,
        "raw additive",
        transform=ax_d.transAxes,
        color=CORE_COLOR["dendritic_additive"],
        fontsize=PT_SMALL,
        ha="left",
        va="top",
    )

    clean_exact = clean_exact[clean_exact.depth.eq("all_depths_seed_mean")].copy()
    clean_order = [
        ("mnist", "additive", "MNIST\nraw add."),
        ("mnist", "shunting", "\nshunt."),
        ("noise_resilience", "additive", "noise\nraw add."),
        ("noise_resilience", "shunting", "\nshunt.\n(ReLU)"),
    ]
    for index, (dataset, core, label) in enumerate(clean_order):
        row = clean_exact[
            clean_exact.dataset.eq(dataset) & clean_exact.core.eq(core)
        ].iloc[0]
        mean = 100 * float(row.mean_exact_minus_backpropagation_accuracy)
        low = 100 * float(row.ci95_low)
        high = 100 * float(row.ci95_high)
        ax_e.errorbar(
            index,
            mean,
            yerr=[[mean - low], [high - mean]],
            color=CORE_COLOR[f"dendritic_{core}"],
            marker=CORE_MARKER[f"dendritic_{core}"],
            ms=MARKER_MS,
            lw=LW_ERR,
            capsize=ERR_CAPSIZE,
        )
    ax_e.axhline(0, color=COLORS["mute"], ls="--", lw=LW_REF)
    ax_e.set_xticks(range(4))
    ax_e.set_xticklabels([item[2] for item in clean_order])
    ax_e.tick_params(axis="x", labelsize=PT_SMALL)
    ax_e.set_ylabel("exact path − BP (pp)")
    panel_title(ax_e, "K", "Same-seed exact path–BP")
    style_axis(ax_e)

    fixed = followup_contrast[
        followup_contrast.study.eq("fixed_budget")
        & followup_contrast.contrast.eq("depth 4 - depth 1")
    ].copy()
    feedback_specs = [
        ("per_soma", "scalar fallback"),
        ("per_soma_shared", "neuron-specific"),
        ("path_transport", "exact path"),
        ("backprop", "backprop"),
    ]
    for index, (feedback, _) in enumerate(feedback_specs):
        row = fixed[
            fixed.core.eq("dendritic_additive") & fixed.feedback.eq(feedback)
        ].iloc[0]
        mean = 100 * float(row.mean_difference)
        low = 100 * float(row.ci95_low)
        high = 100 * float(row.ci95_high)
        ax_f.errorbar(
            index,
            mean,
            yerr=[[mean - low], [high - mean]],
            color=CORE_COLOR["dendritic_additive"],
            marker=CORE_MARKER["dendritic_additive"],
            ms=MARKER_MS,
            lw=LW_ERR,
            capsize=ERR_CAPSIZE,
        )
    ax_f.axhline(0, color=COLORS["mute"], ls="--", lw=LW_REF)
    ax_f.set_xticks(range(4))
    ax_f.set_xticklabels(["MW\nfallback", "neuron-\nspecific",
                          "exact\npath", "BP"])
    ax_f.tick_params(axis="x", labelsize=PT_SMALL)
    ax_f.set_ylabel("depth 4 − depth 1 (pp)")
    panel_title(ax_f, "L", "Fixed-budget depth")
    style_axis(ax_f)
    # Single-core panel: direct colour label instead of a legend.
    ax_f.text(
        1.5,
        -4.4,
        "raw additive",
        color=CORE_COLOR["dendritic_additive"],
        fontsize=PT_LEGEND,
        ha="center",
        va="center",
    )
    # The neuron-indexed, exact and BP contrasts carry 95% CIs of only
    # ±0.13-0.19 pp — narrower than the marker itself, so no whiskers can
    # protrude.  The caption states this; an in-panel QA note would restate it.

    # One colour per routing condition, shared by panels G, H and I, following
    # the manuscript-wide routing taxonomy (figs 3/5/8, the subtree factorial
    # and the full-tree extension): red-brown = exact transport (the ceiling —
    # never an adversarial control), green = anatomy-correct routing, rose =
    # broken permutation control ('shuffle'/derangement slot), blue = random
    # rank-K, gray = gated point control.  Names and hues match the subtree
    # full-factorial legend so the two figures can be cross-read.  Panels G
    # and I carry identity on the category axis; inside the H scatter every
    # condition additionally gets the subtree factorial's marker shape (o/s/v,
    # X for the prospective-only neuron-shared condition), which keeps the one
    # OKLab/Machado-marginal pair (neuron amber vs correct green, worst 7.1)
    # unambiguous.
    subtree_colors = {
        "neuron_shared_k1": COLORS["low_rank"],
        "correct_subtree_k2": COLORS["shunting"],
        "within_neuron_deranged_k2": COLORS["highlight"],
        "random_dense_rank2": COLORS["additive"],
        "exact_transport": COLORS["bp"],
        "gated_point_emulation": COLORS["point_mlp"],
    }
    subtree_markers = {
        "neuron_shared_k1": "X",
        "correct_subtree_k2": "o",
        "within_neuron_deranged_k2": "s",
        "random_dense_rank2": "v",
    }
    condition_order = [
        "neuron_shared_k1",
        "correct_subtree_k2",
        "within_neuron_deranged_k2",
        "random_dense_rank2",
    ]
    condition_short = {
        "neuron_shared_k1": "neuron-\nshared",
        "correct_subtree_k2": "correct\nancestry",
        "within_neuron_deranged_k2": "route\nderangement",
        "random_dense_rank2": "random\nrank-2",
        "exact_transport": "exact",
        "gated_point_emulation": "gated\npoint",
    }

    # Panels G and I are horizontal dot plots: condition names sit as
    # horizontal y tick labels (no crowding, no two-row stagger), and the
    # per-seed strip is drawn just above each row's mean diamond so the seeds
    # are never occluded by the mean marker.
    def seeds_and_mean(ax, index, values, color, seed):
        ordered = np.sort(np.asarray(values, dtype=float))
        ax.scatter(
            ordered,
            index - 0.40 + np.linspace(-0.08, 0.08, ordered.size),
            s=SEED_MS**2,
            color=color,
            alpha=SEED_ALPHA,
            edgecolors="none",
            zorder=3,
        )
        mean, low, high = bootstrap_ci(values, seed=seed)
        ax.errorbar(
            mean,
            index,
            xerr=[[mean - low], [high - mean]],
            color=color,
            marker="D",
            markerfacecolor="white",
            ms=MARKER_MS + 1.2,
            lw=LW_ERR,
            elinewidth=LW_ERR,
            capsize=ERR_CAPSIZE,
            zorder=5,
        )

    subtree_conditions = condition_order + ["exact_transport", "gated_point_emulation"]
    for index, condition in enumerate(subtree_conditions):
        values = subtree[subtree.condition.eq(condition)].test_accuracy.to_numpy(float)
        seeds_and_mean(ax_g, index, values, subtree_colors[condition], 42_000 + index)
    ax_g.set_yticks(
        range(len(subtree_conditions)),
        [condition_short[c] for c in subtree_conditions],
    )
    ax_g.tick_params(axis="y", labelsize=PT_SMALL)
    # Consecutive rows carry two-line names; tightened leading keeps adjacent
    # labels from touching at the 6-row pitch.
    for tick_label in ax_g.get_yticklabels():
        tick_label.set_linespacing(1.02)
    ax_g.set_xlabel("held-out accuracy")
    ax_g.set_xlim(0.10, 0.90)
    ax_g.set_xticks([0.2, 0.4, 0.6, 0.8])
    ax_g.set_ylim(len(subtree_conditions) - 0.45, -0.80)
    panel_title(ax_g, "M", "Within-neuron routing")
    style_axis(ax_g, grid="x")
    marker_key = [
        Line2D(
            [0], [0], ls="none", marker="o", ms=SEED_MS + 0.6,
            markerfacecolor=COLORS["mute"], markeredgecolor="none",
            alpha=0.75, label="seed",
        ),
        Line2D(
            [0], [0], ls="none", marker="D", ms=MARKER_MS + 0.6,
            markerfacecolor="white", markeredgecolor=COLORS["mute"],
            markeredgewidth=LW_ERR, label="mean ± 95% CI",
        ),
    ]
    # The key sits in the empty lower-left band, but its gray sample diamond
    # lands at a plausible accuracy on the gated-point row (same hue as that
    # series), so it is framed: a white card with a hairline mute border
    # separates the key from the row lines and from the data field.
    leg_g = clean_legend(
        ax_g,
        handles=marker_key,
        fontsize=PT_SMALL,
        loc="lower left",
        auto_clear=True,
        frameon=True,
        facecolor="white",
        edgecolor=COLORS["mute"],
        framealpha=1.0,
        borderpad=0.55,
    )
    leg_g.get_frame().set_linewidth(LW_HAIR)

    for condition in condition_order:
        part = subtree[subtree.condition.eq(condition)]
        ax_h.scatter(
            part.initial_gradient_scaled_capture,
            part.initial_norm_matched_one_step_progress,
            s=SEED_MS**2,
            color=subtree_colors[condition],
            marker=subtree_markers[condition],
            alpha=0.66,
            edgecolors="none",
            label=condition_short[condition],
        )
    ax_h.axhline(0, color=COLORS["mute"], ls="--", lw=LW_REF)
    # Seed multiplicity (all ten correct-ancestry seeds land exactly on (1, 1);
    # all ten route-derangement seeds stack at capture 0) is stated in the
    # caption rather than as in-panel QA notes.
    ax_h.set_xticks([0.0, 0.5, 1.0])
    ax_h.set_xlabel("exact-gradient capture")
    ax_h.set_ylabel("one-step progress")
    panel_title(ax_h, "N", "Gradient geometry")
    style_axis(ax_h)
    # No in-panel key: an unframed key's sample markers were indistinguishable
    # from the data field (a sample marker reads as one more point on the
    # descending arc), and a framed four-row card cannot fit any empty band of
    # this panel without occluding the arc.  Panel M (canonical lettering)
    # names all four conditions on its rows in exactly these hues, so a mute
    # cross-reference in the clear lower-right corner carries the key; the
    # marker shapes are a redundant cue on top of those hues, repeating the
    # subtree full-factorial figure's shapes for the shared conditions.  This
    # block is published only inside S19, where the key panel is lettered G.
    ax_h.text(
        0.97,
        0.32,
        "colors as in G",
        transform=ax_h.transAxes,
        color=COLORS["mute"],
        fontsize=PT_SMALL,
        ha="right",
        va="bottom",
    )

    for index, condition in enumerate(condition_order):
        values = subtree[
            subtree.condition.eq(condition)
        ].context_switch_forgetting.to_numpy(float)
        seeds_and_mean(ax_i, index, values, subtree_colors[condition], 43_000 + index)
    ax_i.axvline(0, color=COLORS["mute"], ls="--", lw=LW_REF)
    ax_i.set_yticks(
        range(len(condition_order)),
        [condition_short[c] for c in condition_order],
    )
    ax_i.tick_params(axis="y", labelsize=PT_SMALL)
    for tick_label in ax_i.get_yticklabels():
        tick_label.set_linespacing(1.02)
    ax_i.set_xlabel("context-0 forgetting")
    ax_i.set_xlim(-0.06, 0.68)
    ax_i.set_xticks([0.0, 0.2, 0.4, 0.6])
    ax_i.set_ylim(len(condition_order) - 0.45, -0.80)
    panel_title(ax_i, "O", "Switch interference")
    style_axis(ax_i, grid="x")

    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.canvas.draw()
    audit_layout(fig, "fig_prospective_learning_benefits")
    audit_text_over_data(fig, "fig_prospective_learning_benefits")
    fig.savefig(
        FIGURES / "fig_prospective_learning_benefits.pdf",
        metadata={"CreationDate": None, "ModDate": None},
    )
    fig.savefig(FIGURES / "fig_prospective_learning_benefits.png", dpi=600)
    plt.close(fig)


def _plot_three_claim_main() -> None:
    """Archive the superseded three-claim display without replacing Figure 3."""

    contrast = pd.read_csv(SOURCE_DATA / "paired_contrasts.csv")
    routing = pd.read_csv(
        ROOT / "source_data" / "prospective_routing_control" / "paired_contrasts.csv"
    )
    followup_summary = pd.read_csv(
        ROOT / "source_data" / "prospective_followup" / "condition_summary.csv"
    )

    fig, (ax_a, ax_b, ax_c) = plt.subplots(
        1,
        3,
        figsize=(FIG_W, 3.05),
        gridspec_kw={
            "left": 0.082,
            "right": 0.985,
            "bottom": 0.205,
            "top": 0.86,
            "wspace": 0.50,
        },
    )

    gap = contrast[contrast.family.eq("local_minus_bp")].copy()
    feedback_specs = [
        ("per_soma - backprop", "matched-width fallback", COLORS["scalar"]),
        ("per_soma_shared - backprop", "neuron-specific", COLORS["per_soma"]),
        ("path_transport - backprop", "exact path", COLORS["oracle"]),
    ]
    for contrast_name, label, color in feedback_specs:
        for task, marker, linestyle in [
            ("mnist", "o", "-"),
            ("noise_resilience", "s", "--"),
        ]:
            part = (
                gap[gap.contrast.eq(contrast_name) & gap.task.eq(task)]
                .groupby("depth", as_index=False)["mean_difference"]
                .mean()
                .sort_values("depth")
            )
            task_label = "MNIST" if task == "mnist" else "noise"
            ax_a.plot(
                part.depth,
                100 * part.mean_difference,
                color=color,
                marker=marker,
                linestyle=linestyle,
                ms=3.7,
                lw=LW_DATA,
                label=f"{label} · {task_label}",
            )
    ax_a.axhline(0, color=COLORS["mute"], ls=":", lw=LW_REF)
    ax_a.set_xticks([1, 2, 3, 4])
    ax_a.set_xlabel("dendritic stage count")
    ax_a.set_ylabel("local - backprop (pp)")
    panel_title(ax_a, "A", "Gap to backpropagation")
    style_axis(ax_a)
    hierarchy_handles = [
        Line2D([0], [0], color=COLORS["scalar"], lw=LW_DATA, label="matched-width fallback"),
        Line2D([0], [0], color=COLORS["per_soma"], lw=LW_DATA, label="neuron-specific"),
        Line2D([0], [0], color=COLORS["oracle"], lw=LW_DATA, label="exact path"),
        Line2D([0], [0], color=COLORS["ink"], marker="o", lw=LW_DATA, label="MNIST"),
        Line2D([0], [0], color=COLORS["ink"], marker="s", ls="--", lw=LW_DATA, label="noise task"),
    ]
    clean_legend(
        ax_a,
        handles=hierarchy_handles,
        fontsize=PT_LEGEND - 0.8,
        loc="center",
        bbox_to_anchor=(0.53, 0.69),
        ncol=2,
    )

    labels = []
    for task_index, task in enumerate(("mnist", "noise_resilience")):
        for depth_index, depth in enumerate((2, 4)):
            base = task_index * 2 + depth_index
            labels.append(f"{'MNIST' if task == 'mnist' else 'noise'}\nD{depth}")
            for core, offset in (("dendritic_shunting", -0.09), ("dendritic_additive", 0.09)):
                row = routing[
                    routing.task.eq(task)
                    & routing.core.eq(core)
                    & routing.depth.eq(depth)
                ].iloc[0]
                mean = 100 * float(row.mean_difference)
                low = 100 * float(row.ci95_low)
                high = 100 * float(row.ci95_high)
                ax_b.errorbar(
                    base + offset,
                    mean,
                    yerr=[[mean - low], [high - mean]],
                    color=CORE_COLOR[core],
                    marker="o",
                    ms=4.3,
                    lw=LW_ERR,
                    capsize=ERR_CAPSIZE,
                )
    ax_b.axhline(0, color=COLORS["mute"], ls=":", lw=LW_REF)
    ax_b.set_xticks(range(4), labels)
    ax_b.set_ylabel("correct ownership gain (pp)")
    panel_title(ax_b, "B", "Ownership control")
    style_axis(ax_b)

    dose = followup_summary[followup_summary.family.eq("inhibition")].copy()
    dose_specs = [
        ("local_ca", "per_soma", "matched-width fallback", COLORS["scalar"], "o"),
        ("local_ca", "per_soma_shared", "neuron-specific", COLORS["per_soma"], "s"),
        ("local_ca", "path_transport", "exact path", COLORS["oracle"], "^"),
        ("standard", "backprop", "backprop", COLORS["bp"], "D"),
    ]
    for strategy, feedback, label, color, marker in dose_specs:
        part = dose[dose.strategy.eq(strategy) & dose.feedback.eq(feedback)]
        pivot = part.pivot(
            index="inhibitory_synapses_per_branch",
            columns="core",
            values="mean_accuracy",
        ).sort_index()
        effect = 100 * (pivot["dendritic_shunting"] - pivot["dendritic_additive"])
        ax_c.plot(
            effect.index,
            effect.values,
            color=color,
            marker=marker,
            ms=3.8,
            lw=LW_DATA,
            label=label,
        )
    ax_c.axhline(0, color=COLORS["mute"], ls=":", lw=LW_REF)
    ax_c.set_xticks([0, 10, 20, 40])
    ax_c.set_xlabel("inhibitory contacts / branch")
    ax_c.set_ylabel("shunting - additive (pp)")
    panel_title(ax_c, "C", "Inhibitory dose")
    style_axis(ax_c)
    clean_legend(ax_c, fontsize=PT_LEGEND, loc="lower right")

    superseded = ANALYSIS / "superseded"
    superseded.mkdir(parents=True, exist_ok=True)
    fig.canvas.draw()
    audit_layout(fig, "fig_prospective_three_claim_historical_excluded")
    audit_text_over_data(fig, "fig_prospective_three_claim_historical_excluded")
    fig.savefig(
        superseded / "fig_prospective_three_claim_historical_excluded.pdf",
        metadata={"CreationDate": None, "ModDate": None},
    )
    fig.savefig(
        superseded / "fig_prospective_three_claim_historical_excluded.png",
        dpi=600,
    )
    plt.close(fig)


def plot(frame: pd.DataFrame, summary: pd.DataFrame) -> None:
    _plot_streamlined_main()


def report(summary: pd.DataFrame, contrast: pd.DataFrame) -> str:
    conditioning = contrast[contrast.family == "feedback_conditioning"].copy()
    neuron_indexed = contrast[
        (contrast.family == "feedback")
        & (contrast.contrast == "neuron-indexed - scalar")
    ].copy()
    lines = [
        "# Historical full-cohort prospective artifact summary",
        "",
        "> **Not for publication inference.** This complete artifact summary predates the outcome-independent input-validity audit. Publication results use `source_data/prospective_input_validity/`; every signed-noise positive-conductance shunting condition below is excluded.",
        "",
        "Every frozen execution passed the historical artifact audit. Artificial-network seeds are paired and are the inferential unit.",
        "",
        "## Historical prespecified contrasts (excluded where they require signed shunting)",
        "",
    ]
    for _, row in conditioning.sort_values(["task", "depth"]).iterrows():
        lines.append(
            f"- {TASK_LABEL[row.task]}, depth {int(row.depth)}: the shunting-minus-additive "
            f"advantage under scalar rather than exact feedback was "
            f"{100 * row.mean_difference:.2f} percentage points "
            f"(95% paired bootstrap CI {100 * row.ci95_low:.2f} to "
            f"{100 * row.ci95_high:.2f}; {int(row.wins)}/{int(row.n_pairs)} pairs; "
            f"exact Wilcoxon P={row.wilcoxon_p_two_sided:.4g})."
        )
    lines += ["", "## Value of neuron-indexed feedback", ""]
    for _, row in neuron_indexed.sort_values(["task", "core", "depth"]).iterrows():
        lines.append(
            f"- {TASK_LABEL[row.task]}, {CORE_LABEL[row.core]}, depth {int(row.depth)}: "
            f"{100 * row.mean_difference:.2f} percentage points "
            f"(95% CI {100 * row.ci95_low:.2f} to {100 * row.ci95_high:.2f})."
        )
    lines += [
        "",
        "## Interpretation guardrails",
        "",
        "Depth changes the number of compartments and is not a parameter-matched morphology comparison. "
        "The current paper excludes all signed-noise shunting contrasts regardless of their outcome. The one-seed "
        "canary is excluded from these historical summaries.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--audit-csv",
        type=Path,
        default=ANALYSIS / "prospective_primary_confirmatory_audit.csv",
    )
    parser.add_argument("--expected-seeds", type=int, default=10)
    args = parser.parse_args()

    frame = validate(pd.read_csv(args.audit_csv), args.expected_seeds)
    summary = condition_summary(frame)
    contrast = contrasts(frame)

    SOURCE_DATA.mkdir(parents=True, exist_ok=True)
    frame.to_csv(SOURCE_DATA / "seed_outcomes.csv", index=False)
    summary.to_csv(SOURCE_DATA / "condition_summary.csv", index=False)
    contrast.to_csv(SOURCE_DATA / "paired_contrasts.csv", index=False)
    (ANALYSIS / "prospective_primary_confirmatory_results.md").write_text(
        report(summary, contrast)
    )
    # Rebuild the validity-qualified checkpoint-mechanism figure. The first
    # display produced by this legacy renderer is archived under
    # analysis/superseded and cannot overwrite the current Figure 3.
    _plot_legacy(frame, summary)
    plot(frame, summary)


if __name__ == "__main__":
    main()
