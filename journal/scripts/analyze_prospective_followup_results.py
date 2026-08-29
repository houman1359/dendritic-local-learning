#!/usr/bin/env python3
"""Analyze the frozen topology, inhibition-dose, and fixed-budget cohorts.

Both audit tables are required.  The script refuses to summarize an incomplete,
failed, duplicated, or unbalanced confirmatory cohort.  Seeds are paired for
every contrast and the one-seed canaries are never read.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from neurips_style import (
    COLORS,
    ERR_CAPSIZE,
    FIG_W,
    LW_DATA,
    LW_ERR,
    LW_REF,
    PT_LEGEND,
    apply_neurips_style,
    audit_layout,
    audit_text_over_data,
    clean_legend,
    panel_title,
    style_axis,
)


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "analysis"
SOURCE_DATA = ROOT / "source_data" / "prospective_followup"
FIGURES = ROOT / "figures" / "generated"

CORE_ORDER = ["dendritic_shunting", "dendritic_additive"]
CORE_LABEL = {
    "dendritic_shunting": "Shunting",
    "dendritic_additive": "Additive",
}
CORE_COLOR = {
    "dendritic_shunting": COLORS["shunting"],
    "dendritic_additive": COLORS["additive"],
}
FEEDBACK_ORDER = ["per_soma", "per_soma_shared", "path_transport"]
FEEDBACK_LABEL = {
    "per_soma": "Legacy fallback",
    "per_soma_shared": "Neuron-indexed",
    "per_soma_shuffled": "Shuffled ancestry",
    "path_transport": "Exact transport",
    "backprop": "Backpropagation",
}
TASK_LABEL = {"mnist": "MNIST", "noise_resilience": "Noise task"}
TOPOLOGY_LABEL = {"spatial": "Spatial", "random": "Random"}
TOPOLOGY_STYLE = {"spatial": "-", "random": "--"}

apply_neurips_style()


def bootstrap_ci(values: np.ndarray, *, seed: int, n_boot: int = 20_000):
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        raise ValueError("Cannot bootstrap an empty contrast")
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
            values,
            zero_method="wilcox",
            correction=False,
            alternative="two-sided",
        ).pvalue
    )


def benjamini_hochberg(values: pd.Series) -> pd.Series:
    """Return monotone Benjamini-Hochberg adjusted P values."""
    p_values = values.to_numpy(dtype=float)
    order = np.argsort(p_values)
    ranked = p_values[order]
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    restored = np.empty_like(adjusted)
    restored[order] = np.clip(adjusted, 0.0, 1.0)
    return pd.Series(restored, index=values.index)


def _family(run_dir: str) -> str:
    if "inhibition_dose" in run_dir:
        return "inhibition"
    if "spatial_topology" in run_dir:
        return "spatial"
    if "ancestry_routing" in run_dir:
        return "routing"
    if "fixed_budget_depth" in run_dir:
        return "fixed_budget"
    raise ValueError(f"Unknown prospective family: {run_dir}")


def _require_columns(frame: pd.DataFrame) -> None:
    required = {
        "run_dir",
        "status",
        "task",
        "core",
        "strategy",
        "feedback",
        "routing",
        "topology",
        "depth",
        "branch_factors",
        "non_somatic_compartments_per_soma",
        "inhibitory_synapses_per_branch",
        "seed",
        "test_accuracy",
        "total_parameters",
        "active_synapses",
        "duration_seconds",
        "cuda_peak_memory_allocated_bytes",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise SystemExit(
            "Audit table predates the required stage-resolved metadata; rerun the "
            f"audit. Missing columns: {', '.join(missing)}"
        )


def validate(
    followup: pd.DataFrame,
    fixed: pd.DataFrame,
    expected_seeds: int,
) -> pd.DataFrame:
    frame = pd.concat([followup, fixed], ignore_index=True)
    _require_columns(frame)
    if len(followup) != 880 or len(fixed) != 320:
        raise SystemExit(
            f"Unexpected cohort sizes: followup={len(followup)} (expected 880), "
            f"fixed_budget={len(fixed)} (expected 320)"
        )
    failed = frame[frame.status != "pass"]
    if not failed.empty:
        raise SystemExit(
            f"Refusing partial analysis: {len(failed)} rows did not pass the audit"
        )
    frame = frame.copy()
    frame["family"] = frame.run_dir.map(_family)
    family_sizes = frame.family.value_counts().to_dict()
    expected_sizes = {
        "inhibition": 400,
        "spatial": 320,
        "routing": 160,
        "fixed_budget": 320,
    }
    if family_sizes != expected_sizes:
        raise SystemExit(
            f"Unexpected family sizes: {family_sizes}; expected {expected_sizes}"
        )

    condition_key = [
        "family",
        "task",
        "core",
        "strategy",
        "feedback",
        "routing",
        "topology",
        "depth",
        "branch_factors",
        "inhibitory_synapses_per_branch",
    ]
    if frame.duplicated([*condition_key, "seed"]).any():
        duplicates = frame[frame.duplicated([*condition_key, "seed"], keep=False)]
        raise SystemExit(f"Duplicate condition-by-seed rows:\n{duplicates.to_string()}")
    counts = frame.groupby(condition_key, dropna=False).seed.nunique()
    bad = counts[counts != expected_seeds]
    if not bad.empty:
        raise SystemExit(f"Unbalanced seed counts:\n{bad.to_string()}")
    if not np.isfinite(frame.test_accuracy.to_numpy(dtype=float)).all():
        raise SystemExit("Non-finite test accuracy survived the artifact audit")
    return frame


def validate_followup_only(
    followup: pd.DataFrame,
    expected_seeds: int,
) -> pd.DataFrame:
    """Validate the complete 880-run follow-up independently of fixed depth."""
    _require_columns(followup)
    if len(followup) != 880:
        raise SystemExit(
            f"Unexpected follow-up size: {len(followup)} (expected 880)"
        )
    failed = followup[followup.status != "pass"]
    if not failed.empty:
        raise SystemExit(
            f"Refusing partial analysis: {len(failed)} follow-up rows did not pass"
        )
    frame = followup.copy()
    frame["family"] = frame.run_dir.map(_family)
    expected_sizes = {"inhibition": 400, "spatial": 320, "routing": 160}
    if frame.family.value_counts().to_dict() != expected_sizes:
        raise SystemExit(
            f"Unexpected follow-up family sizes: {frame.family.value_counts().to_dict()}"
        )
    condition_key = [
        "family", "task", "core", "strategy", "feedback", "routing",
        "topology", "depth", "branch_factors", "inhibitory_synapses_per_branch",
    ]
    if frame.duplicated([*condition_key, "seed"]).any():
        raise SystemExit("Duplicate follow-up condition-by-seed rows")
    counts = frame.groupby(condition_key, dropna=False).seed.nunique()
    if not counts.eq(expected_seeds).all():
        raise SystemExit(f"Unbalanced follow-up seed counts:\n{counts.to_string()}")
    if not np.isfinite(frame.test_accuracy.to_numpy(dtype=float)).all():
        raise SystemExit("Non-finite test accuracy survived the follow-up audit")
    return frame


def select(frame: pd.DataFrame, spec: dict[str, object], label: str) -> pd.DataFrame:
    part = frame
    for column, value in spec.items():
        part = part[part[column] == value]
    if part.empty:
        raise ValueError(f"Empty selection for {spec}")
    if part.seed.duplicated().any():
        raise ValueError(f"Selection is not one row per seed: {spec}")
    return part[["seed", "test_accuracy"]].rename(columns={"test_accuracy": label})


def paired(frame: pd.DataFrame, left: dict[str, object], right: dict[str, object]):
    out = select(frame, left, "left").merge(
        select(frame, right, "right"), on="seed", validate="one_to_one"
    )
    out["difference"] = out.left - out.right
    return out


def add_contrast(
    rows: list[dict[str, object]],
    *,
    study: str,
    contrast: str,
    values: np.ndarray,
    seed: int,
    **metadata: object,
) -> None:
    values = np.asarray(values, dtype=float)
    mean, lo, hi = bootstrap_ci(values, seed=seed)
    rows.append(
        {
            "study": study,
            "contrast": contrast,
            **metadata,
            "n_pairs": values.size,
            "mean_difference": mean,
            "ci95_low": lo,
            "ci95_high": hi,
            "wins": int((values > 0).sum()),
            "ties": int(np.isclose(values, 0.0).sum()),
            "wilcoxon_p_two_sided": paired_test(values),
        }
    )


def summaries(frame: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "family",
        "task",
        "core",
        "strategy",
        "feedback",
        "routing",
        "topology",
        "depth",
        "branch_factors",
        "non_somatic_compartments_per_soma",
        "inhibitory_synapses_per_branch",
    ]
    rows: list[dict[str, object]] = []
    for values, part in frame.groupby(keys, dropna=False, sort=True):
        mean, lo, hi = bootstrap_ci(part.test_accuracy.to_numpy(), seed=len(rows))
        rows.append(
            {
                **dict(zip(keys, values)),
                "n_seeds": part.seed.nunique(),
                "mean_accuracy": mean,
                "ci95_low": lo,
                "ci95_high": hi,
                "total_parameters": int(part.total_parameters.iloc[0]),
                "active_synapses": int(part.active_synapses.iloc[0]),
                "mean_duration_seconds": float(part.duration_seconds.mean()),
                "mean_peak_memory_bytes": float(
                    part.cuda_peak_memory_allocated_bytes.mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def inhibition_contrasts(frame: pd.DataFrame, rows: list[dict[str, object]]) -> None:
    part = frame[frame.family == "inhibition"]
    counter = 0
    for core in CORE_ORDER:
        modes = [("standard", "backprop"), *[("local_ca", f) for f in FEEDBACK_ORDER]]
        for strategy, feedback in modes:
            effect = paired(
                part,
                {
                    "core": core,
                    "strategy": strategy,
                    "feedback": feedback,
                    "inhibitory_synapses_per_branch": 40,
                },
                {
                    "core": core,
                    "strategy": strategy,
                    "feedback": feedback,
                    "inhibitory_synapses_per_branch": 0,
                },
            )
            add_contrast(
                rows,
                study="inhibition",
                contrast="dose 40 - dose 0",
                values=effect.difference.to_numpy(),
                seed=counter,
                task="noise_resilience",
                core=core,
                strategy=strategy,
                feedback=feedback,
            )
            counter += 1

    for feedback in ["backprop", *FEEDBACK_ORDER]:
        strategy = "standard" if feedback == "backprop" else "local_ca"
        at_40 = paired(
            part,
            {
                "core": "dendritic_shunting",
                "strategy": strategy,
                "feedback": feedback,
                "inhibitory_synapses_per_branch": 40,
            },
            {
                "core": "dendritic_additive",
                "strategy": strategy,
                "feedback": feedback,
                "inhibitory_synapses_per_branch": 40,
            },
        )
        at_0 = paired(
            part,
            {
                "core": "dendritic_shunting",
                "strategy": strategy,
                "feedback": feedback,
                "inhibitory_synapses_per_branch": 0,
            },
            {
                "core": "dendritic_additive",
                "strategy": strategy,
                "feedback": feedback,
                "inhibitory_synapses_per_branch": 0,
            },
        )
        values = at_40.set_index("seed").difference - at_0.set_index("seed").difference
        add_contrast(
            rows,
            study="inhibition",
            contrast="core by dose interaction",
            values=values.to_numpy(),
            seed=counter,
            task="noise_resilience",
            core="shunting - additive",
            strategy=strategy,
            feedback=feedback,
        )
        counter += 1

    for core in CORE_ORDER:
        for dose in (0, 5, 10, 20, 40):
            scalar = paired(
                part,
                {
                    "core": core,
                    "strategy": "local_ca",
                    "feedback": "per_soma",
                    "inhibitory_synapses_per_branch": dose,
                },
                {
                    "core": core,
                    "strategy": "local_ca",
                    "feedback": "path_transport",
                    "inhibitory_synapses_per_branch": dose,
                },
            )
            add_contrast(
                rows,
                study="inhibition",
                contrast="scalar - exact",
                values=scalar.difference.to_numpy(),
                seed=counter,
                task="noise_resilience",
                core=core,
                strategy="local_ca",
                feedback="per_soma - path_transport",
                dose=dose,
            )
            counter += 1


def spatial_contrasts(frame: pd.DataFrame, rows: list[dict[str, object]]) -> None:
    part = frame[frame.family == "spatial"]
    counter = 100
    for task in ("mnist", "noise_resilience"):
        for core in CORE_ORDER:
            modes = [
                ("standard", "backprop"),
                *[("local_ca", feedback) for feedback in FEEDBACK_ORDER],
            ]
            for strategy, feedback in modes:
                effect = paired(
                    part,
                    {
                        "task": task,
                        "core": core,
                        "strategy": strategy,
                        "feedback": feedback,
                        "topology": "spatial",
                    },
                    {
                        "task": task,
                        "core": core,
                        "strategy": strategy,
                        "feedback": feedback,
                        "topology": "random",
                    },
                )
                add_contrast(
                    rows,
                    study="spatial",
                    contrast="spatial - random",
                    values=effect.difference.to_numpy(),
                    seed=counter,
                    task=task,
                    core=core,
                    strategy=strategy,
                    feedback=feedback,
                )
                counter += 1

    for core in CORE_ORDER:
        for strategy, feedback in [
            ("standard", "backprop"),
            *[("local_ca", mode) for mode in FEEDBACK_ORDER],
        ]:
            effects = {}
            for task in ("mnist", "noise_resilience"):
                effects[task] = (
                    paired(
                        part,
                        {
                            "task": task,
                            "core": core,
                            "strategy": strategy,
                            "feedback": feedback,
                            "topology": "spatial",
                        },
                        {
                            "task": task,
                            "core": core,
                            "strategy": strategy,
                            "feedback": feedback,
                            "topology": "random",
                        },
                    )
                    .set_index("seed")
                    .difference
                )
            interaction = effects["mnist"] - effects["noise_resilience"]
            add_contrast(
                rows,
                study="spatial",
                contrast="topology by task interaction",
                values=interaction.to_numpy(),
                seed=counter,
                task="MNIST - noise_resilience",
                core=core,
                strategy=strategy,
                feedback=feedback,
            )
            counter += 1


def routing_contrasts(frame: pd.DataFrame, rows: list[dict[str, object]]) -> None:
    part = frame[frame.family == "routing"]
    counter = 200
    for task in ("mnist", "noise_resilience"):
        for core in CORE_ORDER:
            effects = {}
            for depth in (2, 4):
                effects[depth] = (
                    paired(
                        part,
                        {
                            "task": task,
                            "core": core,
                            "depth": depth,
                            "routing": "correct",
                        },
                        {
                            "task": task,
                            "core": core,
                            "depth": depth,
                            "routing": "shuffled",
                        },
                    )
                    .set_index("seed")
                    .difference
                )
                add_contrast(
                    rows,
                    study="routing",
                    contrast="correct - shuffled ancestry",
                    values=effects[depth].to_numpy(),
                    seed=counter,
                    task=task,
                    core=core,
                    strategy="local_ca",
                    feedback="matched-bandwidth ancestry",
                    depth=depth,
                )
                counter += 1
            add_contrast(
                rows,
                study="routing",
                contrast="routing by depth interaction",
                values=(effects[4] - effects[2]).to_numpy(),
                seed=counter,
                task=task,
                core=core,
                strategy="local_ca",
                feedback="matched-bandwidth ancestry",
                depth="4 - 2",
            )
            counter += 1


def fixed_budget_contrasts(frame: pd.DataFrame, rows: list[dict[str, object]]) -> None:
    part = frame[frame.family == "fixed_budget"]
    counter = 300
    for core in CORE_ORDER:
        for strategy, feedback in [
            ("standard", "backprop"),
            *[("local_ca", mode) for mode in FEEDBACK_ORDER],
        ]:
            effect = paired(
                part,
                {"core": core, "strategy": strategy, "feedback": feedback, "depth": 4},
                {"core": core, "strategy": strategy, "feedback": feedback, "depth": 1},
            )
            add_contrast(
                rows,
                study="fixed_budget",
                contrast="depth 4 - depth 1",
                values=effect.difference.to_numpy(),
                seed=counter,
                task="noise_resilience",
                core=core,
                strategy=strategy,
                feedback=feedback,
            )
            counter += 1

            wide = part[
                (part.core == core)
                & (part.strategy == strategy)
                & (part.feedback == feedback)
            ].pivot(index="seed", columns="depth", values="test_accuracy")
            slopes = np.polyfit(np.arange(1, 5), wide[[1, 2, 3, 4]].to_numpy().T, 1)[0]
            add_contrast(
                rows,
                study="fixed_budget",
                contrast="linear accuracy change per depth",
                values=slopes,
                seed=counter,
                task="noise_resilience",
                core=core,
                strategy=strategy,
                feedback=feedback,
            )
            counter += 1

    for feedback in ["backprop", *FEEDBACK_ORDER]:
        strategy = "standard" if feedback == "backprop" else "local_ca"
        effect = {}
        for core in CORE_ORDER:
            effect[core] = (
                paired(
                    part,
                    {
                        "core": core,
                        "strategy": strategy,
                        "feedback": feedback,
                        "depth": 4,
                    },
                    {
                        "core": core,
                        "strategy": strategy,
                        "feedback": feedback,
                        "depth": 1,
                    },
                )
                .set_index("seed")
                .difference
            )
        interaction = effect["dendritic_shunting"] - effect["dendritic_additive"]
        add_contrast(
            rows,
            study="fixed_budget",
            contrast="depth by core interaction",
            values=interaction.to_numpy(),
            seed=counter,
            task="noise_resilience",
            core="shunting - additive",
            strategy=strategy,
            feedback=feedback,
        )
        counter += 1

    for core in CORE_ORDER:
        for depth in (1, 2, 3, 4):
            for feedback in FEEDBACK_ORDER:
                gap = paired(
                    part,
                    {
                        "core": core,
                        "strategy": "local_ca",
                        "feedback": feedback,
                        "depth": depth,
                    },
                    {
                        "core": core,
                        "strategy": "standard",
                        "feedback": "backprop",
                        "depth": depth,
                    },
                )
                add_contrast(
                    rows,
                    study="fixed_budget",
                    contrast="local - backprop",
                    values=gap.difference.to_numpy(),
                    seed=counter,
                    task="noise_resilience",
                    core=core,
                    strategy="local_ca - backprop",
                    feedback=feedback,
                    depth=depth,
                )
                counter += 1


def all_contrasts(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    inhibition_contrasts(frame, rows)
    spatial_contrasts(frame, rows)
    routing_contrasts(frame, rows)
    fixed_budget_contrasts(frame, rows)
    result = pd.DataFrame(rows)
    result["fdr_bh_within_study"] = result.groupby("study", group_keys=False)[
        "wilcoxon_p_two_sided"
    ].transform(benjamini_hochberg)
    return result


def _errorbar_line(ax, part: pd.DataFrame, x: str, *, color: str, label: str, ls="-"):
    part = part.sort_values(x)
    xx = part[x].to_numpy(dtype=float)
    yy = part.mean_accuracy.to_numpy(dtype=float)
    lo = part.ci95_low.to_numpy(dtype=float)
    hi = part.ci95_high.to_numpy(dtype=float)
    ax.errorbar(
        xx,
        yy,
        yerr=np.vstack([yy - lo, hi - yy]),
        color=color,
        label=label,
        ls=ls,
        marker="o",
        ms=4.0,
        lw=LW_DATA,
        elinewidth=LW_ERR,
        capsize=ERR_CAPSIZE,
    )


def plot_inhibition(summary: pd.DataFrame, contrasts: pd.DataFrame) -> None:
    data = summary[summary.family == "inhibition"]
    fig, axes = plt.subplots(
        2,
        3,
        figsize=(FIG_W, 5.4),
        gridspec_kw={
            "left": 0.10,
            "right": 0.95,
            "bottom": 0.13,
            "top": 0.91,
            "wspace": 0.68,
            "hspace": 0.68,
        },
    )
    for ax, letter, feedback in zip(axes[0], "ABC", FEEDBACK_ORDER):
        for core in CORE_ORDER:
            _errorbar_line(
                ax,
                data[
                    (data.strategy == "local_ca")
                    & (data.feedback == feedback)
                    & (data.core == core)
                ],
                "inhibitory_synapses_per_branch",
                color=CORE_COLOR[core],
                label=CORE_LABEL[core],
            )
        panel_title(ax, letter, FEEDBACK_LABEL[feedback])
        ax.set_xlabel("inhibitory contacts")
        if ax is axes[0, 0]:
            ax.set_ylabel("test accuracy")
        style_axis(ax)
    clean_legend(axes[0, 2], fontsize=PT_LEGEND, loc="best")

    ax = axes[1, 0]
    for core in CORE_ORDER:
        _errorbar_line(
            ax,
            data[(data.strategy == "standard") & (data.core == core)],
            "inhibitory_synapses_per_branch",
            color=CORE_COLOR[core],
            label=CORE_LABEL[core],
        )
    panel_title(ax, "D", "Backpropagation")
    ax.set_xlabel("inhibitory contacts")
    ax.set_ylabel("test accuracy")
    style_axis(ax)

    ax = axes[1, 1]
    gap = contrasts[
        (contrasts.study == "inhibition") & (contrasts.contrast == "scalar - exact")
    ]
    for core in CORE_ORDER:
        part = gap[gap.core == core].sort_values("dose")
        ax.errorbar(
            part.dose,
            100 * part.mean_difference,
            yerr=np.vstack(
                [
                    100 * (part.mean_difference - part.ci95_low),
                    100 * (part.ci95_high - part.mean_difference),
                ]
            ),
            color=CORE_COLOR[core],
            marker="o",
            ms=4.0,
            lw=LW_DATA,
            elinewidth=LW_ERR,
            capsize=ERR_CAPSIZE,
            label=CORE_LABEL[core],
        )
    ax.axhline(0, color=COLORS["mute"], lw=LW_REF, ls="--")
    panel_title(ax, "E", "Restricted-feedback gap")
    ax.set_xlabel("inhibitory contacts")
    ax.set_ylabel("scalar - exact (pp)")
    style_axis(ax)

    ax = axes[1, 2]
    interaction = contrasts[
        (contrasts.study == "inhibition")
        & (contrasts.contrast == "core by dose interaction")
    ].copy()
    order = ["backprop", *FEEDBACK_ORDER]
    interaction["order"] = interaction.feedback.map({v: i for i, v in enumerate(order)})
    interaction = interaction.sort_values("order")
    x = np.arange(len(interaction))
    y = 100 * interaction.mean_difference.to_numpy()
    ax.errorbar(
        x,
        y,
        yerr=np.vstack(
            [
                100 * (interaction.mean_difference - interaction.ci95_low),
                100 * (interaction.ci95_high - interaction.mean_difference),
            ]
        ),
        fmt="o",
        color=COLORS["highlight"],
        ms=4.4,
        lw=LW_ERR,
        capsize=ERR_CAPSIZE,
    )
    ax.axhline(0, color=COLORS["mute"], lw=LW_REF, ls="--")
    ax.set_xticks(x, ["BP", "Scalar", "Ancestry", "Exact"])
    panel_title(ax, "F", "Dose by core")
    ax.set_ylabel("interaction (pp)")
    style_axis(ax)

    superseded = ANALYSIS / "superseded"
    superseded.mkdir(parents=True, exist_ok=True)
    fig.canvas.draw()
    audit_layout(fig, "fig_prospective_inhibition_dose_historical_excluded")
    audit_text_over_data(fig, "fig_prospective_inhibition_dose_historical_excluded")
    fig.savefig(
        superseded / "fig_prospective_inhibition_dose_historical_excluded.pdf",
        metadata={"CreationDate": None, "ModDate": None},
    )
    fig.savefig(
        superseded / "fig_prospective_inhibition_dose_historical_excluded.png",
        dpi=350,
    )
    plt.close(fig)


def plot_topology(summary: pd.DataFrame, contrasts: pd.DataFrame) -> None:
    spatial = summary[summary.family == "spatial"]
    routing = summary[summary.family == "routing"]
    fig, axes = plt.subplots(
        3,
        3,
        figsize=(FIG_W, 7.7),
        gridspec_kw={
            "left": 0.10,
            "right": 0.95,
            "bottom": 0.08,
            "top": 0.82,
            "wspace": 0.72,
            "hspace": 0.70,
        },
    )
    letters = iter("ABCDEFGHI")
    feedback_x = np.arange(3)
    for col, task in enumerate(("mnist", "noise_resilience")):
        ax = axes[0, col]
        for core, offset in zip(CORE_ORDER, (-0.08, 0.08)):
            for topology in ("spatial", "random"):
                part = (
                    spatial[
                        (spatial.task == task)
                        & (spatial.core == core)
                        & (spatial.strategy == "local_ca")
                        & (spatial.topology == topology)
                    ]
                    .set_index("feedback")
                    .loc[FEEDBACK_ORDER]
                )
                ax.plot(
                    feedback_x + offset,
                    part.mean_accuracy,
                    color=CORE_COLOR[core],
                    ls=TOPOLOGY_STYLE[topology],
                    marker="o" if topology == "spatial" else "s",
                    ms=3.6,
                    lw=LW_DATA,
                    label=f"{CORE_LABEL[core]}, {TOPOLOGY_LABEL[topology].lower()}",
                )
        ax.set_xticks(
            feedback_x,
            ["Scalar", "Ancestry", "Exact"],
        )
        panel_title(ax, next(letters), f"{TASK_LABEL[task]} topology")
        if col == 0:
            ax.set_ylabel("test accuracy")
        style_axis(ax)

    ax = axes[0, 2]
    effects = contrasts[
        (contrasts.study == "spatial")
        & (contrasts.contrast == "spatial - random")
        & (contrasts.strategy == "local_ca")
    ]
    positions = np.arange(3)
    for task, task_offset, marker in (
        ("mnist", -0.12, "o"),
        ("noise_resilience", 0.12, "s"),
    ):
        for core, core_offset in zip(CORE_ORDER, (-0.035, 0.035)):
            part = (
                effects[(effects.task == task) & (effects.core == core)]
                .set_index("feedback")
                .loc[FEEDBACK_ORDER]
            )
            ax.plot(
                positions + task_offset + core_offset,
                100 * part.mean_difference,
                color=CORE_COLOR[core],
                marker=marker,
                ms=4.0,
                lw=LW_DATA,
                label=f"{TASK_LABEL[task]}, {CORE_LABEL[core].lower()}",
            )
    ax.axhline(0, color=COLORS["mute"], lw=LW_REF, ls="--")
    ax.set_xticks(positions, ["Scalar", "Ancestry", "Exact"])
    ax.set_ylabel("spatial - random (pp)")
    panel_title(ax, next(letters), "Topology effect")
    style_axis(ax)

    ax = axes[1, 0]
    bp = contrasts[
        (contrasts.study == "spatial")
        & (contrasts.contrast == "spatial - random")
        & (contrasts.strategy == "standard")
    ]
    x = np.arange(4)
    ordered = []
    labels = []
    for task in ("mnist", "noise_resilience"):
        for core in CORE_ORDER:
            ordered.append(bp[(bp.task == task) & (bp.core == core)].iloc[0])
            labels.append(f"{TASK_LABEL[task]}\n{CORE_LABEL[core]}")
    vals = np.array([row.mean_difference for row in ordered]) * 100
    los = np.array([row.ci95_low for row in ordered]) * 100
    his = np.array([row.ci95_high for row in ordered]) * 100
    ax.errorbar(
        x,
        vals,
        yerr=np.vstack([vals - los, his - vals]),
        fmt="o",
        color=COLORS["bp"],
        lw=LW_ERR,
        capsize=ERR_CAPSIZE,
    )
    ax.axhline(0, color=COLORS["mute"], lw=LW_REF, ls="--")
    ax.set_xticks(x, labels)
    ax.set_ylabel("spatial - random (pp)")
    panel_title(ax, next(letters), "BP topology control")
    style_axis(ax)

    for col, task in enumerate(("mnist", "noise_resilience")):
        ax = axes[1, col + 1]
        for core in CORE_ORDER:
            for routing_label, linestyle in (("correct", "-"), ("shuffled", "--")):
                part = routing[
                    (routing.task == task)
                    & (routing.core == core)
                    & (routing.routing == routing_label)
                ].sort_values("depth")
                ax.plot(
                    part.depth,
                    part.mean_accuracy,
                    color=CORE_COLOR[core],
                    ls=linestyle,
                    marker="o" if routing_label == "correct" else "s",
                    ms=3.8,
                    lw=LW_DATA,
                    label=f"{CORE_LABEL[core]}, {routing_label}",
                )
        ax.set_xticks([2, 4])
        ax.set_xlabel("dendritic depth")
        if col == 0:
            ax.set_ylabel("test accuracy")
        panel_title(ax, next(letters), f"{TASK_LABEL[task]} ancestry")
        style_axis(ax)

    ax = axes[2, 0]
    route_effect = contrasts[
        (contrasts.study == "routing")
        & (contrasts.contrast == "correct - shuffled ancestry")
    ]
    for task, ls in (("mnist", "-"), ("noise_resilience", "--")):
        for core in CORE_ORDER:
            part = route_effect[
                (route_effect.task == task) & (route_effect.core == core)
            ].sort_values("depth")
            ax.plot(
                part.depth,
                100 * part.mean_difference,
                color=CORE_COLOR[core],
                ls=ls,
                marker="o",
                ms=4,
                lw=LW_DATA,
                label=f"{TASK_LABEL[task]}, {CORE_LABEL[core].lower()}",
            )
    ax.axhline(0, color=COLORS["mute"], lw=LW_REF, ls=":")
    ax.set_xticks([2, 4])
    ax.set_xlabel("dendritic depth")
    ax.set_ylabel("correct - shuffled (pp)")
    panel_title(ax, next(letters), "Matched-bandwidth routing")
    style_axis(ax)

    ax = axes[2, 1]
    interaction = contrasts[
        (contrasts.study == "routing")
        & (contrasts.contrast == "routing by depth interaction")
    ]
    x = np.arange(len(interaction))
    vals = 100 * interaction.mean_difference.to_numpy()
    lo = 100 * interaction.ci95_low.to_numpy()
    hi = 100 * interaction.ci95_high.to_numpy()
    colors = [CORE_COLOR[core] for core in interaction.core]
    ax.errorbar(
        x,
        vals,
        yerr=np.vstack([vals - lo, hi - vals]),
        fmt="none",
        ecolor=COLORS["edge"],
        lw=LW_ERR,
        capsize=ERR_CAPSIZE,
    )
    ax.scatter(x, vals, c=colors, s=20, zorder=3)
    ax.axhline(0, color=COLORS["mute"], lw=LW_REF, ls="--")
    labels = [
        f"{TASK_LABEL[t]}\n{CORE_LABEL[c]}"
        for t, c in zip(interaction.task, interaction.core)
    ]
    ax.set_xticks(x, labels)
    ax.set_ylabel("depth interaction (pp)")
    panel_title(ax, next(letters), "Routing by depth")
    style_axis(ax)

    ax = axes[2, 2]
    task_interaction = contrasts[
        (contrasts.study == "spatial")
        & (contrasts.contrast == "topology by task interaction")
    ]
    order = ["backprop", *FEEDBACK_ORDER]
    for core, offset in zip(CORE_ORDER, (-0.08, 0.08)):
        part = (
            task_interaction[task_interaction.core == core]
            .set_index("feedback")
            .loc[order]
        )
        values = 100 * part.mean_difference.to_numpy()
        low = 100 * part.ci95_low.to_numpy()
        high = 100 * part.ci95_high.to_numpy()
        ax.errorbar(
            np.arange(4) + offset,
            values,
            yerr=np.vstack([values - low, high - values]),
            fmt="o",
            color=CORE_COLOR[core],
            ms=4.0,
            lw=LW_ERR,
            capsize=ERR_CAPSIZE,
            label=CORE_LABEL[core],
        )
    ax.axhline(0, color=COLORS["mute"], lw=LW_REF, ls="--")
    ax.set_xticks(np.arange(4), ["BP", "Scalar", "Ancestry", "Exact"])
    ax.set_ylabel("MNIST - noise interaction (pp)")
    panel_title(ax, next(letters), "Topology by task")
    style_axis(ax)

    topology_handles, topology_labels = axes[0, 0].get_legend_handles_labels()
    routing_handles, routing_labels = axes[1, 1].get_legend_handles_labels()
    fig.legend(
        topology_handles,
        topology_labels,
        ncol=4,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        frameon=False,
        fontsize=PT_LEGEND,
    )
    fig.legend(
        routing_handles,
        routing_labels,
        ncol=4,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.935),
        frameon=False,
        fontsize=PT_LEGEND,
    )
    superseded = ANALYSIS / "superseded"
    superseded.mkdir(parents=True, exist_ok=True)
    fig.canvas.draw()
    audit_layout(fig, "fig_prospective_topology_routing_historical_excluded")
    audit_text_over_data(fig, "fig_prospective_topology_routing_historical_excluded")
    fig.savefig(
        superseded / "fig_prospective_topology_routing_historical_excluded.pdf",
        metadata={"CreationDate": None, "ModDate": None},
    )
    fig.savefig(
        superseded / "fig_prospective_topology_routing_historical_excluded.png",
        dpi=350,
    )
    plt.close(fig)


def plot_fixed_budget(summary: pd.DataFrame, contrasts: pd.DataFrame) -> None:
    data = summary[summary.family == "fixed_budget"]
    valid_cores = [core for core in CORE_ORDER if core in set(data.core)]
    if not valid_cores:
        raise ValueError("No input-valid fixed-budget conditions")
    fig, axes = plt.subplots(
        3,
        3,
        figsize=(FIG_W, 7.25),
        gridspec_kw={
            "left": 0.10,
            "right": 0.94,
            "bottom": 0.08,
            "top": 0.90,
            "wspace": 0.78,
            "hspace": 0.68,
        },
    )
    letters = iter("ABCDEFGHI")
    for ax, feedback in zip(axes[0, :3], FEEDBACK_ORDER):
        for core in valid_cores:
            _errorbar_line(
                ax,
                data[
                    (data.strategy == "local_ca")
                    & (data.feedback == feedback)
                    & (data.core == core)
                ],
                "depth",
                color=CORE_COLOR[core],
                label=CORE_LABEL[core],
            )
        panel_title(ax, next(letters), FEEDBACK_LABEL[feedback])
        ax.set_xticks([1, 2, 3, 4], ["D1", "D2", "D3", "D4"])
        ax.set_xlabel("physical depth")
        if ax is axes[0, 0]:
            ax.set_ylabel("test accuracy")
        style_axis(ax)
    ax = axes[1, 0]
    for core in valid_cores:
        _errorbar_line(
            ax,
            data[(data.strategy == "standard") & (data.core == core)],
            "depth",
            color=CORE_COLOR[core],
            label=CORE_LABEL[core],
        )
    panel_title(ax, next(letters), "Backpropagation")
    ax.set_xticks([1, 2, 3, 4], ["D1", "D2", "D3", "D4"])
    ax.set_xlabel("physical depth")
    style_axis(ax)

    ax = axes[1, 1]
    depth_effect = contrasts[
        (contrasts.study == "fixed_budget")
        & (contrasts.contrast == "depth 4 - depth 1")
    ]
    order = ["backprop", *FEEDBACK_ORDER]
    offsets = (0.0,) if len(valid_cores) == 1 else (-0.08, 0.08)
    for core, offset in zip(valid_cores, offsets):
        part = depth_effect[depth_effect.core == core].set_index("feedback").loc[order]
        ax.errorbar(
            np.arange(4) + offset,
            100 * part.mean_difference,
            yerr=np.vstack(
                [
                    100 * (part.mean_difference - part.ci95_low),
                    100 * (part.ci95_high - part.mean_difference),
                ]
            ),
            fmt="o",
            color=CORE_COLOR[core],
            lw=LW_ERR,
            capsize=ERR_CAPSIZE,
            label=CORE_LABEL[core],
        )
    ax.axhline(0, color=COLORS["mute"], lw=LW_REF, ls="--")
    ax.set_xticks(np.arange(4), ["BP", "fallback", "neuron", "exact"])
    ax.tick_params(axis="x", labelrotation=24)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right")
    ax.set_ylabel("D4 - D1 (pp)")
    panel_title(ax, next(letters), "Fixed-contact depth")
    style_axis(ax)

    ax = axes[1, 2]
    gap = contrasts[
        (contrasts.study == "fixed_budget") & (contrasts.contrast == "local - backprop")
    ]
    for core in valid_cores:
        for feedback, ls in zip(FEEDBACK_ORDER, (":", "--", "-")):
            part = gap[(gap.core == core) & (gap.feedback == feedback)].sort_values(
                "depth"
            )
            ax.plot(
                part.depth,
                100 * part.mean_difference,
                color=CORE_COLOR[core],
                ls=ls,
                marker="o",
                ms=3.6,
                lw=LW_DATA,
                label=f"{CORE_LABEL[core]}, {FEEDBACK_LABEL[feedback].lower()}",
            )
    ax.axhline(0, color=COLORS["mute"], lw=LW_REF, ls="-.")
    ax.set_xticks([1, 2, 3, 4], ["D1", "D2", "D3", "D4"])
    ax.set_xlabel("physical depth")
    ax.set_ylabel("local - BP (pp)")
    panel_title(ax, next(letters), "Local-learning gap")
    style_axis(ax)

    ax = axes[2, 0]
    resource = data[
        (data.strategy == "standard") & (data.core == valid_cores[0])
    ].sort_values("depth")
    ax.plot(
        resource.depth,
        resource.active_synapses / 1e6,
        color=COLORS["dend"],
        marker="o",
        lw=LW_DATA,
        label="active synapses",
    )
    ax.plot(
        resource.depth,
        resource.total_parameters / 1e6,
        color=COLORS["oracle"],
        marker="s",
        lw=LW_DATA,
        label="trainable parameters",
    )
    ax.set_xticks([1, 2, 3, 4], ["D1", "D2", "D3", "D4"])
    ax.set_xlabel("physical depth")
    ax.set_ylabel("count (millions)")
    panel_title(ax, next(letters), "Matched input budget")
    style_axis(ax)

    ax = axes[2, 1]
    for core in valid_cores:
        part = data[
            (data.strategy == "local_ca")
            & (data.feedback == "path_transport")
            & (data.core == core)
        ].sort_values("depth")
        ax.plot(
            part.depth,
            part.mean_duration_seconds / 60,
            color=CORE_COLOR[core],
            marker="o",
            lw=LW_DATA,
            label=CORE_LABEL[core],
        )
    ax.set_xticks([1, 2, 3, 4], ["D1", "D2", "D3", "D4"])
    ax.set_xlabel("physical depth")
    ax.set_ylabel("mean runtime (min)")
    panel_title(ax, next(letters), "Runtime")
    style_axis(ax)

    ax = axes[2, 2]
    for core in valid_cores:
        part = data[
            (data.strategy == "local_ca")
            & (data.feedback == "path_transport")
            & (data.core == core)
        ].sort_values("depth")
        ax.plot(
            part.depth,
            part.mean_peak_memory_bytes / 2**20,
            color=CORE_COLOR[core],
            marker="o",
            lw=LW_DATA,
            label=CORE_LABEL[core],
        )
    ax.set_xticks([1, 2, 3, 4], ["D1", "D2", "D3", "D4"])
    ax.set_xlabel("physical depth")
    ax.set_ylabel("peak allocated memory (MiB)")
    panel_title(ax, next(letters), "Memory")
    style_axis(ax)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        ncol=len(valid_cores),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        frameon=False,
        fontsize=PT_LEGEND,
    )

    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.canvas.draw()
    audit_layout(fig, "fig_prospective_fixed_budget_depth")
    audit_text_over_data(fig, "fig_prospective_fixed_budget_depth")
    fig.savefig(
        FIGURES / "fig_prospective_fixed_budget_depth.pdf",
        metadata={"CreationDate": None, "ModDate": None},
    )
    fig.savefig(FIGURES / "fig_prospective_fixed_budget_depth.png", dpi=350)
    canonical = ROOT / "figures" / "supplementary" / "figure_S08_panels_A-I.pdf"
    canonical.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(canonical, metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)


def report(contrasts: pd.DataFrame, *, includes_fixed_budget: bool = True) -> str:
    total = "1,200" if includes_fixed_budget else "880"
    lines = [
        "# Prospective shunting, topology, and fixed-budget results",
        "",
        f"This report is generated only after all {total} included frozen confirmatory runs pass the artifact audit. Seeds are paired and are the inferential unit. Canary runs are excluded.",
        "",
    ]
    sections = [
        (
            "Matched-bandwidth ancestry routing",
            "routing",
            "correct - shuffled ancestry",
        ),
        ("Task-aligned fixed topology", "spatial", "spatial - random"),
        ("Inhibitory dose", "inhibition", "core by dose interaction"),
    ]
    if includes_fixed_budget:
        sections.append(
            ("Fixed-budget dendritic depth", "fixed_budget", "depth 4 - depth 1")
        )
    for title, study, name in sections:
        lines.extend([f"## {title}", ""])
        part = contrasts[(contrasts.study == study) & (contrasts.contrast == name)]
        for _, row in part.iterrows():
            labels = []
            for key in ("task", "core", "feedback", "depth"):
                value = row.get(key)
                if pd.notna(value):
                    labels.append(f"{key}={value}")
            lines.append(
                f"- {', '.join(labels)}: {100 * row.mean_difference:.2f} percentage points "
                f"(95% paired bootstrap CI {100 * row.ci95_low:.2f} to "
                f"{100 * row.ci95_high:.2f}; {int(row.wins)}/{int(row.n_pairs)} "
                f"positive pairs; exact Wilcoxon P={row.wilcoxon_p_two_sided:.4g}; "
                f"within-study BH-adjusted P={row.fdr_bh_within_study:.4g})."
            )
        lines.append("")
    lines.extend(
        [
            "## Interpretation guardrails",
            "",
            "A spatial-topology effect shared by backpropagation and local learning is a forward inductive-bias effect. A correct-versus-shuffled ancestry effect isolates feedback routing at matched bandwidth. An inhibition-dose effect changes the full operating point and is not, by itself, a backward-only causal intervention. The fixed-budget comparison holds distal leaves and active input contacts approximately constant, but deeper trees retain additional coupling and reactivation parameters.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--followup-audit",
        type=Path,
        default=ANALYSIS / "prospective_followup_confirmatory_audit.csv",
    )
    parser.add_argument(
        "--fixed-budget-audit",
        type=Path,
        default=ANALYSIS / "prospective_fixed_budget_confirmatory_audit.csv",
    )
    parser.add_argument("--expected-seeds", type=int, default=10)
    parser.add_argument(
        "--followup-only",
        action="store_true",
        help="Analyze the complete 880-run dose/topology/routing cohort without waiting for fixed-budget depth.",
    )
    args = parser.parse_args()

    followup = pd.read_csv(args.followup_audit)
    if args.followup_only:
        frame = validate_followup_only(followup, args.expected_seeds)
    else:
        frame = validate(
            followup,
            pd.read_csv(args.fixed_budget_audit),
            args.expected_seeds,
        )
    summary = summaries(frame)
    if args.followup_only:
        rows: list[dict[str, object]] = []
        inhibition_contrasts(frame, rows)
        spatial_contrasts(frame, rows)
        routing_contrasts(frame, rows)
        contrast = pd.DataFrame(rows)
        contrast["fdr_bh_within_study"] = contrast.groupby(
            "study", group_keys=False
        )["wilcoxon_p_two_sided"].transform(benjamini_hochberg)
    else:
        contrast = all_contrasts(frame)

    SOURCE_DATA.mkdir(parents=True, exist_ok=True)
    frame.to_csv(SOURCE_DATA / "seed_outcomes.csv", index=False)
    summary.to_csv(SOURCE_DATA / "condition_summary.csv", index=False)
    contrast.to_csv(SOURCE_DATA / "paired_contrasts.csv", index=False)
    (ANALYSIS / "prospective_followup_confirmatory_results.md").write_text(
        report(contrast, includes_fixed_budget=not args.followup_only)
    )
    plot_inhibition(summary, contrast)
    plot_topology(summary, contrast)
    if not args.followup_only:
        plot_fixed_budget(summary, contrast)


if __name__ == "__main__":
    main()
