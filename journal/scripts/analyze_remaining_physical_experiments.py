#!/usr/bin/env python3
"""Analyze the literal grouped-point and second-hierarchy experiments."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path
import sys
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from journal_style import (  # noqa: E402
    COLORS,
    ERR_CAPSIZE,
    FIG_W,
    LW_DATA,
    LW_ERR,
    LW_HAIR,
    MARKER_MS,
    PT_LEGEND,
    PT_SMALL,
    apply_neurips_style,
    audit_layout,
    audit_text_over_data,
    clean_legend,
    panel_title,
    style_axis,
)


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "remaining_physical_experiment_runs"
REFERENCE = (
    ROOT
    / "source_data"
    / "nonlinear_physical_depth_confirmatory"
    / "seed_outcomes.csv"
)
STAR_REFERENCE = (
    ROOT
    / "source_data"
    / "point_dendrite_credit_controls"
    / "combined_seed_outcomes.csv"
)
OUTPUT = ROOT / "source_data" / "remaining_physical_experiments"
FIGURES = ROOT / "figures" / "generated"

RUN_SPECS = {
    "journal_remaining_h3_aligned_grouped_point_bp": (
        3, "aligned", "grouped_point", "full_bp"
    ),
    "journal_remaining_h3_rewired_tree_grouped_point_bp": (
        3, "rewired_tree", "grouped_point", "full_bp"
    ),
    "journal_remaining_h2_aligned_serial_bp": (
        2, "aligned", "serial_tree", "full_bp"
    ),
    "journal_remaining_h2_rewired_tree_serial_bp": (
        2, "rewired_tree", "serial_tree", "full_bp"
    ),
    "journal_remaining_h2_aligned_grouped_point_bp": (
        2, "aligned", "grouped_point", "full_bp"
    ),
    "journal_remaining_h2_rewired_tree_grouped_point_bp": (
        2, "rewired_tree", "grouped_point", "full_bp"
    ),
    "journal_remaining_h2_aligned_serial_local3f": (
        2, "aligned", "serial_tree", "local_auto"
    ),
    "journal_remaining_h2_rewired_tree_serial_local3f": (
        2, "rewired_tree", "serial_tree", "local_auto"
    ),
}

RESOURCE_COLUMNS = [
    "total_parameters",
    "trainable_parameters",
    "active_synapses",
    "candidate_synapse_slots",
    "persistent_state_scalars",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def latest_run(stem: str) -> Path:
    matches = sorted(RUNS.glob(f"{stem}_*"))
    if not matches:
        raise FileNotFoundError(f"No run directory for {stem}")
    return matches[-1]


def metric(final: dict[str, Any], name: str, split: str) -> float:
    return float(final[name][split])


def collect(*, allow_incomplete: bool) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    run_records: list[dict[str, Any]] = []
    for stem, (hierarchy, regime, architecture, credit_default) in RUN_SPECS.items():
        run = latest_run(stem)
        original = yaml.safe_load((run / "original_config.yaml").read_text())
        expected = int(original["sweep_contract"]["expected_config_count"])
        configs = sorted(
            (run / "configs").glob("unified_config_*.yaml"),
            key=lambda p: int(p.stem.rsplit("_", 1)[1]),
        )
        run_records.append(
            {
                "stem": stem,
                "run_dir": str(run.relative_to(ROOT)),
                "expected": expected,
                "generated": len(configs),
                "manifest_sha256": sha256(run / "frozen_sweep_manifest.json"),
            }
        )
        if len(configs) != expected:
            missing.append(f"{stem}: generated {len(configs)}/{expected}")
        for config_path in configs:
            index = int(config_path.stem.rsplit("_", 1)[1])
            result = run / "results" / f"config_{index}"
            final_path = result / "performance" / "final.json"
            resources_path = result / "model_resources.json"
            if not final_path.is_file() or not resources_path.is_file():
                missing.append(f"{stem}/config_{index}")
                continue
            config = yaml.safe_load(config_path.read_text())
            final = json.loads(final_path.read_text())
            resources = json.loads(resources_path.read_text())
            population = config["model"]["core"]["population_network"]["layers"][0][
                "populations"
            ][0]
            factors = list(population["branch_factors"])
            learning = config["training"]["main"]
            if credit_default == "local_auto":
                local = learning["learning_strategy_config"]
                credit = (
                    "local_path"
                    if str(local["error_broadcast_mode"]).lower()
                    == "path_transport"
                    else "local_shared"
                )
            else:
                credit = credit_default
            log_text = "\n".join(
                path.read_text(errors="replace")
                for path in (result / "train.log", result / "dendritic_modeling.log")
                if path.is_file()
            ).lower()
            row: dict[str, Any] = {
                "hierarchy": hierarchy,
                "cohort": stem,
                "run_dir": str(run.relative_to(ROOT)),
                "config_index": index,
                "seed": int(config["experiment"]["seed"]),
                "regime": regime,
                "architecture": architecture,
                "credit": credit,
                "depth": len(factors),
                "branch_factors": "x".join(map(str, factors)),
                "total_parameters": int(resources["total_parameters"]),
                "trainable_parameters": int(resources["trainable_parameters"]),
                "active_synapses": int(resources.get("active_synapses", 0)),
                "candidate_synapse_slots": int(
                    resources.get("candidate_synapse_slots", 0)
                ),
                "persistent_state_scalars": int(
                    resources.get("persistent_state_scalars_per_sample", 0)
                ),
                "config_sha256": sha256(config_path),
                "final_sha256": sha256(final_path),
                "fallback_mentions": int(log_text.count("fallback")),
                "nonfinite_alert": bool(
                    "nan detected" in log_text
                    or "non-finite" in log_text
                    or "nonfinite" in log_text
                ),
            }
            for name in ("accuracy", "auc", "categorical_loglikelihood"):
                for split in ("train", "valid", "test"):
                    row[f"{split}_{name}"] = metric(final, name, split)
            rows.append(row)
    if missing and not allow_incomplete:
        raise RuntimeError(
            f"Experiment matrix incomplete ({len(missing)} missing): "
            + ", ".join(missing[:20])
        )
    frame = pd.DataFrame(rows)
    return frame, {
        "status": "incomplete" if missing else "complete",
        "expected_rows": int(sum(record["expected"] for record in run_records)),
        "observed_rows": len(frame),
        "missing_count": len(missing),
        "missing_examples": missing[:40],
        "runs": run_records,
    }


def h3_reference() -> pd.DataFrame:
    frame = pd.read_csv(REFERENCE)
    frame = frame[
        frame.regime.isin(["aligned", "rewired_tree"])
        & frame.mechanism.eq("shunting")
        & frame.transport.eq("backpropagation")
    ].copy()
    frame["hierarchy"] = 3
    frame["architecture"] = "serial_tree"
    frame["credit"] = "full_bp"
    return frame


def h3_star_reference() -> pd.DataFrame:
    frame = pd.read_csv(STAR_REFERENCE)
    frame = frame[
        frame.regime.isin(["aligned", "rewired_tree"])
        & frame.architecture.eq("all_active_star")
        & frame.credit.eq("full_bp")
    ].copy()
    frame["hierarchy"] = 3
    frame["architecture"] = "grouped_star"
    return frame


def bootstrap_mean(values: np.ndarray, seed: int, draws: int = 50_000) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), size=(draws, len(values)))].mean(1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(values.mean()), float(low), float(high)


def exact_sign_flip_p(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    observed = abs(float(values.mean()))
    null = np.asarray(
        [
            abs(float(np.mean(values * np.asarray(signs))))
            for signs in itertools.product((-1.0, 1.0), repeat=len(values))
        ]
    )
    return float(np.mean(null >= observed - 1e-15))


def select(frame: pd.DataFrame, **filters: Any) -> pd.Series:
    part = frame.copy()
    for column, value in filters.items():
        part = part[part[column].eq(value)]
    if part.seed.duplicated().any():
        raise ValueError(f"Duplicate seed rows for {filters}")
    return part.set_index("seed").test_accuracy.sort_index()


def difference(
    frame: pd.DataFrame,
    left: dict[str, Any],
    right: dict[str, Any],
) -> pd.Series:
    a, b = select(frame, **left).align(select(frame, **right), join="inner")
    if len(a) != 10 or not a.index.equals(b.index):
        raise ValueError(f"Expected ten paired seeds for {left} minus {right}")
    return a - b


def contrast_row(
    *,
    name: str,
    family: str,
    values: pd.Series,
    seed: int,
    detail: str,
) -> dict[str, Any]:
    mean, low, high = bootstrap_mean(values.to_numpy(float), seed)
    return {
        "contrast": name,
        "family": family,
        "detail": detail,
        "n_seeds": len(values),
        "mean_accuracy": mean,
        "mean_pp": 100.0 * mean,
        "ci_low_accuracy": low,
        "ci_high_accuracy": high,
        "ci_low_pp": 100.0 * low,
        "ci_high_pp": 100.0 * high,
        "positive_pairs": int((values > 0).sum()),
        "negative_pairs": int((values < 0).sum()),
        "zero_pairs": int((values == 0).sum()),
        "exact_two_sided_sign_flip_p": exact_sign_flip_p(values.to_numpy(float)),
        "positive_claim_gate": bool(low > 0 and int((values > 0).sum()) >= 8),
        "seed_values_pp": ";".join(f"{100*x:.8f}" for x in values.to_numpy(float)),
    }


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    grouping = ["hierarchy", "regime", "architecture", "credit", "depth"]
    for index, (key, part) in enumerate(frame.groupby(grouping, sort=True)):
        mean, low, high = bootstrap_mean(part.test_accuracy.to_numpy(float), 7100 + index)
        rows.append(
            dict(
                zip(grouping, key),
                n_seeds=part.seed.nunique(),
                mean_test_accuracy=mean,
                ci_low=low,
                ci_high=high,
            )
        )
    return pd.DataFrame(rows)


def build_contrasts(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    seed_values: dict[str, pd.Series] = {}

    def add(name: str, family: str, values: pd.Series, detail: str) -> None:
        seed_values[name] = values
        rows.append(
            contrast_row(
                name=name,
                family=family,
                values=values,
                seed=8200 + len(rows),
                detail=detail,
            )
        )

    # H=3: literal parallel-readout grouped point versus the frozen serial tree.
    for regime in ("aligned", "rewired_tree"):
        for depth in (1, 2, 3):
            add(
                f"h3_serial_minus_grouped__{regime}__d{depth}",
                "h3_architecture",
                difference(
                    frame,
                    dict(hierarchy=3, regime=regime, architecture="serial_tree", credit="full_bp", depth=depth),
                    dict(hierarchy=3, regime=regime, architecture="grouped_point", credit="full_bp", depth=depth),
                ),
                f"H=3 {regime}: serial tree minus literal grouped point at D{depth}",
            )
        add(
            f"h3_serial_depth__{regime}",
            "h3_depth",
            difference(
                frame,
                dict(hierarchy=3, regime=regime, architecture="serial_tree", credit="full_bp", depth=3),
                dict(hierarchy=3, regime=regime, architecture="serial_tree", credit="full_bp", depth=1),
            ),
            f"H=3 {regime}: serial D3 minus D1",
        )
        add(
            f"h3_grouped_depth__{regime}",
            "h3_depth",
            difference(
                frame,
                dict(hierarchy=3, regime=regime, architecture="grouped_point", credit="full_bp", depth=3),
                dict(hierarchy=3, regime=regime, architecture="grouped_point", credit="full_bp", depth=1),
            ),
            f"H=3 {regime}: grouped-point D3 minus D1",
        )
    add(
        "h3_serial_grouped_alignment_interaction__d3",
        "h3_primary",
        seed_values["h3_serial_minus_grouped__aligned__d3"]
        - seed_values["h3_serial_minus_grouped__rewired_tree__d3"],
        "H=3 aligned-minus-reversed interaction for serial minus grouped point at D3",
    )
    add(
        "h3_serial_grouped_depth_interaction__aligned",
        "h3_primary",
        seed_values["h3_serial_depth__aligned"]
        - seed_values["h3_grouped_depth__aligned"],
        "H=3 aligned difference in D3-minus-D1 between serial and grouped point",
    )
    for regime in ("aligned", "rewired_tree"):
        add(
            f"h3_star_minus_grouped__{regime}__d3",
            "h3_architecture",
            difference(
                frame,
                dict(hierarchy=3, regime=regime, architecture="grouped_star", credit="full_bp", depth=3),
                dict(hierarchy=3, regime=regime, architecture="grouped_point", credit="full_bp", depth=3),
            ),
            f"H=3 {regime}: earlier grouped star minus literal grouped point at D3",
        )

    # H=2: depth and placement crossover for BP and both local transports.
    methods = [
        ("serial_bp", "serial_tree", "full_bp"),
        ("grouped_bp", "grouped_point", "full_bp"),
        ("shared_local", "serial_tree", "local_shared"),
        ("path_local", "serial_tree", "local_path"),
    ]
    for method, architecture, credit in methods:
        for regime in ("aligned", "rewired_tree"):
            add(
                f"h2_depth__{method}__{regime}",
                "h2_depth",
                difference(
                    frame,
                    dict(hierarchy=2, regime=regime, architecture=architecture, credit=credit, depth=2),
                    dict(hierarchy=2, regime=regime, architecture=architecture, credit=credit, depth=1),
                ),
                f"H=2 {regime}: D2 minus D1 for {method}",
            )
        add(
            f"h2_alignment_interaction__{method}",
            "h2_primary",
            seed_values[f"h2_depth__{method}__aligned"]
            - seed_values[f"h2_depth__{method}__rewired_tree"],
            f"H=2 aligned-minus-reversed D2-minus-D1 interaction for {method}",
        )
    for regime in ("aligned", "rewired_tree"):
        for depth in (1, 2):
            add(
                f"h2_serial_minus_grouped__{regime}__d{depth}",
                "h2_architecture",
                difference(
                    frame,
                    dict(hierarchy=2, regime=regime, architecture="serial_tree", credit="full_bp", depth=depth),
                    dict(hierarchy=2, regime=regime, architecture="grouped_point", credit="full_bp", depth=depth),
                ),
                f"H=2 {regime}: serial BP minus grouped-point BP at D{depth}",
            )
    add(
        "h2_serial_grouped_alignment_interaction__d2",
        "h2_primary",
        seed_values["h2_serial_minus_grouped__aligned__d2"]
        - seed_values["h2_serial_minus_grouped__rewired_tree__d2"],
        "H=2 aligned-minus-reversed interaction for serial minus grouped point at D2",
    )

    contrast = pd.DataFrame(rows)
    seed_frame = pd.DataFrame(seed_values).rename_axis("seed").reset_index()
    return contrast, seed_frame


def audit(frame: pd.DataFrame, collection: dict[str, Any]) -> dict[str, Any]:
    finite_columns = [
        column
        for column in frame.columns
        if column.endswith("_accuracy")
        or column.endswith("_auc")
        or column.endswith("_categorical_loglikelihood")
    ]
    checks: dict[str, Any] = {
        **collection,
        "unique_rows": int(
            frame[["cohort", "config_index"]].drop_duplicates().shape[0]
        ),
        "duplicate_seed_conditions": int(
            frame.duplicated(
                ["hierarchy", "regime", "architecture", "credit", "depth", "seed"]
            ).sum()
        ),
        "all_metrics_finite": bool(
            len(frame) > 0 and np.isfinite(frame[finite_columns].to_numpy(float)).all()
        ),
        "fallback_mentions": int(frame.fallback_mentions.sum()) if len(frame) else 0,
        "nonfinite_alerts": int(frame.nonfinite_alert.sum()) if len(frame) else 0,
        "seeds_by_hierarchy": {
            str(int(h)): sorted(map(int, part.seed.unique()))
            for h, part in frame.groupby("hierarchy")
        },
    }
    if collection["status"] != "complete":
        checks["status"] = "incomplete"
        return checks

    resource_failures: list[str] = []
    for hierarchy in (2, 3):
        part = frame[frame.hierarchy.eq(hierarchy)]
        for (regime, depth), group in part[part.credit.eq("full_bp")].groupby(
            ["regime", "depth"]
        ):
            for column in RESOURCE_COLUMNS:
                if group[column].nunique() != 1:
                    resource_failures.append(
                        f"H{hierarchy}/{regime}/D{depth}/{column}"
                    )
    h2_serial = frame[
        frame.hierarchy.eq(2)
        & frame.architecture.eq("serial_tree")
        & frame.credit.eq("full_bp")
    ]
    for regime, part in h2_serial.groupby("regime"):
        for column in RESOURCE_COLUMNS:
            if part[column].nunique() != 1:
                resource_failures.append(f"H2/{regime}/fixed-depth-budget/{column}")

    checks["resource_failures"] = resource_failures
    checks["resource_gate"] = not resource_failures
    checks["h2_expected_seeds"] = checks["seeds_by_hierarchy"].get("2") == list(
        range(10300, 10310)
    )
    checks["h3_expected_seeds"] = checks["seeds_by_hierarchy"].get("3") == list(
        range(10200, 10210)
    )
    checks["status"] = (
        "complete_pass"
        if checks["all_metrics_finite"]
        and checks["fallback_mentions"] == 0
        and checks["nonfinite_alerts"] == 0
        and checks["duplicate_seed_conditions"] == 0
        and checks["resource_gate"]
        and checks["h2_expected_seeds"]
        and checks["h3_expected_seeds"]
        else "complete_fail"
    )
    return checks


def line_panel(
    ax: plt.Axes,
    summary: pd.DataFrame,
    *,
    hierarchy: int,
    regime: str,
    title: str,
    letter: str,
    legend: bool = True,
    legend_loc: str = "upper left",
    key_note: str | None = None,
    ylabel: bool = True,
) -> None:
    styles = [
        ("serial_tree", "full_bp", "serial BP", COLORS["shunting"], "o", "-"),
        ("grouped_point", "full_bp", "grouped-point BP", COLORS["point_mlp"], "s", "--"),
    ]
    if hierarchy == 3:
        styles.append(
            ("grouped_star", "full_bp", "grouped star", COLORS["oracle"], "^", ":")
        )
    else:
        styles.extend(
            [
                ("serial_tree", "local_shared", "shared LocalCA", COLORS["local"], "D", "-."),
                ("serial_tree", "local_path", "path LocalCA", COLORS["pathway"], "^", ":"),
            ]
        )
    for architecture, credit, label, color, marker, linestyle in styles:
        part = summary[
            summary.hierarchy.eq(hierarchy)
            & summary.regime.eq(regime)
            & summary.architecture.eq(architecture)
            & summary.credit.eq(credit)
        ].sort_values("depth")
        if part.empty:
            continue
        y = part.mean_test_accuracy.to_numpy(float)
        lo = part.ci_low.to_numpy(float)
        hi = part.ci_high.to_numpy(float)
        ax.errorbar(
            part.depth,
            y,
            yerr=np.vstack([y - lo, hi - y]),
            color=color,
            marker=marker,
            linestyle=linestyle,
            linewidth=LW_DATA,
            elinewidth=LW_ERR,
            capsize=ERR_CAPSIZE,
            markersize=MARKER_MS,
            markeredgecolor="white",
            markeredgewidth=0.5,
            label=label,
        )
    panel_title(ax, letter, title)
    ax.set_xlabel(r"physical stage count $D_{\mathrm{p}}$")
    if ylabel:
        ax.set_ylabel("test accuracy")
    depths = sorted(summary[summary.hierarchy.eq(hierarchy)].depth.unique())
    ax.set_xticks(depths)
    ax.set_xlim(depths[0] - 0.3, depths[-1] + 0.3)
    ax.set_ylim(0.44, 1.06)
    ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    style_axis(ax, grid="y")
    if legend:
        if legend_loc == "below":
            # Compact two-column key that stays inside this panel's grid
            # cell: a wider key crossed the cell boundary and was sliced
            # when the block is recomposed into S18.
            clean_legend(
                ax, loc="upper center", bbox_to_anchor=(0.50, -0.26),
                fontsize=PT_SMALL - 0.4, ncol=2, handlelength=1.1,
                columnspacing=0.7, handletextpad=0.4,
            )
        else:
            clean_legend(ax, loc=legend_loc, fontsize=PT_LEGEND,
                         handlelength=2.0)
    elif key_note:
        ax.text(
            0.03, 0.965, key_note, transform=ax.transAxes, ha="left", va="top",
            fontsize=PT_SMALL, color=COLORS["mute"],
        )


def forest(
    ax: plt.Axes,
    contrast: pd.DataFrame,
    names: list[str],
    labels: list[str],
    *,
    letter: str,
    title: str,
    colors: list[str] | None = None,
) -> None:
    indexed = contrast.set_index("contrast")
    y = np.arange(len(names))[::-1]
    colors = colors or [COLORS["shunting"]] * len(names)
    ax.axvline(0, color=COLORS["mute"], linewidth=LW_HAIR, zorder=0)
    for yi, name, color in zip(y, names, colors):
        row = indexed.loc[name]
        mean = float(row.mean_pp)
        low = float(row.ci_low_pp)
        high = float(row.ci_high_pp)
        ax.errorbar(
            mean,
            yi,
            xerr=np.asarray([[mean - low], [high - mean]]),
            fmt="o",
            color=color,
            markerfacecolor=color,
            markeredgecolor="white",
            markeredgewidth=0.5,
            markersize=MARKER_MS,
            elinewidth=LW_ERR,
            capsize=ERR_CAPSIZE,
        )
    panel_title(ax, letter, title)
    # Line-break long condition names so the third-column forest plots retain
    # a canonical full-width canvas without colliding with neighboring panels.
    ax.set_xlim(-2.5, 35.5)
    ax.set_ylim(-0.6, len(names) - 0.4)
    ax.set_yticks(y, labels)
    ax.set_xlabel("paired difference (pp)")
    style_axis(ax, grid="x")


def make_figure(summary: pd.DataFrame, contrast: pd.DataFrame) -> None:
    apply_neurips_style()
    fig = plt.figure(figsize=(FIG_W, 5.15))
    grid = fig.add_gridspec(
        2, 3, left=0.115, right=0.985, top=0.9125, bottom=0.155,
        wspace=0.62, hspace=0.74,
    )
    axes = [fig.add_subplot(grid[r, c]) for r in range(2) for c in range(3)]

    line_panel(
        axes[0], summary, hierarchy=3, regime="aligned",
        letter="P", title="H=3 aligned hierarchy",
    )
    line_panel(
        axes[1], summary, hierarchy=3, regime="rewired_tree",
        letter="Q", title="H=3 reversed placement",
        # S18 lettering: source panel P is published as S18 F.
        legend=False, key_note="key as in F", ylabel=False,
    )
    forest(
        axes[2], contrast,
        [
            "h3_serial_minus_grouped__aligned__d3",
            "h3_serial_minus_grouped__rewired_tree__d3",
            "h3_serial_grouped_alignment_interaction__d3",
            "h3_star_minus_grouped__aligned__d3",
        ],
        ["serial$-$point\naligned", "serial$-$point\nreversed", "alignment\ninteraction", "star$-$point\naligned"],
        letter="R", title="Composition at H=3",
        colors=[COLORS["shunting"], COLORS["point_mlp"], COLORS["bp"], COLORS["oracle"]],
    )
    line_panel(
        axes[3], summary, hierarchy=2, regime="aligned",
        letter="S", title="Independent H=2 hierarchy", legend_loc="below",
    )
    line_panel(
        axes[4], summary, hierarchy=2, regime="rewired_tree",
        letter="T", title="H=2 reversed placement",
        # S18 lettering: source panel S is published as S18 I.
        legend=False, key_note="key as in I", ylabel=False,
    )
    forest(
        axes[5], contrast,
        [
            "h2_alignment_interaction__serial_bp",
            "h2_alignment_interaction__grouped_bp",
            "h2_alignment_interaction__shared_local",
            "h2_alignment_interaction__path_local",
        ],
        ["serial BP", "grouped BP", "shared\nLocalCA", "path\nLocalCA"],
        letter="U", title="Interactions at H=2",
        colors=[COLORS["shunting"], COLORS["point_mlp"], COLORS["local"], COLORS["pathway"]],
    )

    fig.canvas.draw()
    audit_layout(fig, "fig_remaining_physical_crossovers")
    audit_text_over_data(fig, "fig_remaining_physical_crossovers")
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        FIGURES / "fig_remaining_physical_crossovers.pdf",
        metadata={"CreationDate": None, "ModDate": None},
    )
    fig.savefig(FIGURES / "fig_remaining_physical_crossovers.png", dpi=600)
    plt.close(fig)


def write_report(contrast: pd.DataFrame, audit_record: dict[str, Any]) -> None:
    indexed = contrast.set_index("contrast")

    def sentence(name: str) -> str:
        row = indexed.loc[name]
        return (
            f"{row.mean_pp:.2f} pp "
            f"(95% paired-seed bootstrap interval {row.ci_low_pp:.2f} to "
            f"{row.ci_high_pp:.2f}; {int(row.positive_pairs)}/10 positive; "
            f"exact two-sided sign-flip P={row.exact_two_sided_sign_flip_p:.4f}; "
            f"positive gate={'pass' if row.positive_claim_gate else 'fail'})"
        )

    report = f"""# Remaining physical-depth experiments

Status: **{audit_record['status']}** ({audit_record['observed_rows']}/{audit_record['expected_rows']} new fits).

## H=3 literal grouped-point control

- Serial minus grouped point at aligned D3: {sentence('h3_serial_minus_grouped__aligned__d3')}.
- The same contrast after placement reversal: {sentence('h3_serial_minus_grouped__rewired_tree__d3')}.
- Aligned-minus-reversed architecture interaction: {sentence('h3_serial_grouped_alignment_interaction__d3')}.
- Difference between serial and grouped D3-minus-D1 effects in the aligned task: {sentence('h3_serial_grouped_depth_interaction__aligned')}.
- Earlier grouped star minus the literal grouped point at aligned D3: {sentence('h3_star_minus_grouped__aligned__d3')}.

## Independent H=2 hierarchy

- Serial-BP D2-minus-D1 effect, aligned: {sentence('h2_depth__serial_bp__aligned')}.
- Serial-BP depth-by-placement interaction: {sentence('h2_alignment_interaction__serial_bp')}.
- Serial minus grouped point at aligned D2: {sentence('h2_serial_minus_grouped__aligned__d2')}.
- Architecture-by-placement interaction at D2: {sentence('h2_serial_grouped_alignment_interaction__d2')}.
- Shared-LocalCA D2-minus-D1 effect, aligned: {sentence('h2_depth__shared_local__aligned')}.
- Shared-LocalCA depth-by-placement interaction: {sentence('h2_alignment_interaction__shared_local')}.
- Path-LocalCA D2-minus-D1 effect, aligned: {sentence('h2_depth__path_local__aligned')}.
- Path-LocalCA depth-by-placement interaction: {sentence('h2_alignment_interaction__path_local')}.

The report retains every frozen primary contrast regardless of sign. A positive directional statement is licensed only when the paired interval excludes zero and at least 8/10 seed differences are positive.
"""
    (OUTPUT / "RESULTS.md").write_text(report, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()

    if not RUNS.exists():
        # Frozen-source render: raw run directories are not in this checkout;
        # restyle the figure from the audited source-data summaries.
        summary = pd.read_csv(OUTPUT / "condition_summary.csv")
        contrasts = pd.read_csv(OUTPUT / "paired_contrasts.csv")
        make_figure(summary, contrasts)
        print("Rendered fig_remaining_physical_crossovers from frozen source "
              "data (run directories absent; collection skipped).")
        return

    new, collection = collect(allow_incomplete=args.allow_incomplete)
    if new.empty:
        raise SystemExit("No completed outcomes")
    combined = pd.concat(
        [new, h3_reference(), h3_star_reference()], ignore_index=True, sort=False
    )
    combined = combined[
        combined.hierarchy.eq(2)
        | (
            combined.hierarchy.eq(3)
            & combined.architecture.isin(
                ["serial_tree", "grouped_point", "grouped_star"]
            )
            & combined.credit.eq("full_bp")
        )
    ].copy()
    summary = summarize(combined)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    new.to_csv(OUTPUT / "seed_outcomes_new.csv", index=False)
    combined.to_csv(OUTPUT / "seed_outcomes_with_h3_reference.csv", index=False)
    summary.to_csv(OUTPUT / "condition_summary.csv", index=False)

    if collection["status"] != "complete":
        record = audit(new, collection)
        (OUTPUT / "audit.json").write_text(
            json.dumps(record, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(record, indent=2))
        return

    contrasts, seed_contrasts = build_contrasts(combined)
    contrasts.to_csv(OUTPUT / "paired_contrasts.csv", index=False)
    seed_contrasts.to_csv(OUTPUT / "paired_contrasts_by_seed.csv", index=False)
    record = audit(new, collection)
    (OUTPUT / "audit.json").write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf-8"
    )
    write_report(contrasts, record)
    make_figure(summary, contrasts)
    print(json.dumps(record, indent=2))
    print((OUTPUT / "RESULTS.md").read_text())


if __name__ == "__main__":
    main()
