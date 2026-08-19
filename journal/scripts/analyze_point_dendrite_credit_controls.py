#!/usr/bin/env python3
"""Collect and analyze the point--dendrite and BP--credit control extension."""

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
    PT_ANNOT,
    PT_LEGEND,
    PT_SMALL,
    apply_neurips_style,
    audit_layout,
    audit_text_over_data,
    axis_break_note,
    clean_legend,
    panel_title,
    style_axis,
)
from analyze_nonlinear_physical_depth_confirmatory import draw_tree  # noqa: E402
from credit_tree_schematics import MS_JUNCTION  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "point_dendrite_credit_runs"
REFERENCE = (
    ROOT
    / "source_data"
    / "nonlinear_physical_depth_confirmatory"
    / "seed_outcomes.csv"
)
OUTPUT = ROOT / "source_data" / "point_dendrite_credit_controls"
FIGURES = ROOT / "figures" / "generated"

RUN_SPECS = {
    "aligned_active_param_mlp_bp": ("aligned", "point_mlp_active", "full_bp"),
    "aligned_all_active_star_bp": ("aligned", "all_active_star", "full_bp"),
    "aligned_soma_broadcast_bp": ("aligned", "serial_tree", "soma_broadcast_bp"),
    "aligned_total_param_mlp_bp": ("aligned", "point_mlp_total", "full_bp"),
    "rewired_tree_all_active_star_bp": (
        "rewired_tree",
        "all_active_star",
        "full_bp",
    ),
    "rewired_tree_soma_broadcast_bp": (
        "rewired_tree",
        "serial_tree",
        "soma_broadcast_bp",
    ),
    "aligned_soma_broadcast_matched_optimizer": (
        "aligned",
        "serial_tree",
        "soma_broadcast_matched",
    ),
    "rewired_tree_soma_broadcast_matched_optimizer": (
        "rewired_tree",
        "serial_tree",
        "soma_broadcast_matched",
    ),
}


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def latest_run(stem: str) -> Path:
    matches = sorted(RUNS.glob(f"journal_point_credit_{stem}_*"))
    if not matches:
        raise FileNotFoundError(f"No run directory for {stem}")
    return matches[-1]


def _metric_row(final: dict[str, Any], metric: str, split: str) -> float:
    return float(final[metric][split])


def collect_new(*, allow_incomplete: bool) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    run_records: list[dict[str, Any]] = []
    for stem, (regime, architecture, credit) in RUN_SPECS.items():
        run = latest_run(stem)
        original = yaml.safe_load((run / "original_config.yaml").read_text())
        expected = int(original["sweep_contract"]["expected_config_count"])
        configs = sorted(
            (run / "configs").glob("unified_config_*.yaml"),
            key=lambda path: int(path.stem.rsplit("_", 1)[1]),
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
            population = config["model"]["core"]["population_network"]["layers"][
                0
            ]["populations"][0]
            factors = list(population["branch_factors"])
            is_point = architecture.startswith("point_mlp")
            log_text = "\n".join(
                path.read_text(errors="replace")
                for path in (result / "train.log", result / "dendritic_modeling.log")
                if path.is_file()
            ).lower()
            row: dict[str, Any] = {
                "source": "new_control",
                "cohort": stem,
                "run_dir": str(run.relative_to(ROOT)),
                "config_index": index,
                "seed": int(config["experiment"]["seed"]),
                "regime": regime,
                "architecture": architecture,
                "credit": credit,
                "depth": 0 if is_point else len(factors),
                "reference_depth": 3 if is_point else len(factors),
                "branch_factors": "point" if is_point else "x".join(map(str, factors)),
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
            for metric in ("accuracy", "auc", "categorical_loglikelihood"):
                for split in ("train", "valid", "test"):
                    row[f"{split}_{metric}"] = _metric_row(final, metric, split)
            rows.append(row)
    if missing and not allow_incomplete:
        raise RuntimeError(
            f"Control extension incomplete ({len(missing)} missing): "
            + ", ".join(missing[:12])
        )
    return pd.DataFrame(rows), {
        "status": "incomplete" if missing else "complete",
        "expected_new_rows": int(sum(record["expected"] for record in run_records)),
        "observed_new_rows": len(rows),
        "missing_count": len(missing),
        "missing_examples": missing[:30],
        "runs": run_records,
    }


def reference_rows() -> pd.DataFrame:
    frame = pd.read_csv(REFERENCE)
    frame = frame[
        frame.regime.isin(["aligned", "rewired_tree"])
        & frame.mechanism.eq("shunting")
    ].copy()
    frame["source"] = "frozen_reference"
    frame["architecture"] = "serial_tree"
    frame["credit"] = np.select(
        [
            frame.transport.eq("backpropagation"),
            frame.transport.eq("per_soma_shared"),
            frame.transport.eq("path_transport"),
        ],
        ["full_bp", "local_shared", "local_path"],
        default="exclude",
    )
    frame = frame[frame.credit.ne("exclude")].copy()
    frame["reference_depth"] = frame.depth
    return frame


def bootstrap_mean(
    values: np.ndarray, seed: int, draws: int = 50_000
) -> tuple[float, float, float]:
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


def _select(frame: pd.DataFrame, **filters: Any) -> pd.Series:
    part = frame.copy()
    for column, value in filters.items():
        part = part[part[column].eq(value)]
    if part.seed.duplicated().any():
        raise ValueError(f"Duplicate seed rows for {filters}")
    return part.set_index("seed").test_accuracy.sort_index()


def paired(
    frame: pd.DataFrame,
    *,
    name: str,
    family: str,
    left: dict[str, Any],
    right: dict[str, Any],
    seed: int,
    detail: str,
) -> dict[str, Any]:
    left_values = _select(frame, **left)
    right_values = _select(frame, **right)
    joined = pd.concat([left_values, right_values], axis=1, join="inner")
    joined.columns = ["left", "right"]
    if len(joined) != 10:
        raise ValueError(f"Expected ten paired seeds for {name}, got {len(joined)}")
    values = (joined.left - joined.right).to_numpy(float)
    mean, low, high = bootstrap_mean(values, seed)
    return {
        "contrast": name,
        "family": family,
        "detail": detail,
        "left": json.dumps(left, sort_keys=True),
        "right": json.dumps(right, sort_keys=True),
        "n_pairs": len(values),
        "mean_difference": mean,
        "ci95_low": low,
        "ci95_high": high,
        "positive_pairs": int(np.sum(values > 0)),
        "negative_pairs": int(np.sum(values < 0)),
        "ties": int(np.sum(np.isclose(values, 0))),
        "exact_sign_flip_p_two_sided": exact_sign_flip_p(values),
    }


def summarize(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    group_cols = ["regime", "architecture", "credit", "depth", "reference_depth"]
    summaries: list[dict[str, Any]] = []
    for index, (keys, part) in enumerate(frame.groupby(group_cols, sort=True)):
        row = dict(zip(group_cols, keys))
        row["n_seeds"] = int(part.seed.nunique())
        for metric_index, metric in enumerate(("test_accuracy", "test_auc")):
            mean, low, high = bootstrap_mean(
                part[metric].to_numpy(float), 8_100_000 + 100 * index + metric_index
            )
            row[f"mean_{metric}"] = mean
            row[f"ci95_low_{metric}"] = low
            row[f"ci95_high_{metric}"] = high
        summaries.append(row)

    contrasts: list[dict[str, Any]] = []
    counter = 0
    for regime in ("aligned", "rewired_tree"):
        for depth in (1, 2, 3):
            contrasts.append(
                paired(
                    frame,
                    name=f"serial_minus_star__{regime}__d{depth}",
                    family="architecture",
                    left={
                        "regime": regime,
                        "architecture": "serial_tree",
                        "credit": "full_bp",
                        "depth": depth,
                    },
                    right={
                        "regime": regime,
                        "architecture": "all_active_star",
                        "credit": "full_bp",
                        "depth": depth,
                    },
                    seed=8_200_000 + counter,
                    detail=f"serial tree - resource-identical star, {regime}, D{depth}",
                )
            )
            counter += 1
            contrasts.append(
                paired(
                    frame,
                    name=f"full_bp_minus_soma_broadcast__{regime}__d{depth}",
                    family="credit_coordinate",
                    left={
                        "regime": regime,
                        "architecture": "serial_tree",
                        "credit": "full_bp",
                        "depth": depth,
                    },
                    right={
                        "regime": regime,
                        "architecture": "serial_tree",
                        "credit": "soma_broadcast_bp",
                        "depth": depth,
                    },
                    seed=8_200_000 + counter,
                    detail=f"full BP - soma-broadcast autograd, {regime}, D{depth}",
                )
            )
            counter += 1
            contrasts.append(
                paired(
                    frame,
                    name=f"soma_broadcast_minus_local_shared__{regime}__d{depth}",
                    family="eligibility",
                    left={
                        "regime": regime,
                        "architecture": "serial_tree",
                        "credit": "soma_broadcast_bp",
                        "depth": depth,
                    },
                    right={
                        "regime": regime,
                        "architecture": "serial_tree",
                        "credit": "local_shared",
                        "depth": depth,
                    },
                    seed=8_200_000 + counter,
                    detail=f"autograd eligibility - LocalCA eligibility at shared soma credit, {regime}, D{depth}",
                )
            )
            counter += 1
            contrasts.append(
                paired(
                    frame,
                    name=f"matched_broadcast_minus_local_shared__{regime}__d{depth}",
                    family="optimizer_matched_update_rule",
                    left={
                        "regime": regime,
                        "architecture": "serial_tree",
                        "credit": "soma_broadcast_matched",
                        "depth": depth,
                    },
                    right={
                        "regime": regime,
                        "architecture": "serial_tree",
                        "credit": "local_shared",
                        "depth": depth,
                    },
                    seed=8_200_000 + counter,
                    detail=f"soma-broadcast autograd - LocalCA shared, LocalCA optimizer matched, {regime}, D{depth}",
                )
            )
            counter += 1
            contrasts.append(
                paired(
                    frame,
                    name=f"bp_optimizer_broadcast_minus_matched_broadcast__{regime}__d{depth}",
                    family="optimizer_groups",
                    left={
                        "regime": regime,
                        "architecture": "serial_tree",
                        "credit": "soma_broadcast_bp",
                        "depth": depth,
                    },
                    right={
                        "regime": regime,
                        "architecture": "serial_tree",
                        "credit": "soma_broadcast_matched",
                        "depth": depth,
                    },
                    seed=8_200_000 + counter,
                    detail=f"soma-broadcast BP optimizer - LocalCA optimizer, {regime}, D{depth}",
                )
            )
            counter += 1
            contrasts.append(
                paired(
                    frame,
                    name=f"matched_broadcast_minus_local_path__{regime}__d{depth}",
                    family="optimizer_matched_update_rule",
                    left={
                        "regime": regime,
                        "architecture": "serial_tree",
                        "credit": "soma_broadcast_matched",
                        "depth": depth,
                    },
                    right={
                        "regime": regime,
                        "architecture": "serial_tree",
                        "credit": "local_path",
                        "depth": depth,
                    },
                    seed=8_200_000 + counter,
                    detail=f"soma-broadcast autograd - LocalCA path, LocalCA optimizer matched, {regime}, D{depth}",
                )
            )
            counter += 1
            contrasts.append(
                paired(
                    frame,
                    name=f"local_path_minus_local_shared__{regime}__d{depth}",
                    family="path_transport",
                    left={
                        "regime": regime,
                        "architecture": "serial_tree",
                        "credit": "local_path",
                        "depth": depth,
                    },
                    right={
                        "regime": regime,
                        "architecture": "serial_tree",
                        "credit": "local_shared",
                        "depth": depth,
                    },
                    seed=8_200_000 + counter,
                    detail=f"exact path LocalCA - shared-soma LocalCA, {regime}, D{depth}",
                )
            )
            counter += 1
            contrasts.append(
                paired(
                    frame,
                    name=f"full_bp_minus_local_path__{regime}__d{depth}",
                    family="bp_local_residual",
                    left={
                        "regime": regime,
                        "architecture": "serial_tree",
                        "credit": "full_bp",
                        "depth": depth,
                    },
                    right={
                        "regime": regime,
                        "architecture": "serial_tree",
                        "credit": "local_path",
                        "depth": depth,
                    },
                    seed=8_200_000 + counter,
                    detail=f"full BP - exact-path LocalCA, {regime}, D{depth}",
                )
            )
            counter += 1

    indexed = pd.DataFrame(contrasts).set_index("contrast")
    for depth in (1, 2, 3):
        aligned = indexed.loc[f"serial_minus_star__aligned__d{depth}"]
        rewired = indexed.loc[f"serial_minus_star__rewired_tree__d{depth}"]
        # Compute the interaction from the underlying paired seed differences,
        # not from the two summary means.
        serial_a = _select(
            frame, regime="aligned", architecture="serial_tree", credit="full_bp", depth=depth
        )
        star_a = _select(
            frame, regime="aligned", architecture="all_active_star", credit="full_bp", depth=depth
        )
        serial_r = _select(
            frame, regime="rewired_tree", architecture="serial_tree", credit="full_bp", depth=depth
        )
        star_r = _select(
            frame, regime="rewired_tree", architecture="all_active_star", credit="full_bp", depth=depth
        )
        values = ((serial_a - star_a) - (serial_r - star_r)).to_numpy(float)
        mean, low, high = bootstrap_mean(values, 8_300_000 + depth)
        contrasts.append(
            {
                "contrast": f"serial_star_alignment_interaction__d{depth}",
                "family": "architecture_interaction",
                "detail": "(serial-star) aligned - (serial-star) reversed",
                "left": aligned.detail,
                "right": rewired.detail,
                "n_pairs": len(values),
                "mean_difference": mean,
                "ci95_low": low,
                "ci95_high": high,
                "positive_pairs": int(np.sum(values > 0)),
                "negative_pairs": int(np.sum(values < 0)),
                "ties": int(np.sum(np.isclose(values, 0))),
                "exact_sign_flip_p_two_sided": exact_sign_flip_p(values),
            }
        )

    for match in ("active", "total"):
        contrasts.append(
            paired(
                frame,
                name=f"serial_d3_minus_{match}_matched_point_mlp",
                family="unstructured_point",
                left={
                    "regime": "aligned",
                    "architecture": "serial_tree",
                    "credit": "full_bp",
                    "depth": 3,
                },
                right={
                    "regime": "aligned",
                    "architecture": f"point_mlp_{match}",
                    "credit": "full_bp",
                    "depth": 0,
                },
                seed=8_400_000 + (0 if match == "active" else 1),
                detail=f"serial D3 BP - {match}-parameter-matched point MLP",
            )
        )
    return pd.DataFrame(summaries), pd.DataFrame(contrasts)


def audit_controls(new: pd.DataFrame, combined: pd.DataFrame, audit: dict[str, Any]) -> dict[str, Any]:
    metric_columns = [
        column
        for column in new.columns
        if column.endswith(("_accuracy", "_auc", "_categorical_loglikelihood"))
    ]
    structured = combined[
        combined.architecture.isin(["serial_tree", "all_active_star"])
        & combined.credit.isin(
            ["full_bp", "soma_broadcast_bp", "soma_broadcast_matched"]
        )
    ]
    resource_columns = [
        "trainable_parameters",
        "active_synapses",
        "candidate_synapse_slots",
        "persistent_state_scalars",
    ]
    point_parameter_counts = {}
    if len(new):
        point_parameter_counts = {
            architecture: sorted(map(int, part.trainable_parameters.unique()))
            for architecture, part in new[
                new.architecture.str.startswith("point_mlp")
            ].groupby("architecture")
        }
    audit.update(
        {
            "finite_metrics": bool(
                len(new) == 0
                or np.isfinite(new[metric_columns].to_numpy(float)).all()
            ),
            "nonfinite_alert_rows": int(new.nonfinite_alert.sum()) if len(new) else 0,
            "fallback_mentions": int(new.fallback_mentions.sum()) if len(new) else 0,
            "seeds": sorted(map(int, new.seed.unique())) if len(new) else [],
            "structured_resource_unique_values": {
                column: sorted(map(int, structured[column].unique()))
                for column in resource_columns
            },
            "structured_resource_equal": all(
                structured[column].nunique() == 1 for column in resource_columns
            ),
            "point_parameter_counts": point_parameter_counts,
        }
    )
    audit["all_gates_pass"] = bool(
        audit["status"] == "complete"
        and audit["observed_new_rows"] == audit["expected_new_rows"]
        and audit["finite_metrics"]
        and audit["nonfinite_alert_rows"] == 0
        and audit["structured_resource_equal"]
        and audit["seeds"] == list(range(10200, 10210))
    )
    return audit


def _summary_row(summary: pd.DataFrame, **filters: Any) -> pd.Series:
    part = summary.copy()
    for column, value in filters.items():
        part = part[part[column].eq(value)]
    if len(part) != 1:
        raise ValueError(f"Expected one row for {filters}, got {len(part)}")
    return part.iloc[0]


def _draw_architecture_schematic(ax: plt.Axes) -> None:
    """Three architecture motifs in the credit-tree library vocabulary: an
    unstructured point MLP, parallel pooled arms (grouped star), and the
    serial dendritic tree whose junctions carry the credit path."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    panel_title(ax, "G", "Architecture controls")
    # Point MLP: crossed dense layers, control gray.
    for y, count in ((0.85, 3), (0.51, 4), (0.17, 3)):
        for offset in np.linspace(-0.075, 0.075, count):
            ax.plot(
                [0.17 + offset], [y], marker="o", ms=MS_JUNCTION,
                mfc=COLORS["point_mlp"], mec="none", ls="none", zorder=4,
            )
    for y1, y2 in ((0.85, 0.51), (0.51, 0.17)):
        ax.plot([0.095, 0.245], [y1, y2], color=COLORS["mute"], lw=LW_HAIR)
        ax.plot([0.245, 0.095], [y1, y2], color=COLORS["mute"], lw=LW_HAIR)
    # Grouped star: independent two-segment arms pooled at the soma.
    draw_tree(
        ax, [4, 1], x0=0.50, y0=0.16, radius=0.70, sector_deg=30.0,
        color=COLORS["oracle"],
    )
    # Serial tree: the D3 chain-of-junctions architecture.
    draw_tree(
        ax, [2, 1, 2], x0=0.83, y0=0.16, radius=0.70, sector_deg=28.0,
        color=COLORS["dend"],
    )
    for x, label in (
        (0.17, "point\nMLP"), (0.50, "grouped\nstar"), (0.83, "serial\ntree")
    ):
        ax.text(
            x, 0.055, label, ha="center", va="center", linespacing=1.1,
            fontsize=PT_ANNOT, color=COLORS["ink"],
        )


def _line(
    ax: plt.Axes,
    summary: pd.DataFrame,
    *,
    architecture: str,
    credit: str,
    color: str,
    label: str,
    marker: str,
    linestyle: str = "-",
) -> None:
    part = summary[
        summary.regime.eq("aligned")
        & summary.architecture.eq(architecture)
        & summary.credit.eq(credit)
        & summary.depth.gt(0)
    ].sort_values("depth")
    x = part.depth.to_numpy(float)
    mean = part.mean_test_accuracy.to_numpy(float)
    low = part.ci95_low_test_accuracy.to_numpy(float)
    high = part.ci95_high_test_accuracy.to_numpy(float)
    ax.errorbar(
        x, mean, yerr=np.vstack([mean - low, high - mean]),
        color=color, lw=LW_DATA, elinewidth=LW_ERR, capsize=ERR_CAPSIZE,
        marker=marker, ms=MARKER_MS, markeredgecolor="white", markeredgewidth=0.5,
        linestyle=linestyle, label=label,
    )


def render_figure(summary: pd.DataFrame, contrasts: pd.DataFrame) -> None:
    apply_neurips_style()
    fig, axes = plt.subplots(
        2, 3,
        figsize=(FIG_W, 5.15),
        gridspec_kw={
            "left": 0.115,
            "right": 0.985,
            "bottom": 0.168,
            "top": 0.9125,
            "wspace": 0.62,
            "hspace": 0.74,
        },
    )
    ax_a, ax_b, ax_c, ax_d, ax_e, ax_f = axes.ravel()
    _draw_architecture_schematic(ax_a)

    _line(ax_b, summary, architecture="serial_tree", credit="full_bp", color=COLORS["shunting"], label="serial tree", marker="o")
    # grouped star is always the violet triangle, matching the P-U block.
    _line(ax_b, summary, architecture="all_active_star", credit="full_bp", color=COLORS["oracle"], label="grouped star", marker="^")
    ax_b.set_xlim(0.7, 3.3)
    ax_b.set_xticks([1, 2, 3])
    ax_b.set_ylim(0.44, 1.06)
    ax_b.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax_b.set_xlabel(r"physical depth $D_{\mathrm{p}}$")
    ax_b.set_ylabel("test accuracy")
    panel_title(ax_b, "H", "Serial composition")
    style_axis(ax_b, grid="y")
    # Two series: direct labels instead of a legend.
    ax_b.text(
        3.0, 0.955, "serial tree", ha="right", va="bottom",
        fontsize=PT_LEGEND, color=COLORS["shunting"],
    )
    ax_b.text(
        3.0, 0.572, "grouped star", ha="right", va="top",
        fontsize=PT_LEGEND, color=COLORS["oracle"],
    )

    indexed = contrasts.set_index("contrast")
    names = [
        "serial_minus_star__aligned__d1",
        "serial_minus_star__aligned__d2",
        "serial_minus_star__aligned__d3",
        "serial_star_alignment_interaction__d3",
    ]
    labels = ["D1", "D2", "D3", "D3 × align."]
    part = indexed.loc[names]
    y = np.arange(len(names))[::-1]
    mean = 100 * part.mean_difference.to_numpy(float)
    low = 100 * part.ci95_low.to_numpy(float)
    high = 100 * part.ci95_high.to_numpy(float)
    ax_c.axvline(0, color=COLORS["mute"], lw=LW_HAIR)
    ax_c.errorbar(
        mean, y, xerr=np.vstack([mean - low, high - mean]), fmt="o",
        color=COLORS["shunting"], ecolor=COLORS["shunting"],
        markerfacecolor=COLORS["shunting"], markeredgecolor="white",
        markeredgewidth=0.5, elinewidth=LW_ERR, capsize=ERR_CAPSIZE,
        ms=MARKER_MS,
    )
    ax_c.set_ylim(-0.6, 3.6)
    ax_c.set_yticks(y, labels)
    ax_c.set_xlabel("serial − star (pp)")
    panel_title(ax_c, "I", "Composition contrasts")
    style_axis(ax_c, grid="x")

    credit_specs = [
        ("full_bp", COLORS["bp"], "full BP", "o", "-"),
        ("soma_broadcast_bp", COLORS["additive"], "broadcast (BP opt.)", "s", "--"),
        ("soma_broadcast_matched", COLORS["highlight"], "broadcast (local opt.)", "v", "--"),
        ("local_path", COLORS["pathway"], "path LocalCA", "^", "-."),
        ("local_shared", COLORS["local"], "shared LocalCA", "D", ":"),
    ]
    for credit, color, label, marker, linestyle in credit_specs:
        _line(ax_d, summary, architecture="serial_tree", credit=credit, color=color, label=label, marker=marker, linestyle=linestyle)
    ax_d.set_xlim(0.7, 3.3)
    ax_d.set_xticks([1, 2, 3])
    ax_d.set_ylim(0.44, 1.06)
    ax_d.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax_d.set_xlabel(r"physical depth $D_{\mathrm{p}}$")
    ax_d.set_ylabel("test accuracy")
    panel_title(ax_d, "J", "Credit-coordinate ladder")
    style_axis(ax_d, grid="y")
    clean_legend(
        ax_d, loc="upper center", bbox_to_anchor=(0.50, -0.235),
        fontsize=PT_SMALL, ncol=2, handlelength=2.0,
    )

    names = [
        "full_bp_minus_soma_broadcast__aligned__d3",
        "bp_optimizer_broadcast_minus_matched_broadcast__aligned__d3",
        "matched_broadcast_minus_local_shared__aligned__d3",
        "local_path_minus_local_shared__aligned__d3",
        "full_bp_minus_local_path__aligned__d3",
    ]
    labels = [
        "coordinate\ncost",
        "optimizer\ngroups",
        "eligibility\nresidual",
        "path\nspecificity",
        "BP − path\nLocalCA",
    ]
    part = indexed.loc[names]
    y = np.arange(len(names))[::-1]
    mean = 100 * part.mean_difference.to_numpy(float)
    low = 100 * part.ci95_low.to_numpy(float)
    high = 100 * part.ci95_high.to_numpy(float)
    ax_e.axvline(0, color=COLORS["mute"], lw=LW_HAIR)
    ax_e.errorbar(
        mean, y, xerr=np.vstack([mean - low, high - mean]), fmt="o",
        color=COLORS["additive"], ecolor=COLORS["additive"],
        markerfacecolor=COLORS["additive"], markeredgecolor="white",
        markeredgewidth=0.5, elinewidth=LW_ERR, capsize=ERR_CAPSIZE,
        ms=MARKER_MS,
    )
    ax_e.set_ylim(-0.6, 4.6)
    ax_e.set_yticks(y, labels)
    ax_e.tick_params(axis="y", labelsize=PT_SMALL)
    ax_e.set_xlabel("paired difference (pp)")
    panel_title(ax_e, "K", "BP–local decomposition")
    style_axis(ax_e, grid="x")

    point_rows = summary[
        summary.regime.eq("aligned")
        & (
            ((summary.architecture.eq("serial_tree")) & summary.credit.eq("full_bp") & summary.depth.eq(3))
            | summary.architecture.str.startswith("point_mlp")
        )
    ].copy()
    order = ["point_mlp_active", "point_mlp_total", "serial_tree"]
    point_rows["order"] = point_rows.architecture.map({name: idx for idx, name in enumerate(order)})
    point_rows = point_rows.sort_values("order")
    labels = ["active\nmatch", "total\nmatch", "serial\nD3"]
    x = np.arange(3)
    mean = point_rows.mean_test_accuracy.to_numpy(float)
    low = point_rows.ci95_low_test_accuracy.to_numpy(float)
    high = point_rows.ci95_high_test_accuracy.to_numpy(float)
    colors = [COLORS["point_mlp"], COLORS["point_mlp"], COLORS["shunting"]]
    for xi, mi, li, hi, color in zip(x, mean, low, high, colors):
        ax_f.errorbar(
            [xi], [mi], yerr=np.asarray([[mi - li], [hi - mi]]), fmt="o",
            color=color, ecolor=color, markerfacecolor=color,
            markeredgecolor="white", markeredgewidth=0.5,
            elinewidth=LW_ERR, capsize=ERR_CAPSIZE, ms=MARKER_MS,
        )
    ax_f.set_xlim(-0.7, 2.7)
    ax_f.set_xticks(x, labels)
    ax_f.set_ylim(0.86, 1.015)
    ax_f.set_yticks([0.90, 0.95, 1.00])
    ax_f.set_ylabel("test accuracy")
    panel_title(ax_f, "L", "Point-network controls")
    style_axis(ax_f, grid="y")
    axis_break_note(ax_f, "y axis truncated", loc="lower right")

    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.canvas.draw()
    audit_layout(fig, "fig_point_dendrite_credit_controls")
    audit_text_over_data(fig, "fig_point_dendrite_credit_controls")
    fig.savefig(
        FIGURES / "fig_point_dendrite_credit_controls.pdf",
        metadata={"CreationDate": None, "ModDate": None},
    )
    fig.savefig(FIGURES / "fig_point_dendrite_credit_controls.png", dpi=600)
    plt.close(fig)


def write_report(audit: dict[str, Any], contrasts: pd.DataFrame | None) -> None:
    if contrasts is None:
        text = (
            "# Point--dendrite and BP--local-credit controls\n\n"
            f"Status: incomplete ({audit['observed_new_rows']}/"
            f"{audit['expected_new_rows']} new fits collected).\n"
        )
    else:
        indexed = contrasts.set_index("contrast")
        serial = indexed.loc["serial_minus_star__aligned__d3"]
        interaction = indexed.loc["serial_star_alignment_interaction__d3"]
        broadcast = indexed.loc["full_bp_minus_soma_broadcast__aligned__d3"]
        matched_shared = indexed.loc[
            "matched_broadcast_minus_local_shared__aligned__d3"
        ]
        matched_path = indexed.loc[
            "matched_broadcast_minus_local_path__aligned__d3"
        ]
        optimizer = indexed.loc[
            "bp_optimizer_broadcast_minus_matched_broadcast__aligned__d3"
        ]
        bp_path = indexed.loc["full_bp_minus_local_path__aligned__d3"]
        active_point = indexed.loc["serial_d3_minus_active_matched_point_mlp"]
        total_point = indexed.loc["serial_d3_minus_total_matched_point_mlp"]
        text = f"""# Point--dendrite and BP--local-credit controls

Status: complete; {audit['observed_new_rows']} new fits and the frozen
same-seed BP/LocalCA reference arms passed the numerical and resource gates.

- Serial tree minus resource-identical grouped star at aligned D3:
  {100 * serial.mean_difference:.2f} percentage points
  ({100 * serial.ci95_low:.2f} to {100 * serial.ci95_high:.2f};
  {int(serial.positive_pairs)}/10 positive seeds).
- D3 alignment interaction for serial minus star:
  {100 * interaction.mean_difference:.2f} points
  ({100 * interaction.ci95_low:.2f} to {100 * interaction.ci95_high:.2f}).
- Full BP minus soma-broadcast autograd at aligned D3:
  {100 * broadcast.mean_difference:.2f} points
  ({100 * broadcast.ci95_low:.2f} to {100 * broadcast.ci95_high:.2f}).
- Soma-broadcast autograd minus shared-soma LocalCA at aligned D3, with the
  LocalCA optimizer matched:
  {100 * matched_shared.mean_difference:.2f} points
  ({100 * matched_shared.ci95_low:.2f} to {100 * matched_shared.ci95_high:.2f}).
- Soma-broadcast autograd minus path-transport LocalCA at aligned D3, with the
  LocalCA optimizer matched:
  {100 * matched_path.mean_difference:.2f} points
  ({100 * matched_path.ci95_low:.2f} to {100 * matched_path.ci95_high:.2f}).
- Changing only soma-broadcast from the BP optimizer to the LocalCA optimizer
  cost {100 * optimizer.mean_difference:.2f} points
  ({100 * optimizer.ci95_low:.2f} to {100 * optimizer.ci95_high:.2f}); after
  exact path transport, full BP retained a {100 * bp_path.mean_difference:.2f}
  point advantage ({100 * bp_path.ci95_low:.2f} to
  {100 * bp_path.ci95_high:.2f}).
- Active- and total-parameter-matched point MLP minus serial D3 BP:
  {-100 * active_point.mean_difference:.2f} and
  {-100 * total_point.mean_difference:.2f} points, respectively.

Interpretation is conditional on the frozen calibrated hierarchical gain--load
task. A positive serial--star interaction supports serial divisive composition;
a null contrast assigns the result to grouped computation rather than serial
dendritic depth. The point MLPs test whether this structural resource is an
unconstrained expressivity advantage. The soma-broadcast comparisons separate
teaching-coordinate restriction from the remaining optimizer-matched LocalCA
update-rule gap; they do not turn autograd into a biological mechanism.
"""
    (OUTPUT / "report.md").write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    if not RUNS.exists():
        # Frozen-source render: raw run directories are not in this checkout;
        # restyle the figure from the audited source-data summaries.
        summary = pd.read_csv(OUTPUT / "condition_summary.csv")
        contrasts = pd.read_csv(OUTPUT / "paired_contrasts.csv")
        render_figure(summary, contrasts)
        print("Rendered fig_point_dendrite_credit_controls from frozen source "
              "data (run directories absent; collection skipped).")
        return
    new, audit = collect_new(allow_incomplete=args.allow_incomplete)
    references = reference_rows()
    combined = pd.concat([references, new], ignore_index=True, sort=False)
    audit = audit_controls(new, combined, audit)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    new.to_csv(OUTPUT / "new_seed_outcomes.csv", index=False, float_format="%.10g")
    combined.to_csv(
        OUTPUT / "combined_seed_outcomes.csv", index=False, float_format="%.10g"
    )
    contrasts: pd.DataFrame | None = None
    if audit["status"] == "complete":
        summary, contrasts = summarize(combined)
        summary.to_csv(OUTPUT / "condition_summary.csv", index=False, float_format="%.10g")
        contrasts.to_csv(OUTPUT / "paired_contrasts.csv", index=False, float_format="%.10g")
        render_figure(summary, contrasts)
    (OUTPUT / "audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_report(audit, contrasts)
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
