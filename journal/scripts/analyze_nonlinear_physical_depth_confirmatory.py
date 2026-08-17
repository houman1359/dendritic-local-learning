#!/usr/bin/env python3
"""Audit, analyze, and render the nonlinear physical-depth replication."""

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
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyBboxPatch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from journal_style import (
    COLORS,
    ERR_CAPSIZE,
    FIG_W,
    LW_DATA,
    LW_EDGE,
    LW_ERR,
    LW_HAIR,
    MARKERS,
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
    wrap_ticklabels,
)


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "nonlinear_physical_depth_runs"
SOURCE = ROOT / "source_data" / "nonlinear_physical_depth_confirmatory"
FIGURES = ROOT / "figures" / "generated"

RUN_SPECS = {
    "aligned_additive_bp": ("aligned", "additive", "bp"),
    "aligned_shunting_bp": ("aligned", "shunting", "bp"),
    "aligned_shunting_local3f": ("aligned", "shunting", "local3f"),
    "rewired_tree_shunting_bp": ("rewired_tree", "shunting", "bp"),
    "rewired_tree_shunting_local3f": ("rewired_tree", "shunting", "local3f"),
    "sensor_shuffled_shunting_bp": ("sensor_shuffled", "shunting", "bp"),
    "zero_alignment_shunting_bp": ("zero_alignment", "shunting", "bp"),
}
REGIME_LABEL = {
    "aligned": "aligned",
    "zero_alignment": "zero alignment",
    "sensor_shuffled": "sensor shuffled",
    "rewired_tree": "tree reversed",
}
REGIME_COLOR = {
    "aligned": COLORS["shunting"],
    "zero_alignment": COLORS["mute"],
    "sensor_shuffled": COLORS["highlight"],
    "rewired_tree": COLORS["pathway"],
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def latest_run(stem: str) -> Path:
    matches = sorted(RUNS.glob(f"journal_confirmatory_physical_depth_{stem}_*"))
    if not matches:
        raise FileNotFoundError(f"No run directory for {stem}")
    return matches[-1]


def collect(*, allow_incomplete: bool) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    run_records: list[dict[str, Any]] = []
    for stem, (regime, mechanism, method) in RUN_SPECS.items():
        run = latest_run(stem)
        original = yaml.safe_load((run / "original_config.yaml").read_text())
        expected = int(original["sweep_contract"]["expected_config_count"])
        manifest = run / "frozen_sweep_manifest.json"
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
                "manifest_sha256": sha256(manifest),
            }
        )
        if len(configs) != expected:
            missing.append(f"{stem}: generated {len(configs)}/{expected}")
        for config_path in configs:
            index = int(config_path.stem.rsplit("_", 1)[1])
            result_dir = run / "results" / f"config_{index}"
            final_path = result_dir / "performance" / "final.json"
            resources_path = result_dir / "model_resources.json"
            if not final_path.exists() or not resources_path.exists():
                missing.append(f"{stem}/config_{index}")
                continue
            config = yaml.safe_load(config_path.read_text())
            final = json.loads(final_path.read_text())
            resources = json.loads(resources_path.read_text())
            layer = config["model"]["core"]["population_network"]["layers"][0]
            factors = layer["populations"][0]["branch_factors"]
            strategy = config["training"]["main"]["strategy"]
            transport = (
                "backpropagation"
                if strategy == "standard"
                else config["training"]["main"]["learning_strategy_config"][
                    "error_broadcast_mode"
                ]
            )
            log_text = "\n".join(
                p.read_text(errors="replace")
                for p in (result_dir / "train.log", result_dir / "dendritic_modeling.log")
                if p.exists()
            ).lower()
            row: dict[str, Any] = {
                "cohort": stem,
                "run_dir": str(run.relative_to(ROOT)),
                "config_index": index,
                "seed": int(config["experiment"]["seed"]),
                "regime": regime,
                "mechanism": mechanism,
                "method": method,
                "transport": transport,
                "depth": len(factors),
                "branch_factors": "x".join(map(str, factors)),
                "signal_delta": float(
                    config["data"]["dataset_params"]["hierarchical_gain_load"][
                        "e_signal_delta"
                    ]
                ),
                "train_gain_sigma": float(
                    config["data"]["dataset_params"]["hierarchical_gain_load"][
                        "train_gain_sigma"
                    ]
                ),
                "test_gain_sigma": float(
                    config["data"]["dataset_params"]["hierarchical_gain_load"][
                        "test_gain_sigma"
                    ]
                ),
                "total_parameters": int(resources["total_parameters"]),
                "trainable_parameters": int(resources["trainable_parameters"]),
                "active_synapses": int(resources["active_synapses"]),
                "candidate_synapse_slots": int(resources["candidate_synapse_slots"]),
                "persistent_state_scalars": int(
                    resources["persistent_state_scalars_per_sample"]
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
                    row[f"{split}_{metric}"] = float(final[metric][split])
            rows.append(row)
    if missing and not allow_incomplete:
        raise RuntimeError(
            f"Confirmatory cohort incomplete ({len(missing)} missing): "
            + ", ".join(missing[:12])
        )
    frame = pd.DataFrame(rows)
    audit = {
        "status": "incomplete" if missing else "complete",
        "expected_rows": int(sum(record["expected"] for record in run_records)),
        "observed_rows": len(frame),
        "missing_count": len(missing),
        "missing_examples": missing[:30],
        "runs": run_records,
    }
    return frame, audit


def bootstrap_mean(values: np.ndarray, seed: int, draws: int = 50_000):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(draws, len(values)), replace=True).mean(axis=1)
    low, high = np.quantile(samples, [0.025, 0.975])
    return float(values.mean()), float(low), float(high)


def exact_sign_flip_p(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    observed = abs(float(values.mean()))
    null = [
        abs(float(np.mean(values * np.asarray(signs))))
        for signs in itertools.product((-1.0, 1.0), repeat=len(values))
    ]
    return float(np.mean(np.asarray(null) >= observed - 1e-15))


def record_contrast(
    *, name: str, family: str, values: pd.Series, seed: int, detail: str
) -> dict[str, Any]:
    values = values.sort_index().astype(float)
    mean, low, high = bootstrap_mean(values.to_numpy(), seed)
    return {
        "contrast": name,
        "family": family,
        "detail": detail,
        "n_pairs": len(values),
        "mean_difference": mean,
        "ci95_low": low,
        "ci95_high": high,
        "positive_pairs": int((values > 0).sum()),
        "negative_pairs": int((values < 0).sum()),
        "exact_sign_flip_p_two_sided": exact_sign_flip_p(values.to_numpy()),
    }


def depth_effect(frame: pd.DataFrame, **filters: Any) -> pd.Series:
    part = frame.copy()
    for column, value in filters.items():
        part = part[part[column].eq(value)]
    wide = part.pivot(index="seed", columns="depth", values="test_accuracy")
    if not {1, 3}.issubset(wide.columns):
        raise ValueError(f"Missing depth for {filters}")
    return wide[3] - wide[1]


def summarize(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    group_cols = ["regime", "mechanism", "method", "transport", "depth", "branch_factors"]
    summary_rows = []
    for index, (keys, part) in enumerate(frame.groupby(group_cols, sort=True)):
        row = dict(zip(group_cols, keys))
        row["n_seeds"] = int(part.seed.nunique())
        for metric_index, metric in enumerate(("train_accuracy", "valid_accuracy", "test_accuracy", "test_auc")):
            mean, low, high = bootstrap_mean(
                part[metric].to_numpy(float), 7_100_000 + 100 * index + metric_index
            )
            row[f"mean_{metric}"] = mean
            row[f"ci95_low_{metric}"] = low
            row[f"ci95_high_{metric}"] = high
        summary_rows.append(row)

    contrasts: list[dict[str, Any]] = []
    counter = 0
    for regime in ("aligned", "zero_alignment", "sensor_shuffled", "rewired_tree"):
        values = depth_effect(
            frame, regime=regime, mechanism="shunting", method="bp",
            transport="backpropagation"
        )
        contrasts.append(
            record_contrast(
                name=f"bp_depth_d3_minus_d1__{regime}",
                family="bp_depth",
                values=values,
                seed=7_200_000 + counter,
                detail=f"shunting BP D3-D1, {regime}",
            )
        )
        counter += 1
    aligned_bp = depth_effect(
        frame, regime="aligned", mechanism="shunting", method="bp",
        transport="backpropagation"
    )
    for control in ("zero_alignment", "sensor_shuffled", "rewired_tree"):
        controlled = depth_effect(
            frame, regime=control, mechanism="shunting", method="bp",
            transport="backpropagation"
        )
        contrasts.append(
            record_contrast(
                name=f"bp_depth_interaction__aligned_minus_{control}",
                family="bp_interaction",
                values=aligned_bp - controlled,
                seed=7_200_000 + counter,
                detail=f"(D3-D1) aligned - {REGIME_LABEL[control]}",
            )
        )
        counter += 1

    for transport in ("per_soma_shared", "path_transport"):
        aligned = depth_effect(
            frame, regime="aligned", mechanism="shunting", method="local3f",
            transport=transport
        )
        rewired = depth_effect(
            frame, regime="rewired_tree", mechanism="shunting", method="local3f",
            transport=transport
        )
        contrasts.append(
            record_contrast(
                name=f"local_depth_d3_minus_d1__aligned__{transport}",
                family="local_depth",
                values=aligned,
                seed=7_200_000 + counter,
                detail=f"aligned LocalCA D3-D1, {transport}",
            )
        )
        counter += 1
        contrasts.append(
            record_contrast(
                name=f"local_depth_interaction__aligned_minus_rewired__{transport}",
                family="local_interaction",
                values=aligned - rewired,
                seed=7_200_000 + counter,
                detail=f"LocalCA aligned-reversed depth effect, {transport}",
            )
        )
        counter += 1

    for depth in (1, 2, 3):
        part = frame[
            frame.regime.eq("aligned")
            & frame.method.eq("bp")
            & frame.transport.eq("backpropagation")
            & frame.depth.eq(depth)
        ]
        wide = part.pivot(index="seed", columns="mechanism", values="test_accuracy")
        contrasts.append(
            record_contrast(
                name=f"mechanism_shunting_minus_additive__d{depth}",
                family="mechanism",
                values=wide["shunting"] - wide["additive"],
                seed=7_200_000 + counter,
                detail=f"aligned BP shunting-additive, D{depth}",
            )
        )
        counter += 1
    return pd.DataFrame(summary_rows), pd.DataFrame(contrasts)


def audit_resources(frame: pd.DataFrame, audit: dict[str, Any]) -> dict[str, Any]:
    finite_columns = [
        column
        for column in frame.columns
        if column.endswith(("_accuracy", "_auc", "_categorical_loglikelihood"))
    ]
    resource_columns = [
        "total_parameters",
        "trainable_parameters",
        "active_synapses",
        "candidate_synapse_slots",
        "persistent_state_scalars",
    ]
    aligned_bp = frame[
        frame.regime.eq("aligned")
        & frame.mechanism.eq("shunting")
        & frame.method.eq("bp")
        & frame.transport.eq("backpropagation")
    ]
    accessibility = {
        f"D{depth}": {
            "mean_test_accuracy": float(part.test_accuracy.mean()),
            "seeds_above_0_55": int((part.test_accuracy > 0.55).sum()),
            "n_seeds": int(part.seed.nunique()),
        }
        for depth in (1, 2, 3)
        for part in [aligned_bp[aligned_bp.depth.eq(depth)]]
    }
    accessibility_gate = all(
        row["n_seeds"] == 10
        and row["seeds_above_0_55"] >= 9
        and row["mean_test_accuracy"] < 0.98
        for row in accessibility.values()
    )
    audit.update(
        {
            "finite_metrics": bool(np.isfinite(frame[finite_columns].to_numpy(float)).all()),
            "nonfinite_alert_rows": int(frame.nonfinite_alert.sum()),
            "local_fallback_mentions": int(
                frame.loc[frame.method.eq("local3f"), "fallback_mentions"].sum()
            ),
            "resource_unique_values": {
                column: sorted(map(int, frame[column].unique()))
                for column in resource_columns
            },
            "resource_equal": all(frame[column].nunique() == 1 for column in resource_columns),
            "seeds": sorted(map(int, frame.seed.unique())),
            "aligned_bp_accessibility": accessibility,
            "aligned_bp_accessibility_gate": accessibility_gate,
        }
    )
    audit["all_gates_pass"] = bool(
        audit["status"] == "complete"
        and audit["observed_rows"] == 270
        and audit["finite_metrics"]
        and audit["nonfinite_alert_rows"] == 0
        and audit["local_fallback_mentions"] == 0
        and audit["resource_equal"]
        and audit["seeds"] == list(range(10200, 10210))
        and audit["aligned_bp_accessibility_gate"]
    )
    return audit


def audit_inference(contrasts: pd.DataFrame, audit: dict[str, Any]) -> dict[str, Any]:
    """Apply the frozen claim gates without changing the prespecified estimates."""
    indexed = contrasts.set_index("contrast")
    names = [
        "bp_depth_d3_minus_d1__aligned",
        "bp_depth_interaction__aligned_minus_zero_alignment",
        "bp_depth_interaction__aligned_minus_sensor_shuffled",
        "bp_depth_interaction__aligned_minus_rewired_tree",
        "local_depth_d3_minus_d1__aligned__per_soma_shared",
        "local_depth_d3_minus_d1__aligned__path_transport",
        "local_depth_interaction__aligned_minus_rewired__per_soma_shared",
        "local_depth_interaction__aligned_minus_rewired__path_transport",
    ]
    claim_gates: dict[str, dict[str, Any]] = {}
    for name in names:
        row = indexed.loc[name]
        interval_is_positive = bool(row.ci95_low > 0)
        sign_count = int(row.positive_pairs)
        claim_gates[name] = {
            "mean_difference": float(row.mean_difference),
            "ci95_low": float(row.ci95_low),
            "ci95_high": float(row.ci95_high),
            "positive_seed_signs": sign_count,
            "claim_gate_pass": bool(interval_is_positive and sign_count >= 8),
        }
    audit["claim_gates"] = claim_gates
    return audit


def _summary_row(summary: pd.DataFrame, **filters: Any) -> pd.Series:
    part = summary.copy()
    for column, value in filters.items():
        part = part[part[column].eq(value)]
    if len(part) != 1:
        raise ValueError(f"Expected one summary row, got {len(part)} for {filters}")
    return part.iloc[0]


def draw_tree(
    ax: plt.Axes,
    factors: list[int],
    *,
    x0: float,
    y0: float,
    radius: float,
    sector_deg: float,
    color: str,
    node_r: float = 0.011,
) -> list[list[tuple[float, float]]]:
    """Radial dendritic tree: soma disc at the root, tapering dend strokes.

    ``branch_factors`` is ordered soma-to-distal in the production DendriNet
    planner; that order is preserved.  Leaves are spread evenly on an arc so
    an 8-way fan stays readable, and stroke width tapers with the cumulative
    branch count (a unary stage reads as a serial compartment: a straight
    continuation through a junction glyph).  Returns node coordinates per
    level (level 0 is the soma) so callers can anchor annotations.
    """
    n_leaves = int(np.prod(factors))
    leaf_angles = (
        np.deg2rad(np.linspace(-sector_deg / 2.0, sector_deg / 2.0, n_leaves))
        if n_leaves > 1
        else np.zeros(1)
    )
    step = radius / len(factors)
    current: list[tuple[float, float, int, int]] = [(x0, y0, 0, n_leaves)]
    coords: list[list[tuple[float, float]]] = [[(x0, y0)]]
    cumulative = 1
    for level, factor in enumerate(factors, start=1):
        cumulative *= factor
        lw = max(LW_HAIR, LW_DATA * cumulative ** -0.45)
        r = level * step
        next_nodes: list[tuple[float, float, int, int]] = []
        for px, py, lo, hi in current:
            span = (hi - lo) // factor
            for child in range(factor):
                clo, chi = lo + child * span, lo + (child + 1) * span
                angle = float(leaf_angles[clo:chi].mean())
                cx, cy = x0 + r * np.sin(angle), y0 + r * np.cos(angle)
                ax.plot(
                    [px, cx], [py, cy], color=color, lw=lw,
                    solid_capstyle="round", zorder=3,
                )
                next_nodes.append((cx, cy, clo, chi))
        coords.append([(nx, ny) for nx, ny, _, _ in next_nodes])
        current = next_nodes
    for level_nodes in coords[1:]:
        for nx, ny in level_nodes:
            ax.add_patch(
                Circle((nx, ny), node_r, facecolor="white",
                       edgecolor=color, lw=LW_EDGE, zorder=4)
            )
    ax.add_patch(
        Circle((x0, y0), 0.030, facecolor=COLORS["soma"],
               edgecolor=COLORS["edge"], lw=LW_EDGE, zorder=5)
    )
    return coords


def panel_schematic(ax: plt.Axes) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    panel_title(ax, "A", "Matched-resource depth")
    specs = [
        (0.18, [8], 48.0, "D1"),
        (0.50, [2, 3], 44.0, "D2"),
        (0.82, [2, 1, 2], 40.0, "D3"),
    ]
    for x, factors, sector, name in specs:
        draw_tree(
            ax, factors, x0=x, y0=0.40, radius=0.30, sector_deg=sector,
            color=COLORS["dend"],
        )
        ax.text(
            x, 0.915, name,
            ha="center", va="center", fontsize=PT_ANNOT, color=COLORS["ink"],
        )
        ax.text(
            x, 0.845, rf"$D_{{\mathrm{{p}}}}={name[-1]}$",
            ha="center", va="center", fontsize=PT_SMALL, color=COLORS["mute"],
        )
        ax.text(
            x, 0.780, "[" + ",".join(map(str, factors)) + "]",
            ha="center", va="center", fontsize=PT_SMALL, color=COLORS["mute"],
        )
    ax.text(
        0.50,
        0.325,
        "entries: soma → distal · 8 branch units each\n"
        "contacts · parameters · states matched",
        ha="center",
        va="top",
        linespacing=1.2,
        fontsize=PT_SMALL,
        color=COLORS["mute"],
    )


def _scope_box(
    ax: plt.Axes,
    points: list[tuple[float, float]],
    pad: float,
    edge: str,
) -> tuple[float, float, float, float]:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x_lo, x_hi = min(xs) - pad, max(xs) + pad
    y_lo, y_hi = min(ys) - pad, max(ys) + pad
    ax.add_patch(
        FancyBboxPatch(
            (x_lo, y_lo), x_hi - x_lo, y_hi - y_lo,
            boxstyle="round,pad=0.012,rounding_size=0.035",
            facecolor="#20509E", alpha=0.05, edgecolor="none", zorder=1,
        )
    )
    ax.add_patch(
        FancyBboxPatch(
            (x_lo, y_lo), x_hi - x_lo, y_hi - y_lo,
            boxstyle="round,pad=0.012,rounding_size=0.035",
            facecolor="none", edgecolor=edge, lw=LW_EDGE, zorder=2,
        )
    )
    return x_lo, x_hi, y_lo, y_hi


def panel_task(ax: plt.Axes) -> None:
    """Nested gain scopes drawn directly on a dendritic tree (fine ⊂ coarse
    ⊂ global), with leader lines to the gain symbols they modulate."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    panel_title(ax, "B", "Nested divisive task")
    coords = draw_tree(
        ax, [2, 1, 2], x0=0.28, y0=0.40, radius=0.31, sector_deg=50.0,
        color=COLORS["dend"],
    )
    soma = coords[0][0]
    n1_r = coords[1][1]
    n2_r = coords[2][1]
    leaves = coords[3]
    # Nested scopes, innermost darkest: one distal branch (fine), one soma
    # subtree (coarse), the whole tree (global).  Stacked light fills shade
    # the nesting; containment encodes the hierarchy, so no extra hues.
    fine = _scope_box(ax, [n2_r, leaves[3]], 0.032, "#20509E")
    coarse = _scope_box(ax, [n1_r, n2_r, leaves[2], leaves[3]], 0.050, "#5580BE")
    tree_pts = [soma, (soma[0], soma[1] - 0.030)] + [p for lvl in coords for p in lvl]
    whole = _scope_box(ax, tree_pts, 0.085, "#93B3DB")
    for (x_hi, y_at), label in (
        ((fine[1], 0.64), "fine  $G_f$"),
        ((coarse[1], 0.47), "coarse  $G_c$"),
        ((whole[1], 0.30), "global  $G_g$"),
    ):
        ax.plot(
            [x_hi + 0.012, 0.625], [y_at, y_at],
            color=COLORS["mute"], lw=LW_HAIR, zorder=2,
        )
        ax.text(
            0.64, y_at, label,
            ha="left", va="center", fontsize=PT_ANNOT, color=COLORS["ink"],
        )
    ax.text(
        0.50, 0.205,
        r"$E_{\mathrm{distal}}=s_y\,G_f\,G_c\,G_g$",
        ha="center", va="center", fontsize=PT_ANNOT, color=COLORS["ink"],
    )
    ax.text(
        0.50, 0.080,
        "matched sensors · shuffled/reversed controls",
        ha="center", va="center", fontsize=PT_SMALL, color=COLORS["mute"],
    )


def line_summary(
    ax: plt.Axes,
    summary: pd.DataFrame,
    handles: list[Line2D],
    *,
    regime: str,
    mechanism: str,
    method: str,
    transport: str,
    color: str,
    label: str,
    ls: Any = "-",
    marker: str = "o",
    dx: float = 0.0,
) -> None:
    """Mean line at true x plus 95% CI error bars under the series marker.

    ``dx`` shifts only the marker/error-bar column so honestly coincident
    series stay visible; the shift is declared in an in-panel note.  ``ls``
    accepts an (offset, dash) tuple so two coincident curves can interleave
    their dashes instead of silently occluding one another.
    """
    part = summary[
        summary.regime.eq(regime)
        & summary.mechanism.eq(mechanism)
        & summary.method.eq(method)
        & summary.transport.eq(transport)
    ].sort_values("depth")
    x = part.depth.to_numpy(float)
    mean = part.mean_test_accuracy.to_numpy(float)
    low = part.ci95_low_test_accuracy.to_numpy(float)
    high = part.ci95_high_test_accuracy.to_numpy(float)
    ax.plot(x, mean, color=color, lw=LW_DATA, ls=ls, zorder=2)
    ax.errorbar(
        x + dx, mean, yerr=np.vstack([mean - low, high - mean]),
        fmt=marker, ms=MARKER_MS, color=color,
        markerfacecolor=color, markeredgecolor="white", markeredgewidth=0.5,
        ecolor=color, elinewidth=LW_ERR, capsize=ERR_CAPSIZE, zorder=3,
    )
    legend_ls = ls if isinstance(ls, str) else (0.0, ls[1])
    handles.append(
        Line2D(
            [], [], color=color, lw=LW_DATA, ls=legend_ls,
            marker=marker, ms=MARKER_MS, markerfacecolor=color,
            markeredgecolor="white", markeredgewidth=0.5, label=label,
        )
    )


def render_figure(summary: pd.DataFrame, contrasts: pd.DataFrame) -> None:
    apply_neurips_style()
    fig, axes = plt.subplots(
        2, 3, figsize=(FIG_W, 5.00),
        gridspec_kw={"left": 0.115, "right": 0.985, "bottom": 0.10, "top": 0.91, "wspace": 0.62, "hspace": 0.74},
    )
    ax_a, ax_b, ax_c, ax_d, ax_e, ax_f = axes.ravel()
    panel_schematic(ax_a)
    panel_task(ax_b)

    # C — the two flat controls coincide to <0.1 pp, so their dashes are
    # interleaved and every marker column is dodged by a declared offset.
    dashes = {
        "aligned": "-",
        "zero_alignment": (0.0, (3.2, 3.2)),
        "sensor_shuffled": (3.2, (3.2, 3.2)),
        "rewired_tree": (0.0, (3.2, 1.8)),
    }
    offsets = {
        "aligned": -0.055,
        "zero_alignment": -0.055,
        "sensor_shuffled": 0.055,
        "rewired_tree": 0.055,
    }
    handles_c: list[Line2D] = []
    for marker, regime in zip(
        MARKERS, ("aligned", "zero_alignment", "sensor_shuffled", "rewired_tree")
    ):
        line_summary(
            ax_c, summary, handles_c, regime=regime, mechanism="shunting",
            method="bp", transport="backpropagation", color=REGIME_COLOR[regime],
            label=REGIME_LABEL[regime], ls=dashes[regime], marker=marker,
            dx=offsets[regime],
        )
    ax_c.set_xlim(0.7, 3.3)
    ax_c.set_xticks([1, 2, 3])
    ax_c.set_xlabel(r"physical depth $D_{\mathrm{p}}$")
    ax_c.set_ylabel("test accuracy")
    ax_c.set_ylim(0.44, 1.06)
    ax_c.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    panel_title(ax_c, "C", "Backprop depth test")
    style_axis(ax_c, grid="y")
    clean_legend(
        ax_c, fontsize=PT_SMALL, loc="upper left", handles=handles_c,
        handlelength=2.6,
    )
    ax_c.text(
        0.02, 0.085, "flat controls coincide\nmarkers offset · bars: 95% CI",
        transform=ax_c.transAxes, ha="left", va="center", linespacing=1.25,
        fontsize=PT_SMALL, color=COLORS["mute"],
    )

    # D — the four prespecified contrasts all sit near +31 pp, so the axis
    # is truncated (and says so) to make the seed-bootstrap CIs readable.
    forest_names = [
        "bp_depth_d3_minus_d1__aligned",
        "bp_depth_interaction__aligned_minus_zero_alignment",
        "bp_depth_interaction__aligned_minus_sensor_shuffled",
        "bp_depth_interaction__aligned_minus_rewired_tree",
    ]
    labels = [
        "aligned D3−D1", "vs zero alignment",
        "vs sensor shuffled", "vs tree reversed",
    ]
    part = contrasts.set_index("contrast").loc[forest_names]
    y = np.arange(len(part))[::-1]
    mean = 100 * part.mean_difference.to_numpy(float)
    low = 100 * part.ci95_low.to_numpy(float)
    high = 100 * part.ci95_high.to_numpy(float)
    ax_d.errorbar(
        mean, y, xerr=np.vstack([mean - low, high - mean]),
        fmt=MARKERS[3], ms=MARKER_MS, color=COLORS["shunting"],
        markerfacecolor=COLORS["shunting"], markeredgecolor="white",
        markeredgewidth=0.5, ecolor=COLORS["shunting"], elinewidth=LW_ERR,
        capsize=ERR_CAPSIZE, zorder=3,
    )
    ax_d.set_yticks(y)
    ax_d.set_yticklabels(wrap_ticklabels(labels, width=10))
    ax_d.set_ylim(-0.9, 3.6)
    ax_d.set_xlim(30.3, 32.05)
    ax_d.set_xticks([30.5, 31.0, 31.5, 32.0])
    ax_d.tick_params(axis="y", labelsize=PT_SMALL)
    ax_d.set_xlabel("paired effect\n(percentage points)")
    panel_title(ax_d, "D", "Prespecified contrasts")
    style_axis(ax_d, grid="x")
    ax_d.text(
        0.03, 0.965, "all 95% CIs exclude zero",
        transform=ax_d.transAxes, ha="left", va="top",
        fontsize=PT_SMALL, color=COLORS["mute"],
    )
    axis_break_note(ax_d, "x axis truncated", loc="lower right")

    # E — both reversed curves coincide, and all four series meet at depth 1;
    # dashes interleave and marker columns are dodged as declared below.
    handles_e: list[Line2D] = []
    for index, (transport, color, short) in enumerate((
        ("per_soma_shared", COLORS["local"], "shared"),
        ("path_transport", COLORS["oracle"], "path"),
    )):
        line_summary(
            ax_e, summary, handles_e, regime="aligned", mechanism="shunting",
            method="local3f", transport=transport, color=color,
            label=f"{short}, aligned", ls="-", marker=MARKERS[2 * index],
            dx=-0.165 + 0.11 * index,
        )
        line_summary(
            ax_e, summary, handles_e, regime="rewired_tree",
            mechanism="shunting", method="local3f", transport=transport,
            color=color, label=f"{short}, reversed",
            ls=(3.2 * index, (3.2, 3.2)), marker=MARKERS[2 * index + 1],
            dx=0.165 - 0.11 * index,
        )
    ax_e.set_xlim(0.7, 3.3)
    ax_e.set_xticks([1, 2, 3])
    ax_e.set_xlabel(r"physical depth $D_{\mathrm{p}}$")
    ax_e.set_ylabel("test accuracy")
    ax_e.set_ylim(0.44, 1.06)
    ax_e.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    panel_title(ax_e, "E", "Local credit transport")
    style_axis(ax_e, grid="y")
    clean_legend(
        ax_e, fontsize=PT_SMALL, loc="upper left", handles=handles_e,
        handlelength=2.6,
    )
    ax_e.text(
        0.02, 0.085, "reversed curves coincide\nmarkers offset · bars: 95% CI",
        transform=ax_e.transAxes, ha="left", va="center", linespacing=1.25,
        fontsize=PT_SMALL, color=COLORS["mute"],
    )

    # F — same uncertainty treatment as C and E (95% CI error bars).
    handles_f: list[Line2D] = []
    line_summary(
        ax_f, summary, handles_f, regime="aligned", mechanism="shunting",
        method="bp", transport="backpropagation", color=COLORS["shunting"],
        label="shunting", marker=MARKERS[0],
    )
    line_summary(
        ax_f, summary, handles_f, regime="aligned", mechanism="additive",
        method="bp", transport="backpropagation", color=COLORS["additive"],
        label="raw additive", marker=MARKERS[1],
    )
    ax_f.set_xlim(0.7, 3.3)
    ax_f.set_xticks([1, 2, 3])
    ax_f.set_xlabel(r"physical depth $D_{\mathrm{p}}$")
    ax_f.set_ylabel("test accuracy")
    ax_f.set_ylim(0.44, 1.06)
    ax_f.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    panel_title(ax_f, "F", "Divisive control")
    style_axis(ax_f, grid="y")
    clean_legend(ax_f, fontsize=PT_LEGEND, loc="upper left", handles=handles_f)
    ax_f.text(
        0.02, 0.05, "bars: 95% CI",
        transform=ax_f.transAxes, ha="left", va="center",
        fontsize=PT_SMALL, color=COLORS["mute"],
    )

    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.canvas.draw()
    audit_layout(fig, "fig_nonlinear_physical_depth")
    audit_text_over_data(fig, "fig_nonlinear_physical_depth")
    fig.savefig(FIGURES / "fig_nonlinear_physical_depth.pdf", metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(FIGURES / "fig_nonlinear_physical_depth.png", dpi=600)
    plt.close(fig)


def write_report(summary: pd.DataFrame, contrasts: pd.DataFrame, audit: dict[str, Any]) -> str:
    indexed = contrasts.set_index("contrast")
    primary = indexed.loc["bp_depth_d3_minus_d1__aligned"]
    interactions = [
        indexed.loc["bp_depth_interaction__aligned_minus_zero_alignment"],
        indexed.loc["bp_depth_interaction__aligned_minus_sensor_shuffled"],
        indexed.loc["bp_depth_interaction__aligned_minus_rewired_tree"],
    ]
    shared = indexed.loc["local_depth_d3_minus_d1__aligned__per_soma_shared"]
    path = indexed.loc["local_depth_d3_minus_d1__aligned__path_transport"]
    shared_interaction = indexed.loc[
        "local_depth_interaction__aligned_minus_rewired__per_soma_shared"
    ]
    path_interaction = indexed.loc[
        "local_depth_interaction__aligned_minus_rewired__path_transport"
    ]
    condition_lines = []
    for depth in (1, 2, 3):
        row = _summary_row(
            summary, regime="aligned", mechanism="shunting", method="bp",
            transport="backpropagation", depth=depth
        )
        condition_lines.append(f"D{depth} {row.mean_test_accuracy:.4f}")
    interaction_text = "; ".join(
        f"{row['detail']}: {100*row['mean_difference']:.2f} pp "
        f"({100*row['ci95_low']:.2f}, {100*row['ci95_high']:.2f})"
        for row in interactions
    )
    control_depth_text = "; ".join(
        f"{REGIME_LABEL[regime]}: {100*indexed.loc[f'bp_depth_d3_minus_d1__{regime}', 'mean_difference']:.2f} pp"
        for regime in ("zero_alignment", "sensor_shuffled", "rewired_tree")
    )
    additive_lines = []
    for depth in (1, 2, 3):
        row = _summary_row(
            summary, regime="aligned", mechanism="additive", method="bp",
            transport="backpropagation", depth=depth
        )
        additive_lines.append(f"D{depth} {row.mean_test_accuracy:.4f}")
    report = f"""# Confirmatory nonlinear physical-depth result

Completeness and numerical gate: **{'pass' if audit['all_gates_pass'] else 'fail'}**.
The collector found {audit['observed_rows']}/{audit['expected_rows']} frozen fits,
finite metrics, {audit['local_fallback_mentions']} LocalCA fallback mentions, and
resource equality={audit['resource_equal']}.

Aligned shunting BP accuracy was {', '.join(condition_lines)}. The paired D3-D1
effect was {100*primary.mean_difference:.2f} percentage points (seed-bootstrap
95% interval {100*primary.ci95_low:.2f} to {100*primary.ci95_high:.2f};
{int(primary.positive_pairs)}/{int(primary.n_pairs)} positive pairs; exact
sign-flip p={primary.exact_sign_flip_p_two_sided:.4g}).

Topology controls: {interaction_text}.
Their raw D3-D1 effects were {control_depth_text}. Aligned raw-additive BP was
{', '.join(additive_lines)}.

Under LocalCA, aligned D3-D1 was {100*shared.mean_difference:.2f} pp
({100*shared.ci95_low:.2f}, {100*shared.ci95_high:.2f}) for shared-soma feedback
    and {100*path.mean_difference:.2f} pp ({100*path.ci95_low:.2f},
    {100*path.ci95_high:.2f}) for exact path transport. Their aligned-minus-reversed
depth interactions were {100*shared_interaction.mean_difference:.2f} pp
({100*shared_interaction.ci95_low:.2f}, {100*shared_interaction.ci95_high:.2f})
and {100*path_interaction.mean_difference:.2f} pp
({100*path_interaction.ci95_low:.2f}, {100*path_interaction.ci95_high:.2f}),
respectively.

Interpretation must follow the interaction signs. A positive aligned depth
effect alone shows forward capacity, not task-tree specificity. Raw additive
and grouped-point controls bound shunting-specific and dendrite-specific claims.
The operating point was selected in transparent exploratory pilots; inference
here uses only fresh seeds 10200--10209.
"""
    (SOURCE / "report.md").write_text(report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--no-figure", action="store_true")
    args = parser.parse_args()
    frame, audit = collect(allow_incomplete=args.allow_incomplete)
    SOURCE.mkdir(parents=True, exist_ok=True)
    frame.to_csv(SOURCE / "seed_outcomes.csv", index=False, float_format="%.10g")
    if audit["status"] != "complete":
        (SOURCE / "audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True))
        print(json.dumps(audit, indent=2, sort_keys=True))
        return
    audit = audit_resources(frame, audit)
    summary, contrasts = summarize(frame)
    audit = audit_inference(contrasts, audit)
    summary.to_csv(SOURCE / "condition_summary.csv", index=False, float_format="%.10g")
    contrasts.to_csv(SOURCE / "paired_contrasts.csv", index=False, float_format="%.10g")
    (SOURCE / "audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True))
    report = write_report(summary, contrasts, audit)
    if not args.no_figure:
        render_figure(summary, contrasts)
    print(report)
    if not audit["all_gates_pass"]:
        raise SystemExit("Confirmatory artifact gate failed")


if __name__ == "__main__":
    main()
