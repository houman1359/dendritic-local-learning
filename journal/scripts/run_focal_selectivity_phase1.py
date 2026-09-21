#!/usr/bin/env python3
"""Run the frozen passive focal-shunt dose/selectivity matrix.

This phase crosses electrotonic regime, distributed background conductance,
fixed-relative dose, fixed-absolute dose, and input-conductance-normalized dose.
Rank-one updates are evaluated exactly with Sherman--Morrison while somatic
voltage and the scalar output error are restored after each intervention.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from journal_style import (
    COLORS,
    FIG_W,
    LW_DATA,
    PT_LEGEND,
    apply_neurips_style,
    audit_layout,
    audit_text_over_data,
    clean_legend,
    panel_title,
    style_axis,
)


ROOT = Path(__file__).resolve().parents[1]
RECON = ROOT / "code" / "reconstructed_tree"
if str(RECON) not in sys.path:
    sys.path.insert(0, str(RECON))

from analyze_physical_cable_sensitivity import physical_conductance_system  # noqa: E402
from run_focal_shunting_credit_perturbation import (  # noqa: E402
    choose_focal_sites,
    stable_rng,
)


CONFIG = ROOT / "configs" / "focal_selectivity" / "phase1_passive_matrix.json"
SEGMENTS = ROOT / "source_data" / "figure3" / "segment_metrics.csv"
SOURCE = ROOT / "source_data" / "focal_selectivity_phase1"
FIGURES = ROOT / "figures" / "generated"
CANARY = ROOT / "analysis" / "focal_selectivity_phase1_canary.json"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def intervention_state(
    inverse: np.ndarray,
    rhs: np.ndarray,
    baseline_voltage: np.ndarray,
    soma_index: int,
    focal_index: int,
    eta: float,
    inhibitory_reversal: float,
    perturbation: str,
) -> tuple[np.ndarray, np.ndarray, float]:
    column = inverse[:, focal_index]
    soma_column = inverse[:, soma_index]
    if perturbation == "focal shunt":
        denominator = 1.0 + eta * inverse[focal_index, focal_index]

        def solve(vector: np.ndarray) -> np.ndarray:
            first = inverse @ vector
            return first - eta * column * (column @ vector) / denominator

        changed_rhs = rhs.copy()
        changed_rhs[focal_index] += eta * inhibitory_reversal
        voltage_unclamped = solve(changed_rhs)
        soma_response = soma_column - eta * column * column[soma_index] / denominator
        adjoint = soma_response
    elif perturbation == "matched additive":
        voltage_unclamped = baseline_voltage + eta * (
            inhibitory_reversal - baseline_voltage[focal_index]
        ) * column
        soma_response = soma_column
        adjoint = soma_column
    else:
        raise ValueError(perturbation)
    clamp = (
        baseline_voltage[soma_index] - voltage_unclamped[soma_index]
    ) / soma_response[soma_index]
    voltage = voltage_unclamped + clamp * soma_response
    residual = abs(voltage[soma_index] - baseline_voltage[soma_index])
    return voltage, adjoint, float(residual)


def scheme_eta(
    scheme: str,
    value: float,
    local_scale: float,
    input_resistance: float,
) -> float:
    if scheme == "relative_local":
        return float(value * local_scale)
    if scheme == "fixed_absolute_ns":
        return float(value)
    if scheme == "input_conductance_normalized":
        return float(value / input_resistance)
    raise ValueError(scheme)


def relation_metrics(
    base_gradient: np.ndarray,
    changed_gradient: np.ndarray,
    indices: np.ndarray,
) -> dict[str, float]:
    base = base_gradient[indices]
    changed = changed_gradient[indices]
    epsilon = 1e-15 * max(float(np.max(np.abs(base_gradient))), 1.0)
    signed_log = np.log((np.abs(changed) + epsilon) / (np.abs(base) + epsilon))
    return {
        "median_abs_log_change": float(np.median(np.abs(signed_log))),
        "median_signed_log_change": float(np.median(signed_log)),
        "attenuated_fraction": float(np.mean(signed_log < 0)),
        "enhanced_fraction": float(np.mean(signed_log > 0)),
        "sign_flip_fraction": float(np.mean(np.signbit(changed) != np.signbit(base))),
        "gradient_energy_ratio": float(
            np.sum(changed * changed) / max(float(np.sum(base * base)), 1e-30)
        ),
    }


def run_cell_regime(
    root_id: int,
    segments: pd.DataFrame,
    cfg: dict,
    rm: float,
    background_multiplier: float,
    *,
    canary: bool,
) -> list[dict]:
    electrical, matrix, rhs, soma_index, parents, children = physical_conductance_system(
        segments,
        float(cfg["e_scale"]),
        float(cfg["i_scale"]),
        float(cfg["excitatory_reversal"]),
        float(cfg["inhibitory_reversal"]),
        axial_resistivity_ohm_cm=float(cfg["axial_resistivity_ohm_cm"]),
        membrane_resistance_ohm_cm2=float(rm),
    )
    background = background_multiplier * electrical.g_leak.to_numpy(float)
    matrix = matrix.copy()
    matrix[np.diag_indices(len(matrix))] += background
    rhs = rhs + background * float(cfg["background_reversal"])
    minimum_eigenvalue = float(np.linalg.eigvalsh(matrix)[0])
    if minimum_eigenvalue <= 0:
        raise RuntimeError("non-positive conductance matrix")
    inverse = np.linalg.inv(matrix)
    baseline_voltage = inverse @ rhs
    soma_error = 1.0
    baseline_adjoint = inverse[:, soma_index] * soma_error
    baseline_gradient_all = baseline_adjoint * (
        float(cfg["excitatory_reversal"]) - baseline_voltage
    )
    ids = electrical.segment_id.astype(int).tolist()
    index = {segment: position for position, segment in enumerate(ids)}
    e_segments = electrical.loc[electrical.E_size > 0, "segment_id"].astype(int).tolist()
    e_indices = np.asarray([index[segment] for segment in e_segments], dtype=int)
    base_gradient = baseline_gradient_all[e_indices]
    focal_sites, relations = choose_focal_sites(
        electrical,
        e_segments,
        parents,
        children,
        stable_rng(int(cfg["seed"]), root_id, stream=17),
        1 if canary else int(cfg["max_focal_sites_per_cell"]),
        int(cfg["minimum_descendant_and_comparison_sites"]),
    )
    rows: list[dict] = []
    dose_schemes = cfg["dose_schemes"]
    if canary:
        dose_schemes = {name: [values[len(values) // 2]] for name, values in dose_schemes.items()}
    for focal in focal_sites:
        focal_index = index[int(focal)]
        c_k = inverse[:, focal_index]
        input_resistance = float(inverse[focal_index, focal_index])
        local_scale = float(
            electrical.loc[focal_index, "g_leak"]
            * (1.0 + background_multiplier)
            + electrical.loc[focal_index, "g_e"]
            + electrical.loc[focal_index, "g_i"]
        )
        descendant_indices = relations[focal]["descendant"]
        comparison_indices = relations[focal]["depth-matched unrelated"]
        cable_at_e = np.abs(c_k[e_indices])
        selectivity = float(
            np.median(cable_at_e[descendant_indices])
            / max(float(np.median(cable_at_e[comparison_indices])), 1e-30)
        )
        for scheme, values in dose_schemes.items():
            for dose_value in values:
                eta = scheme_eta(
                    scheme,
                    float(dose_value),
                    local_scale,
                    input_resistance,
                )
                for perturbation in ("matched additive", "focal shunt"):
                    voltage, adjoint, clamp_residual = intervention_state(
                        inverse,
                        rhs,
                        baseline_voltage,
                        soma_index,
                        focal_index,
                        eta,
                        float(cfg["inhibitory_reversal"]),
                        perturbation,
                    )
                    gradient = adjoint * (
                        float(cfg["excitatory_reversal"]) - voltage
                    )
                    changed = gradient[e_indices]
                    descendant = relation_metrics(
                        base_gradient, changed, descendant_indices
                    )
                    comparison = relation_metrics(
                        base_gradient, changed, comparison_indices
                    )
                    row = {
                        "root_id": root_id,
                        "focal_segment_id": int(focal),
                        "focal_topological_depth": int(
                            electrical.loc[focal_index, "topological_depth"]
                        ),
                        "membrane_resistance_ohm_cm2": float(rm),
                        "background_leak_multiplier": float(background_multiplier),
                        "dose_scheme": scheme,
                        "dose_value": float(dose_value),
                        "delta_conductance_ns": eta,
                        "perturbation": perturbation,
                        "local_input_resistance_gohm": input_resistance,
                        "local_input_conductance_ns": 1.0 / input_resistance,
                        "transport_selectivity": selectivity,
                        "baseline_focal_adjoint": float(baseline_adjoint[focal_index]),
                        "matrix_minimum_eigenvalue": minimum_eigenvalue,
                        "soma_state_residual": clamp_residual,
                    }
                    for name, value in descendant.items():
                        row[f"descendant_{name}"] = value
                    for name, value in comparison.items():
                        row[f"matched_{name}"] = value
                    row["localization_index"] = (
                        descendant["median_abs_log_change"]
                        - comparison["median_abs_log_change"]
                    )
                    row["signed_localization"] = (
                        descendant["median_signed_log_change"]
                        - comparison["median_signed_log_change"]
                    )
                    rows.append(row)
    return rows


def bootstrap(values: np.ndarray, seed: int) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(20_000, len(values)), replace=True).mean(axis=1)
    low, high = np.quantile(draws, [0.025, 0.975])
    return float(values.mean()), float(low), float(high)


def summarize(rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics = [
        "localization_index",
        "signed_localization",
        "descendant_attenuated_fraction",
        "descendant_enhanced_fraction",
        "descendant_sign_flip_fraction",
        "descendant_gradient_energy_ratio",
        "transport_selectivity",
        "local_input_resistance_gohm",
    ]
    keys = [
        "membrane_resistance_ohm_cm2",
        "background_leak_multiplier",
        "dose_scheme",
        "dose_value",
        "perturbation",
    ]
    cell = rows.groupby(["root_id", *keys], as_index=False)[metrics].mean()
    summary_rows: list[dict] = []
    for condition_index, (values, part) in enumerate(cell.groupby(keys, sort=True)):
        row = dict(zip(keys, values))
        row["n_cells"] = int(part.root_id.nunique())
        for metric_index, metric in enumerate(metrics):
            mean, low, high = bootstrap(
                part[metric].to_numpy(float), 60_000 + 20 * condition_index + metric_index
            )
            row[f"mean_{metric}"] = mean
            row[f"ci95_low_{metric}"] = low
            row[f"ci95_high_{metric}"] = high
        summary_rows.append(row)
    return cell, pd.DataFrame(summary_rows)


def paired_contrasts(cell: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "membrane_resistance_ohm_cm2",
        "background_leak_multiplier",
        "dose_scheme",
        "dose_value",
    ]
    metrics = [
        "localization_index",
        "signed_localization",
        "descendant_attenuated_fraction",
        "descendant_enhanced_fraction",
        "descendant_sign_flip_fraction",
        "descendant_gradient_energy_ratio",
    ]
    rows: list[dict] = []
    for condition_index, (values, part) in enumerate(cell.groupby(keys, sort=True)):
        wide = part.pivot(index="root_id", columns="perturbation", values=metrics)
        for metric_index, metric in enumerate(metrics):
            differences = (
                wide[(metric, "focal shunt")] - wide[(metric, "matched additive")]
            ).to_numpy(float)
            mean, low, high = bootstrap(
                differences, 70_000 + 20 * condition_index + metric_index
            )
            rows.append(
                {
                    **dict(zip(keys, values)),
                    "metric": metric,
                    "n_cells": len(differences),
                    "mean_shunt_minus_additive": mean,
                    "ci95_low": low,
                    "ci95_high": high,
                    "cells_positive": int(np.sum(differences > 0)),
                }
            )
    return pd.DataFrame(rows)


def plot(summary: pd.DataFrame, rows: pd.DataFrame) -> None:
    apply_neurips_style()
    fig, (ax_a, ax_b, ax_c) = plt.subplots(
        1,
        3,
        figsize=(FIG_W, 2.95),
        gridspec_kw={
            "left": 0.085,
            "right": 0.985,
            "bottom": 0.20,
            "top": 0.84,
            "wspace": 0.55,
        },
    )
    absolute = summary[
        summary.dose_scheme.eq("fixed_absolute_ns")
        & summary.perturbation.eq("focal shunt")
        & np.isclose(summary.background_leak_multiplier, 0.0)
    ]
    regime_colors = {300.0: COLORS["shunting"], 1000.0: COLORS["per_soma"], 15000.0: COLORS["additive"]}
    for rm, part in absolute.groupby("membrane_resistance_ohm_cm2"):
        part = part.sort_values("dose_value")
        ax_a.plot(
            part.dose_value,
            part.mean_localization_index,
            color=regime_colors[float(rm)],
            marker="o",
            lw=LW_DATA,
            label=rf"$R_m={rm:g}$",
        )
    ax_a.set_xscale("log")
    ax_a.axhline(0, color=COLORS["mute"], ls="--", lw=0.8)
    ax_a.set_xlabel("fixed shunt conductance (nS)")
    ax_a.set_ylabel("localization index")
    panel_title(ax_a, "A", "Absolute dose")
    style_axis(ax_a)
    clean_legend(ax_a, fontsize=PT_LEGEND - 0.4, loc="best")

    selected = rows[
        rows.perturbation.eq("focal shunt")
        & rows.dose_scheme.eq("input_conductance_normalized")
        & np.isclose(rows.dose_value, 1.0)
    ]
    cell_sites = selected.groupby(
        ["root_id", "focal_segment_id", "membrane_resistance_ohm_cm2", "background_leak_multiplier"],
        as_index=False,
    )[["transport_selectivity", "localization_index"]].mean()
    ax_b.scatter(
        cell_sites.transport_selectivity,
        cell_sites.localization_index,
        s=9,
        color=COLORS["per_soma"],
        alpha=0.35,
        edgecolors="none",
    )
    ax_b.axhline(0, color=COLORS["mute"], ls="--", lw=0.8)
    ax_b.axvline(1, color=COLORS["mute"], ls=":", lw=0.8)
    ax_b.set_xscale("log")
    ax_b.set_xlabel(r"transport selectivity $S_k$")
    ax_b.set_ylabel("localization index")
    panel_title(ax_b, "B", "Selectivity predicts spread")
    style_axis(ax_b)

    central = summary[
        summary.dose_scheme.eq("input_conductance_normalized")
        & np.isclose(summary.dose_value, 1.0)
        & np.isclose(summary.membrane_resistance_ohm_cm2, 1000.0)
        & np.isclose(summary.background_leak_multiplier, 1.0)
        & summary.perturbation.eq("focal shunt")
    ].iloc[0]
    metrics = ["attenuated", "enhanced", "sign flip"]
    values = [
        central.mean_descendant_attenuated_fraction,
        central.mean_descendant_enhanced_fraction,
        central.mean_descendant_sign_flip_fraction,
    ]
    ax_c.bar(range(3), values, color=[COLORS["shunting"], COLORS["additive"], COLORS["mute"]], width=0.68)
    ax_c.set_xticks(range(3), metrics, rotation=20, ha="right")
    ax_c.set_ylabel("descendant fraction")
    ax_c.set_ylim(0, 1)
    panel_title(ax_c, "C", "Signed gradient outcomes")
    style_axis(ax_c, grid="y")

    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.canvas.draw()
    audit_layout(fig, "fig_focal_selectivity_matrix")
    audit_text_over_data(fig, "fig_focal_selectivity_matrix")
    fig.savefig(
        FIGURES / "fig_focal_selectivity_matrix.pdf",
        metadata={"CreationDate": None, "ModDate": None},
    )
    fig.savefig(FIGURES / "fig_focal_selectivity_matrix.png", dpi=600)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["canary", "confirmatory"], required=True)
    args = parser.parse_args()
    cfg = json.loads(CONFIG.read_text())
    segments = pd.read_csv(SEGMENTS)
    if args.phase == "canary":
        roots = [int(sorted(segments.root_id.unique())[0])]
        rms = [float(cfg["membrane_resistance_ohm_cm2"][1])]
        backgrounds = [float(cfg["background_leak_multipliers"][1])]
    else:
        roots = [int(value) for value in sorted(segments.root_id.unique())]
        rms = [float(value) for value in cfg["membrane_resistance_ohm_cm2"]]
        backgrounds = [float(value) for value in cfg["background_leak_multipliers"]]
    rows: list[dict] = []
    for root_id in roots:
        cell = segments[segments.root_id.eq(root_id)].copy()
        for rm in rms:
            for background in backgrounds:
                rows.extend(
                    run_cell_regime(
                        root_id,
                        cell,
                        cfg,
                        rm,
                        background,
                        canary=args.phase == "canary",
                    )
                )
    frame = pd.DataFrame(rows)
    if frame.empty or not np.isfinite(frame.select_dtypes(include=[np.number])).all().all():
        raise SystemExit("non-finite or empty focal-selectivity output")
    gate = {
        "rows": int(len(frame)),
        "minimum_matrix_eigenvalue": float(frame.matrix_minimum_eigenvalue.min()),
        "maximum_soma_state_residual": float(frame.soma_state_residual.max()),
        "passed": bool(
            frame.matrix_minimum_eigenvalue.min() > 0
            and frame.soma_state_residual.max() < 1e-10
        ),
    }
    if args.phase == "canary":
        payload = {
            "phase": "canary",
            "config_sha256": digest(CONFIG),
            "script_sha256": digest(Path(__file__).resolve()),
            "artifact_and_numerical_gates": gate,
            "outcomes_are_not_confirmatory": True,
        }
        CANARY.write_text(json.dumps(payload, indent=2) + "\n")
        print(json.dumps(payload, indent=2))
        if not gate["passed"]:
            raise SystemExit("canary failed")
        return
    canary = json.loads(CANARY.read_text()) if CANARY.exists() else None
    if not canary or not canary["artifact_and_numerical_gates"]["passed"]:
        raise SystemExit("passing canary required")
    if canary["config_sha256"] != digest(CONFIG) or canary["script_sha256"] != digest(Path(__file__).resolve()):
        raise SystemExit("code or configuration changed after canary")
    SOURCE.mkdir(parents=True, exist_ok=True)
    cell, summary = summarize(frame)
    contrasts = paired_contrasts(cell)
    frame.to_csv(SOURCE / "site_outcomes.csv.gz", index=False, compression="gzip", float_format="%.10g")
    cell.to_csv(SOURCE / "cell_condition_metrics.csv", index=False, float_format="%.10g")
    summary.to_csv(SOURCE / "condition_summary.csv", index=False, float_format="%.10g")
    contrasts.to_csv(SOURCE / "paired_contrasts.csv", index=False, float_format="%.10g")
    metadata = {
        "study": cfg["study"],
        "status": "complete_phase1_confirmatory",
        "scope_boundary": cfg["scope_boundary"],
        "config_sha256": digest(CONFIG),
        "script_sha256": digest(Path(__file__).resolve()),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "n_cells": int(frame.root_id.nunique()),
        "n_focal_sites": int(frame[["root_id", "focal_segment_id"]].drop_duplicates().shape[0]),
        "n_site_condition_rows": int(len(frame)),
        "minimum_matrix_eigenvalue": gate["minimum_matrix_eigenvalue"],
        "maximum_soma_state_residual": gate["maximum_soma_state_residual"],
    }
    (SOURCE / "summary.json").write_text(json.dumps(metadata, indent=2) + "\n")
    plot(summary, frame)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
