#!/usr/bin/env python3
"""Focal shunting perturbations and exact credit on real MICrONS trees.

Each compressed morphology is instantiated as a passive conductance network.
Exact gradients of a somatic squared-error objective with respect to local
excitatory conductances are computed by an adjoint solve.  A focal inhibitory
conductance increase is compared with a current perturbation matched to its
first-order current at the unperturbed local voltage.  Somatic current clamp
restores the baseline somatic voltage so changes in the global error magnitude
cannot explain changes in the spatial credit field.

This is a mechanistic in-silico perturbation on measured anatomy, not an in-vivo
learning experiment.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from analyze_microns_morphology_credit import ancestry_matrix, electrical_geometry, parent_map


PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_SEGMENTS = PROJECT / "results" / "microns_morphology_credit" / "segment_metrics.csv"
DEFAULT_OUTDIR = PROJECT / "results" / "microns_focal_shunting_credit"
PERTURBATIONS = ["matched additive", "focal shunt"]
CATEGORIES = ["descendant", "depth-matched unrelated", "sister", "ancestor", "unrelated"]
COLORS = {"matched additive": "#d99b2b", "focal shunt": "#26828e"}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def stable_rng(seed: int, root_id: int, stream: int) -> np.random.Generator:
    root_id = int(root_id)
    return np.random.default_rng(
        np.random.SeedSequence(
            [
                int(seed) & 0xFFFFFFFF,
                root_id & 0xFFFFFFFF,
                (root_id >> 32) & 0xFFFFFFFF,
                int(stream) & 0xFFFFFFFF,
            ]
        )
    )


def conductance_system(
    segments: pd.DataFrame,
    e_scale: float,
    i_scale: float,
    excitatory_reversal: float,
    inhibitory_reversal: float,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, int, dict[int, int], dict[int, list[int]]]:
    electrical = electrical_geometry(segments, e_scale=e_scale, i_scale=i_scale)
    electrical = electrical.sort_values("topological_depth").reset_index(drop=True)
    root, parents, children = parent_map(electrical)
    ids = electrical["segment_id"].astype(int).tolist()
    index = {segment: position for position, segment in enumerate(ids)}
    n = len(electrical)
    matrix = np.zeros((n, n), dtype=float)
    rhs = (
        electrical["g_e"].to_numpy(dtype=float) * float(excitatory_reversal)
        + electrical["g_i"].to_numpy(dtype=float) * float(inhibitory_reversal)
    )
    matrix[np.diag_indices(n)] += (
        electrical["g_leak"].to_numpy(dtype=float)
        + electrical["g_e"].to_numpy(dtype=float)
        + electrical["g_i"].to_numpy(dtype=float)
    )
    for child, parent in parents.items():
        child_index = index[int(child)]
        parent_index = index[int(parent)]
        coupling = float(electrical.loc[child_index, "g_edge"])
        matrix[child_index, child_index] += coupling
        matrix[parent_index, parent_index] += coupling
        matrix[child_index, parent_index] -= coupling
        matrix[parent_index, child_index] -= coupling
    root_index = index[int(root)]
    return electrical, matrix, rhs, root_index, parents, children


def solve_with_soma_clamp(
    matrix: np.ndarray,
    rhs: np.ndarray,
    root_index: int,
    target_soma_voltage: float,
) -> tuple[np.ndarray, float, float]:
    unit_soma = np.zeros(len(rhs), dtype=float)
    unit_soma[root_index] = 1.0
    solutions = np.linalg.solve(matrix, np.column_stack([rhs, unit_soma]))
    voltage = solutions[:, 0]
    soma_response = solutions[:, 1]
    clamp_current = (float(target_soma_voltage) - voltage[root_index]) / soma_response[root_index]
    voltage = voltage + clamp_current * soma_response
    return voltage, float(clamp_current), float(abs(voltage[root_index] - target_soma_voltage))


def exact_e_gradient(
    matrix: np.ndarray,
    voltage: np.ndarray,
    root_index: int,
    target_output: float,
    excitatory_reversal: float,
) -> np.ndarray:
    root_vector = np.zeros(len(voltage), dtype=float)
    root_vector[root_index] = voltage[root_index] - float(target_output)
    adjoint = np.linalg.solve(matrix.T, root_vector)
    return adjoint * (float(excitatory_reversal) - voltage)


def descendant_set(segment: int, children: dict[int, list[int]]) -> set[int]:
    found: set[int] = set()
    stack = [int(segment)]
    while stack:
        node = int(stack.pop())
        if node in found:
            continue
        found.add(node)
        stack.extend(children.get(node, []))
    return found


def ancestor_set(segment: int, parents: dict[int, int]) -> set[int]:
    found: set[int] = set()
    cursor = int(segment)
    while cursor in parents:
        cursor = int(parents[cursor])
        found.add(cursor)
    return found


def relation_indices(
    focal: int,
    e_segments: list[int],
    path_lengths: dict[int, float],
    parents: dict[int, int],
    children: dict[int, list[int]],
) -> dict[str, np.ndarray]:
    descendants = descendant_set(focal, children)
    ancestors = ancestor_set(focal, parents)
    sisters: set[int] = set()
    if focal in parents:
        parent = parents[focal]
        for sibling in children.get(parent, []):
            if int(sibling) != int(focal):
                sisters.update(descendant_set(int(sibling), children))
    categories = {
        "descendant": np.asarray([i for i, segment in enumerate(e_segments) if segment in descendants], dtype=int),
        "ancestor": np.asarray([i for i, segment in enumerate(e_segments) if segment in ancestors], dtype=int),
        "sister": np.asarray([i for i, segment in enumerate(e_segments) if segment in sisters], dtype=int),
    }
    excluded = descendants | ancestors | sisters
    unrelated = np.asarray([i for i, segment in enumerate(e_segments) if segment not in excluded], dtype=int)
    categories["unrelated"] = unrelated
    if len(categories["descendant"]) and len(unrelated):
        unrelated_depth = np.asarray([path_lengths[e_segments[i]] for i in unrelated], dtype=float)
        matched = []
        for index in categories["descendant"]:
            depth = path_lengths[e_segments[int(index)]]
            matched.append(int(unrelated[np.argmin(np.abs(unrelated_depth - depth))]))
        categories["depth-matched unrelated"] = np.asarray(matched, dtype=int)
    else:
        categories["depth-matched unrelated"] = np.asarray([], dtype=int)
    return categories


def choose_focal_sites(
    electrical: pd.DataFrame,
    e_segments: list[int],
    parents: dict[int, int],
    children: dict[int, list[int]],
    rng: np.random.Generator,
    maximum: int,
    minimum_sites: int,
) -> tuple[list[int], dict[int, dict[str, np.ndarray]]]:
    path_lengths = electrical.set_index("segment_id")["path_length_um"].astype(float).to_dict()
    eligible: list[int] = []
    relations: dict[int, dict[str, np.ndarray]] = {}
    for focal in electrical.loc[electrical["g_i"] > 0, "segment_id"].astype(int):
        current = relation_indices(focal, e_segments, path_lengths, parents, children)
        if (
            len(current["descendant"]) >= int(minimum_sites)
            and len(current["depth-matched unrelated"]) >= int(minimum_sites)
        ):
            eligible.append(int(focal))
            relations[int(focal)] = current
    if len(eligible) <= int(maximum):
        return eligible, relations
    lookup = electrical.set_index("segment_id")
    depth = lookup.loc[eligible, "topological_depth"].to_numpy(dtype=float)
    edges = np.unique(np.quantile(depth, [0.0, 0.25, 0.5, 0.75, 1.0]))
    bins = np.digitize(depth, edges[1:-1], right=True)
    selected: list[int] = []
    per_bin = max(1, int(np.ceil(maximum / max(len(np.unique(bins)), 1))))
    for depth_bin in np.unique(bins):
        candidates = np.asarray(eligible, dtype=int)[bins == depth_bin]
        selected.extend(rng.permutation(candidates)[:per_bin].astype(int).tolist())
    if len(selected) > int(maximum):
        selected = rng.permutation(np.asarray(selected, dtype=int))[: int(maximum)].astype(int).tolist()
    return sorted(selected), relations


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    cumulative = np.cumsum(weights) / max(float(np.sum(weights)), 1e-12)
    return float(values[np.searchsorted(cumulative, 0.5, side="left")])


def category_record(
    root_id: int,
    focal: int,
    dose: float,
    perturbation: str,
    category: str,
    indices: np.ndarray,
    change: np.ndarray,
    signed_change: np.ndarray,
    sign_flip: np.ndarray,
    e_weights: np.ndarray,
) -> dict[str, Any] | None:
    if not len(indices):
        return None
    values = change[indices]
    weights = e_weights[indices]
    return {
        "root_id": int(root_id),
        "focal_segment_id": int(focal),
        "dose": float(dose),
        "perturbation": perturbation,
        "category": category,
        "n_sites": int(len(indices)),
        "median_abs_log_gradient_change": float(np.median(values)),
        "mean_abs_log_gradient_change": float(np.mean(values)),
        "synapse_weighted_median_abs_log_gradient_change": weighted_median(values, weights),
        "median_signed_log_gradient_change": float(np.median(signed_change[indices])),
        "gradient_sign_flip_fraction": float(np.mean(sign_flip[indices])),
    }


def analyze_cell(
    root_id: int,
    segments: pd.DataFrame,
    doses: list[float],
    seed: int,
    max_focal_sites: int,
    minimum_sites: int,
    e_scale: float,
    i_scale: float,
    excitatory_reversal: float,
    inhibitory_reversal: float,
    conductance_builder: Callable[..., tuple[pd.DataFrame, np.ndarray, np.ndarray, int, dict[int, int], dict[int, list[int]]]] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    builder = conductance_system if conductance_builder is None else conductance_builder
    electrical, baseline_matrix, baseline_rhs, root_index, parents, children = builder(
        segments, e_scale, i_scale, excitatory_reversal, inhibitory_reversal
    )
    ids = electrical["segment_id"].astype(int).tolist()
    index = {segment: position for position, segment in enumerate(ids)}
    baseline_voltage = np.linalg.solve(baseline_matrix, baseline_rhs)
    target_output = float(baseline_voltage[root_index] - 1.0)
    baseline_gradient = exact_e_gradient(
        baseline_matrix, baseline_voltage, root_index, target_output, excitatory_reversal
    )
    e_segments = electrical.loc[electrical["E_size"] > 0, "segment_id"].astype(int).tolist()
    e_indices = np.asarray([index[segment] for segment in e_segments], dtype=int)
    e_weights = electrical.loc[e_indices, "E_size"].to_numpy(dtype=float)
    focal_sites, relations = choose_focal_sites(
        electrical,
        e_segments,
        parents,
        children,
        stable_rng(seed, root_id, stream=1),
        max_focal_sites,
        minimum_sites,
    )
    focal_depth = electrical.set_index("segment_id").loc[focal_sites, "topological_depth"].to_numpy(dtype=float)
    if len(focal_sites) > 1:
        depth_edges = np.unique(np.quantile(focal_depth, [0.0, 0.25, 0.5, 0.75, 1.0]))
        focal_bins = np.digitize(focal_depth, depth_edges[1:-1], right=True)
    else:
        focal_bins = np.zeros(len(focal_sites), dtype=int)
    focal_bin_lookup = {focal: int(focal_bins[i]) for i, focal in enumerate(focal_sites)}
    shuffle_rng = stable_rng(seed, root_id, stream=3)
    shuffled_relation_templates: dict[int, list[int]] = {}
    for focal in focal_sites:
        candidates = [
            other
            for other in focal_sites
            if other != focal and focal_bin_lookup[other] == focal_bin_lookup[focal]
        ]
        if not candidates:
            candidates = [other for other in focal_sites if other != focal]
        shuffled_relation_templates[focal] = (
            shuffle_rng.choice(candidates, size=200, replace=True).astype(int).tolist()
            if candidates
            else []
        )
    category_rows: list[dict[str, Any]] = []
    focal_rows: list[dict[str, Any]] = []
    max_clamp_error = 0.0
    for focal in focal_sites:
        focal_index = index[focal]
        local_scale = float(
            electrical.loc[focal_index, "g_leak"]
            + electrical.loc[focal_index, "g_e"]
            + electrical.loc[focal_index, "g_i"]
        )
        for dose in doses:
            delta_conductance = float(dose) * local_scale
            for perturbation in PERTURBATIONS:
                matrix = baseline_matrix.copy()
                rhs = baseline_rhs.copy()
                if perturbation == "focal shunt":
                    matrix[focal_index, focal_index] += delta_conductance
                    rhs[focal_index] += delta_conductance * float(inhibitory_reversal)
                else:
                    rhs[focal_index] += delta_conductance * (
                        float(inhibitory_reversal) - baseline_voltage[focal_index]
                    )
                voltage, clamp_current, clamp_error = solve_with_soma_clamp(
                    matrix, rhs, root_index, baseline_voltage[root_index]
                )
                max_clamp_error = max(max_clamp_error, clamp_error)
                gradient = exact_e_gradient(
                    matrix, voltage, root_index, target_output, excitatory_reversal
                )
                base = baseline_gradient[e_indices]
                changed = gradient[e_indices]
                epsilon = 1e-15 * max(float(np.max(np.abs(base))), 1.0)
                signed_log_change = np.log((np.abs(changed) + epsilon) / (np.abs(base) + epsilon))
                absolute_log_change = np.abs(signed_log_change)
                sign_flip = np.signbit(changed) != np.signbit(base)
                records: dict[str, dict[str, Any]] = {}
                for category in CATEGORIES:
                    record = category_record(
                        root_id,
                        focal,
                        dose,
                        perturbation,
                        category,
                        relations[focal][category],
                        absolute_log_change,
                        signed_log_change,
                        sign_flip,
                        e_weights,
                    )
                    if record is not None:
                        category_rows.append(record)
                        records[category] = record
                if "descendant" in records and "depth-matched unrelated" in records:
                    shuffled_localization = []
                    for template in shuffled_relation_templates[focal]:
                        template_descendant = relations[int(template)]["descendant"]
                        template_unrelated = relations[int(template)]["depth-matched unrelated"]
                        shuffled_localization.append(
                            float(
                                np.median(absolute_log_change[template_descendant])
                                - np.median(absolute_log_change[template_unrelated])
                            )
                        )
                    focal_rows.append(
                        {
                            "root_id": int(root_id),
                            "focal_segment_id": int(focal),
                            "focal_topological_depth": int(electrical.loc[focal_index, "topological_depth"]),
                            "focal_path_length_um": float(electrical.loc[focal_index, "path_length_um"]),
                            "dose": float(dose),
                            "perturbation": perturbation,
                            "delta_conductance": float(delta_conductance),
                            "soma_clamp_current": float(clamp_current),
                            "localization_index": float(
                                records["descendant"]["median_abs_log_gradient_change"]
                                - records["depth-matched unrelated"]["median_abs_log_gradient_change"]
                            ),
                            "descendant_change": float(records["descendant"]["median_abs_log_gradient_change"]),
                            "matched_unrelated_change": float(
                                records["depth-matched unrelated"]["median_abs_log_gradient_change"]
                            ),
                            "depth_shuffled_localization_mean": float(np.mean(shuffled_localization))
                            if shuffled_localization
                            else float("nan"),
                            "depth_shuffled_localization_sd": float(np.std(shuffled_localization, ddof=1))
                            if len(shuffled_localization) > 1
                            else float("nan"),
                        }
                    )
    finite_difference_checks = []
    for e_index in stable_rng(seed, root_id, stream=2).choice(
        e_indices, size=min(3, len(e_indices)), replace=False
    ):
        epsilon = 1e-6 * max(float(electrical.loc[e_index, "g_e"]), 1.0)
        changed_matrix = baseline_matrix.copy()
        changed_rhs = baseline_rhs.copy()
        changed_matrix[e_index, e_index] += epsilon
        changed_rhs[e_index] += epsilon * float(excitatory_reversal)
        changed_voltage = np.linalg.solve(changed_matrix, changed_rhs)
        base_loss = 0.5 * (baseline_voltage[root_index] - target_output) ** 2
        changed_loss = 0.5 * (changed_voltage[root_index] - target_output) ** 2
        finite = (changed_loss - base_loss) / epsilon
        analytic = baseline_gradient[e_index]
        finite_difference_checks.append(
            {
                "segment_id": int(ids[e_index]),
                "analytic": float(analytic),
                "finite_difference": float(finite),
                "relative_error": float(abs(finite - analytic) / max(abs(analytic), 1e-12)),
            }
        )
    validation = {
        "root_id": int(root_id),
        "n_segments": int(len(electrical)),
        "n_e_segments": int(len(e_segments)),
        "n_eligible_focal_sites": int(len(relations)),
        "n_selected_focal_sites": int(len(focal_sites)),
        "selected_focal_segments": [int(value) for value in focal_sites],
        "matrix_symmetry_error": float(np.max(np.abs(baseline_matrix - baseline_matrix.T))),
        "matrix_minimum_eigenvalue": float(np.linalg.eigvalsh(baseline_matrix)[0]),
        "maximum_soma_clamp_error": float(max_clamp_error),
        "finite_difference_checks": finite_difference_checks,
    }
    return category_rows, focal_rows, validation


def bootstrap_ci(values: np.ndarray, rng: np.random.Generator) -> list[float]:
    draws = rng.choice(values, size=(20_000, len(values)), replace=True).mean(axis=1)
    return [float(x) for x in np.quantile(draws, [0.025, 0.975])]


def summarize_primary(focal: pd.DataFrame, primary_dose: float, seed: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    selected = focal[np.isclose(focal["dose"], float(primary_dose))]
    wide = selected.pivot_table(
        index=["root_id", "focal_segment_id"], columns="perturbation", values="localization_index"
    ).dropna()
    shuffled = selected.pivot_table(
        index=["root_id", "focal_segment_id"],
        columns="perturbation",
        values="depth_shuffled_localization_mean",
    ).add_suffix(" depth-shuffled")
    wide = wide.join(shuffled).dropna()
    wide["shunt_minus_additive"] = wide["focal shunt"] - wide["matched additive"]
    wide["shunt_topology_minus_depth_shuffle"] = (
        wide["focal shunt"] - wide["focal shunt depth-shuffled"]
    )
    cell = wide.groupby(level="root_id").agg(
        focal_shunt_localization=("focal shunt", "mean"),
        matched_additive_localization=("matched additive", "mean"),
        shunt_minus_additive=("shunt_minus_additive", "mean"),
        shunt_depth_shuffled_localization=("focal shunt depth-shuffled", "mean"),
        shunt_topology_minus_depth_shuffle=("shunt_topology_minus_depth_shuffle", "mean"),
        n_focal_sites=("shunt_minus_additive", "size"),
    ).reset_index()
    values = cell["shunt_minus_additive"].to_numpy(dtype=float)
    test = stats.wilcoxon(values, alternative="two-sided") if len(values) >= 2 and not np.allclose(values, 0) else None
    topology_values = cell["shunt_topology_minus_depth_shuffle"].to_numpy(dtype=float)
    topology_test = (
        stats.wilcoxon(topology_values, alternative="two-sided")
        if len(topology_values) >= 2 and not np.allclose(topology_values, 0)
        else None
    )
    summary = {
        "dose": float(primary_dose),
        "n_cells": int(len(cell)),
        "n_focal_sites": int(len(wide)),
        "mean_shunt_localization_index": float(cell["focal_shunt_localization"].mean()),
        "mean_matched_additive_localization_index": float(cell["matched_additive_localization"].mean()),
        "mean_shunt_minus_additive": float(np.mean(values)),
        "cell_bootstrap_ci95": bootstrap_ci(values, np.random.default_rng(seed)),
        "cells_positive": int(np.sum(values > 0)),
        "wilcoxon_two_sided_p": float(test.pvalue) if test is not None else float("nan"),
        "depth_shuffled_relation_control": {
            "mean_shunt_depth_shuffled_localization": float(
                cell["shunt_depth_shuffled_localization"].mean()
            ),
            "mean_true_minus_depth_shuffled": float(np.mean(topology_values)),
            "cell_bootstrap_ci95": bootstrap_ci(topology_values, np.random.default_rng(seed + 1)),
            "cells_positive": int(np.sum(topology_values > 0)),
            "wilcoxon_two_sided_p": float(topology_test.pvalue)
            if topology_test is not None
            else float("nan"),
            "procedure": "200 relation templates sampled from other focal sites in the same topological-depth quartile",
        },
    }
    return cell, summary


def make_figure(category: pd.DataFrame, focal: pd.DataFrame, cell: pd.DataFrame, primary_dose: float, outdir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.8))
    ax = axes[0]
    positions = {"matched additive": 0, "focal shunt": 1}
    for _, row in cell.iterrows():
        values = [row["matched_additive_localization"], row["focal_shunt_localization"]]
        ax.plot([0, 1], values, color="#aaaaaa", lw=0.8, alpha=0.75)
        ax.scatter([0, 1], values, color=[COLORS["matched additive"], COLORS["focal shunt"]], s=25, zorder=3)
    ax.axhline(0, color="black", lw=0.8, ls="--")
    ax.set_xticks([0, 1], ["matched\nadditive", "focal\nshunt"])
    ax.set_ylabel("descendant − matched-unrelated\nabsolute log-gradient change")
    ax.set_title(f"A  Localization at dose {primary_dose:g}")

    ax = axes[1]
    selected = category[category["category"].isin(["descendant", "depth-matched unrelated"])]
    cell_category = selected.groupby(["root_id", "dose", "perturbation", "category"], as_index=False)[
        "median_abs_log_gradient_change"
    ].mean()
    for perturbation in PERTURBATIONS:
        for category_name, linestyle in [("descendant", "-"), ("depth-matched unrelated", "--")]:
            current = cell_category[
                (cell_category["perturbation"] == perturbation) & (cell_category["category"] == category_name)
            ]
            means = current.groupby("dose")["median_abs_log_gradient_change"].mean()
            ax.plot(
                means.index,
                means.values,
                marker="o",
                color=COLORS[perturbation],
                ls=linestyle,
                label=f"{perturbation}, {'desc.' if category_name == 'descendant' else 'matched'}",
            )
    ax.set_xscale("log", base=2)
    ax.set_xlabel("perturbation dose (local conductance units)")
    ax.set_ylabel("median absolute log-gradient change")
    ax.set_title("B  Dose and tree relation")
    ax.legend(frameon=False, fontsize=7)

    ax = axes[2]
    for _, row in cell.iterrows():
        values = [row["shunt_depth_shuffled_localization"], row["focal_shunt_localization"]]
        ax.plot([0, 1], values, color="#aaaaaa", lw=0.8, alpha=0.75)
        ax.scatter([0, 1], values, color=["#cc79a7", COLORS["focal shunt"]], s=25, zorder=3)
    ax.axhline(0, color="black", lw=0.8, ls="--")
    ax.set_xticks([0, 1], ["depth-shuffled\nrelation", "true focal\nrelation"])
    ax.set_ylabel("shunt localization index")
    ax.set_title("C  Topology control")
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Focal shunting perturbations of exact credit on real dendritic topology", weight="bold", fontsize=11)
    fig.tight_layout()
    fig.savefig(outdir / "microns_focal_shunting_credit.png", dpi=280, bbox_inches="tight")
    fig.savefig(outdir / "microns_focal_shunting_credit.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--segments", type=Path, default=DEFAULT_SEGMENTS)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--doses", type=float, nargs="+", default=[0.25, 0.5, 1.0, 2.0])
    parser.add_argument("--primary-dose", type=float, default=1.0)
    parser.add_argument("--max-focal-sites", type=int, default=16)
    parser.add_argument("--minimum-sites", type=int, default=3)
    parser.add_argument("--e-scale", type=float, default=0.35)
    parser.add_argument("--i-scale", type=float, default=0.35)
    parser.add_argument("--excitatory-reversal", type=float, default=1.0)
    parser.add_argument("--inhibitory-reversal", type=float, default=-0.2)
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    segments = pd.read_csv(args.segments)
    category_rows: list[dict[str, Any]] = []
    focal_rows: list[dict[str, Any]] = []
    validation: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for root_id, cell_segments in segments.groupby("root_id"):
        try:
            rows, focal, checks = analyze_cell(
                int(root_id),
                cell_segments.copy(),
                [float(value) for value in args.doses],
                args.seed,
                args.max_focal_sites,
                args.minimum_sites,
                args.e_scale,
                args.i_scale,
                args.excitatory_reversal,
                args.inhibitory_reversal,
            )
            category_rows.extend(rows)
            focal_rows.extend(focal)
            validation.append(checks)
        except Exception as exc:
            errors.append({"root_id": str(root_id), "error": f"{type(exc).__name__}: {exc}"})
    category = pd.DataFrame(category_rows)
    focal = pd.DataFrame(focal_rows)
    if category.empty or focal.empty:
        raise RuntimeError(f"no perturbations completed: {errors}")
    category.to_csv(args.outdir / "category_effects.csv", index=False)
    focal.to_csv(args.outdir / "focal_localization.csv", index=False)
    cell, primary = summarize_primary(focal, args.primary_dose, args.seed + 1)
    cell.to_csv(args.outdir / "cell_primary_contrasts.csv", index=False)
    maximum_fd_error = max(
        check["relative_error"]
        for item in validation
        for check in item["finite_difference_checks"]
    )
    summary = {
        "analysis": "focal shunting perturbation of exact credit on real MICrONS morphologies",
        "status": "completed" if not errors else "completed_with_exclusions",
        "interpretation_level": "mechanistic passive-network test on measured anatomy",
        "n_cells": int(category["root_id"].nunique()),
        "n_focal_sites": int(focal[["root_id", "focal_segment_id"]].drop_duplicates().shape[0]),
        "doses": [float(value) for value in args.doses],
        "primary_contrast": primary,
        "validation": {
            "maximum_adjoint_finite_difference_relative_error": float(maximum_fd_error),
            "maximum_soma_clamp_error": float(max(item["maximum_soma_clamp_error"] for item in validation)),
            "minimum_matrix_eigenvalue": float(min(item["matrix_minimum_eigenvalue"] for item in validation)),
            "maximum_matrix_symmetry_error": float(max(item["matrix_symmetry_error"] for item in validation)),
            "cell_checks": validation,
        },
        "errors": errors,
        "parameters": {
            "seed": int(args.seed),
            "max_focal_sites_per_cell": int(args.max_focal_sites),
            "minimum_descendant_and_comparison_sites": int(args.minimum_sites),
            "e_scale": float(args.e_scale),
            "i_scale": float(args.i_scale),
            "excitatory_reversal": float(args.excitatory_reversal),
            "inhibitory_reversal": float(args.inhibitory_reversal),
        },
    }
    write_json(args.outdir / "summary.json", summary)
    lines = [
        "# Focal shunting and exact credit",
        "",
        f"Completed {summary['n_focal_sites']} focal perturbations across {summary['n_cells']} real reconstructed cells.",
        "Somatic current clamp held the output voltage—and therefore the scalar output error—fixed.",
        "",
        "## Frozen primary contrast",
        "",
        f"At dose {primary['dose']:g}, mean shunt localization was {primary['mean_shunt_localization_index']:.4f} "
        f"and matched-additive localization was {primary['mean_matched_additive_localization_index']:.4f}.",
        f"The cell-level shunt-minus-additive contrast was {primary['mean_shunt_minus_additive']:.4f}, "
        f"95% bootstrap CI [{primary['cell_bootstrap_ci95'][0]:.4f}, {primary['cell_bootstrap_ci95'][1]:.4f}], "
        f"with {primary['cells_positive']}/{primary['n_cells']} cells positive and Wilcoxon p={primary['wilcoxon_two_sided_p']:.4g}.",
        f"True shunt localization exceeded the depth-shuffled relation control by "
        f"{primary['depth_shuffled_relation_control']['mean_true_minus_depth_shuffled']:.4f}, "
        f"95% CI [{primary['depth_shuffled_relation_control']['cell_bootstrap_ci95'][0]:.4f}, "
        f"{primary['depth_shuffled_relation_control']['cell_bootstrap_ci95'][1]:.4f}], "
        f"Wilcoxon p={primary['depth_shuffled_relation_control']['wilcoxon_two_sided_p']:.4g}.",
        "",
        "## Numerical validation",
        "",
        f"Maximum adjoint/finite-difference relative error: {maximum_fd_error:.3e}.",
        f"Maximum residual state-matched somatic-voltage error: {summary['validation']['maximum_soma_clamp_error']:.3e}.",
        "",
        "## Boundary",
        "",
        "This demonstrates a mechanistic signature in a calibrated passive model on real anatomy. It does not show that inhibition carried teaching signals in vivo.",
    ]
    (args.outdir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    make_figure(category, focal, cell, args.primary_dose, args.outdir)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
