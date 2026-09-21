#!/usr/bin/env python3
"""Test ancestry dictionaries against independently generated cable responses.

The original route-capacity experiment generated fields from the direct
ancestry kernel and reconstructed them with columns of that same kernel.  This
analysis instead generates the target covariance from exact finite-difference
responses of a reciprocal passive cable system to focal shunts.  The tested
feedback dictionaries remain sparse ancestry routes.  A degree- and
depth-matched surrogate-tree null preserves every node, edge attribute,
topological depth, and parent out-degree while reassigning child subtrees among
parents at the preceding depth.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


PROJECT = Path(__file__).resolve().parents[1]
RECONSTRUCTED = PROJECT / "code" / "reconstructed_tree"
if str(RECONSTRUCTED) not in sys.path:
    sys.path.insert(0, str(RECONSTRUCTED))

from analyze_microns_morphology_credit import ancestry_matrix, parent_map  # noqa: E402
from run_focal_shunting_credit_perturbation import (  # noqa: E402
    conductance_system,
    exact_e_gradient,
    solve_with_soma_clamp,
)


DEFAULT_SEGMENTS = PROJECT / "source_data" / "figure3" / "segment_metrics.csv"
DEFAULT_OUTDIR = PROJECT / "source_data" / "reciprocal_routing"
CHANNELS = [1, 2, 4, 8, 16]
METHODS = [
    "dense SVD oracle",
    "morphology paths",
    "random real paths",
    "depth bins",
    "row-shuffled paths",
    "degree-depth surrogate tree",
]


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


def degree_depth_matched_surrogate(
    segments: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Reassign children to same-depth parent slots without changing degrees."""

    result = segments.copy()
    original_parent = result.set_index("segment_id")["parent_segment_id"].astype(int).to_dict()
    depth = result.set_index("segment_id")["topological_depth"].astype(int).to_dict()
    replacement: dict[int, int] = {}
    for current_depth in sorted(set(depth.values())):
        if current_depth == 0:
            continue
        children = np.asarray(
            sorted(segment for segment, value in depth.items() if value == current_depth),
            dtype=int,
        )
        slots = np.asarray([original_parent[int(child)] for child in children], dtype=int)
        if len(children) > 1:
            children = rng.permutation(children)
        for child, parent in zip(children, slots, strict=True):
            replacement[int(child)] = int(parent)
    result["parent_segment_id"] = result["segment_id"].map(
        lambda segment: replacement.get(int(segment), -1)
    )

    _, surrogate_parent, surrogate_children = parent_map(result)
    if len(surrogate_parent) != len(result) - 1:
        raise AssertionError("surrogate is not a connected rooted tree")
    original_outdegree = pd.Series(
        [parent for parent in original_parent.values() if int(parent) >= 0]
    ).value_counts().sort_index()
    surrogate_outdegree = pd.Series(surrogate_parent).value_counts().sort_index()
    if not original_outdegree.equals(surrogate_outdegree):
        raise AssertionError("surrogate changed parent out-degrees")
    for child, parent in surrogate_parent.items():
        if depth[int(child)] != depth[int(parent)] + 1:
            raise AssertionError("surrogate changed topological depth")
    result["n_children"] = result["segment_id"].map(
        lambda segment: len(surrogate_children.get(int(segment), []))
    )
    return result


def exact_reciprocal_response(
    segments: pd.DataFrame,
    e_scale: float,
    i_scale: float,
    derivative_dose: float,
    excitatory_reversal: float,
    inhibitory_reversal: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[int], list[int], dict[int, int]]:
    """Return the exact local log-gradient response to every focal shunt."""

    electrical, matrix, rhs, root_index, parents, _ = conductance_system(
        segments,
        e_scale,
        i_scale,
        excitatory_reversal,
        inhibitory_reversal,
    )
    ids = electrical["segment_id"].astype(int).tolist()
    index = {segment: position for position, segment in enumerate(ids)}
    voltage = np.linalg.solve(matrix, rhs)
    target_output = float(voltage[root_index] - 1.0)
    gradient = exact_e_gradient(matrix, voltage, root_index, target_output, excitatory_reversal)
    e_segments = electrical.loc[electrical["E_size"] > 0, "segment_id"].astype(int).tolist()
    i_segments = electrical.loc[electrical["I_size"] > 0, "segment_id"].astype(int).tolist()
    e_indices = np.asarray([index[segment] for segment in e_segments], dtype=int)
    base = gradient[e_indices]
    epsilon = 1e-15 * max(float(np.max(np.abs(base))), 1.0)
    response = np.zeros((len(e_segments), len(i_segments)), dtype=float)
    for column, focal in enumerate(i_segments):
        focal_index = index[int(focal)]
        local_scale = float(
            electrical.loc[focal_index, "g_leak"]
            + electrical.loc[focal_index, "g_e"]
            + electrical.loc[focal_index, "g_i"]
        )
        delta = float(derivative_dose) * local_scale
        changed_matrix = matrix.copy()
        changed_rhs = rhs.copy()
        changed_matrix[focal_index, focal_index] += delta
        changed_rhs[focal_index] += delta * float(inhibitory_reversal)
        changed_voltage, _, _ = solve_with_soma_clamp(
            changed_matrix,
            changed_rhs,
            root_index,
            voltage[root_index],
        )
        changed_gradient = exact_e_gradient(
            changed_matrix,
            changed_voltage,
            root_index,
            target_output,
            excitatory_reversal,
        )[e_indices]
        response[:, column] = (
            np.log((np.abs(changed_gradient) + epsilon) / (np.abs(base) + epsilon))
            / float(derivative_dose)
        )
    indexed = electrical.set_index("segment_id")
    e_weights = indexed.loc[e_segments, "E_size"].to_numpy(dtype=float)
    e_depth = indexed.loc[e_segments, "path_length_um"].to_numpy(dtype=float)
    beta = (
        indexed.loc[i_segments, "g_i"].to_numpy(dtype=float)
        / indexed.loc[i_segments, "g_total"].to_numpy(dtype=float)
    )
    return response, e_weights, e_depth, beta, e_segments, i_segments, parents


def weighted_operator(response: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    sqrt_weight = np.sqrt(np.clip(np.asarray(weights, dtype=float), 0.0, None))
    scale = float(np.sqrt(np.mean(sqrt_weight**2)))
    if scale > 0:
        sqrt_weight /= scale
    return response * sqrt_weight[:, None], sqrt_weight


def capture(dictionary: np.ndarray, weighted_response: np.ndarray, sqrt_weight: np.ndarray) -> float:
    weighted_dictionary = np.asarray(dictionary, dtype=float) * sqrt_weight[:, None]
    if weighted_dictionary.size == 0 or not np.any(weighted_dictionary):
        return 0.0
    basis, singular, _ = np.linalg.svd(weighted_dictionary, full_matrices=False)
    tolerance = max(weighted_dictionary.shape) * np.finfo(float).eps * float(singular[0])
    rank = int(np.sum(singular > tolerance))
    if rank == 0:
        return 0.0
    projected = basis[:, :rank].T @ weighted_response
    denominator = float(np.sum(weighted_response**2))
    return float(np.clip(np.sum(projected**2) / max(denominator, 1e-30), 0.0, 1.0))


def depth_dictionary(depth: np.ndarray, count: int) -> np.ndarray:
    if count <= 1:
        return np.ones((len(depth), 1), dtype=float)
    ranks = pd.Series(depth).rank(method="first", pct=True).to_numpy(dtype=float)
    bins = np.minimum((ranks * int(count)).astype(int), int(count) - 1)
    return np.eye(int(count), dtype=float)[bins]


def direct_dictionary(
    e_segments: list[int],
    i_segments: list[int],
    parents: dict[int, int],
    beta: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    ancestry = ancestry_matrix(e_segments, i_segments, parents)
    return ancestry * beta[None, :], ancestry


def summarize_cell(
    root_id: int,
    segments: pd.DataFrame,
    seed: int,
    derivative_dose: float,
    n_controls: int,
    e_scale: float,
    i_scale: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    response, e_weight, e_depth, beta, e_segments, i_segments, parents = exact_reciprocal_response(
        segments,
        e_scale,
        i_scale,
        derivative_dose,
        1.0,
        -0.2,
    )
    weighted_response, sqrt_weight = weighted_operator(response, e_weight)
    direct, ancestry = direct_dictionary(e_segments, i_segments, parents, beta)
    domain = ancestry.T @ e_weight / max(float(e_weight.sum()), 1e-30)
    leverage = beta * domain
    order = np.argsort(-leverage)
    singular = np.linalg.svd(weighted_response, compute_uv=False)
    singular_energy = singular**2
    total_energy = float(singular_energy.sum())
    rows: list[dict[str, Any]] = []
    random_rng = stable_rng(seed, root_id, 1)
    shuffle_rng = stable_rng(seed, root_id, 2)
    surrogate_rng = stable_rng(seed, root_id, 3)
    maximum_count = min(len(e_segments), len(i_segments))

    for count in [value for value in CHANNELS if value <= maximum_count]:
        oracle_capture = float(np.sum(singular_energy[:count]) / max(total_energy, 1e-30))
        rows.append({"root_id": root_id, "channels": count, "method": "dense SVD oracle", "capture": oracle_capture})
        selected = order[:count]
        selected_dictionary = direct[:, selected]
        rows.append(
            {
                "root_id": root_id,
                "channels": count,
                "method": "morphology paths",
                "capture": capture(selected_dictionary, weighted_response, sqrt_weight),
                "wiring_nonzeros": int(np.count_nonzero(selected_dictionary)),
            }
        )
        random_values: list[float] = []
        shuffled_values: list[float] = []
        surrogate_values: list[float] = []
        for _ in range(int(n_controls)):
            choice = random_rng.choice(len(i_segments), size=count, replace=False)
            random_values.append(capture(direct[:, choice], weighted_response, sqrt_weight))
            shuffled = selected_dictionary.copy()
            for column in range(shuffled.shape[1]):
                shuffled[:, column] = shuffle_rng.permutation(shuffled[:, column])
            shuffled_values.append(capture(shuffled, weighted_response, sqrt_weight))

            surrogate = degree_depth_matched_surrogate(segments, surrogate_rng)
            _, surrogate_parents, _ = parent_map(surrogate)
            surrogate_dictionary, surrogate_ancestry = direct_dictionary(
                e_segments,
                i_segments,
                surrogate_parents,
                beta,
            )
            surrogate_domain = surrogate_ancestry.T @ e_weight / max(float(e_weight.sum()), 1e-30)
            surrogate_order = np.argsort(-(beta * surrogate_domain))[:count]
            surrogate_values.append(
                capture(surrogate_dictionary[:, surrogate_order], weighted_response, sqrt_weight)
            )
        for method, values in [
            ("random real paths", random_values),
            ("row-shuffled paths", shuffled_values),
            ("degree-depth surrogate tree", surrogate_values),
        ]:
            rows.append(
                {
                    "root_id": root_id,
                    "channels": count,
                    "method": method,
                    "capture": float(np.mean(values)),
                    "control_sd": float(np.std(values, ddof=1)),
                }
            )
        rows.append(
            {
                "root_id": root_id,
                "channels": count,
                "method": "depth bins",
                "capture": capture(depth_dictionary(e_depth, count), weighted_response, sqrt_weight),
            }
        )

    ancestry_mask = ancestry.astype(bool)
    response_energy = response**2
    audit = {
        "root_id": int(root_id),
        "n_e_segments": int(len(e_segments)),
        "n_i_segments": int(len(i_segments)),
        "reciprocal_operator_rank": int(np.linalg.matrix_rank(weighted_response)),
        "off_ancestry_response_energy_fraction": float(
            response_energy[~ancestry_mask].sum() / max(float(response_energy.sum()), 1e-30)
        ),
        "derivative_dose": float(derivative_dose),
    }
    return rows, audit


def bootstrap_interval(values: np.ndarray, rng: np.random.Generator) -> list[float]:
    draws = rng.choice(values, size=(20_000, len(values)), replace=True).mean(axis=1)
    return [float(x) for x in np.quantile(draws, [0.025, 0.975])]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--segments", type=Path, default=DEFAULT_SEGMENTS)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--seed", type=int, default=20260804)
    parser.add_argument("--derivative-dose", type=float, default=1e-3)
    parser.add_argument("--n-controls", type=int, default=200)
    parser.add_argument("--e-scale", type=float, default=0.35)
    parser.add_argument("--i-scale", type=float, default=0.35)
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    segments = pd.read_csv(args.segments)
    rows: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    for root_id, group in segments.groupby("root_id", sort=True):
        print(f"reciprocal routing: {int(root_id)}", flush=True)
        cell_rows, audit = summarize_cell(
            int(root_id),
            group.copy(),
            args.seed,
            args.derivative_dose,
            args.n_controls,
            args.e_scale,
            args.i_scale,
        )
        rows.extend(cell_rows)
        audits.append(audit)
    curves = pd.DataFrame(rows)
    audit_frame = pd.DataFrame(audits)
    curves.to_csv(args.outdir / "cell_method_capture.csv", index=False)
    audit_frame.to_csv(args.outdir / "operator_audit.csv", index=False)

    focus = curves[curves["channels"] == 8].pivot(
        index="root_id", columns="method", values="capture"
    )
    comparisons: dict[str, Any] = {}
    rng = np.random.default_rng(args.seed + 1)
    for control in ["random real paths", "depth bins", "row-shuffled paths", "degree-depth surrogate tree"]:
        difference = (focus["morphology paths"] - focus[control]).to_numpy(dtype=float)
        comparisons[control] = {
            "mean_morphology_minus_control": float(difference.mean()),
            "cell_bootstrap_ci95": bootstrap_interval(difference, rng),
            "cells_positive": int((difference > 0).sum()),
            "n_cells": int(len(difference)),
            "wilcoxon_two_sided_p": float(
                stats.wilcoxon(difference, alternative="two-sided", zero_method="wilcox").pvalue
            ),
        }
    means = focus.mean().sort_values(ascending=False).to_dict()
    summary = {
        "status": "complete",
        "analysis": "ancestry dictionaries evaluated on an exact reciprocal-cable shunt-response operator",
        "target_generator": (
            "finite-difference derivative of the exact clamped-soma excitatory log-gradient field "
            "with respect to a focal shunt in the reciprocal passive cable system"
        ),
        "n_cells": int(curves["root_id"].nunique()),
        "focus_channels": 8,
        "focus_method_mean_capture": {key: float(value) for key, value in means.items()},
        "comparisons": comparisons,
        "operator_audit": {
            "mean_off_ancestry_response_energy_fraction": float(
                audit_frame["off_ancestry_response_energy_fraction"].mean()
            ),
            "minimum_off_ancestry_response_energy_fraction": float(
                audit_frame["off_ancestry_response_energy_fraction"].min()
            ),
            "cell_operator_ranks": audit_frame["reciprocal_operator_rank"].astype(int).tolist(),
        },
        "scope": [
            "The target operator is no longer generated by the tested ancestry dictionary.",
            "The operator remains an in-silico passive-cable response on MICrONS morphology, not measured learning credit.",
            "Surrogates preserve node depths and parent out-degrees but not full three-dimensional embedding constraints.",
        ],
    }
    write_json(args.outdir / "summary.json", summary)
    lines = [
        "# Reciprocal-cable routing control",
        "",
        "The target covariance is generated by exact focal-shunt responses of the reciprocal cable model, not by columns of the tested ancestry dictionary.",
        "",
        "## Eight-channel mean capture",
        "",
    ]
    for method, value in means.items():
        lines.append(f"- {method}: {value:.4f}")
    lines.extend(["", "## Morphology contrasts", ""])
    for method, item in comparisons.items():
        lines.append(
            f"- versus {method}: {item['mean_morphology_minus_control']:.4f} "
            f"[{item['cell_bootstrap_ci95'][0]:.4f}, {item['cell_bootstrap_ci95'][1]:.4f}], "
            f"{item['cells_positive']}/{item['n_cells']} cells, Wilcoxon p={item['wilcoxon_two_sided_p']:.4g}"
        )
    lines.extend(
        [
            "",
            f"Mean off-ancestry response energy: {summary['operator_audit']['mean_off_ancestry_response_energy_fraction']:.4f}.",
            "",
            "This is an independent-generator structural control, but it remains a modeled cable response rather than an in-vivo credit measurement.",
        ]
    )
    (args.outdir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
