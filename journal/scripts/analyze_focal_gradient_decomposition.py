#!/usr/bin/env python3
"""Decompose focal-shunt gradient changes into adjoint and driving-force terms.

For an excitatory conductance at compartment i, the passive-model gradient is

    dL/dg_i = q_i (E_E - V_i),

where q is the soma-derived adjoint.  The primary focal experiment changes
both q and the dendritic voltage field V.  This analysis recomputes the frozen
focal sites and separately substitutes the post-shunt adjoint or post-shunt
driving force while holding the other factor at baseline.  It therefore
quantifies, rather than assumes, how much descendant localization is present
in each factor.  The state-matching compensatory somatic current is retained
in all post-perturbation solves.
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


JOURNAL = Path(__file__).resolve().parents[1]
HELPERS = JOURNAL / "code" / "reconstructed_tree"
if str(HELPERS) not in sys.path:
    sys.path.insert(0, str(HELPERS))

from run_focal_shunting_credit_perturbation import (  # noqa: E402
    choose_focal_sites,
    conductance_system,
    solve_with_soma_clamp,
    stable_rng,
)


DEFAULT_SEGMENTS = JOURNAL / "source_data" / "figure3" / "segment_metrics.csv"
DEFAULT_OUTDIR = JOURNAL / "source_data" / "focal_decomposition"
MODES = (
    "full shunt",
    "adjoint only",
    "driving force only",
    "matched additive",
)


def portable_path(path: Path) -> str:
    """Return a manuscript-relative path when the file belongs to this package."""
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(JOURNAL))
    except ValueError:
        return path.name


def adjoint(
    matrix: np.ndarray,
    root_index: int,
    scalar_output_error: float,
) -> np.ndarray:
    source = np.zeros(matrix.shape[0], dtype=float)
    source[root_index] = float(scalar_output_error)
    return np.linalg.solve(matrix.T, source)


def localization(
    changed: np.ndarray,
    baseline: np.ndarray,
    descendant: np.ndarray,
    matched: np.ndarray,
) -> tuple[float, float, float]:
    epsilon = 1e-15 * max(float(np.max(np.abs(baseline))), 1.0)
    magnitude = np.abs(np.log((np.abs(changed) + epsilon) / (np.abs(baseline) + epsilon)))
    descendant_change = float(np.median(magnitude[descendant]))
    matched_change = float(np.median(magnitude[matched]))
    return descendant_change - matched_change, descendant_change, matched_change


def analyze_cell(
    root_id: int,
    segments: pd.DataFrame,
    dose: float,
    seed: int,
    max_focal_sites: int,
    minimum_sites: int,
    e_scale: float,
    i_scale: float,
    excitatory_reversal: float,
    inhibitory_reversal: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    electrical, matrix0, rhs0, root_index, parents, children = conductance_system(
        segments,
        e_scale,
        i_scale,
        excitatory_reversal,
        inhibitory_reversal,
    )
    ids = electrical["segment_id"].astype(int).tolist()
    index = {segment: position for position, segment in enumerate(ids)}
    voltage0 = np.linalg.solve(matrix0, rhs0)
    output_error = 1.0
    q0 = adjoint(matrix0, root_index, output_error)
    driving0 = float(excitatory_reversal) - voltage0
    gradient0 = q0 * driving0

    e_segments = electrical.loc[electrical["E_size"] > 0, "segment_id"].astype(int).tolist()
    e_indices = np.asarray([index[segment] for segment in e_segments], dtype=int)
    focal_sites, relations = choose_focal_sites(
        electrical,
        e_segments,
        parents,
        children,
        stable_rng(seed, root_id, stream=1),
        max_focal_sites,
        minimum_sites,
    )

    rows: list[dict[str, Any]] = []
    maximum_factorization_error = 0.0
    maximum_clamp_error = 0.0
    for focal in focal_sites:
        focal_index = index[int(focal)]
        local_scale = float(
            electrical.loc[focal_index, "g_leak"]
            + electrical.loc[focal_index, "g_e"]
            + electrical.loc[focal_index, "g_i"]
        )
        delta = float(dose) * local_scale

        matrix_shunt = matrix0.copy()
        rhs_shunt = rhs0.copy()
        matrix_shunt[focal_index, focal_index] += delta
        rhs_shunt[focal_index] += delta * float(inhibitory_reversal)
        voltage_shunt, clamp_shunt, clamp_error_shunt = solve_with_soma_clamp(
            matrix_shunt, rhs_shunt, root_index, voltage0[root_index]
        )
        q_shunt = adjoint(matrix_shunt, root_index, output_error)
        driving_shunt = float(excitatory_reversal) - voltage_shunt

        rhs_additive = rhs0.copy()
        rhs_additive[focal_index] += delta * (
            float(inhibitory_reversal) - voltage0[focal_index]
        )
        voltage_additive, clamp_additive, clamp_error_additive = solve_with_soma_clamp(
            matrix0, rhs_additive, root_index, voltage0[root_index]
        )
        driving_additive = float(excitatory_reversal) - voltage_additive

        fields = {
            "full shunt": q_shunt * driving_shunt,
            "adjoint only": q_shunt * driving0,
            "driving force only": q0 * driving_shunt,
            "matched additive": q0 * driving_additive,
        }
        direct_full = q_shunt * (float(excitatory_reversal) - voltage_shunt)
        maximum_factorization_error = max(
            maximum_factorization_error,
            float(np.max(np.abs(fields["full shunt"] - direct_full))),
        )
        maximum_clamp_error = max(
            maximum_clamp_error,
            float(clamp_error_shunt),
            float(clamp_error_additive),
        )

        descendant = relations[int(focal)]["descendant"]
        matched = relations[int(focal)]["depth-matched unrelated"]
        base = gradient0[e_indices]
        for mode, full_field in fields.items():
            loc, desc, other = localization(
                full_field[e_indices], base, descendant, matched
            )
            rows.append(
                {
                    "root_id": int(root_id),
                    "focal_segment_id": int(focal),
                    "dose": float(dose),
                    "mode": mode,
                    "localization_index": loc,
                    "descendant_change": desc,
                    "matched_unrelated_change": other,
                    "delta_conductance": delta,
                    "soma_clamp_current": (
                        float(clamp_additive)
                        if mode == "matched additive"
                        else float(clamp_shunt)
                    ),
                }
            )

    return rows, {
        "root_id": int(root_id),
        "n_focal_sites": int(len(focal_sites)),
        "maximum_factorization_absolute_error": maximum_factorization_error,
        "maximum_soma_clamp_error": maximum_clamp_error,
    }


def bootstrap_ci(values: np.ndarray, rng: np.random.Generator) -> list[float]:
    draws = rng.choice(values, size=(20_000, len(values)), replace=True).mean(axis=1)
    return [float(value) for value in np.quantile(draws, [0.025, 0.975])]


def summarize(rows: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    cell = (
        rows.groupby(["root_id", "mode"], as_index=False)["localization_index"]
        .mean()
    )
    wide = cell.pivot(index="root_id", columns="mode", values="localization_index")
    mode_summary: dict[str, Any] = {}
    rng = np.random.default_rng(seed)
    for mode in MODES:
        values = wide[mode].to_numpy(dtype=float)
        test = stats.wilcoxon(values, alternative="two-sided")
        mode_summary[mode] = {
            "mean_localization": float(values.mean()),
            "cell_bootstrap_ci95": bootstrap_ci(values, rng),
            "cells_positive": int((values > 0).sum()),
            "n_cells": int(len(values)),
            "wilcoxon_two_sided_p": float(test.pvalue),
        }

    contrasts: dict[str, Any] = {}
    for left, right in (
        ("adjoint only", "driving force only"),
        ("adjoint only", "matched additive"),
        ("full shunt", "matched additive"),
    ):
        values = (wide[left] - wide[right]).to_numpy(dtype=float)
        test = stats.wilcoxon(values, alternative="two-sided")
        name = f"{left} minus {right}"
        contrasts[name] = {
            "mean_difference": float(values.mean()),
            "cell_bootstrap_ci95": bootstrap_ci(values, rng),
            "cells_positive": int((values > 0).sum()),
            "n_cells": int(len(values)),
            "wilcoxon_two_sided_p": float(test.pvalue),
        }
    return cell, {"mode_summary": mode_summary, "contrasts": contrasts}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--segments", type=Path, default=DEFAULT_SEGMENTS)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--dose", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--max-focal-sites", type=int, default=16)
    parser.add_argument("--minimum-sites", type=int, default=3)
    parser.add_argument("--e-scale", type=float, default=0.35)
    parser.add_argument("--i-scale", type=float, default=0.35)
    parser.add_argument("--excitatory-reversal", type=float, default=1.0)
    parser.add_argument("--inhibitory-reversal", type=float, default=-0.2)
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    segments = pd.read_csv(args.segments)
    records: list[dict[str, Any]] = []
    checks: list[dict[str, Any]] = []
    for root_id, group in segments.groupby("root_id", sort=True):
        current, validation = analyze_cell(
            int(root_id),
            group.copy(),
            args.dose,
            args.seed,
            args.max_focal_sites,
            args.minimum_sites,
            args.e_scale,
            args.i_scale,
            args.excitatory_reversal,
            args.inhibitory_reversal,
        )
        records.extend(current)
        checks.append(validation)

    site = pd.DataFrame(records)
    if site.empty:
        raise RuntimeError("no eligible focal sites")
    cell, result = summarize(site, args.seed + 101)
    payload = {
        "analysis": "factorized focal-gradient decomposition at fixed somatic voltage and output error",
        "segments": portable_path(args.segments),
        "dose": float(args.dose),
        "n_cells": int(site["root_id"].nunique()),
        "n_focal_sites": int(site[["root_id", "focal_segment_id"]].drop_duplicates().shape[0]),
        "parameters": {
            "seed": int(args.seed),
            "max_focal_sites_per_cell": int(args.max_focal_sites),
            "minimum_descendant_and_comparison_sites": int(args.minimum_sites),
            "e_scale": float(args.e_scale),
            "i_scale": float(args.i_scale),
            "excitatory_reversal": float(args.excitatory_reversal),
            "inhibitory_reversal": float(args.inhibitory_reversal),
        },
        **result,
        "validation": {
            "maximum_factorization_absolute_error": float(
                max(item["maximum_factorization_absolute_error"] for item in checks)
            ),
            "maximum_soma_clamp_error": float(
                max(item["maximum_soma_clamp_error"] for item in checks)
            ),
            "cell_checks": checks,
        },
        "interpretive_boundary": (
            "The adjoint-only and driving-force-only substitutions are algebraic "
            "factor decompositions of the modeled gradient. They do not constitute "
            "independent biological interventions."
        ),
    }
    site.to_csv(args.outdir / "site_decomposition.csv", index=False)
    cell.to_csv(args.outdir / "cell_decomposition.csv", index=False)
    (args.outdir / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
