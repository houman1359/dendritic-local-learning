#!/usr/bin/env python3
"""Attribute focal gradient localization to adjoint and driving-force factors.

The focal decomposition stores four localization payoffs at every modeled site:

* a first-order-current-matched additive perturbation;
* the post-shunt adjoint with the baseline driving force (adjoint only);
* the baseline adjoint with the post-shunt driving force (driving force only);
* the post-shunt adjoint and driving force together (full shunt).

For a payoff ``v`` and reference ``v00``, the exact two-factor Shapley values are

    phi_adjoint = 1/2 [(v10 - v00) + (v11 - v01)]
    phi_driving = 1/2 [(v01 - v00) + (v11 - v10)].

Their sum equals ``v11 - v00`` for every focal site.  The primary estimand uses
the matched additive perturbation as ``v00`` and therefore allocates the
full-shunt-versus-additive localization contrast.  Because that additive
control is not the literal state in which neither shunt factor changes, we
also report a prespecified sensitivity with unperturbed localization
(``v00 = 0``).  The latter is the conventional two-factor decomposition of
the full shunt response itself.

Uncertainty is estimated by a hierarchical percentile bootstrap.  Cells are
the sampling units and are resampled with replacement; focal sites are then
resampled within every sampled cell occurrence.  Each cell receives equal
weight regardless of its number of focal sites.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


JOURNAL = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = JOURNAL / "source_data" / "focal_decomposition" / "site_decomposition.csv"
DEFAULT_OUTDIR = JOURNAL / "source_data" / "focal_decomposition"

KEY_COLUMNS = ["root_id", "focal_segment_id", "dose"]
MODE_COLUMN = "mode"
PAYOFF_COLUMN = "localization_index"
MODES = (
    "matched additive",
    "adjoint only",
    "driving force only",
    "full shunt",
)
METRICS = (
    "total_contrast",
    "adjoint_shapley",
    "driving_force_shapley",
    "factor_interaction",
)
ESTIMANDS = (
    "full_shunt_minus_matched_additive",
    "full_shunt_minus_unperturbed",
)


def portable_path(path: Path) -> str:
    """Return a manuscript-relative path when the file belongs to this package."""
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(JOURNAL))
    except ValueError:
        return path.name


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_payoffs(path: Path) -> pd.DataFrame:
    rows = pd.read_csv(path)
    required = set(KEY_COLUMNS) | {MODE_COLUMN, PAYOFF_COLUMN}
    missing = sorted(required - set(rows.columns))
    if missing:
        raise ValueError(f"missing required columns: {missing}")

    relevant = rows.loc[rows[MODE_COLUMN].isin(MODES)].copy()
    duplicates = relevant.duplicated(KEY_COLUMNS + [MODE_COLUMN], keep=False)
    if duplicates.any():
        examples = relevant.loc[duplicates, KEY_COLUMNS + [MODE_COLUMN]].head().to_dict("records")
        raise ValueError(f"duplicate site-mode rows, for example: {examples}")

    mode_sets = relevant.groupby(KEY_COLUMNS, sort=False)[MODE_COLUMN].agg(set)
    expected = set(MODES)
    incomplete = mode_sets[mode_sets.map(lambda value: value != expected)]
    if not incomplete.empty:
        example = incomplete.index[0]
        raise ValueError(
            f"site {example} does not have exactly the four required modes: "
            f"{sorted(incomplete.iloc[0])}"
        )

    wide = relevant.pivot(index=KEY_COLUMNS, columns=MODE_COLUMN, values=PAYOFF_COLUMN)
    wide = wide.loc[:, list(MODES)].sort_index().reset_index()
    values = wide.loc[:, list(MODES)].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("non-finite localization payoff")
    return wide


def shapley_table(payoffs: pd.DataFrame, estimand: str) -> pd.DataFrame:
    if estimand not in ESTIMANDS:
        raise ValueError(f"unknown estimand: {estimand}")

    matched = payoffs["matched additive"].to_numpy(dtype=float)
    adjoint = payoffs["adjoint only"].to_numpy(dtype=float)
    driving = payoffs["driving force only"].to_numpy(dtype=float)
    full = payoffs["full shunt"].to_numpy(dtype=float)
    if estimand == "full_shunt_minus_matched_additive":
        reference = matched
    else:
        reference = np.zeros_like(full)

    adjoint_first = adjoint - reference
    adjoint_second = full - driving
    driving_first = driving - reference
    driving_second = full - adjoint
    adjoint_phi = 0.5 * (adjoint_first + adjoint_second)
    driving_phi = 0.5 * (driving_first + driving_second)
    total = full - reference
    interaction = full - adjoint - driving + reference
    residual = adjoint_phi + driving_phi - total

    result = payoffs.loc[:, KEY_COLUMNS].copy()
    result.insert(0, "estimand", estimand)
    result["matched_additive_localization"] = matched
    result["reference_localization"] = reference
    result["adjoint_only_localization"] = adjoint
    result["driving_force_only_localization"] = driving
    result["full_shunt_localization"] = full
    result["adjoint_first_marginal"] = adjoint_first
    result["adjoint_second_marginal"] = adjoint_second
    result["driving_force_first_marginal"] = driving_first
    result["driving_force_second_marginal"] = driving_second
    result["factor_interaction"] = interaction
    result["adjoint_shapley"] = adjoint_phi
    result["driving_force_shapley"] = driving_phi
    result["total_contrast"] = total
    result["reconciliation_residual"] = residual
    return result


def cell_table(site: pd.DataFrame) -> pd.DataFrame:
    value_columns = [
        column
        for column in site.columns
        if column not in {"estimand", "root_id", "focal_segment_id", "dose"}
    ]
    grouped = (
        site.groupby(["estimand", "root_id"], as_index=False, sort=True)[value_columns]
        .mean()
    )
    counts = (
        site.groupby(["estimand", "root_id"], as_index=False, sort=True)
        .size()
        .rename(columns={"size": "n_focal_sites"})
    )
    return counts.merge(grouped, on=["estimand", "root_id"], validate="one_to_one")


def hierarchical_bootstrap(
    site: pd.DataFrame,
    draws: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Resample cells, then sites within sampled cells, with equal cell weight."""

    groups = [
        group.loc[:, list(METRICS)].to_numpy(dtype=float)
        for _, group in site.groupby("root_id", sort=True)
    ]
    n_cells = len(groups)
    if n_cells < 2:
        raise ValueError("hierarchical bootstrap requires at least two cells")

    output = np.empty((draws, len(METRICS)), dtype=float)
    for draw in range(draws):
        sampled_cells = rng.integers(0, n_cells, size=n_cells)
        cell_means = np.empty((n_cells, len(METRICS)), dtype=float)
        for position, cell_index in enumerate(sampled_cells):
            values = groups[int(cell_index)]
            sampled_sites = rng.integers(0, len(values), size=len(values))
            cell_means[position] = values[sampled_sites].mean(axis=0)
        output[draw] = cell_means.mean(axis=0)

    result = pd.DataFrame(output, columns=METRICS)
    result.insert(0, "draw", np.arange(draws, dtype=int))
    denominator = result["total_contrast"].to_numpy(dtype=float)
    result["adjoint_share_of_total"] = result["adjoint_shapley"] / denominator
    result["driving_force_share_of_total"] = (
        result["driving_force_shapley"] / denominator
    )
    result["reconciliation_residual"] = (
        result["adjoint_shapley"]
        + result["driving_force_shapley"]
        - result["total_contrast"]
    )
    return result


def percentile_ci(values: pd.Series) -> list[float]:
    return [float(value) for value in np.quantile(values.to_numpy(dtype=float), [0.025, 0.975])]


def assert_reconciliation(frame: pd.DataFrame, label: str) -> None:
    residual = frame["reconciliation_residual"].to_numpy(dtype=float)
    scale = max(1.0, float(frame["total_contrast"].abs().max()))
    tolerance = 64.0 * np.finfo(float).eps * scale
    maximum = float(np.max(np.abs(residual)))
    if not np.isfinite(maximum) or maximum > tolerance:
        raise ArithmeticError(
            f"{label} Shapley reconciliation failed: maximum residual "
            f"{maximum:.17g} exceeds tolerance {tolerance:.17g}"
        )


def signed_rank(values: np.ndarray) -> float:
    if np.allclose(values, 0.0, atol=0.0, rtol=0.0):
        return 1.0
    return float(stats.wilcoxon(values, alternative="two-sided", method="auto").pvalue)


def summarize_estimand(
    site: pd.DataFrame,
    cell: pd.DataFrame,
    bootstrap: pd.DataFrame,
) -> dict[str, Any]:
    observed = {metric: float(cell[metric].mean()) for metric in METRICS}
    total = observed["total_contrast"]
    observed["adjoint_share_of_total"] = observed["adjoint_shapley"] / total
    observed["driving_force_share_of_total"] = observed["driving_force_shapley"] / total

    inference: dict[str, Any] = {}
    for metric in METRICS:
        cell_values = cell[metric].to_numpy(dtype=float)
        inference[metric] = {
            "hierarchical_bootstrap_ci95": percentile_ci(bootstrap[metric]),
            "cells_positive": int((cell_values > 0).sum()),
            "cells_negative": int((cell_values < 0).sum()),
            "wilcoxon_two_sided_p": signed_rank(cell_values),
        }
    for metric in ("adjoint_share_of_total", "driving_force_share_of_total"):
        inference[metric] = {
            "hierarchical_bootstrap_ci95": percentile_ci(bootstrap[metric])
        }

    return {
        "n_cells": int(cell["root_id"].nunique()),
        "n_focal_sites": int(site[["root_id", "focal_segment_id"]].drop_duplicates().shape[0]),
        "observed_equal_cell_mean": observed,
        "inference": inference,
        "validation": {
            "maximum_absolute_site_reconciliation_residual": float(
                site["reconciliation_residual"].abs().max()
            ),
            "maximum_absolute_cell_reconciliation_residual": float(
                cell["reconciliation_residual"].abs().max()
            ),
            "maximum_absolute_bootstrap_reconciliation_residual": float(
                bootstrap["reconciliation_residual"].abs().max()
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--bootstrap-draws", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260731)
    args = parser.parse_args()
    if args.bootstrap_draws < 1:
        raise ValueError("--bootstrap-draws must be positive")
    args.outdir.mkdir(parents=True, exist_ok=True)

    payoffs = load_payoffs(args.input)
    site_parts: list[pd.DataFrame] = []
    cell_parts: list[pd.DataFrame] = []
    bootstrap_parts: list[pd.DataFrame] = []
    summaries: dict[str, Any] = {}
    seed_sequence = np.random.SeedSequence(args.seed)

    for estimand, child_seed in zip(ESTIMANDS, seed_sequence.spawn(len(ESTIMANDS))):
        site_current = shapley_table(payoffs, estimand)
        cell_current = cell_table(site_current)
        bootstrap_current = hierarchical_bootstrap(
            site_current,
            args.bootstrap_draws,
            np.random.default_rng(child_seed),
        )
        bootstrap_current.insert(0, "estimand", estimand)
        assert_reconciliation(site_current, f"{estimand} site")
        assert_reconciliation(cell_current, f"{estimand} cell")
        assert_reconciliation(bootstrap_current, f"{estimand} bootstrap")
        summaries[estimand] = summarize_estimand(
            site_current,
            cell_current,
            bootstrap_current,
        )
        site_parts.append(site_current)
        cell_parts.append(cell_current)
        bootstrap_parts.append(bootstrap_current)

    site_all = pd.concat(site_parts, ignore_index=True)
    cell_all = pd.concat(cell_parts, ignore_index=True)
    bootstrap_all = pd.concat(bootstrap_parts, ignore_index=True)

    output_paths = {
        "site": args.outdir / "site_shapley.csv",
        "cell": args.outdir / "cell_shapley.csv",
        "bootstrap": args.outdir / "shapley_bootstrap_draws.csv.gz",
        "summary": args.outdir / "shapley_summary.json",
    }
    site_all.to_csv(output_paths["site"], index=False, float_format="%.17g")
    cell_all.to_csv(output_paths["cell"], index=False, float_format="%.17g")
    bootstrap_all.to_csv(
        output_paths["bootstrap"],
        index=False,
        compression={"method": "gzip", "compresslevel": 6, "mtime": 0},
        float_format="%.12g",
    )

    payload = {
        "analysis": "two-factor Shapley attribution of focal gradient localization",
        "input": portable_path(args.input),
        "input_sha256": sha256(args.input),
        "payoff": PAYOFF_COLUMN,
        "primary_estimand": "full_shunt_minus_matched_additive",
        "sensitivity_estimand": "full_shunt_minus_unperturbed",
        "shapley_definition": {
            "adjoint": "0.5 * [(v10 - v00) + (v11 - v01)]",
            "driving_force": "0.5 * [(v01 - v00) + (v11 - v10)]",
            "interaction": "v11 - v10 - v01 + v00",
            "identity": "adjoint + driving_force = v11 - v00",
            "states": {
                "v00_primary": "matched additive localization",
                "v00_sensitivity": "unperturbed localization, exactly zero by definition",
                "v10": "adjoint-only localization",
                "v01": "driving-force-only localization",
                "v11": "full-shunt localization",
            },
        },
        "bootstrap": {
            "method": (
                "hierarchical percentile bootstrap: resample cells with replacement, "
                "then focal sites within each sampled cell occurrence; average sites "
                "within cell and weight cells equally"
            ),
            "draws": int(args.bootstrap_draws),
            "seed": int(args.seed),
            "interval": "2.5th and 97.5th percentiles",
            "sampling_unit": "cell",
        },
        "estimands": summaries,
        "interpretive_boundary": (
            "The matched-additive reference gives an exact Shapley allocation of the "
            "observed full-shunt-versus-additive localization contrast, but it is not "
            "a literal no-factor factorial corner. The unperturbed-reference result is "
            "the conventional factorial sensitivity. Both operate on the nonlinear "
            "localization-index payoff. Adjoint-only and driving-force-only states are "
            "algebraic substitutions in the passive model, not independent biological "
            "interventions or evidence of in vivo plasticity."
        ),
        "outputs": {name: portable_path(path) for name, path in output_paths.items()},
    }
    output_paths["summary"].write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
