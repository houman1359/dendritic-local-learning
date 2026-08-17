#!/usr/bin/env python3
"""Wiring-normalized capture reanalysis of the MICrONS routing-capacity data.

This is a deterministic secondary analysis of the frozen Figure 3 source
table (``source_data/figure3/routing_capacity_curves.csv.gz``).  For every
reconstructed cell, feedback dictionary, and channel count it normalizes
field capture by the wiring cost of the dictionary (``wiring_density``, the
fraction of dense-feedback nonzeros) and reports capture per unit wiring,
both in absolute terms and relative to the dense PCA oracle evaluated on the
same cell at the same channel count.

Conventions follow the paper: the cell (``root_id``) is the unit of
inference; Monte Carlo streams (``seed``) are nested within cell and are
averaged within cell before any derived ratio is formed.  Uncertainty is a
cell-level bootstrap (20,000 resamples of the 8 cells) drawn once from
``numpy.random.default_rng(20260812)`` and shared across every dictionary,
channel count, and metric, so intervals are paired over the same resampled
cells.  Before writing any output, the script verifies that the pipeline
reproduces the published 8-channel numbers (capture and wiring density per
dictionary and the morphology/oracle capture fraction) to 5e-3 and aborts if
it cannot.  It does not alter the frozen source table.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "source_data" / "figure3" / "routing_capacity_curves.csv.gz"
OUTPUT = ROOT / "source_data" / "capture_per_wire"

BOOTSTRAP_SEED = 20260812
BOOTSTRAP_DRAWS = 20_000
CHANNELS = (1, 2, 4, 8)
ORACLE = "dense PCA oracle"
METHODS = (
    "dense PCA oracle",
    "morphology-aware paths",
    "random paths",
    "depth-only bins",
    "shuffled ancestry",
)

# Published 8-channel anchors (main text, Figure 3): the reanalysis must
# reproduce these from the frozen table before its outputs can be trusted.
PUBLISHED_8CH = {
    "morphology-aware paths": {"credit_capture": 0.487, "wiring_density": 0.0692},
    "random paths": {"credit_capture": 0.285},
    "depth-only bins": {"credit_capture": 0.222},
    "shuffled ancestry": {"credit_capture": 0.198},
}
PUBLISHED_MORPH_ORACLE_FRACTION = 0.850
TOLERANCE = 5e-3


def cell_level_table(curves: pd.DataFrame) -> pd.DataFrame:
    """Average Monte Carlo streams within cell, then form wiring ratios."""

    kept = curves[curves.channels.isin(CHANNELS)].copy()
    cell = kept.groupby(["root_id", "method", "channels"], as_index=False).agg(
        n_streams=("seed", "nunique"),
        credit_capture=("credit_capture", "mean"),
        wiring_nonzeros=("wiring_nonzeros", "mean"),
        wiring_density=("wiring_density", "mean"),
        oracle_capture=("oracle_capture", "mean"),
        oracle_fraction=("oracle_fraction", "mean"),
    )
    # The oracle is re-estimated per Monte Carlo stream, but within a stream
    # every method row must reference the same oracle capture, and the
    # oracle's own capture must equal that reference.
    oracle_spread = (
        kept.groupby(["root_id", "channels", "seed"]).oracle_capture.std(ddof=0).max()
    )
    if not float(oracle_spread) < 1e-12:
        raise SystemExit("oracle reference differs across methods within a stream")
    oracle_rows = kept[kept.method.eq(ORACLE)]
    if not np.allclose(oracle_rows.credit_capture, oracle_rows.oracle_capture):
        raise SystemExit("dense-oracle rows disagree with their oracle reference")

    cell["capture_per_wire"] = cell.credit_capture / cell.wiring_density
    oracle = cell[cell.method.eq(ORACLE)][
        ["root_id", "channels", "credit_capture", "wiring_density"]
    ].rename(
        columns={
            "credit_capture": "oracle_credit_capture",
            "wiring_density": "oracle_wiring_density",
        }
    )
    cell = cell.merge(oracle, on=["root_id", "channels"], validate="many_to_one")
    cell["oracle_capture_per_wire"] = (
        cell.oracle_credit_capture / cell.oracle_wiring_density
    )
    # Capture relative to the same-cell dense oracle, per unit wiring.
    cell["relative_capture_per_wire"] = (
        cell.credit_capture / cell.oracle_credit_capture
    ) / cell.wiring_density
    # Factor by which the dictionary beats the dense oracle in capture
    # delivered per unit of wiring (identically 1 for the oracle itself).
    cell["capture_per_wire_vs_oracle"] = (
        cell.capture_per_wire / cell.oracle_capture_per_wire
    )
    return cell.drop(columns=["oracle_credit_capture", "oracle_wiring_density"])


def verify_published_anchors(cell: pd.DataFrame) -> dict:
    """Reproduce the published 8-channel numbers or abort."""

    checks = {}
    eight = cell[cell.channels.eq(8)]
    for method, anchors in PUBLISHED_8CH.items():
        part = eight[eight.method.eq(method)]
        for column, published in anchors.items():
            observed = float(part[column].mean())
            checks[f"{method}: mean {column}"] = {
                "published": published,
                "reproduced": observed,
            }
            if abs(observed - published) > TOLERANCE:
                raise SystemExit(
                    f"verification failed for {method} {column}: "
                    f"reproduced {observed:.6f}, published {published}"
                )
    morph_fraction = float(
        eight[eight.method.eq("morphology-aware paths")].oracle_fraction.mean()
    )
    checks["morphology-aware paths: mean oracle_fraction"] = {
        "published": PUBLISHED_MORPH_ORACLE_FRACTION,
        "reproduced": morph_fraction,
    }
    if abs(morph_fraction - PUBLISHED_MORPH_ORACLE_FRACTION) > TOLERANCE:
        raise SystemExit(
            "verification failed for morphology oracle fraction: "
            f"reproduced {morph_fraction:.6f}, "
            f"published {PUBLISHED_MORPH_ORACLE_FRACTION}"
        )
    return checks


def summarize(cell: pd.DataFrame) -> pd.DataFrame:
    """Cell-bootstrap means and 95% intervals per dictionary and channels."""

    cells = np.array(sorted(cell.root_id.unique()))
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    # One shared resample-index matrix: intervals are paired across every
    # dictionary, channel count, and metric.
    resamples = rng.integers(0, len(cells), size=(BOOTSTRAP_DRAWS, len(cells)))
    metrics = (
        "credit_capture",
        "wiring_density",
        "oracle_fraction",
        "capture_per_wire",
        "relative_capture_per_wire",
        "capture_per_wire_vs_oracle",
    )
    rows = []
    for method in METHODS:
        for channels in CHANNELS:
            part = cell[
                cell.method.eq(method) & cell.channels.eq(channels)
            ].set_index("root_id").loc[cells]
            row = {
                "method": method,
                "channels": int(channels),
                "n_cells": int(len(part)),
            }
            for metric in metrics:
                values = part[metric].to_numpy(float)
                sampled = values[resamples].mean(axis=1)
                low, high = np.quantile(sampled, [0.025, 0.975])
                row[f"mean_{metric}"] = float(values.mean())
                row[f"ci95_low_{metric}"] = float(low)
                row[f"ci95_high_{metric}"] = float(high)
            rows.append(row)
    return pd.DataFrame(rows)


def write_report(summary: pd.DataFrame, checks: dict) -> str:
    eight = summary[summary.channels.eq(8)].set_index("method")

    def line(method: str) -> str:
        row = eight.loc[method]
        return (
            f"| {method} | {row.mean_credit_capture:.3f} | "
            f"{100.0 * row.mean_wiring_density:.2f}% | "
            f"{row.mean_capture_per_wire:.2f} | "
            f"{row.mean_capture_per_wire_vs_oracle:.2f} "
            f"({row.ci95_low_capture_per_wire_vs_oracle:.2f}"
            f"-{row.ci95_high_capture_per_wire_vs_oracle:.2f}) |"
        )

    morph = eight.loc["morphology-aware paths"]
    verification = "\n".join(
        f"- {name}: published {values['published']}, "
        f"reproduced {values['reproduced']:.4f}"
        for name, values in checks.items()
    )
    return f"""# Wiring-normalized capture (capture per wire)

Secondary analysis of the frozen Figure 3 routing-capacity table
(`source_data/figure3/routing_capacity_curves.csv.gz`; 8 reconstructed
cells, Monte Carlo streams averaged within cell before any ratio).
Capture per wire divides a dictionary's field capture by its wiring
density (fraction of dense-feedback nonzeros).  The oracle-relative
factor divides each cell's capture-per-wire by the dense PCA oracle's
capture-per-wire on the same cell at the same channel count; intervals
are paired cell bootstraps ({BOOTSTRAP_DRAWS:,} resamples of the 8 cells,
`numpy.random.default_rng({BOOTSTRAP_SEED})`).

## Headline (8 feedback channels, mean over 8 cells)

| dictionary | capture | wiring density | capture per wire | vs dense oracle (95% CI) |
|---|---|---|---|---|
{line('morphology-aware paths')}
{line('random paths')}
{line('depth-only bins')}
{line('shuffled ancestry')}
{line('dense PCA oracle')}

- Ancestry routes reach {100.0 * morph.mean_oracle_fraction:.1f}% of the
  dense oracle's capture while using {100.0 * morph.mean_wiring_density:.2f}% of
  its wiring, a {morph.mean_capture_per_wire_vs_oracle:.1f}x advantage in
  capture per unit wiring (95% CI
  {morph.ci95_low_capture_per_wire_vs_oracle:.1f}-{morph.ci95_high_capture_per_wire_vs_oracle:.1f}).

## Verification against published 8-channel numbers (tolerance {TOLERANCE})

{verification}

## Scope

Wiring density counts feedback nonzeros, not axonal path length or
conduction delay, so capture per wire is a connectivity-budget measure,
not a metabolic one.  The 16-channel rows present in the frozen curves
are outside the published 1-8 channel grid and are not reanalysed here.
The normalization rewards sparse dictionaries by construction; the
substantive comparison is against the equally sparse controls
(ancestry-shuffled paths share the morphology dictionary's exact wiring
density), not against the dense oracle alone.
"""


def main() -> None:
    curves = pd.read_csv(INPUT)
    missing = set(METHODS) - set(curves.method.unique())
    if missing:
        raise SystemExit(f"methods missing from frozen table: {sorted(missing)}")
    if curves.root_id.nunique() != 8:
        raise SystemExit(
            f"expected the 8-cell primary cohort, found {curves.root_id.nunique()}"
        )
    cell = cell_level_table(curves)
    expected = 8 * len(METHODS) * len(CHANNELS)
    if len(cell) != expected:
        raise SystemExit(f"expected {expected} cell rows, found {len(cell)}")
    if not np.isfinite(cell.select_dtypes(include=[np.number])).all().all():
        raise SystemExit("non-finite cell-level metric")
    checks = verify_published_anchors(cell)
    summary = summarize(cell)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    cell.to_csv(OUTPUT / "cell_method_channel.csv", index=False, float_format="%.10g")
    summary.to_csv(OUTPUT / "summary.csv", index=False, float_format="%.10g")
    (OUTPUT / "report.md").write_text(write_report(summary, checks), encoding="utf-8")

    headline = summary[summary.channels.eq(8)][
        [
            "method",
            "mean_credit_capture",
            "mean_wiring_density",
            "mean_capture_per_wire",
            "mean_capture_per_wire_vs_oracle",
            "ci95_low_capture_per_wire_vs_oracle",
            "ci95_high_capture_per_wire_vs_oracle",
        ]
    ]
    print(json.dumps(
        {
            "verification": checks,
            "headline_8_channels": headline.to_dict(orient="records"),
        },
        indent=2,
        sort_keys=True,
    ))


if __name__ == "__main__":
    main()
