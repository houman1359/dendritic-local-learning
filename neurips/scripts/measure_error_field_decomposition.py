#!/usr/bin/env python3
"""Scope-correct decomposition of exact dendritic compartment-error fields.

This checkpoint-only analysis deliberately keeps feedforward dendritic
populations separate.  Equal soma indices in different network layers are
never treated as one neuron.  For each recorded stage and each population it
reports:

* unrestricted rank-1 geometry;
* oracle global, neuron-indexed, local-template, and neuron x template fields;
* the actual submitted matched-width/scalar-fallback field;
* the available neuron-wise field formed from the model's soma coordinates.

Captured-energy fractions are computed per run as ``1 - residual**2`` and only
then averaged in the summary.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
DRAFT_ROOT = SCRIPT_DIR.parent
REPO_ROOT = SCRIPT_DIR.parents[3]
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(REPO_ROOT / "src"))

from measure_error_rank_diagnostics import (  # noqa: E402
    _collect_exact_error_matrices,
    _cosine_to_exact,
    _group_segments_by_soma,
    _residual_to_exact,
    _ungroup_soma_segments,
)


_LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)")
_STAGE_RE = re.compile(r"(?:^|\.)branch_layers\.(\d+)(?:\.|$)")


@dataclass
class Scope:
    name: str
    scope_type: str
    population: int
    stage: int | None
    stage_role: str
    exact: torch.Tensor
    template: torch.Tensor
    scalar_broadcast: torch.Tensor | None
    ancestry_broadcast: torch.Tensor | None
    actual_broadcast: torch.Tensor | None
    n_soma: int
    segment_widths: list[int]
    stage_names: list[str]
    population_energy: float


def _default_run_dirs() -> list[Path]:
    mnist = (
        DRAFT_ROOT
        / "local_sweep_runs"
        / "gradient_fidelity_vs_ie_nonnegativeinput_fix_20260409164042"
        / "results"
    )
    noise = (
        DRAFT_ROOT
        / "local_sweep_runs"
        / "noise_resilience_rank_bridge_nonnegativeinput_fix_20260409164042"
        / "results"
    )
    return (
        [mnist / f"config_{idx}" for idx in range(10, 15)]
        + [mnist / f"config_{idx}" for idx in range(60, 65)]
        + [noise / f"config_{idx}" for idx in range(0, 5)]
    )


def _population_and_stage(name: str, fallback_stage: int) -> tuple[int, int]:
    pop_match = _LAYER_RE.search(name)
    stage_match = _STAGE_RE.search(name)
    population = int(pop_match.group(1)) if pop_match else 0
    stage = int(stage_match.group(1)) if stage_match else fallback_stage
    return population, stage


def _cat_optional(values: list[torch.Tensor | None]) -> torch.Tensor | None:
    if not values or any(value is None for value in values):
        return None
    return torch.cat([value for value in values if value is not None], dim=1)


def _stage_role(stage: int, max_stage: int) -> str:
    if stage == max_stage:
        return "soma"
    distance = max_stage - stage
    if distance == 1:
        return "proximal"
    if distance == 2:
        return "distal"
    return f"distal_{distance}"


def _build_scopes(
    matrices: dict[str, torch.Tensor],
    broadcasts: dict[str, dict[str, Any]],
) -> list[Scope]:
    stage_records: list[dict[str, Any]] = []
    layer_keys = sorted(
        (key for key in matrices if key.startswith("layer_")),
        key=lambda key: int(key.split("_")[1]),
    )
    for fallback_stage, key in enumerate(layer_keys):
        broadcast = broadcasts[key]
        layer_name = str(broadcast.get("layer_name", key))
        population, stage = _population_and_stage(layer_name, fallback_stage)
        exact = matrices[key].detach().float()
        stage_records.append(
            {
                "key": key,
                "layer_name": layer_name,
                "population": population,
                "stage": stage,
                "exact": exact,
                "template": broadcast["template"].detach().float(),
                "scalar_broadcast": broadcast.get("scalar_broadcast"),
                "ancestry_broadcast": broadcast.get("per_soma_broadcast"),
                "actual_broadcast": broadcast.get("actual_broadcast"),
                "n_soma": int(broadcast["n_soma"]),
            }
        )

    scopes: list[Scope] = []
    populations = sorted({int(record["population"]) for record in stage_records})
    for population in populations:
        records = sorted(
            (
                record
                for record in stage_records
                if int(record["population"]) == population
            ),
            key=lambda record: int(record["stage"]),
        )
        max_stage = max(int(record["stage"]) for record in records)
        population_energy = float(
            sum(record["exact"].square().sum().item() for record in records)
        )

        for record in records:
            exact = record["exact"]
            scopes.append(
                Scope(
                    name=f"population_{population}/stage_{record['stage']}",
                    scope_type="stage",
                    population=population,
                    stage=int(record["stage"]),
                    stage_role=_stage_role(int(record["stage"]), max_stage),
                    exact=exact,
                    template=record["template"],
                    scalar_broadcast=record["scalar_broadcast"],
                    ancestry_broadcast=record["ancestry_broadcast"],
                    actual_broadcast=record["actual_broadcast"],
                    n_soma=int(record["n_soma"]),
                    segment_widths=[int(exact.size(1))],
                    stage_names=[str(record["layer_name"])],
                    population_energy=population_energy,
                )
            )

        for scope_type, selected in (
            (
                "population_dendritic",
                [record for record in records if int(record["stage"]) != max_stage],
            ),
            (
                "population_soma",
                [record for record in records if int(record["stage"]) == max_stage],
            ),
            ("population_all", records),
        ):
            if not selected:
                continue
            exact = torch.cat([record["exact"] for record in selected], dim=1)
            scopes.append(
                Scope(
                    name=f"population_{population}/{scope_type}",
                    scope_type=scope_type,
                    population=population,
                    stage=None,
                    stage_role=(
                        "dendritic"
                        if scope_type == "population_dendritic"
                        else "soma"
                        if scope_type == "population_soma"
                        else "all"
                    ),
                    exact=exact,
                    template=torch.cat(
                        [record["template"] for record in selected],
                        dim=1,
                    ),
                    scalar_broadcast=_cat_optional(
                        [record["scalar_broadcast"] for record in selected]
                    ),
                    ancestry_broadcast=_cat_optional(
                        [record["ancestry_broadcast"] for record in selected]
                    ),
                    actual_broadcast=_cat_optional(
                        [record["actual_broadcast"] for record in selected]
                    ),
                    n_soma=int(selected[0]["n_soma"]),
                    segment_widths=[
                        int(record["exact"].size(1)) for record in selected
                    ],
                    stage_names=[
                        str(record["layer_name"]) for record in selected
                    ],
                    population_energy=population_energy,
                )
            )
    return scopes


def _global_projection(exact: torch.Tensor) -> torch.Tensor:
    return exact.mean(dim=1, keepdim=True).expand_as(exact)


def _template_projection(
    exact: torch.Tensor,
    template: torch.Tensor,
) -> torch.Tensor:
    denom = template.square().sum(dim=1, keepdim=True).clamp_min(1e-30)
    coeff = (exact * template).sum(dim=1, keepdim=True) / denom
    return coeff * template


def _neuron_projection(
    exact: torch.Tensor,
    *,
    n_soma: int,
    segment_widths: list[int],
    template: torch.Tensor | None,
) -> torch.Tensor:
    grouped_exact = _group_segments_by_soma(
        exact,
        n_soma=n_soma,
        segment_widths=segment_widths,
    )
    if grouped_exact is None:
        raise ValueError("Scope cannot be grouped by soma")
    exact_blocks, widths = grouped_exact
    if template is None:
        projected_blocks = exact_blocks.mean(dim=2, keepdim=True).expand_as(
            exact_blocks
        )
    else:
        grouped_template = _group_segments_by_soma(
            template,
            n_soma=n_soma,
            segment_widths=segment_widths,
        )
        if grouped_template is None:
            raise ValueError("Template cannot be grouped by soma")
        template_blocks, _ = grouped_template
        denom = template_blocks.square().sum(dim=2, keepdim=True).clamp_min(1e-30)
        coeff = (exact_blocks * template_blocks).sum(
            dim=2,
            keepdim=True,
        ) / denom
        projected_blocks = coeff * template_blocks
    return _ungroup_soma_segments(
        projected_blocks,
        segment_compartments_per_soma=widths,
    )


def _metric_row(
    family: str,
    approx: torch.Tensor,
    exact: torch.Tensor,
    *,
    oracle: bool,
) -> dict[str, Any]:
    residual = _residual_to_exact(approx, exact)
    cosine = _cosine_to_exact(approx, exact)
    denom = float(approx.square().sum().item())
    scale = (
        float((approx * exact).sum().item() / denom)
        if denom > 0
        else float("nan")
    )
    rescaled = scale * approx if denom > 0 else approx
    rescaled_residual = _residual_to_exact(rescaled, exact)
    return {
        "family": family,
        "oracle_coefficients": bool(oracle),
        "residual": residual,
        "captured_energy": 1.0 - residual**2,
        "cosine": cosine,
        "optimal_global_scale": scale,
        "rescaled_residual": rescaled_residual,
        "rescaled_captured_energy": 1.0 - rescaled_residual**2,
    }


def _rank1_row(exact: torch.Tensor) -> dict[str, Any]:
    singular = torch.linalg.svdvals(exact.detach().float())
    total = float(singular.square().sum().item())
    captured = float(singular[0].square().item() / total) if total > 0 else float("nan")
    residual = (max(0.0, 1.0 - captured)) ** 0.5
    return {
        "family": "rank1_svd",
        "oracle_coefficients": True,
        "residual": residual,
        "captured_energy": captured,
        "cosine": captured**0.5,
        "optimal_global_scale": 1.0,
        "rescaled_residual": residual,
        "rescaled_captured_energy": captured,
    }


def analyze_run(
    run_dir: Path,
    *,
    batch_size: int,
    split: str,
    device: torch.device,
) -> list[dict[str, Any]]:
    meta, matrices, broadcasts = _collect_exact_error_matrices(
        run_dir,
        batch_size,
        split,
        device,
    )
    rows: list[dict[str, Any]] = []
    for scope in _build_scopes(matrices, broadcasts):
        exact_energy = float(scope.exact.square().sum().item())
        common = {
            **meta,
            "split": split,
            "batch_size": batch_size,
            "scope": scope.name,
            "scope_type": scope.scope_type,
            "population": scope.population,
            "stage": scope.stage,
            "stage_role": scope.stage_role,
            "stage_names": "|".join(scope.stage_names),
            "n_samples": int(scope.exact.size(0)),
            "n_compartments": int(scope.exact.size(1)),
            "n_soma": int(scope.n_soma),
            "exact_error_energy": exact_energy,
            "population_error_energy_fraction": (
                exact_energy / scope.population_energy
                if scope.population_energy > 0
                else float("nan")
            ),
        }
        approximations = [
            _rank1_row(scope.exact),
            _metric_row(
                "global_oracle",
                _global_projection(scope.exact),
                scope.exact,
                oracle=True,
            ),
            _metric_row(
                "neuron_oracle",
                _neuron_projection(
                    scope.exact,
                    n_soma=scope.n_soma,
                    segment_widths=scope.segment_widths,
                    template=None,
                ),
                scope.exact,
                oracle=True,
            ),
            _metric_row(
                "template_oracle",
                _template_projection(scope.exact, scope.template),
                scope.exact,
                oracle=True,
            ),
            _metric_row(
                "neuron_x_template_oracle",
                _neuron_projection(
                    scope.exact,
                    n_soma=scope.n_soma,
                    segment_widths=scope.segment_widths,
                    template=scope.template,
                ),
                scope.exact,
                oracle=True,
            ),
        ]
        for family, approx in (
            ("scalar_available", scope.scalar_broadcast),
            ("neuron_available", scope.ancestry_broadcast),
            ("submitted_mw", scope.actual_broadcast),
        ):
            if isinstance(approx, torch.Tensor):
                approximations.append(
                    _metric_row(
                        family,
                        approx,
                        scope.exact,
                        oracle=False,
                    )
                )
        rows.extend([{**common, **metrics} for metrics in approximations])
    return rows


def _summarize(frame: pd.DataFrame) -> pd.DataFrame:
    group_cols = [
        "dataset",
        "network_type",
        "strategy",
        "rule_variant",
        "error_broadcast_mode",
        "split",
        "scope",
        "scope_type",
        "population",
        "stage",
        "stage_role",
        "family",
        "oracle_coefficients",
    ]
    metric_cols = [
        "residual",
        "captured_energy",
        "cosine",
        "optimal_global_scale",
        "rescaled_residual",
        "rescaled_captured_energy",
        "exact_error_energy",
        "population_error_energy_fraction",
    ]
    return (
        frame.groupby(group_cols, dropna=False)[metric_cols]
        .agg(["mean", "std", "count"])
        .reset_index()
        .pipe(
            lambda table: table.set_axis(
                [
                    "_".join(str(part) for part in col if str(part))
                    if isinstance(col, tuple)
                    else str(col)
                    for col in table.columns
                ],
                axis=1,
            )
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        action="append",
        default=[],
        help="Completed checkpoint directory; repeat for multiple runs.",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--split",
        choices=["train", "valid", "test"],
        nargs="+",
        default=["test"],
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--output-runs",
        type=Path,
        default=DRAFT_ROOT / "figures" / "data" / "error_field_decomposition_runs.csv",
    )
    parser.add_argument(
        "--output-summary",
        type=Path,
        default=DRAFT_ROOT / "figures" / "data" / "error_field_decomposition_summary.csv",
    )
    args = parser.parse_args()

    run_dirs = args.run_dir or _default_run_dirs()
    missing = [path for path in run_dirs if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing run directories: {missing}")
    device = torch.device(
        args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu"
    )
    rows: list[dict[str, Any]] = []
    total = len(run_dirs) * len(args.split)
    completed = 0
    for run_dir in run_dirs:
        for split in args.split:
            completed += 1
            print(f"[{completed}/{total}] {run_dir} ({split})")
            rows.extend(
                analyze_run(
                    run_dir,
                    batch_size=args.batch_size,
                    split=split,
                    device=device,
                )
            )
    frame = pd.DataFrame(rows)
    summary = _summarize(frame)
    args.output_runs.parent.mkdir(parents=True, exist_ok=True)
    args.output_summary.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_runs, index=False)
    summary.to_csv(args.output_summary, index=False)
    print(f"Wrote {args.output_runs}")
    print(f"Wrote {args.output_summary}")


if __name__ == "__main__":
    main()
