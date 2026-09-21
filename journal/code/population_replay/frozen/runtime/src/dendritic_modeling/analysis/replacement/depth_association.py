"""Depth-vs-degradation association for single-layer replacement screens.

Given a completed layerwise replacement screen (one directory per layer, up
to four family arms per layer, each with a ``benchmark_metrics.json`` and a
``metrics.json``), this analysis:

1. Verifies that all pooled cells were benchmarked on an identical
   evaluation protocol (``evaluation_provenance`` plus the trained-side
   window layout), and pools only the modal protocol group.
2. Resolves a teacher baseline on that same protocol.  The per-run
   ``initial`` block is used when it is labelled ``original_teacher_model``
   and is constant across pooled cells; degradation is then the perplexity
   ratio ``exp(trained_lm_loss - teacher_lm_loss)``.  If no shared baseline
   exists, raw ``trained.lm_loss`` is analyzed directly (a monotone proxy on
   identical windows) and the artifact says so.
3. Reduces each layer to its best (minimum trained LM loss) arm and
   computes Pearson and Spearman correlations of depth with best-arm
   degradation, with percentile-bootstrap confidence intervals obtained by
   resampling layers (the experimental unit), never arms.
4. Fits an arm-level OLS of degradation on depth controlling for
   log(density), population width, and family indicators, reporting
   classical and layer-clustered (CR1) standard errors.
5. Repeats the headline correlations with a caller-specified sensitivity
   band of layers excluded (defaults to the ledger's
   ``tail_validation_required_layers`` minus quarantined layers).
6. Correlates each layer's local fit quality (best-arm ``relative_mse``)
   with its end-to-end degradation, to contrast local and end-to-end views.

Layers listed as quarantined in the ledger are excluded entirely.

Example
-------
python -m dendritic_modeling.analysis.replacement.depth_association \\
    --screen-root outputs/canonical_population_fmi \\
    --pattern "olmo2_32b_alloc_screen_L*_v1" \\
    --ledger outputs/canonical_population_fmi/olmo2_32b_alloc_screen_ledger.json \\
    --output-json outputs/canonical_population_fmi/olmo2_32b_depth_association_v1.json
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

ARM_NAME_PATTERN = re.compile(r"^(?P<family>[a-z_]+?)_d(?P<density>\d+p\d+)_")
LAYER_DIR_PATTERN = re.compile(r"_L(?P<layer>\d+)_v\d+$")
KNOWN_FAMILIES = (
    "signed_flat",
    "signed_ei_flat",
    "gated_signed_flat",
    "gated_signed_ei_flat",
)
TEACHER_INITIAL_LABEL = "original_teacher_model"


@dataclass
class ArmRecord:
    """One completed screen cell (a single family arm at a single layer)."""

    layer: int
    arm_name: str
    family: str
    density: float
    population_width: float | None
    trained_lm_loss: float
    initial_lm_loss: float
    initial_label: str
    local_relative_mse: float | None
    dendritic_active_params: int | None
    dendritic_stored_params: int | None
    teacher_dense_params: int | None
    provenance_key: str
    result_dir: str
    degradation: float = math.nan

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe record (without the bulky provenance key)."""

        record = {
            "layer": self.layer,
            "arm_name": self.arm_name,
            "family": self.family,
            "density": self.density,
            "population_width": self.population_width,
            "trained_lm_loss": self.trained_lm_loss,
            "initial_lm_loss": self.initial_lm_loss,
            "local_relative_mse": self.local_relative_mse,
            "dendritic_active_params": self.dendritic_active_params,
            "dendritic_stored_params": self.dendritic_stored_params,
            "teacher_dense_params": self.teacher_dense_params,
            "degradation": self.degradation,
            "result_dir": self.result_dir,
        }
        return record


@dataclass
class CorrelationResult:
    """A correlation with its bootstrap confidence interval."""

    statistic: str
    r: float
    p_value: float
    n: int
    ci_low: float
    ci_high: float
    bootstrap_draws: int
    bootstrap_seed: int
    degenerate_draws: int

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe record."""

        return {
            "statistic": self.statistic,
            "r": self.r,
            "p_value": self.p_value,
            "n": self.n,
            "ci95_low": self.ci_low,
            "ci95_high": self.ci_high,
            "bootstrap_draws": self.bootstrap_draws,
            "bootstrap_seed": self.bootstrap_seed,
            "degenerate_bootstrap_draws_dropped": self.degenerate_draws,
        }


@dataclass
class ScreenTable:
    """All pooled arm records plus the provenance/pooling decisions made."""

    arms: list[ArmRecord]
    provenance_notes: dict[str, Any] = field(default_factory=dict)

    @property
    def layers(self) -> list[int]:
        """Sorted distinct layer indices present in the pooled table."""

        return sorted({arm.layer for arm in self.arms})

    def best_arm_by_layer(self) -> dict[int, ArmRecord]:
        """Reduce to the best (minimum trained LM loss) arm per layer."""

        best: dict[int, ArmRecord] = {}
        for arm in self.arms:
            incumbent = best.get(arm.layer)
            if incumbent is None or arm.trained_lm_loss < incumbent.trained_lm_loss:
                best[arm.layer] = arm
        return best


def parse_arm_name(arm_name: str) -> tuple[str, float]:
    """Extract (family, density) from an arm name.

    Density is encoded with ``p`` for the decimal point, e.g.
    ``signed_flat_d0p27646617624050124_indexed_rewire`` has family
    ``signed_flat`` and density ``0.27646617624050124``.
    """

    match = ARM_NAME_PATTERN.match(arm_name)
    if match is None:
        raise ValueError(f"Cannot parse family/density from arm name: {arm_name}")
    family = match.group("family")
    density = float(match.group("density").replace("p", ".", 1))
    if family not in KNOWN_FAMILIES:
        raise ValueError(f"Unknown family '{family}' in arm name: {arm_name}")
    return family, density


def parse_layer_index(directory_name: str) -> int | None:
    """Extract the layer index from a screen directory name, if present."""

    match = LAYER_DIR_PATTERN.search(directory_name)
    if match is None:
        return None
    return int(match.group("layer"))


def _read_population_width(config_path: Path) -> float | None:
    """Recover the population width from an arm's config YAML.

    The compiled plan stores it as ``population_width`` (some older configs
    use ``n_units``).  A lightweight regex scan avoids a YAML dependency and
    tolerates partially written configs.
    """

    if not config_path.exists():
        return None
    text = config_path.read_text()
    for key in ("n_units", "population_width"):
        match = re.search(rf"^\s*{key}:\s*([0-9.eE+-]+)\s*$", text, re.MULTILINE)
        if match is not None:
            return float(match.group(1))
    return None


def _provenance_key(benchmark: dict[str, Any]) -> str:
    """Canonical string identifying the evaluation protocol of one cell."""

    trained = benchmark.get("trained", {})
    payload = {
        "evaluation_provenance": benchmark.get("evaluation_provenance"),
        "teacher_source": benchmark.get("teacher_source"),
        "window_starts_by_batch": trained.get("window_starts_by_batch"),
        "token_offset": trained.get("token_offset"),
        "requested_token_range": trained.get("requested_token_range"),
    }
    return json.dumps(payload, sort_keys=True)


def _read_local_relative_mse(result_dir: Path, layer: int) -> float | None:
    """Read the local (layer-output) relative MSE from ``metrics.json``."""

    metrics_path = result_dir / "metrics.json"
    if not metrics_path.exists():
        return None
    with metrics_path.open() as handle:
        metrics = json.load(handle)
    layer_metrics = metrics.get("layer_metrics", {}).get(f"layer_{layer}")
    if not isinstance(layer_metrics, dict):
        return None
    value = layer_metrics.get("relative_mse")
    return float(value) if value is not None else None


def load_screen(
    screen_root: Path,
    pattern: str,
    ledger_path: Path,
) -> ScreenTable:
    """Load all complete, non-quarantined cells and pool by protocol.

    Quarantined layers from the ledger are excluded entirely.  Cells are
    grouped by their evaluation protocol; only the modal group is pooled and
    any excluded cells are recorded in ``provenance_notes``.
    """

    with ledger_path.open() as handle:
        ledger = json.load(handle)
    quarantined = set(ledger.get("quarantined_layers", []))
    tail_flagged = set(ledger.get("tail_validation_required_layers", []))

    arms: list[ArmRecord] = []
    skipped_incomplete: list[str] = []
    skipped_quarantined_layers: set[int] = set()
    ledger_layers = ledger.get("layers", {})

    for layer_dir in sorted(screen_root.glob(pattern)):
        layer = parse_layer_index(layer_dir.name)
        if layer is None:
            continue
        if layer in quarantined:
            skipped_quarantined_layers.add(layer)
            continue
        ledger_cells = ledger_layers.get(str(layer), {}).get("cells", {})
        for arm_dir in sorted((layer_dir / "results").glob("*")):
            arm_name = arm_dir.name
            if ledger_cells and ledger_cells.get(arm_name) != "complete":
                skipped_incomplete.append(str(arm_dir))
                continue
            benchmark_path = arm_dir / "benchmark_metrics.json"
            if not benchmark_path.exists():
                skipped_incomplete.append(str(arm_dir))
                continue
            with benchmark_path.open() as handle:
                benchmark = json.load(handle)
            family, density = parse_arm_name(arm_name)
            width = _read_population_width(layer_dir / f"{arm_name}.yaml")
            arms.append(
                ArmRecord(
                    layer=layer,
                    arm_name=arm_name,
                    family=family,
                    density=density,
                    population_width=width,
                    trained_lm_loss=float(benchmark["trained"]["lm_loss"]),
                    initial_lm_loss=float(benchmark["initial"]["lm_loss"]),
                    initial_label=str(benchmark.get("initial_label", "")),
                    local_relative_mse=_read_local_relative_mse(arm_dir, layer),
                    dendritic_active_params=benchmark.get("dendritic_active_params"),
                    dendritic_stored_params=benchmark.get("dendritic_stored_params"),
                    teacher_dense_params=benchmark.get("teacher_dense_params"),
                    provenance_key=_provenance_key(benchmark),
                    result_dir=str(arm_dir),
                )
            )

    if not arms:
        raise RuntimeError(f"No complete cells found under {screen_root}/{pattern}")

    groups: dict[str, list[ArmRecord]] = {}
    for arm in arms:
        groups.setdefault(arm.provenance_key, []).append(arm)
    modal_key = max(groups, key=lambda key: len(groups[key]))
    pooled = groups[modal_key]
    excluded_other_protocol = [
        arm.result_dir
        for key, members in groups.items()
        if key != modal_key
        for arm in members
    ]

    notes: dict[str, Any] = {
        "distinct_evaluation_protocols": len(groups),
        "pooled_cells": len(pooled),
        "cells_excluded_for_protocol_mismatch": excluded_other_protocol,
        "cells_skipped_incomplete": skipped_incomplete,
        "quarantined_layers_excluded": sorted(skipped_quarantined_layers),
        "tail_validation_required_layers": sorted(tail_flagged),
        "modal_protocol": json.loads(modal_key),
    }
    return ScreenTable(arms=pooled, provenance_notes=notes)


def resolve_teacher_baseline(
    table: ScreenTable,
    tolerance: float = 1e-9,
) -> float | None:
    """Resolve a shared teacher LM loss on the pooled protocol, if any.

    The in-run ``initial`` benchmark is accepted as the teacher baseline
    only when it is labelled ``original_teacher_model`` in every pooled cell
    and its LM loss is bitwise-constant (within ``tolerance``) across cells,
    which certifies that it measured the intact teacher on the shared
    windows rather than a partially patched model.
    """

    labels = {arm.initial_label for arm in table.arms}
    if labels != {TEACHER_INITIAL_LABEL}:
        return None
    losses = np.array([arm.initial_lm_loss for arm in table.arms])
    if float(losses.max() - losses.min()) > tolerance:
        return None
    return float(losses.mean())


def bootstrap_correlation(
    x: np.ndarray,
    y: np.ndarray,
    statistic: str,
    draws: int,
    seed: int,
) -> CorrelationResult:
    """Correlation of paired unit-level values with a percentile-bootstrap CI.

    Units (rows of ``x``/``y``) are resampled with replacement; draws in
    which either resampled margin is constant are dropped and counted.
    """

    if statistic == "pearson":
        point = stats.pearsonr(x, y)
    elif statistic == "spearman":
        point = stats.spearmanr(x, y)
    else:
        raise ValueError(f"Unknown statistic: {statistic}")

    rng = np.random.default_rng(seed)
    n = len(x)
    replicates: list[float] = []
    degenerate = 0
    for _ in range(draws):
        idx = rng.integers(0, n, size=n)
        xb, yb = x[idx], y[idx]
        if np.ptp(xb) == 0 or np.ptp(yb) == 0:
            degenerate += 1
            continue
        if statistic == "pearson":
            replicates.append(float(stats.pearsonr(xb, yb).statistic))
        else:
            replicates.append(float(stats.spearmanr(xb, yb).statistic))
    lo, hi = np.percentile(np.array(replicates), [2.5, 97.5])
    return CorrelationResult(
        statistic=statistic,
        r=float(point.statistic),
        p_value=float(point.pvalue),
        n=n,
        ci_low=float(lo),
        ci_high=float(hi),
        bootstrap_draws=draws,
        bootstrap_seed=seed,
        degenerate_draws=degenerate,
    )


def ols_with_layer_clusters(
    y: np.ndarray,
    design: np.ndarray,
    column_names: list[str],
    cluster_ids: np.ndarray,
) -> dict[str, Any]:
    """OLS with classical and layer-clustered (CR1) standard errors.

    Implemented directly with numpy because statsmodels is not installed in
    the analysis environment.  The CR1 estimator applies the usual
    ``G/(G-1) * (n-1)/(n-k)`` small-sample correction, where clusters are
    layers.
    """

    n, k = design.shape
    xtx_inv = np.linalg.inv(design.T @ design)
    beta = xtx_inv @ design.T @ y
    residuals = y - design @ beta
    dof = n - k
    sigma2 = float(residuals @ residuals) / dof
    classical_se = np.sqrt(np.diag(sigma2 * xtx_inv))

    clusters = np.unique(cluster_ids)
    meat = np.zeros((k, k))
    for cluster in clusters:
        mask = cluster_ids == cluster
        xg = design[mask]
        ug = residuals[mask]
        score = xg.T @ ug
        meat += np.outer(score, score)
    n_clusters = len(clusters)
    correction = (n_clusters / (n_clusters - 1)) * ((n - 1) / dof)
    clustered_cov = correction * xtx_inv @ meat @ xtx_inv
    clustered_se = np.sqrt(np.diag(clustered_cov))

    total_ss = float(np.sum((y - y.mean()) ** 2))
    residual_ss = float(residuals @ residuals)
    r_squared = 1.0 - residual_ss / total_ss

    t_stats = beta / clustered_se
    p_values = 2.0 * stats.t.sf(np.abs(t_stats), df=n_clusters - 1)
    return {
        "n_observations": int(n),
        "n_clusters": int(n_clusters),
        "cluster_unit": "layer",
        "r_squared": r_squared,
        "coefficients": {
            name: {
                "estimate": float(beta[i]),
                "classical_se": float(classical_se[i]),
                "layer_clustered_se": float(clustered_se[i]),
                "p_value_clustered": float(p_values[i]),
            }
            for i, name in enumerate(column_names)
        },
        "standard_error_note": (
            "Clustered p-values use a t distribution with n_clusters - 1 "
            "degrees of freedom (CR1 small-sample correction)."
        ),
    }


def build_arm_regression(table: ScreenTable) -> dict[str, Any]:
    """Arm-level OLS of degradation on depth with allocation controls."""

    usable = [arm for arm in table.arms if arm.population_width is not None]
    dropped = len(table.arms) - len(usable)
    reference_family = "signed_flat"
    dummy_families = [f for f in KNOWN_FAMILIES if f != reference_family]

    y = np.array([arm.degradation for arm in usable])
    columns = ["intercept", "depth", "log_density", "population_width"]
    columns += [f"family[{name}]" for name in dummy_families]
    design = np.column_stack(
        [
            np.ones(len(usable)),
            np.array([arm.layer for arm in usable], dtype=float),
            np.log(np.array([arm.density for arm in usable])),
            np.array([arm.population_width for arm in usable], dtype=float),
        ]
        + [
            np.array([1.0 if arm.family == name else 0.0 for arm in usable])
            for name in dummy_families
        ]
    )
    clusters = np.array([arm.layer for arm in usable])
    result = ols_with_layer_clusters(y, design, columns, clusters)
    result["reference_family"] = reference_family
    result["arms_dropped_missing_width"] = dropped
    return result


def analyze(
    table: ScreenTable,
    tail_band: set[int],
    draws: int,
    seed: int,
) -> dict[str, Any]:
    """Compute all headline statistics for a pooled screen table."""

    best = table.best_arm_by_layer()
    layers = np.array(sorted(best), dtype=float)
    degradation = np.array([best[int(layer)].degradation for layer in layers])

    def correlation_block(x: np.ndarray, y: np.ndarray) -> dict[str, Any]:
        return {
            "pearson": bootstrap_correlation(x, y, "pearson", draws, seed).to_dict(),
            "spearman": bootstrap_correlation(x, y, "spearman", draws, seed).to_dict(),
        }

    headline = correlation_block(layers, degradation)

    keep = np.array([int(layer) not in tail_band for layer in layers])
    sensitivity = correlation_block(layers[keep], degradation[keep])

    local_mse = np.array(
        [
            (
                best[int(layer)].local_relative_mse
                if best[int(layer)].local_relative_mse is not None
                else np.nan
            )
            for layer in layers
        ]
    )
    has_local = ~np.isnan(local_mse)
    if has_local.sum() >= 3:
        local_contrast = {
            "local_relative_mse_vs_degradation": correlation_block(
                local_mse[has_local], degradation[has_local]
            ),
            "depth_vs_local_relative_mse": correlation_block(
                layers[has_local], local_mse[has_local]
            ),
            "n_layers_with_local_metric": int(has_local.sum()),
        }
    else:
        local_contrast = {"note": "local relative_mse unavailable"}

    return {
        "n_layers": len(layers),
        "layers": [int(layer) for layer in layers],
        "depth_vs_best_arm_degradation": headline,
        "tail_band_excluded_layers": sorted(tail_band),
        "depth_vs_best_arm_degradation_tail_band_excluded": {
            "n_layers": int(keep.sum()),
            **sensitivity,
        },
        "local_fit_contrast": local_contrast,
        "arm_level_regression": build_arm_regression(table),
        "best_arm_per_layer": {
            str(int(layer)): best[int(layer)].to_dict() for layer in layers
        },
    }


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--screen-root", type=Path, required=True)
    parser.add_argument("--pattern", type=str, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    parser.add_argument(
        "--tail-band",
        type=str,
        default=None,
        help=(
            "Comma-separated layer indices to exclude in the sensitivity "
            "check; defaults to the ledger's tail_validation_required "
            "layers minus quarantined layers."
        ),
    )
    args = parser.parse_args(argv)

    table = load_screen(args.screen_root, args.pattern, args.ledger)

    teacher_lm_loss = resolve_teacher_baseline(table)
    if teacher_lm_loss is not None:
        teacher_baseline_source = (
            "in-run 'initial' benchmark: labelled original_teacher_model and "
            "bitwise-constant across all pooled cells, so it measures the "
            "intact teacher on the shared evaluation windows; no separate "
            "teacher benchmark JSON on this protocol was required"
        )
        degradation_definition = (
            "perplexity_ratio: exp(trained_lm_loss - teacher_lm_loss), where "
            "teacher_lm_loss is the shared original_teacher_model LM loss "
            "measured in-run on the identical evaluation windows"
        )
        for arm in table.arms:
            arm.degradation = math.exp(arm.trained_lm_loss - teacher_lm_loss)
    else:
        teacher_baseline_source = None
        degradation_definition = (
            "trained_lm_loss (no shared teacher baseline was found on the "
            "pooled evaluation protocol; raw trained LM loss is a monotone "
            "proxy for degradation on identical windows)"
        )
        for arm in table.arms:
            arm.degradation = arm.trained_lm_loss

    if args.tail_band is not None:
        tail_band = {int(part) for part in args.tail_band.split(",") if part}
    else:
        tail_band = set(
            table.provenance_notes["tail_validation_required_layers"]
        ) - set(table.provenance_notes["quarantined_layers_excluded"])

    results = analyze(
        table,
        tail_band=tail_band,
        draws=args.bootstrap_draws,
        seed=args.bootstrap_seed,
    )

    artifact = {
        "schema": "dendritic_depth_association/v1",
        "screen_root": str(args.screen_root),
        "pattern": args.pattern,
        "ledger": str(args.ledger),
        "degradation_definition": degradation_definition,
        "teacher_lm_loss": teacher_lm_loss,
        "teacher_baseline_source": teacher_baseline_source,
        "best_arm_convention": (
            "per layer, degradation is taken from the arm with the minimum "
            "trained LM loss across that layer's pooled arms"
        ),
        "provenance": table.provenance_notes,
        "claim_boundary": (
            "MEASURED ASSOCIATION on a completed-but-single-seed screen: one "
            "training seed per cell, one evaluation protocol; depth is not "
            "randomized, so no causal claim is made. Layers flagged "
            "tail-validation-required are included in the headline numbers; "
            "the tail-band-excluded block is the sensitivity result for that "
            "flag. Quarantined layers are excluded entirely."
        ),
        "results": results,
        "arm_records": [arm.to_dict() for arm in table.arms],
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w") as handle:
        json.dump(artifact, handle, indent=1)
    print(f"Wrote {args.output_json}")


if __name__ == "__main__":
    main()
