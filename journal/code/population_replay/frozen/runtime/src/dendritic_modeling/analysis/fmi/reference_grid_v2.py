"""Scoring utilities for the PopulationNetwork FMI-v2 reference grid.

The frozen FMI-v1 grid and its hand-built proxy cells are deliberately not
modified by this module.  FMI-v2 evaluates executable PopulationNetwork
candidates and keeps three questions separate:

* did FMI select the best measured candidate (top-k and regret)?
* what quality did each candidate attain after the declared training budget?
* is the selected candidate Pareto-efficient in quality and measured cost?

No weighted quality/cost score is invented here.  Downstream reports can pick
an operating point only after declaring their resource constraint.
"""

from __future__ import annotations

import math
import statistics
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

DEFAULT_RESOURCE_METRICS = (
    "compact_checkpoint_bytes",
    "active_parameters",
    "latency_ms",
    "energy_mj",
)


def _as_finite_float(value: Any, *, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return result


def _is_biological(record: Mapping[str, Any]) -> bool:
    """Resolve an explicitly recorded constraint class.

    Grid manifests should always store ``biological_neuron``.  The family-name
    fallback makes older generated candidate records readable, while refusing
    to silently classify an unknown family as biological.
    """

    if "biological_neuron" in record:
        return bool(record["biological_neuron"])
    family = str(record.get("family", "")).lower()
    if "positive_ei" in family:
        return True
    if family.startswith("signed") or "signed_ei" in family:
        return False
    raise ValueError(
        "Each reference-grid record must provide biological_neuron or use a "
        "recognized positive_ei/signed family name"
    )


def _selection_candidate_id(selection: Any) -> str:
    if isinstance(selection, str):
        return selection
    if not isinstance(selection, Mapping):
        raise TypeError("selection entries must be candidate ids or mappings")
    for key in ("selected_candidate", "candidate_id", "compiled_candidate"):
        value = selection.get(key)
        if value:
            return str(value)
    raise ValueError(
        "selection mapping must contain selected_candidate, candidate_id, or "
        "compiled_candidate"
    )


def _ranked(values: Mapping[str, float], *, lower_is_better: bool) -> list[str]:
    return sorted(
        values,
        key=lambda candidate_id: (
            values[candidate_id] if lower_is_better else -values[candidate_id],
            candidate_id,
        ),
    )


def _absolute_regret(
    selected: float,
    best: float,
    *,
    lower_is_better: bool,
) -> float:
    return max(0.0, selected - best if lower_is_better else best - selected)


def _dominates(
    left: Mapping[str, float],
    right: Mapping[str, float],
    *,
    dimensions: Sequence[str],
    quality_key: str,
    lower_is_better: bool,
) -> bool:
    comparisons = []
    for dimension in dimensions:
        left_value = left[dimension]
        right_value = right[dimension]
        if dimension == quality_key and not lower_is_better:
            comparisons.append((left_value >= right_value, left_value > right_value))
        else:
            comparisons.append((left_value <= right_value, left_value < right_value))
    return all(no_worse for no_worse, _ in comparisons) and any(
        strictly_better for _, strictly_better in comparisons
    )


def _pareto_frontier(
    candidates: Mapping[str, Mapping[str, float]],
    *,
    dimensions: Sequence[str],
    quality_key: str,
    lower_is_better: bool,
) -> list[str]:
    return sorted(
        candidate_id
        for candidate_id, values in candidates.items()
        if not any(
            other_id != candidate_id
            and _dominates(
                other_values,
                values,
                dimensions=dimensions,
                quality_key=quality_key,
                lower_is_better=lower_is_better,
            )
            for other_id, other_values in candidates.items()
        )
    )


def score_reference_grid(
    records: Iterable[Mapping[str, Any]],
    selections: Mapping[str, Any],
    *,
    quality_key: str = "valid_loss",
    lower_is_better: bool = True,
    allow_non_biological: bool = False,
    resource_metrics: Sequence[str] = DEFAULT_RESOURCE_METRICS,
) -> dict[str, Any]:
    """Score prospective selections against completed candidate measurements.

    Records are grouped by ``target_id`` and ``candidate_id``. Replicated
    seeds are aggregated by the median before candidates are ranked. By
    default, relaxed candidates are outside the admissible domain; passing
    ``allow_non_biological=True`` expands the comparison set and also reports
    the relative quality cost of the best strict positive-E/I candidate.
    """

    grouped: dict[str, dict[str, list[Mapping[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for record in records:
        if "target_id" not in record or "candidate_id" not in record:
            raise ValueError("records require target_id and candidate_id")
        if quality_key not in record:
            raise ValueError(f"records require the quality metric {quality_key!r}")
        _as_finite_float(record[quality_key], name=quality_key)
        grouped[str(record["target_id"])][str(record["candidate_id"])].append(record)

    if not grouped:
        raise ValueError("reference-grid records must not be empty")
    missing = sorted(set(grouped) - {str(key) for key in selections})
    if missing:
        raise ValueError(f"No prospective selection was supplied for targets {missing}")

    per_target: list[dict[str, Any]] = []
    for target_id in sorted(grouped):
        candidate_records = grouped[target_id]
        aggregates: dict[str, dict[str, Any]] = {}
        for candidate_id, replicas in sorted(candidate_records.items()):
            biological_values = {_is_biological(record) for record in replicas}
            if len(biological_values) != 1:
                raise ValueError(
                    f"Candidate {candidate_id!r} changes biological constraint "
                    "class across replicas"
                )
            biological = biological_values.pop()
            metrics: dict[str, float] = {
                quality_key: statistics.median(
                    _as_finite_float(record[quality_key], name=quality_key)
                    for record in replicas
                )
            }
            for metric in resource_metrics:
                present = [record[metric] for record in replicas if metric in record]
                if present:
                    if len(present) != len(replicas):
                        raise ValueError(
                            f"Metric {metric!r} is missing from some replicas of "
                            f"candidate {candidate_id!r}"
                        )
                    metrics[metric] = statistics.median(
                        _as_finite_float(value, name=metric) for value in present
                    )
            aggregates[candidate_id] = {
                "candidate_id": candidate_id,
                "family": str(replicas[0].get("family", candidate_id)),
                "biological_neuron": biological,
                "replicates": len(replicas),
                "metrics": metrics,
            }

        eligible = {
            candidate_id: aggregate
            for candidate_id, aggregate in aggregates.items()
            if allow_non_biological or aggregate["biological_neuron"]
        }
        if not eligible:
            raise ValueError(f"Target {target_id!r} has no eligible candidates")
        selected_id = _selection_candidate_id(selections[target_id])
        if selected_id not in eligible:
            available = sorted(eligible)
            raise ValueError(
                f"Selected candidate {selected_id!r} is not in the admissible grid "
                f"for {target_id!r}; eligible candidates are {available}"
            )

        quality = {
            candidate_id: float(aggregate["metrics"][quality_key])
            for candidate_id, aggregate in eligible.items()
        }
        ranking = _ranked(quality, lower_is_better=lower_is_better)
        best_id = ranking[0]
        best_value = quality[best_id]
        selected_value = quality[selected_id]
        median_value = statistics.median(quality.values())
        absolute_regret = _absolute_regret(
            selected_value,
            best_value,
            lower_is_better=lower_is_better,
        )
        median_gap = _absolute_regret(
            median_value,
            best_value,
            lower_is_better=lower_is_better,
        )
        normalization = (
            median_gap if median_gap > 1e-12 else max(abs(best_value), 1e-12)
        )

        common_resources = [
            metric
            for metric in resource_metrics
            if all(metric in aggregate["metrics"] for aggregate in eligible.values())
        ]
        pareto_dimensions = [quality_key, *common_resources]
        pareto_values = {
            candidate_id: {
                dimension: float(aggregate["metrics"][dimension])
                for dimension in pareto_dimensions
            }
            for candidate_id, aggregate in eligible.items()
        }
        frontier = _pareto_frontier(
            pareto_values,
            dimensions=pareto_dimensions,
            quality_key=quality_key,
            lower_is_better=lower_is_better,
        )

        target_result: dict[str, Any] = {
            "target_id": target_id,
            "domain": (
                "biological_and_relaxed" if allow_non_biological else "biological_only"
            ),
            "candidate_count": len(eligible),
            "selected_candidate": selected_id,
            "best_candidate": best_id,
            "selected_rank": ranking.index(selected_id) + 1,
            "top1": selected_id == best_id,
            "top3": selected_id in ranking[:3],
            "quality_metric": quality_key,
            "lower_is_better": lower_is_better,
            "selected_quality": selected_value,
            "best_quality": best_value,
            "absolute_regret": absolute_regret,
            "relative_regret_to_best": absolute_regret / max(abs(best_value), 1e-12),
            "normalized_regret": absolute_regret / normalization,
            "normalized_regret_denominator": normalization,
            "ranking": ranking,
            "pareto_dimensions": pareto_dimensions,
            "pareto_frontier": frontier,
            "selected_is_pareto": selected_id in frontier,
            "candidates": [eligible[candidate_id] for candidate_id in sorted(eligible)],
        }

        if allow_non_biological:
            strict_ids = [
                candidate_id
                for candidate_id, aggregate in eligible.items()
                if aggregate["biological_neuron"]
            ]
            if strict_ids:
                strict_best_id = _ranked(
                    {
                        candidate_id: quality[candidate_id]
                        for candidate_id in strict_ids
                    },
                    lower_is_better=lower_is_better,
                )[0]
                strict_regret = _absolute_regret(
                    quality[strict_best_id],
                    best_value,
                    lower_is_better=lower_is_better,
                )
                target_result["positive_ei_best_candidate"] = strict_best_id
                target_result["positive_ei_absolute_regret"] = strict_regret
                target_result["positive_ei_regret"] = strict_regret / max(
                    abs(best_value), 1e-12
                )

        per_target.append(target_result)

    normalized_regrets = [result["normalized_regret"] for result in per_target]
    report = {
        "schema": "dendritic_fmi_reference_grid_report/v2",
        "quality_metric": quality_key,
        "lower_is_better": lower_is_better,
        "allow_non_biological": bool(allow_non_biological),
        "target_count": len(per_target),
        "top1_rate": statistics.mean(float(result["top1"]) for result in per_target),
        "top3_rate": statistics.mean(float(result["top3"]) for result in per_target),
        "mean_normalized_regret": statistics.mean(normalized_regrets),
        "median_normalized_regret": statistics.median(normalized_regrets),
        "pareto_selection_rate": statistics.mean(
            float(result["selected_is_pareto"]) for result in per_target
        ),
        "targets": per_target,
    }
    if allow_non_biological:
        strict_regrets = [
            result["positive_ei_regret"]
            for result in per_target
            if "positive_ei_regret" in result
        ]
        report["positive_ei_targets_compared"] = len(strict_regrets)
        report["median_positive_ei_regret"] = (
            statistics.median(strict_regrets) if strict_regrets else None
        )
    return report


__all__ = ["DEFAULT_RESOURCE_METRICS", "score_reference_grid"]
