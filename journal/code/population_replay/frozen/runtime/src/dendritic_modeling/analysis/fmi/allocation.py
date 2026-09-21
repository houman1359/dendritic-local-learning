"""Exact global allocation of measured replacement candidates.

The historical ``outputs/vit_dino_pilot/maximal_assignment.py`` established a
useful experimental pattern: different teacher layers should receive different
replacement budgets, and the choices should be made jointly rather than by a
uniform density rule.  That script was tied to one OLMo proxy-cell table.  This
module retains the pattern while requiring measured, canonical candidate
records as input.

The allocator deliberately does not invent a weighted quality/resource score.
It enumerates the exact nondominated frontier under an explicitly additive
quality-cost model and then selects either the least-resource assignment below
a declared quality budget or the least-quality assignment below a declared
resource budget.  The full composed model must still be evaluated: additive
local costs are an allocation model, not an end-to-end quality measurement.
"""

from __future__ import annotations

import math
import statistics
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class AllocationState:
    """One point on a global quality/resource frontier."""

    quality_cost: float
    resource_cost: float
    choices: tuple[tuple[str, str], ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "quality_cost": float(self.quality_cost),
            "resource_cost": float(self.resource_cost),
            "choices": dict(self.choices),
        }


def log_ratio_excess(ratio: float) -> float:
    """Return the nonnegative log excess used by historical maximal assignment."""

    value = float(ratio)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"quality ratio must be positive and finite, got {ratio!r}")
    return max(0.0, math.log(value))


def _finite_nonnegative(value: Any, *, name: str) -> float:
    resolved = float(value)
    if not math.isfinite(resolved) or resolved < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative, got {value!r}")
    return resolved


def _aggregate_options(
    records: Iterable[Mapping[str, Any]],
    *,
    quality_cost_key: str,
    resource_cost_key: str,
    allow_non_biological: bool,
    require_replacement: bool,
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, dict[str, list[Mapping[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for record in records:
        for key in ("target_id", "candidate_id", quality_cost_key, resource_cost_key):
            if key not in record:
                raise ValueError(f"allocation record is missing {key!r}")
        grouped[str(record["target_id"])][str(record["candidate_id"])].append(record)
    if not grouped:
        raise ValueError("allocation records must not be empty")

    result: dict[str, list[dict[str, Any]]] = {}
    for target_id, candidates in sorted(grouped.items()):
        options = []
        for candidate_id, replicas in sorted(candidates.items()):
            biological_values = {
                bool(record.get("biological_neuron", True)) for record in replicas
            }
            dense_values = {bool(record.get("is_dense", False)) for record in replicas}
            if len(biological_values) != 1 or len(dense_values) != 1:
                raise ValueError(
                    f"Candidate {candidate_id!r} changes constraint class across replicas"
                )
            biological = biological_values.pop()
            is_dense = dense_values.pop()
            if not allow_non_biological and not biological:
                continue
            if require_replacement and is_dense:
                continue
            plans = [record.get("plan") for record in replicas if "plan" in record]
            if plans and (
                len(plans) != len(replicas) or any(plan != plans[0] for plan in plans)
            ):
                raise ValueError(
                    f"Candidate {candidate_id!r} changes compiled plan across replicas"
                )
            options.append(
                {
                    "target_id": target_id,
                    "candidate_id": candidate_id,
                    "quality_cost": statistics.median(
                        _finite_nonnegative(
                            record[quality_cost_key], name=quality_cost_key
                        )
                        for record in replicas
                    ),
                    "resource_cost": statistics.median(
                        _finite_nonnegative(
                            record[resource_cost_key], name=resource_cost_key
                        )
                        for record in replicas
                    ),
                    "biological_neuron": biological,
                    "is_dense": is_dense,
                    "replicates": len(replicas),
                    "plan": plans[0] if plans else None,
                }
            )
        if not options:
            qualifier = "non-dense " if require_replacement else ""
            raise ValueError(
                f"Target {target_id!r} has no eligible {qualifier}replacement candidates"
            )
        result[target_id] = options
    return result


def _prune_frontier(states: list[AllocationState]) -> list[AllocationState]:
    """Remove exactly dominated states in two minimization dimensions."""

    ordered = sorted(
        states,
        key=lambda state: (state.quality_cost, state.resource_cost, state.choices),
    )
    frontier: list[AllocationState] = []
    best_resource_at_lower_quality = math.inf
    cursor = 0
    while cursor < len(ordered):
        quality = ordered[cursor].quality_cost
        group = []
        while cursor < len(ordered) and ordered[cursor].quality_cost == quality:
            group.append(ordered[cursor])
            cursor += 1
        group_best_resource = min(state.resource_cost for state in group)
        if group_best_resource < best_resource_at_lower_quality:
            # Within one quality coordinate, only the minimum-resource states
            # survive. Preserve assignments tied on both coordinates: their
            # local allocation scores are identical, but their composed-model
            # behavior can differ and must be tested rather than discarded.
            frontier.extend(
                state for state in group if state.resource_cost == group_best_resource
            )
        best_resource_at_lower_quality = min(
            best_resource_at_lower_quality, group_best_resource
        )
    return frontier


def enumerate_allocation_frontier(
    records: Iterable[Mapping[str, Any]],
    *,
    quality_cost_key: str = "quality_cost",
    resource_cost_key: str = "resource_cost",
    allow_non_biological: bool = False,
    require_replacement: bool = True,
    max_frontier_states: int = 1_000_000,
) -> dict[str, Any]:
    """Enumerate the exact global frontier with one candidate per target.

    Replicate records are aggregated by their median. Both costs must already
    be nonnegative and additive under the caller's declared allocation model.
    No approximation is made when the frontier grows large: the function fails
    closed instead of silently truncating scientifically relevant states.
    """

    options = _aggregate_options(
        records,
        quality_cost_key=quality_cost_key,
        resource_cost_key=resource_cost_key,
        allow_non_biological=allow_non_biological,
        require_replacement=require_replacement,
    )
    states = [AllocationState(0.0, 0.0, ())]
    stage_sizes = []
    for target_id, target_options in options.items():
        expanded = [
            AllocationState(
                quality_cost=state.quality_cost + float(option["quality_cost"]),
                resource_cost=state.resource_cost + float(option["resource_cost"]),
                choices=(
                    *state.choices,
                    (str(target_id), str(option["candidate_id"])),
                ),
            )
            for state in states
            for option in target_options
        ]
        states = _prune_frontier(expanded)
        stage_sizes.append(
            {
                "target_id": target_id,
                "candidate_count": len(target_options),
                "frontier_states": len(states),
            }
        )
        if len(states) > int(max_frontier_states):
            raise RuntimeError(
                "Exact allocation frontier exceeded max_frontier_states after "
                f"{target_id!r}: {len(states)} > {max_frontier_states}. Increase "
                "the explicit limit or reduce the candidate menu; no approximate "
                "assignment was returned."
            )

    plan_lookup = {
        target_id: {
            str(option["candidate_id"]): option.get("plan") for option in target_options
        }
        for target_id, target_options in options.items()
    }
    return {
        "schema": "dendritic_global_replacement_frontier/v1",
        "status": "allocation_model_not_end_to_end_measurement",
        "quality_cost_key": quality_cost_key,
        "resource_cost_key": resource_cost_key,
        "quality_composition": "additive_across_targets",
        "allow_non_biological": bool(allow_non_biological),
        "require_replacement": bool(require_replacement),
        "target_count": len(options),
        "stage_sizes": stage_sizes,
        "frontier": [state.as_dict() for state in states],
        "plan_lookup": plan_lookup,
        "required_confirmation": (
            "Evaluate every selected assignment as a fully composed model on "
            "frozen held-out and task-level endpoints."
        ),
    }


def select_quality_budget(
    report: Mapping[str, Any], max_quality_cost: float
) -> dict[str, Any]:
    """Select the least-resource frontier point below a quality-cost budget."""

    budget = _finite_nonnegative(max_quality_cost, name="max_quality_cost")
    eligible = [
        state
        for state in report.get("frontier", [])
        if float(state["quality_cost"]) <= budget
    ]
    if not eligible:
        raise ValueError(f"No global assignment satisfies quality budget {budget}")
    selected = min(
        eligible,
        key=lambda state: (
            float(state["resource_cost"]),
            float(state["quality_cost"]),
            tuple(sorted(state["choices"].items())),
        ),
    )
    return {
        **dict(selected),
        "selection_rule": "minimum_resource_subject_to_quality_budget",
        "max_quality_cost": budget,
        "required_confirmation": report.get("required_confirmation"),
    }


def select_resource_budget(
    report: Mapping[str, Any], max_resource_cost: float
) -> dict[str, Any]:
    """Select the least-quality frontier point below a resource-cost budget."""

    budget = _finite_nonnegative(max_resource_cost, name="max_resource_cost")
    eligible = [
        state
        for state in report.get("frontier", [])
        if float(state["resource_cost"]) <= budget
    ]
    if not eligible:
        raise ValueError(f"No global assignment satisfies resource budget {budget}")
    selected = min(
        eligible,
        key=lambda state: (
            float(state["quality_cost"]),
            float(state["resource_cost"]),
            tuple(sorted(state["choices"].items())),
        ),
    )
    return {
        **dict(selected),
        "selection_rule": "minimum_quality_subject_to_resource_budget",
        "max_resource_cost": budget,
        "required_confirmation": report.get("required_confirmation"),
    }


def selected_plans(
    report: Mapping[str, Any], selection: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    """Resolve a selected frontier point to its frozen compiler plans."""

    lookup = report.get("plan_lookup", {})
    result = {}
    for target_id, candidate_id in selection.get("choices", {}).items():
        try:
            plan = lookup[str(target_id)][str(candidate_id)]
        except KeyError as exc:
            raise KeyError(
                f"No plan is recorded for {target_id!r}/{candidate_id!r}"
            ) from exc
        if plan is None:
            raise ValueError(
                f"Allocation record {target_id!r}/{candidate_id!r} has no compiled plan"
            )
        result[str(target_id)] = dict(plan)
    return result


__all__ = [
    "AllocationState",
    "enumerate_allocation_frontier",
    "log_ratio_excess",
    "select_quality_budget",
    "select_resource_budget",
    "selected_plans",
]
