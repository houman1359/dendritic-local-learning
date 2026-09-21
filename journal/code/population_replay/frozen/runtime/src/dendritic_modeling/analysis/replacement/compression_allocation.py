"""Allocate mixed module formats against an explicit whole-model byte budget.

This thin accounting layer reuses the exact FMI allocation engine. Modules may
remain intact, and every result includes the untouched model remainder. Local
quality costs are only additive *search proxies*: neither this frontier nor
passing single-module gates establishes composed-model capability.

Tensor bytes, logical parameter counts and serialized file bytes have separate
ledgers. Quantized logical parameters count represented weights, not uint8
storage elements or the number of registered ``nn.Parameter`` elements. Module
files can be summed only for an explicitly specified bundle; their sum does
not predict the size of a newly serialized monolithic checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Iterable, Mapping
from fractions import Fraction
from pathlib import Path
from typing import Any

from dendritic_modeling.analysis.fmi.allocation import enumerate_allocation_frontier

REQUIRED_CONFIRMATION = (
    "Local quality costs are additive search proxies, not additive capability "
    "errors. Evaluate each selected composition against the same teacher on "
    "frozen capability gates after export/reload; measure its full artifact "
    "bytes and peak memory separately. No composition is admitted by this tool."
)


def _count(value: Any, name: str, *, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if value < int(positive):
        raise ValueError(f"{name} must be {'positive' if positive else 'nonnegative'}")
    return value


def allocate_compression_frontier(
    records: Iterable[Mapping[str, Any]],
    *,
    teacher_tensor_bytes: int,
    teacher_parameter_count: int,
    teacher_checkpoint_file_bytes: int | None = None,
    keep_intact: Iterable[str] = (),
    target_compression: float = 3.0,
    fixed_bundle_file_bytes: int | None = None,
    max_frontier_states: int = 100_000,
) -> dict[str, Any]:
    """Return an exact byte/proxy Pareto frontier and the physical byte ceiling.

    Each disjoint module requires one intact record. Records contain
    ``module_id``, ``candidate_id``, ``is_intact``, ``tensor_bytes``,
    ``logical_parameter_count`` and nonnegative ``local_quality_cost``. Optional
    source bindings and other metadata survive in ``candidate_records``.
    Replicate measurements must be summarized before allocation: candidate IDs
    are unique per module, preventing median byte counts from inventing formats.

    ``fixed_bundle_file_bytes``, when supplied, must cover all unmodified state
    and other required bundle files; every candidate must then supply its
    measured ``module_file_bytes``. Otherwise candidate file bytes stay unknown.
    The tensor budget excludes transient buffers and allocator reservations.
    """
    teacher_tensor_bytes = _count(
        teacher_tensor_bytes, "teacher_tensor_bytes", positive=True
    )
    teacher_parameter_count = _count(
        teacher_parameter_count, "teacher_parameter_count", positive=True
    )
    if teacher_checkpoint_file_bytes is not None:
        _count(
            teacher_checkpoint_file_bytes,
            "teacher_checkpoint_file_bytes",
            positive=True,
        )
    if fixed_bundle_file_bytes is not None:
        _count(fixed_bundle_file_bytes, "fixed_bundle_file_bytes")
    _count(max_frontier_states, "max_frontier_states", positive=True)
    if isinstance(target_compression, bool) or not math.isfinite(target_compression):
        raise ValueError("target_compression must be finite and at least 1")
    target = Fraction(str(target_compression))
    if target < 1:
        raise ValueError("target_compression must be finite and at least 1")

    grouped: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for source in records:
        row = dict(source)
        for key in ("module_id", "candidate_id"):
            if not isinstance(row.get(key), str) or not row[key]:
                raise ValueError(f"{key} must be a nonempty string")
        module, candidate = row["module_id"], row["candidate_id"]
        if candidate in grouped[module]:
            raise ValueError(f"duplicate candidate {module}/{candidate}")
        if type(row.get("is_intact")) is not bool:
            raise ValueError("is_intact must be explicitly boolean")
        for key in ("tensor_bytes", "logical_parameter_count"):
            _count(row.get(key), key)
        cost = row.get("local_quality_cost")
        if isinstance(cost, bool) or not isinstance(cost, (int, float)):
            raise ValueError("local_quality_cost must be finite and nonnegative")
        if not math.isfinite(cost) or cost < 0:
            raise ValueError("local_quality_cost must be finite and nonnegative")
        if row["is_intact"] and cost != 0:
            raise ValueError("intact local_quality_cost must be zero")
        if "module_file_bytes" in row:
            _count(row["module_file_bytes"], "module_file_bytes")
        elif fixed_bundle_file_bytes is not None:
            raise ValueError("bundle accounting requires every module_file_bytes")
        grouped[module][candidate] = row
    if not grouped:
        raise ValueError("candidate records must not be empty")
    modules = sorted(grouped)
    for index, module in enumerate(modules):
        if any(other.startswith(module + ".") for other in modules[index + 1 :]):
            raise ValueError("module paths must be disjoint, not ancestor/descendant")
    locked = set(keep_intact)
    if locked - grouped.keys():
        raise ValueError(
            f"unknown keep_intact modules: {sorted(locked - grouped.keys())}"
        )

    intact = {}
    for module, options in grouped.items():
        baseline = [row for row in options.values() if row["is_intact"]]
        if len(baseline) != 1:
            raise ValueError(f"module {module} requires exactly one intact candidate")
        intact[module] = baseline[0]
    remainder_bytes = teacher_tensor_bytes - sum(
        r["tensor_bytes"] for r in intact.values()
    )
    remainder_parameters = teacher_parameter_count - sum(
        r["logical_parameter_count"] for r in intact.values()
    )
    if remainder_bytes < 0 or remainder_parameters < 0:
        raise ValueError("intact module totals exceed the whole teacher inventory")
    options = {
        module: (
            [intact[module]] if module in locked else list(grouped[module].values())
        )
        for module in modules
    }
    # The existing exact engine stores resources as floats. Enforce the integer
    # range in which all possible sums are exact instead of rounding byte costs.
    if (
        remainder_bytes
        + sum(max(r["tensor_bytes"] for r in rows) for rows in options.values())
        > 2**53
    ):
        raise ValueError(
            "tensor byte sums exceed exact integer range of allocation engine"
        )
    engine = enumerate_allocation_frontier(
        [
            {
                "target_id": module,
                "candidate_id": row["candidate_id"],
                "quality_cost": row["local_quality_cost"],
                "resource_cost": row["tensor_bytes"],
                "is_dense": row["is_intact"],
            }
            for module, rows in options.items()
            for row in rows
        ],
        allow_non_biological=True,
        require_replacement=False,
        max_frontier_states=max_frontier_states,
    )

    def account(choices: Mapping[str, str]) -> dict[str, Any]:
        chosen = [
            grouped[module][candidate] for module, candidate in sorted(choices.items())
        ]
        tensors = remainder_bytes + sum(r["tensor_bytes"] for r in chosen)
        files = (
            None
            if fixed_bundle_file_bytes is None
            else fixed_bundle_file_bytes + sum(r["module_file_bytes"] for r in chosen)
        )
        return {
            "choices": dict(sorted(choices.items())),
            "intact_modules": [r["module_id"] for r in chosen if r["is_intact"]],
            "local_quality_cost_sum": sum(r["local_quality_cost"] for r in chosen),
            "whole_model_tensor_bytes": tensors,
            "whole_model_logical_parameter_count": remainder_parameters
            + sum(r["logical_parameter_count"] for r in chosen),
            "tensor_compression_factor": (
                teacher_tensor_bytes / tensors if tensors else None
            ),
            "tensor_byte_reduction": 1 - tensors / teacher_tensor_bytes,
            "candidate_bundle_file_bytes": files,
            "file_compression_factor": (
                teacher_checkpoint_file_bytes / files
                if teacher_checkpoint_file_bytes is not None and files
                else None
            ),
            "capability_status": "unmeasured_composition",
        }

    minimum = account(
        {
            module: min(
                rows,
                key=lambda r: (
                    r["tensor_bytes"],
                    r["local_quality_cost"],
                    r["candidate_id"],
                ),
            )["candidate_id"]
            for module, rows in options.items()
        }
    )
    budget = teacher_tensor_bytes * target.denominator // target.numerator
    unavoidable = remainder_bytes + sum(intact[m]["tensor_bytes"] for m in locked)
    return {
        "schema": "whole_model_compression_allocation/v1",
        "status": "byte_accounting_and_search_proxy_not_capability_admission",
        "teacher_tensor_bytes": teacher_tensor_bytes,
        "teacher_parameter_count": teacher_parameter_count,
        "teacher_checkpoint_file_bytes": teacher_checkpoint_file_bytes,
        "uncompressed_remainder_tensor_bytes": remainder_bytes,
        "uncompressed_remainder_parameter_count": remainder_parameters,
        "fixed_bundle_file_bytes": fixed_bundle_file_bytes,
        "file_accounting": "explicit_bundle_only; monolithic candidate file size is unmeasured",
        "keep_intact": sorted(locked),
        "minimum_bytes_for_supplied_options": minimum,
        "zero_byte_replaceable_module_ceiling": (
            teacher_tensor_bytes / unavoidable if unavoidable else None
        ),
        "zero_byte_replaceable_module_ceiling_is_unbounded": unavoidable == 0,
        "target": {
            "compression_factor": float(target),
            "whole_model_integer_tensor_byte_budget": budget,
            "physically_feasible_for_supplied_options": minimum[
                "whole_model_tensor_bytes"
            ]
            <= budget,
            "additional_bytes_to_save": max(
                0, minimum["whole_model_tensor_bytes"] - budget
            ),
            "capability_status": "unmeasured_composition",
        },
        "frontier": [account(point["choices"]) for point in engine["frontier"]],
        "engine_stage_sizes": engine["stage_sizes"],
        "candidate_records": [
            grouped[m][c] for m in modules for c in sorted(grouped[m])
        ],
        "required_confirmation": REQUIRED_CONFIRMATION,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "request", type=Path, help="JSON with records and keyword arguments"
    )
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    raw = args.request.read_bytes()
    report = allocate_compression_frontier(**json.loads(raw))
    report["source_request"] = {
        "path": str(args.request.resolve()),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(args.output), "target": report["target"]}))


if __name__ == "__main__":
    main()
