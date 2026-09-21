"""General structural prune-and-recover plans for replacement networks.

The one-off OLMo experiments established a useful protocol: initialize a
larger sparse replacement, train it, remove the lowest-magnitude retained
contacts, recover, and repeat.  This module expresses the structural part of
that protocol without depending on a teacher architecture or experiment
driver.  Training loops remain responsible for rebuilding their optimizer
after every rung because pruning replaces Parameter objects.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import torch.nn as nn

from dendritic_modeling.deployment.pruning import (
    IndexedPruningTarget,
    apply_indexed_pruning_targets_,
    resolve_indexed_pruning_targets,
)

PRUNING_LADDER_SCHEMA = "dendritic_replacement_pruning_ladder/v1"


@dataclass(frozen=True)
class ResolvedReplacementPruningRung:
    """One immutable structural rung resolved against named replacements."""

    name: str
    recovery_steps: int
    targets_by_unit: dict[str, tuple[IndexedPruningTarget, ...]]
    source_config: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": PRUNING_LADDER_SCHEMA,
            "name": self.name,
            "recovery_steps": self.recovery_steps,
            "targets_by_unit": {
                unit: [asdict(target) for target in targets]
                for unit, targets in self.targets_by_unit.items()
            },
            "source_config": self.source_config,
        }


def _plain_mapping(value: Any, *, label: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    return {str(key): item for key, item in value.items()}


def _merged_target_config(
    rung: Mapping[str, Any],
    unit_override: Mapping[str, Any] | None,
) -> dict[str, Any]:
    result = {
        "default_density": rung.get("default_density"),
        "path_densities": _plain_mapping(
            rung.get("path_densities", {}), label="path_densities"
        ),
        "path_contacts": _plain_mapping(
            rung.get("path_contacts", {}), label="path_contacts"
        ),
        "min_k": int(rung.get("min_k", 1)),
    }
    if unit_override is None:
        return result
    override = _plain_mapping(unit_override, label="per_unit override")
    if "default_density" in override:
        result["default_density"] = override["default_density"]
    if "min_k" in override:
        result["min_k"] = int(override["min_k"])
    result["path_densities"].update(
        _plain_mapping(override.get("path_densities", {}), label="path_densities")
    )
    result["path_contacts"].update(
        _plain_mapping(override.get("path_contacts", {}), label="path_contacts")
    )
    return result


def resolve_replacement_pruning_rung(
    units: Mapping[str, nn.Module],
    rung_config: Mapping[str, Any],
    *,
    rung_index: int = 0,
) -> ResolvedReplacementPruningRung:
    """Resolve path and per-unit selectors before mutating any replacement."""

    rung = _plain_mapping(rung_config, label="pruning rung")
    name = str(rung.get("name") or f"rung_{int(rung_index):02d}")
    recovery_steps = int(rung.get("recovery_steps", 0))
    if recovery_steps < 0:
        raise ValueError("pruning rung recovery_steps must be >= 0")
    per_unit = _plain_mapping(rung.get("per_unit", {}), label="per_unit")
    unknown_units = set(per_unit) - set(units)
    if unknown_units:
        raise ValueError(
            f"Pruning rung {name!r} has unknown per_unit keys: {sorted(unknown_units)}"
        )

    has_global_target = any(
        rung.get(key) not in (None, {}, [])
        for key in ("default_density", "path_densities", "path_contacts")
    )
    targets_by_unit: dict[str, tuple[IndexedPruningTarget, ...]] = {}
    for unit_name, model in units.items():
        override = per_unit.get(unit_name)
        if override is None and not has_global_target:
            continue
        target_config = _merged_target_config(rung, override)
        targets = resolve_indexed_pruning_targets(model, **target_config)
        targets_by_unit[unit_name] = tuple(targets)
    if not targets_by_unit:
        raise ValueError(f"Pruning rung {name!r} selects no replacement unit")
    return ResolvedReplacementPruningRung(
        name=name,
        recovery_steps=recovery_steps,
        targets_by_unit=targets_by_unit,
        source_config=rung,
    )


def apply_replacement_pruning_rung_(
    units: Mapping[str, nn.Module],
    rung: ResolvedReplacementPruningRung,
) -> dict[str, dict[str, dict[str, float]]]:
    """Apply one resolved rung atomically with respect to target validation.

    All paths and current contact counts are checked before the first module is
    changed.  This avoids partially applying a stale plan.
    """

    for unit_name, targets in rung.targets_by_unit.items():
        if unit_name not in units:
            raise ValueError(f"Resolved pruning unit {unit_name!r} is missing")
        current = dict(units[unit_name].named_modules())
        for target in targets:
            module = current.get(target.path)
            if module is None or int(getattr(module, "K", -1)) != target.k_before:
                found = None if module is None else getattr(module, "K", None)
                raise ValueError(
                    f"Stale pruning target {unit_name}.{target.path}: expected "
                    f"K={target.k_before}, found {found}"
                )
    return {
        unit_name: apply_indexed_pruning_targets_(units[unit_name], targets)
        for unit_name, targets in rung.targets_by_unit.items()
    }


def validate_pruning_ladder_config(value: Any) -> list[dict[str, Any]]:
    """Return normalized enabled rungs and reject ambiguous ladder configs."""

    config = _plain_mapping(value, label="pruning_ladder")
    if not bool(config.get("enabled", False)):
        return []
    raw_rungs = config.get("rungs", [])
    if not isinstance(raw_rungs, Sequence) or isinstance(raw_rungs, (str, bytes)):
        raise TypeError("pruning_ladder.rungs must be a sequence of mappings")
    rungs = [
        _plain_mapping(rung, label=f"pruning_ladder.rungs[{index}]")
        for index, rung in enumerate(raw_rungs)
    ]
    if not rungs:
        raise ValueError("Enabled pruning_ladder requires at least one rung")
    names = [
        str(rung.get("name") or f"rung_{index:02d}") for index, rung in enumerate(rungs)
    ]
    if len(names) != len(set(names)):
        raise ValueError("Pruning rung names must be unique")
    return rungs


__all__ = [
    "PRUNING_LADDER_SCHEMA",
    "ResolvedReplacementPruningRung",
    "apply_replacement_pruning_rung_",
    "resolve_replacement_pruning_rung",
    "validate_pruning_ladder_config",
]
