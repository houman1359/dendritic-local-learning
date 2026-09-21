"""Exact resource matching for compiled PopulationNetwork replacements.

Compression comparisons must not infer deployed resources from a nominal
density or soma width.  E/I pathways, reactivation parameters, sparse-index
dtypes, and readout topology can all change the realized budget.  This module
therefore instantiates the same canonical replacement used by training and
measures its active parameters and persistent runtime bytes directly.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from copy import deepcopy
from dataclasses import dataclass

import torch

from dendritic_modeling.deployment.ledger import module_storage_ledger
from dendritic_modeling.networks.architectures.replacement.compiler import (
    CompiledReplacementPlan,
)
from dendritic_modeling.networks.architectures.transformer import (
    GatedPopulationNetworkFFNReplacement,
    PopulationNetworkFFNReplacement,
    build_population_network_ffn_kwargs_from_core_config,
)


@dataclass(frozen=True)
class ReplacementResourceMeasurement:
    """Exact resources of one newly instantiated compiled replacement."""

    active_parameters: int
    stored_parameters: int
    parameter_bytes: int
    buffer_bytes: int
    index_bytes: int
    runtime_bytes: int

    def metric(self, name: str) -> int:
        """Return a named match metric with a fail-closed vocabulary."""

        if name not in {"active_parameters", "stored_parameters", "runtime_bytes"}:
            raise ValueError(
                "resource metric must be active_parameters, stored_parameters, "
                "or runtime_bytes"
            )
        return int(getattr(self, name))


@dataclass(frozen=True)
class CompiledReplacementResourceMatch:
    """A selected candidate and its differences from a reference plan."""

    reference_plan: CompiledReplacementPlan
    candidate_plan: CompiledReplacementPlan
    reference: ReplacementResourceMeasurement
    candidate: ReplacementResourceMeasurement
    primary_metric: str
    secondary_metric: str
    primary_relative_error: float
    secondary_relative_error: float


def measure_compiled_replacement_resources(
    plan: CompiledReplacementPlan,
    *,
    dtype: torch.dtype = torch.float32,
    resource_only_initialization: bool = True,
) -> ReplacementResourceMeasurement:
    """Measure one compiler plan through the canonical executable module.

    The nested ``replacement_kwargs`` mapping is intentional: the normal
    config translator receives the full ``transformer_replacement`` section,
    not the constructor overrides directly.  Using the wrong level silently
    drops sparse-readout and topology settings and can invalidate a match.
    """

    core_config = deepcopy(plan.core_config)
    if resource_only_initialization:
        population_network = core_config.get("population_network")
        if not isinstance(population_network, dict):
            raise ValueError(
                "compiled PopulationNetwork plan is missing population_network"
            )
        layers = population_network.get("layers")
        if not isinstance(layers, list) or not layers:
            raise ValueError("compiled PopulationNetwork plan has no layers")
        for layer in layers:
            if not isinstance(layer, dict):
                raise ValueError("compiled PopulationNetwork layer must be a mapping")
            defaults = layer.setdefault("population_defaults", {})
            if not isinstance(defaults, dict):
                raise ValueError("population_defaults must be a mapping")
            # Resource measurement depends only on registered tensor shapes,
            # dtypes, persistence, and realized topology.  Analytical dendritic
            # value initialization can require thousands of numerical root/
            # expectation evaluations for extreme structural widths without
            # changing any of those quantities.  Mechanism-neutral
            # initialization preserves the executable module and topology but
            # avoids that irrelevant value-only work.
            defaults["dbl_init_method"] = "mechanism_neutral"

    transformer_replacement = {"replacement_kwargs": plan.replacement_kwargs}
    kwargs = build_population_network_ffn_kwargs_from_core_config(
        core_config,
        transformer_replacement=transformer_replacement,
    )
    replacement_class = (
        GatedPopulationNetworkFFNReplacement
        if plan.replacement_kind == "gated_population_network"
        else PopulationNetworkFFNReplacement
    )
    replacement = replacement_class(hidden_size=plan.hidden_size, **kwargs).to(
        dtype=dtype
    )
    estimate = replacement.parameter_estimate()
    ledger = module_storage_ledger(replacement)
    return ReplacementResourceMeasurement(
        active_parameters=int(estimate["active_total"]),
        stored_parameters=int(estimate["stored_total"]),
        parameter_bytes=int(ledger["parameter_bytes"]),
        buffer_bytes=int(ledger["buffer_bytes"]),
        index_bytes=int(ledger["index_bytes"]),
        runtime_bytes=int(ledger["total_bytes"]),
    )


def _relative_error(candidate: int, reference: int) -> float:
    if reference < 1:
        raise ValueError("reference resource metric must be positive")
    return abs(int(candidate) - int(reference)) / int(reference)


def match_compiled_population_width(
    reference_plan: CompiledReplacementPlan,
    *,
    candidate_factory: Callable[[int], CompiledReplacementPlan],
    widths: Iterable[int],
    primary_metric: str = "active_parameters",
    secondary_metric: str = "runtime_bytes",
    max_primary_relative_error: float | None = None,
    max_secondary_relative_error: float | None = None,
) -> CompiledReplacementResourceMatch:
    """Select a width by exact executable resources, with optional gates.

    ``candidate_factory`` keeps scientific axes explicit at the call site.  It
    should hold support budgets, readout contacts, topology method, and other
    intended controls fixed while changing only the requested width.
    """

    resolved_widths = sorted({int(width) for width in widths})
    if not resolved_widths or resolved_widths[0] < 1:
        raise ValueError("widths must contain at least one positive integer")
    reference = measure_compiled_replacement_resources(reference_plan)
    reference_primary = reference.metric(primary_metric)
    reference_secondary = reference.metric(secondary_metric)

    ranked: list[
        tuple[
            tuple[float, float, int],
            CompiledReplacementPlan,
            ReplacementResourceMeasurement,
        ]
    ] = []
    for width in resolved_widths:
        plan = candidate_factory(width)
        if int(plan.population_width) != width:
            raise ValueError(
                "candidate_factory returned population_width "
                f"{plan.population_width} for requested width {width}"
            )
        measurement = measure_compiled_replacement_resources(plan)
        primary_error = _relative_error(
            measurement.metric(primary_metric), reference_primary
        )
        secondary_error = _relative_error(
            measurement.metric(secondary_metric), reference_secondary
        )
        ranked.append(((primary_error, secondary_error, width), plan, measurement))

    score, plan, measurement = min(ranked, key=lambda item: item[0])
    primary_error, secondary_error, _ = score
    if max_primary_relative_error is not None and primary_error > float(
        max_primary_relative_error
    ):
        raise ValueError(
            f"best {primary_metric} relative error {primary_error:.6g} exceeds "
            f"limit {float(max_primary_relative_error):.6g}"
        )
    if max_secondary_relative_error is not None and secondary_error > float(
        max_secondary_relative_error
    ):
        raise ValueError(
            f"best {secondary_metric} relative error {secondary_error:.6g} exceeds "
            f"limit {float(max_secondary_relative_error):.6g}"
        )
    return CompiledReplacementResourceMatch(
        reference_plan=reference_plan,
        candidate_plan=plan,
        reference=reference,
        candidate=measurement,
        primary_metric=primary_metric,
        secondary_metric=secondary_metric,
        primary_relative_error=float(primary_error),
        secondary_relative_error=float(secondary_error),
    )


__all__ = [
    "CompiledReplacementResourceMatch",
    "ReplacementResourceMeasurement",
    "match_compiled_population_width",
    "measure_compiled_replacement_resources",
]
