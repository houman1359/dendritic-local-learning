"""Exact model-resource accounting for confirmatory experiments."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, fields, is_dataclass
from typing import Any

import torch

from dendritic_modeling.analysis.utils.dendritic_depth import (
    soma_relative_dendritic_depth,
)
from dendritic_modeling.analysis.utils.effective_synapses import (
    effective_synapse_snapshot,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.dendritic.branch_layer import (
    DendriticBranchLayer,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.synapse import (
    DenseToSparseLinear,
    TopKLinear,
)

_SYNAPSE_PATHWAYS = {
    "branch_excitation": "ff_excitatory",
    "branch_inhibition": "ff_inhibitory",
    "branch_recurrent": "rec_excitatory",
    "branch_rec_inhibition": "rec_inhibitory",
}


@dataclass(frozen=True)
class SynapseResourceRecord:
    module: str
    pathway: str
    soma_relative_depth: int
    out_features: int
    in_features: int
    candidate_parameters: int
    candidate_slots: int
    active_synapses: int
    realized_k_min: int
    realized_k_max: int
    realized_k_mean: float
    selection_policy: str
    mask_source: str


@dataclass(frozen=True)
class StaticDendriticResourceSummary:
    """Config-derived resources for a regular dendritic population.

    This summary is intentionally limited to quantities that are determined
    exactly by morphology, pathway input widths, and retained contacts.  It
    does not infer trainable parameter counts for a particular sparse-layer
    implementation; checkpoint-backed analyses should use
    :func:`model_resource_summary` for those quantities.
    """

    level_compartments_per_soma: tuple[int, ...]
    nonsomatic_compartments_per_soma: int
    nonsomatic_compartments_total: int
    synaptic_compartments_total: int
    active_synapses_by_pathway: dict[str, int]
    active_synapses_total: int
    candidate_slots_by_pathway: dict[str, int]
    candidate_synapse_slots_total: int
    child_couplings_total: int
    reactivation_parameters_total: int


def _resource_integer(value: Any, *, name: str, allow_zero: bool) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer, not bool")
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        raise TypeError(f"{name} must be an integer, got {value!r}") from None
    if parsed != value:
        raise ValueError(f"{name} must be an exact integer, got {value!r}")
    minimum = 0 if allow_zero else 1
    if parsed < minimum:
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be {qualifier}, got {parsed}")
    return parsed


def dendritic_level_widths(branch_factors: Sequence[int]) -> tuple[int, ...]:
    """Return nonsomatic compartment counts per soma from proximal to distal."""

    if isinstance(branch_factors, (str, bytes)):
        raise TypeError("branch_factors must be a sequence of positive integers")
    factors = tuple(
        _resource_integer(value, name=f"branch_factors[{index}]", allow_zero=False)
        for index, value in enumerate(branch_factors)
    )
    if not factors:
        raise ValueError("branch_factors must contain at least one dendritic level")
    level_widths = []
    width = 1
    for factor in factors:
        width *= factor
        level_widths.append(width)
    return tuple(level_widths)


def static_dendritic_resource_summary(
    *,
    branch_factors: Sequence[int],
    n_somas: int,
    pathway_input_dims: Mapping[str, int],
    synapses_per_compartment: Mapping[str, int],
    somatic_synapses: bool = False,
    reactivation_scalars_per_compartment: int = 2,
    reactivation_includes_soma: bool = True,
) -> StaticDendriticResourceSummary:
    """Compute exact config-level resources for one regular population.

    ``candidate_slots`` counts structurally available input/output pairs, not
    necessarily trainable parameters.  For standard dense-candidate Top-K the
    two counts coincide; indexed and structured implementations can differ.
    """

    level_widths = dendritic_level_widths(branch_factors)
    soma_count = _resource_integer(n_somas, name="n_somas", allow_zero=False)
    reactivation_width = _resource_integer(
        reactivation_scalars_per_compartment,
        name="reactivation_scalars_per_compartment",
        allow_zero=True,
    )

    input_dims = {
        str(pathway): _resource_integer(
            value,
            name=f"pathway_input_dims[{pathway!r}]",
            allow_zero=True,
        )
        for pathway, value in pathway_input_dims.items()
    }
    retained = {
        str(pathway): _resource_integer(
            value,
            name=f"synapses_per_compartment[{pathway!r}]",
            allow_zero=True,
        )
        for pathway, value in synapses_per_compartment.items()
    }
    if not input_dims:
        raise ValueError("At least one synaptic pathway must be specified")
    if set(input_dims) != set(retained):
        raise ValueError(
            "pathway_input_dims and synapses_per_compartment must have identical keys"
        )
    for pathway in sorted(input_dims):
        if retained[pathway] > input_dims[pathway]:
            raise ValueError(
                f"Retained contacts for {pathway!r} exceed its input width: "
                f"{retained[pathway]} > {input_dims[pathway]}"
            )

    nonsomatic_per_soma = sum(level_widths)
    nonsomatic_total = nonsomatic_per_soma * soma_count
    synaptic_compartments = nonsomatic_total + (
        soma_count if bool(somatic_synapses) else 0
    )
    active_by_pathway = {
        pathway: synaptic_compartments * count for pathway, count in retained.items()
    }
    candidate_by_pathway = {
        pathway: synaptic_compartments * input_dims[pathway] for pathway in input_dims
    }
    reactivation_compartments = nonsomatic_total + (
        soma_count if bool(reactivation_includes_soma) else 0
    )
    return StaticDendriticResourceSummary(
        level_compartments_per_soma=level_widths,
        nonsomatic_compartments_per_soma=nonsomatic_per_soma,
        nonsomatic_compartments_total=nonsomatic_total,
        synaptic_compartments_total=synaptic_compartments,
        active_synapses_by_pathway=active_by_pathway,
        active_synapses_total=sum(active_by_pathway.values()),
        candidate_slots_by_pathway=candidate_by_pathway,
        candidate_synapse_slots_total=sum(candidate_by_pathway.values()),
        child_couplings_total=nonsomatic_total,
        reactivation_parameters_total=(reactivation_compartments * reactivation_width),
    )


def synapse_resource_records(model: torch.nn.Module) -> list[SynapseResourceRecord]:
    """Return one realized-connectivity record per dendritic pathway."""

    records: list[SynapseResourceRecord] = []
    seen: set[int] = set()
    for branch_name, branch_layer in model.named_modules():
        if not isinstance(branch_layer, DendriticBranchLayer):
            continue
        depth = soma_relative_dendritic_depth(branch_layer)
        for attr, pathway in _SYNAPSE_PATHWAYS.items():
            synapse = getattr(branch_layer, attr, None)
            if synapse is None or id(synapse) in seen:
                continue
            seen.add(id(synapse))
            if type(synapse) in {TopKLinear, DenseToSparseLinear}:
                counts = _dense_candidate_topk_resource_counts(synapse)
                out_features = int(counts["out_features"])
                in_features = int(counts["in_features"])
                candidate_slots = int(counts["candidate_slots"])
                active_synapses = int(counts["active_synapses"])
                realized_k_min = int(counts["realized_k_min"])
                realized_k_max = int(counts["realized_k_max"])
                realized_k_mean = float(counts["realized_k_mean"])
                selection_policy = str(counts["selection_policy"])
                mask_source = str(counts["mask_source"])
            elif callable(
                compact_counts_fn := getattr(
                    synapse,
                    "connectivity_resource_counts",
                    None,
                )
            ):
                counts = compact_counts_fn()
                out_features = int(counts["out_features"])
                in_features = int(counts["in_features"])
                candidate_slots = int(counts["candidate_slots"])
                active_synapses = int(counts["active_synapses"])
                realized_k_min = int(counts["realized_k_min"])
                realized_k_max = int(counts["realized_k_max"])
                realized_k_mean = float(counts["realized_k_mean"])
                selection_policy = str(counts["selection_policy"])
                mask_source = str(counts["mask_source"])
            else:
                snapshot = effective_synapse_snapshot(synapse)
                realized_k = snapshot.realized_k.to(dtype=torch.float32)
                out_features = int(snapshot.active_mask.shape[0])
                in_features = int(snapshot.active_mask.shape[1])
                candidate_slots = int(snapshot.candidate_mask.ne(0).sum().item())
                active_synapses = int(snapshot.active_mask.ne(0).sum().item())
                realized_k_min = int(realized_k.min().item())
                realized_k_max = int(realized_k.max().item())
                realized_k_mean = float(realized_k.mean().item())
                selection_policy = snapshot.selection_policy
                mask_source = snapshot.mask_source
            records.append(
                SynapseResourceRecord(
                    module=f"{branch_name}.{attr}",
                    pathway=pathway,
                    soma_relative_depth=int(depth),
                    out_features=out_features,
                    in_features=in_features,
                    candidate_parameters=int(
                        sum(parameter.numel() for parameter in synapse.parameters())
                    ),
                    candidate_slots=candidate_slots,
                    active_synapses=active_synapses,
                    realized_k_min=realized_k_min,
                    realized_k_max=realized_k_max,
                    realized_k_mean=realized_k_mean,
                    selection_policy=selection_policy,
                    mask_source=mask_source,
                )
            )
    return records


def _dense_candidate_topk_resource_counts(
    synapse: TopKLinear,
) -> dict[str, int | float | str]:
    """Count deterministic dense-candidate Top-K support without a dense mask."""

    if type(synapse) is DenseToSparseLinear:
        selected_k = int(synapse.current_k)
        selection_policy = "dense_to_sparse"
    elif type(synapse) is TopKLinear:
        selected_k = int(synapse.K)
        selection_policy = "topk"
    else:  # pragma: no cover - guarded by the caller
        raise TypeError(type(synapse))

    allowed_mask = synapse.connection_mask
    forbidden = synapse._forbidden_input_index_per_output
    if allowed_mask.numel() > 0:
        allowed_per_output = allowed_mask.count_nonzero(dim=1).to(torch.long)
        if forbidden.numel() > 0:
            valid = forbidden >= 0
            if bool(valid.any()):
                rows = torch.arange(synapse.out_features, device=forbidden.device)
                forbidden_allowed = allowed_mask[
                    rows[valid],
                    forbidden[valid].to(torch.long),
                ].to(torch.long)
                allowed_per_output[valid] -= forbidden_allowed
        mask_source = "structured_candidate_mask"
    else:
        allowed_per_output = torch.full(
            (synapse.out_features,),
            synapse.in_features,
            dtype=torch.long,
            device=synapse.pre_w.device,
        )
        if forbidden.numel() > 0:
            allowed_per_output -= (forbidden >= 0).to(
                device=allowed_per_output.device,
                dtype=torch.long,
            )
        mask_source = "dense_candidates"
    realized = allowed_per_output.clamp(max=selected_k)
    return {
        "out_features": int(synapse.out_features),
        "in_features": int(synapse.in_features),
        "candidate_slots": int(allowed_per_output.sum().item()),
        "active_synapses": int(realized.sum().item()),
        "realized_k_min": int(realized.min().item()),
        "realized_k_max": int(realized.max().item()),
        "realized_k_mean": float(realized.to(torch.float64).mean().item()),
        "selection_policy": selection_policy,
        "mask_source": mask_source,
    }


def _tensor_tree_numel(value: Any, seen: set[int]) -> int:
    if isinstance(value, torch.Tensor):
        if id(value) in seen:
            return 0
        seen.add(id(value))
        return int(value.numel())
    if is_dataclass(value) and not isinstance(value, type):
        return sum(
            _tensor_tree_numel(getattr(value, field.name), seen)
            for field in fields(value)
        )
    if isinstance(value, dict):
        return sum(_tensor_tree_numel(item, seen) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_tensor_tree_numel(item, seen) for item in value)
    return 0


def persistent_state_scalars(core: torch.nn.Module) -> int:
    """Count recurrent state scalars retained per sample between timesteps."""

    init_state = getattr(core, "init_state", None)
    if callable(init_state):
        state = init_state(
            batch_size=1,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        return _tensor_tree_numel(state, set())

    config = getattr(core, "config", None)
    cell_type = str(getattr(config, "cell_type", "")).lower()
    hidden_dim = int(getattr(config, "hidden_dim", 0) or 0)
    num_layers = int(getattr(config, "num_layers", 1) or 1)
    if cell_type in {"vanilla", "gru"}:
        return hidden_dim * num_layers
    if cell_type == "lstm":
        return 2 * hidden_dim * num_layers
    return 0


def model_resource_summary(model: torch.nn.Module) -> dict[str, Any]:
    """Return auditable parameter, connectivity, and state accounting."""

    core = getattr(model, "core_network", model)
    records = synapse_resource_records(model)
    record_provider = getattr(core, "connectivity_resource_records", None)
    if callable(record_provider):
        provided = record_provider()
        if not isinstance(provided, list):
            raise TypeError("connectivity_resource_records() must return a list")
        records.extend(
            (
                record
                if isinstance(record, SynapseResourceRecord)
                else SynapseResourceRecord(**record)
            )
            for record in provided
        )
    total_parameters = int(sum(parameter.numel() for parameter in model.parameters()))
    trainable_parameters = int(
        sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        )
    )
    core_total_parameters = int(
        sum(parameter.numel() for parameter in core.parameters())
    )
    core_trainable_parameters = int(
        sum(
            parameter.numel()
            for parameter in core.parameters()
            if parameter.requires_grad
        )
    )
    synapse_candidate_parameters = int(
        sum(record.candidate_parameters for record in records)
    )
    summary = {
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
        "core_total_parameters": core_total_parameters,
        "core_trainable_parameters": core_trainable_parameters,
        "synapse_candidate_parameters": synapse_candidate_parameters,
        "non_synaptic_parameters": total_parameters - synapse_candidate_parameters,
        "candidate_synapse_slots": int(
            sum(record.candidate_slots for record in records)
        ),
        "active_synapses": int(sum(record.active_synapses for record in records)),
        "persistent_state_scalars_per_sample": persistent_state_scalars(core),
        "synapses": [asdict(record) for record in records],
    }
    ledger_provider = getattr(core, "resource_ledger", None)
    if callable(ledger_provider):
        summary["mechanism_ledger"] = ledger_provider()
    return summary


__all__ = [
    "StaticDendriticResourceSummary",
    "SynapseResourceRecord",
    "dendritic_level_widths",
    "model_resource_summary",
    "persistent_state_scalars",
    "static_dendritic_resource_summary",
    "synapse_resource_records",
]
