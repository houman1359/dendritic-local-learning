"""Replacement builders for transformer distillation units."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
import torch.nn as nn

from dendritic_modeling.config import Config
from dendritic_modeling.networks.architectures.transformer import (
    DendriticFFNReplacement,
    DenseSwiGLUSurrogate,
    EIStackMLPSlot,
    GatedDendriticFFNReplacement,
    GatedPopulationNetworkFFNReplacement,
    LayerwiseEIStackReplacement,
    PopulationNetworkFFNReplacement,
    TiedPopulationNetworkFFNSite,
    build_dendritic_ffn_kwargs_from_core_config,
    build_ei_stack_kwargs_from_core_config,
    build_population_network_ffn_kwargs_from_core_config,
    build_transformer_replacement_for_mlp,
    unwrap_shared_population_replacement,
    validate_parameter_tied_replacement_groups,
)
from dendritic_modeling.networks.architectures.transformer.dense_surrogate import (
    dense_swiglu_options,
)
from dendritic_modeling.networks.architectures.transformer.patching import (
    _parameter_tied_replacement_signature,
)
from dendritic_modeling.training._transformer_replacement.common import (
    DistillationUnit,
    SyntheticTransformerMLP,
    _to_plain_mapping,
    resolve_configured_replacement_layers,
)


def _replacement_kind(config: Config) -> str:
    replacement_kwargs = _to_plain_mapping(
        config.model.transformer_replacement.replacement_kwargs
    )
    return str(
        replacement_kwargs.get("kind", replacement_kwargs.get("type", "dendritic_ffn"))
    ).lower()


def _make_single_layer_ei_stack_replacement(
    *,
    hidden_size: int,
    config: Config,
    dtype: torch.dtype,
    device: torch.device,
) -> EIStackMLPSlot:
    kwargs = build_ei_stack_kwargs_from_core_config(
        config.model.core,
        transformer_replacement=config.model.transformer_replacement,
        num_layers=1,
    )
    stack = LayerwiseEIStackReplacement(hidden_size=hidden_size, **kwargs)
    stack = stack.to(device=device, dtype=dtype)
    return EIStackMLPSlot(stack, 0)


def _make_replacement(
    *,
    hidden_size: int,
    config: Config,
    dtype: torch.dtype,
    device: torch.device,
) -> nn.Module:
    kind = _replacement_kind(config)
    if kind == "dense_swiglu_surrogate":
        model_cfg = config.model.transformer_replacement
        if (
            _to_plain_mapping(model_cfg.selection).get("enabled", False)
            or model_cfg.compiled_plan
            or model_cfg.compiled_plans_by_layer
        ):
            raise ValueError(
                "Dense SwiGLU surrogate cannot inherit sparse selection or compiler plans"
            )
        options = dense_swiglu_options(_to_plain_mapping(model_cfg.replacement_kwargs))
        return DenseSwiGLUSurrogate(hidden_size=hidden_size, **options).to(
            device=device, dtype=dtype
        )
    if kind in {"dense_mlp_control", "dense_control", "dense_mlp"}:
        raise ValueError(
            "dense_mlp_control is a synthetic training baseline; construct it "
            "through _make_synthetic_units"
        )
    if kind in {"ei_stack", "layerwise_ei_stack", "einet_stack"}:
        return _make_single_layer_ei_stack_replacement(
            hidden_size=hidden_size,
            config=config,
            dtype=dtype,
            device=device,
        )

    if kind in {
        "population_network",
        "population_ffn",
        "population_network_ffn",
        "gated_population_network",
        "gated_population_ffn",
        "population_glu",
    }:
        kwargs = build_population_network_ffn_kwargs_from_core_config(
            config.model.core,
            transformer_replacement=config.model.transformer_replacement,
        )
        replacement_cls = (
            GatedPopulationNetworkFFNReplacement
            if kind
            in {
                "gated_population_network",
                "gated_population_ffn",
                "population_glu",
            }
            else PopulationNetworkFFNReplacement
        )
        replacement = replacement_cls(hidden_size=hidden_size, **kwargs)
        compiled_plan = _to_plain_mapping(
            config.model.transformer_replacement.compiled_plan
        )
        if compiled_plan:
            replacement.compiled_replacement_plan = compiled_plan
        return replacement.to(device=device, dtype=dtype)

    kwargs = build_dendritic_ffn_kwargs_from_core_config(
        config.model.core,
        transformer_replacement=config.model.transformer_replacement,
    )
    replacement_cls = (
        GatedDendriticFFNReplacement
        if kind in {"gated_dendritic_ffn", "gated_dendritic", "dendritic_glu"}
        else DendriticFFNReplacement
    )
    replacement = replacement_cls(hidden_size=hidden_size, **kwargs)
    return replacement.to(device=device, dtype=dtype)


def _make_synthetic_units(
    config: Config,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> list[DistillationUnit]:
    train_cfg = config.training.transformer_replacement
    teacher_cfg = _to_plain_mapping(train_cfg.synthetic_teacher)
    hidden_size = int(train_cfg.hidden_size or 32)
    intermediate_size = int(train_cfg.intermediate_size or hidden_size * 4)
    layer_indices = resolve_configured_replacement_layers(config, default=[0])

    units: list[DistillationUnit] = []
    for layer_index in layer_indices:
        planted_options = {
            "support_fraction": float(teacher_cfg.get("support_fraction", 0.25)),
            "inhibitory_fraction": float(teacher_cfg.get("inhibitory_fraction", 0.25)),
            "output_rank": teacher_cfg.get("output_rank"),
            "branch_factors": teacher_cfg.get("branch_factors", (2, 2)),
            "weight_scale": float(teacher_cfg.get("weight_scale", 1.0)),
            "seed": int(teacher_cfg.get("seed", train_cfg.seed or 0))
            + int(layer_index),
        }
        teacher = SyntheticTransformerMLP(
            hidden_size,
            intermediate_size,
            kind=str(teacher_cfg.get("kind", "gated_mlp")),
            activation=str(teacher_cfg.get("activation", "silu")),
            bias=bool(teacher_cfg.get("bias", False)),
            **planted_options,
        ).to(device=device, dtype=dtype)
        teacher.eval()
        for param in teacher.parameters():
            param.requires_grad_(False)

        if _replacement_kind(config) in {
            "dense_mlp_control",
            "dense_control",
            "dense_mlp",
        }:
            replacement = SyntheticTransformerMLP(
                hidden_size,
                intermediate_size,
                kind=str(teacher_cfg.get("kind", "gated_mlp")),
                activation=str(teacher_cfg.get("activation", "silu")),
                bias=bool(teacher_cfg.get("bias", False)),
            ).to(device=device, dtype=dtype)
        else:
            selection = _to_plain_mapping(
                config.model.transformer_replacement.selection
            )
            plans_by_layer = _to_plain_mapping(
                config.model.transformer_replacement.compiled_plans_by_layer
            )
            if bool(selection.get("enabled", False)) or plans_by_layer:
                replacement = build_transformer_replacement_for_mlp(
                    teacher,
                    config.model.transformer_replacement,
                    config.model.core,
                    layer_index=int(layer_index),
                ).to(device=device, dtype=dtype)
            else:
                replacement = _make_replacement(
                    hidden_size=hidden_size,
                    config=config,
                    dtype=dtype,
                    device=device,
                )
            if (
                isinstance(replacement, GatedDendriticFFNReplacement)
                and replacement.teacher_topk_init
            ):
                replacement.initialize_from_mlp(teacher)
        units.append(
            DistillationUnit(
                layer_index=int(layer_index),
                teacher_mlp=teacher,
                replacement=replacement,
            )
        )
    return _apply_parameter_tied_population_groups_to_units(config, units)


def _apply_parameter_tied_population_groups_to_units(
    config: Config,
    units: list[DistillationUnit],
) -> list[DistillationUnit]:
    """Tie standalone layerwise units using the model placement contract."""

    raw_groups = getattr(
        config.model.transformer_replacement,
        "parameter_tied_replacement_groups",
        [],
    )
    groups = validate_parameter_tied_replacement_groups(
        raw_groups, [unit.layer_index for unit in units]
    )
    if not groups:
        return units
    if bool(config.model.transformer_replacement.selection.get("enabled", False)):
        raise ValueError(
            "runtime FMI selection cannot be combined with parameter-tied groups; "
            "freeze identical compiled_plans_by_layer first"
        )
    by_layer = {int(unit.layer_index): unit for unit in units}
    for group_index, group in enumerate(groups):
        members = [by_layer[layer] for layer in group]
        signatures = [
            _parameter_tied_replacement_signature(unit.replacement) for unit in members
        ]
        if any(signature != signatures[0] for signature in signatures[1:]):
            raise ValueError(
                "parameter-tied replacement group has incompatible compiled plans, "
                f"devices, or boundary shapes: {list(group)}"
            )
        shared_core = unwrap_shared_population_replacement(members[0].replacement)
        for unit in members:
            unit.replacement = TiedPopulationNetworkFFNSite(
                shared_core,
                layer_index=int(unit.layer_index),
                group_index=group_index,
                group_layers=group,
            )
            unit.tied_group_index = int(group_index)
            unit.tied_group_leader = int(group[0])
            unit.tied_group_layers = tuple(group)
            unit.is_tied_alias = int(unit.layer_index) != int(group[0])
    return units


def _records_to_units(records: Sequence[Any]) -> list[DistillationUnit]:
    return [
        DistillationUnit(
            layer_index=int(record.layer_index),
            replacement=record.replacement,
            teacher_mlp=None,
            tied_group_index=getattr(record, "tied_group_index", None),
            tied_group_leader=getattr(record, "tied_group_leader", None),
            tied_group_layers=tuple(getattr(record, "tied_group_layers", ())),
            is_tied_alias=bool(getattr(record, "is_tied_alias", False)),
            collapsed_span_index=getattr(record, "collapsed_span_index", None),
            collapsed_span_layers=tuple(getattr(record, "collapsed_span_layers", ())),
            collapsed_span_original_mlps=tuple(
                getattr(record, "collapsed_span_original_mlps", ())
            ),
            collapsed_span_post_mlp_norm_attr=getattr(
                record, "collapsed_span_post_mlp_norm_attr", ""
            ),
            collapsed_span_original_post_mlp_norms=tuple(
                getattr(record, "collapsed_span_original_post_mlp_norms", ())
            ),
        )
        for record in records
    ]


def _offload_collapsed_span_original_mlps_(records: Sequence[Any]) -> int:
    """Move detached dense span FFNs and removed norms off accelerator memory.

    Collapsed-span records retain the removed dense modules solely for exact
    storage accounting and restoration.  The joint path has a separate dense
    teacher, so keeping these detached student modules on the accelerator
    silently consumes the memory that the collapse is intended to save.
    """

    moved = 0
    seen: set[int] = set()
    for record in records:
        originals = tuple(getattr(record, "collapsed_span_original_mlps", ()) or ())
        originals += tuple(
            getattr(record, "collapsed_span_original_post_mlp_norms", ()) or ()
        )
        for module in originals:
            if id(module) in seen:
                continue
            seen.add(id(module))
            module.to(device=torch.device("cpu"))
            moved += 1
    if moved and torch.cuda.is_available():
        torch.cuda.empty_cache()
    return moved
