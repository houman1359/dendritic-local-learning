"""Patching utilities for replacing transformer MLP modules."""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise
from numbers import Integral
from typing import Any

import torch
import torch.nn as nn

from dendritic_modeling.networks.architectures.replacement.cells import (
    require_runtime_tensor_contract,
)
from dendritic_modeling.networks.architectures.replacement.compiler import (
    CompiledReplacementPlan,
    compiled_replacement_plan_from_mapping,
)
from dendritic_modeling.networks.architectures.transformer.config_translation import (
    build_dendritic_ffn_kwargs_from_core_config,
    build_ei_stack_kwargs_from_core_config,
    build_population_network_ffn_kwargs_from_core_config,
)
from dendritic_modeling.networks.architectures.transformer.dense_surrogate import (
    DenseSwiGLUSurrogate,
    dense_swiglu_options,
)
from dendritic_modeling.networks.architectures.transformer.modules import (
    CollapsedPopulationNetworkSpanExit,
    DendriticFFNReplacement,
    DenseMLPControl,
    EIStackMLPSlot,
    GatedDendriticFFNReplacement,
    GatedPopulationNetworkFFNReplacement,
    LayerwiseEIStackReplacement,
    PopulationNetworkFFNReplacement,
    TiedPopulationNetworkFFNSite,
    ZeroFFNResidualBranch,
    unwrap_shared_population_replacement,
)
from dendritic_modeling.networks.architectures.transformer.utils import (
    _first_parameter,
    _get_attr_path,
    _infer_mlp_dims,
    _set_attr_path,
    _to_plain_mapping,
)

logger = logging.getLogger(__name__)


def resolve_transformer_layers(
    model: nn.Module,
    *,
    layers_attr: str | None = None,
) -> nn.ModuleList | list[nn.Module]:
    """Resolve a decoder layer list from common Hugging Face model layouts.

    DeepSeek/Llama-style causal language models usually expose decoder layers
    as ``model.model.layers``.  Passing ``layers_attr`` makes the lookup
    explicit for custom wrappers.
    """
    if layers_attr is not None:
        layers = _get_attr_path(model, layers_attr)
        if not isinstance(layers, (nn.ModuleList, list)):
            raise TypeError(f"{layers_attr!r} does not resolve to a layer list")
        return layers

    candidates = (
        "model.layers",
        "model.model.layers",
        "layers",
        "transformer.h",
        "gpt_neox.layers",
        # Vision transformers (ViT/DeiT-style encoders).
        "vit.encoder.layer",
        "encoder.layer",
    )
    for path in candidates:
        try:
            layers = _get_attr_path(model, path)
        except AttributeError:
            continue
        if isinstance(layers, (nn.ModuleList, list)):
            return layers
    raise ValueError(
        "Could not find transformer decoder layers. Pass layers_attr explicitly."
    )


class _ViTResidualOutput(nn.Module):
    """Replaces ``ViTOutput`` when the FFN is replaced as a whole.

    HF ViT splits the FFN across ``intermediate`` (dense+act) and ``output``
    (dense+dropout) with the residual add INSIDE ``ViTOutput.forward``. When
    the replacement cell computes the entire FFN in ``intermediate``'s slot,
    this module keeps the teacher's dropout and residual semantics:
    ``forward(hidden_states, input_tensor) = dropout(hidden_states) +
    input_tensor``.
    """

    def __init__(self, dropout_p: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout_p)

    def forward(
        self, hidden_states: torch.Tensor, input_tensor: torch.Tensor
    ) -> torch.Tensor:
        return self.dropout(hidden_states) + input_tensor


@dataclass
class ReplacementRecord:
    """Metadata for one patched transformer target module."""

    layer_index: int
    mlp_attr: str
    original_mlp: nn.Module
    replacement: nn.Module
    tied_group_index: int | None = None
    tied_group_leader: int | None = None
    tied_group_layers: tuple[int, ...] = ()
    is_tied_alias: bool = False
    collapsed_span_index: int | None = None
    collapsed_span_layers: tuple[int, ...] = ()
    collapsed_span_original_mlps: tuple[nn.Module, ...] = ()
    collapsed_span_installed_modules: tuple[nn.Module, ...] = ()
    collapsed_span_post_mlp_norm_attr: str = ""
    collapsed_span_original_post_mlp_norms: tuple[nn.Module, ...] = ()
    collapsed_span_installed_post_mlp_norms: tuple[nn.Module, ...] = ()


def validate_parameter_tied_replacement_groups(
    groups: Any,
    layer_indices: Sequence[int],
) -> list[tuple[int, ...]]:
    """Normalize strict, non-overlapping contiguous parameter-tying groups."""

    if groups is None:
        return []
    if isinstance(groups, (str, bytes)) or not isinstance(groups, Sequence):
        raise TypeError("parameter_tied_replacement_groups must be a sequence")
    declared = {int(index) for index in layer_indices}
    normalized: list[tuple[int, ...]] = []
    seen: set[int] = set()
    for group_index, raw_group in enumerate(groups):
        if isinstance(raw_group, (str, bytes)) or not isinstance(raw_group, Sequence):
            raise TypeError(
                f"parameter_tied_replacement_groups[{group_index}] must be a sequence"
            )
        if any(
            isinstance(value, bool) or not isinstance(value, Integral)
            for value in raw_group
        ):
            raise TypeError("parameter-tied replacement indices must be integers")
        group = tuple(int(value) for value in raw_group)
        if len(group) < 2:
            raise ValueError(
                "each parameter-tied replacement group must contain at least two layers"
            )
        if list(group) != sorted(group) or len(group) != len(set(group)):
            raise ValueError(
                "each parameter-tied replacement group must be strictly increasing"
            )
        if any(right != left + 1 for left, right in pairwise(group)):
            raise ValueError("each parameter-tied replacement group must be contiguous")
        missing = sorted(set(group) - declared)
        if missing:
            raise ValueError(
                "parameter-tied replacement group members must also be declared "
                "in layers; "
                f"missing {missing}"
            )
        overlap = sorted(set(group) & seen)
        if overlap:
            raise ValueError(
                "parameter-tied replacement groups must not overlap; "
                f"repeated {overlap}"
            )
        seen.update(group)
        normalized.append(group)
    return normalized


def validate_collapsed_replacement_spans(
    spans: Any,
    layer_indices: Sequence[int],
) -> list[tuple[int, ...]]:
    """Validate true one-application FFN-collapse spans."""

    if spans is None:
        return []
    if isinstance(spans, (str, bytes)) or not isinstance(spans, Sequence):
        raise TypeError("collapsed_replacement_spans must be a sequence")
    if not spans:
        return []
    if any(
        isinstance(index, bool) or not isinstance(index, Integral)
        for index in layer_indices
    ):
        raise TypeError("collapsed replacement layers must be integers")
    declared_order = [int(index) for index in layer_indices]
    if (
        not declared_order
        or any(index < 0 for index in declared_order)
        or declared_order != sorted(declared_order)
        or len(declared_order) != len(set(declared_order))
    ):
        raise ValueError(
            "collapsed replacement layers must be unique sorted nonnegative indices"
        )
    declared = set(declared_order)
    normalized: list[tuple[int, ...]] = []
    seen: set[int] = set()
    for span_index, raw_span in enumerate(spans):
        if isinstance(raw_span, (str, bytes)) or not isinstance(raw_span, Sequence):
            raise TypeError(
                f"collapsed_replacement_spans[{span_index}] must be a sequence"
            )
        if any(
            isinstance(value, bool) or not isinstance(value, Integral)
            for value in raw_span
        ):
            raise TypeError("collapsed replacement span indices must be integers")
        span = tuple(int(value) for value in raw_span)
        if len(span) < 2:
            raise ValueError(
                "each collapsed replacement span must contain at least two layers"
            )
        if list(span) != sorted(span) or len(span) != len(set(span)):
            raise ValueError(
                "each collapsed replacement span must be strictly increasing"
            )
        if any(right != left + 1 for left, right in pairwise(span)):
            raise ValueError("each collapsed replacement span must be contiguous")
        missing = sorted(set(span) - declared)
        if missing:
            raise ValueError(
                "collapsed replacement span members must also be declared in layers; "
                f"missing {missing}"
            )
        overlap = sorted(set(span) & seen)
        if overlap:
            raise ValueError(
                f"collapsed replacement spans must not overlap; repeated {overlap}"
            )
        seen.update(span)
        normalized.append(span)
    if [span[0] for span in normalized] != sorted(span[0] for span in normalized):
        raise ValueError("collapsed replacement spans must be ordered by depth")
    uncovered = sorted(declared - seen)
    if normalized and uncovered:
        raise ValueError(
            "collapsed replacement mode requires every declared layer to belong "
            f"to exactly one collapsed span; uncovered {uncovered}"
        )
    return normalized


def _parameter_tied_replacement_signature(module: nn.Module) -> tuple[Any, ...]:
    """Return the executable structure that must agree before parameters tie."""

    core = unwrap_shared_population_replacement(module)
    if not isinstance(
        core,
        (PopulationNetworkFFNReplacement, GatedPopulationNetworkFFNReplacement),
    ):
        raise TypeError(
            "parameter-tied replacement groups require PopulationNetwork FFN "
            "replacements"
        )
    state_signature = tuple(
        (name, tuple(value.shape), str(value.dtype), str(value.device))
        for name, value in core.state_dict().items()
    )
    plan = getattr(core, "compiled_replacement_plan", {}) or {}
    plan_json = json.dumps(plan, sort_keys=True, separators=(",", ":"))
    population_json = json.dumps(
        getattr(core, "population_network_config", {}) or {},
        sort_keys=True,
        separators=(",", ":"),
    )
    return (
        type(core),
        int(getattr(core, "hidden_size", -1)),
        int(getattr(core, "output_size", getattr(core, "output_dim", -1))),
        getattr(core, "reference_intermediate_size", None),
        str(getattr(core, "input_transform", "")),
        plan_json,
        population_json,
        state_signature,
    )


def apply_parameter_tied_population_groups_to_records(
    model: nn.Module,
    records: list[ReplacementRecord],
    groups: Sequence[Sequence[int]],
    *,
    layers_attr: str | None = None,
) -> list[ReplacementRecord]:
    """Install unique site wrappers around one physical core per group.

    All compatibility checks complete before any wrapper is installed.  This
    helper is public so reconstruction tools can apply the exact same alias
    graph after constructing ordinary PopulationNetwork records.
    """

    normalized = validate_parameter_tied_replacement_groups(
        groups, [record.layer_index for record in records]
    )
    if not normalized:
        return records
    try:
        by_layer = {int(record.layer_index): record for record in records}
        prepared: list[tuple[int, tuple[int, ...], nn.Module]] = []
        for group_index, group in enumerate(normalized):
            group_records = [by_layer[layer] for layer in group]
            signatures = [
                _parameter_tied_replacement_signature(record.replacement)
                for record in group_records
            ]
            if any(signature != signatures[0] for signature in signatures[1:]):
                raise ValueError(
                    "parameter-tied replacement group has incompatible compiled "
                    f"plans, devices, or boundary shapes: {list(group)}"
                )
            prepared.append(
                (
                    group_index,
                    group,
                    unwrap_shared_population_replacement(group_records[0].replacement),
                )
            )

        layers = resolve_transformer_layers(model, layers_attr=layers_attr)
        for group_index, group, shared_core in prepared:
            for layer_index in group:
                record = by_layer[layer_index]
                wrapper = TiedPopulationNetworkFFNSite(
                    shared_core,
                    layer_index=layer_index,
                    group_index=group_index,
                    group_layers=group,
                )
                layer = layers[layer_index]
                _set_attr_path(layer, record.mlp_attr, wrapper)
                record.replacement = wrapper
                record.tied_group_index = int(group_index)
                record.tied_group_leader = int(group[0])
                record.tied_group_layers = tuple(group)
                record.is_tied_alias = layer_index != group[0]
    except Exception:
        restore_transformer_mlp_layers(model, records, layers_attr=layers_attr)
        raise
    return records


def build_selected_transformer_replacement_from_dims(
    *,
    hidden_size: int,
    teacher_intermediate_size: int,
    transformer_replacement: Any,
    layer_index: int | None = None,
    output_size: int | None = None,
    replacement_kwargs: dict[str, Any] | None = None,
) -> nn.Module:
    """Build an FMI-selected cell when only cached boundary shapes exist."""

    from dendritic_modeling.networks.architectures.replacement.selection import (
        resolve_replacement_selection,
    )

    tr = _to_plain_mapping(transformer_replacement)
    selection = _to_plain_mapping(tr.get("selection", {}))
    selected = resolve_replacement_selection(
        selection,
        hidden_size=int(hidden_size),
        teacher_intermediate_size=int(teacher_intermediate_size),
        layer_index=layer_index,
    )
    return build_compiled_population_replacement(
        selected,
        hidden_size=int(hidden_size),
        teacher_intermediate_size=int(teacher_intermediate_size),
        output_size=output_size,
        transformer_replacement=tr,
        replacement_kwargs=replacement_kwargs,
    )


def build_compiled_population_replacement(
    selected: Any,
    *,
    hidden_size: int,
    teacher_intermediate_size: int,
    output_size: int | None = None,
    transformer_replacement: Any = None,
    replacement_kwargs: dict[str, Any] | None = None,
) -> nn.Module:
    """Instantiate one resolved plan at a vector-valued model boundary.

    Transformer residual FFNs keep ``output_size == hidden_size``. Classical
    and vision spans may expose a different suffix dimension. Both routes use
    the same gated/single-path PopulationNetwork and compiled sparse readout.
    """

    return build_population_replacement_from_compiled_plan(
        selected.plan,
        hidden_size=hidden_size,
        teacher_intermediate_size=teacher_intermediate_size,
        output_size=output_size,
        transformer_replacement=transformer_replacement,
        replacement_kwargs=replacement_kwargs,
        selection_manifest=selected.manifest,
    )


def build_population_replacement_from_compiled_plan(
    plan: CompiledReplacementPlan,
    *,
    hidden_size: int,
    teacher_intermediate_size: int,
    output_size: int | None = None,
    transformer_replacement: Any = None,
    replacement_kwargs: dict[str, Any] | None = None,
    selection_manifest: dict[str, Any] | None = None,
) -> nn.Module:
    """Instantiate one immutable compiled plan at a model boundary."""

    if int(plan.hidden_size) != int(hidden_size):
        raise ValueError(
            "Compiled replacement hidden size does not match its target: "
            f"{plan.hidden_size} != {hidden_size}"
        )
    tr = _to_plain_mapping(transformer_replacement)
    selection = _to_plain_mapping(tr.get("selection", {}))
    selected_tr = dict(tr)
    selected_tr["input_transform"] = plan.replacement_kwargs["input_transform"]
    selected_tr["replacement_kwargs"] = dict(plan.replacement_kwargs)
    runtime_overrides = _to_plain_mapping(selection.get("runtime_overrides", {}))
    runtime_overrides.update(_to_plain_mapping(tr.get("runtime_overrides", {})))
    if replacement_kwargs:
        runtime_overrides.update(replacement_kwargs)
    kind = str(plan.replacement_kind).lower()
    kwargs = build_population_network_ffn_kwargs_from_core_config(
        plan.core_config,
        transformer_replacement=selected_tr,
        overrides=runtime_overrides,
    )
    replacement_cls = (
        GatedPopulationNetworkFFNReplacement
        if kind == "gated_population_network"
        else PopulationNetworkFFNReplacement
    )
    replacement = replacement_cls(
        hidden_size=int(hidden_size),
        output_size=output_size,
        **kwargs,
    )
    replacement.reference_intermediate_size = int(teacher_intermediate_size)
    replacement.teacher_support_metric = str(plan.teacher_support_metric or "")
    replacement.selection_manifest = dict(selection_manifest or {})
    replacement.compiled_replacement_plan = plan.as_dict()
    require_runtime_tensor_contract(
        replacement,
        boundary="compiled PopulationNetwork replacement",
    )
    return replacement


def build_transformer_replacement_for_mlp(
    mlp: nn.Module,
    transformer_replacement: Any,
    core_config: Any,
    *,
    layer_index: int | None = None,
    replacement_kwargs: dict[str, Any] | None = None,
) -> nn.Module:
    """Build one replacement through the shared manual/FMI/sweep pathway.

    When ``selection.enabled`` is set, the stored fingerprint is resolved for
    this layer and compiled into a full ``PopulationNetwork``. Otherwise this
    preserves the historical config translation. The returned module carries
    its selection manifest so checkpoints can retain scientific provenance.
    """

    tr = _to_plain_mapping(transformer_replacement)
    selection = _to_plain_mapping(tr.get("selection", {}))
    plans_by_layer = _to_plain_mapping(tr.get("compiled_plans_by_layer", {}))
    declared_options = _to_plain_mapping(tr.get("replacement_kwargs", {}))
    declared_kind = str(
        declared_options.get("kind", declared_options.get("type", ""))
    ).lower()
    if declared_kind == "dense_swiglu_surrogate":
        if (
            bool(selection.get("enabled", False))
            or plans_by_layer
            or tr.get("compiled_plan")
        ):
            raise ValueError(
                "Dense SwiGLU surrogate cannot inherit sparse selection or compiler plans"
            )
        options = {**declared_options, **dict(replacement_kwargs or {})}
        return DenseSwiGLUSurrogate.from_mlp(mlp, **dense_swiglu_options(options))
    if bool(selection.get("enabled", False)) and plans_by_layer:
        raise ValueError(
            "Runtime FMI selection and compiled_plans_by_layer are mutually exclusive"
        )
    if plans_by_layer:
        if layer_index is None:
            raise ValueError("compiled_plans_by_layer requires a target layer index")
        raw_plan = plans_by_layer.get(str(layer_index), plans_by_layer.get(layer_index))
        if raw_plan is None:
            raise KeyError(
                f"No compiled replacement plan exists for layer {layer_index}"
            )
        plan = compiled_replacement_plan_from_mapping(_to_plain_mapping(raw_plan))
        canonical = json.dumps(
            plan.as_dict(), sort_keys=True, separators=(",", ":")
        ).encode()
        hidden_size, intermediate_size = _infer_mlp_dims(mlp)
        if intermediate_size is None:
            raise ValueError(
                "A compiled per-layer plan requires a recognizable teacher FFN "
                "intermediate size"
            )
        return build_population_replacement_from_compiled_plan(
            plan,
            hidden_size=hidden_size,
            teacher_intermediate_size=intermediate_size,
            transformer_replacement=tr,
            replacement_kwargs=replacement_kwargs,
            selection_manifest={
                "schema": "dendritic_frozen_layer_plan/v1",
                "source": "compiled_plans_by_layer",
                "layer_index": int(layer_index),
                "plan_sha256": hashlib.sha256(canonical).hexdigest(),
                "status": "prospectively_frozen_before_training",
            },
        )
    if bool(selection.get("enabled", False)):
        hidden_size, intermediate_size = _infer_mlp_dims(mlp)
        if intermediate_size is None:
            raise ValueError(
                "FMI selection requires the teacher intermediate size; the "
                "target module did not expose a recognizable FFN projection"
            )
        return build_selected_transformer_replacement_from_dims(
            hidden_size=hidden_size,
            teacher_intermediate_size=intermediate_size,
            transformer_replacement=tr,
            layer_index=layer_index,
            replacement_kwargs=replacement_kwargs,
        )

    replacement_overrides = _to_plain_mapping(tr.get("replacement_kwargs", {}))
    kind = str(
        replacement_overrides.get(
            "kind", replacement_overrides.get("type", "dendritic_ffn")
        )
    ).lower()
    population_kinds = {
        "population_network",
        "population_ffn",
        "population_network_ffn",
        "gated_population_network",
        "gated_population_ffn",
        "population_glu",
    }
    if kind in population_kinds:
        kwargs = build_population_network_ffn_kwargs_from_core_config(
            core_config,
            transformer_replacement=tr,
            overrides=replacement_kwargs,
        )
        replacement_cls = (
            GatedPopulationNetworkFFNReplacement
            if kind
            in {"gated_population_network", "gated_population_ffn", "population_glu"}
            else PopulationNetworkFFNReplacement
        )
    elif kind in {"ei_stack", "layerwise_ei_stack", "einet_stack"}:
        raise ValueError(
            "A layerwise EI stack is a multi-site object and cannot be built "
            "as one MLP. Use PopulationNetwork replacements for per-layer FMI."
        )
    else:
        kwargs = build_dendritic_ffn_kwargs_from_core_config(
            core_config,
            transformer_replacement=tr,
            overrides=replacement_kwargs,
        )
        replacement_cls = (
            GatedDendriticFFNReplacement
            if kind in {"gated_dendritic_ffn", "gated_dendritic", "dendritic_glu"}
            else DendriticFFNReplacement
        )
    replacement = replacement_cls.from_mlp(mlp, **kwargs)
    compiled_plan = _to_plain_mapping(tr.get("compiled_plan", {}))
    if compiled_plan:
        replacement.compiled_replacement_plan = compiled_plan
    require_runtime_tensor_contract(
        replacement,
        boundary="transformer FFN replacement",
    )
    return replacement


def _resolve_layer_target(
    layers: nn.ModuleList | list[nn.Module],
    layer_index: int,
    mlp_attr: str,
) -> tuple[nn.Module, nn.Module]:
    """Return a transformer layer and its target module with consistent errors."""
    if layer_index < 0 or layer_index >= len(layers):
        raise IndexError(
            f"layer index {layer_index} out of range for {len(layers)} layers"
        )
    layer = layers[layer_index]
    try:
        target = _get_attr_path(layer, mlp_attr)
    except AttributeError as exc:
        raise AttributeError(
            f"layer {layer_index} has no target module path {mlp_attr!r}"
        ) from exc
    if not isinstance(target, nn.Module):
        raise TypeError(f"layer {layer_index} target {mlp_attr!r} is not an nn.Module")
    return layer, target


def _move_replacement_like_source(
    replacement: nn.Module,
    source: nn.Module,
) -> nn.Module:
    """Move replacement to the first source parameter's device and floating dtype."""
    param = _first_parameter(source)
    if param is None:
        return replacement
    dtype = param.dtype if param.dtype.is_floating_point else None
    return replacement.to(device=param.device, dtype=dtype)


def _validate_layer_indices(
    layer_indices: Sequence[int],
) -> list[int]:
    """Return ordered unique layer indices or fail before mutating a model."""
    ordered = [int(index) for index in layer_indices]
    if not ordered:
        raise ValueError("layer_indices must contain at least one layer")
    if len(ordered) != len(set(ordered)):
        raise ValueError("layer_indices must not contain duplicates")
    return ordered


def _annotate_uniform_compiled_plan(
    records: list[ReplacementRecord],
    transformer_replacement: dict[str, Any],
) -> list[ReplacementRecord]:
    """Attach one frozen sweep plan to every identically configured layer."""

    raw_plan = _to_plain_mapping(transformer_replacement.get("compiled_plan", {}))
    if not raw_plan:
        return records
    plan = compiled_replacement_plan_from_mapping(raw_plan)
    plan_mapping = plan.as_dict()
    canonical = json.dumps(plan_mapping, sort_keys=True, separators=(",", ":")).encode()
    plan_sha256 = hashlib.sha256(canonical).hexdigest()
    for record in records:
        record.replacement.compiled_replacement_plan = plan_mapping
        record.replacement.selection_manifest = {
            "schema": "dendritic_frozen_uniform_plan/v1",
            "source": "compiled_plan",
            "layer_index": int(record.layer_index),
            "plan_sha256": plan_sha256,
            "status": "prospectively_frozen_before_training",
        }
    return records


def replace_transformer_mlp_layers(
    model: nn.Module,
    layer_indices: Sequence[int],
    *,
    replacement_kwargs: dict[str, object] | None = None,
    layers_attr: str | None = None,
    mlp_attr: str = "mlp",
    preserve_device_dtype: bool = True,
    replacement_class: type[nn.Module] = DendriticFFNReplacement,
) -> list[ReplacementRecord]:
    """Replace selected transformer layer target modules with dendritic FFNs.

    This is intentionally model-object based rather than config-file based so
    it can work with Hugging Face DeepSeek/Llama-like modules after loading.
    ``mlp_attr`` may be a nested path relative to each transformer layer.
    """
    replacement_kwargs = dict(replacement_kwargs or {})
    layers = resolve_transformer_layers(model, layers_attr=layers_attr)
    ordered_indices = _validate_layer_indices(layer_indices)

    # Resolve and construct every replacement first. If any layer path, shape,
    # or constructor is invalid, the source model remains completely untouched.
    targets = [
        (*_resolve_layer_target(layers, layer_index, mlp_attr), layer_index)
        for layer_index in ordered_indices
    ]
    prepared: list[tuple[nn.Module, nn.Module, nn.Module, int]] = []
    for layer, original_mlp, layer_index in targets:
        from_mlp = getattr(replacement_class, "from_mlp", None)
        if not callable(from_mlp):
            raise TypeError("replacement_class must provide a callable from_mlp")
        replacement = from_mlp(original_mlp, **replacement_kwargs)

        if preserve_device_dtype:
            replacement = _move_replacement_like_source(replacement, original_mlp)
        require_runtime_tensor_contract(
            replacement,
            boundary=f"transformer layer {layer_index} target {mlp_attr!r}",
        )
        prepared.append((layer, original_mlp, replacement, layer_index))

    records: list[ReplacementRecord] = []
    for layer, original_mlp, replacement, layer_index in prepared:
        _set_attr_path(layer, mlp_attr, replacement)
        records.append(
            ReplacementRecord(
                layer_index=int(layer_index),
                mlp_attr=mlp_attr,
                original_mlp=original_mlp,
                replacement=replacement,
            )
        )

    return records


def replace_transformer_mlp_layers_with_ei_stack(
    model: nn.Module,
    layer_indices: Sequence[int],
    *,
    replacement_kwargs: dict[str, object] | None = None,
    layers_attr: str | None = None,
    mlp_attr: str = "mlp",
    preserve_device_dtype: bool = True,
) -> list[ReplacementRecord]:
    """Replace selected target modules with slots from one shared EI stack."""
    replacement_kwargs = dict(replacement_kwargs or {})
    layers = resolve_transformer_layers(model, layers_attr=layers_attr)
    ordered_indices = _validate_layer_indices(layer_indices)
    for layer_index in ordered_indices:
        _resolve_layer_target(layers, layer_index, mlp_attr)

    _, first_mlp = _resolve_layer_target(layers, ordered_indices[0], mlp_attr)
    hidden_size, _ = _infer_mlp_dims(first_mlp)
    stack_num_layers = int(replacement_kwargs.pop("num_layers", len(ordered_indices)))
    if stack_num_layers != len(ordered_indices):
        raise ValueError(
            "EI stack num_layers must match the number of replaced transformer "
            f"layers ({stack_num_layers} != {len(ordered_indices)})"
        )
    stack = LayerwiseEIStackReplacement(
        hidden_size=hidden_size,
        num_layers=stack_num_layers,
        **replacement_kwargs,
    )
    if preserve_device_dtype:
        stack = _move_replacement_like_source(stack, first_mlp)
    require_runtime_tensor_contract(stack, boundary="layerwise EI stack")

    records: list[ReplacementRecord] = []
    for stack_pos, layer_index in enumerate(ordered_indices):
        layer, original_mlp = _resolve_layer_target(layers, layer_index, mlp_attr)
        replacement = EIStackMLPSlot(stack, stack_pos)
        _set_attr_path(layer, mlp_attr, replacement)
        records.append(
            ReplacementRecord(
                layer_index=int(layer_index),
                mlp_attr=mlp_attr,
                original_mlp=original_mlp,
                replacement=replacement,
            )
        )
    return records


def replace_vit_ffn_layers(
    model: nn.Module,
    layer_indices: Sequence[int],
    *,
    replacement_kwargs: dict[str, object] | None = None,
    layers_attr: str | None = None,
    preserve_device_dtype: bool = True,
    replacement_class: type[nn.Module] = DendriticFFNReplacement,
) -> list[ReplacementRecord]:
    """Replace ViT-style split FFNs (``intermediate`` + ``output``) per layer.

    HF vision transformers split the FFN across two modules with the
    residual add inside ``ViTOutput.forward``, so the decoder-style
    single-attribute patcher cannot be used. Here the replacement cell takes
    ``intermediate``'s slot (computing the whole FFN, d -> d) and ``output``
    becomes :class:`_ViTResidualOutput` (teacher dropout + residual add).
    Both originals are kept in the record (as an ``nn.ModuleDict``) so
    :func:`restore_transformer_mlp_layers`-style unpatching can reinstall
    them.
    """
    replacement_kwargs = dict(replacement_kwargs or {})
    layers = resolve_transformer_layers(model, layers_attr=layers_attr)
    ordered_indices = _validate_layer_indices(layer_indices)

    prepared = []
    for layer_index in ordered_indices:
        layer = layers[layer_index]
        intermediate = getattr(layer, "intermediate", None)
        output = getattr(layer, "output", None)
        if intermediate is None or output is None:
            raise ValueError(
                f"Layer {layer_index} lacks ViT-style intermediate/output "
                "modules; use the decoder-style target_module path instead."
            )
        shim = nn.Module()
        shim.fc1 = intermediate.dense
        shim.fc2 = output.dense
        from_mlp = getattr(replacement_class, "from_mlp", None)
        if not callable(from_mlp):
            raise TypeError("replacement_class must provide a callable from_mlp")
        replacement = from_mlp(shim, **replacement_kwargs)
        if preserve_device_dtype:
            replacement = _move_replacement_like_source(replacement, intermediate)
        require_runtime_tensor_contract(
            replacement,
            boundary=f"ViT layer {layer_index} FFN replacement",
        )
        dropout_p = 0.0
        dropout = getattr(output, "dropout", None)
        if dropout is not None and hasattr(dropout, "p"):
            dropout_p = float(dropout.p)
        prepared.append(
            (layer, layer_index, intermediate, output, replacement, dropout_p)
        )

    records: list[ReplacementRecord] = []
    for layer, layer_index, intermediate, output, replacement, dropout_p in prepared:
        layer.intermediate = replacement
        layer.output = _ViTResidualOutput(dropout_p)
        records.append(
            ReplacementRecord(
                layer_index=int(layer_index),
                mlp_attr="intermediate+output",
                original_mlp=nn.ModuleDict(
                    {"intermediate": intermediate, "output": output}
                ),
                replacement=replacement,
            )
        )
        logger.info(
            "Replaced ViT FFN at layer %d (dropout_p=%.3f)", layer_index, dropout_p
        )
    return records


def _collapsed_span_key(span: Sequence[int]) -> str:
    return f"{int(span[0])}:{int(span[-1])}"


def apply_collapsed_population_spans(
    model: nn.Module,
    transformer_replacement: Any,
    core_config: Any,
    *,
    replacement_kwargs: dict[str, Any] | None = None,
) -> list[ReplacementRecord]:
    """Collapse each declared FFN span to one cell at its exit.

    Transformer blocks themselves are never removed.  Earlier FFN branches
    return exact zeros, preserving attention and residual routing. The exit
    FFN applies one canonical PopulationNetwork cell. An explicitly configured
    post-MLP norm is included in the replaced residual branch at every site.
    """

    tr = _to_plain_mapping(transformer_replacement)
    layer_indices = tr.get("layers", tr.get("layer_indices", []))
    spans = validate_collapsed_replacement_spans(
        tr.get("collapsed_replacement_spans", []), layer_indices
    )
    if not spans:
        return []
    if tr.get("parameter_tied_replacement_groups"):
        raise ValueError(
            "collapsed_replacement_spans and parameter_tied_replacement_groups "
            "are mutually exclusive"
        )
    if str(tr.get("model_family", "causal_lm")).lower() != "causal_lm":
        raise NotImplementedError(
            "collapsed replacement spans currently support causal_lm decoder "
            "blocks only"
        )
    if bool(_to_plain_mapping(tr.get("selection", {})).get("enabled", False)):
        raise ValueError(
            "runtime FMI selection is not admissible for collapsed spans; freeze "
            "a compiled_plans_by_collapsed_span prescription prospectively"
        )
    if _to_plain_mapping(tr.get("compiled_plans_by_layer", {})):
        raise ValueError(
            "compiled_plans_by_layer cannot define a collapsed span; use "
            "compiled_plans_by_collapsed_span"
        )
    if list(tr.get("pre_patched", []) or []):
        raise ValueError(
            "collapsed replacement spans cannot compose with single-site "
            "pre_patched replacements; use hash-pinned "
            "pre_patched_collapsed_spans for staged span composition"
        )
    replacement_overrides = _to_plain_mapping(tr.get("replacement_kwargs", {}))
    replacement_kind = str(
        replacement_overrides.get(
            "kind", replacement_overrides.get("type", "dendritic_ffn")
        )
    ).lower()
    population_kinds = {
        "population_network",
        "population_ffn",
        "population_network_ffn",
        "gated_population_network",
        "gated_population_ffn",
        "population_glu",
    }
    if replacement_kind not in population_kinds:
        raise ValueError(
            "collapsed replacement spans support only canonical PopulationNetwork "
            "replacement kinds"
        )

    frozen = _to_plain_mapping(tr.get("compiled_plans_by_collapsed_span", {}))
    uniform_plan = _to_plain_mapping(tr.get("compiled_plan", {}))
    if frozen and uniform_plan:
        raise ValueError(
            "compiled_plan and compiled_plans_by_collapsed_span are mutually exclusive"
        )
    if not frozen and not uniform_plan:
        raise ValueError(
            "collapsed replacement spans require a compiler-produced compiled_plan "
            "or complete compiled_plans_by_collapsed_span mapping"
        )
    selection_runtime_overrides = _to_plain_mapping(
        _to_plain_mapping(tr.get("selection", {})).get("runtime_overrides", {})
    )
    top_level_runtime_overrides = _to_plain_mapping(tr.get("runtime_overrides", {}))
    if selection_runtime_overrides or top_level_runtime_overrides or replacement_kwargs:
        raise ValueError(
            "compiled collapsed-span plans are immutable: runtime or call-time "
            "replacement overrides would make the executable cell disagree with "
            "the hash-bound compiler plan"
        )
    expected_keys = {_collapsed_span_key(span) for span in spans}
    if frozen and set(frozen) != expected_keys:
        raise ValueError(
            "compiled_plans_by_collapsed_span must cover the declared spans exactly; "
            f"expected={sorted(expected_keys)}, got={sorted(frozen)}"
        )
    layers_attr = tr.get("layers_attr")
    mlp_attr = str(tr.get("target_module", tr.get("mlp_attr", "mlp")))
    layers = resolve_transformer_layers(model, layers_attr=layers_attr)
    preserve = bool(tr.get("preserve_device_dtype", True))
    norm_attr = tr.get("collapsed_span_post_mlp_norm_attr", "")
    if not isinstance(norm_attr, str) or (
        any(not part.isidentifier() for part in norm_attr.split(".")) and norm_attr
    ):
        raise ValueError("collapsed_span_post_mlp_norm_attr must be a module path")
    if norm_attr and (
        norm_attr == mlp_attr
        or norm_attr.startswith(mlp_attr + ".")
        or mlp_attr.startswith(norm_attr + ".")
    ):
        raise ValueError("collapsed span MLP and post-MLP norm paths must be disjoint")
    prepared: list[
        tuple[
            int,
            tuple[int, ...],
            tuple[nn.Module, ...],
            tuple[nn.Module, ...],
            tuple[nn.Module, ...],
            tuple[nn.Module, ...],
        ]
    ] = []
    for span_index, span in enumerate(spans):
        originals = tuple(
            _resolve_layer_target(layers, layer_index, mlp_attr)[1]
            for layer_index in span
        )
        original_norms = (
            tuple(
                _resolve_layer_target(layers, layer_index, norm_attr)[1]
                for layer_index in span
            )
            if norm_attr
            else ()
        )
        installed_norms = tuple(nn.Identity() for _ in original_norms)
        exit_mlp = originals[-1]
        key = _collapsed_span_key(span)
        raw_plan = frozen[key] if frozen else uniform_plan
        plan = compiled_replacement_plan_from_mapping(_to_plain_mapping(raw_plan))
        hidden_size, intermediate_size = _infer_mlp_dims(exit_mlp)
        if intermediate_size is None:
            raise ValueError(
                "a compiled collapsed-span plan requires a recognizable teacher "
                "FFN intermediate size"
            )
        canonical = json.dumps(
            plan.as_dict(), sort_keys=True, separators=(",", ":")
        ).encode()
        if frozen:
            manifest_source = "compiled_plans_by_collapsed_span"
        else:
            manifest_source = "compiled_plan"
        cell = build_population_replacement_from_compiled_plan(
            plan,
            hidden_size=hidden_size,
            teacher_intermediate_size=intermediate_size,
            transformer_replacement=tr,
            replacement_kwargs=replacement_kwargs,
            selection_manifest={
                "schema": "dendritic_frozen_collapsed_span_plan/v1",
                "source": manifest_source,
                "span_key": key,
                "span_layers": list(span),
                "exit_layer": int(span[-1]),
                "plan_sha256": hashlib.sha256(canonical).hexdigest(),
                "status": "prospectively_frozen_before_training",
            },
        )
        if not isinstance(
            cell,
            (PopulationNetworkFFNReplacement, GatedPopulationNetworkFFNReplacement),
        ):
            raise TypeError("collapsed span compiler did not build a PopulationNetwork")
        if preserve:
            cell = _move_replacement_like_source(cell, exit_mlp)
        installed: list[nn.Module] = [
            ZeroFFNResidualBranch(
                layer_index=layer_index,
                span_index=span_index,
                span_layers=span,
            )
            for layer_index in span[:-1]
        ]
        installed.append(
            CollapsedPopulationNetworkSpanExit(
                cell,
                span_index=span_index,
                span_layers=span,
                post_mlp_norm_attr=norm_attr,
            )
        )
        prepared.append(
            (
                span_index,
                span,
                originals,
                tuple(installed),
                original_norms,
                installed_norms,
            )
        )

    records: list[ReplacementRecord] = []
    try:
        for (
            span_index,
            span,
            originals,
            installed,
            original_norms,
            installed_norms,
        ) in prepared:
            for layer_index, module in zip(span, installed):
                _set_attr_path(layers[layer_index], mlp_attr, module)
            for layer_index, module in zip(span, installed_norms):
                _set_attr_path(layers[layer_index], norm_attr, module)
            records.append(
                ReplacementRecord(
                    layer_index=int(span[-1]),
                    mlp_attr=mlp_attr,
                    original_mlp=originals[-1],
                    replacement=installed[-1],
                    collapsed_span_index=int(span_index),
                    collapsed_span_layers=tuple(span),
                    collapsed_span_original_mlps=originals,
                    collapsed_span_installed_modules=installed,
                    collapsed_span_post_mlp_norm_attr=norm_attr,
                    collapsed_span_original_post_mlp_norms=original_norms,
                    collapsed_span_installed_post_mlp_norms=installed_norms,
                )
            )
    except Exception:
        for (
            _span_index,
            span,
            originals,
            _installed,
            original_norms,
            _norms,
        ) in prepared:
            for layer_index, original in zip(span, originals):
                _set_attr_path(layers[layer_index], mlp_attr, original)
            for layer_index, original in zip(span, original_norms):
                _set_attr_path(layers[layer_index], norm_attr, original)
        raise
    return records


def _collapsed_span_layer_hidden(output: Any) -> torch.Tensor:
    if torch.is_tensor(output):
        return output
    if isinstance(output, (tuple, list)) and output and torch.is_tensor(output[0]):
        return output[0]
    raise TypeError("transformer layer output does not expose a hidden-state tensor")


def validate_collapsed_span_additive_contract(
    model: nn.Module,
    record: ReplacementRecord,
    input_ids: torch.Tensor,
    *,
    layers_attr: str | None = None,
    rtol: float = 5e-3,
    atol: float = 5e-3,
) -> dict[str, Any]:
    """Measure the additive FFN-residual contract for one collapsed span.

    The check runs the same token window with the span-exit cell active and
    replaced by an exact zero branch.  It fails unless the resulting block
    output difference equals the cell output itself.  Training and benchmark
    paths call this independently of topology initialization, including after
    loading a learned checkpoint.
    """

    if not record.collapsed_span_layers:
        raise ValueError("the record does not describe a collapsed replacement span")
    layers = resolve_transformer_layers(model, layers_attr=layers_attr)
    exit_layer = int(record.layer_index)
    span_layers = tuple(int(layer) for layer in record.collapsed_span_layers)
    layer = layers[exit_layer]
    installed = _get_attr_path(layer, record.mlp_attr)
    if installed is not record.replacement:
        raise RuntimeError("collapsed span exit changed before contract validation")
    _validate_collapsed_span_norm_placement(layers, record)
    zero = ZeroFFNResidualBranch(
        layer_index=exit_layer,
        span_index=int(record.collapsed_span_index),
        span_layers=span_layers,
    )
    cell_inputs: list[torch.Tensor] = []
    base_exits: list[torch.Tensor] = []
    active_exits: list[torch.Tensor] = []

    def capture_input(_module, args):
        cell_inputs.append(args[0].detach())

    def capture_base(_module, _args, output):
        base_exits.append(_collapsed_span_layer_hidden(output).detach())

    def capture_active(_module, _args, output):
        active_exits.append(_collapsed_span_layer_hidden(output).detach())

    training_states = [(module, bool(module.training)) for module in model.modules()]
    model.eval()
    zero_handles: list[Any] = []
    active_handle = None
    try:
        zero_handles = [
            zero.register_forward_pre_hook(capture_input),
            layer.register_forward_hook(capture_base),
        ]
        _set_attr_path(layer, record.mlp_attr, zero)
        try:
            with torch.no_grad():
                model(input_ids=input_ids)
        finally:
            _set_attr_path(layer, record.mlp_attr, installed)
            for handle in zero_handles:
                handle.remove()
            zero_handles.clear()
        active_handle = layer.register_forward_hook(capture_active)
        with torch.no_grad():
            model(input_ids=input_ids)
            if not cell_inputs:
                raise RuntimeError("collapsed span contract input capture is empty")
            cell_contribution = installed(cell_inputs[0]).detach()
    finally:
        if _get_attr_path(layer, record.mlp_attr) is zero:
            _set_attr_path(layer, record.mlp_attr, installed)
        for handle in zero_handles:
            handle.remove()
        if active_handle is not None:
            active_handle.remove()
        for module, was_training in training_states:
            module.training = was_training
    if len(base_exits) != 1 or len(active_exits) != 1:
        raise RuntimeError("collapsed span contract exit capture is incomplete")
    observed_contribution = active_exits[0] - base_exits[0]
    absolute_error = (observed_contribution.float() - cell_contribution.float()).abs()
    max_abs_error = float(absolute_error.max().item())
    denominator = cell_contribution.float().abs().clamp_min(float(atol))
    max_relative_error = float((absolute_error / denominator).max().item())
    passed = torch.allclose(
        observed_contribution.float(),
        cell_contribution.float(),
        rtol=float(rtol),
        atol=float(atol),
    )
    diagnostics = {
        "schema": "dendritic_collapsed_span_additive_contract/v1",
        "span_layers": list(span_layers),
        "exit_layer": exit_layer,
        "probe_shape": list(input_ids.shape),
        "rtol": float(rtol),
        "atol": float(atol),
        "max_abs_error": max_abs_error,
        "max_relative_error": max_relative_error,
        "passed": bool(passed),
    }
    if not passed:
        raise NotImplementedError(
            "collapsed spans require an additive FFN residual branch: the "
            "measured block-output contribution does not equal the cell output "
            f"(max_abs_error={max_abs_error:.6g}, "
            f"max_relative_error={max_relative_error:.6g})"
        )
    return diagnostics


def _validate_collapsed_span_norm_placement(
    layers: Sequence[nn.Module], record: ReplacementRecord
) -> None:
    """Check complete norm ownership before validation or any restoration."""

    norm_attr = record.collapsed_span_post_mlp_norm_attr
    originals = record.collapsed_span_original_post_mlp_norms
    installed = record.collapsed_span_installed_post_mlp_norms
    if not norm_attr:
        if originals or installed:
            raise RuntimeError("collapsed replacement norm record is incomplete")
        return
    if len(originals) != len(record.collapsed_span_layers) or len(installed) != len(
        originals
    ):
        raise RuntimeError("collapsed replacement norm record is incomplete")
    for layer_index, module in zip(record.collapsed_span_layers, installed):
        if (
            not isinstance(module, nn.Identity)
            or _get_attr_path(layers[layer_index], norm_attr) is not module
        ):
            raise RuntimeError(
                f"collapsed span layer {layer_index}: installed post-MLP norm changed after patching"
            )


def restore_transformer_mlp_layers(
    model: nn.Module,
    records: Sequence[ReplacementRecord],
    *,
    layers_attr: str | None = None,
) -> None:
    """Reinstall every record's original module(s), undoing the patcher.

    Handles both record shapes: decoder-style records whose ``mlp_attr`` names
    one nested attribute on the layer, and ViT records whose
    ``mlp_attr == "intermediate+output"`` store both originals in an
    ``nn.ModuleDict``.  Fails closed when a layer does not currently hold the
    record's replacement -- that means the model was mutated again after
    patching (or this record was already restored), and overwriting whatever
    sits there would corrupt that later state.
    """

    layers = resolve_transformer_layers(model, layers_attr=layers_attr)
    for record in records:
        if record.collapsed_span_layers:
            _validate_collapsed_span_norm_placement(layers, record)
            if len(record.collapsed_span_layers) != len(
                record.collapsed_span_installed_modules
            ) or len(record.collapsed_span_layers) != len(
                record.collapsed_span_original_mlps
            ):
                raise RuntimeError("collapsed replacement record is incomplete")
            for layer_index, installed in zip(
                record.collapsed_span_layers,
                record.collapsed_span_installed_modules,
            ):
                current = _get_attr_path(layers[int(layer_index)], record.mlp_attr)
                if current is not installed:
                    raise RuntimeError(
                        f"cannot restore collapsed span layer {layer_index}: the "
                        "installed FFN module changed after patching"
                    )
            continue
        layer = layers[int(record.layer_index)]
        if record.mlp_attr == "intermediate+output":
            current = getattr(layer, "intermediate", None)
        else:
            current = _get_attr_path(layer, record.mlp_attr)
        if current is not record.replacement:
            raise RuntimeError(
                f"cannot restore layer {record.layer_index} "
                f"({record.mlp_attr!r}): the module currently installed "
                f"({type(current).__name__}) is not this record's "
                "replacement -- the model was mutated after patching or the "
                "record was already restored"
            )
    for record in records:
        if record.collapsed_span_layers:
            for layer_index, original in zip(
                record.collapsed_span_layers,
                record.collapsed_span_original_mlps,
            ):
                _set_attr_path(layers[int(layer_index)], record.mlp_attr, original)
            for layer_index, original in zip(
                record.collapsed_span_layers,
                record.collapsed_span_original_post_mlp_norms,
            ):
                _set_attr_path(
                    layers[int(layer_index)],
                    record.collapsed_span_post_mlp_norm_attr,
                    original,
                )
            continue
        layer = layers[int(record.layer_index)]
        if record.mlp_attr == "intermediate+output":
            originals = record.original_mlp
            layer.intermediate = originals["intermediate"]
            layer.output = originals["output"]
        else:
            _set_attr_path(layer, record.mlp_attr, record.original_mlp)


def apply_transformer_replacement_config(
    model: nn.Module,
    transformer_replacement: Any,
    core_config: Any,
    *,
    replacement_kwargs: dict[str, Any] | None = None,
) -> list[ReplacementRecord]:
    """Apply a YAML-style transformer replacement config to a loaded model."""
    tr = _to_plain_mapping(transformer_replacement)
    if not bool(tr.get("enabled", True)):
        return []

    if "layer_indices" in tr and "layers" not in tr:
        logger.warning(
            "transformer_replacement.layer_indices is deprecated; use "
            "transformer_replacement.layers instead."
        )
    if "mlp_attr" in tr and "target_module" not in tr:
        logger.warning(
            "transformer_replacement.mlp_attr is deprecated; use "
            "transformer_replacement.target_module instead."
        )
    layer_indices = tr.get("layers", tr.get("layer_indices", []))
    if not layer_indices:
        raise ValueError(
            "transformer_replacement.layers must contain at least one layer"
        )
    tied_groups = validate_parameter_tied_replacement_groups(
        tr.get("parameter_tied_replacement_groups", []), layer_indices
    )
    collapsed_spans = validate_collapsed_replacement_spans(
        tr.get("collapsed_replacement_spans", []), layer_indices
    )
    if collapsed_spans:
        return apply_collapsed_population_spans(
            model,
            tr,
            core_config,
            replacement_kwargs=replacement_kwargs,
        )

    if tr.get("collapsed_span_post_mlp_norm_attr", ""):
        raise ValueError("collapsed_span_post_mlp_norm_attr requires collapsed spans")

    replacement_overrides = _to_plain_mapping(tr.get("replacement_kwargs", {}))
    replacement_kind = str(
        replacement_overrides.get(
            "kind", replacement_overrides.get("type", "dendritic_ffn")
        )
    ).lower()
    population_kinds = {
        "population_network",
        "population_ffn",
        "population_network_ffn",
        "gated_population_network",
        "gated_population_ffn",
        "population_glu",
    }
    if tied_groups:
        if replacement_kind not in population_kinds:
            raise ValueError(
                "parameter_tied_replacement_groups supports only canonical "
                "PopulationNetwork replacement kinds"
            )
        if str(tr.get("model_family", "causal_lm")).lower() == "vit":
            raise NotImplementedError(
                "parameter_tied_replacement_groups does not yet support split ViT FFNs"
            )
        if bool(_to_plain_mapping(tr.get("selection", {})).get("enabled", False)):
            raise ValueError(
                "runtime FMI selection cannot be combined with parameter-tied groups; "
                "freeze identical compiled_plans_by_layer first"
            )
    if replacement_kind in {"dense_mlp_control", "dense_control", "dense_mlp"}:
        if str(tr.get("model_family", "causal_lm")).lower() == "vit":
            raise ValueError("dense_mlp_control does not yet support split ViT FFNs")
        return replace_transformer_mlp_layers(
            model,
            [int(idx) for idx in layer_indices],
            layers_attr=tr.get("layers_attr"),
            mlp_attr=tr.get("target_module", tr.get("mlp_attr", "mlp")),
            preserve_device_dtype=bool(tr.get("preserve_device_dtype", True)),
            replacement_class=DenseMLPControl,
        )

    selection = _to_plain_mapping(tr.get("selection", {}))
    plans_by_layer = _to_plain_mapping(tr.get("compiled_plans_by_layer", {}))
    if bool(selection.get("enabled", False)) and plans_by_layer:
        raise ValueError(
            "Runtime FMI selection and compiled_plans_by_layer are mutually exclusive"
        )
    if (
        bool(selection.get("enabled", False))
        or plans_by_layer
        or replacement_kind == "dense_swiglu_surrogate"
    ):
        layers = resolve_transformer_layers(model, layers_attr=tr.get("layers_attr"))
        ordered_indices = _validate_layer_indices(layer_indices)
        preserve = bool(tr.get("preserve_device_dtype", True))
        is_vit = str(tr.get("model_family", "causal_lm")).lower() == "vit"
        prepared: list[tuple[Any, ...]] = []
        for layer_index in ordered_indices:
            layer = layers[layer_index]
            if is_vit:
                intermediate = getattr(layer, "intermediate", None)
                output = getattr(layer, "output", None)
                if intermediate is None or output is None:
                    raise ValueError(
                        f"Layer {layer_index} lacks ViT intermediate/output modules"
                    )
                shim = nn.Module()
                shim.fc1 = intermediate.dense
                shim.fc2 = output.dense
                replacement = build_transformer_replacement_for_mlp(
                    shim,
                    tr,
                    core_config,
                    layer_index=int(layer_index),
                    replacement_kwargs=replacement_kwargs,
                )
                if preserve:
                    replacement = _move_replacement_like_source(
                        replacement, intermediate
                    )
                dropout = getattr(output, "dropout", None)
                dropout_p = float(getattr(dropout, "p", 0.0))
                prepared.append(
                    (
                        "vit",
                        layer,
                        layer_index,
                        intermediate,
                        output,
                        replacement,
                        dropout_p,
                    )
                )
            else:
                mlp_attr = tr.get("target_module", tr.get("mlp_attr", "mlp"))
                layer, original = _resolve_layer_target(layers, layer_index, mlp_attr)
                replacement = build_transformer_replacement_for_mlp(
                    original,
                    tr,
                    core_config,
                    layer_index=int(layer_index),
                    replacement_kwargs=replacement_kwargs,
                )
                if preserve:
                    replacement = _move_replacement_like_source(replacement, original)
                prepared.append(
                    ("decoder", layer, layer_index, mlp_attr, original, replacement)
                )

        records: list[ReplacementRecord] = []
        for entry in prepared:
            if entry[0] == "vit":
                _, layer, layer_index, intermediate, output, replacement, p = entry
                layer.intermediate = replacement
                layer.output = _ViTResidualOutput(p)
                records.append(
                    ReplacementRecord(
                        layer_index=int(layer_index),
                        mlp_attr="intermediate+output",
                        original_mlp=nn.ModuleDict(
                            {"intermediate": intermediate, "output": output}
                        ),
                        replacement=replacement,
                    )
                )
            else:
                _, layer, layer_index, mlp_attr, original, replacement = entry
                _set_attr_path(layer, mlp_attr, replacement)
                records.append(
                    ReplacementRecord(
                        layer_index=int(layer_index),
                        mlp_attr=str(mlp_attr),
                        original_mlp=original,
                        replacement=replacement,
                    )
                )
        return apply_parameter_tied_population_groups_to_records(
            model,
            records,
            tied_groups,
            layers_attr=tr.get("layers_attr"),
        )

    if replacement_kind in {"ei_stack", "layerwise_ei_stack", "einet_stack"}:
        kwargs = build_ei_stack_kwargs_from_core_config(
            core_config,
            transformer_replacement=tr,
            num_layers=len(layer_indices),
            overrides=replacement_kwargs,
        )
        return replace_transformer_mlp_layers_with_ei_stack(
            model,
            [int(idx) for idx in layer_indices],
            replacement_kwargs=kwargs,
            layers_attr=tr.get("layers_attr"),
            mlp_attr=tr.get("target_module", tr.get("mlp_attr", "mlp")),
            preserve_device_dtype=bool(tr.get("preserve_device_dtype", True)),
        )

    if replacement_kind in population_kinds:
        kwargs = build_population_network_ffn_kwargs_from_core_config(
            core_config,
            transformer_replacement=tr,
            overrides=replacement_kwargs,
        )
        replacement_class = (
            GatedPopulationNetworkFFNReplacement
            if replacement_kind
            in {
                "gated_population_network",
                "gated_population_ffn",
                "population_glu",
            }
            else PopulationNetworkFFNReplacement
        )
        if str(tr.get("model_family", "causal_lm")).lower() == "vit":
            records = replace_vit_ffn_layers(
                model,
                [int(idx) for idx in layer_indices],
                replacement_kwargs=kwargs,
                layers_attr=tr.get("layers_attr"),
                preserve_device_dtype=bool(tr.get("preserve_device_dtype", True)),
                replacement_class=replacement_class,
            )
        else:
            records = replace_transformer_mlp_layers(
                model,
                [int(idx) for idx in layer_indices],
                replacement_kwargs=kwargs,
                layers_attr=tr.get("layers_attr"),
                mlp_attr=tr.get("target_module", tr.get("mlp_attr", "mlp")),
                preserve_device_dtype=bool(tr.get("preserve_device_dtype", True)),
                replacement_class=replacement_class,
            )
        records = _annotate_uniform_compiled_plan(
            records,
            tr,
        )
        return apply_parameter_tied_population_groups_to_records(
            model,
            records,
            tied_groups,
            layers_attr=tr.get("layers_attr"),
        )

    kwargs = build_dendritic_ffn_kwargs_from_core_config(
        core_config,
        transformer_replacement=tr,
        overrides=replacement_kwargs,
    )
    if str(tr.get("model_family", "causal_lm")).lower() == "vit":
        return replace_vit_ffn_layers(
            model,
            [int(idx) for idx in layer_indices],
            replacement_kwargs=kwargs,
            layers_attr=tr.get("layers_attr"),
            preserve_device_dtype=bool(tr.get("preserve_device_dtype", True)),
            replacement_class=(
                GatedDendriticFFNReplacement
                if replacement_kind
                in {"gated_dendritic_ffn", "gated_dendritic", "dendritic_glu"}
                else DendriticFFNReplacement
            ),
        )
    return replace_transformer_mlp_layers(
        model,
        [int(idx) for idx in layer_indices],
        replacement_kwargs=kwargs,
        layers_attr=tr.get("layers_attr"),
        mlp_attr=tr.get("target_module", tr.get("mlp_attr", "mlp")),
        preserve_device_dtype=bool(tr.get("preserve_device_dtype", True)),
        replacement_class=(
            GatedDendriticFFNReplacement
            if replacement_kind
            in {"gated_dendritic_ffn", "gated_dendritic", "dendritic_glu"}
            else DendriticFFNReplacement
        ),
    )


def load_transformer_with_dendritic_replacements(
    transformer_replacement: Any,
    core_config: Any,
) -> tuple[nn.Module, list[ReplacementRecord]]:
    """Load a Hugging Face causal LM and apply dendritic MLP replacements."""
    tr = _to_plain_mapping(transformer_replacement)
    if "backbone" in tr and "model_name" not in tr:
        logger.warning(
            "transformer_replacement.backbone is deprecated; use "
            "transformer_replacement.model_name instead."
        )
    model_name = tr.get("model_name", tr.get("backbone", ""))
    if not model_name:
        raise ValueError("transformer_replacement.model_name is required")

    try:
        from transformers import AutoModel, AutoModelForCausalLM
    except ImportError as exc:
        raise ImportError(
            "transformers is required to load transformer_replacement models"
        ) from exc

    model_kwargs = _to_plain_mapping(tr.get("model_kwargs", {}))
    family = str(tr.get("model_family", "causal_lm")).strip().lower()
    loader = str(tr.get("model_loader", "auto")).strip().lower()
    if loader == "auto":
        loader = {
            "vit": "image_classification",
            "dino": "base_model",
            "dinov2": "base_model",
            "vision_backbone": "base_model",
        }.get(family, "causal_lm")
    if loader == "image_classification":
        from transformers import AutoModelForImageClassification

        model = AutoModelForImageClassification.from_pretrained(
            model_name, **model_kwargs
        )
    elif loader == "base_model":
        model = AutoModel.from_pretrained(model_name, **model_kwargs)
    elif loader == "causal_lm":
        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    else:
        raise ValueError(
            "transformer_replacement.model_loader must be auto, causal_lm, "
            "image_classification, or base_model"
        )
    records = apply_transformer_replacement_config(model, tr, core_config)
    return model, records
