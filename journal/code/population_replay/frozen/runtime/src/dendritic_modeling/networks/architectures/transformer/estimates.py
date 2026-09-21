"""Parameter estimates for transformer feed-forward replacements."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from dendritic_modeling.config.conversion import normalize_sparsity_type


def _normalize_topk_type(value: Any) -> str:
    return normalize_sparsity_type(value)


def estimate_deepseek_dense_ffn_params(
    hidden_size: int = 7168,
    intermediate_size: int = 18432,
    *,
    gated: bool = True,
    bias: bool = False,
) -> int:
    """Estimate parameters in a DeepSeek/Llama-style dense FFN block."""
    projections = 3 if gated else 2
    params = projections * int(hidden_size) * int(intermediate_size)
    if bias:
        params += (2 if gated else 1) * int(intermediate_size) + int(hidden_size)
    return params


def estimate_dendritic_ffn_params(
    *,
    hidden_size: int,
    dendritic_units: int,
    branch_factors: Sequence[int],
    synapses_per_branch: int,
    candidate_size: int | None = None,
    input_transform: str = "signed_split",
    topk_type: str = "indexed",
    output_topk: int | None = None,
    output_bias: bool = False,
    somatic_synapses: bool = True,
    reactivation_parameters_per_unit: int = 2,
    pre_norm: bool = False,
) -> dict[str, int]:
    """Estimate stored and active parameters for a dendritic FFN replacement.

    ``reactivation_parameters_per_unit`` defaults to two for ``param_tanh``
    (one slope and one midpoint per branch unit). Set it to zero for a fixed
    activation, or one for ``param_tanh_only_m``. ``pre_norm`` accounts for
    the weight and bias of the optional input LayerNorm.
    """
    topk_type = _normalize_topk_type(topk_type)
    input_dim = int(hidden_size) * (2 if input_transform == "signed_split" else 1)
    layer_sizes = [int(dendritic_units)]
    for branch_factor in branch_factors:
        layer_sizes.append(layer_sizes[-1] * int(branch_factor))
    dendritic_layer_sizes = list(reversed(layer_sizes))
    if not somatic_synapses:
        dendritic_layer_sizes = dendritic_layer_sizes[:-1]
    branch_units = sum(dendritic_layer_sizes)

    active_synapses = branch_units * int(synapses_per_branch)
    if topk_type in {"indexed", "indexed_rewire"}:
        stored_synapses = active_synapses
    elif topk_type == "indexed_dynamic":
        if candidate_size is None:
            candidate_size = min(
                input_dim,
                max(int(synapses_per_branch), 4 * int(synapses_per_branch)),
            )
        stored_synapses = branch_units * int(candidate_size)
    else:
        stored_synapses = branch_units * input_dim

    blocklinear = sum(
        int(parent) * int(branch_factor)
        for parent, branch_factor in zip(layer_sizes, branch_factors)
    )
    if output_topk is not None:
        output_topk = int(output_topk)
        if output_topk < 1 or output_topk > int(dendritic_units):
            raise ValueError("output_topk must be in [1, dendritic_units]")
        if output_bias:
            raise ValueError("output_topk currently requires output_bias=false")
    projection_width = int(dendritic_units) if output_topk is None else output_topk
    projection = projection_width * int(hidden_size)
    if output_bias:
        projection += int(hidden_size)
    reactivation = branch_units * max(int(reactivation_parameters_per_unit), 0)
    pre_norm_params = 2 * int(hidden_size) if pre_norm else 0
    stored_total = stored_synapses + blocklinear + projection + reactivation
    active_total = active_synapses + blocklinear + projection + reactivation
    stored_total += pre_norm_params
    active_total += pre_norm_params
    return {
        "input_dim": input_dim,
        "branch_units": branch_units,
        "active_synapses": active_synapses,
        "stored_synapses": stored_synapses,
        "blocklinear": blocklinear,
        "reactivation": reactivation,
        "pre_norm": pre_norm_params,
        "output_projection": projection,
        "stored_total": stored_total,
        "active_total": active_total,
    }


def estimate_gated_dendritic_ffn_params(**kwargs: Any) -> dict[str, int]:
    """Estimate a two-core gated dendritic FFN with one shared projection.

    The gate and value cores have identical dendritic shapes.  Their input
    adapter has no learned parameters; optional pre-normalization and the
    output projection are shared and therefore counted only once.
    """
    single_kwargs = dict(kwargs)
    output_topk = single_kwargs.pop("output_topk", None)
    single = estimate_dendritic_ffn_params(**single_kwargs)
    hidden_size = int(single_kwargs["hidden_size"])
    dendritic_units = int(single_kwargs["dendritic_units"])
    if output_topk is not None:
        output_topk = int(output_topk)
        if output_topk < 1 or output_topk > dendritic_units:
            raise ValueError("output_topk must be in [1, dendritic_units]")
    projection = hidden_size * (dendritic_units if output_topk is None else output_topk)
    if bool(single_kwargs.get("output_bias", False)):
        projection += hidden_size
    pre_norm = 2 * hidden_size if bool(single_kwargs.get("pre_norm", False)) else 0
    doubled_fields = (
        "branch_units",
        "active_synapses",
        "stored_synapses",
        "blocklinear",
        "reactivation",
    )
    estimate = {
        "input_dim": int(single["input_dim"]),
        **{field: 2 * int(single[field]) for field in doubled_fields},
        "pre_norm": pre_norm,
        "output_projection": projection,
    }
    estimate["stored_total"] = (
        2
        * (
            int(single["stored_total"])
            - int(single["output_projection"])
            - int(single["pre_norm"])
        )
        + projection
        + pre_norm
    )
    estimate["active_total"] = (
        2
        * (
            int(single["active_total"])
            - int(single["output_projection"])
            - int(single["pre_norm"])
        )
        + projection
        + pre_norm
    )
    return estimate
