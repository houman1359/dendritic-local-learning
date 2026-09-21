"""Reusable output projections for replacement cells.

The replacement core and its readout are separate resource decisions.  This
module keeps dense, low-rank, fixed-index, rewiring, candidate-pool, and
developmental dense-to-sparse readouts on one package-owned construction path.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from dendritic_modeling.networks.architectures.excitation_inhibition.synapse import (
    DenseToSparseLinear,
    IndexedDynamicTopKLinear,
    IndexedRewireLinear,
    IndexedSparseLinear,
    StochasticTopKLinear,
    TopKLinear,
    VarianceTopKLinear,
)
from dendritic_modeling.networks.architectures.replacement.cells import (
    make_linear_output_projection,
)
from dendritic_modeling.networks.utils.weight_transforms import (
    POSITIVE_WEIGHT_TRANSFORMS,
    apply_weight_transform,
    inverse_weight_transform,
)

OUTPUT_TOPOLOGY_MODES = (
    "standard",
    "stochastic",
    "variance",
    "indexed",
    "indexed_rewire",
    "indexed_dynamic",
    "dense_to_sparse",
)


class TransformedLinear(nn.Module):
    """Dense linear map whose effective weights obey a declared transform."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        *,
        bias: bool,
        init_std: float,
        weight_transform: str,
    ):
        super().__init__()
        self.in_features = int(input_dim)
        self.out_features = int(output_dim)
        self.weight_transform = str(weight_transform).lower()
        desired = (
            torch.empty(self.out_features, self.in_features)
            .normal_(mean=0.0, std=float(init_std))
            .abs_()
            .clamp_min(1e-6)
        )
        self.pre_w = nn.Parameter(
            inverse_weight_transform(desired, self.weight_transform)
        )
        self.pre_bias = nn.Parameter(torch.zeros(self.out_features)) if bias else None

    @property
    def weight(self) -> torch.Tensor:
        return apply_weight_transform(self.pre_w, self.weight_transform)

    @property
    def bias(self) -> torch.Tensor | None:
        if self.pre_bias is None:
            return None
        return apply_weight_transform(self.pre_bias, self.weight_transform)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return F.linear(inputs, self.weight, self.bias)


class PositiveEIOutputProjection(nn.Module):
    """Signed readout implemented as two nonnegative E/I projections.

    The interface output may be signed, but every stored pathway has a
    nonnegative effective weight: ``y = W_E h - W_I h``.  This prevents a
    nominally biological core from quietly using an unrestricted signed
    output matrix to recover the teacher.
    """

    def __init__(self, excitatory: nn.Module, inhibitory: nn.Module):
        super().__init__()
        self.excitatory = excitatory
        self.inhibitory = inhibitory

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.excitatory(inputs) - self.inhibitory(inputs)


class PositiveEICoreOutputAdapter(nn.Module):
    """Wrap a core with a strict positive-path E/I interface readout."""

    def __init__(
        self,
        core: nn.Module,
        output_dim: int,
        *,
        weight_transform: str = "softplus",
    ):
        super().__init__()
        self.core = core
        self.input_dim = getattr(core, "input_dim", None)
        self.output_dim = int(output_dim)
        self.output_adapter = make_positive_ei_output_projection(
            int(core.output_dim),
            self.output_dim,
            weight_transform=weight_transform,
            bias=False,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.output_adapter(self.core(inputs))

    def decay_weights(self, weight_decay: float, weight_boosting: bool = False) -> None:
        decay = getattr(self.core, "decay_weights", None)
        if callable(decay):
            decay(weight_decay, weight_boosting)

    def apply_rewiring(self) -> None:
        rewire = getattr(self.core, "apply_rewiring", None)
        if callable(rewire):
            rewire()


def make_output_projection(
    input_dim: int,
    output_dim: int,
    *,
    rank: int | None = None,
    topk: int | None = None,
    topology_mode: str = "indexed",
    bias: bool = False,
    init_std: float = 0.02,
    init_method: str = "xavier_normal",
    seed: int | None = None,
    candidate_size: int | None = None,
    selection: str = "standard",
    noise_level: float = 0.0,
    temperature: float = 0.5,
    ultrafast: bool = True,
    variance_momentum: float = 0.9,
    rewire_frequency: int = 100,
    rewire_quantile: float = 0.05,
    rewire_until_step: int | None = None,
    dense_to_sparse_initial_density: float = 1.0,
    dense_to_sparse_initial_k: int | None = None,
    dense_to_sparse_start_step: int = 0,
    dense_to_sparse_end_step: int = 1000,
    dense_to_sparse_update_interval: int = 1,
    dense_to_sparse_schedule: str = "cubic",
    dense_to_sparse_freeze_on_end: bool = False,
    dense_to_sparse_advance_on_forward: bool = False,
    dense_to_sparse_prune_metric: str = "magnitude",
    output_chunk_size: int = 2048,
    index_dtype: str = "int64",
    workspace_mb: float | None = None,
    cache_transformed_weights: bool = False,
    recompute_backward: bool = False,
    projection_backend: str = "eager",
    persistent_indices: bool = True,
    init_mode: str = "per_rank",
    support_group_rows: int = 1,
    support_col_block: int = 1,
    weight_transform: str = "identity",
) -> nn.Module:
    """Build one readout while enforcing mutually exclusive budgets.

    ``rank`` selects a two-factor dense readout, ``topk`` selects one of the
    sparse topology modes, and omitting both selects a dense linear readout.
    The function is domain-independent and can be reused by transformer,
    vision, or synthetic replacement cells.
    """

    input_dim = int(input_dim)
    output_dim = int(output_dim)
    if input_dim < 1 or output_dim < 1:
        raise ValueError("input_dim and output_dim must be positive")
    if rank is not None and topk is not None:
        raise ValueError("rank and topk are mutually exclusive")

    normalized_transform = str(weight_transform).strip().lower()
    if normalized_transform not in {"identity", *POSITIVE_WEIGHT_TRANSFORMS}:
        raise ValueError(f"unsupported output weight_transform {weight_transform!r}")

    if topk is None:
        if rank is None:
            if normalized_transform != "identity":
                return TransformedLinear(
                    input_dim,
                    output_dim,
                    bias=bias,
                    init_std=init_std,
                    weight_transform=normalized_transform,
                )
            return make_linear_output_projection(
                input_dim,
                output_dim,
                bias=bias,
                init_std=init_std,
            )
        rank = int(rank)
        if rank < 1 or rank > min(input_dim, output_dim):
            raise ValueError("rank must be in [1, min(input_dim, output_dim)]")
        if normalized_transform == "identity":
            first = nn.Linear(input_dim, rank, bias=False)
            second = nn.Linear(rank, output_dim, bias=bool(bias))
            nn.init.normal_(first.weight, mean=0.0, std=float(init_std))
            nn.init.normal_(second.weight, mean=0.0, std=float(init_std))
            if second.bias is not None:
                nn.init.zeros_(second.bias)
        else:
            first = TransformedLinear(
                input_dim,
                rank,
                bias=False,
                init_std=init_std,
                weight_transform=normalized_transform,
            )
            second = TransformedLinear(
                rank,
                output_dim,
                bias=bool(bias),
                init_std=init_std,
                weight_transform=normalized_transform,
            )
        return nn.Sequential(first, second)

    topk = int(topk)
    if topk < 1 or topk > input_dim:
        raise ValueError("topk must be in [1, input_dim]")
    if bias:
        raise ValueError("sparse output projections currently require bias=false")

    mode = str(topology_mode).strip().lower()
    if mode not in OUTPUT_TOPOLOGY_MODES:
        raise ValueError(
            f"topology_mode must be one of {OUTPUT_TOPOLOGY_MODES}, got {mode!r}"
        )
    common = {
        "in_features": input_dim,
        "out_features": output_dim,
        "K": topk,
        "init_method": str(init_method),
        "weight_transform": normalized_transform,
    }
    indexed = {
        "seed": seed,
        "output_chunk_size": int(output_chunk_size),
        "index_dtype": str(index_dtype),
        "workspace_mb": workspace_mb,
        "cache_transformed_weights": bool(cache_transformed_weights),
        "recompute_backward": bool(recompute_backward),
        "projection_backend": str(projection_backend),
        "persistent_indices": bool(persistent_indices),
        "init_mode": str(init_mode),
    }
    if mode == "standard":
        return TopKLinear(
            **common,
            noise_level=float(noise_level),
        )
    if mode == "stochastic":
        return StochasticTopKLinear(
            **common,
            noise_level=float(noise_level),
            temperature=float(temperature),
            ultrafast=bool(ultrafast),
        )
    if mode == "variance":
        return VarianceTopKLinear(
            **common,
            noise_level=float(noise_level),
            momentum=float(variance_momentum),
        )
    if mode == "indexed":
        return IndexedSparseLinear(
            **common,
            **indexed,
            support_group_rows=int(support_group_rows),
            support_col_block=int(support_col_block),
        )
    if mode == "indexed_rewire":
        return IndexedRewireLinear(
            **common,
            **indexed,
            rewire_frequency=int(rewire_frequency),
            rewire_quantile=float(rewire_quantile),
            rewire_until_step=rewire_until_step,
            support_group_rows=int(support_group_rows),
            support_col_block=int(support_col_block),
        )
    if mode == "indexed_dynamic":
        return IndexedDynamicTopKLinear(
            **common,
            **indexed,
            candidate_size=candidate_size,
            selection=str(selection),
            noise_level=0.0,
        )
    return DenseToSparseLinear(
        **common,
        noise_level=0.0,
        initial_density=float(dense_to_sparse_initial_density),
        initial_k=dense_to_sparse_initial_k,
        start_step=int(dense_to_sparse_start_step),
        end_step=int(dense_to_sparse_end_step),
        update_interval=int(dense_to_sparse_update_interval),
        schedule=str(dense_to_sparse_schedule),
        freeze_on_end=bool(dense_to_sparse_freeze_on_end),
        advance_on_forward=bool(dense_to_sparse_advance_on_forward),
        prune_metric=str(dense_to_sparse_prune_metric),
    )


def make_positive_ei_output_projection(
    input_dim: int,
    output_dim: int,
    **kwargs: object,
) -> PositiveEIOutputProjection:
    """Build matched positive E and I readouts with independent topology."""

    options = dict(kwargs)
    transform = str(options.pop("weight_transform", "softplus")).lower()
    if transform not in POSITIVE_WEIGHT_TRANSFORMS:
        raise ValueError("positive E/I output readout needs a positive transform")
    seed = options.get("seed")
    excitatory = make_output_projection(
        input_dim,
        output_dim,
        weight_transform=transform,
        **options,
    )
    if seed is not None:
        options["seed"] = int(seed) + 104_729
    inhibitory = make_output_projection(
        input_dim,
        output_dim,
        weight_transform=transform,
        **options,
    )
    return PositiveEIOutputProjection(excitatory, inhibitory)


def output_projection_parameter_counts(projection: nn.Module) -> tuple[int, int]:
    """Return stored and currently active learned projection parameters."""

    if isinstance(projection, PositiveEIOutputProjection):
        e_stored, e_active = output_projection_parameter_counts(projection.excitatory)
        i_stored, i_active = output_projection_parameter_counts(projection.inhibitory)
        return e_stored + i_stored, e_active + i_active
    stored = sum(param.numel() for param in projection.parameters())
    pre_w = getattr(projection, "pre_w", None)
    mask_fn = getattr(projection, "weight_mask", None)
    if pre_w is None or not callable(mask_fn):
        return int(stored), int(stored)
    active_weights = int(mask_fn().ne(0).sum().item())
    return int(stored), int(stored - pre_w.numel() + active_weights)


__all__ = [
    "OUTPUT_TOPOLOGY_MODES",
    "PositiveEICoreOutputAdapter",
    "PositiveEIOutputProjection",
    "TransformedLinear",
    "make_output_projection",
    "make_positive_ei_output_projection",
    "output_projection_parameter_counts",
]
