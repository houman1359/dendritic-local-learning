"""Replacement modules for transformer feed-forward blocks."""

from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from dendritic_modeling.networks.architectures.excitation_inhibition.dendritic import (
    DendriNet,
    DendriNetConfig,
    DendriticSynapseConfig,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.ei_layer import (
    ExcitationInhibitionLayer,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.synapse import (
    IndexedSparseLinear,
)
from dendritic_modeling.networks.architectures.replacement import (
    RUNTIME_TENSOR_CONTRACT_SCHEMA,
    TEACHER_TOPOLOGY_METRICS,
    LayerwiseTokenReplacementStack,
    NonNegativeInputAdapter,
    TokenReplacementCell,
    copy_topk_weight_,
    make_linear_output_projection,
    make_output_projection,
    make_positive_ei_output_projection,
    output_projection_parameter_counts,
    preserve_runtime_tensor_contract,
    transformed_feature_dim,
)
from dendritic_modeling.networks.architectures.transformer.estimates import (
    estimate_dendritic_ffn_params,
    estimate_gated_dendritic_ffn_params,
)
from dendritic_modeling.networks.architectures.transformer.utils import (
    _as_list,
    _infer_mlp_dims,
)
from dendritic_modeling.networks.utils.weight_transforms import (
    POSITIVE_WEIGHT_TRANSFORMS,
)


class DenseMLPControl(nn.Module):
    """Exact teacher-MLP copy used as an adaptation-matched dense control.

    The wrapper preserves the teacher module's forward semantics and initial
    parameters while exposing the checkpoint and storage interface expected of
    a replacement cell. Under ``train_target=replacement_only``, only this copy
    is optimized; the rest of the language model stays frozen exactly as it
    does for a dendritic replacement.
    """

    runtime_tensor_contract_schema = RUNTIME_TENSOR_CONTRACT_SCHEMA

    def __init__(self, mlp: nn.Module):
        super().__init__()
        self.mlp = deepcopy(mlp)

    @classmethod
    def from_mlp(cls, mlp: nn.Module, **_: object) -> DenseMLPControl:
        return cls(mlp)

    def forward(self, *args: object, **kwargs: object) -> object:
        references = [
            value
            for value in (*args, *kwargs.values())
            if torch.is_tensor(value) and value.is_floating_point()
        ]
        if not references:
            raise TypeError("DenseMLPControl requires a floating activation tensor")
        reference = references[0]
        parameter = next(self.mlp.parameters(), None)
        if parameter is None or not parameter.is_floating_point():
            raise RuntimeError("DenseMLPControl has no floating master parameter")

        def _to_master(value: object) -> object:
            if torch.is_tensor(value) and value.is_floating_point():
                return value.to(device=parameter.device, dtype=parameter.dtype)
            return value

        output = self.mlp(
            *(_to_master(value) for value in args),
            **{key: _to_master(value) for key, value in kwargs.items()},
        )
        return preserve_runtime_tensor_contract(
            output,
            reference,
            boundary=type(self).__name__,
        )

    def parameter_estimate(self) -> dict[str, int]:
        total = sum(int(parameter.numel()) for parameter in self.parameters())
        return {
            "stored_total": total,
            "active_total": total,
            "dense_control": total,
        }


class DendriticFFNReplacement(TokenReplacementCell):
    """Token-wise DendriNet replacement for transformer MLP/FFN modules.

    The module preserves the transformer contract ``[..., hidden_size] ->
    [..., hidden_size]``.  It is intended to replace the MLP submodule inside a
    residual transformer block; the residual connection itself remains outside
    this module.
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        dendritic_units: int,
        branch_factors: Sequence[int] = (3, 3),
        synapses_per_branch: int = 40,
        input_transform: str = "signed_split",
        topk_type: str = "indexed_rewire",
        topk_noise_level: float = 0.0,
        topk_temperature: float = 0.5,
        topk_ultrafast: bool = False,
        topk_weight_norm_order: int | None = None,
        topk_gamma: float = 1.0,
        dense_to_sparse_initial_density: float = 1.0,
        dense_to_sparse_initial_k: int | None = None,
        dense_to_sparse_start_step: int = 0,
        dense_to_sparse_end_step: int = 1000,
        dense_to_sparse_update_interval: int = 1,
        dense_to_sparse_schedule: str = "cubic",
        dense_to_sparse_freeze_on_end: bool = False,
        dense_to_sparse_advance_on_forward: bool = True,
        dense_to_sparse_prune_metric: str = "weight",
        indexed_seed: int | None = None,
        indexed_candidate_size: int | None = None,
        indexed_selection: str = "standard",
        indexed_output_chunk_size: int = 2048,
        indexed_index_dtype: str = "int64",
        indexed_workspace_mb: float | None = None,
        indexed_cache_transformed_weights: bool = False,
        indexed_recompute_backward: bool = False,
        indexed_projection_backend: str = "eager",
        indexed_persistent_indices: bool = True,
        indexed_init_mode: str = "per_rank",
        indexed_rewire_frequency: int = 100,
        indexed_rewire_quantile: float = 0.05,
        indexed_rewire_until_step: int | None = None,
        use_shunting: bool = True,
        use_additive_normalization: bool = False,
        reactivate: bool = True,
        reactivation_type: str = "param_tanh",
        somatic_synapses: bool = True,
        reactivation_init_m: float = 1.0,
        reactivation_init_b: float = 0.5,
        reactivation_init_policy: str = "analytical",
        reactivation_occupancy_quantile_low: float | None = None,
        reactivation_occupancy_quantile_high: float | None = None,
        reactivation_occupancy_target_low: float | None = None,
        reactivation_occupancy_target_high: float | None = None,
        reactivation_calibration_min_quantile_width: float = 1e-3,
        reactivation_calibration_max_m: float = 50.0,
        reactivation_calibration_revert_on_invalid: bool = True,
        reactivation_sigma_aware_k: float = 0.25,
        reactivation_memory_efficient: bool = False,
        weight_transform: str = "softplus",
        topk_init_method: str = "xavier_normal",
        efficient_blocklinear: bool = False,
        output_topk: int | None = None,
        output_bias: bool = False,
        output_init_std: float = 0.02,
        output_scale: float = 1.0,
        pre_norm: bool = False,
    ):
        if dendritic_units < 1:
            raise ValueError("dendritic_units must be >= 1")
        if synapses_per_branch < 1:
            raise ValueError("synapses_per_branch must be >= 1")

        hidden_size = int(hidden_size)
        dendritic_units = int(dendritic_units)
        output_topk = None if output_topk is None else int(output_topk)
        branch_factors = _as_list(branch_factors)
        synapses_per_branch = int(synapses_per_branch)
        input_dim = transformed_feature_dim(hidden_size, input_transform)
        synapse_config = DendriticSynapseConfig.from_kwargs(
            {
                "topk_init_method": topk_init_method,
                "topk_noise_level": topk_noise_level,
                "topk_type": topk_type,
                "topk_temperature": topk_temperature,
                "topk_ultrafast": topk_ultrafast,
                "topk_weight_norm_order": topk_weight_norm_order,
                "topk_gamma": topk_gamma,
                "dense_to_sparse_initial_density": (dense_to_sparse_initial_density),
                "dense_to_sparse_initial_k": dense_to_sparse_initial_k,
                "dense_to_sparse_start_step": dense_to_sparse_start_step,
                "dense_to_sparse_end_step": dense_to_sparse_end_step,
                "dense_to_sparse_update_interval": (dense_to_sparse_update_interval),
                "dense_to_sparse_schedule": dense_to_sparse_schedule,
                "dense_to_sparse_freeze_on_end": dense_to_sparse_freeze_on_end,
                "dense_to_sparse_advance_on_forward": (
                    dense_to_sparse_advance_on_forward
                ),
                "dense_to_sparse_prune_metric": dense_to_sparse_prune_metric,
                "indexed_seed": indexed_seed,
                "indexed_candidate_size": indexed_candidate_size,
                "indexed_selection": indexed_selection,
                "indexed_output_chunk_size": indexed_output_chunk_size,
                "indexed_index_dtype": indexed_index_dtype,
                "indexed_workspace_mb": indexed_workspace_mb,
                "indexed_cache_transformed_weights": indexed_cache_transformed_weights,
                "indexed_recompute_backward": indexed_recompute_backward,
                "indexed_projection_backend": indexed_projection_backend,
                "indexed_persistent_indices": indexed_persistent_indices,
                "indexed_init_mode": indexed_init_mode,
                "indexed_rewire_frequency": indexed_rewire_frequency,
                "indexed_rewire_quantile": indexed_rewire_quantile,
                "indexed_rewire_until_step": indexed_rewire_until_step,
                "use_shunting": use_shunting,
                "use_additive_normalization": use_additive_normalization,
                "reactivate": reactivate,
                "reactivation_type": reactivation_type,
                "reactivation_init_m": reactivation_init_m,
                "reactivation_init_b": reactivation_init_b,
                "reactivation_init_policy": reactivation_init_policy,
                "reactivation_occupancy_quantile_low": (
                    reactivation_occupancy_quantile_low
                ),
                "reactivation_occupancy_quantile_high": (
                    reactivation_occupancy_quantile_high
                ),
                "reactivation_occupancy_target_low": (
                    reactivation_occupancy_target_low
                ),
                "reactivation_occupancy_target_high": (
                    reactivation_occupancy_target_high
                ),
                "reactivation_calibration_min_quantile_width": (
                    reactivation_calibration_min_quantile_width
                ),
                "reactivation_calibration_max_m": reactivation_calibration_max_m,
                "reactivation_calibration_revert_on_invalid": (
                    reactivation_calibration_revert_on_invalid
                ),
                "reactivation_sigma_aware_k": reactivation_sigma_aware_k,
                "reactivation_memory_efficient": reactivation_memory_efficient,
                "weight_transform": weight_transform,
                "efficient_blocklinear": bool(efficient_blocklinear),
            }
        )
        core = DendriNet(
            config=DendriNetConfig(
                n_soma=dendritic_units,
                branch_factors=tuple(branch_factors),
                excitatory_input_dim=input_dim,
                excitatory_synapses_per_branch=synapses_per_branch,
                inhibitory_input_dim=None,
                inhibitory_synapses_per_branch=None,
                somatic_synapses=bool(somatic_synapses),
                synapse_config=synapse_config,
            ),
        )

        super().__init__(
            input_dim=hidden_size,
            output_dim=hidden_size,
            input_transform=input_transform,
            core=core,
            core_output_dim=dendritic_units,
            output_bias=output_bias,
            output_init_std=output_init_std,
            output_scale=output_scale,
            pre_norm=pre_norm,
        )
        if output_topk is not None:
            output_seed = None if indexed_seed is None else int(indexed_seed) + 1
            self.output_projection = make_output_projection(
                dendritic_units,
                hidden_size,
                topk=output_topk,
                topology_mode="indexed",
                bias=output_bias,
                init_std=output_init_std,
                init_method=topk_init_method,
                seed=output_seed,
                output_chunk_size=indexed_output_chunk_size,
                index_dtype=indexed_index_dtype,
                workspace_mb=indexed_workspace_mb,
                cache_transformed_weights=indexed_cache_transformed_weights,
                recompute_backward=indexed_recompute_backward,
                projection_backend=indexed_projection_backend,
                persistent_indices=indexed_persistent_indices,
                init_mode=indexed_init_mode,
            )

        self.hidden_size = hidden_size
        self.dendritic_units = dendritic_units
        self.branch_factors = branch_factors
        self.synapses_per_branch = synapses_per_branch
        self.indexed_candidate_size = indexed_candidate_size
        self.indexed_selection = indexed_selection
        self.indexed_rewire_frequency = int(indexed_rewire_frequency)
        self.indexed_rewire_quantile = float(indexed_rewire_quantile)
        self.indexed_rewire_until_step = indexed_rewire_until_step
        self.topk_type = topk_type
        self.topk_noise_level = float(topk_noise_level)
        self.topk_temperature = float(topk_temperature)
        self.topk_ultrafast = bool(topk_ultrafast)
        self.somatic_synapses = bool(somatic_synapses)
        self.efficient_blocklinear = bool(efficient_blocklinear)
        self.output_topk = None if output_topk is None else int(output_topk)
        self.reactivate = bool(reactivate)
        self.reactivation_type = str(reactivation_type)
        self.uses_pre_norm = not isinstance(self.pre_norm, nn.Identity)

    @classmethod
    def from_mlp(cls, mlp: nn.Module, **kwargs: object) -> DendriticFFNReplacement:
        """Construct a replacement by inferring ``hidden_size`` from an MLP."""
        hidden_size, intermediate_size = _infer_mlp_dims(mlp)
        replacement = cls(hidden_size=hidden_size, **kwargs)
        replacement.reference_intermediate_size = intermediate_size
        return replacement

    def forward(self, hidden_states: torch.Tensor, *args: object, **kwargs: object):
        """Run the dendritic FFN on all leading token/batch dimensions."""
        return super().forward(hidden_states, *args, **kwargs)

    def parameter_estimate(self) -> dict[str, int]:
        """Return a parameter estimate matching this module's configuration."""
        reactivation_params = sum(
            param.numel()
            for branch_layer in self.core.branch_layers
            for param in branch_layer.reactivation.parameters()
        )
        reactivation_units = sum(
            int(branch_layer.branch_config.output_dim)
            for branch_layer in self.core.branch_layers
        )
        params_per_unit = (
            reactivation_params // reactivation_units
            if reactivation_units and reactivation_params % reactivation_units == 0
            else 0
        )
        return estimate_dendritic_ffn_params(
            hidden_size=self.hidden_size,
            dendritic_units=self.dendritic_units,
            branch_factors=self.branch_factors,
            synapses_per_branch=self.synapses_per_branch,
            candidate_size=self.indexed_candidate_size,
            input_transform=self.input_transform,
            topk_type=self.topk_type,
            output_topk=self.output_topk,
            output_bias=(
                isinstance(self.output_projection, nn.Linear)
                and self.output_projection.bias is not None
            ),
            somatic_synapses=self.somatic_synapses,
            reactivation_parameters_per_unit=params_per_unit,
            pre_norm=self.uses_pre_norm,
        )


def _offset_population_indexed_seeds(
    population_network: dict[str, Any], offset: int
) -> dict[str, Any]:
    """Return a deep copy with every configured ``indexed_seed`` shifted.

    ``GatedPopulationNetworkFFNReplacement`` builds its gate and value cores
    from one compiled ``population_network`` payload.  ``IndexedSparseLinear``
    samples ``connection_indices`` deterministically from ``indexed_seed`` (as
    namespaced by layer, population, pathway, and depth -- all identical
    between the two cores), so without an offset the value core starts with
    bit-identical supports to the gate core.  The teacher SwiGLU this family
    mirrors has independent gate/up supports, and the legacy
    ``GatedDendriticFFNReplacement`` already offsets its value path
    (``value_seed_offset``); this helper gives the canonical family the same
    guarantee.  Seeds left unset (``None``) stay unset: those layers already
    draw from the global RNG and are not deterministic between the cores.
    """

    def _walk(node: Any) -> Any:
        if isinstance(node, dict):
            out: dict[str, Any] = {}
            for key, value in node.items():
                if key == "indexed_seed" and isinstance(value, int):
                    out[key] = int(value) + int(offset)
                else:
                    out[key] = _walk(value)
            return out
        if isinstance(node, list):
            return [_walk(item) for item in node]
        return node

    return _walk(deepcopy(dict(population_network)))


def _build_population_network_core(
    population_network: dict[str, Any],
    *,
    input_dim: int,
) -> nn.Module:
    """Build the canonical population-network implementation without flattening it.

    Transformer configs historically accepted a ``population_network`` block
    but translated its first population into a single ``DendriNet``.  This
    helper deliberately preserves every configured layer, population,
    connection, polarity, and per-population morphology.
    """

    from dendritic_modeling.networks.architectures.factory import get_architecture

    payload = deepcopy(dict(population_network))
    # ``input_transform`` is a transformer replacement adapter option, not a
    # PopulationNetworkConfig field.  Accepting it here keeps generated configs
    # backward compatible while preventing an unexpected-key constructor error.
    payload.pop("input_transform", None)
    core = get_architecture(
        "population_network",
        {"population_network": payload},
        input_dim=int(input_dim),
    )
    if bool(getattr(core, "is_recurrent", False)):
        raise ValueError(
            "PopulationNetworkFFNReplacement currently requires a feedforward "
            "population_network. Recurrent population dynamics must define "
            "their transformer token/state semantics explicitly."
        )
    return core


def _validate_biological_population_contract(
    population_network: dict[str, Any],
    *,
    input_transform: str,
) -> None:
    """Reject nominal biological cells that contain signed hidden shortcuts."""

    if str(input_transform) not in {"relu", "signed_split"}:
        raise ValueError(
            "biological_neuron=true requires input_transform='relu' or "
            "'signed_split' so every dendritic input channel is nonnegative"
        )
    layers = list(population_network.get("layers", []) or [])
    if not layers:
        raise ValueError("biological population_network requires at least one layer")
    nonnegative_reactivations = {
        "param_tanh",
        "param_tanh_only_m",
        "param_relu",
        "relu",
        "sigmoid",
        "softplus",
    }
    for layer_index, layer in enumerate(layers):
        defaults = dict(layer.get("population_defaults", {}) or {})
        for population in list(layer.get("populations", []) or []):
            overrides = dict(population.get("population", {}) or {})
            weight_transform = str(
                overrides.get(
                    "weight_transform", defaults.get("weight_transform", "exp")
                )
            ).lower()
            if weight_transform not in POSITIVE_WEIGHT_TRANSFORMS:
                raise ValueError(
                    "biological_neuron=true requires positive effective "
                    f"weights; layer {layer_index} population "
                    f"{population.get('name')!r} uses {weight_transform!r}"
                )
            reactivate = bool(
                overrides.get("reactivate", defaults.get("reactivate", True))
            )
            reactivation = str(
                overrides.get(
                    "reactivation_type",
                    defaults.get("reactivation_type", "param_tanh"),
                )
            ).lower()
            if not reactivate or reactivation not in nonnegative_reactivations:
                raise ValueError(
                    "biological_neuron=true requires a nonnegative population "
                    f"reactivation; layer {layer_index} population "
                    f"{population.get('name')!r} has reactivate={reactivate}, "
                    f"reactivation_type={reactivation!r}"
                )


class PopulationNetworkFFNReplacement(TokenReplacementCell):
    """Transformer FFN replacement backed by the real ``PopulationNetwork``.

    Unlike the compatibility translator, this class does not select one
    population and discard the rest of the graph.  The complete configured
    population network is retained, including explicit E/I pathways,
    per-population branch factors, shunting/additive rules, and rewiring or
    dense-to-sparse synapses.
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        output_size: int | None = None,
        population_network: dict[str, Any],
        input_transform: str = "signed_split",
        output_rank: int | None = None,
        output_topk: int | None = None,
        output_topology_mode: str = "indexed",
        output_seed: int | None = None,
        output_candidate_size: int | None = None,
        output_selection: str = "standard",
        output_noise_level: float = 0.0,
        output_temperature: float = 0.5,
        output_ultrafast: bool = True,
        output_variance_momentum: float = 0.9,
        output_rewire_frequency: int = 100,
        output_rewire_quantile: float = 0.05,
        output_rewire_until_step: int | None = None,
        output_dense_to_sparse_initial_density: float = 1.0,
        output_dense_to_sparse_start_step: int = 0,
        output_dense_to_sparse_end_step: int = 1000,
        output_dense_to_sparse_update_interval: int = 1,
        output_dense_to_sparse_schedule: str = "cubic",
        output_dense_to_sparse_freeze_on_end: bool = False,
        output_dense_to_sparse_advance_on_forward: bool = False,
        output_dense_to_sparse_prune_metric: str = "magnitude",
        output_chunk_size: int = 2048,
        output_index_dtype: str = "auto",
        output_workspace_mb: float | None = None,
        output_projection_backend: str = "recompute",
        output_bias: bool = False,
        output_init_std: float = 0.02,
        output_scale: float = 1.0,
        pre_norm: bool = False,
        biological_neuron: bool | None = None,
        teacher_support_metric: str = "",
        affine_bypass_topk: int | None = None,
        affine_bypass_rank: int | None = None,
        affine_bypass_topology_mode: str | None = None,
        affine_bypass_seed: int | None = None,
    ):
        hidden_size = int(hidden_size)
        output_size = hidden_size if output_size is None else int(output_size)
        if output_size < 1:
            raise ValueError("output_size must be positive")
        if biological_neuron is True:
            _validate_biological_population_contract(
                population_network,
                input_transform=input_transform,
            )
        adapted_dim = transformed_feature_dim(hidden_size, input_transform)
        core = _build_population_network_core(
            population_network,
            input_dim=adapted_dim,
        )
        super().__init__(
            input_dim=hidden_size,
            output_dim=output_size,
            input_transform=input_transform,
            core=core,
            core_output_dim=int(core.output_dim),
            output_bias=output_bias,
            output_init_std=output_init_std,
            output_scale=output_scale,
            pre_norm=pre_norm,
        )
        projection_factory = (
            make_positive_ei_output_projection
            if biological_neuron is True
            else make_output_projection
        )
        self.output_projection = projection_factory(
            int(core.output_dim),
            output_size,
            rank=output_rank,
            topk=output_topk,
            topology_mode=output_topology_mode,
            seed=output_seed,
            candidate_size=output_candidate_size,
            selection=output_selection,
            noise_level=output_noise_level,
            temperature=output_temperature,
            ultrafast=output_ultrafast,
            variance_momentum=output_variance_momentum,
            rewire_frequency=output_rewire_frequency,
            rewire_quantile=output_rewire_quantile,
            rewire_until_step=output_rewire_until_step,
            dense_to_sparse_initial_density=output_dense_to_sparse_initial_density,
            dense_to_sparse_start_step=output_dense_to_sparse_start_step,
            dense_to_sparse_end_step=output_dense_to_sparse_end_step,
            dense_to_sparse_update_interval=output_dense_to_sparse_update_interval,
            dense_to_sparse_schedule=output_dense_to_sparse_schedule,
            dense_to_sparse_freeze_on_end=output_dense_to_sparse_freeze_on_end,
            dense_to_sparse_advance_on_forward=output_dense_to_sparse_advance_on_forward,
            dense_to_sparse_prune_metric=output_dense_to_sparse_prune_metric,
            output_chunk_size=output_chunk_size,
            index_dtype=output_index_dtype,
            workspace_mb=output_workspace_mb,
            projection_backend=output_projection_backend,
            bias=output_bias,
            init_std=output_init_std,
            **({"weight_transform": "softplus"} if biological_neuron is True else {}),
        )
        self.affine_bypass = None
        if affine_bypass_topk is not None or affine_bypass_rank is not None:
            self.affine_bypass = projection_factory(
                adapted_dim,
                output_size,
                rank=affine_bypass_rank,
                topk=affine_bypass_topk,
                topology_mode=(
                    output_topology_mode
                    if affine_bypass_topology_mode is None
                    else affine_bypass_topology_mode
                ),
                seed=(
                    (None if output_seed is None else int(output_seed) + 2_000_003)
                    if affine_bypass_seed is None
                    else int(affine_bypass_seed)
                ),
                candidate_size=output_candidate_size,
                selection=output_selection,
                noise_level=output_noise_level,
                temperature=output_temperature,
                ultrafast=output_ultrafast,
                variance_momentum=output_variance_momentum,
                rewire_frequency=output_rewire_frequency,
                rewire_quantile=output_rewire_quantile,
                rewire_until_step=output_rewire_until_step,
                dense_to_sparse_initial_density=(
                    output_dense_to_sparse_initial_density
                ),
                dense_to_sparse_start_step=output_dense_to_sparse_start_step,
                dense_to_sparse_end_step=output_dense_to_sparse_end_step,
                dense_to_sparse_update_interval=(
                    output_dense_to_sparse_update_interval
                ),
                dense_to_sparse_schedule=output_dense_to_sparse_schedule,
                dense_to_sparse_freeze_on_end=(output_dense_to_sparse_freeze_on_end),
                dense_to_sparse_advance_on_forward=(
                    output_dense_to_sparse_advance_on_forward
                ),
                dense_to_sparse_prune_metric=output_dense_to_sparse_prune_metric,
                output_chunk_size=output_chunk_size,
                index_dtype=output_index_dtype,
                workspace_mb=output_workspace_mb,
                projection_backend=output_projection_backend,
                bias=False,
                init_std=output_init_std,
                **(
                    {"weight_transform": "softplus"}
                    if biological_neuron is True
                    else {}
                ),
            )
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.population_network_config = deepcopy(dict(population_network))
        self.output_rank = None if output_rank is None else int(output_rank)
        self.output_topk = None if output_topk is None else int(output_topk)
        self.biological_neuron = biological_neuron
        self.teacher_support_metric = str(teacher_support_metric).strip().lower()
        if self.teacher_support_metric and (
            self.teacher_support_metric not in TEACHER_TOPOLOGY_METRICS
        ):
            raise ValueError(
                "teacher_support_metric must be one of "
                f"{TEACHER_TOPOLOGY_METRICS} or empty"
            )
        self.reference_intermediate_size: int | None = None

    def forward(self, hidden_states: torch.Tensor, *args: object, **kwargs: object):
        del args, kwargs
        normalized = self.pre_norm(hidden_states)
        adapted = self.input_adapter(normalized)
        core_output = self.output_projection(self.run_core(adapted))
        if self.affine_bypass is not None:
            core_output = core_output + self.affine_bypass(adapted)
        return preserve_runtime_tensor_contract(
            core_output * self.output_scale,
            hidden_states,
            boundary=type(self).__name__,
        )

    @classmethod
    def from_mlp(
        cls,
        mlp: nn.Module,
        **kwargs: object,
    ) -> PopulationNetworkFFNReplacement:
        hidden_size, intermediate_size = _infer_mlp_dims(mlp)
        replacement = cls(hidden_size=hidden_size, **kwargs)
        replacement.reference_intermediate_size = int(intermediate_size)
        return replacement

    def run_core(self, adapted: torch.Tensor) -> torch.Tensor:
        leading_shape = adapted.shape[:-1]
        flat = adapted.reshape(-1, adapted.shape[-1])
        output = self.core(flat)
        return output.reshape(*leading_shape, output.shape[-1])

    def parameter_estimate(self) -> dict[str, int]:
        stored_core = sum(param.numel() for param in self.core.parameters())
        effective_core = int(self.core.get_effective_params())
        projection, active_projection = output_projection_parameter_counts(
            self.output_projection
        )
        bypass, active_bypass = (
            (0, 0)
            if self.affine_bypass is None
            else output_projection_parameter_counts(self.affine_bypass)
        )
        pre_norm = sum(param.numel() for param in self.pre_norm.parameters())
        return {
            "stored_core": int(stored_core),
            "active_core": int(effective_core),
            "output_projection": int(projection),
            "active_output_projection": int(active_projection),
            "affine_bypass": int(bypass),
            "active_affine_bypass": int(active_bypass),
            "pre_norm": int(pre_norm),
            "stored_total": int(stored_core + projection + bypass + pre_norm),
            "active_total": int(
                effective_core + active_projection + active_bypass + pre_norm
            ),
        }


class ZeroFFNResidualBranch(nn.Module):
    """Exact zero contribution for an FFN residual branch in a collapsed span."""

    runtime_tensor_contract_schema = RUNTIME_TENSOR_CONTRACT_SCHEMA

    def __init__(
        self,
        *,
        layer_index: int,
        span_index: int,
        span_layers: Sequence[int],
    ) -> None:
        super().__init__()
        self.layer_index = int(layer_index)
        self.collapsed_span_index = int(span_index)
        self.collapsed_span_layers = tuple(int(layer) for layer in span_layers)

    def forward(self, hidden_states: torch.Tensor, *args: object, **kwargs: object):
        del args, kwargs
        return torch.zeros_like(hidden_states)


class CollapsedPopulationNetworkSpanExit(nn.Module):
    """One PopulationNetwork application at a collapsed FFN span's exit."""

    runtime_tensor_contract_schema = RUNTIME_TENSOR_CONTRACT_SCHEMA

    def __init__(
        self,
        span_cell: nn.Module,
        *,
        span_index: int,
        span_layers: Sequence[int],
        post_mlp_norm_attr: str = "",
    ) -> None:
        super().__init__()
        if not isinstance(
            span_cell,
            (PopulationNetworkFFNReplacement, GatedPopulationNetworkFFNReplacement),
        ):
            raise TypeError(
                "collapsed replacement spans require a canonical "
                "PopulationNetwork FFN replacement cell"
            )
        layers = tuple(int(layer) for layer in span_layers)
        if len(layers) < 2:
            raise ValueError("a collapsed replacement span needs at least two layers")
        self.span_cell = span_cell
        self.collapsed_span_index = int(span_index)
        self.collapsed_span_layers = layers
        self.collapsed_span_start = int(layers[0])
        self.collapsed_span_exit = int(layers[-1])
        self._collapsed_metadata = {
            "schema": "dendritic_collapsed_replacement_span/v1",
            "span_index": self.collapsed_span_index,
            "span_layers": list(layers),
            "zero_ffn_layers": list(layers[:-1]),
            "exit_layer": self.collapsed_span_exit,
            "cell_application_count": 1,
            "execution_semantics": "one_cell_applied_only_at_span_exit",
            "zero_branch_implementation": "torch_zeros_like_tensor_write",
            "runtime_claim_status": "requires_measured_family_specific_benchmark",
        }
        if post_mlp_norm_attr:
            self._collapsed_metadata.update(
                post_mlp_norm_attr=post_mlp_norm_attr,
                post_mlp_norm_removed_layers=list(layers),
                cell_output_boundary="post_mlp_norm_residual_branch",
            )

    @property
    def teacher_support_metric(self) -> str:
        return str(getattr(self.span_cell, "teacher_support_metric", ""))

    @property
    def selection_manifest(self) -> dict[str, Any]:
        manifest = deepcopy(
            dict(getattr(self.span_cell, "selection_manifest", {}) or {})
        )
        manifest["layer_index"] = self.collapsed_span_exit
        manifest["collapsed_replacement_span"] = dict(self._collapsed_metadata)
        return manifest

    @property
    def compiled_replacement_plan(self) -> dict[str, Any]:
        return getattr(self.span_cell, "compiled_replacement_plan", {})

    @property
    def teacher_topk_diagnostics(self) -> dict[str, Any]:
        return getattr(self.span_cell, "teacher_topk_diagnostics", {})

    def parameter_estimate(self) -> dict[str, int]:
        return dict(self.span_cell.parameter_estimate())

    def forward(self, *args: object, **kwargs: object) -> torch.Tensor:
        return self.span_cell(*args, **kwargs)


def unwrap_collapsed_population_replacement(module: nn.Module) -> nn.Module:
    """Return the PopulationNetwork cell owned by a collapsed-span exit."""

    if isinstance(module, CollapsedPopulationNetworkSpanExit):
        return module.span_cell
    return module


class TiedPopulationNetworkFFNSite(nn.Module):
    """Site-specific view of one physically shared PopulationNetwork FFN.

    A distinct wrapper is installed at every transformer site so forward hooks
    and replacement records remain unambiguous.  ``shared_core`` is the exact
    same module object in every wrapper in the group; consequently parameters,
    buffers, sparse topology state, and gradients are shared rather than
    copied.  The shared core is still applied independently at every FFN site;
    this is parameter tying, not a single cell that replaces a multi-layer
    teacher span.  The wrapper intentionally accepts only canonical
    PopulationNetwork replacement cores.
    """

    runtime_tensor_contract_schema = RUNTIME_TENSOR_CONTRACT_SCHEMA

    def __init__(
        self,
        shared_core: nn.Module,
        *,
        layer_index: int,
        group_index: int,
        group_layers: Sequence[int],
    ) -> None:
        super().__init__()
        if not isinstance(
            shared_core,
            (PopulationNetworkFFNReplacement, GatedPopulationNetworkFFNReplacement),
        ):
            raise TypeError(
                "parameter-tied replacement groups require a canonical "
                "PopulationNetwork FFN replacement core"
            )
        layers = tuple(int(value) for value in group_layers)
        if int(layer_index) not in layers:
            raise ValueError("site layer must be a member of its parameter-tied group")
        self.shared_core = shared_core
        self.site_layer_index = int(layer_index)
        self.tied_group_index = int(group_index)
        self.tied_group_layers = layers
        self.tied_group_leader = int(layers[0])
        self._shared_selection_metadata = {
            "schema": "dendritic_parameter_tied_replacement_group/v1",
            "group_index": self.tied_group_index,
            "leader_layer": self.tied_group_leader,
            "layers": list(self.tied_group_layers),
            "role": "alias" if self.is_tied_alias else "leader",
            "physical_parameter_owner": self.tied_group_leader,
            "execution_semantics": "shared_cell_applied_at_every_ffn_site",
            "reduces_cell_applications": False,
        }

    @property
    def is_tied_alias(self) -> bool:
        return self.site_layer_index != self.tied_group_leader

    @property
    def teacher_support_metric(self) -> str:
        return str(getattr(self.shared_core, "teacher_support_metric", ""))

    @property
    def selection_manifest(self) -> dict[str, Any]:
        manifest = deepcopy(
            dict(getattr(self.shared_core, "selection_manifest", {}) or {})
        )
        manifest["layer_index"] = self.site_layer_index
        manifest["parameter_tied_replacement"] = dict(self._shared_selection_metadata)
        return manifest

    @property
    def compiled_replacement_plan(self) -> dict[str, Any]:
        return getattr(self.shared_core, "compiled_replacement_plan", {})

    @property
    def teacher_topk_diagnostics(self) -> dict[str, Any]:
        return getattr(self.shared_core, "teacher_topk_diagnostics", {})

    def parameter_estimate(self) -> dict[str, int]:
        return dict(self.shared_core.parameter_estimate())

    def forward(self, *args: object, **kwargs: object) -> torch.Tensor:
        return self.shared_core(*args, **kwargs)


def unwrap_shared_population_replacement(module: nn.Module) -> nn.Module:
    """Return the physical cell underlying a site wrapper, if present."""

    if isinstance(module, TiedPopulationNetworkFFNSite):
        return module.shared_core
    return module


class GatedPopulationNetworkFFNReplacement(nn.Module):
    """SwiGLU/GELU-style replacement with two full population-network paths."""

    runtime_tensor_contract_schema = RUNTIME_TENSOR_CONTRACT_SCHEMA

    def __init__(
        self,
        *,
        hidden_size: int,
        output_size: int | None = None,
        population_network: dict[str, Any],
        input_transform: str = "identity",
        gate_activation: str = "silu",
        output_rank: int | None = None,
        output_topk: int | None = None,
        output_topology_mode: str = "indexed",
        output_seed: int | None = None,
        output_candidate_size: int | None = None,
        output_selection: str = "standard",
        output_noise_level: float = 0.0,
        output_temperature: float = 0.5,
        output_ultrafast: bool = True,
        output_variance_momentum: float = 0.9,
        output_rewire_frequency: int = 100,
        output_rewire_quantile: float = 0.05,
        output_rewire_until_step: int | None = None,
        output_dense_to_sparse_initial_density: float = 1.0,
        output_dense_to_sparse_start_step: int = 0,
        output_dense_to_sparse_end_step: int = 1000,
        output_dense_to_sparse_update_interval: int = 1,
        output_dense_to_sparse_schedule: str = "cubic",
        output_dense_to_sparse_freeze_on_end: bool = False,
        output_dense_to_sparse_advance_on_forward: bool = False,
        output_dense_to_sparse_prune_metric: str = "magnitude",
        output_chunk_size: int = 2048,
        output_index_dtype: str = "auto",
        output_workspace_mb: float | None = None,
        output_projection_backend: str = "recompute",
        output_bias: bool = False,
        output_init_std: float = 0.02,
        output_scale: float = 1.0,
        pre_norm: bool = False,
        biological_neuron: bool | None = None,
        teacher_support_metric: str = "",
        affine_bypass_topk: int | None = None,
        affine_bypass_rank: int | None = None,
        affine_bypass_topology_mode: str | None = None,
        affine_bypass_seed: int | None = None,
        value_seed_offset: int = 1,
    ):
        super().__init__()
        hidden_size = int(hidden_size)
        output_size = hidden_size if output_size is None else int(output_size)
        if output_size < 1:
            raise ValueError("output_size must be positive")
        if biological_neuron is True:
            _validate_biological_population_contract(
                population_network,
                input_transform=input_transform,
            )
        adapted_dim = transformed_feature_dim(hidden_size, input_transform)
        self.pre_norm = nn.LayerNorm(hidden_size) if pre_norm else nn.Identity()
        self.input_adapter = NonNegativeInputAdapter(hidden_size, input_transform)
        self.gate_core = _build_population_network_core(
            population_network,
            input_dim=adapted_dim,
        )
        self.value_seed_offset = int(value_seed_offset)
        self.value_core = _build_population_network_core(
            _offset_population_indexed_seeds(
                population_network, self.value_seed_offset
            ),
            input_dim=adapted_dim,
        )
        if int(self.gate_core.output_dim) != int(self.value_core.output_dim):
            raise ValueError(
                "gate and value population networks must have equal output dims"
            )
        projection_factory = (
            make_positive_ei_output_projection
            if biological_neuron is True
            else make_output_projection
        )
        self.output_projection = projection_factory(
            int(self.gate_core.output_dim),
            output_size,
            rank=output_rank,
            topk=output_topk,
            topology_mode=output_topology_mode,
            seed=output_seed,
            candidate_size=output_candidate_size,
            selection=output_selection,
            noise_level=output_noise_level,
            temperature=output_temperature,
            ultrafast=output_ultrafast,
            variance_momentum=output_variance_momentum,
            rewire_frequency=output_rewire_frequency,
            rewire_quantile=output_rewire_quantile,
            rewire_until_step=output_rewire_until_step,
            dense_to_sparse_initial_density=output_dense_to_sparse_initial_density,
            dense_to_sparse_start_step=output_dense_to_sparse_start_step,
            dense_to_sparse_end_step=output_dense_to_sparse_end_step,
            dense_to_sparse_update_interval=output_dense_to_sparse_update_interval,
            dense_to_sparse_schedule=output_dense_to_sparse_schedule,
            dense_to_sparse_freeze_on_end=output_dense_to_sparse_freeze_on_end,
            dense_to_sparse_advance_on_forward=output_dense_to_sparse_advance_on_forward,
            dense_to_sparse_prune_metric=output_dense_to_sparse_prune_metric,
            output_chunk_size=output_chunk_size,
            index_dtype=output_index_dtype,
            workspace_mb=output_workspace_mb,
            projection_backend=output_projection_backend,
            bias=output_bias,
            init_std=output_init_std,
            **({"weight_transform": "softplus"} if biological_neuron is True else {}),
        )
        self.affine_bypass = None
        if affine_bypass_topk is not None or affine_bypass_rank is not None:
            self.affine_bypass = projection_factory(
                adapted_dim,
                output_size,
                rank=affine_bypass_rank,
                topk=affine_bypass_topk,
                topology_mode=(
                    output_topology_mode
                    if affine_bypass_topology_mode is None
                    else affine_bypass_topology_mode
                ),
                seed=(
                    (None if output_seed is None else int(output_seed) + 2_000_003)
                    if affine_bypass_seed is None
                    else int(affine_bypass_seed)
                ),
                candidate_size=output_candidate_size,
                selection=output_selection,
                noise_level=output_noise_level,
                temperature=output_temperature,
                ultrafast=output_ultrafast,
                variance_momentum=output_variance_momentum,
                rewire_frequency=output_rewire_frequency,
                rewire_quantile=output_rewire_quantile,
                rewire_until_step=output_rewire_until_step,
                dense_to_sparse_initial_density=(
                    output_dense_to_sparse_initial_density
                ),
                dense_to_sparse_start_step=output_dense_to_sparse_start_step,
                dense_to_sparse_end_step=output_dense_to_sparse_end_step,
                dense_to_sparse_update_interval=(
                    output_dense_to_sparse_update_interval
                ),
                dense_to_sparse_schedule=output_dense_to_sparse_schedule,
                dense_to_sparse_freeze_on_end=(output_dense_to_sparse_freeze_on_end),
                dense_to_sparse_advance_on_forward=(
                    output_dense_to_sparse_advance_on_forward
                ),
                dense_to_sparse_prune_metric=output_dense_to_sparse_prune_metric,
                output_chunk_size=output_chunk_size,
                index_dtype=output_index_dtype,
                workspace_mb=output_workspace_mb,
                projection_backend=output_projection_backend,
                bias=False,
                init_std=output_init_std,
                **(
                    {"weight_transform": "softplus"}
                    if biological_neuron is True
                    else {}
                ),
            )
        normalized_activation = str(gate_activation).lower()
        if normalized_activation not in {
            "silu",
            "swish",
            "gelu",
            "relu",
            "identity",
            "linear",
            "none",
        }:
            raise ValueError(f"Unsupported gate_activation {gate_activation!r}")
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.output_dim = output_size
        self.input_transform = str(input_transform)
        self.gate_activation = normalized_activation
        self.output_scale = float(output_scale)
        self.output_rank = None if output_rank is None else int(output_rank)
        self.output_topk = None if output_topk is None else int(output_topk)
        self.biological_neuron = biological_neuron
        self.teacher_support_metric = str(teacher_support_metric).strip().lower()
        if self.teacher_support_metric and (
            self.teacher_support_metric not in TEACHER_TOPOLOGY_METRICS
        ):
            raise ValueError(
                "teacher_support_metric must be one of "
                f"{TEACHER_TOPOLOGY_METRICS} or empty"
            )
        self.population_network_config = deepcopy(dict(population_network))
        self.reference_intermediate_size: int | None = None

    @classmethod
    def from_mlp(
        cls,
        mlp: nn.Module,
        **kwargs: object,
    ) -> GatedPopulationNetworkFFNReplacement:
        hidden_size, intermediate_size = _infer_mlp_dims(mlp)
        replacement = cls(hidden_size=hidden_size, **kwargs)
        replacement.reference_intermediate_size = int(intermediate_size)
        return replacement

    def _activate_gate(self, x: torch.Tensor) -> torch.Tensor:
        if self.gate_activation in {"silu", "swish"}:
            return F.silu(x)
        if self.gate_activation == "gelu":
            return F.gelu(x)
        if self.gate_activation == "relu":
            return F.relu(x)
        return x

    @staticmethod
    def _run_population_core(core: nn.Module, adapted: torch.Tensor) -> torch.Tensor:
        leading_shape = adapted.shape[:-1]
        flat = adapted.reshape(-1, adapted.shape[-1])
        output = core(flat)
        return output.reshape(*leading_shape, output.shape[-1])

    def forward(self, hidden_states: torch.Tensor, *args: object, **kwargs: object):
        del args, kwargs
        if hidden_states.shape[-1] != self.hidden_size:
            raise ValueError(
                f"Expected hidden size {self.hidden_size}, got {hidden_states.shape[-1]}"
            )
        adapted = self.input_adapter(self.pre_norm(hidden_states))
        gate = self._run_population_core(self.gate_core, adapted)
        value = self._run_population_core(self.value_core, adapted)
        output = self.output_projection(self._activate_gate(gate) * value)
        if self.affine_bypass is not None:
            output = output + self.affine_bypass(adapted)
        return preserve_runtime_tensor_contract(
            output * self.output_scale,
            hidden_states,
            boundary=type(self).__name__,
        )

    def parameter_estimate(self) -> dict[str, int]:
        stored_core = sum(
            param.numel()
            for core in (self.gate_core, self.value_core)
            for param in core.parameters()
        )
        active_core = int(self.gate_core.get_effective_params()) + int(
            self.value_core.get_effective_params()
        )
        projection, active_projection = output_projection_parameter_counts(
            self.output_projection
        )
        bypass, active_bypass = (
            (0, 0)
            if self.affine_bypass is None
            else output_projection_parameter_counts(self.affine_bypass)
        )
        pre_norm = sum(param.numel() for param in self.pre_norm.parameters())
        return {
            "stored_core": int(stored_core),
            "active_core": int(active_core),
            "output_projection": int(projection),
            "active_output_projection": int(active_projection),
            "affine_bypass": int(bypass),
            "active_affine_bypass": int(active_bypass),
            "pre_norm": int(pre_norm),
            "stored_total": int(stored_core + projection + bypass + pre_norm),
            "active_total": int(
                active_core + active_projection + active_bypass + pre_norm
            ),
        }


class GatedDendriticFFNReplacement(nn.Module):
    """Two-path dendritic FFN with a SwiGLU-style multiplicative bottleneck.

    A conventional Llama/OLMo MLP learns separate gate and value projections,
    multiplies them after the gate activation, and projects the result back to
    the residual stream.  The original :class:`DendriticFFNReplacement` has a
    single dendritic path, so it cannot express that factorization directly.
    This variant uses two independently wired DendriNet cores while sharing the
    input adapter and the final output projection::

        y = output_projection(activation(gate_core(x)) * value_core(x))

    It intentionally reuses the same dendritic configuration for both paths so
    TopK, reactivation calibration, and rewiring infrastructure work unchanged.
    """

    runtime_tensor_contract_schema = RUNTIME_TENSOR_CONTRACT_SCHEMA

    def __init__(
        self,
        *,
        hidden_size: int,
        dendritic_units: int,
        gate_activation: str = "silu",
        value_seed_offset: int = 1,
        teacher_topk_init: bool = False,
        teacher_topk_metric: str = "weight",
        output_topk: int | None = None,
        **dendritic_kwargs: object,
    ):
        super().__init__()
        hidden_size = int(hidden_size)
        dendritic_units = int(dendritic_units)
        output_topk = None if output_topk is None else int(output_topk)
        gate_activation = str(gate_activation).lower()
        if gate_activation not in {
            "silu",
            "swish",
            "gelu",
            "relu",
            "identity",
            "linear",
            "none",
        }:
            raise ValueError(
                f"Unsupported gated dendritic activation {gate_activation!r}"
            )

        gate_kwargs = dict(dendritic_kwargs)
        value_kwargs = dict(dendritic_kwargs)
        indexed_seed = value_kwargs.get("indexed_seed")
        if indexed_seed is not None:
            value_kwargs["indexed_seed"] = int(indexed_seed) + int(value_seed_offset)

        gate_path = DendriticFFNReplacement(
            hidden_size=hidden_size,
            dendritic_units=dendritic_units,
            **gate_kwargs,
        )
        value_path = DendriticFFNReplacement(
            hidden_size=hidden_size,
            dendritic_units=dendritic_units,
            **value_kwargs,
        )
        # Keep only the modules used by this factorization.  The temporary path
        # projections are deliberately discarded rather than left as unused
        # parameters (which would also break strict DDP configurations).
        self.pre_norm = gate_path.pre_norm
        self.input_adapter = gate_path.input_adapter
        self.gate_core = gate_path.core
        self.value_core = value_path.core
        output_bias = bool(dendritic_kwargs.get("output_bias", False))
        output_seed = dendritic_kwargs.get("indexed_seed")
        if output_seed is not None:
            output_seed = int(output_seed) + 2 * int(value_seed_offset)
        self.output_projection = make_output_projection(
            dendritic_units,
            hidden_size,
            topk=output_topk,
            topology_mode="indexed",
            bias=output_bias,
            init_std=float(dendritic_kwargs.get("output_init_std", 0.02)),
            init_method=str(dendritic_kwargs.get("topk_init_method", "xavier_normal")),
            seed=output_seed,
            output_chunk_size=int(
                dendritic_kwargs.get("indexed_output_chunk_size", 2048)
            ),
            index_dtype=str(dendritic_kwargs.get("indexed_index_dtype", "int64")),
            workspace_mb=dendritic_kwargs.get("indexed_workspace_mb"),
            cache_transformed_weights=bool(
                dendritic_kwargs.get("indexed_cache_transformed_weights", False)
            ),
            recompute_backward=bool(
                dendritic_kwargs.get("indexed_recompute_backward", False)
            ),
            projection_backend=str(
                dendritic_kwargs.get("indexed_projection_backend", "eager")
            ),
            persistent_indices=bool(
                dendritic_kwargs.get("indexed_persistent_indices", True)
            ),
            init_mode=str(dendritic_kwargs.get("indexed_init_mode", "per_rank")),
        )

        self.hidden_size = hidden_size
        self.dendritic_units = dendritic_units
        self.input_transform = gate_path.input_transform
        self.output_scale = float(dendritic_kwargs.get("output_scale", 1.0))
        self.gate_activation = gate_activation
        self.value_seed_offset = int(value_seed_offset)
        self.teacher_topk_init = bool(teacher_topk_init)
        teacher_topk_metric = str(teacher_topk_metric).strip().lower()
        if teacher_topk_metric not in {
            "weight",
            "activation_weighted",
            "activation_weighted_structured",
        }:
            raise ValueError(
                "teacher_topk_metric must be 'weight', 'activation_weighted', "
                f"or 'activation_weighted_structured'; got {teacher_topk_metric!r}"
            )
        self.teacher_topk_metric = teacher_topk_metric
        self.output_topk = output_topk
        self.teacher_topk_diagnostics: dict[str, float | int] = {}
        self.uses_pre_norm = not isinstance(self.pre_norm, nn.Identity)
        self.reference_intermediate_size: int | None = None
        self._estimate_kwargs = {
            "hidden_size": hidden_size,
            "dendritic_units": dendritic_units,
            "branch_factors": list(gate_path.branch_factors),
            "synapses_per_branch": gate_path.synapses_per_branch,
            "candidate_size": gate_path.indexed_candidate_size,
            "input_transform": gate_path.input_transform,
            "topk_type": gate_path.topk_type,
            "output_bias": (
                isinstance(self.output_projection, nn.Linear)
                and self.output_projection.bias is not None
            ),
            "output_topk": self.output_topk,
            "somatic_synapses": gate_path.somatic_synapses,
            "reactivation_parameters_per_unit": self._reactivation_params_per_unit(),
            "pre_norm": self.uses_pre_norm,
        }

    @classmethod
    def from_mlp(
        cls,
        mlp: nn.Module,
        **kwargs: object,
    ) -> GatedDendriticFFNReplacement:
        """Construct a gated replacement by inferring the source hidden size."""
        hidden_size, intermediate_size = _infer_mlp_dims(mlp)
        replacement = cls(hidden_size=hidden_size, **kwargs)
        replacement.reference_intermediate_size = intermediate_size
        if replacement.teacher_topk_init:
            replacement.initialize_from_mlp(mlp)
        return replacement

    def _single_synapse_layer(self, core: DendriNet, *, path: str) -> nn.Module:
        if len(core.branch_layers) != 1:
            raise ValueError(
                "teacher_topk_init requires branch_factors=[] so each gated "
                f"{path} path has exactly one somatic synapse layer"
            )
        branch = core.branch_layers[0]
        synapse = branch.branch_excitation
        if synapse is None or not all(
            hasattr(synapse, name)
            for name in ("connection_indices", "pre_w", "K", "weight_transform")
        ):
            raise TypeError(
                "teacher_topk_init requires an indexed or indexed_rewire "
                f"synapse in the {path} path"
            )
        if not isinstance(branch.reactivation, nn.Identity):
            raise ValueError("teacher_topk_init requires reactivate=false")
        if str(synapse.weight_transform).lower() != "identity":
            raise ValueError("teacher_topk_init requires weight_transform='identity'")
        if getattr(branch, "use_shunting", False):
            raise ValueError(
                "teacher_topk_init requires use_shunting=false: the shunting "
                "rule divides the branch voltage by its total conductance, so "
                "a TopK-copied signed linear map cannot reproduce the dense "
                "teacher exactly even without inhibition."
            )
        return synapse

    def initialize_from_mlp(
        self,
        mlp: nn.Module,
        *,
        input_scales: torch.Tensor | None = None,
        intermediate_scales: torch.Tensor | None = None,
    ) -> dict[str, float | int]:
        """Magnitude-TopK initialize a one-level gated cell from a dense MLP.

        This is exact when ``K == hidden_size`` and otherwise initializes each
        gate/value row from its largest-magnitude teacher weights.  The dense
        down projection is copied without pruning. With calibration scales
        (``teacher_topk_metric='activation_weighted'``), rows select supports
        by |W_ij| * rms(x_j) instead of raw magnitude; ``input_scales`` covers
        gate/value inputs and ``intermediate_scales`` the down projection.
        Re-invoking with scales re-selects supports idempotently.
        """
        structured_selection = (
            getattr(self, "teacher_topk_metric", "weight")
            == "activation_weighted_structured"
        )
        if self.input_transform != "identity":
            raise ValueError("teacher_topk_init requires input_transform='identity'")
        if self.output_scale != 1.0:
            raise ValueError("teacher_topk_init requires output_scale=1.0")
        required = ("gate_proj", "up_proj", "down_proj")
        if not all(
            isinstance(getattr(mlp, name, None), nn.Linear) for name in required
        ):
            raise TypeError(
                "teacher_topk_init expects gate_proj, up_proj, and down_proj "
                "nn.Linear modules"
            )
        gate_synapse = self._single_synapse_layer(self.gate_core, path="gate")
        value_synapse = self._single_synapse_layer(self.value_core, path="value")
        gate_proj = mlp.gate_proj
        up_proj = mlp.up_proj
        down_proj = mlp.down_proj
        if self.dendritic_units != int(gate_proj.out_features):
            raise ValueError(
                "teacher_topk_init requires dendritic_units to equal the "
                f"teacher intermediate size ({gate_proj.out_features})"
            )
        for name, projection in (("gate_proj", gate_proj), ("up_proj", up_proj)):
            if projection.bias is not None:
                raise ValueError(
                    f"teacher_topk_init cannot represent a bias in teacher {name}"
                )
        gate_energy = copy_topk_weight_(
            gate_proj.weight,
            gate_synapse,
            path="gate_proj",
            input_scales=input_scales,
            structured=structured_selection,
        )
        value_energy = copy_topk_weight_(
            up_proj.weight,
            value_synapse,
            path="up_proj",
            input_scales=input_scales,
            structured=structured_selection,
        )
        if isinstance(self.output_projection, IndexedSparseLinear):
            if down_proj.bias is not None:
                raise ValueError(
                    "teacher_topk_init cannot represent a bias in teacher down_proj"
                )
            down_energy = copy_topk_weight_(
                down_proj.weight,
                self.output_projection,
                path="down_proj",
                input_scales=intermediate_scales,
                structured=structured_selection,
            )
        else:
            if tuple(down_proj.weight.shape) != tuple(
                self.output_projection.weight.shape
            ):
                raise ValueError(
                    "Teacher down_proj weight shape does not match the gated "
                    "dendritic output projection"
                )
            with torch.no_grad():
                self.output_projection.weight.copy_(
                    down_proj.weight.to(
                        device=self.output_projection.weight.device,
                        dtype=self.output_projection.weight.dtype,
                    )
                )
                if down_proj.bias is not None:
                    if self.output_projection.bias is None:
                        raise ValueError(
                            "Teacher down_proj has a bias but output_bias is disabled"
                        )
                    self.output_projection.bias.copy_(
                        down_proj.bias.to(
                            device=self.output_projection.bias.device,
                            dtype=self.output_projection.bias.dtype,
                        )
                    )
                elif self.output_projection.bias is not None:
                    self.output_projection.bias.zero_()
            down_energy = 1.0
        diagnostics: dict[str, float | int] = {
            "k": int(gate_synapse.K),
            "hidden_size": self.hidden_size,
            "intermediate_size": self.dendritic_units,
            "gate_weight_energy_retained": gate_energy,
            "value_weight_energy_retained": value_energy,
            "down_weight_energy_retained": down_energy,
            "activation_weighted_selection": int(input_scales is not None),
        }
        self.teacher_topk_diagnostics = diagnostics
        return diagnostics

    def _activate_gate(self, x: torch.Tensor) -> torch.Tensor:
        if self.gate_activation in {"silu", "swish"}:
            return F.silu(x)
        if self.gate_activation == "gelu":
            return F.gelu(x)
        if self.gate_activation == "relu":
            return F.relu(x)
        return x

    def _reactivation_params_per_unit(self) -> int:
        parameters = sum(
            param.numel()
            for branch_layer in self.gate_core.branch_layers
            for param in branch_layer.reactivation.parameters()
        )
        units = sum(
            int(branch_layer.branch_config.output_dim)
            for branch_layer in self.gate_core.branch_layers
        )
        return parameters // units if units and parameters % units == 0 else 0

    def forward(self, hidden_states: torch.Tensor, *args: object, **kwargs: object):
        """Run both dendritic paths on every leading token/batch dimension."""
        del args, kwargs
        if hidden_states.shape[-1] != self.hidden_size:
            raise ValueError(
                f"Expected hidden size {self.hidden_size}, "
                f"got {hidden_states.shape[-1]}"
            )
        adapted = self.input_adapter(self.pre_norm(hidden_states))
        gated = self._activate_gate(self.gate_core(adapted))
        value = self.value_core(adapted)
        out = self.output_projection(gated * value) * self.output_scale
        return preserve_runtime_tensor_contract(
            out,
            hidden_states,
            boundary=type(self).__name__,
        )

    def parameter_estimate(self) -> dict[str, int]:
        """Estimate stored/active parameters across both dendritic paths."""
        return estimate_gated_dendritic_ffn_params(**self._estimate_kwargs)


def _effective_params_for_dendrinet(module: nn.Module | None) -> int:
    if module is None or not isinstance(module, DendriNet):
        return 0
    effective = 0
    for branch_layer in module.branch_layers:
        for attr in ("branch_excitation", "branch_inhibition"):
            synapse = getattr(branch_layer, attr, None)
            if synapse is None:
                continue
            if hasattr(synapse, "K") and hasattr(synapse, "out_features"):
                effective += int(synapse.out_features) * int(synapse.K)
            else:
                effective += sum(param.numel() for param in synapse.parameters())
        block = getattr(branch_layer, "branches_to_output", None)
        if block is not None:
            effective += sum(param.numel() for param in block.parameters())
        reactivation = getattr(branch_layer, "reactivation", None)
        if reactivation is not None and not isinstance(reactivation, nn.Identity):
            effective += sum(param.numel() for param in reactivation.parameters())
    return effective


class LayerwiseEIStackReplacement(LayerwiseTokenReplacementStack):
    """A shared EI stack whose layers replace transformer MLP slots.

    OLMo/Llama-style MLPs are interleaved with attention and residual blocks.
    This module therefore exposes one callable slot per transformer depth:
    layer ``k`` receives the hidden state at transformer block ``k``, runs the
    corresponding EI layer with its own E/I populations, and projects the
    excitatory population back to the model hidden size for the residual path.
    """

    runtime_tensor_contract_schema = RUNTIME_TENSOR_CONTRACT_SCHEMA

    def __init__(
        self,
        *,
        hidden_size: int,
        num_layers: int,
        excitatory_cells: int = 200,
        inhibitory_cells: int = 50,
        excitatory_branch_factors: Sequence[int] = (2, 2),
        inhibitory_branch_factors: Sequence[int] = (2,),
        ee_synapses_per_branch: int = 32,
        ei_synapses_per_branch: int = 16,
        ie_synapses_per_branch: int = 16,
        ii_synapses_per_branch: int = 8,
        input_transform: str = "signed_split",
        use_shunting: bool = True,
        synapse_mode: str = "ei",
        reactivation_type: str = "param_tanh",
        reactivate: bool = True,
        topk_init_method: str = "xavier_normal",
        topk_noise_level: float = 0.0,
        topk_type: str = "indexed_rewire",
        topk_temperature: float = 0.25,
        topk_weight_norm_order: int | None = None,
        topk_gamma: float = 1.0,
        weight_transform: str = "softplus",
        indexed_seed: int | None = None,
        output_bias: bool = False,
        output_init_std: float = 0.02,
        output_scale: float = 1.0,
        pre_norm: bool = False,
        efficient_blocklinear: bool = False,
        use_noise: bool = False,
        compile_forward: bool = False,
        **kwargs: object,
    ):
        super().__init__(
            input_dim=hidden_size,
            output_dim=hidden_size,
            num_layers=num_layers,
            input_transform=input_transform,
            output_scale=output_scale,
            pre_norm=pre_norm,
        )
        if int(excitatory_cells) < 1:
            raise ValueError("excitatory_cells must be >= 1")
        if int(inhibitory_cells) < 1:
            raise ValueError("inhibitory_cells must be >= 1")

        self.hidden_size = int(hidden_size)
        self.excitatory_cells = int(excitatory_cells)
        self.inhibitory_cells = int(inhibitory_cells)
        self.efficient_blocklinear = bool(efficient_blocklinear)
        transformed_dim = self.adapted_input_dim
        self.layers = nn.ModuleList()
        self.output_projections = nn.ModuleList()
        extra_kwargs = dict(kwargs)
        for layer_idx in range(self.num_layers):
            layer_seed = (
                None if indexed_seed is None else int(indexed_seed) + int(layer_idx)
            )
            self.layers.append(
                ExcitationInhibitionLayer(
                    n_excitatory_cells=self.excitatory_cells,
                    n_inhibitory_cells=self.inhibitory_cells,
                    excitatory_branch_factors=list(excitatory_branch_factors),
                    inhibitory_branch_factors=list(inhibitory_branch_factors),
                    excitatory_input_dim=transformed_dim,
                    inhibitory_input_dim=transformed_dim,
                    ee_synapses_per_branch=int(ee_synapses_per_branch),
                    ei_synapses_per_branch=int(ei_synapses_per_branch),
                    ie_synapses_per_branch=int(ie_synapses_per_branch),
                    ii_synapses_per_branch=int(ii_synapses_per_branch),
                    build_inhibitory_cells=True,
                    inhibitory_network_type="dendritic",
                    reactivate=bool(reactivate),
                    topk_init_method=str(topk_init_method),
                    use_shunting=bool(use_shunting),
                    synapse_mode=str(synapse_mode),
                    reactivation_type=str(reactivation_type),
                    topk_noise_level=float(topk_noise_level),
                    topk_type=str(topk_type),
                    topk_temperature=float(topk_temperature),
                    topk_weight_norm_order=topk_weight_norm_order,
                    topk_gamma=float(topk_gamma),
                    weight_transform=str(weight_transform),
                    use_noise=bool(use_noise),
                    indexed_seed=layer_seed,
                    efficient_blocklinear=self.efficient_blocklinear,
                    compile_forward=bool(compile_forward),
                    **extra_kwargs,
                )
            )
            projection = make_linear_output_projection(
                self.excitatory_cells,
                self.hidden_size,
                bias=bool(output_bias),
                init_std=float(output_init_std),
            )
            self.output_projections.append(projection)

    def forward_layer(
        self, layer_index: int, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        idx = self._validate_layer_index(layer_index)
        original_shape = hidden_states.shape
        x = self.encode_layer_input(idx, hidden_states)
        x = x.reshape(-1, self.adapted_input_dim)
        excitatory, _ = self.layers[idx](x, x)
        output = self.decode_layer_output(idx, excitatory)
        return preserve_runtime_tensor_contract(
            output.reshape(original_shape),
            hidden_states,
            boundary=f"{type(self).__name__}.forward_layer",
        )

    def layer_parameter_estimate(self, layer_index: int) -> dict[str, int]:
        idx = int(layer_index)
        layer = self.layers[idx]
        projection = self.output_projections[idx]
        norm = self.pre_norm[idx]
        stored = (
            sum(param.numel() for param in layer.parameters())
            + sum(param.numel() for param in projection.parameters())
            + sum(param.numel() for param in norm.parameters())
        )
        active = (
            _effective_params_for_dendrinet(getattr(layer, "excitatory_cells", None))
            + _effective_params_for_dendrinet(getattr(layer, "inhibitory_cells", None))
            + sum(param.numel() for param in projection.parameters())
            + sum(param.numel() for param in norm.parameters())
        )
        return {
            "stored_total": int(stored),
            "active_total": int(active),
        }


class EIStackMLPSlot(nn.Module):
    """One OLMo MLP replacement slot backed by a shared EI stack."""

    runtime_tensor_contract_schema = RUNTIME_TENSOR_CONTRACT_SCHEMA

    def __init__(self, stack: LayerwiseEIStackReplacement, layer_index: int):
        super().__init__()
        self.stack = stack
        self.layer_index = int(layer_index)

    def forward(self, hidden_states: torch.Tensor, *args: object, **kwargs: object):
        del args, kwargs
        return self.stack.forward_layer(self.layer_index, hidden_states)

    def parameter_estimate(self) -> dict[str, int]:
        return self.stack.layer_parameter_estimate(self.layer_index)
