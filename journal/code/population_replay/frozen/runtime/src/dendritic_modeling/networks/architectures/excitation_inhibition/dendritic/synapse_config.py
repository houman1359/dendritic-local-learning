"""Typed configuration objects for dendritic branch synapse construction."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any


@dataclass(frozen=True)
class TopKConfig:
    """Sparse synapse selection and initialization options."""

    init_method: str = "xavier_normal"
    noise_level: float = 0.0
    type: str = "standard"
    weight_norm_order: int | None = None
    gamma: float = 1.0
    temperature: float = 0.5
    ultrafast: bool = (
        True  # ultrafast stochastic-topk default (set False for rank-probabilistic)
    )
    strategy: str = "none"


@dataclass(frozen=True)
class DenseToSparseConfig:
    """Annealed dense-to-sparse schedule options."""

    initial_density: float = 1.0
    initial_k: int | None = None
    start_step: int = 0
    end_step: int = 1000
    update_interval: int = 1
    schedule: str = "cubic"
    freeze_on_end: bool = False
    advance_on_forward: bool = True
    prune_metric: str = "weight"


@dataclass(frozen=True)
class IndexedSynapseConfig:
    """Indexed sparse-synapse and indexed-rewire options."""

    seed: int | None = None
    output_chunk_size: int = 2048
    candidate_size: int | None = None
    selection: str = "standard"
    rewire_frequency: int = 100
    rewire_quantile: float = 0.05
    rewire_until_step: int | None = None
    index_dtype: str = "int64"
    workspace_mb: float | None = None
    cache_transformed_weights: bool = False
    recompute_backward: bool = False
    persistent_indices: bool = True
    init_mode: str = "per_rank"
    projection_backend: str = "eager"
    # Hardware-structured support (see sample_structured_indices): defaults
    # of 1 preserve unstructured sampling exactly.
    support_group_rows: int = 1
    support_col_block: int = 1

    def __post_init__(self) -> None:
        from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.kernels import (
            normalize_indexed_projection_backend,
        )

        normalized = normalize_indexed_projection_backend(self.projection_backend)
        if self.recompute_backward and normalized not in {"eager", "recompute"}:
            raise ValueError(
                "indexed_recompute_backward cannot be combined with "
                f"indexed_projection_backend={normalized!r}"
            )
        object.__setattr__(self, "projection_backend", normalized)


@dataclass(frozen=True)
class ReactivationConfig:
    """Dendritic reactivation options."""

    enabled: bool = False
    type: str = "param_tanh"
    dendritic_activation: str | None = None
    init_m: float = 1
    init_b: float = 0.5
    init_policy: str = "analytical"
    # Optional override for the propagated soma output stage only: when set,
    # the FINAL branch layer of an excitatory DendriNet uses this activation
    # type (with soma_init_m / soma_init_b as its fixed-policy parameters)
    # while every other branch level keeps ``type``. ``None`` preserves the
    # historical uniform gate exactly. Inhibitory populations always keep the
    # base gate; the override targets the emitted excitatory code.
    soma_type: str | None = None
    soma_init_m: float = 1.0
    soma_init_b: float = 0.0
    occupancy_quantile_low: float | None = None
    occupancy_quantile_high: float | None = None
    occupancy_target_low: float | None = None
    occupancy_target_high: float | None = None
    calibration_min_quantile_width: float = 1e-3
    calibration_max_m: float = 50.0
    calibration_revert_on_invalid: bool = True
    sigma_aware_k: float = 0.25
    strategy: str = "none"
    memory_efficient: bool = False


@dataclass(frozen=True)
class MorphologyConfig:
    """Branch voltage, morphology, and output aggregation options."""

    use_shunting: bool = True
    use_additive_normalization: bool = False
    additive_mode: str = "raw"
    additive_tangent_n0: float | None = None
    additive_tangent_t0: float | None = None
    epsilon: float = 1e-8
    weight_transform: str = "exp"
    efficient_blocklinear: bool = False
    blocklinear_strategy: str = "none"


@dataclass(frozen=True)
class AdaptiveInitConfig:
    """Branch-layer initialization options."""

    dbl_init_method: str = "analytical_expectation"
    adaptive_initialization: bool = True
    adaptive_initialization_policy: str = "preserve_shunting_center"
    adaptive_target_conductance: float = 5.0
    initial_child_conductance: float = 1.0
    branch_factors: tuple[int, ...] | None = None
    initialization_seed: int | None = None
    initialization_namespace: str = ""


@dataclass(frozen=True)
class DeepSTConfig:
    """DeepST sparse-synapse options retained for compatibility."""

    use_noise: bool = True
    rewiring_mode: str = "global"
    rewire_frequency: int = 1
    synapses_per_branch: int | None = None
    sigma: float = 0.05
    excitatory_target_density: float = 0.1
    inhibitory_target_density: float = 0.1
    freeze_excitatory_connectivity: bool = False
    freeze_inhibitory_connectivity: bool = False
    init_method: str | None = None
    weight_threshold: float = 1e-6
    credit_trace_decay: float = 0.95
    credit_candidate_pool_size: int = 8
    credit_turnover_fraction: float = 0.1
    credit_weak_active_pool_fraction: float = 0.5
    credit_min_observations: int = 5
    credit_swap_margin: float = 0.0
    credit_force_turnover: bool = False
    credit_selection: str = "credit"
    credit_warmup_steps: int = 5


def _topk_config_from_kwargs(source: Mapping[str, Any]) -> TopKConfig:
    """Build TopK options from legacy branch-layer kwargs."""
    return TopKConfig(
        init_method=source.get("topk_init_method", "xavier_normal"),
        noise_level=source.get("topk_noise_level", 0.0),
        type=source.get("topk_type", "standard"),
        weight_norm_order=source.get("topk_weight_norm_order", None),
        gamma=source.get("topk_gamma", 1.0),
        temperature=source.get("topk_temperature", 0.5),
        ultrafast=source.get("topk_ultrafast", True),
        strategy=source.get("topk_strategy", "none"),
    )


def _dense_to_sparse_config_from_kwargs(
    source: Mapping[str, Any],
) -> DenseToSparseConfig:
    """Build dense-to-sparse schedule options from legacy kwargs."""
    return DenseToSparseConfig(
        initial_density=source.get("dense_to_sparse_initial_density", 1.0),
        initial_k=source.get("dense_to_sparse_initial_k", None),
        start_step=source.get("dense_to_sparse_start_step", 0),
        end_step=source.get("dense_to_sparse_end_step", 1000),
        update_interval=source.get("dense_to_sparse_update_interval", 1),
        schedule=source.get("dense_to_sparse_schedule", "cubic"),
        freeze_on_end=source.get("dense_to_sparse_freeze_on_end", False),
        advance_on_forward=source.get("dense_to_sparse_advance_on_forward", True),
        prune_metric=source.get("dense_to_sparse_prune_metric", "weight"),
    )


def _indexed_config_from_kwargs(source: Mapping[str, Any]) -> IndexedSynapseConfig:
    """Build indexed sparse-synapse options from legacy kwargs."""
    return IndexedSynapseConfig(
        seed=source.get("indexed_seed", None),
        output_chunk_size=source.get("indexed_output_chunk_size", 2048),
        candidate_size=source.get("indexed_candidate_size", None),
        selection=source.get("indexed_selection", "standard"),
        rewire_frequency=source.get("indexed_rewire_frequency", 100),
        rewire_quantile=source.get("indexed_rewire_quantile", 0.05),
        rewire_until_step=source.get("indexed_rewire_until_step", None),
        index_dtype=source.get("indexed_index_dtype", "int64"),
        workspace_mb=source.get("indexed_workspace_mb", None),
        cache_transformed_weights=source.get(
            "indexed_cache_transformed_weights", False
        ),
        recompute_backward=source.get("indexed_recompute_backward", False),
        projection_backend=source.get("indexed_projection_backend", "eager"),
        persistent_indices=source.get("indexed_persistent_indices", True),
        init_mode=source.get("indexed_init_mode", "per_rank"),
        support_group_rows=source.get("indexed_support_group_rows", 1),
        support_col_block=source.get("indexed_support_col_block", 1),
    )


def _reactivation_config_from_kwargs(source: Mapping[str, Any]) -> ReactivationConfig:
    """Build dendritic reactivation options from legacy kwargs."""
    return ReactivationConfig(
        enabled=source.get("reactivate", False),
        type=source.get("reactivation_type", "param_tanh"),
        dendritic_activation=source.get("dendritic_activation", None),
        init_m=source.get("reactivation_init_m", 1),
        init_b=source.get("reactivation_init_b", 0.5),
        init_policy=source.get("reactivation_init_policy", "analytical"),
        soma_type=source.get("reactivation_soma_type", None),
        soma_init_m=source.get("reactivation_soma_init_m", 1.0),
        soma_init_b=source.get("reactivation_soma_init_b", 0.0),
        occupancy_quantile_low=source.get("reactivation_occupancy_quantile_low", None),
        occupancy_quantile_high=source.get(
            "reactivation_occupancy_quantile_high", None
        ),
        occupancy_target_low=source.get("reactivation_occupancy_target_low", None),
        occupancy_target_high=source.get("reactivation_occupancy_target_high", None),
        calibration_min_quantile_width=source.get(
            "reactivation_calibration_min_quantile_width", 1e-3
        ),
        calibration_max_m=source.get("reactivation_calibration_max_m", 50.0),
        calibration_revert_on_invalid=source.get(
            "reactivation_calibration_revert_on_invalid", True
        ),
        sigma_aware_k=source.get("reactivation_sigma_aware_k", 0.25),
        strategy=source.get("reactivation_strategy", "none"),
        memory_efficient=source.get("reactivation_memory_efficient", False),
    )


def _morphology_config_from_kwargs(source: Mapping[str, Any]) -> MorphologyConfig:
    """Build morphology and voltage options from legacy kwargs."""
    return MorphologyConfig(
        use_shunting=source.get("use_shunting", True),
        use_additive_normalization=source.get("use_additive_normalization", False),
        additive_mode=source.get("additive_mode", "raw"),
        additive_tangent_n0=source.get("additive_tangent_n0", None),
        additive_tangent_t0=source.get("additive_tangent_t0", None),
        epsilon=source.get("epsilon", 1e-8),
        weight_transform=source.get("weight_transform", "exp"),
        efficient_blocklinear=source.get("efficient_blocklinear", False),
        blocklinear_strategy=source.get("blocklinear_strategy", "none"),
    )


def _initialization_config_from_kwargs(source: Mapping[str, Any]) -> AdaptiveInitConfig:
    """Build branch-layer initialization options from legacy kwargs."""
    branch_factors = source.get("branch_factors", None)
    if branch_factors is not None:
        branch_factors = tuple(branch_factors)

    return AdaptiveInitConfig(
        dbl_init_method=source.get("dbl_init_method", "analytical_expectation"),
        adaptive_initialization=source.get("adaptive_initialization", True),
        adaptive_initialization_policy=source.get(
            "adaptive_initialization_policy", "preserve_shunting_center"
        ),
        adaptive_target_conductance=source.get("adaptive_target_conductance", 5.0),
        initial_child_conductance=source.get("initial_child_conductance", 1.0),
        branch_factors=branch_factors,
        initialization_seed=source.get("initialization_seed", None),
        initialization_namespace=str(source.get("initialization_namespace", "") or ""),
    )


def _deepst_config_from_kwargs(source: Mapping[str, Any]) -> DeepSTConfig:
    """Build DeepST compatibility options from legacy kwargs."""
    return DeepSTConfig(
        use_noise=source.get("use_noise", True),
        rewiring_mode=source.get("rewiring_mode", "global"),
        rewire_frequency=source.get("rewire_frequency", 1),
        synapses_per_branch=source.get("synapses_per_branch", None),
        sigma=source.get("sigma", 0.05),
        excitatory_target_density=source.get("excitatory_target_density", 0.1),
        inhibitory_target_density=source.get("inhibitory_target_density", 0.1),
        freeze_excitatory_connectivity=source.get(
            "freeze_excitatory_connectivity", False
        ),
        freeze_inhibitory_connectivity=source.get(
            "freeze_inhibitory_connectivity", False
        ),
        init_method=source.get("init_method", None),
        weight_threshold=source.get("weight_threshold", 1e-6),
        credit_trace_decay=source.get("credit_trace_decay", 0.95),
        credit_candidate_pool_size=source.get("credit_candidate_pool_size", 8),
        credit_turnover_fraction=source.get("credit_turnover_fraction", 0.1),
        credit_weak_active_pool_fraction=source.get(
            "credit_weak_active_pool_fraction", 0.5
        ),
        credit_min_observations=source.get("credit_min_observations", 5),
        credit_swap_margin=source.get("credit_swap_margin", 0.0),
        credit_force_turnover=source.get("credit_force_turnover", False),
        credit_selection=source.get("credit_selection", "credit"),
        credit_warmup_steps=source.get("credit_warmup_steps", 5),
    )


def _topk_kwargs(config: TopKConfig) -> dict[str, Any]:
    """Convert TopK options to legacy branch-layer kwargs."""
    return {
        "topk_init_method": config.init_method,
        "topk_noise_level": config.noise_level,
        "topk_type": config.type,
        "topk_weight_norm_order": config.weight_norm_order,
        "topk_gamma": config.gamma,
        "topk_temperature": config.temperature,
        "topk_ultrafast": config.ultrafast,
        "topk_strategy": config.strategy,
    }


def _dense_to_sparse_kwargs(config: DenseToSparseConfig) -> dict[str, Any]:
    """Convert dense-to-sparse options to legacy kwargs."""
    return {
        "dense_to_sparse_initial_density": config.initial_density,
        "dense_to_sparse_initial_k": config.initial_k,
        "dense_to_sparse_start_step": config.start_step,
        "dense_to_sparse_end_step": config.end_step,
        "dense_to_sparse_update_interval": config.update_interval,
        "dense_to_sparse_schedule": config.schedule,
        "dense_to_sparse_freeze_on_end": config.freeze_on_end,
        "dense_to_sparse_advance_on_forward": config.advance_on_forward,
        "dense_to_sparse_prune_metric": config.prune_metric,
    }


def _indexed_kwargs(config: IndexedSynapseConfig) -> dict[str, Any]:
    """Convert indexed sparse-synapse options to legacy kwargs."""
    return {
        "indexed_seed": config.seed,
        "indexed_output_chunk_size": config.output_chunk_size,
        "indexed_candidate_size": config.candidate_size,
        "indexed_selection": config.selection,
        "indexed_rewire_frequency": config.rewire_frequency,
        "indexed_rewire_quantile": config.rewire_quantile,
        "indexed_rewire_until_step": config.rewire_until_step,
        "indexed_index_dtype": config.index_dtype,
        "indexed_workspace_mb": config.workspace_mb,
        "indexed_cache_transformed_weights": config.cache_transformed_weights,
        "indexed_recompute_backward": config.recompute_backward,
        "indexed_projection_backend": config.projection_backend,
        "indexed_persistent_indices": config.persistent_indices,
        "indexed_init_mode": config.init_mode,
        "indexed_support_group_rows": config.support_group_rows,
        "indexed_support_col_block": config.support_col_block,
    }


def _reactivation_kwargs(config: ReactivationConfig) -> dict[str, Any]:
    """Convert reactivation options to legacy kwargs."""
    return {
        "reactivate": config.enabled,
        "reactivation_type": config.type,
        "dendritic_activation": config.dendritic_activation,
        "reactivation_init_m": config.init_m,
        "reactivation_init_b": config.init_b,
        "reactivation_init_policy": config.init_policy,
        "reactivation_soma_type": config.soma_type,
        "reactivation_soma_init_m": config.soma_init_m,
        "reactivation_soma_init_b": config.soma_init_b,
        "reactivation_occupancy_quantile_low": config.occupancy_quantile_low,
        "reactivation_occupancy_quantile_high": config.occupancy_quantile_high,
        "reactivation_occupancy_target_low": config.occupancy_target_low,
        "reactivation_occupancy_target_high": config.occupancy_target_high,
        "reactivation_calibration_min_quantile_width": (
            config.calibration_min_quantile_width
        ),
        "reactivation_calibration_max_m": config.calibration_max_m,
        "reactivation_calibration_revert_on_invalid": (
            config.calibration_revert_on_invalid
        ),
        "reactivation_sigma_aware_k": config.sigma_aware_k,
        "reactivation_strategy": config.strategy,
        "reactivation_memory_efficient": config.memory_efficient,
    }


def _morphology_kwargs(config: MorphologyConfig) -> dict[str, Any]:
    """Convert morphology and voltage options to legacy kwargs."""
    return {
        "use_shunting": config.use_shunting,
        "use_additive_normalization": config.use_additive_normalization,
        "additive_mode": config.additive_mode,
        "additive_tangent_n0": config.additive_tangent_n0,
        "additive_tangent_t0": config.additive_tangent_t0,
        "epsilon": config.epsilon,
        "weight_transform": config.weight_transform,
        "efficient_blocklinear": config.efficient_blocklinear,
        "blocklinear_strategy": config.blocklinear_strategy,
    }


def _initialization_kwargs(config: AdaptiveInitConfig) -> dict[str, Any]:
    """Convert initialization options to legacy kwargs."""
    return {
        "dbl_init_method": config.dbl_init_method,
        "adaptive_initialization": config.adaptive_initialization,
        "adaptive_initialization_policy": config.adaptive_initialization_policy,
        "adaptive_target_conductance": config.adaptive_target_conductance,
        "initial_child_conductance": config.initial_child_conductance,
        "branch_factors": (
            None if config.branch_factors is None else list(config.branch_factors)
        ),
        "initialization_seed": config.initialization_seed,
        "initialization_namespace": config.initialization_namespace,
    }


def _deepst_kwargs(config: DeepSTConfig) -> dict[str, Any]:
    """Convert DeepST compatibility options to legacy kwargs."""
    return {
        "use_noise": config.use_noise,
        "rewiring_mode": config.rewiring_mode,
        "rewire_frequency": config.rewire_frequency,
        "synapses_per_branch": config.synapses_per_branch,
        "sigma": config.sigma,
        "excitatory_target_density": config.excitatory_target_density,
        "inhibitory_target_density": config.inhibitory_target_density,
        "freeze_excitatory_connectivity": config.freeze_excitatory_connectivity,
        "freeze_inhibitory_connectivity": config.freeze_inhibitory_connectivity,
        "init_method": config.init_method,
        "weight_threshold": config.weight_threshold,
        "credit_trace_decay": config.credit_trace_decay,
        "credit_candidate_pool_size": config.credit_candidate_pool_size,
        "credit_turnover_fraction": config.credit_turnover_fraction,
        "credit_weak_active_pool_fraction": config.credit_weak_active_pool_fraction,
        "credit_min_observations": config.credit_min_observations,
        "credit_swap_margin": config.credit_swap_margin,
        "credit_force_turnover": config.credit_force_turnover,
        "credit_selection": config.credit_selection,
        "credit_warmup_steps": config.credit_warmup_steps,
    }


def _coerce_section_config(
    section_type,
    value: Any,
    *,
    section_name: str,
    base_config: Any,
):
    """Normalize a nested section config object or mapping."""
    if isinstance(value, section_type):
        return value
    if isinstance(value, Mapping):
        return replace(base_config, **value)
    raise TypeError(
        f"{section_name} config must be a {section_type.__name__} or mapping"
    )


def _has_nested_section_config(mapping: Mapping[str, Any]) -> bool:
    """Return whether a mapping carries any nested typed section payload."""
    return any(
        key in SECTION_CONFIG_TYPES
        and (isinstance(value, Mapping) or isinstance(value, SECTION_CONFIG_TYPES[key]))
        for key, value in mapping.items()
    )


@dataclass(frozen=True)
class DendriticSynapseConfig:
    """Grouped branch-layer options with legacy-kwarg conversion helpers."""

    topk: TopKConfig = field(default_factory=TopKConfig)
    dense_to_sparse: DenseToSparseConfig = field(default_factory=DenseToSparseConfig)
    indexed: IndexedSynapseConfig = field(default_factory=IndexedSynapseConfig)
    reactivation: ReactivationConfig = field(default_factory=ReactivationConfig)
    morphology: MorphologyConfig = field(default_factory=MorphologyConfig)
    initialization: AdaptiveInitConfig = field(default_factory=AdaptiveInitConfig)
    deepst: DeepSTConfig = field(default_factory=DeepSTConfig)
    print_hooks: bool = False

    @classmethod
    def from_config(
        cls,
        config: DendriticSynapseConfig | Mapping[str, Any],
    ) -> DendriticSynapseConfig:
        """Normalize an existing config object or legacy kwarg mapping."""
        if isinstance(config, cls):
            return config
        if isinstance(config, Mapping):
            if _has_nested_section_config(config):
                return cls.from_sections(config)
            return cls.from_kwargs(config)
        raise TypeError("synapse_config must be a DendriticSynapseConfig or mapping")

    @classmethod
    def from_kwargs(
        cls,
        mapping: Mapping[str, Any] | None = None,
        /,
        **kwargs: Any,
    ) -> DendriticSynapseConfig:
        """Build a typed config from legacy ``DendriticBranchLayer`` kwargs."""
        source = dict(mapping or {})
        source.update(kwargs)

        return cls(
            topk=_topk_config_from_kwargs(source),
            dense_to_sparse=_dense_to_sparse_config_from_kwargs(source),
            indexed=_indexed_config_from_kwargs(source),
            reactivation=_reactivation_config_from_kwargs(source),
            morphology=_morphology_config_from_kwargs(source),
            initialization=_initialization_config_from_kwargs(source),
            deepst=_deepst_config_from_kwargs(source),
            print_hooks=source.get("print_hooks", False),
        )

    @classmethod
    def from_population_config(cls, population_config: Any) -> DendriticSynapseConfig:
        """Build branch-layer synapse options from a recurrent PopulationConfig.

        The recurrent population config carries both synapse construction fields
        and recurrent dynamics fields. This bridge deliberately selects only the
        fields consumed by ``DendriticBranchLayer`` so recurrent builders do not
        need to duplicate the long legacy keyword mapping.
        """

        return cls.from_kwargs(
            {
                key: getattr(population_config, key)
                for key in POPULATION_SYNAPSE_KEYS
                if hasattr(population_config, key)
            }
        )

    @classmethod
    def from_sections(
        cls,
        mapping: Mapping[str, Any] | None = None,
        /,
        **kwargs: Any,
    ) -> DendriticSynapseConfig:
        """Build from nested typed sections, with flat legacy kwargs as defaults."""
        source = dict(mapping or {})
        source.update(kwargs)
        config = cls.from_kwargs(source)
        replacements = {
            section_name: _coerce_section_config(
                section_type,
                source[section_name],
                section_name=section_name,
                base_config=getattr(config, section_name),
            )
            for section_name, section_type in SECTION_CONFIG_TYPES.items()
            if section_name in source
        }
        if "print_hooks" in source:
            replacements["print_hooks"] = bool(source["print_hooks"])
        return replace(config, **replacements)

    @classmethod
    def split_kwargs(
        cls,
        kwargs: Mapping[str, Any],
    ) -> tuple[DendriticSynapseConfig, dict[str, Any]]:
        """Return the config built from known keys plus unconsumed kwargs."""
        known = cls.legacy_kwargs_from_mapping(kwargs)
        extra = {key: value for key, value in kwargs.items() if key not in LEGACY_KEYS}
        return cls.from_kwargs(known), extra

    @classmethod
    def legacy_kwargs_from_mapping(cls, mapping: Mapping[str, Any]) -> dict[str, Any]:
        """Select legacy synapse kwargs from a larger mapping."""
        return {key: mapping[key] for key in LEGACY_KEY_ORDER if key in mapping}

    def to_kwargs(self) -> dict[str, Any]:
        """Convert back to legacy branch-layer kwargs."""
        return {
            **_topk_kwargs(self.topk),
            **_dense_to_sparse_kwargs(self.dense_to_sparse),
            **_indexed_kwargs(self.indexed),
            **_morphology_kwargs(self.morphology),
            **_reactivation_kwargs(self.reactivation),
            **_initialization_kwargs(self.initialization),
            **_deepst_kwargs(self.deepst),
            "print_hooks": self.print_hooks,
        }

    def with_branch_factors(
        self, branch_factors: Sequence[int] | None
    ) -> DendriticSynapseConfig:
        """Return a copy carrying the concrete dendritic morphology factors."""

        normalized = None if branch_factors is None else tuple(branch_factors)
        return replace(
            self,
            initialization=replace(
                self.initialization,
                branch_factors=normalized,
            ),
        )


LEGACY_KEY_ORDER = tuple(DendriticSynapseConfig().to_kwargs())
LEGACY_KEYS = frozenset(LEGACY_KEY_ORDER)
POPULATION_SYNAPSE_KEYS = tuple(sorted(LEGACY_KEYS - {"dendritic_activation"}))
SECTION_CONFIG_TYPES = {
    "topk": TopKConfig,
    "dense_to_sparse": DenseToSparseConfig,
    "indexed": IndexedSynapseConfig,
    "reactivation": ReactivationConfig,
    "morphology": MorphologyConfig,
    "initialization": AdaptiveInitConfig,
    "deepst": DeepSTConfig,
}


__all__ = [
    "AdaptiveInitConfig",
    "DeepSTConfig",
    "DendriticSynapseConfig",
    "DenseToSparseConfig",
    "IndexedSynapseConfig",
    "MorphologyConfig",
    "ReactivationConfig",
    "TopKConfig",
]
