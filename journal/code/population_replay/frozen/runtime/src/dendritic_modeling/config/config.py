"""
config.py
---------
Configuration file for dendritic_modeling.
This module defines the configuration dataclasses for the model, training, task, etc.
It loads and saves configurations using OmegaConf.
"""

import logging
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field, fields
from functools import lru_cache
from typing import Any, Optional

from omegaconf import OmegaConf

from dendritic_modeling.analysis.registry import get_registered_analyzers
from dendritic_modeling.config.analysis import (
    AnalysisConfig,
    AnalysisRuntimeConfig,
    EvaluationRuntimeConfig,
    PerformanceAnalysisConfig,
    PerformanceAnalysisParams,
)
from dendritic_modeling.config.base import BaseConfig
from dendritic_modeling.config.conversion import to_plain_dict as _to_plain_dict
from dendritic_modeling.config.legacy import normalize_transfer_config
from dendritic_modeling.config.local_learning import (
    build_local_rule_config as _build_local_rule_config,
)
from dendritic_modeling.config.model import (
    ArchitectureConfig,
    BlockLinearConfig,
    ConnectivityConfig,
    CoreConfig,
    DecoderConfig,
    DecoderParamsConfig,
    EncoderConfig,
    EncoderParamsConfig,
    ImplementationConfig,
    ModelConfig,
    MorphologyConfig,
    PretrainedReplacementConfig,
    ReactivationConfig,
    ReplacementRegionConfig,
    SegmentTrainabilityConfig,
    SparsityConfig,
    SpatialConfig,
    StructuredConnectivityConfig,
    TrainabilityConfig,
    TransferFunctionConfig,
    TransformerReplacementConfig,
)
from dendritic_modeling.config.reactivation import (
    DEFAULT_ADDITIVE_REACTIVATION_INIT_POLICY,
    normalize_reactivation_init_policy,
)
from dendritic_modeling.config.recurrent import RecurrentConfig
from dendritic_modeling.config.training import (
    CommonTrainerConfig,
    EncoderTrainingConfig,
    FreezeBranchesStrategyConfig,
    FreezeBranchKLStrategyConfig,
    FreezeLayersStrategyConfig,
    HomeostaticControlStrategyConfig,
    LocalRuleConfig,
    MainTrainingConfig,
    MultiStageStrategyConfig,
    OptimizerConfig,
    ParamGroupsConfig,
    RecurrentStrategyConfig,
    StandardStrategyConfig,
    TrainingConfig,
    TrainingStrategiesConfig,
    TransformerReplacementTrainingConfig,
    VisionDistillationStrategyConfig,
    VisionReplacementTrainingConfig,
    VoltageStabilizationStrategyConfig,
)
from dendritic_modeling.config.validation import validate_loaded_config

logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig(BaseConfig):
    """Basic experiment settings."""

    seed: int = 42
    # Independent confirmatory RNG streams. ``None`` preserves legacy config
    # ergonomics and resolves to ``seed`` at runtime after CLI overrides.
    dataset_seed: Optional[int] = None
    split_seed: Optional[int] = None
    model_seed: Optional[int] = None
    topology_seed: Optional[int] = None
    loader_seed: Optional[int] = None
    evaluation_seed: Optional[int] = None
    probe_seed: Optional[int] = None
    strict_deterministic: bool = False
    # Optional reviewer-facing test seal. Legacy runs retain unrestricted test
    # access; confirmatory harnesses can expose only an explicitly capped audit
    # set and disable every analysis path that could adapt to test examples.
    sealed_test: bool = False
    allow_test_data: bool = True
    max_test_samples: Optional[int] = None
    # Opt-in content hashes over the realized train/validation/test datasets.
    # Confirmatory generated-data runs use this to prove which exact tensors
    # reached the training process without serializing the examples twice.
    record_dataset_fingerprints: bool = False
    train_valid_split: float = 0.8
    fast_mode: bool = False
    deterministic: bool = True
    cudnn_benchmark: bool = False
    allow_tf32: bool = False
    float32_matmul_precision: str = "highest"
    enable_profiling: bool = False
    profiling_output_dir: str = "results/profiling/"
    enable_hooks: bool = True

    checkpointing: dict[str, Any] = field(
        default_factory=lambda: {
            "enabled": False,
            "save_every_n_epochs": 10,
            "checkpoint_dir": "results/checkpoints/",
        }
    )


@dataclass
class DataProcessingConfig(BaseConfig):
    """Data processing parameters."""

    flatten: bool = True
    normalize: bool = False
    label_noise_rate: float = (
        0.0  # Fraction of training labels to randomize (0 = clean)
    )


@dataclass
class DatasetParamsConfig(BaseConfig):
    """Dataset-specific configuration parameters."""

    # MNIST variants
    mnist_modulo10: dict[str, Any] = field(
        default_factory=lambda: {"shuffle_iterations": 2}
    )
    cifar10_modulo10: dict[str, Any] = field(
        default_factory=lambda: {"shuffle_iterations": 1, "pair_seed": 0}
    )
    double_mnist_contextual_mod10: dict[str, Any] = field(
        default_factory=lambda: {
            "shuffle_iterations": 1,
            "pair_seed": 0,
            "context_seed": 123,
        }
    )
    double_cifar10_contextual_mod10: dict[str, Any] = field(
        default_factory=lambda: {
            "shuffle_iterations": 1,
            "pair_seed": 0,
            "context_seed": 123,
        }
    )
    random_flip_mnist: dict[str, Any] = field(
        default_factory=lambda: {"flip_probability": 0.5}
    )

    # Synthetic datasets
    poisson_generator: dict[str, Any] = field(
        default_factory=lambda: {
            "base_dataset": "mnist",
            "multiplicative_gain": True,
            "fixed_gain_factor": None,
            "max_gain_factor": None,
            "max_gain_tau_ratio": None,
            "uniform_gain": True,
            "gain_sampling": "log_uniform",
            "poisson_sampling": True,
            "stimulus_duration": 1.0,
        }
    )

    multixor: dict[str, Any] = field(default_factory=lambda: {"n_bits": 8})
    orientation_bars: dict[str, Any] = field(
        default_factory=lambda: {"n_orient_classes": 8}
    )
    orthonet: dict[str, Any] = field(
        default_factory=lambda: {
            "input_dim": 784,
            "n_layers": 5,
            "nonlinearity": "sigmoid",
            "repeat_layers": False,
            "noise_std": 0.1,
        }
    )
    contextual_stream_gain_shift: dict[str, Any] = field(
        default_factory=lambda: {
            "n_samples": 4000,
            "stream_dim": 64,
            "n_classes": 2,
            "signal_strength": 0.35,
            "relevant_noise_std": 0.08,
            "irrelevant_noise_std": 0.18,
            "train_gain_relevant_min": 0.9,
            "train_gain_relevant_max": 1.1,
            "train_gain_irrelevant_min": 0.8,
            "train_gain_irrelevant_max": 1.2,
            "test_gain_relevant": 1.0,
            "test_gain_irrelevant": 3.0,
            "test_irrelevant_alignment_alpha": 1.0,
            "ood_mode": "irrelevant",
            "valid_split_mode": "ood",
            "context_signal_scale": 1.0,
        }
    )
    branch_local_gain_load: dict[str, Any] = field(
        default_factory=lambda: {
            "n_samples": 6000,
            "stream_dim": 64,
            "signal_fraction": 0.5,
            "n_gain_groups": 8,
            "n_classes": 2,
            "e_baseline": 5.0,
            "i_baseline": 2.0,
            "e_signal_delta": 1.0,
            "i_signal_delta": 0.0,
            "signal_mode": "e_only",
            "support_mode": "prefix",
            "train_gain_support_mode": "signal",
            "gain_fraction": -1.0,
            "load_fraction": 1.0,
            "load_alignment_alpha": 1.0,
            "independent_noise_std": 0.05,
            "train_gain_sigma": 0.8,
            "test_gain_sigma": 0.8,
            "gain_alignment_alpha": 1.0,
            "load_mean": 0.0,
            "load_noise_sigma": 0.0,
            "valid_split_mode": "test",
        }
    )
    hierarchical_gain_load: dict[str, Any] = field(
        default_factory=lambda: {
            "n_samples": 6000,
            "stream_dim": 64,
            "n_levels": 3,
            "hierarchy_branching": 2,
            "n_flat_groups": 8,
            "n_classes": 2,
            "e_baseline": 5.0,
            "i_baseline": 2.0,
            "e_signal_delta": 0.35,
            "i_signal_delta": 0.35,
            "signal_mode": "e_only",
            "signal_profile": "all",
            "nuisance_layout": "factorized_sensors",
            "sensor_e_baseline": 0.02,
            "gain_structure": "hierarchical",
            "gain_scale_decay": 1.0,
            "train_gain_sigma": 0.4,
            "test_gain_sigma": 1.4,
            "sensor_alignment_alpha": 1.0,
            "sensor_support_mode": "matched",
            "private_gain_sigma": 0.0,
            "independent_noise_std": 0.08,
            "load_mean": 0.0,
            "load_noise_sigma": 0.0,
            "valid_split_mode": "test",
        }
    )
    imagenet: dict[str, Any] = field(
        default_factory=lambda: {
            "loader_backend": "auto",
            "train_samples_per_class": None,
            "val_samples_per_class": None,
        }
    )


@dataclass
class DataConfig(BaseConfig):
    """Data configuration including dataset selection and processing."""

    dataset_name: str = "cifar10"
    base_dir: str = ""

    processing: DataProcessingConfig = field(default_factory=DataProcessingConfig)
    dataset_params: DatasetParamsConfig = field(default_factory=DatasetParamsConfig)


@dataclass
class WandbConfig(BaseConfig):
    """Weights & Biases logging configuration."""

    use_wandb: bool = False
    entity: str = ""
    project: str = "dendritic-sweeps"
    group: str = "unified_sweep"
    tags: list[str] = field(default_factory=lambda: ["debug"])


@dataclass
class OutputsConfig(BaseConfig):
    """Output configuration."""

    run_name: str = "ei_sweep"
    results_dir: str = "results"
    # Use ``results_dir`` itself as the run directory. This is intended for
    # manifest-controlled runs whose output path has already been atomically
    # claimed; the legacy timestamped behavior remains the default.
    exact_run_dir: bool = False


@dataclass
class FSDPConfig(BaseConfig):
    """FSDP-specific tuning knobs.

    These only take effect when ``distributed.mode`` is ``"fsdp"``.
    """

    sharding_strategy: str = "FULL_SHARD"
    mixed_precision: bool = True
    cpu_offload: bool = False
    min_num_params: int = 1000000
    # ``indexed_synapses`` nests FSDP around each indexed sparse synapse
    # module. This bounds parameter all-gathers for contact-heavy models while
    # leaving the historical whole-model policy unchanged by default.
    auto_wrap_policy: str = "none"
    # Let FSDP move and shard wrapped units directly from CPU instead of first
    # materializing the complete, unsharded model on every GPU.
    cpu_init: bool = False
    reduce_communication_overhead: bool = True
    limit_all_gathers: bool = True
    forward_prefetch: bool = True
    sync_module_states: bool = True
    use_orig_params: bool = True
    backward_prefetch: str = "BACKWARD_PRE"
    gradient_checkpointing: bool = False
    # Readout-only mode for contact-heavy recurrent models. Freezing before
    # FSDP wrapping avoids core gradients and optimizer state while preserving
    # the full dendritic forward dynamics.
    freeze_core: bool = False
    # Cache deterministic frozen-core outputs once per local dataset shard.
    # Later epochs train the readout without repeating the recurrent forward.
    cache_frozen_core_outputs: bool = False
    # Optional local batch size used after the frozen states are cached.
    cached_core_batch_size: Optional[int] = None
    # Mixed-precision dtype control (only used when mixed_precision=True). The
    # single knob `mixed_precision_dtype` sets param/reduce/buffer together; the
    # granular fields override it individually (None = follow the knob). Default
    # "fp16" preserves historical behavior; set "bf16" (or reduce_dtype="fp32")
    # to avoid fp16 gradient-all-reduce overflow on bf16-capable GPUs.
    mixed_precision_dtype: str = "fp16"
    param_dtype: Optional[str] = None
    reduce_dtype: Optional[str] = None
    buffer_dtype: Optional[str] = None


@dataclass
class DistributedConfig(BaseConfig):
    """Distributed training configuration.

    ``mode`` selects the runtime:

        - ``"none"`` — single-GPU, plain ``python`` (default)
        - ``"ddp"``  — DistributedDataParallel via ``torchrun``
        - ``"fsdp"`` — FullyShardedDataParallel via ``torchrun``

    ``fsdp`` contains FSDP-specific tuning knobs (only used when mode=fsdp).

    Node and GPU counts are **not** part of training config — they are
    scheduler/launcher concerns.  Use ``sbatch --nodes=`` and
    ``--gpus-per-node=`` (or the script's CLI argument) to control resources.
    """

    mode: str = "none"  # none | ddp | fsdp
    fsdp: FSDPConfig = field(default_factory=FSDPConfig)


@dataclass
class Config:
    """Main configuration class with clean structure."""

    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    outputs: OutputsConfig = field(default_factory=OutputsConfig)
    distributed: DistributedConfig = field(default_factory=DistributedConfig)


def _build_core_config(core_data: Any) -> CoreConfig:
    """Build a typed CoreConfig from possibly nested dict/OmegaConf payload."""
    core_dict = _to_plain_dict(core_data)
    core_type = str(core_dict.get("type", "einet"))
    lower_core_type = core_type.lower()
    if lower_core_type in {"ei_net", "unified_einet"}:
        logger.warning(
            "model.core.type=%r is deprecated; use 'unified_ei' instead.",
            core_type,
        )
    unified_ei = core_dict.get("unified_ei", core_dict.get("ei_unified", {}))
    connectivity_dict = _to_plain_dict(core_dict.get("connectivity", {}))
    structured_alias_present = "structured_connectivity" in connectivity_dict
    structured_alias = connectivity_dict.pop("structured_connectivity", {})
    if structured_alias_present:
        logger.warning(
            "model.core.connectivity.structured_connectivity is deprecated; "
            "use model.core.connectivity.structured instead."
        )
    structured_connectivity = connectivity_dict.pop("structured", structured_alias)
    morphology_dict = _to_plain_dict(core_dict.get("morphology", {}))
    sparsity_dict = _to_plain_dict(core_dict.get("sparsity", {}))
    annealed_topk_alias = sparsity_dict.pop("annealed_topk", None)
    if annealed_topk_alias is not None:
        logger.warning(
            "model.core.sparsity.annealed_topk is deprecated; use "
            "model.core.sparsity.dense_to_sparse instead."
        )
        sparsity_dict.setdefault("dense_to_sparse", annealed_topk_alias)
    reactivation_dict = _to_plain_dict(core_dict.get("reactivation", {}))
    additive_aliases = {
        "dendritic_additive",
        "flat_additive",
        "dendritic_normalized_additive",
        "flat_normalized_additive",
    }
    is_explicit_additive_einet = (
        lower_core_type in {"einet", "ei_net", "unified_einet"}
        and morphology_dict.get("use_shunting") is False
    )
    if "init_policy" not in reactivation_dict and (
        lower_core_type in additive_aliases or is_explicit_additive_einet
    ):
        reactivation_dict["init_policy"] = DEFAULT_ADDITIVE_REACTIVATION_INIT_POLICY
    elif "init_policy" in reactivation_dict:
        reactivation_dict["init_policy"] = normalize_reactivation_init_policy(
            reactivation_dict.get("init_policy")
        )

    return CoreConfig(
        type=core_type,
        target_active_parameters=(
            None
            if core_dict.get("target_active_parameters") is None
            else int(core_dict["target_active_parameters"])
        ),
        biological_neuron=(
            None
            if core_dict.get("biological_neuron") is None
            else bool(core_dict["biological_neuron"])
        ),
        initialization_seed=(
            None
            if core_dict.get("initialization_seed") is None
            else int(core_dict["initialization_seed"])
        ),
        architecture=ArchitectureConfig(
            **_to_plain_dict(core_dict.get("architecture", {}))
        ),
        connectivity=ConnectivityConfig(
            **connectivity_dict,
            structured=StructuredConnectivityConfig(
                **_to_plain_dict(structured_connectivity)
            ),
        ),
        transfer=TransferFunctionConfig(
            **normalize_transfer_config(core_dict.get("transfer", {}))
        ),
        morphology=MorphologyConfig(**morphology_dict),
        sparsity=SparsityConfig(**sparsity_dict),
        reactivation=ReactivationConfig(**reactivation_dict),
        blocklinear=BlockLinearConfig(
            **_to_plain_dict(core_dict.get("blocklinear", {}))
        ),
        implementation=ImplementationConfig(
            **_to_plain_dict(core_dict.get("implementation", {}))
        ),
        spatial=SpatialConfig(**_to_plain_dict(core_dict.get("spatial", {}))),
        synapse_types=_to_plain_dict(core_dict.get("synapse_types", {})),
        dynamics=_to_plain_dict(
            core_dict.get("dynamics", core_dict.get("spiking", {}))
        ),
        dendritic_spikes=_to_plain_dict(core_dict.get("dendritic_spikes", {})),
        soma_feedback=_to_plain_dict(core_dict.get("soma_feedback", {})),
        recurrent=RecurrentConfig(**_to_plain_dict(core_dict.get("recurrent", {}))),
        unified_ei=_to_plain_dict(unified_ei),
        recurrent_ei=_to_plain_dict(core_dict.get("recurrent_ei", {})),
        baseline_rnn=_to_plain_dict(core_dict.get("baseline_rnn", {})),
        heterogeneous_leak_ctrnn=_to_plain_dict(
            core_dict.get("heterogeneous_leak_ctrnn", {})
        ),
        legendre_memory=_to_plain_dict(core_dict.get("legendre_memory", {})),
        population_network=_to_plain_dict(core_dict.get("population_network", {})),
        vision=_to_plain_dict(core_dict.get("vision", {})),
    )


def _build_pretrained_replacement_config(
    pr_data: Any,
    *,
    legacy_flat_output_adapter: str | None = None,
) -> PretrainedReplacementConfig:
    """Build a typed PretrainedReplacementConfig from dict/OmegaConf."""
    pr_dict = _to_plain_dict(pr_data)
    replace_dict = _to_plain_dict(pr_dict.get("replace", {}))
    train_dict = _to_plain_dict(pr_dict.get("trainability", {}))

    encoder_dict = _to_plain_dict(train_dict.get("encoder", {}))
    decoder_dict = _to_plain_dict(train_dict.get("decoder", {}))
    trainability = TrainabilityConfig(
        encoder=SegmentTrainabilityConfig(
            mode=str(encoder_dict.get("mode", "frozen")),
        ),
        decoder=SegmentTrainabilityConfig(
            mode=str(decoder_dict.get("mode", "trainable")),
        ),
    )

    configured_adapter = pr_dict.get("flat_output_adapter")
    if configured_adapter is not None and legacy_flat_output_adapter is not None:
        if str(configured_adapter).strip().lower() != legacy_flat_output_adapter:
            raise ValueError(
                "Conflicting flat output adapters: the canonical vision replacement "
                f"config requests {configured_adapter!r}, while legacy core.transfer "
                f"fields request {legacy_flat_output_adapter!r}"
            )
    flat_output_adapter = (
        str(configured_adapter or legacy_flat_output_adapter or "auto").strip().lower()
    )
    if flat_output_adapter not in {"auto", "zero_pad", "linear"}:
        raise ValueError(
            "flat_output_adapter must be 'auto', 'zero_pad', or 'linear'; "
            f"got {flat_output_adapter!r}"
        )

    return PretrainedReplacementConfig(
        enabled=bool(pr_dict.get("enabled", False)),
        backbone=str(pr_dict.get("backbone", "alexnet")),
        weights=pr_dict.get("weights", "IMAGENET1K_V1"),
        core_source=str(pr_dict.get("core_source", "configured")),
        input_shape=list(pr_dict.get("input_shape", [3, 224, 224])),
        replace=ReplacementRegionConfig(
            start=str(replace_dict.get("start", "")),
            end=str(replace_dict.get("end", "")),
        ),
        target_modules=[str(name) for name in pr_dict.get("target_modules", [])],
        regions=[_to_plain_dict(region) for region in pr_dict.get("regions", []) or []],
        omit_layers=list(pr_dict.get("omit_layers", [])),
        flat_output_adapter=flat_output_adapter,
        teacher_init=str(pr_dict.get("teacher_init", "none")),
        selection=_to_plain_dict(pr_dict.get("selection", {})),
        trainability=trainability,
    )


def _build_transformer_replacement_config(
    tr_data: Any,
) -> TransformerReplacementConfig:
    """Build a typed TransformerReplacementConfig from dict/OmegaConf."""
    tr_dict = _to_plain_dict(tr_data)
    if "backbone" in tr_dict and "model_name" not in tr_dict:
        logger.warning(
            "model.transformer_replacement.backbone is deprecated; use "
            "model.transformer_replacement.model_name instead."
        )
    if "layer_indices" in tr_dict and "layers" not in tr_dict:
        logger.warning(
            "model.transformer_replacement.layer_indices is deprecated; use "
            "model.transformer_replacement.layers instead."
        )
    if "mlp_attr" in tr_dict and "target_module" not in tr_dict:
        logger.warning(
            "model.transformer_replacement.mlp_attr is deprecated; use "
            "model.transformer_replacement.target_module instead."
        )
    return TransformerReplacementConfig(
        enabled=bool(tr_dict.get("enabled", False)),
        model_name=str(tr_dict.get("model_name", tr_dict.get("backbone", ""))),
        model_family=str(tr_dict.get("model_family", "causal_lm")),
        model_loader=str(tr_dict.get("model_loader", "auto")),
        model_source=_to_plain_dict(tr_dict.get("model_source", {})),
        layers=[
            int(idx) for idx in tr_dict.get("layers", tr_dict.get("layer_indices", []))
        ],
        # Preserve the declared scalar types here.  The placement validator
        # deliberately rejects strings, booleans, and other coercible values
        # instead of silently changing the alias graph.
        parameter_tied_replacement_groups=[
            list(group)
            for group in tr_dict.get("parameter_tied_replacement_groups", []) or []
        ],
        collapsed_replacement_spans=[
            list(span) for span in tr_dict.get("collapsed_replacement_spans", []) or []
        ],
        collapsed_span_post_mlp_norm_attr=tr_dict.get(
            "collapsed_span_post_mlp_norm_attr", ""
        ),
        target_module=str(tr_dict.get("target_module", tr_dict.get("mlp_attr", "mlp"))),
        layers_attr=tr_dict.get("layers_attr"),
        input_transform=str(tr_dict.get("input_transform", "signed_split")),
        preserve_device_dtype=bool(tr_dict.get("preserve_device_dtype", True)),
        model_kwargs=_to_plain_dict(tr_dict.get("model_kwargs", {})),
        replacement_kwargs=_to_plain_dict(tr_dict.get("replacement_kwargs", {})),
        compiled_plan=_to_plain_dict(tr_dict.get("compiled_plan", {})),
        compiled_plans_by_layer={
            str(layer): _to_plain_dict(plan)
            for layer, plan in _to_plain_dict(
                tr_dict.get("compiled_plans_by_layer", {})
            ).items()
        },
        compiled_plans_by_collapsed_span={
            str(span): _to_plain_dict(plan)
            for span, plan in _to_plain_dict(
                tr_dict.get("compiled_plans_by_collapsed_span", {})
            ).items()
        },
        selection=_to_plain_dict(tr_dict.get("selection", {})),
        pre_patched=[_to_plain_dict(entry) for entry in tr_dict.get("pre_patched", [])],
        pre_patched_collapsed_spans=[
            _to_plain_dict(entry)
            for entry in tr_dict.get("pre_patched_collapsed_spans", [])
        ],
    )


def _build_param_groups_config(param_groups_data: Any) -> ParamGroupsConfig:
    """Build a typed ParamGroupsConfig from dict/OmegaConf payload."""
    param_groups_dict = _to_plain_dict(param_groups_data)
    return ParamGroupsConfig(**param_groups_dict)


def _build_common_trainer_config(common_data: Any) -> CommonTrainerConfig:
    """Build a typed CommonTrainerConfig with typed param groups."""
    common_dict = _to_plain_dict(common_data)
    param_group_keys = set(ParamGroupsConfig.__dataclass_fields__.keys())
    param_groups_dict = _to_plain_dict(common_dict.get("param_groups", {}))

    # Backward compatibility: historical configs sometimes put LR knobs
    # directly under `training.*.common` instead of under `param_groups`.
    for key in list(common_dict.keys()):
        if key in param_group_keys:
            param_groups_dict.setdefault(key, common_dict.pop(key))

    common_dict["param_groups"] = _build_param_groups_config(param_groups_dict)
    return CommonTrainerConfig(
        **_filter_dataclass_kwargs(CommonTrainerConfig, common_dict)
    )


def _build_training_strategies_config(strategies_data: Any) -> TrainingStrategiesConfig:
    """Build typed nested strategy configs from dict/OmegaConf payload."""
    strategies_dict = _to_plain_dict(strategies_data)
    strategy_classes = {
        "standard": StandardStrategyConfig,
        "multi_stage": MultiStageStrategyConfig,
        "recurrent": RecurrentStrategyConfig,
        "voltage_stabilization": VoltageStabilizationStrategyConfig,
        "homeostatic_control": HomeostaticControlStrategyConfig,
        "freeze_layers": FreezeLayersStrategyConfig,
        "freeze_branches": FreezeBranchesStrategyConfig,
        "freeze_branch_kl": FreezeBranchKLStrategyConfig,
        "vision_distillation": VisionDistillationStrategyConfig,
    }

    built: dict[str, Any] = {}
    for name, config_cls in strategy_classes.items():
        raw_value = strategies_dict.get(name)
        if raw_value is None:
            continue
        if isinstance(raw_value, config_cls):
            built[name] = raw_value
        else:
            built[name] = config_cls(**_to_plain_dict(raw_value))

    return TrainingStrategiesConfig(**built)


def _build_encoder_training_config(encoder_data: Any) -> EncoderTrainingConfig:
    """Build a typed EncoderTrainingConfig with typed common config."""
    encoder_dict = _to_plain_dict(encoder_data)
    encoder_dict["common"] = _build_common_trainer_config(
        encoder_dict.get("common", {})
    )
    if (
        "learning_strategy_config" in encoder_dict
        and encoder_dict["learning_strategy_config"] is not None
        and not isinstance(encoder_dict["learning_strategy_config"], LocalRuleConfig)
    ):
        encoder_dict["learning_strategy_config"] = _build_local_rule_config(
            _to_plain_dict(encoder_dict["learning_strategy_config"])
        )
    strategies_data = encoder_dict.get("strategies")
    if strategies_data is not None and not isinstance(
        strategies_data, TrainingStrategiesConfig
    ):
        strategies_dict = _to_plain_dict(strategies_data)
        if "local_ca" in strategies_dict:
            logger.warning(
                "training.encoder.strategies.local_ca is no longer supported; "
                "use training.encoder.learning_strategy_config instead."
            )
        encoder_dict["strategies"] = _build_training_strategies_config(strategies_dict)
    return EncoderTrainingConfig(**encoder_dict)


def _build_main_training_config(main_data: Any) -> MainTrainingConfig:
    """Build a typed MainTrainingConfig with typed common/optimizer config."""
    main_dict = _to_plain_dict(main_data)
    explicit_strategy_overrides = _to_plain_dict(main_dict.get("strategies", {}))
    main_dict["common"] = _build_common_trainer_config(main_dict.get("common", {}))
    optimizer_data = main_dict.get("optimizer")
    if optimizer_data is not None and not isinstance(optimizer_data, OptimizerConfig):
        main_dict["optimizer"] = OptimizerConfig(**_to_plain_dict(optimizer_data))
    if (
        "learning_strategy_config" in main_dict
        and main_dict["learning_strategy_config"] is not None
        and not isinstance(main_dict["learning_strategy_config"], LocalRuleConfig)
    ):
        main_dict["learning_strategy_config"] = _build_local_rule_config(
            _to_plain_dict(main_dict["learning_strategy_config"])
        )
    strategies_data = main_dict.get("strategies")
    if strategies_data is not None and not isinstance(
        strategies_data, TrainingStrategiesConfig
    ):
        strategies_dict = _to_plain_dict(strategies_data)
        if "local_ca" in strategies_dict:
            logger.warning(
                "training.main.strategies.local_ca is no longer supported; "
                "use training.main.learning_strategy_config instead."
            )
        main_dict["strategies"] = _build_training_strategies_config(strategies_dict)
    main_config = MainTrainingConfig(**main_dict)
    # Preserve the raw user-provided strategy block so downstream config
    # preparation can distinguish "explicitly set to the default value" from
    # "not specified at all".
    main_config._explicit_strategy_overrides = explicit_strategy_overrides
    return main_config


def _build_transformer_replacement_training_config(
    training_data: Any,
) -> TransformerReplacementTrainingConfig:
    """Build typed config for transformer replacement distillation."""
    return _build_filtered_config(
        TransformerReplacementTrainingConfig,
        training_data,
    )


def _build_vision_replacement_training_config(
    training_data: Any,
) -> VisionReplacementTrainingConfig:
    """Build typed config for vision replacement distillation."""
    return _build_filtered_config(VisionReplacementTrainingConfig, training_data)


def _build_model_config(model_data: Any) -> ModelConfig:
    """Build a typed ModelConfig from possibly nested dict/OmegaConf payload."""
    model_dict = _to_plain_dict(model_data)
    encoder_dict = _to_plain_dict(model_dict.get("encoder", {}))
    decoder_dict = _to_plain_dict(model_dict.get("decoder", {}))

    core_dict = _to_plain_dict(model_dict.get("core", {}))
    transfer_dict = _to_plain_dict(core_dict.get("transfer", {}))
    legacy_force_projection = bool(transfer_dict.pop("force_projection", False))
    legacy_zero_pad = bool(transfer_dict.pop("zero_pad_output", False))
    if legacy_force_projection:
        raise ValueError(
            "Legacy core.transfer.force_projection=true cannot be migrated "
            "losslessly; retain the historical source revision for that projection "
            "control or provide a canonical replacement configuration"
        )
    legacy_flat_output_adapter = "zero_pad" if legacy_zero_pad else None
    core_dict["transfer"] = transfer_dict

    pr_data = model_dict.get("pretrained_replacement", {})
    pr_dict = _to_plain_dict(pr_data)
    vr_data = model_dict.get("vision_replacement", {})
    vr_dict = _to_plain_dict(vr_data)
    legacy_pr_adapter = (
        legacy_flat_output_adapter
        if bool(pr_dict.get("enabled", False))
        and not bool(vr_dict.get("enabled", False))
        else None
    )
    legacy_vr_adapter = (
        legacy_flat_output_adapter if bool(vr_dict.get("enabled", False)) else None
    )
    pr_config = _build_pretrained_replacement_config(
        pr_data,
        legacy_flat_output_adapter=legacy_pr_adapter,
    )
    vr_config = _build_pretrained_replacement_config(
        vr_data,
        legacy_flat_output_adapter=legacy_vr_adapter,
    )
    tr_data = model_dict.get("transformer_replacement", {})
    tr_config = _build_transformer_replacement_config(tr_data)

    return ModelConfig(
        task=str(model_dict.get("task", "classification")),
        initial_checkpoint=model_dict.get("initial_checkpoint"),
        initial_checkpoint_sha256=model_dict.get("initial_checkpoint_sha256"),
        initial_checkpoint_strict=bool(
            model_dict.get("initial_checkpoint_strict", True)
        ),
        learned_output_scale=bool(model_dict.get("learned_output_scale", True)),
        fixed_output_scale=float(model_dict.get("fixed_output_scale", 1.0)),
        output_scale_mode=model_dict.get("output_scale_mode"),
        encoder=EncoderConfig(
            type=str(encoder_dict.get("type", "identity")),
            load_save_root=str(
                encoder_dict.get("load_save_root", "./trained_encoder_networks/")
            ),
            params=EncoderParamsConfig(
                **_to_plain_dict(encoder_dict.get("params", {}))
            ),
        ),
        core=_build_core_config(core_dict),
        decoder=DecoderConfig(
            type=str(decoder_dict.get("type", "MLP")),
            params=DecoderParamsConfig(
                **_to_plain_dict(decoder_dict.get("params", {}))
            ),
        ),
        vision_replacement=vr_config,
        pretrained_replacement=pr_config,
        transformer_replacement=tr_config,
    )


def _build_data_config(data_data: Any) -> DataConfig:
    """Build a typed DataConfig while preserving custom dataset sections."""
    data_dict = _to_plain_dict(data_data)
    return DataConfig(
        dataset_name=str(data_dict.get("dataset_name", "cifar10")),
        base_dir=str(data_dict.get("base_dir", "")),
        processing=_build_data_processing_config(data_dict.get("processing", {})),
        dataset_params=_coerce_dataset_params(data_dict.get("dataset_params")),
    )


def _build_data_processing_config(processing_data: Any) -> DataProcessingConfig:
    """Build typed data-processing config from a config-like payload."""
    return DataProcessingConfig(**_to_plain_dict(processing_data))


def _coerce_dataset_params(dataset_params: Any) -> DatasetParamsConfig | Any:
    """Return typed default params or preserve custom dataset parameter sections."""
    if dataset_params is None:
        return DatasetParamsConfig()
    if hasattr(dataset_params, "_metadata"):
        return OmegaConf.to_container(dataset_params, resolve=True)
    return dataset_params


def _filter_dataclass_kwargs(config_cls: type, raw_data: Any) -> dict[str, Any]:
    """Drop unknown keys when loading historical configs into dataclasses."""
    data = raw_data or {}
    if isinstance(data, Mapping):
        pass  # dict, OmegaConf DictConfig, etc.
    elif hasattr(data, "_metadata"):
        data = OmegaConf.to_container(data, resolve=True) or {}
    else:
        return {}
    allowed = {f.name for f in fields(config_cls)}
    return {key: value for key, value in data.items() if key in allowed}


def _build_filtered_config(config_cls: type, raw_data: Any) -> Any:
    """Instantiate a dataclass config after dropping unknown load-time keys."""
    return config_cls(**_filter_dataclass_kwargs(config_cls, raw_data))


@lru_cache(maxsize=10)
def _load_config_cached(path: Optional[str] = None) -> Config:
    """Load configuration from a file or use defaults.

    This internal helper is cached because YAML parsing and typed conversion are
    relatively expensive. Public callers receive a deepcopy via ``load_config``
    so runtime mutations cannot leak into later loads of the same path.
    """
    if path:
        raw_conf = OmegaConf.load(path)

        # Handle both regular configs and sweep configs (which have base_config)
        if "base_config" in raw_conf:
            # This is a sweep config file, use base_config
            conf_data = raw_conf["base_config"]
        else:
            # This is a regular config file
            conf_data = raw_conf

        # Handle analysis config field mapping
        analysis_data = conf_data.get("analysis", {})
        distributed_config = _build_distributed_config(conf_data)
        mapped_analysis = _map_analysis_config(
            analysis_data,
            distributed_mode=distributed_config.mode,
        )

        # Handle training config field mapping
        training_data = conf_data.get("training", {})
        mapped_training = _map_training_config(training_data)

        # Explicitly construct nested training configs
        encoder_config = _build_encoder_training_config(
            mapped_training.get("encoder", {})
        )
        main_config = _build_main_training_config(mapped_training.get("main", {}))
        if "transformer_distillation" in mapped_training:
            logger.warning(
                "training.transformer_distillation is deprecated; use "
                "training.transformer_replacement instead."
            )
        transformer_training = _build_transformer_replacement_training_config(
            mapped_training.get(
                "transformer_replacement",
                mapped_training.get("transformer_distillation", {}),
            )
        )
        vision_training = _build_vision_replacement_training_config(
            mapped_training.get("vision_replacement", {})
        )
        training_config = TrainingConfig(
            encoder=encoder_config,
            main=main_config,
            transformer_replacement=transformer_training,
            vision_replacement=vision_training,
        )

        config = Config(
            experiment=_build_filtered_config(
                ExperimentConfig, conf_data.get("experiment", {})
            ),
            data=_build_data_config(conf_data.get("data", {})),
            model=_build_model_config(conf_data.get("model", {})),
            training=training_config,
            analysis=AnalysisConfig(**mapped_analysis),
            wandb=_build_filtered_config(WandbConfig, conf_data.get("wandb", {})),
            outputs=OutputsConfig(**conf_data.get("outputs", {})),
            distributed=distributed_config,
        )
        validate_loaded_config(config)
        return config
    config = Config()
    validate_loaded_config(config)
    return config


def load_config(path: Optional[str] = None) -> Config:
    """Load configuration from a file or defaults as an independent object."""
    return deepcopy(_load_config_cached(path))


def load_analysis_config(
    path: str,
    *,
    distributed_mode: str = "none",
) -> AnalysisConfig:
    """Load a standalone, typed ``analysis:`` YAML profile.

    Checkpoint analyses often need to reuse an immutable training config while
    applying a newer measurement profile.  Keeping the analysis profile in a
    separate YAML avoids copying or mutating the scientific training config.
    The input may either contain an ``analysis:`` root or be the analysis
    mapping itself.
    """

    raw_conf = OmegaConf.load(path)
    analysis_data = raw_conf.get("analysis", raw_conf)
    if not isinstance(analysis_data, Mapping):
        raise ValueError(f"Analysis profile must be a mapping: {path}")
    mapped = _map_analysis_config(
        analysis_data,
        distributed_mode=distributed_mode,
    )
    return AnalysisConfig(**mapped)


# Preserve the lru_cache testing/debug interface that callers already use.
load_config.cache_clear = _load_config_cached.cache_clear  # type: ignore[attr-defined]
load_config.cache_info = _load_config_cached.cache_info  # type: ignore[attr-defined]


def _build_distributed_config(conf_data: dict) -> DistributedConfig:
    """Build DistributedConfig from the ``distributed:`` YAML block.

    Expected layout::

        distributed:
          mode: ddp
          fsdp:
            sharding_strategy: FULL_SHARD

    Raises ``ValueError`` if a legacy top-level ``fsdp:`` block is present
    (it must be nested under ``distributed:`` now).
    """
    # Reject legacy top-level fsdp: block.
    if "fsdp" in conf_data and "distributed" not in conf_data:
        raise ValueError(
            "Top-level 'fsdp:' block is no longer supported. "
            "Move FSDP settings under 'distributed.fsdp:' and set "
            "'distributed.mode: fsdp'. See docs/configuration_guide.rst."
        )
    if "fsdp" in conf_data and "fsdp" not in conf_data.get("distributed", {}):
        logger.warning(
            "Ignoring top-level 'fsdp:' block. FSDP settings must be under "
            "'distributed.fsdp:'. See docs/configuration_guide.rst."
        )

    dist_raw = dict(conf_data.get("distributed", {}))
    fsdp_raw = dict(dist_raw.pop("fsdp", {}))
    fsdp_config = _build_filtered_config(FSDPConfig, fsdp_raw)
    return DistributedConfig(
        **_filter_dataclass_kwargs(DistributedConfig, dist_raw),
        fsdp=fsdp_config,
    )


def _default_analysis_runtime_config(distributed_mode: str) -> AnalysisRuntimeConfig:
    """Return the implicit shared analysis runtime for a distributed mode."""
    if str(distributed_mode).lower() == "none":
        default_runtime = EvaluationRuntimeConfig(mode="materialize")
        return AnalysisRuntimeConfig(
            training=default_runtime,
            final=EvaluationRuntimeConfig(mode="materialize"),
        )
    return AnalysisRuntimeConfig()


_LEGACY_ANALYSIS_ALIASES = {
    "spike_trains": "spike_train_analysis",
}


def _analysis_field_mapping() -> dict[str, str]:
    """Return YAML analysis aliases keyed by registry name."""
    mapping = {
        name: spec.config_field for name, spec in get_registered_analyzers().items()
    }
    mapping.update(_LEGACY_ANALYSIS_ALIASES)
    return mapping


def _build_analysis_runtime_section(
    runtime_data: Any,
) -> tuple[EvaluationRuntimeConfig | None, Any]:
    """Build one analysis runtime section and return removed legacy splits."""
    if not isinstance(runtime_data, Mapping):
        return None, None
    runtime_dict = dict(runtime_data)
    splits = runtime_dict.pop("splits", None)
    return EvaluationRuntimeConfig(**runtime_dict), splits


def _build_analysis_section_config(
    section_data: Mapping[str, Any],
    config_cls: type,
    params_cls: type | None = None,
) -> Any:
    """Build an analysis config section, including nested params when present."""
    section = _filter_dataclass_kwargs(config_cls, section_data)
    if params_cls is not None and isinstance(section.get("params"), Mapping):
        section["params"] = params_cls(
            **_filter_dataclass_kwargs(params_cls, section["params"])
        )
    return config_cls(**section)


def _analysis_section_config_class(field_name: str) -> type | None:
    """Return the dataclass type for an ``AnalysisConfig`` field."""
    for field_def in fields(AnalysisConfig):
        if field_def.name == field_name:
            return field_def.type if isinstance(field_def.type, type) else None
    return None


def _analysis_section_params_class(config_cls: type) -> type | None:
    """Return the nested params dataclass type for an analysis section."""
    for field_def in fields(config_cls):
        if field_def.name == "params":
            return field_def.type if isinstance(field_def.type, type) else None
    return None


def _build_mapped_analysis_sections(mapped: dict[str, Any]) -> None:
    """Convert mapped analysis dicts into their typed config dataclasses."""
    for field_name, section_data in list(mapped.items()):
        if field_name == "runtime" or not isinstance(section_data, Mapping):
            continue
        config_cls = _analysis_section_config_class(field_name)
        if config_cls is None:
            continue
        params_cls = _analysis_section_params_class(config_cls)
        mapped[field_name] = _build_analysis_section_config(
            section_data,
            config_cls,
            params_cls,
        )


def _map_analysis_config(
    analysis_data: dict,
    distributed_mode: str = "none",
) -> dict:
    """Map analysis config fields to expected AnalysisConfig structure."""
    mapped = {}

    for new_name, old_name in _analysis_field_mapping().items():
        if new_name in analysis_data:
            mapped[old_name] = analysis_data[new_name]
        elif old_name in analysis_data:
            mapped[old_name] = analysis_data[old_name]

    if "runtime" in analysis_data and isinstance(analysis_data["runtime"], Mapping):
        runtime_data = dict(analysis_data["runtime"])
        default_runtime = _default_analysis_runtime_config(distributed_mode)
        training_runtime, _ = _build_analysis_runtime_section(
            runtime_data.get("training")
        )
        if training_runtime is not None:
            runtime_data["training"] = training_runtime
        elif "training" not in runtime_data:
            runtime_data["training"] = default_runtime.training
        final_runtime, _ = _build_analysis_runtime_section(runtime_data.get("final"))
        if final_runtime is not None:
            runtime_data["final"] = final_runtime
        elif "final" not in runtime_data:
            runtime_data["final"] = default_runtime.final
        mapped["runtime"] = AnalysisRuntimeConfig(**runtime_data)

    # Instantiate PerformanceAnalysisConfig and migrate legacy performance runtime
    # config into the shared top-level analysis.runtime surface.
    if "performance_analysis" in mapped and isinstance(
        mapped["performance_analysis"], Mapping
    ):
        pa = dict(mapped["performance_analysis"])

        legacy_training_runtime = None
        legacy_final_runtime = None
        legacy_training_splits = None
        legacy_final_splits = None
        legacy_training_runtime, legacy_training_splits = (
            _build_analysis_runtime_section(pa.get("training_runtime"))
        )
        legacy_final_runtime, legacy_final_splits = _build_analysis_runtime_section(
            pa.get("final_runtime")
        )

        if "runtime" not in mapped and (
            legacy_training_runtime is not None or legacy_final_runtime is not None
        ):
            mapped["runtime"] = AnalysisRuntimeConfig(
                training=legacy_training_runtime
                or _default_analysis_runtime_config(distributed_mode).training,
                final=legacy_final_runtime
                or _default_analysis_runtime_config(distributed_mode).final,
            )

        if legacy_training_splits is not None and "training_splits" not in pa:
            pa["training_splits"] = list(legacy_training_splits)
        if legacy_final_splits is not None and "final_splits" not in pa:
            pa["final_splits"] = list(legacy_final_splits)

        pa.pop("training_runtime", None)
        pa.pop("final_runtime", None)
        mapped["performance_analysis"] = _build_analysis_section_config(
            pa,
            PerformanceAnalysisConfig,
            PerformanceAnalysisParams,
        )

    if "runtime" not in mapped:
        mapped["runtime"] = _default_analysis_runtime_config(distributed_mode)

    _build_mapped_analysis_sections(mapped)

    return mapped


def _map_training_config(training_data: dict) -> dict:
    """Map training config fields and handle nested learning_strategy_config."""
    # Convert OmegaConf to regular dict
    mapped = (
        OmegaConf.to_container(training_data, resolve=True)
        if hasattr(training_data, "_metadata")
        else dict(training_data)
    )

    # Handle nested learning_strategy_config and optimizer in main config
    if "main" in mapped and isinstance(mapped["main"], dict):
        mapped["main"] = _map_main_training_config(mapped["main"])

    return mapped


def _map_main_training_config(main_data: dict[str, Any]) -> dict[str, Any]:
    """Map legacy nested fields in ``training.main``."""
    mapped_main = dict(main_data)
    _hoist_legacy_main_learning_strategy_config(mapped_main)
    _coerce_main_learning_strategy_config(mapped_main)
    _coerce_main_optimizer_config(mapped_main)
    return mapped_main


def _hoist_legacy_main_learning_strategy_config(main_data: dict[str, Any]) -> None:
    """Move legacy ``training.main.common.learning_strategy_config`` if present."""
    common = main_data.get("common")
    if not isinstance(common, dict) or "learning_strategy_config" not in common:
        return
    main_data.setdefault(
        "learning_strategy_config",
        common.pop("learning_strategy_config"),
    )
    logger.warning(
        "learning_strategy_config found under training.main.common — "
        "hoisting to training.main. Please update your config."
    )


def _coerce_main_learning_strategy_config(main_data: dict[str, Any]) -> None:
    """Build LocalRuleConfig for dict-valued main learning strategy config."""
    strategy_config = main_data.get("learning_strategy_config")
    if strategy_config is not None and isinstance(strategy_config, dict):
        main_data["learning_strategy_config"] = _build_local_rule_config(
            strategy_config
        )


def _coerce_main_optimizer_config(main_data: dict[str, Any]) -> None:
    """Build OptimizerConfig for dict-valued main optimizer config."""
    optimizer = main_data.get("optimizer")
    if isinstance(optimizer, dict):
        main_data["optimizer"] = OptimizerConfig(**optimizer)
