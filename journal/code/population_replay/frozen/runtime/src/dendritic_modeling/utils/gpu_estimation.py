#!/usr/bin/env python3
"""
GPU Memory Estimation for Dendritic Models.

This module provides comprehensive GPU memory estimation for dendritic neural networks,
including support for FSDP distributed training.
"""

import logging
import math
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, ClassVar, Optional

import numpy as np
import torch

from dendritic_modeling.config.conversion import to_plain_dict
from dendritic_modeling.config.recurrence import is_recurrent_core_config
from dendritic_modeling.datasets.dimensions import (
    dataset_class_count,
    dataset_input_shape,
)
from dendritic_modeling.utils.hooks import iter_modules_matching
from dendritic_modeling.utils.reproducibility import resolve_experiment_seeds

logger = logging.getLogger(__name__)

BYTES_PER_GIB = 1024**3
DEFAULT_DTYPE_BYTES = 2
DTYPE_BYTES = {
    "fp32": 4,
    "fp16": 2,
    "bf16": 2,
    "int8": 1,
}
DEFAULT_OPTIMIZER_STATE_MULTIPLIER = 2.0
OPTIMIZER_STATE_MULTIPLIERS = {
    "sgd": 1.0,  # Only momentum
    "adam": 2.0,  # Momentum + variance
    "adamw": 2.0,  # Same as Adam
    "lamb": 2.0,  # Similar to Adam
    "rmsprop": 1.0,  # Running average
}
DENDRITIC_BRANCHING_OVERHEAD = 10.0
TOPK_INDEX_VALUE_FACTOR = 2
TOPK_SPARSE_GRADIENT_FACTOR = 4
DEFAULT_TOPK_SYNAPSES = 100
BLOCKLINEAR_ACTIVATION_FACTOR = 2
LINEAR_ACTIVATION_FACTOR = 2
PARAMETER_FALLBACK_ACTIVATION_MULTIPLIER = 60
GRADIENT_GRAPH_OVERHEAD = 2.5
FSDP_COMMUNICATION_OVERHEAD = 0.5
FSDP_ACTIVATION_CHECKPOINT_FACTOR = 0.7
FSDP_CONFIG_ACTIVATION_FACTOR = 0.8
FSDP_CONFIG_OVERHEAD_FACTOR = 0.1
DEFAULT_BATCH_SIZE = 32
DEFAULT_TRAINING_DTYPE = "fp16"
DEFAULT_ESTIMATION_OPTIMIZER = "adam"
DEFAULT_CLASS_COUNT = 10
REGRESSION_OUTPUT_DIM = 1
DEFAULT_TRAIN_VALID_SPLIT = 0.8
DEFAULT_INPUT_DIM = 784
GPU_MEMORY_SAFETY_FACTOR = 0.85
PRACTICAL_FSDP_GPU_LIMIT = 16
FSDP_CANDIDATE_CONFIGURATIONS = (
    (2, 1, 2),
    (4, 1, 4),
    (8, 2, 4),
    (16, 4, 4),
    (32, 8, 4),
)
FSDP_FALLBACK_NODES = 8
FSDP_FALLBACK_GPUS_PER_NODE = 4
FSDP_FALLBACK_TOTAL_GPUS = 32
_PARAMETER_CATEGORY_KEYWORDS = (
    ("TopK", "TopK Layers"),
    ("BlockLinear", "Block Linear Layers"),
    ("Linear", "Linear Layers"),
    ("Conv", "Convolutional Layers"),
    ("Reactivation", "Reactivation Layers"),
    ("ParametricTanh", "Reactivation Layers"),
    ("BatchNorm", "Normalization Layers"),
    ("LayerNorm", "Normalization Layers"),
)


def _builtin_input_shape(dataset_name: str, *, flatten: bool = True) -> tuple[int, ...]:
    """Return a required built-in dataset shape from the canonical dimension API."""
    shape = dataset_input_shape(dataset_name, flatten=flatten)
    if shape is None:  # pragma: no cover - protects module constants from drift.
        raise RuntimeError(f"Missing built-in input shape for {dataset_name!r}")
    return shape


MNIST_FLAT_INPUT_SHAPE = _builtin_input_shape("mnist")
CIFAR10_FLAT_INPUT_SHAPE = _builtin_input_shape("cifar10")
IMAGENET_FLAT_INPUT_SHAPE = _builtin_input_shape("imagenet")
IMAGENET_IMAGE_INPUT_SHAPE = _builtin_input_shape("imagenet", flatten=False)


def _bytes_to_gib(num_bytes: float) -> float:
    """Convert bytes to GiB using the estimator's historical convention."""
    return num_bytes / BYTES_PER_GIB


def _resolve_batch_size(config: Any, batch_size: Optional[int] = None) -> int:
    """Resolve the batch size from explicit, current, or legacy config paths."""
    if batch_size is not None:
        return batch_size

    resolved_batch_size = DEFAULT_BATCH_SIZE

    if hasattr(config, "training") and hasattr(config.training, "main"):
        if hasattr(config.training.main, "common"):
            resolved_batch_size = getattr(
                config.training.main.common, "batch_size", resolved_batch_size
            )
        elif hasattr(config.training.main, "trainer"):
            trainer_params = config.training.main.trainer.__dict__
            resolved_batch_size = trainer_params.get("batch_size", resolved_batch_size)
    elif hasattr(config, "train") and hasattr(config.train, "main"):
        if hasattr(config.train.main, "trainer"):
            trainer_params = config.train.main.trainer.__dict__
            resolved_batch_size = trainer_params.get("batch_size", resolved_batch_size)

    return resolved_batch_size


def _resolve_task_config(config: Any) -> tuple[Any, Any]:
    """Resolve legacy task config or synthesize one from the data config."""
    if hasattr(config, "task"):
        task_cfg = config.task
        task_params = to_plain_dict(getattr(task_cfg, "parameters", {}))
        # Dataset loading requires Mapping semantics (not merely attributes).
        # Keep the legacy object identity while normalizing its boundary field.
        task_cfg.parameters = task_params
        return task_cfg, task_params

    processing_params = to_plain_dict(getattr(config.data, "processing", {}))
    dataset_name = config.data.dataset_name
    all_dataset_params = to_plain_dict(getattr(config.data, "dataset_params", {}))
    dataset_params = to_plain_dict(all_dataset_params.get(dataset_name, {}))
    task_params = {**processing_params, **dataset_params}

    experiment = getattr(config, "experiment", None)
    if experiment is not None and hasattr(experiment, "seed"):
        resolved_seeds = resolve_experiment_seeds(experiment)
        task_params["label_noise_seed"] = resolved_seeds.dataset_seed
        task_params["split_seed"] = resolved_seeds.split_seed
        task_params.setdefault("seed", resolved_seeds.dataset_seed)

    task_cfg = type("TaskConfig", (), {})()
    task_cfg.dataset = dataset_name
    task_cfg.data_path = getattr(config.data, "base_dir", None)
    task_cfg.parameters = task_params
    task_cfg.train_valid_split = getattr(
        config.experiment, "train_valid_split", DEFAULT_TRAIN_VALID_SPLIT
    )
    return task_cfg, task_params


def _fallback_input_shape(task_cfg: Any, task_params: Any) -> tuple[int, ...]:
    """Return the historical hardcoded input shape when dataset loading fails."""
    task_params = to_plain_dict(task_params)
    dataset_name = getattr(task_cfg, "dataset", "cifar10")
    shape = dataset_input_shape(
        dataset_name,
        flatten=task_params.get("flatten", False),
    )
    if shape is not None:
        return shape

    return (task_params.get("input_dim", DEFAULT_INPUT_DIM),)


def _flat_input_dim(input_shape: tuple[int, ...]) -> int:
    """Return the flattened input dimension used by flat architectures."""
    return input_shape[0] if len(input_shape) == 1 else int(np.prod(input_shape))


def _config_params(config_node: Any) -> dict[str, Any]:
    """Return explicit legacy/current params from a config node."""
    return to_plain_dict(
        getattr(config_node, "parameters", None) or getattr(config_node, "params", {})
    )


def _core_params_from_config(core_config: Any) -> dict[str, Any]:
    """Merge non-structured core parameter sections into one parameter dict."""
    if hasattr(core_config, "parameters"):
        return dict(core_config.parameters)

    core_params = {}
    for section_name in ("architecture", "connectivity", "morphology"):
        if hasattr(core_config, section_name):
            core_params.update(dict(getattr(core_config, section_name).__dict__))
    return core_params


def _resolve_n_classes(task_type: str, task_cfg: Any, task_params: Any) -> int:
    """Resolve decoder output size from task type and dataset metadata."""
    if task_type != "classification":
        return REGRESSION_OUTPUT_DIM

    task_params = to_plain_dict(task_params)
    dataset_name = getattr(task_cfg, "dataset", "cifar10")
    return dataset_class_count(
        dataset_name,
        default_class_count=task_params.get("n_classes", DEFAULT_CLASS_COUNT),
    )


def _resolve_trainer_params(config: Any) -> dict[str, Any]:
    """Resolve trainer params from current or legacy config paths."""
    if hasattr(config, "training") and hasattr(config.training, "main"):
        if hasattr(config.training.main, "common"):
            return config.training.main.common.__dict__
        if hasattr(config.training.main, "trainer"):
            return config.training.main.trainer.__dict__
    elif hasattr(config, "train") and hasattr(config.train, "main"):
        if hasattr(config.train.main, "trainer"):
            return config.train.main.trainer.__dict__
    return {}


def _resolve_training_dtype(config: Any) -> str:
    """Resolve training dtype used by the memory estimator."""
    return _resolve_trainer_params(config).get("dtype", DEFAULT_TRAINING_DTYPE)


def _resolve_estimation_input_shape(
    task_cfg: Any,
    task_params: dict[str, Any],
) -> tuple[int, ...] | torch.Size:
    """Resolve input shape without downloading built-in datasets.

    Canonical metadata is sufficient for built-in datasets.  Custom datasets
    still use one real sample so temporal and structured input shapes remain
    accurate.
    """
    task_params = to_plain_dict(task_params)
    dataset_name = getattr(task_cfg, "dataset", "cifar10")
    canonical_shape = dataset_input_shape(
        dataset_name,
        flatten=task_params.get("flatten", False),
    )
    if canonical_shape is not None:
        logger.info(
            "Using canonical input shape for %s: %s",
            dataset_name,
            canonical_shape,
        )
        return canonical_shape

    from dendritic_modeling.datasets import get_unified_datasets

    try:
        train_ds, _, _ = get_unified_datasets(task_cfg=task_cfg)
        sample_input = train_ds[0][0]
        input_shape = sample_input.shape
        logger.info(f"Using real dataset input shape: {input_shape}")
        return input_shape
    except Exception as e:
        logger.warning(
            "Failed to load dataset, falling back to canonical built-in shapes: %s", e
        )
        return _fallback_input_shape(task_cfg, task_params)


def _estimation_input_dim(
    model_config: Any, input_shape: tuple[int, ...] | torch.Size
) -> int:
    """Match training's feature-dimension handling for recurrent inputs."""
    if is_recurrent_core_config(model_config.core):
        return int(input_shape[-1])
    return _flat_input_dim(tuple(input_shape))


def _build_estimation_encoder(
    model_config: Any, input_shape: tuple[int, ...] | torch.Size
):
    """Build the encoder used for memory estimation."""
    from dendritic_modeling.networks import Identity
    from dendritic_modeling.networks.architectures.factory import get_architecture

    encoder_config = (
        getattr(model_config, "encoder_network", None) or model_config.encoder
    )
    input_dim = _estimation_input_dim(model_config, input_shape)
    if encoder_config.type.lower() == "identity":
        return Identity(input_dim)

    encoder_params = _config_params(encoder_config)
    encoder_params["input_dim"] = input_dim
    return get_architecture(encoder_config.type, encoder_params)


def _build_estimation_core(model_config: Any, encoder: torch.nn.Module):
    """Build the core network used for memory estimation."""
    from dendritic_modeling.networks.architectures.factory import get_architecture

    core_config = getattr(model_config, "core_network", None) or model_config.core
    core_type = core_config.type.lower()
    structured_core_types = {
        "einet",
        "ei_unified",
        "unified_ei",
        "ei_net",
        "unified_einet",
        "rnn_dendritic_shunting",
        "rnn_dendritic_additive",
        "rnn_dendritic_normalized_additive",
        "rnn_flat_shunting",
        "rnn_flat_additive",
        "rnn_flat_normalized_additive",
    }
    is_structured_core = hasattr(core_config, "architecture") or (
        core_type == "population_network" and hasattr(core_config, "population_network")
    )
    if (core_type in structured_core_types or core_type == "population_network") and (
        is_structured_core
    ):
        return get_architecture(core_config.type, core_config, encoder.output_dim)

    core_params = _core_params_from_config(core_config)
    core_params["input_dim"] = encoder.output_dim
    return get_architecture(core_config.type, core_params)


def _build_estimation_decoder(
    model_config: Any,
    core_network: torch.nn.Module,
    n_classes: int,
):
    """Build the decoder used for memory estimation."""
    from dendritic_modeling.networks import Identity
    from dendritic_modeling.networks.architectures.factory import get_architecture

    decoder_config = (
        getattr(model_config, "decoder_network", None) or model_config.decoder
    )
    if decoder_config.type.lower() == "identity":
        return Identity(core_network.output_dim)

    decoder_params = _config_params(decoder_config)
    decoder_params["input_dim"] = core_network.output_dim
    if decoder_params.get("output_dim") is None:
        decoder_params["output_dim"] = n_classes
    return get_architecture(decoder_config.type, decoder_params)


def _build_estimation_model(
    config: Any,
    input_shape: tuple[int, ...],
    task_cfg: Any,
    task_params: dict[str, Any],
):
    """Build the model used for GPU-memory estimation."""
    from dendritic_modeling.models import Classifier, Regressor

    model_config = config.model
    encoder = _build_estimation_encoder(model_config, input_shape)
    core_network = _build_estimation_core(model_config, encoder)

    task_type = model_config.task
    n_classes = _resolve_n_classes(task_type, task_cfg, task_params)
    decoder = _build_estimation_decoder(model_config, core_network, n_classes)

    if task_type == "classification":
        return Classifier(
            encoder_network=encoder,
            core_network=core_network,
            decoder_network=decoder,
            learned_output_scale=getattr(model_config, "learned_output_scale", True),
            fixed_output_scale=getattr(model_config, "fixed_output_scale", 1.0),
            output_scale_mode=getattr(model_config, "output_scale_mode", None),
        )

    return Regressor(
        encoder_network=encoder,
        core_network=core_network,
        decoder_network=decoder,
    )


def _is_leaf_module(module: torch.nn.Module) -> bool:
    """Return whether a module has no child modules."""
    return len(list(module.children())) == 0


def _iter_leaf_modules(model: torch.nn.Module) -> Iterator[torch.nn.Module]:
    """Yield leaf modules in PyTorch module traversal order."""
    yield from iter_modules_matching(model, _is_leaf_module)


def _module_parameter_category(module: torch.nn.Module) -> str:
    """Return the historical parameter-count category for a module."""
    module_type = type(module).__name__

    for keyword, category in _PARAMETER_CATEGORY_KEYWORDS:
        if keyword in module_type:
            return category
    return "Other Layers"


def _extract_einet_activation_config(module: torch.nn.Module) -> dict[str, Any] | None:
    """Extract the legacy EINet activation-estimation fields from a module."""
    if not hasattr(module, "excitatory_layer_sizes"):
        return None

    return {
        "e_layers": module.excitatory_layer_sizes,
        "i_layers": module.inhibitory_layer_sizes,
        "e_branches": module.excitatory_branch_factors,
        "i_branches": module.inhibitory_branch_factors,
        "ee_synapses": module.ee_synapses_per_branch_per_layer,
        "ei_synapses": module.ei_synapses_per_branch_per_layer,
        "ie_synapses": module.ie_synapses_per_branch_per_layer,
        "ii_synapses": module.ii_synapses_per_branch_per_layer,
    }


def _find_einet_activation_config(
    model: torch.nn.Module,
) -> dict[str, Any] | None:
    """Return activation-estimation config for the first EINet module, if present."""
    for module in iter_modules_matching(
        model,
        lambda module: "ExcitationInhibitionNetwork" in type(module).__name__,
    ):
        return _extract_einet_activation_config(module)
    return None


def _topk_activation_bytes(
    module: torch.nn.Module,
    *,
    batch_size: int,
    bytes_per_param: int,
) -> float:
    """Estimate sparse TopK activation and gradient storage in bytes."""
    out_features = getattr(module, "out_features", 0)
    k_synapses = getattr(module, "K", DEFAULT_TOPK_SYNAPSES)
    return (
        batch_size
        * out_features
        * k_synapses
        * bytes_per_param
        * TOPK_SPARSE_GRADIENT_FACTOR
    )


def _blocklinear_activation_bytes(
    module: torch.nn.Module,
    *,
    batch_size: int,
    bytes_per_param: int,
) -> float:
    """Estimate BlockLinear branch-combination activation storage in bytes."""
    if not hasattr(module, "weight"):
        return 0.0

    weight = module.weight
    if callable(weight) or not hasattr(weight, "shape"):
        return 0.0

    return (
        batch_size
        * np.prod(weight.shape)
        * bytes_per_param
        * BLOCKLINEAR_ACTIVATION_FACTOR
    )


def _linear_activation_bytes(
    module: torch.nn.Linear,
    *,
    batch_size: int,
    bytes_per_param: int,
) -> float:
    """Estimate dense Linear input/output activation storage in bytes."""
    return (
        batch_size
        * (module.in_features + module.out_features)
        * bytes_per_param
        * LINEAR_ACTIVATION_FACTOR
    )


def _parameter_fallback_activation_bytes(
    model: torch.nn.Module,
    *,
    bytes_per_param: int,
) -> float:
    """Estimate activations from parameters when layer-wise structure is opaque."""
    total_params = sum(p.numel() for p in model.parameters())
    return total_params * bytes_per_param * PARAMETER_FALLBACK_ACTIVATION_MULTIPLIER


@dataclass
class MemoryBreakdown:
    """Memory breakdown for model training."""

    parameters: float  # GB
    gradients: float  # GB
    optimizer_states: float  # GB
    activations: float  # GB
    fsdp_overhead: float  # GB
    total: float  # GB

    def __str__(self):
        lines = [
            "Memory Breakdown:",
            f"  Parameters:        {self.parameters:.2f} GB",
            f"  Gradients:         {self.gradients:.2f} GB",
            f"  Optimizer states:  {self.optimizer_states:.2f} GB",
            f"  Activations:       {self.activations:.2f} GB",
            f"  FSDP overhead:     {self.fsdp_overhead:.2f} GB",
            f"  Total:             {self.total:.2f} GB",
        ]
        return "\n".join(lines)


@dataclass
class GPURecommendation:
    """GPU recommendations for training."""

    gpu_model: str
    gpu_memory_gb: int
    min_gpus_needed: int
    recommended_gpus: int  # Power of 2 for efficiency
    memory_per_gpu_gb: float
    utilization_percent: float
    fits: bool

    def __str__(self):
        status = "[FITS]" if self.fits else "[NO FIT]"
        return (
            f"{status} {self.gpu_model} ({self.gpu_memory_gb}GB): "
            f"min {self.min_gpus_needed} GPUs, "
            f"recommended {self.recommended_gpus} GPUs, "
            f"{self.memory_per_gpu_gb:.2f}GB/GPU ({self.utilization_percent:.1f}% utilization)"
        )


class GPUMemoryEstimator:
    """Estimate GPU memory requirements for dendritic models."""

    GPU_SPECS: ClassVar[dict[str, dict[str, Any]]] = {
        "A100-40GB": {"memory_gb": 40, "compute": "high"},
        "H100": {"memory_gb": 80, "compute": "high"},
    }

    def __init__(self, dtype: str = "fp16", optimizer: str = "adam"):
        """
        Initialize the estimator.

        Args:
            dtype: Data type ('fp16', 'fp32', 'bf16')
            optimizer: Optimizer type ('adam', 'sgd', 'adamw')
        """
        self.dtype = dtype
        self.optimizer = optimizer
        self.bytes_per_param = self._get_bytes_per_param(dtype)
        self.optimizer_state_multiplier = self._get_optimizer_multiplier(optimizer)

    def _get_bytes_per_param(self, dtype: str) -> int:
        """Get bytes per parameter for given dtype."""
        return DTYPE_BYTES.get(dtype, DEFAULT_DTYPE_BYTES)

    def _get_optimizer_multiplier(self, optimizer: str) -> float:
        """Get memory multiplier for optimizer states."""
        return OPTIMIZER_STATE_MULTIPLIERS.get(
            optimizer.lower(), DEFAULT_OPTIMIZER_STATE_MULTIPLIER
        )

    def estimate_model_parameters(self, model: torch.nn.Module) -> dict[str, int]:
        """
        Count parameters by layer type.

        Args:
            model: PyTorch model

        Returns:
            Dictionary mapping layer types to parameter counts
        """
        param_counts = {}
        total_params = 0

        for module in _iter_leaf_modules(model):
            # Count parameters for this module
            module_params = sum(p.numel() for p in module.parameters())
            if module_params == 0:
                continue

            category = _module_parameter_category(module)
            param_counts[category] = param_counts.get(category, 0) + module_params
            total_params += module_params

        param_counts["Total"] = total_params
        return param_counts

    def estimate_activation_memory(
        self, model: torch.nn.Module, batch_size: int, input_shape: tuple[int, ...]
    ) -> float:
        """
        Estimate activation memory for dendritic networks.

        Args:
            model: PyTorch model
            batch_size: Batch size
            input_shape: Input shape (excluding batch dimension)

        Returns:
            Activation memory in GB
        """
        activation_memory_bytes = 0

        # For dendritic networks (EINet), we need to analyze the structure more carefully
        # The network has multiple layers with branch factors creating massive intermediate activations

        # First, check if this is an EINet model
        einet_config = _find_einet_activation_config(model)

        if einet_config:
            # Calculate activation memory for EINet architecture
            input_dim = np.prod(input_shape)

            # Transfer layer activations
            activation_memory_bytes += batch_size * input_dim * self.bytes_per_param

            # Calculate branch factors product for each layer
            e_branch_product = 1
            for bf in einet_config["e_branches"]:
                e_branch_product *= bf

            i_branch_product = 1
            if einet_config["i_branches"]:
                for bf in einet_config["i_branches"]:
                    i_branch_product *= bf

            # For each excitatory layer
            for i, layer_size in enumerate(einet_config["e_layers"]):
                # Dendritic branch activations
                # Each neuron has branch_product branches
                branch_activations = (
                    batch_size * layer_size * e_branch_product * self.bytes_per_param
                )
                activation_memory_bytes += branch_activations

                # TopK connectivity activations (sparse weight indices and values)
                if i < len(einet_config["ee_synapses"]):
                    synapses = einet_config["ee_synapses"][i]
                    # Each branch has K synapses, storing indices and values
                    topk_memory = (
                        batch_size
                        * layer_size
                        * e_branch_product
                        * synapses
                        * self.bytes_per_param
                        * TOPK_INDEX_VALUE_FACTOR
                    )
                    activation_memory_bytes += topk_memory

            # For each inhibitory layer (if present)
            if einet_config["i_layers"]:
                for i, layer_size in enumerate(einet_config["i_layers"]):
                    if layer_size:  # Skip None entries
                        # Similar calculation for inhibitory neurons
                        branch_activations = (
                            batch_size
                            * layer_size
                            * i_branch_product
                            * self.bytes_per_param
                        )
                        activation_memory_bytes += branch_activations

                        # TopK connectivity for inhibitory neurons
                        if i < len(einet_config["ii_synapses"]):
                            synapses = einet_config["ii_synapses"][i]
                            topk_memory = (
                                batch_size
                                * layer_size
                                * i_branch_product
                                * synapses
                                * self.bytes_per_param
                                * TOPK_INDEX_VALUE_FACTOR
                            )
                            activation_memory_bytes += topk_memory

            # Add substantial overhead for gradient computation graphs in dendritic networks
            # Dendritic networks have complex backward passes with branch-wise operations
            activation_memory_bytes *= DENDRITIC_BRANCHING_OVERHEAD

        else:
            # Fallback: Analyze layer by layer
            for module in _iter_leaf_modules(model):
                # TopK layers - these are the main memory consumers in dendritic networks
                if "TopK" in type(module).__name__ or hasattr(module, "K"):
                    activation_memory_bytes += _topk_activation_bytes(
                        module,
                        batch_size=batch_size,
                        bytes_per_param=self.bytes_per_param,
                    )

                elif "BlockLinear" in type(module).__name__:
                    activation_memory_bytes += _blocklinear_activation_bytes(
                        module,
                        batch_size=batch_size,
                        bytes_per_param=self.bytes_per_param,
                    )

                elif isinstance(module, torch.nn.Linear):
                    activation_memory_bytes += _linear_activation_bytes(
                        module,
                        batch_size=batch_size,
                        bytes_per_param=self.bytes_per_param,
                    )

            # If we couldn't get a good estimate, use parameter-based estimation
            if activation_memory_bytes == 0:
                activation_memory_bytes = _parameter_fallback_activation_bytes(
                    model,
                    bytes_per_param=self.bytes_per_param,
                )

            # Add gradient graph overhead
            activation_memory_bytes *= GRADIENT_GRAPH_OVERHEAD

        return _bytes_to_gib(activation_memory_bytes)

    def estimate_memory_breakdown(
        self,
        model: torch.nn.Module,
        batch_size: int,
        input_shape: tuple[int, ...],
        use_fsdp: bool = False,
        num_gpus: int = 1,
    ) -> MemoryBreakdown:
        """
        Estimate complete memory breakdown for training.

        Args:
            model: PyTorch model
            batch_size: Batch size
            input_shape: Input shape (excluding batch dimension)
            use_fsdp: Whether using FSDP
            num_gpus: Number of GPUs for FSDP

        Returns:
            MemoryBreakdown object
        """
        # Count parameters
        param_counts = self.estimate_model_parameters(model)
        total_params = param_counts["Total"]

        # Parameter memory
        param_memory_gb = _bytes_to_gib(total_params * self.bytes_per_param)

        # Gradient memory (same as parameters)
        gradient_memory_gb = param_memory_gb

        # Optimizer state memory
        optimizer_memory_gb = param_memory_gb * self.optimizer_state_multiplier

        # Activation memory
        activation_memory_gb = self.estimate_activation_memory(
            model, batch_size, input_shape
        )

        # FSDP overhead
        fsdp_overhead_gb = 0.0
        if use_fsdp:
            # FSDP adds communication buffers and sharding metadata
            # Approximately 10-20% overhead
            base_memory = param_memory_gb + gradient_memory_gb + optimizer_memory_gb
            fsdp_overhead_gb = (
                base_memory * FSDP_COMMUNICATION_OVERHEAD
            )  # Increased overhead for communication/sharding

            # With FSDP, memory is sharded across GPUs
            param_memory_gb /= num_gpus
            gradient_memory_gb /= num_gpus
            optimizer_memory_gb /= num_gpus
            # Activations are not sharded (each GPU computes full forward pass)
            # But FSDP can use gradient checkpointing to reduce activation memory
            activation_memory_gb *= FSDP_ACTIVATION_CHECKPOINT_FACTOR

        # Total memory
        total_memory_gb = (
            param_memory_gb
            + gradient_memory_gb
            + optimizer_memory_gb
            + activation_memory_gb
            + fsdp_overhead_gb
        )

        return MemoryBreakdown(
            parameters=param_memory_gb,
            gradients=gradient_memory_gb,
            optimizer_states=optimizer_memory_gb,
            activations=activation_memory_gb,
            fsdp_overhead=fsdp_overhead_gb,
            total=total_memory_gb,
        )

    def get_gpu_recommendations(
        self,
        memory_breakdown: MemoryBreakdown,
        safety_factor: float = GPU_MEMORY_SAFETY_FACTOR,
    ) -> list[GPURecommendation]:
        """
        Get GPU recommendations based on memory requirements.

        Args:
            memory_breakdown: Memory breakdown from estimate_memory_breakdown
            safety_factor: Safety factor for memory (0.85 = use 85% of GPU memory)

        Returns:
            List of GPU recommendations sorted by efficiency
        """
        recommendations = []

        for gpu_name, gpu_spec in self.GPU_SPECS.items():
            gpu_memory = gpu_spec["memory_gb"]
            usable_memory = gpu_memory * safety_factor

            # Single GPU scenario
            single_gpu_fits = memory_breakdown.total <= usable_memory

            # Multi-GPU scenario (with FSDP)
            if usable_memory <= 0 or memory_breakdown.total <= 0:
                logger.warning(
                    f" Invalid memory values: total={memory_breakdown.total:.2f}GB, usable={usable_memory:.2f}GB"
                )
                min_gpus_needed = 1  # Default to 1 GPU if calculation fails
            else:
                min_gpus_needed = math.ceil(memory_breakdown.total / usable_memory)

            # Recommend power of 2 for communication efficiency
            recommended_gpus = 1
            while recommended_gpus < min_gpus_needed:
                recommended_gpus *= 2

            # Memory per GPU with FSDP
            if recommended_gpus > 0:
                memory_per_gpu = memory_breakdown.total / recommended_gpus
                utilization = (
                    (memory_per_gpu / gpu_memory) * 100 if gpu_memory > 0 else 0
                )
            else:
                memory_per_gpu = memory_breakdown.total
                utilization = 0

            recommendation = GPURecommendation(
                gpu_model=gpu_name,
                gpu_memory_gb=gpu_memory,
                min_gpus_needed=min_gpus_needed,
                recommended_gpus=recommended_gpus,
                memory_per_gpu_gb=memory_per_gpu,
                utilization_percent=utilization,
                fits=single_gpu_fits or recommended_gpus <= PRACTICAL_FSDP_GPU_LIMIT,
            )

            recommendations.append(recommendation)

        recommendations.sort(key=lambda r: (r.recommended_gpus, -r.utilization_percent))

        return recommendations

    def get_optimal_gpu_config(self, single_gpu_breakdown, h100_memory, a100_memory):
        """
        Determine optimal GPU configuration for FSDP training.

        Returns:
            dict: Optimal configuration with nodes, gpus_per_node, total_gpus, etc.
        """

        if single_gpu_breakdown.total <= h100_memory:
            return {
                "single_gpu_fits": True,
                "gpu_type": "H100",
                "nodes": 1,
                "gpus_per_node": 1,
                "total_gpus": 1,
                "memory_per_gpu": single_gpu_breakdown.total,
            }
        elif single_gpu_breakdown.total <= a100_memory:
            return {
                "single_gpu_fits": True,
                "gpu_type": "A100-40GB",
                "nodes": 1,
                "gpus_per_node": 1,
                "total_gpus": 1,
                "memory_per_gpu": single_gpu_breakdown.total,
            }

        for total_gpus, nodes, gpus_per_node in FSDP_CANDIDATE_CONFIGURATIONS:
            param_memory = single_gpu_breakdown.parameters / total_gpus
            grad_memory = single_gpu_breakdown.gradients / total_gpus
            optimizer_memory = single_gpu_breakdown.optimizer_states / total_gpus
            activation_memory = (
                single_gpu_breakdown.activations * FSDP_CONFIG_ACTIVATION_FACTOR
            )
            fsdp_overhead = FSDP_CONFIG_OVERHEAD_FACTOR * (param_memory + grad_memory)

            total_memory = (
                param_memory
                + grad_memory
                + optimizer_memory
                + activation_memory
                + fsdp_overhead
            )

            if total_memory <= a100_memory:
                return {
                    "single_gpu_fits": False,
                    "gpu_type": "A100-40GB",
                    "nodes": nodes,
                    "gpus_per_node": gpus_per_node,
                    "total_gpus": total_gpus,
                    "memory_per_gpu": total_memory,
                }

            elif total_memory <= h100_memory:
                return {
                    "single_gpu_fits": False,
                    "gpu_type": "H100",
                    "nodes": nodes,
                    "gpus_per_node": gpus_per_node,
                    "total_gpus": total_gpus,
                    "memory_per_gpu": total_memory,
                }

        return {
            "single_gpu_fits": False,
            "gpu_type": "H100",
            "nodes": FSDP_FALLBACK_NODES,
            "gpus_per_node": FSDP_FALLBACK_GPUS_PER_NODE,
            "total_gpus": FSDP_FALLBACK_TOTAL_GPUS,
            "memory_per_gpu": total_memory,
            "warning": "Model may be too large - consider optimization",
        }

    def print_estimation_report(
        self,
        model: torch.nn.Module,
        batch_size: int,
        input_shape: tuple[int, ...],
        config: Optional[Any] = None,
    ):
        """
        Print a simplified GPU memory estimation report.

        Args:
            model: PyTorch model
            batch_size: Batch size
            input_shape: Input shape (excluding batch dimension)
            config: Optional configuration object
        """
        # Get total parameters
        total_params = sum(p.numel() for p in model.parameters())

        # Single GPU estimation
        single_gpu_breakdown = self.estimate_memory_breakdown(
            model, batch_size, input_shape, use_fsdp=False
        )

        logger.info("=" * 80)
        logger.info("GPU MEMORY ESTIMATION")
        logger.info("=" * 80)
        logger.info(f"Model: {total_params:,} parameters")
        logger.info(f"Single GPU memory: {single_gpu_breakdown.total:.2f} GB")

        # Test 4-GPU FSDP (common setup)
        fsdp_4gpu = self.estimate_memory_breakdown(
            model, batch_size, input_shape, use_fsdp=True, num_gpus=4
        )
        logger.info(f"4-GPU FSDP memory per GPU: {fsdp_4gpu.total:.2f} GB")

        a100_memory = (
            self.GPU_SPECS["A100-40GB"]["memory_gb"] * GPU_MEMORY_SAFETY_FACTOR
        )
        h100_memory = self.GPU_SPECS["H100"]["memory_gb"] * GPU_MEMORY_SAFETY_FACTOR

        if single_gpu_breakdown.total <= a100_memory:
            logger.info("[FITS] Fits on single A100-40GB (40GB)")
        elif single_gpu_breakdown.total <= h100_memory:
            logger.info("[FITS] Fits on single H100 (80GB)")
        else:
            logger.info("[NO FIT] Requires multi-GPU setup")
        optimal_config = self.get_optimal_gpu_config(
            single_gpu_breakdown, h100_memory, a100_memory
        )

        if optimal_config["single_gpu_fits"]:
            logger.info(f"[FITS] Single {optimal_config['gpu_type']} sufficient")
        else:
            logger.info(
                f"[RECOMMENDED] {optimal_config['total_gpus']}x {optimal_config['gpu_type']} FSDP"
            )
            logger.info(
                f"[CONFIG] {optimal_config['nodes']} nodes x {optimal_config['gpus_per_node']} GPUs/node"
            )

        logger.info("=" * 80)

        return optimal_config


def estimate_from_config(config_path: str, batch_size: Optional[int] = None):
    """
    Estimate GPU requirements from a configuration file.

    Uses canonical metadata for built-in datasets and one real sample for
    custom datasets that may have temporal or structured inputs.

    Args:
        config_path: Path to configuration file
        batch_size: Override batch size (uses config value if None)
    """
    from dendritic_modeling.config import load_config

    config = load_config(config_path)
    batch_size = _resolve_batch_size(config, batch_size)
    task_cfg, task_params = _resolve_task_config(config)
    input_shape = _resolve_estimation_input_shape(task_cfg, task_params)
    model = _build_estimation_model(config, input_shape, task_cfg, task_params)

    dtype = _resolve_training_dtype(config)
    estimator = GPUMemoryEstimator(dtype=dtype, optimizer=DEFAULT_ESTIMATION_OPTIMIZER)
    estimator.print_estimation_report(model, batch_size, input_shape, config)

    return estimator, model
