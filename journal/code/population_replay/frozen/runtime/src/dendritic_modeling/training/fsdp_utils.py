"""
FullyShardedDataParallel Utilities for Large Dendritic Models
=========================================

This module provides utilities for training extremely large dendritic models
that cannot fit on a single GPU using PyTorch's Fully Sharded Data Parallel (FullyShardedDataParallel).
"""

import logging
from types import MethodType
from typing import Any, Optional

import torch
from torch.distributed.fsdp import (
    BackwardPrefetch,
    CPUOffload,
    FullStateDictConfig,
    FullyShardedDataParallel,
    MixedPrecision,
    ShardingStrategy,
    StateDictType,
)
from torch.distributed.fsdp.wrap import ModuleWrapPolicy

from dendritic_modeling.networks.architectures.excitation_inhibition import (
    DendriticBranchLayer,
    ExcitationInhibitionLayer,
    ExcitationInhibitionNetwork,
    IndexedDynamicTopKLinear,
    IndexedSparseLinear,
)
from dendritic_modeling.networks.architectures.recurrent.ei_layer import EILayer
from dendritic_modeling.networks.architectures.recurrent.ei_network import EINetwork
from dendritic_modeling.utils.hooks import iter_named_modules_matching

logger = logging.getLogger(__name__)


_FSDP_DTYPE_ALIASES = {
    "fp16": torch.float16,
    "float16": torch.float16,
    "half": torch.float16,
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
    "fp32": torch.float32,
    "float32": torch.float32,
}


def _resolve_fsdp_dtype(value: Any, *, field_name: str) -> torch.dtype:
    """Resolve a config dtype alias while preserving old fp16 defaults."""
    if isinstance(value, torch.dtype):
        return value
    key = str(value).strip().lower()
    if key not in _FSDP_DTYPE_ALIASES:
        raise ValueError(
            f"Unknown FSDP {field_name}={value!r}. "
            f"Expected one of {sorted(_FSDP_DTYPE_ALIASES)}."
        )
    return _FSDP_DTYPE_ALIASES[key]


def _is_dendritic_fsdp_module(module: torch.nn.Module) -> bool:
    """Return whether a module should be treated as an FSDP dendritic unit."""
    return isinstance(
        module,
        (DendriticBranchLayer, ExcitationInhibitionLayer, EINetwork, EILayer),
    )


def _is_large_linear_checkpoint_candidate(module: torch.nn.Module) -> bool:
    """Return whether a Linear module meets the checkpointing size rule."""
    return isinstance(module, torch.nn.Linear) and module.in_features > 1000


def _is_gradient_checkpoint_candidate(_name: str, module: torch.nn.Module) -> bool:
    """Return whether a module should have activation checkpointing applied."""
    return _is_dendritic_fsdp_module(module) or _is_large_linear_checkpoint_candidate(
        module
    )


def create_fsdp_wrap_policy(min_num_params: int = 1e6):
    """
    Create an auto-wrap policy for FullyShardedDataParallel that intelligently wraps dendritic layers.

    Args:
        min_num_params: Minimum number of parameters for a layer to be wrapped

    Returns:
        Auto-wrap policy function
    """

    def dendritic_auto_wrap_policy(module, recurse, nonwrapped_numel):
        # Always wrap these large layers
        if _is_dendritic_fsdp_module(module):
            return True

        # Wrap based on parameter count
        if nonwrapped_numel >= min_num_params:
            return True

        # Don't wrap small modules
        return False

    return dendritic_auto_wrap_policy


def get_fsdp_config(
    model_size: str = "large",
    mixed_precision: bool = True,
    cpu_offload: bool = False,
    sharding_strategy: str = "FULL_SHARD",
    reduce_communication_overhead: bool = True,
    fsdp_config_dict: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """
    Get FullyShardedDataParallel configuration based on model size and requirements.

    Args:
        model_size: "small", "large", or "xlarge"
        mixed_precision: Whether to use mixed precision training
        cpu_offload: Whether to offload parameters to CPU
        sharding_strategy: FullyShardedDataParallel sharding strategy
        reduce_communication_overhead: Whether to enable communication optimizations
        fsdp_config_dict: Direct FullyShardedDataParallel configuration from config file (overrides other params)

    Returns:
        FullyShardedDataParallel configuration dictionary
    """
    # If a direct config dict is provided, use those values with fallbacks
    if fsdp_config_dict is not None:
        mixed_precision = fsdp_config_dict.get("mixed_precision", mixed_precision)
        cpu_offload = fsdp_config_dict.get("cpu_offload", cpu_offload)
        sharding_strategy = fsdp_config_dict.get("sharding_strategy", sharding_strategy)
        reduce_communication_overhead = fsdp_config_dict.get(
            "reduce_communication_overhead", reduce_communication_overhead
        )
        min_num_params = fsdp_config_dict.get("min_num_params", None)
        auto_wrap_policy_name = str(
            fsdp_config_dict.get("auto_wrap_policy", "none")
        ).lower()
        sync_module_states = fsdp_config_dict.get("sync_module_states", True)
        use_orig_params = fsdp_config_dict.get("use_orig_params", True)
        backward_prefetch = fsdp_config_dict.get("backward_prefetch", "BACKWARD_PRE")
        limit_all_gathers = fsdp_config_dict.get("limit_all_gathers", True)
        forward_prefetch = fsdp_config_dict.get("forward_prefetch", True)
        gradient_checkpointing = fsdp_config_dict.get("gradient_checkpointing", False)
        # `or` (not dict.get default) so explicit None granular fields — the
        # FSDPConfig defaults — fall back to the single mixed_precision_dtype knob.
        mp_dtype_knob = fsdp_config_dict.get("mixed_precision_dtype") or "fp16"
        mp_param_dtype = fsdp_config_dict.get("param_dtype") or mp_dtype_knob
        mp_reduce_dtype = fsdp_config_dict.get("reduce_dtype") or mp_param_dtype
        mp_buffer_dtype = fsdp_config_dict.get("buffer_dtype") or mp_param_dtype
    else:
        # Use defaults
        min_num_params = None
        auto_wrap_policy_name = "none"
        sync_module_states = True
        use_orig_params = True
        backward_prefetch = "BACKWARD_PRE"
        limit_all_gathers = True
        forward_prefetch = True
        gradient_checkpointing = False
        mp_param_dtype = "fp16"
        mp_reduce_dtype = "fp16"
        mp_buffer_dtype = "fp16"

    # Sharding strategies
    strategy_map = {
        "FULL_SHARD": ShardingStrategy.FULL_SHARD,
        "SHARD_GRAD_OP": ShardingStrategy.SHARD_GRAD_OP,
        "NO_SHARD": ShardingStrategy.NO_SHARD,
        "HYBRID_SHARD": ShardingStrategy.HYBRID_SHARD,  # Reduces communication overhead
    }

    if reduce_communication_overhead and sharding_strategy == "FULL_SHARD":
        logger.info(
            "HYBRID_SHARD auto-upgrade disabled - using FULL_SHARD with communication optimizations"
        )

    # Mixed precision configuration
    mp_config = None
    if mixed_precision:
        mp_config = MixedPrecision(
            param_dtype=_resolve_fsdp_dtype(mp_param_dtype, field_name="param_dtype"),
            reduce_dtype=_resolve_fsdp_dtype(
                mp_reduce_dtype, field_name="reduce_dtype"
            ),
            buffer_dtype=_resolve_fsdp_dtype(
                mp_buffer_dtype, field_name="buffer_dtype"
            ),
        )

    # CPU offload configuration
    cpu_offload_config = None
    if cpu_offload:
        cpu_offload_config = CPUOffload(offload_params=True)

    # Auto-wrap policy based on model size or explicit min_num_params
    if min_num_params is not None:
        # Use explicit value from config
        pass
    elif model_size == "xlarge":
        pass  # Wrap layers with >100K params
    elif model_size == "large":
        pass  # Wrap layers with >1M params
    else:
        pass  # Wrap layers with >10M params

    if auto_wrap_policy_name == "none":
        # Preserve the historical whole-model wrapping behavior by default.
        auto_wrap_policy = None
    elif auto_wrap_policy_name == "indexed_synapses":
        # This narrowly targets leaf sparse-synapse modules. It avoids the
        # parent/child wrapping ambiguity of the old parameter-count policy
        # while bounding full-parameter all-gathers for very large RNNs.
        auto_wrap_policy = ModuleWrapPolicy(
            {IndexedSparseLinear, IndexedDynamicTopKLinear}
        )
    else:
        raise ValueError(
            "FSDP auto_wrap_policy must be 'none' or 'indexed_synapses', "
            f"got {auto_wrap_policy_name!r}"
        )

    # Convert backward_prefetch string to enum
    backward_prefetch_map = {
        "BACKWARD_PRE": BackwardPrefetch.BACKWARD_PRE,
        "BACKWARD_POST": BackwardPrefetch.BACKWARD_POST,
        None: BackwardPrefetch.BACKWARD_PRE,  # Default
    }
    backward_prefetch_enum = backward_prefetch_map.get(
        backward_prefetch, BackwardPrefetch.BACKWARD_PRE
    )

    # Communication optimization settings
    config = {
        "sharding_strategy": strategy_map[sharding_strategy],
        "mixed_precision": mp_config,
        "cpu_offload": cpu_offload_config,
        "auto_wrap_policy": auto_wrap_policy,
        "backward_prefetch": backward_prefetch_enum,
        "device_id": torch.cuda.current_device(),
        "sync_module_states": sync_module_states,
        "use_orig_params": use_orig_params,  # Important for dendritic models
    }

    # Add communication optimizations if enabled
    if reduce_communication_overhead:
        config.update(
            {
                "limit_all_gathers": limit_all_gathers,  # Reduce memory pressure during communication
                "forward_prefetch": forward_prefetch,  # Prefetch parameters for forward pass
            }
        )

        # Additional optimizations for newer PyTorch versions
        try:
            # These may not be available in older versions
            config.update(
                {
                    "ignored_modules": [],  # Can specify modules to not shard
                    "param_init_fn": None,  # Custom parameter initialization
                }
            )
        except Exception:
            pass  # Ignore if not available

    # Store gradient checkpointing flag for later use (not directly an FullyShardedDataParallel param)
    config["_gradient_checkpointing"] = gradient_checkpointing

    # cpu_init: leave the model on CPU until FSDP wraps it (see _wrap_model).
    # The 1M gate died INSIDE FSDP.__init__: every rank ran `model.to(device)`
    # before wrapping, materializing the full unsharded model (37.6 GB at 1M)
    # on each GPU, and _sync_module_params_and_buffers then OOMed asking for
    # 2.98 GiB more. With cpu_init, FSDP's own device_id machinery moves states
    # per wrapped unit, so combined with auto_wrap_policy=indexed_synapses the
    # per-rank init peak is bounded by the largest synapse module plus the
    # accumulated shards -- never the whole model. Default off.
    config["_cpu_init"] = bool(
        fsdp_config_dict.get("cpu_init", False) if fsdp_config_dict else False
    )

    return config


def wrap_model_with_fsdp(
    model: torch.nn.Module,
    fsdp_config: Optional[dict[str, Any]] = None,
) -> FullyShardedDataParallel:
    """
    Wrap a dendritic model with FullyShardedDataParallel.

    Args:
        model: The model to wrap
        fsdp_config: FullyShardedDataParallel configuration (uses defaults if None)

    Returns:
        FullyShardedDataParallel-wrapped model
    """
    if fsdp_config is None:
        fsdp_config = get_fsdp_config()
    else:
        # Private strategy flags must not reach FSDP, but callers may reuse the
        # same configuration while wrapping multiple model components.
        fsdp_config = dict(fsdp_config)

    # Extract private flags (not FullyShardedDataParallel parameters)
    gradient_checkpointing = fsdp_config.pop("_gradient_checkpointing", False)
    fsdp_config.pop("_cpu_init", None)  # consumed by the strategy's _wrap_model

    # Special handling for dendritic models
    if hasattr(model, "net") and isinstance(model.net, ExcitationInhibitionNetwork):
        logger.info(
            "Detected E/I Network - applying FullyShardedDataParallel with dendritic-aware wrapping"
        )

        # Estimate model size
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Total model parameters: {total_params:,}")

        if total_params > 1e9:  # > 1B parameters
            logger.info("Large model detected - using aggressive sharding")
            fsdp_config["sharding_strategy"] = ShardingStrategy.FULL_SHARD

    # Apply gradient checkpointing if enabled
    if gradient_checkpointing:
        logger.info("Enabling gradient checkpointing for memory savings")
        # Apply gradient checkpointing to the model before FullyShardedDataParallel wrapping

        # For dendritic models, apply checkpointing to major components
        if hasattr(model, "net"):
            # Apply to the core network
            _apply_gradient_checkpointing(model.net)
        else:
            # Apply to the entire model
            _apply_gradient_checkpointing(model)

    # Wrap the model with FullyShardedDataParallel
    fsdp_model = FullyShardedDataParallel(model, **fsdp_config)

    return fsdp_model


def _checkpoint_module_forward(module: torch.nn.Module) -> None:
    """Patch a module forward method to run through activation checkpointing."""
    from torch.utils.checkpoint import checkpoint

    original_forward = module.forward

    def checkpointed_forward(self, *args, orig_forward=original_forward, **kwargs):
        return checkpoint(orig_forward, *args, use_reentrant=False, **kwargs)

    module.forward = MethodType(checkpointed_forward, module)


def _apply_gradient_checkpointing(model: torch.nn.Module):
    """
    Apply gradient checkpointing to suitable modules in the model.

    Args:
        model: Model to apply checkpointing to
    """
    # Checkpoint the outermost eligible module only. Nested checkpoint regions
    # recompute the same dendritic operations more than once and can make
    # non-reentrant checkpointing observe different saved-tensor metadata.
    checkpointed_names: list[str] = []
    for name, module in iter_named_modules_matching(
        model,
        _is_gradient_checkpoint_candidate,
    ):
        is_nested = any(
            bool(name) and (not parent_name or name.startswith(f"{parent_name}."))
            for parent_name in checkpointed_names
        )
        if is_nested:
            logger.debug(
                "Skipping nested gradient checkpoint target %s",
                name,
            )
            continue

        if _is_dendritic_fsdp_module(module):
            _checkpoint_module_forward(module)
            checkpointed_names.append(name)
            logger.debug("Applied gradient checkpointing to %s", name)
        elif _is_large_linear_checkpoint_candidate(module):
            _checkpoint_module_forward(module)
            checkpointed_names.append(name)
            logger.debug(
                "Applied gradient checkpointing to large linear layer %s",
                name,
            )


def save_fsdp_checkpoint(
    model: FullyShardedDataParallel,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    save_path: str,
    rank: int = 0,
):
    """
    Save FullyShardedDataParallel model checkpoint.

    Args:
        model: FullyShardedDataParallel-wrapped model
        optimizer: Optimizer
        epoch: Current epoch
        save_path: Path to save checkpoint
        rank: Process rank
    """
    # Configure state dict saving
    save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)

    with FullyShardedDataParallel.state_dict_type(
        model, StateDictType.FULL_STATE_DICT, save_policy
    ):
        state_dict = model.state_dict()
        try:
            optimizer_state_dict = FullyShardedDataParallel.optim_state_dict(
                model, optimizer
            )
        except Exception as exc:
            logger.warning(
                "Falling back to local optimizer.state_dict() while saving FSDP "
                "checkpoint because FSDP optimizer-state gather failed: %s",
                exc,
            )
            optimizer_state_dict = optimizer.state_dict()

        # Only save on rank 0
        if rank == 0:
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": state_dict,
                "optimizer_state_dict": optimizer_state_dict,
            }
            torch.save(checkpoint, save_path)
            logger.info(f"Saved FullyShardedDataParallel checkpoint to {save_path}")


def load_fsdp_checkpoint(
    model: FullyShardedDataParallel,
    optimizer: torch.optim.Optimizer,
    checkpoint_path: str,
):
    """
    Load FullyShardedDataParallel model checkpoint.

    Args:
        model: FullyShardedDataParallel-wrapped model
        optimizer: Optimizer
        checkpoint_path: Path to checkpoint

    Returns:
        Epoch number from checkpoint
    """
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # Configure state dict loading
    with FullyShardedDataParallel.state_dict_type(model, StateDictType.FULL_STATE_DICT):
        model.load_state_dict(checkpoint["model_state_dict"])

    try:
        optimizer_state_dict = FullyShardedDataParallel.optim_state_dict_to_load(
            model, optimizer, checkpoint["optimizer_state_dict"]
        )
    except Exception as exc:
        logger.warning(
            "Falling back to direct optimizer.load_state_dict() for FSDP "
            "checkpoint because optimizer-state remapping failed: %s",
            exc,
        )
        optimizer_state_dict = checkpoint["optimizer_state_dict"]
    optimizer.load_state_dict(optimizer_state_dict)

    return checkpoint["epoch"]


def estimate_memory_requirements(
    num_neurons: list[int],
    num_synapses_per_neuron: list[int],
    num_branches: list[int],
    dtype: torch.dtype = torch.float32,
) -> dict[str, float]:
    """
    Estimate memory requirements for a dendritic model.

    Args:
        num_neurons: Number of neurons per layer
        num_synapses_per_neuron: Synapses per neuron per layer
        num_branches: Number of branches per layer
        dtype: Data type for parameters

    Returns:
        Dictionary with memory estimates in GB
    """
    bytes_per_param = 4 if dtype == torch.float32 else 2

    total_params = 0
    layer_params = []

    for i in range(len(num_neurons) - 1):
        # TopKLinear parameters
        n_in = num_neurons[i]
        n_out = num_neurons[i + 1]
        n_branches_out = num_branches[i + 1] if i + 1 < len(num_branches) else 1

        # Each branch has its own TopKLinear layers
        topk_params = n_in * n_out * n_branches_out

        # Branch aggregation parameters
        if i > 0:
            branch_agg_params = num_branches[i] * n_branches_out
        else:
            branch_agg_params = 0

        layer_total = topk_params + branch_agg_params
        layer_params.append(layer_total)
        total_params += layer_total

    # Convert to GB
    param_memory_gb = (total_params * bytes_per_param) / 1e9

    # Estimate additional memory needs
    # Gradients: same as parameters
    gradient_memory_gb = param_memory_gb

    # Optimizer states (Adam): 2x parameters
    optimizer_memory_gb = 2 * param_memory_gb

    # Activations (rough estimate): 10% of parameters
    activation_memory_gb = 0.1 * param_memory_gb

    total_memory_gb = (
        param_memory_gb
        + gradient_memory_gb
        + optimizer_memory_gb
        + activation_memory_gb
    )

    return {
        "parameters_gb": param_memory_gb,
        "gradients_gb": gradient_memory_gb,
        "optimizer_gb": optimizer_memory_gb,
        "activations_gb": activation_memory_gb,
        "total_gb": total_memory_gb,
        "total_params": total_params,
        "layer_params": layer_params,
    }


def get_recommended_gpu_count(memory_per_gpu_gb: float, total_memory_gb: float) -> int:
    """
    Get recommended number of GPUs based on memory requirements.

    Args:
        memory_per_gpu_gb: Available memory per GPU (e.g., 80 for H100)
        total_memory_gb: Total memory required

    Returns:
        Recommended number of GPUs
    """
    # Leave 10% headroom
    usable_memory_per_gpu = memory_per_gpu_gb * 0.9

    # Calculate minimum GPUs needed
    min_gpus = int(total_memory_gb / usable_memory_per_gpu) + 1

    # Round up to power of 2 for better communication efficiency
    if min_gpus <= 2:
        return 2
    elif min_gpus <= 4:
        return 4
    elif min_gpus <= 8:
        return 8
    elif min_gpus <= 16:
        return 16
    else:
        return ((min_gpus + 15) // 16) * 16  # Round up to nearest 16
